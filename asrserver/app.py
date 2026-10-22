"""Server chép lời video HLS theo job bất đồng bộ, kết quả khớp ml.VideoTranscript v2.

Chạy: python -m asrserver.app  (cwd = Projects/qwen3-asr). Biến môi trường: xem phần CẤU HÌNH.
"""
import asyncio
import concurrent.futures
import gc
import http.client
import json
import logging
import os
import secrets
import socket
import threading
import time
import urllib.error
import urllib.request
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Literal

import av
from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, PlainTextResponse
from pydantic import BaseModel, ConfigDict, Field

from . import cues, edge_dns, hls, pipeline

# ---------- CẤU HÌNH ----------
ROOT = Path(__file__).resolve().parent.parent
MODEL = os.environ.get("ASR_MODEL", "Qwen/Qwen3-ASR-1.7B")
HOST = os.environ.get("ASR_HOST", "127.0.0.1")
PORT = int(os.environ.get("ASR_PORT", "8000"))
GPU_UTIL = float(os.environ.get("ASR_GPU_UTIL", "0.45"))
MAX_ACTIVE = int(os.environ.get("ASR_MAX_ACTIVE", "6"))     # job tải/chép cùng lúc
HIGH_EXTRA = int(os.environ.get("ASR_HIGH_EXTRA", "4"))     # slot thêm chỉ cho job high
MAX_QUEUE = int(os.environ.get("ASR_MAX_QUEUE", "2000"))    # quá thì 429
LANG_LOCK = os.environ.get("ASR_LANG_LOCK", "1") == "1"
JOBS_DIR = ROOT / "jobs"
KEEP = timedelta(hours=24)
API_KEY = (ROOT / ".api_key").read_text().strip()
VERSION = pipeline.VERSION if LANG_LOCK else f"{pipeline.VERSION}-nolock"
PRIORITY = {"high": 0, "normal": 1, "low": 2}
FINISHED = ("succeeded", "failed", "cancelled")

log = logging.getLogger("asrserver")
POOL = concurrent.futures.ThreadPoolExecutor((MAX_ACTIVE + HIGH_EXTRA) * 2 + 4, thread_name_prefix="job")


def now():
    return datetime.now(timezone.utc)


def iso(dt):
    return dt.isoformat(timespec="seconds").replace("+00:00", "Z") if dt else None


def parse_iso(s):
    return datetime.fromisoformat(s.replace("Z", "+00:00")) if s else None


# ---------- BODY ----------
class Options(BaseModel):
    model_config = ConfigDict(extra="forbid")
    languages: list[str] = ["Vietnamese", "English"]
    window_s: float = Field(20.0, gt=0)
    window_slack_s: float = Field(5.0, ge=0)
    chunk_s: float = Field(6.0, gt=0)
    chunk_slack_s: float = Field(3.0, ge=0)
    min_voiced_s: float = Field(0.3, ge=0)
    max_chars: int = Field(84, gt=0)
    line_chars: int = Field(42, gt=0)
    tokens_per_second: float = Field(15.0, gt=0)


class JobIn(BaseModel):
    model_config = ConfigDict(extra="forbid")
    audio_url: str
    ref: str | int | None = None
    max_seconds: float = Field(0, ge=0)
    priority: Literal["high", "normal", "low"] = "normal"
    options: Options = Options()
    callback_url: str | None = None


# ---------- LỖI ----------
class Cancelled(Exception):
    pass


def classify(e):
    """→ (code, message, retryable). retryable = lỗi server/mạng, client gửi lại sau."""
    from vllm.v1.engine.exceptions import EngineDeadError

    msg = f"{type(e).__name__}: {e}"[:500]
    if isinstance(e, urllib.error.HTTPError):
        return "download_error", f"HTTP {e.code} {e.url}", e.code >= 500 or e.code in (408, 429)
    if isinstance(e, (urllib.error.URLError, TimeoutError, socket.timeout, ConnectionError, http.client.HTTPException)):
        return "download_error", msg, True
    if isinstance(e, ValueError) and "playlist" in str(e):
        return "download_error", msg, False
    if isinstance(e, (av.error.FFmpegError, UnicodeDecodeError)):
        return "decode_error", msg, False
    if "out of memory" in str(e).lower():
        return "gpu_oom", msg, True
    if isinstance(e, EngineDeadError):
        return "model_load", msg, True
    return "internal", msg, True


# ---------- JOB ----------
class Job:
    FIELDS = ("job_id", "ref", "status", "audio_url", "max_seconds", "priority", "options", "callback_url",
              "idempotency_key", "seq", "created_at", "started_at", "finished_at", "expires_at",
              "result", "timing", "error")

    def __init__(self, **kw):
        for f in self.FIELDS:
            setattr(self, f, kw.get(f))
        self.progress = None
        self.cancel = False
        self.tasks = []

    def save(self):
        p = JOBS_DIR / f"{self.job_id}.json"
        tmp = p.with_suffix(".tmp")
        tmp.write_text(json.dumps({f: getattr(self, f) for f in self.FIELDS}, ensure_ascii=False))
        tmp.replace(p)

    def finish(self, status):
        self.status = status
        self.finished_at = iso(now())
        self.expires_at = iso(now() + KEEP)
        self.progress = None
        self.save()


class Server:
    def __init__(self):
        self.engine = None
        self.jobs: dict[str, Job] = {}
        self.idem: dict[str, str] = {}
        self.queued: set[str] = set()
        self.running: set[str] = set()
        self.seq = 0
        self.wake = asyncio.Event()
        self.loop = None

    # --- trạng thái ---
    def order(self):
        return sorted(self.queued, key=lambda j: (PRIORITY[self.jobs[j].priority], self.jobs[j].seq))

    def view(self, job):
        out = {"job_id": job.job_id, "ref": job.ref, "status": job.status}
        if job.status == "queued":
            out["queue_position"] = self.order().index(job.job_id) + 1
        if job.status == "running":
            out["progress"] = job.progress
        if job.status == "succeeded":
            out["result"] = job.result
            out["timing"] = job.timing
        if job.status == "failed":
            out["error"] = job.error
        out["created_at"] = job.created_at
        if job.started_at:
            out["started_at"] = job.started_at
        if job.status in FINISHED:
            out["finished_at"] = job.finished_at
            out["expires_at"] = job.expires_at
        out["poll_url"] = f"/v1/asr/jobs/{job.job_id}"
        return out

    def load(self):
        JOBS_DIR.mkdir(exist_ok=True)
        for p in JOBS_DIR.glob("*.json"):
            try:
                job = Job(**json.loads(p.read_text()))
            except Exception:
                log.exception("bỏ file job hỏng %s", p)
                continue
            if job.status in FINISHED and parse_iso(job.expires_at) < now():
                p.unlink(missing_ok=True)
                continue
            if job.status in ("queued", "running"):   # server tắt giữa chừng → chép lại
                job.status, job.started_at = "queued", None
                self.queued.add(job.job_id)
            self.jobs[job.job_id] = job
            self.seq = max(self.seq, job.seq or 0)
            if job.idempotency_key:
                self.idem[job.idempotency_key] = job.job_id
        log.info("nạp %d job (%d chờ)", len(self.jobs), len(self.queued))

    def submit(self, body: JobIn, key: str | None):
        if key and (old := self.jobs.get(self.idem.get(key, ""))) and old.status not in ("failed", "cancelled") \
                and (old.result or {}).get("status") != "no_video":
            return old   # no_video có thể là video đang chuyển mã, nên gửi lại thì tải lại
        if len(self.queued) >= MAX_QUEUE:
            raise HTTPException(429, "hàng đợi đầy", headers={"Retry-After": "60"})
        self.seq += 1
        job = Job(job_id=f"asr_{secrets.token_hex(8)}", ref=body.ref, status="queued", audio_url=body.audio_url,
                  max_seconds=body.max_seconds, priority=body.priority, options=body.options.model_dump(),
                  callback_url=body.callback_url, idempotency_key=key, seq=self.seq, created_at=iso(now()))
        job.save()
        self.jobs[job.job_id] = job
        self.queued.add(job.job_id)
        if key:
            self.idem[key] = job.job_id
        self.wake.set()
        return job

    def cancel(self, job):
        if job.status == "queued":
            self.queued.discard(job.job_id)
            job.finish("cancelled")
        elif job.status == "running":
            job.cancel = True
            for t in job.tasks:
                t.cancel()

    # --- chạy ---
    async def dispatch(self):
        while True:
            await self.wake.wait()
            self.wake.clear()
            while self.queued:
                jid = self.order()[0]
                limit = MAX_ACTIVE + (HIGH_EXTRA if self.jobs[jid].priority == "high" else 0)
                if len(self.running) >= limit:
                    break
                self.queued.discard(jid)
                self.running.add(jid)
                asyncio.create_task(self.run(self.jobs[jid]))

    async def cleanup(self):
        while True:
            await asyncio.sleep(600)
            for jid, job in list(self.jobs.items()):
                if job.status in FINISHED and parse_iso(job.expires_at) < now():
                    del self.jobs[jid]
                    if self.idem.get(job.idempotency_key) == jid:
                        del self.idem[job.idempotency_key]
                    (JOBS_DIR / f"{jid}.json").unlink(missing_ok=True)

    async def chunks(self, job, clips, lang_ids, tps, prio):
        """Gửi mọi khúc một lượt cùng lúc vào vLLM; huỷ job → huỷ các request còn lại."""
        async def one(c):
            r = await self.engine.one(c, lang_ids, tps, prio)
            job.progress["done"] += 1
            job.progress["percent"] = int(100 * job.progress["done"] / job.progress["total"])
            return r

        job.tasks = [asyncio.create_task(one(c)) for c in clips]
        try:
            return await asyncio.gather(*job.tasks)
        except BaseException:
            for t in job.tasks:
                t.cancel()
            raise
        finally:
            job.tasks = []

    def transcriber(self, job, cfg, prio):
        """Hàm đồng bộ cho pipeline.transcribe (chạy trong luồng job), gọi sang event loop."""
        lang_ids = self.engine.lang_ids(cfg["languages"])
        passes = iter(["pass1", "pass2"])

        def run(clips):
            if job.cancel:
                raise Cancelled
            job.progress = {"stage": next(passes, "pass2"), "done": 0, "total": len(clips), "percent": 0}
            fut = asyncio.run_coroutine_threadsafe(
                self.chunks(job, clips, lang_ids, cfg["tokens_per_second"], prio), self.loop)
            try:
                return fut.result()
            except (concurrent.futures.CancelledError, asyncio.CancelledError):
                raise Cancelled

        return run

    def work(self, job):
        """Toàn bộ job (luồng riêng) → (result, timing)."""
        cfg = job.options

        def dl_progress(done, total):
            if job.cancel:
                raise Cancelled
            job.progress = {"stage": "download", "done": done, "total": total, "percent": int(100 * done / total)}

        t0 = time.monotonic()
        job.progress = {"stage": "download", "done": 0, "total": 0, "percent": 0}
        try:
            audio = hls.load_audio(job.audio_url, job.max_seconds, progress=dl_progress)
        except hls.NoVideo as e:
            return {"status": "no_video", "reason": str(e), "duration_sec": None, "transcribed_sec": None,
                    "playlist": None, "language": ",,", "cues": [], "vtt": None,
                    "model": MODEL, "version": VERSION}, {"download_sec": round(time.monotonic() - t0, 2)}
        t1 = time.monotonic()
        status, out = pipeline.transcribe(audio, self.transcriber(job, cfg, PRIORITY[job.priority]), cfg)
        job.progress = {"stage": "cues", "done": 0, "total": 1, "percent": 99}
        result = {"status": status, "reason": None, "duration_sec": audio.duration,
                  "transcribed_sec": audio.seconds, "playlist": audio.playlist,
                  "language": cues.languages(out), "cues": out,
                  "vtt": cues.to_vtt(out, cfg["line_chars"]) if out else None,
                  "model": MODEL, "version": VERSION}
        t2 = time.monotonic()
        return result, {"download_sec": round(t1 - t0, 2), "asr_sec": round(t2 - t1, 2),
                        "processing_sec": round(t2 - t0, 2),
                        "rtf": round((t2 - t0) / audio.seconds, 4) if audio.seconds else None}

    async def run(self, job):
        job.status, job.started_at = "running", iso(now())
        job.save()
        queued_sec = round((parse_iso(job.started_at) - parse_iso(job.created_at)).total_seconds(), 1)
        fatal = False
        try:
            job.result, timing = await self.loop.run_in_executor(POOL, self.work, job)
            job.timing = {"queued_sec": queued_sec, **timing}
            job.finish("succeeded")
            log.info("xong %s ref=%s %s %.0fs rtf=%s", job.job_id, job.ref, job.result["status"],
                     job.result["transcribed_sec"] or 0, timing.get("rtf"))
        except Cancelled:
            job.finish("cancelled")
        except Exception as e:
            from vllm.v1.engine.exceptions import EngineDeadError

            code, msg, retry = classify(e)
            job.error = {"code": code, "message": msg, "retryable": retry}
            job.finish("failed")
            log.warning("lỗi %s ref=%s %s %s", job.job_id, job.ref, code, msg)
            gc.collect()   # lỗi giữa lúc tải để lại vòng tham chiếu future → exception → socket
            fatal = isinstance(e, EngineDeadError)
        finally:
            self.running.discard(job.job_id)
            self.wake.set()
        if job.callback_url:
            POOL.submit(callback, job.callback_url, self.view(job))
        if fatal:   # engine chết → thoát để systemd khởi động lại
            log.error("EngineCore chết, thoát")
            os._exit(1)


def callback(url, body):
    data = json.dumps(body, ensure_ascii=False).encode()
    for wait in (0, 5, 30, 120):
        time.sleep(wait)
        try:
            req = urllib.request.Request(url, data=data, method="POST",
                                         headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=20) as r:
                if r.status < 300:
                    return
        except Exception as e:
            log.warning("callback %s lỗi: %s", url, e)


# ---------- HTTP ----------
S = Server()


@asynccontextmanager
async def lifespan(app):
    from .engine import Engine

    S.loop = asyncio.get_running_loop()
    if os.environ.get("ASR_CDN_DNS"):   # tắt mặc định: edge VN đo không nhanh hơn (xem edge_dns.py)
        log.info("edge CDN lessonvid: %s", edge_dns.install())
    S.load()
    S.engine = Engine(MODEL, gpu_memory_utilization=GPU_UTIL, lang_lock=LANG_LOCK)
    await S.engine.warmup()
    tasks = [asyncio.create_task(S.dispatch()), asyncio.create_task(S.cleanup())]
    S.wake.set()
    log.info("sẵn sàng: %s %s, %d job song song", MODEL, VERSION, MAX_ACTIVE)
    yield
    for t in tasks:
        t.cancel()


app = FastAPI(title="Qwen3-ASR job server", lifespan=lifespan, docs_url=None, redoc_url=None, openapi_url=None)


@app.exception_handler(RequestValidationError)
async def bad_body(request, exc):
    return JSONResponse({"error": "invalid_body", "detail": json.loads(json.dumps(exc.errors(), default=str))},
                        status_code=400)


def auth(authorization: str | None = Header(None), x_api_key: str | None = Header(None)):
    key = x_api_key or (authorization or "").removeprefix("Bearer ").strip()
    if not key or not secrets.compare_digest(key, API_KEY):
        raise HTTPException(401, "thiếu hoặc sai API key", headers={"WWW-Authenticate": "Bearer"})


def get_job(job_id):
    job = S.jobs.get(job_id)
    if not job:
        raise HTTPException(404, "không có job")
    return job


@app.get("/health")
async def health():
    return {"status": "ok", "queue": len(S.queued), "running": len(S.running), "model": MODEL, "version": VERSION}


@app.post("/v1/asr/jobs", status_code=202, dependencies=[Depends(auth)])
async def create(body: JobIn, idempotency_key: str | None = Header(None)):
    try:
        S.engine.lang_ids(body.options.languages)
    except ValueError as e:
        raise HTTPException(400, str(e))
    if body.callback_url and not body.callback_url.startswith(("http://", "https://")):
        raise HTTPException(400, "callback_url phải là http(s)")
    job = S.submit(body, idempotency_key)
    return {k: v for k, v in S.view(job).items() if k in
            ("job_id", "ref", "status", "queue_position", "created_at", "poll_url")}


@app.get("/v1/asr/jobs/{job_id}", dependencies=[Depends(auth)])
async def status(job_id: str):
    return S.view(get_job(job_id))


@app.delete("/v1/asr/jobs/{job_id}", dependencies=[Depends(auth)])
async def cancel(job_id: str):
    job = get_job(job_id)
    S.cancel(job)
    for _ in range(50):   # job đang chạy dừng ở mốc kiểm tra kế tiếp (nhóm .ts hoặc lượt chép)
        if job.status in FINISHED:
            break
        await asyncio.sleep(0.2)
    return S.view(job)


@app.get("/v1/asr/jobs/{job_id}/result.vtt", dependencies=[Depends(auth)])
async def vtt(job_id: str):
    job = get_job(job_id)
    if job.status != "succeeded":
        raise HTTPException(409, f"job đang {job.status}")
    return PlainTextResponse(job.result["vtt"] or "WEBVTT\n\n", media_type="text/vtt; charset=utf-8")


if __name__ == "__main__":
    import uvicorn

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    uvicorn.run(app, host=HOST, port=PORT, log_level="warning")

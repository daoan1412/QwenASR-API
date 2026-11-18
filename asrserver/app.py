"""Server ASR dùng chung theo job bất đồng bộ: chép lời video HLS (kết quả khớp ml.VideoTranscript v2), tệp upload hoặc
link tệp tiếng/video; chế độ cues (câu phụ đề, mặc định) | text (chỉ chữ) | align (mốc từng từ, Qwen3-ForcedAligner).

Chạy: python -m asrserver.app  (cwd = thư mục repo). Biến môi trường: xem phần CẤU HÌNH.
"""
import asyncio
import concurrent.futures
import gc
import hashlib
import http.client
import json
import logging
import os
import re
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
from fastapi import Depends, FastAPI, Header, HTTPException, Query, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, PlainTextResponse
from pydantic import BaseModel, ConfigDict, Field, model_validator

from . import aligner, asr, cues, edge_dns, hls, media, pipeline
from .asr import SR

# ---------- CẤU HÌNH ----------
ROOT = Path(__file__).resolve().parent.parent
MODEL = os.environ.get("ASR_MODEL", "Qwen/Qwen3-ASR-1.7B")
HOSTS = [h.strip() for h in os.environ.get("ASR_HOST", "127.0.0.1").split(",") if h.strip()]  # vd. "127.0.0.1,172.17.0.1"
PORT = int(os.environ.get("ASR_PORT", "8000"))
GPU_UTIL = float(os.environ.get("ASR_GPU_UTIL", "0.45"))
MAX_ACTIVE = int(os.environ.get("ASR_MAX_ACTIVE", "12"))     # job tải/chép cùng lúc
HIGH_EXTRA = int(os.environ.get("ASR_HIGH_EXTRA", "4"))     # slot thêm chỉ cho job high
MAX_QUEUE = int(os.environ.get("ASR_MAX_QUEUE", "2000"))    # quá thì 429
LANG_LOCK = os.environ.get("ASR_LANG_LOCK", "1") == "1"
KEY_MAX_QUEUE = int(os.environ.get("ASR_KEY_MAX_QUEUE", "200"))  # job chờ tối đa của mỗi khoá phụ (.api_keys)
ALIGN_PRELOAD = os.environ.get("ASR_ALIGN_PRELOAD", "1") == "1"
KEYS_DIR = Path(os.environ.get("ASR_KEYS_DIR", ROOT))   # chứa .api_key (+ .api_keys); test đặt thư mục tạm
JOBS_DIR = Path(os.environ.get("ASR_JOBS_DIR", ROOT / "jobs"))
UPLOADS = JOBS_DIR / "uploads"
KEEP = timedelta(hours=24)
UPLOAD_ID = re.compile(r"[0-9a-f]{32}")


def load_keys():
    """{khoá: tên}: .api_key là "main" (không giới hạn); .api_keys thêm khoá phụ, mỗi dòng "tên khoá" (vd. sandbox Xưởng
    video) — job chờ của mỗi khoá phụ tối đa ASR_KEY_MAX_QUEUE."""
    keys = {(KEYS_DIR / ".api_key").read_text().strip(): "main"}
    extra = KEYS_DIR / ".api_keys"
    if extra.exists():
        for ln in extra.read_text().splitlines():
            if ln.strip() and not ln.lstrip().startswith("#"):
                name, key = ln.split()
                keys[key] = name
    return keys


KEYS = load_keys()
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
    words: bool = False              # cues: thêm `words` [[từ, s, e], …] — mốc từng từ bằng ForcedAligner (aligner.align_cues)
    align_chunk_s: float = Field(0.0, ge=0, le=170)   # words / align+cues: gom câu thành khúc ≤ bấy nhiêu giây; 0 = mỗi câu một khúc


class JobIn(BaseModel):
    model_config = ConfigDict(extra="forbid")
    audio_url: str | None = None     # link HLS (như cũ) hoặc link tệp tiếng/video
    upload_id: str | None = None     # tệp đã gửi qua POST /v1/uploads
    mode: Literal["cues", "text", "align"] = "cues"
    text: str | None = None          # align: lời đã biết (vd. chữ TTS đọc)
    cues: list[dict] | None = None   # align: câu phụ đề đã chép [{s, e, t, lang}] (vd. ml.VideoTranscript) — gán mốc từng từ
                                     #   cho lời có sẵn mà không chép lại; tiếng dài bao nhiêu cũng được (chia khúc ≤ 150 s)
    language: str | None = None      # text: ép ngôn ngữ (như qwen_asr language=…); align: mặc định Vietnamese
    ref: str | int | None = None
    max_seconds: float = Field(0, ge=0)
    priority: Literal["high", "normal", "low"] = "normal"
    options: Options = Options()
    callback_url: str | None = None

    @model_validator(mode="after")
    def check(self):
        if (self.audio_url is None) == (self.upload_id is None):
            raise ValueError("cần đúng một trong audio_url / upload_id")
        if self.upload_id is not None and not UPLOAD_ID.fullmatch(self.upload_id):
            raise ValueError("upload_id sai dạng")
        if self.mode == "align" and not (self.text or "").strip() and not self.cues:
            raise ValueError("mode align cần text hoặc cues")
        if self.cues is not None and (self.mode != "align" or (self.text or "").strip()):
            raise ValueError("cues chỉ dùng với mode align, không kèm text")
        return self


# ---------- LỖI ----------
class Cancelled(Exception):
    pass


def classify(e):
    """→ (code, message, retryable). retryable = lỗi server/mạng, client gửi lại sau."""
    from vllm.v1.engine.exceptions import EngineDeadError

    msg = f"{type(e).__name__}: {e}"[:500]
    if isinstance(e, media.NotMedia):
        return "not_media", msg, False
    if isinstance(e, ValueError) and str(e).startswith(("align:", "upload")):
        return "bad_input", msg, False
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
    FIELDS = ("job_id", "ref", "status", "audio_url", "upload_id", "mode", "text", "cues", "language", "key_name",
              "max_seconds", "priority", "options", "callback_url",
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

    def submit(self, body: JobIn, key: str | None, key_name: str = "main"):
        if key and (old := self.jobs.get(self.idem.get(key, ""))) and old.status not in ("failed", "cancelled") \
                and (old.result or {}).get("status") != "no_video":
            return old   # no_video có thể là video đang chuyển mã, nên gửi lại thì tải lại
        if len(self.queued) >= MAX_QUEUE:
            raise HTTPException(429, "hàng đợi đầy", headers={"Retry-After": "60"})
        if key_name != "main" and sum(self.jobs[j].key_name == key_name for j in self.queued) >= KEY_MAX_QUEUE:
            raise HTTPException(429, f"khoá {key_name}: quá {KEY_MAX_QUEUE} job chờ", headers={"Retry-After": "30"})
        if body.upload_id:
            p = UPLOADS / body.upload_id
            if not p.exists():
                raise HTTPException(404, "không có upload_id (hết hạn 24 h hoặc chưa gửi)")
            os.utime(p)
        self.seq += 1
        job = Job(job_id=f"asr_{secrets.token_hex(8)}", ref=body.ref, status="queued", audio_url=body.audio_url,
                  upload_id=body.upload_id, mode=body.mode, text=body.text, cues=body.cues, language=body.language, key_name=key_name,
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
            # tệp upload: giữ KEEP kể từ lần dùng cuối (tạo job / job xong), job còn chờ/chạy thì giữ
            busy = {self.jobs[j].upload_id for j in self.queued | self.running}
            for p in UPLOADS.glob("*"):
                if p.name not in busy and now() - datetime.fromtimestamp(p.stat().st_mtime, timezone.utc) > KEEP:
                    p.unlink(missing_ok=True)

    async def chunks(self, job, clips, lang_ids, tps, prio, force_language=None):
        """Gửi mọi khúc một lượt cùng lúc vào vLLM; huỷ job → huỷ các request còn lại."""
        async def one(c):
            r = await self.engine.one(c, lang_ids, tps, prio, force_language)
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

    def transcriber(self, job, cfg, prio, force_language=None):
        """Hàm đồng bộ cho pipeline.transcribe (chạy trong luồng job), gọi sang event loop."""
        lang_ids = self.engine.lang_ids(cfg["languages"])
        passes = iter(["pass1", "pass2"])

        def run(clips):
            if job.cancel:
                raise Cancelled
            job.progress = {"stage": next(passes, "pass2"), "done": 0, "total": len(clips), "percent": 0}
            fut = asyncio.run_coroutine_threadsafe(
                self.chunks(job, clips, lang_ids, cfg["tokens_per_second"], prio, force_language), self.loop)
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
        mode = job.mode or "cues"
        try:
            if job.upload_id:
                p = UPLOADS / job.upload_id
                if not p.exists():
                    raise ValueError("upload đã hết hạn / bị xoá")
                audio = media.decode(str(p), job.max_seconds)
            else:
                audio = media.load_url(job.audio_url, job.max_seconds, progress=dl_progress)
        except hls.NoVideo as e:
            return {"status": "no_video", "reason": str(e), "duration_sec": None, "transcribed_sec": None,
                    "playlist": None, "language": ",,", "cues": [], "vtt": None,
                    "model": MODEL, "version": VERSION}, {"download_sec": round(time.monotonic() - t0, 2)}
        t1 = time.monotonic()
        if mode != "cues":
            result = self.work_text(job, audio, cfg) if mode == "text" else self.work_align(job, audio)
            t2 = time.monotonic()
            return result, {"download_sec": round(t1 - t0, 2), "asr_sec": round(t2 - t1, 2),
                            "processing_sec": round(t2 - t0, 2)}
        status, out = pipeline.transcribe(audio, self.transcriber(job, cfg, PRIORITY[job.priority]), cfg)
        job.progress = {"stage": "cues", "done": 0, "total": 1, "percent": 99}
        result = {"status": status, "reason": None, "duration_sec": audio.duration,
                  "transcribed_sec": audio.seconds, "playlist": audio.playlist,
                  "language": cues.languages(out), "cues": out,
                  "vtt": cues.to_vtt(out, cfg["line_chars"]) if out else None,
                  "model": MODEL, "version": VERSION}
        t2 = time.monotonic()
        timing = {"download_sec": round(t1 - t0, 2), "asr_sec": round(t2 - t1, 2)}
        if cfg.get("words") and out:
            job.progress = {"stage": "align", "done": 0, "total": 1, "percent": 99}
            try:
                result["words"] = aligner.align_cues(ALIGNER, audio.wav, out, cfg.get("align_chunk_s", 0.0))
            except Exception as e:  # mốc từng từ hỏng không làm hỏng câu phụ đề
                log.warning("align từ hỏng %s: %r", job.job_id, e)
                result["words"], result["words_error"] = None, f"{type(e).__name__}: {e}"[:300]
            timing["align_sec"] = round(time.monotonic() - t2, 2)
        t3 = time.monotonic()
        return result, {**timing, "processing_sec": round(t3 - t0, 2),
                        "rtf": round((t3 - t0) / audio.seconds, 4) if audio.seconds else None}

    def work_text(self, job, audio, cfg):
        """Chỉ chữ (vd. soát giọng TTS): một lượt khúc ~window_s giây cắt ở chỗ nhỏ tiếng, bỏ khúc lặng, nối chữ."""
        base = {"reason": None, "duration_sec": audio.duration, "transcribed_sec": audio.seconds,
                "playlist": audio.playlist, "model": MODEL, "version": VERSION}
        if audio.wav is None:
            return {"status": "no_audio", "text": "", "language": ",,", **base}
        wav = audio.wav
        rms = asr.frame_rms(wav)
        thr = asr.threshold(rms)
        wins = [(s, e) for s, e in asr.cut(rms, len(wav) / SR, cfg["window_s"], cfg["window_slack_s"])
                if asr.voiced_span(rms, s, e, thr, cfg["min_voiced_s"])]
        if not wins:
            return {"status": "silent", "text": "", "language": ",,", **base}
        out = self.transcriber(job, cfg, PRIORITY[job.priority], job.language)(
            [wav[int(s * SR):int(e * SR)] for s, e in wins])
        text = " ".join(t for _, t in out if t)
        lang = cues.languages([{"lang": cues.lang_code(l)} for l, t in out if t])
        return {"status": "ok" if text else "silent", "text": text, "language": lang, **base}

    def work_align(self, job, audio):
        base = {"reason": None, "duration_sec": audio.duration, "transcribed_sec": audio.seconds,
                "playlist": audio.playlist, "model": aligner.MODEL, "version": VERSION}
        if audio.wav is None:
            return {"status": "no_audio", "words": [], **base}
        job.progress = {"stage": "align", "done": 0, "total": 1, "percent": 50}
        if job.cues:
            return {"status": "ok", "words": aligner.align_cues(ALIGNER, audio.wav, job.cues, job.options.get("align_chunk_s", 0.0)), **base}
        path = str(UPLOADS / job.upload_id) if job.upload_id else None
        return {"status": "ok", "words": ALIGNER.align(audio.wav, job.text, job.language or "Vietnamese", path), **base}

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
            if job.upload_id and (UPLOADS / job.upload_id).exists():
                os.utime(UPLOADS / job.upload_id)
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
ALIGNER = aligner.Aligner()


@asynccontextmanager
async def lifespan(app):
    from .engine import Engine

    S.loop = asyncio.get_running_loop()
    if os.environ.get("ASR_CDN_DNS"):   # tắt mặc định: edge VN đo không nhanh hơn (xem edge_dns.py)
        log.info("edge CDN lessonvid: %s", edge_dns.install())
    S.load()
    UPLOADS.mkdir(parents=True, exist_ok=True)
    S.engine = Engine(MODEL, gpu_memory_utilization=GPU_UTIL, lang_lock=LANG_LOCK)
    await S.engine.warmup()
    if ALIGN_PRELOAD:   # nạp aligner nền: job align đầu tiên khỏi chờ ~10 s, lỗi GPU thấy ngay trong log
        POOL.submit(lambda: (ALIGNER.load(), log.info("aligner sẵn sàng: %s %s", aligner.MODEL, aligner.DEVICE)))
    tasks = [asyncio.create_task(S.dispatch()), asyncio.create_task(S.cleanup())]
    S.wake.set()
    log.info("sẵn sàng: %s %s, %d job song song, %d khoá, nghe %s:%d", MODEL, VERSION, MAX_ACTIVE, len(KEYS),
             ",".join(HOSTS), PORT)
    yield
    for t in tasks:
        t.cancel()


app = FastAPI(title="Qwen3-ASR job server", lifespan=lifespan, docs_url=None, redoc_url=None, openapi_url=None)


@app.exception_handler(RequestValidationError)
async def bad_body(request, exc):
    return JSONResponse({"error": "invalid_body", "detail": json.loads(json.dumps(exc.errors(), default=str))},
                        status_code=400)


def auth(authorization: str | None = Header(None), x_api_key: str | None = Header(None)) -> str:
    """→ tên khoá ("main" hoặc tên trong .api_keys)."""
    key = x_api_key or (authorization or "").removeprefix("Bearer ").strip()
    for k, name in KEYS.items():
        if key and secrets.compare_digest(key, k):
            return name
    raise HTTPException(401, "thiếu hoặc sai API key", headers={"WWW-Authenticate": "Bearer"})


def get_job(job_id):
    job = S.jobs.get(job_id)
    if not job:
        raise HTTPException(404, "không có job")
    return job


@app.get("/health")
async def health():
    return {"status": "ok", "queue": len(S.queued), "running": len(S.running), "model": MODEL, "version": VERSION}


@app.post("/v1/uploads", status_code=201)
async def upload(request: Request, key_name: str = Depends(auth)):
    """Thân request là nguyên tệp tiếng/video (wav, mp3, mp4…) → {upload_id, bytes}; dùng trong job ≤ 24 h kể từ lần dùng
    cuối. Cùng nội dung → cùng upload_id (sha256)."""
    UPLOADS.mkdir(parents=True, exist_ok=True)
    h, n = hashlib.sha256(), 0
    tmp = UPLOADS / f".{secrets.token_hex(8)}.part"
    try:
        with open(tmp, "wb") as f:
            async for chunk in request.stream():
                n += len(chunk)
                if n > media.MAX_BYTES:
                    raise HTTPException(413, f"tệp lớn hơn {media.MAX_BYTES >> 20} MB")
                h.update(chunk)
                f.write(chunk)
        if n == 0:
            raise HTTPException(400, "thân request rỗng")
        uid = h.hexdigest()[:32]
        dst = UPLOADS / uid
        if dst.exists():
            os.utime(dst)
        else:
            tmp.replace(dst)
    finally:
        tmp.unlink(missing_ok=True)
    log.info("upload %s %d byte (khoá %s)", uid, n, key_name)
    return {"upload_id": uid, "bytes": n}


@app.post("/v1/asr/jobs", status_code=202)
async def create(body: JobIn, idempotency_key: str | None = Header(None), key_name: str = Depends(auth)):
    try:
        S.engine.lang_ids(body.options.languages)
        if body.language:
            S.engine.lang_ids([body.language])
    except ValueError as e:
        raise HTTPException(400, str(e))
    if body.callback_url and not body.callback_url.startswith(("http://", "https://")):
        raise HTTPException(400, "callback_url phải là http(s)")
    job = S.submit(body, idempotency_key, key_name)
    return {k: v for k, v in S.view(job).items() if k in
            ("job_id", "ref", "status", "queue_position", "created_at", "poll_url")}


@app.get("/v1/asr/jobs/{job_id}", dependencies=[Depends(auth)])
async def status(job_id: str, wait: float = Query(0, ge=0, le=60)):
    """wait > 0: giữ request tới khi job xong hoặc hết bấy nhiêu giây (client khúc ngắn khỏi hỏi đi hỏi lại)."""
    job = get_job(job_id)
    end = time.monotonic() + wait
    while job.status not in FINISHED and time.monotonic() < end:
        await asyncio.sleep(0.2)
    return S.view(job)


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
    socks = []
    for h in HOSTS:   # nhiều địa chỉ: 127.0.0.1 cho máy này + 172.17.0.1 (docker0) cho container/sandbox
        sk = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sk.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sk.bind((h, PORT))
        socks.append(sk)
    uvicorn.Server(uvicorn.Config(app, log_level="warning")).run(sockets=socks)

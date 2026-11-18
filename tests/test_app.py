"""Test API không cần GPU: engine vLLM và aligner giả, thư mục job/khoá tạm.

    .venv/bin/python -m pytest -q tests
"""
import asyncio
import http.server
import io
import os
import shutil
import subprocess
import sys
import threading
import types
import wave
from functools import partial

import numpy as np
import pytest

SR = 16_000


def tone_wav(seconds=2.0, silence=1.0) -> bytes:
    t = np.arange(int(seconds * SR)) / SR
    x = np.concatenate([0.3 * np.sin(2 * np.pi * 220 * t), np.zeros(int(silence * SR))])
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes((x * 32767).astype(np.int16).tobytes())
    return buf.getvalue()


class FakeEngine:
    calls: list = []

    def __init__(self, *a, **kw):
        pass

    def lang_ids(self, names):
        if "Klingon" in names:
            raise ValueError("ngôn ngữ không hợp lệ: 'Klingon'")
        return list(range(len(names) + 1))

    async def warmup(self):
        pass

    async def one(self, wav, lang_ids, tps=15.0, priority=1, force_language=None):
        FakeEngine.calls.append(force_language)
        await asyncio.sleep(0)
        return (force_language or "Vietnamese"), f"chữ {len(wav) / SR:.1f} giây"


class FakeAligner:
    def align(self, wav, text, language, path=None):
        assert path is None or os.path.exists(path)   # upload: đường dẫn tệp; align_cues: khúc wav
        return [[w, i * 0.5, i * 0.5 + 0.4] for i, w in enumerate(text.split())]


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    d = tmp_path_factory.mktemp("asr")
    (d / ".api_key").write_text("k-main\n")
    (d / ".api_keys").write_text("# tên khoá\nstudio k-studio\n")
    os.environ.update(ASR_KEYS_DIR=str(d), ASR_JOBS_DIR=str(d / "jobs"), ASR_ALIGN_PRELOAD="0", ASR_KEY_MAX_QUEUE="2")
    sys.modules["asrserver.engine"] = types.SimpleNamespace(Engine=FakeEngine)
    sys.modules.pop("asrserver.app", None)
    from fastapi.testclient import TestClient

    from asrserver import app as appmod

    appmod.ALIGNER = FakeAligner()
    with TestClient(appmod.app) as c:
        c.mod = appmod
        yield c


H = {"Authorization": "Bearer k-main"}


def run_job(client, body, headers=H):
    r = client.post("/v1/asr/jobs", json=body, headers=headers)
    assert r.status_code == 202, r.text
    j = client.get(f"/v1/asr/jobs/{r.json()['job_id']}?wait=20", headers=headers).json()
    assert j["status"] in ("succeeded", "failed"), j
    return j


def test_auth(client):
    assert client.get("/health").status_code == 200
    assert client.post("/v1/uploads", content=b"x").status_code == 401
    assert client.post("/v1/uploads", content=b"x", headers={"X-API-Key": "sai"}).status_code == 401


def test_upload_dedupe(client):
    data = tone_wav()
    a = client.post("/v1/uploads", content=data, headers=H).json()
    b = client.post("/v1/uploads", content=data, headers={"Authorization": "Bearer k-studio"}).json()
    assert a == b and a["bytes"] == len(data) and len(a["upload_id"]) == 32
    assert client.post("/v1/uploads", content=b"", headers=H).status_code == 400


def test_text_mode_forced_language(client):
    uid = client.post("/v1/uploads", content=tone_wav(), headers=H).json()["upload_id"]
    FakeEngine.calls.clear()
    j = run_job(client, {"upload_id": uid, "mode": "text", "language": "Vietnamese"})
    res = j["result"]
    assert j["status"] == "succeeded" and res["status"] == "ok"
    assert res["text"].startswith("chữ") and res["language"] == ",vi,"
    assert FakeEngine.calls == ["Vietnamese"]   # 3 s tiếng → một khúc, ép ngôn ngữ
    assert abs(res["duration_sec"] - 3.0) < 0.05


def test_text_mode_silent(client):
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1), w.setsampwidth(2), w.setframerate(SR), w.writeframes(bytes(2 * SR * 2))
    uid = client.post("/v1/uploads", content=buf.getvalue(), headers=H).json()["upload_id"]
    assert run_job(client, {"upload_id": uid, "mode": "text"})["result"]["status"] == "silent"


def test_cues_mode_upload(client):
    uid = client.post("/v1/uploads", content=tone_wav(4, 1), headers=H).json()["upload_id"]
    res = run_job(client, {"upload_id": uid})["result"]
    assert res["status"] == "ok" and res["cues"] and res["vtt"].startswith("WEBVTT")


def test_align_mode(client):
    uid = client.post("/v1/uploads", content=tone_wav(), headers=H).json()["upload_id"]
    res = run_job(client, {"upload_id": uid, "mode": "align", "text": "xin chào các em"})["result"]
    assert res["status"] == "ok" and [w[0] for w in res["words"]] == ["xin", "chào", "các", "em"]


def test_validation(client):
    uid = client.post("/v1/uploads", content=tone_wav(), headers=H).json()["upload_id"]
    bad = [{"upload_id": uid, "audio_url": "https://x/a.m3u8"}, {}, {"upload_id": uid, "mode": "align"},
           {"upload_id": "../../etc/passwd"}, {"upload_id": uid, "mode": "text", "language": "Klingon"}]
    for body in bad:
        assert client.post("/v1/asr/jobs", json=body, headers=H).status_code == 400, body
    assert client.post("/v1/asr/jobs", json={"upload_id": "0" * 32}, headers=H).status_code == 404


def test_key_queue_cap(client):
    """Khoá phụ chỉ được KEY_MAX_QUEUE job chờ (khoá main không giới hạn)."""
    mod = client.mod
    uid = client.post("/v1/uploads", content=tone_wav(), headers=H).json()["upload_id"]
    hs = {"Authorization": "Bearer k-studio"}
    mod.S.queued.update({"giả1", "giả2"})
    mod.S.jobs.update({j: mod.Job(job_id=j, key_name="studio", priority="normal", seq=0) for j in ("giả1", "giả2")})
    try:
        assert client.post("/v1/asr/jobs", json={"upload_id": uid}, headers=hs).status_code == 429
        assert client.post("/v1/asr/jobs", json={"upload_id": uid, "ref": "main"}, headers=H).status_code == 202
    finally:
        mod.S.queued.difference_update({"giả1", "giả2"})


@pytest.fixture(scope="module")
def http_dir(tmp_path_factory):
    d = tmp_path_factory.mktemp("www")
    handler = partial(http.server.SimpleHTTPRequestHandler, directory=str(d))
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield d, f"http://127.0.0.1:{srv.server_address[1]}"
    srv.shutdown()


def test_media_routing(http_dir, monkeypatch):
    from asrserver import hls, media

    d, base = http_dir
    seen = []
    monkeypatch.setattr(hls, "load_audio", lambda url, *a, **kw: seen.append(url) or "hls")
    (d / "v.m3u8").write_text("#EXTM3U\n")
    (d / "video1").write_text("#EXTM3U\n#EXT-X-STREAM-INF:BANDWIDTH=1\nx.m3u8\n")
    (d / "page.html").write_text("<!DOCTYPE html><html><body>TikTok</body></html>")
    (d / "a.wav").write_bytes(tone_wav())
    assert media.load_url(f"{base}/v.m3u8") == "hls"                 # HLS đi thẳng, không tải trước
    assert media.load_url(f"{base}/video1?id=1&type=1") == "hls"     # playlist không đuôi .m3u8: nhìn nội dung
    assert seen == [f"{base}/v.m3u8", f"{base}/video1?id=1&type=1"]
    with pytest.raises(media.NotMedia):
        media.load_url(f"{base}/page.html")
    with pytest.raises(hls.NoVideo):
        media.load_url(f"{base}/khong-co.mp4")
    a = media.load_url(f"{base}/a.wav")
    assert a.wav is not None and abs(a.seconds - 3.0) < 0.05 and a.playlist == f"{base}/a.wav"
    assert abs(media.load_url(f"{base}/a.wav", max_seconds=1).seconds - 1.0) < 0.01


@pytest.mark.skipif(not shutil.which("ffmpeg"), reason="cần ffmpeg để tạo mp4")
def test_decode_mp4_and_no_audio(tmp_path):
    from asrserver import media

    mp4, silent = tmp_path / "a.mp4", tmp_path / "v.mp4"
    subprocess.run(["ffmpeg", "-loglevel", "error", "-f", "lavfi", "-i", "sine=frequency=300:duration=2", "-f", "lavfi",
                    "-i", "color=c=black:s=64x64:d=2", "-shortest", "-c:v", "libx264", "-c:a", "aac", str(mp4)], check=True)
    subprocess.run(["ffmpeg", "-loglevel", "error", "-f", "lavfi", "-i", "color=c=black:s=64x64:d=2", "-c:v", "libx264",
                    str(silent)], check=True)
    a = media.decode(str(mp4))
    assert a.wav is not None and abs(a.seconds - 2.0) < 0.1
    assert media.decode(str(silent)).wav is None


def test_align_cues_chunks_and_offsets():
    """align_cues: câu gom thành khúc ≤ chunk_s, mốc từ cộng đầu khúc (trừ đệm), ngôn ngữ theo số đông câu của khúc."""
    from asrserver import aligner

    seen = []

    class Al:
        def align(self, wav, text, language, path=None):
            seen.append((round(len(wav) / SR, 1), text, language))
            return [[w, i * 1.0, i * 1.0 + 0.5] for i, w in enumerate(text.split())]

    cues = [{"s": 10.0, "e": 14.0, "t": "câu một", "lang": "vi"}, {"s": 15.0, "e": 19.0, "t": "câu hai", "lang": "vi"},
            {"s": 200.0, "e": 204.0, "t": "hello there", "lang": "en"}]
    words = aligner.align_cues(Al(), np.zeros(300 * SR, np.float32), cues, chunk_s=150)
    assert [x[1:] for x in seen] == [("câu một câu hai", "Vietnamese"), ("hello there", "English")]
    assert seen[0][0] == 9.6                                   # 9,7 → 19,3 (đệm 0,3 s)
    assert words[0] == ["câu", 9.7, 10.2, 0] and words[-1] == ["there", 200.7, 201.2, 2]
    assert [w[3] for w in words] == [0, 0, 1, 1, 2, 2]                     # khúc nhiều câu: chia theo số từ
    one = aligner.align_cues(Al(), np.zeros(300 * SR, np.float32), cues)   # mặc định mỗi câu một khúc
    assert [w[3] for w in one] == [0, 0, 1, 1, 2, 2] and len(seen) == 5


def test_cues_mode_words_option(client):
    uid = client.post("/v1/uploads", content=tone_wav(4.0), headers=H).json()["upload_id"]
    res = run_job(client, {"upload_id": uid, "options": {"words": True}})["result"]
    assert res["cues"] and res["words"] and all(len(w) == 4 for w in res["words"])
    res = run_job(client, {"upload_id": uid})["result"]
    assert "words" not in res


def test_align_mode_with_cues(client):
    uid = client.post("/v1/uploads", content=tone_wav(4.0), headers=H).json()["upload_id"]
    cues = [{"s": 0.2, "e": 2.0, "t": "xin chào", "lang": "vi"}]
    res = run_job(client, {"upload_id": uid, "mode": "align", "cues": cues})["result"]
    assert res["status"] == "ok" and [w[0] for w in res["words"]] == ["xin", "chào"]
    bad = client.post("/v1/asr/jobs", json={"upload_id": uid, "cues": cues}, headers=H)   # cues mà không phải align
    assert bad.status_code == 400

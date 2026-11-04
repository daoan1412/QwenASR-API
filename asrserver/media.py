"""Nguồn tiếng ngoài HLS: tệp đã upload, hoặc link tới tệp tiếng/video (mp4, mp3, wav, m4a…) → Audio như hls.load_audio.

Link HLS (.m3u8, .ts kiểu cũ) vẫn đi thẳng hls.load_audio như trước, wav ra không đổi. Link khác thì tải về rồi nhìn nội
dung: playlist (#EXTM3U, vd. media.moon.vn/video1?id=…) → hls.load_audio; trang web → NotMedia; còn lại giải mã bằng PyAV.
"""
import io
import os
import urllib.error
import urllib.request
from urllib.parse import urlsplit

import av
import numpy as np

from . import hls
from .hls import SR, Audio, NoVideo

MAX_BYTES = int(os.environ.get("ASR_MAX_UPLOAD_MB", "512")) << 20


class NotMedia(ValueError):
    """Nội dung không phải tiếng/video (trang web, tệp quá lớn) — client gửi sai, không thử lại."""


def is_hls(url):
    path = urlsplit(url).path
    return path.endswith((".m3u8", ".ts"))


def fetch(url, limit=MAX_BYTES, timeout=60):
    """Tải cả tệp (≤ limit byte). 403/404/410 → NoVideo như HLS."""
    try:
        with urllib.request.urlopen(urllib.request.Request(url, headers=hls.UA), timeout=timeout) as r:
            data = r.read(limit + 1)
    except urllib.error.HTTPError as e:
        e.close()
        if e.code in hls.NO_VIDEO_HTTP:
            raise NoVideo(f"HTTP {e.code}") from None
        raise
    if len(data) > limit:
        raise NotMedia(f"tệp lớn hơn {limit >> 20} MB")
    return data


def load_url(url, max_seconds=0, progress=None):
    if not url or not str(url).startswith("http") or is_hls(url):
        return hls.load_audio(url, max_seconds, progress=progress)
    data = fetch(url)
    head = data[:1024].lstrip().lower()
    if head.startswith(b"#extm3u"):
        return hls.load_audio(url, max_seconds, progress=progress)
    if head.startswith((b"<!doctype html", b"<html")) or b"<html" in head:
        raise NotMedia("link là trang web, không phải tệp tiếng/video")
    return decode(io.BytesIO(data), max_seconds, playlist=url)


def decode(src, max_seconds=0, playlist=None):
    """Tệp (đường dẫn hoặc file-like) → mono float32 16 kHz, chỉ `max_seconds` giây đầu (0 = cả tệp)."""
    with av.open(src) as c:
        duration = round(c.duration / av.time_base, 2) if c.duration else None
        if not c.streams.audio:
            return Audio(playlist, duration or 0.0, 0.0, None)
        rs, parts, n = av.AudioResampler(format="flt", layout="mono", rate=SR), [], 0
        cap = int(max_seconds * SR) if max_seconds else None
        for f in c.decode(audio=0):
            for o in rs.resample(f):
                parts.append(o.to_ndarray().reshape(-1))
                n += parts[-1].size
            if cap and n >= cap:
                break
        else:
            parts += [o.to_ndarray().reshape(-1) for o in rs.resample(None)]
    wav = np.concatenate(parts).astype(np.float32) if parts else np.zeros(0, np.float32)
    if cap:
        wav = wav[:cap]
    seconds = round(len(wav) / SR, 2)
    return Audio(playlist, duration or seconds, seconds, wav)

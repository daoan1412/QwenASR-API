"""HLS → audio 16 kHz. Chép từ moon-ml/projects/video_transcript v2; chỉ thêm tham số progress."""
import io, re, time, urllib.error, urllib.request
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from urllib.parse import urljoin
import av
import numpy as np

SR = 16_000
UA = {"User-Agent": "Mozilla/5.0"}
NO_VIDEO_HTTP = (403, 404, 410)

class NoVideo(Exception): pass

@dataclass
class Audio:
    playlist: str
    duration: float        # cả video, theo EXTINF
    seconds: float         # phần đã tải (≤ max_seconds)
    wav: np.ndarray | None  # None = video không có luồng tiếng

def get(url, timeout=60, tries=3):
    """Thử lại lỗi mạng tạm thời (thêm cho server); 4xx trả ngay."""
    for k in range(tries):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=timeout) as r:
                return r.read()
        except urllib.error.HTTPError as e:
            if e.code < 500 or k == tries - 1:
                raise
        except (urllib.error.URLError, TimeoutError, ConnectionError, OSError):
            if k == tries - 1:
                raise
        time.sleep(2 * (k + 1))

def media_playlist(url):
    """master → luồng băng thông thấp nhất; link .ts kiểu cũ (…/stream_1/stream_1-3.ts) → playlist con."""
    if url.endswith(".ts"):
        m = re.match(r"(.*)/(stream_\d+)/\2-\d+\.ts$", url)
        if not m:
            raise ValueError(f"Không đoán được playlist từ {url}")
        url = f"{m.group(1)}/{m.group(2)}.m3u8"
    text = get(url).decode()
    if "#EXT-X-STREAM-INF" not in text:
        return url
    variants, bw = [], None
    for ln in text.splitlines():
        if ln.startswith("#EXT-X-STREAM-INF"):
            m = re.search(r"BANDWIDTH=(\d+)", ln)
            bw = int(m.group(1)) if m else 10**9
        elif ln.strip() and not ln.startswith("#") and bw is not None:
            variants.append((bw, urljoin(url, ln.strip())))
            bw = None
    return min(variants)[1]

def segments(playlist):
    out, t, dur = [], 0.0, None
    for ln in get(playlist).decode().splitlines():
        if ln.startswith("#EXTINF:"):
            dur = float(ln[8:].split(",")[0])
        elif ln.strip() and not ln.startswith("#") and dur is not None:
            out.append((t, dur, urljoin(playlist, ln.strip())))
            t += dur
            dur = None
    return out

def decode_audio(data):
    """Khúc .ts liền mạch → mono float32 16 kHz; None nếu không có luồng tiếng."""
    chunks = []
    with av.open(io.BytesIO(data), format="mpegts") as c:
        if not c.streams.audio:
            return None
        rs = av.AudioResampler(format="flt", layout="mono", rate=SR)
        for f in c.decode(audio=0):
            for o in rs.resample(f):
                chunks.append(o.to_ndarray().reshape(-1))
        for o in rs.resample(None):
            chunks.append(o.to_ndarray().reshape(-1))
    return np.concatenate(chunks).astype(np.float32) if chunks else np.zeros(0, np.float32)

def fit(wav, seconds):
    n = int(round(seconds * SR))
    return wav[:n] if len(wav) >= n else np.concatenate([wav, np.zeros(n - len(wav), np.float32)])

def load_audio(url, max_seconds=0, workers=16, group=60, progress=None, ahead=2):
    """progress(done, total): gọi sau mỗi nhóm đoạn .ts. Server tải trước `ahead` nhóm trong lúc giải mã;
    ranh giới nhóm giữ như bản gốc nên wav không đổi."""
    if not url or not str(url).startswith("http"):
        raise NoVideo("không có link video")
    try:
        playlist = media_playlist(url)
        segs = segments(playlist)
    except urllib.error.HTTPError as e:
        if e.code in NO_VIDEO_HTTP:
            raise NoVideo(f"HTTP {e.code}") from e
        raise
    if not segs:
        raise NoVideo("HLS rỗng")
    duration = segs[-1][0] + segs[-1][1]
    if max_seconds:
        segs = [s for s in segs if s[0] < max_seconds]
    parts, has_audio = [], False
    groups = [segs[i:i + group] for i in range(0, len(segs), group)]
    ex = ThreadPoolExecutor(workers)
    try:
        futs = {}
        for k in range(len(groups)):
            for j in range(k, min(k + ahead + 1, len(groups))):
                if j not in futs:
                    futs[j] = [ex.submit(get, s[2]) for s in groups[j]]
            g = groups[k]
            wav = decode_audio(b"".join(f.result() for f in futs.pop(k)))
            has_audio |= wav is not None
            parts.append(fit(wav if wav is not None else np.zeros(0, np.float32), sum(s[1] for s in g)))
            if progress:
                progress(min((k + 1) * group, len(segs)), len(segs))
    finally:
        ex.shutdown(wait=False, cancel_futures=True)
    seconds = segs[-1][0] + segs[-1][1]
    return Audio(playlist, round(duration, 2), round(seconds, 2), np.concatenate(parts) if has_audio else None)

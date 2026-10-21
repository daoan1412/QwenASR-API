"""Chia khúc + gán từ. Chép từ moon-ml/projects/video_transcript v2 (bỏ only_languages bản HF, xem langlock.py)."""
import difflib, re
import numpy as np
SR = 16_000
HOP = 0.05

def frame_rms(wav):
    win = int(HOP * SR)
    n = len(wav) // win
    return np.sqrt((wav[: n * win].reshape(n, win) ** 2).mean(axis=1)) if n else np.zeros(0, np.float32)

def cut(rms, end, target=6.0, slack=3.0, start=0.0):
    """Chia [start, end] thành khúc ~target giây, cắt ở ô nhỏ tiếng nhất trong target ± slack."""
    bounds, s = [], start
    while end - s > target + slack:
        lo, hi = int((s + target - slack) / HOP), int((s + target + slack) / HOP)
        c = (lo + int(np.argmin(rms[lo:hi]))) * HOP if lo < hi <= len(rms) else s + target
        bounds.append((round(s, 2), round(c, 2)))
        s = c
    bounds.append((round(s, 2), round(end, 2)))
    return bounds

def threshold(rms):
    return max(1e-3, 0.1 * float(np.percentile(rms, 95))) if len(rms) else 1e-3

def voiced_span(rms, s, e, thr, min_voiced=0.3):
    a, b = int(s / HOP), int(e / HOP)
    loud = np.flatnonzero(rms[a:b] > thr)
    if len(loud) * HOP < min_voiced:
        return None
    return round(max(s, s + loud[0] * HOP - 0.1), 2), round(min(e, s + (loud[-1] + 1) * HOP + 0.1), 2)

def _norm(w):
    return re.sub(r"[^\w]", "", w.lower())

def assign(words, subs, weights=None):
    """Từ của lượt khúc dài → chỉ số khúc ngắn (không giảm)."""
    if not words or not subs:
        return [0] * len(words)
    flat = [(_norm(w), k) for k, ws in enumerate(subs) for w in ws]
    sm = difflib.SequenceMatcher(None, [_norm(w) for w in words], [f[0] for f in flat], autojunk=False)
    idx = [None] * len(words)
    for a, b, n in sm.get_matching_blocks():
        for j in range(n):
            idx[a + j] = flat[b + j][1]
    known = [i for i, k in enumerate(idx) if k is not None]
    if not known:
        w = np.asarray(weights or [1.0] * len(subs), float)
        edges = np.cumsum(w) / w.sum()
        return [int(np.searchsorted(edges, (i + 0.5) / len(words))) for i in range(len(words))]
    first, last = known[0], known[-1]
    for i in range(first):
        idx[i] = idx[first]
    for i in range(last + 1, len(words)):
        idx[i] = idx[last]
    for left, right in zip(known, known[1:]):
        m, kl, kr = right - left - 1, idx[left], idx[right]
        for r in range(m):
            idx[left + 1 + r] = kl + round((r + 1) * (kr - kl) / (m + 1))
    return idx

"""Hai lượt chép (khúc ~20 s rồi ~6 s) → câu phụ đề. Chép nguyên từ moon-ml/projects/video_transcript v2."""
from . import asr, cues
from .asr import SR

VERSION = "v2"

def transcribe(audio, transcriber, cfg):
    """→ (Status, cues). transcriber: list khúc float32 → [(lang, text)]."""
    if audio.wav is None:
        return "no_audio", []
    wav, mv = audio.wav, cfg.get("min_voiced_s", 0.3)
    rms = asr.frame_rms(wav)
    thr = asr.threshold(rms)
    windows = []  # (s, e, [(s, e, phần có tiếng)])
    for s, e in asr.cut(rms, len(wav) / SR, cfg.get("window_s", 20.0), cfg.get("window_slack_s", 5.0)):
        subs = [(a, b, v) for a, b in asr.cut(rms, e, cfg.get("chunk_s", 6.0), cfg.get("chunk_slack_s", 3.0), start=s)
                if (v := asr.voiced_span(rms, a, b, thr, mv))]
        if subs:
            windows.append((s, e, subs))
    if not windows:
        return "silent", []
    clip = lambda s, e: wav[int(s * SR):int(e * SR)]
    first = transcriber([clip(s, e) for s, e, _ in windows])
    need = [(w, k) for w, ((_, _, subs), (_, text)) in enumerate(zip(windows, first)) if text and len(subs) > 1
            for k in range(len(subs))]
    second = dict(zip(need, transcriber([clip(*windows[w][2][k][:2]) for w, k in need]))) if need else {}
    chunks = []
    for w, ((_, _, subs), (lang, text)) in enumerate(zip(windows, first)):
        words = text.split()
        if not words:
            continue
        if len(subs) == 1:
            idx = [0] * len(words)
        else:
            idx = asr.assign(words, [second[(w, k)][1].split() for k in range(len(subs))],
                             [v[1] - v[0] for _, _, v in subs])
        for k in sorted(set(idx)):
            v = subs[k][2]
            chunks.append({"s": v[0], "e": v[1], "lang": lang,
                           "text": " ".join(x for x, i in zip(words, idx) if i == k)})
    out = cues.build(chunks, cfg.get("max_chars", 84))
    return ("ok" if out else "silent"), out

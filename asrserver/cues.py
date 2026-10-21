"""Câu phụ đề + VTT. Chép nguyên từ moon-ml/projects/video_transcript v2."""
import math
from collections import Counter
LANG_CODE = {"Vietnamese": "vi", "English": "en", "Chinese": "zh", "Cantonese": "yue", "Japanese": "ja", "Korean": "ko",
             "French": "fr", "German": "de", "Spanish": "es", "Russian": "ru", "Thai": "th", "Indonesian": "id"}
PUNCT = ".,?!;:…"

def lang_code(name):
    return None if not name else LANG_CODE.get(name, name.lower()[:8])

def split_text(text, max_chars):
    text = " ".join(text.split())
    if len(text) <= max_chars:
        return [text] if text else []
    target = len(text) / math.ceil(len(text) / max_chars)
    parts, cur = [], ""
    for w in text.split(" "):
        if cur and (len(cur) + 1 + len(w) > min(max_chars, target * 1.15)
                    or (cur[-1] in PUNCT and len(cur) >= 0.6 * target)):
            parts.append(cur)
            cur = w
        else:
            cur = f"{cur} {w}" if cur else w
    if cur:
        parts.append(cur)
    return parts

def build(chunks, max_chars=84):
    """chunks [{s,e,lang,text}] → câu phụ đề [{s,e,t,lang}]."""
    out = []
    for c in chunks:
        parts = split_text(c["text"], max_chars)
        total = sum(len(p) for p in parts)
        t = c["s"]
        for p in parts:
            e = c["e"] if p is parts[-1] else round(t + (c["e"] - c["s"]) * len(p) / total, 2)
            out.append({"s": round(t, 2), "e": e, "t": p, "lang": lang_code(c.get("lang"))})
            t = e
    return out

def languages(cues):
    n = Counter(c["lang"] for c in cues if c.get("lang"))
    return "," + ",".join(k for k, _ in n.most_common()) + ","

def wrap(text, line_chars=42):
    if len(text) <= line_chars or " " not in text:
        return text
    mid = len(text) / 2
    i = min((i for i, ch in enumerate(text) if ch == " "), key=lambda i: abs(i - mid))
    return text[:i] + "\n" + text[i + 1:]

def ts(sec):
    ms = int(round(sec * 1000))
    return f"{ms // 3_600_000:02d}:{ms // 60_000 % 60:02d}:{ms // 1000 % 60:02d}.{ms % 1000:03d}"

def escape(text):
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

def to_vtt(cues, line_chars=42):
    body = "".join(f"{ts(c['s'])} --> {ts(c['e'])}\n{escape(wrap(c['t'], line_chars))}\n\n" for c in cues)
    return "WEBVTT\n\n" + body

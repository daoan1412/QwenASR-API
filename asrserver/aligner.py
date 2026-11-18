"""Mốc từng từ: Qwen3-ForcedAligner-0.6B (bản transformers của qwen-asr) chạy trong tiến trình API, một yêu cầu một lúc.

Thay align_words.py --worker của ai-video-skills (trước đây mỗi máy tự nạp mô hình trên CPU/GPU). ~1,5 GB VRAM bf16.
"""
import os
import threading
from collections import Counter

from .hls import SR

MODEL = os.environ.get("ASR_ALIGN_MODEL", "Qwen/Qwen3-ForcedAligner-0.6B")
DEVICE = os.environ.get("ASR_ALIGN_DEVICE", "cuda:0")
MAX_SECONDS = 180  # qwen_asr MAX_FORCE_ALIGN_INPUT_SECONDS


class Aligner:
    def __init__(self):
        self.model = None
        self.lock = threading.Lock()

    def load(self):
        with self.lock:
            if self.model is None:
                import torch
                from qwen_asr import Qwen3ForcedAligner

                self.model = Qwen3ForcedAligner.from_pretrained(
                    MODEL, dtype=torch.float32 if DEVICE == "cpu" else torch.bfloat16, device_map=DEVICE)
        return self.model

    def align(self, wav, text, language, path=None):
        """→ [[từ, giây bắt đầu, giây kết thúc], …]. `path` (tệp upload): để qwen_asr tự đọc tệp như align_words.py cũ —
        đo 31/10/2026 trên 3 đoạn TTS: giống hệt từng mốc; đưa wav PyAV thì lệch TB 22–51 ms, có từ 480 ms."""
        if len(wav) > MAX_SECONDS * SR:
            raise ValueError(f"align: tiếng dài hơn {MAX_SECONDS} s")
        model = self.load()
        with self.lock:
            items = model.align(audio=path or (wav, SR), text=text, language=language)[0]
            if DEVICE != "cpu":   # trả bộ đệm activation (GPU chung với vLLM Qwen 27B + moon-embed, còn rất ít chỗ)
                import torch

                torch.cuda.empty_cache()
        return [[it.text, round(float(it.start_time), 3), round(float(it.end_time), 3)] for it in items]


    def align_many(self, wavs, texts, languages):
        """Nhiều khúc một lượt (qwen_asr nhận lô) → [[[từ, s, e], …] mỗi khúc]."""
        model = self.load()
        with self.lock:
            res = model.align(audio=[(w, SR) for w in wavs], text=list(texts), language=list(languages))
            if DEVICE != "cpu":
                import torch

                torch.cuda.empty_cache()
        return [[[it.text, round(float(it.start_time), 3), round(float(it.end_time), 3)] for it in r] for r in res]

# mã ngôn ngữ của câu phụ đề (cues.lang_code) → tên ngôn ngữ của aligner; mã khác → Vietnamese
LANG_NAMES = {"vi": "Vietnamese", "en": "English"}
CHUNK_S = 0.0     # gom câu thành khúc ≤ bấy nhiêu giây; 0 = mỗi câu một khúc (xem align_cues)
PAD_S = 0.3       # đệm hai đầu khúc để từ đầu / cuối không bị cắt
BATCH = 32        # số khúc mỗi lượt gọi aligner


def align_cues(al, wav, cues, chunk_s=CHUNK_S, pad=PAD_S, batch=BATCH):
    """Câu phụ đề [{s, e, t, lang}] + tiếng cả video (16 kHz, cùng trục thời gian với câu) → [[từ, s, e, câu], …] mốc tính từ
    đầu video, `câu` = chỉ số câu (từ 0) chứa từ — aligner có thể tách / gộp từ khác cách tách theo khoảng trắng (đo 200 video:
    9 video lệch 3–19 từ), nên ghép từ ↔ câu theo chỉ số này, không theo thứ tự. Khúc nhiều câu (chunk_s > 0) thì chỉ số câu
    chia theo số từ nếu khớp, không thì theo giây. Câu liền nhau gom thành khúc ≤ `chunk_s`
    giây (0 = mỗi câu một khúc, nằm trong khung giờ của chính câu); mỗi khúc một mẫu aligner, gửi theo lô `batch`; ngôn ngữ
    = ngôn ngữ nhiều câu nhất của khúc. Thêm 10/11/2026 cho mục lục video (moon-ml video_segment_v2): ranh giới đoạn chốt
    tới từ thay vì tới câu phụ đề (~3 s). Đo video 12242547 (53 phút): khúc 150 s thì aligner trôi — 10% câu có từ đầu lệch
    > 26 s so với mốc câu, một chuỗi 211 từ dồn một mốc."""
    groups, cur = [], []
    for c in cues:
        if cur and float(c["e"]) - float(cur[0]["s"]) > chunk_s:
            groups.append(cur)
            cur = []
        cur.append(c)
    if cur:
        groups.append(cur)
    total, items = len(wav) / SR, []
    pos = {id(c): k for k, c in enumerate(cues)}
    for g in groups:
        text = " ".join(str(c.get("t") or "").strip() for c in g).strip()
        if not text:
            continue
        a, b = max(0.0, float(g[0]["s"]) - pad), min(total, float(g[-1]["e"]) + pad)
        b = min(b, a + MAX_SECONDS)   # nhóm chỉ vượt khi một câu đơn quá dài — cắt bớt đuôi
        lang = LANG_NAMES.get(Counter(c.get("lang") for c in g).most_common(1)[0][0], "Vietnamese")
        items.append((a, wav[int(a * SR):int(b * SR)], text, lang, [pos[id(c)] for c in g], g))
    out = []
    for k in range(0, len(items), batch):
        part = items[k:k + batch]
        if hasattr(al, "align_many"):
            res = al.align_many([x[1] for x in part], [x[2] for x in part], [x[3] for x in part])
        else:
            res = [al.align(x[1], x[2], x[3]) for x in part]
        for (a, _, _, _, idx, g), words in zip(part, res):
            out += [[w, round(s + a, 3), round(e + a, 3), k] for (w, s, e), k in zip(words, cue_of(words, g, idx, a))]
    return out


def cue_of(words, group, idx, a):
    """Chỉ số câu của từng từ trong một khúc: khúc một câu → câu đó; nhiều câu → theo số từ mỗi câu nếu tổng khớp, không
    thì câu có khung giờ chứa giữa từ (gần nhất)."""
    if len(idx) == 1:
        return [idx[0]] * len(words)
    counts = [len(str(c.get("t") or "").split()) for c in group]
    if sum(counts) == len(words):
        return [k for k, n in zip(idx, counts) for _ in range(n)]
    out = []
    for _, s, e in words:
        t = a + (s + e) / 2
        out.append(min(zip(idx, group), key=lambda x: 0 if float(x[1]["s"]) <= t <= float(x[1]["e"])
                       else min(abs(t - float(x[1]["s"])), abs(t - float(x[1]["e"]))))[0])
    return out

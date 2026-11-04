"""Mốc từng từ: Qwen3-ForcedAligner-0.6B (bản transformers của qwen-asr) chạy trong tiến trình API, một yêu cầu một lúc.

Thay align_words.py --worker của ai-video-skills (trước đây mỗi máy tự nạp mô hình trên CPU/GPU). ~1,5 GB VRAM bf16.
"""
import os
import threading

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

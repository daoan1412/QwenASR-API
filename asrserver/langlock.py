"""Khoá ngôn ngữ: ở token ngay sau "language" chỉ cho các tên trong extra_args["allowed_lang_ids"].

Tương đương OnlyLanguages bản HF (chỉ bước thứ hai của câu trả lời). Chạy trong tiến trình EngineCore.
"""
import torch
from vllm.v1.sample.logits_processor import AdapterLogitsProcessor

LANG_TOK = 11528  # "language"


class LangLock(AdapterLogitsProcessor):
    def is_argmax_invariant(self) -> bool:
        return False

    def new_req_logits_processor(self, params):
        allowed = (params.extra_args or {}).get("allowed_lang_ids")
        if not allowed:
            return None

        def lock(output_ids, logits):
            if len(output_ids) != 1 or output_ids[0] != LANG_TOK:
                return logits
            mask = torch.full_like(logits, float("-inf"))
            mask[allowed] = 0
            return logits + mask

        return lock

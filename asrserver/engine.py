"""Gọi Qwen3-ASR qua vLLM AsyncLLM: mỗi khúc là một request, khúc của mọi job được batch chung."""
import asyncio
import uuid

import numpy as np

from qwen_asr.core.transformers_backend import Qwen3ASRProcessor
from qwen_asr.core.vllm_backend import Qwen3ASRForConditionalGeneration
from vllm import ModelRegistry, SamplingParams
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.sampling_params import RequestOutputKind
from vllm.v1.engine.async_llm import AsyncLLM

from .asr import SR
from .langlock import LANG_TOK, LangLock

ModelRegistry.register_model("Qwen3ASRForConditionalGeneration", Qwen3ASRForConditionalGeneration)


def parse(raw):
    """"language X<asr_text>chữ" → (X | None, chữ)."""
    if "<asr_text>" not in raw:
        return None, "" if raw.startswith("language") else raw.strip()
    head, text = raw.split("<asr_text>", 1)
    lang = head.removeprefix("language").strip()
    return (None if lang in ("", "None") else lang), text.strip()


class Engine:
    def __init__(self, model, gpu_memory_utilization=0.6, max_model_len=4096, max_num_seqs=128, lang_lock=True):
        self.model = model
        self.lang_lock = lang_lock
        processor = Qwen3ASRProcessor.from_pretrained(model, fix_mistral_regex=True)
        self.tokenizer = processor.tokenizer
        assert self.tokenizer.encode("language", add_special_tokens=False) == [LANG_TOK]
        msgs = [{"role": "system", "content": ""}, {"role": "user", "content": [{"type": "audio", "audio": ""}]}]
        self.prompt = processor.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
        self.llm = AsyncLLM.from_engine_args(AsyncEngineArgs(
            model=model,
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=max_model_len,
            max_num_seqs=max_num_seqs,
            scheduling_policy="priority",
            logits_processors=[LangLock] if lang_lock else None,
            limit_mm_per_prompt={"audio": 1},
            attention_backend="FLASH_ATTN",   # bắt buộc cho VLLM_BATCH_INVARIANT
        ))

    def lang_ids(self, names):
        """["Vietnamese", "English"] → id token " Vietnamese", " English", " None"; ValueError nếu tên không là 1 token."""
        ids = []
        for n in [*names, "None"]:
            t = self.tokenizer.encode(f" {n}", add_special_tokens=False)
            if len(t) != 1:
                raise ValueError(f"ngôn ngữ không hợp lệ: {n!r}")
            ids.append(t[0])
        return ids

    async def one(self, wav, lang_ids, tokens_per_second=15.0, priority=1):
        sp = SamplingParams(
            temperature=0.0,
            max_tokens=int(tokens_per_second * len(wav) / SR + 32),
            output_kind=RequestOutputKind.FINAL_ONLY,
            extra_args={"allowed_lang_ids": lang_ids} if self.lang_lock else None,
        )
        final = None
        async for out in self.llm.generate({"prompt": self.prompt, "multi_modal_data": {"audio": [wav]}},
                                           sp, uuid.uuid4().hex, priority=priority):
            final = out
        return parse(final.outputs[0].text)

    async def warmup(self):
        """Request đầu tiên với mỗi độ dài khúc chậm ~10 s; chạy trước khi nhận job."""
        rng = np.random.default_rng(0)
        ids = self.lang_ids(["Vietnamese", "English"])
        for n in (1, 8, 32):
            clips = [0.01 * rng.standard_normal(int(sec * SR)).astype(np.float32)
                     for sec in np.linspace(0.5, 25, n)]
            await asyncio.gather(*(self.one(c, ids) for c in clips))

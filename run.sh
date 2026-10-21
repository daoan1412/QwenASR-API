#!/usr/bin/env bash
# Qwen3-ASR-1.7B job server (asrserver/app.py) on 127.0.0.1:8000
# Public at https://asr.daoan.online via the Cloudflare tunnel; key in .api_key
cd "$(dirname "$0")"
export VLLM_BATCH_INVARIANT="${VLLM_BATCH_INVARIANT:-1}"   # kết quả không phụ thuộc cách gom batch
exec .venv/bin/python -m asrserver.app "$@"

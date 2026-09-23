#!/usr/bin/env bash
# llama-server launcher for the native build with the tuned Qwen3.8-Flash-Next config:
# 4060 Ti only (the 2060 SUPER costs more than it gives), experts of 4 layers on the GPU (5 do not leave room for the 80k-context compute buffers at ub 2048), large ubatch for prefill.
# usage: [OT=<tensor overrides>] [LLAMA_HOT_EXPERTS=<placement>] customware/serve.sh [port] [extra llama-server args...]
#   hot experts: OT='.*_exps=CPU' LLAMA_HOT_EXPERTS=customware/hot-experts-mixed.txt customware/serve.sh   (default port 5001 so it can run next to the daily 5000)
set -euo pipefail
cd "$(dirname "$0")/.."
port=${1:-5001}; shift || true
model="C:/dev/textgen/user_data/models/Qwen3.8-Flash-Next-GGUF-UD-Q4_K_XL/Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf"
exec build/bin/llama-server.exe \
  -m "$model" \
  --jinja -fa on -ngl 99 -dev CUDA0 \
  -ot "${OT:-blk\.(2[8-9]|3[0-1])\..*_exps=CUDA0,.*_exps=CPU}" \
  --cache-type-k q8_0 --cache-type-v q8_0 \
  --load-mode mmap -np 1 -c 80000 -ub 2048 -b 2048 --no-warmup --cache-ram 2048 \
  --host 0.0.0.0 --port "$port" -t 6 "$@"

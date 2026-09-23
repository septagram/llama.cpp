#!/usr/bin/env bash
# llama-server launcher for the native build with the daily Qwen3.8-Flash-Next config.
# usage: customware/serve.sh [port] [extra llama-server args...]   (default port 5001 so it can run next to the daily 5000)
set -euo pipefail
cd "$(dirname "$0")/.."
port=${1:-5001}; shift || true
model="C:/dev/textgen/user_data/models/Qwen3.8-Flash-Next-GGUF-UD-Q4_K_XL/Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf"
exec build/bin/llama-server.exe \
  -m "$model" \
  --jinja -fa on -sm layer -ngl 99 \
  -ot 'blk\.(4[2-6])\..*_exps=CUDA0,blk\.(39|4[0-1])\..*_exps=CUDA1,.*_exps=CPU' \
  --cache-type-k q8_0 --cache-type-v q8_0 \
  --load-mode mmap -np 1 -c 80000 --no-warmup --cache-ram 2048 \
  --host 0.0.0.0 --port "$port" -t 6 "$@"

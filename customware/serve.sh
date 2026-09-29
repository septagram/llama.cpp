#!/usr/bin/env bash
# llama-server launcher for the native build with the tuned Qwen3.8-Flash-Next config:
# 4060 Ti only (the 2060 SUPER costs more than it gives), experts of 4 layers on the GPU (5 do not leave room for the 80k-context compute buffers at ub 2048), large ubatch for prefill.
# usage: [OT=<overrides>] [LLAMA_HOT_EXPERTS=<placement>] [CTX=160000] [NP=2] [UB=2048] [CACHE_RAM=2048] customware/serve.sh [port] [args...]
#   hot experts: OT='.*_exps=CPU' LLAMA_HOT_EXPERTS=customware/hot-experts-mixed.txt customware/serve.sh   (default port 5001 so it can run next to the daily 5000)
set -euo pipefail
cd "$(dirname "$0")/.."
port=${1:-5001}; shift || true
model="C:/dev/textgen/user_data/models/Qwen3.8-Flash-Next-GGUF-UD-Q4_K_XL/Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf"
# BIN: another binary for an A/B, e.g. the released one in C:/dev/llama.cpp-customware
exec "${BIN:-build/bin/llama-server.exe}" \
  -m "$model" \
  --jinja -fa on -ngl 99 -dev CUDA0 \
  -ot "${OT:-blk\.(2[8-9]|3[0-1])\..*_exps=CUDA0,.*_exps=CPU}" \
  --cache-type-k q8_0 --cache-type-v q8_0 \
  --load-mode mmap -np "${NP:-2}" --kv-unified -c "${CTX:-160000}" -ub "${UB:-2048}" -b 2048 --no-warmup --cache-ram "${CACHE_RAM:-2048}" \
  --host 0.0.0.0 --port "$port" -t 6 "$@"

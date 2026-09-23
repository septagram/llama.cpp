#!/usr/bin/env bash
# llama-bench wrapper for the Qwen3.8-Flash-Next expert-offload config used on this machine.
# usage: customware/bench.sh <label> <llama-bench.exe> [extra llama-bench args...]
# results append to customware/bench-results.md (markdown rows) and bench-results.jsonl.
set -euo pipefail
cd "$(dirname "$0")/.."
label=${1:?label}; bin=${2:?llama-bench path}; shift 2
model="C:/dev/textgen/user_data/models/Qwen3.8-Flash-Next-GGUF-UD-Q4_K_XL/Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf"
ot='blk\.(4[2-6])\..*_exps=CUDA0,blk\.(39|4[0-1])\..*_exps=CUDA1,.*_exps=CPU'
args=(-m "$model" -ngl 99 -sm layer -fa 1 -ctk q8_0 -ctv q8_0 -t 6 -ot "$ot" -r 5 --progress)
echo "### $label ($(date '+%F %T'))" >> customware/bench-results.md
echo "\`$bin $* \`" >> customware/bench-results.md
"$bin" "${args[@]}" "$@" -o md 2>customware/bench-last.err | tee -a customware/bench-results.md
"$bin" "${args[@]}" "$@" -o jsonl 2>/dev/null | sed "s/^{/{\"label\":\"$label\",/" >> customware/bench-results.jsonl || true

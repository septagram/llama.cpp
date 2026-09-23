#!/usr/bin/env bash
# llama-bench wrapper for the Qwen3.8-Flash-Next expert-offload config used on this machine.
# usage: customware/bench.sh <label> <llama-bench.exe> [extra llama-bench args...]
# raw jsonl appends to customware/bench-results.jsonl, a compact row with per-sample t/s to customware/bench-results.md,
# stderr (load log, profiler output) goes to customware/bench-<label>.err
set -euo pipefail
cd "$(dirname "$0")/.."
label=${1:?label}; bin=${2:?llama-bench path}; shift 2
model="C:/dev/textgen/user_data/models/Qwen3.8-Flash-Next-GGUF-UD-Q4_K_XL/Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf"
# llama-bench: ';' separates overrides inside one set, ',' would start another test variant
# env OT overrides the tensor placement, env DEV the device list (e.g. DEV=CUDA0 to leave the 2060 SUPER out)
ot=${OT:-'blk\.(4[2-6])\..*_exps=CUDA0;blk\.(39|4[0-1])\..*_exps=CUDA1;.*_exps=CPU'}
args=(-m "$model" -ngl 99 -sm layer -fa 1 -ctk q8_0 -ctv q8_0 -t 6 -ot "$ot" -r 5 --progress)
if [ -n "${DEV:-}" ]; then args+=(-dev "$DEV"); fi
"$bin" "${args[@]}" "$@" -o jsonl 2>"customware/bench-$label.err" | python customware/bench-record.py "$label" "$*"

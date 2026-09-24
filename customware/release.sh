#!/usr/bin/env bash
# Freeze the current native build as the "current best" runnable outside the repo, so rebuilds never
# lock or break the binary the daily launcher uses.
#   binaries -> C:\dev\llama.cpp-customware\   (llama-server.exe, llama-bench.exe, hot-experts.txt, VERSION.txt)
#   launcher -> C:\dev\textgen\Qwen3.8-Flash-Next-UD-Q4_K_XL.customware.sh
set -euo pipefail
cd "$(dirname "$0")/.."
dst=/c/dev/llama.cpp-customware
mkdir -p "$dst"
cp -f build/bin/llama-server.exe build/bin/llama-bench.exe "$dst"/
cp -f "${HOT_PLACEMENT:-customware/hot-experts-mixed-4000.txt}" "$dst/hot-experts.txt"
{ echo "commit: $(git rev-parse --short HEAD)"; echo "date:   $(date '+%F %T')"; git log --oneline -1; } > "$dst/VERSION.txt"
cat > /c/dev/textgen/Qwen3.8-Flash-Next-UD-Q4_K_XL.customware.sh <<'LAUNCH'
#!/bin/bash
# Current best config from the customware work (see C:\dev\llama.cpp-customware\VERSION.txt).
# Same context, slots and unified KV as the daily script (160k, 2 slots). Measured at 22k depth on a quiet box:
# ~160 t/s prefill, ~12.9 t/s generation (daily script before: ~70 / ~7), ~12.9 GB of VRAM. What changed and why:
#   -dev CUDA0                 the 2060 SUPER costs more than it gives (generation 7.8 -> 12 t/s without it)
#   LLAMA_HOT_EXPERTS          the most used ~1350 experts (4 GB) live on the 4060 Ti instead of whole layers;
#                              the placement was learned from real prompts. To regenerate it from your own sessions:
#                              run with GGML_CPU_PROFILE=5000 for a while (histogram lines in the server log), then
#                              python customware/expert-place.py --model <gguf> --budget-mib 4000 --out <file> <server log>
#   LLAMA_HOT_EXPERTS_ALIAS=1  the hot experts have no VRAM of their own: they are parked in the tail of the GPU
#                              compute buffer. A big prompt evicts them and one restore (~0.7 s) brings them back
#                              before generation, so 160k context + -ub 2048 + the hot set fit together
#   -ub 2048 -b 2048           expert weights are streamed to the GPU once per ubatch, bigger ubatches = faster prefill
#   GGML_OP_OFFLOAD_MIN_BATCH  prompt chunks below 512 tokens run on the CPU with the hot experts (faster than
#                              streaming 65 GB of experts for a short turn, and no restore afterwards)
# Fallbacks:
#   GPU shared with other work (diffusion etc): a smaller placement (--budget-mib 3000), or drop both LLAMA_HOT_EXPERTS
#   lines and use -ot "blk\.(2[8-9]|30)\..*_exps=CUDA0,.*_exps=CPU" with -ub 1024 instead of -ot ".*_exps=CPU"
# Note: anything else rendering on the 4060 Ti or spinning a core (a GUI app with continuous repaint, for example)
#   costs up to a third of the token rate; check Task Manager's GPU column if generation is slower than ~12 t/s.
cat $0
export LLAMA_HOT_EXPERTS=../llama.cpp-customware/hot-experts.txt
export LLAMA_HOT_EXPERTS_ALIAS=1
export GGML_OP_OFFLOAD_MIN_BATCH=512
../llama.cpp-customware/llama-server \
	-m user_data/models/Qwen3.8-Flash-Next-GGUF-UD-Q4_K_XL/Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf \
	--jinja \
	-fa on \
	-ngl 99 \
	-dev CUDA0 \
	-ot ".*_exps=CPU" \
	--cache-type-k q8_0 --cache-type-v q8_0 \
	--load-mode mmap \
	-np 2 \
	--kv-unified \
	-c 160000 \
	-ub 2048 -b 2048 \
	--no-warmup -fit off \
	--cache-ram 16384 \
	--host 0.0.0.0 --port 5000 \
	-lv 4 \
	-t 6
LAUNCH
chmod +x /c/dev/textgen/Qwen3.8-Flash-Next-UD-Q4_K_XL.customware.sh
echo "released $(git rev-parse --short HEAD) to $dst; launcher: C:\dev\textgen\Qwen3.8-Flash-Next-UD-Q4_K_XL.customware.sh"

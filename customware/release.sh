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
# Same slots and unified KV as the daily script; context 120k instead of 160k (agreed trade: it is what
# lets -ub 2048 fit, +50% prefill). Measured at 22k depth, 2 slots: ~169 t/s prefill, ~12.5 t/s generation
# (daily script before: ~70 / ~7). What changed and why:
#   -dev CUDA0             the 2060 SUPER costs more than it gives (generation 7.8 -> 12 t/s without it)
#   LLAMA_HOT_EXPERTS      the most used ~1400 experts (4 GB) live on the 4060 Ti instead of whole layers;
#                          the placement was learned from real prompts. To regenerate it from your own sessions:
#                          run with GGML_CPU_PROFILE=5000 for a while (histogram lines in the server log), then
#                          python customware/expert-place.py --model <gguf> --budget-mib 4000 --out <file> <server log>
#   -ub 2048 -b 2048       expert weights are streamed to the GPU once per ubatch, so bigger ubatches are faster;
#                          the QSA compute buffers scale with context x ubatch, 160k + 2048 does not fit on 16 GB
# VRAM ends up at ~14.6 of 16.4 GB (above ~15 GB Windows starts paging VRAM and everything crawls). Fallbacks:
#   need 160k context:     -c 160000 -ub 1024        (measured ~110 t/s prefill, ~12.9 t/s generation)
#   GPU shared with other  smaller placement (--budget-mib 3000) or drop LLAMA_HOT_EXPERTS and use
#   work (diffusion etc):  -ot "blk\.(2[8-9]|30)\..*_exps=CUDA0,.*_exps=CPU" instead of -ot ".*_exps=CPU"
cat $0
export LLAMA_HOT_EXPERTS=../llama.cpp-customware/hot-experts.txt
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
	-c 120000 \
	-ub 2048 -b 2048 \
	--no-warmup -fit off \
	--cache-ram 16384 \
	--host 0.0.0.0 --port 5000 \
	-lv 4 \
	-t 6
LAUNCH
chmod +x /c/dev/textgen/Qwen3.8-Flash-Next-UD-Q4_K_XL.customware.sh
echo "released $(git rev-parse --short HEAD) to $dst; launcher: C:\dev\textgen\Qwen3.8-Flash-Next-UD-Q4_K_XL.customware.sh"

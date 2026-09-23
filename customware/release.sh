#!/usr/bin/env bash
# Freeze the current native build as the "current best" runnable outside the repo, so rebuilds never
# lock or break the binary the daily launcher uses.
#   binaries -> C:\dev\llama.cpp-customware\   (llama-server.exe, llama-bench.exe, VERSION.txt)
#   launcher -> C:\dev\textgen\Qwen3.8-Flash-Next-UD-Q4_K_XL.customware.sh
set -euo pipefail
cd "$(dirname "$0")/.."
dst=/c/dev/llama.cpp-customware
mkdir -p "$dst"
cp -f build/bin/llama-server.exe build/bin/llama-bench.exe "$dst"/
{ echo "commit: $(git rev-parse --short HEAD)"; echo "date:   $(date '+%F %T')"; git log --oneline -1; } > "$dst/VERSION.txt"
cat > /c/dev/textgen/Qwen3.8-Flash-Next-UD-Q4_K_XL.customware.sh <<'LAUNCH'
#!/bin/bash
# Current best config from the customware work (see C:\dev\llama.cpp-customware\VERSION.txt).
# What changed vs the old script and why:
#   -dev CUDA0            the 2060 SUPER costs more than it gives (tg 7.8 -> 12 t/s without it)
#   experts 28-31 on GPU  4 layers is what fits next to the 80k-context buffers at -ub 2048
#   -ub 2048 -b 2048      expert weights are copied to the GPU once per ubatch, so big ubatches ~3x prefill
# VRAM sits at ~15.8 of 16.4 GB: if something else needs the 4060 Ti (e.g. diffusion), drop to 3 expert
# layers: blk\.(29|3[0-1])
cat $0
../llama.cpp-customware/llama-server \
	-m user_data/models/Qwen3.8-Flash-Next-GGUF-UD-Q4_K_XL/Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf \
	--jinja \
	-fa on \
	-ngl 99 \
	-dev CUDA0 \
	-ot "blk\.(2[8-9]|3[0-1])\..*_exps=CUDA0,.*_exps=CPU" \
	--cache-type-k q8_0 --cache-type-v q8_0 \
	--load-mode mmap \
	-np 1 \
	-c 80000 \
	-ub 2048 -b 2048 \
	--no-warmup \
	--cache-ram 2048 \
	--host 0.0.0.0 --port 5000 \
	-lv 4 \
	-t 6
LAUNCH
chmod +x /c/dev/textgen/Qwen3.8-Flash-Next-UD-Q4_K_XL.customware.sh
echo "released $(git rev-parse --short HEAD) to $dst; launcher: C:\dev\textgen\Qwen3.8-Flash-Next-UD-Q4_K_XL.customware.sh"

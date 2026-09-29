#!/usr/bin/env bash
# Entry 008 A/B: the plain depth series (22k / 44k / 88k, 64 greedy tokens, no profiling) on the released binary
# and on the build, same launcher-shaped config on port 5011. Prints ms/token per depth for both.
#   usage: customware/test-depth-ab.sh <label> [old binary]
set -uo pipefail
cd "$(dirname "$0")/.."
label=${1:?label}
old=${2:-/c/dev/llama.cpp-customware/llama-server.exe}
echo "== old: $old"
BIN="$old" PROF=0 bash customware/test-depth-prof.sh "$label-old" | grep -E "^(22k|44k|88k):|server up"
echo "== new: build/bin/llama-server.exe"
PROF=0 bash customware/test-depth-prof.sh "$label-new" | grep -E "^(22k|44k|88k):|server up"
echo "depth A/B $label done"

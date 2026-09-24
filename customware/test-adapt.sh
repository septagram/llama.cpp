#!/usr/bin/env bash
# Entry 004 test: adaptive hot experts. Phase A starts from the shipped placement with a fresh stats file, runs the
# code prompt twice and the prose prompt twice (coverage must rise on repeats and recover after the switch), and
# checks that the stats file was written. Phase B restarts with the placement computed from the stats and runs the
# code prompt once (coverage must start near where phase A ended).
#   usage: customware/test-adapt.sh <label> [CTX=160000] [UB=2048] [PORT=5011]
set -uo pipefail
cd "$(dirname "$0")/.."
label=${1:?label}
CTX=${CTX:-160000}; UB=${UB:-2048}; PORT=${PORT:-5011}
STATS=customware/hot-experts-$label.stats
rm -f "$STATS"

start() { # <phase> <env...>
  local phase=$1; shift
  log=customware/serve-$label-$phase.log
  : > "$log"
  (env "$@" OT='.*_exps=CPU' LLAMA_HOT_EXPERTS_ALIAS=1 GGML_OP_OFFLOAD_MIN_BATCH=512 LLAMA_HOT_EXPERTS_STATS="$STATS" LLAMA_HOT_EXPERTS_LOG="${LOG:-256}" CTX="$CTX" UB="$UB" bash customware/serve.sh "$PORT" -lv 4 > "$log" 2>&1 &)
  for i in $(seq 1 180); do
    sleep 5
    if curl -s -m 2 "http://127.0.0.1:$PORT/health" 2>/dev/null | grep -q '"ok"'; then echo "health ok after $((i*5)) s"; break; fi
    if grep -q "CUDA error\|error loading model\|failed to load\|GGML_ASSERT" "$log"; then echo "server failed"; grep -E "CUDA error|error loading|failed to load|GGML_ASSERT" "$log" | head -3; taskkill //F //IM llama-server.exe >/dev/null 2>&1; exit 1; fi
  done
  grep -E "hot experts|placement from|no usable stats|compute buffer size" "$log" | cut -c1-220
}
req() { # <name> <request file>
  curl -s -m 1800 "http://127.0.0.1:$PORT/v1/chat/completions" -H "Content-Type: application/json" -d @"$2" > "customware/ab-$label-$1.json"
  python -c "
import json
d=json.load(open('customware/ab-$label-$1.json'))
t=d.get('timings',{})
print('$label $1: prompt_n=%d (%.1f t/s) predicted_n=%d (%.2f t/s)' % (t.get('prompt_n',0), t.get('prompt_per_second',0), t.get('predicted_n',0), t.get('predicted_per_second',0)))
" 2>&1 | head -2
  grep -E "^.{0,16}hot experts: [0-9]+ tokens" "$log" | tail -2 | cut -c1-200
}
stop() { taskkill //F //IM llama-server.exe >/dev/null 2>&1; sleep 3; }

echo "== $label phase A: shipped placement + fresh stats (ctx=$CTX ub=$UB)"
start A LLAMA_HOT_EXPERTS=customware/hot-experts-mixed-4000.txt
req code1  customware/smoke-request-code.json
req code2  customware/smoke-request-code.json
req prose1 customware/smoke-request-prose.json
req prose2 customware/smoke-request-prose.json
echo "vram: $(nvidia-smi --query-gpu=memory.used --format=csv,noheader -i 1)"
grep -E "hot experts: [0-9]+ tokens|hot_refill|prefill mode" "$log" | cut -c1-200 | tail -20
stop
ls -la "$STATS" 2>&1; head -c 300 "$STATS" 2>/dev/null | head -3

echo "== $label phase B: placement from the stats"
start B LLAMA_HOT_EXPERTS_MIB=4000
req code3 customware/smoke-request-code.json
grep -E "hot experts: [0-9]+ tokens" "$log" | cut -c1-200 | tail -6
stop
echo "== $label done"

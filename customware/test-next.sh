#!/usr/bin/env bash
# Three experiments on the launcher config, one after the other, on a quiet box:
#   base : 160k, parked 4 GB set placed from a copy of the launcher's learned stats, swaps on   (reference)
#   spec : the same + n-gram speculative decoding (--spec-type ngram-mod)
#   big  : 120k context, -ub 4096 -b 4096, a 6 GB set from the same stats
# Each phase: 22k code prompt twice, 12k prose prompt once. Timings, draft acceptance, VRAM, hot-expert log lines.
#   usage: customware/test-next.sh <label> [PORT=5011] [PHASES="base spec big"]
set -uo pipefail
cd "$(dirname "$0")/.."
label=${1:?label}
PORT=${PORT:-5011}
PHASES=${PHASES:-"base spec big"}
STATS_SRC=/c/dev/llama.cpp-customware/hot-experts.stats
[ -s "$STATS_SRC" ] || STATS_SRC=customware/hot-experts-seed.stats

start() { # <phase> <ctx> <ub> <mib> <extra server args...>
  local phase=$1 ctx=$2 ub=$3 mib=$4; shift 4
  log=customware/serve-$label-$phase.log
  stats=customware/hot-experts-$label-$phase.stats
  cp -f "$STATS_SRC" "$stats"
  : > "$log"
  (OT='.*_exps=CPU' LLAMA_HOT_EXPERTS_ALIAS=1 GGML_OP_OFFLOAD_MIN_BATCH=512 LLAMA_HOT_EXPERTS_STATS="$stats" LLAMA_HOT_EXPERTS_MIB="$mib" LLAMA_HOT_EXPERTS_LOG=256 LLAMA_HOT_EXPERTS_TRACE="customware/trace-$label-$phase.bin" CTX="$ctx" UB="$ub" bash customware/serve.sh "$PORT" -lv 4 "$@" > "$log" 2>&1 &)
  for i in $(seq 1 180); do
    sleep 5
    if curl -s -m 2 "http://127.0.0.1:$PORT/health" 2>/dev/null | grep -q '"ok"'; then echo "health ok after $((i*5)) s"; break; fi
    if grep -q "CUDA error\|error loading model\|failed to load\|GGML_ASSERT\|error: " "$log"; then echo "server failed"; grep -E "CUDA error|error loading|failed to load|GGML_ASSERT|error: " "$log" | head -3; bash customware/kill-port.sh "$PORT" > /dev/null 2>&1; return 1; fi
  done
  grep -E "placement from|hot experts: [0-9]+ experts|hot_alias_place|compute buffer size|spec|draft" "$log" | grep -v "^.*srv.*params" | cut -c1-200 | head -8
  echo "vram: $(nvidia-smi --query-gpu=memory.used --format=csv,noheader -i 1)"
}
req() { # <name> <request file>
  curl -s -m 1800 "http://127.0.0.1:$PORT/v1/chat/completions" -H "Content-Type: application/json" -d @"$2" > "customware/ab-$label-$1.json"
  python -c "
import json
d=json.load(open('customware/ab-$label-$1.json'))
t=d.get('timings',{})
extra=''
if t.get('draft_n'): extra=' draft_n=%d accepted=%d (%.0f%%)' % (t['draft_n'], t.get('draft_n_accepted',0), 100.0*t.get('draft_n_accepted',0)/max(1,t['draft_n']))
print('$label $1: prompt_n=%d (%.1f t/s) predicted_n=%d (%.2f t/s)%s' % (t.get('prompt_n',0), t.get('prompt_per_second',0), t.get('predicted_n',0), t.get('predicted_per_second',0), extra))
" 2>&1 | head -2
  grep -E "hot experts: [0-9]+ tokens" "$log" | grep -E "over the last [0-9]{3} " | tail -1 | cut -c1-160
}
stop() { bash customware/kill-port.sh "$PORT" > /dev/null 2>&1; sleep 3; grep -E "draft acceptance|mean_acc_len|draft_ratio|n_draft" "$log" | tail -2 | cut -c1-200; }

for phase in $PHASES; do
  case $phase in
    base) echo "== $label base: 160k ub2048 4 GB"; start base 160000 2048 4000 || continue ;;
    spec) echo "== $label spec: 160k ub2048 4 GB + ngram-mod"; start spec 160000 2048 4000 --spec-type ngram-mod || continue ;;
    big)  echo "== $label big: 120k ub4096 6 GB";  start big  120000 4096 6000 -b 4096 || continue ;;
    *) echo "unknown phase $phase"; continue ;;
  esac
  req code1  customware/smoke-request-code.json
  req code2  customware/smoke-request-code.json
  req prose1 customware/smoke-request-prose.json
  echo "vram: $(nvidia-smi --query-gpu=memory.used --format=csv,noheader -i 1)"
  stop
done
echo "== $label done"

#!/usr/bin/env bash
# Entry 007 admission fix (2026-09-29): a request never waits for cells nobody holds, and the generation reserve is
# capped (clients send max_tokens 32000 as a matter of course).
#   run 1, pool 24576, LLAMA_KV_RESERVE=100000: a 22k prompt with max_tokens 3000 needs more than the pool on paper,
#          nothing else is active -> must start at once (before: "waiting for cells" forever)
#   run 2, pool 40960, default reserve 2048: A generates from a 12k prompt, B (22k prompt, max_tokens 100000)
#          arrives -> B must start at once next to A: 12597 + 22225 + 2048 fit, 12597 + 22225 + 100000 would not
#          (the first run of this test used a 32k pool, where B genuinely did not fit next to A and waited: correct,
#          but it tested nothing)
#   usage: customware/test-admit.sh <label>
set -uo pipefail
cd "$(dirname "$0")/.."
label=${1:?label}
PORT=${PORT:-5011}
start() { # <ctx> <log> [env...]
  local ctx=$1 log=$2; shift 2
  : > "$log"
  (env "$@" OT='.*_exps=CPU' LLAMA_HOT_EXPERTS=customware/hot-experts-mixed-4000.txt LLAMA_HOT_EXPERTS_ALIAS=0 LLAMA_HOT_EXPERTS_SWAP=0 GGML_OP_OFFLOAD_MIN_BATCH=512 CTX="$ctx" UB=2048 CACHE_RAM=4096 bash customware/serve.sh "$PORT" -lv 4 > "$log" 2>&1 &)
  for i in $(seq 1 120); do sleep 5; curl -s -m 2 "http://127.0.0.1:$PORT/health" 2>/dev/null | grep -q '"ok"' && { echo "server up after $((i*5)) s"; break; }; done
}
stop() { bash customware/kill-port.sh "$PORT"; sleep 3; }
mk() { # <out> <base request> <n_predict> <tag>
  python -c "
import json;d=json.load(open('$2'))
d['max_tokens']=$3; d['temperature']=0; d['top_k']=1; d['seed']=7
d['chat_template_kwargs']={'enable_thinking': False}; d['reasoning_budget']=0
d['messages'][0]['content']='[$4] '+d['messages'][0]['content']
json.dump(d,open('$1','w'))"; }
post() { # <name> <request>
  local t0=$(date +%s)
  curl -s -m 7200 "http://127.0.0.1:$PORT/v1/chat/completions" -H "Content-Type: application/json" -d @"$2" > "customware/ab-$label-$1.json"
  python -c "
import json
d=json.load(open('customware/ab-$label-$1.json'))
if 'error' in d: print('$1: ERROR %s: %s' % (d['error'].get('type'), d['error'].get('message','')[:100])); raise SystemExit
t=d.get('timings',{})
print('$1: prompt_n=%d (%.1f t/s) predicted_n=%d (%.2f t/s) finish=%s, %d s wall' % (t.get('prompt_n',0),t.get('prompt_per_second',0),t.get('predicted_n',0),t.get('predicted_per_second',0),d['choices'][0].get('finish_reason'),$(date +%s)-$t0))"
}
admits() { grep -c "waiting for cells" "$1"; }

echo "== run 1: alone in a 24576 pool, reserve uncapped, need > pool"
LOG=customware/serve-$label-1.log
start 24576 "$LOG" LLAMA_KV_RESERVE=100000
mk customware/admit-A1.json customware/depth-request-22k.json 3000 r1A
post A1 customware/admit-A1.json
echo "deferrals in run 1: $(admits "$LOG") (expected 0)"
stop

echo "== run 2: 32768 pool, default reserve; A generates, B (22k, max_tokens 100000) must start next to it"
LOG=customware/serve-$label-2.log
start 32768 "$LOG"
mk customware/admit-A2.json customware/smoke-request-prose.json 400 r2A
mk customware/admit-B2.json customware/depth-request-22k.json 100000 r2B
post A2 customware/admit-A2.json & sleep 12
post B2 customware/admit-B2.json
wait
echo "deferrals in run 2: $(admits "$LOG") (expected 0)"
grep -E "prompt processing, n_tokens|waiting for cells" "$LOG" | sed -E 's/^[0-9.]+ [IWE] //' | cut -c1-140 | head -8
stop
echo "admit test $label done"

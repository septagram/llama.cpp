#!/usr/bin/env bash
# Entry 010, on the real model: under --kv-unified a cached prompt must be restored whenever it exists, even when
# another idle slot still holds its cells. A 24576-cell pool, the 22k depth prompt (A) and the 12k prose prompt (B)
# cannot share it; requested A, B, A, B, the second A and B must come back from the prompt cache (a restore of a
# few seconds), never from a re-prefill. Classic hot experts (the 24k compute buffer is too small to park the set),
# prompt cache 8 GB, port 5011.
#   usage: [BIN=...] customware/test-cache-hit-real.sh <label>
set -uo pipefail
cd "$(dirname "$0")/.."
label=${1:?label}
PORT=${PORT:-5011}
LOG=customware/serve-$label.log
: > "$LOG"
(env OT='.*_exps=CPU' LLAMA_HOT_EXPERTS=customware/hot-experts-mixed-4000.txt LLAMA_HOT_EXPERTS_ALIAS=0 LLAMA_HOT_EXPERTS_SWAP=0 GGML_OP_OFFLOAD_MIN_BATCH=512 CTX=24576 UB=2048 CACHE_RAM=8192 bash customware/serve.sh "$PORT" -lv 4 > "$LOG" 2>&1 &)
for i in $(seq 1 120); do sleep 5; curl -s -m 2 "http://127.0.0.1:$PORT/health" 2>/dev/null | grep -q '"ok"' && { echo "server up after $((i*5)) s"; break; }; done
mk() { # <out> <base> <tag>
  python -c "
import json;d=json.load(open('$2'))
d['max_tokens']=16; d['temperature']=0; d['top_k']=1; d['seed']=7; d['chat_template_kwargs']={'enable_thinking': False}; d['reasoning_budget']=0
d['messages'][0]['content']='[$3] '+d['messages'][0]['content']
json.dump(d,open('$1','w'))"; }
mk customware/hitr-A.json customware/depth-request-22k.json hitA
mk customware/hitr-B.json customware/smoke-request-prose.json hitB
post() { # <name> <request>
  local t0=$(date +%s)
  curl -s -m 1800 "http://127.0.0.1:$PORT/v1/chat/completions" -H "Content-Type: application/json" -d @"$2" > "customware/ab-$label-$1.json"
  python -c "
import json
d=json.load(open('customware/ab-$label-$1.json'))
if 'error' in d: print('  $1: ERROR', d['error']); raise SystemExit
t=d.get('timings',{}); print('  $1: prompt_n=%d cache_n=%d predicted_n=%d, %d s wall' % (t.get('prompt_n',0), t.get('cache_n',0), t.get('predicted_n',0), $(date +%s)-$t0))"
}
echo "== A, B, A, B"
post A1 customware/hitr-A.json; post B1 customware/hitr-B.json; post A2 customware/hitr-A.json; post B2 customware/hitr-B.json
echo "== verdict"
echo "restore failures: $(grep -c 'failed to find\|failed to load prompt from cache' "$LOG")"
grep -E "cached n_tokens =|freeing idle slot|kept for later|saving idle slot|purging slot|waiting for cells" "$LOG" | sed -E 's/^[0-9.]+ [IWE] //' | cut -c1-140
bash customware/kill-port.sh "$PORT" > /dev/null; sleep 2
echo "cache hit test $label done"

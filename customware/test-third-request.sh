#!/usr/bin/env bash
# Two large conversations that cannot share the pool take turns; a small third request arrives while one generates
# and the other waits. Does the small one run at once next to the active one, or wait behind the queued one?
# Small scale, real model, port 5011: pool 24576; A and B are the 12.3k prose prompt under two tags (together
# they do not fit); C is the 42-token request. A generates 400 tokens; B is posted once A is generating (it must
# queue); C is posted two seconds later.
#   usage: customware/test-third-request.sh <label>
set -uo pipefail
cd "$(dirname "$0")/.."
label=${1:?label}
PORT=${PORT:-5011}
LOG=customware/serve-$label.log
: > "$LOG"
(env OT='.*_exps=CPU' LLAMA_HOT_EXPERTS=customware/hot-experts-mixed-4000.txt LLAMA_HOT_EXPERTS_ALIAS=0 LLAMA_HOT_EXPERTS_SWAP=0 GGML_OP_OFFLOAD_MIN_BATCH=512 CTX=24576 UB=2048 CACHE_RAM=8192 bash customware/serve.sh "$PORT" -lv 4 > "$LOG" 2>&1 &)
for i in $(seq 1 120); do sleep 5; curl -s -m 2 "http://127.0.0.1:$PORT/health" 2>/dev/null | grep -q '"ok"' && { echo "server up after $((i*5)) s"; break; }; done
mk() { # <out> <base> <n_predict> <tag> [ignore_eos]
  python -c "
import json;d=json.load(open('$2'))
d['max_tokens']=$3; d['temperature']=0; d['top_k']=1; d['seed']=7; d['chat_template_kwargs']={'enable_thinking': False}; d['reasoning_budget']=0
if '${5:-}'=='ignore_eos': d['ignore_eos']=True
d['messages'][0]['content']='[$4] '+d['messages'][0]['content']
json.dump(d,open('$1','w'))"; }
mk customware/third-A.json customware/smoke-request-prose.json 400 bigA ignore_eos
mk customware/third-B.json customware/smoke-request-prose.json 64  bigB ignore_eos
mk customware/third-C.json customware/equiv-request-lp.json    64  smallC ignore_eos
T0=$(date +%s)
post() { # <name> <request>
  local t0=$(date +%s)
  curl -s -m 2400 "http://127.0.0.1:$PORT/v1/chat/completions" -H "Content-Type: application/json" -d @"$2" > "customware/ab-$label-$1.json"
  python -c "
import json
d=json.load(open('customware/ab-$label-$1.json'))
if 'error' in d: print('  $1: ERROR', d['error']); raise SystemExit
t=d.get('timings',{}); print('  $1: posted at +%d s, answered at +%d s (prompt_n=%d cache_n=%d predicted_n=%d at %.1f t/s)' % ($t0-$T0, $(date +%s)-$T0, t.get('prompt_n',0), t.get('cache_n',0), t.get('predicted_n',0), t.get('predicted_per_second',0)))"
}
post A customware/third-A.json &
for i in $(seq 1 120); do sleep 2; grep -q "n_gen = " "$LOG" && break; done
echo "A is generating at +$(( $(date +%s)-T0 )) s; posting B (must queue), then C"
post B customware/third-B.json & sleep 2
post C customware/third-C.json
wait
echo "== order of events"
grep -E "launch_slot_.*processing task|waiting for cells|kept for later|release: .* stop processing|freeing idle slot|suspended|purging" "$LOG" | sed -E 's/^([0-9]+\.[0-9]+)\.[0-9]+\.[0-9]+ [IWE] /\1 /' | cut -c1-140
bash customware/kill-port.sh "$PORT" > /dev/null; sleep 2
echo "third request test $label done"

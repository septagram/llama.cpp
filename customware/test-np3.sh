#!/usr/bin/env bash
# How much does a third generation at once cost? Three 12k conversations (the prose prompt under three tags) are
# prefilled one by one, then 300-token continuations are posted together: first two at once, then all three at
# once (LLAMA_SCHED_MAX_ACTIVE=3 for this run). Reports each request's t/s and the sum. Port 5011, 3 slots, pool 49152.
#   usage: customware/test-np3.sh <label>
set -uo pipefail
cd "$(dirname "$0")/.."
label=${1:?label}
PORT=${PORT:-5011}
LOG=customware/serve-$label.log
: > "$LOG"
(env LLAMA_SCHED_MAX_ACTIVE=3 OT='.*_exps=CPU' LLAMA_HOT_EXPERTS=customware/hot-experts-mixed-4000.txt LLAMA_HOT_EXPERTS_ALIAS=0 LLAMA_HOT_EXPERTS_SWAP=0 GGML_OP_OFFLOAD_MIN_BATCH=512 CTX=49152 NP=3 UB=2048 CACHE_RAM=8192 bash customware/serve.sh "$PORT" -lv 4 > "$LOG" 2>&1 &)
for i in $(seq 1 120); do sleep 5; curl -s -m 2 "http://127.0.0.1:$PORT/health" 2>/dev/null | grep -q '"ok"' && { echo "server up after $((i*5)) s"; break; }; done
mk() { # <out> <base> <n_predict> <tag> [ignore_eos]
  python -c "
import json;d=json.load(open('$2'))
d['max_tokens']=$3; d['temperature']=0; d['top_k']=1; d['seed']=7; d['chat_template_kwargs']={'enable_thinking': False}; d['reasoning_budget']=0
if '${5:-}'=='ignore_eos': d['ignore_eos']=True
d['messages'][0]['content']='[$4] '+d['messages'][0]['content']
json.dump(d,open('$1','w'))"; }
post() { # <name> <request>
  curl -s -m 3600 "http://127.0.0.1:$PORT/v1/chat/completions" -H "Content-Type: application/json" -d @"$2" > "customware/ab-$label-$1.json"
  python -c "
import json
d=json.load(open('customware/ab-$label-$1.json'))
if 'error' in d: print('  $1: ERROR', d['error']); raise SystemExit
t=d.get('timings',{}); print('  $1: cache_n=%d predicted_n=%d at %.2f t/s' % (t.get('cache_n',0), t.get('predicted_n',0), t.get('predicted_per_second',0)))"
}
total() { python -c "
import json,sys; s=0
for n in sys.argv[1:]:
    t=json.load(open('customware/ab-$label-%s.json'%n)).get('timings',{}); s+=t.get('predicted_per_second',0)
print('  together: %.1f t/s' % s)" "$@"; }
for t in 1 2 3; do mk customware/np3-P$t.json customware/smoke-request-prose.json 8 np3-$t; post P$t customware/np3-P$t.json; done
for t in 1 2 3; do mk customware/np3-G$t.json customware/smoke-request-prose.json 300 np3-$t ignore_eos; done
echo "== one alone"
post G1a customware/np3-G1.json
echo "== two at once"
post G1b customware/np3-G1.json & post G2b customware/np3-G2.json & wait
total G1b G2b
echo "== three at once"
post G1c customware/np3-G1.json & post G2c customware/np3-G2.json & post G3c customware/np3-G3.json & wait
total G1c G2c G3c
grep -c "waiting for cells" "$LOG" | sed 's/^/  waiting for cells: /'
bash customware/kill-port.sh "$PORT" > /dev/null; sleep 2
echo "np3 test $label done"

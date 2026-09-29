#!/usr/bin/env bash
# Entry 007 scenarios on a small unified pool: -c 32768 -np 2 --kv-unified, classic hot experts (the compute buffer
# at 32k is too small to park the set), prompt cache 8 GB, port 5011.
#   1. busy pool + new request: A generates from a 22k prompt (n_predict 3000); B (12k prompt) arrives -> B waits, no error
#   2. two active outgrow the pool: A (12k, n_predict 6000) and B (12k, n_predict 6000) -> one is suspended, both finish;
#      B's greedy text equals its uninterrupted run
#   3. idle retention: A done and retained; a 22k request runs (idle slot purged), no wait, no error
#   4. genuinely too large: a 44k prompt -> EXCEED_CONTEXT_SIZE error, nothing else affected
#   usage: customware/test-kvu.sh <label>
set -uo pipefail
cd "$(dirname "$0")/.."
label=${1:?label}
PORT=${PORT:-5011}
LOG=customware/serve-$label.log
: > "$LOG"
(OT='.*_exps=CPU' LLAMA_HOT_EXPERTS=customware/hot-experts-mixed-4000.txt LLAMA_HOT_EXPERTS_ALIAS=0 LLAMA_HOT_EXPERTS_SWAP=0 GGML_OP_OFFLOAD_MIN_BATCH=512 CTX=32768 UB=2048 CACHE_RAM=8192 bash customware/serve.sh "$PORT" -lv 4 > "$LOG" 2>&1 &)
for i in $(seq 1 120); do sleep 5; curl -s -m 2 "http://127.0.0.1:$PORT/health" 2>/dev/null | grep -q '"ok"' && { echo "server up after $((i*5)) s"; break; }; done
grep -E "unified KV pool" "$LOG" | cut -c1-200

mk() { # <out> <base request> <n_predict> <temperature> <tag>
  python -c "
import json;d=json.load(open('$2'))
d['max_tokens']=$3; d['temperature']=$4; d['top_k']=1 if $4==0 else 40; d['seed']=7
d['chat_template_kwargs']={'enable_thinking': False}; d['reasoning_budget']=0
d['messages'][0]['content']='[$5] '+d['messages'][0]['content']
json.dump(d,open('$1','w'))"; }
post() { # <name> <request> ; prints timings or the error
  curl -s -m 7200 "http://127.0.0.1:$PORT/v1/chat/completions" -H "Content-Type: application/json" -d @"$2" > "customware/ab-$label-$1.json"
  python -c "
import json
d=json.load(open('customware/ab-$label-$1.json'))
if 'error' in d: print('$1: ERROR %s: %s' % (d['error'].get('type'), d['error'].get('message','')[:100])); raise SystemExit
t=d.get('timings',{}); m=d['choices'][0]['message']
print('$1: prompt_n=%d (%.1f t/s) predicted_n=%d (%.2f t/s) finish=%s content=%d chars' % (t.get('prompt_n',0),t.get('prompt_per_second',0),t.get('predicted_n',0),t.get('predicted_per_second',0),d['choices'][0].get('finish_reason'),len(m.get('content') or '')))"
}
kvlog() { grep -E "waiting for cells|KV pool overflow|suspended:|resumed after|Context size has been exceeded|purging slot" "$LOG" | sed -E 's/^[0-9.]+ [IWE] //' | cut -c1-160 | tail -${1:-6}; }

echo "== scenario 1: busy pool, new request waits"
mk customware/kvu-A1.json customware/depth-request-22k.json 3000 0.7 s1A
mk customware/kvu-B1.json customware/smoke-request-prose.json 300 0.7 s1B
post A1 customware/kvu-A1.json & sleep 45
post B1 customware/kvu-B1.json
wait
kvlog

echo "== scenario 2: two active sequences outgrow the pool"
mk customware/kvu-A2.json customware/smoke-request-prose.json 6000 0.7 s2A
mk customware/kvu-B2.json customware/agentic-request-30k.json 6000 0 s2B
python -c "
import json;d=json.load(open('customware/kvu-B2.json'))
d['messages']=d['messages'][:30]; json.dump(d,open('customware/kvu-B2.json','w'))"
post A2 customware/kvu-A2.json & sleep 20
post B2 customware/kvu-B2.json
wait
kvlog 8
echo "-- B2 uninterrupted reference (same greedy request, alone)"
post B2ref customware/kvu-B2.json
python -c "
import json
a=json.load(open('customware/ab-$label-B2.json'))['choices'][0]['message'].get('content') or ''
b=json.load(open('customware/ab-$label-B2ref.json'))['choices'][0]['message'].get('content') or ''
n=0
for x,y in zip(a,b):
    if x!=y: break
    n+=1
print('B2 vs reference: identical prefix %d chars of %d / %d' % (n, len(a), len(b)))"

echo "== scenario 3: idle retention"
mk customware/kvu-A3.json customware/depth-request-22k.json 50 0 s3A
post A3 customware/kvu-A3.json
kvlog 3

echo "== scenario 4: genuinely too large (44k prompt on a 32k pool)"
mk customware/kvu-A4.json customware/depth-request-44k.json 20 0 s4A
post A4 customware/kvu-A4.json
kvlog 3

curl -s "http://127.0.0.1:$PORT/slots" | python -c "import json,sys; s=json.load(sys.stdin); print('slots:', [(x['id'], x['is_processing'], x.get('suspended')) for x in s])"
taskkill //F //IM llama-server.exe >/dev/null 2>&1
echo "== $label done"

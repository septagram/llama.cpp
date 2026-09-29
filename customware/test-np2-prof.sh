#!/usr/bin/env bash
# Entry 009 / H12: what a second generator costs. Two generators on one server (160k launcher shape: ub 2048, the
# 4 GB alias set fits the compute buffer), the per-node profile of the two-token graphs, compared with the one-token
# graphs of the same run. GGML_SCHED_PROFILE=N dumps every N-th graph node by node.
#   usage: [PROF=16] customware/test-np2-prof.sh <label>
set -uo pipefail
cd "$(dirname "$0")/.."
label=${1:?label}
PORT=${PORT:-5011}
LOG=customware/serve-$label.log
: > "$LOG"
(env GGML_SCHED_PROFILE="${PROF:-16}" GGML_SCHED_PROFILE_NODES=1 GGML_CUDA_DISABLE_GRAPHS=1 \
 OT='.*_exps=CPU' LLAMA_HOT_EXPERTS=customware/hot-experts-mixed-4000.txt LLAMA_HOT_EXPERTS_ALIAS=1 LLAMA_HOT_EXPERTS_SWAP=0 \
 GGML_OP_OFFLOAD_MIN_BATCH=512 CTX="${CTX:-160000}" UB="${UB:-2048}" bash customware/serve.sh "$PORT" -lv 4 > "$LOG" 2>&1 &)
for i in $(seq 1 120); do sleep 5; curl -s -m 2 "http://127.0.0.1:$PORT/health" 2>/dev/null | grep -q '"ok"' && { echo "server up after $((i*5)) s"; break; }; done
grep -E "hot_alias_place|hot experts disabled" "$LOG" | cut -c1-160

mk() { # <out> <base request> <n_predict> <tag>
  python -c "
import json;d=json.load(open('$2'))
d['max_tokens']=$3; d['temperature']=0; d['top_k']=1; d['seed']=7; d['ignore_eos']=True
d['chat_template_kwargs']={'enable_thinking': False}; d['reasoning_budget']=0
d['messages'][0]['content']='[$4] '+d['messages'][0]['content']
json.dump(d,open('$1','w'))"; }
post() { # <name> <request>
  curl -s -m 7200 "http://127.0.0.1:$PORT/v1/chat/completions" -H "Content-Type: application/json" -d @"$2" > "customware/ab-$label-$1.json"
  python -c "
import json
d=json.load(open('customware/ab-$label-$1.json'))
if 'error' in d: print('$1: ERROR', d['error']); raise SystemExit
t=d.get('timings',{})
print('$1: prompt_n=%d (%.1f t/s) predicted_n=%d %.1f ms/token' % (t.get('prompt_n',0),t.get('prompt_per_second',0),t.get('predicted_n',0),t.get('predicted_per_token_ms',0)))"
}

# A alone first (one-token graphs), then A and B together (two-token graphs once B's prompt is in)
mk customware/np2-A.json customware/depth-request-22k.json 120 npA
mk customware/np2-B.json customware/smoke-request-prose.json 400 npB
echo "== A alone"
post A1 customware/np2-A.json
start=$(wc -l < "$LOG")
echo "== A and B together"
post B customware/np2-B.json & sleep 100
post A2 customware/np2-A.json
wait
end=$(wc -l < "$LOG")
bash customware/kill-port.sh "$PORT" > /dev/null 2>&1
sleep 2
echo "== one-token graphs of the whole run"
NTOK=1 python customware/prof-summary.py "$LOG" "np1:0-$end" | head -12
echo "== two-token graphs while both generate"
NTOK=2 python customware/prof-summary.py "$LOG" "np2:$start-$end" | head -16
echo "np2 profile $label done"

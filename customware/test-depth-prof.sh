#!/usr/bin/env bash
# Entry 008: rank the decode graph's ops at 22k / 44k / 88k depth. GGML_SCHED_PROFILE=N prints every graph's split
# summary and dumps every N-th graph; GGML_SCHED_PROFILE_NODES=1 computes the dumped graphs node by node with a sync
# after each node (a ranking, not a measurement) and times set_inputs; CUDA graphs are off so the one-node graphs are
# not captured. Launcher-shaped config (alias hot experts, 120k pool) on port 5011; the summary is per depth, decode
# graphs only (n_tokens = 1), via customware/prof-summary.py.
#   usage: [PROF=16] [NPRED=64] customware/test-depth-prof.sh <label>
set -uo pipefail
cd "$(dirname "$0")/.."
label=${1:?label}
PORT=${PORT:-5011}
LOG=customware/serve-$label.log
: > "$LOG"
PROF=${PROF:-16}
# PROF=0: no profiling at all, CUDA graphs on: the plain depth series for a before/after comparison
if [ "$PROF" -gt 0 ]; then prof_env=(GGML_SCHED_PROFILE="$PROF" GGML_SCHED_PROFILE_NODES=1 GGML_CUDA_DISABLE_GRAPHS=1); else prof_env=(); fi
(env "${prof_env[@]}" \
 OT='.*_exps=CPU' LLAMA_HOT_EXPERTS=customware/hot-experts-mixed-4000.txt LLAMA_HOT_EXPERTS_ALIAS=1 LLAMA_HOT_EXPERTS_SWAP=0 \
 GGML_OP_OFFLOAD_MIN_BATCH=512 CTX="${CTX:-120000}" UB="${UB:-2048}" bash customware/serve.sh "$PORT" -lv 4 > "$LOG" 2>&1 &)
for i in $(seq 1 120); do sleep 5; curl -s -m 2 "http://127.0.0.1:$PORT/health" 2>/dev/null | grep -q '"ok"' && { echo "server up after $((i*5)) s"; break; }; done

ranges=()
start=$(wc -l < "$LOG")
for d in ${DEPTHS:-22k 44k 88k}; do
  python -c "
import json;d=json.load(open('customware/depth-request-$d.json'))
d['max_tokens']=${NPRED:-64}; d['temperature']=0; d['top_k']=1; d['seed']=7
d['chat_template_kwargs']={'enable_thinking': False}; d['reasoning_budget']=0
json.dump(d,open('customware/prof-$d.json','w'))"
  curl -s -m 7200 "http://127.0.0.1:$PORT/v1/chat/completions" -H "Content-Type: application/json" -d @customware/prof-$d.json > "customware/ab-$label-$d.json"
  python -c "
import json
d=json.load(open('customware/ab-$label-$d.json'))
if 'error' in d: print('$d: ERROR', d['error']); raise SystemExit
t=d.get('timings',{})
print('$d: prompt_n=%d (%.1f t/s) predicted_n=%d %.1f ms/token' % (t.get('prompt_n',0),t.get('prompt_per_second',0),t.get('predicted_n',0),t.get('predicted_per_token_ms',0)))"
  end=$(wc -l < "$LOG")
  ranges+=("$d:$start-$end")
  start=$end
done
bash customware/kill-port.sh "$PORT" > /dev/null 2>&1
sleep 2
python customware/prof-summary.py "$LOG" "${ranges[@]}"
echo "depth profile $label done"

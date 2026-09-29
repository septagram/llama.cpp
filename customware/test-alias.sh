#!/usr/bin/env bash
# Server A/B for the aliased hot experts (entry 005): start the server with the given context/ubatch, run the 22k code
# prompt twice (cold prefill + cached), then a short follow-up on the cached prefix (must stay in decode mode), and
# report timings, VRAM and the hot-expert log lines.
#   usage: customware/test-alias.sh <label> [CTX=160000] [UB=2048] [PLACEMENT=customware/hot-experts-mixed-4000.txt] [ALIAS=1] [PORT=5011]
set -uo pipefail
cd "$(dirname "$0")/.."
label=${1:?label}
CTX=${CTX:-160000}; UB=${UB:-2048}; PLACEMENT=${PLACEMENT:-customware/hot-experts-mixed-4000.txt}; ALIAS=${ALIAS:-1}; PORT=${PORT:-5011}
log=customware/serve-$label.log
: > "$log"
echo "== $label: ctx=$CTX ub=$UB placement=$PLACEMENT alias=$ALIAS"
(OT='.*_exps=CPU' LLAMA_HOT_EXPERTS="$PLACEMENT" LLAMA_HOT_EXPERTS_ALIAS="$ALIAS" CTX="$CTX" UB="$UB" bash customware/serve.sh "$PORT" -lv 4 > "$log" 2>&1 &)
for i in $(seq 1 180); do
  sleep 5
  if curl -s -m 2 "http://127.0.0.1:$PORT/health" 2>/dev/null | grep -q '"ok"'; then echo "health ok after $((i*5)) s"; break; fi
  if grep -q "CUDA error\|error loading model\|failed to load\|GGML_ASSERT" "$log"; then echo "server failed"; grep -E "CUDA error|error loading|failed to load|GGML_ASSERT" "$log" | head -3; bash customware/kill-port.sh "$PORT" > /dev/null 2>&1; exit 1; fi
done
grep -E "hot experts|compute buffer size|KV buffer size" "$log" | cut -c1-220
echo "vram: $(nvidia-smi --query-gpu=memory.used --format=csv,noheader -i 1)"
req() { # <name> <request file>
  curl -s -m 1800 "http://127.0.0.1:$PORT/v1/chat/completions" -H "Content-Type: application/json" -d @"$2" > "customware/ab-$label-$1.json"
  python -c "
import json,sys
d=json.load(open('customware/ab-$label-$1.json'))
t=d.get('timings',{})
print('$label $1: prompt_n=%d (%.1f t/s) predicted_n=%d (%.2f t/s)' % (t.get('prompt_n',0), t.get('prompt_per_second',0), t.get('predicted_n',0), t.get('predicted_per_second',0)))
" 2>&1 | head -2
}
req run1 customware/smoke-request-code.json
req run2 customware/smoke-request-code.json
python - "$label" <<'EOF'
import json,sys
label=sys.argv[1]
d=json.load(open('customware/smoke-request-code.json'))
r=json.load(open('customware/ab-%s-run2.json' % label))
m=r['choices'][0]['message']
reply=(m.get('content') or '') or ('(thinking) ' + (m.get('reasoning_content') or '')[:2000])
d['messages']=d['messages']+[{'role':'assistant','content':reply},{'role':'user','content':'Thanks. In two sentences: what is the single riskiest part of this code, and why?'}]
d['max_tokens']=120
json.dump(d,open('customware/smoke-request-followup.json','w'))
EOF
req run3 customware/smoke-request-followup.json
echo "vram: $(nvidia-smi --query-gpu=memory.used --format=csv,noheader -i 1)"
grep -E "hot experts|prefill mode|restored|re-parking|reallocated" "$log" | cut -c1-200 | tail -12
bash customware/kill-port.sh "$PORT" > /dev/null 2>&1
echo "== $label done"

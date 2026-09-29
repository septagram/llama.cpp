#!/usr/bin/env bash
# Entry 009: the fused get_rows + avg pool (indexer block mean) on vs off on one binary, every other CUDA fusion on
# in both runs (GGML_CUDA_DISABLE_FUSION=1 would switch those off too and change the numerics everywhere).
# Launcher shape (ub 4096, alias set), port 5011: the 48-token log-prob request and a 12k greedy prompt; both
# must match bit for bit, and the log must show the fusion announcing itself in the "on" run.
#   usage: customware/test-fusion-ab.sh <label>
set -uo pipefail
cd "$(dirname "$0")/.."
label=${1:?label}
PORT=${PORT:-5011}
python -c "
import json;d=json.load(open('customware/smoke-request-prose.json'))
d['max_tokens']=64; d['temperature']=0; d['top_k']=1; d['seed']=7; d['chat_template_kwargs']={'enable_thinking': False}; d['reasoning_budget']=0
json.dump(d,open('customware/fuse-prose.json','w'))"
for mode in on off; do
  LOG=customware/serve-$label-$mode.log; : > "$LOG"
  extra=(); [ "$mode" = off ] && extra=(GGML_CUDA_NO_GET_ROWS_POOL=1)
  (env "${extra[@]}" OT='.*_exps=CPU' LLAMA_HOT_EXPERTS=customware/hot-experts-mixed-4000.txt LLAMA_HOT_EXPERTS_ALIAS=1 LLAMA_HOT_EXPERTS_SWAP=0 GGML_OP_OFFLOAD_MIN_BATCH=512 CTX="${CTX:-120000}" UB="${UB:-4096}" bash customware/serve.sh "$PORT" -b "${UB:-4096}" -lv 4 > "$LOG" 2>&1 &)
  for i in $(seq 1 60); do sleep 5; curl -s -m 2 "http://127.0.0.1:$PORT/health" 2>/dev/null | grep -q '"ok"' && { echo "[$mode] server up after $((i*5)) s"; break; }; done
  curl -s -m 600 "http://127.0.0.1:$PORT/v1/chat/completions" -H "Content-Type: application/json" -d @customware/equiv-request-lp.json > "customware/ab-$label-$mode-equiv.json"
  curl -s -m 2400 "http://127.0.0.1:$PORT/v1/chat/completions" -H "Content-Type: application/json" -d @customware/fuse-prose.json > "customware/ab-$label-$mode-prose.json"
  python -c "
import json
d=json.load(open('customware/ab-$label-$mode-prose.json')); t=d.get('timings',{})
print('[$mode] prose 12k: prompt_n=%d (%.1f t/s) predicted_n=%d %.1f ms/token' % (t.get('prompt_n',0),t.get('prompt_per_second',0),t.get('predicted_n',0),t.get('predicted_per_token_ms',0)))"
  echo "[$mode] fusion announced: $(grep -c 'fused get_rows + avg pool active' "$LOG")"
  bash customware/kill-port.sh "$PORT" > /dev/null; sleep 3
done
echo "== 48-token log-probs, off vs on"; python customware/equiv-compare.py "customware/ab-$label-off-equiv.json" "customware/ab-$label-on-equiv.json" 2>&1 | head -3
python - <<EOF
import json
a=json.load(open('customware/ab-$label-off-prose.json'))['choices'][0]['message']; b=json.load(open('customware/ab-$label-on-prose.json'))['choices'][0]['message']
ta=(a.get('content') or '')+(a.get('reasoning_content') or ''); tb=(b.get('content') or '')+(b.get('reasoning_content') or '')
n=0
while n<min(len(ta),len(tb)) and ta[n]==tb[n]: n+=1
print("prose greedy off vs on:", "IDENTICAL" if ta==tb else f"DIFFER after {n} chars", len(ta), len(tb))
EOF
echo "fusion A/B $label done"

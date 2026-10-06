#!/usr/bin/env bash
# Entry 010: the prompt cache yields to the model's page cache. Two server runs on port 5011 (--cache-ram 16384):
#   run 1: LLAMA_CACHE_RAM_RESERVE_MIB=200000 (more than the box has): every save must be skipped with the log line
#   run 2: reserve = what is available at start minus 2000 MiB: the small prompt's save fits, the 12k prompt's save
#          must evict it or be skipped; the log shows the decisions with numbers
# Requests: A (48 tokens), B (12k prose), C (48 tokens, other tag). A save happens when a slot with an idle prompt is
# reused or when the next task starts, so the log is read after C.
#   usage: customware/test-cache-reserve.sh <label>
set -uo pipefail
cd "$(dirname "$0")/.."
label=${1:?label}
PORT=${PORT:-5011}
avail_mib=$(powershell -NoProfile -Command "[int]((Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory/1KB)" | tr -d '\r')
echo "available at start: $avail_mib MiB"
mk() { # <out> <base> <n_predict> <tag>
  python -c "
import json;d=json.load(open('$2'))
d['max_tokens']=$3; d['temperature']=0; d['top_k']=1; d['seed']=7; d['chat_template_kwargs']={'enable_thinking': False}; d['reasoning_budget']=0
d['messages'][0]['content']='[$4] '+d['messages'][0]['content']
json.dump(d,open('$1','w'))"; }
mk customware/cr-A.json customware/equiv-request-lp.json 16 crA
mk customware/cr-B.json customware/smoke-request-prose.json 16 crB
mk customware/cr-C.json customware/equiv-request-lp.json 16 crC
post() { curl -s -m 2400 "http://127.0.0.1:$PORT/v1/chat/completions" -H "Content-Type: application/json" -d @"$1" > "customware/ab-$label-$2.json"; python -c "
import json; d=json.load(open('customware/ab-$label-$2.json')); t=d.get('timings',{}); print('  $2: prompt_n=%d predicted_n=%d' % (t.get('prompt_n',0),t.get('predicted_n',0)))"; }
for run in 1 2; do
  if [ $run = 1 ]; then reserve=200000; else reserve=$((avail_mib - 2000)); fi
  LOG=customware/serve-$label-$run.log; : > "$LOG"
  echo "== run $run: LLAMA_CACHE_RAM_RESERVE_MIB=$reserve"
  (env LLAMA_CACHE_RAM_RESERVE_MIB=$reserve OT='.*_exps=CPU' LLAMA_HOT_EXPERTS=customware/hot-experts-mixed-4000.txt LLAMA_HOT_EXPERTS_ALIAS=1 LLAMA_HOT_EXPERTS_SWAP=0 GGML_OP_OFFLOAD_MIN_BATCH=512 CTX=120000 UB=2048 CACHE_RAM=16384 bash customware/serve.sh "$PORT" -lv 4 > "$LOG" 2>&1 &)
  for i in $(seq 1 60); do sleep 5; curl -s -m 2 "http://127.0.0.1:$PORT/health" 2>/dev/null | grep -q '"ok"' && { echo "  server up after $((i*5)) s"; break; }; done
  post customware/cr-A.json A; post customware/cr-B.json B; post customware/cr-C.json C
  sleep 2
  echo "  -- guard lines"; grep -E "prompt cache keeps|reserved for the model|not saving this prompt|saving prompt with length|cache state:" "$LOG" | sed -E 's/^[0-9.]+ [IWE] //' | cut -c1-170
  bash customware/kill-port.sh "$PORT" > /dev/null; sleep 3
done
echo "cache reserve test $label done"

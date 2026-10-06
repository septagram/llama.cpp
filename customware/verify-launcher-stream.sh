#!/usr/bin/env bash
# Release check for the streaming path on the deployed launcher (port 5000, default ping interval of 30 s):
# a large request generates; a second large one that cannot fit is posted as a stream and must get its headers
# and pings while it waits, then its answer. Also the plain 22k code prompt for the speed line.
#   usage: [LAUNCHER=...customware-120k.sh] customware/verify-launcher-stream.sh <label>
if curl -s -m 2 http://127.0.0.1:5000/health 2>/dev/null | grep -q '"ok"'; then echo "port 5000 is busy (the daily server?): not starting a second launcher, not killing anything"; exit 1; fi
set -uo pipefail
cd "$(dirname "$0")/.."
repo=$PWD
label=${1:?label}
LOG=$repo/customware/serve-$label.log
: > "$LOG"
LAUNCHER=${LAUNCHER:-Qwen3.8-Flash-Next-UD-Q4_K_XL.customware-120k.sh}
(cd /c/dev/textgen && bash "./$LAUNCHER" >> "$LOG" 2>&1 &)
for i in $(seq 1 180); do
  sleep 5
  if curl -s -m 2 http://127.0.0.1:5000/health 2>/dev/null | grep -q '"ok"'; then echo "health ok after $((i*5)) s"; break; fi
  if grep -q "CUDA error\|error loading model\|failed to load\|GGML_ASSERT" "$LOG"; then echo "LAUNCH FAILED"; grep -E "CUDA error|error loading|failed to load|GGML_ASSERT" "$LOG" | head -3; bash customware/kill-port.sh 5000 > /dev/null 2>&1; exit 1; fi
done
echo "vram: $(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader | tr '\n' ' ')"
mk() { # <out> <base> <n_predict> <tag> [ignore_eos]
  python -c "
import json;d=json.load(open('$2'))
d['max_tokens']=$3; d['temperature']=0; d['top_k']=1; d['seed']=7; d['chat_template_kwargs']={'enable_thinking': False}; d['reasoning_budget']=0
if '${5:-}'=='ignore_eos': d['ignore_eos']=True
d['messages'][0]['content']='[$4] '+d['messages'][0]['content']
json.dump(d,open('$1','w'))"; }
mk customware/vls-A.json customware/depth-request-88k.json 700 vlsA ignore_eos
mk customware/vls-B.json customware/depth-request-44k.json 32  vlsB ignore_eos
clean() { sed -E 's/\x1b\[[0-9;]*m//g' "$LOG"; }
T0=$(date +%s)
(curl -s -m 3600 http://127.0.0.1:5000/v1/chat/completions -H "Content-Type: application/json" -d @customware/vls-A.json > "customware/ab-$label-A.json"; python -c "
import json
d=json.load(open('customware/ab-$label-A.json')); t=d.get('timings',{})
print('  A: answered at +%d s (prompt_n=%d at %.1f t/s, predicted_n=%d at %.2f t/s)' % ($(date +%s)-$T0, t.get('prompt_n',0), t.get('prompt_per_second',0), t.get('predicted_n',0), t.get('predicted_per_second',0)))") &
for i in $(seq 1 900); do sleep 1; clean | grep -q "n_gen = " && break; done
echo "  A is generating at +$(( $(date +%s)-T0 )) s; B (44k, cannot fit next to it) posted as a stream"
python customware/sse-client.py 5000 customware/vls-B.json B
wait
echo "streams started with a ping: $(clean | grep -c 'the stream starts with a ping')   waiting for cells: $(clean | grep -c 'waiting for cells')   restore failures: $(clean | grep -c 'failed to find [0-9]* available')"
echo "vram: $(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader | tr '\n' ' ')"
bash customware/kill-port.sh 5000 > /dev/null 2>&1
sleep 2
echo "launcher stream check $label done"

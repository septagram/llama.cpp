#!/usr/bin/env bash
# never touch a server that is already on 5000 (the daily one): this script starts and stops its own
if curl -s -m 2 http://127.0.0.1:5000/health 2>/dev/null | grep -q "\"ok\""; then echo "port 5000 is busy (the daily server?): not starting a second launcher, not killing anything"; exit 1; fi
# Start the released launcher exactly as the user does (C:\dev\textgen, port 5000), check health, compare the greedy
# log-probs of the fixed 48-token request with the earlier classic-hot and plain results, run the 22k code prompt once,
# print the hot-expert log lines, stop the server.
set -uo pipefail
cd "$(dirname "$0")/.."
repo=$PWD
LOG=$repo/customware/launcher-verify.log
: > "$LOG"
LAUNCHER=${LAUNCHER:-Qwen3.8-Flash-Next-UD-Q4_K_XL.customware.sh}
(cd /c/dev/textgen && bash "./$LAUNCHER" >> "$LOG" 2>&1 &)
for i in $(seq 1 180); do
  sleep 5
  if curl -s -m 2 http://127.0.0.1:5000/health 2>/dev/null | grep -q '"ok"'; then echo "health ok after $((i*5)) s"; break; fi
  if grep -q "CUDA error\|error loading model\|failed to load\|GGML_ASSERT" "$LOG"; then echo "LAUNCH FAILED"; grep -E "CUDA error|error loading|failed to load|GGML_ASSERT" "$LOG" | head -3; bash customware/kill-port.sh 5000 > /dev/null 2>&1; exit 1; fi
done
grep -E "hot experts|hot_alias_place|compute buffer size|KV buffer size|placement from|no usable stats" "$LOG" | cut -c1-200
echo "vram: $(nvidia-smi --query-gpu=memory.used --format=csv,noheader -i 1)"
curl -s -m 600 http://127.0.0.1:5000/v1/chat/completions -H "Content-Type: application/json" -d @customware/equiv-request-lp.json > customware/equiv-lp-launcher.json
echo "== greedy log-probs vs classic hot (160k / ub 1024)"; python customware/equiv-compare.py customware/equiv-lp-hot4000-ub1024.json customware/equiv-lp-launcher.json 2>&1 | head -6
echo "== greedy log-probs vs plain 3 layers"; python customware/equiv-compare.py customware/equiv-lp-plain3-ub1024.json customware/equiv-lp-launcher.json 2>&1 | head -6
curl -s -m 1800 http://127.0.0.1:5000/v1/chat/completions -H "Content-Type: application/json" -d @customware/smoke-request-code.json > customware/ab-launcher-code.json
python -c "
import json
d=json.load(open('customware/ab-launcher-code.json'))
t=d.get('timings',{})
print('launcher code prompt: prompt_n=%d (%.1f t/s) predicted_n=%d (%.2f t/s)' % (t.get('prompt_n',0), t.get('prompt_per_second',0), t.get('predicted_n',0), t.get('predicted_per_second',0)))
"
echo "vram: $(nvidia-smi --query-gpu=memory.used --format=csv,noheader -i 1)"
grep -E "hot experts: [0-9]+ tokens|hot_refill|prefill mode" "$LOG" | cut -c1-200 | tail -8
bash customware/kill-port.sh 5000 > /dev/null 2>&1
sleep 2
ls -la /c/dev/llama.cpp-customware/hot-experts.stats 2>&1 | head -1
echo "verify done"

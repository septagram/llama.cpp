#!/usr/bin/env bash
# never touch a server that is already on 5000 (the daily one): this script starts and stops its own
if curl -s -m 2 http://127.0.0.1:5000/health 2>/dev/null | grep -q "\"ok\""; then echo "port 5000 is busy (the daily server?): not starting a second launcher, not killing anything"; exit 1; fi
# Problem 1 measurement on the 120k launcher (port 5000):
#   a) agentic-shaped 30k prompt (82 short turns) alone
#   b) a 1500-token generation in flight, then the agentic prompt (variant, uncached)
#   c) a 1500-token generation in flight, then the plain 22k prompt (variant, uncached)
# Prints prefill/decode timings and how many prompt batches the server used ("prompt processing progress" lines).
set -uo pipefail
cd "$(dirname "$0")/.."
LOG=$PWD/customware/serve-conc.log
: > "$LOG"
(cd /c/dev/textgen && LLAMA_HOT_EXPERTS_LOG=100000 bash ./Qwen3.8-Flash-Next-UD-Q4_K_XL.customware-120k.sh >> "$LOG" 2>&1 &)
for i in $(seq 1 60); do sleep 5; curl -s -m 2 http://127.0.0.1:5000/health 2>/dev/null | grep -q '"ok"' && { echo "launcher up after $((i*5)) s"; break; }; done
timing() { python -c "
import json,sys
d=json.load(open('$1'));t=d.get('timings',{})
print('$2: prompt_n=%d (%.1f t/s, %.1f s) predicted_n=%d (%.2f t/s)' % (t.get('prompt_n',0),t.get('prompt_per_second',0),t.get('prompt_ms',0)/1000,t.get('predicted_n',0),t.get('predicted_per_second',0)))"; }
variant() { python -c "
import json;d=json.load(open('$1'));d['messages'][0]['content']='[$3] '+d['messages'][0]['content'];json.dump(d,open('$2','w'))"; }
batches() { local before=$1; local now; now=$(grep -c "prompt processing, n_tokens" "$LOG"); echo "  prompt batches: $((now-before))"; }

echo "== a) agentic 30k alone"
b0=$(grep -c "prompt processing, n_tokens" "$LOG")
curl -s -m 3600 http://127.0.0.1:5000/v1/chat/completions -H "Content-Type: application/json" -d @customware/agentic-request-30k.json > customware/ab-conc-a.json
timing customware/ab-conc-a.json "a"; batches $b0

echo "== b) long generation in flight + agentic 30k (variant)"
variant customware/agentic-request-30k.json customware/agentic-request-30k-v2.json v2
b0=$(grep -c "prompt processing, n_tokens" "$LOG")
curl -s -m 3600 http://127.0.0.1:5000/v1/chat/completions -H "Content-Type: application/json" -d @customware/longgen-request.json > customware/ab-conc-b-gen.json &
sleep 12
curl -s -m 3600 http://127.0.0.1:5000/v1/chat/completions -H "Content-Type: application/json" -d @customware/agentic-request-30k-v2.json > customware/ab-conc-b.json
timing customware/ab-conc-b.json "b prefill"; batches $b0
wait
timing customware/ab-conc-b-gen.json "b generator"

echo "== c) long generation in flight + plain 22k (variant)"
variant customware/depth-request-22k.json customware/depth-request-22k-v2.json v2
b0=$(grep -c "prompt processing, n_tokens" "$LOG")
curl -s -m 3600 http://127.0.0.1:5000/v1/chat/completions -H "Content-Type: application/json" -d @customware/longgen-request.json > customware/ab-conc-c-gen.json &
sleep 12
curl -s -m 3600 http://127.0.0.1:5000/v1/chat/completions -H "Content-Type: application/json" -d @customware/depth-request-22k-v2.json > customware/ab-conc-c.json
timing customware/ab-conc-c.json "c prefill"; batches $b0
wait
timing customware/ab-conc-c-gen.json "c generator"

echo "== d) plain 22k alone (variant 3)"
variant customware/depth-request-22k.json customware/depth-request-22k-v3.json v3
b0=$(grep -c "prompt processing, n_tokens" "$LOG")
curl -s -m 3600 http://127.0.0.1:5000/v1/chat/completions -H "Content-Type: application/json" -d @customware/depth-request-22k-v3.json > customware/ab-conc-d.json
timing customware/ab-conc-d.json "d"; batches $b0
grep -c "hot_refill" "$LOG" | sed 's/^/  refills total: /'
bash customware/kill-port.sh 5000 > /dev/null 2>&1
echo "conc test done"

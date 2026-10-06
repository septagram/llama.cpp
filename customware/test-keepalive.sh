#!/usr/bin/env bash
# Entry 011: a waiting streaming request is kept alive, and a slot release retries the whole deferred queue.
# Real model, port 5011, pool 24576, --sse-ping-interval 5. customware/sse-client.py timestamps the headers,
# the pings and the data events.
#   Q: A (12.3k, 400 tokens) generates; B (12.3k, other tag, streaming) cannot fit and queues, then prefills cold.
#      Pass: B's headers arrive after about one ping interval, pings follow every ~5 s through the queue wait and
#      the prefill, then the normal stream ends with [DONE]; the longest silence stays near the interval.
#   E: a prompt larger than the pool, streaming. Pass: an HTTP error status with a JSON error, not a stream.
#   F: a short streaming request that answers at once. Pass: no ping before the first data event.
#   R: X (12.3k cached, 700 tokens) and Y (tiny, 150 tokens) occupy both slots; Z (12.3k, cannot fit next to X) and
#      D (tiny) are queued in that order. Pass: D is answered before X finishes (when Y's slot is released the
#      whole queue is retried: Z still waits, D starts). Before this entry only Z was retried and D waited for X.
#   usage: [BIN=...] [PING=5] customware/test-keepalive.sh <label>
set -uo pipefail
cd "$(dirname "$0")/.."
label=${1:?label}
PORT=${PORT:-5011}
PING=${PING:-5}
LOG=customware/serve-$label.log
: > "$LOG"
(env OT='.*_exps=CPU' LLAMA_HOT_EXPERTS=customware/hot-experts-mixed-4000.txt LLAMA_HOT_EXPERTS_ALIAS=0 LLAMA_HOT_EXPERTS_SWAP=0 GGML_OP_OFFLOAD_MIN_BATCH=512 CTX=24576 UB=2048 CACHE_RAM=8192 bash customware/serve.sh "$PORT" -lv 4 --sse-ping-interval "$PING" > "$LOG" 2>&1 &)
for i in $(seq 1 120); do sleep 5; curl -s -m 2 "http://127.0.0.1:$PORT/health" 2>/dev/null | grep -q '"ok"' && { echo "server up after $((i*5)) s"; break; }; done
mk() { # <out> <base> <n_predict> <tag> [ignore_eos]
  python -c "
import json;d=json.load(open('$2'))
d['max_tokens']=$3; d['temperature']=0; d['top_k']=1; d['seed']=7; d['chat_template_kwargs']={'enable_thinking': False}; d['reasoning_budget']=0
if '${5:-}'=='ignore_eos': d['ignore_eos']=True
d['messages'][0]['content']='[$4] '+d['messages'][0]['content']
json.dump(d,open('$1','w'))"; }
mk customware/ka-A.json  customware/smoke-request-prose.json 400 kaA ignore_eos
mk customware/ka-B.json  customware/smoke-request-prose.json 48  kaB ignore_eos
mk customware/ka-E.json  customware/depth-request-44k.json   16  kaE
mk customware/ka-F.json  customware/equiv-request-lp.json    32  kaF ignore_eos
mk customware/ka-X.json  customware/smoke-request-prose.json 700 kaA ignore_eos
mk customware/ka-Y.json  customware/equiv-request-lp.json    150 kaY ignore_eos
mk customware/ka-Z.json  customware/smoke-request-prose.json 32  kaB ignore_eos
mk customware/ka-D.json  customware/equiv-request-lp.json    48  kaD ignore_eos
T0=$(date +%s)
plain() { # <name> <request> : non-streaming, prints when it was answered
  curl -s -m 2400 "http://127.0.0.1:$PORT/v1/chat/completions" -H "Content-Type: application/json" -d @"$2" > "customware/ab-$label-$1.json"
  python -c "
import json
d=json.load(open('customware/ab-$label-$1.json'))
if 'error' in d: print('  $1: ERROR', d['error']); raise SystemExit
t=d.get('timings',{}); print('  $1: answered at +%d s (prompt_n=%d cache_n=%d predicted_n=%d)' % ($(date +%s)-$T0, t.get('prompt_n',0), t.get('cache_n',0), t.get('predicted_n',0)))"
}
wait_gen() { # wait until the log shows <n> generation progress lines more than now
  local base; base=$(grep -c "n_gen = " "$LOG")
  for i in $(seq 1 300); do sleep 1; [ "$(grep -c "n_gen = " "$LOG")" -gt "$base" ] && return 0; done
}

echo "== F: a request that answers at once"
python customware/sse-client.py "$PORT" customware/ka-F.json F
echo "== E: a prompt larger than the pool"
python customware/sse-client.py "$PORT" customware/ka-E.json E
echo "== Q: a streaming request queued behind a generating one, then a cold prefill"
plain A customware/ka-A.json &
wait_gen
echo "  A is generating at +$(( $(date +%s)-T0 )) s; B posted now"
python customware/sse-client.py "$PORT" customware/ka-B.json B --detail | tail -4
wait
echo "== R: both slots busy, Z (large) and D (small) queued in that order"
T0=$(date +%s)
plain X customware/ka-X.json &
wait_gen
plain Y customware/ka-Y.json & sleep 5
plain Z customware/ka-Z.json & sleep 1
plain D customware/ka-D.json
wait
echo "== server log"
echo "streams started with a ping: $(grep -c 'the stream starts with a ping' "$LOG")   waiting for cells: $(grep -c 'waiting for cells' "$LOG")   failures: $(grep -c 'failed to find [0-9]* available\|GGML_ASSERT' "$LOG")"
bash customware/kill-port.sh "$PORT" > /dev/null; sleep 2
echo "keepalive test $label done"

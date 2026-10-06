#!/usr/bin/env bash
# Entry 010, the two paths test-cache-hit-real.sh does not reach, at small scale on port 5011 (real model).
#   E, "early restore" (the trace that cost the user most): pool 24576. B (12k) is cached. C (22k, new) starts
#      prefilling; B2 arrives a few seconds in, when the pool still has room for B's 12k state but will not once
#      C's prompt is in. The restore must wait ("does not fit next to ... kept for later"), B2 queues, and after C
#      it comes back from the cache. Before the fix the state was restored, purged unsaved when C's prefill ran
#      out of cells, and B2 prefilled from zero. Pass: B2 cache_n ~12.3k, no purge, no restore failure.
#   P, "purge keeps the prompt": pool 34816, LLAMA_KV_MARGIN=0. B (22k, 400 tokens, ignore_eos) and A (12k) run
#      together; A finishes and sits idle with 12.4k cells; B's generation runs the pool out ~234 tokens in and
#      A's idle slot is purged. Pass: the purge is preceded by a save and A2 comes back with cache_n ~12.3k.
#   usage: [BIN=...] [ONLY=E|P] customware/test-cache-hit-paths.sh <label>
set -uo pipefail
cd "$(dirname "$0")/.."
label=${1:?label}
PORT=${PORT:-5011}
ONLY=${ONLY:-EP}
mk() { # <out> <base> <n_predict> <tag> [ignore_eos]
  python -c "
import json;d=json.load(open('$2'))
d['max_tokens']=$3; d['temperature']=0; d['top_k']=1; d['seed']=7; d['chat_template_kwargs']={'enable_thinking': False}; d['reasoning_budget']=0
if '${5:-}'=='ignore_eos': d['ignore_eos']=True
d['messages'][0]['content']='[$4] '+d['messages'][0]['content']
json.dump(d,open('$1','w'))"; }
mk customware/hitp-A.json      customware/smoke-request-prose.json 16  pathA
mk customware/hitp-B12.json    customware/smoke-request-prose.json 16  pathB
mk customware/hitp-C22.json    customware/depth-request-22k.json   16  pathC
mk customware/hitp-B22long.json customware/depth-request-22k.json  400 pathBL ignore_eos
start() { # <ctx> <log> [env...]
  local ctx=$1 log=$2; shift 2
  : > "$log"
  (env "$@" OT='.*_exps=CPU' LLAMA_HOT_EXPERTS=customware/hot-experts-mixed-4000.txt LLAMA_HOT_EXPERTS_ALIAS=0 LLAMA_HOT_EXPERTS_SWAP=0 GGML_OP_OFFLOAD_MIN_BATCH=512 CTX="$ctx" UB=2048 CACHE_RAM=8192 bash customware/serve.sh "$PORT" -lv 4 > "$log" 2>&1 &)
  for i in $(seq 1 120); do sleep 5; curl -s -m 2 "http://127.0.0.1:$PORT/health" 2>/dev/null | grep -q '"ok"' && { echo "server up after $((i*5)) s"; break; }; done
}
post() { # <name> <request>
  local t0=$(date +%s)
  curl -s -m 2400 "http://127.0.0.1:$PORT/v1/chat/completions" -H "Content-Type: application/json" -d @"$2" > "customware/ab-$label-$1.json"
  python -c "
import json
d=json.load(open('customware/ab-$label-$1.json'))
if 'error' in d: print('  $1: ERROR', d['error']); raise SystemExit
t=d.get('timings',{}); print('  $1: prompt_n=%d cache_n=%d predicted_n=%d, %d s wall' % (t.get('prompt_n',0), t.get('cache_n',0), t.get('predicted_n',0), $(date +%s)-$t0))"
}
verdict() { # <log>
  echo "  restore failures: $(grep -c 'failed to find [0-9]* available cells\|failed to load prompt from cache' "$1")   kept for later: $(grep -c 'kept for later' "$1")   purges: $(grep -c 'purging slot' "$1")   deferrals: $(grep -c 'waiting for cells' "$1")   idle slots freed for a restore: $(grep -c 'freeing idle slot' "$1")"
  grep -E "kept for later|purging slot|waiting for cells|freeing idle slot|saving prompt with length|failed to find [0-9]* available" "$1" | sed -E 's/^[0-9.]+ [IWE] //' | cut -c1-150
}
if [[ $ONLY == *E* ]]; then
  echo "== E: a cached prompt returns early in another slot's prefill"
  LOG=customware/serve-$label-E.log
  start 24576 "$LOG"
  post E-B1 customware/hitp-B12.json
  post E-C  customware/hitp-C22.json & sleep 14
  post E-B2 customware/hitp-B12.json
  wait
  verdict "$LOG"
  bash customware/kill-port.sh "$PORT" > /dev/null; sleep 3
fi
if [[ $ONLY == *P* ]]; then
  echo "== P: an idle slot purged by a pool overflow keeps its prompt"
  LOG=customware/serve-$label-P.log
  start 34816 "$LOG" LLAMA_KV_MARGIN=0
  post P-B  customware/hitp-B22long.json & sleep 4
  post P-A1 customware/hitp-A.json
  wait
  post P-A2 customware/hitp-A.json
  verdict "$LOG"
  bash customware/kill-port.sh "$PORT" > /dev/null; sleep 3
fi
echo "cache hit paths $label done"

#!/usr/bin/env bash
# Entry 010 on the deployed launcher (port 5000, 120k unified pool): the user's failure, both steps.
#   A = the 88k depth prompt (~92k tokens), B = the 44k depth prompt, tagged so they share no prefix; together they
#   exceed the pool. A1, B1 fill the prompt cache. Then A2 (400 tokens, ignore_eos) is started and B2 is posted
#   while A2 generates: B2's cached state cannot fit next to A2, so the restore must not be attempted ("kept for
#   later"), B2 must wait ("waiting for cells"), and on the retry the idle A slot must be freed first and B2
#   restored from the cache. Pass: A2 and B2 report cache_n near their prompt length, zero "failed to find".
#   SCEN=early replays the other field trace instead: B (44k) is cached, C (89k, new) prefills from zero, B2 arrives
#   25 s in. Pass: B2 waits ("does not fit next to ... kept for later"), no purge, and comes back from the cache.
#   usage: [SCEN=two-step|early] [LAUNCHER=...customware-120k.sh] customware/test-cache-hit-launcher.sh <label>
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
mk customware/hitl-A.json     customware/depth-request-88k.json 16  hitA
mk customware/hitl-A-long.json customware/depth-request-88k.json 400 hitA ignore_eos
mk customware/hitl-B.json     customware/depth-request-44k.json 16  hitB
post() { # <name> <request>
  local t0=$(date +%s)
  curl -s -m 3600 http://127.0.0.1:5000/v1/chat/completions -H "Content-Type: application/json" -d @"$2" > "customware/ab-$label-$1.json"
  python -c "
import json
d=json.load(open('customware/ab-$label-$1.json'))
if 'error' in d: print('  $1: ERROR', d['error']); raise SystemExit
t=d.get('timings',{}); print('  $1: prompt_n=%d cache_n=%d (%.1f t/s) predicted_n=%d (%.2f t/s), %d s wall' % (t.get('prompt_n',0), t.get('cache_n',0), t.get('prompt_per_second',0), t.get('predicted_n',0), t.get('predicted_per_second',0), $(date +%s)-$t0))"
}
if [ "${SCEN:-two-step}" = early ]; then
  # the trace that cost the most in the field: B's cached state returns a few seconds into C's prefill from zero,
  # when the pool still has room for it but will not once C's prompt is in
  mk customware/hitl-C.json customware/depth-request-88k.json 16 hitC
  echo "== fill the cache: B1"
  post B1 customware/hitl-B.json
  echo "== C prefilling from zero, B2 arrives 25 s in"
  post C customware/hitl-C.json & sleep 25
  post B2 customware/hitl-B.json
  wait
else
  echo "== fill the cache: A1 then B1"
  post A1 customware/hitl-A.json
  post B1 customware/hitl-B.json
  echo "== A2 generating, B2 arrives behind it"
  post A2 customware/hitl-A-long.json & sleep 15
  post B2 customware/hitl-B.json
  wait
fi
echo "== verdict"
clean() { sed -E 's/\x1b\[[0-9;]*m//g' "$LOG"; }
echo "restore failures (failed to find / failed to load): $(clean | grep -c 'failed to find\|failed to load prompt from cache')"
echo "restores skipped until they fit (kept for later):   $(clean | grep -c 'kept for later')"
echo "idle slots freed for a cached prompt:               $(clean | grep -c 'freeing idle slot')"
echo "idle slots purged by a pool overflow:               $(clean | grep -c 'purging slot')"
echo "admission deferrals (waiting for cells):            $(clean | grep -c 'waiting for cells')"
clean | grep -E "freeing idle slot|kept for later|waiting for cells|failed to find|purging slot|saving prompt with length|found better prompt" | sed -E 's/^[0-9.]+ [IWE] //' | cut -c1-150
echo "vram: $(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader | tr '\n' ' ')"
bash customware/kill-port.sh 5000 > /dev/null 2>&1
sleep 2
echo "launcher cache hit test $label done"

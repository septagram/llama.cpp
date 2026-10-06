#!/usr/bin/env bash
# Entry 013, the scheduler, on the real model (port 5011, pool 24576, 3 slots, --sse-ping-interval 5, -lv 4).
#   P1 priority preemption: A (12.3k prose prompt, 600 tokens, ignore_eos) generates; H (a 12.3k prompt with another
#      tag, priority 5, 48 tokens) arrives: it cannot fit next to A, so A is suspended ("displaced by a higher
#      priority"), H runs at once, A resumes afterwards and finishes. Pass: H answered before A, A's
#      predicted_n = 600, "suspended"/"resumed" in the log, no error.
#   P2 idle conversations stay: after P1 the finished conversations sit in their slots; S (a 42-token request)
#      launches and the log shows no "saving idle slot" at the launch; then A2 = A's conversation + a follow-up
#      continues from the slot (cache_n ~12.9k) without a cache restore ("found better prompt" count unchanged).
#   P3 aging: LLAMA_SCHED_AGE_S=20. B (12.3k, 600 tokens, ignore_eos) generates; W (12.3k, other tag, prio 0,
#      48 tokens) cannot fit and waits; after ~20 s W is promoted and B suspended; W answers; B resumes. Pass: W's
#      wall time well under B's, "displaced" in the log once more.
#   P4 overflow still handled: two 2200-token tails on a shared 12.4k prefix (as test-fork.sh F4): suspended, resumed, both finish.
#   usage: [ONLY=1234] customware/test-sched.sh <label>
set -uo pipefail
cd "$(dirname "$0")/.."
label=${1:?label}
PORT=${PORT:-5011}
ONLY=${ONLY:-1234}
LOG=customware/serve-$label.log
: > "$LOG"
(env LLAMA_SCHED_AGE_S=20 OT='.*_exps=CPU' LLAMA_HOT_EXPERTS=customware/hot-experts-mixed-4000.txt LLAMA_HOT_EXPERTS_ALIAS=0 LLAMA_HOT_EXPERTS_SWAP=0 GGML_OP_OFFLOAD_MIN_BATCH=512 CTX=24576 NP=3 UB=2048 CACHE_RAM=8192 bash customware/serve.sh "$PORT" -lv 4 --sse-ping-interval 5 > "$LOG" 2>&1 &)
for i in $(seq 1 120); do sleep 5; curl -s -m 2 "http://127.0.0.1:$PORT/health" 2>/dev/null | grep -q '"ok"' && { echo "server up after $((i*5)) s"; break; }; done
grep -E "n_slots = " "$LOG" | sed -E 's/^[0-9.]+ [IWE] //' | cut -c1-100
mk() { # <out> <base> <n_predict> <tag> <priority> [ignore_eos]
  python -c "
import json;d=json.load(open('$2'))
d['max_tokens']=$3; d['temperature']=0; d['top_k']=1; d['seed']=7; d['chat_template_kwargs']={'enable_thinking': False}; d['reasoning_budget']=0
d['priority']=$5
if '${6:-}'=='ignore_eos': d['ignore_eos']=True
d['messages'][0]['content']='[$4] '+d['messages'][0]['content']
json.dump(d,open('$1','w'))"; }
T0=$(date +%s)
post() { # <name> <request>
  local t0=$(date +%s)
  curl -s -m 3600 "http://127.0.0.1:$PORT/v1/chat/completions" -H "Content-Type: application/json" -d @"$2" > "customware/ab-$label-$1.json"
  python -c "
import json
d=json.load(open('customware/ab-$label-$1.json'))
if 'error' in d: print('  $1: ERROR', d['error']); raise SystemExit
t=d.get('timings',{}); print('  $1: posted +%d s, done +%d s: prompt_n=%d cache_n=%d predicted_n=%d at %.2f t/s' % ($t0-$T0, $(date +%s)-$T0, t.get('prompt_n',0), t.get('cache_n',0), t.get('predicted_n',0), t.get('predicted_per_second',0)))"
}
clean() { sed -E 's/\x1b\[[0-9;]*m//g' "$LOG"; }
wait_gen() { local base; base=$(clean | grep -c "n_gen = "); for i in $(seq 1 600); do sleep 1; [ "$(clean | grep -c "n_gen = ")" -gt "$base" ] && return 0; done; }
counts() { echo "  displaced: $(clean | grep -c 'displaced by a higher priority')   suspended: $(clean | grep -c 'suspended:')   resumed: $(clean | grep -c 'resumed after')   purges: $(clean | grep -c 'purging slot')   saves at launch: $(clean | grep -c 'saving idle slot')   restores: $(clean | grep -c 'found better prompt')   waiting: $(clean | grep -c 'waiting for cells')   errors: $(clean | grep -c 'Context size has been exceeded\|GGML_ASSERT\|error loading state')"; }

if [[ $ONLY == *1* ]]; then
  echo "== P1: a higher priority request displaces the running one"
  mk customware/sc-A.json customware/smoke-request-prose.json 600 schedA 0 ignore_eos
  mk customware/sc-H.json customware/smoke-request-prose.json 48  schedH 5 ignore_eos
  post A customware/sc-A.json &
  wait_gen
  echo "  A is generating at +$(( $(date +%s)-T0 )) s; H (priority 5) posted"
  post H customware/sc-H.json
  wait
  counts
  clean | grep -E "scheduler \(|displaced|suspended:|resumed after|waiting for cells" | sed -E 's/^[0-9.]+ [IWE] //' | cut -c1-160 | head -8
fi
if [[ $ONLY == *2* ]]; then
  echo "== P2: idle conversations stay in their slots across a launch"
  mk customware/sc-S.json customware/equiv-request-lp.json 16 schedS 0
  saves0=$(clean | grep -c 'saving idle slot'); restores0=$(clean | grep -c 'found better prompt')
  post S customware/sc-S.json
  python customware/fork-requests.py customware/sc-A.json customware/ab-$label-A.json customware/sc 32 A2 > /dev/null
  post A2 customware/sc-A2.json
  echo "  saves at launch during P2: $(( $(clean | grep -c 'saving idle slot') - saves0 ))   cache restores during P2: $(( $(clean | grep -c 'found better prompt') - restores0 ))"
fi
if [[ $ONLY == *3* ]]; then
  echo "== P3: aging promotes a waiter after LLAMA_SCHED_AGE_S=20"
  mk customware/sc-B.json customware/smoke-request-prose.json 600 schedB 0 ignore_eos
  mk customware/sc-W.json customware/smoke-request-prose.json 48  schedW 0 ignore_eos
  post B customware/sc-B.json &
  wait_gen
  echo "  B is generating at +$(( $(date +%s)-T0 )) s; W (priority 0) posted"
  post W customware/sc-W.json
  wait
  counts
fi
if [[ $ONLY == *4* ]]; then
  echo "== P4: overflow of a shared pair"
  mk customware/sc-P.json customware/smoke-request-prose.json 48 schedP 0
  post P customware/sc-P.json
  python customware/fork-requests.py customware/sc-P.json customware/ab-$label-P.json customware/sc 2200 X Y > /dev/null
  post X customware/sc-X.json & sleep 3
  post Y customware/sc-Y.json
  wait
  counts
fi
echo "== /slots"; curl -s "http://127.0.0.1:$PORT/slots" | python -c "import json,sys; [print('  slot', s['id'], 'processing', s['is_processing'], 'priority', s.get('priority'), 'fork_of', s.get('fork_of')) for s in json.load(sys.stdin)]"
bash customware/kill-port.sh "$PORT" > /dev/null; sleep 2
echo "sched test $label done"

#!/usr/bin/env bash
# Entry 012, forking at small scale on the real model (port 5011, --sse-ping-interval 5, -lv 4).
#   F1 fan-out, pool 16384: P (the 12.3k prose prompt, 48 greedy tokens) ends; A and B, both P + the assistant
#      turn + a different follow-up (fork-requests.py), 200 tokens each, posted 3 s apart. Pass: A continues in
#      P's slot (cache_n ~12.4k), B is a fork ("fork of slot" in the log, cache_n ~12.4k), both generate at the
#      same time, no "waiting for cells" although two private copies would not fit the pool.
#   F3 exactness: C = the same request as B, served alone after A and B finished (from the cache or the slot).
#      Reports the common prefix of B's and C's texts (a near-tie flip is expected at some point, as in 007).
#   F4 pressure, pool 16384: A2 and B2 (same prefix, N4 = 2200 tokens each, ignore_eos) overflow the pool together
#      (12.4k shared + 2 x 2200 > 16384; 1500 each fitted).
#      Pass: one is suspended and resumed, both finish, no "Context size has been exceeded".
#   F5 (entry 014) the pair leaves the pool and comes back shared: after F1 (A continues P in one slot, B is a fork in
#      the other, both idle) an unrelated 12.3k request U needs their cells: the scheduler purges them (one prefix
#      segment written, two tail saves); U answers. B's request again (B3): no live holder, so its prefix comes back
#      from the segment; A's request again (A3): B is live, so A's tail is restored next to it and re-attached.
#      Pass: 1 segment, 2 tail saves, 1 segment restore, 1 re-attached restore, no restore failure, the pool after
#      A3's restore well under 2 x 12.4k cells (the restore line prints it), A3/B3 texts match A/B (modulo a near-tie).
#   usage: [ONLY=1534] customware/test-fork.sh <label>
set -uo pipefail
cd "$(dirname "$0")/.."
label=${1:?label}
PORT=${PORT:-5011}
ONLY=${ONLY:-1534}
N4=${N4:-2200}
LOG=customware/serve-$label.log
: > "$LOG"
(env OT='.*_exps=CPU' LLAMA_HOT_EXPERTS=customware/hot-experts-mixed-4000.txt LLAMA_HOT_EXPERTS_ALIAS=0 LLAMA_HOT_EXPERTS_SWAP=0 GGML_OP_OFFLOAD_MIN_BATCH=512 CTX=16384 UB=2048 CACHE_RAM=8192 bash customware/serve.sh "$PORT" -lv 4 --sse-ping-interval 5 > "$LOG" 2>&1 &)
for i in $(seq 1 120); do sleep 5; curl -s -m 2 "http://127.0.0.1:$PORT/health" 2>/dev/null | grep -q '"ok"' && { echo "server up after $((i*5)) s"; break; }; done
mk() { # <out> <base> <n_predict> <tag>
  python -c "
import json;d=json.load(open('$2'))
d['max_tokens']=$3; d['temperature']=0; d['top_k']=1; d['seed']=7; d['chat_template_kwargs']={'enable_thinking': False}; d['reasoning_budget']=0
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
overlap() { # how many generation progress lines of slot 0 and slot 1 are interleaved
  clean | grep -E "n_gen = " | sed -E 's/.*id +([0-9]+) \|.*/\1/' | awk 'NR>1 && $1!=prev {c++} {prev=$1} END {print c+0}'
}

if [[ $ONLY == *1* ]]; then
  echo "== F1: fan-out"
  mk customware/fk-P.json customware/smoke-request-prose.json 48 forkP
  post P customware/fk-P.json
  python customware/fork-requests.py customware/fk-P.json customware/ab-$label-P.json customware/fk 200 A B > /dev/null
  post A customware/fk-A.json & sleep 3
  post B customware/fk-B.json
  wait
  echo "  forks: $(clean | grep -c 'fork of slot')   waiting for cells: $(clean | grep -c 'waiting for cells')   restore failures: $(clean | grep -c 'failed to find [0-9]* available')   turn-end checkpoints: $(clean | grep -c 'created context checkpoint')   slot switches while generating: $(overlap)"
  clean | grep -E "fork of slot|forked from slot|not forking|waiting for cells|no slot is free" | sed -E 's/^[0-9.]+ [IWE] //' | cut -c1-150
fi
if [[ $ONLY == *5* ]]; then
  echo "== F5: the pair leaves the pool and comes back shared"
  mk customware/fk-U.json customware/smoke-request-prose.json 20 forkU
  post U customware/fk-U.json
  post B3 customware/fk-B.json
  post A3 customware/fk-A.json
  echo "  segments written: $(clean | grep -c 'wrote a .*prefix segment')   tail saves: $(clean | grep -c 'saved slot .* as a tail')   segment restores: $(clean | grep -c 'restored from a prefix segment')   re-attached restores: $(clean | grep -c 'restored a tail .* shared with slot')   restore failures: $(clean | grep -c 'failed to load prompt from cache')"
  clean | grep -E "prefix segment|as a tail|restored a tail|purging slot|restoring the tail here" | sed -E 's/^[0-9.]+ [IWE] //' | cut -c1-170
  python - <<EOF2
import json
for x in ('A', 'B'):
    a = json.load(open('customware/ab-$label-%s.json' % x))['choices'][0]['message'].get('content') or ''
    b = json.load(open('customware/ab-$label-%s3.json' % x))['choices'][0]['message'].get('content') or ''
    n = 0
    while n < min(len(a), len(b)) and a[n] == b[n]: n += 1
    print('  %s3 vs %s: %s (%d and %d chars)' % (x, x, 'IDENTICAL' if a == b else 'same for the first %d chars' % n, len(a), len(b)))
EOF2
fi
if [[ $ONLY == *3* ]]; then
  echo "== F3: the forked text against the same request served alone"
  post C customware/fk-B.json
  python - <<EOF
import json
b = json.load(open('customware/ab-$label-B.json'))['choices'][0]['message']; c = json.load(open('customware/ab-$label-C.json'))['choices'][0]['message']
tb = (b.get('content') or ''); tc = (c.get('content') or '')
n = 0
while n < min(len(tb), len(tc)) and tb[n] == tc[n]: n += 1
print('  B (fork, next to A) vs C (alone): %s (%d and %d chars)' % ('IDENTICAL' if tb == tc else 'same for the first %d chars' % n, len(tb), len(tc)))
EOF
fi
if [[ $ONLY == *4* ]]; then
  echo "== F4: the pair outgrows the pool"
  python customware/fork-requests.py customware/fk-P.json customware/ab-$label-P.json customware/fk $N4 A2 B2 > /dev/null
  post A2 customware/fk-A2.json & sleep 3
  post B2 customware/fk-B2.json
  wait
  echo "  suspended: $(clean | grep -c 'KV pool overflow: suspended')   resumed: $(clean | grep -c 'resumed after')   collective errors: $(clean | grep -c 'Context size has been exceeded')   purges: $(clean | grep -c 'purging slot')"
  clean | grep -E "suspended slot|resumed after|purging slot|Context size" | sed -E 's/^[0-9.]+ [IWE] //' | cut -c1-150 | tail -6
fi
echo "== /slots at the end"; curl -s "http://127.0.0.1:$PORT/slots" | python -c "import json,sys; [print('  slot', s['id'], 'fork_of', s.get('fork_of'), 'n_fork_shared', s.get('n_fork_shared')) for s in json.load(sys.stdin)]"
bash customware/kill-port.sh "$PORT" > /dev/null; sleep 2
echo "fork test $label done"

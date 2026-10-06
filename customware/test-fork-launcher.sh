#!/usr/bin/env bash
# Entry 012 on the deployed 120k launcher (port 5000): the precompaction flow.
#   M = the 88k depth prompt, 16 greedy tokens: the conversation ends (turn-end checkpoint).
#   M2 = M + the assistant turn + "continue", 400 tokens ignore_eos: the main agent's next turn, in M's slot.
#   C  = M + the assistant turn + "summarize", 200 tokens, posted 10 s later: the compaction fork.
# Pass: C is a fork of M2's slot ("fork of slot" in the log) with cache_n ~88.8k, both generate at the same time
# (interleaved progress lines), no "waiting for cells", VRAM unchanged. Both tails run near 7.5 t/s.
#   usage: [LAUNCHER=...customware-120k.sh] customware/test-fork-launcher.sh <label>
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
clean() { sed -E 's/\x1b\[[0-9;]*m//g' "$LOG"; }
mk() { # <out> <base> <n_predict> <tag>
  python -c "
import json;d=json.load(open('$2'))
d['max_tokens']=$3; d['temperature']=0; d['top_k']=1; d['seed']=7; d['chat_template_kwargs']={'enable_thinking': False}; d['reasoning_budget']=0
d['messages'][0]['content']='[$4] '+d['messages'][0]['content']
json.dump(d,open('$1','w'))"; }
T0=$(date +%s)
post() { # <name> <request>
  local t0=$(date +%s)
  curl -s -m 3600 http://127.0.0.1:5000/v1/chat/completions -H "Content-Type: application/json" -d @"$2" > "customware/ab-$label-$1.json"
  python -c "
import json
d=json.load(open('customware/ab-$label-$1.json'))
if 'error' in d: print('  $1: ERROR', d['error']); raise SystemExit
t=d.get('timings',{}); print('  $1: posted +%d s, done +%d s: prompt_n=%d (%.1f t/s) cache_n=%d predicted_n=%d at %.2f t/s' % ($t0-$T0, $(date +%s)-$T0, t.get('prompt_n',0), t.get('prompt_per_second',0), t.get('cache_n',0), t.get('predicted_n',0), t.get('predicted_per_second',0)))"
}
echo "== M: the conversation, cold"
mk customware/fkl-M.json customware/depth-request-88k.json 16 forkM
post M customware/fkl-M.json
python customware/fork-requests.py customware/fkl-M.json customware/ab-$label-M.json customware/fkl 400 M2 > /dev/null
python customware/fork-requests.py customware/fkl-M.json customware/ab-$label-M.json customware/fkl 200 C > /dev/null
echo "== M2 continues, C forks 10 s later"
post M2 customware/fkl-M2.json & sleep 10
post C customware/fkl-C.json
wait
echo "== verdict"
echo "forks: $(clean | grep -c 'fork of slot')   waiting for cells: $(clean | grep -c 'waiting for cells')   restore failures: $(clean | grep -c 'failed to find [0-9]* available')   slot switches while generating: $(clean | grep -E 'n_gen = ' | sed -E 's/.*id +([0-9]+) \|.*/\1/' | awk 'NR>1 && $1!=prev {c++} {prev=$1} END {print c+0}')"
clean | grep -E "fork of slot|forked from slot|not forking|waiting for cells|no slot is free|created context checkpoint" | sed -E 's/^[0-9.]+ [IWE] //' | cut -c1-150 | tail -8
echo "vram: $(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader | tr '\n' ' ')"
curl -s http://127.0.0.1:5000/slots | python -c "import json,sys; [print('  slot', s['id'], 'fork_of', s.get('fork_of'), 'n_fork_shared', s.get('n_fork_shared')) for s in json.load(sys.stdin)]"
bash customware/kill-port.sh 5000 > /dev/null 2>&1
sleep 2
echo "launcher fork test $label done"

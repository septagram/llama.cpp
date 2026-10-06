#!/usr/bin/env bash
# Fork feasibility probe (2026-10-06): does this build decode two sequences that share prefix cells, for qwen4exp?
# Upstream's own fork path is a request with n = 2: the parent slot processes the prompt, the child slot gets the
# state by llama_memory_seq_cp (labels only in a unified pool) and both generate. The pool here (16384 cells) cannot
# hold the 12.3k-token prompt twice, so the request can only complete if the prefix cells are shared.
# Then the same prompt with n = 1, for comparison of the texts (greedy).
#   usage: customware/test-fork-probe.sh <label>
set -uo pipefail
cd "$(dirname "$0")/.."
label=${1:?label}
PORT=${PORT:-5011}
LOG=customware/serve-$label.log
: > "$LOG"
(env OT='.*_exps=CPU' LLAMA_HOT_EXPERTS=customware/hot-experts-mixed-4000.txt LLAMA_HOT_EXPERTS_ALIAS=0 LLAMA_HOT_EXPERTS_SWAP=0 GGML_OP_OFFLOAD_MIN_BATCH=512 CTX=16384 UB=2048 CACHE_RAM=4096 bash customware/serve.sh "$PORT" -lv 4 > "$LOG" 2>&1 &)
for i in $(seq 1 120); do sleep 5; curl -s -m 2 "http://127.0.0.1:$PORT/health" 2>/dev/null | grep -q '"ok"' && { echo "server up after $((i*5)) s"; break; }; done
grep -E "n_ctx_slot|unified KV pool" "$LOG" | sed -E 's/^[0-9.]+ [IWE] //' | cut -c1-120 | head -2
mk() { # <out> <n>
  python -c "
import json;d=json.load(open('customware/smoke-request-prose.json'))
d['max_tokens']=48; d['temperature']=0; d['top_k']=1; d['seed']=7; d['n']=$2; d['chat_template_kwargs']={'enable_thinking': False}; d['reasoning_budget']=0
d['messages'][0]['content']='[fork] '+d['messages'][0]['content']
json.dump(d,open('$1','w'))"; }
mk customware/fork-n2.json 2
mk customware/fork-n1.json 1
for r in n2 n1; do
  curl -s -m 1800 "http://127.0.0.1:$PORT/v1/chat/completions" -H "Content-Type: application/json" -d @customware/fork-$r.json > "customware/ab-$label-$r.json"
done
python - <<EOF
import json
a = json.load(open('customware/ab-$label-n2.json')); b = json.load(open('customware/ab-$label-n1.json'))
for name, d in (('n=2', a), ('n=1', b)):
    if 'error' in d: print(name, 'ERROR', d['error']); continue
    t = d.get('timings', {}); u = d.get('usage', {})
    print(name, 'choices:', len(d['choices']), 'usage:', u, 'prompt_n=%s predicted_n=%s' % (t.get('prompt_n'), t.get('predicted_n')))
    for c in d['choices']:
        print('   [%d] %r' % (c.get('index', 0), (c['message'].get('content') or '')[:150]))
if 'error' not in a and 'error' not in b:
    ref = b['choices'][0]['message'].get('content') or ''
    for c in a['choices']:
        txt = c['message'].get('content') or ''
        n = 0
        while n < min(len(txt), len(ref)) and txt[n] == ref[n]: n += 1
        print('   choice %d vs the n=1 text: %s' % (c.get('index', 0), 'IDENTICAL' if txt == ref else 'same for %d of %d chars' % (n, len(ref))))
EOF
echo "-- server log"; grep -E "copying state to child|is_child = 1|failed to find|GGML_ASSERT|error|waiting for cells|purging|n_tokens = 12[0-9]{3}, truncated" "$LOG" | sed -E 's/^[0-9.]+ [IWE] //' | cut -c1-140 | head -12
bash customware/kill-port.sh "$PORT" > /dev/null; sleep 2
echo "fork probe $label done"

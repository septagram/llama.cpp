#!/usr/bin/env bash
# Speculative decoding on an echo-style request (the model returns a 60 KB file with a comment above every function):
# the launcher config without and with --spec-type ngram-mod, greedy, up to 1500 output tokens.
#   usage: customware/test-spec.sh <label> [PORT=5011] [SPEC_ARGS="--spec-type ngram-mod"]
set -uo pipefail
cd "$(dirname "$0")/.."
label=${1:?label}
PORT=${PORT:-5011}
SPEC_ARGS=${SPEC_ARGS:---spec-type ngram-mod}
STATS_SRC=/c/dev/llama.cpp-customware/hot-experts.stats

run() { # <phase> <server args...>
  local phase=$1; shift
  log=customware/serve-$label-$phase.log
  stats=customware/hot-experts-$label-$phase.stats
  cp -f "$STATS_SRC" "$stats"
  : > "$log"
  (OT='.*_exps=CPU' LLAMA_HOT_EXPERTS_ALIAS=1 GGML_OP_OFFLOAD_MIN_BATCH=512 LLAMA_HOT_EXPERTS_STATS="$stats" LLAMA_HOT_EXPERTS_MIB=4000 LLAMA_HOT_EXPERTS_LOG=256 CTX=160000 UB=2048 bash customware/serve.sh "$PORT" -lv 4 "$@" > "$log" 2>&1 &)
  for i in $(seq 1 180); do
    sleep 5
    if curl -s -m 2 "http://127.0.0.1:$PORT/health" 2>/dev/null | grep -q '"ok"'; then echo "health ok after $((i*5)) s"; break; fi
    if grep -q "CUDA error\|error loading model\|failed to load\|GGML_ASSERT\|error: " "$log"; then echo "server failed"; grep -E "CUDA error|error loading|failed to load|GGML_ASSERT|error: " "$log" | head -3; taskkill //F //IM llama-server.exe >/dev/null 2>&1; return 1; fi
  done
  for r in 1 2; do
    curl -s -m 3600 "http://127.0.0.1:$PORT/v1/chat/completions" -H "Content-Type: application/json" -d @customware/smoke-request-echo.json > "customware/ab-$label-$phase-echo$r.json"
    python -c "
import json
d=json.load(open('customware/ab-$label-$phase-echo$r.json'))
t=d.get('timings',{}); m=d['choices'][0]['message']
extra=''
if t.get('draft_n'): extra=' draft_n=%d accepted=%d (%.0f%%)' % (t['draft_n'], t.get('draft_n_accepted',0), 100.0*t.get('draft_n_accepted',0)/max(1,t['draft_n']))
print('$label $phase echo$r: prompt_n=%d (%.1f t/s) predicted_n=%d (%.2f t/s)%s | content %d chars, reasoning %d chars' % (t.get('prompt_n',0), t.get('prompt_per_second',0), t.get('predicted_n',0), t.get('predicted_per_second',0), extra, len(m.get('content') or ''), len(m.get('reasoning_content') or '')))
" 2>&1 | head -2
  done
  grep -E "draft acceptance|mean_acc_len|draft_ratio" "$log" | tail -2 | cut -c1-200
  taskkill //F //IM llama-server.exe >/dev/null 2>&1; sleep 3
}

echo "== $label nospec"; run nospec
echo "== $label spec ($SPEC_ARGS)"; run spec $SPEC_ARGS
echo "== $label done"

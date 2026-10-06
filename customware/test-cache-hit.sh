#!/usr/bin/env bash
# Entry 010: under --kv-unified a cached prompt must be restored whenever it exists, even when another idle slot
# still holds its cells (before: the restore ran first, failed for lack of cells, and the prompt was re-prefilled
# with its state sitting in RAM). Small scale, CPU only, so it runs next to the daily server: a 0.5B model, a
# 4096-cell pool, two prompts of ~2600 and ~2200 tokens that cannot share it, requested A, B, A, B.
# Pass: no "failed to find ... available cells", no "failed to load prompt from cache", and the second A and B
# start from their full cached length. BIN=<other server> reproduces the bug on an older build.
#   usage: [BIN=...] customware/test-cache-hit.sh <label>
set -uo pipefail
cd "$(dirname "$0")/.."
label=${1:?label}
PORT=${PORT:-5012}
MODEL=${MODEL:-/c/dev/textgen/user_data/models/tiny/Qwen2.5-0.5B-Instruct-Q4_K_M.gguf}
BIN=${BIN:-build/bin/llama-server.exe}
LOG=customware/serve-$label.log
: > "$LOG"
("$BIN" -m "$MODEL" -dev none -c 4096 -np 2 --kv-unified --cache-ram 1024 --cache-idle-slots -t 2 --no-warmup -fa off \
    --host 127.0.0.1 --port "$PORT" -lv 4 > "$LOG" 2>&1 &)
for i in $(seq 1 30); do sleep 2; curl -s -m 2 "http://127.0.0.1:$PORT/health" 2>/dev/null | grep -q '"ok"' && { echo "server up after $((i*2)) s"; break; }; done
python - <<'EOF'
import json
a = "The quick brown fox jumps over the lazy dog near the river bank. " * 240
b = "Seven silver spoons sat silently on the sunny southern shelf all day. " * 200
for name, text in (("A", a), ("B", b)):
    json.dump({"messages": [{"role": "user", "content": text + "\nReply with one word."}], "max_tokens": 8, "temperature": 0,
               "cache_prompt": True}, open(f"customware/hit-{name}.json", "w"))
EOF
post() { # <name> <request>
  curl -s -m 600 "http://127.0.0.1:$PORT/v1/chat/completions" -H "Content-Type: application/json" -d @"$2" > "customware/ab-$label-$1.json"
  python -c "
import json
d=json.load(open('customware/ab-$label-$1.json'))
if 'error' in d: print('  $1: ERROR', d['error']); raise SystemExit
t=d.get('timings',{}); print('  $1: prompt_n=%d cache_n=%d predicted_n=%d' % (t.get('prompt_n',0), t.get('cache_n',0), t.get('predicted_n',0)))"
}
echo "== A, B, A, B"
post A1 customware/hit-A.json; post B1 customware/hit-B.json; post A2 customware/hit-A.json; post B2 customware/hit-B.json
echo "== verdict"
fails=$(grep -c "failed to find\|failed to load prompt from cache" "$LOG")
echo "restore failures: $fails"
grep -E "cached n_tokens =|freeing idle slot|kept for later|saving idle slot|purging slot" "$LOG" | sed -E 's/^[0-9.]+ [IWE] //' | cut -c1-140
bash customware/kill-port.sh "$PORT" > /dev/null; sleep 1
echo "cache hit test $label done"

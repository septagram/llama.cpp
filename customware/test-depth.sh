#!/usr/bin/env bash
# Decode cost vs context depth on the running launcher (port 5000): 22k, 44k and 88k prompts, 64 greedy tokens each,
# thinking off. Prints prefill and per-token decode time; run twice per depth so the second is prompt-cached.
set -uo pipefail
cd "$(dirname "$0")/.."
for f in depth-request-22k depth-request-44k depth-request-88k; do
  for r in 1 2; do
    curl -s -m 3600 http://127.0.0.1:5000/v1/chat/completions -H "Content-Type: application/json" -d @customware/$f.json > customware/ab-depth-$f-$r.json
    python -c "
import json
d=json.load(open('customware/ab-depth-$f-$r.json'));t=d.get('timings',{})
print('$f run $r: prompt_n=%d (%.1f t/s) predicted_n=%d %.1f ms/token (%.2f t/s)' % (t.get('prompt_n',0),t.get('prompt_per_second',0),t.get('predicted_n',0),t.get('predicted_per_token_ms',0),t.get('predicted_per_second',0)))"
  done
done
echo "depth test done"

#!/usr/bin/env bash
# Does loading the multimodal projector on the second GPU cost the text path anything?
# Same server config with (MM=1) or without (MM=0) `--mmproj ... -mmdev CUDA1`; one 22k text request; per-GPU VRAM.
# With MM=1 an image request runs first (customware/image-request.json: red/blue rectangles on a 96x64 PNG).
#   usage: MM=0|1 customware/test-mmproj.sh <label>
set -uo pipefail
cd "$(dirname "$0")/.."
label=${1:?label}
PORT=${PORT:-5011}
MM=${MM:-1}
MMPROJ="/v/Model/Qwen3.8-Flash-Next-GGUF-UD-Q4_K_XL/qwen-3.8-flash-next-mmproj-F16.gguf"
LOG=customware/serve-$label.log
: > "$LOG"
mm_args=()
if [ "$MM" = 1 ]; then mm_args=(--mmproj "$MMPROJ" -mmdev CUDA1); fi
(OT='.*_exps=CPU' LLAMA_HOT_EXPERTS=customware/hot-experts-mixed-4000.txt LLAMA_HOT_EXPERTS_ALIAS=1 LLAMA_HOT_EXPERTS_SWAP=0 GGML_OP_OFFLOAD_MIN_BATCH=512 CTX="${CTX:-120000}" UB="${UB:-4096}" bash customware/serve.sh "$PORT" -b "${UB:-4096}" "${mm_args[@]}" > "$LOG" 2>&1 &)
for i in $(seq 1 120); do sleep 5; curl -s -m 2 "http://127.0.0.1:$PORT/health" 2>/dev/null | grep -q '"ok"' && { echo "server up after $((i*5)) s"; break; }; done
grep -E "loaded multimodal|unified KV pool" "$LOG" | cut -c1-200
vram() { nvidia-smi --query-gpu=index,memory.used --format=csv,noheader | tr '\n' ' '; echo; }
post() { # <name> <request>
  curl -s -m 7200 "http://127.0.0.1:$PORT/v1/chat/completions" -H "Content-Type: application/json" -d @"$2" > "customware/ab-$label-$1.json"
  python -c "
import json
d=json.load(open('customware/ab-$label-$1.json'))
if 'error' in d: print('$1: ERROR %s: %s' % (d['error'].get('type'), d['error'].get('message','')[:100])); raise SystemExit
t=d.get('timings',{}); m=d['choices'][0]['message']
print('$1: prompt_n=%d (%.1f t/s) predicted_n=%d (%.2f t/s)' % (t.get('prompt_n',0),t.get('prompt_per_second',0),t.get('predicted_n',0),t.get('predicted_per_second',0)))
if '$1'=='image': print('reply:', (m.get('content') or '')[:300])"
}
echo "vram before: $(vram)"
if [ "$MM" = 1 ]; then post image customware/image-request.json; fi
post text22k customware/depth-request-22k.json
echo "vram after: $(vram)"
bash customware/kill-port.sh "$PORT" > /dev/null 2>&1
sleep 3
echo "mmproj test $label done (MM=$MM)"

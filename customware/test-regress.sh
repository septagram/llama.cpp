#!/usr/bin/env bash
# The regression suite of the server-side entries, run one after the other on port 5011 (the GPU must be free):
#   entry 007 queueing/preemption/resume (test-kvu.sh, scenarios 1 2 3 4 5, exactness included),
#   entry 010 cache hits (test-cache-hit-real.sh), entry 011 keep-alive (test-keepalive.sh),
#   entry 012 forking (test-fork.sh F1+F4), the small request next to a large one (test-third-request.sh).
# Prints each script's verdict lines; the pass criteria are in the scripts' headers.
#   usage: customware/test-regress.sh <label>
set -uo pipefail
cd "$(dirname "$0")/.."
label=${1:?label}
run() { # <name> <command...>
  local name=$1; shift
  echo "#################### $name"
  "$@" > "customware/regress-$label-$name.out" 2>&1
  grep -E "ERROR|identical|IDENTICAL|restore failures|deferrals|forks:|suspended:|resumed:|collective|purges|waiting for cells|headers after|answered at|done$|kept for later|freeing idle|failures:" "customware/regress-$label-$name.out" | cut -c1-170 | head -24
}
run kvu          env SCEN="1 2 3 4 5" bash customware/test-kvu.sh "$label-kvu"
run cachehit     bash customware/test-cache-hit-real.sh "$label-hit"
run keepalive    bash customware/test-keepalive.sh "$label-ka"
run fork         env ONLY=154 bash customware/test-fork.sh "$label-fork"
run third        bash customware/test-third-request.sh "$label-third"
echo "regression suite $label done"

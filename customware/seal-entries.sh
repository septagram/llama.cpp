#!/usr/bin/env bash
# Seal customware entries: one incremental patch per entry (its own code commits, customware/ excluded), so the
# patches apply in numeric order onto the fork point. Works on a scratch index: the working tree and the real index
# are never touched. Fails if an entry's commits do not apply without a later entry's code, or if the patches
# together do not reproduce master's code exactly.
#   usage: [DRY=1] [BASE=v0.5.0-customware.0] customware/seal-entries.sh
set -euo pipefail
cd "$(dirname "$0")/.."
BASE=${BASE:-v0.5.0-customware.0}
MAP=${MAP:-customware/entry-commits.txt}
DRY=${DRY:-0}
idx=$(mktemp); err=$(mktemp)
trap 'rm -f "$idx" "$err"' EXIT
export GIT_INDEX_FILE="$idx"

git read-tree "$BASE"
prev=$(git write-tree)
patches=()
while read -r entry commits; do
  [[ -z "${entry:-}" || "$entry" == \#* ]] && continue
  for c in $commits; do
    [ -z "$(git diff --name-only "$c^" "$c" -- ':(exclude)customware')" ] && continue
    if git diff --binary --full-index "$c^" "$c" -- ':(exclude)customware' | git apply --cached --check 2>/dev/null; then
      git diff --binary --full-index "$c^" "$c" -- ':(exclude)customware' | git apply --cached
    else
      # the commit was written on top of a later-numbered entry's code (history interleaved the two): its own changes
      # are carried over by a three-way merge; a conflict stops the seal
      if ! git diff --binary --full-index "$c^" "$c" -- ':(exclude)customware' | git apply --cached --3way 2>"$err" ||
         ! git write-tree >/dev/null 2>&1; then
        echo "FAIL: entry $entry, commit $c does not apply even with a three-way merge:"; sed 's/^/  /' "$err" | head -8; exit 1
      fi
      echo "    $entry: $c applied by a three-way merge (written on top of a later entry's code)"
    fi
  done
  cur=$(git write-tree)
  md=$(ls customware/"$entry"-*.md | head -1)
  patch=${md%.md}.patch
  if [ "$prev" = "$cur" ]; then
    printf "%s %-40s %s\n" "$entry" "(no patch)" "no code of its own (carried by another entry's patch)"
    if [ "$DRY" = 0 ]; then rm -f "$patch"; fi
  else
    printf "%s %-40s %s\n" "$entry" "$(basename "$patch")" "$(git diff --shortstat "$prev" "$cur" -- ':(exclude)customware' | sed 's/^ //')"
    if [ "$DRY" = 0 ]; then git diff --binary --full-index "$prev" "$cur" -- ':(exclude)customware' > "$patch"; patches+=("$patch"); fi
  fi
  prev=$cur
done < "$MAP"

if git diff --quiet "$prev" master -- ':(exclude)customware'; then
  echo "replayed entries reproduce master's code exactly"
else
  echo "MISMATCH with master:"; git diff --stat "$prev" master -- ':(exclude)customware' | tail -5; exit 1
fi

# the written patch files, applied in numeric order onto the base, must give the same tree
if [ "$DRY" = 0 ]; then
  git read-tree "$BASE"
  for p in "${patches[@]}"; do
    git apply --cached --check "$p" 2>"$err" || { echo "FAIL: $p does not apply in sequence"; sed 's/^/  /' "$err" | head -5; exit 1; }
    git apply --cached "$p"
  done
  [ "$(git write-tree)" = "$prev" ] && echo "the ${#patches[@]} patch files apply in order onto $BASE and reproduce master's code" || { echo "MISMATCH after applying the patch files"; exit 1; }
fi

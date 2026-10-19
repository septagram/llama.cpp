# customware/012: forking v1 - a request that branches from a live conversation shares its prefix cells

## Goal

Two requests that branch from the same state both get a cache hit, cheaply, and run at the same time when the
pool holds the shared prefix plus both tails. Covers the user's two flows: subagent fan-out (several requests
with the parent's context arriving seconds apart) and precompaction (fork at ~100k; the fork writes the summary
while the main agent continues). Design agreed on 2026-10-06 (`customware/notes-forking.md`): fork in the pool
by sharing cells, sharing lasts while both sequences stay in the pool.

## What the engine gives (read and probed)

- `llama_memory_seq_cp(src, dst, 0, c)` in a unified pool adds the label `dst` to the cells of `src` at
  positions [0, c): no copy, no new cells. The indexer cache follows. The recurrent memory ignores the range and
  shares `src`'s current state.
- A checkpoint (host copy of the recurrent state at a position) restores into any sequence with
  `LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY`; the restore detaches `dst` from the shared recurrent cell first.
- Between task-processing steps every token in a slot's list has been decoded (the decode retry loop is inside
  `update_slots`, and completion tasks are declined while a decode runs), so a prefix up to any checkpoint of
  any slot is in the pool.
- `test-fork-probe.sh`: two sequences sharing a 12.3k prefix in a 16k pool decode correctly on this model.

## Design

1. **Fork step in `get_available_slot`** (`--kv-unified`, completion tasks, `cache_prompt`, no pinned slot).
   Candidates: every slot that holds tokens and is not suspended or waiting for a parent, busy or idle, by the
   common prefix with the new prompt (`f_sim > slot_prompt_similarity`). An idle candidate with the longest
   prefix is taken over as today (ties go to idle). When the longest prefix is in a busy slot `src`: take an idle
   slot `dst` (empty first, else least recently used; none -> defer), choose the fork point `c` = `src`'s latest
   checkpoint with `n_tokens <= lcp` and `n_tokens < prompt length` (none worth it -> today's path), check the
   fit (`kv_committed(true) + kv_need(prompt - c, n_predict) <= pool`, else today's path and admission defers),
   save `dst`'s old prompt to the cache as the LRU path does, then `dst.fork_from(src, ckpt)`:
   `seq_rm(dst)`, `seq_cp(src, dst, 0, c)`, restore the checkpoint into `dst`, `dst.tokens = src.tokens[0, c)`,
   `dst.checkpoints = { ckpt }`. The prompt loop then sees a full prefix match and processes only the suffix.
2. **Eager turn-end checkpoint**: when a completion task releases its slot, a checkpoint at the end of the
   generation is taken (same model conditions as the prompt-loop checkpoints; skipped if one is already there),
   so the recurrent state at a fork point exists before the fork arrives. The lazy one the next request would
   make at the same position is skipped.
3. **Share-aware accounting**: new `llama_memory_n_cells_owned(mem, ids, n)` (cells whose labels are all in the
   set). `kv_idle_retained` becomes the cells the idle slots own together (purgeable), optionally excluding one
   slot; admission charges a request only for what its slot does not already hold
   (`kv_need(prompt - lcp, n_predict)`) with that slot not counted as purgeable; `pick_victim` compares owned
   cells (what a suspension frees); `try_clear_idle_slots` purges the idle slot that owns the most cells and
   reports nothing to free when none owns any.
4. **Observability**: a log line per fork, `n_forks` next to the entry 007 counters, `fork_of` and
   `n_fork_shared` in `/slots`.

Not in v1 (documented): an idle source whose tokens diverge before its end is taken over in place as today (a
rewritten AI turn and a fork sibling look the same there; the fork() flows always find the source busy); a
fork saved to the prompt cache comes back with private cells; a cache entry longer than the busy source's prefix
is not preferred over the fork; pinned requests (`id_slot`) do not fork; non-unified pools do not fork.

## Steps

1. Engine: `llama_memory_n_cells_owned` (llama.h, llama-context.cpp).
2. Server: `fork_from`, slot fields and JSON, the turn-end checkpoint hook, the accounting, the fork step, the
   admission need, the counters.
3. `customware/test-fork.sh` (real model, port 5011): F1 fan-out in a 16,384-cell pool (12.4k conversation,
   two follow-ups 3 s apart: both `cache_n` ~12.4k, both generating at once, no deferral: the per-slot sum would
   not fit); F3 the forked text against the same request served alone from the cache; F4 pressure (both tails
   grow until the pair overflows: one is suspended, the other finishes, the suspended one resumes, no error).
   `customware/test-fork-launcher.sh` on the deployed 120k launcher: F2 precompaction (88k conversation, a
   400-token continuation and a summary request 10 s apart, both generating together, no deferral).
4. Release, launcher check, note `customware/012-forking.md`, NEXT.md, dashboard, memory.

## Verification

- F1: two "cache_n" near the conversation length, one "forked from slot" line, interleaved generation, zero
  "waiting for cells" in a pool that could not hold two private copies.
- F2 on the launcher: same at 88k, VRAM unchanged, both tails generating at ~7.5 t/s.
- F4: "suspended" then "resumed", both requests complete, no "Context size has been exceeded".
- All earlier tests still pass: test-cache-hit-real.sh, test-cache-hit-paths.sh, test-keepalive.sh.

## Risks

- A fork point not on a multiple of 4 leaves one indexer block incomplete (treated as always visible): fine for
  the model, not bit-identical to a lone prefill. Reported by F3, not fixed.
- A restore that cannot find a recurrent cell aborts the server (`load_tgt` aborts on size mismatch): the fork
  frees `dst`'s own recurrent cell first and the restore takes it; with `-np` slots there are `-np` cells.
- The 112 MiB checkpoint copy per fork and per turn end (host memcpy / device read, tens of ms).

## Results (2026-10-06)

Build `00ec137ff`. `customware/test-fork.sh fork1`, real model, pool 16,384 cells, `-np 2`:

- **F1 fan-out.** P (12,341 tokens, 48 generated) ends; A and B, both P + the assistant turn + a different
  follow-up, posted 3 s apart. A continued in P's slot (`cache_n` 12337, 79 tokens processed); B became a fork
  of A's busy slot at the turn-end checkpoint (12,394 tokens; common prefix 12,397 of 12,416; 22 tokens to
  process; `cache_n` 12394). Both generated at the same time (8.6 and 9.0 t/s, 7 slot switches in the progress
  lines), zero "waiting for cells" in a pool that could not hold two private copies (2 x 12.4k > 16,384), zero
  restore failures, 8 checkpoints created including the turn-end ones. `/slots` shows `fork_of` and
  `n_fork_shared` 12394 on both slots (the second fork happened in F4, from the other direction).
- **F3 exactness.** The same request as B, served alone afterwards from the slot (`cache_n` 12412): the texts
  agree for the first 130 characters, then diverge (1068 vs 1020 characters). The near-tie flip of a different
  batch composition (entry 007's caveat), not a defect of the fork. B's fork point (12,394) is not a multiple of 4, so the straddling indexer
  block (treated as always visible) was in play, and the text stayed coherent.
- **F4 pressure, first attempt.** 1,500 tokens per tail fitted next to the 12.4k prefix (15.4k < 16,384):
  no overflow, both finished at 7.8 t/s. Re-run with 2,200 per tail below.
- **F4 pressure, second attempt (2,200 tokens per tail), build `00ec137ff`.** The pair overflowed at ~14.4k
  tokens each: the fork (slot 0, owning only its tail) was suspended ("14376 tokens parked in RAM (314.6 MiB)":
  a fork is saved whole, as designed), A2 finished (2,200 tokens at 8.0 t/s), the idle slot was purged to make
  room ("14617 cells of its own"), and the resume **crashed the server**: `state_seq_set_data: error loading
  state: unexpectedly reached end of buffer`, then the indexer assert on the next decode.

  Cause, not in the fork code: `server_prompt_cache::alloc` removes "any cached prompt fully contained in the
  new one"; a pinned entry (a suspended slot's parked state, entry 007) has an empty token list, which is
  contained in every prompt, so the purge's save (added in entry 010) erased it: the log line "removing obsolete
  cached prompt with length 0" is the parked state going away while the slot still pointed at it. The resume
  then read freed memory. Any suspension followed by a save had this; entry 010 made the sequence reachable
  (a purge saves first) and it was live on the launcher since this morning. Fix (`patch-012b.py`): `alloc` and
  `select` skip pinned entries.
- **F4 pressure, third run, build `6f50e7617` (the pinned-entry fix).** Same sequence: overflow at ~14.4k
  tokens each, the fork (slot 0, "0 in this batch") suspended, A2 finished its 2,200 tokens at 7.9 t/s, the
  idle slot purged ("14617 cells of its own") and saved, the fork resumed after 20.7 s with its 14,376 tokens
  and finished its 2,200 at 7.4 t/s. No error, no restore failure, the server alive at the end. F1 again:
  fork, both generating together, zero deferrals.
- **F2 precompaction on the deployed 120k launcher** (`test-fork-launcher.sh fkl1`, release `6f50e7617`).
  M (88,834 tokens, cold at 282 t/s, 16 generated) ends with a turn-end checkpoint at 88,849. M2 (M + the
  assistant turn + "continue", 400 tokens) takes M's slot; C (M + the assistant turn + "summarize", 200 tokens)
  arrives 10 s later while M2 generates: "fork of slot 1 at 88852 tokens (common prefix 88855 of 88874), 22
  tokens to process", `cache_n` 88852. Both generated at the same time (10 slot switches in the progress
  lines): C at 7.45 t/s, M2 at 11.4 t/s before C arrived and ~7.5 t/s next to it (8.8 overall). Zero
  "waiting for cells", zero restore failures, VRAM 15.1 -> 15.6 GB as for any two active slots. The fork point
  88,852 was M2's last-user-message checkpoint, 3 tokens past the eager turn-end one at 88,849: a generation
  stops when the end-of-turn token is sampled and that token is not pushed into the slot, while the echoed
  turn is rendered with `<|im_end|>
<|im_start|>user
` after it. The echo is faithful; the fork takes the
  latest checkpoint at or before the divergence, so it lands on whichever is later.

## Status

Released 2026-10-06 as `6f50e7617` (both launchers). v1 as agreed: sharing lasts while both sequences stay in
the pool; a fork pushed to the prompt cache comes back whole with private cells. What the user's harness can
rely on: a request whose prefix is a busy conversation's state starts as a fork within seconds and runs next
to it; several such requests take turns on the two slots; an idle source is still taken over in place.

Left for later (notes-forking.md "Later"): parking, refcounted checkpoints, range-limited save/restore, forking
from an idle source that diverges before its end, preferring a longer cache entry over a shorter busy prefix.

## Final commit list

Sealed 2026-10-20 into `012-forking.patch`: this entry's own diff (customware/ excluded), applied after the entries before it onto upstream `6e60f3560` (tag `v0.5.0-customware.0`); `customware/seal-entries.sh` rebuilds and checks it from `customware/entry-commits.txt`.

- `44b40fca2` customware: NEXT.md: forking facts and direction, and the dead pre-compaction entries in the prompt cache (notes only)
- `e8cb48ab7` customware: notes on forking (architecture in progress) and on checkpoint policy (parked); fork probe (notes only)
- `1c59b2b4c` customware: fork notes: behaviour under pressure, how the queue admits a small request next to a large one (notes only)
- `00ec137ff` server : forking v1 under --kv-unified: a request that branches from a busy conversation shares its prefix cells
- `6f50e7617` server : the prompt cache never erases a pinned entry through its prompt matching (use-after-free on resume)
- `1699c0159` customware: entry 012 verified at 16k (fan-out, overflow) and at 88k on the deployed launcher; notes closed (notes only)
- `ee3260db5` customware: entry 012 note: the 3-token gap between the turn-end and the fork point is the template's end-of-turn tokens, not the echo (notes only)

What shipped, measured results and deviations from the plan are recorded in the sections above (this entry predates the "Implementation notes" heading).

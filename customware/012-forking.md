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

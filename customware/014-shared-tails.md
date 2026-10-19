# customware/014: a fork pushed to RAM stays shared (tail saves and prefix segments)

Plan `26d-bash-ocean-plaza` (VibeMaster, 2026-10-07, assumed approved per the standing night rule); implementation notes and results below the plan.

## Goal

Today a conversation that shares its first `c` cells with another sequence (a fork of entry 012, or the parent
of one) is saved to RAM whole and comes back with `n` private cells: after one round trip through the prompt
cache an 88k pair needs 176k cells, which the 120k pool does not have, so the pair that generated side by side
before the save cannot after it. After this entry only the private tail `[c, n)` (plus the recurrent state at
`n`) is saved when the prefix `[0, c)` is covered by a live sequence or by a prefix already in RAM; the prefix
itself is kept once in RAM as a *segment* (attention cells only, written at the first tail save of its family).
On restore the prefix re-attaches to a live holder (labels only, no cells) or comes back from the segment, and
the tail is restored behind it. A fork pair survives any number of round trips with the pool holding the
prefix once, and RAM holds it once too. Symmetric: the parent saved while the fork is live is a tail as well.

Expected outcome on the launcher: the compaction-style fork at 88k (entry 012, F2) keeps both sides
generating after either side is purged and restored; the 16k fork test shows a restore of a ~200-token tail
behind a 12.4k prefix with the pool at 12.6k cells, not 25k.

## What the engine gives (traced 2026-10-07)

- `llama_kv_cache::state_write(io, seq_id)` collects the sequence's cells by label into index ranges and
  writes `cell_count`, meta (pos, one seq id) and the K/V rows; `state_read_meta` for one sequence does
  `seq_rm(dest, -1, -1)`, then `find_slot(ubatch, false)` for `cell_count` cells at the saved positions (any
  free cells, no position check against the sequence's other cells), `apply_ubatch`, and asserts only the
  restored cells. `cell_count == 0` is skipped entirely. A failed read removes the whole sequence.
- `llama_memory_hybrid_idx` writes attention, recurrent, indexer; the indexer read adopts the attention
  read's cell layout (`sinfo`), so both must see the same cells. `PARTIAL_ONLY` = recurrent only.
- `llama_memory_recurrent::state_read_meta` does `seq_rm(dest, -1, -1)` then `find_slot` for its one cell: a
  label shared through `seq_cp` is dropped and the restored state becomes the sequence's own (what
  `fork_from` relies on).
- `seq_cp(src, dst, 0, c)` on the hybrid adds labels on the attention and indexer cells and shares the
  recurrent tail cell; `llama_memory_seq_pos_min/max` say whether a sequence holds `[0, c)`.

## Design

### Engine (`src/llama-kv-cache.h/.cpp`, `src/llama-memory-hybrid*.cpp`, `src/llama-context.cpp`, `include/llama.h`)

1. A position window on the attention cache for the next per-sequence state I/O, as a member rather than a
   change to the virtual interface:
   ```cpp
   // llama_kv_cache, public
   void set_state_range(llama_pos p0, llama_pos p1);   // -1, -1 = whole sequence (default)
   // private
   llama_pos state_p0 = -1, state_p1 = -1;
   ```
   `state_write` (seq_id != -1) keeps a cell only when `state_p0 < 0 || (pos >= state_p0 && (state_p1 < 0 ||
   pos < state_p1))`. `state_read_meta` (single sequence) does `seq_rm(dest, state_p0 < 0 ? -1 : state_p0, -1)`
   instead of removing everything, so cells below `p0` (re-attached before the read) stay. Whole-cache I/O
   (`seq_id == -1`) ignores the window.
2. `LLAMA_STATE_SEQ_FLAGS_ATTN_ONLY` (`1u << 8`, customware) in `llama.h`: `llama_memory_hybrid::state_write/
   state_read` and `llama_memory_hybrid_idx::state_read` skip the recurrent part under it (the indexer part
   is written and read as for a whole save). Used for segments.
3. `LLAMA_API bool llama_memory_set_state_range(llama_memory_t mem, llama_pos p0, llama_pos p1)`: sets the
   window on the attention cache (hybrid: `get_mem_attn()`, and `get_mem_idx()` when the memory is a
   `llama_memory_hybrid_idx`), only for caches whose dynamic type is exactly `llama_kv_cache` (the dsa/iswa/
   msa variants have their own state paths and do not honour it); returns false otherwise, and the server then
   saves whole sequences as today. Same `dynamic_cast` shape as `llama_memory_n_cells_owned`.

### Server (`tools/server/server-task.h/.cpp`, `server-context.cpp`)

4. Cache entries (`server_prompt_cache_state`): `llama_pos p0 = 0` (first position in `data.main`; 0 = the
   whole sequence, as today) and `bool segment = false` (a prefix: attention cells `[0, tokens.size())` of its
   tokens, written `ATTN_ONLY`, no recurrent state, no checkpoints; not a conversation). Draft-context data
   stays whole in every case.
5. Prefix holders: the sequences in the pool a tail can re-attach to.
   ```cpp
   struct server_prefix_holder { int id; const server_tokens * tokens; int32_t n_tokens; };
   using server_prefix_holders = std::vector<server_prefix_holder>;
   ```
   `server_context::prefix_holders(exclude)` lists every slot with `prompt.n_tokens() > 0` that is not
   `SUSPENDED`/`WAIT_OTHER`, except `exclude`. A holder covers `[0, c)` of a token list when its lcp with it is
   `>= c`, `n_tokens >= c`, and `llama_memory_seq_pos_min(mem, id) == 0 && seq_pos_max >= c - 1`.
6. Save (`server_slot::prompt_save(cache, holders)` and `suspend()`, both through one
   `server_prompt_cache::save(prompt, holders, id, ctx_tgt, ctx_dft, pinned)`):
   `c = split_point(tokens, holders)` = the longest prefix covered by a holder (`min(lcp, holder.n_tokens)`)
   or by an existing segment (`min(lcp, segment.size)`), 0 when below `LLAMA_TAIL_MIN` tokens (env, default
   4096). If `c > 0` and `llama_memory_set_state_range` is supported: when no segment covers `[0, c)`, write
   one from this sequence's own cells (window `[0, c)`, `ATTN_ONLY`, tokens `clone().keep_first(c)`,
   `segment = true`; through `alloc` with the usual limit/room checks, never through the obsolete-removal);
   then the entry with window `[c, -1)` and `p0 = c`. Else whole, as today. A small RAII `server_state_range`
   sets the window and resets it to `(-1, -1)` on scope exit, around every `get_size/get_data/set_data`.
7. Restore (`server_prompt_cache::restore(entry, holders, ctx_tgt, ctx_dft, id_slot, out)`, used by `load`
   and `load_exact`): `seq_rm(id_slot, -1, -1)`; if `p0 > 0`: a holder covering `[0, p0)` (not `id_slot`) →
   `llama_memory_seq_cp(mem, holder.id, id_slot, 0, p0)`, `out.fork_of = holder.id`, `out.n_shared = p0`;
   else a segment covering `[0, p0)` → `set_data` of its `data.main` with window `[0, -1)` and `ATTN_ONLY`
   (the segment's data is kept, it serves the family); else fail (return false, nothing restored). Then the
   entry's data with window `[p0, -1)`, draft whole. Any failure → `seq_rm(id_slot, -1, -1)`, false.
   `load` consumes the entry as today (`prompt = std::move(...)`, erase); `load_exact` erases the pinned one.
8. Selection: `select` skips pinned entries and segments, and skips a tail entry (`p0 > 0`) with no holder
   and no segment covering its prefix (it cannot be restored; it waits for eviction). `peek(prompt, tokens_new,
   holders)` returns `{ n_tokens, n_cells, holder }` with `n_cells = n_tokens - (holder >= 0 ? p0 : 0)`.
   `alloc`'s two loops skip segments like pinned entries. `evict_oldest` skips a segment while any tail entry
   (pinned or not) is covered by it, so tails go first and their segment last.
9. Server context: `plan_placement`'s cache case uses `n_cells + max(0, n - n_tokens)`; `get_available_slot`'s
   restore path uses `peek(...).n_cells`, never purges the holder to make room, and the LRU slot choice avoids
   the holder when another idle slot qualifies (a restore into the holder's slot would lose the sharing, not
   correctness). `build_sched_input`: a `SUSPENDED` request with a pinned tail (`p0 > 0`) and a live holder
   needs `n - p0 + resume_margin`, else `n + resume_margin`. All `prompt_save` callers pass
   `prefix_holders(&slot)`. Log lines: `saved as a tail: %d of %d tokens behind a prefix held by slot %d` /
   `... behind a %d-token segment in RAM`, `wrote a %d-token prefix segment (%.1f MiB)`, `restored a tail of
   %d tokens behind the %d shared with slot %d; pool %d of %d cells` / `... behind %d cells restored from a
   segment`. `/slots` shows `fork_of`/`n_fork_shared` from the restore.

### Not in this entry

Parking proper (sequence ids beyond recurrent cells); segment sharing for RAM-only families (two whole
entries with a common prefix are not split after the fact); a prefix-only cache hit (a segment has no
recurrent state, so it is never restored for a new conversation by itself); the draft context's state.

## Steps

1. Engine: window members and filters, `ATTN_ONLY`, `llama_memory_set_state_range`. Build `llama`.
2. Server cache: entry fields, holders, `split_point`, `save`, `restore`, `select`/`peek`/`alloc`/
   `evict_oldest` rules. Build `llama-server`.
3. Server context: holders, the save/suspend/restore call sites, `plan_placement`, `get_available_slot`,
   `build_sched_input`, the log lines.
4. Tests: `test-fork.sh` F5 (below), F4 reads the tail lines; the regression set.
5. Release, launcher check, `customware/014-shared-tails.md`, `NEXT.md`, memory, dashboard.

## Verification

- `test-fork.sh` F1 and F4 as before (F4's suspended side is now a tail: `suspended: ... tail`, `resumed`).
- New F5 (pool 16384, `-np 2`): after F1 (A continues P in slot 0, B is a fork in slot 1, both idle), an
  unrelated 12.3k request C arrives (the prose prompt with a different first message): the scheduler purges
  A then B (LRU) → log shows one segment write and two tail saves; C answers. Then B's request again: no live
  holder → `restored ... from a segment`; then A's request: B is live → `restored a tail ... shared with
  slot 1`, pool well under 2 × 12.4k cells (the restore line prints it; pass: `< 16000`). Both answers carry
  `cache_n` ≈ their full prompt (no re-prefill) and match the F1 texts (same-prefix check as F3; a near-tie
  flip is expected at some point).
- `test-kvu.sh` 1-5, `test-cache-hit-real.sh`, `test-sched.sh`, `test-third-request.sh`, `test-keepalive.sh`
  unchanged in outcome.
- Launcher (`verify-launcher.sh`, `test-fork-launcher.sh`): the 88k fork pair still generates side by side;
  no change in VRAM; the launcher log shows no `failed to restore`.

## Risks

- A restore into a window with the prefix re-attached relies on `find_slot(ubatch, false)` not checking the
  sequence's other positions: verified in the code; the asserts after `apply_ubatch` cover only the restored
  cells. A failed read drops the whole sequence (prefix labels too), which the server treats as a miss.
- The segment of a family can be evicted only after every tail that needs it; a cache limit that cannot hold
  the segment plus one tail falls back to whole saves (the entry is skipped like an over-limit one today).
- `LLAMA_TAIL_MIN` keeps short common prefixes (system prompts) out: a tail with a tiny prefix would cost a
  segment write for nothing.

## Execution policy

- Implementer: the planner (Fable), as for entries 012 and 013: the user asked for Fable on the forking work,
  the engine path is small and traced above, and the server part reuses the entry 012/013 code the planner
  wrote. No subagent for the implementation.
- Commits: one for the engine window and flag, one for the server cache and context, one for the tests, the
  note and the release; each with the `Plan:` trailer. Never pushed.
- Review: this entry is part of the review cadence over entries 010-014 that follows tonight (Opus
  reviewer and tribunal, until the second full review converges), per the user's instructions.

## Implementation notes (2026-10-07)

- Engine: commit `88de1d50e`. `llama_kv_cache::set_state_range`, the window in `state_write` (cells outside
  `[state_p0, state_p1)` skipped) and `state_read_meta` (`seq_rm(dest, state_p0, -1)` instead of all),
  `LLAMA_STATE_SEQ_FLAGS_ATTN_ONLY` (`1u << 8`) gating the recurrent part in both hybrid memories,
  `llama_memory_set_state_range` (plain `llama_kv_cache` only, by `typeid`; sets the indexer cache as well).
- Server: commits `de2360b44`, `56b24a07b`. As planned, with these details found on the way:
  - `prefix_holders()` must not test `llama_memory_seq_pos_min`: for a hybrid memory it is the larger of the
    attention and recurrent minima, and the recurrent cell sits at the last position, so it is never 0. The first
    run of F5 showed no tail at all. A slot's tokens start at 0 and are all in the pool; the count comes from
    `seq_pos_max + 1` capped at the token list.
  - A pinned (suspended) tail keeps a clone of the slot's tokens in its entry, so the segment-coverage check in
    `evict_oldest` can see it; `save()` holds the segment it is about to reference (`seg_hold`) across the
    eviction `alloc`/`alloc_pinned` may run, since the new tail is not in the list yet.
  - The segment is written from the saved sequence's own cells (window `[0, c)`, `ATTN_ONLY`), never from the
    holder: nothing of another slot is touched by a save.
  - `restore()` clears the destination first, re-attaches or restores the prefix, then reads the tail with the
    window `[p0, -1)`; the entry's buffers are freed on success and kept on failure (a retry needs them).
  - `get_available_slot`: when an idle slot holds a prefix of the request but the cache holds the request's own
    tail behind that prefix and a live holder exists, the tail is restored into another idle slot next to it
    ("restoring the tail here, next to it") instead of the takeover cutting the sibling's tail. Without another
    idle slot the takeover stands.
- Patches as applied: `patch-014a.py`, `patch-014a2.py` (its hybrid anchor needed the blank line), `patch-014b.py`
  (one raw newline in a log string broke the first build; fixed in place).

## Results (2026-10-07, real model, `test-fork.sh r3-fork`, pool 16384, `-np 2`)

- F1 fan-out as in entry 012: one fork, no waiting for cells, both generating together (8.8 / 9.1 t/s).
- **F5**: U (unrelated 12.3k) purges the idle pair: `wrote a 12397-token prefix segment (174.2 MiB) from slot 1`,
  `saved slot 1 as a tail: 218 of 12615 tokens behind the prefix held by slot 0 (115.6 MiB)`, then
  `saved slot 0 as a tail: 218 of 12615 tokens behind a 12397-token segment in RAM (115.6 MiB)`. B's request
  again: `restored a tail of 218 tokens behind 12397 cells restored from a prefix segment; pool 12615 of 16384`.
  A's request again: `the cache holds 12615 tokens of this conversation behind the 12397 that slot 0 has:
  restoring the tail here, next to it`, `restored a tail of 218 tokens behind the 12397 shared with slot 0; pool
  12833 of 16384 cells` (two private copies would need 25230). Neither re-prefilled: `prompt_n=4` each, both
  at 13.5 t/s. Texts agree with the F1 answers for the first 367 / 221 characters (the batch-composition flip
  of entry 007, as in F3). Counts: 1 segment, 2 tail saves, 1 segment restore, 1 re-attached restore, 0 failures.
- F3: B (fork, next to A) vs C (alone): same for the first 221 chars, as before.
- F4 (A2/B2 with 2200-token tails outgrow the pool): `suspended slot 0 (14373 tokens)` after
  `saved slot 0 as a tail: 1976 of 14373 tokens behind the prefix held by slot 1 (140.3 MiB)` (a whole save was
  ~450 MiB); when B2 finished, the decision purged it (`saved slot 1 as a tail: 2220 of 14617 tokens behind a
  12397-token segment`: the pair's tails plus the prefix, 16593, exceed the pool) and A2 `resumed after 20.8 s`
  from the segment (`restored a parked tail of 1976 tokens behind 12397 cells restored from a prefix segment`).
  Both finish, no collective error.
- A refinement left for later: the decision costs a parked tail as "behind a live holder" and may purge that
  holder in the same pass (F4 above); the resume then needs the whole sequence, and if that does not fit, the
  resume-retry rule covers it (next pass, lower occupancy). A cost that knows a purged holder raises its tails'
  need would predict it.

Regression on the same build (`r3`/`r4`): kvu 2 and 5 (suspend, resume after 95 s, both complete), kvu 4 and
keep-alive E (a prompt larger than the pool refused at once: HTTP 400 after 0.1 s, server idle after), keep-alive
F/Q/R (pings every 5 s, headers after 5.1 s, order kept); scheduler P1 (a priority-5 request displaces the
running one, which resumes after 67-69 s and finishes); cache-hit (the second A and B restore from RAM: `cached
n_tokens = 22221` / `12337`, no prefill); the third request (a 42-token request runs next to the generating 12.3k
one while the second 12.3k waits). Rerun on the review-fixed build (`r5`, commits `46b6c069d`..`9441764e1`): fork
F1/F5/F3/F4 (same counts; F4's victim is now displaced by the decision at a tick, before the overflow, since the
shared prefix is charged: `14993 of 16384 cells committed` where the first build logged 2600), kvu 2 and 5,
scheduler P1, the third request: all pass.

Released as `b3fe31f24` on 2026-10-07 23:50 (both launchers). Launcher check: health in 20 s, log-probs as in every
earlier check, VRAM 15.5-15.9 GiB. The 88k precompaction pair on the 120k launcher (`fkl-014`): the compaction request
forked at 88852 tokens (22 to process), both generated together, no waiting for cells; the pair stays in the 120k pool,
so no tail save happens there (that needs a third conversation to push them out, as F5 does at 16k).

## Final commit list

Sealed 2026-10-20 into `014-shared-tails.patch`: this entry's own diff (customware/ excluded), applied after the entries before it onto upstream `6e60f3560` (tag `v0.5.0-customware.0`); `customware/seal-entries.sh` rebuilds and checks it from `customware/entry-commits.txt`.

- `88de1d50e` llama : a position window for per-sequence state I/O of the attention cache; attention-only state flag
- `de2360b44` server : a fork pushed to RAM stays shared: tail saves behind live holders or prefix segments, re-attached on restore
- `c7c82797d` customware: entry 014 plan and patches; fork test F5 (the pair leaves the pool and comes back shared); the regression suite runs it (notes only)
- `56b24a07b` server : prefix holders are the slots' token lists, not the hybrid memory's position minimum
- `46b6c069d` server : the scheduler charges shared cells, caps a request's cost at the pool, charges a parked prefill its whole prompt; the slot refusals stop counting suspended slots
- `6e708180e` server : infill goes through the scheduler; no tail across a media chunk; the owned-cells comment back in place; the suite header names what the fork stage runs
- `9441764e1` customware: entry 014 implementation notes and results (fork F1/F5/F3/F4, kvu, keep-alive on the real model) (notes only)
- `3d68d48f6` server : a takeover that keeps a prefix shared only between idle conversations is charged it
- `b3fe31f24` server : comments on kv_resident, kv_committed and the restore block say what the scheduler does now; fork test reads UTF-8 and counts displacements
- `078c1a86c` customware: entry 014 results on the review-fixed build and the launcher; order of work marks 014 done and released (notes only)
- `739fa64e0` customware: the report of 2026-10-07 (scheduler finished, entry 014 released, review cadence over entries 010-014 paused after the second full review) (notes only)
- `5c6a27f06` server : a parked request resumes as generating only if it was generating when parked
- `8cb144dc4` customware: the checkpoint-policy item gets its first field evidence (a cancelled prefill leaves cells without a checkpoint) (notes only)
- `fba9e9252` customware: the report records the iteration 4 fix taken on 2026-10-19 (notes only)

This patch also carries the fixes of the review cadence over entries 010-014 (`46b6c069d`, `6e708180e`, `3d68d48f6`, `b3fe31f24`, `5c6a27f06`): they touch the code of 010-014 together and were written on top of 014. The findings, verdicts and what each fix is for: `customware/REPORT-2026-10-07.md`.

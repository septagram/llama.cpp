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

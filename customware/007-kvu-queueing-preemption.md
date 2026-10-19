# customware/007: KV-aware admission, preemption and resume under `--kv-unified`

## Goal

A request that fits the pool on its own never fails because of other requests' KV usage; contention costs latency,
never correctness, and the API consumer sees nothing but a pause. Verified in this tree: `decode()` in
`tools/server/server-context.cpp` purges idle slots one at a time when `llama_decode` reports no KV space, halves the
batch down to 1, then sends "Context size has been exceeded." to every processing slot, releases and wipes them and
aborts the batch (the upstream TODO at that site admits it). Admission checks a prompt only against `slot.n_ctx`,
which under `-kvu` is the whole pool.

## Design

One primitive, "suspend a slot / resume a slot", used by the overflow path now and available to future policies
(API key, explicit priority) through one victim-selection function.

- **Admission (A).** In the `SERVER_TASK_TYPE_COMPLETION` path, after a free slot is found and before
  `launch_slot_with_task`: if `kv_unified`, compute the resident cells of *active* slots (their `prompt.n_tokens()`;
  idle retained slots are purgeable and not counted, and with `--cache-idle-slots` they are cleared at launch anyway)
  and defer the task (`queue_tasks.defer`) when `resident + prompt tokens + reserve > pool`. The reserve is the
  request's `n_predict` when set, else 256. Deferred tasks are already retried when a slot releases, which is also
  when cells free up. FIFO, no queue cap (the client's HTTP timeout is the cap). Log line on deferral.
- **Preemption (B).** In `decode()`, when `llama_decode` returns 1 (no KV space) and no idle slot could be cleared:
  before halving below `n_batch / 4`, call `pick_victim()`: the processing slot with the smallest resident context
  (ties: most recently admitted), excluding slots with children, the slot currently in prompt processing when another
  generating slot exists, and slots already suspended. Suspend it: save its state pinned into the prompt cache's RAM
  pool (`prompt_save` with a `pinned` flag, so `update()` never evicts it), `seq_rm` its cells, drop its tokens from
  the current batch (rebuild the batch without that seq id), set `state = SLOT_STATE_SUSPENDED`, keep its task,
  sampler, `generated_text`, `n_sent_text` and `sampled` untouched. Nothing is sent to its client. Then retry the
  decode with the same `n_batch`. If no victim exists, fall through to today's path (only reachable when a single
  sequence alone does not fit, which admission and the per-slot `n_ctx` logic already prevent).
- **Resume (C).** Whenever a slot releases or a suspended slot could fit (checked at the top of `update_slots`):
  suspended slots first, oldest first, then deferred tasks. Resume = `llama_state_seq_set_data_ext` from the pinned
  entry, erase the entry, `state = SLOT_STATE_GENERATING`; the next batch adds `slot.sampled` at
  `prompt.tokens.pos_next()` and generation continues where it stopped. Works for the hybrid memory because the
  per-sequence state includes the recurrent state (the prompt cache already relies on it).
- **Errors (D).** Unchanged: a prompt larger than the pool is rejected with `ERROR_TYPE_EXCEED_CONTEXT_SIZE` at
  admission; a single sequence outgrowing the pool alone hits the existing per-slot context-shift / stop logic. The
  collective "Context size has been exceeded." becomes unreachable in normal operation.
- **Observability (E).** `/slots`: `"suspended": true/false` per slot; `/metrics` and the metrics task: counts of
  suspended slots and KV-deferred tasks; log lines for defer (task, prompt tokens, resident), preempt (victim,
  cells freed, reason) and resume (slot, wait time).
- **Interplay.** Suspended slots are skipped by the batch loop and by `get_available_slot`; a suspended slot's
  `is_processing()` stays true so its client keeps its stream. Speculative decoding and `n_cmpl > 1` children are
  never chosen as victims. Entry 006's prefill rule is unaffected.

## Steps

1. Slot state `SLOT_STATE_SUSPENDED`, `suspend()` / `resume()` on `server_slot`, pinned entries in
   `server_prompt_cache` (`alloc` with `pinned`, `update()` skips them, `load_exact` restores one and erases it).
2. Admission check in the completion path.
3. Preemption in `decode()`, batch rebuild without the victim's tokens, `pick_victim()`.
4. Resume pass at the top of `update_slots` and on slot release.
5. `/slots`, `/metrics`, log lines.
6. `customware/test-kvu.sh`: `-c 32768 -np 2 -kvu`, scenarios 1-5 of the spec (busy pool + new request queued; two
   active outgrow the pool; idle retention; genuinely too large; hybrid resume equals the uninterrupted output for a
   greedy request).

## Verification

- All five scenarios pass with no error responses except scenario 4's context-size error.
- Scenario 5: the resumed request's text equals an uninterrupted greedy run of the same request.
- The 120k launcher: a 100k-resident generation plus a 50k request queues and completes; the log shows the defer and
  the admission after the first completes.

## Execution policy

- Implementer: the planner (Fable). Commits per step on `master`, `Assisted-by: Claude Fable 5.1`,
  `Plan: 26c-update-saucy-wing`. No review mid-way; stop and report when measured.

## Implementation notes (what actually shipped)

- Shipped as planned: `SLOT_STATE_SUSPENDED`, `server_slot::suspend()` / `resume()`, pinned entries in
  `server_prompt_cache` (`alloc_pinned`, `load_exact`, `erase`, eviction skips pinned), the KV accounting helpers
  (`kv_pool_size`, `kv_need`, `kv_resident`), the admission check in the completion path (deferral with a log
  line), `pick_victim()` + `try_preempt(off)` in the decode failure path (before the collective error, only under
  `--kv-unified`, only when a 1-token batch still finds no cells and no idle slot could be purged),
  `try_resume_suspended()` at the top of `update_slots` (oldest first, purging idle slots to make room),
  `server_batch::remove_slot_tokens()` with the batch-index shift for the other slots, `"suspended"` in `/slots`,
  a cancelled suspended slot drops its parked state and its token list.
- Scenarios on a 32k pool (`customware/test-kvu.sh`, classic hot experts, prompt cache 8 GB):
  1. 22k prompt generating, 12k request arrives: "task 4 needs 12641 cells (prompt 12341) but 22481 of 32768 are
     spoken for: waiting for cells", admitted after the first finished, completed normally, no error.
  3. idle retained 22k slot, a new 22k request: purged and run, no wait, no error.
  4. 44k prompt on the 32k pool: `exceed_context_size_error` immediately, slots idle after.
  2. first attempt did not overflow (both answers ended early); rerun with `ignore_eos` and a 24k pool, below.
- Scenario 2 rerun (24k pool, 12k + 10k prompts, 1500 forced tokens each): the overflow fired as designed, the
  smaller sequence (11217 tokens, 270 MiB) was parked and the other ran to completion. The resume then failed:
  the finished sequence stayed resident as an idle retained slot, and the resume check did not count idle retained
  cells, so the restore ran into a nearly full pool (`state_seq_set_data: failed to restore kv cache`) and the
  request got an error, the exact outcome this entry exists to prevent. Fixes: `kv_idle_retained()` is counted
  and idle slots are purged one by one before a resume; `load_exact` keeps the parked state on failure; a failed
  restore is retried on later passes and only after 8 failures does the client see an error. The restore itself
  does not need contiguous cells (`find_slot(..., false)`), so fragmentation cannot break it.
- Hardening from the mid-prefill question: the victim is always a generating slot (a slot in prompt processing is
  never parked, its prefill continues), suspended slots are excluded from the old collective-kill path, and a
  resumed sequence needs `LLAMA_KV_RESUME_MARGIN` (1024) of headroom so two sequences that cannot coexist do not
  ping-pong through RAM every few hundred tokens. `LLAMA_KV_MARGIN` (256) is the growth margin per active sequence;
  a negative value makes admission optimistic, which is how scenario 5 forces an overflow during a prefill.
- Final accounting: `llama_memory_n_cells_used()` (new customware API, the attention cache's occupancy behind the
  hybrid memory) replaces the bookkeeping for the present; admission adds the rest of prompts in flight, a growth
  margin per active sequence (`LLAMA_KV_MARGIN`, 256) and the suspended sequences; resume uses the occupancy plus the
  rest of prompts in flight plus the resume headroom (`LLAMA_KV_RESUME_MARGIN`, 1024), purging idle retained slots one
  by one, and retries a failed restore only after the occupancy has dropped.
- Results on the fixed build (24k pool unless noted):
  - Scenario 5 (overflow while the second request had just finished its prefill, forced with `LLAMA_KV_MARGIN=-2000`):
    the smaller sequence parked at 12343 tokens (286 MiB), the other finished its 800 tokens, the parked one resumed
    after 83 s and completed, no error.
  - Scenario 2, three runs: parked at 11217 tokens (270 MiB), resumed after 47-51 s, completed 1500 tokens, no error,
    identical 4969-character output every time.
  - Exactness: the same concurrent pair on a 32k pool (no overflow) produced byte-identical output for both the
    parked sequence (4969/4969 characters) and the survivor (5525/5525). A solo run of the same request differs from
    character 31 on: llama.cpp's logits depend on the batch composition (one sequence per batch vs two), which is
    outside what any server can promise; the comparison that isolates the preemption is the concurrent one.
  - Scenarios 1, 3, 4 as before (waited and completed; idle retained purged; context-size error only for the
    44k prompt).

## Final commit list

Sealed 2026-10-20 into `007-kvu-queueing-preemption.patch`: this entry's own diff (customware/ excluded), applied after the entries before it onto upstream `6e60f3560` (tag `v0.5.0-customware.0`); `customware/seal-entries.sh` rebuilds and checks it from `customware/entry-commits.txt`.

- `df79272fd` server : KV-aware admission, preemption and resume under --kv-unified; full prompt batches

Its commit `df79272fd` also carries entry 006's server-side batch cut. Its admission follow-up (a request never waits for cells nobody holds; the reserve capped at `LLAMA_KV_RESERVE`) landed in `f6a59d2e8` together with entry 008's profiling groundwork, and travels in `008-depth-profile.patch`.

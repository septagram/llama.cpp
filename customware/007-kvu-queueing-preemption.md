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

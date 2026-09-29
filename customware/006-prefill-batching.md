# customware/006: prefill batching, concurrency and launcher logs

## Goal

Agentic prompts prefill at the full-batch rate, with or without another request generating, and the launcher keeps
a log of every session. Measured today on the 120k launcher: a plain 22k prompt runs at 277 t/s alone and 251 with
a 1500-token generation in flight; an agentic 20k prompt (82 short turns) runs at 141-199 t/s because the server
cuts it at message boundaries into batches of 14, 496, 572, 954 and 19 tokens between the full 4096-token ones, and
every batch above the CPU threshold streams the whole 65 GB expert set (572 tokens: 9.5 s; 4096 tokens: 13.5 s).
Concurrency costs ~10%: each mixed step runs the generator's 2-token ubatch in decode mode, which restores the parked
experts (0.66 s) that the next prefill batch evicts again (16 restores in the test run).

## Changes

1. **Coarser prompt batches** (`tools/server/server-context.cpp`, prompt loop in `update_slots`): the message-boundary
   cut for a checkpoint (`spans.is_user_start` with `checkpoints.empty()` / `checkpoint_min_step` / `last_user_pos`)
   is honoured only when the batch already holds at least `n_batch / 2` tokens (2048 on the launcher); otherwise the
   loop keeps filling. The end-of-prompt cuts (`4 + n_ubatch` and `4` tokens before the end) stay, they are what the
   rollback checkpoint needs. A checkpoint then lands at the end of a batch instead of exactly at a message start, so
   a prefix mismatch may re-prefill up to a few thousand tokens more than today. A debug line logs every cut with its
   reason, so the batching of any prompt can be read from the log.
2. **No restore inside a prefill** (`src/llama-context.cpp`): `decode()` marks the batch as a prefill when its total
   token count exceeds `hot_threshold`; `process_ubatch` then builds every ubatch of that call without the hot split.
   The generator's token in a mixed step runs on the plain CPU path (~150 ms) instead of decode mode plus a restore
   (~750 ms). Default behaviour, no flag.
3. **Launcher logs** (`customware/release.sh`): both launchers pass `--log-file` with a timestamped path under
   `C:\dev\llama.cpp-customware\logs\`; the console keeps its colours (the file sink writes uncoloured lines).

## Expected

- Agentic prompts alone: 250-280 t/s (from 140-250 depending on message layout).
- With a generator in flight: 1-3% below alone (from ~10%); the generator keeps its one token per prompt batch and
  its stream stays alive; both decode at ~7 t/s each afterwards, unchanged.

## Verification

- `customware/test-concurrent.sh` before/after: phases a-d, plus the per-batch breakdown from the server log
  (`prompt processing, n_tokens` lines) showing no batch under 2048 tokens except the end-of-prompt ones.
- Restore count in the log during phase c: 1 per prompt (after it), not one per batch.
- A prefix-mismatch follow-up (the echo request after the code prompt) still reuses the cache up to the last
  checkpoint.

## Execution policy

- Implementer: the planner (Fable). Commits per change on `master`, `Assisted-by: Claude Fable 5.1`,
  `Plan: 26c-update-retry-little`. No review mid-way; stop and report when 006 and 007 are measured.

## Implementation notes (what actually shipped)

- Measured before (chain24, 120k launcher): plain 22k prompt 277 t/s alone, 251 with a 1500-token generation in
  flight; agentic 20k prompt 141 alone / 199 with a generator, cut into batches of 510, 4096, 3985, 572, 4096, 3002,
  954, 3119, 19 tokens (full batches 270-305 t/s, 572 tokens 60 t/s, 954 tokens 88 t/s); 16 restores of the parked
  experts during the run.
- The batch-level rule "no hot split for any ubatch of a prefill-sized batch" (chain25): restores 16 -> 5, plain
  prompt with a generator 251 -> 267 t/s (alone 288), agentic with a generator 199 -> 213.
- The first cut change (cut at a user message only once the batch holds n_batch/2) still produced 2048-2500-token
  batches: the checkpoint for a batch start is created after that batch's token loop, so the loop always sees a list
  one batch behind and its spacing rule fires at almost every message start. Second change: only the last-user-message
  cut is kept (the checkpoint before the last user message is what regenerations need); checkpoints are still created
  at batch starts with the same spacing, so a prefix mismatch inside a batch re-prefills up to one batch more.
- Launcher logs: `--log-file ../llama.cpp-customware/logs/server-<timestamp>.log` in both launchers; the file is plain
  text, the console keeps its colours.

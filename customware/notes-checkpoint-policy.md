# Checkpoint policy: state of research (parked 2026-10-06)

Parked by the user: the decision points are ambiguous, harnesses differ a lot, and it is not needed for forking.
Take it up only with an architectural discussion first. This file records what is known.

## What a checkpoint is

A host-RAM copy of one sequence's recurrent state (the 36 Gated DeltaNet layers), 112.57 MiB, taken just before a
batch is decoded. It is what lets this model go back to a position: the attention K/V and indexer keys are
append-only and can simply be cut, the recurrent state cannot be rewound. Without a checkpoint at or before the
point where a new prompt diverges from the slot, the slot is reprocessed from zero.

## Where they are made (tools/server/server-context.cpp, prompt loop)

- A: where a request starts adding to the slot. For a normal continuation that is the end of the previous
  generation. Made lazily, by the next request, not when the generation ends.
- B: 4 tokens before the end of the prompt, just before the model starts generating.
- C: one batch (4 + n_ubatch tokens) before the end of the prompt; only when the request's new content is that long.
- D: at the start of the last user-role message (the prompt loop cuts the batch there); only during a prefill that
  crosses it. Tool results are not user-role, so in an agent loop this can be far up the prompt.

Limits: `--ctx-checkpoints` (32 per conversation, 3.6 GiB) and `checkpoint_min_step` (8192). When the list is
full, checkpoints closer than 8192 tokens to the previous kept one are dropped (those of the current request
excepted), then the oldest. A rollback erases every checkpoint past the rollback point.

Result on a real 89k session (51 requests, one agent): 7115, 14512, 17333, 29095, 37613, 47611, 55868, 64723,
73384, 79955, then 22 between 80.9k and 89.4k. Sparse history, dense recent turns.

## What the launcher logs say (all sessions 2026-09-30 .. 2026-10-06)

| kind | made | used for a rollback |
|------|------|---------------------|
| A, previous turn end            |  865 |  3 |
| B, just before the AI turn      | 1032 | 62 |
| C, one batch before prompt end  |  146 | 12 |
| D, other mid-prefill            |  150 |  9 |

- 919 continuing requests: 827 start exactly at the end of the previous generation (no rollback at all), 92 do
  not (median 451 tokens of the slot discarded, maximum 103,476).
- Of the 92: in 53 the new prompt diverges exactly where the previous prompt ended (the AI turn came back dropped
  or rewritten by the client), in 10 the previous request had been cancelled by the client, 29 other.
- The checkpoint used was made by the previous request in 71 of 86 rollbacks, 2-3 requests back in 10, 4 or more
  back in 5. Tokens processed after a rollback: median 983, p90 7,880.
- One compaction missed its rollback by 19 tokens (common prefix 8,420, earliest checkpoint 8,439) and was
  reprocessed from zero.

Cost side: in a saved conversation the checkpoints are two thirds of the bytes (116k tokens: 1,743 MiB of state +
31 x 112.57 MiB = 5,233 MiB), and every save deep-copies them although they are already in host RAM (a save
takes 0.56 s with up to 4 checkpoints, 1.7 s with 20 or more).

## Ideas, none decided

- Retention by recency: dense for the last few requests, sparse beyond; then a lower cap (8-12).
- A flag for kinds C and D.
- A taken eagerly when a generation ends (forking needs this one regardless; see notes-forking.md).
- Refcounted checkpoint data instead of deep copies (also in notes-forking.md).

Open questions: how other harnesses echo the AI turn (the 53 rewritten turns are harness behaviour), what a
rewind or edit workflow needs from old checkpoints, whether the cap should be bytes rather than a count.

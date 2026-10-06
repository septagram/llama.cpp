# Forking: architecture notes (discussed 2026-10-06; v1 implemented the same day as entry 012, see 012-forking.md)

Goal (user): two requests that branch from the same state both get a cache hit, cheaply; subagent fan-out and a
"precompaction" flow (fork at ~100k, the fork writes the summary while the main agent continues). High priority,
after the keep-alive entry. Nothing here is implemented.

## Vocabulary

- **Pool**: under `--kv-unified` there is one K/V buffer for the whole server, not one per slot: 120,064 *cells*
  in the 120k launcher, one VRAM allocation made at startup (1,494.94 MiB attention K/V + 186.87 MiB indexer
  keys). A cell holds one token position: its keys and values for the 12 attention layers. Every cell carries
  labels: its position and the set of sequences it belongs to (a 256-bit set). A conversation is the set of cells
  labelled with its sequence id, anywhere in the buffer. Attention for a sequence reads the buffer through a mask
  that hides the cells of other sequences. `n_ctx` is the number of cells; "waiting for cells" and the
  preemption of entry 007 count them. Without `--kv-unified` the buffer is cut into fixed per-slot partitions.
- **Recurrent state**: 112.57 MiB per sequence (36 Gated DeltaNet layers), in VRAM, rewritten at every token.
- **Checkpoint**: a host-RAM copy of the recurrent state at one position (notes-checkpoint-policy.md).
- **Prompt cache**: host-RAM copies of whole sequences that are not in the pool (K/V read back from VRAM + the
  recurrent state + the checkpoint list).

Nothing per-conversation lives in host RAM during inference. The CPU computes only the expert matmuls: a pure
function of the current hidden vector, which travels GPU -> CPU -> GPU once per layer. The expert weights and
the n-gram table in RAM are weights, shared by every conversation.

## Today (traced in the code, 2026-10-06)

Only the first fork hits. A busy slot is invisible to the slot picker; a cache entry is consumed by its load; a
saved descendant deletes its cached prefix as obsolete; an idle sibling's slot is taken over in place, unsaved
when the new prompt keeps at least half of it. Short fork: the second fork prefills everything. Long fork at
~100k: strictly serialized, the main agent stalls while the compaction fork generates. Cold fork: works only when
the divergence sits at a checkpoint of the first request.

## What the engine already does

- `llama_memory_seq_cp(src, dst, p0, p1)` in a unified pool adds the label `dst` to the cells of `src` in the
  range: no data copied, no cells allocated. Removing a sequence removes its label; a cell is free when it has
  none. No owner, no parent pointer.
- The recurrent memory shares a state between two sequences the same way and separates them at the next decode
  (the sequence being decoded gets a fresh cell initialised from the shared one).
- A checkpoint is sequence-agnostic: it can be restored into any sequence id, and the restore first detaches that
  sequence from whatever state it shared.
- Upstream uses all of this for `n > 1` (parent and child slots, `copy_state_to`), only for identical prompts in
  one request. customware/test-fork-probe.sh runs that path on this model in a pool too small for two copies.
- The indexer's block tables already key a block on (set of sequences, position bucket), so shared cells are
  anticipated. A fork point that is not a multiple of the block size (4) leaves one straddling block incomplete;
  incomplete blocks are treated as always visible. Fine for the model, not bit-identical to the same prompt
  prefilled alone. To be measured.

## Probe (customware/test-fork-probe.sh forkp1, 2026-10-06)

A request with `n = 2`, a 12,340-token prompt, a pool of 16,384 cells (too small for the prompt twice), real
model, this build with hot experts and the sparse attention path. Result: both sequences finish at 12,387 tokens
without an error or an assert, so the prefix cells were shared; the two greedy texts are identical to each other;
both generate at 10.6 t/s at the same time. Against the same prompt decoded alone the text agrees for 61 of 235
characters and then takes another near-tie, the batch-composition effect already known from entry 007. The
10.6 t/s is a best case: identical continuations route to the same experts, so the second token adds no memory
traffic; two different continuations should land near the 7.6 t/s each measured for two agents.

## Decisions so far

1. Fork in the pool by sharing cells, not by copying state. A fork is then an ordinary slot: scheduled like any
   other, and parent and fork fit together whenever shared prefix + both tails fit (100k + two tails in 120k).
   Running them at the same time is not a separate feature.
2. No always-mirrored RAM copy of live conversations (the user's 2a): it spends scarce RAM to save a transfer
   that measures 0.2 s up and 0.5 s down for 1.5 GiB.
3. Fork from wherever the source is (2b), reduced to one rule: get the prefix into the pool once, then share it.
   Source live or idle in the pool: share. Source only in the prompt cache: load it as today, later forks share
   from the first one's cells. Cache entries stay single-use.
4. Shared chunks in host RAM (prefixes stored once across cache entries) are deferred: forks are short-lived and
   mostly stay in the pool.
5. Checkpoint policy is out of scope, except: a checkpoint is taken when a generation ends, because a fork needs
   the recurrent state at that point and today it only appears when the next request starts.

## v1 semantics (proposed)

Sharing lasts while both sequences stay in the pool. A fork that is saved to the prompt cache is saved whole and
comes back with private cells: correct, but from then on it is an independent conversation and takes turns like
one. Known consequences: with two slots, parent and fork occupy both, and a third conversation pushes one of them
through RAM; a preemption (entry 007) of one of two sharing sequences frees only its tail, and its resume needs
private cells for all of it, so it waits until the other leaves. The harness should fork with headroom.

The child starts with one checkpoint, at the fork point.

## Under pressure (read in the code, 2026-10-06; v1 accepted by the user)

The shared prefix is reclaimed only when every sequence labelled on it is gone. Two cases:

- The pair overflows the pool by itself (both generating, no idle slot): entry 007 suspends the smallest
  generating sequence. That frees its tail only; the survivor is now alone on the prefix and continues until its
  own context limit. No cascade. The suspended one comes back with private cells, so after the survivor's turn.
  A suspended sequence keeps its slot.
- An outsider needs more cells than a tail: both members of the pair must leave. Today every idle slot is saved
  and cleared at each launch, so this happens by itself for idle ones; what has to change is the arithmetic
  (evicting one of a pair frees its tail, not its token count) and the victim choice, which compares token
  counts and should compare what a suspension would free.

## How the queue behaves today (relevant to forks taking turns)

Admission asks two things: is a slot free, and does the request fit next to what the active and suspended
sequences are committed to. It does not look at who else is waiting. A new request is handled at once; the
deferred queue is only retried when a slot is released, one request per release (one that asked for that slot,
else the first in the queue), and a request that still does not fit goes to the back.

So a small request slips in next to a large active one while another large one waits. Measured
(customware/test-third-request.sh, pool 24576): B (12.3k) queued behind A (12.3k, generating) at +84 s; C
(42 tokens) posted at +86 s started at once and answered at +96 s at 7.7 t/s; B was retried when C finished
(still waiting), started when A finished at +111 s and answered at +177 s.

Weak spot: one retry per release. A small request that was itself queued (both slots busy when it arrived) sits
behind a large one that still does not fit until the next release. Retrying the whole queue on each release
fixes it; planned with the keep-alive entry, which touches the same code. There is no preemption for fairness:
a queued request never interrupts a running generation, only a real overflow suspends one.

## Work items for the entry

- The fork step in slot selection: when the best prefix of a new prompt is in another slot (busy or idle), take a
  free slot, share [0, lcp), restore the fork-point checkpoint into it, process only the suffix.
- Siblings are never taken over in place.
- Cell accounting that understands shared cells: admission, the idle-retained subtraction, preemption, the purge.
- The eager turn-end checkpoint.
- Tests: fan-out at 20k, precompaction at ~100k with both generating, the indexer block boundary.

## Later

- Parking: an idle conversation keeps its cells but gives up its slot and live recurrent state (its turn-end
  checkpoint stands in); resume is one 112 MiB upload. Replaces save-and-clear at every launch. Needs sequence
  ids beyond the slot count (cells allow 256; unverified for the recurrent memory).
- Refcounted checkpoint data (no deep copy on save, shared between forks).
- Range-limited save and restore, so a saved fork keeps or regains its shared prefix; then shared chunks in RAM.

## Slots

`-np` is the number of conversations active at once; each has a live recurrent state (112.57 MiB of VRAM). It
stays 2. The number of forks alive is a different thing and, with parking, not bounded by it.

## Handoff (2026-10-06): for whoever plans and implements v1 (done: entry 012, same day; kept as the record of what was known before)

The user will plan and implement forking with Fable. State at handoff: nothing of forking is implemented; the
design above was agreed in discussion; entries 010 and 011 are released (`17138311b`).

Verified by running:
- two sequences sharing a prefix decode correctly on this build and model (`test-fork-probe.sh`, upstream's
  `n = 2` path): shared cells, identical greedy texts, no assert.
- the queue and cache behaviour that forks will ride on: `test-cache-hit-real.sh`, `test-cache-hit-paths.sh`,
  `test-cache-hit-launcher.sh`, `test-third-request.sh`, `test-keepalive.sh`.

Read in the code, not run:
- forking from a parent that has moved past the fork point: `llama_memory_seq_cp(src, dst, 0, N)` followed by a
  PARTIAL_ONLY restore of the checkpoint at N into `dst`. The recurrent `seq_cp` ignores the range and shares the
  parent's current state; the restore first detaches `dst` from it (`state_read_meta` calls `seq_rm(dst)`), so the
  order copy-then-restore is the safe one. With two slots the recurrent memory has exactly two cells.
- the indexer block boundary when the fork point is not a multiple of 4 (`set_input_qsa`, incomplete groups go
  to the always-visible block).
- more sequence ids than slots (needed for parking, not for v1).

Where the server decides things today (all in `tools/server/server-context.cpp`): `get_available_slot` (slot
choice, cache save/restore, the entry 010 fit check), the admission block in the `SERVER_TASK_TYPE_COMPLETION`
case, the idle-slot save-and-clear right after a launch ([TAG_IDLE_SLOT_CLEAR]), the decode failure block
(`try_clear_idle_slots`, `try_preempt`), `try_resume_suspended`, and the checkpoint logic in the prompt loop.
Upstream's fork mechanics to reuse: `copy_state_to`, `launch_slots_with_parent_task`, `SLOT_STATE_WAIT_OTHER`.
Cell accounting helpers: `kv_cells_used` (real, counts a shared cell once), `kv_resident` and
`kv_idle_retained` (per-slot token sums: these double count shared cells), `kv_committed`.

Suggested first steps: (1) a probe for the unverified fork-past-the-fork-point path, before any server logic;
(2) the eager turn-end checkpoint; (3) the fork step in `get_available_slot` with a test at 20k (fan-out) and at
~100k in the 120k launcher (parent and fork generating together); (4) the accounting.

Working rules that cost time this week (also in the memory notes): prove a build by its log, the exe timestamp
and a new string in the exe; never `taskkill /IM llama-server.exe`, use `customware/kill-port.sh`; check port
5000 before using the GPU; in commands sent through the shell tool a doubled backslash arrives single, so write
C escapes into patches with `chr(92)` or the edit tool.

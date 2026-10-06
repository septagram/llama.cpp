# Forking: architecture notes (discussion in progress, 2026-10-06)

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

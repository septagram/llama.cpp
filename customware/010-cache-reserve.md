# customware/010: the prompt cache yields to the model's page cache

## Goal

The server must coexist with the Ordis VM (16 GB, permanent: it is the OpenClaw client that will use the server)
and with whatever else runs on the box. Last night showed the failure mode: the mmap'd experts live in the page
cache, the OS counts that memory as available and reclaims it first, so a prompt cache that grows to its byte
limit (16 GB in the launchers) pushes the experts out silently and prefill drops to disk speed (21 t/s, a 22k
prompt "hangs" for half an hour). A fixed lower `--cache-ram` would only move the cliff; the cache has to give
way to the RAM that is actually there.

## Design

- `LLAMA_CACHE_RAM_RESERVE_MIB` (default 0 = off): physical RAM that must stay *available* (free plus file
  cache, `GlobalMemoryStatusEx().ullAvailPhys` on Windows, `MemAvailable` on Linux) after a save.
- `server_prompt_cache::make_room(bytes_new)`: while `available < reserve + bytes_new`, evict the oldest
  unpinned entry and re-read the number (the freed vectors go back to the OS at once). A regular save that still
  does not fit is skipped with a warning; a suspended slot (entry 007) is parked regardless, older prompts first.
- `update()` calls `make_room(0)`: outside pressure (the VM growing, other agents) shrinks the cache too.
- The launchers set 76000 MiB: the CPU-side tensors are 105 GB, of which the n-gram table's untouched rows do
  not need residency; prefill and decode touch about 75 GB. One log line at start says the guard is on; each
  eviction or skipped save logs the numbers.

## Steps

1. Header, helper, `make_room`, the three call sites, the startup line, the launcher env and comment.
2. `customware/test-cache-reserve.sh`: on port 5011 with the reserve set just under what is available, two
   prompts: the first save must evict or be skipped with the log line; with a small reserve both saves go in.
3. Release, verify the 120k launcher (health, the guard line in its log), notes, memory, dashboard.

## Verification

- The log shows the guard at start and its decisions when RAM is short; `/slots` and the cache-state trace show
  the cache shrinking instead of the experts leaving.
- No change in behaviour when the variable is unset (upstream semantics kept).

## Implementation notes

- `tools/server/server-task.h/.cpp`: `reserve_bytes` read from the environment in the constructor,
  `server_available_ram_bytes()` (Windows `GlobalMemoryStatusEx`, Linux `/proc/meminfo MemAvailable`, else 0 =
  guard off), `make_room()` called in `alloc()` (skip the save when it cannot fit), `alloc_pinned()` (park anyway,
  older prompts first) and `update()` (outside pressure). `tools/server/server-context.cpp` logs the reserve at
  start. `customware/release.sh`: both launchers export `LLAMA_CACHE_RAM_RESERVE_MIB=76000`.
- Why "available" and not "free": the OS keeps the model's pages in the file cache and reports them as available
  either way; what the guard defends is the *sum* of free and cached memory, which is exactly the room the experts
  need. When private memory grows anywhere on the box (the VM, an agent, the prompt cache itself), that sum
  shrinks below the reserve and the cache gives way.
- Why not derive the reserve from the model: the CPU-side tensors are 105 GB but the n-gram table's rows are
  touched sparsely; 76 GB is what a long session actually keeps warm. Tune it per model in the launcher.

## The cache miss the user caught (2026-09-30 morning)

OpenCode at 98,770 tokens in slot 1, OpenClaw's 69,964-token prompt arrives: slot 1 is saved to the cache and
cleared, slot 0 prefills the 70k. OpenCode's continuation (98,998 tokens) arrives during that prefill: the slot
picker restores the cached 98,770-token state into slot 1 *before* anything is freed, `state_read_meta` cannot
find 98,770 cells (70k are in use), the task is deferred (right). When the 70k finishes the deferred task is
retried, the picker restores again, and slot 0's 69,964 idle cells are still there: the restore fails once more
and the continuation starts from zero, a five-minute prefill with its state sitting in RAM. The launch-time step
that saves and clears idle slots under `--kv-unified` runs *after* the picker's restore, which is the wrong
order when the pool is shared. A second, smaller leak: `try_clear_idle_slots` (the decode-time purge) dropped
prompts without saving them.

Fix: `server_prompt_cache::peek()` says how many tokens the entry `load()` would pick has; the picker, under
`--kv-unified`, frees other idle slots (saved first, as the launch would) until that many cells are free, and if
they still are not (a processing slot holds them) it does not attempt the restore: the entry stays for the retry
that follows the admission deferral. The purge saves before it clears. Test: `customware/test-cache-hit-real.sh`
(a 24k pool, the 22k and 12k prompts requested A, B, A, B; the second A and B must come back from the cache).

Reproduced on the released binary (ea2d1ce84) with `test-cache-hit-real.sh`: A 22,225 tokens prefilled in
136 s, B 12,341 in 67 s, then A again "found better prompt ... failed to find 22240 available cells" and
re-prefilled in 119 s, B again the same, 4 restore failures in all: the exact shape of the user's log. (The first
"new build" run of that test exercised the same old code: the build had failed on a missing `<psapi.h>` and the
exe on disk was the previous one. The build check now looks for the log's FAILED line, the exe's timestamp and
a string that only the fix contains.)

## The guard's measure on Windows

`GlobalMemoryStatusEx().ullAvailPhys` counts the standby list (file cache) as available but not the file pages
mapped into a process's working set, and after a prefill the experts sit in the server's own working set: the
first test run showed 12.3 GiB "available" on a box with the whole model warm. The server's mapped (non-private)
working set (`K32GetProcessMemoryInfo`: WorkingSetSize minus PrivateUsage) is added back; Linux's MemAvailable
already counts mapped page cache. The "box below the reserve" warning from `update()`'s probe is logged once per
episode, a refused save every time.

## Verification log (2026-09-30 afternoon, GPU freed by the user)

- `test-cache-hit-real.sh hitr-new2` on the fixed build: A 22,225 tokens prefilled in 138 s, B 12,341 in 167 s,
  then **A again in 2 s** (`cache_n` 22221, prompt 294 ms) and **B again in 2 s** (`cache_n` 12337), zero
  restore failures; the log reads "freeing idle slot (12356 tokens) so the cached prompt of 22240 tokens fits"
  and the reverse. The released binary on the same script: 4 restore failures, A and B re-prefilled (119 s, 66 s).
- `test-cache-reserve.sh cr2` with the corrected measure: available RAM reads 70-76.5 GiB throughout (the first
  version dipped to 12.3 GiB after a prefill); with an unattainable reserve every save is refused with the numbers,
  with a 45 GiB reserve both prompts are saved. Nothing changes when the variable is unset.
- The launchers ship `LLAMA_CACHE_RAM_RESERVE_MIB=70000`, not 76000: the guard's "available" sat at 70-76 GiB on
  this box with Ordis and the day's agents running, and a cache that pushes a few GB of experts out costs seconds
  of streaming on the next prefill, while the 16 GB cliff cost minutes. 76000 would have refused nearly every save.

## Status at deployment (2026-10-06)

The cache-hit fix sat uncommitted and unreleased from 2026-09-30 14:30 (the session ran out of tokens between the
passing test and the commit) until 2026-10-06; the user kept running ea2d1ce84 and hit the bug 22 times in three
sessions (`failed to find N available cells` in the launcher logs of 09-30, 10-05 and 10-05/06, always the same
two steps: a restore attempted while the other slot was processing, then a second one on the retry with that slot
idle but not yet cleared, then a prefill from zero).

Released with the RAM guard **switched off** in both launchers (`LLAMA_CACHE_RAM_RESERVE_MIB` not exported; the
code is inert without it). Reason found while taking inventory: under `--kv-unified` an idle slot is cleared
after its save whether or not the save succeeded ([TAG_IDLE_SLOT_CLEAR]), so a save the guard refuses loses that
conversation's state, i.e. the guard can cause exactly the re-prefill this entry exists to remove. On this box the
guard's "available" reads 70-76 GiB and a 99k-token save is 4 GiB, so a 70000 MiB reserve would refuse some of
them. Before it is switched on it needs one of: never refuse the save of a live slot's prompt (evict older
entries only), or keep the slot's cells when the save is refused.

Known gap, not fixed: the cache picks an entry when at least a quarter of it is a prefix of the new prompt, and a
restore needs cells for the whole entry. After a compaction (a 99k entry, a 35k new prompt sharing its first
30k) the new prompt can pass admission while the entry does not fit next to a busy slot; the restore is then
skipped and the 35k are prefilled from zero. Deferring the task until the entry fits would close it, at the
price of waiting for the other slot.

## What the user's logs showed (read on 2026-10-06), and the refinement

The launcher log of 2026-10-05 21:25 (12.4 hours, OpenCode and OpenClaw at ~93k tokens each on the 120k pool)
has three shapes of the same bug, not one:

1. `failed to find N available cells` while the other slot is processing: the restore is attempted, fails, the
   task is deferred (16 failures with shape 2).
2. the same on the retry, with the other slot idle but its cells not cleared yet: prefill from zero.
3. no error at all: the returning task arrives a few seconds into the other slot's prefill, the pool still has
   room, the restore **succeeds** and the cache entry is consumed; the task is deferred (the other prompt is
   committed); the other prefill reaches the point where both do not fit, `decode` fails to find cells, and
   `try_clear_idle_slots` purges the restored slot without saving it. 30 purges of 80-94k tokens.

Cost in that session: 890,532 tokens prefilled again (about 65 minutes), and of 70 queued requests 28 were
cancelled by the client after 300 s (its timeout), most of them waiting behind a prefill that should have been a
restore. Two ~93k conversations cannot be active together in 120k cells, so they alternate; each swap is a
4 GiB save and a 4 GiB restore, seconds when it works.

d8a68dfff handles 1 and 2 directly and rescues 3 through the purge's save. The refinement removes the round trip
of 3: the picker restores only when the entry fits next to what the active sequences are *committed* to
(`kv_committed`: their whole prompts plus margins, suspended slots included), not the cells free at that
instant; otherwise "does not fit next to the N cells the active sequences need: kept for later" and the task
queues with its state still in the cache.

Verification of d8a68dfff on the deployed 120k launcher (`test-cache-hit-launcher.sh hitl1`): A (88,834 tokens)
and B (44,432) fill the cache; A2 comes back with `cache_n` 88830 and generates 400 tokens in 37 s wall; B2
arrives while A2 generates, is "kept for later" and deferred, then restored with `cache_n` 44428 after A2;
zero restore failures. Prefill on the way: 248.7 and 305.2 t/s; decode at 89k depth 11.2 t/s.

Not addressed: a queued request that waits longer than the client's timeout is still cancelled by the client
(a turn of 3000 generated tokens is five minutes on its own). Raising the clients' request timeout is the
immediate remedy; keep-alive bytes on a queued streaming response would be the server-side one.

## Out of scope

Deriving the reserve from the model automatically (the table's touched rows are not knowable up front); a
`--cache-ram` CLI flag for it (env, like the other customware knobs).

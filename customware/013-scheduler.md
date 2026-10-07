# customware/013: one scheduler instead of five rules

## Goal

The decisions about the unified KV pool (who starts, who waits, who is suspended, which idle conversation is
purged) come from one pure function over one ordered list, with a reconcile step that drives the existing
mechanisms (launch, fork, suspend, resume, save, purge). It replaces the five places that decide today
(`get_available_slot`'s fit check, the admission block, the idle-slot save-and-clear at launch, the decode
failure path, the resume pass) and brings what they could not: explicit priority with preemption across
classes, a starvation guard, running requests sticky among equals, idle conversations kept in their slots
until their cells or slot are needed. Design: `customware/notes-scheduler.md` (the user's model, discussed
2026-10-06).

## The decision (`tools/server/server-sched.h/.cpp`, no llama dependency)

Input: pool size, slot count, now; the requests (running with the cells they own alone and their growth
reserve; suspended with what a private restore needs; waiting with what their placement needs and the idle
conversation they would continue); the idle conversations (cells owned, last use). Order: effective priority
(explicit, plus one class after `t_age` of waiting) descending, then running before suspended before waiting,
then arrival. Walk: keep what fits in cells and slots; a running request is displaced only when the cells it
holds do not fit next to higher-ranked ones (its reserve is charged either way, so nothing lower starts into
it); suspended requests keep their slot; a waiting request takes a free slot or an idle conversation's. After
the walk: purge idle conversations least-recent-first until the kept set fits in cells and slots, never the
one a kept request continues. Output: keep, start, suspend, purge, and the overflow victim (lowest class,
fewest held cells, youngest). `tests/test-server-sched.cpp` covers the shapes from the logs: the small request
next to a large one, a fork pair, class preemption, sticky equals, aging, LRU purges, slots, suspended first,
the victim, the reserve.

## The server (`tools/server/server-context.cpp`, `server-queue.cpp`, `server-task.*`)

1. `plan_placement(task)`: a pure version of `get_available_slot`'s choice: continue/take over an idle slot
   (need = suffix + reserve, reuse = that slot), fork from a busy one (need = suffix past the fork point), a
   cache entry (need = entry + suffix), cold (need = prompt + reserve). `get_available_slot` keeps executing.
2. `schedule(reason)`: builds the input (running slots, suspended slots, the deferred queue, idle slots),
   decides, reconciles: suspensions first (`try_preempt` on the named requests, now allowed for prompt-
   processing slots too: on resume they return to `SLOT_STATE_STARTED` and the prompt loop continues), purges
   (`purge_idle`: save then clear), resumes (`resume()`), then the waiting starts are pulled from the deferred
   queue (`pop_deferred_tasks(ids)`) and come back through `process_single_task`, where the decision is
   confirmed and the task launched through `get_available_slot`. Runs on: a completion task arriving, a slot
   release, a decode that found no cells (emergency: purge, else the victim), and in `update_slots` when
   something changed or every 2 s (aging).
3. Removed under `--kv-unified`: the admission block, the idle-slot save-and-clear at launch,
   `try_resume_suspended`, the whole-queue re-post on release. Without `--kv-unified` nothing changes.
4. `priority` request field (int, default 0; JSON), `t_arrival` on the task; `LLAMA_SCHED_AGE_S` (default
   300) for the aging; log lines per decision that changes something; `/slots` shows priority.
5. Launchers: a third slot (one 112 MiB recurrent cell) so a higher class can start while two run; VRAM
   measured in the launcher check, the hot set trimmed if needed.

## Not in this entry

Parking (idle conversations without a slot; needs sequence ids beyond recurrent cells), the slot/request
split (a suspended request releasing its slot), priority from API keys, a replay harness (the unit test
cases stand in for it: they are the logged shapes).

## Verification

- `test-server-sched` passes.
- Every earlier scenario still passes: `test-kvu.sh` (scenarios 1-5, exactness), `test-cache-hit-real.sh`,
  `test-cache-hit-paths.sh`, `test-third-request.sh`, `test-keepalive.sh`, `test-fork.sh` F1/F4.
- New, `test-sched.sh`: P1 priority preemption (a prio-5 request arriving with no room suspends the running
  prio-0 one and starts at once; the suspended one resumes after); P2 idle conversations stay in their slots
  across a launch (no save at launch, a continuation hits without a restore); P3 aging (a waiter promoted
  after `LLAMA_SCHED_AGE_S` displaces an equal); P4 emergency overflow still purges, then suspends the victim.
- On the deployed launcher: the two-agent pattern from the logs (alternating large conversations, a small
  third request slips in), VRAM with the third slot.

## Risks

- A request started by the decision but refused by `get_available_slot` (no slot to fork into, a restore that
  does not fit) is deferred again; the next decision runs on the next event, not in a loop.
- Suspending a prompt-processing slot is new: its in-flight batch tokens are removed as for a generating one
  and its processed prefix is saved; the resume path through `SLOT_STATE_STARTED` re-sends the begin marker
  (a ping after entry 011).
- The reserve per running request is one margin re-decided every pass; the emergency path stays as the net.

## Implementation notes (2026-10-07)

- `tools/server/server-sched.{h,cpp}`: `sched_decide`. The walk charges a running request `held + need`
  (its cells and its growth reserve) but displaces it only when `held` alone does not fit next to the
  higher-ranked ones; so a waiter of a lower rank never starts into a running request's reserve, and a
  running request is never suspended for lack of reserve. Slots: suspended requests count as holding one
  whether or not they resume now; idle conversations do not count, they are purged or taken over. Purges
  are settled after the walk, least recent first, only as far as the kept set needs, never the conversation a
  kept request continues. `tests/test-server-sched.cpp` builds without the model
  (`customware/build-target.cmd test-server-sched`).
- `server-context.cpp`: `plan_placement` mirrors `get_available_slot`'s choice without side effects (idle
  prefix: suffix + reserve and the slot as `reuse`; busy prefix with a checkpoint: suffix past the fork point;
  a cache entry: `max(entry, prompt)` + reserve; cold: prompt + reserve). `schedule()` builds the input from
  the slots and the deferred queue (plus the arriving task), decides, and reconciles in order: suspensions
  (FULL only: between decodes every token of a slot is decoded, so `suspend()` saves the state as it stands),
  purges, resumes (with entry 007's retry rule), then the waiting starts are pulled from the deferred queue
  and come back through `process_single_task`, where the decision is confirmed before `get_available_slot`
  executes the placement. Modes: FULL (task arrival, the 2 s tick), LIGHT (the release callback: it runs
  inside the slot loop, so it only pulls), EMERGENCY (a decode found no cells: purge what the decision
  purges; the victim, a displaced request if any else the lowest-ranked running one, goes through the
  batch-aware `try_preempt`).
- Gone under `--kv-unified`: the admission block, the idle-slot save-and-clear at launch (idle conversations
  keep their cells until purged), `try_resume_suspended`, the whole-queue re-post on release.
- Suspending a slot that is processing its prompt: its in-flight batch tokens are removed as for a generating
  one (`try_preempt`) or none are in flight (FULL); `resume()` sends it back to `SLOT_STATE_STARTED`, where the
  prompt loop finds the processed prefix and continues (the begin marker is sent again: a ping after 011).
- `priority` (request JSON, default 0), `t_arrival` set when the scheduler first sees a task,
  `LLAMA_SCHED_AGE_S` (300). The launchers go to `-np 3`.

## Results

`customware/test-sched.sh sched1` on `d39d22e07` (before the cap and the kept-cells fix), real model, pool
24,576, three slots, `--sse-ping-interval 5`, `LLAMA_SCHED_AGE_S=20`:

- **P1 priority preemption.** A (12.3k prompt, 600 tokens) generating; H (12.3k, other tag, priority 5)
  arrives: "scheduler (task): keep 0, start 1, suspend 1", A is "displaced by a higher priority request:
  suspending (12443 tokens)" (287 MiB parked), H prefills and generates alone at 13.2 t/s and answers at
  +162 s; on H's release the decision purges H's idle conversation for A's resume ("resumed after 66.9 s"),
  A finishes its 600 tokens at +202 s. No error.
- **P2 idle conversations stay.** A 42-token request launches with zero "saving idle slot" lines (before:
  every idle slot was saved and cleared at each launch). It took over A's conversation by a 10-token common
  prefix, as the slot picker does, so A's continuation came back through one cache restore (`cache_n`
  12337). The decision had charged the takeover with all 12,940 cells of A's conversation and purged H's for
  nothing: fixed in `3511e93a0` (a takeover keeps the common prefix, the rest is freed).
- **P3 aging.** B generating; W (priority 0) cannot fit: "waiting for cells" with the decision's numbers;
  20 s later "scheduler (tick): ... suspend 1", B displaced, W answers at 12.6 t/s, B resumes and finishes.
- **P4** ran the shared pair (2,200-token tails) to completion together at 7.8 t/s each, but with this
  script's 24k pool the pair does not overflow; the overflow path under the scheduler is covered by
  `test-fork.sh` F4 (16k pool) in the regression suite.

`customware/test-sched.sh sched2` on `3511e93a0` (the cap and the kept-cells fix): the same four scenarios
pass (H at 13.5 t/s alone while A is parked, A back and done at +190 s; zero saves at launch; the aged waiter at
13.5 t/s; the shared pair at 7.8 t/s each).

**Three generations at once** (`customware/test-np3.sh np31`, `LLAMA_SCHED_MAX_ACTIVE=3`, three 12.3k
conversations, 300-token continuations from the cache):

| generating at once | each | together |
|---|---|---|
| one | 12.8 t/s | 12.8 |
| two | 8.8 t/s | 17.6 (x1.37) |
| three | 6.6 t/s | 19.7 (x1.54) |

A third concurrent generation adds 12% of total throughput and costs every slot 25%; the user does not want it
by default, so `LLAMA_SCHED_MAX_ACTIVE` stays 2 and the third slot serves priority starts, suspended requests
and idle conversations that keep their cells. The projection from the two-way cost model (5.6 each) was a
little pessimistic: the third token adds ~43 ms to a 114 ms step here, not 46.

## Follow-ups from the regression suite (2026-10-07, commit `e49f3c06a`)

The night's `test-regress.sh r1` (build of `3511e93a0`): kvu scenarios 1, 2, 3, 5 passed (scenario 2: the
displaced sequence suspended, resumed after 104.6 s, both complete; the "identical prefix" check stays weak, as
in entry 007: logits depend on the batch composition), cache-hit passed, keep-alive F passed. Three gaps:

- **A prompt the pool cannot hold even alone waited forever** (kvu scenario 4, keep-alive E: a 44k prompt on a
  32k / 24k pool). The decision never starts it, the deferral never ends, the client gives up. Now refused at
  once in `process_single_task` with the context-exceeded error, as the launch would refuse it
  (`task.n_tokens() >= kv_pool_size()`).
- **A request whose growth reserve exceeds the pool never fit** although its prompt does. Its need is charged
  at most the pool in `build_sched_input`, so it runs alone, as entry 007's admission let it.
- **An overflow victim that could never come back was parked forever**: `try_preempt` suspended whatever the
  scheduler named, and a sequence whose own cells plus the resume margin exceed the pool never resumes. Such a
  victim now fails alone with the context error, its conversation goes to the cache like any purged one, and the
  others go on with its cells.

The suite itself was stopped by the harness at its 2-hour limit while the keep-alive stage hung on scenario E;
its child script kept running into the next morning (see `check-box.sh`'s leftover-process check).

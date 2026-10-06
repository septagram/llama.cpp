# One scheduler instead of five rules: design note (discussed 2026-10-06; implemented 2026-10-07 as entry 013, see 013-scheduler.md)

Idea from the user; discussed, judged worth doing, implemented right after forking v1 (entry 013). What follows is the design as discussed; the implementation notes and results are in `013-scheduler.md`. One addition made during implementation: a cap on the requests running at once (`LLAMA_SCHED_MAX_ACTIVE`, default 2), because slots are request contexts, not parallelism.

## The model

Keep one ordered list of everything that wants to run: requests that are running, waiting, or were evicted.
On every event (a request arrives, a turn finishes, a sequence is about to outgrow the pool) do two steps:

1. **Decide.** Walk the list in order and keep a running total of slots and cells. A request that fits is marked
   to be active; one that does not is skipped and the walk continues. The result is the set that should be
   active.
2. **Reconcile.** Suspend what is active and not in the set; bring in what is in the set and not active.

Nothing is remembered between events except the list, so there are no stale flags to disagree with each other.

## Why (compared with the server as of dfa8b58dc)

| | today | this model |
|---|---|---|
| where decisions live | slot picking, admission, idle-slot clearing at launch, overflow handling, resume: five places | one function: list and capacity in, active set out |
| a small request next to a large one | admitted because admission does not look at the queue; not if it was itself queued (one retry per release) | by definition |
| priorities, preemption | none; only a real overflow suspends | the order of the list |
| shared prefixes (forks) | each of the five places has to learn the arithmetic | the walk computes the marginal cells of each request given the ones already kept |
| overflow | found when a decode fails, then purge or suspend | the same function, run on projected sizes before the decode |
| testing | needs the model and the GPU | a pure function; the launcher logs can be replayed through it |

The three cache-miss shapes fixed in entry 010 were all interactions between the separate rules.

## Agreed adjustments

- **Order = (explicit priority, running, was evicted, arrival).** A running request ranks above a waiting one of
  the same priority, so a returning evicted request does not kick out an equal that slipped in meanwhile: that
  would cost a save and a restore (about 2 s) and stall a client for nothing a turn boundary would not give.
- **Preemption follows from the order, no flag.** A strictly higher priority class may evict a running lower one;
  an equal class never does.
- **Starvation guard.** "Does not fit, skip" lets a stream of small requests keep a large one waiting forever.
  Once the first waiting request has waited longer than a threshold, nothing behind it is admitted if that would
  delay it. The current server has the same hole.
- **Idle conversations are not in the list.** They are cache: their cells are reclaimable, least valuable first.
  Keeping them from falling out of host RAM is cache policy (NEXT.md item 13), not scheduling.
- **No advantage for requests whose state is resident**, for now: a legitimate tie-break, but it works against
  new conversations. Revisit with data.

## Needs first

- Keep-alive: a scheduler that pauses more requests makes more clients wait.
- Forking v1: the shared-cell accounting the fit calculation uses.
- Suspending a request during its prefill; today only a generating sequence can be suspended.

## How to build it

A separate module with no model dependency: plain structs in (the list, pool size, slots, who shares what),
the active set out. Replay the two-agent session of 2026-10-05 (12.4 h, 70 queued requests, 28 client timeouts)
through it and compare waits, evictions and prefilled tokens before it replaces anything in the server. The
mechanisms it drives stay as they are: suspend, resume, save, restore, fork.

## Open

- Whether the list should hold conversations rather than requests (a turn that ends with a tool call is very
  likely followed by another request within seconds; today the conversation leaves the list between them).
- The threshold of the starvation guard, and whether explicit priority comes from the API key, a request field,
  or both.
- Reservation for growth: how much headroom a kept request is charged for its generation.

# What is next

The work queue of the customware fork: the order the user set, then every open item with what is known about it.
Items keep their numbers; the order of work is the list right below, not the numbering.

## Order of work (set by the user, 2026-10-06)

1. **Keep-alive for queued requests**, with the queue retry fix (item 9). Done: entry 011, released 2026-10-06.
2. **Forking v1** (item 12, `notes-forking.md`). Done 2026-10-06: entry 012 (`012-forking.md`), released as `6f50e7617`, verified at 12k (fan-out, overflow) and at 88k on the launcher (the compaction fork generating next to the main agent).
3. **One scheduler instead of five rules** (item 15, `notes-scheduler.md`).
4. The remaining optimization hypotheses below (items 2-8, 10, 11, 13), in the order of their expected value.
5. **Checkpoint policy** (item 14, `notes-checkpoint-policy.md`): last, the least defined of them.

The 160k launcher: "not yet" (the user may switch, mainly because the model often ends one or two messages short
of finishing a chunk under 120k).

## Items

1. **H16: decode cost at depth** (entry 008, done 2026-09-29). The per-node profile (`GGML_SCHED_PROFILE_NODES=1`,
   `test-depth-prof.sh`, `prof1.out`) put the slope in the flash attention (with a q8_0 cache the CUDA dispatcher
   took the vector kernel, which walks the whole context) and the indexer pooling copies; `top_k` was under 2 ms.
   Fixed: `BEST_FATTN_KERNEL_MMA_F16_GATHER` + `ggml_pool_2d`: decode 97.9 / 100.7 / 119.2 -> 95.5 / 93.7 / 102.5
   ms/token at 22k / 44k / 88k, slope 0.32 -> 0.11 ms per 1k tokens. **Still on the slope** (in order of size at
   88k): `set_input_qsa`'s CPU scan of every cell per ubatch (0.7 -> 2.8 ms per token: make the block tables
   incremental), the indexer's `get_rows` over every cell (12 MB read, 46 MB written per layer: a block-key cache or
   a fused gather-and-mean), the mask fill / set_rows / add over `n_kv` per layer, the per-token mask upload. On the
   prefill side the sort over `ub x n_kv` rows costs ~5% of a ubatch at 88k (27 ms per layer per 2048 tokens).
2. **H17: prompt cache vs page cache.** Each idle-slot save carries 19-31 checkpoints of 112 MB; after a 30-minute
   agent session the cache held 10.6 GB of host RAM and free RAM fell to 6 GB. Measure `--cache-ram 4096` and
   `--ctx-checkpoints 8` against today's 16384 in an agent session; second-order on decode, first-order on RAM.
   **Evidence from 2026-09-30 night** (entry 009 note): with the Ordis VM (16 GB) and WSL running the commit
   charge is 70 GB before the server starts, the 94 GB of mmap'd model cannot stay resident, and the launcher's
   16 GB prompt cache pushed the experts out: prefill fell to 21 t/s (every ubatch faults the whole expert set
   in from disk) and a 22k prompt looked hung for 30 minutes. The failure mode is a cliff, not a slope: as long
   as the experts fit in what the page cache has left, nothing is lost; one more consumer and prefill collapses
   tenfold. **Resolved by entry 010 (2026-09-30)**: `LLAMA_CACHE_RAM_RESERVE_MIB` makes the prompt cache evict
   and skip saves whenever available RAM would drop below the reserve (76000 MiB in the launchers), so the
   server never pushes its own experts out; the Ordis VM (permanent) is accommodated the same way. What is left
   of H17 is only the measurement of `--ctx-checkpoints 8` vs 32 on an agent session (checkpoint count sets the
   size of every save: 112 MiB each). Same entry: under `--kv-unified` the prompt-cache restore now runs after the
   other idle slots are freed (and is not attempted when it cannot fit), so a cached prompt is never re-prefilled
   because another slot still held its cells; the decode-time purge saves before it clears.
3. **Adaptive placement tuning on real traces** (`LLAMA_HOT_EXPERTS_TRACE`, `customware/hot-sim.py`). On the first
   agentic trace a 1024-token half-life beat 4096 by ~1.5 points of coverage at 0.9 MB of copies per token; the band
   sits 2-4 points under the hindsight ceiling. Sweep half-life, hysteresis, per-layer budget and the prefill weight
   on a few real sessions before changing defaults. Also: per-layer budgets follow the stats only at restart.
4. **Fused CPU expert op**: gate, up, activation and down in one pass per expert, removing three of the five
   barriers per layer and keeping the intermediate in cache. Kernel work; expected 5-8% of decode.
5. **Small-batch CPU expert kernel (H12)**: batch 2-31 runs at 10-15 GB/s effective vs 29-34 at batch 1. Unlocks
   speculation (measured +7-18% on echo-heavy output today, capped by the verify batch) and `-np 2`.
   **Field data, 2026-10-05/06 launcher logs (1,646 samples with two generators)**: one generator 11.5-11.8 t/s
   median, two generators 7.5-7.7 t/s each, 15.0-15.4 t/s together (x1.27-1.34). A step goes from 86 ms to
   132 ms: the second token costs 46 ms, which is the CPU expert time of one token (47 ms in the entry 008
   profile) and nothing else. So at batch 2 the cost is the memory traffic of ten more experts per layer, not a
   kernel deficiency; a small-batch kernel has nothing to win there. What is left of H12 is the larger verify
   batches of speculation, unmeasured.
6. **Overlap the GPU hot half with the CPU cold half** of each layer's experts in the scheduler (launch the GPU split
   before computing the CPU split when they are independent). Bounded by the GPU hot time: 2-4 ms per token, 3-5%.
7. **Second-GPU probe**: hot tensors of half the layers on the 2060 SUPER as a pessimistic stand-in for a second
   4060 Ti; measures the hop cost before any purchase.
8. **Steering** (parked): control vectors are not wired for this architecture (four residual streams, no cvec hook);
   ~10 lines to apply at the single-stream layer output, a matching capture point for the generator, and a per-token
   strength schedule that ramps with the length of the thinking block. The hard part is the ruler (deliberation
   tokens before the first tool call, task outcomes).

9. **Queued requests vs client timeouts** (from the logs of 2026-10-05): 28 of 70 queued requests were cancelled
   by the client after 300 s. With entry 010 a swap between two large conversations is a restore of seconds, so
   the wait is the other agent's turn; a 3000-token turn is still five minutes. Raise the clients' request
   timeout; server side, keep-alive bytes on a queued streaming response would make the queue invisible.
   **Done in entry 011 (2026-10-06)**, `customware/011-keepalive.md`: a stream that has no result after one ping
   interval (30 s) starts with a ping and is pinged until its first result; a slot release retries the whole
   deferred queue in order. What is left: non-streaming requests cannot be kept alive (nothing may precede the
   JSON body), and a client's own limit on the time to the first token is not affected by pings (four long
   prefills in the logs ended that way although headers and pings were being sent).
10. **RAM guard for the prompt cache** (`LLAMA_CACHE_RAM_RESERVE_MIB`, built, off): before it is switched on it
   must never refuse the save of a live slot's prompt (evict older entries only), or the slot must keep its cells
   when the save is refused. See 010-cache-reserve.md.
11. **A cached entry much longer than the new prompt** (after a compaction): the cache picks an entry when a
   quarter of it is a prefix of the new prompt and the restore needs cells for all of it; next to a busy slot the
   restore is skipped and the prompt prefilled from zero. Deferring until it fits would close the gap. Below a
   quarter the entry is not used at all (upstream rule), so a compacted conversation re-prefills its unchanged
   system prompt and tools.

12. **Proper forking** (done 2026-10-06, entry 012; design in `customware/notes-forking.md`, results in
   `012-forking.md`). v1: a request whose longest common prefix is a busy slot's conversation forks from it in
   the pool (shared cells + the fork-point checkpoint) and runs next to it; share-aware cell accounting; a
   checkpoint at every turn end. Later: parking (idle conversations keep their cells without a slot),
   refcounted checkpoints, range-limited save/restore so a saved fork keeps its shared prefix, forking from an
   idle source that diverges before its end (today taken over in place), preferring a longer cache entry over a
   shorter busy prefix.
13. **The prompt cache fills with dead pre-compaction states.** At a compaction the new prompt shares almost
   nothing with the old one (41 tokens in one harness, ~8.4k in another), so it goes to the other slot and the
   old 116k conversation is saved as a 5.2 GiB entry that nothing will ask for again. Three of them fill the
   16 GiB limit; the next save evicts two entries at once, possibly a live agent's. In the session of
   2026-09-30: 28 saves, 11 evictions, cache at 12-15 GiB throughout. This is also most of the RAM pressure of
   H17. Options: an age or count limit for entries nobody extended, saving such entries without checkpoints, or
   a harness hint.

14. **Checkpoint policy** (parked by the user, 2026-10-06): which checkpoints to make and keep. The research so far
   is in `customware/notes-checkpoint-policy.md` (the "just before the AI turn" checkpoint served 62 of 89
   rollbacks in the launcher logs; retention by recency and a lower cap are the candidates). Needs an
   architectural discussion before anything is changed; not part of the fork work except the eager turn-end
   checkpoint.

15. **One scheduler instead of five rules** (user's idea, 2026-10-06; after forking): one ordered list of
   everything that wants to run, a single function that walks it and decides the active set, then a reconcile
   step. Brings priorities and preemption, makes the small-request-next-to-a-large-one case true by definition,
   and puts the shared-cell arithmetic in one place. `customware/notes-scheduler.md`.

Tooling that exists: `test-next.sh`, `test-spec.sh`, `test-depth.sh`, `test-adapt.sh`, `test-concurrent.sh`,
`verify-launcher.sh`, `run-pi-task.ps1` (pi needs `SHELL` pointed at Git Bash), `hot-sim.py`, `stats-seed.py`,
`expert-place.py`, `equiv-compare.py`, `prof-summary.py`.

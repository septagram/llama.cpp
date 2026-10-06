# Next optimizations (parked 2026-09-29 while the server correctness work runs)

Resume from here when the queueing/preemption work (entry 006) and the concurrent-prefill fix are done. Order is
by expected value on the user's real workload (agent sessions at 40-120k depth on the 120k launcher).

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
6. **Overlap the GPU hot half with the CPU cold half** of each layer's experts in the scheduler (launch the GPU split
   before computing the CPU split when they are independent). Bounded by the GPU hot time: 2-4 ms per token, 3-5%.
7. **Second-GPU probe**: hot tensors of half the layers on the 2060 SUPER as a pessimistic stand-in for a second
   4060 Ti; measures the hop cost before any purchase.
8. **Steering** (parked): control vectors are not wired for this architecture (four residual streams, no cvec hook);
   ~10 lines to apply at the single-stream layer output, a matching capture point for the generator, and a per-token
   strength schedule that ramps with the length of the thinking block. The hard part is the ruler (deliberation
   tokens before the first tool call, task outcomes).

Tooling that exists: `test-next.sh`, `test-spec.sh`, `test-depth.sh`, `test-adapt.sh`, `test-concurrent.sh`,
`verify-launcher.sh`, `run-pi-task.ps1` (pi needs `SHELL` pointed at Git Bash), `hot-sim.py`, `stats-seed.py`,
`expert-place.py`, `equiv-compare.py`, `prof-summary.py`.

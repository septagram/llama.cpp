# Next optimizations (parked 2026-09-29 while the server correctness work runs)

Resume from here when the queueing/preemption work (entry 006) and the concurrent-prefill fix are done. Order is
by expected value on the user's real workload (agent sessions at 40-120k depth on the 120k launcher).

1. **H16: decode cost at depth.** Measured on the fresh launcher: 80.8 ms/token at 22k, 94.8 at 44k, 116.2 at 92k
   (~0.5 ms per 1k tokens of context). Suspect: the QSA indexer scores and top-k selects over all n_kv positions per
   token per layer (a full argsort per head). Profile per op at 90k depth (`GGML_SCHED_PROFILE` + a CUDA op-level
   breakdown), then a radix/partial top-k kernel or a cached-score path. Biggest remaining decode lever for long
   sessions.
2. **H17: prompt cache vs page cache.** Each idle-slot save carries 19-31 checkpoints of 112 MB; after a 30-minute
   agent session the cache held 10.6 GB of host RAM and free RAM fell to 6 GB. Measure `--cache-ram 4096` and
   `--ctx-checkpoints 8` against today's 16384 in an agent session; second-order on decode, first-order on RAM.
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

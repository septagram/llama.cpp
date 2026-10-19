# customware/003: hot experts on the GPU (per-expert placement instead of per-layer)

## Goal

Spend the ~6.3 GB of VRAM that today holds the routed experts of 4 whole layers on the *most used*
individual experts of every CPU layer instead. The expert histogram shows heavy routing skew (per
layer the top 10% of experts take 53-87% of routed slots, top 20% take 78-95%), so the same VRAM
can remove far more than 4/44 of the CPU's expert traffic. Expected: CPU expert bytes per token
-50..60%, generation ~12 -> 16-18 t/s if the skew holds on real prompts. Needs no pinned memory.

## Context

- With the 4060 Ti only: 83 ms/token = CPU experts ~50 ms (44 layers x 1.15 ms, bandwidth-bound at
  ~30 GB/s), GPU ~28 ms, split syncs ~8 ms. Each CPU layer reads 10 experts x ~3 MB.
- `cpu-prof-experts` lines (GGML_CPU_PROFILE) give per-layer counts of routed expert ids.
- Pinned/streamed experts are out on Windows (RAM), so the hot set must be GPU-resident.

## Steps

### 1. Placement from statistics: `customware/expert-place.py`

Input: one or more logs with `cpu-prof-experts: <layer> <count> x n_expert` lines (the last block per
log wins, counts are cumulative), a VRAM budget in MiB, the per-layer expert byte size (from the
GGUF: gate + up + down row-block sizes, layers 2/4/30/46/47 are bigger). Output: a text file
`<layer> <id> <id> ...` chosen greedily by count/bytes across all CPU layers until the budget is used,
plus a report: share of routed slots covered per layer and overall. Also a mode that ranks whole
layers, to compare against the current per-layer placement.

### 2. Model side: load the hot experts

- `llama_layer` gets `ffn_gate_hot`, `ffn_up_hot`, `ffn_down_hot` (3D, `n_hot` experts), `ffn_hot_map`
  (I32 [n_expert]: id -> local slot, cold -> 0), `ffn_hot_mask` (F32 [n_expert]: 1 hot / 0 cold),
  `ffn_cold_map` (I32 [n_expert]: cold -> id, hot -> -1).
- `LLAMA_HOT_EXPERTS=<file>` (env, read once in llama-model.cpp): for each listed layer whose full
  expert tensors live on a host buffer, create the six tensors in the CUDA0 context before buffer
  allocation (they are not GGUF weights, so the loader skips them), and after `load_all_data` fill
  them: expert rows copied with `ggml_backend_tensor_set` from the mmap'd full tensors, maps/masks
  computed from the list. Log the VRAM used and the covered share.

### 3. Graph: split every configured layer's MoE into a GPU part and a CPU part

In `llm_graph_context::build_moe_ffn`, after `weights` [1, n_used, n_tokens] and `selected_experts`
are known, when the layer has hot tensors:

```cpp
ggml_tensor * ids_gpu = ggml_get_rows(ctx0, hot_map,  ids);          // I32 [n_used, n_tokens]
ggml_tensor * ids_cpu = ggml_get_rows(ctx0, cold_map, ids);          // I32, -1 for hot slots
ggml_tensor * m       = ggml_get_rows(ctx0, hot_mask, ids);          // F32 [1, n_used, n_tokens]
ggml_tensor * w_gpu   = ggml_mul(ctx0, weights, m);
ggml_tensor * w_cpu   = ggml_sub(ctx0, weights, w_gpu);
out = run_experts(gate_hot, up_hot, down_hot, ids_gpu, w_gpu)        // lands on CUDA0 (weights there)
    + run_experts(gate_exps, up_exps, down_exps, ids_cpu, w_cpu);    // lands on the CPU
```

`run_experts` is the existing up/gate/activation/down/weighted/reduce tail of `build_moe_ffn`, factored
into a lambda. The scheduler places each part by its weights, no forcing needed.

- CPU `mul_mat_id` (ggml-cpu.c, and the repack/iqp variants assert): a negative id means "skip": the
  row is not grouped, and the destination rows of skipped slots are zeroed by thread 0 so the
  masked weight (0) meets a finite value.
- The GPU part maps cold slots to local slot 0 and multiplies by 0; one extra expert of GPU compute.

### 4. Measure

- llama-bench tg128 and pp2048 (ub 2048) with `LLAMA_HOT_EXPERTS` vs the 4-whole-layer placement,
  same VRAM. `GGML_CPU_PROFILE` for the CPU expert bytes per token; `GGML_SCHED_PROFILE` for the
  extra GPU nodes. Then the server at 22k depth.
- Robustness of the placement: statistics from the code prompt vs the prose prompt; measure the
  coverage each placement achieves on the other prompt.

## Verification

- Greedy generation (temperature 0, fixed prompt) with and without the split produces the same tokens
  for at least the first 64 tokens (float reassociation only).
- llama-bench prints the same "size/params"; VRAM use within +5% of today's 4-layer config.
- No NaN/inf in the outputs with all-hot and all-cold routed sets (unit check through the graph:
  a layer with n_hot = 0 configured is a no-op).

## Open questions

- How stable is the hot set across prompt types; whether a small per-layer floor (e.g. top 5 always)
  plus a global greedy remainder is the better policy.
- Whether to keep a fully hot layer (all 512 on the GPU) for the most skewed layers instead of the
  split, to avoid the split's extra nodes there.

## Execution policy

- Implementer: the planner (Fable), as in entries 001 and 002.
- Commit policy: one commit per step (placement script; model loading; graph split + CPU kernel;
  results), directly on `master`, `Assisted-by: Claude Fable 5.1`, `Plan: 26c-tiptop-latch-uneven`.
- Review policy: none mid-way; stop and report before any review when the optimization work is done.

## Implementation notes (what actually shipped)

- Placement script, loader (compacted `ffn_{gate,up,down}_hot` + `ffn_hot_map`/`ffn_cold_map`/`ffn_hot_mask` on
  the first GPU, filled from the mmap'd full tensors after the load), and the graph split in `build_moe_ffn`
  shipped as planned. The maps are looked up with a flattened index tensor (get_rows treats a 2-D index as
  batched, which only worked for one token). Loader contexts got headroom for the extra tensors.
- The split is numerically equivalent: greedy output byte-identical over 64 tokens vs the whole-layer config.
- Deviation: "-1 = skip" is honoured on the GPU too, not only by the CPU kernel. The first prefill version mapped
  masked slots to a dummy expert, which produced duplicate ids within a token and crashed the CUDA MMQ ids helper
  (it assumes one slot per expert per token). Now every mul_mat_id path leaves zeros for negative ids: CPU kernel,
  MMQ ids helper (+ inverse map memset and scatter skip), MMVQ (both kernels), the host fallback (extra zero row),
  and the scheduler's "copy only used experts" scan. `op_params[3] = 1` marks such ops so MMQ zeroes its output.
  With that the split also runs during prefill: hot experts compute on the resident copy and only cold experts
  are streamed, which lowers both the PCIe bytes per batch and the page-cache working set of the experts.
- Measurements: llama-bench with a placement learned from its own routing (77% coverage): tg128 12.0 -> 15.3 t/s,
  CPU 57 -> 28 ms/token, GPU unchanged (the split's extra nodes are free). Server, code prompt at 22k depth, mixed
  placement (45% coverage on both prompts): CPU expert ops halved (q4_K 292 -> 149 us, q5_1 402 -> 200 us), but
  tg stayed ~10 t/s because the n-gram table (28.8 GB mmap, 24 random rows per token) page-faulted for up to 35 ms
  per token; the box cannot keep experts + n-gram table + buffers resident. Locking the table (`LLAMA_LOCK_TENSORS`)
  fixed its lookups (0.16 ms) but made prefill thrash (38 t/s) and decode 6 t/s until the cache settled.
- Coverage depends on the workload: per-prompt placements cover 57-61% of their own prompt, 34-38% of a different
  one; the mixed placement 53-55%. `GGML_CPU_PROFILE` in the server logs the histogram, so the placement can be
  regenerated from real sessions with `customware/expert-place.py`.
- Real-config A/B (server, 22.2k code prompt, 300 generated, 160k context, 2 slots, unified KV, `-ub 1024`, page cache warm):
  hot experts 4 GB (`hot-experts-mixed-4000.txt`, 41.5% of routed slots served from the GPU) 110 t/s prefill, 12.85/12.93 t/s
  generation at 14.2 GB VRAM; plain 3 whole expert layers (28-30) at the same settings 115 / 11.51/11.48 at 15.2 GB: +12%
  generation for 1 GB less VRAM. Bigger placements do not fit next to the 160k buffers: 4.5 GB + ub 1024 and 6 GB + ub 512
  push VRAM past ~15 GB, where WDDM starts paging and everything crawls.
- Launcher config (agreed trade of context for speed: 120k instead of 160k): `-c 120000 -np 2 --kv-unified -ub 2048 -b 2048`
  + 4 GB hot experts = 14.6 GB VRAM, 169 t/s prefill and 12.5 t/s generation at 22k depth (the daily script did ~70 / ~7).
  160k + ub 1024 (110 / 12.9) is documented in the launcher as the fallback. `customware/release.sh` writes both the frozen
  binary + placement (`C:\dev\llama.cpp-customware`) and the launcher (`C:\dev\textgen\...customware.sh`).
- Numerics with the split at prefill: greedy log-probs vs the plain config agree to 0.0025 nats (mean) over the identical
  prefix and then a near-tie flips at token 2 (' answer' -0.63/-0.95 vs ' to' -1.15/-0.74); the earlier "byte-identical over
  64 tokens" check compared token ids only and predates the prefill split. The GPU half (MMQ/MMVQ) and the CPU kernels
  accumulate in different orders, so bit-equality is not expected; the perplexity comparison below is the check that matters.
- Perplexity, same settings for both (`-c 8192 -ub 1024 -b 2048 --chunks 4`, 390 KB of repo docs + source, q8_0 KV, CUDA0 only):
  plain 3 whole layers 2.1431 +/- 0.0301, hot 4 GB split (prefill through the split too) 2.1345 +/- 0.0298. Equal within
  the error bar, so the split is numerically sound at prefill and decode; the token-2 flip above is a near-tie.
- The launcher that crashed for the user ("out of memory while offloading") was the frozen binary from commit 5f782a926,
  where `-ot ...=CPU` still landed the experts in pinned host memory (`CUDA_Host model buffer size = 67200 MiB` in the
  log) and the WDDM shared budget ran out at the first GPU upload. Fixed by defc32010 in the repo; the release was
  simply never refreshed. `release.sh` now ships 13684c74e and the launcher sets `-fit off` (the fit step aborts anyway).

## Final commit list

Sealed 2026-10-20 into `003-hot-experts.patch`: this entry's own diff (customware/ excluded), applied after the entries before it onto upstream `6e60f3560` (tag `v0.5.0-customware.0`); `customware/seal-entries.sh` rebuilds and checks it from `customware/entry-commits.txt`.

- `4cbfd07e7` llama : hot experts on the GPU (LLAMA_HOT_EXPERTS)
- `13684c74e` CUDA : mul_mat_id treats a negative expert id as "skip" (hot experts at prefill)
- `df604b39f` customware: entry 003 results, refreshed release (120k / ub 2048 / 4 GB hot), entry 004 plan (notes only)

# customware/008: decode cost at depth (H16): profile the QSA indexer, then cut it

## Goal

Decode costs 80.8 ms/token at 22k depth, 94.8 at 44k and 116.2 at 92k on the launcher config: 0.5 ms per 1k tokens
of context, 35 ms/token by the time an agent has filled 90k. Nothing on the expert side depends on depth, so the slope
comes from the 12 QSA attention layers. Find where, by measurement, and remove the avoidable part.

## What the code says (facts, not the measurement)

- `build_qsa_top_k` (src/models/qwen4exp.cpp) re-derives every block's pooled indexer key on every token: a
  `ggml_get_rows` of all `n_kv` cells of the q8_0 indexer cache (dequantised to f32: 12 MB read, 46 MB written at 90k),
  four slice copies and three adds for the mean over the ratio-4 block, rms_norm and rope over the 22.5k blocks, the
  score mul_mat, then a per-cell expansion and `ggml_top_k` over the `n_kv`-wide row. Rough traffic: ~300 MB per
  layer, ~3.6 GB per token at 90k, ~12 ms on the 4060 Ti's 288 GB/s. The pooled key of a complete block only changes
  when its cells change.
- `ggml_top_k` on this CUDA 12.3 build (CCCL 2.x, no `DeviceTopK`) is a full `DeviceRadixSort::SortPairsDescending`
  of the 90k row into three pool-allocated temporaries plus a strided copy of the first 2051 indices, 12 times per
  token. `ggml/src/ggml-cuda/top-k.cu` already carries a 5-pass radix select (`top_k_radix_cuda`) but only compiles it
  for HIP.
- `set_input_qsa` (src/llama-memory-hybrid-idx.cpp) scans every cell on the CPU per ubatch: its own TODO measured
  865 us at 33k, so ~2.4 ms per token at 90k.
- The flash attention itself is already sparse (`n_kv_max` -> mask compaction, O(2048) per query).

## Steps

1. **Per-node profile mode** in the split profiler (`ggml/src/ggml-backend.cpp`): `GGML_SCHED_PROFILE_NODES=1` on
   top of `GGML_SCHED_PROFILE=N` computes each split node by node with a backend sync after every node and
   accumulates time by (op, tensor name without the `-<layer>` suffix); every dumped graph prints the top rows with
   count, total and mean. Node-by-node sync exposes launch latency, so it ranks costs rather than measuring them
   exactly. `llama_context::process_ubatch` times `set_inputs` under the same variable so the CPU scan shows up too.
2. **Profile decode at 22k / 45k / 90k** with the launcher config on port 5011 (`customware/test-depth-prof.sh`:
   classic hot experts, the depth requests, two dumps per depth), and write the ranking into the entry note.
3. **Fix in order of measured cost**; the candidates, cheapest first:
   a. Radix-select top-k for wide rows on NVIDIA: compile `top_k_radix_cuda` for CUDA as well and use it when
      `ncols > 1024` and `DeviceTopK` is unavailable. Output order is unspecified, as it already is on the `DeviceTopK`
      path (`output_ordering::unsorted`); the only consumer here is `ggml_set_rows` into the mask. Ties at the k-th
      score resolve arbitrarily in both implementations; in this model they only occur inside the tail block.
   b. Cheaper pooling: gather once and mean in one pass instead of four slice copies and three adds.
   c. If pooling still dominates: a per-layer block-key cache (f16 `[128, n_blocks_max]`, ~8 MB per layer) filled by
      `set_rows` for the blocks the ubatch touches, invalidated from the same per-ubatch cell scan, recomputed lazily
      after state restore. Only if the profile says the pooling is the bulk and (b) is not enough.
   d. Incremental `set_input_qsa` (keep the block tables across ubatches) if the CPU scan is a visible slice.
4. **Verify and ship**: `test-backend-ops -o TOP_K` with rows of 90k and 131k; the depth series again; greedy output
   at 22k compared with the previous run (tail-block ties may legitimately differ, report if they do); release into
   both launchers; dashboard; memory; NEXT.md.

## Verification

- The profile at 90k names the top costs with numbers; the note records them.
- After the fix: ms/token at 22k / 44k / 88k, the slope in ms per 1k tokens before and after; VRAM unchanged.
- `test-backend-ops` TOP_K passes on CUDA for wide rows; the full-run perplexity spot check (one 8k chunk) matches.

## Out of scope

Prefill at depth (the sort over `ub x n_kv` rows) unless the profile shows it; the two-GPU probe; the fused CPU
expert op.

## Profile (2026-09-29, `customware/prof1.out`, launcher-shaped config, decode graphs only)

Per-split summaries (mean over ~44 decode graphs, a sync per split): CUDA0 compute 60.6 / 61.7 / 81.4 ms at
22k / 44k / 88k, CPU experts 47.3 / 46.4 / 45.3 ms, `set_inputs` 0.74 / 1.41 / 2.78 ms. The node ranking of the
dumped graphs (unnamed nodes merged per op) puts the depth cost where the plan did not expect it:

| node                              | 22k     | 44k     | 88k      |
|-----------------------------------|---------|---------|----------|
| FLASH_ATTN_EXT (12 layers)        | < 1.9   | 4.4 ms  | 14.5 ms  |
| CONT of the pooling slices (48)   | 2.0 ms  | 2.3 ms  | 5.3 ms   |
| TOP_K (12)                        | < 1.7   | < 1.7   | < 1.7 ms |
| set_inputs (CPU, per ubatch)      | 0.7 ms  | 1.4 ms  | 2.8 ms   |

The flash attention grows linearly with the context although the op carries `n_kv_max` = 2051: with a q8_0 cache
`ggml_cuda_get_best_fattn_kernel` picks the *vector* kernel for decode (`Q->ne[1] <= 2` on Ada), and the vector
kernel has no index list: it reads every K and V row of the cache. The sparse path (mask compaction, O(n_kv_max))
exists only in the f16 MMA kernel, and taking it with a quantized cache would first convert the whole cache to f16
(`launch_fattn`, `need_f16_K/V`). The top-k is a non-issue at decode (one row); at prefill it is a real one
(27 ms per layer per 2048-token ubatch at 88k: a full sort of `ub x n_kv` pairs), about 5% of a ubatch.

## Implementation

- `ggml/src/ggml-cuda/fattn.cu`: `BEST_FATTN_KERNEL_MMA_F16_GATHER`, chosen on NVIDIA Turing+ when K or V is
  quantized, the op carries `n_kv_max`, the query batch is 1 or 2 (decode), head size 64/128/256, no ALiBi or
  softcap, and the context holds at least 4 x n_kv_max x queries cells. It compacts the mask into one index list
  per query (`ggml_cuda_flash_attn_ext_compact_mask`, the existing kernel), gathers and dequantizes those rows of K
  and V into f16 pool scratch (`get_rows_cuda`, one list after the other, padded to a multiple of
  `FATTN_KQ_STRIDE`), writes a compact mask (the original value at the gathered cell for the owning query, -inf
  elsewhere), and runs the dense MMA kernel on shallow tensor copies pointing at the scratch. `get_alloc_size`
  requests nothing for this path; the scratch is pool memory. test-backend-ops cases added for the qwen4exp shape
  (256/256, gqa 12, budget 2048, one and two queries, q8_0 and q4_0).
- `src/models/qwen4exp.cpp`: the block mean is `ggml_pool_2d(AVG, 1 x r)` over the gathered keys instead of four
  slice copies and three adds (5 ms per token at 88k for 12 layers).
- `customware/build-win.cmd` configures with tests on and builds `test-backend-ops` too.

## Verification log

- `test-backend-ops test -b CUDA0 -o FLASH_ATTN_EXT`: 3986/3986 pass, including the four gather cases (256/256
  gqa 12 q8_0 at 8k with one query and at 16k with two, q4_0 at 8k, 128/128 gqa 8 q8_0 with two queries);
  `-o POOL_2D`: 128/128.
- Launcher verification of the admission-fix release (f6a59d2e8, projector on CUDA1): 120k launcher up, 22k code
  prompt 222.9 t/s prefill / 11.55 t/s generation at 15.4-15.8 GB VRAM (the desktop holds 1.4 GB now, so the 6 GB
  set sits in the paging regime; `LLAMA_HOT_EXPERTS_MIB=5000` is the advice).
- **Depth A/B** (`customware/test-depth-ab.sh ab1`, `depth-ab1.out`): released binary vs the build, 120k pool,
  ub 2048, the 4 GB alias set, 12.0 GB VRAM (no paging), 64 greedy tokens, decode ms/token:

  | depth | old   | new   | change   |
  |-------|-------|-------|----------|
  | 22k   |  97.9 |  95.5 | -2.4 ms  |
  | 44k   | 100.7 |  93.7 | -7.0 ms  |
  | 88k   | 119.2 | 102.5 | -16.7 ms (-14%) |

  Slope: 0.32 -> 0.11 ms per 1k tokens of context. Prefill unchanged (166-187 t/s both). Greedy outputs identical
  at 44k and 88k; at 22k they share 171 characters and then diverge: the gathered K/V rows are f16 where the vector
  kernel used the int8 dot products of the q8_0 cache, and the block mean is now r partial products instead of one
  sum, so near-ties can flip. What remains of the slope: `set_inputs` (the QSA block tables, 0.7 -> 2.8 ms), the
  indexer's `get_rows` over every cell (12 MB read, 46 MB written per layer at 88k), the mask fill/set_rows/add
  over `n_kv`, and the per-token mask upload.
- **The op alone** (`test-backend-ops perf -o FLASH_ATTN_EXT`, 256/256, gqa 12, q8_0 K/V, one query, per layer):

  | cells  | vector kernel (n_kv_max 0) | gather (n_kv_max 2048) |
  |--------|----------------------------|------------------------|
  | 24576  | 324 us                     | 44 us                  |
  | 49152  | 685 us                     | 50 us                  |
  | 90112  | 1327 us                    | 61 us                  |

  Twelve layers at 88k: 15.9 ms -> 0.7 ms per token; the op no longer depends on the context to speak of.
- Launcher verification of the 008 release (8a3a84e8c): 120k launcher up in 25 s, 22k code prompt 203.4 t/s
  prefill / 12.25 t/s generation at 15.4-15.8 GB (11.55 t/s with the previous binary in the same VRAM state).

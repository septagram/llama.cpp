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

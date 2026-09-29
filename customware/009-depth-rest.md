# customware/009: the rest of the depth slope, and what a second generator really costs

## Goal

After entry 008 decode costs 95.5 / 93.7 / 102.5 ms/token at 22k / 44k / 88k (0.11 ms per 1k tokens of context).
Two questions remain that decide where the next hours go, and both are measurements first:

1. What is still on the slope, and is any of it cheap to remove? The candidates from the 008 profile: the
   indexer's `get_rows` over every cell of the q8_0 indexer cache (12 MB read, 46 MB of f32 written per layer at
   88k, then read again by the pool), the QSA block tables built on the CPU every ubatch (`set_input_qsa`, 0.7 ->
   2.8 ms), and a handful of O(n_kv) mask ops with tiny bytes.
2. H12: with `-np 2` the second generator was measured at 10-15 GB/s "effective" against 29-34 at batch 1. Whether
   that is a kernel deficiency or just the routing (two tokens touch ~20 distinct experts, so the weight traffic
   doubles) decides whether a small-batch CPU expert kernel is worth writing. The user runs `-np 2` for side
   questions next to an agent.

## Steps

1. **Fresh profile on the 008 binary** (`customware/test-depth-prof.sh prof2`): the node ranking at 22k / 44k /
   88k with the gather path and the pool in place. Record the remaining depth-dependent rows in the note.
2. **Fused gather-and-mean** for the indexer keys: a CUDA fusion of `GET_ROWS` (quantized or f16 source) followed
   by `POOL_2D` (avg, window 1 x r, stride 1 x r, no padding) into one kernel that reads each gathered row once
   and writes the f32 mean: `ggml_cuda_can_fuse` pattern `{GET_ROWS, POOL_2D}` with the shape checks, dispatch
   next to the other fusions, kernel in `getrows.cu`. test-backend-ops cannot see fusions directly; the check is
   the greedy output of the depth requests before/after and the profile rows.
3. **Batch-2 decode profile** (`customware/test-np2-prof.sh`): two generators on one server, the per-node ranking
   of the two-token graphs (prof-summary.py gains a token-count selector). Compare `MUL_MAT_ID` per layer at batch
   2 with batch 1 from step 1: ~2x means bandwidth (the routing), 3-4x means the kernel. Write the verdict on H12
   into NEXT.md either way.
4. Depending on 1 and 3: either the `set_input_qsa` constant-factor work (persistent scratch, one pass less) or
   the small-batch kernel investigation. Not both tonight.
5. A/B depth series, commit, release, notes, dashboard, memory.

## Verification

- Depth series ms/token at 22k / 44k / 88k before/after step 2 (same config as the 008 A/B, hot set disabled there,
  which is the same on both sides).
- Greedy outputs of the three depth requests unchanged by the fusion (the mean is computed in the same order;
  differences would point at a bug).
- The batch-2 profile answers H12 with numbers.

## Profile on the 008 binary (`prof2.out`, 2026-09-30 00:15, under the load described below)

Decode 145.3 / 120.9 / 131.0 ms/token at 22k / 44k / 88k under profiling (the 22k point ran while Zed spun
hardest). Rows that grow from 22k to 88k in the node ranking (node-by-node, a sync after each): the unnamed
`GET_ROWS` group +4.4 ms (174 nodes per graph, 61 us each: the indexer key gather of every cell, and the
expansion of one score per block to every cell, an 88k-rows-of-one-float gather that the get_rows kernel runs as
88k blocks of 256 threads), `POOL_2D` 3.4 ms at 88k, `set_inputs` 0.8 -> 2.8 ms. Everything else is flat.

## Implementation

- `ggml/src/ggml-cuda/getrows.cu`: `k_get_rows_float_tall`: gathers with rows of at most 8 elements and at least
  1024 rows get one thread per element (the score expansion at decode has rows of one float).
- `ggml/src/ggml-cuda/getrows.cu`: `k_get_rows_mean_q` / `k_get_rows_mean_float` and `ggml_cuda_op_get_rows_pool2d`:
  one thread pair per output element, the r gathered rows read once, each term scaled by 1/r and summed in the
  same order as `pool2d_nchw_kernel`, so the fused result matches the unfused one bit for bit.
  `ggml_cuda_can_fuse_get_rows_pool2d` admits: source f32/f16/q4_0/q4_1/q5_0/q5_1/q8_0 with ne[3] = 1, I32
  indices `[r*n_out, n_stream]`, avg pool with k0 = s0 = 1, k1 = s1 = r, no padding, f32 output of
  `[ne00, n_out, n_stream]`.
- `ggml/src/ggml-cuda/ggml-cuda.cu`: the `{GET_ROWS, POOL_2D}` pattern in `ggml_cuda_can_fuse` (after the generic
  single-use check) and its dispatch ahead of the rms_norm patterns.
- `customware/test-np2-prof.sh` + `NTOK=2 prof-summary.py`: the two-token graph ranking for H12.

## Verification log

- test-backend-ops on CUDA0: GET_ROWS 219/219 (three tall cases: 90112 rows of one f32, 20000 rows of two with
  two batches, 90112 rows of one f16), POOL_2D 128/128, FLASH_ATTN_EXT 3986/3986.
- Depth A/B `ab2` (008 release vs this build, same config, hot set disabled on both, 12.0 GB VRAM), decode
  ms/token at 22k / 44k / 88k: **old 125.2 / 124.4 / 139.4 -> new 113.3 / 116.1 / 126.3**. Greedy outputs
  identical at all three depths (the fused mean is bit-for-bit the pool's). Caveat: both sides ran under the
  background load below, which moved between the two runs; the gain shows at every depth (-12, -8, -13 ms)
  while the 22k -> 88k slope is nearly unchanged (14.2 -> 13.0 ms), so part of the level shift is the load.
  A reversed-order pair at 22k follows to separate the two.

## The "hang" (01:20-02:10): page cache, not kernels

The launcher verification of the 009 release stalled on its 22k code prompt for 30 minutes with no progress
line, while the depth A/B on port 5011 had run the same prompts normally. Bisecting on 5011 with the launcher's
shape (ub 4096, alias set active): fusion on and off, then the tall gather on and off, all four the same: the
first 4096-token batch completes after 184-194 s, 21-22 t/s, GPU at 100%. That is the expert weights streaming
from disk, not a kernel (the disk itself reads the 112 GB of shards sequentially in 79 s; the 340 MB/s effective
during a squeezed prefill is page-fault churn: every ubatch faults the whole expert set through a cache window
smaller than it). The box tonight has 70 GB of commit charge with no server
running: the Ordis VM (16 GB assigned) and WSL NixOS are running (vmmem 11 GB private), rust-analyzer 3 GB,
Telegram and Zed 2 GB each. The 94 GB of mmap'd model cannot stay resident next to that, and the launcher's
16 GB prompt cache (H17) pushed the experts out entirely. Two more findings from the same bisect:

- the `{GET_ROWS, POOL_2D}` fusion never fired: `ggml_can_fuse` demands consecutive nodes of the same shape,
  and the pool's output is r times shorter than the gather's. The rope/set_rows patterns use
  `ggml_can_fuse_subgraph` for the same reason; the pattern now does too, ahead of the generic gate, and the
  op logs once when it first runs.
- the tall gather is behind `GGML_CUDA_TALL_GET_ROWS=1` until it is A/B'd on its own; it is not the stall.

The A/B numbers of `ab2` therefore compare the 008 binary with the tall gather only (fusion inert), under a
page cache that was being squeezed. They are not the fusion's numbers.

With the gate fixed the fusion fires (the log says so), and the first on/off comparison showed different
outputs: the 48-token log-probs diverged at token 2 with a different distribution, the 12k greedy text after 30
characters. That comparison was confounded: its "off" side used `GGML_CUDA_DISABLE_FUSION=1`, which switches
off every CUDA fusion (rms_norm+mul, rope+set_rows, mul_mat+glu, ...) and changes the numerics of all 48 layers.
`test_get_rows_pool2d` (a whole-graph case in test-backend-ops, CUDA fused vs the reference unfused) passes
5/5 including 90112 cells, two streams and q4_0 with r = 8, so the kernel itself is exact. The clean A/B is
`customware/test-fusion-ab.sh`: `GGML_CUDA_NO_GET_ROWS_POOL=1` disables only this fusion. The released binary
(b4ba3b3fa) carries the inert gate, so nothing wrong is live; the fixed gate is released only when that A/B
shows identical outputs.

## Environment tonight (2026-09-30, 00:30)

Zed spins 1.35 cores continuously, llmsniff 0.5, vorpControl 0.25, sst-prototype 0.2: about 2.3 of 8 cores
busy next to the server's six barrier-synchronized threads. The 22k profile point came out 25% slower than the
same config yesterday (145 vs 120 ms/token under profiling). At 01:10 sst-prototype was at 1.8 cores and Zed at
1.2: the load moves by a core or two between runs, which is the size of the effects measured here. The user's processes are left alone; every number
tonight is read as an A/B pair run back to back under the same load, never against yesterday's absolute values.

## Out of scope

The prefill-side sort over `ub x n_kv` rows; a block-key cache (VRAM); anything that needs the user's sessions
(H17, adaptive tuning on real traces).

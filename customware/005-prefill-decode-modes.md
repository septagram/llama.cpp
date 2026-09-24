# customware/005: prefill mode / decode mode (hot experts in the tail of the compute buffer)

## Goal

Remove the context-vs-ubatch trade on the 4060 Ti. Today 160k context forces `-ub 1024` (110 t/s prefill) because the
QSA indexer compute buffers scale with context x ubatch (5.0 GB at 160k / ub 2048) and do not fit next to the 4 GB hot
set; the shipped launcher trades 40k of context for ub 2048 (169 t/s). The hot experts and the big prefill compute
buffer are never needed at the same time: a decode graph uses ~200 MB of the compute buffer, and a big prefill ubatch
gains little from the hot set (it streams all 512 experts per ubatch anyway; the 4 GB set is 6% of those bytes).

## Design

- `LLAMA_HOT_EXPERTS_ALIAS=1`: the model creates the three big hot tensors per layer (`ffn_{gate,up,down}_hot`) in a
  no-alloc context of its own, without storage; the maps stay ordinary CUDA0 weights (6 KB per layer). The context
  places them in the **tail** of the scheduler's CUDA0 compute buffer after the worst-case reserve, at
  `[size - hot_bytes, size)` (`ggml_backend_tensor_alloc`), and fills them from the mmap'd host tensors.
- **Decode mode**: the hot split is built (`llm_graph_params.hot_experts = true`) and the hot tensors are valid.
  **Prefill mode**: `hot_experts = false`, every layer runs the plain path (the scheduler streams all experts) and the
  compute buffer may clobber the hot region. The mode is part of the graph params, so graph reuse never mixes them.
- The decision is made per ubatch in `process_ubatch`, without prediction magic, from what the allocator reports:
  1. Build in decode mode when `n_tokens <= hot_threshold` (initially `n_ubatch / 2`, so a decode-mode graph can never
     need more than the plain worst-case reserve and the compute buffer is never reallocated), else in prefill mode.
  2. After `ggml_backend_sched_alloc_graph`: if the CUDA0 compute buffer handle or size changed (reallocation), re-place
     the hot tensors at the new tail and mark them dirty. Read the allocator's high-water mark for this graph (new
     customware accessors `ggml_backend_sched_get_buffer` / `ggml_backend_sched_get_buffer_max_size`, backed by
     `ggml_gallocr_get_buffer_chunk` / `ggml_gallocr_get_max_size`). If it reaches into the hot region: a decode-mode
     graph is rebuilt in prefill mode (`hot_threshold = n_tokens - 1`, so the next ubatch of that size does not retry)
     and the region is marked dirty.
  3. Before computing a decode-mode graph with a dirty region: synchronize and refill (4 GB, 0.4-2 s), once per big
     prompt, not per token.
- The hot `mul_mat_id` nodes are pinned to the GPU backend with `ggml_backend_sched_set_tensor_backend`: the scheduler's
  "run on the weights' backend" rule only fires for buffers with the WEIGHTS usage, and the compute buffer is COMPUTE.
- Re-aliasing also runs at the end of every `sched_reserve` (new scheduler, new buffers).
- One context per model when the hot set is context-owned (the server has one); logged.

## Expected

- 160k + ub 2048 + 4 GB hot: weights 4.6 + KV 2.0 + RS 0.2 + compute 5.0 (hot inside) = 11.8 GB + desktop. 169 t/s
  prefill and ~12.5 t/s decode at the user's preferred context, 3 GB of VRAM to spare for a bigger set or ub 4096.
- 120k + ub 4096 + up to 6 GB hot (compute 7.5 GB minus the decode need): prefill streams the experts half as often
  (ub 1024 -> 2048 gave +53% at 22k depth), decode gains from the bigger set (53% coverage measured for 6 GB mixed).

## Steps

1. ggml accessors (ggml-alloc, ggml-backend): buffer chunk handle and high-water mark of the last allocated graph.
2. llama-graph: `hot_experts` in the params (+ `allow_reuse`), in the context, pin the hot nodes to the GPU backend;
   qwen4exp passes the split only in decode mode.
3. llama-model: alias mode (no-alloc hot context, spec kept, `hot_alias()`, `hot_alias_bytes()`,
   `hot_alias_place()`, `hot_fill()`), the existing fill refactored to be callable from the context.
4. llama-context: state (`hot_buf`, `hot_offset`, `hot_dirty`, `hot_threshold`), placement after `sched_reserve`,
   the per-ubatch decision and refill in `process_ubatch`, log lines for mode switches and refills.
5. Launcher back to `-c 160000 -ub 2048`; measure; try 120k + ub 4096 + 6 GB.

## Verification

- Perplexity plain vs alias mode (same settings as entry 003's check); greedy log-probs on the code prompt.
- A 22k prompt followed by 300 tokens, then a 300-token follow-up: the follow-up runs in decode mode (no refill), the
  log shows exactly one refill per big prompt.
- VRAM at 160k / ub 2048 measured under 13 GB; no WDDM paging; tg and pp at 22k depth vs entry 003 (12.5 / 169).

## Execution policy

- Implementer: the planner (Fable). Commits per step on `master`, `Assisted-by: Claude Fable 5.1`,
  `Plan: 26c-trial-squad-tree`. No review mid-way; stop and report before any review when the work is measured.

## Implementation notes (what actually shipped)

- Shipped as planned: `LLAMA_HOT_EXPERTS_ALIAS=1` creates the three big hot tensors per layer in a no-alloc context of the
  model; `llama_context::hot_alias_place()` (end of `sched_reserve`) parks them in the tail of the CUDA0 compute buffer
  and `process_ubatch` decides the mode per ubatch: decode mode below `hot_threshold`, checked after allocation against
  the allocator's high-water mark (new `ggml_backend_sched_get_buffer` / `_get_buffer_max_size` accessors), rebuilt in
  prefill mode when the graph reaches into the parked experts, refilled (`hot_refill`) before the next decode-mode graph.
  The hot matmuls are pinned to the GPU with `ggml_backend_sched_set_tensor_backend` because the scheduler's "run where
  the weights are" rule only looks at WEIGHTS buffers. `ggml_backend_tensor_memset_padding` zeroes the quantization
  padding of the parked tensors after every refill (compute buffers do not do it on init).
- Deviation, and the important finding: the reason a ubatch reaches into the tail is not the QSA buffers but the
  scheduler's op-offload copies. Above `GGML_OP_OFFLOAD_MIN_BATCH` (32 by default) every used expert of every layer is
  staged in the compute buffer for the GPU matmul: 866 MiB for a 42-token ubatch, 1825 MiB for 311 tokens. So decode
  mode holds below the offload threshold, and any bigger ubatch runs in prefill mode and costs one refill (~0.66 s from
  the page cache, 1.5 s cold) afterwards. Follow-up turns of a few hundred tokens therefore either go through the
  streaming path (~46 t/s for 341 tokens) or, with a higher threshold, through the CPU with the hot split (H15, measured
  below).
- Measurements at 160k context, 2 slots, unified KV, 22.2k code prompt then 300 tokens, then a 341-token follow-up,
  all on the same evening with a background process spinning one core (see below):
  - classic hot experts (entry 003), `-ub 1024`: 101.5 t/s prefill, 9.8 / 9.6 t/s decode, follow-up 48.9 t/s, VRAM 14.5-15.0 GB.
  - alias, `-ub 2048`: 156.4 t/s prefill (+54%), 9.7 / 9.8 t/s decode, follow-up 46.5 t/s, VRAM 12.9-13.4 GB.
  - first alias run earlier the same evening: 162 t/s prefill, 12.9 t/s decode on the first request.
- The decode numbers of the evening (9.2-9.8 t/s) are below the afternoon's 12.5-12.9 for every configuration,
  including the unchanged classic one. The machine had `sst-prototype.exe` at ~100% of one core and Zed at ~60% of
  one core during these runs; with 6 barrier-synchronized llama threads on 8 cores, one straggler thread explains a
  loss of that size. The profiled run below measures barrier wait per op to confirm.
- Confirmed by the profiled run (`GGML_CPU_PROFILE=100 GGML_SCHED_PROFILE=100`, 7.1-7.4 t/s under profiling): the CPU
  ops themselves take 26 ms per token (comp 23.4 + barrier wait 2.8; q4_K gate/up carry all the wait), the n-gram
  gather 1.1 ms, but the scheduler sees CUDA0 compute at 62.5 ms per token (45 ms in the afternoon profile at the same
  depth) and the CPU splits at 30-100 ms alternating (thread wake-up latency across 49 CPU splits per token). Windows'
  per-process GPU counters during the run: `sst-prototype.exe` 58% of the 4060 Ti's 3D engine (continuous repaint),
  llama-server 46% of its CUDA engine, Zed 3%. The display GPU is time-sliced between the two, and the spinning core
  delays llama's threads at every split. So the evening's decode numbers measure the machine, not the code; the
  afternoon's 12.9 t/s stands for this configuration on a quiet box. Reported to the user: closing or throttling the
  SST Prototype window while the model runs is worth roughly a third of the token rate.
- H15 measured (same evening, same load): the 341-token follow-up with `GGML_OP_OFFLOAD_MIN_BATCH=512` stays in decode
  mode, runs on the CPU with the hot split at 52.8 t/s and needs no restore; through the streaming path it was 45-46.5
  t/s plus a 0.66 s restore. The launcher sets 512: below that a prompt chunk runs on the CPU, above it the scheduler
  streams the experts (155 t/s for 2048-token chunks). Break-even is somewhere around 600-800 tokens per chunk.
- Launcher (`release.sh`) now: `-c 160000 -ub 2048`, `LLAMA_HOT_EXPERTS_ALIAS=1`, `GGML_OP_OFFLOAD_MIN_BATCH=512`,
  the 4 GB placement; ~12.9 GB of VRAM, so there is room for a bigger placement or a second GPU consumer.

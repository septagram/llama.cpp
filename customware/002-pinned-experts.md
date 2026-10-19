# customware/002: pinned host memory for the CPU-resident experts, and a decode-time CPU/GPU expert split

## Goal

Use the PCIe link and the GPU's DMA engine as a second path to the expert weights that live in host
RAM. Two stages, each measured on its own:

1. **Pinned experts.** Let the CPU-resident expert tensors be loaded into CUDA pinned host memory
   (`-ot '.*_exps=CUDA_Host'`). Prefill already streams these tensors to CUDA0 per ubatch; from
   pageable mmap memory the copies run at ~16 GB/s (large ranges) and ~6 GB/s (scattered experts).
   From pinned memory they become real asynchronous DMA at the PCIe 4.0 x8 rate (~20+ GB/s).
   Expected: prefill +20-40% (its copy share is 76%), no change for decode by itself.
2. **Decode-time split (H9).** With pinned experts, stream roughly a third of each layer's selected
   experts to CUDA0 while the CPU computes the other two thirds, so both the cores (~30 GB/s) and the
   GPU DMA (~20 GB/s) read DRAM at the same time. Expected: CPU phase per layer 1.15 ms -> ~0.7 ms,
   i.e. about -20 ms per token (12 -> ~15 t/s). Toggleable, default off.

## Context

- Measured (entry 001): per token 83 ms with the 4060 Ti only: CPU experts ~50 ms, GPU ~28 ms,
  split syncs ~8 ms. CPU kernels stream at 29-34 GB/s of a 40.7 GB/s ceiling (membw probe).
- `cudaMallocHost` of one 48 GB block fails on Windows/WDDM, `cudaHostRegister` on a file mapping is
  unsupported, but 60 GB pinned fine as 2 GB chunks (`customware/tools/pinprobe.exe loop`).
- The loader assigns tensors to one buffer per (context, buffer type), split by the buffer type's
  `get_max_size`; the CUDA host buffer type reports SIZE_MAX today.
- With mmap on, the loader redirects any host-buffer choice to plain CPU memory; with an explicit
  override that redirect must be skipped, and the data is then copied out of the mapping by
  `ggml_backend_tensor_set` (memcpy into pinned memory, ~15-30 s for 65 GB, page cache warm).
- RAM budget: 65 GB pinned (non-pageable) + 28.8 GB n-gram table (mmap) + ~5 GB other. Fits in
  128 GB with ~20 GB slack; opt-in only.

## Steps

### 1. Pinned experts (done in the tree, to be built and measured)

- `ggml/src/ggml-cuda/ggml-cuda.cu`: `ggml_backend_cuda_host_buffer_type` gets `get_max_size`
  (env `GGML_CUDA_HOST_MAX_SIZE_MB`, default 2048), so `ggml_backend_alloc_ctx_tensors_from_buft`
  splits the expert context into 2 GB pinned buffers.
- `common/arg.cpp` and `tools/llama-bench/llama-bench.cpp`: `-ot` accepts the host buffer type
  names (`CUDA_Host`) next to the device names.
- `src/llama-model-loader.cpp`: keep an explicit host-buffer override under mmap
  (`buft_from_override`).
- Verify: `load_tensors` log shows `CUDA_Host model buffer size` lines (many, <= 2 GiB each),
  tg unchanged, pp512/pp2048 copy time per split drops (GGML_SCHED_PROFILE), pinprobe's 60 GB
  headroom holds with the model loaded.

### 2. Decode-time split, toggle `LLAMA_MOE_GPU_SHARE=<0..1>` (env, default 0 = off)

In `llm_graph_context::build_moe_ffn` (or a qwen4exp-local wrapper), when the share is > 0 and
`n_tokens == 1` (decode) and the expert tensor is host-resident:

```cpp
// static tables, built once per model: expert e goes to the GPU if e < n_gpu = share * n_expert
// ids_gpu = get_rows(map_gpu, ids)  where map_gpu[e] = e < n_gpu ? e : 0      (I32)
// ids_cpu = get_rows(map_cpu, ids)  where map_cpu[e] = e < n_gpu ? n_gpu : e  (I32)
// w_gpu   = weights * get_rows(mask_gpu, ids), w_cpu = weights * get_rows(mask_cpu, ids)  (F32 0/1)
// out     = moe(ids_gpu, w_gpu) [forced to CUDA0, weights streamed by the scheduler]
//         + moe(ids_cpu, w_cpu) [CPU]
```

- The GPU op is forced to CUDA0 through a scheduler hint: a tensor flag (`GGML_TENSOR_FLAG_OFFLOAD`)
  honoured in `ggml_backend_sched_backend_id_from_cur` before the weight-based choice.
- The scheduler's existing "copy only the used experts" path streams the selected subset. The dummy
  expert (0 or n_gpu) that masked slots map to costs one extra expert per matmul.
- The scheduler executes CUDA splits asynchronously, so the CPU split runs while the GPU streams
  and computes, and the weighted-sum on the GPU waits for both.
- Measure the share at 0.25, 0.33, 0.5 with tg128 (depth 0 and 8192) and the split profiler.

### 3. Record

- Results to `customware/bench-results.md` and the dashboard; `customware/002-*.md` implementation
  notes at seal time.

## Verification

- Same output tokens with and without pinning (llama-bench prints identical results; a fixed-seed
  completion in the server matches within sampling).
- `GGML_SCHED_PROFILE`: per-token copy time and CUDA0 compute time; per-layer CPU time from
  `GGML_CPU_PROFILE`.
- Memory: `nvidia-smi` and the Windows commit charge before and after load; the server must still
  start with `-c 80000 -ub 2048`.

## Open questions

- Whether 65 GB of pinned memory slows the desktop (non-pageable) or the n-gram table's mmap.
- Whether DRAM keeps ~40 GB/s to the cores while the DMA engine also reads it (the split's premise).

## Execution policy

- Implementer: the planner (Fable), same reasoning as entry 001: every step is chosen from the
  previous measurement.
- Commit policy: one commit per step (pinned experts; decode split), directly on `master`,
  `Assisted-by: Claude Fable 5.1`, `Plan: 26c-tiny-whacky-family`.
- Review policy: none mid-way; the session stops and reports before any review once the
  optimization work is done.

## Implementation notes (what actually shipped)

- Step 1 shipped as planned (`GGML_CUDA_HOST_MAX_SIZE_MB`, `-ot ...=CUDA_Host`, loader keeps a host-buffer
  override under mmap) plus a follow-up fix: an override to `CPU` still goes through the CPU buffer list
  and must be redirected off the host buffer type, otherwise every CPU override becomes pinned.
- Measured on this machine: pinning is mechanically excellent (CPU reads pinned q4_K at 32 GB/s, prefill
  copies of pinned tensors enqueue in ~1 ms per layer as true async DMA, ~25 GB/s implied) but Windows
  and this box's RAM make it a loss: one `cudaMallocHost` above ~2 GB fails, total pinned is capped at
  ~62 GB and shares a ~64 GB WDDM budget with VRAM backing (pinning 67 GB made a 2 GB `cudaMalloc`
  fail), and pinning 41 GB starved the file cache for the remaining mmap'd weights: down experts fell
  to 18.7 GB/s and the n-gram table lookups went from 0.09 ms to 118 ms per token. tg 12.0 -> 8.6-10.4.
- Step 2 (decode-time split with streaming) was not implemented: without pinning, scattered expert
  copies run at ~6.7 GB/s through the driver's staging path and also burn CPU-side bandwidth. The
  scheduler would additionally need the GPU-subset split on a second CUDA stream for real overlap.
- Verdict: keep the code (useful on Linux, where pinning is uncapped and `cudaHostRegister` works on
  mappings), do not use it here. The VRAM budget went to entry 003 (hot experts) instead.

## Final commit list

Sealed 2026-10-20 into `002-pinned-experts.patch`: this entry's own diff (customware/ excluded), applied after the entries before it onto upstream `6e60f3560` (tag `v0.5.0-customware.0`); `customware/seal-entries.sh` rebuilds and checks it from `customware/entry-commits.txt`.

- `5f782a926` llama : allow pinned host memory for overridden weights (-ot ...=CUDA_Host)
- `defc32010` llama : keep -ot ...=CPU off pinned memory under mmap

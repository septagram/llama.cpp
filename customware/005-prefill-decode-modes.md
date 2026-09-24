# customware/005: prefill mode / decode mode (hot experts in the tail of the compute buffer)

Status: planned (draft, not yet submitted to VibeMaster). Ordered before entry 004 because it decides where the hot
tensors live, which 004's runtime swaps must respect.

## Goal

Remove the context-vs-ubatch trade on the 4060 Ti. Today 160k context forces `-ub 1024` (110 t/s prefill) because the
QSA indexer compute buffers scale with context x ubatch (5.0 GB at 160k / ub 2048) and do not fit next to the 4 GB hot
set; the shipped launcher trades 40k of context for ub 2048 (169 t/s). The hot experts and the big prefill compute
buffer are never needed at the same time: a decode graph uses ~200 MB of the compute buffer, a big prefill ubatch does
not benefit much from the hot set (it streams all 512 experts per ubatch anyway; the 4 GB set is 6% of the bytes).

## Design

- The hot expert tensors (`ffn_{gate,up,down}_hot`) are created by the model without storage and allocated by the
  context into the **tail** of the scheduler's CUDA0 compute buffer after the worst-case reserve: `[size - hot_bytes,
  size)`, via `ggml_backend_tensor_alloc`. The maps stay ordinary model tensors (6 KB per layer).
- A ubatch runs in **decode mode** (hot split, hot tensors valid) when the graph's high-water mark on CUDA0 stays below
  the hot region, otherwise in **prefill mode** (plain path for every layer, the scheduler streams all experts; the
  compute buffer may clobber the hot region). The mode is decided before the graph is built from the ubatch size against
  a threshold calibrated at init (reserve graphs at 32, 64, ..., n_ubatch tokens and record the max size of each),
  and verified after allocation (customware accessor for the gallocr high-water mark; violation -> log and rebuild
  plain).
- After a prefill-mode ubatch the hot region is dirty. Before the next decode-mode ubatch the context re-uploads the
  hot experts from the mmap'd host tensors (4 GB: 0.4 s from the page cache, up to ~2 s from NVMe if the prefill
  stream evicted them). Restored once per big prompt, not per token.
- The mode is part of the graph params so graph reuse never mixes them. CUDA graphs are unaffected: the hot tensors'
  addresses are stable, and a decode graph after a prefill is re-captured today already.
- If the scheduler ever reallocates the CUDA0 compute buffer (a graph needs more than reserved), the hot tensors are
  re-aliased and re-uploaded (detected by comparing the buffer handle each ubatch).
- One context per model when the hot set is context-owned (the server has one); logged and asserted.

## Expected

- 160k + ub 2048 + 4 GB hot: model 4.6 + KV 2.0 + RS 0.2 + compute 5.0 = 11.8 GB + desktop -> fits with margin,
  169 t/s prefill and 12.5 t/s decode at the user's preferred context.
- 120k + ub 4096 + up to 6 GB hot (hot region = compute 7.5 GB minus decode need): prefill streams the experts half as
  often (measured ub 1024 -> 2048 gave +53% at 22k depth), decode gains from the bigger set (53% coverage measured
  for 6 GB mixed). To be measured; VRAM must stay under ~15 GB.

## Steps

1. ggml: `ggml_backend_sched_get_buffer(sched, backend)` and `ggml_backend_sched_get_buffer_max_size(sched, backend)`
   (gallocr high-water mark of the last allocated graph) as customware accessors.
2. llama-model: `LLAMA_HOT_EXPERTS_ALIAS=1` creates the hot tensors in a no-alloc context (data = null) instead of the
   CUDA0 weights context; the fill moves to the context.
3. llama-context: after `graph_reserve`, alias + fill; the calibration table; `hot_mode(ubatch)`; the dirty flag and
   the restore; `llm_graph_params.hot_mode`; `build_moe_ffn` gets `hot = nullptr` in prefill mode.
4. Launcher: back to `-c 160000` with `-ub 2048`; measure; try 120k + ub 4096 + 6 GB.

## Verification

- Perplexity plain vs alias mode (same settings as entry 003's check), greedy log-probs on the code prompt.
- A 22k prompt followed by 300 tokens, then a 300-token follow-up: the follow-up must run in decode mode (no restore).
- VRAM at 160k / ub 2048 measured under 13 GB; no WDDM paging.

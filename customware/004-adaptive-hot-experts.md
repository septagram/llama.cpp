# customware/004: adaptive hot experts (learned from the user's own sessions)

## Goal

Entry 003 puts a fixed set of experts on the 4060 Ti, chosen offline from two of my prompts: it serves ~42% of the
routed slots at 4 GB. A set fitted to the prompts it actually runs on covered 77-87%. Make the placement learn from
the user's real sessions, continuously and automatically: the server collects the router's choices for every token,
swaps experts in and out of the GPU slots at runtime, and rebalances the per-layer budgets at the next start from the
accumulated statistics. Target: coverage 60%+ on the user's workload, generation ~15 t/s at the launcher config
(12.5 today), no user action beyond running the launcher.

## Context (measured, entry 003)

- Per token the CPU streams 10 experts x ~3 MB for each of the 48 layers minus what is hot; the CPU expert bytes are
  the largest slice of a decode token (~45 ms of 80 at 42% coverage). Each 10 points of coverage is worth ~5% of tg.
- Hot tensors are compacted per layer (`ffn_{gate,up,down}_hot`, n_hot slots each) with three small map tensors
  (`ffn_hot_map`, `ffn_cold_map`, `ffn_hot_mask`) that decide which expert lives in which slot. The graph does not
  depend on the maps' contents, so rewriting them between two decodes is safe and keeps the CUDA graphs valid.
- A swap = 3 `ggml_backend_tensor_set` of ~1 MB each from the mmap'd host tensor (page-cache resident, since the
  candidate expert was just used by the CPU) + 3 map updates. ~0.3 ms per expert at PCIe speed.
- The routing tensor (`selected_experts`, I32 [n_expert_used, n_tokens]) is already computed per layer; the hot
  branch makes a contiguous copy `sel` for the map lookups.

## Design (two tiers)

1. **Collector** (always on when hot experts are configured or `LLAMA_HOT_EXPERTS_STATS` is set):
   `build_moe_ffn` marks the contiguous `sel` of every MoE layer as a graph output and records `(il, sel)` in
   `llm_graph_result::t_moe_sel`. After each ubatch `llama_context::decode` reads them back asynchronously
   (48 x n_expert_used x n_tokens ints: 2 KB per decode token, 4 MB per 2048-token prefill ubatch), synchronizes, and
   feeds them to the model's `llama_hot_experts` runtime object, which keeps two float histograms [n_layer][n_expert]:
   `recent` (halved every 4096 tokens, drives the swaps) and `total` (halved every 262144 tokens, persisted, drives the
   next start). It also counts hits (id was hot at observation time) so coverage is a logged number, not an estimate.
2. **Runtime swaps, within a layer**: every 64 decoded tokens and after every prefill ubatch, for each layer with hot
   slots: candidates = highest `recent` non-resident experts, victims = lowest `recent` residents; swap while
   `recent[cand] > 1.5 * recent[victim] + 2` (hysteresis against noise), newest residents are protected for 2 ticks
   (no ping-pong). Global budget per tick: 48 swaps (~150 MB, ~15 ms), taken by largest gain first, so the amortized
   cost is < 0.3 ms per token. Swap = copy the three expert slices into the victim's slot, update the maps in host
   mirrors and upload the three 2 KB map tensors of that layer. Done under a mutex, only between graph computations.
3. **Restart rebalance**: `LLAMA_HOT_EXPERTS_STATS=<file>` names the stats file. The server rewrites it (tmp + rename)
   every 8192 tokens and at exit with `total`, one line per layer `<layer> <count x n_expert>` (same format as the
   `cpu-prof-experts` lines, so `customware/expert-place.py` can read it too). At load, if the file exists, the
   placement is computed in C++ from it (byte-aware greedy: score = count / expert bytes, until `LLAMA_HOT_EXPERTS_MIB`
   is used; layers 2/4/30/46/47 have bigger experts) and `LLAMA_HOT_EXPERTS=<placement file>` is only the fallback
   for a fresh install. Logged: layers, experts, MiB, predicted coverage on the stats.

## Steps

### 1. `src/llama-hot-experts.{h,cpp}` (new, customware): the runtime object

`llama_hot_experts` owned by `llama_model_base::impl`, accessor `hot_experts()` (const model, mutable state, as with
mlock). Holds per-layer: pointers to the six tensors + the full host tensors, `n_hot`, host mirrors of the maps and the
slot -> expert table, protection counters; the two histograms; token counters; the stats path/budget; a mutex.
API: `configure(model, spec)` after load, `observe(il, const int32_t * ids, n)`, `tick(n_tokens, is_prefill)` (decay,
swaps, flush), `flush()`, `placement_from_stats(...)` (static, used by the loader), `log_summary()`.

### 2. Loader: placement from stats

In `load_tensors`, replace `llama_hot_experts_read()` by a function that returns the placement from the stats file
when present (needs the per-layer expert bytes: `ffn_{gate,up,down}_exps->nb[2]`, available after
`done_getting_tensors`) and from the placement file otherwise. Keep the tensor creation and the fill; after the fill,
construct the runtime object.

### 3. Graph + context: export the routing ids

- `llm_graph_result`: `std::vector<std::pair<int, ggml_tensor *>> t_moe_sel;` cleared in `reset()`, set as outputs in
  `set_outputs()`.
- `build_moe_ffn`: when `model.hot_experts()` collects, make `sel` contiguous for every MoE layer (not only the split
  ones), push it. The hot branch reuses the same tensor.
- `llama_context::decode`, after each successful `process_ubatch`: `ggml_backend_tensor_get_async` each `sel` into a
  staging buffer, `ggml_backend_sched_synchronize`, then `observe` + `tick`. The server samples right after decode,
  which synchronizes anyway, so the extra sync costs nothing there.

### 4. Launcher and tools

- `release.sh` template: `LLAMA_HOT_EXPERTS_STATS=../llama.cpp-customware/hot-experts.stats`,
  `LLAMA_HOT_EXPERTS_MIB=4000`, keep `LLAMA_HOT_EXPERTS` as the first-start fallback.
- `expert-place.py`: accept the stats file (already the same line format; verify).
- Log lines (`hot experts: ...`) every 8192 tokens: coverage over the window, swaps done, bytes moved.

### 5. Measure

- Server at the launcher config, code prompt: run 1 with the shipped mixed placement (coverage ~42%), run 2 after the
  swaps settled, then the prose prompt (adaptation speed: coverage per 1k tokens). Then restart with the stats file and
  confirm the computed placement's coverage on the same prompt (should start where run 2 ended).
- tg/pp at 22k depth, 2 slots, vs entry 003 (12.5 / 169). Expect tg 14-15.
- Perplexity with the collector + swaps on vs off (same settings as the 003 check): must match within the error bar.

## Verification

- Coverage logged by the collector rises monotonically on a repeated prompt and does not collapse on a prompt switch.
- The stats file round-trips (`expert-place.py` reads it; the C++ placement from it lists the same top experts).
- Greedy log-probs after a few hundred swaps stay within the GPU/CPU accumulation noise (~0.003 nats) of the fixed
  placement on the same prompt.
- No swap ever touches a tensor while a graph is running (mutex + only inside `decode` after the sync).

## Open questions / risks

- `-np 2`: two sequences per ubatch mix into the same histogram; fine for placement, noted.
- Page faults on the swap source if the candidate expert was evicted from the page cache (unlikely: it was just used).
  Bounded by the per-tick budget either way.
- Persistence across model files: the stats file carries the model's expert count and layer count in its header;
  mismatch -> ignored.

## Execution policy

- Implementer: the planner (Fable). Commits per step on `master`, `Assisted-by: Claude Fable 5.1`,
  `Plan: 26c-tremor-fade-shell`. No review mid-way; stop and report before any review when the work is measured.

## Implementation notes (what actually shipped)

- Shipped as planned: `src/llama-hot-experts.{h,cpp}` (histograms, in-layer swaps, stats file, placement from stats),
  the routing ids exported per layer as graph outputs (`llm_graph_result::t_moe_sel`, built when the model has the
  runtime object), read back in `llama_context::hot_observe` after each ubatch, `tick` runs the swap policy every 64
  decoded tokens and after every prefill ubatch. The slot lists the runtime edits are the model's placement spec, so
  a refill of parked tensors (entry 005) copies the current set; while the tensors are parked out, a swap updates the
  spec and the maps only. Env: `LLAMA_HOT_EXPERTS_STATS=<file>` (read at start, rewritten every 8192 tokens and at
  exit, `cpu-prof-experts:` line format so `expert-place.py` reads it too), `LLAMA_HOT_EXPERTS_MIB=<n>` (placement
  from the stats within that budget; the placement file is the first-start fallback), `LLAMA_HOT_EXPERTS_SWAP=0`
  (collect only), `LLAMA_HOT_EXPERTS_LOG=<tokens>` (coverage / swap log cadence).
- First run at the launcher config (160k, alias, 4 GB): mechanics all work. Code prompt twice, prose prompt twice:
  1392 swaps, 48 per tick during prefill (maps only, 0 bytes), 141-1417 MiB of copies during the decode windows, four
  refills at 0.65-0.70 s, no errors. The stats file (97 KB) was written, and a restart with `LLAMA_HOT_EXPERTS_MIB=4000`
  and no placement file computed its own set from it (1352 experts on 47 layers, predicted coverage 32.6%).
- Deviation found in that run: a 22k prefill floods the recent histogram (half-life 4096 tokens) with prefill routing,
  so the set is tuned to the document by the time generation starts, and the decode-time coverage of the shipped
  placement dropped rather than rose. Prefill observations (ubatches of 32+ tokens) now weigh 1/16 in both histograms
  and in the decay clock, so generation dominates what the set adapts to; that is also what the offline placements
  were measured on (the CPU profiler only saw decode). The log cadence for the test is 256 tokens so the decode
  windows are visible.
- Rerun with the prefill weighting and 256-token log windows (the box was quieter by then, ~12 t/s decode in every
  run): code prompt 163 t/s prefill / 12.0 decode, cached 12.1, prose 190 / 11.9, cached 12.2; restart placed from
  the stats: 170 / 11.9 with 1 swap in the decode window (converged). Decode-window coverage: 27.7% and 29.3% on the
  code prompt, 15.8% and 19.3% after the switch to prose, 24.1% after the restart.
- Second flaw, the cold start: the shipped placement serves 41.5% of decode routing when fixed, but a fresh session
  starts with empty histograms, so its experts have no evidence to defend them and the 768 prefill-time swaps (even at
  1/16 weight) reshuffle the set before generation starts; 300 generated tokens are not enough to recover. Fix: ship
  the statistics the placement was learned from instead of the placement. `customware/stats-seed.py` merges the
  offline histograms (x4) into `hot-experts-seed.stats`; `release.sh` installs it as the stats file when none has been
  learned yet (a learned file is never overwritten), the placement at start comes from it, and `load_stats` seeds the
  fast histogram with a quarter of the slow one, so the resident set is defended by ~2.5k tokens' worth of evidence
  and a new workload takes over within a few thousand generated tokens.
- The greedy log-prob comparison on the released launcher (alias, swaps on, short prompt through the CPU split):
  41 of 48 tokens identical to the classic hot run, mean |delta| 0.015 nats over the prefix (0.002 vs plain up to the
  known near-tie at token 2). Small-context perplexity runs could not exercise the parked path (the compute buffer at
  8k-32k context is 290-980 MiB, too small to park even a 600 MiB set); each of them matched its reference exactly.
- With the seed (160k, alias, 256-token windows, code prompt twice): decode-window coverage 37.0% then 39.4% with 10
  and 4 swaps per window (stable, slowly improving), 170.6 t/s prefill, 12.61 / 12.97 t/s decode; 142 swaps during the
  first prefill instead of 768. The released launcher (`verify-launcher.sh`): placement from the seed = the shipped
  1348 experts (predicted 42.4%), 181 / 12.2 t/s on the first run of the evening, 171 / 12.8 after the seed fix,
  12.9 GB VRAM, and the learned stats file (79 KB after one session) survives a re-release.
- What is not measured yet: the long-run gain of a set learned from the user's own sessions (the offline mixed
  placement was fitted to the two test prompts, so on them the adaptive set cannot beat it by much). The mechanism
  and its cost are: ~2 MB of PCIe copies per generated token at the swap budget, one 0.65 s restore per big prompt.

## Final commit list

- 35299131e llama : adaptive hot experts (LLAMA_HOT_EXPERTS_STATS, runtime swaps, placement from stats)
- a14357033 customware: seed the adaptive hot experts with the placement's own statistics

## First agentic trace (2026-09-25, pi driving the 120k launcher on a Rust refactor task, 10 minutes)

- 16 requests, 7009 generated tokens, 49k prefill tokens, conversation depth up to 44k. Live decode-window coverage
  36-47% with the 6 GB set placed from the learned stats; the run crashed on pi's side (its bash tool spawns
  `/bin/sh`, which does not exist when pi is started from cmd), not on the server's.
- Simulator sweep on that trace (placement approximated from the post-run stats): fixed set 39.1%, half-life 1024
  41.7% (0.9 MB of copies per generated token), 2048 40.8%, 4096 40.1%, 16384 39.6%, hindsight optimum 43.6%. On
  agentic traffic a shorter half-life is worth about +1.5 points of coverage (~1% of decode) for six times the
  copies, still cheap. The band is 2-4 points under the ceiling; the set and the per-layer budgets matter more
  than the half-life.
- Decode slows with depth: 83 ms per token at ~6k, 86 at ~12k, 95 at ~44k (the synthetic 22k prompt runs at 73).
  Suspects: the QSA indexer scoring and top-k selection over the whole context per token per layer (a full argsort
  of 44k positions per head), and the n-gram table under memory pressure. To profile at depth (H16).
- Prompt cache entries are huge with the hybrid model's checkpoints: one idle 12k-token slot saved 2.5 GB (20
  checkpoints of 112 MB). With `--cache-ram 16384` that is RAM taken from the page cache the cold experts live in;
  `--cache-ram 4096` or fewer checkpoints may be the better daily setting (H17, to measure).

# customware/001: native build, profiling instrumentation, baselines for qwen4exp CPU-expert generation

## Goal

Find out, with numbers taken on this machine, where the ~143 ms per generated token goes when
Qwen3.8-Flash-Next (arch `qwen4exp`, UD-Q4_K_XL) runs with `-ngl 99` and the routed experts on
the CPU. Every later optimization entry (kernel work, split/launch overhead, repack, speculation)
is chosen from this profile rather than from architecture reasoning.

Expected outcome: a native build of this repo that is at least as fast as the prebuilt binary,
two env-gated profilers in ggml, a baseline table (prebuilt vs native, thread/knob sweep), and a
per-token breakdown: CPU expert matmul time vs GPU split time vs copy/sync time.

## Context (established before this plan)

- Daily-driver binary is prebuilt (build 10757, Clang, generic `haswell` CPU variant, OpenMP).
- With `-ngl 99` and only `*_exps` on CPU, the DeltaNet scan runs on the GPU. CPU work per token
  is ~38 layers x (gate/up `mul_mat_id` on Q4_K + down `mul_mat_id` on Q5_1 + swiglu) plus a
  24-row gather from the 28.8 GB IQ4_NL n-gram table. ~1.3 GB weights per token, ~9 GB/s at 7 t/s.
- Prefill offloads expert matmuls to CUDA0 for batches >= 32 tokens by copying each layer's
  expert weights over PCIe per ubatch, so prefill is a copy problem, not a CPU one.
- CUDA graphs are cached per split key in this tree; whether they actually replay here is unknown.
- The run script's `ple_ngram_embd=CPU` override matches no tensor (the table is named
  `per_layer_token_embd.weight`); where it lands is to be read from the load log.

## Steps

### 1. Native build (`customware/build-win.cmd`)

- clang-cl 21 for C/C++ with `/clang:-march=znver2`, VS2019 `cl` as the nvcc host compiler,
  CUDA 12.3, `CMAKE_CUDA_ARCHITECTURES=75-real;89-real`, static libs, tools only.
- `GGML_CCACHE=OFF` (ggml auto-detects the sccache on PATH; it cannot parse nvcc command lines).
- OpenMP is not found for clang-cl by CMake, so this build uses ggml's own threadpool. Recorded as
  a variable to compare against the OpenMP prebuilt; not fixed in this entry.
- Deliverables: `build/bin/llama-bench.exe`, `build/bin/llama-server.exe`.

### 2. Baseline benchmarks (`customware/bench.sh`)

llama-bench, fixed config identical across arms: `-ngl 99 -sm layer -fa 1 -ctk q8_0 -ctv q8_0 -t 6
-ot 'blk\.(4[2-6])\..*_exps=CUDA0,blk\.(39|4[0-1])\..*_exps=CUDA1,.*_exps=CPU' -r 5`, tests
`tg128` and `pp512`. Arms:

- A. prebuilt `C:\dev\llama.cpp\llama-bench.exe`
- B. native build

Both are run after the page cache is warm (first run discarded). Results appended to
`customware/bench-results.md` and the dashboard.

### 3. Profiling instrumentation (env-gated, zero cost when unset)

3a. `GGML_SCHED_PROFILE=N` in `ggml/src/ggml-backend.cpp`, `ggml_backend_sched_compute_splits`:

```cpp
struct sched_prof_split { int backend_id; int n_nodes; const char * name; int64_t us_copy; int64_t us_comp; };
// when enabled: time input copies, then graph_compute_async + ggml_backend_synchronize(split_backend)
// accumulate per backend: n_splits, us_copy, us_comp; keep per-split rows for the dump
// after the loop: every N-th graph compute print one summary line, and the per-split rows
```

Output: `sched-prof: graph #k total=... ms splits=... | CUDA0 n=.. copy=.. comp=.. | CUDA1 ... | CPU ...`
plus the per-split rows on the dump iterations. The forced sync only happens when enabled.

3b. `GGML_CPU_PROFILE=N` in `ggml/src/ggml-cpu/ggml-cpu.c`, `ggml_graph_compute_thread` (thread 0
only): per node, `t0` before compute, `t1` after compute, `t2` after the barrier. Accumulate into a
static table keyed by (op, src0 type, batch class: rows of src1 == 1 vs > 1): `us_compute`,
`us_barrier`, `count`. Print the table every N CPU graph computes and from `ggml_threadpool_free`.
Fused-op paths are attributed to the first node's op.

Neither profiler changes numerics. Both are a single `static int` env read per process.

### 4. Knob sweep with llama-bench (native build)

- threads: `-t 4,5,6,7,8`
- `--prio 0` vs `--prio 2`, `--poll 50` vs `--poll 0` (threadpool build only)
- prefill ubatch: `-p 2048 -ub 512,1024,2048 -b 2048`
- `-mmp 0` vs `-mmp 1` (`--load-mode none` vs mmap) once, tg only
- one `-d 16384` depth run to see how much attention/indexer cost grows with context

### 5. Record

- `customware/bench-results.md` gets every table; `perf-dashboard.html` gets the summary and the
  hypothesis verdicts (H1 kernel-bound, H2 launch/sync-bound, H4 prefill/ub, H5 threads, H7 native).

## Verification

- `llama-bench` prints identical token counts for A and B; server smoke test: start
  `build/bin/llama-server.exe` with the daily config on a spare port, one chat completion returns
  coherent text.
- Profilers: run tg128 with and without the env vars; tg t/s within noise when unset; with
  `GGML_SCHED_PROFILE=1` the per-graph totals sum to about the measured ms/token.
- Build is clean of warnings in the touched files.

## Open questions

- Where does `per_layer_token_embd.weight` land (CPU mmap vs CUDA_Host pinned copy)? Read the load
  log; it decides whether the pinned-host idea for prefill is already partly in effect.
- Whether the threadpool build's spin-polling between GPU splits hurts or helps vs OpenMP.

## Execution policy

- Implementer: the planner (Fable) implements directly. This is a measure-modify-measure loop where
  every change is chosen from the previous number; handing the loop to a lower-class subagent would
  cost the context that makes the decisions. Subagents are used for independent side tasks only.
- Commit policy: one commit per completed step (build script, bench script, each profiler, results),
  directly on `master`, `Assisted-by: Claude Fable 5.1` trailer, `Plan: 26c-tiny-chaps-evade`.
- Review policy: per the user's instruction, no review is requested mid-way; the session stops and
  reports before any review once the optimization work is done. This entry is sealed as
  `customware/001` together with the other entries at the end.

## Implementation notes (what actually shipped)

- Build: three toolchain detours, all captured in `build-win.cmd`: `GGML_CCACHE=OFF` (ggml auto-picked the
  sccache on PATH as the nvcc launcher), MSVC default flags kept next to `-march=znver2` (overriding
  `CMAKE_CXX_FLAGS` had dropped `/EHsc`), and clang-cl 21 compiling against the VS2026 (14.51) STL via
  `-imsvc` + `-fms-compatibility-version=19.51` because it rejects a cast in the VS2019 (14.28) STL
  `<functional>`; nvcc keeps cl 14.28. OpenMP is not found for clang-cl, so the build uses ggml's own
  threadpool (measured: no difference vs the OpenMP prebuilt).
- Profilers landed as planned (`GGML_SCHED_PROFILE`, `GGML_CPU_PROFILE`), plus a per-layer expert usage
  histogram in the CPU profiler (`cpu-prof-experts` lines) that the plan did not have.
- llama-bench `-ot` uses `;` between overrides inside one set and `,` to start another test variant; the
  bench wrapper documents that. Results are recorded as jsonl with per-sample t/s (`bench-record.py`)
  because run-to-run spread was ~10% on tg with the two-GPU config.
- Findings that changed the direction (all in `bench-results.md` and the dashboard):
  - H1 refuted: CPU expert kernels stream at 29-34 GB/s of a 40.7 GB/s DRAM ceiling (membw probe);
    they were 29% of the token with both GPUs, ~55% with one.
  - The RTX 2060 SUPER was a net loss: tg 7.8 -> 12.0 t/s by leaving it out (`-dev CUDA0`) and putting
    4-5 expert layers on the 4060 Ti instead. Cross-GPU hops were a minor part (8.1 with hops removed).
  - Prefill is 76% PCIe copies of expert weights, once per ubatch: `-ub 2048` gives 312 t/s at depth 0
    vs 106 (both GPUs, ub 512). At 22k depth in the server: ~190 t/s and ~11 t/s generation.
  - CUDA graphs are active (12 ms/token saved); threads 6 stays optimal; poll level irrelevant;
    large pages irrelevant; native Zen 2 build equal to the prebuilt.
- Not done from the plan: `-mmp 0` vs `-mmp 1` (superseded by the pinned-memory work in entry 002).

## Final commit list

Sealed 2026-10-20 into `001-profilers-and-baseline.patch`: this entry's own diff (customware/ excluded), applied after the entries before it onto upstream `6e60f3560` (tag `v0.5.0-customware.0`); `customware/seal-entries.sh` rebuilds and checks it from `customware/entry-commits.txt`.

- `0721c0eb9` customware: fix the Windows build script for this toolchain (notes only)
- `93f079ad6` ggml : add env-gated split and per-op profilers, bench tooling
- `3a43bb23c` customware: DRAM/pinning probes and the first benchmark results (notes only)

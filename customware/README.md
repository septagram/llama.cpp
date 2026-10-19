# customware/

This directory captures the fork-local patches that ride on top of
upstream. Each numbered entry has two files:

- `NNN-<slug>.md` — the plan, including any deviations noted under
  "Implementation notes" after the work shipped.
- `NNN-<slug>.patch` — this entry's own diff (customware/ excluded), so
  the patches apply in numeric order onto the fork point (upstream
  `6e60f3560`, tag `v0.5.0-customware.0`); the primary rebase tool on the
  next upstream update. Each entry's "Final commit list" names its commits
  and any code that travels in another entry's patch (history sometimes
  interleaved two entries; 005 has no patch of its own for that reason).

`entry-commits.txt` maps every commit to its entry; `seal-entries.sh`
regenerates the patches from it on a scratch index and fails unless they
apply in order and reproduce `master`'s code exactly; `seal-notes.py`
writes the "Final commit list" sections.

Tag format on `master`: `v<upstream-version>-customware.<n>`.
Branches named `<upstream-version>` (no `customware` suffix) are
created at the start of an `/customware update` and preserve the
pre-update state.

This fork is tuned for one machine: Ryzen 7 3800X + RTX 4060 Ti 16 GB +
RTX 2060 SUPER 8 GB, 128 GB DDR4-3600, Windows 10 (Linux support kept).
Target workload: Qwen3.8-Flash-Next (qwen4exp) UD-Q4_K_XL with experts on CPU.
`build-win.cmd` is the build entry point on Windows.

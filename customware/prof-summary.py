#!/usr/bin/env python3
"""Summarize GGML_SCHED_PROFILE / GGML_CPU_PROFILE output captured from stderr.

usage: prof-summary.py <log> [--skip N] [--min-splits N]
  --skip N        ignore the first N sched-prof graphs (warmup / prompt)
  --min-splits N  only count graphs with at least N splits (drop tiny graphs like sampling)
Prints mean per-graph totals and per-backend copy/compute time, and the per-split rows of the
largest dumped graph grouped by backend. The cpu-prof table (last one in the log) is echoed as-is.
"""
import re, sys, collections, statistics

args = sys.argv[1:]
path = args[0]
skip = int(args[args.index('--skip') + 1]) if '--skip' in args else 0
min_splits = int(args[args.index('--min-splits') + 1]) if '--min-splits' in args else 2

graph_re = re.compile(r'sched-prof: graph (\d+) total=([\d.]+) ms splits=(\d+)(.*)')
be_re = re.compile(r'\| (\S+) n=(\d+) nodes=(\d+) copy=([\d.]+) comp=([\d.]+)')
split_re = re.compile(r'sched-prof:\s+split\s+(\d+) (\S+)\s+nodes=\s*(\d+) copy=\s*([\d.]+) us comp=\s*([\d.]+) us\s+(.*)')

graphs = []
cur_rows = []
last_dump = []
cpu_prof = []
for line in open(path, errors='replace'):
    m = graph_re.match(line.strip())
    if m:
        g = {'id': int(m.group(1)), 'total': float(m.group(2)), 'splits': int(m.group(3)), 'be': {}}
        for b in be_re.finditer(m.group(4)):
            g['be'][b.group(1)] = (int(b.group(2)), int(b.group(3)), float(b.group(4)), float(b.group(5)))
        graphs.append(g)
        if cur_rows:
            last_dump = cur_rows
            cur_rows = []
        continue
    m = split_re.match(line)
    if m:
        cur_rows.append((int(m.group(1)), m.group(2), int(m.group(3)), float(m.group(4)), float(m.group(5)), m.group(6).strip()))
        continue
    if line.startswith('cpu-prof:'):
        cpu_prof.append(line.rstrip())

sel = [g for g in graphs[skip:] if g['splits'] >= min_splits]
print(f'{len(graphs)} graphs in log, {len(sel)} selected (skip={skip}, min_splits={min_splits})')
if sel:
    tot = [g['total'] for g in sel]
    print(f'per-graph total: mean {statistics.mean(tot):.2f} ms, median {statistics.median(tot):.2f}, min {min(tot):.2f}, max {max(tot):.2f}, splits {sel[0]["splits"]}..{sel[-1]["splits"]}')
    names = sorted({n for g in sel for n in g['be']})
    for n in names:
        rows = [g['be'][n] for g in sel if n in g['be']]
        print(f'  {n:6s} splits={statistics.mean(r[0] for r in rows):6.1f} nodes={statistics.mean(r[1] for r in rows):7.1f} copy={statistics.mean(r[2] for r in rows):7.2f} ms comp={statistics.mean(r[3] for r in rows):7.2f} ms')
if last_dump:
    print(f'\nlast dumped graph: {len(last_dump)} splits')
    by_be = collections.defaultdict(lambda: [0, 0.0, 0.0, 0])
    for _, be, nodes, copy, comp, _ in last_dump:
        by_be[be][0] += 1; by_be[be][1] += copy; by_be[be][2] += comp; by_be[be][3] += nodes
    for be, (n, copy, comp, nodes) in by_be.items():
        print(f'  {be:6s} splits={n:4d} nodes={nodes:5d} copy={copy/1000:7.2f} ms comp={comp/1000:7.2f} ms  avg comp/split={comp/n:7.1f} us')
    top = sorted(last_dump, key=lambda r: -r[4])[:12]
    print('  slowest splits:')
    for i, be, nodes, copy, comp, name in top:
        print(f'    #{i:3d} {be:6s} nodes={nodes:4d} copy={copy:8.1f} comp={comp:8.1f}  {name}')
if cpu_prof:
    # print only the last table
    starts = [i for i, l in enumerate(cpu_prof) if 'graph computes' in l]
    print('\n' + '\n'.join(cpu_prof[starts[-1]:] if starts else cpu_prof))

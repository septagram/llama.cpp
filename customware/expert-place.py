#!/usr/bin/env python
"""Choose which routed experts to keep on the GPU from routing statistics.

usage: expert-place.py --model <first gguf split> --budget-mib N --out <placement.txt> <log-or-stats>...
  <log-or-stats>  files with `cpu-prof-experts: <layer> <count>...` lines (server/bench stderr with
                  GGML_CPU_PROFILE set); the last block per file wins, counts are cumulative
  --layers-off L  comma-separated layers whose experts already live on the GPU (excluded)
  --min-per-layer K  floor of K experts per layer before the global greedy fill (default 0)
Output: one line per layer: `<layer> <id> <id> ...`, and a coverage report on stderr.
"""
import argparse, sys, collections
import numpy as np
sys.path.insert(0, 'gguf-py')
from gguf import GGUFReader

ap = argparse.ArgumentParser()
ap.add_argument('--model', required=True); ap.add_argument('--budget-mib', type=float, required=True)
ap.add_argument('--out', required=True); ap.add_argument('--layers-off', default='')
ap.add_argument('--min-per-layer', type=int, default=0); ap.add_argument('logs', nargs='+')
a = ap.parse_args()

# bytes per expert per layer = sum over gate/up/down of nbytes / n_expert
per_expert = {}
base = a.model
for i in range(1, 9):
    p = base.replace('00001-of', '%05d-of' % i)
    try: r = GGUFReader(p)
    except Exception: break
    for t in r.tensors:
        if '_exps' in t.name and t.name.startswith('blk.'):
            il = int(t.name.split('.')[1]); n_exp = int(t.shape[2])
            per_expert[il] = per_expert.get(il, 0) + int(t.n_bytes) // n_exp
print(f'expert bytes per layer: min {min(per_expert.values())/2**20:.2f} MiB, max {max(per_expert.values())/2**20:.2f} MiB', file=sys.stderr)

stats = collections.defaultdict(lambda: None)
for path in a.logs:
    block = {}
    for l in open(path, errors='replace'):
        if l.startswith('cpu-prof-experts:'):
            parts = l.split(); il = int(parts[1]); arr = np.array([int(x) for x in parts[2:]], dtype=np.int64)
            if il == 0 and 0 in block and len(block) > 1: block = {}
            block[il] = arr
    for il, arr in block.items():
        stats[il] = arr if stats[il] is None else stats[il] + arr
off = {int(x) for x in a.layers_off.split(',') if x}
layers = [il for il in sorted(stats) if il not in off]
total = sum(int(stats[il].sum()) for il in layers)

chosen = {il: [] for il in layers}
budget = a.budget_mib * 2**20
used = 0.0
if a.min_per_layer:
    for il in layers:
        for e in np.argsort(-stats[il])[:a.min_per_layer]:
            if stats[il][e] > 0: chosen[il].append(int(e)); used += per_expert[il]
items = sorted(((int(stats[il][e]) / per_expert[il], il, int(e)) for il in layers for e in range(len(stats[il])) if stats[il][e] > 0), reverse=True)
taken = {(il, e) for il in layers for e in chosen[il]}
for _, il, e in items:
    if (il, e) in taken: continue
    if used + per_expert[il] > budget: continue
    chosen[il].append(e); taken.add((il, e)); used += per_expert[il]
hit = sum(int(stats[il][chosen[il]].sum()) for il in layers)
with open(a.out, 'w') as f:
    for il in layers:
        if chosen[il]: f.write(f'{il} ' + ' '.join(str(e) for e in sorted(chosen[il])) + '\n')
print(f'{sum(len(v) for v in chosen.values())} experts on {len(layers)} layers, {used/2**20:.0f} MiB of {a.budget_mib:.0f}, covering {hit/total:.1%} of routed slots', file=sys.stderr)
for il in layers:
    c = chosen[il]; s = int(stats[il].sum())
    print(f'  layer {il:2d}: {len(c):3d} experts, {int(stats[il][c].sum())/s if s else 0:5.1%} of its routing', file=sys.stderr)

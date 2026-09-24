#!/usr/bin/env python
"""Merge expert-routing histograms (server/bench logs with `cpu-prof-experts: <layer> <count>...` lines, or
llama-hot-experts stats files) into one stats file that seeds LLAMA_HOT_EXPERTS_STATS on a fresh install.

  python customware/stats-seed.py --out <stats> [--scale S] [--n-layer 48] [--n-expert 512] <inputs...>

The last block of lines per input wins (the profiler prints cumulative counts). Counts are summed across inputs and
multiplied by --scale: the seed must outweigh the first few hundred generated tokens of a new session, otherwise the
runtime swaps churn the shipped set out before it has proven itself.
"""
import argparse, sys

def read_hist(path, n_layer, n_expert):
    blocks = []  # list of dict layer -> counts, one per block
    cur = {}
    with open(path, encoding='utf-8', errors='replace') as f:
        for line in f:
            i = line.find('cpu-prof-experts:')
            if i < 0:
                continue
            parts = line[i + len('cpu-prof-experts:'):].split()
            if not parts:
                continue
            try:
                il = int(parts[0])
                counts = [float(x) for x in parts[1:1 + n_expert]]
            except ValueError:
                continue
            if il in cur:          # a layer repeats: a new block starts
                blocks.append(cur)
                cur = {}
            cur[il] = counts
    if cur:
        blocks.append(cur)
    return blocks[-1] if blocks else {}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    ap.add_argument('--scale', type=float, default=1.0)
    ap.add_argument('--n-layer', type=int, default=48)
    ap.add_argument('--n-expert', type=int, default=512)
    ap.add_argument('inputs', nargs='+')
    a = ap.parse_args()

    total = [[0.0] * a.n_expert for _ in range(a.n_layer)]
    for path in a.inputs:
        h = read_hist(path, a.n_layer, a.n_expert)
        if not h:
            print('no histogram in', path, file=sys.stderr)
            continue
        for il, counts in h.items():
            if 0 <= il < a.n_layer:
                for e, c in enumerate(counts[:a.n_expert]):
                    total[il][e] += c
        print('%s: %d layers, %.0f slots' % (path, len(h), sum(sum(c) for c in h.values())))

    with open(a.out, 'w', encoding='utf-8', newline='\n') as f:
        f.write('# llama-hot-experts-stats n_layer=%d n_expert=%d tokens=0\n' % (a.n_layer, a.n_expert))
        f.write('# seed merged by customware/stats-seed.py (scale %g) from: %s\n' % (a.scale, ' '.join(a.inputs)))
        for il in range(a.n_layer):
            f.write('cpu-prof-experts: %d %s\n' % (il, ' '.join(str(int(round(c * a.scale))) for c in total[il])))
    print('wrote', a.out, '%.0f slots' % (sum(sum(r) for r in total) * a.scale))

if __name__ == '__main__':
    main()

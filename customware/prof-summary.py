#!/usr/bin/env python
"""Summarise a server log written with GGML_SCHED_PROFILE=N GGML_SCHED_PROFILE_NODES=1 (customware/test-depth-prof.sh).

usage: prof-summary.py <log> <label>:<first line>-<end line> ...
Per range: the decode graphs (ubatches of one token) - mean graph time, per-backend copy/compute, mean set_inputs -
and the node ranking of the dumped decode graphs, averaged per graph."""
import re
import sys
from collections import defaultdict

RE_INPUTS = re.compile(r"sched-prof: set_inputs ([0-9.]+) ms for (\d+) tokens")
RE_NODE   = re.compile(r"sched-prof:\s+node (\S+)\s+n=\s*(\d+) total=\s*([0-9.]+) us mean=\s*([0-9.]+) us\s+(.*)$")
RE_GRAPH  = re.compile(r"sched-prof: graph (\d+) total=([0-9.]+) ms splits=(\d+)(.*)$")
RE_BACK   = re.compile(r"\| (\S+) n=(\d+) nodes=(\d+) copy=([0-9.]+) comp=([0-9.]+)")


def summarise(lines, label):
    ntok = None
    t_inputs = []
    graphs = []          # (total_ms, {backend: (copy, comp)})
    dumps = []           # list of {key: (n, total_us)}
    cur_nodes = {}
    for line in lines:
        m = RE_INPUTS.search(line)
        if m:
            ntok = int(m.group(2))
            if ntok == 1:
                t_inputs.append(float(m.group(1)))
            continue
        m = RE_NODE.search(line)
        if m:
            # unnamed nodes (node_1234) differ only by their number: one row per op
            key = re.sub(r"\bnode_\d+$", "?", m.group(5).strip())
            prev = cur_nodes.get(key)
            n, us = int(m.group(2)), float(m.group(3))
            if prev:
                n, us = prev[1] + n, prev[2] + us
            cur_nodes[key] = (m.group(1), n, us)
            continue
        m = RE_GRAPH.search(line)
        if m:
            if ntok == 1:
                backs = {b.group(1): (float(b.group(4)), float(b.group(5))) for b in RE_BACK.finditer(m.group(4))}
                graphs.append((float(m.group(2)), backs))
                if cur_nodes:
                    dumps.append(cur_nodes)
            cur_nodes = {}
    print(f"== {label}: {len(graphs)} decode graphs, {len(dumps)} dumped node by node")
    if not graphs:
        return
    # the node-by-node graphs are slower; report the mean of the others as the graph time
    plain = [g for g in graphs if True]
    mean_total = sum(g[0] for g in plain) / len(plain)
    backs = defaultdict(lambda: [0.0, 0.0, 0])
    for _, b in plain:
        for name, (copy, comp) in b.items():
            backs[name][0] += copy
            backs[name][1] += comp
            backs[name][2] += 1
    back_s = "  ".join(f"{n}: copy {v[0]/v[2]:.2f} comp {v[1]/v[2]:.2f} ms" for n, v in backs.items())
    inp = sum(t_inputs) / len(t_inputs) if t_inputs else 0.0
    print(f"   graph mean {mean_total:.2f} ms (with a sync per split; the dumped ones included)  {back_s}  set_inputs {inp:.2f} ms")
    if not dumps:
        return
    acc = defaultdict(lambda: [None, 0, 0.0])
    for d in dumps:
        for key, (backend, n, us) in d.items():
            acc[key][0] = backend
            acc[key][1] += n
            acc[key][2] += us
    rows = sorted(acc.items(), key=lambda kv: -kv[1][2])
    total = sum(v[2] for v in acc.values()) / len(dumps)
    print(f"   node ranking, per dumped graph (sum {total/1000:.2f} ms with a sync after every node):")
    for key, (backend, n, us) in rows[:30]:
        print(f"   {backend:6s} n={n/len(dumps):5.1f} {us/len(dumps)/1000:8.3f} ms  {us/n:8.1f} us/node  {key}")


def main():
    log = sys.argv[1]
    with open(log, encoding="utf-8", errors="replace") as f:
        lines = f.readlines()
    for arg in sys.argv[2:]:
        label, rng = arg.split(":")
        a, b = (int(x) for x in rng.split("-"))
        summarise(lines[a:b], label)


if __name__ == "__main__":
    main()

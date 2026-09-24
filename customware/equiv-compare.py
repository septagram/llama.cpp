#!/usr/bin/env python
"""Compare two greedy chat completions with top log-probs: token agreement and, at the first divergence, the
competing candidates' log-probs on both sides. usage: equiv-compare.py <a.json> <b.json>"""
import json, sys
def load(p):
    r = json.load(open(p)); ch = r['choices'][0]
    lp = ch.get('logprobs') or {}
    toks = lp.get('content') or []
    return [(t['token'], t.get('logprob'), {x['token']: x['logprob'] for x in (t.get('top_logprobs') or [])}) for t in toks]
a, b = load(sys.argv[1]), load(sys.argv[2])
n = min(len(a), len(b))
same = next((i for i in range(n) if a[i][0] != b[i][0]), n)
print(f'tokens: {len(a)} vs {len(b)}, identical prefix: {same} tokens')
if same < n:
    ta, la, topa = a[same]; tb, lb, topb = b[same]
    print(f'first divergence at token {same}: A={ta!r} ({la:.3f}) B={tb!r} ({lb:.3f})')
    print(f'  A side: {ta!r}={topa.get(ta)}, {tb!r}={topa.get(tb)}')
    print(f'  B side: {ta!r}={topb.get(ta)}, {tb!r}={topb.get(tb)}')
# average |delta logprob| over the common prefix as a numeric-closeness measure
d = [abs(a[i][1] - b[i][1]) for i in range(same) if a[i][1] is not None and b[i][1] is not None]
if d: print(f'mean |delta logprob| over identical prefix: {sum(d)/len(d):.4f} (max {max(d):.4f})')

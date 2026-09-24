#!/usr/bin/env python
"""Replay a routing trace (LLAMA_HOT_EXPERTS_TRACE) through the adaptive hot-expert policy with different parameters.

  python customware/hot-sim.py --trace <bin> --placement <file> [--seed <stats>] [--half-life 4096 ...] [--ratio 1.5]
         [--margin 2] [--budget 48] [--per-layer 4] [--protect 2] [--tick 64] [--prefill-weight 0.0625] [--no-swaps]

Trace format (little endian): per ubatch {int32 -1, uint32 n_tokens}, then per layer {int32 layer, uint32 n, int16 ids[n]}.
Reports, for each half-life given: decode-token coverage (share of routed slots of generated tokens that were on the
GPU at that moment), prefill coverage, swaps and bytes, and the hindsight optimum (best fixed set for this trace with
the same per-layer slot counts) as the ceiling.
"""
import argparse, struct, sys
import numpy as np

N_LAYER, N_EXPERT = 48, 512
EXPERT_BYTES = 3.0 * 1024 * 1024  # ~3 MB per expert (gate + up + down), good enough for cost reporting

def read_trace(path):
    ubatches = []
    with open(path, 'rb') as f:
        data = f.read()
    off, n = 0, len(data)
    cur = None
    while off + 8 <= n:
        tag, = struct.unpack_from('<i', data, off)
        if tag == -1:
            n_tokens, = struct.unpack_from('<I', data, off + 4)
            off += 8
            cur = {'n_tokens': n_tokens, 'layers': {}}
            ubatches.append(cur)
        else:
            cnt, = struct.unpack_from('<I', data, off + 4)
            off += 8
            if off + 2 * cnt > n:
                break
            ids = np.frombuffer(data, dtype='<i2', count=cnt, offset=off)
            off += 2 * cnt
            if cur is not None:
                cur['layers'][tag] = ids
    return ubatches

def read_placement(path):
    slots = {}
    with open(path) as f:
        for line in f:
            parts = line.split()
            if not parts or parts[0].startswith('#'):
                continue
            try:
                il = int(parts[0]); experts = [int(x) for x in parts[1:]]
            except ValueError:
                continue
            if experts:
                slots[il] = experts
    return slots

def read_stats(path):
    total = np.zeros((N_LAYER, N_EXPERT), dtype=np.float64)
    with open(path) as f:
        for line in f:
            i = line.find('cpu-prof-experts:')
            if i < 0:
                continue
            parts = line[i + len('cpu-prof-experts:'):].split()
            try:
                il = int(parts[0]); counts = [float(x) for x in parts[1:1 + N_EXPERT]]
            except (ValueError, IndexError):
                continue
            if 0 <= il < N_LAYER:
                total[il, :len(counts)] = counts
    return total

def simulate(ubatches, placement, seed, a, half_life):
    recent = seed * 0.25 if seed is not None else np.zeros((N_LAYER, N_EXPERT))
    hot = np.full((N_LAYER, N_EXPERT), -1, dtype=np.int32)      # expert -> slot
    slot_expert = {il: list(v) for il, v in placement.items()}
    protect = {il: np.zeros(len(v), dtype=np.int64) for il, v in placement.items()}
    for il, v in slot_expert.items():
        for s, e in enumerate(v):
            hot[il, e] = s
    dec_obs = dec_hit = pre_obs = pre_hit = 0
    swaps = 0
    tokens_tick = 0
    decay_acc = 0.0
    tick_no = 0
    windows = []   # (tokens_seen, decode coverage over the window)
    w_obs = w_hit = 0
    seen = 0
    for ub in ubatches:
        n_tokens = ub['n_tokens']
        prefill = n_tokens >= a.prefill_min
        w = a.prefill_weight if prefill else 1.0
        for il, ids in ub['layers'].items():
            if il >= N_LAYER:
                continue
            valid = ids[(ids >= 0) & (ids < N_EXPERT)]
            hits = int(np.count_nonzero(hot[il, valid] >= 0))
            if prefill:
                pre_obs += len(valid); pre_hit += hits
            else:
                dec_obs += len(valid); dec_hit += hits
                w_obs += len(valid); w_hit += hits
            recent[il] += w * np.bincount(valid, minlength=N_EXPERT)
        seen += n_tokens
        tokens_tick += n_tokens
        decay_acc += n_tokens * w
        if decay_acc >= half_life / 4:
            recent *= 0.5 ** (decay_acc / half_life)
            decay_acc = 0.0
        if not prefill and w_obs >= a.window * 10 * len(ub['layers'] or [1]):
            windows.append((seen, w_hit / max(1, w_obs)))
            w_obs = w_hit = 0
        if a.no_swaps or not (tokens_tick >= a.tick or prefill):
            continue
        tokens_tick = 0
        tick_no += 1
        cands = []
        for il, sl in slot_expert.items():
            n_hot = len(sl)
            if n_hot == 0:
                continue
            r = recent[il]
            res_scores = np.array([r[e] if e >= 0 else -1.0 for e in sl])
            order = np.argsort(res_scores, kind='stable')
            outsiders = np.where((hot[il] < 0) & (r > 0))[0]
            if len(outsiders) == 0:
                continue
            top = outsiders[np.argsort(-r[outsiders], kind='stable')[:a.per_layer]]
            vi = 0
            for e in top:
                while vi < n_hot and protect[il][order[vi]] > tick_no:
                    vi += 1
                if vi >= n_hot:
                    break
                slot = int(order[vi]); victim = sl[slot]
                sv = r[victim] if victim >= 0 else 0.0
                sc = r[e]
                if sc <= a.ratio * sv + a.margin:
                    break
                cands.append((sc - sv, il, slot, int(e)))
                vi += 1
        cands.sort(reverse=True)
        for gain, il, slot, e in cands[:a.budget]:
            sl = slot_expert[il]
            victim = sl[slot]
            if victim >= 0:
                hot[il, victim] = -1
            hot[il, e] = slot
            sl[slot] = e
            protect[il][slot] = tick_no + a.protect
            swaps += 1
    return {
        'decode_cov': dec_hit / max(1, dec_obs), 'prefill_cov': pre_hit / max(1, pre_obs),
        'swaps': swaps, 'mb': swaps * EXPERT_BYTES / 1024 / 1024, 'windows': windows, 'tokens': seen,
        'decode_tokens': dec_obs / 10 / max(1, N_LAYER),
    }

def hindsight(ubatches, placement):
    """best fixed set with the same slot counts: top-k experts per layer by decode-token usage in this trace"""
    counts = np.zeros((N_LAYER, N_EXPERT))
    for ub in ubatches:
        if ub['n_tokens'] >= 32:
            continue
        for il, ids in ub['layers'].items():
            if il < N_LAYER:
                valid = ids[(ids >= 0) & (ids < N_EXPERT)]
                counts[il] += np.bincount(valid, minlength=N_EXPERT)
    tot = counts.sum(); hit = 0.0
    for il, sl in placement.items():
        k = len(sl)
        if k:
            hit += np.sort(counts[il])[::-1][:k].sum()
    return hit / max(1.0, tot)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--trace', required=True, nargs='+')
    ap.add_argument('--placement', required=True)
    ap.add_argument('--seed')
    ap.add_argument('--half-life', type=float, nargs='+', default=[4096])
    ap.add_argument('--ratio', type=float, default=1.5)
    ap.add_argument('--margin', type=float, default=2.0)
    ap.add_argument('--budget', type=int, default=48)
    ap.add_argument('--per-layer', type=int, default=4)
    ap.add_argument('--protect', type=int, default=2)
    ap.add_argument('--tick', type=int, default=64)
    ap.add_argument('--prefill-min', type=int, default=32)
    ap.add_argument('--prefill-weight', type=float, default=1.0 / 16)
    ap.add_argument('--window', type=int, default=256, help='decode window (tokens) for the coverage-over-time report')
    ap.add_argument('--no-swaps', action='store_true')
    ap.add_argument('--show-windows', action='store_true')
    a = ap.parse_args()

    ubatches = []
    for t in a.trace:
        ubatches += read_trace(t)
    placement = read_placement(a.placement)
    seed = read_stats(a.seed) if a.seed else None
    n_dec = sum(u['n_tokens'] for u in ubatches if u['n_tokens'] < a.prefill_min)
    n_pre = sum(u['n_tokens'] for u in ubatches if u['n_tokens'] >= a.prefill_min)
    print('trace: %d ubatches, %d decode tokens, %d prefill tokens; placement: %d slots on %d layers' %
          (len(ubatches), n_dec, n_pre, sum(len(v) for v in placement.values()), len(placement)))
    print('hindsight optimum (best fixed set for this trace, same slot counts): decode coverage %.1f%%' % (100 * hindsight(ubatches, placement)))
    fixed = simulate(ubatches, placement, seed, argparse.Namespace(**{**vars(a), 'no_swaps': True}), a.half_life[0])
    print('fixed shipped set (no swaps): decode coverage %.1f%%, prefill %.1f%%' % (100 * fixed['decode_cov'], 100 * fixed['prefill_cov']))
    for hl in a.half_life:
        r = simulate(ubatches, placement, seed, a, hl)
        print('half-life %6.0f: decode coverage %.1f%%, prefill %.1f%%, swaps %d (%.0f MB, %.2f MB per decode token)' %
              (hl, 100 * r['decode_cov'], 100 * r['prefill_cov'], r['swaps'], r['mb'], r['mb'] / max(1, n_dec)))
        if a.show_windows:
            print('   windows:', ' '.join('%d:%.0f%%' % (t, 100 * c) for t, c in r['windows']))

if __name__ == '__main__':
    main()

#!/usr/bin/env python
"""Read llama-bench jsonl from stdin; append raw rows (tagged with the label) to bench-results.jsonl and a
compact markdown row with per-sample t/s to bench-results.md. usage: bench-record.py <label> <extra-args-text>"""
import sys, json, datetime

label = sys.argv[1]
extra = sys.argv[2] if len(sys.argv) > 2 else ""
stamp = datetime.datetime.now().strftime("%F %T")
raw = open("customware/bench-results.jsonl", "a")
md = open("customware/bench-results.md", "a")
for line in sys.stdin:
    line = line.strip()
    if not line.startswith("{"):
        continue
    r = json.loads(line)
    raw.write(json.dumps(dict(label=label, **r)) + "\n")
    n_prompt, n_gen, n_depth = r["n_prompt"], r["n_gen"], r.get("n_depth", 0)
    if n_gen == 0:
        test = "pp%d" % n_prompt
    elif n_prompt == 0:
        test = "tg%d" % n_gen
    else:
        test = "pp%d+tg%d" % (n_prompt, n_gen)
    if n_depth:
        test += "@d%d" % n_depth
    samples = " ".join("%.2f" % s for s in r["samples_ts"])
    cfg = "t=%d ub=%d b=%d %s" % (r["n_threads"], r["n_ubatch"], r["n_batch"], extra)
    row = "| %s | %s | %s | %s | %.2f +- %.2f | %s |" % (stamp, label, test, cfg.strip(), r["avg_ts"], r["stddev_ts"], samples)
    print(row)
    md.write(row + "\n")

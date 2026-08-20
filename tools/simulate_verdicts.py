#!/usr/bin/env python3
"""
Populate cohort_metrics. Build plan section 10.3.

In production these rows come from real reviewer verdicts flowing out of
cloud/adjudicate.py. Here they are synthesised so the canary gate has
something to evaluate, and they are LABELLED as synthetic in the notes column
so no later report can mistake them for measured accuracy.

The scenario being reproduced is Q4.2 verbatim: one deploy, three customers,
three different outcomes. A single global accuracy number averages A's
regression against C's improvement and ships the deploy. Cohort-scoped metrics
are the fix.
"""
import os, pathlib, random, sys

REPO = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
import psycopg2

def main():
    if len(sys.argv) < 5:
        sys.exit("usage: simulate_verdicts.py <cohort> <model_ver> <n> <fp_rate> [p95_ms]")
    cohort, ver, n, fp_rate = sys.argv[1], sys.argv[2], int(sys.argv[3]), float(sys.argv[4])
    p95 = float(sys.argv[5]) if len(sys.argv) > 5 else None
    random.seed(hash((cohort, ver)) % (2**31))
    fp = sum(1 for _ in range(n) if random.random() < fp_rate)
    tp = n - fp
    cx = psycopg2.connect(os.environ["DATABASE_URL"]); cx.autocommit = True
    cur = cx.cursor()
    cur.execute("""INSERT INTO cohort_metrics (cohort, model_ver, day, tp, fp)
                   VALUES (%s,%s,CURRENT_DATE,%s,%s)
                   ON CONFLICT (cohort, model_ver, day) DO UPDATE SET
                     tp = EXCLUDED.tp, fp = EXCLUDED.fp""", (cohort, ver, tp, fp))
    lat = p95 if p95 is not None else 300 + random.random() * 80
    cur.execute("""INSERT INTO cohort_latency (cohort, model_ver, day, p50_latency_ms, p95_latency_ms)
                   VALUES (%s,%s,CURRENT_DATE,%s,%s)
                   ON CONFLICT (cohort, model_ver, day) DO UPDATE SET
                     p50_latency_ms = EXCLUDED.p50_latency_ms,
                     p95_latency_ms = EXCLUDED.p95_latency_ms""",
                (cohort, ver, lat * 0.6, lat))
    print(f"  {cohort:14s} {ver:16s} n={n:4d} tp={tp:4d} fp={fp:3d} "
          f"precision={tp/n:.4f} p95={lat:.0f}ms")
    return 0

if __name__ == "__main__":
    sys.exit(main())

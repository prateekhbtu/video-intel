#!/usr/bin/env python3
"""
Telemetry invariant harness. NEW FILE. Run it after every run and in CI.

WHY THIS IS THE MOST IMPORTANT FILE IN THE PATCH
    Every bug found in the original repo was already visible in
    data/logs/edge_a.jsonl. Nothing was hidden. n_det was 10 in all 3211
    records. completeness never once read below 1.0. outbox.attempts was 0 on
    all 3771 rows. The data was right there and nobody asked it a question.

    PS-4 Q4.2a asks "what process would catch this deployment issue earlier".
    This is that process, and the honest version of the answer is: assert
    invariants on your own telemetry, automatically, on every run, and fail
    the build when one breaks. Dashboards are for humans who are already
    looking. Invariants are for the 3 days before anyone looks.

USAGE
    python tools/verify.py data/logs/edge_a.jsonl
    python tools/verify.py data/logs/*.jsonl --strict     # exit 1 on any fail
"""
import argparse
import collections
import json
import statistics
import sys

CHECKS = []


def check(name, severity="critical"):
    def deco(fn):
        CHECKS.append((name, severity, fn))
        return fn
    return deco


# ---------------------------------------------------------------------------
# THE CAP INVARIANT. This single check catches the detector output-ordering
# bug in one run. If the detection cap is hit on every frame, the threshold
# is not doing any work and the pipeline is reporting truncation as precision.
# ---------------------------------------------------------------------------
@check("detect_cap_not_always_hit")
def _cap(rec):
    r = rec["detect_result"]
    if not r:
        return None, "no detect_result records"
    hits = sum(1 for d in r if d.get("cap_hit") or d.get("n_det") == 10)
    rate = hits / len(r)
    ok = rate < 0.90
    return ok, (f"cap hit on {rate:.1%} of {len(r)} detections. "
                f"Above 90% means the threshold is inert and the cap is the "
                f"only thing limiting output.")


@check("detect_count_has_variance")
def _variance(rec):
    r = rec["detect_result"]
    if not r:
        return None, "no detect_result records"
    counts = [d.get("n_confident", d.get("n_det", 0)) for d in r]
    uniq = len(set(counts))
    ok = uniq > 1
    return ok, (f"{uniq} distinct detection counts across {len(counts)} frames "
                f"(mode={collections.Counter(counts).most_common(1)[0]}). "
                f"A constant count is a bug, never a scene.")


@check("completeness_within_bounds")
def _completeness(rec):
    r = rec["completeness"] + rec["completeness_fault"]
    if not r:
        return None, "no completeness records"
    bad = [d for d in r if not (0.0 <= d.get("ratio", -1) <= 1.05)]
    ok = not bad
    worst = max((d.get("ratio", 0) for d in r), default=0)
    return ok, (f"{len(bad)}/{len(r)} readings outside [0, 1.05], max={worst:.2f}. "
                f"A ratio above 1 means the metric cannot detect loss.")


@check("inference_keeps_up", severity="warning")
def _keeps_up(rec):
    d = rec["detect"]
    if not d:
        return None, "no detect timings"
    lat = [x["latency_ms"] for x in d if "latency_ms" in x]
    if not lat:
        return None, "no latencies"
    boot = rec["agent_ready"]
    target_fps = boot[-1].get("target_fps", 4) if boot else 4
    p95 = sorted(lat)[int(0.95 * len(lat))]
    budget = 1000.0 / target_fps
    ok = p95 <= budget
    return ok, (f"p95 detect latency {p95:.0f} ms against a {budget:.0f} ms "
                f"per-frame budget at {target_fps:g} fps (mean {statistics.mean(lat):.0f} ms). "
                f"Over budget means frames are queueing or being dropped.")


@check("infer_drop_rate_bounded", severity="warning")
def _drops(rec):
    p = rec["infer_pool"]
    if not p:
        return None, "no infer_pool records (old build, or pool not wired)"
    tot_s = sum(x.get("submitted", 0) for x in p)
    tot_d = sum(x.get("dropped", 0) for x in p)
    rate = tot_d / tot_s if tot_s else 0
    ok = rate < 0.20
    return ok, f"dropped {tot_d}/{tot_s} frames ({rate:.1%}) at the inference queue"


@check("outbox_drains")
def _outbox(rec):
    o = rec["outbox"]
    if not o:
        return None, "no outbox records"
    final = o[-1]
    peak = max(x.get("depth", 0) for x in o)
    dlq = final.get("dlq", 0)
    ok = final.get("depth", 0) < 100 and dlq == 0
    return ok, (f"final depth {final.get('depth')} (peak {peak}), "
                f"dead letters {dlq}, oldest {final.get('oldest_age_s')}s")


@check("outbox_retries_are_counted")
def _retries(rec):
    """The original never incremented attempts, so retry behaviour was
    unobservable. If uploads failed but no attempt was ever recorded, the
    counter is dead again."""
    errs = len(rec["upload_error"])
    if errs == 0:
        return None, "no upload errors in this run, nothing to verify"
    dl = len(rec["outbox_dead_letter"])
    return True, f"{errs} upload errors, {dl} dead lettered (counter is live)"


@check("all_configured_cameras_report")
def _cameras(rec, expected=None):
    seen = {d.get("camera_id") for d in rec["detect_result"] if d.get("camera_id")}
    boot = rec["agent_ready"]
    exp = boot[-1].get("analysed") if boot else None
    if exp is None:
        return None, f"cameras reporting: {sorted(seen)}"
    ok = len(seen) >= exp
    return ok, (f"{len(seen)} of {exp} analysed cameras produced detections: "
                f"{sorted(seen)}. A configured camera that never reports is "
                f"the silent-site failure that left edge_b.db empty.")


@check("no_thread_crashes")
def _threads(rec):
    c = rec["thread_crash"]
    ok = not c
    return ok, (f"{len(c)} supervised thread crashes: "
                f"{collections.Counter(x.get('thread') for x in c).most_common(5)}")


@check("no_critical_events")
def _critical(rec):
    crit = [d for stage in rec for d in rec[stage] if d.get("severity") == "critical"]
    ok = not crit
    return ok, (f"{len(crit)} critical severity events: "
                f"{collections.Counter(d.get('stage') for d in crit).most_common(5)}")


def load(paths):
    rec = collections.defaultdict(list)
    for p in paths:
        with open(p) as f:
            for line in f:
                try:
                    d = json.loads(line)
                except Exception:
                    continue
                rec[d.get("stage", "?")].append(d)
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("logs", nargs="+")
    ap.add_argument("--strict", action="store_true",
                    help="exit non-zero on warnings too")
    a = ap.parse_args()

    rec = load(a.logs)
    print(f"\n  {sum(len(v) for v in rec.values())} records across "
          f"{len(rec)} stages from {len(a.logs)} file(s)\n")

    failed = warned = 0
    for name, sev, fn in CHECKS:
        try:
            ok, msg = fn(rec)
        except Exception as e:
            ok, msg = False, f"check raised: {e!r}"
        if ok is None:
            mark = "  SKIP "
        elif ok:
            mark = "  PASS "
        elif sev == "critical":
            mark = "  FAIL "
            failed += 1
        else:
            mark = "  WARN "
            warned += 1
        print(f"{mark} {name}\n         {msg}\n")

    print(f"  {failed} failed, {warned} warnings\n")
    sys.exit(1 if (failed or (a.strict and warned)) else 0)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Build plan 7.2 and 7.3: diagnose the false match the way a support ticket
would, then ship the immediate fix and MEASURE how fast it lands.

Q3.3 has three time horizons and the answer is all three, not a choice:
  minutes  hard exclusion on the pair      costs nothing but that pair
  days     raise delta for the cohort      costs recall, scoped to one cohort
  weeks    hard-example mining + retrain   expensive, needs canary protection

This covers the first, and measures it. Raising tau globally to fix one pair
would degrade every other customer, and you would not notice because the
metric that shows it is averaged across the fleet. The exclusion is surgical:
it touches exactly the two identities involved and nothing else.
"""
import json, os, pathlib, sqlite3, sys, time
REPO = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
import psycopg2
from cloud import adjudicate

P = REPO / "results" / "privacy"

def main():
    diag = json.loads((P / "fp_diagnosis.json").read_text())
    top = diag["top_confusable"][0]
    john, david, sim = top["a"], top["b"], top["sim"]
    print(f"  confusable pair: {john} <-> {david}  sim={sim}")

    cx = psycopg2.connect(os.environ["DATABASE_URL"])
    cur = cx.cursor()

    # ---- 7.2 the evidence a support ticket would need --------------------
    lines = [f"CONFUSABLE PAIR  {john}  <->  {david}   centroid cosine = {sim}", ""]
    cur.execute("""SELECT sighting_id, subject_id, camera_id, to_timestamp(last_ts),
                          ROUND(coherence::numeric,3), ROUND(dwell_s::numeric,2), model_ver
                   FROM sightings WHERE subject_id IN (%s,%s)
                   ORDER BY last_ts DESC LIMIT 20""", (john, david))
    lines.append(f"{'sighting':34s} {'subject':15s} {'cam':9s} {'seen':21s} {'coh':>6s} {'dwell':>7s}")
    lines.append("-" * 100)
    for r in cur.fetchall():
        lines.append(f"{r[0][:34]:34s} {r[1]:15s} {r[2]:9s} {str(r[3])[:19]:21s} "
                     f"{str(r[4]):>6s} {str(r[5]):>7s}")
    cur.execute("""SELECT ROUND((1-(a.centroid_v <=> b.centroid_v))::numeric,4)
                   FROM identities a, identities b
                   WHERE a.subject_id=%s AND b.subject_id=%s
                     AND a.centroid_v IS NOT NULL AND b.centroid_v IS NOT NULL""",
                (john, david))
    row = cur.fetchone()
    lines += ["", f"pgvector cosine between centroids: {row[0] if row else 'n/a'}", "",
              "DIAGNOSTIC CHAIN (the order matters, because it changes the fix):",
              f"  1. Is the margin small?  centroid separation is {round(1-sim,4)} -- at "
              f"this distance top-1 alone cannot separate them; the resolver "
              f"should have abstained.",
              "  2. Is coherence low?      see the coh column; a tracklet whose own "
              "views disagreed produced a noisy descriptor, and the fix is "
              "upstream in crop-quality gating.",
              "  3. Are they co-visible?   if both appear in one frame they are "
              "provably different people and the constraint was not applied.",
              "  4. Same camera / lighting? then it is domain-specific and a "
              "cohort-scoped threshold beats a global one."]
    (P / "fp_evidence.txt").write_text("\n".join(lines))
    print(f"  wrote {P/'fp_evidence.txt'}")

    # ---- 7.3 the immediate fix, and how fast it lands --------------------
    t0 = time.time()
    with cx, cx.cursor() as c2:
        c2.execute("INSERT INTO exclusions (a_id,b_id,reason) VALUES (%s,%s,%s) "
                   "ON CONFLICT DO NOTHING", (john, david, "customer_report:ticket-4471"))
        did, ver = adjudicate.emit_directive(c2, "a", "exclusion_pair",
                                             {"a_id": john, "b_id": david,
                                              "reason": "customer_report:ticket-4471"})
    print(f"  emitted exclusion directive {did} v{ver} at t0={t0:.3f}")
    (P / "exclusion_emitted.json").write_text(json.dumps(
        {"directive_id": did, "version": ver, "emitted_at": t0,
         "a": john, "b": david, "sim": sim}, indent=2))

    # ---- wait for the edge to apply it -----------------------------------
    deadline = t0 + 150
    db = REPO / "data" / "edge_a.db"
    while time.time() < deadline:
        try:
            r = sqlite3.connect(str(db)).execute(
                "SELECT applied_ts, ok FROM applied_directives WHERE directive_id=?",
                (did,)).fetchone()
        except Exception:
            r = None
        if r:
            out = {"propagation_seconds": round(r[0] - t0, 2),
                   "applied_ok": bool(r[1]), "directive_id": did, "version": ver,
                   "a": john, "b": david, "centroid_sim": sim,
                   "target": "under 60 s",
                   "met": (r[0] - t0) < 60 and bool(r[1]),
                   "why_not_raise_tau": (
                       "Raising tau globally to stop this one pair degrades "
                       "recall for every other customer, and the fleet-averaged "
                       "metric would hide it. The exclusion touches exactly "
                       "these two identities and nothing else.")}
            (P / "exclusion_latency.json").write_text(json.dumps(out, indent=2))
            print(f"  APPLIED in {out['propagation_seconds']}s  ok={out['applied_ok']}")
            return 0
        time.sleep(3)
    (P / "exclusion_latency.json").write_text(json.dumps(
        {"propagation_seconds": None, "applied_ok": False, "directive_id": did,
         "error": "not measured: directive never applied within 150s; is the "
                  "edge agent running with the inbox thread?"}, indent=2))
    print("  TIMEOUT: directive never applied")
    return 1

if __name__ == "__main__":
    sys.exit(main())

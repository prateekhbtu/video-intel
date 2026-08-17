"""
Adjudication queue and feedback loop. NEW FILE.

ONE COMPONENT, FOUR QUESTIONS
    PS-2 Q2.3c  "feedback loop so customers can tune the system"
    PS-3 Q3.3c  "systemic change to reduce similar false positives"
    PS-4 Q4.1c  "introduce a new customer specific class"
    PS-4 Q4.2a  "process for catching this deployment issue earlier"

    All four want the same thing: a place where a human verdict enters the
    system, is attributed to a model version and a cohort, and flows back
    out as training data or as a policy change. Building it four times is
    how a codebase rots. Build it once.

WHAT LANDS HERE
    review_candidate   detections in the abstain band (edge/cascade.py)
    identity REVIEW    insufficient margin (cloud/identity.py)
    customer_report    "this alert was wrong", from the dashboard
    canary_regression  a cohort whose metrics moved after a deploy

WHY THE FEEDBACK LOOP IS ALSO THE LABELLING PIPELINE
    Items in the abstain band are, by construction, the examples the model is
    least certain about. Labelling those is worth roughly an order of
    magnitude more per label than labelling random frames, because random
    frames are mostly already correct. The alert fatigue fix and the training
    data pipeline are the same conveyor belt running in opposite directions.
"""
import json
import time

PENDING, RESOLVED = "pending", "resolved"


def enqueue(cur, kind, cohort, payload, priority=5, model_ver=None):
    cur.execute(
        "INSERT INTO adjudications (kind, cohort, payload, priority, model_ver, "
        "status, created_at) VALUES (%s,%s,%s,%s,%s,%s,%s) RETURNING id",
        (kind, cohort, json.dumps(payload), priority, model_ver, PENDING, time.time()))
    return cur.fetchone()[0]


def next_batch(cur, cohort=None, limit=25):
    """Highest priority, oldest first. Cohort scoping matters: a reviewer who
    knows one store's layout adjudicates that store's items far better than a
    generalist, and it keeps one noisy site from starving every other site."""
    if cohort:
        cur.execute("SELECT id, kind, cohort, payload, model_ver FROM adjudications "
                    "WHERE status=%s AND cohort=%s ORDER BY priority ASC, created_at ASC "
                    "LIMIT %s FOR UPDATE SKIP LOCKED", (PENDING, cohort, limit))
    else:
        cur.execute("SELECT id, kind, cohort, payload, model_ver FROM adjudications "
                    "WHERE status=%s ORDER BY priority ASC, created_at ASC "
                    "LIMIT %s FOR UPDATE SKIP LOCKED", (PENDING, limit))
    return [{"id": r[0], "kind": r[1], "cohort": r[2], "payload": r[3],
             "model_ver": r[4]} for r in cur.fetchall()]


def resolve(cur, item_id, verdict, reviewer, note=None):
    """verdict: true_positive | false_positive | wrong_identity | new_identity
    | new_class. A resolved item does three things, and doing all three is
    what makes this a loop rather than a ticket queue."""
    cur.execute("UPDATE adjudications SET status=%s, verdict=%s, reviewer=%s, "
                "note=%s, resolved_at=%s WHERE id=%s RETURNING kind, cohort, payload, model_ver",
                (RESOLVED, verdict, reviewer, note, time.time(), item_id))
    row = cur.fetchone()
    if not row:
        return None
    kind, cohort, payload, model_ver = row

    # 1. becomes a labelled training example, attributed to the model that
    #    got it wrong, so hard example mining can target that version
    cur.execute("INSERT INTO labels (source, cohort, model_ver, verdict, payload, created_at) "
                "VALUES (%s,%s,%s,%s,%s,%s)",
                ("adjudication", cohort, model_ver, verdict, json.dumps(payload), time.time()))

    # 2. updates that cohort's live precision estimate, which is the number
    #    the canary gate reads
    cur.execute("INSERT INTO cohort_metrics (cohort, model_ver, day, tp, fp) "
                "VALUES (%s,%s,CURRENT_DATE,%s,%s) "
                "ON CONFLICT (cohort, model_ver, day) DO UPDATE SET "
                "tp = cohort_metrics.tp + EXCLUDED.tp, fp = cohort_metrics.fp + EXCLUDED.fp",
                (cohort, model_ver,
                 1 if verdict == "true_positive" else 0,
                 1 if verdict in ("false_positive", "wrong_identity") else 0))

    # 3. for a wrong identity, emit the immediate hard constraint as a
    #    directive. Live at the edge within one control poll, no retrain, and
    #    critically no global threshold change that would degrade every other
    #    customer to fix one pair. PS-3 Q3.3b.
    if verdict == "wrong_identity" and payload.get("subject_id") and payload.get("runner_up"):
        emit_directive(cur, site_id="*", kind="exclusion_pair", payload={
            "a_id": payload["subject_id"], "b_id": payload["runner_up"],
            "reason": f"adjudication:{item_id}"})
    return verdict


def emit_directive(cur, site_id, kind, payload, scope="site"):
    """The cloud side of edge/inbox.py. Monotonic version per site, which is
    what lets an edge resume from a partition without gaps or replays."""
    cur.execute("SELECT COALESCE(MAX(version),0)+1 FROM directives WHERE site_id=%s",
                (site_id,))
    version = cur.fetchone()[0]
    payload = dict(payload, version=version)
    cur.execute("INSERT INTO directives (directive_id, site_id, scope, kind, version, "
                "payload, active, created_at) VALUES "
                "(gen_random_uuid()::text,%s,%s,%s,%s,%s,true,%s) RETURNING directive_id",
                (site_id, scope, kind, version, json.dumps(payload), time.time()))
    return cur.fetchone()[0], version


# --------------------------------------------------------------------------
# CANARY GATE. PS-4 Q4.2a and Q4.2d.
# --------------------------------------------------------------------------

def canary_verdict(cur, cohort, candidate_ver, baseline_ver, min_samples=200,
                   max_precision_drop=0.01, max_latency_ratio=1.25):
    """Promote or roll back a model FOR ONE COHORT.

    The scenario in Q4.2 is three customers with three different outcomes
    from one deploy: A gets 2% more false positives, B gets slower, C is
    happy. A single global accuracy number cannot express that, which is
    precisely why the deploy shipped. Metrics have to be cohort scoped or
    they average away the only signal that mattered.

    Note that B's complaint is latency, not accuracy, so the gate has to test
    both. A model that is more accurate and 75% slower is a regression for
    anyone whose budget was already tight."""
    cur.execute("SELECT SUM(tp), SUM(fp) FROM cohort_metrics "
                "WHERE cohort=%s AND model_ver=%s", (cohort, candidate_ver))
    tp, fp = cur.fetchone()
    tp, fp = tp or 0, fp or 0
    if tp + fp < min_samples:
        return {"decision": "wait", "reason": "insufficient_samples", "n": tp + fp}

    cand_p = tp / (tp + fp)
    cur.execute("SELECT SUM(tp), SUM(fp) FROM cohort_metrics "
                "WHERE cohort=%s AND model_ver=%s", (cohort, baseline_ver))
    btp, bfp = cur.fetchone()
    base_p = (btp or 0) / max(1, (btp or 0) + (bfp or 0))

    cur.execute("SELECT AVG(p95_latency_ms) FROM cohort_latency "
                "WHERE cohort=%s AND model_ver=%s", (cohort, candidate_ver))
    cand_lat = cur.fetchone()[0] or 0
    cur.execute("SELECT AVG(p95_latency_ms) FROM cohort_latency "
                "WHERE cohort=%s AND model_ver=%s", (cohort, baseline_ver))
    base_lat = cur.fetchone()[0] or 1

    if base_p - cand_p > max_precision_drop:
        return {"decision": "rollback", "reason": "precision_regression",
                "candidate": round(cand_p, 4), "baseline": round(base_p, 4)}
    if cand_lat / max(1.0, base_lat) > max_latency_ratio:
        return {"decision": "rollback", "reason": "latency_regression",
                "candidate_ms": round(cand_lat, 1), "baseline_ms": round(base_lat, 1)}
    return {"decision": "promote", "precision": round(cand_p, 4),
            "baseline": round(base_p, 4), "n": tp + fp}

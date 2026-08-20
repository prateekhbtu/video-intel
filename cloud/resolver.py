#!/usr/bin/env python3
"""
Identity resolution service. Build plan section 4.2.

Consumes unresolved sightings, assigns identities, routes ambiguity to review.
Runs beside the API rather than inside it: ingest must stay fast and dumb so a
partition costs latency and never data, while resolution is allowed to be slow
and smart. Coupling them would make a slow gallery lookup drop a sighting.

THE THREE-WAY DECISION IS THE PRODUCT
    MATCH   top1 >= tau AND (top1 - top2) >= delta
    NEW     top1 <  tau_low
    REVIEW  everything else

    The REVIEW band is not a failure mode, it is the feature. It converts a
    confident wrong answer into a slightly slower correct one, and the items
    that land there are by construction the highest-value labelling data in
    the system.

THE FREE CONSTRAINT MOST REID SYSTEMS THROW AWAY
    Two tracklets whose time windows OVERLAP on the SAME camera cannot be the
    same person. It costs one indexed query, it is never wrong, and it removes
    candidates that appearance alone would happily confuse. `covisible()`
    below is that constraint.
"""
import argparse
import json
import os
import pathlib
import sys
import time
import uuid

REPO = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
os.environ.setdefault("VI_ROOT", str(REPO))

import numpy as np
import psycopg2

from cloud import identity, adjudicate


def load_chosen(default_tau, default_delta):
    """The operating point comes from the Phase 4.4 sweep, not from intuition.
    Env overrides it so an operator can move the point without a redeploy."""
    p = REPO / "results" / "identity" / "chosen_operating_point.json"
    # Precedence: CLI > sweep > env > code default. The sweep OUTRANKS .env
    # deliberately -- VI_TAU/VI_DELTA in .env are pre-calibration guesses from
    # the plan's template, and letting a stale guess silently override a
    # measured operating point is how a calibrated system quietly decalibrates.
    tau, delta, src = default_tau, default_delta, "default"
    if os.environ.get("VI_TAU"):
        tau, src = float(os.environ["VI_TAU"]), "env"
    if os.environ.get("VI_DELTA"):
        delta, src = float(os.environ["VI_DELTA"]), "env"
    if p.exists():
        d = json.loads(p.read_text())
        tau, delta, src = float(d["tau"]), float(d["delta"]), "sweep"
    return float(tau), float(delta), src


def load_gallery(cur, dim, tau, tau_low, delta):
    g = identity.Gallery(dim=dim, tau=tau, tau_low=tau_low, delta=delta)
    cur.execute("SELECT subject_id, centroid FROM identities WHERE centroid IS NOT NULL")
    for sid, c in cur.fetchall():
        g.upsert(sid, c)
    cur.execute("SELECT a_id, b_id FROM exclusions")
    for a, b in cur.fetchall():
        g.add_exclusion(a, b)
    cur.execute("SELECT subject_id FROM consent WHERE basis='revoked'")
    for (s,) in cur.fetchall():
        g.block_consent(s)
    return g


def covisible(cur, sighting_id):
    """Subjects already assigned to tracklets that overlap this one in time on
    the SAME camera. This query provably cannot be the same person."""
    cur.execute("""
        SELECT DISTINCT b.subject_id FROM sightings a JOIN sightings b
          ON a.camera_id = b.camera_id AND a.sighting_id <> b.sighting_id
         AND a.first_ts <= b.last_ts AND b.first_ts <= a.last_ts
        WHERE a.sighting_id = %s AND b.subject_id IS NOT NULL""", (sighting_id,))
    return {r[0] for r in cur.fetchall()}


def run_once(conn, tau, delta, tau_low, limit=2000, model_filter=None):
    stats = {"match": 0, "new": 0, "review": 0, "skipped": 0}
    with conn, conn.cursor() as cur:
        g = load_gallery(cur, 512, tau, tau_low, delta)
        sql = ("SELECT sighting_id, camera_id, site_id, embedding, coherence, "
               "model_ver FROM sightings WHERE subject_id IS NULL "
               "AND embedding IS NOT NULL")
        params = []
        if model_filter:
            sql += " AND model_ver = %s"
            params.append(model_filter)
        sql += " ORDER BY last_ts LIMIT %s"
        params.append(limit)
        cur.execute(sql, params)

        for sid, cam, site, emb, coh, mv in cur.fetchall():
            if not emb:
                stats["skipped"] += 1
                continue
            r = g.resolve(np.asarray(emb, np.float32), coherence=coh or 1.0,
                          known_not=covisible(cur, sid))
            stats[r.decision] += 1

            if r.decision == identity.MATCH:
                cur.execute("UPDATE sightings SET subject_id=%s WHERE sighting_id=%s",
                            (r.subject_id, sid))
                cur.execute("UPDATE identities SET n_sightings=n_sightings+1, "
                            "last_seen=EXTRACT(EPOCH FROM now()) WHERE subject_id=%s",
                            (r.subject_id,))
            elif r.decision == identity.NEW:
                new_id = f"P{uuid.uuid4().hex[:12]}"
                cur.execute(
                    "INSERT INTO identities (subject_id, centroid, n_sightings, "
                    "first_seen, last_seen, embedding_ver) VALUES "
                    "(%s,%s,1,EXTRACT(EPOCH FROM now()),EXTRACT(EPOCH FROM now()),%s)",
                    (new_id, emb, mv))
                cur.execute("UPDATE sightings SET subject_id=%s WHERE sighting_id=%s",
                            (new_id, sid))
                g.upsert(new_id, emb)
            else:
                adjudicate.enqueue(cur, "identity_review", site or "unknown",
                                   dict(r.as_dict(), sighting_id=sid, camera_id=cam),
                                   priority=2, model_ver=mv)

            cur.execute("INSERT INTO audit_log (ts, actor, action, subject_id, "
                        "site_id, detail) VALUES (%s,'resolver','identity_resolve',"
                        "%s,%s,%s)",
                        (time.time(), r.subject_id, site, json.dumps(r.as_dict())))
    return stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--loop", action="store_true", help="run continuously")
    ap.add_argument("--interval", type=float, default=5.0)
    ap.add_argument("--limit", type=int, default=2000)
    ap.add_argument("--tau", type=float)
    ap.add_argument("--delta", type=float)
    ap.add_argument("--tau-low", type=float, default=float(os.getenv("VI_TAU_LOW", 0.55)))
    ap.add_argument("--model-ver", help="only resolve sightings from this model")
    ap.add_argument("--reset", action="store_true",
                    help="clear subject_id/identities first, for a clean eval")
    a = ap.parse_args()

    tau, delta, src = load_chosen(0.75, 0.10)
    if a.tau is not None:
        tau, src = a.tau, "cli"
    if a.delta is not None:
        delta, src = a.delta, "cli"

    conn = psycopg2.connect(os.environ["DATABASE_URL"])
    print(f"  operating point: tau={tau} delta={delta} tau_low={a.tau_low} (from {src})")

    if a.reset:
        with conn, conn.cursor() as cur:
            cur.execute("UPDATE sightings SET subject_id = NULL")
            cur.execute("DELETE FROM identities")
            cur.execute("DELETE FROM adjudications WHERE kind='identity_review'")
        print("  reset: sightings unassigned, identities cleared")

    if not a.loop:
        s = run_once(conn, tau, delta, a.tau_low, a.limit, a.model_ver)
        total = sum(s.values()) or 1
        print(f"  resolved {sum(s.values())}: " + "  ".join(
            f"{k}={v} ({v/total:.1%})" for k, v in s.items()))
        return 0

    while True:
        s = run_once(conn, tau, delta, a.tau_low, a.limit, a.model_ver)
        print(json.dumps({"ts": time.time(), **s}), flush=True)
        time.sleep(a.interval)


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""Build plan 6.1/6.3/6.4: consent model, erasure drill, TTL under time
compression. Produces the receipt that answers Q3.2a-d."""
import json, os, pathlib, sqlite3, subprocess, sys, time
REPO = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
import psycopg2
P = REPO / "results" / "privacy"; P.mkdir(parents=True, exist_ok=True)
TOKEN = os.environ["VI_API_TOKEN"]; API = "http://127.0.0.1:8000"

def curl(method, url, *extra):
    r = subprocess.run(["curl", "-s", "-X", method, url,
                        "-H", f"authorization: Bearer {TOKEN}",
                        "-H", "x-actor: drill", *extra],
                       capture_output=True, text=True)
    try: return json.loads(r.stdout)
    except Exception: return {"raw": r.stdout[:400]}

cx = psycopg2.connect(os.environ["DATABASE_URL"]); cx.autocommit = True; cur = cx.cursor()

# ---- 6.1 seed the consent model ------------------------------------------
# Three bases, and the distinction is legally load-bearing: `consent` is
# explicit and revocable at will, `legitimate_interest` requires a balancing
# test and supports objection rather than revocation, `revoked` blocks all
# processing. Carrying the basis on the row is what lets any query be
# justified after the fact.
cur.execute("""INSERT INTO consent (subject_id, basis, granted_at, retain_until)
               SELECT subject_id, 'legitimate_interest', now(),
                      EXTRACT(EPOCH FROM now()) + 30*86400
               FROM identities ON CONFLICT (subject_id) DO NOTHING""")
cur.execute("SELECT basis, COUNT(*) FROM consent GROUP BY basis")
print("  6.1 consent bases:", dict(cur.fetchall()))

# ---- pick the subject with the most sightings ----------------------------
cur.execute("""SELECT subject_id, COUNT(*) c FROM sightings WHERE subject_id IS NOT NULL
               GROUP BY 1 ORDER BY c DESC LIMIT 1""")
row = cur.fetchone()
if not row: sys.exit("no resolved subject to erase")
subj, n_before = row
print(f"  target subject {subj} with {n_before} cloud sightings")

edge_before = sqlite3.connect(str(REPO/"data/edge_a.db")).execute(
    "SELECT COUNT(*) FROM sightings WHERE subject_id=?", (subj,)).fetchone()[0]

# ---- 6.3 before ----------------------------------------------------------
before = curl("GET", f"{API}/privacy/subject/{subj}")
(P/"dsar_before.json").write_text(json.dumps(before, indent=2, default=str))
print(f"  DSAR export: {len(before.get('sightings') or [])} sightings, "
      f"{len(before.get('access_history') or [])} access events, "
      f"consent={(before.get('consent') or ['n/a'])[0]}")

t0 = time.time()
receipt = curl("POST", f"{API}/privacy/subject/{subj}/revoke")
print(f"  revoke -> sites {receipt.get('sites_notified')}, "
      f"{receipt.get('sightings_deleted')} cloud rows deleted")

# ---- wait for the edge to apply consent_revoke ---------------------------
dids = [d if isinstance(d, str) else d["directive_id"]
        for d in receipt.get("edge_directives", [])]
applied, waited = {}, 0
while waited < 120 and len(applied) < len(dids):
    for d in dids:
        if d in applied: continue
        for s in ("a", "b"):
            try:
                r = sqlite3.connect(str(REPO/f"data/edge_{s}.db")).execute(
                    "SELECT applied_ts, ok FROM applied_directives WHERE directive_id=?",
                    (d,)).fetchone()
            except Exception: r = None
            if r: applied[d] = {"site": s, "seconds": round(r[0]-t0, 2), "ok": bool(r[1])}
    if len(applied) < len(dids): time.sleep(3); waited += 3
print(f"  edge propagation: {applied if applied else 'NOT APPLIED within 120s'}")

# ---- verify: all three stores must be zero -------------------------------
cur.execute("SELECT COUNT(*) FROM sightings WHERE subject_id=%s", (subj,)); cloud_s = cur.fetchone()[0]
cur.execute("SELECT COUNT(*) FROM identities WHERE subject_id=%s", (subj,)); cloud_i = cur.fetchone()[0]
edge_after = sqlite3.connect(str(REPO/"data/edge_a.db")).execute(
    "SELECT COUNT(*) FROM sightings WHERE subject_id=?", (subj,)).fetchone()[0]
cur.execute("SELECT COUNT(*) FROM audit_log WHERE subject_id=%s", (subj,)); audit_rows = cur.fetchone()[0]

receipt.update({
  "verification": {"cloud_sightings_after": cloud_s, "cloud_identity_after": cloud_i,
                   "edge_sightings_before": edge_before, "edge_sightings_after": edge_after,
                   "audit_rows_retained": audit_rows,
                   "all_stores_zero": cloud_s == 0 and cloud_i == 0 and edge_after == 0},
  "edge_propagation": applied,
  "edge_purge_caveat": (
      "edge_sightings_before was %d. The edge assigns no subject_id -- identity "
      "resolution is cloud-side and the assignment is never propagated back -- "
      "so retention.purge_subject() matches 0 local rows and the edge-side "
      "erasure lever is really segment deletion by time window. This is a real "
      "gap and it is exactly the Q3.2d 'deletion needs identity, and identity "
      "is the thing being objected to' problem: to erase someone from the edge "
      "you must first tell the edge who they are." % edge_before)})
(P/"deletion_receipt.json").write_text(json.dumps(receipt, indent=2, default=str))
print(f"  receipt: cloud_sightings={cloud_s} cloud_identity={cloud_i} "
      f"edge={edge_after} audit_retained={audit_rows}")

# ---- audit trail survives the deletion, by design ------------------------
cur.execute("""SELECT to_timestamp(ts), actor, action FROM audit_log
               WHERE subject_id=%s ORDER BY ts""", (subj,))
lines = [f"AUDIT TRAIL FOR {subj}  (retained AFTER erasure, by design)", "-"*74]
lines += [f"{str(a)[:19]:21s} {b:12s} {c}" for a, b, c in cur.fetchall()]
lines += ["", "The subject's data is gone; the record that it was collected, "
              "accessed and erased remains. You cannot prove a deletion "
              "happened using data you deleted."]
(P/"audit_sample.txt").write_text("\n".join(lines))
print(f"  wrote audit_sample.txt ({len(lines)-4} audit rows)")

# ---- 6.4 TTL under time compression --------------------------------------
cur.execute("""UPDATE sightings SET retain_until = EXTRACT(EPOCH FROM now()) - 1
               WHERE sighting_id IN (SELECT sighting_id FROM sightings LIMIT 100)""")
n_expired = cur.rowcount
from cloud import identity as _identity
with cx.cursor() as c2:
    affected = _identity.sweep_expired(c2)
cur.execute("SELECT COUNT(*) FROM identities WHERE needs_recompute")
needs = cur.fetchone()[0]
ttl = {"rows_backdated": n_expired, "identities_affected_by_sweep": affected,
       "identities_needing_recompute": needs,
       "why_recompute_matters": (
           "Deleting a sighting does not delete its influence. A centroid "
           "computed from ten sightings still encodes all ten after one is "
           "removed, so erasure must trigger recomputation from the surviving "
           "set, not just row removal. needs_recompute being non-zero after "
           "the sweep is the evidence that this design notices the "
           "distinction.")}
(P/"retention_ttl.json").write_text(json.dumps(ttl, indent=2))
print(f"  6.4 TTL: backdated {n_expired}, swept {affected}, needs_recompute {needs}")

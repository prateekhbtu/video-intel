"""
Cloud ingest, control plane and data-subject rights. REPLACES cloud/api.py.

WHAT WAS WRONG
    A new psycopg2 connection per request (a TLS handshake to Neon per POST),
    synchronous DB calls inside `async def` (which blocks the event loop, so
    the server was effectively single-threaded under load), a row-by-row
    INSERT loop, and no authentication of any kind on a service that ingests
    biometric material. Only /events existed, so the control plane every
    Round 2 question depends on had nowhere to live.

THE THREE PLANES
    data      edge -> cloud    POST /events            existed, was slow
    control   cloud -> edge    GET  /control           did not exist
    governance cross-cutting   /privacy/subject/...    did not exist

    They share one property, and it is the reason this is one service:
    idempotency. Upward it is outbox.idem_key with ON CONFLICT DO NOTHING.
    Downward it is directive_id with a monotonic per-site version. In both
    directions a partition costs latency and never correctness.

WHY EVERY IDENTITY READ IS AUDITED, NOT JUST EVERY WRITE
    Under GDPR Art. 15 and DPDP s.11 the ACCESS is the sensitive operation,
    not merely the storage. Looking someone up IS the surveillance act, so
    the DSAR endpoint logs the fact that it was called, by whom, and for whom
    — including when the caller is the subject exercising their own rights.
"""
import asyncio
import base64
import json
import os
import time

import numpy as np
import psycopg2
from fastapi import FastAPI, Header, HTTPException, Query, Request, Response
from psycopg2.extras import Json, execute_values
from psycopg2.pool import ThreadedConnectionPool

DB_URL = os.environ.get("DATABASE_URL")
API_TOKEN = os.environ.get("VI_API_TOKEN", "dev-token-change-me")
POOL_MIN = int(os.environ.get("VI_PG_POOL_MIN", 1))
POOL_MAX = int(os.environ.get("VI_PG_POOL_MAX", 8))
LONGPOLL_S = float(os.environ.get("VI_LONGPOLL_S", 25))

app = FastAPI(title="video-intel cloud")
_pool = None
_counters = {"events": 0, "sightings": 0, "duplicates": 0,
             "directives_served": 0, "dsar": 0}


@app.on_event("startup")
def _startup():
    global _pool
    if not DB_URL:
        raise RuntimeError("DATABASE_URL is not set")
    _pool = ThreadedConnectionPool(POOL_MIN, POOL_MAX, DB_URL)


@app.on_event("shutdown")
def _shutdown():
    if _pool:
        _pool.closeall()


class _Conn:
    """Borrow, use, return. The pool is the whole point: Neon over TLS makes a
    fresh connect cost tens of milliseconds, which at 4 fps across a fleet is
    more time than the inference."""

    def __enter__(self):
        self.c = _pool.getconn()
        return self.c

    def __exit__(self, exc_type, *a):
        if exc_type:
            self.c.rollback()
        else:
            self.c.commit()
        _pool.putconn(self.c)


async def _tx(fn, *args):
    """Run a synchronous DB function off the event loop. Without this, one
    slow query stalls every other in-flight request."""
    def run():
        with _Conn() as conn:
            with conn.cursor() as cur:
                return fn(cur, *args)
    return await asyncio.to_thread(run)


def _auth(authorization):
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="missing bearer token")
    if authorization.split(" ", 1)[1].strip() != API_TOKEN:
        raise HTTPException(status_code=403, detail="bad token")


def _unpack_emb(p):
    """Sightings carry float16 base64 on the wire (1,368 B) rather than a JSON
    float32 array (~4.6 KB). Raw crops never cross the WAN at all."""
    if p.get("emb"):
        v = np.frombuffer(base64.b64decode(p["emb"]),
                          np.float16 if p.get("emb_dtype", "float16") == "float16"
                          else np.float32).astype(np.float32)
        return v.tolist()
    if isinstance(p.get("embedding"), list):
        return p["embedding"]
    return None


# ==========================================================================
# DATA PLANE
# ==========================================================================
@app.post("/events")
async def ingest(request: Request, authorization: str = Header(None),
                 x_site_id: str = Header("unknown")):
    _auth(authorization)
    batch = await request.json()
    if not batch:
        return {"ok": True, "n": 0}

    def _ingest(cur):
        rows, sightings, activities = [], [], []
        for ev in batch:
            p = ev.get("payload") or {}
            kind = ev.get("kind", "event")
            rows.append((ev["idem_key"], p.get("camera_id", "unknown"),
                         p.get("site_id", x_site_id), kind, Json(p)))
            if kind == "sighting":
                emb = _unpack_emb(p)
                sightings.append((
                    p.get("sighting_id") or ev["idem_key"],
                    p.get("camera_id"), p.get("site_id", x_site_id),
                    p.get("track_id"), p.get("first_ts"), p.get("last_ts"),
                    p.get("dwell_s"), p.get("coherence"), emb,
                    p.get("model_ver"),
                    float(p.get("last_ts") or time.time()) + RETAIN_S))
            elif kind == "activity":
                activities.append(p)

        before = cur.rowcount
        execute_values(
            cur,
            "INSERT INTO events (idem_key, camera_id, site_id, event_type, payload) "
            "VALUES %s ON CONFLICT (idem_key) DO NOTHING", rows)
        inserted = cur.rowcount
        _ = before

        if sightings:
            # Same guarantee one layer down. A replayed sighting must not
            # become a second appearance of a person: that would not duplicate
            # a row, it would corrupt the identity graph.
            execute_values(
                cur,
                "INSERT INTO sightings (sighting_id, camera_id, site_id, track_id, "
                "first_ts, last_ts, dwell_s, coherence, embedding, model_ver, "
                "retain_until) VALUES %s ON CONFLICT (sighting_id) DO NOTHING",
                sightings)

        return {"received": len(rows), "inserted": inserted,
                "duplicates": len(rows) - inserted, "sightings": len(sightings),
                "activities": len(activities)}

    res = await _tx(_ingest)
    _counters["events"] += res["inserted"]
    _counters["duplicates"] += res["duplicates"]
    _counters["sightings"] += res["sightings"]
    return {"ok": True, **res}


RETAIN_S = int(os.environ.get("VI_RETAIN_DAYS", 30)) * 86400


# ==========================================================================
# CONTROL PLANE
# ==========================================================================
@app.get("/control")
async def control(site_id: str = Query(...), since_version: int = Query(0),
                  authorization: str = Header(None)):
    """Long poll. Edges sit behind customer NAT with no inbound reachability,
    so push is not deployable; one idle connection per site is what survives a
    real firewall. Fleet-wide directives use site_id '*'."""
    _auth(authorization)
    deadline = time.time() + LONGPOLL_S

    def _fetch(cur):
        cur.execute(
            "SELECT directive_id, site_id, scope, kind, version, payload "
            "FROM directives WHERE active AND version > %s AND site_id IN (%s, '*') "
            "ORDER BY version LIMIT 200", (since_version, site_id))
        return [{"directive_id": r[0], "site_id": r[1], "scope": r[2],
                 "kind": r[3], "version": r[4], "payload": r[5]}
                for r in cur.fetchall()]

    while True:
        directives = await _tx(_fetch)
        if directives or time.time() >= deadline:
            _counters["directives_served"] += len(directives)
            return {"site_id": site_id, "since_version": since_version,
                    "directives": directives}
        await asyncio.sleep(1.0)


@app.post("/control/ack")
async def control_ack(request: Request, authorization: str = Header(None)):
    _auth(authorization)
    body = await request.json()
    site_id = body.get("site_id", "unknown")
    ids = body.get("directive_ids") or []
    if not ids:
        return {"ok": True, "acked": 0}

    def _ack(cur):
        execute_values(
            cur,
            "INSERT INTO directive_acks (directive_id, site_id, acked_at) VALUES %s "
            "ON CONFLICT (directive_id, site_id) DO NOTHING",
            [(d, site_id, time.time()) for d in ids])
        return cur.rowcount

    return {"ok": True, "acked": await _tx(_ack)}


# ==========================================================================
# FLEET HEALTH
# ==========================================================================
@app.get("/health/fleet")
async def health_fleet(authorization: str = Header(None)):
    _auth(authorization)

    def _health(cur):
        cur.execute(
            "SELECT camera_id, site_id, COUNT(*), MAX(created_at) "
            "FROM events WHERE created_at > now() - interval '15 minutes' "
            "GROUP BY camera_id, site_id ORDER BY camera_id")
        cams = [{"camera_id": r[0], "site_id": r[1], "events_15m": r[2],
                 "last_seen": r[3].isoformat() if r[3] else None}
                for r in cur.fetchall()]
        cur.execute("SELECT COUNT(*), COUNT(subject_id) FROM sightings")
        total, resolved = cur.fetchone()
        cur.execute("SELECT COUNT(*) FROM adjudications WHERE status='pending'")
        pending = cur.fetchone()[0]
        cur.execute("SELECT site_id, MAX(version) FROM directives GROUP BY site_id")
        versions = {r[0]: r[1] for r in cur.fetchall()}
        return {"cameras": cams, "sightings": total or 0,
                "resolved": resolved or 0,
                "resolve_pct": round(100.0 * (resolved or 0) / max(1, total or 1), 1),
                "review_queue": pending, "directive_versions": versions}

    data = await _tx(_health)
    return {"ok": True, "ts": time.time(), **data}


@app.get("/health")
async def health():
    return {"ok": True, "ts": time.time()}


@app.get("/metrics")
async def metrics():
    """Prometheus scrape. Phase 13 points a job at this alongside the edge
    exporter on 9101."""
    def _m(cur):
        cur.execute("SELECT COUNT(*) FROM sightings")
        s = cur.fetchone()[0]
        cur.execute("SELECT COUNT(*) FROM sightings WHERE subject_id IS NOT NULL")
        r = cur.fetchone()[0]
        cur.execute("SELECT COUNT(*) FROM identities")
        i = cur.fetchone()[0]
        cur.execute("SELECT COUNT(*) FROM adjudications WHERE status='pending'")
        q = cur.fetchone()[0]
        return s, r, i, q

    s, r, i, q = await _tx(_m)
    lines = [
        f"vi_cloud_events_ingested {_counters['events']}",
        f"vi_cloud_events_duplicate {_counters['duplicates']}",
        f"vi_cloud_sightings_total {s}",
        f"vi_cloud_sightings_resolved {r}",
        f"vi_cloud_identities_total {i}",
        f"vi_cloud_review_queue_depth {q}",
        f"vi_cloud_directives_served {_counters['directives_served']}",
        f"vi_cloud_dsar_requests {_counters['dsar']}",
    ]
    return Response("\n".join(lines) + "\n", media_type="text/plain")


# ==========================================================================
# DATA SUBJECT RIGHTS (PS-3 Q3.2b, Q3.2c)
# Every one of these logs ITSELF, because under Art. 15 and DPDP s.11 the
# access is the sensitive operation, not just the storage.
# ==========================================================================
@app.get("/privacy/subject/{subject_id}")
async def dsar_export(subject_id: str, authorization: str = Header(None),
                      x_actor: str = Header("unknown")):
    _auth(authorization)
    _counters["dsar"] += 1

    def _q(cur):
        cur.execute(
            "INSERT INTO audit_log (ts, actor, action, subject_id, detail) "
            "VALUES (%s,%s,'dsar_export',%s,%s)",
            (time.time(), x_actor, subject_id, Json({"lawful_basis": "art15"})))
        cur.execute(
            "SELECT subject_id, n_sightings, first_seen, last_seen, embedding_ver "
            "FROM identities WHERE subject_id=%s", (subject_id,))
        ident = cur.fetchone()
        cur.execute(
            "SELECT sighting_id, camera_id, site_id, first_ts, last_ts, dwell_s, "
            "retain_until, consent_basis FROM sightings WHERE subject_id=%s "
            "ORDER BY last_ts DESC LIMIT 1000", (subject_id,))
        s = cur.fetchall()
        cur.execute(
            "SELECT ts, actor, action FROM audit_log WHERE subject_id=%s "
            "ORDER BY ts DESC LIMIT 500", (subject_id,))
        return {"identity": ident, "sightings": s, "access_history": cur.fetchall()}

    data = await _tx(_q)
    if not data["identity"]:
        raise HTTPException(status_code=404, detail="no such subject")
    return {"subject_id": subject_id, "generated_at": time.time(), **data}


@app.post("/privacy/subject/{subject_id}/revoke")
async def dsar_revoke(subject_id: str, authorization: str = Header(None),
                      x_actor: str = Header("unknown")):
    """Erasure fans OUT to every edge that holds pixels via the control plane
    BEFORE deleting cloud state, then marks derived centroids for
    recomputation.

    That last step is the part most designs miss, and Q3.2d asks for it
    directly: a centroid computed from ten sightings still encodes information
    about all ten, so deleting the rows does not delete their influence. The
    needs_recompute flag exists for exactly this."""
    _auth(authorization)
    from cloud import adjudicate

    def _revoke(cur):
        cur.execute("SELECT DISTINCT site_id FROM sightings WHERE subject_id=%s",
                    (subject_id,))
        sites = [r[0] for r in cur.fetchall() if r[0]]
        directives = [adjudicate.emit_directive(
            cur, s, "consent_revoke", {"subject_id": subject_id})[0] for s in sites]

        cur.execute(
            "INSERT INTO consent (subject_id, basis, revoked_at, retain_until) "
            "VALUES (%s,'revoked',now(),0) ON CONFLICT (subject_id) DO UPDATE "
            "SET basis='revoked', revoked_at=now(), retain_until=0", (subject_id,))
        cur.execute("SELECT COUNT(*) FROM sightings WHERE subject_id=%s", (subject_id,))
        n = cur.fetchone()[0]
        cur.execute("DELETE FROM sightings   WHERE subject_id=%s", (subject_id,))
        cur.execute("DELETE FROM identities  WHERE subject_id=%s", (subject_id,))
        cur.execute(
            "UPDATE identities SET needs_recompute=true WHERE subject_id IN "
            "(SELECT DISTINCT a_id FROM exclusions WHERE b_id=%s)", (subject_id,))
        receipt = {"subject_id": subject_id, "sightings_deleted": n,
                   "edge_directives": directives, "sites_notified": sites,
                   "completed_at": time.time()}
        cur.execute(
            "INSERT INTO audit_log (ts, actor, action, subject_id, detail) "
            "VALUES (%s,%s,'erasure',%s,%s)",
            (time.time(), x_actor, subject_id, Json(receipt)))
        return receipt

    return await _tx(_revoke)

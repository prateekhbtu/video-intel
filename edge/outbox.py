"""
Store-and-forward, edge to cloud. REPLACES the Round 1 edge/outbox.py.

WHAT WAS WRONG, VISIBLE IN YOUR OWN DATA
    SELECT MAX(attempts) FROM outbox  ->  0, across all 3,771 rows.

    The column existed and was never incremented, so retry behaviour was
    unobservable by construction. Worse, a non-2xx response fell through to
    the next loop iteration with no sleep and no state change, so a permanent
    4xx became a hot loop that re-POSTed the same rejected batch forever while
    reporting nothing. Peak queue depth reached 934 and the only reason it
    drained is that the cloud never actually returned an error.

THE THREE PROPERTIES THAT MATTER
    1. attempts is incremented and persisted, so "is it retrying" is a query.
    2. 4xx dead-letters. A malformed payload will never become well-formed;
       retrying it forever starves the rows behind it. 5xx and transport
       errors back off exponentially, because those genuinely do recover.
    3. next_attempt is a column, not a sleep. A crash mid-backoff resumes the
       schedule instead of resetting it, and one poisoned row cannot block
       the queue because the SELECT skips rows that are not yet due.

IDEMPOTENCY IS THE CONTRACT
    idem_key is UNIQUE here and ON CONFLICT DO NOTHING in the cloud. That pair
    is what makes a replay after a partition a no-op. PS-3 depends on it more
    than PS-1 did: a duplicated sighting is not a duplicated row, it is a
    phantom second appearance of a person, which corrupts the identity graph.
"""
import json
import time

import requests

from common import telem
from edge import config

MAX_ATTEMPTS = 12          # ~2 h of backoff before a 5xx is given up on
BASE_BACKOFF = 2.0
MAX_BACKOFF = 300.0
ALERT_DEPTH = 500


def enqueue(conn, kind, payload, idem_key):
    """INSERT OR IGNORE, so re-enqueueing the same logical event is free."""
    conn.execute(
        "INSERT OR IGNORE INTO outbox(idem_key, kind, payload, created, next_attempt) "
        "VALUES(?,?,?,?,0)",
        (idem_key, kind, json.dumps(payload, separators=(",", ":")), time.time()))
    conn.commit()


def _backoff(attempts):
    return min(MAX_BACKOFF, BASE_BACKOFF * (2 ** min(attempts, 8)))


def _stats(conn):
    row = conn.execute(
        "SELECT COUNT(*), MIN(created) FROM outbox WHERE sent=0 AND dead=0").fetchone()
    dlq = conn.execute("SELECT COUNT(*) FROM outbox WHERE dead=1").fetchone()[0]
    return row[0] or 0, row[1], dlq


def drain(conn, site_id, api=None, batch=20, rate_cap_bps=None, token=None,
          interval=2.0):
    api = api or config.CLOUD_API
    token = token or config.API_TOKEN
    session = requests.Session()
    session.headers.update({"content-type": "application/json",
                            "authorization": f"Bearer {token}",
                            "x-site-id": site_id})

    while True:
        now = time.time()
        rows = conn.execute(
            "SELECT id, idem_key, kind, payload, attempts FROM outbox "
            "WHERE sent=0 AND dead=0 AND next_attempt <= ? ORDER BY id LIMIT ?",
            (now, batch)).fetchall()

        if not rows:
            depth, oldest, dlq = _stats(conn)
            telem.emit("outbox", site_id=site_id, depth=depth, dlq=dlq,
                       oldest_age_s=round(now - oldest, 1) if oldest else 0)
            time.sleep(interval)
            continue

        body = [{"idem_key": r[1], "kind": r[2], "payload": json.loads(r[3])}
                for r in rows]
        raw = json.dumps(body, separators=(",", ":"))
        ids = [(r[0],) for r in rows]
        status = None

        try:
            with telem.Timer("upload", site_id=site_id, n_events=len(rows),
                             bytes=len(raw)):
                resp = session.post(f"{api}/events", data=raw, timeout=30)
            status = resp.status_code
        except Exception as e:
            telem.emit("upload_error", site_id=site_id, err=repr(e),
                       n_events=len(rows))

        if status is not None and status < 300:
            conn.executemany(
                "UPDATE outbox SET sent=1, sent_ts=?, attempts=attempts+1, "
                "last_status=? WHERE id=?",
                [(now, status, i[0]) for i in ids])
            conn.commit()

        elif status is not None and 400 <= status < 500 and status != 429:
            # Permanent. Retrying a rejected payload forever is how a queue
            # dies; dead-lettering it keeps the rows behind it moving and
            # turns a silent stall into a countable, alertable number.
            conn.executemany(
                "UPDATE outbox SET dead=1, attempts=attempts+1, last_status=? "
                "WHERE id=?", [(status, i[0]) for i in ids])
            conn.commit()
            telem.emit("outbox_dead_letter", site_id=site_id, status=status,
                       n=len(rows), severity="critical",
                       sample=body[0]["idem_key"])

        else:
            # Transient: 5xx, 429, or transport failure. Back off per row so a
            # crash resumes the schedule rather than restarting it.
            for (rid, _k, _kind, _p, attempts) in rows:
                a = (attempts or 0) + 1
                if a >= MAX_ATTEMPTS:
                    conn.execute(
                        "UPDATE outbox SET dead=1, attempts=?, last_status=? WHERE id=?",
                        (a, status, rid))
                    telem.emit("outbox_dead_letter", site_id=site_id,
                               status=status, reason="max_attempts",
                               attempts=a, severity="critical")
                else:
                    conn.execute(
                        "UPDATE outbox SET attempts=?, last_status=?, next_attempt=? "
                        "WHERE id=?", (a, status, now + _backoff(a), rid))
            conn.commit()
            telem.emit("upload_retry", site_id=site_id, status=status,
                       n=len(rows), next_in_s=round(_backoff(rows[0][4] + 1), 1))

        depth, oldest, dlq = _stats(conn)
        telem.emit("outbox", site_id=site_id, depth=depth, dlq=dlq,
                   oldest_age_s=round(time.time() - oldest, 1) if oldest else 0,
                   last_status=status)
        if depth > ALERT_DEPTH:
            telem.emit("outbox_alert", site_id=site_id, depth=depth,
                       threshold=ALERT_DEPTH, severity="warning")

        if rate_cap_bps:
            # Honour the stated 5 Mbps site uplink instead of assuming it.
            time.sleep(len(raw) * 8 / rate_cap_bps)


def purge_sent(conn, keep_hours=24):
    """Sent rows are kept briefly as replay evidence, then dropped. Dead
    letters are NEVER auto-purged: they are the incident record."""
    cutoff = time.time() - keep_hours * 3600
    n = conn.execute("DELETE FROM outbox WHERE sent=1 AND sent_ts < ?",
                     (cutoff,)).rowcount
    conn.commit()
    if n:
        telem.emit("outbox_purge", n=n, keep_hours=keep_hours)
    return n

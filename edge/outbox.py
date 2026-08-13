import time, json, requests
from common import telem

def enqueue(conn, kind, payload, idem_key):
    conn.execute("INSERT OR IGNORE INTO outbox(idem_key,kind,payload,created) VALUES(?,?,?,?)",
                 (idem_key, kind, json.dumps(payload), time.time()))
    conn.commit()

def drain(conn, site_id, api, batch=20, rate_cap_bps=None):
    while True:
        rows = conn.execute(
            "SELECT id,idem_key,kind,payload FROM outbox WHERE sent=0 ORDER BY id LIMIT ?",
            (batch,)
        ).fetchall()
        if not rows:
            time.sleep(2); continue
        
        body = [{"idem_key": r[1], "kind": r[2], "payload": json.loads(r[3])} for r in rows]
        raw = json.dumps(body)
        try:
            with telem.Timer("upload", site_id=site_id, n_events=len(rows), bytes=len(raw)):
                r = requests.post(f"{api}/events", data=raw, timeout=10, headers={"content-type": "application/json"})
                if r.status_code < 300:
                    conn.executemany("UPDATE outbox SET sent=1 WHERE id=?", [(x[0],) for x in rows])
                    conn.commit()
        except Exception:
            time.sleep(5)
        
        depth = conn.execute("SELECT COUNT(*) FROM outbox WHERE sent=0").fetchone()[0]
        oldest = conn.execute("SELECT MIN(created) FROM outbox WHERE sent=0").fetchone()[0]
        
        telem.emit("outbox", site_id=site_id, depth=depth, oldest_age_s=round(time.time()-oldest,1) if oldest else 0)
        
        if rate_cap_bps:
            time.sleep(len(raw) * 8 / rate_cap_bps)

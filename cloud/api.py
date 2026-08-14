import os
from fastapi import FastAPI, Request
import psycopg2
from psycopg2.extras import Json

app = FastAPI()
DB_URL = os.getenv("DATABASE_URL")

def get_db():
    return psycopg2.connect(DB_URL)

@app.post("/events")
async def receive_events(request: Request):
    events = await request.json()
    if not events:
        return {"ok": True}
    
    conn = get_db()
    cur = conn.cursor()
    
    try:
        for ev in events:
            cur.execute("""
                INSERT INTO events (idem_key, camera_id, site_id, event_type, payload)
                VALUES (%s, %s, %s, %s, %s)
                ON CONFLICT (idem_key) DO NOTHING;
            """, (
                ev["idem_key"], 
                ev["payload"].get("camera_id", "unknown"),
                ev["payload"].get("site_id", "unknown"),
                ev["kind"],
                Json(ev["payload"])
            ))
        conn.commit()
    except Exception as e:
        conn.rollback()
        print("Database Error:", e)
        return {"ok": False, "error": str(e)}
    finally:
        cur.close()
        conn.close()
        
    return {"ok": True, "inserted": len(events)}

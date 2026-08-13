import subprocess, threading, time, sqlite3, os, glob
from common import telem

def watch(camera_id, site_id, segdir, conn):
    seen, seq = set(), 0
    while True:
        for p in sorted(glob.glob(f"{segdir}/*.mp4")):
            if p in seen:
                continue
            # skip the file currently being written
            if time.time() - os.path.getmtime(p) < 2:
                continue
            seen.add(p); seq += 1
            size = os.path.getsize(p)
            conn.execute(
                "INSERT OR IGNORE INTO segments(camera_id,seq,path,start_ts,bytes) VALUES(?,?,?,?,?)",
                (camera_id, seq, p, os.path.getctime(p), size)
            )
            conn.commit()
            telem.emit("segment_write", camera_id=camera_id, site_id=site_id, seq=seq, bytes=size, duration_s=10)
        time.sleep(2)

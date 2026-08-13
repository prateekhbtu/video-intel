import argparse, sqlite3, threading, time, os
from common import telem
from edge import recorder, completeness, outbox

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--site", required=True)
    p.add_argument("--cameras", required=True)
    p.add_argument("--db", required=True)
    p.add_argument("--segdir", required=True)
    p.add_argument("--log", required=True)
    p.add_argument("--api", required=True)
    return p.parse_args()

def run_outbox(db, site, api):
    outbox.drain(sqlite3.connect(db, timeout=15), site, api)

def run_recorder(cam, site, segdir, db):
    recorder.watch(cam, site, segdir, sqlite3.connect(db, timeout=15))

def run_completeness(cam, site, db):
    completeness.completeness_loop(cam, site, sqlite3.connect(db, timeout=15))

def main():
    args = parse_args()
    telem.init(args.log)
    
    # Initialize Database Schema
    os.makedirs(os.path.dirname(args.db), exist_ok=True)
    conn = sqlite3.connect(args.db)
    conn.execute("PRAGMA journal_mode = WAL")
    conn.execute("PRAGMA synchronous = NORMAL")
    with open("/workspaces/video-intel/edge/schema.sql") as f:
        conn.executescript(f.read())
    conn.close()
    
    cameras = args.cameras.split(",")
    os.makedirs(args.segdir, exist_ok=True)

    # 1. Start Outbox Drain
    threading.Thread(target=run_outbox, args=(args.db, args.site, args.api), daemon=True).start()

    # 2. Start Recorder and Completeness Loops per Camera
    for cam in cameras:
        cam_segdir = os.path.join(args.segdir, cam)
        os.makedirs(cam_segdir, exist_ok=True)
        
        threading.Thread(target=run_recorder, args=(cam, args.site, cam_segdir, args.db), daemon=True).start()
        threading.Thread(target=run_completeness, args=(cam, args.site, args.db), daemon=True).start()

    print(f"Agent for site {args.site} running. Monitoring {len(cameras)} cameras.")
    while True:
        time.sleep(1)

if __name__ == "__main__":
    main()

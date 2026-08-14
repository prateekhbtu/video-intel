import argparse, sqlite3, threading, time, os, yaml
from common import telem
from edge import recorder, completeness, outbox, cascade

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
    
    conn = sqlite3.connect(args.db)
    with open("/workspaces/video-intel/edge/schema.sql") as f:
        conn.executescript(f.read())
    conn.close()
    
    with open("/workspaces/video-intel/edge/roster.yaml") as f:
        roster = yaml.safe_load(f)
    analyse_cams = set(roster.get("analyse", []))

    cameras = args.cameras.split(",")
    threading.Thread(target=run_outbox, args=(args.db, args.site, args.api), daemon=True).start()

    for cam in cameras:
        cam_segdir = os.path.join(args.segdir, cam)
        os.makedirs(cam_segdir, exist_ok=True)
        
        threading.Thread(target=run_recorder, args=(cam, args.site, cam_segdir, args.db), daemon=True).start()
        threading.Thread(target=run_completeness, args=(cam, args.site, args.db), daemon=True).start()
        
        if cam in analyse_cams:
            rtsp = f"rtsp://127.0.0.1:8554/{cam}"
            model = "/workspaces/video-intel/data/models/rf-detr-nano-int8.onnx"
            zones = "/workspaces/video-intel/edge/zones.yaml"
            threading.Thread(target=cascade.run_cascade, args=(cam, args.site, rtsp, model, args.db, zones), daemon=True).start()

    print(f"Agent {args.site} running. Analyzing {len([c for c in cameras if c in analyse_cams])} / {len(cameras)} cameras.")
    while True:
        time.sleep(1)

if __name__ == "__main__":
    main()

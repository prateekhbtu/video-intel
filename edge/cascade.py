import time, yaml, uuid, sqlite3, traceback
from common import telem
from edge import gate, decode, detect, outbox, tracker

def run_cascade(camera_id, site_id, rtsp_url, model_path, db_path, zones_yaml):
    db_conn = sqlite3.connect(db_path, timeout=15)
    
    try:
        g = gate.MotionGate(camera_id)
        d = detect.Detector(model_path)
        t = tracker.CentroidTracker()
    except Exception as e:
        print(f"[{camera_id}] Init error: {e}")
        return
        
    while True:
        try:
            print(f"[{camera_id}] Attempting to decode {rtsp_url}...")
            for frame, wall_ts in decode.frames(rtsp_url, camera_id):
                if g.passes(frame):
                    boxes = d(frame, camera_id)
                    tracked_objects = t.update(boxes)
                    
                    if len(tracked_objects) > 0:
                        payload = {
                            "camera_id": camera_id,
                            "site_id": site_id,
                            "ts": wall_ts,
                            "active_ids": list(tracked_objects.keys()),
                            "count": len(tracked_objects)
                        }
                        outbox.enqueue(db_conn, "inference_event", payload, str(uuid.uuid4()))
                        print(f"[{camera_id}] Tracked {len(tracked_objects)} objects, queued to outbox.")
        except Exception as e:
            print(f"[{camera_id}] Cascade error: {e}")
            time.sleep(5)

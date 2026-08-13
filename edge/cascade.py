import time, yaml, uuid, sqlite3
from common import telem
from edge import gate, decode, detect, outbox

def run_cascade(camera_id, site_id, rtsp_url, model_path, db_path, zones_yaml):
    db_conn = sqlite3.connect(db_path, timeout=15)
    g = gate.MotionGate(camera_id)
    d = detect.Detector(model_path)
    
    while True:
        try:
            for frame, wall_ts in decode.frames(rtsp_url, camera_id):
                if g.passes(frame):
                    boxes = d(frame, camera_id)
                    if boxes:
                        payload = {
                            "camera_id": camera_id,
                            "site_id": site_id,
                            "ts": wall_ts,
                            "detections": len(boxes)
                        }
                        outbox.enqueue(db_conn, "inference_event", payload, str(uuid.uuid4()))
        except Exception as e:
            time.sleep(5)

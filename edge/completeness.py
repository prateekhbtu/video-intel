import time
from common import telem

def completeness_loop(camera_id, site_id, conn, window=60, seg_s=10):
    while True:
        time.sleep(window)
        expected = window // seg_s
        cutoff = time.time() - window
        actual = conn.execute(
            "SELECT COUNT(*) FROM segments WHERE camera_id=? AND start_ts>?",
            (camera_id, cutoff)
        ).fetchone()[0]
        ratio = actual / expected if expected else 0
        telem.emit("completeness", camera_id=camera_id, site_id=site_id, expected=expected, actual=actual, ratio=round(ratio, 3))

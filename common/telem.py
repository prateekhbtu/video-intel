import json, os, sys, time, uuid, threading

RUN_ID = os.environ.get("RUN_ID") or str(uuid.uuid4())[:8]
lock = threading.Lock()
fh = None

def init(logfile):
    global fh
    os.makedirs(os.path.dirname(logfile), exist_ok=True)
    fh = open(logfile, "a", buffering=1)

def emit(stage, **kw):
    rec = {"ts": round(time.time(), 3), "run_id": RUN_ID, "stage": stage}
    rec.update(kw)
    line = json.dumps(rec, separators=(",", ":"))
    with lock:
        if fh:
            fh.write(line + "\n")
        else:
            sys.stdout.write(line + "\n")

class Timer:
    def __init__(self, stage, **kw):
        self.stage, self.kw = stage, kw
    def __enter__(self):
        self.t0 = time.perf_counter(); return self
    def __exit__(self, *a):
        emit(self.stage, latency_ms=round((time.perf_counter() - self.t0) * 1000, 2), **self.kw)

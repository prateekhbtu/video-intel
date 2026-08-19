"""
Shared, bounded inference pool. NEW FILE.

THE PROBLEM IT SOLVES
    cascade.py created one detect.Detector per camera thread, each with its
    own ONNX session and intra_op_num_threads=2, on a 1 vCPU box.

    Measured on this exact model and machine:
        1 concurrent session   ->  619 ms
        3 concurrent sessions  -> 1902 ms   (3.07x)
    Your logged mean of 1493 ms over 3 cameras is almost entirely queueing
    against oversubscribed CPU, not model cost.

    Worse, there was no backpressure and no drop metric. decode.frames() is a
    generator, so a slow detector silently stalls the decode loop and frames
    pile up in the PyAV buffer. At 4 fps x 3 cameras you demand 12 inferences
    per second and deliver 0.67. The 18x gap was invisible because nothing
    measured it.

THE DESIGN
    One session, N workers sized to the box, one bounded queue.
    When the queue is full the frame is DROPPED and counted. Dropping under
    load is correct behaviour for a real time system; silently falling
    behind is not. drop_rate becomes a first class SLI, and it is the metric
    that would have caught this on day one.
"""
import queue
import threading
import time

from common import telem
from edge import config, detect


class InferencePool:
    def __init__(self, model_path=None, workers=None, queue_max=None, classes=None):
        self.workers_n = workers or config.INFER_WORKERS
        self.q = queue.Queue(maxsize=queue_max or config.INFER_QUEUE_MAX)
        self.classes = classes
        self.detector = detect.Detector(model_path, classes=classes)
        self._lock = threading.Lock()      # ORT sessions are thread safe, but
                                           # we serialise to keep CPU honest
                                           # when workers_n == 1
        self.submitted = 0
        self.dropped = 0
        self.completed = 0
        self._stop = threading.Event()
        self._threads = [
            threading.Thread(target=self._worker, name=f"infer-{i}", daemon=True)
            for i in range(self.workers_n)
        ]
        for t in self._threads:
            t.start()
        threading.Thread(target=self._report_loop, daemon=True).start()
        telem.emit("infer_pool_start", workers=self.workers_n,
                   queue_max=self.q.maxsize, threads_per_worker=config.INFER_THREADS)

    # ---- producer side --------------------------------------------------
    def submit(self, frame, camera_id, wall_ts, callback):
        """Non blocking. Returns True if accepted, False if dropped."""
        self.submitted += 1
        try:
            self.q.put_nowait((frame, camera_id, wall_ts, callback))
            return True
        except queue.Full:
            self.dropped += 1
            telem.emit("infer_drop", camera_id=camera_id, qsize=self.q.qsize())
            return False

    # ---- consumer side --------------------------------------------------
    def _worker(self):
        while not self._stop.is_set():
            try:
                frame, cam, wall_ts, cb = self.q.get(timeout=1.0)
            except queue.Empty:
                continue
            try:
                queue_age_ms = (time.time() - wall_ts) * 1000
                with self._lock:
                    confident, unsure = self.detector(frame, cam)
                self.completed += 1
                telem.emit("infer_done", camera_id=cam,
                           queue_age_ms=round(queue_age_ms, 1),
                           qsize=self.q.qsize())
                cb(confident, unsure, wall_ts)
            except Exception as e:
                telem.emit("infer_error", camera_id=cam, err=repr(e))
            finally:
                self.q.task_done()

    def reload(self, model_path, model_ver=None):
        """Hot-swap the weights behind the pool. Called by ModelManager after
        an atomic activate, which is what makes PS-4 Q4.2b's per-tenant
        rollback take one control poll instead of one deploy.

        The new session is BUILT FIRST and only then swapped in under the
        lock. If the load raises, the pool keeps serving the old model and the
        rollback fails loudly rather than taking the site offline."""
        old = self.detector
        fresh = detect.Detector(model_path, classes=self.classes)
        with self._lock:
            self.detector = fresh
        telem.emit("infer_pool_reload", model_ver=model_ver or fresh.ver,
                   previous=getattr(old, "ver", None), queued=self.q.qsize())
        return fresh

    def _report_loop(self, period=10):
        last = (0, 0, 0)
        while not self._stop.is_set():
            time.sleep(period)
            s, d, c = self.submitted, self.dropped, self.completed
            ds, dd, dc = s - last[0], d - last[1], c - last[2]
            last = (s, d, c)
            telem.emit("infer_pool",
                       submitted=ds, dropped=dd, completed=dc,
                       drop_rate=round(dd / ds, 4) if ds else 0.0,
                       qsize=self.q.qsize(),
                       throughput_fps=round(dc / period, 2))

    def stop(self):
        self._stop.set()

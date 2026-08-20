#!/usr/bin/env python3
"""
Prometheus exporter over the JSONL telemetry. Build plan 13.1.

WHY IT READS THE LOG RATHER THAN INSTRUMENTING A SECOND TIME
    Instrumenting the code separately gives you two sources of truth that
    drift, and the dashboard and the invariant harness then disagree about
    what happened. Reading the same JSONL that tools/verify.py reads means
    they can never disagree: one write path, two readers.
"""
import collections, glob, json, os, pathlib, sys, threading, time
from http.server import HTTPServer, BaseHTTPRequestHandler

REPO = pathlib.Path(os.environ.get("VI_ROOT",
                    pathlib.Path(__file__).resolve().parent.parent))
PORT = int(os.environ.get("VI_EXPORTER_PORT", 9101))
M = collections.defaultdict(float)      # counters and gauges
H = collections.defaultdict(list)       # observations, summarised as quantiles
_lock = threading.Lock()


def tail():
    pos = {}
    while True:
        for path in glob.glob(str(REPO / "data/logs/edge_*.jsonl")):
            try:
                f = open(path)
            except OSError:
                continue
            f.seek(pos.get(path, 0))
            for line in f:
                try:
                    d = json.loads(line)
                except Exception:
                    continue
                s = d.get("stage")
                cam = d.get("camera_id", "all")
                site = d.get("site_id", "unknown")
                with _lock:
                    if s == "detect" and "latency_ms" in d:
                        H[f'vi_detect_latency_ms{{camera="{cam}"}}'].append(d["latency_ms"])
                        M[f'vi_detect_total{{camera="{cam}"}}'] += 1
                    elif s == "motion_gate":
                        M[f'vi_gate_total{{camera="{cam}"}}'] += 1
                        M[f'vi_gate_passed{{camera="{cam}"}}'] += bool(d.get("passed"))
                    elif s == "infer_pool":
                        M[f'vi_infer_dropped{{site="{site}"}}'] += d.get("dropped", 0)
                        M[f'vi_infer_submitted{{site="{site}"}}'] += d.get("submitted", 0)
                    elif s == "outbox":
                        M[f'vi_outbox_depth{{site="{site}"}}'] = d.get("depth", 0)
                        M[f'vi_outbox_dlq{{site="{site}"}}'] = d.get("dlq", 0)
                        M[f'vi_outbox_oldest_s{{site="{site}"}}'] = d.get("oldest_age_s", 0)
                    elif s == "completeness":
                        M[f'vi_completeness_ratio{{camera="{cam}"}}'] = d.get("ratio", 0)
                    elif s == "completeness_fault":
                        M[f'vi_completeness_faults{{camera="{cam}"}}'] += 1
                    elif s == "activity":
                        M[f'vi_activity_total{{camera="{cam}",kind="{d.get("activity")}"}}'] += 1
                    elif s == "sighting":
                        M[f'vi_sightings_total{{camera="{cam}"}}'] += 1
                    elif s == "directive_applied":
                        M[f'vi_directives_applied{{kind="{d.get("kind")}"}}'] += 1
                    elif s == "thread_crash":
                        M[f'vi_thread_crashes{{thread="{d.get("thread")}"}}'] += 1
                    if d.get("camera_id"):
                        M[f'vi_camera_last_event_seconds{{camera="{cam}"}}'] = d.get("ts", 0)
            pos[path] = f.tell()
            f.close()
        time.sleep(3)


class H_(BaseHTTPRequestHandler):
    def do_GET(self):
        out = []
        with _lock:
            for k, v in sorted(M.items()):
                out.append(f"{k} {v}")
            for k, vals in sorted(H.items()):
                if not vals:
                    continue
                v = sorted(vals[-2000:])
                base, lbl = k.split("{", 1)
                lbl = "{" + lbl
                for q, idx in (("0.5", len(v)//2), ("0.95", int(.95*len(v))),
                               ("0.99", int(.99*len(v)))):
                    out.append(f'{base}_q{q.replace(".","")}{lbl} {v[min(idx, len(v)-1)]}')
        body = ("\n".join(out) + "\n").encode()
        self.send_response(200)
        self.send_header("Content-Type", "text/plain; version=0.0.4")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *a):
        pass


if __name__ == "__main__":
    threading.Thread(target=tail, daemon=True).start()
    print(f"exporter on :{PORT}", flush=True)
    HTTPServer(("0.0.0.0", PORT), H_).serve_forever()

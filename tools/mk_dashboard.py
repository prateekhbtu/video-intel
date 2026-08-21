#!/usr/bin/env python3
"""Provision the Grafana dashboard. Build plan 13.3.

Rows map to questions, not to whatever the metric names happened to be. A
dashboard whose panels do not answer a stated question is decoration.
"""
import json, sys, urllib.request, base64

GRAF = "http://127.0.0.1:3000"
AUTH = base64.b64encode(b"admin:admin").decode()

def api(path, payload=None, method="POST"):
    req = urllib.request.Request(GRAF + path, method=method,
        data=json.dumps(payload).encode() if payload else None,
        headers={"Content-Type": "application/json",
                 "Authorization": f"Basic {AUTH}"})
    try:
        return json.load(urllib.request.urlopen(req, timeout=25))
    except Exception as e:
        body = getattr(e, "read", lambda: b"")()
        return {"error": repr(e), "body": body.decode()[:200]}

print(" ", api("/api/datasources", {
    "name": "prom", "type": "prometheus", "access": "proxy",
    "url": "http://host.docker.internal:9090", "isDefault": True}).get("message", "datasource ok"))

def panel(title, exprs, x, y, w=8, h=6, unit="short", legend=None):
    return {"type": "timeseries", "title": title,
            "gridPos": {"x": x, "y": y, "w": w, "h": h},
            "fieldConfig": {"defaults": {"unit": unit}, "overrides": []},
            "targets": [{"expr": e, "legendFormat": (legend or "{{camera}}{{site}}{{kind}}"),
                         "refId": chr(65+i)} for i, e in enumerate(exprs)]}

def row(title, y):
    return {"type": "row", "title": title, "gridPos": {"x": 0, "y": y, "w": 24, "h": 1},
            "collapsed": False, "panels": []}

panels, y = [], 0
# ---- Fleet health (Q1.3) ------------------------------------------------
panels.append(row("Fleet health  ·  Q1.3", y)); y += 1
panels += [
 panel("Cameras reporting", ["count(vi_camera_last_event_seconds)"], 0, y, legend="cameras"),
 panel("Seconds since last event", ["time() - vi_camera_last_event_seconds"], 8, y, unit="s"),
 panel("Completeness ratio (bound 1.05)", ["vi_completeness_ratio"], 16, y),
]; y += 6
# ---- Pipeline (Q4.3) ----------------------------------------------------
panels.append(row("Pipeline  ·  Q4.3", y)); y += 1
panels += [
 panel("Detect latency p50 / p95", ["vi_detect_latency_ms_q50", "vi_detect_latency_ms_q95"], 0, y, unit="ms"),
 panel("Motion gate pass rate", ["vi_gate_passed / clamp_min(vi_gate_total,1)"], 8, y, unit="percentunit"),
 panel("Inference drop rate", ["vi_infer_dropped / clamp_min(vi_infer_submitted,1)"], 16, y, unit="percentunit"),
]; y += 6
# ---- Delivery (Q1.2) ----------------------------------------------------
panels.append(row("Delivery  ·  Q1.2", y)); y += 1
panels += [
 panel("Outbox depth", ["vi_outbox_depth"], 0, y),
 panel("Oldest undelivered (s)", ["vi_outbox_oldest_s"], 8, y, unit="s"),
 panel("Dead letters", ["vi_outbox_dlq"], 16, y),
]; y += 6
# ---- Identity (Q3.1, Q3.3) ---------------------------------------------
panels.append(row("Identity  ·  Q3.1, Q3.3", y)); y += 1
panels += [
 panel("Sightings emitted", ["vi_sightings_total"], 0, y),
 panel("Sightings per minute", ["rate(vi_sightings_total[5m]) * 60"], 8, y),
 panel("Activity events", ["vi_activity_total"], 16, y),
]; y += 6
# ---- Lifecycle (Q4.2) ---------------------------------------------------
panels.append(row("Lifecycle  ·  Q4.2", y)); y += 1
panels += [
 panel("Control directives applied", ["vi_directives_applied"], 0, y, legend="{{kind}}"),
 panel("Supervised thread crashes", ["vi_thread_crashes"], 8, y),
 panel("Completeness faults", ["vi_completeness_faults"], 16, y),
]

dash = {"dashboard": {"title": "video-intel fleet", "uid": "video-intel",
                      "timezone": "browser", "schemaVersion": 39,
                      "refresh": "10s", "time": {"from": "now-6h", "to": "now"},
                      "panels": panels},
        "overwrite": True}
r = api("/api/dashboards/db", dash)
print("  dashboard:", r.get("status", r))
print(f"  rows: Fleet health, Pipeline, Delivery, Identity, Lifecycle "
      f"({sum(1 for p in panels if p['type']!='row')} panels)")

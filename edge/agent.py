"""
Edge agent entry point. REPLACES the Round 1 edge/agent.py.

THREE THINGS WERE WRONG, AND ONE OF THEM COST A WHOLE SITE
    1. Hardcoded /workspaces/video-intel paths in three places, which makes
       the repo Codespaces-only and directly contradicts the "operational
       within 30 minutes on arbitrary hardware" requirement. Everything now
       resolves through edge/config.py.

    2. Unsupervised daemon threads. A thread that raised died silently and the
       agent kept reporting healthy, because nothing was watching. edge_b.db
       ended a five-hour run with zero segments and the log had fourteen
       lines in it. Nothing alerted, because "a thread stopped" was not an
       observable event. Every thread now runs under supervise(), which
       restarts it with backoff and emits thread_crash, and tools/verify.py
       fails the run on any such record.

    3. No boot heartbeat. verify.py's all_configured_cameras_report check
       needs to know how many cameras were SUPPOSED to produce detections
       before it can tell you that one did not. agent_ready carries that
       number, which is the check that would have caught the empty edge_b.db
       on day one instead of at the post-mortem.

WHAT STARTS HERE
    one shared inference pool     sized to the box, not one session per camera
    one embedder                  shared, tracklet-scoped
    one policy store              per-tenant versioned flags
    one gallery cache             identities, exclusions, class prototypes
    one model manager             staged, checksummed, atomically swapped
    one retention manager         TTL sweep and disk-pressure eviction
    one control-plane inbox       the return path all of Round 2 depends on
    per camera                    recorder, completeness, and (if analysed) a
                                  cascade
"""
import argparse
import os
import sqlite3
import threading
import time

import yaml

from common import telem
from edge import (cascade, completeness, config, embed, gallery as gallery_mod,
                  inbox, infer_pool, models, outbox, policy, recorder, retention)

HEARTBEAT_S = 30


def connect(db_path):
    """One connection per thread. WAL lets readers and the writer proceed
    concurrently, which is what makes a single SQLite file workable for a
    dozen threads on one box."""
    conn = sqlite3.connect(db_path, timeout=30, check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("PRAGMA busy_timeout=30000")
    return conn


def supervise(name, target, *args, backoff=5.0, max_backoff=120.0, **kwargs):
    """Run target forever. A crash is logged as an EVENT, not a stack trace on
    a terminal nobody is reading, and the thread comes back."""
    def loop():
        delay = backoff
        while True:
            t0 = time.time()
            try:
                target(*args, **kwargs)
                telem.emit("thread_exit", thread=name, ran_s=round(time.time() - t0, 1))
            except Exception as e:
                telem.emit("thread_crash", thread=name, err=repr(e),
                           ran_s=round(time.time() - t0, 1), severity="critical")
            delay = backoff if time.time() - t0 > 60 else min(max_backoff, delay * 2)
            time.sleep(delay)

    t = threading.Thread(target=loop, name=name, daemon=True)
    t.start()
    return t


def load_roster(path=None):
    """Supports the Round 2 `sites:` topology and the flat Round 1 form, so an
    old roster still boots instead of dying on a KeyError."""
    with open(path or config.ROSTER) as f:
        r = yaml.safe_load(f) or {}
    return {
        "sites": r.get("sites") or {},
        "analyse": set(r.get("analyse") or []),
        "reid_enabled": set(r.get("reid_enabled") or r.get("analyse") or []),
        "record_all": bool(r.get("record_all", True)),
    }


def apply_schema(db_path):
    head = open(config.EDGE_SCHEMA).readline()
    if "Edge schema" not in head:
        raise SystemExit(
            f"{config.EDGE_SCHEMA} does not look like the edge schema (line 1: "
            f"{head.strip()!r}). Applying the Postgres schema to SQLite aborts "
            f"on CREATE EXTENSION and leaves a half-built database.")
    conn = connect(db_path)
    conn.executescript(open(config.EDGE_SCHEMA).read())
    conn.commit()
    n = conn.execute("SELECT COUNT(*) FROM sqlite_master WHERE type='table'").fetchone()[0]
    conn.close()
    return n


def parse_args():
    p = argparse.ArgumentParser(description="video-intel edge agent")
    p.add_argument("--site", required=True)
    p.add_argument("--cameras", default=None,
                   help="comma list; defaults to the roster's cameras for this site")
    p.add_argument("--db", default=None)
    p.add_argument("--segdir", default=None)
    p.add_argument("--log", default=None)
    p.add_argument("--api", default=None)
    p.add_argument("--roster", default=None)
    p.add_argument("--no-reid", action="store_true")
    return p.parse_args()


def main():
    a = parse_args()
    site = a.site
    db_path = a.db or str(config.DATA / f"edge_{site}.db")
    segroot = a.segdir or str(config.SEG / site)
    log_path = a.log or str(config.LOGS / f"edge_{site}.jsonl")
    api = a.api or config.CLOUD_API

    telem.init(log_path)
    os.makedirs(os.path.dirname(db_path) or ".", exist_ok=True)
    os.makedirs(segroot, exist_ok=True)

    n_tables = apply_schema(db_path)
    roster = load_roster(a.roster)
    site_cfg = roster["sites"].get(site, {})
    tenant = site_cfg.get("tenant", os.environ.get("VI_TENANT", "default"))

    if a.cameras:
        cameras = [c.strip() for c in a.cameras.split(",") if c.strip()]
    else:
        cameras = list(site_cfg.get("cameras") or
                       [c for c in roster["analyse"] if c.startswith(f"{site}_")])
    analysed = [c for c in cameras if c in roster["analyse"]]

    telem.emit("agent_boot", site_id=site, tenant=tenant, db=db_path,
               tables=n_tables, cameras=len(cameras), analysed=len(analysed),
               cpu_count=config.CPU_COUNT,
               infer_workers=config.INFER_WORKERS,
               infer_threads=config.INFER_THREADS)

    # ---- shared, long-lived components ----------------------------------
    ctl_conn = connect(db_path)
    pol = policy.PolicyStore(ctl_conn, tenant_id=tenant)
    gal = gallery_mod.EdgeGallery(ctl_conn)

    pool = infer_pool.InferencePool(classes={1})        # COCO person
    mgr = models.ModelManager(on_activate=lambda ver, path: pool.reload(path, ver))
    emb = None if a.no_reid else embed.Embedder()
    if emb is not None and emb.kind == "fallback":
        telem.emit("reid_fallback_active", severity="warning",
                   note="HSV placeholder descriptor, NOT re-identification. "
                        "Export a ReID ONNX to data/models/reid-osnet-int8.onnx.")

    ret = retention.RetentionManager(
        ctl_conn, site, days=int(os.environ.get("VI_RETAIN_DAYS", 30)),
        seg_root=segroot)

    ctx = {"policy": pol, "gallery": gal, "retention": ret,
           "model_manager": mgr, "detectors": {"*": pool.detector},
           "pool": pool, "embedder": emb, "site_id": site, "tenant": tenant}

    # ---- threads ---------------------------------------------------------
    supervise("outbox", outbox.drain, connect(db_path), site, api)
    supervise("inbox", inbox.poll_loop, connect(db_path), site, ctx, api)
    supervise("retention", retention.retention_loop, ret)

    for cam in cameras:
        cam_seg = os.path.join(segroot, cam)
        os.makedirs(cam_seg, exist_ok=True)
        supervise(f"recorder:{cam}", recorder.watch, cam, site, cam_seg, connect(db_path))
        supervise(f"completeness:{cam}", completeness.completeness_loop,
                  cam, site, connect(db_path))

    for cam in analysed:
        rtsp = f"{config.RTSP_BASE}/{cam}"
        supervise(f"cascade:{cam}", cascade.run_cascade, cam, site, rtsp,
                  connect(db_path), pool, pol, emb,
                  cam in roster["reid_enabled"] and emb is not None)

    # THE record that makes a silent site detectable. verify.py compares the
    # cameras that actually produced detections against `analysed`.
    telem.emit("agent_ready", site_id=site, tenant=tenant,
               cameras=len(cameras), analysed=len(analysed),
               analysed_cameras=sorted(analysed),
               reid=bool(emb) and emb.kind == "onnx",
               model_ver=pool.detector.ver)
    print(f"agent {site}: {len(cameras)} cameras, {len(analysed)} analysed, "
          f"reid={'onnx' if emb and emb.kind == 'onnx' else 'off/fallback'}")

    hb = connect(db_path)
    while True:
        time.sleep(HEARTBEAT_S)
        try:
            depth = hb.execute(
                "SELECT COUNT(*) FROM outbox WHERE sent=0 AND dead=0").fetchone()[0]
            dead = hb.execute("SELECT COUNT(*) FROM outbox WHERE dead=1").fetchone()[0]
            segs, seg_bytes = hb.execute(
                "SELECT COUNT(*), COALESCE(SUM(bytes),0) FROM segments").fetchone()
            sights = hb.execute("SELECT COUNT(*) FROM sightings").fetchone()[0]
            telem.emit("heartbeat", site_id=site, outbox_depth=depth,
                       outbox_dead=dead, segments=segs,
                       segment_gb=round(seg_bytes / 1e9, 3), sightings=sights,
                       threads_alive=threading.active_count(),
                       infer_submitted=pool.submitted, infer_dropped=pool.dropped)
        except Exception as e:
            telem.emit("heartbeat_error", site_id=site, err=repr(e))


if __name__ == "__main__":
    main()

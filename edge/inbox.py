"""
Control plane inbox. NEW FILE. This is the single most important addition
for Round 2.

WHY
    The PS-1 system is strictly one directional: edge -> cloud. Every Round 2
    question needs the return path.

        PS-3 Q3.1d  update identity embeddings as people age    -> gallery delta
        PS-3 Q3.2a  enforce consent at scale                    -> consent revoke
        PS-3 Q3.2c  time based deletion at scale                -> retention policy
        PS-3 Q3.3b  immediate fix for one bad match             -> exclusion pair
        PS-4 Q4.1c  introduce a new customer specific class     -> class manifest
        PS-4 Q4.2b  roll back for some customers, forward for others -> model pin
        PS-2 Q2.3c  customer tunable feedback loop              -> threshold update

    Seven separate questions, one mechanism. That is the answer worth giving:
    they are not seven features, they are one versioned control channel.

SEMANTICS: exactly the outbox, mirrored
    versioned, idempotent, resumable across partition, acknowledged.
    The edge stores applied_version per (scope, key). It polls with that
    version, applies anything newer inside a transaction, and acks. If the
    link dies mid-apply the next poll replays from the last acked version,
    so a partition costs latency and never correctness. Same guarantee the
    outbox gives upward, which is why it is safe to reason about both with
    one mental model.

PULL, NOT PUSH
    Edges sit behind customer NAT with no inbound reachability, so push is
    not deployable. Long poll costs one idle connection per site and is what
    survives a real firewall.
"""
import json
import time

import requests

from common import telem
from edge import config

APPLIERS = {}


def applier(kind):
    def deco(fn):
        APPLIERS[kind] = fn
        return fn
    return deco


def current_version(conn, scope="site"):
    row = conn.execute(
        "SELECT COALESCE(MAX(version),0) FROM applied_directives WHERE scope=?",
        (scope,)).fetchone()
    return row[0] if row else 0


def poll_loop(conn, site_id, ctx, api=None, token=None, interval=15):
    """ctx carries live handles the appliers mutate: detectors, policy cache,
    gallery cache, retention config."""
    api = api or config.CLOUD_API
    token = token or config.API_TOKEN
    session = requests.Session()
    session.headers.update({"authorization": f"Bearer {token}",
                            "x-site-id": site_id})

    while True:
        try:
            since = current_version(conn)
            r = session.get(f"{api}/control",
                            params={"site_id": site_id, "since_version": since},
                            timeout=45)
            if r.status_code >= 300:
                telem.emit("inbox_error", site_id=site_id, status=r.status_code)
                time.sleep(interval)
                continue

            directives = r.json().get("directives", [])
            if not directives:
                time.sleep(interval)
                continue

            applied = []
            for d in sorted(directives, key=lambda x: x["version"]):
                kind, ver = d["kind"], d["version"]
                already = conn.execute(
                    "SELECT 1 FROM applied_directives WHERE directive_id=?",
                    (d["directive_id"],)).fetchone()
                if already:
                    continue                     # idempotent replay
                fn = APPLIERS.get(kind)
                if not fn:
                    telem.emit("inbox_unknown_kind", kind=kind, version=ver)
                    continue
                try:
                    fn(conn, ctx, d["payload"])
                    conn.execute(
                        "INSERT INTO applied_directives"
                        "(directive_id,scope,kind,version,payload,applied_ts,ok) "
                        "VALUES(?,?,?,?,?,?,1)",
                        (d["directive_id"], d.get("scope", "site"), kind, ver,
                         json.dumps(d["payload"]), time.time()))
                    conn.commit()
                    applied.append(d["directive_id"])
                    telem.emit("directive_applied", kind=kind, version=ver,
                               directive_id=d["directive_id"])
                except Exception as e:
                    conn.execute(
                        "INSERT OR REPLACE INTO applied_directives"
                        "(directive_id,scope,kind,version,payload,applied_ts,ok,err) "
                        "VALUES(?,?,?,?,?,?,0,?)",
                        (d["directive_id"], d.get("scope", "site"), kind, ver,
                         json.dumps(d["payload"]), time.time(), repr(e)))
                    conn.commit()
                    telem.emit("directive_failed", kind=kind, version=ver,
                               err=repr(e), severity="critical")

            if applied:
                session.post(f"{api}/control/ack",
                             json={"site_id": site_id, "directive_ids": applied},
                             timeout=15)
        except Exception as e:
            telem.emit("inbox_error", site_id=site_id, err=repr(e))
            time.sleep(interval)


# --------------------------------------------------------------------------
# Appliers. Each one is small on purpose: the value is in the shared channel,
# not in any individual handler.
# --------------------------------------------------------------------------

@applier("threshold")
def _apply_threshold(conn, ctx, p):
    """PS-2 Q2.3c. Customer turns the sensitivity dial, edge honours it."""
    cam = p.get("camera_id", "*")
    for key in ("conf_high", "conf_low", "min_hits", "dwell_seconds"):
        if key in p:
            ctx["policy"].set(conn, cam, "detect", key, p[key], p["version"])
    if cam in ctx["detectors"]:
        ctx["detectors"][cam].set_thresholds(p.get("conf_high"), p.get("conf_low"))


@applier("model_pin")
def _apply_model_pin(conn, ctx, p):
    """PS-4 Q4.2b. Per tenant model version. Rolling Customer A back while
    Customer C rolls forward is a policy write, not a deploy. The weights
    file is fetched and checksummed BEFORE the pin flips, so a failed
    download can never leave a site with no model."""
    ctx["model_manager"].stage(p["model_uri"], p["sha256"], p["model_ver"])
    ctx["model_manager"].activate(p["model_ver"])          # atomic swap
    ctx["policy"].set(conn, "*", "model", "active_ver", p["model_ver"], p["version"])


@applier("consent_revoke")
def _apply_consent_revoke(conn, ctx, p):
    """PS-3 Q3.2a and Q3.2c. Enforcement has to reach the EDGE, because the
    edge is what holds raw pixels. A cloud only deletion is theatre."""
    conn.execute("INSERT OR REPLACE INTO consent(subject_id, basis, revoked_ts, "
                 "retain_until) VALUES(?,?,?,?)",
                 (p["subject_id"], "revoked", time.time(), 0))
    conn.commit()
    ctx["retention"].purge_subject(p["subject_id"])


@applier("exclusion_pair")
def _apply_exclusion_pair(conn, ctx, p):
    """PS-3 Q3.3b. The immediate fix for 'you matched John as David': a hard
    never-match constraint on that specific pair, live within one poll, with
    no retrain and no global threshold change that would hurt everyone else."""
    conn.execute("INSERT OR IGNORE INTO exclusions(a_id,b_id,reason,created) "
                 "VALUES(?,?,?,?)",
                 (p["a_id"], p["b_id"], p.get("reason", "customer_report"), time.time()))
    conn.commit()
    ctx["gallery"].add_exclusion(p["a_id"], p["b_id"])


@applier("class_manifest")
def _apply_class_manifest(conn, ctx, p):
    """PS-4 Q4.1c. A new customer specific class arrives as a manifest of
    class ids plus, for open set classes, a set of reference embeddings.
    Closed set classes need a retrain; embedding classes are live instantly.
    That difference is the whole answer to Q4.1b."""
    ctx["policy"].set(conn, "*", "classes", "manifest",
                      json.dumps(p["classes"]), p["version"])
    if p.get("reference_embeddings"):
        ctx["gallery"].upsert_prototypes(p["reference_embeddings"])


@applier("gallery_delta")
def _apply_gallery_delta(conn, ctx, p):
    """PS-3 Q3.1d. Representation drift. The cloud recomputes centroids as
    people age or change appearance and ships only the delta, so a site
    cache stays fresh without ever downloading the full gallery."""
    ctx["gallery"].apply_delta(p["upserts"], p["deletes"], p["version"])


@applier("retention")
def _apply_retention(conn, ctx, p):
    ctx["retention"].configure(days=p["days"], classes=p.get("classes"))
    ctx["policy"].set(conn, "*", "retention", "days", p["days"], p["version"])

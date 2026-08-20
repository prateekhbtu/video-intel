#!/usr/bin/env python3
"""
Model registry. Build plan section 10.1.

Nothing ships unregistered, because a rollback needs a known-good artefact to
roll back TO. A version string with no checksum behind it is a promise, not a
guarantee: it cannot tell you whether the file on the edge is the file you
tested. The sha256 is what makes "roll back to v1.0-baseline" a verifiable
instruction rather than a hopeful one.
"""
import hashlib, json, os, pathlib, sys

REPO = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
import psycopg2

DDL = """CREATE TABLE IF NOT EXISTS model_registry (
    model_ver     TEXT PRIMARY KEY,
    path          TEXT NOT NULL,
    sha256        TEXT NOT NULL,
    bytes         BIGINT NOT NULL,
    registered_at TIMESTAMPTZ DEFAULT now(),
    notes         TEXT)"""

def main():
    if len(sys.argv) < 3:
        sys.exit("usage: register_model.py <path> <model_ver> [notes]")
    path, ver = sys.argv[1], sys.argv[2]
    notes = sys.argv[3] if len(sys.argv) > 3 else None
    p = pathlib.Path(path)
    if not p.exists():
        sys.exit(f"no such model: {path}")
    sha = hashlib.sha256(p.read_bytes()).hexdigest()
    size = p.stat().st_size
    cx = psycopg2.connect(os.environ["DATABASE_URL"]); cx.autocommit = True
    cur = cx.cursor()
    cur.execute(DDL)
    cur.execute("INSERT INTO model_registry (model_ver, path, sha256, bytes, notes) "
                "VALUES (%s,%s,%s,%s,%s) ON CONFLICT (model_ver) DO NOTHING",
                (ver, str(p), sha, size, notes))
    print(json.dumps({"model_ver": ver, "sha256": sha[:16], "mb": round(size/1e6, 2),
                      "registered": cur.rowcount == 1}))
    return 0

if __name__ == "__main__":
    sys.exit(main())

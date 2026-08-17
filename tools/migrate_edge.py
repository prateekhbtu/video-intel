#!/usr/bin/env python3
"""
Edge SQLite migration. RUN THIS BEFORE edge/schema.sql, NOT INSTEAD OF IT.

WHY IT IS NEEDED
    edge/schema.sql uses CREATE TABLE IF NOT EXISTS throughout, which is
    idempotent for NEW tables and a silent no-op for tables that already
    exist. Your edge_a.db and edge_b.db were created by the Round 1 schema, so
    the three carried-over tables are missing every column the Round 2 code
    expects:

        policies  ->  tenant_id, updated   (and the PRIMARY KEY is wrong)
        segments  ->  duration_s, legal_hold
        outbox    ->  sent_ts, dead, last_status, next_attempt

    Symptom if you skip this: policy.py raises
    "table policies has no column named tenant_id" the moment the first
    control directive lands, and outbox.py raises "no such column:
    next_attempt" on its first drain loop.

USAGE
    python tools/migrate_edge.py data/edge_a.db
    python tools/migrate_edge.py data/edge_b.db
    # then, and only then:
    sqlite3 data/edge_a.db < edge/schema.sql

Safe to run repeatedly. Takes a timestamped backup first.
"""
import sqlite3
import shutil
import sys
import time
from pathlib import Path


def cols(conn, table):
    return {r[1] for r in conn.execute(f"PRAGMA table_info({table})")}


def has_table(conn, table):
    return conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (table,)
    ).fetchone() is not None


def add_column(conn, table, name, decl):
    """SQLite ALTER TABLE ADD COLUMN is cheap and non-locking. It cannot add a
    non-constant DEFAULT, which is why every default below is a literal."""
    if not has_table(conn, table):
        print(f"  {table}: absent, schema.sql will create it")
        return False
    if name in cols(conn, table):
        print(f"  {table}.{name}: already present")
        return False
    conn.execute(f"ALTER TABLE {table} ADD COLUMN {name} {decl}")
    print(f"  {table}.{name}: ADDED ({decl})")
    return True


def rebuild_policies(conn):
    """The policies primary key must widen from (camera_id, zone, field) to
    (tenant_id, camera_id, zone, field). SQLite cannot alter a primary key, so
    this is the standard create-copy-drop-rename dance. Wrapped in the caller's
    transaction, so a failure leaves the original intact."""
    if not has_table(conn, "policies"):
        print("  policies: absent, schema.sql will create it")
        return
    existing = cols(conn, "policies")
    if "tenant_id" in existing:
        print("  policies: already migrated")
        return

    n = conn.execute("SELECT COUNT(*) FROM policies").fetchone()[0]
    conn.execute("""
        CREATE TABLE policies_new (
            tenant_id TEXT    NOT NULL DEFAULT 'default',
            camera_id TEXT    NOT NULL,
            zone      TEXT    NOT NULL,
            field     TEXT    NOT NULL,
            value     TEXT    NOT NULL,
            version   INTEGER NOT NULL,
            updated   REAL,
            PRIMARY KEY (tenant_id, camera_id, zone, field)
        )""")
    conn.execute("""
        INSERT INTO policies_new (tenant_id, camera_id, zone, field, value, version, updated)
        SELECT 'default', camera_id, zone, field, value, version, ?
        FROM policies""", (time.time(),))
    conn.execute("DROP TABLE policies")
    conn.execute("ALTER TABLE policies_new RENAME TO policies")
    print(f"  policies: REBUILT with tenant_id in the primary key ({n} rows preserved)")


def backfill_durations(conn):
    """completeness.py measures coverage from recorded media duration. Round 1
    never stored it, so old rows would read as zero coverage and trip the
    bounds assertion. Estimate from the segment window, which is exact for
    every segment except the last one per camera."""
    if not has_table(conn, "segments") or "duration_s" not in cols(conn, "segments"):
        return
    n = conn.execute(
        "UPDATE segments SET duration_s = MAX(0, COALESCE(end_ts, start_ts) - start_ts) "
        "WHERE (duration_s IS NULL OR duration_s = 0) AND end_ts IS NOT NULL"
    ).rowcount
    m = conn.execute(
        "UPDATE segments SET duration_s = 10.0 "
        "WHERE (duration_s IS NULL OR duration_s = 0) AND end_ts IS NULL"
    ).rowcount
    print(f"  segments.duration_s: backfilled {n} exact, {m} assumed at 10s")


def main(path):
    p = Path(path)
    if not p.exists():
        print(f"{path}: does not exist yet, nothing to migrate")
        return 0

    backup = p.with_suffix(p.suffix + f".bak-{int(time.time())}")
    shutil.copy2(p, backup)
    print(f"backup: {backup}")

    conn = sqlite3.connect(path)
    conn.execute("PRAGMA foreign_keys=OFF")
    print(f"\nmigrating {path}")
    try:
        with conn:
            print(" segments:")
            add_column(conn, "segments", "duration_s", "REAL DEFAULT 0")
            add_column(conn, "segments", "legal_hold", "INTEGER DEFAULT 0")
            backfill_durations(conn)

            print(" outbox:")
            add_column(conn, "outbox", "sent_ts", "REAL")
            add_column(conn, "outbox", "dead", "INTEGER DEFAULT 0")
            add_column(conn, "outbox", "last_status", "INTEGER")
            add_column(conn, "outbox", "next_attempt", "REAL DEFAULT 0")

            print(" policies:")
            rebuild_policies(conn)
    except Exception as e:
        print(f"\nMIGRATION FAILED: {e!r}")
        print(f"restore with:  cp {backup} {path}")
        return 1

    # Verify the post-conditions the Round 2 code depends on.
    print("\nverification:")
    required = {
        "segments": {"duration_s", "legal_hold"},
        "outbox":   {"sent_ts", "dead", "last_status", "next_attempt"},
        "policies": {"tenant_id", "updated"},
    }
    ok = True
    for table, need in required.items():
        if not has_table(conn, table):
            print(f"  {table}: absent (schema.sql will create it)")
            continue
        missing = need - cols(conn, table)
        if missing:
            print(f"  {table}: STILL MISSING {sorted(missing)}")
            ok = False
        else:
            print(f"  {table}: ok")

    pk = [r[1] for r in conn.execute("PRAGMA table_info(policies)") if r[5]]
    if has_table(conn, "policies"):
        print(f"  policies primary key: {pk}")
        if pk and pk[0] != "tenant_id":
            print("  policies: PRIMARY KEY NOT WIDENED")
            ok = False

    conn.close()
    print("\nMIGRATION OK" if ok else "\nMIGRATION INCOMPLETE")
    return 0 if ok else 1


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(2)
    sys.exit(max(main(a) for a in sys.argv[1:]))

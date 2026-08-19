-- Edge schema (SQLite). 11 tables. Additive and idempotent.
--
-- PAIRED WITH tools/migrate_edge.py, WHICH MUST RUN FIRST ON AN EXISTING DB.
--     Every statement here is CREATE TABLE IF NOT EXISTS, which is idempotent
--     for new tables and a SILENT NO-OP for tables that already exist. The
--     Round 1 databases already have segments, outbox and policies, so the
--     columns added below would never appear on them. migrate_edge.py does the
--     ALTER TABLE work; this file does the CREATE work. Run migrate first.
--
-- DO NOT APPLY cloud/schema.sql TO SQLITE. It is Postgres and its
-- CREATE EXTENSION line aborts the script, leaving a half-built database.

-- ===================== RECORDING ========================================
CREATE TABLE IF NOT EXISTS segments (
    camera_id  TEXT    NOT NULL,
    seq        INTEGER NOT NULL,
    path       TEXT    NOT NULL,
    start_ts   REAL    NOT NULL,
    end_ts     REAL,
    duration_s REAL    DEFAULT 0,   -- REAL duration, probed, not assumed 10.
                                    -- completeness.py divides by this, and the
                                    -- Round 1 ratio of 2.33 came from assuming
                                    -- it instead of measuring it.
    bytes      INTEGER DEFAULT 0,
    uploaded   INTEGER DEFAULT 0,
    legal_hold INTEGER DEFAULT 0,   -- retention.py will never delete these
    PRIMARY KEY (camera_id, seq)
);
CREATE INDEX IF NOT EXISTS idx_seg_up    ON segments(camera_id, uploaded);
CREATE INDEX IF NOT EXISTS idx_seg_start ON segments(camera_id, start_ts);
CREATE INDEX IF NOT EXISTS idx_seg_ttl   ON segments(start_ts) WHERE legal_hold = 0;

-- ===================== STORE AND FORWARD (UP) ===========================
-- idem_key UNIQUE is the single most load-bearing constraint in the repo.
-- It is what makes a partition replay a no-op instead of a phantom second
-- appearance of a person, which would corrupt the identity graph rather than
-- merely duplicating a row.
CREATE TABLE IF NOT EXISTS outbox (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    idem_key     TEXT UNIQUE NOT NULL,
    kind         TEXT NOT NULL,
    payload      TEXT NOT NULL,
    created      REAL NOT NULL,
    attempts     INTEGER DEFAULT 0,   -- was never incremented in Round 1, so
                                      -- retry behaviour was unobservable
    sent         INTEGER DEFAULT 0,
    sent_ts      REAL,
    dead         INTEGER DEFAULT 0,   -- 4xx: will never succeed, stop retrying
    last_status  INTEGER,
    next_attempt REAL DEFAULT 0       -- backoff deadline; 0 means "now"
);
CREATE INDEX IF NOT EXISTS idx_out_pending ON outbox(sent, dead, next_attempt, id);
CREATE INDEX IF NOT EXISTS idx_out_dead    ON outbox(dead) WHERE dead = 1;

-- ===================== CONTROL PLANE (DOWN) =============================
-- The mirror image of outbox. directive_id is the downward idem_key, which is
-- what makes an interrupted apply safe to replay from the last acked version.
CREATE TABLE IF NOT EXISTS applied_directives (
    directive_id TEXT PRIMARY KEY,
    scope        TEXT NOT NULL DEFAULT 'site',
    kind         TEXT NOT NULL,
    version      INTEGER NOT NULL,
    payload      TEXT,
    applied_ts   REAL NOT NULL,
    ok           INTEGER DEFAULT 1,
    err          TEXT
);
CREATE INDEX IF NOT EXISTS idx_appdir_scope ON applied_directives(scope, version);
CREATE INDEX IF NOT EXISTS idx_appdir_kind  ON applied_directives(kind, applied_ts DESC);

-- Per tenant / per camera versioned flags. This table existed in Round 1 with
-- zero rows in it. Widened with tenant_id it IS the per-customer feature flag
-- system PS-4 Q4.2b asks for: rolling Customer A back while Customer C rolls
-- forward is a row write with a version bump, not a deploy.
CREATE TABLE IF NOT EXISTS policies (
    tenant_id TEXT    NOT NULL DEFAULT 'default',
    camera_id TEXT    NOT NULL,          -- '*' = all cameras for this tenant
    zone      TEXT    NOT NULL,          -- scope: detect | model | classes | retention | zone:<name>
    field     TEXT    NOT NULL,
    value     TEXT    NOT NULL,
    version   INTEGER NOT NULL,
    updated   REAL,
    PRIMARY KEY (tenant_id, camera_id, zone, field)
);

-- ===================== IDENTITY (PS-3) ==================================
-- The edge holds a LOCAL sighting row so that a consent purge can find the
-- segments that contain a subject's pixels. The cloud holds the graph; the
-- edge holds the evidence. Deleting only in the cloud destroys your ability
-- to prove compliance while preserving the thing you were meant to delete.
CREATE TABLE IF NOT EXISTS sightings (
    sighting_id TEXT PRIMARY KEY,       -- same value as the outbox idem_key
    camera_id   TEXT NOT NULL,
    site_id     TEXT,
    track_id    INTEGER,
    subject_id  TEXT,                   -- NULL until the cloud resolves it
    ts          REAL NOT NULL,          -- first_ts; retention.py matches on this
    last_ts     REAL,
    dwell_s     REAL,
    coherence   REAL,
    n_samples   INTEGER DEFAULT 0,
    model_ver   TEXT,
    created     REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_sight_subject ON sightings(subject_id);
CREATE INDEX IF NOT EXISTS idx_sight_cam_ts  ON sightings(camera_id, ts);

-- Local mirror of the cloud gallery, kept fresh by the gallery_delta
-- directive. Lets the edge answer "have I seen this person" during a cloud
-- partition instead of going blind.
CREATE TABLE IF NOT EXISTS identities (
    subject_id  TEXT PRIMARY KEY,
    centroid    BLOB,                   -- float32 vector, .tobytes()
    dim         INTEGER DEFAULT 512,
    n_sightings INTEGER DEFAULT 0,
    last_seen   REAL,
    version     INTEGER DEFAULT 0,
    updated     REAL
);

CREATE TABLE IF NOT EXISTS embeddings_cache (
    sighting_id TEXT PRIMARY KEY,
    subject_id  TEXT,
    vec         BLOB NOT NULL,
    dim         INTEGER DEFAULT 512,
    created     REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_embcache_subject ON embeddings_cache(subject_id);

-- Hard never-match constraints. PS-3 Q3.3b: the immediate fix for "you matched
-- John as David" is surgical, costs exactly the two identities involved, and
-- lands in one control poll. Raising the global threshold instead degrades
-- recall for every other customer to fix one pair.
CREATE TABLE IF NOT EXISTS exclusions (
    a_id    TEXT NOT NULL,
    b_id    TEXT NOT NULL,
    reason  TEXT,
    created REAL,
    PRIMARY KEY (a_id, b_id)
);

-- ===================== GOVERNANCE =======================================
-- The lawful basis lives on the row, so every query can be justified after
-- the fact. 'consent' is revocable at will, 'legitimate_interest' requires a
-- balancing test and supports objection, 'revoked' blocks all processing.
CREATE TABLE IF NOT EXISTS consent (
    subject_id   TEXT PRIMARY KEY,
    basis        TEXT NOT NULL DEFAULT 'legitimate_interest',
    granted_ts   REAL,
    revoked_ts   REAL,
    retain_until REAL
);

-- Append only, and deliberately EXEMPT from the TTL sweep: you must be able to
-- prove a deletion happened after the data itself is gone.
CREATE TABLE IF NOT EXISTS audit_log (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    ts         REAL NOT NULL,
    site_id    TEXT,
    actor      TEXT DEFAULT 'edge',
    action     TEXT NOT NULL,
    subject_id TEXT,
    detail     TEXT
);
CREATE INDEX IF NOT EXISTS idx_audit_ts      ON audit_log(ts DESC);
CREATE INDEX IF NOT EXISTS idx_audit_subject ON audit_log(subject_id, ts DESC);
CREATE INDEX IF NOT EXISTS idx_audit_action  ON audit_log(action, ts DESC);

-- ===================== ACTIVITY (PS-2 carry-over) =======================
-- The layer where a pixel becomes something a human can act on. Round 1
-- reported "N objects", which is a detector output, not an event.
CREATE TABLE IF NOT EXISTS activity (
    id        INTEGER PRIMARY KEY AUTOINCREMENT,
    camera_id TEXT NOT NULL,
    site_id   TEXT,
    zone      TEXT,
    activity  TEXT NOT NULL,            -- loitering | crowd_formation | unauthorized_entry
    track_id  INTEGER,
    ts        REAL NOT NULL,
    detail    TEXT
);
CREATE INDEX IF NOT EXISTS idx_activity_ts  ON activity(ts DESC);
CREATE INDEX IF NOT EXISTS idx_activity_cam ON activity(camera_id, ts DESC);

-- Cloud schema. REPLACES cloud/schema.sql. Additive and idempotent.
-- Requires: CREATE EXTENSION IF NOT EXISTS pgcrypto;   (for gen_random_uuid)
-- Optional at >100K identities: CREATE EXTENSION IF NOT EXISTS vector;

CREATE EXTENSION IF NOT EXISTS pgcrypto;

-- ===================== EXISTING =========================================
CREATE TABLE IF NOT EXISTS events (
    id          BIGSERIAL PRIMARY KEY,
    idem_key    VARCHAR(255) UNIQUE,     -- the property everything leans on
    camera_id   VARCHAR(50),
    site_id     VARCHAR(50),
    event_type  VARCHAR(50),
    payload     JSONB,
    created_at  TIMESTAMPTZ DEFAULT now()
);
-- These indexes did not exist. Without them the dashboard query
-- (ORDER BY created_at DESC LIMIT 200) does a full scan of the table, which
-- is fine at 3771 rows and fatal at 10M.
CREATE INDEX IF NOT EXISTS idx_events_created ON events(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_events_cam_ts  ON events(camera_id, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_events_type    ON events(event_type, created_at DESC);

-- ===================== IDENTITY (PS-3) ==================================
CREATE TABLE IF NOT EXISTS sightings (
    sighting_id   VARCHAR(64) PRIMARY KEY,   -- = the edge idem_key, so a
                                             -- partition replay is a no-op
    camera_id     VARCHAR(50) NOT NULL,
    site_id       VARCHAR(50),
    track_id      INTEGER,
    subject_id    VARCHAR(64),               -- NULL until resolved
    first_ts      DOUBLE PRECISION NOT NULL,
    last_ts       DOUBLE PRECISION NOT NULL,
    dwell_s       REAL,
    coherence     REAL,
    embedding     REAL[],                    -- swap to vector(512) with pgvector
    model_ver     VARCHAR(64),
    retain_until  DOUBLE PRECISION NOT NULL, -- retention is a column, not a job
    consent_basis VARCHAR(32) DEFAULT 'legitimate_interest',
    created_at    TIMESTAMPTZ DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_sight_subject ON sightings(subject_id, last_ts DESC);
CREATE INDEX IF NOT EXISTS idx_sight_retain  ON sightings(retain_until);
CREATE INDEX IF NOT EXISTS idx_sight_site    ON sightings(site_id, last_ts DESC);

CREATE TABLE IF NOT EXISTS identities (
    subject_id      VARCHAR(64) PRIMARY KEY,
    centroid        REAL[],
    n_sightings     INTEGER DEFAULT 0,
    first_seen      DOUBLE PRECISION,
    last_seen       DOUBLE PRECISION,
    needs_recompute BOOLEAN DEFAULT false,   -- set when a source sighting is
                                             -- deleted: a centroid derived
                                             -- from deleted data still
                                             -- encodes the subject
    embedding_ver   VARCHAR(64),
    updated_at      TIMESTAMPTZ DEFAULT now()
);

CREATE TABLE IF NOT EXISTS exclusions (
    a_id VARCHAR(64) NOT NULL,
    b_id VARCHAR(64) NOT NULL,
    reason TEXT, created_at TIMESTAMPTZ DEFAULT now(),
    PRIMARY KEY (a_id, b_id)
);

CREATE TABLE IF NOT EXISTS consent (
    subject_id   VARCHAR(64) PRIMARY KEY,
    basis        VARCHAR(32) NOT NULL,
    granted_at   TIMESTAMPTZ,
    revoked_at   TIMESTAMPTZ,
    retain_until DOUBLE PRECISION
);

-- ===================== CONTROL PLANE ====================================
CREATE TABLE IF NOT EXISTS directives (
    directive_id VARCHAR(64) PRIMARY KEY,
    site_id      VARCHAR(50) NOT NULL,     -- '*' = fleet wide
    scope        VARCHAR(32) DEFAULT 'site',
    kind         VARCHAR(32) NOT NULL,
    version      BIGINT NOT NULL,          -- monotonic per site
    payload      JSONB NOT NULL,
    active       BOOLEAN DEFAULT true,
    created_at   DOUBLE PRECISION,
    UNIQUE (site_id, version)
);
CREATE INDEX IF NOT EXISTS idx_dir_site ON directives(site_id, version);

CREATE TABLE IF NOT EXISTS directive_acks (
    directive_id VARCHAR(64) NOT NULL,
    site_id      VARCHAR(50) NOT NULL,
    acked_at     DOUBLE PRECISION,
    PRIMARY KEY (directive_id, site_id)
);

-- ===================== FEEDBACK LOOP ====================================
CREATE TABLE IF NOT EXISTS adjudications (
    id          BIGSERIAL PRIMARY KEY,
    kind        VARCHAR(32) NOT NULL,
    cohort      VARCHAR(64) NOT NULL,      -- site, tenant, or gallery
    payload     JSONB NOT NULL,
    priority    INTEGER DEFAULT 5,
    model_ver   VARCHAR(64),
    status      VARCHAR(16) DEFAULT 'pending',
    verdict     VARCHAR(32),
    reviewer    VARCHAR(64),
    note        TEXT,
    created_at  DOUBLE PRECISION,
    resolved_at DOUBLE PRECISION
);
CREATE INDEX IF NOT EXISTS idx_adj_queue ON adjudications(status, cohort, priority, created_at);

CREATE TABLE IF NOT EXISTS labels (
    id BIGSERIAL PRIMARY KEY,
    source VARCHAR(32), cohort VARCHAR(64), model_ver VARCHAR(64),
    verdict VARCHAR(32), payload JSONB, created_at DOUBLE PRECISION
);

-- Cohort scoped metrics. A single global accuracy number is exactly what let
-- the Q4.2 deploy ship: it averaged Customer A's regression away against
-- Customer C's improvement.
CREATE TABLE IF NOT EXISTS cohort_metrics (
    cohort VARCHAR(64) NOT NULL, model_ver VARCHAR(64) NOT NULL,
    day DATE NOT NULL, tp INTEGER DEFAULT 0, fp INTEGER DEFAULT 0,
    fn INTEGER DEFAULT 0,
    PRIMARY KEY (cohort, model_ver, day)
);

CREATE TABLE IF NOT EXISTS cohort_latency (
    cohort VARCHAR(64) NOT NULL, model_ver VARCHAR(64) NOT NULL,
    day DATE NOT NULL, p50_latency_ms REAL, p95_latency_ms REAL,
    PRIMARY KEY (cohort, model_ver, day)
);

-- ===================== AUDIT ============================================
-- Append only. Every READ of an identity is logged, not just every write:
-- under PS-3 Q3.2b the query is the sensitive operation, because looking
-- someone up is itself the surveillance act.
CREATE TABLE IF NOT EXISTS audit_log (
    id BIGSERIAL PRIMARY KEY,
    ts DOUBLE PRECISION NOT NULL,
    actor VARCHAR(64), action VARCHAR(64) NOT NULL,
    subject_id VARCHAR(64), site_id VARCHAR(50),
    detail JSONB
);
CREATE INDEX IF NOT EXISTS idx_audit_subject ON audit_log(subject_id, ts DESC);
CREATE INDEX IF NOT EXISTS idx_audit_actor   ON audit_log(actor, ts DESC);

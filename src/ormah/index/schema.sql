-- Ormah - SQLite Index Schema

CREATE TABLE IF NOT EXISTS nodes (
    id TEXT PRIMARY KEY,
    type TEXT NOT NULL,
    tier TEXT NOT NULL DEFAULT 'working',
    source TEXT NOT NULL,
    space TEXT,
    space_locked INTEGER NOT NULL DEFAULT 0,
    title TEXT,
    content TEXT,
    created TEXT NOT NULL,
    updated TEXT NOT NULL,
    last_accessed TEXT NOT NULL,
    access_count INTEGER DEFAULT 0,
    confidence REAL DEFAULT 1.0,
    importance REAL DEFAULT 0.5,
    valid_until TEXT,
    stability REAL DEFAULT 1.0,
    last_review TEXT,
    file_path TEXT NOT NULL,
    file_hash TEXT NOT NULL,
    seq INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS edges (
    source_id TEXT NOT NULL REFERENCES nodes(id) ON DELETE CASCADE,
    target_id TEXT NOT NULL REFERENCES nodes(id) ON DELETE CASCADE,
    edge_type TEXT NOT NULL,
    weight REAL DEFAULT 0.5,
    created TEXT NOT NULL,
    reason TEXT,
    PRIMARY KEY (source_id, target_id, edge_type)
);

CREATE TABLE IF NOT EXISTS node_tags (
    node_id TEXT NOT NULL REFERENCES nodes(id) ON DELETE CASCADE,
    tag TEXT NOT NULL,
    PRIMARY KEY (node_id, tag)
);

CREATE INDEX IF NOT EXISTS idx_nodes_type ON nodes(type);
CREATE INDEX IF NOT EXISTS idx_nodes_tier ON nodes(tier);
CREATE INDEX IF NOT EXISTS idx_nodes_space ON nodes(space);
-- idx_nodes_seq is created in Database._migrate (after the seq column is guaranteed),
-- never here: on a legacy DB this script runs before the migration adds the column,
-- so indexing nodes(seq) at executescript time would fail with "no such column: seq".
CREATE INDEX IF NOT EXISTS idx_edges_source ON edges(source_id);
CREATE INDEX IF NOT EXISTS idx_edges_target ON edges(target_id);
CREATE INDEX IF NOT EXISTS idx_node_tags_tag ON node_tags(tag);

CREATE VIRTUAL TABLE IF NOT EXISTS nodes_fts USING fts5(
    id UNINDEXED, title, content, tags,
    tokenize='porter unicode61'
);

CREATE TABLE IF NOT EXISTS proposals (
    id TEXT PRIMARY KEY,
    type TEXT NOT NULL,
    status TEXT DEFAULT 'pending',
    source_nodes TEXT NOT NULL,
    proposed_action TEXT NOT NULL,
    reason TEXT,
    created TEXT NOT NULL,
    resolved TEXT
);

CREATE TABLE IF NOT EXISTS meta (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS merge_history (
    id TEXT PRIMARY KEY,
    proposal_id TEXT,              -- NULL for auto-merges
    kept_node_id TEXT NOT NULL,
    removed_node_id TEXT NOT NULL,
    removed_node_snapshot TEXT NOT NULL,  -- full MemoryNode JSON
    original_edges TEXT NOT NULL,         -- JSON array of edge dicts
    merged_at TEXT NOT NULL,
    undone_at TEXT                        -- NULL until rollback
);

CREATE TABLE IF NOT EXISTS auto_link_checked (
    node_a TEXT NOT NULL,
    node_b TEXT NOT NULL,
    result TEXT NOT NULL,           -- 'supports', 'contradicts', 'related_to', 'none', 'error'
    checked_at TEXT NOT NULL,
    PRIMARY KEY (node_a, node_b)
);

CREATE TABLE IF NOT EXISTS audit_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    operation TEXT NOT NULL,        -- 'delete', 'update', 'mark_outdated'
    node_id TEXT NOT NULL,
    node_snapshot TEXT,             -- full node JSON before the operation
    detail TEXT,                    -- operation-specific context
    performed_at TEXT NOT NULL
);

-- Prompt-level payload shared by every candidate produced by one retrieval.
-- Candidate rows keep their own stable whisper_log.id because feedback and
-- signals refer to that exact surfaced/rejected event.
CREATE TABLE IF NOT EXISTS retrieval_events (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    surface       TEXT NOT NULL,
    session_id    TEXT NOT NULL,
    space         TEXT,
    prompt_hash   TEXT NOT NULL,
    prompt_text   TEXT,
    prompt_vec    BLOB NOT NULL,
    logged_at     TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_retrieval_events_session ON retrieval_events(session_id);
CREATE INDEX IF NOT EXISTS idx_retrieval_events_logged  ON retrieval_events(logged_at);

CREATE TABLE IF NOT EXISTS whisper_log (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id          TEXT NOT NULL,
    space               TEXT,
    prompt_hash         TEXT NOT NULL,
    prompt_text         TEXT,
    prompt_vec          BLOB NOT NULL,
    node_id             TEXT NOT NULL,
    score               REAL NOT NULL,
    retrieval_score     REAL,
    raw_cosine          REAL,
    cross_encoder_score REAL,
    ce_absolute         REAL,
    gate_score          REAL,
    source              TEXT,
    retrieval_rank      INTEGER,
    final_rank          INTEGER,
    decision_stage      TEXT NOT NULL DEFAULT 'legacy',
    was_injected        INTEGER NOT NULL,
    logged_at           TEXT NOT NULL,
    retrieval_event_id  INTEGER REFERENCES retrieval_events(id) ON DELETE RESTRICT
);
CREATE INDEX IF NOT EXISTS idx_whisper_log_session ON whisper_log(session_id);
CREATE INDEX IF NOT EXISTS idx_whisper_log_node    ON whisper_log(node_id);
CREATE INDEX IF NOT EXISTS idx_whisper_log_logged  ON whisper_log(logged_at);
CREATE INDEX IF NOT EXISTS idx_whisper_log_retention
    ON whisper_log(was_injected, logged_at);
CREATE INDEX IF NOT EXISTS idx_whisper_log_session_injected
    ON whisper_log(session_id, was_injected);
CREATE INDEX IF NOT EXISTS idx_whisper_log_node_injected_logged
    ON whisper_log(node_id, was_injected, logged_at);

CREATE TABLE IF NOT EXISTS affinity (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    prompt_vec   BLOB NOT NULL,
    prompt_text  TEXT,
    node_id      TEXT NOT NULL,
    signal       INTEGER NOT NULL,
    source       TEXT NOT NULL DEFAULT 'explicit',
    confirmed_at TEXT NOT NULL,
    space        TEXT,
    session_id   TEXT NOT NULL,
    whisper_log_id INTEGER REFERENCES whisper_log(id) ON DELETE SET NULL
);
CREATE INDEX IF NOT EXISTS idx_affinity_node ON affinity(node_id);
-- Indexes on affinity.whisper_log_id are created by the migration code
-- (_ensure_affinity_indexes), NOT here. On a pre-feedback DB the affinity table
-- already exists without whisper_log_id, so "CREATE TABLE IF NOT EXISTS affinity"
-- above is a no-op and the column is missing. executescript() runs this file
-- before _migrate() adds the column, so creating these indexes here would crash
-- with "no such column: whisper_log_id". _migrate_affinity_schema() rebuilds the
-- legacy table and then creates these indexes, which is the correct ordering.

CREATE TABLE IF NOT EXISTS signals (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    whisper_log_id INTEGER REFERENCES whisper_log(id) ON DELETE SET NULL,
    node_id        TEXT NOT NULL,
    signal_type    TEXT NOT NULL,
    polarity       INTEGER NOT NULL,
    strength       REAL NOT NULL DEFAULT 1.0,
    source         TEXT NOT NULL,
    session_id     TEXT,
    agent_id       TEXT,
    surface        TEXT,
    space          TEXT,
    prompt_hash    TEXT,
    evidence       TEXT,
    created        TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_signals_node ON signals(node_id);
CREATE INDEX IF NOT EXISTS idx_signals_session ON signals(session_id);
CREATE INDEX IF NOT EXISTS idx_signals_created ON signals(created);
CREATE INDEX IF NOT EXISTS idx_signals_whisper_log ON signals(whisper_log_id);
CREATE UNIQUE INDEX IF NOT EXISTS idx_signals_whisper_type_source_unique
    ON signals(whisper_log_id, signal_type, source)
    WHERE whisper_log_id IS NOT NULL;

-- One row per whisper call recording the outcome — including silence.
-- whisper_log (above) only records candidates; prompts where whisper stayed
-- silent left no trace, making silence rate uncomputable. This table is the
-- per-prompt denominator. Stores prompt_hash only (no text, no vectors).
CREATE TABLE IF NOT EXISTS whisper_decisions (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id      TEXT,
    space           TEXT,
    prompt_hash     TEXT NOT NULL,
    intent          TEXT,               -- comma-joined intent categories, NULL if unclassified
    outcome         TEXT NOT NULL,      -- 'injected' | 'silent_short' | 'silent_conversational'
                                        -- | 'silent_topic_shift' | 'silent_no_candidates'
                                        -- | 'silent_gate' | 'silent_blackout' | 'silent_error'
    candidate_count INTEGER DEFAULT 0,  -- results returned by search before filtering
    injected_count  INTEGER DEFAULT 0,  -- memories actually injected
    max_gate_score  REAL,               -- best absolute gate score among candidates
    logged_at       TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_whisper_decisions_session ON whisper_decisions(session_id);
CREATE INDEX IF NOT EXISTS idx_whisper_decisions_logged  ON whisper_decisions(logged_at);

CREATE TABLE IF NOT EXISTS review_log (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    node_id     TEXT NOT NULL,
    session_id  TEXT NOT NULL,
    surfaced_at TEXT NOT NULL,
    answered    INTEGER DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_review_log_node ON review_log(node_id);

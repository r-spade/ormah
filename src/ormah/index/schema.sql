-- Ormah - SQLite Index Schema

CREATE TABLE IF NOT EXISTS nodes (
    id TEXT PRIMARY KEY,
    type TEXT NOT NULL,
    tier TEXT NOT NULL DEFAULT 'working',
    source TEXT NOT NULL,
    space TEXT,
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
    file_hash TEXT NOT NULL
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

CREATE TABLE IF NOT EXISTS whisper_log (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id   TEXT NOT NULL,
    space        TEXT,
    prompt_hash  TEXT NOT NULL,
    prompt_text  TEXT,
    prompt_vec   BLOB NOT NULL,
    node_id      TEXT NOT NULL,
    score        REAL NOT NULL,
    was_injected INTEGER NOT NULL,
    logged_at    TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_whisper_log_session ON whisper_log(session_id);
CREATE INDEX IF NOT EXISTS idx_whisper_log_node    ON whisper_log(node_id);
CREATE INDEX IF NOT EXISTS idx_whisper_log_logged  ON whisper_log(logged_at);

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
    UNIQUE (node_id, session_id)
);
CREATE INDEX IF NOT EXISTS idx_affinity_node ON affinity(node_id);

CREATE TABLE IF NOT EXISTS review_log (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    node_id     TEXT NOT NULL,
    session_id  TEXT NOT NULL,
    surfaced_at TEXT NOT NULL,
    answered    INTEGER DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_review_log_node ON review_log(node_id);

CREATE TABLE IF NOT EXISTS gate_state (
    key         TEXT PRIMARY KEY,      -- always 'injection_gate'
    value       REAL NOT NULL,
    updated_at  TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS decay_alert_log (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    node_id        TEXT NOT NULL,
    alerted_at     TEXT NOT NULL,
    retrievability REAL NOT NULL,
    tier           TEXT NOT NULL,
    acknowledged   INTEGER DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_decay_alert_node ON decay_alert_log(node_id);

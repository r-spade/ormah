"""Tests for whisper_log, affinity, and review_log schema additions."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from ormah.index.db import Database


def _table_columns(db: Database, table: str) -> list[str]:
    rows = db.conn.execute(f"PRAGMA table_info({table})").fetchall()
    return [row[1] for row in rows]


def _table_exists(db: Database, table: str) -> bool:
    row = db.conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name=?", (table,)
    ).fetchone()
    return row is not None


def _index_exists(db: Database, index_name: str) -> bool:
    row = db.conn.execute(
        "SELECT name FROM sqlite_master WHERE type='index' AND name=?", (index_name,)
    ).fetchone()
    return row is not None


# ---------------------------------------------------------------------------
# Tests for fresh schema (via init_schema)
# ---------------------------------------------------------------------------


def test_whisper_log_table_exists(db):
    assert _table_exists(db, "whisper_log")


def test_whisper_log_columns(db):
    cols = _table_columns(db, "whisper_log")
    for expected in [
        "id",
        "session_id",
        "space",
        "prompt_hash",
        "prompt_text",
        "prompt_vec",
        "node_id",
        "score",
        "retrieval_score",
        "raw_cosine",
        "cross_encoder_score",
        "ce_absolute",
        "gate_score",
        "source",
        "retrieval_rank",
        "final_rank",
        "decision_stage",
        "was_injected",
        "logged_at",
    ]:
        assert expected in cols, f"Missing column '{expected}' in whisper_log"


def test_whisper_log_indexes(db):
    assert _index_exists(db, "idx_whisper_log_session")
    assert _index_exists(db, "idx_whisper_log_node")
    assert _index_exists(db, "idx_whisper_log_logged")


def test_affinity_table_exists(db):
    assert _table_exists(db, "affinity")


def test_affinity_columns(db):
    cols = _table_columns(db, "affinity")
    for expected in [
        "id",
        "prompt_vec",
        "prompt_text",
        "node_id",
        "signal",
        "source",
        "confirmed_at",
        "space",
        "session_id",
        "whisper_log_id",
    ]:
        assert expected in cols, f"Missing column '{expected}' in affinity"


def test_affinity_unique_constraint_is_per_whisper_log(db):
    """Feedback is capped per whisper event, not per whole session."""
    import datetime

    now = datetime.datetime.now(datetime.UTC).isoformat()
    cursor = db.conn.execute(
        "INSERT INTO whisper_log "
        "(session_id, space, prompt_hash, prompt_text, prompt_vec, node_id, score, "
        "was_injected, logged_at) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
        ("sess-1", "space", "hash-1", "prompt one", b"\x00" * 4, "node-1", 0.8, 1, now),
    )
    first_whisper_log_id = cursor.lastrowid
    cursor = db.conn.execute(
        "INSERT INTO whisper_log "
        "(session_id, space, prompt_hash, prompt_text, prompt_vec, node_id, score, "
        "was_injected, logged_at) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
        ("sess-1", "space", "hash-2", "prompt two", b"\x00" * 4, "node-1", 0.8, 1, now),
    )
    second_whisper_log_id = cursor.lastrowid

    db.conn.execute(
        "INSERT INTO affinity "
        "(prompt_vec, node_id, signal, source, confirmed_at, session_id, whisper_log_id) "
        "VALUES (?, ?, ?, ?, ?, ?, ?)",
        (b"\x00" * 4, "node-1", 1, "explicit", now, "sess-1", first_whisper_log_id),
    )
    db.conn.execute(
        "INSERT INTO affinity "
        "(prompt_vec, node_id, signal, source, confirmed_at, session_id, whisper_log_id) "
        "VALUES (?, ?, ?, ?, ?, ?, ?)",
        (b"\x00" * 4, "node-1", 1, "explicit", now, "sess-1", second_whisper_log_id),
    )

    with pytest.raises(sqlite3.IntegrityError):
        db.conn.execute(
            "INSERT INTO affinity "
            "(prompt_vec, node_id, signal, source, confirmed_at, session_id, whisper_log_id) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (b"\x00" * 4, "node-1", 1, "explicit", now, "sess-1", first_whisper_log_id),
        )


def test_affinity_index(db):
    assert _index_exists(db, "idx_affinity_node")
    assert _index_exists(db, "idx_affinity_whisper_log")
    assert _index_exists(db, "idx_affinity_node_whisper_log_unique")


def test_signals_table_exists(db):
    assert _table_exists(db, "signals")


def test_signals_columns(db):
    cols = _table_columns(db, "signals")
    for expected in [
        "id",
        "whisper_log_id",
        "node_id",
        "signal_type",
        "polarity",
        "strength",
        "source",
        "session_id",
        "agent_id",
        "surface",
        "space",
        "prompt_hash",
        "evidence",
        "created",
    ]:
        assert expected in cols, f"Missing column '{expected}' in signals"


def test_signals_indexes(db):
    assert _index_exists(db, "idx_signals_node")
    assert _index_exists(db, "idx_signals_session")
    assert _index_exists(db, "idx_signals_created")
    assert _index_exists(db, "idx_signals_whisper_log")
    assert _index_exists(db, "idx_signals_whisper_type_source_unique")


def test_review_log_table_exists(db):
    assert _table_exists(db, "review_log")


def test_review_log_columns(db):
    cols = _table_columns(db, "review_log")
    for expected in ["id", "node_id", "session_id", "surfaced_at", "answered"]:
        assert expected in cols, f"Missing column '{expected}' in review_log"


def test_review_log_index(db):
    assert _index_exists(db, "idx_review_log_node")


# ---------------------------------------------------------------------------
# Migration tests: tables created on existing DB that lacked them
# ---------------------------------------------------------------------------


def _make_db_without_new_tables(tmp_path: Path) -> Database:
    """Create a DB, init schema, then drop the three new tables to simulate
    an older database that predates their introduction."""
    database = Database(tmp_path / "index.db")
    database.init_schema()
    # Drop the new tables to simulate an old DB
    database.conn.executescript(
        "DROP TABLE IF EXISTS whisper_log;"
        "DROP TABLE IF EXISTS affinity;"
        "DROP TABLE IF EXISTS signals;"
        "DROP TABLE IF EXISTS review_log;"
    )
    return database


def test_migrate_creates_whisper_log(tmp_path):
    db = _make_db_without_new_tables(tmp_path)
    assert not _table_exists(db, "whisper_log")
    db._migrate()
    assert _table_exists(db, "whisper_log")
    db.close()


def test_migrate_adds_whisper_candidate_diagnostics(tmp_path):
    path = tmp_path / "index.db"
    conn = sqlite3.connect(path)
    conn.execute(
        """
        CREATE TABLE whisper_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL,
            space TEXT,
            prompt_hash TEXT NOT NULL,
            prompt_text TEXT,
            prompt_vec BLOB NOT NULL,
            node_id TEXT NOT NULL,
            score REAL NOT NULL,
            was_injected INTEGER NOT NULL,
            logged_at TEXT NOT NULL
        )
        """
    )
    conn.execute(
        "INSERT INTO whisper_log "
        "(session_id, prompt_hash, prompt_vec, node_id, score, was_injected, logged_at) "
        "VALUES ('s1', 'h1', X'00', 'n1', 0.5, 0, '2026-01-01T00:00:00Z')"
    )
    conn.commit()
    conn.close()

    db = Database(path)
    db.init_schema()

    cols = _table_columns(db, "whisper_log")
    assert "decision_stage" in cols
    assert "ce_absolute" in cols
    row = db.conn.execute(
        "SELECT node_id, decision_stage FROM whisper_log WHERE node_id = 'n1'"
    ).fetchone()
    assert tuple(row) == ("n1", "legacy")
    db.close()


def test_migrate_creates_affinity(tmp_path):
    db = _make_db_without_new_tables(tmp_path)
    assert not _table_exists(db, "affinity")
    db._migrate()
    assert _table_exists(db, "affinity")
    db.close()


def test_migrate_creates_review_log(tmp_path):
    db = _make_db_without_new_tables(tmp_path)
    assert not _table_exists(db, "review_log")
    db._migrate()
    assert _table_exists(db, "review_log")
    db.close()


def test_migrate_creates_signals(tmp_path):
    db = _make_db_without_new_tables(tmp_path)
    assert not _table_exists(db, "signals")
    db._migrate()
    assert _table_exists(db, "signals")
    db.close()


def test_migrate_is_idempotent(tmp_path):
    """Calling _migrate() on an already-migrated DB must not raise."""
    db = _make_db_without_new_tables(tmp_path)
    db._migrate()
    db._migrate()  # second call should be a no-op
    assert _table_exists(db, "whisper_log")
    assert _table_exists(db, "affinity")
    assert _table_exists(db, "signals")
    assert _table_exists(db, "review_log")


# ---------------------------------------------------------------------------
# Regression: a *pre-feedback* DB whose affinity table still has the legacy
# schema (no whisper_log_id, UNIQUE(node_id, session_id)) must survive the full
# init_schema() path. The earlier migration tests drop the affinity table
# entirely and call _migrate() directly, so they never exercised executescript()
# running schema.sql against a pre-existing legacy affinity table -- which is
# exactly what crashed in 0.12.0/0.12.1 with "no such column: whisper_log_id".
# ---------------------------------------------------------------------------

_LEGACY_AFFINITY_DDL = """
CREATE TABLE affinity (
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
CREATE INDEX idx_affinity_node ON affinity(node_id);
"""


def _make_legacy_affinity_db(tmp_path: Path) -> Path:
    """Build a DB with the pre-feedback affinity table and a seed row, without
    any of the new feedback tables/columns."""
    path = tmp_path / "index.db"
    conn = sqlite3.connect(path)
    conn.executescript(_LEGACY_AFFINITY_DDL)
    conn.execute(
        "INSERT INTO affinity "
        "(prompt_vec, prompt_text, node_id, signal, source, confirmed_at, "
        " space, session_id) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        (b"\x00", "hi", "node-1", 1, "explicit", "2026-01-01T00:00:00Z",
         "proj", "sess-1"),
    )
    conn.commit()
    conn.close()
    return path


def test_init_schema_migrates_legacy_affinity_table(tmp_path):
    """Full init_schema() on a legacy affinity table must not raise and must
    add whisper_log_id, create the signals table, and preserve existing rows."""
    path = _make_legacy_affinity_db(tmp_path)

    db = Database(path)
    db.init_schema()  # regression: this used to raise OperationalError

    assert "whisper_log_id" in _table_columns(db, "affinity")
    assert _table_exists(db, "signals")
    assert _index_exists(db, "idx_affinity_whisper_log")
    assert _index_exists(db, "idx_affinity_node_whisper_log_unique")
    # existing data survives the table rebuild
    rows = db.conn.execute("SELECT node_id, session_id FROM affinity").fetchall()
    assert [tuple(r) for r in rows] == [("node-1", "sess-1")]

    # idempotent: a second init_schema() on the now-migrated DB is a no-op
    db.init_schema()
    assert "whisper_log_id" in _table_columns(db, "affinity")
    db.close()
    db.close()

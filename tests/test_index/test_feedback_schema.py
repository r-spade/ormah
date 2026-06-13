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
    db.close()

"""Regression tests for the whisper eval miner.

Build a temp SQLite DB with the real whisper_log / whisper_decisions / nodes
schema and exercise the mining invariants: per-session grouping, exclusion of
recall-contamination rows, deterministic candidate truncation, and provisional
labelling.
"""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from eval.whisper.miner import _MAX_MEMORIES_PER_CASE, MinerError, import_labels, mine


def _make_db(path: Path) -> None:
    conn = sqlite3.connect(path)
    conn.executescript(
        """
        CREATE TABLE whisper_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL, space TEXT,
            prompt_hash TEXT NOT NULL, prompt_text TEXT, prompt_vec BLOB,
            node_id TEXT NOT NULL, score REAL NOT NULL,
            was_injected INTEGER NOT NULL, logged_at TEXT NOT NULL
        );
        CREATE TABLE whisper_decisions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT, space TEXT, prompt_hash TEXT NOT NULL,
            intent TEXT, outcome TEXT NOT NULL,
            candidate_count INTEGER DEFAULT 0, injected_count INTEGER DEFAULT 0,
            max_gate_score REAL, logged_at TEXT NOT NULL
        );
        CREATE TABLE nodes (
            id TEXT PRIMARY KEY, type TEXT NOT NULL, tier TEXT NOT NULL DEFAULT 'working',
            source TEXT NOT NULL DEFAULT 'test', space TEXT, title TEXT, content TEXT,
            confidence REAL DEFAULT 1.0
        );
        CREATE TABLE node_tags (node_id TEXT NOT NULL, tag TEXT NOT NULL);
        """
    )
    conn.commit()
    conn.close()


def _log(conn, session_id, prompt_hash, prompt_text, node_id, score, injected, ts):
    conn.execute(
        "INSERT INTO whisper_log (session_id, space, prompt_hash, prompt_text, prompt_vec, "
        "node_id, score, was_injected, logged_at) VALUES (?,?,?,?,?,?,?,?,?)",
        (session_id, "ormah", prompt_hash, prompt_text, b"", node_id, score, injected, ts),
    )


def _decision(conn, session_id, prompt_hash, ts, outcome="injected"):
    conn.execute(
        "INSERT INTO whisper_decisions (session_id, space, prompt_hash, intent, outcome, "
        "candidate_count, injected_count, max_gate_score, logged_at) VALUES (?,?,?,?,?,?,?,?,?)",
        (session_id, "ormah", prompt_hash, None, outcome, 1, 1, 0.6, ts),
    )


def _node(conn, node_id, title):
    conn.execute(
        "INSERT INTO nodes (id, type, tier, source, space, title, content, confidence) "
        "VALUES (?, 'fact', 'working', 'test', 'ormah', ?, ?, 1.0)",
        (node_id, title, f"content for {title}"),
    )


def _run_mine(tmp_path: Path, db: Path, limit: int = 50):
    out = tmp_path / "mined.jsonl"
    mine(db, limit=limit, out_path=out)
    return [json.loads(line) for line in out.read_text().splitlines() if line.strip()]


def test_same_prompt_different_sessions_yields_two_cases(tmp_path):
    db = tmp_path / "live.db"
    _make_db(db)
    conn = sqlite3.connect(db)
    for node in ("n1", "n2"):
        _node(conn, node, node)
    # Same prompt_hash, two sessions — must not collapse into one case.
    for sess in ("sess-A", "sess-B"):
        _log(conn, sess, "hashX", "same prompt", "n1", 0.7, 1, "2026-07-01T10:00:00Z")
        _log(conn, sess, "hashX", "same prompt", "n2", 0.4, 0, "2026-07-01T10:00:00Z")
        _decision(conn, sess, "hashX", "2026-07-01T10:00:00Z")
    conn.commit()
    conn.close()

    cases = _run_mine(tmp_path, db)
    assert len(cases) == 2


def test_recall_contamination_excluded(tmp_path):
    db = tmp_path / "live.db"
    _make_db(db)
    conn = sqlite3.connect(db)
    for node in ("n1", "n2"):
        _node(conn, node, node)
    # Group WITH a matching whisper_decisions row → mined.
    _log(conn, "sess-1", "whisper", "a whisper prompt", "n1", 0.7, 1, "2026-07-01T10:00:00Z")
    _log(conn, "sess-1", "whisper", "a whisper prompt", "n2", 0.4, 0, "2026-07-01T10:00:00Z")
    _decision(conn, "sess-1", "whisper", "2026-07-01T10:00:00Z")
    # Group with NO whisper_decisions row (recall exposure / pre-table) → excluded.
    _log(conn, "sess-1", "recall", "a recall exposure", "n1", 0.9, 1, "2026-07-01T11:00:00Z")
    _log(conn, "sess-1", "recall", "a recall exposure", "n2", 0.8, 1, "2026-07-01T11:00:00Z")
    conn.commit()
    conn.close()

    cases = _run_mine(tmp_path, db)
    prompts = {c["prompts"][0]["text"] for c in cases}
    assert prompts == {"a whisper prompt"}


def test_normalized_retrieval_event_supplies_prompt_payload(tmp_path):
    db = tmp_path / "live.db"
    _make_db(db)
    conn = sqlite3.connect(db)
    conn.executescript(
        """
        CREATE TABLE retrieval_events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            surface TEXT NOT NULL,
            session_id TEXT NOT NULL,
            space TEXT,
            prompt_hash TEXT NOT NULL,
            prompt_text TEXT,
            prompt_vec BLOB NOT NULL,
            logged_at TEXT NOT NULL
        );
        ALTER TABLE whisper_log ADD COLUMN retrieval_event_id INTEGER;
        """
    )
    for node in ("n1", "n2"):
        _node(conn, node, node)
    cursor = conn.execute(
        """
        INSERT INTO retrieval_events
            (surface, session_id, space, prompt_hash, prompt_text, prompt_vec, logged_at)
        VALUES ('whisper', 'sess', 'ormah', 'hash', 'normalized prompt', X'01', ?)
        """,
        ("2026-07-01T10:00:00Z",),
    )
    event_id = cursor.lastrowid
    for node_id, score, injected in (("n1", 0.7, 1), ("n2", 0.2, 0)):
        conn.execute(
            """
            INSERT INTO whisper_log
                (session_id, space, prompt_hash, prompt_text, prompt_vec,
                 node_id, score, was_injected, logged_at, retrieval_event_id)
            VALUES ('sess', 'ormah', 'hash', NULL, X'', ?, ?, ?, ?, ?)
            """,
            (node_id, score, injected, "2026-07-01T10:00:00Z", event_id),
        )
    _decision(conn, "sess", "hash", "2026-07-01T10:00:00Z")
    conn.commit()
    conn.close()

    cases = _run_mine(tmp_path, db)

    assert len(cases) == 1
    assert cases[0]["prompts"][0]["text"] == "normalized prompt"


def test_deterministic_truncation_keeps_injected_node(tmp_path):
    db = tmp_path / "live.db"
    _make_db(db)
    conn = sqlite3.connect(db)
    # One injected node plus more than _MAX_MEMORIES_PER_CASE weak candidates.
    _node(conn, "injected", "the injected one")
    _log(conn, "s", "h", "wide pool prompt", "injected", 0.62, 1, "2026-07-01T10:00:00Z")
    for i in range(_MAX_MEMORIES_PER_CASE + 5):
        nid = f"weak{i:02d}"
        _node(conn, nid, nid)
        _log(conn, "s", "h", "wide pool prompt", nid, 0.10, 0, "2026-07-01T10:00:00Z")
    _decision(conn, "s", "h", "2026-07-01T10:00:00Z")
    conn.commit()
    conn.close()

    cases = _run_mine(tmp_path, db)
    assert len(cases) == 1
    mem_ids = {m["node_id"] for m in cases[0]["memories"]}
    assert "injected" in mem_ids
    assert len(mem_ids) <= _MAX_MEMORIES_PER_CASE


def test_missing_whisper_decisions_raises_clear_error(tmp_path):
    """A live DB predating the whisper_decisions table must fail cleanly, not
    crash with a raw OperationalError, and must not mine contaminated data."""
    db = tmp_path / "old.db"
    conn = sqlite3.connect(db)
    conn.executescript(
        "CREATE TABLE whisper_log (id INTEGER PRIMARY KEY, session_id TEXT, space TEXT, "
        "prompt_hash TEXT, prompt_text TEXT, prompt_vec BLOB, node_id TEXT, score REAL, "
        "was_injected INTEGER, logged_at TEXT);"
    )
    conn.commit()
    conn.close()

    with pytest.raises(MinerError, match="whisper_decisions"):
        mine(db, limit=5, out_path=tmp_path / "mined.jsonl")


def test_mined_cases_are_provisional_until_import(tmp_path):
    db = tmp_path / "live.db"
    _make_db(db)
    conn = sqlite3.connect(db)
    for node in ("n1", "n2"):
        _node(conn, node, node)
    _log(conn, "s", "h", "a prompt", "n1", 0.7, 1, "2026-07-01T10:00:00Z")
    _log(conn, "s", "h", "a prompt", "n2", 0.4, 0, "2026-07-01T10:00:00Z")
    _decision(conn, "s", "h", "2026-07-01T10:00:00Z")
    conn.commit()
    conn.close()

    out = tmp_path / "mined.jsonl"
    mine(db, limit=10, out_path=out)
    cases = [json.loads(line) for line in out.read_text().splitlines() if line.strip()]
    assert cases and all(c.get("provisional") is True for c in cases)

    confirmed = import_labels(out)
    assert confirmed == len(cases)
    cases_after = [json.loads(line) for line in out.read_text().splitlines() if line.strip()]
    assert all("provisional" not in c for c in cases_after)

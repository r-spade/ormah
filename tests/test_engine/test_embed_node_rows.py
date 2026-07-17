"""Tests for MemoryEngine._embed_node_rows (extracted embedding core, #32)."""
from __future__ import annotations

import numpy as np
import pytest

from ormah.models.node import CreateNodeRequest


def test_embed_node_rows_returns_embedded_ids(engine):
    nid, _ = engine.remember(CreateNodeRequest(title="Alpha", content="hello world"))
    with engine.db.transaction() as conn:
        conn.execute("DELETE FROM node_vectors WHERE id = ?", (nid,))
    rows = engine.db.conn.execute(
        "SELECT id, title, content FROM nodes WHERE id = ?", (nid,)
    ).fetchall()

    embedded_ids, failed_ids = engine._embed_node_rows(rows)

    assert embedded_ids == [nid]
    assert failed_ids == []
    assert engine.db.conn.execute(
        "SELECT count(*) FROM node_vectors_rowids WHERE id = ?", (nid,)
    ).fetchone()[0] == 1


def test_embed_node_rows_reports_failed_ids(engine, monkeypatch):
    nid, _ = engine.remember(CreateNodeRequest(title="Boom", content="payload"))
    with engine.db.transaction() as conn:
        conn.execute("DELETE FROM node_vectors WHERE id = ?", (nid,))
    rows = engine.db.conn.execute(
        "SELECT id, title, content FROM nodes WHERE id = ?", (nid,)
    ).fetchall()

    class _DeadEncoder:
        def encode(self, text):
            raise RuntimeError("encoder down")

    monkeypatch.setattr("ormah.embeddings.encoder.get_encoder", lambda settings: _DeadEncoder())

    embedded_ids, failed_ids = engine._embed_node_rows(rows)

    assert embedded_ids == []
    assert failed_ids == [nid]


def test_embed_node_rows_empty_list_is_noop(engine):
    embedded_ids, failed_ids = engine._embed_node_rows([])
    assert embedded_ids == []
    assert failed_ids == []


def test_embed_node_rows_persists_incrementally(engine, monkeypatch):
    """A hard interrupt mid-encode must leave already-encoded chunks persisted,
    not lose everything. Simulate the kill with a BaseException on encode #101."""
    dim = engine.settings.embedding_dim
    for i in range(150):
        engine.remember(CreateNodeRequest(title=f"n{i}", content=f"content {i}"))
    with engine.db.transaction() as conn:
        conn.execute("DELETE FROM node_vectors")

    calls = {"n": 0}

    class _KillAt101:
        def encode(self, text):
            calls["n"] += 1
            if calls["n"] == 101:
                raise KeyboardInterrupt("hard kill mid-encode")
            return np.ones(dim, dtype=np.float32)

    monkeypatch.setattr("ormah.embeddings.encoder.get_encoder", lambda s: _KillAt101())
    rows = engine.db.conn.execute("SELECT id, title, content FROM nodes").fetchall()

    with pytest.raises(KeyboardInterrupt):
        engine._embed_node_rows(rows)

    persisted = engine.db.conn.execute(
        "SELECT count(*) FROM node_vectors_rowids"
    ).fetchone()[0]
    assert persisted == 100  # first full chunk landed before the kill


def test_embed_node_rows_flushes_pending_on_cooperative_cancel(engine, monkeypatch):
    """stop_event set mid-run: everything encoded so far is persisted (the final
    flush runs on the cooperative-cancel path too)."""
    import threading

    dim = engine.settings.embedding_dim
    for i in range(120):
        engine.remember(CreateNodeRequest(title=f"c{i}", content=f"cancel {i}"))
    with engine.db.transaction() as conn:
        conn.execute("DELETE FROM node_vectors")

    stop = threading.Event()
    calls = {"n": 0}

    class _StopAt105:
        def encode(self, text):
            calls["n"] += 1
            if calls["n"] == 105:
                stop.set()  # cancellation arrives after this encode returns
            return np.ones(dim, dtype=np.float32)

    monkeypatch.setattr("ormah.embeddings.encoder.get_encoder", lambda s: _StopAt105())
    rows = engine.db.conn.execute("SELECT id, title, content FROM nodes").fetchall()

    embedded_ids, failed_ids = engine._embed_node_rows(rows, stop_event=stop)

    assert len(embedded_ids) == 105  # 100 flushed at the boundary + 5 pending flushed on exit
    assert failed_ids == []
    persisted = engine.db.conn.execute(
        "SELECT count(*) FROM node_vectors_rowids"
    ).fetchone()[0]
    assert persisted == 105


def test_persistence_failure_propagates_not_marked_failed(engine, monkeypatch):
    """An upsert_batch error is a JOB failure, not a per-node encode failure: it
    must propagate (tracked() records it) and the node must NOT enter failed_ids
    — a wrong failed_ids entry would get its vector deleted by the schema-mode
    failed-node cleanup. (council r3, codex high)

    Uses >100 nodes so the in-loop chunk-boundary flush actually fires during
    the loop (with <=100 nodes only the post-loop flush runs, which is
    structurally outside any try regardless of where the in-loop call sits —
    that would make this test pass even if _flush() were wrongly moved inside
    the encode try). Asserts upsert_batch was called exactly once: if the
    first flush's error were swallowed into failed_ids instead of propagating,
    the loop would continue and a second (post-loop) flush would call
    upsert_batch again, catching the exact regression this test guards
    against."""
    import sqlite3

    dim = engine.settings.embedding_dim
    for i in range(101):
        engine.remember(CreateNodeRequest(title=f"pf{i}", content=f"persist fail {i}"))
    with engine.db.transaction() as conn:
        conn.execute("DELETE FROM node_vectors")

    class _OkEncoder:
        def encode(self, text):
            return np.ones(dim, dtype=np.float32)

    monkeypatch.setattr("ormah.embeddings.encoder.get_encoder", lambda s: _OkEncoder())

    calls = {"n": 0}

    def _boom(self, items):
        calls["n"] += 1
        raise sqlite3.OperationalError("disk I/O error")

    monkeypatch.setattr(
        "ormah.embeddings.vector_store.VectorStore.upsert_batch", _boom
    )
    rows = engine.db.conn.execute("SELECT id, title, content FROM nodes").fetchall()

    with pytest.raises(sqlite3.OperationalError):
        engine._embed_node_rows(rows)

    assert calls["n"] == 1  # propagated immediately; no second (post-loop) flush attempt

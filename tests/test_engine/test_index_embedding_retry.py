"""Tests for _index_embedding bounded retry (#32)."""
from __future__ import annotations

from ormah.models.node import CreateNodeRequest


class _FlakyEncoder:
    """Fails `fail_times` then succeeds, returning a fixed-dim vector."""
    def __init__(self, fail_times, dim):
        self.fail_times = fail_times
        self.calls = 0
        self._dim = dim

    def encode(self, text):
        self.calls += 1
        if self.calls <= self.fail_times:
            raise RuntimeError("transient encoder failure")
        import numpy as np
        return np.ones(self._dim, dtype="float32")


def test_index_embedding_retries_then_succeeds(engine, monkeypatch):
    nid, _ = engine.remember(CreateNodeRequest(title="Retry", content="payload"))
    with engine.db.transaction() as conn:
        conn.execute("DELETE FROM node_vectors WHERE id = ?", (nid,))
    node = engine.file_store.load(nid)

    enc = _FlakyEncoder(fail_times=2, dim=engine.settings.embedding_dim)
    monkeypatch.setattr("ormah.embeddings.encoder.get_encoder", lambda settings: enc)
    monkeypatch.setattr("time.sleep", lambda s: None)  # no real backoff in tests

    engine._index_embedding(node)  # max_retries default 2 -> 3 attempts total

    assert enc.calls == 3
    assert engine.db.conn.execute(
        "SELECT count(*) FROM node_vectors_rowids WHERE id = ?", (nid,)
    ).fetchone()[0] == 1


def test_index_embedding_gives_up_without_raising(engine, monkeypatch):
    nid, _ = engine.remember(CreateNodeRequest(title="Down", content="payload"))
    with engine.db.transaction() as conn:
        conn.execute("DELETE FROM node_vectors WHERE id = ?", (nid,))
    node = engine.file_store.load(nid)

    class _DeadEncoder:
        def encode(self, text):
            raise RuntimeError("encoder permanently down")

    monkeypatch.setattr("ormah.embeddings.encoder.get_encoder", lambda settings: _DeadEncoder())
    monkeypatch.setattr("time.sleep", lambda s: None)

    engine._index_embedding(node)  # must NOT raise

    assert engine.db.conn.execute(
        "SELECT count(*) FROM node_vectors_rowids WHERE id = ?", (nid,)
    ).fetchone()[0] == 0

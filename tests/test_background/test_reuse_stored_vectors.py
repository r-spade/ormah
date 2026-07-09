"""Issue #88: pairwise jobs must reuse stored vectors, not re-encode probes."""
import re
import numpy as np
import pytest

from ormah.embeddings.vector_store import VectorStore, stored_or_encoded


def _vec_dim(db) -> int:
    sql = db.conn.execute(
        "SELECT sql FROM sqlite_master WHERE name='node_vectors'"
    ).fetchone()[0]
    return int(re.search(r"float\[(\d+)\]", sql, re.IGNORECASE).group(1))


class _ExplodingEncoder:
    def encode(self, text):
        raise AssertionError("encoder.encode must not run when a stored vector exists")


class _CountingEncoder:
    def __init__(self, dim):
        self.dim = dim
        self.calls = 0

    def encode(self, text):
        self.calls += 1
        return np.ones(self.dim, dtype=np.float32)


def test_stored_or_encoded_prefers_stored_vector(db):
    db.init_vec_table(dim=8)
    store = VectorStore(db)
    dim = _vec_dim(db)
    stored = np.full(dim, 0.5, dtype=np.float32)
    store.upsert("node-1", stored)
    out = stored_or_encoded(store, _ExplodingEncoder(), "node-1", "some text")
    assert np.allclose(out, stored)


def test_stored_or_encoded_falls_back_and_warns(db, caplog):
    db.init_vec_table(dim=8)
    store = VectorStore(db)
    enc = _CountingEncoder(_vec_dim(db))
    with caplog.at_level("WARNING"):
        out = stored_or_encoded(store, enc, "missing-node", "some text")
    assert enc.calls == 1
    assert out.shape == (enc.dim,)
    assert any("re-encoding" in r.message for r in caplog.records)


def test_find_link_candidates_does_not_reencode(engine, monkeypatch, caplog):
    from ormah.background import auto_linker
    from ormah.models.node import CreateNodeRequest

    for i in range(2):
        engine.remember(CreateNodeRequest(content=f"same fact repeated {i}", title="same fact"))
    monkeypatch.setattr(
        "ormah.embeddings.encoder.get_encoder", lambda s: _ExplodingEncoder()
    )
    # must not raise: every probe vector already sits in node_vectors
    with caplog.at_level("WARNING"):
        auto_linker._find_link_candidates(engine, limit=4)
    assert not [r for r in caplog.records if "failed" in r.message]


def test_find_merge_candidates_does_not_reencode(engine, monkeypatch, caplog):
    from ormah.background import duplicate_merger
    from ormah.models.node import CreateNodeRequest

    for i in range(2):
        engine.remember(CreateNodeRequest(content=f"same fact repeated {i}", title="same fact"))
    monkeypatch.setattr(
        "ormah.embeddings.encoder.get_encoder", lambda s: _ExplodingEncoder()
    )
    with caplog.at_level("WARNING"):
        duplicate_merger._find_merge_candidates(engine, limit=4)
    assert not [r for r in caplog.records if "failed" in r.message]


def test_find_conflict_candidates_does_not_reencode(engine, monkeypatch, caplog):
    from ormah.background import conflict_detector
    from ormah.models.node import CreateNodeRequest

    for i in range(2):
        engine.remember(CreateNodeRequest(
            content=f"same fact repeated {i}", title="same fact", type="preference"
        ))
    monkeypatch.setattr(
        "ormah.embeddings.encoder.get_encoder", lambda s: _ExplodingEncoder()
    )
    with caplog.at_level("WARNING"):
        conflict_detector._find_conflict_candidates(engine, limit=4)
    assert not [r for r in caplog.records if "failed" in r.message]

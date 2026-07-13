"""Guard: a dim mismatch must never silently DROP a populated vector store."""
from __future__ import annotations

import numpy as np
import pytest

from ormah.embeddings.vector_store import VectorStore


def _count(engine) -> int:
    return engine.db.conn.execute("SELECT count(*) FROM node_vectors").fetchone()[0]


def test_dim_mismatch_on_populated_store_raises_not_drops(engine):
    dim = engine.settings.embedding_dim
    VectorStore(engine.db).upsert("n1", np.ones(dim, dtype=np.float32))
    before = _count(engine)
    assert before > 0

    with pytest.raises(RuntimeError, match="dimension mismatch"):
        engine.db.init_vec_table(dim + 256)  # e.g. config default 768 vs stored 1024

    assert _count(engine) == before  # nothing dropped


def test_dim_mismatch_on_empty_store_recreates(engine):
    with engine.db.transaction() as conn:
        conn.execute("DELETE FROM node_vectors")
    new_dim = engine.settings.embedding_dim + 256

    engine.db.init_vec_table(new_dim)  # empty → safe to recreate, no raise

    row = engine.db.conn.execute(
        "SELECT sql FROM sqlite_master WHERE name='node_vectors'"
    ).fetchone()
    assert f"FLOAT[{new_dim}]" in row[0]


def test_allow_drop_authorizes_reindex(engine):
    dim = engine.settings.embedding_dim
    VectorStore(engine.db).upsert("n1", np.ones(dim, dtype=np.float32))

    engine.db.init_vec_table(dim + 256, allow_drop=True)  # explicit → drops

    assert _count(engine) == 0
    row = engine.db.conn.execute(
        "SELECT sql FROM sqlite_master WHERE name='node_vectors'"
    ).fetchone()
    assert f"FLOAT[{dim + 256}]" in row[0]


def test_unparseable_ddl_raises_and_preserves_rows(engine):
    """A node_vectors table whose DDL has no FLOAT[dim] (corrupt/foreign schema)
    must fail closed: never guess-drop, never boot into broken vector search."""
    with engine.db.transaction() as conn:
        conn.execute("DROP TABLE node_vectors")
        conn.execute("CREATE TABLE node_vectors (id TEXT PRIMARY KEY, embedding BLOB)")
        conn.execute("INSERT INTO node_vectors (id, embedding) VALUES ('n1', x'00')")

    with pytest.raises(RuntimeError, match="FLOAT"):
        engine.db.init_vec_table(engine.settings.embedding_dim)

    assert _count(engine) == 1  # rows preserved for inspection/recovery


def test_startup_wiring_respects_reindex_flag(settings):
    """MemoryEngine.__init__ authorizes the drop only when the flag equals the
    configured dim; a stale flag from a previous migration keeps refusing."""
    from ormah.engine.memory_engine import MemoryEngine

    eng = MemoryEngine(settings)
    dim = settings.embedding_dim
    VectorStore(eng.db).upsert("n1", np.ones(dim, dtype=np.float32))
    eng.db.close()

    # Stale flag: authorizes the OLD dim while config asks a new one → refuse.
    settings.embedding_dim = dim + 256
    settings.reindex_on_dim_change = dim  # stale value != new dim
    with pytest.raises(RuntimeError, match="dimension mismatch"):
        MemoryEngine(settings)

    # Correct flag: equals the NEW dim → authorized drop.
    settings.reindex_on_dim_change = dim + 256
    eng2 = MemoryEngine(settings)
    assert eng2.db.conn.execute("SELECT count(*) FROM node_vectors").fetchone()[0] == 0
    eng2.db.close()

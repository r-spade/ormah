"""Tests for index builder."""

import pytest

from ormah.index.builder import IndexBuilder
from ormah.models.node import MemoryNode, NodeType


def test_full_rebuild(db, file_store):
    # Create some nodes on disk
    for i in range(3):
        node = MemoryNode(
            type=NodeType.fact,
            source="agent:test",
            content=f"Fact {i} for indexing.",
            title=f"Fact {i}",
        )
        file_store.save(node)

    builder = IndexBuilder(db, file_store)
    count = builder.full_rebuild()
    assert count == 3

    # Verify in DB
    rows = db.conn.execute("SELECT COUNT(*) FROM nodes").fetchone()
    assert rows[0] == 3


def test_incremental_update(db, file_store):
    builder = IndexBuilder(db, file_store)

    # Initial build
    node = MemoryNode(
        type=NodeType.fact,
        source="agent:test",
        content="Original content.",
        title="Original",
    )
    file_store.save(node)
    builder.full_rebuild()

    # Add another node
    node2 = MemoryNode(
        type=NodeType.decision,
        source="agent:test",
        content="New decision.",
        title="Decision",
    )
    file_store.save(node2)

    added, updated = builder.incremental_update()
    assert added == 1
    assert updated == 0

    rows = db.conn.execute("SELECT COUNT(*) FROM nodes").fetchone()
    assert rows[0] == 2


def test_full_rebuild_aborts_and_preserves_data_on_total_failure(db, file_store, monkeypatch):
    """A rebuild where every file fails to index must NOT persist a truncated index —
    it must ROLLBACK, preserving whatever was committed before (the nodes-empty incident)."""
    for i in range(3):
        node = MemoryNode(
            type=NodeType.fact,
            source="agent:test",
            content=f"Fact {i} for indexing.",
            title=f"Fact {i}",
        )
        file_store.save(node)

    builder = IndexBuilder(db, file_store)
    builder.full_rebuild()  # seed the index from the fixture store
    before = db.conn.execute("SELECT COUNT(*) FROM nodes").fetchone()[0]
    assert before == 3

    def boom(_path):
        raise OSError(24, "Too many open files")

    monkeypatch.setattr(builder, "_index_file_nodes_only", boom)

    with pytest.raises(RuntimeError, match=r"0/3 files"):
        builder.full_rebuild()

    after = db.conn.execute("SELECT COUNT(*) FROM nodes").fetchone()[0]
    assert after == before  # ROLLBACK preserved the prior committed state


def test_full_rebuild_aborts_and_preserves_data_on_partial_failure(db, file_store, monkeypatch):
    """One file succeeding out of many must still abort the rebuild (not just count==0) —
    a partial index committed as "complete" is the exact silent-degradation risk the
    count==0 guard misses."""
    for i in range(3):
        node = MemoryNode(
            type=NodeType.fact,
            source="agent:test",
            content=f"Fact {i} for indexing.",
            title=f"Fact {i}",
        )
        file_store.save(node)

    builder = IndexBuilder(db, file_store)
    builder.full_rebuild()  # seed the index from the fixture store
    before = db.conn.execute("SELECT COUNT(*) FROM nodes").fetchone()[0]
    assert before == 3

    original = builder._index_file_nodes_only
    calls = {"n": 0}

    def flaky(path):
        calls["n"] += 1
        if calls["n"] == 1:
            return original(path)
        raise OSError(24, "Too many open files")

    monkeypatch.setattr(builder, "_index_file_nodes_only", flaky)

    with pytest.raises(RuntimeError, match=r"1/3 files"):
        builder.full_rebuild()

    after = db.conn.execute("SELECT COUNT(*) FROM nodes").fetchone()[0]
    assert after == before  # ROLLBACK preserved the prior committed state, not a 1-node index


def test_full_rebuild_allow_partial_accepts_incomplete_pass(db, file_store, monkeypatch):
    """allow_partial=True is the explicit opt-out: a partial pass is committed instead of
    raising, for callers that intentionally tolerate known-corrupt files."""
    for i in range(3):
        node = MemoryNode(
            type=NodeType.fact,
            source="agent:test",
            content=f"Fact {i} for indexing.",
            title=f"Fact {i}",
        )
        file_store.save(node)

    builder = IndexBuilder(db, file_store)

    original = builder._index_file_nodes_only
    calls = {"n": 0}

    def flaky(path):
        calls["n"] += 1
        if calls["n"] == 1:
            return original(path)
        raise OSError(24, "Too many open files")

    monkeypatch.setattr(builder, "_index_file_nodes_only", flaky)

    count = builder.full_rebuild(allow_partial=True)
    assert count == 1

    rows = db.conn.execute("SELECT COUNT(*) FROM nodes").fetchone()
    assert rows[0] == 1


def test_full_rebuild_edge_failure_does_not_abort_but_is_surfaced(db, file_store, monkeypatch, caplog):
    """A per-file edge-indexing failure must NOT abort the rebuild (edges are derived and
    self-healing; aborting on one bad link would roll back every good node and risk an empty
    store). But the aggregate failure must be surfaced, not swallowed silently (council-pr H1)."""
    import logging

    for i in range(3):
        node = MemoryNode(
            type=NodeType.fact,
            source="agent:test",
            content=f"Fact {i} for indexing.",
            title=f"Fact {i}",
        )
        file_store.save(node)

    builder = IndexBuilder(db, file_store)
    original = builder._index_file_edges

    def flaky_edges(path):
        if "Fact 1" in path.read_text(encoding="utf-8"):
            raise RuntimeError("bad link")
        return original(path)

    monkeypatch.setattr(builder, "_index_file_edges", flaky_edges)

    with caplog.at_level(logging.ERROR, logger="ormah.index.builder"):
        count = builder.full_rebuild()

    assert count == 3  # all nodes committed despite the edge failure
    assert db.conn.execute("SELECT COUNT(*) FROM nodes").fetchone()[0] == 3
    assert any("failed edge indexing" in r.message for r in caplog.records)  # surfaced, not silent

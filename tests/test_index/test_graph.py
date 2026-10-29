"""Tests for GraphIndex batch methods."""

from __future__ import annotations

import pytest

from ormah.index.graph import GraphIndex


@pytest.fixture
def graph(db):
    """GraphIndex backed by a real test database."""
    return GraphIndex(db.conn)


def _insert_node(db, node_id: str, content: str = "test", **kwargs):
    """Insert a minimal node row directly."""
    db.conn.execute(
        """INSERT INTO nodes (id, type, tier, content, title, source, space,
           created, updated, last_accessed, access_count, importance, confidence,
           file_path, file_hash)
           VALUES (?, ?, ?, ?, ?, ?, ?, datetime('now'), datetime('now'),
           datetime('now'), 0, 0.5, 1.0, '/fake/path', 'abc123')""",
        (
            node_id,
            kwargs.get("type", "fact"),
            kwargs.get("tier", "working"),
            content,
            kwargs.get("title", content),
            kwargs.get("source", "test"),
            kwargs.get("space"),
        ),
    )
    db.conn.commit()


def _insert_tag(db, node_id: str, tag: str):
    """Insert a tag for a node."""
    db.conn.execute(
        "INSERT INTO node_tags (node_id, tag) VALUES (?, ?)", (node_id, tag)
    )
    db.conn.commit()


# ---------------------------------------------------------------------------
# get_nodes_batch
# ---------------------------------------------------------------------------


def test_get_nodes_batch(db, graph):
    _insert_node(db, "n1", "first")
    _insert_node(db, "n2", "second")
    _insert_node(db, "n3", "third")

    result = graph.get_nodes_batch(["n1", "n3"])
    assert set(result.keys()) == {"n1", "n3"}
    assert result["n1"]["content"] == "first"
    assert result["n3"]["content"] == "third"


def test_get_nodes_batch_empty(graph):
    result = graph.get_nodes_batch([])
    assert result == {}


def test_get_nodes_batch_missing_ids(db, graph):
    _insert_node(db, "n1", "exists")
    result = graph.get_nodes_batch(["n1", "missing"])
    assert "n1" in result
    assert "missing" not in result


# ---------------------------------------------------------------------------
# get_tags_batch
# ---------------------------------------------------------------------------


def test_get_tags_batch(db, graph):
    _insert_node(db, "n1", "first")
    _insert_node(db, "n2", "second")
    _insert_tag(db, "n1", "alpha")
    _insert_tag(db, "n1", "beta")
    _insert_tag(db, "n2", "gamma")

    result = graph.get_tags_batch(["n1", "n2"])
    assert result["n1"] == {"alpha", "beta"}
    assert result["n2"] == {"gamma"}


def test_get_tags_batch_empty(graph):
    result = graph.get_tags_batch([])
    assert result == {}


def test_get_tags_batch_no_tags(db, graph):
    _insert_node(db, "n1", "no tags")
    result = graph.get_tags_batch(["n1"])
    # Node exists but has no tags — should not appear in result
    assert result == {}


# ---------------------------------------------------------------------------
# get_node — Prefix resolution
# ---------------------------------------------------------------------------

FIRST = "deadbeef-0000-0000-0000-00000000000a"
SECOND = "deadbeef-0000-0000-0000-00000000000b"


def test_get_node_resolves_an_ambiguous_prefix_to_nothing(db, graph):
    """Two nodes share the Short id `deadbeef`: the prefix names neither of them."""
    _insert_node(db, FIRST, "first")
    _insert_node(db, SECOND, "second")

    assert graph.get_node("deadbeef") is None


def test_get_node_resolves_a_unique_prefix(db, graph):
    _insert_node(db, FIRST, "first")
    _insert_node(db, "cafebabe-0000-0000-0000-00000000000b", "other")

    assert graph.get_node("deadbeef")["id"] == FIRST


def test_get_node_by_full_id_ignores_a_node_sharing_its_prefix(db, graph):
    _insert_node(db, FIRST, "first")
    _insert_node(db, SECOND, "second")

    assert graph.get_node(SECOND)["content"] == "second"
    assert graph.get_node("deadbeef-0000-0000-0000-00000000000c") is None

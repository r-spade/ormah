"""Tests for FileStore in-memory ID-to-path cache."""

from __future__ import annotations

from ormah.models.node import MemoryNode, NodeType, Tier
from ormah.store.file_store import FileStore


def _make_node(title="Test", content="Test content") -> MemoryNode:
    return MemoryNode(type=NodeType.fact, tier=Tier.working, title=title, content=content, source="test")


def test_cache_populated_on_save(tmp_path):
    store = FileStore(tmp_path / "nodes")
    node = _make_node()
    store.save(node)

    assert node.id in store._id_cache
    assert store._id_cache[node.id].exists()


def test_cache_hit_on_load(tmp_path):
    store = FileStore(tmp_path / "nodes")
    node = _make_node()
    store.save(node)

    # Load should use cache — no glob needed
    loaded = store.load(node.id)
    assert loaded is not None
    assert loaded.id == node.id


def test_cache_cleared_on_delete(tmp_path):
    store = FileStore(tmp_path / "nodes")
    node = _make_node()
    store.save(node)
    assert node.id in store._id_cache

    store.delete(node.id)
    assert node.id not in store._id_cache


def test_stale_cache_entry_recovers(tmp_path):
    """If the cached path no longer exists, _find_file still finds via glob."""
    store = FileStore(tmp_path / "nodes")
    node = _make_node()
    path = store.save(node)

    # Corrupt the cache entry to point to a nonexistent path
    store._id_cache[node.id] = tmp_path / "nodes" / "ghost.md"

    # Should recover via glob fallback
    found = store._find_file(node.id)
    assert found is not None
    assert found.exists()
    # Cache should be updated
    assert store._id_cache[node.id] == found


def test_full_cache_build_finds_all_nodes(tmp_path):
    store = FileStore(tmp_path / "nodes")
    nodes = [_make_node(title=f"Node {i}", content=f"Content {i}") for i in range(5)]
    for n in nodes:
        store.save(n)

    # Clear cache, force rebuild
    store._id_cache.clear()
    store._cache_built = False

    # Access a node whose short_id glob won't match (simulate rename)
    # Actually just verify _build_cache populates all
    store._build_cache()

    assert len(store._id_cache) == 5
    for n in nodes:
        assert n.id in store._id_cache


def test_find_file_returns_none_for_missing(tmp_path):
    store = FileStore(tmp_path / "nodes")
    assert store._find_file("nonexistent-id") is None
    # Cache should be fully built after the miss
    assert store._cache_built is True

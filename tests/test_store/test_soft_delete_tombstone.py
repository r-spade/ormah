from __future__ import annotations

import frontmatter

from ormah.models.node import MemoryNode, NodeType
from ormah.store.file_store import FileStore


def _store(tmp_path):
    return FileStore(tmp_path / "nodes")


def _make(store, content="x"):
    node = MemoryNode(type=NodeType.fact, content=content, title=content)
    store.save(node)
    return node


def test_soft_delete_stamps_deleted_at(tmp_path):
    store = _store(tmp_path)
    node = _make(store)

    assert store.soft_delete(node.id) is True

    deleted_dir = tmp_path / "deleted"
    files = list(deleted_dir.glob("*.md"))
    assert len(files) == 1
    meta = frontmatter.loads(files[0].read_text(encoding="utf-8")).metadata
    assert "deleted_at" in meta and meta["deleted_at"]


def test_list_deleted_returns_id_and_deleted_at(tmp_path):
    store = _store(tmp_path)
    node = _make(store)
    store.soft_delete(node.id)

    entries = store.list_deleted()
    assert len(entries) == 1
    node_id, deleted_at, path = entries[0]
    assert node_id == node.id
    assert deleted_at is not None
    assert path.exists()


def test_purge_removes_tombstone(tmp_path):
    store = _store(tmp_path)
    node = _make(store)
    store.soft_delete(node.id)

    assert store.purge(node.id) is True
    assert store.list_deleted() == []
    assert store.purge(node.id) is False  # idempotent: already gone

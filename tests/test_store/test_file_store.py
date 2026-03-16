"""Tests for file store CRUD operations."""

from ormah.models.node import MemoryNode, NodeType, Tier


def test_save_and_load(file_store):
    node = MemoryNode(
        type=NodeType.fact,
        tier=Tier.working,
        source="agent:test",
        content="Test fact content.",
        title="Test fact",
    )

    path = file_store.save(node)
    assert path.exists()
    assert path.suffix == ".md"

    loaded = file_store.load(node.id)
    assert loaded is not None
    assert loaded.id == node.id
    assert loaded.content == "Test fact content."


def test_delete(file_store):
    node = MemoryNode(
        type=NodeType.fact,
        source="agent:test",
        content="To be deleted.",
    )
    file_store.save(node)
    assert file_store.load(node.id) is not None

    result = file_store.delete(node.id)
    assert result is True
    assert file_store.load(node.id) is None


def test_list_all(file_store):
    for i in range(3):
        node = MemoryNode(
            type=NodeType.fact,
            source="agent:test",
            content=f"Fact number {i}",
        )
        file_store.save(node)

    nodes = file_store.list_all()
    assert len(nodes) == 3


def test_soft_delete_moves_file(file_store):
    node = MemoryNode(
        type=NodeType.fact,
        source="agent:test",
        content="To be soft deleted.",
        title="Soft delete me",
    )
    path = file_store.save(node)
    assert path.exists()

    result = file_store.soft_delete(node.id)
    assert result is True

    # Original file gone
    assert not path.exists()

    # File exists in deleted/ directory
    deleted_dir = file_store.nodes_dir.parent / "deleted"
    dest = deleted_dir / path.name
    assert dest.exists()


def test_soft_delete_nonexistent_returns_false(file_store):
    result = file_store.soft_delete("nonexistent-id")
    assert result is False


def test_soft_delete_clears_cache(file_store):
    node = MemoryNode(
        type=NodeType.fact,
        source="agent:test",
        content="Cache test.",
        title="Cache node",
    )
    file_store.save(node)
    assert file_store.load(node.id) is not None

    file_store.soft_delete(node.id)
    assert file_store.load(node.id) is None


def test_soft_deleted_not_in_list_all(file_store):
    node = MemoryNode(
        type=NodeType.fact,
        source="agent:test",
        content="Listed then gone.",
        title="Listed node",
    )
    file_store.save(node)
    assert len(file_store.list_all()) == 1

    file_store.soft_delete(node.id)
    assert len(file_store.list_all()) == 0


def test_soft_deleted_not_in_list_paths(file_store):
    node = MemoryNode(
        type=NodeType.fact,
        source="agent:test",
        content="Paths test.",
        title="Paths node",
    )
    file_store.save(node)
    assert len(file_store.list_paths()) == 1

    file_store.soft_delete(node.id)
    assert len(file_store.list_paths()) == 0


def test_touch_access(file_store):
    node = MemoryNode(
        type=NodeType.fact,
        source="agent:test",
        content="Access me.",
        access_count=0,
    )
    file_store.save(node)

    updated = file_store.touch_access(node.id)
    assert updated is not None
    assert updated.access_count == 1

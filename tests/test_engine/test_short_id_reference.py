"""An update addressed by a Short id still resolves end to end (#280)."""

from __future__ import annotations

from ormah.models.node import CreateNodeRequest, MemoryNode, NodeType, Tier, UpdateNodeRequest

COLLIDING_SHORT_ID = "deadbeef"


def test_update_node_by_short_id_reaches_the_right_node(engine):
    """update_node forwards node_id straight to FileStore.load — this seam breaks
    if the engine ever normalises or pre-validates the reference before the store
    sees it. A Short id that matches exactly one node must resolve to that node."""
    node_id, _ = engine.remember(
        CreateNodeRequest(content="original content", type=NodeType.fact, title="Short id target")
    )
    short_id = engine.file_store.load(node_id).short_id

    result = engine.update_node(short_id, UpdateNodeRequest(content="updated content"))

    assert result is not None
    node = engine.file_store.load(node_id)
    assert node.content == "updated content"


def _index_collider(engine, suffix: str, title: str) -> MemoryNode:
    """Save and index a node whose Full id starts with the shared Short id."""
    node = MemoryNode(
        id=f"{COLLIDING_SHORT_ID}-0000-0000-0000-00000000000{suffix}",
        type=NodeType.fact,
        tier=Tier.working,
        title=title,
        content=f"content of {title}",
        source="test",
    )
    engine.builder.index_single(engine.file_store.save(node))
    return node


def _live_files(engine) -> list[str]:
    return sorted(p.name for p in engine.file_store.nodes_dir.glob("*.md"))


def test_delete_by_unique_short_id_acts_on_the_resolved_full_id(engine):
    """The index row, the audit entry and the file must all follow the Full id the
    store resolved — not the Short id the caller typed, which names no index row."""
    target = _index_collider(engine, "a", "Unique target")

    result = engine.delete_node(COLLIDING_SHORT_ID)

    assert result is not None and result.startswith("Deleted")
    assert engine.graph.get_node(target.id) is None
    assert engine.file_store.load(target.id) is None
    [entry] = engine.list_audit_log(operation="delete")
    assert entry["node_id"] == target.id


def test_delete_by_ambiguous_short_id_mutates_nothing(engine):
    """Ambiguity is not absence: the delete must refuse before the index fallback
    picks one of the colliders, and before it writes an audit entry."""
    first = _index_collider(engine, "a", "Collider A")
    second = _index_collider(engine, "b", "Collider B")
    files_before = _live_files(engine)

    result = engine.delete_node(COLLIDING_SHORT_ID)

    assert result is not None and not result.startswith("Deleted")
    assert _live_files(engine) == files_before
    assert engine.graph.get_node(first.id) is not None
    assert engine.graph.get_node(second.id) is not None
    assert engine.list_audit_log(operation="delete") == []


def test_delete_of_a_node_whose_file_is_malformed_leaves_the_index_alone(engine):
    """A file that will not parse confirms nothing, but it does not confirm absence
    either: it may be this very node. The index fallback must not remove the row
    while the file is still in nodes/ to be repaired and reindexed."""
    target = _index_collider(engine, "a", "Soon corrupt")
    path = engine.file_store._find_file(target.id)
    path.write_text("this is not a node", encoding="utf-8")
    engine.file_store._id_cache.clear()  # cold, as after a restart
    engine.file_store._cache_built = False

    result = engine.delete_node(target.id)

    assert result is None or not result.startswith("Deleted")
    assert engine.graph.get_node(target.id) is not None
    assert path.exists()
    assert engine.list_audit_log(operation="delete") == []


def test_delete_by_a_prefix_the_store_does_not_resolve_mutates_nothing(engine):
    """The store resolves only a Full id or an 8-character Short id, so a shorter prefix
    is a confirmed absence there. The index fallback must not turn it into a target:
    `get_node` answers a prefix with `LIKE ... LIMIT 1`, one arbitrary row of many."""
    first = _index_collider(engine, "a", "Collider A")
    second = _index_collider(engine, "b", "Collider B")
    files_before = _live_files(engine)

    result = engine.delete_node(COLLIDING_SHORT_ID[:4])

    assert result is None or not result.startswith("Deleted")
    assert _live_files(engine) == files_before
    assert engine.graph.get_node(first.id) is not None
    assert engine.graph.get_node(second.id) is not None
    assert engine.list_audit_log(operation="delete") == []


def test_delete_by_the_self_nodes_short_id_is_refused(engine):
    """The Self guard must see the Full id the store resolved, not the reference the
    caller typed: a Short id would otherwise slip past it."""
    self_id = engine.user_node_id

    result = engine.delete_node(self_id.split("-")[0])

    assert result == "Cannot delete the user self node."
    assert engine.file_store.load(self_id) is not None
    assert engine.graph.get_node(self_id) is not None
    assert engine.list_audit_log(operation="delete") == []

"""Tests for execute_merge and undo_merge operations."""

from __future__ import annotations

from ormah.models.node import CreateNodeRequest, NodeType, EdgeType, ConnectRequest


def _create_node(engine, title="Test", content="Test content", node_type=NodeType.fact, **kwargs):
    """Helper to create a node, returns (id, slug)."""
    req = CreateNodeRequest(content=content, type=node_type, title=title, tags=["test"], **kwargs)
    return engine.remember(req, agent_id="test")


# --- Basic merge ---

def test_merge_keeps_higher_tier_node(engine):
    """When merging nodes of different tiers, the higher-tier node is kept."""
    id_a, _ = _create_node(engine, title="Core fact", content="Important info")
    id_b, _ = _create_node(engine, title="Working fact", content="Less important info")

    # Promote node A to core
    node_a = engine.file_store.load(id_a)
    from ormah.models.node import Tier
    node_a.tier = Tier.core
    engine.file_store.save(node_a)
    engine.db.conn.execute("UPDATE nodes SET tier = 'core' WHERE id = ?", (id_a,))
    engine.db.conn.commit()

    result = engine.execute_merge(id_a, id_b)

    assert id_a[:8] in result
    assert "kept" in result
    assert engine.file_store.load(id_a) is not None
    assert engine.file_store.load(id_b) is None


def test_merge_keeps_longer_content_same_tier(engine):
    """When tiers match, the node with more content is kept."""
    id_a, _ = _create_node(engine, title="Short", content="Short.")
    id_b, _ = _create_node(engine, title="Long", content="This is a much longer description with detail.")

    engine.execute_merge(id_a, id_b)

    assert engine.file_store.load(id_b) is not None
    assert engine.file_store.load(id_a) is None


def test_merge_applies_llm_content(engine):
    """Merged content/title from LLM is applied to the kept node."""
    id_a, _ = _create_node(engine, title="Python", content="Python is a language.")
    id_b, _ = _create_node(engine, title="Python lang", content="Python is popular.")

    engine.execute_merge(
        id_a, id_b,
        merged_content="Python is a popular programming language.",
        merged_title="Python Programming Language",
    )

    kept = engine.file_store.load(id_a) or engine.file_store.load(id_b)
    assert kept is not None
    assert kept.content == "Python is a popular programming language."
    assert kept.title == "Python Programming Language"


def test_merge_combines_tags(engine):
    """Tags from the removed node are merged into the kept node."""
    id_a, _ = _create_node(engine, title="A", content="Content A long enough")
    id_b, _ = _create_node(engine, title="B", content="Content B")

    # Add unique tags to B
    node_b = engine.file_store.load(id_b)
    node_b.tags.append("unique-tag")
    engine.file_store.save(node_b)

    engine.execute_merge(id_a, id_b)

    kept = engine.file_store.load(id_a)
    assert kept is not None
    assert "unique-tag" in kept.tags


# --- Edge remapping ---

def test_merge_remaps_edges(engine):
    """Edges from the removed node are remapped to the kept node."""
    id_a, _ = _create_node(engine, title="Kept", content="This node will be kept because longer")
    id_b, _ = _create_node(engine, title="Removed", content="Shorter content")
    id_c, _ = _create_node(engine, title="Neighbor", content="A neighbor node")

    # Create edge: B -> C
    engine.connect(ConnectRequest(
        source_id=id_b, target_id=id_c, edge=EdgeType.supports, weight=0.8
    ))

    engine.execute_merge(id_a, id_b)

    # Edge should now be A -> C (remapped)
    edges = engine.db.conn.execute(
        "SELECT * FROM edges WHERE source_id = ? AND target_id = ? AND edge_type = 'supports'",
        (id_a, id_c),
    ).fetchall()
    assert len(edges) >= 1


def test_merge_skips_self_loop_edges(engine):
    """When remapping creates a self-loop, the edge is dropped."""
    id_a, _ = _create_node(engine, title="A", content="Longer content for keeping")
    id_b, _ = _create_node(engine, title="B", content="Short content")

    # Create edge: B -> A (after merge, would become A -> A)
    engine.connect(ConnectRequest(
        source_id=id_b, target_id=id_a, edge=EdgeType.related_to, weight=0.5
    ))

    engine.execute_merge(id_a, id_b)

    # No self-loop should exist
    self_loops = engine.db.conn.execute(
        "SELECT * FROM edges WHERE source_id = ? AND target_id = ?",
        (id_a, id_a),
    ).fetchall()
    assert len(self_loops) == 0


def test_merge_skips_duplicate_edges(engine):
    """When remapping would duplicate an existing edge, it's skipped."""
    id_a, _ = _create_node(engine, title="A", content="Longer content for keeping")
    id_b, _ = _create_node(engine, title="B", content="Short content")
    id_c, _ = _create_node(engine, title="C", content="Neighbor node content")

    # Both A and B have supports edges to C
    engine.connect(ConnectRequest(
        source_id=id_a, target_id=id_c, edge=EdgeType.supports, weight=0.9
    ))
    engine.connect(ConnectRequest(
        source_id=id_b, target_id=id_c, edge=EdgeType.supports, weight=0.7
    ))

    engine.execute_merge(id_a, id_b)

    # Should only have one supports edge from A to C (not two)
    edges = engine.db.conn.execute(
        "SELECT * FROM edges WHERE source_id = ? AND target_id = ? AND edge_type = 'supports'",
        (id_a, id_c),
    ).fetchall()
    assert len(edges) == 1


# --- Merge history ---

def test_merge_creates_history_entry(engine):
    """execute_merge creates a record in merge_history."""
    id_a, _ = _create_node(engine, title="A", content="Content A")
    id_b, _ = _create_node(engine, title="B", content="Content B")

    engine.execute_merge(id_a, id_b)

    merges = engine.list_merges()
    assert len(merges) == 1
    assert merges[0]["undone_at"] is None


# --- Undo merge ---

def test_undo_restores_removed_node(engine):
    """Undoing a merge restores the removed node."""
    id_a, _ = _create_node(engine, title="A", content="Longer content for keeping")
    id_b, _ = _create_node(engine, title="B", content="Short content")

    original_b = engine.file_store.load(id_b)

    engine.execute_merge(id_a, id_b)
    assert engine.file_store.load(id_b) is None

    merge = engine.list_merges()[0]
    engine.undo_merge(merge["id"])

    restored = engine.file_store.load(id_b)
    assert restored is not None
    assert restored.content == original_b.content
    assert restored.title == original_b.title


def test_undo_restores_original_edges(engine):
    """Undoing a merge restores the removed node's original edges."""
    id_a, _ = _create_node(engine, title="A", content="Longer content for keeping")
    id_b, _ = _create_node(engine, title="B", content="Short content")
    id_c, _ = _create_node(engine, title="C", content="Neighbor node content")

    # Create edge: B -> C
    engine.connect(ConnectRequest(
        source_id=id_b, target_id=id_c, edge=EdgeType.supports, weight=0.8
    ))

    engine.execute_merge(id_a, id_b)

    merge = engine.list_merges()[0]
    engine.undo_merge(merge["id"])

    # Original edge B -> C should be restored
    edges = engine.db.conn.execute(
        "SELECT * FROM edges WHERE source_id = ? AND target_id = ? AND edge_type = 'supports'",
        (id_b, id_c),
    ).fetchall()
    assert len(edges) >= 1


def test_undo_removes_remapped_edges(engine):
    """Undoing a merge removes edges that were remapped during the merge."""
    id_a, _ = _create_node(engine, title="A", content="Longer content for keeping")
    id_b, _ = _create_node(engine, title="B", content="Short content")
    id_c, _ = _create_node(engine, title="C", content="Neighbor node content")

    # Create edge: B -> C (will be remapped to A -> C during merge)
    engine.connect(ConnectRequest(
        source_id=id_b, target_id=id_c, edge=EdgeType.supports, weight=0.8
    ))

    # Verify no A -> C supports edge before merge
    pre_edges = engine.db.conn.execute(
        "SELECT * FROM edges WHERE source_id = ? AND target_id = ? AND edge_type = 'supports'",
        (id_a, id_c),
    ).fetchall()
    assert len(pre_edges) == 0

    engine.execute_merge(id_a, id_b)

    # A -> C should exist after merge (remapped)
    post_edges = engine.db.conn.execute(
        "SELECT * FROM edges WHERE source_id = ? AND target_id = ? AND edge_type = 'supports'",
        (id_a, id_c),
    ).fetchall()
    assert len(post_edges) >= 1

    merge = engine.list_merges()[0]
    engine.undo_merge(merge["id"])

    # Remapped A -> C should be gone
    after_undo = engine.db.conn.execute(
        "SELECT * FROM edges WHERE source_id = ? AND target_id = ? AND edge_type = 'supports'",
        (id_a, id_c),
    ).fetchall()
    assert len(after_undo) == 0


def test_undo_prefix_match(engine):
    """undo_merge supports prefix matching on merge IDs."""
    id_a, _ = _create_node(engine, title="A", content="Content A")
    id_b, _ = _create_node(engine, title="B", content="Content B")

    engine.execute_merge(id_a, id_b)

    merge = engine.list_merges()[0]
    prefix = merge["id"][:8]
    result = engine.undo_merge(prefix)

    assert "Undone" in result


def test_double_undo_rejected(engine):
    """Undoing an already-undone merge returns an error."""
    id_a, _ = _create_node(engine, title="A", content="Content A")
    id_b, _ = _create_node(engine, title="B", content="Content B")

    engine.execute_merge(id_a, id_b)

    merge = engine.list_merges()[0]
    engine.undo_merge(merge["id"])

    result = engine.undo_merge(merge["id"])
    assert "already undone" in result


def test_undo_marks_undone_at(engine):
    """After undo, the merge_history entry has undone_at set."""
    id_a, _ = _create_node(engine, title="A", content="Content A")
    id_b, _ = _create_node(engine, title="B", content="Content B")

    engine.execute_merge(id_a, id_b)

    merge = engine.list_merges()[0]
    engine.undo_merge(merge["id"])

    updated = engine.list_merges()[0]
    assert updated["undone_at"] is not None


# --- Error cases ---

def test_merge_missing_node_returns_error(engine):
    """Merging with a non-existent node returns an error string."""
    id_a, _ = _create_node(engine, title="A", content="Content A")

    result = engine.execute_merge(id_a, "nonexistent-id")
    assert "not found" in result


def test_undo_missing_merge_returns_error(engine):
    """Undoing a non-existent merge returns an error string."""
    result = engine.undo_merge("nonexistent-id")
    assert "not found" in result

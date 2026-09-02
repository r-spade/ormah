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


def test_undo_does_not_delete_a_preexisting_collided_edge(engine):
    """Undo must not delete an edge undo never created (Codex council finding, #123).

    C declares BOTH C->A and C->B with the SAME edge_type in its own markdown. During the
    merge, the remap loop for B->A skips the INSERT because C->A already exists (the
    "already exists in either direction" check). undo_merge must not delete that
    pre-existing C->A row just because it appears, remapped, in original_edges — it was
    never inserted by execute_merge.
    """
    from ormah.models.node import Connection

    original_threshold = engine.settings.auto_link_similarity_threshold
    engine.settings.auto_link_similarity_threshold = 999.0
    try:
        id_a, _ = _create_node(
            engine, title="Kept", content="This node is kept because its content is much longer"
        )
        id_b, _ = _create_node(engine, title="Removed", content="Short")
        id_c, _ = _create_node(engine, title="Third", content="An unrelated third node entirely")
    finally:
        engine.settings.auto_link_similarity_threshold = original_threshold

    # C -> A (weight 0.8) and C -> B (weight 0.6), same edge_type — the collision.
    node_c = engine.file_store.load(id_c)
    node_c.connections.append(Connection(target=id_a, edge=EdgeType.supports, weight=0.8))
    node_c.connections.append(Connection(target=id_b, edge=EdgeType.supports, weight=0.6))
    engine.file_store.save(node_c)
    engine.builder.index_single(engine.file_store._path_for(node_c))

    def c_to_a():
        return engine.db.conn.execute(
            "SELECT weight FROM edges WHERE source_id = ? AND target_id = ? "
            "AND edge_type = 'supports'",
            (id_c, id_a),
        ).fetchall()

    before = c_to_a()
    assert len(before) == 1, "sanity: C -> A exists before the merge"

    engine.execute_merge(id_a, id_b)
    assert len(c_to_a()) == 1, "C -> A must survive the merge"

    merge = engine.list_merges()[0]
    engine.undo_merge(merge["id"])

    after = c_to_a()
    assert len(after) == 1, (
        "undo deleted a pre-existing edge it never created: C -> A existed before the merge "
        "and execute_merge skipped remapping it (already exists), but undo removed it anyway"
    )
    assert after[0]["weight"] == 0.8, "C -> A came back with the wrong weight"


def test_undo_falls_back_when_remapped_edges_is_null(engine):
    """Merge history rows written before this change (remapped_edges NULL) still undo
    correctly: undo falls back to deriving the remapped key from original_edges."""
    id_a, _ = _create_node(engine, title="A", content="Longer content for keeping")
    id_b, _ = _create_node(engine, title="B", content="Short content")
    id_c, _ = _create_node(engine, title="C", content="Neighbor node content")

    # Create edge: B -> C (will be remapped to A -> C during merge)
    engine.connect(ConnectRequest(
        source_id=id_b, target_id=id_c, edge=EdgeType.supports, weight=0.8
    ))

    engine.execute_merge(id_a, id_b)

    post_edges = engine.db.conn.execute(
        "SELECT * FROM edges WHERE source_id = ? AND target_id = ? AND edge_type = 'supports'",
        (id_a, id_c),
    ).fetchall()
    assert len(post_edges) >= 1, "sanity: A -> C exists after merge (remapped)"

    merge = engine.list_merges()[0]

    # Simulate a merge_history row written before this change.
    engine.db.conn.execute(
        "UPDATE merge_history SET remapped_edges = NULL WHERE id = ?", (merge["id"],)
    )
    engine.db.conn.commit()

    engine.undo_merge(merge["id"])

    after_undo = engine.db.conn.execute(
        "SELECT * FROM edges WHERE source_id = ? AND target_id = ? AND edge_type = 'supports'",
        (id_a, id_c),
    ).fetchall()
    assert len(after_undo) == 0, "fallback path must still remove the remapped edge"


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


def test_merge_preserves_third_party_incoming_edge(engine):
    """A third node pointing AT the kept node must survive the merge (#123).

    The merge path calls index_single(kept). Before #123 that destroyed every edge pointing
    at kept, and merge_nodes hand-restored them. This test is what makes deleting that
    workaround a proof rather than an assumption: no existing merge test creates D -> kept,
    so the suite could stay green while exactly this edge was lost.
    """
    from ormah.models.node import Connection

    id_a, _ = _create_node(engine, title="Kept", content="This node will be kept because longer")
    id_b, _ = _create_node(engine, title="Removed", content="Shorter content")
    id_d, _ = _create_node(engine, title="Third", content="An unrelated third node")

    # D -> kept, declared in D's own markdown — the path index_single re-reads.
    node_d = engine.file_store.load(id_d)
    node_d.connections.append(
        Connection(target=id_a, edge=EdgeType.supports, weight=0.8)
    )
    engine.file_store.save(node_d)
    engine.builder.index_single(engine.file_store._path_for(node_d))

    def third_party():
        return engine.db.conn.execute(
            "SELECT source_id, target_id, edge_type, weight, created FROM edges "
            "WHERE source_id = ? AND target_id = ?",
            (id_d, id_a),
        ).fetchall()

    before = third_party()
    assert len(before) == 1, "sanity: D -> kept must exist before the merge"

    engine.execute_merge(id_a, id_b)

    after = third_party()
    assert len(after) == 1, "merge destroyed the third-party incoming edge (#123)"
    assert after[0]["edge_type"] == "supports"
    assert after[0]["weight"] == 0.8, "weight 0.8, not the 0.5 default — this is D's declared row"
    assert after[0]["created"] == before[0]["created"], "the row was recreated, not preserved"


def test_merge_retargets_neighbour_markdown_when_the_remap_is_skipped(engine):
    """A neighbour's markdown must be retargeted even when its edge row already exists (#123).

    Since #123, index_single(kept) no longer wipes the kept node's incoming edges. That means
    a neighbour's edge into the kept node can already be present in the DB before the remap
    loop runs, so the "edge already exists" check short-circuits that row and the INSERT is
    skipped. affected_node_ids must still be recorded for that neighbour, or the markdown
    rewrite pass never fires and the neighbour's connections keep pointing at the soft-deleted
    removed node.
    """
    from ormah.models.node import Connection

    original_threshold = engine.settings.auto_link_similarity_threshold
    engine.settings.auto_link_similarity_threshold = 999.0
    try:
        id_a, _ = _create_node(
            engine, title="Kept", content="This node will be kept because it is much longer"
        )
        id_b, _ = _create_node(engine, title="Removed", content="Short")
        id_c, _ = _create_node(engine, title="Third", content="An unrelated third node entirely")
    finally:
        engine.settings.auto_link_similarity_threshold = original_threshold

    # C declares C -> A and C -> B in its own markdown, so index_single(C) creates the A edge
    # in the DB up front — the exact precondition that makes the "already exists" check fire
    # for the remapped B -> A row during the merge.
    node_c = engine.file_store.load(id_c)
    node_c.connections.append(Connection(target=id_a, edge=EdgeType.supports, weight=0.8))
    node_c.connections.append(Connection(target=id_b, edge=EdgeType.supports, weight=0.6))
    engine.file_store.save(node_c)
    engine.builder.index_single(engine.file_store._path_for(node_c))

    engine.execute_merge(id_a, id_b)

    after = engine.file_store.load(id_c)
    targets = [c.target for c in after.connections]
    assert id_b not in targets, "C's markdown still points at the removed node"
    assert id_a in targets, "C's markdown should be retargeted at the kept node"


def test_merge_does_not_duplicate_a_collided_neighbour_connection(engine):
    """A neighbour that already declares both C->kept and C->removed with the same edge type
    must not end up with two connections to kept after the merge (#123).

    Before the fix, execute_merge's neighbour markdown pass blindly retargets every
    connection whose target is `removed`, so C ends up declaring `C -> A` twice with the
    same edge type (once pre-existing, once retargeted from `C -> B`). execute_merge itself
    does not reindex C, so this is invisible until the next incremental_update(): reindexing
    C clears its outgoing edges and reinserts from markdown via INSERT OR REPLACE on
    (source_id, target_id, edge_type) — last one wins, silently overwriting the pre-existing
    0.8-weight edge with the retargeted 0.6-weight one.
    """
    from ormah.models.node import Connection

    original_threshold = engine.settings.auto_link_similarity_threshold
    engine.settings.auto_link_similarity_threshold = 999.0
    try:
        id_a, _ = _create_node(
            engine, title="Kept", content="This node will be kept because it is much longer"
        )
        id_b, _ = _create_node(engine, title="Removed", content="Short")
        id_c, _ = _create_node(engine, title="Third", content="An unrelated third node entirely")
    finally:
        engine.settings.auto_link_similarity_threshold = original_threshold

    # C declares C -> A (supports, 0.8) and C -> B (supports, 0.6) — same edge type, so
    # retargeting the B connection collides with the pre-existing A connection.
    node_c = engine.file_store.load(id_c)
    node_c.connections.append(Connection(target=id_a, edge=EdgeType.supports, weight=0.8))
    node_c.connections.append(Connection(target=id_b, edge=EdgeType.supports, weight=0.6))
    engine.file_store.save(node_c)
    engine.builder.index_single(engine.file_store._path_for(node_c))

    engine.execute_merge(id_a, id_b)

    # This is the step that surfaces the clobber in production: the 60s index updater
    # reindexes any markdown file that changed, including C's rewritten markdown.
    engine.builder.incremental_update()

    after = engine.file_store.load(id_c)
    a_connections = [c for c in after.connections if c.target == id_a]
    assert len(a_connections) == 1, (
        f"C's markdown should have exactly one connection to A, got {len(a_connections)}"
    )

    row = engine.db.conn.execute(
        "SELECT weight FROM edges WHERE source_id = ? AND target_id = ? AND edge_type = ?",
        (id_c, id_a, EdgeType.supports.value),
    ).fetchone()
    assert row is not None, "C -> A supports edge should exist"
    assert row["weight"] == 0.8, (
        f"C -> A supports edge should keep its original weight 0.8, got {row['weight']}"
    )


def test_merge_retargets_a_neighbour_connection_whose_edge_type_does_not_collide(engine):
    """A neighbour connection to `removed` must still be retargeted when its edge type does
    not collide with any connection the neighbour already has pointing at `kept` (#123).

    Pins that the collision-avoidance fix does not over-correct into dropping every
    retarget: C -> B (contradicts) has no matching C -> A (contradicts), so it must survive
    as its own retargeted edge, distinct from the pre-existing C -> A (supports).
    """
    from ormah.models.node import Connection

    original_threshold = engine.settings.auto_link_similarity_threshold
    engine.settings.auto_link_similarity_threshold = 999.0
    try:
        id_a, _ = _create_node(
            engine, title="Kept", content="This node will be kept because it is much longer"
        )
        id_b, _ = _create_node(engine, title="Removed", content="Short")
        id_c, _ = _create_node(engine, title="Third", content="An unrelated third node entirely")
    finally:
        engine.settings.auto_link_similarity_threshold = original_threshold

    # C declares C -> A (supports, 0.8) and C -> B (contradicts, 0.6) — different edge
    # types, so no collision: both must survive after the merge.
    node_c = engine.file_store.load(id_c)
    node_c.connections.append(Connection(target=id_a, edge=EdgeType.supports, weight=0.8))
    node_c.connections.append(Connection(target=id_b, edge=EdgeType.contradicts, weight=0.6))
    engine.file_store.save(node_c)
    engine.builder.index_single(engine.file_store._path_for(node_c))

    engine.execute_merge(id_a, id_b)
    engine.builder.incremental_update()

    after = engine.file_store.load(id_c)
    a_edges = {c.edge for c in after.connections if c.target == id_a}
    assert EdgeType.supports in a_edges, "C -> A supports should still exist"
    assert EdgeType.contradicts in a_edges, "C -> A contradicts should have been retargeted from B"

    supports_row = engine.db.conn.execute(
        "SELECT weight FROM edges WHERE source_id = ? AND target_id = ? AND edge_type = ?",
        (id_c, id_a, EdgeType.supports.value),
    ).fetchone()
    contradicts_row = engine.db.conn.execute(
        "SELECT weight FROM edges WHERE source_id = ? AND target_id = ? AND edge_type = ?",
        (id_c, id_a, EdgeType.contradicts.value),
    ).fetchone()
    assert supports_row is not None and supports_row["weight"] == 0.8
    assert contradicts_row is not None and contradicts_row["weight"] == 0.6


def test_merge_coalesces_duplicate_removed_connections_to_the_last_declaration(engine):
    """When a neighbour declares the SAME edge type toward `removed` twice, the retarget must
    keep the LAST declaration, matching what _index_file_edges makes effective (#123).

    `_index_file_edges` writes edges with INSERT OR REPLACE on
    (source_id, target_id, edge_type), so when a markdown file declares the same
    (target, edge_type) pair twice, the LAST declaration wins at reindex time. C has no
    connection to A at all here — only two declarations of C -> B, same edge type, different
    weights — so the retargeted C -> A connection must carry the weight of the second
    (last) one, not the first.
    """
    from ormah.models.node import Connection

    original_threshold = engine.settings.auto_link_similarity_threshold
    engine.settings.auto_link_similarity_threshold = 999.0
    try:
        id_a, _ = _create_node(
            engine, title="Kept", content="This node will be kept because it is much longer"
        )
        id_b, _ = _create_node(engine, title="Removed", content="Short")
        id_c, _ = _create_node(engine, title="Third", content="An unrelated third node entirely")
    finally:
        engine.settings.auto_link_similarity_threshold = original_threshold

    # C declares C -> B (supports, 0.8) then C -> B (supports, 0.6) — same edge type, no
    # connection to A at all. Last declaration (0.6) is the effective one.
    node_c = engine.file_store.load(id_c)
    node_c.connections.append(Connection(target=id_b, edge=EdgeType.supports, weight=0.8))
    node_c.connections.append(Connection(target=id_b, edge=EdgeType.supports, weight=0.6))
    engine.file_store.save(node_c)
    engine.builder.index_single(engine.file_store._path_for(node_c))

    engine.execute_merge(id_a, id_b)

    # This is the step that surfaces the wrong winner in production: the 60s index updater
    # reindexes any markdown file that changed, including C's rewritten markdown.
    engine.builder.incremental_update()

    after = engine.file_store.load(id_c)
    a_connections = [c for c in after.connections if c.target == id_a]
    assert len(a_connections) == 1, (
        f"C's markdown should have exactly one connection to A, got {len(a_connections)}"
    )
    assert a_connections[0].weight == 0.6, (
        f"C -> A should carry the LAST declaration's weight 0.6, got {a_connections[0].weight}"
    )

    row = engine.db.conn.execute(
        "SELECT weight FROM edges WHERE source_id = ? AND target_id = ? AND edge_type = ?",
        (id_c, id_a, EdgeType.supports.value),
    ).fetchone()
    assert row is not None, "C -> A supports edge should exist"
    assert row["weight"] == 0.6, (
        f"C -> A supports edge should carry the last declaration's weight 0.6, got {row['weight']}"
    )


def test_merge_keeps_the_preexisting_declaration_when_duplicates_collide(engine):
    """When a neighbour already declares the edge type toward `kept`, that pre-existing
    declaration wins over BOTH duplicate declarations toward `removed`, no matter which of
    the duplicates would otherwise be the last one (#123).

    C declares C -> A (supports, 0.9) first, then two C -> B (supports) declarations. The
    pre-existing C -> A connection must survive with its own weight 0.9; both C -> B
    declarations are dropped entirely, not coalesced into a competing retarget.
    """
    from ormah.models.node import Connection

    original_threshold = engine.settings.auto_link_similarity_threshold
    engine.settings.auto_link_similarity_threshold = 999.0
    try:
        id_a, _ = _create_node(
            engine, title="Kept", content="This node will be kept because it is much longer"
        )
        id_b, _ = _create_node(engine, title="Removed", content="Short")
        id_c, _ = _create_node(engine, title="Third", content="An unrelated third node entirely")
    finally:
        engine.settings.auto_link_similarity_threshold = original_threshold

    # C declares C -> A (supports, 0.9), then C -> B (supports, 0.8), then C -> B
    # (supports, 0.6). The pre-existing C -> A declaration must win.
    node_c = engine.file_store.load(id_c)
    node_c.connections.append(Connection(target=id_a, edge=EdgeType.supports, weight=0.9))
    node_c.connections.append(Connection(target=id_b, edge=EdgeType.supports, weight=0.8))
    node_c.connections.append(Connection(target=id_b, edge=EdgeType.supports, weight=0.6))
    engine.file_store.save(node_c)
    engine.builder.index_single(engine.file_store._path_for(node_c))

    engine.execute_merge(id_a, id_b)
    engine.builder.incremental_update()

    after = engine.file_store.load(id_c)
    a_connections = [c for c in after.connections if c.target == id_a]
    assert len(a_connections) == 1, (
        f"C's markdown should have exactly one connection to A, got {len(a_connections)}"
    )
    assert a_connections[0].weight == 0.9, (
        f"C -> A should keep its pre-existing weight 0.9, got {a_connections[0].weight}"
    )

    row = engine.db.conn.execute(
        "SELECT weight FROM edges WHERE source_id = ? AND target_id = ? AND edge_type = ?",
        (id_c, id_a, EdgeType.supports.value),
    ).fetchone()
    assert row is not None, "C -> A supports edge should exist"
    assert row["weight"] == 0.9, (
        f"C -> A supports edge should keep the pre-existing weight 0.9, got {row['weight']}"
    )

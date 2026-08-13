"""Mutation-stamping guarantees (Sync v1 Step 0).

Every content mutation must advance `updated` before save; soft-deletes must
stamp `deleted_at`; read-side access metadata must never advance `updated`.
Sync's last-write-wins merge depends on these invariants.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from unittest.mock import patch

from ormah.models.node import ConnectRequest, CreateNodeRequest, EdgeType, NodeType
from ormah.store.markdown import parse_node

PAST = datetime(2020, 1, 1, tzinfo=timezone.utc)

_LLM_PATCH = "ormah.background.llm_client.llm_generate"


def _create(engine, title, content, **kwargs):
    """Create a node with auto-linking suppressed, return its id."""
    original = engine.settings.auto_link_similarity_threshold
    engine.settings.auto_link_similarity_threshold = 999.0
    try:
        node_id, _ = engine.remember(
            CreateNodeRequest(content=content, type=NodeType.fact, title=title, **kwargs),
            agent_id="test",
        )
    finally:
        engine.settings.auto_link_similarity_threshold = original
    return node_id


def _backdate(file_store, node_id):
    node = file_store.load(node_id)
    node.updated = PAST
    file_store.save(node)


def _updated(file_store, node_id):
    return file_store.load(node_id).updated


def _tombstone(file_store, node_id):
    """Parse the tombstone file for a node from deleted/."""
    deleted_dir = file_store.nodes_dir.parent / "deleted"
    short_id = node_id.split("-")[0]
    matches = list(deleted_dir.glob(f"*_{short_id}.md"))
    assert matches, f"no tombstone found for {node_id}"
    return parse_node(matches[0].read_text(encoding="utf-8"))


def _reset_adapter():
    from ormah.background.llm_client import reset_adapter

    reset_adapter()


# ---------------------------------------------------------------------------
# Store level
# ---------------------------------------------------------------------------


def test_soft_delete_stamps_deleted_at(file_store):
    from ormah.models.node import MemoryNode

    node = MemoryNode(type=NodeType.fact, content="delete me", title="Doomed")
    node.updated = PAST
    file_store.save(node)

    assert file_store.soft_delete(node.id) is True

    tombstone = _tombstone(file_store, node.id)
    assert tombstone.deleted_at is not None
    assert tombstone.deleted_at.tzinfo is not None
    assert tombstone.deleted_at > PAST
    # Deletion ordering uses deleted_at, never updated
    assert tombstone.updated == PAST


def test_soft_delete_unparseable_file_still_moves(file_store):
    from ormah.models.node import MemoryNode

    node = MemoryNode(type=NodeType.fact, content="will be corrupted", title="Corrupt")
    path = file_store.save(node)
    path.write_text("::: not valid frontmatter :::", encoding="utf-8")

    assert file_store.soft_delete(node.id) is True
    deleted_dir = file_store.nodes_dir.parent / "deleted"
    assert (deleted_dir / path.name).exists()


def test_touch_access_does_not_advance_updated(file_store):
    from ormah.models.node import MemoryNode

    node = MemoryNode(type=NodeType.fact, content="access me", title="Accessed")
    node.updated = PAST
    file_store.save(node)

    result = file_store.touch_access(node.id)
    assert result.access_count == 1
    assert _updated(file_store, node.id) == PAST


# ---------------------------------------------------------------------------
# Engine mutations
# ---------------------------------------------------------------------------


def test_connect_advances_source_updated(engine):
    id_a = _create(engine, "Source", "Source node content.")
    id_b = _create(engine, "Target", "Target node content.")
    _backdate(engine.file_store, id_a)

    engine.connect(ConnectRequest(source_id=id_a, target_id=id_b, edge=EdgeType.related_to))

    assert _updated(engine.file_store, id_a) > PAST


def test_update_node_advances_updated(engine):
    from ormah.models.node import UpdateNodeRequest

    node_id = _create(engine, "Editable", "Original content.")
    _backdate(engine.file_store, node_id)

    engine.update_node(node_id, UpdateNodeRequest(content="New content."))

    assert _updated(engine.file_store, node_id) > PAST


def test_mark_outdated_advances_updated(engine):
    node_id = _create(engine, "Stale", "Old truth.")
    _backdate(engine.file_store, node_id)

    engine.mark_outdated(node_id, reason="superseded")

    assert _updated(engine.file_store, node_id) > PAST


def test_confirmed_use_does_not_advance_updated(engine):
    node_id = _create(engine, "Read often", "Frequently accessed fact.")
    _backdate(engine.file_store, node_id)

    engine._record_confirmed_use(node_id)

    node = engine.file_store.load(node_id)
    assert node.updated == PAST
    assert node.access_count == 1
    assert node.last_review is not None


def test_merge_stamps_kept_node_and_tombstone(engine):
    id_short = _create(engine, "Short", "Brief.")
    id_long = _create(engine, "Long", "Much longer content that wins the keep decision.")
    _backdate(engine.file_store, id_short)
    _backdate(engine.file_store, id_long)

    engine.execute_merge(id_short, id_long)

    # Kept node stamped
    assert _updated(engine.file_store, id_long) > PAST
    # Removed node's tombstone carries deleted_at
    tombstone = _tombstone(engine.file_store, id_short)
    assert tombstone.deleted_at is not None
    assert tombstone.deleted_at > PAST


def test_merge_neighbor_remap_advances_updated(engine):
    id_removed = _create(engine, "Removed", "Brief.")
    id_kept = _create(engine, "Kept", "Much longer content that wins the keep decision.")
    id_neighbor = _create(engine, "Neighbor", "Points at the removed node.")
    engine.connect(
        ConnectRequest(source_id=id_neighbor, target_id=id_removed, edge=EdgeType.supports)
    )
    _backdate(engine.file_store, id_neighbor)

    engine.execute_merge(id_removed, id_kept)

    neighbor = engine.file_store.load(id_neighbor)
    assert neighbor.updated > PAST
    assert any(c.target == id_kept for c in neighbor.connections)


def test_undo_merge_restored_node_outranks_tombstone(engine):
    id_removed = _create(engine, "Removed", "Brief.")
    id_kept = _create(engine, "Kept", "Much longer content that wins the keep decision.")
    engine.execute_merge(id_removed, id_kept)
    tombstone = _tombstone(engine.file_store, id_removed)

    merge = engine.list_merges(limit=1)[0]
    engine.undo_merge(merge["id"])

    restored = engine.file_store.load(id_removed)
    assert restored is not None
    assert restored.deleted_at is None
    # The live node must win the node-vs-tombstone rule (deleted_at >= updated)
    assert restored.updated > tombstone.deleted_at


def test_link_to_self_advances_self_node_updated(engine):
    _backdate(engine.file_store, engine.user_node_id)

    _create(engine, "About me", "I prefer dark roast coffee.", about_self=True)

    assert _updated(engine.file_store, engine.user_node_id) > PAST


def test_tier_change_advances_updated(engine):
    from ormah.models.node import Tier

    node_id = _create(engine, "Promotable", "Should move tiers.")
    node = engine.file_store.load(node_id)
    node.updated = PAST

    assert engine.tier_manager.promote(node, Tier.core) is True
    assert node.updated > PAST


# ---------------------------------------------------------------------------
# Background jobs
# ---------------------------------------------------------------------------


def test_importance_scorer_advances_updated(engine):
    node_id = _create(engine, "Scored", "A node whose importance will change.")
    _backdate(engine.file_store, node_id)
    # Force a meaningful importance delta so the scorer rewrites the file
    engine.db.conn.execute("UPDATE nodes SET importance = 0.0 WHERE id = ?", (node_id,))
    engine.db.conn.commit()

    from ormah.background.importance_scorer import run_importance_scoring

    run_importance_scoring(engine)

    node = engine.file_store.load(node_id)
    assert node.importance > 0.0
    assert node.updated > PAST


def test_auto_linker_advances_updated(engine):
    id_a = _create(engine, "Python language", "Python is a programming language.")
    id_b = _create(engine, "Python lang", "Python is a popular programming language.")
    _backdate(engine.file_store, id_a)
    _backdate(engine.file_store, id_b)

    engine.settings.llm_provider = "ollama"
    engine.settings.auto_link_similarity_threshold = 0.0
    _reset_adapter()
    llm_response = json.dumps({"relationship": "supports", "reason": "Same topic."})

    with patch(_LLM_PATCH, return_value=llm_response):
        from ormah.background.auto_linker import run_auto_linker

        run_auto_linker(engine)

    stamped = [
        nid
        for nid in (id_a, id_b)
        if engine.file_store.load(nid).connections and _updated(engine.file_store, nid) > PAST
    ]
    assert stamped, "the node that received the markdown connection must be stamped"


def test_conflict_detector_advances_updated(engine):
    id_a = _create(engine, "Use PostgreSQL", "We decided to use PostgreSQL for the database.")
    id_b = _create(engine, "Use MySQL", "We decided to use MySQL for the database.")
    _backdate(engine.file_store, id_a)
    _backdate(engine.file_store, id_b)

    engine.settings.llm_provider = "ollama"
    _reset_adapter()
    llm_response = json.dumps(
        {"conflict": True, "same_subject": True, "type": "tension", "explanation": "Conflicting choices."}
    )

    with patch(_LLM_PATCH, return_value=llm_response):
        from ormah.background.conflict_detector import run_conflict_detection

        run_conflict_detection(engine)

    stamped = [
        nid
        for nid in (id_a, id_b)
        if engine.file_store.load(nid).connections and _updated(engine.file_store, nid) > PAST
    ]
    assert stamped, "the node that received the conflict edge must be stamped"


def test_auto_cluster_advances_updated(engine):
    id_anchor = _create(engine, "Anchor", "Node with a space.", space="projx")
    id_floating = _create(engine, "Floating", "Node without a space.")
    engine.connect(
        ConnectRequest(source_id=id_anchor, target_id=id_floating, edge=EdgeType.related_to)
    )
    # Clear any space the engine assigned at creation, then backdate
    engine.db.conn.execute("UPDATE nodes SET space = NULL WHERE id = ?", (id_floating,))
    engine.db.conn.commit()
    node = engine.file_store.load(id_floating)
    node.space = None
    node.updated = PAST
    engine.file_store.save(node)

    from ormah.background.auto_cluster import run_auto_cluster

    run_auto_cluster(engine)

    node = engine.file_store.load(id_floating)
    assert node.space == "projx"
    assert node.updated > PAST

# ---------------------------------------------------------------------------
# Review fixes (Codex review of PR #105)
# ---------------------------------------------------------------------------


def test_update_node_noop_does_not_advance_updated(engine):
    from ormah.models.node import UpdateNodeRequest

    node_id = _create(engine, "Stable", "Unchanging content.")
    _backdate(engine.file_store, node_id)

    # Empty request
    engine.update_node(node_id, UpdateNodeRequest())
    assert _updated(engine.file_store, node_id) == PAST

    # Request assigning identical values
    node = engine.file_store.load(node_id)
    engine.update_node(
        node_id, UpdateNodeRequest(content=node.content, title=node.title)
    )
    assert _updated(engine.file_store, node_id) == PAST

    # A real change still stamps
    engine.update_node(node_id, UpdateNodeRequest(content="Actually new content."))
    assert _updated(engine.file_store, node_id) > PAST


def test_identity_tier_migration_stamps_updated(engine):
    from ormah.models.node import Tier

    node_id = _create(engine, "About me", "I prefer tea.", about_self=True)

    # Force the pre-migration state: core tier on disk and in the index
    node = engine.file_store.load(node_id)
    node.tier = Tier.core
    node.updated = PAST
    engine.file_store.save(node)
    engine.db.conn.execute("UPDATE nodes SET tier = 'core' WHERE id = ?", (node_id,))
    engine.db.conn.execute("DELETE FROM meta WHERE key = 'identity_tier_migrated'")
    engine.db.conn.commit()

    engine._migrate_identity_tiers()

    migrated = engine.file_store.load(node_id)
    assert migrated.tier == Tier.working
    assert migrated.updated > PAST


def test_identity_edge_repair_persists_to_markdown(engine):
    """Phase-2 repaired defines edges must live in the self node's markdown so
    they survive a full index rebuild (edges table is derived)."""
    node_id = _create(engine, "Orphaned identity fact", "I am left-handed.")

    # Tag as about_self without a defines edge (the orphan scenario)
    node = engine.file_store.load(node_id)
    node.tags.append("about_self")
    engine.file_store.save(node)
    engine.db.conn.execute(
        "INSERT OR IGNORE INTO node_tags (node_id, tag) VALUES (?, 'about_self')",
        (node_id,),
    )
    engine.db.conn.execute("DELETE FROM meta WHERE key = 'identity_edges_repaired'")
    engine.db.conn.commit()
    _backdate(engine.file_store, engine.user_node_id)

    engine._migrate_identity_tiers()

    # Markdown source of truth carries the edge and the stamp
    self_node = engine.file_store.load(engine.user_node_id)
    assert any(
        c.target == node_id and c.edge == EdgeType.defines
        for c in self_node.connections
    )
    assert self_node.updated > PAST

    # And the edge survives a full rebuild
    engine.builder.full_rebuild()
    row = engine.db.conn.execute(
        "SELECT 1 FROM edges WHERE source_id = ? AND target_id = ? AND edge_type = 'defines'",
        (engine.user_node_id, node_id),
    ).fetchone()
    assert row is not None

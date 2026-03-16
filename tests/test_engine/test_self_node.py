"""Tests for the user self node feature."""

from ormah.models.node import CreateNodeRequest, NodeType, Tier


def test_self_node_created_on_startup(engine):
    """Self node should be created on startup with correct properties."""
    assert engine.user_node_id is not None

    node = engine.file_store.load(engine.user_node_id)
    assert node is not None
    assert node.type == NodeType.person
    assert node.tier == Tier.core
    assert node.source == "system:self"
    assert node.title == "Self"
    assert node.space is None
    assert "self" in node.tags
    assert "identity" in node.tags


def test_self_node_persisted_in_meta(engine):
    """Self node ID should be stored in the meta table."""
    row = engine.db.conn.execute(
        "SELECT value FROM meta WHERE key = 'user_node_id'"
    ).fetchone()
    assert row is not None
    assert row["value"] == engine.user_node_id


def test_self_node_survives_restart(settings):
    """Self node should be reused across engine restarts."""
    from ormah.engine.memory_engine import MemoryEngine

    # First startup
    eng1 = MemoryEngine(settings)
    eng1.startup()
    first_id = eng1.user_node_id
    eng1.shutdown()

    # Second startup — should reuse same node
    eng2 = MemoryEngine(settings)
    eng2.startup()
    assert eng2.user_node_id == first_id
    eng2.shutdown()


def test_self_node_survives_core_cap(engine):
    """Self node should never be demoted when core cap is enforced."""
    # Fill up core beyond cap (default 50)
    for i in range(55):
        req = CreateNodeRequest(
            content=f"Core fact number {i}.",
            type=NodeType.fact,
            tier=Tier.core,
            title=f"Core fact {i}",
        )
        engine.remember(req)

    # Self node should still be core
    self_node = engine.file_store.load(engine.user_node_id)
    assert self_node is not None
    assert self_node.tier == Tier.core


def test_about_self_creates_defines_edge(engine):
    """remember() with about_self=True should create a defines edge from self node."""
    req = CreateNodeRequest(
        content="User's name is Alice.",
        type=NodeType.fact,
        title="User name",
        about_self=True,
    )
    node_id, _ = engine.remember(req)

    # Check defines edge exists in DB
    edge = engine.db.conn.execute(
        "SELECT * FROM edges WHERE source_id = ? AND target_id = ? AND edge_type = 'defines'",
        (engine.user_node_id, node_id),
    ).fetchone()
    assert edge is not None
    assert edge["weight"] == 1.0


def test_about_self_false_no_defines_edge(engine):
    """remember() without about_self should NOT create a defines edge."""
    req = CreateNodeRequest(
        content="Python is a programming language.",
        type=NodeType.fact,
        title="Python",
    )
    node_id, _ = engine.remember(req)

    edge = engine.db.conn.execute(
        "SELECT * FROM edges WHERE source_id = ? AND target_id = ? AND edge_type = 'defines'",
        (engine.user_node_id, node_id),
    ).fetchone()
    assert edge is None


def test_about_self_preference_stays_working(engine):
    """Preference with about_self=True stays working tier (consolidator handles dedup)."""
    req = CreateNodeRequest(
        content="User prefers dark mode.",
        type=NodeType.preference,
        title="Dark mode preference",
        about_self=True,
    )
    node_id, _ = engine.remember(req)

    node = engine.file_store.load(node_id)
    assert node is not None
    assert node.tier == Tier.working


def test_about_self_promotes_person_to_core(engine):
    """Person type with about_self=True should be promoted to core tier."""
    req = CreateNodeRequest(
        content="User is a software engineer.",
        type=NodeType.person,
        title="User profession",
        about_self=True,
    )
    node_id, _ = engine.remember(req)

    node = engine.file_store.load(node_id)
    assert node is not None
    assert node.tier == Tier.core


def test_about_self_fact_stays_working(engine):
    """Fact with about_self=True stays working tier (only preference/person promoted)."""
    req = CreateNodeRequest(
        content="User lives in London.",
        type=NodeType.fact,
        title="Location",
        about_self=True,
    )
    node_id, _ = engine.remember(req)

    node = engine.file_store.load(node_id)
    assert node is not None
    assert node.tier == Tier.working


def test_self_node_skipped_by_decay(engine):
    """Decay manager should skip the self node."""
    from ormah.background.decay_manager import run_decay

    # Artificially make self node look stale
    engine.db.conn.execute(
        "UPDATE nodes SET last_accessed = '2020-01-01T00:00:00+00:00', tier = 'working' WHERE id = ?",
        (engine.user_node_id,),
    )
    engine.db.conn.commit()

    run_decay(engine)

    # Self node should have no decay proposal
    proposals = engine.db.conn.execute(
        "SELECT * FROM proposals WHERE source_nodes LIKE ?",
        (f"%{engine.user_node_id}%",),
    ).fetchall()
    assert len(proposals) == 0


def test_get_self_empty(engine):
    """get_self with no identity nodes returns appropriate message."""
    text = engine.get_self()
    assert "No user identity" in text


def test_get_self_with_identity(engine):
    """get_self returns formatted identity nodes."""
    req = CreateNodeRequest(
        content="User's name is Alice.",
        type=NodeType.fact,
        title="User name",
        about_self=True,
    )
    engine.remember(req)

    text = engine.get_self()
    assert "About the User" in text
    assert "User name" in text


def test_context_includes_identity_section(engine):
    """get_context should include identity section at top."""
    req = CreateNodeRequest(
        content="User prefers dark mode.",
        type=NodeType.preference,
        title="Dark mode preference",
        about_self=True,
    )
    engine.remember(req)

    text = engine.get_context()
    assert "About the User" in text
    # Identity section should come before core memories
    identity_pos = text.find("About the User")
    core_pos = text.find("Core Memories")
    if core_pos >= 0:
        assert identity_pos < core_pos

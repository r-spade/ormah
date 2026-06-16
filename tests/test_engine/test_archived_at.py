from __future__ import annotations

from ormah.models.node import CreateNodeRequest, NodeType, Tier, UpdateNodeRequest


def _archived_at(engine, node_id):
    row = engine.db.conn.execute(
        "SELECT archived_at FROM nodes WHERE id = ?", (node_id,)
    ).fetchone()
    return row["archived_at"]


def test_demotion_to_archival_stamps_archived_at(engine):
    node_id, _ = engine.remember(CreateNodeRequest(
        content="demote me", type=NodeType.fact, tier=Tier.working, title="d"))
    assert _archived_at(engine, node_id) is None

    engine.update_node(node_id, UpdateNodeRequest(tier=Tier.archival))

    assert _archived_at(engine, node_id) is not None
    assert engine.file_store.load(node_id).archived_at is not None  # source of truth


def test_non_archival_update_does_not_stamp(engine):
    node_id, _ = engine.remember(CreateNodeRequest(
        content="rename me", type=NodeType.fact, tier=Tier.working, title="r"))
    engine.update_node(node_id, UpdateNodeRequest(title="renamed"))
    assert _archived_at(engine, node_id) is None


def test_metadata_edit_while_archival_keeps_archived_at(engine):
    """A metadata edit (no tier change) must not move the clock."""
    node_id, _ = engine.remember(CreateNodeRequest(
        content="x", type=NodeType.fact, tier=Tier.working, title="x"))
    engine.update_node(node_id, UpdateNodeRequest(tier=Tier.archival))
    first = _archived_at(engine, node_id)
    engine.update_node(node_id, UpdateNodeRequest(title="x2"))  # no tier change
    assert _archived_at(engine, node_id) == first


def test_leaving_archival_clears_archived_at(engine):
    node_id, _ = engine.remember(CreateNodeRequest(
        content="y", type=NodeType.fact, tier=Tier.working, title="y"))
    engine.update_node(node_id, UpdateNodeRequest(tier=Tier.archival))
    assert _archived_at(engine, node_id) is not None
    engine.update_node(node_id, UpdateNodeRequest(tier=Tier.working))  # promoted out
    assert _archived_at(engine, node_id) is None


def test_re_entering_archival_restamps_fresh(engine):
    """archival → working → archival must reset the clock, not keep the old one."""
    node_id, _ = engine.remember(CreateNodeRequest(
        content="z", type=NodeType.fact, tier=Tier.working, title="z"))
    engine.update_node(node_id, UpdateNodeRequest(tier=Tier.archival))
    engine.update_node(node_id, UpdateNodeRequest(tier=Tier.working))   # clears
    engine.update_node(node_id, UpdateNodeRequest(tier=Tier.archival))  # re-stamps
    assert _archived_at(engine, node_id) is not None

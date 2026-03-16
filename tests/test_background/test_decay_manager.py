"""Tests for the decay manager background job."""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

import pytest

from ormah.background.decay_manager import run_decay
from ormah.models.node import CreateNodeRequest, NodeType, Tier


def _make_stale(engine, node_id: str, days: int = 30) -> None:
    """Set a node's last_accessed to `days` ago."""
    stale_date = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
    engine.db.conn.execute(
        "UPDATE nodes SET last_accessed = ? WHERE id = ?", (stale_date, node_id)
    )
    engine.db.conn.commit()


def _get_tier(engine, node_id: str) -> str:
    row = engine.db.conn.execute(
        "SELECT tier FROM nodes WHERE id = ?", (node_id,)
    ).fetchone()
    return row["tier"] if row else None


def test_high_importance_node_not_decayed(engine):
    """A stale node with high importance should not be demoted."""
    node_id, _ = engine.remember(CreateNodeRequest(
        content="Important stale node",
        type=NodeType.fact,
        tier=Tier.working,
        title="Important",
    ))

    _make_stale(engine, node_id)
    engine.db.conn.execute(
        "UPDATE nodes SET importance = 0.9 WHERE id = ?", (node_id,)
    )
    engine.db.conn.commit()

    run_decay(engine)

    assert _get_tier(engine, node_id) == "working"


def test_low_importance_stale_node_decayed(engine):
    """A stale node with low importance should be demoted to archival."""
    node_id, _ = engine.remember(CreateNodeRequest(
        content="Unimportant stale node",
        type=NodeType.fact,
        tier=Tier.working,
        title="Unimportant",
    ))

    _make_stale(engine, node_id)
    engine.db.conn.execute(
        "UPDATE nodes SET importance = 0.2 WHERE id = ?", (node_id,)
    )
    engine.db.conn.commit()

    run_decay(engine)

    assert _get_tier(engine, node_id) == "archival"


def test_decay_still_works_without_importance(engine):
    """Low importance (0.3) + stale should trigger decay (0.3 < 0.5 threshold)."""
    node_id, _ = engine.remember(CreateNodeRequest(
        content="Default importance stale node",
        type=NodeType.fact,
        tier=Tier.working,
        title="Default",
    ))

    _make_stale(engine, node_id)
    engine.db.conn.execute(
        "UPDATE nodes SET importance = 0.3 WHERE id = ?", (node_id,)
    )
    engine.db.conn.commit()

    run_decay(engine)

    assert _get_tier(engine, node_id) == "archival"


def test_decay_is_idempotent(engine):
    """Running decay twice should not error; node stays archival after both runs."""
    node_id, _ = engine.remember(CreateNodeRequest(
        content="Node that will go stale",
        type=NodeType.fact,
        tier=Tier.working,
        title="Stale node",
    ))

    _make_stale(engine, node_id)
    engine.db.conn.execute(
        "UPDATE nodes SET importance = 0.2 WHERE id = ?", (node_id,)
    )
    engine.db.conn.commit()

    run_decay(engine)
    assert _get_tier(engine, node_id) == "archival"

    # Second run should not error
    run_decay(engine)
    assert _get_tier(engine, node_id) == "archival"


def test_decay_writes_audit_log(engine):
    """Demoted nodes should have an audit log entry recording the tier change."""
    node_id, _ = engine.remember(CreateNodeRequest(
        content="Node with known importance",
        type=NodeType.fact,
        tier=Tier.working,
        title="Audit test",
    ))

    _make_stale(engine, node_id)
    engine.db.conn.execute(
        "UPDATE nodes SET importance = 0.35 WHERE id = ?", (node_id,)
    )
    engine.db.conn.commit()

    run_decay(engine)

    row = engine.db.conn.execute(
        "SELECT detail FROM audit_log WHERE node_id = ? AND operation = 'update' "
        "ORDER BY performed_at DESC LIMIT 1",
        (node_id,),
    ).fetchone()
    assert row is not None
    detail = json.loads(row["detail"])
    assert "tier" in detail["changed_fields"]


def test_decay_cleans_pending_proposals(engine):
    """Legacy pending decay proposals should be cleaned up on run."""
    # Insert a fake legacy decay proposal
    engine.db.conn.execute(
        "INSERT INTO proposals (id, type, status, source_nodes, proposed_action, reason, created) "
        "VALUES ('legacy-1', 'decay', 'pending', '[\"fake-id\"]', 'Demote to archival: test', 'test', ?)",
        (datetime.now(timezone.utc).isoformat(),),
    )
    engine.db.conn.commit()

    count_before = engine.db.conn.execute(
        "SELECT COUNT(*) FROM proposals WHERE type = 'decay' AND status = 'pending'"
    ).fetchone()[0]
    assert count_before == 1

    run_decay(engine)

    count_after = engine.db.conn.execute(
        "SELECT COUNT(*) FROM proposals WHERE type = 'decay' AND status = 'pending'"
    ).fetchone()[0]
    assert count_after == 0

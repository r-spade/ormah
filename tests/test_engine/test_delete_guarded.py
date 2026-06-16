from __future__ import annotations

from datetime import datetime, timezone

from ormah.models.node import CreateNodeRequest, NodeType, Tier


def _archival(engine):
    node_id, _ = engine.remember(CreateNodeRequest(
        content="g", type=NodeType.fact, tier=Tier.archival, title="g"))
    return node_id


def _exists(engine, node_id):
    return engine.db.conn.execute(
        "SELECT 1 FROM nodes WHERE id=?", (node_id,)).fetchone() is not None


def test_guard_false_aborts_deletion(engine):
    node_id = _archival(engine)
    res = engine.delete_node_guarded(node_id, lambda conn: False)
    assert res is None
    assert _exists(engine, node_id) is True


def test_guard_true_deletes(engine):
    node_id = _archival(engine)
    res = engine.delete_node_guarded(node_id, lambda conn: True)
    assert res is not None and res.startswith("Deleted")
    assert _exists(engine, node_id) is False


def test_guard_observes_writes_in_same_transaction(engine):
    """A +feedback row inserted inside the guard's txn is visible to the guard's recheck."""
    node_id = _archival(engine)

    def guard(conn):
        conn.execute(
            "INSERT INTO affinity (prompt_vec, node_id, signal, source, confirmed_at, session_id) "
            "VALUES (?, ?, 1, 'explicit', ?, 's1')",
            (b"\x00", node_id, datetime.now(timezone.utc).isoformat()))
        row = conn.execute(
            "SELECT 1 FROM affinity WHERE node_id=? AND signal>0 LIMIT 1", (node_id,)
        ).fetchone()
        return row is None  # protected (has feedback) → guard returns False → abort

    res = engine.delete_node_guarded(node_id, guard)
    assert res is None
    assert _exists(engine, node_id) is True


def test_guard_never_deletes_user_node(engine):
    res = engine.delete_node_guarded(engine.user_node_id, lambda conn: True)
    assert res is None

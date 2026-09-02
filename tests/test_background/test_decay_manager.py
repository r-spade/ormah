"""Tests for the decay manager background job."""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

import pytest

from ormah.background.decay_manager import run_decay
from ormah.background.importance_scorer import run_importance_scoring
from ormah.models.node import ConnectRequest, CreateNodeRequest, EdgeType, NodeType, Tier


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


def test_high_importance_stale_node_is_decayed(engine):
    """#222: importance is no longer a pre-gate — a stale node decays regardless.

    Before #222 a node with importance >= decay_importance_threshold (0.5) could
    never leave working, however stale it became.
    """
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

    assert _get_tier(engine, node_id) == "archival"


def test_invalid_timestamp_skips_one_node_without_aborting_decay(engine, caplog):
    """A malformed row must not prevent later valid nodes from decaying.

    Removing the importance pre-gate exposes high-importance rows to timestamp
    arithmetic. A quoted/imported timestamp without timezone information parses
    successfully but cannot be subtracted from timezone-aware ``now``. Keep that
    failure scoped to the affected node instead of aborting the whole batch.
    """
    poisoned_id, _ = engine.remember(CreateNodeRequest(
        content="High-importance node with an invalid recency anchor",
        type=NodeType.fact,
        tier=Tier.working,
        title="Invalid timestamp",
    ))
    healthy_id, _ = engine.remember(CreateNodeRequest(
        content="Valid stale node that must still be processed",
        type=NodeType.fact,
        tier=Tier.working,
        title="Valid timestamp",
    ))

    aware_old = (datetime.now(timezone.utc) - timedelta(days=30)).isoformat()
    naive_old = (datetime.now() - timedelta(days=30)).isoformat()
    engine.db.conn.execute(
        "UPDATE nodes SET importance = 0.9, stability = 1.0, "
        "last_review = NULL, last_accessed = ? WHERE id = ?",
        (naive_old, poisoned_id),
    )
    engine.db.conn.execute(
        "UPDATE nodes SET importance = 0.1, stability = 1.0, "
        "last_review = ?, last_accessed = ? WHERE id = ?",
        (aware_old, aware_old, healthy_id),
    )
    engine.db.conn.commit()

    run_decay(engine)

    assert _get_tier(engine, poisoned_id) == "working"
    assert _get_tier(engine, healthy_id) == "archival"
    assert "skipped node" in caplog.text
    assert "Decay manager failed" not in caplog.text


def test_decay_retrievability_respects_stability(engine):
    """At the same age, low stability decays while high stability survives."""
    low_id, _ = engine.remember(CreateNodeRequest(
        content="Thirty-day-old low-stability memory",
        type=NodeType.fact,
        tier=Tier.working,
        title="Low stability",
    ))
    high_id, _ = engine.remember(CreateNodeRequest(
        content="Thirty-day-old high-stability memory",
        type=NodeType.fact,
        tier=Tier.working,
        title="High stability",
    ))

    old = (datetime.now(timezone.utc) - timedelta(days=30)).isoformat()
    engine.db.conn.execute(
        "UPDATE nodes SET stability = 1.0, last_review = NULL, last_accessed = ? "
        "WHERE id = ?",
        (old, low_id),
    )
    engine.db.conn.execute(
        "UPDATE nodes SET stability = 100.0, last_review = NULL, last_accessed = ? "
        "WHERE id = ?",
        (old, high_id),
    )
    engine.db.conn.commit()

    run_decay(engine)

    assert _get_tier(engine, low_id) == "archival"
    assert _get_tier(engine, high_id) == "working"


def test_accumulated_access_and_edges_do_not_pin_a_node_to_working(engine):
    """The reported case, driven end-to-end: 50 accesses + 4 edges on a stale
    hub node make the real importance_scorer compute importance ~= 0.5892
    (measured), above the old decay_importance_threshold gate of 0.5, and the
    node must still decay once run_decay sees it — proving retrievability
    alone, not importance, now controls the working->archival demotion.

    `engine.remember()`/`engine.connect()` never touch `last_review`, so it
    stays None on this node; `run_importance_scoring`'s recency anchor is
    `last_review or last_accessed`, so making the node stale via
    `_make_stale` (which only rewrites `last_accessed`) is enough for the
    scorer to see a genuinely 30-day-old node.
    """
    node_id, _ = engine.remember(CreateNodeRequest(
        content="Hub node with long access history",
        type=NodeType.concept,
        tier=Tier.working,
        title="Hub",
    ))

    for i in range(4):
        sat_id, _ = engine.remember(CreateNodeRequest(
            content=f"Satellite of the hub number {i}",
            type=NodeType.fact,
            tier=Tier.working,
        ))
        engine.connect(ConnectRequest(
            source_id=node_id,
            target_id=sat_id,
            edge=EdgeType.related_to,
        ))

    engine.db.conn.execute(
        "UPDATE nodes SET access_count = 50 WHERE id = ?", (node_id,)
    )
    engine.db.conn.commit()
    _make_stale(engine, node_id)

    run_importance_scoring(engine)

    row = engine.db.conn.execute(
        "SELECT importance FROM nodes WHERE id = ?", (node_id,)
    ).fetchone()
    computed_importance = row["importance"]
    assert computed_importance >= 0.5, (
        f"expected the reported profile (50 accesses + 4 edges, stale) to "
        f"score >= the old decay_importance_threshold of 0.5, got "
        f"{computed_importance}"
    )

    run_decay(engine)

    assert _get_tier(engine, node_id) == "archival"


def test_fresh_high_importance_node_stays_working(engine):
    """The negative case (council I2): retrievability still decides.

    Removing the importance gate must not turn decay into "demote everything".
    A fresh node has R ~= 1.0 and stays working whatever its importance. Without
    this test, deleting the retrievability check would leave the suite green.
    """
    node_id, _ = engine.remember(CreateNodeRequest(
        content="Fresh node that must not decay",
        type=NodeType.fact,
        tier=Tier.working,
        title="Fresh",
    ))

    # Deliberately NOT made stale.
    engine.db.conn.execute(
        "UPDATE nodes SET importance = 0.9 WHERE id = ?", (node_id,)
    )
    engine.db.conn.commit()

    run_decay(engine)

    assert _get_tier(engine, node_id) == "working"


def test_fresh_low_importance_node_stays_working(engine):
    """Same guard from the other side: low importance alone never demotes."""
    node_id, _ = engine.remember(CreateNodeRequest(
        content="Fresh unimportant node that must not decay",
        type=NodeType.fact,
        tier=Tier.working,
        title="Fresh unimportant",
    ))

    engine.db.conn.execute(
        "UPDATE nodes SET importance = 0.05 WHERE id = ?", (node_id,)
    )
    engine.db.conn.commit()

    run_decay(engine)

    assert _get_tier(engine, node_id) == "working"


def test_self_node_is_never_decayed(engine):
    """Identity protection survives the removal of the importance gate."""
    user_node_id = getattr(engine, "user_node_id", None)
    # Fail closed, not skip (council I2): MemoryEngine.startup() calls
    # _ensure_self_node(), which creates the self node if absent, so a missing
    # one means the fixture broke — silently skipping would hide that.
    assert user_node_id is not None, "engine fixture must provide a self node"

    _make_stale(engine, user_node_id)
    engine.db.conn.execute(
        "UPDATE nodes SET tier = 'working', importance = 0.1 WHERE id = ?",
        (user_node_id,),
    )
    engine.db.conn.commit()

    run_decay(engine)

    assert _get_tier(engine, user_node_id) == "working"


def test_low_importance_stale_node_decayed(engine):
    """Low importance is deliberately irrelevant to decay now: pairs with
    test_high_importance_stale_node_is_decayed (importance=0.9) to show a
    stale node decays the same way at either end of the importance range."""
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
    """Decay does not require importance to be set at all: a stale node with
    its importance left untouched still decays on retrievability alone."""
    node_id, _ = engine.remember(CreateNodeRequest(
        content="Default importance stale node",
        type=NodeType.fact,
        tier=Tier.working,
        title="Default",
    ))

    _make_stale(engine, node_id)

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


def _make_decayable(engine, node_id: str) -> None:
    """Lower importance under decay_importance_threshold.

    Both the node default and the threshold are 0.5, and the gate is `>=`, so a
    node left at its default is skipped before retrievability is ever computed.
    """
    engine.db.conn.execute(
        "UPDATE nodes SET importance = 0.2 WHERE id = ?", (node_id,)
    )
    engine.db.conn.commit()


def test_decay_uses_the_shared_retrievability_implementation(engine, monkeypatch):
    """AC5: one exponential curve, shared with the reinforcement path."""
    from ormah.background import decay_manager

    node_id, _ = engine.remember(CreateNodeRequest(
        content="Node whose retrievability we intercept",
        type=NodeType.fact,
        tier=Tier.working,
        title="Intercepted",
    ))
    _make_stale(engine, node_id)
    _make_decayable(engine, node_id)

    calls = []
    real = decay_manager.lifecycle.retrievability

    def _spy(days_since, stability, **kwargs):
        calls.append((days_since, stability))
        return real(days_since, stability, **kwargs)

    monkeypatch.setattr(decay_manager.lifecycle, "retrievability", _spy)
    run_decay(engine)

    assert calls, "run_decay computed retrievability without the shared helper"
    days_since, _stability = calls[0]
    assert days_since == pytest.approx(30, abs=1)


def test_a_node_used_today_is_not_decayed_while_its_review_lags(engine):
    """The cooldown can leave last_review a day behind; use must win the anchor.

    With the old `last_review or last_accessed` order this node reads as 30 days
    stale and is demoted. Step 2 pins that: the test must FAIL before the flip.
    """
    node_id, _ = engine.remember(CreateNodeRequest(
        content="Used today, reviewed a month ago",
        type=NodeType.fact,
        tier=Tier.working,
        title="Fresh use, stale review",
    ))
    now = datetime.now(timezone.utc)
    engine.db.conn.execute(
        "UPDATE nodes SET last_accessed = ?, last_review = ?, stability = 1.0 WHERE id = ?",
        (now.isoformat(), (now - timedelta(days=30)).isoformat(), node_id),
    )
    _make_decayable(engine, node_id)

    run_decay(engine)

    assert _get_tier(engine, node_id) == "working"


def test_zero_stability_decays_on_the_configured_initial_stability(engine, monkeypatch):
    """Decay must use the same zero fallback reinforcement uses (council round 3, I3).

    Node.stability is Field(ge=0.0), so 0 is a real state. With the hardcoded 1.0
    fallback a seven-day-old zero-stability node reads R = exp(-7/1) ~= 0.0009 and
    is archived; with fsrs_initial_stability = 30 it reads exp(-7/30) ~= 0.79 and
    must survive. The 0.3 decay threshold sits between the two, so this test can
    only pass on the shared fallback.
    """
    monkeypatch.setattr(engine.settings, "fsrs_initial_stability", 30.0)

    node_id, _ = engine.remember(CreateNodeRequest(
        content="Zero stability, used a week ago",
        type=NodeType.fact,
        tier=Tier.working,
        title="Zero stability",
    ))
    now = datetime.now(timezone.utc)
    engine.db.conn.execute(
        "UPDATE nodes SET stability = 0.0, last_accessed = ?, last_review = NULL WHERE id = ?",
        ((now - timedelta(days=7)).isoformat(), node_id),
    )
    engine.db.conn.commit()
    _make_decayable(engine, node_id)

    run_decay(engine)

    assert _get_tier(engine, node_id) == "working"


def test_decay_takes_the_lock_once_per_demoted_node_not_once_per_run(engine):
    """The default-install bug: no LLM anywhere, yet L_mem is held for the whole run."""
    from tests.test_background.lock_probe import install_probe

    ids = []
    for i in range(3):
        nid, _ = engine.remember(CreateNodeRequest(
            content=f"stale node {i}", type=NodeType.fact, tier=Tier.working,
            title=f"stale {i}"))
        _make_stale(engine, nid)
        ids.append(nid)

    probe = install_probe(engine)
    run_decay(engine)

    assert all(_get_tier(engine, nid) == "archival" for nid in ids)
    # Before the fix: exactly 1, whatever the node count. After: one per demotion.
    assert probe.acquisitions >= 3


def test_decay_does_not_demote_a_node_promoted_after_the_snapshot(engine):
    """#257's canary, written here: revalidate tier inside the apply step.

    Decay snapshots the node as a stale 'working' candidate. Between that snapshot
    and the locked apply step, a foreground promotion refreshes it. Without
    per-apply-step revalidation the demotion would land anyway and silently undo
    the promotion.

    The hook point matters: it must land the promotion between the unlocked outer
    scan and the locked `_still_decays` re-check — not between `_still_decays` and
    `update_node`, which are both inside the same `memory_operation_at(epoch)` and
    so can never observe an interleaved write (every engine mutator takes the same
    lock). `lifecycle.retrievability` runs in the outer scan, once per candidate
    row, which makes it the right seam: patching it fires exactly once, before
    that node's locked apply step, and does not touch `_still_decays` or
    `update_node` — the code under test.
    """
    from ormah import lifecycle

    node_id, _ = engine.remember(CreateNodeRequest(
        content="about to be promoted", type=NodeType.fact, tier=Tier.working,
        title="promoted"))
    _make_stale(engine, node_id)

    real_retrievability = lifecycle.retrievability
    promoted = {"done": False}

    def promote_then_compute(days_since, stability, **kwargs):
        """Stand in for a concurrent foreground promotion landing right after the
        outer scan snapshots this node as a stale candidate."""
        if not promoted["done"]:
            promoted["done"] = True
            fresh = datetime.now(timezone.utc).isoformat()
            engine.db.conn.execute(
                "UPDATE nodes SET last_accessed = ?, last_review = ?, tier = 'working' "
                "WHERE id = ?", (fresh, fresh, node_id))
            engine.db.conn.commit()
        return real_retrievability(days_since, stability, **kwargs)

    lifecycle.retrievability = promote_then_compute
    try:
        run_decay(engine)
    finally:
        lifecycle.retrievability = real_retrievability

    assert _get_tier(engine, node_id) == "working"


def test_decay_aborts_the_run_when_a_restore_lands_mid_run(engine):
    """Abort, do not skip: the whole snapshot is stale, and nothing may be written."""
    ids = []
    for i in range(3):
        nid, _ = engine.remember(CreateNodeRequest(
            content=f"stale {i}", type=NodeType.fact, tier=Tier.working, title=f"s{i}"))
        _make_stale(engine, nid)
        ids.append(nid)

    real_update = engine.update_node
    demotions = {"count": 0}

    def bump_after_first(nid, req, *args, **kwargs):
        result = real_update(nid, req, *args, **kwargs)
        demotions["count"] += 1
        if demotions["count"] == 1:
            engine._restore_epoch += 1
        return result

    engine.update_node = bump_after_first
    run_decay(engine)  # returns cleanly, does not raise

    assert demotions["count"] == 1
    assert sum(_get_tier(engine, nid) == "archival" for nid in ids) == 1

"""Tests for the importance scoring background job."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from ormah.background.importance_scorer import run_importance_scoring
from ormah.models.node import ConnectRequest, CreateNodeRequest, EdgeType, NodeType, Tier


@pytest.fixture
def scored_engine(engine):
    """Engine with a few nodes of varying profiles."""
    # Core node with many accesses
    req_core = CreateNodeRequest(
        content="Important core fact",
        type=NodeType.fact,
        tier=Tier.core,
        title="Core fact",
    )
    engine.remember(req_core)

    # Working node with few accesses
    req_work = CreateNodeRequest(
        content="Working memory",
        type=NodeType.fact,
        tier=Tier.working,
        title="Working fact",
    )
    engine.remember(req_work)

    # Archival node
    req_arch = CreateNodeRequest(
        content="Old archival memory",
        type=NodeType.fact,
        tier=Tier.archival,
        title="Archival fact",
    )
    engine.remember(req_arch)

    return engine


def test_basic_scoring_runs(scored_engine):
    """Importance scorer should run without errors."""
    run_importance_scoring(scored_engine)


def test_all_new_nodes_get_recency_signal(scored_engine):
    """All freshly created nodes should have non-zero importance from recency."""
    run_importance_scoring(scored_engine)

    rows = scored_engine.db.conn.execute(
        "SELECT importance FROM nodes WHERE source != 'system:self'"
    ).fetchall()

    for r in rows:
        assert r["importance"] > 0.0, "Fresh nodes should have non-zero importance"


def test_empty_graph(engine):
    """Scorer should handle empty graph gracefully (only self node exists)."""
    run_importance_scoring(engine)
    # No crash is the assertion


def test_markdown_updated(scored_engine):
    """Importance values should be persisted to markdown files."""
    # Set a node's importance to something very different so it gets updated
    row = scored_engine.db.conn.execute(
        "SELECT id FROM nodes WHERE source != 'system:self' LIMIT 1"
    ).fetchone()
    if row is None:
        pytest.skip("No non-self nodes")

    node_id = row["id"]
    scored_engine.db.conn.execute(
        "UPDATE nodes SET importance = 0.0 WHERE id = ?", (node_id,)
    )
    scored_engine.db.conn.commit()

    node = scored_engine.file_store.load(node_id)
    if node:
        node.importance = 0.0
        scored_engine.file_store.save(node)

    run_importance_scoring(scored_engine)

    node_after = scored_engine.file_store.load(node_id)
    assert node_after is not None
    # It should have been updated from 0.0 to something > 0
    assert node_after.importance > 0.0


def test_edge_centrality_boosts_importance(engine):
    """A hub node with edges should score higher than an isolated node."""
    hub_id, _ = engine.remember(CreateNodeRequest(
        content="Hub node with many connections",
        type=NodeType.concept,
        tier=Tier.working,
        title="Hub",
    ))

    isolated_id, _ = engine.remember(CreateNodeRequest(
        content="Isolated node no connections",
        type=NodeType.concept,
        tier=Tier.working,
        title="Isolated",
    ))

    # Create 5 satellite nodes connected to the hub
    for i in range(5):
        sat_id, _ = engine.remember(CreateNodeRequest(
            content=f"Satellite node {i}",
            type=NodeType.fact,
            tier=Tier.working,
        ))
        engine.connect(ConnectRequest(
            source_id=hub_id,
            target_id=sat_id,
            edge=EdgeType.related_to,
        ))

    run_importance_scoring(engine)

    hub_imp = engine.db.conn.execute(
        "SELECT importance FROM nodes WHERE id = ?", (hub_id,)
    ).fetchone()["importance"]
    iso_imp = engine.db.conn.execute(
        "SELECT importance FROM nodes WHERE id = ?", (isolated_id,)
    ).fetchone()["importance"]

    assert hub_imp > iso_imp


def test_adding_outlier_does_not_shift_existing_scores(engine):
    """Absolute normalization: adding a high-access outlier shouldn't shift existing scores."""
    node_id, _ = engine.remember(CreateNodeRequest(
        content="Normal node with moderate access",
        type=NodeType.fact,
        tier=Tier.working,
        title="Normal",
    ))

    # Give it moderate access
    engine.db.conn.execute(
        "UPDATE nodes SET access_count = 10 WHERE id = ?", (node_id,)
    )
    engine.db.conn.commit()

    run_importance_scoring(engine)
    score_before = engine.db.conn.execute(
        "SELECT importance FROM nodes WHERE id = ?", (node_id,)
    ).fetchone()["importance"]

    # Now add an outlier with 10000 accesses
    outlier_id, _ = engine.remember(CreateNodeRequest(
        content="Outlier node with massive access",
        type=NodeType.fact,
        tier=Tier.working,
    ))
    engine.db.conn.execute(
        "UPDATE nodes SET access_count = 10000 WHERE id = ?", (outlier_id,)
    )
    engine.db.conn.commit()

    run_importance_scoring(engine)
    score_after = engine.db.conn.execute(
        "SELECT importance FROM nodes WHERE id = ?", (node_id,)
    ).fetchone()["importance"]

    # Score should be stable within 0.1 delta (edge counts may shift slightly
    # due to auto-linking, but access normalization is absolute)
    assert abs(score_after - score_before) < 0.1


def test_recency_decay(engine):
    """Recently accessed node should score higher than an old one."""
    recent_id, _ = engine.remember(CreateNodeRequest(
        content="Recently accessed node",
        type=NodeType.fact,
        tier=Tier.working,
        title="Recent",
    ))

    old_id, _ = engine.remember(CreateNodeRequest(
        content="Old node not accessed in a while",
        type=NodeType.fact,
        tier=Tier.working,
        title="Old",
    ))

    # Make the old node's last_accessed 30 days ago
    old_date = (datetime.now(timezone.utc) - timedelta(days=30)).isoformat()
    engine.db.conn.execute(
        "UPDATE nodes SET last_accessed = ? WHERE id = ?", (old_date, old_id)
    )
    engine.db.conn.commit()

    run_importance_scoring(engine)

    recent_imp = engine.db.conn.execute(
        "SELECT importance FROM nodes WHERE id = ?", (recent_id,)
    ).fetchone()["importance"]
    old_imp = engine.db.conn.execute(
        "SELECT importance FROM nodes WHERE id = ?", (old_id,)
    ).fetchone()["importance"]

    assert recent_imp > old_imp


def test_batch_query_used(engine):
    """Edge count should be fetched with batch queries, not N per-node queries."""
    # Create a few nodes
    for i in range(5):
        engine.remember(CreateNodeRequest(
            content=f"Node {i}",
            type=NodeType.fact,
            tier=Tier.working,
        ))

    # Patch the thread-local connection with a tracking wrapper to count edge queries.
    db = engine.db
    original_conn = db.conn  # ensures the thread-local conn is initialised

    query_log = []

    class TrackingConnection:
        """Wrapper that logs queries while delegating to real connection."""
        def __init__(self, real_conn):
            self._real = real_conn

        def execute(self, sql, *args, **kwargs):
            if "edges" in str(sql).lower():
                query_log.append(str(sql))
            return self._real.execute(sql, *args, **kwargs)

        def __getattr__(self, name):
            return getattr(self._real, name)

    db._local.conn = TrackingConnection(original_conn)
    try:
        run_importance_scoring(engine)
    finally:
        db._local.conn = original_conn

    # Should have at most 1 edge-related query (the batch), not N
    assert len(query_log) <= 1


def test_weight_normalization(engine):
    """Custom weights that don't sum to 1.0 should still produce valid [0, 1] scores."""
    node_id, _ = engine.remember(CreateNodeRequest(
        content="Test node for weight normalization",
        type=NodeType.fact,
        tier=Tier.core,
        title="Weight test",
    ))

    # Set all weights to 0.5 (total = 1.5)
    engine.settings.importance_access_weight = 0.5
    engine.settings.importance_edge_weight = 0.5
    engine.settings.importance_recency_weight = 0.5

    # Force update by setting importance far from expected
    engine.db.conn.execute(
        "UPDATE nodes SET importance = 0.0 WHERE id = ?", (node_id,)
    )
    engine.db.conn.commit()

    run_importance_scoring(engine)

    rows = engine.db.conn.execute(
        "SELECT importance FROM nodes WHERE source != 'system:self'"
    ).fetchall()

    for r in rows:
        assert 0.0 <= r["importance"] <= 1.0


def test_confidence_affects_search_ranking(engine):
    """Confidence factor in search should rank high-confidence above low-confidence."""
    # Create two nodes with identical content but different confidence
    high_id, _ = engine.remember(CreateNodeRequest(
        content="Architecture decision about caching layer",
        type=NodeType.decision,
        tier=Tier.working,
        title="Caching architecture",
        confidence=1.0,
    ))

    low_id, _ = engine.remember(CreateNodeRequest(
        content="Architecture decision about caching layer",
        type=NodeType.decision,
        tier=Tier.working,
        title="Caching architecture uncertain",
        confidence=0.1,
    ))

    # Search for the content — confidence_factor in search should differentiate
    results = engine.recall_search_structured(
        "caching architecture", limit=10,
    )

    # Find both nodes in results
    result_ids = [r["node"]["id"] for r in results]
    assert high_id in result_ids, "High-confidence node should appear in results"
    assert low_id in result_ids, "Low-confidence node should appear in results"

    high_rank = result_ids.index(high_id)
    low_rank = result_ids.index(low_id)
    assert high_rank < low_rank, (
        f"High-confidence node (rank {high_rank}) should rank above "
        f"low-confidence node (rank {low_rank})"
    )


def test_importance_range_with_new_signals(engine):
    """Verify score range: fresh node ~0.33, hub >0.7, stale disconnected <0.3.

    Stale expected value: 0.34·ln(6)/ln(51) + 0.33·exp(-ln2·30/14) ≈ 0.23.
    """
    from datetime import timedelta

    # Fresh node — no access, no edges, just created
    fresh_id, _ = engine.remember(CreateNodeRequest(
        content="Fresh node just created",
        type=NodeType.fact,
        tier=Tier.working,
        title="Fresh",
    ))

    # Hub node — many accesses and edges
    hub_id, _ = engine.remember(CreateNodeRequest(
        content="Hub node with connections",
        type=NodeType.concept,
        tier=Tier.working,
        title="Hub",
    ))
    engine.db.conn.execute(
        "UPDATE nodes SET access_count = 50 WHERE id = ?", (hub_id,)
    )
    for i in range(10):
        sat_id, _ = engine.remember(CreateNodeRequest(
            content=f"Satellite {i}", type=NodeType.fact, tier=Tier.working,
        ))
        engine.connect(ConnectRequest(
            source_id=hub_id, target_id=sat_id, edge=EdgeType.related_to,
        ))

    # Stale node — old, no edges, few accesses
    stale_id, _ = engine.remember(CreateNodeRequest(
        content="Stale disconnected node",
        type=NodeType.fact,
        tier=Tier.working,
        title="Stale",
    ))
    old_date = (datetime.now(timezone.utc) - timedelta(days=30)).isoformat()
    engine.db.conn.execute(
        "UPDATE nodes SET access_count = 5, last_accessed = ?, last_review = ? WHERE id = ?",
        (old_date, old_date, stale_id),
    )
    # Auto-linking can attach edges non-deterministically; this node is meant to be
    # disconnected, and importance reads edge counts — strip them so it really is.
    engine.db.conn.execute(
        "DELETE FROM edges WHERE source_id = ? OR target_id = ?",
        (stale_id, stale_id),
    )
    engine.db.conn.commit()

    run_importance_scoring(engine)

    fresh_imp = engine.db.conn.execute(
        "SELECT importance FROM nodes WHERE id = ?", (fresh_id,)
    ).fetchone()["importance"]
    hub_imp = engine.db.conn.execute(
        "SELECT importance FROM nodes WHERE id = ?", (hub_id,)
    ).fetchone()["importance"]
    stale_imp = engine.db.conn.execute(
        "SELECT importance FROM nodes WHERE id = ?", (stale_id,)
    ).fetchone()["importance"]

    assert fresh_imp > 0.2, f"Fresh node should be ~0.33, got {fresh_imp}"
    assert hub_imp > 0.7, f"Hub node should be >0.7, got {hub_imp}"
    assert stale_imp < 0.3, f"Stale node should be <0.3, got {stale_imp}"


def test_recency_signal_follows_configured_half_life():
    """At exactly one half-life the signal is 0.5; at zero days it is 1.0."""
    from ormah.background.importance_scorer import _recency_signal

    assert _recency_signal(0.0, 14.0) == pytest.approx(1.0)
    assert _recency_signal(14.0, 14.0) == pytest.approx(0.5)
    assert _recency_signal(28.0, 14.0) == pytest.approx(0.25)

    # A different configured half-life moves the 0.5 point with it.
    assert _recency_signal(7.0, 7.0) == pytest.approx(0.5)


def test_importance_job_uses_configured_recency_half_life(engine):
    """The full scorer reads the setting rather than hardcoding 14 days."""
    node_id, _ = engine.remember(CreateNodeRequest(
        content="Memory used exactly one configured half-life ago",
        type=NodeType.fact,
        tier=Tier.working,
        title="Configured half-life",
    ))
    old = (datetime.now(timezone.utc) - timedelta(days=7)).isoformat()
    engine.settings.importance_access_weight = 0.0
    engine.settings.importance_edge_weight = 0.0
    engine.settings.importance_recency_weight = 1.0
    engine.settings.importance_recency_half_life_days = 7.0
    engine.db.conn.execute(
        "UPDATE nodes SET importance = 0.0, last_review = ?, last_accessed = ? "
        "WHERE id = ?",
        (old, old, node_id),
    )
    engine.db.conn.commit()

    run_importance_scoring(engine)

    importance = engine.db.conn.execute(
        "SELECT importance FROM nodes WHERE id = ?", (node_id,)
    ).fetchone()["importance"]
    assert importance == pytest.approx(0.5, abs=0.001)


def test_recency_ignores_a_lagging_last_review(engine):
    """A node used today must not read as stale because reinforcement is on cooldown."""
    node_id, _ = engine.remember(CreateNodeRequest(
        content="Used today, reinforced a month ago",
        type=NodeType.fact,
        tier=Tier.working,
        title="Lagging review",
    ))
    now = datetime.now(timezone.utc)
    engine.db.conn.execute(
        "UPDATE nodes SET last_accessed = ?, last_review = ?, access_count = 0 "
        "WHERE id = ?",
        (now.isoformat(), (now - timedelta(days=30)).isoformat(), node_id),
    )
    # Keep the assertion about recency only; auto-linking can otherwise add an
    # unrelated edge contribution and make the old timestamp order look healthy.
    engine.db.conn.execute(
        "DELETE FROM edges WHERE source_id = ? OR target_id = ?", (node_id, node_id)
    )
    engine.db.conn.commit()

    run_importance_scoring(engine)

    importance = engine.db.conn.execute(
        "SELECT importance FROM nodes WHERE id = ?", (node_id,)
    ).fetchone()["importance"]
    settings = engine.settings
    total = (
        settings.importance_access_weight
        + settings.importance_edge_weight
        + settings.importance_recency_weight
    )
    assert importance == pytest.approx(
        settings.importance_recency_weight / total, abs=0.02
    )


def test_recency_signal_survives_an_invalid_half_life():
    """Defence in depth (council I1): the validator should make these unreachable,
    but the guard must hold anyway. A zero or negative half-life must never raise
    ZeroDivisionError and kill the job. A NaN half-life skips the '<= 0' check
    (nan <= 0 is False) and would otherwise reach math.exp, returning NaN — which
    the caller's clamp, max(0.0, min(1.0, x)), turns into 1.0 rather than
    propagating, silently saturating importance at maximum. An infinite half-life
    also skips the check and would make every node read as perfectly fresh
    (recency_signal == 1.0) regardless of age."""
    from ormah.background.importance_scorer import _recency_signal

    assert _recency_signal(10.0, 0.0) == 0.0
    assert _recency_signal(10.0, -14.0) == 0.0
    assert _recency_signal(10.0, float("nan")) == 0.0
    assert _recency_signal(10.0, float("inf")) == 0.0


def test_importance_recency_is_independent_of_stability(engine):
    """Two nodes of identical age and profile score the same regardless of stability.

    Before #222 the recency term was exp(-days/stability), so a high-stability node
    scored far higher than a low-stability one at the same age. It must not anymore.
    """
    low_id, _ = engine.remember(CreateNodeRequest(
        content="Alpha node about zebras and telescopes",
        type=NodeType.fact,
        tier=Tier.working,
        title="Low stability",
    ))
    high_id, _ = engine.remember(CreateNodeRequest(
        content="Beta node about zebras and telescopes",
        type=NodeType.fact,
        tier=Tier.working,
        title="High stability",
    ))

    old_date = (datetime.now(timezone.utc) - timedelta(days=20)).isoformat()
    engine.db.conn.execute(
        "UPDATE nodes SET last_accessed = ?, last_review = ?, access_count = 3, "
        "stability = 1.0 WHERE id = ?",
        (old_date, old_date, low_id),
    )
    engine.db.conn.execute(
        "UPDATE nodes SET last_accessed = ?, last_review = ?, access_count = 3, "
        "stability = 100.0 WHERE id = ?",
        (old_date, old_date, high_id),
    )
    # Auto-linking can attach edges non-deterministically; importance reads edge
    # counts, so strip them to isolate the recency term.
    engine.db.conn.execute(
        "DELETE FROM edges WHERE source_id IN (?, ?) OR target_id IN (?, ?)",
        (low_id, high_id, low_id, high_id),
    )
    engine.db.conn.commit()

    run_importance_scoring(engine)

    low_imp = engine.db.conn.execute(
        "SELECT importance FROM nodes WHERE id = ?", (low_id,)
    ).fetchone()["importance"]
    high_imp = engine.db.conn.execute(
        "SELECT importance FROM nodes WHERE id = ?", (high_id,)
    ).fetchone()["importance"]

    assert low_imp == pytest.approx(high_imp), (
        f"stability must not affect importance recency: {low_imp} vs {high_imp}"
    )


def test_importance_scoring_takes_the_lock_per_node_and_per_chunk(engine):
    """No LLM in this job at all — the hold is pure whole-run retention."""
    from tests.test_background.lock_probe import install_probe

    for i in range(4):
        nid, _ = engine.remember(CreateNodeRequest(
            content=f"node {i} with enough content to score", type=NodeType.fact,
            title=f"node {i}"))
        engine.db.conn.execute(
            "UPDATE nodes SET access_count = ?, importance = 0.0 WHERE id = ?", (i * 5, nid))
    engine.db.conn.commit()

    probe = install_probe(engine)
    run_importance_scoring(engine)

    # Before the fix: exactly 1 for any node count.
    assert probe.acquisitions >= 4


def test_importance_scoring_aborts_when_a_restore_lands_mid_run(engine):
    from ormah.background.memory_lock import RestoredUnderfoot  # noqa: F401  (documents intent)

    for i in range(4):
        nid, _ = engine.remember(CreateNodeRequest(
            content=f"node {i} with enough content to score", type=NodeType.fact,
            title=f"node {i}"))
        engine.db.conn.execute(
            "UPDATE nodes SET access_count = ?, importance = 0.0 WHERE id = ?", (i * 5, nid))
    engine.db.conn.commit()

    real_save = engine.file_store.save
    saves = {"count": 0}

    def bump_after_first(node):
        path = real_save(node)
        saves["count"] += 1
        if saves["count"] == 1:
            engine._restore_epoch += 1
        return path

    engine.file_store.save = bump_after_first
    run_importance_scoring(engine)  # returns cleanly

    assert saves["count"] == 1

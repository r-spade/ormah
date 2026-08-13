"""Deterministic tests for the versioned lifecycle policy and confirmed use."""

from __future__ import annotations

import math
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, patch

import pytest

from ormah.engine.lifecycle import (
    DEFAULT_INITIAL_STABILITY,
    LIFECYCLE_MODEL_VERSION,
    archival_deadline,
    bounded_stability_update,
    importance_recency,
    reinforcement_spacing,
    retrievability,
)
from ormah.config import Settings
from ormah.models.node import ConnectRequest, CreateNodeRequest, EdgeType, NodeType, Tier


def _backdate(engine, node_id: str, *, days: float, stability: float, tier: Tier) -> None:
    anchor = datetime.now(timezone.utc) - timedelta(days=days)
    node = engine.file_store.load(node_id)
    node.stability = stability
    node.tier = tier
    node.last_accessed = anchor
    node.last_review = anchor
    engine.file_store.save(node)
    with engine.db.transaction() as conn:
        conn.execute(
            """
            UPDATE nodes
            SET tier = ?, stability = ?, last_accessed = ?, last_review = ?
            WHERE id = ?
            """,
            (tier.value, stability, anchor.isoformat(), anchor.isoformat(), node_id),
        )


def test_default_initial_stability_is_the_seven_day_threshold_anchor():
    assert DEFAULT_INITIAL_STABILITY == pytest.approx(5.814084815577761)
    assert retrievability(7.0, DEFAULT_INITIAL_STABILITY, fallback=1.0) == pytest.approx(0.3)


def test_lifecycle_settings_are_explicit_and_validated(tmp_path):
    settings = Settings(memory_dir=tmp_path)
    assert settings.fsrs_reinforcement_gain == 0.5
    assert settings.fsrs_reinforcement_saturation_exponent == 0.5
    assert settings.fsrs_reinforcement_spacing_cap == 2.0
    with pytest.raises(ValueError):
        Settings(memory_dir=tmp_path, fsrs_reinforcement_spacing_cap=0.5)


def test_production_creation_uses_configured_initial_stability(engine):
    engine.settings.fsrs_initial_stability = 12.5
    node_id, _ = engine.remember(CreateNodeRequest(
        content="Configured initial stability", type=NodeType.fact, title="Configured"
    ))
    assert engine.file_store.load(node_id).stability == pytest.approx(12.5)


def test_bounded_update_uses_old_stability_before_promotion_floor():
    reinforced = bounded_stability_update(
        1.0,
        30.0,
        gain=0.5,
        saturation_exponent=0.5,
        spacing_cap=2.0,
        max_stability=365.0,
        fallback=1.0,
    )
    assert reinforced == pytest.approx(2.0)
    assert max(reinforced, DEFAULT_INITIAL_STABILITY) == pytest.approx(
        DEFAULT_INITIAL_STABILITY
    )


def test_spacing_is_finite_and_matches_near_threshold_value():
    spacing = reinforcement_spacing(
        10_000_000.0,
        1.0,
        spacing_cap=2.0,
        fallback=1.0,
    )
    assert math.isfinite(spacing)
    assert spacing == pytest.approx(2.0)
    assert reinforcement_spacing(
        -math.log(0.3),
        1.0,
        spacing_cap=2.0,
        fallback=1.0,
    ) == pytest.approx(1.272, abs=0.001)


def test_saturated_policy_reaches_default_cap_in_about_74_updates():
    stability = 1.0
    updates = 0
    while stability < 365.0:
        stability = bounded_stability_update(
            stability,
            0.0,
            gain=0.5,
            saturation_exponent=0.5,
            spacing_cap=2.0,
            max_stability=365.0,
            fallback=1.0,
        )
        updates += 1
    assert updates == 74


def test_importance_recency_has_its_own_half_life():
    assert importance_recency(14.0, 14.0) == pytest.approx(0.5)
    assert importance_recency(14.0, 7.0) == pytest.approx(0.25)


def test_archival_deadline_is_explicit_for_future_model_migrations():
    anchor = datetime(2026, 1, 1, tzinfo=timezone.utc)
    deadline = archival_deadline(
        anchor,
        DEFAULT_INITIAL_STABILITY,
        threshold=0.3,
        fallback=1.0,
    )
    assert (deadline - (anchor + timedelta(days=7))).total_seconds() == pytest.approx(0, abs=1)


def test_confirmed_use_promotes_after_reinforcing_old_stability(engine):
    node_id, _ = engine.remember(CreateNodeRequest(
        content="An archival memory that was deliberately recalled.",
        type=NodeType.fact,
        title="Recall me",
    ))
    _backdate(engine, node_id, days=30, stability=1.0, tier=Tier.archival)

    assert engine._record_confirmed_use(node_id) is True

    node = engine.file_store.load(node_id)
    assert node.tier == Tier.working
    assert node.stability == pytest.approx(DEFAULT_INITIAL_STABILITY)
    assert node.access_count == 1
    assert node.last_review is not None
    indexed = engine.db.conn.execute(
        "SELECT tier, stability, access_count FROM nodes WHERE id = ?", (node_id,)
    ).fetchone()
    assert indexed["tier"] == "working"
    assert indexed["stability"] == pytest.approx(DEFAULT_INITIAL_STABILITY)
    assert indexed["access_count"] == 1
    version = engine.db.conn.execute(
        "SELECT value FROM meta WHERE key = 'lifecycle_model_version'"
    ).fetchone()
    assert version["value"] == str(LIFECYCLE_MODEL_VERSION)


def test_repeated_same_day_confirmed_use_updates_anchor_but_not_stability(engine):
    node_id, _ = engine.remember(CreateNodeRequest(
        content="A node used repeatedly in one session.",
        type=NodeType.fact,
        title="Repeated use",
    ))
    _backdate(engine, node_id, days=2, stability=1.0, tier=Tier.working)

    for _ in range(10):
        assert engine._record_confirmed_use(node_id) is True

    node = engine.file_store.load(node_id)
    assert node.access_count == 10
    assert node.last_review is not None
    assert node.last_accessed > node.last_review - timedelta(seconds=1)
    first_update_stability = node.stability
    first_review = node.last_review

    assert engine._record_confirmed_use(node_id) is True
    node_after = engine.file_store.load(node_id)
    assert node_after.access_count == 11
    assert node_after.stability == pytest.approx(first_update_stability)
    assert node_after.last_review == first_review


def test_generic_derived_from_does_not_block_promotion_but_consolidation_does(engine):
    generic_id, _ = engine.remember(CreateNodeRequest(
        content="A generally derived memory.", type=NodeType.fact, title="Generic"
    ))
    replacement_id, _ = engine.remember(CreateNodeRequest(
        content="Another memory.", type=NodeType.fact, title="Replacement"
    ))
    _backdate(engine, generic_id, days=30, stability=1.0, tier=Tier.archival)
    engine.connect(ConnectRequest(
        source_id=replacement_id,
        target_id=generic_id,
        edge=EdgeType.derived_from,
        weight=1.0,
    ))

    engine._record_confirmed_use(generic_id)
    assert engine.file_store.load(generic_id).tier == Tier.working

    consolidated_id, _ = engine.remember(CreateNodeRequest(
        content="A deliberately superseded original.", type=NodeType.fact, title="Superseded"
    ))
    _backdate(engine, consolidated_id, days=30, stability=1.0, tier=Tier.archival)
    assert engine.mark_consolidated(consolidated_id, replacement_id) is True
    engine._record_confirmed_use(consolidated_id)
    consolidated = engine.file_store.load(consolidated_id)
    assert consolidated.consolidated_into == replacement_id
    assert consolidated.tier == Tier.archival


@pytest.mark.parametrize("use_fts_fallback", [False, True])
def test_broad_recall_is_non_mutating_on_hybrid_and_fts_paths(engine, use_fts_fallback):
    node_id, _ = engine.remember(CreateNodeRequest(
        content="Search result surfacing is not confirmed use.",
        type=NodeType.fact,
        title="Surfacing semantics",
    ))
    before = engine.file_store.load(node_id)
    result = {"node": engine.graph.get_node(node_id), "score": 1.0, "source": "hybrid"}

    if use_fts_fallback:
        with patch.object(engine, "_get_hybrid_search", return_value=None), \
             patch.object(engine.graph, "fts_search", return_value=[{"id": node_id, "score": 1.0}]):
            engine.recall_search_structured(
                "surfacing semantics", limit=1, spread_activation=False, min_relevance=0.0
            )
    else:
        fake_search = MagicMock()
        fake_search.search.return_value = [result]
        with patch.object(engine, "_get_hybrid_search", return_value=fake_search):
            engine.recall_search_structured(
                "surfacing semantics", limit=1, spread_activation=False, min_relevance=0.0
            )

    after = engine.file_store.load(node_id)
    assert after.access_count == before.access_count == 0
    assert after.last_accessed == before.last_accessed
    assert after.last_review == before.last_review
    assert after.stability == before.stability


def test_recall_node_confirms_only_requested_node_not_neighbors(engine):
    requested_id, _ = engine.remember(CreateNodeRequest(
        content="The deliberately fetched memory.", type=NodeType.fact, title="Requested"
    ))
    neighbor_id, _ = engine.remember(CreateNodeRequest(
        content="A surfaced graph neighbor.", type=NodeType.fact, title="Neighbor"
    ))
    engine.connect(ConnectRequest(
        source_id=requested_id, target_id=neighbor_id, edge=EdgeType.related_to
    ))

    engine.recall_node(requested_id)

    assert engine.file_store.load(requested_id).access_count == 1
    assert engine.file_store.load(neighbor_id).access_count == 0

"""Tests for spreading activation in recall."""

from datetime import datetime, timezone

from ormah.engine.traversal import format_search_results
from ormah.models.node import (
    ConnectRequest,
    CreateNodeRequest,
    EdgeType,
    NodeType,
)


def _remember(engine, content, title=None, tags=None):
    req = CreateNodeRequest(
        content=content,
        type=NodeType.fact,
        title=title or content[:40],
        tags=tags or [],
    )
    node_id, _ = engine.remember(req, agent_id="test")
    return node_id


def _connect(engine, source_id, target_id, edge_type, weight=0.8):
    req = ConnectRequest(
        source_id=source_id,
        target_id=target_id,
        edge=edge_type,
        weight=weight,
    )
    engine.connect(req)


def _make_result(engine, node_id, score=0.9):
    node = engine.graph.get_node(node_id)
    return {"node": node, "score": score, "source": "hybrid"}


def _filter_user_node(enriched, engine):
    """Remove the auto-created user/self node from results for cleaner assertions."""
    user_id = getattr(engine, "user_node_id", None)
    if user_id is None:
        return enriched
    return [r for r in enriched if r["node"]["id"] != user_id]


def test_basic_activation(engine):
    """Neighbor of a top hit appears as activated."""
    seed_id = _remember(engine, "PostgreSQL database config")
    neighbor_id = _remember(engine, "Database migration decision")
    _connect(engine, seed_id, neighbor_id, EdgeType.supports, weight=0.8)

    results = [_make_result(engine, seed_id, score=0.9)]
    enriched = _filter_user_node(engine._spread_activation(results, limit=10), engine)

    assert len(enriched) == 2
    assert enriched[0]["node"]["id"] == seed_id
    activated = enriched[1]
    assert activated["source"] == "activated"
    assert activated["node"]["id"] == neighbor_id
    assert activated["activated_by"] == seed_id
    assert activated["activation_edge"] == "supports"


def test_deduplication(engine):
    """Node already in direct hits is not duplicated as activated."""
    id_a = _remember(engine, "Node A - direct hit")
    id_b = _remember(engine, "Node B - also direct hit")
    _connect(engine, id_a, id_b, EdgeType.related_to, weight=0.9)

    results = [
        _make_result(engine, id_a, score=0.9),
        _make_result(engine, id_b, score=0.7),
    ]
    enriched = _filter_user_node(engine._spread_activation(results, limit=10), engine)

    # No activated results since B is already a direct hit
    ids = [r["node"]["id"] for r in enriched]
    assert ids.count(id_b) == 1
    assert len(enriched) == 2


def test_max_per_seed_cap(engine):
    """Only top 3 neighbors per seed are activated."""
    seed_id = _remember(engine, "Seed node")
    neighbor_ids = []
    for i in range(5):
        nid = _remember(engine, f"Neighbor {i}")
        _connect(engine, seed_id, nid, EdgeType.supports, weight=round(0.9 - i * 0.1, 1))
        neighbor_ids.append(nid)

    results = [_make_result(engine, seed_id, score=0.9)]
    enriched = engine._spread_activation(results, limit=20)

    activated = [r for r in enriched if r.get("source") == "activated"]
    assert len(activated) == 3  # max_per_seed default is 3


def test_multiple_seeds_max_score(engine):
    """When multiple seeds activate the same neighbor, max score wins."""
    seed_a = _remember(engine, "Seed A")
    seed_b = _remember(engine, "Seed B")
    shared = _remember(engine, "Shared neighbor")

    _connect(engine, seed_a, shared, EdgeType.supports, weight=0.9)
    _connect(engine, seed_b, shared, EdgeType.related_to, weight=0.5)

    results = [
        _make_result(engine, seed_a, score=0.8),
        _make_result(engine, seed_b, score=0.6),
    ]
    enriched = _filter_user_node(engine._spread_activation(results, limit=10), engine)

    activated = [r for r in enriched if r.get("source") == "activated"]
    assert len(activated) == 1
    assert activated[0]["node"]["id"] == shared

    # supports (factor=1.0): 0.8 * 0.9 * 1.0 * 0.5 = 0.36
    # related_to (factor=0.7): 0.6 * 0.5 * 0.7 * 0.5 = 0.105
    # Max should be 0.36
    assert abs(activated[0]["score"] - 0.36) < 0.01


def test_edge_type_factors(engine):
    """contradicts propagates less activation than supports, and is labelled as conflict."""
    seed_id = _remember(engine, "PostgreSQL tuning parameters for production")
    supports_id = _remember(engine, "Solar panel efficiency in desert climates")
    contradicts_id = _remember(engine, "Recipes for sourdough bread with rye flour")

    # Remove any auto-created edges so only our explicit edges exist
    for nid in (seed_id, supports_id, contradicts_id):
        engine.db.conn.execute(
            "DELETE FROM edges WHERE source_id = ? OR target_id = ?", (nid, nid)
        )
    engine.db.conn.commit()

    _connect(engine, seed_id, supports_id, EdgeType.supports, weight=0.8)
    _connect(engine, seed_id, contradicts_id, EdgeType.contradicts, weight=0.8)

    results = [_make_result(engine, seed_id, score=0.9)]
    enriched = _filter_user_node(engine._spread_activation(results, limit=10), engine)

    # supports -> source="activated", contradicts -> source="conflict"
    by_id = {r["node"]["id"]: r for r in enriched if r["node"]["id"] != seed_id}
    assert supports_id in by_id
    assert contradicts_id in by_id
    assert by_id[supports_id]["source"] == "activated"
    assert by_id[contradicts_id]["source"] == "conflict"
    # supports factor=1.0, contradicts factor=0.4
    assert by_id[supports_id]["score"] > by_id[contradicts_id]["score"]


def test_empty_results(engine):
    """Empty results in, empty results out."""
    enriched = engine._spread_activation([], limit=10)
    assert enriched == []


def test_format_output_distinguishes_activated(engine):
    """format_search_results separates direct hits from activated."""
    seed_id = _remember(engine, "Direct hit node")
    neighbor_id = _remember(engine, "Activated neighbor node")
    _connect(engine, seed_id, neighbor_id, EdgeType.defines, weight=0.7)

    results = [_make_result(engine, seed_id, score=0.9)]
    enriched = _filter_user_node(engine._spread_activation(results, limit=10), engine)

    formatted = format_search_results(enriched)
    assert "Found 1 memories (+1 related):" in formatted
    assert "--- Related (via graph) ---" in formatted
    assert "via defines from" in formatted


def test_format_output_shows_conflicting_context(engine):
    """format_search_results shows contradicts-activated results under 'Conflicting context'."""
    # Use dissimilar content so auto-linker doesn't create a related_to edge
    seed_id = _remember(engine, "The sky is often blue during clear days")
    conflict_id = _remember(engine, "Bananas are a tropical fruit grown in warm climates")

    # Remove any auto-created edges, then add only a contradicts edge
    engine.db.conn.execute(
        "DELETE FROM edges WHERE (source_id = ? OR target_id = ?) AND (source_id = ? OR target_id = ?)",
        (seed_id, seed_id, conflict_id, conflict_id),
    )
    engine.db.conn.commit()
    _connect(engine, seed_id, conflict_id, EdgeType.contradicts, weight=0.8)

    results = [_make_result(engine, seed_id, score=0.9)]
    enriched = engine._spread_activation(results, limit=10)

    formatted = format_search_results(enriched)
    assert "--- Conflicting context ---" in formatted
    assert "contradicts" in formatted


def test_merged_sort_order(engine):
    """Activated result with high score ranks above direct hit with low score."""
    low_hit = _remember(engine, "Low relevance node")
    seed_id = _remember(engine, "High relevance seed")
    neighbor_id = _remember(engine, "Well-connected neighbor")
    _connect(engine, seed_id, neighbor_id, EdgeType.supports, weight=0.95)

    results = [
        _make_result(engine, seed_id, score=0.9),
        _make_result(engine, low_hit, score=0.05),
    ]
    enriched = engine._spread_activation(results, limit=10)

    # Activated neighbor (0.9 * 0.95 * 1.0 * 0.5 = 0.4275) should rank above
    # the low-score direct hit (0.05)
    scores = [(r["node"]["id"], r["score"]) for r in enriched]
    neighbor_pos = next(i for i, (nid, _) in enumerate(scores) if nid == neighbor_id)
    low_pos = next(i for i, (nid, _) in enumerate(scores) if nid == low_hit)
    assert neighbor_pos < low_pos, f"Activated neighbor should rank above low-score direct hit"

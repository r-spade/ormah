"""Tests for recency, access frequency, and tier scoring signals in hybrid search."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, patch

import pytest

from ormah.config import Settings
from ormah.embeddings.hybrid_search import HybridSearch


def _make_node(node_id, tier="working", access_count=0, last_accessed=None):
    """Build a minimal node dict with scoring-relevant fields."""
    if last_accessed is None:
        last_accessed = datetime.now(timezone.utc)
    return {
        "id": node_id,
        "type": "fact",
        "tier": tier,
        "content": f"content of {node_id}",
        "access_count": access_count,
        "last_accessed": last_accessed.isoformat(),
    }


@pytest.fixture
def scored_hybrid(tmp_path):
    """HybridSearch with controllable node metadata for scoring signal tests."""
    settings = Settings(
        memory_dir=tmp_path,
        fts_weight=0.4,
        vector_weight=0.6,
        similarity_threshold=0.4,
        rrf_k=60,
        recency_boost=0.05,
        recency_half_life_days=7.0,
        access_boost=0.1,
        tier_boost_core=0.1,
        tier_boost_working=0.0,
        tier_boost_archival=-0.1,
    )
    with (
        patch("ormah.embeddings.hybrid_search.GraphIndex") as MockGraph,
        patch("ormah.embeddings.hybrid_search.VectorStore"),
        patch("ormah.embeddings.hybrid_search.get_encoder"),
    ):
        conn = MagicMock()
        hs = HybridSearch(conn, settings)
        hs.graph = MockGraph.return_value
    return hs


def _run_search(hs, fts_results, vec_results, node_map):
    """Run search with controlled FTS, vector, and node metadata."""
    hs.graph.fts_search.return_value = fts_results
    hs.vec_store.search.return_value = vec_results
    hs.encoder.encode.return_value = "fake_vec"
    hs.graph.get_node.side_effect = lambda nid: node_map.get(nid)
    hs.graph.get_nodes_batch.side_effect = lambda ids: {
        nid: node_map[nid] for nid in ids if nid in node_map
    }
    hs.graph.get_tags_batch.side_effect = lambda ids: {}
    return hs.search("test query", limit=10)


def test_recently_accessed_ranks_higher(scored_hybrid):
    """A recently accessed node should outrank an older one with the same base score."""
    now = datetime.now(timezone.utc)
    nodes = {
        "recent": _make_node("recent", last_accessed=now),
        "stale": _make_node("stale", last_accessed=now - timedelta(days=30)),
    }
    results = _run_search(
        scored_hybrid,
        fts_results=[
            {"id": "recent", "score": 5.0},
            {"id": "stale", "score": 5.0},
        ],
        vec_results=[
            {"id": "recent", "similarity": 0.8},
            {"id": "stale", "similarity": 0.8},
        ],
        node_map=nodes,
    )
    assert results[0]["node"]["id"] == "recent"
    assert results[0]["score"] > results[1]["score"]


def test_frequently_accessed_ranks_higher(scored_hybrid):
    """A frequently accessed node should outrank a rarely accessed one."""
    now = datetime.now(timezone.utc)
    nodes = {
        "popular": _make_node("popular", access_count=15, last_accessed=now),
        "ignored": _make_node("ignored", access_count=0, last_accessed=now),
    }
    results = _run_search(
        scored_hybrid,
        fts_results=[
            {"id": "popular", "score": 5.0},
            {"id": "ignored", "score": 5.0},
        ],
        vec_results=[
            {"id": "popular", "similarity": 0.8},
            {"id": "ignored", "similarity": 0.8},
        ],
        node_map=nodes,
    )
    assert results[0]["node"]["id"] == "popular"
    assert results[0]["score"] > results[1]["score"]


def test_core_tier_outranks_archival(scored_hybrid):
    """A core node should outrank an archival node with the same base score."""
    now = datetime.now(timezone.utc)
    nodes = {
        "core_node": _make_node("core_node", tier="core", last_accessed=now),
        "archival_node": _make_node("archival_node", tier="archival", last_accessed=now),
    }
    results = _run_search(
        scored_hybrid,
        fts_results=[
            {"id": "core_node", "score": 5.0},
            {"id": "archival_node", "score": 5.0},
        ],
        vec_results=[
            {"id": "core_node", "similarity": 0.8},
            {"id": "archival_node", "similarity": 0.8},
        ],
        node_map=nodes,
    )
    assert results[0]["node"]["id"] == "core_node"
    # core boost (+10%) vs archival boost (-10%) = ~20% proportional difference
    core_score = results[0]["score"]
    archival_score = results[1]["score"]
    assert core_score > archival_score
    ratio = core_score / archival_score if archival_score > 0 else float("inf")
    assert ratio == pytest.approx(1.1 / 0.9, rel=0.05)


def test_strong_relevance_beats_boosts(tmp_path):
    """Boosts should not override a large relevance gap.

    RRF base scores are small (~0.01–0.03), so boost magnitudes must be
    proportional.  This test uses realistic small boosts and verifies that
    a node ranked #1 in both retrievers still beats a maximally-boosted
    node that only appears in one list at a low rank.
    """
    settings = Settings(
        memory_dir=tmp_path,
        fts_weight=0.4,
        vector_weight=0.6,
        similarity_threshold=0.4,
        rrf_k=60,
        recency_boost=0.001,
        recency_half_life_days=7.0,
        access_boost=0.001,
        tier_boost_core=0.001,
        tier_boost_working=0.0,
        tier_boost_archival=-0.001,
        # Disable score blending so this test exercises pure boost behavior
        similarity_blend_weight=0.0,
        fts_only_dampening=1.0,
        min_result_score=0.0,
    )
    with (
        patch("ormah.embeddings.hybrid_search.GraphIndex") as MockGraph,
        patch("ormah.embeddings.hybrid_search.VectorStore"),
        patch("ormah.embeddings.hybrid_search.get_encoder"),
    ):
        conn = MagicMock()
        hs = HybridSearch(conn, settings)
        hs.graph = MockGraph.return_value

    now = datetime.now(timezone.utc)
    filler_ids = [f"filler{i}" for i in range(8)]
    nodes = {
        "relevant": _make_node("relevant", tier="archival", access_count=0,
                               last_accessed=now - timedelta(days=60)),
        "boosted": _make_node("boosted", tier="core", access_count=20,
                              last_accessed=now),
    }
    for fid in filler_ids:
        nodes[fid] = _make_node(fid, last_accessed=now)

    results = _run_search(
        hs,
        fts_results=[
            {"id": "relevant", "score": 10.0},
            *[{"id": fid, "score": 5.0 - i * 0.1} for i, fid in enumerate(filler_ids)],
            {"id": "boosted", "score": 0.5},
        ],
        vec_results=[
            {"id": "relevant", "similarity": 0.95},
            *[{"id": fid, "similarity": 0.8 - i * 0.03} for i, fid in enumerate(filler_ids)],
            # "boosted" not in vector results — only FTS at rank 10
        ],
        node_map=nodes,
    )
    # "relevant" is rank 1 in both lists → strong RRF score
    # "boosted" is rank 10 in FTS only → weak RRF score
    relevant_result = next(r for r in results if r["node"]["id"] == "relevant")
    boosted_result = next(r for r in results if r["node"]["id"] == "boosted")
    assert relevant_result["score"] > boosted_result["score"]


def test_access_boost_is_logarithmic(scored_hybrid):
    """Going from 0→5 accesses should give a larger boost than 15→20."""
    now = datetime.now(timezone.utc)
    nodes = {
        "low": _make_node("low", access_count=5, last_accessed=now),
        "high": _make_node("high", access_count=20, last_accessed=now),
    }
    results = _run_search(
        scored_hybrid,
        fts_results=[
            {"id": "low", "score": 5.0},
            {"id": "high", "score": 5.0},
        ],
        vec_results=[
            {"id": "low", "similarity": 0.8},
            {"id": "high", "similarity": 0.8},
        ],
        node_map=nodes,
    )
    scores = {r["node"]["id"]: r["score"] for r in results}
    # Both get boosted but the gap between 5 and 20 accesses should be small
    # (logarithmic, not linear)
    gap = scores["high"] - scores["low"]
    assert gap > 0
    assert gap < 0.05  # small due to log scaling


def test_zero_boosts_preserve_base_ranking(tmp_path):
    """With all boosts set to 0, ranking should match pure fusion."""
    settings = Settings(
        memory_dir=tmp_path,
        fts_weight=0.4,
        vector_weight=0.6,
        similarity_threshold=0.4,
        rrf_k=60,
        recency_boost=0.0,
        access_boost=0.0,
        tier_boost_core=0.0,
        tier_boost_working=0.0,
        tier_boost_archival=0.0,
    )
    with (
        patch("ormah.embeddings.hybrid_search.GraphIndex") as MockGraph,
        patch("ormah.embeddings.hybrid_search.VectorStore"),
        patch("ormah.embeddings.hybrid_search.get_encoder"),
    ):
        conn = MagicMock()
        hs = HybridSearch(conn, settings)
        hs.graph = MockGraph.return_value

    now = datetime.now(timezone.utc)
    nodes = {
        "a": _make_node("a", tier="core", access_count=100, last_accessed=now),
        "b": _make_node("b", tier="archival", access_count=0,
                        last_accessed=now - timedelta(days=90)),
    }
    results = _run_search(
        hs,
        fts_results=[
            {"id": "a", "score": 3.0},
            {"id": "b", "score": 8.0},
        ],
        vec_results=[
            {"id": "a", "similarity": 0.5},
            {"id": "b", "similarity": 0.9},
        ],
        node_map=nodes,
    )
    # "b" has higher base score in both retrievers — with zero boosts, it wins
    assert results[0]["node"]["id"] == "b"


# ---------------------------------------------------------------------------
# Space scoring tests
# ---------------------------------------------------------------------------


class TestSpaceScoring:
    """Tests for score-based space prioritization in MemoryEngine."""

    def _make_result(self, node_id, score, space=None):
        return {
            "node": {"id": node_id, "space": space, "content": f"content of {node_id}"},
            "score": score,
            "source": "hybrid",
        }

    def test_space_scoring_demotes_other_project(self, tmp_path):
        """Same-space results score higher than otherwise-identical cross-project results."""
        from ormah.engine.memory_engine import MemoryEngine

        settings = Settings(
            memory_dir=tmp_path,
            space_boost_global=0.85,
            space_boost_other=0.6,
        )
        engine = MemoryEngine.__new__(MemoryEngine)
        engine.settings = settings

        results = [
            self._make_result("same", 1.0, space="myproject"),
            self._make_result("other", 1.0, space="otherproject"),
        ]
        adjusted = engine._apply_space_scores(results, "myproject")
        scores = {r["node"]["id"]: r["score"] for r in adjusted}
        assert scores["same"] == pytest.approx(1.0)
        assert scores["other"] == pytest.approx(0.6)
        assert adjusted[0]["node"]["id"] == "same"

    def test_space_scoring_preserves_strong_cross_project(self, tmp_path):
        """A high-relevance cross-project result still beats a weak current-project result."""
        from ormah.engine.memory_engine import MemoryEngine

        settings = Settings(
            memory_dir=tmp_path,
            space_boost_global=0.85,
            space_boost_other=0.6,
        )
        engine = MemoryEngine.__new__(MemoryEngine)
        engine.settings = settings

        results = [
            self._make_result("weak_local", 0.3, space="myproject"),
            self._make_result("strong_other", 1.0, space="otherproject"),
        ]
        adjusted = engine._apply_space_scores(results, "myproject")
        scores = {r["node"]["id"]: r["score"] for r in adjusted}
        # strong_other: 1.0 * 0.6 = 0.6 > weak_local: 0.3 * 1.0 = 0.3
        assert scores["strong_other"] > scores["weak_local"]
        assert adjusted[0]["node"]["id"] == "strong_other"

    def test_global_space_gets_moderate_penalty(self, tmp_path):
        """Global (space=None) results get the global boost factor."""
        from ormah.engine.memory_engine import MemoryEngine

        settings = Settings(
            memory_dir=tmp_path,
            space_boost_global=0.85,
            space_boost_other=0.6,
        )
        engine = MemoryEngine.__new__(MemoryEngine)
        engine.settings = settings

        results = [
            self._make_result("local", 1.0, space="myproject"),
            self._make_result("global", 1.0, space=None),
            self._make_result("other", 1.0, space="otherproject"),
        ]
        adjusted = engine._apply_space_scores(results, "myproject")
        scores = {r["node"]["id"]: r["score"] for r in adjusted}
        assert scores["local"] > scores["global"] > scores["other"]
        assert scores["global"] == pytest.approx(0.85)


# ---------------------------------------------------------------------------
# Spreading activation access isolation tests
# ---------------------------------------------------------------------------


class TestSpreadingActivationAccessIsolation:
    """Verify that _touch_access is only called for direct search matches,
    not for nodes injected by spreading activation."""

    def test_touch_access_skips_activated_nodes(self, engine):
        """Spreading activation results (source=activated/conflict) must not
        get their access_count bumped."""
        # Create three nodes via the engine
        from ormah.models.node import CreateNodeRequest

        id_a, _ = engine.remember(CreateNodeRequest(
            content="Alpha direct match node",
            title="Alpha",
            type="fact",
            tier="working",
        ))
        id_b, _ = engine.remember(CreateNodeRequest(
            content="Beta neighbor node",
            title="Beta",
            type="fact",
            tier="working",
        ))
        id_c, _ = engine.remember(CreateNodeRequest(
            content="Gamma direct match node",
            title="Gamma",
            type="fact",
            tier="working",
        ))

        # Record initial access counts
        node_a = engine.file_store.load(id_a)
        node_b = engine.file_store.load(id_b)
        node_c = engine.file_store.load(id_c)
        initial_a = node_a.access_count
        initial_b = node_b.access_count
        initial_c = node_c.access_count

        # Mock _spread_activation to inject B as an activated neighbor
        original_spread = engine._spread_activation

        def mock_spread(results, limit):
            out = original_spread(results, limit)
            # Inject B as if it were pulled in by spreading activation from A
            out.append({
                "node": engine.graph.get_node(id_b),
                "score": 0.5,
                "source": "activated",
                "activated_by": id_a,
                "activation_edge": "related_to",
            })
            return out

        # Mock hybrid search to return A and C as direct matches
        def mock_search(query, limit=10, **filters):
            return [
                {"node": engine.graph.get_node(id_a), "score": 1.0, "source": "hybrid"},
                {"node": engine.graph.get_node(id_c), "score": 0.8, "source": "hybrid"},
            ]

        search_obj = engine._get_hybrid_search()
        if search_obj is not None:
            with patch.object(search_obj, "search", side_effect=mock_search), \
                 patch.object(engine, "_spread_activation", side_effect=mock_spread):
                engine.recall_search_structured("test query", limit=10)
        else:
            # FTS-only path
            engine.graph.fts_search = MagicMock(return_value=[
                {"id": id_a, "score": 5.0},
                {"id": id_c, "score": 4.0},
            ])
            with patch.object(engine, "_spread_activation", side_effect=mock_spread):
                engine.recall_search_structured("test query", limit=10)

        # Verify: A and C should have access_count incremented
        node_a_after = engine.file_store.load(id_a)
        node_c_after = engine.file_store.load(id_c)
        assert node_a_after.access_count == initial_a + 1
        assert node_c_after.access_count == initial_c + 1

        # B (activated neighbor) should NOT have access_count incremented
        node_b_after = engine.file_store.load(id_b)
        assert node_b_after.access_count == initial_b

    def test_touch_access_skips_conflict_nodes(self, engine):
        """Nodes with source=conflict should also be skipped."""
        from ormah.models.node import CreateNodeRequest

        id_a, _ = engine.remember(CreateNodeRequest(
            content="Statement one",
            title="Statement",
            type="fact",
            tier="working",
        ))
        id_conflict, _ = engine.remember(CreateNodeRequest(
            content="Contradicting statement",
            title="Contradiction",
            type="fact",
            tier="working",
        ))

        initial_conflict = engine.file_store.load(id_conflict).access_count

        def mock_spread(results, limit):
            out = list(results)
            out.append({
                "node": engine.graph.get_node(id_conflict),
                "score": 0.3,
                "source": "conflict",
                "activated_by": id_a,
                "activation_edge": "contradicts",
            })
            return out

        def mock_search(query, limit=10, **filters):
            return [
                {"node": engine.graph.get_node(id_a), "score": 1.0, "source": "hybrid"},
            ]

        search_obj = engine._get_hybrid_search()
        if search_obj is not None:
            with patch.object(search_obj, "search", side_effect=mock_search), \
                 patch.object(engine, "_spread_activation", side_effect=mock_spread):
                engine.recall_search_structured("test query", limit=10)
        else:
            engine.graph.fts_search = MagicMock(return_value=[
                {"id": id_a, "score": 5.0},
            ])
            with patch.object(engine, "_spread_activation", side_effect=mock_spread):
                engine.recall_search_structured("test query", limit=10)

        node_conflict_after = engine.file_store.load(id_conflict)
        assert node_conflict_after.access_count == initial_conflict

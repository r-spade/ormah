"""Tests for temporal query filters (created_after / created_before)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from ormah.config import Settings
from ormah.embeddings.hybrid_search import HybridSearch


def _make_node(node_id, created="2026-01-15T00:00:00Z", node_type="fact"):
    return {
        "id": node_id,
        "type": node_type,
        "tier": "working",
        "content": f"content of {node_id}",
        "created": created,
    }


@pytest.fixture
def mock_hybrid(tmp_path):
    """HybridSearch with mocked internals — no real DB or encoder."""
    settings = Settings(
        memory_dir=tmp_path,
        fts_weight=0.4,
        vector_weight=0.6,
        similarity_threshold=0.4,
        rrf_k=60,
    )

    nodes = {
        "old": _make_node("old", created="2026-01-01T00:00:00Z"),
        "mid": _make_node("mid", created="2026-01-15T00:00:00Z"),
        "new": _make_node("new", created="2026-02-01T00:00:00Z"),
    }

    with (
        patch("ormah.embeddings.hybrid_search.GraphIndex") as MockGraph,
        patch("ormah.embeddings.hybrid_search.VectorStore"),
        patch("ormah.embeddings.hybrid_search.get_encoder"),
    ):
        conn = MagicMock()
        hs = HybridSearch(conn, settings)
        MockGraph.return_value.get_node.side_effect = lambda nid: nodes.get(nid)
        MockGraph.return_value.get_nodes_batch.side_effect = lambda ids: {
            nid: nodes[nid] for nid in ids if nid in nodes
        }
        MockGraph.return_value.get_tags_batch.side_effect = lambda ids: {}
        hs.graph = MockGraph.return_value
    return hs


def _run_search(hs, **kwargs):
    """Run search with all three nodes returned by both retrievers."""
    hs.graph.fts_search.return_value = [
        {"id": "old", "score": 10.0},
        {"id": "mid", "score": 8.0},
        {"id": "new", "score": 6.0},
    ]
    hs.vec_store.search.return_value = [
        {"id": "old", "similarity": 0.9},
        {"id": "mid", "similarity": 0.8},
        {"id": "new", "similarity": 0.7},
    ]
    hs.encoder.encode.return_value = "fake_vec"
    return hs.search("test query", limit=10, **kwargs)


class TestCreatedAfter:
    def test_created_after_filters_old(self, mock_hybrid):
        results = _run_search(mock_hybrid, created_after="2026-01-10T00:00:00Z")
        ids = [r["node"]["id"] for r in results]
        assert "old" not in ids
        assert "mid" in ids
        assert "new" in ids


class TestCreatedBefore:
    def test_created_before_filters_new(self, mock_hybrid):
        results = _run_search(mock_hybrid, created_before="2026-01-20T00:00:00Z")
        ids = [r["node"]["id"] for r in results]
        assert "old" in ids
        assert "mid" in ids
        assert "new" not in ids


class TestCreatedRange:
    def test_created_range(self, mock_hybrid):
        results = _run_search(
            mock_hybrid,
            created_after="2026-01-10T00:00:00Z",
            created_before="2026-01-20T00:00:00Z",
        )
        ids = [r["node"]["id"] for r in results]
        assert ids == ["mid"]


class TestNoTemporalFilter:
    def test_no_temporal_filter_returns_all(self, mock_hybrid):
        results = _run_search(mock_hybrid)
        ids = [r["node"]["id"] for r in results]
        assert len(ids) == 3
        assert set(ids) == {"old", "mid", "new"}


class TestTemporalWithTypeFilter:
    def test_temporal_with_type_filter(self, tmp_path):
        """Temporal + type filters should combine with AND semantics."""
        settings = Settings(
            memory_dir=tmp_path,
            fts_weight=0.4,
            vector_weight=0.6,
            similarity_threshold=0.4,
            rrf_k=60,
        )

        nodes = {
            "fact-old": _make_node("fact-old", created="2026-01-01T00:00:00Z", node_type="fact"),
            "decision-new": _make_node("decision-new", created="2026-02-01T00:00:00Z", node_type="decision"),
            "fact-new": _make_node("fact-new", created="2026-02-01T00:00:00Z", node_type="fact"),
        }

        with (
            patch("ormah.embeddings.hybrid_search.GraphIndex") as MockGraph,
            patch("ormah.embeddings.hybrid_search.VectorStore"),
            patch("ormah.embeddings.hybrid_search.get_encoder"),
        ):
            conn = MagicMock()
            hs = HybridSearch(conn, settings)
            MockGraph.return_value.get_node.side_effect = lambda nid: nodes.get(nid)
            MockGraph.return_value.get_nodes_batch.side_effect = lambda ids: {
                nid: nodes[nid] for nid in ids if nid in nodes
            }
            MockGraph.return_value.get_tags_batch.side_effect = lambda ids: {}
            hs.graph = MockGraph.return_value

        hs.graph.fts_search.return_value = [
            {"id": "fact-old", "score": 10.0},
            {"id": "decision-new", "score": 8.0},
            {"id": "fact-new", "score": 6.0},
        ]
        hs.vec_store.search.return_value = [
            {"id": "fact-old", "similarity": 0.9},
            {"id": "decision-new", "similarity": 0.8},
            {"id": "fact-new", "similarity": 0.7},
        ]
        hs.encoder.encode.return_value = "fake_vec"

        results = hs.search(
            "test query",
            limit=10,
            types=["fact"],
            created_after="2026-01-15T00:00:00Z",
        )
        ids = [r["node"]["id"] for r in results]
        # Only "fact-new" is both type=fact AND created after Jan 15
        assert ids == ["fact-new"]

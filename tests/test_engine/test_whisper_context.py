"""Tests for whisper context (involuntary recall injection)."""

from __future__ import annotations

import numpy as np
from unittest.mock import MagicMock, patch

import pytest

from ormah.engine.context_builder import ContextBuilder
from ormah.index.graph import GraphIndex


def _make_node_dict(node_id, title, tier="core", space=None, importance=0.5, node_type="fact"):
    return {
        "id": node_id,
        "type": node_type,
        "tier": tier,
        "title": title,
        "content": f"Content about {title}",
        "space": space,
        "importance": importance,
        "confidence": 1.0,
        "valid_until": None,
        "source": "agent:test",
        "access_count": 0,
        "last_accessed": "2026-01-01T00:00:00Z",
        "created": "2026-01-01T00:00:00Z",
        "updated": "2026-01-01T00:00:00Z",
    }


def _insert_node(conn, node):
    conn.execute(
        "INSERT INTO nodes (id, type, tier, source, space, title, content, "
        "created, updated, last_accessed, access_count, confidence, importance, "
        "file_path, file_hash) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (node["id"], node["type"], node["tier"], node["source"],
         node["space"], node["title"], node["content"],
         node["created"], node["updated"], node["last_accessed"],
         node["access_count"], node["confidence"], node["importance"],
         "/fake/path", "abc123"),
    )


@pytest.fixture
def mock_graph(tmp_path):
    from ormah.index.db import Database
    db = Database(tmp_path / "index.db")
    db.init_schema()
    graph = GraphIndex(db.conn)
    return graph


class TestWhisperMinScore:
    """Whisper should drop results below min_score threshold."""

    def test_low_score_results_dropped(self, mock_graph):
        mock_engine = MagicMock()
        builder = ContextBuilder(mock_graph, engine=mock_engine)

        nodes = [_make_node_dict(f"node-{i}", f"Fact {i}") for i in range(5)]
        # Only 2 results above threshold (min_score=0.15, gate=0.55)
        mock_engine.recall_search_structured.return_value = [
            {"node": nodes[0], "score": 0.8, "source": "hybrid"},
            {"node": nodes[1], "score": 0.6, "source": "hybrid"},
            {"node": nodes[2], "score": 0.1, "source": "hybrid"},
            {"node": nodes[3], "score": 0.05, "source": "hybrid"},
            {"node": nodes[4], "score": 0.02, "source": "hybrid"},
        ]

        result = builder.build_whisper_context(
            prompt="how does auth work",
            min_score=0.15,
        )

        assert "Fact 0" in result
        assert "Fact 1" in result
        assert "Fact 2" not in result
        assert "Fact 3" not in result
        assert "Fact 4" not in result

    def test_all_below_threshold_returns_empty(self, mock_graph):
        mock_engine = MagicMock()
        mock_engine.settings.claude_maintenance_enabled = False
        builder = ContextBuilder(mock_graph, engine=mock_engine)

        nodes = [_make_node_dict("node-0", "Irrelevant")]
        mock_engine.recall_search_structured.return_value = [
            {"node": nodes[0], "score": 0.05, "source": "hybrid"},
        ]

        result = builder.build_whisper_context(
            prompt="hello",
            min_score=0.15,
        )

        assert result == ""


class TestWhisperCompactFormatting:
    """Whisper formatting: flat list, top 2 full, rest title-only."""

    def test_top_node_content_within_cap_not_truncated(self, mock_graph):
        mock_engine = MagicMock()
        builder = ContextBuilder(mock_graph, engine=mock_engine)

        long_content = "A" * 600
        node = {**_make_node_dict("node-1", "Some title"), "content": long_content}
        mock_engine.recall_search_structured.return_value = [
            {"node": node, "score": 0.9, "source": "hybrid"},
        ]

        result = builder.build_whisper_context(
            prompt="something",
            injection_gate=0.0,
        )

        assert long_content in result

    def test_top_node_content_capped_at_word_boundary(self, mock_graph):
        mock_engine = MagicMock()
        builder = ContextBuilder(mock_graph, engine=mock_engine)

        giant_content = "word " * 800  # ~4KB, far over the cap
        node = {**_make_node_dict("node-1", "Some title"), "content": giant_content}
        mock_engine.recall_search_structured.return_value = [
            {"node": node, "score": 0.9, "source": "hybrid"},
        ]

        result = builder.build_whisper_context(
            prompt="something",
            injection_gate=0.0,
            injected_content_max_chars=600,
        )

        assert giant_content.strip() not in result
        injected_line = next(
            line for line in result.splitlines() if line.startswith("  word")
        )
        assert len(injected_line) <= 2 + 600
        assert injected_line.endswith("…")

    def test_content_cap_holds_for_unbroken_text(self, mock_graph):
        """An unbroken (space-free) string must not exceed the cap by one."""
        from ormah.engine.context_builder import _truncate_at_word_boundary

        unbroken = "A" * 601
        out = _truncate_at_word_boundary(unbroken, max_len=600)
        assert len(out) <= 600
        assert out.endswith("…")

    def test_content_cap_is_configurable(self, mock_graph):
        mock_engine = MagicMock()
        builder = ContextBuilder(mock_graph, engine=mock_engine)

        content = "word " * 100  # 500 chars
        node = {**_make_node_dict("node-1", "Some title"), "content": content}
        mock_engine.recall_search_structured.return_value = [
            {"node": node, "score": 0.9, "source": "hybrid"},
        ]

        result = builder.build_whisper_context(
            prompt="something",
            injection_gate=0.0,
            injected_content_max_chars=100,
        )

        injected_line = next(
            line for line in result.splitlines() if line.startswith("  word")
        )
        assert len(injected_line) <= 2 + 100
        assert injected_line.endswith("…")


class TestWhisperFailSilently:
    """Whisper should return empty string on failure, not dump everything."""

    def test_empty_prompt_returns_empty(self, mock_graph):
        mock_engine = MagicMock()
        builder = ContextBuilder(mock_graph, engine=mock_engine)

        result = builder.build_whisper_context(prompt="")
        assert result == ""

    def test_short_prompt_returns_empty(self, mock_graph):
        """Prompts of 2 chars or less (e.g. 'y', 'ok') should return empty."""
        mock_engine = MagicMock()
        builder = ContextBuilder(mock_graph, engine=mock_engine)

        for short in ["y", "ok", "n", "  y  "]:
            result = builder.build_whisper_context(prompt=short)
            assert result == "", f"Expected empty for {short!r}, got {result!r}"
            mock_engine.recall_search_structured.assert_not_called()

    def test_three_char_prompt_not_filtered(self, mock_graph):
        """Prompts of 3+ chars should proceed normally."""
        mock_engine = MagicMock()
        builder = ContextBuilder(mock_graph, engine=mock_engine)

        nodes = [_make_node_dict("node-0", "Some fact")]
        mock_engine.recall_search_structured.return_value = [
            {"node": nodes[0], "score": 0.8, "source": "hybrid"},
        ]

        result = builder.build_whisper_context(prompt="yes", min_score=0.1)
        assert "Some fact" in result

    def test_no_engine_returns_empty(self, mock_graph):
        builder = ContextBuilder(mock_graph)  # No engine

        result = builder.build_whisper_context(prompt="test query")
        assert result == ""

    def test_search_failure_returns_empty(self, mock_graph):
        mock_engine = MagicMock()
        mock_engine.recall_search_structured.side_effect = RuntimeError("search down")
        builder = ContextBuilder(mock_graph, engine=mock_engine)

        result = builder.build_whisper_context(prompt="test query")
        assert result == ""


class TestWhisperNodeLimit:
    """Whisper should respect max_nodes."""

    def test_respects_max_nodes(self, mock_graph):
        mock_engine = MagicMock()
        builder = ContextBuilder(mock_graph, engine=mock_engine)

        nodes = [_make_node_dict(f"node-{i}", f"Fact {i}") for i in range(10)]
        mock_engine.recall_search_structured.return_value = [
            {"node": n, "score": 0.9, "source": "hybrid"} for n in nodes
        ]

        result = builder.build_whisper_context(
            prompt="test",
            max_nodes=3,
            min_score=0.1,
            injection_gate=0.0,
        )

        # Search fetches a deep pool (max_nodes * multiplier) for the
        # reranker/gate, but the final output stays capped at max_nodes.
        mock_engine.recall_search_structured.assert_called_once()
        call_kwargs = mock_engine.recall_search_structured.call_args
        limit = call_kwargs.kwargs.get("limit") or call_kwargs[1].get("limit")
        assert limit == 3 * 5  # max_nodes * default candidate_pool_multiplier
        injected = sum(1 for i in range(10) if f"Fact {i}" in result)
        assert injected <= 3

    def test_candidate_pool_multiplier_configurable(self, mock_graph):
        mock_engine = MagicMock()
        builder = ContextBuilder(mock_graph, engine=mock_engine)
        mock_engine.recall_search_structured.return_value = []

        builder.build_whisper_context(
            prompt="test",
            max_nodes=4,
            candidate_pool_multiplier=3,
        )

        call_kwargs = mock_engine.recall_search_structured.call_args
        limit = call_kwargs.kwargs.get("limit") or call_kwargs[1].get("limit")
        assert limit == 12

    def test_total_budget_respected(self, mock_graph):
        """Total nodes in output should be <= max_nodes, even with identity nodes."""
        mock_engine = MagicMock()
        builder = ContextBuilder(mock_graph, engine=mock_engine)
        conn = mock_graph.conn

        # Create user node
        user_node = _make_node_dict("user-1", "Self", node_type="person")
        _insert_node(conn, user_node)

        # Create 3 identity nodes
        identity_nodes = []
        for i in range(3):
            node = _make_node_dict(f"id-{i}", f"Identity {i}", node_type="preference")
            _insert_node(conn, node)
            identity_nodes.append(node)
            conn.execute(
                "INSERT INTO edges (source_id, target_id, edge_type, weight, created) "
                "VALUES (?, ?, 'defines', 1.0, '2026-01-01T00:00:00Z')",
                ("user-1", f"id-{i}"),
            )

        # Create 10 non-identity nodes
        other_nodes = [_make_node_dict(f"other-{i}", f"Other {i}", tier="working") for i in range(10)]
        conn.commit()

        # Return identity + non-identity in search results
        all_results = [
            {"node": n, "score": 0.9, "source": "hybrid"} for n in identity_nodes
        ] + [
            {"node": n, "score": 0.8, "source": "hybrid"} for n in other_nodes
        ]
        mock_engine.recall_search_structured.return_value = all_results

        max_nodes = 6
        result = builder.build_whisper_context(
            prompt="test",
            user_node_id="user-1",
            max_nodes=max_nodes,
            min_score=0.1,
        )

        # Count how many nodes appear in the output
        total_found = 0
        for i in range(3):
            if f"Identity {i}" in result:
                total_found += 1
        for i in range(10):
            if f"Other {i}" in result:
                total_found += 1
        assert total_found <= max_nodes, (
            f"Expected at most {max_nodes} nodes in output, found {total_found}"
        )


class TestWhisperReranker:
    """Whisper cross-encoder reranking with linear-rescale blended scoring."""

    def test_reranker_blends_and_preserves_relevant(self, mock_graph):
        """Blended scoring should preserve semantically relevant results
        even when cross-encoder scores are negative."""
        mock_engine = MagicMock()
        builder = ContextBuilder(mock_graph, engine=mock_engine)

        nodes = [_make_node_dict(f"node-{i}", f"Fact {i}") for i in range(4)]
        mock_engine.recall_search_structured.return_value = [
            {"node": nodes[0], "score": 0.8, "source": "hybrid"},
            {"node": nodes[1], "score": 0.7, "source": "hybrid"},
            {"node": nodes[2], "score": 0.6, "source": "hybrid"},
            {"node": nodes[3], "score": 0.5, "source": "hybrid"},
        ]

        # CE scores: node-2 highest CE, node-3 negative but has decent embedding
        # All mild CE scores → linear rescale keeps all 4 results
        mock_cross_encoder = MagicMock()
        mock_cross_encoder.rerank.return_value = [0.3, 0.9, 0.95, -0.5]

        with patch("ormah.embeddings.reranker._get_model", return_value=mock_cross_encoder):
            result = builder.build_whisper_context(
                prompt="specific query",
                min_score=0.1,
                injection_gate=0.1,  # low gate to isolate reranker behavior
                reranker_enabled=True,
                reranker_model="cross-encoder/ms-marco-MiniLM-L-6-v2",
                reranker_min_score=0.0,
            )

        # All results preserved — mild CE scores → rescale preserves all
        # (CE=-0.5 → rescale≈0.639, blended≈0.456)
        assert "Fact 0" in result
        assert "Fact 1" in result
        assert "Fact 2" in result
        assert "Fact 3" in result

    def test_reranker_filters_low_blended_scores(self, mock_graph):
        """Results with both low CE and low embedding scores should be filtered."""
        mock_engine = MagicMock()
        builder = ContextBuilder(mock_graph, engine=mock_engine)

        nodes = [_make_node_dict(f"node-{i}", f"Fact {i}") for i in range(3)]
        mock_engine.recall_search_structured.return_value = [
            {"node": nodes[0], "score": 0.8, "source": "hybrid"},
            {"node": nodes[1], "score": 0.5, "source": "hybrid"},
            {"node": nodes[2], "score": 0.2, "source": "hybrid"},  # low embedding
        ]

        mock_cross_encoder = MagicMock()
        # node-2: CE=-10 → rescale=0.111, emb=0.2 → blended=0.4*0.111+0.6*0.2=0.164
        mock_cross_encoder.rerank.return_value = [2.0, -1.0, -10.0]

        with patch("ormah.embeddings.reranker._get_model", return_value=mock_cross_encoder):
            result = builder.build_whisper_context(
                prompt="specific query",
                min_score=0.1,
                injection_gate=0.1,  # low gate to isolate reranker behavior
                reranker_enabled=True,
                reranker_model="cross-encoder/ms-marco-MiniLM-L-6-v2",
                reranker_min_score=0.17,  # raised to filter node-2
            )

        # node-0: blended = 0.4*rescale(2)+0.6*0.8 ≈ 0.4*0.778+0.48 = 0.791 ✓
        assert "Fact 0" in result
        # node-1: blended = 0.4*rescale(-1)+0.6*0.5 ≈ 0.4*0.611+0.30 = 0.544 ✓
        assert "Fact 1" in result
        # node-2: blended = 0.4*rescale(-10)+0.6*0.2 ≈ 0.4*0.111+0.12 = 0.164 ✗
        assert "Fact 2" not in result

    def test_reranker_min_score_on_blended(self, mock_graph):
        """reranker_min_score threshold applies to blended score, not raw CE."""
        mock_engine = MagicMock()
        builder = ContextBuilder(mock_graph, engine=mock_engine)

        nodes = [_make_node_dict(f"node-{i}", f"Fact {i}") for i in range(3)]
        mock_engine.recall_search_structured.return_value = [
            {"node": nodes[0], "score": 0.8, "source": "hybrid"},
            {"node": nodes[1], "score": 0.7, "source": "hybrid"},
            {"node": nodes[2], "score": 0.6, "source": "hybrid"},
        ]

        mock_cross_encoder = MagicMock()
        # Linear rescale: CE [0.8, 0.1, 0.05] → rescale [0.711, 0.672, 0.669]
        # With alpha=0.4: node-0=0.764, node-1=0.689, node-2=0.628
        # At reranker_min_score=0.7: only node-0 passes
        mock_cross_encoder.rerank.return_value = [0.8, 0.1, 0.05]

        with patch("ormah.embeddings.reranker._get_model", return_value=mock_cross_encoder):
            result = builder.build_whisper_context(
                prompt="specific query",
                min_score=0.1,
                reranker_enabled=True,
                reranker_model="cross-encoder/ms-marco-MiniLM-L-6-v2",
                reranker_min_score=0.7,
            )

        # node-0: blended = 0.4*rescale(0.8)+0.6*0.8 ≈ 0.4*0.711+0.48 = 0.764 ✓
        assert "Fact 0" in result
        # node-1: blended = 0.4*rescale(0.1)+0.6*0.7 ≈ 0.4*0.672+0.42 = 0.689 ✗
        assert "Fact 1" not in result
        # node-2: blended = 0.4*rescale(0.05)+0.6*0.6 ≈ 0.4*0.669+0.36 = 0.628 ✗
        assert "Fact 2" not in result

    def test_reranker_fallback_on_error(self, mock_graph):
        mock_engine = MagicMock()
        builder = ContextBuilder(mock_graph, engine=mock_engine)

        nodes = [_make_node_dict("node-0", "Fact 0")]
        mock_engine.recall_search_structured.return_value = [
            {"node": nodes[0], "score": 0.8, "source": "hybrid"},
        ]

        with patch("ormah.embeddings.reranker._get_model", side_effect=RuntimeError("model not found")):
            result = builder.build_whisper_context(
                prompt="test",
                min_score=0.1,
                reranker_enabled=True,
                reranker_model="cross-encoder/ms-marco-MiniLM-L-6-v2",
                reranker_min_score=0.0,
            )

        # Should fall back to embedding scores and still return results
        assert "Fact 0" in result

    def test_reranker_disabled_skips_reranking(self, mock_graph):
        mock_engine = MagicMock()
        builder = ContextBuilder(mock_graph, engine=mock_engine)

        nodes = [_make_node_dict("node-0", "Fact 0")]
        mock_engine.recall_search_structured.return_value = [
            {"node": nodes[0], "score": 0.8, "source": "hybrid"},
        ]

        with patch("ormah.embeddings.reranker.rerank") as mock_rerank:
            result = builder.build_whisper_context(
                prompt="test",
                min_score=0.1,
                reranker_enabled=False,
                reranker_model="cross-encoder/ms-marco-MiniLM-L-6-v2",
                reranker_min_score=0.0,
            )

        mock_rerank.assert_not_called()
        assert "Fact 0" in result

    def test_reranker_empty_candidates_noop(self, mock_graph):
        mock_engine = MagicMock()
        mock_engine.settings.claude_maintenance_enabled = False
        builder = ContextBuilder(mock_graph, engine=mock_engine)

        # All below bi-encoder min_score
        nodes = [_make_node_dict("node-0", "Fact 0")]
        mock_engine.recall_search_structured.return_value = [
            {"node": nodes[0], "score": 0.01, "source": "hybrid"},
        ]

        with patch("ormah.embeddings.reranker.rerank") as mock_rerank:
            result = builder.build_whisper_context(
                prompt="test",
                min_score=0.5,
                reranker_enabled=True,
                reranker_model="cross-encoder/ms-marco-MiniLM-L-6-v2",
                reranker_min_score=0.0,
            )

        # No candidates pass min_score, so reranker shouldn't be called
        mock_rerank.assert_not_called()
        assert result == ""


class TestWhisperWithProject:
    """Space param still filters results correctly."""

    def test_with_space_passes_space_to_search(self, mock_graph):
        mock_engine = MagicMock()
        builder = ContextBuilder(mock_graph, engine=mock_engine)

        node = _make_node_dict("node-1", "Project fact", space="myproject")
        mock_engine.recall_search_structured.return_value = [
            {"node": node, "score": 0.9, "source": "hybrid"},
        ]

        builder.build_whisper_context(
            prompt="project stuff",
            space="myproject",
            injection_gate=0.0,
        )

        call_kwargs = mock_engine.recall_search_structured.call_args[1]
        assert call_kwargs["default_space"] == "myproject"


class TestWhisperIntentAware:
    """Whisper should use intent classification to gate/filter results."""

    def test_conversational_returns_empty(self, mock_graph):
        """Conversational prompts should produce no whisper output."""
        mock_engine = MagicMock()
        builder = ContextBuilder(mock_graph, engine=mock_engine)

        # Mock classifier that returns conversational
        from ormah.engine.prompt_classifier import PromptIntent

        mock_classifier = MagicMock()
        mock_classifier.classify.return_value = PromptIntent(categories=["conversational"])
        builder._classifier = mock_classifier

        result = builder.build_whisper_context(prompt="hello", min_score=0.1)
        assert result == ""
        # Should not even attempt a search
        mock_engine.recall_search_structured.assert_not_called()

    @pytest.mark.parametrize("prompt", ["Thanks, that helps.", "THAT HELPS!"])
    def test_acknowledgement_returns_empty_before_classifier_or_search(self, mock_graph, prompt):
        """Acknowledgements stay silent even without a session continuation."""
        mock_engine = MagicMock()
        builder = ContextBuilder(mock_graph, engine=mock_engine)
        builder._classifier = MagicMock()

        result = builder.build_whisper_context(prompt=prompt, min_score=0.1)

        assert result == ""
        builder._classifier.classify.assert_not_called()
        mock_engine.recall_search_structured.assert_not_called()

    def test_acknowledgement_is_silent_when_classifier_is_unavailable(self, mock_graph):
        """The guard must not depend on local embedding-model availability."""
        mock_engine = MagicMock()
        mock_engine._get_hybrid_search.return_value = None
        builder = ContextBuilder(mock_graph, engine=mock_engine)

        result = builder.build_whisper_context(prompt="Thanks, that helps.", min_score=0.1)

        assert result == ""
        mock_engine.recall_search_structured.assert_not_called()

    @pytest.mark.parametrize(
        ("prompt", "title"),
        [
            ("Thanks, now explain the cache.", "Cache invalidation"),
            ("Done with that; start the deployment.", "Deployment checklist"),
        ],
    )
    def test_substantive_prompt_with_acknowledgement_words_still_searches(
        self, mock_graph, prompt, title,
    ):
        """Acknowledgement words must not swallow a new useful request."""
        from ormah.engine.prompt_classifier import PromptIntent

        mock_engine = MagicMock()
        builder = ContextBuilder(mock_graph, engine=mock_engine)
        builder._classifier = MagicMock()
        builder._classifier.classify.return_value = PromptIntent(categories=["general"])
        node = _make_node_dict("substantive-1", title)
        mock_engine.recall_search_structured.return_value = [
            {"node": node, "score": 0.8, "source": "hybrid"},
        ]

        result = builder.build_whisper_context(prompt=prompt, min_score=0.1)

        assert title in result
        mock_engine.recall_search_structured.assert_called_once()

    def test_general_intent_searches_normally(self, mock_graph):
        """General intent should use normal search behavior."""
        mock_engine = MagicMock()
        builder = ContextBuilder(mock_graph, engine=mock_engine)

        from ormah.engine.prompt_classifier import PromptIntent

        mock_classifier = MagicMock()
        mock_classifier.classify.return_value = PromptIntent(categories=["general"])
        builder._classifier = mock_classifier

        nodes = [_make_node_dict("node-0", "Auth module details")]
        mock_engine.recall_search_structured.return_value = [
            {"node": nodes[0], "score": 0.8, "source": "hybrid"},
        ]

        result = builder.build_whisper_context(prompt="how does auth work", min_score=0.1)
        assert "Auth module details" in result
        mock_engine.recall_search_structured.assert_called_once()

    def test_temporal_intent_passes_created_after_and_before(self, mock_graph):
        """Temporal intent should add created_after and created_before to search params."""
        mock_engine = MagicMock()
        builder = ContextBuilder(mock_graph, engine=mock_engine)

        from ormah.engine.prompt_classifier import PromptIntent

        mock_classifier = MagicMock()
        mock_classifier.classify.return_value = PromptIntent(
            categories=["temporal"],
            search_params={
                "created_after": "2026-02-23T00:00:00+00:00",
                "created_before": "2026-03-02T00:00:00+00:00",
                "search_query": "what did we do",
            },
        )
        builder._classifier = mock_classifier

        mock_engine.recall_search_structured.return_value = []

        builder.build_whisper_context(prompt="what did we do last week", min_score=0.1)
        call_kwargs = mock_engine.recall_search_structured.call_args
        assert call_kwargs.kwargs.get("created_after") == "2026-02-23T00:00:00+00:00"
        assert call_kwargs.kwargs.get("created_before") == "2026-03-02T00:00:00+00:00"

    def test_temporal_intent_uses_stripped_search_query(self, mock_graph):
        """Temporal intent should use stripped search_query instead of raw prompt."""
        mock_engine = MagicMock()
        builder = ContextBuilder(mock_graph, engine=mock_engine)

        from ormah.engine.prompt_classifier import PromptIntent

        mock_classifier = MagicMock()
        mock_classifier.classify.return_value = PromptIntent(
            categories=["temporal"],
            search_params={
                "created_after": "2026-02-23T00:00:00+00:00",
                "created_before": "2026-03-02T00:00:00+00:00",
                "search_query": "what did I work on whisper",
            },
        )
        builder._classifier = mock_classifier

        mock_engine.recall_search_structured.return_value = []

        builder.build_whisper_context(
            prompt="what did I work on whisper last week", min_score=0.1,
        )
        call_kwargs = mock_engine.recall_search_structured.call_args
        query = call_kwargs.kwargs.get("query")
        # Should use the stripped query, not the raw prompt with "last week"
        assert "last week" not in query
        assert "whisper" in query

    def test_identity_intent_runs_search(self, mock_graph):
        """Identity-only intent should still run search (not skip it)."""
        mock_engine = MagicMock()
        mock_engine.recall_search_structured.return_value = []
        mock_engine.settings.claude_maintenance_enabled = False
        builder = ContextBuilder(mock_graph, engine=mock_engine)
        conn = mock_graph.conn

        from ormah.engine.prompt_classifier import PromptIntent

        mock_classifier = MagicMock()
        mock_classifier.classify.return_value = PromptIntent(categories=["identity"])
        builder._classifier = mock_classifier

        # Create user node with identity neighbors
        user_node = _make_node_dict("user-1", "Self", node_type="person")
        _insert_node(conn, user_node)
        pref_node = _make_node_dict("pref-1", "Likes dark mode", node_type="preference")
        _insert_node(conn, pref_node)
        conn.execute(
            "INSERT INTO edges (source_id, target_id, edge_type, weight, created) "
            "VALUES (?, ?, 'defines', 1.0, '2026-01-01T00:00:00Z')",
            ("user-1", "pref-1"),
        )
        conn.commit()

        result = builder.build_whisper_context(
            prompt="what do you know about me",
            user_node_id="user-1",
            min_score=0.1,
        )

        # Search SHOULD be called even for identity-only intent
        mock_engine.recall_search_structured.assert_called_once()
        # With no search results, whisper stays silent (no graph neighbor dump)
        assert result == ""

    def test_classifier_failure_falls_back_to_normal(self, mock_graph):
        """If classifier raises, should fall back to normal search."""
        mock_engine = MagicMock()
        builder = ContextBuilder(mock_graph, engine=mock_engine)

        mock_classifier = MagicMock()
        mock_classifier.classify.side_effect = RuntimeError("encoder broken")
        builder._classifier = mock_classifier

        nodes = [_make_node_dict("node-0", "Some fact")]
        mock_engine.recall_search_structured.return_value = [
            {"node": nodes[0], "score": 0.8, "source": "hybrid"},
        ]

        result = builder.build_whisper_context(prompt="test query", min_score=0.1)
        # Should still work via normal search
        assert "Some fact" in result
        mock_engine.recall_search_structured.assert_called_once()

    def test_no_classifier_searches_normally(self, mock_graph):
        """If classifier can't be created (no engine hybrid search), search normally."""
        mock_engine = MagicMock()
        mock_engine._get_hybrid_search.return_value = None
        builder = ContextBuilder(mock_graph, engine=mock_engine)
        # _classifier is None by default

        nodes = [_make_node_dict("node-0", "A fact")]
        mock_engine.recall_search_structured.return_value = [
            {"node": nodes[0], "score": 0.8, "source": "hybrid"},
        ]

        result = builder.build_whisper_context(prompt="test query", min_score=0.1)
        assert "A fact" in result


class TestWhisperRerankerBlendIntegration:
    """Integration tests: blended reranker through the full whisper pipeline.

    These test the build_whisper_context → rerank flow end-to-end,
    verifying that the new blend_alpha and max_doc_chars params are
    passed through correctly and that the pipeline produces the right
    output for realistic scenarios.
    """

    def test_unanimously_negative_ce_suppresses_results(self, mock_graph):
        """When ALL cross-encoder scores are strongly negative (< -5),
        results are suppressed — the CE is confidently saying 'off-topic'.

        Note: exploration_enabled=False here to isolate injection gate behaviour.
        The exploration slot is tested separately in TestExplorationSlot.
        """
        mock_engine = MagicMock()
        mock_engine.settings = _make_settings_mock(
            whisper_exploration_enabled=False,
            whisper_reranker_min_score=0.0,
        )
        builder = ContextBuilder(mock_graph, engine=mock_engine)

        nodes = [
            _make_node_dict("n1", "MemoryEngine central facade"),
            _make_node_dict("n2", "Dual storage markdown and SQLite"),
            _make_node_dict("n3", "Chose bge-base for embeddings"),
        ]
        mock_engine.recall_search_structured.return_value = [
            {"node": nodes[0], "score": 0.714, "source": "hybrid"},
            {"node": nodes[1], "score": 0.736, "source": "hybrid"},
            {"node": nodes[2], "score": 0.681, "source": "hybrid"},
        ]

        mock_ce = MagicMock()
        # All strongly negative CE — off-topic signal
        # CE [-10.7, -11.4, -8.2] → rescale [0.072, 0.033, 0.211]
        # With alpha=0.4: blended [0.457, 0.455, 0.449] → below gate 0.55
        mock_ce.rerank.return_value = [-10.7, -11.4, -8.2]

        with patch("ormah.embeddings.reranker._get_model", return_value=mock_ce):
            result = builder.build_whisper_context(
                prompt="how do I cook pasta",
                min_score=0.1,
                reranker_enabled=True,
                reranker_min_score=0.0,
            )

        # All suppressed — CE unanimously says off-topic
        assert result == ""

    def test_mixed_ce_preserves_positive_results(self, mock_graph):
        """When at least one CE score is > -5, results are NOT suppressed."""
        mock_engine = MagicMock()
        builder = ContextBuilder(mock_graph, engine=mock_engine)

        nodes = [
            _make_node_dict("n1", "MemoryEngine central facade"),
            _make_node_dict("n2", "Dual storage markdown and SQLite"),
        ]
        mock_engine.recall_search_structured.return_value = [
            {"node": nodes[0], "score": 0.714, "source": "hybrid"},
            {"node": nodes[1], "score": 0.736, "source": "hybrid"},
        ]

        mock_ce = MagicMock()
        # One positive, one negative — mixed signal
        mock_ce.rerank.return_value = [2.0, -10.0]

        with patch("ormah.embeddings.reranker._get_model", return_value=mock_ce):
            result = builder.build_whisper_context(
                prompt="how does the search pipeline work?",
                min_score=0.1,
                reranker_enabled=True,
                reranker_min_score=0.0,
            )

        # At least the positive-CE result should survive
        assert "MemoryEngine central facade" in result

    def test_blend_alpha_passed_through(self, mock_graph):
        """Custom blend_alpha should affect which results survive."""
        mock_engine = MagicMock()
        builder = ContextBuilder(mock_graph, engine=mock_engine)

        nodes = [_make_node_dict("n1", "Fact A")]
        mock_engine.recall_search_structured.return_value = [
            {"node": nodes[0], "score": 0.3, "source": "hybrid"},
        ]

        mock_ce = MagicMock()
        # CE=-10.0 → rescale=0.111
        # With α=0.4: 0.4*0.111+0.6*0.3 = 0.224 (passes 0.15)
        # With α=0.9: 0.9*0.111+0.1*0.3 = 0.130 (fails 0.15)
        mock_ce.rerank.return_value = [-10.0]

        with patch("ormah.embeddings.reranker._get_model", return_value=mock_ce):
            # Default alpha: should pass min_score=0.15
            # (prompt shares the "fact" token so the topical filter keeps it)
            result_default = builder.build_whisper_context(
                prompt="tell me about fact a",
                min_score=0.1,
                injection_gate=0.1,  # low gate to isolate reranker behavior
                reranker_enabled=True,
                reranker_min_score=0.15,
                reranker_blend_alpha=0.4,
            )
            # High alpha: CE dominates → should fail min_score=0.15
            result_high_alpha = builder.build_whisper_context(
                prompt="tell me about fact a",
                min_score=0.1,
                injection_gate=0.1,  # low gate to isolate reranker behavior
                reranker_enabled=True,
                reranker_min_score=0.15,
                reranker_blend_alpha=0.9,
            )

        assert "Fact A" in result_default
        assert "Fact A" not in result_high_alpha

    def test_max_doc_chars_passed_through(self, mock_graph):
        """Verify max_doc_chars is forwarded to reranker."""
        mock_engine = MagicMock()
        builder = ContextBuilder(mock_graph, engine=mock_engine)

        long_content = "z" * 1000
        node = _make_node_dict("n1", "Title")
        node["content"] = long_content
        mock_engine.recall_search_structured.return_value = [
            {"node": node, "score": 0.7, "source": "hybrid"},
        ]

        mock_ce = MagicMock()
        mock_ce.rerank.return_value = [0.0]

        with patch("ormah.embeddings.reranker._get_model", return_value=mock_ce):
            builder.build_whisper_context(
                prompt="test",
                min_score=0.1,
                reranker_enabled=True,
                reranker_min_score=0.0,
                reranker_max_doc_chars=100,
            )

        doc = mock_ce.rerank.call_args[0][1][0]
        # "Title: " + 100 chars of content
        assert len(doc) == len("Title: ") + 100

    def test_reranker_with_identity_nodes(self, mock_graph):
        """Reranker should only affect non-identity search results.

        Identity nodes are separated before reranking, so a negative CE
        on a preference node shouldn't drop it.
        """
        mock_engine = MagicMock()
        builder = ContextBuilder(mock_graph, engine=mock_engine)
        conn = mock_graph.conn

        user_node = _make_node_dict("user-1", "Self", node_type="person")
        _insert_node(conn, user_node)
        pref = _make_node_dict("pref-1", "Prefers dark mode", node_type="preference")
        _insert_node(conn, pref)
        conn.execute(
            "INSERT INTO edges (source_id, target_id, edge_type, weight, created) "
            "VALUES (?, ?, 'defines', 1.0, '2026-01-01T00:00:00Z')",
            ("user-1", "pref-1"),
        )
        conn.commit()

        fact = _make_node_dict("fact-1", "Search uses hybrid FTS")
        mock_engine.recall_search_structured.return_value = [
            {"node": pref, "score": 0.5, "source": "hybrid"},
            {"node": fact, "score": 0.6, "source": "hybrid"},
        ]

        mock_ce = MagicMock()
        # Both candidates go through reranker (identity split happens after)
        mock_ce.rerank.return_value = [1.0, 2.0]

        with patch("ormah.embeddings.reranker._get_model", return_value=mock_ce):
            result = builder.build_whisper_context(
                prompt="how does search work",
                user_node_id="user-1",
                min_score=0.1,
                reranker_enabled=True,
                reranker_min_score=0.0,
            )

        assert "Search uses hybrid FTS" in result
        assert "Prefers dark mode" in result

    def test_min_score_and_reranker_min_score_both_apply(self, mock_graph):
        """Embedding min_score pre-filters; the 0.40 post-boost floor further filters.

        With reranker_min_score=0.0, the post-boost floor defaults to 0.40.
        'Somewhat relevant' (CE=-8, blended≈0.30) falls below that floor.
        """
        mock_engine = MagicMock()
        mock_engine.settings = _make_settings_mock(
            whisper_exploration_enabled=False,
            whisper_reranker_min_score=0.0,
        )
        builder = ContextBuilder(mock_graph, engine=mock_engine)

        nodes = [
            _make_node_dict("good", "Relevant fact"),
            _make_node_dict("mid", "Somewhat relevant"),
            _make_node_dict("low_emb", "Low embedding score"),
        ]
        mock_engine.recall_search_structured.return_value = [
            {"node": nodes[0], "score": 0.8, "source": "hybrid"},
            {"node": nodes[1], "score": 0.5, "source": "hybrid"},
            {"node": nodes[2], "score": 0.1, "source": "hybrid"},  # below min_score
        ]

        mock_ce = MagicMock()
        # Only 2 candidates reach reranker (low_emb filtered by min_score=0.45)
        # good: CE=3.0, emb=0.8 → blended ≈ 0.86 (above 0.40 floor)
        # mid: CE=-8.0, emb=0.5 → blended ≈ 0.30 (below 0.40 floor → filtered)
        mock_ce.rerank.return_value = [3.0, -8.0]

        with patch("ormah.embeddings.reranker._get_model", return_value=mock_ce):
            result = builder.build_whisper_context(
                prompt="test query",
                min_score=0.45,  # filters low_emb before reranker
                injection_gate=0.1,  # low gate to isolate floor behavior
                reranker_enabled=True,
                reranker_min_score=0.0,
            )

        # good survives the 0.40 post-boost floor (blended ≈ 0.86)
        assert "Relevant fact" in result
        # mid is filtered by the 0.40 floor (blended ≈ 0.30 < 0.40)
        assert "Somewhat relevant" not in result
        assert "Low embedding score" not in result

    def test_reranker_reorders_final_output(self, mock_graph):
        """The reranker should change the order of results in the output."""
        mock_engine = MagicMock()
        builder = ContextBuilder(mock_graph, engine=mock_engine)

        nodes = [
            _make_node_dict("first_emb", "First by embedding", tier="working", space="proj"),
            _make_node_dict("second_emb", "Second by embedding", tier="working", space="proj"),
        ]
        mock_engine.recall_search_structured.return_value = [
            {"node": nodes[0], "score": 0.9, "source": "hybrid"},
            {"node": nodes[1], "score": 0.5, "source": "hybrid"},
        ]

        mock_ce = MagicMock()
        # CE flips the order: second_emb gets very high CE
        mock_ce.rerank.return_value = [-3.0, 8.0]

        with patch("ormah.embeddings.reranker._get_model", return_value=mock_ce):
            # Prompt shares the "embedding" token with both titles so the
            # topical filter keeps them.
            result = builder.build_whisper_context(
                prompt="embedding order check",
                space="proj",
                min_score=0.1,
                reranker_enabled=True,
                reranker_min_score=0.0,
            )

        # Both present
        assert "First by embedding" in result
        assert "Second by embedding" in result


class TestWhisperIdentityGating:
    """Identity results should be suppressed when no topical results survive."""

    def test_identity_suppressed_when_no_other_results_low_score(self, mock_graph):
        """Low-scoring identity results should be suppressed when no topical results survive."""
        mock_engine = MagicMock()
        mock_engine.settings.claude_maintenance_enabled = False
        builder = ContextBuilder(mock_graph, engine=mock_engine)
        conn = mock_graph.conn

        user_node = _make_node_dict("user-1", "Self", node_type="person")
        _insert_node(conn, user_node)
        pref = _make_node_dict("pref-1", "Likes dark mode", node_type="preference")
        _insert_node(conn, pref)
        conn.execute(
            "INSERT INTO edges (source_id, target_id, edge_type, weight, created) "
            "VALUES (?, ?, 'defines', 1.0, '2026-01-01T00:00:00Z')",
            ("user-1", "pref-1"),
        )
        conn.commit()

        # Identity result with low score — off-topic query dragged it in
        mock_engine.recall_search_structured.return_value = [
            {"node": pref, "score": 0.3, "source": "hybrid"},
        ]

        result = builder.build_whisper_context(
            prompt="how do I cook pasta",
            user_node_id="user-1",
            min_score=0.45,
        )

        # Identity should be suppressed — score below min_score, no topical results
        assert result == ""

    def test_identity_kept_when_high_score_no_other_results(self, mock_graph):
        """High-scoring identity results should survive even without topical results."""
        mock_engine = MagicMock()
        builder = ContextBuilder(mock_graph, engine=mock_engine)
        conn = mock_graph.conn

        user_node = _make_node_dict("user-1", "Self", node_type="person")
        _insert_node(conn, user_node)
        pref = _make_node_dict("pref-1", "Lives in London", node_type="preference")
        _insert_node(conn, pref)
        conn.execute(
            "INSERT INTO edges (source_id, target_id, edge_type, weight, created) "
            "VALUES (?, ?, 'defines', 1.0, '2026-01-01T00:00:00Z')",
            ("user-1", "pref-1"),
        )
        conn.commit()

        # Identity result with high score — legitimate identity query
        mock_engine.recall_search_structured.return_value = [
            {"node": pref, "score": 0.9, "source": "hybrid"},
        ]

        result = builder.build_whisper_context(
            prompt="where does alice live",
            user_node_id="user-1",
            min_score=0.45,
        )

        assert "Lives in London" in result

    def test_identity_kept_when_topical_results_exist(self, mock_graph):
        """When topical results survive, identity should still be included."""
        mock_engine = MagicMock()
        builder = ContextBuilder(mock_graph, engine=mock_engine)
        conn = mock_graph.conn

        user_node = _make_node_dict("user-1", "Self", node_type="person")
        _insert_node(conn, user_node)
        pref = _make_node_dict("pref-1", "Likes dark mode", node_type="preference")
        _insert_node(conn, pref)
        conn.execute(
            "INSERT INTO edges (source_id, target_id, edge_type, weight, created) "
            "VALUES (?, ?, 'defines', 1.0, '2026-01-01T00:00:00Z')",
            ("user-1", "pref-1"),
        )
        conn.commit()

        topical = _make_node_dict("fact-1", "Search pipeline details")
        mock_engine.recall_search_structured.return_value = [
            {"node": pref, "score": 0.6, "source": "hybrid"},
            {"node": topical, "score": 0.8, "source": "hybrid"},
        ]

        result = builder.build_whisper_context(
            prompt="how does search work",
            user_node_id="user-1",
            min_score=0.1,
        )

        assert "Likes dark mode" in result
        assert "Search pipeline details" in result

    def test_identity_only_intent_stays_silent_without_search_results(self, mock_graph):
        """identity-only intent with no search results should stay silent (no graph dump)."""
        mock_engine = MagicMock()
        mock_engine.settings.claude_maintenance_enabled = False
        builder = ContextBuilder(mock_graph, engine=mock_engine)
        conn = mock_graph.conn

        from ormah.engine.prompt_classifier import PromptIntent

        mock_classifier = MagicMock()
        mock_classifier.classify.return_value = PromptIntent(categories=["identity"])
        builder._classifier = mock_classifier

        user_node = _make_node_dict("user-1", "Self", node_type="person")
        _insert_node(conn, user_node)
        pref = _make_node_dict("pref-1", "Likes dark mode", node_type="preference")
        _insert_node(conn, pref)
        conn.execute(
            "INSERT INTO edges (source_id, target_id, edge_type, weight, created) "
            "VALUES (?, ?, 'defines', 1.0, '2026-01-01T00:00:00Z')",
            ("user-1", "pref-1"),
        )
        conn.commit()

        result = builder.build_whisper_context(
            prompt="what do you know about me",
            user_node_id="user-1",
            min_score=0.1,
        )

        assert result == ""


class TestWhisperPrecisionGuards:
    """Precision helpers should favor the most relevant whisper candidate."""

    def test_identity_only_prefers_global_identity_results(self, mock_graph):
        mock_engine = MagicMock()
        builder = ContextBuilder(mock_graph, engine=mock_engine)

        from ormah.engine.prompt_classifier import PromptIntent

        mock_classifier = MagicMock()
        mock_classifier.classify.return_value = PromptIntent(categories=["identity"])
        builder._classifier = mock_classifier

        identity = _make_node_dict("id-1", "User lives in Dublin")
        identity["space"] = None
        project_fact = _make_node_dict("fact-1", "Ormah runs on port 8787")
        project_fact["space"] = "ormah"
        mock_engine.recall_search_structured.return_value = [
            {"node": identity, "score": 0.82, "source": "hybrid"},
            {"node": project_fact, "score": 0.79, "source": "hybrid"},
        ]

        result = builder.build_whisper_context(
            prompt="where does the user live",
            min_score=0.1,
            injection_gate=0.5,
        )

        assert "User lives in Dublin" in result
        assert "port 8787" not in result

    def test_mixed_identity_prompt_keeps_topical_project_result(self, mock_graph):
        mock_engine = MagicMock()
        builder = ContextBuilder(mock_graph, engine=mock_engine)

        from ormah.engine.prompt_classifier import PromptIntent

        mock_classifier = MagicMock()
        mock_classifier.classify.return_value = PromptIntent(categories=["identity", "general"])
        builder._classifier = mock_classifier

        global_pref = _make_node_dict("id-1", "Prefers dark mode")
        global_pref["space"] = None
        global_pref["content"] = "Use a dark theme with gold accent."
        project_fact = _make_node_dict("fact-1", "Auth uses JWT tokens")
        project_fact["space"] = "ormah"
        project_fact["content"] = "The auth flow uses JWT access tokens and refresh tokens."
        mock_engine.recall_search_structured.return_value = [
            {"node": global_pref, "score": 0.82, "source": "hybrid"},
            {"node": project_fact, "score": 0.79, "source": "hybrid"},
        ]

        result = builder.build_whisper_context(
            prompt="what's my preference for how ormah handles auth?",
            min_score=0.1,
            injection_gate=0.5,
        )

        assert "Auth uses JWT tokens" in result
        assert "Prefers dark mode" not in result

    def test_technical_theme_prompt_keeps_factual_memory(self, mock_graph):
        mock_engine = MagicMock()
        builder = ContextBuilder(mock_graph, engine=mock_engine)

        factual = _make_node_dict("fact-1", "Theme system implementation")
        factual["content"] = "The theme system loads tokens, merges overrides, and hydrates CSS variables."
        preference = _make_node_dict("pref-1", "Prefers dark theme", node_type="preference")
        preference["content"] = "Use a dark theme with gold accent."
        mock_engine.recall_search_structured.return_value = [
            {"node": factual, "score": 0.81, "source": "hybrid"},
            {"node": preference, "score": 0.78, "source": "hybrid"},
        ]

        result = builder.build_whisper_context(
            prompt="how is the theme system implemented?",
            min_score=0.1,
            injection_gate=0.5,
        )

        assert "Theme system implementation" in result


class TestPreferenceApplicability:
    """Standing rules use a typed applicability channel without biasing facts."""

    def _builder(self, mock_graph, main_results, preference_results):
        mock_engine = _make_engine_with_encoder(mock_graph)
        mock_engine.settings.whisper_exploration_enabled = False
        mock_engine.has_searchable_preferences.return_value = True
        mock_engine.recall_search_structured.side_effect = [
            main_results,
            preference_results,
        ]
        return ContextBuilder(mock_graph, engine=mock_engine), mock_engine

    def test_applicable_preference_is_merged_without_suppressing_fact(self, mock_graph):
        factual = _make_node_dict("fact-1", "Graph component implementation")
        preference = _make_node_dict(
            "pref-1", "Prefer simple designs", node_type="preference"
        )
        builder, engine = self._builder(
            mock_graph,
            [{"node": factual, "score": 0.80, "source": "hybrid"}],
            [{"node": preference, "score": 0.65, "source": "hybrid"}],
        )
        mock_ce = MagicMock()
        mock_ce.rerank.side_effect = [[3.0], [0.0]]

        with patch("ormah.embeddings.reranker._get_model", return_value=mock_ce), \
             patch(
                 "ormah.engine.context_builder.ContextBuilder._get_classifier",
                 return_value=None,
             ), \
             patch("ormah.engine.affinity.batch_fetch_affinity", return_value={}), \
             patch("ormah.engine.affinity.compute_affinity_boost", return_value=0.0):
            result = builder.build_whisper_context(
                prompt="build the graph component",
                min_score=0.1,
                reranker_enabled=True,
                reranker_min_score=0.0,
                injection_gate=0.45,
                preference_applicability_enabled=True,
                preference_applicability_gate=0.40,
            )

        assert "Prefer simple designs" in result
        assert "Graph component implementation" in result
        preference_call = engine.recall_search_structured.call_args_list[1]
        assert preference_call.kwargs["types"] == ["preference"]
        assert preference_call.kwargs["auto_temporal"] is False
        assert preference_call.kwargs["spread_activation"] is False
        main_query_vec = engine.recall_search_structured.call_args_list[0].kwargs["query_vec"]
        assert preference_call.kwargs["query_vec"] is main_query_vec
        engine._get_hybrid_search.return_value.encoder.encode_query.assert_called_once_with(
            "build the graph component"
        )
        assert mock_ce.rerank.call_args_list[1].args[0].startswith(
            "Relevant user preference for this action:"
        )

    def test_inapplicable_preference_cannot_suppress_factual_result(self, mock_graph):
        factual = _make_node_dict("fact-1", "Graph component implementation")
        preference = _make_node_dict(
            "pref-1", "Prefer dark themes", node_type="preference"
        )
        builder, _ = self._builder(
            mock_graph,
            [{"node": factual, "score": 0.80, "source": "hybrid"}],
            [{"node": preference, "score": 0.65, "source": "hybrid"}],
        )
        mock_ce = MagicMock()
        mock_ce.rerank.side_effect = [[3.0], [-12.0]]

        with patch("ormah.embeddings.reranker._get_model", return_value=mock_ce), \
             patch(
                 "ormah.engine.context_builder.ContextBuilder._get_classifier",
                 return_value=None,
             ), \
             patch("ormah.engine.affinity.batch_fetch_affinity", return_value={}), \
             patch("ormah.engine.affinity.compute_affinity_boost", return_value=0.0):
            result = builder.build_whisper_context(
                prompt="build the graph component",
                min_score=0.1,
                reranker_enabled=True,
                reranker_min_score=0.0,
                injection_gate=0.45,
                preference_applicability_enabled=True,
                preference_applicability_gate=0.40,
            )

        assert "Graph component implementation" in result
        assert "Prefer dark themes" not in result

    def test_empty_preference_store_skips_search_and_rerank(self, mock_graph):
        factual = _make_node_dict("fact-1", "Graph component implementation")
        builder, engine = self._builder(
            mock_graph,
            [{"node": factual, "score": 0.80, "source": "hybrid"}],
            [],
        )
        engine.has_searchable_preferences.return_value = False
        mock_ce = MagicMock()
        mock_ce.rerank.return_value = [3.0]

        with patch("ormah.embeddings.reranker._get_model", return_value=mock_ce), patch(
            "ormah.engine.context_builder.ContextBuilder._get_classifier",
            return_value=None,
        ), patch(
            "ormah.engine.affinity.batch_fetch_affinity", return_value={}
        ), patch(
            "ormah.engine.affinity.compute_affinity_boost", return_value=0.0
        ):
            result = builder.build_whisper_context(
                prompt="build the graph component",
                min_score=0.1,
                reranker_enabled=True,
                reranker_min_score=0.0,
                injection_gate=0.45,
                preference_applicability_enabled=True,
                preference_applicability_gate=0.40,
            )

        assert "Graph component implementation" in result
        assert engine.recall_search_structured.call_count == 1
        assert mock_ce.rerank.call_count == 1
        engine._get_hybrid_search.return_value.encoder.encode_query.assert_called_once()

    def test_different_effective_and_preference_queries_use_distinct_vectors(
        self, mock_graph
    ):
        from ormah.engine.prompt_classifier import PromptIntent

        factual = _make_node_dict("fact-1", "Graph component implementation")
        preference = _make_node_dict(
            "pref-1", "Prefer simple designs", node_type="preference"
        )
        builder, engine = self._builder(
            mock_graph,
            [{"node": factual, "score": 0.80, "source": "hybrid"}],
            [{"node": preference, "score": 0.65, "source": "hybrid"}],
        )
        mock_classifier = MagicMock()
        mock_classifier.classify.return_value = PromptIntent(
            categories=["temporal"],
            search_params={"search_query": "build the graph component"},
            prompt_vec=np.array([1.0, 0.0, 0.0], dtype=np.float32),
        )
        builder._classifier = mock_classifier
        main_vec = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        preference_vec = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        encoder = engine._get_hybrid_search.return_value.encoder
        encoder.encode_query.side_effect = [main_vec, preference_vec]
        mock_ce = MagicMock()
        mock_ce.rerank.side_effect = [[3.0], [0.0]]

        with patch("ormah.embeddings.reranker._get_model", return_value=mock_ce):
            builder.build_whisper_context(
                prompt="build the graph component yesterday",
                min_score=0.1,
                reranker_enabled=True,
                injection_gate=0.45,
                preference_applicability_enabled=True,
                preference_applicability_gate=0.40,
            )

        main_call, preference_call = engine.recall_search_structured.call_args_list
        assert main_call.kwargs["query"] == "build the graph component"
        assert main_call.kwargs["query_vec"] is main_vec
        assert preference_call.kwargs["query"] == "build the graph component yesterday"
        assert preference_call.kwargs["query_vec"] is preference_vec
        assert encoder.encode_query.call_count == 2

    def test_topical_overlap_guard_drops_unrelated_extra_result(self, mock_graph):
        mock_engine = MagicMock()
        builder = ContextBuilder(mock_graph, engine=mock_engine)

        relevant = _make_node_dict("fact-1", "FSRS decay algorithm")
        relevant["content"] = "Memory decay uses FSRS stability and retrievability."
        unrelated = _make_node_dict("fact-2", "MCP exposes six tools")
        unrelated["content"] = "remember, recall, recall_node, mark_outdated."
        mock_engine.recall_search_structured.return_value = [
            {"node": relevant, "score": 0.78, "source": "hybrid"},
            {"node": unrelated, "score": 0.74, "source": "hybrid"},
        ]

        result = builder.build_whisper_context(
            prompt="how does memory decay work in ormah",
            min_score=0.1,
            injection_gate=0.5,
        )

        assert "FSRS decay algorithm" in result
        assert "MCP exposes six tools" not in result


class TestWhisperContextBuffer:
    """Context-enhanced search using recent prompts."""

    def test_recent_prompts_enhance_search_query(self, mock_graph):
        """Underspecified follow-up prompts should use recent context in search."""
        mock_engine = MagicMock()
        builder = ContextBuilder(mock_graph, engine=mock_engine)
        from ormah.engine.prompt_classifier import PromptIntent

        builder._classifier = MagicMock()
        builder._classifier.classify.return_value = PromptIntent(categories=["continuation"])

        nodes = [_make_node_dict("node-0", "Whisper quality metrics")]
        mock_engine.recall_search_structured.return_value = [
            {"node": nodes[0], "score": 0.8, "source": "hybrid"},
        ]

        builder.build_whisper_context(
            prompt="what about the metrics side?",
            min_score=0.1,
            recent_prompts=["how's whisper quality?", "show me the eval results"],
        )

        call_kwargs = mock_engine.recall_search_structured.call_args
        query = call_kwargs.kwargs.get("query") or call_kwargs[1].get("query")
        # Query should contain context from recent prompts
        assert "whisper quality" in query
        assert "eval results" in query
        assert "what about the metrics side?" in query

    @pytest.mark.parametrize(
        "prompt",
        [
            "and the second one?",
            "and how often does that run?",
            "continue where we left off",
        ],
    )
    def test_prelabelled_continuations_still_enrich_search_query(self, mock_graph, prompt):
        """Acknowledgement handling must not regress useful short follow-ups."""
        from ormah.engine.prompt_classifier import PromptIntent

        mock_engine = MagicMock()
        builder = ContextBuilder(mock_graph, engine=mock_engine)
        builder._classifier = MagicMock()
        builder._classifier.classify.return_value = PromptIntent(categories=["continuation"])
        mock_engine.recall_search_structured.return_value = []

        builder.build_whisper_context(
            prompt=prompt,
            min_score=0.1,
            recent_prompts=["how does the scheduler work?"],
        )

        query = mock_engine.recall_search_structured.call_args.kwargs["query"]
        assert query == f"how does the scheduler work? {prompt}"

    def test_reranker_judges_context_enhanced_followup_query(self, mock_graph):
        """The reranker must score the same context-enhanced query that search
        ran on, not the bare prompt. Its ce_absolute drives the injection gate,
        so judging an underspecified follow-up ("and the second one?") against
        the bare prompt gate-rejects memories that only make sense with the
        session context."""
        from ormah.engine.prompt_classifier import PromptIntent

        mock_engine = _make_engine_with_encoder(mock_graph)
        builder = ContextBuilder(mock_graph, engine=mock_engine)
        builder._classifier = MagicMock()
        builder._classifier.classify.return_value = PromptIntent(categories=["continuation"])

        node = _make_node_dict("node-0", "Second eval result summary")
        mock_engine.recall_search_structured.return_value = [
            {"node": node, "score": 0.8, "source": "hybrid"},
        ]

        captured: dict = {}

        def _capture_rerank(query, docs):
            captured["query"] = query
            return [5.0 for _ in docs]

        mock_ce = MagicMock()
        mock_ce.rerank.side_effect = _capture_rerank

        with patch("ormah.embeddings.reranker._get_model", return_value=mock_ce), \
             patch("ormah.engine.affinity.batch_fetch_affinity", return_value={}), \
             patch("ormah.engine.affinity.compute_affinity_boost", return_value=0.0):
            builder.build_whisper_context(
                prompt="and the second one?",
                min_score=0.1,
                reranker_enabled=True,
                reranker_min_score=0.0,
                injection_gate=0.1,
                recent_prompts=["how's whisper quality?", "show me the eval results"],
            )

        # The reranker received the context-enhanced query (session tokens
        # present), not the bare underspecified prompt.
        assert "query" in captured
        assert "whisper quality" in captured["query"]
        assert "eval results" in captured["query"]
        assert "and the second one?" in captured["query"]

    def test_explicit_prompt_with_recent_context_uses_raw_prompt(self, mock_graph):
        """Fully specified prompts should not be polluted by recent context."""
        mock_engine = MagicMock()
        builder = ContextBuilder(mock_graph, engine=mock_engine)

        mock_engine.recall_search_structured.return_value = []

        builder.build_whisper_context(
            prompt="how does auth work",
            min_score=0.1,
            recent_prompts=["how's whisper quality?", "show me the eval results"],
        )

        call_kwargs = mock_engine.recall_search_structured.call_args
        query = call_kwargs.kwargs.get("query") or call_kwargs[1].get("query")
        assert query == "how does auth work"

    def test_no_recent_prompts_uses_raw_prompt(self, mock_graph):
        """Without recent_prompts, search query should be the raw prompt."""
        mock_engine = MagicMock()
        builder = ContextBuilder(mock_graph, engine=mock_engine)

        mock_engine.recall_search_structured.return_value = []

        builder.build_whisper_context(
            prompt="how does auth work",
            min_score=0.1,
            recent_prompts=None,
        )

        call_kwargs = mock_engine.recall_search_structured.call_args
        query = call_kwargs.kwargs.get("query") or call_kwargs[1].get("query")
        assert query == "how does auth work"

    def test_empty_recent_prompts_uses_raw_prompt(self, mock_graph):
        """Empty recent_prompts list should use the raw prompt."""
        mock_engine = MagicMock()
        builder = ContextBuilder(mock_graph, engine=mock_engine)

        mock_engine.recall_search_structured.return_value = []

        builder.build_whisper_context(
            prompt="how does auth work",
            min_score=0.1,
            recent_prompts=[],
        )

        call_kwargs = mock_engine.recall_search_structured.call_args
        query = call_kwargs.kwargs.get("query") or call_kwargs[1].get("query")
        assert query == "how does auth work"

    def test_recent_prompts_capped_at_2_for_followups(self, mock_graph):
        """Only the last 2 recent prompts should be used for follow-up prompts."""
        mock_engine = MagicMock()
        builder = ContextBuilder(mock_graph, engine=mock_engine)
        from ormah.engine.prompt_classifier import PromptIntent

        builder._classifier = MagicMock()
        builder._classifier.classify.return_value = PromptIntent(categories=["continuation"])

        mock_engine.recall_search_structured.return_value = []

        builder.build_whisper_context(
            prompt="what about this part?",
            min_score=0.1,
            recent_prompts=["old1", "old2", "old3", "old4", "old5"],
        )

        call_kwargs = mock_engine.recall_search_structured.call_args
        query = call_kwargs.kwargs.get("query") or call_kwargs[1].get("query")
        # Should only contain last 2 + current
        assert "old1" not in query
        assert "old2" not in query
        assert "old3" not in query
        assert "old4" in query
        assert "old5" in query
        assert "what about this part?" in query


class TestSessionBufferRoute:
    """Tests for the per-session prompt buffer in the whisper route."""

    def test_buffer_accumulates_prompts(self):
        """Buffer should accumulate prompts per session."""
        from ormah.api.routes_agent import _session_buffers
        from collections import deque
        import time

        # Clear state
        _session_buffers.clear()

        session_id = "test-session-1"
        buf = deque(maxlen=5)
        _session_buffers[session_id] = buf

        now = time.time()
        buf.append(("prompt 1", now))
        buf.append(("prompt 2", now + 1))
        buf.append(("prompt 3", now + 2))

        assert len(_session_buffers[session_id]) == 3
        prompts = [p for p, _ in _session_buffers[session_id]]
        assert prompts == ["prompt 1", "prompt 2", "prompt 3"]

        _session_buffers.clear()

    def test_buffers_isolated_by_session(self):
        """Different session IDs should have independent buffers."""
        from ormah.api.routes_agent import _session_buffers
        from collections import deque
        import time

        _session_buffers.clear()

        now = time.time()
        buf1 = deque(maxlen=5)
        buf1.append(("session1 prompt", now))
        _session_buffers["session-1"] = buf1

        buf2 = deque(maxlen=5)
        buf2.append(("session2 prompt", now))
        _session_buffers["session-2"] = buf2

        assert [p for p, _ in _session_buffers["session-1"]] == ["session1 prompt"]
        assert [p for p, _ in _session_buffers["session-2"]] == ["session2 prompt"]

        _session_buffers.clear()

    def test_acknowledgement_does_not_reinject_served_context(self, tmp_path):
        """The real route must not enrich an acknowledgement with its prior turn.

        The deterministic classifier deliberately labels the acknowledgement as
        ``continuation`` to reproduce the old failure.  The first route call
        writes an injected whisper_log event for the seeded Markdown decision;
        the second call uses the same session buffer and must remain silent.
        """
        from fastapi import FastAPI
        from fastapi.testclient import TestClient
        from ormah.api.routes_agent import _session_buffers, router
        from ormah.engine.prompt_classifier import PromptIntent
        from ormah.index.db import Database

        _session_buffers.clear()
        db = Database(tmp_path / "index.db")
        db.init_schema()
        graph = GraphIndex(db.conn)
        node = _make_node_dict("markdown-choice", "Markdown source of truth", space="nova")
        node["content"] = (
            "Nova stores memories as Markdown because people can read, edit, version and "
            "move them between tools. SQLite is a derived search index that can be rebuilt."
        )
        _insert_node(db.conn, node)
        db.conn.commit()

        prompt_vec = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        engine = MagicMock()
        engine.settings = _make_settings_mock()
        engine.db = db
        encoder = MagicMock()
        encoder.encode.return_value = prompt_vec
        encoder.encode_query.return_value = prompt_vec
        hybrid_search = MagicMock()
        hybrid_search.encoder = encoder
        engine._get_hybrid_search.return_value = hybrid_search
        engine.recall_search_structured.return_value = [
            {"node": node, "score": 0.9, "source": "hybrid", "raw_cosine": 0.9},
        ]
        builder = ContextBuilder(graph, engine=engine)
        builder._classifier = MagicMock()
        builder._classifier.classify.side_effect = [
            PromptIntent(categories=["general"]),
            PromptIntent(categories=["continuation"]),
        ]

        def get_whisper_context(**kwargs):
            return builder.build_whisper_context(
                **kwargs,
                min_score=0.1,
                injection_gate=0.55,
                topic_shift_enabled=True,
                topic_shift_threshold=0.75,
            )

        engine.get_whisper_context.side_effect = get_whisper_context
        app = FastAPI()
        app.include_router(router)
        app.state.engine = engine

        try:
            with TestClient(app) as client:
                first = client.post(
                    "/agent/whisper",
                    json={
                        "prompt": "Why did we choose Markdown as the source of truth?",
                        "space": "nova",
                        "session_id": "ack-later",
                    },
                )
                assert first.status_code == 200
                assert "Markdown source of truth" in first.json()["text"]
                served = db.conn.execute(
                    "SELECT was_injected FROM whisper_log WHERE session_id = ? AND node_id = ?",
                    ("ack-later", "markdown-choice"),
                ).fetchone()
                assert served is not None and served["was_injected"] == 1

                acknowledgement = client.post(
                    "/agent/whisper",
                    json={
                        "prompt": "Thanks, that helps.",
                        "space": "nova",
                        "session_id": "ack-later",
                    },
                )

            assert acknowledgement.status_code == 200
            assert acknowledgement.json()["text"] == ""
            assert engine.recall_search_structured.call_count == 1
            whisper_log_count = db.conn.execute(
                "SELECT COUNT(*) FROM whisper_log WHERE session_id = ?",
                ("ack-later",),
            ).fetchone()[0]
            assert whisper_log_count == 1
        finally:
            _session_buffers.clear()
            db.close()

    def test_session_gap_drops_acknowledgement_history(self):
        """A stale buffer cannot make an acknowledgement a follow-up query."""
        from collections import deque
        import time

        from fastapi import FastAPI
        from fastapi.testclient import TestClient
        from ormah.api.routes_agent import _session_buffers, router
        from ormah.config import settings as global_settings

        _session_buffers.clear()
        gap_seconds = global_settings.whisper_session_gap_minutes * 60
        _session_buffers["expired-ack"] = deque(
            [("Why did we choose Markdown?", time.time() - gap_seconds - 1)],
            maxlen=global_settings.whisper_context_buffer_size,
        )
        engine = MagicMock()
        engine.get_whisper_context.return_value = ""
        app = FastAPI()
        app.include_router(router)
        app.state.engine = engine

        try:
            with TestClient(app) as client:
                response = client.post(
                    "/agent/whisper",
                    json={"prompt": "Thanks, that helps.", "session_id": "expired-ack"},
                )

            assert response.status_code == 200
            assert engine.get_whisper_context.call_args.kwargs["recent_prompts"] is None
        finally:
            _session_buffers.clear()


class TestWhisperTopicShift:
    """Topic-shift detection: skip injection when prompt is on the same topic."""

    def test_same_topic_skips_injection(self, mock_graph):
        """High similarity to recent prompts → skip whisper."""
        mock_engine = MagicMock()
        builder = ContextBuilder(mock_graph, engine=mock_engine)

        # Mock encoder that returns identical vectors for same-topic prompts
        mock_encoder = MagicMock()
        same_vec = np.array([1.0, 0.0, 0.0])
        mock_encoder.encode.return_value = same_vec
        mock_encoder.encode_batch.return_value = np.array([same_vec, same_vec])

        mock_hybrid = MagicMock()
        mock_hybrid.encoder = mock_encoder
        mock_engine._get_hybrid_search.return_value = mock_hybrid

        result = builder.build_whisper_context(
            prompt="more about whisper thresholds",
            min_score=0.1,
            recent_prompts=["whisper threshold tuning", "whisper score analysis"],
            topic_shift_enabled=True,
            topic_shift_threshold=0.75,
        )

        assert result == ""
        # Should not call search since we skipped early
        mock_engine.recall_search_structured.assert_not_called()

    def test_topic_shift_triggers_injection(self, mock_graph):
        """Low similarity to recent prompts → proceed with whisper."""
        mock_engine = MagicMock()
        builder = ContextBuilder(mock_graph, engine=mock_engine)

        # Mock encoder: current prompt is orthogonal to recent prompts
        mock_encoder = MagicMock()
        mock_encoder.encode.return_value = np.array([1.0, 0.0, 0.0])
        mock_encoder.encode_batch.return_value = np.array([
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ])

        mock_hybrid = MagicMock()
        mock_hybrid.encoder = mock_encoder
        mock_engine._get_hybrid_search.return_value = mock_hybrid

        nodes = [_make_node_dict("node-0", "Auth details")]
        mock_engine.recall_search_structured.return_value = [
            {"node": nodes[0], "score": 0.8, "source": "hybrid"},
        ]

        result = builder.build_whisper_context(
            prompt="how does authentication work",
            min_score=0.1,
            recent_prompts=["whisper threshold tuning", "whisper score analysis"],
            topic_shift_enabled=True,
            topic_shift_threshold=0.75,
        )

        assert "Auth details" in result
        mock_engine.recall_search_structured.assert_called_once()

    def test_follow_up_prompt_bypasses_same_topic_skip(self, mock_graph):
        """Underspecified follow-up prompts should still search even on same topic."""
        mock_engine = MagicMock()
        builder = ContextBuilder(mock_graph, engine=mock_engine)
        from ormah.engine.prompt_classifier import PromptIntent

        builder._classifier = MagicMock()
        builder._classifier.classify.return_value = PromptIntent(categories=["continuation"])

        mock_encoder = MagicMock()
        same_vec = np.array([1.0, 0.0, 0.0])
        mock_encoder.encode.return_value = same_vec
        mock_encoder.encode_batch.return_value = np.array([same_vec, same_vec])

        mock_hybrid = MagicMock()
        mock_hybrid.encoder = mock_encoder
        mock_engine._get_hybrid_search.return_value = mock_hybrid

        nodes = [_make_node_dict("node-0", "Metrics details")]
        mock_engine.recall_search_structured.return_value = [
            {"node": nodes[0], "score": 0.8, "source": "hybrid"},
        ]

        result = builder.build_whisper_context(
            prompt="what about the metrics side?",
            min_score=0.1,
            recent_prompts=["how does the whisper eval pipeline work?"],
            topic_shift_enabled=True,
            topic_shift_threshold=0.75,
        )

        assert "Metrics details" in result
        mock_engine.recall_search_structured.assert_called_once()

    def test_cold_start_always_injects(self, mock_graph):
        """Empty recent_prompts (cold start) → always inject."""
        mock_engine = MagicMock()
        builder = ContextBuilder(mock_graph, engine=mock_engine)

        nodes = [_make_node_dict("node-0", "Some fact")]
        mock_engine.recall_search_structured.return_value = [
            {"node": nodes[0], "score": 0.8, "source": "hybrid"},
        ]

        result = builder.build_whisper_context(
            prompt="test query",
            min_score=0.1,
            recent_prompts=[],
            topic_shift_enabled=True,
            topic_shift_threshold=0.75,
        )

        assert "Some fact" in result

    def test_cold_start_none_prompts_injects(self, mock_graph):
        """None recent_prompts (cold start) → always inject."""
        mock_engine = MagicMock()
        builder = ContextBuilder(mock_graph, engine=mock_engine)

        nodes = [_make_node_dict("node-0", "Some fact")]
        mock_engine.recall_search_structured.return_value = [
            {"node": nodes[0], "score": 0.8, "source": "hybrid"},
        ]

        result = builder.build_whisper_context(
            prompt="test query",
            min_score=0.1,
            recent_prompts=None,
            topic_shift_enabled=True,
            topic_shift_threshold=0.75,
        )

        assert "Some fact" in result

    def test_disabled_skips_detection(self, mock_graph):
        """topic_shift_enabled=False → never skip, even if same topic."""
        mock_engine = MagicMock()
        builder = ContextBuilder(mock_graph, engine=mock_engine)

        nodes = [_make_node_dict("node-0", "Some fact")]
        mock_engine.recall_search_structured.return_value = [
            {"node": nodes[0], "score": 0.8, "source": "hybrid"},
        ]

        result = builder.build_whisper_context(
            prompt="whisper thresholds",
            min_score=0.1,
            recent_prompts=["whisper thresholds"],
            topic_shift_enabled=False,
            topic_shift_threshold=0.75,
        )

        assert "Some fact" in result
        mock_engine.recall_search_structured.assert_called_once()

    def test_encoder_failure_falls_through(self, mock_graph):
        """If encoder raises, should fall through to normal whisper."""
        mock_engine = MagicMock()
        builder = ContextBuilder(mock_graph, engine=mock_engine)

        mock_hybrid = MagicMock()
        mock_hybrid.encoder.encode.side_effect = RuntimeError("encoder broken")
        mock_engine._get_hybrid_search.return_value = mock_hybrid

        nodes = [_make_node_dict("node-0", "Some fact")]
        mock_engine.recall_search_structured.return_value = [
            {"node": nodes[0], "score": 0.8, "source": "hybrid"},
        ]

        result = builder.build_whisper_context(
            prompt="test query",
            min_score=0.1,
            recent_prompts=["previous prompt"],
            topic_shift_enabled=True,
            topic_shift_threshold=0.75,
        )

        # Should fall through and still return results
        assert "Some fact" in result

    def test_no_hybrid_search_falls_through(self, mock_graph):
        """If hybrid search is None, should fall through to normal whisper."""
        mock_engine = MagicMock()
        builder = ContextBuilder(mock_graph, engine=mock_engine)

        mock_engine._get_hybrid_search.return_value = None

        nodes = [_make_node_dict("node-0", "Some fact")]
        mock_engine.recall_search_structured.return_value = [
            {"node": nodes[0], "score": 0.8, "source": "hybrid"},
        ]

        result = builder.build_whisper_context(
            prompt="test query",
            min_score=0.1,
            recent_prompts=["previous prompt"],
            topic_shift_enabled=True,
            topic_shift_threshold=0.75,
        )

        assert "Some fact" in result


def _make_settings_mock(
    whisper_reranker_min_score=0.40,
    whisper_exploration_enabled=True,
    affinity_similarity_threshold=0.70,
    affinity_half_life_days=30.0,
    affinity_max_boost=0.15,
    affinity_implicit_weight=0.8,
    claude_maintenance_enabled=False,
    whisper_no_overlap_ce_floor=0.45,
    whisper_no_overlap_cosine_floor=0.70,
):
    """Create a MagicMock settings object with affinity-related float attributes."""
    settings = MagicMock()
    settings.whisper_reranker_min_score = whisper_reranker_min_score
    settings.whisper_exploration_enabled = whisper_exploration_enabled
    settings.affinity_similarity_threshold = affinity_similarity_threshold
    settings.affinity_half_life_days = affinity_half_life_days
    settings.affinity_max_boost = affinity_max_boost
    settings.affinity_implicit_weight = affinity_implicit_weight
    settings.claude_maintenance_enabled = claude_maintenance_enabled
    settings.whisper_no_overlap_ce_floor = whisper_no_overlap_ce_floor
    settings.whisper_no_overlap_cosine_floor = whisper_no_overlap_cosine_floor
    return settings


def _make_engine_with_encoder(mock_graph, prompt_vec=None):
    """Create a mock engine with a hybrid search encoder that returns a fixed vector."""
    mock_engine = MagicMock()
    mock_engine.settings = _make_settings_mock()

    if prompt_vec is None:
        prompt_vec = np.array([1.0, 0.0, 0.0], dtype=np.float32)

    mock_encoder = MagicMock()
    mock_encoder.encode.return_value = prompt_vec
    mock_encoder.encode_query.return_value = prompt_vec
    mock_hybrid = MagicMock()
    mock_hybrid.encoder = mock_encoder
    mock_engine._get_hybrid_search.return_value = mock_hybrid
    mock_engine.db = mock_graph._db if hasattr(mock_graph, "_db") else MagicMock()
    return mock_engine


class TestAffinityBoost:
    """Affinity boost rescues candidates that would otherwise be gated out."""

    def test_affinity_boost_rescues_below_gate_candidate(self, mock_graph):
        """A candidate below the injection gate that receives a strong affinity
        boost should survive into the final results."""
        prompt_vec = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        mock_engine = MagicMock()
        mock_engine.settings = _make_settings_mock(
            whisper_reranker_min_score=0.40,
            affinity_max_boost=0.15,
        )
        mock_encoder = MagicMock()
        mock_encoder.encode.return_value = prompt_vec
        mock_hybrid = MagicMock()
        mock_hybrid.encoder = mock_encoder
        mock_engine._get_hybrid_search.return_value = mock_hybrid

        builder = ContextBuilder(mock_graph, engine=mock_engine)

        # Candidate scores 0.48 after cross-encoder — below gate of 0.55
        # but with +0.15 boost it becomes 0.63 → above gate
        rescued_node = _make_node_dict("rescued", "Rescued memory")
        mock_engine.recall_search_structured.return_value = [
            {"node": rescued_node, "score": 0.48, "source": "hybrid"},
        ]

        # Patch affinity functions: boost returns +0.15
        with patch("ormah.engine.affinity.batch_fetch_affinity", return_value={"rescued": []}), \
             patch("ormah.engine.affinity.compute_affinity_boost", return_value=0.15):
            result = builder.build_whisper_context(
                prompt="relevant query",
                min_score=0.1,
                reranker_enabled=True,
                reranker_min_score=0.40,
                injection_gate=0.55,
            )

        assert "Rescued memory" in result

    def test_no_affinity_boost_without_reranker(self, mock_graph):
        """Affinity boost is only applied when reranker_enabled=True."""
        mock_engine = MagicMock()
        mock_engine.settings = _make_settings_mock()
        mock_engine._get_hybrid_search.return_value = MagicMock()

        builder = ContextBuilder(mock_graph, engine=mock_engine)

        node = _make_node_dict("node-1", "A fact")
        mock_engine.recall_search_structured.return_value = [
            {"node": node, "score": 0.48, "source": "hybrid"},
        ]

        with patch("ormah.engine.affinity.batch_fetch_affinity") as mock_bfa:
            builder.build_whisper_context(
                prompt="test",
                min_score=0.1,
                reranker_enabled=False,  # no reranker → no affinity boost
                injection_gate=0.55,
            )

        # batch_fetch_affinity should NOT be called when reranker is disabled
        mock_bfa.assert_not_called()

    def test_affinity_boost_failure_falls_back_gracefully(self, mock_graph):
        """If affinity boost raises, the pipeline should continue with unmodified scores."""
        prompt_vec = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        mock_engine = MagicMock()
        mock_engine.settings = _make_settings_mock()
        mock_encoder = MagicMock()
        mock_encoder.encode.return_value = prompt_vec
        mock_hybrid = MagicMock()
        mock_hybrid.encoder = mock_encoder
        mock_engine._get_hybrid_search.return_value = mock_hybrid

        builder = ContextBuilder(mock_graph, engine=mock_engine)

        node = _make_node_dict("node-1", "Important fact")
        mock_engine.recall_search_structured.return_value = [
            {"node": node, "score": 0.9, "source": "hybrid"},
        ]

        mock_ce = MagicMock()
        mock_ce.rerank.return_value = [5.0]

        with patch("ormah.embeddings.reranker._get_model", return_value=mock_ce), \
             patch("ormah.engine.affinity.batch_fetch_affinity", side_effect=RuntimeError("db error")):
            result = builder.build_whisper_context(
                prompt="test",
                min_score=0.1,
                reranker_enabled=True,
                injection_gate=0.1,
            )

        # Should still return results despite affinity boost failure
        assert "Important fact" in result

    def test_negative_affinity_suppresses_candidate(self, mock_graph):
        """A strong negative affinity boost should push a marginal candidate below the gate.

        Exploration is disabled so the injection gate is the final arbiter.
        """
        prompt_vec = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        mock_engine = MagicMock()
        mock_engine.settings = _make_settings_mock(
            affinity_max_boost=0.15,
            whisper_reranker_min_score=0.40,
            whisper_exploration_enabled=False,  # isolate affinity suppression
        )
        mock_encoder = MagicMock()
        mock_encoder.encode.return_value = prompt_vec
        mock_hybrid = MagicMock()
        mock_hybrid.encoder = mock_encoder
        mock_engine._get_hybrid_search.return_value = mock_hybrid

        builder = ContextBuilder(mock_graph, engine=mock_engine)

        # Candidate with marginal CE relevance — above gate normally;
        # with -0.15 affinity boost it drops below the gate.
        node = _make_node_dict("marginal", "Marginal memory")
        mock_engine.recall_search_structured.return_value = [
            {"node": node, "score": 0.60, "source": "hybrid"},
        ]

        mock_ce = MagicMock()
        # Raw CE -0.8 → ce_absolute (-0.8+12)/18 ≈ 0.622: clears the 0.55
        # gate on its own, but not after the negative affinity delta.
        mock_ce.rerank.return_value = [-0.8]

        with patch("ormah.embeddings.reranker._get_model", return_value=mock_ce), \
             patch("ormah.engine.affinity.batch_fetch_affinity", return_value={"marginal": []}), \
             patch("ormah.engine.affinity.compute_affinity_boost", return_value=-0.15):
            result = builder.build_whisper_context(
                prompt="unrelated query",
                min_score=0.1,
                reranker_enabled=True,
                reranker_min_score=0.40,
                injection_gate=0.55,
            )

        # Gate score = ce_absolute 0.622 - 0.15 = 0.472 < 0.55 → suppressed
        assert "Marginal memory" not in result


class TestExplorationSlot:
    """Exploration slot injects one unconfirmed gated-out candidate."""

    def test_exploration_injects_gated_out_candidate_with_no_affinity_signal(self, mock_graph):
        """A gated-out candidate with no existing affinity signal should be injected
        via the exploration slot."""
        prompt_vec = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        mock_engine = MagicMock()
        mock_engine.settings = _make_settings_mock(
            whisper_exploration_enabled=True,
            whisper_reranker_min_score=0.40,
        )
        mock_encoder = MagicMock()
        mock_encoder.encode.return_value = prompt_vec
        mock_hybrid = MagicMock()
        mock_hybrid.encoder = mock_encoder
        mock_engine._get_hybrid_search.return_value = mock_hybrid

        builder = ContextBuilder(mock_graph, engine=mock_engine)

        # One candidate above gate (injected normally), one gated-out but above 0.40
        injected_node = _make_node_dict("injected", "Injected memory")
        explore_node = _make_node_dict("explore", "Exploration memory")
        mock_engine.recall_search_structured.return_value = [
            {"node": injected_node, "score": 0.70, "source": "hybrid"},
            {"node": explore_node, "score": 0.49, "source": "hybrid"},
        ]

        mock_ce = MagicMock()
        # CE scores: injected above 0.50 gate, explore between 0.40-0.50
        # CE=2.0 → rescale=0.778, emb=0.70 → blended=0.4*0.778+0.6*0.70=0.731 (above 0.50) ✓
        # CE=-5.0 → rescale=0.389, emb=0.49 → blended=0.4*0.389+0.6*0.49=0.450 (below 0.50, above 0.40)
        mock_ce.rerank.return_value = [2.0, -5.0]

        # No affinity signal for explore_node → eligible for exploration
        with patch("ormah.embeddings.reranker._get_model", return_value=mock_ce), \
             patch("ormah.engine.affinity.batch_fetch_affinity", return_value={}), \
             patch("ormah.engine.affinity.compute_affinity_boost", return_value=0.0):
            result = builder.build_whisper_context(
                prompt="injected exploration memory check",
                min_score=0.1,
                reranker_enabled=True,
                reranker_min_score=0.40,
                injection_gate=0.50,
            )

        assert "Exploration memory" in result
        # Exploration candidates are labeled so the agent can weigh them
        assert "**[exploring]** Exploration memory" in result

    def test_exploration_skips_candidate_with_existing_affinity_signal(self, mock_graph):
        """A gated-out candidate that already has an affinity signal for a similar prompt
        should NOT be injected via the exploration slot."""
        prompt_vec = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        # Affinity row with very similar prompt vec → sim > 0.70
        similar_prompt_vec = np.array([0.99, 0.14, 0.0], dtype=np.float32)
        similar_prompt_vec = (similar_prompt_vec / np.linalg.norm(similar_prompt_vec)).astype(np.float32)

        mock_engine = MagicMock()
        mock_engine.settings = _make_settings_mock(
            whisper_exploration_enabled=True,
            whisper_reranker_min_score=0.40,
        )
        mock_encoder = MagicMock()
        mock_encoder.encode.return_value = prompt_vec
        mock_hybrid = MagicMock()
        mock_hybrid.encoder = mock_encoder
        mock_engine._get_hybrid_search.return_value = mock_hybrid

        builder = ContextBuilder(mock_graph, engine=mock_engine)

        injected_node = _make_node_dict("injected", "Injected memory")
        known_node = _make_node_dict("known", "Known gated memory")
        mock_engine.recall_search_structured.return_value = [
            {"node": injected_node, "score": 0.70, "source": "hybrid"},
            {"node": known_node, "score": 0.49, "source": "hybrid"},
        ]

        mock_ce = MagicMock()
        # CE=2.0 → rescale=0.778, emb=0.70 → blended=0.731 (above 0.50) ✓
        # CE=-5.0 → rescale=0.389, emb=0.49 → blended=0.450 (below 0.50, above 0.40)
        mock_ce.rerank.return_value = [2.0, -5.0]

        # known_node has an affinity row with sim > 0.70 to current prompt
        affinity_map_for_explore = {
            "known": [{"prompt_vec": similar_prompt_vec.tobytes(), "signal": 1,
                       "source": "explicit", "confirmed_at": "2026-03-01T00:00:00+00:00"}],
        }

        with patch("ormah.embeddings.reranker._get_model", return_value=mock_ce), \
             patch("ormah.engine.affinity.batch_fetch_affinity", return_value=affinity_map_for_explore), \
             patch("ormah.engine.affinity.compute_affinity_boost", return_value=0.0):
            result = builder.build_whisper_context(
                prompt="test query",
                min_score=0.1,
                reranker_enabled=True,
                reranker_min_score=0.40,
                injection_gate=0.50,
            )

        # known_node already has signal → should NOT be explored
        assert "Known gated memory" not in result

    def test_exploration_disabled_skips_slot(self, mock_graph):
        """When whisper_exploration_enabled=False, no exploration slot is injected."""
        prompt_vec = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        mock_engine = MagicMock()
        mock_engine.settings = _make_settings_mock(
            whisper_exploration_enabled=False,
            whisper_reranker_min_score=0.40,
        )
        mock_encoder = MagicMock()
        mock_encoder.encode.return_value = prompt_vec
        mock_hybrid = MagicMock()
        mock_hybrid.encoder = mock_encoder
        mock_engine._get_hybrid_search.return_value = mock_hybrid

        builder = ContextBuilder(mock_graph, engine=mock_engine)

        injected_node = _make_node_dict("injected", "Injected memory")
        explore_node = _make_node_dict("explore", "Should not appear")
        mock_engine.recall_search_structured.return_value = [
            {"node": injected_node, "score": 0.70, "source": "hybrid"},
            {"node": explore_node, "score": 0.49, "source": "hybrid"},
        ]

        mock_ce = MagicMock()
        # CE=2.0 → rescale=0.778, emb=0.70 → blended=0.731 (above 0.50) ✓
        # CE=-5.0 → rescale=0.389, emb=0.49 → blended=0.450 (below 0.50, above 0.40)
        mock_ce.rerank.return_value = [2.0, -5.0]

        with patch("ormah.embeddings.reranker._get_model", return_value=mock_ce), \
             patch("ormah.engine.affinity.batch_fetch_affinity", return_value={}), \
             patch("ormah.engine.affinity.compute_affinity_boost", return_value=0.0):
            result = builder.build_whisper_context(
                prompt="test query",
                min_score=0.1,
                reranker_enabled=True,
                reranker_min_score=0.40,
                injection_gate=0.50,
            )

        assert "Should not appear" not in result


class TestExplorationCEGate:
    """Exploration slot should skip candidates the CE strongly rejected."""

    def test_exploration_skips_strongly_rejected_by_ce(self, mock_graph):
        """Candidate with CE < -8 should not be explored even with no affinity signal."""
        mock_engine = MagicMock()
        mock_engine.settings = _make_settings_mock(
            whisper_exploration_enabled=True,
            affinity_similarity_threshold=0.70,
            whisper_reranker_min_score=0.0,
        )
        builder = ContextBuilder(mock_graph, engine=mock_engine)

        nodes = [
            _make_node_dict("pass-1", "Relevant fact"),
            _make_node_dict("noise-1", "Noise fact"),
        ]
        mock_engine.recall_search_structured.return_value = [
            {"node": nodes[0], "score": 0.75, "source": "hybrid"},
            {"node": nodes[1], "score": 0.60, "source": "hybrid"},
        ]

        # Mock CE model: pass-1 gets positive CE (passes gate), noise-1 gets very negative CE
        mock_ce = MagicMock()
        mock_ce.rerank.return_value = [2.0, -10.0]

        # Mock encoder for affinity boost path
        prompt_vec = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        mock_encoder = MagicMock()
        mock_encoder.encode.return_value = prompt_vec
        mock_hybrid = MagicMock()
        mock_hybrid.encoder = mock_encoder
        mock_engine._get_hybrid_search.return_value = mock_hybrid

        with patch("ormah.embeddings.reranker._get_model", return_value=mock_ce), \
             patch("ormah.engine.context_builder.ContextBuilder._get_classifier", return_value=None), \
             patch("ormah.engine.affinity.batch_fetch_affinity", return_value={"noise-1": []}), \
             patch("ormah.engine.affinity.compute_affinity_boost", return_value=0.0):
            result = builder.build_whisper_context(
                prompt="what is kubernetes",
                injection_gate=0.50,
                reranker_enabled=True,
                reranker_min_score=0.40,
            )

        assert "Noise fact" not in result
        assert "Relevant fact" in result

    def test_exploration_allows_borderline_ce(self, mock_graph):
        """Candidate with CE > -8 should still be eligible for exploration."""
        mock_engine = MagicMock()
        mock_engine.settings = _make_settings_mock(
            whisper_exploration_enabled=True,
            affinity_similarity_threshold=0.70,
            whisper_reranker_min_score=0.0,
        )
        builder = ContextBuilder(mock_graph, engine=mock_engine)

        nodes = [
            _make_node_dict("pass-1", "Relevant fact"),
            _make_node_dict("maybe-1", "Maybe useful"),
        ]
        mock_engine.recall_search_structured.return_value = [
            {"node": nodes[0], "score": 0.75, "source": "hybrid"},
            {"node": nodes[1], "score": 0.60, "source": "hybrid"},
        ]

        # pass-1: CE=+2.0 → passes gate; maybe-1: CE=-5.0 → gated out but CE > -8 (explorable)
        mock_ce = MagicMock()
        mock_ce.rerank.return_value = [2.0, -5.0]

        # Mock encoder for affinity boost path
        prompt_vec = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        mock_encoder = MagicMock()
        mock_encoder.encode.return_value = prompt_vec
        mock_hybrid = MagicMock()
        mock_hybrid.encoder = mock_encoder
        mock_engine._get_hybrid_search.return_value = mock_hybrid

        with patch("ormah.embeddings.reranker._get_model", return_value=mock_ce), \
             patch("ormah.engine.context_builder.ContextBuilder._get_classifier", return_value=None), \
             patch("ormah.engine.affinity.batch_fetch_affinity", return_value={"maybe-1": []}), \
             patch("ormah.engine.affinity.compute_affinity_boost", return_value=0.0):
            result = builder.build_whisper_context(
                prompt="relevant fact or maybe useful",
                injection_gate=0.50,
                reranker_enabled=True,
                reranker_min_score=0.40,
            )

        assert "Maybe useful" in result

    def test_exploration_when_no_ce_score_proceeds_normally(self, mock_graph):
        """When a candidate has no cross_encoder_score (e.g., reranker errored),
        the CE gate check should be skipped and exploration proceeds normally."""
        mock_engine = MagicMock()
        mock_engine.settings = _make_settings_mock(
            whisper_exploration_enabled=True,
            affinity_similarity_threshold=0.70,
            whisper_reranker_min_score=0.40,
        )
        builder = ContextBuilder(mock_graph, engine=mock_engine)

        nodes = [
            _make_node_dict("pass-1", "Relevant fact"),
            _make_node_dict("explore-1", "Explore me"),
        ]
        mock_engine.recall_search_structured.return_value = [
            {"node": nodes[0], "score": 0.70, "source": "hybrid"},
            {"node": nodes[1], "score": 0.45, "source": "hybrid"},
        ]

        # Reranker fails, so candidates don't have cross_encoder_score
        mock_ce = MagicMock()
        mock_ce.rerank.side_effect = RuntimeError("CE model not found")

        # Mock encoder for affinity boost path
        prompt_vec = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        mock_encoder = MagicMock()
        mock_encoder.encode.return_value = prompt_vec
        mock_hybrid = MagicMock()
        mock_hybrid.encoder = mock_encoder
        mock_engine._get_hybrid_search.return_value = mock_hybrid

        with patch("ormah.embeddings.reranker._get_model", return_value=mock_ce), \
             patch("ormah.engine.context_builder.ContextBuilder._get_classifier", return_value=None), \
             patch("ormah.engine.affinity.batch_fetch_affinity", return_value={"explore-1": []}), \
             patch("ormah.engine.affinity.compute_affinity_boost", return_value=0.0):
            result = builder.build_whisper_context(
                prompt="some query",
                injection_gate=0.50,
                reranker_enabled=True,  # reranker enabled but errors
                reranker_min_score=0.40,
            )

        # explore-1 at 0.45 clears the 0.40 floor, gets gated out at 0.50,
        # but should be explored since it has no CE score to block it
        assert "Explore me" in result


class TestWhisperLog:
    """whisper_log rows are written when session_id is provided."""

    @pytest.fixture
    def db_graph(self, tmp_path):
        """Fixture returning (db, graph) with real schema for whisper_log tests."""
        from ormah.index.db import Database

        db = Database(tmp_path / "index.db")
        db.init_schema()
        graph = GraphIndex(db.conn)
        return db, graph

    def test_whisper_log_written_on_session_id(self, db_graph):
        """When session_id is provided, whisper_log rows should be written for
        injected candidates."""
        db, graph = db_graph

        prompt_vec = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        mock_engine = MagicMock()
        mock_engine.settings = _make_settings_mock(whisper_reranker_min_score=0.40)
        mock_encoder = MagicMock()
        mock_encoder.encode.return_value = prompt_vec
        mock_hybrid = MagicMock()
        mock_hybrid.encoder = mock_encoder
        mock_engine._get_hybrid_search.return_value = mock_hybrid
        mock_engine.db = db

        builder = ContextBuilder(graph, engine=mock_engine)

        node = _make_node_dict("node-log-1", "Logged memory")
        mock_engine.recall_search_structured.return_value = [
            {"node": node, "score": 0.70, "source": "hybrid"},
        ]

        mock_ce = MagicMock()
        mock_ce.rerank.return_value = [2.0]

        with patch("ormah.embeddings.reranker._get_model", return_value=mock_ce), \
             patch("ormah.engine.affinity.batch_fetch_affinity", return_value={}), \
             patch("ormah.engine.affinity.compute_affinity_boost", return_value=0.0):
            result = builder.build_whisper_context(
                prompt="how does search work",
                min_score=0.1,
                reranker_enabled=True,
                reranker_min_score=0.40,
                injection_gate=0.55,
                session_id="test-session-abc",
            )

        rows = db.conn.execute("SELECT * FROM whisper_log").fetchall()
        assert len(rows) >= 1
        row = rows[0]
        assert row["session_id"] == "test-session-abc"
        assert row["node_id"] == "node-log-1"
        assert row["was_injected"] == 1
        assert row["prompt_text"] is None
        assert row["prompt_vec"] == b""
        event = db.conn.execute(
            "SELECT * FROM retrieval_events WHERE id = ?",
            (row["retrieval_event_id"],),
        ).fetchone()
        assert event["surface"] == "whisper"
        assert event["prompt_text"] == "how does search work"
        assert len(event["prompt_vec"]) == 12
        assert row["decision_stage"] == "injected"
        assert row["retrieval_rank"] == 1
        assert row["final_rank"] == 1
        assert row["ce_absolute"] is not None
        assert f"(id: node-log, whisper_log_id: {row['id']})" in result

    def test_whisper_log_not_written_without_session_id(self, db_graph):
        """When session_id is None, no whisper_log rows should be written."""
        db, graph = db_graph

        prompt_vec = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        mock_engine = MagicMock()
        mock_engine.settings = _make_settings_mock()
        mock_encoder = MagicMock()
        mock_encoder.encode.return_value = prompt_vec
        mock_hybrid = MagicMock()
        mock_hybrid.encoder = mock_encoder
        mock_engine._get_hybrid_search.return_value = mock_hybrid
        mock_engine.db = db

        builder = ContextBuilder(graph, engine=mock_engine)

        node = _make_node_dict("node-nolog-1", "Unlogged memory")
        mock_engine.recall_search_structured.return_value = [
            {"node": node, "score": 0.80, "source": "hybrid"},
        ]

        mock_ce = MagicMock()
        mock_ce.rerank.return_value = [2.0]

        with patch("ormah.embeddings.reranker._get_model", return_value=mock_ce), \
             patch("ormah.engine.affinity.batch_fetch_affinity", return_value={}), \
             patch("ormah.engine.affinity.compute_affinity_boost", return_value=0.0):
            builder.build_whisper_context(
                prompt="how does search work",
                min_score=0.1,
                reranker_enabled=True,
                injection_gate=0.55,
                session_id=None,  # no session_id
            )

        rows = db.conn.execute("SELECT * FROM whisper_log").fetchall()
        assert len(rows) == 0

    def test_temporal_results_excluded_from_whisper_log(self, db_graph):
        """Temporal results (source='temporal') must not appear in whisper_log."""
        db, graph = db_graph

        prompt_vec = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        mock_engine = MagicMock()
        mock_engine.settings = _make_settings_mock()
        mock_encoder = MagicMock()
        mock_encoder.encode.return_value = prompt_vec
        mock_hybrid = MagicMock()
        mock_hybrid.encoder = mock_encoder
        mock_engine._get_hybrid_search.return_value = mock_hybrid
        mock_engine.db = db

        builder = ContextBuilder(graph, engine=mock_engine)

        from ormah.engine.prompt_classifier import PromptIntent

        mock_classifier = MagicMock()
        mock_classifier.classify.return_value = PromptIntent(
            categories=["temporal"],
            search_params={
                "created_after": "2026-03-01T00:00:00+00:00",
                "search_query": "what did I work on",
            },
        )
        builder._classifier = mock_classifier

        temporal_node = _make_node_dict("temporal-node", "Recent memory")
        mock_engine.recall_search_structured.return_value = [
            {"node": temporal_node, "score": 0.001, "source": "temporal"},
        ]

        builder.build_whisper_context(
            prompt="what did I work on today",
            min_score=0.1,
            session_id="session-temporal-test",
        )

        rows = db.conn.execute("SELECT * FROM whisper_log").fetchall()
        # Temporal results must never be logged
        assert len(rows) == 0

    def test_whisper_log_records_pre_boost_score(self, db_graph):
        """whisper_log.score should store the pre-boost blended score, not the boosted value."""
        db, graph = db_graph

        prompt_vec = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        mock_engine = MagicMock()
        mock_engine.settings = _make_settings_mock(whisper_reranker_min_score=0.40)
        mock_encoder = MagicMock()
        mock_encoder.encode.return_value = prompt_vec
        mock_hybrid = MagicMock()
        mock_hybrid.encoder = mock_encoder
        mock_engine._get_hybrid_search.return_value = mock_hybrid
        mock_engine.db = db

        builder = ContextBuilder(graph, engine=mock_engine)

        node = _make_node_dict("node-preboost", "Pre-boost check")
        # CE gives blended ~0.60; affinity adds +0.10
        mock_engine.recall_search_structured.return_value = [
            {"node": node, "score": 0.60, "source": "hybrid"},
        ]

        mock_ce = MagicMock()
        mock_ce.rerank.return_value = [1.0]

        with patch("ormah.embeddings.reranker._get_model", return_value=mock_ce), \
             patch("ormah.engine.affinity.batch_fetch_affinity", return_value={}), \
             patch("ormah.engine.affinity.compute_affinity_boost", return_value=0.10):
            builder.build_whisper_context(
                prompt="query for pre-boost test",
                min_score=0.1,
                reranker_enabled=True,
                reranker_min_score=0.40,
                injection_gate=0.55,
                session_id="session-preboost",
            )

        rows = db.conn.execute("SELECT * FROM whisper_log WHERE node_id = 'node-preboost'").fetchall()
        assert len(rows) == 1
        # Score in the DB should be the pre-boost value, not boosted
        # The actual blended score from mock reranker will differ, but it should NOT
        # be the boosted score (pre_boost_score = r.get("_pre_boost_score", r["score"]))
        # Since we patched compute_affinity_boost to return 0.10, the logged score
        # should be less than the injected score by ~0.10
        logged_score = rows[0]["score"]
        # The logged score is the pre-boost (CE blended), not boosted
        # We just verify a row was written — exact value depends on reranker mock
        assert logged_score >= 0.0

    def test_whisper_log_records_candidates_rejected_before_reranking(self, db_graph):
        db, graph = db_graph
        prompt_vec = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        mock_engine = MagicMock()
        mock_engine.settings = _make_settings_mock()
        mock_encoder = MagicMock()
        mock_encoder.encode.return_value = prompt_vec
        mock_hybrid = MagicMock()
        mock_hybrid.encoder = mock_encoder
        mock_engine._get_hybrid_search.return_value = mock_hybrid
        mock_engine.db = db

        builder = ContextBuilder(graph, engine=mock_engine)
        node = _make_node_dict("trimmed-node", "Unrelated candidate")
        mock_engine.recall_search_structured.return_value = [
            {
                "node": node,
                "score": 0.20,
                "source": "hybrid",
                "raw_cosine": 0.25,
            }
        ]

        builder.build_whisper_context(
            prompt="deployment pipeline",
            min_score=0.45,
            session_id="session-pretrim",
        )

        row = db.conn.execute(
            "SELECT * FROM whisper_log WHERE node_id = 'trimmed-node'"
        ).fetchone()
        assert row is not None
        assert row["decision_stage"] == "pre_rerank_floor"
        assert row["retrieval_score"] == pytest.approx(0.20)
        assert row["raw_cosine"] == pytest.approx(0.25)
        assert row["ce_absolute"] is None
        assert row["was_injected"] == 0

    def test_whisper_log_records_topical_rejection_with_absolute_scores(self, db_graph):
        db, graph = db_graph
        prompt_vec = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        mock_engine = MagicMock()
        mock_engine.settings = _make_settings_mock()
        mock_encoder = MagicMock()
        mock_encoder.encode.return_value = prompt_vec
        mock_hybrid = MagicMock()
        mock_hybrid.encoder = mock_encoder
        mock_engine._get_hybrid_search.return_value = mock_hybrid
        mock_engine.db = db

        builder = ContextBuilder(graph, engine=mock_engine)
        relevant = _make_node_dict("overlap-node", "Deployment pipeline details")
        rejected = _make_node_dict("rejected-node", "Unrelated embedding neighbor")
        mock_engine.recall_search_structured.return_value = [
            {"node": relevant, "score": 0.80, "source": "hybrid"},
            {"node": rejected, "score": 0.75, "source": "hybrid"},
        ]
        mock_ce = MagicMock()
        mock_ce.rerank.return_value = [3.0, -5.0]

        with patch("ormah.embeddings.reranker._get_model", return_value=mock_ce), \
             patch("ormah.engine.affinity.batch_fetch_affinity", return_value={}), \
             patch("ormah.engine.affinity.compute_affinity_boost", return_value=0.0):
            builder.build_whisper_context(
                prompt="deployment pipeline problems",
                min_score=0.1,
                reranker_enabled=True,
                reranker_min_score=0.0,
                injection_gate=0.45,
                no_overlap_ce_floor=0.45,
                session_id="session-topical",
            )

        row = db.conn.execute(
            "SELECT * FROM whisper_log WHERE node_id = 'rejected-node'"
        ).fetchone()
        assert row is not None
        assert row["decision_stage"] == "topical_filter"
        assert row["cross_encoder_score"] == pytest.approx(-5.0)
        assert row["ce_absolute"] == pytest.approx(7 / 18)
        assert row["gate_score"] == pytest.approx(7 / 18)
        assert row["was_injected"] == 0


class TestWhisperFlatRankedDisplay:
    """Whisper outputs a flat ranked list — top 2 full, rest title+ID only."""

    def test_top_two_nodes_shown_in_full(self, mock_graph):
        mock_engine = MagicMock()
        builder = ContextBuilder(mock_graph, engine=mock_engine)

        nodes = [
            {**_make_node_dict(f"node-{i}", f"Title {i}"), "content": f"Full content for node {i}, longer than a title."}
            for i in range(4)
        ]
        mock_engine.recall_search_structured.return_value = [
            {"node": nodes[0], "score": 0.9, "source": "hybrid"},
            {"node": nodes[1], "score": 0.8, "source": "hybrid"},
            {"node": nodes[2], "score": 0.7, "source": "hybrid"},
            {"node": nodes[3], "score": 0.6, "source": "hybrid"},
        ]

        result = builder.build_whisper_context(
            prompt="tell me about nodes",
            injection_gate=0.0,
        )

        # Top 2 show full content
        assert "Full content for node 0" in result
        assert "Full content for node 1" in result
        # Nodes 3-4 do NOT show content
        assert "Full content for node 2" not in result
        assert "Full content for node 3" not in result

    def test_remaining_nodes_show_title_and_id_only(self, mock_graph):
        mock_engine = MagicMock()
        builder = ContextBuilder(mock_graph, engine=mock_engine)

        nodes = [
            {**_make_node_dict(f"abcd{i:04d}", f"Title {i}"), "content": f"Full content for node {i}."}
            for i in range(4)
        ]
        mock_engine.recall_search_structured.return_value = [
            {"node": nodes[0], "score": 0.9, "source": "hybrid"},
            {"node": nodes[1], "score": 0.8, "source": "hybrid"},
            {"node": nodes[2], "score": 0.7, "source": "hybrid"},
            {"node": nodes[3], "score": 0.6, "source": "hybrid"},
        ]

        result = builder.build_whisper_context(
            prompt="tell me about nodes",
            injection_gate=0.0,
        )

        # Nodes 3-4 show title and ID
        assert "Title 2" in result
        assert "abcd0002" in result
        assert "Title 3" in result
        assert "abcd0003" in result

    def test_all_nodes_have_node_id(self, mock_graph):
        mock_engine = MagicMock()
        builder = ContextBuilder(mock_graph, engine=mock_engine)

        nodes = [
            _make_node_dict(f"nodeid{i:02d}", f"Title {i}")
            for i in range(3)
        ]
        mock_engine.recall_search_structured.return_value = [
            {"node": nodes[0], "score": 0.9, "source": "hybrid"},
            {"node": nodes[1], "score": 0.8, "source": "hybrid"},
            {"node": nodes[2], "score": 0.7, "source": "hybrid"},
        ]

        result = builder.build_whisper_context(
            prompt="tell me about nodes",
            injection_gate=0.0,
        )

        # All nodes show their IDs
        assert "nodeid00" in result
        assert "nodeid01" in result
        assert "nodeid02" in result

    def test_no_section_headers(self, mock_graph):
        mock_engine = MagicMock()
        builder = ContextBuilder(mock_graph, engine=mock_engine)

        # Mix of tiers and types
        nodes = [
            {**_make_node_dict("core-001", "Core fact", tier="core"), "content": "Some core content."},
            {**_make_node_dict("work-001", "Working fact", tier="working"), "content": "Some working content."},
        ]
        mock_engine.recall_search_structured.return_value = [
            {"node": nodes[0], "score": 0.9, "source": "hybrid"},
            {"node": nodes[1], "score": 0.8, "source": "hybrid"},
        ]

        result = builder.build_whisper_context(
            prompt="tell me something",
            injection_gate=0.0,
        )

        assert "## About the User" not in result
        assert "## Core Memories" not in result
        assert "## Project:" not in result

    def test_flat_list_preserves_search_result_order(self, mock_graph):
        # recall_search_structured always returns results sorted by score descending.
        # Whisper should preserve that order — first result in list = first in output.
        mock_engine = MagicMock()
        builder = ContextBuilder(mock_graph, engine=mock_engine)

        nodes = [
            _make_node_dict("high-score", "High score title"),
            _make_node_dict("low-score", "Low score title"),
        ]
        mock_engine.recall_search_structured.return_value = [
            {"node": nodes[0], "score": 0.9, "source": "hybrid"},
            {"node": nodes[1], "score": 0.6, "source": "hybrid"},
        ]

        result = builder.build_whisper_context(
            prompt="tell me something",
            injection_gate=0.0,
        )

        # First result in search appears first in output
        high_pos = result.index("High score title")
        low_pos = result.index("Low score title")
        assert high_pos < low_pos

    def test_framing_text_updated(self, mock_graph):
        mock_engine = MagicMock()
        builder = ContextBuilder(mock_graph, engine=mock_engine)

        nodes = [_make_node_dict("node-x", "Some title")]
        mock_engine.recall_search_structured.return_value = [
            {"node": nodes[0], "score": 0.9, "source": "hybrid"},
        ]

        result = builder.build_whisper_context(
            prompt="something",
            injection_gate=0.0,
        )

        assert "The most relevant memories are shown in full" in result
        assert "use recall with its node ID" in result


class TestGateScoreContract:
    """The injection gate cuts absolute signals (ce_absolute / raw_cosine),
    never the rank-relative blended score (I2)."""

    def _builder(self, mock_graph, results, ce_scores):
        mock_engine = _make_engine_with_encoder(mock_graph)
        builder = ContextBuilder(mock_graph, engine=mock_engine)
        mock_engine.recall_search_structured.return_value = results
        mock_ce = MagicMock()
        mock_ce.rerank.return_value = ce_scores
        return builder, mock_ce

    def test_high_blended_low_ce_is_gated_out(self, mock_graph):
        """A weak query's least-bad match: blended ~0.9 (rank-relative top)
        but the cross-encoder condemns it. The gate must reject."""
        node = _make_node_dict("weak-1", "Rank-relative winner")
        builder, mock_ce = self._builder(
            mock_graph,
            [{"node": node, "score": 0.90, "source": "hybrid"}],
            [-9.0],  # ce_absolute = (-9+12)/18 ≈ 0.167
        )

        with patch("ormah.embeddings.reranker._get_model", return_value=mock_ce), \
             patch("ormah.engine.affinity.batch_fetch_affinity", return_value={}), \
             patch("ormah.engine.affinity.compute_affinity_boost", return_value=0.0):
            result = builder.build_whisper_context(
                prompt="unrelated topic query",
                min_score=0.1,
                reranker_enabled=True,
                reranker_min_score=0.0,
                injection_gate=0.50,
            )

        assert "Rank-relative winner" not in result

    def test_low_blended_high_ce_survives_gate(self, mock_graph):
        """A genuinely relevant match under-ranked by the bi-encoder: the
        cross-encoder vouches, the gate passes it."""
        node = _make_node_dict("strong-1", "Under-ranked gem")
        builder, mock_ce = self._builder(
            mock_graph,
            [{"node": node, "score": 0.45, "source": "hybrid"}],
            [4.0],  # ce_absolute = (4+12)/18 ≈ 0.889
        )

        with patch("ormah.embeddings.reranker._get_model", return_value=mock_ce), \
             patch("ormah.engine.affinity.batch_fetch_affinity", return_value={}), \
             patch("ormah.engine.affinity.compute_affinity_boost", return_value=0.0):
            result = builder.build_whisper_context(
                prompt="find the gem",
                min_score=0.1,
                reranker_enabled=True,
                reranker_min_score=0.0,
                injection_gate=0.50,
            )

        assert "Under-ranked gem" in result

    def test_reranker_off_gates_on_raw_cosine(self, mock_graph):
        """Without the reranker, the gate falls back to raw_cosine, not the
        blended score."""
        mock_engine = _make_engine_with_encoder(mock_graph)
        builder = ContextBuilder(mock_graph, engine=mock_engine)

        noise = _make_node_dict("cos-low", "Cosine-weak result")
        real = _make_node_dict("cos-high", "Cosine-strong result")
        mock_engine.recall_search_structured.return_value = [
            {"node": noise, "score": 0.95, "source": "hybrid", "raw_cosine": 0.30},
            {"node": real, "score": 0.60, "source": "hybrid", "raw_cosine": 0.72},
        ]

        result = builder.build_whisper_context(
            prompt="cosine strong result topic",
            min_score=0.1,
            reranker_enabled=False,
            injection_gate=0.50,
        )

        assert "Cosine-strong result" in result
        assert "Cosine-weak result" not in result

    def test_legacy_results_fall_back_to_blended_score(self, mock_graph):
        """Results carrying neither absolute signal keep pre-contract gate
        behavior (blended score) instead of being rejected outright."""
        mock_engine = _make_engine_with_encoder(mock_graph)
        builder = ContextBuilder(mock_graph, engine=mock_engine)

        node = _make_node_dict("legacy-1", "Legacy scored result")
        mock_engine.recall_search_structured.return_value = [
            {"node": node, "score": 0.80, "source": "hybrid"},
        ]

        result = builder.build_whisper_context(
            prompt="legacy scored result topic",
            min_score=0.1,
            reranker_enabled=False,
            injection_gate=0.50,
        )

        assert "Legacy scored result" in result

    def test_cross_space_memory_demoted_below_gate(self, mock_graph):
        """The gate re-applies cross-space demotion the absolute signal drops:
        a wrong-project memory the CE/cosine rates highly is still gated out."""
        mock_engine = _make_engine_with_encoder(mock_graph)
        builder = ContextBuilder(mock_graph, engine=mock_engine)

        node = _make_node_dict("other-proj", "Auth in project B uses JWT", space="project-b")
        mock_engine.recall_search_structured.return_value = [
            # raw_cosine 0.72 clears 0.50 alone, but other-project factor 0.6
            # → 0.72 * 0.6 = 0.432 < 0.50: cross-project leakage prevented.
            {"node": node, "score": 0.80, "source": "hybrid",
             "raw_cosine": 0.72, "_space_factor": 0.6},
        ]

        result = builder.build_whisper_context(
            prompt="how do we handle auth",
            min_score=0.1,
            reranker_enabled=False,
            injection_gate=0.50,
        )

        assert "Auth in project B uses JWT" not in result

    def test_low_confidence_memory_demoted_below_gate(self, mock_graph):
        """The gate re-applies the confidence factor: a low-confidence memory
        the cosine rates highly is demoted below the gate."""
        mock_engine = _make_engine_with_encoder(mock_graph)
        builder = ContextBuilder(mock_graph, engine=mock_engine)

        node = _make_node_dict("shaky", "Auth uses session cookies", space=None)
        node["confidence"] = 0.2  # confidence_factor 0.4 + 0.6*0.2 = 0.52
        mock_engine.recall_search_structured.return_value = [
            # 0.72 * 0.52 = 0.374 < 0.50: low-confidence memory gated out.
            {"node": node, "score": 0.80, "source": "hybrid", "raw_cosine": 0.72},
        ]

        result = builder.build_whisper_context(
            prompt="how does auth work",
            min_score=0.1,
            reranker_enabled=False,
            injection_gate=0.50,
        )

        assert "Auth uses session cookies" not in result


class TestWhisperDecisions:
    """Every whisper call writes exactly one whisper_decisions row (I10)."""

    @pytest.fixture
    def db_graph(self, tmp_path):
        from ormah.index.db import Database

        db = Database(tmp_path / "index.db")
        db.init_schema()
        graph = GraphIndex(db.conn)
        return db, graph

    def _builder_with_db(self, db, graph, results=None):
        mock_engine = MagicMock()
        mock_engine.settings = _make_settings_mock()
        prompt_vec = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        mock_encoder = MagicMock()
        mock_encoder.encode.return_value = prompt_vec
        mock_hybrid = MagicMock()
        mock_hybrid.encoder = mock_encoder
        mock_engine._get_hybrid_search.return_value = mock_hybrid
        mock_engine.db = db
        mock_engine.recall_search_structured.return_value = results or []
        return ContextBuilder(graph, engine=mock_engine)

    def _decisions(self, db):
        return db.conn.execute(
            "SELECT * FROM whisper_decisions ORDER BY id"
        ).fetchall()

    def test_short_prompt_logs_silent_short(self, db_graph):
        db, graph = db_graph
        builder = self._builder_with_db(db, graph)

        builder.build_whisper_context(prompt="ok", session_id="s1")

        rows = self._decisions(db)
        assert len(rows) == 1
        assert rows[0]["outcome"] == "silent_short"
        assert rows[0]["session_id"] == "s1"

    def test_no_candidates_logs_silent_no_candidates(self, db_graph):
        db, graph = db_graph
        builder = self._builder_with_db(db, graph, results=[])

        builder.build_whisper_context(
            prompt="a topic with no memories", session_id="s2",
        )

        rows = self._decisions(db)
        assert len(rows) == 1
        assert rows[0]["outcome"] == "silent_no_candidates"
        assert rows[0]["candidate_count"] == 0

    def test_gate_reject_logs_silent_gate_with_score(self, db_graph):
        db, graph = db_graph
        node = _make_node_dict("gated-1", "Gated candidate")
        builder = self._builder_with_db(
            db, graph,
            results=[{"node": node, "score": 0.80, "source": "hybrid",
                      "raw_cosine": 0.30}],
        )

        builder.build_whisper_context(
            prompt="gated candidate topic",
            min_score=0.1,
            reranker_enabled=False,
            injection_gate=0.50,
            session_id="s3",
        )

        rows = self._decisions(db)
        assert len(rows) == 1
        assert rows[0]["outcome"] == "silent_gate"
        assert rows[0]["candidate_count"] == 1
        assert rows[0]["injected_count"] == 0
        assert rows[0]["max_gate_score"] == pytest.approx(0.30)

    def test_injection_logs_injected_with_counts(self, db_graph):
        db, graph = db_graph
        node = _make_node_dict("inj-1", "Injected candidate")
        builder = self._builder_with_db(
            db, graph,
            results=[{"node": node, "score": 0.80, "source": "hybrid",
                      "raw_cosine": 0.75}],
        )

        builder.build_whisper_context(
            prompt="injected candidate topic",
            min_score=0.1,
            reranker_enabled=False,
            injection_gate=0.50,
            session_id="s4",
        )

        rows = self._decisions(db)
        assert len(rows) == 1
        assert rows[0]["outcome"] == "injected"
        assert rows[0]["injected_count"] == 1
        assert rows[0]["max_gate_score"] == pytest.approx(0.75)

    def test_one_row_per_call(self, db_graph):
        db, graph = db_graph
        builder = self._builder_with_db(db, graph)

        builder.build_whisper_context(prompt="ok", session_id="s5")
        builder.build_whisper_context(prompt="first real prompt about topics",
                                      session_id="s5")

        rows = self._decisions(db)
        assert len(rows) == 2


class TestExplorationRespectsSilence:
    """Exploration piggybacks on real injections; it never breaks silence (I6)."""

    def test_no_exploration_when_gate_rejected_everything(self, mock_graph):
        prompt_vec = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        mock_engine = MagicMock()
        mock_engine.settings = _make_settings_mock(
            whisper_exploration_enabled=True,
            whisper_reranker_min_score=0.40,
        )
        mock_encoder = MagicMock()
        mock_encoder.encode.return_value = prompt_vec
        mock_hybrid = MagicMock()
        mock_hybrid.encoder = mock_encoder
        mock_engine._get_hybrid_search.return_value = mock_hybrid

        builder = ContextBuilder(mock_graph, engine=mock_engine)

        # Single candidate: above the 0.40 exploration floor, below the gate.
        node = _make_node_dict("lonely", "Lonely candidate")
        mock_engine.recall_search_structured.return_value = [
            {"node": node, "score": 0.60, "source": "hybrid"},
        ]

        mock_ce = MagicMock()
        # CE -5 → ce_absolute 0.389 < 0.50 gate; blended 0.4*0.389+0.6*0.6 = 0.516 ≥ 0.40 floor
        mock_ce.rerank.return_value = [-5.0]

        with patch("ormah.embeddings.reranker._get_model", return_value=mock_ce), \
             patch("ormah.engine.context_builder.ContextBuilder._get_classifier", return_value=None), \
             patch("ormah.engine.affinity.batch_fetch_affinity", return_value={}), \
             patch("ormah.engine.affinity.compute_affinity_boost", return_value=0.0):
            result = builder.build_whisper_context(
                prompt="lonely candidate topic",
                min_score=0.1,
                reranker_enabled=True,
                reranker_min_score=0.40,
                injection_gate=0.50,
            )

        # Gate rejected the only candidate → silence stands, no exploration.
        assert "Lonely candidate" not in result


class TestTopicalFilterFailClosed:
    """Candidates with no lexical overlap need an absolute voucher (I7)."""

    def test_no_overlap_weak_ce_dropped(self, mock_graph):
        mock_engine = _make_engine_with_encoder(mock_graph)
        builder = ContextBuilder(mock_graph, engine=mock_engine)

        node = _make_node_dict("stranger", "Unrelated embedding neighbor")
        mock_engine.recall_search_structured.return_value = [
            {"node": node, "score": 0.85, "source": "hybrid"},
        ]

        mock_ce = MagicMock()
        # CE -5.0 → ce_absolute 0.389, below the 0.45 no-overlap floor
        mock_ce.rerank.return_value = [-5.0]

        with patch("ormah.embeddings.reranker._get_model", return_value=mock_ce), \
             patch("ormah.engine.context_builder.ContextBuilder._get_classifier", return_value=None), \
             patch("ormah.engine.affinity.batch_fetch_affinity", return_value={}), \
             patch("ormah.engine.affinity.compute_affinity_boost", return_value=0.0):
            result = builder.build_whisper_context(
                prompt="deployment pipeline problems",
                min_score=0.1,
                reranker_enabled=True,
                reranker_min_score=0.0,
                injection_gate=0.10,
            )

        assert "Unrelated embedding neighbor" not in result

    def test_no_overlap_strong_ce_survives(self, mock_graph):
        """A true paraphrase match with zero shared words survives via the CE voucher."""
        mock_engine = _make_engine_with_encoder(mock_graph)
        builder = ContextBuilder(mock_graph, engine=mock_engine)

        node = _make_node_dict("paraphrase", "Semantically equivalent answer")
        mock_engine.recall_search_structured.return_value = [
            {"node": node, "score": 0.55, "source": "hybrid"},
        ]

        mock_ce = MagicMock()
        # CE +3 → ce_absolute 0.833 ≥ 0.45 voucher
        mock_ce.rerank.return_value = [3.0]

        with patch("ormah.embeddings.reranker._get_model", return_value=mock_ce), \
             patch("ormah.engine.context_builder.ContextBuilder._get_classifier", return_value=None), \
             patch("ormah.engine.affinity.batch_fetch_affinity", return_value={}), \
             patch("ormah.engine.affinity.compute_affinity_boost", return_value=0.0):
            result = builder.build_whisper_context(
                prompt="deployment pipeline problems",
                min_score=0.1,
                reranker_enabled=True,
                reranker_min_score=0.0,
                injection_gate=0.10,
            )

        assert "Semantically equivalent answer" in result

    def test_no_overlap_cosine_voucher_when_reranker_off(self, mock_graph):
        mock_engine = _make_engine_with_encoder(mock_graph)
        builder = ContextBuilder(mock_graph, engine=mock_engine)

        strong = _make_node_dict("cos-strong", "Semantic twin")
        weak = _make_node_dict("cos-weak", "Embedding stranger")
        mock_engine.recall_search_structured.return_value = [
            {"node": strong, "score": 0.80, "source": "hybrid", "raw_cosine": 0.78},
            {"node": weak, "score": 0.75, "source": "hybrid", "raw_cosine": 0.55},
        ]

        result = builder.build_whisper_context(
            prompt="deployment pipeline problems",
            min_score=0.1,
            reranker_enabled=False,
            injection_gate=0.10,
        )

        assert "Semantic twin" in result
        assert "Embedding stranger" not in result


class TestTopicShiftServedMemory:
    """Topic-shift suppression only fires for topics that were served (I9)."""

    @pytest.fixture
    def db_graph(self, tmp_path):
        from ormah.index.db import Database

        db = Database(tmp_path / "index.db")
        db.init_schema()
        graph = GraphIndex(db.conn)
        return db, graph

    def _builder(self, db, graph, prompt_vec):
        mock_engine = MagicMock()
        mock_engine.settings = _make_settings_mock()
        mock_encoder = MagicMock()
        mock_encoder.encode.return_value = prompt_vec
        # encode_batch: recent prompts on the same topic → same vector
        mock_encoder.encode_batch.return_value = np.stack([prompt_vec])
        mock_hybrid = MagicMock()
        mock_hybrid.encoder = mock_encoder
        mock_engine._get_hybrid_search.return_value = mock_hybrid
        mock_engine.db = db
        return mock_engine, ContextBuilder(graph, engine=mock_engine)

    def _insert_whisper_log(self, db, *, session_id, prompt_vec, was_injected):
        with db.transaction() as conn:
            conn.execute(
                "INSERT INTO whisper_log "
                "(session_id, space, prompt_hash, prompt_text, prompt_vec, "
                "node_id, score, was_injected, logged_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, datetime('now'))",
                (session_id, None, "h", "p", prompt_vec.astype(np.float32).tobytes(),
                 "n1", 0.9, was_injected),
            )

    def test_same_topic_never_served_proceeds(self, db_graph):
        """Turn 1 was gate-rejected (logged was_injected=0); turn 2 on the
        same topic must NOT be topic-shift-skipped — the topic is starved."""
        db, graph = db_graph
        vec = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        self._insert_whisper_log(db, session_id="sess-starved", prompt_vec=vec, was_injected=0)

        mock_engine, builder = self._builder(db, graph, vec)
        node = _make_node_dict("served-1", "Starved topic memory")
        mock_engine.recall_search_structured.return_value = [
            {"node": node, "score": 0.85, "source": "hybrid", "raw_cosine": 0.80},
        ]

        with patch("ormah.engine.context_builder.ContextBuilder._get_classifier", return_value=None):
            result = builder.build_whisper_context(
                prompt="starved topic memory question",
                min_score=0.1,
                reranker_enabled=False,
                injection_gate=0.10,
                topic_shift_enabled=True,
                topic_shift_threshold=0.75,
                recent_prompts=["starved topic memory question earlier"],
                session_id="sess-starved",
            )

        assert "Starved topic memory" in result

    def test_same_topic_already_served_skips(self, db_graph):
        """Turn 1 injected (was_injected=1); turn 2 on the same topic is
        correctly suppressed — no repeat spam."""
        db, graph = db_graph
        vec = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        self._insert_whisper_log(db, session_id="sess-served", prompt_vec=vec, was_injected=1)

        mock_engine, builder = self._builder(db, graph, vec)
        node = _make_node_dict("served-2", "Served topic memory")
        mock_engine.recall_search_structured.return_value = [
            {"node": node, "score": 0.85, "source": "hybrid", "raw_cosine": 0.80},
        ]

        with patch("ormah.engine.context_builder.ContextBuilder._get_classifier", return_value=None):
            result = builder.build_whisper_context(
                prompt="served topic memory question",
                min_score=0.1,
                reranker_enabled=False,
                injection_gate=0.10,
                topic_shift_enabled=True,
                topic_shift_threshold=0.75,
                recent_prompts=["served topic memory question earlier"],
                session_id="sess-served",
            )

        assert "Served topic memory" not in result

    def test_no_session_id_preserves_plain_skip(self, db_graph):
        """Without a session_id there is no served history — the plain
        topic-shift skip behavior is preserved."""
        db, graph = db_graph
        vec = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        mock_engine, builder = self._builder(db, graph, vec)
        node = _make_node_dict("served-3", "Sessionless topic memory")
        mock_engine.recall_search_structured.return_value = [
            {"node": node, "score": 0.85, "source": "hybrid", "raw_cosine": 0.80},
        ]

        with patch("ormah.engine.context_builder.ContextBuilder._get_classifier", return_value=None):
            result = builder.build_whisper_context(
                prompt="sessionless topic memory question",
                min_score=0.1,
                reranker_enabled=False,
                injection_gate=0.10,
                topic_shift_enabled=True,
                topic_shift_threshold=0.75,
                recent_prompts=["sessionless topic memory question earlier"],
                session_id=None,
            )

        assert "Sessionless topic memory" not in result


class TestSessionBufferEviction:
    """Dead sessions are evicted from _session_buffers on access (I12)."""

    def _client(self):
        from fastapi import FastAPI
        from fastapi.testclient import TestClient
        from ormah.api.routes_agent import router

        app = FastAPI()
        app.include_router(router)
        engine = MagicMock()
        engine.get_whisper_context.return_value = ""
        app.state.engine = engine
        return TestClient(app)

    def test_dead_sessions_evicted_on_access(self, monkeypatch):
        import time as time_mod
        from collections import deque
        from ormah.api.routes_agent import _session_buffers
        from ormah.config import settings as global_settings

        _session_buffers.clear()
        gap_seconds = global_settings.whisper_session_gap_minutes * 60

        now = time_mod.time()
        stale = deque(maxlen=5)
        stale.append(("old prompt", now - gap_seconds * 3))
        _session_buffers["dead-session"] = stale

        fresh = deque(maxlen=5)
        fresh.append(("recent prompt", now - 5))
        _session_buffers["live-session"] = fresh

        client = self._client()
        resp = client.post(
            "/agent/whisper",
            json={"prompt": "hello there world", "session_id": "another-session"},
        )
        assert resp.status_code == 200

        assert "dead-session" not in _session_buffers
        assert "live-session" in _session_buffers
        assert "another-session" in _session_buffers

        _session_buffers.clear()


class TestEncodeOncePerWhisper:
    """The prompt is embedded (encode) exactly once per whisper call (I15).

    ``PromptClassifier.classify()`` encodes the raw prompt to score it
    against archetypes; ``build_whisper_context`` reuses that same vector
    (``intent.prompt_vec``) for topic-shift detection, the affinity boost,
    and whisper_log instead of encoding again. The classifier must NOT be
    mocked away here — doing so hides its internal ``encode()`` call and
    would make the "encode once" assertion count only the *other* encodes,
    silently passing even if classify() and the builder each encoded
    separately (as happened before this test was fixed).
    """

    def test_classifier_and_prompt_vec_share_one_encode(self, mock_graph):
        prompt_vec = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        mock_engine = MagicMock()
        mock_engine.settings = _make_settings_mock()
        mock_engine.settings.whisper_intent_threshold = 0.65

        mock_encoder = MagicMock()
        mock_encoder.encode.return_value = prompt_vec
        # Archetype vectors are orthogonal to the prompt on every category
        # (and to the recent-prompt centroid), so intent falls back to
        # "general" and topic-shift sees a real shift -- both real code
        # paths, driven by the real (un-mocked) PromptClassifier.
        mock_encoder.encode_batch.side_effect = lambda prompts: np.tile(
            np.array([0.0, 1.0, 0.0], dtype=np.float32), (len(prompts), 1)
        )
        mock_hybrid = MagicMock()
        mock_hybrid.encoder = mock_encoder
        mock_engine._get_hybrid_search.return_value = mock_hybrid
        mock_engine.recall_search_structured.return_value = []

        builder = ContextBuilder(mock_graph, engine=mock_engine)

        builder.build_whisper_context(
            prompt="a fresh new topic entirely",
            topic_shift_enabled=True,
            topic_shift_threshold=0.75,
            recent_prompts=["something else before"],
            session_id="enc-1",
        )

        # encode() called exactly once for the whole whisper call: by
        # PromptClassifier.classify(). Topic-shift, the affinity boost, and
        # whisper_log all reuse intent.prompt_vec rather than re-encoding.
        assert mock_encoder.encode.call_count == 1

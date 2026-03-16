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
        # Only 2 results above threshold
        mock_engine.recall_search_structured.return_value = [
            {"node": nodes[0], "score": 0.8, "source": "hybrid"},
            {"node": nodes[1], "score": 0.5, "source": "hybrid"},
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


class TestWhisperIdentityCap:
    """Whisper should cap identity nodes tightly."""

    def test_identity_capped_to_max(self, mock_graph):
        mock_engine = MagicMock()
        builder = ContextBuilder(mock_graph, engine=mock_engine)
        conn = mock_graph.conn

        # Create user node
        user_node = _make_node_dict("user-1", "Self", node_type="person")
        _insert_node(conn, user_node)

        # Create 8 identity nodes (mix of person/preference/fact)
        identity_nodes = []
        for i in range(8):
            ntype = "preference" if i < 2 else "fact"
            node = _make_node_dict(f"id-{i}", f"Identity {i}", node_type=ntype)
            _insert_node(conn, node)
            identity_nodes.append(node)
            conn.execute(
                "INSERT INTO edges (source_id, target_id, edge_type, weight, created) "
                "VALUES (?, ?, 'defines', 1.0, '2026-01-01T00:00:00Z')",
                ("user-1", f"id-{i}"),
            )
        conn.commit()

        # All identity nodes returned by search with high scores
        mock_engine.recall_search_structured.return_value = [
            {"node": n, "score": 0.9, "source": "hybrid"} for n in identity_nodes
        ]

        result = builder.build_whisper_context(
            prompt="tell me about myself",
            user_node_id="user-1",
            identity_max=5,
            min_score=0.1,
        )

        # 2 preferences (always kept) + 3 facts (capped) = 5 total
        identity_count = sum(1 for i in range(8) if f"Identity {i}" in result)
        assert identity_count <= 5

    def test_person_preference_always_kept(self, mock_graph):
        """Preferences should be kept when topical results also exist."""
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

        topical = _make_node_dict("fact-1", "Auth module info")
        mock_engine.recall_search_structured.return_value = [
            {"node": pref, "score": 0.5, "source": "hybrid"},
            {"node": topical, "score": 0.7, "source": "hybrid"},
        ]

        result = builder.build_whisper_context(
            prompt="how does auth work",
            user_node_id="user-1",
            identity_max=5,
            min_score=0.1,
        )

        assert "Prefers dark mode" in result


class TestWhisperCompactFormatting:
    """Whisper should use compact content truncation."""

    def test_content_truncated_to_max_content_len(self, mock_graph):
        mock_engine = MagicMock()
        builder = ContextBuilder(mock_graph, engine=mock_engine)

        long_content = "A" * 500
        node = _make_node_dict("node-1", "Long content node")
        node["content"] = long_content

        mock_engine.recall_search_structured.return_value = [
            {"node": node, "score": 0.9, "source": "hybrid"},
        ]

        result = builder.build_whisper_context(
            prompt="test",
            max_content_len=150,
            min_score=0.1,
        )

        # The formatted output should not contain the full 500-char content
        assert long_content not in result
        # But should contain a truncated version
        assert "A" * 150 in result


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
            identity_max=5,
            min_score=0.1,
        )

        # recall_search_structured called with limit=max_nodes+identity_max=8
        mock_engine.recall_search_structured.assert_called_once()
        call_kwargs = mock_engine.recall_search_structured.call_args
        limit = call_kwargs.kwargs.get("limit") or call_kwargs[1].get("limit")
        assert limit == 8  # max_nodes(3) + identity_max(5)

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
            identity_max=5,
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
    """Whisper cross-encoder reranking with sigmoid-blended scoring."""

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

        # All results preserved — blended scores fall back to embedding
        # when CE is negative (sigmoid(-0.5)≈0.38, blended≈0.45)
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
        # node-2: CE=-10 → sigmoid≈0, emb=0.2 → blended=0.4*0+0.6*0.2=0.12
        mock_cross_encoder.rerank.return_value = [2.0, -1.0, -10.0]

        with patch("ormah.embeddings.reranker._get_model", return_value=mock_cross_encoder):
            result = builder.build_whisper_context(
                prompt="specific query",
                min_score=0.1,
                injection_gate=0.1,  # low gate to isolate reranker behavior
                reranker_enabled=True,
                reranker_model="cross-encoder/ms-marco-MiniLM-L-6-v2",
                reranker_min_score=0.15,
            )

        # node-0: blended = 0.4*sigmoid(2)+0.6*0.8 ≈ 0.4*0.88+0.48 = 0.83 ✓
        assert "Fact 0" in result
        # node-1: blended = 0.4*sigmoid(-1)+0.6*0.5 ≈ 0.4*0.27+0.30 = 0.41 ✓
        assert "Fact 1" in result
        # node-2: blended = 0.4*sigmoid(-10)+0.6*0.2 ≈ 0.4*0.00+0.12 = 0.12 ✗
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
        mock_cross_encoder.rerank.return_value = [0.8, 0.1, 0.05]

        with patch("ormah.embeddings.reranker._get_model", return_value=mock_cross_encoder):
            result = builder.build_whisper_context(
                prompt="specific query",
                min_score=0.1,
                reranker_enabled=True,
                reranker_model="cross-encoder/ms-marco-MiniLM-L-6-v2",
                reranker_min_score=0.7,
            )

        # node-0: blended = 0.4*sigmoid(0.8)+0.6*0.8 ≈ 0.4*0.69+0.48 = 0.756 ✓
        assert "Fact 0" in result
        # node-1: blended = 0.4*sigmoid(0.1)+0.6*0.7 ≈ 0.4*0.525+0.42 = 0.630 ✗
        assert "Fact 1" not in result
        # node-2: blended = 0.4*sigmoid(0.05)+0.6*0.6 ≈ 0.4*0.512+0.36 = 0.565 ✗
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
    """Whisper with space should format project section."""

    def test_with_space_formats_project_section(self, mock_graph):
        mock_engine = MagicMock()
        builder = ContextBuilder(mock_graph, engine=mock_engine)

        core_node = _make_node_dict("core-1", "Core fact", tier="core")
        working_node = _make_node_dict("work-1", "Project detail", tier="working", space="myproj")

        mock_engine.recall_search_structured.return_value = [
            {"node": core_node, "score": 0.8, "source": "hybrid"},
            {"node": working_node, "score": 0.7, "source": "hybrid"},
        ]

        result = builder.build_whisper_context(
            prompt="project stuff",
            space="myproj",
            min_score=0.1,
        )

        assert "Core fact" in result
        assert "Project detail" in result
        assert "myproj" in result


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
        # Should still return identity info from graph neighbors
        assert "Likes dark mode" in result

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
        results are suppressed — the CE is confidently saying 'off-topic'."""
        mock_engine = MagicMock()
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
        mock_ce.rerank.return_value = [-5.0]

        # With α=0.4 (default): 0.4*sigmoid(-5)+0.6*0.3 ≈ 0.003+0.18 = 0.183
        # With α=0.9: 0.9*sigmoid(-5)+0.1*0.3 ≈ 0.006+0.03 = 0.036
        with patch("ormah.embeddings.reranker._get_model", return_value=mock_ce):
            # Default alpha: should pass min_score=0.15
            result_default = builder.build_whisper_context(
                prompt="test",
                min_score=0.1,
                injection_gate=0.1,  # low gate to isolate reranker behavior
                reranker_enabled=True,
                reranker_min_score=0.15,
                reranker_blend_alpha=0.4,
            )
            # High alpha: CE dominates → should fail min_score=0.15
            result_high_alpha = builder.build_whisper_context(
                prompt="test",
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
        """Embedding min_score filters before reranker; reranker min_score
        filters blended scores after."""
        mock_engine = MagicMock()
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
        mock_ce.rerank.return_value = [3.0, -8.0]

        with patch("ormah.embeddings.reranker._get_model", return_value=mock_ce):
            result = builder.build_whisper_context(
                prompt="test query",
                min_score=0.45,  # filters low_emb before reranker
                injection_gate=0.1,  # low gate to isolate reranker behavior
                reranker_enabled=True,
                reranker_min_score=0.0,
            )

        assert "Relevant fact" in result
        assert "Somewhat relevant" in result
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
            result = builder.build_whisper_context(
                prompt="test query",
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

    def test_identity_only_intent_bypasses_gating(self, mock_graph):
        """identity-only intent should return identity even without topical results."""
        mock_engine = MagicMock()
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

        assert "Likes dark mode" in result


class TestWhisperContextBuffer:
    """Context-enhanced search using recent prompts."""

    def test_recent_prompts_enhance_search_query(self, mock_graph):
        """recent_prompts should be joined with current prompt for search."""
        mock_engine = MagicMock()
        builder = ContextBuilder(mock_graph, engine=mock_engine)

        nodes = [_make_node_dict("node-0", "Whisper quality metrics")]
        mock_engine.recall_search_structured.return_value = [
            {"node": nodes[0], "score": 0.8, "source": "hybrid"},
        ]

        builder.build_whisper_context(
            prompt="closer to you liking it more?",
            min_score=0.1,
            recent_prompts=["how's whisper quality?", "show me the eval results"],
        )

        call_kwargs = mock_engine.recall_search_structured.call_args
        query = call_kwargs.kwargs.get("query") or call_kwargs[1].get("query")
        # Query should contain context from recent prompts
        assert "whisper quality" in query
        assert "eval results" in query
        assert "closer to you liking it more?" in query

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

    def test_recent_prompts_capped_at_3(self, mock_graph):
        """Only the last 3 recent prompts should be used."""
        mock_engine = MagicMock()
        builder = ContextBuilder(mock_graph, engine=mock_engine)

        mock_engine.recall_search_structured.return_value = []

        builder.build_whisper_context(
            prompt="current",
            min_score=0.1,
            recent_prompts=["old1", "old2", "old3", "old4", "old5"],
        )

        call_kwargs = mock_engine.recall_search_structured.call_args
        query = call_kwargs.kwargs.get("query") or call_kwargs[1].get("query")
        # Should only contain last 3 + current
        assert "old1" not in query
        assert "old2" not in query
        assert "old3" in query
        assert "old4" in query
        assert "old5" in query
        assert "current" in query


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


class TestWhisperDynamicContentBudget:
    """Dynamic content budget: distribute total chars across results."""

    def test_few_results_get_more_chars(self, mock_graph):
        """1-2 results should get up to max_per_node chars each."""
        mock_engine = MagicMock()
        builder = ContextBuilder(mock_graph, engine=mock_engine)

        long_content = "A" * 600
        node = _make_node_dict("node-1", "Single result")
        node["content"] = long_content

        mock_engine.recall_search_structured.return_value = [
            {"node": node, "score": 0.9, "source": "hybrid"},
        ]

        result = builder.build_whisper_context(
            prompt="test",
            min_score=0.1,
            content_total_budget=1200,
            content_min_per_node=100,
            content_max_per_node=500,
        )

        # With 1 result: 1200/1 = 1200, clamped to 500
        # Should contain at least 500 chars of content (capped at max_per_node)
        assert "A" * 500 in result
        # But NOT the full 600
        assert "A" * 600 not in result

    def test_many_results_get_fewer_chars(self, mock_graph):
        """8+ results should get ~150 chars each (like current behavior)."""
        mock_engine = MagicMock()
        builder = ContextBuilder(mock_graph, engine=mock_engine)

        nodes = []
        for i in range(8):
            node = _make_node_dict(f"node-{i}", f"Fact {i}")
            node["content"] = "B" * 300
            nodes.append(node)

        mock_engine.recall_search_structured.return_value = [
            {"node": n, "score": 0.9, "source": "hybrid"} for n in nodes
        ]

        result = builder.build_whisper_context(
            prompt="test",
            min_score=0.1,
            content_total_budget=1200,
            content_min_per_node=100,
            content_max_per_node=500,
        )

        # With 8 results: 1200/8 = 150, clamped between 100 and 500
        # Should contain 150 chars of content, not 300
        assert "B" * 300 not in result
        assert "B" * 150 in result

    def test_budget_zero_uses_max_content_len(self, mock_graph):
        """content_total_budget=0 → fall back to max_content_len."""
        mock_engine = MagicMock()
        builder = ContextBuilder(mock_graph, engine=mock_engine)

        node = _make_node_dict("node-1", "A fact")
        node["content"] = "C" * 300

        mock_engine.recall_search_structured.return_value = [
            {"node": node, "score": 0.9, "source": "hybrid"},
        ]

        result = builder.build_whisper_context(
            prompt="test",
            min_score=0.1,
            max_content_len=150,
            content_total_budget=0,
        )

        # Should use max_content_len=150 (budget disabled)
        assert "C" * 150 in result
        assert "C" * 300 not in result

    def test_budget_respects_min_per_node(self, mock_graph):
        """With many results, per-node budget should not go below min_per_node."""
        mock_engine = MagicMock()
        builder = ContextBuilder(mock_graph, engine=mock_engine)

        nodes = []
        for i in range(20):
            node = _make_node_dict(f"node-{i}", f"Fact {i}")
            node["content"] = "D" * 200
            nodes.append(node)

        mock_engine.recall_search_structured.return_value = [
            {"node": n, "score": 0.9, "source": "hybrid"} for n in nodes
        ]

        result = builder.build_whisper_context(
            prompt="test",
            min_score=0.1,
            max_nodes=20,
            identity_max=0,
            content_total_budget=1200,
            content_min_per_node=100,
            content_max_per_node=500,
        )

        # With 20 results: 1200/20 = 60, clamped to min 100
        # Should contain at least 100 chars of content per node
        assert "D" * 100 in result

    def test_budget_with_identity_and_other_results(self, mock_graph):
        """Budget should count both identity and non-identity results."""
        mock_engine = MagicMock()
        builder = ContextBuilder(mock_graph, engine=mock_engine)
        conn = mock_graph.conn

        user_node = _make_node_dict("user-1", "Self", node_type="person")
        _insert_node(conn, user_node)

        pref = _make_node_dict("pref-1", "User preference")
        pref["content"] = "E" * 400
        _insert_node(conn, pref)
        conn.execute(
            "INSERT INTO edges (source_id, target_id, edge_type, weight, created) "
            "VALUES (?, ?, 'defines', 1.0, '2026-01-01T00:00:00Z')",
            ("user-1", "pref-1"),
        )
        conn.commit()

        other = _make_node_dict("fact-1", "Tech fact")
        other["content"] = "F" * 400

        mock_engine.recall_search_structured.return_value = [
            {"node": pref, "score": 0.8, "source": "hybrid"},
            {"node": other, "score": 0.7, "source": "hybrid"},
        ]

        result = builder.build_whisper_context(
            prompt="test",
            user_node_id="user-1",
            min_score=0.1,
            content_total_budget=1200,
            content_min_per_node=100,
            content_max_per_node=500,
        )

        # 2 total results → 1200/2 = 600, clamped to 500
        # Both should have up to 500 chars, not full 400 (both under cap)
        assert "User preference" in result
        assert "Tech fact" in result

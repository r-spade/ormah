"""Tests for _return_debug mode on build_whisper_context."""
from __future__ import annotations

import pytest
from unittest.mock import MagicMock

from ormah.engine.context_builder import ContextBuilder
from ormah.index.graph import GraphIndex


def _make_node(node_id, title, node_type="fact"):
    return {
        "id": node_id, "type": node_type, "tier": "working",
        "title": title, "content": f"Content for {title}.",
        "space": None, "importance": 0.5, "confidence": 1.0,
        "valid_until": None, "source": "test", "access_count": 0,
        "last_accessed": "2026-01-01T00:00:00Z",
        "created": "2026-01-01T00:00:00Z",
        "updated": "2026-01-01T00:00:00Z",
    }


@pytest.fixture
def mock_graph(tmp_path):
    from ormah.index.db import Database
    db = Database(tmp_path / "index.db")
    db.init_schema()
    return GraphIndex(db.conn)


class TestWhisperDebugMode:
    def test_debug_returns_tuple(self, mock_graph):
        mock_engine = MagicMock()
        builder = ContextBuilder(mock_graph, engine=mock_engine)
        node = _make_node("node-abc12345", "Port fact")
        mock_engine._search_structured.return_value = [
            {"node": node, "score": 0.9, "source": "hybrid"},
        ]
        result = builder.build_whisper_context(
            prompt="what port", injection_gate=0.0, _return_debug=True
        )
        assert isinstance(result, tuple)
        whisper_text, injected_ids = result
        assert isinstance(whisper_text, str)
        assert injected_ids == ["node-abc12345"]

    def test_debug_suppressed_returns_empty_ids(self, mock_graph):
        mock_engine = MagicMock()
        mock_engine.settings.claude_maintenance_enabled = False
        builder = ContextBuilder(mock_graph, engine=mock_engine)
        mock_engine._search_structured.return_value = []
        whisper_text, injected_ids = builder.build_whisper_context(
            prompt="what port", _return_debug=True
        )
        assert whisper_text == ""
        assert injected_ids == []

    def test_debug_multiple_nodes_returns_all_ids(self, mock_graph):
        mock_engine = MagicMock()
        builder = ContextBuilder(mock_graph, engine=mock_engine)
        nodes = [_make_node(f"node-{i:08d}", f"Fact {i}") for i in range(3)]
        mock_engine._search_structured.return_value = [
            {"node": n, "score": 0.9 - i * 0.1, "source": "hybrid"}
            for i, n in enumerate(nodes)
        ]
        _, injected_ids = builder.build_whisper_context(
            prompt="some query", injection_gate=0.0, _return_debug=True
        )
        assert injected_ids == ["node-00000000", "node-00000001", "node-00000002"]

    def test_no_debug_returns_string(self, mock_graph):
        mock_engine = MagicMock()
        builder = ContextBuilder(mock_graph, engine=mock_engine)
        node = _make_node("node-1", "A fact")
        mock_engine._search_structured.return_value = [
            {"node": node, "score": 0.9, "source": "hybrid"},
        ]
        result = builder.build_whisper_context(
            prompt="query", injection_gate=0.0, _return_debug=False
        )
        assert isinstance(result, str)

    def test_debug_ids_respect_gate(self, mock_graph):
        """Nodes that don't clear the injection gate should not appear in injected_ids."""
        mock_engine = MagicMock()
        builder = ContextBuilder(mock_graph, engine=mock_engine)
        nodes = [_make_node(f"node-{i}", f"Fact {i}") for i in range(2)]
        mock_engine._search_structured.return_value = [
            {"node": nodes[0], "score": 0.8, "source": "hybrid"},
            {"node": nodes[1], "score": 0.3, "source": "hybrid"},
        ]
        _, injected_ids = builder.build_whisper_context(
            prompt="query", min_score=0.1, injection_gate=0.55, _return_debug=True
        )
        assert "node-0" in injected_ids
        assert "node-1" not in injected_ids

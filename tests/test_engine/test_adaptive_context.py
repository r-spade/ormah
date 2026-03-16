"""Tests for adaptive context injection with task_hint."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from ormah.engine.context_builder import ContextBuilder
from ormah.index.graph import GraphIndex


def _make_node_dict(node_id, title, tier="core", space=None, importance=0.5):
    return {
        "id": node_id,
        "type": "fact",
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


class TestWithoutHint:
    """Without task_hint, context is capped to max_nodes."""

    def test_without_hint_returns_core_within_cap(self, mock_graph):
        """With fewer nodes than the cap, all core nodes appear."""
        builder = ContextBuilder(mock_graph)
        conn = mock_graph.conn

        for i in range(5):
            _insert_node(conn, _make_node_dict(f"node-{i}", f"Fact {i}"))
        conn.commit()

        result = builder.build_core_context(max_nodes=20)
        for i in range(5):
            assert f"Fact {i}" in result

    def test_without_hint_caps_to_max_nodes(self, mock_graph):
        """When core + working exceeds max_nodes, output is capped."""
        builder = ContextBuilder(mock_graph)
        conn = mock_graph.conn

        for i in range(30):
            _insert_node(conn, _make_node_dict(
                f"node-{i}", f"Fact {i}", importance=30 - i,
            ))
        conn.commit()

        result = builder.build_core_context(max_nodes=10)
        # Count how many "Fact N" appear — should be at most 10
        count = sum(1 for i in range(30) if f"Fact {i}" in result)
        assert count <= 10

    def test_without_hint_prioritizes_current_space(self, mock_graph):
        """Core nodes from the current space appear before other projects."""
        builder = ContextBuilder(mock_graph)
        conn = mock_graph.conn

        # Current-space node with low importance
        _insert_node(conn, _make_node_dict(
            "current-1", "Current Space Fact", space="myproject", importance=0.1,
        ))
        # Other-project node with high importance
        _insert_node(conn, _make_node_dict(
            "other-1", "Other Project Fact", space="otherproject", importance=0.9,
        ))
        conn.commit()

        # Cap to 1 — current-space node should win
        result = builder.build_core_context(space="myproject", max_nodes=1)
        assert "Current Space Fact" in result
        assert "Other Project Fact" not in result


class TestWithHint:
    """With task_hint, results should be filtered by relevance."""

    def test_hint_uses_hybrid_search(self, mock_graph):
        """When task_hint and engine are given, recall_search_structured should be used."""
        mock_engine = MagicMock()
        builder = ContextBuilder(mock_graph, engine=mock_engine)
        conn = mock_graph.conn

        nodes = []
        for i in range(10):
            node = _make_node_dict(f"node-{i}", f"Fact {i}", importance=0.5)
            nodes.append(node)
            _insert_node(conn, node)
        conn.commit()

        # Engine returns a subset as search results
        mock_engine.recall_search_structured.return_value = [
            {"node": nodes[0], "score": 0.9, "source": "hybrid"},
            {"node": nodes[1], "score": 0.8, "source": "hybrid"},
        ]

        result = builder.build_core_context(task_hint="python debugging", max_nodes=3)

        mock_engine.recall_search_structured.assert_called_once_with(
            query="python debugging",
            limit=3,
            default_space=None,
            tiers=["core", "working"],
            touch_access=False,
        )
        # Only filtered nodes appear
        assert "Fact 0" in result
        assert "Fact 1" in result

    def test_hint_falls_back_without_engine(self, mock_graph):
        """ContextBuilder without engine falls back to _filter_by_hint."""
        builder = ContextBuilder(mock_graph)  # No engine
        conn = mock_graph.conn

        for i in range(10):
            _insert_node(conn, _make_node_dict(f"node-{i}", f"Fact {i}", importance=0.5))
        conn.commit()

        subset = [_make_node_dict("node-0", "Fact 0"), _make_node_dict("node-1", "Fact 1")]
        with patch.object(builder, "_filter_by_hint", return_value=subset) as mock_filter:
            result = builder.build_core_context(task_hint="python debugging", max_nodes=3)
            mock_filter.assert_called_once()

    def test_hint_falls_back_on_search_failure(self, mock_graph):
        """When engine.recall_search_structured raises, result is capped (not full dump)."""
        mock_engine = MagicMock()
        mock_engine.recall_search_structured.side_effect = RuntimeError("search failed")
        builder = ContextBuilder(mock_graph, engine=mock_engine)
        conn = mock_graph.conn

        for i in range(30):
            _insert_node(conn, _make_node_dict(
                f"node-{i}", f"Fact {i}", importance=30 - i,
            ))
        conn.commit()

        result = builder.build_core_context(task_hint="python debugging", max_nodes=5)
        # Capped to max_nodes, not a full dump of all 30
        count = sum(1 for i in range(30) if f"Fact {i}" in result)
        assert count <= 5

    def test_hint_filter_returns_none_falls_back(self, mock_graph):
        """When _filter_by_hint returns None, result is capped (not full dump)."""
        builder = ContextBuilder(mock_graph)
        conn = mock_graph.conn

        for i in range(30):
            _insert_node(conn, _make_node_dict(
                f"node-{i}", f"Fact {i}", importance=30 - i,
            ))
        conn.commit()

        with patch.object(builder, "_filter_by_hint", return_value=None):
            result = builder.build_core_context(task_hint="python debugging", max_nodes=5)
            count = sum(1 for i in range(30) if f"Fact {i}" in result)
            assert count <= 5


class TestIdentityFiltering:
    """Identity nodes: person/preference always kept, others filtered by hint."""

    def test_person_and_preference_always_kept(self, mock_graph):
        """Person and preference identity nodes are always included."""
        mock_engine = MagicMock()
        builder = ContextBuilder(mock_graph, engine=mock_engine)
        conn = mock_graph.conn

        # Create a user node (person type)
        user_node = _make_node_dict("user-1", "Self", tier="core")
        user_node["type"] = "person"
        _insert_node(conn, user_node)

        # Create preference identity node
        pref = _make_node_dict("pref-1", "Prefers dark mode", tier="core")
        pref["type"] = "preference"
        _insert_node(conn, pref)

        # Create fact identity node (should be filtered)
        fact = _make_node_dict("fact-1", "Lives in London", tier="core")
        fact["type"] = "fact"
        _insert_node(conn, fact)

        # Link both via defines
        for nid in ("pref-1", "fact-1"):
            conn.execute(
                "INSERT INTO edges (source_id, target_id, edge_type, weight, created) "
                "VALUES (?, ?, 'defines', 1.0, '2026-01-01T00:00:00Z')",
                ("user-1", nid),
            )
        conn.commit()

        mock_engine.recall_search_structured.return_value = []

        # Mock _filter_by_hint to return empty (simulating low relevance)
        with patch.object(builder, "_filter_by_hint", return_value=[]):
            result = builder.build_core_context(
                user_node_id="user-1", task_hint="python debugging", max_nodes=1
            )
        # Person and preference survive
        assert "Prefers dark mode" in result
        # Fact filtered out (low relevance)
        assert "Lives in London" not in result

    def test_relevant_facts_kept_by_hint(self, mock_graph):
        """Identity facts that score well against the hint are kept."""
        mock_engine = MagicMock()
        builder = ContextBuilder(mock_graph, engine=mock_engine)
        conn = mock_graph.conn

        user_node = _make_node_dict("user-1", "Self", tier="core")
        user_node["type"] = "person"
        _insert_node(conn, user_node)

        # Relevant fact
        fact_relevant = _make_node_dict("fact-1", "Works at Acme on Python APIs", tier="core")
        _insert_node(conn, fact_relevant)

        # Irrelevant fact
        fact_irrelevant = _make_node_dict("fact-2", "Likes red grapes", tier="core")
        _insert_node(conn, fact_irrelevant)

        for nid in ("fact-1", "fact-2"):
            conn.execute(
                "INSERT INTO edges (source_id, target_id, edge_type, weight, created) "
                "VALUES (?, ?, 'defines', 1.0, '2026-01-01T00:00:00Z')",
                ("user-1", nid),
            )
        conn.commit()

        mock_engine.recall_search_structured.return_value = []

        # _filter_by_hint returns only the relevant fact
        with patch.object(builder, "_filter_by_hint", return_value=[fact_relevant]):
            result = builder.build_core_context(
                user_node_id="user-1", task_hint="python debugging", max_nodes=5
            )
        assert "Works at Acme" in result
        assert "red grapes" not in result

    def test_identity_filter_fallback_on_failure(self, mock_graph):
        """When _filter_by_hint returns None, all identity nodes are preserved."""
        mock_engine = MagicMock()
        builder = ContextBuilder(mock_graph, engine=mock_engine)
        conn = mock_graph.conn

        user_node = _make_node_dict("user-1", "Self", tier="core")
        user_node["type"] = "person"
        _insert_node(conn, user_node)

        fact = _make_node_dict("fact-1", "Home in Portland", tier="core")
        _insert_node(conn, fact)

        conn.execute(
            "INSERT INTO edges (source_id, target_id, edge_type, weight, created) "
            "VALUES (?, ?, 'defines', 1.0, '2026-01-01T00:00:00Z')",
            ("user-1", "fact-1"),
        )
        conn.commit()

        mock_engine.recall_search_structured.return_value = []

        # _filter_by_hint returns None (failure)
        with patch.object(builder, "_filter_by_hint", return_value=None):
            result = builder.build_core_context(
                user_node_id="user-1", task_hint="python debugging", max_nodes=5
            )
        # All identity nodes preserved on failure
        assert "Home in Portland" in result

    def test_identity_not_filtered_without_hint(self, mock_graph):
        """Without task_hint, all identity nodes are included."""
        builder = ContextBuilder(mock_graph)
        conn = mock_graph.conn

        user_node = _make_node_dict("user-1", "Self", tier="core")
        user_node["type"] = "person"
        _insert_node(conn, user_node)

        fact = _make_node_dict("fact-1", "Likes red grapes", tier="core")
        _insert_node(conn, fact)

        conn.execute(
            "INSERT INTO edges (source_id, target_id, edge_type, weight, created) "
            "VALUES (?, ?, 'defines', 1.0, '2026-01-01T00:00:00Z')",
            ("user-1", "fact-1"),
        )
        conn.commit()

        result = builder.build_core_context(user_node_id="user-1", max_nodes=20)
        assert "Likes red grapes" in result

    def test_identity_preserved_with_hybrid(self, mock_graph):
        """Person/preference identity nodes appear even when hybrid search returns unrelated results."""
        mock_engine = MagicMock()
        builder = ContextBuilder(mock_graph, engine=mock_engine)
        conn = mock_graph.conn

        # Create identity setup
        user_node = _make_node_dict("user-1", "Self", tier="core")
        user_node["type"] = "person"
        _insert_node(conn, user_node)

        identity = _make_node_dict("identity-1", "User Preference: Dark Mode", tier="core")
        identity["type"] = "preference"
        _insert_node(conn, identity)

        conn.execute(
            "INSERT INTO edges (source_id, target_id, edge_type, weight, created) "
            "VALUES (?, ?, 'defines', 1.0, '2026-01-01T00:00:00Z')",
            ("user-1", "identity-1"),
        )

        # Create some non-identity core nodes
        for i in range(3):
            _insert_node(conn, _make_node_dict(f"node-{i}", f"Fact {i}"))
        conn.commit()

        # Engine returns only node-0
        mock_engine.recall_search_structured.return_value = [
            {"node": _make_node_dict("node-0", "Fact 0"), "score": 0.9, "source": "hybrid"},
        ]

        result = builder.build_core_context(
            user_node_id="user-1", task_hint="python debugging", max_nodes=5
        )
        # Preference identity is always present
        assert "User Preference: Dark Mode" in result
        # Filtered node is present
        assert "Fact 0" in result
        # Non-matching nodes are excluded
        assert "Fact 1" not in result
        assert "Fact 2" not in result


class TestDeduplication:
    """Working/core nodes that duplicate identity entries should be removed."""

    def test_working_node_with_same_title_as_identity_excluded(self, mock_graph):
        """A working node with the same title as an identity node should not appear."""
        builder = ContextBuilder(mock_graph)
        conn = mock_graph.conn

        user_node = _make_node_dict("user-1", "Self", tier="core")
        user_node["type"] = "person"
        _insert_node(conn, user_node)

        # Identity preference (core, linked via defines)
        identity_pref = _make_node_dict("pref-core", "Communication style", tier="core")
        identity_pref["type"] = "preference"
        _insert_node(conn, identity_pref)

        conn.execute(
            "INSERT INTO edges (source_id, target_id, edge_type, weight, created) "
            "VALUES (?, ?, 'defines', 1.0, '2026-01-01T00:00:00Z')",
            ("user-1", "pref-core"),
        )

        # Duplicate working node with same title
        working_dupe = _make_node_dict(
            "pref-working", "Communication style", tier="working", space="myproj"
        )
        working_dupe["type"] = "preference"
        _insert_node(conn, working_dupe)
        conn.commit()

        result = builder.build_core_context(
            user_node_id="user-1", space="myproj", max_nodes=20
        )
        # Bullet marker for this title should appear exactly once
        assert result.count("]** Communication style") == 1

    def test_core_node_with_same_title_as_identity_excluded(self, mock_graph):
        """A non-identity core node with same title as identity node is excluded."""
        builder = ContextBuilder(mock_graph)
        conn = mock_graph.conn

        user_node = _make_node_dict("user-1", "Self", tier="core")
        user_node["type"] = "person"
        _insert_node(conn, user_node)

        # Identity node
        identity = _make_node_dict("id-1", "Likes Python", tier="core")
        identity["type"] = "preference"
        _insert_node(conn, identity)

        conn.execute(
            "INSERT INTO edges (source_id, target_id, edge_type, weight, created) "
            "VALUES (?, ?, 'defines', 1.0, '2026-01-01T00:00:00Z')",
            ("user-1", "id-1"),
        )

        # Non-identity core node with same title (different ID, no defines edge)
        dupe = _make_node_dict("dupe-1", "Likes Python", tier="core")
        dupe["type"] = "preference"
        _insert_node(conn, dupe)
        conn.commit()

        result = builder.build_core_context(user_node_id="user-1", max_nodes=20)
        # Bullet marker for this title should appear exactly once
        assert result.count("]** Likes Python") == 1

    def test_different_titles_not_affected(self, mock_graph):
        """Nodes with different titles from identity are not removed."""
        builder = ContextBuilder(mock_graph)
        conn = mock_graph.conn

        user_node = _make_node_dict("user-1", "Self", tier="core")
        user_node["type"] = "person"
        _insert_node(conn, user_node)

        identity = _make_node_dict("id-1", "User Name", tier="core")
        identity["type"] = "preference"
        _insert_node(conn, identity)

        conn.execute(
            "INSERT INTO edges (source_id, target_id, edge_type, weight, created) "
            "VALUES (?, ?, 'defines', 1.0, '2026-01-01T00:00:00Z')",
            ("user-1", "id-1"),
        )

        # Unrelated working node
        other = _make_node_dict("other-1", "API Architecture", tier="working", space="proj")
        _insert_node(conn, other)
        conn.commit()

        result = builder.build_core_context(
            user_node_id="user-1", space="proj", max_nodes=20
        )
        assert "User Name" in result
        assert "API Architecture" in result


class TestFilterByHint:
    """Unit tests for _filter_by_hint method (legacy fallback)."""

    def test_returns_none_on_import_error(self, mock_graph):
        builder = ContextBuilder(mock_graph)
        candidates = [_make_node_dict("n1", "Test")]

        with patch.dict("sys.modules", {"ormah.embeddings.encoder": None}):
            result = builder._filter_by_hint(candidates, "test hint")
            assert result is None

    def test_respects_max_nodes(self, mock_graph):
        mock_engine = MagicMock()
        builder = ContextBuilder(mock_graph, engine=mock_engine)
        candidates = [_make_node_dict(f"n{i}", f"Fact {i}") for i in range(10)]

        # Mock the encoder and vector store
        mock_enc = type("MockEncoder", (), {
            "encode": lambda self, x: np.ones(384, dtype=np.float32),
            "encode_query": lambda self, x: np.ones(384, dtype=np.float32),
        })()
        mock_vs = type("MockVS", (), {"get": lambda self, x: np.ones(384, dtype=np.float32)})()

        with patch("ormah.embeddings.encoder.get_encoder", return_value=mock_enc), \
             patch("ormah.embeddings.vector_store.VectorStore", return_value=mock_vs):
            result = builder._filter_by_hint(candidates, "test", max_nodes=3)
            assert result is not None
            assert len(result) == 3


class TestFallback:
    """When encoder fails, should fall back to capped context."""

    def test_fallback_on_encoder_failure(self, mock_graph):
        builder = ContextBuilder(mock_graph)
        conn = mock_graph.conn

        for i in range(30):
            _insert_node(conn, _make_node_dict(
                f"node-{i}", f"Fact {i}", importance=30 - i,
            ))
        conn.commit()

        # Patch _filter_by_hint to return None (simulating encoder failure)
        with patch.object(builder, "_filter_by_hint", return_value=None):
            result = builder.build_core_context(task_hint="python debugging", max_nodes=5)
            count = sum(1 for i in range(30) if f"Fact {i}" in result)
            assert count <= 5


class TestEmptyFilteredResults:
    """When search succeeds but no candidates match, core/working should be empty."""

    def test_hint_empty_results_returns_empty(self, mock_graph):
        """Search returns results but none in candidate set → empty core/working."""
        mock_engine = MagicMock()
        builder = ContextBuilder(mock_graph, engine=mock_engine)
        conn = mock_graph.conn

        # Create identity setup so we can verify identity still appears
        user_node = _make_node_dict("user-1", "Self", tier="core")
        user_node["type"] = "person"
        _insert_node(conn, user_node)

        pref = _make_node_dict("pref-1", "Prefers dark mode", tier="core")
        pref["type"] = "preference"
        _insert_node(conn, pref)

        conn.execute(
            "INSERT INTO edges (source_id, target_id, edge_type, weight, created) "
            "VALUES (?, ?, 'defines', 1.0, '2026-01-01T00:00:00Z')",
            ("user-1", "pref-1"),
        )

        # Create non-identity core nodes
        for i in range(5):
            _insert_node(conn, _make_node_dict(f"node-{i}", f"Fact {i}"))
        conn.commit()

        # Search returns results that are NOT in the candidate set
        mock_engine.recall_search_structured.return_value = [
            {"node": _make_node_dict("unrelated-1", "Unrelated"), "score": 0.9, "source": "hybrid"},
        ]

        result = builder.build_core_context(
            user_node_id="user-1", task_hint="fix login button CSS", max_nodes=20
        )

        # Identity still appears
        assert "Prefers dark mode" in result
        # Non-identity core nodes should NOT appear (empty filtered)
        for i in range(5):
            assert f"Fact {i}" not in result

    def test_hint_search_returns_empty_list(self, mock_graph):
        """Search returns empty list → empty core/working sections."""
        mock_engine = MagicMock()
        builder = ContextBuilder(mock_graph, engine=mock_engine)
        conn = mock_graph.conn

        for i in range(5):
            _insert_node(conn, _make_node_dict(f"node-{i}", f"Fact {i}"))
        conn.commit()

        mock_engine.recall_search_structured.return_value = []

        result = builder.build_core_context(task_hint="fix login button CSS", max_nodes=20)

        # No core nodes should appear
        for i in range(5):
            assert f"Fact {i}" not in result

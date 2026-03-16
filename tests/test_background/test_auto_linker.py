"""Tests for LLM-based edge type classification in auto_linker."""

from __future__ import annotations

import json
from unittest.mock import patch, MagicMock

from ormah.models.node import CreateNodeRequest, NodeType

_LLM_PATCH = "ormah.background.llm_client.llm_generate"


def _create_pair(engine, title_a="Python language", content_a="Python is a programming language.",
                 title_b="Python lang", content_b="Python is a popular programming language.",
                 node_type=NodeType.fact):
    """Helper: create two similar nodes without auto-linking, return their IDs."""
    # Suppress auto-link during creation so run_auto_linker controls edge creation
    original_threshold = engine.settings.auto_link_similarity_threshold
    engine.settings.auto_link_similarity_threshold = 999.0
    try:
        id_a, _ = engine.remember(
            CreateNodeRequest(content=content_a, type=node_type, title=title_a, tags=["test"]),
            agent_id="test",
        )
        id_b, _ = engine.remember(
            CreateNodeRequest(content=content_b, type=node_type, title=title_b, tags=["test"]),
            agent_id="test",
        )
    finally:
        engine.settings.auto_link_similarity_threshold = original_threshold
    return id_a, id_b


def _edges_between(engine, id_a, id_b):
    """Return all edges between two nodes."""
    return engine.db.conn.execute(
        "SELECT edge_type FROM edges WHERE "
        "(source_id = ? AND target_id = ?) OR (source_id = ? AND target_id = ?)",
        (id_a, id_b, id_b, id_a),
    ).fetchall()


def _reset_adapter():
    from ormah.background.llm_client import reset_adapter
    reset_adapter()


def test_llm_classifies_supports(engine):
    """LLM classifies as supports -> edge created with type supports."""
    id_a, id_b = _create_pair(engine)

    llm_response = json.dumps({
        "relationship": "supports",
        "reason": "Both describe Python as a programming language.",
    })

    engine.settings.llm_provider = "ollama"
    engine.settings.auto_link_similarity_threshold = 0.0
    _reset_adapter()

    with patch(_LLM_PATCH, return_value=llm_response):
        from ormah.background.auto_linker import run_auto_linker
        run_auto_linker(engine)

    edges = _edges_between(engine, id_a, id_b)
    assert len(edges) >= 1
    assert edges[0]["edge_type"] == "supports"


def test_llm_classifies_contradicts(engine):
    """LLM classifies as contradicts -> edge created with type contradicts."""
    id_a, id_b = _create_pair(
        engine,
        title_a="Python is fast",
        content_a="Python is the fastest programming language.",
        title_b="Python is slow",
        content_b="Python is one of the slowest programming languages.",
    )

    llm_response = json.dumps({
        "relationship": "contradicts",
        "reason": "They make opposing claims about Python speed.",
    })

    engine.settings.llm_provider = "ollama"
    engine.settings.auto_link_similarity_threshold = 0.0
    _reset_adapter()

    with patch(_LLM_PATCH, return_value=llm_response):
        from ormah.background.auto_linker import run_auto_linker
        run_auto_linker(engine)

    edges = _edges_between(engine, id_a, id_b)
    assert len(edges) >= 1
    assert edges[0]["edge_type"] == "contradicts"


def test_llm_classifies_none_no_edge(engine):
    """LLM classifies as none -> no edge created."""
    id_a, id_b = _create_pair(engine)

    llm_response = json.dumps({
        "relationship": "none",
        "reason": "Not meaningfully related.",
    })

    engine.settings.llm_provider = "ollama"
    engine.settings.auto_link_similarity_threshold = 0.0
    _reset_adapter()

    with patch(_LLM_PATCH, return_value=llm_response):
        from ormah.background.auto_linker import run_auto_linker
        run_auto_linker(engine)

    edges = _edges_between(engine, id_a, id_b)
    assert len(edges) == 0


def test_llm_unavailable_skips_edge(engine):
    """LLM returns None -> no edge created (no heuristic fallback)."""
    id_a, id_b = _create_pair(engine)

    engine.settings.llm_provider = "ollama"
    engine.settings.auto_link_similarity_threshold = 0.0
    _reset_adapter()

    with patch(_LLM_PATCH, return_value=None):
        from ormah.background.auto_linker import run_auto_linker
        run_auto_linker(engine)

    edges = _edges_between(engine, id_a, id_b)
    assert len(edges) == 0


def test_llm_disabled_skips_entirely(engine):
    """With llm_provider='none', LLM is never called and no edges are created."""
    id_a, id_b = _create_pair(engine)

    engine.settings.llm_provider = "none"
    engine.settings.auto_link_similarity_threshold = 0.0
    _reset_adapter()

    mock_llm = MagicMock()
    with patch(_LLM_PATCH, mock_llm):
        from ormah.background.auto_linker import run_auto_linker
        run_auto_linker(engine)

    mock_llm.assert_not_called()

    edges = _edges_between(engine, id_a, id_b)
    assert len(edges) == 0


def test_checked_pairs_not_rechecked(engine):
    """Pairs already checked should not trigger a second LLM call on re-run."""
    id_a, id_b = _create_pair(engine)

    llm_response = json.dumps({
        "relationship": "none",
        "reason": "Not meaningfully related.",
    })

    engine.settings.llm_provider = "ollama"
    engine.settings.auto_link_similarity_threshold = 0.0
    _reset_adapter()

    mock_llm = MagicMock(return_value=llm_response)
    with patch(_LLM_PATCH, mock_llm):
        from ormah.background.auto_linker import run_auto_linker
        run_auto_linker(engine)

    first_call_count = mock_llm.call_count
    assert first_call_count >= 1

    # Run again — the pair should be skipped
    mock_llm.reset_mock()
    with patch(_LLM_PATCH, mock_llm):
        run_auto_linker(engine)

    # LLM should not be called again for the same pair
    assert mock_llm.call_count == 0


def test_checked_pairs_recorded_for_none(engine):
    """Pairs classified as 'none' should be recorded in auto_link_checked."""
    id_a, id_b = _create_pair(engine)

    llm_response = json.dumps({
        "relationship": "none",
        "reason": "Not meaningfully related.",
    })

    engine.settings.llm_provider = "ollama"
    engine.settings.auto_link_similarity_threshold = 0.0
    _reset_adapter()

    with patch(_LLM_PATCH, return_value=llm_response):
        from ormah.background.auto_linker import run_auto_linker
        run_auto_linker(engine)

    pair = tuple(sorted([id_a, id_b]))
    row = engine.db.conn.execute(
        "SELECT result FROM auto_link_checked WHERE node_a = ? AND node_b = ?",
        pair,
    ).fetchone()
    assert row is not None
    assert row["result"] == "none"


def test_checked_pairs_invalidated_on_update(engine):
    """Updating a node's content should clear its checked pairs so it gets re-evaluated."""
    from ormah.models.node import UpdateNodeRequest

    id_a, id_b = _create_pair(engine)

    llm_response = json.dumps({
        "relationship": "none",
        "reason": "Not meaningfully related.",
    })

    engine.settings.llm_provider = "ollama"
    engine.settings.auto_link_similarity_threshold = 0.0
    _reset_adapter()

    with patch(_LLM_PATCH, return_value=llm_response):
        from ormah.background.auto_linker import run_auto_linker
        run_auto_linker(engine)

    pair = tuple(sorted([id_a, id_b]))
    row = engine.db.conn.execute(
        "SELECT 1 FROM auto_link_checked WHERE node_a = ? AND node_b = ?", pair
    ).fetchone()
    assert row is not None  # pair was recorded

    # Update node A's content
    engine.update_node(id_a, UpdateNodeRequest(content="Completely different content now"))

    # Checked pair should be cleared
    row = engine.db.conn.execute(
        "SELECT 1 FROM auto_link_checked WHERE node_a = ? AND node_b = ?", pair
    ).fetchone()
    assert row is None  # pair invalidated

    # Next run should re-evaluate the pair
    mock_llm = MagicMock(return_value=json.dumps({
        "relationship": "supports",
        "reason": "Now they are related.",
    }))
    with patch(_LLM_PATCH, mock_llm):
        run_auto_linker(engine)

    assert mock_llm.call_count >= 1  # LLM was called again for this pair

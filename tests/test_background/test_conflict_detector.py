"""Tests for LLM-based contradiction detection in conflict_detector."""

from __future__ import annotations

import json
from unittest.mock import patch, MagicMock

from ormah.models.node import CreateNodeRequest, NodeType

_LLM_PATCH = "ormah.background.llm_client.llm_generate"


def _create_pair(engine, title_a="Use PostgreSQL", content_a="We decided to use PostgreSQL for the database.",
                 title_b="Use MySQL", content_b="We decided to use MySQL for the database.",
                 node_type=NodeType.decision):
    """Helper: create two similar nodes without auto-linking, return their IDs."""
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


def _reset_adapter():
    from ormah.background.llm_client import reset_adapter
    reset_adapter()


def test_llm_detects_evolution_creates_evolved_from_edge(engine):
    """LLM detects belief evolution -> evolved_from edge created, no proposal."""
    id_a, id_b = _create_pair(
        engine,
        title_a="Dislikes grapes",
        content_a="The user hates grapes.",
        title_b="Loves red grapes",
        content_b="The user loves red grapes but hates green grapes.",
        node_type=NodeType.preference,
    )

    llm_response = json.dumps({
        "same_subject": True,
        "conflict": True,
        "type": "evolution",
        "evolved_node": "b",
        "explanation": "Refined from blanket dislike to nuanced preference by grape type.",
    })

    engine.settings.llm_provider = "ollama"
    _reset_adapter()

    with patch(_LLM_PATCH, return_value=llm_response):
        from ormah.background.conflict_detector import run_conflict_detection
        run_conflict_detection(engine)

    # Should create an evolved_from edge, not a proposal
    edges = engine.db.conn.execute(
        "SELECT * FROM edges WHERE edge_type = 'evolved_from'"
    ).fetchall()
    assert len(edges) >= 1
    edge = edges[0]
    assert edge["source_id"] == id_b  # newer (evolved) node
    assert edge["target_id"] == id_a  # older node
    assert "Refined from blanket dislike" in edge["reason"]

    # No proposals should be created
    proposals = engine.db.conn.execute(
        "SELECT * FROM proposals WHERE type = 'conflict' AND status = 'pending'"
    ).fetchall()
    assert len(proposals) == 0


def test_llm_detects_tension_creates_contradicts_edge(engine):
    """LLM detects genuine tension -> contradicts edge created with reason, no proposal."""
    id_a, id_b = _create_pair(engine, node_type=NodeType.fact)

    llm_response = json.dumps({
        "same_subject": True,
        "conflict": True,
        "type": "tension",
        "explanation": "Cannot use both PostgreSQL and MySQL as the primary database.",
    })

    engine.settings.llm_provider = "ollama"
    _reset_adapter()

    with patch(_LLM_PATCH, return_value=llm_response):
        from ormah.background.conflict_detector import run_conflict_detection
        run_conflict_detection(engine)

    # Should create a contradicts edge with reason
    edges = engine.db.conn.execute(
        "SELECT * FROM edges WHERE edge_type = 'contradicts'"
    ).fetchall()
    assert len(edges) >= 1
    edge = edges[0]
    assert "Cannot use both PostgreSQL and MySQL" in edge["reason"]

    # No proposals should be created
    proposals = engine.db.conn.execute(
        "SELECT * FROM proposals WHERE type = 'conflict' AND status = 'pending'"
    ).fetchall()
    assert len(proposals) == 0


def test_llm_rejects_contradiction_no_edge(engine):
    """LLM rejects contradiction -> no edge, no proposal."""
    id_a, id_b = _create_pair(
        engine,
        title_a="Use PostgreSQL",
        content_a="We decided to use PostgreSQL for the database.",
        title_b="PostgreSQL config",
        content_b="PostgreSQL should be configured with connection pooling.",
        node_type=NodeType.fact,
    )

    llm_response = json.dumps({
        "same_subject": True,
        "conflict": False,
        "type": "none",
        "explanation": "These are complementary — one is a decision, the other is a configuration detail.",
    })

    engine.settings.llm_provider = "ollama"
    _reset_adapter()

    with patch(_LLM_PATCH, return_value=llm_response):
        from ormah.background.conflict_detector import run_conflict_detection
        run_conflict_detection(engine)

    # No edges of conflict type
    edges = engine.db.conn.execute(
        "SELECT * FROM edges WHERE edge_type IN ('contradicts', 'evolved_from')"
    ).fetchall()
    assert len(edges) == 0

    # No proposals
    proposals = engine.db.conn.execute(
        "SELECT * FROM proposals WHERE type = 'conflict' AND status = 'pending'"
    ).fetchall()
    assert len(proposals) == 0


def test_llm_unavailable_skips_pair(engine):
    """LLM returns None -> pair is skipped, no proposals created."""
    id_a, id_b = _create_pair(engine, node_type=NodeType.fact)

    engine.settings.llm_provider = "ollama"
    _reset_adapter()

    with patch(_LLM_PATCH, return_value=None):
        from ormah.background.conflict_detector import run_conflict_detection
        run_conflict_detection(engine)

    # No proposals or edges should be created
    proposals = engine.db.conn.execute(
        "SELECT * FROM proposals WHERE type = 'conflict' AND status = 'pending'"
    ).fetchall()
    assert len(proposals) == 0

    edges = engine.db.conn.execute(
        "SELECT * FROM edges WHERE edge_type IN ('contradicts', 'evolved_from')"
    ).fetchall()
    assert len(edges) == 0


def test_llm_disabled_skips_detection(engine):
    """With llm_provider='none', LLM is never called and no proposals created."""
    id_a, id_b = _create_pair(engine)

    engine.settings.llm_provider = "none"
    _reset_adapter()

    mock_llm = MagicMock()
    with patch(_LLM_PATCH, mock_llm):
        from ormah.background.conflict_detector import run_conflict_detection
        run_conflict_detection(engine)

    mock_llm.assert_not_called()

    proposals = engine.db.conn.execute(
        "SELECT * FROM proposals WHERE type = 'conflict' AND status = 'pending'"
    ).fetchall()
    assert len(proposals) == 0


def test_project_scoped_nodes_checked_when_flag_enabled(engine):
    """With conflict_check_all_spaces=True, project-scoped nodes are checked."""
    original_threshold = engine.settings.auto_link_similarity_threshold
    engine.settings.auto_link_similarity_threshold = 999.0
    try:
        id_a, _ = engine.remember(
            CreateNodeRequest(
                content="We decided to use PostgreSQL for the database.",
                type=NodeType.fact,
                title="Use PostgreSQL",
                space="myproject",
            ),
            agent_id="test",
        )
        id_b, _ = engine.remember(
            CreateNodeRequest(
                content="We decided to use MySQL for the database.",
                type=NodeType.fact,
                title="Use MySQL",
                space="myproject",
            ),
            agent_id="test",
        )
    finally:
        engine.settings.auto_link_similarity_threshold = original_threshold

    llm_response = json.dumps({
        "same_subject": True,
        "conflict": True,
        "type": "tension",
        "explanation": "Cannot use both PostgreSQL and MySQL as the primary database.",
    })

    engine.settings.llm_provider = "ollama"
    engine.settings.conflict_check_all_spaces = True
    _reset_adapter()

    with patch(_LLM_PATCH, return_value=llm_response):
        from ormah.background.conflict_detector import run_conflict_detection
        run_conflict_detection(engine)

    edges = engine.db.conn.execute(
        "SELECT * FROM edges WHERE edge_type = 'contradicts'"
    ).fetchall()
    assert len(edges) >= 1


def test_project_scoped_nodes_skipped_by_default(engine):
    """By default (conflict_check_all_spaces=False), project-scoped nodes are not checked."""
    original_threshold = engine.settings.auto_link_similarity_threshold
    engine.settings.auto_link_similarity_threshold = 999.0
    try:
        engine.remember(
            CreateNodeRequest(
                content="We decided to use PostgreSQL for the database.",
                type=NodeType.fact,
                title="Use PostgreSQL",
                space="myproject",
            ),
            agent_id="test",
        )
        engine.remember(
            CreateNodeRequest(
                content="We decided to use MySQL for the database.",
                type=NodeType.fact,
                title="Use MySQL",
                space="myproject",
            ),
            agent_id="test",
        )
    finally:
        engine.settings.auto_link_similarity_threshold = original_threshold

    engine.settings.llm_provider = "ollama"
    engine.settings.conflict_check_all_spaces = False
    _reset_adapter()

    mock_llm = MagicMock()
    with patch(_LLM_PATCH, mock_llm):
        from ormah.background.conflict_detector import run_conflict_detection
        run_conflict_detection(engine)

    # LLM should never be called since project-scoped nodes are skipped
    mock_llm.assert_not_called()

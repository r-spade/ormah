"""Tests for LLM-based contradiction detection in conflict_detector."""

from __future__ import annotations

import json
import shutil
from datetime import datetime, timedelta, timezone
from unittest.mock import patch, MagicMock

import pytest

from ormah.background import conflict_detector
from ormah.models.node import CreateNodeRequest, NodeType

_LLM_PATCH = "ormah.background.llm_client.llm_generate"


def _rows():
    a = {"title": "Uses Postgres", "content": "API on Postgres.", "created": "2025-01-01", "space": "p"}
    b = {"title": "Uses SQLite", "content": "API on SQLite.", "created": "2025-06-01", "space": "p"}
    return a, b


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

    # ORDER BY RANDOM() means node_a/node_b ordering is non-deterministic.
    # Use side_effect to return evolved_node="b" when "Loves red grapes" is
    # presented as Memory B, and evolved_node="a" otherwise — so the
    # direction assertion is always semantically correct.
    def dynamic_llm_response(settings, prompt, json_mode=True, **kwargs):
        parts = prompt.split("\nMemory B")
        evolved = "b" if len(parts) > 1 and "Loves red grapes" in parts[1] else "a"
        return json.dumps({
            "same_subject": True,
            "conflict": True,
            "type": "evolution",
            "evolved_node": evolved,
            "explanation": "Refined from blanket dislike to nuanced preference by grape type.",
        })

    engine.settings.llm_provider = "ollama"
    _reset_adapter()

    with patch(_LLM_PATCH, side_effect=dynamic_llm_response):
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


# --- Strict schema + fail-closed (Step 1) ---


def test_conflict_passes_strict_schema(monkeypatch):
    captured = {}

    def spy(settings, prompt, **kwargs):
        captured.update(kwargs)
        return '{"same_subject": true, "conflict": true, "type": "evolution", "evolved_node": "b", "explanation": "x"}'

    monkeypatch.setattr("ormah.background.llm_client.llm_generate", spy)
    a, b = _rows()
    result = conflict_detector._llm_check_conflict(object(), a, b)
    assert captured["response_format"]["json_schema"]["schema"] is conflict_detector._CONFLICT_RESPONSE_SCHEMA
    assert result["conflict"] is True


def test_conflict_fail_closed(monkeypatch):
    monkeypatch.setattr("ormah.background.llm_client.llm_generate", lambda *a, **k: None)
    a, b = _rows()
    assert conflict_detector._llm_check_conflict(object(), a, b) is None


# --- Inconsistent-but-schema-valid combos -> _INCONSISTENT sentinel, not None (Step 5) ---


def test_conflict_inconsistent_no_conflict_but_type_not_none(monkeypatch):
    monkeypatch.setattr(
        "ormah.background.llm_client.llm_generate",
        lambda *a, **k: json.dumps(
            {"same_subject": True, "conflict": False, "type": "tension", "evolved_node": None, "explanation": "x"}
        ),
    )
    a, b = _rows()
    result = conflict_detector._llm_check_conflict(object(), a, b)
    assert result is conflict_detector._INCONSISTENT


def test_conflict_inconsistent_conflict_true_but_not_same_subject(monkeypatch):
    monkeypatch.setattr(
        "ormah.background.llm_client.llm_generate",
        lambda *a, **k: json.dumps(
            {"same_subject": False, "conflict": True, "type": "tension", "evolved_node": None, "explanation": "x"}
        ),
    )
    a, b = _rows()
    result = conflict_detector._llm_check_conflict(object(), a, b)
    assert result is conflict_detector._INCONSISTENT


def test_conflict_inconsistent_conflict_true_type_none(monkeypatch):
    monkeypatch.setattr(
        "ormah.background.llm_client.llm_generate",
        lambda *a, **k: json.dumps(
            {"same_subject": True, "conflict": True, "type": "none", "evolved_node": None, "explanation": "x"}
        ),
    )
    a, b = _rows()
    result = conflict_detector._llm_check_conflict(object(), a, b)
    assert result is conflict_detector._INCONSISTENT


def test_conflict_inconsistent_evolution_missing_evolved_node(monkeypatch):
    monkeypatch.setattr(
        "ormah.background.llm_client.llm_generate",
        lambda *a, **k: json.dumps(
            {"same_subject": True, "conflict": True, "type": "evolution", "evolved_node": None, "explanation": "x"}
        ),
    )
    a, b = _rows()
    result = conflict_detector._llm_check_conflict(object(), a, b)
    assert result is conflict_detector._INCONSISTENT


def test_conflict_tension_forces_evolved_node_none(monkeypatch):
    """tension with a stray evolved_node is NOT inconsistent — it's normalized to None."""
    monkeypatch.setattr(
        "ormah.background.llm_client.llm_generate",
        lambda *a, **k: json.dumps(
            {"same_subject": True, "conflict": True, "type": "tension", "evolved_node": "a", "explanation": "x"}
        ),
    )
    a, b = _rows()
    result = conflict_detector._llm_check_conflict(object(), a, b)
    assert result is not None and result is not conflict_detector._INCONSISTENT
    assert result["evolved_node"] is None


def test_run_conflict_records_none_for_inconsistent_answer_not_error(engine):
    """An inconsistent-but-schema-valid answer must record 'none' (terminal, no backoff churn),
    never 'error' (which would re-LLM every run once the 6h window passes)."""
    id_a, id_b = _create_pair(engine, node_type=NodeType.fact)

    llm_response = json.dumps({
        "same_subject": True, "conflict": False, "type": "tension",
        "evolved_node": None, "explanation": "inconsistent",
    })
    engine.settings.llm_provider = "ollama"
    _reset_adapter()

    with patch(_LLM_PATCH, return_value=llm_response):
        from ormah.background.conflict_detector import run_conflict_detection
        run_conflict_detection(engine)

    rows = engine.db.conn.execute("SELECT result FROM conflict_checked").fetchall()
    assert rows and all(r[0] == "none" for r in rows)


# --- Every terminal path writes a row (Step 8) ---


def test_conflict_records_none_on_all_terminal_paths(engine):
    """Both same_subject=false and no-conflict answers write result='none' (not silently dropped)."""
    id_a, id_b = _create_pair(engine, node_type=NodeType.fact)

    llm_response = json.dumps({
        "same_subject": True, "conflict": False, "type": "none",
        "evolved_node": None, "explanation": "not related",
    })
    engine.settings.llm_provider = "ollama"
    _reset_adapter()

    with patch(_LLM_PATCH, return_value=llm_response):
        from ormah.background.conflict_detector import run_conflict_detection
        run_conflict_detection(engine)

    rows = engine.db.conn.execute("SELECT result FROM conflict_checked").fetchall()
    assert rows and all(r[0] == "none" for r in rows)


# --- Per-run cap + circuit breaker (Step 7/10) ---


def _make_engine_with_many_similar_nodes(tmp_path, n):
    from ormah.config import Settings
    from ormah.engine.memory_engine import MemoryEngine

    nodes_dir = tmp_path / "nodes"
    nodes_dir.mkdir()
    settings = Settings(memory_dir=tmp_path)
    eng = MemoryEngine(settings)
    eng.startup()
    eng.settings.llm_provider = "ollama"  # llm_enabled derives from this
    for i in range(n):
        eng.remember(
            CreateNodeRequest(
                content=f"We decided to use database variant {i} for the API.",
                type=NodeType.fact,
                title=f"Database choice {i}",
                tags=["test"],
            ),
            agent_id="test",
        )
    return eng


def test_run_conflict_stops_at_cap(monkeypatch, tmp_path):
    calls = {"n": 0}
    monkeypatch.setattr(
        conflict_detector, "_llm_check_conflict",
        lambda s, a, b: (calls.__setitem__("n", calls["n"] + 1) or {
            "same_subject": True, "conflict": False, "type": "none",
            "evolved_node": None, "explanation": "x",
        }),
    )
    eng = _make_engine_with_many_similar_nodes(tmp_path, n=10)
    try:
        eng.settings.conflict_check_max_llm_calls_per_run = 3
        conflict_detector.run_conflict_detection(eng)
        assert calls["n"] == 3
    finally:
        eng.shutdown()


def test_conflict_circuit_breaker(monkeypatch, tmp_path):
    calls = {"n": 0}

    def _fail(s, a, b):
        calls["n"] += 1
        return None

    monkeypatch.setattr(conflict_detector, "_llm_check_conflict", _fail)
    eng = _make_engine_with_many_similar_nodes(tmp_path, n=20)
    try:
        object.__setattr__(eng.settings, "conflict_check_max_llm_calls_per_run", 100)
        conflict_detector.run_conflict_detection(eng)
        assert calls["n"] <= 3
        errs = eng.db.conn.execute(
            "SELECT count(*) FROM conflict_checked WHERE result='error'"
        ).fetchone()[0]
        assert errs >= 1
    finally:
        eng.shutdown()


# --- Error-row backoff (Step 6/10) ---


def test_conflict_error_row_backoff(engine):
    """An 'error' row inside the backoff window hides the pair; once the window has
    elapsed the pair is checked again."""
    id_a, id_b = _create_pair(engine, node_type=NodeType.fact)
    pair = tuple(sorted([id_a, id_b]))

    # Fresh error row -> within backoff window -> candidate skipped.
    with engine.db.transaction() as conn:
        conn.execute(
            "INSERT OR REPLACE INTO conflict_checked (node_a, node_b, result, checked_at) "
            "VALUES (?, ?, 'error', ?)",
            (*pair, datetime.now(timezone.utc).isoformat()),
        )
    candidates = conflict_detector._find_conflict_candidates(engine, limit=10)
    assert candidates == []

    # Stale error row (past the 6h backoff window) -> candidate re-appears.
    stale = (datetime.now(timezone.utc) - timedelta(hours=7)).isoformat()
    with engine.db.transaction() as conn:
        conn.execute(
            "UPDATE conflict_checked SET checked_at = ? WHERE node_a = ? AND node_b = ?",
            (stale, *pair),
        )
    candidates = conflict_detector._find_conflict_candidates(engine, limit=10)
    found_pairs = {tuple(sorted([c["node_a"]["id"], c["node_b"]["id"]])) for c in candidates}
    assert pair in found_pairs


# --- Ordered-pair skip (Step 6) ---


def test_conflict_ordered_pair_skip(engine):
    """A terminal row written as (a, b) also skips the reversed candidate (b, a)."""
    id_a, id_b = _create_pair(engine, node_type=NodeType.fact)
    pair = tuple(sorted([id_a, id_b]))

    with engine.db.transaction() as conn:
        conn.execute(
            "INSERT OR REPLACE INTO conflict_checked (node_a, node_b, result, checked_at) "
            "VALUES (?, ?, 'none', ?)",
            (*pair, datetime.now(timezone.utc).isoformat()),
        )

    candidates = conflict_detector._find_conflict_candidates(engine, limit=10)
    found_pairs = {tuple(sorted([c["node_a"]["id"], c["node_b"]["id"]])) for c in candidates}
    assert pair not in found_pairs


# --- Real claude_cli integration (Step 10) ---


@pytest.mark.integration
@pytest.mark.skipif(shutil.which("claude") is None, reason="claude CLI not installed")
def test_real_claude_cli_conflict_check_returns_valid_type(engine):
    """End-to-end: --json-schema -> structured_output round-trips for the conflict prompt."""
    engine.settings.llm_provider = "claude_cli"
    _reset_adapter()

    a, b = _rows()
    result = conflict_detector._llm_check_conflict(engine.settings, a, b)

    assert result is not None
    assert result is not conflict_detector._INCONSISTENT
    assert result["type"] in ("evolution", "tension", "none")

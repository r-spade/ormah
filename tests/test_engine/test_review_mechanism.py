"""Regression coverage for retired automatic retrospective review notes."""

from __future__ import annotations

from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from ormah.engine.context_builder import ContextBuilder
from ormah.engine.maintenance_signal import MAINTENANCE_DUE_SIGNAL
from ormah.index.db import Database
from ormah.index.graph import GraphIndex


_REVIEW_HEADING = "one thing to review when you get a chance"


def _make_node_dict(node_id, title, tier="core", space=None, importance=0.5, **kwargs):
    return {
        "id": node_id,
        "type": kwargs.get("type", "fact"),
        "tier": tier,
        "title": title,
        "content": kwargs.get("content", f"Content about {title}"),
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
        (
            node["id"],
            node["type"],
            node["tier"],
            node["source"],
            node["space"],
            node["title"],
            node["content"],
            node["created"],
            node["updated"],
            node["last_accessed"],
            node["access_count"],
            node["confidence"],
            node["importance"],
            "/fake/path",
            "abc123",
        ),
    )


@pytest.fixture
def mock_graph(tmp_path):
    db = Database(tmp_path / "index.db")
    db.init_schema()
    return GraphIndex(db.conn)


def _make_mock_engine(conn, *, maintenance_enabled=False):
    engine = MagicMock()
    engine.settings = SimpleNamespace(
        claude_maintenance_enabled=maintenance_enabled,
        claude_maintenance_interval_hours=24,
    )
    engine._get_hybrid_search.return_value = None
    engine.recall_search_structured.return_value = []
    engine.has_searchable_preferences.return_value = False

    @contextmanager
    def transaction():
        yield conn

    engine.db.transaction = transaction
    return engine


def _admitted_ordinary_result(node_id="node-selected"):
    node = _make_node_dict(
        node_id,
        "Current authentication design",
        space="myspace",
        content="Authentication uses scoped tokens for the current application.",
    )
    return {"node": node, "score": 0.80, "source": "hybrid"}


def _seed_historical_review_data(conn, node_id="node-held-back"):
    """Seed prior withheld events and review history without producing a review."""
    held_back = _make_node_dict(
        node_id,
        "Historical held-back memory",
        space="myspace",
        content="This historical content must never be appended to a new whisper.",
    )
    _insert_node(conn, held_back)
    conn.execute(
        "INSERT INTO whisper_log "
        "(node_id, score, session_id, space, prompt_text, prompt_hash, prompt_vec, "
        "decision_stage, was_injected, logged_at) "
        "VALUES (?, 0.48, 'old-session', 'myspace', 'old task prompt', 'old-hash', X'', "
        "'injection_gate', 0, datetime('now', '-1 day'))",
        (node_id,),
    )
    conn.execute(
        "INSERT INTO review_log (node_id, session_id, surfaced_at, answered) "
        "VALUES (?, 'historic-review', datetime('now', '-20 days'), 0)",
        (node_id,),
    )
    conn.execute(
        "INSERT INTO review_log (node_id, session_id, surfaced_at, answered) "
        "VALUES (?, 'answered-review', datetime('now', '-30 days'), 1)",
        (node_id,),
    )
    conn.commit()
    return held_back


def _review_rows(conn):
    return [
        tuple(row)
        for row in conn.execute(
            "SELECT node_id, session_id, surfaced_at, answered FROM review_log ORDER BY id"
        ).fetchall()
    ]


@pytest.mark.parametrize(
    ("session_id", "recent_prompts"),
    [
        pytest.param(None, None, id="sessionless"),
        pytest.param("first-session", None, id="first-session-turn"),
        pytest.param("after-gap", None, id="post-gap-turn"),
        pytest.param("ongoing-session", ["earlier task"], id="ongoing-session"),
    ],
)
def test_selected_context_never_appends_historical_review_or_writes_review_log(
    mock_graph, session_id, recent_prompts
):
    """Current-task context remains useful without a retrospective assignment.

    This is the removal regression: at PR head 98ee602, first/sessionless/
    post-gap shapes append the held-back title and add a review_log row even
    though the normal selected memory is already present.
    """
    conn = mock_graph.conn
    held_back = _seed_historical_review_data(conn)
    before = _review_rows(conn)
    engine = _make_mock_engine(conn)
    engine.recall_search_structured.return_value = [_admitted_ordinary_result()]

    result = ContextBuilder(mock_graph, engine=engine).build_whisper_context(
        prompt="how does the current authentication design work",
        space="myspace",
        recent_prompts=recent_prompts,
        session_id=session_id,
    )

    assert "Current authentication design" in result
    assert _REVIEW_HEADING not in result
    assert held_back["title"] not in result
    assert held_back["content"] not in result
    assert _review_rows(conn) == before


def test_preference_only_context_never_appends_historical_review_or_writes_review_log(mock_graph):
    """Applicable preferences remain injectable without bringing back reviews."""
    conn = mock_graph.conn
    held_back = _seed_historical_review_data(conn)
    before = _review_rows(conn)
    preference = _make_node_dict(
        "node-preference",
        "Prefer concise architecture",
        space="myspace",
        type="preference",
        content="Keep architecture decisions concise and written down.",
    )
    engine = _make_mock_engine(conn)
    engine.has_searchable_preferences.return_value = True
    engine.recall_search_structured.side_effect = [
        [],
        [{"node": preference, "score": 0.65, "source": "hybrid"}],
    ]
    cross_encoder = MagicMock()
    cross_encoder.rerank.return_value = [3.0]

    with patch("ormah.embeddings.reranker._get_model", return_value=cross_encoder):
        result = ContextBuilder(mock_graph, engine=engine).build_whisper_context(
            prompt="plan this architecture change",
            space="myspace",
            recent_prompts=None,
            session_id="preference-only",
            reranker_enabled=True,
            preference_applicability_enabled=True,
            preference_applicability_gate=0.40,
        )

    assert "Prefer concise architecture" in result
    assert _REVIEW_HEADING not in result
    assert held_back["title"] not in result
    assert _review_rows(conn) == before


@pytest.mark.parametrize(
    ("prompt", "session_id", "recent_prompts", "kwargs", "expected"),
    [
        pytest.param(
            "an unrelated question with enough words",
            None,
            None,
            {},
            "",
            id="sessionless-no-candidates",
        ),
        pytest.param(
            "an unrelated question with enough words",
            "first-session",
            None,
            {},
            "",
            id="first-session-no-candidates",
        ),
        pytest.param(
            "an unrelated question with enough words",
            "after-gap",
            None,
            {},
            "",
            id="post-gap-no-candidates",
        ),
        pytest.param(
            "an unrelated question with enough words",
            "ongoing-session",
            ["earlier task"],
            {},
            "",
            id="ongoing-no-candidates",
        ),
        pytest.param(
            "ok",
            "short-prompt",
            None,
            {},
            "",
            id="short-prompt",
        ),
    ],
)
def test_silent_paths_preserve_historical_review_records(
    mock_graph, prompt, session_id, recent_prompts, kwargs, expected
):
    """No candidate path can convert old review history into current context."""
    conn = mock_graph.conn
    _seed_historical_review_data(conn)
    before = _review_rows(conn)
    engine = _make_mock_engine(conn)

    result = ContextBuilder(mock_graph, engine=engine).build_whisper_context(
        prompt=prompt,
        space="myspace",
        recent_prompts=recent_prompts,
        session_id=session_id,
        **kwargs,
    )

    assert result == expected
    assert _review_rows(conn) == before


def test_gate_rejection_stays_silent_and_preserves_historical_review_records(mock_graph):
    conn = mock_graph.conn
    _seed_historical_review_data(conn)
    before = _review_rows(conn)
    engine = _make_mock_engine(conn)
    engine.recall_search_structured.return_value = [_admitted_ordinary_result()]

    result = ContextBuilder(mock_graph, engine=engine).build_whisper_context(
        prompt="how does the current authentication design work",
        space="myspace",
        recent_prompts=None,
        session_id="gate-rejected",
        injection_gate=0.90,
    )

    assert result == ""
    assert _review_rows(conn) == before
    outcome = conn.execute(
        "SELECT outcome, injected_count FROM whisper_decisions ORDER BY id DESC LIMIT 1"
    ).fetchone()
    assert tuple(outcome) == ("silent_gate", 0)


def test_maintenance_signal_remains_bare_when_no_context_is_selected(mock_graph):
    conn = mock_graph.conn
    _seed_historical_review_data(conn)
    before = _review_rows(conn)
    engine = _make_mock_engine(conn, maintenance_enabled=True)

    result = ContextBuilder(mock_graph, engine=engine).build_whisper_context(
        prompt="an unrelated question with enough words",
        space="myspace",
        recent_prompts=None,
        session_id="maintenance-only",
    )

    assert result == MAINTENANCE_DUE_SIGNAL
    assert _review_rows(conn) == before


def test_selected_context_and_maintenance_signal_remain_intact_without_review(mock_graph):
    conn = mock_graph.conn
    held_back = _seed_historical_review_data(conn)
    before = _review_rows(conn)
    engine = _make_mock_engine(conn, maintenance_enabled=True)
    engine.recall_search_structured.return_value = [_admitted_ordinary_result()]

    result = ContextBuilder(mock_graph, engine=engine).build_whisper_context(
        prompt="how does the current authentication design work",
        space="myspace",
        recent_prompts=None,
        session_id="selected-maintenance",
    )

    assert "Current authentication design" in result
    assert result.endswith(MAINTENANCE_DUE_SIGNAL)
    assert _REVIEW_HEADING not in result
    assert held_back["title"] not in result
    assert _review_rows(conn) == before

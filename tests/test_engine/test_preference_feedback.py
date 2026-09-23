"""Regression tests for contextual feedback in preference applicability."""

from __future__ import annotations

from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from ormah.engine.context_builder import ContextBuilder
from ormah.engine.prompt_classifier import PromptIntent
from ormah.index.db import Database
from ormah.index.graph import GraphIndex


def _insert_preference(engine, node_id: str) -> dict:
    node = {
        "id": node_id,
        "type": "preference",
        "tier": "working",
        "source": "agent:test",
        "space": "nova",
        "title": "Authentication release announcements",
        "content": "Write authentication release announcements in a playful tone.",
    }
    engine.db.conn.execute(
        "INSERT INTO nodes (id, type, tier, source, space, title, content, "
        "created, updated, last_accessed, access_count, confidence, importance, "
        "file_path, file_hash) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (
            node["id"],
            node["type"],
            node["tier"],
            node["source"],
            node["space"],
            node["title"],
            node["content"],
            "2026-01-01T00:00:00Z",
            "2026-01-01T00:00:00Z",
            "2026-01-01T00:00:00Z",
            0,
            1.0,
            0.5,
            "/tmp/synthetic-preference.md",
            "synthetic-preference",
        ),
    )
    engine.db.conn.commit()
    return node


def _lifecycle_snapshot(engine, node_id: str) -> tuple:
    row = engine.db.conn.execute(
        "SELECT tier, valid_until, access_count, last_accessed, updated, "
        "confidence, importance FROM nodes WHERE id = ?",
        (node_id,),
    ).fetchone()
    return tuple(row)


@pytest.fixture
def mock_graph(tmp_path):
    db = Database(tmp_path / "index.db")
    db.init_schema()
    graph = GraphIndex(db.conn)
    yield graph
    db.close()


def _unit_node(node_id: str, title: str, node_type: str = "preference") -> dict:
    return {
        "id": node_id,
        "type": node_type,
        "tier": "working",
        "space": "nova",
        "title": title,
        "content": f"Content about {title}",
        "confidence": 1.0,
        "importance": 0.5,
    }


def _unit_settings() -> SimpleNamespace:
    return SimpleNamespace(
        affinity_similarity_threshold=0.70,
        affinity_half_life_days=30.0,
        affinity_max_boost=0.15,
        affinity_implicit_weight=0.8,
        whisper_exploration_enabled=False,
        claude_maintenance_enabled=False,
    )


def _unit_builder(mock_graph, main_results, preference_results, prompt_vec=None):
    if prompt_vec is None:
        prompt_vec = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    engine = MagicMock()
    engine.settings = _unit_settings()
    engine.has_searchable_preferences.return_value = True
    engine.recall_search_structured.side_effect = [main_results, preference_results]
    engine.db = MagicMock()
    engine.db.conn = mock_graph.conn
    encoder = MagicMock()
    encoder.encode.return_value = prompt_vec
    encoder.encode_query.return_value = prompt_vec
    hybrid = SimpleNamespace(encoder=encoder)
    engine._get_hybrid_search.return_value = hybrid
    builder = ContextBuilder(mock_graph, engine=engine)
    builder._classifier = SimpleNamespace(
        classify=lambda _: PromptIntent(categories=["general"], prompt_vec=prompt_vec),
    )
    return builder, engine, encoder


def _fake_rerank(score_by_id: dict[str, float], queries: list[str]):
    def rerank(query, candidates, **_kwargs):
        queries.append(query)
        return [
            {
                **candidate,
                "score": score_by_id[candidate["node"]["id"]],
                "cross_encoder_score": -4.08,
                "ce_absolute": score_by_id[candidate["node"]["id"]],
            }
            for candidate in sorted(
                candidates,
                key=lambda candidate: score_by_id[candidate["node"]["id"]],
                reverse=True,
            )
        ]

    return rerank


def _affinity_row(prompt_vec: np.ndarray, signal: int) -> dict:
    return {
        "prompt_vec": prompt_vec.astype(np.float32).tobytes(),
        "signal": signal,
        "source": "implicit",
        "confirmed_at": datetime.now(timezone.utc).isoformat(),
    }


def test_real_negative_feedback_suppresses_preference_applicability(engine):
    """An injected preference's exact negative event affects the second pass."""
    node_id = "303-preference-negative-000000000000000000000001"
    node = _insert_preference(engine, node_id)
    prompt = "Review authentication token validation."
    prompt_vec = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    engine.context_builder._classifier = SimpleNamespace(
        classify=lambda _: PromptIntent(categories=["general"], prompt_vec=prompt_vec),
    )
    hybrid = MagicMock()
    hybrid.encoder.encode_query.return_value = prompt_vec
    main_result = {"node": node, "score": 0.8, "raw_cosine": 0.8, "source": "hybrid"}
    preference_result = {
        "node": node,
        "score": 0.8,
        "raw_cosine": 0.8,
        "source": "hybrid",
    }
    engine.recall_search_structured = MagicMock(
        side_effect=[
            [main_result],
            [preference_result],
            [main_result],
            [preference_result],
        ]
    )
    model = SimpleNamespace(rerank=lambda _query, docs: [-4.08] * len(docs))
    lifecycle_before = _lifecycle_snapshot(engine, node_id)

    def whisper():
        return engine.context_builder.build_whisper_context(
            prompt=prompt,
            space="nova",
            max_nodes=8,
            min_score=0.0,
            candidate_pool_multiplier=5,
            reranker_enabled=True,
            reranker_model="synthetic-reranker",
            reranker_min_score=0.0,
            reranker_blend_alpha=1.0,
            injection_gate=0.45,
            preference_applicability_enabled=True,
            preference_applicability_gate=0.40,
            preference_max_nodes=2,
            session_id="303-negative-feedback",
            _return_debug=True,
        )

    with patch.object(engine, "_get_hybrid_search", return_value=hybrid), \
         patch("ormah.embeddings.reranker._get_model", return_value=model):
        before_text, before_ids = whisper()
        first_log = engine.db.conn.execute(
            "SELECT id, score, gate_score, source, decision_stage, was_injected "
            "FROM whisper_log WHERE node_id = ? ORDER BY id DESC LIMIT 1",
            (node_id,),
        ).fetchone()
        assert before_ids == [node_id]
        assert "playful tone" in before_text
        assert first_log["source"] == "preference_applicability"
        assert first_log["decision_stage"] == "injected"
        assert first_log["was_injected"] == 1
        assert first_log["score"] == pytest.approx(0.44, abs=1e-6)
        assert first_log["gate_score"] == pytest.approx(0.44, abs=1e-6)

        engine.submit_feedback(
            node_id,
            signal=-1,
            source="implicit",
            whisper_log_id=first_log["id"],
        )
        affinity = engine.db.conn.execute(
            "SELECT signal, source, whisper_log_id FROM affinity WHERE node_id = ?",
            (node_id,),
        ).fetchone()
        assert tuple(affinity) == (-1, "implicit", first_log["id"])
        assert _lifecycle_snapshot(engine, node_id) == lifecycle_before

        after_text, after_ids = whisper()
        second_log = engine.db.conn.execute(
            "SELECT score, gate_score, source, decision_stage, was_injected "
            "FROM whisper_log WHERE node_id = ? ORDER BY id DESC LIMIT 1",
            (node_id,),
        ).fetchone()

    assert after_text == ""
    assert after_ids == []
    assert second_log["source"] == "preference_applicability"
    assert second_log["decision_stage"] == "preference_applicability_gate"
    assert second_log["was_injected"] == 0
    assert second_log["score"] == pytest.approx(0.29, abs=1e-6)
    assert second_log["gate_score"] == pytest.approx(0.29, abs=1e-6)
    assert _lifecycle_snapshot(engine, node_id) == lifecycle_before


@pytest.mark.parametrize(
    ("rows", "base_score", "expected_title"),
    [
        ({}, 0.45, "No feedback preference"),
        (
            {
                "pref-neutral": [
                    _affinity_row(np.array([-1.0, 0.0, 0.0], dtype=np.float32), -1),
                ],
            },
            0.45,
            "Nonmatching feedback preference",
        ),
    ],
)
def test_preference_without_matching_feedback_keeps_applicability(
    mock_graph, rows, base_score, expected_title
):
    node_id = next(iter(rows), "pref-no-feedback")
    node = _unit_node(node_id, expected_title)
    builder, _engine, _encoder = _unit_builder(mock_graph, [], [{"node": node, "score": 0.8}])
    queries = []
    fake_rerank = _fake_rerank({node_id: base_score}, queries)
    batch = MagicMock(return_value=rows)

    with patch("ormah.embeddings.reranker.rerank", side_effect=fake_rerank), \
         patch("ormah.engine.affinity.batch_fetch_affinity", batch):
        result = builder.build_whisper_context(
            prompt="apply a preference",
            min_score=0.0,
            reranker_enabled=True,
            reranker_min_score=0.0,
            reranker_blend_alpha=1.0,
            injection_gate=0.55,
            preference_applicability_enabled=True,
            preference_applicability_gate=0.40,
            preference_max_nodes=1,
        )

    assert expected_title in result
    batch.assert_called_once_with(mock_graph.conn, [node_id])
    assert queries == ["Relevant user preference for this action: apply a preference"]


def test_matching_positive_feedback_rescues_preference_below_gate(mock_graph):
    node_id = "pref-positive-rescue"
    node = _unit_node(node_id, "Positive feedback preference")
    builder, _engine, _encoder = _unit_builder(mock_graph, [], [{"node": node, "score": 0.8}])
    positive_rows = {
        node_id: [_affinity_row(np.array([1.0, 0.0, 0.0], dtype=np.float32), 1)],
    }
    fake_rerank = _fake_rerank({node_id: 0.30}, [])
    batch = MagicMock(return_value=positive_rows)

    with patch("ormah.embeddings.reranker.rerank", side_effect=fake_rerank), \
         patch("ormah.engine.affinity.batch_fetch_affinity", batch):
        result = builder.build_whisper_context(
            prompt="apply a preference",
            min_score=0.0,
            reranker_enabled=True,
            reranker_min_score=0.0,
            reranker_blend_alpha=1.0,
            injection_gate=0.55,
            preference_applicability_enabled=True,
            preference_applicability_gate=0.40,
            preference_max_nodes=1,
        )

    assert "Positive feedback preference" in result
    batch.assert_called_once_with(mock_graph.conn, [node_id])


def test_already_admitted_preference_is_not_duplicated_or_refetched(mock_graph):
    node_id = "pref-already-admitted"
    node = _unit_node(node_id, "Already admitted preference")
    candidate = {"node": node, "score": 0.8, "raw_cosine": 0.8, "source": "hybrid"}
    builder, engine, _encoder = _unit_builder(mock_graph, [candidate], [candidate])
    fake_rerank = _fake_rerank({node_id: 0.80}, [])
    batch = MagicMock(return_value={})

    with patch("ormah.embeddings.reranker.rerank", side_effect=fake_rerank), \
         patch("ormah.engine.affinity.batch_fetch_affinity", batch):
        result, ids = builder.build_whisper_context(
            prompt="already admitted preference",
            min_score=0.0,
            reranker_enabled=True,
            reranker_min_score=0.0,
            reranker_blend_alpha=1.0,
            injection_gate=0.45,
            preference_applicability_enabled=True,
            preference_applicability_gate=0.40,
            preference_max_nodes=1,
            _return_debug=True,
        )

    assert ids == [node_id]
    assert result.count("id: pref-alr") == 1
    assert engine.recall_search_structured.call_count == 2
    batch.assert_called_once_with(mock_graph.conn, [node_id])


def test_preference_candidate_seen_in_both_paths_is_adjusted_once(mock_graph):
    node_id = "pref-shared-candidate"
    node = _unit_node(node_id, "Shared preference")
    candidate = {"node": node, "score": 0.8, "raw_cosine": 0.8, "source": "hybrid"}
    builder, _engine, _encoder = _unit_builder(mock_graph, [candidate], [candidate])
    negative_rows = {
        node_id: [_affinity_row(np.array([1.0, 0.0, 0.0], dtype=np.float32), -1)],
    }
    batch_calls = []
    compute_calls = []

    def fetch(_conn, node_ids):
        batch_calls.append(list(node_ids))
        return {node_id: negative_rows[node_id]}

    from ormah.engine.affinity import compute_affinity_boost as real_compute

    def compute(current_vec, candidate_id, affinity_rows, settings):
        compute_calls.append((current_vec.copy(), candidate_id))
        return real_compute(current_vec, candidate_id, affinity_rows, settings)

    fake_rerank = _fake_rerank({node_id: 0.44}, [])
    with patch("ormah.embeddings.reranker.rerank", side_effect=fake_rerank), \
         patch("ormah.engine.affinity.batch_fetch_affinity", side_effect=fetch), \
         patch("ormah.engine.affinity.compute_affinity_boost", side_effect=compute):
        result, ids = builder.build_whisper_context(
            prompt="apply a preference",
            min_score=0.0,
            reranker_enabled=True,
            reranker_min_score=0.0,
            reranker_blend_alpha=1.0,
            injection_gate=0.45,
            preference_applicability_enabled=True,
            preference_applicability_gate=0.40,
            preference_max_nodes=1,
            _return_debug=True,
        )

    assert result == ""
    assert ids == []
    assert batch_calls == [[node_id]]
    assert len(compute_calls) == 1


@pytest.mark.parametrize(
    ("rows", "expected_title"),
    [
        ({}, "Raw top preference"),
        (
            {
                "pref-raw-top": [
                    _affinity_row(np.array([1.0, 0.0, 0.0], dtype=np.float32), -1),
                ],
                "pref-adjusted-top": [
                    _affinity_row(np.array([1.0, 0.0, 0.0], dtype=np.float32), 1),
                ],
            },
            "Adjusted top preference",
        ),
    ],
)
def test_preference_feedback_controls_slot_selection_and_keeps_topical_slot(
    mock_graph, rows, expected_title
):
    fact = _unit_node("fact-slot", "Graph component implementation", node_type="fact")
    raw_top = _unit_node("pref-raw-top", "Raw top preference")
    adjusted_top = _unit_node("pref-adjusted-top", "Adjusted top preference")
    main = [{"node": fact, "score": 0.8, "raw_cosine": 0.8, "source": "hybrid"}]
    preferences = [
        {"node": raw_top, "score": 0.8, "raw_cosine": 0.8, "source": "hybrid"},
        {"node": adjusted_top, "score": 0.8, "raw_cosine": 0.8, "source": "hybrid"},
    ]
    builder, _engine, _encoder = _unit_builder(mock_graph, main, preferences)
    score_by_id = {"fact-slot": 0.80, "pref-raw-top": 0.60, "pref-adjusted-top": 0.55}
    fake_rerank = _fake_rerank(score_by_id, [])
    batch_calls = []

    def fetch(_conn, node_ids):
        batch_calls.append(list(node_ids))
        return {node_id: rows.get(node_id, []) for node_id in node_ids}

    with patch("ormah.embeddings.reranker.rerank", side_effect=fake_rerank), \
         patch("ormah.engine.affinity.batch_fetch_affinity", side_effect=fetch):
        result, ids = builder.build_whisper_context(
            prompt="build the graph component",
            max_nodes=3,
            min_score=0.0,
            reranker_enabled=True,
            reranker_min_score=0.0,
            reranker_blend_alpha=1.0,
            injection_gate=0.45,
            preference_applicability_enabled=True,
            preference_applicability_gate=0.40,
            preference_max_nodes=1,
            _return_debug=True,
        )

    expected_id = "pref-adjusted-top" if rows else "pref-raw-top"
    assert ids == [expected_id, "fact-slot"]
    assert expected_title in result
    assert "Graph component implementation" in result
    assert batch_calls == [["fact-slot"], ["pref-raw-top", "pref-adjusted-top"]]


def test_preference_affinity_uses_raw_prompt_vector_and_no_extra_encoder_call(mock_graph):
    raw_prompt_vec = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    query_vec = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    node = _unit_node("pref-raw-vector", "Raw vector preference")
    builder, _engine, encoder = _unit_builder(
        mock_graph,
        [],
        [{"node": node, "score": 0.8}],
        prompt_vec=raw_prompt_vec,
    )
    encoder.encode_query.return_value = query_vec
    observed = []

    def compute(current_vec, _node_id, _rows, _settings):
        observed.append(current_vec.copy())
        return 0.0

    queries = []
    fake_rerank = _fake_rerank({node["id"]: 0.45}, queries)
    with patch("ormah.embeddings.reranker.rerank", side_effect=fake_rerank), \
         patch("ormah.engine.affinity.batch_fetch_affinity", return_value={}), \
         patch("ormah.engine.affinity.compute_affinity_boost", side_effect=compute):
        result = builder.build_whisper_context(
            prompt="review authentication",
            min_score=0.0,
            reranker_enabled=True,
            reranker_min_score=0.0,
            reranker_blend_alpha=1.0,
            injection_gate=0.40,
            preference_applicability_enabled=True,
            preference_applicability_gate=0.40,
            preference_max_nodes=1,
        )

    assert "Raw vector preference" in result
    assert len(observed) == 1
    assert np.array_equal(observed[0], raw_prompt_vec)
    assert encoder.encode.call_count == 0
    assert encoder.encode_query.call_count == 1
    assert queries == ["Relevant user preference for this action: review authentication"]


def test_preference_affinity_failure_keeps_unadjusted_applicability(mock_graph):
    node = _unit_node("pref-affinity-failure", "Affinity failure preference")
    builder, _engine, _encoder = _unit_builder(
        mock_graph, [], [{"node": node, "score": 0.8}]
    )
    fake_rerank = _fake_rerank({node["id"]: 0.45}, [])
    batch = MagicMock(side_effect=RuntimeError("synthetic affinity outage"))

    with patch("ormah.embeddings.reranker.rerank", side_effect=fake_rerank), \
         patch("ormah.engine.affinity.batch_fetch_affinity", batch):
        result = builder.build_whisper_context(
            prompt="apply a preference",
            min_score=0.0,
            reranker_enabled=True,
            reranker_min_score=0.0,
            reranker_blend_alpha=1.0,
            injection_gate=0.55,
            preference_applicability_enabled=True,
            preference_applicability_gate=0.40,
            preference_max_nodes=1,
        )

    assert "Affinity failure preference" in result
    batch.assert_called_once()


def test_missing_raw_prompt_vector_keeps_preference_and_skips_affinity(mock_graph):
    node = _unit_node("pref-missing-vector", "Missing vector preference")
    builder, engine, encoder = _unit_builder(
        mock_graph, [], [{"node": node, "score": 0.8}]
    )
    builder._classifier = SimpleNamespace(
        classify=lambda _: PromptIntent(categories=["general"], prompt_vec=None),
    )
    encoder.encode.return_value = None
    fake_rerank = _fake_rerank({node["id"]: 0.45}, [])
    batch = MagicMock(return_value={})

    with patch("ormah.embeddings.reranker.rerank", side_effect=fake_rerank), \
         patch("ormah.engine.affinity.batch_fetch_affinity", batch):
        result = builder.build_whisper_context(
            prompt="apply a preference",
            min_score=0.0,
            reranker_enabled=True,
            reranker_min_score=0.0,
            reranker_blend_alpha=1.0,
            injection_gate=0.55,
            preference_applicability_enabled=True,
            preference_applicability_gate=0.40,
            preference_max_nodes=1,
        )

    assert "Missing vector preference" in result
    batch.assert_not_called()
    assert engine._get_hybrid_search.return_value.encoder.encode.call_count == 1


@pytest.mark.parametrize(
    ("preference_enabled", "reranker_enabled"),
    [(False, True), (True, False)],
)
def test_disabled_preference_channel_does_not_do_affinity_work(
    mock_graph, preference_enabled, reranker_enabled
):
    node = _unit_node("pref-disabled", "Disabled preference")
    builder, engine, _encoder = _unit_builder(
        mock_graph, [], [{"node": node, "score": 0.8}]
    )
    batch = MagicMock(return_value={})
    fake_rerank = _fake_rerank({node["id"]: 0.45}, [])

    with patch("ormah.embeddings.reranker.rerank", side_effect=fake_rerank), \
         patch("ormah.engine.affinity.batch_fetch_affinity", batch):
        builder.build_whisper_context(
            prompt="apply a preference",
            min_score=0.0,
            reranker_enabled=reranker_enabled,
            reranker_min_score=0.0,
            reranker_blend_alpha=1.0,
            injection_gate=0.55,
            preference_applicability_enabled=preference_enabled,
            preference_applicability_gate=0.40,
            preference_max_nodes=1,
        )

    assert engine.recall_search_structured.call_count == 1
    batch.assert_not_called()

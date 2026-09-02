"""Tests for the transcript watcher — auto-ingestion of agent transcripts."""

from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from ormah import signal_strength
from ormah.background.session_watcher import (
    IngestResult,
    SessionHandler,
    _ingest_session,
    _load_state,
    _record_whisper_usage_signals,
    _save_state,
    _scan_sessions,
    _space_from_encoded_dir,
    start_session_watcher,
    stop_session_watcher,
)
from ormah.engine.memory_engine import MemoryEngine
from ormah.models.node import CreateNodeRequest
from ormah.transcript.parser import parse_transcript

_LLM_PATCH = "ormah.background.llm_client.llm_generate"

_LLM_RESPONSE = json.dumps({"memories": [
    {
        "content": "Chose bge-base-en-v1.5 for embeddings because it needs no task prefixes.",
        "type": "decision",
        "title": "Embedding model choice",
        "tags": ["embeddings"],
    },
]})


def _make_jsonl(path: Path, user_turns: int = 6) -> None:
    """Write a minimal JSONL transcript with the given number of user turns."""
    lines = []
    for i in range(user_turns):
        lines.append(json.dumps({
            "type": "user",
            "message": {"role": "user", "content": f"User message {i} with enough text to parse"},
        }))
        lines.append(json.dumps({
            "type": "assistant",
            "message": {"role": "assistant", "stop_reason": "end_turn", "content": [
                {"type": "text", "text": f"Assistant response {i} with some detail"},
            ]},
        }))
    path.write_text("\n".join(lines) + "\n")


def _mark_idle(path: Path) -> None:
    """Backdate mtime so _ingest_session treats the transcript as finished (idle flush).

    A fresh file is considered active, so its trailing user+assistant block is held back
    until a following user turn (or the idle flush) confirms the response is complete.
    """
    now = time.time()
    os.utime(path, (now, now - 120))


def _write_turn_jsonl(path: Path, prompt: str, response: str) -> None:
    lines = [
        {
            "type": "user",
            "message": {"role": "user", "content": prompt},
        },
        {
            "type": "assistant",
            "message": {
                "role": "assistant",
                "stop_reason": "end_turn",
                "content": [{"type": "text", "text": response}],
            },
        },
    ]
    path.write_text("\n".join(json.dumps(line) for line in lines) + "\n")


def _write_codex_turn_jsonl(path: Path, prompt: str, response: str) -> None:
    lines = [
        {
            "type": "response_item",
            "payload": {
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": prompt}],
            },
        },
        {
            "type": "response_item",
            "payload": {
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": response}],
            },
        },
        {"type": "event_msg", "payload": {"type": "task_complete"}},  # closes the turn
    ]
    path.write_text("\n".join(json.dumps(line) for line in lines) + "\n")


def _insert_injected_whisper_log(
    engine: MemoryEngine,
    *,
    node_id: str,
    session_id: str,
    prompt: str,
    space: str = "myproject",
) -> int:
    cursor = engine.db.conn.execute(
        "INSERT INTO whisper_log "
        "(session_id, space, prompt_hash, prompt_text, prompt_vec, node_id, score, "
        "was_injected, logged_at) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, datetime('now'))",
        (session_id, space, "hash-abc", prompt, b"\x00" * 4, node_id, 0.9, 1),
    )
    engine.db.conn.commit()
    return cursor.lastrowid


# --- Test 1: Space detection from encoded directory names ---

@pytest.mark.parametrize("dirname,expected", [
    ("-Users-johndoe-Projects-ormah", "ormah"),
    ("-Users-alice-Code-my-app", "app"),
    ("-home-bob-projects-foo", "foo"),
    ("", None),
    ("-", None),
    ("simple", "simple"),
])
def test_space_from_encoded_dir(dirname, expected):
    assert _space_from_encoded_dir(dirname) == expected


# --- Test 2: Basic session ingestion ---

def test_ingest_session_basic(engine, tmp_path):
    """A JSONL transcript with enough turns gets ingested and state updated."""
    watch_dir = tmp_path / "projects"
    project_dir = watch_dir / "-Users-alice-Code-myproject"
    project_dir.mkdir(parents=True)
    jsonl = project_dir / "abc123.jsonl"
    _make_jsonl(jsonl, user_turns=6)

    state = {}
    with patch(_LLM_PATCH, return_value=_LLM_RESPONSE):
        result = _ingest_session(engine, jsonl, state, watch_dir, min_turns=5)

    assert result == IngestResult.OK
    rel = str(jsonl.relative_to(watch_dir))
    assert rel in state
    entry = state[rel]
    assert entry["session_id"] == "abc123"
    assert entry["source"] == "claude_code"
    assert entry["space"] == "myproject"
    assert entry["user_turns"] == 6
    assert len(entry["node_ids"]) == 1


def test_subagent_transcript_is_not_ingested(engine, tmp_path):
    """Subagent transcripts (<uuid>/subagents/agent-*.jsonl) are internal agent scratch.

    They must never be ingested as memories, even with turns above min_turns — otherwise
    every Task-tool spawn balloons the store with low-value granular memories.
    """
    watch_dir = tmp_path / "projects"
    sub_dir = watch_dir / "-Users-alice-Code-myproject" / "abc123" / "subagents"
    sub_dir.mkdir(parents=True)
    jsonl = sub_dir / "agent-deadbeef.jsonl"
    _make_jsonl(jsonl, user_turns=6)  # well above min_turns

    state = {}
    with patch(_LLM_PATCH, return_value=_LLM_RESPONSE):
        result = _ingest_session(engine, jsonl, state, watch_dir, min_turns=5)

    assert result == IngestResult.NO_PROGRESS
    assert state == {}


def test_scan_skips_subagents_keeps_primary(engine, tmp_path):
    """A scan ingests the primary session transcript but skips sibling subagent transcripts."""
    watch_dir = tmp_path / "projects"
    project_dir = watch_dir / "-Users-alice-Code-myproject"
    sub_dir = project_dir / "abc123" / "subagents"
    sub_dir.mkdir(parents=True)
    _make_jsonl(project_dir / "abc123.jsonl", user_turns=6)
    _make_jsonl(sub_dir / "agent-deadbeef.jsonl", user_turns=6)

    with patch(_LLM_PATCH, return_value=_LLM_RESPONSE):
        ingested = _scan_sessions(engine, watch_dir, min_turns=5, lookback_hours=9999)

    assert ingested == 1
    state = _load_state(watch_dir)
    sub_rel = str((sub_dir / "agent-deadbeef.jsonl").relative_to(watch_dir))
    assert sub_rel not in state


def test_ingest_codex_session_resolves_rollout_session_id_and_space(engine, tmp_path):
    """Codex rollout filenames are matched back to the whisper_log hook session id."""
    watch_dir = tmp_path / ".codex" / "sessions"
    transcript_dir = watch_dir / "2026" / "06" / "24"
    transcript_dir.mkdir(parents=True)
    jsonl = transcript_dir / "rollout-2026-06-24T12-00-00-sess-456.jsonl"

    prompt = "Why is the Codex watcher less polished?"
    response = (
        "The Codex watcher should resolve rollout filenames through whisper_log session ids "
        "instead of trusting the transcript filename stem."
    )
    _write_codex_turn_jsonl(jsonl, prompt, response)
    _mark_idle(jsonl)  # finished single-turn session → idle flush

    node_id, _ = engine.remember(CreateNodeRequest(
        content="Codex watcher rollout filenames should be resolved through whisper_log session ids.",
        type="fact",
        title="Codex watcher session id resolution",
    ))
    whisper_log_id = _insert_injected_whisper_log(
        engine,
        node_id=node_id,
        session_id="sess-456",
        prompt=prompt,
        space="ormah",
    )

    state = {}
    with patch(_LLM_PATCH, return_value=_LLM_RESPONSE):
        assert _ingest_session(engine, jsonl, state, watch_dir, min_turns=1) == IngestResult.OK

    rel = str(jsonl.relative_to(watch_dir))
    entry = state[rel]
    assert entry["session_id"] == "sess-456"
    assert entry["source"] == "codex"
    assert entry["space"] == "ormah"
    assert entry["signals_recorded"] == 1

    signal = engine.db.conn.execute(
        "SELECT * FROM signals WHERE whisper_log_id = ?", (whisper_log_id,)
    ).fetchone()
    assert signal is not None
    assert signal["session_id"] == "sess-456"
    assert signal["agent_id"] == "codex"
    assert signal["polarity"] == 1


def test_ingest_codex_session_without_whisper_log_does_not_infer_date_space(engine, tmp_path):
    """Codex date folders are storage layout, not project space."""
    watch_dir = tmp_path / ".codex" / "sessions"
    transcript_dir = watch_dir / "2026" / "06" / "24"
    transcript_dir.mkdir(parents=True)
    jsonl = transcript_dir / "rollout-2026-06-24T12-00-00-no-log.jsonl"
    _write_codex_turn_jsonl(jsonl, "Prompt with enough content", "Response with enough content")
    _mark_idle(jsonl)  # finished single-turn session → idle flush

    state = {}
    with patch(_LLM_PATCH, return_value=_LLM_RESPONSE):
        assert _ingest_session(engine, jsonl, state, watch_dir, min_turns=1) == IngestResult.OK

    entry = state[str(jsonl.relative_to(watch_dir))]
    assert entry["source"] == "codex"
    assert entry["space"] is None


def test_record_whisper_usage_signal_promotes_clear_reference(engine, tmp_path):
    """Clear references in an assistant response create a signal and affinity row."""
    prompt = "How should we solve feedback collection?"
    response = "The right fix is the transcript watcher mines feedback usage approach."
    transcript_path = tmp_path / "usage-session.jsonl"
    _write_turn_jsonl(transcript_path, prompt, response)
    transcript = parse_transcript(transcript_path)

    node_id, _ = engine.remember(CreateNodeRequest(
        content="The transcript watcher mines feedback usage from completed transcripts.",
        type="fact",
        title="Transcript watcher mines feedback usage",
    ))
    whisper_log_id = _insert_injected_whisper_log(
        engine,
        node_id=node_id,
        session_id="usage-session",
        prompt=prompt,
    )

    recorded = _record_whisper_usage_signals(engine, transcript)

    assert recorded == 1
    signal = engine.db.conn.execute(
        "SELECT * FROM signals WHERE whisper_log_id = ?", (whisper_log_id,)
    ).fetchone()
    assert signal is not None
    assert signal["signal_type"] == "whisper_referenced"
    assert signal["polarity"] == 1
    assert signal["source"] == "transcript_watcher_heuristic"
    assert signal["surface"] == "transcript_watcher"
    assert signal["agent_id"] == "claude_code"

    affinity = engine.db.conn.execute(
        "SELECT * FROM affinity WHERE whisper_log_id = ?", (whisper_log_id,)
    ).fetchone()
    assert affinity is not None
    assert affinity["node_id"] == node_id
    assert affinity["signal"] == 1
    assert affinity["source"] == "auto_heuristic"


def test_record_whisper_usage_signal_keeps_unreferenced_neutral(engine, tmp_path):
    """Unreferenced whispers are observable but do not become negative affinity."""
    prompt = "How should we solve feedback collection?"
    response = "We should first fix the database uniqueness key."
    transcript_path = tmp_path / "neutral-session.jsonl"
    _write_turn_jsonl(transcript_path, prompt, response)
    transcript = parse_transcript(transcript_path)

    node_id, _ = engine.remember(CreateNodeRequest(
        content="Graph rendering should use level of detail for large datasets.",
        type="fact",
        title="Large graph rendering performance",
    ))
    whisper_log_id = _insert_injected_whisper_log(
        engine,
        node_id=node_id,
        session_id="neutral-session",
        prompt=prompt,
    )

    recorded = _record_whisper_usage_signals(engine, transcript)

    assert recorded == 1
    signal = engine.db.conn.execute(
        "SELECT * FROM signals WHERE whisper_log_id = ?", (whisper_log_id,)
    ).fetchone()
    assert signal is not None
    assert signal["signal_type"] == "whisper_unreferenced"
    assert signal["polarity"] == 0

    affinity = engine.db.conn.execute(
        "SELECT * FROM affinity WHERE whisper_log_id = ?", (whisper_log_id,)
    ).fetchone()
    assert affinity is None


def test_llm_judge_disabled_by_default(engine, tmp_path):
    """The transcript watcher does not call the LLM unless the judge is enabled."""
    prompt = "How should we solve feedback collection?"
    response = "We should first fix the database uniqueness key."
    transcript_path = tmp_path / "judge-disabled-session.jsonl"
    _write_turn_jsonl(transcript_path, prompt, response)
    transcript = parse_transcript(transcript_path)

    node_id, _ = engine.remember(CreateNodeRequest(
        content="Graph rendering should use level of detail for large datasets.",
        type="fact",
        title="Large graph rendering performance",
    ))
    _insert_injected_whisper_log(
        engine,
        node_id=node_id,
        session_id="judge-disabled-session",
        prompt=prompt,
    )
    engine.settings.llm_provider = "ollama"

    mock_llm = MagicMock(return_value=json.dumps({"verdicts": []}))
    with patch(_LLM_PATCH, mock_llm):
        recorded = _record_whisper_usage_signals(engine, transcript)

    assert recorded == 1
    mock_llm.assert_not_called()


def test_llm_judge_promotes_used_verdict(engine, tmp_path):
    """A confident LLM 'used' verdict creates positive affinity for an ambiguous row."""
    prompt = "What deployment marker should we use?"
    response = "That guidance is the right one for the rollout."
    transcript_path = tmp_path / "judge-used-session.jsonl"
    _write_turn_jsonl(transcript_path, prompt, response)
    transcript = parse_transcript(transcript_path)

    node_id, _ = engine.remember(CreateNodeRequest(
        content="Use blue deployment markers when rollback plans need quick visual checks.",
        type="fact",
        title="Blue deployment rollback marker",
    ))
    whisper_log_id = _insert_injected_whisper_log(
        engine,
        node_id=node_id,
        session_id="judge-used-session",
        prompt=prompt,
    )
    engine.settings.llm_provider = "ollama"
    engine.settings.feedback_llm_judge_enabled = True

    llm_response = json.dumps({
        "verdicts": [{
            "whisper_log_id": whisper_log_id,
            "verdict": "used",
            "confidence": 0.88,
            "reason": "The answer endorses the injected deployment guidance.",
        }]
    })
    with patch(_LLM_PATCH, return_value=llm_response) as mock_llm:
        recorded = _record_whisper_usage_signals(engine, transcript)

    assert recorded == 2
    call_kwargs = mock_llm.call_args.kwargs
    assert call_kwargs["response_format"]["type"] == "json_schema"
    assert call_kwargs["response_format"]["json_schema"]["name"] == "whisper_feedback_verdicts"
    assert call_kwargs["temperature"] == 0
    assert call_kwargs["max_tokens"] == 512

    judge_signal = engine.db.conn.execute(
        "SELECT * FROM signals WHERE whisper_log_id = ? "
        "AND source = 'transcript_watcher_llm_judge'",
        (whisper_log_id,),
    ).fetchone()
    assert judge_signal is not None
    assert judge_signal["signal_type"] == "whisper_judged_used"
    assert judge_signal["polarity"] == 1
    # #218: strength is the judge's band position now, not its raw confidence.
    assert judge_signal["strength"] == pytest.approx(
        signal_strength.judge_strength(0.88, engine.settings.feedback_llm_judge_min_confidence, 1)
    )

    affinity = engine.db.conn.execute(
        "SELECT * FROM affinity WHERE whisper_log_id = ?", (whisper_log_id,)
    ).fetchone()
    assert affinity is not None
    assert affinity["signal"] == 1
    assert affinity["source"] == "auto_llm_judge"


def test_llm_judge_falls_back_to_json_object_mode(engine, tmp_path):
    """Providers that reject JSON Schema can still use the JSON-object fallback."""
    prompt = "How should we solve feedback collection?"
    response = "We should first fix the database uniqueness key."
    transcript_path = tmp_path / "judge-schema-fallback-session.jsonl"
    _write_turn_jsonl(transcript_path, prompt, response)
    transcript = parse_transcript(transcript_path)

    node_id, _ = engine.remember(CreateNodeRequest(
        content="Graph rendering should use level of detail for large datasets.",
        type="fact",
        title="Large graph rendering performance",
    ))
    whisper_log_id = _insert_injected_whisper_log(
        engine,
        node_id=node_id,
        session_id="judge-schema-fallback-session",
        prompt=prompt,
    )
    engine.settings.llm_provider = "ollama"
    engine.settings.feedback_llm_judge_enabled = True

    llm_response = json.dumps({
        "verdicts": [{
            "whisper_log_id": whisper_log_id,
            "verdict": "irrelevant",
            "confidence": 0.91,
        }]
    })
    mock_llm = MagicMock(side_effect=[None, llm_response])
    with patch(_LLM_PATCH, mock_llm):
        recorded = _record_whisper_usage_signals(engine, transcript)

    assert recorded == 2
    assert mock_llm.call_count == 2
    first_kwargs = mock_llm.call_args_list[0].kwargs
    second_kwargs = mock_llm.call_args_list[1].kwargs
    assert first_kwargs["response_format"]["type"] == "json_schema"
    assert first_kwargs["temperature"] == 0
    assert first_kwargs["max_tokens"] == 512
    assert "response_format" not in second_kwargs
    assert second_kwargs["json_mode"] is True
    assert second_kwargs["temperature"] == 0
    assert second_kwargs["max_tokens"] == 512

    judge_signal = engine.db.conn.execute(
        "SELECT * FROM signals WHERE whisper_log_id = ? "
        "AND source = 'transcript_watcher_llm_judge'",
        (whisper_log_id,),
    ).fetchone()
    assert judge_signal is not None
    assert judge_signal["signal_type"] == "whisper_judged_irrelevant"


def test_llm_judge_promotes_irrelevant_verdict_as_negative(engine, tmp_path):
    """A confident LLM irrelevant verdict is the automatic negative-feedback path."""
    prompt = "How should we solve feedback collection?"
    response = "We should first fix the database uniqueness key."
    transcript_path = tmp_path / "judge-negative-session.jsonl"
    _write_turn_jsonl(transcript_path, prompt, response)
    transcript = parse_transcript(transcript_path)

    node_id, _ = engine.remember(CreateNodeRequest(
        content="Graph rendering should use level of detail for large datasets.",
        type="fact",
        title="Large graph rendering performance",
    ))
    whisper_log_id = _insert_injected_whisper_log(
        engine,
        node_id=node_id,
        session_id="judge-negative-session",
        prompt=prompt,
    )
    engine.settings.llm_provider = "ollama"
    engine.settings.feedback_llm_judge_enabled = True

    llm_response = json.dumps({
        "verdicts": [{
            "whisper_log_id": whisper_log_id,
            "verdict": "irrelevant",
            "confidence": 0.91,
            "reason": "The memory is about graph UI rendering, not feedback schema work.",
        }]
    })
    with patch(_LLM_PATCH, return_value=llm_response):
        recorded = _record_whisper_usage_signals(engine, transcript)

    assert recorded == 2
    judge_signal = engine.db.conn.execute(
        "SELECT * FROM signals WHERE whisper_log_id = ? "
        "AND source = 'transcript_watcher_llm_judge'",
        (whisper_log_id,),
    ).fetchone()
    assert judge_signal is not None
    assert judge_signal["signal_type"] == "whisper_judged_irrelevant"
    assert judge_signal["polarity"] == -1

    affinity = engine.db.conn.execute(
        "SELECT * FROM affinity WHERE whisper_log_id = ?", (whisper_log_id,)
    ).fetchone()
    assert affinity is not None
    assert affinity["signal"] == -1
    assert affinity["source"] == "auto_llm_judge"


def test_llm_judge_low_confidence_records_uncertain_without_affinity(engine, tmp_path):
    """Low-confidence LLM verdicts remain observable but do not affect ranking."""
    prompt = "How should we solve feedback collection?"
    response = "We should first fix the database uniqueness key."
    transcript_path = tmp_path / "judge-low-confidence-session.jsonl"
    _write_turn_jsonl(transcript_path, prompt, response)
    transcript = parse_transcript(transcript_path)

    node_id, _ = engine.remember(CreateNodeRequest(
        content="Graph rendering should use level of detail for large datasets.",
        type="fact",
        title="Large graph rendering performance",
    ))
    whisper_log_id = _insert_injected_whisper_log(
        engine,
        node_id=node_id,
        session_id="judge-low-confidence-session",
        prompt=prompt,
    )
    engine.settings.llm_provider = "ollama"
    engine.settings.feedback_llm_judge_enabled = True

    llm_response = json.dumps({
        "verdicts": [{
            "whisper_log_id": whisper_log_id,
            "verdict": "irrelevant",
            "confidence": 0.4,
            "reason": "Maybe unrelated, but confidence is low.",
        }]
    })
    with patch(_LLM_PATCH, return_value=llm_response):
        recorded = _record_whisper_usage_signals(engine, transcript)

    assert recorded == 2
    judge_signal = engine.db.conn.execute(
        "SELECT * FROM signals WHERE whisper_log_id = ? "
        "AND source = 'transcript_watcher_llm_judge'",
        (whisper_log_id,),
    ).fetchone()
    assert judge_signal is not None
    assert judge_signal["signal_type"] == "whisper_judged_uncertain"
    assert judge_signal["polarity"] == 0

    affinity = engine.db.conn.execute(
        "SELECT * FROM affinity WHERE whisper_log_id = ?", (whisper_log_id,)
    ).fetchone()
    assert affinity is None


def test_llm_judge_skips_clear_heuristic_positive(engine, tmp_path):
    """The optional judge does not spend an LLM call on clear heuristic positives."""
    prompt = "How should we solve feedback collection?"
    response = "The right fix is the transcript watcher mines feedback usage approach."
    transcript_path = tmp_path / "judge-skip-positive-session.jsonl"
    _write_turn_jsonl(transcript_path, prompt, response)
    transcript = parse_transcript(transcript_path)

    node_id, _ = engine.remember(CreateNodeRequest(
        content="The transcript watcher mines feedback usage from completed transcripts.",
        type="fact",
        title="Transcript watcher mines feedback usage",
    ))
    _insert_injected_whisper_log(
        engine,
        node_id=node_id,
        session_id="judge-skip-positive-session",
        prompt=prompt,
    )
    engine.settings.llm_provider = "ollama"
    engine.settings.feedback_llm_judge_enabled = True

    mock_llm = MagicMock(return_value=json.dumps({"verdicts": []}))
    with patch(_LLM_PATCH, mock_llm):
        recorded = _record_whisper_usage_signals(engine, transcript)

    assert recorded == 1
    mock_llm.assert_not_called()


def test_llm_judge_is_idempotent(engine, tmp_path):
    """Once a judge signal exists, the same whisper row is not judged again."""
    prompt = "How should we solve feedback collection?"
    response = "We should first fix the database uniqueness key."
    transcript_path = tmp_path / "judge-idempotent-session.jsonl"
    _write_turn_jsonl(transcript_path, prompt, response)
    transcript = parse_transcript(transcript_path)

    node_id, _ = engine.remember(CreateNodeRequest(
        content="Graph rendering should use level of detail for large datasets.",
        type="fact",
        title="Large graph rendering performance",
    ))
    whisper_log_id = _insert_injected_whisper_log(
        engine,
        node_id=node_id,
        session_id="judge-idempotent-session",
        prompt=prompt,
    )
    engine.settings.llm_provider = "ollama"
    engine.settings.feedback_llm_judge_enabled = True

    mock_llm = MagicMock(return_value=json.dumps({
        "verdicts": [{
            "whisper_log_id": whisper_log_id,
            "verdict": "irrelevant",
            "confidence": 0.91,
            "reason": "The memory is about graph UI rendering.",
        }]
    }))
    with patch(_LLM_PATCH, mock_llm):
        assert _record_whisper_usage_signals(engine, transcript) == 2
        assert _record_whisper_usage_signals(engine, transcript) == 0

    assert mock_llm.call_count == 1
    signal_count = engine.db.conn.execute(
        "SELECT COUNT(*) AS count FROM signals WHERE whisper_log_id = ?",
        (whisper_log_id,),
    ).fetchone()["count"]
    affinity_count = engine.db.conn.execute(
        "SELECT COUNT(*) AS count FROM affinity WHERE whisper_log_id = ?",
        (whisper_log_id,),
    ).fetchone()["count"]
    assert signal_count == 2
    assert affinity_count == 1


# --- Test 3: Min turns filter ---

def test_min_turns_filter(engine, tmp_path):
    """A session with too few turns is skipped."""
    watch_dir = tmp_path / "projects"
    project_dir = watch_dir / "-Users-alice-Code-myproject"
    project_dir.mkdir(parents=True)
    jsonl = project_dir / "short.jsonl"
    _make_jsonl(jsonl, user_turns=3)

    state = {}
    with patch(_LLM_PATCH, return_value=_LLM_RESPONSE):
        result = _ingest_session(engine, jsonl, state, watch_dir, min_turns=5)

    # A fresh (active, not idle) file below min_turns hits the short-tail branch and
    # defers → TRANSIENT (retry until it grows past min_turns or the session idles).
    assert result == IngestResult.TRANSIENT
    assert str(jsonl.relative_to(watch_dir)) not in state


# --- Test 4: Unchanged session skipped ---

def test_unchanged_session_skipped(engine, tmp_path):
    """Same hash → session not re-ingested."""
    watch_dir = tmp_path / "projects"
    project_dir = watch_dir / "-Users-alice-Code-myproject"
    project_dir.mkdir(parents=True)
    jsonl = project_dir / "session.jsonl"
    _make_jsonl(jsonl, user_turns=6)

    state = {}
    with patch(_LLM_PATCH, return_value=_LLM_RESPONSE):
        assert _ingest_session(engine, jsonl, state, watch_dir, min_turns=5) == IngestResult.OK
        assert _ingest_session(engine, jsonl, state, watch_dir, min_turns=5) == IngestResult.NO_PROGRESS


# --- Test 5: Scan respects lookback ---

def test_scan_respects_lookback(engine, tmp_path):
    """Old files are skipped during catch-up scan, recent ones ingested."""
    watch_dir = tmp_path / "projects"
    project_dir = watch_dir / "-Users-alice-Code-proj"
    project_dir.mkdir(parents=True)

    recent = project_dir / "recent.jsonl"
    _make_jsonl(recent, user_turns=6)

    old = project_dir / "old.jsonl"
    _make_jsonl(old, user_turns=6)
    # Set mtime to 200 hours ago (beyond 72h lookback)
    import os
    old_time = time.time() - (200 * 3600)
    os.utime(old, (old_time, old_time))

    with patch(_LLM_PATCH, return_value=_LLM_RESPONSE):
        count = _scan_sessions(engine, watch_dir, min_turns=5, lookback_hours=72)

    assert count == 1  # only recent
    state = _load_state(watch_dir)
    assert str(recent.relative_to(watch_dir)) in state
    assert str(old.relative_to(watch_dir)) not in state


# --- Test 6: Debounce coalesces writes ---

def test_debounce_coalesces_writes(engine, tmp_path):
    """5 rapid events → 1 ingestion call."""
    watch_dir = tmp_path / "projects"
    project_dir = watch_dir / "-Users-alice-Code-proj"
    project_dir.mkdir(parents=True)

    handler = SessionHandler(engine, watch_dir, debounce_seconds=0.3, min_turns=5)
    jsonl = project_dir / "active.jsonl"

    call_count = 0
    original_ingest = _ingest_session

    def counting_ingest(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        return original_ingest(*args, **kwargs)

    with patch("ormah.background.session_watcher._ingest_session", side_effect=counting_ingest):
        for i in range(5):
            _make_jsonl(jsonl, user_turns=6 + i)
            from watchdog.events import FileModifiedEvent
            handler.on_modified(FileModifiedEvent(str(jsonl)))
            time.sleep(0.05)

        # Wait for debounce
        time.sleep(0.5)

    assert call_count == 1


# --- Test 7: Lifecycle start/stop ---

def test_lifecycle_start_stop(engine, tmp_path):
    """Observer starts and stops cleanly."""
    watch_dir = tmp_path / "projects"
    watch_dir.mkdir()

    engine.settings.session_watcher_enabled = True
    engine.settings.session_watcher_dir = watch_dir
    engine.settings.session_watcher_debounce_seconds = 10.0

    observers = start_session_watcher(engine)
    try:
        assert len(observers) == 1
        assert observers[0].observer.is_alive()
    finally:
        stop_session_watcher(observers)

    # Give observer thread a moment to stop
    time.sleep(0.1)
    assert not observers[0].observer.is_alive()


def test_lifecycle_includes_codex_sessions_when_using_default_agent_dir(
    engine,
    tmp_path,
    monkeypatch,
):
    """Default watcher setup starts observers for existing Claude and Codex session dirs."""
    home = tmp_path / "home"
    claude_dir = home / ".claude" / "projects"
    codex_dir = home / ".codex" / "sessions"
    claude_dir.mkdir(parents=True)
    codex_dir.mkdir(parents=True)
    monkeypatch.setenv("HOME", str(home))

    engine.settings.session_watcher_enabled = True
    engine.settings.session_watcher_dir = Path("~/.claude/projects")
    engine.settings.session_watcher_debounce_seconds = 10.0

    observers = start_session_watcher(engine)
    try:
        assert len(observers) == 2
        assert all(w.observer.is_alive() for w in observers)
    finally:
        stop_session_watcher(observers)


# --- Test 8: Disabled returns empty ---

def test_disabled_returns_empty(engine, tmp_path):
    """session_watcher_enabled=False → empty list."""
    engine.settings.session_watcher_enabled = False
    observers = start_session_watcher(engine)
    assert observers == []


# --- Test 9: State persistence ---

def test_state_persistence(tmp_path):
    """State file survives save/load roundtrip."""
    watch_dir = tmp_path / "projects"
    watch_dir.mkdir()

    state = {
        "proj/abc.jsonl": {
            "hash": "deadbeef",
            "last_ingested": "2024-01-01T00:00:00",
            "session_id": "abc",
            "space": "proj",
            "user_turns": 10,
            "node_ids": ["id-1", "id-2"],
        }
    }
    _save_state(watch_dir, state)

    loaded = _load_state(watch_dir)
    assert loaded == state
    assert loaded["proj/abc.jsonl"]["hash"] == "deadbeef"
    assert loaded["proj/abc.jsonl"]["node_ids"] == ["id-1", "id-2"]


# --- Test 10: Nonexistent watch dir ---

def test_nonexistent_watch_dir(engine, tmp_path):
    """Nonexistent watch dir returns empty, no crash."""
    engine.settings.session_watcher_enabled = True
    engine.settings.session_watcher_dir = tmp_path / "does-not-exist"

    observers = start_session_watcher(engine)
    assert observers == []


def test_start_returns_before_startup_reconcile_finishes(engine, tmp_path):
    """Startup reconcile runs off the bind path after observers are active."""
    import threading

    import ormah.background.session_watcher as sw

    watch_dir = tmp_path / "projects"
    watch_dir.mkdir(parents=True)
    engine.settings.session_watcher_enabled = True
    engine.settings.session_watcher_dir = watch_dir
    engine.settings.session_watcher_debounce_seconds = 10.0

    started = threading.Event()
    release = threading.Event()

    def blocking_reconcile(watches):
        started.set()
        release.wait(5)
        return 0

    with patch.object(sw, "run_session_reconcile", side_effect=blocking_reconcile):
        t0 = time.monotonic()
        watches = sw.start_session_watcher(engine)
        elapsed = time.monotonic() - t0
        try:
            assert elapsed < 1.0
            assert len(watches) == 1
            assert watches[0].observer.is_alive()
            assert watches[0].startup_reconcile_thread is not None
            assert started.wait(2)
        finally:
            release.set()
            sw.stop_session_watcher(watches)


def test_startup_reconcile_uses_live_handler_state(engine, tmp_path):
    """The off-bind startup catch-up reuses SessionHandler.reconcile, not a separate state owner."""
    watch_dir = tmp_path / "projects"
    project_dir = watch_dir / "-Users-alice-Code-myproject"
    project_dir.mkdir(parents=True)
    jsonl = project_dir / "abc123.jsonl"
    _make_jsonl(jsonl, user_turns=6)
    _mark_idle(jsonl)

    engine.settings.session_watcher_enabled = True
    engine.settings.session_watcher_dir = watch_dir
    engine.settings.session_watcher_debounce_seconds = 10.0

    with patch(_LLM_PATCH, return_value=_LLM_RESPONSE):
        watches = start_session_watcher(engine)
        try:
            thread = watches[0].startup_reconcile_thread
            assert thread is not None
            thread.join(timeout=5)
            assert not thread.is_alive()
            rel = str(jsonl.relative_to(watch_dir))
            assert rel in watches[0].handler._state
        finally:
            stop_session_watcher(watches)


def test_stop_drains_live_inflight_ingest(engine, tmp_path):
    """stop_session_watcher waits for live ingest work before returning."""
    import threading

    import ormah.background.session_watcher as sw

    watch_dir = tmp_path / "projects"
    project_dir = watch_dir / "-Users-alice-Code-myproject"
    project_dir.mkdir(parents=True)
    jsonl = project_dir / "abc123.jsonl"
    _make_jsonl(jsonl, user_turns=6)
    _mark_idle(jsonl)

    engine.settings.session_watcher_enabled = True
    engine.settings.session_watcher_dir = watch_dir
    engine.settings.session_watcher_debounce_seconds = 10.0
    engine.settings.session_watcher_lookback_hours = -1

    started = threading.Event()
    release = threading.Event()

    def blocking_ingest(*args, **kwargs):
        started.set()
        release.wait(5)
        return IngestResult.TRANSIENT

    with patch.object(sw, "_ingest_session", side_effect=blocking_ingest):
        watches = sw.start_session_watcher(engine)
        live = threading.Thread(target=watches[0].handler._do_ingest, args=(jsonl,))
        live.start()
        assert started.wait(2)

        stopped = threading.Event()
        stopper = threading.Thread(
            target=lambda: (sw.stop_session_watcher(watches), stopped.set()),
        )
        stopper.start()
        assert not stopped.wait(0.5)
        release.set()
        assert stopped.wait(5)
        live.join(timeout=5)
        stopper.join(timeout=5)

    assert watches[0].handler.in_flight_count() == 0


def test_do_ingest_rejects_work_after_stop(engine, tmp_path):
    """A timer firing after shutdown begins must not touch ingest/DB work."""
    import ormah.background.session_watcher as sw

    watch_dir = tmp_path / "projects"
    project_dir = watch_dir / "-Users-alice-Code-myproject"
    project_dir.mkdir(parents=True)
    jsonl = project_dir / "abc123.jsonl"
    _make_jsonl(jsonl, user_turns=6)
    _mark_idle(jsonl)

    handler = sw.SessionHandler(engine, watch_dir, 60.0, 5, 30.0, 9999)
    handler.request_stop()
    ingest = MagicMock(return_value=IngestResult.OK)

    with patch.object(sw, "_ingest_session", ingest):
        assert handler._do_ingest(jsonl) == IngestResult.TRANSIENT

    ingest.assert_not_called()


# --- Test 11: Incremental — only appended turns are re-ingested ---

def test_incremental_only_new_turns(engine, tmp_path):
    """After the first ingest, a later change feeds ONLY the appended turns to ingest."""
    watch_dir = tmp_path / "projects"
    project_dir = watch_dir / "-Users-alice-Code-proj"
    project_dir.mkdir(parents=True)
    jsonl = project_dir / "active.jsonl"
    _make_jsonl(jsonl, user_turns=6)

    captured: list[str] = []
    real_ingest = engine.ingest_conversation

    def capture(content, **kwargs):
        captured.append(content)
        return real_ingest(content=content, **kwargs)

    state = {}
    with patch(_LLM_PATCH, return_value=_LLM_RESPONSE), \
         patch.object(engine, "ingest_conversation", side_effect=capture):
        assert _ingest_session(engine, jsonl, state, watch_dir, min_turns=5) == IngestResult.OK
        first_offset = state[str(jsonl.relative_to(watch_dir))]["end_offset"]
        assert first_offset > 0

        _make_jsonl(jsonl, user_turns=12)  # identical first 6 turns + 6 appended
        assert _ingest_session(engine, jsonl, state, watch_dir, min_turns=5) == IngestResult.OK

    assert "User message 0 " not in captured[1]
    assert "User message 6 " in captured[1]
    assert state[str(jsonl.relative_to(watch_dir))]["end_offset"] > first_offset


# --- Test 12: Incremental — too-few new turns defers ---

def test_incremental_defers_small_append(engine, tmp_path):
    """A change adding fewer than min_turns new turns does not trigger a second ingest."""
    watch_dir = tmp_path / "projects"
    project_dir = watch_dir / "-Users-alice-Code-proj"
    project_dir.mkdir(parents=True)
    jsonl = project_dir / "active.jsonl"
    _make_jsonl(jsonl, user_turns=6)

    calls = 0
    real_ingest = engine.ingest_conversation

    def counting(content, **kwargs):
        nonlocal calls
        calls += 1
        return real_ingest(content=content, **kwargs)

    state = {}
    with patch(_LLM_PATCH, return_value=_LLM_RESPONSE), \
         patch.object(engine, "ingest_conversation", side_effect=counting):
        assert _ingest_session(engine, jsonl, state, watch_dir, min_turns=5) == IngestResult.OK
        saved = dict(state[str(jsonl.relative_to(watch_dir))])

        _make_jsonl(jsonl, user_turns=8)  # only 2 new turns < min_turns, file still active → TRANSIENT defer
        assert _ingest_session(engine, jsonl, state, watch_dir, min_turns=5) == IngestResult.TRANSIENT

    assert calls == 1
    assert state[str(jsonl.relative_to(watch_dir))] == saved


# --- Test 13: Shrink resets the cursor ---

def test_shrink_resets_cursor(engine, tmp_path):
    """A file that shrinks below the stored offset is re-ingested from the start."""
    watch_dir = tmp_path / "projects"
    project_dir = watch_dir / "-Users-alice-Code-proj"
    project_dir.mkdir(parents=True)
    jsonl = project_dir / "active.jsonl"
    _make_jsonl(jsonl, user_turns=10)

    captured: list[str] = []
    real_ingest = engine.ingest_conversation

    def capture(content, **kwargs):
        captured.append(content)
        return real_ingest(content=content, **kwargs)

    state = {}
    with patch(_LLM_PATCH, return_value=_LLM_RESPONSE), \
         patch.object(engine, "ingest_conversation", side_effect=capture):
        assert _ingest_session(engine, jsonl, state, watch_dir, min_turns=5) == IngestResult.OK

        _make_jsonl(jsonl, user_turns=5)  # smaller file → size < stored end_offset
        assert _ingest_session(engine, jsonl, state, watch_dir, min_turns=5) == IngestResult.OK

    assert "User message 0 " in captured[1]


# --- New tests: safe-payload ingest, idle flush + retry, in-flight guard ---


def _append_pair(path, i):
    with path.open("a") as f:
        f.write(json.dumps({
            "type": "user",
            "message": {"role": "user", "content": f"User message {i} with enough text to parse"},
        }) + "\n")
        f.write(json.dumps({
            "type": "assistant",
            "message": {"role": "assistant", "stop_reason": "end_turn", "content": [
                {"type": "text", "text": f"Assistant response {i} with some detail"},
            ]},
        }) + "\n")


def _append_user(path, i):
    with path.open("a") as f:
        f.write(json.dumps({
            "type": "user",
            "message": {"role": "user", "content": f"User message {i} with enough text to parse"},
        }) + "\n")


def _append_assistant(path, i, stop_reason="end_turn"):
    """Append one assistant text record. stop_reason=None / "tool_use" marks it as a
    non-terminal record of a still-open response (more records to come)."""
    with path.open("a") as f:
        f.write(json.dumps({
            "type": "assistant",
            "message": {"role": "assistant", "stop_reason": stop_reason, "content": [
                {"type": "text", "text": f"Assistant response {i} with some detail"},
            ]},
        }) + "\n")


def _append_codex_turn(path, i, *, records=1, complete=True):
    """Append a Codex turn: a user message, `records` assistant text records (multi-record
    when >1), and a task_complete event unless `complete=False` (still in flight)."""
    with path.open("a") as f:
        f.write(json.dumps({"type": "response_item", "payload": {"type": "message",
            "role": "user", "content": [
                {"type": "input_text", "text": f"User message {i} with enough text to parse"}]}}) + "\n")
        for r in range(records):
            f.write(json.dumps({"type": "response_item", "payload": {"type": "message",
                "role": "assistant", "content": [
                    {"type": "output_text", "text": f"Assistant response {i} part {r} detail"}]}}) + "\n")
        if complete:
            f.write(json.dumps({"type": "event_msg", "payload": {"type": "task_complete"}}) + "\n")


def test_inflight_multirecord_response_not_split(engine, tmp_path):
    """An in-flight response (non-terminal stop_reason) is held back until its terminal
    record arrives, so a multi-record assistant response is never split from its prompt.
    Claude Code detects completion via stop_reason, not the next user turn.
    """
    watch_dir = tmp_path / "projects"
    project_dir = watch_dir / "-Users-alice-Code-proj"
    project_dir.mkdir(parents=True)
    jsonl = project_dir / "active.jsonl"
    rel = str(jsonl.relative_to(watch_dir))

    _make_jsonl(jsonl, user_turns=6)  # 6 complete (end_turn) pairs
    state = {}
    captured: list[str] = []
    real_ingest = engine.ingest_conversation

    def capture(content, **kwargs):
        captured.append(content)
        return real_ingest(content=content, **kwargs)

    with patch(_LLM_PATCH, return_value=_LLM_RESPONSE), \
         patch.object(engine, "ingest_conversation", side_effect=capture):
        # Every pair is terminal -> all committed; the cursor sits after the last one.
        assert _ingest_session(engine, jsonl, state, watch_dir, min_turns=5) == IngestResult.OK
        cursor1 = state[rel]["end_offset"]

        # New turn: prompt + a FIRST assistant record still in flight (tool_use). The
        # response is not complete, so nothing new commits and the cursor must not move
        # into the middle of the response.
        _append_user(jsonl, 6)
        _append_assistant(jsonl, 6, stop_reason="tool_use")
        assert _ingest_session(engine, jsonl, state, watch_dir, min_turns=1) != IngestResult.OK
        assert state[rel]["end_offset"] == cursor1

        # The response completes with a terminal record: prompt + BOTH assistant records
        # commit together — never split.
        _append_assistant(jsonl, 6, stop_reason="end_turn")
        assert _ingest_session(engine, jsonl, state, watch_dir, min_turns=1) == IngestResult.OK

    committed = captured[-1]
    assert "User message 6 " in committed
    assert committed.count("Assistant response 6 ") == 2  # both records, not split
    assert state[rel]["end_offset"] > cursor1


def test_codex_multirecord_turn_committed_whole_via_task_complete(engine, tmp_path):
    """A multi-record Codex turn commits as one block at its task_complete; an in-flight
    turn (no task_complete yet) is held back, never split."""
    watch_dir = tmp_path / ".codex" / "sessions"
    transcript_dir = watch_dir / "2026" / "06" / "25"
    transcript_dir.mkdir(parents=True)
    jsonl = transcript_dir / "rollout-2026-06-25T12-00-00-sess-multi.jsonl"

    _append_codex_turn(jsonl, 0, records=3, complete=True)
    _append_codex_turn(jsonl, 1, records=2, complete=True)
    # In-flight final turn: two assistant records, no task_complete yet.
    _append_codex_turn(jsonl, 2, records=2, complete=False)

    state = {}
    captured: list[str] = []
    real_ingest = engine.ingest_conversation

    def capture(content, **kwargs):
        captured.append(content)
        return real_ingest(content=content, **kwargs)

    with patch(_LLM_PATCH, return_value=_LLM_RESPONSE), \
         patch.object(engine, "ingest_conversation", side_effect=capture):
        # Fresh/active: the two task_complete turns commit whole; the in-flight one waits.
        assert _ingest_session(engine, jsonl, state, watch_dir, min_turns=1) == IngestResult.OK

    committed = captured[-1]
    assert committed.count("Assistant response 0 part ") == 3  # turn 0 not split
    assert committed.count("Assistant response 1 part ") == 2  # turn 1 not split
    assert "User message 2 " not in committed                  # in-flight turn held back


def test_legacy_mid_response_cursor_recovered(engine, tmp_path):
    """A watcher cursor an older version left BETWEEN two assistant records of one response
    triggers a full re-parse so the tail is recovered with its prompt — not orphaned."""
    watch_dir = tmp_path / "projects"
    project_dir = watch_dir / "-Users-alice-Code-proj"
    project_dir.mkdir(parents=True)
    jsonl = project_dir / "active.jsonl"
    rel = str(jsonl.relative_to(watch_dir))

    records = [
        {"type": "user", "message": {"role": "user",
            "content": "Prompt with the memory detail and enough text to parse"}},
        {"type": "assistant", "message": {"role": "assistant", "stop_reason": "tool_use",
            "content": [{"type": "text", "text": "First part of the response"}]}},
        {"type": "assistant", "message": {"role": "assistant", "stop_reason": "end_turn",
            "content": [{"type": "text", "text": "Second part with the actual answer"}]}},
    ]
    jsonl.write_text("\n".join(json.dumps(r) for r in records) + "\n")

    # A legacy state cursor saved mid-response (after the first assistant record), with the
    # CORRECT file hash — the file is unchanged. Recovery must still fire because the stored
    # offset is behind EOF (the hash short-circuit only skips a fully-consumed file).
    from ormah.background.session_watcher import _file_hash
    raw = jsonl.read_bytes().splitlines(keepends=True)
    mid = len(raw[0]) + len(raw[1])
    state = {rel: {"end_offset": mid, "hash": _file_hash(jsonl), "node_ids": [], "user_turns": 1}}

    captured: list[str] = []
    real_ingest = engine.ingest_conversation

    def capture(content, **kwargs):
        captured.append(content)
        return real_ingest(content=content, **kwargs)

    with patch(_LLM_PATCH, return_value=_LLM_RESPONSE), \
         patch.object(engine, "ingest_conversation", side_effect=capture):
        assert _ingest_session(engine, jsonl, state, watch_dir, min_turns=1) == IngestResult.OK

    committed = captured[-1]
    assert "Prompt with the memory detail" in committed   # prompt recovered
    assert "First part of the response" in committed       # both response records,
    assert "Second part with the actual answer" in committed  # paired with the prompt
    assert state[rel]["end_offset"] > mid

    # Recovery is one-time: the cursor is now a safe boundary (file fully consumed), so a
    # second pass on the unchanged file skips without re-recovering.
    assert state[rel]["end_offset"] == jsonl.stat().st_size
    with patch(_LLM_PATCH, return_value=_LLM_RESPONSE):
        assert _ingest_session(engine, jsonl, state, watch_dir, min_turns=1) == IngestResult.NO_PROGRESS


def test_codex_inflight_turn_not_split_on_idle(engine, tmp_path):
    """An in-flight Codex turn (no task_complete yet) is held back even when the file
    looks idle — there is no idle flush that could split it."""
    watch_dir = tmp_path / ".codex" / "sessions"
    transcript_dir = watch_dir / "2026" / "06" / "25"
    transcript_dir.mkdir(parents=True)
    jsonl = transcript_dir / "rollout-2026-06-25T12-30-00-sess-sticky.jsonl"
    rel = str(jsonl.relative_to(watch_dir))

    _append_codex_turn(jsonl, 0, records=2, complete=True)
    state = {}
    with patch(_LLM_PATCH, return_value=_LLM_RESPONSE):
        assert _ingest_session(engine, jsonl, state, watch_dir, min_turns=1) == IngestResult.OK
    cursor = state[rel]["end_offset"]

    # In-flight multi-record turn, file now idle. The turn has no closure signal, so it is
    # held back — never flushed mid-response.
    _append_codex_turn(jsonl, 1, records=2, complete=False)
    now = time.time()
    os.utime(jsonl, (now, now - 120))
    with patch(_LLM_PATCH, return_value=_LLM_RESPONSE):
        assert _ingest_session(engine, jsonl, state, watch_dir, min_turns=1,
                               idle_threshold=30) == IngestResult.NO_PROGRESS
    assert state[rel]["end_offset"] == cursor


def test_idle_tail_with_dangling_user_no_duplicate(engine, tmp_path):
    watch_dir = tmp_path / "projects"
    project_dir = watch_dir / "-Users-alice-Code-proj"
    project_dir.mkdir(parents=True)
    jsonl = project_dir / "active.jsonl"

    _make_jsonl(jsonl, user_turns=6)
    state = {}
    with patch(_LLM_PATCH, return_value=_LLM_RESPONSE):
        _ingest_session(engine, jsonl, state, watch_dir, min_turns=5)

    _append_pair(jsonl, 6)
    _append_pair(jsonl, 7)
    _append_user(jsonl, 8)
    now = time.time()
    os.utime(jsonl, (now, now - 120))

    captured = []
    real_ingest = engine.ingest_conversation

    def capture(content, **kwargs):
        captured.append(content)
        return real_ingest(content=content, **kwargs)

    with patch(_LLM_PATCH, return_value=_LLM_RESPONSE), \
         patch.object(engine, "ingest_conversation", side_effect=capture):
        assert _ingest_session(
            engine, jsonl, state, watch_dir, min_turns=5, idle_threshold=30
        ) == IngestResult.OK
        assert "User message 8 " not in captured[-1]

        _append_assistant(jsonl, 8)
        now2 = time.time()
        os.utime(jsonl, (now2, now2 - 120))
        assert _ingest_session(
            engine, jsonl, state, watch_dir, min_turns=1, idle_threshold=30
        ) == IngestResult.OK

    joined = "\n".join(captured)
    assert joined.count("User message 8 ") == 1


def test_session_tail_idle_ingested(engine, tmp_path):
    watch_dir = tmp_path / "projects"
    project_dir = watch_dir / "-Users-alice-Code-proj"
    project_dir.mkdir(parents=True)
    jsonl = project_dir / "active.jsonl"

    _make_jsonl(jsonl, user_turns=6)
    state = {}
    with patch(_LLM_PATCH, return_value=_LLM_RESPONSE):
        _ingest_session(engine, jsonl, state, watch_dir, min_turns=5)

    _append_pair(jsonl, 6)
    _append_pair(jsonl, 7)
    now = time.time()
    os.utime(jsonl, (now, now - 120))

    calls = 0
    real_ingest = engine.ingest_conversation

    def counting(content, **kwargs):
        nonlocal calls
        calls += 1
        return real_ingest(content=content, **kwargs)

    with patch(_LLM_PATCH, return_value=_LLM_RESPONSE), \
         patch.object(engine, "ingest_conversation", side_effect=counting):
        assert _ingest_session(
            engine, jsonl, state, watch_dir, min_turns=5, idle_threshold=30
        ) == IngestResult.OK
    assert calls == 1


def test_retry_fires_and_ingests_after_idle(engine, tmp_path):
    from ormah.background import session_watcher as sw

    watch_dir = tmp_path / "projects"
    project_dir = watch_dir / "-Users-alice-Code-proj"
    project_dir.mkdir(parents=True)
    jsonl = project_dir / "active.jsonl"

    _make_jsonl(jsonl, user_turns=6)

    captured_timers = []

    class FakeTimer:
        def __init__(self, delay, fn, args=()):
            self.delay = delay
            self.fn = fn
            self.args = args
            self.daemon = False
        def start(self):
            captured_timers.append(self)
        def cancel(self):
            pass

    calls = 0
    real_ingest = engine.ingest_conversation

    def counting(content, **kwargs):
        nonlocal calls
        calls += 1
        return real_ingest(content=content, **kwargs)

    # Seed state outside counting context so the initial 6-pair ingest is not counted.
    with patch(_LLM_PATCH, return_value=_LLM_RESPONSE), \
         patch.object(sw, "Timer", FakeTimer):
        handler = sw.SessionHandler(
            engine, watch_dir, debounce_seconds=60, min_turns=5, idle_threshold=30,
        )
        sw._ingest_session(engine, jsonl, handler._state, watch_dir, min_turns=5)

    _append_pair(jsonl, 6)
    _append_pair(jsonl, 7)

    with patch(_LLM_PATCH, return_value=_LLM_RESPONSE), \
         patch.object(engine, "ingest_conversation", side_effect=counting), \
         patch.object(sw, "Timer", FakeTimer):
        handler._do_ingest(jsonl)
        assert calls == 0
        assert len(captured_timers) == 1
        assert captured_timers[0].delay == 30

        now = time.time()
        os.utime(jsonl, (now, now - 120))
        timer = captured_timers[0]
        timer.fn(*timer.args)

    assert calls == 1


def test_concurrent_ingest_skipped(engine, tmp_path):
    import threading

    from ormah.background import session_watcher as sw

    watch_dir = tmp_path / "projects"
    project_dir = watch_dir / "-Users-alice-Code-proj"
    project_dir.mkdir(parents=True)
    jsonl = project_dir / "active.jsonl"
    _make_jsonl(jsonl, user_turns=6)

    started = threading.Event()
    release = threading.Event()
    calls = 0

    def blocking_ingest(content, **kwargs):
        nonlocal calls
        calls += 1
        started.set()
        release.wait(timeout=5)
        return []

    with patch(_LLM_PATCH, return_value=_LLM_RESPONSE), \
         patch.object(engine, "ingest_conversation", side_effect=blocking_ingest):
        handler = sw.SessionHandler(
            engine, watch_dir, debounce_seconds=60, min_turns=5, idle_threshold=30,
        )
        t1 = threading.Thread(target=handler._do_ingest, args=(jsonl,))
        t1.start()
        assert started.wait(timeout=5)
        handler._do_ingest(jsonl)
        release.set()
        t1.join(timeout=5)

    assert calls == 1


# --- ADR-0003 (#149): gate the rewind on forward progress ---


def test_api_error_orphan_advances_without_reingest(engine, tmp_path, caplog):
    """ADR-0003 regression (bug #149): an assistant 'API Error' record right after a
    terminal end_turn flags leading_orphan on the next tick. The watcher must NOT rewind
    to 0 (36x whole-file re-extractions); it drops the fragment, ingests the tail past
    the boundary, and the following tick is a cheap NO_PROGRESS."""
    watch_dir = tmp_path / "projects"
    project_dir = watch_dir / "-Users-alice-Code-myproject"
    project_dir.mkdir(parents=True)
    jsonl = project_dir / "abc123.jsonl"

    first_turn = [
        {"type": "user", "message": {"content": "Prompt one"}},
        {"type": "assistant", "message": {"stop_reason": "end_turn",
            "content": [{"type": "text", "text": "Answer one"}]}},
    ]
    tail = [
        {"type": "assistant", "message": {"stop_reason": "stop_sequence",
            "content": [{"type": "text",
                "text": "API Error: Connection closed mid-response."}]}},
        {"type": "user", "message": {"content": "continue with the previous response"}},
        {"type": "assistant", "message": {"stop_reason": "end_turn",
            "content": [{"type": "text",
                "text": "Answer two continues with additional detail"}]}},
    ]
    with open(jsonl, "w") as f:
        for line in first_turn:
            f.write(json.dumps(line) + "\n")
    boundary = parse_transcript(jsonl).safe_end_offset  # where tick N parked the cursor
    with open(jsonl, "a") as f:
        for line in tail:
            f.write(json.dumps(line) + "\n")
    _mark_idle(jsonl)

    rel = str(jsonl.relative_to(watch_dir))
    state = {rel: {"end_offset": boundary, "hash": "stale", "user_turns": 1, "node_ids": []}}

    with patch(_LLM_PATCH, return_value=_LLM_RESPONSE) as mock_llm, \
         caplog.at_level(logging.INFO, logger="ormah.background.session_watcher"):
        r1 = _ingest_session(engine, jsonl, state, watch_dir, 1)
        assert r1 == IngestResult.OK
        assert "recovering legacy mid-response cursor" not in caplog.text  # no rewind
        assert state[rel]["end_offset"] == jsonl.stat().st_size            # tail consumed
        assert state[rel]["end_offset"] > boundary                          # monotonic
        assert mock_llm.call_count == 1
        prompt = str(mock_llm.call_args_list[0])
        assert "Answer one" not in prompt   # slice before the cursor NOT re-ingested
        assert "API Error" not in prompt    # orphan fragment dropped, not committed
        assert "continue" in prompt         # previously-stranded tail IS ingested

        r2 = _ingest_session(engine, jsonl, state, watch_dir, 1)
        assert r2 == IngestResult.NO_PROGRESS   # second tick: nothing re-extracted
        assert mock_llm.call_count == 1
        assert state[rel]["end_offset"] == jsonl.stat().st_size


def test_no_progress_orphan_still_rewinds(engine, tmp_path, caplog):
    """A genuine legacy mid-response cursor (orphan AND no forward progress) still
    triggers the one-time whole-file recovery, re-pairing the tail with its prompt."""
    watch_dir = tmp_path / "projects"
    project_dir = watch_dir / "-Users-alice-Code-myproject"
    project_dir.mkdir(parents=True)
    jsonl = project_dir / "abc123.jsonl"
    records = [
        {"type": "user", "message": {"content": "Prompt about the architecture decision"}},
        {"type": "assistant", "message": {"stop_reason": "tool_use",
            "content": [{"type": "text", "text": "First part"}]}},
        {"type": "assistant", "message": {"stop_reason": "end_turn",
            "content": [{"type": "text", "text": "Second part"}]}},
    ]
    with open(jsonl, "w") as f:
        for line in records:
            f.write(json.dumps(line) + "\n")
    raw = jsonl.read_bytes().splitlines(keepends=True)
    mid = len(raw[0]) + len(raw[1])  # cursor parked mid-response by an older version
    _mark_idle(jsonl)

    rel = str(jsonl.relative_to(watch_dir))
    state = {rel: {"end_offset": mid, "hash": "stale", "user_turns": 1, "node_ids": []}}

    with patch(_LLM_PATCH, return_value=_LLM_RESPONSE) as mock_llm, \
         caplog.at_level(logging.INFO, logger="ormah.background.session_watcher"):
        r1 = _ingest_session(engine, jsonl, state, watch_dir, 1)
    assert r1 == IngestResult.OK
    assert "recovering legacy mid-response cursor" in caplog.text
    prompt = str(mock_llm.call_args_list[0])
    assert "Prompt about the architecture decision" in prompt  # re-paired from offset 0
    assert state[rel]["end_offset"] == jsonl.stat().st_size


def test_below_min_turns_orphan_reparse_is_cheap_noop(engine, tmp_path, caplog):
    """ADR-0003 residual: with the guard, an advanced-but-below-min_turns payload on an
    ACTIVE file defers (TRANSIENT) and re-parses on later ticks as a parse-only no-op —
    no rewind, no LLM call, no duplication — until it idles or crosses min_turns."""
    watch_dir = tmp_path / "projects"
    project_dir = watch_dir / "-Users-alice-Code-myproject"
    project_dir.mkdir(parents=True)
    jsonl = project_dir / "abc123.jsonl"

    first_turn = [
        {"type": "user", "message": {"content": "Prompt one"}},
        {"type": "assistant", "message": {"stop_reason": "end_turn",
            "content": [{"type": "text", "text": "Answer one"}]}},
    ]
    tail = [
        {"type": "assistant", "message": {"stop_reason": "stop_sequence",
            "content": [{"type": "text",
                "text": "API Error: Connection closed mid-response."}]}},
        {"type": "user", "message": {"content": "continue"}},
        {"type": "assistant", "message": {"stop_reason": "end_turn",
            "content": [{"type": "text", "text": "Answer two"}]}},
    ]
    with open(jsonl, "w") as f:
        for line in first_turn:
            f.write(json.dumps(line) + "\n")
    boundary = parse_transcript(jsonl).safe_end_offset
    with open(jsonl, "a") as f:
        for line in tail:
            f.write(json.dumps(line) + "\n")
    # NO _mark_idle: mtime is fresh, so the file is ACTIVE and 1 turn < min_turns=5 defers.

    rel = str(jsonl.relative_to(watch_dir))
    state = {rel: {"end_offset": boundary, "hash": "stale", "user_turns": 1, "node_ids": []}}

    with patch(_LLM_PATCH, return_value=_LLM_RESPONSE) as mock_llm, \
         caplog.at_level(logging.INFO, logger="ormah.background.session_watcher"):
        r1 = _ingest_session(engine, jsonl, state, watch_dir, 5)
        r2 = _ingest_session(engine, jsonl, state, watch_dir, 5)
    assert r1 == IngestResult.TRANSIENT and r2 == IngestResult.TRANSIENT  # defer, retry later
    assert "recovering legacy mid-response cursor" not in caplog.text     # never rewinds
    assert mock_llm.call_count == 0                                       # parse-only no-op
    assert state[rel]["end_offset"] == boundary                           # cursor held, not lost


def test_legacy_orphan_with_later_turns_advances_and_drops(engine, tmp_path, caplog):
    """ADR-0003 accepted-loss pinning (watcher level): a genuine legacy mid-response cursor
    in a file that ALSO has later closed turns → no rewind, the fragment tail is dropped
    (bounded, one-time loss), the later turn is ingested, cursor reaches EOF."""
    watch_dir = tmp_path / "projects"
    project_dir = watch_dir / "-Users-alice-Code-myproject"
    project_dir.mkdir(parents=True)
    jsonl = project_dir / "abc123.jsonl"
    records = [
        {"type": "user", "message": {"content": "Prompt one"}},
        {"type": "assistant", "message": {"stop_reason": "tool_use",
            "content": [{"type": "text", "text": "First part"}]}},
        {"type": "assistant", "message": {"stop_reason": "end_turn",
            "content": [{"type": "text", "text": "Second part"}]}},
        {"type": "user", "message": {"content": "Prompt two continues the architecture discussion"}},
        {"type": "assistant", "message": {"stop_reason": "end_turn",
            "content": [{"type": "text", "text": "Answer two follows up with more detail"}]}},
    ]
    with open(jsonl, "w") as f:
        for line in records:
            f.write(json.dumps(line) + "\n")
    raw = jsonl.read_bytes().splitlines(keepends=True)
    mid = len(raw[0]) + len(raw[1])  # legacy cursor parked mid-response
    _mark_idle(jsonl)

    rel = str(jsonl.relative_to(watch_dir))
    state = {rel: {"end_offset": mid, "hash": "stale", "user_turns": 1, "node_ids": []}}

    with patch(_LLM_PATCH, return_value=_LLM_RESPONSE) as mock_llm, \
         caplog.at_level(logging.INFO, logger="ormah.background.session_watcher"):
        r1 = _ingest_session(engine, jsonl, state, watch_dir, 1)
    assert r1 == IngestResult.OK
    assert "recovering legacy mid-response cursor" not in caplog.text  # ADR: no rewind
    assert state[rel]["end_offset"] == jsonl.stat().st_size
    prompt = str(mock_llm.call_args_list[0])
    assert "Second part" not in prompt   # the accepted, bounded loss
    assert "Prompt one" not in prompt    # pre-cursor content not re-ingested
    assert "Prompt two" in prompt        # later turn ingested normally


def test_inflight_orphan_rewind_parks_without_reingest(engine, tmp_path, caplog):
    """ADR-0003 critical regression (Codex review, #149): an orphan with NO forward
    progress whose rewind (full re-parse) ALSO makes no progress — because the tail is a
    still-open (in-flight) response, not a genuinely recoverable one — must park
    (NO_PROGRESS) rather than re-extract the closed prefix on every tick."""
    watch_dir = tmp_path / "projects"
    project_dir = watch_dir / "-Users-alice-Code-myproject"
    project_dir.mkdir(parents=True)
    jsonl = project_dir / "abc123.jsonl"

    closed_turn = [
        {"type": "user", "message": {"content": "Prompt about the release plan"}},
        {"type": "assistant", "message": {"stop_reason": "end_turn",
            "content": [{"type": "text", "text": "Answer about the release plan"}]}},
    ]
    with open(jsonl, "w") as f:
        for line in closed_turn:
            f.write(json.dumps(line) + "\n")
    boundary = parse_transcript(jsonl).safe_end_offset  # cursor parked here by tick N

    # An in-flight response fragment: text-bearing, non-terminal stop_reason, no
    # following user turn and no closure — the response is genuinely still being written.
    with open(jsonl, "a") as f:
        f.write(json.dumps({"type": "assistant", "message": {"stop_reason": "tool_use",
            "content": [{"type": "text", "text": "In-flight fragment"}]}}) + "\n")
    _mark_idle(jsonl)

    rel = str(jsonl.relative_to(watch_dir))
    state = {rel: {"end_offset": boundary, "hash": "stale", "user_turns": 1, "node_ids": []}}

    with patch(_LLM_PATCH, return_value=_LLM_RESPONSE) as mock_llm, \
         caplog.at_level(logging.INFO, logger="ormah.background.session_watcher"):
        r1 = _ingest_session(engine, jsonl, state, watch_dir, 1)
        r2 = _ingest_session(engine, jsonl, state, watch_dir, 1)

    assert r1 == IngestResult.NO_PROGRESS
    assert r2 == IngestResult.NO_PROGRESS
    assert mock_llm.call_count == 0                       # never re-ingested
    assert state[rel]["end_offset"] == boundary            # cursor left untouched


# --- Test 19: in-flight skip reschedules the dropped event (no lost tail) ---

def test_inflight_skip_reschedules(engine, tmp_path):
    """A modify event skipped because an ingest is in flight must be rescheduled, not dropped."""
    import threading

    from ormah.background import session_watcher as sw

    watch_dir = tmp_path / "projects"
    project_dir = watch_dir / "-Users-alice-Code-proj"
    project_dir.mkdir(parents=True)
    jsonl = project_dir / "active.jsonl"
    _make_jsonl(jsonl, user_turns=6)

    scheduled = []

    class FakeTimer:
        def __init__(self, delay, fn, args=()):
            self.delay = delay
            self.fn = fn
            self.args = args
            self.daemon = False
        def start(self):
            scheduled.append(self)
        def cancel(self):
            pass

    started = threading.Event()
    release = threading.Event()

    def blocking_ingest(content, **kwargs):
        started.set()
        release.wait(timeout=5)
        return []

    with patch(_LLM_PATCH, return_value=_LLM_RESPONSE), \
         patch.object(engine, "ingest_conversation", side_effect=blocking_ingest), \
         patch.object(sw, "Timer", FakeTimer):
        handler = sw.SessionHandler(
            engine, watch_dir, debounce_seconds=60, min_turns=5, idle_threshold=30,
        )
        t1 = threading.Thread(target=handler._do_ingest, args=(jsonl,))
        t1.start()
        assert started.wait(timeout=5)   # ingest A is in flight
        handler._do_ingest(jsonl)        # skipped — must mark pending
        assert scheduled == []           # nothing rescheduled while A still runs
        release.set()
        t1.join(timeout=5)

    # After A finishes, the skipped event was rescheduled as a fresh debounce
    assert len(scheduled) == 1
    assert scheduled[0].delay == 60


# --- Test 20: shrink resets node_ids provenance, not just turn count ---

def test_shrink_resets_node_ids(engine, tmp_path):
    """A file that shrinks below the stored offset must not carry stale node_ids forward."""
    watch_dir = tmp_path / "projects"
    project_dir = watch_dir / "-Users-alice-Code-proj"
    project_dir.mkdir(parents=True)
    jsonl = project_dir / "active.jsonl"
    rel = str(jsonl.relative_to(watch_dir))

    _make_jsonl(jsonl, user_turns=10)
    state = {}
    with patch(_LLM_PATCH, return_value=_LLM_RESPONSE):
        assert _ingest_session(engine, jsonl, state, watch_dir, min_turns=5) == IngestResult.OK
    first_nodes = list(state[rel]["node_ids"])
    assert first_nodes  # first ingest produced at least one node

    _make_jsonl(jsonl, user_turns=5)  # smaller file → size < stored end_offset → full re-ingest
    with patch(_LLM_PATCH, return_value=_LLM_RESPONSE):
        assert _ingest_session(engine, jsonl, state, watch_dir, min_turns=5) == IngestResult.OK

    # Full re-ingest (prev_offset reset to 0): stale node_ids must not be concatenated,
    # so the stored provenance carries no duplicates.
    nodes = state[rel]["node_ids"]
    assert len(nodes) == len(set(nodes))


def test_do_ingest_returns_ok_when_it_ingests(engine, tmp_path):
    """_do_ingest reports IngestResult so reconcile can count recoveries and triage failures."""
    watch_dir = tmp_path / "projects"
    project_dir = watch_dir / "-Users-alice-Code-myproject"
    project_dir.mkdir(parents=True)
    jsonl = project_dir / "abc123.jsonl"
    _make_jsonl(jsonl, user_turns=6)
    _mark_idle(jsonl)

    handler = SessionHandler(engine, watch_dir, 60.0, 5, 30.0, 9999)
    with patch(_LLM_PATCH, return_value=_LLM_RESPONSE):
        assert handler._do_ingest(jsonl) == IngestResult.OK
        assert handler._do_ingest(jsonl) == IngestResult.NO_PROGRESS  # nothing new the second time


def test_reconcile_ingests_file_the_live_path_missed(engine, tmp_path):
    """A changed, idle transcript whose fsevent never reached the handler is recovered."""
    watch_dir = tmp_path / "projects"
    project_dir = watch_dir / "-Users-alice-Code-myproject"
    project_dir.mkdir(parents=True)
    jsonl = project_dir / "abc123.jsonl"
    _make_jsonl(jsonl, user_turns=6)
    _mark_idle(jsonl)

    handler = SessionHandler(engine, watch_dir, 60.0, 5, 30.0, 9999)
    rel = str(jsonl.relative_to(watch_dir))
    assert rel not in handler._state  # simulate the dropped event: handler never saw it

    with patch(_LLM_PATCH, return_value=_LLM_RESPONSE):
        recovered = handler.reconcile()

    assert recovered == 1
    assert rel in handler._state
    assert handler._state[rel]["user_turns"] == 6


def test_reconcile_skips_fully_consumed_file_on_second_pass(engine, tmp_path):
    """A second reconcile does not re-ingest a file already consumed to EOF (cheap skip)."""
    watch_dir = tmp_path / "projects"
    project_dir = watch_dir / "-Users-alice-Code-myproject"
    project_dir.mkdir(parents=True)
    jsonl = project_dir / "abc123.jsonl"
    _make_jsonl(jsonl, user_turns=6)
    _mark_idle(jsonl)

    handler = SessionHandler(engine, watch_dir, 60.0, 5, 30.0, 9999)
    with patch(_LLM_PATCH, return_value=_LLM_RESPONSE):
        assert handler.reconcile() == 1
        assert handler.reconcile() == 0


def test_reconcile_does_not_reingest_what_live_path_already_took(engine, tmp_path):
    """reconcile shares handler state, so a file ingested live is not re-ingested."""
    watch_dir = tmp_path / "projects"
    project_dir = watch_dir / "-Users-alice-Code-myproject"
    project_dir.mkdir(parents=True)
    jsonl = project_dir / "abc123.jsonl"
    _make_jsonl(jsonl, user_turns=6)
    _mark_idle(jsonl)

    handler = SessionHandler(engine, watch_dir, 60.0, 5, 30.0, 9999)
    with patch(_LLM_PATCH, return_value=_LLM_RESPONSE):
        handler._do_ingest(jsonl)                      # live path ingests it
        rel = str(jsonl.relative_to(watch_dir))
        node_count = len(handler._state[rel]["node_ids"])
        recovered = handler.reconcile()

    assert recovered == 0
    assert len(handler._state[rel]["node_ids"]) == node_count


def test_reconcile_logs_recovery_heartbeat(engine, tmp_path, caplog):
    """reconcile emits the functional heartbeat when it recovers >0 transcripts."""
    watch_dir = tmp_path / "projects"
    project_dir = watch_dir / "-Users-alice-Code-myproject"
    project_dir.mkdir(parents=True)
    jsonl = project_dir / "abc123.jsonl"
    _make_jsonl(jsonl, user_turns=6)
    _mark_idle(jsonl)

    handler = SessionHandler(engine, watch_dir, 60.0, 5, 30.0, 9999)
    with patch(_LLM_PATCH, return_value=_LLM_RESPONSE), \
         caplog.at_level("INFO", logger="ormah.background.session_watcher"):
        handler.reconcile()
    assert any("reconcile recovered" in r.message for r in caplog.records)


# --- Adversarial regressions for the two HIGH council findings ---

def test_reconcile_retries_seen_file_when_first_do_ingest_fails(engine, tmp_path):
    """A transient ingest failure must NOT strand a seen file: the next tick retries it."""
    watch_dir = tmp_path / "projects"
    project_dir = watch_dir / "-Users-alice-Code-myproject"
    project_dir.mkdir(parents=True)
    jsonl = project_dir / "abc123.jsonl"
    _make_jsonl(jsonl, user_turns=6)
    _mark_idle(jsonl)

    handler = SessionHandler(engine, watch_dir, 60.0, 5, 30.0, 9999)
    # Seed state as a seen file with a pending tail (cursor behind EOF).
    rel = str(jsonl.relative_to(watch_dir))
    handler._state[rel] = {"hash": "stale", "end_offset": 0, "node_ids": [], "user_turns": 0}

    calls = {"n": 0}
    real = _ingest_session

    def flaky(*a, **k):
        calls["n"] += 1
        if calls["n"] == 1:
            return IngestResult.TRANSIENT     # transient failure on the first reconcile
        return real(*a, **k)

    with patch(_LLM_PATCH, return_value=_LLM_RESPONSE), \
            patch("ormah.background.session_watcher._ingest_session", side_effect=flaky):
        assert handler.reconcile() == 0       # first tick: ingest "fails"
        assert handler.reconcile() == 1       # second tick retries (not skipped) and recovers


def test_reconcile_recovers_partial_tail_without_mtime_change(engine, tmp_path):
    """A grown tail with an UNCHANGED mtime is still recovered (cursor != size, not mtime)."""
    watch_dir = tmp_path / "projects"
    project_dir = watch_dir / "-Users-alice-Code-myproject"
    project_dir.mkdir(parents=True)
    jsonl = project_dir / "abc123.jsonl"
    _make_jsonl(jsonl, user_turns=6)
    _mark_idle(jsonl)

    handler = SessionHandler(engine, watch_dir, 60.0, 5, 30.0, 9999)
    with patch(_LLM_PATCH, return_value=_LLM_RESPONSE):
        assert handler.reconcile() == 1               # consumes the first 6 turns
        old_mtime = jsonl.stat().st_mtime
        _make_jsonl(jsonl, user_turns=12)             # append 6 more (size grows)
        os.utime(jsonl, (old_mtime, old_mtime))       # mtime unchanged on purpose
        recovered = handler.reconcile()

    assert recovered == 1                             # picked up via end_offset != size


def test_reconcile_while_live_ingesting_defers_then_retries(engine, tmp_path):
    """If the live path owns the path mid-ingest, reconcile defers, then retries next tick."""
    watch_dir = tmp_path / "projects"
    project_dir = watch_dir / "-Users-alice-Code-myproject"
    project_dir.mkdir(parents=True)
    jsonl = project_dir / "abc123.jsonl"
    _make_jsonl(jsonl, user_turns=6)
    _mark_idle(jsonl)

    handler = SessionHandler(engine, watch_dir, 60.0, 5, 30.0, 9999)
    handler._ingesting.add(str(jsonl))                # simulate live path mid-ingest

    with patch(_LLM_PATCH, return_value=_LLM_RESPONSE):
        assert handler.reconcile() == 0               # deferred: live path owns it

    handler._ingesting.discard(str(jsonl))            # live path finished without ingesting
    handler._pending.discard(str(jsonl))
    with patch(_LLM_PATCH, return_value=_LLM_RESPONSE):
        assert handler.reconcile() == 1               # not poisoned -> retried and recovered


def test_reconcile_bounds_retries_for_abandoned_inflight_tail(engine, tmp_path):
    """A seen tail that never converges (always no-op) is retried a bounded number of
    times, not re-attempted (re-hashed) every tick forever."""
    from ormah.background.session_watcher import MAX_RECONCILE_RETRIES

    watch_dir = tmp_path / "projects"
    project_dir = watch_dir / "-Users-alice-Code-myproject"
    project_dir.mkdir(parents=True)
    jsonl = project_dir / "abc123.jsonl"
    _make_jsonl(jsonl, user_turns=6)
    _mark_idle(jsonl)

    handler = SessionHandler(engine, watch_dir, 60.0, 5, 30.0, 9999)
    rel = str(jsonl.relative_to(watch_dir))
    # Seen file stuck below EOF that never makes progress (abandoned in-flight tail).
    handler._state[rel] = {"hash": "x", "end_offset": 1, "node_ids": [], "user_turns": 0}

    calls = {"n": 0}

    def noop(path):
        calls["n"] += 1
        return IngestResult.NO_PROGRESS  # never makes progress (size + safe boundary frozen)

    handler._do_ingest = noop
    for _ in range(8):
        handler.reconcile()

    assert calls["n"] == MAX_RECONCILE_RETRIES  # bounded, not 8


def test_run_session_reconcile_recreates_dead_observer(engine, tmp_path):
    """A dead Observer is stopped/joined and recreated; reconcile still runs."""
    from ormah.background.session_watcher import SessionWatch, run_session_reconcile

    watch_dir = tmp_path / "projects"
    watch_dir.mkdir(parents=True)
    handler = SessionHandler(engine, watch_dir, 60.0, 5, 30.0, 9999)

    dead = MagicMock()
    dead.is_alive.return_value = False
    watch = SessionWatch(watch_dir=watch_dir, handler=handler, observer=dead)

    with patch("ormah.background.session_watcher.Observer") as MockObserver:
        new_obs = MockObserver.return_value
        total = run_session_reconcile([watch])

    dead.stop.assert_called_once()        # old observer cleaned up before recreate
    dead.join.assert_called_once()
    new_obs.schedule.assert_called_once()
    new_obs.start.assert_called_once()
    assert watch.observer is new_obs
    assert total == 0  # empty dir, nothing to recover


def test_run_session_reconcile_skips_stopping_handler(engine, tmp_path):
    """A shutdown-overlapped reconcile tick must not recreate observers or ingest."""
    from ormah.background.session_watcher import SessionWatch, run_session_reconcile

    watch_dir = tmp_path / "projects"
    watch_dir.mkdir(parents=True)
    handler = SessionHandler(engine, watch_dir, 60.0, 5, 30.0, 9999)
    handler.request_stop()

    dead = MagicMock()
    dead.is_alive.return_value = False
    watch = SessionWatch(watch_dir=watch_dir, handler=handler, observer=dead)

    with patch("ormah.background.session_watcher.Observer") as MockObserver:
        total = run_session_reconcile([watch])

    MockObserver.assert_not_called()
    dead.stop.assert_not_called()
    assert total == 0


def test_run_session_reconcile_runs_reconcile_even_when_recreate_fails(engine, tmp_path):
    """If recreating a dead Observer raises, the reconcile scan still runs (safety-net guarantee)."""
    from ormah.background.session_watcher import SessionWatch, run_session_reconcile

    watch_dir = tmp_path / "projects"
    watch_dir.mkdir(parents=True)
    handler = SessionHandler(engine, watch_dir, 60.0, 5, 30.0, 9999)
    handler.reconcile = MagicMock(return_value=0)

    dead = MagicMock()
    dead.is_alive.return_value = False
    watch = SessionWatch(watch_dir=watch_dir, handler=handler, observer=dead)

    with patch("ormah.background.session_watcher.Observer", side_effect=RuntimeError("boom")):
        total = run_session_reconcile([watch])

    handler.reconcile.assert_called_once()  # safety net ran despite recreate failure
    assert total == 0


def test_reconcile_does_not_starve_valid_file_behind_stuck_never_seen_files(engine, tmp_path):
    """>cap never-seen files that never ingest must not starve a later valid transcript:
    they get parked after MAX_RECONCILE_RETRIES, freeing the per-tick budget for the valid one."""
    from ormah.background.session_watcher import MAX_RECONCILE_RETRIES

    watch_dir = tmp_path / "projects"
    project_dir = watch_dir / "-Users-alice-Code-myproject"
    project_dir.mkdir(parents=True)

    cap = engine.settings.session_watcher_reconcile_max_per_tick
    for i in range(cap):                              # sort BEFORE 'zz-valid' below
        p = project_dir / f"00stuck-{i:03d}.jsonl"
        p.write_text("not a valid transcript line\n")  # ingests nothing -> stays never-seen
        _mark_idle(p)

    valid = project_dir / "zz-valid.jsonl"            # sorts AFTER all stuck files
    _make_jsonl(valid, user_turns=6)
    _mark_idle(valid)
    rel_valid = str(valid.relative_to(watch_dir))

    handler = SessionHandler(engine, watch_dir, 60.0, 5, 30.0, 9999)
    with patch(_LLM_PATCH, return_value=_LLM_RESPONSE):
        for _ in range(MAX_RECONCILE_RETRIES + 2):    # let the stuck files exhaust their budget
            handler.reconcile()
            if rel_valid in handler._state:
                break

    assert rel_valid in handler._state                # reached, not starved


def test_reconcile_never_parks_transient_failures(engine, tmp_path):
    """A TRANSIENT _do_ingest result must never increment _reconcile_attempts — the file
    is retried every tick indefinitely, never parked (unlike NO_PROGRESS which parks after
    MAX_RECONCILE_RETRIES attempts at the same file size)."""
    watch_dir = tmp_path / "projects"
    project_dir = watch_dir / "-Users-alice-Code-myproject"
    project_dir.mkdir(parents=True)
    jsonl = project_dir / "abc123.jsonl"
    _make_jsonl(jsonl, user_turns=6)
    _mark_idle(jsonl)

    handler = SessionHandler(engine, watch_dir, 60.0, 5, 30.0, 9999)
    rel = str(jsonl.relative_to(watch_dir))
    # Seed state: seen file with a pending tail (cursor behind EOF).
    handler._state[rel] = {"hash": "stale", "end_offset": 0, "node_ids": [], "user_turns": 0}

    ingest_calls = {"n": 0}

    def always_transient(path):
        ingest_calls["n"] += 1
        return IngestResult.TRANSIENT

    handler._do_ingest = always_transient
    for _ in range(6):
        handler.reconcile()

    # Must have been attempted every single tick — never parked.
    assert ingest_calls["n"] == 6
    # And _reconcile_attempts must not have accumulated a count for this file.
    assert handler._reconcile_attempts.get(rel) is None


# --- Council-PR H1/H2: change-token park key + TRANSIENT deprioritization ---

def test_reconcile_unparks_after_same_size_content_change(engine, tmp_path):
    """H1: a same-byte-size content rewrite (new mtime) un-parks a NO_PROGRESS file.

    If a parked file's content is repaired but byte length is unchanged, the mtime_ns
    changes. The new (size, mtime_ns) token differs from the parked token, so the file
    is un-parked and _do_ingest is called again on the next tick.
    """
    from ormah.background.session_watcher import MAX_RECONCILE_RETRIES

    watch_dir = tmp_path / "projects"
    project_dir = watch_dir / "-Users-alice-Code-myproject"
    project_dir.mkdir(parents=True)
    jsonl = project_dir / "abc123.jsonl"
    _make_jsonl(jsonl, user_turns=6)
    _mark_idle(jsonl)

    handler = SessionHandler(engine, watch_dir, 60.0, 5, 30.0, 9999)
    rel = str(jsonl.relative_to(watch_dir))
    # Seed state: seen file with a pending tail (cursor behind EOF).
    handler._state[rel] = {"hash": "stale", "end_offset": 0, "node_ids": [], "user_turns": 0}

    ingest_calls = {"n": 0}

    def always_no_progress(path):
        ingest_calls["n"] += 1
        return IngestResult.NO_PROGRESS

    handler._do_ingest = always_no_progress

    # Drive reconcile MAX_RECONCILE_RETRIES times to park the file.
    for _ in range(MAX_RECONCILE_RETRIES):
        handler.reconcile()
    assert ingest_calls["n"] == MAX_RECONCILE_RETRIES

    # File is now parked: further reconcile calls must NOT call _do_ingest.
    handler.reconcile()
    assert ingest_calls["n"] == MAX_RECONCILE_RETRIES  # count did not increase → parked

    # Rewrite with identical byte size but a bumped mtime (content changed, size unchanged).
    original_size = jsonl.stat().st_size
    content = jsonl.read_bytes()
    jsonl.write_bytes(content)  # same bytes = same size; write bumps mtime_ns
    assert jsonl.stat().st_size == original_size  # size unchanged — regression guard

    # Now reconcile must call _do_ingest again (token changed → un-parked).
    handler.reconcile()
    assert ingest_calls["n"] == MAX_RECONCILE_RETRIES + 1


def test_reconcile_deprioritizes_persistent_transient_behind_valid(engine, tmp_path):
    """H2: files that keep returning TRANSIENT are deprioritized, not starved-out, so an
    older valid file is still ingested within a bounded number of ticks.

    Setup: cap=2 newest files → always TRANSIENT; 1 older file → returns OK.
    After TRANSIENT files cross MAX_RECONCILE_RETRIES ticks at their token, they sort
    behind the valid file and the valid file is ingested.
    """
    from ormah.background.session_watcher import MAX_RECONCILE_RETRIES

    watch_dir = tmp_path / "projects"
    project_dir = watch_dir / "-Users-alice-Code-myproject"
    project_dir.mkdir(parents=True)

    cap = 2
    engine.settings.session_watcher_reconcile_max_per_tick = cap

    now = time.time()

    # Two newest TRANSIENT files.
    transient_files = []
    for i in range(cap):
        p = project_dir / f"new-{i:03d}.jsonl"
        _make_jsonl(p, user_turns=6)
        mtime = now - i  # newest first (decreasing by 1s)
        os.utime(p, (mtime, mtime))
        transient_files.append(p)

    # One older valid file.
    valid = project_dir / "old-valid.jsonl"
    _make_jsonl(valid, user_turns=6)
    old_mtime = now - 1000  # clearly older
    os.utime(valid, (old_mtime, old_mtime))

    # Seed all as seen with a pending tail.
    handler = SessionHandler(engine, watch_dir, 60.0, 5, 30.0, 9999)
    for p in transient_files + [valid]:
        rel = str(p.relative_to(watch_dir))
        handler._state[rel] = {"hash": "x", "end_offset": 0, "node_ids": [], "user_turns": 0}

    transient_paths = {str(p) for p in transient_files}
    valid_path = str(valid)
    ingested_paths: list[str] = []

    def selective_ingest(path):
        ingested_paths.append(str(path))
        if str(path) in transient_paths:
            return IngestResult.TRANSIENT
        return IngestResult.OK

    handler._do_ingest = selective_ingest

    # Run enough ticks for TRANSIENT files to cross MAX_RECONCILE_RETRIES at their token
    # and become deprioritized, then for the valid file to be picked up.
    max_ticks = MAX_RECONCILE_RETRIES + 3
    for _ in range(max_ticks):
        handler.reconcile()
        if valid_path in ingested_paths:
            break

    assert valid_path in ingested_paths, (
        f"Valid file was never ingested after {max_ticks} ticks — it was starved. "
        f"ingested_paths={ingested_paths}"
    )


def test_reconcile_deprioritized_transients_retried_oldest_first(engine, tmp_path):
    """H2': among already-deprioritized TRANSIENT files, the OLDEST is retried first (FIFO).

    Non-deprioritized candidates sort newest-first (fresh drops recover soonest), but within the
    deprioritized group oldest-first avoids a long-failing transient that just became recoverable
    being starved behind newer deprioritized peers.
    """
    from ormah.background.session_watcher import MAX_RECONCILE_RETRIES

    watch_dir = tmp_path / "projects"
    project_dir = watch_dir / "-Users-alice-Code-myproject"
    project_dir.mkdir(parents=True)
    now = time.time()

    older = project_dir / "older.jsonl"
    newer = project_dir / "newer.jsonl"
    _make_jsonl(older, user_turns=6)
    _make_jsonl(newer, user_turns=6)
    os.utime(older, (now - 1000, now - 1000))
    os.utime(newer, (now - 1, now - 1))

    handler = SessionHandler(engine, watch_dir, 60.0, 5, 30.0, 9999)
    handler.engine.settings.session_watcher_reconcile_max_per_tick = 1  # one slot per tick

    # Seed both as seen-pending AND already deprioritized at their current token.
    for p in (older, newer):
        rel = str(p.relative_to(watch_dir))
        st = p.stat()
        handler._state[rel] = {"hash": "x", "end_offset": 0, "node_ids": [], "user_turns": 0}
        handler._reconcile_transient[rel] = (st.st_size, st.st_mtime_ns, MAX_RECONCILE_RETRIES)

    calls: list[str] = []

    def record_ingest(path):
        calls.append(str(path))
        return IngestResult.TRANSIENT

    handler._do_ingest = record_ingest
    handler.reconcile()  # cap=1 -> only the first-sorted deprioritized candidate runs

    assert calls == [str(older)], (
        f"Oldest deprioritized file should be retried first (FIFO), got {calls}"
    )


# --- Council-PR F2/F3: per-tick time budget + lookback<0 never-seen guard ---

def test_reconcile_respects_per_tick_time_budget(engine, tmp_path):
    """The per-tick wall-clock budget causes an early break, so not all candidates are processed
    in one reconcile() call even when count < max_per_tick."""
    watch_dir = tmp_path / "projects"
    project_dir = watch_dir / "-Users-alice-Code-myproject"
    project_dir.mkdir(parents=True)

    # Three seen files with a pending tail (end_offset=0, size>0) — all are reconcile candidates.
    files = []
    for i in range(3):
        p = project_dir / f"session-{i:02d}.jsonl"
        _make_jsonl(p, user_turns=6)
        _mark_idle(p)
        files.append(p)

    handler = SessionHandler(engine, watch_dir, 60.0, 5, 30.0, 9999)
    handler.engine.settings.session_watcher_reconcile_max_seconds = 30.0

    # Seed all files as seen-but-pending (cursor at 0, size > 0).
    for p in files:
        rel = str(p.relative_to(watch_dir))
        handler._state[rel] = {"hash": "stale", "end_offset": 0, "node_ids": [], "user_turns": 0}

    ingest_calls = []

    def counting_ingest(path):
        ingest_calls.append(path)
        return IngestResult.OK

    handler._do_ingest = counting_ingest

    # Tie the clock to real progress instead of an exact call count: time stays at 0.0
    # until the first file is ingested, then jumps past the 30s budget so the next
    # loop-check breaks. Robust to any extra time.time() calls reconcile() may add.
    def fake_time():
        return 9999.0 if ingest_calls else 0.0

    with patch("ormah.background.session_watcher.time.time", side_effect=fake_time):
        handler.reconcile()

    # Budget broke after 1 — fewer than all 3 candidates were processed.
    assert len(ingest_calls) == 1


def test_reconcile_skips_never_seen_when_lookback_negative(engine, tmp_path):
    """When lookback_hours < 0 (catch-up disabled), never-seen files must be skipped in
    reconcile() — mirroring the _scan_sessions rule."""
    watch_dir = tmp_path / "projects"
    project_dir = watch_dir / "-Users-alice-Code-myproject"
    project_dir.mkdir(parents=True)
    jsonl = project_dir / "abc123.jsonl"
    _make_jsonl(jsonl, user_turns=6)
    _mark_idle(jsonl)

    # lookback_hours=-1 means no catch-up: never-seen files must be skipped.
    handler = SessionHandler(engine, watch_dir, 60.0, 5, 30.0, lookback_hours=-1)
    rel = str(jsonl.relative_to(watch_dir))
    assert rel not in handler._state  # never seen

    ingest_calls = []

    def counting_ingest(path):
        ingest_calls.append(path)
        return IngestResult.OK

    handler._do_ingest = counting_ingest
    recovered = handler.reconcile()

    assert recovered == 0
    assert ingest_calls == []
    assert rel not in handler._state


# --- Issue #220: confirmed use from the auto_llm_judge path -----------------

_LIFECYCLE_FIELDS = ("access_count", "last_accessed", "stability", "last_review")


def _lifecycle(engine, node_id):
    """The four lifecycle fields, from the markdown file and the SQLite row."""
    node = engine.file_store.load(node_id)
    row = engine.db.conn.execute(
        "SELECT access_count, last_accessed, stability, last_review FROM nodes WHERE id = ?",
        (node_id,),
    ).fetchone()
    return {
        "file": tuple(getattr(node, f) for f in _LIFECYCLE_FIELDS),
        "db": tuple(row[f] for f in _LIFECYCLE_FIELDS),
    }


def test_llm_judge_used_verdict_records_confirmed_use(engine, tmp_path):
    """Issue #220: a positive auto_llm_judge verdict confirms use for its node."""
    prompt = "What deployment marker should we use?"
    response = "That guidance is the right one for the rollout."
    transcript_path = tmp_path / "judge-confirms-session.jsonl"
    _write_turn_jsonl(transcript_path, prompt, response)
    transcript = parse_transcript(transcript_path)

    node_id, _ = engine.remember(CreateNodeRequest(
        content="Use blue deployment markers when rollback plans need quick visual checks.",
        type="fact",
        title="Blue deployment rollback marker",
    ))
    whisper_log_id = _insert_injected_whisper_log(
        engine, node_id=node_id, session_id="judge-confirms-session", prompt=prompt,
    )
    engine.settings.llm_provider = "ollama"
    engine.settings.feedback_llm_judge_enabled = True

    before = _lifecycle(engine, node_id)

    llm_response = json.dumps({
        "verdicts": [{
            "whisper_log_id": whisper_log_id,
            "verdict": "used",
            "confidence": 0.88,
            "reason": "The answer endorses the injected deployment guidance.",
        }]
    })
    with patch(_LLM_PATCH, return_value=llm_response):
        _record_whisper_usage_signals(engine, transcript)

    after = _lifecycle(engine, node_id)
    assert after != before, "the judged-used node was not confirmed"
    assert after["file"][0] == before["file"][0] + 1, "access_count did not advance by one"
    assert after["db"][0] == after["file"][0], "file and DB disagree on access_count"

    # The signal and affinity rows must still be written — confirmed use is
    # additional behaviour, not a replacement for observability.
    affinity = engine.db.conn.execute(
        "SELECT * FROM affinity WHERE whisper_log_id = ?", (whisper_log_id,)
    ).fetchone()
    assert affinity is not None
    assert affinity["source"] == "auto_llm_judge"


def test_llm_judge_unused_verdict_does_not_record_confirmed_use(engine, tmp_path):
    """A negative verdict is affinity evidence only — it never reinforces."""
    prompt = "What deployment marker should we use?"
    response = "Ignore that; we are switching to a completely different scheme."
    transcript_path = tmp_path / "judge-unused-session.jsonl"
    _write_turn_jsonl(transcript_path, prompt, response)
    transcript = parse_transcript(transcript_path)

    node_id, _ = engine.remember(CreateNodeRequest(
        content="Use blue deployment markers when rollback plans need quick visual checks.",
        type="fact",
        title="Blue deployment rollback marker",
    ))
    whisper_log_id = _insert_injected_whisper_log(
        engine, node_id=node_id, session_id="judge-unused-session", prompt=prompt,
    )
    engine.settings.llm_provider = "ollama"
    engine.settings.feedback_llm_judge_enabled = True

    before = _lifecycle(engine, node_id)

    llm_response = json.dumps({
        "verdicts": [{
            "whisper_log_id": whisper_log_id,
            "verdict": "unused",
            "confidence": 0.9,
            "reason": "The answer rejects the injected guidance.",
        }]
    })
    with patch(_LLM_PATCH, return_value=llm_response):
        _record_whisper_usage_signals(engine, transcript)

    assert _lifecycle(engine, node_id) == before, "an unused verdict changed lifecycle fields"


@pytest.mark.parametrize("title,content,response,should_confirm", [
    # title match (0.94) — the title appears verbatim in the response
    (
        "Transcript watcher mines feedback usage",
        "The transcript watcher mines feedback usage from completed transcripts.",
        "The right fix is the transcript watcher mines feedback usage approach.",
        True,
    ),
    # sentence match (0.92) — a content sentence appears verbatim
    (
        "Vector search notes",
        "Sqlite vec stores embeddings inside the same database file as the nodes.",
        "As noted: sqlite vec stores embeddings inside the same database file as the nodes.",
        True,
    ),
])
def test_verbatim_heuristic_match_confirms_use(
    engine, tmp_path, title, content, response, should_confirm,
):
    """#272: a verbatim heuristic hit reinforces the memory. Contract 12, inverted.

    This is the issue's acceptance criterion: 0 of 1,629 positive heuristic pairs
    took a claim, because only the judge block ever called _claim_confirmed_use.
    """
    prompt = "How should we solve feedback collection?"
    transcript_path = tmp_path / "verbatim-confirm-session.jsonl"
    _write_turn_jsonl(transcript_path, prompt, response)
    transcript = parse_transcript(transcript_path)

    node_id, _ = engine.remember(CreateNodeRequest(content=content, type="fact", title=title))
    whisper_log_id = _insert_injected_whisper_log(
        engine, node_id=node_id, session_id="verbatim-confirm-session", prompt=prompt,
    )

    before = _lifecycle(engine, node_id)
    recorded = _record_whisper_usage_signals(engine, transcript)

    assert recorded == 1
    signal = engine.db.conn.execute(
        "SELECT polarity, strength FROM signals WHERE whisper_log_id = ?", (whisper_log_id,)
    ).fetchone()
    assert signal["polarity"] == 1
    assert signal["strength"] >= 0.80, "fixture did not produce a verbatim match — check the text"

    claim = engine.db.conn.execute(
        "SELECT 1 FROM confirmed_use_claims WHERE whisper_log_id = ? AND node_id = ?",
        (whisper_log_id, node_id),
    ).fetchone()
    assert claim is not None, "the heuristic path took no confirmed-use claim"
    assert _lifecycle(engine, node_id) != before, "the claim was taken but nothing reinforced"


def test_node_id_heuristic_match_confirms_use(engine, tmp_path):
    """#272 spec case 1: the strongest match kind (0.98).

    Separate from the parametrized test above because the response must quote the
    node's short id, which only exists after the node is created.
    """
    prompt = "Which memory covers the retention policy?"
    node_id, _ = engine.remember(CreateNodeRequest(
        content="Retention is governed by decay and archival thresholds.",
        type="fact",
        title="Retention policy overview",
    ))
    response = f"That is memory {node_id[:8]}, which covers it."

    transcript_path = tmp_path / "nodeid-confirm-session.jsonl"
    _write_turn_jsonl(transcript_path, prompt, response)
    transcript = parse_transcript(transcript_path)

    whisper_log_id = _insert_injected_whisper_log(
        engine, node_id=node_id, session_id="nodeid-confirm-session", prompt=prompt,
    )

    before = _lifecycle(engine, node_id)
    _record_whisper_usage_signals(engine, transcript)

    signal = engine.db.conn.execute(
        "SELECT strength, evidence FROM signals WHERE whisper_log_id = ?", (whisper_log_id,)
    ).fetchone()
    assert json.loads(signal["evidence"])["match"] == "node_id"
    assert signal["strength"] == signal_strength.VERBATIM_NODE_ID

    claim = engine.db.conn.execute(
        "SELECT 1 FROM confirmed_use_claims WHERE whisper_log_id = ? AND node_id = ?",
        (whisper_log_id, node_id),
    ).fetchone()
    assert claim is not None, "a node_id match — the strongest evidence there is — did not claim"
    assert _lifecycle(engine, node_id) != before


def test_token_overlap_heuristic_match_does_not_confirm(engine, tmp_path):
    """#272 D1: the weak channel records evidence but never reinforces.

    97.4% of heuristic hits are token_overlap; admitting them would give the least
    precise kind the same lifecycle power as a verbatim node_id match.
    """
    prompt = "What about the retention policy?"
    # Overlapping vocabulary, but no verbatim title or sentence.
    response = (
        "The decay process lowers stability, and archival thresholds eventually "
        "move things along."
    )
    transcript_path = tmp_path / "overlap-no-confirm-session.jsonl"
    _write_turn_jsonl(transcript_path, prompt, response)
    transcript = parse_transcript(transcript_path)

    node_id, _ = engine.remember(CreateNodeRequest(
        content="Decay lowers stability until archival thresholds move a node out of working.",
        type="fact",
        title="Decay stability archival thresholds",
    ))
    whisper_log_id = _insert_injected_whisper_log(
        engine, node_id=node_id, session_id="overlap-no-confirm-session", prompt=prompt,
    )

    before = _lifecycle(engine, node_id)
    with patch(_LLM_PATCH, return_value=None):  # judge unavailable — isolate the heuristic
        _record_whisper_usage_signals(engine, transcript)

    signal = engine.db.conn.execute(
        "SELECT polarity, strength, evidence FROM signals WHERE whisper_log_id = ?",
        (whisper_log_id,),
    ).fetchone()
    assert signal["polarity"] == 1, "fixture did not match at all — check the vocabulary overlap"
    assert json.loads(signal["evidence"])["match"] == "token_overlap"
    assert signal["strength"] < 0.80

    claim = engine.db.conn.execute(
        "SELECT 1 FROM confirmed_use_claims WHERE whisper_log_id = ?", (whisper_log_id,)
    ).fetchone()
    assert claim is None, "token_overlap took a claim — it is below the evidence floor"
    assert _lifecycle(engine, node_id) == before


def test_one_nodes_reinforcement_failure_does_not_stop_the_batch(engine, tmp_path):
    """#272: the batch is isolated per node, matching the judge path's contract."""
    prompt = "How should we solve feedback collection?"
    response = (
        "Two things: the transcript watcher mines feedback usage approach, "
        "and sqlite vec stores embeddings inside the same database file as the nodes."
    )
    transcript_path = tmp_path / "batch-failure-session.jsonl"
    _write_turn_jsonl(transcript_path, prompt, response)
    transcript = parse_transcript(transcript_path)

    first, _ = engine.remember(CreateNodeRequest(
        content="The transcript watcher mines feedback usage from completed transcripts.",
        type="fact", title="Transcript watcher mines feedback usage",
    ))
    second, _ = engine.remember(CreateNodeRequest(
        content="Sqlite vec stores embeddings inside the same database file as the nodes.",
        type="fact", title="Vector search notes",
    ))
    for node_id in (first, second):
        _insert_injected_whisper_log(
            engine, node_id=node_id, session_id="batch-failure-session", prompt=prompt,
        )

    before_second = _lifecycle(engine, second)
    real = engine._record_confirmed_use

    def flaky(node_id, *, whisper_log_id):
        if node_id == first:
            raise ZeroDivisionError("simulated mutator failure")
        return real(node_id, whisper_log_id=whisper_log_id)

    with patch.object(engine, "_record_confirmed_use", side_effect=flaky):
        recorded = _record_whisper_usage_signals(engine, transcript)

    assert recorded == 2, "a mutator failure changed the recorded count"
    assert _lifecycle(engine, second) != before_second, "node 2 lost its reinforcement"


def test_heuristic_below_the_floor_does_not_record_confirmed_use(engine, tmp_path):
    """Contract 12, as amended by #272: the floor, not the source, is the gate.

    Before #272 no heuristic hit could confirm. Now a verbatim one does, and only
    evidence below HEURISTIC_CONFIRM_FLOOR is kept out. The verbatim half of this
    contract lives in test_verbatim_heuristic_match_confirms_use.
    """
    prompt = "What about the retention policy?"
    response = (
        "The decay process lowers stability, and archival thresholds eventually "
        "move things along."
    )
    transcript_path = tmp_path / "contract12-session.jsonl"
    _write_turn_jsonl(transcript_path, prompt, response)
    transcript = parse_transcript(transcript_path)

    node_id, _ = engine.remember(CreateNodeRequest(
        content="Decay lowers stability until archival thresholds move a node out of working.",
        type="fact",
        title="Decay stability archival thresholds",
    ))
    whisper_log_id = _insert_injected_whisper_log(
        engine, node_id=node_id, session_id="contract12-session", prompt=prompt,
    )

    before = _lifecycle(engine, node_id)
    with patch(_LLM_PATCH, return_value=None):
        recorded = _record_whisper_usage_signals(engine, transcript)

    assert recorded == 1
    signal = engine.db.conn.execute(
        "SELECT polarity, strength FROM signals WHERE whisper_log_id = ?", (whisper_log_id,)
    ).fetchone()
    assert signal["polarity"] == 1, "the signal is still recorded — this is lifecycle, not observability"
    assert signal["strength"] < 0.80

    assert _lifecycle(engine, node_id) == before, "a below-floor hit confirmed use — it must not"
    claim = engine.db.conn.execute(
        "SELECT 1 FROM confirmed_use_claims WHERE whisper_log_id = ?", (whisper_log_id,)
    ).fetchone()
    assert claim is None, "a below-floor hit took a confirmed-use claim"


def test_replaying_the_judge_does_not_reconfirm(engine, tmp_path):
    """Issue #220: a second pass over the same transcript reinforces nothing.

    has_llm_judge already excludes an event that was judged before, so the
    replay should not even reach the confirm loop — and the claim latch stops it
    a second time if it does. Two independent guards, deliberately.
    """
    prompt = "What deployment marker should we use?"
    response = "That guidance is the right one for the rollout."
    transcript_path = tmp_path / "judge-replay-session.jsonl"
    _write_turn_jsonl(transcript_path, prompt, response)
    transcript = parse_transcript(transcript_path)

    node_id, _ = engine.remember(CreateNodeRequest(
        content="Use blue deployment markers when rollback plans need quick visual checks.",
        type="fact",
        title="Blue deployment rollback marker",
    ))
    whisper_log_id = _insert_injected_whisper_log(
        engine, node_id=node_id, session_id="judge-replay-session", prompt=prompt,
    )
    engine.settings.llm_provider = "ollama"
    engine.settings.feedback_llm_judge_enabled = True

    llm_response = json.dumps({
        "verdicts": [{
            "whisper_log_id": whisper_log_id,
            "verdict": "used",
            "confidence": 0.88,
            "reason": "The answer endorses the injected deployment guidance.",
        }]
    })
    with patch(_LLM_PATCH, return_value=llm_response):
        _record_whisper_usage_signals(engine, transcript)
    after_first = _lifecycle(engine, node_id)

    with patch(_LLM_PATCH, return_value=llm_response):
        _record_whisper_usage_signals(engine, transcript)

    assert _lifecycle(engine, node_id) == after_first, (
        "replaying the judge reinforced the same event twice"
    )


def test_feedback_claim_makes_the_judge_a_noop(engine, tmp_path):
    """Issue #220 cross-caller contract: one event, one reinforcement, two callers.

    This is the case has_llm_judge cannot cover: it only looks at signals whose
    source is transcript_watcher_llm_judge, so it is blind to feedback submitted
    through MCP. Before the claim latch, an implicit +1 followed by a positive
    judge verdict on the same whisper event reinforced it twice.
    """
    prompt = "What deployment marker should we use?"
    response = "That guidance is the right one for the rollout."
    transcript_path = tmp_path / "judge-cross-session.jsonl"
    _write_turn_jsonl(transcript_path, prompt, response)
    transcript = parse_transcript(transcript_path)

    node_id, _ = engine.remember(CreateNodeRequest(
        content="Use blue deployment markers when rollback plans need quick visual checks.",
        type="fact",
        title="Blue deployment rollback marker",
    ))
    whisper_log_id = _insert_injected_whisper_log(
        engine, node_id=node_id, session_id="judge-cross-session", prompt=prompt,
    )
    engine.settings.llm_provider = "ollama"
    engine.settings.feedback_llm_judge_enabled = True

    engine.submit_feedback(node_id, signal=1, source="implicit", whisper_log_id=whisper_log_id)
    after_feedback = _lifecycle(engine, node_id)

    llm_response = json.dumps({
        "verdicts": [{
            "whisper_log_id": whisper_log_id,
            "verdict": "used",
            "confidence": 0.88,
            "reason": "The answer endorses the injected deployment guidance.",
        }]
    })
    with patch(_LLM_PATCH, return_value=llm_response):
        recorded = _record_whisper_usage_signals(engine, transcript)

    # The judge's own signal and affinity rows are still written — observability
    # is not what the claim gates.
    assert recorded >= 1, "the judge signal was not recorded"
    assert _lifecycle(engine, node_id) == after_feedback, (
        "the judge reinforced an event already confirmed through submit_feedback"
    )


def test_one_failing_node_does_not_skip_the_rest_of_the_batch(engine, tmp_path):
    """Issue #220: reinforcement is isolated per node, for any exception.

    The judge signals and the claims are already committed when this loop runs,
    so an escaping exception would abort the slice and — because has_llm_judge is
    now set and the claims are taken — the retry would never reinforce these
    events. The later nodes would lose their only chance at confirmation.

    ZeroDivisionError is the realistic case, not a contrived one: stability is
    Field(default=1.0, ge=0.0), so zero is legal, and the mutator divides by it.
    """
    prompt = "What deployment marker should we use?"
    response = "Both of those notes are exactly right."
    transcript_path = tmp_path / "judge-batch-session.jsonl"
    _write_turn_jsonl(transcript_path, prompt, response)
    transcript = parse_transcript(transcript_path)

    first_id, _ = engine.remember(CreateNodeRequest(
        content="Use blue deployment markers when rollback plans need quick visual checks.",
        type="fact", title="Blue deployment rollback marker",
    ))
    second_id, _ = engine.remember(CreateNodeRequest(
        content="Roll back within one minute when the marker check fails.",
        type="fact", title="Rollback timing",
    ))
    log_ids = [
        _insert_injected_whisper_log(
            engine, node_id=node_id, session_id="judge-batch-session", prompt=prompt,
        )
        for node_id in (first_id, second_id)
    ]
    engine.settings.llm_provider = "ollama"
    engine.settings.feedback_llm_judge_enabled = True

    before_second = _lifecycle(engine, second_id)

    real_mutator = engine._record_confirmed_use

    def failing_for_first(node_id, *, whisper_log_id):
        if node_id == first_id:
            raise ZeroDivisionError("float division by zero")
        return real_mutator(node_id, whisper_log_id=whisper_log_id)

    llm_response = json.dumps({
        "verdicts": [
            {"whisper_log_id": log_id, "verdict": "used", "confidence": 0.9,
             "reason": "endorsed"}
            for log_id in log_ids
        ]
    })
    with patch(_LLM_PATCH, return_value=llm_response), \
         patch.object(engine, "_record_confirmed_use", side_effect=failing_for_first):
        recorded = _record_whisper_usage_signals(engine, transcript)

    # Both nodes go through the heuristic pass unreferenced (the response text
    # matches neither node's id/title/content), then both go to the judge pass:
    # 2 heuristic signals + 2 judge signals.
    assert recorded == 4, "the signals themselves must still be recorded"
    assert _lifecycle(engine, second_id) != before_second, (
        "the first node's failure skipped the second node's reinforcement"
    )


def test_a_weak_heuristic_hit_still_reaches_the_judge(engine, tmp_path):
    """#272 D3: below the floor, the judge is the only route left — do not suppress it.

    Before #272 any heuristic hit suppressed the judge, so a token_overlap match
    could neither claim nor be judged. That is 1,587 of the 1,629 measured rows.
    """
    prompt = "What about the retention policy?"
    # MEASURED by executing _node_usage_evidence, not reasoned about: this text gives
    # match="token_overlap", overlap_ratio 0.6, strength 0.436 — a REAL weak hit, below
    # the 0.80 floor. The previous text ("Retention uses decay, stability and archival
    # thresholds together.") gave overlap_ratio 0.4, under OVERLAP_GATE 0.5, so the
    # match was "none": no hit, polarity 0, and NO affinity row written at all.
    response = (
        "The decay process lowers stability, and archival thresholds eventually "
        "move things along."
    )
    transcript_path = tmp_path / "weak-to-judge-session.jsonl"
    _write_turn_jsonl(transcript_path, prompt, response)
    transcript = parse_transcript(transcript_path)

    node_id, _ = engine.remember(CreateNodeRequest(
        content="Decay lowers stability until archival thresholds move a node out of working.",
        type="fact",
        title="Decay stability archival thresholds",
    ))
    whisper_log_id = _insert_injected_whisper_log(
        engine, node_id=node_id, session_id="weak-to-judge-session", prompt=prompt,
    )
    engine.settings.llm_provider = "ollama"
    engine.settings.feedback_llm_judge_enabled = True

    llm_response = json.dumps({"verdicts": [{
        "whisper_log_id": whisper_log_id,
        "verdict": "used",
        "confidence": 0.95,
        "reason": "The answer applies the retention guidance.",
    }]})
    before = _lifecycle(engine, node_id)
    with patch(_LLM_PATCH, return_value=llm_response) as mock_llm:
        _record_whisper_usage_signals(engine, transcript)

    assert mock_llm.called, "a weak heuristic hit suppressed the judge"
    claim = engine.db.conn.execute(
        "SELECT 1 FROM confirmed_use_claims WHERE whisper_log_id = ?", (whisper_log_id,)
    ).fetchone()
    assert claim is not None, "the judge confirmed nothing for a weak heuristic hit"
    assert _lifecycle(engine, node_id) != before


def test_an_irrelevant_verdict_overrides_the_weak_heuristic_affinity(engine, tmp_path):
    """#272, council (Codex HIGH): the judge outranks the heuristic for the same event.

    This task is what makes the conflict reachable: a token_overlap hit now gets both
    an affinity +1 from the heuristic block AND a trip to the judge. affinity has a
    unique (node_id, whisper_log_id) index and _insert_affinity is ON CONFLICT DO
    NOTHING, so without Step 5 the judge's -1 is silently dropped and retrieval keeps
    consuming a +1 the judge just rejected.

    Red before Step 5 on the final row's polarity, not on the signal: the signals table
    records the negative verdict either way. The affinity row is the falsifier.
    """
    prompt = "What about the retention policy?"
    # MEASURED by executing _node_usage_evidence, not reasoned about: this text gives
    # match="token_overlap", overlap_ratio 0.6, strength 0.436 — a REAL weak hit, below
    # the 0.80 floor. The previous text ("Retention uses decay, stability and archival
    # thresholds together.") gave overlap_ratio 0.4, under OVERLAP_GATE 0.5, so the
    # match was "none": no hit, polarity 0, and NO affinity row written at all.
    response = (
        "The decay process lowers stability, and archival thresholds eventually "
        "move things along."
    )
    transcript_path = tmp_path / "irrelevant-override-session.jsonl"
    _write_turn_jsonl(transcript_path, prompt, response)
    transcript = parse_transcript(transcript_path)

    node_id, _ = engine.remember(CreateNodeRequest(
        content="Decay lowers stability until archival thresholds move a node out of working.",
        type="fact",
        title="Decay stability archival thresholds",
    ))
    whisper_log_id = _insert_injected_whisper_log(
        engine, node_id=node_id, session_id="irrelevant-override-session", prompt=prompt,
    )
    engine.settings.llm_provider = "ollama"
    engine.settings.feedback_llm_judge_enabled = True

    llm_response = json.dumps({"verdicts": [{
        "whisper_log_id": whisper_log_id,
        "verdict": "irrelevant",
        "confidence": 0.95,
        "reason": "The answer never uses the injected memory.",
    }]})
    with patch(_LLM_PATCH, return_value=llm_response) as mock_llm:
        _record_whisper_usage_signals(engine, transcript)

    assert mock_llm.called, "the weak hit never reached the judge — check Step 4"

    affinity = engine.db.conn.execute(
        "SELECT signal, source FROM affinity WHERE node_id = ? AND whisper_log_id = ?",
        (node_id, whisper_log_id),
    ).fetchall()
    assert len(affinity) == 1, "the unique index should keep exactly one row per event"
    assert affinity[0]["signal"] == -1, (
        "the heuristic's +1 survived an irrelevant verdict — retrieval will keep boosting "
        "a memory the judge rejected"
    )
    assert affinity[0]["source"] == "auto_llm_judge"

    claim = engine.db.conn.execute(
        "SELECT 1 FROM confirmed_use_claims WHERE whisper_log_id = ?", (whisper_log_id,)
    ).fetchone()
    assert claim is None, "a negative verdict took a confirmed-use claim"


def test_explicit_feedback_outranks_a_later_judge_verdict(engine, tmp_path):
    """#272: precedence is explicit > auto_llm_judge > auto_heuristic, not last-write-wins.

    Step 5's UPDATE is scoped to source = 'auto_heuristic' precisely so a human's
    explicit feedback is never overwritten by an automated verdict. Drop that WHERE
    clause and this goes red.

    The feedback is NEGATIVE, and that is load-bearing — not a stylistic choice.
    An earlier draft used signal=1 and could not fail: a positive explicit feedback
    takes the _claim_confirmed_use latch synchronously inside _submit_feedback_locked
    (memory_engine.py:2842-2849), so Step 3's already_confirmed reads True, Step 4's
    `settled` is True, the judge is never queued, the patched LLM is never called and
    Step 5's UPDATE never runs at all. The assertion then passed on Task 2's
    ON CONFLICT DO NOTHING alone, with the scoping clause deleted or intact.

    signal=-1 is the path that reaches Step 5: _claim_confirmed_use returns False for
    any signal != 1, so the affinity row is written as 'explicit' while NO claim is
    taken. already_confirmed is False, `confirms` is False (the response only
    token-overlaps, whose band supremum 0.78 sits under the 0.80 floor), so the event
    is unsettled, the judge runs, and its UPDATE fires against a row whose source is
    'explicit'. Only the `AND source = 'auto_heuristic'` clause leaves it standing.

    It is also the real scenario: a human marked a memory as NOT useful, which does
    not settle the event, and the judge's later verdict must not overwrite the
    attribution back to itself.
    """
    prompt = "What about the retention policy?"
    # MEASURED by executing _node_usage_evidence, not reasoned about: this text gives
    # match="token_overlap", overlap_ratio 0.6, strength 0.436 — a REAL weak hit, below
    # the 0.80 floor. The previous text ("Retention uses decay, stability and archival
    # thresholds together.") gave overlap_ratio 0.4, under OVERLAP_GATE 0.5, so the
    # match was "none": no hit, polarity 0, and NO affinity row written at all.
    response = (
        "The decay process lowers stability, and archival thresholds eventually "
        "move things along."
    )
    transcript_path = tmp_path / "explicit-outranks-session.jsonl"
    _write_turn_jsonl(transcript_path, prompt, response)
    transcript = parse_transcript(transcript_path)

    node_id, _ = engine.remember(CreateNodeRequest(
        content="Decay lowers stability until archival thresholds move a node out of working.",
        type="fact",
        title="Decay stability archival thresholds",
    ))
    whisper_log_id = _insert_injected_whisper_log(
        engine, node_id=node_id, session_id="explicit-outranks-session", prompt=prompt,
    )
    engine.settings.llm_provider = "ollama"
    engine.settings.feedback_llm_judge_enabled = True

    # A human says this memory was NOT useful, through MCP. This writes the affinity
    # row as 'explicit' WITHOUT taking the claim (signal != 1), which is what leaves
    # the event unsettled so the judge below actually runs. See the docstring.
    engine.submit_feedback(node_id, signal=-1, source="explicit", whisper_log_id=whisper_log_id)

    llm_response = json.dumps({"verdicts": [{
        "whisper_log_id": whisper_log_id,
        "verdict": "irrelevant",
        "confidence": 0.95,
        "reason": "The answer never uses the injected memory.",
    }]})
    with patch(_LLM_PATCH, return_value=llm_response) as mock_judge:
        _record_whisper_usage_signals(engine, transcript)

    affinity = engine.db.conn.execute(
        "SELECT signal, source FROM affinity WHERE node_id = ? AND whisper_log_id = ?",
        (node_id, whisper_log_id),
    ).fetchone()
    assert affinity["source"] == "explicit", "an automated verdict overwrote explicit feedback"
    assert affinity["signal"] == -1, "the human's negative signal was replaced"

    # The guard on the guard: if the judge never ran, the two assertions above hold
    # trivially and prove nothing about the scoping clause. This is what the earlier
    # signal=1 draft failed silently.
    assert mock_judge.called, (
        "the judge never ran, so Step 5's UPDATE never executed and this test cannot "
        "distinguish a scoped UPDATE from an unscoped one"
    )


def test_a_confirming_heuristic_hit_does_not_reach_the_judge(engine, tmp_path):
    """#272 D3: judging an event that already confirmed is wasted spend.

    Council (Cursor, MEDIUM) on the final plan: asserting only `not mock_llm.called`
    is not enough. The plan itself notes this already passes on the old `referenced`
    rule, so on its own it pins nothing about #272. Worse, an implementation that
    calls _claim_confirmed_use only when `not llm_judge_enabled` would keep every
    Task 2 test green (the judge is off by default there) AND this one green — while
    in production, with the judge armed, a verbatim hit would be neither claimed nor
    judged. The claim and lifecycle assertions below are what falsify that.
    """
    prompt = "How should we solve feedback collection?"
    response = "The right fix is the transcript watcher mines feedback usage approach."
    transcript_path = tmp_path / "strong-skips-judge-session.jsonl"
    _write_turn_jsonl(transcript_path, prompt, response)
    transcript = parse_transcript(transcript_path)

    node_id, _ = engine.remember(CreateNodeRequest(
        content="The transcript watcher mines feedback usage from completed transcripts.",
        type="fact",
        title="Transcript watcher mines feedback usage",
    ))
    whisper_log_id = _insert_injected_whisper_log(
        engine, node_id=node_id, session_id="strong-skips-judge-session", prompt=prompt,
    )
    # The judge is ENABLED here on purpose: with it off, "not called" would be
    # vacuously true and the test would pass against any suppression rule at all.
    engine.settings.llm_provider = "ollama"
    engine.settings.feedback_llm_judge_enabled = True

    before = _lifecycle(engine, node_id)
    with patch(_LLM_PATCH) as mock_llm:
        _record_whisper_usage_signals(engine, transcript)

    assert not mock_llm.called, "a confirming heuristic hit was judged anyway"

    # The judge is ARMED here. These two are what stop a claim-only-when-judge-disabled
    # implementation from shipping green.
    claim = engine.db.conn.execute(
        "SELECT 1 FROM confirmed_use_claims WHERE whisper_log_id = ? AND node_id = ?",
        (whisper_log_id, node_id),
    ).fetchone()
    assert claim is not None, (
        "a verbatim hit took no claim while the judge was enabled — it is now neither "
        "confirmed nor judged"
    )
    assert _lifecycle(engine, node_id) != before, "the claim was taken but nothing reinforced"


def test_an_already_confirmed_event_is_not_rejudged_on_reingest(engine, tmp_path):
    """#272 D3: on RE-INGEST the claim table is the only authority left.

    Council R3 (Cursor, MEDIUM) rejected the first version of this test: it used a
    verbatim response, so the base's old rule suppressed the judge on BOTH passes
    (`referenced` on the first, `heuristic_polarity == 1` on the second) and the test
    passed unchanged. It proved nothing about `already_confirmed`.

    The response is unreferenced instead, and the claim comes from MCP feedback. On
    the second pass `has_heuristic` is true with polarity 0, so the base computes
    `referenced = False` and QUEUES the judge — red today. Only `already_confirmed`
    can suppress it, which is exactly the predicate under test.
    """
    prompt = "How should we solve feedback collection?"
    response = "I don't know."
    transcript_path = tmp_path / "reingest-session.jsonl"
    _write_turn_jsonl(transcript_path, prompt, response)
    transcript = parse_transcript(transcript_path)

    node_id, _ = engine.remember(CreateNodeRequest(
        content="The transcript watcher mines feedback usage from completed transcripts.",
        type="fact",
        title="Transcript watcher mines feedback usage",
    ))
    whisper_log_id = _insert_injected_whisper_log(
        engine, node_id=node_id, session_id="reingest-session", prompt=prompt,
    )
    engine.settings.llm_provider = "ollama"
    engine.settings.feedback_llm_judge_enabled = True

    # The claim arrives through MCP, the one caller has_llm_judge cannot see.
    engine.submit_feedback(node_id, signal=1, source="implicit", whisper_log_id=whisper_log_id)

    # First pass writes the polarity-0 heuristic row that makes the next pass a re-ingest.
    with patch(_LLM_PATCH) as first_llm:
        _record_whisper_usage_signals(engine, transcript)
    assert not first_llm.called, "the judge ran on an event MCP had already confirmed"

    after_first = _lifecycle(engine, node_id)

    # Second pass: has_heuristic is now true, polarity 0. Only the claim can settle it.
    with patch(_LLM_PATCH) as second_llm:
        _record_whisper_usage_signals(engine, transcript)

    assert not second_llm.called, "a settled event was sent to the judge on re-ingest"
    assert _lifecycle(engine, node_id) == after_first, "the event reinforced twice"


def test_mcp_feedback_suppresses_the_judge_for_that_event(engine, tmp_path):
    """#272 D3: closes the cross-caller blindness has_llm_judge cannot see (#220 13a).

    The response is deliberately UNREFERENCED. Council R2 (Cursor, MEDIUM) caught the
    earlier fixture: it overlapped the node, so `referenced` was already true and the
    base's `not referenced` suppressed the judge on its own — the test passed before
    and after, proving nothing. With no textual reference, the base queues the judge
    (red today) and only `already_confirmed` can suppress it (green after).
    """
    prompt = "What about the retention policy?"
    response = "I don't know."
    transcript_path = tmp_path / "mcp-first-session.jsonl"
    _write_turn_jsonl(transcript_path, prompt, response)
    transcript = parse_transcript(transcript_path)

    node_id, _ = engine.remember(CreateNodeRequest(
        content="Decay lowers stability until archival thresholds move a node out of working.",
        type="fact",
        title="Decay stability archival thresholds",
    ))
    whisper_log_id = _insert_injected_whisper_log(
        engine, node_id=node_id, session_id="mcp-first-session", prompt=prompt,
    )
    engine.settings.llm_provider = "ollama"
    engine.settings.feedback_llm_judge_enabled = True

    engine.submit_feedback(node_id, signal=1, source="implicit", whisper_log_id=whisper_log_id)
    after_feedback = _lifecycle(engine, node_id)

    with patch(_LLM_PATCH) as mock_llm:
        _record_whisper_usage_signals(engine, transcript)

    assert not mock_llm.called, "an event already confirmed through MCP was judged again"
    assert _lifecycle(engine, node_id) == after_feedback, "the event reinforced twice"

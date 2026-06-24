"""Tests for the transcript watcher — auto-ingestion of agent transcripts."""

from __future__ import annotations

import json
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from ormah.background.session_watcher import (
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
            "message": {"role": "assistant", "content": [
                {"type": "text", "text": f"Assistant response {i} with some detail"},
            ]},
        }))
    path.write_text("\n".join(lines) + "\n")


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

    assert result is True
    rel = str(jsonl.relative_to(watch_dir))
    assert rel in state
    entry = state[rel]
    assert entry["session_id"] == "abc123"
    assert entry["source"] == "claude_code"
    assert entry["space"] == "myproject"
    assert entry["user_turns"] == 6
    assert len(entry["node_ids"]) == 1


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
        assert _ingest_session(engine, jsonl, state, watch_dir, min_turns=1) is True

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

    state = {}
    with patch(_LLM_PATCH, return_value=_LLM_RESPONSE):
        assert _ingest_session(engine, jsonl, state, watch_dir, min_turns=1) is True

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
    assert judge_signal["strength"] == 0.88

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

    assert result is False
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
        assert _ingest_session(engine, jsonl, state, watch_dir, min_turns=5) is True
        assert _ingest_session(engine, jsonl, state, watch_dir, min_turns=5) is False


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
        assert observers[0].is_alive()
    finally:
        stop_session_watcher(observers)

    # Give observer thread a moment to stop
    time.sleep(0.1)
    assert not observers[0].is_alive()


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
        assert all(observer.is_alive() for observer in observers)
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

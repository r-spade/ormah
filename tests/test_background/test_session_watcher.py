"""Tests for the transcript watcher — auto-ingestion of agent transcripts."""

from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from ormah.background.session_watcher import (
    MAX_EXTRACT_FAILURES,
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

_LLM_PATCH = "ormah.background.llm_client.ingest_llm_generate"
# A slice-specific extraction failure: the LLM responds but the content is unparseable, so
# _extract_memories_llm raises during json.loads and returns its generic error string. This is the
# DETERMINISTIC failure that counts toward the per-slice cap (unlike a provider-wide call failure /
# None, which is transient and never skips the slice — council-pr H1).
_UNPARSEABLE = "this is not json at all"
# The whisper-usage LLM judge uses the global llm_generate (maintenance path), NOT the
# extraction-only ingest_llm_generate. Judge tests patch this; ingest tests patch _LLM_PATCH.
_JUDGE_PATCH = "ormah.background.llm_client.llm_generate"

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

    Recedes past the default session_watcher_idle_threshold (600s, see _ingest_session)
    so callers relying on either that default or a smaller explicit idle_threshold see the
    file as idle.
    """
    now = time.time()
    os.utime(path, (now, now - 700))


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
    _mark_idle(jsonl)  # finished session, below flush_bytes → idle flush

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


def test_ingest_none_is_transient_and_does_not_advance(engine, tmp_path):
    """LLM unavailable (adapter returns None) -> TRANSIENT, cursor must not advance."""
    watch_dir = tmp_path / "projects"
    project_dir = watch_dir / "-Users-alice-Code-myproject"
    project_dir.mkdir(parents=True)
    jsonl = project_dir / "abc123.jsonl"
    _make_jsonl(jsonl, user_turns=6)
    _mark_idle(jsonl)

    state = {}
    # "LLM unavailable" == no provider configured: the failure stays TRANSIENT and is never
    # counted toward the per-slice cap, so no state entry is written. Patch the provider check
    # explicitly so the result does not depend on a cached ingest adapter left by an earlier test.
    with patch(_LLM_PATCH, return_value=None), \
         patch("ormah.background.session_watcher.ingest_provider_configured", return_value=False):
        result = _ingest_session(engine, jsonl, state, watch_dir, min_turns=5)

    assert result == IngestResult.TRANSIENT
    rel = str(jsonl.relative_to(watch_dir))
    assert rel not in state  # no provider -> failure never counted, cursor never written


def test_toxic_slice_skipped_after_max_extract_failures(engine, tmp_path):
    """A slice that fails extraction MAX_EXTRACT_FAILURES times (provider present) must advance
    the cursor past it — not re-drive ingestion forever (the 1393x loop)."""
    watch_dir = tmp_path / "projects"
    project_dir = watch_dir / "-Users-alice-Code-myproject"
    project_dir.mkdir(parents=True)
    jsonl = project_dir / "abc123.jsonl"
    _make_jsonl(jsonl, user_turns=6)
    _mark_idle(jsonl)
    rel = str(jsonl.relative_to(watch_dir))
    state = {}

    # Provider IS configured; the slice deterministically fails extraction (unparseable output).
    with patch(_LLM_PATCH, return_value=_UNPARSEABLE), \
         patch("ormah.background.session_watcher.ingest_provider_configured", return_value=True):
        for i in range(1, MAX_EXTRACT_FAILURES):
            assert _ingest_session(engine, jsonl, state, watch_dir, min_turns=5) == IngestResult.TRANSIENT
            assert state[rel]["extract_fail_count"] == i
            assert state[rel]["end_offset"] == 0  # cursor NOT advanced yet
        # Capped: skip the toxic slice forward. The cursor advanced -> progress, so this is OK,
        # not NO_PROGRESS (which would bump the reconcile-park counter for a slice that just
        # progressed).
        assert _ingest_session(engine, jsonl, state, watch_dir, min_turns=5) == IngestResult.OK

    assert state[rel]["end_offset"] > 0            # cursor advanced past the toxic slice
    assert "extract_fail_count" not in state[rel]  # counter cleared after skip
    # Durable quarantine trail: the skipped range is recorded, not just logged, so it can be
    # replayed after the provider issue is fixed.
    skipped = state[rel]["skipped_slices"]
    assert len(skipped) == 1
    assert skipped[0]["start"] == 0
    assert skipped[0]["end"] == state[rel]["end_offset"]
    assert skipped[0]["reason"] == "extract_failed_x3"


def test_capped_skip_schedules_drain_continuation(engine, tmp_path):
    """When the toxic slice is a CAPPED batch (more closed content follows), the skip must call
    on_defer_active so the rest of the transcript drains on the next tick, not only via reconcile.
    (Council adjustment #3 for Task 04 — mirrors the success-path capped continuation.)"""
    watch_dir = tmp_path / "projects"
    project_dir = watch_dir / "-Users-alice-Code-myproject"
    project_dir.mkdir(parents=True)
    jsonl = project_dir / "abc123.jsonl"
    _make_jsonl(jsonl, user_turns=12)  # large enough that a small flush_bytes caps the first batch
    _mark_idle(jsonl)
    state = {}
    defer_calls: list[int] = []

    result = None
    with patch(_LLM_PATCH, return_value=_UNPARSEABLE), \
         patch("ormah.background.session_watcher.ingest_provider_configured", return_value=True):
        for _ in range(MAX_EXTRACT_FAILURES):
            result = _ingest_session(
                engine, jsonl, state, watch_dir, min_turns=1,
                flush_bytes=300,  # small -> the first closed batch is capped (content past it)
                on_defer_active=lambda: defer_calls.append(1),
            )

    assert result == IngestResult.OK          # capped slice skipped after the cap
    assert defer_calls, "on_defer_active must fire on a capped skip to drain the remainder"


def test_no_provider_failure_never_burns_the_slice(engine, tmp_path):
    """Without a provider, a failure must stay TRANSIENT and never advance the cursor or count —
    the data must survive until a provider returns."""
    watch_dir = tmp_path / "projects"
    project_dir = watch_dir / "-Users-alice-Code-myproject"
    project_dir.mkdir(parents=True)
    jsonl = project_dir / "abc123.jsonl"
    _make_jsonl(jsonl, user_turns=6)
    _mark_idle(jsonl)
    rel = str(jsonl.relative_to(watch_dir))
    state = {}

    with patch(_LLM_PATCH, return_value=None), \
         patch("ormah.background.session_watcher.ingest_provider_configured", return_value=False):
        for _ in range(MAX_EXTRACT_FAILURES + 2):
            assert _ingest_session(engine, jsonl, state, watch_dir, min_turns=5) == IngestResult.TRANSIENT

    # Never counted, never advanced: either no entry, or an entry with cursor still at 0 and no counter.
    entry = state.get(rel, {})
    assert entry.get("end_offset", 0) == 0
    assert "extract_fail_count" not in entry


def test_extract_fail_count_persists_across_restart(engine, tmp_path):
    """The per-slice failure counter must survive a process restart (persisted state), not just
    live in-memory — otherwise a restarted watcher resets the cap and the loop never breaks."""
    watch_dir = tmp_path / "projects"
    project_dir = watch_dir / "-Users-alice-Code-myproject"
    project_dir.mkdir(parents=True)
    jsonl = project_dir / "abc123.jsonl"
    _make_jsonl(jsonl, user_turns=6)
    _mark_idle(jsonl)
    rel = str(jsonl.relative_to(watch_dir))
    state = {}

    with patch(_LLM_PATCH, return_value=_UNPARSEABLE), \
         patch("ormah.background.session_watcher.ingest_provider_configured", return_value=True):
        for i in range(1, MAX_EXTRACT_FAILURES):
            assert _ingest_session(engine, jsonl, state, watch_dir, min_turns=5) == IngestResult.TRANSIENT

        assert state[rel]["extract_fail_count"] == MAX_EXTRACT_FAILURES - 1

        # Simulate a restart: reload state from disk into a fresh dict.
        reloaded_state = _load_state(watch_dir)
        assert reloaded_state[rel]["extract_fail_count"] == MAX_EXTRACT_FAILURES - 1

        # The (MAX_EXTRACT_FAILURES)th failure, on the reloaded state, must still trip the cap.
        assert _ingest_session(engine, jsonl, reloaded_state, watch_dir, min_turns=5) == IngestResult.OK

    assert reloaded_state[rel]["end_offset"] > 0
    assert "extract_fail_count" not in reloaded_state[rel]


def test_success_after_cap_preserves_skipped_slices(engine, tmp_path):
    """A capped slice records a durable skipped_slices entry; a LATER successful slice must not
    wipe that quarantine trail. The success-path state write was building the entry from scratch
    (dropping skipped_slices) while the cap path copied existing state (council-pr C1)."""
    watch_dir = tmp_path / "projects"
    project_dir = watch_dir / "-Users-alice-Code-myproject"
    project_dir.mkdir(parents=True)
    jsonl = project_dir / "abc123.jsonl"
    _make_jsonl(jsonl, user_turns=12)
    _mark_idle(jsonl)
    rel = str(jsonl.relative_to(watch_dir))
    state = {}

    # Phase 1: the first (capped) batch deterministically fails extraction MAX_EXTRACT_FAILURES
    # times (slice-specific, unparseable) -> quarantined + skipped.
    with patch(_LLM_PATCH, return_value=_UNPARSEABLE), \
         patch("ormah.background.session_watcher.ingest_provider_configured", return_value=True):
        for _ in range(MAX_EXTRACT_FAILURES):
            _ingest_session(engine, jsonl, state, watch_dir, min_turns=1, flush_bytes=300)
    assert state[rel]["skipped_slices"], "precondition: first slice quarantined"
    quarantined = list(state[rel]["skipped_slices"])

    # Phase 2: the NEXT batch extracts successfully and writes fresh success state.
    ok = json.dumps({"memories": [{"content": "a genuine memory to store", "type": "fact",
                                   "title": "t"}]})
    with patch(_LLM_PATCH, return_value=ok), \
         patch("ormah.background.session_watcher.ingest_provider_configured", return_value=True):
        result = _ingest_session(engine, jsonl, state, watch_dir, min_turns=1, flush_bytes=300)

    assert result == IngestResult.OK
    # The durable quarantine trail must survive the successful write.
    assert state[rel]["skipped_slices"] == quarantined


def test_ingest_exception_counts_toward_cap(engine, tmp_path):
    """A DETERMINISTIC exception in ingest_conversation must count toward the per-slice cap and
    eventually skip the slice — otherwise it pins the cursor forever, the same loop the string
    path already guards against (council-pr I1)."""
    watch_dir = tmp_path / "projects"
    project_dir = watch_dir / "-Users-alice-Code-myproject"
    project_dir.mkdir(parents=True)
    jsonl = project_dir / "abc123.jsonl"
    _make_jsonl(jsonl, user_turns=6)
    _mark_idle(jsonl)
    rel = str(jsonl.relative_to(watch_dir))
    state = {}

    with patch.object(engine, "ingest_conversation", side_effect=RuntimeError("boom")), \
         patch("ormah.background.session_watcher.ingest_provider_configured", return_value=True):
        for i in range(1, MAX_EXTRACT_FAILURES):
            assert _ingest_session(engine, jsonl, state, watch_dir, min_turns=5) == IngestResult.TRANSIENT
            assert state[rel]["extract_fail_count"] == i
            assert state[rel]["end_offset"] == 0  # cursor pinned until capped
        # The capped attempt skips the slice forward instead of looping forever.
        assert _ingest_session(engine, jsonl, state, watch_dir, min_turns=5) == IngestResult.OK

    assert state[rel]["end_offset"] > 0
    skipped = state[rel]["skipped_slices"]
    assert len(skipped) == 1
    assert skipped[0]["reason"] == "ingest_exception_x3"  # distinguishable from extract failures (M1)


def test_transient_storage_exception_never_skips_slice(engine, tmp_path):
    """A retryable storage exception (SQLite lock under WAL contention) must stay TRANSIENT forever
    and never advance the cursor or count toward the cap — else a lock that clears later loses the
    slice permanently (council-pr H2). Only DETERMINISTIC exceptions may be capped."""
    import sqlite3

    watch_dir = tmp_path / "projects"
    project_dir = watch_dir / "-Users-alice-Code-myproject"
    project_dir.mkdir(parents=True)
    jsonl = project_dir / "abc123.jsonl"
    _make_jsonl(jsonl, user_turns=6)
    _mark_idle(jsonl)
    rel = str(jsonl.relative_to(watch_dir))
    state = {}

    with patch.object(engine, "ingest_conversation",
                      side_effect=sqlite3.OperationalError("database is locked")), \
         patch("ormah.background.session_watcher.ingest_provider_configured", return_value=True):
        for _ in range(MAX_EXTRACT_FAILURES + 2):
            assert _ingest_session(engine, jsonl, state, watch_dir, min_turns=5) == IngestResult.TRANSIENT

    entry = state.get(rel, {})
    assert entry.get("end_offset", 0) == 0        # cursor never advanced
    assert "extract_fail_count" not in entry      # never counted toward the cap
    assert "skipped_slices" not in entry          # never quarantined -> no data loss


def test_provider_wide_call_failure_never_skips_slice(engine, tmp_path):
    """A provider-wide LLM call failure (binary/auth/network/timeout -> raw is None -> CALL_FAILED)
    must stay TRANSIENT and never count toward the cap: during an outage every slice would otherwise
    be skipped after the cap = mass silent loss (council-pr H1). Only slice-specific parse failures
    are capped."""
    watch_dir = tmp_path / "projects"
    project_dir = watch_dir / "-Users-alice-Code-myproject"
    project_dir.mkdir(parents=True)
    jsonl = project_dir / "abc123.jsonl"
    _make_jsonl(jsonl, user_turns=6)
    _mark_idle(jsonl)
    rel = str(jsonl.relative_to(watch_dir))
    state = {}

    # Both provider checks TRUE + call returns None -> _extract_memories_llm returns CALL_FAILED
    # (a provider-wide failure, not a slice defect).
    with patch(_LLM_PATCH, return_value=None), \
         patch("ormah.engine.memory_engine.ingest_provider_configured", return_value=True), \
         patch("ormah.background.session_watcher.ingest_provider_configured", return_value=True):
        for _ in range(MAX_EXTRACT_FAILURES + 3):
            assert _ingest_session(engine, jsonl, state, watch_dir, min_turns=5) == IngestResult.TRANSIENT

    entry = state.get(rel, {})
    assert entry.get("end_offset", 0) == 0    # cursor never advanced during the outage
    assert "extract_fail_count" not in entry  # provider-wide failure never counts toward the cap
    assert "skipped_slices" not in entry       # nothing skipped -> no data loss


def test_ingest_valid_empty_memories_advances(engine, tmp_path):
    """A valid {"memories": []} extraction is a SUCCESS: the slice is consumed and the
    cursor advances, so session_watcher never re-processes a no-memory turn forever."""
    watch_dir = tmp_path / "projects"
    project_dir = watch_dir / "-Users-alice-Code-myproject"
    project_dir.mkdir(parents=True)
    jsonl = project_dir / "abc123.jsonl"
    _make_jsonl(jsonl, user_turns=6)
    _mark_idle(jsonl)

    state = {}
    with patch(_LLM_PATCH, return_value='{"memories": []}'):
        result = _ingest_session(engine, jsonl, state, watch_dir, min_turns=5)

    assert result == IngestResult.OK
    rel = str(jsonl.relative_to(watch_dir))
    entry = state[rel]
    assert entry["end_offset"] > 0  # cursor advanced past the consumed slice
    assert entry["node_ids"] == []


def test_ingest_null_optional_fields_does_not_wedge_cursor(engine, tmp_path):
    """Cursor-wedge regression: the fallback extraction path is not --json-schema-
    constrained, so tags/about_self/confidence can arrive as null. That must not raise
    inside ingest_conversation -> propagate as an error string -> TRANSIENT forever."""
    watch_dir = tmp_path / "projects"
    project_dir = watch_dir / "-Users-alice-Code-myproject"
    project_dir.mkdir(parents=True)
    jsonl = project_dir / "abc123.jsonl"
    _make_jsonl(jsonl, user_turns=6)
    _mark_idle(jsonl)

    null_fields_response = json.dumps({"memories": [
        {"content": "x", "type": "fact", "title": "t",
         "tags": None, "about_self": None, "confidence": None},
    ]})

    state = {}
    with patch(_LLM_PATCH, return_value=null_fields_response):
        result = _ingest_session(engine, jsonl, state, watch_dir, min_turns=5)

    assert result == IngestResult.OK
    rel = str(jsonl.relative_to(watch_dir))
    entry = state[rel]
    assert entry["end_offset"] > 0  # cursor advanced, not wedged
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
    primary = project_dir / "abc123.jsonl"
    _make_jsonl(primary, user_turns=6)
    _mark_idle(primary)  # finished session, below flush_bytes → idle flush
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
    with patch(_JUDGE_PATCH, mock_llm):
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
    with patch(_JUDGE_PATCH, return_value=llm_response) as mock_llm:
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


def test_llm_judge_no_schemaless_fallback_on_schema_failure(engine, tmp_path):
    """When the schema call fails, the judge gives up rather than retrying without a schema."""
    prompt = "How should we solve feedback collection?"
    response = "We should first fix the database uniqueness key."
    transcript_path = tmp_path / "judge-schema-failure-session.jsonl"
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
        session_id="judge-schema-failure-session",
        prompt=prompt,
    )
    engine.settings.llm_provider = "ollama"
    engine.settings.feedback_llm_judge_enabled = True

    mock_llm = MagicMock(return_value=None)
    with patch(_JUDGE_PATCH, mock_llm):
        recorded = _record_whisper_usage_signals(engine, transcript)

    assert recorded == 1
    assert mock_llm.call_count == 1

    judge_signal = engine.db.conn.execute(
        "SELECT * FROM signals WHERE whisper_log_id = ? "
        "AND source = 'transcript_watcher_llm_judge'",
        (whisper_log_id,),
    ).fetchone()
    assert judge_signal is None


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
    with patch(_JUDGE_PATCH, return_value=llm_response):
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
    with patch(_JUDGE_PATCH, return_value=llm_response):
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
    with patch(_JUDGE_PATCH, mock_llm):
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
    with patch(_JUDGE_PATCH, mock_llm):
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

    # A short ACTIVE window below min_turns defers (noise cut) rather than extracting —
    # retry until it crosses min_turns, crosses flush_bytes, or the session idles.
    assert result == IngestResult.TRANSIENT
    assert str(jsonl.relative_to(watch_dir)) not in state


def test_min_turns_skips_short_active_window(engine, tmp_path):
    """A window below min_turns that is NOT idle must defer, not extract (noise cut)."""
    watch_dir = tmp_path / "projects"
    project_dir = watch_dir / "-Users-alice-Code-myproject"
    project_dir.mkdir(parents=True)
    jsonl = project_dir / "abc123.jsonl"
    _make_jsonl(jsonl, user_turns=2)  # below min_turns=5
    # NOT marked idle -> active short window

    state = {}
    with patch(_LLM_PATCH, return_value=_LLM_RESPONSE):
        result = _ingest_session(engine, jsonl, state, watch_dir, min_turns=5)
    assert result != IngestResult.OK
    assert state == {}


def test_min_turns_still_flushes_short_idle_session(engine, tmp_path):
    """A short but FINISHED (idle) session must still be captured — not stranded."""
    watch_dir = tmp_path / "projects"
    project_dir = watch_dir / "-Users-alice-Code-myproject"
    project_dir.mkdir(parents=True)
    jsonl = project_dir / "abc123.jsonl"
    _make_jsonl(jsonl, user_turns=2)
    _mark_idle(jsonl)  # finished

    state = {}
    with patch(_LLM_PATCH, return_value=_LLM_RESPONSE):
        result = _ingest_session(engine, jsonl, state, watch_dir, min_turns=5)
    assert result == IngestResult.OK


# --- Test 4: Unchanged session skipped ---

def test_unchanged_session_skipped(engine, tmp_path):
    """Same hash → session not re-ingested."""
    watch_dir = tmp_path / "projects"
    project_dir = watch_dir / "-Users-alice-Code-myproject"
    project_dir.mkdir(parents=True)
    jsonl = project_dir / "session.jsonl"
    _make_jsonl(jsonl, user_turns=6)
    _mark_idle(jsonl)  # finished session, below flush_bytes → idle flush

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
    _mark_idle(recent)  # finished session, below flush_bytes → idle flush

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
    _mark_idle(jsonl)  # finished session, below flush_bytes → idle flush

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
        _mark_idle(jsonl)  # appended session, below flush_bytes → idle flush
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
    _mark_idle(jsonl)  # finished session, below flush_bytes → idle flush

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
    _mark_idle(jsonl)  # finished session, below flush_bytes → idle flush

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
        _mark_idle(jsonl)  # shrunk session, below flush_bytes → idle flush
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
    _mark_idle(jsonl)  # finished-so-far session, below flush_bytes → idle flush
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
        # into the middle of the response. Mark idle too: this must hold back regardless
        # of idle, because the trailing record is genuinely incomplete (not just small).
        _append_user(jsonl, 6)
        _append_assistant(jsonl, 6, stop_reason="tool_use")
        _mark_idle(jsonl)
        assert _ingest_session(engine, jsonl, state, watch_dir, min_turns=1) != IngestResult.OK
        assert state[rel]["end_offset"] == cursor1

        # The response completes with a terminal record: prompt + BOTH assistant records
        # commit together — never split.
        _append_assistant(jsonl, 6, stop_reason="end_turn")
        _mark_idle(jsonl)
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
    _mark_idle(jsonl)  # below flush_bytes → idle flush for the closed turns

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
    _mark_idle(jsonl)  # finished session, below flush_bytes → idle flush

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
    _mark_idle(jsonl)  # finished-so-far turn, below flush_bytes → idle flush
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
    _mark_idle(jsonl)  # finished session, below flush_bytes → idle flush

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
    _mark_idle(jsonl)  # finished session, below flush_bytes → idle flush

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
    _mark_idle(jsonl)  # finished session, below flush_bytes → idle flush
    state = {}
    with patch(_LLM_PATCH, return_value=_LLM_RESPONSE):
        assert _ingest_session(engine, jsonl, state, watch_dir, min_turns=5) == IngestResult.OK
    first_nodes = list(state[rel]["node_ids"])
    assert first_nodes  # first ingest produced at least one node

    _make_jsonl(jsonl, user_turns=5)  # smaller file → size < stored end_offset → full re-ingest
    _mark_idle(jsonl)  # shrunk session, below flush_bytes → idle flush
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


# --- Merge of #52 (catch-up off bind path) onto the reconcile rework -------------------
# These cover the behavior the merge introduced that NEITHER prior suite tested:
# the off-bind startup catch-up and the shutdown drain that closes the use-after-close
# window (issue #52), now expressed on the reconcile API (list[SessionWatch] + _stop_event).


def test_do_ingest_rejected_after_stop_event(engine, tmp_path):
    """Once _stop_event is set, _do_ingest rejects under the lock before touching the engine —
    the guard that closes the use-after-close window at shutdown (issue #52)."""
    watch_dir = tmp_path / "projects"
    project_dir = watch_dir / "-Users-alice-Code-myproject"
    project_dir.mkdir(parents=True)
    jsonl = project_dir / "abc123.jsonl"
    _make_jsonl(jsonl)
    _mark_idle(jsonl)

    handler = SessionHandler(engine, watch_dir, 60.0, 5, 30.0, 9999)
    handler._stop_event.set()

    with patch("ormah.background.session_watcher._ingest_session") as mock_ingest:
        result = handler._do_ingest(jsonl)

    assert result == IngestResult.TRANSIENT
    mock_ingest.assert_not_called()          # rejected before the heavy work
    assert handler.in_flight_count() == 0     # never claimed the path


def test_stop_session_watcher_drains_inflight_ingest(engine, tmp_path):
    """stop_session_watcher blocks until an in-flight ingest finishes, so nothing writes to the
    DB after the lifespan calls engine.shutdown() right after (use-after-close guard, issue #52)."""
    import threading

    from ormah.background.session_watcher import SessionWatch

    watch_dir = tmp_path / "projects"
    watch_dir.mkdir(parents=True)
    handler = SessionHandler(engine, watch_dir, 60.0, 5, 30.0, 9999)

    entered = threading.Event()
    release = threading.Event()

    def blocking_ingest(*a, **k):
        entered.set()
        release.wait(5)
        return IngestResult.OK

    def worker():
        with patch("ormah.background.session_watcher._ingest_session", side_effect=blocking_ingest):
            handler._do_ingest(watch_dir / "x.jsonl")

    t = threading.Thread(target=worker)
    t.start()
    assert entered.wait(5)                     # an ingest is now in-flight
    assert handler.in_flight_count() == 1

    watch = SessionWatch(
        watch_dir=watch_dir, handler=handler, observer=MagicMock(), startup_reconcile_thread=None,
    )
    stop_returned = threading.Event()

    def stopper():
        stop_session_watcher([watch])
        stop_returned.set()

    s = threading.Thread(target=stopper)
    s.start()

    assert not stop_returned.wait(0.5)         # stop must NOT return while ingest is in-flight
    release.set()                              # let the ingest finish
    assert stop_returned.wait(5)               # now the drain completes and stop returns
    t.join(5)
    s.join(5)
    assert handler.in_flight_count() == 0


def test_start_session_watcher_runs_catchup_off_bind(engine, tmp_path):
    """start_session_watcher ingests a pre-existing backlog via the off-bind startup thread:
    the observer is live immediately (not blocked on a synchronous scan) and the backlog is
    recovered once the startup thread joins (issue #52)."""
    watch_dir = tmp_path / "projects"
    project_dir = watch_dir / "-Users-alice-Code-myproject"
    project_dir.mkdir(parents=True)
    jsonl = project_dir / "abc123.jsonl"
    _make_jsonl(jsonl)
    _mark_idle(jsonl)

    engine.settings.session_watcher_enabled = True
    engine.settings.session_watcher_dir = watch_dir
    engine.settings.session_watcher_debounce_seconds = 10.0
    engine.settings.session_watcher_lookback_hours = 9999

    with patch(_LLM_PATCH, return_value=_LLM_RESPONSE):
        watches = start_session_watcher(engine)
        try:
            assert len(watches) == 1
            assert watches[0].observer.is_alive()        # live from t0, scan did not block the bind
            assert watches[0].startup_reconcile_thread is not None
            watches[0].startup_reconcile_thread.join(10)           # deterministic wait for the off-bind drain
            assert not watches[0].startup_reconcile_thread.is_alive()
            rel = str(jsonl.relative_to(watch_dir))
            assert rel in watches[0].handler._state      # backlog ingested off the bind path
        finally:
            stop_session_watcher(watches)

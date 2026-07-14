"""Session watcher — auto-ingest completed agent JSONL transcripts."""

from __future__ import annotations

import hashlib
import json
import logging
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock, Timer

from watchdog.events import FileSystemEventHandler
from watchdog.observers import Observer

from ormah.engine.memory_engine import MemoryEngine
from ormah.text.tokens import distinctive_tokens
from ormah.transcript.parser import TranscriptResult, TranscriptTurn, parse_transcript

logger = logging.getLogger(__name__)

_STATE_FILENAME = ".session_watcher_state"
_HEURISTIC_SOURCE = "transcript_watcher_heuristic"
_LLM_JUDGE_SOURCE = "transcript_watcher_llm_judge"
_HEURISTIC_AFFINITY_SOURCE = "auto_heuristic"
_LLM_JUDGE_AFFINITY_SOURCE = "auto_llm_judge"
_FENCE_RE = re.compile(r"```(?:json)?\s*\n(.*?)```", re.DOTALL)
_DEFAULT_SESSION_WATCHER_DIR = Path("~/.claude/projects")
_CODEX_SESSION_WATCHER_DIR = Path("~/.codex/sessions")

_LLM_FEEDBACK_JUDGE_PROMPT = """\
You are judging retrieval feedback for Ormah, a memory system.

Given a user prompt, the assistant response, and memories that Ormah injected before the
assistant answered, decide whether each memory was actually useful retrieval context.

Verdicts:
- "used": the assistant response materially uses, cites, paraphrases, or relies on the memory.
- "irrelevant": the memory is clearly unrelated/noisy for this prompt and response.
- "uncertain": there is not enough evidence either way. Silence alone is uncertain, not irrelevant.

Rules:
- Do not mark a memory "used" just because it shares generic words with the response.
- Do not mark a memory "irrelevant" just because the assistant omitted it.
- Use "irrelevant" only when the memory is plainly off-topic for the user's prompt and answer.
- Prefer "uncertain" when the judgment is ambiguous.

Return strict JSON matching this shape:
{
  "verdicts": [
    {
      "whisper_log_id": 123,
      "verdict": "used|irrelevant|uncertain",
      "confidence": 0.0
    }
  ]
}
"""


def _normalise_text(text: str) -> str:
    """Lowercase text and collapse punctuation/whitespace for matching."""
    cleaned = re.sub(r"[^a-zA-Z0-9]+", " ", text.lower())
    return " ".join(cleaned.split())


def _assistant_response_after_prompt(
    turns: list[TranscriptTurn],
    prompt_text: str | None,
) -> str | None:
    """Return assistant text immediately following the matching user prompt."""
    if not prompt_text:
        return None

    wanted = _normalise_text(prompt_text)
    if not wanted:
        return None

    for idx, turn in enumerate(turns):
        if turn.role != "user" or _normalise_text(turn.text) != wanted:
            continue

        responses: list[str] = []
        for next_turn in turns[idx + 1:]:
            if next_turn.role == "user":
                break
            if next_turn.role == "assistant":
                responses.append(next_turn.text)
        return "\n\n".join(responses) if responses else None

    return None


def _node_usage_evidence(row, response_text: str) -> tuple[bool, float, dict]:
    """Detect whether an assistant response clearly referenced an injected memory."""
    response_norm = _normalise_text(response_text)
    response_tokens = distinctive_tokens(response_text, extra_stop_words={"memory", "ormah"})

    node_id = row["node_id"]
    short_id = node_id[:8] if node_id else ""
    if short_id and short_id.lower() in response_text.lower():
        return True, 1.0, {"match": "node_id", "short_id": short_id}

    title = row["title"] or ""
    title_tokens = distinctive_tokens(title, extra_stop_words={"memory", "ormah"})
    title_norm = _normalise_text(title)
    if len(title_tokens) >= 2 and len(title_norm) >= 12 and title_norm in response_norm:
        return True, 0.95, {"match": "title", "title": title}

    content = row["content"] or ""
    for sentence in re.split(r"[\n.!?]+", content):
        sentence = sentence.strip()
        if len(sentence) < 24:
            continue
        sentence_tokens = distinctive_tokens(sentence, extra_stop_words={"memory", "ormah"})
        sentence_norm = _normalise_text(sentence)
        if len(sentence_tokens) >= 4 and sentence_norm in response_norm:
            return True, 0.9, {"match": "sentence", "text": sentence[:160]}

    node_tokens = distinctive_tokens(
        f"{title} {content}",
        extra_stop_words={"memory", "ormah"},
    )
    prompt_tokens = distinctive_tokens(row["prompt_text"] or "")
    candidate_tokens = node_tokens - prompt_tokens
    overlap = sorted(candidate_tokens & response_tokens)
    denominator = min(len(candidate_tokens), 12)
    overlap_ratio = (len(overlap) / denominator) if denominator else 0.0
    if len(overlap) >= 4 and overlap_ratio >= 0.5:
        return True, min(0.85, 0.45 + overlap_ratio), {
            "match": "token_overlap",
            "overlap": overlap[:12],
            "overlap_ratio": round(overlap_ratio, 3),
        }

    return False, 0.0, {
        "match": "none",
        "overlap": overlap[:12],
        "overlap_ratio": round(overlap_ratio, 3),
    }


def _extract_json(raw: str) -> str:
    """Extract JSON from an LLM response that may contain fences or prose."""
    stripped = raw.strip()
    if stripped.startswith(("{", "[")):
        return stripped

    match = _FENCE_RE.search(raw)
    if match:
        return match.group(1).strip()

    for start_char, end_char in [("{", "}"), ("[", "]")]:
        start = raw.find(start_char)
        end = raw.rfind(end_char)
        if start != -1 and end > start:
            return raw[start : end + 1]

    return stripped


def _normalise_judge_verdict(raw: object) -> str:
    """Map loose LLM verdict labels to the canonical feedback verdicts."""
    value = str(raw or "").strip().lower().replace("-", "_").replace(" ", "_")
    if value in {"used", "useful", "referenced", "positive", "relevant"}:
        return "used"
    if value in {
        "irrelevant",
        "clearly_irrelevant",
        "not_useful",
        "negative",
        "noisy",
        "noise",
    }:
        return "irrelevant"
    return "uncertain"


def _confidence(raw: object) -> float:
    """Parse and clamp an LLM confidence value into [0.0, 1.0]."""
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return 0.0
    return max(0.0, min(1.0, value))


def _llm_feedback_judge_response_format() -> dict:
    """Return the compact structured-output schema for whisper feedback judgments."""
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "whisper_feedback_verdicts",
            "strict": True,
            "schema": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "verdicts": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "additionalProperties": False,
                            "properties": {
                                "whisper_log_id": {"type": "integer"},
                                "verdict": {
                                    "type": "string",
                                    "enum": ["used", "irrelevant", "uncertain"],
                                },
                                "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                            },
                            "required": ["whisper_log_id", "verdict", "confidence"],
                        },
                    },
                },
                "required": ["verdicts"],
            },
        },
    }


def _feedback_llm_judge_enabled(engine: MemoryEngine) -> bool:
    settings = engine.settings
    return bool(
        getattr(settings, "feedback_llm_judge_enabled", False)
        and getattr(settings, "llm_enabled", False)
    )


def _llm_judge_whisper_usage(
    engine: MemoryEngine,
    prompt_text: str,
    response_text: str,
    rows: list,
) -> dict[int, dict]:
    """Ask the configured LLM to judge ambiguous whisper usage for one turn."""
    if not rows:
        return {}

    from ormah.background.llm_client import llm_generate

    candidates = [
        {
            "whisper_log_id": row["id"],
            "node_id": (row["node_id"] or "")[:8],
            "title": row["title"] or "",
            "content": (row["content"] or "")[:1200],
        }
        for row in rows
    ]
    payload = {
        "user_prompt": (prompt_text or "")[:2500],
        "assistant_response": response_text[:5000],
        "memories": candidates,
    }
    prompt = (
        _LLM_FEEDBACK_JUDGE_PROMPT
        + "\n\nInput JSON:\n"
        + json.dumps(payload, ensure_ascii=False, indent=2)
    )

    raw = llm_generate(
        engine.settings,
        prompt,
        json_mode=True,
        response_format=_llm_feedback_judge_response_format(),
        temperature=0,
        max_tokens=512,
    )
    if raw is None:
        logger.info("LLM feedback judge schema call failed; falling back to JSON object mode")
        raw = llm_generate(
            engine.settings,
            prompt,
            json_mode=True,
            temperature=0,
            max_tokens=512,
        )
    if raw is None:
        return {}

    try:
        parsed = json.loads(_extract_json(raw))
    except (json.JSONDecodeError, TypeError, ValueError):
        logger.warning("LLM returned invalid JSON for feedback judgment")
        return {}

    verdicts = parsed.get("verdicts") if isinstance(parsed, dict) else parsed
    if not isinstance(verdicts, list):
        return {}

    judgments: dict[int, dict] = {}
    valid_ids = {int(row["id"]) for row in rows}
    for item in verdicts:
        if not isinstance(item, dict):
            continue
        raw_id = item.get("whisper_log_id", item.get("id"))
        try:
            whisper_log_id = int(raw_id)
        except (TypeError, ValueError):
            continue
        if whisper_log_id not in valid_ids:
            continue

        verdict = _normalise_judge_verdict(item.get("verdict"))
        confidence = _confidence(item.get("confidence"))
        judgments[whisper_log_id] = {
            "verdict": verdict,
            "confidence": confidence,
            "reason": str(item.get("reason") or "")[:500],
        }

    return judgments


def _insert_usage_signal(
    conn,
    row,
    transcript: TranscriptResult,
    *,
    signal_type: str,
    polarity: int,
    strength: float,
    source: str,
    evidence: dict,
    created: str,
) -> int:
    conn.execute(
        """
        INSERT INTO signals
            (
                whisper_log_id, node_id, signal_type, polarity, strength,
                source, session_id, agent_id, surface, space, prompt_hash,
                evidence, created
            )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT DO NOTHING
        """,
        (
            row["id"],
            row["node_id"],
            signal_type,
            polarity,
            strength,
            source,
            row["session_id"],
            transcript.source,
            "transcript_watcher",
            row["space"],
            row["prompt_hash"],
            json.dumps(evidence, sort_keys=True),
            created,
        ),
    )
    return conn.execute("SELECT changes()").fetchone()[0]


def _insert_affinity(
    conn,
    row,
    *,
    signal: int,
    source: str,
    confirmed_at: str,
) -> None:
    conn.execute(
        """
        INSERT INTO affinity
            (
                prompt_vec, prompt_text, node_id, signal, source,
                confirmed_at, space, session_id, whisper_log_id
            )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT DO NOTHING
        """,
        (
            row["prompt_vec"],
            row["prompt_text"],
            row["node_id"],
            signal,
            source,
            confirmed_at,
            row["space"],
            row["session_id"],
            row["id"],
        ),
    )


def _record_whisper_usage_signals(
    engine: MemoryEngine,
    transcript: TranscriptResult,
    turns: list | None = None,
) -> int:
    """Mine transcript responses for clear usage of injected whisper memories.

    *turns* restricts mining to a subset of ``transcript.turns`` (e.g. only the
    closed/safe blocks of an active session, so a still-growing assistant response
    is not judged from a partial body). Defaults to all turns.
    """
    if turns is None:
        turns = transcript.turns
    llm_judge_enabled = _feedback_llm_judge_enabled(engine)
    rows = engine.db.conn.execute(
        """
        SELECT
            wl.id, wl.node_id,
            COALESCE(re.prompt_text, wl.prompt_text) AS prompt_text,
            COALESCE(re.prompt_hash, wl.prompt_hash) AS prompt_hash,
            COALESCE(re.prompt_vec, wl.prompt_vec) AS prompt_vec,
            COALESCE(re.session_id, wl.session_id) AS session_id,
            COALESCE(re.space, wl.space) AS space,
            n.title, n.content,
            (
                SELECT s.polarity FROM signals s
                WHERE s.whisper_log_id = wl.id
                  AND s.source = ?
                ORDER BY s.id DESC
                LIMIT 1
            ) AS heuristic_polarity,
            EXISTS (
                SELECT 1 FROM signals s
                WHERE s.whisper_log_id = wl.id
                  AND s.source = ?
            ) AS has_llm_judge
        FROM whisper_log wl
        LEFT JOIN retrieval_events re ON re.id = wl.retrieval_event_id
        JOIN nodes n ON n.id = wl.node_id
        WHERE wl.session_id = ?
          AND wl.was_injected = 1
        ORDER BY wl.logged_at ASC, wl.id ASC
        """,
        (_HEURISTIC_SOURCE, _LLM_JUDGE_SOURCE, transcript.session_id),
    ).fetchall()
    if not rows:
        return 0

    now_iso = datetime.now(timezone.utc).isoformat()
    recorded = 0

    heuristic_records: list[dict] = []
    llm_groups: dict[tuple[str, str], list] = {}
    response_cache: dict[str, str | None] = {}
    for row in rows:
        prompt_text = row["prompt_text"] or ""
        if prompt_text not in response_cache:
            response_cache[prompt_text] = _assistant_response_after_prompt(
                turns,
                prompt_text,
            )
        response = response_cache[prompt_text]
        if response is None:
            continue

        heuristic_polarity = row["heuristic_polarity"]
        has_heuristic = heuristic_polarity is not None
        has_llm_judge = bool(row["has_llm_judge"])

        referenced = False
        if not has_heuristic:
            referenced, strength, evidence = _node_usage_evidence(row, response)
            signal_type = "whisper_referenced" if referenced else "whisper_unreferenced"
            polarity = 1 if referenced else 0
            heuristic_records.append({
                "row": row,
                "signal_type": signal_type,
                "polarity": polarity,
                "strength": strength,
                "evidence": {
                    **evidence,
                    "detector": _HEURISTIC_SOURCE,
                    "response_chars": len(response),
                },
            })
        else:
            referenced = int(heuristic_polarity) == 1

        if llm_judge_enabled and not has_llm_judge and not referenced:
            llm_groups.setdefault((prompt_text, response), []).append(row)

    with engine.db.transaction() as conn:
        for record in heuristic_records:
            row = record["row"]
            recorded += _insert_usage_signal(
                conn,
                row,
                transcript,
                signal_type=record["signal_type"],
                polarity=record["polarity"],
                strength=record["strength"],
                source=_HEURISTIC_SOURCE,
                evidence=record["evidence"],
                created=now_iso,
            )
            if record["polarity"] == 1:
                _insert_affinity(
                    conn,
                    row,
                    signal=1,
                    source=_HEURISTIC_AFFINITY_SOURCE,
                    confirmed_at=now_iso,
                )

    if not llm_groups:
        return recorded

    judge_records: list[dict] = []
    min_confidence = getattr(engine.settings, "feedback_llm_judge_min_confidence", 0.75)
    for (prompt_text, response), group_rows in llm_groups.items():
        judgments = _llm_judge_whisper_usage(engine, prompt_text, response, group_rows)
        for row in group_rows:
            judgment = judgments.get(int(row["id"]))
            if judgment is None:
                continue

            verdict = judgment["verdict"]
            confidence = judgment["confidence"]
            promoted = confidence >= min_confidence and verdict in {"used", "irrelevant"}
            polarity = 0
            signal_type = "whisper_judged_uncertain"
            if promoted and verdict == "used":
                polarity = 1
                signal_type = "whisper_judged_used"
            elif promoted and verdict == "irrelevant":
                polarity = -1
                signal_type = "whisper_judged_irrelevant"

            judge_records.append({
                "row": row,
                "signal_type": signal_type,
                "polarity": polarity,
                "strength": confidence,
                "evidence": {
                    "detector": _LLM_JUDGE_SOURCE,
                    "verdict": verdict,
                    "confidence": confidence,
                    "min_confidence": min_confidence,
                    "reason": judgment["reason"],
                    "promoted": promoted,
                    "response_chars": len(response),
                },
            })

    with engine.db.transaction() as conn:
        for record in judge_records:
            row = record["row"]
            recorded += _insert_usage_signal(
                conn,
                row,
                transcript,
                signal_type=record["signal_type"],
                polarity=record["polarity"],
                strength=record["strength"],
                source=_LLM_JUDGE_SOURCE,
                evidence=record["evidence"],
                created=now_iso,
            )
            if record["polarity"] in (1, -1):
                _insert_affinity(
                    conn,
                    row,
                    signal=record["polarity"],
                    source=_LLM_JUDGE_AFFINITY_SOURCE,
                    confirmed_at=now_iso,
                )

    return recorded


def _space_from_encoded_dir(dirname: str) -> str | None:
    """Extract project space from an encoded transcript directory name.

    Claude Code uses paths like ``-Users-johndoe-Projects-ormah``.
    The current compatibility strategy uses the last ``-`` segment as
    the project name; future transcript sources should provide their
    own space strategy before reaching the watcher.
    Leading ``-`` is stripped before splitting.
    """
    stripped = dirname.lstrip("-")
    if not stripped:
        return None
    parts = stripped.split("-")
    return parts[-1] if parts else None


def _resolve_transcript_session_id(
    engine: MemoryEngine,
    path: Path,
    parsed_session_id: str,
    source: str,
) -> str:
    """Resolve source-specific transcript filenames back to hook session ids.

    Claude Code transcript filenames are the session id. Codex rollout filenames can embed
    the hook session id inside a longer filename, so use recent whisper_log rows to recover
    the id that was used when whispers were injected.
    """
    if not parsed_session_id:
        return parsed_session_id

    exact = engine.db.conn.execute(
        "SELECT 1 FROM whisper_log WHERE session_id = ? LIMIT 1",
        (parsed_session_id,),
    ).fetchone()
    if exact is not None:
        return parsed_session_id

    if source != "codex":
        return parsed_session_id

    row = engine.db.conn.execute(
        """
        SELECT session_id
        FROM whisper_log
        WHERE session_id IS NOT NULL
          AND session_id != ''
          AND length(session_id) >= 6
          AND ? LIKE '%' || session_id || '%'
        ORDER BY length(session_id) DESC, logged_at DESC, id DESC
        LIMIT 1
        """,
        (path.name,),
    ).fetchone()
    return row["session_id"] if row is not None else parsed_session_id


def _space_from_whisper_log(engine: MemoryEngine, session_id: str) -> str | None:
    """Return the most recent non-empty space logged for a whisper session."""
    if not session_id:
        return None

    row = engine.db.conn.execute(
        """
        SELECT space
        FROM whisper_log
        WHERE session_id = ?
          AND space IS NOT NULL
          AND space != ''
        ORDER BY logged_at DESC, id DESC
        LIMIT 1
        """,
        (session_id,),
    ).fetchone()
    return row["space"] if row is not None else None


def _space_for_transcript(
    engine: MemoryEngine,
    path: Path,
    result: TranscriptResult,
) -> str | None:
    """Choose the project space for a parsed transcript."""
    logged_space = _space_from_whisper_log(engine, result.session_id)
    if logged_space:
        return logged_space

    if result.source == "claude_code":
        return _space_from_encoded_dir(path.parent.name)

    return None


def _expand_watch_dir(path: Path) -> Path:
    return Path(path).expanduser().resolve()


def _session_watch_dirs(settings) -> list[Path]:
    """Return existing transcript watch directories for the current settings."""
    primary = _expand_watch_dir(settings.session_watcher_dir)
    candidates = [primary]

    if primary == _expand_watch_dir(_DEFAULT_SESSION_WATCHER_DIR):
        candidates.append(_expand_watch_dir(_CODEX_SESSION_WATCHER_DIR))

    watch_dirs: list[Path] = []
    for candidate in candidates:
        if candidate.exists() and candidate not in watch_dirs:
            watch_dirs.append(candidate)
    return watch_dirs


def _file_hash(path: Path) -> str:
    """Return SHA-256 hex digest of a file's contents."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_state(watch_dir: Path) -> dict:
    """Load persisted state for the watch directory."""
    state_path = watch_dir / _STATE_FILENAME
    if state_path.exists():
        try:
            return json.loads(state_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            logger.warning("Corrupted session watcher state file %s, starting fresh", state_path)
    return {}


def _save_state(watch_dir: Path, state: dict) -> None:
    """Persist state for the watch directory."""
    state_path = watch_dir / _STATE_FILENAME
    state_path.write_text(json.dumps(state, indent=2), encoding="utf-8")


def _ingest_session(
    engine: MemoryEngine,
    path: Path,
    state: dict,
    watch_dir: Path,
    min_turns: int,
    idle_threshold: float = 30.0,
    on_defer_active=None,
) -> bool:
    """Ingest a single JSONL session transcript if changed. Returns True if ingested."""
    rel = str(path.relative_to(watch_dir))

    try:
        h = _file_hash(path)
    except OSError as e:
        logger.warning("Cannot read %s: %s", path, e)
        return False
    try:
        size = path.stat().st_size
    except OSError as e:
        logger.warning("Cannot stat %s: %s", path, e)
        return False

    # Incremental: only parse the turns appended since the last ingest.
    existing = state.get(rel)
    prev_offset = existing.get("end_offset", 0) if existing else 0
    # Skip an unchanged file only if the previous ingest already consumed it whole. A stored
    # offset behind EOF means a pending tail or a legacy mid-response cursor still to process,
    # which must be re-parsed (so recovery can run) even when the hash is unchanged.
    if existing and existing.get("hash") == h and prev_offset >= size:
        return False
    if prev_offset > size:
        prev_offset = 0  # file shrank (compaction/rewrite) -> re-ingest whole

    try:
        result = parse_transcript(path, start_offset=prev_offset)
        if result.leading_orphan:
            # A cursor left mid-response by an older version: re-parse the whole file so
            # the dropped tail is recovered and re-paired with its prompt. A one-time
            # re-ingest of this file; the background dedup jobs reconcile any overlap.
            logger.info("Session watcher recovering legacy mid-response cursor for %s", rel)
            prev_offset = 0
            result = parse_transcript(path, start_offset=0)
    except Exception as e:
        logger.warning("Session transcript parse error for %s: %s", path, e)
        return False

    # Commit only the "safe" payload — the closed boundary, content proven complete by a
    # terminal stop_reason (Claude Code), a Codex task_complete event, or a following user
    # turn. This never splits a multi-record response from its prompt. A trailing block
    # with no completion signal yet is genuinely in-flight and is held back; once it
    # completes the file changes and the next parse picks it up. (A response left forever
    # in-flight — a process killed mid-turn — is intentionally never ingested.)
    payload_offset = result.safe_end_offset
    payload_conversation = result.safe_conversation
    payload_users = result.safe_user_turn_count
    payload_turns = result.safe_turns

    # When the file looks idle/finished, commit whatever is closed even below min_turns,
    # so a short finished session is not stranded.
    try:
        age = time.time() - path.stat().st_mtime
    except OSError:
        age = idle_threshold + 1  # treat unstatable file as idle
    is_idle = age > idle_threshold

    # Nothing new to commit at the closed boundary.
    if payload_offset <= prev_offset:
        # Active session with appended-but-unclosed content (a still-streaming response):
        # schedule a retry so the turn is committed once it completes.
        if not is_idle and result.end_offset > prev_offset and on_defer_active is not None:
            on_defer_active()
        return False

    # Short tail on an active session — defer until more turns close or the session idles.
    if not is_idle and payload_users < min_turns:
        if on_defer_active is not None:
            on_defer_active()  # schedule a retry so the tail is not lost
        return False

    result.session_id = _resolve_transcript_session_id(
        engine,
        path,
        result.session_id,
        result.source,
    )
    space = _space_for_transcript(engine, path, result)
    signals_recorded = _record_whisper_usage_signals(engine, result, turns=payload_turns)

    try:
        ingested = engine.ingest_conversation(
            content=payload_conversation,
            space=space,
            agent_id=result.source,
            extra_tags=["session-transcript"],
        )
        if isinstance(ingested, str):
            logger.warning("Session watcher ingestion failed for %s: %s", path, ingested)
            return False
        count = len(ingested) if isinstance(ingested, list) else 0
    except Exception as e:
        logger.warning("Session watcher ingestion error for %s: %s", path, e)
        return False

    new_node_ids = [m["node_id"] for m in ingested] if isinstance(ingested, list) else []
    # prev_offset == 0 means a fresh/whole re-ingest; don't carry stale cumulative
    # turns or node_ids forward (the new ingest re-covers them).
    carry = existing and prev_offset > 0
    prev_node_ids = existing.get("node_ids", []) if carry else []
    prev_turns = existing.get("user_turns", 0) if carry else 0

    state[rel] = {
        "hash": h,
        "end_offset": payload_offset,
        "last_ingested": datetime.now(timezone.utc).isoformat(),
        "session_id": result.session_id,
        "source": result.source,
        "space": space,
        "user_turns": prev_turns + payload_users,
        "node_ids": prev_node_ids + new_node_ids,
        "signals_recorded": signals_recorded,
    }
    _save_state(watch_dir, state)

    logger.info(
        "Session watcher ingested %s (%d new turns, %d memories extracted, %d signals recorded)",
        rel, payload_users, count, signals_recorded,
    )
    return True


def _scan_sessions(
    engine: MemoryEngine,
    watch_dir: Path,
    min_turns: int,
    lookback_hours: int,
) -> int:
    """Scan for new/changed JSONL transcripts. Returns count ingested."""
    state = _load_state(watch_dir)
    ingested = 0

    now = time.time()
    cutoff = now - (lookback_hours * 3600) if lookback_hours > 0 else 0

    for jsonl_file in sorted(watch_dir.rglob("*.jsonl")):
        rel = str(jsonl_file.relative_to(watch_dir))

        # Lookback cutoff applies only to never-ingested files
        if rel not in state and lookback_hours >= 0 and cutoff > 0:
            try:
                mtime = jsonl_file.stat().st_mtime
            except OSError:
                continue
            if mtime < cutoff:
                continue

        # lookback_hours == -1 means skip all never-ingested files (no catch-up)
        if rel not in state and lookback_hours < 0:
            continue

        if _ingest_session(engine, jsonl_file, state, watch_dir, min_turns):
            ingested += 1

    # Clean stale state entries for deleted files
    stale_keys = [
        rel for rel in list(state.keys())
        if not (watch_dir / rel).exists()
    ]
    for key in stale_keys:
        del state[key]
    if stale_keys:
        _save_state(watch_dir, state)

    return ingested


class SessionHandler(FileSystemEventHandler):
    """Watches for .jsonl file create/modify events with debouncing."""

    def __init__(
        self,
        engine: MemoryEngine,
        watch_dir: Path,
        debounce_seconds: float,
        min_turns: int,
        idle_threshold: float = 30.0,
    ) -> None:
        self.engine = engine
        self.watch_dir = watch_dir
        self.debounce_seconds = debounce_seconds
        self.min_turns = min_turns
        self.idle_threshold = idle_threshold
        self._state = _load_state(watch_dir)
        self._timers: dict[str, Timer] = {}
        self._ingesting: set[str] = set()
        self._pending: set[str] = set()
        self._lock = Lock()

    def _schedule_ingest(self, path: Path) -> None:
        """Schedule a debounced ingestion for the given file."""
        key = str(path)
        with self._lock:
            if key in self._timers:
                self._timers[key].cancel()
            timer = Timer(
                self.debounce_seconds,
                self._do_ingest,
                args=(path,),
            )
            timer.daemon = True
            self._timers[key] = timer
            timer.start()

    def _schedule_retry(self, path: Path) -> None:
        """Re-attempt ingestion after idle_threshold so an active short tail is not lost."""
        key = str(path)
        with self._lock:
            if key in self._timers:
                self._timers[key].cancel()
            timer = Timer(self.idle_threshold, self._do_ingest, args=(path,))
            timer.daemon = True
            self._timers[key] = timer
            timer.start()

    def _do_ingest(self, path: Path) -> None:
        """Actually ingest the session (called after debounce or retry)."""
        key = str(path)
        with self._lock:
            self._timers.pop(key, None)
            if key in self._ingesting:
                # An ingest for this path is already running and has already parsed
                # its slice; mark the path so the new content is re-ingested once it
                # finishes, instead of dropping this event.
                self._pending.add(key)
                return
            self._ingesting.add(key)
        try:
            _ingest_session(
                self.engine, path, self._state, self.watch_dir, self.min_turns,
                idle_threshold=self.idle_threshold,
                on_defer_active=lambda: self._schedule_retry(path),
            )
        finally:
            with self._lock:
                self._ingesting.discard(key)
                rerun = key in self._pending
                self._pending.discard(key)
        if rerun:
            self._schedule_ingest(path)  # re-process content that arrived mid-ingest

    def on_created(self, event):
        if not event.is_directory and event.src_path.endswith(".jsonl"):
            self._schedule_ingest(Path(event.src_path))

    def on_modified(self, event):
        if not event.is_directory and event.src_path.endswith(".jsonl"):
            self._schedule_ingest(Path(event.src_path))


def start_session_watcher(engine: MemoryEngine) -> list[Observer]:
    """Start the session watcher for agent transcript files.

    Performs an initial catch-up scan, then starts a real-time watcher.
    Returns list of Observer instances for shutdown.
    """
    s = engine.settings
    if not s.session_watcher_enabled:
        return []

    watch_dirs = _session_watch_dirs(s)
    if not watch_dirs:
        logger.warning("Session watcher dir does not exist: %s", _expand_watch_dir(s.session_watcher_dir))
        return []

    observers: list[Observer] = []
    for watch_dir in watch_dirs:
        # Catch-up scan
        ingested = _scan_sessions(
            engine, watch_dir, s.session_watcher_min_turns, s.session_watcher_lookback_hours,
        )
        if ingested:
            logger.info("Session watcher catch-up: ingested %d sessions from %s", ingested, watch_dir)

        # Start real-time watcher
        handler = SessionHandler(
            engine, watch_dir, s.session_watcher_debounce_seconds, s.session_watcher_min_turns,
            s.session_watcher_idle_threshold,
        )
        observer = Observer()
        observer.schedule(handler, str(watch_dir), recursive=True)
        observer.start()
        observers.append(observer)
        logger.info("Session watcher started on %s", watch_dir)

    return observers


def stop_session_watcher(observers: list[Observer]) -> None:
    """Stop and join all session watcher observers."""
    for observer in observers:
        observer.stop()
    for observer in observers:
        observer.join(timeout=5)
    if observers:
        logger.info("Session watcher stopped")

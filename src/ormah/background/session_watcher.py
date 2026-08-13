"""Session watcher — auto-ingest completed agent JSONL transcripts."""

from __future__ import annotations

import hashlib
import json
import logging
import re
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import Enum
from pathlib import Path
from threading import Event, Lock, Thread, Timer

from watchdog.events import FileSystemEventHandler
from watchdog.observers import Observer

from ormah.engine.memory_engine import MemoryEngine
from ormah.text.tokens import distinctive_tokens
from ormah.transcript.parser import (
    TranscriptResult,
    TranscriptTurn,
    parse_transcript,
    should_rewind,
)

logger = logging.getLogger(__name__)


class IngestResult(Enum):
    """Why an ingest attempt did/didn't commit, so reconcile parks only files that cannot
    progress (corrupt / frozen safe boundary) and never parks transient external failures."""
    OK = "ok"                    # committed new content
    NO_PROGRESS = "no_progress"  # nothing new at the safe boundary, or unparseable (file's fault) -> park-eligible
    TRANSIENT = "transient"      # external failure (engine error) or defer -> retry, never park

_STATE_FILENAME = ".session_watcher_state"
MAX_RECONCILE_RETRIES = 3
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

    now_iso = datetime.now(UTC).isoformat()
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

    # The LLM judge writes its own observability and affinity rows above. A
    # confident positive verdict is nevertheless confirmed use, so route only
    # that lifecycle mutation through the engine's shared operation. Heuristic
    # positives intentionally remain ranking/observability evidence only.
    for record in judge_records:
        if record["polarity"] == 1:
            engine._record_confirmed_use(record["row"]["node_id"])

    return recorded


def _is_subagent_transcript(path: Path) -> bool:
    """True for subagent transcripts (Claude Code writes them under ``<uuid>/subagents/``).

    These are internal agent scratch, not user-facing sessions — ingesting them balloons
    the store with low-value granular memories under a junk ``subagents`` space. Matches a
    ``subagents`` segment at any depth so nested layouts are covered too.
    """
    return "subagents" in path.parts


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
    state_lock=None,
) -> IngestResult:
    """Ingest a single JSONL session transcript if changed.

    Returns:
        IngestResult.OK         — new content was committed.
        IngestResult.NO_PROGRESS — nothing to commit at the safe boundary (file is frozen,
                                   corrupt, or already fully consumed) — park-eligible by
                                   reconcile after MAX_RECONCILE_RETRIES at the same size.
        IngestResult.TRANSIENT  — external failure (engine error, in-flight defer, or
                                   in-flight skip); never increments the park counter.
    """
    if _is_subagent_transcript(path):
        return IngestResult.NO_PROGRESS
    rel = str(path.relative_to(watch_dir))

    try:
        h = _file_hash(path)
    except OSError as e:
        logger.warning("Cannot read %s: %s", path, e)
        return IngestResult.TRANSIENT
    try:
        size = path.stat().st_size
    except OSError as e:
        logger.warning("Cannot stat %s: %s", path, e)
        return IngestResult.TRANSIENT

    # Incremental: only parse the turns appended since the last ingest.
    existing = state.get(rel)
    prev_offset = existing.get("end_offset", 0) if existing else 0
    # Skip an unchanged file only if the previous ingest already consumed it whole. A stored
    # offset behind EOF means a pending tail or a legacy mid-response cursor still to process,
    # which must be re-parsed (so recovery can run) even when the hash is unchanged.
    if existing and existing.get("hash") == h and prev_offset >= size:
        return IngestResult.NO_PROGRESS
    if prev_offset > size:
        prev_offset = 0  # file shrank (compaction/rewrite) -> re-ingest whole

    try:
        result = parse_transcript(path, start_offset=prev_offset)
        if should_rewind(result, prev_offset):
            # Orphan with NO forward progress: a genuine cursor left mid-response by an
            # older version. Re-parse the whole file so the dropped tail is re-paired with
            # its prompt. With forward progress the orphan is a false positive (ADR-0003,
            # #149): the fragment is dropped and the cursor advances — rewinding there
            # would re-ingest the whole file on every tick forever.
            original_offset = prev_offset
            logger.info("Session watcher recovering legacy mid-response cursor for %s", rel)
            prev_offset = 0
            result = parse_transcript(path, start_offset=0)
            if result.safe_end_offset <= original_offset:
                # The rewind itself made no progress: the "orphan" tail is a still-open
                # in-flight response, not a recoverable one. ADR-0003: a no-progress
                # transcript parks, it does not re-extract the closed prefix every tick.
                return IngestResult.NO_PROGRESS
    except Exception as e:  # noqa: BLE001 - transcript parsers can raise provider-specific errors
        logger.warning("Session transcript parse error for %s: %s", path, e)
        return IngestResult.NO_PROGRESS

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
            return IngestResult.TRANSIENT  # will grow; retry, never park
        return IngestResult.NO_PROGRESS   # idle/frozen safe boundary -> park-eligible

    # Short tail on an active session — defer until more turns close or the session idles.
    if not is_idle and payload_users < min_turns:
        if on_defer_active is not None:
            on_defer_active()  # schedule a retry so the tail is not lost
        return IngestResult.TRANSIENT

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
            return IngestResult.TRANSIENT
        count = len(ingested) if isinstance(ingested, list) else 0
    except Exception as e:  # noqa: BLE001 - engine providers surface heterogeneous failures
        logger.warning("Session watcher ingestion error for %s: %s", path, e)
        return IngestResult.TRANSIENT

    new_node_ids = [m["node_id"] for m in ingested] if isinstance(ingested, list) else []
    # prev_offset == 0 means a fresh/whole re-ingest; don't carry stale cumulative
    # turns or node_ids forward (the new ingest re-covers them).
    carry = existing and prev_offset > 0
    prev_node_ids = existing.get("node_ids", []) if carry else []
    prev_turns = existing.get("user_turns", 0) if carry else 0

    entry = {
        "hash": h,
        "end_offset": payload_offset,
        "last_ingested": datetime.now(UTC).isoformat(),
        "session_id": result.session_id,
        "source": result.source,
        "space": space,
        "user_turns": prev_turns + payload_users,
        "node_ids": prev_node_ids + new_node_ids,
        "signals_recorded": signals_recorded,
    }
    if state_lock is not None:
        with state_lock:
            state[rel] = entry
            _save_state(watch_dir, state)
    else:
        state[rel] = entry
        _save_state(watch_dir, state)

    logger.info(
        "Session watcher ingested %s (%d new turns, %d memories extracted, %d signals recorded)",
        rel, payload_users, count, signals_recorded,
    )
    return IngestResult.OK


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

        if _ingest_session(engine, jsonl_file, state, watch_dir, min_turns) == IngestResult.OK:
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
        lookback_hours: int = 72,
    ) -> None:
        self.engine = engine
        self.watch_dir = watch_dir
        self.debounce_seconds = debounce_seconds
        self.min_turns = min_turns
        self.idle_threshold = idle_threshold
        self.lookback_hours = lookback_hours
        self._state = _load_state(watch_dir)
        self._timers: dict[str, Timer] = {}
        self._ingesting: set[str] = set()
        self._pending: set[str] = set()
        self._lock = Lock()
        self._state_lock = Lock()
        self._reconcile_lock = Lock()
        self._stop_event = Event()
        self._reconcile_attempts: dict[str, tuple[int, int, int]] = {}  # rel -> (size, mtime_ns, no_progress_count)
        self._reconcile_transient: dict[str, tuple[int, int, int]] = {}  # rel -> (size, mtime_ns, count)

    def _schedule_ingest(self, path: Path) -> None:
        """Schedule a debounced ingestion for the given file."""
        key = str(path)
        with self._lock:
            if self._stop_event.is_set():
                return
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
            if self._stop_event.is_set():
                return
            if key in self._timers:
                self._timers[key].cancel()
            timer = Timer(self.idle_threshold, self._do_ingest, args=(path,))
            timer.daemon = True
            self._timers[key] = timer
            timer.start()

    def _do_ingest(self, path: Path) -> IngestResult:
        """Ingest the session (after debounce, retry, or reconcile). Returns IngestResult.

        The heavy work (parse/LLM/DB) runs lock-free; only the state read-modify-write
        serializes via ``self._state_lock`` (passed into ``_ingest_session``), so a backlog
        reconcile never blocks the live fast path.
        """
        key = str(path)
        with self._lock:
            self._timers.pop(key, None)
            if self._stop_event.is_set():
                return IngestResult.TRANSIENT
            if key in self._ingesting:
                self._pending.add(key)
                return IngestResult.TRANSIENT
            self._ingesting.add(key)
        result = IngestResult.NO_PROGRESS
        try:
            result = _ingest_session(
                self.engine, path, self._state, self.watch_dir, self.min_turns,
                idle_threshold=self.idle_threshold,
                on_defer_active=lambda: self._schedule_retry(path),
                state_lock=self._state_lock,
            )
        finally:
            with self._lock:
                self._ingesting.discard(key)
                rerun = key in self._pending
                self._pending.discard(key)
        if rerun and not self._stop_event.is_set():
            self._schedule_ingest(path)
        return result

    def reconcile(self) -> int:
        """Disk-truth safety net: ingest transcripts the live FSEvents path dropped.

        Mechanism-agnostic and cheap: a stat-only scan finds files that still need work —
        never-seen (within lookback) or a state cursor not at EOF — then routes up to
        ``session_watcher_reconcile_max_per_tick`` of them through ``self._do_ingest`` (the
        single state owner, so no clobber / no double-ingest). A file with a pending or failed
        tail (``end_offset != size``) is retried each tick — bounded to
        ``MAX_RECONCILE_RETRIES`` attempts per size so an abandoned in-flight tail is not
        re-hashed forever — so a transient ingest failure never strands it. Returns
        transcripts recovered.

        Candidates are sorted most-recently-modified first so freshly dropped transcripts are
        recovered soonest. A per-tick wall-clock budget
        (``session_watcher_reconcile_max_seconds``) caps scheduler-thread occupancy: remaining
        candidates are picked up on the next tick.

        The park key for NO_PROGRESS is ``(size, mtime_ns)``: a same-size content rewrite
        (e.g. a repaired JSONL line) changes the mtime_ns, producing a new token that un-parks
        the file immediately without waiting for a process restart.  Persistently-TRANSIENT files
        (external failures that repeat at the same token) are deprioritized — sorted behind
        fresh candidates — rather than parked, so they keep being retried but cannot monopolize
        the per-tick cap and starve valid candidates.
        """
        if self._stop_event.is_set():
            return 0
        if not self._reconcile_lock.acquire(blocking=False):
            return 0
        try:
            cutoff = time.time() - (self.lookback_hours * 3600) if self.lookback_hours > 0 else 0
            cap = self.engine.settings.session_watcher_reconcile_max_per_tick
            candidates: list[tuple[bool, float, Path]] = []
            for jsonl_file in sorted(self.watch_dir.rglob("*.jsonl")):
                if self._stop_event.is_set():
                    return 0
                if _is_subagent_transcript(jsonl_file):
                    continue
                try:
                    st = jsonl_file.stat()
                except OSError:
                    continue
                rel = str(jsonl_file.relative_to(self.watch_dir))
                entry = self._state.get(rel)
                if entry is None:
                    # Never-seen: mirror _scan_sessions catch-up rules.
                    if self.lookback_hours < 0:
                        continue  # catch-up disabled -> skip never-seen files
                    if cutoff > 0 and st.st_mtime < cutoff:
                        continue
                elif entry.get("end_offset", 0) == st.st_size:
                    # Fully consumed -> skip cheaply (no hash, no _do_ingest).
                    # ponytail: known limitation (council-pr H1') — a same-size rewrite that PRESERVES
                    # mtime_ns (utime / cp --preserve, or an in-place repair restoring the timestamp)
                    # is invisible here and in the park check below. Closing it means hashing every
                    # consumed file each tick, reintroducing the O(n) scan cost flagged earlier; the
                    # Claude/Codex transcript workload is append-only (rewrites grow the file), so this
                    # pattern does not occur in practice. Upgrade path: content-hash token if it ever does.
                    continue
                # else: seen with cursor not at EOF -> pending/failed tail (or a rewrite).
                token = (st.st_size, st.st_mtime_ns)
                # H1: park NO_PROGRESS by (size, mtime_ns) — a same-size content rewrite (new mtime)
                # changes the token and un-parks the file, so a recoverable tail is never stranded.
                park = self._reconcile_attempts.get(rel)
                if park is not None and (park[0], park[1]) == token and park[2] >= MAX_RECONCILE_RETRIES:
                    continue  # parked at this exact content; skip until the content (token) changes
                # H2: deprioritize (never park) a file that keeps failing TRANSIENT at this token, so a
                # cluster of deterministically-failing files can't monopolize the per-tick cap and
                # starve valid candidates. Deprioritized files still get retried — just behind fresh ones.
                tr = self._reconcile_transient.get(rel)
                deprioritized = (
                    tr is not None and (tr[0], tr[1]) == token and tr[2] >= MAX_RECONCILE_RETRIES
                )
                candidates.append((deprioritized, st.st_mtime, jsonl_file))
            # Non-deprioritized first (newest-first, so freshly dropped transcripts recover soonest),
            # then deprioritized (oldest-first FIFO, so a long-failing transient that just became
            # recoverable is retried before newer deprioritized peers — no intra-group starvation).
            candidates.sort(key=lambda t: (t[0], t[1] if t[0] else -t[1]))
            recovered = 0
            budget = self.engine.settings.session_watcher_reconcile_max_seconds
            start = time.time()
            for _dep, _mtime, jsonl_file in candidates[:cap]:
                if self._stop_event.is_set() or time.time() - start >= budget:
                    break  # yield scheduler thread; remaining picked up on the next run
                rel = str(jsonl_file.relative_to(self.watch_dir))
                try:
                    st2 = jsonl_file.stat()
                except OSError:
                    continue
                size, mtime_ns = st2.st_size, st2.st_mtime_ns
                result = self._do_ingest(jsonl_file)
                if result == IngestResult.OK:
                    recovered += 1
                    self._reconcile_attempts.pop(rel, None)
                    self._reconcile_transient.pop(rel, None)
                elif result == IngestResult.NO_PROGRESS:
                    prev = self._reconcile_attempts.get(rel)
                    count = prev[2] + 1 if (prev is not None and (prev[0], prev[1]) == (size, mtime_ns)) else 1
                    self._reconcile_attempts[rel] = (size, mtime_ns, count)
                    self._reconcile_transient.pop(rel, None)
                else:  # TRANSIENT — never park; count toward deprioritization at this token
                    prev = self._reconcile_transient.get(rel)
                    count = prev[2] + 1 if (prev is not None and (prev[0], prev[1]) == (size, mtime_ns)) else 1
                    self._reconcile_transient[rel] = (size, mtime_ns, count)
                    self._reconcile_attempts.pop(rel, None)
            if recovered:
                logger.info(
                    "Session watcher reconcile recovered %d transcript(s) the live path missed",
                    recovered,
                )
            return recovered
        finally:
            self._reconcile_lock.release()

    def request_stop(self) -> None:
        """Reject new ingest work and cancel debounce/retry timers."""
        self._stop_event.set()
        with self._lock:
            for timer in self._timers.values():
                timer.cancel()
            self._timers.clear()

    def is_stopping(self) -> bool:
        return self._stop_event.is_set()

    def in_flight_count(self) -> int:
        """Number of ingests that have claimed a file and not yet released it."""
        with self._lock:
            return len(self._ingesting)

    def on_created(self, event):
        if not event.is_directory and event.src_path.endswith(".jsonl"):
            path = Path(event.src_path)
            if not _is_subagent_transcript(path):
                self._schedule_ingest(path)

    def on_modified(self, event):
        if not event.is_directory and event.src_path.endswith(".jsonl"):
            path = Path(event.src_path)
            if not _is_subagent_transcript(path):
                self._schedule_ingest(path)


@dataclass
class SessionWatch:
    """A live watcher: its directory, handler, and (swappable) Observer."""
    watch_dir: Path
    handler: SessionHandler
    observer: Observer
    startup_reconcile_thread: Thread | None = None


def _run_startup_reconcile(watches: list[SessionWatch]) -> None:
    """Run the first disk-truth reconcile off the FastAPI bind path."""
    try:
        recovered = run_session_reconcile(watches)
    except Exception as e:  # noqa: BLE001 - keep background thread failures out of stderr
        logger.warning("Session watcher startup reconcile failed: %s", e)
        return
    if recovered:
        logger.info(
            "Session watcher startup reconcile recovered %d transcript(s)",
            recovered,
        )


def start_session_watcher(engine: MemoryEngine) -> list[SessionWatch]:
    """Start the session watcher for agent transcript files.

    Starts real-time observers immediately, then runs the first disk-truth reconcile off the
    FastAPI bind path. Returns list of SessionWatch for shutdown and periodic reconcile.
    """
    s = engine.settings
    if not s.session_watcher_enabled:
        return []

    watch_dirs = _session_watch_dirs(s)
    if not watch_dirs:
        logger.warning("Session watcher dir does not exist: %s", _expand_watch_dir(s.session_watcher_dir))
        return []

    watches: list[SessionWatch] = []
    try:
        for watch_dir in watch_dirs:
            handler = SessionHandler(
                engine, watch_dir, s.session_watcher_debounce_seconds, s.session_watcher_min_turns,
                s.session_watcher_idle_threshold, s.session_watcher_lookback_hours,
            )
            observer = Observer()
            observer.schedule(handler, str(watch_dir), recursive=True)
            observer.start()
            watches.append(SessionWatch(watch_dir=watch_dir, handler=handler, observer=observer))
            logger.info("Session watcher started on %s", watch_dir)

        if watches:
            startup_thread = Thread(
                target=_run_startup_reconcile,
                args=(watches,),
                name="ormah-session-startup-reconcile",
                daemon=False,
            )
            for watch in watches:
                watch.startup_reconcile_thread = startup_thread
            startup_thread.start()
    except Exception:
        stop_session_watcher(watches)
        raise

    return watches


def stop_session_watcher(watches: list[SessionWatch]) -> None:
    """Stop observers and drain in-flight transcript ingest before DB shutdown."""
    for w in watches:
        w.handler.request_stop()
    for w in watches:
        w.observer.stop()
    startup_threads = {
        w.startup_reconcile_thread
        for w in watches
        if w.startup_reconcile_thread is not None
    }
    for thread in startup_threads:
        thread.join()
    _drain_handlers([w.handler for w in watches])
    for w in watches:
        w.observer.join(timeout=5)
    if watches:
        logger.info("Session watcher stopped")


def _drain_handlers(handlers: list[SessionHandler]) -> None:
    """Wait for in-flight ingest work so it cannot touch the DB after shutdown."""
    waited = 0.0
    while any(handler.in_flight_count() > 0 for handler in handlers):
        time.sleep(0.05)
        waited += 0.05
        if waited >= 5.0:
            in_flight = sum(handler.in_flight_count() for handler in handlers)
            logger.warning(
                "Session watcher shutdown still draining %d in-flight ingest(s)",
                in_flight,
            )
            waited = 0.0


def run_session_reconcile(watches: list[SessionWatch]) -> int:
    """Periodic safety net: recreate any dead Observer, then reconcile each watcher.

    Recreating the Observer keeps the fast path alive going forward; the reconcile scan recovers
    anything the live path dropped (Observer death OR FSEvents coalescing). Returns total recovered.
    """
    total = 0
    for w in watches:
        if w.handler.is_stopping():
            continue
        try:
            alive = w.observer.is_alive()
        except Exception:  # noqa: BLE001 - a broken Observer should be treated as dead
            alive = False
        if not alive:
            logger.warning("Session watcher Observer not alive for %s; recreating", w.watch_dir)
            try:
                w.observer.stop()
                w.observer.join(timeout=5)
            except Exception as e:  # noqa: BLE001 - best-effort cleanup of a dead Observer
                logger.debug("Stopping dead Observer for %s failed: %s", w.watch_dir, e)
            try:
                observer = Observer()
                observer.schedule(w.handler, str(w.watch_dir), recursive=True)
                observer.start()
                w.observer = observer
            except Exception as e:  # noqa: BLE001 - reconcile should continue if recreate fails
                logger.warning("Failed to recreate Observer for %s: %s", w.watch_dir, e)
        total += w.handler.reconcile()
    return total

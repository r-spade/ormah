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


def _record_whisper_usage_signals(
    engine: MemoryEngine,
    transcript: TranscriptResult,
) -> int:
    """Mine transcript responses for clear usage of injected whisper memories."""
    rows = engine.db.conn.execute(
        """
        SELECT
            wl.id, wl.node_id, wl.prompt_text, wl.prompt_hash, wl.prompt_vec,
            wl.session_id, wl.space, n.title, n.content
        FROM whisper_log wl
        JOIN nodes n ON n.id = wl.node_id
        WHERE wl.session_id = ?
          AND wl.was_injected = 1
          AND NOT EXISTS (
              SELECT 1 FROM signals s
              WHERE s.whisper_log_id = wl.id
                AND s.source = 'transcript_watcher_heuristic'
          )
        ORDER BY wl.logged_at ASC, wl.id ASC
        """,
        (transcript.session_id,),
    ).fetchall()
    if not rows:
        return 0

    now_iso = datetime.now(timezone.utc).isoformat()
    recorded = 0
    with engine.db.transaction() as conn:
        for row in rows:
            response = _assistant_response_after_prompt(transcript.turns, row["prompt_text"])
            if response is None:
                continue

            referenced, strength, evidence = _node_usage_evidence(row, response)
            signal_type = "whisper_referenced" if referenced else "whisper_unreferenced"
            polarity = 1 if referenced else 0
            evidence = {
                **evidence,
                "detector": "transcript_watcher_heuristic",
                "response_chars": len(response),
            }
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
                    "transcript_watcher_heuristic",
                    row["session_id"],
                    transcript.source,
                    "transcript_watcher",
                    row["space"],
                    row["prompt_hash"],
                    json.dumps(evidence, sort_keys=True),
                    now_iso,
                ),
            )
            recorded += conn.execute("SELECT changes()").fetchone()[0]

            if referenced:
                conn.execute(
                    """
                    INSERT INTO affinity
                        (
                            prompt_vec, prompt_text, node_id, signal, source,
                            confirmed_at, space, session_id, whisper_log_id
                        )
                    VALUES (?, ?, ?, 1, ?, ?, ?, ?, ?)
                    ON CONFLICT DO NOTHING
                    """,
                    (
                        row["prompt_vec"],
                        row["prompt_text"],
                        row["node_id"],
                        "auto_heuristic",
                        now_iso,
                        row["space"],
                        row["session_id"],
                        row["id"],
                    ),
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
) -> bool:
    """Ingest a single JSONL session transcript if changed. Returns True if ingested."""
    rel = str(path.relative_to(watch_dir))

    try:
        h = _file_hash(path)
    except OSError as e:
        logger.warning("Cannot read %s: %s", path, e)
        return False

    existing = state.get(rel)
    if existing and existing.get("hash") == h:
        return False

    try:
        result = parse_transcript(path)
    except Exception as e:
        logger.warning("Session transcript parse error for %s: %s", path, e)
        return False

    if result.user_turn_count < min_turns:
        return False

    # Detect space from parent directory encoding
    space = _space_from_encoded_dir(path.parent.name)
    signals_recorded = _record_whisper_usage_signals(engine, result)

    try:
        ingested = engine.ingest_conversation(
            content=result.conversation,
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
    prev_node_ids = existing.get("node_ids", []) if existing else []

    state[rel] = {
        "hash": h,
        "last_ingested": datetime.now(timezone.utc).isoformat(),
        "session_id": result.session_id,
        "source": result.source,
        "space": space,
        "user_turns": result.user_turn_count,
        "node_ids": prev_node_ids + new_node_ids,
        "signals_recorded": signals_recorded,
    }
    _save_state(watch_dir, state)

    logger.info(
        "Session watcher ingested %s (%d turns, %d memories extracted, %d signals recorded)",
        rel, result.user_turn_count, count, signals_recorded,
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
    ) -> None:
        self.engine = engine
        self.watch_dir = watch_dir
        self.debounce_seconds = debounce_seconds
        self.min_turns = min_turns
        self._state = _load_state(watch_dir)
        self._timers: dict[str, Timer] = {}
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

    def _do_ingest(self, path: Path) -> None:
        """Actually ingest the session (called after debounce)."""
        with self._lock:
            self._timers.pop(str(path), None)
        _ingest_session(self.engine, path, self._state, self.watch_dir, self.min_turns)

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

    watch_dir = Path(s.session_watcher_dir).expanduser().resolve()
    if not watch_dir.exists():
        logger.warning("Session watcher dir does not exist: %s", watch_dir)
        return []

    # Catch-up scan
    ingested = _scan_sessions(
        engine, watch_dir, s.session_watcher_min_turns, s.session_watcher_lookback_hours,
    )
    if ingested:
        logger.info("Session watcher catch-up: ingested %d sessions from %s", ingested, watch_dir)

    # Start real-time watcher
    handler = SessionHandler(
        engine, watch_dir, s.session_watcher_debounce_seconds, s.session_watcher_min_turns,
    )
    observer = Observer()
    observer.schedule(handler, str(watch_dir), recursive=True)
    observer.start()
    logger.info("Session watcher started on %s", watch_dir)

    return [observer]


def stop_session_watcher(observers: list[Observer]) -> None:
    """Stop and join all session watcher observers."""
    for observer in observers:
        observer.stop()
    for observer in observers:
        observer.join(timeout=5)
    if observers:
        logger.info("Session watcher stopped")

"""Session watcher — auto-ingest completed Claude Code JSONL transcripts."""

from __future__ import annotations

import hashlib
import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock, Timer

from watchdog.events import FileSystemEventHandler
from watchdog.observers import Observer

from ormah.engine.memory_engine import MemoryEngine

logger = logging.getLogger(__name__)

_STATE_FILENAME = ".session_watcher_state"


def _space_from_encoded_dir(dirname: str) -> str | None:
    """Extract project space from Claude Code's encoded directory name.

    Claude Code encodes paths like ``-Users-johndoe-Projects-ormah``.
    The last segment after splitting on ``-`` is the project name.
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
    from ormah.transcript.parser import parse_transcript

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

    try:
        ingested = engine.ingest_conversation(
            content=result.conversation,
            space=space,
            agent_id="session-watcher",
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
        "space": space,
        "user_turns": result.user_turn_count,
        "node_ids": prev_node_ids + new_node_ids,
    }
    _save_state(watch_dir, state)

    logger.info(
        "Session watcher ingested %s (%d turns, %d memories extracted)",
        rel, result.user_turn_count, count,
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
    """Start the session watcher for Claude Code transcripts.

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

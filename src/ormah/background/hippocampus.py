"""Hippocampus — real-time file watching & auto-ingestion of .md files."""

from __future__ import annotations

import fnmatch
import hashlib
import json
import logging
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock, Timer

from watchdog.events import FileSystemEventHandler
from watchdog.observers import Observer

from ormah.engine.memory_engine import MemoryEngine

logger = logging.getLogger(__name__)

_STATE_FILENAME = ".hippocampus_state"


def _detect_space(path: Path) -> str | None:
    """Detect the project space from a file path.

    Tries git repo basename first, falls back to parent directory name.
    """
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            cwd=str(path.parent),
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            repo_root = result.stdout.strip()
            if repo_root:
                return Path(repo_root).name
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
        pass
    return path.parent.name or None


def _file_hash(path: Path) -> str:
    """Return SHA-256 hex digest of a file's contents."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_state(watch_dir: Path) -> dict:
    """Load persisted state for a watch directory."""
    state_path = watch_dir / _STATE_FILENAME
    if state_path.exists():
        try:
            return json.loads(state_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            logger.warning("Corrupted state file %s, starting fresh", state_path)
    return {}


def _save_state(watch_dir: Path, state: dict) -> None:
    """Persist state for a watch directory."""
    state_path = watch_dir / _STATE_FILENAME
    state_path.write_text(json.dumps(state, indent=2), encoding="utf-8")


def _matches_ignore(rel_path: str, patterns: list[str]) -> bool:
    """Return True if rel_path matches any of the ignore patterns.

    Supports ``**`` glob syntax by normalising to fnmatch semantics
    (fnmatch ``*`` already matches path separators on Unix).
    """
    for pattern in patterns:
        # Collapse ** to * (fnmatch * already matches /)
        norm = pattern.replace("**", "*")
        if fnmatch.fnmatch(rel_path, norm):
            return True
        # Also try without leading */ so **/dir/** matches dir/file
        if norm.startswith("*/") and fnmatch.fnmatch(rel_path, norm[2:]):
            return True
    return False


def _ingest_file(
    engine: MemoryEngine,
    path: Path,
    state: dict,
    watch_dir: Path,
    ignore_patterns: list[str] | None = None,
) -> bool:
    """Ingest a single file if its content has changed. Returns True if ingested."""
    rel = str(path.relative_to(watch_dir))

    if ignore_patterns and _matches_ignore(rel, ignore_patterns):
        return False

    try:
        content = path.read_text(encoding="utf-8")
    except OSError as e:
        logger.warning("Cannot read %s: %s", path, e)
        return False

    if len(content.strip()) < 10:
        return False

    h = hashlib.sha256(content.encode()).hexdigest()
    existing = state.get(rel)

    if existing and existing.get("hash") == h:
        return False

    space = _detect_space(path)
    try:
        result = engine.ingest_conversation(content=content, space=space)
        if isinstance(result, str):
            logger.warning("Hippocampus ingestion failed for %s: %s", path, result)
            return False
        count = len(result) if isinstance(result, list) else 0
    except Exception as e:
        logger.warning("Hippocampus ingestion error for %s: %s", path, e)
        return False

    # Collect node_ids from ingestion result
    new_node_ids = [m["node_id"] for m in result] if isinstance(result, list) else []

    # Append to existing node_ids (don't replace — old memories are still valid)
    prev_node_ids = existing.get("node_ids", []) if existing else []

    state[rel] = {
        "hash": h,
        "last_ingested": datetime.now(timezone.utc).isoformat(),
        "size": len(content),
        "node_ids": prev_node_ids + new_node_ids,
    }
    _save_state(watch_dir, state)

    logger.info("Hippocampus ingested %s (%d memories extracted)", rel, count)
    return True


class HippocampusHandler(FileSystemEventHandler):
    """Watches for .md file create/modify events with debouncing."""

    def __init__(
        self,
        engine: MemoryEngine,
        watch_dir: Path,
        debounce_seconds: float,
        ignore_patterns: list[str] | None = None,
    ) -> None:
        self.engine = engine
        self.watch_dir = watch_dir
        self.debounce_seconds = debounce_seconds
        self.ignore_patterns = ignore_patterns or []
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
        """Actually ingest the file (called after debounce)."""
        with self._lock:
            self._timers.pop(str(path), None)
        _ingest_file(self.engine, path, self._state, self.watch_dir, self.ignore_patterns)

    def on_created(self, event):
        if not event.is_directory and event.src_path.endswith(".md"):
            self._schedule_ingest(Path(event.src_path))

    def on_modified(self, event):
        if not event.is_directory and event.src_path.endswith(".md"):
            self._schedule_ingest(Path(event.src_path))


def _scan_directory(
    engine: MemoryEngine,
    watch_dir: Path,
    ignore_patterns: list[str] | None = None,
) -> int:
    """Scan a directory for new/changed .md files. Returns count of files ingested."""
    state = _load_state(watch_dir)
    ingested = 0
    for md_file in sorted(watch_dir.rglob("*.md")):
        if md_file.name == _STATE_FILENAME:
            continue
        if _ingest_file(engine, md_file, state, watch_dir, ignore_patterns):
            ingested += 1

    # Clean stale state entries whose files no longer exist on disk
    stale_keys = [
        rel for rel in list(state.keys())
        if not (watch_dir / rel).exists()
    ]
    for key in stale_keys:
        del state[key]
    if stale_keys:
        _save_state(watch_dir, state)

    return ingested


def start_hippocampus(engine: MemoryEngine) -> list[Observer]:
    """Start file watchers for all configured hippocampus directories.

    Performs an initial catch-up scan of each directory, then starts
    real-time watchers. Returns list of Observer instances for shutdown.
    """
    s = engine.settings
    if not s.hippocampus_enabled or not s.hippocampus_watch_dirs:
        return []

    ignore_patterns = s.hippocampus_ignore_patterns

    observers: list[Observer] = []
    for watch_dir in s.hippocampus_watch_dirs:
        watch_dir = Path(watch_dir).expanduser().resolve()
        watch_dir.mkdir(parents=True, exist_ok=True)

        # Catch-up scan
        ingested = _scan_directory(engine, watch_dir, ignore_patterns)
        if ingested:
            logger.info("Hippocampus catch-up: ingested %d files from %s", ingested, watch_dir)

        # Start real-time watcher
        handler = HippocampusHandler(engine, watch_dir, s.hippocampus_debounce_seconds, ignore_patterns)
        observer = Observer()
        observer.schedule(handler, str(watch_dir), recursive=True)
        observer.start()
        observers.append(observer)
        logger.info("Hippocampus watcher started on %s", watch_dir)

    return observers


def stop_hippocampus(observers: list[Observer]) -> None:
    """Stop and join all hippocampus observers."""
    for observer in observers:
        observer.stop()
    for observer in observers:
        observer.join(timeout=5)
    if observers:
        logger.info("Hippocampus watchers stopped")


def run_hippocampus_scan(engine: MemoryEngine) -> None:
    """One-shot scan of all watch dirs (for admin trigger / manual catch-up)."""
    s = engine.settings
    if not s.hippocampus_enabled or not s.hippocampus_watch_dirs:
        return

    ignore_patterns = s.hippocampus_ignore_patterns
    total = 0
    for watch_dir in s.hippocampus_watch_dirs:
        watch_dir = Path(watch_dir).expanduser().resolve()
        if not watch_dir.exists():
            continue
        total += _scan_directory(engine, watch_dir, ignore_patterns)

    if total:
        logger.info("Hippocampus scan: ingested %d files total", total)

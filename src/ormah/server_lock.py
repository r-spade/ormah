"""Single-writer guard: at most one Ormah process owns a memory directory.

The server keeps an in-memory id-to-path cache and an in-process lock, neither of which
another process can see. Two servers on one directory (for example a dev server started with
a different ``ORMAH_PORT``) therefore race each other on the same files. This module takes an
exclusive, non-blocking OS lock for the lifetime of the server so the second one refuses to
start.

The lock lives in ``<memory_dir>/.ormah/server.lock``. It is deliberately separate from
``store.lock`` (see ``ormah.cloud.store_lock``), which is taken briefly and repeatedly by cloud
protection code from other threads and processes while the server runs.

The kernel releases a ``flock`` when the owning process dies, so a crash never leaves a stale
lock. ``server.json`` next to it records who holds the lock purely so the refusal message can
name the holder. It is informational: the lock is the truth and stale contents are harmless.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from filelock import FileLock, Timeout as FileLockTimeout

from ormah.cloud.store_lock import canonical_memory_dir

logger = logging.getLogger(__name__)

ROLE_SERVER = "server"
ROLE_RESTORE = "restore"


class ServerLockHeld(RuntimeError):
    """Raised when another Ormah process already owns the memory directory."""

    def __init__(self, memory_dir: Path, holder: dict[str, Any] | None = None) -> None:
        self.memory_dir = memory_dir
        self.holder = holder
        super().__init__(self._message())

    def _message(self) -> str:
        holder = self.holder or {}
        pid = holder.get("pid")
        role = holder.get("role", ROLE_SERVER)
        if role == ROLE_RESTORE:
            who = "an Ormah backup restore"
        else:
            who = "another Ormah server"
        details = []
        if pid is not None:
            details.append(f"pid {pid}")
        if holder.get("port") is not None:
            details.append(f"port {holder['port']}")
        if holder.get("started_at"):
            details.append(f"started {holder['started_at']}")
        suffix = f" ({', '.join(details)})" if details else ""
        hint = (
            "Only one Ormah server can use a memory directory at a time. "
            "Stop it, or give this one its own directory with ORMAH_MEMORY_DIR."
        )
        if role == ROLE_RESTORE:
            hint = "Wait for the restore to finish and try again."
        return f"{who}{suffix} is already using {self.memory_dir}. {hint}"


def server_lock_path(memory_dir: Path) -> Path:
    return canonical_memory_dir(memory_dir) / ".ormah" / "server.lock"


def _holder_path(memory_dir: Path) -> Path:
    return canonical_memory_dir(memory_dir) / ".ormah" / "server.json"


def _pid_alive(pid: object) -> bool:
    if not isinstance(pid, int) or pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def read_holder(memory_dir: Path) -> dict[str, Any] | None:
    """Best-effort read of the recorded holder. Never trust it as proof of ownership."""
    try:
        data = json.loads(_holder_path(memory_dir).read_text())
    except (OSError, ValueError):
        return None
    if not isinstance(data, dict) or not _pid_alive(data.get("pid")):
        return None
    return data


class ServerLock:
    """Exclusive, non-blocking, process-lifetime lock on one memory directory."""

    def __init__(
        self,
        memory_dir: Path,
        *,
        port: int | None = None,
        role: str = ROLE_SERVER,
    ) -> None:
        self.memory_dir = canonical_memory_dir(memory_dir)
        self.path = server_lock_path(self.memory_dir)
        self.holder_path = _holder_path(self.memory_dir)
        self.port = port
        self.role = role
        self._lock = FileLock(str(self.path))
        self._held = False
        self._recorded = False

    @property
    def held(self) -> bool:
        return self._held

    def acquire(self, *, record: bool = True) -> ServerLock:
        """Take the lock or raise ``ServerLockHeld`` immediately.

        ``record=False`` skips writing the holder sidecar (used by throwaway probes).
        """
        if self._held:
            raise RuntimeError("ServerLock is already held by this object")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        try:
            self._lock.acquire(timeout=0)
        except FileLockTimeout:
            raise ServerLockHeld(self.memory_dir, read_holder(self.memory_dir)) from None
        self._held = True
        try:
            os.chmod(self.path, 0o600)
        except OSError:
            pass
        self._recorded = record
        if record:
            self._write_holder()
        return self

    def release(self) -> None:
        if not self._held:
            return
        self._held = False
        # Drop the sidecar while still holding the lock so it cannot clobber a successor's.
        if self._recorded:
            try:
                self.holder_path.unlink()
            except OSError:
                pass
        self._lock.release()

    def _write_holder(self) -> None:
        record = {
            "pid": os.getpid(),
            "port": self.port,
            "role": self.role,
            "started_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }
        tmp = self.holder_path.with_suffix(f".json.{os.getpid()}.tmp")
        try:
            tmp.write_text(json.dumps(record))
            os.replace(tmp, self.holder_path)
        except OSError:
            logger.debug("Could not record server lock holder", exc_info=True)
            try:
                tmp.unlink()
            except OSError:
                pass

    def __enter__(self) -> ServerLock:
        return self.acquire()

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.release()


def check_available(memory_dir: Path) -> None:
    """Raise ``ServerLockHeld`` if the directory is owned, without keeping the lock.

    A pre-flight for the CLI so the user sees a friendly message instead of a startup
    traceback. The lock taken in the server lifespan is the real guard.
    """
    lock = ServerLock(memory_dir)
    lock.acquire(record=False)
    lock.release()

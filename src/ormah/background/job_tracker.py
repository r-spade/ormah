"""Track background job execution status for observability."""

from __future__ import annotations

import contextlib
import logging
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Callable

logger = logging.getLogger(__name__)


@dataclass
class JobStatus:
    """Snapshot of a single job's health."""

    last_run: datetime | None = None
    last_success: datetime | None = None
    last_error: str | None = None
    last_error_time: datetime | None = None
    run_count: int = 0
    error_count: int = 0
    last_duration_ms: float = 0.0


class JobTracker:
    """Thread-safe registry of background job execution outcomes."""

    def __init__(self) -> None:
        self._jobs: dict[str, JobStatus] = {}
        self._lock = threading.Lock()
        self._running: set[str] = set()

    def record_success(self, job_id: str, duration_ms: float) -> None:
        now = datetime.now(timezone.utc)
        with self._lock:
            status = self._jobs.setdefault(job_id, JobStatus())
            status.last_run = now
            status.last_success = now
            status.run_count += 1
            status.last_duration_ms = duration_ms

    def record_failure(self, job_id: str, error: str, duration_ms: float) -> None:
        now = datetime.now(timezone.utc)
        with self._lock:
            status = self._jobs.setdefault(job_id, JobStatus())
            status.last_run = now
            status.last_error = error
            status.last_error_time = now
            status.run_count += 1
            status.error_count += 1
            status.last_duration_ms = duration_ms

    def is_running(self, job_id: str) -> bool:
        with self._lock:
            return job_id in self._running

    @contextlib.contextmanager
    def run_guard(self, job_id: str):
        """Yield True if this caller claimed the job, False if it was already running.

        An edge-writing job (auto_linker, conflict_detector) must never run twice at
        once: both instances read the same watermark, enumerate the same candidates,
        and race each other's edge writes (#117). APScheduler's max_instances=1 covers
        the scheduled path; this covers the manual admin routes, which call the
        runners directly and never touch the scheduler.
        """
        with self._lock:
            acquired = job_id not in self._running
            if acquired:
                self._running.add(job_id)
        try:
            yield acquired
        finally:
            if acquired:
                with self._lock:
                    self._running.discard(job_id)

    def snapshot(self) -> dict[str, dict[str, Any]]:
        """Return a JSON-serialisable snapshot of all job statuses."""
        with self._lock:
            result = {}
            for job_id, s in self._jobs.items():
                result[job_id] = {
                    "last_run": s.last_run.isoformat() if s.last_run else None,
                    "last_success": s.last_success.isoformat() if s.last_success else None,
                    "last_error": s.last_error,
                    "last_error_time": s.last_error_time.isoformat() if s.last_error_time else None,
                    "run_count": s.run_count,
                    "error_count": s.error_count,
                    "last_duration_ms": round(s.last_duration_ms, 1),
                }
            return result


def failure_reason(result: Any) -> str | None:
    """Return why a runner's result means failure, or None if it succeeded.

    Runners do not raise on failure — they signal it in the return value, and they do
    not all use the same shape:
      * the background jobs return ``{"error": ...}``
      * ``run_restore_verification`` returns a plain ``False``

    Treating everything that is not an error dict as a success recorded a failed run as
    healthy, which is how the #117 outage stayed invisible for a day. ``None`` and
    ``True`` stay successes: most runners return nothing at all.
    """
    if result is False:
        return "run reported failure"
    if isinstance(result, dict) and "error" in result:
        return str(result["error"])
    return None


def tracked(tracker: JobTracker, job_id: str, fn: Callable, *args: Any) -> Callable:
    """Wrap a job function with tracking. Returns a no-arg callable for the scheduler."""

    def _wrapper():
        t0 = time.monotonic()
        with tracker.run_guard(job_id) as acquired:
            if not acquired:
                logger.warning("Job %s is already running; skipping this trigger", job_id)
                return
            try:
                result = fn(*args)
                duration_ms = (time.monotonic() - t0) * 1000
                # The runners catch their own exceptions and signal failure in the RETURN
                # value. Discarding it recorded a dead run as a success, so /admin/health
                # kept reporting ok (#117).
                reason = failure_reason(result)
                if reason is not None:
                    tracker.record_failure(job_id, reason, duration_ms)
                    logger.warning(
                        "Job %s reported an error after %.0fms: %s", job_id, duration_ms, reason
                    )
                else:
                    tracker.record_success(job_id, duration_ms)
            except Exception as e:
                duration_ms = (time.monotonic() - t0) * 1000
                tracker.record_failure(job_id, str(e), duration_ms)
                logger.warning("Job %s failed after %.0fms: %s", job_id, duration_ms, e)

    return _wrapper

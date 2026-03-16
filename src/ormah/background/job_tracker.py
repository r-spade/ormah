"""Track background job execution status for observability."""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
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


def tracked(tracker: JobTracker, job_id: str, fn: Callable, *args: Any) -> Callable:
    """Wrap a job function with tracking. Returns a no-arg callable for the scheduler."""

    def _wrapper():
        t0 = time.monotonic()
        try:
            fn(*args)
            duration_ms = (time.monotonic() - t0) * 1000
            tracker.record_success(job_id, duration_ms)
        except Exception as e:
            duration_ms = (time.monotonic() - t0) * 1000
            tracker.record_failure(job_id, str(e), duration_ms)
            logger.warning("Job %s failed after %.0fms: %s", job_id, duration_ms, e)

    return _wrapper

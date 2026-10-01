"""Background execution manager for agent-driven maintenance."""

from __future__ import annotations

import copy
import logging
import threading
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Callable

from ormah.background.job_tracker import JobTracker
from ormah.engine.memory_engine import MemoryEngine

logger = logging.getLogger(__name__)

_ACTIVE_STATUSES = {"running_phase1", "awaiting_results", "running_phase2"}


@dataclass
class MaintenanceJob:
    """In-memory state for a single maintenance run."""

    job_id: str
    status: str
    started_at: datetime
    phase: str
    batches: dict[str, Any] | None = None
    apply_summary: dict[str, Any] | None = None
    finished_at: datetime | None = None
    last_error: str | None = None
    phase1_finished_at: datetime | None = None
    phase2_started_at: datetime | None = None
    deadline: float | None = None
    expires_at: datetime | None = None


class MaintenanceManager:
    """Run maintenance phases in background threads with single-flight semantics."""

    def __init__(
        self,
        engine: MemoryEngine,
        tracker: JobTracker | None = None,
        *,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._engine = engine
        self._tracker = tracker
        self._lock = threading.Lock()
        self._job: MaintenanceJob | None = None
        self._clock = clock
        self._timeout_seconds = getattr(
            getattr(engine, "settings", None), "maintenance_timeout_minutes", 30
        ) * 60
        engine.maintenance_is_active = self.has_active_reservation

    def _expire_locked(self) -> None:
        """Expire analysis only. Preparation and application keep the reservation."""
        job = self._job
        if (
            job is not None
            and job.status == "awaiting_results"
            and job.deadline is not None
            and self._clock() >= job.deadline
        ):
            job.status = "expired"
            job.finished_at = datetime.now(timezone.utc)
            job.batches = None
            job.last_error = "Maintenance assignment expired. Discard stale analysis; stop this run."

    def has_active_reservation(self) -> bool:
        with self._lock:
            self._expire_locked()
            return self._job is not None and self._job.status in _ACTIVE_STATUSES

    def start_phase1(self) -> dict[str, Any]:
        """Atomically reserve phase 1; only the winner receives an assignment receipt."""
        with self._lock:
            self._expire_locked()
            if self._job is not None and self._job.status in _ACTIVE_STATUSES:
                return {
                    "status": "busy",
                    "message": "Another maintenance run is underway. Stop this run.",
                }

            job = MaintenanceJob(
                job_id=str(uuid.uuid4()),
                status="running_phase1",
                phase="phase1",
                started_at=datetime.now(timezone.utc),
            )
            self._job = job
            payload = self._serialize(job)

        try:
            threading.Thread(
                target=self._run_phase1,
                args=(job.job_id,),
                name=f"ormah-maintenance-phase1-{job.job_id[:8]}",
                daemon=True,
            ).start()
        except Exception as exc:
            self._record_failure(job.job_id, "phase1", exc, 0)
            return self.get_status(job.job_id)
        return payload

    def submit_results(self, results: dict[str, Any], job_id: str | None = None) -> dict[str, Any]:
        """Start phase 2 for the current prepared job."""
        with self._lock:
            self._expire_locked()
            if not job_id:
                raise ValueError(
                    "job_id is required with results; echo the Phase 1 assignment receipt."
                )
            job = self._job
            if job is None:
                raise LookupError(
                    "Maintenance assignment was lost. Discard stale analysis; stop this run."
                )
            if job.job_id != job_id:
                raise ValueError("Maintenance job mismatch. Discard stale analysis; stop this run.")
            if job.status == "running_phase1":
                raise RuntimeError("Maintenance batches are still being prepared")
            if job.status == "running_phase2":
                raise RuntimeError(
                    "Maintenance results are already applying. Poll this job_id; do not resubmit."
                )
            if job.status == "expired":
                raise RuntimeError(job.last_error)
            if job.status != "awaiting_results":
                raise RuntimeError(
                    f"Maintenance job is {job.status}. Results were not accepted; stop this run."
                )

            results = copy.deepcopy(results)
            job.status = "running_phase2"
            job.phase = "phase2"
            job.phase2_started_at = datetime.now(timezone.utc)
            payload = self._serialize(job)

        try:
            threading.Thread(
                target=self._run_phase2,
                args=(job.job_id, results),
                name=f"ormah-maintenance-phase2-{job.job_id[:8]}",
                daemon=True,
            ).start()
        except Exception as exc:
            self._record_failure(job.job_id, "phase2", exc, 0)
            return self.get_status(job.job_id)
        return payload

    def get_status(self, job_id: str | None = None) -> dict[str, Any]:
        """Return the current maintenance job state."""
        with self._lock:
            self._expire_locked()
            if self._job is None:
                return {
                    "status": "idle",
                    "message": "No maintenance assignment exists. Discard stale analysis; stop this run.",
                }
            if job_id and self._job.job_id != job_id:
                return {
                    "status": "replaced",
                    "job_id": job_id,
                    "message": "Maintenance assignment was replaced. Discard stale analysis; stop this run.",
                }
            return self._serialize(self._job)

    def _run_phase1(self, job_id: str) -> None:
        t0 = time.monotonic()
        try:
            batches = self._engine.get_maintenance_batches()
            duration_ms = (time.monotonic() - t0) * 1000
            with self._lock:
                if self._job is None or self._job.job_id != job_id:
                    return
                self._job.status = "awaiting_results"
                self._job.phase = "phase1"
                self._job.batches = batches
                self._job.phase1_finished_at = datetime.now(timezone.utc)
                self._job.deadline = self._clock() + self._timeout_seconds
                self._job.expires_at = self._job.phase1_finished_at + timedelta(
                    seconds=self._timeout_seconds
                )
            if self._tracker is not None:
                self._tracker.record_success("maintenance_phase1", duration_ms)
            logger.info(
                "Maintenance phase 1 ready: %s in %.0fms (%s)",
                job_id[:8],
                duration_ms,
                batches.get("summary", "no summary"),
            )
        except Exception as exc:
            self._record_failure(job_id, "phase1", exc, time.monotonic() - t0)

    def _run_phase2(self, job_id: str, results: dict[str, Any]) -> None:
        t0 = time.monotonic()
        try:
            summary = self._engine.apply_maintenance_results(results)
            duration_ms = (time.monotonic() - t0) * 1000
            with self._lock:
                if self._job is None or self._job.job_id != job_id:
                    return
                self._job.status = "completed"
                self._job.phase = "phase2"
                self._job.apply_summary = summary
                self._job.finished_at = datetime.now(timezone.utc)
            if self._tracker is not None:
                self._tracker.record_success("maintenance_phase2", duration_ms)
            logger.info("Maintenance phase 2 applied: %s in %.0fms", job_id[:8], duration_ms)
        except Exception as exc:
            self._record_failure(job_id, "phase2", exc, time.monotonic() - t0)

    def _record_failure(self, job_id: str, phase: str, exc: Exception, duration_s: float) -> None:
        duration_ms = duration_s * 1000
        with self._lock:
            if self._job is None or self._job.job_id != job_id:
                return
            # A late preparation/telemetry error must not release an applying job
            # or turn an already published success into a failure.
            if self._job.status != f"running_{phase}":
                return
            self._job.status = "failed"
            self._job.phase = phase
            self._job.last_error = str(exc)
            self._job.finished_at = datetime.now(timezone.utc)
        if self._tracker is not None:
            self._tracker.record_failure(f"maintenance_{phase}", str(exc), duration_ms)
        logger.warning("Maintenance %s failed for %s after %.0fms: %s", phase, job_id[:8], duration_ms, exc)

    @staticmethod
    def _serialize(job: MaintenanceJob) -> dict[str, Any]:
        data: dict[str, Any] = {
            "job_id": job.job_id,
            "status": job.status,
            "phase": job.phase,
            "started_at": job.started_at.isoformat(),
            "phase1_finished_at": (
                job.phase1_finished_at.isoformat() if job.phase1_finished_at else None
            ),
            "phase2_started_at": (
                job.phase2_started_at.isoformat() if job.phase2_started_at else None
            ),
            "finished_at": job.finished_at.isoformat() if job.finished_at else None,
            "last_error": job.last_error,
            "expires_at": job.expires_at.isoformat() if job.expires_at else None,
        }
        if job.status == "awaiting_results" and job.batches is not None:
            data["batches"] = job.batches
        if job.status == "completed" and job.apply_summary is not None:
            data["apply_summary"] = job.apply_summary
        return data

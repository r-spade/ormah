"""Background execution manager for agent-driven maintenance."""

from __future__ import annotations

import copy
import logging
import threading
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

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


class MaintenanceManager:
    """Run maintenance phases in background threads with single-flight semantics."""

    def __init__(self, engine: MemoryEngine, tracker: JobTracker | None = None) -> None:
        self._engine = engine
        self._tracker = tracker
        self._lock = threading.Lock()
        self._job: MaintenanceJob | None = None

    def start_phase1(self) -> dict[str, Any]:
        """Start phase 1 if needed, or return the existing job state."""
        with self._lock:
            if self._job is not None and self._job.status in _ACTIVE_STATUSES:
                return self._serialize(self._job)

            job = MaintenanceJob(
                job_id=str(uuid.uuid4()),
                status="running_phase1",
                phase="phase1",
                started_at=datetime.now(timezone.utc),
            )
            self._job = job

        threading.Thread(
            target=self._run_phase1,
            args=(job.job_id,),
            name=f"ormah-maintenance-phase1-{job.job_id[:8]}",
            daemon=True,
        ).start()
        return self._serialize(job)

    def submit_results(self, results: dict[str, Any], job_id: str | None = None) -> dict[str, Any]:
        """Start phase 2 for the current prepared job."""
        with self._lock:
            job = self._job
            if job is None:
                raise LookupError("No maintenance job has been started")
            if job_id and job.job_id != job_id:
                raise ValueError("Maintenance results do not match the active job")
            if job.status == "running_phase1":
                raise RuntimeError("Maintenance batches are still being prepared")
            if job.status == "running_phase2":
                return self._serialize(job)
            if job.status != "awaiting_results":
                raise RuntimeError(f"Maintenance job is not awaiting results (current status: {job.status})")

            job.status = "running_phase2"
            job.phase = "phase2"
            job.phase2_started_at = datetime.now(timezone.utc)

        threading.Thread(
            target=self._run_phase2,
            args=(job.job_id, copy.deepcopy(results)),
            name=f"ormah-maintenance-phase2-{job.job_id[:8]}",
            daemon=True,
        ).start()
        return self._serialize(job)

    def get_status(self, job_id: str | None = None) -> dict[str, Any]:
        """Return the current maintenance job state."""
        with self._lock:
            if self._job is None:
                return {"status": "idle"}
            if job_id and self._job.job_id != job_id:
                return {"status": "idle", "job_id": job_id}
            return self._serialize(self._job)

    def _run_phase1(self, job_id: str) -> None:
        t0 = time.monotonic()
        try:
            batches = self._engine.get_maintenance_batches()
            duration_ms = (time.monotonic() - t0) * 1000
            # batch_errors is additive: healthy batches still flow to phase 2
            # (status stays "awaiting_results", job.batches keeps all four
            # keys), but a finder failure must not look like a clean tracker
            # success — /admin is the #90 observability surface.
            errors = batches.get("batch_errors") or {}
            with self._lock:
                if self._job is None or self._job.job_id != job_id:
                    return
                self._job.status = "awaiting_results"
                self._job.phase = "phase1"
                self._job.batches = batches
                self._job.phase1_finished_at = datetime.now(timezone.utc)
            if self._tracker is not None:
                if errors:
                    self._tracker.record_failure(
                        "maintenance_phase1",
                        f"finder failures: {', '.join(sorted(errors))}",
                        duration_ms,
                        stats={"batch_errors": errors},
                    )
                else:
                    self._tracker.record_success("maintenance_phase1", duration_ms)
            if errors:
                logger.warning(
                    "Maintenance phase 1 ready with failed batches: %s in %.0fms (%s)",
                    job_id[:8],
                    duration_ms,
                    ", ".join(sorted(errors)),
                )
            else:
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
        }
        if job.status == "awaiting_results" and job.batches is not None:
            data["batches"] = job.batches
        if job.status == "completed" and job.apply_summary is not None:
            data["apply_summary"] = job.apply_summary
        return data

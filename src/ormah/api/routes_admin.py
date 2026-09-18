"""Admin API routes for maintenance operations."""

from __future__ import annotations

import contextlib
import logging
from pathlib import Path

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/admin", tags=["admin"])


class BackupSettingsUpdate(BaseModel):
    backup_dir: str = Field(min_length=1)
    retention_count: int = Field(ge=1)


# Map of task IDs to their runner functions (module path, function name)
_TASK_RUNNERS = {
    "auto_linker": ("ormah.background.auto_linker", "run_auto_linker"),
    "decay_manager": ("ormah.background.decay_manager", "run_decay"),
    "conflict_detector": ("ormah.background.conflict_detector", "run_conflict_detection"),
    "duplicate_merger": ("ormah.background.duplicate_merger", "run_duplicate_detection"),
    "auto_cluster": ("ormah.background.auto_cluster", "run_auto_cluster"),
    "hippocampus": ("ormah.background.hippocampus", "run_hippocampus_scan"),
    "importance_scorer": ("ormah.background.importance_scorer", "run_importance_scoring"),
    "consolidator": ("ormah.background.consolidator", "run_consolidation"),
    "memory_backup": ("ormah.backup", "run_auto_backup"),
    "cloud_backup": ("ormah.cloud.jobs", "run_cloud_backup"),
    "restore_verification": ("ormah.cloud.jobs", "run_restore_verification"),
}

_TASK_DESCRIPTIONS = {
    "importance_scorer": "Recalculates importance scores for all memories based on connections, access patterns, and usefulness.",
    "index_updater": "Incrementally updates the full-text and vector search indexes with new or modified memories.",
    "duplicate_merger": "Detects near-duplicate memories and creates merge proposals for review.",
    "conflict_detector": "Finds contradicting memories and creates conflict proposals for resolution.",
    "auto_linker": "Discovers and creates edges between semantically related memories.",
    "auto_cluster": "Groups memories into clusters based on semantic similarity and tags.",
    "consolidator": "Merges or summarizes redundant working-tier memories into consolidated entries.",
    "decay_manager": "Demotes working memories to archival when FSRS retrievability falls below threshold.",
    "hippocampus": "Scans for structural patterns and promotes frequently accessed working memories to core.",
    "memory_backup": "Creates a local backup of memory source files when one is due.",
    "cloud_backup": "Encrypts and uploads a due cloud backup without changing the sync head.",
    "restore_verification": "Downloads, decrypts, rebuilds, and searches the latest cloud backup.",
}

# Order for sleep cycle (full maintenance pass)
_SLEEP_CYCLE_ORDER = [
    "importance_scorer",
    "index_updater",
    "duplicate_merger",
    "conflict_detector",
    "auto_linker",
    "auto_cluster",
    "consolidator",
    "decay_manager",
    "memory_backup",
]


def _backup_to_dict(backup) -> dict:
    return {
        "name": backup.name,
        "path": str(backup.path),
        "created_at": backup.created_at.isoformat(),
        "node_count": backup.node_count,
        "deleted_count": backup.deleted_count,
        "size_bytes": backup.size_bytes,
    }


def _backup_status_payload(settings, service) -> dict:
    latest = service.latest()
    has_memory = service.has_backupable_memory()
    due = has_memory and service.backup_due(interval_hours=settings.backup_interval_hours)
    return {
        "enabled": settings.backup_enabled,
        "backup_dir": str(settings.backup_dir),
        "interval_hours": settings.backup_interval_hours,
        "retention_count": settings.backup_retention_count,
        "has_backupable_memory": has_memory,
        "due": due,
        "latest": _backup_to_dict(latest) if latest else None,
    }


def _backup_service_from_request(request: Request):
    from ormah.backup import service_from_settings

    engine = request.app.state.engine
    return engine.settings, service_from_settings(engine.settings)


def _persist_backup_settings(backup_dir: Path, retention_count: int) -> None:
    from ormah.cloud.settings import update_settings_env

    update_settings_env(
        {
            "ORMAH_BACKUP_DIR": str(backup_dir),
            "ORMAH_BACKUP_RETENTION_COUNT": str(retention_count),
        }
    )


@router.get("/health")
def health(request: Request):
    tracker = getattr(request.app.state, "job_tracker", None)
    result: dict = {"status": "ok"}
    if tracker is not None:
        result["jobs"] = tracker.snapshot()
    manager = getattr(request.app.state, "maintenance_manager", None)
    if manager is not None:
        result["maintenance"] = manager.get_status()
    return result


@router.get("/maintenance-status")
def maintenance_status(request: Request):
    manager = getattr(request.app.state, "maintenance_manager", None)
    if manager is None:
        return {"status": "idle"}
    return manager.get_status()


@router.get("/backup")
def backup_status(request: Request):
    """Return local backup configuration and latest backup metadata."""
    settings, service = _backup_service_from_request(request)
    return _backup_status_payload(settings, service)


@router.get("/cloud-status")
def cloud_status(request: Request):
    """Return per-store cloud backup and restore-verification health."""
    from ormah.cloud.state import cloud_status_payload

    return cloud_status_payload(request.app.state.engine.settings)


@router.post("/backup/create")
def create_backup(request: Request):
    """Create a manual local backup of source-of-truth memory files."""
    from ormah.backup import BackupError

    settings, service = _backup_service_from_request(request)
    try:
        backup = service.create(reason="manual-ui")
    except BackupError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {
        "status": "created",
        "backup": _backup_to_dict(backup),
        "backup_status": _backup_status_payload(settings, service),
    }


@router.post("/backup/settings")
def update_backup_settings(body: BackupSettingsUpdate, request: Request):
    """Update local backup settings for this server and persist them to config."""
    from ormah.backup import service_from_settings

    settings = request.app.state.engine.settings
    try:
        backup_dir = Path(body.backup_dir).expanduser().resolve()
    except OSError as exc:
        raise HTTPException(status_code=400, detail=f"Invalid backup directory: {exc}") from exc

    settings.backup_dir = backup_dir
    settings.backup_retention_count = body.retention_count
    _persist_backup_settings(backup_dir, body.retention_count)
    service = service_from_settings(settings)
    return {
        "status": "updated",
        "backup_status": _backup_status_payload(settings, service),
    }


@router.post("/rebuild")
def rebuild_index(request: Request):
    engine = request.app.state.engine
    count = engine.rebuild_index()
    return {"status": "rebuilt", "nodes_indexed": count}


@router.get("/tasks")
def list_tasks(request: Request):
    """List all registered background tasks and their next run time."""
    scheduler = getattr(request.app.state, "scheduler", None)
    if scheduler is None:
        return {"tasks": [], "note": "Scheduler not running"}
    jobs = []
    for job in scheduler.get_jobs():
        jobs.append({
            "id": job.id,
            "name": job.name,
            "next_run": str(job.next_run_time) if job.next_run_time else None,
            "description": _TASK_DESCRIPTIONS.get(job.id),
            "paused": job.next_run_time is None,
        })
    return {"tasks": jobs}


@router.post("/tasks/{task_id}/pause")
def pause_task(task_id: str, request: Request):
    """Pause a background task by ID."""
    scheduler = getattr(request.app.state, "scheduler", None)
    if scheduler is None:
        raise HTTPException(status_code=503, detail="Scheduler not running")
    try:
        scheduler.pause_job(task_id)
    except Exception as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    return {"status": "paused", "task": task_id}


@router.post("/tasks/{task_id}/resume")
def resume_task(task_id: str, request: Request):
    """Resume a paused background task by ID."""
    scheduler = getattr(request.app.state, "scheduler", None)
    if scheduler is None:
        raise HTTPException(status_code=503, detail="Scheduler not running")
    try:
        scheduler.resume_job(task_id)
    except Exception as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    return {"status": "resumed", "task": task_id}


@router.post("/tasks/pause-all")
def pause_all_tasks(request: Request):
    """Pause all background tasks."""
    scheduler = getattr(request.app.state, "scheduler", None)
    if scheduler is None:
        raise HTTPException(status_code=503, detail="Scheduler not running")
    for job in scheduler.get_jobs():
        scheduler.pause_job(job.id)
    return {"status": "all_paused"}


@router.post("/tasks/resume-all")
def resume_all_tasks(request: Request):
    """Resume all background tasks."""
    scheduler = getattr(request.app.state, "scheduler", None)
    if scheduler is None:
        raise HTTPException(status_code=503, detail="Scheduler not running")
    for job in scheduler.get_jobs():
        scheduler.resume_job(job.id)
    return {"status": "all_resumed"}


@router.post("/tasks/{task_id}/run")
def run_task(task_id: str, request: Request):
    """Manually trigger a background task by ID."""
    engine = request.app.state.engine
    tracker = getattr(request.app.state, "job_tracker", None)

    # index_updater is a method on engine.builder, not a standalone function
    if task_id == "index_updater":
        with _guard(tracker, task_id) as acquired:
            if not acquired:
                raise HTTPException(status_code=409, detail=f"Task {task_id} is already running")
            added, updated = engine.builder.incremental_update()
            return {"status": "completed", "task": task_id, "added": added, "updated": updated}

    if task_id not in _TASK_RUNNERS:
        raise HTTPException(status_code=404, detail=f"Unknown task: {task_id}. Available: {list(_TASK_RUNNERS.keys()) + ['index_updater']}")

    import importlib
    module_path, func_name = _TASK_RUNNERS[task_id]
    module = importlib.import_module(module_path)
    runner = getattr(module, func_name)

    # This route bypasses the scheduler, so APScheduler's max_instances=1 does not
    # apply to it. Without this guard a manual trigger during a scheduled run starts a
    # second concurrent run over the same watermark, and the two race each other's
    # edge writes (#117).
    with _guard(tracker, task_id) as acquired:
        if not acquired:
            raise HTTPException(status_code=409, detail=f"Task {task_id} is already running")
        return _run_and_report(task_id, runner, engine)


def _run_and_report(task_id: str, runner, engine) -> dict:
    """Run a task and report what actually happened.

    The route used to return {"status": "completed"} unconditionally: a run that
    raised, or that returned {"error": ...}, was reported to the caller as a success.
    """
    from ormah.background.job_tracker import failure_reason

    try:
        result = runner(engine)
    except Exception as e:
        logger.warning("Manual run of %s failed: %s", task_id, e)
        raise HTTPException(status_code=500, detail=str(e)) from e

    # Same classifier the tracker uses: runners do not raise, they signal failure in the
    # return value, and not all of them use the same shape (dict with "error", or False).
    reason = failure_reason(result)
    if reason is not None:
        raise HTTPException(status_code=500, detail=reason)

    payload = {"status": "completed", "task": task_id}
    if isinstance(result, dict):
        payload["stats"] = result
    return payload


@contextlib.contextmanager
def _guard(tracker, task_id: str):
    """Claim task_id for the duration of the block, or yield False if it is running.

    A missing tracker (scheduler never started) means no scheduled job can be running
    either, so there is nothing to collide with — but two concurrent HTTP requests
    still could, so the app always builds a JobTracker (main.py), tracker=None is only
    reachable in tests.
    """
    if tracker is None:
        yield True
        return
    with tracker.run_guard(task_id) as acquired:
        yield acquired


@router.post("/tasks/run-all")
def run_all_tasks(request: Request):
    """Run all background tasks sequentially in sleep-cycle order."""
    import importlib

    from ormah.background.job_tracker import failure_reason

    engine = request.app.state.engine
    tracker = getattr(request.app.state, "job_tracker", None)
    results: dict[str, str] = {}

    for task_id in _SLEEP_CYCLE_ORDER:
        with _guard(tracker, task_id) as acquired:
            if not acquired:
                # Same hole as run_task: this bypasses the scheduler, so a job already
                # in flight would get a second concurrent run over the same watermark,
                # racing its edge and markdown writes (#117).
                results[task_id] = "skipped: already running"
                continue
            try:
                if task_id == "index_updater":
                    engine.builder.incremental_update()
                elif task_id in _TASK_RUNNERS:
                    module_path, func_name = _TASK_RUNNERS[task_id]
                    module = importlib.import_module(module_path)
                    runner = getattr(module, func_name)
                    reason = failure_reason(runner(engine))
                    if reason is not None:
                        results[task_id] = f"error: {reason}"
                        continue
                results[task_id] = "ok"
            except Exception as exc:
                results[task_id] = f"error: {exc}"

    has_errors = any(v.startswith("error:") for v in results.values())
    if has_errors:
        return JSONResponse(status_code=503, content={"status": "degraded", "results": results})
    # A skipped task is not an error, but the cycle did NOT run everything it was asked
    # to. Reporting "completed" would tell a cron/script that the sleep cycle ran in full
    # when part of it never started.
    if any(v.startswith("skipped:") for v in results.values()):
        return {"status": "partial", "results": results}
    return {"status": "completed", "results": results}

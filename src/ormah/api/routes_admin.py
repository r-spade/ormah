"""Admin API routes for maintenance operations."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request

router = APIRouter(prefix="/admin", tags=["admin"])

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
}

_TASK_DESCRIPTIONS = {
    "importance_scorer": "Recalculates importance scores for all memories based on connections, access patterns, and usefulness.",
    "index_updater": "Incrementally updates the full-text and vector search indexes with new or modified memories.",
    "duplicate_merger": "Detects near-duplicate memories and creates merge proposals for review.",
    "conflict_detector": "Finds contradicting memories and creates conflict proposals for resolution.",
    "auto_linker": "Discovers and creates edges between semantically related memories.",
    "auto_cluster": "Groups memories into clusters based on semantic similarity and tags.",
    "consolidator": "Merges or summarizes redundant working-tier memories into consolidated entries.",
    "decay_manager": "Applies time-based decay to memory importance, demoting stale unused memories.",
    "hippocampus": "Scans for structural patterns and promotes frequently accessed working memories to core.",
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
]


@router.get("/health")
async def health(request: Request):
    tracker = getattr(request.app.state, "job_tracker", None)
    result: dict = {"status": "ok"}
    if tracker is not None:
        result["jobs"] = tracker.snapshot()
    return result


@router.get("/stats")
async def stats(request: Request):
    engine = request.app.state.engine
    return engine.stats()


@router.post("/rebuild")
async def rebuild_index(request: Request):
    engine = request.app.state.engine
    count = engine.rebuild_index()
    return {"status": "rebuilt", "nodes_indexed": count}


@router.get("/tasks")
async def list_tasks(request: Request):
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
async def pause_task(task_id: str, request: Request):
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
async def resume_task(task_id: str, request: Request):
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
async def pause_all_tasks(request: Request):
    """Pause all background tasks."""
    scheduler = getattr(request.app.state, "scheduler", None)
    if scheduler is None:
        raise HTTPException(status_code=503, detail="Scheduler not running")
    for job in scheduler.get_jobs():
        scheduler.pause_job(job.id)
    return {"status": "all_paused"}


@router.post("/tasks/resume-all")
async def resume_all_tasks(request: Request):
    """Resume all background tasks."""
    scheduler = getattr(request.app.state, "scheduler", None)
    if scheduler is None:
        raise HTTPException(status_code=503, detail="Scheduler not running")
    for job in scheduler.get_jobs():
        scheduler.resume_job(job.id)
    return {"status": "all_resumed"}


@router.post("/tasks/{task_id}/run")
async def run_task(task_id: str, request: Request):
    """Manually trigger a background task by ID."""
    engine = request.app.state.engine

    # index_updater is a method on engine.builder, not a standalone function
    if task_id == "index_updater":
        added, updated = engine.builder.incremental_update()
        return {"status": "completed", "task": task_id, "added": added, "updated": updated}

    if task_id not in _TASK_RUNNERS:
        raise HTTPException(status_code=404, detail=f"Unknown task: {task_id}. Available: {list(_TASK_RUNNERS.keys()) + ['index_updater']}")

    import importlib
    module_path, func_name = _TASK_RUNNERS[task_id]
    module = importlib.import_module(module_path)
    runner = getattr(module, func_name)

    runner(engine)
    return {"status": "completed", "task": task_id}


@router.post("/tasks/run-all")
async def run_all_tasks(request: Request):
    """Run all background tasks sequentially in sleep-cycle order."""
    import importlib

    engine = request.app.state.engine
    results: dict[str, str] = {}

    for task_id in _SLEEP_CYCLE_ORDER:
        try:
            if task_id == "index_updater":
                engine.builder.incremental_update()
            elif task_id in _TASK_RUNNERS:
                module_path, func_name = _TASK_RUNNERS[task_id]
                module = importlib.import_module(module_path)
                runner = getattr(module, func_name)
                runner(engine)
            results[task_id] = "ok"
        except Exception as exc:
            results[task_id] = f"error: {exc}"

    return {"status": "completed", "results": results}

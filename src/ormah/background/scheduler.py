"""APScheduler job registration for background processing."""

from __future__ import annotations

import logging

from apscheduler.schedulers.background import BackgroundScheduler

from ormah.background.job_tracker import JobTracker, tracked
from ormah.engine.memory_engine import MemoryEngine

logger = logging.getLogger(__name__)

# How long (seconds) a misfired job is still allowed to run.
# If the scheduler was blocked longer than this, the run is skipped.
_MISFIRE_GRACE = 120


def start_scheduler(engine: MemoryEngine) -> tuple[BackgroundScheduler, JobTracker]:
    """Register and start all background jobs.

    Returns ``(scheduler, tracker)`` so the caller can inspect job health
    via ``tracker.snapshot()``.
    """
    scheduler = BackgroundScheduler()
    tracker = JobTracker()
    s = engine.settings

    from ormah.background.auto_linker import run_auto_linker

    scheduler.add_job(
        tracked(tracker, "auto_linker", run_auto_linker, engine),
        "interval",
        minutes=s.auto_link_interval_minutes,
        id="auto_linker",
        name="Auto-linker",
        misfire_grace_time=_MISFIRE_GRACE,
    )

    from ormah.background.decay_manager import run_decay

    scheduler.add_job(
        tracked(tracker, "decay_manager", run_decay, engine),
        "interval",
        hours=s.decay_interval_hours,
        id="decay_manager",
        name="Decay manager",
        misfire_grace_time=_MISFIRE_GRACE,
    )

    from ormah.background.conflict_detector import run_conflict_detection

    scheduler.add_job(
        tracked(tracker, "conflict_detector", run_conflict_detection, engine),
        "interval",
        minutes=s.conflict_check_interval_minutes,
        id="conflict_detector",
        name="Conflict detector",
        misfire_grace_time=_MISFIRE_GRACE,
    )

    from ormah.background.duplicate_merger import run_duplicate_detection

    scheduler.add_job(
        tracked(tracker, "duplicate_merger", run_duplicate_detection, engine),
        "interval",
        minutes=s.duplicate_check_interval_minutes,
        id="duplicate_merger",
        name="Duplicate merger",
        misfire_grace_time=_MISFIRE_GRACE,
    )

    from ormah.background.auto_cluster import run_auto_cluster

    scheduler.add_job(
        tracked(tracker, "auto_cluster", run_auto_cluster, engine),
        "interval",
        minutes=s.auto_cluster_interval_minutes,
        id="auto_cluster",
        name="Auto-cluster",
        misfire_grace_time=_MISFIRE_GRACE,
    )

    from ormah.background.consolidator import run_consolidation

    scheduler.add_job(
        tracked(tracker, "consolidator", run_consolidation, engine),
        "interval",
        minutes=s.consolidation_interval_minutes,
        id="consolidator",
        name="Consolidator",
        misfire_grace_time=_MISFIRE_GRACE,
    )

    from ormah.background.importance_scorer import run_importance_scoring

    scheduler.add_job(
        tracked(tracker, "importance_scorer", run_importance_scoring, engine),
        "interval",
        minutes=s.importance_recompute_interval_minutes,
        id="importance_scorer",
        name="Importance scorer",
        misfire_grace_time=_MISFIRE_GRACE,
    )

    scheduler.add_job(
        tracked(tracker, "index_updater", engine.builder.incremental_update),
        "interval",
        minutes=1,
        id="index_updater",
        name="Index updater",
        misfire_grace_time=_MISFIRE_GRACE,
    )

    scheduler.start()
    logger.info("Background scheduler started with %d jobs", len(scheduler.get_jobs()))
    return scheduler, tracker

"""APScheduler job registration for background processing."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
import logging

from apscheduler.schedulers.background import BackgroundScheduler

from ormah.background.job_tracker import JobTracker, tracked
from ormah.engine.memory_engine import MemoryEngine

logger = logging.getLogger(__name__)

# How long (seconds) a misfired job is still allowed to run.
# If the scheduler was blocked longer than this, the run is skipped.
_MISFIRE_GRACE = 120


# Spread the four LLM jobs so they don't burst together 24h after boot (#90).
# Offset is relative to process start, not wall-clock — a restart loop shorter than
# the offset (e.g. crash-looping every 2 min against a 5+ min offset) defers the first run indefinitely.
_STAGGER_REFERENCE_MINUTES = 60


def _stagger_factor(s) -> float:
    """One shared factor for all four jobs, so distinct nominal offsets stay
    distinct regardless of which job has the shortest configured interval
    (council R3 finding 2 — per-job scaling let jobs with different intervals
    collide at the same offset)."""
    shortest = min(
        s.auto_link_interval_minutes,
        s.conflict_check_interval_minutes,
        s.duplicate_check_interval_minutes,
        s.consolidation_interval_minutes,
    )
    return min(1.0, shortest / _STAGGER_REFERENCE_MINUTES)


def _staggered(minutes: int, factor: float) -> datetime:
    """First run is offset to spread the LLM jobs, always inside one interval."""
    return datetime.now(timezone.utc) + timedelta(minutes=minutes * factor)


def start_scheduler(engine: MemoryEngine) -> tuple[BackgroundScheduler, JobTracker]:
    """Register and start all background jobs.

    Returns ``(scheduler, tracker)`` so the caller can inspect job health
    via ``tracker.snapshot()``.
    """
    scheduler = BackgroundScheduler()
    tracker = JobTracker()
    s = engine.settings
    stagger_factor = _stagger_factor(s)

    from ormah.background.auto_linker import run_auto_linker

    scheduler.add_job(
        tracked(tracker, "auto_linker", run_auto_linker, engine),
        "interval",
        minutes=s.auto_link_interval_minutes,
        id="auto_linker",
        name="Auto-linker",
        next_run_time=_staggered(5, stagger_factor),
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
        next_run_time=_staggered(15, stagger_factor),
        misfire_grace_time=_MISFIRE_GRACE,
    )

    from ormah.background.duplicate_merger import run_duplicate_detection

    scheduler.add_job(
        tracked(tracker, "duplicate_merger", run_duplicate_detection, engine),
        "interval",
        minutes=s.duplicate_check_interval_minutes,
        id="duplicate_merger",
        name="Duplicate merger",
        next_run_time=_staggered(30, stagger_factor),
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
        next_run_time=_staggered(45, stagger_factor),
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

    from ormah.background.whisper_log_cleanup import run_whisper_log_cleanup

    scheduler.add_job(
        tracked(tracker, "whisper_log_cleanup", run_whisper_log_cleanup, engine),
        "interval",
        hours=s.whisper_log_cleanup_interval_hours,
        id="whisper_log_cleanup",
        name="Whisper log cleanup",
        misfire_grace_time=_MISFIRE_GRACE,
    )

    from ormah.backup import run_auto_backup

    scheduler.add_job(
        tracked(tracker, "memory_backup", run_auto_backup, engine),
        "interval",
        hours=s.backup_interval_hours,
        id="memory_backup",
        name="Memory backup",
        next_run_time=datetime.now(timezone.utc),
        misfire_grace_time=_MISFIRE_GRACE,
    )

    from ormah.cloud.jobs import run_cloud_backup, run_restore_verification

    scheduler.add_job(
        tracked(tracker, "cloud_backup", run_cloud_backup, engine),
        "interval",
        hours=s.cloud_backup_interval_hours,
        id="cloud_backup",
        name="Encrypted cloud backup",
        next_run_time=datetime.now(timezone.utc),
        misfire_grace_time=_MISFIRE_GRACE,
    )

    scheduler.add_job(
        tracked(tracker, "restore_verification", run_restore_verification, engine),
        "interval",
        hours=168,
        id="restore_verification",
        name="Cloud restore verification",
        misfire_grace_time=_MISFIRE_GRACE,
    )

    scheduler.start()
    logger.info("Background scheduler started with %d jobs", len(scheduler.get_jobs()))
    return scheduler, tracker

"""APScheduler job registration for background processing."""

from __future__ import annotations

from datetime import datetime, timezone
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

    from ormah.background.whisper_log_cleanup import run_whisper_log_cleanup

    scheduler.add_job(
        tracked(tracker, "whisper_log_cleanup", run_whisper_log_cleanup, engine),
        "interval",
        hours=s.whisper_log_cleanup_interval_hours,
        id="whisper_log_cleanup",
        name="Whisper log cleanup",
        misfire_grace_time=_MISFIRE_GRACE,
    )

    from ormah.background.synthetic_pattern_monitor import run_synthetic_pattern_monitor

    scheduler.add_job(
        tracked(tracker, "synthetic_pattern_monitor", run_synthetic_pattern_monitor, engine),
        "interval",
        minutes=s.whisper_pattern_monitor_interval_minutes,
        id="synthetic_pattern_monitor",
        name="Synthetic pattern monitor",
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


def register_session_reconcile_job(scheduler, tracker, watches, interval_minutes: int) -> None:
    """Register the session-watcher reconcile job on an already-started scheduler.

    Registered after the watchers start (they do not exist when start_scheduler runs), so the job
    can reach the live handlers/observers. ``coalesce=True`` + a full-interval misfire grace mean a
    slightly-long tick is never silently dropped; reconcile re-scans disk each run, so a coalesced
    tick loses no work. No-op when there are no watchers.
    """
    if not watches:
        return
    from ormah.background.session_watcher import run_session_reconcile

    scheduler.add_job(
        tracked(tracker, "session_reconcile", run_session_reconcile, watches),
        "interval",
        minutes=interval_minutes,
        id="session_reconcile",
        name="Session reconcile",
        max_instances=1,
        coalesce=True,
        misfire_grace_time=max(_MISFIRE_GRACE, interval_minutes * 60),
    )
    logger.info("Session reconcile job registered (every %d min)", interval_minutes)

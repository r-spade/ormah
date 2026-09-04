"""Background job registration (#143)."""

from __future__ import annotations

from ormah.background.scheduler import start_scheduler


def test_synthetic_pattern_monitor_is_registered(engine):
    scheduler, _tracker = start_scheduler(engine)
    try:
        job_ids = {job.id for job in scheduler.get_jobs()}
        assert "synthetic_pattern_monitor" in job_ids
    finally:
        scheduler.shutdown(wait=False)

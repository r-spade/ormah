# Task 08: Register the forgetting job in the scheduler

**Depends on:** Task 05 (`run_forgetting` exists).

Register `run_forgetting` as a tracked interval job, following the exact pattern of the other
background jobs in `start_scheduler`. The job runs on every interval regardless of the master
switch; `run_forgetting` itself short-circuits to a no-op when `deletion_enabled=False`.

**Files:**
- Modify: `src/ormah/background/scheduler.py`
- Test: `tests/test_background/test_scheduler.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_background/test_scheduler.py` (or append if it exists):

```python
from __future__ import annotations

from ormah.background.scheduler import start_scheduler


def test_forgetting_job_is_registered(engine):
    scheduler, _tracker = start_scheduler(engine)
    try:
        job_ids = {job.id for job in scheduler.get_jobs()}
        assert "forgetting_manager" in job_ids
    finally:
        scheduler.shutdown(wait=False)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_background/test_scheduler.py -v`
Expected: FAIL (`forgetting_manager` job not registered).

- [ ] **Step 3: Register the job**

In `src/ormah/background/scheduler.py`, add this block after the `decay_manager`
registration (it is the conceptual sibling of decay):

```python
    from ormah.background.forgetting_manager import run_forgetting

    scheduler.add_job(
        tracked(tracker, "forgetting_manager", run_forgetting, engine),
        "interval",
        hours=s.forgetting_interval_hours,
        id="forgetting_manager",
        name="Forgetting manager",
        misfire_grace_time=_MISFIRE_GRACE,
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_background/test_scheduler.py -v`
Expected: PASS.

- [ ] **Step 5: Full suite + lint**

Run: `.venv/bin/python -m pytest tests/ -v`
Expected: PASS (whole suite green; default run excludes `integration`).

Run: `.venv/bin/ruff check src/ tests/`
Expected: no errors.

- [ ] **Step 6: Commit**

```bash
git add src/ormah/background/scheduler.py tests/test_background/test_scheduler.py
git commit -m "feat(background): register forgetting manager job (#28)"
```

## Final verification (after all tasks)

- [ ] `.venv/bin/python -m pytest tests/ -v` — whole suite green
- [ ] `.venv/bin/ruff check src/ tests/` — clean
- [ ] Confirm no-op default: with `deletion_enabled=False`, `run_forgetting` returns before any
      deletion (covered by `test_master_switch_off_is_noop` and `test_purge_skipped_when_disabled`).
- [ ] Update issue #28 / open the PR against the base branch.

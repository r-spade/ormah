"""Tests for background job tracker."""

from __future__ import annotations

from ormah.background.job_tracker import JobTracker, tracked


def test_record_success():
    tracker = JobTracker()
    tracker.record_success("test_job", 123.4)

    snap = tracker.snapshot()
    assert "test_job" in snap
    assert snap["test_job"]["run_count"] == 1
    assert snap["test_job"]["error_count"] == 0
    assert snap["test_job"]["last_duration_ms"] == 123.4
    assert snap["test_job"]["last_success"] is not None
    assert snap["test_job"]["last_error"] is None


def test_record_failure():
    tracker = JobTracker()
    tracker.record_failure("test_job", "boom", 50.0)

    snap = tracker.snapshot()
    assert snap["test_job"]["run_count"] == 1
    assert snap["test_job"]["error_count"] == 1
    assert snap["test_job"]["last_error"] == "boom"
    assert snap["test_job"]["last_error_time"] is not None
    assert snap["test_job"]["last_success"] is None


def test_mixed_runs():
    tracker = JobTracker()
    tracker.record_success("job", 10.0)
    tracker.record_success("job", 20.0)
    tracker.record_failure("job", "oops", 5.0)

    snap = tracker.snapshot()
    assert snap["job"]["run_count"] == 3
    assert snap["job"]["error_count"] == 1
    assert snap["job"]["last_error"] == "oops"
    assert snap["job"]["last_success"] is not None


def test_tracked_wrapper_success():
    tracker = JobTracker()
    calls = []

    def my_job(engine):
        calls.append(engine)

    wrapper = tracked(tracker, "my_job", my_job, "fake_engine")
    wrapper()

    assert calls == ["fake_engine"]
    snap = tracker.snapshot()
    assert snap["my_job"]["run_count"] == 1
    assert snap["my_job"]["error_count"] == 0


def test_tracked_wrapper_failure():
    tracker = JobTracker()

    def failing_job(engine):
        raise ValueError("test error")

    wrapper = tracked(tracker, "fail_job", failing_job, "fake_engine")
    wrapper()  # should not raise — tracked() catches exceptions

    snap = tracker.snapshot()
    assert snap["fail_job"]["run_count"] == 1
    assert snap["fail_job"]["error_count"] == 1
    assert "test error" in snap["fail_job"]["last_error"]


def test_snapshot_empty():
    tracker = JobTracker()
    assert tracker.snapshot() == {}


def test_multiple_jobs_independent():
    tracker = JobTracker()
    tracker.record_success("job_a", 10.0)
    tracker.record_failure("job_b", "err", 5.0)

    snap = tracker.snapshot()
    assert snap["job_a"]["error_count"] == 0
    assert snap["job_b"]["error_count"] == 1


def test_tracker_reports_a_job_as_running_while_it_executes():
    from ormah.background.job_tracker import JobTracker, tracked

    tracker = JobTracker()
    seen = {}

    def job():
        seen["running_during"] = tracker.is_running("demo")

    assert tracker.is_running("demo") is False
    tracked(tracker, "demo", job)()
    assert seen["running_during"] is True
    assert tracker.is_running("demo") is False   # cleared after completion


def test_tracker_clears_the_running_flag_when_the_job_raises():
    from ormah.background.job_tracker import JobTracker, tracked

    tracker = JobTracker()

    def boom():
        raise RuntimeError("nope")

    tracked(tracker, "demo", boom)()
    assert tracker.is_running("demo") is False   # a stuck flag would wedge the job forever


def test_run_guard_refuses_a_second_concurrent_claim():
    from ormah.background.job_tracker import JobTracker

    tracker = JobTracker()
    inner = []

    with tracker.run_guard("demo") as acquired:
        assert acquired is True
        with tracker.run_guard("demo") as second:
            inner.append(second)
    assert inner == [False]

    with tracker.run_guard("demo") as third:   # released -> claimable again
        assert third is True


def test_tracked_skips_a_run_when_the_job_is_already_running():
    from ormah.background.job_tracker import JobTracker, tracked

    tracker = JobTracker()
    calls = []

    with tracker.run_guard("demo"):
        tracked(tracker, "demo", lambda: calls.append(1))()

    assert calls == []   # the wrapped function never ran


def test_tracked_records_a_failure_when_the_job_returns_an_error_dict():
    """A runner that dies signals it by RETURNING {'error': ...} (it catches its own
    exceptions), so tracked() must inspect the return value. Ignoring it made the job
    tracker — and therefore /admin/health — report a dead run as a success (#117)."""
    from ormah.background.job_tracker import JobTracker, tracked

    tracker = JobTracker()
    tracked(tracker, "demo", lambda: {"error": "watermark exploded"})()

    snap = tracker.snapshot()["demo"]
    assert snap["last_error"] == "watermark exploded"
    assert snap["error_count"] == 1
    assert snap["last_success"] is None


def test_tracked_records_a_failure_when_the_job_returns_false():
    """run_restore_verification returns a bool — False means the restore could NOT be
    verified. Treating every non-error-dict as success left a failed restore check green
    in /admin/health (Codex, PR B round 2)."""
    from ormah.background.job_tracker import JobTracker, tracked

    tracker = JobTracker()
    tracked(tracker, "restore_verification", lambda: False)()

    snap = tracker.snapshot()["restore_verification"]
    assert snap["error_count"] == 1
    assert snap["last_success"] is None


def test_tracked_still_treats_true_and_none_as_success():
    from ormah.background.job_tracker import JobTracker, tracked

    tracker = JobTracker()
    tracked(tracker, "ok_bool", lambda: True)()
    tracked(tracker, "ok_none", lambda: None)()

    assert tracker.snapshot()["ok_bool"]["error_count"] == 0
    assert tracker.snapshot()["ok_none"]["error_count"] == 0

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

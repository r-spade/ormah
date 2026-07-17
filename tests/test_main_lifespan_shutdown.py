"""Bounded scheduler shutdown + engine.shutdown() policy (Fix A / Fix D).

Tests the invariant: when the scheduler's shutdown(wait=True) does not
complete within _SHUTDOWN_TIMEOUT, engine.shutdown() must NOT be called
(avoids use-after-close). Symmetrically, when scheduler exits in time,
engine.shutdown() IS called (assuming fallback is not alive).

Design: avoids spinning up a full FastAPI + MemoryEngine for speed. We
exercise the decision logic via:
  1. A pure helper `_should_close_engine(fallback_alive, scheduler_alive)`
     extracted from the lifespan — unit-tests the policy table.
  2. A direct test that the scheduler-shutdown bounded path sets
     `scheduler_alive=True` when the scheduler.shutdown() blocks past the
     timeout (mock scheduler whose shutdown() blocks on an Event).
  3. An integration-style test that wires `_stop_backfill_fallback` + the
     scheduler bounded block together and asserts engine.shutdown() is NOT
     called when either is alive.
  4. Per-lifespan stop event (R1): each lifespan execution creates a fresh
     Event in app.state so a reload cannot rearm an orphan worker.
"""
from __future__ import annotations

import threading as _threading
import time as _time

import pytest
from fastapi import FastAPI

from ormah import main

real_sleep = _time.sleep


@pytest.fixture(autouse=True)
def _reset_fallback_state():
    main._stop_backfill_fallback()
    main._fallback_degraded = False
    yield
    main._stop_backfill_fallback()
    main._fallback_degraded = False


def _wait_for(predicate, timeout=2.0):
    waited = 0.0
    while not predicate() and waited < timeout:
        real_sleep(0.02)
        waited += 0.02


# ---------------------------------------------------------------------------
# 1. Pure policy helper — _should_close_engine
# ---------------------------------------------------------------------------

def test_should_close_engine_both_alive():
    assert main._should_close_engine(fallback_alive=True, scheduler_alive=True) is False


def test_should_close_engine_fallback_alive():
    assert main._should_close_engine(fallback_alive=True, scheduler_alive=False) is False


def test_should_close_engine_scheduler_alive():
    assert main._should_close_engine(fallback_alive=False, scheduler_alive=True) is False


def test_should_close_engine_neither_alive():
    assert main._should_close_engine(fallback_alive=False, scheduler_alive=False) is True


# ---------------------------------------------------------------------------
# 2. Bounded scheduler-shutdown: scheduler stuck past timeout → scheduler_alive=True
# ---------------------------------------------------------------------------

def test_scheduler_shutdown_timeout_sets_scheduler_alive(monkeypatch):
    """Fix A: scheduler.shutdown(wait=True) that blocks past _SHUTDOWN_TIMEOUT
    must result in scheduler_alive=True so engine.shutdown() is skipped."""
    monkeypatch.setattr(main, "_SHUTDOWN_TIMEOUT", 0.1)

    release = _threading.Event()

    class _BlockingScheduler:
        def shutdown(self, wait=True):
            # Simulates a job stuck in a non-interruptible encoder.encode()
            release.wait(timeout=10.0)

    scheduler_alive = main._bounded_scheduler_shutdown(_BlockingScheduler())

    assert scheduler_alive is True, (
        "scheduler_alive must be True when shutdown() does not complete within _SHUTDOWN_TIMEOUT"
    )
    # Cleanup
    release.set()


def test_scheduler_shutdown_completes_within_timeout(monkeypatch):
    """Fix A: scheduler.shutdown() that completes before timeout → scheduler_alive=False."""
    monkeypatch.setattr(main, "_SHUTDOWN_TIMEOUT", 5.0)

    class _QuickScheduler:
        def shutdown(self, wait=True):
            return  # exits immediately

    scheduler_alive = main._bounded_scheduler_shutdown(_QuickScheduler())

    assert scheduler_alive is False, (
        "scheduler_alive must be False when shutdown() completes before timeout"
    )


# ---------------------------------------------------------------------------
# 3. Integration: fallback preso → engine NOT closed
# ---------------------------------------------------------------------------

def test_engine_not_closed_when_fallback_alive(monkeypatch):
    """Fix D: when the fallback thread survives the join timeout, engine.shutdown()
    must NOT be called."""
    monkeypatch.setattr(main, "_FALLBACK_JOIN_TIMEOUT", 0.1)
    monkeypatch.setattr(main, "_SHUTDOWN_TIMEOUT", 0.1)
    monkeypatch.setattr(main, "_BACKFILL_FALLBACK_BASE_BACKOFF", 0.001)

    release = _threading.Event()

    def _blocking_run(engine, stop_event=None):
        release.wait(timeout=10.0)

    monkeypatch.setattr(
        "ormah.background.embedding_backfill.run_embedding_backfill",
        _blocking_run,
    )

    main._start_backfill_fallback(object())
    _wait_for(lambda: main._fallback_thread is not None
              and main._fallback_thread.is_alive(), timeout=2.0)

    fallback_alive = main._stop_backfill_fallback()

    shutdown_called = []

    class _FakeEngine:
        def shutdown(self):
            shutdown_called.append(True)

    if not main._should_close_engine(fallback_alive=fallback_alive, scheduler_alive=False):
        pass  # engine.shutdown() deliberately skipped
    else:
        _FakeEngine().shutdown()

    assert fallback_alive is True
    assert shutdown_called == [], "engine.shutdown() must NOT be called when fallback_alive=True"

    # Cleanup
    release.set()
    _wait_for(lambda: main._fallback_thread is None
              or not main._fallback_thread.is_alive(), timeout=3.0)
    with main._fallback_lock:
        main._fallback_thread = None
        main._fallback_stop_event = None


def test_engine_not_closed_when_scheduler_alive(monkeypatch):
    """Fix A: when scheduler shutdown does not complete in time, engine.shutdown()
    must NOT be called."""
    monkeypatch.setattr(main, "_SHUTDOWN_TIMEOUT", 0.1)

    release = _threading.Event()

    class _BlockingScheduler:
        def shutdown(self, wait=True):
            release.wait(timeout=10.0)

    scheduler_alive = main._bounded_scheduler_shutdown(_BlockingScheduler())

    shutdown_called = []

    class _FakeEngine:
        def shutdown(self):
            shutdown_called.append(True)

    if not main._should_close_engine(fallback_alive=False, scheduler_alive=scheduler_alive):
        pass
    else:
        _FakeEngine().shutdown()

    assert scheduler_alive is True
    assert shutdown_called == [], "engine.shutdown() must NOT be called when scheduler_alive=True"

    release.set()


def test_engine_closed_when_both_exit_cleanly(monkeypatch):
    """Positive path: both fallback and scheduler exit cleanly → engine.shutdown() called."""
    monkeypatch.setattr(main, "_FALLBACK_JOIN_TIMEOUT", 5.0)
    monkeypatch.setattr(main, "_SHUTDOWN_TIMEOUT", 5.0)
    monkeypatch.setattr(main, "_BACKFILL_FALLBACK_BASE_BACKOFF", 0.001)

    def _quick_run(engine, stop_event=None):
        return  # exits immediately

    monkeypatch.setattr(
        "ormah.background.embedding_backfill.run_embedding_backfill",
        _quick_run,
    )

    main._start_backfill_fallback(object())
    _wait_for(lambda: main._fallback_thread is not None, timeout=2.0)

    fallback_alive = main._stop_backfill_fallback()

    class _QuickScheduler:
        def shutdown(self, wait=True):
            return

    scheduler_alive = main._bounded_scheduler_shutdown(_QuickScheduler())

    shutdown_called = []

    class _FakeEngine:
        def shutdown(self):
            shutdown_called.append(True)

    if main._should_close_engine(fallback_alive=fallback_alive, scheduler_alive=scheduler_alive):
        _FakeEngine().shutdown()

    assert fallback_alive is False
    assert scheduler_alive is False
    assert shutdown_called == [True], "engine.shutdown() must be called when both exit cleanly"


# ---------------------------------------------------------------------------
# 4. Per-lifespan stop event (R1)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_each_lifespan_gets_its_own_stop_event(tmp_path, monkeypatch):
    """R1: each lifespan execution must create a NEW threading.Event in
    app.state.lifecycle_stop_event. A reload must never reuse (or clear) the
    event from a previous lifespan so that orphan workers from a prior,
    expired shutdown cannot be rearmed.

    We monkeypatch heavy I/O (MemoryEngine, start_scheduler, watchers) so the
    test is fast and hermetic. The key invariants:
      - sched_ev1 is not sched_ev2  (distinct object per lifespan execution)
      - ev1 is sched_ev1 / ev2 is sched_ev2  (app.state holds the same object
        that was passed to start_scheduler)
    """
    import sys
    import threading

    # --- lightweight fakes ---

    class _FakeEngine:
        def startup(self): pass
        def shutdown(self): pass

    class _FakeScheduler:
        def shutdown(self, wait=True): pass

    class _FakeTracker:
        pass

    captured_stop_events: list[threading.Event] = []

    def _fake_start_scheduler(engine, stop_event=None):
        captured_stop_events.append(stop_event)
        return _FakeScheduler(), _FakeTracker()

    monkeypatch.setattr("ormah.main.MemoryEngine", lambda settings: _FakeEngine())
    monkeypatch.setattr(
        "ormah.main.settings",
        type("S", (), {"port": 8787, "memory_dir": str(tmp_path)})(),
    )
    monkeypatch.setattr("ormah.main.MaintenanceManager", lambda *a, **kw: object())

    # Use monkeypatch.setitem so pytest restores sys.modules on teardown,
    # preventing contamination of later tests (e.g. test_ingest_stores_node_ids).
    _fake_hippocampus = type(sys)("_fake_hippo")
    _fake_hippocampus.start_hippocampus = lambda engine: []
    _fake_hippocampus.stop_hippocampus = lambda obs: None
    monkeypatch.setitem(sys.modules, "ormah.background.hippocampus", _fake_hippocampus)

    _fake_session_watcher = type(sys)("_fake_sw")
    _fake_session_watcher.start_session_watcher = lambda engine: []
    _fake_session_watcher.stop_session_watcher = lambda obs: None
    monkeypatch.setitem(sys.modules, "ormah.background.session_watcher", _fake_session_watcher)

    _fake_scheduler_mod = type(sys)("_fake_sched")
    _fake_scheduler_mod.start_scheduler = _fake_start_scheduler
    monkeypatch.setitem(sys.modules, "ormah.background.scheduler", _fake_scheduler_mod)

    app = FastAPI(lifespan=main.lifespan)

    # --- first lifespan execution ---
    async with main.lifespan(app):
        ev1 = app.state.lifecycle_stop_event
        assert isinstance(ev1, threading.Event), "lifecycle_stop_event must be a threading.Event"

    # --- second lifespan execution (simulates in-process reload) ---
    async with main.lifespan(app):
        ev2 = app.state.lifecycle_stop_event
        assert isinstance(ev2, threading.Event), "lifecycle_stop_event must be a threading.Event"

    # Both lifespans must have passed a stop_event to start_scheduler
    assert len(captured_stop_events) == 2, (
        f"Expected 2 captured stop events, got {len(captured_stop_events)}"
    )
    sched_ev1, sched_ev2 = captured_stop_events

    # Invariant 1: each lifespan creates a DISTINCT Event — the R1 bug was
    # reusing (and clear()-ing) a single global Event across lifespans.
    assert sched_ev1 is not sched_ev2, (
        "Each lifespan must create a DISTINCT Event; reload must not reuse the old one (R1)"
    )

    # Invariant 2: app.state holds the same object that was passed to the scheduler
    assert ev1 is sched_ev1, (
        "app.state.lifecycle_stop_event must be the exact event passed to start_scheduler"
    )
    assert ev2 is sched_ev2, (
        "app.state.lifecycle_stop_event must be the exact event passed to start_scheduler"
    )

    # Invariant 3: both events were signalled during their respective shutdowns
    assert sched_ev1.is_set(), "First lifespan stop event must be set after its shutdown"
    assert sched_ev2.is_set(), "Second lifespan stop event must be set after its shutdown"

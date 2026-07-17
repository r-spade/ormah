"""Scheduler-independent embedding backfill fallback (#32, council C2/CH1/CH2).

The fallback retries indefinitely with backoff until the gap closes
(run_embedding_backfill raises while it is incomplete). Lifecycle is
controlled by a stop event and a singleton guard; _fallback_degraded exposes
a persistent outage to /admin/health while no scheduler exists.
"""
from __future__ import annotations

import threading as _threading
import time as _time

import pytest

from ormah import main

real_sleep = _time.sleep


@pytest.fixture(autouse=True)
def _reset_fallback_state():
    main._stop_backfill_fallback()  # tear down any thread a prior test started
    main._fallback_degraded = False
    yield
    main._stop_backfill_fallback()
    main._fallback_degraded = False


def _wait_for(predicate, timeout=2.0):
    waited = 0.0
    while not predicate() and waited < timeout:
        real_sleep(0.02)
        waited += 0.02


def test_fallback_runs_backfill_off_thread(monkeypatch):
    calls = []
    monkeypatch.setattr(
        "ormah.background.embedding_backfill.run_embedding_backfill",
        lambda engine, stop_event=None: calls.append(engine),
    )
    sentinel = object()
    main._start_backfill_fallback(sentinel)  # returns immediately (off-thread)
    _wait_for(lambda: len(calls) >= 1)
    assert calls == [sentinel]


def test_fallback_retries_until_success(monkeypatch):
    calls = []

    def _flaky(engine, stop_event=None):
        calls.append(engine)
        if len(calls) < 2:
            raise RuntimeError("still incomplete")

    monkeypatch.setattr(
        "ormah.background.embedding_backfill.run_embedding_backfill", _flaky)
    monkeypatch.setattr(main, "_BACKFILL_FALLBACK_BASE_BACKOFF", 0.001)

    main._start_backfill_fallback(object())
    _wait_for(lambda: len(calls) >= 2)
    real_sleep(0.05)  # give a spurious 3rd attempt a chance to (not) happen
    assert len(calls) == 2  # retried once, then stopped on success


def test_fallback_keeps_retrying_past_old_budget(monkeypatch):
    """C2: fallback does not give up after 5 attempts — retries until success."""
    calls = []

    def _mostly_failing(engine, stop_event=None):
        calls.append(engine)
        if len(calls) <= 6:  # fail more than the old hard budget of 5
            raise RuntimeError("still incomplete")
        # 7th call succeeds (returns None)

    monkeypatch.setattr(
        "ormah.background.embedding_backfill.run_embedding_backfill",
        _mostly_failing,
    )
    monkeypatch.setattr(main, "_BACKFILL_FALLBACK_BASE_BACKOFF", 0.001)

    main._start_backfill_fallback(object())
    _wait_for(lambda: len(calls) >= 7, timeout=3.0)
    real_sleep(0.05)
    assert len(calls) == 7  # succeeded on the 7th attempt (past the old budget of 5)


def test_fallback_sets_degraded_flag_on_failure_clears_on_success(monkeypatch):
    """CH2: persistent failure is observable via _fallback_degraded; clears on recovery."""
    calls = []

    def _flaky(engine, stop_event=None):
        calls.append(engine)
        if len(calls) < 3:
            raise RuntimeError("still incomplete")

    monkeypatch.setattr(
        "ormah.background.embedding_backfill.run_embedding_backfill", _flaky)
    monkeypatch.setattr(main, "_BACKFILL_FALLBACK_BASE_BACKOFF", 0.001)

    main._start_backfill_fallback(object())
    # After the first failure, before success, the flag is set.
    _wait_for(lambda: main._fallback_degraded is True, timeout=2.0)
    assert main._fallback_degraded is True
    # On eventual success it clears.
    _wait_for(lambda: main._fallback_degraded is False and len(calls) >= 3, timeout=2.0)
    assert main._fallback_degraded is False


def test_fallback_is_singleton(monkeypatch):
    """CH1: a second start while one is alive does not spawn a second thread."""
    started = []

    def _block_forever(engine, stop_event=None):
        started.append(engine)
        raise RuntimeError("never closes")  # keeps the loop alive

    monkeypatch.setattr(
        "ormah.background.embedding_backfill.run_embedding_backfill", _block_forever)
    monkeypatch.setattr(main, "_BACKFILL_FALLBACK_BASE_BACKOFF", 0.001)

    main._start_backfill_fallback(object())
    _wait_for(lambda: main._fallback_thread is not None
              and main._fallback_thread.is_alive(), timeout=2.0)
    first = main._fallback_thread
    main._start_backfill_fallback(object())  # second start — must be a no-op
    assert main._fallback_thread is first  # same thread, not replaced


def test_fallback_stops_on_shutdown(monkeypatch):
    """CH1: _stop_backfill_fallback stops a permanently-failing fallback."""
    calls = []

    def _boom(engine, stop_event=None):
        calls.append(engine)
        raise RuntimeError("permanently incomplete")

    monkeypatch.setattr(
        "ormah.background.embedding_backfill.run_embedding_backfill", _boom)
    monkeypatch.setattr(main, "_BACKFILL_FALLBACK_BASE_BACKOFF", 0.001)

    main._start_backfill_fallback(object())
    _wait_for(lambda: len(calls) >= 1, timeout=2.0)
    main._stop_backfill_fallback()
    assert main._fallback_thread is None or not main._fallback_thread.is_alive()
    settled = len(calls)
    real_sleep(0.1)
    assert len(calls) == settled  # no further attempts after stop


# ---------------------------------------------------------------------------
# Task B — new tests (CRB: atomic singleton, CR1 revert, stop_event forwarding)
# ---------------------------------------------------------------------------


class _FakeEngine:
    """Blocks in backfill_embeddings until stop_event is set or 10s elapses.
    When stop_event is None (pre-implementation, stop_event not forwarded yet),
    blocks for the full safety-net duration so the thread stays alive long enough
    for the concurrent test to count it."""

    def __init__(self):
        self.entered = _threading.Event()
        self._internal_block = _threading.Event()

    def backfill_embeddings(self, stop_event=None):
        self.entered.set()
        # Use stop_event if provided, otherwise block on internal event (safety net 10s)
        waiter = stop_event if stop_event is not None else self._internal_block
        waiter.wait(timeout=10.0)
        return {"missing": 0}


class _QuickEngine:
    """Completes immediately with no missing nodes."""

    def backfill_embeddings(self, stop_event=None):
        return {"missing": 0}


class _CancellableEngine:
    """Loops checking stop_event; records whether it received it."""

    def __init__(self):
        self.entered = _threading.Event()
        self.saw_stop = False

    def backfill_embeddings(self, stop_event=None):
        self.entered.set()
        while True:
            if stop_event is not None and stop_event.is_set():
                self.saw_stop = True
                return {"missing": 0}
            _threading.Event().wait(timeout=0.01)


def _monkeypatch_run_embedding_backfill(monkeypatch):
    """Patch run_embedding_backfill to delegate to engine.backfill_embeddings(stop_event=...)."""
    def _fake_run(engine, stop_event=None):
        result = engine.backfill_embeddings(stop_event=stop_event)
        if result.get("missing", 0) > 0:
            raise RuntimeError(f"backfill incomplete: {result['missing']} missing")

    monkeypatch.setattr(
        "ormah.background.embedding_backfill.run_embedding_backfill",
        _fake_run,
    )


def test_concurrent_start_creates_single_thread(monkeypatch):
    """CRB: 8 threads racing _start_backfill_fallback must produce exactly 1 live thread."""
    monkeypatch.setattr(main, "_BACKFILL_FALLBACK_BASE_BACKOFF", 0.001)
    _monkeypatch_run_embedding_backfill(monkeypatch)

    engine = _FakeEngine()
    barrier = _threading.Barrier(8)

    def racer():
        barrier.wait()
        main._start_backfill_fallback(engine)

    threads = [_threading.Thread(target=racer) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    alive = [
        t for t in _threading.enumerate()
        if t.name == "embedding-backfill-fallback" and t.is_alive()
    ]
    assert len(alive) == 1
    main._stop_backfill_fallback()


def test_stop_clears_handle(monkeypatch):
    """CR1 reverted: handle is always cleared after stop, even on quick completion."""
    monkeypatch.setattr(main, "_BACKFILL_FALLBACK_BASE_BACKOFF", 0.001)
    _monkeypatch_run_embedding_backfill(monkeypatch)

    main._start_backfill_fallback(_QuickEngine())
    main._stop_backfill_fallback()
    assert main._fallback_thread is None


def test_stop_cancels_long_backfill_within_join(monkeypatch):
    """stop_event is forwarded to the engine; handle is cleared; saw_stop is True."""
    monkeypatch.setattr(main, "_BACKFILL_FALLBACK_BASE_BACKOFF", 0.001)
    _monkeypatch_run_embedding_backfill(monkeypatch)

    eng = _CancellableEngine()
    main._start_backfill_fallback(eng)
    assert eng.entered.wait(timeout=2.0), "engine never entered backfill"
    main._stop_backfill_fallback()

    assert main._fallback_thread is None
    assert eng.saw_stop is True


# ---------------------------------------------------------------------------
# C1 — bounded join com política pós-timeout
# ---------------------------------------------------------------------------


def test_stop_returns_true_when_thread_survives_timeout(monkeypatch):
    """C1: se o join expira (encode travado), _stop retorna True e handle é mantido (C-B)."""
    monkeypatch.setattr(main, "_FALLBACK_JOIN_TIMEOUT", 0.1)
    monkeypatch.setattr(main, "_BACKFILL_FALLBACK_BASE_BACKOFF", 0.001)

    release = _threading.Event()

    def _blocking(engine, stop_event=None):
        # Ignora stop_event mid-call — simula encoder.encode() travado
        release.wait(timeout=5.0)

    monkeypatch.setattr(
        "ormah.background.embedding_backfill.run_embedding_backfill",
        _blocking,
    )

    main._start_backfill_fallback(object())
    # Aguarda o thread entrar no blocking
    _wait_for(lambda: main._fallback_thread is not None
              and main._fallback_thread.is_alive(), timeout=2.0)

    result = main._stop_backfill_fallback()

    assert result is True, "_stop deve retornar True quando thread sobrevive ao timeout"
    assert main._fallback_thread is not None, "handle deve ser mantido (C-B) para bloquear segunda instância"

    # Cleanup: libera o encode e espera o thread morrer
    release.set()
    _wait_for(lambda: not main._fallback_thread.is_alive(), timeout=3.0)
    with main._fallback_lock:
        main._fallback_thread = None
        main._fallback_stop_event = None


def test_stop_returns_false_when_thread_exits(monkeypatch):
    """C1: quando o thread sai antes do timeout, _stop retorna False e handle é limpo."""
    monkeypatch.setattr(main, "_FALLBACK_JOIN_TIMEOUT", 5.0)
    monkeypatch.setattr(main, "_BACKFILL_FALLBACK_BASE_BACKOFF", 0.001)

    def _quick(engine, stop_event=None):
        return  # sai imediatamente ao receber stop_event

    monkeypatch.setattr(
        "ormah.background.embedding_backfill.run_embedding_backfill",
        _quick,
    )

    main._start_backfill_fallback(object())
    _wait_for(lambda: main._fallback_thread is not None, timeout=2.0)

    result = main._stop_backfill_fallback()

    assert result is False, "_stop deve retornar False quando thread encerra normalmente"
    assert main._fallback_thread is None, "handle deve ser limpo quando thread está morto"


def test_stop_does_not_return_while_backfill_thread_alive(monkeypatch):
    """M-A: _stop_backfill_fallback must NOT return while the thread is alive.

    Simulates encoder.encode() blocking mid-call (ignores stop_event until
    the encode finishes). With the old join(timeout=5s) the stop returns after
    ~5s and the handle is cleared while the thread is still alive — C-A/C-B.
    With the fix (join without timeout) _stop blocks until the encode releases,
    guaranteeing the engine is never closed while the thread can still touch the DB.

    Protocol:
    - `entered` fires when the fake encode begins (thread is alive, inside encode).
    - `release` blocks the fake encode; setting it simulates the encode finishing.
    - `_stop_backfill_fallback()` runs in a helper thread; `stopped` fires when it returns.
    - Assert: stopped does NOT fire within 6s while release is not set  → FAILS on old code
              (old join(5s) would return after 5s; not stopped.wait(6) would be False).
    - Release the encode, then assert: stopped fires within 3s, handle is None, thread dead.
    """
    monkeypatch.setattr(main, "_BACKFILL_FALLBACK_BASE_BACKOFF", 0.001)

    entered = _threading.Event()
    release = _threading.Event()

    def _blocking_run(engine, stop_event=None):
        entered.set()
        # Simulate a long encoder.encode() that does NOT check stop_event mid-call.
        release.wait()
        # After the blocking encode finishes, honour the stop_event so the loop exits.
        if stop_event is not None and stop_event.is_set():
            return {"mode": "delta", "embedded": 0, "failed": 0, "missing": 1,
                    "vec_count": 0, "node_count": 1}
        return {"mode": "delta", "embedded": 0, "failed": 0, "missing": 0,
                "vec_count": 0, "node_count": 1}

    def _fake_run(engine, stop_event=None):
        result = _blocking_run(engine, stop_event=stop_event)
        if result.get("missing", 0) > 0:
            raise RuntimeError(f"backfill incomplete: {result['missing']} missing")

    monkeypatch.setattr(
        "ormah.background.embedding_backfill.run_embedding_backfill",
        _fake_run,
    )

    main._start_backfill_fallback(object())
    assert entered.wait(timeout=2.0), "thread never entered the blocking encode"

    # Run _stop in a helper thread so we can observe when it returns.
    stopped = _threading.Event()

    def _do_stop():
        main._stop_backfill_fallback()
        stopped.set()

    stop_thread = _threading.Thread(target=_do_stop, daemon=True)
    stop_thread.start()

    # CRITICAL assertion: _stop must NOT return while the encode is still blocked.
    # Old code: join(timeout=5s) returns after ~5s → stopped.wait(6) is True →
    #           `not stopped.wait(6)` is False → test FAILS on old code.
    # Fixed code: join() blocks until release → stopped.wait(6) is False →
    #             `not stopped.wait(6)` is True → test PASSES.
    assert not stopped.wait(timeout=6), (
        "_stop_backfill_fallback returned while the encode was still blocking — "
        "this means join had a timeout and the handle was cleared with the thread alive (C-A/C-B)"
    )

    # Now release the blocking encode.
    release.set()

    # _stop must return promptly after the encode finishes.
    assert stopped.wait(timeout=3), "_stop_backfill_fallback did not return after encode released"

    # Handle must be cleared and thread must be dead.
    assert main._fallback_thread is None
    thread_ref = None
    # Recover thread reference via enumerate to confirm it really died.
    for t in _threading.enumerate():
        if t.name == "embedding-backfill-fallback":
            thread_ref = t
            break
    assert thread_ref is None or not thread_ref.is_alive(), (
        "backfill thread is still alive after _stop_backfill_fallback returned"
    )

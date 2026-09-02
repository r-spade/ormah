"""The restore epoch: apply steps are valid only against the graph they were computed on."""

from __future__ import annotations

import threading

import pytest

from ormah.background.memory_lock import RestoredUnderfoot, restore_aware_job


def test_memory_operation_at_yields_while_the_epoch_holds(engine):
    epoch = engine.restore_epoch
    with engine.memory_operation_at(epoch):
        pass  # no raise


def test_memory_operation_at_raises_once_the_epoch_moves(engine):
    epoch = engine.restore_epoch
    engine._restore_epoch += 1
    with pytest.raises(RestoredUnderfoot):
        with engine.memory_operation_at(epoch):
            pass


def test_memory_operation_at_holds_l_mem_while_it_yields(engine):
    """The check and the mutation must be atomic w.r.t. the restore (spec §2).

    A same-thread ``acquire(blocking=False)`` can't tell "lock held by me" from
    "lock free" because ``_memory_operation_lock`` is reentrant. Use a second
    thread instead: it can only acquire the lock when nobody else holds it.
    """
    epoch = engine.restore_epoch
    entered = threading.Event()
    probe_done = threading.Event()
    result = {}

    def probe_from_other_thread():
        entered.wait(timeout=5)
        acquired = engine._memory_operation_lock.acquire(timeout=0.2)
        if acquired:
            engine._memory_operation_lock.release()
        result["acquired_while_held"] = acquired
        probe_done.set()

    prober = threading.Thread(target=probe_from_other_thread)
    prober.start()
    with engine.memory_operation_at(epoch):
        entered.set()
        # Hold the context open until the other thread has actually tried
        # and failed to acquire the lock, so the exclusion window is real.
        assert probe_done.wait(timeout=5), "probe thread never reported back"
    prober.join(timeout=5)

    assert result.get("acquired_while_held") is False

    # And the lock must be free again once the context manager has exited.
    released = engine._memory_operation_lock.acquire(timeout=1)
    assert released is True
    engine._memory_operation_lock.release()


def test_reload_restored_graph_bumps_the_epoch(engine):
    before = engine.restore_epoch
    engine.reload_restored_graph()
    assert engine.restore_epoch == before + 1


def test_restore_aware_job_passes_the_entry_epoch_to_the_job(engine):
    seen = []

    @restore_aware_job
    def job(eng, epoch):
        seen.append(epoch)

    job(engine)
    assert seen == [engine.restore_epoch]


def test_restore_aware_job_ends_the_run_instead_of_raising(engine, caplog):
    """APScheduler must not see the abort as a job crash."""

    @restore_aware_job
    def job(eng, epoch):
        eng._restore_epoch += 1
        with eng.memory_operation_at(epoch):
            pass

    with caplog.at_level("INFO"):
        assert job(engine) is None
    assert "restore" in caplog.text.lower()


def test_restore_aware_job_forwards_extra_arguments(engine):
    seen = {}

    @restore_aware_job
    def job(eng, epoch, limit, *, dry_run=False):
        seen.update(limit=limit, dry_run=dry_run)

    job(engine, 7, dry_run=True)
    assert seen == {"limit": 7, "dry_run": True}

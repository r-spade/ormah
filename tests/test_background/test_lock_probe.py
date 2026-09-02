"""The probe itself: re-entrancy must not inflate the count."""

from __future__ import annotations

import threading

from ormah.models.node import CreateNodeRequest, NodeType

from tests.test_background.lock_probe import install_probe


def test_reentrant_acquisition_counts_once(engine):
    probe = install_probe(engine)
    with engine.memory_operation():
        with engine.memory_operation():
            pass
    assert probe.acquisitions == 1


def test_sequential_acquisitions_count_separately(engine):
    probe = install_probe(engine)
    with engine.memory_operation():
        pass
    with engine.memory_operation():
        pass
    assert probe.acquisitions == 2


def test_held_reports_the_calling_thread_only(engine):
    probe = install_probe(engine)
    inside = threading.Event()
    other_thread_saw = []

    def observer():
        inside.wait(timeout=5.0)
        other_thread_saw.append(probe.held)

    t = threading.Thread(target=observer, daemon=True)
    t.start()
    with engine.memory_operation():
        assert probe.held is True
        inside.set()
        t.join(timeout=5.0)
    assert probe.held is False
    assert other_thread_saw == [False]


def test_probe_covers_file_store_writes(engine):
    """FileStore keeps its own reference to L_mem; the probe must reach it."""
    probe = install_probe(engine)
    engine.remember(CreateNodeRequest(
        content="probe reaches the file store", type=NodeType.fact, title="probe"))
    assert probe.acquisitions >= 1
    assert probe.held is False

"""The embedding_backfill job must be registered with a post-bind first run (#32)."""
from __future__ import annotations

import threading
from unittest.mock import patch

from ormah.background.scheduler import start_scheduler


def test_embedding_backfill_job_registered(engine):
    scheduler, _tracker = start_scheduler(engine)
    try:
        job = scheduler.get_job("embedding_backfill")
        assert job is not None
        assert job.name == "Embedding backfill"
        # next_run_time is set (post-bind first run), not None/deferred
        assert job.next_run_time is not None
    finally:
        scheduler.shutdown(wait=False)


def test_scheduler_passes_stop_event_to_embedding_backfill(engine):
    """C2: start_scheduler deve injetar o stop_event no job embedding_backfill."""
    evt = threading.Event()
    received_kwargs = {}

    def _fake_run(eng, stop_event=None):
        received_kwargs["stop_event"] = stop_event

    with patch(
        "ormah.background.embedding_backfill.run_embedding_backfill",
        side_effect=_fake_run,
    ):
        scheduler, _tracker = start_scheduler(engine, stop_event=evt)
        try:
            job = scheduler.get_job("embedding_backfill")
            assert job is not None, "job embedding_backfill deve estar registrado"
            # Executa o job diretamente (sem APScheduler thread pool)
            job.func()
        finally:
            scheduler.shutdown(wait=False)

    assert "stop_event" in received_kwargs, "run_embedding_backfill não foi chamado"
    assert received_kwargs["stop_event"] is evt, (
        f"stop_event injetado deve ser o mesmo evento passado a start_scheduler; "
        f"recebido: {received_kwargs['stop_event']!r}"
    )

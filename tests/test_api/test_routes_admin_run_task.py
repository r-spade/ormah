"""The manual task-trigger routes must not start a job that is already running,
and must report what actually happened (#117)."""

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from ormah.api.routes_admin import router as admin_router
from ormah.background.job_tracker import JobTracker
from ormah.config import Settings
from ormah.engine.memory_engine import MemoryEngine


@pytest.fixture
def app_and_client(tmp_memory_dir):
    settings = Settings(memory_dir=tmp_memory_dir, backup_dir=tmp_memory_dir.parent / "backups")
    engine = MemoryEngine(settings)
    engine.startup()

    app = FastAPI()
    app.include_router(admin_router)
    app.state.engine = engine
    app.state.job_tracker = JobTracker()

    with TestClient(app) as c:
        yield app, c

    engine.shutdown()


def test_run_task_rejects_a_job_that_is_already_running(app_and_client):
    """A manual trigger during the scheduled run used to start a second concurrent
    run over the same watermark (#117). It must 409 instead."""
    app, client = app_and_client

    with app.state.job_tracker.run_guard("auto_linker") as acquired:
        assert acquired is True
        resp = client.post("/admin/tasks/auto_linker/run")

    assert resp.status_code == 409
    assert "already running" in resp.json()["detail"].lower()


def test_run_task_reports_a_failure_instead_of_completed(app_and_client, monkeypatch):
    """The route returned {'status': 'completed'} unconditionally — a run that blew
    up was reported to the caller as a success."""
    _app, client = app_and_client
    import ormah.background.auto_linker as al

    def boom(_engine):
        raise RuntimeError("watermark exploded")

    monkeypatch.setattr(al, "run_auto_linker", boom)
    resp = client.post("/admin/tasks/auto_linker/run")

    assert resp.status_code == 500
    assert "watermark exploded" in resp.json()["detail"]


def test_run_task_returns_the_stats_on_success(app_and_client, monkeypatch):
    _app, client = app_and_client
    import ormah.background.auto_linker as al

    monkeypatch.setattr(al, "run_auto_linker", lambda _e: None)
    resp = client.post("/admin/tasks/auto_linker/run")

    assert resp.status_code == 200
    assert resp.json()["status"] == "completed"


def test_run_all_skips_a_task_that_is_already_running(app_and_client, monkeypatch):
    """run-all calls the runners directly too — same hole."""
    app, client = app_and_client
    import ormah.background.auto_linker as al

    calls = []
    monkeypatch.setattr(al, "run_auto_linker", lambda _e: calls.append(1))

    with app.state.job_tracker.run_guard("auto_linker"):
        resp = client.post("/admin/tasks/run-all")

    assert calls == []                                  # never started a second run
    assert resp.json()["results"]["auto_linker"] == "skipped: already running"
    assert resp.json()["status"] == "partial"   # skipped work is not a completed cycle


def test_lifespan_always_creates_a_job_tracker_even_if_the_scheduler_fails(tmp_memory_dir, monkeypatch):
    """The guard is only atomic against a SHARED tracker. It was created inside the
    scheduler's try block, so a failed scheduler startup left app.state without one and
    _guard degraded to a no-op — two concurrent triggers could then run the same
    edge-writing job at once (Codex, PR B)."""
    import asyncio

    from fastapi import FastAPI

    import ormah.background.scheduler as sched
    from ormah.main import lifespan

    def boom(*_a, **_kw):
        raise RuntimeError("scheduler is down")

    monkeypatch.setattr(sched, "start_scheduler", boom)
    monkeypatch.setenv("ORMAH_MEMORY_DIR", str(tmp_memory_dir))
    monkeypatch.setenv("ORMAH_SESSION_WATCHER_ENABLED", "false")

    app = FastAPI()

    async def drive():
        async with lifespan(app):
            return getattr(app.state, "job_tracker", None)

    tracker = asyncio.run(drive())

    assert tracker is not None, "no tracker -> the single-flight guard silently degrades to a no-op"
    assert tracker.is_running("auto_linker") is False

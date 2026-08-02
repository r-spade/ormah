"""Tests for API routes."""

import threading
import time

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from ormah.api.routes_admin import router as admin_router
from ormah.api.routes_agent import router as agent_router
from ormah.api.routes_stats import router as stats_router
from ormah.api.routes_ui import router as ui_router
from ormah.background.maintenance_manager import MaintenanceManager
from ormah.config import Settings
from ormah.engine.memory_engine import MemoryEngine


@pytest.fixture
def client(tmp_memory_dir):
    settings = Settings(memory_dir=tmp_memory_dir, backup_dir=tmp_memory_dir.parent / "backups")
    engine = MemoryEngine(settings)
    engine.startup()

    # Create a fresh app without the production lifespan to avoid
    # writing to the real memory directory
    test_app = FastAPI()
    test_app.include_router(agent_router)
    test_app.include_router(admin_router)
    test_app.include_router(stats_router)
    test_app.include_router(ui_router)
    test_app.state.engine = engine
    test_app.state.maintenance_manager = MaintenanceManager(engine)

    with TestClient(test_app) as c:
        yield c

    engine.shutdown()


def test_health(client):
    resp = client.get("/admin/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


def test_remember_and_recall(client):
    # Remember
    resp = client.post("/agent/remember", json={
        "content": "Test memory content.",
        "type": "fact",
        "title": "Test memory",
    })
    assert resp.status_code == 200
    data = resp.json()
    assert data["node_id"] is not None

    node_id = data["node_id"]

    # Recall by ID
    resp = client.get(f"/agent/recall/{node_id}")
    assert resp.status_code == 200
    assert "Test memory" in resp.json()["text"]


def test_recall_not_found(client):
    resp = client.get("/agent/recall/nonexistent-id")
    assert resp.status_code == 404


def test_stats(client):
    resp = client.get("/stats")
    assert resp.status_code == 200
    body = resp.json()
    assert "total_nodes" in body["store"]
    assert "feedback_health" in body["whisper"]

    assert client.get("/admin/stats").status_code == 404
    assert client.get("/agent/stats").status_code == 404


def test_backup_status_empty_store(client):
    resp = client.get("/admin/backup")
    assert resp.status_code == 200
    data = resp.json()
    assert data["enabled"] is True
    assert data["latest"] is None
    assert data["has_backupable_memory"] is False
    assert data["due"] is False
    assert data["backup_dir"].endswith("backups")


def test_backup_create_returns_updated_status(client):
    resp = client.post("/admin/backup/create")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "created"
    assert data["backup"]["name"].startswith("memory_")
    assert data["backup_status"]["latest"]["name"] == data["backup"]["name"]


def test_backup_settings_updates_runtime_config(client, tmp_path, monkeypatch):
    persisted = {}

    def fake_persist(backup_dir, retention_count):
        persisted["backup_dir"] = backup_dir
        persisted["retention_count"] = retention_count

    monkeypatch.setattr("ormah.api.routes_admin._persist_backup_settings", fake_persist)

    target_dir = tmp_path / "chosen-backups"
    resp = client.post(
        "/admin/backup/settings",
        json={"backup_dir": str(target_dir), "retention_count": 4},
    )

    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "updated"
    assert data["backup_status"]["backup_dir"] == str(target_dir.resolve())
    assert data["backup_status"]["retention_count"] == 4
    assert client.app.state.engine.settings.backup_dir == target_dir.resolve()
    assert client.app.state.engine.settings.backup_retention_count == 4
    assert persisted == {"backup_dir": target_dir.resolve(), "retention_count": 4}


def test_backup_settings_rejects_zero_retention(client):
    resp = client.post(
        "/admin/backup/settings",
        json={"backup_dir": "/tmp/ormah-backups", "retention_count": 0},
    )

    assert resp.status_code == 422


def test_search(client):
    # Add a memory first
    client.post("/agent/remember", json={
        "content": "SQLite is used for the graph index.",
        "type": "decision",
        "title": "SQLite choice",
    })

    resp = client.post("/agent/recall", json={"query": "sqlite graph"})
    assert resp.status_code == 200


def test_context_endpoint_removed(client):
    resp = client.get("/agent/context")
    assert resp.status_code == 404


def test_connect(client):
    r1 = client.post("/agent/remember", json={"content": "A", "type": "fact"})
    r2 = client.post("/agent/remember", json={"content": "B", "type": "fact"})

    resp = client.post("/agent/connect", json={
        "source_id": r1.json()["node_id"],
        "target_id": r2.json()["node_id"],
        "edge": "related_to",
    })
    assert resp.status_code == 200
    assert "Connected" in resp.json()["text"]


def test_maintenance_runs_in_background_and_stats_stay_available(client):
    app = client.app
    started = threading.Event()
    release = threading.Event()
    original = app.state.engine.get_maintenance_batches

    def slow_batches():
        started.set()
        release.wait(timeout=5)
        return {
            "link_candidates": [],
            "conflict_candidates": [],
            "merge_candidates": [],
            "consolidation_clusters": [],
            "summary": "nothing to process",
        }

    app.state.engine.get_maintenance_batches = slow_batches
    try:
        resp = client.post("/agent/maintenance", json={})
        assert resp.status_code == 202
        assert resp.json()["status"] == "running_phase1"
        assert started.wait(timeout=1)

        stats = client.get("/stats")
        assert stats.status_code == 200

        release.set()
        deadline = time.time() + 2
        status = None
        while time.time() < deadline:
            poll = client.get("/agent/maintenance", params={"job_id": resp.json()["job_id"]})
            status = poll.json()
            if status["status"] == "awaiting_results":
                break
            time.sleep(0.05)
        assert status is not None
        assert status["status"] == "awaiting_results"
        assert "batches" in status
    finally:
        app.state.engine.get_maintenance_batches = original
        release.set()


def test_maintenance_reuses_single_inflight_job(client):
    app = client.app
    release = threading.Event()
    original = app.state.engine.get_maintenance_batches

    def slow_batches():
        release.wait(timeout=5)
        return {
            "link_candidates": [],
            "conflict_candidates": [],
            "merge_candidates": [],
            "consolidation_clusters": [],
            "summary": "nothing to process",
        }

    app.state.engine.get_maintenance_batches = slow_batches
    try:
        first = client.post("/agent/maintenance", json={})
        second = client.post("/agent/maintenance", json={})

        assert first.status_code == 202
        assert second.status_code == 202
        assert first.json()["job_id"] == second.json()["job_id"]
    finally:
        app.state.engine.get_maintenance_batches = original
        release.set()


def test_maintenance_phase2_apply_completes_via_routes(client):
    app = client.app
    original_batches = app.state.engine.get_maintenance_batches
    original_apply = app.state.engine.apply_maintenance_results

    def ready_batches():
        return {
            "link_candidates": [],
            "conflict_candidates": [],
            "merge_candidates": [],
            "consolidation_clusters": [],
            "summary": "nothing to process",
        }

    def apply_results(results):
        assert results == {"edges": []}
        return {"edges": 1, "merges": 0, "consolidations": 0, "skipped": 0}

    app.state.engine.get_maintenance_batches = ready_batches
    app.state.engine.apply_maintenance_results = apply_results
    try:
        start = client.post("/agent/maintenance", json={})
        assert start.status_code in {200, 202}
        job_id = start.json()["job_id"]
        phase1 = start.json()
        deadline = time.time() + 2
        while phase1["status"] != "awaiting_results" and time.time() < deadline:
            poll = client.get("/agent/maintenance", params={"job_id": job_id})
            phase1 = poll.json()
            if phase1["status"] == "awaiting_results":
                break
            time.sleep(0.05)
        assert phase1 is not None
        assert phase1["status"] == "awaiting_results"

        phase2 = client.post(
            "/agent/maintenance",
            json={"job_id": job_id, "results": {"edges": []}},
        )
        assert phase2.status_code == 202
        assert phase2.json()["status"] in {"running_phase2", "completed"}

        deadline = time.time() + 2
        final = phase2.json()
        while final["status"] != "completed" and time.time() < deadline:
            poll = client.get("/agent/maintenance", params={"job_id": job_id})
            final = poll.json()
            if final["status"] == "completed":
                break
            time.sleep(0.05)
        assert final is not None
        assert final["status"] == "completed"
        assert final["apply_summary"] == {
            "edges": 1,
            "merges": 0,
            "consolidations": 0,
            "skipped": 0,
        }
    finally:
        app.state.engine.get_maintenance_batches = original_batches
        app.state.engine.apply_maintenance_results = original_apply


def _wait_for_status(manager, job_id, not_status="running_phase1", timeout=2):
    deadline = time.time() + timeout
    status = manager.get_status(job_id=job_id)
    while status["status"] == not_status and time.time() < deadline:
        time.sleep(0.01)
        status = manager.get_status(job_id=job_id)
    return status


def test_phase1_finder_failure_is_recorded_as_tracker_failure(engine):
    """Issue #90 (dev council follow-up): a Phase 1 with a broken finder must
    still deliver the healthy batches to Phase 2 (status stays
    "awaiting_results", job.batches is kept) but must NOT look like a clean
    success in JobTracker — /admin is the #90 observability surface and must
    not lie about a maintenance cycle whose link candidates never computed."""
    from ormah.background.job_tracker import JobTracker

    def batches_with_error():
        return {
            "batch_errors": {"link_candidates": "RuntimeError: boom"},
            "link_candidates": [],
            "conflict_candidates": [{"fake": "candidate"}],
            "merge_candidates": [],
            "consolidation_clusters": [],
            "summary": "1 conflict candidates; FAILED: link_candidates",
        }

    engine.get_maintenance_batches = batches_with_error
    tracker = JobTracker()
    manager = MaintenanceManager(engine, tracker=tracker)

    payload = manager.start_phase1()
    status = _wait_for_status(manager, payload["job_id"])

    # Healthy batches must still flow to phase 2.
    assert status["status"] == "awaiting_results"
    assert status["batches"]["conflict_candidates"] == [{"fake": "candidate"}]

    # But the tracker must show this as a failure, not a clean run.
    snap = tracker.snapshot()["maintenance_phase1"]
    assert snap["error_count"] == 1
    assert snap["last_success"] is None
    assert "link_candidates" in snap["last_error"]
    assert snap["last_stats"]["batch_errors"] == {"link_candidates": "RuntimeError: boom"}


def test_phase1_clean_run_still_records_tracker_success(engine):
    from ormah.background.job_tracker import JobTracker

    def clean_batches():
        return {
            "batch_errors": {},
            "link_candidates": [],
            "conflict_candidates": [],
            "merge_candidates": [],
            "consolidation_clusters": [],
            "summary": "nothing to process",
        }

    engine.get_maintenance_batches = clean_batches
    tracker = JobTracker()
    manager = MaintenanceManager(engine, tracker=tracker)

    payload = manager.start_phase1()
    status = _wait_for_status(manager, payload["job_id"])

    assert status["status"] == "awaiting_results"
    snap = tracker.snapshot()["maintenance_phase1"]
    assert snap["error_count"] == 0
    assert snap["last_success"] is not None

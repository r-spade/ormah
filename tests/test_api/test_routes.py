"""Tests for API routes."""

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from ormah.api.routes_admin import router as admin_router
from ormah.api.routes_agent import router as agent_router
from ormah.api.routes_ui import router as ui_router
from ormah.config import Settings
from ormah.engine.memory_engine import MemoryEngine


@pytest.fixture
def client(tmp_memory_dir):
    settings = Settings(memory_dir=tmp_memory_dir)
    engine = MemoryEngine(settings)
    engine.startup()

    # Create a fresh app without the production lifespan to avoid
    # writing to the real memory directory
    test_app = FastAPI()
    test_app.include_router(agent_router)
    test_app.include_router(admin_router)
    test_app.include_router(ui_router)
    test_app.state.engine = engine

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


def test_context(client):
    resp = client.get("/agent/context")
    assert resp.status_code == 200


def test_stats(client):
    resp = client.get("/admin/stats")
    assert resp.status_code == 200
    assert "total_nodes" in resp.json()


def test_search(client):
    # Add a memory first
    client.post("/agent/remember", json={
        "content": "SQLite is used for the graph index.",
        "type": "decision",
        "title": "SQLite choice",
    })

    resp = client.post("/agent/recall", json={"query": "sqlite graph"})
    assert resp.status_code == 200


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

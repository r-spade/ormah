"""Tests for agent-facing power tools: update_memory, connect_memories, promote_memory."""

from __future__ import annotations

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from ormah.api.routes_agent import router as agent_router
from ormah.config import Settings
from ormah.engine.memory_engine import MemoryEngine
from ormah.models.node import CreateNodeRequest, NodeType, Tier, UpdateNodeRequest


@pytest.fixture
def client(tmp_memory_dir):
    settings = Settings(memory_dir=tmp_memory_dir)
    eng = MemoryEngine(settings)
    eng.startup()

    app = FastAPI()
    app.include_router(agent_router)
    app.state.engine = eng

    with TestClient(app) as c:
        yield c, eng

    eng.shutdown()


def _make_node(eng, content="Test content", tier=Tier.working, title="Test node"):
    req = CreateNodeRequest(content=content, type=NodeType.fact, title=title, tier=tier)
    node_id, _ = eng.remember(req)
    return node_id


# ---------------------------------------------------------------------------
# update_memory route
# ---------------------------------------------------------------------------


class TestUpdateMemoryRoute:

    def _get(self, eng, node_id):
        return eng.graph.get_node(node_id)

    def test_update_content(self, client):
        c, eng = client
        node_id = _make_node(eng, content="Original content")
        resp = c.post(f"/agent/update/{node_id}", json={"content": "Updated content"})
        assert resp.status_code == 200
        assert self._get(eng, node_id)["content"] == "Updated content"

    def test_update_title(self, client):
        c, eng = client
        node_id = _make_node(eng, title="Old title")
        resp = c.post(f"/agent/update/{node_id}", json={"title": "New title"})
        assert resp.status_code == 200
        assert self._get(eng, node_id)["title"] == "New title"

    def test_update_tier(self, client):
        c, eng = client
        node_id = _make_node(eng, tier=Tier.working)
        resp = c.post(f"/agent/update/{node_id}", json={"tier": "archival"})
        assert resp.status_code == 200
        assert self._get(eng, node_id)["tier"] == "archival"

    def test_update_confidence(self, client):
        c, eng = client
        node_id = _make_node(eng)
        resp = c.post(f"/agent/update/{node_id}", json={"confidence": 0.6})
        assert resp.status_code == 200
        assert abs(self._get(eng, node_id)["confidence"] - 0.6) < 1e-6

    def test_update_unknown_node_returns_404(self, client):
        c, _ = client
        resp = c.post("/agent/update/nonexistent-id", json={"content": "x"})
        assert resp.status_code == 404

    def test_update_multiple_fields(self, client):
        c, eng = client
        node_id = _make_node(eng, content="old", title="old title")
        resp = c.post(f"/agent/update/{node_id}", json={"content": "new", "title": "new title"})
        assert resp.status_code == 200
        node = self._get(eng, node_id)
        assert node["content"] == "new"
        assert node["title"] == "new title"


# ---------------------------------------------------------------------------
# connect_memories route
# ---------------------------------------------------------------------------


class TestConnectMemoriesRoute:

    def test_connect_creates_edge(self, client):
        c, eng = client
        src = _make_node(eng, content="Source memory")
        tgt = _make_node(eng, content="Target memory")
        resp = c.post("/agent/connect", json={"source_id": src, "target_id": tgt})
        assert resp.status_code == 200
        edges = eng.db.conn.execute(
            "SELECT edge_type FROM edges WHERE source_id = ? AND target_id = ?", (src, tgt)
        ).fetchall()
        assert len(edges) == 1
        assert edges[0][0] == "related_to"

    def test_connect_with_edge_type(self, client):
        c, eng = client
        src = _make_node(eng, content="Decision A")
        tgt = _make_node(eng, content="Decision B")
        resp = c.post("/agent/connect", json={"source_id": src, "target_id": tgt, "edge": "supports"})
        assert resp.status_code == 200
        row = eng.db.conn.execute(
            "SELECT edge_type FROM edges WHERE source_id = ? AND target_id = ?", (src, tgt)
        ).fetchone()
        assert row[0] == "supports"

    def test_connect_with_weight(self, client):
        c, eng = client
        src = _make_node(eng)
        tgt = _make_node(eng)
        resp = c.post("/agent/connect", json={"source_id": src, "target_id": tgt, "weight": 0.9})
        assert resp.status_code == 200
        row = eng.db.conn.execute(
            "SELECT weight FROM edges WHERE source_id = ? AND target_id = ?", (src, tgt)
        ).fetchone()
        assert abs(row[0] - 0.9) < 1e-6

    def test_connect_invalid_source_returns_error(self, client):
        c, eng = client
        tgt = _make_node(eng)
        resp = c.post("/agent/connect", json={"source_id": "bad-id", "target_id": tgt})
        assert resp.status_code == 200
        assert "not found" in resp.json()["text"].lower()

    def test_connect_invalid_target_returns_error(self, client):
        c, eng = client
        src = _make_node(eng)
        resp = c.post("/agent/connect", json={"source_id": src, "target_id": "bad-id"})
        assert resp.status_code == 200
        assert "not found" in resp.json()["text"].lower()


# ---------------------------------------------------------------------------
# promote_memory route
# ---------------------------------------------------------------------------


class TestPromoteMemoryRoute:

    def _tier(self, eng, node_id):
        return eng.graph.get_node(node_id)["tier"]

    def test_promote_working_to_core(self, client):
        c, eng = client
        node_id = _make_node(eng, tier=Tier.working)
        resp = c.post(f"/agent/promote/{node_id}", json={"tier": "core"})
        assert resp.status_code == 200
        assert self._tier(eng, node_id) == "core"

    def test_promote_archival_to_working(self, client):
        c, eng = client
        node_id = _make_node(eng, tier=Tier.archival)
        resp = c.post(f"/agent/promote/{node_id}", json={"tier": "working"})
        assert resp.status_code == 200
        assert self._tier(eng, node_id) == "working"

    def test_promote_default_tier_is_core(self, client):
        c, eng = client
        node_id = _make_node(eng, tier=Tier.working)
        resp = c.post(f"/agent/promote/{node_id}", json={})
        assert resp.status_code == 200
        assert self._tier(eng, node_id) == "core"

    def test_promote_already_at_tier_returns_ok(self, client):
        c, eng = client
        node_id = _make_node(eng, tier=Tier.core)
        resp = c.post(f"/agent/promote/{node_id}", json={"tier": "core"})
        assert resp.status_code == 200
        assert "already" in resp.json()["text"].lower()

    def test_promote_unknown_node_returns_404(self, client):
        c, _ = client
        resp = c.post("/agent/promote/nonexistent", json={"tier": "core"})
        assert resp.status_code == 404

    def test_promote_invalid_tier_returns_422(self, client):
        c, eng = client
        node_id = _make_node(eng)
        resp = c.post(f"/agent/promote/{node_id}", json={"tier": "invalid"})
        assert resp.status_code == 422

    def test_promote_respects_core_cap(self, client):
        c, eng = client
        # Set a very small cap
        eng.tier_manager.core_cap = 2
        # Fill core tier
        id1 = _make_node(eng, tier=Tier.core, title="Core 1", content="Core memory 1")
        id2 = _make_node(eng, tier=Tier.core, title="Core 2", content="Core memory 2")
        # Promote a 3rd node — cap enforced, least important core gets demoted
        id3 = _make_node(eng, tier=Tier.working, title="Promote me", content="Important new memory")
        resp = c.post(f"/agent/promote/{id3}", json={"tier": "core"})
        assert resp.status_code == 200
        # Count core nodes — should still be <= 2 after cap enforcement
        core_count = eng.db.conn.execute(
            "SELECT COUNT(*) FROM nodes WHERE tier = 'core'"
        ).fetchone()[0]
        assert core_count <= 2

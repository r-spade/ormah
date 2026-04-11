"""Tests for memory versioning — new snapshot storage and recall_history."""

from __future__ import annotations

import json

import pytest

from ormah.models.node import CreateNodeRequest, NodeType, Tier, UpdateNodeRequest


def _make_node(engine, title="Versioned node", content="Original content"):
    req = CreateNodeRequest(content=content, type=NodeType.fact, title=title)
    node_id, _ = engine.remember(req)
    return node_id


# ---------------------------------------------------------------------------
# New snapshot stored in detail
# ---------------------------------------------------------------------------


class TestNewSnapshotStorage:

    def test_update_stores_new_snapshot_in_detail(self, engine):
        node_id = _make_node(engine)
        engine.update_node(node_id, UpdateNodeRequest(content="Updated content"))

        entries = engine.list_audit_log(node_id=node_id, operation="update")
        assert len(entries) == 1
        detail = json.loads(entries[0]["detail"])
        assert "new_snapshot" in detail
        assert detail["new_snapshot"]["content"] == "Updated content"

    def test_update_stores_old_snapshot_in_node_snapshot(self, engine):
        node_id = _make_node(engine, content="Before update")
        engine.update_node(node_id, UpdateNodeRequest(content="After update"))

        entries = engine.list_audit_log(node_id=node_id, operation="update")
        old = json.loads(entries[0]["node_snapshot"])
        assert old["content"] == "Before update"

    def test_update_records_changed_fields(self, engine):
        node_id = _make_node(engine, title="Old title")
        engine.update_node(node_id, UpdateNodeRequest(title="New title"))

        entries = engine.list_audit_log(node_id=node_id, operation="update")
        detail = json.loads(entries[0]["detail"])
        assert "title" in detail["changed_fields"]

    def test_multiple_updates_produce_multiple_entries(self, engine):
        node_id = _make_node(engine)
        engine.update_node(node_id, UpdateNodeRequest(content="v2"))
        engine.update_node(node_id, UpdateNodeRequest(content="v3"))

        entries = engine.list_audit_log(node_id=node_id, operation="update")
        assert len(entries) == 2

    def test_tier_change_stored_in_new_snapshot(self, engine):
        node_id = _make_node(engine)
        engine.update_node(node_id, UpdateNodeRequest(tier=Tier.archival))

        entries = engine.list_audit_log(node_id=node_id, operation="update")
        detail = json.loads(entries[0]["detail"])
        assert detail["new_snapshot"]["tier"] == "archival"


# ---------------------------------------------------------------------------
# recall_history formatting
# ---------------------------------------------------------------------------


class TestRecallHistory:

    def test_no_history_returns_not_found(self, engine):
        result = engine.recall_history("nonexistent-id")
        assert "not found" in result.lower()

    def test_empty_history_message(self, engine):
        node_id = _make_node(engine)
        result = engine.recall_history(node_id)
        assert "No history" in result

    def test_update_history_shows_diff(self, engine):
        node_id = _make_node(engine, content="original body")
        engine.update_node(node_id, UpdateNodeRequest(content="updated body"))

        result = engine.recall_history(node_id)
        assert "UPDATE" in result
        assert "original body" in result
        assert "updated body" in result

    def test_history_shows_title_change(self, engine):
        node_id = _make_node(engine, title="Old title")
        engine.update_node(node_id, UpdateNodeRequest(title="New title"))

        result = engine.recall_history(node_id)
        assert "Old title" in result
        assert "New title" in result

    def test_history_shows_mark_outdated(self, engine):
        node_id = _make_node(engine)
        engine.mark_outdated(node_id, reason="superseded by v2")

        result = engine.recall_history(node_id)
        assert "MARK_OUTDATED" in result
        assert "superseded by v2" in result

    def test_history_chronological_order(self, engine):
        node_id = _make_node(engine, content="v1")
        engine.update_node(node_id, UpdateNodeRequest(content="v2"))
        engine.update_node(node_id, UpdateNodeRequest(content="v3"))

        result = engine.recall_history(node_id)
        v1_pos = result.find("v1")
        v2_pos = result.find("v2")
        v3_pos = result.find("v3")
        assert v1_pos < v2_pos < v3_pos


# ---------------------------------------------------------------------------
# HTTP route
# ---------------------------------------------------------------------------


class TestRecallHistoryRoute:

    @pytest.fixture
    def client(self, tmp_memory_dir):
        from fastapi import FastAPI
        from fastapi.testclient import TestClient

        from ormah.api.routes_agent import router as agent_router
        from ormah.config import Settings
        from ormah.engine.memory_engine import MemoryEngine

        settings = Settings(memory_dir=tmp_memory_dir)
        eng = MemoryEngine(settings)
        eng.startup()

        test_app = FastAPI()
        test_app.include_router(agent_router)
        test_app.state.engine = eng

        with TestClient(test_app) as c:
            yield c, eng

        eng.shutdown()

    def test_history_route_returns_200(self, client):
        c, eng = client
        req = CreateNodeRequest(content="test content", type=NodeType.fact, title="Route test")
        node_id, _ = eng.remember(req)
        eng.update_node(node_id, UpdateNodeRequest(content="updated content"))

        resp = c.get(f"/agent/recall/{node_id}/history")
        assert resp.status_code == 200
        assert "UPDATE" in resp.json()["text"]

    def test_history_route_unknown_node(self, client):
        c, _ = client
        resp = c.get("/agent/recall/nonexistent/history")
        assert resp.status_code == 200
        assert "not found" in resp.json()["text"].lower()

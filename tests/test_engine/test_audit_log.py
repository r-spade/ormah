"""Tests for audit logging on delete, update, and mark_outdated."""

from __future__ import annotations

import json

from ormah.models.node import CreateNodeRequest, NodeType, UpdateNodeRequest


def _create_node(engine, title="Test", content="Test content", **kwargs):
    """Helper to create a node, returns (id, slug)."""
    req = CreateNodeRequest(content=content, type=NodeType.fact, title=title, tags=["test"], **kwargs)
    return engine.remember(req, agent_id="test")


def test_delete_creates_audit_entry(engine):
    """Deleting a node writes a full snapshot to the audit log."""
    node_id, _ = _create_node(engine, title="To delete", content="Will be deleted")

    # Load original for comparison
    original = engine.file_store.load(node_id)
    assert original is not None

    engine.delete_node(node_id)

    entries = engine.list_audit_log()
    assert len(entries) == 1
    entry = entries[0]
    assert entry["operation"] == "delete"
    assert entry["node_id"] == node_id
    assert entry["performed_at"] is not None

    # Verify snapshot contains the original node data
    snapshot = json.loads(entry["node_snapshot"])
    assert snapshot["id"] == node_id
    assert snapshot["title"] == "To delete"
    assert snapshot["content"] == "Will be deleted"


def test_update_creates_audit_entry(engine):
    """Updating a node logs the old state and changed fields."""
    node_id, _ = _create_node(engine, title="Original title", content="Original content")

    engine.update_node(node_id, UpdateNodeRequest(title="New title", content="New content"))

    entries = engine.list_audit_log()
    assert len(entries) == 1
    entry = entries[0]
    assert entry["operation"] == "update"
    assert entry["node_id"] == node_id

    # Snapshot should have the OLD values
    snapshot = json.loads(entry["node_snapshot"])
    assert snapshot["title"] == "Original title"
    assert snapshot["content"] == "Original content"

    # Detail should list what changed
    detail = json.loads(entry["detail"])
    assert "title" in detail["changed_fields"]
    assert "content" in detail["changed_fields"]


def test_mark_outdated_creates_audit_entry(engine):
    """Marking a node outdated logs the reason and old valid_until."""
    node_id, _ = _create_node(engine, title="Active fact", content="Still valid")

    engine.mark_outdated(node_id, reason="No longer accurate")

    entries = engine.list_audit_log()
    assert len(entries) == 1
    entry = entries[0]
    assert entry["operation"] == "mark_outdated"
    assert entry["node_id"] == node_id

    detail = json.loads(entry["detail"])
    assert detail["reason"] == "No longer accurate"
    assert detail["old_valid_until"] is None  # was not set before


def test_list_audit_log_filters(engine):
    """list_audit_log filters by node_id and operation."""
    id_a, _ = _create_node(engine, title="Node A", content="Content A")
    id_b, _ = _create_node(engine, title="Node B", content="Content B")

    engine.update_node(id_a, UpdateNodeRequest(title="Updated A"))
    engine.delete_node(id_b)
    engine.mark_outdated(id_a, reason="old")

    # All entries
    all_entries = engine.list_audit_log()
    assert len(all_entries) == 3

    # Filter by node_id
    a_entries = engine.list_audit_log(node_id=id_a)
    assert len(a_entries) == 2
    assert all(e["node_id"] == id_a for e in a_entries)

    b_entries = engine.list_audit_log(node_id=id_b)
    assert len(b_entries) == 1
    assert b_entries[0]["operation"] == "delete"

    # Filter by operation
    deletes = engine.list_audit_log(operation="delete")
    assert len(deletes) == 1
    assert deletes[0]["node_id"] == id_b

    updates = engine.list_audit_log(operation="update")
    assert len(updates) == 1

    outdated = engine.list_audit_log(operation="mark_outdated")
    assert len(outdated) == 1

    # Filter by both
    both = engine.list_audit_log(node_id=id_a, operation="update")
    assert len(both) == 1


def test_delete_node_soft_deletes_file(engine):
    """delete_node should move the markdown file to deleted/ instead of removing it."""
    node_id, _ = _create_node(engine, title="Soft delete test", content="Should survive in deleted dir")

    # Get the file path before deletion
    path = engine.file_store._find_file(node_id)
    assert path is not None and path.exists()
    filename = path.name

    engine.delete_node(node_id)

    # Original file should be gone
    assert not path.exists()

    # File should exist in deleted/ directory
    deleted_dir = engine.file_store.nodes_dir.parent / "deleted"
    dest = deleted_dir / filename
    assert dest.exists()

    # Node should no longer be loadable from store
    assert engine.file_store.load(node_id) is None

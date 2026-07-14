"""Tests for markdown parsing and serialization."""

from ormah.models.node import Connection, EdgeType, MemoryNode, NodeType, Tier
from ormah.store.markdown import parse_node, serialize_node


def test_roundtrip():
    node = MemoryNode(
        type=NodeType.decision,
        tier=Tier.working,
        source="agent:test",
        title="Test decision",
        content="Chose option A over option B.",
        tags=["test", "demo"],
        space="testing",
        connections=[
            Connection(target="abc123", edge=EdgeType.supports, weight=0.9),
        ],
    )

    text = serialize_node(node)
    parsed = parse_node(text)

    assert parsed.id == node.id
    assert parsed.type == NodeType.decision
    assert parsed.tier == Tier.working
    assert parsed.source == "agent:test"
    assert parsed.title == "Test decision"
    assert parsed.content == "Chose option A over option B."
    assert parsed.tags == ["test", "demo"]
    assert parsed.space == "testing"
    assert len(parsed.connections) == 1
    assert parsed.connections[0].target == "abc123"
    assert parsed.connections[0].edge == EdgeType.supports


def test_parse_minimal():
    text = """---
id: test-id-123
type: fact
source: agent:test
created: 2026-01-01T00:00:00Z
updated: 2026-01-01T00:00:00Z
---
This is a simple fact."""

    node = parse_node(text)
    assert node.id == "test-id-123"
    assert node.type == NodeType.fact
    assert node.content == "This is a simple fact."
    assert node.tier == Tier.working  # default
    assert node.connections == []
    assert node.deleted_at is None  # legacy files without deleted_at parse fine


def test_deleted_at_roundtrip():
    from datetime import datetime, timezone

    stamp = datetime(2026, 7, 12, 10, 30, 0, tzinfo=timezone.utc)
    node = MemoryNode(
        type=NodeType.fact,
        source="agent:test",
        content="Tombstoned fact.",
        deleted_at=stamp,
    )

    text = serialize_node(node)
    assert "deleted_at" in text
    parsed = parse_node(text)
    assert parsed.deleted_at == stamp


def test_deleted_at_omitted_when_none():
    node = MemoryNode(type=NodeType.fact, source="agent:test", content="Live fact.")
    text = serialize_node(node)
    assert "deleted_at" not in text

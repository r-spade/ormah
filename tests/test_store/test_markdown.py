"""Tests for markdown parsing and serialization."""

from datetime import datetime, timezone

import frontmatter
import pytest

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


def test_superseded_by_survives_a_markdown_roundtrip():
    node = MemoryNode(type=NodeType.fact, content="a superseded source")
    node.superseded_by = "11111111-2222-3333-4444-555555555555"

    parsed = parse_node(serialize_node(node))

    assert parsed.superseded_by == "11111111-2222-3333-4444-555555555555"


def test_superseded_by_is_absent_from_frontmatter_when_none():
    """Without this assertion, writing `superseded_by: null` pollutes every node file."""
    node = MemoryNode(type=NodeType.fact, content="an ordinary node")

    text = serialize_node(node)

    assert "superseded_by" not in text
    assert parse_node(text).superseded_by is None


def test_a_file_written_before_223_parses_to_none():
    node = MemoryNode(type=NodeType.fact, content="written by an older build")
    text = serialize_node(node)
    assert parse_node(text).superseded_by is None


def test_stability_is_not_retroactively_rescaled():
    """#191 forbids rescaling values that predate the new default."""
    node = MemoryNode(type=NodeType.fact, content="an old node", stability=1.0)
    assert parse_node(serialize_node(node)).stability == 1.0


def test_a_file_with_no_stability_key_parses_to_one_not_the_new_default():
    """parse_node's `meta.get("stability", 1.0)` fallback must stay 1.0."""
    text = (
        "---\n"
        "id: 99999999-8888-7777-6666-555555555555\n"
        "type: fact\n"
        "tier: working\n"
        "source: agent:test\n"
        "created: 2026-01-01T00:00:00Z\n"
        "updated: 2026-01-01T00:00:00Z\n"
        "last_accessed: 2026-01-01T00:00:00Z\n"
        "---\n"
        "a node from before FSRS\n"
    )
    assert parse_node(text).stability == 1.0


# quoted-naive is the case from r-spade/ormah#237: the only one that parsed naive before the fix.
@pytest.mark.parametrize(
    ("stamp", "raw_type", "raw_naive"),
    [
        ("2026-01-01T12:00:00", datetime, True),
        ("2026-01-01T12:00:00Z", datetime, False),
        ('"2026-01-01T12:00:00"', str, True),
        ('"2026-01-01T12:00:00Z"', str, False),
    ],
    ids=["unquoted-naive", "unquoted-utc", "quoted-naive", "quoted-utc"],
)
def test_a_frontmatter_timestamp_parses_to_utc(stamp, raw_type, raw_naive):
    text = f"""---
id: test-id-123
type: fact
source: agent:test
created: {stamp}
updated: {stamp}
---
A fact."""

    # The YAML loader decides which branch of _parse_dt runs: pin the raw value it hands over.
    raw = frontmatter.loads(text).metadata["created"]
    assert type(raw) is raw_type
    raw_dt = raw if isinstance(raw, datetime) else datetime.fromisoformat(raw)
    assert (raw_dt.tzinfo is None) is raw_naive

    node = parse_node(text)

    expected = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)
    assert node.created.tzinfo is not None
    assert node.created == expected
    assert node.updated == expected

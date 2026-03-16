"""Tests for enriched node fields in markdown serialization."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from ormah.models.node import MemoryNode, NodeType, Tier
from ormah.store.markdown import parse_node, serialize_node


def _make_node(**overrides) -> MemoryNode:
    defaults = dict(
        type=NodeType.fact,
        tier=Tier.working,
        title="Test node",
        content="Some content",
    )
    defaults.update(overrides)
    return MemoryNode(**defaults)


class TestLegacyParsing:
    """Nodes without the new fields should get sensible defaults."""

    def test_missing_confidence_defaults_to_1(self):
        node = _make_node()
        text = serialize_node(node)
        # Simulate a legacy file by stripping confidence line
        lines = [l for l in text.split("\n") if not l.startswith("confidence:")]
        legacy_text = "\n".join(lines)
        parsed = parse_node(legacy_text)
        assert parsed.confidence == 1.0

    def test_missing_importance_defaults_to_0_5(self):
        node = _make_node()
        text = serialize_node(node)
        lines = [l for l in text.split("\n") if not l.startswith("importance:")]
        legacy_text = "\n".join(lines)
        parsed = parse_node(legacy_text)
        assert parsed.importance == 0.5

    def test_missing_valid_until_defaults_to_none(self):
        node = _make_node()
        text = serialize_node(node)
        parsed = parse_node(text)
        assert parsed.valid_until is None


class TestRoundtrip:
    """Roundtrip serialization with all enrichment fields."""

    def test_roundtrip_all_fields(self):
        valid_until = datetime(2026, 12, 31, tzinfo=timezone.utc)
        node = _make_node(
            confidence=0.7,
            importance=0.85,
            valid_until=valid_until,
        )
        text = serialize_node(node)
        parsed = parse_node(text)

        assert parsed.confidence == 0.7
        assert parsed.importance == 0.85
        assert parsed.valid_until == valid_until

    def test_roundtrip_without_valid_until(self):
        node = _make_node(confidence=0.5, importance=0.3)
        text = serialize_node(node)
        parsed = parse_node(text)
        assert parsed.valid_until is None
        assert parsed.confidence == 0.5
        assert parsed.importance == 0.3

    def test_roundtrip_preserves_other_fields(self):
        node = _make_node(
            title="My title",
            content="My content",
            space="myproject",
            tags=["a", "b"],
            confidence=0.9,
        )
        text = serialize_node(node)
        parsed = parse_node(text)

        assert parsed.title == "My title"
        assert parsed.content == "My content"
        assert parsed.space == "myproject"
        assert parsed.tags == ["a", "b"]
        assert parsed.id == node.id


class TestConfidenceValidation:
    """Confidence must be between 0.0 and 1.0."""

    def test_confidence_too_high(self):
        with pytest.raises(Exception):
            _make_node(confidence=1.5)

    def test_confidence_too_low(self):
        with pytest.raises(Exception):
            _make_node(confidence=-0.1)

    def test_confidence_at_bounds(self):
        node_low = _make_node(confidence=0.0)
        assert node_low.confidence == 0.0
        node_high = _make_node(confidence=1.0)
        assert node_high.confidence == 1.0

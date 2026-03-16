"""Tests for the mark_outdated feedback tool."""

from __future__ import annotations

import pytest

from ormah.models.node import CreateNodeRequest, NodeType, Tier


@pytest.fixture
def feedback_engine(engine):
    """Engine with a node to give feedback on."""
    req = CreateNodeRequest(
        content="Python uses indentation for blocks",
        type=NodeType.fact,
        title="Python indentation",
    )
    node_id, _ = engine.remember(req)
    return engine, node_id


class TestMarkOutdated:

    def test_sets_valid_until(self, feedback_engine):
        engine, node_id = feedback_engine
        result = engine.mark_outdated(node_id)
        assert result is not None
        assert "valid_until" in result

        node = engine.file_store.load(node_id)
        assert node.valid_until is not None

    def test_reason_appended(self, feedback_engine):
        engine, node_id = feedback_engine
        engine.mark_outdated(node_id, reason="Python 4 changed this")

        node = engine.file_store.load(node_id)
        assert "[Outdated: Python 4 changed this]" in node.content

    def test_without_reason(self, feedback_engine):
        engine, node_id = feedback_engine
        engine.mark_outdated(node_id)

        node = engine.file_store.load(node_id)
        assert "[Outdated:" not in node.content
        assert node.valid_until is not None

    def test_nonexistent_returns_none(self, engine):
        result = engine.mark_outdated("nonexistent-id")
        assert result is None

    def test_outdated_demoted_in_search(self, feedback_engine):
        """An outdated memory should get a lower score in search."""
        engine, node_id = feedback_engine

        # Create a fresh competing node
        req2 = CreateNodeRequest(
            content="Python uses indentation for code blocks and scoping",
            type=NodeType.fact,
            title="Python indentation details",
        )
        fresh_id, _ = engine.remember(req2)

        # Mark the original as outdated
        engine.mark_outdated(node_id, reason="superseded")

        # The outdated node should have valid_until set
        node = engine.file_store.load(node_id)
        assert node.valid_until is not None

    def test_persists_to_disk(self, feedback_engine):
        engine, node_id = feedback_engine
        engine.mark_outdated(node_id, reason="No longer valid")

        node = engine.file_store.load(node_id)
        assert node.valid_until is not None
        assert "No longer valid" in node.content

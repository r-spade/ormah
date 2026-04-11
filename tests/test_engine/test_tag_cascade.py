"""Tests for tag-cascade invalidation in mark_outdated()."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from ormah.models.node import CreateNodeRequest, NodeType, Tier


class TestTagCascadeInvalidation:
    def test_siblings_surfaced_in_mark_outdated_response(self, engine):
        """mark_outdated() should list nodes that share tags."""
        node_a_id, _ = engine.remember(CreateNodeRequest(
            content="Decision about Python",
            type=NodeType.decision,
            tier=Tier.working,
            title="Python choice",
            tags=["python", "language"],
        ))
        node_b_id, _ = engine.remember(CreateNodeRequest(
            content="Another python note",
            type=NodeType.fact,
            tier=Tier.working,
            title="Python note",
            tags=["python"],
        ))

        result = engine.mark_outdated(node_a_id, reason="Switched to Go")

        assert "python" in result
        assert "Python note" in result
        assert node_b_id[:8] in result

    def test_no_cascade_when_node_has_no_tags(self, engine):
        """Nodes without tags should not trigger any cascade output."""
        node_id, _ = engine.remember(CreateNodeRequest(
            content="Untagged memory",
            type=NodeType.fact,
            tier=Tier.working,
            title="No tags",
            tags=[],
        ))

        result = engine.mark_outdated(node_id)

        assert "share tag" not in result
        assert "consider reviewing" not in result

    def test_already_outdated_siblings_excluded(self, engine):
        """Nodes already marked outdated should not appear in cascade."""
        node_a_id, _ = engine.remember(CreateNodeRequest(
            content="Stale memory",
            type=NodeType.fact,
            tier=Tier.working,
            title="Stale",
            tags=["legacy"],
        ))
        node_b_id, _ = engine.remember(CreateNodeRequest(
            content="Also stale memory",
            type=NodeType.fact,
            tier=Tier.working,
            title="Also stale",
            tags=["legacy"],
        ))

        # Mark B outdated first
        engine.mark_outdated(node_b_id, reason="Old")

        # Now mark A outdated — B should not appear in cascade
        result = engine.mark_outdated(node_a_id, reason="Old too")

        assert "Also stale" not in result

    def test_cascade_disabled_via_config(self, engine):
        """tag_cascade_invalidation_enabled=False skips cascade entirely."""
        engine.settings.tag_cascade_invalidation_enabled = False

        node_a_id, _ = engine.remember(CreateNodeRequest(
            content="First node",
            type=NodeType.fact,
            tier=Tier.working,
            title="Node A",
            tags=["shared-tag"],
        ))
        engine.remember(CreateNodeRequest(
            content="Second node",
            type=NodeType.fact,
            tier=Tier.working,
            title="Node B",
            tags=["shared-tag"],
        ))

        result = engine.mark_outdated(node_a_id)

        assert "share tag" not in result
        assert "consider reviewing" not in result

    def test_self_not_included_in_cascade(self, engine):
        """The invalidated node itself should not appear in its own cascade list."""
        node_id, _ = engine.remember(CreateNodeRequest(
            content="Self check",
            type=NodeType.fact,
            tier=Tier.working,
            title="Self",
            tags=["mytag"],
        ))

        result = engine.mark_outdated(node_id)

        # The node being invalidated should appear as the primary entry, but not
        # again in the cascade siblings list
        lines = result.split("\n")
        cascade_lines = [l for l in lines if l.strip().startswith("·")]
        for line in cascade_lines:
            assert node_id[:8] not in line


class TestFindTagSiblings:
    def test_returns_siblings_with_shared_tags(self, engine):
        """_find_tag_siblings returns nodes sharing any of the queried tags."""
        target_id, _ = engine.remember(CreateNodeRequest(
            content="Target node",
            type=NodeType.fact,
            tier=Tier.working,
            title="Target",
            tags=["a", "b"],
        ))
        sibling_id, _ = engine.remember(CreateNodeRequest(
            content="Sibling node",
            type=NodeType.fact,
            tier=Tier.working,
            title="Sibling",
            tags=["a"],
        ))

        results = engine._find_tag_siblings(node_id=target_id, tags=["a", "b"], space=None)

        ids = [r[0] for r in results]
        assert sibling_id in ids

    def test_respects_space_filter(self, engine):
        """Nodes in a different space should not appear in cascade."""
        target_id, _ = engine.remember(CreateNodeRequest(
            content="Project A node",
            type=NodeType.fact,
            tier=Tier.working,
            title="Proj A",
            tags=["shared"],
            space="project-a",
        ))
        other_id, _ = engine.remember(CreateNodeRequest(
            content="Project B node",
            type=NodeType.fact,
            tier=Tier.working,
            title="Proj B",
            tags=["shared"],
            space="project-b",
        ))

        results = engine._find_tag_siblings(
            node_id=target_id, tags=["shared"], space="project-a"
        )

        ids = [r[0] for r in results]
        assert other_id not in ids

    def test_max_results_limit_respected(self, engine):
        """Result count should not exceed limit."""
        target_id, _ = engine.remember(CreateNodeRequest(
            content="Target",
            type=NodeType.fact,
            tier=Tier.working,
            title="Target",
            tags=["bulk"],
        ))
        for i in range(5):
            engine.remember(CreateNodeRequest(
                content=f"Sibling {i}",
                type=NodeType.fact,
                tier=Tier.working,
                title=f"Sibling {i}",
                tags=["bulk"],
            ))

        results = engine._find_tag_siblings(node_id=target_id, tags=["bulk"], space=None, limit=3)

        assert len(results) <= 3

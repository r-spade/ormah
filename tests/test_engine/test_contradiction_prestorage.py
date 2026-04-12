"""Tests for the pre-storage contradiction check in MemoryEngine."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from ormah.models.node import CreateNodeRequest, NodeType, Tier


class TestContradictionPrestorage:
    def test_warning_appended_when_contradiction_found(self, engine):
        """remember() output should contain a warning when similar nodes exist."""
        with patch.object(
            engine,
            "_check_contradiction_prestorage",
            return_value=[("abc12345", "Conflicting fact", 0.92)],
        ):
            _, text = engine.remember(CreateNodeRequest(
                content="New fact that may conflict",
                type=NodeType.fact,
                title="New Fact",
            ))

        assert "⚠ Possible contradiction" in text
        assert "Conflicting fact" in text
        assert "abc12345" in text

    def test_no_warning_when_no_contradictions(self, engine):
        """remember() output should be clean when check returns empty list."""
        with patch.object(
            engine,
            "_check_contradiction_prestorage",
            return_value=[],
        ):
            _, text = engine.remember(CreateNodeRequest(
                content="Unique new fact",
                type=NodeType.fact,
                title="Unique Fact",
            ))

        assert "⚠" not in text
        assert "contradiction" not in text.lower()

    def test_check_disabled_via_config(self, engine):
        """When contradiction_prestorage_enabled=False, check should not run."""
        engine.settings.contradiction_prestorage_enabled = False

        with patch.object(
            engine, "_check_contradiction_prestorage"
        ) as mock_check:
            engine.remember(CreateNodeRequest(
                content="Fact with disabled check",
                type=NodeType.fact,
                title="Disabled Check",
            ))

        mock_check.assert_not_called()

    def test_check_enabled_by_default(self, engine):
        """contradiction_prestorage_enabled defaults to True."""
        assert engine.settings.contradiction_prestorage_enabled is True


class TestCheckContradictionPrestorage:
    def test_returns_empty_when_vector_store_unavailable(self, engine):
        """Should not raise if encoder/vector store import fails."""
        from ormah.models.node import MemoryNode, NodeType, Tier

        node = MemoryNode(
            type=NodeType.fact,
            tier=Tier.working,
            title="Test node",
            content="Test content",
        )

        with patch("ormah.embeddings.encoder.get_encoder", side_effect=ImportError):
            result = engine._check_contradiction_prestorage(node)

        assert result == []

    def test_cross_space_nodes_not_flagged(self, engine):
        """Nodes from a different space should never be flagged."""
        from ormah.models.node import MemoryNode, NodeType, Tier

        node = MemoryNode(
            type=NodeType.fact,
            tier=Tier.working,
            title="Project A fact",
            content="Using Postgres for project A",
            space="project-a",
        )

        # Simulate a high-similarity match from a different space
        with patch("ormah.embeddings.encoder.get_encoder") as mock_enc, \
             patch("ormah.embeddings.vector_store.VectorStore") as mock_vs:

            import numpy as np
            mock_encoder = mock_enc.return_value
            mock_encoder.encode.return_value = np.zeros(128, dtype="float32")

            mock_vs.return_value.search.return_value = [
                {"id": "other-node-id", "similarity": 0.95},
            ]

            # other node is in a different space
            with patch.object(
                engine.graph,
                "get_node",
                return_value={"id": "other-node-id", "space": "project-b", "title": "Project B fact", "content": ""},
            ):
                result = engine._check_contradiction_prestorage(node, similarity_threshold=0.88)

        assert result == []

    def test_same_space_high_similarity_flagged(self, engine):
        """High-similarity same-space node should appear in flagged list."""
        from ormah.models.node import MemoryNode, NodeType, Tier

        node = MemoryNode(
            type=NodeType.fact,
            tier=Tier.working,
            title="Same space fact",
            content="Uses Redis for caching",
            space="project-x",
        )

        with patch("ormah.embeddings.encoder.get_encoder") as mock_enc, \
             patch("ormah.embeddings.vector_store.VectorStore") as mock_vs:

            import numpy as np
            mock_encoder = mock_enc.return_value
            mock_encoder.encode.return_value = np.zeros(128, dtype="float32")

            mock_vs.return_value.search.return_value = [
                {"id": "existing-node-id", "similarity": 0.93},
            ]

            with patch.object(
                engine.graph,
                "get_node",
                return_value={
                    "id": "existing-node-id",
                    "space": "project-x",
                    "title": "Cache technology choice",
                    "content": "Uses Memcached for caching",
                },
            ):
                result = engine._check_contradiction_prestorage(node, similarity_threshold=0.88)

        assert len(result) == 1
        assert result[0][0] == "existing-node-id"
        assert result[0][2] == 0.93

    def test_below_threshold_not_flagged(self, engine):
        """Similarity below threshold should not produce a warning."""
        from ormah.models.node import MemoryNode, NodeType, Tier

        node = MemoryNode(
            type=NodeType.fact,
            tier=Tier.working,
            title="Some fact",
            content="Some content",
            space="project-x",
        )

        with patch("ormah.embeddings.encoder.get_encoder") as mock_enc, \
             patch("ormah.embeddings.vector_store.VectorStore") as mock_vs:

            import numpy as np
            mock_encoder = mock_enc.return_value
            mock_encoder.encode.return_value = np.zeros(128, dtype="float32")

            mock_vs.return_value.search.return_value = [
                {"id": "existing-node-id", "similarity": 0.75},
            ]

            with patch.object(
                engine.graph,
                "get_node",
                return_value={"id": "existing-node-id", "space": "project-x", "title": "Related", "content": ""},
            ):
                result = engine._check_contradiction_prestorage(node, similarity_threshold=0.88)

        assert result == []

    def test_self_excluded_from_results(self, engine):
        """The node being stored should not flag itself as a contradiction."""
        from ormah.models.node import MemoryNode, NodeType, Tier

        node = MemoryNode(
            type=NodeType.fact,
            tier=Tier.working,
            title="Self check",
            content="Content",
            space="proj",
        )

        with patch("ormah.embeddings.encoder.get_encoder") as mock_enc, \
             patch("ormah.embeddings.vector_store.VectorStore") as mock_vs:

            import numpy as np
            mock_encoder = mock_enc.return_value
            mock_encoder.encode.return_value = np.zeros(128, dtype="float32")

            mock_vs.return_value.search.return_value = [
                {"id": node.id, "similarity": 0.99},  # self
            ]

            result = engine._check_contradiction_prestorage(node, similarity_threshold=0.88)

        assert result == []

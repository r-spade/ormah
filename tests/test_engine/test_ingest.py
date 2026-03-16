"""Tests for conversation ingestion: dry_run, confidence, truncation."""

from __future__ import annotations

import json
from unittest.mock import patch

from ormah.models.node import CreateNodeRequest

_LLM_PATCH = "ormah.background.llm_client.llm_generate"


class TestIngestDryRun:
    def test_dry_run_returns_extracted_without_storing(self, engine):
        """dry_run=True should return extracted memories without calling remember()."""
        fake_llm_response = json.dumps(
            {
                "memories": [
                    {
                        "content": "User prefers dark mode for all editors",
                        "type": "preference",
                        "title": "Dark mode preference",
                        "tags": ["ui"],
                        "about_self": True,
                    },
                    {
                        "content": "Project uses FastAPI for the backend",
                        "type": "fact",
                        "title": "FastAPI backend",
                        "tags": ["architecture"],
                        "about_self": False,
                    },
                ]
            }
        )
        with patch(_LLM_PATCH, return_value=fake_llm_response):
            result = engine.ingest_conversation(
                content="A long enough conversation about preferences and architecture decisions." * 5,
                dry_run=True,
            )

        assert isinstance(result, list)
        assert len(result) == 2
        # Dry run results should NOT have node_id (nothing was stored)
        assert "node_id" not in result[0]
        # But should have content, title, type
        assert result[0]["title"] == "Dark mode preference"
        assert result[0]["type"] == "preference"
        assert result[0]["content"] == "User prefers dark mode for all editors"
        assert "auto-ingested" in result[0]["tags"]

    def test_dry_run_does_not_create_nodes(self, engine):
        """Verify no nodes are created during dry_run."""
        fake_llm_response = json.dumps(
            {
                "memories": [
                    {
                        "content": "Test memory that should not be stored",
                        "type": "fact",
                        "title": "Test memory",
                        "tags": [],
                    },
                ]
            }
        )
        # Count nodes before
        before = engine.db.conn.execute("SELECT COUNT(*) FROM nodes").fetchone()[0]

        with patch(_LLM_PATCH, return_value=fake_llm_response):
            engine.ingest_conversation(
                content="Some conversation content." * 10,
                dry_run=True,
            )

        # Count nodes after — should be the same (only self node)
        after = engine.db.conn.execute("SELECT COUNT(*) FROM nodes").fetchone()[0]
        assert after == before


class TestIngestConfidence:
    def test_auto_ingested_memories_get_default_confidence(self, engine):
        """Auto-ingested memories should default to confidence=0.7."""
        fake_llm_response = json.dumps(
            {
                "memories": [
                    {
                        "content": "The project is called ormah",
                        "type": "fact",
                        "title": "Project name",
                        "tags": ["project"],
                    },
                ]
            }
        )
        with patch(_LLM_PATCH, return_value=fake_llm_response):
            result = engine.ingest_conversation(
                content="A conversation about the project." * 10,
            )

        assert isinstance(result, list)
        assert len(result) == 1
        node_id = result[0]["node_id"]

        # Load the node and check confidence
        node = engine.file_store.load(node_id)
        assert node is not None
        assert node.confidence == 0.7

    def test_llm_specified_confidence_preserved(self, engine):
        """If the LLM specifies confidence, it should be used."""
        fake_llm_response = json.dumps(
            {
                "memories": [
                    {
                        "content": "User might prefer vim keybindings",
                        "type": "preference",
                        "title": "Possible vim preference",
                        "tags": [],
                        "confidence": 0.4,
                    },
                ]
            }
        )
        with patch(_LLM_PATCH, return_value=fake_llm_response):
            result = engine.ingest_conversation(
                content="A conversation about editor preferences." * 10,
            )

        assert len(result) == 1
        node = engine.file_store.load(result[0]["node_id"])
        assert node.confidence == 0.4

    def test_dry_run_includes_confidence(self, engine):
        """dry_run results should include the confidence value."""
        fake_llm_response = json.dumps(
            {
                "memories": [
                    {
                        "content": "Some fact",
                        "type": "fact",
                        "title": "A fact",
                        "tags": [],
                    },
                ]
            }
        )
        with patch(_LLM_PATCH, return_value=fake_llm_response):
            result = engine.ingest_conversation(
                content="A conversation." * 10,
                dry_run=True,
            )

        assert result[0]["confidence"] == 0.7


class TestIngestTruncation:
    def test_content_truncated_to_setting(self, engine):
        """Content passed to LLM should be truncated to ingest_max_content_chars."""
        engine.settings.ingest_max_content_chars = 2000
        marker = "ZQZQ"
        long_content = marker * 2000  # 8000 chars total

        captured_prompt = {}

        def fake_llm(settings, prompt, **kwargs):
            captured_prompt["prompt"] = prompt
            return json.dumps({"memories": []})

        with patch(_LLM_PATCH, side_effect=fake_llm):
            engine.ingest_conversation(content=long_content)

        # The conversation text in the prompt should be truncated to 2000 chars
        # which means 500 full markers (each is 4 chars)
        prompt = captured_prompt["prompt"]
        marker_count = prompt.count(marker)
        assert marker_count == 500

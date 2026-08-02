"""Tests for the shared LLM facade — provider-configured detection."""
from __future__ import annotations

from ormah.background import llm_client
from ormah.background.llm_client import reset_adapter


def test_ingest_provider_configured_reflects_adapter(monkeypatch):
    reset_adapter()
    monkeypatch.setattr(llm_client, "get_adapter", lambda *a, **k: None)
    assert llm_client.ingest_provider_configured(object()) is False

    reset_adapter()
    sentinel = object()
    monkeypatch.setattr(llm_client, "get_adapter", lambda *a, **k: sentinel)

    class _S:
        ingest_llm_provider = "claude_cli"
        llm_provider = "claude_cli"
        ingest_llm_model = "claude-haiku-4-5-20251001"
        llm_model = "claude-haiku-4-5-20251001"

    assert llm_client.ingest_provider_configured(_S()) is True

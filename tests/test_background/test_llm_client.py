"""Tests for the shared LLM facade — adapter routing and input windows."""

from __future__ import annotations

import pytest

from ormah.background import llm_client
from ormah.background.llm.ollama_adapter import OllamaAdapter


class _StubAdapter:
    def generate(self, prompt, **kwargs):
        return "{}"


@pytest.fixture
def captured_num_ctx(monkeypatch):
    """Capture the num_ctx get_adapter is called with, with a clean adapter cache."""
    seen = {}

    def fake_get_adapter(settings, num_ctx=None):
        seen["num_ctx"] = num_ctx
        return _StubAdapter()

    monkeypatch.setattr(llm_client, "get_adapter", fake_get_adapter)
    llm_client.reset_adapter()
    yield seen
    llm_client.reset_adapter()


class TestConsolidationRoute:
    """#192: only the consolidation route pins an input window it can prove."""

    def test_consolidation_route_derives_num_ctx_from_the_budget(self, settings, captured_num_ctx):
        settings.llm_provider = "ollama"
        settings.consolidation_max_prompt_chars = 24000
        settings.llm_num_predict = 4096

        llm_client.llm_generate(settings, "prompt", route="consolidation")

        assert captured_num_ctx["num_ctx"] == 16096  # 24000/2 chars-per-token + 4096 output

    def test_shared_maintenance_adapter_still_omits_num_ctx(self, settings, captured_num_ctx):
        """auto_linker, conflict_detector and duplicate_merger share this adapter and must not
        pay the consolidation KV cache."""
        settings.llm_provider = "ollama"

        llm_client.llm_generate(settings, "prompt")

        assert captured_num_ctx["num_ctx"] is None

    def test_reset_adapter_clears_the_consolidation_cache(self, settings, captured_num_ctx):
        settings.llm_provider = "ollama"
        settings.consolidation_max_prompt_chars = 24000
        settings.llm_num_predict = 4096
        llm_client.llm_generate(settings, "prompt", route="consolidation")

        llm_client.reset_adapter()
        settings.consolidation_max_prompt_chars = 40000
        llm_client.llm_generate(settings, "prompt", route="consolidation")

        assert captured_num_ctx["num_ctx"] == 24096, "the cache was not rebuilt after reset"


class TestOllamaInputWindow:
    """num_ctx=None must OMIT the key, never substitute a default of our own."""

    def test_num_ctx_defaults_to_none(self):
        adapter = OllamaAdapter(model="m")
        assert adapter.num_ctx is None

    def test_num_ctx_reaches_the_ollama_options(self, monkeypatch):
        sent = {}

        class _Resp:
            def raise_for_status(self):
                pass

            def json(self):
                return {"response": "{}"}

        def fake_post(url, json=None, timeout=None):
            sent.update(json)
            return _Resp()

        import httpx

        monkeypatch.setattr(httpx, "post", fake_post)

        OllamaAdapter(model="m", num_ctx=16096).generate("p")
        assert sent["options"]["num_ctx"] == 16096

        sent.clear()
        OllamaAdapter(model="m").generate("p")
        assert "num_ctx" not in sent["options"]

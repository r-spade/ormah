"""Tests for LLM adapter package."""

from __future__ import annotations

import sys
from unittest.mock import patch, MagicMock

from ormah.background.llm import get_adapter
from ormah.background.llm.ollama_adapter import OllamaAdapter
from ormah.background.llm.litellm_adapter import LiteLLMAdapter


class _FakeSettings:
    llm_provider: str = "ollama"
    llm_model: str = "llama3.2"
    llm_base_url: str = "http://localhost:11434"
    llm_num_predict: int = 2048


# --- OllamaAdapter ---

def test_ollama_adapter_success():
    adapter = OllamaAdapter(model="llama3.2")
    mock_resp = MagicMock()
    mock_resp.json.return_value = {"response": '{"answer": 42}'}
    mock_resp.raise_for_status = MagicMock()

    with patch("httpx.post", return_value=mock_resp) as mock_post:
        result = adapter.generate("test prompt", json_mode=True)

    assert result == '{"answer": 42}'
    call_kwargs = mock_post.call_args
    assert call_kwargs[1]["json"]["format"] == "json"
    assert call_kwargs[1]["json"]["options"] == {"num_predict": 4096}


def test_ollama_adapter_custom_num_predict():
    adapter = OllamaAdapter(model="llama3.2", num_predict=1024)
    mock_resp = MagicMock()
    mock_resp.json.return_value = {"response": '{"answer": 42}'}
    mock_resp.raise_for_status = MagicMock()

    with patch("httpx.post", return_value=mock_resp) as mock_post:
        result = adapter.generate("test prompt", json_mode=False)

    assert result == '{"answer": 42}'
    call_kwargs = mock_post.call_args
    payload = call_kwargs[1]["json"]
    assert payload["options"] == {"num_predict": 1024}
    assert "format" not in payload


def test_ollama_adapter_timeout():
    import httpx

    adapter = OllamaAdapter(model="llama3.2")

    with patch("httpx.post", side_effect=httpx.TimeoutException("")):
        result = adapter.generate("test prompt")

    assert result is None


# --- LiteLLMAdapter ---

def test_litellm_adapter_success():
    adapter = LiteLLMAdapter(model="claude-sonnet-4-20250514")
    mock_choice = MagicMock()
    mock_choice.message.content = '{"result": "ok"}'
    mock_response = MagicMock()
    mock_response.choices = [mock_choice]

    mock_litellm = MagicMock()
    mock_litellm.completion.return_value = mock_response
    with patch.dict(sys.modules, {"litellm": mock_litellm}):
        result = adapter.generate("test prompt", json_mode=True)

    assert result == '{"result": "ok"}'
    call_kwargs = mock_litellm.completion.call_args[1]
    assert call_kwargs["response_format"] == {"type": "json_object"}


def test_litellm_adapter_failure():
    adapter = LiteLLMAdapter(model="claude-sonnet-4-20250514")

    mock_litellm = MagicMock()
    mock_litellm.completion.side_effect = Exception("API error")
    with patch.dict(sys.modules, {"litellm": mock_litellm}):
        result = adapter.generate("test prompt")

    assert result is None


# --- get_adapter factory ---

def test_get_adapter_ollama():
    settings = _FakeSettings()
    settings.llm_provider = "ollama"
    adapter = get_adapter(settings)
    assert isinstance(adapter, OllamaAdapter)
    assert adapter.num_predict == 2048


def test_get_adapter_litellm():
    settings = _FakeSettings()
    settings.llm_provider = "litellm"
    adapter = get_adapter(settings)
    assert isinstance(adapter, LiteLLMAdapter)


def test_get_adapter_none():
    settings = _FakeSettings()
    settings.llm_provider = "none"
    adapter = get_adapter(settings)
    assert adapter is None


# --- facade with none provider ---

def test_llm_generate_none_provider():
    from ormah.background.llm_client import llm_generate, reset_adapter

    settings = _FakeSettings()
    settings.llm_provider = "none"
    reset_adapter()
    try:
        result = llm_generate(settings, "test prompt")
        assert result is None
    finally:
        reset_adapter()

import json
import subprocess

from ormah.background.llm import get_adapter
from ormah.background.llm.claude_cli_adapter import ClaudeCliAdapter
from ormah.background.llm.ollama_adapter import OllamaAdapter
from ormah.config import Settings


def test_get_adapter_provider_override_beats_settings():
    s = Settings(llm_provider="ollama")
    assert isinstance(get_adapter(s, provider="claude_cli", model="haiku"), ClaudeCliAdapter)
    assert isinstance(get_adapter(s), OllamaAdapter)  # unchanged when no override


def test_get_adapter_model_override_beats_settings_model():
    s = Settings(llm_provider="ollama", llm_model="settings-model")
    adapter = get_adapter(s, provider="ollama", model="override-model")
    assert adapter.model == "override-model"


def test_settings_accepts_claude_cli_and_ingest_provider():
    s = Settings(llm_provider="ollama", ingest_llm_provider="claude_cli", ingest_llm_model="haiku")
    assert s.ingest_llm_provider == "claude_cli"
    assert s.ingest_llm_model == "haiku"


def test_ingest_provider_falls_back_to_llm_provider_when_empty():
    from ormah.background.llm_client import _resolve_ingest_provider
    assert _resolve_ingest_provider(Settings(llm_provider="ollama")) == "ollama"
    assert _resolve_ingest_provider(
        Settings(llm_provider="ollama", ingest_llm_provider="claude_cli", ingest_llm_model="haiku")
    ) == "claude_cli"


def test_ingest_model_falls_back_to_llm_model_when_empty():
    from ormah.background.llm_client import _resolve_ingest_model
    assert _resolve_ingest_model(Settings(llm_model="fallback-model")) == "fallback-model"
    assert _resolve_ingest_model(
        Settings(llm_model="fallback-model", ingest_llm_provider="claude_cli", ingest_llm_model="haiku")
    ) == "haiku"


def test_extraction_uses_ingest_adapter_not_maintenance(monkeypatch):
    from ormah.background import llm_client
    llm_client.reset_adapter()
    s = Settings(llm_provider="ollama", ingest_llm_provider="claude_cli", ingest_llm_model="haiku")
    captured = {}

    def fake_get_adapter(settings, provider=None, model=None):
        captured["provider"] = provider
        captured["model"] = model
        return None

    monkeypatch.setattr(llm_client, "get_adapter", fake_get_adapter)
    llm_client.ingest_llm_generate(s, "prompt")
    assert captured["provider"] == "claude_cli"
    assert captured["model"] == "haiku"


def test_ingest_path_generate_carries_no_session_persistence(monkeypatch):
    """Proves --no-session-persistence survives the full get_adapter -> generate ingest
    path, so `claude -p` writes no .jsonl transcript under ~/.claude that the session
    watcher could re-ingest (recursion suppressed at the source, not via a guard).
    Overlaps test_claude_cli_adapter.py::test_argv_pins_model_and_json_output, which
    proves the flag on direct construction — this proves it survives the ingest seam."""
    from ormah.background import llm_client

    llm_client.reset_adapter()
    s = Settings(
        llm_provider="ollama",
        ingest_llm_provider="claude_cli",
        ingest_llm_model="claude-haiku-4-5-20251001",
    )

    captured = {}

    def fake_run(argv, **kwargs):
        captured["argv"] = argv
        return subprocess.CompletedProcess(
            argv, 0, stdout=json.dumps({"is_error": False, "result": "{}"}), stderr="",
        )

    monkeypatch.setattr(subprocess, "run", fake_run)
    llm_client.ingest_llm_generate(s, "prompt")
    assert "--no-session-persistence" in captured["argv"]

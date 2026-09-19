import json
import subprocess
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from eval.bench.artifacts import Journal
from eval.bench.cost import Budget, BudgetExceeded, token_cost
from eval.bench.providers import AnthropicProvider, ClaudeCLIProvider, CodexProvider, ProviderError


def test_claude_arguments_and_estimate(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "fake-never-use")

    def fake_run(args, **kwargs):
        assert args[:2] == ["claude", "-p"]
        assert all(
            flag in args
            for flag in ["--safe-mode", "--system-prompt", "--no-session-persistence", "--output-format"]
        )
        # --bare disables subscription OAuth; it must never come back.
        assert "--bare" not in args
        assert args[args.index("--tools") + 1] == ""
        assert "ANTHROPIC_API_KEY" not in kwargs["env"]
        assert kwargs["input"] == "question"
        return SimpleNamespace(
            returncode=0,
            stdout=json.dumps(
                {"result": "answer", "usage": {"input_tokens": 3}, "total_cost_usd": 0.02}
            ),
            stderr="",
        )

    monkeypatch.setattr(subprocess, "run", fake_run)
    result = ClaudeCLIProvider("sonnet").complete("question")
    assert result.usd == 0
    assert result.estimated_usd == 0.02


def test_codex_arguments_stdin_file_and_usage(monkeypatch):
    monkeypatch.setattr(CodexProvider, "executable", "/opt/test/codex")
    def fake_run(args, **kwargs):
        assert args[0] == "/opt/test/codex"
        assert "--ignore-user-config" in args
        assert args[args.index("--sandbox") + 1] == "read-only"
        assert args[args.index("--color") + 1] == "never"
        assert args[-1] == "-"
        assert "question" in kwargs["input"]
        Path(args[args.index("--output-last-message") + 1]).write_text("yes")
        return SimpleNamespace(
            returncode=0, stdout='{"type":"turn.completed","usage":{"input_tokens":8}}', stderr=""
        )

    monkeypatch.setattr(subprocess, "run", fake_run)
    result = CodexProvider("default").complete("question")
    assert result.text == "yes"
    assert result.usage["input_tokens"] == 8


def test_provider_error_recorded(monkeypatch, tmp_path):
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *a, **kw: SimpleNamespace(returncode=1, stdout="", stderr="Not logged in"),
    )
    ledger = Journal(tmp_path / "calls.jsonl")
    with pytest.raises(ProviderError, match="Not logged in"):
        ClaudeCLIProvider("sonnet", ledger=ledger).complete("question")
    assert ledger.rows[0]["error"]
    assert len(ledger.rows) == 1


def test_anthropic_actual_usage_budget_and_cache(tmp_path):
    client = Mock()
    client.messages.create.return_value = SimpleNamespace(
        usage=SimpleNamespace(
            model_dump=lambda: {
                "input_tokens": 10000,
                "output_tokens": 1000,
                "cache_read_input_tokens": 1000,
                "cache_creation_input_tokens": 1000,
            }
        ),
        content=[SimpleNamespace(type="text", text="answer")],
        model="claude-haiku-4-5",
    )
    ledger = Journal(tmp_path / "calls.jsonl")
    budget = Budget(0.01)
    provider = AnthropicProvider("claude-haiku-4-5", client=client, ledger=ledger, budget=budget)
    with pytest.raises(BudgetExceeded):
        provider.complete("q", max_tokens=1)
    assert budget.spent == pytest.approx(0.01635)
    assert ledger.rows[0]["usd"] == pytest.approx(0.01635)
    with pytest.raises(BudgetExceeded):
        provider.complete("q")
    assert client.messages.create.call_count == 1
    assert client.messages.create.call_args.kwargs["temperature"] == 0
    with pytest.raises(ValueError):
        token_cost("unknown", {})


def test_transient_retry_is_bounded_and_recorded(monkeypatch, tmp_path):
    from tests.test_eval_bench.conftest import FakeProvider
    from eval.bench.providers import TextResult

    monkeypatch.setattr("time.sleep", lambda _: None)
    ledger = Journal(tmp_path / "calls.jsonl")
    provider = FakeProvider("fake", ledger=ledger)
    provider._call = Mock(side_effect=[RuntimeError("503 unavailable"), TextResult("ok", "fake")])
    assert provider.complete("q").text == "ok"
    assert len(ledger.rows) == 2
    provider._call = Mock(side_effect=RuntimeError("503 unavailable"))
    with pytest.raises(ProviderError):
        provider.complete("q")
    assert provider._call.call_count == 3


def test_embedding_cache_namespace_isolation(tmp_path):
    from eval.bench.store import EmbeddingCache
    from tests.test_eval_bench.conftest import FakeEncoder

    encoder = FakeEncoder()
    for namespace in ("model-a", "model-a", "model-b"):
        cache = EmbeddingCache(tmp_path / "embeddings.sqlite", namespace)
        cache.encode(["same text"], encoder)
        cache.close()
    assert encoder.calls == 2


def test_retry_latency_measures_each_attempt(monkeypatch, tmp_path):
    from eval.bench.providers import TextResult
    from tests.test_eval_bench.conftest import FakeProvider

    clock = iter([10.0, 12.0, 20.0, 23.0])
    monkeypatch.setattr("time.perf_counter", lambda: next(clock))
    monkeypatch.setattr("time.sleep", lambda _: None)
    ledger = Journal(tmp_path / "calls.jsonl")
    provider = FakeProvider("fake", ledger=ledger)
    provider._call = Mock(side_effect=[RuntimeError("503 unavailable"), TextResult("ok", "fake")])
    assert provider.complete("q").latency_s == 3
    assert [row["latency_s"] for row in ledger.rows] == [2, 3]


def test_optional_api_usage_fields_may_be_null():
    assert token_cost(
        "claude-haiku-4-5",
        {
            "input_tokens": 100,
            "output_tokens": 10,
            "cache_read_input_tokens": None,
            "cache_creation_input_tokens": None,
        },
    ) == pytest.approx(0.00015)


def test_one_hour_cache_writes_use_reported_ttl():
    assert token_cost(
        "claude-haiku-4-5",
        {
            "cache_creation_input_tokens": 1000,
            "cache_creation": {"ephemeral_1h_input_tokens": 1000},
        },
    ) == pytest.approx(0.002)

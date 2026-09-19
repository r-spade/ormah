import json
from unittest.mock import Mock

import pytest

from eval.bench.artifacts import Journal
from eval.bench.runner import phases_for, run
from tests.test_eval_bench.conftest import FakeProvider


def dataset(tmp_path, locomo):
    (tmp_path / "data").mkdir()
    (tmp_path / "data" / "locomo10.json").write_text(json.dumps(locomo))


def test_free_resume_and_separate_llm_phases(args, tmp_path, locomo, bench_engine, monkeypatch):
    dataset(tmp_path, locomo)
    monkeypatch.setattr(bench_engine, "shutdown", lambda: None)
    factory = Mock(return_value=bench_engine)
    provider_factory = Mock(
        side_effect=lambda name, model, **kw: FakeProvider(model or "fake", **kw)
    )
    report = run(args, base=tmp_path, engine_factory=factory, provider_factory=provider_factory)
    assert report["overall"]["questions"] == 3
    assert report["overall"]["recall@30"] == 1
    provider_factory.assert_not_called()
    seeded = bench_engine.db.conn.execute("SELECT COUNT(*) FROM nodes").fetchone()[0]
    assert seeded == 2
    latest = Journal(tmp_path / "artifacts" / "test" / "questions.jsonl").latest()
    args.resume = True
    report = run(args, base=tmp_path, engine_factory=factory, provider_factory=provider_factory)
    assert latest == Journal(tmp_path / "artifacts" / "test" / "questions.jsonl").latest()
    args.phase = "answer,judge,report"
    report = run(args, base=tmp_path, engine_factory=factory, provider_factory=provider_factory)
    assert report["overall"]["accuracy"] == 1
    assert report["overall"]["scored"] == 2
    assert report["overall"]["abstention_accuracy"] == 1
    before = len(Journal(tmp_path / "artifacts" / "test" / "calls.jsonl").rows)
    run(args, base=tmp_path, engine_factory=factory, provider_factory=provider_factory)
    assert len(Journal(tmp_path / "artifacts" / "test" / "calls.jsonl").rows) == before
    args.k = 10
    with pytest.raises(ValueError, match="differ"):
        run(args, base=tmp_path, engine_factory=factory, provider_factory=provider_factory)


def test_extract_requires_explicit_phase(args):
    args.mode = "extract"
    with pytest.raises(ValueError, match="explicitly"):
        phases_for(args)
    args.phase = "all"
    assert "store" in phases_for(args)


def test_api_without_key_estimate_no_engine(args, monkeypatch, capsys, tmp_path):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    args.phase = "answer"
    args.answer_provider = "anthropic"
    factory = Mock()
    with pytest.raises(ValueError, match="ANTHROPIC_API_KEY"):
        run(args, base=tmp_path, engine_factory=factory)
    factory.assert_not_called()
    assert "estimate $" in capsys.readouterr().out


def test_journal_recovers_only_torn_tail(tmp_path):
    path = tmp_path / "test.jsonl"
    path.write_bytes(b'{"question_id":"a"}\n{"question_id":')
    journal = Journal(path)
    journal.append({"question_id": "b"})
    assert list(Journal(path).latest()) == ["a", "b"]
    path.write_text("broken\n")
    with pytest.raises(json.JSONDecodeError):
        Journal(path)


def test_cli_parse(monkeypatch):
    import ormah.cli

    handler = Mock()
    monkeypatch.setattr(ormah.cli, "_cmd_eval_bench", handler)
    monkeypatch.setattr(
        "sys.argv",
        [
            "ormah",
            "eval",
            "bench",
            "run",
            "locomo",
            "--mode",
            "extract",
            "--phase",
            "all",
            "--limit",
            "10",
            "--conversation",
            "0",
        ],
    )
    ormah.cli.main()
    parsed = handler.call_args.args[0]
    assert parsed.judge_provider == "codex"
    assert parsed.answer_provider == "claude-cli"
    assert parsed.limit == 10 and parsed.conversation == 0 and parsed.workers == 4


def test_provider_errors_are_checkpointed(args, tmp_path, locomo, bench_engine, monkeypatch):
    dataset(tmp_path, locomo)
    monkeypatch.setattr(bench_engine, "shutdown", lambda: None)
    args.phase = "all"

    def factory(name, model, **kwargs):
        provider = FakeProvider("fake", **kwargs)
        if kwargs["phase"] == "answer":
            provider._call = Mock(side_effect=RuntimeError("invalid provider output"))
        return provider

    report = run(
        args, base=tmp_path, engine_factory=lambda _: bench_engine, provider_factory=factory
    )
    assert report["overall"]["errors"]["answer"] == 3
    assert report["overall"]["errors"]["judge"] == 3
    assert report["overall"]["accuracy"] is None
    assert report["calls"]["answer"]["errors"] == 3


def test_question_filter_limit(args, tmp_path, locomo):
    from eval.bench.runner import selected_questions

    dataset(tmp_path, locomo)
    args.category = 3
    args.limit = 1
    questions = list(selected_questions(args, tmp_path / "data" / "locomo10.json"))
    assert len(questions) == 1
    assert questions[0].gold == "Yes"


def test_published_runtime_guard(monkeypatch, capsys):
    import builtins
    import ormah.cli

    original = builtins.__import__

    def missing_eval(name, *args, **kwargs):
        if name == "eval.bench.cli":
            raise ModuleNotFoundError("No module named 'eval'", name="eval")
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", missing_eval)
    with pytest.raises(SystemExit) as exc:
        ormah.cli._cmd_eval_bench(None)
    assert exc.value.code == 1
    assert "not installed in the published Ormah runtime" in capsys.readouterr().out


def test_cli_does_not_hide_missing_transitive_dependency(monkeypatch):
    import builtins
    import ormah.cli

    original = builtins.__import__

    def missing_dependency(name, *args, **kwargs):
        if name == "eval.bench.cli":
            raise ModuleNotFoundError("No module named 'numpy'", name="numpy")
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", missing_dependency)
    with pytest.raises(ModuleNotFoundError, match="numpy"):
        ormah.cli._cmd_eval_bench(None)


def test_api_budget_aborts_run_and_preserves_charge(
    args, tmp_path, locomo, bench_engine, monkeypatch
):
    from types import SimpleNamespace
    from eval.bench.cost import BudgetExceeded
    from eval.bench.providers import AnthropicProvider

    dataset(tmp_path, locomo)
    monkeypatch.setattr(bench_engine, "shutdown", lambda: None)
    args.answer_provider = "anthropic"
    run(args, base=tmp_path, engine_factory=lambda _: bench_engine)
    args.resume = True
    args.phase = "answer,judge,report"
    args.max_usd = 0.01
    monkeypatch.setenv("ANTHROPIC_API_KEY", "fake-test-key-never-sent")
    client = Mock()
    client.messages.create.return_value = SimpleNamespace(
        usage=SimpleNamespace(model_dump=lambda: {"input_tokens": 10000, "output_tokens": 1000}),
        content=[SimpleNamespace(type="text", text="blue bike")],
        model="claude-haiku-4-5",
    )

    def factory(name, model, **kwargs):
        if name == "anthropic":
            return AnthropicProvider("claude-haiku-4-5", client=client, **kwargs)
        return FakeProvider("fake", **kwargs)

    with pytest.raises(BudgetExceeded):
        run(args, base=tmp_path, engine_factory=lambda _: bench_engine, provider_factory=factory)
    assert client.messages.create.call_count == 1
    calls = Journal(tmp_path / "artifacts" / "test" / "calls.jsonl").rows
    assert sum(c["usd"] for c in calls) == pytest.approx(0.015)
    summary = json.loads((tmp_path / "artifacts" / "test" / "summary.json").read_text())
    assert summary["calls"]["judge"]["calls"] == 0


@pytest.mark.parametrize("amount", [float("nan"), float("inf"), 0, -1])
def test_budget_must_be_finite_and_positive(args, amount):
    from eval.bench.runner import validate

    args.max_usd = amount
    with pytest.raises(ValueError, match="finite and positive"):
        validate(args)

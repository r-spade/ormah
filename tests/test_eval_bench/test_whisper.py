import json
import logging
from datetime import datetime, timedelta, timezone
from unittest.mock import Mock

import numpy as np
import pytest

from eval.bench.answer import answer_prompt, memory_context
from eval.bench.artifacts import Journal
from eval.bench.metrics import aggregate
from eval.bench.runner import run, settings_for
from eval.bench.store import EmbeddingCache, seed_memories
from eval.bench.whisper import (
    RerankerUnavailable,
    reference_date,
    retrieve_whisper,
    whispered_memories,
)
from eval.settings import WHISPER_EVAL_SETTINGS_OVERRIDES
from tests.test_eval_bench.conftest import FakeProvider
from tests.test_eval_bench.test_runner import dataset


def enable_whisper(engine):
    for key, value in WHISPER_EVAL_SETTINGS_OVERRIDES.items():
        setattr(engine.settings, key, value)
    engine._whisper_reranker_available = True


def seed(engine, tmp_path, n=3):
    memories = [dict(
        id=f"memory{i:02d}-full-id", title=f"Ada bike {i}",
        content="Ada rides a bike. " * 60, type="fact", created="2023-01-10T00:00:00+00:00",
        tags=["session:s1", f"turn:D1:{i}"],
    ) for i in range(n)]
    cache = EmbeddingCache(tmp_path / "embeddings.sqlite", "test")
    try:
        seed_memories(engine, memories, cache)
    finally:
        cache.close()
    return memories


def test_settings_shared_and_recall_default():
    assert settings_for("recall")["whisper_reranker_enabled"] is False
    assert all(settings_for("whisper")[k] == v for k, v in WHISPER_EVAL_SETTINGS_OVERRIDES.items())
    from eval.whisper.cli import _EVAL_SETTINGS_OVERRIDES

    assert _EVAL_SETTINGS_OVERRIDES is WHISPER_EVAL_SETTINGS_OVERRIDES


def test_real_whisper_projection_with_fake_reranker(bench_engine, tmp_path, monkeypatch):
    from ormah.engine.prompt_classifier import PromptIntent

    enable_whisper(bench_engine)
    memories = seed(bench_engine, tmp_path)
    bench_engine.context_builder._classifier = Mock()
    bench_engine.context_builder._classifier.classify.return_value = PromptIntent(["general"])
    monkeypatch.setattr("ormah.embeddings.reranker.rerank", lambda **kw: [
        {**r, "score": 0.9, "ce_absolute": 0.9} for r in kw["candidates"]
    ])
    monkeypatch.setattr(bench_engine, "_maybe_get_onboarding_nudge", lambda **kw: "NUDGE SENTINEL")
    result = retrieve_whisper(bench_engine, "What bike does Ada ride?", "2023-01-20T00:00:00+00:00")
    assert result["injected_count"] == 3
    assert [r["node"]["id"] for r in result["ranked"]] == result["injected_ids"]
    assert all(0 < len(r["node"]["content"]) <= 600 for r in result["ranked"][:2])
    assert result["ranked"][2]["node"]["content"] == ""
    assert "NUDGE SENTINEL" in result["raw_whisper_text"]
    assert "NUDGE SENTINEL" not in result["whisper_text"]
    assert "# Ormah whispers" not in result["whisper_text"]
    assert result["answer_context_chars"] == len(memory_context(result["ranked"]))
    assert result["whisper_context_chars"] == len(result["whisper_text"])
    assert memories[0]["content"] not in memory_context(result["ranked"])
    assert result["reranker_active"] is True


@pytest.mark.parametrize("raw", ["", "Onboarding nudge", "maintenance_due: do maintenance"])
def test_silence_passes_empty_memories_to_answerer(
    raw, args, tmp_path, locomo, bench_engine, monkeypatch
):
    dataset(tmp_path, locomo)
    enable_whisper(bench_engine)
    args.retrieval, args.phase = "whisper", "all"
    monkeypatch.setattr(bench_engine, "shutdown", lambda: None)
    whisper = Mock(return_value=(raw, []))
    monkeypatch.setattr(bench_engine, "get_whisper_context", whisper)
    recall = Mock(side_effect=AssertionError("No recall fallback"))
    monkeypatch.setattr(bench_engine, "recall_search_structured", recall)
    factory = Mock(return_value=bench_engine)
    answerer = Mock(return_value={"text": "Not enough information"})
    monkeypatch.setattr("eval.bench.answer.answer_question", answerer)
    report = run(args, base=tmp_path, engine_factory=factory,
                 provider_factory=lambda name, model, **kw: FakeProvider("fake", **kw))
    assert factory.call_args.kwargs == {"retrieval": "whisper"}
    assert all(call.args[2] == [] for call in answerer.call_args_list)
    assert all(call.kwargs == dict(space=None, recent_prompts=None, session_id=None,
                                   _return_debug=True) for call in whisper.call_args_list)
    recall.assert_not_called()
    assert report["overall"]["injection_rate"] == 0
    assert report["overall"]["mean_whisper_context_chars"] == 0
    assert report["overall"]["recall@whisper"] == 0
    assert report["manifest"]["reranker_active"] is True
    args.resume, args.retrieval = True, "recall"
    with pytest.raises(ValueError, match="differ"):
        run(args, base=tmp_path, engine_factory=factory)


def test_unavailable_reranker_fails_before_seeding(args, tmp_path, locomo, bench_engine, monkeypatch):
    dataset(tmp_path, locomo)
    args.retrieval = "whisper"
    monkeypatch.setattr(bench_engine, "shutdown", lambda: None)
    seeder = Mock()
    monkeypatch.setattr("eval.bench.runner.seed_memories", seeder)
    with pytest.raises(RerankerUnavailable, match="requires a loaded reranker"):
        run(args, base=tmp_path, engine_factory=lambda *a, **kw: bench_engine)
    seeder.assert_not_called()
    manifest = json.loads((tmp_path / "artifacts/test/manifest.json").read_text())
    assert manifest["reranker_active"] is False


def test_runtime_reranker_failure_aborts_and_checkpoints(
    args, tmp_path, locomo, bench_engine, monkeypatch
):
    dataset(tmp_path, locomo)
    args.retrieval = "whisper"
    enable_whisper(bench_engine)
    monkeypatch.setattr(bench_engine, "shutdown", lambda: None)

    def fail(*a, **kw):
        logging.getLogger("ormah.engine.context_builder").warning(
            "Whisper reranker failed, using embedding scores: broken inference"
        )
        return "", []

    monkeypatch.setattr(bench_engine, "get_whisper_context", fail)
    with pytest.raises(RerankerUnavailable, match="broken inference"):
        run(args, base=tmp_path, engine_factory=lambda *a, **kw: bench_engine)
    rows = Journal(tmp_path / "artifacts/test/questions.jsonl").latest()
    assert len(rows) == 1
    assert next(iter(rows.values()))["retrieve"]["reranker_active"] is False
    assert json.loads((tmp_path / "artifacts/test/manifest.json").read_text())["reranker_active"] is False


def test_missing_header_is_error(bench_engine, tmp_path):
    memories = seed(bench_engine, tmp_path, n=1)
    with pytest.raises(ValueError, match="header"):
        whispered_memories(bench_engine, "arbitrary non-memory text", [memories[0]["id"]])


def test_whisper_metrics_whole_injected_set_and_abstention():
    def row(dataset="longmemeval", silent=False, abstention=False):
        ranked = [] if silent else [
            {"node": {"tags": ["session:s1", "turn:D1:1"]}},
            {"node": {"tags": ["session:s2", "turn:D1:2"]}},
        ]
        return dict(dataset=dataset, gold_ids=["s1", "s2"] if dataset == "longmemeval"
                    else ["D1:1", "D1:2"], abstention=abstention,
                    retrieve={"status": "ok", "result": dict(
                        ranked=ranked, silent=silent, whisper_context_chars=0 if silent else 100,
                        answer_context_chars=0 if silent else 150, latency_s=0.1)},
                    judge={"status": "ok", "result": {"correct": not silent}})

    # k must not truncate the whisper set. Silences stay in mean denominators;
    # LME abstentions are excluded from retrieval recall, but not silence metrics.
    result = aggregate([row(), row(silent=True), row(silent=True, abstention=True)], 1, "raw", "whisper")
    assert result["recall@whisper"] == 0.5
    assert result["injection_rate"] == pytest.approx(1 / 3)
    assert result["mean_injected_memories"] == pytest.approx(2 / 3)
    assert result["mean_whisper_context_chars"] == pytest.approx(100 / 3)
    assert result["abstention_silence_rate"] == 1
    assert result["abstention_retrieved"] == 1
    assert result["accuracy"] == pytest.approx(1 / 3)
    assert "production_recall@1" not in result
    assert aggregate([row("locomo")], 1, "raw", "whisper")["recall@whisper"] == 1
    assert aggregate([row("locomo")], 1, "extract", "whisper")["recall@whisper"] is None
    assert aggregate([], 1, "raw", "whisper")["injection_rate"] is None


def test_reference_date_covers_temporal_and_continuation_and_resets(encoder):
    from ormah.engine.prompt_classifier import PromptClassifier, extract_time_params

    reference = datetime(2023, 1, 20, tzinfo=timezone.utc)
    classifier = PromptClassifier(encoder)
    vec = encoder.encode("what happened last week")
    classifier._archetype_vecs = {"temporal": np.array([vec])}
    with pytest.raises(RuntimeError), reference_date(reference.isoformat()):
        direct = extract_time_params("last week")
        assert datetime.fromisoformat(direct["created_after"]) < reference
        assert datetime.fromisoformat(direct["created_before"]) <= reference
        assert classifier.classify("what happened last week").search_params["created_after"] == direct["created_after"]
        classifier._archetype_vecs = {"continuation": np.array([vec])}
        assert classifier.classify("what happened last week").search_params["created_after"] == (
            reference - timedelta(days=3)
        ).isoformat()
        raise RuntimeError("reset even on failure")
    assert datetime.fromisoformat(extract_time_params("today")["created_before"]).year == datetime.now(timezone.utc).year


def test_answer_prompt_silent_has_no_memory_text():
    prompt = answer_prompt({"dataset": "locomo", "question": "Where?", "question_date": "2023"}, [])
    assert "Memories:\n\n\nQuestion:" in prompt


def test_auto_temporal_recall_uses_scoped_date(bench_engine, tmp_path, monkeypatch):
    seed(bench_engine, tmp_path, n=1)
    search = Mock()
    search.search.return_value = []
    monkeypatch.setattr(bench_engine, "_get_hybrid_search", lambda: search)
    with reference_date("2023-01-20T00:00:00+00:00"):
        hits = bench_engine.recall_search_structured("what happened last week")
    assert search.search.call_args.kwargs["created_after"] == "2023-01-06T00:00:00+00:00"
    assert search.search.call_args.kwargs["created_before"] == "2023-01-13T00:00:00+00:00"
    assert hits[0]["node"]["id"] == "memory00-full-id"
    assert bench_engine.recall_search_structured("what happened last week") == []


def test_reference_date_is_context_local():
    from contextvars import Context
    from ormah.engine.prompt_classifier import _temporal_reference_date

    with reference_date("2023-01-20T00:00:00+00:00"):
        assert _temporal_reference_date.get().year == 2023
        assert Context().run(_temporal_reference_date.get) is None
    assert _temporal_reference_date.get() is None


def test_legacy_recall_manifest_can_resume(args, tmp_path, locomo, bench_engine, monkeypatch):
    dataset(tmp_path, locomo)
    monkeypatch.setattr(bench_engine, "shutdown", lambda: None)
    factory = lambda *a, **kw: bench_engine  # noqa: E731
    run(args, base=tmp_path, engine_factory=factory)
    path = tmp_path / "artifacts/test/manifest.json"
    manifest = json.loads(path.read_text())
    del manifest["parameters"]["retrieval"]
    path.write_text(json.dumps(manifest))
    args.resume = True
    assert run(args, base=tmp_path, engine_factory=factory)["overall"]["recall@30"] == 1

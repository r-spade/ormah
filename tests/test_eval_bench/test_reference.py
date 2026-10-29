from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import patch
import subprocess

import numpy as np
import pytest

from eval.bench.answer import answer_prompt, memory_context
from eval.bench.diagnostics import diagnose_question
from eval.bench.reference import budget_context, dense_question, empty_question
from eval.bench.store import EmbeddingCache, seed_memories


def memories():
    return [{"id": str(i), "title": "Same lexical title", "content": text,
             "created": "2023-01-01T00:00:00+00:00", "tags": [f"turn:{i}"]}
            for i, text in enumerate(["alpha blue", "beta red", "alpha blue"])]


def test_dense_uses_cosine_not_lexical_boosts(bench_engine, tmp_path):
    cache = EmbeddingCache(tmp_path / "vectors.sqlite", "fake")
    seed_memories(bench_engine, memories(), cache)
    # Deliberately unnormalized stored vectors; title matches cannot affect rank.
    from ormah.embeddings.vector_store import VectorStore
    vectors = VectorStore(bench_engine.db)
    q = np.zeros(768, dtype=np.float32)
    q[0] = 1
    vectors.upsert("0", 100 * np.roll(q, 1))
    vectors.upsert("1", 2 * q)
    vectors.upsert("2", 7 * q)
    with patch(
        "ormah.embeddings.encoder.get_encoder", return_value=SimpleNamespace(encode_query=lambda _: q)
    ):
        result = dense_question(bench_engine, "Same lexical title", 2)
    assert [x["node"]["id"] for x in result["ranked"]] == ["1", "2"]
    assert [x["score"] for x in result["ranked"]] == [1.0, 1.0]
    assert result["production_gate"] is None
    assert result["ranked"][0]["node"]["content"] == "beta red"
    cache.close()


@pytest.mark.parametrize("cap", [0, 1, 30, 84, 85, 99, 12000])
def test_exact_unicode_budget_includes_dates_and_separators(cap):
    ranked = [{"node": {**m, "content": "é 🐈 " * 30}} for m in memories()]
    original = {"ranked": ranked}
    saved = deepcopy(original)
    result = budget_context(original, cap)
    assert result["answer_context_text"] == memory_context(ranked)[:cap]
    assert result["answer_context_chars"] <= cap
    assert original == saved
    assert sum(s["exposed_content_chars"] for s in result["exposure_spans"]) <= cap
    assert result["pre_budget_ranked"] == ranked


def test_no_memory_and_prompt_keep_question_date_without_gold():
    q = {"dataset": "locomo", "question_date": "2023-10-07", "question": "Where?",
         "gold": "SECRET GOLD", "source_evidence": [{"text": "SECRET EVIDENCE"}]}
    none = budget_context(empty_question(), 12000)
    assert none["answer_context_text"] == ""
    assert none["answer_context_chars"] == 0
    prompt = answer_prompt(q, [], context=none["answer_context_text"])
    assert "Reference date: 2023-10-07\nMemories:\n\n\nQuestion: Where?" in prompt
    assert "SECRET" not in prompt
    # The explicit payload is used verbatim, including a clipped header.
    assert "Memories:\n[2023\n\nQuestion:" in answer_prompt(q, [], context="[2023")


def test_comparison_counts_exposed_reference_memories():
    from eval.bench.compare import _metric
    result = budget_context({"ranked": [{"node": m} for m in memories()]}, 35)
    row = {"retrieve": {"status": "ok", "result": result}}
    assert _metric(row, "injected_count") == 1
    assert _metric(row, "silent") is False
    row["retrieve"]["result"] = budget_context(empty_question(), 12000)
    assert _metric(row, "injected_count") == 0
    assert _metric(row, "silent") is True


def test_budget_diagnostics_distinguish_retrieval_from_exposure():
    stored = memories()
    ranked = [{"node": m, "score": 1} for m in stored]
    result = budget_context({"ranked": ranked}, len(memory_context([ranked[0]])) - 3)
    row = {"question_id": "q", "dataset": "locomo", "question_type": "1",
           "abstention": False, "gold_ids": ["0"],
           "source_evidence": [{"turn_id": "0", "session_id": "s", "text": "alpha blue"}],
           "store": {"status": "ok"}, "retrieve": {"status": "ok", "result": result},
           "judge": {"status": "ok", "result": {"correct": False}}}
    diagnostic = diagnose_question(row, stored, "raw")
    assert diagnostic["retrieve"]["supporting_turn_coverage"]["recall"] == 1
    assert diagnostic["exposure"]["supporting_full_content_coverage"]["recall"] == 0
    assert diagnostic["failure_stage"] == "exposure_loss_or_truncation"
    assert diagnostic["exposure"]["context"] == result["answer_context_text"]
    assert diagnostic["exposure"]["supporting_memories"][0]["exposed_content"] == "alpha b"


def test_all_api_keys_removed_from_subscription_children(monkeypatch):
    from eval.bench.providers import ClaudeCLIProvider
    for key in ("OPENAI_API_KEY", "CODEX_API_KEY", "ANTHROPIC_API_KEY"):
        monkeypatch.setenv(key, "SECRET")
    def fake(args, **kwargs):
        assert not any(k.endswith("API_KEY") for k in kwargs["env"] if k in
                       {"OPENAI_API_KEY", "CODEX_API_KEY", "ANTHROPIC_API_KEY"})
        return SimpleNamespace(returncode=0, stdout="{}", stderr="")
    monkeypatch.setattr(subprocess, "run", fake)
    ClaudeCLIProvider("sonnet").run([], "prompt", "/tmp")


@pytest.mark.parametrize("strategy", ["dense", "none", "recall"])
def test_reference_runner_budget_reaches_answerer(args, locomo, tmp_path, monkeypatch, strategy):
    import json
    from eval.bench.runner import run
    from tests.test_eval_bench.conftest import FakeProvider
    (tmp_path / "data").mkdir()
    (tmp_path / "data" / "locomo10.json").write_text(json.dumps(locomo))
    args.retrieval = strategy
    args.context_chars = 35
    args.phase = "all"
    args.answer_provider = args.judge_provider = "codex"
    seen = []
    class Provider(FakeProvider):
        def _call(self, prompt, max_tokens):
            if self.phase == "answer":
                seen.append(prompt.split("Memories:\n", 1)[1].split("\n\nQuestion:", 1)[0])
            return super()._call(prompt, max_tokens)
    # A synthetic encoder avoids network/model downloads in integration tests.
    from tests.test_eval_bench.conftest import FakeEncoder
    encoder = FakeEncoder()
    monkeypatch.setattr("ormah.embeddings.encoder.get_encoder", lambda *_: encoder)
    monkeypatch.setattr("eval.bench.store.get_encoder", lambda *_: encoder)
    result = run(args, base=tmp_path, provider_factory=lambda name, model, **kw: Provider("fake", **kw))
    assert seen and all(len(text) <= 35 for text in seen)
    if strategy == "none":
        assert all(text == "" for text in seen)
    assert result["overall"]["errors"] == {p: 0 for p in ("store", "retrieve", "answer", "judge")}

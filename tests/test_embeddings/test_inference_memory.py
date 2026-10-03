"""Regression coverage for #322: resource bounds must reach real inference."""

from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from threading import Barrier, Lock, get_ident
import time
from unittest.mock import MagicMock

import numpy as np
import pytest

from ormah.embeddings import local_adapter, reranker
from ormah.embeddings.runtime import inference_request
from ormah.engine.context_builder import ContextBuilder
from ormah.engine.prompt_classifier import ARCHETYPES, PromptClassifier


def candidate(i=0):
    return {"node": {"id": str(i), "title": "Memory", "content": "content"}, "score": 0.5}


@pytest.fixture(autouse=True)
def isolated_model_caches(monkeypatch):
    monkeypatch.setattr(local_adapter, "_model_cache", {})
    monkeypatch.setattr(reranker, "_model_cache", {})


def test_arena_option_reaches_onnx_sessions(monkeypatch, tmp_path):
    """Exercise real FastEmbed argument forwarding, without downloading models.

    Checking only constructor kwargs misses the contributor patch's bug:
    session_options=... is accepted by FastEmbed but silently discarded.
    """
    from fastembed.common.model_management import ModelManagement

    monkeypatch.setattr(ModelManagement, "download_model", lambda *a, **kw: tmp_path)
    for module in [
        "fastembed.text.onnx_text_model",
        "fastembed.rerank.cross_encoder.onnx_text_model",
    ]:
        monkeypatch.setattr(f"{module}.load_tokenizer", lambda **kw: (MagicMock(), {}))
    sessions = []

    def session(*args, **kwargs):
        sessions.append(kwargs["sess_options"])
        return MagicMock()

    monkeypatch.setattr("onnxruntime.InferenceSession", session)
    local_adapter.LocalAdapter().model
    reranker.preload_model("Xenova/ms-marco-MiniLM-L-6-v2")
    assert len(sessions) == 2
    assert all(not options.enable_cpu_mem_arena for options in sessions)


@pytest.mark.parametrize("kind", ["encoder", "reranker"])
def test_concurrent_first_use_loads_one_model(monkeypatch, kind):
    calls = []
    barrier = Barrier(6)

    def construct(*args, **kwargs):
        calls.append(1)
        time.sleep(0.03)  # Release the GIL while the other callers enter.
        return object()

    if kind == "encoder":
        monkeypatch.setattr("fastembed.TextEmbedding", construct)

        def load():
            return local_adapter.LocalAdapter("shared-model").model
    else:
        monkeypatch.setattr("fastembed.rerank.cross_encoder.TextCrossEncoder", construct)

        def load():
            return reranker.preload_model("shared-model")

    def run(i):
        barrier.wait(timeout=5)
        with inference_request("recall" if i % 2 else "general"):
            return load()

    with ThreadPoolExecutor(6) as pool:
        models = list(pool.map(run, range(6)))
    assert len(calls) == 1
    assert all(model is models[0] for model in models)


def test_nested_model_load_does_not_deadlock(monkeypatch):
    inference_threads = []

    class Model:
        def embed(self, texts, **kwargs):
            inference_threads.append(get_ident())
            yield np.ones(3)

    def construct(*args, **kwargs):
        inference_threads.append(get_ident())
        return Model()

    monkeypatch.setattr("fastembed.TextEmbedding", construct)
    adapter = local_adapter.LocalAdapter("nested-model")
    with ThreadPoolExecutor(1) as pool:
        result = pool.submit(adapter.encode, "prompt").result(timeout=5)
    assert result.shape == (3,)
    assert len(set(inference_threads)) == 1


def test_constructor_failure_can_retry(monkeypatch):
    model = object()
    attempts = 0

    def construct(*args, **kwargs):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise RuntimeError("load failed")
        return model

    monkeypatch.setattr("fastembed.TextEmbedding", construct)
    adapter = local_adapter.LocalAdapter("retry-model")
    with pytest.raises(RuntimeError, match="load failed"):
        adapter.model
    assert adapter.model is model
    assert attempts == 2


@pytest.mark.parametrize("origin", ["general", "recall"])
def test_all_local_inference_paths_share_a_lane_limit(monkeypatch, origin):
    """Protect lazy iteration, including encode/batch and different adapters."""
    barrier = Barrier(8)
    counter_lock = Lock()
    active = peak = 0
    inference_threads = set()

    def execute(values):
        nonlocal active, peak
        with counter_lock:
            inference_threads.add(get_ident())
            active += 1
            peak = max(peak, active)
        try:
            time.sleep(0.02)
            yield from values
        finally:
            with counter_lock:
                active -= 1

    class Model:
        def embed(self, texts, **kwargs):
            yield from execute([np.ones(3) for _ in texts])

        query_embed = embed

        def rerank(self, query, docs, **kwargs):
            yield from execute([0.0 for _ in docs])

    model = Model()
    monkeypatch.setattr(reranker, "_get_model", lambda _: model)
    adapters = [local_adapter.LocalAdapter() for _ in range(2)]
    for adapter in adapters:
        adapter._model = model
    calls = [
        lambda: adapters[0].encode("prompt"),
        lambda: adapters[1].encode_query("prompt"),
        lambda: adapters[0].encode_batch(["prompt"] * 2),
        lambda: reranker.rerank("prompt", [candidate()], "test", 0),
    ] * 2

    def run(call):
        barrier.wait(timeout=5)
        with inference_request(origin):
            return call()

    with ThreadPoolExecutor(8) as pool:
        list(pool.map(run, calls))
    assert peak == 1
    assert len(inference_threads) == 1
    assert active == 0


def test_reranker_bounds_batches_without_losing_query_or_candidates(monkeypatch):
    model = MagicMock()
    model.rerank.side_effect = lambda query, docs, **kw: iter(-12 + i / 2 for i in range(len(docs)))
    monkeypatch.setattr(reranker, "_get_model", lambda _: model)
    query = "background " * 500 + "current request about authentication"
    results = reranker.rerank(query, [candidate(i) for i in range(25)], "test", 0)
    args, kwargs = model.rerank.call_args
    assert args[0] == query
    assert len(args[1]) == 25
    assert kwargs["batch_size"] <= 8
    assert {r["node"]["id"] for r in results} == {str(i) for i in range(25)}
    assert results[0]["node"]["id"] == "24"


@pytest.mark.parametrize("requested,expected", [(32, 8), (100, 8), (2, 2)])
def test_embedding_batch_bound_preserves_text_and_order(requested, expected):
    model = MagicMock()
    model.embed.side_effect = lambda texts, **kw: (np.array([i]) for i, _ in enumerate(texts))
    adapter = local_adapter.LocalAdapter()
    adapter._model = model
    texts = [str(i) + " context" * 1000 for i in range(25)]
    result = adapter.encode_batch(texts, batch_size=requested)
    assert model.embed.call_count == 1
    args, kwargs = model.embed.call_args
    assert args[0] == texts
    assert kwargs["batch_size"] == expected
    np.testing.assert_array_equal(result[:, 0], np.arange(25))


@pytest.mark.parametrize("method", ["encode", "encode_query", "encode_batch", "rerank"])
def test_inference_failure_releases_limit_for_other_threads(monkeypatch, method):
    def fail(*args, **kwargs):
        yield from ()
        raise RuntimeError("inference failed")

    model = MagicMock()
    model.embed.side_effect = fail
    model.query_embed.side_effect = fail
    model.rerank.side_effect = fail
    adapter = local_adapter.LocalAdapter()
    adapter._model = model
    monkeypatch.setattr(reranker, "_get_model", lambda _: model)
    with pytest.raises(RuntimeError, match="inference failed"):
        if method == "rerank":
            reranker.rerank("query", [candidate()], "test", 0)
        else:
            getattr(adapter, method)(["query"] if method == "encode_batch" else "query")
    model.embed.side_effect = lambda *a, **kw: iter([np.ones(3)])
    with ThreadPoolExecutor(1) as pool:
        assert pool.submit(adapter.encode, "retry").result(timeout=5).shape == (3,)


def test_concurrent_classifier_cold_start_publishes_complete_archetypes():
    class Encoder:
        def __init__(self):
            self.batch_calls = 0

        def encode_batch(self, prompts):
            self.batch_calls += 1
            time.sleep(0.01)
            return np.ones((len(prompts), 3))

        def encode(self, prompt):
            return np.ones(3)

    encoder = Encoder()
    classifier = PromptClassifier(encoder)
    barrier = Barrier(4)

    def classify(_):
        barrier.wait(timeout=5)
        return classifier.classify("investigate the memory system").categories

    with ThreadPoolExecutor(4) as pool:
        categories = list(pool.map(classify, range(4)))
    assert encoder.batch_calls == len(ARCHETYPES)
    assert set(classifier._archetype_vecs or {}) == set(ARCHETYPES)
    assert all(result == categories[0] for result in categories)


def test_context_builder_concurrent_first_use_creates_one_classifier(monkeypatch):
    constructed = []
    classifier = object()

    def construct(*args, **kwargs):
        constructed.append(1)
        time.sleep(0.02)
        return classifier

    monkeypatch.setattr("ormah.engine.prompt_classifier.PromptClassifier", construct)
    engine = SimpleNamespace(
        _get_hybrid_search=lambda: SimpleNamespace(encoder=object()),
    )
    builder = ContextBuilder(graph=MagicMock(), engine=engine)
    barrier = Barrier(6)

    def get(_):
        barrier.wait(timeout=5)
        return builder._get_classifier()

    with ThreadPoolExecutor(6) as pool:
        results = list(pool.map(get, range(6)))
    assert len(constructed) == 1
    assert all(result is classifier for result in results)

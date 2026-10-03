"""Request routing and bounded concurrency, independent of model speed."""

from concurrent.futures import ThreadPoolExecutor
from contextvars import ContextVar
from threading import Barrier, Event, Lock, current_thread, get_ident
from types import SimpleNamespace

import anyio
import numpy as np
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
import httpx

from ormah.api.routes_agent import router as agent_router
from ormah.api.routes_ui import router as ui_router
from ormah.config import Settings
from ormah.embeddings import local_adapter, reranker, runtime
from ormah.engine.memory_engine import MemoryEngine


@runtime.local_inference
def worker_identity():
    return current_thread().name, get_ident()


@pytest.fixture
def engine(tmp_path, monkeypatch):
    engine = MemoryEngine(Settings(memory_dir=tmp_path, backup_dir=tmp_path / "backup"))
    seen = []

    def search(*args, **kwargs):
        seen.append(worker_identity())
        return []

    monkeypatch.setattr(engine, "_get_hybrid_search", lambda: SimpleNamespace(search=search))
    yield engine, seen
    engine.shutdown()


def test_direct_search_origins_and_error_reset(engine, monkeypatch):
    eng, seen = engine
    eng.recall_search("memory")
    eng.recall_search_structured("memory")
    eng._search_structured("memory")
    assert [name.startswith("ormah-recall") for name, _ in seen] == [True, True, False]

    def fail(*args, **kwargs):
        assert worker_identity()[0].startswith("ormah-recall")
        raise RuntimeError("search failed")

    monkeypatch.setattr(eng, "_search_structured", fail)
    with pytest.raises(RuntimeError, match="search failed"):
        eng.recall_search_structured("memory")
    assert worker_identity()[0].startswith("ormah-inference")


def test_api_ui_and_mcp_use_engine_recall_policy(engine, monkeypatch):
    from ormah.adapters import mcp_adapter

    eng, seen = engine
    app = FastAPI()
    app.state.engine = eng
    app.include_router(agent_router)
    app.include_router(ui_router)
    with TestClient(app) as client:
        assert client.post("/agent/recall", json={"query": "memory"}).status_code == 200
        assert client.get("/ui/search", params={"q": "memory"}).status_code == 200

    # Follow the real MCP HTTP dispatch into the ASGI route and AnyIO worker.
    original = httpx.AsyncClient
    monkeypatch.setattr(mcp_adapter.httpx, "AsyncClient", lambda **kw: original(
        **kw, transport=httpx.ASGITransport(app=app),
    ))
    anyio.run(mcp_adapter._dispatch, "http://test", "recall", {"query": "memory"})
    assert len(seen) == 3
    assert all(name.startswith("ormah-recall") for name, _ in seen)
    assert worker_identity()[0].startswith("ormah-inference")


def test_blocked_whisper_encoding_does_not_block_recall(engine, monkeypatch):
    eng, seen = engine
    entered, release = Event(), Event()
    calls = []

    class Model:
        def embed(self, texts, **kwargs):
            calls.append(current_thread().name)
            yield np.ones(3)

        def query_embed(self, texts, **kwargs):
            calls.append(current_thread().name)
            entered.set()
            assert release.wait(5)
            yield np.ones(3)

    adapter = local_adapter.LocalAdapter()
    adapter._model = Model()

    def build(**kwargs):
        adapter.encode("whisper single text")
        adapter.encode_query("whisper search")
        return ""

    monkeypatch.setattr(eng.context_builder, "build_whisper_context", build)
    monkeypatch.setattr(eng, "_maybe_get_onboarding_nudge", lambda **kw: None)
    eng.settings.whisper_reranker_enabled = False
    with ThreadPoolExecutor(2) as callers:
        whisper = callers.submit(eng.get_whisper_context, "background prompt")
        try:
            assert entered.wait(5)
            callers.submit(eng.recall_search, "memory").result(timeout=2)
            assert not whisper.done()
        finally:
            release.set()
        whisper.result(timeout=5)
    assert all(name.startswith("ormah-inference") for name in calls)
    assert seen[0][0].startswith("ormah-recall")


def test_both_lanes_overlap_but_each_serializes_including_lazy_iteration():
    first_pair = Barrier(2)
    counter_lock = Lock()
    active = {"general": 0, "recall": 0}
    peaks = active.copy()
    total_peak = 0
    threads = {origin: set() for origin in active}

    class Model:
        def embed(self, texts, **kwargs):
            nonlocal total_peak
            origin = runtime._origin.get()
            with counter_lock:
                active[origin] += 1
                peaks[origin] = max(peaks[origin], active[origin])
                total_peak = max(total_peak, sum(active.values()))
                threads[origin].add(get_ident())
            try:
                # Pair one job from each lane; a single global lock deadlocks.
                first_pair.wait(timeout=5)
                yield np.ones(3)
            finally:
                with counter_lock:
                    active[origin] -= 1

    adapter = local_adapter.LocalAdapter()
    adapter._model = Model()

    def run(origin):
        with runtime.inference_request(origin):
            return adapter.encode("synthetic")

    with ThreadPoolExecutor(8) as callers:
        futures = [callers.submit(run, origin) for _ in range(4) for origin in active]
        for future in futures:
            future.result(timeout=10)
    assert peaks == {"general": 1, "recall": 1}
    assert total_peak == 2
    assert all(len(ids) == 1 for ids in threads.values())
    assert threads["general"].isdisjoint(threads["recall"])
    assert active == {"general": 0, "recall": 0}


@pytest.mark.parametrize("origin", ["general", "recall"])
def test_nested_calls_stay_on_worker_and_recover_after_error(origin):
    @runtime.local_inference
    def outer():
        first = worker_identity()
        # Even a nested origin switch must not create cross-worker waits.
        with runtime.inference_request("recall" if origin == "general" else "general"):
            assert worker_identity() == first
        raise RuntimeError("inference failed")

    with runtime.inference_request(origin):
        with pytest.raises(RuntimeError, match="inference failed"):
            outer()
        assert worker_identity()[0].startswith(
            "ormah-recall" if origin == "recall" else "ormah-inference"
        )
    assert worker_identity()[0].startswith("ormah-inference")


def test_anyio_and_executor_copy_and_reset_context():
    marker = ContextVar("request_marker", default=None)

    @runtime.local_inference
    def read_context():
        assert marker.get() == "request-a"
        marker.set("worker-only")
        return worker_identity()

    async def run():
        marker.set("request-a")
        with runtime.inference_request("recall"):
            identity = await anyio.to_thread.run_sync(read_context)
        assert marker.get() == "request-a"
        general = await anyio.to_thread.run_sync(worker_identity)
        return identity, general

    recall, general = anyio.run(run)
    assert recall[0].startswith("ormah-recall")
    assert general[0].startswith("ormah-inference")
    assert marker.get() is None


@pytest.mark.parametrize("kind", ["encoder", "reranker"])
def test_concurrent_constructor_failure_then_retry_across_lanes(monkeypatch, kind):
    monkeypatch.setattr(local_adapter, "_model_cache", {})
    monkeypatch.setattr(reranker, "_model_cache", {})
    entered, release = Event(), Event()
    model = object()
    attempts = []

    def construct(*args, **kwargs):
        attempts.append(1)
        if len(attempts) == 1:
            entered.set()
            assert release.wait(5)
            raise RuntimeError("load failed")
        return model

    if kind == "encoder":
        monkeypatch.setattr("fastembed.TextEmbedding", construct)

        def load():
            return local_adapter.LocalAdapter("retry").model
    else:
        monkeypatch.setattr("fastembed.rerank.cross_encoder.TextCrossEncoder", construct)

        def load():
            return reranker.preload_model("retry")

    def run(origin):
        with runtime.inference_request(origin):
            return load()

    with ThreadPoolExecutor(2) as callers:
        first = callers.submit(run, "general")
        try:
            assert entered.wait(5)
            second = callers.submit(run, "recall")
        finally:
            release.set()
        with pytest.raises(RuntimeError, match="load failed"):
            first.result(timeout=5)
        assert second.result(timeout=5) is model
    assert load() is model
    assert len(attempts) == 2


def test_debug_timing_does_not_include_input(caplog):
    @runtime.local_inference
    def encode(text):
        return len(text)

    caplog.set_level("DEBUG", logger=runtime.__name__)
    with runtime.inference_request("recall"):
        assert encode("private query") == 13
    records = [r.inference for r in caplog.records if hasattr(r, "inference")]
    assert len(records) == 2
    timing = records[-1]
    assert timing["enqueued"] <= timing["started"] <= timing["ended"]
    assert timing["queue_seconds"] >= 0
    assert timing["execution_seconds"] >= 0
    assert timing["origin"] == "recall"
    assert "private query" not in str(records) + caplog.text


@pytest.mark.parametrize("entry", ["direct", "api", "mcp"])
def test_single_node_recall_feedback_embedding_uses_reserved_lane(engine, monkeypatch, entry):
    from ormah.adapters import mcp_adapter
    from ormah.models.node import MemoryNode

    eng, seen = engine
    node = MemoryNode(type="fact", title="Synthetic memory", content="Test content")
    eng.builder.index_single(eng.file_store.save(node))

    def encode_feedback(prompt):
        seen.append(worker_identity())
        return b""

    monkeypatch.setattr(eng, "_encode_feedback_prompt_vec", encode_feedback)
    app = FastAPI()
    app.state.engine = eng
    app.include_router(agent_router)
    if entry == "direct":
        assert eng.recall_node(node.id)
    elif entry == "api":
        with TestClient(app) as client:
            assert client.get(f"/agent/recall/{node.id}").status_code == 200
    else:
        original = httpx.AsyncClient
        monkeypatch.setattr(mcp_adapter.httpx, "AsyncClient", lambda **kw: original(
            **kw, transport=httpx.ASGITransport(app=app),
        ))
        assert anyio.run(mcp_adapter._dispatch, "http://test", "recall_node", {"node_id": node.id})
    assert len(seen) == 1
    assert seen[0][0].startswith("ormah-recall")
    assert worker_identity()[0].startswith("ormah-inference")

from datetime import datetime, timezone
from unittest.mock import Mock

from eval.bench.datasets import Question, Session, Turn
from eval.bench.store import (
    EmbeddingCache,
    ProviderAdapter,
    extract_session,
    prepare_memories,
    raw_memories,
    seed_memories,
)
from eval.bench.retrieve import retrieve_question
from ormah.background.llm_client import reset_adapter, set_adapter
from tests.test_eval_bench.conftest import FakeProvider


def session():
    return Session(
        "s1", "2000-01-01T00:00:00+00:00", [Turn("Ada", "blue bike commuting daily", "D1:1")]
    )


def test_raw_cache_and_backdated_retrieval(bench_engine, tmp_path, encoder):
    cache = EmbeddingCache(tmp_path / "cache.sqlite", "fake")
    memories = raw_memories("locomo", session())
    seed_memories(bench_engine, memories, cache)
    node = bench_engine.file_store.load(memories[0]["id"])
    assert node.created.year == 2000
    assert (datetime.now(timezone.utc) - node.last_accessed).total_seconds() < 30
    assert node.updated.year == datetime.now(timezone.utc).year
    assert node.content.startswith("Ada:")
    first_calls = encoder.calls
    seed_memories(bench_engine, memories, cache)
    assert encoder.calls == first_calls
    result = retrieve_question(bench_engine, "blue bike commuting", 30)
    assert memories[0]["id"] in [r["node"]["id"] for r in result["ranked"]]
    assert any("turn:D1:1" in r["node"]["tags"] for r in result["ranked"])
    assert cache.hits > 0
    cache.close()


def test_extract_uses_real_ingest_prompt_and_cache(bench_engine, tmp_path):
    provider = FakeProvider("fake", phase="extract")
    provider._call = Mock(wraps=provider._call)
    adapter = ProviderAdapter(provider)
    set_adapter(adapter)
    try:
        first = extract_session(bench_engine, "locomo", session(), tmp_path / "extract", adapter)
        again = extract_session(bench_engine, "locomo", session(), tmp_path / "extract", adapter)
        assert first == again
        assert provider._call.call_count == 1
        assert "memory curator" in provider._call.call_args.args[0]
        assert "session:s1" in first[0]["tags"]
        assert first[0]["turn_provenance"] == "unknown"
        changed = session()
        changed.turns[0].text += " and on weekends"
        extract_session(bench_engine, "locomo", changed, tmp_path / "extract", adapter)
        assert provider._call.call_count == 2
    finally:
        reset_adapter()


def test_retrieval_arguments():
    engine = Mock()
    engine.settings.recall_min_relevance_score = 0.35
    engine.recall_search_structured.return_value = []
    retrieve_question(engine, "when?", 7)
    engine.recall_search_structured.assert_called_once_with(
        "when?", limit=7, min_relevance=0.0, auto_temporal=False, default_space=None
    )


def test_prepare_extract_and_seed(bench_engine, tmp_path, encoder):
    provider = FakeProvider("fake", phase="extract")
    adapter = ProviderAdapter(provider)
    set_adapter(adapter)
    q = Question("q", "locomo", "What?", "bike", "4", "2023-01-01", ["D1:1"], sessions=[session()])
    try:
        memories = prepare_memories(bench_engine, q, "extract", tmp_path / "extract", adapter)
        cache = EmbeddingCache(tmp_path / "vec.sqlite", "fake")
        seed_memories(bench_engine, memories, cache)
        assert (
            bench_engine.graph.get_node(memories[0]["id"])["content"]
            == "Ada rides a blue bike to work."
        )
        cache.close()
    finally:
        reset_adapter()


def test_vector_knn_works_on_sqlite_before_limit_pushdown(bench_engine, encoder):
    """SQLite 3.40 does not pass LIMIT constraints to sqlite-vec virtual tables."""
    from ormah.embeddings.vector_store import VectorStore

    vectors = VectorStore(bench_engine.db)
    vectors.upsert_batch([("a", encoder.encode("bike")), ("b", encoder.encode("boat"))])
    result = vectors.search(encoder.encode("bike"), limit=1)
    assert len(result) == 1
    assert result[0]["id"] == "a"
    assert result[0]["similarity"] == 1


def test_malformed_extraction_does_not_poison_cache(bench_engine, tmp_path):
    import pytest
    from eval.bench.providers import TextResult

    provider = FakeProvider("fake", phase="extract")
    provider._call = Mock(return_value=TextResult('{"wrong": "schema"}', "fake"))
    adapter = ProviderAdapter(provider)
    set_adapter(adapter)
    try:
        with pytest.raises(RuntimeError):
            extract_session(bench_engine, "locomo", session(), tmp_path / "extract", adapter)
        assert not list((tmp_path / "extract").glob("*.jsonl"))
    finally:
        reset_adapter()


def test_extraction_preserves_budget_abort_through_ingest(bench_engine, tmp_path):
    import pytest
    from eval.bench.cost import BudgetExceeded

    provider = FakeProvider("fake", phase="extract")
    provider.complete = Mock(side_effect=BudgetExceeded("next call exceeds budget"))
    adapter = ProviderAdapter(provider)
    set_adapter(adapter)
    try:
        with pytest.raises(BudgetExceeded):
            extract_session(bench_engine, "locomo", session(), tmp_path / "extract", adapter)
    finally:
        reset_adapter()


def test_retrieval_does_not_report_silent_lexical_fallback():
    import logging
    import pytest

    engine = Mock()

    def fallback(*args, **kwargs):
        logging.getLogger("ormah.embeddings.hybrid_search").warning("Vector search failed: broken")
        return []

    engine.recall_search_structured.side_effect = fallback
    with pytest.raises(RuntimeError, match="Vector search failed"):
        retrieve_question(engine, "question", 30)

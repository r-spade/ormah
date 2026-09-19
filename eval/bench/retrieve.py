"""Measure the shipped recall path without wall-clock temporal inference."""

import logging
import time

from eval.bench.answer import memory_context


class _VectorFailure(logging.Handler):
    """Production may fall back to FTS; benchmark scores must expose failures."""

    def __init__(self):
        super().__init__(logging.WARNING)
        self.error = None

    def emit(self, record):
        if "Vector search failed:" in record.getMessage():
            self.error = record.getMessage()


def retrieve_question(engine, question: str, k: int) -> dict:
    logger = logging.getLogger("ormah.embeddings.hybrid_search")
    failure = _VectorFailure()
    logger.addHandler(failure)
    try:
        start = time.perf_counter()
        ranked = engine.recall_search_structured(
            question, limit=k, min_relevance=0.0, auto_temporal=False, default_space=None
        )
        latency = time.perf_counter() - start
    finally:
        logger.removeHandler(failure)
    if failure.error:
        raise RuntimeError(failure.error)
    # Structured search nodes are SQL rows; tags live in node_tags instead.
    tags = engine.graph.get_tags_batch([r["node"]["id"] for r in ranked]) if ranked else {}
    # Retain exactly the content seen by the answerer for offline re-judging.
    ranked = [
        {
            "node": {key: r["node"].get(key) for key in ("id", "title", "content", "created")},
            "score": r["score"],
            "source": r.get("source"),
            **({"raw_cosine": r["raw_cosine"]} if "raw_cosine" in r else {}),
        }
        for r in ranked
    ]
    for item in ranked:
        item["node"]["tags"] = sorted(tags.get(item["node"]["id"], []))
    return {
        "strategy": "recall",
        "reranker_active": False,
        "answer_context_chars": len(memory_context(ranked)),
        "ranked": ranked,
        "latency_s": latency,
        "production_gate": engine.settings.recall_min_relevance_score,
        "passing_gate": sum(
            r["score"] >= engine.settings.recall_min_relevance_score for r in ranked
        ),
    }

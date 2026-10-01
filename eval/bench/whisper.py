"""Benchmark adapter for the shipped whisper, without expanding title-only hits."""

from __future__ import annotations

import logging
import re
import time
from contextlib import contextmanager
from datetime import datetime

from eval.bench.answer import memory_context
from eval.bench.retrieve import _VectorFailure


class RerankerUnavailable(RuntimeError):
    pass


def reranker_active(engine):
    return bool(
        engine.settings.whisper_reranker_enabled and engine._whisper_reranker_available
    )


def require_reranker(engine):
    # startup() synchronously preloads the model, but production catches load failures.
    if not reranker_active(engine):
        raise RerankerUnavailable(
            "Whisper benchmark requires a loaded reranker; model "
            f"{engine.settings.whisper_reranker_model!r} is unavailable. "
            "Embedding-only fallback is not a valid whisper benchmark."
        )


@contextmanager
def reference_date(value: str):
    from ormah.engine.prompt_classifier import _temporal_reference_date

    dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if dt.tzinfo is None:
        raise ValueError("Whisper reference date must include a timezone")
    token = _temporal_reference_date.set(dt)
    try:
        yield
    finally:
        _temporal_reference_date.reset(token)


class _WhisperFailure(_VectorFailure):
    def __init__(self):
        super().__init__()
        self.reranker_error = None

    def emit(self, record):
        super().emit(record)
        message = record.getMessage()
        if "reranker failed" in message or "reranker unavailable" in message:
            self.reranker_error = message
        if "Whisper search failed:" in message or "Preference applicability search failed:" in message:
            self.error = message


def whispered_memories(engine, raw_text, injected_ids):
    """Allowlist debug IDs and copy only their actual rendered titles/previews.

    Match each stored title + short ID in output order. A preview is accepted
    only when it exactly matches the production truncator's output immediately
    below that header. Never consume arbitrary trailing output (onboarding,
    maintenance, framing), or expand a title-only result to its stored content.
    Unknown output formats fail visibly instead of silently spending more context.
    """
    from ormah.engine.context_builder import _truncate_at_word_boundary

    nodes = engine.graph.get_nodes_batch(injected_ids)
    tags = engine.graph.get_tags_batch(injected_ids)
    ranked, blocks = [], []
    cursor = 0
    for node_id in injected_ids:
        node = nodes.get(node_id)
        if node is None:
            raise ValueError(f"Whisper injected missing node {node_id}")
        full_content = node.get("content") or ""
        title = node.get("title") or (
            full_content[:60].strip() + ("…" if len(full_content) > 60 else "")
        )
        header = re.compile(
            r"^- \*\*\[[^\]\n]+\]\*\* " + re.escape(title)
            + r" \(id: " + re.escape(node_id[:8]) + r"\)$", re.MULTILINE
        )
        match = header.search(raw_text, cursor)
        if match is None:
            raise ValueError(f"Whisper output missing expected memory header {node_id}")
        cursor = match.end()
        content = ""
        block = match.group()
        if raw_text.startswith("\n  ", cursor):
            preview = _truncate_at_word_boundary(
                full_content.strip(), engine.settings.whisper_injected_content_max_chars
            )
            if not preview or not raw_text.startswith("\n  " + preview, cursor):
                raise ValueError(f"Whisper output has an unexpected preview for {node_id}")
            content = preview
            cursor += len("\n  " + preview)
            block += "\n  " + preview
        ranked.append({
            "node": {
                "id": node_id, "title": title, "content": content,
                "created": node.get("created"), "tags": sorted(tags.get(node_id, [])),
            },
            "source": "whisper",
        })
        blocks.append(block)
    return ranked, "\n\n".join(blocks)


def retrieve_whisper(engine, question: str, question_date: str) -> dict:
    require_reranker(engine)
    failure = _WhisperFailure()
    loggers = [logging.getLogger(name) for name in (
        "ormah.embeddings.hybrid_search", "ormah.engine.context_builder",
        "ormah.engine.memory_engine",
    )]
    for logger in loggers:
        logger.addHandler(failure)
    try:
        with reference_date(question_date):
            start = time.perf_counter()
            raw_text, injected_ids = engine.get_whisper_context(
                question, space=None, recent_prompts=None, session_id=None, _return_debug=True
            )
            latency = time.perf_counter() - start
    finally:
        for logger in loggers:
            logger.removeHandler(failure)
    if failure.reranker_error:
        raise RerankerUnavailable(failure.reranker_error)
    require_reranker(engine)
    if failure.error:
        raise RuntimeError(failure.error)
    ranked, text = whispered_memories(engine, raw_text, injected_ids)
    return {
        "strategy": "whisper", "ranked": ranked, "injected_ids": injected_ids,
        "raw_whisper_text": raw_text, "whisper_text": text,
        "silent": not injected_ids, "injected_count": len(injected_ids),
        "whisper_context_chars": len(text), "raw_whisper_chars": len(raw_text),
        "answer_context_chars": len(memory_context(ranked)),
        "latency_s": latency, "reranker_active": True,
        "temporal_reference_date": question_date,
    }

"""Cross-encoder reranker for whisper context precision.

Uses linear-rescale blended scoring to combine cross-encoder relevance with
the original embedding score. Unlike sigmoid blending, linear rescale preserves
the CE model's ability to suppress noise (negative CE scores pull blended score
down proportionally).
"""

from __future__ import annotations

import logging
from threading import Lock

from ormah.embeddings.cache import get_fastembed_cache_dir, is_model_cached
from ormah.embeddings.runtime import MAX_BATCH_SIZE, local_inference

logger = logging.getLogger(__name__)

_model_cache: dict[str, object] = {}
_model_cache_lock = Lock()

# CE score range for linear rescale normalization.
# Derived from empirical distribution: MS MARCO MiniLM scores range
# from ~-12 (completely irrelevant) to ~+6 (perfect keyword match).
_CE_MIN = -12.0
_CE_MAX = 6.0


def _linear_rescale(ce_score: float) -> float:
    """Rescale raw CE score to [0, 1] using clamped linear interpolation."""
    return max(0.0, min(1.0, (ce_score - _CE_MIN) / (_CE_MAX - _CE_MIN)))


@local_inference
def rerank(
    query: str,
    candidates: list[dict],
    model_name: str,
    min_score: float,
    blend_alpha: float = 0.6,
    max_doc_chars: int = 512,
) -> list[dict]:
    """Rerank search results using a cross-encoder with linear-rescale blended scoring.

    Final score = alpha * linear_rescale(ce_score) + (1-alpha) * embedding_score

    Linear rescale maps the CE score range [-12, +6] to [0, 1], preserving the
    model's ability to both boost relevant results and suppress irrelevant ones.

    Args:
        query: The user's prompt.
        candidates: List of search result dicts ({"node": {...}, "score": float}).
        model_name: CrossEncoder model name.
        min_score: Drop results below this blended score.
        blend_alpha: Weight for cross-encoder component (0–1). Default 0.6.
        max_doc_chars: Max characters of content to feed to cross-encoder.

    Returns:
        Filtered and reordered candidates with updated scores.
    """
    if not candidates:
        return []

    model = _get_model(model_name)

    # Build doc strings for each candidate
    docs = []
    for r in candidates:
        node = r["node"]
        doc = node.get("title") or ""
        content = node.get("content", "").strip()
        if content and content != doc:
            doc = f"{doc}: {content[:max_doc_chars]}" if doc else content[:max_doc_chars]
        docs.append(doc)

    # FastEmbed already truncates pairs to the model's token window. Bound the
    # batch and concurrent executions, without throwing away query context.
    # Consume the generator on the worker: creating it does not run inference.
    ce_scores = list(model.rerank(query, docs, batch_size=MAX_BATCH_SIZE))

    # Linear-rescale blend with original embedding scores, filter, sort
    reranked = []
    for r, ce_score in zip(candidates, ce_scores):
        ce_rescaled = _linear_rescale(float(ce_score))
        emb_score = r.get("score", 0.0)
        blended = blend_alpha * ce_rescaled + (1 - blend_alpha) * emb_score

        if blended >= min_score:
            reranked.append({
                **r,
                "score": blended,
                "cross_encoder_score": float(ce_score),
                # Absolute relevance in [0, 1], independent of the candidate
                # set — the only score here safe to compare against a fixed
                # threshold (blended/embedding scores are rank-relative).
                "ce_absolute": ce_rescaled,
                "embedding_score": emb_score,
            })

    reranked.sort(key=lambda r: r["score"], reverse=True)
    return reranked


@local_inference
def _get_model(model_name: str):
    if model_name in _model_cache:
        return _model_cache[model_name]
    # Construction only: concurrent inference uses the shared immutable model.
    with _model_cache_lock:
        if model_name in _model_cache:
            return _model_cache[model_name]
        from fastembed.rerank.cross_encoder import TextCrossEncoder

        model = TextCrossEncoder(
            model_name,
            cache_dir=str(get_fastembed_cache_dir()),
            # FastEmbed accepts this top-level option; session_options= is ignored.
            enable_cpu_mem_arena=False,
        )
        _model_cache[model_name] = model
        return model


def preload_model(model_name: str):
    """Load the configured reranker model into the in-process cache."""
    return _get_model(model_name)


def model_is_cached(model_name: str) -> bool:
    """Return True if the reranker model is already present on disk."""
    from fastembed.rerank.cross_encoder import TextCrossEncoder

    return is_model_cached(
        model_name,
        TextCrossEncoder.list_supported_models(),
        cache_dir=get_fastembed_cache_dir(),
    )

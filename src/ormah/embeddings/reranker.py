"""Cross-encoder reranker for whisper context precision.

Uses sigmoid-blended scoring to combine cross-encoder relevance with
the original embedding score, preventing the CE model from over-filtering
semantically relevant memories that lack keyword overlap.
"""

from __future__ import annotations

import logging
import math

logger = logging.getLogger(__name__)

_model_cache: dict[str, object] = {}


def rerank(
    query: str,
    candidates: list[dict],
    model_name: str,
    min_score: float,
    blend_alpha: float = 0.4,
    max_doc_chars: int = 512,
) -> list[dict]:
    """Rerank search results using a cross-encoder with sigmoid-blended scoring.

    Final score = α * sigmoid(ce_score) + (1-α) * embedding_score

    This preserves the reranker's ability to boost truly relevant results
    (ce=+6 → sigmoid≈1.0) while preventing it from destroying valid embedding
    matches (ce=-10 → sigmoid≈0 → falls back to embedding score).

    Args:
        query: The user's prompt.
        candidates: List of search result dicts ({"node": {...}, "score": float}).
        model_name: CrossEncoder model name.
        min_score: Drop results below this blended score.
        blend_alpha: Weight for cross-encoder component (0–1). Default 0.4.
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

    # Score all docs in one batch
    ce_scores = list(model.rerank(query, docs))

    # Sigmoid-blend with original embedding scores, filter, sort
    reranked = []
    for r, ce_score in zip(candidates, ce_scores):
        ce_prob = 1.0 / (1.0 + math.exp(-float(ce_score)))
        emb_score = r.get("score", 0.0)
        blended = blend_alpha * ce_prob + (1 - blend_alpha) * emb_score

        if blended >= min_score:
            reranked.append({
                **r,
                "score": blended,
                "cross_encoder_score": float(ce_score),
                "embedding_score": emb_score,
            })

    reranked.sort(key=lambda r: r["score"], reverse=True)
    return reranked


def _get_model(model_name: str):
    if model_name in _model_cache:
        return _model_cache[model_name]
    from fastembed.rerank.cross_encoder import TextCrossEncoder

    model = TextCrossEncoder(model_name)
    _model_cache[model_name] = model
    return model

"""Canonical probe text for embeddings.

Single source of truth: every vector in ``node_vectors`` and every maintenance
probe must be built from this function, or KNN compares vectors of different
texts.
"""

from __future__ import annotations


def embedding_text(title: str | None, content: str, max_content_chars: int = 512) -> str:
    """Build text for embedding. Truncates content to avoid topic averaging in long docs."""
    prefix = title or ""
    truncated = content[:max_content_chars]
    return f"{prefix} {truncated}".strip()

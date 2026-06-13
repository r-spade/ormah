"""Shared lightweight tokenisation helpers."""

from __future__ import annotations

import re
from collections.abc import Iterable

STOP_WORDS = frozenset({
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "shall", "can", "need", "dare", "ought",
    "user", "i", "me", "my", "we", "our", "you", "your", "he", "she", "it",
    "they", "them", "this", "that", "these", "those", "am", "not", "no",
    "nor", "so", "if", "or", "and", "but", "for", "of", "to", "in",
    "on", "at", "by", "with", "from", "as", "into", "about", "what",
    "which", "who", "whom", "when", "where", "why", "how", "all", "any",
    "each", "every", "both", "few", "more", "most", "other", "some",
    "such", "than", "too", "very", "just", "because", "also",
})

IDENTITY_TOKENS = frozenset({"user", "i", "me", "my", "we", "our", "you", "your"})

_TOKEN_RE = re.compile(r"\b[a-zA-Z][a-zA-Z0-9_-]{2,}\b")


def distinctive_tokens(
    text: str,
    *,
    extra_stop_words: Iterable[str] = (),
    min_len: int = 3,
) -> set[str]:
    """Return lowercased content-bearing tokens from free text."""
    stop_words = STOP_WORDS | {word.lower() for word in extra_stop_words}
    return {
        token
        for token in (match.group(0).lower() for match in _TOKEN_RE.finditer(text))
        if len(token) >= min_len and token not in stop_words
    }


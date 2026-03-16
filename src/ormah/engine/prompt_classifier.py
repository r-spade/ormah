"""Embedding-based intent classifier for whisper-inject prompts."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone

import numpy as np

from ormah.embeddings.base import EmbeddingAdapter

logger = logging.getLogger(__name__)

# Archetype prompts per intent category.  More examples = better embedding
# space coverage for paraphrases the user might actually type.
ARCHETYPES: dict[str, list[str]] = {
    "temporal": [
        "what did we do yesterday",
        "recap our recent progress",
        "what happened last week",
        "what were we working on recently",
        "show me recent changes",
        "summarize what we did today",
        "anything new since last time",
        "what was the last thing we worked on",
    ],
    "identity": [
        "what do you know about me",
        "tell me about myself",
        "my preferences",
        "who am I",
        "remind me what I prefer",
        "what are my settings",
        "my personal information",
        "where does the user live",
        "what is the user's name",
        "tell me about the user",
        "what is their email",
        "where do I live",
        "where am I from",
        "where do I work",
        "what did I study",
        "what is my job",
    ],
    "continuation": [
        "continue where we left off",
        "back to what we were doing",
        "where were we",
        "pick up from last time",
        "let's keep going",
        "as we discussed earlier",
    ],
    "conversational": [
        "hello how are you",
        "thanks for the help",
        "ok sounds good",
        "sure go ahead",
        "good morning",
        "bye",
        "no that's all",
        "great thanks",
        "hi there",
        "hey there",
        "please continue",
        "continue please",
    ],
}

# Maps time-reference keywords to (days_start, days_end | None).
# days_start = how far back the window starts (created_after = now - days_start).
# days_end   = how far back the window ends   (created_before = now - days_end).
#              None means "now" (i.e. the window extends to the present).
_TIME_KEYWORDS: list[tuple[re.Pattern, int, int | None]] = [
    (re.compile(r"\btoday\b", re.IGNORECASE), 1, None),           # 24h ago → now
    (re.compile(r"\byesterday\b", re.IGNORECASE), 2, 1),          # 48h ago → 24h ago
    (re.compile(r"\blast\s+week\b", re.IGNORECASE), 14, 7),       # 14d ago → 7d ago
    (re.compile(r"\bthis\s+week\b", re.IGNORECASE), 7, None),     # 7d ago → now
    (re.compile(r"\blast\s+month\b", re.IGNORECASE), 60, 30),     # 60d ago → 30d ago
    (re.compile(r"\brecently\b|\blately\b", re.IGNORECASE), 3, None),  # 3d ago → now
]

_NUMERIC_TIME_RE = re.compile(
    r"\b(?:last|past)\s+(\d+)\s+(hours?|days?|weeks?|months?)\b",
    re.IGNORECASE,
)

_UNIT_TO_DAYS: dict[str, float] = {"hour": 1 / 24, "day": 1, "week": 7, "month": 30}

_DEFAULT_TEMPORAL_DAYS = 3

# Regex to match all temporal phrases for stripping from search queries.
# Combines _TIME_KEYWORDS patterns + _NUMERIC_TIME_RE into one list.
_TEMPORAL_STRIP_PATTERNS: list[re.Pattern] = [
    re.compile(r"\btoday\b", re.IGNORECASE),
    re.compile(r"\byesterday\b", re.IGNORECASE),
    re.compile(r"\blast\s+week\b", re.IGNORECASE),
    re.compile(r"\bthis\s+week\b", re.IGNORECASE),
    re.compile(r"\blast\s+month\b", re.IGNORECASE),
    re.compile(r"\brecently\b|\blately\b|\brecent\b", re.IGNORECASE),
    _NUMERIC_TIME_RE,
]

# Dangling prepositions left after temporal phrase removal.
_DANGLING_PREP_RE = re.compile(
    r"\b(?:in|during|from|over|for)\s+(?:the\s+)?(?=\s*$|\s*,)", re.IGNORECASE
)


def has_temporal_phrases(prompt: str) -> bool:
    """Return True if *prompt* contains explicit temporal phrases.

    Unlike :func:`extract_time_params`, does **not** apply the default
    fallback — only returns True when a concrete time reference is found.
    """
    if _NUMERIC_TIME_RE.search(prompt):
        return True
    for pattern, _, _ in _TIME_KEYWORDS:
        if pattern.search(prompt):
            return True
    return False


def extract_time_params(prompt: str) -> dict:
    """Parse lightweight time references and return ``created_after``/``created_before`` filters.

    Checks numeric expressions like "last 4 days" first, then falls back
    to static keywords.  Final fallback is "search last 3 days".

    Returns a dict with ``created_after`` (always) and ``created_before``
    (when the window has a bounded end).
    """
    now = datetime.now(timezone.utc)

    # Dynamic numeric patterns: "last 4 days", "past 2 weeks", etc.
    m = _NUMERIC_TIME_RE.search(prompt)
    if m:
        n = int(m.group(1))
        unit = m.group(2).rstrip("s").lower()
        days_per_unit = _UNIT_TO_DAYS.get(unit, 1)
        days = n * days_per_unit

        # Rolling previous-period logic for weeks/months with N > 1:
        # "last 2 weeks" = 4 weeks ago → 2 weeks ago (the previous 2-week window)
        # Days/hours extend to now (user wants "the last N days" ending now)
        if unit in ("week", "month") and n > 1:
            start = now - timedelta(days=days * 2)
            end = now - timedelta(days=days)
            return {
                "created_after": start.isoformat(),
                "created_before": end.isoformat(),
            }
        else:
            start = now - timedelta(days=days)
            return {
                "created_after": start.isoformat(),
                "created_before": now.isoformat(),
            }

    for pattern, days_start, days_end in _TIME_KEYWORDS:
        if pattern.search(prompt):
            start = now - timedelta(days=days_start)
            end = (now - timedelta(days=days_end)) if days_end is not None else now
            return {
                "created_after": start.isoformat(),
                "created_before": end.isoformat(),
            }

    # No specific keyword found — default to 3 days → now
    start = now - timedelta(days=_DEFAULT_TEMPORAL_DAYS)
    return {
        "created_after": start.isoformat(),
        "created_before": now.isoformat(),
    }


def strip_temporal_phrases(prompt: str) -> str:
    """Remove temporal phrases from *prompt*, returning the topical residue.

    Examples::

        "what did we do last week"              → "what did we do"
        "what did I work on whisper last week"   → "what did I work on whisper"
        "changes to the API in the last 3 days"  → "changes to the API"
    """
    result = prompt
    for pat in _TEMPORAL_STRIP_PATTERNS:
        result = pat.sub("", result)

    # Clean up dangling prepositions left behind
    result = _DANGLING_PREP_RE.sub("", result)

    # Collapse multiple spaces and strip
    result = re.sub(r"\s{2,}", " ", result).strip()
    return result


@dataclass
class PromptIntent:
    """Result of classifying a user prompt."""

    categories: list[str]
    """Matched intent categories, e.g. ``["temporal"]``. Falls back to ``["general"]``."""

    search_params: dict = field(default_factory=dict)
    """Extra kwargs to merge into ``recall_search_structured`` call."""


class PromptClassifier:
    """Classify prompt intent using cosine similarity to archetype embeddings.

    Lazy-initialises archetype vectors on first ``classify()`` call so the
    encoder isn't invoked during import / construction.
    """

    # If conversational is matched alongside other categories, only suppress
    # it when another category scores within this margin of conversational's
    # score.  With bge-base, cross-category noise is ~0.65–0.68 while genuine
    # matches are 0.9+, so 0.15 cleanly separates signal from noise.
    _CONV_MARGIN: float = 0.15

    def __init__(
        self,
        encoder: EmbeddingAdapter,
        threshold: float = 0.65,
    ) -> None:
        self._encoder = encoder
        self._threshold = threshold
        # category -> (n_archetypes, dim) matrix of archetype embeddings
        self._archetype_vecs: dict[str, np.ndarray] | None = None

    # ------------------------------------------------------------------
    # Lazy archetype encoding
    # ------------------------------------------------------------------

    def _ensure_archetypes(self) -> None:
        if self._archetype_vecs is not None:
            return
        self._archetype_vecs = {}
        for category, prompts in ARCHETYPES.items():
            vecs = self._encoder.encode_batch(prompts)  # (n, dim)
            # Normalise rows (should already be, but be safe)
            norms = np.linalg.norm(vecs, axis=1, keepdims=True)
            norms = np.where(norms == 0, 1, norms)
            self._archetype_vecs[category] = vecs / norms

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def classify(self, prompt: str) -> PromptIntent:
        """Classify *prompt* and return an intent with search-param overrides."""
        self._ensure_archetypes()
        assert self._archetype_vecs is not None

        prompt_vec = self._encoder.encode(prompt)
        prompt_norm = np.linalg.norm(prompt_vec)
        if prompt_norm == 0:
            return PromptIntent(categories=["general"])
        prompt_vec = prompt_vec / prompt_norm

        # Compute max cosine similarity per category
        scores: dict[str, float] = {}
        for category, arch_vecs in self._archetype_vecs.items():
            sims = arch_vecs @ prompt_vec  # (n_archetypes,)
            scores[category] = float(np.max(sims))

        # Collect all categories above threshold
        matched = [cat for cat, score in scores.items() if score >= self._threshold]

        # Conversational is suppressed only when another category is a strong
        # match — i.e. within *_CONV_MARGIN* of conversational's own score.
        # Embedding models often assign high baseline similarity across
        # categories, so a prompt like "hello" might score 0.88 conversational
        # and 0.67 identity.  The 0.67 is noise (barely above threshold) and
        # should not override conversational.
        if "conversational" in matched and len(matched) > 1:
            conv_score = scores["conversational"]
            strong_others = [
                cat for cat in matched
                if cat != "conversational"
                and scores[cat] >= conv_score - self._CONV_MARGIN
            ]
            if strong_others:
                matched.remove("conversational")
            else:
                # Conversational dominates — keep only it
                matched = ["conversational"]

        # Continuation is suppressed when identity is present — "where do I
        # live" matches continuation archetype "where were we" via the shared
        # "where" token, but the continuation `created_after` filter would
        # exclude old identity nodes.  Identity queries need timeless search.
        if "continuation" in matched and "identity" in matched:
            matched.remove("continuation")

        if not matched:
            return PromptIntent(categories=["general"])

        # Build merged search_params from all matched categories
        search_params: dict = {}
        if "temporal" in matched:
            search_params.update(extract_time_params(prompt))
            stripped = strip_temporal_phrases(prompt)
            if stripped != prompt:
                search_params["search_query"] = stripped
        if "continuation" in matched:
            # Recent memories in current space — same as temporal default
            if "created_after" not in search_params:
                now = datetime.now(timezone.utc)
                search_params["created_after"] = (
                    now - timedelta(days=_DEFAULT_TEMPORAL_DAYS)
                ).isoformat()

        return PromptIntent(categories=sorted(matched), search_params=search_params)

    # ------------------------------------------------------------------
    # Time heuristics (only for temporal intent)
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_time_params(prompt: str) -> dict:
        """Backwards-compatible wrapper around module-level :func:`extract_time_params`."""
        return extract_time_params(prompt)

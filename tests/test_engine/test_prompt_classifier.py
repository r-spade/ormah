"""Tests for the embedding-based prompt intent classifier."""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import MagicMock

import numpy as np
import pytest

from ormah.engine.prompt_classifier import (
    PromptClassifier,
    PromptIntent,
    extract_time_params,
    strip_temporal_phrases,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class FakeEncoder:
    """Deterministic encoder that maps specific phrases to known vectors.

    Uses a simple hash-based approach: each unique phrase gets a random-seeded
    unit vector, so cosine similarity between identical/similar phrases is 1.0
    and between unrelated ones is ~0.
    """

    def __init__(self, dim: int = 32):
        self._dim = dim
        self._cache: dict[str, np.ndarray] = {}

    @property
    def dim(self) -> int:
        return self._dim

    def _get_vec(self, text: str) -> np.ndarray:
        if text not in self._cache:
            rng = np.random.RandomState(hash(text) % (2**31))
            vec = rng.randn(self._dim).astype(np.float32)
            vec /= np.linalg.norm(vec)
            self._cache[text] = vec
        return self._cache[text]

    def encode(self, text: str) -> np.ndarray:
        return self._get_vec(text)

    def encode_batch(self, texts: list[str], batch_size: int = 32) -> np.ndarray:
        return np.stack([self._get_vec(t) for t in texts])


class ControlledEncoder:
    """Encoder where we can explicitly set what vectors are returned.

    This gives us full control over cosine similarities for testing
    classification logic independent of real embeddings.
    """

    def __init__(self, dim: int = 8):
        self._dim = dim
        self._batch_results: list[np.ndarray] = []
        self._encode_result: np.ndarray | None = None

    @property
    def dim(self) -> int:
        return self._dim

    def set_encode_result(self, vec: np.ndarray) -> None:
        self._encode_result = vec

    def set_batch_results(self, vecs: list[np.ndarray]) -> None:
        self._batch_results = list(vecs)

    def encode(self, text: str) -> np.ndarray:
        assert self._encode_result is not None
        return self._encode_result

    def encode_batch(self, texts: list[str], batch_size: int = 32) -> np.ndarray:
        # Return pre-set vectors, consuming from the front
        n = len(texts)
        result = np.stack(self._batch_results[:n])
        self._batch_results = self._batch_results[n:]
        return result


def _unit_vec(dim: int, idx: int) -> np.ndarray:
    """Create a unit vector with 1.0 at position idx."""
    v = np.zeros(dim, dtype=np.float32)
    v[idx] = 1.0
    return v


# ---------------------------------------------------------------------------
# Tests: PromptIntent dataclass
# ---------------------------------------------------------------------------

class TestPromptIntent:

    def test_default_search_params(self):
        intent = PromptIntent(categories=["general"])
        assert intent.search_params == {}

    def test_custom_search_params(self):
        intent = PromptIntent(
            categories=["temporal"],
            search_params={"created_after": "2026-01-01T00:00:00"},
        )
        assert "created_after" in intent.search_params


# ---------------------------------------------------------------------------
# Tests: Classification logic (with controlled encoder)
# ---------------------------------------------------------------------------

class TestClassificationLogic:
    """Test classification decisions with controlled cosine similarities."""

    def test_high_similarity_matches_category(self):
        """When prompt vector is identical to an archetype, it should match."""
        dim = 8
        encoder = ControlledEncoder(dim=dim)

        # We need to set up batch results for all archetype categories
        # temporal (8), identity (7), continuation (6), conversational (8) = 29 total
        from ormah.engine.prompt_classifier import ARCHETYPES

        total_archetypes = sum(len(v) for v in ARCHETYPES.values())

        # All archetype vectors point in different directions (orthogonal-ish)
        # Use the first position for temporal archetypes
        temporal_vec = _unit_vec(dim, 0)
        identity_vec = _unit_vec(dim, 1)
        continuation_vec = _unit_vec(dim, 2)
        conversational_vec = _unit_vec(dim, 3)

        batch_vecs = []
        for cat in ARCHETYPES:
            n = len(ARCHETYPES[cat])
            if cat == "temporal":
                batch_vecs.extend([temporal_vec] * n)
            elif cat == "identity":
                batch_vecs.extend([identity_vec] * n)
            elif cat == "continuation":
                batch_vecs.extend([continuation_vec] * n)
            elif cat == "conversational":
                batch_vecs.extend([conversational_vec] * n)

        encoder.set_batch_results(batch_vecs)

        classifier = PromptClassifier(encoder, threshold=0.65)

        # Prompt vector matches temporal exactly
        encoder.set_encode_result(temporal_vec)
        intent = classifier.classify("what did we do yesterday")
        assert "temporal" in intent.categories
        assert "created_after" in intent.search_params
        assert "created_before" in intent.search_params

    def test_no_match_returns_general(self):
        """When prompt doesn't match any archetype, return general."""
        dim = 8
        encoder = ControlledEncoder(dim=dim)

        from ormah.engine.prompt_classifier import ARCHETYPES

        # All archetypes use positions 0-3
        batch_vecs = []
        for cat in ARCHETYPES:
            n = len(ARCHETYPES[cat])
            batch_vecs.extend([_unit_vec(dim, list(ARCHETYPES.keys()).index(cat))] * n)

        encoder.set_batch_results(batch_vecs)
        classifier = PromptClassifier(encoder, threshold=0.65)

        # Prompt in a completely different direction (position 7)
        encoder.set_encode_result(_unit_vec(dim, 7))
        intent = classifier.classify("refactor the auth module")
        assert intent.categories == ["general"]
        assert intent.search_params == {}

    def test_conversational_sole_match(self):
        """Conversational only applies when it's the sole match."""
        dim = 8
        encoder = ControlledEncoder(dim=dim)

        from ormah.engine.prompt_classifier import ARCHETYPES

        cat_list = list(ARCHETYPES.keys())
        conv_idx = cat_list.index("conversational")

        batch_vecs = []
        for cat in ARCHETYPES:
            n = len(ARCHETYPES[cat])
            batch_vecs.extend([_unit_vec(dim, cat_list.index(cat))] * n)

        encoder.set_batch_results(batch_vecs)
        classifier = PromptClassifier(encoder, threshold=0.65)

        # Prompt matches only conversational
        encoder.set_encode_result(_unit_vec(dim, conv_idx))
        intent = classifier.classify("hello")
        assert intent.categories == ["conversational"]

    def test_conversational_overridden_by_other_intent(self):
        """If conversational + temporal both match, conversational is removed."""
        dim = 8
        encoder = ControlledEncoder(dim=dim)

        from ormah.engine.prompt_classifier import ARCHETYPES

        cat_list = list(ARCHETYPES.keys())
        temporal_idx = cat_list.index("temporal")
        conv_idx = cat_list.index("conversational")

        batch_vecs = []
        for cat in ARCHETYPES:
            n = len(ARCHETYPES[cat])
            batch_vecs.extend([_unit_vec(dim, cat_list.index(cat))] * n)

        encoder.set_batch_results(batch_vecs)
        classifier = PromptClassifier(encoder, threshold=0.65)

        # Prompt matches both temporal and conversational
        mixed = _unit_vec(dim, temporal_idx) + _unit_vec(dim, conv_idx)
        mixed /= np.linalg.norm(mixed)
        encoder.set_encode_result(mixed)
        intent = classifier.classify("hey what did we do yesterday")
        # Conversational should be removed; temporal should remain
        assert "conversational" not in intent.categories
        assert "temporal" in intent.categories

    def test_conversational_dominates_when_others_are_weak(self):
        """Conversational should win when other matches are far below its score."""
        dim = 8
        encoder = ControlledEncoder(dim=dim)

        from ormah.engine.prompt_classifier import ARCHETYPES

        cat_list = list(ARCHETYPES.keys())
        conv_idx = cat_list.index("conversational")
        identity_idx = cat_list.index("identity")

        batch_vecs = []
        for cat in ARCHETYPES:
            n = len(ARCHETYPES[cat])
            batch_vecs.extend([_unit_vec(dim, cat_list.index(cat))] * n)

        encoder.set_batch_results(batch_vecs)
        classifier = PromptClassifier(encoder, threshold=0.65)

        # Prompt heavily weighted toward conversational, slight identity bleed
        # conv=0.95, identity=0.31 — identity above threshold via dim bleed
        # but far below conversational
        vec = _unit_vec(dim, conv_idx) * 0.95 + _unit_vec(dim, identity_idx) * 0.31
        vec /= np.linalg.norm(vec)
        encoder.set_encode_result(vec)
        intent = classifier.classify("hello there")
        # Conversational should dominate — identity is too weak
        assert intent.categories == ["conversational"]

    def test_multi_intent_composition(self):
        """Multiple non-conversational intents can co-exist."""
        dim = 8
        encoder = ControlledEncoder(dim=dim)

        from ormah.engine.prompt_classifier import ARCHETYPES

        cat_list = list(ARCHETYPES.keys())
        temporal_idx = cat_list.index("temporal")
        identity_idx = cat_list.index("identity")

        batch_vecs = []
        for cat in ARCHETYPES:
            n = len(ARCHETYPES[cat])
            batch_vecs.extend([_unit_vec(dim, cat_list.index(cat))] * n)

        encoder.set_batch_results(batch_vecs)
        classifier = PromptClassifier(encoder, threshold=0.5)  # lower threshold

        # Prompt matches both temporal and identity
        mixed = _unit_vec(dim, temporal_idx) + _unit_vec(dim, identity_idx)
        mixed /= np.linalg.norm(mixed)  # cosine sim to each = ~0.707
        encoder.set_encode_result(mixed)
        intent = classifier.classify("what did you learn about me recently")
        assert "temporal" in intent.categories
        assert "identity" in intent.categories


# ---------------------------------------------------------------------------
# Tests: Time parameter extraction
# ---------------------------------------------------------------------------

class TestTimeExtraction:
    """Tests for extract_time_params (bounded time windows)."""

    def _days_ago(self, iso: str) -> float:
        dt = datetime.fromisoformat(iso)
        return (datetime.now(timezone.utc) - dt).total_seconds() / 86400

    def test_yesterday_bounded(self):
        params = extract_time_params("what did we do yesterday")
        assert "created_after" in params
        assert "created_before" in params
        # Window: 2d ago → 1d ago
        assert 1.9 < self._days_ago(params["created_after"]) < 2.1
        assert 0.9 < self._days_ago(params["created_before"]) < 1.1

    def test_last_week_bounded(self):
        params = extract_time_params("what happened last week")
        assert "created_after" in params
        assert "created_before" in params
        # Window: 14d ago → 7d ago
        assert 13.9 < self._days_ago(params["created_after"]) < 14.1
        assert 6.9 < self._days_ago(params["created_before"]) < 7.1

    def test_today_extends_to_now(self):
        params = extract_time_params("what did we do today")
        assert "created_after" in params
        assert "created_before" in params
        # Window: 1d ago → now
        assert 0.95 < self._days_ago(params["created_after"]) < 1.05
        assert self._days_ago(params["created_before"]) < 0.01

    def test_this_week_extends_to_now(self):
        params = extract_time_params("what did we do this week")
        assert "created_after" in params
        assert "created_before" in params
        # Window: 7d ago → now
        assert 6.9 < self._days_ago(params["created_after"]) < 7.1
        assert self._days_ago(params["created_before"]) < 0.01

    def test_recently_extends_to_now(self):
        params = extract_time_params("any recent changes")
        assert "created_after" in params
        assert "created_before" in params
        # Window: 3d ago → now
        assert 2.9 < self._days_ago(params["created_after"]) < 3.1
        assert self._days_ago(params["created_before"]) < 0.01

    def test_no_time_word_defaults_to_3_days(self):
        params = extract_time_params("what were we working on")
        assert "created_after" in params
        assert "created_before" in params
        assert 2.9 < self._days_ago(params["created_after"]) < 3.1
        assert self._days_ago(params["created_before"]) < 0.01

    def test_last_month_bounded(self):
        params = extract_time_params("what happened last month")
        assert "created_after" in params
        assert "created_before" in params
        # Window: 60d ago → 30d ago
        assert 59.9 < self._days_ago(params["created_after"]) < 60.1
        assert 29.9 < self._days_ago(params["created_before"]) < 30.1

    def test_numeric_days_extend_to_now(self):
        params = extract_time_params("what did we do in the last 4 days")
        assert "created_after" in params
        assert "created_before" in params
        # "last N days" extends to now
        assert 3.9 < self._days_ago(params["created_after"]) < 4.1
        assert self._days_ago(params["created_before"]) < 0.01

    def test_numeric_weeks_rolling(self):
        """'last 2 weeks' uses rolling previous-period: 4w ago → 2w ago."""
        params = extract_time_params("show me the past 2 weeks")
        assert "created_after" in params
        assert "created_before" in params
        # Rolling: 28d ago → 14d ago
        assert 27.9 < self._days_ago(params["created_after"]) < 28.1
        assert 13.9 < self._days_ago(params["created_before"]) < 14.1

    def test_numeric_1_week_extends_to_now(self):
        """'last 1 week' (N=1) extends to now, not rolling."""
        params = extract_time_params("show me the past 1 week")
        assert "created_after" in params
        assert "created_before" in params
        assert 6.9 < self._days_ago(params["created_after"]) < 7.1
        assert self._days_ago(params["created_before"]) < 0.01

    def test_numeric_hours(self):
        params = extract_time_params("what happened in the last 6 hours")
        assert "created_after" in params
        dt = datetime.fromisoformat(params["created_after"])
        now = datetime.now(timezone.utc)
        diff_hours = (now - dt).total_seconds() / 3600
        assert 5.9 < diff_hours < 6.1

    def test_numeric_months_rolling(self):
        """'last 3 months' uses rolling: 6m ago → 3m ago."""
        params = extract_time_params("last 3 months summary")
        assert "created_after" in params
        assert "created_before" in params
        # Rolling: 180d ago → 90d ago
        assert 179.9 < self._days_ago(params["created_after"]) < 180.1
        assert 89.9 < self._days_ago(params["created_before"]) < 90.1

    def test_past_synonym(self):
        params = extract_time_params("past 5 days of work")
        assert "created_after" in params
        assert "created_before" in params
        assert 4.9 < self._days_ago(params["created_after"]) < 5.1
        assert self._days_ago(params["created_before"]) < 0.01

    def test_backwards_compat_static_method(self):
        """PromptClassifier._extract_time_params still works."""
        params = PromptClassifier._extract_time_params("what did we do today")
        assert "created_after" in params
        assert "created_before" in params


# ---------------------------------------------------------------------------
# Tests: Temporal phrase stripping
# ---------------------------------------------------------------------------

class TestStripTemporalPhrases:

    def test_strip_last_week(self):
        assert strip_temporal_phrases("what did we do last week") == "what did we do"

    def test_strip_mixed_topical(self):
        result = strip_temporal_phrases("what did I work on whisper last week")
        assert "whisper" in result
        assert "last week" not in result

    def test_strip_numeric_days(self):
        result = strip_temporal_phrases("changes to the API in the last 3 days")
        assert "changes to the API" in result
        assert "last 3 days" not in result

    def test_strip_today(self):
        assert strip_temporal_phrases("what did we do today") == "what did we do"

    def test_strip_yesterday(self):
        assert strip_temporal_phrases("work from yesterday") == "work"

    def test_strip_recently(self):
        result = strip_temporal_phrases("any recent changes to auth")
        assert "auth" in result
        assert "recent" not in result

    def test_no_temporal_returns_unchanged(self):
        prompt = "how does the search pipeline work"
        assert strip_temporal_phrases(prompt) == prompt

    def test_pure_temporal_returns_residue(self):
        """Pure temporal queries should leave some residue (stop words)."""
        result = strip_temporal_phrases("what did we do last week")
        assert result == "what did we do"

    def test_strip_collapses_spaces(self):
        result = strip_temporal_phrases("work   from   yesterday")
        assert "  " not in result


# ---------------------------------------------------------------------------
# Tests: Lazy initialization
# ---------------------------------------------------------------------------

class TestLazyInit:

    def test_archetypes_encoded_on_first_classify(self):
        encoder = FakeEncoder(dim=16)
        classifier = PromptClassifier(encoder, threshold=0.65)
        assert classifier._archetype_vecs is None

        classifier.classify("test prompt")
        assert classifier._archetype_vecs is not None

    def test_archetypes_cached_across_calls(self):
        encoder = FakeEncoder(dim=16)
        classifier = PromptClassifier(encoder, threshold=0.65)

        classifier.classify("first prompt")
        vecs_after_first = classifier._archetype_vecs

        classifier.classify("second prompt")
        assert classifier._archetype_vecs is vecs_after_first

    def test_zero_vector_prompt_returns_general(self):
        """An encoder that returns zero vectors should not crash."""
        encoder = MagicMock()
        encoder.encode_batch.return_value = np.zeros((8, 16), dtype=np.float32)
        encoder.encode.return_value = np.zeros(16, dtype=np.float32)

        classifier = PromptClassifier(encoder, threshold=0.65)
        intent = classifier.classify("")
        assert intent.categories == ["general"]


# ---------------------------------------------------------------------------
# Tests: Continuation intent
# ---------------------------------------------------------------------------

class TestContinuationIntent:

    def test_continuation_adds_created_after(self):
        dim = 8
        encoder = ControlledEncoder(dim=dim)

        from ormah.engine.prompt_classifier import ARCHETYPES

        cat_list = list(ARCHETYPES.keys())
        cont_idx = cat_list.index("continuation")

        batch_vecs = []
        for cat in ARCHETYPES:
            n = len(ARCHETYPES[cat])
            batch_vecs.extend([_unit_vec(dim, cat_list.index(cat))] * n)

        encoder.set_batch_results(batch_vecs)
        classifier = PromptClassifier(encoder, threshold=0.65)

        encoder.set_encode_result(_unit_vec(dim, cont_idx))
        intent = classifier.classify("continue where we left off")
        assert "continuation" in intent.categories
        assert "created_after" in intent.search_params

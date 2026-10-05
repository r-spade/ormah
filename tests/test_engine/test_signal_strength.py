"""Unit tests for the #218 ordinal evidence ladder."""

import json
import math

import pytest

from ormah import signal_strength as ss


def test_bands_are_disjoint_and_ordered():
    """The ladder's central assertion: channel dominates within-channel confidence."""
    ordered = sorted(ss.BANDS, key=lambda band: -band[1])
    assert [band[0] for band in ordered] == [
        "explicit",
        "node_id",
        "title",
        "sentence",
        "auto_llm_judge",
        "implicit",
        "token_overlap",
    ]
    for (upper, upper_lo, _), (lower, _, lower_hi) in zip(ordered, ordered[1:]):
        assert lower_hi < upper_lo, f"{lower} band overlaps {upper}"


def test_token_overlap_starts_at_the_band_floor():
    assert ss.token_overlap_strength(ss.OVERLAP_GATE) == pytest.approx(ss.OVERLAP_FLOOR)


def test_token_overlap_separates_the_observed_domain():
    """0.5..7.583 is the range measured on a live store; no two ratios may tie."""
    values = [ss.token_overlap_strength(r) for r in (0.5, 0.55, 1.167, 1.5, 3.0, 7.583)]
    assert values == sorted(values)
    assert len(set(values)) == len(values)
    assert values[-1] < ss.IMPLICIT


def test_token_overlap_fixes_the_defect_218_names():
    """Today min(0.85, 0.45 + ratio) returns exactly 0.85 for both of these."""
    assert ss.token_overlap_strength(0.5) != ss.token_overlap_strength(1.8)


def test_token_overlap_never_exceeds_its_band():
    """The supremum is strict in exact arithmetic, not in float64.

    float64 reaches it around ratio 37 — five times above the 7.583 maximum observed
    on a live store. That crossover is libm-dependent, so it is documented here and
    not asserted; what is asserted is that the band is never exceeded.
    """
    supremum = ss.OVERLAP_FLOOR + ss.OVERLAP_SPAN
    assert all(ss.token_overlap_strength(n / 10) <= supremum for n in range(5, 1000))
    assert supremum < ss.IMPLICIT


@pytest.mark.parametrize("min_confidence", [0.75, 0.80])
def test_judge_band_is_affine_over_the_callers_min_confidence(min_confidence):
    assert ss.judge_strength(min_confidence, min_confidence, 1) == pytest.approx(ss.JUDGE_LO)
    assert ss.judge_strength(1.0, min_confidence, 1) == pytest.approx(ss.JUDGE_HI)
    midpoint = (min_confidence + 1.0) / 2
    assert ss.judge_strength(midpoint, min_confidence, 1) == pytest.approx(
        (ss.JUDGE_LO + ss.JUDGE_HI) / 2
    )


def test_judge_zero_polarity_carries_no_strength():
    """An uncertain verdict asserts nothing. Its confidence survives in evidence."""
    assert ss.judge_strength(0.35, 0.75, 0) == 0.0
    assert ss.judge_strength(0.99, 0.75, 0) == 0.0


def test_judge_negative_polarity_uses_the_same_band():
    """A confident 'irrelevant' is strong evidence for its own polarity."""
    assert ss.judge_strength(1.0, 0.75, -1) == pytest.approx(ss.JUDGE_HI)


def test_judge_degenerate_min_confidence_does_not_divide_by_zero():
    assert ss.judge_strength(1.0, 1.0, 1) == ss.JUDGE_HI


def test_explicit_and_implicit_no_longer_share_a_strength():
    """Today submit_feedback hardcodes 1.0 for both."""
    assert ss.feedback_strength("explicit", 1) == ss.EXPLICIT
    assert ss.feedback_strength("implicit", 1) == ss.IMPLICIT
    assert ss.feedback_strength("explicit", 1) != ss.feedback_strength("implicit", 1)


def test_feedback_zero_signal_carries_no_strength():
    """Unreachable through the HTTP surface; reachable from a direct Python caller."""
    assert ss.feedback_strength("explicit", 0) == 0.0


def test_unknown_source_fails_closed_to_the_bottom_rung():
    assert ss.feedback_strength("something_new", 1) == ss.UNKNOWN
    assert ss.feedback_strength("auto_heuristic", -1) == ss.UNKNOWN


@pytest.mark.parametrize(
    "match,expected",
    [
        ("node_id", ss.VERBATIM_NODE_ID),
        ("title", ss.VERBATIM_TITLE),
        ("sentence", ss.VERBATIM_SENTENCE),
    ],
)
def test_recompute_reads_the_heuristic_match_kind(match, expected):
    evidence = json.dumps({"match": match})
    assert ss.strength_from_evidence(ss.HEURISTIC_SOURCE, 1, evidence) == expected


def test_recompute_reads_the_overlap_ratio():
    evidence = json.dumps({"match": "token_overlap", "overlap_ratio": 1.167})
    assert ss.strength_from_evidence(ss.HEURISTIC_SOURCE, 1, evidence) == pytest.approx(
        ss.token_overlap_strength(1.167)
    )


def test_recompute_uses_the_rows_own_min_confidence():
    """Not today's setting: the judge stamps min_confidence on every row it writes."""
    lenient = json.dumps({"confidence": 0.80, "min_confidence": 0.75})
    strict = json.dumps({"confidence": 0.80, "min_confidence": 0.80})
    assert ss.strength_from_evidence(
        ss.LLM_JUDGE_SOURCE, 1, lenient
    ) > ss.strength_from_evidence(ss.LLM_JUDGE_SOURCE, 1, strict)


def test_recompute_survives_malformed_evidence():
    assert ss.strength_from_evidence(ss.HEURISTIC_SOURCE, 1, "not json") == ss.UNKNOWN
    assert ss.strength_from_evidence(ss.HEURISTIC_SOURCE, 1, None) == ss.UNKNOWN
    assert ss.strength_from_evidence("implicit", 0, None) == 0.0


# --- Non-finite and out-of-range evidence (#218, Codex peer 2026-08-26) ---------
#
# The migration runs on every startup. json.loads accepts NaN and Infinity by
# default and returns an unbounded int for a large integer literal; SQLite stores a
# NaN REAL as NULL and signals.strength is NOT NULL. So one poisoned historical row
# would raise IntegrityError inside the migration and the server would never boot
# again. These pin the coercion that prevents it.

def test_as_float_survives_an_integer_beyond_the_double_range():
    """float() of a huge int raises OverflowError, which is not a ValueError."""
    assert ss._as_float(int("9" * 400)) == 0.0
    assert ss._as_float(int("9" * 400), default=0.75) == 0.75


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_as_float_rejects_non_finite_values(value):
    assert ss._as_float(value) == 0.0
    assert ss._as_float(value, default=0.75) == 0.75


@pytest.mark.parametrize(
    "ratio", [float("nan"), float("inf"), float("-inf"), int("9" * 400)]
)
def test_recompute_stays_finite_and_in_band_for_poisoned_overlap_ratio(ratio):
    evidence = json.dumps({"match": "token_overlap", "overlap_ratio": ratio})
    result = ss.strength_from_evidence(ss.HEURISTIC_SOURCE, 1, evidence)
    assert math.isfinite(result)
    assert ss.OVERLAP_FLOOR <= result <= ss.OVERLAP_FLOOR + ss.OVERLAP_SPAN


@pytest.mark.parametrize("confidence", [float("nan"), float("inf"), int("9" * 400)])
def test_recompute_stays_finite_and_in_band_for_poisoned_confidence(confidence):
    evidence = json.dumps({"confidence": confidence, "min_confidence": 0.75})
    result = ss.strength_from_evidence(ss.LLM_JUDGE_SOURCE, 1, evidence)
    assert math.isfinite(result)
    assert ss.JUDGE_LO <= result <= ss.JUDGE_HI

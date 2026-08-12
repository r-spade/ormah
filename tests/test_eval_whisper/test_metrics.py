"""Tests for eval/whisper/metrics.py."""
from __future__ import annotations
import pytest
from eval.whisper.metrics import (
    injection_recall, injection_precision, f1_score,
    top2_recall, has_false_positive, suppression_correct,
    compute_prompt_metrics,
)


class TestInjectionRecall:
    def test_all_found(self):
        assert injection_recall(["a", "b"], ["b", "a", "c"]) == 1.0

    def test_partial(self):
        assert injection_recall(["a", "b"], ["a", "c"]) == 0.5

    def test_none_found(self):
        assert injection_recall(["a", "b"], ["c", "d"]) == 0.0

    def test_empty_should_inject_returns_none(self):
        assert injection_recall([], ["a", "b"]) is None

    def test_empty_injected(self):
        assert injection_recall(["a"], []) == 0.0


class TestInjectionPrecision:
    def test_all_relevant(self):
        assert injection_precision(["a", "b"], ["a", "b"]) == 1.0

    def test_partial_relevant(self):
        assert injection_precision(["a"], ["a", "b", "c"]) == pytest.approx(1 / 3)

    def test_empty_should_inject_returns_none(self):
        assert injection_precision([], ["a"]) is None

    def test_empty_injected_returns_zero(self):
        assert injection_precision(["a"], []) == 0.0


class TestF1Score:
    def test_perfect(self):
        assert f1_score(1.0, 1.0) == 1.0

    def test_zero_both(self):
        assert f1_score(0.0, 0.0) == 0.0

    def test_none_propagates(self):
        assert f1_score(None, 0.5) is None
        assert f1_score(0.5, None) is None


class TestTop2Recall:
    def test_in_top2(self):
        assert top2_recall(["a"], ["a", "b", "c"]) == 1.0

    def test_not_in_top2(self):
        assert top2_recall(["c"], ["a", "b", "c"]) == 0.0

    def test_empty_should_inject_returns_none(self):
        assert top2_recall([], ["a"]) is None

    def test_second_position(self):
        assert top2_recall(["b"], ["a", "b", "c"]) == 1.0

    def test_third_position_not_counted(self):
        assert top2_recall(["c"], ["a", "b", "c"]) == 0.0


class TestFalsePositive:
    def test_fp_present(self):
        assert has_false_positive(["x"], ["x", "y"]) is True

    def test_no_fp(self):
        assert has_false_positive(["x"], ["a", "b"]) is False

    def test_empty_should_not_inject(self):
        assert has_false_positive([], ["a"]) is False


class TestSuppressionCorrect:
    def test_correctly_suppressed(self):
        assert suppression_correct(should_suppress=True, injection_fired=False) is True

    def test_incorrectly_not_suppressed(self):
        assert suppression_correct(should_suppress=True, injection_fired=True) is False

    def test_non_noise_returns_none(self):
        assert suppression_correct(should_suppress=False, injection_fired=True) is None
        assert suppression_correct(should_suppress=False, injection_fired=False) is None


class TestComputePromptMetrics:
    def test_perfect_result(self):
        m = compute_prompt_metrics(
            should_inject=["a"],
            should_not_inject=["b"],
            should_suppress=False,
            injected_ids=["a"],
            injection_fired=True,
        )
        assert m["injection_recall"] == 1.0
        assert m["injection_precision"] == 1.0
        assert m["f1"] == 1.0
        assert m["top2_recall"] == 1.0
        assert m["false_positive_present"] is False
        assert m["suppression_correct"] is None
        assert m["injection_fired"] is True

    def test_noise_case_suppressed(self):
        m = compute_prompt_metrics(
            should_inject=[],
            should_not_inject=[],
            should_suppress=True,
            injected_ids=[],
            injection_fired=False,
        )
        assert m["suppression_correct"] is True
        assert m["injection_recall"] is None

    def test_noise_case_not_suppressed(self):
        m = compute_prompt_metrics(
            should_inject=[],
            should_not_inject=[],
            should_suppress=True,
            injected_ids=["a"],
            injection_fired=True,
        )
        assert m["suppression_correct"] is False

    def test_abstain_injection_is_a_false_positive_turn_without_forbidden_ids(self):
        m = compute_prompt_metrics(
            should_inject=[],
            should_not_inject=[],
            should_suppress=True,
            injected_ids=["unexpected"],
            injection_fired=True,
            target_decision="abstain",
        )
        assert m["false_positive_turn"] is True
        assert m["false_positive_present"] is True
        assert m["legacy_false_positive_present"] is False

    def test_contract_counters_are_explicit(self):
        m = compute_prompt_metrics(
            should_inject=["critical-a", "critical-b"],
            should_not_inject=["forbidden-a", "forbidden-b"],
            should_suppress=False,
            injected_ids=["critical-a", "acceptable", "forbidden-a"],
            injection_fired=True,
            may_include=["acceptable"],
        )
        assert m["relevant_injected_count"] == 2
        assert m["injected_count"] == 3
        assert m["critical_found_count"] == 1
        assert m["critical_count"] == 2
        assert m["forbidden_suppressed_count"] == 1
        assert m["forbidden_count"] == 2
        assert m["useful_turn"] is False

    def test_useful_turn_requires_all_critical_nodes_but_allows_acceptable_extras(self):
        m = compute_prompt_metrics(
            should_inject=["critical-a"],
            should_not_inject=[],
            should_suppress=False,
            injected_ids=["critical-a", "acceptable"],
            injection_fired=True,
            may_include=["acceptable"],
            target_decision="inject",
        )
        assert m["useful_turn"] is True
        assert m["turn_helpful"] is True

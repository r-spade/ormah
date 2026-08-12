"""Tests for eval/whisper/report.py."""
from __future__ import annotations
from eval.whisper.runner import PromptResult, WhisperEvalResult
from eval.whisper.report import format_report


def _make_result(category, recall, suppression_correct=None, fp=False,
                 case_id="c-1", prompt="q", should_inject=None, injected_ids=None):
    prec = recall
    metrics = {
        "injection_recall": recall,
        "injection_precision": prec,
        "f1": recall,
        "top2_recall": recall,
        "suppression_correct": suppression_correct,
        "false_positive_present": fp,
        "false_positive_turn": fp and category == "noise",
        "injection_fired": bool(recall),
        "injected_count": 1 if recall is not None else 0,
        "relevant_injected_count": 1 if recall else 0,
        "critical_count": 1 if recall is not None else 0,
        "critical_found_count": 1 if recall else 0,
        "forbidden_count": 0,
        "forbidden_suppressed_count": 0,
        "turn_helpful": bool(recall) if recall is not None else None,
        "useful_turn": bool(recall) if category != "noise" else None,
    }
    return PromptResult(
        case_id=case_id, prompt=prompt, category=category,
        should_inject=should_inject or [],
        injected_ids=injected_ids or [],
        metrics=metrics,
        target_decision="abstain" if category == "noise" else "inject",
        schema_version="2.0",
    )


def _make_eval_result(prompt_results):
    from eval.whisper.runner import _aggregate
    from collections import defaultdict
    by_cat = defaultdict(list)
    for r in prompt_results:
        by_cat[r.category].append(r)
    return WhisperEvalResult(
        prompt_results=prompt_results,
        aggregate=_aggregate(prompt_results),
        category_aggregates={cat: _aggregate(rs) for cat, rs in by_cat.items()},
    )


class TestFormatReport:
    def test_report_contains_overall(self):
        result = _make_eval_result([_make_result("factual", 0.8)])
        report = format_report(result)
        assert "OVERALL" in report

    def test_report_contains_category_name(self):
        result = _make_eval_result([_make_result("preference", 0.67)])
        report = format_report(result)
        assert "preference" in report

    def test_report_includes_nonstandard_category(self):
        result = _make_eval_result([_make_result("prompt_injection", 1.0)])
        assert "prompt_injection" in format_report(result)

    def test_report_contains_false_positive_turn_metric(self):
        result = _make_eval_result([
            _make_result("noise", None, suppression_correct=True),
            _make_result("noise", None, suppression_correct=False),
        ])
        report = format_report(result)
        assert "false_positive_turn_rate" in report
        assert "n=2" in report

    def test_report_contains_contract_provenance_when_available(self):
        result = _make_eval_result([_make_result("factual", 1.0)])
        result.metadata = {
            "corpus_schema_mode": "2.0",
            "corpus_sha256": "a" * 64,
        }
        report = format_report(result)
        assert "contract v2.0" in report
        assert "corpus aaaaaaaaaaaa" in report

    def test_failures_shown_when_flag_set(self):
        result = _make_eval_result([
            _make_result(
                "factual", 0.0,
                case_id="w-fact-001", prompt="what port",
                should_inject=["mem-001"], injected_ids=[],
            )
        ])
        report = format_report(result, show_failures=True)
        assert "w-fact-001" in report
        assert "what port" in report

    def test_failures_hidden_by_default(self):
        result = _make_eval_result([
            _make_result("factual", 0.0, case_id="w-fail-001", prompt="failing prompt")
        ])
        report = format_report(result, show_failures=False)
        assert "failing prompt" not in report

    def test_prompt_count_in_header(self):
        results = [_make_result("factual", 1.0) for _ in range(5)]
        result = _make_eval_result(results)
        report = format_report(result)
        assert "5" in report

    def test_legacy_result_uses_legacy_report(self):
        result = _make_eval_result([_make_result("factual", 1.0)])
        from eval.whisper.runner import _aggregate

        result.aggregate = _aggregate(result.prompt_results, schema_mode="legacy_compat")
        result.category_aggregates = {
            "factual": _aggregate(result.prompt_results, schema_mode="legacy_compat")
        }
        report = format_report(result)
        assert "legacy metrics" in report
        assert "Wilson" not in report

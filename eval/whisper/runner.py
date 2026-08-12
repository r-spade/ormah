"""Whisper eval runner — seeds DB, calls full pipeline, collects metrics per prompt."""
from __future__ import annotations
from collections import defaultdict
from dataclasses import dataclass, field
from math import sqrt
from typing import Optional

from eval.whisper.contract import SCHEMA_VERSION, corpus_schema_mode, infer_target_decision
from eval.whisper.metrics import compute_prompt_metrics
from eval.whisper.seeder import seed_case


@dataclass
class PromptResult:
    case_id: str
    prompt: str
    category: str
    should_inject: list[str]
    injected_ids: list[str]
    metrics: dict
    may_include: list[str] = field(default_factory=list)
    should_not_inject: list[str] = field(default_factory=list)
    should_suppress: bool = False
    intent_categories: list[str] | None = None
    has_temporal_phrases: bool | None = None
    session_id: str | None = None
    recent_prompts: list[str] | None = None
    target_decision: str = "inject"
    schema_version: str | None = None


@dataclass
class WhisperEvalResult:
    prompt_results: list[PromptResult] = field(default_factory=list)
    aggregate: dict = field(default_factory=dict)
    category_aggregates: dict = field(default_factory=dict)
    metadata: dict = field(default_factory=dict)


def run_whisper_eval(
    cases: list[dict],
    engine,
    *,
    simulate_session: bool = False,
    preserve_self: bool | None = None,
    metadata: dict | None = None,
) -> WhisperEvalResult:
    """Run the whisper eval pipeline over *cases*."""
    prompt_results: list[PromptResult] = []

    for case in cases:
        effective_preserve_self = (
            bool(case.get("preserve_self", False)) if preserve_self is None else preserve_self
        )
        seed_case(engine, case, preserve_self=effective_preserve_self)
        space = case.get("space")
        case_simulate_session = bool(case.get("simulate_session", False)) or simulate_session
        case_session_id = case.get("session_id")
        session_buf: list[str] = []

        for prompt_obj in case.get("prompts", []):
            text = prompt_obj["text"]
            category = prompt_obj.get("category", "general")
            expected = prompt_obj.get("expected", {})
            should_inject = expected.get("must_include") or expected.get("should_inject", [])
            may_include = expected.get("may_include", [])
            should_not_inject = expected.get("must_not_include") or expected.get("should_not_inject", [])
            target_decision = infer_target_decision(expected)
            should_suppress = expected.get(
                "must_be_silent",
                expected.get("should_suppress", target_decision == "abstain"),
            )

            # Capture intent classification (same one used by whisper) for analysis.
            intent_categories = None
            has_temporal_phrases = None
            try:
                classifier = engine.context_builder._get_classifier()
                if classifier is not None:
                    intent_categories = classifier.classify(text).categories
                from ormah.engine.prompt_classifier import has_temporal_phrases as _htp
                has_temporal_phrases = _htp(text)
            except Exception:
                pass

            # Session simulation: mimic /agent/whisper route behavior.
            session_id = prompt_obj.get("session_id") or case_session_id
            if "recent_prompts" in prompt_obj:
                recent_prompts = prompt_obj.get("recent_prompts")
            elif case_simulate_session and session_id and text.strip():
                # First message: None; subsequent: all prior prompts (bounded by settings buffer size).
                if session_buf:
                    buf_size = getattr(engine.settings, "whisper_context_buffer_size", 5)
                    recent_prompts = session_buf[-buf_size:]
                else:
                    recent_prompts = None
            else:
                recent_prompts = []

            whisper_text, injected_ids = engine.get_whisper_context(
                prompt=text,
                space=space,
                recent_prompts=recent_prompts,
                session_id=session_id,
                _return_debug=True,
            )
            if case_simulate_session and session_id and text.strip():
                session_buf.append(text.strip())

            metrics = compute_prompt_metrics(
                should_inject=should_inject,
                should_not_inject=should_not_inject,
                should_suppress=should_suppress,
                injected_ids=injected_ids,
                injection_fired=bool(whisper_text.strip()),
                may_include=may_include,
                target_decision=target_decision,
            )

            prompt_results.append(PromptResult(
                case_id=case["id"],
                prompt=text,
                category=category,
                should_inject=should_inject,
                may_include=may_include,
                should_not_inject=should_not_inject,
                should_suppress=should_suppress,
                intent_categories=intent_categories,
                has_temporal_phrases=has_temporal_phrases,
                session_id=session_id,
                recent_prompts=recent_prompts,
                target_decision=target_decision,
                schema_version=case.get("schema_version"),
                injected_ids=injected_ids,
                metrics=metrics,
            ))

    schema_mode = corpus_schema_mode(cases)
    if schema_mode == "mixed":
        raise ValueError("Cannot evaluate mixed legacy and schema v2 cases")
    aggregate = _aggregate(prompt_results, schema_mode=schema_mode)
    by_cat = defaultdict(list)
    for r in prompt_results:
        by_cat[r.category].append(r)
    category_aggregates = {
        cat: _aggregate(results, schema_mode=schema_mode) for cat, results in by_cat.items()
    }

    return WhisperEvalResult(
        prompt_results=prompt_results,
        aggregate=aggregate,
        category_aggregates=category_aggregates,
        metadata={"corpus_schema_mode": schema_mode, **(metadata or {})},
    )


def _wilson_interval(successes: int, total: int, z: float = 1.96) -> list[float] | None:
    """Return a two-sided Wilson score interval for a binomial proportion."""
    if total == 0:
        return None
    proportion = successes / total
    denominator = 1 + z**2 / total
    centre = (proportion + z**2 / (2 * total)) / denominator
    margin = (
        z
        * sqrt((proportion * (1 - proportion) + z**2 / (4 * total)) / total)
        / denominator
    )
    return [max(0.0, centre - margin), min(1.0, centre + margin)]


def _ratio(successes: int, total: int) -> float | None:
    return successes / total if total else None


def _aggregate(
    prompt_results: list[PromptResult], *, schema_mode: str | None = None
) -> dict:
    """Dispatch to the metric contract declared by the corpus."""
    if schema_mode is None:
        versions = {result.schema_version for result in prompt_results}
        schema_mode = SCHEMA_VERSION if versions == {SCHEMA_VERSION} else "legacy_compat"
    if schema_mode == "legacy_compat":
        return _aggregate_legacy(prompt_results)
    if schema_mode != SCHEMA_VERSION:
        raise ValueError(f"Unsupported corpus schema mode: {schema_mode}")
    return _aggregate_v2(prompt_results)


def _aggregate_legacy(prompt_results: list[PromptResult]) -> dict:
    """Preserve the pre-C08 metrics so existing baselines and Make gates do not move."""
    non_noise = [r for r in prompt_results if r.metrics["suppression_correct"] is None]
    noise = [r for r in prompt_results if r.metrics["suppression_correct"] is not None]

    def _avg(key: str, results: list[PromptResult]) -> Optional[float]:
        vals = [r.metrics[key] for r in results if r.metrics.get(key) is not None]
        return sum(vals) / len(vals) if vals else None

    result: dict = {
        "total_prompts": len(prompt_results),
        "injection_recall": _avg("injection_recall", non_noise),
        "injection_precision": _avg("injection_precision", non_noise),
        "f1": _avg("f1", non_noise),
        "top2_recall": _avg("top2_recall", non_noise),
        "false_positive_rate": (
            sum(
                1
                for r in prompt_results
                if r.metrics.get(
                    "legacy_false_positive_present",
                    bool(set(r.should_not_inject) & set(r.injected_ids)),
                )
            )
            / len(prompt_results)
            if prompt_results
            else None
        ),
    }
    if noise:
        correct = sum(1 for r in noise if r.metrics["suppression_correct"])
        result["suppression_accuracy"] = correct / len(noise)
        result["suppression_correct_count"] = correct
        result["suppression_count"] = len(noise)
    return result


def _aggregate_v2(prompt_results: list[PromptResult]) -> dict:
    """Aggregate exact contract metrics using their declared denominators.

    Node precision and useful recall are micro-averaged. False-positive turn
    rate is computed only over abstain-labeled turns. The legacy implementation
    macro-averaged prompt scores and divided forbidden-node hits by all turns;
    those values must not be compared as a trend with this contract.
    """
    noise = [r for r in prompt_results if r.metrics["suppression_correct"] is not None]

    def _avg(key: str, results: list[PromptResult]) -> Optional[float]:
        vals = [r.metrics[key] for r in results if r.metrics.get(key) is not None]
        return sum(vals) / len(vals) if vals else None

    injected_count = sum(r.metrics.get("injected_count", 0) for r in prompt_results)
    relevant_injected_count = sum(
        r.metrics.get("relevant_injected_count", 0) for r in prompt_results
    )
    critical_count = sum(r.metrics.get("critical_count", 0) for r in prompt_results)
    critical_found_count = sum(
        r.metrics.get("critical_found_count", 0) for r in prompt_results
    )
    inject_required_turns = [r for r in prompt_results if r.target_decision == "inject"]
    useful_turn_count = sum(
        1 for r in inject_required_turns if r.metrics.get("useful_turn", False)
    )
    abstain = [r for r in prompt_results if r.target_decision == "abstain"]
    false_positive_turn_count = sum(
        1 for r in abstain if r.metrics.get("false_positive_turn", False)
    )
    forbidden_count = sum(r.metrics.get("forbidden_count", 0) for r in prompt_results)
    forbidden_suppressed_count = sum(
        r.metrics.get("forbidden_suppressed_count", 0) for r in prompt_results
    )
    scorable_injected_turns = [
        r for r in prompt_results if r.metrics.get("turn_helpful") is not None
    ]
    injected_turn_count = sum(
        1 for r in prompt_results if r.metrics.get("injection_fired", False)
    )
    unscorable_injected_turn_count = injected_turn_count - len(scorable_injected_turns)
    helpful_turn_count = sum(
        1 for r in scorable_injected_turns if r.metrics["turn_helpful"]
    )
    injection_precision_value = _ratio(relevant_injected_count, injected_count)
    injection_recall_value = _ratio(critical_found_count, critical_count)
    useful_recall_value = _ratio(useful_turn_count, len(inject_required_turns))
    false_positive_turn_rate = _ratio(false_positive_turn_count, len(abstain))
    suppression_accuracy_value = _ratio(forbidden_suppressed_count, forbidden_count)
    strict_label_turn_precision = _ratio(helpful_turn_count, len(scorable_injected_turns))

    result: dict = {
        "total_prompts": len(prompt_results),
        "decision_counts": {
            decision: sum(1 for r in prompt_results if r.target_decision == decision)
            for decision in ("inject", "abstain", "ask_qualify")
        },
        "injection_precision": injection_precision_value,
        # True turn precision requires material-helpfulness or downstream judgment.
        # Exact node labels are only an offline proxy and remain separately named.
        "turn_precision": None,
        "strict_label_turn_precision": strict_label_turn_precision,
        "unscorable_injected_turn_count": unscorable_injected_turn_count,
        "useful_recall": useful_recall_value,
        "injection_recall": injection_recall_value,
        "f1": (
            2 * injection_precision_value * injection_recall_value
            / (injection_precision_value + injection_recall_value)
            if injection_precision_value is not None
            and injection_recall_value is not None
            and injection_precision_value + injection_recall_value > 0
            else 0.0
            if injection_precision_value == injection_recall_value == 0
            else None
        ),
        "top2_recall": _avg("top2_recall", prompt_results),
        "false_positive_turn_rate": false_positive_turn_rate,
        # Compatibility alias; unlike the legacy metric, the denominator is
        # abstain-labeled turns, as required by the quality contract.
        "false_positive_rate": false_positive_turn_rate,
        "suppression_accuracy": suppression_accuracy_value,
        "sample_counts": {
            "injection_precision": injected_count,
            "turn_precision": 0,
            "strict_label_turn_precision": len(scorable_injected_turns),
            "useful_recall": len(inject_required_turns),
            "injection_recall": critical_count,
            "false_positive_turn_rate": len(abstain),
            "suppression_accuracy": forbidden_count,
        },
        "descriptive_wilson_intervals": {
            "injection_precision": _wilson_interval(relevant_injected_count, injected_count),
            "turn_precision": None,
            "strict_label_turn_precision": _wilson_interval(
                helpful_turn_count, len(scorable_injected_turns)
            ),
            "useful_recall": _wilson_interval(
                useful_turn_count, len(inject_required_turns)
            ),
            "injection_recall": _wilson_interval(critical_found_count, critical_count),
            "false_positive_turn_rate": _wilson_interval(
                false_positive_turn_count, len(abstain)
            ),
            "suppression_accuracy": _wilson_interval(
                forbidden_suppressed_count, forbidden_count
            ),
        },
        "unavailable_metrics": [
            "repeat_irrelevance_rate",
            "downstream_lift",
            "delivery_confirmed_turn_precision",
            "turn_precision",
        ],
    }

    if noise:
        correct = sum(1 for r in noise if r.metrics["suppression_correct"])
        result["legacy_silence_accuracy"] = correct / len(noise)
        result["legacy_silence_correct_count"] = correct
        result["legacy_silence_count"] = len(noise)

    return result


def _aggregate_by_category(prompt_results: list[PromptResult]) -> dict:
    by_cat: dict[str, list[PromptResult]] = defaultdict(list)
    for r in prompt_results:
        by_cat[r.category].append(r)
    return {cat: _aggregate(results) for cat, results in by_cat.items()}

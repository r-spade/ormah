"""Format whisper eval results as a human-readable table."""
from __future__ import annotations

from eval.whisper.runner import WhisperEvalResult

_CATEGORY_ORDER = [
    "preference", "factual", "decision", "technical",
    "identity", "temporal", "continuation", "noise",
]


def _fmt(val, width=6) -> str:
    if val is None:
        return " " * width
    return f"{val:.2f}".rjust(width)


def _fmt_interval(interval) -> str:
    if interval is None:
        return "N/A"
    return f"[{interval[0]:.3f}, {interval[1]:.3f}]"


def format_report(result: WhisperEvalResult, show_failures: bool = False) -> str:
    if "descriptive_wilson_intervals" not in result.aggregate:
        return _format_legacy_report(result, show_failures=show_failures)

    lines = []
    total = result.aggregate.get("total_prompts", 0)
    n_cats = len(result.category_aggregates)
    lines.append(f"Whisper Eval  ({total} prompts, {n_cats} categories)")
    schema_version = result.metadata.get("corpus_schema_mode")
    corpus_hash = result.metadata.get("corpus_sha256")
    if schema_version:
        provenance = f"contract v{schema_version}"
        if corpus_hash:
            provenance += f", corpus {corpus_hash[:12]}"
        lines.append(provenance)
    lines.append("═" * 92)
    lines.append(
        f"{'':20s}  {'useful':>7}  {'node-r':>7}  {'node-p':>7}  "
        f"{'strict-p':>8}  {'f1':>7}  {'fp-turn':>7}"
    )

    categories = [cat for cat in _CATEGORY_ORDER if cat in result.category_aggregates]
    categories += sorted(set(result.category_aggregates) - set(_CATEGORY_ORDER))
    for cat in categories:
        agg = result.category_aggregates.get(cat)
        if agg is None:
            continue
        count = agg.get("total_prompts", 0)
        label = f"{cat} ({count})"

        lines.append(
            f"{label:20s}"
            f"  {_fmt(agg.get('useful_recall'))}"
            f"  {_fmt(agg.get('injection_recall'))}"
            f"  {_fmt(agg.get('injection_precision'))}"
            f"  {_fmt(agg.get('strict_label_turn_precision'), width=7)}"
            f"  {_fmt(agg.get('f1'))}"
            f"  {_fmt(agg.get('false_positive_turn_rate'))}"
        )

    agg = result.aggregate
    lines.append("═" * 92)
    lines.append(
        f"{'OVERALL':20s}"
        f"  {_fmt(agg.get('useful_recall'))}"
        f"  {_fmt(agg.get('injection_recall'))}"
        f"  {_fmt(agg.get('injection_precision'))}"
        f"  {_fmt(agg.get('strict_label_turn_precision'), width=7)}"
        f"  {_fmt(agg.get('f1'))}"
        f"  {_fmt(agg.get('false_positive_turn_rate'))}"
    )
    lines.append("")
    lines.append(
        "Descriptive Wilson intervals (naive; observations may cluster by case/session):"
    )
    intervals = agg.get("descriptive_wilson_intervals", {})
    counts = agg.get("sample_counts", {})
    for key in (
        "injection_precision",
        "strict_label_turn_precision",
        "useful_recall",
        "injection_recall",
        "false_positive_turn_rate",
        "suppression_accuracy",
    ):
        lines.append(
            f"  {key:28s} {_fmt_interval(intervals.get(key)):16s}  "
            f"n={counts.get(key, 0)}"
        )
    unavailable = agg.get("unavailable_metrics", [])
    if unavailable:
        lines.append(f"Unavailable (not inferred): {', '.join(unavailable)}")

    if show_failures:
        failures = _collect_failures(result)
        if failures:
            lines.append("")
            lines.append(f"FAILURES ({len(failures)}):")
            for f in failures:
                lines.append(f"  {f['case_id']:20s}  [{f['category']}]  \"{f['prompt']}\"")
                expected_str = str(f['should_inject']) if f['should_inject'] else ("(silent)" if f.get("should_suppress") else "[]")
                got_str = str(f['injected_ids']) if f['injected_ids'] else "[]"
                extra = f.get("extra_injected_ids") or []
                if extra:
                    lines.append(f"    expected: {expected_str}  injected: {got_str}  extra: {extra}")
                else:
                    lines.append(f"    expected: {expected_str}  injected: {got_str}")

    return "\n".join(lines)


def _format_legacy_report(result: WhisperEvalResult, show_failures: bool = False) -> str:
    """Render the unchanged pre-v2 report for local legacy baselines."""
    lines = []
    total = result.aggregate.get("total_prompts", 0)
    n_cats = len(result.category_aggregates)
    lines.append(f"Whisper Eval  ({total} prompts, {n_cats} categories; legacy metrics)")
    lines.append("═" * 72)
    lines.append(f"{'':20s}  {'recall':>7}  {'prec':>7}  {'f1':>7}  {'top2':>7}  {'fp_rate':>7}")
    categories = [cat for cat in _CATEGORY_ORDER if cat in result.category_aggregates]
    categories += sorted(set(result.category_aggregates) - set(_CATEGORY_ORDER))
    for cat in categories:
        aggregate = result.category_aggregates[cat]
        count = aggregate.get("total_prompts", 0)
        if cat == "noise":
            accuracy = aggregate.get("suppression_accuracy")
            correct = aggregate.get("suppression_correct_count", 0)
            total_noise = aggregate.get("suppression_count", 0)
            lines.append("─" * 72)
            lines.append(
                f"{'noise':20s}  suppression_accuracy: {_fmt(accuracy).strip()}"
                f"  ({correct}/{total_noise} correctly silent)"
            )
            lines.append("─" * 72)
        else:
            lines.append(
                f"{f'{cat} ({count})':20s}"
                f"  {_fmt(aggregate.get('injection_recall'))}"
                f"  {_fmt(aggregate.get('injection_precision'))}"
                f"  {_fmt(aggregate.get('f1'))}"
                f"  {_fmt(aggregate.get('top2_recall'))}"
                f"  {_fmt(aggregate.get('false_positive_rate'))}"
            )
    aggregate = result.aggregate
    lines.append("═" * 72)
    lines.append(
        f"{'OVERALL':20s}"
        f"  {_fmt(aggregate.get('injection_recall'))}"
        f"  {_fmt(aggregate.get('injection_precision'))}"
        f"  {_fmt(aggregate.get('f1'))}"
        f"  {_fmt(aggregate.get('top2_recall'))}"
        f"  {_fmt(aggregate.get('false_positive_rate'))}"
    )
    if show_failures:
        failures = _collect_failures(result)
        if failures:
            lines.append("")
            lines.append(f"FAILURES ({len(failures)}):")
            for failure in failures:
                lines.append(
                    f"  {failure['case_id']:20s}  [{failure['category']}]  "
                    f"\"{failure['prompt']}\""
                )
                expected = (
                    str(failure["should_inject"])
                    if failure["should_inject"]
                    else "(silent)"
                    if failure.get("should_suppress")
                    else "[]"
                )
                injected = str(failure["injected_ids"]) if failure["injected_ids"] else "[]"
                extra = failure.get("extra_injected_ids") or []
                suffix = f"  extra: {extra}" if extra else ""
                lines.append(f"    expected: {expected}  injected: {injected}{suffix}")
    return "\n".join(lines)


def _collect_failures(result: WhisperEvalResult) -> list[dict]:
    failures = []
    for r in result.prompt_results:
        m = r.metrics
        is_failure = (
            (m["injection_recall"] is not None and m["injection_recall"] < 1.0)
            or (m.get("extra_injected_count") or 0) > 0
            or m["false_positive_present"]
            or m["suppression_correct"] is False
        )
        if is_failure:
            failures.append({
                "case_id": r.case_id,
                "category": r.category,
                "prompt": r.prompt,
                "should_inject": r.should_inject,
                "should_suppress": getattr(r, "should_suppress", False),
                "injected_ids": r.injected_ids,
                "extra_injected_ids": m.get("extra_injected_ids", []),
            })
    return failures

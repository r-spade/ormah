"""Per-turn primitives for the whisper measurement contract."""
from __future__ import annotations
from typing import Optional


def injection_recall(should_inject: list[str], injected_ids: list[str]) -> Optional[float]:
    """Fraction of should_inject nodes that appeared in injected output."""
    if not should_inject:
        return None
    injected_set = set(injected_ids)
    return sum(1 for nid in should_inject if nid in injected_set) / len(should_inject)


def injection_precision(
    should_inject: list[str],
    injected_ids: list[str],
    may_include: list[str] | None = None,
) -> Optional[float]:
    """Fraction of injected nodes that were relevant.

    By default, relevance is defined as membership in *should_inject*.
    When *may_include* is provided, those nodes are treated as acceptable
    extra injections (useful when multiple answers are reasonable).
    """
    if not should_inject and not (may_include or []):
        return None
    if not injected_ids:
        return 0.0
    relevant = set(should_inject) | set(may_include or [])
    return sum(1 for nid in injected_ids if nid in relevant) / len(injected_ids)


def f1_score(recall: Optional[float], precision: Optional[float]) -> Optional[float]:
    if recall is None or precision is None:
        return None
    if recall + precision == 0:
        return 0.0
    return 2 * recall * precision / (recall + precision)


def top2_recall(should_inject: list[str], injected_ids: list[str]) -> Optional[float]:
    """Fraction of should_inject nodes in top-2 injected positions (shown in full)."""
    if not should_inject:
        return None
    top2 = set(injected_ids[:2])
    return sum(1 for nid in should_inject if nid in top2) / len(should_inject)


def has_false_positive(should_not_inject: list[str], injected_ids: list[str]) -> bool:
    """True if any should_not_inject node appeared in injected output."""
    injected_set = set(injected_ids)
    return any(nid in injected_set for nid in should_not_inject)


def suppression_correct(should_suppress: bool, injection_fired: bool) -> Optional[bool]:
    """For noise cases: True if pipeline correctly stayed silent. None for non-noise."""
    if not should_suppress:
        return None
    return not injection_fired


def compute_prompt_metrics(
    should_inject: list[str],
    should_not_inject: list[str],
    should_suppress: bool,
    injected_ids: list[str],
    injection_fired: bool,
    may_include: list[str] | None = None,
    target_decision: str = "inject",
) -> dict:
    rec = injection_recall(should_inject, injected_ids)
    prec = injection_precision(should_inject, injected_ids, may_include=may_include)
    injected_set = set(injected_ids)
    must_set = set(should_inject)
    allowed_set = must_set | set(may_include or [])
    extras = [nid for nid in injected_ids if nid not in allowed_set]
    forbidden_set = set(should_not_inject)
    forbidden_injected = forbidden_set & injected_set
    critical_found_count = sum(1 for nid in should_inject if nid in injected_set)
    relevant_injected_count = sum(1 for nid in injected_ids if nid in allowed_set)
    useful_turn = None
    if target_decision == "inject":
        useful_turn = injection_fired and critical_found_count == len(should_inject)
    turn_helpful = None
    if injection_fired and target_decision != "ask_qualify":
        turn_helpful = (
            target_decision == "inject"
            and critical_found_count == len(should_inject)
            and not extras
            and not forbidden_injected
        )
    return {
        "injection_recall": rec,
        "injection_precision": prec,
        "f1": f1_score(rec, prec),
        "top2_recall": top2_recall(should_inject, injected_ids),
        "suppression_correct": suppression_correct(should_suppress, injection_fired),
        "false_positive_present": (
            (target_decision == "abstain" and injection_fired)
            or has_false_positive(should_not_inject, injected_ids)
        ),
        "legacy_false_positive_present": has_false_positive(
            should_not_inject, injected_ids
        ),
        "false_positive_turn": target_decision == "abstain" and injection_fired,
        "target_decision": target_decision,
        "useful_turn": useful_turn,
        "turn_helpful": turn_helpful,
        "injection_fired": injection_fired,
        "injected_count": len(injected_ids),
        "relevant_injected_count": relevant_injected_count,
        "critical_count": len(should_inject),
        "critical_found_count": critical_found_count,
        "missing_must_count": len(should_inject) - critical_found_count,
        "forbidden_count": len(forbidden_set),
        "forbidden_suppressed_count": len(forbidden_set - injected_set),
        "extra_injected_count": len(extras),
        "extra_injected_ids": extras,
    }

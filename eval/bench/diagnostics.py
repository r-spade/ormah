"""Observable, non-adjudicating failure-stage diagnostics for public benchmarks."""

from __future__ import annotations

from collections import Counter

from eval.bench.answer import memory_context


def _turn_tags(item: dict) -> set[str]:
    tags = item.get("node", item).get("tags") or []
    if isinstance(tags, str):
        import json

        tags = json.loads(tags)
    return {tag[5:] for tag in tags if tag.startswith("turn:")}


def _session_tags(item: dict) -> set[str]:
    tags = item.get("node", item).get("tags") or []
    if isinstance(tags, str):
        import json

        tags = json.loads(tags)
    return {tag[8:] for tag in tags if tag.startswith("session:")}


def _coverage(expected: set[str], observed: set[str]) -> dict:
    hit = sorted(expected & observed)
    missed = sorted(expected - observed)
    return {
        "expected": len(expected),
        "hit": len(hit),
        "recall": len(hit) / len(expected) if expected else None,
        "hit_ids": hit,
        "missed_ids": missed,
    }


def diagnose_question(row: dict, stored_memories: list[dict], mode: str) -> dict:
    """Describe observable stages without treating evidence labels as answer proof."""
    evidence = row.get("source_evidence") or []
    resolved_turns = {item["turn_id"] for item in evidence}
    # LoCoMo evidence IDs are turn IDs. LongMemEval's gold_ids are session IDs;
    # its supporting turn IDs come exclusively from has_answer annotations.
    expected_turns = (
        set(row.get("gold_ids") or []) if row["dataset"] == "locomo" else resolved_turns
    )
    expected_sessions = (
        set(row.get("gold_ids") or [])
        if row["dataset"] == "longmemeval"
        else {item["session_id"] for item in evidence}
    )
    unresolved_turns = expected_turns - resolved_turns
    retrieved = (
        row.get("retrieve", {}).get("result", {}).get("ranked", [])
        if row.get("retrieve", {}).get("status") == "ok"
        else []
    )
    stored_turns = set().union(*(_turn_tags(item) for item in stored_memories)) if stored_memories else set()
    retrieved_turns = set().union(*(_turn_tags(item) for item in retrieved)) if retrieved else set()
    retrieved_sessions = set().union(*(_session_tags(item) for item in retrieved)) if retrieved else set()

    exposures = []
    fully_exposed_turns: set[str] = set()
    content_exposed_turns: set[str] = set()
    for item in retrieved:
        node = item["node"]
        turn_ids = sorted(_turn_tags(item) & expected_turns)
        if not turn_ids:
            continue
        stored = next((m for m in stored_memories if m.get("id") == node.get("id")), None)
        content = node.get("content") or ""
        stored_content = (stored or {}).get("content") or ""
        if not content:
            status = "title_only"
        elif content == stored_content:
            status = "full_content"
            fully_exposed_turns.update(turn_ids)
        else:
            status = "truncated_content"
        if content:
            content_exposed_turns.update(turn_ids)
        exposures.append({
            "memory_id": node.get("id"),
            "turn_ids": turn_ids,
            "status": status,
            "exposed_title": node.get("title") or "",
            "exposed_content": content,
            "stored_content_chars": len(stored_content),
            "exposed_content_chars": len(content),
        })

    judge = row.get("judge", {})
    judged_wrong = judge.get("status") == "ok" and not judge.get("result", {}).get("correct")
    judged_correct = judge.get("status") == "ok" and judge.get("result", {}).get("correct")
    if judged_correct:
        failure_stage = "judge_marked_correct_not_adjudicated"
    elif unresolved_turns:
        failure_stage = "unknown_unresolved_source_evidence"
    elif not evidence:
        failure_stage = "unknown_needs_review" if judged_wrong else "not_classified"
    elif mode == "extract":
        failure_stage = "unresolvable_extracted_provenance" if judged_wrong else "not_classified"
    elif expected_turns - stored_turns:
        failure_stage = "storage_absence"
    elif expected_turns - retrieved_turns:
        failure_stage = "retrieval_miss"
    elif expected_turns - fully_exposed_turns:
        failure_stage = "exposure_loss_or_truncation"
    elif judged_wrong:
        failure_stage = "downstream_answer_or_judge_needs_review"
    else:
        failure_stage = "not_classified"

    retrieval_result = row.get("retrieve", {}).get("result", {})
    exposed_context = memory_context(retrieved)
    return {
        "question_id": row["question_id"],
        "dataset": row["dataset"],
        "question_type": row["question_type"],
        "mode": mode,
        "abstention": row["abstention"],
        "failure_stage": failure_stage,
        "limitations": [
            "Supporting-turn coverage is not proof that the exposed evidence is sufficient.",
            "Judge verdicts are model outputs; downstream cases require inspection.",
        ] + (["Extracted memories have session-only provenance; turn survival is unknown."]
             if mode == "extract" else []),
        "source": {
            "supporting_turns_labeled": bool(evidence),
            "evidence": evidence,
            "supporting_turn_ids": sorted(expected_turns),
            "supporting_session_ids": sorted(expected_sessions),
            "unresolved_turn_ids": sorted(unresolved_turns),
        },
        "store": {
            "status": row.get("store", {}).get("status", "missing"),
            "memory_count": len(stored_memories),
            "supporting_turn_coverage": _coverage(expected_turns, stored_turns)
            if mode == "raw" else None,
            "memories": stored_memories,
        },
        "retrieve": {
            "status": row.get("retrieve", {}).get("status", "missing"),
            "ranked_ids": [item["node"].get("id") for item in retrieved],
            "ranked": retrieved,
            "supporting_turn_coverage": _coverage(expected_turns, retrieved_turns)
            if mode == "raw" else None,
            "supporting_session_coverage": _coverage(expected_sessions, retrieved_sessions),
        },
        "exposure": {
            "context": exposed_context,
            "context_chars": len(exposed_context),
            "context_tokens_estimate": (len(exposed_context) + 3) // 4,
            "supporting_content_coverage": _coverage(expected_turns, content_exposed_turns)
            if mode == "raw" else None,
            "supporting_full_content_coverage": _coverage(expected_turns, fully_exposed_turns)
            if mode == "raw" else None,
            "supporting_memories": exposures,
            "reported_context_chars": retrieval_result.get("answer_context_chars"),
        },
        "answer": row.get("answer", {"status": "missing"}),
        "judge": judge or {"status": "missing"},
    }


def aggregate_diagnostics(items: list[dict]) -> dict:
    def recalls(path):
        values = []
        for item in items:
            value = item
            for part in path:
                value = value.get(part) if isinstance(value, dict) else None
            if isinstance(value, dict) and value.get("recall") is not None:
                values.append(value["recall"])
        return values

    def mean(values):
        return sum(values) / len(values) if values else None

    turn = recalls(("retrieve", "supporting_turn_coverage"))
    content = recalls(("exposure", "supporting_content_coverage"))
    full = recalls(("exposure", "supporting_full_content_coverage"))
    session = recalls(("retrieve", "supporting_session_coverage"))
    return {
        "questions": len(items),
        "failure_stage_counts": dict(sorted(Counter(i["failure_stage"] for i in items).items())),
        "supporting_turn_retrieval": {"questions": len(turn), "macro_recall": mean(turn)},
        "supporting_turn_content_exposure": {"questions": len(content), "macro_recall": mean(content)},
        "supporting_turn_full_exposure": {"questions": len(full), "macro_recall": mean(full)},
        "supporting_session_retrieval": {"questions": len(session), "macro_recall": mean(session)},
        "diagnostic_errors": sum(
            item[stage].get("status") == "error"
            for item in items
            for stage in ("store", "retrieve", "answer", "judge")
        ),
    }

"""Session-level LongMemEval and turn-level LoCoMo retrieval metrics."""

from __future__ import annotations

import math
import statistics


def retrieval_metrics(question, retrieval, k, mode="raw", gated=False):
    if (question["dataset"] == "longmemeval" and question["abstention"]) or not question[
        "gold_ids"
    ]:
        return {"recall": None, "ndcg": None}
    if question["dataset"] == "locomo" and mode == "extract":
        return {"recall": None, "ndcg": None}
    prefix = "session:" if question["dataset"] == "longmemeval" else "turn:"
    ranked = []
    for item in retrieval.get("ranked", [])[:k]:
        if gated and item["score"] < retrieval["production_gate"]:
            continue
        tags = item["node"].get("tags") or []
        if isinstance(tags, str):
            import json

            tags = json.loads(tags)
        for tag in tags:
            if tag.startswith(prefix) and tag[len(prefix) :] not in ranked:
                ranked.append(tag[len(prefix) :])
    gold = set(question["gold_ids"])
    recall = len(gold.intersection(ranked)) / len(gold)
    dcg = sum(1 / math.log2(i + 2) for i, value in enumerate(ranked) if value in gold)
    ideal = sum(1 / math.log2(i + 2) for i in range(min(k, len(gold))))
    return {"recall": recall, "ndcg": dcg / ideal if ideal else None}


def mean(values):
    values = [v for v in values if v is not None]
    return statistics.mean(values) if values else None


def percentile(values, quantile):
    if not values:
        return None
    values = sorted(values)
    pos = (len(values) - 1) * quantile
    low, high = math.floor(pos), math.ceil(pos)
    return values[low] + (values[high] - values[low]) * (pos - low)


def aggregate(rows, k, mode, strategy="recall"):
    metrics, gates, scores, abstentions, latencies = [], [], [], [], []
    counts, context_chars, whisper_chars, silent_abstentions = [], [], [], []
    for row in rows:
        if row.get("retrieve", {}).get("status") == "ok":
            retrieval = row["retrieve"]["result"]
            limit = max(1, len(retrieval["ranked"])) if strategy == "whisper" else k
            metrics.append(retrieval_metrics(row, retrieval, limit, mode))
            if strategy == "recall":
                gates.append(retrieval_metrics(row, retrieval, k, mode, gated=True))
            counts.append(len(retrieval["ranked"]))
            context_chars.append(retrieval.get("answer_context_chars"))
            if strategy == "whisper":
                whisper_chars.append(retrieval["whisper_context_chars"])
                if row["dataset"] == "longmemeval" and row["abstention"]:
                    silent_abstentions.append(int(retrieval["silent"]))
            latencies.append(retrieval["latency_s"])
        if row.get("judge", {}).get("status") == "ok":
            value = int(row["judge"]["result"]["correct"])
            if row["abstention"]:
                abstentions.append(value)
            if row["dataset"] == "longmemeval" or not row["abstention"]:
                scores.append(value)
    result = {
        "questions": len(rows),
        "retrieved": len(metrics),
        "scored": len(scores),
        "accuracy": mean(scores),
        "abstention_questions": sum(row["abstention"] for row in rows),
        "abstention_scored": len(abstentions),
        "abstention_accuracy": mean(abstentions),
        "injection_rate": mean([int(n > 0) for n in counts]),
        "mean_injected_memories": mean(counts),
        "mean_answer_context_chars": mean(context_chars),
        "retrieval_labeled": sum(m["recall"] is not None for m in metrics),
        "retrieval_p50_s": percentile(latencies, 0.5),
        "retrieval_p95_s": percentile(latencies, 0.95),
        "errors": {
            phase: sum(r.get(phase, {}).get("status") == "error" for r in rows)
            for phase in ("store", "retrieve", "answer", "judge")
        },
    }

    if strategy == "whisper":
        result.update({
            "recall@whisper": mean([m["recall"] for m in metrics]),
            "mean_whisper_context_chars": mean(whisper_chars),
            "abstention_retrieved": len(silent_abstentions),
            "abstention_silence_rate": mean(silent_abstentions),
        })
    else:
        result.update({
            f"recall@{k}": mean([m["recall"] for m in metrics]),
            f"ndcg@{k}": mean([m["ndcg"] for m in metrics]),
            f"production_recall@{k}": mean([m["recall"] for m in gates]),
        })
    return result

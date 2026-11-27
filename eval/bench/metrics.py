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


def aggregate(rows, k, mode):
    metrics, gates, scores, abstentions, latencies = [], [], [], [], []
    for row in rows:
        if row.get("retrieve", {}).get("status") == "ok":
            retrieval = row["retrieve"]["result"]
            metrics.append(retrieval_metrics(row, retrieval, k, mode))
            gates.append(retrieval_metrics(row, retrieval, k, mode, gated=True))
            latencies.append(retrieval["latency_s"])
        if row.get("judge", {}).get("status") == "ok":
            value = int(row["judge"]["result"]["correct"])
            if row["abstention"]:
                abstentions.append(value)
            if row["dataset"] == "longmemeval" or not row["abstention"]:
                scores.append(value)
    return {
        "questions": len(rows),
        "retrieved": len(metrics),
        "scored": len(scores),
        "accuracy": mean(scores),
        "abstention_questions": sum(row["abstention"] for row in rows),
        "abstention_scored": len(abstentions),
        "abstention_accuracy": mean(abstentions),
        f"recall@{k}": mean([m["recall"] for m in metrics]),
        f"ndcg@{k}": mean([m["ndcg"] for m in metrics]),
        "retrieval_labeled": sum(m["recall"] is not None for m in metrics),
        f"production_recall@{k}": mean([m["recall"] for m in gates]),
        "retrieval_p50_s": percentile(latencies, 0.5),
        "retrieval_p95_s": percentile(latencies, 0.95),
        "errors": {
            phase: sum(r.get(phase, {}).get("status") == "error" for r in rows)
            for phase in ("store", "retrieve", "answer", "judge")
        },
    }

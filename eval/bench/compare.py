"""Paired benchmark comparisons with deterministic bootstrap intervals."""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path
from statistics import mean


def _latest(path: Path) -> dict[str, dict]:
    rows = {}
    for line in path.read_text().splitlines():
        if line.strip():
            row = json.loads(line)
            rows[row["question_id"]] = row
    return rows


def _correct(row: dict) -> bool:
    judge = row.get("judge", {})
    value = judge.get("result", {}).get("correct") if judge.get("status") == "ok" else None
    if not isinstance(value, bool):
        raise ValueError(f"Question {row['question_id']} has no completed boolean judge result")
    return value


def _primary(row: dict) -> bool:
    return row["dataset"] != "locomo" or row["question_type"] != "5"


def _abstention(row: dict) -> bool:
    return bool(row.get("abstention")) or (
        row["dataset"] == "locomo" and row["question_type"] == "5"
    )


def _metric(row: dict, name: str, default=0):
    return row.get("retrieve", {}).get("result", {}).get(name, default)


def _paired(rows: list[tuple[dict, dict]]) -> dict:
    pairs = [(_correct(base), _correct(candidate)) for base, candidate in rows]
    return {
        "questions": len(rows),
        "baseline_correct": sum(base for base, _ in pairs),
        "candidate_correct": sum(candidate for _, candidate in pairs),
        "wins": sum(not base and candidate for base, candidate in pairs),
        "losses": sum(base and not candidate for base, candidate in pairs),
        "ties": sum(base == candidate for base, candidate in pairs),
        "accuracy_delta": mean(candidate - base for base, candidate in pairs) if pairs else None,
    }


def _bootstrap_interval(
    rows: list[tuple[dict, dict]], *, clustered: bool, seed: int, samples: int
) -> list[float]:
    groups = defaultdict(list)
    for index, (base, candidate) in enumerate(rows):
        unit = base.get("conversation_id") if clustered else index
        groups[unit].append((_correct(base), _correct(candidate)))
    units = list(groups)
    rng = random.Random(seed)
    deltas = []
    for _ in range(samples):
        selected = [rng.choice(units) for _ in units]
        pairs = [pair for unit in selected for pair in groups[unit]]
        deltas.append(mean(candidate - base for base, candidate in pairs))
    deltas.sort()
    return [
        deltas[int(0.025 * (samples - 1))],
        deltas[int(0.975 * (samples - 1))],
    ]


def compare_runs(
    baseline: Path, candidate: Path, *, seed: int = 306, samples: int = 10000
) -> dict:
    base_rows = _latest(baseline / "questions.jsonl")
    candidate_rows = _latest(candidate / "questions.jsonl")
    if set(base_rows) != set(candidate_rows):
        raise ValueError("Runs do not contain identical question IDs")
    rows = [(base_rows[qid], candidate_rows[qid]) for qid in sorted(base_rows)]
    datasets = {base["dataset"] for base, _ in rows}
    if len(datasets) != 1 or any(base["dataset"] != cand["dataset"] for base, cand in rows):
        raise ValueError("Runs do not contain the same single dataset")
    dataset = datasets.pop()
    primary = [(base, cand) for base, cand in rows if _primary(base)]
    abstentions = [(base, cand) for base, cand in rows if _abstention(base)]
    return {
        "baseline_run": baseline.name,
        "candidate_run": candidate.name,
        "dataset": dataset,
        "all": _paired(rows),
        "primary": _paired(primary),
        "abstentions": _paired(abstentions),
        "by_stratum": {
            stratum: _paired(
                [(base, cand) for base, cand in rows if base["question_type"] == stratum]
            )
            for stratum in sorted({base["question_type"] for base, _ in rows})
        },
        "context": {
            "baseline_mean_answer_chars": mean(
                _metric(base, "answer_context_chars") for base, _ in rows
            ),
            "candidate_mean_answer_chars": mean(
                _metric(cand, "answer_context_chars") for _, cand in rows
            ),
            "baseline_mean_injected": mean(
                _metric(base, "injected_count") for base, _ in rows
            ),
            "candidate_mean_injected": mean(
                _metric(cand, "injected_count") for _, cand in rows
            ),
            "baseline_injection_rate": mean(
                not _metric(base, "silent", True) for base, _ in rows
            ),
            "candidate_injection_rate": mean(
                not _metric(cand, "silent", True) for _, cand in rows
            ),
            "baseline_mean_retrieval_s": mean(
                _metric(base, "latency_s") for base, _ in rows
            ),
            "candidate_mean_retrieval_s": mean(
                _metric(cand, "latency_s") for _, cand in rows
            ),
        },
        "uncertainty": {
            "method": (
                "conversation-cluster bootstrap" if dataset == "locomo" else "question bootstrap"
            ),
            "seed": seed,
            "samples": samples,
            "primary_accuracy_delta_95pct": _bootstrap_interval(
                primary, clustered=dataset == "locomo", seed=seed, samples=samples
            ),
        },
    }


def _run_path(value: str) -> Path:
    path = Path(value)
    return path if path.exists() else Path(__file__).parent / "artifacts" / value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline")
    parser.add_argument("candidate")
    parser.add_argument("--seed", type=int, default=306)
    parser.add_argument("--samples", type=int, default=10000)
    args = parser.parse_args()
    print(
        json.dumps(
            compare_runs(
                _run_path(args.baseline),
                _run_path(args.candidate),
                seed=args.seed,
                samples=args.samples,
            ),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()

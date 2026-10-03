"""Deterministic, outcome-blind development/held-out benchmark selection."""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

from eval.bench.datasets import SOURCES, load_questions, sha256_file


LME_TYPES = (
    "single-session-user",
    "single-session-assistant",
    "single-session-preference",
    "multi-session",
    "temporal-reasoning",
    "knowledge-update",
)


def _shuffle(values, rng):
    values = sorted(values)
    rng.shuffle(values)
    return values


def _locomo_order(questions, category: str, rng) -> list[str]:
    groups = defaultdict(list)
    for question in questions:
        if question.question_type == category:
            groups[question.conversation_id].append(question.question_id)
    for conversation in groups:
        groups[conversation] = _shuffle(groups[conversation], rng)
    conversations = _shuffle(groups, rng)
    ordered = []
    offset = 0
    while len(ordered) < 16:
        progress = False
        for conversation in conversations:
            if offset < len(groups[conversation]):
                ordered.append(groups[conversation][offset])
                progress = True
                if len(ordered) == 16:
                    break
        if not progress:
            raise ValueError(f"LoCoMo category {category} has fewer than 16 questions")
        offset += 1
    return ordered


def build_split_manifest(data_dir: Path, seed: int = 20261003) -> dict:
    rng = random.Random(seed)
    lme_path = data_dir / SOURCES["longmemeval"][0]
    locomo_path = data_dir / SOURCES["locomo"][0]
    lme = list(load_questions(lme_path, "longmemeval"))
    locomo = list(load_questions(locomo_path, "locomo"))
    selected = {
        "longmemeval": {"development": [], "heldout": []},
        "locomo": {"development": [], "heldout": []},
    }
    for question_type in LME_TYPES:
        answerable = _shuffle(
            [q.question_id for q in lme if q.question_type == question_type and not q.abstention],
            rng,
        )
        if len(answerable) < 16:
            raise ValueError(f"LongMemEval {question_type} has fewer than 16 answerable questions")
        selected["longmemeval"]["development"].extend(answerable[:8])
        selected["longmemeval"]["heldout"].extend(answerable[8:16])

    abstention_types = [
        question_type
        for question_type in LME_TYPES
        if any(q.abstention and q.question_type == question_type for q in lme)
    ]
    if len(abstention_types) != 4:
        raise ValueError(f"Expected four LongMemEval abstention strata, got {abstention_types}")
    for question_type in abstention_types:
        abstentions = _shuffle(
            [q.question_id for q in lme if q.question_type == question_type and q.abstention], rng
        )
        if len(abstentions) < 6:
            raise ValueError(f"LongMemEval {question_type} has fewer than six abstentions")
        selected["longmemeval"]["development"].extend(abstentions[:3])
        selected["longmemeval"]["heldout"].extend(abstentions[3:6])

    for category in map(str, range(1, 6)):
        ordered = _locomo_order(locomo, category, rng)
        selected["locomo"]["development"].extend(ordered[:8])
        selected["locomo"]["heldout"].extend(ordered[8:16])

    by_id = {q.question_id: q for q in [*lme, *locomo]}
    datasets = {}
    for dataset, path in (("longmemeval", lme_path), ("locomo", locomo_path)):
        splits = {}
        for name, ids in selected[dataset].items():
            strata = Counter(
                (
                    "abstention"
                    if dataset == "longmemeval" and by_id[qid].abstention
                    else by_id[qid].question_type
                )
                for qid in ids
            )
            conversations = Counter(by_id[qid].conversation_id for qid in ids)
            splits[name] = {
                "question_ids": ids,
                "count": len(ids),
                "strata": dict(sorted(strata.items())),
                **(
                    {"conversation_counts": dict(sorted(conversations.items()))}
                    if dataset == "locomo"
                    else {}
                ),
            }
        datasets[dataset] = {
            "file": path.name,
            "sha256": sha256_file(path),
            "bytes": path.stat().st_size,
            "splits": splits,
        }
    return {
        "version": 1,
        "name": "ormah-diagnostic-v1",
        "seed": seed,
        "selection": (
            "Outcome-blind deterministic shuffle within strata. LongMemEval: eight answerable "
            "questions per six types plus three abstentions per each of four available types, "
            "per split. LoCoMo: eight per category, round-robin across conversations before reuse."
        ),
        "datasets": datasets,
    }


def load_split(path: Path, dataset: str, split: str, dataset_path: Path) -> tuple[list[str], dict]:
    manifest = json.loads(path.read_text())
    try:
        section = manifest["datasets"][dataset]
        ids = section["splits"][split]["question_ids"]
    except KeyError as exc:
        raise ValueError(f"Split manifest has no {dataset}/{split}") from exc
    actual = sha256_file(dataset_path)
    if section["sha256"] != actual:
        raise ValueError(
            f"Split manifest dataset checksum differs for {dataset}: "
            f"expected {section['sha256']}, got {actual}"
        )
    if len(ids) != len(set(ids)):
        raise ValueError(f"Split manifest {dataset}/{split} contains duplicate question IDs")
    return ids, manifest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20261003)
    args = parser.parse_args()
    print(json.dumps(build_split_manifest(args.data_dir, args.seed), indent=2))


if __name__ == "__main__":
    main()

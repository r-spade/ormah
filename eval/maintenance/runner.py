"""Single-vs-batched A/B runner for the maintenance auto-link eval (#87).

Runs the SAME mined pairs through the single-pair path and the batched path and
records each pair's relationship verdict, so report.agreement can compare them.
Uses a bare Settings() so the run picks up the machine's real LLM provider.
"""
from __future__ import annotations

import json

from ormah.background import auto_linker
from ormah.background.llm import normalize_link_type, pair_batch
from ormah.config import Settings


def _load_pairs(pairs_path: str) -> list[dict]:
    pairs = []
    with open(pairs_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                pairs.append(json.loads(line))
    return pairs


def _to_link_pair(raw: dict) -> dict:
    """Adapt a mined pair to the {node, other, match_id} shape the linker uses."""
    a, b = raw["node_a"], raw["node_b"]
    return {"node": a, "other": b, "match_id": b["id"],
            "similarity": raw.get("similarity", 0.0)}


def _relationship(verdict: dict | None) -> str:
    """Normalize a link verdict to a relationship label, 'error' when unusable."""
    if verdict is None:
        return "error"
    raw_rel = verdict.get("relationship", "error")
    return "error" if raw_rel == "error" else normalize_link_type(raw_rel)


def run(pairs_path: str, mode: str, k: int, out_path: str) -> None:
    settings = Settings()
    raw_pairs = _load_pairs(pairs_path)
    pairs = [_to_link_pair(r) for r in raw_pairs]
    verdicts: dict[str, str] = {}

    if mode == "single":
        for raw, pair in zip(raw_pairs, pairs):
            result = auto_linker._llm_classify_link(settings, pair["node"], pair["other"])
            verdicts[raw["pair_id"]] = _relationship(result)
    else:
        settings.maintenance_pairs_per_call = k
        results = pair_batch.judge_pairs(
            settings, auto_linker._LLM_LINK_INSTRUCTIONS, pairs,
            auto_linker._render_link_pair,
            judge_single=lambda p: auto_linker._llm_classify_link(
                settings, p["node"], p["other"]),
            k=k,
        )
        for raw, result in zip(raw_pairs, results):
            verdicts[raw["pair_id"]] = _relationship(result)

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(verdicts, f, indent=2)

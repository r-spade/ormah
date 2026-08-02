"""Agreement metrics for the single-vs-batched maintenance eval (#87 gate).

Gate (issue #87): before raising K anywhere, batched verdicts must agree with
single-call verdicts on the mined pairs — overall agreement >= 0.90 and
"none"->edge flips <= 0.10. Passing authorizes raising ONE job's per-call K,
and only for the schema whose corpus was mined (council C3).
"""
from __future__ import annotations

GATE_MIN_AGREEMENT = 0.90
GATE_MAX_NONE_TO_EDGE = 0.10


def agreement(single: dict[str, str], batched: dict[str, str]) -> dict:
    keys = sorted(set(single) & set(batched))
    n = len(keys)
    agree = sum(1 for k in keys if single[k] == batched[k])
    none_keys = [k for k in keys if single[k] == "none"]
    none_to_edge = sum(1 for k in none_keys if batched[k] not in ("none", "error"))
    flips: dict[str, int] = {}
    for k in keys:
        if single[k] != batched[k]:
            flips[f"{single[k]}->{batched[k]}"] = flips.get(f"{single[k]}->{batched[k]}", 0) + 1
    agree_rate = agree / n if n else 0.0
    none_to_edge_rate = none_to_edge / len(none_keys) if none_keys else 0.0
    return {
        "n": n, "agree_rate": round(agree_rate, 3),
        "none_to_edge_rate": round(none_to_edge_rate, 3), "flips": flips,
        "gate_pass": agree_rate >= GATE_MIN_AGREEMENT
                     and none_to_edge_rate <= GATE_MAX_NONE_TO_EDGE,
    }

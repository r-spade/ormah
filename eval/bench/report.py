"""Reports are regenerated from journals; failures never silently become passes."""

from __future__ import annotations

import json
from pathlib import Path

from eval.bench.artifacts import Journal, write_json
from eval.bench.metrics import aggregate, percentile


def build_report(run_dir: Path):
    manifest = json.loads((run_dir / "manifest.json").read_text())
    rows = list(Journal(run_dir / "questions.jsonl").latest().values())
    k, mode = manifest["parameters"]["k"], manifest["parameters"]["mode"]
    if manifest["parameters"]["dataset"] == "longmemeval":
        from eval.bench.judge import LONGMEMEVAL_RULES

        types = sorted(LONGMEMEVAL_RULES)
    else:
        types = [str(i) for i in range(1, 6)]
    summary = {
        "manifest": manifest,
        "overall": aggregate(rows, k, mode),
        "by_type": {
            t: aggregate([r for r in rows if r["question_type"] == t], k, mode) for t in types
        },
        "calls": {},
    }
    calls = Journal(run_dir / "calls.jsonl").rows
    for phase in ("extract", "answer", "judge"):
        group = [c for c in calls if c["phase"] == phase]
        latencies = [c["latency_s"] for c in group]
        summary["calls"][phase] = {
            "calls": len(group),
            "errors": sum(bool(c.get("error")) for c in group),
            "usage_unknown": sum(c.get("usage") is None for c in group),
            "input_tokens": sum(((c.get("usage") or {}).get("input_tokens") or 0) for c in group),
            "output_tokens": sum(((c.get("usage") or {}).get("output_tokens") or 0) for c in group),
            "cache_read_input_tokens": sum(
                ((c.get("usage") or {}).get("cache_read_input_tokens") or 0) for c in group
            ),
            "cache_creation_input_tokens": sum(
                ((c.get("usage") or {}).get("cache_creation_input_tokens") or 0) for c in group
            ),
            "prompt_tokens_estimate": sum(c.get("prompt_tokens_estimate", 0) for c in group),
            "usd": sum(c.get("usd", 0) for c in group),
            "subscription_usd_estimate": sum(c.get("estimated_usd") or 0 for c in group),
            "latency_p50_s": percentile(latencies, 0.5),
            "latency_p95_s": percentile(latencies, 0.95),
            "latency_total_s": sum(latencies),
        }
    summary["wall_s"] = sum(manifest.get("phase_wall_s", {}).values())
    write_json(run_dir / "summary.json", summary)
    return summary


def format_report(summary):
    manifest = summary["manifest"]
    k = manifest["parameters"]["k"]
    lines = [
        f"Run {manifest['run_id']} ({manifest['parameters']['dataset']}, "
        f"{manifest['parameters']['mode']})",
        f"{'Group':28} {'N':>5} {'Scored':>7} {'Accuracy':>9} {'Recall':>9} {'nDCG':>9}",
    ]
    for name, metrics in [("overall", summary["overall"]), *summary["by_type"].items()]:
        values = [metrics.get(key) for key in ("accuracy", f"recall@{k}", f"ndcg@{k}")]
        rendered = [f"{v:.4f}" if v is not None else "N/A" for v in values]
        lines.append(
            f"{name:28} {metrics['questions']:5} {metrics['scored']:7} "
            + " ".join(f"{v:>9}" for v in rendered)
        )
    lines.append(
        f"Retrieval p50/p95: {summary['overall']['retrieval_p50_s']} / "
        f"{summary['overall']['retrieval_p95_s']} s; wall {summary['wall_s']:.1f} s"
    )
    lines.append(
        f"Errors: {summary['overall']['errors']}; "
        f"abstention accuracy: {summary['overall']['abstention_accuracy']}"
    )
    return "\n".join(lines)

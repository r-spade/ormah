"""Source-only CLI handlers for public benchmarks."""

from __future__ import annotations

import json
import re
import sys

from eval.bench.runner import BASE


def cmd_eval_bench(args):
    from eval.bench.datasets import download
    from eval.bench.report import build_report, format_report
    from eval.bench.runner import run

    try:
        if args.eval_bench_cmd == "download":
            print(json.dumps(download(BASE / "data", args.dataset), indent=2))
            return
        if args.eval_bench_cmd == "report":
            if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", args.run_id):
                raise ValueError("Invalid run id")
            summary = build_report(BASE / "artifacts" / args.run_id)
            print(json.dumps(summary, indent=2) if args.json else format_report(summary))
            return
        summary = run(args)
        print(format_report(summary) if "overall" in summary else json.dumps(summary))
        if summary.get("errors") or (
            "overall" in summary and any(summary["overall"]["errors"].values())
        ):
            sys.exit(1)
    except (ValueError, RuntimeError, OSError) as exc:
        print(f"Benchmark error: {exc}", file=sys.stderr)
        sys.exit(1)

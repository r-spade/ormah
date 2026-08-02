"""CLI: python -m eval.maintenance.cli {mine|run|report} ...

Local A/B eval gate for #87. Mines auto-link candidate pairs from a production
store (read-only), runs them through the single and batched paths, and reports
whether the batched verdicts agree with the single ones (gate for raising K).
Only the aggregate report is safe to share — never the mined pairs.jsonl.
"""
import argparse
import json
import sys

from eval.maintenance import miner, report, runner


def main() -> int:
    p = argparse.ArgumentParser(prog="eval.maintenance")
    sub = p.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("mine")
    m.add_argument("--db", required=True)
    m.add_argument("--out", default="eval/maintenance/corpus/pairs.jsonl")
    m.add_argument("-n", type=int, default=100)
    r = sub.add_parser("run")
    r.add_argument("--pairs", required=True)
    r.add_argument("--mode", choices=["single", "batched"], required=True)
    r.add_argument("--k", type=int, default=10)
    r.add_argument("--out", required=True)
    g = sub.add_parser("report")
    g.add_argument("--single", required=True)
    g.add_argument("--batched", required=True)
    a = p.parse_args()
    if a.cmd == "mine":
        print(f"mined {miner.mine_pairs(a.db, a.out, a.n)} pairs -> {a.out}")
    elif a.cmd == "run":
        runner.run(a.pairs, a.mode, a.k, a.out)
    else:
        res = report.agreement(json.load(open(a.single)), json.load(open(a.batched)))
        print(json.dumps(res, indent=2))
        return 0 if res["gate_pass"] else 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

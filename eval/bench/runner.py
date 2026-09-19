"""Resumable pipeline, serial isolated retrieval and bounded parallel CLI calls."""

from __future__ import annotations

import importlib.metadata
import json
import math
import os
import re
import resource
import sys
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

from eval.bench.artifacts import Journal, write_json
from eval.bench.cost import Budget, BudgetExceeded, token_cost
from eval.bench.datasets import SOURCES, load_questions, sha256_file
from eval.bench.providers import make_provider
from eval.bench.report import build_report
from eval.bench.store import (
    ClaudeCLIAdapter,
    CodexAdapter,
    EmbeddingCache,
    ProviderAdapter,
    digest,
    prepare_memories,
    seed_memories,
)
from eval.settings import RETRIEVAL_EVAL_SETTINGS_OVERRIDES

BASE = Path(__file__).parent
BENCH_SETTINGS_OVERRIDES = {
    **RETRIEVAL_EVAL_SETTINGS_OVERRIDES,
    "embedding_max_content_chars": 512,
    "ingest_max_content_chars": 100000,
}


def phases_for(args):
    if args.phase == "free":
        phases = {"store", "retrieve", "report"}
        if args.mode == "extract":
            raise ValueError(
                "extract mode uses an LLM: explicitly name --phase all or --phase store"
            )
    elif args.phase == "all":
        phases = {"store", "retrieve", "answer", "judge", "report"}
    else:
        phases = set(args.phase.split(","))
        if not phases <= {"store", "retrieve", "answer", "judge", "report"}:
            raise ValueError("Unknown --phase; use store,retrieve,answer,judge,report or all")
    return phases


def make_engine(db_dir):
    from ormah.config import Settings
    from ormah.engine.memory_engine import MemoryEngine

    (db_dir / "nodes").mkdir(parents=True, exist_ok=True)
    settings = Settings(memory_dir=db_dir, **BENCH_SETTINGS_OVERRIDES)
    engine = MemoryEngine(settings)
    engine.startup()
    return engine


def selected_questions(args, path):
    count = 0
    for q in load_questions(path, args.dataset):
        if args.question_type and q.question_type != args.question_type:
            continue
        if args.category and q.question_type != str(args.category):
            continue
        if args.conversation is not None and q.conversation_id != str(args.conversation):
            continue
        yield q
        count += 1
        if args.limit and count >= args.limit:
            return


def parameters(args):
    return {
        k: getattr(args, k)
        for k in (
            "dataset",
            "mode",
            "k",
            "limit",
            "question_type",
            "category",
            "conversation",
            "extract_provider",
            "extract_model",
            "answer_provider",
            "answer_model",
            "judge_provider",
            "judge_model",
        )
    }


def validate(args):
    if args.k <= 0 or args.workers <= 0 or (args.limit is not None and args.limit <= 0):
        raise ValueError("--k, --workers and --limit must be positive")
    if args.conversation is not None and args.conversation < 0:
        raise ValueError("--conversation must be nonnegative")
    if not math.isfinite(args.max_usd) or args.max_usd <= 0:
        raise ValueError("--max-usd must be finite and positive")
    if args.dataset == "longmemeval" and (args.category or args.conversation is not None):
        raise ValueError("--category and --conversation apply only to LoCoMo")
    if args.dataset == "locomo" and args.question_type:
        raise ValueError("--question-type applies only to LongMemEval")
    if args.run_id and not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", args.run_id):
        raise ValueError("Invalid run id")


def run(args, *, base=BASE, engine_factory=make_engine, provider_factory=make_provider):
    invocation_start = time.perf_counter()
    validate(args)
    phases = phases_for(args)
    active = [p for p in ("answer", "judge") if p in phases]
    if args.mode == "extract" and "store" in phases:
        active.insert(0, "extract")
    # Preflight before creating an engine, downloading, or invoking any provider.
    count_estimate = args.limit or (1986 if args.dataset == "locomo" else 500)
    for phase in active:
        provider = getattr(args, phase + "_provider")
        model = getattr(args, phase + "_model") or (
            "claude-haiku-4-5"
            if provider == "anthropic"
            else "sonnet"
            if provider == "claude-cli"
            else "default"
        )
        if provider == "anthropic":
            estimate = token_cost(
                model,
                {
                    "input_tokens": (
                        (600_000 if args.dataset == "locomo" else 79_000_000)
                        if phase == "extract"
                        else count_estimate * 6000
                    ),
                    "output_tokens": (
                        (272 if args.dataset == "locomo" else 19_829) * 4096
                        if phase == "extract"
                        else count_estimate * (256 if phase == "judge" else 1024)
                    ),
                },
            )
            print(
                f"{phase}: API estimate ${estimate:.4f} "
                "(rough token allowance; extraction estimate covers the full dataset)",
                flush=True,
            )
            if not os.environ.get("ANTHROPIC_API_KEY"):
                raise ValueError("anthropic requires ANTHROPIC_API_KEY; no API call made")
        else:
            print(f"{phase}: {provider}/{model}, subscription billed; $0 API spend", flush=True)
    if args.resume and not args.run_id:
        raise ValueError("--resume requires --run-id")
    run_id = (
        args.run_id or f"{args.dataset}-{args.mode}-{datetime.now(timezone.utc):%Y%m%dT%H%M%S%f}"
    )
    run_dir = base / "artifacts" / run_id
    manifest_path = run_dir / "manifest.json"
    path = base / "data" / SOURCES[args.dataset][0]
    if not path.exists():
        raise ValueError(f"Dataset missing: {path}; run 'ormah eval bench download'")
    fingerprint = sha256_file(path)
    if manifest_path.exists():
        if not args.resume:
            raise ValueError("Run exists; use --resume or a new --run-id")
        manifest = json.loads(manifest_path.read_text())
        if manifest["parameters"] != parameters(args) or manifest["dataset_sha256"] != fingerprint:
            raise ValueError("Resume parameters/dataset differ from the saved manifest")
    else:
        manifest = {
            "run_id": run_id,
            "parameters": parameters(args),
            "dataset_sha256": fingerprint,
            "dataset_bytes": path.stat().st_size,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "ormah_version": importlib.metadata.version("ormah"),
            "git_sha": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
            "git_dirty": bool(subprocess.check_output(["git", "status", "--porcelain"], text=True)),
            "settings": BENCH_SETTINGS_OVERRIDES,
            "embedding_model": RETRIEVAL_EVAL_SETTINGS_OVERRIDES["embedding_model"],
            "embedding_batch_size": 16,
            "runtime": {
                "python": __import__("platform").python_version(),
                "sqlite": __import__("sqlite3").sqlite_version,
                "sqlite_vec": importlib.metadata.version("sqlite-vec"),
                "fastembed": importlib.metadata.version("fastembed"),
            },
            "prompt_hashes": {name: sha256_file(BASE / name) for name in ("answer.py", "judge.py")},
            "phase_wall_s": {},
            "providers": {},
            "invocations": [],
        }
    manifest["invocations"].append(
        {
            "phases": sorted(phases),
            "workers": args.workers,
            "max_usd": args.max_usd,
            "resume": args.resume,
            "embedding_batch_size": 16,
            "embedding_length_sorted": True,
        }
    )
    write_json(manifest_path, manifest)
    print(f"Run id: {run_id}", flush=True)
    journal = Journal(run_dir / "questions.jsonl")
    rows = journal.latest()
    ledger = Journal(run_dir / "calls.jsonl")
    budget = Budget(args.max_usd, sum(c.get("usd", 0) for c in ledger.rows))
    providers = {}
    for phase in active:
        provider = provider_factory(
            getattr(args, phase + "_provider"),
            getattr(args, phase + "_model"),
            ledger=ledger,
            phase=phase,
            budget=budget,
        )
        providers[phase] = provider
        manifest["providers"][phase] = {
            "provider": provider.name,
            "model": provider.model,
            "version": provider.version(),
        }
    write_json(manifest_path, manifest)

    def save(row):
        journal.append(row)
        rows[row["question_id"]] = row

    def elapsed(phase, start):
        times = manifest["phase_wall_s"]
        times[phase] = times.get(phase, 0) + time.perf_counter() - start
        write_json(manifest_path, manifest)

    if phases & {"store", "retrieve"}:
        from ormah.background.llm_client import reset_adapter, set_adapter
        from eval.bench.retrieve import retrieve_question

        engine = engine_factory(base / "eval_db" / run_id)
        cache = EmbeddingCache(
            base / "artifacts" / "cache" / "embeddings.sqlite",
            f"{engine.settings.embedding_provider}:{engine.settings.embedding_model}:"
            f"{engine.settings.embedding_dim}",
        )
        adapter = None
        if "extract" in providers:
            cls = {"claude-cli": ClaudeCLIAdapter, "codex": CodexAdapter}.get(
                providers["extract"].name, ProviderAdapter
            )
            adapter = cls(providers["extract"])
            set_adapter(adapter)
        seeded = None
        failed_haystacks = {}
        try:
            for index, q in enumerate(selected_questions(args, path)):
                row = rows.get(q.question_id, q.metadata())
                needed = {
                    p
                    for p in phases & {"store", "retrieve"}
                    if row.get(p, {}).get("status") != "ok"
                }
                if not needed:
                    continue
                haystack_id = q.conversation_id if q.dataset == "locomo" else q.question_id
                haystack_path = run_dir / "haystacks" / f"{digest(haystack_id)}.json"
                start = time.perf_counter()
                try:
                    if haystack_id in failed_haystacks:
                        raise RuntimeError(failed_haystacks[haystack_id])
                    if seeded != haystack_id:
                        if haystack_path.exists():
                            memories = json.loads(haystack_path.read_text())
                        elif "store" in phases:
                            memories = prepare_memories(
                                engine,
                                q,
                                args.mode,
                                base / "artifacts" / "cache" / "extractions",
                                adapter,
                                args.workers,
                            )
                            write_json(haystack_path, memories)
                        else:
                            raise ValueError("Missing stored haystack: run --phase store first")
                        seed_memories(engine, memories, cache)
                        memory_count = len(memories)
                        del memories
                        seeded = haystack_id
                    if "store" in phases:
                        row["store"] = {
                            "status": "ok",
                            "result": {
                                "memories": memory_count,
                                "haystack": str(haystack_path.relative_to(run_dir)),
                            },
                        }
                        save(dict(row))
                    elapsed("store", start)
                except Exception as exc:
                    failed_haystacks[haystack_id] = str(exc)
                    row["store"] = {"status": "error", "error": str(exc)}
                    save(dict(row))
                    elapsed("store", start)
                    if isinstance(exc, BudgetExceeded) or budget.spent >= budget.max_usd:
                        raise BudgetExceeded(str(exc)) from exc
                    continue
                if "retrieve" in needed:
                    start = time.perf_counter()
                    try:
                        row["retrieve"] = {
                            "status": "ok",
                            "result": retrieve_question(engine, q.question, args.k),
                        }
                    except Exception as exc:
                        row["retrieve"] = {"status": "error", "error": str(exc)}
                    save(dict(row))
                    elapsed("retrieve", start)
                if index % 10 == 0:
                    print(f"{index + 1} questions stored/retrieved", flush=True)
                del q, row
        finally:
            previous = manifest.setdefault("embedding_cache", {"hits": 0, "misses": 0})
            previous["hits"] += cache.hits
            previous["misses"] += cache.misses
            write_json(manifest_path, manifest)
            cache.close()
            engine.shutdown()
            reset_adapter()
    if not rows:
        raise ValueError("No questions selected or no completed store phase")
    for phase, dependency in (("answer", "retrieve"), ("judge", "answer")):
        if phase not in phases:
            continue
        from eval.bench.answer import answer_question
        from eval.bench.judge import judge_answer

        start = time.perf_counter()

        def process(row):
            row = dict(row)
            if row.get(dependency, {}).get("status") != "ok":
                row[phase] = {"status": "error", "error": f"Missing successful {dependency}"}
                return row
            try:
                if phase == "answer":
                    result = answer_question(
                        providers[phase], row, row["retrieve"]["result"]["ranked"]
                    )
                else:
                    result = judge_answer(providers[phase], row, row["answer"]["result"]["text"])
                row[phase] = {"status": "ok", "result": result}
            except BudgetExceeded as exc:
                row[phase] = {"status": "error", "error": str(exc), "budget_exceeded": True}
            except Exception as exc:
                row[phase] = {"status": "error", "error": str(exc)}
            return row

        try:
            with ThreadPoolExecutor(max_workers=args.workers) as pool:
                futures = [
                    pool.submit(process, row)
                    for row in rows.values()
                    if row.get(phase, {}).get("status") != "ok"
                ]
                for future in as_completed(futures):
                    save(future.result())
        finally:
            elapsed(phase, start)
        if any(r.get(phase, {}).get("budget_exceeded") for r in rows.values()):
            build_report(run_dir)
            raise BudgetExceeded(f"--max-usd reached; results and charges saved in {run_dir}")
    manifest["invocation_wall_s"] = (
        manifest.get("invocation_wall_s", 0) + time.perf_counter() - invocation_start
    )
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    manifest["peak_rss_kib"] = peak / 1024 if sys.platform == "darwin" else peak
    write_json(manifest_path, manifest)
    if "report" in phases:
        return build_report(run_dir)
    return {
        "run_id": run_id,
        "errors": sum(r.get(p, {}).get("status") == "error" for r in rows.values() for p in phases),
    }

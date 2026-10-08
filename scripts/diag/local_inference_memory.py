"""Measure #322 through /agent/whisper using real models and a temporary store.

Run in a fresh process for each revision. The diagnostic never reads an existing
Ormah store. It requires already-cached default models and can enforce a process
RSS stop so unsafe baseline concurrency does not exhaust a shared host.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import gc
from importlib.metadata import version
import json
import os
from pathlib import Path
import platform
import resource
import subprocess
import sys
from tempfile import TemporaryDirectory
from threading import Event, Thread
import time


def current_rss_mib() -> float:
    if sys.platform == "linux":
        pages = int(Path("/proc/self/statm").read_text().split()[1])
        return pages * os.sysconf("SC_PAGE_SIZE") / 1024**2
    return int(subprocess.check_output(
        ["ps", "-o", "rss=", "-p", str(os.getpid())], text=True,
    ).strip()) / 1024


def report(stage: str, **extra) -> None:
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    peak /= 1024**2 if sys.platform == "darwin" else 1024
    print(json.dumps({
        "stage": stage,
        "rss_mib": round(current_rss_mib(), 1),
        "peak_mib": round(peak, 1),
        **extra,
    }), flush=True)


def start_rss_guard(cap_mib: int) -> Event:
    stopped = Event()

    def monitor() -> None:
        while not stopped.wait(0.05):
            rss = current_rss_mib()
            if rss > cap_mib:
                report("rss_cap_exceeded", cap_mib=cap_mib, observed_rss_mib=round(rss, 1))
                os._exit(86)

    Thread(target=monitor, name="rss-cap", daemon=True).start()
    return stopped


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--rounds", type=int, default=1)
    parser.add_argument("--prompt-chars", type=int, default=5000)
    parser.add_argument("--nodes", type=int, default=45)
    parser.add_argument("--rss-cap-mib", type=int, default=3000)
    parser.add_argument("--idle-seconds", type=float, default=2)
    args = parser.parse_args()
    if min(args.concurrency, args.rounds, args.prompt_chars, args.nodes, args.rss_cap_mib) < 1:
        parser.error("concurrency, rounds, prompt-chars, nodes, and rss-cap-mib must be positive")

    process_started = time.monotonic()
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["HF_HOME"] = str((args.cache_dir / "hf-home").resolve())
    os.environ["FASTEMBED_CACHE_PATH"] = str(args.cache_dir.resolve())
    guard = start_rss_guard(args.rss_cap_mib)

    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from fastembed import TextEmbedding

    from ormah.api.routes_agent import router
    from ormah.config import Settings
    from ormah.embeddings import reranker
    from ormah.embeddings.cache import is_model_cached
    from ormah.engine.memory_engine import MemoryEngine
    from ormah.models.node import CreateNodeRequest

    reranker_candidate_counts: list[int] = []
    original_rerank = reranker.rerank

    def recording_rerank(query, candidates, *args, **kwargs):
        reranker_candidate_counts.append(len(candidates))
        return original_rerank(query, candidates, *args, **kwargs)

    reranker.rerank = recording_rerank

    git_revision = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], text=True,
    ).strip()
    report(
        "environment",
        git_revision=git_revision,
        python=platform.python_version(),
        platform=platform.platform(),
        fastembed=version("fastembed"),
        onnxruntime=version("onnxruntime"),
        sqlite_vec=version("sqlite-vec"),
        cpu_count=os.cpu_count(),
        rss_cap_mib=args.rss_cap_mib,
        prompt_chars=args.prompt_chars,
    )

    if not is_model_cached(
        "BAAI/bge-base-en-v1.5", TextEmbedding.list_supported_models(),
        cache_dir=args.cache_dir,
    ) or not reranker.model_is_cached("Xenova/ms-marco-MiniLM-L-6-v2"):
        parser.error("cache-dir must already contain the default embedding and reranker models")

    unit = (
        "We are diagnosing memory usage in a Python service with concurrent requests, "
        "embedding retrieval, database updates, and cross encoder ranking. "
    )
    with TemporaryDirectory(prefix="ormah-inference-memory-") as store:
        settings = Settings(
            memory_dir=Path(store),
            backup_dir=Path(store) / "backups",
            llm_provider="none",
            # Keep the full 6 * 5 candidate pool for a controlled reranker
            # stress workload even if this host's sqlite-vec/SQLite pairing
            # falls back to FTS-only retrieval.
            whisper_min_relevance_score=0.0,
        )
        engine = MemoryEngine(settings)
        startup_started = time.monotonic()
        engine.startup()
        try:
            for i in range(args.nodes):
                engine.remember(CreateNodeRequest(
                    type="fact",
                    title=f"Memory performance investigation {i}",
                    content=unit * 12,
                ))
            app = FastAPI()
            app.include_router(router)
            app.state.engine = engine
            with TestClient(app) as client:
                report(
                    "ready",
                    nodes=args.nodes,
                    startup_seconds=round(time.monotonic() - startup_started, 3),
                    process_seconds=round(time.monotonic() - process_started, 3),
                )

                def request(round_id: int, i: int) -> dict[str, float | int]:
                    started = time.monotonic()
                    response = client.post("/agent/whisper", json={
                        "session_id": f"memory-repro-{round_id}-{i}",
                        "prompt": (
                            "Please investigate memory usage and concurrency "
                            "in our Python service. " + unit * 50
                        )[:args.prompt_chars],
                    })
                    response.raise_for_status()
                    return {
                        "seconds": round(time.monotonic() - started, 3),
                        "response_chars": len(response.json()["text"]),
                    }

                for round_id in range(args.rounds):
                    started = time.monotonic()
                    count_start = len(reranker_candidate_counts)
                    with ThreadPoolExecutor(args.concurrency) as pool:
                        results = list(pool.map(
                            lambda i: request(round_id, i),
                            range(args.concurrency),
                        ))
                    report(
                        "requests_complete",
                        round=round_id,
                        concurrency=args.concurrency,
                        total_seconds=round(time.monotonic() - started, 3),
                        requests=results,
                        reranker_candidate_counts=reranker_candidate_counts[count_start:],
                    )
                gc.collect()
                time.sleep(args.idle_seconds)
                report("idle_after_gc", idle_seconds=args.idle_seconds)
        finally:
            engine.shutdown()
            guard.set()


if __name__ == "__main__":
    main()

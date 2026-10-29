"""Real-model scheduling comparison. Run via a guarded fresh child per arm.

Requires job-local cache/scratch env (HF_HUB_OFFLINE=1). No server sockets or
production data. Arms share candidate source; only executor mapping changes.
JSONL records contain synthetic node IDs/scores and timing, never prompt text.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from contextvars import ContextVar
from datetime import datetime, timezone
import gc
from importlib.metadata import version
import json
import logging
import os
from pathlib import Path
import resource
import sqlite3
import subprocess
import sys
from threading import Barrier, Event, Lock, Thread, current_thread
import time
import uuid

TAG = ContextVar("benchmark_request", default="setup")
SHORT = ("Please investigate memory usage and concurrency in our Python service. "
         "How can we keep embedding retrieval and cross encoder ranking fast without "
         "retaining too much memory?")
UNIT = ("We are diagnosing memory usage in a Python service with concurrent requests, "
        "embedding retrieval, database updates, and cross encoder ranking. ")
LONG = (SHORT + " " + UNIT * 50)[:5000]
QUERY = "What have we learned about memory usage, embedding retrieval, and concurrent requests?"


def rss():
    return int(Path('/proc/self/statm').read_text().split()[1]) * os.sysconf('SC_PAGE_SIZE') / 2**20


def compact(results):
    return [{"id": r['node']['id'], **{k: float(r[k]) for k in
             ('score', 'raw_cosine', 'cross_encoder_score', 'ce_absolute') if r.get(k) is not None}}
            for r in results]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--arm', choices=['one', 'two_general', 'dedicated'], required=True)
    p.add_argument('--repo', type=Path, required=True)
    p.add_argument('--scratch', type=Path, required=True)
    p.add_argument('--corpus', type=Path, required=True)
    p.add_argument('--mode', choices=['seed', 'warm', 'cold', 'concurrency'], default='warm')
    p.add_argument('--rounds', type=int, default=6)
    p.add_argument('--batch', type=int, default=8)
    p.add_argument('--threads', type=int, default=0)
    p.add_argument('--floor', choices=['default', 'zero'], default='default')
    a = p.parse_args()
    a.repo = a.repo.resolve()
    a.scratch = a.scratch.resolve()
    a.corpus = a.corpus.resolve()
    assert os.environ.get('HF_HUB_OFFLINE') == '1'
    assert Path(os.environ['ORMAH_MEMORY_DIR']).is_relative_to(a.scratch.parent)
    assert Path(os.environ['FASTEMBED_CACHE_PATH']).exists()
    a.scratch.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(a.repo.resolve() / 'src'))
    import numpy as np
    import ormah
    from fastapi import FastAPI, Request
    from fastapi.testclient import TestClient
    from fastembed import TextEmbedding
    from fastembed.rerank.cross_encoder import TextCrossEncoder
    import onnxruntime as ort
    from ormah.api.routes_agent import router
    from ormah.config import Settings
    from ormah.embeddings import runtime, local_adapter, reranker
    from ormah.embeddings.text import embedding_text
    from ormah.embeddings.vector_store import VectorStore
    from ormah.engine.memory_engine import MemoryEngine
    from ormah.models.node import MemoryNode

    assert Path(ormah.__file__).resolve().is_relative_to(a.repo.resolve())
    source_sha = subprocess.check_output(['git', '-C', str(a.repo), 'rev-parse', 'HEAD'], text=True).strip()
    print_lock = Lock()
    events, reranks, vectors, warnings, sessions = [], [], [], [], []
    rerank_started = Event()
    measured_peak = [rss()]
    stop = Event()

    def report(stage, **data):
        with print_lock:
            print(json.dumps(dict(stage=stage, arm=a.arm, mode=a.mode, t=time.perf_counter(),
                rss_mib=rss(), peak_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
                load=os.getloadavg(), native_threads=len(list(Path('/proc/self/task').iterdir())),
                **data)), flush=True)

    def monitor():
        while not stop.wait(.05):
            value = rss()
            measured_peak[0] = max(measured_peak[0], value)
            if value > 2800:
                report('rss_cap', cap=2800)
                os._exit(86)

    Thread(target=monitor, daemon=True).start()
    logging.basicConfig(level=logging.WARNING)

    class TimingHandler(logging.Handler):
        def emit(self, record):
            if hasattr(record, 'inference'):
                event = dict(record.inference, request=TAG.get(), thread=current_thread().name)
                events.append(event)
                if 'ended' not in event and event['operation'] == 'rerank' and TAG.get() != 'setup':
                    rerank_started.set()
            elif record.levelno >= logging.WARNING and record.name.startswith('ormah'):
                warnings.append({'logger': record.name, 'message': record.getMessage()})

    handler = TimingHandler()
    logging.getLogger().addHandler(handler)
    runtime.logger.setLevel(logging.DEBUG)
    runtime.logger.propagate = False
    runtime.logger.addHandler(handler)
    # Isolate scheduling, retaining the candidate's cache safety in all arms.
    runtime._worker.shutdown()
    runtime._recall_worker.shutdown()
    runtime._worker = ThreadPoolExecutor(max_workers=2 if a.arm == 'two_general' else 1,
                                        thread_name_prefix='bench-general')
    runtime._recall_worker = (ThreadPoolExecutor(max_workers=1, thread_name_prefix='bench-recall')
                              if a.arm == 'dedicated' else runtime._worker)
    runtime.MAX_BATCH_SIZE = local_adapter.MAX_BATCH_SIZE = reranker.MAX_BATCH_SIZE = a.batch
    original_session = ort.InferenceSession.__init__

    def session_init(self, *args, **kwargs):
        original_session(self, *args, **kwargs)
        opts = self.get_session_options()
        sessions.append(dict(arena=opts.enable_cpu_mem_arena,
            intra=opts.intra_op_num_threads, inter=opts.inter_op_num_threads,
            providers=self.get_providers(), session_id=id(self)))

    ort.InferenceSession.__init__ = session_init
    for cls in (TextEmbedding, TextCrossEncoder):
        original = cls.__init__

        def initialize(self, *args, _original=original, **kwargs):
            if a.threads:
                kwargs['threads'] = a.threads
            return _original(self, *args, **kwargs)

        cls.__init__ = initialize
    original_vector = VectorStore.search

    def vector_search(self, *args, **kwargs):
        try:
            results = original_vector(self, *args, **kwargs)
        except Exception:
            vectors.append(dict(request=TAG.get(), error=True))
            raise
        vectors.append(dict(request=TAG.get(), count=len(results), error=False))
        return results

    VectorStore.search = vector_search
    original_rerank = reranker.rerank

    def record_rerank(query, candidates, *args, **kwargs):
        start = time.perf_counter()
        results = original_rerank(query, candidates, *args, **kwargs)
        reranks.append(dict(request=TAG.get(), start=start, end=time.perf_counter(),
            count=len(candidates), preference=query.startswith('Relevant user preference'),
            results=compact(results)))
        return results

    reranker.rerank = record_rerank
    report('environment', source_sha=source_sha, imported_file=ormah.__file__,
        dirty=bool(subprocess.check_output(['git', '-C', str(a.repo), 'status', '--porcelain'])),
        python=sys.version, sqlite=sqlite3.sqlite_version,
        versions={x: version(x) for x in ('fastembed', 'onnxruntime', 'tokenizers', 'numpy', 'anyio')},
        batch=a.batch, threads=a.threads, floor=a.floor, affinity=sorted(os.sched_getaffinity(0)),
        cpu_model=next(x for x in Path('/proc/cpuinfo').read_text().splitlines() if x.startswith('model name')),
        short_chars=len(SHORT), long_chars=len(LONG), env={k: os.environ.get(k) for k in
            ('MALLOC_ARENA_MAX', 'OMP_NUM_THREADS', 'TOKENIZERS_PARALLELISM', 'HF_HUB_OFFLINE')})
    settings = Settings(memory_dir=a.scratch / 'store', backup_dir=a.scratch / 'backups',
                        llm_provider='none')
    if a.floor == 'zero':
        settings.whisper_min_relevance_score = 0.0
    engine = MemoryEngine(settings)
    # Explicitly populate a synthetic store without startup warmup. This permits
    # concurrent cold first use against precomputed real vectors in cold mode.
    stamp = datetime(2026, 10, 1, tzinfo=timezone.utc)
    preferences = [
        'I prefer small reversible changes and measurable performance evidence before tuning services.',
        'Please preserve complete user queries when optimizing memory retrieval and ranking.',
        'Use bounded batches for memory intensive tasks and document the batch size limit.',
        'Keep deliberate recall responsive while background inference work is busy.',
        'When changing concurrency, check thread safety and avoid duplicate model loading.',
        'Use scratch directories and synthetic data for benchmarks; never use production memories.',
        'Explain peak versus retained process memory and latency tradeoffs in plain language.',
        'Prefer explicit CPU and memory measurements to assuming each worker owns one core.',
    ]
    facts = [
        'The Python service uses ONNX sessions for embedding retrieval and cross encoder ranking.',
        'A long rerank can block requests waiting on the shared inference executor.',
        'Memory usage depends on sequence length, batch size, and concurrent requests.',
        'Database vector search combines semantic similarity with keyword matches.',
        'A reusable inference worker reduces overlapping temporary activation buffers.',
        'Full user queries retain useful context before the model token limit is applied.',
        'The service records process RSS after concurrent embedding and ranking requests.',
        'A request context can distinguish deliberate recall from background whisper work.',
    ]
    nodes = []
    for i in range(48):
        pref = i >= 40
        text = preferences[i-40] if pref else facts[i % len(facts)] + ' ' + UNIT * (1 + i % 3)
        node = MemoryNode(id=str(uuid.uuid5(uuid.NAMESPACE_DNS, f'ormah-recall-bench-{i}')),
            type='preference' if pref else 'fact', title=f'{"Operating preference" if pref else "Service finding"} {i:02}',
            content=text, created=stamp, updated=stamp, last_accessed=stamp, importance=.5)
        nodes.append(node)
        engine.builder.index_single(engine.file_store.save(node))
    vec_store = VectorStore(engine.db)
    encoder = engine._get_hybrid_search().encoder
    texts = [embedding_text(n.title, n.content, settings.embedding_max_content_chars) for n in nodes]
    if a.mode == 'seed':
        embeddings = encoder.encode_batch(texts)
        np.savez(a.corpus, ids=np.array([n.id for n in nodes]), vectors=embeddings)
        report('seed', node_count=len(nodes), vector_shape=embeddings.shape, sessions=sessions)
    else:
        with np.load(a.corpus, allow_pickle=False) as corpus:
            assert corpus['ids'].tolist() == [n.id for n in nodes]
            vec_store.upsert_batch(list(zip(corpus['ids'].tolist(), corpus['vectors'])))
        assert vec_store.count() == 48
        app = FastAPI()
        app.include_router(router)
        app.state.engine = engine

        @app.middleware('http')
        async def request_context(request: Request, call_next):
            token = TAG.set(request.headers['x-benchmark-request'])
            try:
                return await call_next(request)
            finally:
                TAG.reset(token)

        def health():
            assert not warnings, warnings
            assert all(not v['error'] for v in vectors)
            assert all(not s['arena'] for s in sessions)
            assert len(local_adapter._model_cache) == 1
            assert len(reranker._model_cache) == 1
            assert len(sessions) == 2
            assert engine._whisper_reranker_available

        with TestClient(app) as client:
            def request(kind, tag):
                token = TAG.set(tag)
                start = time.perf_counter()
                try:
                    if kind == 'recall':
                        response = client.post('/agent/recall', json={'query': QUERY, 'limit': 10,
                            'session_id': tag}, headers={'x-benchmark-request': tag})
                    else:
                        response = client.post('/agent/whisper', json={'prompt': LONG if kind == 'long' else SHORT,
                            'session_id': tag}, headers={'x-benchmark-request': tag})
                    response.raise_for_status()
                    end = time.perf_counter()
                    rows = [dict(r) for r in engine.db.conn.execute(
                        'SELECT node_id,score,raw_cosine,cross_encoder_score,gate_score,was_injected,final_rank '
                        'FROM whisper_log WHERE session_id=? ORDER BY node_id', (tag,)).fetchall()]
                    assert rows, f'No retrieval evidence for {tag}'
                    return dict(kind=kind, request=tag, start=start, end=end, seconds=end-start,
                                retrieval=rows)
                finally:
                    TAG.reset(token)

            def phase(label, kinds, offset=None):
                rerank_started.clear()
                indices = len(events), len(reranks), len(vectors)
                measured_peak[0] = rss()
                gate = Barrier(len(kinds))
                start = time.perf_counter()
                cpu_start = time.process_time()

                def run(i, kind):
                    gate.wait(timeout=10)
                    if kind == 'recall':
                        if offset == 'rerank':
                            assert rerank_started.wait(30), 'Whisper never entered rerank'
                            time.sleep(.05)
                        elif offset is not None:
                            time.sleep(float(offset))
                    return request(kind, f'{label}-{i}')

                with ThreadPoolExecutor(len(kinds)) as callers:
                    pending = [callers.submit(run, i, kind) for i, kind in enumerate(kinds)]
                    results = [future.result(timeout=120) for future in pending]
                elapsed = time.perf_counter()-start
                phase_events = events[indices[0]:]
                completed_reranks = [e for e in phase_events if e['operation'] == 'rerank' and 'ended' in e]
                if offset == 'rerank':
                    for result in results:
                        if result['kind'] == 'recall':
                            assert any(e['started'] <= result['start'] <= e['ended'] for e in completed_reranks)
                report('phase', label=label, offset=offset, seconds=elapsed,
                    cpu_seconds=time.process_time()-cpu_start, results=results,
                    events=phase_events, reranks=reranks[indices[1]:], vectors=vectors[indices[2]:],
                    sampled_peak_mib=measured_peak[0])
                health()

            if a.mode == 'cold':
                assert not sessions and not local_adapter._model_cache and not reranker._model_cache
                phase('cold', ['long', 'short', 'recall', 'recall'])
            else:
                engine._warmup_embedder()
                engine._warmup_reranker()
                request('short', 'warmup')
                request('recall', 'warmup-recall')
                health()
                report('ready', sessions=sessions, model_counts=[len(local_adapter._model_cache), len(reranker._model_cache)])
                if a.mode == 'concurrency':
                    # Repeat actual shared-tokenizer and ONNX calls with dissimilar lengths.
                    refs = [encoder.encode_query(QUERY), encoder.encode(SHORT), encoder.encode(LONG)]
                    barrier = Barrier(2)

                    def check(origin):
                        with runtime.inference_request(origin):
                            for _ in range(12):
                                barrier.wait(timeout=30)
                                for idx, text in enumerate((QUERY, SHORT, LONG)):
                                    result = encoder.encode_query(text) if idx == 0 else encoder.encode(text)
                                    np.testing.assert_allclose(result, refs[idx], rtol=1e-5, atol=1e-6)

                    with ThreadPoolExecutor(2) as callers:
                        futures = [callers.submit(check, origin) for origin in ('general', 'recall')]
                        for future in futures:
                            future.result(timeout=120)
                    report('concurrent_equivalence', iterations=24, vectors_checked=72, sessions=sessions)
                else:
                    for i in range(3):
                        phase(f'idle-{i}', ['recall'])
                        phase(f'short-solo-{i}', ['short'])
                    phase('short-four', ['short']*4)
                    for i in range(a.rounds):
                        offset = [.1, .5, 'rerank'][i % 3]
                        phase(f'mixed-long-{i}', ['long', 'long', 'recall'], offset)
                    phase('mixed-short', ['short', 'short', 'recall'], .1)
                    phase('recall-burst', ['long', 'long'] + ['recall']*6, 'rerank')
                    # Background re-encoding of existing nodes: exercises ingestion
                    # model work + vector writes without changing corpus ranking.
                    gate = Event()

                    def ingest():
                        token = TAG.set('background-ingestion')
                        try:
                            gate.set()
                            for node in nodes[:8]:
                                engine._index_embedding(node)
                        finally:
                            TAG.reset(token)

                    with ThreadPoolExecutor(1) as pool:
                        future = pool.submit(ingest)
                        assert gate.wait(5)
                        phase('ingestion-overlap', ['short', 'recall'], .1)
                        future.result(timeout=30)
            # Serial reference and raw candidate evidence for cross-arm equivalence.
            reference = engine.recall_search_structured(QUERY)
            report('reference', results=compact(reference), sessions=sessions,
                model_counts=[len(local_adapter._model_cache), len(reranker._model_cache)],
                vector_calls=len(vectors), vector_errors=sum(v['error'] for v in vectors),
                preference_reranks=sum(r['preference'] and r['count'] > 0 for r in reranks))
            health()
    engine.shutdown()
    gc.collect()
    time.sleep(.3)
    report('complete', retained_mib=rss(), sessions=sessions, warnings=warnings)
    runtime._worker.shutdown()
    if runtime._recall_worker is not runtime._worker:
        runtime._recall_worker.shutdown()
    stop.set()


if __name__ == '__main__':
    main()

"""Isolated file/index/vector seeding with persistent extraction and embedding caches."""

from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from eval.bench.artifacts import Journal
from eval.bench.cost import BudgetExceeded
from eval.bench.datasets import session_dict
from eval.recall.seeder import clear_eval_db
from ormah.background.llm import LLMAdapter
from ormah.embeddings.encoder import get_encoder
from ormah.embeddings.text import embedding_text
from ormah.embeddings.vector_store import VectorStore
from ormah.models.node import MemoryNode, Tier


def digest(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


class ProviderAdapter(LLMAdapter):
    """Bridge the real engine ingest prompt to the benchmark's provider/accounting."""

    def __init__(self, provider):
        self.provider = provider
        self.local = threading.local()

    def generate(
        self, prompt, json_mode=True, *, response_format=None, temperature=None, max_tokens=None
    ):
        try:
            result = self.provider.complete(
                prompt, item_id=getattr(self.local, "item_id", ""), max_tokens=max_tokens or 4096
            )
        except BudgetExceeded as exc:
            # The production ingest facade converts exceptions to error strings.
            # Retain the typed budget signal so the benchmark stops immediately.
            self.local.budget_error = exc
            raise
        # The engine tolerates malformed JSON as no memories. Bench must not cache that.
        from ormah.background.llm_client import extract_json

        parsed = json.loads(extract_json(result.text))
        while isinstance(parsed, dict) and "memories" in parsed:
            parsed = parsed["memories"]
        if not isinstance(parsed, list):
            raise ValueError("Extraction did not return a memories list")
        return result.text


class ClaudeCLIAdapter(ProviderAdapter):
    pass


class CodexAdapter(ProviderAdapter):
    pass


class EmbeddingCache:
    def __init__(self, path: Path, namespace: str):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(path)
        self.db.execute(
            "CREATE TABLE IF NOT EXISTS embeddings "
            "(namespace TEXT, hash TEXT, vector BLOB, PRIMARY KEY(namespace, hash))"
        )
        self.namespace = namespace
        self.hits = self.misses = 0

    def encode(self, texts, encoder):
        vectors = {}
        missing = {}
        keys = [digest(text) for text in texts]
        for key, text in zip(keys, texts):
            if key in vectors or key in missing:
                continue
            row = self.db.execute(
                "SELECT vector FROM embeddings WHERE namespace=? AND hash=?", (self.namespace, key)
            ).fetchone()
            if row:
                vectors[key] = np.frombuffer(row[0], dtype=np.float32)
                self.hits += 1
            else:
                missing[key] = text
                self.misses += 1
        if missing:
            # Group similarly sized texts to reduce attention padding on CPU.
            # Restore caller order below through the content-hash mapping.
            missing = dict(sorted(missing.items(), key=lambda item: len(item[1])))
            encoded = encoder.encode_batch(list(missing.values()), batch_size=16)
            with self.db:
                for key, vec in zip(missing, encoded, strict=True):
                    vec = np.asarray(vec, dtype=np.float32)
                    vectors[key] = vec
                    self.db.execute(
                        "INSERT OR REPLACE INTO embeddings VALUES (?, ?, ?)",
                        (self.namespace, key, vec.tobytes()),
                    )
        return [vectors[key] for key in keys]

    def close(self):
        self.db.close()


def raw_memories(dataset, session):
    return [
        {
            "id": digest(f"{dataset}:{session.session_id}:{t.turn_id}"),
            "title": f"{t.speaker}: {t.text[:80]}",
            "content": f"{t.speaker}: {t.text}" if dataset == "locomo" else t.text,
            "created": session.date,
            "type": "fact",
            "tags": [f"bench:{dataset}", f"session:{session.session_id}", f"turn:{t.turn_id}"],
        }
        for t in session.turns
    ]


def extract_session(engine, dataset, session, cache_dir, adapter):
    from ormah.engine.memory_engine import _INGEST_LLM_PROMPT

    text = f"Session date: {session.date}\n" + "\n".join(
        f"[{t.turn_id}] {t.speaker}: {t.text}" for t in session.turns
    )
    prompt = _INGEST_LLM_PROMPT.format(
        conversation=text[: engine.settings.ingest_max_content_chars]
    )
    prompt_hash = digest(prompt + adapter.provider.name + adapter.provider.model)
    key = digest(session.session_id + prompt_hash)
    journal = Journal(cache_dir / f"{key}.jsonl")
    if journal.rows:
        return journal.rows[-1]["memories"]
    adapter.local.item_id = session.session_id
    adapter.local.budget_error = None
    memories = engine.ingest_conversation(text, dry_run=True)
    if isinstance(memories, str):
        if adapter.local.budget_error is not None:
            raise adapter.local.budget_error
        raise RuntimeError(memories)
    result = []
    for i, mem in enumerate(memories):
        result.append(
            {
                **mem,
                "id": digest(key + str(i)),
                "created": session.date,
                # No invented exact turn provenance: ingest returns session-level facts.
                "tags": mem.get("tags", []) + [f"bench:{dataset}", f"session:{session.session_id}"],
                "turn_provenance": "unknown",
            }
        )
    journal.append(
        {
            "session_id": session.session_id,
            "prompt_hash": prompt_hash,
            "input_chars": len(text),
            "truncated_chars": max(0, len(text) - engine.settings.ingest_max_content_chars),
            "session_hash": digest(json.dumps(session_dict(session), sort_keys=True)),
            "memories": result,
        }
    )
    return result


def prepare_memories(engine, question, mode, cache_dir, adapter=None, workers=4):
    if mode == "raw":
        return [
            mem for session in question.sessions for mem in raw_memories(question.dataset, session)
        ]
    if adapter is None:
        raise ValueError("extract mode requires an explicitly enabled extract phase")
    with ThreadPoolExecutor(max_workers=workers) as pool:
        groups = pool.map(
            lambda s: extract_session(engine, question.dataset, s, cache_dir, adapter),
            question.sessions,
        )
        return [mem for group in groups for mem in group]


def seed_memories(engine, memories, cache):
    clear_eval_db(engine)
    encoder = get_encoder(engine.settings)
    vectors = VectorStore(engine.db)
    now = datetime.now(timezone.utc)
    for start in range(0, len(memories), 64):
        nodes = [
            MemoryNode(
                id=m["id"],
                title=m["title"],
                content=m["content"],
                type=m.get("type", "fact"),
                tier=Tier.working,
                confidence=float(m.get("confidence", 1)),
                tags=m["tags"],
                source="eval:bench",
                created=m["created"],
                # Backdated creation must not trigger FSRS decay/tier exclusion.
                updated=now,
                last_accessed=now,
            )
            for m in memories[start : start + 64]
        ]
        texts = [
            embedding_text(n.title, n.content, engine.settings.embedding_max_content_chars)
            for n in nodes
        ]
        embeddings = cache.encode(texts, encoder)
        for node in nodes:
            path = engine.file_store.save(node)
            engine.builder.index_single(path)
        vectors.upsert_batch([(n.id, v) for n, v in zip(nodes, embeddings, strict=True)])

    # HybridSearch intentionally tolerates vector failures in production. A
    # benchmark must fail visibly rather than silently measure lexical fallback.
    if memories:
        probe = vectors.get(memories[0]["id"])
        if probe is None:
            raise RuntimeError("Seeded haystack is missing its first embedding")
        vectors.search(probe, limit=1)

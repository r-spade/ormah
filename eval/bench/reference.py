"""Independent dense reference and deterministic, auditable context budgets."""

from __future__ import annotations

from copy import deepcopy
import time

import numpy as np

from eval.bench.answer import memory_context


def dense_question(engine, question: str, k: int) -> dict:
    """Exact cosine over the stored vectors, without Ormah ranking signals.

    Use the same production document/query encoder as hybrid recall. Explicit
    normalization makes this independent of sqlite-vec's L2 approximation.
    All candidates are compared, with node ID breaking exact cosine ties.
    """
    from ormah.embeddings.encoder import get_encoder

    if k <= 0:
        raise ValueError("k must be positive")
    start = time.perf_counter()
    query = np.asarray(get_encoder(engine.settings).encode_query(question), dtype=np.float32)
    norm = np.linalg.norm(query)
    if not np.isfinite(query).all() or norm <= 0:
        raise ValueError("Invalid query vector")
    query = query / norm
    candidates = []
    for row in engine.db.conn.execute("SELECT id, embedding FROM node_vectors ORDER BY id"):
        vector = np.frombuffer(row[1], dtype=np.float32)
        norm = np.linalg.norm(vector)
        if vector.shape != query.shape or not np.isfinite(vector).all() or norm <= 0:
            raise ValueError(f"Invalid stored vector: {row[0]}")
        candidates.append((str(row[0]), float(np.dot(query, vector / norm))))
    candidates.sort(key=lambda pair: (-pair[1], pair[0]))
    tags = engine.graph.get_tags_batch([node_id for node_id, _ in candidates[:k]])
    ranked = []
    for node_id, cosine in candidates[:k]:
        node = engine.graph.get_node(node_id)
        if node is None:
            raise ValueError(f"Vector without stored node: {node_id}")
        ranked.append({
            "node": {**{key: node.get(key) for key in ("id", "title", "content", "created")},
                     "tags": sorted(tags.get(node_id, []))},
            "score": cosine, "raw_cosine": cosine, "source": "dense_cosine",
        })
    return {"strategy": "dense", "ranked": ranked, "reranker_active": False,
            "latency_s": time.perf_counter() - start,
            "production_gate": None, "passing_gate": None,
            "answer_context_chars": len(memory_context(ranked))}


def empty_question() -> dict:
    return {"strategy": "none", "ranked": [], "reranker_active": False,
            "latency_s": 0.0, "production_gate": None, "passing_gate": None,
            "answer_context_chars": 0}


def budget_context(result: dict, max_chars: int, k: int = 30) -> dict:
    """Expose the exact prefix of <=k dated entries, counting all formatting.

    Preserve complete pre-budget rankings for retrieval analysis and explicit
    character spans for exposure diagnostics. A cap may bisect any part of the
    last dated entry; the exact text, rather than a reconstruction, is sent.
    """
    if max_chars < 0 or not 0 < k <= 30:
        raise ValueError("Require nonnegative max_chars and 1 <= k <= 30")
    result = deepcopy(result)
    ranked = result["ranked"][:k]
    full = memory_context(ranked)
    text = full[:max_chars]
    spans = []
    start = 0
    for item in ranked:
        node = item["node"]
        block = memory_context([item])
        content = node.get("content", "") or ""
        title = node.get("title", "") or ""
        title_start = start + len(f"[{node.get('created', 'unknown date')}] ")
        content_start = start + len(block) - len(content)
        spans.append({
            "memory_id": node["id"], "entry_start": start,
            "entry_end": start + len(block), "content_start": content_start,
            "content_end": start + len(block),
            "title_start": title_start, "title_end": title_start + len(title),
            "exposed_entry_chars": max(0, min(len(block), len(text) - start)),
            "exposed_title_chars": max(0, min(len(title), len(text) - title_start)),
            "exposed_content_chars": max(0, min(len(content), len(text) - content_start)),
        })
        start += len(block) + 2
    result.update({
        "ranked": ranked, "pre_budget_ranked": result["ranked"],
        "answer_context_text": text, "answer_context_chars": len(text),
        "answer_context_tokens_estimate": (len(text) + 3) // 4,
        "context_budget_chars": max_chars, "context_top_k": k,
        "pre_budget_context_chars": len(full), "exposure_spans": spans,
        "context_clipped": len(text) != len(full),
        "exposed_memory_ids": [s["memory_id"] for s in spans if s["exposed_entry_chars"]],
    })
    return result

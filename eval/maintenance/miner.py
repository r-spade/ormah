"""Mine auto-link candidate pairs from a production store, read-only (#87 eval).

Pairs are sampled across similarity bands so the A/B corpus exercises the whole
decision surface, not just obvious matches. The mined file contains PERSONAL
memory content and must NEVER be committed (see corpus/.gitignore) — only the
aggregate agreement metrics leave the machine.
"""
from __future__ import annotations

import json
import random
import sqlite3
from pathlib import Path  # noqa: F401 -- convenient for callers building paths

import sqlite_vec

BANDS = [(0.65, 0.75), (0.75, 0.85), (0.85, 1.01)]


def _connect_ro(db_path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    conn.enable_load_extension(False)
    return conn


def mine_pairs(db_path: str, out_path: str, n: int = 100, seed: int = 42) -> int:
    conn = _connect_ro(db_path)
    rng = random.Random(seed)
    nodes = conn.execute("SELECT id, title, content, type, space FROM nodes").fetchall()
    rng.shuffle(nodes)
    per_band = n // len(BANDS)
    got = {b: 0 for b in BANDS}
    seen: set[tuple[str, str]] = set()
    out = open(out_path, "w", encoding="utf-8")
    written = 0
    for node in nodes:
        if written >= n:
            break
        vec = conn.execute("SELECT embedding FROM node_vectors WHERE id = ?",
                           (node["id"],)).fetchone()
        if vec is None:
            continue
        for row in conn.execute(
                "SELECT id, distance FROM node_vectors WHERE embedding MATCH ? "
                "ORDER BY distance LIMIT 6", (vec[0],)):
            sim = 1.0 - (row["distance"] ** 2 / 2.0)
            key = tuple(sorted([node["id"], row["id"]]))
            band = next((b for b in BANDS if b[0] <= sim < b[1]), None)
            if row["id"] == node["id"] or key in seen or band is None or got[band] >= per_band + 5:
                continue
            other = conn.execute(
                "SELECT id, title, content, type, space FROM nodes WHERE id = ?",
                (row["id"],)).fetchone()
            if other is None:
                continue
            seen.add(key)
            got[band] += 1
            out.write(json.dumps({
                "pair_id": f"{key[0][:8]}-{key[1][:8]}", "similarity": round(sim, 3),
                "node_a": {k: node[k] for k in node.keys()},
                "node_b": {k: other[k] for k in other.keys()},
            }) + "\n")
            written += 1
    out.close()
    return written

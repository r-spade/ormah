"""Vector storage and search using sqlite-vec."""

from __future__ import annotations

import struct
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from ormah.index.db import Database


def _serialize_f32(vec: np.ndarray) -> bytes:
    """Serialize a numpy float32 vector to bytes for sqlite-vec."""
    return struct.pack(f"{len(vec)}f", *vec.astype(np.float32))


class VectorStore:
    """sqlite-vec backed vector storage."""

    def __init__(self, db: Database) -> None:
        self.db = db

    def upsert(self, node_id: str, embedding: np.ndarray) -> None:
        """Insert or update a vector for a node.

        The DELETE + INSERT pair runs inside a single transaction so that
        a concurrent reader never sees a window where the row is missing.
        """
        vec_bytes = _serialize_f32(embedding)
        with self.db.transaction() as conn:
            conn.execute("DELETE FROM node_vectors WHERE id = ?", (node_id,))
            conn.execute(
                "INSERT INTO node_vectors (id, embedding) VALUES (?, ?)",
                (node_id, vec_bytes),
            )
        # Checkpoint WAL outside the lock — harmless for PASSIVE mode
        self.db.conn.execute("PRAGMA wal_checkpoint(PASSIVE)")

    def upsert_batch(self, items: list[tuple[str, np.ndarray]]) -> None:
        """Batch insert/update vectors in a single transaction."""
        with self.db.transaction() as conn:
            for node_id, embedding in items:
                vec_bytes = _serialize_f32(embedding)
                conn.execute("DELETE FROM node_vectors WHERE id = ?", (node_id,))
                conn.execute(
                    "INSERT INTO node_vectors (id, embedding) VALUES (?, ?)",
                    (node_id, vec_bytes),
                )

    def search(self, query_vec: np.ndarray, limit: int = 10) -> list[dict[str, Any]]:
        """Find nearest neighbors. Returns results with cosine similarity scores."""
        vec_bytes = _serialize_f32(query_vec)
        rows = self.db.conn.execute(
            """
            SELECT id, distance
            FROM node_vectors
            WHERE embedding MATCH ?
            ORDER BY distance
            LIMIT ?
            """,
            (vec_bytes, limit),
        ).fetchall()
        # sqlite-vec returns L2 distance. For normalized vectors:
        # cosine_similarity = 1 - (L2_distance² / 2)
        return [
            {"id": row[0], "distance": row[1], "similarity": 1.0 - (row[1] ** 2 / 2.0)}
            for row in rows
        ]

    def get(self, node_id: str) -> np.ndarray | None:
        """Retrieve the stored embedding for a node, or None if not found."""
        row = self.db.conn.execute(
            "SELECT embedding FROM node_vectors WHERE id = ?", (node_id,)
        ).fetchone()
        if row is None:
            return None
        blob = row[0]
        dim = len(blob) // 4  # float32 = 4 bytes each
        return np.array(struct.unpack(f"{dim}f", blob), dtype=np.float32)

    def delete(self, node_id: str) -> None:
        with self.db.transaction():
            self.db.conn.execute("DELETE FROM node_vectors WHERE id = ?", (node_id,))
        self.db.conn.execute("PRAGMA wal_checkpoint(PASSIVE)")

    def count(self) -> int:
        row = self.db.conn.execute("SELECT COUNT(*) FROM node_vectors").fetchone()
        return row[0] if row else 0

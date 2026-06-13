"""Affinity boost module for the adaptive feedback loop.

Computes per-node score adjustments based on historical user feedback signals
(explicit thumbs up/down and implicit engagement) stored in the affinity table.
"""

from __future__ import annotations

import math
from datetime import datetime, timezone

import numpy as np


def batch_fetch_affinity(conn, node_ids: list[str]) -> dict[str, list[dict]]:
    """Fetch all affinity rows for a list of node_ids in a single query.

    Returns a dict keyed by node_id. Each value is a list of row dicts with at
    minimum: ``prompt_vec`` (bytes blob), ``signal`` (int, +1 or -1),
    ``source`` (str, 'explicit' or 'implicit'), ``confirmed_at`` (str ISO datetime).
    """
    if not node_ids:
        return {}
    placeholders = ",".join("?" * len(node_ids))
    rows = conn.execute(
        f"SELECT node_id, prompt_vec, signal, source, confirmed_at "
        f"FROM affinity WHERE node_id IN ({placeholders})",
        node_ids,
    ).fetchall()
    result: dict[str, list[dict]] = {}
    for row in rows:
        nid = row["node_id"]
        if nid not in result:
            result[nid] = []
        result[nid].append(dict(row))
    return result


def compute_affinity_boost(
    current_prompt_vec: np.ndarray,
    node_id: str,
    affinity_rows: list[dict],
    settings,
) -> float:
    """Compute the affinity boost for a candidate node.

    For each affinity row, a weight is calculated as the product of:
    - cosine similarity between the current prompt vector and the stored prompt vector
      (rows below ``settings.affinity_similarity_threshold`` are skipped)
    - a recency factor that decays with a half-life of ``settings.affinity_half_life_days``
    - a source weight (``settings.affinity_implicit_weight`` for implicit, 1.0 for explicit)

    The normalised weighted sum of signals is then scaled by
    ``settings.affinity_max_boost``.

    Returns a float in the range [-affinity_max_boost, +affinity_max_boost],
    or 0.0 when no rows meet the similarity threshold.
    """
    if not affinity_rows:
        return 0.0

    threshold: float = settings.affinity_similarity_threshold
    half_life: float = settings.affinity_half_life_days
    max_boost: float = settings.affinity_max_boost
    implicit_w: float = settings.affinity_implicit_weight

    ln2 = math.log(2.0)
    now_utc = datetime.now(timezone.utc)

    weighted_sum = 0.0
    weight_total = 0.0

    current_norm = float(np.linalg.norm(current_prompt_vec))

    for row in affinity_rows:
        # Deserialize stored prompt vector
        row_vec = np.frombuffer(row["prompt_vec"], dtype=np.float32)
        row_norm = float(np.linalg.norm(row_vec))

        if current_norm == 0.0 or row_norm == 0.0:
            continue

        sim = float(np.dot(current_prompt_vec, row_vec) / (current_norm * row_norm))

        if sim < threshold:
            continue

        # Parse confirmed_at (may have or lack timezone info)
        confirmed_str: str = row["confirmed_at"]
        try:
            confirmed_dt = datetime.fromisoformat(confirmed_str)
        except ValueError:
            continue

        # Ensure timezone-aware for arithmetic
        if confirmed_dt.tzinfo is None:
            confirmed_dt = confirmed_dt.replace(tzinfo=timezone.utc)

        days_ago = (now_utc - confirmed_dt).total_seconds() / 86400.0
        recency = math.exp(-days_ago * ln2 / half_life)

        source_w = 1.0 if row["source"] == "explicit" else implicit_w
        weight = sim * recency * source_w

        signal = int(row["signal"])
        weighted_sum += signal * weight
        weight_total += weight

    if weight_total == 0.0:
        return 0.0

    normalised = weighted_sum / weight_total  # in [-1, +1]
    return normalised * max_boost

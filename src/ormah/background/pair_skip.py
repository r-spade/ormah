"""Shared ordered-pair skip helpers for pairwise maintenance jobs (dedup, conflict)."""
from __future__ import annotations


def normalize_pair(a_id: str, b_id: str) -> tuple[str, str]:
    s = sorted([a_id, b_id])
    return (s[0], s[1])


def pair_skip_sql(table: str, terminal_results: tuple[str, ...]) -> str:
    # literals only — never interpolate user data (result values are snake_case identifiers)
    assert all(r.replace("_", "").isalpha() for r in terminal_results)
    placeholders = ",".join("'" + r + "'" for r in terminal_results)
    # datetime(checked_at) normalizes ISO 'T'/tz strings to SQLite's comparable format.
    return (
        f"SELECT 1 FROM {table} WHERE node_a = ? AND node_b = ? AND "
        f"(result IN ({placeholders}) OR "
        f"(result = 'error' AND datetime(checked_at) > datetime('now', ?)))"
    )

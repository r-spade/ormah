"""Seed the isolated whisper eval DB with memories from a corpus case."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

from ormah.models.node import Connection, EdgeType, MemoryNode, NodeType, Tier


def _parse_dt(val: str) -> datetime:
    """Parse an ISO/RFC3339 datetime string into a timezone-aware UTC datetime.

    Uses stdlib only (no optional deps). Accepts both trailing "Z" and
    explicit offsets.
    """
    s = val.strip()
    if s.endswith("Z"):
        # `datetime.fromisoformat()` accepts "+00:00" broadly across Python versions.
        s = s[:-1] + "+00:00"
    dt = datetime.fromisoformat(s)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _seed_dt_from_mem(mem: dict, field: str) -> datetime | None:
    """Return a datetime for *field* from corpus memory *mem*.

    Supported formats:
    - ``<field>``: ISO string
    - ``<field>_days_ago``: number (float/int)
    - ``<field>_hours_ago``: number (float/int)
    """
    now = datetime.now(timezone.utc)
    days_ago = mem.get(f"{field}_days_ago")
    if days_ago is not None:
        return now - timedelta(days=float(days_ago))
    hours_ago = mem.get(f"{field}_hours_ago")
    if hours_ago is not None:
        return now - timedelta(hours=float(hours_ago))
    iso = mem.get(field)
    if iso:
        return _parse_dt(str(iso))
    return None


def seed_case(engine, case: dict, *, preserve_self: bool = False) -> None:
    """Clear eval DB and seed with memories from *case*.

    Memories are inserted with their corpus node_id preserved.
    Skips auto-linking and core-cap enforcement.
    """
    clear_eval_db(engine, preserve_self=preserve_self)
    for mem in case.get("memories", []):
        created = _seed_dt_from_mem(mem, "created")
        updated = _seed_dt_from_mem(mem, "updated")
        last_accessed = _seed_dt_from_mem(mem, "last_accessed")
        last_review = _seed_dt_from_mem(mem, "last_review")
        valid_until = _seed_dt_from_mem(mem, "valid_until")

        connections: list[Connection] = []
        for c in mem.get("connections", []) or []:
            # Corpus validation should have checked these, but keep it defensive.
            if not isinstance(c, dict):
                continue
            target = c.get("target")
            if not target:
                continue
            edge = c.get("edge", "related_to")
            weight = float(c.get("weight", 0.5))
            try:
                connections.append(Connection(target=target, edge=EdgeType(edge), weight=weight))
            except Exception:
                continue

        node = MemoryNode(
            id=mem["node_id"],
            type=NodeType(mem.get("type", "fact")),
            tier=Tier(mem.get("tier", "working")),
            title=mem.get("title"),
            content=mem.get("content", ""),
            space=mem.get("space"),
            tags=mem.get("tags", []),
            connections=connections,
            source="eval:corpus",
            confidence=float(mem.get("confidence", 1.0)),
            importance=float(mem.get("importance", 0.5)),
            stability=float(mem.get("stability", 1.0)),
            access_count=int(mem.get("access_count", 0)),
            created=created or datetime.now(timezone.utc),
            updated=updated or datetime.now(timezone.utc),
            last_accessed=last_accessed or (updated or datetime.now(timezone.utc)),
            last_review=last_review,
            valid_until=valid_until,
        )
        path = engine.file_store.save(node)
        engine.builder.index_single(path)
        engine._index_embedding(node)


def clear_eval_db(engine, *, preserve_self: bool = False) -> None:
    """Remove all nodes from the eval DB and file store.

    When *preserve_self* is True, a fresh self node is ensured after clearing
    so eval cases can catch self-node injection regressions.
    """
    nodes_dir = engine.file_store.nodes_dir
    for md_file in nodes_dir.glob("*.md"):
        md_file.unlink()
    engine.file_store._id_cache.clear()
    engine.file_store._cache_built = False
    engine.builder.full_rebuild()
    with engine.db.transaction() as conn:
        # full_rebuild clears nodes/edges/fts; we also clear tables that can
        # leak state across runs and affect scoring (affinity boost, logs, etc).
        # Keep whisper_log before retrieval_events: the FK uses ON DELETE RESTRICT.
        for table in (
            "node_vectors",
            "whisper_log",
            "retrieval_events",
            "whisper_decisions",
            "affinity",
            "review_log",
            "audit_log",
            "auto_link_checked",
        ):
            try:
                conn.execute(f"DELETE FROM {table}")
            except Exception:
                pass

    if preserve_self:
        # Recreate self node (engine.startup() does this normally).
        ensure = getattr(engine, "_ensure_self_node", None)
        if callable(ensure):
            ensure()

"""Parse and serialize memory nodes as markdown files with YAML frontmatter."""

from __future__ import annotations

from datetime import datetime, timezone

import frontmatter

from ormah.models.node import Connection, EdgeType, MemoryNode, NodeType, Tier


def parse_node(text: str) -> MemoryNode:
    """Parse a markdown string with YAML frontmatter into a MemoryNode."""
    post = frontmatter.loads(text)
    meta = post.metadata

    connections = []
    for conn in meta.get("connections", []):
        connections.append(
            Connection(
                target=conn["target"],
                edge=EdgeType(conn.get("edge", "related_to")),
                weight=conn.get("weight", 0.5),
            )
        )

    valid_until_raw = meta.get("valid_until")
    valid_until = _parse_dt(valid_until_raw) if valid_until_raw else None

    return MemoryNode(
        id=meta["id"],
        type=NodeType(meta["type"]),
        tier=Tier(meta.get("tier", "working")),
        source=meta.get("source", "agent:unknown"),
        created=_parse_dt(meta["created"]),
        updated=_parse_dt(meta["updated"]),
        last_accessed=_parse_dt(meta.get("last_accessed", meta["updated"])),
        access_count=meta.get("access_count", 0),
        confidence=meta.get("confidence", 1.0),
        importance=meta.get("importance", 0.5),
        stability=meta.get("stability", 1.0),
        last_review=_parse_dt(meta["last_review"]) if meta.get("last_review") else None,
        valid_until=valid_until,
        space=meta.get("space"),
        tags=meta.get("tags", []),
        connections=connections,
        title=meta.get("title"),
        content=post.content,
    )


def serialize_node(node: MemoryNode) -> str:
    """Serialize a MemoryNode into markdown with YAML frontmatter."""
    meta: dict = {
        "id": node.id,
        "type": node.type.value,
        "tier": node.tier.value,
        "source": node.source,
        "created": _format_dt(node.created),
        "updated": _format_dt(node.updated),
        "last_accessed": _format_dt(node.last_accessed),
        "access_count": node.access_count,
        "confidence": node.confidence,
        "importance": node.importance,
        "stability": node.stability,
    }

    if node.last_review is not None:
        meta["last_review"] = _format_dt(node.last_review)
    if node.valid_until is not None:
        meta["valid_until"] = _format_dt(node.valid_until)
    if node.title:
        meta["title"] = node.title
    if node.space:
        meta["space"] = node.space
    if node.tags:
        meta["tags"] = node.tags
    if node.connections:
        meta["connections"] = [
            {"target": c.target, "edge": c.edge.value, "weight": c.weight}
            for c in node.connections
        ]

    post = frontmatter.Post(content=node.content, **meta)
    return frontmatter.dumps(post)


def _parse_dt(value: str | datetime) -> datetime:
    if isinstance(value, datetime):
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value
    return datetime.fromisoformat(str(value).replace("Z", "+00:00"))


def _format_dt(dt: datetime) -> str:
    return dt.isoformat().replace("+00:00", "Z")

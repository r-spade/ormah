"""Search-related models."""

from __future__ import annotations

from pydantic import BaseModel, Field

from ormah.models.node import MemoryNode, NodeType, Tier


class SearchQuery(BaseModel):
    query: str
    types: list[NodeType] | None = None
    tiers: list[Tier] | None = None
    spaces: list[str] | None = None
    tags: list[str] | None = None
    limit: int = Field(default=10, ge=1, le=100)
    created_after: str | None = None
    created_before: str | None = None


class SearchResult(BaseModel):
    node: MemoryNode
    score: float
    source: str = "hybrid"  # "fts", "vector", "hybrid"
    formatted: str = ""

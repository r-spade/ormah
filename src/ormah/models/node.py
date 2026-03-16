"""Core domain models for memory nodes."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from enum import Enum

from pydantic import BaseModel, Field


class NodeType(str, Enum):
    fact = "fact"
    decision = "decision"
    preference = "preference"
    event = "event"
    person = "person"
    project = "project"
    concept = "concept"
    procedure = "procedure"
    goal = "goal"
    observation = "observation"


class Tier(str, Enum):
    core = "core"
    working = "working"
    archival = "archival"


class EdgeType(str, Enum):
    related_to = "related_to"
    supports = "supports"
    contradicts = "contradicts"
    part_of = "part_of"
    derived_from = "derived_from"
    preceded_by = "preceded_by"
    caused_by = "caused_by"
    depends_on = "depends_on"
    instance_of = "instance_of"
    defines = "defines"
    evolved_from = "evolved_from"


class Connection(BaseModel):
    target: str
    edge: EdgeType = EdgeType.related_to
    weight: float = Field(default=0.5, ge=0.0, le=1.0)


class MemoryNode(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    type: NodeType
    tier: Tier = Tier.working
    source: str = "agent:unknown"
    created: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    last_accessed: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    access_count: int = 0
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    importance: float = Field(default=0.5, ge=0.0, le=1.0)
    stability: float = Field(default=1.0, ge=0.0)  # FSRS: days until ~37% retrievability
    last_review: datetime | None = None  # last stability update (distinct from last_accessed)
    valid_until: datetime | None = None
    space: str | None = None
    tags: list[str] = Field(default_factory=list)
    connections: list[Connection] = Field(default_factory=list)
    title: str | None = None
    content: str = ""

    @property
    def short_id(self) -> str:
        return self.id.split("-")[0]


class CreateNodeRequest(BaseModel):
    content: str
    type: NodeType = NodeType.fact
    tier: Tier = Tier.working
    source: str | None = None
    space: str | None = None
    tags: list[str] = Field(default_factory=list)
    connections: list[Connection] = Field(default_factory=list)
    title: str | None = None
    about_self: bool = False
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)


class UpdateNodeRequest(BaseModel):
    content: str | None = None
    type: NodeType | None = None
    tier: Tier | None = None
    space: str | None = None
    tags: list[str] | None = None
    add_connections: list[Connection] | None = None
    title: str | None = None
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    valid_until: datetime | None = None


class ConnectRequest(BaseModel):
    source_id: str
    target_id: str
    edge: EdgeType = EdgeType.related_to
    weight: float = Field(default=0.5, ge=0.0, le=1.0)

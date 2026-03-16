"""Proposal models for merge/conflict/decay actions."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from enum import Enum

from pydantic import BaseModel, Field


class ProposalType(str, Enum):
    merge = "merge"
    conflict = "conflict"
    decay = "decay"


class ProposalStatus(str, Enum):
    pending = "pending"
    approved = "approved"
    rejected = "rejected"


class Proposal(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    type: ProposalType
    status: ProposalStatus = ProposalStatus.pending
    source_nodes: list[str]
    proposed_action: str
    reason: str | None = None
    created: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    resolved: datetime | None = None


class ResolveProposalRequest(BaseModel):
    action: ProposalStatus  # approved or rejected

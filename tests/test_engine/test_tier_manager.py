"""Tests for the tier manager."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from ormah.engine.tier_manager import TierManager
from ormah.models.node import MemoryNode, NodeType, Tier


def test_enforce_core_cap_demotes_least_important():
    """With cap=2 and 3 core nodes, the least-important one should be demoted."""
    tm = TierManager(core_cap=2)

    nodes = [
        MemoryNode(
            id="node-high",
            type=NodeType.fact,
            tier=Tier.core,
            content="High importance",
            importance=0.9,
        ),
        MemoryNode(
            id="node-low",
            type=NodeType.fact,
            tier=Tier.core,
            content="Low importance",
            importance=0.3,
        ),
        MemoryNode(
            id="node-mid",
            type=NodeType.fact,
            tier=Tier.core,
            content="Mid importance",
            importance=0.7,
        ),
    ]

    demoted = tm.enforce_core_cap(nodes)

    assert len(demoted) == 1
    assert demoted[0].id == "node-low"
    assert demoted[0].tier == Tier.working

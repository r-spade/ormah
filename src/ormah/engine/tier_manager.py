"""Tier promotion/demotion and core cap enforcement."""

from __future__ import annotations

import logging
from datetime import datetime, timezone

from ormah.models.node import MemoryNode, Tier

logger = logging.getLogger(__name__)


class TierManager:
    """Manages tier transitions and enforces the core memory cap."""

    def __init__(self, core_cap: int = 50) -> None:
        self.core_cap = core_cap

    def promote(self, node: MemoryNode, target_tier: Tier) -> bool:
        """Promote a node to a higher tier. Returns True if promoted."""
        tier_order = [Tier.archival, Tier.working, Tier.core]
        current_idx = tier_order.index(node.tier)
        target_idx = tier_order.index(target_tier)

        if target_idx <= current_idx:
            return False

        node.tier = target_tier
        node.updated = datetime.now(timezone.utc)
        return True

    def demote(self, node: MemoryNode, target_tier: Tier) -> bool:
        """Demote a node to a lower tier. Returns True if demoted."""
        tier_order = [Tier.archival, Tier.working, Tier.core]
        current_idx = tier_order.index(node.tier)
        target_idx = tier_order.index(target_tier)

        if target_idx >= current_idx:
            return False

        node.tier = target_tier
        node.updated = datetime.now(timezone.utc)
        return True

    def enforce_core_cap(
        self,
        core_nodes: list[MemoryNode],
        protected_ids: set[str] | None = None,
    ) -> list[MemoryNode]:
        """If core nodes exceed the cap, demote least-accessed ones to working.
        Nodes in protected_ids are never demoted. Returns list of demoted nodes."""
        if len(core_nodes) <= self.core_cap:
            return []

        protected = protected_ids or set()

        # Sort by importance ascending — least important nodes get demoted first
        sorted_nodes = sorted(
            core_nodes,
            key=lambda n: n.importance,
        )

        excess = len(core_nodes) - self.core_cap
        demoted = []
        for node in sorted_nodes:
            if len(demoted) >= excess:
                break
            if node.id in protected:
                continue
            self.demote(node, Tier.working)
            demoted.append(node)
            logger.info("Demoted core node %s to working (cap enforcement)", node.id[:8])

        return demoted

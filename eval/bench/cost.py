"""API billing uses reported tokens; subscription dollar figures are estimates."""

from __future__ import annotations

import threading

# USD / million tokens. Pinned price table, not a claim about future pricing.
# https://platform.claude.com/docs/en/about-claude/pricing
PRICES = {
    "claude-haiku-4-5": (1.0, 5.0),
    "claude-haiku-4-5-20251001": (1.0, 5.0),
    "claude-sonnet-4-5": (3.0, 15.0),
    "claude-sonnet-4-6": (3.0, 15.0),
}


class BudgetExceeded(RuntimeError):
    pass


def token_cost(model: str, usage: dict) -> float:
    if model not in PRICES:
        raise ValueError(f"No price for {model}; add a verified price before API use")
    inp, out = PRICES[model]
    one_hour_writes = (usage.get("cache_creation") or {}).get("ephemeral_1h_input_tokens") or 0
    return (
        (usage.get("input_tokens") or 0) * inp
        + (usage.get("output_tokens") or 0) * out
        + (usage.get("cache_read_input_tokens") or 0) * inp * 0.1
        + (usage.get("cache_creation_input_tokens") or 0) * inp * 1.25
        + one_hour_writes * inp * 0.75
    ) / 1_000_000


class Budget:
    def __init__(self, max_usd: float, spent: float = 0):
        self.max_usd = max_usd
        self.spent = spent
        # API calls are serialized so parallel workers cannot overspend unseen.
        self.lock = threading.Lock()

    def check(self, reserve: float = 0) -> None:
        if self.spent >= self.max_usd or self.spent + reserve > self.max_usd:
            raise BudgetExceeded(
                f"--max-usd {self.max_usd:.4f}: spent ${self.spent:.6f}, "
                f"next-call estimate ${reserve:.6f}"
            )

    def add(self, usd: float) -> None:
        self.spent += usd
        self.check()

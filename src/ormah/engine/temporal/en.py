"""The built-in English temporal locale pack.

Every phrase here carries the priority of its index in the classifier's
keyword table, which is what preserves today's behaviour on a compound prompt:
that table interleaves the two languages, so ``ontem`` (priority 3) beats
``last month`` (priority 8) in "compare ontem with last month". Concatenating
packs in registration order instead would flip it.
"""

from __future__ import annotations

import re

from ormah.engine.temporal.locale import StaticPhrase, TemporalLocale

LOCALE = TemporalLocale(
    code="en",
    static_phrases=(
        StaticPhrase(
            pattern=re.compile(r"\btoday\b", re.IGNORECASE),
            probe="what did we do today",
            priority=0,
            window=(1, None),  # 24h ago -> now
        ),
        StaticPhrase(
            pattern=re.compile(r"\byesterday\b", re.IGNORECASE),
            probe="what did we do yesterday",
            priority=2,
            window=(2, 1),  # 48h ago -> 24h ago
        ),
        StaticPhrase(
            pattern=re.compile(r"\blast\s+week\b", re.IGNORECASE),
            probe="what happened last week",
            priority=4,
            window=(14, 7),  # 14d ago -> 7d ago
        ),
        StaticPhrase(
            pattern=re.compile(r"\bthis\s+week\b", re.IGNORECASE),
            probe="what happened this week",
            priority=6,
            window=(7, None),  # 7d ago -> now
        ),
        StaticPhrase(
            pattern=re.compile(r"\blast\s+month\b", re.IGNORECASE),
            probe="what happened last month",
            priority=8,
            window=(60, 30),  # 60d ago -> 30d ago
        ),
        StaticPhrase(
            pattern=re.compile(r"\brecently\b|\blately\b", re.IGNORECASE),
            probe="what changed recently",
            priority=10,
            window=(3, None),  # 3d ago -> now
        ),
        # Strip-only: "recent" lives in today's strip list and not in the
        # keyword table, so it is removed from the query but never selects a
        # window and never makes temporal detection true. Sharing priority 10
        # with "recently|lately" keeps it where today's single alternation put
        # it in the strip order.
        StaticPhrase(
            pattern=re.compile(r"\brecent\b", re.IGNORECASE),
            probe="show me recent changes",
            priority=10,
        ),
    ),
    numeric_pattern=re.compile(
        r"\b(?:last|past)\s+(\d+)\s+(hours?|days?|weeks?|months?)\b",
        re.IGNORECASE,
    ),
    # English units are already the canonical keys.
    unit_aliases={},
    cleanup_patterns=(
        re.compile(r"\b(?:in|during|from|over|for)\s+(?:the\s+)?", re.IGNORECASE),
    ),
)

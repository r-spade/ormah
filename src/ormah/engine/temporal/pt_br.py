"""The built-in Brazilian Portuguese temporal locale pack.

Priorities are the indices these phrases hold in the classifier's keyword
table today — see the note in :mod:`ormah.engine.temporal.en`.
"""

from __future__ import annotations

import re

from ormah.engine.temporal.locale import StaticPhrase, TemporalLocale

LOCALE = TemporalLocale(
    code="pt-BR",
    static_phrases=(
        StaticPhrase(
            pattern=re.compile(r"\bhoje\b", re.IGNORECASE),
            probe="o que fizemos hoje",
            priority=1,
            window=(1, None),  # 24h atrás -> agora
        ),
        StaticPhrase(
            pattern=re.compile(r"\bontem\b", re.IGNORECASE),
            probe="o que fizemos ontem",
            priority=3,
            window=(2, 1),  # 48h atrás -> 24h atrás
        ),
        StaticPhrase(
            pattern=re.compile(r"\bsemana\s+passada\b", re.IGNORECASE),
            probe="o que fizemos na semana passada",
            priority=5,
            window=(14, 7),
        ),
        StaticPhrase(
            pattern=re.compile(r"\b(?:esta|essa|nesta|nessa)\s+semana\b", re.IGNORECASE),
            probe="o que fizemos nesta semana",
            priority=7,
            window=(7, None),
        ),
        StaticPhrase(
            pattern=re.compile(r"\bm[êe]s\s+passado\b", re.IGNORECASE),
            probe="o que fizemos no mês passado",
            priority=9,
            window=(60, 30),
        ),
        StaticPhrase(
            pattern=re.compile(r"\brecentemente\b|\bultimamente\b", re.IGNORECASE),
            probe="o que fizemos recentemente",
            priority=11,
            window=(3, None),
        ),
    ),
    # Only this language's determiner and units: a pattern shared with the en
    # pack would make both packs claim every numeric match, which is what makes
    # cleanup scoping untestable ("please say no last 3 days"). "meses" is
    # listed before "m[êe]s" so the plural matches whole.
    numeric_pattern=re.compile(
        r"\b(?:[úu]ltim[oa]s?)\s+(\d+)\s+(horas?|dias?|semanas?|meses|m[êe]s)\b",
        re.IGNORECASE,
    ),
    # Applied by the parser after lowercasing and dropping the plural "s".
    # Without these the canonical unit->days map falls back to 1 day, so
    # "últimas 2 semanas" would mean 2 days and the rolling-window branch
    # (which tests for "week"/"month") would never fire.
    unit_aliases={
        "hora": "hour",
        "dia": "day",
        "semana": "week",
        "mese": "month",  # "meses" -> "mese"
        "mê": "month",  # "mês" -> "mê"
        "me": "month",  # "mes" -> "me"
    },
    # Both boundaries are written out. The trailing `\b` is redundant *here*
    # — a word character followed by `\s` already implies one — but the
    # parser's flush rule does not supply it: that rule only requires the match
    # to END at the vacated span, so a cleanup pattern not ending in `\s+`
    # would happily cut a word in half ("ano" + gap, with a bare `no`, leaves
    # "a"). A pack author owes their own boundaries on both sides.
    cleanup_patterns=(re.compile(r"\b(?:na|no|nos|nas|em)\b\s+", re.IGNORECASE),),
)

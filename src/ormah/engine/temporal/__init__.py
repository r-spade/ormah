"""Temporal locale packs and the registry that resolves the enabled ones."""

from __future__ import annotations

from ormah.engine.temporal.locale import (
    StaticPhrase,
    TemporalLocale,
    parse_locale_codes,
    register,
    registered_codes,
    resolve_locales,
)
from ormah.engine.temporal import en, pt_br

# Registration order fixes the tie-break between two static entries of equal
# priority; it is not the window-selection precedence.
register(en.LOCALE)
register(pt_br.LOCALE)

__all__ = [
    "StaticPhrase",
    "TemporalLocale",
    "parse_locale_codes",
    "register",
    "registered_codes",
    "resolve_locales",
]

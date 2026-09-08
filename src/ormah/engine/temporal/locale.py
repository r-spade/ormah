"""Data model for a temporal locale pack.

A pack is the whole temporal grammar of one language, declared as data: the
static phrases it recognises, its own numeric expression, the unit aliases that
fold its unit words onto the canonical English keys the parser's unit->days map
is written in, and the post-strip cleanup patterns for that language.

Grammar is never user-configurable — a pack is code with tests, not a regex an
operator types into ``.env``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field


@dataclass(frozen=True)
class StaticPhrase:
    """One declared phrase — the single source of truth for both recognition and stripping."""

    pattern: re.Pattern
    """The regex that recognises the phrase."""

    probe: str
    """One concrete prompt string this entry's pattern matches.

    A pattern such as ``(?:esta|essa|nesta|nessa)\\s+semana`` is not itself a
    prompt; its probe ("o que fizemos nesta semana") is. Tests drive the entry
    through the public functions with it instead of inspecting derived tables.
    """

    priority: int
    """Where the entry sits in the merged match order, across every enabled pack."""

    window: tuple[int, int | None] | None = None
    """``(days_start, days_end)``, where ``days_end = None`` extends the window to now.

    ``None`` makes the entry **strip-only**: it is removed from the query but
    never selects a window and never makes temporal detection true.
    """

    @property
    def is_strip_only(self) -> bool:
        return self.window is None


@dataclass(frozen=True)
class TemporalLocale:
    """One language's temporal grammar."""

    code: str
    """The pack's identifier as it appears in ``ORMAH_TEMPORAL_LOCALES`` (``en``, ``pt-BR``)."""

    static_phrases: tuple[StaticPhrase, ...] = ()

    numeric_pattern: re.Pattern | None = None
    """This language's numeric expression, matching only its own determiners and units.

    The capture-group contract is fixed and shared across packs, because the
    parser reads the groups: the determiner prefix is non-capturing, **group 1
    is the count** and **group 2 is the unit lexeme**.
    """

    unit_aliases: dict[str, str] = field(default_factory=dict)
    """This language's unit words folded onto the canonical English keys. Empty for ``en``."""

    cleanup_patterns: tuple[re.Pattern, ...] = ()
    """Dangling text this language leaves immediately to the left of a removed phrase.

    These carry no end-of-residue lookahead: the parser anchors them to the
    vacated span and enforces the "end of residue, or immediately before a
    comma" boundary against the final residue itself.
    """

    @property
    def windowed_phrases(self) -> tuple[StaticPhrase, ...]:
        """The entries that declare a window — the keyword table, derived."""
        return tuple(p for p in self.static_phrases if not p.is_strip_only)

    @property
    def strip_patterns(self) -> tuple[re.Pattern, ...]:
        """Every entry's pattern plus the numeric one — the strip table, derived.

        There is no second hand-written list, so a phrase can never be
        recognised for the window and forgotten for the strip.
        """
        patterns = [p.pattern for p in self.static_phrases]
        if self.numeric_pattern is not None:
            patterns.append(self.numeric_pattern)
        return tuple(patterns)


# --- Registry -------------------------------------------------------------
#
# Built-in packs register themselves when ``ormah.engine.temporal`` is
# imported. The registry decides *which* packs are consulted; it never decides
# which window a prompt gets — that is the parser's match order.

_REGISTRY: dict[str, TemporalLocale] = {}


def register(locale: TemporalLocale) -> None:
    """Register *locale* under its code, replacing any pack with the same code."""
    _REGISTRY[locale.code] = locale


def registered_codes() -> tuple[str, ...]:
    """Every registered pack code, in registration order."""
    return tuple(_REGISTRY)


def resolve_locales(codes: tuple[str, ...]) -> tuple[TemporalLocale, ...]:
    """Return the packs named by *codes*, in that order.

    Raises :class:`ValueError` on a code no pack is registered under. Codes are
    compared case-sensitively, so ``pt-br`` is a typo rather than a match.
    """
    unknown = [code for code in codes if code not in _REGISTRY]
    if unknown:
        raise ValueError(
            f"unknown temporal locale(s) {unknown}; registered: {list(_REGISTRY)}"
        )
    return tuple(_REGISTRY[code] for code in codes)


def parse_locale_codes(value: str) -> tuple[str, ...]:
    """Parse the comma-separated ``ORMAH_TEMPORAL_LOCALES`` form into ordered codes.

    Tokens are split on ``,``, trimmed, and empty tokens dropped; duplicates are
    dropped keeping first-seen order. An empty result, or a code no pack is
    registered under, raises :class:`ValueError` rather than silently disabling
    temporal parsing.
    """
    codes = tuple(dict.fromkeys(token.strip() for token in value.split(",") if token.strip()))
    if not codes:
        raise ValueError(f"temporal_locales must name at least one locale, got {value!r}")
    resolve_locales(codes)
    return codes

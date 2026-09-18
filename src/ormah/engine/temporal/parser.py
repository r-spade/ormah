"""The language-agnostic temporal parser.

Owns every rule that is not language-specific — match ordering, window
arithmetic, the canonical unit->days map, the rolling previous-period rule and
the default window — and asks the enabled packs what a phrase means instead of
holding the phrases itself.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from datetime import datetime, timedelta, timezone

from ormah.engine.temporal.locale import StaticPhrase, TemporalLocale

_UNIT_TO_DAYS: dict[str, float] = {"hour": 1 / 24, "day": 1, "week": 7, "month": 30}

# The end-of-residue-or-before-a-comma boundary a cleanup is allowed to fire at.
_BOUNDARY_RE = re.compile(r"\s*$|\s*,")

_DEFAULT_TEMPORAL_DAYS = 3


class TemporalParser:
    """Resolves a prompt into a recall window and a topical residue."""

    def __init__(self, locales: Sequence[TemporalLocale]) -> None:
        self._locales = tuple(locales)
        # Static entries from every pack merge into one sequence ordered by
        # priority, with pack order as the stable tie-break, so no window
        # depends on which packs are enabled or in what order.
        self._static: tuple[StaticPhrase, ...] = tuple(
            sorted(
                (p for locale in self._locales for p in locale.windowed_phrases),
                key=lambda phrase: phrase.priority,
            )
        )

    def has_temporal_phrases(self, prompt: str) -> bool:
        """Whether *prompt* carries an explicit time reference.

        Derived from the windowed entries plus the numeric patterns only, never
        from the strip list: a strip-only phrase is removed from the query and
        stays invisible here, so it never starts date-filtering a recall.
        """
        return self._numeric_match(prompt) is not None or any(
            phrase.pattern.search(prompt) for phrase in self._static
        )

    def extract_time_params(self, prompt: str) -> dict:
        """Return the ``created_after``/``created_before`` window for *prompt*."""
        now = datetime.now(timezone.utc)

        found = self._numeric_match(prompt)
        if found is not None:
            locale, match = found
            count = int(match.group(1))
            # Lowercase before dropping the plural so an uppercased unit still
            # normalises ("DIAS" -> "dias" -> "dia"), then fold the pack's unit
            # word onto the canonical English key this map is written in.
            unit = match.group(2).lower().rstrip("s")
            unit = locale.unit_aliases.get(unit, unit)
            days = count * _UNIT_TO_DAYS.get(unit, 1)

            # Rolling previous-period for weeks/months with N > 1: "last 2
            # weeks" is 4 weeks ago -> 2 weeks ago. Days and hours extend to now.
            if unit in ("week", "month") and count > 1:
                return _window(now, days * 2, days)
            return _window(now, days, 0)

        for phrase in self._static:
            if phrase.pattern.search(prompt):
                days_start, days_end = phrase.window
                return _window(now, days_start, days_end or 0)

        return _window(now, _DEFAULT_TEMPORAL_DAYS, 0)

    def _numeric_match(self, prompt: str) -> tuple[TemporalLocale, re.Match] | None:
        """The numeric match with the smallest offset in *prompt*, ties by pack order.

        Every enabled pack is searched, because "first pack that matched" would
        let the enabled order pick the window for "compare últimas 2 semanas
        with last 3 days".
        """
        best: tuple[TemporalLocale, re.Match] | None = None
        for locale in self._locales:
            if locale.numeric_pattern is None:
                continue
            match = locale.numeric_pattern.search(prompt)
            if match is not None and (best is None or match.start() < best[1].start()):
                best = (locale, match)
        return best

    def strip_temporal_phrases(self, prompt: str) -> str:
        """Remove every enabled pack's temporal phrases, returning the topical residue."""
        residue, gaps = self._remove(prompt)
        residue = self._clean(residue, gaps)
        return re.sub(r"\s{2,}", " ", residue).strip()

    def _remove(self, prompt: str) -> tuple[str, list[tuple[int, TemporalLocale]]]:
        """Apply every strip pattern by re-searching the *current* residue.

        Returns the residue and, per removal, the position of the gap it left
        in that final residue plus the pack that owned it. Collecting the spans
        against the original prompt and deleting them afterwards is not the
        same thing: the patterns overlap on "semana", and right-to-left
        deletion turns "nesta semana passada discutimos autenticação" into
        "timos autenticação".
        """
        residue = prompt
        gaps: list[tuple[int, TemporalLocale]] = []
        for locale in self._locales:
            for pattern in locale.strip_patterns:
                # Resuming the scan at the junction is what `sub` does: a match
                # can never reach back across the text just removed.
                pos = 0
                while (match := pattern.search(residue, pos)) is not None:
                    start, end = match.span()
                    if start == end:
                        break
                    residue = residue[:start] + residue[end:]
                    width = end - start
                    # ponytail: every recorded gap is re-offset per removal, so the
                    # bookkeeping is O(removals^2). A prompt carries a handful;
                    # swap for a segment list if that ever stops being true.
                    gaps = [(g - width if g > start else g, owner) for g, owner in gaps]
                    gaps.append((start, locale))
                    pos = start
        return residue, gaps

    def _clean(self, residue: str, gaps: list[tuple[int, TemporalLocale]]) -> str:
        """Run each owning pack's cleanup patterns at the span it vacated.

        A pattern fires only when it matches immediately to the **left** of the
        gap and, in this final residue, the gap sits at the end or immediately
        before a comma — the boundary today's English lookahead demands, which
        the packs deliberately do not carry because a lookahead on a
        left-context slice cannot see it.
        """
        cuts = []
        for gap, locale in gaps:
            if _BOUNDARY_RE.match(residue, gap) is None:
                continue
            for pattern in locale.cleanup_patterns:
                # The match must end flush with the gap; a plain `search` would
                # take the first match anywhere to its left instead ("mudanças
                # na API nos últimos 3 dias" -> "mudanças API nos").
                match = next(
                    (m for m in pattern.finditer(residue, 0, gap) if m.end() == gap), None
                )
                if match is not None:
                    cuts.append((match.start(), gap))
                    break
        # Right to left, so a cut never invalidates the ones still pending; a
        # gap swallowed by a wider cut is dropped rather than cut again.
        limit = len(residue)
        for start, end in sorted(cuts, reverse=True):
            if end > limit:
                continue
            residue = residue[:start] + residue[end:]
            limit = start
        return residue


def _window(now: datetime, days_start: float, days_end: float) -> dict:
    return {
        "created_after": (now - timedelta(days=days_start)).isoformat(),
        "created_before": (now - timedelta(days=days_end)).isoformat(),
    }

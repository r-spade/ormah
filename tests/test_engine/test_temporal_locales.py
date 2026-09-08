"""Tests for the temporal locale data model and registry."""

from __future__ import annotations

import re

import pytest

from ormah.config import Settings
from ormah.engine.temporal import (
    StaticPhrase,
    TemporalLocale,
    registered_codes,
    resolve_locales,
)
from ormah.engine.prompt_classifier import _TIME_KEYWORDS


class TestStaticPhrase:
    def test_windowed_entry_carries_its_window(self):
        phrase = StaticPhrase(
            pattern=re.compile(r"\byesterday\b", re.IGNORECASE),
            probe="what did we do yesterday",
            priority=2,
            window=(2, 1),
        )
        assert phrase.window == (2, 1)
        assert phrase.is_strip_only is False

    def test_open_ended_window_uses_none_as_its_end(self):
        phrase = StaticPhrase(
            pattern=re.compile(r"\btoday\b", re.IGNORECASE),
            probe="what did we do today",
            priority=0,
            window=(1, None),
        )
        assert phrase.window == (1, None)
        assert phrase.is_strip_only is False

    def test_entry_without_a_window_is_strip_only(self):
        phrase = StaticPhrase(
            pattern=re.compile(r"\brecent\b", re.IGNORECASE),
            probe="show me recent changes",
            priority=10,
        )
        assert phrase.window is None
        assert phrase.is_strip_only is True

    def test_probe_and_priority_are_carried_verbatim(self):
        phrase = StaticPhrase(
            pattern=re.compile(r"\b(?:esta|essa|nesta|nessa)\s+semana\b", re.IGNORECASE),
            probe="o que fizemos nesta semana",
            priority=7,
            window=(7, None),
        )
        assert phrase.priority == 7
        assert phrase.probe == "o que fizemos nesta semana"
        assert phrase.pattern.search(phrase.probe) is not None


class TestRegistry:
    def test_built_in_packs_are_registered(self):
        assert registered_codes() == ("en", "pt-BR")

    def test_resolve_returns_packs_in_the_order_the_codes_name_them(self):
        assert [loc.code for loc in resolve_locales(("pt-BR", "en"))] == ["pt-BR", "en"]
        assert [loc.code for loc in resolve_locales(("en", "pt-BR"))] == ["en", "pt-BR"]

    def test_resolve_with_a_single_code_returns_only_that_pack(self):
        assert [loc.code for loc in resolve_locales(("en",))] == ["en"]

    def test_resolve_raises_on_an_unknown_code(self):
        with pytest.raises(ValueError, match="unknown temporal locale"):
            resolve_locales(("klingon",))

    def test_resolves_the_packs_the_setting_names(self, monkeypatch):
        monkeypatch.setenv("ORMAH_TEMPORAL_LOCALES", "pt-BR,en")
        settings = Settings(memory_dir="/tmp/ormah_test")
        resolved = resolve_locales(settings.temporal_locale_codes)
        assert [loc.code for loc in resolved] == ["pt-BR", "en"]


def _all_phrases():
    """Every static phrase declared by the two built-in packs, with its pack code."""
    return [
        (locale.code, phrase)
        for locale in resolve_locales(("en", "pt-BR"))
        for phrase in locale.static_phrases
    ]


def _pack(code: str) -> TemporalLocale:
    return resolve_locales((code,))[0]


class TestBuiltInPacksCoverTodaysKeywordTable:
    def test_every_keyword_entry_is_declared_by_exactly_one_pack(self):
        for index, (pattern, days_start, days_end) in enumerate(_TIME_KEYWORDS):
            declared = [
                (code, phrase)
                for code, phrase in _all_phrases()
                if phrase.pattern.pattern == pattern.pattern
            ]
            assert len(declared) == 1, f"{pattern.pattern!r} declared {len(declared)} times"
            _, phrase = declared[0]
            assert phrase.priority == index, f"{pattern.pattern!r} priority"
            assert phrase.window == (days_start, days_end), f"{pattern.pattern!r} window"

    def test_no_pack_declares_a_windowed_phrase_outside_the_keyword_table(self):
        table = {pattern.pattern for pattern, _, _ in _TIME_KEYWORDS}
        declared = {
            phrase.pattern.pattern
            for _, phrase in _all_phrases()
            if not phrase.is_strip_only
        }
        assert declared == table

    def test_recent_is_the_only_strip_only_entry_and_lives_in_the_en_pack(self):
        strip_only = [(code, phrase) for code, phrase in _all_phrases() if phrase.is_strip_only]
        assert [code for code, _ in strip_only] == ["en"]
        phrase = strip_only[0][1]
        assert phrase.window is None
        assert phrase.pattern.search("show me recent changes") is not None
        assert phrase.pattern.search("what changed recently") is None

    def test_every_probe_is_matched_by_its_own_pattern(self):
        for code, phrase in _all_phrases():
            assert phrase.pattern.search(phrase.probe) is not None, (
                f"{code}: probe {phrase.probe!r} not matched by {phrase.pattern.pattern!r}"
            )


class TestBuiltInNumericPatterns:
    def test_capture_groups_are_count_then_unit_in_en(self):
        match = _pack("en").numeric_pattern.search("changes to the API in the last 3 days")
        assert match is not None
        assert match.groups() == ("3", "days")

    def test_capture_groups_are_count_then_unit_in_pt_br(self):
        match = _pack("pt-BR").numeric_pattern.search("resumo das últimas 2 semanas")
        assert match is not None
        # A capturing determiner — ``([úu]ltim[oa]s?)`` — shifts every group and
        # would make the parser read "últimas" as the count.
        assert match.groups() == ("2", "semanas")

    @pytest.mark.parametrize(
        "prompt",
        ["resumo das últimas 2 semanas", "last 2 semanas", "últimas 2 weeks"],
    )
    def test_en_numeric_pattern_ignores_pt_br_forms(self, prompt):
        assert _pack("en").numeric_pattern.search(prompt) is None

    @pytest.mark.parametrize(
        "prompt",
        ["changes in the last 3 days", "past 2 weeks", "last 2 semanas", "últimas 2 weeks"],
    )
    def test_pt_br_numeric_pattern_ignores_en_forms(self, prompt):
        assert _pack("pt-BR").numeric_pattern.search(prompt) is None


class TestBuiltInUnitAliases:
    def test_en_pack_needs_no_aliases(self):
        assert _pack("en").unit_aliases == {}

    @pytest.mark.parametrize(
        "prompt, canonical",
        [
            ("últimas 3 horas", "hour"),
            ("última 1 hora", "hour"),
            ("últimos 5 dias", "day"),
            ("último 1 dia", "day"),
            ("últimas 2 semanas", "week"),
            ("última 1 semana", "week"),
            ("últimos 3 meses", "month"),
            ("último 1 mês", "month"),
            ("último 1 mes", "month"),
        ],
    )
    def test_pt_br_units_fold_onto_the_canonical_english_keys(self, prompt, canonical):
        pack = _pack("pt-BR")
        match = pack.numeric_pattern.search(prompt)
        assert match is not None, prompt
        # The normalisation the parser applies: lowercase, drop the plural "s",
        # then fold onto the canonical key.
        unit = match.group(2).lower().rstrip("s")
        assert pack.unit_aliases.get(unit, unit) == canonical


def _cleanup_claims_tail(code: str, left: str) -> bool:
    """Whether *code*'s cleanup claims the text immediately left of a vacated span.

    A cleanup pattern only fires when it matches flush against the end of the
    left-hand context — the parser anchors it to the removed span rather than
    running it over the whole residue.
    """
    return any(
        match.end() == len(left)
        for pattern in _pack(code).cleanup_patterns
        for match in pattern.finditer(left)
    )


class TestBuiltInCleanupPatterns:
    @pytest.mark.parametrize(
        "left", ["changes to the API in the ", "work from ", "notes during ", "logs over the "]
    )
    def test_en_cleanup_matches_its_own_dangling_prepositions(self, left):
        assert _cleanup_claims_tail("en", left)

    @pytest.mark.parametrize(
        "left",
        [
            "o que fizemos na ",
            "o que fizemos no ",
            "mudanças na API nos ",
            "reunião nas ",
            "o que mudou em ",
        ],
    )
    def test_pt_br_cleanup_matches_its_own_dangling_contractions(self, left):
        assert _cleanup_claims_tail("pt-BR", left)

    def test_en_cleanup_does_not_claim_pt_br_contractions(self):
        assert not _cleanup_claims_tail("en", "o que fizemos na ")

    def test_pt_br_cleanup_does_not_claim_the_interior_of_a_phrase(self):
        # "mudanças na API" — the mid-sentence "na" is not flush against the
        # vacated span, so nothing claims it.
        assert not _cleanup_claims_tail("pt-BR", "mudanças na API")

    def test_only_the_owning_pack_would_eat_the_english_no(self):
        # "please say no last 3 days": the en pack owns that removal, and
        # English cleanup leaves "no" alone. Running pt-BR cleanup at the same
        # span — which a shared numeric pattern would cause — returns
        # "please say" instead.
        assert not _cleanup_claims_tail("en", "please say no ")
        assert _cleanup_claims_tail("pt-BR", "please say no ")

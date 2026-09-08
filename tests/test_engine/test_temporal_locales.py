"""Tests for the temporal locale data model and registry."""

from __future__ import annotations

import re

import pytest

from ormah.config import Settings
from ormah.engine.temporal import (
    StaticPhrase,
    registered_codes,
    resolve_locales,
)


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

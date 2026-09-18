"""The temporal contract as the memory engine sees it: the module-level functions.

Every assertion here is on one of the two externally observable outputs — the
window dict returned for a prompt, or the residue string returned for a prompt
— plus the boolean from ``has_temporal_phrases``, the only thing that tells a
strip-only phrase from an unrecognised one. Nothing here reads a derived
table, a pattern's identity, or whether a pack was consulted.

``tests/test_engine/conftest.py`` pins ``ORMAH_TEMPORAL_LOCALES=en,pt-BR``
before every test, so these assertions never depend on the operator's ``.env``.
The one class below that changes the setting owns the cache it dirties, in its
own fixture, rather than leaving the conftest to sweep up after it.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from ormah.engine import prompt_classifier
from ormah.engine.prompt_classifier import (
    extract_time_params,
    has_temporal_phrases,
    strip_temporal_phrases,
)
from ormah.engine.temporal import resolve_locales

# The codes the conftest fixture pins; the probe walk below iterates the packs
# they name rather than reading settings at collection time.
PINNED_CODES = ("en", "pt-BR")


def _days_ago(iso: str) -> float:
    return (datetime.now(timezone.utc) - datetime.fromisoformat(iso)).total_seconds() / 86400


# ---------------------------------------------------------------------------
# Moved from tests/test_engine/test_prompt_classifier.py — the PT-BR
# assertions PR #283 added, expectations unchanged. They live here because
# they only mean anything with the enabled packs pinned.
# ---------------------------------------------------------------------------

class TestPtBrTimeExtraction:
    """Each window below must differ from the 3d->now default, so a passing
    test proves the keyword matched rather than falling through.
    """

    def _days_ago(self, iso: str) -> float:
        dt = datetime.fromisoformat(iso)
        return (datetime.now(timezone.utc) - dt).total_seconds() / 86400

    def test_ptbr_ontem_bounded(self):
        """'ontem' narrows to the same 2d -> 1d window as 'yesterday'."""
        params = extract_time_params("o que fizemos ontem")
        assert 1.9 < self._days_ago(params["created_after"]) < 2.1
        assert 0.9 < self._days_ago(params["created_before"]) < 1.1

    def test_ptbr_hoje_extends_to_now(self):
        params = extract_time_params("o que fizemos hoje")
        assert 0.95 < self._days_ago(params["created_after"]) < 1.05
        assert self._days_ago(params["created_before"]) < 0.01

    def test_ptbr_semana_passada_bounded(self):
        params = extract_time_params("o que aconteceu na semana passada")
        assert 13.9 < self._days_ago(params["created_after"]) < 14.1
        assert 6.9 < self._days_ago(params["created_before"]) < 7.1

    def test_ptbr_esta_semana_extends_to_now(self):
        params = extract_time_params("o que fizemos esta semana")
        assert 6.9 < self._days_ago(params["created_after"]) < 7.1
        assert self._days_ago(params["created_before"]) < 0.01

    def test_ptbr_mes_passado_bounded(self):
        params = extract_time_params("o que aconteceu no mês passado")
        assert 59.9 < self._days_ago(params["created_after"]) < 60.1
        assert 29.9 < self._days_ago(params["created_before"]) < 30.1

    def test_ptbr_numeric_days_extend_to_now(self):
        params = extract_time_params("o que mudou nos últimos 4 dias")
        assert 3.9 < self._days_ago(params["created_after"]) < 4.1
        assert self._days_ago(params["created_before"]) < 0.01

    def test_ptbr_numeric_weeks_rolling(self):
        """'últimas 2 semanas' mirrors the EN rolling window: 28d -> 14d."""
        params = extract_time_params("me mostra as últimas 2 semanas")
        assert 27.9 < self._days_ago(params["created_after"]) < 28.1
        assert 13.9 < self._days_ago(params["created_before"]) < 14.1

    def test_ptbr_numeric_months_rolling(self):
        params = extract_time_params("resumo dos últimos 3 meses")
        assert 179.9 < self._days_ago(params["created_after"]) < 180.1
        assert 89.9 < self._days_ago(params["created_before"]) < 90.1

    def test_ptbr_numeric_hours(self):
        params = extract_time_params("o que aconteceu nas últimas 6 horas")
        dt = datetime.fromisoformat(params["created_after"])
        diff_hours = (datetime.now(timezone.utc) - dt).total_seconds() / 3600
        assert 5.9 < diff_hours < 6.1


class TestPtBrHasTemporalPhrases:
    """'recentemente'/'ultimamente' map to 3d -> now, which is also the
    no-keyword default, so extract_time_params cannot tell a match from a
    fallback. has_temporal_phrases can: it skips the default entirely.
    """

    def test_ptbr_recentemente_is_temporal(self):
        assert has_temporal_phrases("houve mudanças recentemente") is True

    def test_ptbr_ultimamente_is_temporal(self):
        assert has_temporal_phrases("o que mudou ultimamente") is True

    def test_non_temporal_prompt_is_not_temporal(self):
        assert has_temporal_phrases("como funciona o pipeline de busca") is False


class TestPtBrStripTemporalPhrases:

    def test_strip_ptbr_ontem(self):
        assert strip_temporal_phrases("o que fizemos ontem") == "o que fizemos"

    def test_strip_ptbr_hoje(self):
        assert strip_temporal_phrases("o que fizemos hoje") == "o que fizemos"

    def test_strip_ptbr_semana_passada(self):
        result = strip_temporal_phrases("o que fizemos no whisper na semana passada")
        assert "whisper" in result
        assert "semana passada" not in result

    def test_strip_ptbr_mes_passado(self):
        result = strip_temporal_phrases("mudanças no auth no mês passado")
        assert "auth" in result
        assert "mês passado" not in result

    def test_strip_ptbr_numeric_days(self):
        result = strip_temporal_phrases("mudanças no auth nos últimos 3 dias")
        assert "auth" in result
        assert "últimos 3 dias" not in result

    def test_strip_ptbr_recentemente(self):
        result = strip_temporal_phrases("mudanças recentemente no auth")
        assert "auth" in result
        assert "recentemente" not in result


# ---------------------------------------------------------------------------
# Single source of truth: every declared phrase reaches both paths
# ---------------------------------------------------------------------------

_PROBES = [
    (locale.code, phrase)
    for locale in resolve_locales(PINNED_CODES)
    for phrase in locale.static_phrases
]


@pytest.mark.parametrize(
    "code, phrase",
    _PROBES,
    ids=[f"{code}:{phrase.probe}" for code, phrase in _PROBES],
)
def test_every_declared_phrase_is_detected_as_declared_and_always_stripped(code, phrase):
    """Drive every enabled pack's probe through the public functions.

    A windowed entry's probe must make ``has_temporal_phrases`` true; a
    strip-only entry's probe must leave it false. Either way the phrase must
    be gone from the residue. The assertions go through the public functions
    and compare against the probe string, so the test cannot pass by comparing
    a derived table to itself — it fails when a phrase reaches one path and
    not the other.
    """
    assert has_temporal_phrases(phrase.probe) is (not phrase.is_strip_only)

    removed = phrase.pattern.search(phrase.probe).group(0)
    assert removed.lower() not in strip_temporal_phrases(phrase.probe).lower()


def test_a_strip_only_phrase_never_starts_date_filtering():
    """The walk above flips its own expectation with the declaration; this does not.

    ``memory_engine`` applies ``created_after`` only when
    ``has_temporal_phrases`` is true, so promoting "recent" to a windowed
    entry — or deriving detection from the strip list — would start filtering
    this prompt to the last three days. Every window assertion stays green
    under both mistakes, because the default window is the same 3 days.
    """
    assert has_temporal_phrases("show me recent changes") is False
    assert strip_temporal_phrases("show me recent changes") == "show me changes"


# ---------------------------------------------------------------------------
# The setting reaches the production call path
# ---------------------------------------------------------------------------

class TestTheSettingGatesThePublicFunctions:
    """No explicit parser anywhere here: this is the path ``memory_engine`` takes.

    Each test sets the environment variable and relies on the default parser
    being rebuilt from a **fresh** ``Settings()``. Reading the import-time
    ``ormah.config.settings`` singleton instead makes every assertion below
    fail, because that singleton was bound before the fixture ran.
    """

    @pytest.fixture(autouse=True)
    def _isolate_the_default_parser(self):
        """Own the process-global cache these tests dirty, on both sides.

        ``monkeypatch`` restores ``ORMAH_TEMPORAL_LOCALES`` at teardown but
        knows nothing about ``_default_parser``'s ``lru_cache``, so without the
        teardown clear the last test here leaves a pt-BR-only parser behind for
        whatever runs next: ``strip_temporal_phrases("work from yesterday")``
        then returns the prompt unchanged. The setup clear is what makes the
        ``setenv`` in each test body take effect at all — the cache is keyed on
        nothing, so a parser built earlier would otherwise be reused.
        """
        prompt_classifier._default_parser.cache_clear()
        yield
        prompt_classifier._default_parser.cache_clear()

    def test_disabling_pt_br_hides_it_from_detection(self, monkeypatch):
        monkeypatch.setenv("ORMAH_TEMPORAL_LOCALES", "en")
        assert has_temporal_phrases("o que fizemos ontem") is False

    def test_disabling_pt_br_stops_it_being_stripped(self, monkeypatch):
        monkeypatch.setenv("ORMAH_TEMPORAL_LOCALES", "en")
        assert (
            strip_temporal_phrases("o que fizemos na semana passada")
            == "o que fizemos na semana passada"
        )

    def test_disabling_pt_br_falls_back_to_the_default_window(self, monkeypatch):
        monkeypatch.setenv("ORMAH_TEMPORAL_LOCALES", "en")
        params = extract_time_params("o que fizemos ontem")
        assert 2.9 < _days_ago(params["created_after"]) < 3.1
        assert _days_ago(params["created_before"]) < 0.1

    def test_english_still_works_with_pt_br_disabled(self, monkeypatch):
        monkeypatch.setenv("ORMAH_TEMPORAL_LOCALES", "en")
        assert has_temporal_phrases("what did we do yesterday") is True

    def test_disabling_english_hides_it_from_detection(self, monkeypatch):
        monkeypatch.setenv("ORMAH_TEMPORAL_LOCALES", "pt-BR")
        assert has_temporal_phrases("what did we do last week") is False
        assert (
            strip_temporal_phrases("what did we do last week") == "what did we do last week"
        )


# ---------------------------------------------------------------------------
# The accepted regression
# ---------------------------------------------------------------------------

def test_a_dangling_preposition_next_to_another_packs_removal_survives():
    """"work from ontem" keeps its English "from" — deliberate, not a bug.

    Cleanup runs only with the pack that owned the removal, and `ontem` is
    owned by pt-BR. Removing the English `from` here would mean running every
    enabled pack's cleanup at every span, which is exactly what turns
    "please say no last 3 days" into "please say". Whoever "fixes" this
    assertion breaks that one.
    """
    assert strip_temporal_phrases("work from ontem") == "work from"
    assert strip_temporal_phrases("work from yesterday") == "work"

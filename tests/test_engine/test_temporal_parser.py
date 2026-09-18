"""Tests for the language-agnostic TemporalParser.

Every parser here is constructed with an explicit set of packs — no test in
this module reads global configuration or the module-level classifier
functions. Assertions are only ever on the two externally observable outputs:
the window dict for a prompt, and the residue string for a prompt.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from ormah.engine.temporal import TemporalParser, en, pt_br

EN = en.LOCALE
PT = pt_br.LOCALE


def _parser(*locales):
    return TemporalParser(locales)


BOTH_ORDERS = pytest.mark.parametrize(
    "locales",
    [pytest.param((EN, PT), id="en,pt-BR"), pytest.param((PT, EN), id="pt-BR,en")],
)


def _days_ago(iso: str) -> float:
    return (datetime.now(timezone.utc) - datetime.fromisoformat(iso)).total_seconds() / 86400


class TestStripRemovesThePhrase:
    def test_english_phrase_is_removed(self):
        assert _parser(EN).strip_temporal_phrases("what did we do last week") == "what did we do"

    def test_portuguese_phrase_is_removed(self):
        assert _parser(PT).strip_temporal_phrases("o que fizemos ontem") == "o que fizemos"


class TestCleanupIsAnchoredToTheVacatedSpan:
    """The reported bug, its siblings, and the boundary condition that scopes it."""

    def test_the_reported_bug(self):
        assert (
            _parser(EN, PT).strip_temporal_phrases("o que fizemos na semana passada")
            == "o que fizemos"
        )

    @pytest.mark.parametrize(
        "prompt",
        [
            "o que fizemos no mês passado",
            "o que fizemos nos últimos 3 dias",
            "o que fizemos nas últimas 2 semanas",
            "o que fizemos em semana passada",
        ],
    )
    def test_every_sibling_contraction_goes_with_the_phrase(self, prompt):
        assert _parser(EN, PT).strip_temporal_phrases(prompt) == "o que fizemos"

    def test_an_interior_contraction_survives_while_the_dangling_one_goes(self):
        # The falsifier for cleaning up with a plain `search`: that finds the
        # interior "na " first and would return "mudanças API nos".
        assert (
            _parser(EN, PT).strip_temporal_phrases("mudanças na API nos últimos 3 dias")
            == "mudanças na API"
        )

    def test_english_cleanup_fires_at_an_end_positioned_span(self):
        assert (
            _parser(EN, PT).strip_temporal_phrases("changes to the API in the last 3 days")
            == "changes to the API"
        )

    def test_a_mid_clause_gap_keeps_its_preposition(self):
        assert (
            _parser(EN, PT).strip_temporal_phrases("what changed in the last 3 days in auth")
            == "what changed in the in auth"
        )

    def test_the_boundary_is_evaluated_in_the_final_residue(self):
        # "yesterday" is removed too, so the numeric gap ends up at the end of
        # the residue and cleanup fires there. Judging the boundary at the
        # original span position instead leaves "what changed in the".
        assert (
            _parser(EN, PT).strip_temporal_phrases("what changed in the last 3 days yesterday")
            == "what changed"
        )

    @BOTH_ORDERS
    def test_only_the_owning_packs_cleanup_runs_at_a_span(self, locales):
        # `no` is a PT-BR contraction but the removal is owned by `en`, so it
        # survives. A shared numeric pattern, or cleanup gated on "this pack
        # matched somewhere", returns "please say".
        assert (
            TemporalParser(locales).strip_temporal_phrases("please say no last 3 days")
            == "please say no"
        )

    @BOTH_ORDERS
    def test_nothing_removed_means_nothing_cleaned(self, locales):
        assert TemporalParser(locales).strip_temporal_phrases("say no") == "say no"

    def test_a_head_positioned_span_cleans_nowhere_near_the_trailing_no(self):
        assert (
            _parser(EN, PT).strip_temporal_phrases("ontem, explain why we should say no")
            == ", explain why we should say no"
        )


class TestRemovalsAreSequentialReSearches:
    def test_two_disjoint_phrases_are_both_removed(self):
        assert (
            _parser(EN, PT).strip_temporal_phrases("what did we do yesterday and last week")
            == "what did we do and"
        )

    def test_overlapping_phrases_leave_the_determiner(self):
        assert _parser(EN, PT).strip_temporal_phrases("esta semana passada") == "esta"

    def test_overlapping_phrases_do_not_eat_the_topic(self):
        # Collecting spans on the original prompt and deleting them afterwards
        # right-to-left yields "timos autenticação".
        assert (
            _parser(EN, PT).strip_temporal_phrases(
                "nesta semana passada discutimos autenticação"
            )
            == "nesta discutimos autenticação"
        )


class TestNumericBeatsStaticAndTheLeftmostNumericWins:
    @BOTH_ORDERS
    def test_the_leftmost_numeric_match_wins_across_packs(self, locales):
        params = TemporalParser(locales).extract_time_params(
            "compare últimas 2 semanas with last 3 days"
        )
        # The rolling two-week window: 28d ago -> 14d ago. "First pack that
        # matched" would hand this to `en` under the default order.
        assert 27.9 < _days_ago(params["created_after"]) < 28.1
        assert 13.9 < _days_ago(params["created_before"]) < 14.1

    @BOTH_ORDERS
    def test_the_reversed_prompt_selects_the_other_numeric(self, locales):
        params = TemporalParser(locales).extract_time_params(
            "compare last 3 days with últimas 2 semanas"
        )
        assert 2.9 < _days_ago(params["created_after"]) < 3.1
        assert _days_ago(params["created_before"]) < 0.1

    @BOTH_ORDERS
    def test_a_numeric_phrase_beats_another_packs_static_phrase(self, locales):
        params = TemporalParser(locales).extract_time_params(
            "today: resumo das últimas 2 semanas"
        )
        assert 27.9 < _days_ago(params["created_after"]) < 28.1


class TestStaticPriorityDecidesTheWindow:
    @BOTH_ORDERS
    def test_the_earlier_priority_wins_when_it_appears_first(self, locales):
        params = TemporalParser(locales).extract_time_params("compare ontem with last month")
        assert 1.9 < _days_ago(params["created_after"]) < 2.1
        assert 0.9 < _days_ago(params["created_before"]) < 1.1

    @BOTH_ORDERS
    def test_the_earlier_priority_wins_when_it_appears_last(self, locales):
        # Only the reversed prompt separates the priority rule from a
        # leftmost-in-the-prompt rule, which would select last month (60 -> 30).
        params = TemporalParser(locales).extract_time_params("compare last month with ontem")
        assert 1.9 < _days_ago(params["created_after"]) < 2.1
        assert 0.9 < _days_ago(params["created_before"]) < 1.1


class TestWindowArithmetic:
    @pytest.mark.parametrize(
        "prompt,after,before",
        [
            ("show me the past 2 weeks", 28, 14),
            ("mostra as últimas 2 semanas", 28, 14),
            ("last 3 months summary", 180, 90),
            ("resumo dos últimos 3 meses", 180, 90),
        ],
    )
    def test_rolling_previous_period_for_weeks_and_months_above_one(self, prompt, after, before):
        params = _parser(EN, PT).extract_time_params(prompt)
        assert after - 0.1 < _days_ago(params["created_after"]) < after + 0.1
        assert before - 0.1 < _days_ago(params["created_before"]) < before + 0.1

    @pytest.mark.parametrize(
        "prompt,days",
        [
            ("show me the past 1 week", 7),
            ("mostra a última 1 semana", 7),
            ("what did we do in the last 4 days", 4),
            ("o que fizemos nos últimos 4 dias", 4),
            ("what happened in the last 6 hours", 0.25),
            ("o que aconteceu nas últimas 6 horas", 0.25),
            ("last 1 month recap", 30),
            ("resumo do último 1 mês", 30),
        ],
    )
    def test_unit_to_days_and_the_open_ended_branch(self, prompt, days):
        params = _parser(EN, PT).extract_time_params(prompt)
        assert days - 0.1 < _days_ago(params["created_after"]) < days + 0.1
        assert _days_ago(params["created_before"]) < 0.1

    @pytest.mark.parametrize(
        "prompt,after,before",
        [
            ("what did we do today", 1, 0),
            ("o que fizemos hoje", 1, 0),
            ("what did we do yesterday", 2, 1),
            ("o que fizemos ontem", 2, 1),
            ("what happened last week", 14, 7),
            ("o que aconteceu na semana passada", 14, 7),
            ("what happened this week", 7, 0),
            ("o que fizemos nesta semana", 7, 0),
            ("what happened last month", 60, 30),
            ("o que fizemos no mês passado", 60, 30),
            ("what changed recently", 3, 0),
            ("o que mudou recentemente", 3, 0),
        ],
    )
    def test_static_windows(self, prompt, after, before):
        params = _parser(EN, PT).extract_time_params(prompt)
        assert after - 0.1 < _days_ago(params["created_after"]) < after + 0.1
        assert before - 0.1 < _days_ago(params["created_before"]) < before + 0.1

    @pytest.mark.parametrize(
        "prompt", ["what were we working on", "no que estávamos trabalhando", "any recent changes"]
    )
    def test_a_prompt_with_no_window_falls_back_to_three_days(self, prompt):
        params = _parser(EN, PT).extract_time_params(prompt)
        assert 2.9 < _days_ago(params["created_after"]) < 3.1
        assert _days_ago(params["created_before"]) < 0.1


class TestTemporalDetection:
    @pytest.mark.parametrize(
        "prompt",
        [
            "what did we do yesterday",
            "o que fizemos ontem",
            "what did we do in the last 4 days",
            "o que fizemos nos últimos 4 dias",
        ],
    )
    def test_windowed_and_numeric_phrases_are_temporal(self, prompt):
        assert _parser(EN, PT).has_temporal_phrases(prompt) is True

    def test_a_strip_only_phrase_is_not_temporal_but_is_still_stripped(self):
        parser = _parser(EN, PT)
        assert parser.has_temporal_phrases("show me recent changes") is False
        assert parser.strip_temporal_phrases("show me recent changes") == "show me changes"

    def test_a_prompt_with_no_phrase_is_not_temporal(self):
        assert _parser(EN, PT).has_temporal_phrases("what were we working on") is False


class TestThePackSetGatesBehaviour:
    def test_an_english_only_parser_does_not_see_portuguese(self):
        parser = _parser(EN)
        assert parser.has_temporal_phrases("o que fizemos na semana passada") is False
        assert (
            parser.strip_temporal_phrases("o que fizemos na semana passada")
            == "o que fizemos na semana passada"
        )
        params = parser.extract_time_params("o que fizemos ontem")
        assert 2.9 < _days_ago(params["created_after"]) < 3.1

    def test_a_portuguese_only_parser_does_not_see_english(self):
        parser = _parser(PT)
        assert parser.has_temporal_phrases("what did we do last week") is False
        assert (
            parser.strip_temporal_phrases("what did we do last week") == "what did we do last week"
        )

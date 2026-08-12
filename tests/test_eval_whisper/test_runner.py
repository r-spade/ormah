"""Tests for eval/whisper/runner.py."""
from __future__ import annotations
from unittest.mock import MagicMock, patch
import pytest
from eval.whisper.runner import run_whisper_eval, _aggregate


_CASES = [
    {
        "id": "w-fact-001",
        "space": "ormah",
        "memories": [
            {"node_id": "mem-001", "title": "Port fact", "content": "Runs on 8787.", "type": "fact", "tier": "working"},
            {"node_id": "mem-002", "title": "Distractor", "content": "Unrelated.", "type": "fact", "tier": "working"},
        ],
        "prompts": [
            {
                "text": "what port does ormah run on",
                "category": "factual",
                "expected": {
                    "should_inject": ["mem-001"],
                    "should_not_inject": ["mem-002"],
                    "should_suppress": False,
                },
            }
        ],
    },
    {
        "id": "w-noise-001",
        "space": "ormah",
        "memories": [
            {"node_id": "mem-003", "title": "Some fact", "content": "Content.", "type": "fact", "tier": "working"},
        ],
        "prompts": [
            {
                "text": "hello",
                "category": "noise",
                "expected": {"should_inject": [], "should_not_inject": [], "should_suppress": True},
            }
        ],
    },
]


class TestRunWhisperEval:
    def test_returns_result_per_prompt(self):
        mock_engine = MagicMock()
        mock_engine.get_whisper_context.side_effect = [
            ("whisper text", ["mem-001"]),  # factual case: hit
            ("", []),                        # noise case: suppressed
        ]
        with patch("eval.whisper.runner.seed_case"):
            result = run_whisper_eval(_CASES, mock_engine)
        assert len(result.prompt_results) == 2

    def test_factual_hit_metrics(self):
        mock_engine = MagicMock()
        mock_engine.get_whisper_context.side_effect = [
            ("whisper text", ["mem-001"]),
            ("", []),
        ]
        with patch("eval.whisper.runner.seed_case"):
            result = run_whisper_eval(_CASES, mock_engine)
        factual = next(r for r in result.prompt_results if r.category == "factual")
        assert factual.metrics["injection_recall"] == 1.0
        assert factual.metrics["false_positive_present"] is False

    def test_noise_suppression_metrics(self):
        mock_engine = MagicMock()
        mock_engine.get_whisper_context.side_effect = [
            ("whisper text", ["mem-001"]),
            ("", []),
        ]
        with patch("eval.whisper.runner.seed_case"):
            result = run_whisper_eval(_CASES, mock_engine)
        noise = next(r for r in result.prompt_results if r.category == "noise")
        assert noise.metrics["suppression_correct"] is True

    def test_engine_called_with_correct_args(self):
        mock_engine = MagicMock()
        mock_engine.get_whisper_context.return_value = ("", [])
        with patch("eval.whisper.runner.seed_case"):
            run_whisper_eval([_CASES[0]], mock_engine)
        call_kwargs = mock_engine.get_whisper_context.call_args
        assert call_kwargs.kwargs["recent_prompts"] == []
        assert call_kwargs.kwargs["session_id"] is None
        assert call_kwargs.kwargs["_return_debug"] is True
        assert call_kwargs.kwargs["space"] == "ormah"

    def test_seed_called_once_per_case(self):
        mock_engine = MagicMock()
        mock_engine.get_whisper_context.return_value = ("", [])
        with patch("eval.whisper.runner.seed_case") as mock_seed:
            run_whisper_eval(_CASES, mock_engine)
        assert mock_seed.call_count == 2

    def test_simulate_session_passes_prior_prompts_to_follow_up_turns(self):
        mock_engine = MagicMock()
        mock_engine.settings.whisper_context_buffer_size = 5
        mock_engine.get_whisper_context.side_effect = [
            ("first", ["mem-001"]),
            ("second", ["mem-002"]),
        ]
        cases = [{
            "id": "session-case",
            "space": "ormah",
            "simulate_session": True,
            "session_id": "sess-123",
            "memories": [],
            "prompts": [
                {
                    "text": "how does the pipeline work?",
                    "category": "technical",
                    "expected": {"should_inject": ["mem-001"], "should_suppress": False},
                },
                {
                    "text": "what about the report format?",
                    "category": "continuation",
                    "expected": {"should_inject": ["mem-002"], "should_suppress": False},
                },
            ],
        }]

        with patch("eval.whisper.runner.seed_case"):
            run_whisper_eval(cases, mock_engine)

        first_call = mock_engine.get_whisper_context.call_args_list[0]
        second_call = mock_engine.get_whisper_context.call_args_list[1]
        assert first_call.kwargs["recent_prompts"] is None
        assert first_call.kwargs["session_id"] == "sess-123"
        assert second_call.kwargs["recent_prompts"] == ["how does the pipeline work?"]
        assert second_call.kwargs["session_id"] == "sess-123"

    def test_category_aggregates_split_by_category(self):
        mock_engine = MagicMock()
        mock_engine.get_whisper_context.side_effect = [
            ("text", ["mem-001"]),
            ("", []),
        ]
        with patch("eval.whisper.runner.seed_case"):
            result = run_whisper_eval(_CASES, mock_engine)
        assert "factual" in result.category_aggregates
        assert "noise" in result.category_aggregates

    def test_runner_understands_new_expectation_fields(self):
        mock_engine = MagicMock()
        mock_engine.get_whisper_context.return_value = ("text", ["mem-001", "mem-002"])
        cases = [{
            "id": "w-new-fields",
            "space": "ormah",
            "memories": [],
            "prompts": [
                {
                    "text": "q",
                    "category": "technical",
                    "expected": {
                        "must_include": ["mem-001"],
                        "may_include": ["mem-002"],
                        "must_not_include": ["mem-003"],
                        "must_be_silent": False,
                    },
                }
            ],
        }]

        with patch("eval.whisper.runner.seed_case"):
            result = run_whisper_eval(cases, mock_engine)

        prompt_result = result.prompt_results[0]
        assert prompt_result.should_inject == ["mem-001"]
        assert prompt_result.may_include == ["mem-002"]
        assert prompt_result.should_not_inject == ["mem-003"]
        assert prompt_result.should_suppress is False
        assert prompt_result.metrics["injection_precision"] == 1.0

    def test_preserve_self_flows_to_seed_case(self):
        mock_engine = MagicMock()
        mock_engine.get_whisper_context.return_value = ("", [])
        cases = [{"id": "x", "space": "ormah", "memories": [], "prompts": []}]

        with patch("eval.whisper.runner.seed_case") as mock_seed:
            run_whisper_eval(cases, mock_engine, preserve_self=True)

        assert mock_seed.call_args.kwargs["preserve_self"] is True

    def test_case_level_preserve_self_used_when_no_override(self):
        mock_engine = MagicMock()
        mock_engine.get_whisper_context.return_value = ("", [])
        cases = [{"id": "x", "space": "ormah", "preserve_self": True, "memories": [], "prompts": []}]

        with patch("eval.whisper.runner.seed_case") as mock_seed:
            run_whisper_eval(cases, mock_engine)

        assert mock_seed.call_args.kwargs["preserve_self"] is True


class TestAggregate:
    def _make_result(
        self,
        category,
        *,
        injected=0,
        relevant=0,
        critical=0,
        found=0,
        forbidden=0,
        suppressed=0,
        target_decision="inject",
        false_positive_turn=False,
        turn_helpful=None,
        useful_turn=False,
        suppression_correct=None,
    ):
        from eval.whisper.runner import PromptResult
        metrics = {
            "injection_recall": found / critical if critical else None,
            "injection_precision": relevant / injected if injected else None,
            "f1": None,
            "top2_recall": found / critical if critical else None,
            "suppression_correct": suppression_correct,
            "false_positive_present": False,
            "false_positive_turn": false_positive_turn,
            "injection_fired": injected > 0,
            "injected_count": injected,
            "relevant_injected_count": relevant,
            "critical_count": critical,
            "critical_found_count": found,
            "forbidden_count": forbidden,
            "forbidden_suppressed_count": suppressed,
            "turn_helpful": turn_helpful,
            "useful_turn": useful_turn,
        }
        return PromptResult(
            case_id="x", prompt="q", category=category,
            should_inject=[], injected_ids=[], metrics=metrics,
            target_decision=target_decision,
            schema_version="2.0",
        )

    def test_injection_recall_is_micro_averaged(self):
        results = [
            self._make_result("factual", critical=1, found=1),
            self._make_result("factual", critical=3, found=1),
        ]
        agg = _aggregate(results)
        assert agg["injection_recall"] == pytest.approx(0.5)

    def test_useful_recall_counts_inject_turns_receiving_all_critical_memory(self):
        results = [
            self._make_result(
                "factual", critical=2, found=2, useful_turn=True
            ),
            self._make_result("factual", critical=1, found=0),
            self._make_result("noise", target_decision="abstain"),
        ]
        agg = _aggregate(results)
        assert agg["useful_recall"] == pytest.approx(0.5)
        assert agg["sample_counts"]["useful_recall"] == 2

    def test_injection_precision_counts_abstain_turn_nodes(self):
        results = [
            self._make_result("factual", injected=1, relevant=1, critical=1, found=1),
            self._make_result(
                "noise",
                injected=3,
                target_decision="abstain",
                false_positive_turn=True,
            ),
        ]
        assert _aggregate(results)["injection_precision"] == pytest.approx(0.25)

    def test_forbidden_node_suppression_accuracy(self):
        results = [
            self._make_result("factual", forbidden=3, suppressed=3),
            self._make_result("factual", forbidden=1, suppressed=0),
        ]
        agg = _aggregate(results)
        assert agg["suppression_accuracy"] == pytest.approx(0.75)

    def test_false_positive_turn_rate_uses_only_abstain_denominator(self):
        results = [
            self._make_result(
                "noise", target_decision="abstain", false_positive_turn=True
            ),
            self._make_result("noise", target_decision="abstain"),
            *[self._make_result("factual", critical=1, found=1) for _ in range(8)],
        ]
        agg = _aggregate(results)
        assert agg["false_positive_turn_rate"] == pytest.approx(0.5)
        assert agg["sample_counts"]["false_positive_turn_rate"] == 2

    def test_wilson_interval_and_denominator_are_reported(self):
        results = [
            self._make_result("factual", injected=1, relevant=1),
            self._make_result("factual", injected=1, relevant=0),
        ]
        agg = _aggregate(results)
        low, high = agg["descriptive_wilson_intervals"]["injection_precision"]
        assert low < 0.5 < high
        assert agg["sample_counts"]["injection_precision"] == 2

    def test_ask_qualify_injection_makes_full_turn_precision_unavailable(self):
        results = [
            self._make_result(
                "conflict",
                injected=1,
                relevant=1,
                target_decision="ask_qualify",
            )
        ]
        agg = _aggregate(results)
        assert agg["turn_precision"] is None
        assert agg["unscorable_injected_turn_count"] == 1
        assert agg["sample_counts"]["turn_precision"] == 0
        assert agg["strict_label_turn_precision"] is None

    def test_no_labeled_non_noise_returns_none_for_recall(self):
        results = [self._make_result("noise", suppression_correct=True)]
        agg = _aggregate(results)
        assert agg["injection_recall"] is None

    def test_legacy_aggregate_preserves_macro_average(self):
        results = [
            self._make_result("factual", critical=1, found=1),
            self._make_result("factual", critical=2, found=1),
        ]
        aggregate = _aggregate(results, schema_mode="legacy_compat")
        assert aggregate["injection_recall"] == pytest.approx(0.75)
        assert "useful_recall" not in aggregate

    def test_legacy_fp_does_not_count_unlisted_silent_turn_injection(self):
        result = self._make_result(
            "noise",
            injected=1,
            target_decision="abstain",
            false_positive_turn=True,
        )
        result.schema_version = None
        result.metrics["legacy_false_positive_present"] = False
        aggregate = _aggregate([result], schema_mode="legacy_compat")
        assert aggregate["false_positive_rate"] == 0.0

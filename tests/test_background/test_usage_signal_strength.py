"""The heuristic detector places matches on the #218 ordinal ladder."""

import pytest

from ormah import signal_strength as ss
from ormah.background.session_watcher import _node_usage_evidence


def _row(node_id="a1b2c3d4-dead-beef-0000-000000000000", title="", content="", prompt_text=""):
    """_node_usage_evidence reads its row purely by key, so a dict is a valid row."""
    return {"node_id": node_id, "title": title, "content": content, "prompt_text": prompt_text}


def test_node_id_match_takes_the_top_heuristic_rung():
    referenced, strength, evidence = _node_usage_evidence(
        _row(content="anything"), "As memory a1b2c3d4 records, we chose X."
    )
    assert referenced
    assert evidence["match"] == "node_id"
    assert strength == ss.VERBATIM_NODE_ID


def test_title_match_takes_the_title_rung():
    referenced, strength, evidence = _node_usage_evidence(
        _row(
            title="Transcript watcher mines feedback usage",
            content="Some unrelated body text goes here.",
        ),
        "The transcript watcher mines feedback usage, as noted.",
    )
    assert referenced
    assert evidence["match"] == "title"
    assert strength == ss.VERBATIM_TITLE


def test_sentence_match_takes_the_sentence_rung():
    referenced, strength, evidence = _node_usage_evidence(
        _row(title="T", content="The consolidator summarizes from full source content."),
        "Recall that the consolidator summarizes from full source content today.",
    )
    assert referenced
    assert evidence["match"] == "sentence"
    assert strength == ss.VERBATIM_SENTENCE


def test_token_overlap_varies_with_its_ratio():
    """The defect #218 names: every token_overlap match used to report exactly 0.85."""
    referenced, strength, evidence = _node_usage_evidence(
        _row(title="Q", content="quantum entanglement decoherence topology manifold"),
        "We should consider decoherence, then topology, then the manifold, "
        "and finally quantum entanglement in that order.",
    )
    assert referenced
    assert evidence["match"] == "token_overlap"
    assert evidence["overlap_ratio"] == pytest.approx(1.0)
    assert strength == pytest.approx(ss.token_overlap_strength(1.0))
    assert strength != 0.85


def test_no_match_carries_no_strength():
    referenced, strength, evidence = _node_usage_evidence(
        _row(title="Z", content="alpha beta gamma delta"), "Completely unrelated prose here."
    )
    assert not referenced
    assert evidence["match"] == "none"
    assert strength == 0.0


def test_every_heuristic_rung_sits_inside_its_band():
    """The verbatim rungs must stay above the judge band, the overlap rung below implicit."""
    assert ss.VERBATIM_SENTENCE > ss.JUDGE_HI
    assert ss.OVERLAP_FLOOR + ss.OVERLAP_SPAN < ss.IMPLICIT


import json  # noqa: E402
from unittest.mock import patch  # noqa: E402

from ormah.background.session_watcher import _record_whisper_usage_signals  # noqa: E402
from ormah.models.node import CreateNodeRequest  # noqa: E402
from ormah.transcript.parser import parse_transcript  # noqa: E402
from tests.test_background.test_session_watcher import (  # noqa: E402
    _LLM_PATCH,
    _insert_injected_whisper_log,
    _write_turn_jsonl,
)


def _judged(engine, tmp_path, *, verdict, confidence, slug):
    """Drive one whisper through the judge and return its stored signal row."""
    prompt = "How should we handle the blue deployment rollback?"
    response = "Nothing in particular comes to mind about that."
    transcript_path = tmp_path / f"{slug}.jsonl"
    _write_turn_jsonl(transcript_path, prompt, response)
    transcript = parse_transcript(transcript_path)

    node_id, _ = engine.remember(CreateNodeRequest(
        content="Roll back a blue deployment by repointing the load balancer first.",
        type="fact",
        title="Blue deployment rollback marker",
    ))
    whisper_log_id = _insert_injected_whisper_log(
        engine, node_id=node_id, session_id=slug, prompt=prompt
    )
    engine.settings.llm_provider = "ollama"
    engine.settings.feedback_llm_judge_enabled = True

    llm_response = json.dumps({"verdicts": [{
        "whisper_log_id": whisper_log_id,
        "verdict": verdict,
        "confidence": confidence,
        "reason": "fixture",
    }]})
    with patch(_LLM_PATCH, return_value=llm_response):
        _record_whisper_usage_signals(engine, transcript)

    return engine.db.conn.execute(
        "SELECT * FROM signals WHERE whisper_log_id = ? "
        "AND source = 'transcript_watcher_llm_judge'",
        (whisper_log_id,),
    ).fetchone()


def test_a_used_verdict_lands_inside_the_judge_band(engine, tmp_path):
    """It used to store the raw confidence, which collides with other channels."""
    signal = _judged(engine, tmp_path, verdict="used", confidence=0.88, slug="judge-band-used")
    assert signal["polarity"] == 1
    assert signal["strength"] != 0.88
    assert ss.JUDGE_LO <= signal["strength"] <= ss.JUDGE_HI
    assert signal["strength"] == pytest.approx(
        ss.judge_strength(0.88, engine.settings.feedback_llm_judge_min_confidence, 1)
    )


def test_an_uncertain_verdict_carries_no_strength(engine, tmp_path):
    """Below min_confidence the polarity is 0, so the row asserts nothing.

    Its confidence is not lost: it stays in signals.evidence.
    """
    signal = _judged(
        engine, tmp_path, verdict="used", confidence=0.35, slug="judge-band-uncertain"
    )
    assert signal["polarity"] == 0
    assert signal["strength"] == 0.0
    assert json.loads(signal["evidence"])["confidence"] == 0.35


def test_recorded_evidence_reproduces_the_stored_strength():
    """evidence must be a lossless record of what determined strength.

    The #218 backfill recomputes strength from evidence. Deriving the stored strength
    from a more precise ratio than the one recorded makes that recompute perturb a
    correct row instead of confirming it — and makes any future rescan of new rows a
    drift rather than a no-op.

    4 overlaps over 6 candidates is 0.6666..., recorded as 0.667. That gap used to be
    ~1e-04 of strength.
    """
    referenced, strength, evidence = _node_usage_evidence(
        _row(title="Q", content="quantum entanglement decoherence topology manifold curvature"),
        "Consider decoherence, topology, the manifold and curvature when planning.",
    )
    assert referenced
    assert evidence["match"] == "token_overlap"
    assert evidence["overlap_ratio"] == pytest.approx(0.667)
    assert strength == ss.strength_from_evidence(ss.HEURISTIC_SOURCE, 1, json.dumps(evidence))

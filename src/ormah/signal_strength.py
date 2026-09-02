"""The ordinal evidence-strength ladder behind ``signals.strength`` (issue #218).

``strength`` is the strength of the evidence backing a signal row's polarity, on a
single ordinal scale, comparable in RANK across channels. It is NOT a calibrated
probability: 0.86 from the LLM judge and 0.86 from anywhere else mean "the same
rung", not "the same likelihood".

The load-bearing assertion is that the CHANNEL DOMINATES within-channel confidence.
A verbatim match outranks any LLM judgment however confident; an LLM judgment
outranks any agent self-report. Bands are therefore disjoint per channel, and native
confidence modulates only WITHIN a band.

    1.00         explicit ......... the user was actually asked
    0.98         node_id .......... the short id was printed verbatim
    0.94         title ............ the title was printed verbatim
    0.92         sentence ......... a content sentence was printed verbatim
    0.82 - 0.90  auto_llm_judge ... affine over [min_confidence, 1.0]
    0.80         implicit ......... the agent's own self-assessment
    0.40 - 0.78  token_overlap .... asymptotic in overlap_ratio
    0.00         polarity == 0 .... a row that asserts nothing has no evidence

A polarity-zero row asserts nothing, so it carries no evidence strength. Its native
confidence still survives in ``signals.evidence``; nothing is lost.
"""

from __future__ import annotations

import json
import math

# Detector source labels. Owned here because this module maps them onto the ladder;
# session_watcher imports them rather than repeating the literals.
HEURISTIC_SOURCE = "transcript_watcher_heuristic"
LLM_JUDGE_SOURCE = "transcript_watcher_llm_judge"

EXPLICIT = 1.00
VERBATIM_NODE_ID = 0.98
VERBATIM_TITLE = 0.94
VERBATIM_SENTENCE = 0.92
JUDGE_LO = 0.82
JUDGE_HI = 0.90
IMPLICIT = 0.80

# token_overlap: floored at the detector's own entry gate, asymptotic to FLOOR + SPAN.
OVERLAP_GATE = 0.5
OVERLAP_FLOOR = 0.40
OVERLAP_SPAN = 0.38
OVERLAP_K = 1.0

# Bottom of the ladder. Unknown provenance is the weakest evidence there is.
UNKNOWN = OVERLAP_FLOOR

# (channel, band_low, band_high). Disjointness is the executable form of the
# "channel dominates confidence" assertion; test_signal_strength.py pins it.
BANDS = (
    ("explicit", EXPLICIT, EXPLICIT),
    ("node_id", VERBATIM_NODE_ID, VERBATIM_NODE_ID),
    ("title", VERBATIM_TITLE, VERBATIM_TITLE),
    ("sentence", VERBATIM_SENTENCE, VERBATIM_SENTENCE),
    ("auto_llm_judge", JUDGE_LO, JUDGE_HI),
    ("implicit", IMPLICIT, IMPLICIT),
    ("token_overlap", OVERLAP_FLOOR, OVERLAP_FLOOR + OVERLAP_SPAN),
)

_FEEDBACK_LADDER = {
    "explicit": EXPLICIT,
    "implicit": IMPLICIT,
    # submit_feedback carries no confidence, so a judge-sourced row arriving through
    # it lands on the floor of the judge band rather than anywhere inside it.
    "auto_llm_judge": JUDGE_LO,
    "auto_heuristic": UNKNOWN,
}


def _as_float(value: object, default: float = 0.0) -> float:
    """Coerce a stored evidence value to a FINITE float, else *default*.

    Both guards exist because the #218 migration runs on every startup and writes
    into ``signals.strength``, which is ``REAL NOT NULL``:

    - ``OverflowError`` is in the except tuple because ``json.loads`` returns an
      unbounded Python ``int`` for a large integer literal, and ``float()`` of one
      beyond the double range raises it -- not ``ValueError``.
    - Non-finite results are rejected because ``json.loads`` accepts bare ``NaN``
      and ``Infinity``, and SQLite stores a NaN REAL as NULL. Binding one would
      raise ``IntegrityError`` inside the migration, and since the migration runs
      at every boot, a single poisoned historical row would stop the server from
      ever starting again.

    Reported by the Codex peer, /council-pr of 2026-08-26.
    """
    try:
        result = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError, OverflowError):
        return default
    return result if math.isfinite(result) else default


def _finite(value: float) -> float:
    """Last guard before a strength reaches the NOT NULL column.

    ``_as_float`` already keeps non-finite values out of the inputs, so this cannot
    fire today. It guards the OUTPUT side instead: a future change to one of the
    band formulas could produce a non-finite result from finite inputs, and the
    cost of that reaching the every-boot migration is a server that never starts.
    """
    return value if math.isfinite(value) else UNKNOWN


def token_overlap_strength(ratio: float) -> float:
    """Map ``overlap_ratio`` onto the token_overlap band.

    Asymptotic rather than clamped, because ``overlap_ratio`` is unbounded above --
    its denominator is ``min(len(candidate_tokens), 12)`` while its numerator is not
    capped. Any clamp ties every row above it, recreating the exact saturation defect
    #218 reports; a clamp at 1.50 would tie 38% of the rows observed on a live store.

    The supremum FLOOR + SPAN = 0.78 is strict in exact arithmetic but not in float64,
    which reaches it around ratio 37 -- five times the 7.583 maximum observed.
    """
    return OVERLAP_FLOOR + OVERLAP_SPAN * (
        1.0 - math.exp(-OVERLAP_K * max(ratio - OVERLAP_GATE, 0.0))
    )


def judge_strength(confidence: float, min_confidence: float, polarity: int) -> float:
    """Map the judge's confidence onto its band, affine over [min_confidence, 1.0].

    The domain is anchored on the caller's ``min_confidence`` rather than the literal
    0.75, so the band re-anchors if the setting moves. That domain is sound because
    the judge assigns a non-zero polarity only when ``confidence >= min_confidence``,
    so no scoring row can sit below the band floor.
    """
    if polarity == 0:
        return 0.0
    if min_confidence >= 1.0:
        return JUDGE_HI
    span = (confidence - min_confidence) / (1.0 - min_confidence)
    return JUDGE_LO + (JUDGE_HI - JUDGE_LO) * max(0.0, min(1.0, span))


def feedback_strength(source: str, signal: int) -> float:
    """Map a ``submit_feedback`` row onto the ladder.

    Fail-closed: a source the ladder does not know lands on ``UNKNOWN``, the bottom
    rung. ``submit_feedback`` validates ``signal`` only at its HTTP boundary
    (``FeedbackRequest``), so a direct Python caller can still arrive here with 0.
    """
    if signal == 0:
        return 0.0
    return _FEEDBACK_LADDER.get(source, UNKNOWN)


def strength_from_evidence(source: str, polarity: int, evidence_json: str | None) -> float:
    """Recompute a stored row's strength from the ``evidence`` it already carries.

    Used by the #218 backfill, and exact rather than estimated: each channel records
    what its own mapping needs, and the judge stamps the ``min_confidence`` in force
    when the row was written -- so the recompute uses that value, not today's.
    """
    if polarity == 0:
        return 0.0
    try:
        evidence = json.loads(evidence_json) if evidence_json else {}
    except (TypeError, ValueError):
        evidence = {}
    if not isinstance(evidence, dict):
        evidence = {}

    if source == HEURISTIC_SOURCE:
        match = evidence.get("match")
        if match == "node_id":
            return VERBATIM_NODE_ID
        if match == "title":
            return VERBATIM_TITLE
        if match == "sentence":
            return VERBATIM_SENTENCE
        if match == "token_overlap":
            return _finite(token_overlap_strength(_as_float(evidence.get("overlap_ratio"))))
        return UNKNOWN
    if source == LLM_JUDGE_SOURCE:
        return _finite(
            judge_strength(
                _as_float(evidence.get("confidence")),
                _as_float(evidence.get("min_confidence"), default=0.75),
                polarity,
            )
        )
    return feedback_strength(source, polarity)

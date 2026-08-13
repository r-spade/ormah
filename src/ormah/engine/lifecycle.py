"""Versioned lifecycle policy math.

The stored ``stability`` value keeps Ormah's original exponential meaning: it
is the number of days at which retrievability reaches ``exp(-1)`` (about 37%).
This module is deliberately small so decay and confirmed-use reinforcement do
not grow separate interpretations of the policy.
"""

from __future__ import annotations

import math
from datetime import datetime, timedelta


LIFECYCLE_MODEL_VERSION = 1
LIFECYCLE_MODEL_META_KEY = "lifecycle_model_version"
FSRS_BOOTSTRAP_META_KEY = "fsrs_migrated"
DECAY_THRESHOLD = 0.3
DEFAULT_INITIAL_STABILITY = -7.0 / math.log(DECAY_THRESHOLD)


def safe_stability(value: float | int | None, fallback: float) -> float:
    """Return a finite positive stability value for legacy/corrupt rows."""
    try:
        candidate = float(value) if value is not None else fallback
    except (TypeError, ValueError):
        candidate = fallback
    if not math.isfinite(candidate) or candidate <= 0:
        return fallback
    return candidate


def nonnegative_age_days(age_days: float | int | None) -> float:
    """Normalize an elapsed-day value without allowing NaN into policy math."""
    try:
        age = float(age_days) if age_days is not None else 0.0
    except (TypeError, ValueError):
        return 0.0
    if math.isnan(age):
        return 0.0
    return max(age, 0.0)


def elapsed_days(anchor: datetime, now: datetime) -> float:
    """Return non-negative elapsed days between two timezone-aware datetimes."""
    return nonnegative_age_days((now - anchor).total_seconds() / 86400.0)


def retrievability(age_days: float, stability: float, *, fallback: float) -> float:
    """Compute the retained exponential retrievability curve ``exp(-t / S)``."""
    age = nonnegative_age_days(age_days)
    stable = safe_stability(stability, fallback)
    return math.exp(-age / stable)


def importance_recency(age_days: float, half_life_days: float) -> float:
    """Compute importance's independent half-life recency signal."""
    age = nonnegative_age_days(age_days)
    try:
        half_life = float(half_life_days)
    except (TypeError, ValueError):
        half_life = 14.0
    if not math.isfinite(half_life) or half_life <= 0:
        half_life = 14.0
    return math.exp(-math.log(2.0) * age / half_life)


def reinforcement_spacing(
    age_days: float,
    stability: float,
    *,
    spacing_cap: float,
    fallback: float,
) -> float:
    """Compute ``min(R**-0.2, cap)`` safely in log space.

    Evaluating ``R`` first would underflow for old memories and then produce an
    overflow when raising zero to a negative power. The equivalent logarithmic
    form is bounded before exponentiation.
    """
    stable = safe_stability(stability, fallback)
    age = nonnegative_age_days(age_days)
    try:
        cap = float(spacing_cap)
    except (TypeError, ValueError):
        cap = 2.0
    if not math.isfinite(cap) or cap <= 0:
        cap = 2.0
    log_spacing = min(0.2 * age / stable, math.log(cap))
    return math.exp(log_spacing)


def bounded_stability_update(
    stability: float,
    age_days: float,
    *,
    gain: float,
    saturation_exponent: float,
    spacing_cap: float,
    max_stability: float,
    fallback: float,
) -> float:
    """Apply one bounded confirmed-use stability update.

    ``S_new = min(S * (1 + g * S**-w * spacing), max_stability)``.
    """
    stable = safe_stability(stability, fallback)
    maximum = safe_stability(max_stability, fallback)
    if stable >= maximum:
        return maximum
    spacing = reinforcement_spacing(
        age_days,
        stable,
        spacing_cap=spacing_cap,
        fallback=fallback,
    )
    updated = stable * (
        1.0 + float(gain) * (stable ** -float(saturation_exponent)) * spacing
    )
    if not math.isfinite(updated):
        updated = maximum
    return min(updated, maximum)


def archival_deadline(
    anchor: datetime,
    stability: float,
    *,
    threshold: float,
    fallback: float,
) -> datetime:
    """Return the current-model deadline at which retrievability crosses threshold.

    Keeping this relation explicit gives a future model migration a stable
    deadline to preserve without rescaling existing node stability values.
    """
    stable = safe_stability(stability, fallback)
    safe_threshold = min(max(float(threshold), 1e-12), 1.0 - 1e-12)
    return anchor + timedelta(days=stable * -math.log(safe_threshold))

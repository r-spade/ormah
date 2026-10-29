"""Shared eligibility and wording for agent-backed maintenance."""

import logging
from datetime import datetime, timezone

logger = logging.getLogger(__name__)

MAINTENANCE_DUE_MARKER = "maintenance_due"
MAINTENANCE_DUE_SIGNAL = (
    "maintenance_due: run the ormah-maintenance agent in the background; "
    "continue the conversation without blocking the user."
)


def maintenance_due_signal(engine, conn) -> str:
    """Scheduled maintenance is due only when overdue and no reservation is active.

    Engine-only use has no reservation callback. The manager installs one when
    attached; checking it also lazily expires abandoned analysis.
    """
    settings = getattr(engine, "settings", None)
    if not settings or not getattr(settings, "claude_maintenance_enabled", False):
        return ""
    is_active = getattr(engine, "maintenance_is_active", None)
    if is_active is not None and is_active():
        return ""
    try:
        row = conn.execute("SELECT value FROM meta WHERE key = 'last_maintenance_run'").fetchone()
        last_run = row[0] if row else None
        if last_run:
            parsed = datetime.fromisoformat(last_run.replace("Z", "+00:00"))
            if parsed.tzinfo is None:
                parsed = parsed.replace(tzinfo=timezone.utc)
            elapsed = datetime.now(timezone.utc) - parsed.astimezone(timezone.utc)
            if elapsed.total_seconds() <= getattr(settings, "claude_maintenance_interval_hours", 24) * 3600:
                return ""
        return MAINTENANCE_DUE_SIGNAL
    except Exception as exc:
        logger.warning("Failed to compute maintenance_due: %s", exc)
        return ""


def is_maintenance_due_signal(line: str) -> bool:
    """Return true for the current signal line and the legacy bare marker."""
    stripped = line.strip()
    return stripped in {MAINTENANCE_DUE_MARKER, MAINTENANCE_DUE_SIGNAL}

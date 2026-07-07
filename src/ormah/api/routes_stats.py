"""Canonical stats API route."""

from __future__ import annotations

from fastapi import APIRouter, Query, Request

router = APIRouter(tags=["stats"])


@router.get("/stats")
def stats(
    request: Request,
    days: int | None = Query(
        None,
        ge=1,
        le=365,
        description=(
            "Rolling window in days for usage counters. Omit to use the fixed "
            "current calendar week (Mon-Sun UTC)."
        ),
    ),
):
    """Canonical stats payload for tray, CLI, UI, and diagnostics."""
    engine = request.app.state.engine
    return engine.stats(days=days)

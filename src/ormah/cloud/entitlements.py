"""Offline-tolerant advisory entitlement policy from E08 section 6.

=================  ================================  =========  =========
State              Condition                         Uploads    Downloads
=================  ================================  =========  =========
``active``         fresh <=24h and ``backup: true``  run        run
``grace``          stale <=7d, last backup true      run        run
``expired``        fresh false/malformed or >7d      pause      run
``none``           no account token or cache         n/a        n/a
=================  ================================  =========  =========

This policy is advisory UX. The service enforces uploads at presign/finalize;
downloads and local backups are never gated here.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from enum import Enum
from pathlib import Path
from typing import Any

from ormah.cloud.client import CloudClient, client_from_settings

logger = logging.getLogger(__name__)

ENTITLEMENT_CACHE_PATH = Path.home() / ".local" / "share" / "ormah" / "entitlements.json"
FRESH_FOR = timedelta(hours=24)
GRACE_FOR = timedelta(days=7)


class EntitlementStatus(str, Enum):
    ACTIVE = "active"
    GRACE = "grace"
    EXPIRED = "expired"
    NONE = "none"


@dataclass(frozen=True)
class EntitlementCache:
    entitlements: dict[str, Any]
    fetched_at: datetime

    def age(self, now: datetime | None = None) -> timedelta:
        now = _as_utc(now or datetime.now(timezone.utc))
        return max(now - self.fetched_at, timedelta(0))

    @property
    def plan_status(self) -> str | None:
        value = self.entitlements.get("plan_status")
        return value if isinstance(value, str) else None


def _as_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def load_entitlement_cache(path: Path | None = None) -> EntitlementCache | None:
    path = (ENTITLEMENT_CACHE_PATH if path is None else path).expanduser()
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            return None
        fetched_raw = payload.pop("fetched_at", None)
        if not isinstance(fetched_raw, str):
            return None
        fetched_at = _as_utc(datetime.fromisoformat(fetched_raw.replace("Z", "+00:00")))
    except (OSError, ValueError, TypeError):
        return None
    return EntitlementCache(entitlements=payload, fetched_at=fetched_at)


def cache_entitlements(
    entitlements: dict[str, Any],
    *,
    fetched_at: datetime | None = None,
    path: Path | None = None,
) -> EntitlementCache:
    """Atomically cache the raw server response plus its fetch timestamp."""
    if not isinstance(entitlements, dict):
        raise ValueError("Entitlements response must be a JSON object.")
    path = (ENTITLEMENT_CACHE_PATH if path is None else path).expanduser()
    fetched_at = _as_utc(fetched_at or datetime.now(timezone.utc))
    payload = {**entitlements, "fetched_at": fetched_at.isoformat()}

    from ormah.setup import _atomic_write

    path.parent.mkdir(parents=True, exist_ok=True)
    _atomic_write(str(path), json.dumps(payload, indent=2, sort_keys=True) + "\n", mode=0o600)
    return EntitlementCache(entitlements=dict(entitlements), fetched_at=fetched_at)


def clear_entitlement_cache(path: Path | None = None) -> None:
    path = (ENTITLEMENT_CACHE_PATH if path is None else path).expanduser()
    try:
        path.unlink()
    except FileNotFoundError:
        pass


def refresh_entitlements(
    settings,
    *,
    client: CloudClient | None = None,
    now: datetime | None = None,
) -> EntitlementCache:
    """Fetch and cache current entitlements, raising on refresh failure."""
    owned_client = client is None
    client = client or client_from_settings(settings)
    try:
        entitlements = client.get_entitlements()
        return cache_entitlements(entitlements, fetched_at=now)
    finally:
        if owned_client:
            client.close()


def status_from_cache(
    cache: EntitlementCache, *, now: datetime | None = None
) -> EntitlementStatus:
    """Classify one cached response using the canonical E08 policy."""
    now = _as_utc(now or datetime.now(timezone.utc))
    age = cache.age(now)
    backup_enabled = cache.entitlements.get("backup") is True
    if age <= FRESH_FOR:
        return EntitlementStatus.ACTIVE if backup_enabled else EntitlementStatus.EXPIRED
    if age <= GRACE_FOR and backup_enabled:
        return EntitlementStatus.GRACE
    return EntitlementStatus.EXPIRED


def check_entitlement(settings, *, now: datetime | None = None) -> EntitlementStatus:
    """Return ``active|grace|expired|none`` without propagating refresh failures."""
    now = _as_utc(now or datetime.now(timezone.utc))
    cache = load_entitlement_cache()
    if not settings.account_token:
        return EntitlementStatus.NONE

    if cache is not None and cache.age(now) <= FRESH_FOR:
        return status_from_cache(cache, now=now)

    try:
        cache = refresh_entitlements(settings, now=now)
    except Exception as exc:
        logger.debug("Could not refresh Ormah Cloud entitlements; using cache: %s", exc)

    if cache is None:
        return EntitlementStatus.NONE
    return status_from_cache(cache, now=now)

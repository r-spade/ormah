"""Per-store cloud backup state and health derivation."""

from __future__ import annotations

from dataclasses import dataclass, field, fields, replace
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
from typing import Any
import threading
import uuid


CLOUD_STATE_DIR = Path.home() / ".local" / "share" / "ormah" / "cloud"
_STATE_LOCK = threading.RLock()


def _as_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def _parse_time(value: Any) -> datetime | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError("cloud state timestamps must be ISO-8601 strings")
    return _as_utc(datetime.fromisoformat(value.replace("Z", "+00:00")))


def _validate_store_id(store_id: str) -> str:
    try:
        parsed = uuid.UUID(store_id)
    except (ValueError, AttributeError) as exc:
        raise ValueError("store_id must be an RFC 4122 UUIDv4") from exc
    if parsed.version != 4 or parsed.variant != uuid.RFC_4122:
        raise ValueError("store_id must be an RFC 4122 UUIDv4")
    return str(parsed)


@dataclass(frozen=True)
class CloudState:
    """Durable client state for one memory store."""

    last_upload_at: datetime | None = None
    last_upload_snapshot_id: str | None = None
    last_upload_error: str | None = None
    last_verify_at: datetime | None = None
    last_verify_ok: bool | None = None
    last_verify_snapshot_id: str | None = None
    last_verify_error: str | None = None
    extra: dict[str, Any] = field(default_factory=dict, repr=False)

    def to_dict(self) -> dict[str, Any]:
        payload = dict(self.extra)
        payload.update(
            {
                "last_upload_at": (
                    _as_utc(self.last_upload_at).isoformat()
                    if self.last_upload_at is not None
                    else None
                ),
                "last_upload_snapshot_id": self.last_upload_snapshot_id,
                "last_upload_error": self.last_upload_error,
                "last_verify_at": (
                    _as_utc(self.last_verify_at).isoformat()
                    if self.last_verify_at is not None
                    else None
                ),
                "last_verify_ok": self.last_verify_ok,
                "last_verify_snapshot_id": self.last_verify_snapshot_id,
                "last_verify_error": self.last_verify_error,
            }
        )
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> CloudState:
        if not isinstance(payload, dict):
            raise ValueError("cloud state must be a JSON object")
        allowed = {item.name for item in fields(cls)} - {"extra"}
        values = {key: value for key, value in payload.items() if key in allowed}
        values["last_upload_at"] = _parse_time(values.get("last_upload_at"))
        values["last_verify_at"] = _parse_time(values.get("last_verify_at"))
        for key in (
            "last_upload_snapshot_id",
            "last_upload_error",
            "last_verify_snapshot_id",
            "last_verify_error",
        ):
            value = values.get(key)
            if value is not None and not isinstance(value, str):
                raise ValueError(f"{key} must be a string or null")
        verify_ok = values.get("last_verify_ok")
        if verify_ok is not None and not isinstance(verify_ok, bool):
            raise ValueError("last_verify_ok must be a boolean or null")
        extra = {key: value for key, value in payload.items() if key not in allowed}
        return cls(**values, extra=extra)


def state_path(store_id: str, *, state_dir: Path | None = None) -> Path:
    store_id = _validate_store_id(store_id)
    return (state_dir or CLOUD_STATE_DIR).expanduser() / f"{store_id}.json"


def load_state(store_id: str, *, state_dir: Path | None = None) -> CloudState:
    """Load one store's state; missing or corrupt files start empty."""
    path = state_path(store_id, state_dir=state_dir)
    try:
        return CloudState.from_dict(json.loads(path.read_text(encoding="utf-8")))
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return CloudState()


def save_state(
    store_id: str,
    state: CloudState,
    *,
    state_dir: Path | None = None,
) -> CloudState:
    """Atomically persist one store's state with owner-only permissions."""
    path = state_path(store_id, state_dir=state_dir)

    from ormah.setup import _atomic_write

    with _STATE_LOCK:
        path.parent.mkdir(parents=True, exist_ok=True)
        _atomic_write(
            str(path),
            json.dumps(state.to_dict(), indent=2, sort_keys=True) + "\n",
            mode=0o600,
        )
    return state


def update_state(
    store_id: str,
    *,
    state_dir: Path | None = None,
    **changes: Any,
) -> CloudState:
    """Update selected fields while preserving the rest of one store's state."""
    with _STATE_LOCK:
        state = replace(load_state(store_id, state_dir=state_dir), **changes)
        return save_state(store_id, state, state_dir=state_dir)


def _existing_store_id(memory_dir: Path) -> str | None:
    from ormah.cloud.keys import STORE_ID_NAME, get_or_create_store_id

    if not (Path(memory_dir).expanduser() / STORE_ID_NAME).is_file():
        return None
    return get_or_create_store_id(Path(memory_dir))


def cloud_status_payload(
    settings,
    *,
    entitlement: str | None = None,
    now: datetime | None = None,
    state_dir: Path | None = None,
) -> dict[str, Any]:
    """Return state plus derived ages and warnings for CLI/API/UI consumers."""
    now = _as_utc(now or datetime.now(timezone.utc))
    store_error = None
    try:
        store_id = _existing_store_id(settings.memory_dir)
    except Exception as exc:
        store_id = None
        store_error = str(exc)
    state = load_state(store_id, state_dir=state_dir) if store_id else CloudState()

    if entitlement is None:
        from ormah.cloud.entitlements import check_entitlement

        entitlement = check_entitlement(settings).value

    upload_age = None
    if state.last_upload_at is not None:
        upload_age = max(now - state.last_upload_at, timedelta(0))

    warnings: list[str] = []
    if store_error is not None:
        warnings.append(f"Cloud store identity is invalid: {store_error}")
    if settings.cloud_backup_enabled:
        if store_id is None and store_error is None:
            warnings.append("Cloud backup is enabled but this memory store is not initialized.")
        elif upload_age is not None and upload_age > timedelta(
            hours=settings.cloud_backup_interval_hours * 2
        ):
            warnings.append(
                "Cloud backup is stale: the last successful upload is older than "
                f"{settings.cloud_backup_interval_hours * 2} hours."
            )
    if state.last_verify_ok is False:
        detail = f": {state.last_verify_error}" if state.last_verify_error else "."
        warnings.append(f"Cloud restore verification failed{detail}")

    return {
        "enabled": settings.cloud_backup_enabled,
        "store_id": store_id,
        "interval_hours": settings.cloud_backup_interval_hours,
        "entitlement": entitlement,
        "last_upload_at": (
            state.last_upload_at.isoformat() if state.last_upload_at is not None else None
        ),
        "last_upload_snapshot_id": state.last_upload_snapshot_id,
        "last_upload_error": state.last_upload_error,
        "last_upload_age_seconds": (
            int(upload_age.total_seconds()) if upload_age is not None else None
        ),
        "last_verify_at": (
            state.last_verify_at.isoformat() if state.last_verify_at is not None else None
        ),
        "last_verify_ok": state.last_verify_ok,
        "last_verify_snapshot_id": state.last_verify_snapshot_id,
        "last_verify_error": state.last_verify_error,
        "warnings": warnings,
    }

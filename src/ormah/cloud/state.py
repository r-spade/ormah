"""Per-store cloud backup state and health derivation."""

from __future__ import annotations

from dataclasses import dataclass, field, fields, replace
from datetime import datetime, timedelta, timezone
from enum import StrEnum
import json
from pathlib import Path
from typing import Any, Callable
import threading
import uuid


CLOUD_STATE_DIR = Path.home() / ".local" / "share" / "ormah" / "cloud"
CURRENT_CLOUD_STATE_SCHEMA_VERSION = 3
_STATE_LOCK = threading.RLock()


class CloudStateError(RuntimeError):
    """Base error for durable cloud-state failures."""


class CloudStateLoadError(CloudStateError):
    """Raised when an existing cloud-state file cannot be read or parsed safely."""


class CloudStateVersionError(CloudStateError):
    """Raised when an older client would overwrite state from a newer schema."""


class ProtectionState(StrEnum):
    """User-facing states for the paid backup protection journey."""

    LOCAL_ONLY = "local_only"
    SIGN_IN_REQUIRED = "sign_in_required"
    SUBSCRIPTION_REQUIRED = "subscription_required"
    INITIALIZING = "initializing"
    UPLOADING_FIRST_BACKUP = "uploading_first_backup"
    VERIFYING_FIRST_BACKUP = "verifying_first_backup"
    PROTECTED = "protected"
    CHANGES_PENDING = "changes_pending"
    VERIFICATION_PENDING = "verification_pending"
    OFFLINE = "offline"
    PAUSED = "paused"
    STOPPED = "stopped"
    ATTENTION_REQUIRED = "attention_required"


class ProtectionIntentStatus(StrEnum):
    """Durable phases for an explicit Protect action."""

    PENDING = "pending"
    ACCOUNT_BOUND = "account_bound"
    CHECKOUT_PENDING = "checkout_pending"
    READY = "ready"
    RUNNING = "running"
    COMPLETED = "completed"
    CANCELED = "canceled"
    EXPIRED = "expired"


TERMINAL_PROTECTION_INTENT_STATUSES = frozenset(
    {
        ProtectionIntentStatus.COMPLETED,
        ProtectionIntentStatus.CANCELED,
        ProtectionIntentStatus.EXPIRED,
    }
)


class ProtectionOperationKind(StrEnum):
    ENABLE = "enable"
    DISABLE = "disable"
    BACKUP = "backup"
    VERIFY = "verify"
    RESTORE = "restore"


class ProtectionOperationPhase(StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    UPLOADING = "uploading"
    FINALIZING = "finalizing"
    VERIFYING = "verifying"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELED = "canceled"


class UploadJournalPhase(StrEnum):
    """Durable client phases around the only ambiguous upload boundary."""

    RESERVED = "reserved"
    FINALIZING = "finalizing"


class ProtectionReasonCode(StrEnum):
    """Stable machine-readable causes shared by CLI, REST, and desktop clients."""

    NOT_ENABLED = "not_enabled"
    SIGN_IN_REQUIRED = "sign_in_required"
    SUBSCRIPTION_REQUIRED = "subscription_required"
    OFFLINE = "offline"
    ENTITLEMENT_EXPIRED = "entitlement_expired"
    QUOTA_EXCEEDED = "quota_exceeded"
    KEY_MISSING = "key_missing"
    KEY_INVALID = "key_invalid"
    UPLOAD_FAILED = "upload_failed"
    VERIFICATION_FAILED = "verification_failed"
    DOWNLOAD_FAILED = "download_failed"
    CIPHERTEXT_HASH_UNAVAILABLE = "ciphertext_hash_unavailable"
    CIPHERTEXT_HASH_MISMATCH = "ciphertext_hash_mismatch"
    DECRYPT_FAILED = "decrypt_failed"
    MANIFEST_VERIFICATION_FAILED = "manifest_verification_failed"
    NODE_PARSE_FAILED = "node_parse_failed"
    BUNDLE_CORRUPT = "bundle_corrupt"
    INDEX_ENVIRONMENT_UNAVAILABLE = "index_environment_unavailable"
    INDEX_REBUILD_FAILED = "index_rebuild_failed"
    SEARCH_PROBE_FAILED = "search_probe_failed"
    SELF_POINTER_INVALID = "self_pointer_invalid"
    PROCESSING_LIMIT_EXCEEDED = "processing_limit_exceeded"
    DISK_SPACE_INSUFFICIENT = "disk_space_insufficient"
    OPERATION_INTERRUPTED = "operation_interrupted"
    STORE_BUSY = "store_busy"
    PROTECTION_STOPPED = "protection_stopped"
    CLIENT_UPDATE_REQUIRED = "client_update_required"
    SERVICE_UPDATE_REQUIRED = "service_update_required"
    NO_BACKUPABLE_MEMORY = "no_backupable_memory"
    NOT_DUE = "not_due"
    INTENT_EXPIRED = "intent_expired"
    INTENT_CANCELED = "intent_canceled"
    ACCOUNT_CHANGED = "account_changed"
    UPLOAD_STATUS_UNKNOWN = "upload_status_unknown"


@dataclass(frozen=True)
class ProtectionOperation:
    """Token-free operation result returned by shared protection adapters."""

    operation_id: str
    kind: ProtectionOperationKind
    phase: ProtectionOperationPhase
    state: ProtectionState
    reason_code: ProtectionReasonCode | None = None
    message: str | None = None
    snapshot_id: str | None = None
    protection_intent_id: str | None = None


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


def _serialize_time(value: datetime | None) -> str | None:
    return _as_utc(value).isoformat() if value is not None else None


def _parse_enum(value: Any, enum_type: type[StrEnum], field_name: str) -> StrEnum | str | None:
    if value is None:
        return None
    if isinstance(value, enum_type):
        return value
    if not isinstance(value, str):
        raise ValueError(f"{field_name} must be a string or null")
    try:
        return enum_type(value)
    except ValueError:
        # A newer client may add a state before an older reader is upgraded. Preserve it exactly.
        return value


def _serialize_enum(value: StrEnum | str | None) -> str | None:
    return value.value if isinstance(value, StrEnum) else value


def _legacy_protection_state(values: dict[str, Any]) -> ProtectionState:
    uploaded = values.get("last_upload_snapshot_id")
    verified = values.get("last_verify_snapshot_id")
    if uploaded is None:
        return ProtectionState.LOCAL_ONLY
    if values.get("last_verify_ok") is True and uploaded == verified:
        return ProtectionState.PROTECTED
    if values.get("last_verify_ok") is True:
        return ProtectionState.CHANGES_PENDING
    return ProtectionState.ATTENTION_REQUIRED


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

    schema_version: int = CURRENT_CLOUD_STATE_SCHEMA_VERSION
    last_upload_at: datetime | None = None
    last_upload_snapshot_id: str | None = None
    last_upload_error: str | None = None
    last_verify_at: datetime | None = None
    last_verify_ok: bool | None = None
    last_verify_snapshot_id: str | None = None
    last_verify_error: str | None = None
    protection_enabled_at: datetime | None = None
    protection_state: ProtectionState | str = ProtectionState.LOCAL_ONLY
    pending_protection_intent_id: str | None = None
    pending_protection_account_id: str | None = None
    pending_protection_store_id: str | None = None
    pending_protection_created_at: datetime | None = None
    pending_protection_expires_at: datetime | None = None
    pending_protection_status: ProtectionIntentStatus | str | None = None
    pending_protection_snapshot_id: str | None = None
    pending_protection_created_store: bool = False
    pending_protection_origin_state: ProtectionState | str | None = None
    protection_disabled_at: datetime | None = None
    last_operation_id: str | None = None
    last_operation_kind: ProtectionOperationKind | str | None = None
    last_operation_phase: ProtectionOperationPhase | str | None = None
    last_successful_upload_at: datetime | None = None
    last_successful_backup_snapshot_id: str | None = None
    last_successful_verify_at: datetime | None = None
    last_verified_snapshot_id: str | None = None
    last_error_code: ProtectionReasonCode | str | None = None
    last_error_message: str | None = None
    last_error_at: datetime | None = None
    local_graph_fingerprint_at_upload: str | None = None
    recovery_kit_verified_at: datetime | None = None
    next_scheduled_upload_at: datetime | None = None
    next_scheduled_verify_at: datetime | None = None
    pending_upload_id: str | None = None
    pending_upload_snapshot_id: str | None = None
    pending_upload_operation_id: str | None = None
    pending_upload_protection_intent_id: str | None = None
    pending_upload_phase: UploadJournalPhase | str | None = None
    pending_upload_expires_at: datetime | None = None
    extra: dict[str, Any] = field(default_factory=dict, repr=False)

    def to_dict(self) -> dict[str, Any]:
        payload = dict(self.extra)
        payload.update(
            {
                "schema_version": self.schema_version,
                "last_upload_at": _serialize_time(self.last_upload_at),
                "last_upload_snapshot_id": self.last_upload_snapshot_id,
                "last_upload_error": self.last_upload_error,
                "last_verify_at": _serialize_time(self.last_verify_at),
                "last_verify_ok": self.last_verify_ok,
                "last_verify_snapshot_id": self.last_verify_snapshot_id,
                "last_verify_error": self.last_verify_error,
                "protection_enabled_at": _serialize_time(self.protection_enabled_at),
                "protection_state": _serialize_enum(self.protection_state),
                "pending_protection_intent_id": self.pending_protection_intent_id,
                "pending_protection_account_id": self.pending_protection_account_id,
                "pending_protection_store_id": self.pending_protection_store_id,
                "pending_protection_created_at": _serialize_time(
                    self.pending_protection_created_at
                ),
                "pending_protection_expires_at": _serialize_time(
                    self.pending_protection_expires_at
                ),
                "pending_protection_status": _serialize_enum(
                    self.pending_protection_status
                ),
                "pending_protection_snapshot_id": self.pending_protection_snapshot_id,
                "pending_protection_created_store": self.pending_protection_created_store,
                "pending_protection_origin_state": _serialize_enum(
                    self.pending_protection_origin_state
                ),
                "protection_disabled_at": _serialize_time(self.protection_disabled_at),
                "last_operation_id": self.last_operation_id,
                "last_operation_kind": _serialize_enum(self.last_operation_kind),
                "last_operation_phase": _serialize_enum(self.last_operation_phase),
                "last_successful_upload_at": _serialize_time(self.last_successful_upload_at),
                "last_successful_backup_snapshot_id": (
                    self.last_successful_backup_snapshot_id
                ),
                "last_successful_verify_at": _serialize_time(self.last_successful_verify_at),
                "last_verified_snapshot_id": self.last_verified_snapshot_id,
                "last_error_code": _serialize_enum(self.last_error_code),
                "last_error_message": self.last_error_message,
                "last_error_at": _serialize_time(self.last_error_at),
                "local_graph_fingerprint_at_upload": self.local_graph_fingerprint_at_upload,
                "recovery_kit_verified_at": _serialize_time(self.recovery_kit_verified_at),
                "next_scheduled_upload_at": _serialize_time(self.next_scheduled_upload_at),
                "next_scheduled_verify_at": _serialize_time(self.next_scheduled_verify_at),
                "pending_upload_id": self.pending_upload_id,
                "pending_upload_snapshot_id": self.pending_upload_snapshot_id,
                "pending_upload_operation_id": self.pending_upload_operation_id,
                "pending_upload_protection_intent_id": (
                    self.pending_upload_protection_intent_id
                ),
                "pending_upload_phase": _serialize_enum(self.pending_upload_phase),
                "pending_upload_expires_at": _serialize_time(
                    self.pending_upload_expires_at
                ),
            }
        )
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> CloudState:
        if not isinstance(payload, dict):
            raise ValueError("cloud state must be a JSON object")
        allowed = {item.name for item in fields(cls)} - {"extra"}
        values = {key: value for key, value in payload.items() if key in allowed}

        schema_version = values.get("schema_version", 1)
        if (
            not isinstance(schema_version, int)
            or isinstance(schema_version, bool)
            or schema_version < 1
        ):
            raise ValueError("schema_version must be a positive integer")
        values["schema_version"] = max(
            schema_version,
            CURRENT_CLOUD_STATE_SCHEMA_VERSION,
        )

        time_fields = (
            "last_upload_at",
            "last_verify_at",
            "protection_enabled_at",
            "pending_protection_created_at",
            "pending_protection_expires_at",
            "protection_disabled_at",
            "last_successful_upload_at",
            "last_successful_verify_at",
            "last_error_at",
            "recovery_kit_verified_at",
            "next_scheduled_upload_at",
            "next_scheduled_verify_at",
            "pending_upload_expires_at",
        )
        for key in time_fields:
            values[key] = _parse_time(values.get(key))

        string_fields = (
            "last_upload_snapshot_id",
            "last_upload_error",
            "last_verify_snapshot_id",
            "last_verify_error",
            "pending_protection_intent_id",
            "pending_protection_account_id",
            "pending_protection_store_id",
            "pending_protection_snapshot_id",
            "last_operation_id",
            "last_successful_backup_snapshot_id",
            "last_verified_snapshot_id",
            "last_error_message",
            "local_graph_fingerprint_at_upload",
            "pending_upload_id",
            "pending_upload_snapshot_id",
            "pending_upload_operation_id",
            "pending_upload_protection_intent_id",
        )
        for key in string_fields:
            value = values.get(key)
            if value is not None and not isinstance(value, str):
                raise ValueError(f"{key} must be a string or null")

        verify_ok = values.get("last_verify_ok")
        if verify_ok is not None and not isinstance(verify_ok, bool):
            raise ValueError("last_verify_ok must be a boolean or null")

        created_store = values.get("pending_protection_created_store", False)
        if not isinstance(created_store, bool):
            raise ValueError("pending_protection_created_store must be a boolean")
        values["pending_protection_created_store"] = created_store

        enum_fields = {
            "protection_state": ProtectionState,
            "pending_protection_status": ProtectionIntentStatus,
            "pending_protection_origin_state": ProtectionState,
            "last_operation_kind": ProtectionOperationKind,
            "last_operation_phase": ProtectionOperationPhase,
            "last_error_code": ProtectionReasonCode,
            "pending_upload_phase": UploadJournalPhase,
        }
        for key, enum_type in enum_fields.items():
            if key in values:
                values[key] = _parse_enum(values[key], enum_type, key)

        if "last_successful_upload_at" not in payload:
            values["last_successful_upload_at"] = values["last_upload_at"]
        if "last_successful_backup_snapshot_id" not in payload:
            values["last_successful_backup_snapshot_id"] = values.get(
                "last_upload_snapshot_id"
            )
        if "last_successful_verify_at" not in payload and verify_ok is True:
            values["last_successful_verify_at"] = values["last_verify_at"]
        if "last_verified_snapshot_id" not in payload and verify_ok is True:
            values["last_verified_snapshot_id"] = values.get("last_verify_snapshot_id")
        if "protection_state" not in payload:
            values["protection_state"] = _legacy_protection_state(values)

        extra = {key: value for key, value in payload.items() if key not in allowed}
        return cls(**values, extra=extra)


def state_path(store_id: str, *, state_dir: Path | None = None) -> Path:
    store_id = _validate_store_id(store_id)
    return (state_dir or CLOUD_STATE_DIR).expanduser() / f"{store_id}.json"


def load_state(store_id: str, *, state_dir: Path | None = None) -> CloudState:
    """Load one store's state, distinguishing absence from unsafe existing data."""
    path = state_path(store_id, state_dir=state_dir)
    try:
        raw = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return CloudState()
    except OSError as exc:
        raise CloudStateLoadError(f"Could not read cloud state {path}: {exc}") from exc
    try:
        return CloudState.from_dict(json.loads(raw))
    except (ValueError, TypeError, json.JSONDecodeError) as exc:
        raise CloudStateLoadError(f"Cloud state {path} is invalid: {exc}") from exc


def save_state(
    store_id: str,
    state: CloudState,
    *,
    memory_dir: Path,
    state_dir: Path | None = None,
    lock_timeout: float = 30.0,
) -> CloudState:
    """Atomically persist one store's state with owner-only permissions."""
    path = state_path(store_id, state_dir=state_dir)

    from ormah.cloud.store_lock import StoreLock

    with StoreLock(memory_dir, timeout=lock_timeout):
        if path.exists():
            existing = load_state(store_id, state_dir=state_dir)
            _ensure_writable_schema(existing)
        return _write_state(path, state)


def _ensure_writable_schema(state: CloudState) -> None:
    if state.schema_version > CURRENT_CLOUD_STATE_SCHEMA_VERSION:
        raise CloudStateVersionError(
            f"Cloud state schema {state.schema_version} is newer than this client's "
            f"schema {CURRENT_CLOUD_STATE_SCHEMA_VERSION}; update Ormah before writing it."
        )


def _write_state(path: Path, state: CloudState) -> CloudState:
    """Write state after the caller has acquired any required store lock."""

    _ensure_writable_schema(state)

    from ormah.setup import _atomic_write

    with _STATE_LOCK:
        path.parent.mkdir(parents=True, exist_ok=True)
        _atomic_write(
            str(path),
            json.dumps(state.to_dict(), indent=2, sort_keys=True) + "\n",
            mode=0o600,
        )
    return state


def mutate_state(
    store_id: str,
    transform: Callable[[CloudState], CloudState],
    *,
    memory_dir: Path,
    state_dir: Path | None = None,
    lock_timeout: float = 30.0,
) -> CloudState:
    """Apply one read-modify-write transformation without losing concurrent updates."""
    path = state_path(store_id, state_dir=state_dir)

    from ormah.cloud.store_lock import StoreLock

    with StoreLock(memory_dir, timeout=lock_timeout), _STATE_LOCK:
        current = load_state(store_id, state_dir=state_dir)
        updated = transform(current)
        if not isinstance(updated, CloudState):
            raise TypeError("cloud state transform must return CloudState")
        return _write_state(path, updated)


def update_state(
    store_id: str,
    *,
    memory_dir: Path,
    state_dir: Path | None = None,
    lock_timeout: float = 30.0,
    **changes: Any,
) -> CloudState:
    """Update selected fields while preserving the rest of one store's state."""
    return mutate_state(
        store_id,
        lambda state: replace(state, **changes),
        state_dir=state_dir,
        memory_dir=memory_dir,
        lock_timeout=lock_timeout,
    )


def _existing_store_id(memory_dir: Path) -> str | None:
    from ormah.cloud.keys import STORE_ID_NAME, get_or_create_store_id

    if not (Path(memory_dir).expanduser() / STORE_ID_NAME).is_file():
        return None
    return get_or_create_store_id(Path(memory_dir))


def is_protected_and_verified(state: CloudState, *, enabled: bool) -> bool:
    """Return whether durable metadata satisfies the user-facing protection claim."""

    intent_complete = (
        state.pending_protection_intent_id is None
        or state.pending_protection_status in TERMINAL_PROTECTION_INTENT_STATUSES
    )
    snapshot_id = state.last_successful_backup_snapshot_id
    return (
        enabled
        and intent_complete
        and state.pending_upload_phase is not UploadJournalPhase.FINALIZING
        and snapshot_id is not None
        and snapshot_id == state.last_verified_snapshot_id
        and state.last_verify_ok is True
    )


def is_device_loss_recovery_ready(state: CloudState, *, enabled: bool) -> bool:
    """Require both a saved-kit proof and the current verified snapshot invariant."""

    return state.recovery_kit_verified_at is not None and is_protected_and_verified(
        state,
        enabled=enabled,
    )


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
    state_error = None
    try:
        state = load_state(store_id, state_dir=state_dir) if store_id else CloudState()
    except CloudStateError as exc:
        state = CloudState()
        state_error = str(exc)

    if entitlement is None:
        from ormah.cloud.entitlements import check_entitlement

        entitlement = check_entitlement(settings).value

    upload_age = None
    if state.last_upload_at is not None:
        upload_age = max(now - state.last_upload_at, timedelta(0))

    warnings: list[str] = []
    if store_error is not None:
        warnings.append(f"Cloud store identity is invalid: {store_error}")
    if state_error is not None:
        warnings.append(
            "Cloud protection state could not be read and was not overwritten: "
            f"{state_error}"
        )
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

    raw_protection_state = (
        ProtectionState.ATTENTION_REQUIRED
        if state_error is not None or store_error is not None
        else state.protection_state
    )
    protection_state = raw_protection_state
    if not isinstance(protection_state, ProtectionState):
        protection_state = ProtectionState.ATTENTION_REQUIRED
        warnings.append(
            "Cloud protection state requires a newer Ormah version."
        )
    protected_invariant_failed = (
        protection_state is ProtectionState.PROTECTED
        and not is_protected_and_verified(
            state,
            enabled=settings.cloud_backup_enabled,
        )
    )
    if protected_invariant_failed:
        protection_state = ProtectionState.ATTENTION_REQUIRED
        warnings.append(
            "Cloud protection metadata is incomplete; verification must finish before "
            "this memory can be reported as protected."
        )
    if protection_state in {
        ProtectionState.PROTECTED,
        ProtectionState.CHANGES_PENDING,
        ProtectionState.VERIFICATION_PENDING,
    }:
        if entitlement == "expired":
            protection_state = ProtectionState.PAUSED
        elif entitlement == "none":
            protection_state = (
                ProtectionState.OFFLINE
                if getattr(settings, "account_token", None)
                else ProtectionState.SIGN_IN_REQUIRED
            )

    return {
        "schema_version": state.schema_version,
        "enabled": settings.cloud_backup_enabled,
        "store_id": store_id,
        "state_error": state_error,
        "interval_hours": settings.cloud_backup_interval_hours,
        "entitlement": entitlement,
        "protection_state": _serialize_enum(protection_state),
        "protection_enabled_at": _serialize_time(state.protection_enabled_at),
        "protection_disabled_at": _serialize_time(state.protection_disabled_at),
        "protection_intent_id": state.pending_protection_intent_id,
        "protection_intent_status": _serialize_enum(state.pending_protection_status),
        "protection_intent_created_at": _serialize_time(
            state.pending_protection_created_at
        ),
        "protection_intent_expires_at": _serialize_time(
            state.pending_protection_expires_at
        ),
        "last_operation_id": state.last_operation_id,
        "last_operation_kind": _serialize_enum(state.last_operation_kind),
        "last_operation_phase": _serialize_enum(state.last_operation_phase),
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
        "last_successful_upload_at": _serialize_time(state.last_successful_upload_at),
        "last_successful_backup_snapshot_id": state.last_successful_backup_snapshot_id,
        "last_successful_verify_at": _serialize_time(state.last_successful_verify_at),
        "last_verified_snapshot_id": state.last_verified_snapshot_id,
        "recovery_kit_verified_at": _serialize_time(state.recovery_kit_verified_at),
        "device_loss_recovery_ready": is_device_loss_recovery_ready(
            state,
            enabled=settings.cloud_backup_enabled,
        ),
        "last_error_code": _serialize_enum(state.last_error_code),
        "last_error_message": state.last_error_message,
        "warnings": warnings,
    }

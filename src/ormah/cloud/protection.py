"""Shared encrypted cloud-backup and restore-verification orchestration."""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timedelta, timezone
import errno
import logging
from pathlib import Path
import re
import shutil
import tempfile
import uuid

import httpx

from ormah.backup import (
    BackupError,
    BackupInfo,
    resolve_backup_user_node_id,
    resolve_current_user_node_id,
    service_from_settings,
)
from ormah.cloud.bundle import BundleError, build_bundle, open_bundle
from ormah.cloud.billing import validate_protection_intent_id
from ormah.cloud.client import CloudError, client_from_settings
from ormah.cloud.crypto import CloudCryptoError
from ormah.cloud.entitlements import (
    EntitlementStatus,
    cache_entitlements,
    check_entitlement,
)
from ormah.cloud.keys import (
    CloudKeyError,
    RECOVERY_KIT_PATH,
    STORE_ID_NAME,
    current_recipient,
    extract_store_id,
    get_or_create_store_id,
    init_key,
    key_file_exists,
    load_identities,
    write_recovery_kit,
)
from ormah.cloud.recovery import RecoveryKitService
from ormah.cloud.restore import (
    CloudRestoreError,
    CloudRestoreValidationError,
    PreparedCloudRestore,
    prepare_cloud_restore,
    restore_prepared_cloud_snapshot,
    verify_extracted_bundle,
)
from ormah.cloud.settings import set_cloud_backup_enabled
from ormah.cloud.state import (
    CURRENT_CLOUD_STATE_SCHEMA_VERSION,
    CloudState,
    CloudStateVersionError,
    ProtectionIntentStatus,
    ProtectionOperation,
    ProtectionOperationKind,
    ProtectionOperationPhase,
    ProtectionReasonCode,
    ProtectionState,
    TERMINAL_PROTECTION_INTENT_STATUSES,
    UploadJournalPhase,
    is_protected_and_verified,
    load_state,
    update_state,
)
from ormah.cloud.store_lock import StoreLock, StoreLockTimeout
from ormah.cloud.transfer import download_file, put_file, sha256_file


logger = logging.getLogger(__name__)

_URL_RE = re.compile(r"https?://[^\s]+", re.IGNORECASE)
_BEARER_RE = re.compile(r"\bBearer\s+[^\s]+", re.IGNORECASE)
_AGE_SECRET_RE = re.compile(r"AGE-SECRET-KEY-[A-Z0-9]+", re.IGNORECASE)
_QUERY_SECRET_RE = re.compile(
    r"(?i)((?:^|[?&\s])(?:x-amz-signature|x-amz-credential|x-amz-security-token|"
    r"access_token|token|signature)=)[^&\s]+"
)
_SNAPSHOT_ID_RE = re.compile(r"[0-7][0-9A-HJKMNP-TV-Z]{25}")
_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_NODE_PATH_RE = re.compile(r"\b(nodes|deleted)[/\\][^\s'\",)\]]+\.md\b")
_ABSOLUTE_PATH_RE = re.compile(
    r"(?<![A-Za-z0-9])(?:/[A-Za-z0-9._~+@%=-]+)+|"
    r"(?<![A-Za-z0-9])[A-Za-z]:\\(?:[^\\\s:;,]+\\)*[^\\\s:;,]+"
)
_DISK_FULL_ERRNOS = {errno.ENOSPC, getattr(errno, "EDQUOT", errno.ENOSPC)}


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def safe_error_message(value: object, *sensitive_values: str | None) -> str:
    """Return a useful error without returning or logging credential-bearing material."""

    message = str(value)
    message = _URL_RE.sub("<redacted-url>", message)
    message = _QUERY_SECRET_RE.sub(r"\1<redacted>", message)
    message = _BEARER_RE.sub("Bearer <redacted>", message)
    message = _AGE_SECRET_RE.sub("<redacted-age-key>", message)
    message = _NODE_PATH_RE.sub(r"\1/<redacted>.md", message)
    for sensitive in sensitive_values:
        if sensitive:
            message = message.replace(sensitive, "<redacted>")
    return message[:1000]


def safe_product_error_message(value: object, *sensitive_values: str | None) -> str:
    """Redact local paths in addition to secrets before crossing into the webview."""

    message = safe_error_message(value, *sensitive_values)
    return _ABSOLUTE_PATH_RE.sub("<redacted-path>", message)


def _existing_store_id(memory_dir: Path) -> str | None:
    memory_dir = Path(memory_dir).expanduser()
    if not (memory_dir / STORE_ID_NAME).is_file():
        return None
    return get_or_create_store_id(memory_dir)


def _load_writable_state(store_id: str) -> CloudState:
    state = load_state(store_id)
    if state.schema_version > CURRENT_CLOUD_STATE_SCHEMA_VERSION:
        raise CloudStateVersionError(
            f"Cloud state schema {state.schema_version} is newer than this client's schema "
            f"{CURRENT_CLOUD_STATE_SCHEMA_VERSION}; update Ormah before writing it."
        )
    return state


def _snapshot_files(root: Path) -> dict[str, Path]:
    files: dict[str, Path] = {}
    for dirname in ("nodes", "deleted"):
        source = Path(root) / dirname
        if not source.is_dir():
            continue
        for path in source.glob("*.md"):
            if path.is_file():
                files[f"{dirname}/{path.name}"] = path
    return files


def _backup_matches_memory(backup: BackupInfo, memory_dir: Path) -> bool:
    try:
        if resolve_backup_user_node_id(backup.path) != resolve_current_user_node_id(memory_dir):
            return False
    except BackupError:
        return False

    source_files = _snapshot_files(memory_dir)
    backup_files = _snapshot_files(backup.path)
    if source_files.keys() != backup_files.keys():
        return False
    return all(
        source_files[name].stat().st_size == backup_files[name].stat().st_size
        and sha256_file(source_files[name]) == sha256_file(backup_files[name])
        for name in source_files
    )


def _backup_for_upload(service, *, reason: str = "manual") -> BackupInfo:
    latest = service.latest()
    if latest is not None and _backup_matches_memory(latest, service.memory_dir):
        return latest
    return service.create(reason=reason)


def _upload_due(state: CloudState, interval_hours: int, now: datetime) -> bool:
    if state.last_upload_at is None:
        return True
    return now - state.last_upload_at >= timedelta(hours=interval_hours)


def _known_state(state: CloudState) -> ProtectionState:
    value = state.protection_state
    return value if isinstance(value, ProtectionState) else ProtectionState.ATTENTION_REQUIRED


def _state_after_backup(
    state: CloudState,
    *,
    protection_intent_id: str | None = None,
) -> ProtectionState:
    current = _known_state(state)
    if protection_intent_id is not None:
        return ProtectionState.VERIFYING_FIRST_BACKUP
    if current is ProtectionState.LOCAL_ONLY:
        return ProtectionState.VERIFYING_FIRST_BACKUP
    if current is ProtectionState.STOPPED:
        return ProtectionState.STOPPED
    if current is ProtectionState.UPLOADING_FIRST_BACKUP:
        return ProtectionState.VERIFYING_FIRST_BACKUP
    return ProtectionState.VERIFICATION_PENDING


def _state_after_verification(state: CloudState, snapshot_id: str) -> ProtectionState:
    current = _known_state(state)
    if state.pending_upload_phase is UploadJournalPhase.FINALIZING:
        return current
    if snapshot_id != state.last_successful_backup_snapshot_id:
        return current
    if (
        state.pending_protection_intent_id is not None
        and state.pending_protection_status not in TERMINAL_PROTECTION_INTENT_STATUSES
    ):
        return current
    if current in {
        ProtectionState.VERIFYING_FIRST_BACKUP,
        ProtectionState.CHANGES_PENDING,
        ProtectionState.VERIFICATION_PENDING,
        ProtectionState.ATTENTION_REQUIRED,
        ProtectionState.OFFLINE,
        ProtectionState.PROTECTED,
    }:
        return ProtectionState.PROTECTED
    return current


def _state_after_failure(
    state: CloudState,
    requested: ProtectionState,
) -> ProtectionState:
    current = _known_state(state)
    if current is ProtectionState.STOPPED:
        return ProtectionState.STOPPED
    return requested


def _cleared_upload_journal() -> dict[str, object]:
    return {
        "pending_upload_id": None,
        "pending_upload_snapshot_id": None,
        "pending_upload_operation_id": None,
        "pending_upload_protection_intent_id": None,
        "pending_upload_phase": None,
        "pending_upload_expires_at": None,
    }


def _is_offline_error(error: object) -> bool:
    return isinstance(error, httpx.RequestError) or (
        isinstance(error, CloudError) and error.status_code is None
    )


def _is_disk_full_error(error: object) -> bool:
    return isinstance(error, OSError) and error.errno in _DISK_FULL_ERRNOS


def _is_quota_error(error: object) -> bool:
    if not isinstance(error, CloudError) or error.status_code != 413:
        return False
    detail = error.payload.get("detail") if isinstance(error.payload, dict) else error.payload
    return "quota" in str(detail).lower()


def _cloud_error_code(error: object) -> str | None:
    if not isinstance(error, CloudError) or not isinstance(error.payload, dict):
        return None
    detail = error.payload.get("detail")
    if not isinstance(detail, dict):
        return None
    code = detail.get("code")
    return code if isinstance(code, str) else None


def _finalize_is_definitively_expired(state: CloudState, error: object) -> bool:
    """Return whether replacing a failed finalize cannot create a duplicate snapshot."""

    expires_at = state.pending_upload_expires_at
    return (
        _cloud_error_code(error) == "upload_expired"
        and expires_at is not None
        and expires_at <= _utc_now()
    )


def _validated_snapshot_id(value: object) -> str:
    if not isinstance(value, str) or _SNAPSHOT_ID_RE.fullmatch(value) is None:
        raise RuntimeError("Cloud response contained an invalid snapshot id.")
    return value


def _validated_sha256(value: object) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise _VerificationStageError(
            "Cloud metadata does not contain a verified ciphertext hash.",
            ProtectionReasonCode.CIPHERTEXT_HASH_UNAVAILABLE,
        )
    return value


def _verify_extracted_bundle(extracted: Path, expected_store_id: str, info) -> int:
    try:
        return verify_extracted_bundle(extracted, expected_store_id, info)
    except CloudRestoreValidationError as exc:
        raise _VerificationStageError(
            str(exc),
            ProtectionReasonCode(exc.reason_code),
        ) from exc


class CloudProtectionService:
    """Reusable cloud protection operations constructed from application settings."""

    def __init__(self, settings, *, engine=None) -> None:
        self.settings = settings
        self.engine = engine

    @classmethod
    def from_engine(cls, engine) -> CloudProtectionService:
        return cls(engine.settings, engine=engine)

    def _message(self, value: object) -> str:
        return safe_error_message(value, getattr(self.settings, "account_token", None))

    def create_intent(self) -> ProtectionOperation:
        """Create or resume a 30-minute durable Protect action."""

        operation_id = str(uuid.uuid4())
        try:
            with StoreLock(self.settings.memory_dir):
                memory_dir = Path(self.settings.memory_dir).expanduser()
                created_store = not (memory_dir / STORE_ID_NAME).is_file()
                store_id = get_or_create_store_id(memory_dir)
                state = _load_writable_state(store_id)
                if not isinstance(state.protection_state, ProtectionState):
                    return self._client_update_required_operation(
                        operation_id, ProtectionOperationKind.ENABLE
                    )
                signed_in = bool(
                    isinstance(getattr(self.settings, "account_token", None), str)
                    and self.settings.account_token.strip()
                )

                now = _utc_now()
                if (
                    state.pending_protection_intent_id is not None
                    and state.pending_protection_store_id == store_id
                    and (
                        state.pending_protection_status
                        is ProtectionIntentStatus.RUNNING
                        or (
                            state.pending_protection_expires_at is not None
                            and state.pending_protection_expires_at > now
                        )
                    )
                    and state.pending_protection_status
                    not in {
                        ProtectionIntentStatus.COMPLETED,
                        ProtectionIntentStatus.CANCELED,
                        ProtectionIntentStatus.EXPIRED,
                    }
                ):
                    operation_id = state.pending_protection_intent_id
                    return ProtectionOperation(
                        operation_id,
                        ProtectionOperationKind.ENABLE,
                        ProtectionOperationPhase.PENDING,
                        _known_state(state),
                        protection_intent_id=state.pending_protection_intent_id,
                    )

                completed_protection = (
                    state.protection_state is ProtectionState.PROTECTED
                    and is_protected_and_verified(
                        state,
                        enabled=bool(self.settings.cloud_backup_enabled),
                    )
                )
                completed_entitlement = (
                    check_entitlement(self.settings)
                    if completed_protection
                    else None
                )
                if completed_entitlement in {
                    EntitlementStatus.ACTIVE,
                    EntitlementStatus.GRACE,
                } or (
                    signed_in
                    and completed_entitlement is EntitlementStatus.NONE
                ):
                    operation_id = state.pending_protection_intent_id or operation_id
                    return ProtectionOperation(
                        operation_id,
                        ProtectionOperationKind.ENABLE,
                        ProtectionOperationPhase.COMPLETED,
                        ProtectionState.PROTECTED,
                        snapshot_id=state.last_verified_snapshot_id,
                        protection_intent_id=state.pending_protection_intent_id,
                    )

                intent_id = operation_id
                next_state = (
                    ProtectionState.PAUSED
                    if signed_in
                    and completed_entitlement is EntitlementStatus.EXPIRED
                    else state.protection_state
                    if signed_in
                    else ProtectionState.SIGN_IN_REQUIRED
                )
                update_state(
                    store_id,
                    memory_dir=memory_dir,
                    protection_state=next_state,
                    pending_protection_intent_id=intent_id,
                    pending_protection_account_id=None,
                    pending_protection_store_id=store_id,
                    pending_protection_created_at=now,
                    pending_protection_expires_at=now + timedelta(minutes=30),
                    pending_protection_status=ProtectionIntentStatus.PENDING,
                    pending_protection_snapshot_id=None,
                    pending_protection_created_store=(
                        created_store
                        or (
                            state.pending_protection_created_store
                            and state.last_successful_backup_snapshot_id is None
                        )
                    ),
                    pending_protection_origin_state=self._new_intent_origin(state),
                    last_operation_id=operation_id,
                    last_operation_kind=ProtectionOperationKind.ENABLE,
                    last_operation_phase=ProtectionOperationPhase.PENDING,
                    last_error_code=(
                        None if signed_in else ProtectionReasonCode.SIGN_IN_REQUIRED
                    ),
                    last_error_message=None,
                    last_error_at=None,
                )
                return ProtectionOperation(
                    operation_id,
                    ProtectionOperationKind.ENABLE,
                    ProtectionOperationPhase.PENDING,
                    next_state,
                    None if signed_in else ProtectionReasonCode.SIGN_IN_REQUIRED,
                    protection_intent_id=intent_id,
                )
        except StoreLockTimeout:
            return self._store_busy_operation(operation_id, ProtectionOperationKind.ENABLE)
        except CloudStateVersionError:
            return self._client_update_required_operation(
                operation_id, ProtectionOperationKind.ENABLE
            )
        except Exception as exc:
            return self._enable_failure(operation_id, exc)

    def bind_intent(self, intent_id: str) -> ProtectionOperation:
        """Bind an intent to the account from a fresh authenticated response."""

        operation_id = str(uuid.uuid4())
        try:
            intent_id = validate_protection_intent_id(intent_id)
            operation_id = intent_id
        except ValueError:
            return self._invalid_intent_operation(operation_id)
        try:
            with StoreLock(self.settings.memory_dir):
                return self._bind_intent(operation_id, intent_id)
        except StoreLockTimeout:
            return self._store_busy_operation(
                operation_id,
                ProtectionOperationKind.ENABLE,
                protection_intent_id=intent_id,
            )
        except _EnablePrerequisiteError as exc:
            if exc.reason_code is ProtectionReasonCode.INTENT_CANCELED:
                return self._invalid_intent_operation(operation_id, intent_id)
            return self._enable_failure(operation_id, exc, intent_id=intent_id)
        except CloudStateVersionError:
            return self._client_update_required_operation(
                operation_id,
                ProtectionOperationKind.ENABLE,
            )
        except Exception as exc:
            return self._enable_failure(operation_id, exc, intent_id=intent_id)

    def _bind_intent(
        self, operation_id: str, intent_id: str
    ) -> ProtectionOperation:
        store_id, state = self._intent_state(intent_id)
        terminal = self._terminal_intent_operation(operation_id, intent_id, state)
        if terminal is not None:
            return terminal
        if self._expire_intent_if_needed(store_id, state):
            return ProtectionOperation(
                operation_id,
                ProtectionOperationKind.ENABLE,
                ProtectionOperationPhase.CANCELED,
                self._intent_origin_state(state),
                ProtectionReasonCode.INTENT_EXPIRED,
                "This protection request has expired. Start again.",
                protection_intent_id=intent_id,
            )

        token = getattr(self.settings, "account_token", None)
        if not isinstance(token, str) or not token.strip():
            update_state(
                store_id,
                memory_dir=self.settings.memory_dir,
                protection_state=ProtectionState.SIGN_IN_REQUIRED,
                last_operation_id=operation_id,
                last_operation_kind=ProtectionOperationKind.ENABLE,
                last_operation_phase=ProtectionOperationPhase.PENDING,
                last_error_code=ProtectionReasonCode.SIGN_IN_REQUIRED,
                last_error_message=None,
                last_error_at=None,
            )
            return ProtectionOperation(
                operation_id,
                ProtectionOperationKind.ENABLE,
                ProtectionOperationPhase.PENDING,
                ProtectionState.SIGN_IN_REQUIRED,
                ProtectionReasonCode.SIGN_IN_REQUIRED,
                "Sign in to continue protecting this memory.",
                protection_intent_id=intent_id,
            )

        client = None
        try:
            client = client_from_settings(self.settings)
            entitlements = client.get_entitlements()
        except Exception as exc:
            reason = (
                ProtectionReasonCode.SIGN_IN_REQUIRED
                if isinstance(exc, CloudError) and exc.status_code == 401
                else ProtectionReasonCode.OFFLINE
                if _is_offline_error(exc)
                else ProtectionReasonCode.OPERATION_INTERRUPTED
            )
            next_state = (
                ProtectionState.SIGN_IN_REQUIRED
                if reason is ProtectionReasonCode.SIGN_IN_REQUIRED
                else ProtectionState.OFFLINE
                if reason is ProtectionReasonCode.OFFLINE
                else ProtectionState.ATTENTION_REQUIRED
            )
            return self._enable_failure(
                operation_id,
                exc,
                store_id=store_id,
                intent_id=intent_id,
                reason_code=reason,
                protection_state=next_state,
            )
        finally:
            self._close_client(client)

        if not isinstance(entitlements, dict):
            return self._enable_failure(
                operation_id,
                "Ormah Cloud returned malformed account metadata.",
                store_id=store_id,
                intent_id=intent_id,
            )
        try:
            account_id = self._validated_account_id(entitlements.get("account_id"))
        except ValueError as exc:
            return self._enable_failure(
                operation_id,
                exc,
                store_id=store_id,
                intent_id=intent_id,
            )
        try:
            cache_entitlements(entitlements, fetched_at=_utc_now())
        except OSError as exc:
            logger.warning(
                "Could not cache fresh Ormah Cloud entitlement state: %s",
                self._message(exc),
            )

        # Re-read after the network boundary even though this process retains the
        # canonical lock; another process may have completed a pre-existing state write.
        state = _load_writable_state(store_id)
        terminal = self._terminal_intent_operation(operation_id, intent_id, state)
        if terminal is not None:
            return terminal
        if state.pending_protection_account_id not in {None, account_id}:
            origin = self._intent_origin_state(state)
            update_state(
                store_id,
                memory_dir=self.settings.memory_dir,
                protection_state=origin,
                pending_protection_status=ProtectionIntentStatus.CANCELED,
                last_operation_id=operation_id,
                last_operation_kind=ProtectionOperationKind.ENABLE,
                last_operation_phase=ProtectionOperationPhase.CANCELED,
                last_error_code=ProtectionReasonCode.ACCOUNT_CHANGED,
                last_error_message="This protection request belongs to another account.",
                last_error_at=_utc_now(),
            )
            return ProtectionOperation(
                operation_id,
                ProtectionOperationKind.ENABLE,
                ProtectionOperationPhase.CANCELED,
                origin,
                ProtectionReasonCode.ACCOUNT_CHANGED,
                "This protection request was canceled because the account changed.",
                protection_intent_id=intent_id,
            )

        has_subscription = entitlements.get("backup") is True
        if state.pending_protection_status is ProtectionIntentStatus.RUNNING:
            status = ProtectionIntentStatus.RUNNING
            next_state = _known_state(state)
        elif has_subscription:
            status = ProtectionIntentStatus.READY
            next_state = self._intent_origin_state(state)
        elif state.pending_protection_status is ProtectionIntentStatus.CHECKOUT_PENDING:
            status = ProtectionIntentStatus.CHECKOUT_PENDING
            next_state = ProtectionState.SUBSCRIPTION_REQUIRED
        else:
            status = ProtectionIntentStatus.ACCOUNT_BOUND
            next_state = ProtectionState.SUBSCRIPTION_REQUIRED
        changes: dict[str, object] = {
            "protection_state": next_state,
            "pending_protection_account_id": account_id,
            "pending_protection_status": status,
        }
        if status is not ProtectionIntentStatus.RUNNING:
            changes.update(
                last_operation_id=operation_id,
                last_operation_kind=ProtectionOperationKind.ENABLE,
                last_operation_phase=ProtectionOperationPhase.PENDING,
                last_error_code=(
                    None
                    if has_subscription
                    else ProtectionReasonCode.SUBSCRIPTION_REQUIRED
                ),
                last_error_message=None,
                last_error_at=None,
            )
        update_state(
            store_id,
            memory_dir=self.settings.memory_dir,
            **changes,
        )
        return ProtectionOperation(
            operation_id,
            ProtectionOperationKind.ENABLE,
            ProtectionOperationPhase.PENDING,
            next_state,
            None if has_subscription else ProtectionReasonCode.SUBSCRIPTION_REQUIRED,
            protection_intent_id=intent_id,
        )

    def cancel_intent(self, intent_id: str) -> ProtectionOperation:
        """Cancel a pending Protect action without removing recovery material."""

        operation_id = str(uuid.uuid4())
        try:
            intent_id = validate_protection_intent_id(intent_id)
            operation_id = intent_id
        except ValueError:
            return self._invalid_intent_operation(operation_id)
        try:
            with StoreLock(self.settings.memory_dir):
                store_id, state = self._intent_state(intent_id)
                terminal = self._terminal_intent_operation(operation_id, intent_id, state)
                if terminal is not None:
                    return terminal
                if self._expire_intent_if_needed(store_id, state):
                    return ProtectionOperation(
                        operation_id,
                        ProtectionOperationKind.ENABLE,
                        ProtectionOperationPhase.CANCELED,
                        self._intent_origin_state(state),
                        ProtectionReasonCode.INTENT_EXPIRED,
                        protection_intent_id=intent_id,
                    )
                if state.pending_protection_status is ProtectionIntentStatus.RUNNING:
                    return self._enable_failure(
                        operation_id,
                        "Protection has already started; stop cloud protection instead.",
                        store_id=store_id,
                        intent_id=intent_id,
                        reason_code=ProtectionReasonCode.OPERATION_INTERRUPTED,
                    )
                origin = self._intent_origin_state(state)
                update_state(
                    store_id,
                    memory_dir=self.settings.memory_dir,
                    protection_state=origin,
                    pending_protection_status=ProtectionIntentStatus.CANCELED,
                    last_operation_id=operation_id,
                    last_operation_kind=ProtectionOperationKind.ENABLE,
                    last_operation_phase=ProtectionOperationPhase.CANCELED,
                    last_error_code=ProtectionReasonCode.INTENT_CANCELED,
                    last_error_message=None,
                    last_error_at=None,
                )
                return ProtectionOperation(
                    operation_id,
                    ProtectionOperationKind.ENABLE,
                    ProtectionOperationPhase.CANCELED,
                    origin,
                    ProtectionReasonCode.INTENT_CANCELED,
                    protection_intent_id=intent_id,
                )
        except StoreLockTimeout:
            return self._store_busy_operation(
                operation_id,
                ProtectionOperationKind.ENABLE,
                protection_intent_id=intent_id,
            )
        except _EnablePrerequisiteError as exc:
            if exc.reason_code is ProtectionReasonCode.INTENT_CANCELED:
                return self._invalid_intent_operation(operation_id, intent_id)
            return self._enable_failure(operation_id, exc, intent_id=intent_id)
        except CloudStateVersionError:
            return self._client_update_required_operation(
                operation_id,
                ProtectionOperationKind.ENABLE,
            )
        except Exception as exc:
            return self._enable_failure(operation_id, exc, intent_id=intent_id)

    def enable(self, intent_id: str) -> ProtectionOperation:
        """Enable protection, upload immediately, and verify the exact snapshot."""

        operation_id = str(uuid.uuid4())
        try:
            intent_id = validate_protection_intent_id(intent_id)
            operation_id = intent_id
        except ValueError:
            return self._invalid_intent_operation(operation_id)
        try:
            with StoreLock(self.settings.memory_dir):
                bound = self._bind_intent(operation_id, intent_id)
                store_id, state = self._intent_state(intent_id)
                if bound.reason_code is not None or state.pending_protection_status not in {
                    ProtectionIntentStatus.READY,
                    ProtectionIntentStatus.RUNNING,
                }:
                    return bound

                if state.pending_protection_status is ProtectionIntentStatus.READY:
                    update_state(
                        store_id,
                        memory_dir=self.settings.memory_dir,
                        protection_state=ProtectionState.INITIALIZING,
                        pending_protection_status=ProtectionIntentStatus.RUNNING,
                        last_operation_id=operation_id,
                        last_operation_kind=ProtectionOperationKind.ENABLE,
                        last_operation_phase=ProtectionOperationPhase.RUNNING,
                        last_error_code=None,
                        last_error_message=None,
                        last_error_at=None,
                    )

                try:
                    if not key_file_exists():
                        if not state.pending_protection_created_store:
                            raise _EnablePrerequisiteError(
                                "Cloud encryption key is missing; import the recovery kit first.",
                                ProtectionReasonCode.KEY_MISSING,
                            )
                        init_key()
                    load_identities()
                    self._validate_recovery_kit_store(store_id)
                    write_recovery_kit(
                        store_id,
                        account_email=getattr(self.settings, "account_email", None),
                    )
                    RecoveryKitService(self.settings).validate_canonical_kit()
                except _EnablePrerequisiteError:
                    raise
                except CloudKeyError as exc:
                    raise _EnablePrerequisiteError(
                        str(exc), ProtectionReasonCode.KEY_INVALID
                    ) from exc
                except Exception as exc:
                    raise _EnablePrerequisiteError(
                        str(exc), ProtectionReasonCode.KEY_INVALID
                    ) from exc

                set_cloud_backup_enabled(self.settings, True)
                state = _load_writable_state(store_id)
                snapshot_id = self._retry_snapshot(state, intent_id)
                if snapshot_id is None:
                    update_state(
                        store_id,
                        memory_dir=self.settings.memory_dir,
                        protection_state=ProtectionState.UPLOADING_FIRST_BACKUP,
                        last_operation_id=operation_id,
                        last_operation_kind=ProtectionOperationKind.ENABLE,
                        last_operation_phase=ProtectionOperationPhase.UPLOADING,
                    )
                    backup = self._backup_now(
                        operation_id,
                        reason="protection-enable",
                        only_if_due=False,
                        entitlement_verified=True,
                        protection_intent_id=intent_id,
                    )
                    if backup.phase is not ProtectionOperationPhase.COMPLETED:
                        return self._enable_stage_result(
                            store_id,
                            intent_id,
                            backup,
                        )
                    snapshot_id = backup.snapshot_id
                assert snapshot_id is not None

                update_state(
                    store_id,
                    memory_dir=self.settings.memory_dir,
                    protection_state=ProtectionState.VERIFYING_FIRST_BACKUP,
                    last_operation_id=operation_id,
                    last_operation_kind=ProtectionOperationKind.ENABLE,
                    last_operation_phase=ProtectionOperationPhase.VERIFYING,
                )
                verification = self._verify_now(
                    operation_id, requested_snapshot_id=snapshot_id
                )
                if verification.phase is not ProtectionOperationPhase.COMPLETED:
                    return self._enable_stage_result(
                        store_id,
                        intent_id,
                        verification,
                    )

                final = _load_writable_state(store_id)
                if (
                    final.pending_protection_intent_id != intent_id
                    or final.pending_protection_status is not ProtectionIntentStatus.RUNNING
                    or final.last_successful_backup_snapshot_id != snapshot_id
                    or final.pending_protection_snapshot_id != snapshot_id
                    or final.last_verified_snapshot_id != snapshot_id
                    or final.last_verify_ok is not True
                ):
                    return self._enable_failure(
                        operation_id,
                        "Protection verification did not match the uploaded snapshot.",
                        store_id=store_id,
                        intent_id=intent_id,
                        reason_code=ProtectionReasonCode.VERIFICATION_FAILED,
                    )
                now = _utc_now()
                update_state(
                    store_id,
                    memory_dir=self.settings.memory_dir,
                    protection_enabled_at=final.protection_enabled_at or now,
                    protection_disabled_at=None,
                    protection_state=ProtectionState.PROTECTED,
                    pending_protection_status=ProtectionIntentStatus.COMPLETED,
                    last_operation_id=operation_id,
                    last_operation_kind=ProtectionOperationKind.ENABLE,
                    last_operation_phase=ProtectionOperationPhase.COMPLETED,
                    last_error_code=None,
                    last_error_message=None,
                    last_error_at=None,
                )
                return ProtectionOperation(
                    operation_id,
                    ProtectionOperationKind.ENABLE,
                    ProtectionOperationPhase.COMPLETED,
                    ProtectionState.PROTECTED,
                    snapshot_id=snapshot_id,
                    protection_intent_id=intent_id,
                    verified_node_count=verification.verified_node_count,
                )
        except StoreLockTimeout:
            return self._store_busy_operation(
                operation_id,
                ProtectionOperationKind.ENABLE,
                protection_intent_id=intent_id,
            )
        except _EnablePrerequisiteError as exc:
            if exc.reason_code is ProtectionReasonCode.INTENT_CANCELED:
                return self._invalid_intent_operation(operation_id, intent_id)
            return self._enable_failure(
                operation_id,
                exc,
                intent_id=intent_id,
                reason_code=exc.reason_code,
            )
        except CloudStateVersionError:
            return self._client_update_required_operation(
                operation_id,
                ProtectionOperationKind.ENABLE,
            )
        except Exception as exc:
            return self._enable_failure(operation_id, exc, intent_id=intent_id)

    def disable(self) -> ProtectionOperation:
        """Stop future uploads while preserving account and recovery history."""

        operation_id = str(uuid.uuid4())
        try:
            with StoreLock(self.settings.memory_dir):
                store_id = _existing_store_id(self.settings.memory_dir)
                result_state = ProtectionState.LOCAL_ONLY
                if store_id is not None:
                    state = _load_writable_state(store_id)
                    if not isinstance(state.protection_state, ProtectionState):
                        return self._client_update_required_operation(
                            operation_id, ProtectionOperationKind.DISABLE
                        )
                    pending_status = state.pending_protection_status
                    protection_started = (
                        bool(self.settings.cloud_backup_enabled)
                        or state.protection_state is ProtectionState.STOPPED
                        or state.protection_disabled_at is not None
                        or state.last_successful_backup_snapshot_id is not None
                        or pending_status
                        in {
                            ProtectionIntentStatus.RUNNING,
                            ProtectionIntentStatus.COMPLETED,
                        }
                        or state.protection_enabled_at is not None
                    )
                    target_state = (
                        ProtectionState.STOPPED
                        if protection_started
                        else self._intent_origin_state(state)
                    )
                    if pending_status not in {
                        None,
                        ProtectionIntentStatus.COMPLETED,
                        ProtectionIntentStatus.CANCELED,
                        ProtectionIntentStatus.EXPIRED,
                    }:
                        pending_status = ProtectionIntentStatus.CANCELED
                    update_state(
                        store_id,
                        memory_dir=self.settings.memory_dir,
                        protection_state=target_state,
                        protection_disabled_at=(
                            state.protection_disabled_at
                            if target_state is not ProtectionState.STOPPED
                            or state.protection_state is ProtectionState.STOPPED
                            else _utc_now()
                        ),
                        pending_protection_status=pending_status,
                        last_operation_id=operation_id,
                        last_operation_kind=ProtectionOperationKind.DISABLE,
                        last_operation_phase=ProtectionOperationPhase.COMPLETED,
                        last_error_code=None,
                        last_error_message=None,
                        last_error_at=None,
                    )
                    result_state = target_state
                set_cloud_backup_enabled(self.settings, False)
                return ProtectionOperation(
                    operation_id,
                    ProtectionOperationKind.DISABLE,
                    ProtectionOperationPhase.COMPLETED,
                    result_state,
                )
        except StoreLockTimeout:
            return self._store_busy_operation(operation_id, ProtectionOperationKind.DISABLE)
        except CloudStateVersionError:
            return self._client_update_required_operation(
                operation_id,
                ProtectionOperationKind.DISABLE,
            )
        except Exception as exc:
            return self._enable_failure(
                operation_id,
                exc,
                kind=ProtectionOperationKind.DISABLE,
            )

    def _intent_state(self, intent_id: str) -> tuple[str, CloudState]:
        store_id = _existing_store_id(self.settings.memory_dir)
        if store_id is None:
            raise _EnablePrerequisiteError(
                "This protection request does not belong to this memory store.",
                ProtectionReasonCode.INTENT_CANCELED,
            )
        state = _load_writable_state(store_id)
        if not isinstance(state.protection_state, ProtectionState):
            raise CloudStateVersionError("Unknown protection state")
        if (
            state.pending_protection_intent_id != intent_id
            or state.pending_protection_store_id != store_id
        ):
            raise _EnablePrerequisiteError(
                "This protection request does not belong to this memory store.",
                ProtectionReasonCode.INTENT_CANCELED,
            )
        return store_id, state

    @staticmethod
    def _validated_account_id(value: object) -> str:
        if not isinstance(value, str):
            raise ValueError("Ormah Cloud did not return a valid account identifier.")
        try:
            parsed = uuid.UUID(value)
        except ValueError as exc:
            raise ValueError(
                "Ormah Cloud did not return a valid account identifier."
            ) from exc
        if parsed.version != 4 or parsed.variant != uuid.RFC_4122:
            raise ValueError("Ormah Cloud did not return a valid account identifier.")
        return str(parsed)

    @staticmethod
    def _intent_origin_state(state: CloudState) -> ProtectionState:
        origin = state.pending_protection_origin_state
        return origin if isinstance(origin, ProtectionState) else ProtectionState.LOCAL_ONLY

    @staticmethod
    def _new_intent_origin(state: CloudState) -> ProtectionState:
        if (
            state.pending_protection_intent_id is not None
            and state.pending_protection_status is not ProtectionIntentStatus.COMPLETED
            and isinstance(state.pending_protection_origin_state, ProtectionState)
        ):
            return state.pending_protection_origin_state
        if isinstance(state.protection_state, ProtectionState) and state.protection_state not in {
            ProtectionState.SIGN_IN_REQUIRED,
            ProtectionState.SUBSCRIPTION_REQUIRED,
            ProtectionState.INITIALIZING,
            ProtectionState.UPLOADING_FIRST_BACKUP,
            ProtectionState.VERIFYING_FIRST_BACKUP,
        }:
            return state.protection_state
        return ProtectionState.LOCAL_ONLY

    def _expire_intent_if_needed(self, store_id: str, state: CloudState) -> bool:
        if state.pending_protection_status is ProtectionIntentStatus.RUNNING:
            return False
        expires_at = state.pending_protection_expires_at
        if expires_at is not None and expires_at > _utc_now():
            return False
        update_state(
            store_id,
            memory_dir=self.settings.memory_dir,
            protection_state=self._intent_origin_state(state),
            pending_protection_status=ProtectionIntentStatus.EXPIRED,
            last_error_code=ProtectionReasonCode.INTENT_EXPIRED,
            last_error_message=None,
            last_error_at=None,
        )
        return True

    def _terminal_intent_operation(
        self,
        operation_id: str,
        intent_id: str,
        state: CloudState,
    ) -> ProtectionOperation | None:
        status = state.pending_protection_status
        if status is ProtectionIntentStatus.COMPLETED:
            current_state = _known_state(state)
            if (
                current_state is ProtectionState.PROTECTED
                and not is_protected_and_verified(
                    state,
                    enabled=bool(self.settings.cloud_backup_enabled),
                )
            ):
                current_state = ProtectionState.ATTENTION_REQUIRED
            return ProtectionOperation(
                operation_id,
                ProtectionOperationKind.ENABLE,
                ProtectionOperationPhase.COMPLETED,
                current_state,
                snapshot_id=state.last_verified_snapshot_id,
                protection_intent_id=intent_id,
            )
        if status in {ProtectionIntentStatus.CANCELED, ProtectionIntentStatus.EXPIRED}:
            reason = (
                ProtectionReasonCode.INTENT_EXPIRED
                if status is ProtectionIntentStatus.EXPIRED
                else ProtectionReasonCode.INTENT_CANCELED
            )
            return ProtectionOperation(
                operation_id,
                ProtectionOperationKind.ENABLE,
                ProtectionOperationPhase.CANCELED,
                _known_state(state),
                reason,
                protection_intent_id=intent_id,
            )
        return None

    @staticmethod
    def _retry_snapshot(state: CloudState, intent_id: str) -> str | None:
        if state.pending_protection_status is not ProtectionIntentStatus.RUNNING:
            return None
        if state.last_operation_id != intent_id:
            return None
        snapshot_id = state.pending_protection_snapshot_id
        if snapshot_id is None:
            return None
        if snapshot_id != state.last_successful_backup_snapshot_id:
            return None
        if state.protection_state is ProtectionState.VERIFYING_FIRST_BACKUP:
            return snapshot_id
        if state.last_verify_snapshot_id == snapshot_id:
            return snapshot_id
        return None

    @staticmethod
    def _upload_status_is_unknown(state: CloudState) -> bool:
        return (
            state.pending_upload_phase is UploadJournalPhase.FINALIZING
            or state.last_error_code is ProtectionReasonCode.UPLOAD_STATUS_UNKNOWN
        )

    @staticmethod
    def _validate_recovery_kit_store(store_id: str) -> None:
        kit_path = RECOVERY_KIT_PATH.expanduser()
        if not kit_path.is_file():
            return
        existing_store_id = extract_store_id(str(kit_path))
        if existing_store_id != store_id:
            raise _EnablePrerequisiteError(
                "This Ormah installation already has recovery material for another "
                "memory store. Version 1 supports one protected primary memory.",
                ProtectionReasonCode.KEY_INVALID,
            )

    def _enable_stage_result(
        self,
        store_id: str,
        intent_id: str,
        result: ProtectionOperation,
    ) -> ProtectionOperation:
        """Keep child backup/verify failures attached to the stable Protect operation."""

        update_state(
            store_id,
            memory_dir=self.settings.memory_dir,
            last_operation_id=intent_id,
            last_operation_kind=ProtectionOperationKind.ENABLE,
            last_operation_phase=result.phase,
            last_error_code=result.reason_code,
            last_error_message=result.message,
            last_error_at=_utc_now(),
        )
        return replace(
            result,
            operation_id=intent_id,
            kind=ProtectionOperationKind.ENABLE,
            protection_intent_id=intent_id,
        )

    def _enable_failure(
        self,
        operation_id: str,
        error: object,
        *,
        store_id: str | None = None,
        intent_id: str | None = None,
        kind: ProtectionOperationKind = ProtectionOperationKind.ENABLE,
        reason_code: ProtectionReasonCode = ProtectionReasonCode.OPERATION_INTERRUPTED,
        protection_state: ProtectionState = ProtectionState.ATTENTION_REQUIRED,
    ) -> ProtectionOperation:
        if isinstance(error, _EnablePrerequisiteError):
            reason_code = error.reason_code
        message = self._message(error)
        if store_id is None:
            try:
                store_id = _existing_store_id(self.settings.memory_dir)
            except Exception:
                store_id = None
        if store_id is not None:
            try:
                state = _load_writable_state(store_id)
                if isinstance(state.protection_state, ProtectionState):
                    if (
                        kind is ProtectionOperationKind.DISABLE
                        and state.protection_state is ProtectionState.STOPPED
                    ):
                        protection_state = ProtectionState.STOPPED
                    update_state(
                        store_id,
                        memory_dir=self.settings.memory_dir,
                        protection_state=protection_state,
                        last_operation_id=operation_id,
                        last_operation_kind=kind,
                        last_operation_phase=ProtectionOperationPhase.FAILED,
                        last_error_code=reason_code,
                        last_error_message=message,
                        last_error_at=_utc_now(),
                    )
            except Exception as state_error:
                logger.warning(
                    "Could not persist Ormah Cloud enablement failure: %s",
                    self._message(state_error),
                )
        return ProtectionOperation(
            operation_id,
            kind,
            ProtectionOperationPhase.FAILED,
            protection_state,
            reason_code,
            message,
            protection_intent_id=intent_id,
        )

    def _invalid_intent_operation(
        self,
        operation_id: str,
        intent_id: str | None = None,
    ) -> ProtectionOperation:
        """Reject a stale or foreign intent without changing current store health."""

        current_state = ProtectionState.LOCAL_ONLY
        try:
            store_id = _existing_store_id(self.settings.memory_dir)
            if store_id is not None:
                current_state = _known_state(load_state(store_id))
        except Exception:
            current_state = ProtectionState.ATTENTION_REQUIRED
        return ProtectionOperation(
            operation_id,
            ProtectionOperationKind.ENABLE,
            ProtectionOperationPhase.CANCELED,
            current_state,
            ProtectionReasonCode.INTENT_CANCELED,
            "This protection request is no longer active. Start again.",
            protection_intent_id=intent_id,
        )

    def backup_now(
        self,
        reason: str = "manual",
        *,
        only_if_due: bool = False,
    ) -> ProtectionOperation:
        """Create, encrypt, upload, and finalize one backup."""

        operation_id = str(uuid.uuid4())
        try:
            with StoreLock(self.settings.memory_dir):
                return self._backup_now(
                    operation_id,
                    reason=reason,
                    only_if_due=only_if_due,
                    entitlement_verified=False,
                )
        except StoreLockTimeout:
            return self._store_busy_operation(
                operation_id,
                ProtectionOperationKind.BACKUP,
            )
        except Exception as exc:
            return self._backup_failure(operation_id, exc)

    def backup_and_verify(self, reason: str = "manual-ui") -> ProtectionOperation:
        """Upload one recovery point and restore-test that exact snapshot.

        This is the product-facing manual action. Scheduled backups continue to
        use :meth:`backup_now` so they do not download every uploaded snapshot.
        """

        operation_id = str(uuid.uuid4())
        try:
            with StoreLock(self.settings.memory_dir):
                backup = self._backup_now(
                    operation_id,
                    reason=reason,
                    only_if_due=False,
                    entitlement_verified=False,
                )
                if backup.phase is not ProtectionOperationPhase.COMPLETED:
                    return backup
                verification = self._verify_now(
                    operation_id,
                    requested_snapshot_id=backup.snapshot_id,
                )
                return ProtectionOperation(
                    operation_id,
                    ProtectionOperationKind.BACKUP,
                    verification.phase,
                    verification.state,
                    verification.reason_code,
                    verification.message,
                    verification.snapshot_id,
                    verified_node_count=verification.verified_node_count,
                )
        except StoreLockTimeout:
            return self._store_busy_operation(
                operation_id,
                ProtectionOperationKind.BACKUP,
            )
        except Exception as exc:
            return self._verification_failure(operation_id, exc)

    def _record_upload_success(
        self,
        *,
        store_id: str,
        operation_id: str,
        snapshot_id: str,
        protection_intent_id: str | None,
    ) -> ProtectionOperation:
        state = _load_writable_state(store_id)
        next_state = _state_after_backup(
            state,
            protection_intent_id=protection_intent_id,
        )
        uploaded_at = _utc_now()
        changes: dict[str, object] = {
            "last_upload_at": uploaded_at,
            "last_upload_snapshot_id": snapshot_id,
            "last_upload_error": None,
            "last_successful_upload_at": uploaded_at,
            "last_successful_backup_snapshot_id": snapshot_id,
            "protection_state": next_state,
            "last_operation_id": operation_id,
            "last_operation_kind": ProtectionOperationKind.BACKUP,
            "last_operation_phase": ProtectionOperationPhase.COMPLETED,
            "last_error_code": None,
            "last_error_message": None,
            "last_error_at": None,
            **_cleared_upload_journal(),
        }
        if protection_intent_id is not None:
            changes["pending_protection_snapshot_id"] = snapshot_id
        update_state(
            store_id,
            memory_dir=self.settings.memory_dir,
            **changes,
        )
        logger.info("Uploaded encrypted Ormah Cloud snapshot %s", self._message(snapshot_id))
        return ProtectionOperation(
            operation_id,
            ProtectionOperationKind.BACKUP,
            ProtectionOperationPhase.COMPLETED,
            next_state,
            snapshot_id=snapshot_id,
        )

    def _reconcile_finalizing_upload(
        self,
        *,
        store_id: str,
        state: CloudState,
        operation_id: str,
        protection_intent_id: str | None,
        client,
    ) -> ProtectionOperation:
        upload_id = state.pending_upload_id
        snapshot_id = state.pending_upload_snapshot_id
        if not isinstance(upload_id, str) or not upload_id:
            raise RuntimeError("Finalizing upload is missing its upload id.")
        snapshot_id = _validated_snapshot_id(snapshot_id)
        finalized = client.finalize_upload(store_id, upload_id)
        finalized_snapshot = _validated_snapshot_id(finalized.get("snapshot_id"))
        if finalized.get("status") != "committed" or finalized_snapshot != snapshot_id:
            raise RuntimeError("Cloud upload finalize response was malformed.")
        return self._record_upload_success(
            store_id=store_id,
            operation_id=operation_id,
            snapshot_id=snapshot_id,
            protection_intent_id=(
                protection_intent_id
                or state.pending_upload_protection_intent_id
            ),
        )

    def _backup_now(
        self,
        operation_id: str,
        *,
        reason: str,
        only_if_due: bool,
        entitlement_verified: bool = False,
        protection_intent_id: str | None = None,
    ) -> ProtectionOperation:
        settings = self.settings
        store_id: str | None = None
        client = None
        finalize_attempted = False
        try:
            store_id = _existing_store_id(settings.memory_dir)
            try:
                state = _load_writable_state(store_id) if store_id is not None else None
            except CloudStateVersionError:
                return self._client_update_required_operation(
                    operation_id,
                    ProtectionOperationKind.BACKUP,
                )
            if state is not None and not isinstance(state.protection_state, ProtectionState):
                return self._client_update_required_operation(
                    operation_id,
                    ProtectionOperationKind.BACKUP,
                )

            if not settings.cloud_backup_enabled:
                logger.debug("Ormah Cloud backup is disabled")
                return ProtectionOperation(
                    operation_id,
                    ProtectionOperationKind.BACKUP,
                    ProtectionOperationPhase.CANCELED,
                    state.protection_state if state is not None else ProtectionState.LOCAL_ONLY,
                    ProtectionReasonCode.NOT_ENABLED,
                    "Cloud backup is disabled.",
                )

            if state is not None and state.protection_state is ProtectionState.STOPPED:
                return ProtectionOperation(
                    operation_id,
                    ProtectionOperationKind.BACKUP,
                    ProtectionOperationPhase.CANCELED,
                    ProtectionState.STOPPED,
                    ProtectionReasonCode.PROTECTION_STOPPED,
                    "Cloud protection is stopped for this memory store.",
                )
            if state is not None and state.pending_upload_phase is not None:
                if not isinstance(state.pending_upload_phase, UploadJournalPhase):
                    return self._client_update_required_operation(
                        operation_id,
                        ProtectionOperationKind.BACKUP,
                    )
                if state.pending_upload_phase is UploadJournalPhase.FINALIZING:
                    client = client_from_settings(settings)
                    finalize_attempted = True
                    try:
                        return self._reconcile_finalizing_upload(
                            store_id=store_id,
                            state=state,
                            operation_id=operation_id,
                            protection_intent_id=protection_intent_id,
                            client=client,
                        )
                    except CloudError as exc:
                        if not _finalize_is_definitively_expired(state, exc):
                            raise
                        self._close_client(client)
                        client = None
                        finalize_attempted = False
                        state = update_state(
                            store_id,
                            memory_dir=settings.memory_dir,
                            last_upload_error=None,
                            last_error_code=None,
                            last_error_message=None,
                            last_error_at=None,
                            **_cleared_upload_journal(),
                        )
                state = update_state(
                    store_id,
                    memory_dir=settings.memory_dir,
                    **_cleared_upload_journal(),
                )
            if (
                state is not None
                and state.last_error_code is ProtectionReasonCode.UPLOAD_STATUS_UNKNOWN
            ):
                return ProtectionOperation(
                    operation_id,
                    ProtectionOperationKind.BACKUP,
                    ProtectionOperationPhase.FAILED,
                    ProtectionState.ATTENTION_REQUIRED,
                    ProtectionReasonCode.UPLOAD_STATUS_UNKNOWN,
                    "A previous upload has an unknown commit status and must be reconciled.",
                )

            if not key_file_exists():
                return self._backup_failure(
                    operation_id,
                    "Cloud encryption key is missing; run `ormah cloud init`.",
                    store_id=store_id,
                    reason_code=ProtectionReasonCode.KEY_MISSING,
                )

            if store_id is None:
                return self._backup_failure(
                    operation_id,
                    "Cloud store id is missing; run `ormah cloud init`.",
                    reason_code=ProtectionReasonCode.NOT_ENABLED,
                )

            assert state is not None
            entitlement = (
                EntitlementStatus.ACTIVE
                if entitlement_verified
                else check_entitlement(settings)
            )
            if entitlement not in {EntitlementStatus.ACTIVE, EntitlementStatus.GRACE}:
                return self._backup_failure(
                    operation_id,
                    f"Cloud backup paused because entitlement is {entitlement.value}.",
                    store_id=store_id,
                    reason_code=ProtectionReasonCode.ENTITLEMENT_EXPIRED,
                    phase=ProtectionOperationPhase.CANCELED,
                    protection_state=ProtectionState.PAUSED,
                )

            backup_service = service_from_settings(settings)
            if not backup_service.has_backupable_memory():
                logger.debug("Skipping Ormah Cloud backup; no memory nodes exist yet")
                return ProtectionOperation(
                    operation_id,
                    ProtectionOperationKind.BACKUP,
                    ProtectionOperationPhase.CANCELED,
                    state.protection_state
                    if isinstance(state.protection_state, ProtectionState)
                    else ProtectionState.ATTENTION_REQUIRED,
                    ProtectionReasonCode.NO_BACKUPABLE_MEMORY,
                    message="No memory nodes are available to back up.",
                )

            now = _utc_now()
            if only_if_due and not _upload_due(state, settings.cloud_backup_interval_hours, now):
                logger.debug("Skipping Ormah Cloud backup; latest upload is still fresh")
                return ProtectionOperation(
                    operation_id,
                    ProtectionOperationKind.BACKUP,
                    ProtectionOperationPhase.CANCELED,
                    state.protection_state
                    if isinstance(state.protection_state, ProtectionState)
                    else ProtectionState.ATTENTION_REQUIRED,
                    ProtectionReasonCode.NOT_DUE,
                    message="The latest cloud backup is still fresh.",
                    snapshot_id=state.last_upload_snapshot_id,
                )

            operation_kind = (
                ProtectionOperationKind.ENABLE
                if protection_intent_id is not None
                else ProtectionOperationKind.BACKUP
            )
            update_state(
                store_id,
                memory_dir=settings.memory_dir,
                last_operation_id=operation_id,
                last_operation_kind=operation_kind,
                last_operation_phase=ProtectionOperationPhase.PREPARING,
            )
            backup = _backup_for_upload(backup_service, reason=reason)
            with tempfile.TemporaryDirectory(prefix="ormah-cloud-upload-") as tmp:
                bundle = Path(tmp) / f"{backup.name}.age"
                update_state(
                    store_id,
                    memory_dir=settings.memory_dir,
                    last_operation_id=operation_id,
                    last_operation_kind=operation_kind,
                    last_operation_phase=ProtectionOperationPhase.ENCRYPTING,
                )
                build_bundle(
                    backup.path,
                    bundle,
                    [current_recipient()],
                    store_id=store_id,
                    reason=reason,
                )
                size = bundle.stat().st_size
                digest = sha256_file(bundle)
                client = client_from_settings(settings)
                upload = client.create_upload(store_id, size, digest)
                upload_id = upload.get("upload_id")
                snapshot_id = _validated_snapshot_id(upload.get("snapshot_id"))
                put_url = upload.get("put_url")
                expires_at = upload.get("expires_at")
                required_headers = upload.get("required_headers", {})
                if not all(
                    isinstance(value, str) and value for value in (upload_id, put_url)
                ):
                    raise RuntimeError("Cloud upload reservation response was malformed.")
                if not isinstance(expires_at, str):
                    raise RuntimeError("Cloud upload reservation did not include an expiry.")
                parsed_expiry = datetime.fromisoformat(expires_at.replace("Z", "+00:00"))
                if parsed_expiry.tzinfo is None or parsed_expiry <= _utc_now():
                    raise RuntimeError("Cloud upload reservation is already expired.")
                if not isinstance(required_headers, dict) or not all(
                    isinstance(name, str) and isinstance(value, str)
                    for name, value in required_headers.items()
                ):
                    raise RuntimeError("Cloud upload reservation headers were malformed.")

                update_state(
                    store_id,
                    memory_dir=settings.memory_dir,
                    last_operation_id=operation_id,
                    last_operation_kind=operation_kind,
                    last_operation_phase=ProtectionOperationPhase.UPLOADING,
                    pending_upload_id=upload_id,
                    pending_upload_snapshot_id=snapshot_id,
                    pending_upload_operation_id=operation_id,
                    pending_upload_protection_intent_id=protection_intent_id,
                    pending_upload_phase=UploadJournalPhase.RESERVED,
                    pending_upload_expires_at=parsed_expiry,
                )
                put_file(put_url, bundle, required_headers)
                update_state(
                    store_id,
                    memory_dir=settings.memory_dir,
                    last_operation_id=operation_id,
                    last_operation_kind=operation_kind,
                    last_operation_phase=ProtectionOperationPhase.FINALIZING,
                    pending_upload_phase=UploadJournalPhase.FINALIZING,
                )
                finalize_attempted = True
                finalized = client.finalize_upload(store_id, upload_id)
                finalized_snapshot = _validated_snapshot_id(finalized.get("snapshot_id"))
                if finalized.get("status") != "committed" or finalized_snapshot != snapshot_id:
                    raise RuntimeError("Cloud upload finalize response was malformed.")

            return self._record_upload_success(
                store_id=store_id,
                operation_id=operation_id,
                snapshot_id=snapshot_id,
                protection_intent_id=protection_intent_id,
            )
        except CloudStateVersionError:
            return self._client_update_required_operation(
                operation_id,
                ProtectionOperationKind.BACKUP,
            )
        except Exception as exc:
            if not finalize_attempted and store_id is not None:
                try:
                    current = _load_writable_state(store_id)
                    if (
                        current.pending_upload_operation_id == operation_id
                        and current.pending_upload_phase is UploadJournalPhase.RESERVED
                    ):
                        update_state(
                            store_id,
                            memory_dir=settings.memory_dir,
                            **_cleared_upload_journal(),
                        )
                except Exception as cleanup_error:
                    logger.warning(
                        "Could not clear an abandoned upload reservation: %s",
                        self._message(cleanup_error),
                    )
            if finalize_attempted:
                return self._backup_failure(
                    operation_id,
                    exc,
                    store_id=store_id,
                    reason_code=ProtectionReasonCode.UPLOAD_STATUS_UNKNOWN,
                    protection_state=ProtectionState.ATTENTION_REQUIRED,
                )
            if _is_offline_error(exc):
                return self._backup_failure(
                    operation_id,
                    exc,
                    store_id=store_id,
                    reason_code=ProtectionReasonCode.OFFLINE,
                    protection_state=ProtectionState.OFFLINE,
                )
            if _is_quota_error(exc):
                return self._backup_failure(
                    operation_id,
                    exc,
                    store_id=store_id,
                    reason_code=ProtectionReasonCode.QUOTA_EXCEEDED,
                )
            if _is_disk_full_error(exc):
                return self._backup_failure(
                    operation_id,
                    exc,
                    store_id=store_id,
                    reason_code=ProtectionReasonCode.DISK_SPACE_INSUFFICIENT,
                )
            return self._backup_failure(operation_id, exc, store_id=store_id)
        finally:
            self._close_client(client)

    def _backup_failure(
        self,
        operation_id: str,
        error: object,
        *,
        store_id: str | None = None,
        reason_code: ProtectionReasonCode = ProtectionReasonCode.UPLOAD_FAILED,
        phase: ProtectionOperationPhase = ProtectionOperationPhase.FAILED,
        protection_state: ProtectionState = ProtectionState.ATTENTION_REQUIRED,
    ) -> ProtectionOperation:
        message = self._message(error)
        logger.warning("Ormah Cloud backup skipped or failed: %s", message)
        if store_id is not None:
            try:
                current = _load_writable_state(store_id)
                if not isinstance(current.protection_state, ProtectionState):
                    return self._client_update_required_operation(
                        operation_id,
                        ProtectionOperationKind.BACKUP,
                    )
                persisted_state = _state_after_failure(current, protection_state)
                update_state(
                    store_id,
                    memory_dir=self.settings.memory_dir,
                    last_upload_error=message,
                    protection_state=persisted_state,
                    last_operation_id=operation_id,
                    last_operation_kind=ProtectionOperationKind.BACKUP,
                    last_operation_phase=phase,
                    last_error_code=reason_code,
                    last_error_message=message,
                    last_error_at=_utc_now(),
                )
                protection_state = persisted_state
            except CloudStateVersionError:
                return self._client_update_required_operation(
                    operation_id,
                    ProtectionOperationKind.BACKUP,
                )
            except Exception as exc:
                logger.warning("Could not persist Ormah Cloud state: %s", self._message(exc))
        return ProtectionOperation(
            operation_id,
            ProtectionOperationKind.BACKUP,
            phase,
            protection_state,
            reason_code,
            message,
        )

    def prepare_restore(self) -> ProtectionOperation:
        """Find and fully verify the newest locally restorable cloud snapshot."""

        operation_id = str(uuid.uuid4())
        store_id = _existing_store_id(self.settings.memory_dir)
        if store_id is None:
            return ProtectionOperation(
                operation_id,
                ProtectionOperationKind.RESTORE,
                ProtectionOperationPhase.FAILED,
                ProtectionState.ATTENTION_REQUIRED,
                ProtectionReasonCode.KEY_MISSING,
                "Import your recovery kit before restoring memory on this device.",
            )

        def report(phase: str) -> None:
            update_state(
                store_id,
                memory_dir=self.settings.memory_dir,
                last_operation_id=operation_id,
                last_operation_kind=ProtectionOperationKind.RESTORE,
                last_operation_phase=ProtectionOperationPhase(phase),
            )

        try:
            with StoreLock(self.settings.memory_dir):
                state = _load_writable_state(store_id)
                prepared = prepare_cloud_restore(self.settings, progress=report)
                report("ready")
                return ProtectionOperation(
                    operation_id,
                    ProtectionOperationKind.RESTORE,
                    ProtectionOperationPhase.READY,
                    _known_state(state),
                    snapshot_id=prepared.snapshot_id,
                    verified_node_count=prepared.verified_node_count,
                    snapshot_created_at=prepared.snapshot_created_at,
                    skipped_newer_snapshots=prepared.skipped_newer_snapshots,
                    prepared_backup_name=prepared.backup_name,
                )
        except StoreLockTimeout:
            return self._store_busy_operation(
                operation_id,
                ProtectionOperationKind.RESTORE,
            )
        except Exception as exc:
            reason = (
                ProtectionReasonCode(exc.reason_code)
                if isinstance(exc, CloudRestoreError)
                and exc.reason_code in ProtectionReasonCode._value2member_map_
                else ProtectionReasonCode.RESTORE_FAILED
            )
            message = safe_product_error_message(
                exc,
                getattr(self.settings, "account_token", None),
            )
            try:
                state = _load_writable_state(store_id)
                update_state(
                    store_id,
                    memory_dir=self.settings.memory_dir,
                    last_operation_id=operation_id,
                    last_operation_kind=ProtectionOperationKind.RESTORE,
                    last_operation_phase=ProtectionOperationPhase.FAILED,
                    last_error_code=reason,
                    last_error_message=message,
                    last_error_at=_utc_now(),
                )
                protection_state = _known_state(state)
            except Exception:
                protection_state = ProtectionState.ATTENTION_REQUIRED
            return ProtectionOperation(
                operation_id,
                ProtectionOperationKind.RESTORE,
                ProtectionOperationPhase.FAILED,
                protection_state,
                reason,
                message,
            )

    def restore_prepared(self, prepared: ProtectionOperation) -> ProtectionOperation:
        """Apply one coordinator-held preparation through BackupService.restore()."""

        operation_id = str(uuid.uuid4())
        store_id = _existing_store_id(self.settings.memory_dir)
        if (
            store_id is None
            or prepared.kind is not ProtectionOperationKind.RESTORE
            or prepared.phase is not ProtectionOperationPhase.READY
            or not prepared.prepared_backup_name
            or not prepared.snapshot_id
        ):
            return ProtectionOperation(
                operation_id,
                ProtectionOperationKind.RESTORE,
                ProtectionOperationPhase.FAILED,
                ProtectionState.ATTENTION_REQUIRED,
                ProtectionReasonCode.RESTORE_FAILED,
                "The prepared recovery point is no longer available. Check again.",
            )

        def report(phase: str) -> None:
            update_state(
                store_id,
                memory_dir=self.settings.memory_dir,
                last_operation_id=operation_id,
                last_operation_kind=ProtectionOperationKind.RESTORE,
                last_operation_phase=ProtectionOperationPhase(phase),
            )

        try:
            with StoreLock(self.settings.memory_dir):
                state = _load_writable_state(store_id)
                result = restore_prepared_cloud_snapshot(
                    self.settings,
                    PreparedCloudRestore(
                        snapshot_id=prepared.snapshot_id,
                        backup_name=prepared.prepared_backup_name,
                        verified_node_count=prepared.verified_node_count or 0,
                        snapshot_created_at=prepared.snapshot_created_at,
                        skipped_newer_snapshots=prepared.skipped_newer_snapshots,
                    ),
                    rebuild_index=True,
                    engine=self.engine,
                    progress=report,
                )
                update_state(
                    store_id,
                    memory_dir=self.settings.memory_dir,
                    last_operation_id=operation_id,
                    last_operation_kind=ProtectionOperationKind.RESTORE,
                    last_operation_phase=ProtectionOperationPhase.COMPLETED,
                    last_error_code=None,
                    last_error_message=None,
                    last_error_at=None,
                )
                return ProtectionOperation(
                    operation_id,
                    ProtectionOperationKind.RESTORE,
                    ProtectionOperationPhase.COMPLETED,
                    _known_state(state),
                    snapshot_id=result.snapshot_id,
                    verified_node_count=result.restore.rebuilt_nodes,
                    snapshot_created_at=prepared.snapshot_created_at,
                    skipped_newer_snapshots=result.skipped_newer_snapshots,
                    safety_backup_name=(
                        result.restore.safety_backup.name
                        if result.restore.safety_backup is not None
                        else None
                    ),
                )
        except StoreLockTimeout:
            return self._store_busy_operation(
                operation_id,
                ProtectionOperationKind.RESTORE,
                snapshot_id=prepared.snapshot_id,
            )
        except Exception as exc:
            message = safe_product_error_message(
                exc,
                getattr(self.settings, "account_token", None),
            )
            try:
                state = _load_writable_state(store_id)
                update_state(
                    store_id,
                    memory_dir=self.settings.memory_dir,
                    last_operation_id=operation_id,
                    last_operation_kind=ProtectionOperationKind.RESTORE,
                    last_operation_phase=ProtectionOperationPhase.FAILED,
                    last_error_code=ProtectionReasonCode.RESTORE_FAILED,
                    last_error_message=message,
                    last_error_at=_utc_now(),
                )
                protection_state = _known_state(state)
            except Exception:
                protection_state = ProtectionState.ATTENTION_REQUIRED
            return ProtectionOperation(
                operation_id,
                ProtectionOperationKind.RESTORE,
                ProtectionOperationPhase.FAILED,
                protection_state,
                ProtectionReasonCode.RESTORE_FAILED,
                message,
                prepared.snapshot_id,
                verified_node_count=prepared.verified_node_count,
                snapshot_created_at=prepared.snapshot_created_at,
                skipped_newer_snapshots=prepared.skipped_newer_snapshots,
            )

    def verify_now(self, snapshot_id: str | None = None) -> ProtectionOperation:
        """Download and prove one committed snapshot entirely in scratch space."""

        operation_id = str(uuid.uuid4())
        try:
            with StoreLock(self.settings.memory_dir):
                return self._verify_now(operation_id, requested_snapshot_id=snapshot_id)
        except StoreLockTimeout:
            return self._store_busy_operation(
                operation_id,
                ProtectionOperationKind.VERIFY,
                snapshot_id=snapshot_id,
            )
        except Exception as exc:
            return self._verification_failure(operation_id, exc, snapshot_id=snapshot_id)

    def _verify_now(
        self, operation_id: str, *, requested_snapshot_id: str | None
    ) -> ProtectionOperation:
        settings = self.settings
        store_id: str | None = None
        snapshot_id: str | None = requested_snapshot_id
        tracks_latest_snapshot = requested_snapshot_id is None
        client = None
        tmp_root: Path | None = None
        try:
            store_id = _existing_store_id(settings.memory_dir)
            if store_id is None:
                raise _VerificationPrerequisiteError(
                    "Cloud store id is missing; cannot verify restore.",
                    ProtectionReasonCode.NOT_ENABLED,
                )
            try:
                state = _load_writable_state(store_id)
            except CloudStateVersionError:
                return self._client_update_required_operation(
                    operation_id,
                    ProtectionOperationKind.VERIFY,
                    snapshot_id=requested_snapshot_id,
                )
            if not isinstance(state.protection_state, ProtectionState):
                return self._client_update_required_operation(
                    operation_id,
                    ProtectionOperationKind.VERIFY,
                    snapshot_id=requested_snapshot_id,
                )
            if not key_file_exists():
                raise _VerificationPrerequisiteError(
                    "Cloud encryption key is missing; cannot verify restore.",
                    ProtectionReasonCode.KEY_MISSING,
                )
            if not settings.account_token:
                raise _VerificationPrerequisiteError(
                    "Ormah Cloud login is required to verify restore.",
                    ProtectionReasonCode.SIGN_IN_REQUIRED,
                )

            update_state(
                store_id,
                memory_dir=settings.memory_dir,
                last_operation_id=operation_id,
                last_operation_kind=ProtectionOperationKind.VERIFY,
                last_operation_phase=ProtectionOperationPhase.DOWNLOADING,
            )
            client = client_from_settings(settings)
            listing = client.list_blobs(store_id)
            blobs = listing.get("blobs")
            if not isinstance(blobs, list) or not blobs:
                raise RuntimeError("No committed cloud snapshot is available to verify.")
            latest_snapshot_id = _validated_snapshot_id(
                blobs[0].get("snapshot_id") if isinstance(blobs[0], dict) else None
            )
            selected = (
                blobs[0]
                if requested_snapshot_id is None
                else next(
                    (
                        blob
                        for blob in blobs
                        if isinstance(blob, dict)
                        and blob.get("snapshot_id") == requested_snapshot_id
                    ),
                    None,
                )
            )
            if not isinstance(selected, dict):
                raise RuntimeError("The requested cloud snapshot is not available to verify.")
            snapshot_id = _validated_snapshot_id(selected.get("snapshot_id"))
            expected_latest_snapshot_id = (
                state.last_successful_backup_snapshot_id or latest_snapshot_id
            )
            tracks_latest_snapshot = (
                requested_snapshot_id is None or snapshot_id == expected_latest_snapshot_id
            )
            size_bytes = selected.get("size_bytes")
            if not isinstance(size_bytes, int) or isinstance(size_bytes, bool) or size_bytes <= 0:
                raise RuntimeError("Cloud snapshot listing did not include a valid size.")
            if size_bytes > client.processing_limit(require_hardened_write=False):
                raise _VerificationStageError(
                    "Cloud snapshot exceeds this client's safe processing limit.",
                    ProtectionReasonCode.PROCESSING_LIMIT_EXCEEDED,
                )
            if "sha256" not in selected:
                raise _VerificationStageError(
                    "Ormah Cloud must be updated before restore verification can continue.",
                    ProtectionReasonCode.SERVICE_UPDATE_REQUIRED,
                )
            expected_sha256 = _validated_sha256(selected["sha256"])

            presigned = client.presign_download(store_id, snapshot_id)
            get_url = presigned.get("get_url")
            if not isinstance(get_url, str) or not get_url:
                raise RuntimeError("Cloud download response was malformed.")

            tmp_root = Path(tempfile.mkdtemp(prefix="ormah-restore-verify-"))
            bundle = tmp_root / f"{snapshot_id}.age"
            extracted = tmp_root / "snapshot"
            try:
                download_file(get_url, bundle)
            except httpx.RequestError:
                raise
            except OSError as exc:
                if _is_disk_full_error(exc):
                    raise
                raise _VerificationStageError(
                    "The encrypted snapshot could not be downloaded.",
                    ProtectionReasonCode.DOWNLOAD_FAILED,
                ) from exc
            except Exception as exc:
                raise _VerificationStageError(
                    "The encrypted snapshot could not be downloaded.",
                    ProtectionReasonCode.DOWNLOAD_FAILED,
                ) from exc
            if sha256_file(bundle) != expected_sha256:
                raise _VerificationStageError(
                    "The downloaded ciphertext hash did not match cloud metadata.",
                    ProtectionReasonCode.CIPHERTEXT_HASH_MISMATCH,
                )
            update_state(
                store_id,
                memory_dir=settings.memory_dir,
                last_operation_id=operation_id,
                last_operation_kind=ProtectionOperationKind.VERIFY,
                last_operation_phase=ProtectionOperationPhase.VERIFYING,
            )
            try:
                info = open_bundle(bundle, extracted, load_identities())
            except CloudCryptoError as exc:
                raise _VerificationStageError(
                    "The encrypted snapshot could not be decrypted with this recovery key.",
                    ProtectionReasonCode.DECRYPT_FAILED,
                ) from exc
            except BundleError as exc:
                raise _VerificationStageError(
                    "The decrypted snapshot failed manifest and file-integrity verification.",
                    ProtectionReasonCode.MANIFEST_VERIFICATION_FAILED,
                ) from exc
            except (KeyError, TypeError, ValueError, UnicodeError) as exc:
                raise _VerificationStageError(
                    "The decrypted snapshot failed manifest and file-integrity verification.",
                    ProtectionReasonCode.MANIFEST_VERIFICATION_FAILED,
                ) from exc
            update_state(
                store_id,
                memory_dir=settings.memory_dir,
                last_operation_id=operation_id,
                last_operation_kind=ProtectionOperationKind.VERIFY,
                last_operation_phase=ProtectionOperationPhase.REBUILDING,
            )
            verified_node_count = _verify_extracted_bundle(extracted, store_id, info)

            verified_at = _utc_now()
            next_state = _state_after_verification(state, snapshot_id)
            latest_state = _load_writable_state(store_id)
            preserve_upload_ambiguity = self._upload_status_is_unknown(latest_state)
            changes: dict[str, object] = {
                "last_operation_id": operation_id,
                "last_operation_kind": ProtectionOperationKind.VERIFY,
                "last_operation_phase": ProtectionOperationPhase.COMPLETED,
            }
            if not preserve_upload_ambiguity:
                changes.update(
                    last_error_code=None,
                    last_error_message=None,
                    last_error_at=None,
                )
            if tracks_latest_snapshot:
                changes.update(
                    last_verify_at=verified_at,
                    last_verify_ok=True,
                    last_verify_snapshot_id=snapshot_id,
                    last_verify_error=None,
                    last_successful_verify_at=verified_at,
                    last_verified_snapshot_id=snapshot_id,
                    protection_state=next_state,
                )
            update_state(store_id, memory_dir=settings.memory_dir, **changes)
            logger.info(
                "Verified Ormah Cloud snapshot %s is restorable", self._message(snapshot_id)
            )
            return ProtectionOperation(
                operation_id,
                ProtectionOperationKind.VERIFY,
                ProtectionOperationPhase.COMPLETED,
                next_state,
                snapshot_id=snapshot_id,
                verified_node_count=verified_node_count,
            )
        except Exception as exc:
            offline = _is_offline_error(exc)
            reason_code = (
                exc.reason_code
                if isinstance(exc, (_VerificationPrerequisiteError, _VerificationStageError))
                else ProtectionReasonCode.OFFLINE
                if offline
                else ProtectionReasonCode.DISK_SPACE_INSUFFICIENT
                if _is_disk_full_error(exc)
                else ProtectionReasonCode.BUNDLE_CORRUPT
                if isinstance(exc, BundleError)
                else ProtectionReasonCode.VERIFICATION_FAILED
            )
            return self._verification_failure(
                operation_id,
                exc,
                store_id=store_id,
                snapshot_id=snapshot_id,
                reason_code=reason_code,
                record_health=tracks_latest_snapshot and not offline,
                protection_state=(
                    ProtectionState.OFFLINE
                    if offline
                    else ProtectionState.ATTENTION_REQUIRED
                    if tracks_latest_snapshot
                    else None
                ),
            )
        finally:
            self._close_client(client)
            if tmp_root is not None:
                shutil.rmtree(tmp_root, ignore_errors=True)

    def _verification_failure(
        self,
        operation_id: str,
        error: object,
        *,
        store_id: str | None = None,
        snapshot_id: str | None = None,
        reason_code: ProtectionReasonCode = ProtectionReasonCode.VERIFICATION_FAILED,
        record_health: bool = True,
        protection_state: ProtectionState | None = ProtectionState.ATTENTION_REQUIRED,
    ) -> ProtectionOperation:
        message = self._message(error)
        logger.warning("Ormah Cloud restore verification failed: %s", message)
        if store_id is None:
            try:
                store_id = _existing_store_id(self.settings.memory_dir)
            except Exception:
                store_id = None
        if store_id is not None:
            try:
                current = _load_writable_state(store_id)
                if not isinstance(current.protection_state, ProtectionState):
                    return self._client_update_required_operation(
                        operation_id,
                        ProtectionOperationKind.VERIFY,
                        snapshot_id=snapshot_id,
                    )
                persisted_state = (
                    _state_after_failure(current, protection_state)
                    if protection_state is not None
                    else _known_state(current)
                )
                failed_at = _utc_now()
                changes: dict[str, object] = {
                    "last_operation_id": operation_id,
                    "last_operation_kind": ProtectionOperationKind.VERIFY,
                    "last_operation_phase": ProtectionOperationPhase.FAILED,
                    "last_error_code": reason_code,
                    "last_error_message": message,
                    "last_error_at": failed_at,
                }
                if record_health:
                    changes.update(
                        last_verify_at=failed_at,
                        last_verify_ok=False,
                        last_verify_snapshot_id=snapshot_id,
                        last_verify_error=message,
                    )
                if protection_state is not None:
                    changes["protection_state"] = persisted_state
                update_state(store_id, memory_dir=self.settings.memory_dir, **changes)
                failure_state = persisted_state
            except CloudStateVersionError:
                return self._client_update_required_operation(
                    operation_id,
                    ProtectionOperationKind.VERIFY,
                    snapshot_id=snapshot_id,
                )
            except Exception as exc:
                logger.warning("Could not persist Ormah Cloud state: %s", self._message(exc))
                failure_state = ProtectionState.ATTENTION_REQUIRED
        else:
            failure_state = ProtectionState.ATTENTION_REQUIRED
        return ProtectionOperation(
            operation_id,
            ProtectionOperationKind.VERIFY,
            ProtectionOperationPhase.FAILED,
            failure_state,
            reason_code,
            message,
            snapshot_id,
        )

    def _client_update_required_operation(
        self,
        operation_id: str,
        kind: ProtectionOperationKind,
        *,
        snapshot_id: str | None = None,
    ) -> ProtectionOperation:
        """Fail closed when durable state belongs to a newer client."""

        return ProtectionOperation(
            operation_id,
            kind,
            ProtectionOperationPhase.CANCELED,
            ProtectionState.ATTENTION_REQUIRED,
            ProtectionReasonCode.CLIENT_UPDATE_REQUIRED,
            "This cloud state requires a newer Ormah version.",
            snapshot_id,
        )

    def _store_busy_operation(
        self,
        operation_id: str,
        kind: ProtectionOperationKind,
        *,
        snapshot_id: str | None = None,
        protection_intent_id: str | None = None,
    ) -> ProtectionOperation:
        """Return transient contention without changing durable protection health."""

        current_state = ProtectionState.ATTENTION_REQUIRED
        try:
            store_id = _existing_store_id(self.settings.memory_dir)
            if store_id is None:
                current_state = ProtectionState.LOCAL_ONLY
            else:
                current_state = _known_state(load_state(store_id))
        except Exception:
            pass
        logger.info("Ormah Cloud operation canceled because the memory store is busy")
        return ProtectionOperation(
            operation_id,
            kind,
            ProtectionOperationPhase.CANCELED,
            current_state,
            ProtectionReasonCode.STORE_BUSY,
            "Memory store is busy; try again shortly.",
            snapshot_id,
            protection_intent_id,
        )

    @staticmethod
    def _close_client(client) -> None:
        close = getattr(client, "close", None) if client is not None else None
        if close is not None:
            try:
                close()
            except Exception:
                pass


class _VerificationPrerequisiteError(RuntimeError):
    def __init__(self, message: str, reason_code: ProtectionReasonCode) -> None:
        super().__init__(message)
        self.reason_code = reason_code


class _VerificationStageError(RuntimeError):
    def __init__(self, message: str, reason_code: ProtectionReasonCode) -> None:
        super().__init__(message)
        self.reason_code = reason_code


class _EnablePrerequisiteError(RuntimeError):
    def __init__(self, message: str, reason_code: ProtectionReasonCode) -> None:
        super().__init__(message)
        self.reason_code = reason_code

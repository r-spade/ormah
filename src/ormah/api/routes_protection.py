"""Authenticated local API adapters for cloud protection operations."""

from __future__ import annotations

from pathlib import Path
import uuid

from fastapi import APIRouter, Depends, HTTPException, Request, status
from pydantic import BaseModel, ConfigDict, Field

from ormah.api.local_auth import require_local_admin
from ormah.cloud.billing import validate_protection_intent_id
from ormah.cloud.entitlements import load_entitlement_cache, status_from_cache
from ormah.cloud.operations import ProtectionOperationCoordinator
from ormah.cloud.protection import CloudProtectionService, safe_product_error_message
from ormah.cloud.recovery import RecoveryKitError, RecoveryKitService
from ormah.cloud.restore import CloudRestoreError, newest_cloud_snapshot
from ormah.cloud.state import (
    CloudStateError,
    ProtectionOperation,
    ProtectionOperationKind,
    ProtectionReasonCode,
    cloud_status_payload,
)
from ormah.cloud.store_lock import StoreLockTimeout

router = APIRouter(
    prefix="/admin/cloud/protection",
    tags=["cloud-protection"],
    dependencies=[Depends(require_local_admin)],
)


class EmptyRequest(BaseModel):
    """Explicit empty body that rejects accidental protocol inputs."""

    model_config = ConfigDict(extra="forbid")


class VerifyRequest(BaseModel):
    """Optional exact recovery point to verify."""

    model_config = ConfigDict(extra="forbid")

    snapshot_id: str | None = Field(
        default=None,
        pattern=r"^[0-7][0-9A-HJKMNP-TV-Z]{25}$",
    )


class ConfirmRecoveryKitRequest(BaseModel):
    """Proof from the trusted native save/reopen flow."""

    model_config = ConfigDict(extra="forbid")

    sha256_digest: str = Field(pattern=r"^[0-9a-f]{64}$")


class RecoveryReadinessResponse(BaseModel):
    """Purpose-bound response containing no recovery material or locations."""

    model_config = ConfigDict(extra="forbid")

    device_loss_recovery_ready: bool
    recovery_kit_verified_at: str


class RecoveryKitPrepareResponse(BaseModel):
    """Secret-free readiness result for the native save dialog."""

    model_config = ConfigDict(extra="forbid")

    status: str
    regenerated: bool


def _service(request: Request) -> CloudProtectionService:
    injected = getattr(request.app.state, "protection_service", None)
    return injected or CloudProtectionService.from_engine(request.app.state.engine)


def _coordinator(request: Request) -> ProtectionOperationCoordinator:
    coordinator = getattr(request.app.state, "protection_operations", None)
    if not isinstance(coordinator, ProtectionOperationCoordinator):
        raise HTTPException(status_code=503, detail="Protection operations are unavailable.")
    return coordinator


def _recovery_service(request: Request) -> RecoveryKitService:
    injected = getattr(request.app.state, "recovery_kit_service", None)
    return injected or RecoveryKitService(request.app.state.engine.settings)


def _store_key(request: Request, operation: str) -> tuple[str, ...]:
    memory_dir = Path(request.app.state.engine.settings.memory_dir).expanduser().resolve()
    return str(memory_dir), operation


def _intent_id(value: str) -> str:
    try:
        return validate_protection_intent_id(value)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail="Invalid protection intent ID.") from exc


def _operation_payload(operation: ProtectionOperation) -> dict[str, object]:
    return {
        "operation_id": operation.operation_id,
        "kind": operation.kind.value,
        "phase": operation.phase.value,
        "protection_state": operation.state.value,
        "reason_code": operation.reason_code.value if operation.reason_code else None,
        "message": operation.message,
        "snapshot_id": operation.snapshot_id,
        "protection_intent_id": operation.protection_intent_id,
        "verified_node_count": operation.verified_node_count,
        "snapshot_created_at": operation.snapshot_created_at,
        "skipped_newer_snapshots": operation.skipped_newer_snapshots,
        "safety_backup_name": operation.safety_backup_name,
    }


def _cached_entitlement(settings) -> str:
    """Classify local entitlement state without network access during polling."""
    if not getattr(settings, "account_token", None):
        return "none"
    cache = load_entitlement_cache()
    return status_from_cache(cache).value if cache is not None else "none"


def _submit(
    request: Request,
    *,
    operation: str,
    kind: ProtectionOperationKind,
    action,
) -> dict[str, object]:
    try:
        record, deduplicated = _coordinator(request).submit(
            key=_store_key(request, operation),
            kind=kind,
            action=action,
        )
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail="Protection operations are unavailable.") from exc
    payload = record.to_payload()
    payload["deduplicated"] = deduplicated
    return payload


@router.get("")
def protection_status(request: Request):
    """Return durable, token-free protection state for this memory."""
    settings = request.app.state.engine.settings
    payload = cloud_status_payload(settings, entitlement=_cached_entitlement(settings))
    # The shared payload also serves the local CLI, where detailed state paths
    # are useful. The product webview receives only redacted diagnostics.
    payload.pop("state_error", None)
    for field in (
        "last_upload_error",
        "last_verify_error",
        "last_error_message",
    ):
        if payload.get(field) is not None:
            payload[field] = safe_product_error_message(
                payload[field], getattr(settings, "account_token", None)
            )
    payload["warnings"] = [
        safe_product_error_message(warning, getattr(settings, "account_token", None))
        for warning in payload.get("warnings", [])
    ]
    return payload


@router.get("/remote")
def remote_snapshot(request: Request):
    """Describe the newest cloud backup, including one uploaded by another device.

    Local state records only this device's own uploads and verifications, so a
    second machine's backup cannot be seen without asking the cloud. Failures
    degrade to a redacted reason instead of an error status: not knowing what
    the cloud holds must never take the protection panel down.
    """

    settings = request.app.state.engine.settings
    unavailable = {
        "snapshot_id": None,
        "created_at": None,
        "size_bytes": None,
        "from_this_device": False,
        "restore_tested_here": False,
        "reason_code": None,
        "error": None,
    }
    try:
        newest = newest_cloud_snapshot(settings)
    except CloudRestoreError as exc:
        return {
            **unavailable,
            "reason_code": exc.reason_code,
            "error": safe_product_error_message(
                str(exc), getattr(settings, "account_token", None)
            ),
        }
    except Exception as exc:
        return {
            **unavailable,
            "reason_code": "remote_listing_failed",
            "error": safe_product_error_message(
                f"Could not read cloud backups: {exc}",
                getattr(settings, "account_token", None),
            ),
        }
    if newest is None:
        return unavailable

    state = cloud_status_payload(settings, entitlement=_cached_entitlement(settings))
    snapshot_id = newest["snapshot_id"]
    return {
        **newest,
        "from_this_device": snapshot_id == state.get("last_successful_backup_snapshot_id"),
        # A device can only vouch for a check it ran itself.
        "restore_tested_here": snapshot_id == state.get("last_verified_snapshot_id"),
        "reason_code": None,
        "error": None,
    }


@router.post("/intents")
def create_intent(body: EmptyRequest, request: Request):
    """Record explicit upload consent and create or resume a protection intent."""
    del body
    return _operation_payload(_service(request).create_intent())


@router.post("/intents/{intent_id}/bind")
def bind_intent(intent_id: str, body: EmptyRequest, request: Request):
    """Bind an existing protection intent to the currently authenticated account."""
    del body
    return _operation_payload(_service(request).bind_intent(_intent_id(intent_id)))


@router.post("/intents/{intent_id}/cancel")
def cancel_intent(intent_id: str, body: EmptyRequest, request: Request):
    """Cancel a protection intent that has not crossed the upload boundary."""
    del body
    return _operation_payload(_service(request).cancel_intent(_intent_id(intent_id)))


@router.post("/intents/{intent_id}/enable", status_code=status.HTTP_202_ACCEPTED)
def enable_protection(intent_id: str, body: EmptyRequest, request: Request):
    """Start immediate upload and verification without holding the request open."""
    del body
    intent_id = _intent_id(intent_id)
    service = _service(request)
    return _submit(
        request,
        operation=f"enable:{intent_id}",
        kind=ProtectionOperationKind.ENABLE,
        action=lambda: service.enable(intent_id),
    )


@router.post("/disable", status_code=status.HTTP_202_ACCEPTED)
def disable_protection(body: EmptyRequest, request: Request):
    """Stop future uploads while preserving retained recovery points."""
    del body
    service = _service(request)
    return _submit(
        request,
        operation="disable",
        kind=ProtectionOperationKind.DISABLE,
        action=service.disable,
    )


@router.post("/backup", status_code=status.HTTP_202_ACCEPTED)
def backup_now(body: EmptyRequest, request: Request):
    """Create a recovery point and restore-test that exact snapshot."""
    del body
    service = _service(request)
    return _submit(
        request,
        operation="backup",
        kind=ProtectionOperationKind.BACKUP,
        action=lambda: service.backup_and_verify(reason="manual-ui"),
    )


@router.post("/verify", status_code=status.HTTP_202_ACCEPTED)
def verify_now(body: VerifyRequest, request: Request):
    """Start a full scratch restore verification for one recovery point."""
    service = _service(request)
    suffix = body.snapshot_id or "latest"
    return _submit(
        request,
        operation=f"verify:{suffix}",
        kind=ProtectionOperationKind.VERIFY,
        action=lambda: service.verify_now(body.snapshot_id),
    )


@router.post("/restore/prepare", status_code=status.HTTP_202_ACCEPTED)
def prepare_restore(body: EmptyRequest, request: Request):
    """Find and fully verify the newest locally restorable recovery point."""

    del body
    service = _service(request)
    return _submit(
        request,
        operation="restore:prepare",
        kind=ProtectionOperationKind.RESTORE,
        action=service.prepare_restore,
    )


@router.post(
    "/restore/{preparation_operation_id}/confirm",
    status_code=status.HTTP_202_ACCEPTED,
)
def confirm_restore(
    preparation_operation_id: str,
    body: EmptyRequest,
    request: Request,
):
    """Consume one verified preparation and replace live memory once."""

    del body
    try:
        parsed_operation_id = uuid.UUID(preparation_operation_id)
    except (ValueError, AttributeError) as exc:
        raise HTTPException(status_code=422, detail="Invalid restore preparation ID.") from exc
    if parsed_operation_id.version != 4 or str(parsed_operation_id) != preparation_operation_id:
        raise HTTPException(status_code=422, detail="Invalid restore preparation ID.")
    coordinator = _coordinator(request)
    prepared = coordinator.claim_ready_result(
        preparation_operation_id,
        kind=ProtectionOperationKind.RESTORE,
    )
    if prepared is None:
        raise HTTPException(
            status_code=409,
            detail="This restore preparation is unavailable. Check the backup again.",
        )
    service = _service(request)

    def apply_prepared():
        result = service.restore_prepared(prepared)
        if result.reason_code is ProtectionReasonCode.STORE_BUSY:
            coordinator.release_ready_claim(preparation_operation_id)
        return result

    try:
        return _submit(
            request,
            operation=f"restore:confirm:{preparation_operation_id}",
            kind=ProtectionOperationKind.RESTORE,
            action=apply_prepared,
        )
    except Exception:
        coordinator.release_ready_claim(preparation_operation_id)
        raise


@router.post("/restore/{preparation_operation_id}/cancel")
def cancel_restore(
    preparation_operation_id: str,
    body: EmptyRequest,
    request: Request,
):
    """Consume and delete one unneeded verified preparation."""

    del body
    try:
        parsed_operation_id = uuid.UUID(preparation_operation_id)
    except (ValueError, AttributeError) as exc:
        raise HTTPException(status_code=422, detail="Invalid restore preparation ID.") from exc
    if parsed_operation_id.version != 4 or str(parsed_operation_id) != preparation_operation_id:
        raise HTTPException(status_code=422, detail="Invalid restore preparation ID.")
    coordinator = _coordinator(request)
    prepared = coordinator.claim_ready_result(
        preparation_operation_id,
        kind=ProtectionOperationKind.RESTORE,
    )
    if prepared is None:
        raise HTTPException(
            status_code=409,
            detail="This restore preparation is no longer available.",
        )
    try:
        discarded = _service(request).discard_prepared_restore(prepared)
    except Exception:
        coordinator.release_ready_claim(preparation_operation_id)
        raise
    if not discarded:
        raise HTTPException(
            status_code=409,
            detail="This restore preparation is no longer available.",
        )
    return {"status": "discarded"}


@router.post(
    "/recovery-kit/prepare",
    response_model=RecoveryKitPrepareResponse,
)
def prepare_recovery_kit(body: EmptyRequest, request: Request):
    """Ensure the canonical kit matches current keys before native save."""

    del body
    try:
        regenerated = _recovery_service(request).ensure_current_kit()
    except (RecoveryKitError, CloudStateError, StoreLockTimeout, OSError) as exc:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="The recovery kit could not be prepared for saving.",
        ) from exc
    return {"status": "ready_to_save", "regenerated": regenerated}


@router.post(
    "/recovery-kit/confirm",
    response_model=RecoveryReadinessResponse,
)
def confirm_recovery_kit(body: ConfirmRecoveryKitRequest, request: Request):
    """Confirm only the digest produced after the native destination was reopened."""

    try:
        result = _recovery_service(request).confirm_saved_digest(body.sha256_digest)
    except (RecoveryKitError, CloudStateError, StoreLockTimeout, OSError) as exc:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="The saved recovery kit could not be verified.",
        ) from exc
    return {
        "device_loss_recovery_ready": result.device_loss_recovery_ready,
        "recovery_kit_verified_at": result.recovery_kit_verified_at.isoformat(),
    }


@router.get("/operations/{operation_id}")
def operation_status(operation_id: str, request: Request):
    """Poll one process-local invocation; durable protection status is separate."""
    record = _coordinator(request).get(operation_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Protection operation not found.")
    return record.to_payload()

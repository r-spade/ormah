"""Authenticated local API adapters for cloud protection operations."""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, Request, status
from pydantic import BaseModel, ConfigDict, Field

from ormah.api.local_auth import require_local_admin
from ormah.cloud.billing import validate_protection_intent_id
from ormah.cloud.entitlements import load_entitlement_cache, status_from_cache
from ormah.cloud.operations import ProtectionOperationCoordinator
from ormah.cloud.protection import CloudProtectionService
from ormah.cloud.state import (
    ProtectionOperation,
    ProtectionOperationKind,
    cloud_status_payload,
)

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


def _service(request: Request) -> CloudProtectionService:
    injected = getattr(request.app.state, "protection_service", None)
    return injected or CloudProtectionService.from_engine(request.app.state.engine)


def _coordinator(request: Request) -> ProtectionOperationCoordinator:
    coordinator = getattr(request.app.state, "protection_operations", None)
    if not isinstance(coordinator, ProtectionOperationCoordinator):
        raise HTTPException(status_code=503, detail="Protection operations are unavailable.")
    return coordinator


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
    return cloud_status_payload(settings, entitlement=_cached_entitlement(settings))


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
    """Start one user-requested encrypted backup."""
    del body
    service = _service(request)
    return _submit(
        request,
        operation="backup",
        kind=ProtectionOperationKind.BACKUP,
        action=lambda: service.backup_now(reason="manual-ui"),
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


@router.get("/operations/{operation_id}")
def operation_status(operation_id: str, request: Request):
    """Poll one process-local invocation; durable protection status is separate."""
    record = _coordinator(request).get(operation_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Protection operation not found.")
    return record.to_payload()

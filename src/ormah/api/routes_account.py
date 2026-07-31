"""Owner-only local account authentication and hosted billing handoffs.

Handlers are thin wrappers over ``ormah.cloud.account`` and ``ormah.cloud.billing``.
They validate local input, delegate, and translate stable application errors.
No remote request is constructed here, and no response includes an account token.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, ConfigDict, field_validator

from ormah.api.local_auth import require_local_admin
from ormah.cloud.account import (
    AccountError,
    AccountErrorCode,
    get_account_status,
    logout_account,
    normalize_email,
    request_login_code,
    validate_code,
    verify_login_code,
)
from ormah.cloud.billing import (
    BillingError,
    BillingErrorCode,
    get_offer,
    open_portal,
    start_checkout,
    validate_protection_intent_id,
)

router = APIRouter(
    prefix="/admin/account",
    tags=["account"],
    dependencies=[Depends(require_local_admin)],
)

_ERROR_STATUS: dict[BillingErrorCode, int] = {
    BillingErrorCode.SIGN_IN_REQUIRED: 401,
    BillingErrorCode.ACCOUNT_FORBIDDEN: 403,
    BillingErrorCode.INVALID_REQUEST: 400,
    BillingErrorCode.NOT_FOUND: 404,
    BillingErrorCode.CONFLICT: 409,
    BillingErrorCode.RATE_LIMITED: 429,
    BillingErrorCode.CLOUD_UNREACHABLE: 503,
    BillingErrorCode.BILLING_UNAVAILABLE: 502,
    BillingErrorCode.STATE_UNAVAILABLE: 503,
}

_ACCOUNT_ERROR_STATUS: dict[AccountErrorCode, int] = {
    AccountErrorCode.INVALID_REQUEST: 400,
    AccountErrorCode.INVALID_CODE: 401,
    AccountErrorCode.CLOUD_UNREACHABLE: 503,
    AccountErrorCode.ACCOUNT_UNAVAILABLE: 502,
    AccountErrorCode.STATE_UNAVAILABLE: 503,
}


class RequestCodeRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    email: str

    @field_validator("email")
    @classmethod
    def _email(cls, value: str) -> str:
        try:
            return normalize_email(value)
        except AccountError as exc:
            raise ValueError(str(exc)) from exc


class VerifyCodeRequest(RequestCodeRequest):
    code: str

    @field_validator("code")
    @classmethod
    def _code(cls, value: str) -> str:
        try:
            return validate_code(value)
        except AccountError as exc:
            raise ValueError(str(exc)) from exc


class CheckoutRequest(BaseModel):
    """The only accepted Checkout input: which protection intent to pay for."""

    model_config = ConfigDict(extra="forbid")

    protection_intent_id: str

    @field_validator("protection_intent_id")
    @classmethod
    def _canonical_uuid(cls, value: str) -> str:
        return validate_protection_intent_id(value)


class PortalRequest(BaseModel):
    """Portal sessions take no client input: no customer, no return URL."""

    model_config = ConfigDict(extra="forbid")


class LogoutRequest(BaseModel):
    """Logout accepts no token or device identity from the caller."""

    model_config = ConfigDict(extra="forbid")


def _settings(request: Request):
    return request.app.state.engine.settings


def _cloud_client(request: Request):
    """Injection seam for tests and the desktop shell.

    Production leaves this unset, and the billing adapter builds its own
    short-lived authenticated client from settings.
    """
    return getattr(request.app.state, "cloud_client", None)


def _cloud_state_dir(request: Request):
    """Optional test seam; production uses the canonical per-user state directory."""
    return getattr(request.app.state, "cloud_state_dir", None)


def _http_error(exc: BillingError) -> HTTPException:
    """Map a stable billing reason code onto a local HTTP error."""
    return HTTPException(
        status_code=_ERROR_STATUS.get(exc.code, 502),
        detail={"error": exc.code.value, "message": str(exc)},
    )


def _account_http_error(exc: AccountError) -> HTTPException:
    return HTTPException(
        status_code=_ACCOUNT_ERROR_STATUS.get(exc.code, 502),
        detail={"error": exc.code.value, "message": str(exc)},
    )


@router.post("/request-code", status_code=202)
def account_request_code(body: RequestCodeRequest, request: Request):
    try:
        result = request_login_code(
            _settings(request), body.email, client=_cloud_client(request)
        )
    except AccountError as exc:
        raise _account_http_error(exc) from exc
    return result.to_payload()


@router.post("/verify")
def account_verify_code(body: VerifyCodeRequest, request: Request):
    try:
        status = verify_login_code(
            _settings(request),
            body.email,
            body.code,
            client=_cloud_client(request),
        )
    except AccountError as exc:
        raise _account_http_error(exc) from exc
    return status.to_payload()


@router.get("/status")
def account_status(request: Request):
    return get_account_status(_settings(request)).to_payload()


@router.post("/logout")
def account_logout(body: LogoutRequest, request: Request):
    try:
        result = logout_account(
            _settings(request), client=_cloud_client(request)
        )
    except AccountError as exc:
        raise _account_http_error(exc) from exc
    return result.to_payload()


@router.get("/offer")
def account_offer(request: Request):
    """Return the hosted plan offer for the signed-in account.

    Carries no hosted URL, price ID, or customer ID.
    """
    try:
        offer = get_offer(_settings(request), client=_cloud_client(request))
    except BillingError as exc:
        raise _http_error(exc) from exc
    return offer.to_payload()


@router.post("/checkout")
def account_checkout(body: CheckoutRequest, request: Request):
    """Create a hosted Checkout handoff for one protection intent."""
    try:
        handoff = start_checkout(
            _settings(request),
            body.protection_intent_id,
            client=_cloud_client(request),
            state_dir=_cloud_state_dir(request),
        )
    except BillingError as exc:
        raise _http_error(exc) from exc
    return handoff.to_payload()


@router.post("/portal")
def account_portal(body: PortalRequest, request: Request):
    """Create a hosted billing-portal handoff for the signed-in account."""
    try:
        handoff = open_portal(_settings(request), client=_cloud_client(request))
    except BillingError as exc:
        raise _http_error(exc) from exc
    return handoff.to_payload()

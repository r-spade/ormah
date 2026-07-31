"""Shared local account application service for CLI and desktop adapters."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum
import re
from typing import Any

from filelock import Timeout as FileLockTimeout

from ormah.cloud.client import (
    CloudClient,
    CloudError,
    client_from_settings,
    get_device_name,
    get_or_create_device_id,
    persist_account_credentials,
    remove_account_credentials,
)
from ormah.cloud.entitlements import (
    EntitlementCache,
    EntitlementStatus,
    check_entitlement,
    clear_entitlement_cache,
    load_entitlement_cache,
    refresh_entitlements,
    status_from_cache,
)


_OTP_RE = re.compile(r"\d{6}")


class AccountErrorCode(StrEnum):
    """Stable failures shared by local account adapters."""

    INVALID_REQUEST = "invalid_request"
    INVALID_CODE = "invalid_or_expired_code"
    CLOUD_UNREACHABLE = "cloud_unreachable"
    ACCOUNT_UNAVAILABLE = "account_unavailable"
    STATE_UNAVAILABLE = "state_unavailable"


_ERROR_MESSAGES = {
    AccountErrorCode.INVALID_REQUEST: "Enter a valid email address and six-digit code.",
    AccountErrorCode.INVALID_CODE: "The sign-in code is invalid or expired.",
    AccountErrorCode.CLOUD_UNREACHABLE: "Could not reach Ormah Cloud.",
    AccountErrorCode.ACCOUNT_UNAVAILABLE: "Ormah Cloud account service is unavailable.",
    AccountErrorCode.STATE_UNAVAILABLE: "Local account settings could not be updated.",
}


class AccountError(RuntimeError):
    def __init__(self, code: AccountErrorCode) -> None:
        super().__init__(_ERROR_MESSAGES[code])
        self.code = code


@dataclass(frozen=True)
class CodeRequestResult:
    status: str = "code_sent"

    def to_payload(self) -> dict[str, str]:
        return {"status": self.status}


@dataclass(frozen=True)
class AccountStatus:
    signed_in: bool
    email: str | None
    device_name: str
    entitlement: EntitlementStatus
    plan_status: str | None
    cache_age_seconds: int | None
    entitlement_available: bool

    def to_payload(self) -> dict[str, Any]:
        return {
            "signed_in": self.signed_in,
            "email": self.email,
            "device_name": self.device_name,
            "entitlement": self.entitlement.value,
            "plan_status": self.plan_status,
            "cache_age_seconds": self.cache_age_seconds,
            "entitlement_available": self.entitlement_available,
        }


@dataclass(frozen=True)
class LogoutResult:
    revoked_remotely: bool
    warning: str | None = None

    def to_payload(self) -> dict[str, Any]:
        return {
            "signed_in": False,
            "revoked_remotely": self.revoked_remotely,
            "warning": self.warning,
        }


def normalize_email(value: str) -> str:
    email = value.strip().lower()
    if (
        not email
        or len(email) > 320
        or "@" not in email
        or any(ord(char) < 0x20 or ord(char) == 0x7F for char in email)
    ):
        raise AccountError(AccountErrorCode.INVALID_REQUEST)
    return email


def validate_code(value: str) -> str:
    code = value.strip()
    if _OTP_RE.fullmatch(code) is None:
        raise AccountError(AccountErrorCode.INVALID_REQUEST)
    return code


def _map_cloud_error(exc: CloudError, *, verifying: bool = False) -> AccountError:
    if verifying and exc.status_code == 401:
        return AccountError(AccountErrorCode.INVALID_CODE)
    if exc.status_code in {400, 422}:
        return AccountError(AccountErrorCode.INVALID_REQUEST)
    if exc.status_code is None:
        return AccountError(AccountErrorCode.CLOUD_UNREACHABLE)
    return AccountError(AccountErrorCode.ACCOUNT_UNAVAILABLE)


def _close_owned(client: CloudClient | None, owned: bool) -> None:
    if owned and client is not None:
        try:
            client.close()
        except Exception:
            pass


def request_login_code(
    settings,
    email: str,
    *,
    client: CloudClient | None = None,
) -> CodeRequestResult:
    """Request a generic email OTP without exposing account existence."""

    email = normalize_email(email)
    owned = client is None
    try:
        client = client or client_from_settings(settings)
        client.request_code(email)
    except CloudError as exc:
        raise _map_cloud_error(exc) from exc
    except (OSError, ValueError) as exc:
        raise AccountError(AccountErrorCode.ACCOUNT_UNAVAILABLE) from exc
    finally:
        _close_owned(client, owned)
    return CodeRequestResult()


def _status_from_cache(settings, cache: EntitlementCache | None) -> AccountStatus:
    signed_in = bool(
        isinstance(getattr(settings, "account_token", None), str)
        and settings.account_token.strip()
    )
    if not signed_in:
        cache = None
    entitlement = (
        status_from_cache(cache)
        if signed_in and cache is not None
        else EntitlementStatus.NONE
    )
    return AccountStatus(
        signed_in=signed_in,
        email=getattr(settings, "account_email", None) if signed_in else None,
        device_name=get_device_name(),
        entitlement=entitlement,
        plan_status=cache.plan_status if cache is not None else None,
        cache_age_seconds=(
            int(cache.age().total_seconds()) if cache is not None else None
        ),
        entitlement_available=cache is not None,
    )


def verify_login_code(
    settings,
    email: str,
    code: str,
    *,
    client: CloudClient | None = None,
) -> AccountStatus:
    """Verify one OTP, persist the token privately, and update live settings."""

    email = normalize_email(email)
    code = validate_code(code)
    owned = client is None
    try:
        client = client or client_from_settings(settings)
        token = client.verify_code(
            email,
            code,
            get_or_create_device_id(),
            get_device_name(),
        )
        try:
            clear_entitlement_cache()
            persist_account_credentials(token, email)
        except (OSError, FileLockTimeout) as exc:
            try:
                client.revoke_token()
            except Exception:
                pass
            raise AccountError(AccountErrorCode.STATE_UNAVAILABLE) from exc

        settings.account_token = token
        settings.account_email = email
        try:
            cache = refresh_entitlements(settings, client=client)
        except Exception:
            cache = load_entitlement_cache()
        return _status_from_cache(settings, cache)
    except AccountError:
        raise
    except CloudError as exc:
        raise _map_cloud_error(exc, verifying=True) from exc
    except OSError as exc:
        raise AccountError(AccountErrorCode.STATE_UNAVAILABLE) from exc
    except ValueError as exc:
        raise AccountError(AccountErrorCode.ACCOUNT_UNAVAILABLE) from exc
    finally:
        _close_owned(client, owned)


def get_account_status(settings) -> AccountStatus:
    """Return token-free account status using the canonical entitlement policy."""

    entitlement = check_entitlement(settings)
    cache = load_entitlement_cache()
    status = _status_from_cache(settings, cache)
    return AccountStatus(
        signed_in=status.signed_in,
        email=status.email,
        device_name=status.device_name,
        entitlement=entitlement,
        plan_status=status.plan_status,
        cache_age_seconds=status.cache_age_seconds,
        entitlement_available=status.entitlement_available,
    )


def logout_account(
    settings,
    *,
    client: CloudClient | None = None,
    client_factory: Callable[[], CloudClient] | None = None,
) -> LogoutResult:
    """Revoke first when possible, then always remove local account state."""

    revoked = False
    warning = None
    owned = False
    if getattr(settings, "account_token", None):
        try:
            if client is None:
                client = client_factory() if client_factory else client_from_settings(settings)
                owned = True
            client.revoke_token()
            revoked = True
        except Exception:
            warning = "Could not revoke this device while offline."
        finally:
            _close_owned(client, owned)

    try:
        remove_account_credentials()
    except (OSError, FileLockTimeout) as exc:
        raise AccountError(AccountErrorCode.STATE_UNAVAILABLE) from exc
    settings.account_token = None
    settings.account_email = None
    try:
        clear_entitlement_cache()
    except OSError as exc:
        raise AccountError(AccountErrorCode.STATE_UNAVAILABLE) from exc
    return LogoutResult(revoked_remotely=revoked, warning=warning)

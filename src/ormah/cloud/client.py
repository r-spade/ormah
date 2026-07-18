"""HTTP client and local account identity for the Ormah Cloud service."""

from __future__ import annotations

import socket
import time
import uuid
from importlib.metadata import PackageNotFoundError, version as package_version
from pathlib import Path
from typing import Any

import httpx

DATA_DIR = Path.home() / ".local" / "share" / "ormah"
DEVICE_ID_PATH = DATA_DIR / "device_id"
ACCOUNT_TOKEN_KEY = "ORMAH_ACCOUNT_TOKEN"
ACCOUNT_EMAIL_KEY = "ORMAH_ACCOUNT_EMAIL"
PROTOCOL_CACHE_SECONDS = 300
DEFAULT_MAX_CIPHERTEXT_BYTES = 512 * 1024 * 1024


def _client_version() -> str:
    try:
        return package_version("ormah")
    except PackageNotFoundError:
        return "0.0.0"


class CloudError(RuntimeError):
    """A network, protocol, or non-success response from Ormah Cloud."""

    def __init__(
        self,
        message: str,
        *,
        status_code: int | None = None,
        payload: Any = None,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.payload = payload


def _uuid4(value: str, path: Path) -> str:
    try:
        parsed = uuid.UUID(value)
    except ValueError as exc:
        raise CloudError(f"Invalid device id at {path}; expected UUIDv4.") from exc
    if parsed.version != 4 or parsed.variant != uuid.RFC_4122:
        raise CloudError(f"Invalid device id at {path}; expected UUIDv4.")
    return str(parsed)


def get_or_create_device_id(path: Path | None = None) -> str:
    """Return this installation's stable UUIDv4, creating it with mode 0600."""
    path = (DEVICE_ID_PATH if path is None else path).expanduser()
    if path.is_file():
        return _uuid4(path.read_text(encoding="utf-8").strip(), path)

    from ormah.setup import _atomic_write

    device_id = str(uuid.uuid4())
    path.parent.mkdir(parents=True, exist_ok=True)
    _atomic_write(str(path), device_id + "\n", mode=0o600)
    return device_id


def get_device_name() -> str:
    """Human-readable name attached to server-side device tokens."""
    return (socket.gethostname().strip() or "unknown-device")[:200]


def persist_account_credentials(token: str, email: str) -> None:
    """Persist account credentials without dropping unrelated environment keys."""
    from ormah.setup import _read_env_file, _write_env_file

    env = _read_env_file()
    env[ACCOUNT_TOKEN_KEY] = token
    env[ACCOUNT_EMAIL_KEY] = email.strip().lower()
    _write_env_file(env)


def remove_account_credentials() -> None:
    """Remove only Ormah account credentials from the global environment file."""
    from ormah.setup import _read_env_file, _write_env_file

    env = _read_env_file()
    changed = ACCOUNT_TOKEN_KEY in env or ACCOUNT_EMAIL_KEY in env
    env.pop(ACCOUNT_TOKEN_KEY, None)
    env.pop(ACCOUNT_EMAIL_KEY, None)
    if changed:
        _write_env_file(env)


class CloudClient:
    """Small synchronous client for the metadata-only Ormah Cloud API."""

    def __init__(
        self,
        base_url: str,
        token: str | None = None,
        *,
        timeout: float = 10.0,
        device_id: str | None = None,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/") + "/"
        self.token = token
        self.device_id = device_id
        self._protocol_cache: tuple[float, dict[str, Any]] | None = None
        self._http = httpx.Client(
            base_url=self.base_url,
            timeout=httpx.Timeout(timeout),
            transport=transport,
        )

    def close(self) -> None:
        self._http.close()

    def __enter__(self) -> CloudClient:
        return self

    def __exit__(self, *args) -> None:
        self.close()

    def _headers(self, authenticated: bool) -> dict[str, str]:
        headers = {
            "X-Ormah-Client-Version": _client_version(),
            "X-Request-ID": str(uuid.uuid4()),
        }
        if authenticated and self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        return headers

    def _request_json(
        self,
        method: str,
        path: str,
        *,
        authenticated: bool = True,
        **kwargs,
    ) -> dict[str, Any]:
        try:
            headers = self._headers(authenticated)
            supplied_headers = kwargs.pop("headers", None)
            if supplied_headers:
                headers.update(supplied_headers)
            response = self._http.request(
                method,
                path,
                headers=headers,
                **kwargs,
            )
        except httpx.HTTPError as exc:
            raise CloudError(f"Could not reach Ormah Cloud: {exc}") from exc

        try:
            payload = response.json()
        except ValueError as exc:
            raise CloudError(
                "Ormah Cloud returned an invalid JSON response.",
                status_code=response.status_code,
            ) from exc

        if not response.is_success:
            detail = payload.get("detail") if isinstance(payload, dict) else payload
            raise CloudError(
                f"Ormah Cloud request failed ({response.status_code}): {detail}",
                status_code=response.status_code,
                payload=payload,
            )
        if not isinstance(payload, dict):
            raise CloudError(
                "Ormah Cloud returned an invalid response shape.",
                status_code=response.status_code,
                payload=payload,
            )
        return payload

    def request_code(self, email: str) -> dict[str, Any]:
        return self._request_json(
            "POST",
            "auth/request-code",
            authenticated=False,
            json={"email": email.strip().lower()},
        )

    def verify_code(
        self,
        email: str,
        code: str,
        device_id: str,
        device_name: str,
    ) -> str:
        payload = self._request_json(
            "POST",
            "auth/verify",
            authenticated=False,
            json={
                "email": email.strip().lower(),
                "code": code.strip(),
                "device_id": device_id,
                "device_name": device_name,
            },
        )
        token = payload.get("token")
        if not isinstance(token, str) or not token:
            raise CloudError("Ormah Cloud verification response did not include a token.")
        self.token = token
        self.device_id = device_id
        return token

    def get_entitlements(self) -> dict[str, Any]:
        return self._request_json("GET", "me/entitlements")

    def get_protocol(self, *, refresh: bool = False) -> dict[str, Any]:
        now = time.monotonic()
        if (
            not refresh
            and self._protocol_cache is not None
            and now - self._protocol_cache[0] < PROTOCOL_CACHE_SECONDS
        ):
            return self._protocol_cache[1]
        payload = self._request_json("GET", "protocol", authenticated=False)
        self._protocol_cache = (now, payload)
        return payload

    def processing_limit(self, *, require_hardened_write: bool) -> int:
        try:
            protocol = self.get_protocol()
        except CloudError:
            if require_hardened_write:
                raise
            return DEFAULT_MAX_CIPHERTEXT_BYTES
        capabilities = protocol.get("capabilities")
        limit = protocol.get("max_ciphertext_bytes")
        if not isinstance(limit, int) or isinstance(limit, bool) or limit <= 0:
            raise CloudError("Ormah Cloud advertised an invalid ciphertext processing limit.")
        if require_hardened_write and (
            protocol.get("protocol_version") != 2
            or not isinstance(capabilities, list)
            or "immutable-promotion" not in capabilities
        ):
            raise CloudError(
                "Ormah Cloud does not advertise immutable upload promotion; upload stopped."
            )
        return min(limit, DEFAULT_MAX_CIPHERTEXT_BYTES)

    def revoke_token(self) -> dict[str, Any]:
        device_id = self.device_id or get_or_create_device_id()
        payload = self._request_json("GET", "me/tokens")
        tokens = payload.get("tokens")
        if not isinstance(tokens, list):
            raise CloudError("Ormah Cloud token-list response was malformed.")
        token_id = next(
            (
                item.get("token_id")
                for item in tokens
                if isinstance(item, dict)
                and item.get("device_id") == device_id
                and isinstance(item.get("token_id"), str)
            ),
            None,
        )
        if token_id is None:
            raise CloudError("No server token was found for this device.")
        return self._request_json("POST", "me/tokens/revoke", json={"token_id": token_id})

    def create_upload(self, store_id: str, size: int, sha256: str) -> dict[str, Any]:
        limit = self.processing_limit(require_hardened_write=True)
        if size > limit:
            raise CloudError(
                f"Encrypted bundle is larger than the supported {limit}-byte processing limit."
            )
        return self._request_json(
            "POST",
            f"stores/{store_id}/uploads",
            json={"size_bytes": size, "sha256": sha256},
        )

    def finalize_upload(
        self,
        store_id: str,
        upload_id: str,
        advance_head: dict[str, int] | None = None,
    ) -> dict[str, Any]:
        body = {"advance_head": advance_head} if advance_head is not None else {}
        try:
            return self._request_json(
                "POST",
                f"stores/{store_id}/uploads/{upload_id}/finalize",
                json=body,
            )
        except CloudError as exc:
            detail = exc.payload.get("detail") if isinstance(exc.payload, dict) else None
            if exc.status_code == 409 and isinstance(detail, dict) and "head" in detail:
                return detail
            raise

    def list_blobs(self, store_id: str) -> dict[str, Any]:
        return self._request_json("GET", f"stores/{store_id}/blobs")

    def presign_download(self, store_id: str, snapshot_id: str) -> dict[str, Any]:
        return self._request_json(
            "POST",
            f"stores/{store_id}/presign-download",
            json={"snapshot_id": snapshot_id},
        )

    def get_head(self, store_id: str) -> dict[str, Any]:
        return self._request_json("GET", f"stores/{store_id}/head")


def client_from_settings(settings) -> CloudClient:
    return CloudClient(settings.cloud_api_url, settings.account_token)

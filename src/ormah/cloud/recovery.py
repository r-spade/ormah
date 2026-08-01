"""Recovery-kit validation and device-loss readiness confirmation.

This is the only product service that turns a native saved-copy proof into
durable recovery readiness. It never returns kit bytes, identities, digests,
or filesystem locations.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import hmac
import os
from pathlib import Path
import re
import stat
from typing import Callable

from ormah.cloud import keys as cloud_keys
from ormah.cloud.crypto import identity_from_str
from ormah.cloud.keys import (
    MAX_RECOVERY_KIT_BYTES,
    RECOVERY_KIT_FORMAT_VERSION,
    STORE_ID_NAME,
    CloudKeyError,
    extract_recovery_kit_format_version,
    extract_store_id_from_text,
    load_identity_strings,
    rotate_key_and_recovery_kit,
)
from ormah.cloud.state import is_device_loss_recovery_ready, update_state
from ormah.cloud.store_lock import StoreLock


SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


class RecoveryKitError(RuntimeError):
    """Raised when recovery material cannot be proven current and complete."""


@dataclass(frozen=True)
class RecoveryReadiness:
    """Token-free result safe for the local API and product webview."""

    device_loss_recovery_ready: bool
    recovery_kit_verified_at: datetime


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _read_bounded_regular_file(path: Path) -> bytes:
    """Read one fixed sensitive file without following symlinks where supported."""

    flags = os.O_RDONLY
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    if hasattr(os, "O_BINARY"):
        flags |= os.O_BINARY
    try:
        descriptor = os.open(path, flags)
        with os.fdopen(descriptor, "rb") as handle:
            metadata = os.fstat(handle.fileno())
            if not stat.S_ISREG(metadata.st_mode) or metadata.st_size > MAX_RECOVERY_KIT_BYTES:
                raise RecoveryKitError("The recovery kit is invalid.")
            data = handle.read(MAX_RECOVERY_KIT_BYTES + 1)
    except RecoveryKitError:
        raise
    except OSError as exc:
        raise RecoveryKitError("The recovery kit is unavailable.") from exc
    if len(data) > MAX_RECOVERY_KIT_BYTES:
        raise RecoveryKitError("The recovery kit is invalid.")
    return data


def _recovery_kit_account_email(kit_bytes: bytes) -> str | None:
    """Read the descriptive email from the canonical Account section."""

    try:
        lines = kit_bytes.decode("utf-8").splitlines()
    except UnicodeDecodeError:
        return None
    try:
        account_index = lines.index("## Account")
    except ValueError:
        return None
    for line in lines[account_index + 1 :]:
        if line.startswith("## "):
            break
        if line.startswith("Email: "):
            value = line.removeprefix("Email: ").strip()
            return value if value and not value.startswith("<") else None
    return None


class RecoveryKitService:
    """Validate the canonical kit and confirm a reopened native saved copy."""

    def __init__(
        self,
        settings,
        *,
        key_path: Path | None = None,
        kit_path: Path | None = None,
        state_dir: Path | None = None,
        now: Callable[[], datetime] = _utc_now,
    ) -> None:
        self.settings = settings
        self.key_path = cloud_keys.KEY_PATH if key_path is None else key_path
        self.kit_path = cloud_keys.RECOVERY_KIT_PATH if kit_path is None else kit_path
        self.state_dir = state_dir
        self.now = now

    def validate_canonical_kit(self) -> None:
        """Validate the fixed canonical kit without changing readiness."""

        with StoreLock(self.settings.memory_dir):
            self._validate_canonical_kit_locked()

    def ensure_current_kit(self) -> bool:
        """Repair a stale canonical kit before a native save operation.

        Returns ``True`` only when the canonical file had to be regenerated.
        A regenerated file invalidates the proof for any previously saved copy;
        the native save-and-reopen flow establishes a new proof immediately.
        """

        with StoreLock(self.settings.memory_dir):
            try:
                store_id, kit_bytes = self._validate_canonical_kit_locked()
            except RecoveryKitError:
                store_id = self._active_store_id()
            else:
                account_email = getattr(self.settings, "account_email", None)
                if not account_email or _recovery_kit_account_email(kit_bytes) == account_email:
                    return False
            return self._regenerate_current_kit_locked(store_id)

    def _regenerate_current_kit_locked(self, store_id: str) -> bool:
        update_state(
            store_id,
            memory_dir=self.settings.memory_dir,
            state_dir=self.state_dir,
            recovery_kit_verified_at=None,
        )
        cloud_keys.write_recovery_kit(
            store_id,
            key_path=self.key_path,
            kit_path=self.kit_path,
            account_email=getattr(self.settings, "account_email", None),
        )
        self._validate_canonical_kit_locked()
        return True

    def confirm_saved_digest(self, digest: str) -> RecoveryReadiness:
        """Record a saved-copy proof only when its bytes equal the current valid kit."""

        if not isinstance(digest, str) or SHA256_PATTERN.fullmatch(digest) is None:
            raise ValueError("digest must be a lowercase SHA-256 value")

        with StoreLock(self.settings.memory_dir):
            store_id, kit_bytes = self._validate_canonical_kit_locked()
            canonical_digest = hashlib.sha256(kit_bytes).hexdigest()
            if not hmac.compare_digest(digest, canonical_digest):
                raise RecoveryKitError("The saved recovery kit could not be verified.")
            verified_at = self.now().astimezone(timezone.utc)
            state = update_state(
                store_id,
                memory_dir=self.settings.memory_dir,
                state_dir=self.state_dir,
                recovery_kit_verified_at=verified_at,
            )
            return RecoveryReadiness(
                is_device_loss_recovery_ready(
                    state,
                    enabled=bool(getattr(self.settings, "cloud_backup_enabled", False)),
                ),
                verified_at,
            )

    def rotate_current_key(self, *, account_email: str | None = None) -> Path:
        """Clear readiness before installing a recovery-first key rotation.

        The ordering is deliberately fail-safe: if cloud state cannot be made
        writable, neither key nor kit is touched. The prospective kit is then
        installed before the key becomes active, so every identity that could
        encrypt a bundle is recoverable even if the process stops between files.
        """

        with StoreLock(self.settings.memory_dir):
            store_id = self._active_store_id()
            update_state(
                store_id,
                memory_dir=self.settings.memory_dir,
                state_dir=self.state_dir,
                recovery_kit_verified_at=None,
            )
            _, kit_path = rotate_key_and_recovery_kit(
                store_id,
                key_path=self.key_path,
                kit_path=self.kit_path,
                account_email=account_email,
            )
            self._validate_canonical_kit_locked()
            return kit_path

    def _active_store_id(self) -> str:
        store_marker = Path(self.settings.memory_dir).expanduser() / STORE_ID_NAME
        if not store_marker.exists():
            raise RecoveryKitError(
                "Cloud recovery is not initialized; run `ormah cloud init` first."
            )
        try:
            marker_bytes = _read_bounded_regular_file(store_marker)
            if len(marker_bytes) > 128:
                raise RecoveryKitError("The active memory store is invalid.")
            marker_value = marker_bytes.decode("utf-8").strip()
            store_id = extract_store_id_from_text(f"store_id: {marker_value}")
        except (RecoveryKitError, UnicodeDecodeError, CloudKeyError, OSError) as exc:
            raise RecoveryKitError("The active memory store is invalid.") from exc
        if store_id is None or marker_value != store_id:
            raise RecoveryKitError("The active memory store is invalid.")
        return store_id

    def _validate_canonical_kit_locked(self) -> tuple[str, bytes]:
        store_id = self._active_store_id()
        kit_bytes = _read_bounded_regular_file(self.kit_path.expanduser())
        try:
            kit_text = kit_bytes.decode("utf-8")
            kit_store_id = extract_store_id_from_text(kit_text)
            store_claims = [
                line.strip()
                for line in kit_text.splitlines()
                if line.strip().startswith("store_id:")
            ]
            kit_format_version = extract_recovery_kit_format_version(kit_text)
            active_identity_strings = load_identity_strings(self.key_path)
            for identity in active_identity_strings:
                identity_from_str(identity)
            kit_identity_strings = [
                line.strip()
                for line in kit_text.splitlines()
                if line.strip().startswith("AGE-SECRET-KEY-")
            ]
            for identity in kit_identity_strings:
                identity_from_str(identity)
        except (UnicodeDecodeError, CloudKeyError, ValueError, OSError) as exc:
            raise RecoveryKitError("The recovery kit is invalid.") from exc

        if (
            kit_store_id != store_id
            or len(store_claims) != 1
            or kit_format_version != RECOVERY_KIT_FORMAT_VERSION
            or kit_identity_strings != active_identity_strings
        ):
            raise RecoveryKitError("The recovery kit is not current for this memory.")
        return store_id, kit_bytes

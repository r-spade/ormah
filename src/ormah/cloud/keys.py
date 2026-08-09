"""Key lifecycle, store identity, and the recovery kit (E08 §1, §5).

The key file keeps every identity ever generated — the current one first,
older ones retained below so pre-rotation bundles stay decryptable forever.
Nothing here is ever destructive; there is no ``--force``.
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone
from pathlib import Path

from ormah.cloud.crypto import (
    generate_identity,
    identity_from_str,
    identity_to_str,
)

logger = logging.getLogger(__name__)

CONFIG_DIR = Path.home() / ".config" / "ormah"
KEY_PATH = CONFIG_DIR / "cloud.key"
RECOVERY_KIT_PATH = CONFIG_DIR / "ormah-recovery-kit.md"
STORE_ID_NAME = ".store_id"
RECOVERY_KIT_FORMAT_VERSION = 1
MAX_RECOVERY_KIT_BYTES = 256 * 1024


class CloudKeyError(RuntimeError):
    """Raised for key-file lifecycle failures."""


def _atomic_write_0600(path: Path, text: str) -> None:
    from ormah.setup import _atomic_write

    path.parent.mkdir(parents=True, exist_ok=True)
    _atomic_write(str(path), text, mode=0o600)


# --- Key file ---


def key_file_exists(key_path: Path | None = None) -> bool:
    key_path = KEY_PATH if key_path is None else key_path
    return key_path.expanduser().is_file()


def load_identity_strings(key_path: Path | None = None) -> list[str]:
    """All identity strings in the key file, current first."""
    key_path = (KEY_PATH if key_path is None else key_path).expanduser()
    if not key_path.is_file():
        raise CloudKeyError(
            f"No cloud key found at {key_path}. Run `ormah cloud init` first."
        )
    strings = [
        line.strip()
        for line in key_path.read_text(encoding="utf-8").splitlines()
        if line.strip().startswith("AGE-SECRET-KEY-")
    ]
    if not strings:
        raise CloudKeyError(f"Cloud key file {key_path} contains no identities.")
    return strings


def load_identities(key_path: Path | None = None) -> list:
    """All identities for decryption, current first."""
    return [identity_from_str(s) for s in load_identity_strings(key_path)]


def current_recipient(key_path: Path | None = None):
    """The encryption recipient (public key of the current identity)."""
    return load_identities(key_path)[0].to_public()


def _serialize_key_file(identity_strings: list[str], rotated: bool) -> str:
    """Current identity first; older identities retained below it."""
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    lines = [
        "# Ormah cloud encryption identities (age). KEEP THIS FILE SAFE.",
        "# The first identity encrypts new bundles; all of them decrypt.",
        "# Never delete old identities — older backups still need them.",
        f"# current since {now}" + (" (rotation)" if rotated else ""),
        identity_strings[0],
    ]
    for old in identity_strings[1:]:
        lines.append("# retained pre-rotation identity")
        lines.append(old)
    return "\n".join(lines) + "\n"


def init_key(key_path: Path | None = None) -> str:
    """Generate the first identity. Refuses if a key file already exists."""
    key_path = (KEY_PATH if key_path is None else key_path).expanduser()
    if key_path.is_file():
        raise CloudKeyError(
            f"A cloud key already exists at {key_path}. "
            "To get a new encryption key, run `ormah cloud rotate-key` — "
            "it keeps old identities so existing backups stay readable."
        )
    identity_str = identity_to_str(generate_identity())
    _atomic_write_0600(key_path, _serialize_key_file([identity_str], rotated=False))
    return identity_str


def import_key(source: str, key_path: Path | None = None) -> list[str]:
    """Install identities from a recovery kit or key file (fresh machine).

    Accepts a path or raw pasted text; extracts every AGE-SECRET-KEY line,
    preserving order (current first). Refuses if a key file already exists.
    """
    key_path = (KEY_PATH if key_path is None else key_path).expanduser()
    if key_path.is_file():
        raise CloudKeyError(
            f"A cloud key already exists at {key_path}; refusing to overwrite. "
            "Move it aside first if you really mean to replace it."
        )
    source_path = Path(source).expanduser()
    try:
        text = source_path.read_text(encoding="utf-8") if source_path.is_file() else source
    except OSError:
        # Raw recovery-kit text can exceed the platform's maximum path length.
        # `source` is documented to accept either form, so a failed path probe
        # must fall back to parsing it as text.
        text = source
    strings = [
        line.strip()
        for line in text.splitlines()
        if line.strip().startswith("AGE-SECRET-KEY-")
    ]
    if not strings:
        raise CloudKeyError("No age identities found in the provided key material.")
    for s in strings:  # validate before writing anything
        identity_from_str(s)
    _atomic_write_0600(key_path, _serialize_key_file(strings, rotated=False))
    return strings


def _rotate_key_without_recovery_kit(key_path: Path | None = None) -> str:
    """Test/migration helper; product rotation must update the recovery kit first."""
    key_path = KEY_PATH if key_path is None else key_path
    existing = load_identity_strings(key_path)
    new_identity = identity_to_str(generate_identity())
    _atomic_write_0600(
        key_path.expanduser(),
        _serialize_key_file([new_identity, *existing], rotated=True),
    )
    return new_identity


def rotate_key_and_recovery_kit(
    store_id: str,
    *,
    key_path: Path | None = None,
    kit_path: Path | None = None,
    account_email: str | None = None,
) -> tuple[str, Path]:
    """Rotate the active identity without leaving recovery material behind.

    The recovery kit is installed first and contains both the prospective new
    identity and every existing identity. If the process stops before the key
    file is replaced, the old active identity is still present in that kit. A
    failure can therefore make the files temporarily disagree, but cannot make
    a successfully used encryption identity unrecoverable.
    """

    key_path = (KEY_PATH if key_path is None else key_path).expanduser()
    kit_path = (RECOVERY_KIT_PATH if kit_path is None else kit_path).expanduser()
    _ensure_recovery_kit_can_be_rewritten(kit_path)
    existing = load_identity_strings(key_path)
    new_identity = identity_to_str(generate_identity())
    identity_strings = [new_identity, *existing]

    _atomic_write_0600(
        kit_path,
        _serialize_recovery_kit(store_id, identity_strings, account_email),
    )
    _atomic_write_0600(
        key_path,
        _serialize_key_file(identity_strings, rotated=True),
    )
    return new_identity, kit_path


# --- store_id (E08 §1) ---


def _validate_store_id(value: str) -> str:
    """Strict store-id validator: RFC 4122 UUIDv4 only (E08 §1)."""
    try:
        parsed = uuid.UUID(value)
    except ValueError as e:
        raise CloudKeyError(f"Invalid store id (not a UUID): {value!r}") from e
    if parsed.version != 4 or parsed.variant != uuid.RFC_4122:
        raise CloudKeyError(
            f"Invalid store id (must be an RFC 4122 UUIDv4): {value!r}"
        )
    return str(parsed)


def get_or_create_store_id(memory_dir: Path) -> str:
    """UUIDv4 per memory store, persisted at <memory_dir>/.store_id."""
    memory_dir = memory_dir.expanduser()
    store_path = memory_dir / STORE_ID_NAME
    if store_path.is_file():
        value = store_path.read_text(encoding="utf-8").strip()
        try:
            return _validate_store_id(value)
        except CloudKeyError as e:
            raise CloudKeyError(f"Corrupt store id at {store_path}: {value!r}") from e
    memory_dir.mkdir(parents=True, exist_ok=True)
    store_id = str(uuid.uuid4())
    store_path.write_text(store_id + "\n", encoding="utf-8")
    return store_id


def extract_store_id_from_text(text: str) -> str | None:
    """Pull the store id out of already-loaded recovery-kit material.

    Fails closed: an explicit ``store_id:`` line that does not carry a valid
    UUIDv4 raises — a damaged kit must abort the import, not silently mint a
    new namespace. Returns None only when the material genuinely contains no
    store identifier (e.g. a bare key file).
    """
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("store_id:"):
            candidate = stripped.split(":", 1)[1].strip()
            try:
                return _validate_store_id(candidate)
            except CloudKeyError as e:
                raise CloudKeyError(
                    f"Recovery kit carries a malformed store id ({candidate!r}). "
                    "Refusing to import — continuing would mint a new namespace "
                    "and orphan the existing backups. Fix the kit's store_id line "
                    "(or restore an undamaged copy of the kit) and retry."
                ) from e
    # Older kits carried the store id as a bare UUID line.
    for line in text.splitlines():
        stripped = line.strip()
        if len(stripped) == 36:
            try:
                return _validate_store_id(stripped)
            except CloudKeyError:
                continue  # arbitrary 36-char line, not a store id claim
    return None


def extract_store_id(source: str) -> str | None:
    """Pull the store id out of recovery-kit material (path or raw text)."""

    source_path = Path(source).expanduser()
    try:
        text = source_path.read_text(encoding="utf-8") if source_path.is_file() else source
    except OSError:
        text = source
    return extract_store_id_from_text(text)


def install_store_id(memory_dir: Path, store_id: str) -> str:
    """Install a store id from a recovery kit on a fresh machine.

    The store id is the remote namespace (E08 §1) — a restore that generates
    a new one would orphan every existing backup. Fails if a *different* id
    is already installed; installing the same id is a no-op.
    """
    store_id = _validate_store_id(store_id)
    memory_dir = memory_dir.expanduser()
    store_path = memory_dir / STORE_ID_NAME
    if store_path.is_file():
        existing = store_path.read_text(encoding="utf-8").strip()
        if existing == store_id:
            return store_id
        raise CloudKeyError(
            f"This memory store already has store id {existing}, but the recovery "
            f"kit carries {store_id}. Refusing to overwrite — restoring into a "
            "store that belongs to a different remote namespace would orphan its "
            "backups. Use a fresh ORMAH_MEMORY_DIR or remove .store_id deliberately."
        )
    memory_dir.mkdir(parents=True, exist_ok=True)
    store_path.write_text(store_id + "\n", encoding="utf-8")
    return store_id


# --- Recovery kit ---


def extract_recovery_kit_format_version(text: str) -> int | None:
    """Return one declared kit format version, or None for a legacy kit."""

    claims = [
        line.strip().split(":", 1)[1].strip()
        for line in text.splitlines()
        if line.strip().startswith("format_version:")
    ]
    if not claims:
        return None
    if len(claims) != 1:
        raise CloudKeyError("Recovery kit contains multiple format versions.")
    try:
        version = int(claims[0])
    except ValueError as exc:
        raise CloudKeyError("Recovery kit format version is invalid.") from exc
    if version < 1:
        raise CloudKeyError("Recovery kit format version is invalid.")
    return version


def _ensure_recovery_kit_can_be_rewritten(kit_path: Path) -> None:
    """Prevent an older client from discarding a newer kit envelope."""

    if not kit_path.exists():
        return
    try:
        if not kit_path.is_file() or kit_path.stat().st_size > MAX_RECOVERY_KIT_BYTES:
            raise CloudKeyError("The existing recovery kit is invalid.")
        version = extract_recovery_kit_format_version(kit_path.read_text(encoding="utf-8"))
    except UnicodeDecodeError as exc:
        raise CloudKeyError("The existing recovery kit is invalid.") from exc
    if version is not None and version > RECOVERY_KIT_FORMAT_VERSION:
        raise CloudKeyError(
            "The recovery kit was created by a newer Ormah version; update Ormah before "
            "regenerating or rotating recovery material."
        )


def _serialize_recovery_kit(
    store_id: str,
    identity_strings: list[str],
    account_email: str | None,
) -> str:
    if account_email is not None and account_email.splitlines() != [account_email]:
        raise CloudKeyError("The account email cannot be represented safely in the recovery kit.")
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    identities_block = "\n".join(identity_strings)
    return f"""# Ormah Recovery Kit

format_version: {RECOVERY_KIT_FORMAT_VERSION}
Generated: {now}

> **Anyone with this file can read your backups; without it, nobody can —
> including us. Store it offline** (print it, or keep it on a USB drive in a
> drawer). Do not store it in the same cloud account it protects.

## Your encryption identities (all of them — order matters, current first)

```
{identities_block}
```

## Your store id

```
store_id: {store_id}
```

## Account

Email: {account_email or "<your ormah account email>"}

## Restore on a fresh machine

1. Install ormah, then log in:  `ormah account login`
2. Import this kit's keys:      `ormah cloud init --import-key <path-to-this-file>`
3. Restore your memory graph:   `ormah backup restore --cloud`

That's the whole procedure. Every identity listed above is needed to read
backups made before key rotations — never trim this list.
"""


def write_recovery_kit(
    store_id: str,
    key_path: Path | None = None,
    kit_path: Path | None = None,
    account_email: str | None = None,
) -> Path:
    """(Re)generate the versioned recovery kit with every identity."""

    kit_path = (RECOVERY_KIT_PATH if kit_path is None else kit_path).expanduser()
    _ensure_recovery_kit_can_be_rewritten(kit_path)
    identity_strings = load_identity_strings(key_path)
    kit = _serialize_recovery_kit(store_id, identity_strings, account_email)
    _atomic_write_0600(kit_path, kit)
    return kit_path

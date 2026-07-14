"""Encrypted snapshot bundles (E08 §2).

A bundle is ``age-encrypt( gzip-tar( snapshot_dir ) )`` where the snapshot dir
contains ``nodes/*.md``, ``deleted/*.md``, ``backup.json`` and a
``bundle-manifest.json`` with per-file SHA-256 hashes. Opening a bundle always
verifies every extracted file against the manifest — hash checking is not
optional; restore and verification both consume it.
"""

from __future__ import annotations

import hashlib
import io
import json
import logging
import os
import shutil
import tarfile
import tempfile
import unicodedata
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from ormah.cloud.crypto import decrypt_bytes, encrypt_bytes

logger = logging.getLogger(__name__)

BUNDLE_FORMAT_VERSION = 1
MANIFEST_NAME = "bundle-manifest.json"
BACKUP_MANIFEST_NAME = "backup.json"

# Extraction hardening limits (E08 §2)
MAX_MEMBERS = 100_000
MAX_EXPANDED_BYTES = 2 * 1024 * 1024 * 1024  # 2 GB
# Accepted constraint: ciphertext + decrypted compressed tar are held in
# memory (graphs are MBs; streaming decryption is not worth the complexity
# yet), so the encrypted input itself is capped.
MAX_BUNDLE_BYTES = 512 * 1024 * 1024  # 512 MB


class BundleError(RuntimeError):
    """Raised when building or opening a bundle fails."""


@dataclass(frozen=True)
class BundleInfo:
    store_id: str
    created_at: str
    reason: str
    node_count: int
    deleted_count: int
    total_bytes: int
    file_count: int
    sync_base_snapshot_id: str | None
    device_id: str | None


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _iter_bundle_files(backup_dir: Path) -> list[Path]:
    """Files that enter a bundle, as paths relative to backup_dir."""
    files: list[Path] = []
    for sub in ("nodes", "deleted"):
        d = backup_dir / sub
        if d.is_dir():
            files.extend(sorted(p for p in d.glob("*.md") if p.is_file()))
    backup_json = backup_dir / BACKUP_MANIFEST_NAME
    if backup_json.is_file():
        files.append(backup_json)
    return files


def build_bundle(
    backup_dir: Path,
    out_path: Path,
    recipients: list,
    *,
    store_id: str,
    reason: str = "manual",
    sync_base_snapshot_id: str | None = None,
    device_id: str | None = None,
) -> Path:
    """Build an encrypted bundle from a finished local backup directory.

    Returns the written ``.age`` path. The write is atomic (tmp + rename).
    """
    backup_dir = backup_dir.expanduser()
    if not backup_dir.is_dir():
        raise BundleError(f"Backup directory not found: {backup_dir}")

    files = _iter_bundle_files(backup_dir)
    entries = []
    node_count = deleted_count = total_bytes = 0
    payloads: list[tuple[str, bytes]] = []
    for path in files:
        rel = path.relative_to(backup_dir).as_posix()
        data = path.read_bytes()
        entries.append({"path": rel, "size": len(data), "sha256": _sha256(data)})
        payloads.append((rel, data))
        total_bytes += len(data)
        if rel.startswith("nodes/"):
            node_count += 1
        elif rel.startswith("deleted/"):
            deleted_count += 1

    manifest = {
        "format_version": BUNDLE_FORMAT_VERSION,
        "store_id": store_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "reason": reason,
        "node_count": node_count,
        "deleted_count": deleted_count,
        "total_bytes": total_bytes,
        "files": entries,
        "sync": {"base_snapshot_id": sync_base_snapshot_id, "device_id": device_id},
    }
    manifest_bytes = (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode("utf-8")

    tar_buffer = io.BytesIO()
    with tarfile.open(fileobj=tar_buffer, mode="w:gz") as tar:
        for rel, data in payloads:
            _add_member(tar, rel, data)
        _add_member(tar, MANIFEST_NAME, manifest_bytes)

    ciphertext = encrypt_bytes(tar_buffer.getvalue(), recipients)

    out_path = out_path.expanduser()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(out_path.parent), suffix=".tmp", prefix=".ormah_bundle_")
    try:
        os.write(fd, ciphertext)
        os.fsync(fd)
        os.close(fd)
        os.replace(tmp, str(out_path))
    except BaseException:
        try:
            os.close(fd)
        except OSError:
            pass
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise
    return out_path


def _add_member(tar: tarfile.TarFile, name: str, data: bytes) -> None:
    info = tarfile.TarInfo(name=name)
    info.size = len(data)
    info.mtime = 0  # deterministic archives; real times live in frontmatter
    info.mode = 0o600
    tar.addfile(info, io.BytesIO(data))


def _stream_members_to(
    staging: Path,
    plaintext: bytes,
    *,
    max_members: int,
    max_expanded_bytes: int,
) -> tuple[dict[str, str], dict[str, int]]:
    """Validate and stream every tar member into the staging dir in chunks,
    hashing while writing. Returns ({name: sha256}, {name: actual_size}).

    The staging dir is freshly created by the caller, so there is nothing
    pre-existing (symlinks included) for a write to follow. Limits are
    enforced on actual streamed bytes, not tar-declared sizes.
    """
    hashes: dict[str, str] = {}
    sizes: dict[str, int] = {}
    seen_folded: set[str] = set()
    expanded = 0
    try:
        tar = tarfile.open(fileobj=io.BytesIO(plaintext), mode="r:gz")
    except tarfile.TarError as e:
        raise BundleError(f"Bundle payload is not a valid tar.gz archive: {e}") from e
    with tar:
        members = 0
        for member in tar:
            members += 1
            if members > max_members:
                raise BundleError(f"Bundle exceeds member limit ({max_members}).")
            name = member.name
            if not member.isreg():
                raise BundleError(f"Bundle contains non-regular member: {name!r}")
            if name.startswith("/") or ".." in Path(name).parts or "\\" in name:
                raise BundleError(f"Bundle contains unsafe path: {name!r}")
            if not _member_allowed(name):
                raise BundleError(f"Bundle contains disallowed member: {name!r}")
            if name in hashes:
                raise BundleError(f"Bundle contains duplicate member: {name!r}")
            # Unicode-normalize before casefolding so canonically equivalent
            # names (composed vs decomposed) also count as collisions.
            folded = unicodedata.normalize("NFKC", name).casefold()
            if folded in seen_folded:
                raise BundleError(f"Bundle contains case-colliding member: {name!r}")
            fileobj = tar.extractfile(member)
            if fileobj is None:
                raise BundleError(f"Bundle member unreadable: {name!r}")

            target = staging / name
            target.parent.mkdir(parents=True, exist_ok=True)
            digest = hashlib.sha256()
            written = 0
            with open(target, "wb") as out:
                while chunk := fileobj.read(_STREAM_CHUNK):
                    written += len(chunk)
                    expanded += len(chunk)
                    if expanded > max_expanded_bytes:
                        raise BundleError(
                            f"Bundle exceeds expansion limit ({max_expanded_bytes} bytes)."
                        )
                    digest.update(chunk)
                    out.write(chunk)
            hashes[name] = digest.hexdigest()
            sizes[name] = written
            seen_folded.add(folded)
    return hashes, sizes


def _member_allowed(name: str) -> bool:
    """Strict allowlist: exactly nodes/*.md, deleted/*.md, and the two json files."""
    if name in (BACKUP_MANIFEST_NAME, MANIFEST_NAME):
        return True
    for prefix in ("nodes/", "deleted/"):
        if name.startswith(prefix):
            rest = name[len(prefix):]
            return bool(rest) and "/" not in rest and rest.endswith(".md")
    return False


_STREAM_CHUNK = 1024 * 1024


def _check_dest(dest_dir: Path) -> None:
    """The destination must be absent, or an empty real directory.

    ``is_symlink`` uses lstat, so a symlink pointing at an (empty) directory
    elsewhere is rejected rather than silently followed.
    """
    if dest_dir.is_symlink():
        raise BundleError(f"Destination {dest_dir} is a symlink; refusing to follow it.")
    if dest_dir.exists():
        if not dest_dir.is_dir() or any(dest_dir.iterdir()):
            raise BundleError(f"Destination {dest_dir} must be a new or empty directory.")


def open_bundle(
    bundle_path: Path,
    dest_dir: Path,
    identities: list,
    *,
    max_members: int = MAX_MEMBERS,
    max_expanded_bytes: int = MAX_EXPANDED_BYTES,
    max_bundle_bytes: int = MAX_BUNDLE_BYTES,
) -> BundleInfo:
    """Decrypt, safely extract, and hash-verify a bundle into dest_dir.

    Any manifest mismatch (hash, size, missing, or extra file) is a hard
    failure. Members are validated against a strict allowlist and streamed
    into a fresh staging directory created by us — tar metadata is never
    applied (strictly stronger than ``filter="data"``), nothing pre-existing
    can be followed, and memory stays bounded regardless of expanded size.
    dest_dir must not exist or must be empty; files land there only after
    every hash has verified.
    """
    bundle_path = bundle_path.expanduser()
    if not bundle_path.is_file():
        raise BundleError(f"Bundle not found: {bundle_path}")
    if bundle_path.stat().st_size > max_bundle_bytes:
        raise BundleError(
            f"Bundle exceeds size limit ({max_bundle_bytes} bytes); "
            "refusing to load it into memory."
        )

    dest_dir = dest_dir.expanduser()
    _check_dest(dest_dir)

    plaintext = decrypt_bytes(bundle_path.read_bytes(), identities)

    dest_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".ormah_extract_", dir=str(dest_dir.parent)))
    try:
        hashes, sizes = _stream_members_to(
            staging, plaintext, max_members=max_members,
            max_expanded_bytes=max_expanded_bytes,
        )

        if MANIFEST_NAME not in hashes:
            raise BundleError(f"Bundle is missing {MANIFEST_NAME}.")
        try:
            manifest = json.loads((staging / MANIFEST_NAME).read_text(encoding="utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as e:
            raise BundleError(f"Bundle manifest is not valid JSON: {e}") from e
        if manifest.get("format_version") != BUNDLE_FORMAT_VERSION:
            raise BundleError(
                f"Unsupported bundle format_version: {manifest.get('format_version')!r}"
            )

        manifest_files = {entry["path"]: entry for entry in manifest.get("files", [])}
        content_names = set(hashes) - {MANIFEST_NAME}

        missing = sorted(set(manifest_files) - content_names)
        extra = sorted(content_names - set(manifest_files))
        if missing:
            raise BundleError(f"Bundle is missing manifest-listed files: {missing}")
        if extra:
            raise BundleError(f"Bundle contains files not in the manifest: {extra}")

        for name in content_names:
            entry = manifest_files[name]
            if sizes[name] != entry["size"]:
                raise BundleError(
                    f"Size mismatch for {name!r}: manifest {entry['size']}, got {sizes[name]}."
                )
            if hashes[name] != entry["sha256"]:
                raise BundleError(
                    f"Hash mismatch for {name!r} — bundle is corrupt or tampered."
                )

        # Everything verified — move staged files into the (empty) destination.
        # Re-check right before commit so a symlink swapped in after the
        # initial validation cannot redirect the writes.
        _check_dest(dest_dir)
        for name in sorted(hashes):
            target = dest_dir / name
            target.parent.mkdir(parents=True, exist_ok=True)
            os.replace(str(staging / name), str(target))
    finally:
        shutil.rmtree(staging, ignore_errors=True)

    sync = manifest.get("sync") or {}
    return BundleInfo(
        store_id=manifest.get("store_id", ""),
        created_at=manifest.get("created_at", ""),
        reason=manifest.get("reason", ""),
        node_count=manifest.get("node_count", 0),
        deleted_count=manifest.get("deleted_count", 0),
        total_bytes=manifest.get("total_bytes", 0),
        file_count=len(content_names),
        sync_base_snapshot_id=sync.get("base_snapshot_id"),
        device_id=sync.get("device_id"),
    )

"""Reusable cloud snapshot discovery, verification, and full restore workflow."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import errno
import os
from pathlib import Path
import re
import shutil
import tempfile
from typing import Any, Callable
import uuid

import httpx

from ormah.backup import (
    RestoreResult,
    resolve_backup_user_node_id,
    service_from_settings,
)
from ormah.cloud.bundle import BundleError, open_bundle
from ormah.cloud.client import client_from_settings
from ormah.cloud.crypto import CloudCryptoError
from ormah.cloud.keys import (
    STORE_ID_NAME,
    get_or_create_store_id,
    key_file_exists,
    load_identities,
)
from ormah.cloud.transfer import download_file, sha256_file
from ormah.index.builder import IndexBuilder
from ormah.index.db import Database
from ormah.index.graph import GraphIndex
from ormah.store.file_store import FileStore
from ormah.store.markdown import parse_node


_SNAPSHOT_ID_RE = re.compile(r"[0-7][0-9A-HJKMNP-TV-Z]{25}")
_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_PREPARED_PREFIX = ".ormah-cloud-prepared-"
_PREPARED_NAME_RE = re.compile(r"\.ormah-cloud-prepared-[0-9a-f]{32}")
_PREPARED_TTL = timedelta(hours=24)
_FALLBACK_REASON_CODES = frozenset(
    {
        "bundle_corrupt",
        "ciphertext_hash_mismatch",
        "index_rebuild_failed",
        "node_parse_failed",
        "search_probe_failed",
        "self_pointer_invalid",
    }
)
RestoreProgress = Callable[[str], None]


class CloudRestoreError(RuntimeError):
    """Raised when a committed cloud snapshot cannot be restored."""

    def __init__(self, message: str, reason_code: str = "restore_failed") -> None:
        super().__init__(message)
        self.reason_code = reason_code


class CloudRestoreValidationError(CloudRestoreError):
    """Typed local proof failure shared by restore and scheduled verification."""

    def __init__(self, message: str, reason_code: str) -> None:
        super().__init__(message, reason_code)


@dataclass(frozen=True)
class PreparedCloudRestore:
    """A fully verified cloud snapshot held in private temporary staging."""

    snapshot_id: str
    backup_name: str
    verified_node_count: int
    snapshot_created_at: str | None = None
    skipped_newer_snapshots: int = 0


@dataclass(frozen=True)
class CloudRestoreResult:
    snapshot_id: str
    restore: RestoreResult
    skipped_newer_snapshots: int = 0


def _progress(callback: RestoreProgress | None, phase: str) -> None:
    if callback is not None:
        callback(phase)


def _existing_store_id(memory_dir: Path) -> str:
    memory_dir = Path(memory_dir).expanduser()
    if not (memory_dir / STORE_ID_NAME).is_file():
        raise CloudRestoreError(
            "Cloud store id is missing; import a recovery kit first.",
            "key_missing",
        )
    return get_or_create_store_id(memory_dir)


def _committed_blobs(client, store_id: str) -> list[dict[str, Any]]:
    payload = client.list_blobs(store_id)
    blobs = payload.get("blobs")
    if not isinstance(blobs, list):
        raise CloudRestoreError("Cloud snapshot listing was malformed.")
    return [blob for blob in blobs if isinstance(blob, dict)]


def _candidate_blobs(
    blobs: list[dict[str, Any]], requested: str | None
) -> list[dict[str, Any]]:
    if not blobs:
        raise CloudRestoreError(
            "No committed cloud snapshots are available for this store.",
            "no_restorable_backup",
        )
    if requested is None:
        # Snapshot IDs are server-generated ULIDs, so lexical descending order
        # is the protocol's authoritative newest-first order. Calling
        # _snapshot_id here also fails closed on malformed listing entries.
        return sorted(blobs, key=_snapshot_id, reverse=True)
    selected = [blob for blob in blobs if blob.get("snapshot_id") == requested]
    if not selected:
        raise CloudRestoreError(f"Cloud snapshot not found: {requested}")
    return selected


def _snapshot_id(blob: dict[str, Any]) -> str:
    value = blob.get("snapshot_id")
    if not isinstance(value, str) or _SNAPSHOT_ID_RE.fullmatch(value) is None:
        raise CloudRestoreError("Cloud snapshot listing contained an invalid snapshot id.")
    return value


def _snapshot_created_at(blob: dict[str, Any]) -> str | None:
    value = blob.get("created_at")
    if not isinstance(value, str):
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc).isoformat()


def _probe_search(database: Database, nodes: list[Any]) -> None:
    if not nodes:
        raise CloudRestoreError("Restored snapshot has no active memory to test.")
    graph = GraphIndex(database)
    for node in nodes:
        words = re.findall(r"\w{2,}", f"{node.title or ''} {node.content}")
        for word in sorted(words, key=len, reverse=True):
            if any(result["id"] == node.id for result in graph.fts_search(word, limit=10)):
                return
    raise CloudRestoreError("Scratch search did not return a known restored memory.")


def verify_extracted_bundle(extracted: Path, expected_store_id: str, info) -> int:
    """Prove a decrypted snapshot parses, rebuilds, and returns a known node."""

    if info.store_id != expected_store_id:
        raise CloudRestoreValidationError(
            "The encrypted backup belongs to a different memory store.",
            "store_mismatch",
        )

    active_nodes = []
    try:
        for dirname in ("nodes", "deleted"):
            for path in sorted((extracted / dirname).glob("*.md")):
                node = parse_node(path.read_text(encoding="utf-8"))
                if dirname == "nodes":
                    active_nodes.append(node)
    except OSError as exc:
        if exc.errno in {errno.ENOSPC, getattr(errno, "EDQUOT", errno.ENOSPC)}:
            raise
        raise CloudRestoreValidationError(
            "A restored memory node could not be parsed.",
            "node_parse_failed",
        ) from exc
    except Exception as exc:
        raise CloudRestoreValidationError(
            "A restored memory node could not be parsed.",
            "node_parse_failed",
        ) from exc

    try:
        resolve_backup_user_node_id(extracted)
    except Exception as exc:
        raise CloudRestoreValidationError(
            "The backup's active Self pointer does not match the restored memory graph.",
            "self_pointer_invalid",
        ) from exc

    database = Database(extracted / "scratch-index" / "index.db")
    try:
        try:
            database.init_schema()
            rebuilt = IndexBuilder(database, FileStore(extracted / "nodes")).full_rebuild()
        except OSError as exc:
            if exc.errno in {errno.ENOSPC, getattr(errno, "EDQUOT", errno.ENOSPC)}:
                raise
            raise CloudRestoreValidationError(
                "The scratch search index could not be rebuilt in this environment.",
                "index_environment_unavailable",
            ) from exc
        except Exception as exc:
            raise CloudRestoreValidationError(
                "The scratch search index could not be rebuilt in this environment.",
                "index_environment_unavailable",
            ) from exc
        if rebuilt != info.node_count:
            raise CloudRestoreValidationError(
                "The rebuilt memory count did not match the encrypted backup manifest.",
                "index_rebuild_failed",
            )
        try:
            _probe_search(database, active_nodes)
        except Exception as exc:
            raise CloudRestoreValidationError(
                "Scratch search did not return a known restored memory.",
                "search_probe_failed",
            ) from exc
        return rebuilt
    finally:
        database.close()


def _prepare_candidate(
    *,
    client,
    store_id: str,
    blob: dict[str, Any],
    backup_service,
    progress: RestoreProgress | None,
) -> PreparedCloudRestore:
    snapshot_id = _snapshot_id(blob)
    size_bytes = blob.get("size_bytes")
    if not isinstance(size_bytes, int) or isinstance(size_bytes, bool) or size_bytes <= 0:
        raise CloudRestoreError("Cloud snapshot listing did not include a valid size.")
    if size_bytes > client.processing_limit(require_hardened_write=False):
        raise CloudRestoreError("Cloud snapshot exceeds this client's safe processing limit.")
    expected_sha256 = blob.get("sha256")
    if not isinstance(expected_sha256, str) or _SHA256_RE.fullmatch(expected_sha256) is None:
        raise CloudRestoreError("Cloud snapshot listing did not include a verified hash.")

    presigned = client.presign_download(store_id, snapshot_id)
    get_url = presigned.get("get_url")
    if not isinstance(get_url, str) or not get_url:
        raise CloudRestoreError("Cloud download response was malformed.")

    staging = Path(
        tempfile.mkdtemp(prefix=".ormah-cloud-restore-", dir=str(backup_service.backup_dir))
    )
    encrypted = staging / f"{snapshot_id}.age"
    extracted = staging / "extracted"
    try:
        _progress(progress, "downloading")
        download_file(get_url, encrypted)
        if sha256_file(encrypted) != expected_sha256:
            raise CloudRestoreValidationError(
                "The encrypted backup did not match its cloud hash.",
                "ciphertext_hash_mismatch",
            )
        with encrypted.open("rb") as stream:
            has_age_header = stream.read(22) == b"age-encryption.org/v1\n"
        if not has_age_header:
            raise CloudRestoreValidationError(
                "The encrypted backup bundle is corrupt or incompatible.",
                "bundle_corrupt",
            )

        _progress(progress, "decrypting")
        try:
            info = open_bundle(encrypted, extracted, load_identities())
        except OSError:
            raise
        except CloudCryptoError as exc:
            raise CloudRestoreValidationError(
                "The recovery kit on this device cannot decrypt this memory store.",
                "decrypt_failed",
            ) from exc
        except BundleError as exc:
            raise CloudRestoreValidationError(
                "The encrypted backup bundle is corrupt or incompatible.",
                "bundle_corrupt",
            ) from exc
        _progress(progress, "checking")
        verified_node_count = verify_extracted_bundle(extracted, store_id, info)

        # The confirmation screen can remain open. Keep its decrypted staging
        # copy owner-only until the user applies or discards it.
        extracted.chmod(0o700)
        for staged_path in extracted.rglob("*"):
            staged_path.chmod(0o700 if staged_path.is_dir() else 0o600)

        name = f"{_PREPARED_PREFIX}{uuid.uuid4().hex}"
        prepared_backup = backup_service.backup_dir / name
        os.replace(extracted, prepared_backup)
        return PreparedCloudRestore(
            snapshot_id=snapshot_id,
            backup_name=name,
            verified_node_count=verified_node_count,
            snapshot_created_at=_snapshot_created_at(blob),
        )
    finally:
        shutil.rmtree(staging, ignore_errors=True)


def prepare_cloud_restore(
    settings,
    snapshot_id: str | None = None,
    *,
    progress: RestoreProgress | None = None,
) -> PreparedCloudRestore:
    """Find and fully verify the newest locally restorable cloud snapshot.

    When no exact snapshot is requested, corrupt or incompatible newer
    candidates are skipped in newest-first order. Live memory is never read or
    changed by this preparation step.
    """

    if not settings.account_token:
        raise CloudRestoreError(
            "Ormah Cloud login is required before restoring memory.",
            "sign_in_required",
        )
    if not key_file_exists():
        raise CloudRestoreError(
            "Cloud encryption key is missing; import a recovery kit first.",
            "key_missing",
        )

    store_id = _existing_store_id(settings.memory_dir)
    backup_service = service_from_settings(settings)
    backup_service.backup_dir.mkdir(parents=True, exist_ok=True)
    cleanup_abandoned_cloud_restores(backup_service.backup_dir)
    client = None
    try:
        _progress(progress, "discovering")
        client = client_from_settings(settings)
        candidates = _candidate_blobs(_committed_blobs(client, store_id), snapshot_id)
        rejected = 0
        for blob in candidates:
            try:
                prepared = _prepare_candidate(
                    client=client,
                    store_id=store_id,
                    blob=blob,
                    backup_service=backup_service,
                    progress=progress,
                )
            except httpx.HTTPError as exc:
                raise CloudRestoreError(
                    "Ormah Cloud could not be reached. Check your connection and try again.",
                    "offline",
                ) from exc
            except OSError as exc:
                raise CloudRestoreError(
                    "This device could not stage the cloud recovery point.",
                    "restore_failed",
                ) from exc
            except CloudRestoreError as exc:
                rejected += 1
                if snapshot_id is not None or exc.reason_code not in _FALLBACK_REASON_CODES:
                    raise
                continue
            _progress(progress, "ready")
            return PreparedCloudRestore(
                snapshot_id=prepared.snapshot_id,
                backup_name=prepared.backup_name,
                verified_node_count=prepared.verified_node_count,
                snapshot_created_at=prepared.snapshot_created_at,
                skipped_newer_snapshots=rejected,
            )
    finally:
        close = getattr(client, "close", None) if client is not None else None
        if close is not None:
            try:
                close()
            except Exception:
                pass

    raise CloudRestoreError(
        "No committed cloud snapshot passed local restore verification.",
        "no_restorable_backup",
    )


def restore_prepared_cloud_snapshot(
    settings,
    prepared: PreparedCloudRestore,
    *,
    rebuild_index: bool = True,
    engine=None,
    progress: RestoreProgress | None = None,
) -> CloudRestoreResult:
    """Delegate a prepared cloud snapshot to the canonical local restore path."""

    backup_service = service_from_settings(settings)
    prepared_path = _prepared_path(backup_service.backup_dir, prepared.backup_name)
    installed_name = backup_service._unique_backup_name(datetime.now(timezone.utc))
    os.replace(prepared_path, backup_service.backup_dir / installed_name)
    restored = backup_service.restore(
        installed_name,
        rebuild_index=rebuild_index and engine is None,
        progress=progress,
    )
    if rebuild_index and engine is not None:
        _progress(progress, "rebuilding")
        rebuilt_nodes = engine.reload_restored_graph()
        restored = RestoreResult(
            restored=restored.restored,
            safety_backup=restored.safety_backup,
            rebuilt_nodes=rebuilt_nodes,
        )
        _progress(progress, "reloading")
    return CloudRestoreResult(
        snapshot_id=prepared.snapshot_id,
        restore=restored,
        skipped_newer_snapshots=prepared.skipped_newer_snapshots,
    )


def discard_prepared_cloud_restore(settings, prepared_name: str) -> bool:
    """Delete one unconsumed verified preparation without touching real backups."""

    prepared = _prepared_path(service_from_settings(settings).backup_dir, prepared_name)
    if not prepared.exists():
        return False
    shutil.rmtree(prepared)
    return True


def cleanup_abandoned_cloud_restores(
    backup_dir: Path,
    *,
    now: datetime | None = None,
) -> int:
    """Remove crash-left preparations after a bounded local-only lifetime."""

    backup_dir = Path(backup_dir).expanduser()
    if not backup_dir.is_dir():
        return 0
    cutoff = (now or datetime.now(timezone.utc)) - _PREPARED_TTL
    removed = 0
    for candidate in backup_dir.iterdir():
        if (
            candidate.is_symlink()
            or not candidate.is_dir()
            or _PREPARED_NAME_RE.fullmatch(candidate.name) is None
        ):
            continue
        try:
            modified = datetime.fromtimestamp(candidate.stat().st_mtime, timezone.utc)
        except OSError:
            continue
        if modified <= cutoff:
            shutil.rmtree(candidate, ignore_errors=True)
            if not candidate.exists():
                removed += 1
    return removed


def _prepared_path(backup_dir: Path, prepared_name: str) -> Path:
    if _PREPARED_NAME_RE.fullmatch(prepared_name) is None:
        raise CloudRestoreError("The prepared recovery point is invalid or unavailable.")
    backup_dir = Path(backup_dir).expanduser().resolve()
    candidate = backup_dir / prepared_name
    if candidate.is_symlink():
        raise CloudRestoreError("The prepared recovery point is invalid or unavailable.")
    try:
        resolved = candidate.resolve(strict=True)
    except OSError as exc:
        raise CloudRestoreError(
            "The prepared recovery point is invalid or unavailable."
        ) from exc
    if resolved.parent != backup_dir or not resolved.is_dir():
        raise CloudRestoreError("The prepared recovery point is invalid or unavailable.")
    return resolved


def restore_cloud_snapshot(
    settings,
    snapshot_id: str | None = None,
    *,
    rebuild_index: bool = True,
) -> CloudRestoreResult:
    """Prepare a verified cloud snapshot and delegate replacement to BackupService."""

    prepared = prepare_cloud_restore(settings, snapshot_id)
    return restore_prepared_cloud_snapshot(
        settings,
        prepared,
        rebuild_index=rebuild_index,
    )

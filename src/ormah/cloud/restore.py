"""Reusable cloud snapshot discovery, verification, and full restore workflow."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import errno
import os
from pathlib import Path
import re
import shutil
import tempfile
from typing import Any, Callable

import httpx

from ormah.backup import (
    RestoreResult,
    resolve_backup_user_node_id,
    service_from_settings,
)
from ormah.cloud.bundle import open_bundle
from ormah.cloud.client import client_from_settings
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
    """A fully verified cloud snapshot installed as a normal local backup."""

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
        return blobs
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
            "self_pointer_invalid",
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
            "A restored memory file could not be validated.",
            "node_parse_failed",
        ) from exc
    except Exception as exc:
        raise CloudRestoreValidationError(
            "A restored memory file could not be validated.",
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
            raise CloudRestoreError("The encrypted backup did not match its cloud hash.")

        _progress(progress, "decrypting")
        try:
            info = open_bundle(encrypted, extracted, load_identities())
        except OSError:
            raise
        except Exception as exc:
            raise CloudRestoreValidationError(
                "The encrypted backup could not be decrypted and verified.",
                "decrypt_failed",
            ) from exc
        _progress(progress, "checking")
        verified_node_count = verify_extracted_bundle(extracted, store_id, info)

        name = backup_service._unique_backup_name(datetime.now(timezone.utc))
        installed_backup = backup_service.backup_dir / name
        os.replace(extracted, installed_backup)
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
            except CloudRestoreError:
                rejected += 1
                if snapshot_id is not None:
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

    restored = service_from_settings(settings).restore(
        prepared.backup_name,
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

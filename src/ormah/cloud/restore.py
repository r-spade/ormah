"""Reusable cloud snapshot discovery and full restore workflow."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any

from ormah.backup import RestoreResult, service_from_settings
from ormah.cloud.bundle import open_bundle
from ormah.cloud.client import client_from_settings
from ormah.cloud.keys import (
    STORE_ID_NAME,
    get_or_create_store_id,
    key_file_exists,
    load_identities,
)
from ormah.cloud.transfer import download_file


class CloudRestoreError(RuntimeError):
    """Raised when a committed cloud snapshot cannot be restored."""


@dataclass(frozen=True)
class CloudRestoreResult:
    snapshot_id: str
    restore: RestoreResult


def _existing_store_id(memory_dir: Path) -> str:
    memory_dir = Path(memory_dir).expanduser()
    if not (memory_dir / STORE_ID_NAME).is_file():
        raise CloudRestoreError("Cloud store id is missing; import a recovery kit first.")
    return get_or_create_store_id(memory_dir)


def _committed_blobs(client, store_id: str) -> list[dict[str, Any]]:
    payload = client.list_blobs(store_id)
    blobs = payload.get("blobs")
    if not isinstance(blobs, list):
        raise CloudRestoreError("Cloud snapshot listing was malformed.")
    return [blob for blob in blobs if isinstance(blob, dict)]


def _select_snapshot(blobs: list[dict[str, Any]], requested: str | None) -> dict[str, Any]:
    if not blobs:
        raise CloudRestoreError("No committed cloud snapshots are available for this store.")
    if requested:
        if any(blob.get("snapshot_id") == requested for blob in blobs):
            return next(blob for blob in blobs if blob.get("snapshot_id") == requested)
        raise CloudRestoreError(f"Cloud snapshot not found: {requested}")
    snapshot_id = blobs[0].get("snapshot_id")
    if not isinstance(snapshot_id, str) or not snapshot_id:
        raise CloudRestoreError("Cloud snapshot listing was malformed.")
    return blobs[0]


def restore_cloud_snapshot(
    settings,
    snapshot_id: str | None = None,
    *,
    rebuild_index: bool = True,
) -> CloudRestoreResult:
    """Download a verified cloud bundle and delegate replacement to BackupService."""
    if not settings.account_token:
        raise CloudRestoreError("Ormah Cloud login is required; run `ormah account login`.")
    if not key_file_exists():
        raise CloudRestoreError("Cloud encryption key is missing; import a recovery kit first.")

    store_id = _existing_store_id(settings.memory_dir)
    service = service_from_settings(settings)
    service.backup_dir.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=".ormah-cloud-restore-", dir=str(service.backup_dir))
    )
    client = None
    try:
        client = client_from_settings(settings)
        chosen_blob = _select_snapshot(_committed_blobs(client, store_id), snapshot_id)
        chosen = chosen_blob.get("snapshot_id")
        size_bytes = chosen_blob.get("size_bytes")
        if not isinstance(size_bytes, int) or isinstance(size_bytes, bool) or size_bytes <= 0:
            raise CloudRestoreError("Cloud snapshot listing did not include a valid size.")
        if size_bytes > client.processing_limit(require_hardened_write=False):
            raise CloudRestoreError("Cloud snapshot exceeds this client's safe processing limit.")
        presigned = client.presign_download(store_id, chosen)
        get_url = presigned.get("get_url")
        if not isinstance(get_url, str) or not get_url:
            raise CloudRestoreError("Cloud download response was malformed.")

        encrypted = staging / f"{chosen}.age"
        extracted = staging / "extracted"
        download_file(get_url, encrypted)
        info = open_bundle(encrypted, extracted, load_identities())
        if info.store_id != store_id:
            raise CloudRestoreError(
                f"Cloud bundle belongs to store {info.store_id!r}, not local store {store_id!r}."
            )

        name = service._unique_backup_name(datetime.now(timezone.utc))
        installed_backup = service.backup_dir / name
        os.replace(extracted, installed_backup)
        restored = service.restore(name, rebuild_index=rebuild_index)
        return CloudRestoreResult(snapshot_id=chosen, restore=restored)
    except CloudRestoreError:
        raise
    except Exception as exc:
        raise CloudRestoreError(str(exc)) from exc
    finally:
        close = getattr(client, "close", None) if client is not None else None
        if close is not None:
            try:
                close()
            except Exception:
                pass
        shutil.rmtree(staging, ignore_errors=True)

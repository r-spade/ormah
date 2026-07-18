"""Scheduled encrypted cloud backup and restore-verification jobs."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
import logging
from pathlib import Path
import re
import shutil
import tempfile

from ormah.backup import BackupInfo, service_from_settings
from ormah.cloud.bundle import open_bundle, build_bundle
from ormah.cloud.client import CloudError, client_from_settings
from ormah.cloud.entitlements import EntitlementStatus, check_entitlement
from ormah.cloud.keys import (
    STORE_ID_NAME,
    current_recipient,
    get_or_create_store_id,
    key_file_exists,
    load_identities,
)
from ormah.cloud.state import load_state, update_state
from ormah.cloud.transfer import download_file, put_file, sha256_file
from ormah.index.builder import IndexBuilder
from ormah.index.db import Database
from ormah.index.graph import GraphIndex
from ormah.store.file_store import FileStore
from ormah.store.markdown import parse_node


logger = logging.getLogger(__name__)


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _existing_store_id(memory_dir: Path) -> str | None:
    memory_dir = Path(memory_dir).expanduser()
    if not (memory_dir / STORE_ID_NAME).is_file():
        return None
    return get_or_create_store_id(memory_dir)


def _safe_update(store_id: str | None, **changes) -> None:
    if store_id is None:
        return
    try:
        update_state(store_id, **changes)
    except Exception as exc:
        logger.warning("Could not persist Ormah Cloud state: %s", exc)


def _record_upload_error(store_id: str | None, message: str) -> None:
    logger.warning("Ormah Cloud backup skipped or failed: %s", message)
    _safe_update(store_id, last_upload_error=message)


def _snapshot_files(root: Path) -> dict[str, Path]:
    files: dict[str, Path] = {}
    for dirname in ("nodes", "deleted"):
        source = Path(root) / dirname
        if not source.is_dir():
            continue
        for path in source.glob("*.md"):
            if path.is_file():
                files[f"{dirname}/{path.name}"] = path
    return files


def _backup_matches_memory(backup: BackupInfo, memory_dir: Path) -> bool:
    source_files = _snapshot_files(memory_dir)
    backup_files = _snapshot_files(backup.path)
    if source_files.keys() != backup_files.keys():
        return False
    return all(
        source_files[name].stat().st_size == backup_files[name].stat().st_size
        and sha256_file(source_files[name]) == sha256_file(backup_files[name])
        for name in source_files
    )


def _backup_for_upload(service) -> BackupInfo:
    latest = service.latest()
    if latest is not None and _backup_matches_memory(latest, service.memory_dir):
        return latest
    return service.create(reason="cloud")


def _upload_due(store_id: str, interval_hours: int, now: datetime) -> bool:
    last_upload = load_state(store_id).last_upload_at
    if last_upload is None:
        return True
    return now - last_upload >= timedelta(hours=interval_hours)


def run_cloud_backup(engine) -> str | None:
    """Upload one due encrypted backup; log and persist failures without raising."""
    settings = engine.settings
    store_id: str | None = None
    client = None
    try:
        if not settings.cloud_backup_enabled:
            logger.debug("Ormah Cloud backup is disabled")
            return None

        if not key_file_exists():
            store_id = _existing_store_id(settings.memory_dir)
            _record_upload_error(store_id, "Cloud encryption key is missing; run `ormah cloud init`.")
            return None

        store_id = _existing_store_id(settings.memory_dir)
        if store_id is None:
            _record_upload_error(None, "Cloud store id is missing; run `ormah cloud init`.")
            return None

        entitlement = check_entitlement(settings)
        if entitlement not in {EntitlementStatus.ACTIVE, EntitlementStatus.GRACE}:
            _record_upload_error(
                store_id,
                f"Cloud backup paused because entitlement is {entitlement.value}.",
            )
            return None

        service = service_from_settings(settings)
        if not service.has_backupable_memory():
            logger.debug("Skipping Ormah Cloud backup; no memory nodes exist yet")
            return None

        now = _utc_now()
        if not _upload_due(store_id, settings.cloud_backup_interval_hours, now):
            logger.debug("Skipping Ormah Cloud backup; latest upload is still fresh")
            return None

        backup = _backup_for_upload(service)
        with tempfile.TemporaryDirectory(prefix="ormah-cloud-upload-") as tmp:
            bundle = Path(tmp) / f"{backup.name}.age"
            build_bundle(
                backup.path,
                bundle,
                [current_recipient()],
                store_id=store_id,
                reason="cloud-backup",
            )
            size = bundle.stat().st_size
            digest = sha256_file(bundle)
            client = client_from_settings(settings)
            upload = client.create_upload(store_id, size, digest)
            upload_id = upload.get("upload_id")
            snapshot_id = upload.get("snapshot_id")
            put_url = upload.get("put_url")
            expires_at = upload.get("expires_at")
            required_headers = upload.get("required_headers", {})
            if not all(isinstance(value, str) and value for value in (upload_id, snapshot_id, put_url)):
                raise RuntimeError("Cloud upload reservation response was malformed.")
            if not isinstance(expires_at, str):
                raise RuntimeError("Cloud upload reservation did not include an expiry.")
            parsed_expiry = datetime.fromisoformat(expires_at.replace("Z", "+00:00"))
            if parsed_expiry.tzinfo is None or parsed_expiry <= _utc_now():
                raise RuntimeError("Cloud upload reservation is already expired.")
            if not isinstance(required_headers, dict) or not all(
                isinstance(name, str) and isinstance(value, str)
                for name, value in required_headers.items()
            ):
                raise RuntimeError("Cloud upload reservation headers were malformed.")

            put_file(put_url, bundle, required_headers)
            finalized = client.finalize_upload(store_id, upload_id)
            finalized_snapshot = finalized.get("snapshot_id")
            if finalized.get("status") != "committed" or finalized_snapshot != snapshot_id:
                raise RuntimeError("Cloud upload finalize response was malformed.")

        _safe_update(
            store_id,
            last_upload_at=_utc_now(),
            last_upload_snapshot_id=snapshot_id,
            last_upload_error=None,
        )
        logger.info("Uploaded encrypted Ormah Cloud snapshot %s", snapshot_id)
        return snapshot_id
    except Exception as exc:
        _record_upload_error(store_id, str(exc))
        return None
    finally:
        close = getattr(client, "close", None) if client is not None else None
        if close is not None:
            try:
                close()
            except Exception:
                pass


def _probe_search(database: Database, nodes: list) -> None:
    if not nodes:
        raise RuntimeError("Restored snapshot has no active node available for a search probe.")
    graph = GraphIndex(database)
    for node in nodes:
        words = re.findall(r"\w{2,}", f"{node.title or ''} {node.content}")
        for word in sorted(words, key=len, reverse=True):
            if any(result["id"] == node.id for result in graph.fts_search(word, limit=10)):
                return
    raise RuntimeError("Scratch search probe did not return a known restored node.")


def _verify_extracted_bundle(extracted: Path, expected_store_id: str, info) -> int:
    if info.store_id != expected_store_id:
        raise RuntimeError(
            f"Bundle store id {info.store_id!r} does not match local store {expected_store_id!r}."
        )

    active_nodes = []
    for dirname in ("nodes", "deleted"):
        for path in sorted((extracted / dirname).glob("*.md")):
            node = parse_node(path.read_text(encoding="utf-8"))
            if dirname == "nodes":
                active_nodes.append(node)

    database = Database(extracted / "scratch-index" / "index.db")
    try:
        database.init_schema()
        rebuilt = IndexBuilder(database, FileStore(extracted / "nodes")).full_rebuild()
        if rebuilt != info.node_count:
            raise RuntimeError(
                f"Scratch index rebuilt {rebuilt} nodes; bundle manifest declares {info.node_count}."
            )
        _probe_search(database, active_nodes)
        return rebuilt
    finally:
        database.close()


def run_restore_verification(engine) -> bool:
    """Prove the latest committed snapshot decrypts, rebuilds, and searches."""
    settings = engine.settings
    store_id: str | None = None
    snapshot_id: str | None = None
    client = None
    tmp_root: Path | None = None
    try:
        if not settings.cloud_backup_enabled:
            logger.debug("Ormah Cloud restore verification is disabled")
            return False
        if not key_file_exists():
            raise RuntimeError("Cloud encryption key is missing; cannot verify restore.")
        store_id = _existing_store_id(settings.memory_dir)
        if store_id is None:
            raise RuntimeError("Cloud store id is missing; cannot verify restore.")
        if not settings.account_token:
            raise RuntimeError("Ormah Cloud login is required to verify restore.")

        client = client_from_settings(settings)
        listing = client.list_blobs(store_id)
        blobs = listing.get("blobs")
        if not isinstance(blobs, list) or not blobs:
            raise RuntimeError("No committed cloud snapshot is available to verify.")
        latest = blobs[0]
        snapshot_id = latest.get("snapshot_id") if isinstance(latest, dict) else None
        if not isinstance(snapshot_id, str) or not snapshot_id:
            raise RuntimeError("Cloud snapshot listing was malformed.")
        size_bytes = latest.get("size_bytes") if isinstance(latest, dict) else None
        if not isinstance(size_bytes, int) or isinstance(size_bytes, bool) or size_bytes <= 0:
            raise RuntimeError("Cloud snapshot listing did not include a valid size.")
        if size_bytes > client.processing_limit(require_hardened_write=False):
            raise CloudError("Cloud snapshot exceeds this client's safe processing limit.")

        presigned = client.presign_download(store_id, snapshot_id)
        get_url = presigned.get("get_url")
        if not isinstance(get_url, str) or not get_url:
            raise RuntimeError("Cloud download response was malformed.")

        tmp_root = Path(tempfile.mkdtemp(prefix="ormah-restore-verify-"))
        bundle = tmp_root / f"{snapshot_id}.age"
        extracted = tmp_root / "snapshot"
        download_file(get_url, bundle)
        info = open_bundle(bundle, extracted, load_identities())
        _verify_extracted_bundle(extracted, store_id, info)

        verified_at = _utc_now()
        _safe_update(
            store_id,
            last_verify_at=verified_at,
            last_verify_ok=True,
            last_verify_snapshot_id=snapshot_id,
            last_verify_error=None,
        )
        logger.info("Verified Ormah Cloud snapshot %s is restorable", snapshot_id)
        return True
    except Exception as exc:
        message = str(exc)
        logger.warning("Ormah Cloud restore verification failed: %s", message)
        if store_id is None:
            try:
                store_id = _existing_store_id(settings.memory_dir)
            except Exception:
                store_id = None
        _safe_update(
            store_id,
            last_verify_at=_utc_now(),
            last_verify_ok=False,
            last_verify_snapshot_id=snapshot_id,
            last_verify_error=message,
        )
        return False
    finally:
        close = getattr(client, "close", None) if client is not None else None
        if close is not None:
            try:
                close()
            except Exception:
                pass
        if tmp_root is not None:
            shutil.rmtree(tmp_root, ignore_errors=True)

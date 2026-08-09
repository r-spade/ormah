"""Local memory backup and restore support."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
import logging
from pathlib import Path
import re
import shutil
import sqlite3
import threading
from typing import Callable

from ormah.index.builder import IndexBuilder
from ormah.index.db import Database
from ormah.models.node import MemoryNode
from ormah.store.markdown import parse_node
from ormah.store.file_store import FileStore

logger = logging.getLogger(__name__)

BACKUP_PREFIX = "memory_"
BACKUP_TIMESTAMP_FORMAT = "%Y-%m-%d_%H-%M-%S"
BACKUP_NAME_RE = re.compile(
    r"^memory_(?P<timestamp>\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2})(?:_(?P<suffix>\d{2}))?$"
)
MANIFEST_NAME = "backup.json"
BACKUP_FORMAT_VERSION = 2
SOURCE_DIRS = ("nodes", "deleted")
_BACKUP_CREATE_LOCK = threading.RLock()


class BackupError(RuntimeError):
    """Raised when a backup or restore operation cannot be completed."""


@dataclass(frozen=True)
class BackupInfo:
    """Metadata for one local memory backup."""

    name: str
    path: Path
    created_at: datetime
    node_count: int
    deleted_count: int
    size_bytes: int


@dataclass(frozen=True)
class RestoreResult:
    """Result of restoring a memory backup."""

    restored: BackupInfo
    safety_backup: BackupInfo | None
    rebuilt_nodes: int | None


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _directory_size(path: Path) -> int:
    total = 0
    if not path.exists():
        return total
    for item in path.rglob("*"):
        if item.is_file():
            try:
                total += item.stat().st_size
            except OSError:
                continue
    return total


def _count_markdown(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(1 for item in path.glob("*.md") if item.is_file())


def _is_system_self_node(path: Path) -> bool:
    try:
        node = parse_node(path.read_text(encoding="utf-8"))
    except Exception:
        return False
    return node.source == "system:self"


def _count_backupable_markdown(path: Path, *, ignore_system_self: bool = False) -> int:
    if not path.exists():
        return 0

    count = 0
    for item in path.glob("*.md"):
        if not item.is_file():
            continue
        if ignore_system_self and _is_system_self_node(item):
            continue
        count += 1
    return count


def _parse_backup_created_at(path: Path) -> datetime:
    manifest = path / MANIFEST_NAME
    if manifest.exists():
        try:
            data = json.loads(manifest.read_text(encoding="utf-8"))
            created = data.get("created_at")
            if isinstance(created, str):
                parsed = datetime.fromisoformat(created)
                if parsed.tzinfo is None:
                    parsed = parsed.replace(tzinfo=timezone.utc)
                return parsed
        except Exception:
            pass

    match = BACKUP_NAME_RE.match(path.name)
    if match:
        value = match.group("timestamp")
        return datetime.strptime(value, BACKUP_TIMESTAMP_FORMAT).replace(tzinfo=timezone.utc)

    try:
        return datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
    except OSError:
        return datetime.min.replace(tzinfo=timezone.utc)


def _read_index_user_node_id(memory_dir: Path) -> tuple[bool, str | None]:
    """Read the active Self pointer without creating or modifying the index."""
    db_path = Path(memory_dir) / "index.db"
    if not db_path.is_file():
        return False, None

    connection = None
    try:
        connection = sqlite3.connect(f"{db_path.resolve().as_uri()}?mode=ro", uri=True)
        row = connection.execute(
            "SELECT value FROM meta WHERE key = 'user_node_id'"
        ).fetchone()
    except sqlite3.Error:
        return False, None
    finally:
        if connection is not None:
            connection.close()

    if row is None:
        return False, None
    value = row[0]
    if not isinstance(value, str) or not value.strip():
        raise BackupError("The active Self pointer in index.db is invalid.")
    return True, value


def _parsed_nodes(nodes_dir: Path) -> tuple[list[MemoryNode], list[Path]]:
    nodes: list[MemoryNode] = []
    malformed: list[Path] = []
    if not nodes_dir.is_dir():
        return nodes, malformed
    for path in sorted(nodes_dir.glob("*.md")):
        if not path.is_file():
            continue
        try:
            nodes.append(parse_node(path.read_text(encoding="utf-8")))
        except Exception:
            malformed.append(path)
    return nodes, malformed


def _validate_user_node_id(nodes_dir: Path, user_node_id: str, *, context: str) -> str:
    nodes, _malformed = _parsed_nodes(nodes_dir)
    matches = [node for node in nodes if node.id == user_node_id]
    if len(matches) != 1 or matches[0].source != "system:self":
        raise BackupError(
            f"{context} records active Self node {user_node_id!r}, but that exact "
            "system:self node is not present in nodes/."
        )
    return user_node_id


def _infer_user_node_id(nodes_dir: Path, *, context: str) -> str | None:
    nodes, malformed = _parsed_nodes(nodes_dir)
    if malformed:
        raise BackupError(
            f"Cannot determine the active Self node for {context}: "
            f"{len(malformed)} node file(s) could not be parsed."
        )
    self_ids = sorted({node.id for node in nodes if node.source == "system:self"})
    if not self_ids:
        return None
    if len(self_ids) == 1:
        return self_ids[0]
    raise BackupError(
        f"Cannot determine the active Self node for {context}: found {len(self_ids)} "
        "system:self nodes and no portable active pointer. Create a fresh backup on the "
        "source machine before restoring."
    )


def resolve_current_user_node_id(memory_dir: Path) -> str | None:
    """Return the live graph's active Self node, validating it against Markdown."""
    memory_dir = Path(memory_dir).expanduser()
    has_pointer, user_node_id = _read_index_user_node_id(memory_dir)
    if has_pointer:
        assert user_node_id is not None
        return _validate_user_node_id(
            memory_dir / "nodes",
            user_node_id,
            context="the live memory graph",
        )
    return _infer_user_node_id(memory_dir / "nodes", context="the live memory graph")


def resolve_backup_user_node_id(backup_path: Path) -> str | None:
    """Return and validate the active Self pointer carried by a local backup.

    Version-1 backups predate the portable pointer. They remain restorable only
    when their Markdown contains zero or exactly one system:self node.
    """
    backup_path = Path(backup_path)
    manifest_path = backup_path / MANIFEST_NAME
    if not manifest_path.is_file():
        return _infer_user_node_id(backup_path / "nodes", context="the legacy backup")

    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise BackupError(f"Backup manifest is unreadable: {manifest_path}") from exc
    if not isinstance(manifest, dict):
        raise BackupError("Backup manifest must contain a JSON object.")

    version = manifest.get("version", 1)
    if not isinstance(version, int) or isinstance(version, bool) or version not in {1, 2}:
        raise BackupError(f"Unsupported backup manifest version: {version!r}")
    if version == 1:
        return _infer_user_node_id(backup_path / "nodes", context="the legacy backup")
    if "user_node_id" not in manifest:
        raise BackupError("Backup manifest version 2 is missing user_node_id.")

    user_node_id = manifest["user_node_id"]
    if user_node_id is None:
        inferred = _infer_user_node_id(backup_path / "nodes", context="the backup")
        if inferred is not None:
            raise BackupError(
                "Backup manifest records no active Self node, but nodes/ contains a "
                "system:self node."
            )
        return None
    if not isinstance(user_node_id, str) or not user_node_id.strip():
        raise BackupError("Backup manifest user_node_id must be a non-empty string or null.")
    return _validate_user_node_id(
        backup_path / "nodes",
        user_node_id,
        context="the backup",
    )


class BackupService:
    """Creates, lists, prunes, and restores local memory file backups."""

    def __init__(
        self,
        *,
        memory_dir: Path,
        backup_dir: Path,
        retention_count: int = 10,
    ) -> None:
        self.memory_dir = Path(memory_dir).expanduser()
        self.backup_dir = Path(backup_dir).expanduser()
        self.retention_count = retention_count

    def create(
        self,
        *,
        reason: str = "manual",
        now: datetime | None = None,
        prune: bool = True,
    ) -> BackupInfo:
        """Create a timestamped backup, serialized across in-process jobs."""
        with _BACKUP_CREATE_LOCK:
            return self._create(reason=reason, now=now, prune=prune)

    def _create(
        self,
        *,
        reason: str = "manual",
        now: datetime | None = None,
        prune: bool = True,
    ) -> BackupInfo:
        """Create a timestamped backup of source-of-truth memory files."""
        if not self.memory_dir.exists():
            raise BackupError(f"Memory directory not found: {self.memory_dir}")

        user_node_id = resolve_current_user_node_id(self.memory_dir)

        created_at = now or _utc_now()
        name = self._unique_backup_name(created_at)
        destination = self.backup_dir / name
        tmp_destination = self.backup_dir / f".{name}.tmp"

        self.backup_dir.mkdir(parents=True, exist_ok=True)
        if tmp_destination.exists():
            shutil.rmtree(tmp_destination)
        tmp_destination.mkdir(parents=True)

        try:
            for dirname in SOURCE_DIRS:
                source = self.memory_dir / dirname
                target = tmp_destination / dirname
                if source.exists():
                    shutil.copytree(source, target)
                else:
                    target.mkdir()

            if user_node_id is not None:
                _validate_user_node_id(
                    tmp_destination / "nodes",
                    user_node_id,
                    context="the new backup",
                )

            manifest = {
                "version": BACKUP_FORMAT_VERSION,
                "created_at": created_at.isoformat(),
                "reason": reason,
                "source": "ormah backup",
                "source_memory_dir": str(self.memory_dir),
                "user_node_id": user_node_id,
                "included": list(SOURCE_DIRS),
                "excluded": [
                    "index.db",
                    "index.db-shm",
                    "index.db-wal",
                    ".env",
                    ".store_id",
                    "logs",
                    "model_cache",
                ],
            }
            (tmp_destination / MANIFEST_NAME).write_text(
                json.dumps(manifest, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )

            tmp_destination.replace(destination)
        except Exception:
            shutil.rmtree(tmp_destination, ignore_errors=True)
            raise

        info = self._info_for(destination)
        if prune:
            self.prune()
        logger.info("Created Ormah memory backup: %s", info.path)
        return info

    def list(self) -> list[BackupInfo]:
        """Return backups newest first."""
        if not self.backup_dir.exists():
            return []
        backups = [
            self._info_for(path)
            for path in self.backup_dir.iterdir()
            if path.is_dir() and BACKUP_NAME_RE.match(path.name)
        ]
        return sorted(backups, key=lambda backup: backup.created_at, reverse=True)

    def latest(self) -> BackupInfo | None:
        """Return the newest backup, if one exists."""
        backups = self.list()
        return backups[0] if backups else None

    def backup_due(self, *, interval_hours: int, now: datetime | None = None) -> bool:
        """Return True when there is no backup or the newest one is too old."""
        latest = self.latest()
        if latest is None:
            return True
        current_time = now or _utc_now()
        age_seconds = (current_time - latest.created_at).total_seconds()
        return age_seconds >= interval_hours * 3600

    def has_backupable_memory(self) -> bool:
        """Return True when there are active or deleted memory nodes to protect."""
        return _count_backupable_markdown(
            self.memory_dir / "nodes",
            ignore_system_self=True,
        ) > 0 or _count_backupable_markdown(
            self.memory_dir / "deleted"
        ) > 0

    def prune(self) -> list[BackupInfo]:
        """Delete backups beyond retention_count and return what was removed."""
        if self.retention_count < 1:
            raise BackupError("Backup retention count must be at least 1")
        backups = self.list()
        removed: list[BackupInfo] = []
        for backup in backups[self.retention_count:]:
            shutil.rmtree(backup.path)
            removed.append(backup)
        if removed:
            logger.info("Pruned %d old Ormah memory backups", len(removed))
        return removed

    def restore(
        self,
        backup_ref: str,
        *,
        rebuild_index: bool = True,
        progress: Callable[[str], None] | None = None,
    ) -> RestoreResult:
        """Restore source-of-truth memory files from a backup.

        A safety backup of the current memory files is created before any
        destructive replacement happens.
        """
        backup_path = self._resolve_backup_ref(backup_ref)
        restored_info = self._info_for(backup_path)
        user_node_id = resolve_backup_user_node_id(backup_path)
        if progress is not None:
            progress("safety_backup")
        safety_backup = self.create(reason=f"pre-restore:{restored_info.name}", prune=False)

        self.memory_dir.mkdir(parents=True, exist_ok=True)
        restore_tmp = self.memory_dir / f".restore_{restored_info.name}"
        if restore_tmp.exists():
            shutil.rmtree(restore_tmp)
        restore_tmp.mkdir(parents=True)

        try:
            if progress is not None:
                progress("restoring")
            for dirname in SOURCE_DIRS:
                source = backup_path / dirname
                staged = restore_tmp / dirname
                if source.exists():
                    shutil.copytree(source, staged)
                else:
                    staged.mkdir()

            for dirname in SOURCE_DIRS:
                target = self.memory_dir / dirname
                if target.exists():
                    shutil.rmtree(target)
                shutil.move(str(restore_tmp / dirname), str(target))
        finally:
            shutil.rmtree(restore_tmp, ignore_errors=True)

        if progress is not None and rebuild_index:
            progress("rebuilding")
        rebuilt_nodes = self.rebuild_index() if rebuild_index else None
        self._write_user_node_id(user_node_id)
        self.prune()
        logger.info("Restored Ormah memory backup: %s", restored_info.path)
        return RestoreResult(
            restored=restored_info,
            safety_backup=safety_backup,
            rebuilt_nodes=rebuilt_nodes,
        )

    def rebuild_index(self) -> int:
        """Rebuild the SQLite index from restored markdown node files."""
        database = Database(self.memory_dir / "index.db")
        try:
            database.init_schema()
            builder = IndexBuilder(database, FileStore(self.memory_dir / "nodes"))
            return builder.full_rebuild()
        finally:
            database.close()

    def _write_user_node_id(self, user_node_id: str | None) -> None:
        """Replace target-local identity state with the restored graph's pointer."""
        database = Database(self.memory_dir / "index.db")
        try:
            database.init_schema()
            with database.transaction() as connection:
                if user_node_id is None:
                    connection.execute("DELETE FROM meta WHERE key = 'user_node_id'")
                else:
                    connection.execute(
                        "INSERT OR REPLACE INTO meta (key, value) VALUES ('user_node_id', ?)",
                        (user_node_id,),
                    )
        finally:
            database.close()

    def _unique_backup_name(self, created_at: datetime) -> str:
        base = f"{BACKUP_PREFIX}{created_at.strftime(BACKUP_TIMESTAMP_FORMAT)}"
        candidate = base
        suffix = 1
        while (self.backup_dir / candidate).exists() or (self.backup_dir / f".{candidate}.tmp").exists():
            candidate = f"{base}_{suffix:02d}"
            suffix += 1
        return candidate

    def _resolve_backup_ref(self, backup_ref: str) -> Path:
        candidate_name = Path(backup_ref).name
        candidate = self.backup_dir / candidate_name
        try:
            resolved_backup_dir = self.backup_dir.resolve()
            resolved_candidate = candidate.resolve()
        except OSError as exc:
            raise BackupError(f"Backup not found: {backup_ref}") from exc

        if resolved_candidate.parent != resolved_backup_dir:
            raise BackupError("Backup restore only supports backups in the configured backup directory")
        if not resolved_candidate.is_dir() or not BACKUP_NAME_RE.match(resolved_candidate.name):
            raise BackupError(f"Backup not found: {backup_ref}")
        return resolved_candidate

    def _info_for(self, path: Path) -> BackupInfo:
        return BackupInfo(
            name=path.name,
            path=path,
            created_at=_parse_backup_created_at(path),
            node_count=_count_markdown(path / "nodes"),
            deleted_count=_count_markdown(path / "deleted"),
            size_bytes=_directory_size(path),
        )


def service_from_settings(settings) -> BackupService:
    """Build a BackupService from a Settings-like object."""
    return BackupService(
        memory_dir=settings.memory_dir,
        backup_dir=settings.backup_dir,
        retention_count=settings.backup_retention_count,
    )


def run_auto_backup(engine) -> BackupInfo | None:
    """Create a backup when automatic backups are enabled and due."""
    settings = engine.settings
    if not settings.backup_enabled:
        logger.debug("Automatic Ormah memory backups are disabled")
        return None

    service = service_from_settings(settings)
    if not service.has_backupable_memory():
        logger.debug("Skipping Ormah memory backup; no memory nodes exist yet")
        return None

    if not service.backup_due(interval_hours=settings.backup_interval_hours):
        logger.debug("Skipping Ormah memory backup; newest backup is still fresh")
        return None

    try:
        return service.create(reason="automatic")
    except Exception as exc:
        logger.warning("Automatic Ormah memory backup failed: %s", exc)
        return None

"""Small in-process coordinator for long-running cloud protection operations."""

from __future__ import annotations

from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from enum import StrEnum
import logging
from pathlib import Path
import threading
from typing import TYPE_CHECKING, Callable
import uuid

from ormah.cloud.keys import STORE_ID_NAME, get_or_create_store_id
from ormah.cloud.state import (
    CURRENT_CLOUD_STATE_SCHEMA_VERSION,
    ProtectionIntentStatus,
    ProtectionOperation,
    ProtectionOperationKind,
    ProtectionOperationPhase,
    load_state,
)

if TYPE_CHECKING:
    from ormah.engine.memory_engine import MemoryEngine

logger = logging.getLogger(__name__)


class LocalOperationStatus(StrEnum):
    """Lifecycle of work submitted by a local desktop/API caller."""

    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass(frozen=True)
class LocalOperation:
    """Token-free polling record for one local background invocation."""

    operation_id: str
    kind: ProtectionOperationKind
    status: LocalOperationStatus
    submitted_at: datetime
    started_at: datetime | None = None
    finished_at: datetime | None = None
    result: ProtectionOperation | None = None
    error_code: str | None = None
    error_message: str | None = None

    def to_payload(self) -> dict[str, object]:
        result = self.result
        return {
            "operation_id": self.operation_id,
            "kind": self.kind.value,
            "status": self.status.value,
            "submitted_at": self.submitted_at.isoformat(),
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "finished_at": self.finished_at.isoformat() if self.finished_at else None,
            "durable_operation_id": result.operation_id if result else None,
            "phase": result.phase.value if result else None,
            "protection_state": result.state.value if result else None,
            "reason_code": result.reason_code.value if result and result.reason_code else None,
            "message": result.message if result else self.error_message,
            "snapshot_id": result.snapshot_id if result else None,
            "protection_intent_id": result.protection_intent_id if result else None,
            "verified_node_count": result.verified_node_count if result else None,
            "snapshot_created_at": result.snapshot_created_at if result else None,
            "skipped_newer_snapshots": result.skipped_newer_snapshots if result else 0,
            "safety_backup_name": result.safety_backup_name if result else None,
            "error_code": self.error_code,
        }


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


class ProtectionOperationCoordinator:
    """Run blocking protection services without holding an HTTP request open.

    The coordinator intentionally stores only a bounded, process-local polling
    history. Durable recovery and product truth remain in ``CloudState`` and
    ``CloudProtectionService``; callers must refresh the protection status if a
    server restart makes a local operation ID unavailable.
    """

    def __init__(self, *, max_workers: int = 4, max_history: int = 128) -> None:
        if max_workers < 1 or max_history < 1:
            raise ValueError("operation coordinator limits must be positive")
        self._executor = ThreadPoolExecutor(
            max_workers=max_workers,
            thread_name_prefix="ormah-protection",
        )
        self._max_history = max_history
        self._lock = threading.Lock()
        self._operations: dict[str, LocalOperation] = {}
        self._active_by_key: dict[tuple[str, ...], str] = {}
        self._finished: deque[str] = deque()
        self._claimed_results: set[str] = set()
        self._closed = False

    def submit(
        self,
        *,
        key: tuple[str, ...],
        kind: ProtectionOperationKind,
        action: Callable[[], ProtectionOperation],
    ) -> tuple[LocalOperation, bool]:
        """Submit work or return the matching operation already in progress."""
        if not key:
            raise ValueError("operation deduplication key must not be empty")
        with self._lock:
            if self._closed:
                raise RuntimeError("operation coordinator is shutting down")
            active_id = self._active_by_key.get(key)
            if active_id is not None:
                return self._operations[active_id], True

            operation = LocalOperation(
                operation_id=str(uuid.uuid4()),
                kind=kind,
                status=LocalOperationStatus.QUEUED,
                submitted_at=_utc_now(),
            )
            self._operations[operation.operation_id] = operation
            self._active_by_key[key] = operation.operation_id
            self._executor.submit(self._run, operation.operation_id, key, action)
            return operation, False

    def get(self, operation_id: str) -> LocalOperation | None:
        with self._lock:
            operation = self._operations.get(operation_id)
            return replace(operation) if operation is not None else None

    def claim_ready_result(
        self,
        operation_id: str,
        *,
        kind: ProtectionOperationKind,
    ) -> ProtectionOperation | None:
        """Atomically claim one completed preparation for a follow-up action."""

        with self._lock:
            operation = self._operations.get(operation_id)
            if (
                operation is None
                or operation.kind is not kind
                or operation.status is not LocalOperationStatus.COMPLETED
                or operation.result is None
                or operation.result.phase is not ProtectionOperationPhase.READY
                or operation_id in self._claimed_results
            ):
                return None
            self._claimed_results.add(operation_id)
            return replace(operation.result)

    def shutdown(self, *, wait: bool = True) -> None:
        with self._lock:
            self._closed = True
        self._executor.shutdown(wait=wait, cancel_futures=False)

    def _run(
        self,
        operation_id: str,
        key: tuple[str, ...],
        action: Callable[[], ProtectionOperation],
    ) -> None:
        self._update(
            operation_id,
            status=LocalOperationStatus.RUNNING,
            started_at=_utc_now(),
        )
        try:
            result = action()
            if not isinstance(result, ProtectionOperation):
                raise TypeError("protection operation returned an invalid result")
        except Exception as exc:
            # Exception strings can contain local paths, tokens, or provider URLs.
            # Keep the API generic; durable service failures use safe typed results.
            logger.warning(
                "Local protection operation failed; kind=%s error_type=%s",
                self._operations[operation_id].kind.value,
                type(exc).__name__,
            )
            self._finish(
                operation_id,
                key,
                status=LocalOperationStatus.FAILED,
                error_code="operation_failed",
                error_message="The protection operation could not be completed.",
            )
            return
        self._finish(
            operation_id,
            key,
            status=LocalOperationStatus.COMPLETED,
            result=result,
        )

    def _update(self, operation_id: str, **changes: object) -> None:
        with self._lock:
            self._operations[operation_id] = replace(
                self._operations[operation_id],
                **changes,
            )

    def _finish(
        self,
        operation_id: str,
        key: tuple[str, ...],
        **changes: object,
    ) -> None:
        with self._lock:
            self._operations[operation_id] = replace(
                self._operations[operation_id],
                finished_at=_utc_now(),
                **changes,
            )
            if self._active_by_key.get(key) == operation_id:
                del self._active_by_key[key]
            self._finished.append(operation_id)
            while len(self._finished) > self._max_history:
                expired_id = self._finished.popleft()
                self._operations.pop(expired_id, None)
                self._claimed_results.discard(expired_id)


def resume_interrupted_enable(
    engine: MemoryEngine,
    coordinator: ProtectionOperationCoordinator,
) -> LocalOperation | None:
    """Queue the one durable Protect operation that may survive a process crash.

    Upload reservation/finalize recovery and exact-snapshot verification are
    already idempotent inside ``CloudProtectionService.enable``. Startup only
    needs to rediscover a ``running`` intent and hand it back to that service.
    """

    memory_dir = Path(engine.settings.memory_dir).expanduser().resolve()
    if not (memory_dir / STORE_ID_NAME).is_file():
        return None

    try:
        store_id = get_or_create_store_id(memory_dir)
        state = load_state(store_id)
        if state.schema_version > CURRENT_CLOUD_STATE_SCHEMA_VERSION:
            logger.warning(
                "Interrupted cloud protection was not resumed because its state "
                "requires a newer Ormah version."
            )
            return None
    except Exception as exc:
        logger.warning(
            "Interrupted cloud protection could not be inspected at startup; error_type=%s",
            type(exc).__name__,
        )
        return None

    intent_id = state.pending_protection_intent_id
    if (
        state.pending_protection_status is not ProtectionIntentStatus.RUNNING
        or not isinstance(intent_id, str)
        or not intent_id
    ):
        return None

    try:
        from ormah.cloud.protection import CloudProtectionService

        service = CloudProtectionService.from_engine(engine)
        operation, deduplicated = coordinator.submit(
            key=(str(memory_dir), f"enable:{intent_id}"),
            kind=ProtectionOperationKind.ENABLE,
            action=lambda: service.enable(intent_id),
        )
    except Exception as exc:
        logger.warning(
            "Interrupted cloud protection could not be resumed at startup; error_type=%s",
            type(exc).__name__,
        )
        return None
    logger.info(
        "%s interrupted cloud protection intent %s",
        "Reused" if deduplicated else "Resuming",
        intent_id,
    )
    return operation

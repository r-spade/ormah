from __future__ import annotations

import threading
import time
from types import SimpleNamespace
import uuid

from ormah.cloud import protection, state
from ormah.cloud.keys import get_or_create_store_id
from ormah.cloud.operations import (
    LocalOperationStatus,
    ProtectionOperationCoordinator,
    resume_interrupted_enable,
)
from ormah.cloud.state import (
    ProtectionIntentStatus,
    ProtectionOperation,
    ProtectionOperationKind,
    ProtectionOperationPhase,
    ProtectionState,
)


def _result(operation_id: str = "durable-operation") -> ProtectionOperation:
    return ProtectionOperation(
        operation_id=operation_id,
        kind=ProtectionOperationKind.BACKUP,
        phase=ProtectionOperationPhase.COMPLETED,
        state=ProtectionState.VERIFICATION_PENDING,
        snapshot_id="01ARZ3NDEKTSV4RRFFQ69G5FAV",
        verified_node_count=42,
    )


def _wait_for_terminal(
    coordinator: ProtectionOperationCoordinator,
    operation_id: str,
):
    deadline = time.monotonic() + 2
    while time.monotonic() < deadline:
        operation = coordinator.get(operation_id)
        assert operation is not None
        if operation.status in {
            LocalOperationStatus.COMPLETED,
            LocalOperationStatus.FAILED,
        }:
            return operation
        time.sleep(0.005)
    raise AssertionError("operation did not finish")


def test_coordinator_returns_immediately_and_deduplicates_active_work():
    coordinator = ProtectionOperationCoordinator(max_workers=1)
    release = threading.Event()
    calls = 0

    def action():
        nonlocal calls
        calls += 1
        assert release.wait(timeout=2)
        return _result()

    try:
        first, first_deduplicated = coordinator.submit(
            key=("store", "backup"),
            kind=ProtectionOperationKind.BACKUP,
            action=action,
        )
        second, second_deduplicated = coordinator.submit(
            key=("store", "backup"),
            kind=ProtectionOperationKind.BACKUP,
            action=action,
        )

        assert first.operation_id == second.operation_id
        assert first_deduplicated is False
        assert second_deduplicated is True
        release.set()
        completed = _wait_for_terminal(coordinator, first.operation_id)
        assert completed.status is LocalOperationStatus.COMPLETED
        assert completed.result == _result()
        assert completed.to_payload()["verified_node_count"] == 42
        assert calls == 1
    finally:
        release.set()
        coordinator.shutdown()


def test_coordinator_redacts_unexpected_exception_details():
    coordinator = ProtectionOperationCoordinator(max_workers=1)

    def action():
        raise RuntimeError("bearer-secret https://signed.example.test/private")

    try:
        operation, _ = coordinator.submit(
            key=("store", "verify"),
            kind=ProtectionOperationKind.VERIFY,
            action=action,
        )
        failed = _wait_for_terminal(coordinator, operation.operation_id)
        payload = failed.to_payload()
        serialized = str(payload)
        assert failed.status is LocalOperationStatus.FAILED
        assert payload["error_code"] == "operation_failed"
        assert payload["message"] == "The protection operation could not be completed."
        assert "bearer-secret" not in serialized
        assert "signed.example" not in serialized
    finally:
        coordinator.shutdown()


def test_coordinator_bounds_finished_history():
    coordinator = ProtectionOperationCoordinator(max_workers=1, max_history=1)
    try:
        first, _ = coordinator.submit(
            key=("store", "backup"),
            kind=ProtectionOperationKind.BACKUP,
            action=_result,
        )
        _wait_for_terminal(coordinator, first.operation_id)
        second, _ = coordinator.submit(
            key=("store", "verify"),
            kind=ProtectionOperationKind.VERIFY,
            action=_result,
        )
        _wait_for_terminal(coordinator, second.operation_id)

        assert coordinator.get(first.operation_id) is None
        assert coordinator.get(second.operation_id) is not None
    finally:
        coordinator.shutdown()


def test_ready_restore_preparation_can_only_be_claimed_once():
    coordinator = ProtectionOperationCoordinator(max_workers=1)

    def prepared_result():
        return ProtectionOperation(
            operation_id="durable-restore-preparation",
            kind=ProtectionOperationKind.RESTORE,
            phase=ProtectionOperationPhase.READY,
            state=ProtectionState.PROTECTED,
            snapshot_id="01ARZ3NDEKTSV4RRFFQ69G5FAV",
            verified_node_count=42,
            prepared_backup_name="private-prepared-backup",
        )

    try:
        operation, _ = coordinator.submit(
            key=("store", "restore", "prepare"),
            kind=ProtectionOperationKind.RESTORE,
            action=prepared_result,
        )
        completed = _wait_for_terminal(coordinator, operation.operation_id)
        assert "private-prepared-backup" not in str(completed.to_payload())

        claimed = coordinator.claim_ready_result(
            operation.operation_id,
            kind=ProtectionOperationKind.RESTORE,
        )
        repeated = coordinator.claim_ready_result(
            operation.operation_id,
            kind=ProtectionOperationKind.RESTORE,
        )

        assert claimed is not None
        assert claimed.prepared_backup_name == "private-prepared-backup"
        assert repeated is None
    finally:
        coordinator.shutdown()


def test_startup_resumes_and_deduplicates_running_enable(tmp_path, monkeypatch):
    memory_dir = tmp_path / "memory"
    memory_dir.mkdir()
    monkeypatch.setattr(state, "CLOUD_STATE_DIR", tmp_path / "cloud-state")
    store_id = get_or_create_store_id(memory_dir)
    intent_id = str(uuid.uuid4())
    state.update_state(
        store_id,
        memory_dir=memory_dir,
        protection_state=ProtectionState.UPLOADING_FIRST_BACKUP,
        pending_protection_intent_id=intent_id,
        pending_protection_store_id=store_id,
        pending_protection_status=ProtectionIntentStatus.RUNNING,
    )
    release = threading.Event()
    calls: list[str] = []

    class FakeService:
        def enable(self, resumed_intent_id: str):
            calls.append(resumed_intent_id)
            assert release.wait(timeout=2)
            return _result(resumed_intent_id)

    monkeypatch.setattr(
        protection.CloudProtectionService,
        "from_engine",
        lambda engine: FakeService(),
    )
    engine = SimpleNamespace(settings=SimpleNamespace(memory_dir=memory_dir))
    coordinator = ProtectionOperationCoordinator(max_workers=1)
    try:
        first = resume_interrupted_enable(engine, coordinator)
        second = resume_interrupted_enable(engine, coordinator)

        assert first is not None
        assert second is not None
        assert first.operation_id == second.operation_id
        release.set()
        completed = _wait_for_terminal(coordinator, first.operation_id)
        assert completed.status is LocalOperationStatus.COMPLETED
        assert calls == [intent_id]
    finally:
        release.set()
        coordinator.shutdown()


def test_startup_ignores_non_running_or_uninitialized_stores(tmp_path, monkeypatch):
    monkeypatch.setattr(state, "CLOUD_STATE_DIR", tmp_path / "cloud-state")
    coordinator = ProtectionOperationCoordinator(max_workers=1)
    try:
        empty_engine = SimpleNamespace(
            settings=SimpleNamespace(memory_dir=tmp_path / "empty-memory")
        )
        assert resume_interrupted_enable(empty_engine, coordinator) is None

        memory_dir = tmp_path / "initialized-memory"
        memory_dir.mkdir()
        store_id = get_or_create_store_id(memory_dir)
        state.update_state(
            store_id,
            memory_dir=memory_dir,
            protection_state=ProtectionState.SUBSCRIPTION_REQUIRED,
            pending_protection_intent_id=str(uuid.uuid4()),
            pending_protection_store_id=store_id,
            pending_protection_status=ProtectionIntentStatus.ACCOUNT_BOUND,
        )
        initialized_engine = SimpleNamespace(
            settings=SimpleNamespace(memory_dir=memory_dir)
        )
        assert resume_interrupted_enable(initialized_engine, coordinator) is None
    finally:
        coordinator.shutdown()


def test_startup_resume_failure_never_blocks_server_start(tmp_path, monkeypatch):
    memory_dir = tmp_path / "memory"
    memory_dir.mkdir()
    monkeypatch.setattr(state, "CLOUD_STATE_DIR", tmp_path / "cloud-state")
    store_id = get_or_create_store_id(memory_dir)
    intent_id = str(uuid.uuid4())
    state.update_state(
        store_id,
        memory_dir=memory_dir,
        protection_state=ProtectionState.INITIALIZING,
        pending_protection_intent_id=intent_id,
        pending_protection_store_id=store_id,
        pending_protection_status=ProtectionIntentStatus.RUNNING,
    )
    monkeypatch.setattr(
        protection.CloudProtectionService,
        "from_engine",
        lambda engine: (_ for _ in ()).throw(RuntimeError("startup failure")),
    )
    engine = SimpleNamespace(settings=SimpleNamespace(memory_dir=memory_dir))
    coordinator = ProtectionOperationCoordinator(max_workers=1)
    try:
        assert resume_interrupted_enable(engine, coordinator) is None
    finally:
        coordinator.shutdown()

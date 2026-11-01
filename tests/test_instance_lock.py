from __future__ import annotations

import json
import multiprocessing
import os
from pathlib import Path
from typing import Any

import pytest

from automation_safety import ENTRY_INTENT, OrderRequest, SubmissionGate
from tradeos.execution_lock import (
    ExecutionLock,
    LockContendedError,
    LockOwnershipError,
    LockState,
    assert_current_process_holds_lock,
    current_process_holds_lock,
    inspect_lock,
)


def _hold_lock_until_released(
    lock_path: str,
    ready_connection: Any,
    release_event: Any,
) -> None:
    try:
        lock = ExecutionLock(lock_path)
        lock.acquire()
        ready_connection.send(
            {
                "result": "acquired",
                "pid": os.getpid(),
                "registered": current_process_holds_lock(lock_path),
            }
        )
        # Windows ``spawn`` can spend more than ten seconds importing the test
        # module on a cold/contended host.  Keep the holder alive long enough
        # for the contender to exercise the kernel lock rather than timing out
        # during interpreter startup.
        release_event.wait(60)
        lock.release()
    except BaseException as exc:
        ready_connection.send(
            {"result": "error", "error_type": type(exc).__name__, "message": str(exc)}
        )
        raise
    finally:
        ready_connection.close()


def _attempt_lock_once(lock_path: str, result_connection: Any) -> None:
    import tradeos.execution_lock as execution_lock_module

    # Point the gate's default ownership check at this test's isolated lock.
    execution_lock_module.DEFAULT_EXECUTION_LOCK_PATH = Path(lock_path)
    lock = ExecutionLock(lock_path)
    try:
        lock.acquire()
    except LockContendedError as exc:
        class CountingBroker:
            def __init__(self) -> None:
                self.submissions = 0

            def submit_market_order(self, **kwargs: Any) -> None:
                self.submissions += 1
                raise AssertionError("contending process reached broker submission")

        class NeverReconciler:
            def reconcile(self) -> None:
                raise AssertionError("lock rejection must occur before reconciliation")

        broker = CountingBroker()
        gate = SubmissionGate(
            broker=broker,  # type: ignore[arg-type]
            storage=object(),  # type: ignore[arg-type]
            reconciler=NeverReconciler(),  # type: ignore[arg-type]
            submission_enabled=True,
        )
        outcome = gate.submit(
            OrderRequest("COP", "sell", "1", ENTRY_INTENT, "contender-decision")
        )
        result_connection.send(
            {
                "result": "contended",
                "error": str(exc),
                "registered": current_process_holds_lock(lock_path),
                "status_state": exc.status.state.value,
                "holder_pid": (exc.status.metadata or {}).get("pid"),
                "gate_allowed": outcome.allowed,
                "gate_reason": outcome.reason,
                "broker_submissions": broker.submissions,
            }
        )
    else:
        result_connection.send(
            {
                "result": "acquired",
                "registered": current_process_holds_lock(lock_path),
            }
        )
        lock.release()
    finally:
        result_connection.close()


def _acquire_and_crash(lock_path: str, ready_connection: Any) -> None:
    lock = ExecutionLock(lock_path)
    lock.acquire()
    ready_connection.send(
        {
            "pid": os.getpid(),
            "instance_id": (lock.metadata or {}).get("instance_id"),
        }
    )
    ready_connection.close()
    os._exit(23)


def _receive(connection: Any, timeout: float = 30.0) -> Any:
    assert connection.poll(timeout), "spawned lock process did not respond in time"
    return connection.recv()


def _stop_process(process: multiprocessing.Process) -> None:
    if process.pid is None:
        return
    process.join(5)
    if process.is_alive():
        process.terminate()
        process.join(5)


def test_context_manager_registers_current_holder_and_keeps_file(
    tmp_path: Path,
) -> None:
    lock_path = tmp_path / "automation.lock"

    missing = inspect_lock(lock_path)
    assert missing.state is LockState.UNLOCKED
    assert missing.file_exists is False
    assert not lock_path.exists(), "a read-only probe must not create the lock file"

    with ExecutionLock(lock_path) as lock:
        assert lock.held is True
        assert current_process_holds_lock(lock_path) is True
        assert_current_process_holds_lock(lock_path)
        status = inspect_lock(lock_path)
        assert status.state is LockState.HELD
        assert status.held_by_current_process is True
        assert status.metadata_valid is True
        assert status.metadata is not None
        assert status.metadata["state"] == "held"
        assert status.metadata["pid"] == os.getpid()

    assert lock_path.exists(), "the persistent audit file must not be deleted"
    assert current_process_holds_lock(lock_path) is False
    with pytest.raises(LockOwnershipError, match="does not hold"):
        assert_current_process_holds_lock(lock_path)
    released = inspect_lock(lock_path)
    assert released.state is LockState.UNLOCKED
    assert released.metadata_valid is True
    assert released.metadata is not None
    assert released.metadata["state"] == "released"
    assert "released_at" in released.metadata


def test_second_lock_object_in_same_process_is_blocked(tmp_path: Path) -> None:
    lock_path = tmp_path / "automation.lock"
    first = ExecutionLock(lock_path).acquire()
    try:
        with pytest.raises(LockContendedError, match="already held") as captured:
            ExecutionLock(lock_path).acquire()
        assert captured.value.status.held_by_current_process is True
        assert first.acquire() is first
    finally:
        first.release()


def test_probe_does_not_rewrite_persistent_metadata(tmp_path: Path) -> None:
    lock_path = tmp_path / "automation.lock"
    with ExecutionLock(lock_path):
        pass
    before = lock_path.read_bytes()

    status = inspect_lock(lock_path)

    assert status.state is LockState.UNLOCKED
    assert lock_path.read_bytes() == before


def test_two_spawned_processes_cannot_both_hold_execution_lock(
    tmp_path: Path,
) -> None:
    """Exercise the real Windows/POSIX kernel lock, not a thread-only fake."""

    context = multiprocessing.get_context("spawn")
    lock_path = str(tmp_path / "automation.lock")
    release_holder = context.Event()
    holder_receive, holder_send = context.Pipe(duplex=False)
    holder = context.Process(
        target=_hold_lock_until_released,
        args=(lock_path, holder_send, release_holder),
        name="execution-lock-holder",
    )
    contender_receive, contender_send = context.Pipe(duplex=False)
    contender = context.Process(
        target=_attempt_lock_once,
        args=(lock_path, contender_send),
        name="execution-lock-contender",
    )

    holder.start()
    holder_send.close()
    holder_result = _receive(holder_receive)
    assert holder_result["result"] == "acquired"
    assert holder_result["registered"] is True

    try:
        observed = inspect_lock(lock_path)
        assert observed.state is LockState.HELD
        assert observed.held_by_current_process is False
        assert observed.metadata_valid is True
        assert observed.metadata is not None
        assert observed.metadata["pid"] == holder_result["pid"]

        contender.start()
        contender_send.close()
        contender_result = _receive(contender_receive)
        contender.join(10)

        assert contender.exitcode == 0
        assert contender_result["result"] == "contended"
        assert contender_result["registered"] is False
        assert contender_result["status_state"] == "held"
        assert contender_result["holder_pid"] == holder_result["pid"]
        assert "already held" in contender_result["error"]
        assert contender_result["gate_allowed"] is False
        assert contender_result["gate_reason"] == "trading_lock_not_held"
        assert contender_result["broker_submissions"] == 0
    finally:
        release_holder.set()
        _stop_process(contender)
        _stop_process(holder)
        holder_receive.close()
        contender_receive.close()

    assert holder.exitcode == 0
    after_release = inspect_lock(lock_path)
    assert after_release.state is LockState.UNLOCKED
    assert after_release.metadata is not None
    assert after_release.metadata["state"] == "released"


def test_crash_releases_kernel_lock_and_next_holder_audits_stale_metadata(
    tmp_path: Path,
) -> None:
    context = multiprocessing.get_context("spawn")
    lock_path = tmp_path / "automation.lock"
    ready_receive, ready_send = context.Pipe(duplex=False)
    crashing_holder = context.Process(
        target=_acquire_and_crash,
        args=(str(lock_path), ready_send),
        name="execution-lock-crash",
    )

    crashing_holder.start()
    ready_send.close()
    crashed = _receive(ready_receive)
    crashing_holder.join(10)
    ready_receive.close()

    assert crashing_holder.exitcode == 23
    assert inspect_lock(lock_path).state is LockState.UNLOCKED

    replacement = ExecutionLock(lock_path).acquire()
    try:
        metadata = dict(replacement.metadata or {})
        recovery = metadata["stale_recovery"]
        assert recovery["reason"] == "kernel_lock_was_free_despite_held_metadata"
        assert recovery["previous_holder"]["pid"] == crashed["pid"]
        assert recovery["previous_holder"]["instance_id"] == crashed["instance_id"]
    finally:
        replacement.release()


def test_invalid_old_metadata_is_replaced_only_after_acquisition(
    tmp_path: Path,
) -> None:
    lock_path = tmp_path / "automation.lock"
    lock_path.write_text("not-json", encoding="utf-8")

    before = lock_path.read_bytes()
    status = inspect_lock(lock_path)
    assert status.state is LockState.UNLOCKED
    assert status.metadata_valid is False
    assert lock_path.read_bytes() == before

    with ExecutionLock(lock_path) as holder:
        metadata = dict(holder.metadata or {})
        assert metadata["stale_recovery"]["reason"].startswith(
            "invalid_previous_metadata_replaced"
        )
        assert json.loads(lock_path.read_text(encoding="utf-8"))["state"] == "held"

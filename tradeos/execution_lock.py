"""Cross-process execution lock for order-submitting automation.

The lock file is deliberately persistent.  Process exclusion comes from the
kernel lock held on the open file handle, not from the presence, age, or
contents of the file.  The JSON document in the file is an audit aid only.
"""

from __future__ import annotations

import errno
import json
import os
import socket
import stat
import threading
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, BinaryIO, Mapping

from tradeos.runtime_paths import TRADEOS_REPOSITORY_ROOT


DEFAULT_EXECUTION_LOCK_PATH = TRADEOS_REPOSITORY_ROOT / ".automation_v1.lock"
_METADATA_SCHEMA_VERSION = 1
_MAX_METADATA_BYTES = 64 * 1024
_WINDOWS_LOCK_BYTE_OFFSET = _MAX_METADATA_BYTES + 1


class LockState(str, Enum):
    """Observed kernel-lock state at the instant of a probe."""

    HELD = "held"
    UNLOCKED = "unlocked"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class LockStatus:
    """Safe, credential-free snapshot of an execution lock."""

    path: Path
    state: LockState
    held_by_current_process: bool
    file_exists: bool
    metadata: Mapping[str, Any] | None = None
    metadata_valid: bool = False
    detail: str = ""

    @property
    def is_locked(self) -> bool | None:
        """Return true/false for a conclusive probe, otherwise ``None``."""

        if self.state is LockState.HELD:
            return True
        if self.state is LockState.UNLOCKED:
            return False
        return None

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly representation for CLI/dashboard callers."""

        return {
            "path": str(self.path),
            "state": self.state.value,
            "is_locked": self.is_locked,
            "held_by_current_process": self.held_by_current_process,
            "file_exists": self.file_exists,
            "metadata": dict(self.metadata) if self.metadata is not None else None,
            "metadata_valid": self.metadata_valid,
            "detail": self.detail,
        }


class ExecutionLockError(RuntimeError):
    """Base error for execution-lock failures."""


class LockContendedError(ExecutionLockError):
    """Raised when another holder already owns the kernel lock."""

    def __init__(self, status: LockStatus) -> None:
        self.status = status
        holder = "unknown holder"
        if status.metadata_valid and status.metadata is not None:
            pid = status.metadata.get("pid")
            acquired_at = status.metadata.get("acquired_at")
            holder = f"pid={pid}, acquired_at={acquired_at}"
        super().__init__(
            "Automation execution lock is already held "
            f"({holder}; path={status.path})."
        )


class LockOwnershipError(ExecutionLockError):
    """Raised when a process tries to use a lock it does not own."""


_REGISTRY_GUARD = threading.RLock()
_HELD_LOCKS: dict[Path, "ExecutionLock"] = {}


def _canonical_path(path: str | os.PathLike[str] | None) -> Path:
    selected = DEFAULT_EXECUTION_LOCK_PATH if path is None else Path(path)
    return selected.expanduser().resolve(strict=False)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _open_lock_file(path: Path, *, create: bool) -> BinaryIO:
    flags = os.O_RDWR
    if create:
        flags |= os.O_CREAT
    if hasattr(os, "O_BINARY"):
        flags |= os.O_BINARY

    descriptor = os.open(path, flags, 0o600)
    try:
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise ExecutionLockError(f"Execution lock path is not a regular file: {path}")
        os.set_inheritable(descriptor, False)
        return os.fdopen(descriptor, "r+b", buffering=0)
    except BaseException:
        os.close(descriptor)
        raise


if os.name == "nt":
    import msvcrt

    def _try_kernel_lock(handle: BinaryIO) -> bool:
        # Windows denies reads that overlap a locked byte.  Keep the kernel
        # sentinel beyond the bounded JSON area so contenders can still read
        # holder metadata without weakening exclusion.
        handle.seek(_WINDOWS_LOCK_BYTE_OFFSET)
        try:
            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        except OSError as exc:
            if exc.errno in {errno.EACCES, errno.EAGAIN}:
                return False
            raise
        return True

    def _release_kernel_lock(handle: BinaryIO) -> None:
        handle.seek(_WINDOWS_LOCK_BYTE_OFFSET)
        msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)

else:
    import fcntl

    def _try_kernel_lock(handle: BinaryIO) -> bool:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            if exc.errno in {errno.EACCES, errno.EAGAIN}:
                return False
            raise
        return True

    def _release_kernel_lock(handle: BinaryIO) -> None:
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _validate_metadata(value: object) -> bool:
    if not isinstance(value, dict):
        return False
    if value.get("schema_version") != _METADATA_SCHEMA_VERSION:
        return False
    if value.get("state") not in {"held", "released"}:
        return False
    if not isinstance(value.get("instance_id"), str):
        return False
    if not isinstance(value.get("pid"), int):
        return False
    if not isinstance(value.get("hostname"), str):
        return False
    if not isinstance(value.get("acquired_at"), str):
        return False
    return True


def _read_metadata(handle: BinaryIO) -> tuple[dict[str, Any] | None, bool, str]:
    try:
        handle.seek(0)
        raw = handle.read(_MAX_METADATA_BYTES + 1)
    except OSError as exc:
        return None, False, f"metadata read failed: {type(exc).__name__}"

    if not raw:
        return None, False, "lock file has no metadata"
    if len(raw) > _MAX_METADATA_BYTES:
        return None, False, "lock metadata exceeds the safety limit"
    try:
        parsed = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return None, False, "lock metadata is incomplete or malformed"
    if not _validate_metadata(parsed):
        return None, False, "lock metadata does not match the expected schema"
    return parsed, True, "lock metadata is valid"


def _write_metadata(handle: BinaryIO, metadata: Mapping[str, Any]) -> None:
    encoded = (json.dumps(metadata, sort_keys=True, separators=(",", ":")) + "\n").encode(
        "utf-8"
    )
    if len(encoded) > _MAX_METADATA_BYTES:
        raise ExecutionLockError("Execution lock metadata exceeds the safety limit.")
    handle.seek(0)
    handle.truncate(0)
    handle.write(encoded)
    handle.flush()
    os.fsync(handle.fileno())


def _safe_previous_holder(metadata: Mapping[str, Any]) -> dict[str, Any]:
    """Retain only known, non-sensitive fields from abandoned metadata."""

    fields = ("instance_id", "pid", "hostname", "acquired_at", "state")
    return {field: metadata[field] for field in fields if field in metadata}


class ExecutionLock:
    """Nonblocking, process-exclusive lock for submission-capable automation."""

    def __init__(self, path: str | os.PathLike[str] | None = None) -> None:
        self.path = _canonical_path(path)
        self._handle: BinaryIO | None = None
        self._owner_pid: int | None = None
        self._metadata: dict[str, Any] | None = None

    @property
    def held(self) -> bool:
        """Whether this object is the registered holder in this process."""

        return (
            self._handle is not None
            and self._owner_pid == os.getpid()
            and _HELD_LOCKS.get(self.path) is self
        )

    @property
    def metadata(self) -> Mapping[str, Any] | None:
        """Return a defensive copy of current holder metadata."""

        return dict(self._metadata) if self._metadata is not None else None

    def acquire(self) -> "ExecutionLock":
        """Acquire immediately or raise :class:`LockContendedError`.

        Existing metadata is never trusted as proof of ownership.  It is read
        and, when stale, summarized only after the kernel lock is acquired.
        """

        with _REGISTRY_GUARD:
            if self.held:
                return self

            registered = _HELD_LOCKS.get(self.path)
            if registered is not None and registered.held:
                raise LockContendedError(_status_for_registered_holder(registered))

            if self._handle is not None:
                raise LockOwnershipError(
                    "ExecutionLock cannot be reacquired from a different process."
                )

            try:
                handle = _open_lock_file(self.path, create=True)
            except OSError as exc:
                raise ExecutionLockError(
                    f"Unable to open execution lock file {self.path}: {type(exc).__name__}."
                ) from exc

            try:
                acquired = _try_kernel_lock(handle)
            except OSError as exc:
                handle.close()
                raise ExecutionLockError(
                    f"Unable to acquire the kernel execution lock: {type(exc).__name__}."
                ) from exc

            if not acquired:
                metadata, valid, detail = _read_metadata(handle)
                handle.close()
                raise LockContendedError(
                    LockStatus(
                        path=self.path,
                        state=LockState.HELD,
                        held_by_current_process=False,
                        file_exists=True,
                        metadata=metadata,
                        metadata_valid=valid,
                        detail=detail,
                    )
                )

            try:
                previous, previous_valid, previous_detail = _read_metadata(handle)
                acquired_at = _utc_now()
                metadata: dict[str, Any] = {
                    "schema_version": _METADATA_SCHEMA_VERSION,
                    "state": "held",
                    "instance_id": str(uuid.uuid4()),
                    "pid": os.getpid(),
                    "hostname": socket.gethostname(),
                    "acquired_at": acquired_at,
                }
                if previous_valid and previous is not None and previous.get("state") == "held":
                    metadata["stale_recovery"] = {
                        "observed_at": acquired_at,
                        "reason": "kernel_lock_was_free_despite_held_metadata",
                        "previous_holder": _safe_previous_holder(previous),
                    }
                elif previous is None and previous_detail != "lock file has no metadata":
                    metadata["stale_recovery"] = {
                        "observed_at": acquired_at,
                        "reason": "invalid_previous_metadata_replaced_after_kernel_acquisition",
                        "previous_metadata_detail": previous_detail,
                    }
                _write_metadata(handle, metadata)
            except BaseException:
                try:
                    _release_kernel_lock(handle)
                finally:
                    handle.close()
                raise

            self._handle = handle
            self._owner_pid = os.getpid()
            self._metadata = metadata
            _HELD_LOCKS[self.path] = self
            return self

    def release(self) -> None:
        """Write release metadata, unlock explicitly, and close the handle."""

        with _REGISTRY_GUARD:
            if self._handle is None:
                return
            if self._owner_pid != os.getpid() or _HELD_LOCKS.get(self.path) is not self:
                raise LockOwnershipError(
                    "This process is not the registered owner of the execution lock."
                )

            handle = self._handle
            failure: BaseException | None = None
            released_metadata = dict(self._metadata or {})
            released_metadata["state"] = "released"
            released_metadata["released_at"] = _utc_now()
            try:
                _write_metadata(handle, released_metadata)
            except BaseException as exc:
                failure = exc
            try:
                _release_kernel_lock(handle)
            except BaseException as exc:
                if failure is None:
                    failure = exc
            finally:
                handle.close()
                _HELD_LOCKS.pop(self.path, None)
                self._handle = None
                self._owner_pid = None
                self._metadata = released_metadata

            if failure is not None:
                raise ExecutionLockError(
                    "Execution lock released, but its audit update failed: "
                    f"{type(failure).__name__}."
                ) from failure

    def __enter__(self) -> "ExecutionLock":
        return self.acquire()

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self.release()


def _status_for_registered_holder(lock: ExecutionLock) -> LockStatus:
    metadata = lock.metadata
    return LockStatus(
        path=lock.path,
        state=LockState.HELD,
        held_by_current_process=True,
        file_exists=lock.path.exists(),
        metadata=metadata,
        metadata_valid=metadata is not None and _validate_metadata(dict(metadata)),
        detail="kernel lock is registered to this process",
    )


def current_process_holds_lock(
    path: str | os.PathLike[str] | None = None,
) -> bool:
    """Return true only for a live holder registered in this process."""

    canonical = _canonical_path(path)
    with _REGISTRY_GUARD:
        holder = _HELD_LOCKS.get(canonical)
        return holder is not None and holder.held


def assert_current_process_holds_lock(
    path: str | os.PathLike[str] | None = None,
) -> None:
    """Fail closed when the central submission gate lacks lock ownership."""

    canonical = _canonical_path(path)
    if not current_process_holds_lock(canonical):
        raise LockOwnershipError(
            f"Current process does not hold the automation execution lock: {canonical}"
        )


def inspect_lock(path: str | os.PathLike[str] | None = None) -> LockStatus:
    """Probe lock state without changing lock metadata.

    ``UNKNOWN`` is returned for any inspection error so callers cannot mistake
    an unreadable lock for permission to submit.
    """

    canonical = _canonical_path(path)
    with _REGISTRY_GUARD:
        registered = _HELD_LOCKS.get(canonical)
        if registered is not None and registered.held:
            return _status_for_registered_holder(registered)

    try:
        handle = _open_lock_file(canonical, create=False)
    except FileNotFoundError:
        return LockStatus(
            path=canonical,
            state=LockState.UNLOCKED,
            held_by_current_process=False,
            file_exists=False,
            detail="lock file does not exist and no holder was observed",
        )
    except (OSError, ExecutionLockError) as exc:
        return LockStatus(
            path=canonical,
            state=LockState.UNKNOWN,
            held_by_current_process=False,
            file_exists=canonical.exists(),
            detail=f"lock probe failed: {type(exc).__name__}",
        )

    try:
        acquired = _try_kernel_lock(handle)
    except OSError as exc:
        handle.close()
        return LockStatus(
            path=canonical,
            state=LockState.UNKNOWN,
            held_by_current_process=False,
            file_exists=True,
            detail=f"kernel lock probe failed: {type(exc).__name__}",
        )
    metadata, valid, detail = _read_metadata(handle)
    try:
        if acquired:
            _release_kernel_lock(handle)
    except OSError as exc:
        handle.close()
        return LockStatus(
            path=canonical,
            state=LockState.UNKNOWN,
            held_by_current_process=False,
            file_exists=True,
            metadata=metadata,
            metadata_valid=valid,
            detail=f"kernel lock probe cleanup failed: {type(exc).__name__}",
        )
    handle.close()

    return LockStatus(
        path=canonical,
        state=LockState.UNLOCKED if acquired else LockState.HELD,
        held_by_current_process=False,
        file_exists=True,
        metadata=metadata,
        metadata_valid=valid,
        detail=detail,
    )


probe_lock = inspect_lock


def _reset_registry_after_fork() -> None:
    """A forked child must never inherit process-local ownership authority."""

    global _REGISTRY_GUARD
    inherited = tuple(_HELD_LOCKS.values())
    _HELD_LOCKS.clear()
    for lock in inherited:
        handle = lock._handle
        if handle is not None:
            try:
                handle.close()
            except OSError:
                pass
        lock._handle = None
        lock._owner_pid = None
    _REGISTRY_GUARD = threading.RLock()


if hasattr(os, "register_at_fork"):
    os.register_at_fork(after_in_child=_reset_registry_after_fork)


__all__ = [
    "DEFAULT_EXECUTION_LOCK_PATH",
    "ExecutionLock",
    "ExecutionLockError",
    "LockContendedError",
    "LockOwnershipError",
    "LockState",
    "LockStatus",
    "assert_current_process_holds_lock",
    "current_process_holds_lock",
    "inspect_lock",
    "probe_lock",
]

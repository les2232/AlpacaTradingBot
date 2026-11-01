"""Minimal pytest recorder used by :mod:`tradeos.readiness`.

The plugin writes only to the isolated result path supplied by the readiness
runner.  It does not create a readiness attestation itself; the parent process
performs all repository binding and fail-closed validation.
"""

from __future__ import annotations

import json
import os
from pathlib import Path, PurePosixPath
from typing import Any
from uuid import uuid4


RESULT_PATH_ENV = "TRADEOS_READINESS_RESULT_PATH"

_collected_node_ids: set[str] = set()
_deselected_node_ids: set[str] = set()
_passed_node_ids: set[str] = set()
_failed_node_ids: set[str] = set()
_error_node_ids: set[str] = set()
_skipped_node_ids: set[str] = set()
_collection_error_count = 0


def _normalized_node_id(value: object) -> str:
    return str(value).replace("\\", "/")


def _test_file(node_id: str) -> str:
    return PurePosixPath(node_id.split("::", 1)[0]).as_posix()


def _atomic_write(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f".{path.name}.{uuid4().hex}.tmp"
    try:
        with temporary.open("x", encoding="utf-8", newline="\n") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def pytest_collection_finish(session: Any) -> None:
    _collected_node_ids.update(
        _normalized_node_id(item.nodeid) for item in session.items
    )


def pytest_deselected(items: list[Any]) -> None:
    _deselected_node_ids.update(_normalized_node_id(item.nodeid) for item in items)


def pytest_collectreport(report: Any) -> None:
    global _collection_error_count
    if bool(getattr(report, "failed", False)):
        _collection_error_count += 1


def pytest_runtest_logreport(report: Any) -> None:
    node_id = _normalized_node_id(report.nodeid)
    if bool(report.skipped):
        _skipped_node_ids.add(node_id)
        return
    if bool(report.failed):
        if report.when == "call":
            _failed_node_ids.add(node_id)
        else:
            _error_node_ids.add(node_id)
        return
    if report.when == "call" and bool(report.passed):
        _passed_node_ids.add(node_id)


def pytest_sessionfinish(session: Any, exitstatus: int) -> None:
    raw_path = os.environ.get(RESULT_PATH_ENV, "").strip()
    if not raw_path:
        return
    all_collected_node_ids = sorted(
        _collected_node_ids | _deselected_node_ids
    )
    test_files = sorted({_test_file(node_id) for node_id in all_collected_node_ids})
    payload: dict[str, object] = {
        "exitstatus": int(exitstatus),
        "collected_node_ids": all_collected_node_ids,
        "test_files": test_files,
        "collected": len(all_collected_node_ids),
        "passed": len(_passed_node_ids),
        "failed": len(_failed_node_ids),
        "errors": len(_error_node_ids) + _collection_error_count,
        "skipped": len(_skipped_node_ids),
        "deselected": len(_deselected_node_ids),
    }
    _atomic_write(Path(raw_path), payload)

"""Generate and validate fail-closed full-suite readiness attestations.

The attestation runner deliberately has no test-selection arguments.  It runs
the repository's complete ``tests/`` tree with third-party pytest plugin
autoloading disabled, then binds the successful result to the exact worktree,
HEAD, and runtime configuration that were tested.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path, PurePosixPath
import re
import subprocess
import sys
import tempfile
from typing import Mapping, Sequence
from uuid import uuid4


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SCHEMA_VERSION = 1
SUITE_ID = "tradeos-offline-full-v1"
DEFAULT_MAX_AGE = timedelta(hours=24)
DEFAULT_EVIDENCE_DIR = REPOSITORY_ROOT / "logs" / "readiness"
PYTEST_RESULT_ENV = "TRADEOS_READINESS_RESULT_PATH"
FIXED_COMMAND: tuple[str, ...] = (
    sys.executable,
    "-m",
    "pytest",
    "-q",
    "--tb=short",
    "-p",
    "tradeos.readiness_pytest_plugin",
    "tests",
)

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_HEAD_SHA_RE = re.compile(r"^[0-9a-f]{40}$")
_UTC_TIMESTAMP_RE = re.compile(
    r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{6}Z$"
)
_EVIDENCE_NAME_RE = re.compile(
    r"^readiness\.(?P<timestamp>\d{8}T\d{6}\.\d{6}Z)\.json$"
)
_TRACKED_TEST_RE = re.compile(r"^tests/test_[^/]+\.py$")

_EVIDENCE_KEYS = frozenset(
    {
        "schema_version",
        "suite_id",
        "command",
        "plugin_autoload_disabled",
        "status",
        "completed_at_utc",
        "head_sha",
        "source_sha256",
        "config_path",
        "config_sha256",
        "collected_node_ids_sha256",
        "tracked_test_files",
        "collected_test_files",
        "collected",
        "passed",
        "failed",
        "errors",
        "skipped",
        "deselected",
        "payload_sha256",
    }
)

_PLUGIN_RESULT_KEYS = frozenset(
    {
        "exitstatus",
        "collected_node_ids",
        "test_files",
        "collected",
        "passed",
        "failed",
        "errors",
        "skipped",
        "deselected",
    }
)


class ReadinessError(RuntimeError):
    """Base error for readiness generation and validation."""


class ReadinessValidationError(ReadinessError):
    """Raised when persisted evidence is not positively valid."""


@dataclass(frozen=True)
class RepositoryBinding:
    """Current repository and configuration identity."""

    head_sha: str
    source_sha256: str
    config_path: str
    config_sha256: str
    tracked_test_files: tuple[str, ...]


@dataclass(frozen=True)
class ValidatedReadinessEvidence:
    """A readiness artifact that passed every current-state check."""

    path: Path
    payload: Mapping[str, object]


@dataclass(frozen=True)
class EvidenceSelection:
    """Newest valid evidence plus reasons rejected candidates were ignored."""

    evidence: ValidatedReadinessEvidence | None
    rejected: tuple[tuple[Path, str], ...]


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _require_aware_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        raise ReadinessError("readiness timestamps must be timezone-aware")
    return value.astimezone(timezone.utc)


def _format_utc(value: datetime) -> str:
    return _require_aware_utc(value).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def _parse_utc(value: object) -> datetime:
    if type(value) is not str or not _UTC_TIMESTAMP_RE.fullmatch(value):
        raise ReadinessValidationError(
            "completed_at_utc must be a canonical UTC timestamp"
        )
    try:
        parsed = datetime.strptime(value, "%Y-%m-%dT%H:%M:%S.%fZ")
    except ValueError as exc:
        raise ReadinessValidationError("completed_at_utc is invalid") from exc
    return parsed.replace(tzinfo=timezone.utc)


def _validate_evidence_filename(
    path: Path, payload: Mapping[str, object]
) -> None:
    match = _EVIDENCE_NAME_RE.fullmatch(path.name)
    if match is None:
        raise ReadinessValidationError("readiness evidence filename is invalid")
    try:
        filename_time = datetime.strptime(
            match.group("timestamp"), "%Y%m%dT%H%M%S.%fZ"
        ).replace(tzinfo=timezone.utc)
    except ValueError as exc:
        raise ReadinessValidationError(
            "readiness evidence filename timestamp is invalid"
        ) from exc
    if filename_time != _parse_utc(payload.get("completed_at_utc")):
        raise ReadinessValidationError(
            "readiness filename and completion timestamp do not match"
        )


def _canonical_json_bytes(payload: Mapping[str, object]) -> bytes:
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")


def _payload_sha256(payload: Mapping[str, object]) -> str:
    without_digest = {
        key: value for key, value in payload.items() if key != "payload_sha256"
    }
    return hashlib.sha256(_canonical_json_bytes(without_digest)).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _digest_string_list(values: Sequence[str]) -> str:
    return hashlib.sha256(
        json.dumps(
            list(values),
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def _run_git(repo_root: Path, *arguments: str) -> bytes:
    completed = subprocess.run(
        ["git", *arguments],
        cwd=repo_root,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if completed.returncode != 0:
        detail = completed.stderr.decode("utf-8", errors="replace").strip()
        raise ReadinessError(f"git {' '.join(arguments)} failed: {detail}")
    return completed.stdout


def _head_sha(repo_root: Path) -> str:
    value = _run_git(repo_root, "rev-parse", "--verify", "HEAD").decode(
        "ascii", errors="strict"
    ).strip().lower()
    if not _HEAD_SHA_RE.fullmatch(value):
        raise ReadinessError("git returned an invalid HEAD SHA")
    return value


def _tracked_paths(repo_root: Path) -> tuple[str, ...]:
    # Bind the index plus non-ignored untracked files.  New implementation and
    # test files must be covered before an operator stages or commits them.
    # Ignored machine-local files (including the paper-shadow config) remain
    # outside this set and are bound separately by config_path/config_sha256.
    raw = _run_git(
        repo_root,
        "ls-files",
        "-z",
        "--cached",
        "--others",
        "--exclude-standard",
        "--",
    )
    paths: list[str] = []
    for encoded in raw.split(b"\0"):
        if not encoded:
            continue
        relative = encoded.decode("utf-8", errors="surrogateescape").replace(
            "\\", "/"
        )
        pure = PurePosixPath(relative)
        if pure.is_absolute() or ".." in pure.parts:
            raise ReadinessError("git returned a non-canonical versionable path")
        paths.append(pure.as_posix())
    if len(paths) != len(set(paths)):
        raise ReadinessError("git returned duplicate versionable paths")
    return tuple(sorted(paths))


def _tracked_source_sha256(repo_root: Path, tracked_paths: Sequence[str]) -> str:
    digest = hashlib.sha256(b"tradeos-tracked-worktree-v1\0")
    for relative in tracked_paths:
        path = repo_root / PurePosixPath(relative)
        if not path.is_file():
            raise ReadinessError(f"versionable file is missing: {relative}")
        path_bytes = relative.encode("utf-8", errors="surrogateescape")
        contents = path.read_bytes()
        digest.update(len(path_bytes).to_bytes(8, "big"))
        digest.update(path_bytes)
        digest.update(len(contents).to_bytes(8, "big"))
        digest.update(contents)
    return digest.hexdigest()


def _canonical_config_path(repo_root: Path, config_path: str | Path) -> Path:
    requested = Path(config_path)
    if not requested.is_absolute():
        requested = repo_root / requested
    try:
        resolved = requested.resolve(strict=True)
    except OSError as exc:
        raise ReadinessError(f"runtime config does not exist: {requested}") from exc
    if not resolved.is_file():
        raise ReadinessError(f"runtime config is not a file: {resolved}")
    try:
        resolved.relative_to(repo_root.resolve(strict=True))
    except ValueError as exc:
        raise ReadinessError("runtime config must remain inside the repository") from exc
    return resolved


def _capture_repository_binding(
    repo_root: Path, config_path: str | Path
) -> RepositoryBinding:
    root = repo_root.resolve(strict=True)
    tracked_paths = _tracked_paths(root)
    tracked_tests = tuple(path for path in tracked_paths if _TRACKED_TEST_RE.fullmatch(path))
    if not tracked_tests:
        raise ReadinessError("no tracked tests/test_*.py files were found")
    canonical_config = _canonical_config_path(root, config_path)
    return RepositoryBinding(
        head_sha=_head_sha(root),
        source_sha256=_tracked_source_sha256(root, tracked_paths),
        config_path=str(canonical_config),
        config_sha256=_file_sha256(canonical_config),
        tracked_test_files=tracked_tests,
    )


def _atomic_write_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f".{path.name}.{uuid4().hex}.tmp"
    try:
        with temporary.open("x", encoding="utf-8", newline="\n") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, ensure_ascii=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def _invoke_pytest(repo_root: Path, result_path: Path, temp_dir: Path) -> int:
    environment = os.environ.copy()
    environment["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = "1"
    # Ambient selectors or explicitly requested plugins would make the command
    # look fixed while changing what actually ran.
    environment["PYTEST_ADDOPTS"] = ""
    environment["PYTEST_PLUGINS"] = ""
    environment[PYTEST_RESULT_ENV] = str(result_path)
    for name in ("TEMP", "TMP", "TMPDIR"):
        environment[name] = str(temp_dir)
    completed = subprocess.run(
        list(FIXED_COMMAND),
        cwd=repo_root,
        env=environment,
        check=False,
    )
    return int(completed.returncode)


def _load_json_object(path: Path, *, label: str) -> dict[str, object]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ReadinessValidationError(f"{label} is missing") from exc
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ReadinessValidationError(f"{label} is unreadable or malformed") from exc
    if type(payload) is not dict:
        raise ReadinessValidationError(f"{label} must be a JSON object")
    return payload


def _strict_nonnegative_int(payload: Mapping[str, object], key: str) -> int:
    value = payload.get(key)
    if type(value) is not int or value < 0:
        raise ReadinessValidationError(f"{key} must be a non-negative integer")
    return value


def _strict_string_list(payload: Mapping[str, object], key: str) -> tuple[str, ...]:
    value = payload.get(key)
    if type(value) is not list or any(
        type(item) is not str or not item for item in value
    ):
        raise ReadinessValidationError(f"{key} must be a list of non-empty strings")
    result = tuple(value)
    if result != tuple(sorted(set(result))):
        raise ReadinessValidationError(f"{key} must be sorted and unique")
    return result


def _test_file_from_node_id(node_id: str) -> str:
    path = node_id.split("::", 1)[0].replace("\\", "/")
    return PurePosixPath(path).as_posix()


def _validate_plugin_result(
    payload: Mapping[str, object],
    *,
    returncode: int,
    binding: RepositoryBinding,
) -> tuple[str, ...]:
    if set(payload) != _PLUGIN_RESULT_KEYS:
        raise ReadinessValidationError("pytest result has an invalid schema")
    exitstatus = _strict_nonnegative_int(payload, "exitstatus")
    if exitstatus != returncode or exitstatus != 0:
        raise ReadinessValidationError("pytest result exit status is not successful")

    node_ids = _strict_string_list(payload, "collected_node_ids")
    test_files = _strict_string_list(payload, "test_files")
    derived_files = tuple(sorted({_test_file_from_node_id(item) for item in node_ids}))
    if test_files != derived_files:
        raise ReadinessValidationError("pytest result test file metadata is inconsistent")

    counts = {
        key: _strict_nonnegative_int(payload, key)
        for key in (
            "collected",
            "passed",
            "failed",
            "errors",
            "skipped",
            "deselected",
        )
    }
    if counts["collected"] != len(node_ids):
        raise ReadinessValidationError("pytest collected count does not match node IDs")
    if counts["collected"] <= 0 or counts["passed"] != counts["collected"]:
        raise ReadinessValidationError("the complete collected suite did not pass")
    if any(counts[key] != 0 for key in ("failed", "errors", "skipped", "deselected")):
        raise ReadinessValidationError(
            "failed, errored, skipped, or deselected tests make readiness invalid"
        )
    if test_files != binding.tracked_test_files:
        raise ReadinessValidationError(
            "not every tracked tests/test_*.py file was represented"
        )
    return node_ids


def _build_evidence_payload(
    *,
    binding: RepositoryBinding,
    plugin_result: Mapping[str, object],
    node_ids: Sequence[str],
    completed_at_utc: datetime,
) -> dict[str, object]:
    payload: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "suite_id": SUITE_ID,
        "command": list(FIXED_COMMAND),
        "plugin_autoload_disabled": True,
        "status": "passed",
        "completed_at_utc": _format_utc(completed_at_utc),
        "head_sha": binding.head_sha,
        "source_sha256": binding.source_sha256,
        "config_path": binding.config_path,
        "config_sha256": binding.config_sha256,
        "collected_node_ids_sha256": _digest_string_list(node_ids),
        "tracked_test_files": list(binding.tracked_test_files),
        "collected_test_files": list(plugin_result["test_files"]),
        "collected": plugin_result["collected"],
        "passed": plugin_result["passed"],
        "failed": plugin_result["failed"],
        "errors": plugin_result["errors"],
        "skipped": plugin_result["skipped"],
        "deselected": plugin_result["deselected"],
    }
    payload["payload_sha256"] = _payload_sha256(payload)
    return payload


def _validate_payload(
    payload: Mapping[str, object],
    *,
    binding: RepositoryBinding,
    now_utc: datetime,
    max_age: timedelta,
) -> None:
    if set(payload) != _EVIDENCE_KEYS:
        raise ReadinessValidationError("readiness evidence has an invalid schema")
    if type(payload.get("schema_version")) is not int or payload["schema_version"] != SCHEMA_VERSION:
        raise ReadinessValidationError("schema_version is invalid")
    if type(payload.get("plugin_autoload_disabled")) is not bool:
        raise ReadinessValidationError("plugin_autoload_disabled must be a boolean")
    if payload["plugin_autoload_disabled"] is not True:
        raise ReadinessValidationError("pytest plugin autoload was not disabled")
    if type(payload.get("suite_id")) is not str or payload["suite_id"] != SUITE_ID:
        raise ReadinessValidationError("suite_id is invalid")
    command = payload.get("command")
    if type(command) is not list or any(type(item) is not str for item in command):
        raise ReadinessValidationError("command must be a list of strings")
    if command != list(FIXED_COMMAND):
        raise ReadinessValidationError("readiness command was not the fixed full-suite command")
    if type(payload.get("status")) is not str or payload["status"] != "passed":
        raise ReadinessValidationError("readiness status is not passed")

    for key, pattern in (
        ("head_sha", _HEAD_SHA_RE),
        ("source_sha256", _SHA256_RE),
        ("config_sha256", _SHA256_RE),
        ("collected_node_ids_sha256", _SHA256_RE),
        ("payload_sha256", _SHA256_RE),
    ):
        value = payload.get(key)
        if type(value) is not str or not pattern.fullmatch(value):
            raise ReadinessValidationError(f"{key} is invalid")
    if type(payload.get("config_path")) is not str or not payload["config_path"]:
        raise ReadinessValidationError("config_path is invalid")

    tracked_files = _strict_string_list(payload, "tracked_test_files")
    collected_files = _strict_string_list(payload, "collected_test_files")
    counts = {
        key: _strict_nonnegative_int(payload, key)
        for key in (
            "collected",
            "passed",
            "failed",
            "errors",
            "skipped",
            "deselected",
        )
    }
    if counts["collected"] <= 0 or counts["passed"] != counts["collected"]:
        raise ReadinessValidationError("attested suite was not completely passed")
    if any(counts[key] != 0 for key in ("failed", "errors", "skipped", "deselected")):
        raise ReadinessValidationError("attested suite was partial or non-passing")
    if tracked_files != collected_files:
        raise ReadinessValidationError("attested test file sets are inconsistent")

    expected_payload_sha = _payload_sha256(payload)
    if payload["payload_sha256"] != expected_payload_sha:
        raise ReadinessValidationError("payload_sha256 does not match the evidence")

    completed_at = _parse_utc(payload.get("completed_at_utc"))
    current_time = _require_aware_utc(now_utc)
    if max_age <= timedelta(0):
        raise ReadinessError("max_age must be positive")
    if completed_at > current_time:
        raise ReadinessValidationError("readiness evidence is from the future")
    if current_time - completed_at > max_age:
        raise ReadinessValidationError("readiness evidence is stale")

    if payload["head_sha"] != binding.head_sha:
        raise ReadinessValidationError("HEAD SHA does not match current HEAD")
    if payload["source_sha256"] != binding.source_sha256:
        raise ReadinessValidationError("source digest does not match the current worktree")
    if payload["config_path"] != binding.config_path:
        raise ReadinessValidationError("config path does not match the requested config")
    if payload["config_sha256"] != binding.config_sha256:
        raise ReadinessValidationError("config digest does not match the current config")
    if tracked_files != binding.tracked_test_files:
        raise ReadinessValidationError("tracked test files changed after the attestation")


def validate_readiness_evidence(
    path: str | Path,
    *,
    config_path: str | Path,
    repo_root: str | Path = REPOSITORY_ROOT,
    now_utc: datetime | None = None,
    max_age: timedelta = DEFAULT_MAX_AGE,
) -> ValidatedReadinessEvidence:
    """Validate one artifact against the current repository and config."""

    root = Path(repo_root).resolve(strict=True)
    unresolved_path = Path(path)
    if unresolved_path.is_symlink():
        raise ReadinessValidationError("readiness evidence cannot be a symlink")
    try:
        evidence_path = unresolved_path.resolve(strict=True)
    except OSError as exc:
        raise ReadinessValidationError("readiness evidence is missing") from exc
    payload = _load_json_object(evidence_path, label="readiness evidence")
    _validate_evidence_filename(evidence_path, payload)
    binding = _capture_repository_binding(root, config_path)
    _validate_payload(
        payload,
        binding=binding,
        now_utc=now_utc or _utc_now(),
        max_age=max_age,
    )
    return ValidatedReadinessEvidence(path=evidence_path, payload=dict(payload))


def select_latest_valid_evidence(
    evidence_dir: str | Path,
    *,
    config_path: str | Path,
    repo_root: str | Path = REPOSITORY_ROOT,
    now_utc: datetime | None = None,
    max_age: timedelta = DEFAULT_MAX_AGE,
) -> EvidenceSelection:
    """Return the newest valid timestamped artifact, rejecting invalid newer ones."""

    root = Path(repo_root).resolve(strict=True)
    directory = Path(evidence_dir)
    if not directory.is_absolute():
        directory = root / directory
    if not directory.exists():
        return EvidenceSelection(evidence=None, rejected=())
    if not directory.is_dir():
        raise ReadinessError(f"readiness evidence path is not a directory: {directory}")

    binding = _capture_repository_binding(root, config_path)
    current_time = now_utc or _utc_now()
    candidates = sorted(
        (
            path
            for path in directory.iterdir()
            if path.is_file()
            and not path.is_symlink()
            and _EVIDENCE_NAME_RE.fullmatch(path.name)
        ),
        key=lambda item: item.name,
        reverse=True,
    )
    rejected: list[tuple[Path, str]] = []
    for candidate in candidates:
        try:
            payload = _load_json_object(candidate, label="readiness evidence")
            _validate_evidence_filename(candidate, payload)
            _validate_payload(
                payload,
                binding=binding,
                now_utc=current_time,
                max_age=max_age,
            )
        except ReadinessValidationError as exc:
            rejected.append((candidate, str(exc)))
            continue
        return EvidenceSelection(
            evidence=ValidatedReadinessEvidence(
                path=candidate.resolve(strict=True), payload=dict(payload)
            ),
            rejected=tuple(rejected),
        )
    return EvidenceSelection(evidence=None, rejected=tuple(rejected))


def run_full_suite_readiness(
    *,
    config_path: str | Path,
    repo_root: str | Path = REPOSITORY_ROOT,
    evidence_dir: str | Path | None = None,
    completed_at_utc: datetime | None = None,
) -> ValidatedReadinessEvidence:
    """Run the fixed complete suite and atomically persist passing evidence."""

    root = Path(repo_root).resolve(strict=True)
    destination = (
        Path(evidence_dir)
        if evidence_dir is not None
        else root / "logs" / "readiness"
    )
    if not destination.is_absolute():
        destination = root / destination

    temp_parent = root / ".tmp"
    temp_parent.mkdir(parents=True, exist_ok=True)
    before = _capture_repository_binding(root, config_path)
    with tempfile.TemporaryDirectory(prefix="readiness-", dir=temp_parent) as raw_temp:
        temp_dir = Path(raw_temp)
        result_path = temp_dir / "pytest-result.json"
        returncode = _invoke_pytest(root, result_path, temp_dir)
        if returncode != 0:
            raise ReadinessError(
                f"fixed readiness suite failed with pytest exit status {returncode}"
            )
        plugin_result = _load_json_object(result_path, label="pytest readiness result")

    after = _capture_repository_binding(root, config_path)
    if before != after:
        raise ReadinessError("repository or config changed while the suite was running")
    node_ids = _validate_plugin_result(
        plugin_result,
        returncode=returncode,
        binding=after,
    )
    completed = _require_aware_utc(completed_at_utc or _utc_now())
    validation_time = _utc_now()
    if completed > validation_time:
        raise ReadinessError("completion timestamp cannot be in the future")
    payload = _build_evidence_payload(
        binding=after,
        plugin_result=plugin_result,
        node_ids=node_ids,
        completed_at_utc=completed,
    )
    filename = f"readiness.{completed.strftime('%Y%m%dT%H%M%S.%fZ')}.json"
    evidence_path = destination / filename
    _atomic_write_json(evidence_path, payload)
    try:
        return validate_readiness_evidence(
            evidence_path,
            config_path=config_path,
            repo_root=root,
            now_utc=validation_time,
        )
    except Exception:
        try:
            evidence_path.unlink()
        except FileNotFoundError:
            pass
        raise


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the fixed offline suite and emit bound readiness evidence."
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Repository-local runtime config whose exact bytes must be attested.",
    )
    parser.add_argument(
        "--evidence-dir",
        default=None,
        help="Optional repository-local or absolute evidence directory.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        evidence = run_full_suite_readiness(
            config_path=args.config,
            evidence_dir=args.evidence_dir,
        )
    except ReadinessError as exc:
        print(f"READINESS_BLOCKED: {exc}", file=sys.stderr)
        return 1
    print(f"READINESS_PASSED: {evidence.path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "DEFAULT_EVIDENCE_DIR",
    "DEFAULT_MAX_AGE",
    "EvidenceSelection",
    "FIXED_COMMAND",
    "ReadinessError",
    "ReadinessValidationError",
    "RepositoryBinding",
    "SCHEMA_VERSION",
    "SUITE_ID",
    "ValidatedReadinessEvidence",
    "run_full_suite_readiness",
    "select_latest_valid_evidence",
    "validate_readiness_evidence",
]

from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from tradeos import readiness


NOW = datetime(2026, 8, 17, 18, 0, tzinfo=timezone.utc)
NODE_IDS = (
    "tests/test_alpha.py::test_alpha",
    "tests/test_beta.py::test_beta",
)
TEST_FILES = ("tests/test_alpha.py", "tests/test_beta.py")


def _binding(tmp_path: Path, **updates: object) -> readiness.RepositoryBinding:
    values: dict[str, object] = {
        "head_sha": "a" * 40,
        "source_sha256": "b" * 64,
        "config_path": str((tmp_path / "config" / "paper.json").resolve()),
        "config_sha256": "c" * 64,
        "tracked_test_files": TEST_FILES,
    }
    values.update(updates)
    return readiness.RepositoryBinding(**values)  # type: ignore[arg-type]


def _plugin_result(**updates: object) -> dict[str, object]:
    values: dict[str, object] = {
        "exitstatus": 0,
        "collected_node_ids": list(NODE_IDS),
        "test_files": list(TEST_FILES),
        "collected": 2,
        "passed": 2,
        "failed": 0,
        "errors": 0,
        "skipped": 0,
        "deselected": 0,
    }
    values.update(updates)
    return values


def _evidence_payload(
    tmp_path: Path,
    *,
    binding: readiness.RepositoryBinding | None = None,
    completed_at: datetime = NOW,
) -> dict[str, object]:
    current_binding = binding or _binding(tmp_path)
    plugin_result = _plugin_result()
    return readiness._build_evidence_payload(
        binding=current_binding,
        plugin_result=plugin_result,
        node_ids=NODE_IDS,
        completed_at_utc=completed_at,
    )


def _rewrite_digest(payload: dict[str, object]) -> None:
    payload["payload_sha256"] = readiness._payload_sha256(payload)


def _write_payload(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _patch_binding(
    monkeypatch: pytest.MonkeyPatch, binding: readiness.RepositoryBinding
) -> None:
    monkeypatch.setattr(
        readiness,
        "_capture_repository_binding",
        lambda repo_root, config_path: binding,
    )


def test_invoke_pytest_uses_fixed_unfiltered_isolated_command(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    captured: dict[str, object] = {}

    def fake_run(command, *, cwd, env, check):
        captured.update(command=command, cwd=cwd, env=env, check=check)
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(readiness.subprocess, "run", fake_run)
    result_path = tmp_path / "isolated" / "result.json"
    isolated = tmp_path / "isolated"

    assert readiness._invoke_pytest(tmp_path, result_path, isolated) == 0
    assert captured["command"] == list(readiness.FIXED_COMMAND)
    assert captured["cwd"] == tmp_path
    assert captured["check"] is False
    environment = captured["env"]
    assert environment["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] == "1"
    assert environment["PYTEST_ADDOPTS"] == ""
    assert environment["PYTEST_PLUGINS"] == ""
    assert environment[readiness.PYTEST_RESULT_ENV] == str(result_path)
    assert environment["TEMP"] == str(isolated)
    assert environment["TMP"] == str(isolated)
    assert environment["TMPDIR"] == str(isolated)
    assert readiness.FIXED_COMMAND[-1] == "tests"
    assert "-k" not in readiness.FIXED_COMMAND


def test_successful_complete_run_writes_valid_bound_evidence(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    binding = _binding(tmp_path)
    _patch_binding(monkeypatch, binding)

    def fake_invoke(repo_root: Path, result_path: Path, temp_dir: Path) -> int:
        assert temp_dir.parent == tmp_path / ".tmp"
        _write_payload(result_path, _plugin_result())
        return 0

    monkeypatch.setattr(readiness, "_invoke_pytest", fake_invoke)
    monkeypatch.setattr(
        readiness, "_utc_now", lambda: NOW + timedelta(seconds=1)
    )
    evidence_dir = tmp_path / "evidence"

    evidence = readiness.run_full_suite_readiness(
        config_path="config/paper.json",
        repo_root=tmp_path,
        evidence_dir=evidence_dir,
        completed_at_utc=NOW,
    )

    assert evidence.path.parent == evidence_dir
    assert evidence.path.name == "readiness.20260817T180000.000000Z.json"
    assert evidence.payload["status"] == "passed"
    assert evidence.payload["head_sha"] == binding.head_sha
    assert evidence.payload["source_sha256"] == binding.source_sha256
    assert evidence.payload["config_sha256"] == binding.config_sha256
    assert evidence.payload["collected"] == evidence.payload["passed"] == 2
    assert evidence.payload["tracked_test_files"] == list(TEST_FILES)
    assert evidence.payload["collected_test_files"] == list(TEST_FILES)
    assert evidence.payload["command"] == list(readiness.FIXED_COMMAND)
    assert evidence.payload["plugin_autoload_disabled"] is True


def test_failed_pytest_run_never_writes_evidence(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _patch_binding(monkeypatch, _binding(tmp_path))
    monkeypatch.setattr(readiness, "_invoke_pytest", lambda *args: 1)
    evidence_dir = tmp_path / "evidence"

    with pytest.raises(readiness.ReadinessError, match="exit status 1"):
        readiness.run_full_suite_readiness(
            config_path="config/paper.json",
            repo_root=tmp_path,
            evidence_dir=evidence_dir,
            completed_at_utc=NOW,
        )

    assert not list(evidence_dir.glob("readiness.*.json"))


@pytest.mark.parametrize("field", ["failed", "errors", "skipped", "deselected"])
def test_nonzero_failure_skip_or_deselection_never_writes_evidence(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, field: str
) -> None:
    _patch_binding(monkeypatch, _binding(tmp_path))

    def fake_invoke(repo_root: Path, result_path: Path, temp_dir: Path) -> int:
        _write_payload(result_path, _plugin_result(**{field: 1}))
        return 0

    monkeypatch.setattr(readiness, "_invoke_pytest", fake_invoke)
    evidence_dir = tmp_path / "evidence"

    with pytest.raises(readiness.ReadinessValidationError):
        readiness.run_full_suite_readiness(
            config_path="config/paper.json",
            repo_root=tmp_path,
            evidence_dir=evidence_dir,
            completed_at_utc=NOW,
        )

    assert not list(evidence_dir.glob("readiness.*.json"))


def test_missing_tracked_test_file_never_writes_evidence(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    binding = _binding(tmp_path)
    _patch_binding(monkeypatch, binding)

    def fake_invoke(repo_root: Path, result_path: Path, temp_dir: Path) -> int:
        _write_payload(
            result_path,
            _plugin_result(
                collected_node_ids=[NODE_IDS[0]],
                test_files=[TEST_FILES[0]],
                collected=1,
                passed=1,
            ),
        )
        return 0

    monkeypatch.setattr(readiness, "_invoke_pytest", fake_invoke)

    with pytest.raises(
        readiness.ReadinessValidationError,
        match="not every tracked",
    ):
        readiness.run_full_suite_readiness(
            config_path="config/paper.json",
            repo_root=tmp_path,
            evidence_dir=tmp_path / "evidence",
            completed_at_utc=NOW,
        )


def test_repository_change_during_run_prevents_evidence(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    before = _binding(tmp_path)
    after = replace(before, source_sha256="d" * 64)
    bindings = iter((before, after))
    monkeypatch.setattr(
        readiness,
        "_capture_repository_binding",
        lambda repo_root, config_path: next(bindings),
    )

    def fake_invoke(repo_root: Path, result_path: Path, temp_dir: Path) -> int:
        _write_payload(result_path, _plugin_result())
        return 0

    monkeypatch.setattr(readiness, "_invoke_pytest", fake_invoke)
    evidence_dir = tmp_path / "evidence"

    with pytest.raises(readiness.ReadinessError, match="changed while"):
        readiness.run_full_suite_readiness(
            config_path="config/paper.json",
            repo_root=tmp_path,
            evidence_dir=evidence_dir,
            completed_at_utc=NOW,
        )

    assert not list(evidence_dir.glob("readiness.*.json"))


@pytest.mark.parametrize(
    ("field", "invalid_value", "message"),
    [
        ("plugin_autoload_disabled", 1, "must be a boolean"),
        ("collected", True, "non-negative integer"),
    ],
)
def test_validation_rejects_bool_int_type_confusion(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    field: str,
    invalid_value: object,
    message: str,
) -> None:
    binding = _binding(tmp_path)
    _patch_binding(monkeypatch, binding)
    payload = _evidence_payload(tmp_path, binding=binding)
    payload[field] = invalid_value
    _rewrite_digest(payload)
    path = tmp_path / "readiness.20260817T180000.000000Z.json"
    _write_payload(path, payload)

    with pytest.raises(readiness.ReadinessValidationError, match=message):
        readiness.validate_readiness_evidence(
            path,
            config_path="config/paper.json",
            repo_root=tmp_path,
            now_utc=NOW,
        )


@pytest.mark.parametrize(
    ("binding_update", "message"),
    [
        ({"head_sha": "d" * 40}, "HEAD SHA"),
        ({"source_sha256": "d" * 64}, "source digest"),
        ({"config_sha256": "d" * 64}, "config digest"),
        (
            {"config_path": "C:\\different\\paper.json"},
            "config path",
        ),
    ],
)
def test_validation_recomputes_current_head_source_and_config_binding(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    binding_update: dict[str, object],
    message: str,
) -> None:
    attested = _binding(tmp_path)
    current = replace(attested, **binding_update)
    _patch_binding(monkeypatch, current)
    payload = _evidence_payload(tmp_path, binding=attested)
    path = tmp_path / "readiness.20260817T180000.000000Z.json"
    _write_payload(path, payload)

    with pytest.raises(readiness.ReadinessValidationError, match=message):
        readiness.validate_readiness_evidence(
            path,
            config_path="config/paper.json",
            repo_root=tmp_path,
            now_utc=NOW,
        )


@pytest.mark.parametrize(
    ("completed_at", "message"),
    [
        (NOW + timedelta(microseconds=1), "from the future"),
        (NOW - timedelta(hours=24, microseconds=1), "is stale"),
    ],
)
def test_validation_rejects_future_and_stale_evidence(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    completed_at: datetime,
    message: str,
) -> None:
    binding = _binding(tmp_path)
    _patch_binding(monkeypatch, binding)
    payload = _evidence_payload(
        tmp_path, binding=binding, completed_at=completed_at
    )
    path = tmp_path / (
        "readiness." + completed_at.strftime("%Y%m%dT%H%M%S.%fZ") + ".json"
    )
    _write_payload(path, payload)

    with pytest.raises(readiness.ReadinessValidationError, match=message):
        readiness.validate_readiness_evidence(
            path,
            config_path="config/paper.json",
            repo_root=tmp_path,
            now_utc=NOW,
        )


def test_validation_rejects_tampered_payload(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    binding = _binding(tmp_path)
    _patch_binding(monkeypatch, binding)
    payload = _evidence_payload(tmp_path, binding=binding)
    payload["passed"] = 1
    path = tmp_path / "readiness.20260817T180000.000000Z.json"
    _write_payload(path, payload)

    with pytest.raises(readiness.ReadinessValidationError):
        readiness.validate_readiness_evidence(
            path,
            config_path="config/paper.json",
            repo_root=tmp_path,
            now_utc=NOW,
        )


def test_selection_rejects_invalid_newer_candidate_and_uses_valid_older(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    binding = _binding(tmp_path)
    _patch_binding(monkeypatch, binding)
    evidence_dir = tmp_path / "evidence"
    older_time = NOW - timedelta(hours=2)
    newer_time = NOW - timedelta(hours=1)
    older = _evidence_payload(
        tmp_path, binding=binding, completed_at=older_time
    )
    newer = _evidence_payload(
        tmp_path, binding=binding, completed_at=newer_time
    )
    newer["plugin_autoload_disabled"] = 1
    _rewrite_digest(newer)
    older_path = evidence_dir / (
        "readiness." + older_time.strftime("%Y%m%dT%H%M%S.%fZ") + ".json"
    )
    newer_path = evidence_dir / (
        "readiness." + newer_time.strftime("%Y%m%dT%H%M%S.%fZ") + ".json"
    )
    _write_payload(older_path, older)
    _write_payload(newer_path, newer)

    selection = readiness.select_latest_valid_evidence(
        evidence_dir,
        config_path="config/paper.json",
        repo_root=tmp_path,
        now_utc=NOW,
    )

    assert selection.evidence is not None
    assert selection.evidence.path == older_path.resolve()
    assert selection.rejected == (
        (newer_path, "plugin_autoload_disabled must be a boolean"),
    )


def test_tracked_source_digest_includes_dirty_worktree_bytes(tmp_path: Path) -> None:
    tracked = tmp_path / "tradeos.py"
    tracked.write_text("before\n", encoding="utf-8")
    before = readiness._tracked_source_sha256(tmp_path, ("tradeos.py",))

    tracked.write_text("after\n", encoding="utf-8")
    after = readiness._tracked_source_sha256(tmp_path, ("tradeos.py",))

    assert before != after


def test_bound_paths_include_cached_and_nonignored_untracked_files(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    observed: list[tuple[str, ...]] = []

    def fake_git(repo_root: Path, *arguments: str) -> bytes:
        observed.append(arguments)
        return (
            b"tests/test_existing.py\0"
            b"tests/test_readiness.py\0"
            b"tradeos/readiness.py\0"
        )

    monkeypatch.setattr(readiness, "_run_git", fake_git)

    paths = readiness._tracked_paths(tmp_path)

    assert paths == (
        "tests/test_existing.py",
        "tests/test_readiness.py",
        "tradeos/readiness.py",
    )
    assert observed == [
        (
            "ls-files",
            "-z",
            "--cached",
            "--others",
            "--exclude-standard",
            "--",
        )
    ]

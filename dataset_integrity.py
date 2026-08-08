from __future__ import annotations

import hashlib
import hmac
import json
import os
import tempfile
from pathlib import Path
from typing import Any


BARS_SHA256_FIELD = "bars_sha256"
SNAPSHOT_SHA256_FIELD = "snapshot_sha256"
_VOLATILE_MANIFEST_FIELDS = {
    "created_at_utc",
    BARS_SHA256_FIELD,
    SNAPSHOT_SHA256_FIELD,
}


class DatasetIntegrityError(RuntimeError):
    """Raised when a versioned dataset is incomplete, changed, or inconsistent."""


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_sha256(value: object, *, field: str) -> str:
    normalized = str(value or "").strip().lower()
    if len(normalized) != 64 or any(char not in "0123456789abcdef" for char in normalized):
        raise DatasetIntegrityError(f"Dataset manifest field {field!r} is not a valid SHA-256 digest.")
    return normalized


def _manifest_fingerprint(manifest: dict[str, Any]) -> str:
    canonical_manifest = {
        key: value
        for key, value in manifest.items()
        if key != SNAPSHOT_SHA256_FIELD
    }
    try:
        encoded = json.dumps(
            canonical_manifest,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise DatasetIntegrityError("Dataset manifest is not canonical JSON.") from exc
    return hashlib.sha256(encoded).hexdigest()


def finalize_manifest(manifest: dict[str, Any], *, bars_sha256: str) -> dict[str, Any]:
    finalized = dict(manifest)
    finalized[BARS_SHA256_FIELD] = _require_sha256(bars_sha256, field=BARS_SHA256_FIELD)
    finalized.pop(SNAPSHOT_SHA256_FIELD, None)
    finalized[SNAPSHOT_SHA256_FIELD] = _manifest_fingerprint(finalized)
    return finalized


def load_verified_manifest(dataset_path: Path) -> dict[str, Any]:
    dataset_path = Path(dataset_path)
    bars_path = dataset_path / "bars.parquet"
    manifest_path = dataset_path / "manifest.json"
    if not bars_path.is_file():
        raise DatasetIntegrityError(f"Dataset bars file not found: {bars_path}")
    if not manifest_path.exists():
        return {}

    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise DatasetIntegrityError(f"Dataset manifest is unreadable: {manifest_path}") from exc
    if not isinstance(manifest, dict):
        raise DatasetIntegrityError(f"Dataset manifest must contain a JSON object: {manifest_path}")

    expected_bars_hash = manifest.get(BARS_SHA256_FIELD)
    expected_snapshot_hash = manifest.get(SNAPSHOT_SHA256_FIELD)
    if expected_snapshot_hash is not None and expected_bars_hash is None:
        raise DatasetIntegrityError(
            f"Dataset manifest has {SNAPSHOT_SHA256_FIELD} without {BARS_SHA256_FIELD}: {manifest_path}"
        )
    if expected_bars_hash is not None:
        expected = _require_sha256(expected_bars_hash, field=BARS_SHA256_FIELD)
        actual = file_sha256(bars_path)
        if not hmac.compare_digest(actual, expected):
            raise DatasetIntegrityError(
                f"Dataset bars hash mismatch for {bars_path}; the versioned snapshot was modified."
            )
    if expected_snapshot_hash is not None:
        expected = _require_sha256(expected_snapshot_hash, field=SNAPSHOT_SHA256_FIELD)
        actual = _manifest_fingerprint(manifest)
        if not hmac.compare_digest(actual, expected):
            raise DatasetIntegrityError(
                f"Dataset manifest fingerprint mismatch for {manifest_path}; the snapshot metadata was modified."
            )
    return manifest


def _semantic_manifest(manifest: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in manifest.items()
        if key not in _VOLATILE_MANIFEST_FIELDS
    }


def _validate_existing_snapshot(
    dataset_dir: Path,
    *,
    candidate_bars_sha256: str,
    candidate_manifest: dict[str, Any],
) -> tuple[Path, Path]:
    parquet_path = dataset_dir / "bars.parquet"
    manifest_path = dataset_dir / "manifest.json"
    if not parquet_path.is_file() or not manifest_path.is_file():
        raise DatasetIntegrityError(
            f"Dataset directory already exists but is incomplete; refusing to overwrite it: {dataset_dir}"
        )

    existing_manifest = load_verified_manifest(dataset_dir)
    existing_bars_sha256 = file_sha256(parquet_path)
    if not hmac.compare_digest(existing_bars_sha256, candidate_bars_sha256):
        raise DatasetIntegrityError(
            f"Dataset id {dataset_dir.name!r} already exists with different bars; "
            "versioned snapshots are immutable."
        )
    if _semantic_manifest(existing_manifest) != _semantic_manifest(candidate_manifest):
        raise DatasetIntegrityError(
            f"Dataset id {dataset_dir.name!r} already exists with different metadata; "
            "versioned snapshots are immutable."
        )
    return parquet_path, manifest_path


def write_immutable_dataset(
    dataset: Any,
    manifest: dict[str, Any],
    output_root: Path,
    dataset_id: str,
) -> tuple[Path, Path]:
    """Write one content-fingerprinted snapshot, or accept an identical existing snapshot."""

    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    dataset_dir = output_root / dataset_id

    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            prefix=f".{dataset_id}.",
            suffix=".parquet.tmp",
            dir=output_root,
            delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
        dataset.to_parquet(temporary_path, index=False)
        bars_sha256 = file_sha256(temporary_path)
        finalized_manifest = finalize_manifest(manifest, bars_sha256=bars_sha256)

        if dataset_dir.exists():
            return _validate_existing_snapshot(
                dataset_dir,
                candidate_bars_sha256=bars_sha256,
                candidate_manifest=finalized_manifest,
            )

        try:
            dataset_dir.mkdir(parents=False, exist_ok=False)
        except FileExistsError:
            return _validate_existing_snapshot(
                dataset_dir,
                candidate_bars_sha256=bars_sha256,
                candidate_manifest=finalized_manifest,
            )

        parquet_path = dataset_dir / "bars.parquet"
        manifest_path = dataset_dir / "manifest.json"
        os.replace(temporary_path, parquet_path)
        temporary_path = None

        manifest_temp_path = dataset_dir / ".manifest.json.tmp"
        manifest_temp_path.write_text(
            json.dumps(finalized_manifest, indent=2, ensure_ascii=False, allow_nan=False),
            encoding="utf-8",
        )
        os.replace(manifest_temp_path, manifest_path)
        return parquet_path, manifest_path
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from backtest_runner import load_dataset
from dataset_integrity import DatasetIntegrityError, file_sha256, write_immutable_dataset


def _bars(close: float = 100.0) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "symbol": ["AMD"],
            "timestamp": [pd.Timestamp("2026-01-02T14:30:00Z")],
            "open": [close],
            "high": [close],
            "low": [close],
            "close": [close],
            "volume": [100],
        }
    )


def _manifest(created_at: str = "2026-01-03T00:00:00+00:00") -> dict[str, object]:
    return {
        "dataset_id": "snapshot-one",
        "created_at_utc": created_at,
        "symbols": ["AMD"],
        "timeframe": "15Min",
        "row_count": 1,
    }


def test_new_snapshot_is_fingerprinted_and_verified(tmp_path: Path) -> None:
    parquet_path, manifest_path = write_immutable_dataset(
        _bars(), _manifest(), tmp_path, "snapshot-one"
    )

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["bars_sha256"] == file_sha256(parquet_path)
    assert len(manifest["snapshot_sha256"]) == 64

    loaded, loaded_manifest = load_dataset(parquet_path.parent)
    assert loaded["close"].tolist() == [100.0]
    assert loaded_manifest == manifest


def test_identical_rerun_is_idempotent_but_conflicting_rewrite_is_rejected(
    tmp_path: Path,
) -> None:
    parquet_path, manifest_path = write_immutable_dataset(
        _bars(), _manifest(), tmp_path, "snapshot-one"
    )
    original_bars = parquet_path.read_bytes()
    original_manifest = manifest_path.read_bytes()

    # Creation time is intentionally excluded from semantic identity.
    same_paths = write_immutable_dataset(
        _bars(),
        _manifest("2026-01-04T00:00:00+00:00"),
        tmp_path,
        "snapshot-one",
    )
    assert same_paths == (parquet_path, manifest_path)
    assert parquet_path.read_bytes() == original_bars
    assert manifest_path.read_bytes() == original_manifest

    with pytest.raises(DatasetIntegrityError, match="immutable"):
        write_immutable_dataset(_bars(101.0), _manifest(), tmp_path, "snapshot-one")
    assert parquet_path.read_bytes() == original_bars
    assert manifest_path.read_bytes() == original_manifest


def test_modified_snapshot_is_rejected_before_parquet_is_read(tmp_path: Path) -> None:
    parquet_path, _ = write_immutable_dataset(
        _bars(), _manifest(), tmp_path, "snapshot-one"
    )
    parquet_path.write_bytes(parquet_path.read_bytes() + b"tampered")

    with pytest.raises(DatasetIntegrityError, match="bars hash mismatch"):
        load_dataset(parquet_path.parent)


def test_legacy_snapshot_without_hashes_remains_readable(tmp_path: Path) -> None:
    dataset_path = tmp_path / "legacy"
    dataset_path.mkdir()
    _bars().to_parquet(dataset_path / "bars.parquet", index=False)
    (dataset_path / "manifest.json").write_text(
        json.dumps({"dataset_id": "legacy", "symbols": ["AMD"]}),
        encoding="utf-8",
    )

    loaded, manifest = load_dataset(dataset_path)
    assert loaded["symbol"].tolist() == ["AMD"]
    assert manifest == {"dataset_id": "legacy", "symbols": ["AMD"]}

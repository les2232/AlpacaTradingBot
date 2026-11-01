from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd
import pytest

from dataset_snapshotter import (
    align_symbols_on_shared_timestamps,
    build_dataset_id,
    build_symbol_retention_report,
    format_utc_z,
    normalize_symbol_sequence,
)


def test_symbol_normalization_preserves_contract_order_and_rejects_ambiguity() -> None:
    assert normalize_symbol_sequence([" tsla ", "AMD", "msft"]) == [
        "TSLA",
        "AMD",
        "MSFT",
    ]
    for invalid in (
        ["AMD", "amd"],
        ["AMD", "   "],
        ["AMD", 123],
        ["../AMD"],
        ["C:\\AMD"],
        [],
    ):
        with pytest.raises(ValueError):
            normalize_symbol_sequence(invalid)


def test_dataset_id_and_alignment_bind_requested_symbol_order() -> None:
    start = datetime(2026, 7, 21, tzinfo=timezone.utc)
    end = datetime(2026, 8, 20, tzinfo=timezone.utc)
    forward = build_dataset_id(
        ["TSLA", "AMD", "MSFT"],
        "15Min",
        start,
        end,
        "iex",
        "abc123",
    )
    reversed_order = build_dataset_id(
        ["MSFT", "AMD", "TSLA"],
        "15Min",
        start,
        end,
        "iex",
        "abc123",
    )
    assert forward != reversed_order
    assert forward.startswith("TSLA-AMD-MSFT__")

    timestamps = pd.to_datetime(
        ["2026-08-20T00:00:00Z", "2026-08-20T00:15:00Z"],
        utc=True,
    )
    frame = pd.DataFrame(
        [
            {"symbol": symbol, "timestamp": timestamp, "close": 100.0}
            for symbol in ["AMD", "MSFT", "TSLA"]
            for timestamp in reversed(timestamps)
        ]
    )
    aligned = align_symbols_on_shared_timestamps(
        frame,
        ["TSLA", "AMD", "MSFT"],
    )
    assert aligned["symbol"].drop_duplicates().tolist() == ["TSLA", "AMD", "MSFT"]
    assert aligned.groupby("symbol", sort=False)["timestamp"].apply(list).tolist() == [
        list(timestamps),
        list(timestamps),
        list(timestamps),
    ]

    report = build_symbol_retention_report(
        ["TSLA", "AMD", "MSFT"],
        {"TSLA": 2, "AMD": 2, "MSFT": 2},
        {"TSLA": 2, "AMD": 1, "MSFT": 2},
    )
    assert [entry["symbol"] for entry in report] == ["TSLA", "AMD", "MSFT"]


def test_manifest_timestamp_spelling_is_canonical_utc_z() -> None:
    assert format_utc_z(datetime(2026, 8, 20, tzinfo=timezone.utc)) == (
        "2026-08-20T00:00:00Z"
    )
    assert format_utc_z(
        datetime.fromisoformat("2026-08-20T02:30:00+02:30")
    ) == "2026-08-20T00:00:00Z"
    with pytest.raises(ValueError):
        format_utc_z(datetime(2026, 8, 20))

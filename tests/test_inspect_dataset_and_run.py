import json
from pathlib import Path
import uuid

import pandas as pd

from inspect_dataset_and_run import (
    build_warnings,
    format_inspection_output,
    inspect_dataset,
    inspect_run_configuration,
)


def _write_dataset(dataset_dir: Path, *, with_manifest: bool = True) -> None:
    df = pd.DataFrame(
        [
            {"symbol": "AMD", "timestamp": pd.Timestamp("2026-01-02T14:30:00Z"), "open": 1.0, "high": 1.1, "low": 0.9, "close": 1.05, "volume": 10, "trade_count": 1, "vwap": 1.02},
            {"symbol": "AMD", "timestamp": pd.Timestamp("2026-01-02T14:45:00Z"), "open": 1.05, "high": 1.2, "low": 1.0, "close": 1.15, "volume": 12, "trade_count": 1, "vwap": 1.10},
            {"symbol": "JPM", "timestamp": pd.Timestamp("2026-01-02T14:30:00Z"), "open": 2.0, "high": 2.1, "low": 1.9, "close": 2.05, "volume": 15, "trade_count": 1, "vwap": 2.01},
            {"symbol": "JPM", "timestamp": pd.Timestamp("2026-01-02T14:45:00Z"), "open": 2.05, "high": 2.2, "low": 2.0, "close": 2.10, "volume": 16, "trade_count": 1, "vwap": 2.08},
        ]
    )
    df.to_parquet(dataset_dir / "bars.parquet", index=False)
    if with_manifest:
        manifest = {
            "symbols": ["AMD", "JPM"],
            "timeframe": "15Min",
            "feed": "iex",
            "start_utc": "2026-01-02T00:00:00+00:00",
            "end_utc": "2026-03-31T00:00:00+00:00",
            "row_count": 4,
        }
        (dataset_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")


def _make_workspace_tmpdir() -> Path:
    base = Path("tmp_test_artifacts")
    base.mkdir(exist_ok=True)
    path = (base / f"inspect_dataset_{uuid.uuid4().hex}").resolve()
    path.mkdir(parents=True, exist_ok=False)
    return path


def test_inspect_dataset_uses_manifest_when_present() -> None:
    tmp_path = _make_workspace_tmpdir()
    dataset_dir = tmp_path / "sample__15Min__20260102T000000Z__20260331T000000Z__iex__abc123"
    dataset_dir.mkdir()
    _write_dataset(dataset_dir, with_manifest=True)

    info = inspect_dataset(dataset_dir)

    assert info["timeframe"] == "15Min"
    assert info["feed"] == "iex"
    assert info["symbol_count"] == 2
    assert info["total_bars"] == 4


def test_inspect_dataset_infers_without_manifest() -> None:
    tmp_path = _make_workspace_tmpdir()
    dataset_dir = tmp_path / "sample__15Min__20260102T000000Z__20260331T000000Z__sip__abc123"
    dataset_dir.mkdir()
    _write_dataset(dataset_dir, with_manifest=False)

    info = inspect_dataset(dataset_dir)

    assert info["timeframe"] == "15Min"
    assert info["feed"] == "sip"
    assert info["symbol_count"] == 2


def test_inspect_run_configuration_detects_symbol_mismatch() -> None:
    tmp_path = _make_workspace_tmpdir()
    dataset_dir = tmp_path / "sample__15Min__20260102T000000Z__20260331T000000Z__iex__abc123"
    dataset_dir.mkdir()
    _write_dataset(dataset_dir, with_manifest=True)
    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps(
            {
                "source": {"dataset": str(dataset_dir)},
                "runtime": {"strategy_mode": "volatility_expansion", "symbols": ["AMD", "MSFT"]},
            }
        ),
        encoding="utf-8",
    )

    dataset_info = inspect_dataset(dataset_dir)
    run_info = inspect_run_configuration(
        dataset_info=dataset_info,
        config_path=config_path,
        strategy_override=None,
        cli_symbols=None,
        expected_timeframe=None,
        hold_bars_override=None,
    )

    assert run_info["missing_symbols"] == ["MSFT"]


def test_build_warnings_triggers_sample_and_time_bucket_messages() -> None:
    tmp_path = _make_workspace_tmpdir()
    dataset_dir = tmp_path / "sample__15Min__20260102T000000Z__20260215T000000Z__iex__abc123"
    dataset_dir.mkdir()
    _write_dataset(dataset_dir, with_manifest=True)
    dataset_info = inspect_dataset(dataset_dir)
    dataset_info["date_range_days"] = 44
    run_info = {
        "missing_symbols": [],
        "unused_dataset_symbols": [],
        "source_dataset": str(dataset_dir),
        "symbol_source": "config_runtime",
        "symbols_used": ["AMD", "JPM"],
        "timeframe_assumption": "15Min",
    }
    signals_df = pd.DataFrame([{"time_bucket": "open_30m"} for _ in range(4)])
    trades_df = pd.DataFrame([{"realized_pnl": -1.0} for _ in range(3)])

    warnings = build_warnings(
        dataset_info=dataset_info,
        run_info=run_info,
        expected_timeframe="15Min",
        signals_df=signals_df,
        trades_df=trades_df,
    )

    assert any("only 4 signals" in warning.lower() for warning in warnings)
    assert any("all signals occur in open_30m" in warning.lower() for warning in warnings)
    assert any("only 3 closed trades" in warning.lower() for warning in warnings)
    assert any("iex" in warning.lower() for warning in warnings)


def test_format_output_handles_empty_dataset_summary() -> None:
    dataset_info = {
        "dataset_path": Path("dummy"),
        "timeframe": None,
        "feed": None,
        "start_ts": None,
        "end_ts": None,
        "total_bars": 0,
        "symbol_count": 0,
        "symbols": [],
        "evenly_spaced": None,
        "missing_values": pd.DataFrame(columns=["column", "missing_count"]),
    }
    run_info = {
        "config_path": None,
        "strategy_mode": "unknown",
        "symbols_used": [],
        "symbol_source": "dataset_inferred",
        "timeframe_assumption": None,
        "hold_bars": None,
        "active_filters": [],
        "missing_symbols": [],
        "unused_dataset_symbols": [],
    }

    output = format_inspection_output(dataset_info=dataset_info, run_info=run_info, warnings=[])

    assert "DATASET SUMMARY" in output
    assert "Missing values: none detected" in output
    assert "WARNINGS" in output

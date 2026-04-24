from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from backtest_runner import load_dataset
from run_edge_audit import _frame_text, _load_runtime_payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inspect dataset metadata and resolved run configuration for research sanity checks."
    )
    parser.add_argument("--dataset", help="Dataset directory. Defaults to config source.dataset when available.")
    parser.add_argument("--config", help="Optional runtime config JSON.")
    parser.add_argument("--strategy", help="Optional strategy override shown in the run summary.")
    parser.add_argument("--symbols", nargs="*", help="Optional symbol override.")
    parser.add_argument("--expected-timeframe", help="Optional expected timeframe label, e.g. 15Min.")
    parser.add_argument("--hold-bars", help="Optional hold horizon override, e.g. 4 or 4,8,10.")
    parser.add_argument("--signals-csv", help="Optional signals CSV from a research run.")
    parser.add_argument("--trades-csv", help="Optional closed-trades CSV from a research run.")
    return parser.parse_args()


def _normalize_symbol_list(symbols: list[str] | tuple[str, ...] | None) -> list[str]:
    if not symbols:
        return []
    normalized: list[str] = []
    seen: set[str] = set()
    for raw_symbol in symbols:
        symbol = str(raw_symbol).strip().upper()
        if not symbol or symbol in seen:
            continue
        normalized.append(symbol)
        seen.add(symbol)
    return normalized


def _parse_timeframe_from_minutes(minutes: int | None) -> str | None:
    if minutes is None or minutes <= 0:
        return None
    mapping = {
        1: "1Min",
        5: "5Min",
        15: "15Min",
        30: "30Min",
        60: "1H",
        240: "4H",
        1440: "1D",
    }
    return mapping.get(minutes, f"{minutes}Min")


def _infer_feed_from_path(dataset_path: Path) -> str | None:
    parts = dataset_path.name.split("__")
    if len(parts) >= 5:
        return parts[4]
    return None


def _infer_timeframe_from_path(dataset_path: Path) -> str | None:
    parts = dataset_path.name.split("__")
    if len(parts) >= 2:
        return parts[1]
    return None


def _infer_timeframe_from_df(df: pd.DataFrame) -> str | None:
    if df.empty or "timestamp" not in df.columns:
        return None
    sample = df.sort_values(["symbol", "timestamp"]).copy()
    sample["day"] = sample["timestamp"].dt.tz_convert("America/New_York").dt.normalize()
    diffs = (
        sample.groupby(["symbol", "day"], dropna=False)["timestamp"]
        .diff()
        .dropna()
    )
    if diffs.empty:
        return None
    minutes = int(diffs.dt.total_seconds().div(60).round().mode().iloc[0])
    return _parse_timeframe_from_minutes(minutes)


def _format_date(value: pd.Timestamp | None) -> str:
    if value is None or pd.isna(value):
        return "unknown"
    return pd.Timestamp(value).tz_convert("UTC").strftime("%Y-%m-%d")


def _dataset_duration_days(start_ts: pd.Timestamp | None, end_ts: pd.Timestamp | None) -> int | None:
    if start_ts is None or end_ts is None or pd.isna(start_ts) or pd.isna(end_ts):
        return None
    return int((pd.Timestamp(end_ts) - pd.Timestamp(start_ts)).days)


def _missing_value_summary(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["column", "missing_count"])
    missing = df.isna().sum()
    rows = [{"column": col, "missing_count": int(count)} for col, count in missing.items() if int(count) > 0]
    return pd.DataFrame(rows).sort_values(["missing_count", "column"], ascending=[False, True]) if rows else pd.DataFrame(
        columns=["column", "missing_count"]
    )


def _spacing_summary(df: pd.DataFrame, timeframe_label: str | None) -> tuple[bool | None, pd.DataFrame]:
    columns = ["symbol", "dominant_spacing", "off_spacing_rows"]
    if df.empty or "timestamp" not in df.columns:
        return None, pd.DataFrame(columns=columns)

    expected = None
    if timeframe_label and timeframe_label.endswith("Min"):
        try:
            expected = int(timeframe_label.replace("Min", ""))
        except ValueError:
            expected = None
    if timeframe_label == "1H":
        expected = 60
    elif timeframe_label == "1D":
        expected = 1440

    ordered = df.sort_values(["symbol", "timestamp"]).copy()
    ordered["day"] = ordered["timestamp"].dt.tz_convert("America/New_York").dt.normalize()
    ordered["diff_min"] = (
        ordered.groupby(["symbol", "day"], dropna=False)["timestamp"]
        .diff()
        .dt.total_seconds()
        .div(60)
    )
    rows: list[dict[str, Any]] = []
    evenly_spaced = True
    for symbol, group in ordered.groupby("symbol", sort=True):
        diffs = group["diff_min"].dropna()
        if diffs.empty:
            rows.append({"symbol": symbol, "dominant_spacing": "unknown", "off_spacing_rows": 0})
            continue
        dominant = int(diffs.round().mode().iloc[0])
        off_rows = int((diffs.round() != dominant).sum())
        if expected is not None and dominant != expected:
            evenly_spaced = False
        if off_rows > 0:
            evenly_spaced = False
        rows.append(
            {
                "symbol": symbol,
                "dominant_spacing": f"{dominant}Min",
                "off_spacing_rows": off_rows,
            }
        )
    return evenly_spaced, pd.DataFrame(rows, columns=columns)


def inspect_dataset(dataset_path: Path) -> dict[str, Any]:
    df, manifest = load_dataset(dataset_path)
    manifest = manifest if isinstance(manifest, dict) else {}
    symbols = _normalize_symbol_list(manifest.get("symbols")) or _normalize_symbol_list(
        df["symbol"].dropna().astype(str).tolist() if "symbol" in df.columns else []
    )
    start_ts = pd.Timestamp(manifest["start_utc"], tz="UTC") if isinstance(manifest.get("start_utc"), str) else (
        pd.Timestamp(df["timestamp"].min()) if not df.empty and "timestamp" in df.columns else None
    )
    end_ts = pd.Timestamp(manifest["end_utc"], tz="UTC") if isinstance(manifest.get("end_utc"), str) else (
        pd.Timestamp(df["timestamp"].max()) if not df.empty and "timestamp" in df.columns else None
    )
    timeframe = (
        str(manifest.get("timeframe"))
        if isinstance(manifest.get("timeframe"), str)
        else _infer_timeframe_from_path(dataset_path) or _infer_timeframe_from_df(df)
    )
    feed = (
        str(manifest.get("feed"))
        if isinstance(manifest.get("feed"), str)
        else _infer_feed_from_path(dataset_path)
    )
    evenly_spaced, spacing_table = _spacing_summary(df, timeframe)
    missing_table = _missing_value_summary(df)
    return {
        "dataset_path": dataset_path,
        "manifest": manifest,
        "timeframe": timeframe,
        "feed": feed,
        "start_ts": start_ts,
        "end_ts": end_ts,
        "date_range_days": _dataset_duration_days(start_ts, end_ts),
        "total_bars": int(len(df)),
        "symbol_count": int(len(symbols)),
        "symbols": symbols,
        "evenly_spaced": evenly_spaced,
        "spacing_table": spacing_table,
        "missing_values": missing_table,
        "df": df,
    }


def inspect_run_configuration(
    *,
    dataset_info: dict[str, Any],
    config_path: Path | None,
    strategy_override: str | None,
    cli_symbols: list[str] | None,
    expected_timeframe: str | None,
    hold_bars_override: str | None,
) -> dict[str, Any]:
    payload = _load_runtime_payload(config_path) if config_path else {"runtime": {}, "source": {}}
    runtime = dict(payload["runtime"])
    source_dataset = payload["source"].get("dataset") if isinstance(payload["source"].get("dataset"), str) else None

    strategy_mode = str(strategy_override or runtime.get("strategy_mode") or "unknown")
    config_symbols = _normalize_symbol_list(runtime.get("symbols")) if isinstance(runtime.get("symbols"), list) else []
    symbols_used = _normalize_symbol_list(cli_symbols) or config_symbols or dataset_info["symbols"]
    symbol_source = "cli_override" if cli_symbols else ("config_runtime" if config_symbols else "dataset_inferred")

    timeframe_assumption = expected_timeframe
    if timeframe_assumption is None:
        timeframe_assumption = _parse_timeframe_from_minutes(
            int(runtime["bar_timeframe_minutes"])
        ) if "bar_timeframe_minutes" in runtime else None

    hold_bars = hold_bars_override
    if hold_bars is None:
        for key in (
            "volatility_expansion_hold_bars",
            "trend_pullback_hold_bars",
            "momentum_breakout_hold_bars",
        ):
            if key in runtime:
                hold_bars = str(runtime[key])
                break

    active_filters: list[str] = []
    for key in (
        "regime_filter_enabled",
        "time_window_mode",
        "atr_percentile_threshold",
        "volatility_expansion_max_atr_percentile",
        "volatility_expansion_trend_filter",
        "volatility_expansion_use_volume_confirm",
        "trend_pullback_min_adx",
        "trend_pullback_min_atr_percentile",
        "momentum_breakout_min_adx",
        "momentum_breakout_min_atr_percentile",
    ):
        value = runtime.get(key)
        if isinstance(value, bool) and value:
            active_filters.append(f"{key}=true")
        elif isinstance(value, (int, float)) and float(value) > 0:
            active_filters.append(f"{key}={value}")
        elif isinstance(value, str) and value and key == "time_window_mode":
            active_filters.append(f"{key}={value}")

    dataset_symbols = set(dataset_info["symbols"])
    missing_symbols = [symbol for symbol in symbols_used if symbol not in dataset_symbols]
    unused_dataset_symbols = [symbol for symbol in dataset_info["symbols"] if symbol not in set(symbols_used)]

    return {
        "config_path": config_path,
        "source_dataset": source_dataset,
        "strategy_mode": strategy_mode,
        "symbols_used": symbols_used,
        "symbol_source": symbol_source,
        "timeframe_assumption": timeframe_assumption,
        "hold_bars": hold_bars,
        "active_filters": active_filters,
        "missing_symbols": missing_symbols,
        "unused_dataset_symbols": unused_dataset_symbols,
        "dataset_arg_fell_back": config_path is not None and source_dataset is not None,
    }


def load_optional_run_artifacts(signals_csv: str | None, trades_csv: str | None) -> tuple[pd.DataFrame, pd.DataFrame]:
    signals_df = pd.read_csv(signals_csv) if signals_csv else pd.DataFrame()
    trades_df = pd.read_csv(trades_csv) if trades_csv else pd.DataFrame()
    return signals_df, trades_df


def build_warnings(
    *,
    dataset_info: dict[str, Any],
    run_info: dict[str, Any],
    expected_timeframe: str | None,
    signals_df: pd.DataFrame,
    trades_df: pd.DataFrame,
) -> list[str]:
    warnings: list[str] = []

    dataset_timeframe = dataset_info.get("timeframe")
    if expected_timeframe and dataset_timeframe and str(expected_timeframe).lower() != str(dataset_timeframe).lower():
        warnings.append(
            f"Dataset timeframe {dataset_timeframe} does not match expected timeframe {expected_timeframe} -> strategy timing assumptions may be invalid."
        )
    config_timeframe = run_info.get("timeframe_assumption")
    if config_timeframe and dataset_timeframe and str(config_timeframe).lower() != str(dataset_timeframe).lower():
        warnings.append(
            f"Config/runtime timeframe assumption {config_timeframe} does not match dataset timeframe {dataset_timeframe} -> results may be based on the wrong bar size."
        )

    if run_info["missing_symbols"]:
        warnings.append(
            f"Requested symbols are missing from the dataset/date range: {', '.join(run_info['missing_symbols'])}."
        )
    if run_info["unused_dataset_symbols"]:
        warnings.append(
            f"Dataset contains symbols that are not used in this run: {', '.join(run_info['unused_dataset_symbols'][:12])}"
            + (" ..." if len(run_info["unused_dataset_symbols"]) > 12 else "")
        )
    if run_info["symbol_source"] != "cli_override" and run_info["source_dataset"]:
        warnings.append("Dataset path was resolved from config source.dataset rather than an explicit CLI dataset argument.")

    if dataset_info.get("feed") and str(dataset_info["feed"]).lower() == "iex":
        warnings.append("Feed is IEX -> single-exchange data can bias liquidity, volume, and intraday breakout behavior versus SIP.")

    duration_days = dataset_info.get("date_range_days")
    if duration_days is not None and duration_days < 75:
        warnings.append(
            f"Dataset covers only about {duration_days} days -> limited regime diversity and weak statistical confidence."
        )

    if len(run_info["symbols_used"]) < 4:
        warnings.append("Only a few symbols are used in the run -> conclusions may be symbol-specific rather than broad.")

    if dataset_info.get("evenly_spaced") is False:
        warnings.append("Bars are not evenly spaced within sessions -> forward-return and hold-bar logic may be distorted.")

    missing_table = dataset_info.get("missing_values")
    if isinstance(missing_table, pd.DataFrame) and not missing_table.empty:
        warnings.append("Dataset contains missing/NaN values -> indicator calculations or signal counts may be unreliable.")

    if not signals_df.empty:
        if len(signals_df) < 20:
            warnings.append(f"Only {len(signals_df)} signals generated -> sample size too small for strong conclusions.")
        if "time_bucket" in signals_df.columns:
            bucket_counts = signals_df["time_bucket"].fillna("unknown").value_counts(dropna=False)
            if len(bucket_counts) == 1:
                warnings.append(f"All signals occur in {bucket_counts.index[0]} -> possible time-window bias.")
            elif not bucket_counts.empty and (bucket_counts.iloc[0] / max(1, len(signals_df))) >= 0.8:
                warnings.append(
                    f"Signals are heavily concentrated in {bucket_counts.index[0]} ({bucket_counts.iloc[0]}/{len(signals_df)}) -> possible time-of-day bias."
                )

    if not trades_df.empty and len(trades_df) < 20:
        warnings.append(f"Only {len(trades_df)} closed trades -> realized performance is not statistically stable.")

    return warnings


def format_inspection_output(
    *,
    dataset_info: dict[str, Any],
    run_info: dict[str, Any],
    warnings: list[str],
) -> str:
    lines = [
        "DATASET SUMMARY",
        "---------------",
        f"Path: {dataset_info['dataset_path']}",
        f"Timeframe: {dataset_info.get('timeframe') or 'unknown'}",
        f"Feed: {dataset_info.get('feed') or 'unknown'}",
        f"Date range: {_format_date(dataset_info.get('start_ts'))} -> {_format_date(dataset_info.get('end_ts'))}",
        f"Bars: {dataset_info.get('total_bars', 0)}",
        f"Symbols: {dataset_info.get('symbol_count', 0)}",
        f"Symbol list: {', '.join(dataset_info.get('symbols', [])) or 'none'}",
        f"Evenly spaced: {dataset_info.get('evenly_spaced') if dataset_info.get('evenly_spaced') is not None else 'unknown'}",
    ]
    missing_table = dataset_info.get("missing_values")
    if isinstance(missing_table, pd.DataFrame) and not missing_table.empty:
        lines.append("Missing values:")
        lines.append(_frame_text(missing_table, max_rows=16))
    else:
        lines.append("Missing values: none detected")

    lines.extend(
        [
            "",
            "RUN CONFIG",
            "----------",
            f"Config: {run_info['config_path'] or 'none'}",
            f"Strategy: {run_info['strategy_mode']}",
            f"Symbols used ({run_info['symbol_source']}): {', '.join(run_info['symbols_used']) or 'none'}",
            f"Timeframe assumption: {run_info.get('timeframe_assumption') or 'not specified'}",
            f"Hold bars: {run_info.get('hold_bars') or 'not specified'}",
            f"Filters enabled: {', '.join(run_info['active_filters']) or 'none'}",
        ]
    )
    if run_info["missing_symbols"]:
        lines.append(f"Missing requested symbols: {', '.join(run_info['missing_symbols'])}")
    if run_info["unused_dataset_symbols"]:
        lines.append(
            "Dataset symbols not used: "
            + ", ".join(run_info["unused_dataset_symbols"][:12])
            + (" ..." if len(run_info["unused_dataset_symbols"]) > 12 else "")
        )

    lines.extend(["", "WARNINGS", "--------"])
    if warnings:
        lines.extend([f"- {warning}" for warning in warnings])
    else:
        lines.append("- none")
    return "\n".join(lines)


def resolve_dataset_path(dataset_arg: str | None, config_path: Path | None) -> Path:
    if dataset_arg:
        return Path(dataset_arg)
    if config_path:
        payload = _load_runtime_payload(config_path)
        source_dataset = payload["source"].get("dataset")
        if isinstance(source_dataset, str) and source_dataset:
            return Path(source_dataset)
    raise RuntimeError("Dataset path not provided and config source.dataset is missing.")


def main() -> None:
    args = parse_args()
    config_path = Path(args.config) if args.config else None
    dataset_path = resolve_dataset_path(args.dataset, config_path)
    dataset_info = inspect_dataset(dataset_path)
    run_info = inspect_run_configuration(
        dataset_info=dataset_info,
        config_path=config_path,
        strategy_override=args.strategy,
        cli_symbols=args.symbols,
        expected_timeframe=args.expected_timeframe,
        hold_bars_override=args.hold_bars,
    )
    signals_df, trades_df = load_optional_run_artifacts(args.signals_csv, args.trades_csv)
    warnings = build_warnings(
        dataset_info=dataset_info,
        run_info=run_info,
        expected_timeframe=args.expected_timeframe,
        signals_df=signals_df,
        trades_df=trades_df,
    )
    print(format_inspection_output(dataset_info=dataset_info, run_info=run_info, warnings=warnings))


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from backtest_runner import run_backtest
from daily_report import _dedupe, _load
from research.run_trade_path_diagnostics import pair_round_trip_trades


DEFAULT_CONFIG_PATH = Path("config") / "live_config.json"
DEFAULT_LOG_ROOT = Path("logs")
DEFAULT_OUTPUT_DIR = Path("results") / "live_mismatch_report"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare current runtime replay results against live logs over the same date range."
    )
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH), help="Path to runtime config JSON.")
    parser.add_argument("--dataset", help="Optional dataset override. Defaults to config source.dataset.")
    parser.add_argument("--log-root", default=str(DEFAULT_LOG_ROOT), help="Live log root directory.")
    parser.add_argument("--start-date", required=True, help="UTC start date (YYYY-MM-DD).")
    parser.add_argument("--end-date", required=True, help="UTC end date (YYYY-MM-DD).")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR), help="Directory for CSV/JSON artifacts.")
    parser.add_argument("--commission", type=float, default=0.01, help="Per-order commission for replay.")
    parser.add_argument("--slippage", type=float, default=0.05, help="Per-share slippage for replay.")
    return parser.parse_args()


def _load_runtime_payload(config_path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"Runtime config must be a JSON object: {config_path}")
    runtime = payload.get("runtime", payload)
    if not isinstance(runtime, dict):
        raise RuntimeError(f"Runtime config field 'runtime' must be a JSON object: {config_path}")
    source = payload.get("source")
    return runtime, source if isinstance(source, dict) else {}


def _normalize_symbols(raw_symbols: Any) -> list[str]:
    if not isinstance(raw_symbols, list):
        return []
    normalized: list[str] = []
    seen: set[str] = set()
    for raw_symbol in raw_symbols:
        symbol = str(raw_symbol).strip().upper()
        if not symbol or symbol in seen:
            continue
        normalized.append(symbol)
        seen.add(symbol)
    return normalized


def _effective_symbols(runtime: dict[str, Any]) -> list[str]:
    symbols = _normalize_symbols(runtime.get("symbols"))
    excluded = set(_normalize_symbols(runtime.get("excluded_symbols")))
    return [symbol for symbol in symbols if symbol not in excluded]


def _build_backtest_kwargs(
    runtime: dict[str, Any],
    *,
    dataset_path: Path,
    start_date: str,
    end_date: str,
    commission: float,
    slippage: float,
) -> dict[str, Any]:
    return {
        "dataset_path": dataset_path,
        "symbols": _effective_symbols(runtime) or None,
        "sma_bars": int(runtime.get("sma_bars", 15) or 15),
        "commission": commission,
        "slippage": slippage,
        "entry_threshold_pct": float(runtime.get("entry_threshold_pct", 0.0015) or 0.0015),
        "threshold_mode": str(runtime.get("threshold_mode", "static_pct") or "static_pct"),
        "atr_multiple": float(runtime.get("atr_multiple", 1.0) or 1.0),
        "atr_percentile_threshold": float(runtime.get("atr_percentile_threshold", 0.0) or 0.0),
        "time_window_mode": str(runtime.get("time_window_mode", "full_day") or "full_day"),
        "regime_filter_enabled": bool(runtime.get("regime_filter_enabled", False)),
        "orb_filter_mode": str(runtime.get("orb_filter_mode", "none") or "none"),
        "breakout_exit_style": str(runtime.get("breakout_exit_style", "target_1x_stop_low") or "target_1x_stop_low"),
        "breakout_tight_stop_fraction": float(runtime.get("breakout_tight_stop_fraction", 0.5) or 0.5),
        "mean_reversion_exit_style": str(runtime.get("mean_reversion_exit_style", "sma") or "sma"),
        "mean_reversion_max_atr_percentile": float(runtime.get("mean_reversion_max_atr_percentile", 0.0) or 0.0),
        "strategy_mode": str(runtime.get("strategy_mode", "mean_reversion") or "mean_reversion"),
        "symbol_strategy_modes": runtime.get("symbol_strategy_modes"),
        "ml_probability_buy": float(runtime.get("ml_probability_buy", 0.55) or 0.55),
        "ml_probability_sell": float(runtime.get("ml_probability_sell", 0.45) or 0.45),
        "bb_period": int(runtime.get("bb_period", 20) or 20),
        "bb_stddev_mult": float(runtime.get("bb_stddev_mult", 2.0) or 2.0),
        "bb_width_lookback": int(runtime.get("bb_width_lookback", 100) or 100),
        "bb_squeeze_quantile": float(runtime.get("bb_squeeze_quantile", 0.2) or 0.2),
        "bb_slope_lookback": int(runtime.get("bb_slope_lookback", 3) or 3),
        "bb_use_volume_confirm": bool(runtime.get("bb_use_volume_confirm", True)),
        "bb_volume_mult": float(runtime.get("bb_volume_mult", 1.2) or 1.2),
        "bb_exit_mode": str(runtime.get("bb_exit_mode", "middle_band") or "middle_band"),
        "start_date": start_date,
        "end_date": end_date,
    }


def _load_live_frames(log_root: Path, start_date: str, end_date: str) -> dict[str, pd.DataFrame]:
    days = pd.date_range(start=start_date, end=end_date, freq="D")
    signal_frames: list[pd.DataFrame] = []
    risk_frames: list[pd.DataFrame] = []
    position_frames: list[pd.DataFrame] = []
    for day in days:
        log_dir = log_root / day.strftime("%Y-%m-%d")
        if not log_dir.exists():
            continue
        signal_frames.append(_dedupe(_load(log_dir / "signals.jsonl")))
        risk_frames.append(_dedupe(_load(log_dir / "risk.jsonl")))
        position_frames.append(_dedupe(_load(log_dir / "positions.jsonl")))
    return {
        "signals": pd.concat(signal_frames, ignore_index=True) if signal_frames else pd.DataFrame(),
        "risk": pd.concat(risk_frames, ignore_index=True) if risk_frames else pd.DataFrame(),
        "positions": pd.concat(position_frames, ignore_index=True) if position_frames else pd.DataFrame(),
    }


def _filter_frame_symbols(frame: pd.DataFrame, allowed_symbols: list[str]) -> pd.DataFrame:
    if frame.empty or "symbol" not in frame.columns:
        return frame
    return frame[frame["symbol"].isin(allowed_symbols)].copy()


def _build_live_summary(
    frames: dict[str, pd.DataFrame],
    *,
    allowed_symbols: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    signals = _filter_frame_symbols(frames["signals"], allowed_symbols)
    risk = _filter_frame_symbols(frames["risk"], allowed_symbols)
    positions = _filter_frame_symbols(frames["positions"], allowed_symbols)

    buy_signals = signals[(signals.get("event") == "signal.evaluated") & (signals.get("action") == "BUY")].copy()
    blocked = risk[(risk.get("event") == "risk.check") & (risk.get("allowed") == False)].copy()  # noqa: E712
    closes = positions[positions.get("event") == "position.closed"].copy()

    buy_counts = (
        buy_signals.groupby("symbol")
        .size()
        .rename("live_buy_signals")
        .reset_index()
        if not buy_signals.empty and "symbol" in buy_signals.columns
        else pd.DataFrame(columns=["symbol", "live_buy_signals"])
    )
    close_summary = (
        closes.groupby("symbol", dropna=False)
        .agg(
            live_closed_trades=("symbol", "size"),
            live_realized_pnl=("pnl_usd", "sum"),
            live_win_rate=("winner", "mean"),
        )
        .reset_index()
        if not closes.empty and "symbol" in closes.columns
        else pd.DataFrame(columns=["symbol", "live_closed_trades", "live_realized_pnl", "live_win_rate"])
    )
    blocked_summary = (
        blocked.groupby(["symbol", "block_reason"], dropna=False)
        .size()
        .rename("count")
        .reset_index()
        .sort_values(["symbol", "count", "block_reason"], ascending=[True, False, True])
        if not blocked.empty and {"symbol", "block_reason"} <= set(blocked.columns)
        else pd.DataFrame(columns=["symbol", "block_reason", "count"])
    )

    by_symbol = buy_counts.merge(close_summary, on="symbol", how="outer")
    for column, default in (
        ("live_buy_signals", 0),
        ("live_closed_trades", 0),
        ("live_realized_pnl", 0.0),
        ("live_win_rate", 0.0),
    ):
        if column in by_symbol.columns:
            by_symbol.loc[by_symbol[column].isna(), column] = default
    return by_symbol, blocked_summary


def _build_backtest_summary(backtest_result: dict[str, Any], *, configured_hold_bars: int) -> pd.DataFrame:
    closed = pair_round_trip_trades(backtest_result.get("trades", []), configured_hold_bars=configured_hold_bars)
    if closed.empty:
        return pd.DataFrame(
            columns=[
                "symbol",
                "backtest_closed_trades",
                "backtest_realized_pnl",
                "backtest_win_rate",
                "backtest_avg_holding_bars",
            ]
        )
    return (
        closed.groupby("symbol", dropna=False)
        .agg(
            backtest_closed_trades=("symbol", "size"),
            backtest_realized_pnl=("realized_pnl", "sum"),
            backtest_win_rate=("realized_pnl", lambda values: float((values > 0).mean()) if len(values) else 0.0),
            backtest_avg_holding_bars=("holding_bars", "mean"),
        )
        .reset_index()
    )


def _top_block_reasons(blocked_summary: pd.DataFrame) -> dict[str, list[dict[str, Any]]]:
    if blocked_summary.empty:
        return {}
    top: dict[str, list[dict[str, Any]]] = {}
    for symbol, group in blocked_summary.groupby("symbol", dropna=False):
        rows = []
        for _, row in group.head(3).iterrows():
            rows.append(
                {
                    "block_reason": str(row.get("block_reason") or "unknown"),
                    "count": int(row.get("count", 0) or 0),
                }
            )
        top[str(symbol)] = rows
    return top


def main() -> int:
    args = parse_args()
    config_path = Path(args.config)
    runtime, source = _load_runtime_payload(config_path)
    dataset_raw = args.dataset or source.get("dataset")
    if not dataset_raw:
        raise RuntimeError("Dataset path is required. Pass --dataset or provide source.dataset in the config.")
    dataset_path = Path(str(dataset_raw))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    effective_symbols = _effective_symbols(runtime)

    backtest_kwargs = _build_backtest_kwargs(
        runtime,
        dataset_path=dataset_path,
        start_date=args.start_date,
        end_date=args.end_date,
        commission=args.commission,
        slippage=args.slippage,
    )
    backtest_result = run_backtest(**backtest_kwargs)
    backtest_by_symbol = _build_backtest_summary(backtest_result, configured_hold_bars=0)

    live_frames = _load_live_frames(Path(args.log_root), args.start_date, args.end_date)
    live_by_symbol, blocked_summary = _build_live_summary(live_frames, allowed_symbols=effective_symbols)

    merged = (
        backtest_by_symbol.merge(live_by_symbol, on="symbol", how="outer")
        .sort_values("symbol")
    )
    for column, default in (
        ("backtest_closed_trades", 0),
        ("backtest_realized_pnl", 0.0),
        ("backtest_win_rate", 0.0),
        ("backtest_avg_holding_bars", 0.0),
        ("live_buy_signals", 0),
        ("live_closed_trades", 0),
        ("live_realized_pnl", 0.0),
        ("live_win_rate", 0.0),
    ):
        if column in merged.columns:
            merged.loc[merged[column].isna(), column] = default
    if not merged.empty:
        merged["closed_trade_gap"] = merged["backtest_closed_trades"] - merged["live_closed_trades"]
        merged["realized_pnl_gap"] = merged["backtest_realized_pnl"] - merged["live_realized_pnl"]

    overview = {
        "config_path": str(config_path),
        "dataset_path": str(dataset_path),
        "start_date": args.start_date,
        "end_date": args.end_date,
        "effective_symbols": effective_symbols,
        "backtest": {
            "realized_pnl": float(backtest_result.get("realized_pnl", 0.0) or 0.0),
            "total_return_pct": float(backtest_result.get("total_return_pct", 0.0) or 0.0),
            "win_rate": float(backtest_result.get("win_rate", 0.0) or 0.0),
            "profit_factor": float(backtest_result.get("profit_factor", 0.0) or 0.0),
            "total_trades": int(backtest_result.get("total_trades", 0) or 0),
            "closed_trades": int(merged["backtest_closed_trades"].sum()) if not merged.empty else 0,
        },
        "live": {
            "buy_signals": int(merged["live_buy_signals"].sum()) if not merged.empty else 0,
            "closed_trades": int(merged["live_closed_trades"].sum()) if not merged.empty else 0,
            "realized_pnl": float(merged["live_realized_pnl"].sum()) if not merged.empty else 0.0,
        },
        "gaps": {
            "closed_trade_gap": int(merged["closed_trade_gap"].sum()) if not merged.empty else 0,
            "realized_pnl_gap": float(merged["realized_pnl_gap"].sum()) if not merged.empty else 0.0,
        },
        "top_live_block_reasons": _top_block_reasons(blocked_summary),
    }

    merged.to_csv(output_dir / "by_symbol.csv", index=False)
    blocked_summary.to_csv(output_dir / "live_block_reasons.csv", index=False)
    (output_dir / "overview.json").write_text(json.dumps(overview, indent=2), encoding="utf-8")

    print(json.dumps(overview, indent=2))
    print(f"Saved by-symbol report to {output_dir / 'by_symbol.csv'}")
    print(f"Saved block-reason report to {output_dir / 'live_block_reasons.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

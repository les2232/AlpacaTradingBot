from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from backtest_runner import run_backtest
from research.run_trade_path_diagnostics import pair_round_trip_trades


DEFAULT_CONFIG_PATH = Path("config") / "live_config.json"
DEFAULT_OUTPUT_DIR = Path("results") / "mean_reversion_exit_comparison"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare mean-reversion exit styles using the current runtime config as the baseline."
    )
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH), help="Path to runtime config JSON.")
    parser.add_argument("--dataset", help="Optional dataset override. Defaults to config source.dataset.")
    parser.add_argument("--start-date", required=True, help="UTC start date (YYYY-MM-DD).")
    parser.add_argument("--end-date", required=True, help="UTC end date (YYYY-MM-DD).")
    parser.add_argument("--exit-styles", default="sma,eod", help="Comma-separated exit styles to compare.")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR), help="Directory for CSV/JSON artifacts.")
    parser.add_argument("--commission", type=float, default=0.01, help="Per-order commission.")
    parser.add_argument("--slippage", type=float, default=0.05, help="Per-share slippage.")
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
    exit_style: str,
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
        "mean_reversion_exit_style": exit_style,
        "mean_reversion_max_atr_percentile": float(runtime.get("mean_reversion_max_atr_percentile", 0.0) or 0.0),
        "strategy_mode": str(runtime.get("strategy_mode", "mean_reversion") or "mean_reversion"),
        "symbol_strategy_modes": runtime.get("symbol_strategy_modes"),
        "ml_probability_buy": float(runtime.get("ml_probability_buy", 0.55) or 0.55),
        "ml_probability_sell": float(runtime.get("ml_probability_sell", 0.45) or 0.45),
        "start_date": start_date,
        "end_date": end_date,
    }


def _parse_exit_styles(raw_value: str) -> list[str]:
    styles = []
    seen: set[str] = set()
    for chunk in raw_value.split(","):
        style = chunk.strip().lower()
        if not style or style in seen:
            continue
        styles.append(style)
        seen.add(style)
    if not styles:
        raise RuntimeError("At least one exit style is required.")
    return styles


def _summarize_result(exit_style: str, result: dict[str, Any]) -> dict[str, Any]:
    closed = pair_round_trip_trades(result.get("trades", []), configured_hold_bars=0)
    return {
        "exit_style": exit_style,
        "realized_pnl": float(result.get("realized_pnl", 0.0) or 0.0),
        "total_return_pct": float(result.get("total_return_pct", 0.0) or 0.0),
        "win_rate": float(result.get("win_rate", 0.0) or 0.0),
        "profit_factor": float(result.get("profit_factor", 0.0) or 0.0),
        "max_drawdown_pct": float(result.get("max_drawdown_pct", 0.0) or 0.0),
        "closed_trades": int(len(closed)),
        "avg_holding_bars": float(closed["holding_bars"].mean()) if not closed.empty else 0.0,
    }


def _summarize_per_symbol(exit_style: str, result: dict[str, Any]) -> pd.DataFrame:
    closed = pair_round_trip_trades(result.get("trades", []), configured_hold_bars=0)
    if closed.empty:
        return pd.DataFrame(columns=["exit_style", "symbol", "closed_trades", "realized_pnl", "win_rate", "avg_holding_bars"])
    summary = (
        closed.groupby("symbol", dropna=False)
        .agg(
            closed_trades=("symbol", "size"),
            realized_pnl=("realized_pnl", "sum"),
            win_rate=("realized_pnl", lambda values: float((values > 0).mean()) if len(values) else 0.0),
            avg_holding_bars=("holding_bars", "mean"),
        )
        .reset_index()
    )
    summary.insert(0, "exit_style", exit_style)
    return summary


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

    summaries: list[dict[str, Any]] = []
    per_symbol_frames: list[pd.DataFrame] = []
    for exit_style in _parse_exit_styles(args.exit_styles):
        result = run_backtest(
            **_build_backtest_kwargs(
                runtime,
                dataset_path=dataset_path,
                start_date=args.start_date,
                end_date=args.end_date,
                commission=args.commission,
                slippage=args.slippage,
                exit_style=exit_style,
            )
        )
        summaries.append(_summarize_result(exit_style, result))
        per_symbol_frames.append(_summarize_per_symbol(exit_style, result))

    summary_df = pd.DataFrame(summaries).sort_values(["realized_pnl", "profit_factor", "exit_style"], ascending=[False, False, True])
    per_symbol_df = pd.concat(per_symbol_frames, ignore_index=True) if per_symbol_frames else pd.DataFrame()
    best = summary_df.iloc[0].to_dict() if not summary_df.empty else {}

    overview = {
        "config_path": str(config_path),
        "dataset_path": str(dataset_path),
        "start_date": args.start_date,
        "end_date": args.end_date,
        "effective_symbols": _effective_symbols(runtime),
        "baseline_exit_style": str(runtime.get("mean_reversion_exit_style", "sma") or "sma"),
        "best_exit_style": str(best.get("exit_style") or ""),
        "best_realized_pnl": float(best.get("realized_pnl", 0.0) or 0.0),
    }

    summary_df.to_csv(output_dir / "summary.csv", index=False)
    per_symbol_df.to_csv(output_dir / "per_symbol.csv", index=False)
    (output_dir / "overview.json").write_text(json.dumps(overview, indent=2), encoding="utf-8")

    print(summary_df.to_string(index=False))
    print(f"Saved summary to {output_dir / 'summary.csv'}")
    print(f"Saved per-symbol report to {output_dir / 'per_symbol.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

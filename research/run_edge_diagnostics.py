from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from run_edge_audit import (
    VARIANT_CHOICES,
    VARIANT_EXPLICIT_CONFIG,
    VARIANT_LIVE_EFFECTIVE,
    _backtest_exit_summary,
    _frame_text,
    _load_runtime_payload,
    _parse_horizons,
    _print_strategy_summary,
    _prepare_inputs_and_state,
    build_audit_context,
    evaluate_raw_entry_signals,
)
from backtest_runner import run_backtest
from strategy import STRATEGY_MODE_MEAN_REVERSION


DEFAULT_CONFIG_PATH = Path("config") / "live_config.json"


@dataclass(frozen=True)
class DiagnosticRun:
    variant: str
    context: Any
    evaluations_df: pd.DataFrame
    signals_df: pd.DataFrame
    backtest_result: dict[str, Any]
    realized_per_symbol_df: pd.DataFrame


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run focused second-stage diagnostics on the current strategy: fixed-horizon exits, "
            "symbol quality, regime splits, and live-vs-config behavior comparison."
        )
    )
    parser.add_argument("--dataset", help="Path to dataset directory. Defaults to config source.dataset.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH), help="Path to runtime config JSON.")
    parser.add_argument("--symbols", nargs="*", help="Optional symbol override.")
    parser.add_argument("--start-date", help="Optional UTC date filter (YYYY-MM-DD).")
    parser.add_argument("--end-date", help="Optional UTC date filter (YYYY-MM-DD).")
    parser.add_argument(
        "--horizons",
        default="1,2,4",
        help="Comma-separated fixed exit horizons in bars (default: 1,2,4).",
    )
    parser.add_argument("--commission-per-order", type=float, default=0.01)
    parser.add_argument("--slippage-per-share", type=float, default=0.05)
    parser.add_argument("--position-size", type=float, default=1000.0)
    parser.add_argument(
        "--variant",
        default="both",
        choices=["both", *VARIANT_CHOICES],
        help="Behavior variant to run: live_effective, explicit_config, or both (default).",
    )
    parser.add_argument("--output-dir", help="Optional directory to save structured artifacts.")
    return parser.parse_args()


def _variant_list(raw_variant: str) -> tuple[str, ...]:
    if raw_variant == "both":
        return (VARIANT_LIVE_EFFECTIVE, VARIANT_EXPLICIT_CONFIG)
    return (raw_variant,)


def _profit_factor(values: pd.Series) -> float:
    if values.empty:
        return 0.0
    gross_profit = float(values[values > 0].sum())
    gross_loss = abs(float(values[values < 0].sum()))
    if gross_loss == 0.0:
        return float("inf") if gross_profit > 0 else 0.0
    return gross_profit / gross_loss


def summarize_fixed_horizon_metrics(
    signals_df: pd.DataFrame,
    *,
    group_cols: list[str],
    horizons: tuple[int, ...],
) -> pd.DataFrame:
    columns = group_cols + [
        "horizon_bars",
        "sample_count",
        "avg_gross_return_pct",
        "median_gross_return_pct",
        "gross_win_rate_pct",
        "avg_net_expectancy_pct",
        "median_net_return_pct",
        "net_win_rate_pct",
        "net_profit_factor",
    ]
    if signals_df.empty:
        return pd.DataFrame(columns=columns)

    rows: list[dict[str, Any]] = []
    grouped = [((), signals_df)] if not group_cols else signals_df.groupby(group_cols, dropna=False, sort=True)
    for group_key, group_df in grouped:
        key_values = group_key if isinstance(group_key, tuple) else (group_key,)
        key_map = {col: key_values[idx] for idx, col in enumerate(group_cols)}
        for horizon in horizons:
            gross = pd.to_numeric(group_df[f"fwd_{horizon}b_gross_pct"], errors="coerce").dropna()
            net = pd.to_numeric(group_df[f"fwd_{horizon}b_net_pct"], errors="coerce").dropna()
            rows.append({
                **key_map,
                "horizon_bars": horizon,
                "sample_count": int(len(net)),
                "avg_gross_return_pct": float(gross.mean()) if not gross.empty else 0.0,
                "median_gross_return_pct": float(gross.median()) if not gross.empty else 0.0,
                "gross_win_rate_pct": float((gross > 0).mean() * 100.0) if not gross.empty else 0.0,
                "avg_net_expectancy_pct": float(net.mean()) if not net.empty else 0.0,
                "median_net_return_pct": float(net.median()) if not net.empty else 0.0,
                "net_win_rate_pct": float((net > 0).mean() * 100.0) if not net.empty else 0.0,
                "net_profit_factor": _profit_factor(net),
            })
    return pd.DataFrame(rows)


def build_symbol_quality_ranking(
    fixed_horizon_by_symbol: pd.DataFrame,
    realized_per_symbol_df: pd.DataFrame,
) -> pd.DataFrame:
    columns = [
        "symbol",
        "best_horizon_bars",
        "best_net_expectancy_pct",
        "avg_net_expectancy_pct",
        "avg_gross_return_pct",
        "best_net_profit_factor",
        "best_net_win_rate_pct",
        "realized_pnl",
        "realized_expectancy",
        "realized_trades",
    ]
    if fixed_horizon_by_symbol.empty:
        return pd.DataFrame(columns=columns)

    rows: list[dict[str, Any]] = []
    realized_lookup = (
        realized_per_symbol_df.set_index("symbol").to_dict("index")
        if not realized_per_symbol_df.empty
        else {}
    )
    for symbol, symbol_df in fixed_horizon_by_symbol.groupby("symbol", sort=True):
        ranked = symbol_df.sort_values(
            ["avg_net_expectancy_pct", "net_profit_factor", "horizon_bars"],
            ascending=[False, False, True],
        )
        best = ranked.iloc[0]
        realized = realized_lookup.get(symbol, {})
        rows.append({
            "symbol": symbol,
            "best_horizon_bars": int(best["horizon_bars"]),
            "best_net_expectancy_pct": float(best["avg_net_expectancy_pct"]),
            "avg_net_expectancy_pct": float(symbol_df["avg_net_expectancy_pct"].mean()),
            "avg_gross_return_pct": float(symbol_df["avg_gross_return_pct"].mean()),
            "best_net_profit_factor": float(best["net_profit_factor"]),
            "best_net_win_rate_pct": float(best["net_win_rate_pct"]),
            "realized_pnl": float(realized.get("realized_pnl", 0.0)),
            "realized_expectancy": float(realized.get("expectancy", 0.0)),
            "realized_trades": int(realized.get("trades", 0)),
        })
    return pd.DataFrame(rows).sort_values(
        ["best_net_expectancy_pct", "realized_pnl", "symbol"],
        ascending=[False, False, True],
    )


def summarize_variant_comparison(runs: list[DiagnosticRun], horizons: tuple[int, ...]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for run in runs:
        overall = summarize_fixed_horizon_metrics(run.signals_df, group_cols=[], horizons=horizons)
        ranked = overall.sort_values(
            ["avg_net_expectancy_pct", "horizon_bars"],
            ascending=[False, True],
        )
        best = ranked.iloc[0] if not ranked.empty else None
        rows.append({
            "variant": run.variant,
            "signals": int(len(run.signals_df)),
            "realized_pnl": float(run.backtest_result.get("realized_pnl", 0.0)),
            "realized_expectancy": float(run.backtest_result.get("expectancy", 0.0)),
            "realized_win_rate": float(run.backtest_result.get("win_rate", 0.0)),
            "realized_trades": int(run.backtest_result.get("total_trades", 0)),
            "best_horizon_bars": int(best["horizon_bars"]) if best is not None else 0,
            "best_fixed_net_expectancy_pct": float(best["avg_net_expectancy_pct"]) if best is not None else 0.0,
            "best_fixed_gross_return_pct": float(best["avg_gross_return_pct"]) if best is not None else 0.0,
        })
    return pd.DataFrame(rows)


def compare_variant_parameters(runs: list[DiagnosticRun]) -> pd.DataFrame:
    if len(runs) < 2:
        return pd.DataFrame(columns=["field", "live_effective", "explicit_config"])
    by_variant = {run.variant: run.context.backtest_kwargs for run in runs}
    left = by_variant.get(VARIANT_LIVE_EFFECTIVE, {})
    right = by_variant.get(VARIANT_EXPLICIT_CONFIG, {})
    changed_fields = sorted(set(left) | set(right))
    rows = []
    for field in changed_fields:
        left_value = left.get(field)
        right_value = right.get(field)
        if left_value != right_value:
            rows.append({
                "field": field,
                "live_effective": left_value,
                "explicit_config": right_value,
            })
    return pd.DataFrame(rows)


def _slice_tables(signals_df: pd.DataFrame, horizons: tuple[int, ...]) -> dict[str, pd.DataFrame]:
    return {
        "overall": summarize_fixed_horizon_metrics(signals_df, group_cols=[], horizons=horizons),
        "by_symbol": summarize_fixed_horizon_metrics(signals_df, group_cols=["symbol"], horizons=horizons),
        "by_trend_proxy": summarize_fixed_horizon_metrics(signals_df, group_cols=["trend_proxy"], horizons=horizons),
        "by_volatility_regime": summarize_fixed_horizon_metrics(signals_df, group_cols=["volatility_regime"], horizons=horizons),
        "by_time_bucket": summarize_fixed_horizon_metrics(signals_df, group_cols=["time_bucket"], horizons=horizons),
        "by_month": summarize_fixed_horizon_metrics(signals_df, group_cols=["month"], horizons=horizons),
    }


def _interpret_concentration(symbol_ranking_df: pd.DataFrame) -> str:
    if symbol_ranking_df.empty:
        return "No signals fired."
    positive = int((symbol_ranking_df["best_net_expectancy_pct"] > 0).sum())
    total = len(symbol_ranking_df)
    if positive == 0:
        return "No symbol shows positive fixed-horizon net expectancy."
    if positive <= max(2, total // 4):
        return "Edge appears concentrated in a narrow subset of symbols."
    if positive >= max(1, int(total * 0.6)):
        return "Edge appears relatively broad across the symbol set."
    return "Edge appears mixed: some symbols help, several others dilute it."


def _save_variant_outputs(
    run: DiagnosticRun,
    *,
    output_dir: Path,
    slice_tables: dict[str, pd.DataFrame],
    symbol_ranking_df: pd.DataFrame,
) -> None:
    variant_dir = output_dir / run.variant
    variant_dir.mkdir(parents=True, exist_ok=True)
    run.evaluations_df.to_csv(variant_dir / "entry_evaluations.csv", index=False)
    run.signals_df.to_csv(variant_dir / "entry_signals.csv", index=False)
    run.realized_per_symbol_df.to_csv(variant_dir / "realized_per_symbol.csv", index=False)
    symbol_ranking_df.to_csv(variant_dir / "symbol_quality_ranking.csv", index=False)
    for name, table in slice_tables.items():
        table.to_csv(variant_dir / f"{name}.csv", index=False)


def _print_variant_section(run: DiagnosticRun, horizons: tuple[int, ...]) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    print(f"\n=== Variant: {run.variant} ===")
    _print_strategy_summary(run.context)
    slice_tables = _slice_tables(run.signals_df, horizons)
    symbol_ranking_df = build_symbol_quality_ranking(slice_tables["by_symbol"], run.realized_per_symbol_df)

    print("\nFixed-horizon study:")
    print(_frame_text(slice_tables["overall"]))
    print("\nFixed-horizon by symbol:")
    print(_frame_text(slice_tables["by_symbol"], max_rows=50))
    print("\nTop symbols:")
    print(_frame_text(symbol_ranking_df.head(8)))
    print("\nBottom symbols:")
    print(_frame_text(symbol_ranking_df.tail(8).sort_values("best_net_expectancy_pct")))
    print(f"\nSymbol concentration: {_interpret_concentration(symbol_ranking_df)}")
    print("\nTrend/range regimes:")
    print(_frame_text(slice_tables["by_trend_proxy"], max_rows=24))
    print("\nVolatility regimes:")
    print(_frame_text(slice_tables["by_volatility_regime"], max_rows=24))
    print("\nTime-of-day buckets:")
    print(_frame_text(slice_tables["by_time_bucket"], max_rows=24))
    print("\nMonthly buckets:")
    print(_frame_text(slice_tables["by_month"], max_rows=24))
    return slice_tables, symbol_ranking_df


def _run_variant(
    *,
    config_path: Path,
    payload: dict[str, Any],
    args: argparse.Namespace,
    variant: str,
    horizons: tuple[int, ...],
) -> DiagnosticRun:
    source_dataset = payload["source"].get("dataset") if isinstance(payload["source"].get("dataset"), str) else None
    context = build_audit_context(
        config_path=config_path,
        runtime=dict(payload["runtime"]),
        source_dataset=source_dataset,
        dataset_override=args.dataset,
        symbols_override=args.symbols,
        start_date=args.start_date,
        end_date=args.end_date,
        commission_per_order=args.commission_per_order,
        slippage_per_share=args.slippage_per_share,
        position_size=args.position_size,
        output_dir=args.output_dir,
        variant=variant,
    )
    inputs, state = _prepare_inputs_and_state(context)
    evaluations_df, signals_df = evaluate_raw_entry_signals(
        inputs=inputs,
        state=state,
        sma_bars=context.backtest_kwargs["sma_bars"],
        time_window_mode=context.backtest_kwargs["time_window_mode"],
        slippage=context.backtest_kwargs["slippage"],
        commission=context.backtest_kwargs["commission"],
        position_size=context.backtest_kwargs["position_size"],
        horizons=horizons,
    )
    backtest_result = run_backtest(context.dataset_path, **context.backtest_kwargs)
    realized_per_symbol_df = _backtest_exit_summary(backtest_result)
    return DiagnosticRun(
        variant=variant,
        context=context,
        evaluations_df=evaluations_df,
        signals_df=signals_df,
        backtest_result=backtest_result,
        realized_per_symbol_df=realized_per_symbol_df,
    )


def main() -> None:
    args = parse_args()
    horizons = _parse_horizons(args.horizons)
    config_path = Path(args.config)
    payload = _load_runtime_payload(config_path)
    runs = [
        _run_variant(
            config_path=config_path,
            payload=payload,
            args=args,
            variant=variant,
            horizons=horizons,
        )
        for variant in _variant_list(args.variant)
    ]

    saved_outputs: list[str] = []
    for run in runs:
        slice_tables, symbol_ranking_df = _print_variant_section(run, horizons)
        if args.output_dir:
            _save_variant_outputs(
                run,
                output_dir=Path(args.output_dir),
                slice_tables=slice_tables,
                symbol_ranking_df=symbol_ranking_df,
            )
            saved_outputs.append(str(Path(args.output_dir) / run.variant))

    if len(runs) > 1:
        print("\n=== Variant Comparison ===")
        parameter_diff_df = compare_variant_parameters(runs)
        print("\nBehavior parameter differences:")
        print(_frame_text(parameter_diff_df))
        variant_summary_df = summarize_variant_comparison(runs, horizons)
        print("\nVariant metric summary:")
        print(_frame_text(variant_summary_df))

        explicit_run = next(run for run in runs if run.variant == VARIANT_EXPLICIT_CONFIG)
        live_run = next(run for run in runs if run.variant == VARIANT_LIVE_EFFECTIVE)
        if explicit_run.context.backtest_kwargs["strategy_mode"] == STRATEGY_MODE_MEAN_REVERSION:
            print(
                "\nLive mismatch note:\n"
                "The config file does not explicitly set VWAP/ADX mean-reversion fields, but the live bot constructs "
                "StrategyConfig without passing those fields, so live inherits StrategyConfig defaults "
                "(VWAP z-entry 1.5, min ATR percentile 20, max ADX 25). The explicit-config variant treats omitted "
                "fields as neutral/backtest-style defaults instead."
            )
            print(
                "Current run impact:\n"
                f"live_effective signals={len(live_run.signals_df)} vs explicit_config signals={len(explicit_run.signals_df)}; "
                f"realized_pnl={live_run.backtest_result.get('realized_pnl', 0.0):.2f} vs "
                f"{explicit_run.backtest_result.get('realized_pnl', 0.0):.2f}."
            )

    if saved_outputs:
        print("\nSaved outputs:")
        for path in saved_outputs:
            print(path)


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from inspect_dataset_and_run import (
    build_warnings,
    format_inspection_output,
    inspect_dataset,
    inspect_run_configuration,
)
from research.experiment_log import log_experiment_run
from run_edge_audit import _frame_text, _print_strategy_summary
from run_edge_diagnostics import summarize_fixed_horizon_metrics
from run_volatility_expansion_validation import (
    DEFAULT_CONFIG_PATH,
    STRATEGY_MODE_VOLATILITY_EXPANSION,
    ValidationRun,
    _average_holding_period,
    _annotate_compression_quality,
    _build_closed_trades_df,
    _load_config_runtime,
    _normalize_symbol_list,
    _resolve_symbols,
    _run_validation_scenario,
    _runtime_with_overrides,
    attach_signal_context,
    summarize_slice_validation,
)


DEFAULT_FORWARD_HORIZONS = (4, 6, 8, 10, 15, 20, 30)
DEFAULT_HOLD_HORIZONS = (4, 8, 10, 15, 20, 30)
DEFAULT_STRATEGY_MODE = STRATEGY_MODE_VOLATILITY_EXPANSION


@dataclass(frozen=True)
class HoldRun:
    hold_bars: int
    run: ValidationRun


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a longer-horizon validation pass for existing strategy logic."
    )
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH), help="Runtime config JSON.")
    parser.add_argument(
        "--strategy",
        default=DEFAULT_STRATEGY_MODE,
        help="Strategy mode to evaluate. Default: volatility_expansion.",
    )
    parser.add_argument("--dataset", help="Dataset directory. Defaults to config source.dataset.")
    parser.add_argument("--symbols", nargs="*", help="Optional symbol override.")
    parser.add_argument("--start-date", help="Optional UTC date filter (YYYY-MM-DD).")
    parser.add_argument("--end-date", help="Optional UTC date filter (YYYY-MM-DD).")
    parser.add_argument(
        "--forward-horizons",
        default="4,6,8,10,15,20,30",
        help="Comma-separated forward-return horizons in bars.",
    )
    parser.add_argument(
        "--hold-horizons",
        default="4,8,10,15,20,30",
        help="Comma-separated realized hold horizons in bars.",
    )
    parser.add_argument("--commission-per-order", type=float, default=0.01)
    parser.add_argument("--slippage-per-share", type=float, default=0.05)
    parser.add_argument("--position-size", type=float, default=1000.0)
    parser.add_argument(
        "--sections",
        default="all",
        help="Comma-separated subset: all,raw,holds,slices.",
    )
    parser.add_argument(
        "--debug-data",
        action="store_true",
        help="Print dataset/config sanity-check summary before running the validation.",
    )
    parser.add_argument(
        "--inspect-only",
        action="store_true",
        help="Print dataset/config sanity-check summary and exit without running the validation.",
    )
    parser.add_argument("--output-dir", help="Optional directory for CSV/JSON artifacts.")
    return parser.parse_args()


def _parse_horizon_list(raw: str) -> tuple[int, ...]:
    values = tuple(int(part.strip()) for part in raw.split(",") if part.strip())
    if not values or any(value <= 0 for value in values):
        raise ValueError("Horizons must be a comma-separated list of positive integers.")
    return tuple(sorted(set(values)))


def _strategy_hold_key(strategy_mode: str) -> str:
    if strategy_mode == "volatility_expansion":
        return "volatility_expansion_hold_bars"
    if strategy_mode == "trend_pullback":
        return "trend_pullback_hold_bars"
    if strategy_mode == "momentum_breakout":
        return "momentum_breakout_hold_bars"
    raise RuntimeError(f"Unsupported long-horizon strategy: {strategy_mode}")


def _run_long_horizon_scenario(
    *,
    config_path: Path,
    runtime: dict[str, Any],
    source_dataset: str | None,
    dataset_override: str | None,
    symbols: tuple[str, ...],
    start_date: str | None,
    end_date: str | None,
    commission_per_order: float,
    slippage_per_share: float,
    position_size: float,
    output_dir: str | None,
    variant_name: str,
    horizons: tuple[int, ...],
) -> ValidationRun:
    run = _run_validation_scenario(
        config_path=config_path,
        runtime=runtime,
        source_dataset=source_dataset,
        dataset_override=dataset_override,
        symbols=symbols,
        start_date=start_date,
        end_date=end_date,
        commission_per_order=commission_per_order,
        slippage_per_share=slippage_per_share,
        position_size=position_size,
        output_dir=output_dir,
        variant_name=variant_name,
    )
    if horizons == (1, 2, 3, 4, 6, 8):
        return run

    from run_edge_audit import _prepare_inputs_and_state, build_audit_context, evaluate_raw_entry_signals

    context = build_audit_context(
        config_path=config_path,
        runtime=runtime,
        source_dataset=source_dataset,
        dataset_override=dataset_override,
        symbols_override=list(symbols),
        start_date=start_date,
        end_date=end_date,
        commission_per_order=commission_per_order,
        slippage_per_share=slippage_per_share,
        position_size=position_size,
        output_dir=output_dir,
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
    if context.backtest_kwargs["strategy_mode"] == STRATEGY_MODE_VOLATILITY_EXPANSION:
        signals_df = _annotate_compression_quality(signals_df, state)
    hold_bars = int(
        context.backtest_kwargs.get("volatility_expansion_hold_bars")
        or context.backtest_kwargs.get("momentum_breakout_hold_bars")
        or context.backtest_kwargs.get("trend_pullback_hold_bars")
        or 0
    )
    closed_trades_df = attach_signal_context(_build_closed_trades_df(run.backtest_result, hold_bars), signals_df, horizons)
    return ValidationRun(
        name=run.name,
        strategy_mode=run.strategy_mode,
        context=context,
        evaluations_df=evaluations_df,
        signals_df=signals_df,
        backtest_result=run.backtest_result,
        closed_trades_df=closed_trades_df,
    )


def build_hold_horizon_specs(strategy_mode: str, hold_horizons: tuple[int, ...]) -> list[dict[str, Any]]:
    hold_key = _strategy_hold_key(strategy_mode)
    return [{"hold_bars": hold_bars, "runtime_overrides": {hold_key: hold_bars}} for hold_bars in hold_horizons]


def summarize_hold_ladder(hold_runs: list[HoldRun]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for item in hold_runs:
        result = item.run.backtest_result
        rows.append(
            {
                "hold_bars": item.hold_bars,
                "signal_count": int(len(item.run.signals_df)),
                "trade_count": int(len(item.run.closed_trades_df)),
                "realized_pnl": float(result.get("realized_pnl", 0.0)),
                "expectancy": float(result.get("expectancy", 0.0)),
                "win_rate": float(result.get("win_rate", 0.0)),
                "profit_factor": float(result.get("profit_factor", 0.0)),
                "max_drawdown_pct": float(result.get("max_drawdown_pct", 0.0)),
                "avg_holding_period": _average_holding_period(item.run.closed_trades_df),
            }
        )
    return pd.DataFrame(rows).sort_values("hold_bars") if rows else pd.DataFrame(
        columns=[
            "hold_bars",
            "signal_count",
            "trade_count",
            "realized_pnl",
            "expectancy",
            "win_rate",
            "profit_factor",
            "max_drawdown_pct",
            "avg_holding_period",
        ]
    )


def choose_representative_horizon(horizons: tuple[int, ...]) -> int:
    if 20 in horizons:
        return 20
    return max(horizons)


def classify_long_horizon(
    *,
    raw_overall: pd.DataFrame,
    hold_ladder: pd.DataFrame,
) -> tuple[str, str]:
    if raw_overall.empty:
        return "not ready", "No signals were generated for the requested horizon study."

    positive_long_raw = raw_overall[raw_overall["avg_net_expectancy_pct"] > 0.0]
    best_raw_row = raw_overall.sort_values(
        ["avg_net_expectancy_pct", "horizon_bars"],
        ascending=[False, True],
    ).iloc[0]
    best_raw_horizon = int(best_raw_row["horizon_bars"])
    best_raw = float(best_raw_row["avg_net_expectancy_pct"])

    if hold_ladder.empty:
        best_expectancy = 0.0
        best_hold = 0
    else:
        best_hold_row = hold_ladder.sort_values(["expectancy", "hold_bars"], ascending=[False, True]).iloc[0]
        best_expectancy = float(best_hold_row["expectancy"])
        best_hold = int(best_hold_row["hold_bars"])

    if positive_long_raw.empty and best_expectancy <= 0:
        return (
            "reject",
            f"Raw forward returns stay negative through the longer horizon ladder and realized expectancy remains non-positive; best raw horizon was {best_raw_horizon} bars at {best_raw:.3f}%.",
        )
    if positive_long_raw.empty:
        return (
            "keep as research-only",
            f"Realized results improved somewhat, but raw forward returns never turned positive through the longer horizon ladder; best realized hold was {best_hold} bars.",
        )
    if best_expectancy <= 0:
        return (
            "keep as research-only",
            f"Longer raw horizons improved enough to produce a positive slice, but realized expectancy stayed non-positive; best raw horizon was {best_raw_horizon} bars at {best_raw:.3f}%.",
        )
    return (
        "longer-horizon version is promising enough for deeper research",
        f"Raw and realized performance both improved with longer horizons; best raw horizon was {best_raw_horizon} bars and best realized hold was {best_hold} bars.",
    )


def _print_results(
    *,
    baseline_run: ValidationRun,
    raw_overall: pd.DataFrame,
    raw_by_symbol: pd.DataFrame,
    raw_by_time_bucket: pd.DataFrame,
    raw_by_month: pd.DataFrame,
    hold_ladder: pd.DataFrame,
    slice_tables: dict[str, pd.DataFrame],
    classification: tuple[str, str],
) -> None:
    print("\n=== Long Horizon Validation ===")
    _print_strategy_summary(baseline_run.context)
    print("\nSignal generation:")
    print(f"Bars evaluated: {len(baseline_run.evaluations_df)}")
    print(f"Signals fired:  {len(baseline_run.signals_df)}")
    print(f"Closed trades:  {len(baseline_run.closed_trades_df)}")
    print("\nForward returns overall:")
    print(_frame_text(raw_overall, max_rows=32))
    print("\nForward returns by symbol:")
    print(_frame_text(raw_by_symbol, max_rows=64))
    print("\nForward returns by time bucket:")
    print(_frame_text(raw_by_time_bucket, max_rows=32))
    print("\nForward returns by month:")
    print(_frame_text(raw_by_month, max_rows=32))
    print("\nLong-hold realized ladder:")
    print(_frame_text(hold_ladder, max_rows=32))
    print("\nSlice summaries (representative longer horizon):")
    for title, table in slice_tables.items():
        print(f"\n{title}:")
        print(_frame_text(table, max_rows=32))
    print(f"\nDiagnosis: {classification[0]}")
    print(classification[1])


def _save_outputs(
    output_dir: Path,
    *,
    baseline_run: ValidationRun,
    raw_overall: pd.DataFrame,
    raw_by_symbol: pd.DataFrame,
    raw_by_time_bucket: pd.DataFrame,
    raw_by_month: pd.DataFrame,
    hold_ladder: pd.DataFrame,
    slice_tables: dict[str, pd.DataFrame],
    classification: tuple[str, str],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    baseline_run.evaluations_df.to_csv(output_dir / "entry_evaluations.csv", index=False)
    baseline_run.signals_df.to_csv(output_dir / "entry_signals.csv", index=False)
    baseline_run.closed_trades_df.to_csv(output_dir / "closed_trades.csv", index=False)
    raw_overall.to_csv(output_dir / "forward_returns_overall.csv", index=False)
    raw_by_symbol.to_csv(output_dir / "forward_returns_by_symbol.csv", index=False)
    raw_by_time_bucket.to_csv(output_dir / "forward_returns_by_time_bucket.csv", index=False)
    raw_by_month.to_csv(output_dir / "forward_returns_by_month.csv", index=False)
    hold_ladder.to_csv(output_dir / "hold_ladder.csv", index=False)
    for name, table in slice_tables.items():
        table.to_csv(output_dir / f"{name}.csv", index=False)
    payload = {
        "strategy_mode": baseline_run.strategy_mode,
        "classification": classification[0],
        "reason": classification[1],
        "forward_horizons": raw_overall["horizon_bars"].tolist() if not raw_overall.empty else [],
        "hold_horizons": hold_ladder["hold_bars"].tolist() if not hold_ladder.empty else [],
    }
    (output_dir / "summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main() -> None:
    args = parse_args()
    sections = {chunk.strip().lower() for chunk in args.sections.split(",") if chunk.strip()}
    run_all = "all" in sections

    config_path = Path(args.config)
    base_runtime, source_dataset = _load_config_runtime(config_path)
    base_runtime["strategy_mode"] = args.strategy
    symbols = _resolve_symbols(base_runtime, args.symbols)
    if not symbols:
        raise RuntimeError("No symbols resolved for long-horizon validation.")

    if args.debug_data or args.inspect_only:
        dataset_path = Path(args.dataset) if args.dataset else (Path(source_dataset) if source_dataset else None)
        if dataset_path is None:
            raise RuntimeError("Dataset path not provided and config source.dataset is missing.")
        dataset_info = inspect_dataset(dataset_path)
        run_info = inspect_run_configuration(
            dataset_info=dataset_info,
            config_path=config_path,
            strategy_override=args.strategy,
            cli_symbols=list(symbols),
            expected_timeframe=None,
            hold_bars_override=args.hold_horizons,
        )
        warnings = build_warnings(
            dataset_info=dataset_info,
            run_info=run_info,
            expected_timeframe=None,
            signals_df=pd.DataFrame(),
            trades_df=pd.DataFrame(),
        )
        print(format_inspection_output(dataset_info=dataset_info, run_info=run_info, warnings=warnings))
        if args.inspect_only:
            return

    forward_horizons = _parse_horizon_list(args.forward_horizons)
    hold_horizons = _parse_horizon_list(args.hold_horizons)

    baseline_runtime = _runtime_with_overrides(base_runtime, {}, symbols)
    baseline_run = _run_long_horizon_scenario(
        config_path=config_path,
        runtime=baseline_runtime,
        source_dataset=source_dataset,
        dataset_override=args.dataset,
        symbols=symbols,
        start_date=args.start_date,
        end_date=args.end_date,
        commission_per_order=args.commission_per_order,
        slippage_per_share=args.slippage_per_share,
        position_size=args.position_size,
        output_dir=args.output_dir,
        variant_name=f"{args.strategy}_long_horizon_baseline",
        horizons=forward_horizons,
    )

    raw_overall = summarize_fixed_horizon_metrics(baseline_run.signals_df, group_cols=[], horizons=forward_horizons)
    raw_by_symbol = summarize_fixed_horizon_metrics(baseline_run.signals_df, group_cols=["symbol"], horizons=forward_horizons)
    raw_by_time_bucket = summarize_fixed_horizon_metrics(baseline_run.signals_df, group_cols=["time_bucket"], horizons=forward_horizons)
    raw_by_month = summarize_fixed_horizon_metrics(baseline_run.signals_df, group_cols=["month"], horizons=forward_horizons)

    hold_runs: list[HoldRun] = []
    if run_all or "holds" in sections:
        for spec in build_hold_horizon_specs(args.strategy, hold_horizons):
            runtime = _runtime_with_overrides(base_runtime, spec["runtime_overrides"], symbols)
            hold_runs.append(
                HoldRun(
                    hold_bars=int(spec["hold_bars"]),
                    run=_run_long_horizon_scenario(
                        config_path=config_path,
                        runtime=runtime,
                        source_dataset=source_dataset,
                        dataset_override=args.dataset,
                        symbols=symbols,
                        start_date=args.start_date,
                        end_date=args.end_date,
                        commission_per_order=args.commission_per_order,
                        slippage_per_share=args.slippage_per_share,
                        position_size=args.position_size,
                        output_dir=args.output_dir,
                        variant_name=f"{args.strategy}_hold_{int(spec['hold_bars'])}",
                        horizons=forward_horizons,
                    ),
                )
            )
    hold_ladder = summarize_hold_ladder(hold_runs)

    representative_horizon = choose_representative_horizon(forward_horizons)
    representative_hold = next((item.run.closed_trades_df for item in hold_runs if item.hold_bars == representative_horizon), baseline_run.closed_trades_df)
    slice_tables = {}
    if run_all or "slices" in sections:
        slice_tables = {
            "by_symbol": summarize_slice_validation(baseline_run.signals_df, representative_hold, group_cols=["symbol"], horizon_bars=representative_horizon),
            "by_month": summarize_slice_validation(baseline_run.signals_df, representative_hold, group_cols=["month"], horizon_bars=representative_horizon),
            "by_time_bucket": summarize_slice_validation(baseline_run.signals_df, representative_hold, group_cols=["time_bucket"], horizon_bars=representative_horizon),
            "by_volatility_regime": summarize_slice_validation(baseline_run.signals_df, representative_hold, group_cols=["volatility_regime"], horizon_bars=representative_horizon),
        }

    classification = classify_long_horizon(raw_overall=raw_overall, hold_ladder=hold_ladder)
    _print_results(
        baseline_run=baseline_run,
        raw_overall=raw_overall,
        raw_by_symbol=raw_by_symbol,
        raw_by_time_bucket=raw_by_time_bucket,
        raw_by_month=raw_by_month,
        hold_ladder=hold_ladder,
        slice_tables=slice_tables,
        classification=classification,
    )

    if args.output_dir:
        _save_outputs(
            Path(args.output_dir),
            baseline_run=baseline_run,
            raw_overall=raw_overall,
            raw_by_symbol=raw_by_symbol,
            raw_by_time_bucket=raw_by_time_bucket,
            raw_by_month=raw_by_month,
            hold_ladder=hold_ladder,
            slice_tables=slice_tables,
            classification=classification,
        )
    log_experiment_run(
        run_type="long_horizon_validation",
        script_path=__file__,
        strategy_name=baseline_run.strategy_mode,
        dataset_path=args.dataset or source_dataset,
        symbols=symbols,
        params={
            "config": args.config,
            "strategy": args.strategy,
            "start_date": args.start_date,
            "end_date": args.end_date,
            "forward_horizons": list(forward_horizons),
            "hold_horizons": list(hold_horizons),
            "commission_per_order": args.commission_per_order,
            "slippage_per_share": args.slippage_per_share,
            "position_size": args.position_size,
            "sections": args.sections,
        },
        metrics={
            "profit_factor": baseline_run.backtest_result.get("profit_factor"),
            "sharpe": baseline_run.backtest_result.get("sharpe_ratio"),
            "win_rate": baseline_run.backtest_result.get("win_rate"),
            "max_drawdown_pct": baseline_run.backtest_result.get("max_drawdown_pct"),
            "trade_count": len(baseline_run.closed_trades_df),
            "expectancy": baseline_run.backtest_result.get("expectancy"),
            "realized_pnl": baseline_run.backtest_result.get("realized_pnl"),
        },
        output_path=args.output_dir,
        summary_path=(Path(args.output_dir) / "summary.json") if args.output_dir else None,
        extra_fields={
            "classification": {"label": classification[0], "reason": classification[1]},
            "best_raw_horizon_bars": (
                int(raw_overall.sort_values(["avg_net_expectancy_pct", "horizon_bars"], ascending=[False, True]).iloc[0]["horizon_bars"])
                if not raw_overall.empty else None
            ),
            "best_raw_net_expectancy_pct": (
                float(raw_overall["avg_net_expectancy_pct"].max()) if not raw_overall.empty else None
            ),
        },
    )


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from backtest_runner import run_backtest
from research.experiment_log import log_experiment_run
from run_edge_audit import (
    _backtest_exit_summary,
    _frame_text,
    _load_runtime_payload,
    _prepare_inputs_and_state,
    _print_strategy_summary,
    build_audit_context,
    evaluate_raw_entry_signals,
)
from run_edge_diagnostics import summarize_fixed_horizon_metrics
from run_trade_path_diagnostics import pair_round_trip_trades
from strategy import STRATEGY_MODE_MOMENTUM_BREAKOUT, STRATEGY_MODE_TREND_PULLBACK


DEFAULT_CONFIG_PATH = Path("config") / "momentum_breakout.example.json"
DEFAULT_TREND_PULLBACK_CONFIG = Path("config") / "trend_pullback.example.json"
DEFAULT_HORIZONS = (1, 2, 3, 4, 8)


@dataclass(frozen=True)
class ScenarioSpec:
    name: str
    section: str
    runtime_overrides: dict[str, Any]
    symbols: tuple[str, ...]


@dataclass(frozen=True)
class ValidationRun:
    name: str
    strategy_mode: str
    context: Any
    evaluations_df: pd.DataFrame
    signals_df: pd.DataFrame
    backtest_result: dict[str, Any]
    closed_trades_df: pd.DataFrame


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run focused validation for the explicit momentum_breakout strategy."
    )
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH), help="Momentum breakout runtime config JSON.")
    parser.add_argument("--trend-pullback-config", default=str(DEFAULT_TREND_PULLBACK_CONFIG), help="Trend pullback comparison config JSON.")
    parser.add_argument("--dataset", help="Dataset directory. Defaults to config source.dataset.")
    parser.add_argument("--symbols", nargs="*", help="Optional symbol override.")
    parser.add_argument("--start-date", help="Optional UTC date filter (YYYY-MM-DD).")
    parser.add_argument("--end-date", help="Optional UTC date filter (YYYY-MM-DD).")
    parser.add_argument("--commission-per-order", type=float, default=0.01)
    parser.add_argument("--slippage-per-share", type=float, default=0.05)
    parser.add_argument("--position-size", type=float, default=1000.0)
    parser.add_argument(
        "--sections",
        default="all",
        help="Comma-separated subset: all,baseline,compare,perturb.",
    )
    parser.add_argument("--output-dir", help="Optional directory for CSV/JSON artifacts.")
    return parser.parse_args()


def _normalize_symbol_list(symbols: list[str] | tuple[str, ...]) -> tuple[str, ...]:
    normalized: list[str] = []
    seen: set[str] = set()
    for raw_symbol in symbols:
        symbol = str(raw_symbol).strip().upper()
        if not symbol or symbol in seen:
            continue
        normalized.append(symbol)
        seen.add(symbol)
    return tuple(normalized)


def _resolve_symbols(runtime: dict[str, Any], cli_symbols: list[str] | None) -> tuple[str, ...]:
    if cli_symbols:
        return _normalize_symbol_list(cli_symbols)
    runtime_symbols = runtime.get("symbols")
    if isinstance(runtime_symbols, list) and runtime_symbols:
        return _normalize_symbol_list(runtime_symbols)
    return ()


def _runtime_with_overrides(runtime: dict[str, Any], overrides: dict[str, Any], symbols: tuple[str, ...]) -> dict[str, Any]:
    merged = dict(runtime)
    merged.update(overrides)
    merged["symbols"] = list(symbols)
    return merged


def _load_config_runtime(config_path: Path) -> tuple[dict[str, Any], str | None]:
    payload = _load_runtime_payload(config_path)
    dataset = payload["source"].get("dataset") if isinstance(payload["source"].get("dataset"), str) else None
    return dict(payload["runtime"]), dataset


def _closed_trade_count(backtest_result: dict[str, Any]) -> int:
    return sum(1 for trade in backtest_result.get("trades", []) if trade.get("side") == "SELL")


def _build_closed_trades_df(backtest_result: dict[str, Any], configured_hold_bars: int) -> pd.DataFrame:
    closed = pair_round_trip_trades(
        backtest_result.get("trades", []),
        configured_hold_bars=configured_hold_bars,
    ).copy()
    if closed.empty:
        return closed
    closed["entry_ts"] = pd.to_datetime(closed["entry_ts"], utc=True)
    closed["exit_ts"] = pd.to_datetime(closed["exit_ts"], utc=True)
    return closed


def attach_signal_context(closed_trades_df: pd.DataFrame, signals_df: pd.DataFrame) -> pd.DataFrame:
    if closed_trades_df.empty:
        return closed_trades_df.copy()
    signal_subset = signals_df[[
        "symbol",
        "entry_ts",
        "time_bucket",
        "trend_proxy",
        "volatility_regime",
        "month",
        "signal_hour",
        "fwd_1b_net_pct",
        "fwd_2b_net_pct",
        "fwd_3b_net_pct",
        "fwd_4b_net_pct",
        "fwd_8b_net_pct",
    ]].copy()
    signal_subset["entry_ts"] = pd.to_datetime(signal_subset["entry_ts"], utc=True)
    return closed_trades_df.merge(signal_subset, on=["symbol", "entry_ts"], how="left")


def summarize_realized_by_symbol(backtest_result: dict[str, Any]) -> pd.DataFrame:
    rows = []
    for symbol, stats in backtest_result.get("per_symbol", {}).items():
        rows.append(
            {
                "symbol": symbol,
                "realized_pnl": float(stats.get("realized_pnl", 0.0)),
                "expectancy": float(stats.get("expectancy", 0.0)),
                "win_rate": float(stats.get("win_rate", 0.0)),
                "profit_factor": float(stats.get("profit_factor", 0.0)),
                "trades": int(stats.get("total_trades", 0)),
                "avg_return_per_trade_pct": float(stats.get("avg_return_per_trade_pct", 0.0)),
            }
        )
    return pd.DataFrame(rows).sort_values(["realized_pnl", "symbol"], ascending=[False, True]) if rows else pd.DataFrame(
        columns=["symbol", "realized_pnl", "expectancy", "win_rate", "profit_factor", "trades", "avg_return_per_trade_pct"]
    )


def summarize_slice_validation(
    signals_df: pd.DataFrame,
    closed_trades_df: pd.DataFrame,
    *,
    group_cols: list[str],
    horizon_bars: int,
) -> pd.DataFrame:
    columns = group_cols + [
        "signal_count",
        "realized_trade_count",
        "avg_forward_return_pct",
        "realized_expectancy",
        "realized_win_rate",
    ]
    if signals_df.empty:
        return pd.DataFrame(columns=columns)

    signal_metric = summarize_fixed_horizon_metrics(signals_df, group_cols=group_cols, horizons=(horizon_bars,))
    if signal_metric.empty:
        return pd.DataFrame(columns=columns)
    signal_metric = signal_metric.rename(
        columns={
            "sample_count": "signal_count",
            "avg_net_expectancy_pct": "avg_forward_return_pct",
        }
    )[group_cols + ["signal_count", "avg_forward_return_pct"]]

    if closed_trades_df.empty:
        realized_metric = pd.DataFrame(columns=group_cols + ["realized_trade_count", "realized_expectancy", "realized_win_rate"])
    else:
        rows: list[dict[str, Any]] = []
        grouped = [((), closed_trades_df)] if not group_cols else closed_trades_df.groupby(group_cols, dropna=False, sort=True)
        for group_key, group_df in grouped:
            key_values = group_key if isinstance(group_key, tuple) else (group_key,)
            record = {col: key_values[idx] for idx, col in enumerate(group_cols)}
            rows.append(
                {
                    **record,
                    "realized_trade_count": int(len(group_df)),
                    "realized_expectancy": float(group_df["realized_pnl"].mean()) if not group_df.empty else 0.0,
                    "realized_win_rate": float((group_df["realized_pnl"] > 0).mean() * 100.0) if not group_df.empty else 0.0,
                }
            )
        realized_metric = pd.DataFrame(rows)

    merged = signal_metric.merge(realized_metric, on=group_cols, how="left")
    merged["realized_trade_count"] = merged["realized_trade_count"].fillna(0).astype(int)
    merged["realized_expectancy"] = merged["realized_expectancy"].fillna(0.0)
    merged["realized_win_rate"] = merged["realized_win_rate"].fillna(0.0)
    return merged.sort_values(group_cols or ["signal_count"], ascending=True if group_cols else False)


def build_parameter_perturbation_specs(base_runtime: dict[str, Any], *, symbols: tuple[str, ...]) -> list[ScenarioSpec]:
    lookback = int(base_runtime.get("momentum_breakout_lookback_bars", 20))
    buffer_pct = float(base_runtime.get("momentum_breakout_entry_buffer_pct", 0.001))
    hold_bars = int(base_runtime.get("momentum_breakout_hold_bars", 3))
    min_adx = float(base_runtime.get("momentum_breakout_min_adx", 20.0))

    def spec(name: str, overrides: dict[str, Any]) -> ScenarioSpec:
        return ScenarioSpec(name=name, section="perturb", runtime_overrides=overrides, symbols=symbols)

    buffer_values = tuple(
        sorted(
            {
                round(max(0.0, buffer_pct - 0.0005), 6),
                round(buffer_pct, 6),
                round(buffer_pct + 0.0005, 6),
            }
        )
    )
    return [
        spec(f"lookback={value}", {"momentum_breakout_lookback_bars": value})
        for value in sorted({max(5, lookback - 5), lookback, lookback + 5})
    ] + [
        spec(f"entry_buffer={value:.4f}", {"momentum_breakout_entry_buffer_pct": value})
        for value in buffer_values
    ] + [
        spec(f"hold_bars={value}", {"momentum_breakout_hold_bars": value})
        for value in sorted({max(1, hold_bars - 1), hold_bars, hold_bars + 1})
    ] + [
        spec(f"min_adx={value:.1f}", {"momentum_breakout_min_adx": value})
        for value in sorted({max(0.0, min_adx - 5.0), min_adx, min_adx + 5.0})
    ]


def classify_validation(
    *,
    baseline_run: ValidationRun,
    perturbation_df: pd.DataFrame,
    comparison_df: pd.DataFrame,
) -> tuple[str, str]:
    baseline_expectancy = float(baseline_run.backtest_result.get("expectancy", 0.0))
    closed_trades = _closed_trade_count(baseline_run.backtest_result)
    if len(baseline_run.signals_df) == 0 or closed_trades == 0:
        return "not ready", "The baseline configuration did not produce enough actionable trades."

    raw_overall = summarize_fixed_horizon_metrics(baseline_run.signals_df, group_cols=[], horizons=DEFAULT_HORIZONS)
    best_raw = float(raw_overall["avg_net_expectancy_pct"].max()) if not raw_overall.empty else 0.0
    positive_perturb_ratio = float((perturbation_df["expectancy"] > 0).mean()) if not perturbation_df.empty else 0.0
    better_than_pullback = False
    if not comparison_df.empty and {"momentum_breakout", "trend_pullback"}.issubset(set(comparison_df["strategy_mode"])):
        lookup = comparison_df.set_index("strategy_mode").to_dict("index")
        better_than_pullback = float(lookup["momentum_breakout"]["realized_pnl"]) > float(lookup["trend_pullback"]["realized_pnl"])

    if best_raw <= 0 and baseline_expectancy <= 0:
        return "not ready", "The strategy does not show positive raw continuation or realized expectancy on this sample."
    if baseline_expectancy <= 0 and positive_perturb_ratio < 0.35:
        return "weak", "Some raw continuation exists, but realized performance stays negative across most nearby checks."
    if baseline_expectancy <= 0:
        return "promising but narrow", "Raw continuation exists, but realized performance is still too narrow and unstable."
    if positive_perturb_ratio < 0.5:
        return "promising but narrow", "The baseline is positive, but small local changes degrade it too often."
    if better_than_pullback:
        return "stable enough for deeper research", "Breakout is outperforming the recent pullback path and nearby checks are not collapsing."
    return "promising but narrow", "The baseline is better than zero, but the evidence is still narrow rather than robust."


def _save_outputs(
    output_dir: Path,
    *,
    baseline_run: ValidationRun,
    comparison_df: pd.DataFrame,
    perturbation_df: pd.DataFrame,
    slice_tables: dict[str, pd.DataFrame],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    baseline_run.evaluations_df.to_csv(output_dir / "momentum_breakout_entry_evaluations.csv", index=False)
    baseline_run.signals_df.to_csv(output_dir / "momentum_breakout_entry_signals.csv", index=False)
    baseline_run.closed_trades_df.to_csv(output_dir / "momentum_breakout_closed_trades.csv", index=False)
    comparison_df.to_csv(output_dir / "strategy_comparison.csv", index=False)
    perturbation_df.to_csv(output_dir / "perturbation_summary.csv", index=False)
    for name, table in slice_tables.items():
        table.to_csv(output_dir / f"{name}.csv", index=False)


def _run_validation_scenario(
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
) -> ValidationRun:
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
        horizons=DEFAULT_HORIZONS,
    )
    result = run_backtest(dataset_path=context.dataset_path, **context.backtest_kwargs)
    hold_bars = int(
        context.backtest_kwargs.get("momentum_breakout_hold_bars")
        or context.backtest_kwargs.get("trend_pullback_hold_bars")
        or 0
    )
    closed_trades_df = attach_signal_context(_build_closed_trades_df(result, hold_bars), signals_df)
    return ValidationRun(
        name=variant_name,
        strategy_mode=str(context.backtest_kwargs["strategy_mode"]),
        context=context,
        evaluations_df=evaluations_df,
        signals_df=signals_df,
        backtest_result=result,
        closed_trades_df=closed_trades_df,
    )


def _comparison_summary_df(runs: list[ValidationRun]) -> pd.DataFrame:
    rows = []
    for run in runs:
        overall = summarize_fixed_horizon_metrics(run.signals_df, group_cols=[], horizons=DEFAULT_HORIZONS)
        best_horizon = int(overall.sort_values(["avg_net_expectancy_pct", "horizon_bars"], ascending=[False, True]).iloc[0]["horizon_bars"]) if not overall.empty else 0
        best_raw = float(overall["avg_net_expectancy_pct"].max()) if not overall.empty else 0.0
        rows.append(
            {
                "strategy_mode": run.strategy_mode,
                "signals": int(len(run.signals_df)),
                "closed_trades": int(len(run.closed_trades_df)),
                "realized_pnl": float(run.backtest_result.get("realized_pnl", 0.0)),
                "expectancy": float(run.backtest_result.get("expectancy", 0.0)),
                "win_rate": float(run.backtest_result.get("win_rate", 0.0)),
                "profit_factor": float(run.backtest_result.get("profit_factor", 0.0)),
                "max_drawdown_pct": float(run.backtest_result.get("max_drawdown_pct", 0.0)),
                "best_raw_horizon_bars": best_horizon,
                "best_raw_net_expectancy_pct": best_raw,
            }
        )
    return pd.DataFrame(rows).sort_values("strategy_mode")


def _perturbation_summary(run: ValidationRun) -> dict[str, Any]:
    overall = summarize_fixed_horizon_metrics(run.signals_df, group_cols=[], horizons=DEFAULT_HORIZONS)
    best_raw = float(overall["avg_net_expectancy_pct"].max()) if not overall.empty else 0.0
    best_horizon = int(overall.sort_values(["avg_net_expectancy_pct", "horizon_bars"], ascending=[False, True]).iloc[0]["horizon_bars"]) if not overall.empty else 0
    return {
        "scenario": run.name,
        "strategy_mode": run.strategy_mode,
        "signal_count": int(len(run.signals_df)),
        "closed_trades": int(len(run.closed_trades_df)),
        "realized_pnl": float(run.backtest_result.get("realized_pnl", 0.0)),
        "expectancy": float(run.backtest_result.get("expectancy", 0.0)),
        "win_rate": float(run.backtest_result.get("win_rate", 0.0)),
        "profit_factor": float(run.backtest_result.get("profit_factor", 0.0)),
        "max_drawdown_pct": float(run.backtest_result.get("max_drawdown_pct", 0.0)),
        "best_raw_horizon_bars": best_horizon,
        "best_raw_net_expectancy_pct": best_raw,
    }


def _print_validation_results(
    baseline_run: ValidationRun,
    comparison_df: pd.DataFrame,
    perturbation_df: pd.DataFrame,
    slice_tables: dict[str, pd.DataFrame],
    classification: tuple[str, str],
) -> None:
    print("\n=== Baseline Momentum Breakout ===")
    _print_strategy_summary(baseline_run.context)
    print("\nSignal generation:")
    print(f"Bars evaluated: {len(baseline_run.evaluations_df)}")
    print(f"Signals fired:  {len(baseline_run.signals_df)}")
    print(f"Closed trades:  {len(baseline_run.closed_trades_df)}")
    print("\nForward returns overall:")
    print(_frame_text(summarize_fixed_horizon_metrics(baseline_run.signals_df, group_cols=[], horizons=DEFAULT_HORIZONS)))
    print("\nForward returns by symbol:")
    print(_frame_text(summarize_fixed_horizon_metrics(baseline_run.signals_df, group_cols=["symbol"], horizons=DEFAULT_HORIZONS), max_rows=32))
    print("\nSignal frequency by symbol:")
    print(_frame_text(baseline_run.signals_df.groupby("symbol", dropna=False).size().reset_index(name="signal_count"), max_rows=32))
    print("\nSignal frequency by time bucket:")
    print(_frame_text(baseline_run.signals_df.groupby("time_bucket", dropna=False).size().reset_index(name="signal_count"), max_rows=16))
    print("\nRealized performance:")
    print(_frame_text(_backtest_exit_summary(baseline_run.backtest_result)))
    print("\nPer-symbol realized performance:")
    print(_frame_text(summarize_realized_by_symbol(baseline_run.backtest_result), max_rows=32))
    print("\nSlice summaries (baseline hold horizon):")
    for title, table in slice_tables.items():
        print(f"\n{title}:")
        print(_frame_text(table, max_rows=32))
    print("\nStrategy comparison:")
    print(_frame_text(comparison_df, max_rows=16))
    print("\nSmall perturbation stability:")
    print(_frame_text(perturbation_df, max_rows=32))
    print(f"\nDiagnosis: {classification[0]}")
    print(classification[1])


def main() -> None:
    args = parse_args()
    sections = {chunk.strip().lower() for chunk in args.sections.split(",") if chunk.strip()}
    run_all = "all" in sections

    config_path = Path(args.config)
    tp_config_path = Path(args.trend_pullback_config)
    base_runtime, source_dataset = _load_config_runtime(config_path)
    baseline_symbols = _resolve_symbols(base_runtime, args.symbols)
    if not baseline_symbols:
        raise RuntimeError("No symbols resolved for validation.")
    base_runtime["strategy_mode"] = STRATEGY_MODE_MOMENTUM_BREAKOUT

    baseline_run = _run_validation_scenario(
        config_path=config_path,
        runtime=_runtime_with_overrides(base_runtime, {}, baseline_symbols),
        source_dataset=source_dataset,
        dataset_override=args.dataset,
        symbols=baseline_symbols,
        start_date=args.start_date,
        end_date=args.end_date,
        commission_per_order=args.commission_per_order,
        slippage_per_share=args.slippage_per_share,
        position_size=args.position_size,
        output_dir=args.output_dir,
        variant_name="momentum_breakout_baseline",
    )

    baseline_horizon = int(baseline_run.context.backtest_kwargs["momentum_breakout_hold_bars"])
    slice_tables = {
        "by_symbol": summarize_slice_validation(baseline_run.signals_df, baseline_run.closed_trades_df, group_cols=["symbol"], horizon_bars=baseline_horizon),
        "by_trend_proxy": summarize_slice_validation(baseline_run.signals_df, baseline_run.closed_trades_df, group_cols=["trend_proxy"], horizon_bars=baseline_horizon),
        "by_volatility_regime": summarize_slice_validation(baseline_run.signals_df, baseline_run.closed_trades_df, group_cols=["volatility_regime"], horizon_bars=baseline_horizon),
        "by_time_bucket": summarize_slice_validation(baseline_run.signals_df, baseline_run.closed_trades_df, group_cols=["time_bucket"], horizon_bars=baseline_horizon),
        "by_month": summarize_slice_validation(baseline_run.signals_df, baseline_run.closed_trades_df, group_cols=["month"], horizon_bars=baseline_horizon),
    }

    comparison_runs = [baseline_run]
    if run_all or "compare" in sections:
        tp_runtime, _ = _load_config_runtime(tp_config_path)
        tp_runtime["strategy_mode"] = STRATEGY_MODE_TREND_PULLBACK
        comparison_runs.append(
            _run_validation_scenario(
                config_path=tp_config_path,
                runtime=_runtime_with_overrides(tp_runtime, {}, baseline_symbols),
                source_dataset=args.dataset or source_dataset,
                dataset_override=args.dataset,
                symbols=baseline_symbols,
                start_date=args.start_date,
                end_date=args.end_date,
                commission_per_order=args.commission_per_order,
                slippage_per_share=args.slippage_per_share,
                position_size=args.position_size,
                output_dir=args.output_dir,
                variant_name="trend_pullback_comparison",
            )
        )
    comparison_df = _comparison_summary_df(comparison_runs)

    perturbation_df = pd.DataFrame(columns=["scenario", "strategy_mode", "signal_count", "closed_trades", "realized_pnl", "expectancy", "win_rate", "profit_factor", "max_drawdown_pct", "best_raw_horizon_bars", "best_raw_net_expectancy_pct"])
    if run_all or "perturb" in sections:
        perturbation_runs = []
        for spec in build_parameter_perturbation_specs(base_runtime, symbols=baseline_symbols):
            runtime = _runtime_with_overrides(base_runtime, spec.runtime_overrides, spec.symbols)
            perturbation_runs.append(
                _run_validation_scenario(
                    config_path=config_path,
                    runtime=runtime,
                    source_dataset=source_dataset,
                    dataset_override=args.dataset,
                    symbols=spec.symbols,
                    start_date=args.start_date,
                    end_date=args.end_date,
                    commission_per_order=args.commission_per_order,
                    slippage_per_share=args.slippage_per_share,
                    position_size=args.position_size,
                    output_dir=args.output_dir,
                    variant_name=spec.name,
                )
            )
        perturbation_df = pd.DataFrame([_perturbation_summary(run) for run in perturbation_runs]).sort_values(
            ["expectancy", "realized_pnl", "scenario"],
            ascending=[False, False, True],
        )

    classification = classify_validation(
        baseline_run=baseline_run,
        perturbation_df=perturbation_df,
        comparison_df=comparison_df,
    )
    _print_validation_results(baseline_run, comparison_df, perturbation_df, slice_tables, classification)

    if args.output_dir:
        output_dir = Path(args.output_dir)
        _save_outputs(
            output_dir,
            baseline_run=baseline_run,
            comparison_df=comparison_df,
            perturbation_df=perturbation_df,
            slice_tables=slice_tables,
        )
        payload = {
            "classification": {"label": classification[0], "reason": classification[1]},
            "baseline": {
                "strategy_mode": baseline_run.strategy_mode,
                "signal_count": int(len(baseline_run.signals_df)),
                "closed_trade_count": int(len(baseline_run.closed_trades_df)),
                "backtest_result": {
                    "realized_pnl": float(baseline_run.backtest_result.get("realized_pnl", 0.0)),
                    "expectancy": float(baseline_run.backtest_result.get("expectancy", 0.0)),
                    "win_rate": float(baseline_run.backtest_result.get("win_rate", 0.0)),
                    "profit_factor": float(baseline_run.backtest_result.get("profit_factor", 0.0)),
                    "max_drawdown_pct": float(baseline_run.backtest_result.get("max_drawdown_pct", 0.0)),
                },
            },
        }
        (output_dir / "validation_summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"\nSaved validation outputs to: {output_dir}")
    log_experiment_run(
        run_type="strategy_validation",
        script_path=__file__,
        strategy_name=baseline_run.strategy_mode,
        dataset_path=args.dataset or source_dataset,
        symbols=baseline_symbols,
        params={
            "config": args.config,
            "trend_pullback_config": args.trend_pullback_config,
            "start_date": args.start_date,
            "end_date": args.end_date,
            "commission_per_order": args.commission_per_order,
            "slippage_per_share": args.slippage_per_share,
            "position_size": args.position_size,
            "sections": args.sections,
        },
        metrics={
            "total_return_pct": baseline_run.backtest_result.get("total_return_pct"),
            "profit_factor": baseline_run.backtest_result.get("profit_factor"),
            "sharpe": baseline_run.backtest_result.get("sharpe_ratio"),
            "win_rate": baseline_run.backtest_result.get("win_rate"),
            "max_drawdown_pct": baseline_run.backtest_result.get("max_drawdown_pct"),
            "trade_count": len(baseline_run.closed_trades_df),
            "expectancy": baseline_run.backtest_result.get("expectancy"),
            "realized_pnl": baseline_run.backtest_result.get("realized_pnl"),
        },
        output_path=args.output_dir,
        summary_path=(Path(args.output_dir) / "validation_summary.json") if args.output_dir else None,
        extra_fields={
            "classification": {"label": classification[0], "reason": classification[1]},
            "comparison_rows": len(comparison_df),
            "perturbation_rows": len(perturbation_df),
        },
    )


if __name__ == "__main__":
    main()

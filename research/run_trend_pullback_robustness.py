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
    _load_runtime_payload,
    _prepare_inputs_and_state,
    build_audit_context,
    evaluate_raw_entry_signals,
)
from run_edge_diagnostics import _frame_text, summarize_fixed_horizon_metrics


DEFAULT_CONFIG_PATH = Path("config") / "trend_pullback.example.json"
DEFAULT_BASELINE_SYMBOLS = ("AMD", "HON", "C", "JPM")
DEFAULT_HORIZON = 4


@dataclass(frozen=True)
class ScenarioSpec:
    name: str
    section: str
    symbols: tuple[str, ...]
    runtime_overrides: dict[str, Any]
    start_date: str | None = None
    end_date: str | None = None


@dataclass(frozen=True)
class ScenarioRun:
    spec: ScenarioSpec
    context: Any
    evaluations_df: pd.DataFrame
    signals_df: pd.DataFrame
    backtest_result: dict[str, Any]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run a focused robustness validation pass for the explicit trend_pullback strategy."
        )
    )
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH), help="Runtime config JSON.")
    parser.add_argument("--dataset", help="Dataset directory. Defaults to config source.dataset.")
    parser.add_argument(
        "--symbols",
        nargs="*",
        help="Baseline symbol subset. Default: runtime symbols, otherwise AMD HON C JPM.",
    )
    parser.add_argument(
        "--adjacent-symbols",
        nargs="*",
        default=[],
        help="Optional adjacent candidates to test as baseline-plus cases.",
    )
    parser.add_argument("--start-date", help="Optional UTC date filter (YYYY-MM-DD).")
    parser.add_argument("--end-date", help="Optional UTC date filter (YYYY-MM-DD).")
    parser.add_argument("--commission-per-order", type=float, default=0.01)
    parser.add_argument("--slippage-per-share", type=float, default=0.05)
    parser.add_argument("--position-size", type=float, default=1000.0)
    parser.add_argument(
        "--sections",
        default="all",
        help="Comma-separated subset of checks: all,time,params,symbols,regime.",
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


def _resolve_baseline_symbols(runtime: dict[str, Any], cli_symbols: list[str] | None) -> tuple[str, ...]:
    if cli_symbols:
        return _normalize_symbol_list(cli_symbols)
    runtime_symbols = runtime.get("symbols")
    if isinstance(runtime_symbols, list) and runtime_symbols:
        return _normalize_symbol_list(runtime_symbols)
    return DEFAULT_BASELINE_SYMBOLS


def _runtime_with_overrides(runtime: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    merged = dict(runtime)
    merged.update(overrides)
    return merged


def _closed_trade_count(backtest_result: dict[str, Any]) -> int:
    return sum(1 for trade in backtest_result.get("trades", []) if trade.get("side") == "SELL")


def _scenario_summary(run: ScenarioRun, *, horizon_bars: int) -> dict[str, Any]:
    available_horizons = sorted(
        int(column.removeprefix("fwd_").removesuffix("b_gross_pct"))
        for column in run.signals_df.columns
        if column.startswith("fwd_") and column.endswith("b_gross_pct")
    )
    summary_horizon = horizon_bars if horizon_bars in available_horizons else (available_horizons[0] if available_horizons else horizon_bars)
    raw_fixed = summarize_fixed_horizon_metrics(run.signals_df, group_cols=[], horizons=(summary_horizon,))
    raw_row = raw_fixed.iloc[0] if not raw_fixed.empty else None
    return {
        "section": run.spec.section,
        "scenario": run.spec.name,
        "symbols": ",".join(run.spec.symbols),
        "start_date": run.spec.start_date,
        "end_date": run.spec.end_date,
        "signal_count": int(len(run.signals_df)),
        "filled_trades": _closed_trade_count(run.backtest_result),
        "total_pnl": float(run.backtest_result.get("realized_pnl", 0.0)),
        "expectancy": float(run.backtest_result.get("expectancy", 0.0)),
        "win_rate": float(run.backtest_result.get("win_rate", 0.0)),
        "profit_factor": float(run.backtest_result.get("profit_factor", 0.0)),
        "max_drawdown_pct": float(run.backtest_result.get("max_drawdown_pct", 0.0)),
        "raw_fixed_sample_count": int(raw_row["sample_count"]) if raw_row is not None else 0,
        "raw_fixed_horizon_bars": summary_horizon,
        "raw_fixed_net_expectancy_pct": float(raw_row["avg_net_expectancy_pct"]) if raw_row is not None else 0.0,
        "raw_fixed_win_rate_pct": float(raw_row["net_win_rate_pct"]) if raw_row is not None else 0.0,
    }


def build_month_slice_specs(
    evaluations_df: pd.DataFrame,
    *,
    runtime_overrides: dict[str, Any],
    symbols: tuple[str, ...],
) -> list[ScenarioSpec]:
    if evaluations_df.empty or "month" not in evaluations_df.columns:
        return []
    specs: list[ScenarioSpec] = []
    for month in sorted(str(value) for value in evaluations_df["month"].dropna().unique()):
        period = pd.Period(month, freq="M")
        specs.append(
            ScenarioSpec(
                name=f"month:{month}",
                section="time",
                symbols=symbols,
                runtime_overrides=dict(runtime_overrides),
                start_date=period.start_time.strftime("%Y-%m-%d"),
                end_date=period.end_time.strftime("%Y-%m-%d"),
            )
        )
    return specs


def build_parameter_perturbation_specs(
    base_runtime: dict[str, Any],
    *,
    symbols: tuple[str, ...],
) -> list[ScenarioSpec]:
    hold_baseline = int(base_runtime.get("trend_pullback_hold_bars", DEFAULT_HORIZON))
    threshold_baseline = float(base_runtime.get("trend_pullback_entry_threshold", 0.0015))
    min_adx_baseline = float(base_runtime.get("trend_pullback_min_adx", 20.0))

    def scenario(name: str, overrides: dict[str, Any]) -> ScenarioSpec:
        return ScenarioSpec(
            name=name,
            section="params",
            symbols=symbols,
            runtime_overrides=overrides,
        )

    threshold_values = tuple(
        sorted(
            {
                round(max(0.0, threshold_baseline - 0.0005), 6),
                round(threshold_baseline, 6),
                round(threshold_baseline + 0.0005, 6),
            }
        )
    )
    return [
        scenario(f"hold_bars={value}", {"trend_pullback_hold_bars": value})
        for value in (max(1, hold_baseline - 1), hold_baseline, hold_baseline + 1)
    ] + [
        scenario(f"entry_threshold={value:.4f}", {"trend_pullback_entry_threshold": value})
        for value in threshold_values
    ] + [
        scenario(f"min_adx={value:.1f}", {"trend_pullback_min_adx": value})
        for value in (
            max(0.0, min_adx_baseline - 5.0),
            min_adx_baseline,
            min_adx_baseline + 5.0,
        )
    ]


def build_symbol_robustness_specs(
    baseline_symbols: tuple[str, ...],
    *,
    runtime_overrides: dict[str, Any],
    adjacent_symbols: tuple[str, ...],
) -> list[ScenarioSpec]:
    specs = [
        ScenarioSpec(
            name="baseline",
            section="symbols",
            symbols=baseline_symbols,
            runtime_overrides=dict(runtime_overrides),
        )
    ]
    if len(baseline_symbols) > 1:
        for removed_symbol in baseline_symbols:
            symbols = tuple(symbol for symbol in baseline_symbols if symbol != removed_symbol)
            specs.append(
                ScenarioSpec(
                    name=f"leave_out:{removed_symbol}",
                    section="symbols",
                    symbols=symbols,
                    runtime_overrides=dict(runtime_overrides),
                )
            )
    for adjacent_symbol in adjacent_symbols:
        if adjacent_symbol in baseline_symbols:
            continue
        specs.append(
            ScenarioSpec(
                name=f"plus:{adjacent_symbol}",
                section="symbols",
                symbols=baseline_symbols + (adjacent_symbol,),
                runtime_overrides=dict(runtime_overrides),
            )
        )
    return specs


def classify_fragility(
    *,
    baseline_summary: dict[str, Any],
    time_slice_df: pd.DataFrame,
    param_df: pd.DataFrame,
    symbol_df: pd.DataFrame,
    regime_tables: dict[str, pd.DataFrame],
) -> tuple[str, str]:
    baseline_expectancy = float(baseline_summary.get("expectancy", 0.0))
    baseline_pnl = float(baseline_summary.get("total_pnl", 0.0))
    if baseline_summary.get("filled_trades", 0) <= 0:
        return "not ready", "The baseline configuration did not produce closed trades."

    time_positive_ratio = 0.0
    if not time_slice_df.empty:
        time_positive_ratio = float((time_slice_df["expectancy"] > 0).mean())

    param_positive_ratio = 0.0
    if not param_df.empty:
        param_positive_ratio = float((param_df["expectancy"] > 0).mean())

    leave_one_out = symbol_df[symbol_df["scenario"].str.startswith("leave_out:")] if not symbol_df.empty else pd.DataFrame()
    leave_one_out_positive_ratio = float((leave_one_out["expectancy"] > 0).mean()) if not leave_one_out.empty else 0.0

    trend_table = regime_tables.get("trend_proxy", pd.DataFrame())
    midday_table = regime_tables.get("time_bucket", pd.DataFrame())
    trend_positive = False
    midday_negative = False
    if not trend_table.empty:
        trend_rows = trend_table[trend_table["trend_proxy"] == "trend"]
        trend_positive = bool(not trend_rows.empty and float(trend_rows.iloc[0]["avg_net_expectancy_pct"]) > 0)
    if not midday_table.empty:
        midday_rows = midday_table[midday_table["time_bucket"] == "midday"]
        midday_negative = bool(not midday_rows.empty and float(midday_rows.iloc[0]["avg_net_expectancy_pct"]) <= 0)

    if baseline_expectancy <= 0 or baseline_pnl <= 0:
        if param_positive_ratio < 0.4 or time_positive_ratio < 0.4:
            return "not ready", "Baseline realized performance is non-positive and nearby checks do not rescue it."
        return "fragile", "Baseline is weak and only a minority of nearby checks stay positive."

    if time_positive_ratio <= 0.34:
        return "sample-dependent", "Only a small fraction of time slices stayed positive."

    if param_positive_ratio < 0.5:
        return "fragile", "Small local parameter changes materially degrade the strategy."

    if leave_one_out_positive_ratio < 0.5:
        return "fragile", "Removing a single baseline symbol frequently breaks the result."

    if trend_positive and midday_negative:
        return "promising but narrow", "The strategy still looks real, but the edge remains trend-dependent and anti-midday."

    if time_positive_ratio < 0.67 or param_positive_ratio < 0.67:
        return "promising but narrow", "The strategy is positive, but its robustness still looks narrow rather than broad."

    return "stable", "Performance remained positive across most nearby time, parameter, and symbol checks."


def _regime_confirmation(signals_df: pd.DataFrame, *, horizon_bars: int) -> dict[str, pd.DataFrame]:
    return {
        "trend_proxy": summarize_fixed_horizon_metrics(signals_df, group_cols=["trend_proxy"], horizons=(horizon_bars,)),
        "volatility_regime": summarize_fixed_horizon_metrics(signals_df, group_cols=["volatility_regime"], horizons=(horizon_bars,)),
        "time_bucket": summarize_fixed_horizon_metrics(signals_df, group_cols=["time_bucket"], horizons=(horizon_bars,)),
    }


def _run_scenario(
    *,
    config_path: Path,
    source_dataset: str | None,
    base_runtime: dict[str, Any],
    spec: ScenarioSpec,
    dataset_override: str | None,
    commission_per_order: float,
    slippage_per_share: float,
    position_size: float,
    output_dir: str | None,
    horizon_bars: int,
) -> ScenarioRun:
    runtime = _runtime_with_overrides(base_runtime, spec.runtime_overrides)
    context = build_audit_context(
        config_path=config_path,
        runtime=runtime,
        source_dataset=source_dataset,
        dataset_override=dataset_override,
        symbols_override=list(spec.symbols),
        start_date=spec.start_date,
        end_date=spec.end_date,
        commission_per_order=commission_per_order,
        slippage_per_share=slippage_per_share,
        position_size=position_size,
        output_dir=output_dir,
        variant="live_effective",
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
        horizons=(horizon_bars,),
    )
    backtest_result = run_backtest(context.dataset_path, **context.backtest_kwargs)
    return ScenarioRun(
        spec=spec,
        context=context,
        evaluations_df=evaluations_df,
        signals_df=signals_df,
        backtest_result=backtest_result,
    )


def _summaries_frame(runs: list[ScenarioRun], *, horizon_bars: int) -> pd.DataFrame:
    rows = [_scenario_summary(run, horizon_bars=horizon_bars) for run in runs]
    if not rows:
        return pd.DataFrame(
            columns=[
                "section",
                "scenario",
                "symbols",
                "start_date",
                "end_date",
                "signal_count",
                "filled_trades",
                "total_pnl",
                "expectancy",
                "win_rate",
                "profit_factor",
                "max_drawdown_pct",
                "raw_fixed_sample_count",
                "raw_fixed_horizon_bars",
                "raw_fixed_net_expectancy_pct",
                "raw_fixed_win_rate_pct",
            ]
        )
    return pd.DataFrame(rows)


def _save_outputs(
    *,
    output_dir: Path,
    baseline_run: ScenarioRun,
    summary_tables: dict[str, pd.DataFrame],
    regime_tables: dict[str, pd.DataFrame],
    classification: tuple[str, str],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, table in summary_tables.items():
        table.to_csv(output_dir / f"{name}.csv", index=False)
    for name, table in regime_tables.items():
        table.to_csv(output_dir / f"regime_{name}.csv", index=False)
    baseline_run.evaluations_df.to_csv(output_dir / "baseline_entry_evaluations.csv", index=False)
    baseline_run.signals_df.to_csv(output_dir / "baseline_entry_signals.csv", index=False)
    payload = {
        "fragility_label": classification[0],
        "fragility_reason": classification[1],
        "baseline_summary": _scenario_summary(baseline_run, horizon_bars=int(baseline_run.context.backtest_kwargs["trend_pullback_hold_bars"])),
    }
    (output_dir / "robustness_summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _print_section(title: str, df: pd.DataFrame, *, max_rows: int | None = None) -> None:
    print(f"\n=== {title} ===")
    print(_frame_text(df, max_rows=max_rows))


def _parse_sections(raw: str) -> set[str]:
    if raw.strip().lower() == "all":
        return {"time", "params", "symbols", "regime"}
    return {part.strip().lower() for part in raw.split(",") if part.strip()}


def main() -> None:
    args = parse_args()
    config_path = Path(args.config)
    payload = _load_runtime_payload(config_path)
    base_runtime = dict(payload["runtime"])
    base_runtime["strategy_mode"] = "trend_pullback"
    baseline_symbols = _resolve_baseline_symbols(base_runtime, args.symbols)
    adjacent_symbols = _normalize_symbol_list(args.adjacent_symbols)
    selected_sections = _parse_sections(args.sections)

    baseline_spec = ScenarioSpec(
        name="baseline",
        section="baseline",
        symbols=baseline_symbols,
        runtime_overrides={},
        start_date=args.start_date,
        end_date=args.end_date,
    )
    baseline_hold_bars = int(base_runtime.get("trend_pullback_hold_bars", DEFAULT_HORIZON))
    source_dataset = payload["source"].get("dataset") if isinstance(payload.get("source", {}).get("dataset"), str) else None
    baseline_run = _run_scenario(
        config_path=config_path,
        source_dataset=source_dataset,
        base_runtime=base_runtime,
        spec=baseline_spec,
        dataset_override=args.dataset,
        commission_per_order=args.commission_per_order,
        slippage_per_share=args.slippage_per_share,
        position_size=args.position_size,
        output_dir=args.output_dir,
        horizon_bars=baseline_hold_bars,
    )
    baseline_summary = _scenario_summary(baseline_run, horizon_bars=baseline_hold_bars)

    print("\n=== Trend Pullback Robustness ===")
    print(f"Config:   {config_path}")
    print(f"Dataset:  {baseline_run.context.dataset_path}")
    print(f"Symbols:  {', '.join(baseline_symbols)}")
    if not isinstance(payload["runtime"].get("symbols"), list) and not args.symbols:
        print("Baseline symbols defaulted to AMD, HON, C, JPM because the config did not provide a symbol list.")
    print("\nBaseline summary:")
    print(_frame_text(pd.DataFrame([baseline_summary])))

    summary_tables: dict[str, pd.DataFrame] = {}

    time_runs: list[ScenarioRun] = []
    if "time" in selected_sections:
        for spec in build_month_slice_specs(
            baseline_run.evaluations_df,
            runtime_overrides={},
            symbols=baseline_symbols,
        ):
            time_runs.append(
                _run_scenario(
                    config_path=config_path,
                    source_dataset=source_dataset,
                    base_runtime=base_runtime,
                    spec=spec,
                    dataset_override=args.dataset,
                    commission_per_order=args.commission_per_order,
                    slippage_per_share=args.slippage_per_share,
                    position_size=args.position_size,
                    output_dir=args.output_dir,
                    horizon_bars=baseline_hold_bars,
                )
            )
        summary_tables["time_slices"] = _summaries_frame(time_runs, horizon_bars=baseline_hold_bars)
        _print_section("Time Slice Stability", summary_tables["time_slices"], max_rows=24)

    param_runs: list[ScenarioRun] = []
    if "params" in selected_sections:
        for spec in build_parameter_perturbation_specs(base_runtime, symbols=baseline_symbols):
            param_runs.append(
                _run_scenario(
                    config_path=config_path,
                    source_dataset=source_dataset,
                    base_runtime=base_runtime,
                    spec=spec,
                    dataset_override=args.dataset,
                    commission_per_order=args.commission_per_order,
                    slippage_per_share=args.slippage_per_share,
                    position_size=args.position_size,
                    output_dir=args.output_dir,
                    horizon_bars=int(spec.runtime_overrides.get("trend_pullback_hold_bars", baseline_hold_bars)),
                )
            )
        summary_tables["parameter_perturbations"] = _summaries_frame(
            param_runs,
            horizon_bars=baseline_hold_bars,
        )
        _print_section("Parameter Perturbations", summary_tables["parameter_perturbations"], max_rows=32)

    symbol_runs: list[ScenarioRun] = []
    if "symbols" in selected_sections:
        for spec in build_symbol_robustness_specs(
            baseline_symbols,
            runtime_overrides={},
            adjacent_symbols=adjacent_symbols,
        ):
            symbol_runs.append(
                _run_scenario(
                    config_path=config_path,
                    source_dataset=source_dataset,
                    base_runtime=base_runtime,
                    spec=spec,
                    dataset_override=args.dataset,
                    commission_per_order=args.commission_per_order,
                    slippage_per_share=args.slippage_per_share,
                    position_size=args.position_size,
                    output_dir=args.output_dir,
                    horizon_bars=baseline_hold_bars,
                )
            )
        summary_tables["symbol_robustness"] = _summaries_frame(symbol_runs, horizon_bars=baseline_hold_bars)
        _print_section("Symbol Robustness", summary_tables["symbol_robustness"], max_rows=32)

    regime_tables = _regime_confirmation(baseline_run.signals_df, horizon_bars=baseline_hold_bars) if "regime" in selected_sections else {
        "trend_proxy": pd.DataFrame(),
        "volatility_regime": pd.DataFrame(),
        "time_bucket": pd.DataFrame(),
    }
    if "regime" in selected_sections:
        _print_section("Regime Confirmation: Trend Proxy", regime_tables["trend_proxy"], max_rows=24)
        _print_section("Regime Confirmation: Volatility", regime_tables["volatility_regime"], max_rows=24)
        _print_section("Regime Confirmation: Time Bucket", regime_tables["time_bucket"], max_rows=24)

    classification = classify_fragility(
        baseline_summary=baseline_summary,
        time_slice_df=summary_tables.get("time_slices", pd.DataFrame()),
        param_df=summary_tables.get("parameter_perturbations", pd.DataFrame()),
        symbol_df=summary_tables.get("symbol_robustness", pd.DataFrame()),
        regime_tables=regime_tables,
    )
    print("\n=== Fragility Summary ===")
    print(f"Label: {classification[0]}")
    print(f"Why:   {classification[1]}")

    if args.output_dir:
        _save_outputs(
            output_dir=Path(args.output_dir),
            baseline_run=baseline_run,
            summary_tables=summary_tables,
            regime_tables=regime_tables,
            classification=classification,
        )
        print(f"\nArtifacts written to: {Path(args.output_dir)}")
    log_experiment_run(
        run_type="robustness_validation",
        script_path=__file__,
        strategy_name="trend_pullback",
        dataset_path=args.dataset or source_dataset,
        symbols=baseline_symbols,
        params={
            "config": args.config,
            "adjacent_symbols": list(args.adjacent_symbols),
            "start_date": args.start_date,
            "end_date": args.end_date,
            "commission_per_order": args.commission_per_order,
            "slippage_per_share": args.slippage_per_share,
            "position_size": args.position_size,
            "sections": args.sections,
        },
        metrics={
            "total_return_pct": baseline_summary.get("total_return_pct"),
            "profit_factor": baseline_summary.get("profit_factor"),
            "sharpe": baseline_summary.get("sharpe_ratio"),
            "win_rate": baseline_summary.get("win_rate"),
            "max_drawdown_pct": baseline_summary.get("max_drawdown_pct"),
            "trade_count": baseline_summary.get("closed_trade_count"),
            "expectancy": baseline_summary.get("expectancy"),
            "realized_pnl": baseline_summary.get("realized_pnl"),
        },
        output_path=args.output_dir,
        summary_path=(Path(args.output_dir) / "robustness_summary.json") if args.output_dir else None,
        extra_fields={
            "baseline_scenario": baseline_summary.get("scenario"),
            "sections_run": sorted({item.spec.section for item in scenario_runs}),
        },
    )


if __name__ == "__main__":
    main()

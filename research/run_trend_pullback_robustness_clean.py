from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from run_edge_audit import _frame_text, _prepare_inputs_and_state
from run_long_horizon_validation import _parse_horizon_list, _run_long_horizon_scenario, _runtime_with_overrides
from run_trade_path_diagnostics import (
    attach_trade_path_metrics,
    build_signal_path_table,
    build_trade_shape_summary,
    summarize_hold_delta,
    summarize_opportunity_capture,
)
from run_trend_pullback_robustness import build_month_slice_specs, build_symbol_robustness_specs
from run_volatility_expansion_validation import (
    _average_holding_period,
    _load_config_runtime,
    _resolve_symbols,
    summarize_realized_by_symbol,
    summarize_slice_validation,
)
from strategy import STRATEGY_MODE_TREND_PULLBACK


DEFAULT_CONFIG_PATH = Path("config") / "trend_pullback.example.json"
DEFAULT_BASELINE_SYMBOLS = ("AMD", "HON", "C", "JPM")
DEFAULT_FORWARD_HORIZONS = (4, 6, 8, 10, 15, 20, 30)
DEFAULT_BASELINE_HOLD = 20


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
    validation: Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run robustness and generalization validation for trend_pullback on the clean SIP regular-session dataset."
    )
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH), help="Trend-pullback runtime config JSON.")
    parser.add_argument("--dataset", required=True, help="Clean SIP regular-session dataset directory.")
    parser.add_argument("--symbols", nargs="*", help="Baseline symbol subset.")
    parser.add_argument("--adjacent-symbols", nargs="*", default=[], help="Optional adjacent symbols to test as baseline-plus cases.")
    parser.add_argument("--baseline-hold-bars", type=int, default=DEFAULT_BASELINE_HOLD, help="Baseline realized hold horizon.")
    parser.add_argument("--forward-horizons", default="4,6,8,10,15,20,30", help="Comma-separated forward-return horizons.")
    parser.add_argument("--start-date", help="Optional UTC date filter (YYYY-MM-DD).")
    parser.add_argument("--end-date", help="Optional UTC date filter (YYYY-MM-DD).")
    parser.add_argument("--commission-per-order", type=float, default=0.01)
    parser.add_argument("--slippage-per-share", type=float, default=0.05)
    parser.add_argument("--position-size", type=float, default=1000.0)
    parser.add_argument(
        "--sections",
        default="all",
        help="Comma-separated subset: all,time,params,symbols,path.",
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
    runtime_symbols = _resolve_symbols(runtime, None)
    if runtime_symbols:
        return runtime_symbols
    return DEFAULT_BASELINE_SYMBOLS


def _parse_sections(raw: str) -> set[str]:
    if raw.strip().lower() == "all":
        return {"time", "params", "symbols", "path"}
    return {part.strip().lower() for part in raw.split(",") if part.strip()}


def build_clean_parameter_specs(base_runtime: dict[str, Any], *, symbols: tuple[str, ...], baseline_hold_bars: int) -> list[ScenarioSpec]:
    threshold_baseline = float(base_runtime.get("trend_pullback_entry_threshold", 0.0015))
    min_adx_baseline = float(base_runtime.get("trend_pullback_min_adx", 20.0))

    def spec(name: str, overrides: dict[str, Any]) -> ScenarioSpec:
        return ScenarioSpec(name=name, section="params", symbols=symbols, runtime_overrides=overrides)

    threshold_values = tuple(
        sorted(
            {
                round(max(0.0, threshold_baseline - 0.0005), 6),
                round(threshold_baseline, 6),
                round(threshold_baseline + 0.0005, 6),
            }
        )
    )
    hold_values = tuple(sorted({max(4, baseline_hold_bars - 5), baseline_hold_bars, baseline_hold_bars + 10}))

    return (
        [spec(f"hold_bars={value}", {"trend_pullback_hold_bars": value}) for value in hold_values]
        + [spec(f"entry_threshold={value:.4f}", {"trend_pullback_entry_threshold": value}) for value in threshold_values]
        + [
            spec(f"min_adx={value:.1f}", {"trend_pullback_min_adx": value})
            for value in (max(0.0, min_adx_baseline - 5.0), min_adx_baseline, min_adx_baseline + 5.0)
        ]
    )


def _run_scenario(
    *,
    config_path: Path,
    base_runtime: dict[str, Any],
    dataset_path: str,
    spec: ScenarioSpec,
    forward_horizons: tuple[int, ...],
    commission_per_order: float,
    slippage_per_share: float,
    position_size: float,
) -> ScenarioRun:
    runtime = _runtime_with_overrides(base_runtime, spec.runtime_overrides, spec.symbols)
    validation = _run_long_horizon_scenario(
        config_path=config_path,
        runtime=runtime,
        source_dataset=dataset_path,
        dataset_override=dataset_path,
        symbols=spec.symbols,
        start_date=spec.start_date,
        end_date=spec.end_date,
        commission_per_order=commission_per_order,
        slippage_per_share=slippage_per_share,
        position_size=position_size,
        output_dir=None,
        variant_name=spec.name,
        horizons=forward_horizons,
    )
    return ScenarioRun(spec=spec, validation=validation)


def _best_raw_summary(signals_df: pd.DataFrame, horizons: tuple[int, ...]) -> tuple[int, float]:
    rows = []
    for horizon in horizons:
        column = f"fwd_{horizon}b_net_pct"
        if column not in signals_df.columns:
            continue
        series = pd.to_numeric(signals_df[column], errors="coerce").dropna()
        if series.empty:
            continue
        rows.append((horizon, float(series.mean())))
    if not rows:
        return 0, 0.0
    return max(rows, key=lambda item: (item[1], -item[0]))


def _scenario_summary(run: ScenarioRun, *, forward_horizons: tuple[int, ...]) -> dict[str, Any]:
    best_horizon, best_raw = _best_raw_summary(run.validation.signals_df, forward_horizons)
    result = run.validation.backtest_result
    return {
        "section": run.spec.section,
        "scenario": run.spec.name,
        "symbols": ",".join(run.spec.symbols),
        "start_date": run.spec.start_date,
        "end_date": run.spec.end_date,
        "signal_count": int(len(run.validation.signals_df)),
        "trade_count": int(len(run.validation.closed_trades_df)),
        "realized_pnl": float(result.get("realized_pnl", 0.0)),
        "expectancy": float(result.get("expectancy", 0.0)),
        "win_rate": float(result.get("win_rate", 0.0)),
        "profit_factor": float(result.get("profit_factor", 0.0)),
        "max_drawdown_pct": float(result.get("max_drawdown_pct", 0.0)),
        "avg_holding_period": _average_holding_period(run.validation.closed_trades_df),
        "best_raw_horizon_bars": best_horizon,
        "best_raw_net_expectancy_pct": best_raw,
    }


def _summaries_frame(runs: list[ScenarioRun], *, forward_horizons: tuple[int, ...]) -> pd.DataFrame:
    rows = [_scenario_summary(run, forward_horizons=forward_horizons) for run in runs]
    return pd.DataFrame(rows) if rows else pd.DataFrame(
        columns=[
            "section",
            "scenario",
            "symbols",
            "start_date",
            "end_date",
            "signal_count",
            "trade_count",
            "realized_pnl",
            "expectancy",
            "win_rate",
            "profit_factor",
            "max_drawdown_pct",
            "avg_holding_period",
            "best_raw_horizon_bars",
            "best_raw_net_expectancy_pct",
        ]
    )


def compute_symbol_concentration(per_symbol_df: pd.DataFrame) -> dict[str, Any]:
    if per_symbol_df.empty or "realized_pnl" not in per_symbol_df.columns:
        return {"top_symbol": None, "top_positive_share": 0.0}
    positive = per_symbol_df[per_symbol_df["realized_pnl"] > 0].copy()
    if positive.empty:
        return {"top_symbol": None, "top_positive_share": 0.0}
    positive = positive.sort_values(["realized_pnl", "symbol"], ascending=[False, True]).reset_index(drop=True)
    total_positive = float(positive["realized_pnl"].sum())
    top_row = positive.iloc[0]
    return {
        "top_symbol": str(top_row["symbol"]),
        "top_positive_share": (float(top_row["realized_pnl"]) / total_positive) if total_positive > 0 else 0.0,
    }


def summarize_trade_path_sanity(
    *,
    baseline_run: ScenarioRun,
    baseline_hold_bars: int,
    forward_horizons: tuple[int, ...],
    commission_per_order: float,
    slippage_per_share: float,
    position_size: float,
) -> tuple[pd.DataFrame, dict[str, Any], dict[str, Any], pd.DataFrame]:
    inputs, state = _prepare_inputs_and_state(baseline_run.validation.context)
    compare_hold = max(4, baseline_hold_bars - 5)
    path_horizon = max(max(forward_horizons), baseline_hold_bars)
    signal_paths_df = build_signal_path_table(
        baseline_run.validation.signals_df,
        state=state,
        slippage=slippage_per_share,
        commission=commission_per_order,
        position_size=position_size,
        path_horizon=path_horizon,
        horizons=tuple(sorted({compare_hold, baseline_hold_bars})),
    )
    trade_paths_df = attach_trade_path_metrics(baseline_run.validation.closed_trades_df, signal_paths_df)
    hold_delta = summarize_hold_delta(signal_paths_df, hold_a=compare_hold, hold_b=baseline_hold_bars)
    opportunity = summarize_opportunity_capture(trade_paths_df)
    shape_summary = build_trade_shape_summary(trade_paths_df)
    sanity_row = pd.DataFrame(
        [
            {
                "compare_hold_bars": compare_hold,
                "baseline_hold_bars": baseline_hold_bars,
                "trade_count": int(len(trade_paths_df)),
                "avg_realized_return_pct": float(opportunity.get("avg_realized_return_pct", 0.0)),
                "avg_best_net_exit_pct": float(opportunity.get("avg_best_net_exit_pct", 0.0)),
                "avg_missed_opportunity_pct": float(opportunity.get("avg_missed_opportunity_pct", 0.0)),
                "materially_worse_than_best_frac": float(opportunity.get("materially_worse_than_best_frac", 0.0)),
                "avg_delta_baseline_minus_compare_pct": float(hold_delta.get(f"avg_delta_{baseline_hold_bars}m{compare_hold}_pct", 0.0)),
                "giveback_frac_pct": float(hold_delta.get(f"giveback_between_{compare_hold}b_{baseline_hold_bars}b_frac", 0.0)),
                "peak_before_baseline_exit_frac": float(hold_delta.get("peak_before_later_exit_frac", 0.0)),
            }
        ]
    )
    return sanity_row, hold_delta, opportunity, shape_summary


def classify_clean_robustness(
    *,
    baseline_summary: dict[str, Any],
    time_df: pd.DataFrame,
    param_df: pd.DataFrame,
    symbol_df: pd.DataFrame,
    symbol_concentration: dict[str, Any],
    path_sanity: pd.DataFrame,
) -> tuple[str, str]:
    trade_count = int(baseline_summary.get("trade_count", 0))
    expectancy = float(baseline_summary.get("expectancy", 0.0))
    best_raw = float(baseline_summary.get("best_raw_net_expectancy_pct", 0.0))

    if trade_count <= 0 or best_raw <= 0 or expectancy <= 0:
        return "reject", "Baseline clean-SIP performance is not convincingly positive after stress begins."

    if trade_count < 20:
        return "research-only", "The baseline is positive but the closed-trade sample is too small to trust."

    time_positive_ratio = float((time_df["expectancy"] > 0).mean()) if not time_df.empty else 0.0
    param_positive_ratio = float((param_df["expectancy"] > 0).mean()) if not param_df.empty else 0.0
    leave_df = symbol_df[symbol_df["scenario"].str.startswith("leave_out:")] if not symbol_df.empty else pd.DataFrame()
    leave_positive_ratio = float((leave_df["expectancy"] > 0).mean()) if not leave_df.empty else 0.0
    top_positive_share = float(symbol_concentration.get("top_positive_share", 0.0))
    avg_missed = float(path_sanity.iloc[0]["avg_missed_opportunity_pct"]) if not path_sanity.empty else 0.0
    giveback_frac = float(path_sanity.iloc[0]["giveback_frac_pct"]) if not path_sanity.empty else 0.0

    if time_positive_ratio < 0.4 or param_positive_ratio < 0.4 or leave_positive_ratio < 0.4:
        return "research-only", "The edge turns inconsistent once you stress time slices, nearby parameters, or leave-one-out symbols."

    if top_positive_share > 0.70:
        return "research-only", "Too much of the positive contribution comes from a single symbol."

    if giveback_frac > 70.0 and avg_missed > 0.10:
        return "research-only", "The edge is positive, but the realized path still gives too much back late."

    if time_positive_ratio < 0.67 or param_positive_ratio < 0.67 or leave_positive_ratio < 0.67 or top_positive_share > 0.50:
        return "promising but narrow", "The edge survives, but it still looks concentrated or sensitive rather than broad."

    if expectancy > 1.5 and best_raw > 0.15 and top_positive_share < 0.40 and min(time_positive_ratio, param_positive_ratio, leave_positive_ratio) >= 0.75:
        return "candidate for constrained paper testing", "The clean-SIP result stayed positive across most nearby stress checks with limited concentration."

    return "stable enough for deeper research", "The edge stayed positive across most nearby checks and does not look like a single-slice artifact."


def _print_section(title: str, df: pd.DataFrame, *, max_rows: int | None = None) -> None:
    print(f"\n=== {title} ===")
    print(_frame_text(df, max_rows=max_rows))


def _save_outputs(
    *,
    output_dir: Path,
    summary_tables: dict[str, pd.DataFrame],
    classification: tuple[str, str],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, df in summary_tables.items():
        df.to_csv(output_dir / f"{name}.csv", index=False)
    payload = {"classification": classification[0], "reason": classification[1]}
    (output_dir / "robustness_clean_summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main() -> None:
    args = parse_args()
    config_path = Path(args.config)
    dataset_path = str(Path(args.dataset))
    runtime, _ = _load_config_runtime(config_path)
    runtime["strategy_mode"] = STRATEGY_MODE_TREND_PULLBACK
    runtime["trend_pullback_hold_bars"] = int(args.baseline_hold_bars)
    baseline_symbols = _resolve_baseline_symbols(runtime, args.symbols)
    adjacent_symbols = _normalize_symbol_list(args.adjacent_symbols)
    forward_horizons = _parse_horizon_list(args.forward_horizons)
    selected_sections = _parse_sections(args.sections)

    baseline_spec = ScenarioSpec(
        name="baseline",
        section="baseline",
        symbols=baseline_symbols,
        runtime_overrides={"trend_pullback_hold_bars": int(args.baseline_hold_bars)},
        start_date=args.start_date,
        end_date=args.end_date,
    )
    baseline_run = _run_scenario(
        config_path=config_path,
        base_runtime=runtime,
        dataset_path=dataset_path,
        spec=baseline_spec,
        forward_horizons=forward_horizons,
        commission_per_order=args.commission_per_order,
        slippage_per_share=args.slippage_per_share,
        position_size=args.position_size,
    )
    baseline_summary = _scenario_summary(baseline_run, forward_horizons=forward_horizons)
    per_symbol_df = summarize_realized_by_symbol(baseline_run.validation.backtest_result)
    symbol_concentration = compute_symbol_concentration(per_symbol_df)

    print("\n=== Trend Pullback Clean Robustness ===")
    print(f"Config:   {config_path}")
    print(f"Dataset:  {dataset_path}")
    print(f"Symbols:  {', '.join(baseline_symbols)}")
    print(f"Hold:     {args.baseline_hold_bars} bars")
    print("\nBaseline summary:")
    print(_frame_text(pd.DataFrame([baseline_summary])))

    summary_tables: dict[str, pd.DataFrame] = {"baseline": pd.DataFrame([baseline_summary]), "baseline_per_symbol": per_symbol_df}

    if "time" in selected_sections:
        time_runs: list[ScenarioRun] = []
        for spec in build_month_slice_specs(baseline_run.validation.evaluations_df, runtime_overrides={}, symbols=baseline_symbols):
            time_runs.append(
                _run_scenario(
                    config_path=config_path,
                    base_runtime=runtime,
                    dataset_path=dataset_path,
                    spec=spec,
                    forward_horizons=forward_horizons,
                    commission_per_order=args.commission_per_order,
                    slippage_per_share=args.slippage_per_share,
                    position_size=args.position_size,
                )
            )
        time_df = _summaries_frame(time_runs, forward_horizons=forward_horizons)
        summary_tables["time_slices"] = time_df
        summary_tables["baseline_month_slices"] = summarize_slice_validation(
            baseline_run.validation.signals_df,
            baseline_run.validation.closed_trades_df,
            group_cols=["month"],
            horizon_bars=int(args.baseline_hold_bars),
        )
        summary_tables["baseline_time_buckets"] = summarize_slice_validation(
            baseline_run.validation.signals_df,
            baseline_run.validation.closed_trades_df,
            group_cols=["time_bucket"],
            horizon_bars=int(args.baseline_hold_bars),
        )
        summary_tables["baseline_volatility_regimes"] = summarize_slice_validation(
            baseline_run.validation.signals_df,
            baseline_run.validation.closed_trades_df,
            group_cols=["volatility_regime"],
            horizon_bars=int(args.baseline_hold_bars),
        )
        _print_section("Time Slice Robustness", time_df, max_rows=24)
        _print_section("Baseline Month Slices", summary_tables["baseline_month_slices"], max_rows=16)

    else:
        time_df = pd.DataFrame()

    if "params" in selected_sections:
        param_runs: list[ScenarioRun] = []
        for spec in build_clean_parameter_specs(runtime, symbols=baseline_symbols, baseline_hold_bars=int(args.baseline_hold_bars)):
            param_runs.append(
                _run_scenario(
                    config_path=config_path,
                    base_runtime=runtime,
                    dataset_path=dataset_path,
                    spec=spec,
                    forward_horizons=forward_horizons,
                    commission_per_order=args.commission_per_order,
                    slippage_per_share=args.slippage_per_share,
                    position_size=args.position_size,
                )
            )
        param_df = _summaries_frame(param_runs, forward_horizons=forward_horizons)
        summary_tables["parameter_perturbations"] = param_df
        _print_section("Parameter Perturbations", param_df, max_rows=24)
    else:
        param_df = pd.DataFrame()

    if "symbols" in selected_sections:
        symbol_runs: list[ScenarioRun] = []
        for spec in build_symbol_robustness_specs(
            baseline_symbols,
            runtime_overrides={"trend_pullback_hold_bars": int(args.baseline_hold_bars)},
            adjacent_symbols=adjacent_symbols,
        ):
            symbol_runs.append(
                _run_scenario(
                    config_path=config_path,
                    base_runtime=runtime,
                    dataset_path=dataset_path,
                    spec=spec,
                    forward_horizons=forward_horizons,
                    commission_per_order=args.commission_per_order,
                    slippage_per_share=args.slippage_per_share,
                    position_size=args.position_size,
                )
            )
        symbol_df = _summaries_frame(symbol_runs, forward_horizons=forward_horizons)
        summary_tables["symbol_generalization"] = symbol_df
        _print_section("Symbol Generalization", symbol_df, max_rows=24)
    else:
        symbol_df = pd.DataFrame()

    if "path" in selected_sections:
        path_sanity, hold_delta, opportunity, shape_summary = summarize_trade_path_sanity(
            baseline_run=baseline_run,
            baseline_hold_bars=int(args.baseline_hold_bars),
            forward_horizons=forward_horizons,
            commission_per_order=args.commission_per_order,
            slippage_per_share=args.slippage_per_share,
            position_size=args.position_size,
        )
        summary_tables["trade_path_sanity"] = path_sanity
        summary_tables["trade_shape_summary"] = shape_summary
        _print_section("Trade Path Sanity", path_sanity)
        _print_section("Trade Shape Summary", shape_summary, max_rows=12)
    else:
        path_sanity = pd.DataFrame()
        hold_delta = {}
        opportunity = {}

    classification = classify_clean_robustness(
        baseline_summary=baseline_summary,
        time_df=time_df,
        param_df=param_df,
        symbol_df=symbol_df,
        symbol_concentration=symbol_concentration,
        path_sanity=path_sanity,
    )

    print("\n=== Classification ===")
    print(classification[0])
    print(classification[1])

    print("\n=== Symbol Concentration ===")
    print(_frame_text(pd.DataFrame([symbol_concentration])))
    if hold_delta:
        print("\n=== Hold Horizon Sanity ===")
        print(_frame_text(pd.DataFrame([hold_delta])))
    if opportunity:
        print("\n=== Opportunity Capture ===")
        print(_frame_text(pd.DataFrame([opportunity])))

    if args.output_dir:
        _save_outputs(output_dir=Path(args.output_dir), summary_tables=summary_tables, classification=classification)


if __name__ == "__main__":
    main()

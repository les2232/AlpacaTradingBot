from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from audit_dataset_spacing import audit_spacing
from research.experiment_log import log_experiment_run
from run_edge_diagnostics import summarize_fixed_horizon_metrics
from run_long_horizon_validation import (
    _parse_horizon_list,
    _run_long_horizon_scenario,
    _runtime_with_overrides,
    build_hold_horizon_specs,
    choose_representative_horizon,
    classify_long_horizon,
    summarize_hold_ladder,
)
from run_volatility_expansion_validation import (
    DEFAULT_HORIZONS,
    STRATEGY_MODE_TREND_PULLBACK,
    STRATEGY_MODE_VOLATILITY_EXPANSION,
    ValidationRun,
    _average_holding_period,
    _comparison_summary_df,
    _load_config_runtime,
    _resolve_symbols,
    _run_validation_scenario,
    classify_validation,
)


DEFAULT_FORWARD_HORIZONS = "4,6,8,10,15,20,30"
DEFAULT_HOLD_HORIZONS = "4,8,10,15,20,30"


@dataclass(frozen=True)
class FeedStrategySummary:
    strategy_mode: str
    feed_label: str
    signal_count: int
    trade_count: int
    realized_pnl: float
    expectancy: float
    win_rate: float
    profit_factor: float
    best_raw_horizon_bars: int
    best_raw_net_expectancy_pct: float
    classification_label: str
    classification_reason: str
    spacing_label: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Re-run a small SIP vs IEX research comparison for the most informative strategies."
    )
    parser.add_argument("--iex-dataset", required=True, help="Reference IEX dataset.")
    parser.add_argument("--sip-dataset", required=True, help="Rebuilt SIP dataset.")
    parser.add_argument("--trend-config", default="config\\trend_pullback.example.json", help="Trend-pullback config JSON.")
    parser.add_argument("--volatility-config", default="config\\volatility_expansion.example.json", help="Volatility-expansion config JSON.")
    parser.add_argument("--symbols", nargs="*", help="Optional symbol override shared across both strategies.")
    parser.add_argument("--forward-horizons", default=DEFAULT_FORWARD_HORIZONS)
    parser.add_argument("--hold-horizons", default=DEFAULT_HOLD_HORIZONS)
    parser.add_argument("--commission-per-order", type=float, default=0.01)
    parser.add_argument("--slippage-per-share", type=float, default=0.05)
    parser.add_argument("--position-size", type=float, default=1000.0)
    parser.add_argument("--output-dir", help="Optional directory for CSV/JSON artifacts.")
    return parser.parse_args()


def _best_raw_summary(signals_df: pd.DataFrame, horizons: tuple[int, ...]) -> tuple[int, float]:
    overall = summarize_fixed_horizon_metrics(signals_df, group_cols=[], horizons=horizons)
    if overall.empty:
        return 0, 0.0
    row = overall.sort_values(["avg_net_expectancy_pct", "horizon_bars"], ascending=[False, True]).iloc[0]
    return int(row["horizon_bars"]), float(row["avg_net_expectancy_pct"])


def _build_shared_symbols(
    *,
    cli_symbols: list[str] | None,
    volatility_runtime: dict[str, Any],
    trend_runtime: dict[str, Any],
) -> tuple[str, ...]:
    if cli_symbols:
        return tuple(dict.fromkeys(str(symbol).strip().upper() for symbol in cli_symbols if str(symbol).strip()))
    symbols = _resolve_symbols(volatility_runtime, None)
    if symbols:
        return symbols
    symbols = _resolve_symbols(trend_runtime, None)
    if symbols:
        return symbols
    raise RuntimeError("No symbols resolved. Pass --symbols or add symbols to one of the configs.")


def _run_trend_pullback_long(
    *,
    config_path: Path,
    dataset_path: Path,
    runtime: dict[str, Any],
    symbols: tuple[str, ...],
    forward_horizons: tuple[int, ...],
    hold_horizons: tuple[int, ...],
    commission_per_order: float,
    slippage_per_share: float,
    position_size: float,
) -> FeedStrategySummary:
    run = _run_long_horizon_scenario(
        config_path=config_path,
        runtime=runtime,
        source_dataset=str(dataset_path),
        dataset_override=str(dataset_path),
        symbols=symbols,
        start_date=None,
        end_date=None,
        commission_per_order=commission_per_order,
        slippage_per_share=slippage_per_share,
        position_size=position_size,
        output_dir=None,
        variant_name="trend_pullback_long",
        horizons=forward_horizons,
    )
    hold_runs = []
    for spec in build_hold_horizon_specs(STRATEGY_MODE_TREND_PULLBACK, hold_horizons):
        hold_runtime = _runtime_with_overrides(runtime, spec["runtime_overrides"], symbols)
        hold_run = _run_long_horizon_scenario(
            config_path=config_path,
            runtime=hold_runtime,
            source_dataset=str(dataset_path),
            dataset_override=str(dataset_path),
            symbols=symbols,
            start_date=None,
            end_date=None,
            commission_per_order=commission_per_order,
            slippage_per_share=slippage_per_share,
            position_size=position_size,
            output_dir=None,
            variant_name=f"trend_pullback_hold_{spec['hold_bars']}",
            horizons=forward_horizons,
        )
        hold_runs.append(type("HoldRunLike", (), {"hold_bars": int(spec["hold_bars"]), "run": hold_run})())
    hold_ladder = summarize_hold_ladder(hold_runs)
    label, reason = classify_long_horizon(
        raw_overall=summarize_fixed_horizon_metrics(run.signals_df, group_cols=[], horizons=forward_horizons),
        hold_ladder=hold_ladder,
    )
    best_horizon, best_raw = _best_raw_summary(run.signals_df, forward_horizons)
    spacing_label = audit_spacing(dataset_path, list(symbols), "15Min")["classification"][0]
    best_hold_row = hold_ladder.sort_values(["expectancy", "hold_bars"], ascending=[False, True]).iloc[0] if not hold_ladder.empty else None
    return FeedStrategySummary(
        strategy_mode=STRATEGY_MODE_TREND_PULLBACK,
        feed_label=str(dataset_path),
        signal_count=len(run.signals_df),
        trade_count=len(run.closed_trades_df),
        realized_pnl=float(best_hold_row["realized_pnl"]) if best_hold_row is not None else 0.0,
        expectancy=float(best_hold_row["expectancy"]) if best_hold_row is not None else 0.0,
        win_rate=float(best_hold_row["win_rate"]) if best_hold_row is not None else 0.0,
        profit_factor=float(best_hold_row["profit_factor"]) if best_hold_row is not None else 0.0,
        best_raw_horizon_bars=best_horizon,
        best_raw_net_expectancy_pct=best_raw,
        classification_label=label,
        classification_reason=reason,
        spacing_label=spacing_label,
    )


def _run_volatility_expansion_validation(
    *,
    config_path: Path,
    dataset_path: Path,
    runtime: dict[str, Any],
    symbols: tuple[str, ...],
    commission_per_order: float,
    slippage_per_share: float,
    position_size: float,
) -> FeedStrategySummary:
    run = _run_validation_scenario(
        config_path=config_path,
        runtime=runtime,
        source_dataset=str(dataset_path),
        dataset_override=str(dataset_path),
        symbols=symbols,
        start_date=None,
        end_date=None,
        commission_per_order=commission_per_order,
        slippage_per_share=slippage_per_share,
        position_size=position_size,
        output_dir=None,
        variant_name="volatility_expansion_validation",
    )
    comparison_df = _comparison_summary_df([run])
    label, reason = classify_validation(
        baseline_run=run,
        perturbation_df=pd.DataFrame(columns=["expectancy"]),
        comparison_df=comparison_df,
    )
    best_horizon, best_raw = _best_raw_summary(run.signals_df, DEFAULT_HORIZONS)
    spacing_label = audit_spacing(dataset_path, list(symbols), "15Min")["classification"][0]
    return FeedStrategySummary(
        strategy_mode=STRATEGY_MODE_VOLATILITY_EXPANSION,
        feed_label=str(dataset_path),
        signal_count=len(run.signals_df),
        trade_count=len(run.closed_trades_df),
        realized_pnl=float(run.backtest_result.get("realized_pnl", 0.0)),
        expectancy=float(run.backtest_result.get("expectancy", 0.0)),
        win_rate=float(run.backtest_result.get("win_rate", 0.0)),
        profit_factor=float(run.backtest_result.get("profit_factor", 0.0)),
        best_raw_horizon_bars=best_horizon,
        best_raw_net_expectancy_pct=best_raw,
        classification_label=label,
        classification_reason=reason,
        spacing_label=spacing_label,
    )


def classify_material_difference(iex: FeedStrategySummary, sip: FeedStrategySummary) -> tuple[str, str]:
    label_changed = iex.classification_label != sip.classification_label
    raw_sign_changed = (iex.best_raw_net_expectancy_pct > 0) != (sip.best_raw_net_expectancy_pct > 0)
    expectancy_sign_changed = (iex.expectancy > 0) != (sip.expectancy > 0)
    signal_delta_ratio = abs(sip.signal_count - iex.signal_count) / max(1, iex.signal_count)
    expectancy_delta = sip.expectancy - iex.expectancy
    if label_changed or raw_sign_changed or expectancy_sign_changed:
        return "material difference", "The feed switch changes the directional research conclusion or the qualitative classification."
    if signal_delta_ratio >= 0.25 or abs(expectancy_delta) >= 1.0:
        return "requires re-evaluation", "The headline classification stayed similar, but the signal count or expectancy shifted enough to warrant a closer follow-up."
    if abs(expectancy_delta) >= 0.25 or abs(sip.best_raw_net_expectancy_pct - iex.best_raw_net_expectancy_pct) >= 0.05:
        return "slightly cleaner but same conclusion", "SIP changes the metrics somewhat, but not enough to overturn the core conclusion."
    return "mostly unchanged", "SIP leaves the strategy conclusion effectively the same on this sample."


def _summary_frame(rows: list[FeedStrategySummary]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "strategy_mode": row.strategy_mode,
                "feed": Path(row.feed_label).name,
                "signal_count": row.signal_count,
                "trade_count": row.trade_count,
                "realized_pnl": row.realized_pnl,
                "expectancy": row.expectancy,
                "win_rate": row.win_rate,
                "profit_factor": row.profit_factor,
                "best_raw_horizon_bars": row.best_raw_horizon_bars,
                "best_raw_net_expectancy_pct": row.best_raw_net_expectancy_pct,
                "classification": row.classification_label,
                "spacing": row.spacing_label,
            }
            for row in rows
        ]
    )


def _print_strategy_comparison(strategy: str, iex: FeedStrategySummary, sip: FeedStrategySummary, materiality: tuple[str, str]) -> None:
    print(f"\nFEED COMPARISON · {strategy}")
    print("--------------------------------")
    print("IEX:")
    print(f"  raw best horizon: {iex.best_raw_horizon_bars} bars @ {iex.best_raw_net_expectancy_pct:.3f}%")
    print(f"  realized PnL: {iex.realized_pnl:.2f}")
    print(f"  expectancy/trade: {iex.expectancy:.2f}")
    print(f"  signals/trades: {iex.signal_count}/{iex.trade_count}")
    print(f"  classification: {iex.classification_label}")
    print("SIP:")
    print(f"  raw best horizon: {sip.best_raw_horizon_bars} bars @ {sip.best_raw_net_expectancy_pct:.3f}%")
    print(f"  realized PnL: {sip.realized_pnl:.2f}")
    print(f"  expectancy/trade: {sip.expectancy:.2f}")
    print(f"  signals/trades: {sip.signal_count}/{sip.trade_count}")
    print(f"  classification: {sip.classification_label}")
    print("DIFFERENCE:")
    print(f"  signal count delta: {sip.signal_count - iex.signal_count}")
    print(f"  expectancy delta: {sip.expectancy - iex.expectancy:.2f}")
    print(f"  conclusion: {materiality[0]}")
    print(f"  why: {materiality[1]}")


def main() -> None:
    args = parse_args()
    iex_dataset = Path(args.iex_dataset)
    sip_dataset = Path(args.sip_dataset)
    trend_config = Path(args.trend_config)
    volatility_config = Path(args.volatility_config)
    forward_horizons = _parse_horizon_list(args.forward_horizons)
    hold_horizons = _parse_horizon_list(args.hold_horizons)

    trend_runtime, _ = _load_config_runtime(trend_config)
    vol_runtime, _ = _load_config_runtime(volatility_config)
    shared_symbols = _build_shared_symbols(cli_symbols=args.symbols, volatility_runtime=vol_runtime, trend_runtime=trend_runtime)

    trend_runtime["strategy_mode"] = STRATEGY_MODE_TREND_PULLBACK
    vol_runtime["strategy_mode"] = STRATEGY_MODE_VOLATILITY_EXPANSION
    trend_runtime = _runtime_with_overrides(trend_runtime, {}, shared_symbols)
    vol_runtime = _runtime_with_overrides(vol_runtime, {}, shared_symbols)

    trend_iex = _run_trend_pullback_long(
        config_path=trend_config,
        dataset_path=iex_dataset,
        runtime=trend_runtime,
        symbols=shared_symbols,
        forward_horizons=forward_horizons,
        hold_horizons=hold_horizons,
        commission_per_order=args.commission_per_order,
        slippage_per_share=args.slippage_per_share,
        position_size=args.position_size,
    )
    trend_sip = _run_trend_pullback_long(
        config_path=trend_config,
        dataset_path=sip_dataset,
        runtime=trend_runtime,
        symbols=shared_symbols,
        forward_horizons=forward_horizons,
        hold_horizons=hold_horizons,
        commission_per_order=args.commission_per_order,
        slippage_per_share=args.slippage_per_share,
        position_size=args.position_size,
    )
    vol_iex = _run_volatility_expansion_validation(
        config_path=volatility_config,
        dataset_path=iex_dataset,
        runtime=vol_runtime,
        symbols=shared_symbols,
        commission_per_order=args.commission_per_order,
        slippage_per_share=args.slippage_per_share,
        position_size=args.position_size,
    )
    vol_sip = _run_volatility_expansion_validation(
        config_path=volatility_config,
        dataset_path=sip_dataset,
        runtime=vol_runtime,
        symbols=shared_symbols,
        commission_per_order=args.commission_per_order,
        slippage_per_share=args.slippage_per_share,
        position_size=args.position_size,
    )

    trend_materiality = classify_material_difference(trend_iex, trend_sip)
    vol_materiality = classify_material_difference(vol_iex, vol_sip)

    _print_strategy_comparison(STRATEGY_MODE_TREND_PULLBACK, trend_iex, trend_sip, trend_materiality)
    _print_strategy_comparison(STRATEGY_MODE_VOLATILITY_EXPANSION, vol_iex, vol_sip, vol_materiality)

    summary_df = _summary_frame([trend_iex, trend_sip, vol_iex, vol_sip])
    print("\nSUMMARY TABLE")
    print("-------------")
    print(summary_df.to_string(index=False))

    if args.output_dir:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        summary_df.to_csv(output_dir / "feed_comparison_summary.csv", index=False)
        payload = {
            "trend_pullback": {"materiality": trend_materiality[0], "reason": trend_materiality[1]},
            "volatility_expansion": {"materiality": vol_materiality[0], "reason": vol_materiality[1]},
        }
        (output_dir / "feed_comparison_summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    log_experiment_run(
        run_type="feed_comparison_validation",
        script_path=__file__,
        strategy_name="multi_strategy",
        symbols=shared_symbols,
        params={
            "iex_dataset": args.iex_dataset,
            "sip_dataset": args.sip_dataset,
            "trend_config": args.trend_config,
            "volatility_config": args.volatility_config,
            "forward_horizons": list(forward_horizons),
            "hold_horizons": list(hold_horizons),
            "commission_per_order": args.commission_per_order,
            "slippage_per_share": args.slippage_per_share,
            "position_size": args.position_size,
        },
        metrics={
            "trade_count": float(summary_df["trade_count"].sum()) if "trade_count" in summary_df.columns else None,
            "expectancy": float(summary_df["expectancy"].mean()) if "expectancy" in summary_df.columns else None,
            "profit_factor": float(summary_df["profit_factor"].mean()) if "profit_factor" in summary_df.columns else None,
            "win_rate": float(summary_df["win_rate"].mean()) if "win_rate" in summary_df.columns else None,
        },
        output_path=args.output_dir,
        summary_path=(Path(args.output_dir) / "feed_comparison_summary.json") if args.output_dir else None,
        extra_fields={
            "trend_pullback_materiality": {"label": trend_materiality[0], "reason": trend_materiality[1]},
            "volatility_expansion_materiality": {"label": vol_materiality[0], "reason": vol_materiality[1]},
        },
    )


if __name__ == "__main__":
    main()

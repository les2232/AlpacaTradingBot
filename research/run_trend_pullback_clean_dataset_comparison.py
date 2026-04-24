from __future__ import annotations

import argparse
import json
from pathlib import Path

from research.experiment_log import log_experiment_run
from run_feed_comparison_validation import _run_trend_pullback_long, _summary_frame
from run_long_horizon_validation import _parse_horizon_list, _runtime_with_overrides
from run_volatility_expansion_validation import (
    STRATEGY_MODE_TREND_PULLBACK,
    _load_config_runtime,
    _resolve_symbols,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare trend_pullback on IEX shared, SIP shared, and cleaner SIP datasets."
    )
    parser.add_argument("--iex-dataset", required=True)
    parser.add_argument("--sip-shared-dataset", required=True)
    parser.add_argument("--sip-clean-dataset", required=True)
    parser.add_argument("--config", default="config\\trend_pullback.example.json")
    parser.add_argument("--symbols", nargs="*")
    parser.add_argument("--forward-horizons", default="4,6,8,10,15,20,30")
    parser.add_argument("--hold-horizons", default="4,8,10,15,20,30")
    parser.add_argument("--commission-per-order", type=float, default=0.01)
    parser.add_argument("--slippage-per-share", type=float, default=0.05)
    parser.add_argument("--position-size", type=float, default=1000.0)
    parser.add_argument("--output-dir")
    return parser.parse_args()


def classify_clean_dataset_materiality(iex_summary, sip_shared_summary, sip_clean_summary) -> tuple[str, str]:
    if sip_clean_summary.classification_label != iex_summary.classification_label:
        return (
            "materially different, re-evaluate",
            "The cleaner SIP dataset changes the qualitative trend_pullback conclusion versus the IEX baseline.",
        )
    if sip_clean_summary.expectancy > sip_shared_summary.expectancy and sip_clean_summary.expectancy > iex_summary.expectancy:
        return (
            "cleaner and modestly better",
            "The cleaner SIP dataset improves realized expectancy while preserving the same strategy thesis.",
        )
    if (
        abs(sip_clean_summary.expectancy - iex_summary.expectancy) < 0.5
        and abs(sip_clean_summary.best_raw_net_expectancy_pct - iex_summary.best_raw_net_expectancy_pct) < 0.05
    ):
        return (
            "mostly cleaner, same conclusion",
            "The cleaner dataset improves structure more than it changes the research result.",
        )
    return (
        "not worth the added complexity",
        "The cleaner dataset does not improve the conclusion enough to justify a more complex data standard right now.",
    )


def main() -> None:
    args = parse_args()
    config_path = Path(args.config)
    runtime, _ = _load_config_runtime(config_path)
    runtime["strategy_mode"] = STRATEGY_MODE_TREND_PULLBACK
    symbols = _resolve_symbols(runtime, args.symbols)
    if not symbols:
        raise RuntimeError("No symbols resolved for trend_pullback comparison.")
    runtime = _runtime_with_overrides(runtime, {}, symbols)
    forward_horizons = _parse_horizon_list(args.forward_horizons)
    hold_horizons = _parse_horizon_list(args.hold_horizons)

    iex_summary = _run_trend_pullback_long(
        config_path=config_path,
        dataset_path=Path(args.iex_dataset),
        runtime=runtime,
        symbols=symbols,
        forward_horizons=forward_horizons,
        hold_horizons=hold_horizons,
        commission_per_order=args.commission_per_order,
        slippage_per_share=args.slippage_per_share,
        position_size=args.position_size,
    )
    sip_shared_summary = _run_trend_pullback_long(
        config_path=config_path,
        dataset_path=Path(args.sip_shared_dataset),
        runtime=runtime,
        symbols=symbols,
        forward_horizons=forward_horizons,
        hold_horizons=hold_horizons,
        commission_per_order=args.commission_per_order,
        slippage_per_share=args.slippage_per_share,
        position_size=args.position_size,
    )
    sip_clean_summary = _run_trend_pullback_long(
        config_path=config_path,
        dataset_path=Path(args.sip_clean_dataset),
        runtime=runtime,
        symbols=symbols,
        forward_horizons=forward_horizons,
        hold_horizons=hold_horizons,
        commission_per_order=args.commission_per_order,
        slippage_per_share=args.slippage_per_share,
        position_size=args.position_size,
    )

    materiality = classify_clean_dataset_materiality(iex_summary, sip_shared_summary, sip_clean_summary)

    print("\nTREND PULLBACK - DATASET COMPARISON")
    print("-----------------------------------")
    for label, summary in [
        ("IEX shared", iex_summary),
        ("SIP shared", sip_shared_summary),
        ("SIP clean", sip_clean_summary),
    ]:
        print(f"{label}:")
        print(f"  raw best horizon: {summary.best_raw_horizon_bars} bars @ {summary.best_raw_net_expectancy_pct:.3f}%")
        print(f"  realized PnL: {summary.realized_pnl:.2f}")
        print(f"  expectancy/trade: {summary.expectancy:.2f}")
        print(f"  signals/trades: {summary.signal_count}/{summary.trade_count}")
        print(f"  classification: {summary.classification_label}")
        print(f"  spacing audit: {summary.spacing_label}")

    print("\nMATERIALITY")
    print("-----------")
    print(materiality[0])
    print(materiality[1])

    summary_df = _summary_frame([iex_summary, sip_shared_summary, sip_clean_summary])
    print("\nSUMMARY TABLE")
    print(summary_df.to_string(index=False))

    if args.output_dir:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        summary_df.to_csv(output_dir / "trend_pullback_clean_dataset_comparison.csv", index=False)
        (output_dir / "trend_pullback_clean_dataset_comparison.json").write_text(
            json.dumps({"materiality": materiality[0], "reason": materiality[1]}, indent=2),
            encoding="utf-8",
        )
    log_experiment_run(
        run_type="dataset_comparison",
        script_path=__file__,
        strategy_name=STRATEGY_MODE_TREND_PULLBACK,
        symbols=symbols,
        params={
            "config": args.config,
            "iex_dataset": args.iex_dataset,
            "sip_shared_dataset": args.sip_shared_dataset,
            "sip_clean_dataset": args.sip_clean_dataset,
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
        summary_path=(Path(args.output_dir) / "trend_pullback_clean_dataset_comparison.json") if args.output_dir else None,
        extra_fields={
            "materiality": {"label": materiality[0], "reason": materiality[1]},
        },
    )


if __name__ == "__main__":
    main()

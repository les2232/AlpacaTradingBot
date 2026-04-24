from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from backtest_runner import (
    TREND_PULLBACK_RESEARCH_EXIT_FILL_BAR_CLOSE,
    TREND_PULLBACK_RESEARCH_EXIT_FILL_NEXT_OPEN,
    run_backtest,
)
from research.experiment_log import log_experiment_run
from run_edge_audit import _frame_text, _load_runtime_payload, _prepare_inputs_and_state, build_audit_context, evaluate_raw_entry_signals
from run_trade_path_diagnostics import (
    attach_trade_path_metrics,
    build_cost_drag_table,
    build_signal_path_table,
    pair_round_trip_trades,
    summarize_opportunity_capture,
)
from strategy import (
    STRATEGY_MODE_TREND_PULLBACK,
    TREND_PULLBACK_EXIT_FIXED_BARS,
    TREND_PULLBACK_EXIT_HYBRID_TP_OR_TIME,
)


DEFAULT_CONFIG_PATH = Path("config") / "trend_pullback.example.json"
DEFAULT_SYMBOLS = ("AMD", "JPM", "HON", "C")
DEFAULT_BASELINE_HOLD = 20
DEFAULT_TP_TARGETS = (0.005, 0.0075, 0.01)


@dataclass(frozen=True)
class ExitVariant:
    name: str
    section: str
    exit_style: str
    hold_bars: int
    take_profit_pct: float
    research_exit_fill: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run clean-SIP trend_pullback monetization and exit-capture comparisons."
    )
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH), help="Trend-pullback runtime config JSON.")
    parser.add_argument("--dataset", required=True, help="Clean SIP regular-session dataset directory.")
    parser.add_argument("--symbols", nargs="*", help="Optional symbol override.")
    parser.add_argument("--baseline-hold-bars", type=int, default=DEFAULT_BASELINE_HOLD)
    parser.add_argument("--tp-targets", default="0.005,0.0075,0.01", help="Comma-separated hybrid TP targets as decimal returns.")
    parser.add_argument("--commission-per-order", type=float, default=0.01)
    parser.add_argument("--slippage-per-share", type=float, default=0.05)
    parser.add_argument("--position-size", type=float, default=1000.0)
    parser.add_argument(
        "--sections",
        default="all",
        help="Comma-separated subset: all,baseline,hybrid.",
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
    return DEFAULT_SYMBOLS


def _parse_sections(raw: str) -> set[str]:
    if raw.strip().lower() == "all":
        return {"baseline", "hybrid"}
    return {part.strip().lower() for part in raw.split(",") if part.strip()}


def _parse_tp_targets(raw: str) -> tuple[float, ...]:
    values = tuple(sorted({float(part.strip()) for part in raw.split(",") if part.strip()}))
    if not values or any(value <= 0 for value in values):
        raise ValueError("TP targets must be positive decimal values.")
    return values


def build_exit_variants(*, baseline_hold_bars: int, tp_targets: tuple[float, ...], sections: set[str]) -> list[ExitVariant]:
    variants: list[ExitVariant] = []
    if "baseline" in sections:
        variants.extend(
            [
                ExitVariant(
                    name=f"fixed_{baseline_hold_bars}_next_open",
                    section="baseline",
                    exit_style=TREND_PULLBACK_EXIT_FIXED_BARS,
                    hold_bars=baseline_hold_bars,
                    take_profit_pct=0.0,
                    research_exit_fill=TREND_PULLBACK_RESEARCH_EXIT_FILL_NEXT_OPEN,
                ),
                ExitVariant(
                    name=f"fixed_{max(4, baseline_hold_bars - 5)}_next_open",
                    section="baseline",
                    exit_style=TREND_PULLBACK_EXIT_FIXED_BARS,
                    hold_bars=max(4, baseline_hold_bars - 5),
                    take_profit_pct=0.0,
                    research_exit_fill=TREND_PULLBACK_RESEARCH_EXIT_FILL_NEXT_OPEN,
                ),
                ExitVariant(
                    name=f"fixed_{baseline_hold_bars + 5}_next_open",
                    section="baseline",
                    exit_style=TREND_PULLBACK_EXIT_FIXED_BARS,
                    hold_bars=baseline_hold_bars + 5,
                    take_profit_pct=0.0,
                    research_exit_fill=TREND_PULLBACK_RESEARCH_EXIT_FILL_NEXT_OPEN,
                ),
                ExitVariant(
                    name=f"fixed_{baseline_hold_bars}_bar_close_control",
                    section="baseline",
                    exit_style=TREND_PULLBACK_EXIT_FIXED_BARS,
                    hold_bars=baseline_hold_bars,
                    take_profit_pct=0.0,
                    research_exit_fill=TREND_PULLBACK_RESEARCH_EXIT_FILL_BAR_CLOSE,
                ),
            ]
        )
    if "hybrid" in sections:
        variants.extend(
            [
                ExitVariant(
                    name=f"hybrid_tp_{target * 100:.2f}pct_hold{baseline_hold_bars}_bar_close",
                    section="hybrid",
                    exit_style=TREND_PULLBACK_EXIT_HYBRID_TP_OR_TIME,
                    hold_bars=baseline_hold_bars,
                    take_profit_pct=target,
                    research_exit_fill=TREND_PULLBACK_RESEARCH_EXIT_FILL_BAR_CLOSE,
                )
                for target in tp_targets
            ]
        )
    return variants


def _average_cost_drag_pct(cost_drag_df: pd.DataFrame) -> float:
    if cost_drag_df.empty:
        return 0.0
    return float(cost_drag_df["cost_drag_pct"].mean())


def summarize_exit_variant(
    *,
    variant: ExitVariant,
    backtest_result: dict[str, Any],
    signal_count: int,
    signal_paths_df: pd.DataFrame,
    slippage: float,
) -> dict[str, Any]:
    closed_trades_df = pair_round_trip_trades(
        backtest_result.get("trades", []),
        configured_hold_bars=variant.hold_bars,
    )
    if not closed_trades_df.empty:
        closed_trades_df["entry_ts"] = pd.to_datetime(closed_trades_df["entry_ts"], utc=True)
        closed_trades_df["slippage_per_share"] = slippage
    trade_paths_df = attach_trade_path_metrics(closed_trades_df, signal_paths_df)
    opportunity = summarize_opportunity_capture(trade_paths_df)
    cost_drag_df = build_cost_drag_table(closed_trades_df)

    return {
        "variant": variant.name,
        "section": variant.section,
        "exit_style": variant.exit_style,
        "research_exit_fill": variant.research_exit_fill,
        "hold_bars": variant.hold_bars,
        "take_profit_pct": variant.take_profit_pct * 100.0,
        "signal_count": int(signal_count),
        "trade_count": int(len(closed_trades_df)),
        "total_pnl": float(backtest_result.get("realized_pnl", 0.0)),
        "expectancy": float(backtest_result.get("expectancy", 0.0)),
        "win_rate": float(backtest_result.get("win_rate", 0.0)),
        "profit_factor": float(backtest_result.get("profit_factor", 0.0)),
        "max_drawdown_pct": float(backtest_result.get("max_drawdown_pct", 0.0)),
        "avg_realized_return_pct": float(trade_paths_df["realized_return_pct"].mean()) if not trade_paths_df.empty else 0.0,
        "avg_holding_bars": float(closed_trades_df["holding_bars"].mean()) if not closed_trades_df.empty else 0.0,
        "avg_best_available_return_pct": float(opportunity.get("avg_best_net_exit_pct", 0.0)),
        "avg_missed_opportunity_pct": float(opportunity.get("avg_missed_opportunity_pct", 0.0)),
        "materially_below_best_frac": float(opportunity.get("materially_worse_than_best_frac", 0.0)),
        "avg_cost_drag_pct": _average_cost_drag_pct(cost_drag_df),
    }


def classify_exit_capture(summary_df: pd.DataFrame) -> tuple[str, str]:
    if summary_df.empty:
        return "No variants produced trades.", "keep as research-only"

    baseline = summary_df[summary_df["variant"].str.contains("fixed_20_next_open|fixed_20_")]
    baseline = baseline[baseline["research_exit_fill"] == TREND_PULLBACK_RESEARCH_EXIT_FILL_NEXT_OPEN]
    baseline_row = baseline.iloc[0] if not baseline.empty else summary_df.iloc[0]

    ranked = summary_df.sort_values(["expectancy", "total_pnl", "variant"], ascending=[False, False, True]).reset_index(drop=True)
    best_row = ranked.iloc[0]

    expectancy_delta = float(best_row["expectancy"] - baseline_row["expectancy"])
    pnl_delta = float(best_row["total_pnl"] - baseline_row["total_pnl"])
    missed_delta = float(baseline_row["avg_missed_opportunity_pct"] - best_row["avg_missed_opportunity_pct"])
    below_best_delta = float(baseline_row["materially_below_best_frac"] - best_row["materially_below_best_frac"])

    if float(best_row["expectancy"]) <= 0 or float(best_row["total_pnl"]) <= 0:
        return (
            f"{best_row['variant']} was the least bad variant, but no exit style was convincingly positive on the clean SIP baseline.",
            "keep as research-only",
        )

    if best_row["section"] == "hybrid" and best_row["research_exit_fill"] == TREND_PULLBACK_RESEARCH_EXIT_FILL_BAR_CLOSE:
        if expectancy_delta >= 0.75 and pnl_delta >= 25.0 and missed_delta >= 0.30 and below_best_delta >= 10.0:
            return (
                f"{best_row['variant']} materially improved capture, but the gain depends on a research-only same-bar target-touch assumption.",
                "continue with deeper out-of-sample validation",
            )
        return (
            f"{best_row['variant']} improved capture somewhat, but most of the gain still looks too dependent on optimistic research-only fill assumptions.",
            "keep as research-only",
        )

    if expectancy_delta >= 0.75 and pnl_delta >= 25.0 and missed_delta >= 0.30:
        return (
            f"{best_row['variant']} materially improved realized capture without relying on large parameter changes.",
            "continue with deeper out-of-sample validation",
        )

    return (
        f"{best_row['variant']} performed best, but the improvement over the clean-SIP baseline was modest.",
        "keep as research-only",
    )


def _save_outputs(
    *,
    output_dir: Path,
    summary_df: pd.DataFrame,
    diagnosis: tuple[str, str],
    variants: list[ExitVariant],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(output_dir / "trend_pullback_exit_capture_clean.csv", index=False)
    (output_dir / "trend_pullback_exit_capture_clean.json").write_text(
        json.dumps(
            {
                "variants": [asdict(variant) for variant in variants],
                "summary": summary_df.to_dict("records"),
                "diagnosis": diagnosis[0],
                "recommendation": diagnosis[1],
            },
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    payload = _load_runtime_payload(Path(args.config))
    runtime = dict(payload["runtime"])
    runtime["strategy_mode"] = STRATEGY_MODE_TREND_PULLBACK
    runtime["symbols"] = list(_resolve_symbols(runtime, args.symbols))
    runtime["trend_pullback_hold_bars"] = int(args.baseline_hold_bars)
    sections = _parse_sections(args.sections)
    tp_targets = _parse_tp_targets(args.tp_targets)

    context = build_audit_context(
        config_path=Path(args.config),
        runtime=runtime,
        source_dataset=payload.get("source", {}).get("dataset"),
        dataset_override=args.dataset,
        symbols_override=list(runtime["symbols"]),
        start_date=None,
        end_date=None,
        commission_per_order=args.commission_per_order,
        slippage_per_share=args.slippage_per_share,
        position_size=args.position_size,
        output_dir=args.output_dir,
        variant="explicit_config",
    )
    inputs, state = _prepare_inputs_and_state(context)
    _, signals_df = evaluate_raw_entry_signals(
        inputs=inputs,
        state=state,
        sma_bars=context.backtest_kwargs["sma_bars"],
        time_window_mode=context.backtest_kwargs["time_window_mode"],
        slippage=context.backtest_kwargs["slippage"],
        commission=context.backtest_kwargs["commission"],
        position_size=context.backtest_kwargs["position_size"],
        horizons=(15, 20, 25),
    )
    signal_paths_df = build_signal_path_table(
        signals_df,
        state=state,
        slippage=context.backtest_kwargs["slippage"],
        commission=context.backtest_kwargs["commission"],
        position_size=context.backtest_kwargs["position_size"],
        path_horizon=max(25, int(args.baseline_hold_bars)),
        horizons=(15, 20, 25),
    )

    variants = build_exit_variants(
        baseline_hold_bars=int(args.baseline_hold_bars),
        tp_targets=tp_targets,
        sections=sections,
    )

    rows: list[dict[str, Any]] = []
    for variant in variants:
        kwargs = dict(context.backtest_kwargs)
        kwargs.update(
            {
                "trend_pullback_exit_style": variant.exit_style,
                "trend_pullback_hold_bars": variant.hold_bars,
                "trend_pullback_take_profit_pct": variant.take_profit_pct,
                "trend_pullback_research_exit_fill": variant.research_exit_fill,
            }
        )
        result = run_backtest(context.dataset_path, **kwargs)
        rows.append(
            summarize_exit_variant(
                variant=variant,
                backtest_result=result,
                signal_count=len(signals_df),
                signal_paths_df=signal_paths_df,
                slippage=float(context.backtest_kwargs["slippage"]),
            )
        )

    summary_df = pd.DataFrame(rows).sort_values(["expectancy", "total_pnl", "variant"], ascending=[False, False, True])
    diagnosis = classify_exit_capture(summary_df)

    print("\n=== Trend Pullback Exit Capture (Clean SIP) ===")
    print(f"Dataset: {context.dataset_path}")
    print(f"Symbols: {', '.join(runtime['symbols'])}")
    print(f"Baseline hold: {args.baseline_hold_bars} bars")
    print("\nExit comparison:")
    print(_frame_text(summary_df, max_rows=20))
    print("\nDiagnosis:")
    print(_frame_text(pd.DataFrame([{"diagnosis": diagnosis[0], "recommendation": diagnosis[1]}]), max_rows=5))

    if args.output_dir:
        _save_outputs(output_dir=Path(args.output_dir), summary_df=summary_df, diagnosis=diagnosis, variants=variants)
    best_row = summary_df.iloc[0].to_dict() if not summary_df.empty else {}
    log_experiment_run(
        run_type="exit_capture_clean",
        script_path=__file__,
        strategy_name=STRATEGY_MODE_TREND_PULLBACK,
        dataset_path=context.dataset_path,
        symbols=runtime["symbols"],
        params={
            "config": args.config,
            "baseline_hold_bars": args.baseline_hold_bars,
            "tp_targets": list(tp_targets),
            "sections": sorted(sections),
            "commission_per_order": args.commission_per_order,
            "slippage_per_share": args.slippage_per_share,
            "position_size": args.position_size,
        },
        metrics={
            "trade_count": best_row.get("trade_count"),
            "expectancy": best_row.get("expectancy"),
            "realized_pnl": best_row.get("total_pnl"),
            "win_rate": best_row.get("win_rate"),
            "profit_factor": best_row.get("profit_factor"),
            "max_drawdown_pct": best_row.get("max_drawdown_pct"),
        },
        output_path=args.output_dir,
        summary_path=(Path(args.output_dir) / "trend_pullback_exit_capture_clean.json") if args.output_dir else None,
        extra_fields={
            "diagnosis": diagnosis[0],
            "recommendation": diagnosis[1],
            "best_variant": best_row.get("variant"),
        },
    )


if __name__ == "__main__":
    main()

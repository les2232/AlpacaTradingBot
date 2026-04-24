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
from run_edge_audit import (
    _load_runtime_payload,
    _prepare_inputs_and_state,
    build_audit_context,
    evaluate_raw_entry_signals,
)
from run_edge_diagnostics import _frame_text
from run_trade_path_diagnostics import (
    attach_trade_path_metrics,
    build_cost_drag_table,
    build_signal_path_table,
    pair_round_trip_trades,
)
from strategy import (
    STRATEGY_MODE_TREND_PULLBACK,
    TREND_PULLBACK_EXIT_FIXED_BARS,
    TREND_PULLBACK_EXIT_HYBRID_TP_OR_TIME,
    TREND_PULLBACK_EXIT_TAKE_PROFIT,
)


DEFAULT_CONFIG_PATH = Path("config") / "trend_pullback.example.json"
DEFAULT_SYMBOLS = ("AMD", "HON", "C", "JPM")


@dataclass(frozen=True)
class ExitVariant:
    name: str
    exit_style: str
    hold_bars: int
    take_profit_pct: float
    research_exit_fill: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare a small set of explicit trend_pullback exit variants."
    )
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH), help="Runtime config JSON.")
    parser.add_argument("--dataset", help="Dataset directory. Defaults to config source.dataset.")
    parser.add_argument("--symbols", nargs="*", help="Optional symbol override.")
    parser.add_argument("--start-date", help="Optional UTC date filter (YYYY-MM-DD).")
    parser.add_argument("--end-date", help="Optional UTC date filter (YYYY-MM-DD).")
    parser.add_argument("--commission-per-order", type=float, default=0.01)
    parser.add_argument("--slippage-per-share", type=float, default=0.05)
    parser.add_argument("--position-size", type=float, default=1000.0)
    parser.add_argument("--take-profit-pct", type=float, default=0.0025)
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


def build_exit_variants(take_profit_pct: float) -> list[ExitVariant]:
    return [
        ExitVariant(
            name="fixed_3_next_open",
            exit_style=TREND_PULLBACK_EXIT_FIXED_BARS,
            hold_bars=3,
            take_profit_pct=0.0,
            research_exit_fill=TREND_PULLBACK_RESEARCH_EXIT_FILL_NEXT_OPEN,
        ),
        ExitVariant(
            name="fixed_4_next_open",
            exit_style=TREND_PULLBACK_EXIT_FIXED_BARS,
            hold_bars=4,
            take_profit_pct=0.0,
            research_exit_fill=TREND_PULLBACK_RESEARCH_EXIT_FILL_NEXT_OPEN,
        ),
        ExitVariant(
            name="fixed_3_bar_close",
            exit_style=TREND_PULLBACK_EXIT_FIXED_BARS,
            hold_bars=3,
            take_profit_pct=0.0,
            research_exit_fill=TREND_PULLBACK_RESEARCH_EXIT_FILL_BAR_CLOSE,
        ),
        ExitVariant(
            name="fixed_4_bar_close",
            exit_style=TREND_PULLBACK_EXIT_FIXED_BARS,
            hold_bars=4,
            take_profit_pct=0.0,
            research_exit_fill=TREND_PULLBACK_RESEARCH_EXIT_FILL_BAR_CLOSE,
        ),
        ExitVariant(
            name=f"take_profit_{take_profit_pct * 100:.2f}pct",
            exit_style=TREND_PULLBACK_EXIT_TAKE_PROFIT,
            hold_bars=0,
            take_profit_pct=take_profit_pct,
            research_exit_fill=TREND_PULLBACK_RESEARCH_EXIT_FILL_BAR_CLOSE,
        ),
        ExitVariant(
            name=f"hybrid_tp_or_time_{take_profit_pct * 100:.2f}pct_hold3",
            exit_style=TREND_PULLBACK_EXIT_HYBRID_TP_OR_TIME,
            hold_bars=3,
            take_profit_pct=take_profit_pct,
            research_exit_fill=TREND_PULLBACK_RESEARCH_EXIT_FILL_BAR_CLOSE,
        ),
    ]


def _summarize_cost_drag(cost_drag_df: pd.DataFrame) -> float:
    if cost_drag_df.empty:
        return 0.0
    return float(cost_drag_df["cost_drag_pct"].mean())


def summarize_exit_variant(
    *,
    variant: ExitVariant,
    backtest_result: dict[str, Any],
    signal_paths_df: pd.DataFrame,
    slippage: float,
) -> dict[str, Any]:
    closed_trades_df = pair_round_trip_trades(
        backtest_result.get("trades", []),
        configured_hold_bars=variant.hold_bars,
    )
    if not closed_trades_df.empty:
        closed_trades_df["slippage_per_share"] = slippage
    trade_paths_df = attach_trade_path_metrics(closed_trades_df, signal_paths_df)
    cost_drag_df = build_cost_drag_table(closed_trades_df)

    gave_back = (
        trade_paths_df["best_net_exit_pct"] - trade_paths_df["realized_return_pct"]
        if not trade_paths_df.empty and {"best_net_exit_pct", "realized_return_pct"} <= set(trade_paths_df.columns)
        else pd.Series(dtype=float)
    )
    capture_base = trade_paths_df[
        ["realized_return_pct", "best_net_exit_pct"]
    ].dropna() if not trade_paths_df.empty else pd.DataFrame()
    if not capture_base.empty:
        capture_base = capture_base[capture_base["best_net_exit_pct"] > 0]
    if not capture_base.empty:
        capture_ratio = (
            capture_base["realized_return_pct"].clip(lower=0.0) / capture_base["best_net_exit_pct"]
        ) * 100.0
        capture_ratio = capture_ratio.clip(lower=0.0, upper=100.0)
    else:
        capture_ratio = pd.Series(dtype=float)

    return {
        "variant": variant.name,
        "exit_style": variant.exit_style,
        "research_exit_fill": variant.research_exit_fill,
        "hold_bars": variant.hold_bars,
        "take_profit_pct": variant.take_profit_pct * 100.0,
        "trade_count": int(len(closed_trades_df)),
        "total_pnl": float(backtest_result.get("realized_pnl", 0.0)),
        "expectancy": float(backtest_result.get("expectancy", 0.0)),
        "win_rate": float(backtest_result.get("win_rate", 0.0)),
        "profit_factor": float(backtest_result.get("profit_factor", 0.0)),
        "max_drawdown_pct": float(backtest_result.get("max_drawdown_pct", 0.0)),
        "avg_realized_return_pct": float(trade_paths_df["realized_return_pct"].mean()) if not trade_paths_df.empty else 0.0,
        "avg_holding_bars": float(closed_trades_df["holding_bars"].mean()) if not closed_trades_df.empty else 0.0,
        "avg_best_net_exit_pct": float(trade_paths_df["best_net_exit_pct"].mean()) if not trade_paths_df.empty else 0.0,
        "avg_mfe_capture_pct": float(capture_ratio.mean()) if not capture_ratio.empty else 0.0,
        "avg_gave_back_pct": float(gave_back.mean()) if not gave_back.empty else 0.0,
        "avg_cost_drag_pct": _summarize_cost_drag(cost_drag_df),
    }


def diagnose_exit_results(summary_df: pd.DataFrame) -> tuple[str, str]:
    if summary_df.empty:
        return "No exit variants produced trades.", "keep as research-only because edge is still too small"

    ranked = summary_df.sort_values(["total_pnl", "expectancy", "variant"], ascending=[False, False, True])
    best = ranked.iloc[0]
    fixed3_next_open = summary_df[summary_df["variant"] == "fixed_3_next_open"]
    fixed3_bar_close = summary_df[summary_df["variant"] == "fixed_3_bar_close"]
    leakage = 0.0
    if not fixed3_next_open.empty and not fixed3_bar_close.empty:
        leakage = float(fixed3_bar_close.iloc[0]["total_pnl"] - fixed3_next_open.iloc[0]["total_pnl"])

    if float(best["total_pnl"]) <= 0 or float(best["expectancy"]) <= 0:
        return (
            f"{best['variant']} was the least bad, but no variant was convincingly positive after costs.",
            "keep as research-only because edge is still too small",
        )
    if leakage > 20.0:
        return (
            f"{best['variant']} monetized the move best, and bar-close research fills improved fixed-horizon capture materially versus next-open execution.",
            "keep researching with improved exits",
        )
    return (
        f"{best['variant']} performed best, but the improvement was modest and still needs caution around costs and sample stability.",
        "keep as research-only because edge is still too small",
    )


def main() -> None:
    args = parse_args()
    runtime_payload = _load_runtime_payload(Path(args.config))
    runtime = dict(runtime_payload["runtime"])
    runtime["strategy_mode"] = STRATEGY_MODE_TREND_PULLBACK
    runtime["symbols"] = list(_resolve_symbols(runtime, args.symbols))
    if args.start_date:
        runtime["start_date"] = args.start_date
    if args.end_date:
        runtime["end_date"] = args.end_date

    context = build_audit_context(
        config_path=Path(args.config),
        runtime=runtime,
        source_dataset=runtime_payload.get("source", {}).get("dataset"),
        dataset_override=args.dataset,
        symbols_override=list(runtime["symbols"]),
        start_date=args.start_date,
        end_date=args.end_date,
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
        horizons=(1, 2, 3, 4, 5),
    )
    signal_paths_df = build_signal_path_table(
        signals_df,
        state=state,
        slippage=context.backtest_kwargs["slippage"],
        commission=context.backtest_kwargs["commission"],
        position_size=context.backtest_kwargs["position_size"],
        path_horizon=5,
        horizons=(1, 2, 3, 4, 5),
    )

    variants = build_exit_variants(args.take_profit_pct)
    summaries: list[dict[str, Any]] = []
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
        summaries.append(
            summarize_exit_variant(
                variant=variant,
                backtest_result=result,
                signal_paths_df=signal_paths_df,
                slippage=float(context.backtest_kwargs["slippage"]),
            )
        )

    summary_df = pd.DataFrame(summaries).sort_values(
        ["total_pnl", "expectancy", "variant"],
        ascending=[False, False, True],
    )
    diagnosis, recommendation = diagnose_exit_results(summary_df)
    diagnosis_df = pd.DataFrame([{"diagnosis": diagnosis, "recommendation": recommendation}])

    print("\nTrend Pullback Exit Comparison")
    print(_frame_text(summary_df, max_rows=20))
    print("\nExit Diagnosis")
    print(_frame_text(diagnosis_df, max_rows=5))

    if args.output_dir:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        summary_df.to_csv(output_dir / "trend_pullback_exit_comparison.csv", index=False)
        (output_dir / "trend_pullback_exit_comparison.json").write_text(
            json.dumps(
                {
                    "variants": [asdict(variant) for variant in variants],
                    "summary": summary_df.to_dict("records"),
                    "diagnosis": diagnosis,
                    "recommendation": recommendation,
                },
                indent=2,
                default=str,
            ),
            encoding="utf-8",
        )
    best_row = summary_df.iloc[0].to_dict() if not summary_df.empty else {}
    log_experiment_run(
        run_type="exit_comparison",
        script_path=__file__,
        strategy_name=STRATEGY_MODE_TREND_PULLBACK,
        dataset_path=context.dataset_path,
        symbols=runtime["symbols"],
        params={
            "config": args.config,
            "start_date": args.start_date,
            "end_date": args.end_date,
            "take_profit_pct": args.take_profit_pct,
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
        summary_path=(Path(args.output_dir) / "trend_pullback_exit_comparison.json") if args.output_dir else None,
        extra_fields={
            "diagnosis": diagnosis,
            "recommendation": recommendation,
            "best_variant": best_row.get("variant"),
        },
    )


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from backtest_runner import run_backtest
from research.experiment_log import log_experiment_run
from run_edge_audit import (
    _load_runtime_payload,
    _prepare_inputs_and_state,
    build_audit_context,
    compute_forward_return_pcts,
    evaluate_raw_entry_signals,
)
from run_edge_diagnostics import _frame_text
from strategy import STRATEGY_MODE_TREND_PULLBACK


DEFAULT_CONFIG_PATH = Path("config") / "trend_pullback.example.json"
DEFAULT_SYMBOLS = ("AMD", "HON", "C", "JPM")
DEFAULT_PATH_HORIZON = 5


@dataclass(frozen=True)
class HoldRun:
    hold_bars: int
    result: dict[str, Any]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run trade-level edge decomposition diagnostics for trend_pullback."
    )
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH), help="Runtime config JSON.")
    parser.add_argument("--dataset", help="Dataset directory. Defaults to config source.dataset.")
    parser.add_argument("--symbols", nargs="*", help="Optional symbol override.")
    parser.add_argument("--start-date", help="Optional UTC date filter (YYYY-MM-DD).")
    parser.add_argument("--end-date", help="Optional UTC date filter (YYYY-MM-DD).")
    parser.add_argument("--commission-per-order", type=float, default=0.01)
    parser.add_argument("--slippage-per-share", type=float, default=0.05)
    parser.add_argument("--position-size", type=float, default=1000.0)
    parser.add_argument("--compare-holds", default="3,4")
    parser.add_argument("--path-horizon", type=int, default=DEFAULT_PATH_HORIZON)
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


def _parse_compare_holds(raw_holds: str) -> tuple[int, ...]:
    values = []
    for chunk in raw_holds.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        value = int(chunk)
        if value <= 0:
            raise ValueError("compare holds must be positive integers")
        values.append(value)
    unique = tuple(sorted(set(values)))
    if len(unique) < 2:
        raise ValueError("compare holds must include at least two horizons")
    return unique


def pair_round_trip_trades(
    trades: list[dict[str, Any]],
    *,
    configured_hold_bars: int,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    open_entries: dict[str, list[dict[str, Any]]] = {}
    for trade in trades:
        symbol = str(trade.get("symbol", "") or "")
        if not symbol:
            continue
        side = str(trade.get("side", "") or "").upper()
        if side == "BUY":
            open_entries.setdefault(symbol, []).append(trade)
            continue
        if side != "SELL":
            continue
        queue = open_entries.setdefault(symbol, [])
        if not queue:
            continue
        entry_trade = queue.pop(0)
        entry_price = float(entry_trade.get("price", 0.0) or 0.0)
        shares = float(trade.get("shares", entry_trade.get("shares", 0.0)) or 0.0)
        notional = shares * entry_price if shares > 0 and entry_price > 0 else 0.0
        exit_reason = trade.get("exit_reason")
        if not exit_reason:
            if trade.get("forced_close"):
                exit_reason = "forced_close"
            elif trade.get("eod_exit"):
                exit_reason = "eod_exit"
            elif configured_hold_bars > 0 and int(trade.get("holding_bars") or 0) >= configured_hold_bars:
                exit_reason = "trend_pullback_fixed_bars_exit"
            else:
                exit_reason = "unknown"
        pnl = float(trade.get("pnl", 0.0) or 0.0)
        rows.append(
            {
                "symbol": symbol,
                "entry_ts": pd.Timestamp(entry_trade["timestamp"]),
                "exit_ts": pd.Timestamp(trade["timestamp"]),
                "entry_price": entry_price,
                "exit_price": float(trade.get("price", 0.0) or 0.0),
                "shares": shares,
                "position_notional": notional,
                "holding_bars": int(trade.get("holding_bars") or 0),
                "realized_pnl": pnl,
                "realized_return_pct": (pnl / notional * 100.0) if notional > 0 else 0.0,
                "entry_branch": entry_trade.get("entry_branch"),
                "exit_reason": exit_reason,
                "forced_close": bool(trade.get("forced_close", False)),
                "eod_exit": bool(trade.get("eod_exit", False)),
            }
        )
    return pd.DataFrame(rows)


def _path_lookup_for_symbol(state: Any, symbol: str) -> dict[pd.Timestamp, int]:
    return {
        pd.Timestamp(ts): idx
        for idx, ts in enumerate(state.symbols_dfs[symbol]["timestamp"].tolist())
    }


def compute_entry_path_metrics(
    *,
    symbol: str,
    entry_ts: pd.Timestamp,
    entry_fill: float,
    state: Any,
    slippage: float,
    commission: float,
    position_size: float,
    path_horizon: int,
    horizons: tuple[int, ...],
    lookup: dict[pd.Timestamp, int],
) -> dict[str, Any]:
    index = lookup.get(pd.Timestamp(entry_ts))
    if index is None:
        return {
            "path_bar_count": 0,
            "signal_close": None,
            "signal_to_entry_gap_pct": None,
            "mfe_pct": 0.0,
            "mae_pct": 0.0,
            "peak_favorable_bar": 0,
            "worst_adverse_bar": 0,
            "best_net_exit_pct": 0.0,
            "best_exit_bar": 0,
            "worst_net_exit_pct": 0.0,
            "worst_exit_bar": 0,
            "drawdown_before_best_pct": 0.0,
        }

    symbol_df = state.symbols_dfs[symbol]
    closes = symbol_df["close"].tolist()
    highs = symbol_df["high"].tolist()
    lows = symbol_df["low"].tolist()
    signal_close = float(closes[index - 1]) if index > 0 else None
    path_end = min(len(symbol_df), index + max(path_horizon, max(horizons, default=0)))
    best_high = entry_fill
    worst_low = entry_fill
    best_bar = 0
    worst_bar = 0
    best_net_exit = None
    best_exit_bar = 0
    worst_net_exit = None
    worst_exit_bar = 0
    drawdown_before_best = 0.0
    metrics: dict[str, Any] = {
        "path_bar_count": max(0, path_end - index),
        "signal_close": signal_close,
        "signal_to_entry_gap_pct": ((entry_fill - signal_close) / signal_close * 100.0) if signal_close else None,
    }
    net_path_values: list[tuple[int, float]] = []

    for offset, bar_index in enumerate(range(index, path_end), start=1):
        close_price = float(closes[bar_index])
        high_price = float(highs[bar_index])
        low_price = float(lows[bar_index])
        if high_price > best_high:
            best_high = high_price
            best_bar = offset
        if low_price < worst_low:
            worst_low = low_price
            worst_bar = offset
        gross_close_return_pct = ((close_price - entry_fill) / entry_fill * 100.0) if entry_fill > 0 else 0.0
        _, net_exit_return_pct = compute_forward_return_pcts(
            entry_open=entry_fill,
            entry_fill=entry_fill,
            future_close=close_price,
            slippage=slippage,
            commission=commission,
            position_size=position_size,
        )
        net_path_values.append((offset, net_exit_return_pct))
        if best_net_exit is None or net_exit_return_pct > best_net_exit:
            best_net_exit = net_exit_return_pct
            best_exit_bar = offset
            drawdown_before_best = min((value for bar, value in net_path_values if bar <= offset), default=0.0)
        if worst_net_exit is None or net_exit_return_pct < worst_net_exit:
            worst_net_exit = net_exit_return_pct
            worst_exit_bar = offset
        if offset in horizons:
            metrics[f"gross_close_{offset}b_return_pct"] = gross_close_return_pct
            metrics[f"net_exit_{offset}b_return_pct"] = net_exit_return_pct

    for horizon in horizons:
        metrics.setdefault(f"gross_close_{horizon}b_return_pct", None)
        metrics.setdefault(f"net_exit_{horizon}b_return_pct", None)

    metrics.update(
        {
            "mfe_pct": ((best_high - entry_fill) / entry_fill * 100.0) if entry_fill > 0 else 0.0,
            "mae_pct": ((worst_low - entry_fill) / entry_fill * 100.0) if entry_fill > 0 else 0.0,
            "peak_favorable_bar": best_bar,
            "worst_adverse_bar": worst_bar,
            "best_net_exit_pct": float(best_net_exit) if best_net_exit is not None else 0.0,
            "best_exit_bar": best_exit_bar,
            "worst_net_exit_pct": float(worst_net_exit) if worst_net_exit is not None else 0.0,
            "worst_exit_bar": worst_exit_bar,
            "drawdown_before_best_pct": float(drawdown_before_best),
        }
    )
    return metrics


def build_signal_path_table(
    signals_df: pd.DataFrame,
    *,
    state: Any,
    slippage: float,
    commission: float,
    position_size: float,
    path_horizon: int,
    horizons: tuple[int, ...],
) -> pd.DataFrame:
    if signals_df.empty:
        return pd.DataFrame()
    lookups = {symbol: _path_lookup_for_symbol(state, symbol) for symbol in signals_df["symbol"].dropna().unique()}
    rows = []
    for record in signals_df.to_dict("records"):
        metrics = compute_entry_path_metrics(
            symbol=str(record["symbol"]),
            entry_ts=pd.Timestamp(record["entry_ts"]),
            entry_fill=float(record["entry_fill"]),
            state=state,
            slippage=slippage,
            commission=commission,
            position_size=position_size,
            path_horizon=path_horizon,
            horizons=horizons,
            lookup=lookups[str(record["symbol"])],
        )
        rows.append({**record, **metrics})
    return pd.DataFrame(rows)


def attach_trade_path_metrics(
    closed_trades_df: pd.DataFrame,
    signal_paths_df: pd.DataFrame,
) -> pd.DataFrame:
    if closed_trades_df.empty:
        return closed_trades_df.copy()
    join_cols = ["symbol", "entry_ts"]
    signal_subset = signal_paths_df.drop_duplicates(join_cols)
    return closed_trades_df.merge(signal_subset, on=join_cols, how="left", suffixes=("", "_signal"))


def build_cost_drag_table(closed_trades_df: pd.DataFrame) -> pd.DataFrame:
    if closed_trades_df.empty:
        return pd.DataFrame(
            columns=[
                "symbol",
                "entry_ts",
                "realized_return_pct",
                "gross_realized_return_pct",
                "cost_drag_pct",
            ]
        )
    rows = []
    for record in closed_trades_df.to_dict("records"):
        entry_price = float(record.get("entry_price", 0.0) or 0.0)
        exit_price = float(record.get("exit_price", 0.0) or 0.0)
        shares = float(record.get("shares", 0.0) or 0.0)
        notional = float(record.get("position_notional", 0.0) or 0.0)
        gross_entry = max(entry_price - record.get("slippage_per_share", 0.0), 0.0)
        gross_exit = exit_price + record.get("slippage_per_share", 0.0)
        gross_pnl = shares * (gross_exit - gross_entry)
        gross_return_pct = (gross_pnl / notional * 100.0) if notional > 0 else 0.0
        realized_return_pct = float(record.get("realized_return_pct", 0.0) or 0.0)
        rows.append(
            {
                "symbol": record["symbol"],
                "entry_ts": record["entry_ts"],
                "realized_return_pct": realized_return_pct,
                "gross_realized_return_pct": gross_return_pct,
                "cost_drag_pct": gross_return_pct - realized_return_pct,
            }
        )
    return pd.DataFrame(rows)


def summarize_hold_delta(signal_paths_df: pd.DataFrame, *, hold_a: int, hold_b: int) -> dict[str, Any]:
    col_a = f"net_exit_{hold_a}b_return_pct"
    col_b = f"net_exit_{hold_b}b_return_pct"
    empty = {
        "sample_count": 0,
        f"avg_{hold_a}b_net_return_pct": 0.0,
        f"avg_{hold_b}b_net_return_pct": 0.0,
        f"avg_delta_{hold_b}m{hold_a}_pct": 0.0,
        f"profitable_{hold_a}b_not_{hold_b}b": 0,
        f"profitable_{hold_b}b_not_{hold_a}b": 0,
        f"giveback_between_{hold_a}b_{hold_b}b_frac": 0.0,
        "peak_before_later_exit_frac": 0.0,
    }
    if signal_paths_df.empty or col_a not in signal_paths_df.columns or col_b not in signal_paths_df.columns:
        return empty
    base = signal_paths_df[[col_a, col_b, "best_exit_bar"]].dropna()
    if base.empty:
        return empty
    delta = base[col_b] - base[col_a]
    return {
        "sample_count": int(len(base)),
        f"avg_{hold_a}b_net_return_pct": float(base[col_a].mean()),
        f"avg_{hold_b}b_net_return_pct": float(base[col_b].mean()),
        f"avg_delta_{hold_b}m{hold_a}_pct": float(delta.mean()),
        f"profitable_{hold_a}b_not_{hold_b}b": int(((base[col_a] > 0) & (base[col_b] <= 0)).sum()),
        f"profitable_{hold_b}b_not_{hold_a}b": int(((base[col_b] > 0) & (base[col_a] <= 0)).sum()),
        f"giveback_between_{hold_a}b_{hold_b}b_frac": float((delta < 0).mean() * 100.0),
        "peak_before_later_exit_frac": float((base["best_exit_bar"] > 0).lt(hold_b).mean() * 100.0),
    }


def summarize_opportunity_capture(trade_paths_df: pd.DataFrame) -> dict[str, Any]:
    empty = {
        "trade_count": 0,
        "avg_realized_return_pct": 0.0,
        "avg_best_net_exit_pct": 0.0,
        "avg_worst_net_exit_pct": 0.0,
        "avg_missed_opportunity_pct": 0.0,
        "avg_drawdown_before_best_pct": 0.0,
        "materially_worse_than_best_frac": 0.0,
    }
    if trade_paths_df.empty:
        return empty
    base = trade_paths_df[
        [
            "realized_return_pct",
            "best_net_exit_pct",
            "worst_net_exit_pct",
            "drawdown_before_best_pct",
        ]
    ].dropna()
    if base.empty:
        return empty
    missed = base["best_net_exit_pct"] - base["realized_return_pct"]
    return {
        "trade_count": int(len(base)),
        "avg_realized_return_pct": float(base["realized_return_pct"].mean()),
        "avg_best_net_exit_pct": float(base["best_net_exit_pct"].mean()),
        "avg_worst_net_exit_pct": float(base["worst_net_exit_pct"].mean()),
        "avg_missed_opportunity_pct": float(missed.mean()),
        "avg_drawdown_before_best_pct": float(base["drawdown_before_best_pct"].mean()),
        "materially_worse_than_best_frac": float((missed > 0.10).mean() * 100.0),
    }


def bucket_trade_shape(record: dict[str, Any]) -> str:
    net_1b = float(record.get("net_exit_1b_return_pct") or 0.0)
    best_net = float(record.get("best_net_exit_pct") or 0.0)
    realized = float(record.get("realized_return_pct") or 0.0)
    mfe = float(record.get("mfe_pct") or 0.0)
    best_bar = int(record.get("best_exit_bar") or 0)
    worst_bar = int(record.get("worst_adverse_bar") or 0)
    mae = float(record.get("mae_pct") or 0.0)
    if best_net <= 0 and mfe <= 0:
        return "never_worked"
    if net_1b > 0 and realized > 0 and best_bar <= 2:
        return "immediate_follow_through_winner"
    if best_net > 0 and realized <= 0:
        return "gave_back_winner"
    if net_1b <= 0 and best_net > 0 and realized > 0:
        return "delayed_winner"
    if net_1b < 0 and mae < 0 and worst_bar <= 2 and realized < 0:
        return "quick_reversal"
    return "chop_then_drift"


def build_trade_shape_summary(trade_paths_df: pd.DataFrame) -> pd.DataFrame:
    columns = ["shape", "trade_count", "avg_realized_return_pct", "avg_best_net_exit_pct", "avg_mfe_pct", "avg_mae_pct"]
    if trade_paths_df.empty:
        return pd.DataFrame(columns=columns)
    enriched = trade_paths_df.copy()
    enriched["shape"] = [bucket_trade_shape(record) for record in enriched.to_dict("records")]
    rows = []
    for shape, shape_df in enriched.groupby("shape", sort=True):
        rows.append(
            {
                "shape": shape,
                "trade_count": int(len(shape_df)),
                "avg_realized_return_pct": float(shape_df["realized_return_pct"].mean()),
                "avg_best_net_exit_pct": float(shape_df["best_net_exit_pct"].mean()),
                "avg_mfe_pct": float(shape_df["mfe_pct"].mean()),
                "avg_mae_pct": float(shape_df["mae_pct"].mean()),
            }
        )
    return pd.DataFrame(rows).sort_values(["trade_count", "shape"], ascending=[False, True])


def build_realized_hold_summary(runs: list[HoldRun]) -> pd.DataFrame:
    rows = []
    for run in runs:
        result = run.result
        rows.append(
            {
                "hold_bars": run.hold_bars,
                "total_trades": int(result.get("total_trades", 0)),
                "realized_pnl": float(result.get("realized_pnl", 0.0)),
                "expectancy": float(result.get("expectancy", 0.0)),
                "win_rate": float(result.get("win_rate", 0.0)),
                "profit_factor": float(result.get("profit_factor", 0.0)),
                "max_drawdown_pct": float(result.get("max_drawdown_pct", 0.0)),
            }
        )
    return pd.DataFrame(rows).sort_values("hold_bars")


def diagnose_trade_path(
    *,
    trade_paths_df: pd.DataFrame,
    hold_delta_summary: dict[str, Any],
    realized_hold_df: pd.DataFrame,
    cost_drag_summary: dict[str, Any],
) -> str:
    if trade_paths_df.empty:
        return "No closed trades were available, so this remains research-only."

    avg_best = float(trade_paths_df["best_net_exit_pct"].mean()) if "best_net_exit_pct" in trade_paths_df else 0.0
    avg_realized = float(trade_paths_df["realized_return_pct"].mean()) if "realized_return_pct" in trade_paths_df else 0.0
    avg_missed = avg_best - avg_realized
    avg_cost_drag = float(cost_drag_summary.get("avg_cost_drag_pct", 0.0))
    giveback_frac = float(hold_delta_summary.get("giveback_between_3b_4b_frac", 0.0))
    hold3 = realized_hold_df.loc[realized_hold_df["hold_bars"] == 3, "realized_pnl"]
    hold4 = realized_hold_df.loc[realized_hold_df["hold_bars"] == 4, "realized_pnl"]

    if not hold3.empty and not hold4.empty and float(hold3.iloc[0]) > float(hold4.iloc[0]) and giveback_frac >= 50.0:
        if avg_cost_drag > 0 and avg_best > 0 and avg_realized <= 0:
            return (
                "Mixed case: there is some short-horizon edge, but it decays before bar 4 and the remaining "
                "edge is small enough that friction finishes the job."
            )
        return (
            "The best fit is an exit-timing problem: the path tends to peak before bar 4, and the strategy gives "
            "back too much between bars 3 and 4."
        )
    if avg_cost_drag >= max(avg_best, 0.05):
        return "The signal looks too small to survive current fill and cost assumptions."
    if avg_best <= 0:
        return "There is no meaningful edge even before execution; most trades never develop usable upside."
    if avg_missed > 0.15:
        return "The strategy is producing some favorable excursion, but it is not capturing it reliably."
    return "The current sample looks mixed and fragile; keep it research-only until entry/exit timing is clarified."


def _summarize_cost_drag(cost_drag_df: pd.DataFrame) -> dict[str, Any]:
    if cost_drag_df.empty:
        return {
            "trade_count": 0,
            "avg_realized_return_pct": 0.0,
            "avg_gross_realized_return_pct": 0.0,
            "avg_cost_drag_pct": 0.0,
        }
    return {
        "trade_count": int(len(cost_drag_df)),
        "avg_realized_return_pct": float(cost_drag_df["realized_return_pct"].mean()),
        "avg_gross_realized_return_pct": float(cost_drag_df["gross_realized_return_pct"].mean()),
        "avg_cost_drag_pct": float(cost_drag_df["cost_drag_pct"].mean()),
    }


def _to_single_row_df(values: dict[str, Any]) -> pd.DataFrame:
    return pd.DataFrame([values])


def main() -> None:
    args = parse_args()
    compare_holds = _parse_compare_holds(args.compare_holds)
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
    path_horizon = max(args.path_horizon, max(compare_holds))
    horizons = tuple(sorted(set((1, 2, 3, 4, *compare_holds, path_horizon))))
    _, signals_df = evaluate_raw_entry_signals(
        inputs=inputs,
        state=state,
        sma_bars=context.backtest_kwargs["sma_bars"],
        time_window_mode=context.backtest_kwargs["time_window_mode"],
        slippage=context.backtest_kwargs["slippage"],
        commission=context.backtest_kwargs["commission"],
        position_size=context.backtest_kwargs["position_size"],
        horizons=horizons,
    )
    signal_paths_df = build_signal_path_table(
        signals_df,
        state=state,
        slippage=context.backtest_kwargs["slippage"],
        commission=context.backtest_kwargs["commission"],
        position_size=context.backtest_kwargs["position_size"],
        path_horizon=path_horizon,
        horizons=horizons,
    )

    baseline_result = run_backtest(context.dataset_path, **context.backtest_kwargs)
    closed_trades_df = pair_round_trip_trades(
        baseline_result.get("trades", []),
        configured_hold_bars=int(context.backtest_kwargs["trend_pullback_hold_bars"]),
    )
    if not closed_trades_df.empty:
        closed_trades_df["slippage_per_share"] = float(context.backtest_kwargs["slippage"])
    trade_paths_df = attach_trade_path_metrics(closed_trades_df, signal_paths_df)
    cost_drag_df = build_cost_drag_table(closed_trades_df)

    hold_runs = []
    for hold_bars in compare_holds:
        hold_kwargs = dict(context.backtest_kwargs)
        hold_kwargs["trend_pullback_hold_bars"] = hold_bars
        hold_runs.append(HoldRun(hold_bars=hold_bars, result=run_backtest(context.dataset_path, **hold_kwargs)))
    realized_hold_df = build_realized_hold_summary(hold_runs)
    hold_delta_summary = summarize_hold_delta(signal_paths_df, hold_a=compare_holds[0], hold_b=compare_holds[1])
    opportunity_summary = summarize_opportunity_capture(trade_paths_df)
    shape_summary_df = build_trade_shape_summary(trade_paths_df)
    cost_drag_summary = _summarize_cost_drag(cost_drag_df)
    diagnosis = diagnose_trade_path(
        trade_paths_df=trade_paths_df,
        hold_delta_summary=hold_delta_summary,
        realized_hold_df=realized_hold_df,
        cost_drag_summary=cost_drag_summary,
    )

    summary_df = pd.DataFrame(
        [
            {
                "raw_signal_count": int(len(signal_paths_df)),
                "closed_trade_count": int(len(trade_paths_df)),
                "avg_signal_to_entry_gap_pct": float(signal_paths_df["signal_to_entry_gap_pct"].dropna().mean())
                if not signal_paths_df.empty and signal_paths_df["signal_to_entry_gap_pct"].notna().any()
                else 0.0,
                "avg_best_net_exit_pct": float(signal_paths_df["best_net_exit_pct"].mean()) if not signal_paths_df.empty else 0.0,
                "avg_realized_return_pct": float(trade_paths_df["realized_return_pct"].mean()) if not trade_paths_df.empty else 0.0,
                "diagnosis": diagnosis,
            }
        ]
    )

    print("\nTrade Path Summary")
    print(_frame_text(summary_df, max_rows=10))
    print("\nClosed Trade Paths")
    closed_columns = [
        "symbol", "entry_ts", "exit_ts", "entry_price", "exit_price", "holding_bars",
        "realized_pnl", "realized_return_pct", "exit_reason", "net_exit_1b_return_pct",
        "net_exit_2b_return_pct", "net_exit_3b_return_pct", "net_exit_4b_return_pct",
        "mfe_pct", "mae_pct", "peak_favorable_bar", "worst_adverse_bar",
    ]
    print(_frame_text(trade_paths_df[closed_columns] if not trade_paths_df.empty else pd.DataFrame(), max_rows=20))
    print(f"\nHold {compare_holds[0]} vs Hold {compare_holds[1]}: Same Entry Set")
    print(_frame_text(_to_single_row_df(hold_delta_summary), max_rows=10))
    print("\nRealized Hold Comparison")
    print(_frame_text(realized_hold_df, max_rows=10))
    print("\nOpportunity Capture")
    print(_frame_text(_to_single_row_df(opportunity_summary), max_rows=10))
    print("\nCost / Friction Drag")
    print(_frame_text(_to_single_row_df(cost_drag_summary), max_rows=10))
    print("\nTrade Shapes")
    print(_frame_text(shape_summary_df, max_rows=10))

    if args.output_dir:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        summary_df.to_csv(output_dir / "trade_path_summary.csv", index=False)
        trade_paths_df.to_csv(output_dir / "closed_trade_paths.csv", index=False)
        signal_paths_df.to_csv(output_dir / "signal_paths.csv", index=False)
        realized_hold_df.to_csv(output_dir / "hold_comparison.csv", index=False)
        shape_summary_df.to_csv(output_dir / "trade_shapes.csv", index=False)
        (output_dir / "trade_path_summary.json").write_text(
            json.dumps(
                {
                    "summary": summary_df.to_dict("records"),
                    "hold_delta_summary": hold_delta_summary,
                    "opportunity_summary": opportunity_summary,
                    "cost_drag_summary": cost_drag_summary,
                    "diagnosis": diagnosis,
                },
                indent=2,
                default=str,
            ),
            encoding="utf-8",
        )
    log_experiment_run(
        run_type="trade_path_diagnostics",
        script_path=__file__,
        strategy_name=STRATEGY_MODE_TREND_PULLBACK,
        dataset_path=context.dataset_path,
        symbols=symbols,
        params={
            "config": args.config,
            "start_date": args.start_date,
            "end_date": args.end_date,
            "commission_per_order": args.commission_per_order,
            "slippage_per_share": args.slippage_per_share,
            "position_size": args.position_size,
            "compare_holds": list(compare_holds),
            "path_horizon": args.path_horizon,
        },
        metrics={
            "trade_count": int(len(trade_paths_df)),
            "expectancy": float(trade_paths_df["realized_pnl"].mean()) if not trade_paths_df.empty else None,
            "realized_pnl": float(trade_paths_df["realized_pnl"].sum()) if not trade_paths_df.empty else None,
        },
        output_path=args.output_dir,
        summary_path=(Path(args.output_dir) / "trade_path_summary.json") if args.output_dir else None,
        extra_fields={
            "diagnosis": diagnosis,
            "hold_delta_summary": hold_delta_summary,
            "opportunity_summary": opportunity_summary,
        },
    )


if __name__ == "__main__":
    main()

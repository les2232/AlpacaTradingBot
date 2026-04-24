"""
Controlled experiment: SMA mean reversion vs VWAP Z-score mean reversion.

Both runs use identical execution assumptions (same dataset, symbols, capital,
commission, slippage, position size, time window).  The ONLY difference is the
entry/exit signal logic.

SMA baseline
------------
  Entry : price < SMA_20 * (1 - entry_threshold_pct)
  Exit  : price >= SMA_20
  Stop  : price <= entry_price * (1 - mean_reversion_stop_pct)

VWAP Z-score
------------
  Entry : z = (close - vwap) / atr  <  -vwap_z_entry_threshold
  Exit  : price >= session VWAP
  Stop  : price <= entry_price - vwap_z_stop_atr_multiple * ATR

Run:
    python compare_mr_strategies.py
"""

from __future__ import annotations

import math
from datetime import datetime, timezone
from pathlib import Path

from backtest_runner import run_backtest

# ---------------------------------------------------------------------------
# Shared experiment settings — identical for both runs
# ---------------------------------------------------------------------------

DATASET = Path(
    "datasets/"
    "AAPL-AMD-AMZN-GOOGL-JPM-KO-META-MSFT-NVDA-TSLA-XOM"
    "__15Min__20251004T000000Z__20260404T000000Z__sip__7bdcdff15c5f"
)

COMMON = dict(
    dataset_path=DATASET,
    strategy_mode="mean_reversion",
    sma_bars=20,
    starting_capital=10_000.0,
    position_size=1_000.0,
    commission=0.01,
    slippage=0.05,
    time_window_mode="full_day",
    regime_filter_enabled=False,
)

# SMA-specific
SMA_CONFIG = dict(
    **COMMON,
    entry_threshold_pct=0.01,      # enter when price is 1% below SMA_20
    mean_reversion_exit_style="sma",
    mean_reversion_stop_pct=0.02,  # 2% hard stop below entry
    vwap_z_entry_threshold=0.0,    # 0 = disable VWAP path → SMA fallback
    vwap_z_stop_atr_multiple=2.0,
)

# VWAP Z-score
VWAP_CONFIG = dict(
    **COMMON,
    vwap_z_entry_threshold=1.5,    # enter when z < -1.5
    vwap_z_stop_atr_multiple=2.0,  # stop = entry - 2 * ATR
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_ts(ts_str: str) -> datetime:
    """Parse ISO timestamp string, strip microseconds for clean display."""
    try:
        return datetime.fromisoformat(str(ts_str).replace("Z", "+00:00"))
    except Exception:
        return datetime(1970, 1, 1, tzinfo=timezone.utc)


def _fmt_ts(ts_str: str) -> str:
    try:
        dt = _parse_ts(ts_str)
        return dt.astimezone(timezone.utc).strftime("%Y-%m-%d %H:%M ET")
    except Exception:
        return str(ts_str)


def compute_round_trips(trades: list[dict]) -> list[dict]:
    """
    Pair BUY and SELL records into completed round-trip trades.

    Returns a list of dicts with:
        symbol, entry_price, exit_price, entry_ts, exit_ts,
        pnl, hold_bars (estimated), exit_reason
    """
    pending: dict[str, dict] = {}  # symbol → last BUY record
    round_trips: list[dict] = []

    for t in trades:
        sym = t["symbol"]
        if t["side"] == "BUY":
            pending[sym] = t
        elif t["side"] == "SELL" and sym in pending:
            buy = pending.pop(sym)
            entry_dt = _parse_ts(buy["timestamp"])
            exit_dt = _parse_ts(t["timestamp"])
            duration_mins = (exit_dt - entry_dt).total_seconds() / 60.0

            if t.get("eod_exit"):
                reason = "EOD force-close"
            elif t.get("forced_close"):
                reason = "Final forced close"
            elif t.get("pnl", 0.0) > 0:
                reason = "Profit target (anchor return)"
            else:
                reason = "Stop loss"

            round_trips.append({
                "symbol": sym,
                "entry_price": buy["price"],
                "exit_price": t["price"],
                "entry_ts": buy["timestamp"],
                "exit_ts": t["timestamp"],
                "pnl": t.get("pnl", 0.0),
                "hold_mins": duration_mins,
                "exit_reason": reason,
            })

    return round_trips


def avg_holding_mins(round_trips: list[dict]) -> float:
    if not round_trips:
        return 0.0
    return sum(rt["hold_mins"] for rt in round_trips) / len(round_trips)


def print_metrics(label: str, r: dict, round_trips: list[dict]) -> None:
    wins = [rt["pnl"] for rt in round_trips if rt["pnl"] > 0]
    losses = [rt["pnl"] for rt in round_trips if rt["pnl"] <= 0]
    gross_profit = sum(wins)
    gross_loss = abs(sum(losses))
    profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else (
        float("inf") if gross_profit > 0 else 0.0
    )

    print(f"\n### {label}")
    print(f"  Total return      : {r.get('total_return_pct', 0.0):+.2f}%")
    print(f"  Realized P&L      : ${r.get('realized_pnl', 0.0):+.2f}")
    print(f"  Total trades      : {r.get('total_trades', 0)}")
    print(f"  Trades / day      : {r.get('trades_per_day', 0.0):.2f}")
    print(f"  Win rate          : {r.get('win_rate', 0.0):.1f}%")
    print(f"  Avg win           : ${r.get('avg_winning_trade', 0.0):+.2f}")
    print(f"  Avg loss          : ${r.get('avg_losing_trade', 0.0):+.2f}")
    print(f"  Expectancy/trade  : ${r.get('expectancy', 0.0):+.2f}")
    print(f"  Profit factor     : {profit_factor:.2f}")
    print(f"  Max drawdown      : {r.get('max_drawdown_pct', 0.0):.2f}%")
    print(f"  Sharpe ratio      : {r.get('sharpe_ratio', 0.0):.2f}")
    print(f"  Avg hold time     : {avg_holding_mins(round_trips):.0f} min")
    print(f"  Round trips       : {len(round_trips)}")


def print_example_trades(
    label: str,
    round_trips: list[dict],
    n: int = 5,
) -> None:
    if not round_trips:
        print(f"\n  [No completed trades for {label}]")
        return

    sorted_by_pnl = sorted(round_trips, key=lambda rt: rt["pnl"])
    worst = sorted_by_pnl[:3]           # 3 worst losses
    best = sorted_by_pnl[-3:][::-1]     # 3 best wins
    # pick 2 "interesting middle" trades (longest holds, losers above median pnl)
    median_pnl = sorted(rt["pnl"] for rt in round_trips)[len(round_trips) // 2]
    slow_losses = [rt for rt in round_trips if rt["pnl"] < 0 and rt["hold_mins"] > 60]
    slow_losses.sort(key=lambda rt: rt["hold_mins"], reverse=True)
    interesting = slow_losses[:2]

    showcase: list[tuple[str, list[dict]]] = [
        ("Worst trades (largest losses)", worst),
        ("Best trades (largest gains)", best),
    ]
    if interesting:
        showcase.append(("Slow-moving losses (held >60 min before stop)", interesting))

    print(f"\n#### Example trades — {label}")
    for section_label, trades in showcase:
        if not trades:
            continue
        print(f"\n  [{section_label}]")
        for rt in trades:
            entry_pct = ((rt["exit_price"] - rt["entry_price"]) / rt["entry_price"]) * 100
            print(
                f"    {rt['symbol']:5s}"
                f"  entry {rt['entry_price']:8.2f}"
                f"  exit {rt['exit_price']:8.2f}"
                f"  move {entry_pct:+.2f}%"
                f"  held {rt['hold_mins']:5.0f}min"
                f"  pnl ${rt['pnl']:+.2f}"
                f"  [{rt['exit_reason']}]"
            )
            print(
                f"         entered {_fmt_ts(rt['entry_ts'])}"
                f"  -> exited {_fmt_ts(rt['exit_ts'])}"
            )


def _pct_diff(a: float, b: float) -> str:
    if a == 0:
        return "N/A (baseline is 0)"
    diff = (b - a) / abs(a) * 100
    return f"{diff:+.1f}%"


def print_comparison(
    sma_r: dict,
    vwap_r: dict,
    sma_rt: list[dict],
    vwap_rt: list[dict],
) -> None:
    sma_pf = _profit_factor_from_rt(sma_rt)
    vwap_pf = _profit_factor_from_rt(vwap_rt)
    sma_avg_hold = avg_holding_mins(sma_rt)
    vwap_avg_hold = avg_holding_mins(vwap_rt)

    sma_trades = sma_r.get("total_trades", 0)
    vwap_trades = vwap_r.get("total_trades", 0)
    sma_wr = sma_r.get("win_rate", 0.0)
    vwap_wr = vwap_r.get("win_rate", 0.0)
    sma_dd = sma_r.get("max_drawdown_pct", 0.0)
    vwap_dd = vwap_r.get("max_drawdown_pct", 0.0)
    sma_ret = sma_r.get("total_return_pct", 0.0)
    vwap_ret = vwap_r.get("total_return_pct", 0.0)

    print("\n" + "=" * 72)
    print("### Comparison Analysis")
    print("=" * 72)
    print(f"  Total return        : SMA {sma_ret:+.2f}%  vs  VWAP {vwap_ret:+.2f}%"
          f"  (delta {vwap_ret - sma_ret:+.2f}pp)")
    print(f"  Trade frequency     : SMA {sma_trades}  vs  VWAP {vwap_trades}"
          f"  ({_pct_diff(sma_trades, vwap_trades)} change)")
    print(f"  Win rate            : SMA {sma_wr:.1f}%  vs  VWAP {vwap_wr:.1f}%"
          f"  (delta {vwap_wr - sma_wr:+.1f}pp)")
    print(f"  Profit factor       : SMA {sma_pf:.2f}  vs  VWAP {vwap_pf:.2f}")
    print(f"  Max drawdown        : SMA {sma_dd:.2f}%  vs  VWAP {vwap_dd:.2f}%")
    print(f"  Avg hold time       : SMA {sma_avg_hold:.0f}min  vs  VWAP {vwap_avg_hold:.0f}min")

    print("\n  Observed behavioral differences:")

    # Trade frequency interpretation
    if vwap_trades > sma_trades * 1.3:
        print("  - VWAP generates significantly MORE trades — its intraday anchor")
        print("    refreshes daily, creating more entry opportunities than the")
        print("    rolling SMA which can drift far from price over days.")
    elif vwap_trades < sma_trades * 0.7:
        print("  - VWAP generates significantly FEWER trades — z-score normalization")
        print("    by ATR means extreme moves must be large relative to recent")
        print("    volatility, not just a fixed % from a rolling mean.")
    else:
        print("  - Trade frequency is broadly similar between the two strategies.")

    # Win rate interpretation
    wr_delta = vwap_wr - sma_wr
    if abs(wr_delta) > 5:
        better = "VWAP" if wr_delta > 0 else "SMA"
        print(f"  - {better} shows meaningfully higher win rate ({abs(wr_delta):.1f}pp diff).")
        if better == "VWAP":
            print("    VWAP as exit target is more reachable intraday than SMA,")
            print("    since VWAP is computed within the same session.")
        else:
            print("    SMA exit is more forgiving — allows reversion beyond the session.")

    # Hold time interpretation
    hold_delta = vwap_avg_hold - sma_avg_hold
    if vwap_avg_hold < sma_avg_hold * 0.7:
        print("  - VWAP trades close much faster — VWAP reversion is an intraday")
        print("    anchor that can be reached within 1–3 bars, while SMA exit may")
        print("    require waiting through multiple sessions.")
    elif vwap_avg_hold > sma_avg_hold * 1.3:
        print("  - VWAP trades hold longer — the session VWAP moves with volume")
        print("    distribution and may be further from entry than the SMA.")

    # Drawdown interpretation
    dd_delta = vwap_dd - sma_dd
    if vwap_dd < sma_dd - 1.0:
        print("  - VWAP has lower max drawdown: ATR-based stops scale with current")
        print("    volatility, cutting losses faster on high-volatility bars.")
    elif vwap_dd > sma_dd + 1.0:
        print("  - VWAP has higher max drawdown: ATR-scaled stops can be wider on")
        print("    low-volatility entries, allowing more adverse excursion.")

    # Profit factor
    if vwap_pf > sma_pf + 0.15:
        print("  - VWAP has better profit factor: the ATR-normalized entry selects")
        print("    for moves that are extreme relative to current realized volatility,")
        print("    filtering out small wiggles that the fixed-% SMA threshold catches.")
    elif sma_pf > vwap_pf + 0.15:
        print("  - SMA has better profit factor: fixed-% entry combined with SMA")
        print("    exit provides longer runway for recovery vs ATR stop which may")
        print("    exit prematurely during volatile mean-reverting moves.")

    print()
    print("  Signal quality note:")
    print("  " + "-" * 65)
    print("  The VWAP path normalizes entry threshold by ATR (z-score), making")
    print("  signal sensitivity adaptive to daily volatility. The SMA path uses")
    print("  a fixed %-below-SMA threshold, which is more selective in quiet")
    print("  markets and less selective during volatile periods. Neither is")
    print("  strictly superior -- the right choice depends on the volatility")
    print("  regime you expect to trade in.")
    print("  " + "-" * 65)
    print()
    print("  [!] These results are IN-SAMPLE on the training dataset. Parameters")
    print("      were NOT optimized here, but the strategy architectures were")
    print("      designed with knowledge of this data period. Out-of-sample")
    print("      walk-forward validation is required before drawing conclusions.")


def _profit_factor_from_rt(round_trips: list[dict]) -> float:
    wins = [rt["pnl"] for rt in round_trips if rt["pnl"] > 0]
    losses = [rt["pnl"] for rt in round_trips if rt["pnl"] <= 0]
    gross_profit = sum(wins)
    gross_loss = abs(sum(losses))
    if gross_loss > 0:
        return gross_profit / gross_loss
    return float("inf") if gross_profit > 0 else 0.0


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("=" * 72)
    print("MR STRATEGY COMPARISON: SMA Baseline vs VWAP Z-Score")
    print(f"Dataset : {DATASET}")
    print(f"Symbols : AAPL AMD AMZN GOOGL JPM KO META MSFT NVDA TSLA XOM")
    print(f"Period  : Oct 2025 – Apr 2026 (6 months, 15-min bars)")
    print(f"Capital : $10,000 total, $1,000 per position")
    print(f"Costs   : $0.01 commission + $0.05/share slippage per side")
    print("=" * 72)

    # ---- Run 1: SMA baseline ----
    print("\n[1/2] Running SMA mean reversion baseline...")
    sma_results = run_backtest(**SMA_CONFIG)
    sma_rt = compute_round_trips(sma_results.get("trades", []))

    # ---- Run 2: VWAP Z-score ----
    print("\n[2/2] Running VWAP Z-score mean reversion...")
    vwap_results = run_backtest(**VWAP_CONFIG)
    vwap_rt = compute_round_trips(vwap_results.get("trades", []))

    # ---- Report ----
    print("\n" + "=" * 72)
    print_metrics("SMA Mean Reversion", sma_results, sma_rt)
    print_metrics("VWAP Z-Score Mean Reversion", vwap_results, vwap_rt)

    print_example_trades("SMA Mean Reversion", sma_rt, n=5)
    print_example_trades("VWAP Z-Score Mean Reversion", vwap_rt, n=5)

    print_comparison(sma_results, vwap_results, sma_rt, vwap_rt)


if __name__ == "__main__":
    main()

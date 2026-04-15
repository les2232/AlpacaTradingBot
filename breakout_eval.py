"""
Breakout strategy evaluation — pivot from VWAP MR to trend-following.

Context:
    Three stages of VWAP MR sweeps produced a best OOS profit factor of 0.960
    (entry=1.50, stop=1.00, atr>=50) — structurally better than SMA MR but
    still unprofitable.  The flat PF surface across all parameter combinations
    indicates a weak underlying signal rather than a parameter tuning problem.
    This script evaluates the existing ORB breakout mode on the same dataset
    and IS/OOS split to determine whether trend-following is a better fit.

Grid (32 combos):
    breakout_exit_style   : [target_1x_stop_low, target_1.5x_stop_low,
                             target_1x_tight_stop, trailing_stop_half_range]
    breakout_max_stop_pct : [0.02, 0.03]
    time_window_mode      : [morning_only, full_day]
    regime_filter_enabled : [False, True]

Reference:
    Best VWAP MR config from stage 2 (OOS PF 0.960) included for comparison.
    SMA baseline also included.

Run:
    python breakout_eval.py
"""

from __future__ import annotations

import contextlib
import io
import itertools
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from backtest_runner import run_backtest
from strategy import (
    BREAKOUT_EXIT_TARGET_1X_STOP_LOW,
    BREAKOUT_EXIT_TARGET_1_5X_STOP_LOW,
    BREAKOUT_EXIT_TARGET_1X_TIGHT_STOP,
    BREAKOUT_EXIT_TRAILING_HALF_RANGE,
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DATASET = Path(
    "datasets/"
    "AAPL-AMD-AMZN-GOOGL-JPM-KO-META-MSFT-NVDA-TSLA-XOM"
    "__15Min__20251004T000000Z__20260404T000000Z__sip__7bdcdff15c5f"
)

IS_START  = "2025-10-04"
IS_END    = "2026-02-07"
OOS_START = "2026-02-08"
OOS_END   = "2026-04-04"

EXIT_STYLES = [
    BREAKOUT_EXIT_TARGET_1X_STOP_LOW,
    BREAKOUT_EXIT_TARGET_1_5X_STOP_LOW,
    BREAKOUT_EXIT_TARGET_1X_TIGHT_STOP,
    BREAKOUT_EXIT_TRAILING_HALF_RANGE,
]

EXIT_LABELS = {
    BREAKOUT_EXIT_TARGET_1X_STOP_LOW:      "1x_or_stop",
    BREAKOUT_EXIT_TARGET_1_5X_STOP_LOW:    "1.5x_or_stop",
    BREAKOUT_EXIT_TARGET_1X_TIGHT_STOP:    "1x_tight_stop",
    BREAKOUT_EXIT_TRAILING_HALF_RANGE:     "trail_half",
}

MAX_STOP_PCTS    = [0.02, 0.03]
TIME_WINDOWS     = ["morning_only", "full_day"]
REGIME_FILTERS   = [False, True]

COMMON = dict(
    dataset_path=DATASET,
    strategy_mode="breakout",
    sma_bars=20,
    starting_capital=10_000.0,
    position_size=1_000.0,
    commission=0.01,
    slippage=0.05,
)

# Best VWAP MR config from stage-2 sweep (reference)
VWAP_MR_REF = dict(
    **{k: v for k, v in COMMON.items() if k != "strategy_mode"},
    strategy_mode="mean_reversion",
    time_window_mode="full_day",
    regime_filter_enabled=False,
    vwap_z_entry_threshold=1.5,
    vwap_z_stop_atr_multiple=1.0,
    min_atr_percentile=50.0,
    max_adx_threshold=0.0,
)

# SMA baseline (carried forward from stage 1)
SMA_REF = dict(
    **{k: v for k, v in COMMON.items() if k != "strategy_mode"},
    strategy_mode="mean_reversion",
    time_window_mode="full_day",
    regime_filter_enabled=False,
    entry_threshold_pct=0.01,
    mean_reversion_exit_style="sma",
    mean_reversion_stop_pct=0.02,
    vwap_z_entry_threshold=0.0,
    vwap_z_stop_atr_multiple=2.0,
)

MIN_MEANINGFUL_OOS_TRADES = 20


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class EvalResult:
    exit_style: str
    max_stop_pct: float
    time_window: str
    regime_filter: bool
    is_pf: float
    oos_pf: float
    is_return: float
    oos_return: float
    is_dd: float
    oos_dd: float
    is_trades: int
    oos_trades: int
    oos_wr: float


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _profit_factor(trades: list[dict]) -> float:
    pending: dict[str, dict] = {}
    pnls: list[float] = []
    for t in trades:
        sym = t["symbol"]
        if t["side"] == "BUY":
            pending[sym] = t
        elif t["side"] == "SELL" and sym in pending:
            pnls.append(t.get("pnl", 0.0))
            pending.pop(sym)
    gross_win  = sum(p for p in pnls if p > 0)
    gross_loss = abs(sum(p for p in pnls if p <= 0))
    if gross_loss == 0:
        return float("inf") if gross_win > 0 else 0.0
    return gross_win / gross_loss


def _run_silent(kwargs: dict) -> dict:
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        return run_backtest(**kwargs)


def _collect(r: dict) -> tuple[float, float, float, int, float]:
    return (
        _profit_factor(r.get("trades", [])),
        r.get("total_return_pct", 0.0),
        r.get("max_drawdown_pct", 0.0),
        r.get("total_trades", 0),
        r.get("win_rate", 0.0),
    )


def _pf_str(pf: float) -> str:
    if pf == float("inf"):
        return "  inf"
    return f"{min(pf, 9.999):.3f}"


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def run_eval() -> tuple[list[EvalResult], dict, dict, dict, dict]:
    grid = list(itertools.product(EXIT_STYLES, MAX_STOP_PCTS, TIME_WINDOWS, REGIME_FILTERS))
    total = len(grid)
    results: list[EvalResult] = []

    print(f"\n[eval] {total} combos x 2 windows = {total * 2} breakout runs")
    print(f"[eval] IS : {IS_START} to {IS_END}")
    print(f"[eval] OOS: {OOS_START} to {OOS_END}\n")

    for i, (exit_style, max_stop, time_win, regime) in enumerate(grid):
        run_is  = i * 2 + 1
        run_oos = i * 2 + 2
        label   = f"{EXIT_LABELS[exit_style]} stop={max_stop:.2f} {time_win[:3]} regime={regime}"

        print(f"[{run_is:3d}/{total * 2}] {label}  IS ...", end=" ", flush=True)
        kwargs_base = dict(
            **COMMON,
            breakout_exit_style=exit_style,
            breakout_max_stop_pct=max_stop,
            time_window_mode=time_win,
            regime_filter_enabled=regime,
        )
        if i == 0:
            is_r = run_backtest(**kwargs_base, start_date=IS_START, end_date=IS_END)
        else:
            is_r = _run_silent({**kwargs_base, "start_date": IS_START, "end_date": IS_END})
        print("done")

        print(f"[{run_oos:3d}/{total * 2}] {label}  OOS ...", end=" ", flush=True)
        oos_r = _run_silent({**kwargs_base, "start_date": OOS_START, "end_date": OOS_END})
        print("done")

        is_pf,  is_ret,  is_dd,  is_tr,  _      = _collect(is_r)
        oos_pf, oos_ret, oos_dd, oos_tr, oos_wr = _collect(oos_r)
        results.append(EvalResult(
            exit_style=exit_style, max_stop_pct=max_stop,
            time_window=time_win,  regime_filter=regime,
            is_pf=is_pf,   oos_pf=oos_pf,
            is_return=is_ret, oos_return=oos_ret,
            is_dd=is_dd,   oos_dd=oos_dd,
            is_trades=is_tr, oos_trades=oos_tr, oos_wr=oos_wr,
        ))

    # Reference runs
    print()
    for label, cfg in [("VWAP MR ref IS ", {**VWAP_MR_REF, "start_date": IS_START,  "end_date": IS_END}),
                       ("VWAP MR ref OOS", {**VWAP_MR_REF, "start_date": OOS_START, "end_date": OOS_END}),
                       ("SMA ref IS     ", {**SMA_REF,     "start_date": IS_START,  "end_date": IS_END}),
                       ("SMA ref OOS    ", {**SMA_REF,     "start_date": OOS_START, "end_date": OOS_END})]:
        print(f"[ref] {label} ...", end=" ", flush=True)
        r = _run_silent(cfg)
        print("done")

    vwap_is_r  = _run_silent({**VWAP_MR_REF, "start_date": IS_START,  "end_date": IS_END})
    vwap_oos_r = _run_silent({**VWAP_MR_REF, "start_date": OOS_START, "end_date": OOS_END})
    sma_is_r   = _run_silent({**SMA_REF,     "start_date": IS_START,  "end_date": IS_END})
    sma_oos_r  = _run_silent({**SMA_REF,     "start_date": OOS_START, "end_date": OOS_END})

    return results, vwap_is_r, vwap_oos_r, sma_is_r, sma_oos_r


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def _rank_key(r: EvalResult) -> tuple:
    return (-min(r.oos_pf, 9.999), r.oos_dd, -r.oos_return)


def print_results(
    results: list[EvalResult],
    vwap_is: dict, vwap_oos: dict,
    sma_is: dict,  sma_oos: dict,
) -> None:
    vwap_oos_pf  = _profit_factor(vwap_oos.get("trades", []))
    vwap_is_pf   = _profit_factor(vwap_is.get("trades", []))
    vwap_oos_ret = vwap_oos.get("total_return_pct", 0.0)
    vwap_is_ret  = vwap_is.get("total_return_pct", 0.0)
    vwap_oos_dd  = vwap_oos.get("max_drawdown_pct", 0.0)
    vwap_oos_tr  = vwap_oos.get("total_trades", 0)
    vwap_is_tr   = vwap_is.get("total_trades", 0)

    sma_oos_pf   = _profit_factor(sma_oos.get("trades", []))
    sma_is_pf    = _profit_factor(sma_is.get("trades", []))
    sma_oos_ret  = sma_oos.get("total_return_pct", 0.0)
    sma_is_ret   = sma_is.get("total_return_pct", 0.0)
    sma_oos_dd   = sma_oos.get("max_drawdown_pct", 0.0)
    sma_oos_tr   = sma_oos.get("total_trades", 0)
    sma_is_tr    = sma_is.get("total_trades", 0)

    ranked = sorted(results, key=_rank_key)

    # --- Top 10 table ---
    print()
    print("=" * 100)
    print("  BREAKOUT EVALUATION  —  TOP CONFIGURATIONS  (ranked by OOS profit factor)")
    print("=" * 100)

    hdr = (
        f"{'Rank':>4}  {'Exit Style':>14}  {'Stop':>4}  {'Window':>7}  {'Rgm':>3}  "
        f"{'IS-PF':>6}  {'OOS-PF':>6}  "
        f"{'IS-Ret':>7}  {'OOS-Ret':>7}  {'OOS-DD':>7}  {'IS-Tr':>6}  {'OOS-Tr':>6}"
    )
    sep = "-" * 100
    print(hdr)
    print(sep)

    for rank, r in enumerate(ranked[:10], 1):
        beats_vwap = r.oos_pf > vwap_oos_pf
        low_tr     = r.oos_trades < MIN_MEANINGFUL_OOS_TRADES
        flag = " *" if beats_vwap else "  "
        if low_tr:
            flag = " !"
        print(
            f"{rank:>4}{flag} {EXIT_LABELS[r.exit_style]:>14}  {r.max_stop_pct:.2f}"
            f"  {r.time_window[:7]:>7}  {'Y' if r.regime_filter else 'N':>3}  "
            f"{_pf_str(r.is_pf):>6}  {_pf_str(r.oos_pf):>6}  "
            f"{r.is_return:>+7.2f}%  {r.oos_return:>+7.2f}%"
            f"  {r.oos_dd:>7.2f}%  {r.is_trades:>6}  {r.oos_trades:>6}"
        )

    print(sep)
    # Reference rows
    for label, is_pf, oos_pf, is_ret, oos_ret, oos_dd, is_tr, oos_tr in [
        ("VWAP-MR-best", vwap_is_pf, vwap_oos_pf, vwap_is_ret, vwap_oos_ret, vwap_oos_dd, vwap_is_tr, vwap_oos_tr),
        ("SMA-baseline ", sma_is_pf,  sma_oos_pf,  sma_is_ret,  sma_oos_ret,  sma_oos_dd,  sma_is_tr,  sma_oos_tr),
    ]:
        print(
            f"{'ref':>4}   {label:>14}  {'--':>4}  {'--':>7}  {'--':>3}  "
            f"{_pf_str(is_pf):>6}  {_pf_str(oos_pf):>6}  "
            f"{is_ret:>+7.2f}%  {oos_ret:>+7.2f}%"
            f"  {oos_dd:>7.2f}%  {is_tr:>6}  {oos_tr:>6}"
        )
    print(sep)
    print("  * = beats VWAP MR best OOS PF    ! = fewer than 20 OOS trades (low confidence)")


def print_analysis(
    results: list[EvalResult],
    vwap_is: dict, vwap_oos: dict,
    sma_is: dict,  sma_oos: dict,
) -> None:
    vwap_oos_pf = _profit_factor(vwap_oos.get("trades", []))
    vwap_oos_dd = vwap_oos.get("max_drawdown_pct", 0.0)
    vwap_oos_tr = vwap_oos.get("total_trades", 0)
    sma_oos_pf  = _profit_factor(sma_oos.get("trades", []))

    ranked_all   = sorted(results, key=_rank_key)
    reliable     = [r for r in results if r.oos_trades >= MIN_MEANINGFUL_OOS_TRADES]
    above_1      = [r for r in reliable if r.oos_pf > 1.0]
    beats_vwap   = [r for r in reliable if r.oos_pf > vwap_oos_pf]

    def avg(lst: list) -> float:
        return sum(lst) / len(lst) if lst else 0.0

    # Per-exit-style aggregates
    pf_by_exit:  dict[str, list[float]] = defaultdict(list)
    tr_by_exit:  dict[str, list[int]]   = defaultdict(list)
    dd_by_exit:  dict[str, list[float]] = defaultdict(list)
    ret_by_exit: dict[str, list[float]] = defaultdict(list)
    gap_by_exit: dict[str, list[float]] = defaultdict(list)

    for r in results:
        if r.oos_pf != float("inf"):
            pf_by_exit[r.exit_style].append(r.oos_pf)
        tr_by_exit[r.exit_style].append(r.oos_trades)
        dd_by_exit[r.exit_style].append(r.oos_dd)
        ret_by_exit[r.exit_style].append(r.oos_return)
        gap_by_exit[r.exit_style].append(r.is_pf - r.oos_pf)

    # Per-window aggregates
    pf_by_win:  dict[str, list[float]] = defaultdict(list)
    tr_by_win:  dict[str, list[int]]   = defaultdict(list)
    for r in results:
        if r.oos_pf != float("inf"):
            pf_by_win[r.time_window].append(r.oos_pf)
        tr_by_win[r.time_window].append(r.oos_trades)

    # Per-regime aggregates
    pf_by_regime: dict[bool, list[float]] = defaultdict(list)
    for r in results:
        if r.oos_pf != float("inf"):
            pf_by_regime[r.regime_filter].append(r.oos_pf)

    print()
    print("=" * 72)
    print("  ANALYSIS")
    print("=" * 72)

    # --- Q1: Does PF exceed 1.0? ---
    print()
    print("  1. Does any breakout configuration exceed OOS profit factor 1.0?")
    print()

    if above_1:
        above_1_sorted = sorted(above_1, key=_rank_key)
        print(f"  -> YES. {len(above_1)} of {len(reliable)} reliable configurations exceed OOS PF 1.0.")
        print()
        for r in above_1_sorted[:6]:
            print(f"     {EXIT_LABELS[r.exit_style]:>14}  stop={r.max_stop_pct:.2f}"
                  f"  {r.time_window[:7]}  regime={'Y' if r.regime_filter else 'N'}"
                  f"  OOS PF={_pf_str(r.oos_pf)}"
                  f"  OOS ret={r.oos_return:+.2f}%"
                  f"  OOS DD={r.oos_dd:.2f}%"
                  f"  trades={r.oos_trades}")
        if len(above_1_sorted) > 6:
            print(f"     ... and {len(above_1_sorted) - 6} more.")
    else:
        best = ranked_all[0]
        print(f"  -> NO. No reliable configuration reaches OOS PF 1.0.")
        print(f"     Best: {EXIT_LABELS[best.exit_style]}  stop={best.max_stop_pct:.2f}"
              f"  {best.time_window[:7]}  OOS PF={_pf_str(best.oos_pf)}"
              f"  trades={best.oos_trades}")

    if beats_vwap:
        print()
        print(f"  {len(beats_vwap)} of {len(reliable)} configs beat VWAP MR best OOS PF ({_pf_str(vwap_oos_pf)}).")
    else:
        print()
        print(f"  0 breakout configs beat VWAP MR best OOS PF ({_pf_str(vwap_oos_pf)}).")

    # --- Q2: Trade count vs VWAP MR ---
    print()
    print("  2. Trade count vs VWAP MR and SMA baselines:")
    print()

    all_oos_tr = [r.oos_trades for r in reliable]
    avg_breakout_tr = avg(all_oos_tr)
    print(f"     Avg breakout OOS trades   : {avg_breakout_tr:.0f}")
    print(f"     VWAP MR best OOS trades   : {vwap_oos_tr}")
    print(f"     SMA baseline OOS trades   : {sma_oos.get('total_trades', 0)}")

    if avg_breakout_tr < vwap_oos_tr * 0.5:
        reduction = (vwap_oos_tr - avg_breakout_tr) / vwap_oos_tr * 100
        print()
        print(f"  -> Breakout trades {reduction:.0f}% fewer than VWAP MR.")
        print("     ORB fires at most once per symbol per day (single setup per session).")
        print("     This is expected — breakout quality > quantity.")
    else:
        print()
        print(f"  -> Breakout trade count is broadly comparable to VWAP MR.")

    # --- Q3: Exit-style breakdown — are losses reduced or just fewer trades? ---
    print()
    print("  3. Per-exit-style breakdown (are losses reduced or just fewer trades?):")
    print()
    print(f"     {'Exit Style':>14}  {'Avg OOS PF':>10}  {'Avg OOS Trades':>14}  "
          f"{'Avg OOS DD':>10}  {'Avg OOS Ret':>11}  {'IS-OOS gap':>10}")
    print("     " + "-" * 72)

    exit_order = sorted(pf_by_exit.keys(), key=lambda e: -avg(pf_by_exit[e]))
    for exit_style in exit_order:
        avg_pf  = avg(pf_by_exit[exit_style])
        avg_tr  = avg(tr_by_exit[exit_style])
        avg_dd  = avg(dd_by_exit[exit_style])
        avg_ret = avg(ret_by_exit[exit_style])
        avg_gap = avg(gap_by_exit[exit_style])
        label   = EXIT_LABELS[exit_style]
        print(f"     {label:>14}  {avg_pf:>10.3f}  {avg_tr:>14.1f}  "
              f"{avg_dd:>10.2f}%  {avg_ret:>+10.2f}%  {avg_gap:>+10.3f}")

    print()
    best_exit = exit_order[0]
    best_exit_pf = avg(pf_by_exit[best_exit])
    worst_exit = exit_order[-1]
    worst_exit_pf = avg(pf_by_exit[worst_exit])
    exit_spread = best_exit_pf - worst_exit_pf

    if exit_spread > 0.15:
        print(f"  -> Strong exit-style differentiation ({exit_spread:.3f} PF spread).")
        print(f"     '{EXIT_LABELS[best_exit]}' dominates — this exit structure matches the")
        print(f"     dataset's intraday move dynamics better than the alternatives.")
        print(f"     '{EXIT_LABELS[worst_exit]}' underperforms — likely exits too early or")
        print(f"     holds too long relative to the average ORB move size.")
    elif exit_spread > 0.05:
        print(f"  -> Moderate exit-style differentiation ({exit_spread:.3f} PF spread).")
        print(f"     '{EXIT_LABELS[best_exit]}' has a mild edge but the difference is not")
        print(f"     large enough to be decisive — focus on window and stop tuning instead.")
    else:
        print(f"  -> Flat exit-style surface ({exit_spread:.3f} spread) — exit logic")
        print(f"     is not the primary driver.  The entry signal itself determines outcome.")

    # IS-OOS gap check — is there overfitting?
    top10 = ranked_all[:10]
    gaps  = [r.is_pf - r.oos_pf for r in top10]
    avg_gap = avg(gaps)
    overfit_count = sum(1 for g in gaps if g > 0.15)
    stable_count  = sum(1 for g in gaps if abs(g) < 0.05)
    print()
    print(f"  IS-OOS PF gap (top 10): avg={avg_gap:+.3f}  "
          f"overfit(>0.15)={overfit_count}/10  stable(<0.05)={stable_count}/10")

    # --- Q4: Time window and regime filter ---
    print()
    print("  4. Does the equity curve improve with window/regime filtering?")
    print()

    for win in ["morning_only", "full_day"]:
        avg_pf = avg(pf_by_win[win])
        avg_tr = avg(tr_by_win[win])
        print(f"     {win:>12}  avg OOS PF: {avg_pf:.3f}  avg OOS trades: {avg_tr:.0f}")

    print()
    for regime_on in [False, True]:
        avg_pf = avg(pf_by_regime[regime_on])
        label  = "regime ON " if regime_on else "regime OFF"
        print(f"     {label}  avg OOS PF: {avg_pf:.3f}")

    win_delta    = avg(pf_by_win.get("morning_only", [0])) - avg(pf_by_win.get("full_day", [0]))
    regime_delta = avg(pf_by_regime.get(True, [0]))        - avg(pf_by_regime.get(False, [0]))

    print()
    if abs(win_delta) > 0.05:
        better_win = "morning_only" if win_delta > 0 else "full_day"
        print(f"  -> Window matters ({abs(win_delta):.3f} PF delta): '{better_win}' is clearly better.")
        if better_win == "morning_only":
            print("     ORB setups are strongest in the first 2 hours — later entries off a")
            print("     stale opening range dilute quality and generate losing afternoon trades.")
        else:
            print("     Afternoon continuation setups add value on this dataset — the opening")
            print("     range remains a useful reference well into the afternoon session.")
    else:
        print(f"  -> Window makes little difference ({abs(win_delta):.3f} delta).")

    if abs(regime_delta) > 0.05:
        print(f"  -> Regime filter {'helps' if regime_delta > 0 else 'hurts'}"
              f" ({regime_delta:+.3f} PF delta).")
        if regime_delta > 0:
            print("     Requiring a bullish hourly regime correctly eliminates breakout entries")
            print("     on bearish-regime days where upside follow-through is weaker.")
        else:
            print("     Regime filter is over-restrictive on this dataset — it eliminates")
            print("     profitable breakout days that happen to open below the hourly SMA.")
    else:
        print(f"  -> Regime filter has minimal effect ({regime_delta:+.3f} delta).")

    # --- Final verdict ---
    print()
    print("=" * 72)
    print("  VERDICT: BREAKOUT vs VWAP MR")
    print("=" * 72)
    print()

    best_reliable = next((r for r in ranked_all if r.oos_trades >= MIN_MEANINGFUL_OOS_TRADES), None)

    if not best_reliable:
        print("  Insufficient OOS trades across all breakout configs — no reliable comparison.")
        return

    if above_1 and best_reliable.oos_pf > vwap_oos_pf + 0.05:
        top = sorted(above_1, key=_rank_key)[0]
        print(f"  BREAKOUT WINS. Best config significantly outperforms VWAP MR best.")
        print()
        print(f"  Best breakout config:")
        print(f"    exit={EXIT_LABELS[top.exit_style]}  stop={top.max_stop_pct:.2f}"
              f"  window={top.time_window}  regime={'on' if top.regime_filter else 'off'}")
        print(f"    OOS PF={_pf_str(top.oos_pf)}  OOS ret={top.oos_return:+.2f}%"
              f"  OOS DD={top.oos_dd:.2f}%  trades={top.oos_trades}")
        print()
        is_oos_gap = top.is_pf - top.oos_pf
        if abs(is_oos_gap) < 0.08:
            print("  IS-OOS gap is stable. Proceed to stage 3: breakout parameter sweep")
            print("  targeting gap_pct_min and or_range_pct_min quality filters.")
        else:
            print(f"  IS-OOS gap={is_oos_gap:+.3f} — moderate overfitting. Run on a second")
            print("  OOS window (Apr 4 - present) before committing to breakout strategy.")

    elif above_1:
        top = sorted(above_1, key=_rank_key)[0]
        print(f"  BREAKOUT WINS on PF (>{_pf_str(top.oos_pf)} > 1.0) but advantage over VWAP MR is small.")
        print()
        print(f"  Best: {EXIT_LABELS[top.exit_style]}  stop={top.max_stop_pct:.2f}"
              f"  {top.time_window}  OOS PF={_pf_str(top.oos_pf)}  trades={top.oos_trades}")
        print()
        print("  The strategy is profitable OOS — that's the important signal.")
        print("  Proceed: run a breakout parameter sweep targeting OR quality filters")
        print("  (gap_pct_min, or_range_pct_min) to improve selectivity.")

    elif best_reliable.oos_pf > vwap_oos_pf:
        print(f"  BREAKOUT EDGES OUT VWAP MR (OOS PF {_pf_str(best_reliable.oos_pf)}"
              f" vs {_pf_str(vwap_oos_pf)}) but neither reaches 1.0.")
        print()
        print("  Neither strategy is profitable at current parameters.")
        print("  Consider: ADX > 25 as an entry confirmation for breakout (require a trending")
        print("  market to take ORB trades — inverse of the MR ADX filter).")

    else:
        print(f"  VWAP MR REMAINS BETTER (breakout best OOS PF={_pf_str(best_reliable.oos_pf)}"
              f" < VWAP MR {_pf_str(vwap_oos_pf)}).")
        print()
        print("  The dataset may not be well-suited to ORB breakout as implemented.")
        print("  Possible explanations:")
        print("  - Oct 2025 - Apr 2026 was a high-IV, mean-reverting market overall;")
        print("    ORB gaps faded rather than continued.")
        print("  - 15-min bars are too coarse to capture the ORB breakout move precisely;")
        print("    a 5-min bar dataset would allow tighter entries and stops.")
        print("  - The opening range window (first 15 min) may be too narrow; a 30-min")
        print("    OR would produce wider, more meaningful breakout levels.")
        print()
        print("  Recommended next steps:")
        print("  a) Re-evaluate with a 5-min bar dataset if available.")
        print("  b) Test OR window = 30 min (2 bars) as a config option.")
        print("  c) If neither works, pivot to Option A (VWAP MR with ATR-multiple exit)")
        print("     or the wick-fade strategy from the original research document.")

    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    grid = list(itertools.product(EXIT_STYLES, MAX_STOP_PCTS, TIME_WINDOWS, REGIME_FILTERS))
    print("=" * 72)
    print("  BREAKOUT STRATEGY EVALUATION")
    print(f"  Dataset : AAPL AMD AMZN GOOGL JPM KO META MSFT NVDA TSLA XOM")
    print(f"  Period  : Oct 2025 - Apr 2026  (15-min bars, SIP)")
    print(f"  Split   : IS {IS_START} to {IS_END}  |  OOS {OOS_START} to {OOS_END}")
    print(f"  Grid    : {len(EXIT_STYLES)} exits x {len(MAX_STOP_PCTS)} stops"
          f" x {len(TIME_WINDOWS)} windows x {len(REGIME_FILTERS)} regime = {len(grid)} combos")
    print(f"  Capital : $10,000 total  |  $1,000 per position")
    print(f"  Costs   : $0.01 commission + $0.05/share slippage per side")
    print("=" * 72)

    results, vwap_is, vwap_oos, sma_is, sma_oos = run_eval()
    print_results(results, vwap_is, vwap_oos, sma_is, sma_oos)
    print_analysis(results, vwap_is, vwap_oos, sma_is, sma_oos)


if __name__ == "__main__":
    main()

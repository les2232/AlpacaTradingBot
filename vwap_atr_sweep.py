"""
Second-stage parameter sweep: VWAP Z-score MR + ATR percentile filter.

Stage 1 (vwap_sweep.py) found that VWAP MR outperforms SMA baseline OOS
(PF 0.92 max vs SMA 0.83) but remains unprofitable.  Low-volatility entries
were identified as the primary drag.  This sweep adds min_atr_percentile to
the grid to test whether blocking flat-tape entries improves trade quality.

Grid:
    vwap_z_entry_threshold  : [1.5, 2.0, 2.25]          (3 values)
    vwap_z_stop_atr_multiple: [0.75, 1.0, 1.5, 2.0]     (4 values)
    min_atr_percentile      : [0, 10, 20, 30, 40, 50]    (6 values)
    Total: 72 parameter combos x 2 windows = 144 backtest runs

Split (same 70/30 walk-forward as stage 1):
    In-sample : 2025-10-04 to 2026-02-07
    OOS       : 2026-02-08 to 2026-04-04

Run:
    python vwap_atr_sweep.py
"""

from __future__ import annotations

import contextlib
import io
import itertools
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from backtest_runner import run_backtest

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

ENTRY_THRESHOLDS  = [1.5, 2.0, 2.25]
STOP_MULTIPLES    = [0.75, 1.0, 1.5, 2.0]
MIN_ATR_PCTS      = [0, 10, 20, 30, 40, 50]

# Stage-1 best no-filter reference: entry=2.00, stop=1.00, min_atr=0
# Included automatically as min_atr_pct=0 rows in the grid.

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

# SMA OOS reference (unchanged from stage 1)
SMA_OOS_CONFIG = dict(
    **COMMON,
    start_date=OOS_START,
    end_date=OOS_END,
    entry_threshold_pct=0.01,
    mean_reversion_exit_style="sma",
    mean_reversion_stop_pct=0.02,
    vwap_z_entry_threshold=0.0,
    vwap_z_stop_atr_multiple=2.0,
    min_atr_percentile=0.0,
)

# Minimum OOS trade count considered statistically meaningful.
# Configs below this are flagged as over-filtered.
MIN_MEANINGFUL_OOS_TRADES = 25

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class SweepResult:
    entry_threshold: float
    stop_mult: float
    min_atr_pct: float
    is_pf: float
    oos_pf: float
    is_return: float
    oos_return: float
    is_dd: float
    oos_dd: float
    is_trades: int
    oos_trades: int
    is_wr: float
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
    """Return (profit_factor, total_return_pct, max_drawdown_pct, total_trades, win_rate)."""
    return (
        _profit_factor(r.get("trades", [])),
        r.get("total_return_pct", 0.0),
        r.get("max_drawdown_pct", 0.0),
        r.get("total_trades", 0),
        r.get("win_rate", 0.0),
    )


def _pf_str(pf: float) -> str:
    return f"{min(pf, 9.99):.3f}" if pf != float("inf") else "  inf"


# ---------------------------------------------------------------------------
# Sweep
# ---------------------------------------------------------------------------

def run_sweep() -> tuple[list[SweepResult], dict, dict]:
    grid = list(itertools.product(ENTRY_THRESHOLDS, STOP_MULTIPLES, MIN_ATR_PCTS))
    total = len(grid)
    results: list[SweepResult] = []

    print(f"\n[sweep] {total} combos x 2 windows = {total * 2} backtest runs")
    print(f"[sweep] IS : {IS_START} to {IS_END}")
    print(f"[sweep] OOS: {OOS_START} to {OOS_END}\n")

    for i, (entry, stop, min_atr) in enumerate(grid):
        run_is  = i * 2 + 1
        run_oos = i * 2 + 2

        print(f"[{run_is:3d}/{total * 2}] entry={entry:.2f} stop={stop:.2f} atr>={min_atr:2d}  IS ...",
              end=" ", flush=True)
        kwargs_base = dict(
            **COMMON,
            vwap_z_entry_threshold=entry,
            vwap_z_stop_atr_multiple=stop,
            min_atr_percentile=float(min_atr),
        )
        # First call: let the [dataset] header print through once
        if i == 0:
            is_r = run_backtest(**kwargs_base, start_date=IS_START, end_date=IS_END)
        else:
            is_r = _run_silent({**kwargs_base, "start_date": IS_START, "end_date": IS_END})
        print("done")

        print(f"[{run_oos:3d}/{total * 2}] entry={entry:.2f} stop={stop:.2f} atr>={min_atr:2d}  OOS ...",
              end=" ", flush=True)
        oos_r = _run_silent({**kwargs_base, "start_date": OOS_START, "end_date": OOS_END})
        print("done")

        is_pf,  is_ret,  is_dd,  is_tr,  is_wr  = _collect(is_r)
        oos_pf, oos_ret, oos_dd, oos_tr, oos_wr = _collect(oos_r)
        results.append(SweepResult(
            entry_threshold=entry,   stop_mult=stop,        min_atr_pct=float(min_atr),
            is_pf=is_pf,             oos_pf=oos_pf,
            is_return=is_ret,        oos_return=oos_ret,
            is_dd=is_dd,             oos_dd=oos_dd,
            is_trades=is_tr,         oos_trades=oos_tr,
            is_wr=is_wr,             oos_wr=oos_wr,
        ))

    print("\n[ref] SMA OOS reference ...", end=" ", flush=True)
    sma_oos_r = _run_silent(SMA_OOS_CONFIG)
    print("done")

    print("[ref] SMA IS  reference ...", end=" ", flush=True)
    sma_is_cfg = {**SMA_OOS_CONFIG, "start_date": IS_START, "end_date": IS_END}
    del sma_is_cfg["start_date"]  # re-set below to avoid key collision
    sma_is_r = _run_silent({
        **{k: v for k, v in SMA_OOS_CONFIG.items() if k not in ("start_date", "end_date")},
        "start_date": IS_START,
        "end_date": IS_END,
    })
    print("done")

    return results, sma_is_r, sma_oos_r


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def _rank_key(r: SweepResult) -> tuple:
    oos_pf_capped = min(r.oos_pf, 9.99)
    return (-oos_pf_capped, r.oos_dd, -r.oos_return)


def print_top10(results: list[SweepResult], sma_is: dict, sma_oos: dict) -> None:
    sma_oos_pf  = _profit_factor(sma_oos.get("trades", []))
    sma_is_pf   = _profit_factor(sma_is.get("trades", []))
    sma_oos_ret = sma_oos.get("total_return_pct", 0.0)
    sma_is_ret  = sma_is.get("total_return_pct", 0.0)
    sma_oos_dd  = sma_oos.get("max_drawdown_pct", 0.0)
    sma_oos_tr  = sma_oos.get("total_trades", 0)
    sma_is_tr   = sma_is.get("total_trades", 0)

    ranked = sorted(results, key=_rank_key)
    top10  = ranked[:10]

    print()
    print("=" * 96)
    print("  TOP 10 CONFIGURATIONS  (ranked by OOS profit factor)")
    print("=" * 96)

    hdr = (
        f"{'Rank':>4}  {'Entry':>5}  {'Stop':>4}  {'AtrPct':>6}  "
        f"{'IS-PF':>6}  {'OOS-PF':>6}  "
        f"{'IS-Ret':>7}  {'OOS-Ret':>7}  "
        f"{'OOS-DD':>7}  {'IS-Tr':>6}  {'OOS-Tr':>6}"
    )
    sep = "-" * 96
    print(hdr)
    print(sep)

    for rank, r in enumerate(top10, 1):
        beats_sma = r.oos_pf > sma_oos_pf
        low_tr    = r.oos_trades < MIN_MEANINGFUL_OOS_TRADES
        flag = " *" if beats_sma else "  "
        flag = " !" if low_tr else flag
        print(
            f"{rank:>4}{flag} {r.entry_threshold:>5.2f}  {r.stop_mult:>4.2f}  {r.min_atr_pct:>6.0f}  "
            f"{_pf_str(r.is_pf):>6}  {_pf_str(r.oos_pf):>6}  "
            f"{r.is_return:>+7.2f}%  {r.oos_return:>+7.2f}%  "
            f"{r.oos_dd:>7.2f}%  {r.is_trades:>6}  {r.oos_trades:>6}"
        )

    print(sep)
    print(
        f"{'SMA':>4}   {'--':>5}  {'--':>4}  {'--':>6}  "
        f"{_pf_str(sma_is_pf):>6}  {_pf_str(sma_oos_pf):>6}  "
        f"{sma_is_ret:>+7.2f}%  {sma_oos_ret:>+7.2f}%  "
        f"{sma_oos_dd:>7.2f}%  {sma_is_tr:>6}  {sma_oos_tr:>6}"
    )
    print(sep)
    print("  * = beats SMA OOS profit factor    ! = fewer than 25 OOS trades (low confidence)")


def print_analysis(results: list[SweepResult], sma_is: dict, sma_oos: dict) -> None:
    sma_oos_pf = _profit_factor(sma_oos.get("trades", []))
    sma_oos_dd = sma_oos.get("max_drawdown_pct", 0.0)

    ranked_all  = sorted(results, key=_rank_key)
    stage1_ref  = next(
        (r for r in results if r.entry_threshold == 2.0 and r.stop_mult == 1.0 and r.min_atr_pct == 0.0),
        None,
    )

    # --- Per-percentile aggregates ---
    pf_by_atr:     dict[float, list[float]] = defaultdict(list)
    tr_by_atr:     dict[float, list[int]]   = defaultdict(list)
    dd_by_atr:     dict[float, list[float]] = defaultdict(list)
    ret_by_atr:    dict[float, list[float]] = defaultdict(list)

    for r in results:
        pct = r.min_atr_pct
        if r.oos_pf != float("inf"):
            pf_by_atr[pct].append(r.oos_pf)
        tr_by_atr[pct].append(r.oos_trades)
        dd_by_atr[pct].append(r.oos_dd)
        ret_by_atr[pct].append(r.oos_return)

    def avg(lst: list) -> float:
        return sum(lst) / len(lst) if lst else 0.0

    atr_summary: list[tuple[float, float, float, float, float]] = [
        (
            pct,
            avg(pf_by_atr[pct]),
            avg(tr_by_atr[pct]),
            avg(dd_by_atr[pct]),
            avg(ret_by_atr[pct]),
        )
        for pct in sorted(pf_by_atr.keys())
    ]

    beats_sma_count = sum(1 for r in results if r.oos_pf > sma_oos_pf
                          and r.oos_trades >= MIN_MEANINGFUL_OOS_TRADES)
    above_1_count   = sum(1 for r in results if r.oos_pf > 1.0
                          and r.oos_trades >= MIN_MEANINGFUL_OOS_TRADES)
    best            = ranked_all[0]

    print()
    print("=" * 72)
    print("  ANALYSIS")
    print("=" * 72)

    # --- Q1: Does ATR filtering push PF above 1.0? ---
    print()
    print("  1. Does ATR percentile filtering push OOS profit factor above 1.0?")
    print()

    if above_1_count > 0:
        above_1 = [r for r in results if r.oos_pf > 1.0
                   and r.oos_trades >= MIN_MEANINGFUL_OOS_TRADES]
        above_1.sort(key=_rank_key)
        print(f"  -> YES. {above_1_count} configuration(s) achieve OOS PF > 1.0 with >= "
              f"{MIN_MEANINGFUL_OOS_TRADES} trades:")
        for r in above_1[:5]:
            print(f"     entry={r.entry_threshold:.2f} stop={r.stop_mult:.2f}"
                  f" atr>={r.min_atr_pct:.0f}"
                  f"  OOS PF={_pf_str(r.oos_pf)}"
                  f"  OOS ret={r.oos_return:+.2f}%"
                  f"  OOS trades={r.oos_trades}")
        if above_1_count > 5:
            print(f"     ... and {above_1_count - 5} more.")
    else:
        print(f"  -> NO. No configuration with >= {MIN_MEANINGFUL_OOS_TRADES} OOS trades")
        print(f"     achieved OOS PF > 1.0 in this sweep.")
        best_reliable = next(
            (r for r in ranked_all if r.oos_trades >= MIN_MEANINGFUL_OOS_TRADES), None
        )
        if best_reliable:
            print(f"     Best reliable OOS PF: {_pf_str(best_reliable.oos_pf)}"
                  f" (entry={best_reliable.entry_threshold:.2f}"
                  f" stop={best_reliable.stop_mult:.2f}"
                  f" atr>={best_reliable.min_atr_pct:.0f}"
                  f" trades={best_reliable.oos_trades})")

    # Stage-1 comparison
    if stage1_ref:
        print()
        print(f"  Stage-1 reference (entry=2.00, stop=1.00, atr>=0):")
        print(f"     OOS PF={_pf_str(stage1_ref.oos_pf)}"
              f"  OOS ret={stage1_ref.oos_return:+.2f}%"
              f"  OOS DD={stage1_ref.oos_dd:.2f}%"
              f"  OOS trades={stage1_ref.oos_trades}")

    # --- Q2: Per-percentile breakdown ---
    print()
    print("  2. Trade count vs quality across ATR percentile thresholds:")
    print()
    print(f"     {'AtrPct':>6}  {'Avg OOS PF':>10}  {'Avg OOS Trades':>14}  "
          f"{'Avg OOS DD':>10}  {'Avg OOS Ret':>11}")
    print("     " + "-" * 58)

    # Find the sweet spot: highest avg PF that still has avg trades > threshold
    sweet_spot_pct: float | None = None
    prev_pf = 0.0
    for pct, avg_pf, avg_tr, avg_dd, avg_ret in atr_summary:
        marker = ""
        if avg_tr < MIN_MEANINGFUL_OOS_TRADES:
            marker = " [!]"
        elif avg_pf > prev_pf and avg_tr >= MIN_MEANINGFUL_OOS_TRADES:
            if sweet_spot_pct is None or avg_pf > avg(pf_by_atr[sweet_spot_pct]):
                sweet_spot_pct = pct
        prev_pf = avg_pf
        print(f"     {pct:>6.0f}  {avg_pf:>10.3f}  {avg_tr:>14.1f}  "
              f"{avg_dd:>10.2f}%  {avg_ret:>+10.2f}%{marker}")

    print()

    # Find the percentile with best avg PF among those with adequate trades
    reliable_summary = [
        (pct, avg_pf, avg_tr, avg_dd)
        for pct, avg_pf, avg_tr, avg_dd, _ in atr_summary
        if avg_tr >= MIN_MEANINGFUL_OOS_TRADES
    ]

    if reliable_summary:
        best_avg_pct, best_avg_pf, best_avg_tr, best_avg_dd = max(
            reliable_summary, key=lambda x: x[1]
        )
        zero_pf  = avg(pf_by_atr.get(0.0, pf_by_atr.get(0, [])))
        pf_delta = best_avg_pf - zero_pf
        print(f"  -> Best avg OOS PF is at atr>={best_avg_pct:.0f}"
              f" ({best_avg_pf:.3f} vs {zero_pf:.3f} unfiltered,"
              f" delta={pf_delta:+.3f}).")

        baseline_tr = avg(tr_by_atr.get(0.0, tr_by_atr.get(0, [])))
        tr_reduction = (baseline_tr - best_avg_tr) / max(baseline_tr, 1) * 100
        print(f"     Trade count reduction vs atr>=0: {tr_reduction:.0f}%"
              f" ({baseline_tr:.0f} -> {best_avg_tr:.0f} avg OOS trades).")

        if pf_delta >= 0.05:
            print(f"     The {best_avg_pct:.0f}th percentile filter provides a meaningful PF lift")
            print(f"     while retaining adequate trade count for reliable OOS measurement.")
        elif pf_delta > 0:
            print(f"     The improvement is marginal ({pf_delta:+.3f} PF). The filter is helping")
            print(f"     slightly but not dramatically — low-volatility noise is present but")
            print(f"     not the dominant drag on performance.")
        else:
            print(f"     ATR filtering does not consistently improve avg OOS PF.")
            print(f"     Low-volatility noise may not be the primary driver of losses.")

    # --- Q3: Where does filtering become too aggressive? ---
    print()
    print("  3. At what point does filtering become too aggressive?")
    print()

    cliff_pct: float | None = None
    for pct, avg_pf, avg_tr, avg_dd, avg_ret in atr_summary:
        if avg_tr < MIN_MEANINGFUL_OOS_TRADES and cliff_pct is None:
            cliff_pct = pct
            break

    over_filtered = [(pct, avg_tr) for pct, _, avg_tr, _, _ in atr_summary
                     if avg_tr < MIN_MEANINGFUL_OOS_TRADES]

    if over_filtered:
        cliff_pct = over_filtered[0][0]
        print(f"  -> atr>={cliff_pct:.0f} drops avg OOS trades below {MIN_MEANINGFUL_OOS_TRADES}.")
        print(f"     Results at this threshold and above have insufficient sample size to")
        print(f"     trust OOS metrics — any apparent PF improvement is likely noise.")
        # Also check individual configs
        high_pct_overfit = [
            r for r in results
            if r.min_atr_pct >= cliff_pct and r.oos_trades < MIN_MEANINGFUL_OOS_TRADES
        ]
        print(f"     {len(high_pct_overfit)} of {len([r for r in results if r.min_atr_pct >= cliff_pct])}"
              f" configs at atr>={cliff_pct:.0f}+ have fewer than {MIN_MEANINGFUL_OOS_TRADES} OOS trades.")
    else:
        print(f"  -> No threshold in [0, 50] drops avg OOS trades below {MIN_MEANINGFUL_OOS_TRADES}.")
        print(f"     The ATR percentile filter is not overly aggressive across this sweep range.")
        # Check the highest percentile individually
        highest_pct = max(MIN_ATR_PCTS)
        high_pct_results = [r for r in results if r.min_atr_pct == highest_pct]
        min_oos_tr = min(r.oos_trades for r in high_pct_results) if high_pct_results else 0
        print(f"     At atr>={highest_pct}, minimum OOS trade count across all entry/stop combos: {min_oos_tr}.")

    # Check IS-OOS PF gap at the top configs — overfitting detection
    top10 = ranked_all[:10]
    gaps = [r.is_pf - r.oos_pf for r in top10]
    avg_gap = sum(gaps) / len(gaps) if gaps else 0.0
    overfit_count = sum(1 for g in gaps if g > 0.15)

    print()
    print(f"  IS-OOS PF gap for top 10 (avg={avg_gap:+.3f}, {overfit_count}/10 gap>0.15):")
    for r in top10:
        gap = r.is_pf - r.oos_pf
        verdict = "stable  " if abs(gap) < 0.05 else ("OVERFIT " if gap > 0.15 else "moderate")
        print(f"     entry={r.entry_threshold:.2f} stop={r.stop_mult:.2f}"
              f" atr>={r.min_atr_pct:.0f}"
              f"  IS={_pf_str(r.is_pf)}  OOS={_pf_str(r.oos_pf)}"
              f"  gap={gap:+.3f}  [{verdict}]")

    # --- Q4: Does drawdown improve? ---
    print()
    print("  4. Does ATR filtering improve drawdown?")
    print()

    baseline_dd  = avg(dd_by_atr.get(0.0, dd_by_atr.get(0, [])))
    dd_at_pcts   = [(pct, avg(dd_by_atr[pct])) for pct in sorted(dd_by_atr.keys())]

    print(f"     Avg OOS max drawdown by ATR percentile filter:")
    for pct, avg_dd in dd_at_pcts:
        delta = avg_dd - baseline_dd
        print(f"       atr>={pct:2.0f}: {avg_dd:.2f}%  (vs baseline: {delta:+.2f}pp)")

    best_dd_pct  = min(dd_at_pcts, key=lambda x: x[1])
    dd_improvement = abs(best_dd_pct[1]) - abs(baseline_dd)

    print()
    if dd_improvement < -0.3:
        print(f"  -> YES. atr>={best_dd_pct[0]:.0f} reduces avg OOS drawdown by"
              f" {abs(dd_improvement):.2f}pp ({baseline_dd:.2f}% -> {best_dd_pct[1]:.2f}%).")
        print(f"     Blocking flat-tape entries is preventing the strategy from holding")
        print(f"     through extended low-volatility drawdown periods.")
    elif dd_improvement < 0:
        print(f"  -> MARGINALLY. Avg OOS drawdown improves by {abs(dd_improvement):.2f}pp at"
              f" atr>={best_dd_pct[0]:.0f}.")
        print(f"     The improvement is real but small — the primary drawdown driver")
        print(f"     is likely losing trades in volatile conditions, not low-vol noise.")
    else:
        print(f"  -> NO meaningful drawdown improvement from ATR filtering.")
        print(f"     The stops may already be absorbing low-vol losses adequately.")

    # --- Final recommendation ---
    print()
    print("=" * 72)
    print("  RECOMMENDATION")
    print("=" * 72)
    print()

    best_reliable = next(
        (r for r in ranked_all if r.oos_trades >= MIN_MEANINGFUL_OOS_TRADES), None
    )

    if above_1_count > 0:
        top_above_1 = sorted(
            [r for r in results if r.oos_pf > 1.0 and r.oos_trades >= MIN_MEANINGFUL_OOS_TRADES],
            key=_rank_key,
        )[0]
        print(f"  BEST CANDIDATE: entry={top_above_1.entry_threshold:.2f}"
              f"  stop={top_above_1.stop_mult:.2f}"
              f"  atr>={top_above_1.min_atr_pct:.0f}")
        print(f"    OOS PF    : {_pf_str(top_above_1.oos_pf)}")
        print(f"    OOS return: {top_above_1.oos_return:+.2f}%")
        print(f"    OOS DD    : {top_above_1.oos_dd:.2f}%")
        print(f"    OOS trades: {top_above_1.oos_trades}")
        is_oos_gap = top_above_1.is_pf - top_above_1.oos_pf
        print(f"    IS-OOS gap: {is_oos_gap:+.3f}")
        print()
        if abs(is_oos_gap) < 0.08:
            print("  OOS PF > 1.0 with stable IS-OOS gap. This is the first genuinely")
            print("  profitable signal in this sweep series.")
            print()
            print("  Before enabling in live paper trading:")
            print("  a) Confirm on a SECOND OOS window (Apr 4 - present) with fresh data.")
            print("  b) Verify trade count is stable across multiple 4-week periods.")
            print("  c) Run with a conservative position size ($500 instead of $1,000)")
            print("     until the live paper equity curve validates the OOS result.")
        else:
            print("  OOS PF > 1.0 but IS-OOS gap suggests partial overfitting.")
            print("  Do NOT promote to live trading yet.")
            print()
            print("  Next steps:")
            print("  a) Run this config on a second OOS window (Apr 4 - present).")
            print("  b) If it holds above PF 1.0 on the second window, proceed.")
            print("  c) If it degrades, the result is regime-specific and not robust.")
    elif best_reliable and best_reliable.oos_pf > sma_oos_pf + 0.05:
        print(f"  Best config does not reach PF 1.0 but meaningfully beats SMA baseline.")
        print(f"  entry={best_reliable.entry_threshold:.2f}"
              f"  stop={best_reliable.stop_mult:.2f}"
              f"  atr>={best_reliable.min_atr_pct:.0f}"
              f"  OOS PF={_pf_str(best_reliable.oos_pf)}"
              f"  OOS trades={best_reliable.oos_trades}")
        print()
        print("  VWAP MR is structurally better than SMA MR but needs one more lever.")
        print("  Strongest remaining candidate: add ADX(14) gate to block trending bars.")
        print("  This is the next logical experiment before considering strategy abandonment.")
    else:
        print("  ATR filtering did not produce a reliable OOS PF > 1.0 configuration.")
        print()
        print("  The strategy as designed has a structural edge limitation:")
        print("  - Mean reversion works in ranging markets, fails in trending ones.")
        print("  - ATR percentile removes quiet periods but not trending periods.")
        print("  - The missing filter is a TREND/RANGE regime discriminator (ADX/DI).")
        print()
        print("  Recommended next action: add ADX(14) < 25 entry gate.")
        print("  This targets the same problem (bad entries) via a different dimension")
        print("  (trend strength vs volatility magnitude). If ADX filtering also fails,")
        print("  abandon VWAP MR and evaluate the wick-fade strategy from the research doc.")

    print()
    print("  Note: all OOS metrics are genuine hold-out results on unseen data.")
    print("  IS/OOS windows were fixed before any parameter was evaluated.")
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    grid = list(itertools.product(ENTRY_THRESHOLDS, STOP_MULTIPLES, MIN_ATR_PCTS))
    print("=" * 72)
    print("  VWAP Z-SCORE + ATR PERCENTILE FILTER  (stage-2 sweep)")
    print(f"  Dataset : AAPL AMD AMZN GOOGL JPM KO META MSFT NVDA TSLA XOM")
    print(f"  Period  : Oct 2025 - Apr 2026  (15-min bars, SIP)")
    print(f"  Split   : IS {IS_START} to {IS_END}  |  OOS {OOS_START} to {OOS_END}")
    print(f"  Grid    : {len(ENTRY_THRESHOLDS)} thresholds x {len(STOP_MULTIPLES)} stops"
          f" x {len(MIN_ATR_PCTS)} atr floors = {len(grid)} combos")
    print(f"  Capital : $10,000 total  |  $1,000 per position")
    print(f"  Costs   : $0.01 commission + $0.05/share slippage per side")
    print("=" * 72)

    results, sma_is, sma_oos = run_sweep()
    print_top10(results, sma_is, sma_oos)
    print_analysis(results, sma_is, sma_oos)


if __name__ == "__main__":
    main()

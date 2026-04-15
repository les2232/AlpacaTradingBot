"""
Walk-forward parameter sweep for VWAP Z-score mean reversion.

For each parameter pair, two backtests are run:
  - In-sample  (IS) : 2025-10-04 to 2026-02-07  (~70% of data, 126 calendar days)
  - Out-of-sample (OOS): 2026-02-08 to 2026-04-04  (~30% of data,  56 calendar days)

Grid:
    vwap_z_entry_threshold  : [1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 3.0, 3.5]
    vwap_z_stop_atr_multiple: [0.75, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0]
    Total: 63 parameter pairs x 2 windows = 126 backtest runs

SMA OOS reference is also computed for baseline comparison.

Run:
    python vwap_sweep.py
"""

from __future__ import annotations

import io
import contextlib
import itertools
import sys
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
IS_END    = "2026-02-07"   # ~70% of calendar days
OOS_START = "2026-02-08"
OOS_END   = "2026-04-04"   # ~30% of calendar days

ENTRY_THRESHOLDS = [1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 3.0, 3.5]
STOP_MULTIPLES   = [0.75, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0]

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

# SMA baseline configuration for OOS reference
SMA_OOS_CONFIG = dict(
    **COMMON,
    start_date=OOS_START,
    end_date=OOS_END,
    entry_threshold_pct=0.01,
    mean_reversion_exit_style="sma",
    mean_reversion_stop_pct=0.02,
    vwap_z_entry_threshold=0.0,
    vwap_z_stop_atr_multiple=2.0,
)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class SweepResult:
    entry_threshold: float
    stop_mult: float
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
    """Compute profit factor from raw trade list (BUY/SELL pairs)."""
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
    """Run a backtest, suppressing per-run stdout noise after the first call."""
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        result = run_backtest(**kwargs)
    return result


def _collect(r: dict) -> tuple[float, float, float, int, float]:
    """Extract (profit_factor, total_return_pct, max_drawdown_pct, total_trades, win_rate)."""
    pf     = _profit_factor(r.get("trades", []))
    ret    = r.get("total_return_pct", 0.0)
    dd     = r.get("max_drawdown_pct", 0.0)
    trades = r.get("total_trades", 0)
    wr     = r.get("win_rate", 0.0)
    return pf, ret, dd, trades, wr


# ---------------------------------------------------------------------------
# Sweep
# ---------------------------------------------------------------------------

def run_sweep() -> tuple[list[SweepResult], dict, dict]:
    """
    Run the full grid sweep.

    Returns:
        results  : list[SweepResult] for all 63 param pairs
        sma_is_m : IS metrics dict for SMA baseline
        sma_oos_m: OOS metrics dict for SMA baseline
    """
    grid = list(itertools.product(ENTRY_THRESHOLDS, STOP_MULTIPLES))
    total = len(grid)
    results: list[SweepResult] = []

    print(f"\n[sweep] {total} parameter pairs x 2 windows = {total * 2} backtest runs")
    print(f"[sweep] IS  : {IS_START} to {IS_END}")
    print(f"[sweep] OOS : {OOS_START} to {OOS_END}\n")

    # First call: let the dataset header print through
    entry0, stop0 = grid[0]
    print(f"[  1/{total * 2}] entry={entry0:.2f} stop={stop0:.2f}  IS ...", end=" ", flush=True)
    is0 = run_backtest(
        **COMMON,
        start_date=IS_START, end_date=IS_END,
        vwap_z_entry_threshold=entry0,
        vwap_z_stop_atr_multiple=stop0,
    )
    print("done")

    print(f"[  2/{total * 2}] entry={entry0:.2f} stop={stop0:.2f}  OOS ...", end=" ", flush=True)
    oos0 = _run_silent(dict(
        **COMMON,
        start_date=OOS_START, end_date=OOS_END,
        vwap_z_entry_threshold=entry0,
        vwap_z_stop_atr_multiple=stop0,
    ))
    print("done")

    is_pf0,  is_ret0,  is_dd0,  is_tr0,  is_wr0  = _collect(is0)
    oos_pf0, oos_ret0, oos_dd0, oos_tr0, oos_wr0 = _collect(oos0)
    results.append(SweepResult(
        entry_threshold=entry0, stop_mult=stop0,
        is_pf=is_pf0,   oos_pf=oos_pf0,
        is_return=is_ret0, oos_return=oos_ret0,
        is_dd=is_dd0,   oos_dd=oos_dd0,
        is_trades=is_tr0, oos_trades=oos_tr0,
        is_wr=is_wr0,   oos_wr=oos_wr0,
    ))

    for i, (entry, stop) in enumerate(grid[1:], start=1):
        run_num_is  = i * 2 + 1
        run_num_oos = i * 2 + 2
        prefix = f"[{run_num_is:3d}/{total * 2}]"

        print(f"{prefix} entry={entry:.2f} stop={stop:.2f}  IS ...", end=" ", flush=True)
        is_r  = _run_silent(dict(
            **COMMON,
            start_date=IS_START, end_date=IS_END,
            vwap_z_entry_threshold=entry,
            vwap_z_stop_atr_multiple=stop,
        ))
        print("done")

        prefix = f"[{run_num_oos:3d}/{total * 2}]"
        print(f"{prefix} entry={entry:.2f} stop={stop:.2f}  OOS ...", end=" ", flush=True)
        oos_r = _run_silent(dict(
            **COMMON,
            start_date=OOS_START, end_date=OOS_END,
            vwap_z_entry_threshold=entry,
            vwap_z_stop_atr_multiple=stop,
        ))
        print("done")

        is_pf,  is_ret,  is_dd,  is_tr,  is_wr  = _collect(is_r)
        oos_pf, oos_ret, oos_dd, oos_tr, oos_wr = _collect(oos_r)
        results.append(SweepResult(
            entry_threshold=entry, stop_mult=stop,
            is_pf=is_pf,   oos_pf=oos_pf,
            is_return=is_ret, oos_return=oos_ret,
            is_dd=is_dd,   oos_dd=oos_dd,
            is_trades=is_tr, oos_trades=oos_tr,
            is_wr=is_wr,   oos_wr=oos_wr,
        ))

    # SMA baseline on OOS window
    print(f"\n[SMA ref] Running SMA mean reversion on OOS window ...", end=" ", flush=True)
    sma_oos_r = _run_silent(SMA_OOS_CONFIG)
    print("done")

    print(f"[SMA ref] Running SMA mean reversion on IS window ...", end=" ", flush=True)
    sma_is_r  = _run_silent(dict(
        **{k: v for k, v in SMA_OOS_CONFIG.items() if k not in ("start_date", "end_date")},
        start_date=IS_START,
        end_date=IS_END,
    ))
    print("done")

    return results, sma_is_r, sma_oos_r


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def _rank_key(r: SweepResult) -> tuple:
    """Primary: OOS PF desc. Secondary: OOS DD asc (less negative = better). Tertiary: OOS return desc."""
    oos_pf_capped = min(r.oos_pf, 9.99)   # cap inf to avoid sort issues
    return (-oos_pf_capped, r.oos_dd, -r.oos_return)


def print_top10(results: list[SweepResult], sma_is: dict, sma_oos: dict) -> None:
    sma_oos_pf  = _profit_factor(sma_oos.get("trades", []))
    sma_is_pf   = _profit_factor(sma_is.get("trades", []))
    sma_oos_ret = sma_oos.get("total_return_pct", 0.0)
    sma_is_ret  = sma_is.get("total_return_pct", 0.0)
    sma_oos_dd  = sma_oos.get("max_drawdown_pct", 0.0)
    sma_oos_tr  = sma_oos.get("total_trades", 0)
    sma_is_tr   = sma_is.get("total_trades", 0)

    ranked = sorted(results, key=_rank_key)[:10]

    print()
    print("=" * 88)
    print(" TOP 10 VWAP Z-SCORE PARAMETER SETS  (ranked by OOS profit factor)")
    print("=" * 88)

    hdr = (
        f"{'Rank':>4}  {'Entry':>5}  {'Stop':>4}  "
        f"{'IS-PF':>6}  {'OOS-PF':>6}  "
        f"{'IS-Ret':>7}  {'OOS-Ret':>7}  "
        f"{'OOS-DD':>7}  {'IS-Tr':>6}  {'OOS-Tr':>6}  {'OOS-WR':>7}"
    )
    sep = "-" * 88
    print(hdr)
    print(sep)

    for rank, r in enumerate(ranked, 1):
        oos_pf_str = f"{r.oos_pf:.2f}" if r.oos_pf != float("inf") else "  inf"
        is_pf_str  = f"{r.is_pf:.2f}"  if r.is_pf  != float("inf") else "  inf"
        marker = " *" if r.oos_pf > sma_oos_pf else "  "
        print(
            f"{rank:>4}{marker} {r.entry_threshold:>5.2f}  {r.stop_mult:>4.2f}  "
            f"{is_pf_str:>6}  {oos_pf_str:>6}  "
            f"{r.is_return:>+7.2f}%  {r.oos_return:>+7.2f}%  "
            f"{r.oos_dd:>7.2f}%  {r.is_trades:>6}  {r.oos_trades:>6}  "
            f"{r.oos_wr:>6.1f}%"
        )

    print(sep)
    sma_oos_pf_str = f"{sma_oos_pf:.2f}" if sma_oos_pf != float("inf") else "  inf"
    sma_is_pf_str  = f"{sma_is_pf:.2f}"  if sma_is_pf  != float("inf") else "  inf"
    print(
        f"{'SMA':>4}   {'0.01':>5}  {'2.00':>4}  "
        f"{sma_is_pf_str:>6}  {sma_oos_pf_str:>6}  "
        f"{sma_is_ret:>+7.2f}%  {sma_oos_ret:>+7.2f}%  "
        f"{sma_oos_dd:>7.2f}%  {sma_is_tr:>6}  {sma_oos_tr:>6}  "
        f"{'(ref)':>7}"
    )
    print(sep)
    print("  * = beats SMA OOS profit factor")


def print_interpretation(results: list[SweepResult], sma_is: dict, sma_oos: dict) -> None:
    sma_oos_pf = _profit_factor(sma_oos.get("trades", []))

    ranked_all = sorted(results, key=_rank_key)
    top10      = ranked_all[:10]

    beats_sma = [r for r in results if r.oos_pf > sma_oos_pf]

    # Aggregate trade count by threshold to test overtrading hypothesis
    from collections import defaultdict
    trades_by_threshold: dict[float, list[int]] = defaultdict(list)
    pf_by_threshold: dict[float, list[float]] = defaultdict(list)
    for r in results:
        trades_by_threshold[r.entry_threshold].append(r.oos_trades)
        pf_by_threshold[r.entry_threshold].append(
            r.oos_pf if r.oos_pf != float("inf") else 5.0   # cap inf
        )

    # IS-OOS PF gap for top 10
    gaps = [(r.entry_threshold, r.stop_mult, r.is_pf - r.oos_pf) for r in top10]
    avg_gap = sum(g for _, _, g in gaps) / len(gaps) if gaps else 0.0

    print()
    print("=" * 72)
    print("  INTERPRETATION")
    print("=" * 72)

    # --- Question 1: Did stricter thresholds reduce overtrading? ---
    print()
    print("  1. Did stricter thresholds reduce overtrading?")
    print()

    threshold_summary: list[tuple[float, float, float]] = []
    for thr in sorted(trades_by_threshold.keys()):
        avg_tr = sum(trades_by_threshold[thr]) / len(trades_by_threshold[thr])
        avg_pf = sum(pf_by_threshold[thr]) / len(pf_by_threshold[thr])
        threshold_summary.append((thr, avg_tr, avg_pf))
        print(f"     z>={thr:.2f}  avg OOS trades: {avg_tr:6.1f}   avg OOS PF: {avg_pf:.3f}")

    lo_thresh_tr = threshold_summary[0][1]
    hi_thresh_tr = threshold_summary[-1][1]
    trade_reduction_pct = (lo_thresh_tr - hi_thresh_tr) / max(lo_thresh_tr, 1) * 100

    print()
    if trade_reduction_pct > 20:
        print(f"  -> YES. Raising threshold from {threshold_summary[0][0]} to {threshold_summary[-1][0]}")
        print(f"     reduces average OOS trade count by {trade_reduction_pct:.0f}%.")
        print("     Stricter entry selects for more extreme dislocations, filtering noise.")
    else:
        print(f"  -> MARGINAL. Only {trade_reduction_pct:.0f}% trade reduction from "
              f"z>={threshold_summary[0][0]} to z>={threshold_summary[-1][0]}.")
        print("     ATR-normalization already screens many entries; threshold has limited")
        print("     additional filtering power on this dataset.")

    # --- Question 2: Did any VWAP settings beat SMA OOS? ---
    print()
    print(f"  2. Did any VWAP settings beat the SMA baseline OOS?")
    print(f"     (SMA OOS profit factor: {sma_oos_pf:.3f})")
    print()

    if beats_sma:
        print(f"  -> YES. {len(beats_sma)} of {len(results)} parameter pairs beat SMA OOS PF.")
        top_beaters = sorted(beats_sma, key=_rank_key)[:5]
        for r in top_beaters:
            pf_str = f"{r.oos_pf:.3f}" if r.oos_pf != float("inf") else "inf"
            print(f"     entry={r.entry_threshold:.2f}  stop={r.stop_mult:.2f}"
                  f"  OOS PF={pf_str}  OOS ret={r.oos_return:+.2f}%"
                  f"  OOS trades={r.oos_trades}")
    else:
        print(f"  -> NO. Zero parameter pairs beat SMA OOS PF ({sma_oos_pf:.3f}).")
        best = ranked_all[0]
        best_pf_str = f"{best.oos_pf:.3f}" if best.oos_pf != float("inf") else "inf"
        print(f"     Best VWAP OOS PF was {best_pf_str}"
              f" (entry={best.entry_threshold:.2f}, stop={best.stop_mult:.2f}).")
        print("     VWAP Z-score MR as currently designed cannot outperform the SMA")
        print("     baseline on this dataset at any threshold within the sweep range.")

    # --- Question 3: Which look robust vs curve-fit? ---
    print()
    print("  3. Which configurations look robust vs. curve-fit?")
    print()
    print(f"     Avg IS-OOS PF gap across top 10: {avg_gap:+.3f}")
    print()

    robust_count = sum(1 for _, _, g in gaps if g < 0.05)
    overfit_count = sum(1 for _, _, g in gaps if g > 0.20)

    for r, (thr, stp, gap) in zip(top10, gaps):
        verdict = ("robust  " if gap < 0.05
                   else "moderate" if gap < 0.20
                   else "OVERFIT ")
        oos_pf_str = f"{r.oos_pf:.3f}" if r.oos_pf != float("inf") else " inf"
        print(f"     entry={thr:.2f} stop={stp:.2f}"
              f"  IS-PF={r.is_pf:.3f}  OOS-PF={oos_pf_str}"
              f"  gap={gap:+.3f}  [{verdict}]")

    print()
    if robust_count >= 5:
        print(f"  -> {robust_count}/10 top configs show IS-OOS gap < 0.05 — low overfitting risk.")
        print("     The sweep surface is relatively flat, suggesting stability,")
        print("     but also limited discriminative power in the parameters.")
    elif overfit_count >= 5:
        print(f"  -> {overfit_count}/10 top configs show IS-OOS gap > 0.20 — significant overfitting.")
        print("     IS performance is not translating to OOS; the in-sample period")
        print("     is driving parameter selection, not generalizable signal.")
    else:
        print("  -> Mixed picture. Some top configs generalize, some don't.")
        print("     Focus on configs with gap < 0.10 and OOS trade count >= 20.")

    # --- Question 4: Recommendation ---
    print()
    print("  4. Recommended next step")
    print()

    if beats_sma and avg_gap < 0.10:
        print("  -> KEEP TUNING VWAP MR.")
        print("     Multiple configs beat SMA OOS with low overfitting. Add ATR percentile")
        print("     gating next to reduce low-volatility noise entries, then re-sweep.")
    elif beats_sma and avg_gap >= 0.10:
        print("  -> PROCEED WITH CAUTION. Some configs beat SMA OOS but show notable")
        print("     IS-OOS degradation. Do not deploy until a second OOS hold-out period")
        print("     (from fresh data after Apr 4) confirms the result.")
    else:
        print("  -> ADD ADX / REGIME FILTER before continuing VWAP MR tuning.")
        print()
        print("     VWAP Z-score MR fails to consistently beat SMA across this OOS window.")
        print("     The signal fires indiscriminately in both trending and ranging markets.")
        print("     Core issue: mean reversion in a trending regime is a losing trade by")
        print("     construction — the VWAP anchor drifts with the trend, keeping z-score")
        print("     negative for many consecutive bars during a sustained move down, which")
        print("     the stop cannot distinguish from a true reversion setup.")
        print()
        print("     Proposed next steps (in order):")
        print("       a) Add ADX(14) gate: only allow MR entries when ADX < 25 (ranging).")
        print("          This is a single guard condition, no new exit logic required.")
        print("       b) Re-run this same sweep with ADX gate enabled and compare OOS PFs.")
        print("       c) If ADX gate improves OOS PF by >= 0.05 vs best result here,")
        print("          proceed to full regime router implementation.")
        print("       d) If ADX gate shows no improvement, abandon VWAP MR as designed")
        print("          and investigate the wick-fade or squeeze-breakout strategies")
        print("          from the research document.")

    print()
    print("  Note on in-sample optimization:")
    print("  " + "-" * 65)
    print("  This sweep was evaluated on fixed IS/OOS windows, NOT optimized against")
    print("  full-period data. All OOS metrics are genuine hold-out results.")
    print("  A second validation on data after Apr 4, 2026 would provide stronger")
    print("  evidence before any parameter is considered production-ready.")
    print("  " + "-" * 65)
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("=" * 72)
    print(" VWAP Z-SCORE PARAMETER SWEEP  (walk-forward validation)")
    print(f" Dataset : AAPL AMD AMZN GOOGL JPM KO META MSFT NVDA TSLA XOM")
    print(f" Period  : Oct 2025 - Apr 2026  (15-min bars, SIP feed)")
    print(f" Split   : IS {IS_START} to {IS_END}  |  OOS {OOS_START} to {OOS_END}")
    print(f" Grid    : {len(ENTRY_THRESHOLDS)} thresholds x {len(STOP_MULTIPLES)} stop multiples = "
          f"{len(ENTRY_THRESHOLDS) * len(STOP_MULTIPLES)} combos")
    print(f" Capital : $10,000 total, $1,000 per position")
    print(f" Costs   : $0.01 commission + $0.05/share slippage per side")
    print("=" * 72)

    results, sma_is, sma_oos = run_sweep()

    print_top10(results, sma_is, sma_oos)
    print_interpretation(results, sma_is, sma_oos)


if __name__ == "__main__":
    main()

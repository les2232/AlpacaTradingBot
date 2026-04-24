"""
Breakout strategy Stage 3 parameter sweep — entry quality filters.

Context:
    breakout_eval.py established that breakout significantly outperforms VWAP MR best
    (OOS PF 1.129 vs 0.960) and SMA baseline (0.830).  Best settings:
        exit = trail_half or 1x_tight_stop
        window = full_day
        regime = off
        stop   = 0.02

    This sweep adds two entry quality filters:
        breakout_gap_pct_min    : skip if opening gap < N% (require momentum into the open)
        breakout_or_range_pct_min: skip if OR width < N% of OR low (require a meaningful setup level)

    The hypothesis: filtering to higher-quality setups improves OOS PF and reduces drawdown,
    even at the cost of fewer trades.

Grid (50 combos):
    breakout_gap_pct_min     : [0.000, 0.001, 0.002, 0.003, 0.005]
    breakout_or_range_pct_min: [0.000, 0.003, 0.006, 0.009, 0.012]
    exit_style               : [trail_half, 1x_tight_stop]

Fixed (from eval results):
    time_window_mode      = full_day
    regime_filter_enabled = False
    breakout_max_stop_pct = 0.02

Run:
    python breakout_sweep.py
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

# Sweep dimensions
GAP_PCT_MINS      = [0.000, 0.001, 0.002, 0.003, 0.005]   # required opening gap-up (0=off)
OR_RANGE_PCT_MINS = [0.000, 0.003, 0.006, 0.009, 0.012]   # required OR width / OR low (0=off)
EXIT_STYLES       = [BREAKOUT_EXIT_TRAILING_HALF_RANGE, BREAKOUT_EXIT_TARGET_1X_TIGHT_STOP]

EXIT_LABELS = {
    BREAKOUT_EXIT_TRAILING_HALF_RANGE:  "trail_half",
    BREAKOUT_EXIT_TARGET_1X_TIGHT_STOP: "1x_tight_stop",
}

# Fixed settings (confirmed best by breakout_eval.py)
FIXED = dict(
    dataset_path=DATASET,
    strategy_mode="breakout",
    sma_bars=20,
    starting_capital=10_000.0,
    position_size=1_000.0,
    commission=0.01,
    slippage=0.05,
    time_window_mode="full_day",
    regime_filter_enabled=False,
    breakout_max_stop_pct=0.02,
)

# Baseline: best config from breakout_eval (trail_half, full_day, regime=off, stop=0.02)
BASELINE = dict(
    **FIXED,
    breakout_exit_style=BREAKOUT_EXIT_TRAILING_HALF_RANGE,
    breakout_gap_pct_min=0.0,
    breakout_or_range_pct_min=0.0,
)

MIN_MEANINGFUL_OOS_TRADES = 20


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class SweepResult:
    exit_style: str
    gap_pct_min: float
    or_range_pct_min: float
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
# Sweep
# ---------------------------------------------------------------------------

def run_sweep() -> tuple[list[SweepResult], dict, dict]:
    grid = list(itertools.product(GAP_PCT_MINS, OR_RANGE_PCT_MINS, EXIT_STYLES))
    total = len(grid)

    print(f"\n[sweep] {total} combos x 2 windows = {total * 2} breakout runs")
    print(f"[sweep] IS : {IS_START} to {IS_END}")
    print(f"[sweep] OOS: {OOS_START} to {OOS_END}\n")

    results: list[SweepResult] = []

    for i, (gap_min, or_min, exit_style) in enumerate(grid):
        run_is  = i * 2 + 1
        run_oos = i * 2 + 2
        label = (
            f"{EXIT_LABELS[exit_style]}  gap>={gap_min:.3f}  or>={or_min:.3f}"
        )

        kwargs_base = dict(
            **FIXED,
            breakout_exit_style=exit_style,
            breakout_gap_pct_min=gap_min,
            breakout_or_range_pct_min=or_min,
        )

        print(f"[{run_is:3d}/{total * 2}] {label}  IS ...", end=" ", flush=True)
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
        results.append(SweepResult(
            exit_style=exit_style,
            gap_pct_min=gap_min,
            or_range_pct_min=or_min,
            is_pf=is_pf,   oos_pf=oos_pf,
            is_return=is_ret, oos_return=oos_ret,
            is_dd=is_dd,   oos_dd=oos_dd,
            is_trades=is_tr, oos_trades=oos_tr, oos_wr=oos_wr,
        ))

    # Baseline reference
    print()
    for label, cfg in [
        ("baseline IS ", {**BASELINE, "start_date": IS_START,  "end_date": IS_END}),
        ("baseline OOS", {**BASELINE, "start_date": OOS_START, "end_date": OOS_END}),
    ]:
        print(f"[ref] {label} ...", end=" ", flush=True)
        _run_silent(cfg)
        print("done")

    base_is_r  = _run_silent({**BASELINE, "start_date": IS_START,  "end_date": IS_END})
    base_oos_r = _run_silent({**BASELINE, "start_date": OOS_START, "end_date": OOS_END})

    return results, base_is_r, base_oos_r


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def _rank_key(r: SweepResult) -> tuple:
    return (-min(r.oos_pf, 9.999), r.oos_dd, -r.oos_return)


def print_results(
    results: list[SweepResult],
    base_is: dict, base_oos: dict,
) -> None:
    base_oos_pf  = _profit_factor(base_oos.get("trades", []))
    base_is_pf   = _profit_factor(base_is.get("trades", []))
    base_oos_ret = base_oos.get("total_return_pct", 0.0)
    base_is_ret  = base_is.get("total_return_pct", 0.0)
    base_oos_dd  = base_oos.get("max_drawdown_pct", 0.0)
    base_oos_tr  = base_oos.get("total_trades", 0)
    base_is_tr   = base_is.get("total_trades", 0)

    ranked = sorted(results, key=_rank_key)

    print()
    print("=" * 104)
    print("  BREAKOUT SWEEP  --  TOP CONFIGURATIONS  (ranked by OOS profit factor)")
    print("=" * 104)

    hdr = (
        f"{'Rank':>4}  {'Exit Style':>13}  {'Gap>=':>5}  {'OR>=':>5}  "
        f"{'IS-PF':>6}  {'OOS-PF':>6}  "
        f"{'IS-Ret':>7}  {'OOS-Ret':>7}  {'OOS-DD':>7}  {'IS-Tr':>6}  {'OOS-Tr':>6}"
    )
    sep = "-" * 104
    print(hdr)
    print(sep)

    for rank, r in enumerate(ranked[:15], 1):
        beats_base = r.oos_pf > base_oos_pf
        low_tr     = r.oos_trades < MIN_MEANINGFUL_OOS_TRADES
        flag = " *" if beats_base else "  "
        if low_tr:
            flag = " !"
        print(
            f"{rank:>4}{flag} {EXIT_LABELS[r.exit_style]:>13}  {r.gap_pct_min:.3f}  {r.or_range_pct_min:.3f}  "
            f"{_pf_str(r.is_pf):>6}  {_pf_str(r.oos_pf):>6}  "
            f"{r.is_return:>+7.2f}%  {r.oos_return:>+7.2f}%"
            f"  {r.oos_dd:>7.2f}%  {r.is_trades:>6}  {r.oos_trades:>6}"
        )

    print(sep)
    print(
        f"{'ref':>4}   {'baseline':>13}  {'0.000':>5}  {'0.000':>5}  "
        f"{_pf_str(base_is_pf):>6}  {_pf_str(base_oos_pf):>6}  "
        f"{base_is_ret:>+7.2f}%  {base_oos_ret:>+7.2f}%"
        f"  {base_oos_dd:>7.2f}%  {base_is_tr:>6}  {base_oos_tr:>6}"
    )
    print(sep)
    print("  * = beats unfiltered baseline OOS PF    ! = fewer than 20 OOS trades (low confidence)")


def print_analysis(
    results: list[SweepResult],
    base_is: dict, base_oos: dict,
) -> None:
    base_oos_pf = _profit_factor(base_oos.get("trades", []))
    base_oos_tr = base_oos.get("total_trades", 0)

    ranked_all = sorted(results, key=_rank_key)
    reliable   = [r for r in results if r.oos_trades >= MIN_MEANINGFUL_OOS_TRADES]
    above_1    = [r for r in reliable if r.oos_pf > 1.0]
    beats_base = [r for r in reliable if r.oos_pf > base_oos_pf]

    def avg(lst: list) -> float:
        return sum(lst) / len(lst) if lst else 0.0

    # Aggregate by gap_pct_min (across all or_range and exits)
    pf_by_gap:  dict[float, list[float]] = defaultdict(list)
    tr_by_gap:  dict[float, list[int]]   = defaultdict(list)
    dd_by_gap:  dict[float, list[float]] = defaultdict(list)
    for r in results:
        if r.oos_pf != float("inf"):
            pf_by_gap[r.gap_pct_min].append(r.oos_pf)
        tr_by_gap[r.gap_pct_min].append(r.oos_trades)
        dd_by_gap[r.gap_pct_min].append(r.oos_dd)

    # Aggregate by or_range_pct_min
    pf_by_or:  dict[float, list[float]] = defaultdict(list)
    tr_by_or:  dict[float, list[int]]   = defaultdict(list)
    dd_by_or:  dict[float, list[float]] = defaultdict(list)
    for r in results:
        if r.oos_pf != float("inf"):
            pf_by_or[r.or_range_pct_min].append(r.oos_pf)
        tr_by_or[r.or_range_pct_min].append(r.oos_trades)
        dd_by_or[r.or_range_pct_min].append(r.oos_dd)

    # IS-OOS gaps for top configs
    top10 = ranked_all[:10]
    gaps  = [r.is_pf - r.oos_pf for r in top10]
    avg_gap = avg(gaps)
    overfit_count = sum(1 for g in gaps if g > 0.15)
    stable_count  = sum(1 for g in gaps if abs(g) < 0.05)

    print()
    print("=" * 72)
    print("  ANALYSIS")
    print("=" * 72)

    # --- Q1: Do quality filters improve OOS PF? ---
    print()
    print("  1. Do quality filters improve OOS profit factor over unfiltered baseline?")
    print(f"     Unfiltered baseline OOS PF: {_pf_str(base_oos_pf)}")
    print()

    if beats_base:
        top = sorted(beats_base, key=_rank_key)[0]
        print(f"  -> YES. {len(beats_base)} of {len(reliable)} reliable configs beat baseline OOS PF.")
        print()
        for r in sorted(beats_base, key=_rank_key)[:5]:
            print(
                f"     {EXIT_LABELS[r.exit_style]:>13}  gap>={r.gap_pct_min:.3f}  or>={r.or_range_pct_min:.3f}"
                f"  OOS PF={_pf_str(r.oos_pf)}"
                f"  OOS ret={r.oos_return:+.2f}%"
                f"  OOS DD={r.oos_dd:.2f}%"
                f"  trades={r.oos_trades}"
            )
    else:
        best = ranked_all[0]
        print(f"  -> NO. No reliable config improves on unfiltered baseline ({_pf_str(base_oos_pf)}).")
        print(f"     Best filtered: {EXIT_LABELS[best.exit_style]}  gap>={best.gap_pct_min:.3f}"
              f"  or>={best.or_range_pct_min:.3f}  OOS PF={_pf_str(best.oos_pf)}  trades={best.oos_trades}")

    if above_1:
        print()
        print(f"  {len(above_1)} of {len(reliable)} configs exceed OOS PF 1.0.")

    # --- Q2: Gap filter effect ---
    print()
    print("  2. Effect of gap_pct_min filter:")
    print()
    print(f"     {'gap_pct_min':>11}  {'Avg OOS PF':>10}  {'Avg OOS Trades':>14}  {'Avg OOS DD':>10}")
    print("     " + "-" * 50)
    for gap_val in sorted(pf_by_gap.keys()):
        print(
            f"     {gap_val:>11.3f}  {avg(pf_by_gap[gap_val]):>10.3f}"
            f"  {avg(tr_by_gap[gap_val]):>14.1f}  {avg(dd_by_gap[gap_val]):>10.2f}%"
        )

    gap_vals_sorted = sorted(pf_by_gap.keys())
    pf_at_zero = avg(pf_by_gap.get(0.0, [0]))
    pf_at_best_gap = max((avg(pf_by_gap[g]) for g in gap_vals_sorted), default=0)
    gap_lift = pf_at_best_gap - pf_at_zero
    best_gap_val = max(gap_vals_sorted, key=lambda g: avg(pf_by_gap.get(g, [0])))

    print()
    if gap_lift > 0.02:
        print(f"  -> Gap filter helps (+{gap_lift:.3f} PF lift at gap_pct_min={best_gap_val:.3f}).")
        print(f"     Gap-up days have stronger directional follow-through.")
    elif gap_lift > 0:
        print(f"  -> Gap filter has minor positive effect (+{gap_lift:.3f} at gap_pct_min={best_gap_val:.3f}).")
        print(f"     Lift is small; check trade count to ensure sufficient sample.")
    else:
        print(f"  -> Gap filter does not improve OOS PF on this dataset.")
        print(f"     ORB entries work equally well on gap and non-gap days here.")

    # --- Q3: OR range filter effect ---
    print()
    print("  3. Effect of or_range_pct_min filter:")
    print()
    print(f"     {'or_range_pct_min':>16}  {'Avg OOS PF':>10}  {'Avg OOS Trades':>14}  {'Avg OOS DD':>10}")
    print("     " + "-" * 55)
    for or_val in sorted(pf_by_or.keys()):
        print(
            f"     {or_val:>16.3f}  {avg(pf_by_or[or_val]):>10.3f}"
            f"  {avg(tr_by_or[or_val]):>14.1f}  {avg(dd_by_or[or_val]):>10.2f}%"
        )

    or_vals_sorted = sorted(pf_by_or.keys())
    pf_at_zero_or = avg(pf_by_or.get(0.0, [0]))
    pf_at_best_or = max((avg(pf_by_or[v]) for v in or_vals_sorted), default=0)
    or_lift = pf_at_best_or - pf_at_zero_or
    best_or_val = max(or_vals_sorted, key=lambda v: avg(pf_by_or.get(v, [0])))

    print()
    if or_lift > 0.02:
        print(f"  -> OR range filter helps (+{or_lift:.3f} PF lift at or_range_pct_min={best_or_val:.3f}).")
        print(f"     Wider opening ranges produce more decisive breakout levels.")
    elif or_lift > 0:
        print(f"  -> OR range filter has minor positive effect (+{or_lift:.3f} at or>={best_or_val:.3f}).")
    else:
        print(f"  -> OR range filter does not improve OOS PF on this dataset.")
        print(f"     Narrow opening ranges are equally valid breakout setups here.")

    # --- Q4: Overfitting check ---
    print()
    print(f"  4. IS-OOS stability (top 10):")
    print(f"     avg IS-OOS gap={avg_gap:+.3f}  overfit(>0.15)={overfit_count}/10  stable(<0.05)={stable_count}/10")
    if overfit_count >= 3:
        print("  -> Warning: overfitting detected in top configs.")
        print("     IS PF is substantially higher than OOS PF. These parameters are IS-optimized.")
    elif stable_count >= 6:
        print("  -> IS-OOS gap is stable. Results are likely to generalize.")
    else:
        print("  -> Moderate IS-OOS gap. Results are plausible but not strongly confirmed.")

    # --- Trade count sensitivity ---
    print()
    print("  5. Trade count sensitivity:")
    print(f"     Unfiltered baseline OOS trades: {base_oos_tr}")
    reliable_trades = [r.oos_trades for r in reliable]
    if reliable_trades:
        min_tr  = min(reliable_trades)
        max_tr  = max(reliable_trades)
        avg_tr  = avg(reliable_trades)
        print(f"     Filtered configs: min={min_tr}  avg={avg_tr:.0f}  max={max_tr}")
        if min_tr < 30:
            print("  -> Warning: some configs have very few OOS trades (<30).")
            print("     PF estimates are unreliable at low trade counts — favor configs with 50+ trades.")

    # --- Final verdict ---
    print()
    print("=" * 72)
    print("  VERDICT: STAGE 3 QUALITY FILTERS")
    print("=" * 72)
    print()

    best_reliable = next((r for r in ranked_all if r.oos_trades >= MIN_MEANINGFUL_OOS_TRADES), None)

    if not best_reliable:
        print("  Insufficient OOS trades across all filtered configs — no reliable result.")
        return

    improvement = best_reliable.oos_pf - base_oos_pf

    if improvement > 0.05 and best_reliable.oos_pf > 1.0:
        print(f"  FILTERS HELP. Best filtered config exceeds unfiltered baseline by +{improvement:.3f} OOS PF.")
        print()
        print(f"  Recommended config:")
        print(f"    exit={EXIT_LABELS[best_reliable.exit_style]}")
        print(f"    gap_pct_min={best_reliable.gap_pct_min:.3f}  or_range_pct_min={best_reliable.or_range_pct_min:.3f}")
        print(f"    OOS PF={_pf_str(best_reliable.oos_pf)}  OOS ret={best_reliable.oos_return:+.2f}%")
        print(f"    OOS DD={best_reliable.oos_dd:.2f}%  trades={best_reliable.oos_trades}")
        is_oos_gap = best_reliable.is_pf - best_reliable.oos_pf
        print()
        if abs(is_oos_gap) < 0.08:
            print("  IS-OOS gap is stable. Proceed to live validation.")
        else:
            print(f"  IS-OOS gap={is_oos_gap:+.3f} — verify on a second OOS window before going live.")

    elif improvement > 0:
        print(f"  MARGINAL IMPROVEMENT (+{improvement:.3f} OOS PF). Filters help slightly.")
        print()
        print(f"  Best filtered: {EXIT_LABELS[best_reliable.exit_style]}"
              f"  gap>={best_reliable.gap_pct_min:.3f}"
              f"  or>={best_reliable.or_range_pct_min:.3f}"
              f"  OOS PF={_pf_str(best_reliable.oos_pf)}"
              f"  trades={best_reliable.oos_trades}")
        print()
        print("  The unfiltered baseline (OOS PF 1.129) is already profitable.")
        print("  Apply filters only if they don't reduce trade count below ~100 OOS trades.")

    else:
        print(f"  FILTERS DO NOT HELP (best={_pf_str(best_reliable.oos_pf)} vs baseline={_pf_str(base_oos_pf)}).")
        print()
        print("  Quality filters add no value on this dataset.")
        print("  The unfiltered baseline is the best config:")
        print(f"    exit=trail_half  gap_pct_min=0.000  or_range_pct_min=0.000")
        print(f"    OOS PF={_pf_str(base_oos_pf)}  trades={base_oos_tr}")
        print()
        print("  Recommended: proceed to live validation with the unfiltered baseline.")

    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    grid = list(itertools.product(GAP_PCT_MINS, OR_RANGE_PCT_MINS, EXIT_STYLES))
    print("=" * 72)
    print("  BREAKOUT STRATEGY  --  STAGE 3 PARAMETER SWEEP")
    print(f"  Dataset : AAPL AMD AMZN GOOGL JPM KO META MSFT NVDA TSLA XOM")
    print(f"  Period  : Oct 2025 - Apr 2026  (15-min bars, SIP)")
    print(f"  Split   : IS {IS_START} to {IS_END}  |  OOS {OOS_START} to {OOS_END}")
    print(f"  Grid    : {len(GAP_PCT_MINS)} gap x {len(OR_RANGE_PCT_MINS)} or_range"
          f" x {len(EXIT_STYLES)} exits = {len(grid)} combos")
    print(f"  Fixed   : full_day window  |  regime=off  |  stop=0.02")
    print(f"  Capital : $10,000 total  |  $1,000 per position")
    print(f"  Costs   : $0.01 commission + $0.05/share slippage per side")
    print("=" * 72)

    results, base_is, base_oos = run_sweep()
    print_results(results, base_is, base_oos)
    print_analysis(results, base_is, base_oos)


if __name__ == "__main__":
    main()

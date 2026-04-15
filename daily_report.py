#!/usr/bin/env python3
"""
daily_report.py
---------------
Reads the bot's JSONL logs for one trading day and prints a diagnostic
report comparing live behavior to backtest expectations.

Usage:
    python daily_report.py                      # today
    python daily_report.py --date 2026-04-09
    python daily_report.py --date 2026-04-09 --log-root logs
    python daily_report.py --date 2026-04-09 --json   # machine-readable output

Requires: pandas
"""

import argparse
import json
import sys
from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd

# ---------------------------------------------------------------------------
# Backtest expectations
# Update these whenever the validated config changes.
# These are the OOS numbers from the mean-reversion validation.
# ---------------------------------------------------------------------------

BACKTEST: dict[str, Any] = {
    "symbols":                         15,
    "bars_per_day":                    26,     # 390-min session ÷ 15-min bars
    "signal_rate_per_symbol_per_day":  2.1,    # avg BUY signals per symbol per session
    "win_rate":                        0.65,   # fraction of closed trades that are winners
    "avg_hold_bars":                   4.2,    # bars held per trade
    "rejection_rate":                  0.72,   # fraction of evaluated bars → HOLD
    "avg_slippage_bps":                5.0,    # expected adverse slippage (bps)
}

# ---------------------------------------------------------------------------
# Alert thresholds
# ---------------------------------------------------------------------------

WARN: dict[str, Any] = {
    "avg_slippage_bps":          15.0,   # bps — execution cost is eroding edge
    "worst_slippage_bps":        40.0,   # bps — single bad fill
    "signal_rate_pct_of_bt":     0.50,   # live rate below 50% of backtest → filter drift
    "signal_rate_pct_of_bt_hi":  2.00,   # live rate above 200% of backtest → filter not working
    "win_rate_low":               0.50,   # below this → strategy deteriorating
    "win_rate_high":              0.90,   # above this → too few trades, not meaningful
    "eod_exit_rate":              0.30,   # >30% of exits from EOD flatten → exits not firing
    "stale_bar_count":            2,      # stale bars during market hours = data feed issue
    "avg_bar_age_s":              240.0,  # seconds — IEX feed has ~4-5 min publication delay; warn if approaching the 300s stale cutoff
    "min_trades_for_win_rate":    5,      # don't report win rate below this sample size
}


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------

def _load(path: Path) -> pd.DataFrame:
    """Read a JSONL file into a DataFrame. Returns empty DataFrame if missing."""
    if not path.exists():
        return pd.DataFrame()
    lines = [l.strip() for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
    if not lines:
        return pd.DataFrame()
    return pd.DataFrame([json.loads(l) for l in lines])


def load_logs(log_dir: Path) -> dict[str, pd.DataFrame]:
    return {
        "bars":      _load(log_dir / "bars.jsonl"),
        "signals":   _load(log_dir / "signals.jsonl"),
        "risk":      _load(log_dir / "risk.jsonl"),
        "execution": _load(log_dir / "execution.jsonl"),
        "positions": _load(log_dir / "positions.jsonl"),
    }


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------

def _col(df: pd.DataFrame, col: str) -> pd.Series:
    """Return column or empty Series if it doesn't exist."""
    return df[col] if col in df.columns else pd.Series(dtype=object)


def _flag(value: float, warn_threshold: float, low_is_bad: bool = True) -> str:
    """Return 'WARN' or 'ok' based on whether value breaches threshold."""
    if low_is_bad:
        return "WARN" if value < warn_threshold else "ok"
    else:
        return "WARN" if value > warn_threshold else "ok"


# ---------------------------------------------------------------------------
# Section computations
# ---------------------------------------------------------------------------

def compute_bar_quality(bars: pd.DataFrame) -> dict:
    received = bars[_col(bars, "event") == "bar.received"] if not bars.empty else pd.DataFrame()
    if received.empty:
        return {"evaluated": 0, "avg_age_s": None, "stale_count": 0, "symbols": []}

    return {
        "evaluated":   len(received),
        "avg_age_s":   received["bar_age_s"].mean() if "bar_age_s" in received.columns else None,
        "stale_count": int(received["stale"].sum()) if "stale" in received.columns else 0,
        "symbols":     sorted(received["symbol"].unique().tolist()) if "symbol" in received.columns else [],
    }


def compute_signal_behavior(signals: pd.DataFrame) -> dict:
    evaluated = signals[_col(signals, "event") == "signal.evaluated"] if not signals.empty else pd.DataFrame()
    if evaluated.empty:
        return {
            "total_evaluated": 0, "buy_signals": 0, "sell_signals": 0,
            "by_symbol": {}, "rejections": {}, "rejection_rate": None,
        }

    buys  = evaluated[_col(evaluated, "action") == "BUY"]
    sells = evaluated[_col(evaluated, "action") == "SELL"]
    holds = evaluated[_col(evaluated, "action") == "HOLD"]

    # Rejection breakdown (why HOLD fired when a signal might have been expected)
    rejections: dict[str, int] = {}
    if "rejection" in evaluated.columns:
        rej = evaluated["rejection"].dropna()
        rejections = rej.value_counts().to_dict()

    # Per-symbol BUY signal count
    by_symbol: dict[str, int] = {}
    if "symbol" in buys.columns:
        by_symbol = buys["symbol"].value_counts().to_dict()

    return {
        "total_evaluated": len(evaluated),
        "buy_signals":     len(buys),
        "sell_signals":    len(sells),
        "hold_count":      len(holds),
        "by_symbol":       by_symbol,
        "rejections":      rejections,
        "rejection_rate":  len(holds) / len(evaluated) if len(evaluated) > 0 else None,
    }


def compute_execution(execution: pd.DataFrame) -> dict:
    if execution.empty:
        return {"submitted_buys": 0, "submitted_sells": 0,
                "avg_slippage_bps": None, "worst_slippage_bps": None,
                "partial_fills": 0, "avg_latency_ms": None}

    submitted = execution[_col(execution, "event") == "order.submitted"]
    filled    = execution[_col(execution, "event") == "order.filled"]
    partials  = execution[_col(execution, "event") == "order.partial_fill"]

    buys  = submitted[_col(submitted, "side") == "buy"]  if not submitted.empty else pd.DataFrame()
    sells = submitted[_col(submitted, "side") == "sell"] if not submitted.empty else pd.DataFrame()

    slippage_series = (
        filled["slippage_bps"].dropna()
        if not filled.empty and "slippage_bps" in filled.columns
        else pd.Series(dtype=float)
    )
    latency_series = (
        filled["submit_to_fill_ms"].dropna()
        if not filled.empty and "submit_to_fill_ms" in filled.columns
        else pd.Series(dtype=float)
    )

    return {
        "submitted_buys":    len(buys),
        "submitted_sells":   len(sells),
        "avg_slippage_bps":  float(slippage_series.mean())  if len(slippage_series) > 0 else None,
        "worst_slippage_bps":float(slippage_series.max())   if len(slippage_series) > 0 else None,
        "partial_fills":     len(partials),
        "avg_latency_ms":    float(latency_series.mean())   if len(latency_series) > 0 else None,
    }


def compute_risk(risk: pd.DataFrame) -> dict:
    if risk.empty:
        return {
            "kill_switch_triggered": False,
            "buy_blocks_by_reason": {},
            "sell_blocks_by_reason": {},
        }

    checks = risk[_col(risk, "event") == "risk.check"] if not risk.empty else pd.DataFrame()
    triggered = risk[_col(risk, "event") == "kill_switch.triggered"] if not risk.empty else pd.DataFrame()

    buy_blocks: dict[str, int] = {}
    sell_blocks: dict[str, int] = {}
    if not checks.empty and "block_reason" in checks.columns:
        blocked = checks[checks["block_reason"].notna()]
        if not blocked.empty and "action" in blocked.columns:
            buy_blocked  = blocked[blocked["action"] == "BUY"]
            sell_blocked = blocked[blocked["action"] == "SELL"]
            buy_blocks   = buy_blocked["block_reason"].value_counts().to_dict()
            sell_blocks  = sell_blocked["block_reason"].value_counts().to_dict()
        else:
            buy_blocks = blocked["block_reason"].value_counts().to_dict()

    return {
        "kill_switch_triggered": len(triggered) > 0,
        "buy_blocks_by_reason":  buy_blocks,
        "sell_blocks_by_reason": sell_blocks,
    }


def compute_positions(positions: pd.DataFrame) -> dict:
    if positions.empty:
        return {
            "opened": 0, "closed": 0, "win_rate": None,
            "avg_hold_bars": None, "avg_pnl_usd": None,
            "exit_reasons": {}, "total_pnl_usd": None,
        }

    opened = positions[_col(positions, "event") == "position.opened"]
    closed = positions[_col(positions, "event") == "position.closed"]

    if closed.empty:
        return {
            "opened": len(opened), "closed": 0, "win_rate": None,
            "avg_hold_bars": None, "avg_pnl_usd": None,
            "exit_reasons": {}, "total_pnl_usd": None,
        }

    winners = closed["winner"].sum() if "winner" in closed.columns else 0
    win_rate = float(winners / len(closed)) if len(closed) > 0 else None

    avg_hold = float(closed["holding_bars"].mean()) if "holding_bars" in closed.columns else None
    avg_pnl  = float(closed["pnl_usd"].mean())     if "pnl_usd" in closed.columns else None
    total_pnl= float(closed["pnl_usd"].sum())      if "pnl_usd" in closed.columns else None

    exit_reasons: dict[str, int] = {}
    if "exit_reason" in closed.columns:
        exit_reasons = closed["exit_reason"].value_counts().to_dict()

    return {
        "opened":        len(opened),
        "closed":        len(closed),
        "win_rate":      win_rate,
        "avg_hold_bars": avg_hold,
        "avg_pnl_usd":   avg_pnl,
        "total_pnl_usd": total_pnl,
        "exit_reasons":  exit_reasons,
    }


# ---------------------------------------------------------------------------
# Concern detection
# ---------------------------------------------------------------------------

def detect_concerns(bar_q: dict, sig: dict, ex: dict, pos: dict) -> list[str]:
    """
    Return a list of concern strings.  Empty list = everything looks normal.
    """
    concerns = []
    n_sym = BACKTEST["symbols"]

    # --- Slippage ---
    if ex["avg_slippage_bps"] is not None:
        if ex["avg_slippage_bps"] > WARN["avg_slippage_bps"]:
            concerns.append(
                f"SLIPPAGE HIGH: avg {ex['avg_slippage_bps']:.1f} bps "
                f"(threshold {WARN['avg_slippage_bps']} bps) — execution cost eroding edge"
            )
    if ex["worst_slippage_bps"] is not None:
        if ex["worst_slippage_bps"] > WARN["worst_slippage_bps"]:
            concerns.append(
                f"SLIPPAGE SPIKE: worst {ex['worst_slippage_bps']:.1f} bps "
                f"(threshold {WARN['worst_slippage_bps']} bps) — check that fill for anomalies"
            )

    # --- Signal rate ---
    if sig["buy_signals"] > 0 and sig["total_evaluated"] > 0:
        live_rate = sig["buy_signals"] / n_sym
        bt_rate   = BACKTEST["signal_rate_per_symbol_per_day"]
        ratio     = live_rate / bt_rate if bt_rate > 0 else 0
        if ratio < WARN["signal_rate_pct_of_bt"]:
            concerns.append(
                f"SIGNAL RATE LOW: {live_rate:.1f}/symbol/day vs backtest {bt_rate} "
                f"({ratio:.0%} of backtest) — live filters may differ from backtest"
            )
        if ratio > WARN["signal_rate_pct_of_bt_hi"]:
            concerns.append(
                f"SIGNAL RATE HIGH: {live_rate:.1f}/symbol/day vs backtest {bt_rate} "
                f"({ratio:.0%} of backtest) — filter may not be applying correctly in live"
            )

    # --- Win rate ---
    n_closed = pos.get("closed", 0)
    if n_closed >= WARN["min_trades_for_win_rate"] and pos["win_rate"] is not None:
        wr = pos["win_rate"]
        if wr < WARN["win_rate_low"]:
            concerns.append(
                f"WIN RATE LOW: {wr:.0%} (threshold {WARN['win_rate_low']:.0%}) "
                f"— strategy may not be working in current regime"
            )
        if wr > WARN["win_rate_high"]:
            concerns.append(
                f"WIN RATE HIGH: {wr:.0%} ({n_closed} trades) "
                f"— sample too small or streak, not statistically significant"
            )

    # --- EOD exit rate ---
    total_exits = sum(pos.get("exit_reasons", {}).values())
    eod_exits   = pos.get("exit_reasons", {}).get("eod_flatten", 0)
    if total_exits >= 3:
        eod_rate = eod_exits / total_exits
        if eod_rate > WARN["eod_exit_rate"]:
            concerns.append(
                f"EOD EXITS HIGH: {eod_exits}/{total_exits} exits ({eod_rate:.0%}) via EOD flatten "
                f"— positions not recovering within session; exit logic or SMA level may differ from backtest"
            )

    # --- Bar quality ---
    if bar_q["stale_count"] >= WARN["stale_bar_count"]:
        concerns.append(
            f"STALE BARS: {bar_q['stale_count']} stale bar(s) detected "
            f"— data feed latency or Alpaca connectivity issue"
        )
    if bar_q["avg_age_s"] is not None and bar_q["avg_age_s"] > WARN["avg_bar_age_s"]:
        concerns.append(
            f"BAR LATENCY: avg bar age {bar_q['avg_age_s']:.1f}s "
            f"(threshold {WARN['avg_bar_age_s']}s) — bot is processing stale data"
        )

    # --- Partial fills ---
    if ex["partial_fills"] > 0:
        concerns.append(
            f"PARTIAL FILLS: {ex['partial_fills']} partial fill(s) detected "
            f"— universe liquidity may have declined; check fill quality"
        )

    # --- Kill switch ---
    # Reported separately in the report, not as a "concern" here.

    return concerns


# ---------------------------------------------------------------------------
# Report formatter
# ---------------------------------------------------------------------------

def fmt(value: Any, fmt_str: str = "", fallback: str = "n/a") -> str:
    if value is None:
        return fallback
    if isinstance(value, float):
        return format(value, fmt_str) if fmt_str else f"{value:.2f}"
    return str(value)


def print_report(
    report_date: str,
    bar_q: dict,
    sig: dict,
    ex: dict,
    risk: dict,
    pos: dict,
    concerns: list[str],
) -> None:
    n_sym     = BACKTEST["symbols"]
    bt_bars   = BACKTEST["bars_per_day"]
    divider   = "─" * 60

    def section(title: str) -> None:
        print(f"\n{divider}")
        print(f"  {title}")
        print(divider)

    def row(label: str, value: str, note: str = "", flag: str = "") -> None:
        flag_str = f"  [{flag}]" if flag else ""
        note_str = f"  {note}" if note else ""
        print(f"  {label:<38} {value:<14}{flag_str}{note_str}")

    print(f"\n{'═' * 60}")
    print(f"  DAILY BOT REPORT  ·  {report_date}")
    print(f"{'═' * 60}")

    # ── Data quality ──────────────────────────────────────────────
    section("DATA QUALITY")
    expected_bars = n_sym * bt_bars
    row("Bars evaluated",        fmt(bar_q["evaluated"]),
        f"(expect ~{expected_bars} = {n_sym} syms × {bt_bars} bars)")
    avg_age = bar_q["avg_age_s"]
    row("Avg bar age (s)",        fmt(avg_age, ".1f"),
        flag=_flag(avg_age or 0, WARN["avg_bar_age_s"], low_is_bad=False) if avg_age else "")
    row("Stale bars",             fmt(bar_q["stale_count"]),
        flag=_flag(bar_q["stale_count"], WARN["stale_bar_count"] - 1, low_is_bad=False)
             if bar_q["stale_count"] > 0 else "")

    # ── Signal behavior ───────────────────────────────────────────
    section("SIGNAL BEHAVIOR")
    total_eval = sig["total_evaluated"]
    buy_count  = sig["buy_signals"]
    bt_total   = n_sym * BACKTEST["signal_rate_per_symbol_per_day"]
    live_per_sym = buy_count / n_sym if n_sym > 0 else 0
    bt_per_sym   = BACKTEST["signal_rate_per_symbol_per_day"]
    sig_ratio    = live_per_sym / bt_per_sym if bt_per_sym > 0 else 0

    row("Total bars evaluated",   fmt(total_eval))
    row("BUY signals fired",      fmt(buy_count),
        f"({live_per_sym:.1f}/sym/day  bt={bt_per_sym:.1f}  ratio={sig_ratio:.0%})",
        flag=_flag(sig_ratio, WARN["signal_rate_pct_of_bt"], low_is_bad=True)
             if buy_count > 0 else "")
    row("SELL signals fired",     fmt(sig["sell_signals"]))
    rej_rate = sig["rejection_rate"]
    row("Rejection rate",         fmt(rej_rate, ".0%") if rej_rate is not None else "n/a",
        f"(backtest ~{BACKTEST['rejection_rate']:.0%})")

    if sig["rejections"]:
        print()
        print("  Rejection breakdown:")
        for reason, count in sorted(sig["rejections"].items(), key=lambda x: -x[1]):
            pct = count / total_eval * 100 if total_eval > 0 else 0
            print(f"    {reason:<32} {count:>4}  ({pct:.1f}%)")

    if sig["by_symbol"]:
        print()
        print("  BUY signals by symbol:")
        for sym, count in sorted(sig["by_symbol"].items(), key=lambda x: -x[1]):
            print(f"    {sym:<8} {count:>3}")

    # ── Execution ─────────────────────────────────────────────────
    section("EXECUTION")
    row("Orders submitted (buy)",  fmt(ex["submitted_buys"]))
    row("Orders submitted (sell)", fmt(ex["submitted_sells"]))
    row("Partial fills",           fmt(ex["partial_fills"]),
        flag="WARN" if ex["partial_fills"] > 0 else "")
    avg_slip = ex["avg_slippage_bps"]
    worst_slip = ex["worst_slippage_bps"]
    row("Avg slippage (bps)",      fmt(avg_slip, ".1f"),
        f"(backtest ~{BACKTEST['avg_slippage_bps']:.1f} bps)",
        flag=_flag(avg_slip or 0, WARN["avg_slippage_bps"], low_is_bad=False) if avg_slip else "")
    row("Worst slippage (bps)",    fmt(worst_slip, ".1f"),
        flag=_flag(worst_slip or 0, WARN["worst_slippage_bps"], low_is_bad=False) if worst_slip else "")
    row("Avg fill latency (ms)",   fmt(ex["avg_latency_ms"], ".0f"))

    # ── Risk ──────────────────────────────────────────────────────
    section("RISK CONTROLS")
    ks = "TRIGGERED ⚠" if risk["kill_switch_triggered"] else "not triggered"
    row("Kill switch",             ks,
        flag="WARN" if risk["kill_switch_triggered"] else "")
    if risk["buy_blocks_by_reason"]:
        print()
        print("  BUY blocks by reason:")
        for reason, count in sorted(risk["buy_blocks_by_reason"].items(), key=lambda x: -x[1]):
            print(f"    {reason:<38} {count:>3}")
    if risk["sell_blocks_by_reason"]:
        print()
        print("  SELL blocks by reason:")
        for reason, count in sorted(risk["sell_blocks_by_reason"].items(), key=lambda x: -x[1]):
            print(f"    {reason:<38} {count:>3}")

    # ── Trade outcomes ────────────────────────────────────────────
    section("TRADE OUTCOMES")
    n_closed = pos["closed"]
    row("Positions opened",        fmt(pos["opened"]))
    row("Positions closed",        fmt(n_closed))

    if n_closed >= WARN["min_trades_for_win_rate"] and pos["win_rate"] is not None:
        wr = pos["win_rate"]
        row("Win rate",            fmt(wr, ".0%"),
            f"(backtest {BACKTEST['win_rate']:.0%})",
            flag=_flag(wr, WARN["win_rate_low"], low_is_bad=True))
    elif n_closed > 0:
        row("Win rate",            f"{pos['win_rate']:.0%}" if pos["win_rate"] else "n/a",
            f"(only {n_closed} trade(s) — not meaningful yet)")
    else:
        row("Win rate",            "n/a", "(no closed positions today)")

    avg_hold = pos["avg_hold_bars"]
    row("Avg hold (bars)",         fmt(avg_hold, ".1f"),
        f"(backtest {BACKTEST['avg_hold_bars']:.1f})")
    row("Avg PnL per trade ($)",   fmt(pos["avg_pnl_usd"], ".2f"))
    row("Total PnL today ($)",     fmt(pos["total_pnl_usd"], ".2f"))

    if pos["exit_reasons"]:
        print()
        print("  Exit reason breakdown:")
        total_exits = sum(pos["exit_reasons"].values())
        for reason, count in sorted(pos["exit_reasons"].items(), key=lambda x: -x[1]):
            pct = count / total_exits * 100 if total_exits > 0 else 0
            flag_str = ""
            if reason == "eod_flatten" and pct > WARN["eod_exit_rate"] * 100:
                flag_str = "  [WARN]"
            print(f"    {reason:<32} {count:>3}  ({pct:.0f}%){flag_str}")

    # ── Backtest comparison ───────────────────────────────────────
    section("vs BACKTEST COMPARISON")
    print(f"  {'Metric':<32} {'Live':>10}  {'Backtest':>10}  {'Delta':>10}")
    print(f"  {'─'*32} {'─'*10}  {'─'*10}  {'─'*10}")

    def compare_row(label: str, live: Any, bt: Any, fmt_s: str = ".2f", invert: bool = False) -> None:
        if live is None:
            live_str, delta_str = "n/a", "n/a"
        else:
            live_str = format(live, fmt_s)
            delta = live - bt
            sign = "+" if delta >= 0 else ""
            delta_str = f"{sign}{delta:{fmt_s}}"
            if invert:
                delta_str = f"{'−' if delta >= 0 else '+'}{abs(delta):{fmt_s}}"
        bt_str = format(bt, fmt_s)
        print(f"  {label:<32} {live_str:>10}  {bt_str:>10}  {delta_str:>10}")

    compare_row("BUY signals/sym/day",   live_per_sym if buy_count > 0 else None,
                BACKTEST["signal_rate_per_symbol_per_day"])
    compare_row("Win rate",              pos["win_rate"], BACKTEST["win_rate"], ".0%")
    compare_row("Avg hold (bars)",       pos["avg_hold_bars"], BACKTEST["avg_hold_bars"])
    compare_row("Rejection rate",        sig["rejection_rate"], BACKTEST["rejection_rate"], ".0%")
    compare_row("Avg slippage (bps)",    ex["avg_slippage_bps"], BACKTEST["avg_slippage_bps"],
                invert=True)

    # ── Concerns ──────────────────────────────────────────────────
    section("CONCERNS")
    if not concerns:
        print("  No concerns. Live behavior is within expected parameters.")
    else:
        for i, c in enumerate(concerns, 1):
            print(f"  {i}. {c}")

    print(f"\n{'═' * 60}\n")


# ---------------------------------------------------------------------------
# JSON output (machine-readable)
# ---------------------------------------------------------------------------

def build_json_output(
    report_date: str,
    bar_q: dict,
    sig: dict,
    ex: dict,
    risk: dict,
    pos: dict,
    concerns: list[str],
) -> dict:
    return {
        "date":          report_date,
        "bar_quality":   bar_q,
        "signals":       sig,
        "execution":     ex,
        "risk":          risk,
        "positions":     pos,
        "concerns":      concerns,
        "backtest":      BACKTEST,
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Daily bot diagnostic report")
    parser.add_argument(
        "--date", default=date.today().strftime("%Y-%m-%d"),
        help="Trading date to analyse (YYYY-MM-DD). Defaults to today."
    )
    parser.add_argument(
        "--log-root", default="logs",
        help="Root directory containing per-date log subdirectories."
    )
    parser.add_argument(
        "--json", action="store_true",
        help="Output machine-readable JSON instead of the formatted report."
    )
    args = parser.parse_args()

    log_dir = Path(args.log_root) / args.date
    if not log_dir.exists():
        print(f"No log directory found for {args.date}: {log_dir}", file=sys.stderr)
        print("Has the bot run today?", file=sys.stderr)
        sys.exit(1)

    logs = load_logs(log_dir)

    bar_q    = compute_bar_quality(logs["bars"])
    sig      = compute_signal_behavior(logs["signals"])
    ex       = compute_execution(logs["execution"])
    risk_    = compute_risk(logs["risk"])
    pos      = compute_positions(logs["positions"])
    concerns = detect_concerns(bar_q, sig, ex, pos)

    if args.json:
        import json as _json
        output = build_json_output(args.date, bar_q, sig, ex, risk_, pos, concerns)
        print(_json.dumps(output, indent=2, default=str))
    else:
        print_report(args.date, bar_q, sig, ex, risk_, pos, concerns)


if __name__ == "__main__":
    main()

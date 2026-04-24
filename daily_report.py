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

DEFAULT_BACKTEST: dict[str, Any] = {
    "symbols":                         15,
    "bars_per_day":                    26,     # 390-min session ÷ 15-min bars
    "signal_rate_per_symbol_per_day":  2.1,    # avg BUY signals per symbol per session
    "win_rate":                        0.65,   # fraction of closed trades that are winners
    "avg_hold_bars":                   4.2,    # bars held per trade
    "rejection_rate":                  0.72,   # fraction of evaluated bars → HOLD
    "avg_slippage_bps":                5.0,    # expected adverse slippage (bps)
}


def _load_json_file(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def _to_ratio(value: Any) -> float | None:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return numeric / 100.0 if numeric > 1.0 else numeric


def _to_optional_bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "1", "yes"}:
            return True
        if lowered in {"false", "0", "no"}:
            return False
    return None


def _normalize_runtime_for_comparison(runtime: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(runtime, dict):
        return {}
    normalized: dict[str, Any] = {}
    for key in (
        "strategy_mode",
        "sma_bars",
        "entry_threshold_pct",
        "ml_probability_buy",
        "ml_probability_sell",
        "threshold_mode",
        "atr_multiple",
        "atr_percentile_threshold",
        "time_window_mode",
        "regime_filter_enabled",
        "orb_filter_mode",
        "breakout_exit_style",
        "breakout_tight_stop_fraction",
        "mean_reversion_exit_style",
        "mean_reversion_max_atr_percentile",
        "mean_reversion_trend_filter",
        "mean_reversion_trend_slope_filter",
        "mean_reversion_stop_pct",
        "symbols",
        "bar_timeframe_minutes",
    ):
        if key not in runtime:
            continue
        value = runtime[key]
        normalized[key] = list(value) if isinstance(value, list) else value
    return normalized


def _normalize_research_config_for_comparison(config: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(config, dict):
        return {}
    normalized: dict[str, Any] = {}
    for key in (
        "strategy_mode",
        "sma_bars",
        "entry_threshold_pct",
        "ml_probability_buy",
        "ml_probability_sell",
        "threshold_mode",
        "atr_multiple",
        "atr_percentile_threshold",
        "time_window_mode",
        "regime_filter_enabled",
        "orb_filter_mode",
        "breakout_exit_style",
        "breakout_tight_stop_fraction",
        "mean_reversion_exit_style",
        "mean_reversion_max_atr_percentile",
        "mean_reversion_trend_filter",
        "mean_reversion_trend_slope_filter",
        "mean_reversion_stop_pct",
    ):
        if key in config:
            normalized[key] = config[key]
    return normalized


def load_backtest_baseline(repo_root: Path | None = None) -> tuple[dict[str, Any], dict[str, Any]]:
    base = dict(DEFAULT_BACKTEST)
    source = {
        "mode": "defaults",
        "live_config_path": None,
        "research_config_path": None,
        "research_metrics_used": False,
        "live_config_approved": None,
        "research_approved": None,
        "research_matches_live_runtime": None,
        "validation_errors": [],
        "valid_for_comparison": True,
    }

    root = repo_root or Path(__file__).resolve().parent
    live_config = _load_json_file(root / "config" / "live_config.json")
    live_runtime: dict[str, Any] | None = None
    if live_config is not None:
        runtime = live_config.get("runtime")
        if isinstance(runtime, dict):
            live_runtime = runtime
            symbols = runtime.get("symbols")
            if isinstance(symbols, list) and symbols:
                base["symbols"] = len(symbols)
            try:
                timeframe_minutes = int(runtime.get("bar_timeframe_minutes"))
            except (TypeError, ValueError):
                timeframe_minutes = None
            if timeframe_minutes and timeframe_minutes > 0:
                base["bars_per_day"] = max(1, int((6.5 * 60) / timeframe_minutes))
            source["mode"] = "live_config"
            source["live_config_path"] = str(root / "config" / "live_config.json")
        live_source = live_config.get("source")
        if isinstance(live_source, dict) and "approved" in live_source:
            source["live_config_approved"] = _to_optional_bool(live_source.get("approved"))
            if source["live_config_approved"] is False:
                reasons = live_source.get("rejection_reasons") or []
                reasons_text = ", ".join(str(item) for item in reasons if str(item))
                detail = f"live config is approved=false"
                if reasons_text:
                    detail = f"{detail} ({reasons_text})"
                source["validation_errors"].append(detail)

    research = _load_json_file(root / "results" / "best_config_latest.json")
    if research is not None:
        source["research_config_path"] = str(root / "results" / "best_config_latest.json")
        if "approved" in research:
            source["research_approved"] = _to_optional_bool(research.get("approved"))
            if source["research_approved"] is False:
                reasons = research.get("rejection_reasons") or []
                reasons_text = ", ".join(str(item) for item in reasons if str(item))
                detail = f"research artifact is approved=false"
                if reasons_text:
                    detail = f"{detail} ({reasons_text})"
                source["validation_errors"].append(detail)

        research_config = _normalize_research_config_for_comparison(research.get("config"))
        live_runtime_cmp = _normalize_runtime_for_comparison(live_runtime)
        if live_runtime_cmp and research_config:
            mismatches: list[str] = []
            for key, research_value in research_config.items():
                if key not in live_runtime_cmp:
                    continue
                if live_runtime_cmp[key] != research_value:
                    mismatches.append(f"{key}: live={live_runtime_cmp[key]!r} research={research_value!r}")
            source["research_matches_live_runtime"] = len(mismatches) == 0
            if mismatches:
                source["validation_errors"].append(
                    "research config does not match live runtime (" + "; ".join(mismatches) + ")"
                )

        performance = research.get("performance")
        research_valid = (
            isinstance(performance, dict)
            and source["research_approved"] is not False
            and source["research_matches_live_runtime"] is not False
            and source["live_config_approved"] is not False
        )
        if isinstance(performance, dict) and research_valid:
            try:
                trades_per_day_value = float(performance.get("trades_per_day"))
            except (TypeError, ValueError):
                trades_per_day_value = None
            if trades_per_day_value is not None and base["symbols"] > 0:
                base["signal_rate_per_symbol_per_day"] = trades_per_day_value / base["symbols"]
            win_rate = _to_ratio(performance.get("win_rate"))
            if win_rate is not None:
                base["win_rate"] = win_rate
            source["research_metrics_used"] = True
            source["mode"] = "live_config+research" if source["mode"] == "live_config" else "research"

    source["valid_for_comparison"] = len(source["validation_errors"]) == 0

    return base, source


BACKTEST, BACKTEST_SOURCE = load_backtest_baseline()

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
    rows: list[dict[str, Any]] = []
    malformed_lines = 0
    total_lines = 0
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        total_lines += 1
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            malformed_lines += 1
            continue
        if isinstance(payload, dict):
            rows.append(payload)

    df = pd.DataFrame(rows)
    df.attrs["source_path"] = str(path)
    df.attrs["raw_line_count"] = total_lines
    df.attrs["malformed_line_count"] = malformed_lines
    df.attrs["loaded_row_count"] = len(rows)
    return df


def _dedupe(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        df.attrs["duplicate_row_count"] = 0
        df.attrs["unique_row_count"] = 0
        return df

    key_columns = [col for col in ("event", "trace") if col in df.columns]
    deduped = df.drop_duplicates(subset=key_columns or None, keep="first").copy()
    deduped.attrs.update(df.attrs)
    deduped.attrs["duplicate_row_count"] = len(df) - len(deduped)
    deduped.attrs["unique_row_count"] = len(deduped)
    return deduped


def load_logs(log_dir: Path) -> dict[str, pd.DataFrame]:
    return {
        "bars":      _dedupe(_load(log_dir / "bars.jsonl")),
        "signals":   _dedupe(_load(log_dir / "signals.jsonl")),
        "risk":      _dedupe(_load(log_dir / "risk.jsonl")),
        "execution": _dedupe(_load(log_dir / "execution.jsonl")),
        "positions": _dedupe(_load(log_dir / "positions.jsonl")),
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


def _session_signal_rate(sig: dict) -> tuple[float | None, float | None, float | None]:
    total_evaluated = sig.get("total_evaluated", 0) or 0
    buy_signals = sig.get("buy_signals", 0) or 0
    if total_evaluated <= 0 or BACKTEST["symbols"] <= 0:
        return None, None, None

    bars_per_symbol_seen = total_evaluated / BACKTEST["symbols"]
    if bars_per_symbol_seen <= 0:
        return None, None, None

    coverage_pct = bars_per_symbol_seen / BACKTEST["bars_per_day"]
    full_day_equiv = (buy_signals / BACKTEST["symbols"]) * (BACKTEST["bars_per_day"] / bars_per_symbol_seen)
    return bars_per_symbol_seen, coverage_pct, full_day_equiv


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
            "hold_count": 0, "by_symbol": {}, "rejections": {}, "rejection_rate": None,
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


def compute_log_health(logs: dict[str, pd.DataFrame]) -> dict:
    files: dict[str, dict[str, int]] = {}
    for name, df in logs.items():
        files[name] = {
            "raw_lines": int(df.attrs.get("raw_line_count", len(df))),
            "loaded_rows": int(df.attrs.get("loaded_row_count", len(df))),
            "unique_rows": int(df.attrs.get("unique_row_count", len(df))),
            "malformed_lines": int(df.attrs.get("malformed_line_count", 0)),
            "duplicate_rows": int(df.attrs.get("duplicate_row_count", 0)),
        }

    return {
        "files": files,
        "malformed_lines": sum(item["malformed_lines"] for item in files.values()),
        "duplicate_rows": sum(item["duplicate_rows"] for item in files.values()),
    }


# ---------------------------------------------------------------------------
# Concern detection
# ---------------------------------------------------------------------------

def detect_concerns(bar_q: dict, sig: dict, ex: dict, pos: dict, log_health: dict | None = None) -> list[str]:
    """
    Return a list of concern strings.  Empty list = everything looks normal.
    """
    concerns = []
    n_sym = BACKTEST["symbols"]
    if not BACKTEST_SOURCE.get("valid_for_comparison", True):
        validation_errors = BACKTEST_SOURCE.get("validation_errors") or []
        detail = "; ".join(str(item) for item in validation_errors if str(item)) or "baseline validation failed"
        concerns.append(f"BASELINE INVALID: {detail}")

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
    _, coverage_pct, full_day_signal_rate = _session_signal_rate(sig)
    if full_day_signal_rate is not None:
        live_rate = full_day_signal_rate
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
        if coverage_pct is not None and coverage_pct < 1.0:
            concerns.append(
                f"PARTIAL SESSION: observed {coverage_pct:.0%} of a full session "
                f"({sig['total_evaluated']} unique signal evaluations) — normalized signal-rate estimate used"
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

    if log_health:
        malformed = log_health.get("malformed_lines", 0)
        duplicates = log_health.get("duplicate_rows", 0)
        if malformed > 0:
            concerns.append(
                f"LOG PARSE WARNINGS: skipped {malformed} malformed JSONL line(s) — inspect raw logs if this day matters"
            )
        if duplicates > 0:
            concerns.append(
                f"RESTART DUPLICATES: removed {duplicates} duplicate log row(s) by trace/event before analysis"
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
    log_health: dict | None = None,
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
    row("Baseline source",       str(BACKTEST_SOURCE.get("mode", "defaults")))
    if not BACKTEST_SOURCE.get("valid_for_comparison", True):
        print("  Baseline validation:")
        for item in BACKTEST_SOURCE.get("validation_errors", []):
            print(f"    - {item}")
    avg_age = bar_q["avg_age_s"]
    row("Avg bar age (s)",        fmt(avg_age, ".1f"),
        flag=_flag(avg_age or 0, WARN["avg_bar_age_s"], low_is_bad=False) if avg_age else "")
    row("Stale bars",             fmt(bar_q["stale_count"]),
        flag=_flag(bar_q["stale_count"], WARN["stale_bar_count"] - 1, low_is_bad=False)
             if bar_q["stale_count"] > 0 else "")
    if log_health is not None:
        row("Malformed log lines", fmt(log_health.get("malformed_lines", 0)),
            flag="WARN" if log_health.get("malformed_lines", 0) > 0 else "")
        row("Duplicate rows removed", fmt(log_health.get("duplicate_rows", 0)),
            flag="WARN" if log_health.get("duplicate_rows", 0) > 0 else "")

    # ── Signal behavior ───────────────────────────────────────────
    section("SIGNAL BEHAVIOR")
    total_eval = sig["total_evaluated"]
    buy_count  = sig["buy_signals"]
    bars_per_symbol_seen, coverage_pct, live_per_sym = _session_signal_rate(sig)
    if live_per_sym is None:
        live_per_sym = buy_count / n_sym if n_sym > 0 else 0
    bt_per_sym   = BACKTEST["signal_rate_per_symbol_per_day"]
    sig_ratio    = live_per_sym / bt_per_sym if bt_per_sym > 0 else 0

    row("Total bars evaluated",   fmt(total_eval))
    row("BUY signals fired",      fmt(buy_count),
        f"({live_per_sym:.1f}/sym/day equiv  bt={bt_per_sym:.1f}  ratio={sig_ratio:.0%})",
        flag=_flag(sig_ratio, WARN["signal_rate_pct_of_bt"], low_is_bad=True)
             if buy_count > 0 else "")
    row("SELL signals fired",     fmt(sig["sell_signals"]))
    if coverage_pct is not None:
        row("Session coverage",    f"{coverage_pct:.0%}",
            f"(~{bars_per_symbol_seen:.1f}/{BACKTEST['bars_per_day']} bars per symbol observed)")
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
    log_health: dict | None = None,
) -> dict:
    return {
        "date":          report_date,
        "backtest_source": BACKTEST_SOURCE,
        "bar_quality":   bar_q,
        "signals":       sig,
        "execution":     ex,
        "risk":          risk,
        "positions":     pos,
        "concerns":      concerns,
        "log_health":    log_health or {},
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
    parser.add_argument(
        "--require-approved-baseline", action="store_true",
        help="Exit with an error if the report baseline is unapproved or mismatched."
    )
    args = parser.parse_args()

    log_dir = Path(args.log_root) / args.date
    if not log_dir.exists():
        print(f"No log directory found for {args.date}: {log_dir}", file=sys.stderr)
        print("Has the bot run today?", file=sys.stderr)
        sys.exit(1)

    logs = load_logs(log_dir)

    bar_q      = compute_bar_quality(logs["bars"])
    sig        = compute_signal_behavior(logs["signals"])
    ex         = compute_execution(logs["execution"])
    risk_      = compute_risk(logs["risk"])
    pos        = compute_positions(logs["positions"])
    log_health = compute_log_health(logs)
    concerns   = detect_concerns(bar_q, sig, ex, pos, log_health)
    if args.require_approved_baseline and not BACKTEST_SOURCE.get("valid_for_comparison", True):
        for item in BACKTEST_SOURCE.get("validation_errors", []):
            print(f"Baseline validation error: {item}", file=sys.stderr)
        sys.exit(2)

    if args.json:
        import json as _json
        output = build_json_output(args.date, bar_q, sig, ex, risk_, pos, concerns, log_health)
        print(_json.dumps(output, indent=2, default=str))
    else:
        print_report(args.date, bar_q, sig, ex, risk_, pos, concerns, log_health)


if __name__ == "__main__":
    main()

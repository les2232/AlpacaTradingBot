"""
run_research.py
---------------
Automated research pipeline:
  1. Fetch a fresh dataset via dataset_snapshotter.py
  2. Run a parameter sweep via backtest_runner.py
  3. Rank results into leaderboard CSVs
  4. Evaluate approval, stability, and market regime
  5. Write decision artifacts to /results

Usage:
    python run_research.py

Edit the SNAPSHOT CONFIG and SWEEP CONFIG sections below to change parameters.
All paths are resolved relative to this script's directory, not the shell CWD.

Primary outputs written under /results:
  - research_<window>d_<timestamp>.csv
  - leaderboard_<window>d_<timestamp>.csv
  - best_config_latest.json
  - stability_report.json
  - trade_decision.json

This script is intentionally different from run_compare_suite.ps1:
  - run_research.py is the configurable multi-window research workflow
  - run_compare_suite.ps1 is the fixed benchmark suite for repeatable comparisons
"""

import json
import subprocess
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------------------
# PATHS
# Resolved relative to this file so the script works from any working directory.
# ---------------------------------------------------------------------------

SCRIPT_DIR           = Path(__file__).resolve().parent
DATASETS_DIR         = SCRIPT_DIR / "datasets"
RESULTS_DIR          = SCRIPT_DIR / "results"
BEST_CONFIG_PATH      = RESULTS_DIR / "best_config_latest.json"
STABILITY_REPORT_PATH = RESULTS_DIR / "stability_report.json"
TRADE_DECISION_PATH   = RESULTS_DIR / "trade_decision.json"

# ---------------------------------------------------------------------------
# SNAPSHOT CONFIG
# ---------------------------------------------------------------------------

SNAPSHOT_SYMBOLS   = ["AAPL", "MSFT", "NVDA"]
SNAPSHOT_TIMEFRAME = "15Min"

# Lookback windows (in calendar days) run in order, shortest first.
# The longest window is treated as the primary result for approval and
# best_config_latest.json.  Add or remove entries freely.
VALIDATION_WINDOWS = [30, 60, 90]
SNAPSHOT_FEED          = "iex"        # choices: iex | sip | otc
SNAPSHOT_ADJUSTMENT    = "raw"        # choices: raw | split | all
SNAPSHOT_ALIGN_MODE    = "shared"     # choices: shared | none

# ---------------------------------------------------------------------------
# SWEEP CONFIG
# Each entry is (cli-flag-name-without-dashes, value).
# Add or remove entries here — nothing else needs to change.
# ---------------------------------------------------------------------------

SWEEP_PARAMS: list[tuple[str, str]] = [
    ("strategy-mode-list",        "sma,hybrid,breakout"),
    ("sma-bars-list",             "10,15,20,30"),
    ("entry-threshold-pct-list",  "0.001,0.0025,0.005"),
    ("time-window-mode-list",     "full_day,morning_only"),
]

# Fixed (non-sweep) backtest parameters.
FIXED_PARAMS: list[tuple[str, str]] = [
    ("starting-capital", "10000"),
    ("position-size",    "1000"),
]

# ---------------------------------------------------------------------------
# APPROVAL THRESHOLDS
# Applied to the single best config from the longest validation window.
# ---------------------------------------------------------------------------

APPROVAL_MIN_PROFIT_FACTOR    = 1.2   # profit_factor must be >= this
APPROVAL_MAX_DRAWDOWN_PCT     = 8.0   # max_drawdown_pct must be <= this
APPROVAL_MIN_TOTAL_RETURN_PCT = 2.0   # total_return_pct must be >= this
APPROVAL_MIN_TRADES_PER_DAY   = 0.5   # trades_per_day must be >= this

# ---------------------------------------------------------------------------
# STABILITY THRESHOLDS
# Applied to the best config from EVERY window.  Intentionally slightly
# looser than approval thresholds — shorter windows are noisier.
# ---------------------------------------------------------------------------

STABILITY_MIN_PROFIT_FACTOR = 1.1    # profit_factor must be >= this in all windows
STABILITY_MAX_DRAWDOWN_PCT  = 10.0   # max_drawdown_pct must be <= this in all windows

# Parameters that must match across all windows for the config to be stable.
# ── VERIFY THESE match CONFIG_COLS ────────────────────────────────────────
STABILITY_PARAMS = [
    "strategy_mode",
    "sma_bars",
    "entry_threshold_pct",
    "time_window_mode",
]

# ---------------------------------------------------------------------------
# REGIME CONFIG
# Computed from the primary (longest) dataset's close prices.
# All thresholds are expressed in raw per-bar units (no annualisation).
#   Calibration reference (15-Min bars, SIP feed):
#     rolling vol  median ≈ 0.00185  p75 ≈ 0.00260  max ≈ 0.013
#     sma slope    sideways market ≈ -0.002 to -0.007 (well below 0.02)
# ---------------------------------------------------------------------------

REGIME_VOL_PERIOD         = 20      # bars in the rolling volatility window
REGIME_TREND_SMA_PERIOD   = 50      # bars in the trend SMA
REGIME_VOL_HIGH_THRESHOLD = 0.003   # rolling std > this  → high_volatility
REGIME_SLOPE_THRESHOLD    = 0.02    # |SMA % change over SMA period| > this → trending

# Regimes listed here disable trading regardless of approval / stability.
# Remove "high_volatility" or add "sideways" to adjust behaviour.
REGIME_DISABLE_ON: set[str] = {"high_volatility"}


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def _log(msg: str) -> None:
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def _section(title: str) -> None:
    """Print a visible section header to break up dense subprocess output."""
    bar = "─" * 60
    print(f"\n{bar}", flush=True)
    print(f"  {title}", flush=True)
    print(bar, flush=True)


# ---------------------------------------------------------------------------
# Path utilities
# ---------------------------------------------------------------------------

def _ensure_dir(path: Path) -> Path:
    """Create directory if absent and return it."""
    path.mkdir(parents=True, exist_ok=True)
    return path


def _timestamped_csv(results_dir: Path, prefix: str) -> Path:
    """Return results/<prefix>_YYYYMMDD_HHMMSS.csv (directory is created here)."""
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return _ensure_dir(results_dir) / f"{prefix}_{stamp}.csv"


def _newest_dataset(datasets_dir: Path) -> Path:
    """Return the most-recently-modified subdirectory of datasets_dir."""
    if not datasets_dir.is_dir():
        raise RuntimeError(f"Datasets directory not found: '{datasets_dir}'")
    candidates = [p for p in datasets_dir.iterdir() if p.is_dir()]
    if not candidates:
        raise RuntimeError(
            f"No dataset folders found in '{datasets_dir}'. "
            "Make sure the snapshot step completed successfully."
        )
    return max(candidates, key=lambda p: p.stat().st_mtime)


# ---------------------------------------------------------------------------
# Subprocess utilities
# ---------------------------------------------------------------------------

def _params_to_flags(params: list[tuple[str, str]]) -> list[str]:
    """Convert [(flag, value), ...] → ['--flag', 'value', ...]."""
    flags: list[str] = []
    for flag, value in params:
        flags += [f"--{flag}", value]
    return flags


def _run(cmd: list[str], *, label: str) -> None:
    """
    Run a subprocess, streaming its output live.
    Raises RuntimeError with a clear message on non-zero exit.
    cmd[0] must be an executable path or name — no shell expansion is performed.
    """
    _log(f"Command: {' '.join(cmd)}")
    print()
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(
            f"Step '{label}' failed (exit code {exc.returncode}).\n"
            "See subprocess output above for details."
        ) from exc
    print()
    _log(f"Completed: {label}")


# ---------------------------------------------------------------------------
# Date utilities
# ---------------------------------------------------------------------------

def _snapshot_date_range(lookback_days: int) -> tuple[str, str]:
    """
    Return (start, end) as UTC timestamp strings accepted by dataset_snapshotter.py.
    end  = today at UTC midnight
    start = end − lookback_days
    """
    fmt = "%Y-%m-%dT%H:%M:%SZ"
    end   = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
    start = end - timedelta(days=lookback_days)
    return start.strftime(fmt), end.strftime(fmt)


# ---------------------------------------------------------------------------
# Pipeline steps
# ---------------------------------------------------------------------------

def step_snapshot(lookback_days: int) -> Path:
    """
    Fetch a dataset for the given lookback window via dataset_snapshotter.py.
    Returns the Path of the newly created dataset directory.
    Uses set-difference (before vs. after) to identify the new folder reliably
    even when multiple snapshots run back-to-back.
    """
    # Record what already exists so we can identify the new folder afterwards.
    existing: set[Path] = (
        {p for p in DATASETS_DIR.iterdir() if p.is_dir()}
        if DATASETS_DIR.is_dir() else set()
    )

    start_str, end_str = _snapshot_date_range(lookback_days)

    cmd = [
        sys.executable, str(SCRIPT_DIR / "dataset_snapshotter.py"),
        "--symbols",    *SNAPSHOT_SYMBOLS,
        "--start",      start_str,
        "--end",        end_str,
        "--timeframe",  SNAPSHOT_TIMEFRAME,
        "--feed",       SNAPSHOT_FEED,
        "--adjustment", SNAPSHOT_ADJUSTMENT,
        "--align-mode", SNAPSHOT_ALIGN_MODE,
        "--output-dir", str(DATASETS_DIR),
    ]

    _run(cmd, label=f"Dataset snapshot ({lookback_days}d)")

    # Identify the newly created dataset folder.
    current = {p for p in DATASETS_DIR.iterdir() if p.is_dir()}
    new_dirs = current - existing
    if len(new_dirs) == 1:
        return new_dirs.pop()
    if len(new_dirs) > 1:
        # More than one new folder — pick the newest by mtime.
        return max(new_dirs, key=lambda p: p.stat().st_mtime)
    # Fallback: snapshotter re-used an existing folder (same date range + hash).
    return _newest_dataset(DATASETS_DIR)


def step_backtest(output_csv: Path, dataset_path: Path) -> None:
    """Run the parameter sweep backtest against dataset_path."""
    _log(f"Using dataset: {dataset_path.name}")

    cmd = [
        sys.executable, str(SCRIPT_DIR / "backtest_runner.py"),
        "--dataset",    str(dataset_path),
        "--output-csv", str(output_csv),
        *_params_to_flags(SWEEP_PARAMS),
        *_params_to_flags(FIXED_PARAMS),
    ]

    _run(cmd, label="Backtest sweep")


# ---------------------------------------------------------------------------
# Post-processing: ranking and leaderboard
# ---------------------------------------------------------------------------

# Columns used for ranking and approval.
# ── VERIFY THESE if you change backtest_runner.py's output schema ──────────
COL_RETURN         = "total_return_pct"   # higher is better
COL_PF             = "profit_factor"      # higher is better
COL_DRAWDOWN       = "max_drawdown_pct"   # lower  is better
COL_TRADES_PER_DAY = "trades_per_day"     # higher is better

# Config columns saved to best_config_latest.json and shown in the top-5 table.
# ── VERIFY THESE match the CSV header ─────────────────────────────────────
CONFIG_COLS = [
    "strategy_mode",
    "sma_bars",
    "entry_threshold_pct",
    "threshold_mode",
    "time_window_mode",
    "regime_filter_enabled",
    "orb_filter_mode",
    "breakout_exit_style",
    "mean_reversion_exit_style",
]

METRIC_COLS = [
    COL_RETURN,
    COL_PF,
    COL_DRAWDOWN,
    "sharpe_ratio",
    "win_rate",
    "trades_per_day",
    "realized_pnl",
    "combined_score",
]

DISPLAY_COLS = [
    "strategy_mode",
    "sma_bars",
    "entry_threshold_pct",
    "time_window_mode",
    COL_RETURN,
    COL_PF,
    COL_DRAWDOWN,
    "combined_score",
]


def step_rank_results(sweep_csv: Path, leaderboard_csv: Path) -> dict:
    """
    Load sweep_csv, compute per-metric ranks and a combined score,
    save a leaderboard CSV, print the top 5 rows, and return the best
    row as a plain dict for downstream use.

    combined_score = mean of the three per-metric ranks (lower is better).
    """
    df = pd.read_csv(sweep_csv)

    # Validate that expected columns exist before proceeding.
    missing = [c for c in (COL_RETURN, COL_PF, COL_DRAWDOWN) if c not in df.columns]
    if missing:
        raise RuntimeError(
            f"Ranking columns not found in {sweep_csv.name}: {missing}\n"
            f"Available columns: {list(df.columns)}"
        )

    # Rank each metric independently (method='min' gives ties the same rank).
    df["rank_return"]   = df[COL_RETURN].rank(ascending=False, method="min")
    df["rank_pf"]       = df[COL_PF].rank(ascending=False, method="min")
    df["rank_drawdown"] = df[COL_DRAWDOWN].rank(ascending=True, method="min")

    df["combined_score"] = (
        df["rank_return"] + df["rank_pf"] + df["rank_drawdown"]
    ) / 3.0

    df_sorted = df.sort_values("combined_score").reset_index(drop=True)
    df_sorted.to_csv(leaderboard_csv, index=False)
    _log(f"Leaderboard : {leaderboard_csv}")

    # Print top 5 using only columns that actually exist in the output.
    top5_cols = [c for c in DISPLAY_COLS if c in df_sorted.columns]
    top5 = df_sorted.head(5)[top5_cols].copy()

    # Round floats for readability.
    for col in (COL_RETURN, COL_PF, COL_DRAWDOWN, "combined_score"):
        if col in top5.columns:
            top5[col] = top5[col].round(4)

    print()
    print("  Top 5 configurations by combined score (lower = better)")
    print("  " + "-" * 70)
    print(top5.to_string(index=True))
    print()

    return df_sorted.iloc[0].to_dict()


def step_approve_config(best_row: dict) -> tuple[bool, list[str]]:
    """
    Evaluate the best-ranked config against the approval thresholds defined
    at the top of this file.

    Each check is: (display_label, actual_value, threshold, passed, description).
    Returns (approved, rejection_reasons) where rejection_reasons is empty when approved.
    """
    approval_cols = [COL_PF, COL_DRAWDOWN, COL_RETURN, COL_TRADES_PER_DAY]
    missing = [c for c in approval_cols if c not in best_row]
    if missing:
        raise RuntimeError(
            f"Cannot evaluate approval — columns missing from best row: {missing}"
        )

    # Each tuple: (label, actual, threshold, passed, human-readable condition)
    checks = [
        (
            COL_PF,
            best_row[COL_PF],
            APPROVAL_MIN_PROFIT_FACTOR,
            best_row[COL_PF] >= APPROVAL_MIN_PROFIT_FACTOR,
            f"profit_factor {best_row[COL_PF]:.4f} >= {APPROVAL_MIN_PROFIT_FACTOR}",
        ),
        (
            COL_DRAWDOWN,
            best_row[COL_DRAWDOWN],
            APPROVAL_MAX_DRAWDOWN_PCT,
            best_row[COL_DRAWDOWN] <= APPROVAL_MAX_DRAWDOWN_PCT,
            f"max_drawdown_pct {best_row[COL_DRAWDOWN]:.4f}% <= {APPROVAL_MAX_DRAWDOWN_PCT}%",
        ),
        (
            COL_RETURN,
            best_row[COL_RETURN],
            APPROVAL_MIN_TOTAL_RETURN_PCT,
            best_row[COL_RETURN] >= APPROVAL_MIN_TOTAL_RETURN_PCT,
            f"total_return_pct {best_row[COL_RETURN]:.4f}% >= {APPROVAL_MIN_TOTAL_RETURN_PCT}%",
        ),
        (
            COL_TRADES_PER_DAY,
            best_row[COL_TRADES_PER_DAY],
            APPROVAL_MIN_TRADES_PER_DAY,
            best_row[COL_TRADES_PER_DAY] >= APPROVAL_MIN_TRADES_PER_DAY,
            f"trades_per_day {best_row[COL_TRADES_PER_DAY]:.4f} >= {APPROVAL_MIN_TRADES_PER_DAY}",
        ),
    ]

    rejection_reasons = [desc for _, _, _, passed, desc in checks if not passed]
    approved = len(rejection_reasons) == 0

    # Console summary
    verdict = "APPROVED" if approved else "REJECTED"
    print()
    print(f"  Approval verdict: {verdict}")
    print("  " + "-" * 55)
    for _, _, _, passed, desc in checks:
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] {desc}")
    print()

    return approved, rejection_reasons


def step_save_best_config(
    best_row: dict,
    dataset_path: Path,
    output_path: Path,
    approved: bool,
    rejection_reasons: list[str],
) -> None:
    """
    Extract config and performance fields from best_row and write a
    human-readable JSON file to output_path.
    """
    # Check that all expected columns are present before building the dict.
    all_expected = CONFIG_COLS + [c for c in METRIC_COLS if c != "combined_score"]
    missing = [c for c in all_expected if c not in best_row]
    if missing:
        raise RuntimeError(
            f"Cannot save best config — columns missing from leaderboard row: {missing}"
        )

    config = {col: best_row[col] for col in CONFIG_COLS}
    performance = {
        col: round(best_row[col], 6)
        for col in METRIC_COLS
        if col in best_row
    }

    payload = {
        "saved_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "dataset": str(dataset_path),
        "approved": approved,
        "rejection_reasons": rejection_reasons,
        "config": config,
        "performance": performance,
    }

    _ensure_dir(output_path.parent)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    _log(f"Best config : {output_path}")


# ---------------------------------------------------------------------------
# Market regime detection
# ---------------------------------------------------------------------------

def _classify_regime_for_symbol(closes: pd.Series) -> dict:
    """
    Classify one symbol's current market regime from its close price series.
    Returns a dict with regime label and the two numeric signals.

    Classification priority (first match wins):
      1. high_volatility — rolling std of returns > REGIME_VOL_HIGH_THRESHOLD
      2. trending        — |SMA % change over SMA period| > REGIME_SLOPE_THRESHOLD
      3. sideways        — everything else
    """
    returns     = closes.pct_change().dropna()
    rolling_vol = returns.rolling(REGIME_VOL_PERIOD).std()
    current_vol = rolling_vol.iloc[-1] if len(rolling_vol) >= REGIME_VOL_PERIOD else float("nan")

    sma = closes.rolling(REGIME_TREND_SMA_PERIOD).mean().dropna()
    if len(sma) >= REGIME_TREND_SMA_PERIOD:
        # Total % change of the SMA over the last REGIME_TREND_SMA_PERIOD bars.
        slope_pct = float((sma.iloc[-1] - sma.iloc[-REGIME_TREND_SMA_PERIOD]) / sma.iloc[-REGIME_TREND_SMA_PERIOD])
    else:
        slope_pct = 0.0

    if pd.isna(current_vol):
        regime = "unknown"
    elif current_vol > REGIME_VOL_HIGH_THRESHOLD:
        regime = "high_volatility"
    elif abs(slope_pct) > REGIME_SLOPE_THRESHOLD:
        regime = "trending"
    else:
        regime = "sideways"

    return {
        "regime":             regime,
        "current_volatility": round(float(current_vol), 6) if not pd.isna(current_vol) else None,
        "sma_slope_pct":      round(slope_pct, 6),
    }


def step_detect_regime(dataset_path: Path) -> dict:
    """
    Load the primary dataset, classify each symbol's regime, and return
    an aggregate summary.  Majority vote determines the overall label.
    """
    bars_path = dataset_path / "bars.parquet"
    if not bars_path.exists():
        raise RuntimeError(f"Cannot detect regime — bars file not found: {bars_path}")

    df = pd.read_parquet(bars_path)
    required = {"symbol", "timestamp", "close"}
    missing  = required - set(df.columns)
    if missing:
        raise RuntimeError(f"Regime detection requires columns {missing} — not in dataset")

    per_symbol: dict[str, dict] = {}
    for symbol, grp in df.groupby("symbol"):
        closes = grp.sort_values("timestamp")["close"]
        per_symbol[symbol] = _classify_regime_for_symbol(closes)

    if not per_symbol:
        raise RuntimeError("Regime detection found no symbols in dataset")

    # --- aggregate across symbols (majority vote) --------------------------
    labels = [r["regime"] for r in per_symbol.values() if r["regime"] != "unknown"]
    if not labels:
        overall = "unknown"
    elif labels.count("high_volatility") > len(labels) / 2:
        overall = "high_volatility"
    elif labels.count("trending") > len(labels) / 2:
        overall = "trending"
    else:
        overall = "sideways"

    valid_vols  = [r["current_volatility"] for r in per_symbol.values() if r["current_volatility"] is not None]
    avg_vol     = round(sum(valid_vols) / len(valid_vols), 6) if valid_vols else None
    avg_slope   = round(sum(r["sma_slope_pct"] for r in per_symbol.values()) / len(per_symbol), 6)
    disabled    = overall in REGIME_DISABLE_ON

    # --- console summary ---------------------------------------------------
    print()
    print(f"  Market regime: {overall.upper()}")
    print("  " + "-" * 55)
    for sym, info in sorted(per_symbol.items()):
        vol_str = f"{info['current_volatility']:.5f}" if info["current_volatility"] is not None else "n/a"
        print(f"  {sym:>6}  {info['regime']:<16}  vol={vol_str}  slope={info['sma_slope_pct']:+.4f}")
    vol_str = f"{avg_vol:.5f}" if avg_vol is not None else "n/a"
    print(f"\n  Avg vol={vol_str}  avg slope={avg_slope:+.4f}")
    if disabled:
        print(f"  [WARN] Regime '{overall}' is in REGIME_DISABLE_ON — trading will be blocked")
    print()

    return {
        "regime":            overall,
        "disabled_by_regime": disabled,
        "avg_volatility":    avg_vol,
        "avg_sma_slope_pct": avg_slope,
        "per_symbol":        per_symbol,
    }


# ---------------------------------------------------------------------------
# Multi-window stability evaluation
# ---------------------------------------------------------------------------

def step_evaluate_stability(window_results: list[dict]) -> dict:
    """
    Compare best configs across all validation windows and return a stability
    summary dict.

    A config is STABLE when:
      - profit_factor >= STABILITY_MIN_PROFIT_FACTOR in every window
      - max_drawdown_pct <= STABILITY_MAX_DRAWDOWN_PCT in every window
      - all STABILITY_PARAMS are identical across every window
    """
    best_rows = [w["best_row"] for w in window_results]

    # --- metric checks (must hold in every window) -------------------------
    pf_failures = [
        f"{w['lookback_days']}d window: profit_factor {w['best_row'][COL_PF]:.4f}"
        f" < {STABILITY_MIN_PROFIT_FACTOR}"
        for w in window_results
        if w["best_row"].get(COL_PF, 0) < STABILITY_MIN_PROFIT_FACTOR
    ]

    dd_failures = [
        f"{w['lookback_days']}d window: max_drawdown_pct {w['best_row'][COL_DRAWDOWN]:.4f}%"
        f" > {STABILITY_MAX_DRAWDOWN_PCT}%"
        for w in window_results
        if w["best_row"].get(COL_DRAWDOWN, 999) > STABILITY_MAX_DRAWDOWN_PCT
    ]

    # --- parameter consistency check ---------------------------------------
    # For each stability param, collect the value seen in each window.
    param_values: dict[str, list] = {}
    for param in STABILITY_PARAMS:
        vals = [row.get(param) for row in best_rows]
        if any(v is not None for v in vals):
            param_values[param] = vals

    # A parameter is inconsistent if it takes more than one distinct value.
    inconsistent_params: dict[str, list] = {
        param: vals
        for param, vals in param_values.items()
        if len({str(v) for v in vals}) > 1
    }
    param_failures = [
        f"Inconsistent {param}: {vals}"
        for param, vals in inconsistent_params.items()
    ]

    all_reasons  = pf_failures + dd_failures + param_failures
    stable       = len(all_reasons) == 0
    params_consistent = len(inconsistent_params) == 0

    # --- average metrics across windows ------------------------------------
    avg_metric_keys = [
        COL_RETURN, COL_PF, COL_DRAWDOWN,
        "sharpe_ratio", "win_rate", "trades_per_day", "realized_pnl",
    ]
    average_metrics: dict[str, float] = {}
    for key in avg_metric_keys:
        vals = [row[key] for row in best_rows if key in row]
        if vals:
            average_metrics[key] = round(sum(vals) / len(vals), 6)

    # --- console summary ---------------------------------------------------
    verdict = "STABLE" if stable else "UNSTABLE"
    print()
    print(f"  Stability verdict: {verdict}")
    print("  " + "-" * 55)
    for w in window_results:
        row = w["best_row"]
        pf  = row.get(COL_PF, float("nan"))
        dd  = row.get(COL_DRAWDOWN, float("nan"))
        ret = row.get(COL_RETURN, float("nan"))
        sm  = row.get("strategy_mode", "?")
        sma = row.get("sma_bars", "?")
        thr = row.get("entry_threshold_pct", "?")
        print(
            f"  {w['lookback_days']:>3}d  strategy={sm} sma={sma} thresh={thr}"
            f"  pf={pf:.4f}  dd={dd:.4f}%  ret={ret:.4f}%"
        )
    if all_reasons:
        print()
        for reason in all_reasons:
            print(f"  [FAIL] {reason}")
    else:
        print()
        print("  [PASS] All windows meet metric thresholds")
        print(f"  [{'PASS' if params_consistent else 'FAIL'}] Parameters consistent across windows")
    print()

    return {
        "stable": stable,
        "reasons": all_reasons,
        "params_consistent": params_consistent,
        "inconsistent_params": {k: [str(v) for v in vs] for k, vs in inconsistent_params.items()},
        "average_metrics": average_metrics,
    }


def step_save_stability_report(
    window_results: list[dict],
    stability: dict,
    regime: dict,
    output_path: Path,
) -> None:
    """Write the full stability report to a JSON file."""
    per_window = []
    for w in window_results:
        row = w["best_row"]
        per_window.append({
            "lookback_days": w["lookback_days"],
            "dataset": str(w["dataset"]),
            "config":      {col: row[col] for col in CONFIG_COLS if col in row},
            "performance": {
                col: round(row[col], 6) for col in METRIC_COLS if col in row
            },
        })

    payload = {
        "saved_at":           datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "validation_windows": [w["lookback_days"] for w in window_results],
        "per_window":         per_window,
        "average_performance": stability["average_metrics"],
        "stability": {
            "stable":              stability["stable"],
            "reasons":             stability["reasons"],
            "params_consistent":   stability["params_consistent"],
            "inconsistent_params": stability["inconsistent_params"],
        },
        "market_regime": {
            "regime":             regime["regime"],
            "disabled_by_regime": regime["disabled_by_regime"],
            "avg_volatility":     regime["avg_volatility"],
            "avg_sma_slope_pct":  regime["avg_sma_slope_pct"],
            "per_symbol":         regime["per_symbol"],
        },
    }

    _ensure_dir(output_path.parent)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    _log(f"Stability report: {output_path}")


# ---------------------------------------------------------------------------
# Final trade decision
# ---------------------------------------------------------------------------

def step_make_trade_decision(
    approved: bool,
    rejection_reasons: list[str],
    stability: dict,
    regime: dict,
    best_row: dict,
) -> dict:
    """
    Combine approval, stability, and market regime into a single actionable decision.

    should_trade = approved AND stable AND (regime not in REGIME_DISABLE_ON)
    All three gates must pass. Returns a plain dict — no file I/O here.
    """
    stable            = stability["stable"]
    stability_reasons = stability["reasons"]
    disabled_by_regime = regime["disabled_by_regime"]

    should_trade = approved and stable and not disabled_by_regime

    # Console summary
    verdict = "TRADE ENABLED" if should_trade else "TRADE DISABLED"
    border  = "*" * (len(verdict) + 6)
    print()
    print(f"  {border}")
    print(f"  ** {verdict} **")
    print(f"  {border}")

    if should_trade:
        print(f"  Strategy approved, stable, and regime is '{regime['regime']}' (allowed).")
    else:
        print("  Reason(s) trading is disabled:")
        if not approved:
            for r in rejection_reasons:
                print(f"    [approval]   {r}")
        if not stable:
            for r in stability_reasons:
                print(f"    [stability]  {r}")
        if disabled_by_regime:
            print(f"    [regime]     '{regime['regime']}' is in REGIME_DISABLE_ON")
    print()

    perf_keys = [COL_RETURN, COL_PF, COL_DRAWDOWN, "sharpe_ratio", "trades_per_day"]
    performance = {k: round(best_row[k], 6) for k in perf_keys if k in best_row}

    return {
        "timestamp":           datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "should_trade":        should_trade,
        "approved":            approved,
        "stable":              stable,
        "disabled_by_regime":  disabled_by_regime,
        "rejection_reasons":   rejection_reasons,
        "stability_reasons":   stability_reasons,
        "market_regime":       regime["regime"],
        "regime_avg_vol":      regime["avg_volatility"],
        "regime_avg_slope_pct": regime["avg_sma_slope_pct"],
        "selected_config":     {col: best_row[col] for col in CONFIG_COLS if col in best_row},
        "performance":         performance,
    }


def step_save_trade_decision(decision: dict, output_path: Path) -> None:
    """Write trade_decision.json.  Always overwrites so there is one canonical file."""
    _ensure_dir(output_path.parent)
    output_path.write_text(json.dumps(decision, indent=2), encoding="utf-8")
    _log(f"Trade decision  : {output_path}")


def _report_outputs(output_csv: Path) -> None:
    """Print paths of all CSV files the backtest runner produced."""
    # backtest_runner.py appends these suffixes when --output-csv is given.
    side_cars = ["_per_symbol", "_robust_top10", "_winner_by_symbol"]
    _log(f"Primary CSV : {output_csv}")
    for suffix in side_cars:
        path = output_csv.with_name(output_csv.stem + suffix + ".csv")
        if path.exists():
            _log(f"Side-car    : {path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    _section("run_research.py  —  starting")

    n = len(VALIDATION_WINDOWS)
    _log(f"Validation windows : {VALIDATION_WINDOWS} days ({n} windows)")
    _log(f"Best config target : {BEST_CONFIG_PATH}")
    _log(f"Stability report   : {STABILITY_REPORT_PATH}")
    _log(f"Trade decision     : {TRADE_DECISION_PATH}")

    window_results: list[dict] = []
    primary_sweep_csv: Path | None = None   # longest window — used for _report_outputs

    try:
        # ── Phase 1: run snapshot + backtest + rank for every window ──────
        for i, days in enumerate(VALIDATION_WINDOWS, 1):
            _section(f"Window {i} of {n} ({days} days) — Snapshot")
            dataset_path = step_snapshot(days)

            sweep_csv       = _timestamped_csv(RESULTS_DIR, f"research_{days}d")
            leaderboard_csv = _timestamped_csv(RESULTS_DIR, f"leaderboard_{days}d")

            _section(f"Window {i} of {n} ({days} days) — Backtest")
            step_backtest(sweep_csv, dataset_path)

            _section(f"Window {i} of {n} ({days} days) — Rank")
            best_row = step_rank_results(sweep_csv, leaderboard_csv)

            window_results.append({
                "lookback_days": days,
                "dataset":       dataset_path,
                "sweep_csv":     sweep_csv,
                "best_row":      best_row,
            })

            if days == VALIDATION_WINDOWS[-1]:
                primary_sweep_csv = sweep_csv

        # ── Phase 2: approve using the longest (most data-rich) window ────
        primary = window_results[-1]

        _section(f"Approval — {primary['lookback_days']}-day window (primary)")
        approved, rejection_reasons = step_approve_config(primary["best_row"])

        _section("Save best config")
        step_save_best_config(
            primary["best_row"], primary["dataset"],
            BEST_CONFIG_PATH, approved, rejection_reasons,
        )

        # ── Phase 3: stability evaluation across all windows ──────────────
        _section("Stability evaluation")
        stability = step_evaluate_stability(window_results)

        # ── Phase 4: market regime detection (primary dataset) ────────────
        _section("Market regime detection")
        regime = step_detect_regime(primary["dataset"])

        step_save_stability_report(window_results, stability, regime, STABILITY_REPORT_PATH)

        # ── Phase 5: combine into a single trade decision ─────────────────
        _section("Trade decision")
        decision = step_make_trade_decision(
            approved, rejection_reasons, stability, regime, primary["best_row"]
        )
        step_save_trade_decision(decision, TRADE_DECISION_PATH)

    except RuntimeError as exc:
        _section("FAILED")
        _log(f"Error: {exc}")
        sys.exit(1)

    _section("Done")
    if primary_sweep_csv:
        _report_outputs(primary_sweep_csv)


if __name__ == "__main__":
    main()

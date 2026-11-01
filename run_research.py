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
    python run_research.py --dataset datasets\\YOUR_DATASET

Edit the SNAPSHOT CONFIG and SWEEP CONFIG sections below to change parameters.
All paths are resolved relative to this script's directory, not the shell CWD.

Primary outputs written under /results:
  - research_<window>d_<timestamp>.csv
  - leaderboard_<window>d_<timestamp>.csv
  - best_config_latest.json
  - stability_report.json
  - trade_decision.json
  - drift_profile.json

This script is intentionally different from run_compare_suite.ps1:
  - run_research.py is the configurable multi-window research workflow
  - run_compare_suite.ps1 is the fixed benchmark suite for repeatable comparisons
"""

import argparse
import io
import json
import hashlib
import os
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Callable, Mapping
from uuid import uuid4

import pandas as pd
from dataset_integrity import load_verified_manifest
from daily_report import DEFAULT_BACKTEST, WARN
from dotenv import dotenv_values
from research.experiment_log import log_experiment_run
from tradeos.research_bundle import (
    ApprovalPolicy,
    BacktestCanonicalStrategy,
    BacktestModelExecutionEvidence,
    DRIFT_PROFILE_POLICY_V1,
    DRIFT_PROFILE_SCHEMA_NAME,
    DRIFT_PROFILE_SCHEMA_VERSION,
    RegimePolicy,
    ResearchBundle,
    ResearchBundleValidationError,
    ResearchPublicationError,
    RISK_EXECUTION_CONTEXT_FIELDS,
    StabilityPolicy,
    assert_dataset_checkpoint_unchanged,
    assert_source_identity_unchanged,
    build_backtest_selection_evidence,
    build_dataset_provenance_set,
    build_latest_research_state,
    build_live_config_payload,
    build_promotion_pointer,
    build_research_config_identity,
    build_research_bundle,
    canonical_strategy_from_backtest_row,
    capture_dataset_checkpoint,
    capture_dataset_provenance,
    capture_source_identity,
    compute_research_generation_sha256,
    evaluate_approval_policy,
    evaluate_stability_policy,
    generation_directory_component,
    json_file_bytes,
    load_verified_promotion_pointer,
    model_execution_evidence_from_backtest_row,
    normalize_metrics,
    publish_files_atomically,
    resolve_risk_execution_context,
    strict_json_loads,
    validate_backtest_selection_artifacts,
    verify_latest_research_state_bytes,
    write_immutable_bundle,
    write_immutable_generation,
)
from tradeos.execution_lock import ExecutionLock
from tradeos.strategy_contract import PROMOTED_STRATEGY_FIELDS
import strategy_report

# ---------------------------------------------------------------------------
# PATHS
# Resolved relative to this file so the script works from any working directory.
# ---------------------------------------------------------------------------

SCRIPT_DIR           = Path(__file__).resolve().parent
DATASETS_DIR         = SCRIPT_DIR / "datasets"
RESULTS_DIR          = SCRIPT_DIR / "results"
CONFIG_DIR           = SCRIPT_DIR / "config"
BEST_CONFIG_PATH      = RESULTS_DIR / "best_config_latest.json"
STABILITY_REPORT_PATH = RESULTS_DIR / "stability_report.json"
TRADE_DECISION_PATH   = RESULTS_DIR / "trade_decision.json"
LIVE_CONFIG_PATH      = CONFIG_DIR / "live_config.json"
DRIFT_PROFILE_PATH    = RESULTS_DIR / "drift_profile.json"
PROMOTION_BUNDLE_DIR  = RESULTS_DIR / "promotions"
PROMOTION_GENERATION_DIR = PROMOTION_BUNDLE_DIR / "generations"
PROMOTION_POINTER_PATH = RESULTS_DIR / "promotion_current.json"
LATEST_RESEARCH_STATE_PATH = RESULTS_DIR / "research_latest.json"
RESEARCH_LOCK_PATH = SCRIPT_DIR / ".research_pipeline.lock"

APPROVAL_POLICY_NAME    = "tradeos.primary_research_approval"
APPROVAL_POLICY_VERSION = 1
STABILITY_POLICY_NAME    = "tradeos.multi_window_stability"
STABILITY_POLICY_VERSION = 1
REGIME_POLICY_NAME       = "tradeos.market_regime_gate"
REGIME_POLICY_VERSION    = 1


def _load_live_runtime_defaults(
    live_config_path: Path,
    *,
    dotenv_path: Path | None = None,
    process_environment: Mapping[str, str] | None = None,
) -> dict[str, object]:
    path = Path(live_config_path)
    if not path.exists():
        raise RuntimeError(f"Live config not found: {path}")
    try:
        payload = strict_json_loads(
            path.read_text(encoding="utf-8"),
            label="research live config",
        )
    except (OSError, UnicodeError) as exc:
        raise RuntimeError(f"Live config is unreadable: {path}") from exc
    if not isinstance(payload, dict):
        raise RuntimeError(f"Live config must contain an object: {path}")
    runtime = payload.get("runtime")
    if not isinstance(runtime, dict):
        raise RuntimeError(f"Live config is missing a 'runtime' object: {path}")

    environment_file = dotenv_path if dotenv_path is not None else SCRIPT_DIR / ".env"
    try:
        dotenv_environment = dotenv_values(environment_file)
    except (OSError, UnicodeError) as exc:
        raise RuntimeError("Research dotenv inputs are unreadable.") from exc
    effective_environment: dict[str, object] = {
        key: value
        for key, value in dotenv_environment.items()
        if type(key) is str and value is not None
    }
    effective_environment.update(
        os.environ if process_environment is None else process_environment
    )

    symbols = runtime.get("symbols")
    if not isinstance(symbols, list) or not symbols:
        raise RuntimeError(f"Live config is missing runtime.symbols: {path}")

    strategy_mode = str(runtime.get("strategy_mode") or "").strip()
    if not strategy_mode:
        raise RuntimeError(f"Live config is missing runtime.strategy_mode: {path}")

    normalized_symbols: list[str] = []
    seen_symbols: set[str] = set()
    for raw_symbol in symbols:
        if type(raw_symbol) is not str:
            raise RuntimeError(f"Live config research symbols must be strings: {path}")
        symbol = raw_symbol.strip().upper()
        if not symbol or symbol in seen_symbols:
            raise RuntimeError(
                "Live config research symbols must be nonblank and unique after "
                f"normalization: {path}"
            )
        seen_symbols.add(symbol)
        normalized_symbols.append(symbol)
    raw_timeframe = runtime.get("bar_timeframe_minutes")
    if (
        not normalized_symbols
        or type(raw_timeframe) is not int
        or raw_timeframe < 1
    ):
        raise RuntimeError(f"Live config research symbols/timeframe are invalid: {path}")
    return {
        "symbols": normalized_symbols,
        "bar_timeframe_minutes": raw_timeframe,
        "strategy_mode": strategy_mode,
        "risk_execution_context": resolve_risk_execution_context(
            runtime,
            effective_environment,
        ),
    }


@dataclass(frozen=True)
class ResolvedResearchInputs:
    symbols: tuple[str, ...]
    bar_timeframe_minutes: int
    strategy_mode: str
    risk_execution_context: tuple[tuple[str, int | float], ...]
    input_sha256: str

    @property
    def timeframe(self) -> str:
        return f"{self.bar_timeframe_minutes}Min"

    @property
    def sweep_params(self) -> tuple[tuple[str, str], ...]:
        return (("strategy-mode-list", self.strategy_mode), *SWEEP_PARAMETER_TEMPLATE)

    def to_payload(self) -> dict[str, object]:
        return {
            "symbols": list(self.symbols),
            "bar_timeframe_minutes": self.bar_timeframe_minutes,
            "strategy_mode": self.strategy_mode,
            "risk_execution_context": dict(self.risk_execution_context),
            "input_sha256": self.input_sha256,
        }


def _resolve_research_inputs(
    live_config_path: Path = LIVE_CONFIG_PATH,
) -> ResolvedResearchInputs:
    values = _load_live_runtime_defaults(live_config_path)
    body = {
        "symbols": values["symbols"],
        "bar_timeframe_minutes": values["bar_timeframe_minutes"],
        "strategy_mode": values["strategy_mode"],
        "risk_execution_context": values["risk_execution_context"],
    }
    encoded = json.dumps(
        body,
        sort_keys=True,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
    ).encode("utf-8")
    return ResolvedResearchInputs(
        symbols=tuple(body["symbols"]),  # type: ignore[arg-type]
        bar_timeframe_minutes=body["bar_timeframe_minutes"],  # type: ignore[arg-type]
        strategy_mode=body["strategy_mode"],  # type: ignore[arg-type]
        risk_execution_context=tuple(
            (field, body["risk_execution_context"][field])  # type: ignore[index]
            for field in RISK_EXECUTION_CONTEXT_FIELDS
        ),
        input_sha256=hashlib.sha256(encoded).hexdigest(),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the research workflow with fresh snapshots by default, or reuse an existing "
            "dataset with --dataset."
        )
    )
    parser.add_argument(
        "--dataset",
        help=(
            "Optional existing dataset directory. When provided, run research directly against "
            "that dataset instead of creating fresh snapshots."
        ),
    )
    return parser.parse_args()

# ---------------------------------------------------------------------------
# SNAPSHOT CONFIG
# ---------------------------------------------------------------------------

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

SWEEP_PARAMETER_TEMPLATE: tuple[tuple[str, str], ...] = (
    # Broader SMA range to find a tighter or looser mean.
    ("sma-bars-list",             "10,15,20,25,30"),
    # Tighter entries (0.001) and slightly wider (0.003) added to the search space.
    ("entry-threshold-pct-list",  "0.001,0.0015,0.002,0.0025,0.003"),
    ("time-window-mode-list",     "full_day,morning_only"),
    # Sweep all exit styles: sma (wait for recross), midpoint (take profit earlier),
    # eod (hold until forced flatten).
    ("mean-reversion-exit-style-list", "sma,midpoint,eod"),
    # Sweep ATR percentile filter — lower values only trade lower-vol pullbacks,
    # which tend to revert more cleanly.  80 (previous default) is included.
    ("mean-reversion-max-atr-percentile-list", "40,50,60,70,80"),
)

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

# Compatibility projection only.  Stability and promotion use the exact
# canonical strategy envelope/hash emitted by backtest_runner.py, never a
# hand-picked parameter subset.
STABILITY_PARAMS = list(PROMOTED_STRATEGY_FIELDS)

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
REGIME_MAX_DATA_AGE_HOURS = 96.0    # accommodates weekends; older bars fail closed

# Regimes listed here disable trading regardless of approval / stability.
# Remove "high_volatility" or add "sideways" to adjust behaviour.
REGIME_DISABLE_ON: set[str] = {"high_volatility", "stale", "unknown"}


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def _log(msg: str) -> None:
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def _section(title: str) -> None:
    """Print a visible section header to break up dense subprocess output."""
    bar = "-" * 60
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


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _timestamped_csv(results_dir: Path, prefix: str) -> Path:
    """Return a collision-resistant research CSV path."""
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
    return _ensure_dir(results_dir) / f"{prefix}_{stamp}_{uuid4().hex[:8]}.csv"


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
        subprocess.run(cmd, check=True, cwd=SCRIPT_DIR)
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

def _resolve_research_anchor_utc(
    now_utc: datetime | None = None,
) -> str:
    anchor = (now_utc or datetime.now(timezone.utc)).astimezone(timezone.utc).replace(
        hour=0,
        minute=0,
        second=0,
        microsecond=0,
    )
    return anchor.strftime("%Y-%m-%dT%H:%M:%SZ")


def _canonical_utc_text(value: object, *, field: str) -> str:
    if type(value) is not str or not value:
        raise ResearchBundleValidationError(f"{field} is missing or invalid.")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ResearchBundleValidationError(f"{field} is not a valid timestamp.") from exc
    if parsed.tzinfo is None or parsed.utcoffset() != timedelta(0):
        raise ResearchBundleValidationError(f"{field} must resolve to UTC.")
    return parsed.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _snapshot_date_range(
    lookback_days: int,
    *,
    research_anchor_utc: str,
) -> tuple[str, str]:
    """
    Return (start, end) as UTC timestamp strings accepted by dataset_snapshotter.py.
    end  = today at UTC midnight
    start = end − lookback_days
    """
    fmt = "%Y-%m-%dT%H:%M:%SZ"
    try:
        end = datetime.fromisoformat(research_anchor_utc.replace("Z", "+00:00"))
    except (TypeError, ValueError) as exc:
        raise ResearchBundleValidationError("Research UTC anchor is invalid.") from exc
    if end.tzinfo is None or end.utcoffset() != timedelta(0):
        raise ResearchBundleValidationError("Research UTC anchor must be timezone-aware UTC.")
    start = end - timedelta(days=lookback_days)
    return start.strftime(fmt), end.strftime(fmt)


def _fresh_window_ranges(research_anchor_utc: str) -> list[dict[str, object]]:
    return [
        {
            "lookback_days": days,
            "start_utc": _snapshot_date_range(
                days,
                research_anchor_utc=research_anchor_utc,
            )[0],
            "end_utc": research_anchor_utc,
        }
        for days in VALIDATION_WINDOWS
    ]


# ---------------------------------------------------------------------------
# Pipeline steps
# ---------------------------------------------------------------------------

def step_snapshot(
    lookback_days: int,
    *,
    resolved_inputs: ResolvedResearchInputs | None = None,
    research_anchor_utc: str | None = None,
) -> Path:
    """
    Fetch a dataset for the given lookback window via dataset_snapshotter.py.
    Returns the Path of the newly created dataset directory.
    Computes and verifies the exact content-addressed dataset target.  It never
    selects by directory mtime or an unrelated concurrent snapshot.
    """
    inputs = resolved_inputs or _resolve_research_inputs()
    anchor = research_anchor_utc or _resolve_research_anchor_utc()
    start_str, end_str = _snapshot_date_range(
        lookback_days,
        research_anchor_utc=anchor,
    )
    start = datetime.fromisoformat(start_str.replace("Z", "+00:00"))
    end = datetime.fromisoformat(end_str.replace("Z", "+00:00"))
    from dataset_snapshotter import build_dataset_id, compute_code_hash

    code_hash = compute_code_hash(SCRIPT_DIR)
    expected_dataset_id = build_dataset_id(
        list(inputs.symbols),
        inputs.timeframe,
        start,
        end,
        SNAPSHOT_FEED.lower(),
        code_hash,
    )
    expected_dataset = DATASETS_DIR / expected_dataset_id

    cmd = [
        sys.executable, str(SCRIPT_DIR / "dataset_snapshotter.py"),
        "--symbols",    *inputs.symbols,
        "--start",      start_str,
        "--end",        end_str,
        "--timeframe",  inputs.timeframe,
        "--feed",       SNAPSHOT_FEED,
        "--adjustment", SNAPSHOT_ADJUSTMENT,
        "--align-mode", SNAPSHOT_ALIGN_MODE,
        "--output-dir", str(DATASETS_DIR),
    ]

    _run(cmd, label=f"Dataset snapshot ({lookback_days}d)")
    if not expected_dataset.is_dir():
        raise RuntimeError(
            "Snapshotter did not create/reuse the deterministic requested dataset: "
            f"{expected_dataset}"
        )
    manifest = load_verified_manifest(expected_dataset)
    expected_symbols = list(inputs.symbols)
    try:
        manifest_start = datetime.fromisoformat(
            str(manifest["start_utc"]).replace("Z", "+00:00")
        )
        manifest_end = datetime.fromisoformat(
            str(manifest["end_utc"]).replace("Z", "+00:00")
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise RuntimeError("Snapshot manifest start/end evidence is invalid.") from exc
    if (
        manifest.get("dataset_id") != expected_dataset_id
        or manifest_start != start
        or manifest_end != end
        or manifest.get("symbols") != expected_symbols
        or manifest.get("timeframe") != inputs.timeframe
        or manifest.get("feed") != SNAPSHOT_FEED.lower()
        or manifest.get("adjustment") != SNAPSHOT_ADJUSTMENT.lower()
        or manifest.get("align_mode") != SNAPSHOT_ALIGN_MODE
        or manifest.get("code_hash") != code_hash
    ):
        raise RuntimeError(
            "Snapshot manifest contradicts the exact requested research window/configuration."
        )
    return expected_dataset


def step_backtest(
    output_csv: Path,
    dataset_path: Path,
    *,
    resolved_inputs: ResolvedResearchInputs | None = None,
) -> None:
    """Run the parameter sweep backtest against dataset_path."""
    _log(f"Using dataset: {dataset_path.name}")

    inputs = resolved_inputs or _resolve_research_inputs()
    cmd = [
        sys.executable, str(SCRIPT_DIR / "backtest_runner.py"),
        "--dataset",    str(dataset_path),
        "--output-csv", str(output_csv),
        *_params_to_flags(list(inputs.sweep_params)),
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

# Kept for compatibility with reporting code, but derived from the explicit
# versioned contract.  It is not an evidence boundary.
CONFIG_COLS = list(PROMOTED_STRATEGY_FIELDS)

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
    "mean_reversion_exit_style",
    "mean_reversion_max_atr_percentile",
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
    try:
        sweep_bytes = sweep_csv.read_bytes()
    except OSError as exc:
        raise RuntimeError(f"Sweep CSV is unreadable: {sweep_csv}") from exc
    df = pd.read_csv(io.BytesIO(sweep_bytes))
    df.insert(0, "source_row_index", range(len(df)))

    # Validate that expected columns exist before proceeding.
    missing = [
        c
        for c in (
            COL_RETURN,
            COL_PF,
            COL_DRAWDOWN,
            "canonical_strategy_json",
            "canonical_strategy_sha256",
            "model_execution_evidence_json",
            "model_execution_evidence_sha256",
        )
        if c not in df.columns
    ]
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

    # Ranking ties are deterministic and explicitly bound by research_config.
    df_sorted = df.sort_values(
        [
            "combined_score",
            "canonical_strategy_sha256",
            "model_execution_evidence_sha256",
            "source_row_index",
        ],
        ascending=[True, True, True, True],
        kind="mergesort",
    ).reset_index(drop=True)
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

    best_row = df_sorted.iloc[0].to_dict()
    # Promotion is impossible unless the selected row carries the exact
    # canonical envelope and matching digest produced by the backtest.
    canonical_strategy_from_backtest_row(best_row)
    model_execution_evidence_from_backtest_row(best_row)
    candidates: list[dict[str, object]] = []
    for _, row in df.sort_values("source_row_index").iterrows():
        row_mapping = row.to_dict()
        candidate_contract = canonical_strategy_from_backtest_row(row_mapping)
        candidate_model = model_execution_evidence_from_backtest_row(row_mapping)
        candidate_metrics = _selected_metrics(row_mapping)
        candidates.append(
            {
                "source_row_index": int(row_mapping["source_row_index"]),
                "total_return_pct": candidate_metrics[COL_RETURN],
                "profit_factor": candidate_metrics[COL_PF],
                "max_drawdown_pct": candidate_metrics[COL_DRAWDOWN],
                "canonical_strategy_sha256": candidate_contract.sha256,
                "model_execution_evidence_sha256": candidate_model.sha256,
            }
        )
    try:
        leaderboard_bytes = leaderboard_csv.read_bytes()
    except OSError as exc:
        raise ResearchBundleValidationError(
            "Leaderboard CSV became unreadable after ranking."
        ) from exc
    selection_evidence = build_backtest_selection_evidence(
        sweep_csv_sha256=hashlib.sha256(sweep_bytes).hexdigest(),
        leaderboard_csv_sha256=hashlib.sha256(leaderboard_bytes).hexdigest(),
        selected_source_row_index=int(best_row["source_row_index"]),
        candidates=candidates,
    )
    validate_backtest_selection_artifacts(
        selection_evidence,
        sweep_csv_bytes=sweep_bytes,
        leaderboard_csv_bytes=leaderboard_bytes,
    )
    best_row["research_selection_evidence"] = selection_evidence
    return best_row


def _approval_policy() -> ApprovalPolicy:
    return ApprovalPolicy(
        name=APPROVAL_POLICY_NAME,
        version=APPROVAL_POLICY_VERSION,
        min_profit_factor=float(APPROVAL_MIN_PROFIT_FACTOR),
        max_drawdown_pct=float(APPROVAL_MAX_DRAWDOWN_PCT),
        min_total_return_pct=float(APPROVAL_MIN_TOTAL_RETURN_PCT),
        min_trades_per_day=float(APPROVAL_MIN_TRADES_PER_DAY),
    )


def _stability_policy() -> StabilityPolicy:
    return StabilityPolicy(
        name=STABILITY_POLICY_NAME,
        version=STABILITY_POLICY_VERSION,
        min_profit_factor=float(STABILITY_MIN_PROFIT_FACTOR),
        max_drawdown_pct=float(STABILITY_MAX_DRAWDOWN_PCT),
        minimum_window_count=max(2, len(VALIDATION_WINDOWS)),
        required_lookback_days=tuple(VALIDATION_WINDOWS),
        require_consistent_canonical_strategy=True,
        require_consistent_model_execution=True,
    )


def _regime_policy() -> RegimePolicy:
    return RegimePolicy(
        name=REGIME_POLICY_NAME,
        version=REGIME_POLICY_VERSION,
        volatility_period_bars=REGIME_VOL_PERIOD,
        trend_sma_period_bars=REGIME_TREND_SMA_PERIOD,
        high_volatility_threshold=float(REGIME_VOL_HIGH_THRESHOLD),
        slope_threshold=float(REGIME_SLOPE_THRESHOLD),
        max_data_age_hours=float(REGIME_MAX_DATA_AGE_HOURS),
        disable_on=tuple(sorted(REGIME_DISABLE_ON)),
    )


def _selected_metrics(best_row: Mapping[str, object]) -> dict[str, float]:
    selected_names = tuple(
        dict.fromkeys(
            (
                *METRIC_COLS,
                "total_trades",
                "expectancy",
                "avg_return_per_trade_pct",
            )
        )
    )
    return normalize_metrics(
        {name: best_row[name] for name in selected_names if name in best_row}
    )


def step_approve_config(best_row: dict) -> tuple[bool, list[str]]:
    """Evaluate one selected row through the versioned approval policy."""

    metrics = _selected_metrics(best_row)
    policy = _approval_policy()
    outcome = evaluate_approval_policy(metrics, policy)
    approved = outcome.approved
    rejection_reasons = list(outcome.rejection_reasons)

    verdict = "APPROVED" if approved else "REJECTED"
    print()
    print(f"  Approval verdict: {verdict}")
    print("  " + "-" * 55)
    checks = (
        metrics[COL_PF] >= policy.min_profit_factor,
        metrics[COL_DRAWDOWN] <= policy.max_drawdown_pct,
        metrics[COL_RETURN] >= policy.min_total_return_pct,
        metrics[COL_TRADES_PER_DAY] >= policy.min_trades_per_day,
    )
    descriptions = (
        f"profit_factor {metrics[COL_PF]:.4f} >= {policy.min_profit_factor}",
        f"max_drawdown_pct {metrics[COL_DRAWDOWN]:.4f}% <= {policy.max_drawdown_pct}%",
        f"total_return_pct {metrics[COL_RETURN]:.4f}% >= {policy.min_total_return_pct}%",
        f"trades_per_day {metrics[COL_TRADES_PER_DAY]:.4f} >= {policy.min_trades_per_day}",
    )
    for passed, description in zip(checks, descriptions, strict=True):
        print(f"  [{'PASS' if passed else 'FAIL'}] {description}")
    print()

    return approved, rejection_reasons


def step_save_best_config(
    best_row: dict,
    dataset_path: Path,
    output_path: Path,
    approved: bool,
    rejection_reasons: list[str],
) -> None:
    """Deprecated unsafe writer; promotion must commit through the pointer workflow."""

    del best_row, dataset_path, output_path, approved, rejection_reasons
    raise ResearchPublicationError(
        "Direct best-config writes are forbidden; finalize a validated bundle and "
        "commit its generation pointer."
    )


def _load_dataset_manifest_and_bars(dataset_path: Path) -> tuple[dict, pd.DataFrame]:
    bars_path = dataset_path / "bars.parquet"
    manifest = load_verified_manifest(dataset_path)
    df = pd.read_parquet(bars_path, columns=["symbol"])
    return manifest, df


def _normalize_symbols(symbols: list[str] | None) -> list[str]:
    if symbols is None:
        return []

    normalized: list[str] = []
    seen: set[str] = set()
    for raw_symbol in symbols:
        symbol = str(raw_symbol).strip().upper()
        if not symbol or symbol in seen:
            continue
        seen.add(symbol)
        normalized.append(symbol)
    return normalized


def _coerce_boolish(value: object, field_name: str) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and value in {0, 1}:
        return bool(value)
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "1", "yes"}:
            return True
        if lowered in {"false", "0", "no"}:
            return False
    raise RuntimeError(f"Expected boolean-like value for {field_name}, got {value!r}")


def _resolve_dataset_symbols(dataset_path: Path) -> tuple[list[str], str, dict]:
    manifest, df = _load_dataset_manifest_and_bars(dataset_path)
    manifest_symbols = _normalize_symbols(manifest.get("symbols"))
    if manifest_symbols:
        return manifest_symbols, "dataset_metadata", manifest

    content_symbols = _normalize_symbols(df["symbol"].dropna().astype(str).tolist())
    if content_symbols:
        return content_symbols, "dataset_content", manifest

    raise RuntimeError(f"Unable to resolve symbols from dataset at {dataset_path}")


def _parse_timeframe_to_minutes(timeframe_text: str) -> int:
    normalized = timeframe_text.strip().lower()
    suffix_to_minutes = {
        "minutes": 1,
        "minute": 1,
        "mins": 1,
        "min": 1,
        "hours": 60,
        "hour": 60,
        "days": 390,
        "day": 390,
    }
    for suffix, multiplier in suffix_to_minutes.items():
        if normalized.endswith(suffix):
            amount_text = normalized[: -len(suffix)].strip()
            if amount_text.isdigit() and int(amount_text) > 0:
                return int(amount_text) * multiplier
    raise RuntimeError(f"Unsupported dataset timeframe for live config: {timeframe_text!r}")


def _log_dataset_details(dataset_path: Path, *, context: str) -> None:
    symbols, symbol_source, manifest = _resolve_dataset_symbols(dataset_path)
    timeframe = manifest.get("timeframe", "unknown")
    _log(f"{context} dataset    : {dataset_path}")
    _log(f"{context} symbols    : {', '.join(symbols)}")
    _log(f"{context} sym source : {symbol_source}")
    _log(f"{context} timeframe  : {timeframe}")


def step_write_live_config(
    best_row: dict,
    dataset_path: Path,
    output_path: Path,
    approved: bool,
    rejection_reasons: list[str],
) -> None:
    """Deprecated unsafe writer; promotion must commit through the pointer workflow."""

    del best_row, dataset_path, output_path, approved, rejection_reasons
    raise ResearchPublicationError(
        "Direct live-config writes are forbidden; finalize a validated bundle and "
        "commit its generation pointer."
    )


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
        # Retain exact finite signals so bundle-side policy recomputation cannot
        # flip a near-threshold label due to display rounding.
        "current_volatility": float(current_vol) if not pd.isna(current_vol) else None,
        "sma_slope_pct":      float(slope_pct),
    }


def step_detect_regime(dataset_path: Path) -> dict:
    """
    Load the primary dataset, classify each symbol's regime, and return
    an aggregate summary.  Majority vote determines the overall label.
    """
    bars_path = dataset_path / "bars.parquet"
    if not bars_path.exists():
        raise RuntimeError(f"Cannot detect regime — bars file not found: {bars_path}")

    load_verified_manifest(dataset_path)
    df = pd.read_parquet(bars_path)
    required = {"symbol", "timestamp", "close"}
    missing  = required - set(df.columns)
    if missing:
        raise RuntimeError(f"Regime detection requires columns {missing} — not in dataset")
    parsed_timestamps = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    latest_timestamp = parsed_timestamps.max()
    if pd.isna(latest_timestamp):
        raise RuntimeError("Regime detection found no valid bar timestamps")
    latest_bar_utc = latest_timestamp.to_pydatetime().astimezone(timezone.utc).strftime(
        "%Y-%m-%dT%H:%M:%SZ"
    )

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
    policy = _regime_policy()
    latest_bar_time = datetime.fromisoformat(latest_bar_utc.replace("Z", "+00:00"))
    data_age_hours = (
        datetime.now(timezone.utc) - latest_bar_time
    ).total_seconds() / 3600.0
    if data_age_hours < 0 or data_age_hours > policy.max_data_age_hours:
        overall = "stale"
    disabled = overall in policy.disable_on

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
        "policy":            policy.to_payload(),
        "regime":            overall,
        "disabled_by_regime": disabled,
        "latest_bar_utc":    latest_bar_utc,
        "avg_volatility":    avg_vol,
        "avg_sma_slope_pct": avg_slope,
        "per_symbol":        per_symbol,
    }


# ---------------------------------------------------------------------------
# Multi-window stability evaluation
# ---------------------------------------------------------------------------

def step_evaluate_stability(window_results: list[dict]) -> dict:
    """Recompute stability from policy-bound strategy/model/dataset evidence."""

    if not window_results:
        raise ResearchBundleValidationError("Stability requires at least one window.")
    resolved_windows: list[
        tuple[
            dict,
            BacktestCanonicalStrategy,
            BacktestModelExecutionEvidence,
            dict[str, float],
        ]
    ] = []
    evidence_windows: list[dict[str, object]] = []
    for window in window_results:
        row = window.get("best_row")
        if not isinstance(row, Mapping):
            raise ResearchBundleValidationError("Stability window is missing its best row.")
        canonical = canonical_strategy_from_backtest_row(row)
        model_evidence = model_execution_evidence_from_backtest_row(row)
        metrics = _selected_metrics(row)
        provenance = window.get("dataset_provenance")
        selection_evidence = row.get("research_selection_evidence")
        if not isinstance(provenance, Mapping):
            raise ResearchBundleValidationError(
                "Stability window is missing captured dataset provenance."
            )
        if not isinstance(selection_evidence, Mapping):
            raise ResearchBundleValidationError(
                "Stability window is missing deterministic selection evidence."
            )
        provenance_sha = provenance.get("provenance_sha256")
        evidence_windows.append(
            {
                "lookback_days": window["lookback_days"],
                "canonical_strategy_sha256": canonical.sha256,
                "dataset_provenance_sha256": provenance_sha,
                "model_execution_evidence_json": model_evidence.canonical_json,
                "model_execution_evidence_sha256": model_evidence.sha256,
                "selection_evidence": selection_evidence,
                "metrics": metrics,
            }
        )
        resolved_windows.append(
            (window, canonical, model_evidence, metrics)
        )

    stability = evaluate_stability_policy(evidence_windows, _stability_policy())

    # --- console summary ---------------------------------------------------
    verdict = "STABLE" if stability["stable"] else "UNSTABLE"
    print()
    print(f"  Stability verdict: {verdict}")
    print("  " + "-" * 55)
    for window, canonical, _model_evidence, metrics in resolved_windows:
        values = canonical.contract.promoted_values
        print(
            f"  {window['lookback_days']:>3}d  strategy={values['strategy_mode']} "
            f"sma={values['sma_bars']} thresh={values['entry_threshold_pct']}"
            f"  pf={metrics[COL_PF]:.4f}  dd={metrics[COL_DRAWDOWN]:.4f}% "
            f"ret={metrics[COL_RETURN]:.4f}%"
        )
    if stability["reasons"]:
        print()
        for reason in stability["reasons"]:
            print(f"  [FAIL] {reason}")
    else:
        print()
        print("  [PASS] All windows meet metric thresholds")
        print("  [PASS] Canonical strategy and model execution are consistent")
    print()
    return stability


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

    canonical = canonical_strategy_from_backtest_row(best_row)

    return {
        "should_trade":        should_trade,
        "approved":            approved,
        "stable":              stable,
        "disabled_by_regime":  disabled_by_regime,
        "market_regime":       regime["regime"],
        "canonical_strategy_sha256": canonical.sha256,
    }


def step_save_trade_decision(decision: dict, output_path: Path) -> None:
    """Write trade_decision.json.  Always overwrites so there is one canonical file."""
    _ensure_dir(output_path.parent)
    output_path.write_text(json.dumps(decision, indent=2), encoding="utf-8")
    _log(f"Trade decision  : {output_path}")


def _safe_metric(row: dict, key: str) -> float | None:
    value = row.get(key)
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError, OverflowError):
        return None


def _build_hybrid_branch_participation(best_row: dict) -> dict[str, float] | None:
    mr_share = _safe_metric(best_row, "mr_branch_trade_share_pct")
    bb_share = _safe_metric(best_row, "bb_branch_trade_share_pct")
    participation: dict[str, float] = {}
    if mr_share is not None:
        participation["mean_reversion"] = mr_share / 100.0 if mr_share > 1.0 else mr_share
    if bb_share is not None:
        participation["bollinger_breakout"] = bb_share / 100.0 if bb_share > 1.0 else bb_share
    return participation or None


def build_drift_profile_payload(
    *,
    best_row: dict,
    dataset_path: Path,
    approved: bool,
    stability: dict,
    regime: dict,
) -> dict:
    symbols, symbol_source, manifest = _resolve_dataset_symbols(dataset_path)
    symbol_count = len(symbols)
    timeframe_text = str(manifest.get("timeframe") or "").strip()
    timeframe_minutes = _parse_timeframe_to_minutes(timeframe_text) if timeframe_text else None
    trades_per_day = _safe_metric(best_row, "trades_per_day")
    buy_rate = (
        trades_per_day / symbol_count
        if trades_per_day is not None and symbol_count > 0
        else None
    )
    hybrid_branch_participation = _build_hybrid_branch_participation(best_row)

    return {
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "approved": approved,
        "source_artifacts": {
            "dataset": str(dataset_path),
            "dataset_symbol_source": symbol_source,
            "best_config": str(BEST_CONFIG_PATH),
            "trade_decision": str(TRADE_DECISION_PATH),
            "stability_report": str(STABILITY_REPORT_PATH),
            "live_config": str(LIVE_CONFIG_PATH),
        },
        "context": {
            "strategy_mode": best_row.get("strategy_mode"),
            "symbols": symbols,
            "symbol_count": symbol_count,
            "timeframe": timeframe_text or None,
            "bar_timeframe_minutes": timeframe_minutes,
            "stable": bool(stability.get("stable", False)),
            "market_regime": regime.get("regime"),
        },
        "performance": {
            "trades_per_day": trades_per_day,
            "win_rate": _safe_metric(best_row, "win_rate"),
            "profit_factor": _safe_metric(best_row, "profit_factor"),
            "sharpe_ratio": _safe_metric(best_row, "sharpe_ratio"),
            "realized_pnl": _safe_metric(best_row, "realized_pnl"),
            "combined_score": _safe_metric(best_row, "combined_score"),
        },
        "metrics": {
            "buy_signal_rate_per_symbol_per_day": buy_rate,
            "sell_signal_rate_per_symbol_per_day": buy_rate,
            "rejection_rate": float(DEFAULT_BACKTEST["rejection_rate"]),
            "rejection_breakdown": None,
            "hybrid_branch_participation": hybrid_branch_participation,
            "avg_bar_age_s": float(WARN["avg_bar_age_s"]),
            "stale_bar_rate": 0.0,
        },
        "metric_sources": {
            "buy_signal_rate_per_symbol_per_day": "best_row.trades_per_day / promoted symbol_count",
            "sell_signal_rate_per_symbol_per_day": "derived from long-only trades_per_day cadence",
            "rejection_rate": "daily_report.DEFAULT_BACKTEST.rejection_rate",
            "rejection_breakdown": "not available from current research artifacts",
            "hybrid_branch_participation": (
                "best_row branch share columns"
                if hybrid_branch_participation is not None
                else "not available for this promoted strategy"
            ),
            "avg_bar_age_s": "daily_report.WARN.avg_bar_age_s",
            "stale_bar_rate": "validated operator expectation: zero stale bars",
        },
    }


def step_save_drift_profile(
    best_row: dict,
    dataset_path: Path,
    approved: bool,
    stability: dict,
    regime: dict,
    output_path: Path,
) -> None:
    payload = build_drift_profile_payload(
        best_row=best_row,
        dataset_path=dataset_path,
        approved=approved,
        stability=stability,
        regime=regime,
    )
    _ensure_dir(output_path.parent)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    _log(f"Drift profile   : {output_path}")


@dataclass(frozen=True)
class ResearchPublicationTargets:
    repo_root: Path
    bundle_directory: Path
    generation_directory: Path
    pointer_path: Path
    latest_state_path: Path
    best_config_path: Path
    stability_report_path: Path
    trade_decision_path: Path
    drift_profile_path: Path
    live_config_path: Path


@dataclass(frozen=True)
class ResearchPublicationResult:
    bundle_path: Path
    bundle_sha256: str
    research_generation_sha256: str
    published: bool
    pointer_path: Path | None
    latest_state_path: Path


def _default_publication_targets() -> ResearchPublicationTargets:
    return ResearchPublicationTargets(
        repo_root=SCRIPT_DIR,
        bundle_directory=PROMOTION_BUNDLE_DIR,
        generation_directory=PROMOTION_GENERATION_DIR,
        pointer_path=PROMOTION_POINTER_PATH,
        latest_state_path=LATEST_RESEARCH_STATE_PATH,
        best_config_path=BEST_CONFIG_PATH,
        stability_report_path=STABILITY_REPORT_PATH,
        trade_decision_path=TRADE_DECISION_PATH,
        drift_profile_path=DRIFT_PROFILE_PATH,
        live_config_path=LIVE_CONFIG_PATH,
    )


def _repo_relative(path: Path, repo_root: Path) -> str:
    root = Path(repo_root).resolve(strict=True)
    candidate = Path(path).resolve(strict=False)
    try:
        return candidate.relative_to(root).as_posix()
    except ValueError as exc:
        raise ResearchPublicationError(
            f"Publication target escapes the repository root: {candidate}"
        ) from exc


def _research_config_inputs(
    args: argparse.Namespace,
    *,
    resolved_inputs: ResolvedResearchInputs | None = None,
    research_anchor_utc: str | None = None,
    window_ranges: list[dict[str, object]] | None = None,
) -> dict[str, object]:
    inputs = resolved_inputs or _resolve_research_inputs()
    anchor = research_anchor_utc or _resolve_research_anchor_utc()
    if args.dataset:
        existing_path = Path(args.dataset)
        if not existing_path.is_absolute():
            existing_path = SCRIPT_DIR / existing_path
        existing_reference = _repo_relative(existing_path, SCRIPT_DIR)
        if window_ranges is None:
            manifest = load_verified_manifest(existing_path)
            window_ranges = [
                {
                    "lookback_days": 0,
                    "start_utc": _canonical_utc_text(
                        manifest.get("start_utc", manifest.get("start")),
                        field="existing dataset start_utc",
                    ),
                    "end_utc": _canonical_utc_text(
                        manifest.get("end_utc", manifest.get("end")),
                        field="existing dataset end_utc",
                    ),
                }
            ]
    else:
        existing_reference = None
        if window_ranges is None:
            window_ranges = _fresh_window_ranges(anchor)
    assert window_ranges is not None
    return {
        "dataset_request": {
            "mode": "existing_dataset" if args.dataset else "fresh_snapshots",
            "research_anchor_utc": anchor,
            "validation_windows": [0] if args.dataset else list(VALIDATION_WINDOWS),
            "window_ranges": window_ranges,
            "existing_dataset_reference": existing_reference,
            "snapshot": (
                None
                if args.dataset
                else {
                    "symbols": list(inputs.symbols),
                    "timeframe": inputs.timeframe,
                    "feed": SNAPSHOT_FEED.lower(),
                    "adjustment": SNAPSHOT_ADJUSTMENT.lower(),
                    "align_mode": SNAPSHOT_ALIGN_MODE,
                }
            ),
        },
        "search": {
            "sweep_parameters": [
                {"flag": flag, "value": value} for flag, value in inputs.sweep_params
            ],
            "fixed_parameters": [
                {"flag": flag, "value": value} for flag, value in FIXED_PARAMS
            ],
        },
        "ranking_policy": {
            "name": "tradeos.mean_metric_rank",
            "version": 1,
            "combined_score_formula": "mean(rank_return,rank_pf,rank_drawdown)",
            "tie_method": "min",
            "tie_breakers": [
                "canonical_strategy_sha256",
                "model_execution_evidence_sha256",
                "source_row_index",
            ],
            "metric_directions": {
                "max_drawdown_pct": "ascending",
                "profit_factor": "descending",
                "total_return_pct": "descending",
            },
        },
        "approval_policy": _approval_policy().to_payload(),
        "stability_policy": _stability_policy().to_payload(),
        "regime_policy": _regime_policy().to_payload(),
        "evidence_contract": {
            "canonical_strategy_json_column": "canonical_strategy_json",
            "canonical_strategy_sha256_column": "canonical_strategy_sha256",
            "model_execution_evidence_json_column": "model_execution_evidence_json",
            "model_execution_evidence_sha256_column": "model_execution_evidence_sha256",
        },
        "resolved_live_inputs": inputs.to_payload(),
    }


def _assert_run_inputs_unchanged(
    *,
    repo_root: Path,
    source_identity: Mapping[str, object],
    window_results: list[dict],
) -> None:
    assert_source_identity_unchanged(
        source_identity,
        capture_source_identity(repo_root),
    )
    for window in window_results:
        checkpoint = window.get("dataset_checkpoint")
        provenance = window.get("dataset_provenance")
        row = window.get("best_row")
        if (
            not isinstance(checkpoint, Mapping)
            or not isinstance(provenance, Mapping)
            or not isinstance(row, Mapping)
        ):
            raise ResearchBundleValidationError(
                "Research window is missing its pre/post execution evidence."
            )
        selection = row.get("research_selection_evidence")
        sweep_csv = window.get("sweep_csv")
        leaderboard_csv = window.get("leaderboard_csv")
        if (
            not isinstance(selection, Mapping)
            or not isinstance(sweep_csv, Path)
            or not isinstance(leaderboard_csv, Path)
        ):
            raise ResearchBundleValidationError(
                "Research window is missing sweep/leaderboard selection identity."
            )
        try:
            sweep_bytes = sweep_csv.read_bytes()
            leaderboard_bytes = leaderboard_csv.read_bytes()
        except OSError as exc:
            raise ResearchBundleValidationError(
                "Sweep or leaderboard artifact became unreadable."
            ) from exc
        if (
            selection.get("sweep_csv_sha256") != hashlib.sha256(sweep_bytes).hexdigest()
            or selection.get("leaderboard_csv_sha256")
            != hashlib.sha256(leaderboard_bytes).hexdigest()
        ):
            raise ResearchBundleValidationError(
                "Sweep or leaderboard bytes changed after deterministic selection."
            )
        validate_backtest_selection_artifacts(
            selection,
            sweep_csv_bytes=sweep_bytes,
            leaderboard_csv_bytes=leaderboard_bytes,
        )
        assert_dataset_checkpoint_unchanged(
            checkpoint,
            capture_dataset_checkpoint(window["dataset"]),
        )
        canonical = canonical_strategy_from_backtest_row(row)
        observed_provenance = capture_dataset_provenance(
            window["dataset"],
            canonical.contract,
        )
        if observed_provenance != provenance:
            raise ResearchBundleValidationError(
                "Dataset provenance changed after stability evaluation."
            )


def _assert_resolved_live_inputs_unchanged(
    *,
    live_config_path: Path,
    research_config_inputs: Mapping[str, object],
) -> None:
    declared = research_config_inputs.get("resolved_live_inputs")
    observed = _resolve_research_inputs(live_config_path).to_payload()
    if not isinstance(declared, Mapping) or dict(declared) != observed:
        raise ResearchBundleValidationError(
            "Resolved live research inputs changed during the locked research run."
        )


def finalize_research_bundle_and_publish(
    *,
    window_results: list[dict],
    approved: bool,
    rejection_reasons: list[str],
    stability: Mapping[str, object],
    regime: Mapping[str, object],
    final_decision: Mapping[str, object],
    source_identity: Mapping[str, object],
    research_config_inputs: Mapping[str, object],
    targets: ResearchPublicationTargets,
    generated_at_utc: str | None = None,
    failure_injector: Callable[[str, int, Path], None] | None = None,
) -> ResearchPublicationResult:
    """Finalize immutable evidence, then commit one pointer-backed generation."""

    if not window_results:
        raise ResearchBundleValidationError("Research publication has no windows.")
    root = Path(targets.repo_root).resolve(strict=True)
    _assert_resolved_live_inputs_unchanged(
        live_config_path=targets.live_config_path,
        research_config_inputs=research_config_inputs,
    )
    _assert_run_inputs_unchanged(
        repo_root=root,
        source_identity=source_identity,
        window_results=window_results,
    )
    primary = window_results[-1]
    primary_row = primary.get("best_row")
    if not isinstance(primary_row, Mapping):
        raise ResearchBundleValidationError("Primary research row is missing.")
    canonical = canonical_strategy_from_backtest_row(primary_row)
    model_evidence = model_execution_evidence_from_backtest_row(primary_row)
    metrics = _selected_metrics(primary_row)
    approval_policy = _approval_policy()
    expected_approval = evaluate_approval_policy(metrics, approval_policy)
    if (
        type(approved) is not bool
        or approved != expected_approval.approved
        or tuple(rejection_reasons) != expected_approval.rejection_reasons
    ):
        raise ResearchBundleValidationError(
            "Publication approval contradicts bound metrics and policy."
        )
    stable = stability.get("stable")
    disabled_by_regime = regime.get("disabled_by_regime")
    market_regime = regime.get("regime")
    if (
        type(stable) is not bool
        or type(disabled_by_regime) is not bool
        or type(market_regime) is not str
        or not market_regime
    ):
        raise ResearchBundleValidationError(
            "Publication stability or regime verdict types are invalid."
        )
    eligible = bool(approved and stable and not disabled_by_regime)
    expected_final_decision = {
        "should_trade": eligible,
        "approved": approved,
        "stable": stable,
        "disabled_by_regime": disabled_by_regime,
        "market_regime": market_regime,
        "canonical_strategy_sha256": canonical.sha256,
    }
    if (
        not isinstance(final_decision, Mapping)
        or set(final_decision) != set(expected_final_decision)
        or any(
            type(final_decision[field]) is not bool
            for field in (
                "should_trade",
                "approved",
                "stable",
                "disabled_by_regime",
            )
        )
        or dict(final_decision) != expected_final_decision
    ):
        raise ResearchBundleValidationError(
            "Final decision contradicts approval, stability, regime, or strategy identity."
        )
    dataset_set = build_dataset_provenance_set(
        [
            {
                "lookback_days": window["lookback_days"],
                "provenance": window["dataset_provenance"],
            }
            for window in window_results
        ],
        primary_window_index=len(window_results) - 1,
    )
    research_config = build_research_config_identity(research_config_inputs)
    generated_at = generated_at_utc or datetime.now(timezone.utc).strftime(
        "%Y-%m-%dT%H:%M:%SZ"
    )
    approval_payload = {
        "policy": approval_policy.to_payload(),
        "approved": approved,
        "rejection_reasons": list(rejection_reasons),
    }
    generation_sha256 = compute_research_generation_sha256(
        generated_at_utc=generated_at,
        source_content_sha256=source_identity["content_sha256"],  # type: ignore[arg-type]
        runtime_environment_sha256=source_identity["runtime_environment"][
            "environment_sha256"
        ],  # type: ignore[index,arg-type]
        dataset_provenance_sha256=dataset_set["provenance_sha256"],  # type: ignore[arg-type]
        research_config_sha256=research_config["config_sha256"],  # type: ignore[arg-type]
        canonical_strategy_sha256=canonical.sha256,
        model_execution_evidence_sha256=model_evidence.sha256,
        metrics=metrics,
        approval=approval_payload,
        stability=stability,
        regime=regime,
        final_decision=final_decision,
    )

    live_original_bytes: bytes | None = None
    live_bytes: bytes | None = None
    if eligible:
        try:
            live_original_bytes = targets.live_config_path.read_bytes()
            live_text = live_original_bytes.decode("utf-8", errors="strict")
        except (OSError, UnicodeError) as exc:
            raise ResearchPublicationError("Live config base is unreadable.") from exc
        live_base = strict_json_loads(live_text, label="live config base")
        if not isinstance(live_base, Mapping):
            raise ResearchBundleValidationError("Live config base must be an object.")
        live_payload = build_live_config_payload(
            live_base,
            canonical,
            risk_execution_context=research_config_inputs[
                "resolved_live_inputs"
            ]["risk_execution_context"],  # type: ignore[index,arg-type]
            source_updates={
                "approved": True,
                "rejection_reasons": [],
                "saved_at": generated_at,
                "dataset": _repo_relative(primary["dataset"], root),
                "approval_policy": approval_policy.to_payload(),
                "research_generation_sha256": generation_sha256,
                "research_config_sha256": research_config["config_sha256"],
                "dataset_provenance_sha256": dataset_set["provenance_sha256"],
                "source_git_commit": source_identity["git_commit"],
                "source_content_sha256": source_identity["content_sha256"],
                "model_execution_evidence_sha256": model_evidence.sha256,
            },
        )
        live_payload["saved_at"] = generated_at
        live_bytes = json_file_bytes(live_payload)

    live_hash = hashlib.sha256(live_bytes).hexdigest() if live_bytes is not None else None
    bundle = build_research_bundle(
        canonical_strategy=canonical,
        model_execution_evidence=model_evidence,
        metrics=metrics,
        approval_policy=approval_policy,
        approved=approved,
        rejection_reasons=rejection_reasons,
        stability=stability,
        regime=regime,
        final_decision=final_decision,
        dataset_provenance=dataset_set,
        source_identity=source_identity,
        research_config_inputs=research_config_inputs,
        live_config_sha256=live_hash,
        generated_at_utc=generated_at,
    )
    bundle_payload = bundle.payload
    publication = bundle_payload["publication"]
    assert isinstance(publication, dict)
    if publication["generation_sha256"] != generation_sha256:
        raise ResearchBundleValidationError(
            "Precomputed generation identity differs from the finalized bundle."
        )
    bundle_path = write_immutable_bundle(bundle, targets.bundle_directory)
    bundle_relative = _repo_relative(bundle_path, root)
    bundle_file_bytes = bundle.file_bytes()
    bundle_file_sha256 = hashlib.sha256(bundle_file_bytes).hexdigest()
    if not eligible:
        latest_state = build_latest_research_state(
            research_generation_sha256=generation_sha256,
            bundle_path=bundle_relative,
            bundle_sha256=bundle.sha256,
            bundle_file_sha256=bundle_file_sha256,
            eligible=False,
            completed_at_utc=generated_at,
        )
        latest_state_bytes = latest_state.file_bytes()
        verify_latest_research_state_bytes(
            latest_state_bytes,
            repo_root=root,
        )
        _assert_run_inputs_unchanged(
            repo_root=root,
            source_identity=source_identity,
            window_results=window_results,
        )
        _assert_resolved_live_inputs_unchanged(
            live_config_path=targets.live_config_path,
            research_config_inputs=research_config_inputs,
        )
        publish_files_atomically(
            {},
            commit_pointer=(targets.latest_state_path, latest_state_bytes),
            failure_injector=failure_injector,
        )
        return ResearchPublicationResult(
            bundle_path=bundle_path,
            bundle_sha256=bundle.sha256,
            research_generation_sha256=generation_sha256,
            published=False,
            pointer_path=None,
            latest_state_path=targets.latest_state_path,
        )
    assert live_bytes is not None and live_original_bytes is not None
    bundle_identity = bundle_payload["strategy"]
    assert isinstance(bundle_identity, dict)
    best_payload = {
        "schema_name": "tradeos.promoted_best_config_projection",
        "schema_version": 1,
        "saved_at": generated_at,
        "research_generation_sha256": generation_sha256,
        "bundle_path": bundle_relative,
        "bundle_sha256": bundle.sha256,
        "approved": approved,
        "rejection_reasons": list(rejection_reasons),
        "approval_policy": approval_policy.to_payload(),
        "dataset_provenance_sha256": dataset_set["provenance_sha256"],
        "config": {
            "symbols": list(canonical.contract.symbols),
            **canonical.contract.promoted_values,
        },
        "performance": metrics,
        "canonical_strategy_contract": strict_json_loads(
            canonical.canonical_json,
            label="canonical strategy",
        ),
        "canonical_strategy_json": canonical.canonical_json,
        "canonical_strategy_sha256": canonical.sha256,
        "strategy_config_sha256": bundle_identity["strategy_config_sha256"],
        "model_execution_evidence_sha256": model_evidence.sha256,
    }
    stability_payload = {
        "schema_name": "tradeos.promoted_stability_projection",
        "schema_version": 1,
        "saved_at": generated_at,
        "research_generation_sha256": generation_sha256,
        "bundle_path": bundle_relative,
        "bundle_sha256": bundle.sha256,
        "validation_windows": [window["lookback_days"] for window in window_results],
        "per_window": [
            {
                "lookback_days": window["lookback_days"],
                "dataset_provenance_sha256": window["dataset_provenance"][
                    "provenance_sha256"
                ],
                "canonical_strategy_sha256": window["best_row"][
                    "canonical_strategy_sha256"
                ],
                "model_execution_evidence_sha256": window["best_row"][
                    "model_execution_evidence_sha256"
                ],
                "selection_evidence_sha256": window["best_row"][
                    "research_selection_evidence"
                ]["evidence_sha256"],
                "performance": _selected_metrics(window["best_row"]),
            }
            for window in window_results
        ],
        "average_performance": stability["average_metrics"],
        "stability": dict(stability),
        "market_regime": dict(regime),
    }
    decision_payload = {
        "schema_name": "tradeos.promoted_trade_decision_projection",
        "schema_version": 1,
        "timestamp": generated_at,
        "research_generation_sha256": generation_sha256,
        "bundle_path": bundle_relative,
        "bundle_sha256": bundle.sha256,
        **dict(final_decision),
        "rejection_reasons": list(rejection_reasons),
        "stability_reasons": list(stability["reasons"]),
    }
    projected_generation_path = (
        targets.generation_directory
        / generation_directory_component(generation_sha256)
    )
    immutable_projection_paths = {
        "best_config": _repo_relative(
            projected_generation_path / "best_config_latest.json",
            root,
        ),
        "trade_decision": _repo_relative(
            projected_generation_path / "trade_decision.json",
            root,
        ),
        "stability_report": _repo_relative(
            projected_generation_path / "stability_report.json",
            root,
        ),
        "live_config": _repo_relative(
            projected_generation_path / "live_config.json",
            root,
        ),
    }
    dataset_windows = dataset_set["windows"]
    assert isinstance(dataset_windows, list)
    primary_dataset_window = dataset_windows[dataset_set["primary_window_index"]]
    assert isinstance(primary_dataset_window, dict)
    primary_provenance = primary_dataset_window["provenance"]
    assert isinstance(primary_provenance, dict)
    primary_observed = primary_provenance["observed"]
    assert isinstance(primary_observed, dict)
    drift_performance_names = (
        "trades_per_day",
        "win_rate",
        "profit_factor",
        "sharpe_ratio",
        "realized_pnl",
        "combined_score",
    )
    if any(name not in metrics for name in drift_performance_names):
        raise ResearchBundleValidationError(
            "Promoted drift projection requires all bound performance metrics."
        )
    signal_rate = metrics["trades_per_day"] / len(canonical.contract.symbols)
    drift_payload = {
        "schema_name": DRIFT_PROFILE_SCHEMA_NAME,
        "schema_version": DRIFT_PROFILE_SCHEMA_VERSION,
        "generated_at": generated_at,
        "approved": approved,
        "source_artifacts": {
        "dataset_provenance_sha256": dataset_set["provenance_sha256"],
            **immutable_projection_paths,
        "bundle": bundle_relative,
        },
        "context": {
            "strategy_mode": canonical.contract.promoted_values["strategy_mode"],
            "symbols": list(canonical.contract.symbols),
            "symbol_count": len(canonical.contract.symbols),
            "timeframe": primary_observed["timeframe"],
            "bar_timeframe_minutes": canonical.contract.promoted_values[
                "bar_timeframe_minutes"
            ],
            "stable": stability["stable"],
            "market_regime": regime["regime"],
        },
        "performance": {
            name: metrics[name] for name in drift_performance_names
        },
        "metrics": {
            "buy_signal_rate_per_symbol_per_day": signal_rate,
            "sell_signal_rate_per_symbol_per_day": signal_rate,
            "rejection_rate": DRIFT_PROFILE_POLICY_V1.rejection_rate,
            "rejection_breakdown": None,
            "hybrid_branch_participation": None,
            "avg_bar_age_s": DRIFT_PROFILE_POLICY_V1.avg_bar_age_s,
            "stale_bar_rate": DRIFT_PROFILE_POLICY_V1.stale_bar_rate,
        },
        "metric_sources": DRIFT_PROFILE_POLICY_V1.metric_sources(),
        "research_generation_sha256": generation_sha256,
        "bundle_path": bundle_relative,
        "bundle_sha256": bundle.sha256,
    }
    artifact_bytes = {
        "best_config_latest": json_file_bytes(best_payload),
        "live_config": live_bytes,
        "stability_report": json_file_bytes(stability_payload),
        "trade_decision": json_file_bytes(decision_payload),
        "drift_profile": json_file_bytes(drift_payload),
    }
    immutable_generation_path = write_immutable_generation(
        generation_sha256,
        {
            "best_config_latest.json": artifact_bytes["best_config_latest"],
            "live_config.json": artifact_bytes["live_config"],
            "stability_report.json": artifact_bytes["stability_report"],
            "trade_decision.json": artifact_bytes["trade_decision"],
            "drift_profile.json": artifact_bytes["drift_profile"],
        },
        targets.generation_directory,
    )
    if immutable_generation_path != projected_generation_path:
        raise ResearchPublicationError(
            "Immutable generation path differs from its full-digest projection."
        )
    pointer_artifacts = {
        "best_config_latest": (
            _repo_relative(
                immutable_generation_path / "best_config_latest.json",
                root,
            ),
            artifact_bytes["best_config_latest"],
        ),
        "live_config": (
            _repo_relative(immutable_generation_path / "live_config.json", root),
            artifact_bytes["live_config"],
        ),
        "stability_report": (
            _repo_relative(
                immutable_generation_path / "stability_report.json",
                root,
            ),
            artifact_bytes["stability_report"],
        ),
        "trade_decision": (
            _repo_relative(
                immutable_generation_path / "trade_decision.json",
                root,
            ),
            artifact_bytes["trade_decision"],
        ),
        "drift_profile": (
            _repo_relative(immutable_generation_path / "drift_profile.json", root),
            artifact_bytes["drift_profile"],
        ),
    }
    pointer = build_promotion_pointer(
        research_generation_sha256=generation_sha256,
        bundle_path=bundle_relative,
        bundle_sha256=bundle.sha256,
        bundle_file_sha256=bundle_file_sha256,
        artifacts=pointer_artifacts,
    )
    pointer_bytes = pointer.file_bytes()
    targets.pointer_path.parent.mkdir(parents=True, exist_ok=True)
    pointer_verification_path = (
        targets.pointer_path.parent
        / f".{targets.pointer_path.name}.{uuid4().hex}.verify"
    )
    try:
        with pointer_verification_path.open("xb") as handle:
            handle.write(pointer_bytes)
            handle.flush()
            os.fsync(handle.fileno())
        load_verified_promotion_pointer(
            pointer_verification_path,
            repo_root=root,
            require_latest_state=False,
        )
    finally:
        pointer_verification_path.unlink(missing_ok=True)
    latest_state = build_latest_research_state(
        research_generation_sha256=generation_sha256,
        bundle_path=bundle_relative,
        bundle_sha256=bundle.sha256,
        bundle_file_sha256=bundle_file_sha256,
        eligible=True,
        promotion_pointer_path=_repo_relative(targets.pointer_path, root),
        promotion_pointer=pointer,
        promotion_pointer_file_sha256=hashlib.sha256(pointer_bytes).hexdigest(),
        completed_at_utc=generated_at,
    )
    latest_state_bytes = latest_state.file_bytes()
    verify_latest_research_state_bytes(
        latest_state_bytes,
        repo_root=root,
        promotion_pointer_path=targets.pointer_path,
        promotion_pointer_bytes=pointer_bytes,
    )
    _assert_run_inputs_unchanged(
        repo_root=root,
        source_identity=source_identity,
        window_results=window_results,
    )
    _assert_resolved_live_inputs_unchanged(
        live_config_path=targets.live_config_path,
        research_config_inputs=research_config_inputs,
    )
    publications = {
        targets.best_config_path: artifact_bytes["best_config_latest"],
        targets.live_config_path: artifact_bytes["live_config"],
        targets.stability_report_path: artifact_bytes["stability_report"],
        targets.trade_decision_path: artifact_bytes["trade_decision"],
        targets.drift_profile_path: artifact_bytes["drift_profile"],
        targets.pointer_path: pointer_bytes,
    }
    publish_files_atomically(
        publications,
        commit_pointer=(targets.latest_state_path, latest_state_bytes),
        expected_originals={targets.live_config_path: live_original_bytes},
        failure_injector=failure_injector,
    )
    return ResearchPublicationResult(
        bundle_path=bundle_path,
        bundle_sha256=bundle.sha256,
        research_generation_sha256=generation_sha256,
        published=True,
        pointer_path=targets.pointer_path,
        latest_state_path=targets.latest_state_path,
    )


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

def _main_locked(args: argparse.Namespace) -> None:
    _section("run_research.py  —  starting")

    _log(f"Best config target : {BEST_CONFIG_PATH}")
    _log(f"Stability report   : {STABILITY_REPORT_PATH}")
    _log(f"Trade decision     : {TRADE_DECISION_PATH}")
    _log(f"Live config target : {LIVE_CONFIG_PATH}")
    _log(f"Drift profile      : {DRIFT_PROFILE_PATH}")

    window_results: list[dict] = []
    primary_sweep_csv: Path | None = None
    # `_main_locked` is entered only after ExecutionLock acquisition. Resolve
    # the one UTC anchor and all live-derived inputs here, never at import time.
    research_anchor_utc = _resolve_research_anchor_utc()
    resolved_inputs = _resolve_research_inputs(LIVE_CONFIG_PATH)
    existing_dataset_path: Path | None = None
    if args.dataset:
        existing_dataset_path = Path(args.dataset)
        if not existing_dataset_path.is_absolute():
            existing_dataset_path = SCRIPT_DIR / existing_dataset_path
        existing_dataset_path = existing_dataset_path.resolve(strict=False)
        if not existing_dataset_path.exists() or not existing_dataset_path.is_dir():
            raise RuntimeError(
                "Provided dataset path does not exist or is not a directory: "
                f"{existing_dataset_path}"
            )
        existing_manifest = load_verified_manifest(existing_dataset_path)
        window_ranges = [
            {
                "lookback_days": 0,
                "start_utc": _canonical_utc_text(
                    existing_manifest.get(
                        "start_utc",
                        existing_manifest.get("start"),
                    ),
                    field="existing dataset start_utc",
                ),
                "end_utc": _canonical_utc_text(
                    existing_manifest.get(
                        "end_utc",
                        existing_manifest.get("end"),
                    ),
                    field="existing dataset end_utc",
                ),
            }
        ]
    else:
        window_ranges = _fresh_window_ranges(research_anchor_utc)
    source_identity = capture_source_identity(SCRIPT_DIR)
    research_config_inputs = _research_config_inputs(
        args,
        resolved_inputs=resolved_inputs,
        research_anchor_utc=research_anchor_utc,
        window_ranges=window_ranges,
    )
    # Validate the exact declared search/policy inputs before any snapshot or
    # backtest work can begin.
    build_research_config_identity(research_config_inputs)

    try:
        if args.dataset:
            assert existing_dataset_path is not None
            dataset_path = existing_dataset_path

            _log("Research mode      : existing dataset")
            _log_dataset_details(dataset_path, context="Existing")

            sweep_csv = _timestamped_csv(RESULTS_DIR, "research_dataset")
            leaderboard_csv = _timestamped_csv(RESULTS_DIR, "leaderboard_dataset")
            dataset_checkpoint = capture_dataset_checkpoint(dataset_path)

            _section("Existing Dataset — Backtest")
            step_backtest(
                sweep_csv,
                dataset_path,
                resolved_inputs=resolved_inputs,
            )

            _section("Existing Dataset — Rank")
            best_row = step_rank_results(sweep_csv, leaderboard_csv)
            assert_dataset_checkpoint_unchanged(
                dataset_checkpoint,
                capture_dataset_checkpoint(dataset_path),
            )
            canonical = canonical_strategy_from_backtest_row(best_row)
            dataset_provenance = capture_dataset_provenance(
                dataset_path,
                canonical.contract,
            )

            window_results.append({
                "lookback_days": 0,
                "dataset": dataset_path,
                "sweep_csv": sweep_csv,
                "leaderboard_csv": leaderboard_csv,
                "best_row": best_row,
                "dataset_checkpoint": dataset_checkpoint,
                "dataset_provenance": dataset_provenance,
            })
            primary_sweep_csv = sweep_csv
        else:
            n = len(VALIDATION_WINDOWS)
            _log("Research mode      : fresh snapshots")
            _log(f"Validation windows : {VALIDATION_WINDOWS} days ({n} windows)")
            _log(f"Snapshot symbols   : {', '.join(resolved_inputs.symbols)}")

            # ── Phase 1: run snapshot + backtest + rank for every window ──────
            for i, days in enumerate(VALIDATION_WINDOWS, 1):
                _section(f"Window {i} of {n} ({days} days) — Snapshot")
                dataset_path = step_snapshot(
                    days,
                    resolved_inputs=resolved_inputs,
                    research_anchor_utc=research_anchor_utc,
                )
                _log_dataset_details(dataset_path, context=f"Window {days}d")

                sweep_csv = _timestamped_csv(RESULTS_DIR, f"research_{days}d")
                leaderboard_csv = _timestamped_csv(RESULTS_DIR, f"leaderboard_{days}d")
                dataset_checkpoint = capture_dataset_checkpoint(dataset_path)

                _section(f"Window {i} of {n} ({days} days) — Backtest")
                step_backtest(
                    sweep_csv,
                    dataset_path,
                    resolved_inputs=resolved_inputs,
                )

                _section(f"Window {i} of {n} ({days} days) — Rank")
                best_row = step_rank_results(sweep_csv, leaderboard_csv)
                assert_dataset_checkpoint_unchanged(
                    dataset_checkpoint,
                    capture_dataset_checkpoint(dataset_path),
                )
                canonical = canonical_strategy_from_backtest_row(best_row)
                dataset_provenance = capture_dataset_provenance(
                    dataset_path,
                    canonical.contract,
                )

                window_results.append({
                    "lookback_days": days,
                    "dataset": dataset_path,
                    "sweep_csv": sweep_csv,
                    "leaderboard_csv": leaderboard_csv,
                    "best_row": best_row,
                    "dataset_checkpoint": dataset_checkpoint,
                    "dataset_provenance": dataset_provenance,
                })

                if days == VALIDATION_WINDOWS[-1]:
                    primary_sweep_csv = sweep_csv

        # ── Phase 2: approve using the longest (most data-rich) window ────
        primary = window_results[-1]
        approval_label = (
            f"{primary['lookback_days']}-day window (primary)"
            if primary["lookback_days"] > 0
            else "existing dataset (primary)"
        )

        _section(f"Approval — {approval_label}")
        approved, rejection_reasons = step_approve_config(primary["best_row"])

        # ── Phase 3: stability evaluation across all windows ──────────────
        _section("Stability evaluation")
        stability = step_evaluate_stability(window_results)

        # ── Phase 4: market regime detection (primary dataset) ────────────
        _section("Market regime detection")
        regime = step_detect_regime(primary["dataset"])

        # ── Phase 5: combine into a single trade decision ─────────────────
        _section("Trade decision")
        decision = step_make_trade_decision(
            approved, rejection_reasons, stability, regime, primary["best_row"]
        )

        # ── Phase 6: immutable bundle/projections, latest-state marker LAST ─
        _section("Finalize research bundle")
        publication_result = finalize_research_bundle_and_publish(
            window_results=window_results,
            approved=approved,
            rejection_reasons=rejection_reasons,
            stability=stability,
            regime=regime,
            final_decision=decision,
            source_identity=source_identity,
            research_config_inputs=research_config_inputs,
            targets=_default_publication_targets(),
        )
        _log(f"Immutable bundle : {publication_result.bundle_path}")
        _log(f"Bundle SHA       : {publication_result.bundle_sha256}")
        _log(f"Generation SHA   : {publication_result.research_generation_sha256}")
        _log(f"Latest state     : {publication_result.latest_state_path}")
        if publication_result.published:
            _log(f"Commit pointer   : {publication_result.pointer_path}")
        else:
            _log("Current aliases unchanged: final decision is not promotion-eligible.")

        if publication_result.published:
            _section("Generate operator reports")
            try:
                for report_path in strategy_report.write_reports(SCRIPT_DIR):
                    _log(f"Report       : {report_path}")
            except Exception as exc:
                _log(
                    "Warning: committed promotion is valid, but operator report "
                    f"generation failed ({type(exc).__name__})."
                )

    except RuntimeError as exc:
        raise RuntimeError(str(exc)) from exc

    _section("Done")
    if primary_sweep_csv:
        _report_outputs(primary_sweep_csv)
    if window_results:
        primary = window_results[-1]
        try:
            log_experiment_run(
                run_type="research_pipeline",
                script_path=__file__,
                entrypoint=sys.argv[0],
                strategy_name=str(primary["best_row"].get("strategy_mode") or "unknown"),
                dataset_path=primary["dataset"],
                symbols=list(resolved_inputs.symbols),
                params={
                    "dataset_arg": args.dataset,
                    "validation_windows": VALIDATION_WINDOWS,
                    "snapshot_timeframe": resolved_inputs.timeframe,
                    "snapshot_feed": SNAPSHOT_FEED,
                    "snapshot_adjustment": SNAPSHOT_ADJUSTMENT,
                    "snapshot_align_mode": SNAPSHOT_ALIGN_MODE,
                    "sweep_params": dict(resolved_inputs.sweep_params),
                    "fixed_params": dict(FIXED_PARAMS),
                },
                metrics={
                    "total_return_pct": primary["best_row"].get("total_return_pct"),
                    "profit_factor": primary["best_row"].get("profit_factor"),
                    "sharpe": primary["best_row"].get("sharpe_ratio"),
                    "win_rate": primary["best_row"].get("win_rate"),
                    "max_drawdown_pct": primary["best_row"].get("max_drawdown_pct"),
                    "trade_count": primary["best_row"].get("total_trades"),
                    "expectancy": primary["best_row"].get("expectancy"),
                    "realized_pnl": primary["best_row"].get("realized_pnl"),
                },
                output_path=RESULTS_DIR,
                summary_path=publication_result.bundle_path,
                extra_fields={
                    "approved": approved,
                    "approval_reasons": rejection_reasons,
                    "stable": stability["stable"],
                    "stability_reasons": stability["reasons"],
                    "market_regime": regime["regime"],
                    "primary_sweep_csv": str(primary_sweep_csv) if primary_sweep_csv else None,
                    "promotion_bundle_path": str(publication_result.bundle_path),
                    "research_generation_sha256": (
                        publication_result.research_generation_sha256
                    ),
                },
            )
        except Exception as exc:
            _log(f"Warning: experiment log failed ({type(exc).__name__}).")


def main() -> None:
    args = parse_args()
    try:
        with ExecutionLock(RESEARCH_LOCK_PATH):
            _main_locked(args)
    except RuntimeError as exc:
        _section("FAILED")
        _log(f"Error: {exc}")
        sys.exit(1)


if __name__ == "__main__":
    main()

# Project Map

## Start Here

If you are new to this repo, read these in order:

1. `README.md` for the top-level workflows and safety boundaries.
2. `OPERATIONS.md` for the operator workflow used during live or paper sessions.
3. `TRADING_SPEC.md` for the frozen trading behavior and risk constraints.
4. `alpaca_trading_bot/cli.py` to understand the supported commands and runtime startup flow.
5. `trading_bot.py`, `strategy.py`, `dashboard.py`, and `backtest_runner.py` for the core runtime, strategy, UI, and offline simulation paths.

## What This Project Does

This repository is a trading system workspace centered on a shared strategy engine and a single CLI surface, `tradeos`.

Today it supports:

- live or paper execution through Alpaca
- monitoring through a Streamlit dashboard
- dataset snapshotting into versioned offline datasets
- backtesting against saved datasets
- research and validation scripts that write outputs under `results/`

The repo currently mixes core runtime code and many experiment scripts in the same top-level directory, so understanding "what is operational" versus "what is exploratory" is important.

## Canonical Boundaries

Use this mental model when deciding where code belongs:

- `live runtime`
  - execution, broker access, locks, session guards, risk controls, persistence
- `shared strategy`
  - indicators, signal rules, mode normalization, strategy configuration
- `dashboard`
  - read-only monitoring, drilldowns, and operator diagnostics
- `research`
  - backtests, validations, audits, sweeps, ranking, and experiment logging
- `artifacts`
  - `datasets/`, `logs/`, `results/`, and disposable local scratch output

If a change would affect live trading behavior, it belongs on the runtime or shared-strategy side and should be treated as high-risk. If a script exists only to answer a research question, it should live under `research/` even if a compatibility wrapper remains at the repo root.

## Main Workflows

### 1. Live or paper session

- Operator launches `.\start_dashboard.ps1` or uses `tradeos live` / `tradeos preview`.
- `alpaca_trading_bot/cli.py` validates the runtime, acquires the live lock, and delegates to `trading_bot.py`.
- `trading_bot.py` loads config, pulls market/account data through the broker adapter, applies shared strategy logic, enforces runtime protections, and persists snapshots.
- `dashboard.py` reads persisted state and startup artifacts for monitoring.

### 2. Dashboard-only monitoring

- Operator launches `.\start_dashboard.ps1 -DashboardOnly` or `tradeos dashboard`.
- `dashboard.py` renders current and historical state from SQLite plus log/startup artifacts.

### 3. Offline research and backtesting

- `dataset_snapshotter.py` creates versioned datasets under `datasets/`.
- `backtest_runner.py` replays those datasets through the shared strategy layer.
- `run_research.py` and the many `run_*` validation scripts orchestrate sweeps, diagnostics, and comparison studies.
- Outputs land under `results/`.

## Key Entry Points

### CLI and launchers

- `alpaca_trading_bot/cli.py`
  Main CLI implementation for `tradeos`.
- `tradeos/__main__.py`
  Package entry point for `python -m tradeos`.
- `tradeos/cli.py`
  Thin compatibility re-export.
- `start_dashboard.ps1`
  Preferred operator launcher for the live bot and dashboard.
- `run_research.ps1`
  PowerShell wrapper around the research pipeline.
- `run_compare_suite.ps1`
  Fixed comparison wrapper for a standard benchmark run.
- `run_breakout_research.ps1`
  Older focused wrapper for breakout experiments.

### Core runtime and shared logic

- `trading_bot.py`
  Live execution loop, runtime config loading, risk gates, broker interaction, logging, persistence, and session protections.
- `strategy.py`
  Shared strategy engine used by live trading and backtests.
- `storage.py`
  SQLite persistence for snapshots, orders, and dashboard history.
- `botlog.py`
  Structured JSONL logging under `logs/<date>/`.
- `dashboard.py`
  Streamlit monitoring UI.
- `dashboard_state.py`
  Dashboard-side loading, normalization, drilldowns, and derived UI state.

### Data, broker, and ML support

- `tradeos/brokers/base.py`
  Broker abstraction.
- `tradeos/brokers/alpaca_broker.py`
  Alpaca implementation used by the runtime.
- `dataset_snapshotter.py`
  Historical market data snapshotting into `datasets/`.
- `ml/feature_pipeline.py`
  Offline ML feature alignment.
- `ml/train.py`
  Offline model training.
- `ml/predict.py`
  Offline inference helpers used by live and backtest paths.

### Research and experiment scripts

Current evidence suggests these are exploratory or validation-oriented rather than core runtime entry points:

- `research/`
  Home for grouped research-only validation, audit, and experiment scripts.
- `research/legacy_experiments/`
  Home for older sweep-era and historical experiment runners preserved for archival reproducibility.
- root-level wrappers preserved for compatibility for:
  - `run_edge_audit.py`
  - `run_edge_diagnostics.py`
  - `run_cross_sectional_edge_audit.py`
  - `run_feed_comparison_validation.py`
  - `run_long_horizon_validation.py`
  - `run_momentum_breakout_validation.py`
  - `run_trade_path_diagnostics.py`
  - `run_trend_pullback_*`
  - `run_volatility_expansion_validation.py`
  - `run_hybrid_bb_mr_research.py`
  - `audit_dataset_spacing.py`
  - `inspect_dataset_and_run.py`
  - `investigate_missing_bars.py`
  - `rebuild_research_dataset_sip.py`
  - `rebuild_research_dataset_sip_clean.py`
  - `breakout_eval.py`
  - `breakout_sweep.py`
  - `compare_mr_strategies.py`
  - `vwap_sweep.py`
  - `vwap_atr_sweep.py`

Still intentionally at root for now:

- `run_research.py`
- `run_backtest_experiments.py`

They are important to preserve, but they are not part of the normal live operator workflow.

## Module Responsibilities

### Current operational path

- `alpaca_trading_bot/cli.py` owns command parsing, startup validation, lock management, and delegation.
- `trading_bot.py` owns live-cycle execution, runtime safety checks, bar processing, and broker-backed order flow.
- `strategy.py` owns signal generation and strategy configuration.
- `tradeos/brokers/alpaca_broker.py` owns account, market data, and order APIs.
- `storage.py` and `botlog.py` own persistent runtime observability.
- `dashboard.py` and `dashboard_state.py` own operator-facing visibility.

### Supporting but not on the critical live path

- `backtest_runner.py` owns offline simulation and sweep output generation.
- `dataset_snapshotter.py` owns historical data capture.
- `daily_report.py` and `drift_monitor.py` provide diagnostics and post-run analysis.
- `universe.py` appears intentionally standalone and not wired into live symbol selection yet.

### Compatibility and legacy surfaces

- `alpaca_trading_bot/`
  Legacy package name retained for compatibility with the CLI implementation.
- `tradeos/`
  Current package namespace and broker abstraction home.
- `repo_quarantine/`
  Deliberately quarantined legacy launcher and desktop UI code that is not part of the active operator path.

## Config Flow

There are three main config paths:

### 1. Secrets and live risk controls

- Source: `.env`
- Used by: `trading_bot.py`
- Examples: `ALPACA_API_KEY`, `ALPACA_API_SECRET`, `ALPACA_PAPER`, trade/risk limits

### 2. Runtime trading settings

- Source: `config/live_config.json`
- Used by: `trading_bot.py` and surfaced in launcher/dashboard metadata
- Examples: symbol universe, strategy mode, timeframe, strategy-specific parameters

### 3. Offline CLI arguments

- Source: command-line flags
- Used by: `backtest_runner.py`, `dataset_snapshotter.py`, and the various `run_*` scripts

The important mental model is:

- `.env` is for secrets and guardrails
- `config/live_config.json` is the runtime trading-settings source of truth
- CLI flags drive offline and research workflows

## How Live, Research, and Backtesting Relate

- `strategy.py` is the shared center of gravity.
- `trading_bot.py` uses it for live or preview decisions.
- `backtest_runner.py` uses it for offline simulation against saved datasets.
- `dataset_snapshotter.py` creates those datasets.
- research scripts call into the backtest and dataset tooling to generate outputs and compare strategy variants.

This shared-logic setup is good for consistency, but it means edits in `strategy.py`, `trading_bot.py`, and any broker-facing runtime assumptions can affect both research outputs and live behavior.

## Data, Logs, and Generated Outputs

### Runtime and monitoring state

- `bot_history.db`
  Default SQLite history database for runtime snapshots.
- `logs/`
  JSONL runtime logs and startup artifacts.

### Offline research data

- `datasets/`
  Versioned datasets, typically `bars.parquet` plus `manifest.json`.
- `results/`
  Shared artifact area for current promoted artifacts plus research output directories.
- `results/archive/`
  Historical research outputs preserved for reference.

### Test and local transient artifacts

The repo currently contains many untracked local artifacts such as:

- `test_botlog_*/`
- `test_logs_*/`
- `test_dashboard_logs_*/`
- `test_dashboard_state_*.db`
- `inspect_dataset_*/`
- `tmp_test_artifacts/`
- `output/`

These appear to be generated by tests or local analysis runs rather than source code.

### Artifact policy notes

- `results/` root is intentionally reserved for a small set of current promoted artifacts and shared summaries
- active research outputs should prefer dedicated subdirectories under `results/`
- `results/experiments/` is the existing home for batch experiment suites
- `results/archive/` is the conservative archival home for older historical outputs
- scratch paths such as `output/`, `.tmp/`, `tmp_test_artifacts/`, and `inspect_dataset_*/` should remain disposable

## What Looks Current vs Legacy

### Current

- `tradeos` CLI workflow
- `start_dashboard.ps1`
- `trading_bot.py`
- `strategy.py`
- `dashboard.py`
- `dashboard_state.py`
- `storage.py`
- `dataset_snapshotter.py`
- `backtest_runner.py`
- `OPERATIONS.md`
- `TRADING_SPEC.md`
- `V2_CLEANUP_EXECUTION_PLAN.md`

### Compatibility-only

- `alpaca_trading_bot/`
  Still active because the CLI implementation and script entrypoints point here, but the user-facing name is now `tradeos`.
- `alpaca-bot` script alias in `pyproject.toml`
  Kept for compatibility, not the preferred current name.

### Legacy or quarantined

- `repo_quarantine/`
  Holding area for intentionally removed or retired material. The old desktop runtime has already been removed.

### Current but structurally messy

- root-level research compatibility wrappers
- some historical sweep-era entry points still exist at root as wrappers for compatibility
- experiment-specific result trees under `results/`

These appear useful, but they make the top level noisy.

## Current Cleanup Direction

The near-term plan is not a full rewrite. It is a controlled `v2-in-place`:

- preserve the current `tradeos` CLI and operator workflow
- preserve live trading behavior unless a task explicitly targets behavior
- extract shared internals into smaller modules before attempting larger moves
- prefer wrappers and compatibility layers over breaking command paths

See `V2_CLEANUP_EXECUTION_PLAN.md` for the staged execution plan.

## High-Risk Files

Edit these carefully:

- `trading_bot.py`
  Live execution loop, kill switch behavior, order gating, flatten logic, and session protections live here.
- `strategy.py`
  Shared by live and backtest paths.
- `alpaca_trading_bot/cli.py`
  Live lock behavior, startup validation, and command routing.
- `start_dashboard.ps1`
  Operator entrypoint and runtime approval behavior.
- `tradeos/brokers/alpaca_broker.py`
  Broker-facing behavior and market data assumptions.
- `storage.py`
  Dashboard/runtime persistence contract.
- `dashboard_state.py`
  Large state-normalization layer with many tests and UI dependencies.
- `config/live_config.json`
  Active runtime trading settings.
- `TRADING_SPEC.md`
  Behavioral contract document tied to sensitive trading behavior.
- `OPERATIONS.md`
  Operator runbook for safe day-of usage.

## Likely Cleanup Targets

- `.gitignore`
  Missing ignores for several generated local/test artifacts.
- top-level research script sprawl
  Too many focused `run_*` scripts at the repo root for easy orientation.
- naming leftovers
  `tradeos` is the current name, but compatibility aliases and some quarantine docs still use older `alpaca-bot` / `AlpacaTradingBot` wording.
- `results/`
  Mixes current decision artifacts and many experiment outputs in one tree.
- `README.md`
  Needs to act as a shorter front door now that the repo has grown.

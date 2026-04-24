# Research Boundary Audit

## Summary

This repo currently mixes three different kinds of Python at the top level:

- runtime-sensitive system code
- core-adjacent offline utilities
- research and validation scripts

The live/runtime path is already fairly clear, but the root directory is noisy because many research-only scripts live beside the operational code. The safest cleanup direction is to group the clearly research-only scripts while leaving compatibility wrappers at their old locations.

## 1. Core / Runtime

These files and paths are part of the active operator/runtime surface and should remain easy to find:

- `alpaca_trading_bot/`
- `tradeos/`
- `trading_bot.py`
- `strategy.py`
- `storage.py`
- `dashboard.py`
- `dashboard_state.py`
- `botlog.py`
- `symbol_state.py`
- `start_dashboard.ps1`
- `config/live_config.json`
- `OPERATIONS.md`
- `TRADING_SPEC.md`

Core-adjacent but not live-execution code:

- `dataset_snapshotter.py`
- `backtest_runner.py`
- `daily_report.py`
- `drift_monitor.py`
- `ml/`

These are important enough to keep visible for now.

## 2. Research / Experiments

### Clearly research-only root scripts

These are focused on validation, audits, sweeps, diagnostics, or one-off research analysis rather than the active operator path:

- `audit_dataset_spacing.py`
- `inspect_dataset_and_run.py`
- `investigate_missing_bars.py`
- `rebuild_research_dataset_sip.py`
- `rebuild_research_dataset_sip_clean.py`
- `run_cross_sectional_edge_audit.py`
- `run_edge_audit.py`
- `run_edge_diagnostics.py`
- `run_feed_comparison_validation.py`
- `run_hybrid_bb_mr_research.py`
- `run_long_horizon_validation.py`
- `run_momentum_breakout_validation.py`
- `run_trade_path_diagnostics.py`
- `run_trend_pullback_clean_dataset_comparison.py`
- `run_trend_pullback_exit_capture_clean.py`
- `run_trend_pullback_exit_comparison.py`
- `run_trend_pullback_oos_validation.py`
- `run_trend_pullback_robustness.py`
- `run_trend_pullback_robustness_clean.py`
- `run_volatility_expansion_validation.py`

### Research scripts that should stay at root for now

- `run_research.py`
  Important top-level research pipeline already referenced by docs, tests, CLI passthrough, and PowerShell wrappers.
- `run_backtest_experiments.py`
  Closely tied to the `tradeos experiments` CLI route and wrapper scripts.

### Historical research scripts suitable for a legacy archival area

- `breakout_eval.py`
- `breakout_sweep.py`
- `compare_mr_strategies.py`
- `vwap_sweep.py`
- `vwap_atr_sweep.py`

These are older sweep-era experiment runners. They are clearly research-only, but they still need compatibility wrappers because they are listed in packaging metadata and may still be used by old commands or notes.

### Research files that should stay put for now

- `universe.py`

This file appears exploratory, but it is small, standalone, and not obviously part of the same historical experiment cluster as the sweep runners.

## 3. Docs / Runbooks

- `README.md`
- `PROJECT_MAP.md`
- `CLEANUP_AUDIT.md`
- `DESKTOP_REMOVAL_AUDIT.md`
- `OPERATIONS.md`
- `TRADING_SPEC.md`
- `repo_quarantine/README.md`

## 4. Generated Artifacts / Results / Logs

- `results/`
- `datasets/`
- `logs/`
- `output/`
- `test_botlog_*/`
- `test_logs_*/`
- `test_dashboard_logs_*/`
- `test_dashboard_state_*.db`
- `inspect_dataset_*/`
- `tmp*`
- `bot_history.db`

These should not be treated as source code boundaries.

## 5. Legacy / Quarantine

- `repo_quarantine/`

The repo no longer contains the old desktop runtime, but this path still serves as a holding area for retired material.

## 6. Tests

- `tests/`

The tests currently mirror the mixed source layout: they cover both runtime-sensitive code and many research-only scripts.

## Root-Level Files That Must Stay Where They Are For Now

- `trading_bot.py`
- `strategy.py`
- `storage.py`
- `dashboard.py`
- `dashboard_state.py`
- `backtest_runner.py`
- `dataset_snapshotter.py`
- `daily_report.py`
- `alpaca_trading_bot/`
- `tradeos/`
- `start_dashboard.ps1`
- `run_research.py`
- `run_backtest_experiments.py`

## Files That Could Be Safely Grouped Into a Dedicated Research Area

The safest first-wave candidates are the clearly research-only validation and audit scripts listed above, especially the cluster around:

- edge audits and diagnostics
- feed and horizon validation
- trend-pullback validation
- dataset inspection and spacing audits
- SIP dataset rebuild helpers

The next low-risk archival step is the older sweep-era cluster:

- `breakout_eval.py`
- `breakout_sweep.py`
- `compare_mr_strategies.py`
- `vwap_sweep.py`
- `vwap_atr_sweep.py`

These files already behave like a family and are heavily represented in tests.

## Risks And Constraints

### Import compatibility

Many tests import research scripts directly from their current root-level module names. Some scripts also import each other by root-level name. Hard moves without compatibility wrappers would break those imports.

### CLI and wrapper compatibility

`run_research.py` and `run_backtest_experiments.py` are routed through the current CLI implementation. Moving them is possible, but not necessary for the first low-risk pass.

### Packaging metadata

Several older root modules are listed directly in `pyproject.toml` under `py-modules`. That raises the cost of moving older scripts compared with newer research-only scripts where wrappers can absorb most of the change.

### Runtime safety boundary

Anything that can influence live behavior, strategy behavior, broker behavior, or runtime config flow should remain outside this reorganization.

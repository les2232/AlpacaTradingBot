# Artifact Policy Audit

## Summary

This repo already has most of the important artifact locations in stable, recognizable places, but the boundaries are only implicit today.

The main finding is:

- `logs/` is runtime and operational
- `datasets/` is offline research input data
- `results/` is mixed and should be treated as a controlled shared artifact area, not a dumping ground
- `results/archive/` already exists and is the safest home for historical research outputs
- local scratch paths such as `output/`, `.tmp/`, `tmp_test_artifacts/`, `inspect_dataset_*/`, and `test_*` artifacts should remain ignored and disposable

## Audited Paths

### `logs/`

- What it appears to contain: dated runtime JSONL logs under `logs/<YYYY-MM-DD>/`, plus startup and launcher artifacts under `logs/launches/`
- Classification: runtime / operational
- Recommendation: stay
- Risk notes: active runtime code and operator workflows reference this path directly; do not rename or relocate it casually

### `logs/launches/`

- What it appears to contain: launch metadata JSON plus transient stdout/stderr log captures from launcher flows
- Classification: runtime / operational
- Recommendation: stay
- Risk notes: useful for operator diagnostics; treat as local operational history, not curated repo content

### `datasets/`

- What it appears to contain: versioned dataset snapshots named with symbol set, timeframe, date range, feed, and hash
- Classification: active research input data
- Recommendation: stay
- Risk notes: config examples, research scripts, backtests, and docs reference this path directly; changing it would create broad compatibility risk

### `results/`

- What it appears to contain: a mix of current promoted research artifacts, report files, validation subdirectories, and historical comparison outputs
- Classification: mixed shared artifact area
- Recommendation: stay, but document internal policy more clearly
- Risk notes: several files at the `results/` root are read directly by scripts and docs, so a hard split would break existing workflows

### `results/best_config_latest.json`

- What it appears to contain: canonical current promoted research configuration
- Classification: current operational artifact
- Recommendation: stay at `results/` root for now
- Risk notes: read directly by `run_research.py`, `daily_report.py`, `drift_monitor.py`, tests, and docs

### `results/trade_decision.json`

- What it appears to contain: current repo-level trade decision artifact aligned with promoted research output
- Classification: current operational artifact
- Recommendation: stay at `results/` root for now
- Risk notes: read directly by runtime-adjacent reporting and monitoring helpers

### `results/strategy_status.md`

- What it appears to contain: human-readable summary of the current promoted research state and supporting context
- Classification: current operational artifact
- Recommendation: stay at `results/` root for now
- Risk notes: referenced by operator docs; moving it would create unnecessary churn

### `results/stability_report.json`

- What it appears to contain: shared research summary used as a current reference artifact
- Classification: current operational artifact
- Recommendation: stay at `results/` root for now
- Risk notes: still part of the repo-level "current state" story even if it may need refreshes later

### `results/experiments/`

- What it appears to contain: timestamped or named experiment batches created by `run_backtest_experiments.py` and related research flows
- Classification: active research outputs
- Recommendation: stay and continue using it for batched experiment runs
- Risk notes: currently ignored in `.gitignore`, which is appropriate for large local experiment runs

### `results/<named_validation_dir>/`

- What it appears to contain: active research run outputs such as `edge_*`, `trend_pullback_*`, `dataset_spacing_audit*`, and similar validation directories
- Classification: active research outputs
- Recommendation: stay under `results/`, but prefer one directory per run/topic
- Risk notes: many current research scripts already write into clearly named subdirectories here; this is a useful pattern to preserve

### `results/archive/`

- What it appears to contain: older flat-file sweep outputs and historical comparison CSVs preserved for reference
- Classification: archival research
- Recommendation: stay and use as the manual archival destination for older outputs that are no longer part of the current decision surface
- Risk notes: do not mass-move files into it automatically without a human review pass

### `output/`

- What it appears to contain: ad hoc generated deliverables such as the earlier PDF output under `output/pdf/`
- Classification: disposable local output
- Recommendation: ignore
- Risk notes: useful as a scratch delivery area, but should not become a source-of-truth artifact location

### `.tmp/`

- What it appears to contain: temporary cache directories such as pytest cache spillover
- Classification: disposable local output
- Recommendation: ignore
- Risk notes: safe to treat as scratch only

### `tmp_test_artifacts/`

- What it appears to contain: temporary test-created artifact trees, especially dataset inspection and rebuild scratch directories
- Classification: disposable local output
- Recommendation: ignore
- Risk notes: tests intentionally create and discard under this path; do not treat as durable research output

### `inspect_dataset_*/`

- What it appears to contain: top-level temporary inspection directories from dataset inspection tooling
- Classification: disposable local output
- Recommendation: ignore
- Risk notes: local convenience output only

### `test_botlog_*/`, `test_logs_*/`, `test_dashboard_logs_*/`, `test_dashboard_state_*.db`, `test_decision_claim_*.db`

- What it appears to contain: test-generated runtime/log/database fixtures materialized on disk
- Classification: disposable local output
- Recommendation: ignore
- Risk notes: these are noisy but expected; deletion should be optional and manual, not part of normal repo cleanup

### `bot_history.db`

- What it appears to contain: local runtime SQLite history database
- Classification: runtime / operational
- Recommendation: stay local and ignored
- Risk notes: operationally useful, but should not be treated as committed source data

## Overall Recommendation

Use a conservative policy centered on the existing paths:

1. Keep `logs/` and `datasets/` exactly where they are.
2. Treat a small set of root-level `results/` files as the current operational decision surface.
3. Keep active research runs in dedicated subdirectories under `results/`, with `results/experiments/` reserved for batch experiment suites.
4. Use `results/archive/` for manually archived historical outputs.
5. Keep scratch and test artifacts ignored rather than trying to curate them in-repo.

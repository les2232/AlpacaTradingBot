# Artifact Storage Policy

## Purpose

This policy defines where generated artifacts belong so the repo stays understandable without breaking existing TradeOS workflows.

It is intentionally conservative. Existing runtime and research paths remain valid unless a later change explicitly migrates them.

## Policy

### 1. Current operational artifacts

These are the small set of artifacts that represent the repo's current promoted research and operator-facing state.

They stay at the `results/` root for compatibility:

- `results/best_config_latest.json`
- `results/trade_decision.json`
- `results/strategy_status.md`
- `results/stability_report.json`
- closely related current-state reports written by the main research pipeline

Rules:

- keep these files small and human-reviewable
- do not turn the `results/` root into a general experiment dump
- do not move these files without updating all code, tests, and docs that read them directly

### 2. Active research outputs

Active research outputs belong in dedicated subdirectories under `results/`.

Preferred patterns:

- `results/experiments/<run_name>/` for batch experiment suites
- `results/<research_topic>/` for named validation, audit, or diagnostics runs

Examples already in use:

- `results/experiments/...`
- `results/edge_audit_smoke/`
- `results/feed_comparison_validation/`
- `results/trend_pullback_oos_validation/`
- `results/dataset_spacing_audit/`

Rules:

- prefer one folder per run, topic, or validation pass
- keep related JSON, CSV, and markdown outputs together
- choose folder names that describe the research question, not just "output" or "test"

### 3. Archived historical experiment outputs

Historical experiment outputs that are no longer part of the active decision surface belong in:

- `results/archive/`

Use this area for:

- older sweep outputs
- superseded comparison CSVs
- preserved historical experiment evidence

Rules:

- archive manually after review; do not auto-migrate large result sets blindly
- preserve filenames when reproducibility matters
- prefer adding a short note or README before large archival moves

### 4. Runtime and operational logs

Operational logs stay under:

- `logs/<YYYY-MM-DD>/`
- `logs/launches/`

Rules:

- treat `logs/` as local operational history
- do not commit routine log output
- do not repurpose `logs/` for research summaries or long-term reports

### 5. Offline datasets

Offline datasets stay under:

- `datasets/<dataset_name>/`

Rules:

- keep dataset naming descriptive and stable
- include symbol set, timeframe, date range, feed, and distinguishing suffix/hash where available
- treat datasets as local/generated data, not lightweight source files

### 6. Disposable local and debug artifacts

These should remain untracked and disposable:

- `output/`
- `.tmp/`
- `tmp_test_artifacts/`
- `inspect_dataset_*/`
- `test_botlog_*/`
- `test_logs_*/`
- `test_dashboard_logs_*/`
- `test_dashboard_state_*.db`
- `test_decision_claim_*.db`
- similar one-off scratch directories

Rules:

- use these for local debugging, ad hoc exports, and test scratch only
- do not store canonical research outputs here
- cleanup should be manual and optional

## Naming Conventions

Use these naming preferences for future artifacts:

- current promoted files: explicit names such as `best_config_latest.json` or `trade_decision.json`
- active research folders: `<topic>`, `<topic>_<variant>`, or `<topic>_<timestamp>` when repeated runs matter
- experiment batches: `results/experiments/<preset>_<timestamp>/`
- archived files: keep original names unless a human-reviewed archival pass introduces a stronger convention

Avoid names like:

- `output`
- `misc`
- `new_results`
- `test2`

unless they are explicitly inside ignored scratch space.

## What Should Not Be Committed

By default, do not commit:

- routine runtime logs
- local datasets created for experimentation
- disposable scratch exports
- transient test artifact directories
- large experiment batches that only support local iteration

Commit only when the artifact is intentionally part of the repo's shared current-state story or a curated historical record.

## Current Compatibility Constraints

This policy does not change the existing compatibility surface:

- code and docs still expect some canonical files at `results/` root
- runtime logs still live under `logs/`
- datasets still live under `datasets/`

If a future cleanup wants stricter physical separation such as `results/current/`, that should be a dedicated migration with code, test, and documentation updates.

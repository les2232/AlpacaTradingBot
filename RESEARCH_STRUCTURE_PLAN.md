# Research Structure Plan

## Proposed Low-Risk Structure

```text
repo root
|- alpaca_trading_bot/        # CLI implementation and compatibility package
|- tradeos/                   # current package namespace and broker abstractions
|- research/                  # research-only scripts and helpers
|  |- __init__.py
|  |- audit_dataset_spacing.py
|  |- inspect_dataset_and_run.py
|  |- investigate_missing_bars.py
|  |- legacy_experiments/     # older sweep-era and archival experiment runners
|  |- rebuild_research_dataset_sip.py
|  |- rebuild_research_dataset_sip_clean.py
|  |- run_cross_sectional_edge_audit.py
|  |- run_edge_audit.py
|  |- run_edge_diagnostics.py
|  |- run_feed_comparison_validation.py
|  |- run_hybrid_bb_mr_research.py
|  |- run_long_horizon_validation.py
|  |- run_momentum_breakout_validation.py
|  |- run_trade_path_diagnostics.py
|  |- run_trend_pullback_*.py
|  |- run_volatility_expansion_validation.py
|- tests/
|- results/
|- datasets/
|- logs/
```

## Rationale

- Keep runtime-sensitive code visible at the repo root.
- Group clearly research-only scripts into one obvious area.
- Preserve current imports and commands through root-level compatibility wrappers.
- Avoid broad package rewrites or changes to the active operator path.

## Safe Moves Now

Move the clearly research-only validation/audit cluster into `research/`:

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

## Moves To Defer

Keep these at root for now:

- `run_research.py`
- `run_backtest_experiments.py`
- `backtest_runner.py`
- `dataset_snapshotter.py`
- `daily_report.py`
- `universe.py`

Reasons to defer:

- direct CLI routing
- existing PowerShell wrappers
- packaging metadata already points at them
- less clarity about external expectations

## Compatibility Strategy

For each moved file:

1. Move the real implementation into `research/`.
2. Leave a root-level wrapper module at the old path.
3. The wrapper should:
   - re-export the moved module contents
   - preserve `python old_script.py` behavior by calling `main()` when executed directly

This keeps:

- test imports stable
- docs mostly valid during the transition
- ad hoc command usage stable

## Staged Rollout Plan

### Stage 1

- Create `research/` package.
- Move the clearly research-only validation/audit scripts.
- Add root-level wrappers.
- Update docs to explain the boundary.

### Stage 2

- After imports and habits settle, decide whether `run_research.py` and `run_backtest_experiments.py` should also become wrappers.
- Status: older sweep-era scripts now live in `research/legacy_experiments/` with root-level compatibility wrappers.

### Stage 3

- If desired later, reorganize tests to mirror the source boundary, for example:
  - `tests/core/`
  - `tests/research/`
  - `tests/ops/`

## Constraints

- Do not move live/runtime-sensitive files.
- Do not change strategy or broker behavior.
- Do not change runtime config semantics.
- Prefer wrappers over import churn.

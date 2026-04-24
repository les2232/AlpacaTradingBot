# Results Directory

This directory is a shared artifact area with three different roles:

- current promoted research artifacts kept at the `results/` root for compatibility
- active research runs stored in dedicated subdirectories
- archived historical outputs stored under `results/archive/`

Current root-level files such as `best_config_latest.json`, `trade_decision.json`, and `strategy_status.md` are part of the repo's current decision surface and should stay easy to find.

`results/experiment_log.jsonl` is also a root-level shared artifact now. It is the append-only research experiment history used by the dashboard's experiment-history view.

For new outputs:

- use a dedicated subdirectory for active research runs
- use `results/experiments/` for batch experiment suites
- use `results/archive/` for reviewed historical material that is no longer part of the active decision surface

Recently archived historical material includes older `compare_suite_*` runs, earlier `phase*` experiment bundles, and older root-level comparison and walk-forward artifacts.

Avoid writing ad hoc one-off files directly into the `results/` root unless they are intended to become shared current-state artifacts.

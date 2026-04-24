# Results Curation Audit

## Summary

This pass inventories the current `results/` tree and classifies the important files and directories without moving anything.

Evidence used for classification:

- direct code references in `run_research.py`, `daily_report.py`, `drift_monitor.py`, tests, and docs
- `results/strategy_status.md`
- current folder naming patterns under `results/`, `results/archive/`, and `results/experiments/`
- timestamps showing which outputs are part of the newer research passes versus older sweep-era work

## Root-Level Current / Compatibility-Sensitive Artifacts

### `results/best_config_latest.json`

- Type of artifact: JSON promoted research config
- Likely purpose: canonical current promoted strategy/research artifact
- Last-known role if inferable: directly read by `run_research.py`, `daily_report.py`, `drift_monitor.py`, tests, and referenced by docs
- Classification: current
- Reason for classification: explicitly part of the repo's current decision surface and compatibility-sensitive
- Risk if moved: high; code, tests, and runbooks would break or drift

### `results/trade_decision.json`

- Type of artifact: JSON current decision artifact
- Likely purpose: current repo-level trade decision summary aligned with promoted research
- Last-known role if inferable: directly read by `run_research.py`, `drift_monitor.py`, and described in `strategy_status.md`
- Classification: current
- Reason for classification: actively part of the current promoted artifact set
- Risk if moved: high; compatibility and monitoring/reporting assumptions would break

### `results/strategy_status.md`

- Type of artifact: markdown summary
- Likely purpose: operator-facing summary of the currently promoted strategy and supporting research
- Last-known role if inferable: referenced in `OPERATIONS.md`
- Classification: current
- Reason for classification: explicitly used as a human-facing current-state artifact
- Risk if moved: medium to high; operator docs would immediately become stale

### `results/stability_report.json`

- Type of artifact: JSON research summary
- Likely purpose: repo-level stability evidence artifact from the research pipeline
- Last-known role if inferable: written by `run_research.py`; called out in `strategy_status.md` as stale but still relevant
- Classification: current
- Reason for classification: still part of the canonical root-level promoted artifact set even if it needs a refresh later
- Risk if moved: high; `run_research.py` and tests assume this path

## Root-Level Active Research Families

### `results/live_study_2026-04-13_to_2026-04-16.md`

- Type of artifact: markdown study/report
- Likely purpose: focused investigation derived from recent live logs and runtime history
- Last-known role if inferable: references `logs/2026-04-13` through `logs/2026-04-16` and `bot_history.db`
- Classification: ambiguous
- Reason for classification: recent and potentially useful, but not part of the canonical current artifact set and not clearly archived
- Risk if moved: medium; likely low code risk, but human context could be lost

### `results/bollinger_squeeze_smoke*.csv`

- Type of artifact: root-level CSV family
- Likely purpose: smoke-test experiment outputs for a Bollinger squeeze variant
- Last-known role if inferable: recent root outputs dated 2026-04-17
- Classification: active research
- Reason for classification: recent experiment evidence, but not promoted to canonical current artifacts
- Risk if moved: low code risk, medium workflow risk if someone is still comparing them manually

### `results/hybrid_bb_mr_smoke*.csv`

- Type of artifact: root-level CSV family
- Likely purpose: smoke-test results for hybrid Bollinger-band mean reversion research
- Last-known role if inferable: recent root outputs dated 2026-04-17
- Classification: active research
- Reason for classification: recent research support files, not canonical promoted artifacts
- Risk if moved: low code risk, medium workflow risk

### `results/hybrid_bb_mr_branch_persist_smoke*.csv`

- Type of artifact: root-level CSV family
- Likely purpose: branch-persistence variant smoke-test results for hybrid research
- Last-known role if inferable: recent root outputs dated 2026-04-17
- Classification: active research
- Reason for classification: recent, clearly experimental, and still near active hybrid research
- Risk if moved: low code risk, medium workflow risk

### `results/walk_forward_summary.json` and `results/walk_forward_windows.csv`

- Type of artifact: root-level walk-forward summary pair
- Likely purpose: generic walk-forward evaluation outputs
- Last-known role if inferable: dated 2026-04-14; not directly referenced by active docs
- Classification: ambiguous
- Reason for classification: looks important enough to preserve, but not clearly part of the current promoted artifact set
- Risk if moved: medium; could break a human analyst's mental model even if code is unaffected

### `results/wf_*.json` and `results/wf_*.csv`

- Type of artifact: root-level walk-forward family
- Likely purpose: named walk-forward trials such as breakout baseline, SMA trend, stage 3, test, and wick-fade comparisons
- Last-known role if inferable: dated 2026-04-14 to 2026-04-15; looks like intermediate research evidence
- Classification: archive candidate
- Reason for classification: older, named comparative runs that do not appear in current compatibility-sensitive code or docs
- Risk if moved: low code risk, medium reproducibility-context risk if moved without notes

### `results/strategy_compare_global*.csv` and `results/strategy_compare_mixed*.csv`

- Type of artifact: root-level comparison CSV families
- Likely purpose: earlier strategy comparison outputs across baseline modes
- Last-known role if inferable: dated 2026-04-04; older than current mean-reversion promotion
- Classification: archive candidate
- Reason for classification: older flat-file comparisons at root with no evidence of current operational use
- Risk if moved: low code risk, low to medium analyst-history risk

## Root-Level Active Research Directories

### `results/cross_sectional_edge_audit/`

- Type of artifact: research result directory
- Likely purpose: active edge audit diagnostics across symbols/time slices
- Last-known role if inferable: recent 2026-04-18 research directory with JSON/CSV diagnostics
- Classification: active research
- Reason for classification: clearly part of the newer organized research output pattern
- Risk if moved: low code risk, medium workflow risk if recent analysis is still ongoing

### `results/cross_sectional_edge_audit_block_a/`

- Type of artifact: research result directory
- Likely purpose: first split/block variant of cross-sectional edge audit
- Last-known role if inferable: recent 2026-04-18 run
- Classification: active research
- Reason for classification: recent structured research output
- Risk if moved: low code risk, medium workflow risk

### `results/cross_sectional_edge_audit_block_b/`

- Type of artifact: research result directory
- Likely purpose: second split/block variant of cross-sectional edge audit
- Last-known role if inferable: recent 2026-04-18 run
- Classification: active research
- Reason for classification: recent structured research output
- Risk if moved: low code risk, medium workflow risk

### `results/dataset_spacing_audit/`

- Type of artifact: research result directory
- Likely purpose: dataset spacing validation against a baseline dataset
- Last-known role if inferable: recent 2026-04-18 audit
- Classification: active research
- Reason for classification: part of the current dataset-validation research pass
- Risk if moved: low code risk, medium workflow risk

### `results/dataset_spacing_audit_amd/`

- Type of artifact: research result directory
- Likely purpose: symbol-specific AMD spacing audit
- Last-known role if inferable: recent 2026-04-18 audit
- Classification: active research
- Reason for classification: recent focused validation output
- Risk if moved: low code risk, medium workflow risk

### `results/dataset_spacing_audit_sip/`

- Type of artifact: research result directory
- Likely purpose: SIP-feed spacing audit
- Last-known role if inferable: recent 2026-04-18 audit
- Classification: active research
- Reason for classification: recent feed-quality validation output
- Risk if moved: low code risk, medium workflow risk

### `results/dataset_spacing_audit_sip_clean/`

- Type of artifact: research result directory
- Likely purpose: clean regular-session SIP spacing audit
- Last-known role if inferable: recent 2026-04-18 audit
- Classification: active research
- Reason for classification: recent refined dataset-validation output
- Risk if moved: low code risk, medium workflow risk

### `results/edge_audit_momentum_breakout_smoke/`

- Type of artifact: research result directory
- Likely purpose: smoke-level edge audit for momentum breakout
- Last-known role if inferable: recent 2026-04-18 run
- Classification: active research
- Reason for classification: recent strategy-validation output
- Risk if moved: low code risk, medium workflow risk

### `results/edge_audit_smoke/`

- Type of artifact: research result directory
- Likely purpose: generic smoke-level edge audit
- Last-known role if inferable: recent 2026-04-17 run
- Classification: active research
- Reason for classification: recent validation output
- Risk if moved: low code risk, medium workflow risk

### `results/edge_audit_volatility_expansion/`

- Type of artifact: research result directory
- Likely purpose: edge audit for volatility expansion research
- Last-known role if inferable: recent 2026-04-18 run
- Classification: active research
- Reason for classification: recent structured validation output
- Risk if moved: low code risk, medium workflow risk

### `results/edge_diagnostics_current/`

- Type of artifact: research result directory
- Likely purpose: compare diagnostics under explicit config vs live-effective config
- Last-known role if inferable: recent 2026-04-17 directory with `explicit_config/` and `live_effective/`
- Classification: ambiguous
- Reason for classification: the name suggests current relevance, but it is still research/diagnostic rather than canonical runtime output
- Risk if moved: medium; could hide a still-useful operator-facing diagnostic trail

### `results/feed_comparison_validation/`

- Type of artifact: research result directory
- Likely purpose: compare feed-dependent backtest/validation outcomes
- Last-known role if inferable: recent 2026-04-18 validation
- Classification: active research
- Reason for classification: current feed-validation work
- Risk if moved: low code risk, medium workflow risk

### `results/long_horizon_validation_volatility_expansion/`

- Type of artifact: research result directory
- Likely purpose: long-horizon validation for volatility expansion strategy
- Last-known role if inferable: recent 2026-04-18 validation
- Classification: active research
- Reason for classification: recent organized validation output
- Risk if moved: low code risk, medium workflow risk

### `results/missing_bar_analysis/`

- Type of artifact: research result directory
- Likely purpose: missing-bar investigation output
- Last-known role if inferable: recent 2026-04-18 analysis
- Classification: active research
- Reason for classification: recent diagnostic output tied to dataset quality work
- Risk if moved: low code risk, medium workflow risk

### `results/momentum_breakout_validation/`

- Type of artifact: research result directory
- Likely purpose: momentum breakout validation outputs
- Last-known role if inferable: recent 2026-04-18 validation
- Classification: active research
- Reason for classification: current organized strategy-validation output
- Risk if moved: low code risk, medium workflow risk

### `results/trade_path_diagnostics/`

- Type of artifact: research result directory
- Likely purpose: trade-path level diagnostic outputs
- Last-known role if inferable: recent 2026-04-17 run
- Classification: active research
- Reason for classification: recent deeper-dive diagnostics
- Risk if moved: low code risk, medium workflow risk

### `results/trend_pullback_clean_dataset_comparison/`

- Type of artifact: research result directory
- Likely purpose: clean-dataset comparison for trend-pullback behavior
- Last-known role if inferable: recent 2026-04-18 run
- Classification: active research
- Reason for classification: current strategy-validation output
- Risk if moved: low code risk, medium workflow risk

### `results/trend_pullback_exit_capture_clean/`

- Type of artifact: research result directory
- Likely purpose: clean-dataset exit capture analysis for trend pullback
- Last-known role if inferable: recent 2026-04-18 run
- Classification: active research
- Reason for classification: recent structured strategy analysis
- Risk if moved: low code risk, medium workflow risk

### `results/trend_pullback_exit_comparison/`

- Type of artifact: research result directory
- Likely purpose: exit-style comparison for trend pullback
- Last-known role if inferable: recent 2026-04-17 run
- Classification: active research
- Reason for classification: recent active validation output
- Risk if moved: low code risk, medium workflow risk

### `results/trend_pullback_oos_validation/`

- Type of artifact: research result directory
- Likely purpose: out-of-sample validation for trend pullback
- Last-known role if inferable: recent 2026-04-18 run
- Classification: active research
- Reason for classification: current organized validation output
- Risk if moved: low code risk, medium workflow risk

### `results/trend_pullback_oos_validation_second_block/`

- Type of artifact: research result directory
- Likely purpose: second OOS block validation for trend pullback
- Last-known role if inferable: recent 2026-04-18 run
- Classification: active research
- Reason for classification: recent follow-on validation output
- Risk if moved: low code risk, medium workflow risk

### `results/trend_pullback_robustness/`

- Type of artifact: research result directory
- Likely purpose: robustness checks for trend pullback
- Last-known role if inferable: recent 2026-04-17 run
- Classification: active research
- Reason for classification: current validation output
- Risk if moved: low code risk, medium workflow risk

### `results/trend_pullback_robustness_clean/`

- Type of artifact: research result directory
- Likely purpose: clean-dataset robustness checks for trend pullback
- Last-known role if inferable: recent 2026-04-18 run
- Classification: active research
- Reason for classification: current validation output
- Risk if moved: low code risk, medium workflow risk

### `results/volatility_expansion_validation/`

- Type of artifact: research result directory
- Likely purpose: organized volatility-expansion validation outputs
- Last-known role if inferable: recent 2026-04-18 run with summary, regime, and signal files
- Classification: active research
- Reason for classification: recent and clearly part of the newer research workflow
- Risk if moved: low code risk, medium workflow risk

## Results Directories With Likely Historical Status

### `results/compare_suite_20260407_185508/`

- Type of artifact: timestamped comparison suite directory
- Likely purpose: older compare-suite run across baseline, ML threshold, and dataset variants
- Last-known role if inferable: explicitly called stale in `results/strategy_status.md`
- Classification: archive candidate
- Reason for classification: older compare-suite work that predates the current mean-reversion promotion
- Risk if moved: low code risk, low to medium history-reference risk

### `results/compare_suite_20260407_190253/`

- Type of artifact: timestamped comparison suite directory
- Likely purpose: follow-up compare-suite run with ML threshold sweeps
- Last-known role if inferable: explicitly called stale in `results/strategy_status.md`
- Classification: archive candidate
- Reason for classification: older compare-suite work that predates the current promoted strategy
- Risk if moved: low code risk, low to medium history-reference risk

### `results/phase1/`

- Type of artifact: early-phase result directory
- Likely purpose: earlier SMA-focused exploratory sweep results
- Last-known role if inferable: dated 2026-04-05; older than current research organization
- Classification: archive candidate
- Reason for classification: older phase-labeled experiment bundle with no sign of current compatibility sensitivity
- Risk if moved: low code risk, medium analyst-history risk

### `results/phase2/`

- Type of artifact: early-phase result directory
- Likely purpose: second exploratory phase, likely time-filter experiments
- Last-known role if inferable: dated 2026-04-05; older than current research organization
- Classification: archive candidate
- Reason for classification: older phase-labeled experiment bundle with no sign of current compatibility sensitivity
- Risk if moved: low code risk, medium analyst-history risk

## `results/experiments/`

### `results/experiments/`

- Type of artifact: experiment-suite parent directory
- Likely purpose: dedicated home for batched experiment runs created by `run_backtest_experiments.py` and related flows
- Last-known role if inferable: directly referenced by code and current storage policy docs
- Classification: active research
- Reason for classification: already the intended active location for experiment-batch outputs
- Risk if moved: high; code and docs would need updates

### `results/experiments/breakout_baseline_compare_20260410_234133/`

- Type of artifact: timestamped batch experiment directory
- Likely purpose: breakout baseline comparison batch generated by `run_backtest_experiments.py`
- Last-known role if inferable: current experiment-suite format with report and comparison summaries
- Classification: archive candidate
- Reason for classification: structurally active, but the content is tied to older breakout baseline work
- Risk if moved: low code risk, low to medium history-reference risk

### `results/experiments/hybrid_bb_mr_research_smoke/`

- Type of artifact: named experiment directory
- Likely purpose: smoke run for current hybrid BB/MR research
- Last-known role if inferable: recent 2026-04-17 active research batch
- Classification: active research
- Reason for classification: recent and aligned with ongoing hybrid research naming
- Risk if moved: low code risk, medium workflow risk

### `results/experiments/hybrid_bb_mr_research_extended/`

- Type of artifact: named experiment directory
- Likely purpose: extended hybrid BB/MR research run
- Last-known role if inferable: recent 2026-04-17 active research batch
- Classification: active research
- Reason for classification: recent organized experiment suite
- Risk if moved: low code risk, medium workflow risk

### `results/experiments/hybrid_bb_mr_research_q50_q60/`

- Type of artifact: named experiment directory
- Likely purpose: quantile-focused hybrid BB/MR research batch
- Last-known role if inferable: recent 2026-04-17 experiment suite
- Classification: active research
- Reason for classification: recent and clearly tied to current hybrid research exploration
- Risk if moved: low code risk, medium workflow risk

### `results/experiments/hybrid_bb_mr_quality_q50_q60/`

- Type of artifact: named experiment directory
- Likely purpose: quality-focused follow-on run for hybrid BB/MR quantile research
- Last-known role if inferable: recent 2026-04-17 experiment suite
- Classification: active research
- Reason for classification: recent and closely tied to current hybrid research flow
- Risk if moved: low code risk, medium workflow risk

### `results/experiments/smoke_check/`

- Type of artifact: named experiment directory
- Likely purpose: early smoke validation of the experiment-suite wrapper
- Last-known role if inferable: dated 2026-04-10; generic name suggests harness validation rather than promoted research
- Classification: ambiguous
- Reason for classification: probably archiveable, but the generic name makes it less obvious whether it is still kept as a tooling sanity check
- Risk if moved: low code risk, medium process-history risk

### `results/experiments/smoke_wrapper/`

- Type of artifact: named experiment directory
- Likely purpose: wrapper-path smoke validation for experiment orchestration
- Last-known role if inferable: dated 2026-04-10; generic tooling-oriented name
- Classification: ambiguous
- Reason for classification: likely older harness validation, but not enough evidence to move confidently without human review
- Risk if moved: low code risk, medium process-history risk

## `results/archive/`

### `results/archive/`

- Type of artifact: archive directory
- Likely purpose: reviewed holding area for historical sweep outputs and flat comparison artifacts
- Last-known role if inferable: already documented as the archival home in current repo docs
- Classification: archive candidate
- Reason for classification: this is the intended archive destination itself
- Risk if moved: high; would undercut the current storage policy

### Files under `results/archive/`

- Type of artifact: archived flat-file CSV families
- Likely purpose: older breakout, ML, leaderboard, mean-reversion, filter, ORB, and research sweep outputs preserved for reproducibility
- Last-known role if inferable: mostly dated 2026-04-04 through 2026-04-14; names match older sweep-era work and prior leaderboard runs
- Classification: archive candidate
- Reason for classification: these files are already curated into the archive area and should generally remain there
- Risk if moved: low code risk, medium reproducibility risk if reorganized without a naming/index plan

## Overall Notes

- Definitely current and compatibility-sensitive: `best_config_latest.json`, `trade_decision.json`, `strategy_status.md`, `stability_report.json`
- Clearly active research: the newer named validation directories under `results/` and most recent hybrid experiment suites under `results/experiments/`
- Strong archive candidates: `compare_suite_*`, `phase1/`, `phase2/`, older root-level comparison/walk-forward families, and the older breakout experiment batch under `results/experiments/`
- Ambiguous and worth human review before any move: `live_study_2026-04-13_to_2026-04-16.md`, `edge_diagnostics_current/`, generic walk-forward root summaries, and `results/experiments/smoke_check/` / `smoke_wrapper/`

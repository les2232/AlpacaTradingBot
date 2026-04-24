# Results Curation Plan

## What Should Remain At `results/` Root

Keep these at the root because they are either compatibility-sensitive or explicitly part of the current promoted artifact surface:

- `results/best_config_latest.json`
- `results/trade_decision.json`
- `results/strategy_status.md`
- `results/stability_report.json`

Keep these at the root for now unless a human explicitly decides otherwise:

- `results/live_study_2026-04-13_to_2026-04-16.md`
- `results/walk_forward_summary.json`
- `results/walk_forward_windows.csv`

Reason:

- the first group is directly referenced by code, tests, or runbooks
- the second group may still be useful as recent shared analysis, but they are not safe enough to archive blindly

## Good Archive Candidates

These look like the safest candidates for a later manual archival move:

- `results/compare_suite_20260407_185508/`
- `results/compare_suite_20260407_190253/`
- `results/phase1/`
- `results/phase2/`
- root-level `strategy_compare_global*`
- root-level `strategy_compare_mixed*`
- root-level `wf_*`
- `results/experiments/breakout_baseline_compare_20260410_234133/`

Why:

- they predate the current promoted mean-reversion state
- they are not directly referenced by current code paths
- several are explicitly described as older or stale in `results/strategy_status.md`

## Items That Should Stay In Place For Compatibility

Do not move these without a dedicated compatibility migration:

- `results/best_config_latest.json`
- `results/trade_decision.json`
- `results/stability_report.json`
- `results/experiments/`

Do not move casually because of operator/documentation expectations:

- `results/strategy_status.md`

## Items That Need Manual Review Before Any Move

- `results/live_study_2026-04-13_to_2026-04-16.md`
- `results/edge_diagnostics_current/`
- `results/walk_forward_summary.json`
- `results/walk_forward_windows.csv`
- root-level `bollinger_squeeze_smoke*`
- root-level `hybrid_bb_mr_smoke*`
- root-level `hybrid_bb_mr_branch_persist_smoke*`
- `results/experiments/smoke_check/`
- `results/experiments/smoke_wrapper/`

Why:

- these look non-canonical, but they are either recent, generically named, or potentially still useful to current ongoing analysis

## Staged Migration Proposal

### Stage 1: Manual labeling only

- keep the current structure intact
- use this audit as the decision record
- if helpful, add README guidance to `results/experiments/`

### Stage 2: Human-reviewed archival move

After a quick human confirmation, archive the clearest old material:

- `compare_suite_*`
- `phase1/`
- `phase2/`
- root-level older comparison families
- root-level `wf_*`
- older breakout batch under `results/experiments/`

Preferred destination:

- `results/archive/`

### Stage 3: Root cleanup of non-canonical recent outputs

After the old material is archived, review whether recent root-level research families should be grouped into dedicated subdirectories instead of remaining as loose CSVs at the root.

Priority review candidates:

- `bollinger_squeeze_smoke*`
- `hybrid_bb_mr_smoke*`
- `hybrid_bb_mr_branch_persist_smoke*`

### Stage 4: Optional future migration

If the team later wants stronger separation, consider a dedicated migration to something like:

- `results/current/`
- `results/research/`
- `results/archive/`

That should only happen with code, tests, docs, and compatibility updates in the same pass.

## Safest Next Manual Cleanup Step

The safest next manual cleanup step is:

1. review and approve archiving `compare_suite_*`, `phase1/`, `phase2/`, `strategy_compare_*`, and `wf_*`
2. leave all compatibility-sensitive root files where they are
3. defer recent diagnostic and smoke outputs until it is clearer whether they are still part of ongoing analysis

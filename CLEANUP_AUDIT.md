# Cleanup Audit

This audit is intentionally conservative. Findings are grouped by how safely they can be addressed without risking live behavior or destroying potentially useful research material.

## Safe Now

### 1. Missing ignore rules for generated local artifacts

- File/path: `.gitignore`
- Issue: The repo already ignores many generated files, but it does not ignore `test_botlog_*/`, `tmp_test_artifacts/`, `inspect_dataset_*/`, `.tmp/`, or `output/`.
- Why it matters: These directories are currently showing up as untracked clutter and make it harder to distinguish real source changes from local artifacts.
- Recommended action: Add narrowly targeted ignore rules for those generated paths.

### 2. Quarantine docs still use older CLI naming

- File/path: `repo_quarantine/README.md`
- Issue: The README says nothing in `alpaca-bot live`, `alpaca-bot preview`, or `alpaca-bot dashboard` should depend on the quarantined files.
- Why it matters: The active user-facing workflow is now `tradeos`, so the quarantine docs create avoidable naming confusion.
- Recommended action: Update the wording to reference `tradeos`, and note the older name only as compatibility context.

### 3. Legacy desktop quarantine README references the old command name as if it were current

- File/path: `repo_quarantine/legacy_desktop_runtime/README.md`
- Issue: The README describes the retired workflow as conflicting with `alpaca-bot live` and `alpaca-bot dashboard`.
- Why it matters: New engineers may read the quarantine docs while orienting and get the wrong impression about the supported operator path.
- Recommended action: Rewrite the wording around `tradeos`, while explicitly noting that `alpaca-bot` remains only as a compatibility alias.

### 4. Top-level onboarding docs need a repo map and explicit safety guidance

- File/path: `README.md`
- Issue: The existing README contains useful detail, but the repo now has enough moving parts that a clearer front door and safer orientation structure are needed.
- Why it matters: New engineers need a short path to the operator docs, trading spec, key modules, and sensitive files before touching code.
- Recommended action: Tighten the README and add a dedicated `PROJECT_MAP.md`.

## Safe With Review

### 5. Compatibility naming layers are still mixed into the main repo

- File/path: `alpaca_trading_bot/`, `tradeos/`, `pyproject.toml`
- Issue: The current product name is `tradeos`, but the implementation still lives partly under `alpaca_trading_bot`, and `pyproject.toml` still exposes an `alpaca-bot` script alias.
- Why it matters: This is a frequent source of confusion when tracing imports or deciding which name is canonical.
- Recommended action: Keep behavior as-is for now. Later, decide whether to keep the compatibility alias indefinitely or migrate fully to a single package/script name with a deliberate compatibility plan.

### 6. Root-level research script sprawl makes the repo hard to scan

- File/path: top-level `run_*.py`, `audit_dataset_spacing.py`, `investigate_missing_bars.py`, `inspect_dataset_and_run.py`, `rebuild_research_dataset_sip*.py`
- Issue: Many focused research and validation scripts live beside core runtime code at the repo root.
- Why it matters: It blurs the line between operational code and experiments, especially for new contributors.
- Recommended action: Partially addressed by grouping the clearest research-only scripts under `research/` and archiving older sweep-era runners under `research/legacy_experiments/`, both with compatibility wrappers left at the root. A later pass can decide whether any remaining historical helpers should also move.

### 7. Results tree mixes current decision artifacts and historical experiments

- File/path: `results/`
- Issue: `results/` contains current status artifacts such as `best_config_latest.json` alongside many historical validation and comparison outputs.
- Why it matters: Engineers have to mentally separate current decision support from archived experiment output.
- Recommended action: Addressed conservatively by defining an artifact policy: keep compatibility-sensitive current artifacts at the `results/` root, keep active research in dedicated subdirectories, and use `results/archive/` for reviewed historical material.

### 8. Backward-looking names are still present in compatibility layers

- File/path: `alpaca_trading_bot/__init__.py`, `trading_bot.py`
- Issue: Older `AlpacaTradingBot` naming still appears in compatibility shims.
- Why it matters: Some of this is intentional, but not all occurrences are equally important.
- Recommended action: Leave code references alone unless they affect active workflows. Prefer documenting the naming history rather than renaming internals casually.

### 9. Test coverage mirrors the growing script surface area, but the structure is noisy

- File/path: `tests/`
- Issue: The tests directory now includes many focused test files for one-off research scripts as well as core runtime tests.
- Why it matters: This is good coverage, but it reinforces the repo-root script sprawl and makes core-versus-experiment boundaries harder to see.
- Recommended action: Keep the tests. Later, consider grouping tests by `core`, `research`, and `ops` if the code itself is reorganized.

## Risky / Do Not Touch Yet

### 11. Live bot safeguards are intentionally duplicated

- File/path: `trading_bot.py`
- Issue: End-of-day flatten protection exists both in the live decision path and in a background thread fail-safe.
- Why it matters: This looks redundant at first glance, but the duplication is explicitly intentional for runtime safety.
- Recommended action: Do not deduplicate without a specific design review and production-style validation.

### 12. Dashboard and runtime remain tightly coupled

- File/path: `dashboard.py`, `dashboard_state.py`, `trading_bot.py`
- Issue: The dashboard still relies on the live bot's config and initialization path rather than a separate read-only service layer.
- Why it matters: Untangling that relationship would be a real architectural change and could easily break the operator workflow.
- Recommended action: Document the coupling and leave it alone for now.

### 13. Research and runtime share the strategy engine

- File/path: `strategy.py`, `backtest_runner.py`, `trading_bot.py`
- Issue: The same strategy layer is used by both live trading and offline simulation.
- Why it matters: This is desirable for consistency, but edits here carry cross-cutting behavioral risk.
- Recommended action: Treat `strategy.py` as a sensitive shared contract; do not refactor it casually as a cleanup exercise.

### 14. Active runtime config should not be normalized by cleanup work

- File/path: `config/live_config.json`
- Issue: This file is the source of truth for the active runtime trading setup.
- Why it matters: Even well-intentioned cleanup edits could change live or preview behavior.
- Recommended action: Do not rewrite it as part of repo tidying. Only change it as part of an explicit strategy/runtime decision.

### 15. Untracked research outputs should not be deleted just because they are noisy

- File/path: untracked directories under `results/`, `datasets/`, `test_*`, `inspect_dataset_*`, `tmp*`
- Issue: Many local artifacts are clearly generated, but some may still be supporting active analysis.
- Why it matters: Deleting them without explicit confirmation could destroy useful local state or work in progress.
- Recommended action: Prefer ignore rules and documentation over deletion unless the owner explicitly wants the local artifacts cleaned out.

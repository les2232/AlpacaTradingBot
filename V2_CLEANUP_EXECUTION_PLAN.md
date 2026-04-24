# V2 Cleanup Execution Plan

## Goal

Clean up the repository architecture without restarting from zero and without destabilizing the live trading path.

This plan assumes:

- the current repo remains the source of truth
- live behavior is preserved unless a change is explicitly marked as a behavior change
- we improve structure first, then decide whether deeper redesign is still needed

## Recommendation

Do a controlled `v2-in-place`, not a full rewrite.

Keep:

- the current trading specification
- the shared strategy engine
- the live CLI and launcher flow
- the broker adapter and persistence/logging contracts
- the existing tests as the safety net

Refactor:

- repository boundaries
- giant multi-responsibility files
- config normalization between live and backtest
- research script placement
- artifact hygiene

Do not treat cleanup as strategy redesign. The first pass is about shape, ownership, and safety.

## Success Criteria For Week 1

By the end of this cleanup pass:

- a new contributor can tell what is `live`, `research`, `dashboard`, and `artifacts`
- the repo root is quieter and easier to scan
- generated artifacts are less invasive
- the highest-risk files are mapped into smaller internal modules or have a staged extraction plan
- tests still pass for the touched surface area
- no live trading behavior changes are introduced accidentally

## Architectural Direction

Target mental model:

```text
repo root
|- alpaca_trading_bot/        # CLI compatibility layer
|- tradeos/
|  |- brokers/               # broker integrations
|  |- runtime/               # live bot runtime internals
|  |- strategy/              # shared strategy logic and indicators
|  |- config/                # shared config models/parsing
|  |- reporting/             # daily report and diagnostics helpers
|  |- dashboard/             # dashboard-side state and rendering helpers
|- research/                 # research-only runners and helpers
|- tests/
|  |- core/
|  |- research/
|  |- ops/
|- config/
|- datasets/
|- logs/
|- results/
```

Important note:

- this is the direction, not a one-shot rename plan
- week 1 should prefer extraction and wrappers over broad renames
- `alpaca_trading_bot` can remain as the CLI compatibility entrypoint for now

## Safety Rules

Do not change these as part of cleanup unless the task explicitly becomes behavior work:

- `config/live_config.json`
- session windows and flatten behavior
- daily-loss kill switch behavior
- broker order semantics
- one-live-process lock behavior
- price collar and stale-data protections
- signal thresholds unless the task is explicitly strategy tuning

Treat these files as high-risk:

- `trading_bot.py`
- `strategy.py`
- `backtest_runner.py`
- `dashboard.py`
- `dashboard_state.py`
- `alpaca_trading_bot/cli.py`
- `tradeos/brokers/alpaca_broker.py`
- `storage.py`
- `start_dashboard.ps1`

## Proposed Week 1 Plan

### Day 1: Freeze Boundaries

Objective:

- make the architecture explicit before moving code

Tasks:

- confirm the canonical boundaries:
  - live runtime
  - shared strategy engine
  - dashboard
  - research
  - artifacts
- update the top-level docs so they match reality
- add or tighten `.gitignore` rules for known generated local artifacts
- define a small protected-files list in docs for anything tied to live behavior

Deliverables:

- updated repo map
- updated ignore rules
- no code-path changes yet

### Day 2: Clean The Repo Surface

Objective:

- reduce repo-root noise without changing behavior

Tasks:

- move clearly research-only scripts under `research/` if any still remain at root
- leave root-level compatibility wrappers where existing workflows depend on them
- keep `run_research.py` and `run_backtest_experiments.py` at root unless the wrappers are trivial and tested
- group result and archive docs so current vs historical outputs are obvious

Deliverables:

- quieter repo root
- preserved command compatibility
- no runtime behavior changes

### Day 3: Extract Shared Config Models

Objective:

- stop duplicating large config surfaces between live and backtest

Tasks:

- identify overlapping config structures in `trading_bot.py` and `backtest_runner.py`
- create a shared config module under `tradeos/config/` or similar
- move normalization helpers and shared defaults that are clearly cross-cutting
- keep live-only settings and backtest-only settings separate where behavior differs

Guardrails:

- preserve JSON and CLI compatibility
- do not silently change defaults
- add or update tests around config parsing

Deliverables:

- shared config layer introduced
- duplicated config logic reduced

### Day 4: Break Up `strategy.py`

Objective:

- reduce risk and cognitive load in the shared strategy engine

Tasks:

- extract pure indicator/math helpers into focused modules
- extract strategy mode constants and normalization helpers into a dedicated module
- keep a stable import surface from `strategy.py` during the transition
- avoid changing signal behavior unless a test proves parity

Suggested internal split:

- `tradeos/strategy/constants.py`
- `tradeos/strategy/indicators.py`
- `tradeos/strategy/time_windows.py`
- `tradeos/strategy/modes.py`

Deliverables:

- smaller, more navigable strategy code
- stable external imports preserved

### Day 5: Break Up `trading_bot.py`

Objective:

- separate orchestration from policy and I/O

Tasks:

- extract runtime config loading
- extract session/calendar guards
- extract order gating and protection checks
- extract symbol evaluation preparation
- keep the top-level live loop as the orchestrator

Suggested internal split:

- `tradeos/runtime/config_loader.py`
- `tradeos/runtime/session_guards.py`
- `tradeos/runtime/order_guards.py`
- `tradeos/runtime/evaluation.py`
- `tradeos/runtime/models.py`

Guardrails:

- preserve the current lock, kill-switch, and flatten behavior
- do not deduplicate safety checks just because they look repetitive

Deliverables:

- `trading_bot.py` becomes smaller and more readable
- safety logic becomes easier to test directly

### Day 6: Break Up `backtest_runner.py` And Dashboard Dependencies

Objective:

- separate offline simulation plumbing from reporting and data preparation

Tasks:

- extract dataset preparation/loading helpers
- extract metrics/reporting helpers
- extract strategy-runner logic from CLI argument and file output handling
- identify dashboard helpers that can move into `tradeos/dashboard/` without changing the Streamlit entrypoint

Suggested internal split:

- `tradeos/backtest/data_loader.py`
- `tradeos/backtest/runner.py`
- `tradeos/backtest/reporting.py`
- `tradeos/dashboard/state.py`
- `tradeos/dashboard/rendering.py`

Deliverables:

- smaller offline runner
- clearer separation between UI rendering and state shaping

### Day 7: Stabilize, Test, And Decide What Comes Next

Objective:

- finish the week with a safer repo, not an incomplete refactor pile

Tasks:

- run the highest-signal tests for touched areas
- verify CLI paths still resolve
- verify docs still match the code
- list deferred work instead of pushing more structural change into the same pass

Required review questions:

- did we improve navigation?
- did we reduce duplication?
- did we preserve live behavior?
- are any wrappers now safe to retire?
- is a future repo split still necessary, or has `v2-in-place` solved most of the pain?

Deliverables:

- updated architecture status note
- list of deferred items for week 2

## Order Of Operations

Preferred sequencing:

1. docs and ignore hygiene
2. root-structure cleanup
3. shared config extraction
4. strategy extraction
5. runtime extraction
6. backtest/dashboard extraction
7. verification

This order matters because it creates clarity before it creates motion.

## What To Postpone

Do not try to solve these in the first week:

- renaming every `alpaca_trading_bot` reference to `tradeos`
- changing live strategy behavior
- rewriting the dashboard UX from scratch
- migrating every historical result into a perfect archive taxonomy
- deleting old local artifacts unless their owner explicitly approves it
- moving to multiple repos immediately

Those are all reasonable later, but they are not the best first move.

## Immediate Candidate File Moves

If we keep the repo single-package for now, these are the highest-value extractions:

- from `strategy.py`
  - indicators
  - normalization helpers
  - time-window logic
  - strategy-mode constants
- from `trading_bot.py`
  - dataclasses and config parsing
  - session guard logic
  - order/risk guard logic
  - symbol evaluation helpers
- from `backtest_runner.py`
  - dataset loading
  - feature preparation
  - result summarization/report generation
- from `dashboard.py`
  - render helper blocks
  - state formatting helpers

## Concrete Definition Of Done

Week 1 is successful if:

- the repo root is easier to understand in under five minutes
- at least one of the giant core files has been meaningfully decomposed
- shared config duplication has been reduced
- compatibility wrappers keep existing commands working
- touched tests pass
- no runtime safety protections were weakened

## Recommended First Implementation Slice

If we want the safest first coding pass, do this first:

1. tighten `.gitignore`
2. finish boundary docs
3. extract shared config models from `trading_bot.py` and `backtest_runner.py`
4. extract strategy constants and indicator helpers from `strategy.py`

That gives a real win without putting the live bot loop at immediate risk.

## Bottom Line

The project does not need to be thrown away.

It needs:

- stronger boundaries
- smaller modules
- calmer artifact hygiene
- less duplication between live and research paths

The safest path is to preserve the current behavior while reorganizing the internals into a more intentional `v2` shape.

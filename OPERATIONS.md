# Operations

## Current Strategy Selection

Active live mode: **`mean_reversion`** (set in `config/live_config.json`, validated 2026-04-08).

- IS profit factor: 1.189 → OOS profit factor: 1.431 (15-symbol universe, Oct 2025–Apr 2026 / Apr–Oct 2025)
- `hybrid` mode is available but not the current live selection.
- See `results/strategy_status.md` for supporting research files and caveats.

## Source Of Truth

- `.env` is for secrets, `ALPACA_PAPER`, and account/risk settings.
- `config/live_config.json` is the source of truth for active runtime trading settings:
  - symbols
  - strategy
  - timeframe
  - indicator/filter settings
- If `config/live_config.json` differs from `.env`, the bot will override `.env` for supported runtime fields and print which fields were overridden at startup.

## Supported Interfaces

- The supported operational interface is the `tradeos` CLI.
- The supported monitoring UI is the Streamlit dashboard.
- The older desktop UI layer has been intentionally removed and should not be recreated as part of routine operational work.

## One Operator Workflow

Use exactly one execution process and one monitoring process.

1. Start the full live session with one click:

```powershell
.\start_dashboard.ps1
```

This is the preferred launcher. By default it starts `tradeos live`, then starts or reuses the dashboard, and opens the browser when the dashboard is ready. It also keeps the live approval gate on by default.

If you are deliberately testing an unapproved research config, use the explicit override:

```powershell
.\start_dashboard.ps1 -AllowUnapprovedRuntime
```

Treat that as a temporary research-mode launch, not the normal production path.

2. Confirm the startup summary matches expectations:
   - `account=paper`
   - `strategy=mean_reversion`
   - `timeframe=15m`
   - `symbols=15`

3. During market hours, inspect:
   - `logs/<date>/bars.jsonl`
   - `logs/<date>/risk.jsonl`
   - `logs/<date>/signals.jsonl`

4. If you need monitoring without starting the bot:

```powershell
.\start_dashboard.ps1 -DashboardOnly
```

## What Not To Do

- Do not run multiple `tradeos live` processes.
- Do not use the dashboard to execute trading cycles.
- Do not change `.env` and `config/live_config.json` independently without checking the startup summary.
- Do not treat historical logs in `logs/` as safe to commit by default.

## First-Cycle Checks

On the first 1-2 bars after the market opens, confirm:

- `bars.jsonl` shows current completed bars, not stale historical ones
- `risk.jsonl` shows one `cycle.summary` per `decision_ts`
- `signals.jsonl` reflects the expected symbol universe and strategy
- there are no duplicate execution attempts for the same bar
- there are no stale-data failures unless the feed is genuinely delayed

## Current Phase

The project is currently in a late validation and paper-trading hardening phase.

What that means in practice:

- the live strategy has already been selected and frozen in `config/live_config.json`
- the bot has an operator workflow, dashboard, CLI entry point, and automated tests
- the main work now is validating day-of behavior and removing operational friction, not inventing a new baseline strategy

## Next Phase Checklist

Before promoting this bot beyond controlled paper-trading use, focus on these items:

1. Run pre-session preview consistently.
   - Start each session with `tradeos preview`
   - Confirm startup summary, symbol count, and strategy mode before any live paper run

2. Validate multi-day operational behavior.
   - Confirm one decision per completed bar
   - Confirm no duplicate entries or exits
   - Confirm forced end-of-day flatten always triggers when needed
   - Confirm kill-switch behavior is visible and understandable in logs and dashboard

3. Keep research artifacts aligned with the active strategy.
   - Refresh `results/strategy_status.md`, `results/best_config_latest.json`, and related reports after meaningful research updates
   - Retire or clearly label stale comparison outputs that predate the current mean-reversion promotion

4. Tighten repo-level test and startup confidence.
   - Keep `python -m pytest -q` green from the repo root
   - Keep `python -m tradeos --help` green as a packaging smoke test
   - Run targeted checks for `trading_bot`, `dashboard`, and `cli` when changing runtime behavior

5. Prepare for a production-readiness gate.
   - Review secret handling, runtime logs, and config override visibility
   - Confirm startup lock behavior and dashboard-only monitoring workflow remain intact
   - Decide what evidence is required before any real-money deployment discussion

## Pre-Move Test Set

Run this set before moving the project forward after meaningful runtime or dashboard changes:

```powershell
python -m pytest -q
python -m pytest -q tests/test_trading_bot.py tests/test_dashboard.py tests/test_cli.py
python -m tradeos --help
```

For a runtime smoke test before market hours:

```powershell
tradeos preview
```

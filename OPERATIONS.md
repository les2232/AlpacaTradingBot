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
- `config/paper_shadow_config.json` is the machine-specific named profile for supervised paper-shadow sessions. Keep it untracked.

## Supported Interfaces

- The supported operational interface is the `tradeos` CLI.
- The supported monitoring UI is the Streamlit dashboard.
- The older desktop UI layer has been intentionally removed and should not be recreated as part of routine operational work.

## One Operator Workflow

Use one monitoring process. Start a shadow process only for a manually supervised no-submit session.

1. Start the read-only dashboard:

```powershell
.\start_dashboard.ps1
```

This is the default launcher behavior. It starts only the dashboard and opens
the browser when the dashboard is ready; it does not start a trading or shadow
process. It refuses an already-occupied dashboard port because the existing
process's workspace and database authority cannot be verified.

2. Only for an explicitly supervised no-submit observation session, start the
paper shadow and dashboard together:

```powershell
.\start_dashboard.ps1 -PaperShadow
```

This starts `tradeos paper-shadow` with broker submission denied. Keep the
session manually supervised and stop it when observation is complete. Never
substitute `tradeos live` or `tradeos paper`; both are order-capable and outside
this workflow.

3. If paper shadow was started, confirm the startup summary matches expectations:
   - `account=paper`
   - `strategy=mean_reversion`
   - `timeframe=15m`
   - `symbols=15`

4. During the supervised shadow session, inspect:
   - `logs/<date>/bars.jsonl`
   - `logs/<date>/risk.jsonl`
   - `logs/<date>/signals.jsonl`

5. The explicit dashboard-only form is:

```powershell
.\start_dashboard.ps1 -DashboardOnly
```

## What Not To Do

- Do not run multiple `tradeos live` processes.
- Do not use `tradeos live` or `tradeos paper` for a no-submit session; both are order-capable.
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

## Durable Decision-Cycle Authority

Normal paper execution uses one canonical, timezone-aware UTC decision timestamp
as the identity for the entire configured multi-symbol cycle. Before normal
strategy evaluation, the bot atomically claims that identity in the configured
SQLite history database and records a `claimed` then `running` lifecycle. A
normal return becomes `completed`; an exception before an order-capable boundary
becomes `failed`; an exception after that boundary becomes `ambiguous`.

Every existing identity is fail-closed across restarts, regardless of lifecycle
status. An unseen timestamp older than the database's durable high-water mark is
also rejected. Failed, ambiguous, claimed, and running cycles are never
automatically released or retried. A hard process failure can therefore leave a
cycle in `claimed` or `running`; operators must treat that outcome as uncertain.

This mechanism provides one local cycle authority when all launchers use the
same SQLite database. It does not prove broker-level exactly-once order
execution: a process can fail after the broker accepts an order but before the
local response or lifecycle update is recorded. Resolving that window requires
separate broker reconciliation or broker-supported idempotency work.

`BOT_DB_PATH` defaults to the absolute, normalized `bot_history.db` at the
TradeOS repository root. Relative values are also anchored to that repository
root, regardless of the launcher's working directory. Relative paths that
escape the repository through `..` and blank values fail closed; use an
explicitly absolute path when an external database authority is intentional.
The live and preview CLI report the resolved authority; a live launch persists
that exact absolute path in startup metadata, and the dashboard displays the
exact path it opened. The dashboard refuses to initialize storage when an
explicit `BOT_DB_PATH` conflicts with the latest startup metadata.

Path precedence is an inherited `BOT_DB_PATH`, then the `BOT_DB_PATH` key from
the repository-root `.env`, then the repository-root default. The dashboard
prefers the latest startup artifact, but the explicit process/repository value
must match it. Dashboard code uses only `BOT_DB_PATH` from parsed repository
dotenv values and does not export other dotenv values into the process
environment; the normal inherited process environment remains unchanged.

Dashboard startup-artifact discovery is also launcher-invariant:
`BOT_LOG_ROOT` defaults to the repository `logs` directory, and relative values
are repository-anchored with the same blank/escape safeguards.

Forced EOD and kill-switch exits in an accepted cycle remain part of that
cycle's lifecycle and make a later failure ambiguous. Separately, a rejected
normal cycle cannot suppress a forced safety exit: it uses a no-signal snapshot
and can reach the fallback only after the existing startup resync, market-clock,
and regular-hours exit gates pass. A fallback exit does not release or mutate
the rejected cycle. Forced exits rely on broker-observed positions and open
orders and therefore do not provide broker-level idempotency across an uncertain
broker response.

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

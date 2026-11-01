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

## Ask TradeOS

The **Ask TradeOS** tab in the dashboard is a read-only analyst for historical sessions. It reads
`logs/<date>/*.jsonl`, the session's own `startup_config.*.json`, and `bot_history.db` (opened read-only).
It never submits orders, changes trading rules or risk limits, repairs state, or promotes research
configurations.

### Configure

1. Optional: add `OPENAI_API_KEY=...` to the repository `.env`. `TRADEOS_ASK_MODEL` defaults to
   `gpt-6.1-sol`, and `TRADEOS_ASK_TIMEOUT_SECONDS` defaults to `60`. A process environment variable
   overrides the `.env` value. Only these three keys are read from `.env`, and they are not exported.
2. Start the dashboard as usual: `.\start_dashboard.ps1` (or `tradeos dashboard`).
3. Open the **Ask TradeOS** tab, choose a session and question, then press **Ask TradeOS**. Without a key,
   the button is **Show calculated diagnostics**.

### How to read the answer

- **Observation** findings are directly shown by the cited records. **Hypothesis** findings are plausible
  but unproven explanations. Every finding lists evidence IDs. *Supporting evidence* shows each cited item
  and its source, such as `logs/2026-10-15/signals.jsonl:L57` or a startup artifact.
- **Bound sessions** have a `session_id`. Only records carrying that id are used. Records on the same day
  without an id are counted and excluded. **Unbound log days** (older logs without ids) are offered
  separately. They describe the whole day and have no known configuration.
- A historical session's configuration comes only from its own startup artifact. The current
  `config/live_config.json` is never applied to an old session.
- **Comparison unavailable** appears unless every check passes:
  - the session is bound and has a valid startup artifact with a canonical strategy hash;
  - the environment is identified;
  - at least 50% of the expected decisions were observed;
  - the baseline was valid at launch and passes validation now;
  - the baseline's bundle and generation hashes match the ones recorded at launch;
  - the pointer-bound drift profile is valid.

  When unavailable, the tab lists the reasons and suppresses underperformance, drift, and improvement
  conclusions. Observed metrics are still reported. The `daily_report.DEFAULT_BACKTEST` fallback numbers
  are never used as a comparison target.
- Win rate is marked not interpretable below 5 closed trades. Partial sessions (under 90% of the
  09:45-15:45 ET decisions) are flagged.

### What is sent to the model

Only the evidence package for the selected question goes to OpenAI's Responses API, with `store=false`.
It contains allowlisted event fields, calculated metrics, configuration values, and limitations. Order
ids, account fingerprints, balances, file system paths, and credentials are excluded, and the request is
refused if a credential-like value appears. The *Exact model request body* expander shows the payload.

### Tablet access (Android)

Streamlit is a web app, so the tablet only needs a browser:

1. Start the dashboard on the Windows PC as usual.
2. Find the PC's LAN address with `ipconfig` (IPv4 address, for example `192.168.1.20`).
3. Allow inbound TCP 8501 for **Private** networks only, in an elevated PowerShell:
   `New-NetFirewallRule -DisplayName "TradeOS dashboard (LAN)" -Direction Inbound -Protocol TCP -LocalPort 8501 -Profile Private -Action Allow`
4. On the tablet (same Wi-Fi), open `http://192.168.1.20:8501` in Chrome. Use **Add to Home screen** for
   an app icon.

`.streamlit/config.toml` does not set `server.address`, so Streamlit listens on all interfaces, and the
firewall rule is what grants LAN access. The dashboard has no login: anyone on that network can view it
and use Ask TradeOS, which spends OpenAI credit when a key is configured. Keep the Windows network profile
**Private** and the rule scoped to Private. For access away from home, use a private VPN such as Tailscale
rather than opening a router port. Remove the rule with
`Remove-NetFirewallRule -DisplayName "TradeOS dashboard (LAN)"`.

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
python -m pytest -q tests/test_ask_tradeos.py
python -m tradeos --help
```

For a runtime smoke test before market hours:

```powershell
tradeos preview
```

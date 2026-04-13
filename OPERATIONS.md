# Operations

## Source Of Truth

- `.env` is for secrets, `ALPACA_PAPER`, and account/risk settings.
- `config/live_config.json` is the source of truth for active runtime trading settings:
  - symbols
  - strategy
  - timeframe
  - indicator/filter settings
- If `config/live_config.json` differs from `.env`, the bot will override `.env` for supported runtime fields and print which fields were overridden at startup.

## One Operator Workflow

Use exactly one execution process and one monitoring process.

1. Start preview before the session:

```powershell
alpaca-bot preview
```

2. Confirm the startup summary matches expectations:
   - `account=paper`
   - `strategy=mean_reversion`
   - `timeframe=15m`
   - `symbols=15`

3. During market hours, inspect:
   - `logs/<date>/bars.jsonl`
   - `logs/<date>/risk.jsonl`
   - `logs/<date>/signals.jsonl`

4. If preview looks clean, stop it and start the real paper session:

```powershell
alpaca-bot live
```

5. In a separate terminal, launch monitoring only:

```powershell
alpaca-bot dashboard
```

## What Not To Do

- Do not run multiple `alpaca-bot live` processes.
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

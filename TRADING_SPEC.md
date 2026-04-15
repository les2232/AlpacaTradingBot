# Trading Specification

This document freezes the default trading behavior for the bot. The symbol universe and bar frequency remain configurable, but the bot should operate within the constraints below unless this spec is revised.

## Scope

- Venue: Alpaca paper trading by default.
- Asset class: US listed equities only.
- Strategy style: intraday long-only signal trading.
- Execution style: market orders only.

## Configurable Inputs

These settings may be changed without changing the spec itself.

- Universe: `BOT_SYMBOLS`
- Bar frequency: `BAR_TIMEFRAME_MINUTES`
- SMA lookback: `SMA_BARS`
- Strategy mode: `STRATEGY_MODE`
- Max trade budget: `MAX_USD_PER_TRADE`
- Max simultaneous positions: `MAX_OPEN_POSITIONS`
- Max daily realized plus unrealized loss: `MAX_DAILY_LOSS_USD`
- ML thresholds

## Frozen Defaults

These are the code-level fallback defaults when no override is present.
The currently active live strategy is selected via `config/live_config.json`, not from these defaults.
See `config/live_config.json` for the runtime source of truth.

- Default universe: `AAPL`, `MSFT`, `NVDA`
- Default bar frequency: `15` minutes
- Default SMA lookback: `20` bars
- Historical code-default strategy mode: `hybrid`
  - **Current live selection: `mean_reversion`** (set in `config/live_config.json`)
- Default max trade budget: `$200`
- Default max simultaneous positions: `3`
- Default max daily loss: `$300`
- Default account mode: paper

## Decision Cadence

- The bot evaluates once per decision interval.
- The default decision interval is one completed bar.
- With the default configuration, this means one decision pass every `15` minutes.
- The decision timestamp is the UTC close time of the most recent completed bar at evaluation start, floored to the configured bar interval.
- Example: if evaluation begins at `14:07:12` UTC on a `15` minute interval, the decision timestamp is `14:00:00` UTC.
- The bot must not submit more than one net new entry order per symbol per decision interval.
- If a symbol has an open order in flight, the bot must skip that symbol until the order is resolved.

## Bar Closure And Time Alignment

- Features, labels, SMA values, and the decision price must use completed bars only.
- The current in-progress bar must never be used for signal generation or model labels.
- For each symbol, the decision price is the close of the latest completed bar at the decision timestamp.
- Training labels must be formed only from completed-bar close to next completed-bar close movement.
- Historical snapshots should record the decision timestamp, not wall-clock capture time, as the canonical evaluation time.

## Trading Session Constraints

- Trade only during regular market hours.
- Default eligible decision window: `09:45` to `15:45` America/New_York.
- No new entries after `15:45` America/New_York.
- Forced flatten deadline: `15:55` America/New_York.
- All positions should be flat by the end of the regular session.

## Position Constraints

- Direction: long only.
- Position sizing: integer shares only.
- Max gross exposure at entry: `min(MAX_USD_PER_TRADE, available buying power)`.
- One open position per symbol at a time.
- No averaging down or pyramiding.
- No overnight holdings.

## Holding Time Constraints

- Default max holding time: same trading day only.
- A position opened intraday must be closed no later than the forced flatten deadline.
- If a stricter holding limit is later introduced, it must be less than or equal to one trading day.

## Entry Rules

- `sma` mode:
  - Buy when price is above SMA and no position is open.
- `ml` mode:
  - Buy when `probability_up >= ML_PROBABILITY_BUY` and no position is open.
- `hybrid` mode:
  - Buy only when both the SMA trend is bullish and the ML probability meets the buy threshold.
- `mean_reversion` mode (**current live mode**):
  - Buy when price has pulled back below SMA by at least `entry_threshold_pct` and ATR percentile is within `mean_reversion_max_atr_percentile`. Optional trend filter via `mean_reversion_trend_filter`.

## Exit Rules

- `sma` mode:
  - Exit when price is below SMA.
- `ml` mode:
  - Exit when `probability_up <= ML_PROBABILITY_SELL`.
- `hybrid` mode:
  - Exit when either the SMA trend turns bearish or the ML probability falls to or below the sell threshold.
- `mean_reversion` mode (**current live mode**):
  - Exit style is determined by `mean_reversion_exit_style` (currently `sma`: exit when price recrosses above SMA).
- Risk exit:
  - Stop all new trading when daily PnL is less than or equal to `-MAX_DAILY_LOSS_USD`.
- Session exit:
  - Close all open positions before the forced flatten deadline regardless of signal state.

## Risk Controls

- Kill switch: active when daily PnL breaches the configured loss limit.
- Flatten on kill: if execution is enabled and the kill switch is active, the bot must attempt to liquidate open positions immediately.
- Do not place a buy order if one share cannot be funded.
- Do not exceed the configured max per-symbol exposure.
- Do not exceed the configured max order rate.
- Do not trade when live execution price data is stale.
- Do not trade when the live execution price breaches the configured collar versus the decision price.
- Do not trade when the latest completed bar is older than the configured data-delay limit.
- Do not exceed `MAX_OPEN_POSITIONS`.
- Do not place duplicate entry or exit orders for a symbol with an open order already pending.

## Data And State Requirements

- Every decision cycle must persist:
  - account snapshot
  - symbol price and SMA
  - ML probability and confidence
  - action taken or proposed
  - recent order states
- Historical records must be sufficient to reconstruct why the bot was long, flat, or blocked.

## Current Implementation Status

- Enforced now:
  - configurable universe and bar frequency
  - centralized decision logic
  - max trade budget
  - max per-symbol exposure
  - max open positions
  - daily loss kill switch
  - flatten on kill
  - max order rate
  - price collars
  - no trade on stale data
  - skip symbols with open orders in flight
  - persistence of ML fields to SQLite
  - persistence of holding duration snapshots to SQLite
  - regular-hours-only trading window (`09:45-15:45` ET)
  - Alpaca market-clock check before live execution
  - one-decision-per-completed-bar scheduler
  - forced end-of-day flatten window at `15:55` ET
  - duplicated end-of-day flatten protection in the live bot:
    - inside the main decision path as an execution guard
    - inside a dedicated background thread as a wall-clock fail-safe
- Not yet enforced:
  - a stricter holding-time exit earlier than the end-of-day flatten deadline

## Research Workflow Notes

- [dataset_snapshotter.py](dataset_snapshotter.py) creates versioned offline datasets under `datasets/`.
- [backtest_runner.py](backtest_runner.py) replays those datasets through the shared strategy layer and writes result CSVs under `results/`.
- [run_research.py](run_research.py) is the configurable research pipeline:
  - it snapshots fresh data
  - runs sweeps
  - ranks results
  - writes decision artifacts such as `best_config_latest.json`, `stability_report.json`, and `trade_decision.json`
- [run_compare_suite.ps1](run_compare_suite.ps1) is a fixed benchmarking wrapper:
  - it uses predefined datasets
  - compares `sma`, `ml`, and `hybrid`
  - writes all outputs into one timestamped folder under `results/`

## Revision Rule

Any change to session window, flattening behavior, holding time, position direction, or execution style should be treated as a spec revision, not a routine parameter change.

# Bot Reevaluation

_Generated: 2026-04-23 20:47 UTC_

## Verdict

- `NO-GO`
- promoted research config does not match the current live runtime
- live-effective edge diagnostics are negative after costs across all measured short horizons
- recent one-week forward validation is too close to break-even to trust with slippage and regime drift

## Current Mean-Reversion Evidence

- Promoted research window return: `5.46%` with PF `1.189`
- Recent week (`2026-04-14` to `2026-04-21`) return: `0.46%` with PF `1.041`
- Live-effective horizon `1` bars: net expectancy `-0.055%`, PF `0.688`
- Live-effective horizon `2` bars: net expectancy `-0.045%`, PF `0.800`
- Live-effective horizon `4` bars: net expectancy `-0.057%`, PF `0.817`

## Runtime Trust Check

- `mean_reversion_trend_filter` live-only=True (missing from promoted research config)
- `mean_reversion_trend_slope_filter` live-only=True (missing from promoted research config)
- `mean_reversion_stop_pct` live-only=0.01 (missing from promoted research config)

## Symbol Pressure

- Worst 1-bar live-effective net expectancy names:
  - BAC: `-0.182%`
  - WFC: `-0.140%`
  - AMD: `-0.116%`
  - TSLA: `-0.081%`
  - COP: `-0.079%`
- Best 1-bar live-effective net expectancy names:
  - HD: `0.008%`
  - C: `-0.003%`
  - GOOGL: `-0.008%`
  - LLY: `-0.014%`
  - JPM: `-0.015%`

## Candidate Replacement Check

- `trend_pullback_oos_validation` classification: `research-only`
- `trend_pullback_oos_validation` reason: The frozen baseline is inconsistent across OOS windows and remains too fragile to trust.
- Recent `trend_pullback` live mismatch comparison: expectancy `-0.852`, PF `0.766`, realized PnL `-63.93`
- Conclusion: the newer candidate lane is more interesting than the current mean-reversion lane, but it is still research-only rather than production-ready.

## Recommended Next Step

- Pause or heavily throttle the current live mean-reversion bot.
- Refresh research artifacts using the exact live runtime fields now present in `config/live_config.json`.
- Re-test with symbol pruning and stricter production promotion criteria before re-enabling normal size.


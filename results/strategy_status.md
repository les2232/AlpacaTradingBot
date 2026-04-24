# Strategy Status

_Last generated: 2026-04-23 20:47 UTC_

## Current Live Strategy

| Field | Value |
|---|---|
| Mode | `mean_reversion` |
| Symbols | 15 (ABBV, AMD, BAC, C, COP, GOOGL, HD, HON, JPM, LLY, MSFT, QCOM, TSLA, WFC, XOM) |
| Bar timeframe | 15 minutes |
| SMA bars | `15` |
| Entry threshold | 0.15% pullback below SMA |
| Mean-reversion exit | `sma` |
| Mean-reversion ATR cap | 80.0 |
| Trend filter | enabled |
| Trend slope filter | enabled |
| Mean-reversion stop | 1.00% |
| Regime filter | disabled |
| Config source | `config/live_config.json` |

## Promoted Research Snapshot

- `results/best_config_latest.json` approval: `True`
- Profit factor: `1.189`
- Win rate: `68.8%`
- Trades/day: `29.93`
- Max drawdown: `3.88%`

## Config Alignment

- Status: `MISMATCHED`
- `mean_reversion_trend_filter` live-only=True (missing from promoted research config)
- `mean_reversion_trend_slope_filter` live-only=True (missing from promoted research config)
- `mean_reversion_stop_pct` live-only=0.01 (missing from promoted research config)

## Caveats

- This file is generated from `config/live_config.json` and `results/best_config_latest.json`.
- If this status disagrees with recent diagnostics, trust the diagnostics and reevaluate before trading live capital.


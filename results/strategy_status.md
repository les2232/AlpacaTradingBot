# Strategy Status

_Last updated: 2026-04-16_

## Current Live Strategy

| Field | Value |
|---|---|
| Mode | `mean_reversion` |
| Symbols | 15 (AMD TSLA GOOGL MSFT QCOM BAC C WFC JPM ABBV LLY HON XOM COP HD) |
| Bar timeframe | 15 minutes |
| SMA bars | 15 |
| Entry threshold | 0.15% pullback below SMA |
| ATR percentile cap | 80 |
| Exit style | SMA recross |
| Trend filter | disabled |
| Config source | `config/live_config.json` (runtime source of truth) |

## Supporting Research

- `config/live_config.json` - runtime source of truth; now aligned with the promoted mean-reversion settings.
- `results/best_config_latest.json` - canonical research-pipeline format; mean-reversion config with profit factor 1.189, win rate 68.8%, and trades/day 29.93.
- Research CSVs with `15sym` or `mean_reversion` in the filename under `results/` reflect the supporting sweep results.

## Known Caveats

- OOS window is Apr-Oct 2025; regime conditions may differ from current market.
- Trend filter is disabled (`mean_reversion_trend_filter=false`); this was the validated setting but means entries are not regime-gated.
- ATR percentile filter caps volatility exposure but does not eliminate it.
- `hybrid` mode remains available in `strategy.py` and can be reactivated via `config/live_config.json`.

## Stale Artifacts (Reference Only)

- Older `results/compare_suite_*/` folders compare `sma`, `ml`, and `hybrid` and predate the mean-reversion validation.
- `results/stability_report.json` and `results/trade_decision.json` reflect the 2026-04-06 SMA smoketest run; they predate the mean-reversion promotion and will be refreshed by the next `run_research.py` cycle.

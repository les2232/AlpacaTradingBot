# Execution Checklist

_Generated: 2026-04-23 20:47 UTC_

## Immediate

- Pause or heavily throttle the current `mean_reversion` live lane.
- Do not allow live startup when `config/live_config.json` and `results/best_config_latest.json` fail baseline validation.
- Use `results/bot_reevaluation.md` as the current go/no-go summary.
- Treat the current loser cluster as first-pass prune candidates: `BAC`, `WFC`, `AMD`, `TSLA`, `COP`.

## This Week

- Revalidate the exact live runtime fields now present in `config/live_config.json`.
- Re-run recent-window evaluation with costs included and reject near-break-even configs.
- Test symbol pruning before larger rule changes.
- Test exit changes separately from entry changes.
- Keep replacement candidates in research-only status unless they hold up across recent windows.

## Only If Needed

- Consider a broader strategy overhaul only if cleaned-up config parity still leaves backtest/live divergence.
- Consider an architecture overhaul only if multiple strategy families remain fragile after the current controls cleanup.


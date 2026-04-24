# Ranking-First Validation Note

## Executive Summary

This pass does not support the idea that simple ranking-only entries are the main edge. On the current walk-forward setup, `ranking_only` finished behind both `pullback` and `baseline` on stitched out-of-sample results, and it was negative overall.

## Code Path Summary

- Ranking is computed in `prepare_panel_features(...)` and ordered cross-sectionally by descending `score_return`.
- Eligible longs are defined in `assign_cross_sectional_ranks(...)` as the top `ceil(universe_size * eligible_percent)` symbols at each timestamp.
- Existing score modes in the canonical lane are:
  - `return_20`
  - `return_20_plus_60`
  - `trend_consistency_20`
  - `trend_consistency_20_x_slope_20`
  - `trend_consistency_20_x_slope_20_over_atr_20`
- `baseline` uses ranked eligibility plus the existing uptrend gate.
- `pullback` adds pullback / reclaim timing on top of the baseline eligibility.
- `ranking_only` uses ranked eligibility only, with next-bar entry and fixed-hold exit.
- Hold periods are applied mechanically through the existing fixed-bar exit logic.
- Walk-forward selection and stitched OOS reporting are handled by the canonical validation runner.

## Family Comparison

Stitched selected OOS results:

- `pullback`: expectancy `4.5635`, profit factor `1.7004`, trade count `50`, total pnl `228.18`, max drawdown `2.0272%`
- `baseline`: expectancy `0.9110`, profit factor `1.0987`, trade count `60`, total pnl `54.66`, max drawdown `4.3324%`
- `ranking_only`: expectancy `-1.1558`, profit factor `0.8961`, trade count `59`, total pnl `-68.19`, max drawdown `4.7717%`

Fold winners:

- Fold 1: `pullback`
- Fold 2: `ranking_only`
- Fold 3: `pullback`

Family win counts:

- `pullback`: `2`
- `ranking_only`: `1`

## Hold Horizon Read

Best stitched variants by family:

- `pullback`: `top30 hold2 pb0.5`, avg expectancy `3.9329`, total pnl `243.79`
- `baseline`: `top30 hold5`, avg expectancy `4.8922`, total pnl `293.53`
- `ranking_only`: `top30 hold10`, avg expectancy `1.8851`, total pnl `113.11`

Within `ranking_only`, the 10-bar holds were the only clearly positive horizon. The 2-bar and 5-bar variants were negative overall.

## Symbol Concentration

`ranking_only` winners were concentrated in a small set:

- `AMD`: `+86.72`
- `GOOGL`: `+36.17`
- `ABBV`: `+23.54`
- `LLY`: `+17.48`

Largest `ranking_only` drags:

- `XOM`: `-127.62`
- `COP`: `-41.87`
- `HON`: `-24.18`

That concentration is materially worse than the pullback family, which still had losers but produced broader positive contribution from `C`, `GOOGL`, `BAC`, and `WFC`.

## Judgment

The current evidence does not show that selection alone is the main edge. Ranking helps, but the best results here still require some form of timing / gating, and the pure ranking-only path looks too fragile to promote.

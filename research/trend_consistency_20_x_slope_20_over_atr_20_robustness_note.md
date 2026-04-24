## Executive Summary

`trend_consistency_20_x_slope_20_over_atr_20` looks better than the earlier failed signals, but the robustness read is mixed rather than clean. Across 3 walk-forward folds, the pullback implementation stays positive overall and materially beats baseline, yet there is still one clear failure fold and most of the strength comes from the later windows.

## Multi-Fold Results

- Stitched selected OOS baseline: `57` trades, expectancy `-0.1138`, profit factor `0.9897`, max drawdown `6.6838%`, total PnL `-6.48`
- Stitched selected OOS pullback: `50` trades, expectancy `4.5635`, profit factor `1.7004`, max drawdown `2.0272%`, total PnL `228.18`
- Fold 1:
  - baseline expectancy `-17.0469`, profit factor `0.1645`, drawdown `6.6838%`
  - pullback expectancy `-6.9728`, profit factor `0.3415`, drawdown `1.7385%`
- Fold 2:
  - baseline expectancy `8.8768`, profit factor `4.9005`, drawdown `0.4703%`
  - pullback expectancy `7.2158`, profit factor `3.8635`, drawdown `0.5015%`
- Fold 3:
  - baseline expectancy `7.7706`, profit factor `1.7193`, drawdown `2.4568%`
  - pullback expectancy `11.6722`, profit factor `2.6928`, drawdown `1.4323%`

## Time-Slice Stability

- March 2026:
  - baseline expectancy `-3.7527`, profit factor `0.6621`, drawdown `6.6838%`
  - pullback expectancy `0.5649`, profit factor `1.0896`, drawdown `1.9920%`
- April 2026:
  - baseline expectancy `7.7706`, profit factor `1.7193`, drawdown `2.4568%`
  - pullback expectancy `11.6722`, profit factor `2.6928`, drawdown `1.4323%`

The signal is not evenly strong through time. March is weak, especially for baseline, while April is clearly stronger for both families.

## Failure Cases

- Fold 1 is a real failure case for both baseline and pullback, not just a mild soft patch
- Baseline collapses outright in stitched multi-fold OOS, ending slightly negative overall
- Pullback helps materially, but its robustness still depends on the later folds offsetting the early failure
- Performance is not dominated by one single symbol, but the positive contribution set is still concentrated in a handful of names such as `C`, `GOOGL`, `BAC`, and `WFC`

## Overall Robustness Classification

somewhat robust

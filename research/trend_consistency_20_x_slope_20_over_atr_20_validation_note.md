## Executive Summary

`trend_consistency_20_x_slope_20_over_atr_20` is the first canonical ranking signal in this lane that held up well in stitched out-of-sample validation. Both baseline and pullback stayed clearly positive, and pullback improved quality metrics, but the evidence is still limited by having only one walk-forward fold.

## Stitched OOS Results

- Baseline selected OOS: `48` trades, expectancy `5.8342`, profit factor `1.8678`, max drawdown `2.6897%`, total PnL `280.04`
- Pullback selected OOS: `42` trades, expectancy `6.0149`, profit factor `1.9284`, max drawdown `1.9764%`, total PnL `252.62`
- Relative to prior weak signals, this is materially better because stitched OOS expectancy is positive for both families and drawdowns remain controlled.

## Baseline vs Pullback

- Pullback improved expectancy by `+0.1807`
- Pullback improved profit factor by `+0.0606`
- Pullback reduced max drawdown by `0.7133` percentage points
- Pullback gave up `6` trades versus baseline
- This is a real quality improvement, not just a trade-count collapse

## Fold / Symbol Robustness

- Current robustness evidence is limited because the run contains only `1` walk-forward fold
- Both fold winners were selected from negative in-sample expectancies, yet the stitched test fold came back strongly positive, which is encouraging but also unstable as a single-fold read
- Symbol concentration does not look extreme in the stitched summaries:
  - baseline top positive symbol share: `0.2924`
  - pullback top positive symbol share: `0.2797`
- Per-symbol results are still uneven, with strong contributions from a handful of names such as `C`, `BAC`, `WFC`, and `GOOGL`, while names like `AMD`, `QCOM`, `XOM`, and `LLY` remain weak in at least one family

## Final Classification

promising enough to continue

## Recommended Next Step

Keep pursuing this signal through the existing research lane, but treat it as an early positive read rather than a settled result because the current validation still rests on a single walk-forward fold.

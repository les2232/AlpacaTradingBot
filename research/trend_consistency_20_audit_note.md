## Executive Summary

`trend_consistency_20` is better than the earlier return-based scores, but it is still not strong enough to call promising. The signal only shows clean leadership at 10- and 20-bar horizons, while 5-bar ordering remains broken and the spreads are still small.

## Key Audit Results

- 5 bars: top `+0.0001%`, middle `-0.0069%`, bottom `+0.0011%`, top-minus-bottom `-0.0009%`
- 10 bars: top `+0.0132%`, middle `-0.0231%`, bottom `-0.0081%`, top-minus-bottom `+0.0213%`
- 20 bars: top `+0.0422%`, middle `-0.0714%`, bottom `-0.0221%`, top-minus-bottom `+0.0642%`

## Classification

weak

## Decision

try one more ranking refinement

## Why

The main improvement is that top now beats both middle and bottom at 10 and 20 bars. That said, the 5-bar horizon is still misordered because bottom slightly beats top, and the positive spreads are modest enough that overfitting risk is still high. This is not a credible promotion candidate yet, but it is also not as clearly broken as `return_20_plus_60`.

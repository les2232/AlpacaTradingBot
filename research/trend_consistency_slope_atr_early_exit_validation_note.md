## Executive Summary

The `early_no_follow_through_exit` change was harmful. It did not make the strategy more robust. Instead, it turned the previously positive pullback result into a negative one, increased trade count sharply, and worsened drawdown.

## Impact on Expectancy / PF

Reference pullback without early exit:

- stitched OOS expectancy: `4.5635`
- stitched OOS profit factor: `1.7004`
- trade count: `50`

Pullback with early exit:

- stitched OOS expectancy: `-2.0160`
- stitched OOS profit factor: `0.6092`
- trade count: `146`

Change versus prior pullback:

- expectancy worsened by about `-6.58`
- profit factor worsened by about `-1.09`
- trade count increased by `96`

This is not a small degradation. It breaks the edge.

## Impact on Drawdown

Reference pullback without early exit:

- max drawdown: `2.0272%`

Pullback with early exit:

- max drawdown: `8.6920%`

Change:

- drawdown worsened by about `+6.66` percentage points

This is the opposite of the intended effect.

## Fold-Level Comparison

Prior pullback without early exit:

- Fold 1: expectancy `-6.9728`, drawdown `1.7385%`, trade count `15`
- Fold 2: expectancy `7.2158`, drawdown `0.5015%`, trade count `17`
- Fold 3: expectancy `11.6722`, drawdown `1.4323%`, trade count `18`

Pullback with early exit:

- Fold 1: expectancy `-3.2699`, drawdown `2.8039%`, trade count `45`
- Fold 2: expectancy `-2.5116`, drawdown `3.7559%`, trade count `59`
- Fold 3: expectancy `0.0239`, drawdown `3.4311%`, trade count `42`

Interpretation:

- Fold 1 loss was reduced in expectancy terms, but drawdown still got worse and trade count tripled.
- Folds 2 and 3, which had previously been clearly positive, were damaged badly.
- The strategy became much less consistent across folds.

## Trade Count Impact

Trade count rose from `50` to `146`.

That suggests the early exit is not just cutting bad trades. It is also creating many more opportunities to re-enter, which likely recycled the strategy through repeated low-quality trades. This increased churn appears to be the main reason the modification became harmful.

## Final Verdict

harmful

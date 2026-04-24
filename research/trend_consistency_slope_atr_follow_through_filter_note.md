## Executive Summary

The `use_recent_follow_through_filter` change was harmful. It did not improve the weak regime, and it damaged the previously stronger folds. The pullback strategy went from a positive stitched OOS result without the filter to a negative stitched OOS result with the filter.

## Impact on Expectancy / PF

Reference pullback without filter:

- expectancy `4.5635`
- profit factor `1.7004`
- trade count `50`

Pullback with recent follow-through filter:

- expectancy `-2.3414`
- profit factor `0.7397`
- trade count `40`

Change:

- expectancy worsened by about `-6.90`
- profit factor worsened by about `-0.96`
- trade count fell by `10`

The filter reduced trading somewhat, but not in a way that preserved edge.

## Impact on Drawdown

Reference pullback without filter:

- max drawdown `2.0272%`

Pullback with filter:

- max drawdown `3.5254%`

Change:

- drawdown worsened by about `+1.50` percentage points

So the filter did not improve downside behavior overall.

## Fold-Level Comparison

Reference pullback without filter:

- Fold 1: expectancy `-6.9728`, drawdown `1.7385%`, trade count `15`
- Fold 2: expectancy `7.2158`, drawdown `0.5015%`, trade count `17`
- Fold 3: expectancy `11.6722`, drawdown `1.4323%`, trade count `18`

Pullback with filter:

- Fold 1: expectancy `-13.4116`, drawdown `2.6767%`, trade count `11`
- Fold 2: expectancy `-2.0279`, drawdown `0.8031%`, trade count `14`
- Fold 3: expectancy `5.4841`, drawdown `0.5684%`, trade count `15`

Interpretation:

- Fold 1 did not improve. It became worse in expectancy and worse in drawdown.
- Fold 2 was the biggest casualty. It flipped from clearly positive to negative.
- Fold 3 stayed positive, but it was materially weaker than before.

## Trade Count Impact

Trade count fell from `50` to `40`.

That is a moderate reduction, not a collapse. The problem is not that the filter removed too many trades. The problem is that the trades it removed were not the right ones, and the trades it allowed were still not strong enough to preserve the edge.

## Final Verdict

harmful

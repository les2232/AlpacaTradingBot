## 1. Executive Summary

Using the available results, `return_20_plus_60` did not improve signal separation. The only available `return_20` baseline folder was `results/canonical_cross_sectional_rank_audit_smoke/`, and against that baseline the new score was worse: top still failed to beat middle, top no longer beat bottom, and top-minus-bottom spread turned negative at every tested horizon. This is not good enough to justify moving forward to validation.

## 2. Baseline (return_20)

Baseline comparison used:

- `results/canonical_cross_sectional_rank_audit_smoke/bucket_summary.csv`

Key observations:

- 5-bar horizon:
  - top: `-0.0074%`
  - middle: `+0.0154%`
  - bottom: `-0.0137%`
  - ordering: `middle > top > bottom`
- 10-bar horizon:
  - top: `-0.0174%`
  - middle: `+0.0159%`
  - bottom: `-0.0174%`
  - ordering: `middle > top ≈ bottom`
- Top-minus-bottom spread:
  - 5 bars: `+0.0063%`
  - 10 bars: `+0.0000%`

Interpretation:

- The baseline was already weak because top did not beat middle.
- But it still had one limited positive property: top was at least slightly better than bottom.
- Even that edge was economically tiny.

## 3. New Score (return_20_plus_60)

New score folder:

- `results/canonical_cross_sectional_rank_audit_r20_r60/bucket_summary.csv`

Key observations:

- 5-bar horizon:
  - top: `-0.0132%`
  - middle: `-0.0043%`
  - bottom: `+0.0061%`
  - ordering: `bottom > middle > top`
- 10-bar horizon:
  - top: `-0.0336%`
  - middle: `-0.0018%`
  - bottom: `+0.0092%`
  - ordering: `bottom > middle > top`
- 20-bar horizon:
  - top: `-0.0873%`
  - middle: `+0.0271%`
  - bottom: `+0.0081%`
  - ordering: `middle > bottom > top`
- Top-minus-bottom spread:
  - 5 bars: `-0.0194%`
  - 10 bars: `-0.0429%`
  - 20 bars: `-0.0954%`

Interpretation:

- The new score did not fix the middle-bucket problem.
- It made the top bucket clearly worse than bottom.
- Spread was not just small; it flipped negative across all tested horizons.

## 4. Direct Comparison

### Ordering

- Baseline `return_20`:
  - weak ordering
  - top failed to beat middle
  - but top still slightly beat bottom
- New `return_20_plus_60`:
  - worse ordering
  - top lost to both middle and bottom
  - at 5 and 10 bars it was effectively inverted

### Spread

- Baseline spread:
  - 5 bars: `+0.0063%`
  - 10 bars: `+0.0000%`
- New score spread:
  - 5 bars: `-0.0194%`
  - 10 bars: `-0.0429%`
  - 20 bars: `-0.0954%`

The new score is materially worse on spread direction. Baseline was weak-but-positive versus bottom. New score is negative across the board.

### Middle-Bucket Problem

- Baseline:
  - middle still beat top
- New score:
  - middle still beat top
  - and bottom also beat top

So the original problem was not fixed. It became more severe.

### Horizon Behavior

- Baseline:
  - slight positive spread at 5 bars
  - effectively flat by 10 bars
- New score:
  - negative at 5, 10, and 20 bars

There is no horizon where `return_20_plus_60` improved the ordering.

### Magnitude

- Baseline magnitudes were already tiny and probably too small to matter.
- New-score magnitudes are larger in absolute value, but in the wrong direction.

This is not a case of “slight economic improvement.” It is deterioration.

## 5. Verdict

`worse than baseline`

## 6. Recommended Next Step

`try one more ranking refinement`

Do not proceed to validation with `return_20_plus_60`. It made the ranking signal worse, so the work should stay at the signal layer rather than moving forward.

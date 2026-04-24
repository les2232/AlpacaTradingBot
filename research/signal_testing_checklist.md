# Signal Testing Checklist

Use this checklist every time you test one new ranking signal.

## Phase 0 - Setup

- Confirm the canonical audit runner exists.
- Confirm the canonical validation runner exists.
- Confirm outputs are standardized and saved to a dedicated results folder.
- Confirm you are changing only one `score_mode` for this experiment.
- Write down the signal hypothesis before running anything.

## Phase 1 - Signal Audit

For each signal:

- Define the hypothesis in one sentence.
- Implement one `score_mode` only.
- Run the canonical cross-sectional audit.
- Open `bucket_summary.csv` first.
- Check whether `top > middle > bottom` holds.
- Check whether `top_minus_bottom > 0`.
- Check multiple horizons, not just one row.
- If needed, use `time_slice_summary.csv` only to confirm whether the result is broad or narrow.

Classify the signal:

- `promising`
  - top beats middle and bottom in a reasonably clean way
  - top-minus-bottom is positive across multiple horizons
- `weak`
  - some positive signal exists, but ordering is incomplete or magnitude is tiny
- `no signal`
  - ordering is mixed and spread is near zero
- `inverted`
  - bottom beats top or top-minus-bottom is negative

Decision after audit:

- `advance`
  - only if the audit is clearly good enough
- `refine`
  - if the signal is weak but not obviously inverted
- `abandon`
  - if the signal is inverted or repeatedly fails cleanly

## Phase 2 - Validation Gate

Only run this if Phase 1 passes.

- Run canonical validation.
- Compare baseline vs pullback.
- Inspect stitched OOS first.
- Check whether pullback improves quality without collapsing trade count.
- Decide whether the candidate is promotion-ready or still research-only.

## Anti-Overfitting Rules

- Change one signal at a time.
- Do not optimize execution on a weak signal.
- Do not widen the grid casually.
- Do not reinterpret weak results as promising.
- Do not use one lucky horizon or one lucky month as proof.
- Do not move to validation just because the signal is “interesting.”

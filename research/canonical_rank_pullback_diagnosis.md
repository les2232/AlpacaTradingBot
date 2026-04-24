## 1. Executive Summary

The current idea looks weak because the cross-sectional ranking does not produce a clean monotonic gradient. Top only marginally beats bottom, does not beat middle, and that weak ordering does not survive into tradable stitched OOS results. The primary failure mode is weak / diluted signal, not a pullback-entry implementation problem.

## 2. What the Audit Shows

Only the smoke audit folder was available: `results/canonical_cross_sectional_rank_audit_smoke/`.

From `bucket_summary.csv`:

- 5-bar forward return:
  - top: `-0.0074%`
  - middle: `+0.0154%`
  - bottom: `-0.0137%`
- 10-bar forward return:
  - top: `-0.0174%`
  - middle: `+0.0159%`
  - bottom: `-0.0174%`

This is not `top > middle > bottom`. It is effectively `middle > top ≈ bottom`.

From `top_minus_bottom`:

- 5 bars: `+0.0063%`
- 10 bars: `+0.0000%`

Those spreads are directionally positive against bottom, but the magnitude is trivial. At 10 bars the spread is basically zero.

From `time_slice_summary.csv`:

- January 2026:
  - 5 bars: top `+0.0034%`, middle `+0.0037%`, bottom `+0.0603%`, spread `-0.0569%`
  - 10 bars: top `+0.0048%`, middle `+0.0034%`, bottom `+0.1162%`, spread `-0.1114%`
- February 2026:
  - 5 bars: top `-0.0179%`, middle `-0.0274%`, bottom `-0.0391%`, spread `+0.0212%`
  - 10 bars: top `-0.0559%`, middle `-0.0207%`, bottom `-0.0891%`, spread `+0.0332%`
- March 2026:
  - 5 bars: top `-0.0204%`, middle `-0.0054%`, bottom `-0.0205%`, spread `+0.0001%`
  - 10 bars: top `-0.0232%`, middle `-0.0295%`, bottom `-0.0422%`, spread `+0.0190%`
- April 2026:
  - 5 bars: top `+0.0395%`, middle `+0.1954%`, bottom `-0.0396%`, spread `+0.0791%`
  - 10 bars: top `+0.0626%`, middle `+0.2579%`, bottom `+0.0355%`, spread `+0.0271%`

Interpretation:

- January is outright wrong-signed versus bottom.
- February and March have only small positive spread.
- April improves, but middle is still much stronger than top.

From `fold_spread_summary.csv`:

- The only available fold, `2026-03-13` to `2026-04-10`, shows:
  - 5 bars: `+0.1039%`
  - 10 bars: `+0.1317%`

That fold is better than the full-smoke aggregate, which suggests some recency concentration rather than a broad stable edge.

From `symbol_contribution.csv`:

- Top-bucket membership concentration is not extreme by count. Largest shares:
  - `COP` `9.80%`
  - `XOM` `8.91%`
  - `AMD` `8.41%`
- But top-bucket forward returns are mixed to poor:
  - strong: `GOOGL` `+0.0980%` / `+0.1636%`, `XOM` `+0.1025%` / `+0.1957%`
  - weak/negative: `HD` `-0.1301%` / `-0.2944%`, `QCOM` `-0.1332%` / `-0.3370%`, `TSLA` `-0.0560%` / `-0.1267%`

So the problem is not that one symbol is single-handedly creating a fake positive signal. The problem is that the top bucket itself is noisy and mixed.

## 3. What the Validation Shows

Only the smoke validation folder was available: `results/canonical_rank_pullback_validation_smoke/`.

From `stitched_selected_summary.csv`:

- Baseline stitched OOS:
  - trade count: `83`
  - expectancy: `-0.3207`
  - win rate: `46.99%`
  - profit factor: `0.9636`
  - total pnl: `-26.62`
  - max drawdown: `3.17%`
- Pullback stitched OOS:
  - trade count: `75`
  - expectancy: `-1.5214`
  - win rate: `48.00%`
  - profit factor: `0.8526`
  - total pnl: `-114.10`
  - max drawdown: `3.69%`

From `baseline_vs_pullback.csv`:

- pullback minus baseline expectancy: `-1.2007`
- pullback minus baseline profit factor: `-0.1110`
- trade count change: `-8`
- drawdown change: `+0.5204%`

So pullback did not rescue a good signal. It made a bad result worse while only modestly reducing trade count.

From `stitched_variant_summary.csv`:

- baseline variant `baseline_lb20_top20_hold5`:
  - avg expectancy: `-0.3207`
  - positive fold ratio: `0.0`
- pullback variant `pullback_lb20_top20_hold5_pb0.5`:
  - avg expectancy: `-1.5214`
  - positive fold ratio: `0.0`

From `fold_winners.csv`:

- baseline train selection expectancy: `-1.2183`
- pullback train selection expectancy: `-1.3660`

Even the in-sample fold winner for each family was negative. That is a bad sign. This is not a case where in-sample looked good and OOS collapsed. It already looked weak before OOS.

From `per_symbol_breakdown.csv`:

- Baseline positive contributors:
  - `GOOGL` `+97.96`
  - `C` `+59.60`
  - `BAC` `+46.66`
- Baseline negative contributors:
  - `HD` `-61.97`
  - `TSLA` `-56.89`
  - `LLY` `-49.47`
  - `QCOM` `-48.49`
  - `XOM` `-43.27`
- Pullback positive contributors:
  - `GOOGL` `+94.02`
  - `C` `+43.53`
  - `ABBV` `+39.01`
- Pullback negative contributors:
  - `TSLA` `-98.47`
  - `COP` `-88.86`
  - `QCOM` `-70.79`
  - `LLY` `-53.61`

This again points to a mixed and fragile signal, not a small execution mistake.

## 4. Primary Failure Mode

`2. Weak / diluted signal`

The rank signal is not producing a clean hierarchy. Top is not better than middle at either tested horizon, and the top-minus-bottom spread is too small to support a reliable tradable edge. Since the immediate-entry baseline is already negative, the deeper problem is the signal itself, not just the pullback entry timing.

## 5. Supporting Evidence

- The audit gradient is broken:
  - 5 bars: middle `+0.0154%` > top `-0.0074%` > bottom `-0.0137%`
  - 10 bars: middle `+0.0159%` > top `-0.0174%` ≈ bottom `-0.0174%`
- Top-minus-bottom spread is tiny:
  - `+0.0063%` at 5 bars
  - `~0.0000%` at 10 bars
- January is wrong-signed, and March is almost flat.
- April is the best-looking month, but middle still dominates top.
- The only available fold is positive on spread, which suggests local strength, not broad robustness.
- Baseline stitched OOS expectancy is already negative at `-0.3207`.
- Pullback stitched OOS expectancy is much worse at `-1.5214`.
- Pullback only cuts trade count from `83` to `75`, so this is not a “good filter ruined by over-pruning” story.
- Both family winners were already negative in train:
  - baseline `-1.2183`
  - pullback `-1.3660`

## 6. What This Means

The ranking score is not separating the universe well enough. It can sometimes distinguish top from bottom a little, but it is not identifying a genuinely stronger top cohort. That leaves the tradable strategy trying to monetize a very soft statistical difference, and once realistic entry/exit mechanics are applied, the edge disappears.

In plain English: the current score is too noisy to be a foundation. The pullback layer is not the main problem. It is just sitting on top of a weak underlying ranking signal.

## 7. Minimum Next Step

`refine ranking signal`

The baseline is already negative, so the next step should be to improve or replace the ranking signal before doing more execution work.

## 8. Do NOT Do Next

- Do not add more filters to try to rescue this.
- Do not optimize exits.
- Do not widen the parameter sweep.
- Do not tune per-symbol behavior.
- Do not reinterpret the single positive fold as confirmation.
- Do not promote anything from these results.

# Signal Experiment Log

Purpose:

- keep a durable record of what signal was tested
- record why it was tested
- record how it was tested
- capture the audit result and next decision

Use the template below for every new `score_mode`.

---

## Experiment: <signal_name>
Date:
Status: planned / running / completed / abandoned

### Hypothesis
What this signal is trying to capture.

### Signal Definition
Exact score formula.

### Files / Commands
- Audit command:
- Output folder:
- First file inspected:

### Audit Results
- Horizon:
  - Top:
  - Middle:
  - Bottom:
  - Top minus bottom:

### Classification
- promising / weak / no signal / inverted

### Decision
- advance to validation / refine signal / abandon

### Notes
Anything important learned.

---

## Experiment: return_20
Date: 2026-04-18
Status: completed

### Hypothesis
Recent 20-bar relative strength might separate stronger names from weaker names well enough to support the canonical research lane.

### Signal Definition
`score = return(20)`

### Files / Commands
- Audit command:
  - `python run_canonical_cross_sectional_rank_audit.py --score-mode return_20 --output-dir results/canonical_cross_sectional_rank_audit_r20`
- Output folder actually available in repo:
  - `results/canonical_cross_sectional_rank_audit_smoke/`
- First file inspected:
  - `bucket_summary.csv`

### Audit Results
- Horizon: 5 bars
  - Top: `-0.0074%`
  - Middle: `+0.0154%`
  - Bottom: `-0.0137%`
  - Top minus bottom: `+0.0063%`
- Horizon: 10 bars
  - Top: `-0.0174%`
  - Middle: `+0.0159%`
  - Bottom: `-0.0174%`
  - Top minus bottom: `+0.0000%`

### Classification
- weak

### Decision
- refine signal

### Notes
Top did not beat middle. The only positive feature was that top was slightly better than bottom, but the spread was tiny and not good enough to justify confidence.

---

## Experiment: return_20_plus_60
Date: 2026-04-18
Status: completed

### Hypothesis
Combining 20-bar and 60-bar returns might improve ranking separation by blending short and medium-horizon strength.

### Signal Definition
`score = return(20) + return(60)`

### Files / Commands
- Audit command:
  - `python run_canonical_cross_sectional_rank_audit.py --score-mode return_20_plus_60 --output-dir results/canonical_cross_sectional_rank_audit_r20_r60`
- Output folder:
  - `results/canonical_cross_sectional_rank_audit_r20_r60/`
- First file inspected:
  - `bucket_summary.csv`

### Audit Results
- Horizon: 5 bars
  - Top: `-0.0132%`
  - Middle: `-0.0043%`
  - Bottom: `+0.0061%`
  - Top minus bottom: `-0.0194%`
- Horizon: 10 bars
  - Top: `-0.0336%`
  - Middle: `-0.0018%`
  - Bottom: `+0.0092%`
  - Top minus bottom: `-0.0429%`
- Horizon: 20 bars
  - Top: `-0.0873%`
  - Middle: `+0.0271%`
  - Bottom: `+0.0081%`
  - Top minus bottom: `-0.0954%`

### Classification
- inverted

### Decision
- abandon

### Notes
This did not fix the middle-bucket problem. It made the signal worse than the `return_20` baseline by pushing top below both middle and bottom.

---

## Experiment: trend_consistency_20
Date: 2026-04-18
Status: completed

### Hypothesis
Persistent time spent above the 20-bar SMA might separate steadier trend leaders from noisier names better than raw trailing returns.

### Signal Definition
`score = rolling_mean((close > SMA(20)) ? 1 : 0, window=20)`

### Files / Commands
- Audit command:
  - `python run_canonical_cross_sectional_rank_audit.py --score-mode trend_consistency_20 --output-dir results/canonical_cross_sectional_rank_audit_trend_consistency`
- Output folder:
  - `results/canonical_cross_sectional_rank_audit_trend_consistency/`
- First file inspected:
  - `bucket_summary.csv`

### Audit Results
- Horizon: 5 bars
  - Top: `+0.0001%`
  - Middle: `-0.0069%`
  - Bottom: `+0.0011%`
  - Top minus bottom: `-0.0009%`
- Horizon: 10 bars
  - Top: `+0.0132%`
  - Middle: `-0.0231%`
  - Bottom: `-0.0081%`
  - Top minus bottom: `+0.0213%`
- Horizon: 20 bars
  - Top: `+0.0422%`
  - Middle: `-0.0714%`
  - Bottom: `-0.0221%`
  - Top minus bottom: `+0.0642%`

### Classification
- weak

### Decision
- refine signal

### Notes
This is better than the prior return-based scores because top leads at 10 and 20 bars, but it still fails clean ordering at 5 bars and the spreads remain small. That is enough to keep researching the signal layer, but not enough to justify moving forward to validation.

---

## Experiment: trend_consistency_20_x_slope_20_over_atr_20
Date: 2026-04-18
Status: completed

### Hypothesis
Names that stay consistently above a 20-bar trend and also have a rising trend after volatility normalization might separate genuine leaders from noisy movers better than consistency alone.

### Signal Definition
`score = trend_consistency_20 * (slope_20 / atr_20)`

### Files / Commands
- Audit command:
  - `python run_canonical_cross_sectional_rank_audit.py --score-mode trend_consistency_20_x_slope_20_over_atr_20 --output-dir results/canonical_cross_sectional_rank_audit_trend_consistency_slope_atr`
- Audit output folder:
  - `results/canonical_cross_sectional_rank_audit_trend_consistency_slope_atr/`
- Validation command:
  - `python run_canonical_rank_pullback_validation.py --score-mode trend_consistency_20_x_slope_20_over_atr_20 --output-dir results/canonical_rank_pullback_validation_trend_consistency_slope_atr`
- Validation output folder:
  - `results/canonical_rank_pullback_validation_trend_consistency_slope_atr/`
- First files inspected:
  - `bucket_summary.csv`
  - `stitched_selected_summary.csv`

### Audit Results
- Horizon: 5 bars
  - Top: `+0.0067%`
  - Middle: `-0.0105%`
  - Bottom: `-0.0069%`
  - Top minus bottom: `+0.0136%`
- Horizon: 10 bars
  - Top: `+0.0310%`
  - Middle: `-0.0492%`
  - Bottom: `-0.0054%`
  - Top minus bottom: `+0.0364%`
- Horizon: 20 bars
  - Top: `+0.0512%`
  - Middle: `-0.0888%`
  - Bottom: `-0.0131%`
  - Top minus bottom: `+0.0644%`

### Classification
- promising

### Decision
- advance to validation

### Validation Results
- Stitched OOS baseline:
  - Trade count: `48`
  - Expectancy: `5.8342`
  - Profit factor: `1.8678`
  - Max drawdown: `2.6897%`
  - Total PnL: `280.04`
- Stitched OOS pullback:
  - Trade count: `42`
  - Expectancy: `6.0149`
  - Profit factor: `1.9284`
  - Max drawdown: `1.9764%`
  - Total PnL: `252.62`
- Baseline vs pullback verdict:
  - Pullback improved expectancy, profit factor, and drawdown, while giving up `6` trades.
- Framework promotion flag:
  - `promotion_ready = true`

### Notes
This is the first signal in the canonical lane that stayed meaningfully positive in stitched OOS validation. The encouraging part is that both baseline and pullback remained profitable and pullback improved quality metrics. The caution is that the current run still has only one walk-forward fold, so this should be treated as promising enough to continue, not as fully proven.

### Robustness Validation
- Robustness command:
  - `python run_canonical_rank_pullback_validation.py --score-mode trend_consistency_20_x_slope_20_over_atr_20 --train-days 30 --test-days 10 --step-days 10 --output-dir results/canonical_rank_pullback_validation_trend_consistency_slope_atr_robust`
- Robustness output folder:
  - `results/canonical_rank_pullback_validation_trend_consistency_slope_atr_robust/`
- Multi-fold stitched OOS baseline:
  - Trade count: `57`
  - Expectancy: `-0.1138`
  - Profit factor: `0.9897`
  - Max drawdown: `6.6838%`
  - Total PnL: `-6.48`
- Multi-fold stitched OOS pullback:
  - Trade count: `50`
  - Expectancy: `4.5635`
  - Profit factor: `1.7004`
  - Max drawdown: `2.0272%`
  - Total PnL: `228.18`
- Fold pattern:
  - Fold 1 failed for both families
  - Folds 2 and 3 were positive
- Time-slice pattern:
  - March was weak, especially for baseline
  - April was clearly stronger
- Robustness classification:
  - somewhat robust
- Final decision:
  - continue, but do not treat as production-ready yet

### Failure Analysis Note
- Robustness status:
  - somewhat robust, with pullback holding up better than baseline
- Main failure conditions:
  - early March / fold 1 follow-through failure
  - longer-hold fold 1 selections that stayed weak instead of recovering
  - loss concentration in names like `XOM`, `COP`, and `QCOM`
- What pullback appears to do:
  - reduce bad-entry damage and improve average trade quality more than improve raw hit rate
- Next planned research step:
  - compare fold-1 trades versus folds 2 and 3 to isolate whether the weakness came mostly from symbol mix, poor follow-through, or longer hold exposure

### Trade-Level Failure Analysis
- Trade-level result:
  - fold 1 weakness was mainly a poor follow-through regime
- Evidence:
  - fold 1 pullback average MFE `+1.6237%` vs `+2.4424%` in folds 2-3
  - fold 1 pullback average MAE `-2.3908%` vs `-1.5636%` in folds 2-3
  - fold 1 entry distance below recent high was only modestly worse, so entry location alone does not explain the failure
- Concentration:
  - fold 1 losses clustered in `QCOM`, `COP`, and `XOM`
- Current diagnosis:
  - primary mechanism is poor follow-through after entry, with symbol concentration making it worse

### Early Exit Check
- Modification tested:
  - `early_no_follow_through_exit`
  - exit next bar open if `bars_since_entry >= 3` and trade is still non-positive
- Validation result:
  - harmful
- Key outcomes versus prior pullback:
  - expectancy fell from `4.5635` to `-2.0160`
  - profit factor fell from `1.7004` to `0.6092`
  - max drawdown worsened from `2.0272%` to `8.6920%`
  - trade count jumped from `50` to `146`
- Interpretation:
  - the rule appears to increase churn and repeated re-entry more than it improves trade quality

### Follow-Through Filter Check
- Modification tested:
  - `use_recent_follow_through_filter`
  - only allow new pullback entries when recent realized top-bucket follow-through is positive
- Validation result:
  - harmful
- Key outcomes versus prior pullback:
  - expectancy fell from `4.5635` to `-2.3414`
  - profit factor fell from `1.7004` to `0.7397`
  - max drawdown worsened from `2.0272%` to `3.5254%`
  - trade count fell from `50` to `40`
- Interpretation:
  - the filter reduced trades moderately, but it did not isolate the bad regime cleanly and it damaged the previously positive folds

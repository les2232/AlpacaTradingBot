# Canonical Cross-Sectional Rank Pullback v1

## Hypothesis

Recent cross-sectional strength contains usable information in the TradeOS research universe.

If that signal is real, a simple long-only implementation should show:

- top-ranked symbols outperforming middle and bottom buckets on forward returns
- positive top-minus-bottom spread across multiple horizons and date slices
- a tradable implementation that retains the edge after a simple trend and pullback entry refinement

The intent is to test one transparent edge, not to maximize performance.

## Canonical v1 Rules

### Phase 1: Signal Audit

- Universe: existing research universe conventions already used in the repo
- Score: rolling close-to-close return over `rank_lookback_bars`
- Ranking: descending by score at each timestamp across symbols with valid data
- Buckets: top / middle / bottom thirds
- Forward returns: configurable fixed horizons measured from signal bar close

### Phase 2: Tradable Strategy

- Side: long-only
- Eligible longs: symbols in the top `eligible_percent` of the cross-sectional ranking
- Trend filter: `close > SMA(50)`
- Immediate-entry baseline:
  - if a symbol is eligible and in trend, queue entry for the next bar open
- Pullback-entry canonical v1:
  - symbol must remain eligible
  - symbol must remain in trend
  - pullback depth = `(recent_high - close) / ATR`
  - recent high uses a fixed rolling lookback
  - pullback is valid when depth is at least `pullback_depth_atr`
  - entry trigger is simple reclaim behavior: previous bar was in valid pullback, current close is above the previous bar high
  - fill at next bar open
- Exit: fixed holding period only
- Positioning:
  - equal notional per position
  - one position per symbol
  - no pyramiding
  - fixed `max_positions`

## Parameters Intentionally Exposed

- `rank_lookback_bars`
- `eligible_percent`
- `forward_horizons`
- `pullback_depth_atr`
- `hold_bars`
- walk-forward window sizes

## Parameters Intentionally Not Optimized Yet

- trend SMA length stays fixed at 50
- ATR length stays fixed at 14
- recent-high lookback stays fixed at 20 bars
- no advanced exits
- no trailing logic
- no symbol-specific settings
- no extra volatility, volume, or market-regime filters beyond reporting slices
- no ML

## Validation Philosophy

- keep the grid deliberately small
- separate signal validity from execution refinement
- use walk-forward train/test windows
- log every tested variant
- report stitched out-of-sample results
- compare pullback against an immediate-entry baseline with the same rank and hold settings
- do not promote a candidate from the single best in-sample result

## Default Narrow Grid

- `rank_lookback_bars`: 20, 40
- `eligible_percent`: 0.20, 0.30
- `pullback_depth_atr`: 0.5, 1.0
- `hold_bars`: 5, 10

## Success Criteria

### Signal Validity

- top bucket mean forward return is better than middle and bottom for most tested horizons
- top-minus-bottom spread is positive across multiple horizons
- edge remains directionally consistent across multiple date slices and walk-forward windows

### Strategy Validity

- pullback stitched out-of-sample results improve at least one quality metric versus the immediate-entry baseline
- trade count does not collapse to near zero

### Robustness

- stitched out-of-sample behavior remains reasonable across folds
- performance is not dominated by one symbol
- nearby parameter variants are not wildly unstable

## Failure Criteria

- top bucket does not outperform middle and bottom in a durable way
- top-minus-bottom spread is inconsistent or mostly negative
- pullback only looks good by severely reducing trade count
- results are dominated by one symbol or one short date slice
- best-looking result comes from a lone parameter combination with weak stitched OOS support

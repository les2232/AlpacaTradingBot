## Executive Summary

Fold 1 failure was mainly a poor follow-through regime with materially worse adverse movement after entry, not just a case of entries being slightly farther from the recent high. The pullback variant still entered trades that went against it quickly in fold 1, and the losses were amplified by a bad symbol mix led by `QCOM`, `COP`, and `XOM`. Later folds were stronger because trades had better upside follow-through, shallower drawdowns after entry, and more gains from `C`, `BAC`, `WFC`, `GOOGL`, and `AMD`.

## MFE / MAE Comparison

Pullback trades only:

- Fold 1:
  - `15` trades
  - average return `-0.6973%`
  - median return `-0.4167%`
  - average MFE `+1.6237%`
  - median MFE `+1.2552%`
  - average MAE `-2.3908%`
  - median MAE `-2.9180%`
  - `66.7%` of trades reached more than `+1%` MFE
  - `73.3%` of trades saw worse than `-1%` MAE

- Folds 2-3:
  - `35` trades
  - average return `+0.9508%`
  - median return `+0.9454%`
  - average MFE `+2.4424%`
  - median MFE `+2.0114%`
  - average MAE `-1.5636%`
  - median MAE `-0.9667%`
  - `82.9%` of trades reached more than `+1%` MFE
  - `45.7%` of trades saw worse than `-1%` MAE

This is the clearest trade-level difference. Fold 1 had lower upside follow-through and much worse adverse movement after entry.

## Entry Quality Differences

Entry quality proxy at trade entry:

- Fold 1 average distance below recent high: `+0.6345%`
- Folds 2-3 average distance below recent high: `+0.4163%`

ATR-normalized entry distance:

- Fold 1 average: `1.2698 ATR`
- Folds 2-3 average: `1.1378 ATR`

Interpretation:

- Fold 1 entries were somewhat farther below the recent high on average, but the gap is not large enough by itself to explain the whole failure.
- Entry placement looks like a secondary issue.
- The larger problem is what happened after entry: fold 1 trades still suffered much deeper adverse movement and weaker realized follow-through.

## Holding Period Behavior

- Fold 1 average hold bars: `10.0`
- Folds 2-3 average hold bars: `7.43`
- Fold 1 average duration: `50.0` hours
- Folds 2-3 average duration: `51.0` hours

By fold:

- Fold 1 used only the `hold10` pullback variant
- Fold 2 also used `hold10` and was still positive
- Fold 3 used `hold5` and was the strongest fold

Interpretation:

- Longer hold exposure may have made fold 1 losses linger, but it is not the whole story because fold 2 also used `hold10` and worked.
- The stronger evidence is that fold 1 was a bad follow-through environment, not just a bad holding-period setting.

## Symbol Contribution Differences

Fold 1 pullback PnL by symbol:

- winners:
  - `GOOGL +26.17`
  - `LLY +17.15`
  - `WFC +3.91`
  - `C +2.51`
- losers:
  - `QCOM -50.53`
  - `COP -37.82`
  - `XOM -24.19`
  - `AMD -18.42`
  - `TSLA -16.56`

Folds 2-3 pullback PnL by symbol:

- winners:
  - `C +140.52`
  - `BAC +79.18`
  - `GOOGL +77.20`
  - `WFC +50.75`
  - `AMD +44.61`
  - `COP +32.75`
- losers:
  - `XOM -69.86`
  - `QCOM -25.08`
  - `ABBV -6.97`

Interpretation:

- Fold 1 losses were concentrated in a small cluster of names, especially `QCOM`, `COP`, and `XOM`.
- Later folds benefited from a much stronger contribution mix led by financials and `GOOGL`.
- `XOM` remained a persistent weak point even in the stronger folds.

## Most Likely Failure Mechanism

The evidence points most strongly to poor follow-through after entry, with bad symbol concentration as a secondary driver.

Why:

- Fold 1 had lower MFE and much worse MAE than folds 2-3.
- Entry distance from recent high was only modestly worse, so entry location alone does not explain the failure.
- Fold 1 used a longer hold, which likely increased exposure to losses, but longer hold is not sufficient by itself because fold 2 also used `hold10` and still worked.
- The worst losses came from a concentrated group of names, especially `QCOM`, `COP`, and `XOM`.

Answer to the core question:

Fold 1 failure was mainly due to poor follow-through, with symbol concentration making it worse. Holding-period exposure contributed, but it does not look like the primary mechanism.

## Executive Summary

The strategy did not fail randomly. The main weak patch was the early March window, where both baseline and pullback selected longer-hold variants and both families lost money, with baseline losing much more. Later windows were stronger because trade quality improved sharply, drawdowns stayed contained, and the best contributions came from a more favorable mix of names such as `C`, `BAC`, `WFC`, and `GOOGL`.

## Fold-Level Failure Analysis

Fold 1 was the clear failure case.

- Baseline fold 1: `19` trades, expectancy `-17.0469`, win rate `21.05%`, profit factor `0.1645`, max drawdown `6.6838%`
- Pullback fold 1: `15` trades, expectancy `-6.9728`, win rate `33.33%`, profit factor `0.3415`, max drawdown `1.7385%`

Why fold 1 was weak:

- Both families selected `hold10` variants in fold 1, which suggests the chosen trades needed more time to work but instead carried losses.
- The hit rates were extremely low, especially for baseline.
- Pullback helped materially, but only by shrinking the damage. It did not turn the fold positive.

Why later folds were stronger:

- Fold 2 flipped positive for both families:
  - baseline expectancy `8.8768`, profit factor `4.9005`
  - pullback expectancy `7.2158`, profit factor `3.8635`
- Fold 3 was stronger still for pullback:
  - baseline expectancy `7.7706`, profit factor `1.7193`
  - pullback expectancy `11.6722`, profit factor `2.6928`
- The later folds used a mix of `hold5` and `hold10`, but the winning windows showed much higher win rates and much smaller drawdowns than fold 1.

The simplest interpretation is that the strategy is vulnerable when the ranked names do not follow through after entry. In that regime, the longer holds in fold 1 let losses persist instead of resolving quickly.

## Monthly / Time-Slice Analysis

March was weak:

- Baseline March: `39` trades, expectancy `-3.7527`, profit factor `0.6621`, max drawdown `6.6838%`
- Pullback March: `32` trades, expectancy `0.5649`, profit factor `1.0896`, max drawdown `1.9920%`

April was stronger:

- Baseline April: `18` trades, expectancy `7.7706`, profit factor `1.7193`, max drawdown `2.4568%`
- Pullback April: `18` trades, expectancy `11.6722`, profit factor `2.6928`, max drawdown `1.4323%`

Why March was weak and April stronger:

- March contains the full fold 1 failure and still drags baseline negative overall.
- Pullback barely stayed positive in March, so this was not a clean edge month.
- April had stronger follow-through and much better trade quality for both families, especially pullback.

This means the strategy is not evenly strong through time. It appears sensitive to regime quality, with March acting like a hostile follow-through environment and April acting like a cooperative one.

## Symbol Concentration Review

Wins and losses are not coming from one single name, but they are concentrated in a few symbols.

Main positive contributors:

- baseline: `C` `+133.06`, `BAC` `+48.79`, `GOOGL` `+38.82`, `WFC` `+28.48`
- pullback: `C` `+143.02`, `GOOGL` `+103.37`, `BAC` `+79.18`, `WFC` `+54.66`

Main negative contributors:

- baseline: `XOM` `-137.89`, `COP` `-86.69`, `HON` `-32.26`, `QCOM` `-23.38`
- pullback: `XOM` `-94.05`, `QCOM` `-75.61`, `TSLA` `-16.56`, `ABBV` `-8.44`

Sector-style pattern:

- Financials did well: `C`, `BAC`, `WFC`, and even `JPM` on small count
- Energy was the main weak cluster: `XOM` and `COP`
- Some tech names split: `GOOGL` worked well, while `QCOM` was poor and `AMD` improved mainly under pullback

So the strategy is not dominated by one symbol, but it is clearly exposed to cross-sectional sector dispersion. When energy names score well and fail to follow through, the strategy suffers.

## What Pullback Seems To Be Doing

Pullback is helping more by improving average trade quality and limiting damage than by raising raw hit rate.

Stitched selected OOS:

- baseline: expectancy `-0.1138`, win rate `63.16%`, profit factor `0.9897`, max drawdown `6.6838%`
- pullback: expectancy `4.5635`, win rate `62.00%`, profit factor `1.7004`, max drawdown `2.0272%`

Important point:

- Pullback win rate is actually slightly lower than baseline.
- But expectancy and profit factor are much better, and drawdown is dramatically lower.

That suggests pullback is mainly helping by avoiding worse entries and reducing the size of bad trades, not by simply making more trades win. Fold 1 makes this especially clear:

- baseline fold 1 drawdown `6.6838%`
- pullback fold 1 drawdown `1.7385%`

## Most Likely Failure Conditions

- Weak follow-through regime after ranking, especially when selected names do not continue after entry
- Earlier March-style environment where the chosen signals require longer holds and still fail
- Cross-sectional leadership concentrated in weaker groups such as energy names that rank well but do not monetize well

## Recommended Next Step

Stay in diagnosis mode and compare the actual fold-1 trade list against folds 2 and 3 to see whether the main problem was failed follow-through after entry, symbol mix, or longer hold exposure. This is the cleanest next step before considering any strategy changes.

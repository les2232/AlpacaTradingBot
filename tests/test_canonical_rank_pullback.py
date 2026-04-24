import pandas as pd
import pytest

from research.canonical_rank_pullback import (
    StrategyVariant,
    assign_cross_sectional_ranks,
    build_spread_observations,
    evaluate_strategy_success,
    generate_entry_candidates,
    generate_strategy_variants,
    normalize_rank_lookbacks_for_score_mode,
    normalize_score_mode,
    prepare_panel_features,
    run_strategy_backtest,
    select_best_variant,
    summarize_bucket_forward_returns,
    summarize_spread,
)


def _sample_panel() -> pd.DataFrame:
    rows = []
    base_ts = pd.Timestamp("2026-01-05T15:30:00Z")
    for symbol, closes in {
        "AAA": [100 + idx for idx in range(25)],
        "BBB": [100 for _ in range(25)],
        "CCC": [100 - idx for idx in range(25)],
    }.items():
        for idx, close in enumerate(closes):
            rows.append(
                {
                    "symbol": symbol,
                    "timestamp": base_ts + pd.Timedelta(minutes=15 * idx),
                    "open": close,
                    "high": close + 0.5,
                    "low": close - 0.5,
                    "close": close,
                    "volume": 1000 + idx,
                }
            )
    return pd.DataFrame(rows)


def test_prepare_panel_features_adds_core_columns() -> None:
    prepared = prepare_panel_features(_sample_panel(), rank_lookback_bars=20, audit_horizons=(1, 2))
    assert {
        "score_return",
        "return_20",
        "return_60",
        "sma_20",
        "close_above_sma_20",
        "trend_consistency_20",
        "slope_20",
        "score_mode",
        "trend_sma",
        "recent_high",
        "atr",
        "pullback_distance_atr",
        "fwd_1b_return_pct",
        "fwd_2b_return_pct",
    } <= set(prepared.columns)


def test_prepare_panel_features_computes_return_20_plus_60_score() -> None:
    rows = []
    base_ts = pd.Timestamp("2026-01-05T15:30:00Z")
    closes = list(range(100, 171))
    for idx, close in enumerate(closes):
        rows.append(
            {
                "symbol": "AAA",
                "timestamp": base_ts + pd.Timedelta(minutes=15 * idx),
                "open": float(close),
                "high": float(close) + 0.5,
                "low": float(close) - 0.5,
                "close": float(close),
                "volume": 1000 + idx,
            }
        )
    prepared = prepare_panel_features(
        pd.DataFrame(rows),
        rank_lookback_bars=20,
        audit_horizons=(1,),
        score_mode="return_20_plus_60",
    )
    row = prepared.iloc[-1]
    expected_20 = closes[-1] / closes[-21] - 1.0
    expected_60 = closes[-1] / closes[-61] - 1.0
    assert round(float(row["return_20"]), 10) == round(expected_20, 10)
    assert round(float(row["return_60"]), 10) == round(expected_60, 10)
    assert round(float(row["score_return"]), 10) == round(expected_20 + expected_60, 10)


def test_prepare_panel_features_computes_trend_consistency_20_score() -> None:
    rows = []
    base_ts = pd.Timestamp("2026-01-05T15:30:00Z")
    closes = [100.0] * 20 + [101.0] * 19 + [99.0]
    for idx, close in enumerate(closes):
        rows.append(
            {
                "symbol": "AAA",
                "timestamp": base_ts + pd.Timedelta(minutes=15 * idx),
                "open": close,
                "high": close + 0.5,
                "low": close - 0.5,
                "close": close,
                "volume": 1000 + idx,
            }
        )
    prepared = prepare_panel_features(
        pd.DataFrame(rows),
        rank_lookback_bars=20,
        audit_horizons=(1,),
        score_mode="trend_consistency_20",
    )
    valid = prepared["trend_consistency_20"].dropna()
    assert not valid.empty
    assert ((valid >= 0.0) & (valid <= 1.0)).all()
    row = prepared.iloc[-1]
    assert float(row["close_above_sma_20"]) == 0.0
    assert round(float(row["trend_consistency_20"]), 10) == round(19.0 / 20.0, 10)
    assert round(float(row["score_return"]), 10) == round(19.0 / 20.0, 10)


def test_prepare_panel_features_computes_slope_20_and_composite_score() -> None:
    rows = []
    base_ts = pd.Timestamp("2026-01-05T15:30:00Z")
    closes = [100.0] * 20 + [120.0] * 20
    for idx, close in enumerate(closes):
        rows.append(
            {
                "symbol": "AAA",
                "timestamp": base_ts + pd.Timedelta(minutes=15 * idx),
                "open": close,
                "high": close + 0.5,
                "low": close - 0.5,
                "close": close,
                "volume": 1000 + idx,
            }
        )
    prepared = prepare_panel_features(
        pd.DataFrame(rows),
        rank_lookback_bars=20,
        audit_horizons=(1,),
        score_mode="trend_consistency_20_x_slope_20",
    )
    row = prepared.iloc[-1]
    expected_sma_20 = sum(closes[-20:]) / 20.0
    expected_prior_sma_20 = sum(closes[:20]) / 20.0
    expected_slope_20 = expected_sma_20 / expected_prior_sma_20 - 1.0
    expected_consistency = 19.0 / 20.0
    assert round(float(row["slope_20"]), 10) == round(expected_slope_20, 10)
    assert round(float(row["trend_consistency_20"]), 10) == round(expected_consistency, 10)
    assert round(float(row["score_return"]), 10) == round(expected_consistency * expected_slope_20, 10)


def test_prepare_panel_features_computes_atr_normalized_composite_score() -> None:
    rows = []
    base_ts = pd.Timestamp("2026-01-05T15:30:00Z")
    closes = [100.0] * 20 + [120.0] * 20
    for idx, close in enumerate(closes):
        rows.append(
            {
                "symbol": "AAA",
                "timestamp": base_ts + pd.Timedelta(minutes=15 * idx),
                "open": close,
                "high": close + 1.0,
                "low": close - 1.0,
                "close": close,
                "volume": 1000 + idx,
            }
        )
    prepared = prepare_panel_features(
        pd.DataFrame(rows),
        rank_lookback_bars=20,
        audit_horizons=(1,),
        score_mode="trend_consistency_20_x_slope_20_over_atr_20",
        atr_bars=20,
    )
    row = prepared.iloc[-1]
    expected_sma_20 = sum(closes[-20:]) / 20.0
    expected_prior_sma_20 = sum(closes[:20]) / 20.0
    expected_slope_20 = expected_sma_20 / expected_prior_sma_20 - 1.0
    expected_consistency = 19.0 / 20.0
    expected_atr_20 = ((19.0 * 2.0) + 21.0) / 20.0
    assert round(float(row["atr"]), 10) == round(expected_atr_20, 10)
    assert round(float(row["score_return"]), 10) == round(
        expected_consistency * (expected_slope_20 / expected_atr_20), 10
    )


def test_invalid_score_mode_fails_clearly() -> None:
    with pytest.raises(ValueError, match="Unsupported score_mode"):
        normalize_score_mode("bad_mode")


def test_fixed_score_mode_rejects_non_20_rank_lookback() -> None:
    with pytest.raises(ValueError, match="fixed 20-bar anchor"):
        normalize_rank_lookbacks_for_score_mode("return_20_plus_60", (20, 40))


def test_rank_assignment_and_spread_summary_work() -> None:
    prepared = prepare_panel_features(_sample_panel(), rank_lookback_bars=20, audit_horizons=(1,), score_mode="return_20")
    ranked, diagnostics = assign_cross_sectional_ranks(prepared, eligible_percent=0.34)
    assert diagnostics["timestamps_ranked"] > 0

    bucket_summary = summarize_bucket_forward_returns(ranked, horizons=(1,))
    spread_summary = summarize_spread(build_spread_observations(ranked, horizons=(1,)))
    top = bucket_summary[bucket_summary["bucket"] == "top"].iloc[0]
    bottom = bucket_summary[bucket_summary["bucket"] == "bottom"].iloc[0]
    assert float(top["mean_return_pct"]) > float(bottom["mean_return_pct"])
    assert float(spread_summary.iloc[0]["mean_return_pct"]) > 0


def test_generate_strategy_variants_counts_small_grid() -> None:
    variants = generate_strategy_variants(
        score_mode="return_20",
        rank_lookbacks=(20, 40),
        eligible_percents=(0.2, 0.3),
        hold_bars_list=(5, 10),
        pullback_depths=(0.5, 1.0),
    )
    ranking_only = [variant for variant in variants if variant.family == "ranking_only"]
    baseline = [variant for variant in variants if variant.family == "baseline"]
    pullback = [variant for variant in variants if variant.family == "pullback"]
    assert len(ranking_only) == 8
    assert len(baseline) == 8
    assert len(pullback) == 16


def test_generate_entry_candidates_respects_recent_follow_through_filter() -> None:
    timestamps = pd.date_range("2026-01-05T15:30:00Z", periods=30, freq="15min", tz="UTC")
    ranked_df = pd.DataFrame(
        [
            {
                "symbol": "AAA",
                "timestamp": ts,
                "eligible_long": True,
                "in_uptrend": True,
                "next_open": 100.0 if idx < 29 else pd.NA,
                "prev_pullback_distance_atr": 0.6,
                "prev_high": 99.0,
                "close": 100.0,
                "bucket": "top",
                "fwd_5b_return_pct": 1.0,
            }
            for idx, ts in enumerate(timestamps)
        ]
    )
    variant = StrategyVariant(
        family="pullback",
        score_mode="trend_consistency_20_x_slope_20_over_atr_20",
        rank_lookback_bars=20,
        eligible_percent=0.2,
        hold_bars=5,
        pullback_depth_atr=0.5,
        use_recent_follow_through_filter=True,
    )
    candidates = generate_entry_candidates(ranked_df, variant=variant)
    assert len(candidates) == 5
    assert candidates["timestamp"].min() == timestamps[24]


def test_generate_entry_candidates_ranking_only_ignores_uptrend_gate() -> None:
    ranked_df = pd.DataFrame(
        [
            {
                "symbol": "AAA",
                "timestamp": pd.Timestamp("2026-01-05T15:30:00Z"),
                "eligible_long": True,
                "in_uptrend": False,
                "next_open": 100.0,
                "prev_pullback_distance_atr": 0.6,
                "prev_high": 99.0,
                "close": 100.0,
            }
        ]
    )
    ranking_variant = StrategyVariant(
        family="ranking_only",
        score_mode="trend_consistency_20_x_slope_20_over_atr_20",
        rank_lookback_bars=20,
        eligible_percent=0.2,
        hold_bars=5,
    )
    baseline_variant = StrategyVariant(
        family="baseline",
        score_mode="trend_consistency_20_x_slope_20_over_atr_20",
        rank_lookback_bars=20,
        eligible_percent=0.2,
        hold_bars=5,
    )
    assert len(generate_entry_candidates(ranked_df, variant=ranking_variant)) == 1
    assert generate_entry_candidates(ranked_df, variant=baseline_variant).empty


def test_run_strategy_backtest_applies_early_no_follow_through_exit() -> None:
    timestamps = pd.date_range("2026-01-05T15:30:00Z", periods=7, freq="15min", tz="UTC")
    ranked_df = pd.DataFrame(
        [
            {
                "symbol": "AAA",
                "timestamp": ts,
                "open": open_,
                "high": high,
                "low": low,
                "close": close,
                "bar_index": idx,
                "eligible_long": True,
                "in_uptrend": True,
                "next_open": next_open,
                "next_timestamp": timestamps[idx + 1] if idx + 1 < len(timestamps) else pd.NaT,
                "prev_pullback_distance_atr": 0.6,
                "pullback_distance_atr": 0.6,
                "prev_high": prev_high,
                "score_return": 1.0,
                "cross_section_rank": 1,
                "cross_section_size": 3,
            }
            for idx, (ts, open_, high, low, close, next_open, prev_high) in enumerate(
                [
                    (timestamps[0], 100.0, 100.4, 99.8, 100.2, 100.0, 100.1),
                    (timestamps[1], 100.0, 100.2, 99.7, 99.9, 99.8, 99.8),
                    (timestamps[2], 99.8, 100.0, 99.4, 99.7, 99.6, 99.6),
                    (timestamps[3], 99.6, 99.9, 99.2, 99.5, 99.4, 99.4),
                    (timestamps[4], 99.4, 99.6, 99.0, 99.3, 99.2, 99.2),
                    (timestamps[5], 99.2, 99.4, 98.9, 99.1, 99.0, 99.1),
                    (timestamps[6], 99.0, 99.2, 98.7, 98.9, None, 98.9),
                ]
            )
        ]
    )
    variant = StrategyVariant(
        family="pullback",
        score_mode="trend_consistency_20_x_slope_20_over_atr_20",
        rank_lookback_bars=20,
        eligible_percent=0.2,
        hold_bars=5,
        pullback_depth_atr=0.5,
        use_early_no_follow_through_exit=True,
    )
    trades_df, _, _ = run_strategy_backtest(ranked_df, variant=variant, max_positions=1, position_size=1000.0, commission_per_order=0.0, slippage_per_share=0.0)
    assert len(trades_df) == 1
    trade = trades_df.iloc[0]
    assert pd.Timestamp(trade["entry_ts"]) == timestamps[1]
    assert pd.Timestamp(trade["exit_ts"]) == timestamps[5]
    assert int(trade["hold_bars"]) == 4
    assert bool(trade["use_early_no_follow_through_exit"]) is True


def test_select_best_variant_prefers_positive_expectancy_with_trade_count() -> None:
    winner = select_best_variant(
        [
            {"variant_id": "a", "expectancy": 0.5, "profit_factor": 1.2, "trade_count": 2},
            {"variant_id": "b", "expectancy": 0.3, "profit_factor": 1.1, "trade_count": 5},
        ]
    )
    assert winner is not None
    assert winner["variant_id"] == "b"


def test_evaluate_strategy_success_checks_trade_count_and_quality() -> None:
    summary_df = pd.DataFrame(
        [
            {"family": "baseline", "expectancy": 5.0, "profit_factor": 1.1, "max_drawdown_pct": 8.0, "trade_count": 10},
            {"family": "pullback", "expectancy": 6.0, "profit_factor": 1.3, "max_drawdown_pct": 6.0, "trade_count": 7},
        ]
    )
    trades_df = pd.DataFrame(
        [
            {"family": "baseline", "symbol": "AAA", "realized_pnl": 5.0},
            {"family": "baseline", "symbol": "BBB", "realized_pnl": 3.0},
            {"family": "pullback", "symbol": "AAA", "realized_pnl": 6.0},
            {"family": "pullback", "symbol": "BBB", "realized_pnl": 4.0},
        ]
    )
    result = evaluate_strategy_success(summary_df, trades_df)
    assert result["pullback_improves_quality_metric"] is True
    assert result["pullback_trade_count_ok"] is True
    assert result["promotion_ready"] is True

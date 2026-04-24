import pandas as pd

from run_cross_sectional_edge_audit import (
    assign_rank_buckets,
    build_spread_observations,
    prepare_cross_sectional_frame,
    summarize_buckets,
    summarize_spread,
)


def test_assign_rank_buckets_is_deterministic_with_equal_count_buckets() -> None:
    frame = pd.DataFrame(
        [
            {"timestamp": pd.Timestamp("2026-01-01T15:30:00Z"), "symbol": "AAA", "lookback_return": 0.04, "month": "2026-01", "date": "2026-01-01", "time_bucket": "afternoon", "fwd_5b_return_pct": 1.0},
            {"timestamp": pd.Timestamp("2026-01-01T15:30:00Z"), "symbol": "BBB", "lookback_return": 0.04, "month": "2026-01", "date": "2026-01-01", "time_bucket": "afternoon", "fwd_5b_return_pct": 0.5},
            {"timestamp": pd.Timestamp("2026-01-01T15:30:00Z"), "symbol": "CCC", "lookback_return": 0.01, "month": "2026-01", "date": "2026-01-01", "time_bucket": "afternoon", "fwd_5b_return_pct": -0.2},
            {"timestamp": pd.Timestamp("2026-01-01T15:30:00Z"), "symbol": "DDD", "lookback_return": -0.03, "month": "2026-01", "date": "2026-01-01", "time_bucket": "afternoon", "fwd_5b_return_pct": -0.8},
        ]
    )

    result, diagnostics = assign_rank_buckets(frame)

    assert diagnostics["timestamps_ranked"] == 1
    assert result.loc[result["symbol"] == "AAA", "bucket"].iloc[0] == "top"
    assert result.loc[result["symbol"] == "BBB", "bucket"].iloc[0] == "middle"
    assert result.loc[result["symbol"] == "DDD", "bucket"].iloc[0] == "bottom"


def test_prepare_cross_sectional_frame_computes_lookback_and_forward_returns() -> None:
    df = pd.DataFrame(
        [
            {"symbol": "AAA", "timestamp": "2026-01-01T15:30:00Z", "close": 100.0},
            {"symbol": "AAA", "timestamp": "2026-01-01T15:45:00Z", "close": 101.0},
            {"symbol": "AAA", "timestamp": "2026-01-01T16:00:00Z", "close": 103.0},
            {"symbol": "BBB", "timestamp": "2026-01-01T15:30:00Z", "close": 200.0},
            {"symbol": "BBB", "timestamp": "2026-01-01T15:45:00Z", "close": 198.0},
            {"symbol": "BBB", "timestamp": "2026-01-01T16:00:00Z", "close": 197.0},
        ]
    )

    prepared, diagnostics = prepare_cross_sectional_frame(df, lookback_bars=1, horizons=(1,))
    aaa_mid = prepared[(prepared["symbol"] == "AAA") & (prepared["timestamp"] == pd.Timestamp("2026-01-01T15:45:00Z"))].iloc[0]

    assert round(float(aaa_mid["lookback_return"]), 4) == 0.01
    assert round(float(aaa_mid["fwd_1b_return_pct"]), 4) == round((103.0 / 101.0 - 1.0) * 100.0, 4)
    assert diagnostics["rows_with_lookback"] == 4


def test_spread_summary_reflects_top_minus_bottom_difference() -> None:
    observations = pd.DataFrame(
        [
            {"timestamp": pd.Timestamp("2026-01-01T15:30:00Z"), "symbol": "AAA", "bucket": "top", "month": "2026-01", "date": "2026-01-01", "time_bucket": "afternoon", "fwd_5b_return_pct": 1.0},
            {"timestamp": pd.Timestamp("2026-01-01T15:30:00Z"), "symbol": "BBB", "bucket": "bottom", "month": "2026-01", "date": "2026-01-01", "time_bucket": "afternoon", "fwd_5b_return_pct": -1.0},
            {"timestamp": pd.Timestamp("2026-01-01T15:45:00Z"), "symbol": "AAA", "bucket": "top", "month": "2026-01", "date": "2026-01-01", "time_bucket": "afternoon", "fwd_5b_return_pct": 0.5},
            {"timestamp": pd.Timestamp("2026-01-01T15:45:00Z"), "symbol": "BBB", "bucket": "bottom", "month": "2026-01", "date": "2026-01-01", "time_bucket": "afternoon", "fwd_5b_return_pct": 0.0},
        ]
    )

    spread_obs = build_spread_observations(observations, horizons=(5,))
    spread_summary = summarize_spread(spread_obs)
    row = spread_summary.iloc[0]

    assert row["observation_count"] == 2
    assert row["mean_return_pct"] == 1.25
    assert row["hit_rate_pct"] == 100.0


def test_summaries_handle_empty_input() -> None:
    empty_bucket = summarize_buckets(pd.DataFrame(), horizons=(5, 10))
    empty_spread = summarize_spread(pd.DataFrame())

    assert empty_bucket.empty
    assert empty_spread.empty

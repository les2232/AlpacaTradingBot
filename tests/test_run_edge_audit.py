import pandas as pd

from run_edge_audit import (
    classify_trend_proxy,
    classify_volatility_regime,
    compute_forward_return_pcts,
    summarize_forward_returns,
)


def test_compute_forward_return_pcts_includes_costs() -> None:
    gross_pct, net_pct = compute_forward_return_pcts(
        entry_open=100.0,
        entry_fill=100.5,
        future_close=102.0,
        slippage=0.5,
        commission=1.0,
        position_size=1000.0,
    )

    assert round(gross_pct, 4) == 2.0
    assert round(net_pct, 4) == round((((1000.0 / 100.5) * 101.5) - 1.0 - 1001.0) / 1000.0 * 100.0, 4)


def test_classify_regimes() -> None:
    assert classify_volatility_regime(None) == "unknown"
    assert classify_volatility_regime(10.0) == "low_vol"
    assert classify_volatility_regime(50.0) == "mid_vol"
    assert classify_volatility_regime(90.0) == "high_vol"

    assert classify_trend_proxy(None) == "unknown"
    assert classify_trend_proxy(10.0) == "range"
    assert classify_trend_proxy(22.0) == "mixed"
    assert classify_trend_proxy(30.0) == "trend"


def test_summarize_forward_returns_handles_zero_signals() -> None:
    result = summarize_forward_returns(
        pd.DataFrame(),
        group_cols=["symbol"],
        horizons=(1, 2, 4, 8),
    )

    assert list(result.columns) == [
        "symbol",
        "horizon_bars",
        "sample_count",
        "avg_gross_return_pct",
        "median_gross_return_pct",
        "gross_win_rate_pct",
        "avg_net_return_pct",
        "median_net_return_pct",
        "net_win_rate_pct",
    ]
    assert result.empty


def test_summarize_forward_returns_groups_correctly() -> None:
    signals = pd.DataFrame(
        [
            {
                "symbol": "AAA",
                "fwd_1b_gross_pct": 1.0,
                "fwd_1b_net_pct": 0.5,
                "fwd_2b_gross_pct": 2.0,
                "fwd_2b_net_pct": 1.5,
            },
            {
                "symbol": "AAA",
                "fwd_1b_gross_pct": -1.0,
                "fwd_1b_net_pct": -1.5,
                "fwd_2b_gross_pct": 0.0,
                "fwd_2b_net_pct": -0.5,
            },
            {
                "symbol": "BBB",
                "fwd_1b_gross_pct": 3.0,
                "fwd_1b_net_pct": 2.5,
                "fwd_2b_gross_pct": 4.0,
                "fwd_2b_net_pct": 3.5,
            },
        ]
    )

    result = summarize_forward_returns(signals, group_cols=["symbol"], horizons=(1, 2))

    aaa_1b = result[(result["symbol"] == "AAA") & (result["horizon_bars"] == 1)].iloc[0]
    bbb_2b = result[(result["symbol"] == "BBB") & (result["horizon_bars"] == 2)].iloc[0]

    assert aaa_1b["sample_count"] == 2
    assert aaa_1b["avg_gross_return_pct"] == 0.0
    assert aaa_1b["gross_win_rate_pct"] == 50.0
    assert aaa_1b["avg_net_return_pct"] == -0.5
    assert bbb_2b["sample_count"] == 1
    assert bbb_2b["avg_gross_return_pct"] == 4.0
    assert bbb_2b["net_win_rate_pct"] == 100.0

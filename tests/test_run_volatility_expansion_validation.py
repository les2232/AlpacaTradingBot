import pandas as pd

from run_volatility_expansion_validation import (
    ValidationRun,
    attach_signal_context,
    build_parameter_perturbation_specs,
    classify_validation,
    summarize_compression_quality,
    summarize_slice_validation,
)


def test_attach_signal_context_joins_on_symbol_and_entry_ts() -> None:
    closed = pd.DataFrame(
        [
            {"symbol": "AMD", "entry_ts": pd.Timestamp("2026-01-02T15:45:00Z"), "realized_pnl": 10.0},
        ]
    )
    signals = pd.DataFrame(
        [
            {
                "symbol": "AMD",
                "entry_ts": pd.Timestamp("2026-01-02T15:45:00Z"),
                "time_bucket": "morning",
                "trend_proxy": "trend",
                "volatility_regime": "high_vol",
                "month": "2026-01",
                "signal_hour": "09:00",
                "compression_quality": "strongest_compression",
                "bb_width": 0.01,
                "fwd_1b_net_pct": 0.2,
                "fwd_2b_net_pct": 0.3,
                "fwd_3b_net_pct": 0.4,
                "fwd_4b_net_pct": 0.5,
                "fwd_6b_net_pct": 0.55,
                "fwd_8b_net_pct": 0.6,
            }
        ]
    )

    merged = attach_signal_context(closed, signals, (1, 2, 3, 4, 6, 8))

    assert merged.iloc[0]["time_bucket"] == "morning"
    assert merged.iloc[0]["compression_quality"] == "strongest_compression"
    assert merged.iloc[0]["fwd_6b_net_pct"] == 0.55


def test_summarize_slice_validation_handles_empty_inputs() -> None:
    summary = summarize_slice_validation(pd.DataFrame(), pd.DataFrame(), group_cols=["symbol"], horizon_bars=4)

    assert summary.empty
    assert list(summary.columns) == [
        "symbol",
        "signal_count",
        "realized_trade_count",
        "avg_forward_return_pct",
        "realized_expectancy",
        "realized_win_rate",
    ]


def test_summarize_slice_validation_combines_signal_and_realized_metrics() -> None:
    signals = pd.DataFrame(
        [
            {"symbol": "AMD", "fwd_4b_gross_pct": 0.4, "fwd_4b_net_pct": 0.2},
            {"symbol": "AMD", "fwd_4b_gross_pct": 0.1, "fwd_4b_net_pct": -0.1},
            {"symbol": "JPM", "fwd_4b_gross_pct": 0.2, "fwd_4b_net_pct": 0.05},
        ]
    )
    closed = pd.DataFrame(
        [
            {"symbol": "AMD", "realized_pnl": 10.0},
            {"symbol": "AMD", "realized_pnl": -4.0},
            {"symbol": "JPM", "realized_pnl": 3.0},
        ]
    )

    summary = summarize_slice_validation(signals, closed, group_cols=["symbol"], horizon_bars=4)
    amd = summary[summary["symbol"] == "AMD"].iloc[0]
    jpm = summary[summary["symbol"] == "JPM"].iloc[0]

    assert amd["signal_count"] == 2
    assert amd["realized_trade_count"] == 2
    assert round(float(amd["avg_forward_return_pct"]), 6) == 0.05
    assert round(float(amd["realized_expectancy"]), 6) == 3.0
    assert round(float(amd["realized_win_rate"]), 6) == 50.0
    assert jpm["signal_count"] == 1
    assert jpm["realized_trade_count"] == 1


def test_summarize_compression_quality_orders_buckets() -> None:
    signals = pd.DataFrame(
        [
            {"compression_quality": "moderate_compression", "fwd_4b_gross_pct": 0.1, "fwd_4b_net_pct": 0.02},
            {"compression_quality": "strongest_compression", "fwd_4b_gross_pct": 0.3, "fwd_4b_net_pct": 0.10},
            {"compression_quality": "weak_compression", "fwd_4b_gross_pct": -0.1, "fwd_4b_net_pct": -0.05},
        ]
    )

    summary = summarize_compression_quality(signals, horizons=(4,))

    assert list(summary["compression_quality"]) == [
        "strongest_compression",
        "moderate_compression",
        "weak_compression",
    ]


def test_build_parameter_perturbation_specs_stays_local() -> None:
    specs = build_parameter_perturbation_specs(
        {
            "volatility_expansion_lookback_bars": 20,
            "volatility_expansion_entry_buffer_pct": 0.001,
            "volatility_expansion_hold_bars": 4,
            "volatility_expansion_max_atr_percentile": 35.0,
        },
        symbols=("AMD", "HON"),
    )

    names = {spec.name for spec in specs}

    assert "lookback=15" in names
    assert "lookback=20" in names
    assert "lookback=25" in names
    assert "entry_buffer=0.0005" in names
    assert "entry_buffer=0.0010" in names
    assert "entry_buffer=0.0015" in names
    assert "hold_bars=3" in names
    assert "hold_bars=4" in names
    assert "hold_bars=6" in names
    assert "max_atr_pct=25.0" in names
    assert "max_atr_pct=35.0" in names
    assert "max_atr_pct=45.0" in names


def test_classify_validation_rejects_when_raw_and_realized_are_negative() -> None:
    baseline_run = ValidationRun(
        name="baseline",
        strategy_mode="volatility_expansion",
        context=None,
        evaluations_df=pd.DataFrame(),
        signals_df=pd.DataFrame(
            [
                {"fwd_1b_gross_pct": -0.2, "fwd_1b_net_pct": -0.3, "fwd_2b_gross_pct": -0.1, "fwd_2b_net_pct": -0.2, "fwd_3b_gross_pct": -0.2, "fwd_3b_net_pct": -0.1, "fwd_4b_gross_pct": -0.1, "fwd_4b_net_pct": -0.2, "fwd_6b_gross_pct": -0.1, "fwd_6b_net_pct": -0.3, "fwd_8b_gross_pct": -0.2, "fwd_8b_net_pct": -0.4},
            ]
        ),
        backtest_result={"expectancy": -1.0, "trades": [{"side": "SELL"}]},
        closed_trades_df=pd.DataFrame([{"realized_pnl": -1.0}]),
    )

    label, reason = classify_validation(
        baseline_run=baseline_run,
        perturbation_df=pd.DataFrame([{"expectancy": -1.0}]),
        comparison_df=pd.DataFrame(
            [
                {"strategy_mode": "volatility_expansion", "realized_pnl": -10.0},
                {"strategy_mode": "momentum_breakout", "realized_pnl": -5.0},
                {"strategy_mode": "trend_pullback", "realized_pnl": -3.0},
            ]
        ),
    )

    assert label == "reject"
    assert "no positive raw continuation" in reason.lower()

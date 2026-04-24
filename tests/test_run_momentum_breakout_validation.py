import pandas as pd

from run_momentum_breakout_validation import (
    ScenarioSpec,
    ValidationRun,
    attach_signal_context,
    build_parameter_perturbation_specs,
    classify_validation,
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
                "fwd_1b_net_pct": 0.2,
                "fwd_2b_net_pct": 0.3,
                "fwd_3b_net_pct": 0.4,
                "fwd_4b_net_pct": 0.5,
                "fwd_8b_net_pct": 0.6,
            }
        ]
    )

    merged = attach_signal_context(closed, signals)

    assert merged.iloc[0]["time_bucket"] == "morning"
    assert merged.iloc[0]["trend_proxy"] == "trend"
    assert merged.iloc[0]["fwd_3b_net_pct"] == 0.4


def test_summarize_slice_validation_handles_empty_inputs() -> None:
    summary = summarize_slice_validation(pd.DataFrame(), pd.DataFrame(), group_cols=["symbol"], horizon_bars=3)

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
            {"symbol": "AMD", "fwd_3b_gross_pct": 0.4, "fwd_3b_net_pct": 0.2},
            {"symbol": "AMD", "fwd_3b_gross_pct": 0.1, "fwd_3b_net_pct": -0.1},
            {"symbol": "JPM", "fwd_3b_gross_pct": 0.2, "fwd_3b_net_pct": 0.05},
        ]
    )
    closed = pd.DataFrame(
        [
            {"symbol": "AMD", "realized_pnl": 10.0},
            {"symbol": "AMD", "realized_pnl": -4.0},
            {"symbol": "JPM", "realized_pnl": 3.0},
        ]
    )

    summary = summarize_slice_validation(signals, closed, group_cols=["symbol"], horizon_bars=3)
    amd = summary[summary["symbol"] == "AMD"].iloc[0]
    jpm = summary[summary["symbol"] == "JPM"].iloc[0]

    assert amd["signal_count"] == 2
    assert amd["realized_trade_count"] == 2
    assert round(float(amd["avg_forward_return_pct"]), 6) == 0.05
    assert round(float(amd["realized_expectancy"]), 6) == 3.0
    assert round(float(amd["realized_win_rate"]), 6) == 50.0
    assert jpm["signal_count"] == 1
    assert jpm["realized_trade_count"] == 1


def test_build_parameter_perturbation_specs_stays_local() -> None:
    specs = build_parameter_perturbation_specs(
        {
            "momentum_breakout_lookback_bars": 20,
            "momentum_breakout_entry_buffer_pct": 0.001,
            "momentum_breakout_hold_bars": 3,
            "momentum_breakout_min_adx": 20.0,
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
    assert "hold_bars=2" in names
    assert "hold_bars=3" in names
    assert "hold_bars=4" in names
    assert "min_adx=15.0" in names
    assert "min_adx=20.0" in names
    assert "min_adx=25.0" in names


def test_classify_validation_marks_weak_when_raw_exists_but_perturbations_fail() -> None:
    baseline_run = ValidationRun(
        name="baseline",
        strategy_mode="momentum_breakout",
        context=None,
        evaluations_df=pd.DataFrame(),
        signals_df=pd.DataFrame(
            [
                {"fwd_1b_gross_pct": 0.2, "fwd_1b_net_pct": -0.1, "fwd_2b_gross_pct": 0.3, "fwd_2b_net_pct": -0.05, "fwd_3b_gross_pct": 0.4, "fwd_3b_net_pct": 0.02, "fwd_4b_gross_pct": 0.1, "fwd_4b_net_pct": -0.02, "fwd_8b_gross_pct": 0.0, "fwd_8b_net_pct": -0.03},
            ]
        ),
        backtest_result={"expectancy": -1.0, "trades": [{"side": "SELL"}]},
        closed_trades_df=pd.DataFrame([{"realized_pnl": -1.0}]),
    )
    perturbation_df = pd.DataFrame(
        [
            {"expectancy": -1.0},
            {"expectancy": -0.5},
            {"expectancy": 0.0},
        ]
    )
    comparison_df = pd.DataFrame(
        [
            {"strategy_mode": "momentum_breakout", "realized_pnl": -10.0},
            {"strategy_mode": "trend_pullback", "realized_pnl": -20.0},
        ]
    )

    label, reason = classify_validation(
        baseline_run=baseline_run,
        perturbation_df=perturbation_df,
        comparison_df=comparison_df,
    )

    assert label == "weak"
    assert "raw continuation exists" in reason.lower()

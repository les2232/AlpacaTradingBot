import pandas as pd

from run_trend_pullback_robustness import (
    ScenarioRun,
    ScenarioSpec,
    _scenario_summary,
    build_month_slice_specs,
    build_parameter_perturbation_specs,
    build_symbol_robustness_specs,
    classify_fragility,
)


def test_build_month_slice_specs_handles_empty_input() -> None:
    result = build_month_slice_specs(pd.DataFrame(), runtime_overrides={}, symbols=("AMD", "HON"))
    assert result == []


def test_build_month_slice_specs_builds_month_windows() -> None:
    evaluations = pd.DataFrame({"month": ["2026-01", "2026-02", "2026-01"]})
    result = build_month_slice_specs(evaluations, runtime_overrides={}, symbols=("AMD", "HON"))

    assert [spec.name for spec in result] == ["month:2026-01", "month:2026-02"]
    assert result[0].start_date == "2026-01-01"
    assert result[0].end_date == "2026-01-31"


def test_build_parameter_perturbation_specs_stays_local() -> None:
    specs = build_parameter_perturbation_specs(
        {
            "trend_pullback_hold_bars": 4,
            "trend_pullback_entry_threshold": 0.0015,
            "trend_pullback_min_adx": 20.0,
        },
        symbols=("AMD", "HON", "C", "JPM"),
    )

    names = {spec.name for spec in specs}
    assert {"hold_bars=3", "hold_bars=4", "hold_bars=5"} <= names
    assert {"entry_threshold=0.0010", "entry_threshold=0.0015", "entry_threshold=0.0020"} <= names
    assert {"min_adx=15.0", "min_adx=20.0", "min_adx=25.0"} <= names


def test_build_symbol_robustness_specs_includes_leave_one_out_and_plus() -> None:
    specs = build_symbol_robustness_specs(
        ("AMD", "HON", "C", "JPM"),
        runtime_overrides={},
        adjacent_symbols=("ABBV", "AMD"),
    )

    names = [spec.name for spec in specs]
    assert names[0] == "baseline"
    assert "leave_out:AMD" in names
    assert "leave_out:HON" in names
    assert "leave_out:C" in names
    assert "leave_out:JPM" in names
    assert "plus:ABBV" in names
    assert "plus:AMD" not in names


def test_classify_fragility_handles_no_trades() -> None:
    label, reason = classify_fragility(
        baseline_summary={"filled_trades": 0, "expectancy": 0.0, "total_pnl": 0.0},
        time_slice_df=pd.DataFrame(),
        param_df=pd.DataFrame(),
        symbol_df=pd.DataFrame(),
        regime_tables={},
    )

    assert label == "not ready"
    assert "did not produce closed trades" in reason


def test_classify_fragility_marks_positive_but_narrow_setup() -> None:
    label, reason = classify_fragility(
        baseline_summary={"filled_trades": 10, "expectancy": 2.0, "total_pnl": 20.0},
        time_slice_df=pd.DataFrame([{"expectancy": 1.0}, {"expectancy": -1.0}, {"expectancy": 2.0}]),
        param_df=pd.DataFrame([{"expectancy": 1.0}, {"expectancy": 1.5}, {"expectancy": 0.5}]),
        symbol_df=pd.DataFrame(
            [
                {"scenario": "baseline", "expectancy": 2.0},
                {"scenario": "leave_out:AMD", "expectancy": 1.0},
                {"scenario": "leave_out:HON", "expectancy": 0.5},
                {"scenario": "leave_out:C", "expectancy": 0.2},
                {"scenario": "leave_out:JPM", "expectancy": 0.1},
            ]
        ),
        regime_tables={
            "trend_proxy": pd.DataFrame([{"trend_proxy": "trend", "avg_net_expectancy_pct": 0.1}]),
            "time_bucket": pd.DataFrame([{"time_bucket": "midday", "avg_net_expectancy_pct": -0.05}]),
        },
    )

    assert label == "promising but narrow"
    assert "trend-dependent" in reason


def test_scenario_summary_falls_back_to_available_horizon() -> None:
    run = ScenarioRun(
        spec=ScenarioSpec(name="hold_bars=3", section="params", symbols=("AMD",), runtime_overrides={}),
        context=None,
        evaluations_df=pd.DataFrame(),
        signals_df=pd.DataFrame(
            [
                {"fwd_3b_gross_pct": 0.1, "fwd_3b_net_pct": 0.05},
                {"fwd_3b_gross_pct": -0.2, "fwd_3b_net_pct": -0.1},
            ]
        ),
        backtest_result={"trades": [{"side": "SELL"}], "realized_pnl": 1.0, "expectancy": 1.0, "win_rate": 100.0, "profit_factor": 1.5, "max_drawdown_pct": 0.0},
    )

    summary = _scenario_summary(run, horizon_bars=4)
    assert summary["raw_fixed_horizon_bars"] == 3
    assert summary["raw_fixed_sample_count"] == 2

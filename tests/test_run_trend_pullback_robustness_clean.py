import pandas as pd

from run_trend_pullback_robustness_clean import (
    build_clean_parameter_specs,
    classify_clean_robustness,
    compute_symbol_concentration,
)


def test_build_clean_parameter_specs_stays_local_to_long_horizon_baseline() -> None:
    specs = build_clean_parameter_specs(
        {
            "trend_pullback_entry_threshold": 0.0015,
            "trend_pullback_min_adx": 20.0,
        },
        symbols=("AMD", "HON", "C", "JPM"),
        baseline_hold_bars=20,
    )

    names = {spec.name for spec in specs}
    assert {"hold_bars=15", "hold_bars=20", "hold_bars=30"} <= names
    assert {"entry_threshold=0.0010", "entry_threshold=0.0015", "entry_threshold=0.0020"} <= names
    assert {"min_adx=15.0", "min_adx=20.0", "min_adx=25.0"} <= names


def test_compute_symbol_concentration_handles_positive_and_empty_cases() -> None:
    df = pd.DataFrame(
        [
            {"symbol": "AMD", "realized_pnl": 30.0},
            {"symbol": "JPM", "realized_pnl": 10.0},
            {"symbol": "HON", "realized_pnl": -5.0},
        ]
    )
    result = compute_symbol_concentration(df)
    assert result["top_symbol"] == "AMD"
    assert round(result["top_positive_share"], 4) == 0.75

    empty_result = compute_symbol_concentration(pd.DataFrame())
    assert empty_result["top_symbol"] is None
    assert empty_result["top_positive_share"] == 0.0


def test_classify_clean_robustness_marks_narrow_positive_setup() -> None:
    label, reason = classify_clean_robustness(
        baseline_summary={"trade_count": 40, "expectancy": 1.2, "best_raw_net_expectancy_pct": 0.20},
        time_df=pd.DataFrame([{"expectancy": 1.0}, {"expectancy": 0.5}, {"expectancy": -0.2}]),
        param_df=pd.DataFrame([{"expectancy": 1.2}, {"expectancy": 0.7}, {"expectancy": 0.3}]),
        symbol_df=pd.DataFrame(
            [
                {"scenario": "baseline", "expectancy": 1.2},
                {"scenario": "leave_out:AMD", "expectancy": 0.8},
                {"scenario": "leave_out:HON", "expectancy": 0.4},
                {"scenario": "leave_out:C", "expectancy": -0.1},
                {"scenario": "leave_out:JPM", "expectancy": 0.2},
            ]
        ),
        symbol_concentration={"top_symbol": "AMD", "top_positive_share": 0.55},
        path_sanity=pd.DataFrame([{"avg_missed_opportunity_pct": 0.05, "giveback_frac_pct": 45.0}]),
    )

    assert label == "promising but narrow"
    assert "concentrated or sensitive" in reason


def test_classify_clean_robustness_rejects_non_positive_baseline() -> None:
    label, reason = classify_clean_robustness(
        baseline_summary={"trade_count": 25, "expectancy": -0.5, "best_raw_net_expectancy_pct": 0.05},
        time_df=pd.DataFrame(),
        param_df=pd.DataFrame(),
        symbol_df=pd.DataFrame(),
        symbol_concentration={"top_symbol": None, "top_positive_share": 0.0},
        path_sanity=pd.DataFrame(),
    )

    assert label == "reject"
    assert "not convincingly positive" in reason

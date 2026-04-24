import pandas as pd

from run_trend_pullback_oos_validation import (
    build_contiguous_window_specs,
    build_month_window_specs,
    interpret_oos_results,
)


def test_build_month_window_specs_uses_existing_months() -> None:
    evaluations = pd.DataFrame({"month": ["2026-01", "2026-02", "2026-01"]})
    specs = build_month_window_specs(evaluations)
    assert [spec.label for spec in specs] == ["month:2026-01", "month:2026-02"]
    assert specs[0].start_date == "2026-01-01"
    assert specs[0].end_date == "2026-01-31"


def test_build_contiguous_window_specs_splits_dates_into_three_segments() -> None:
    evaluations = pd.DataFrame(
        {
            "date": [
                "2026-01-01",
                "2026-01-02",
                "2026-01-03",
                "2026-01-04",
                "2026-01-05",
                "2026-01-06",
            ]
        }
    )
    specs = build_contiguous_window_specs(evaluations)
    assert [spec.label for spec in specs] == ["segment:early", "segment:middle", "segment:late"]
    assert specs[0].start_date == "2026-01-01"
    assert specs[-1].end_date == "2026-01-06"


def test_interpret_oos_results_flags_collapse() -> None:
    summary = pd.DataFrame(
        [
            {"period_label": "month:2026-01", "trade_count": 5, "expectancy": -1.0, "pnl": -10.0, "top_positive_share": 0.0},
            {"period_label": "month:2026-02", "trade_count": 0, "expectancy": 0.0, "pnl": 0.0, "top_positive_share": 0.0},
            {"period_label": "month:2026-03", "trade_count": 4, "expectancy": -0.5, "pnl": -4.0, "top_positive_share": 0.0},
        ]
    )
    label, reason, diagnostics = interpret_oos_results(summary)
    assert label == "demote"
    assert "collapses" in reason
    assert diagnostics["zero_trade_periods"] == ["month:2026-02"]


def test_interpret_oos_results_allows_stable_positive_case() -> None:
    summary = pd.DataFrame(
        [
            {"period_label": "month:2026-01", "trade_count": 6, "expectancy": 1.0, "pnl": 10.0, "top_positive_share": 0.30},
            {"period_label": "month:2026-02", "trade_count": 5, "expectancy": 0.8, "pnl": 8.0, "top_positive_share": 0.25},
            {"period_label": "month:2026-03", "trade_count": 4, "expectancy": 0.6, "pnl": 6.0, "top_positive_share": 0.35},
        ]
    )
    label, reason, diagnostics = interpret_oos_results(summary)
    assert label == "stable enough for deeper research"
    assert diagnostics["negative_expectancy_periods"] == []

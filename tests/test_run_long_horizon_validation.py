import pandas as pd

from run_long_horizon_validation import (
    build_hold_horizon_specs,
    choose_representative_horizon,
    classify_long_horizon,
    summarize_hold_ladder,
    _parse_horizon_list,
)


def test_parse_horizon_list_sorts_and_dedupes() -> None:
    assert _parse_horizon_list("20,4,10,4,30") == (4, 10, 20, 30)


def test_build_hold_horizon_specs_uses_requested_ladder() -> None:
    specs = build_hold_horizon_specs("volatility_expansion", (4, 8, 15))

    assert specs == [
        {"hold_bars": 4, "runtime_overrides": {"volatility_expansion_hold_bars": 4}},
        {"hold_bars": 8, "runtime_overrides": {"volatility_expansion_hold_bars": 8}},
        {"hold_bars": 15, "runtime_overrides": {"volatility_expansion_hold_bars": 15}},
    ]


def test_choose_representative_horizon_prefers_twenty() -> None:
    assert choose_representative_horizon((4, 8, 10, 20, 30)) == 20
    assert choose_representative_horizon((4, 8, 10, 30)) == 30


def test_summarize_hold_ladder_handles_empty_runs() -> None:
    summary = summarize_hold_ladder([])

    assert summary.empty
    assert list(summary.columns) == [
        "hold_bars",
        "signal_count",
        "trade_count",
        "realized_pnl",
        "expectancy",
        "win_rate",
        "profit_factor",
        "max_drawdown_pct",
        "avg_holding_period",
    ]


def test_classify_long_horizon_rejects_when_raw_and_realized_stay_negative() -> None:
    raw_overall = pd.DataFrame(
        [
            {"horizon_bars": 4, "avg_net_expectancy_pct": -0.2},
            {"horizon_bars": 10, "avg_net_expectancy_pct": -0.1},
            {"horizon_bars": 20, "avg_net_expectancy_pct": -0.05},
        ]
    )
    hold_ladder = pd.DataFrame(
        [
            {"hold_bars": 4, "expectancy": -2.0},
            {"hold_bars": 10, "expectancy": -1.0},
        ]
    )

    label, reason = classify_long_horizon(raw_overall=raw_overall, hold_ladder=hold_ladder)

    assert label == "reject"
    assert "stay negative" in reason.lower()


def test_classify_long_horizon_flags_research_only_when_raw_improves_but_realized_does_not() -> None:
    raw_overall = pd.DataFrame(
        [
            {"horizon_bars": 4, "avg_net_expectancy_pct": -0.2},
            {"horizon_bars": 20, "avg_net_expectancy_pct": 0.15},
        ]
    )
    hold_ladder = pd.DataFrame(
        [
            {"hold_bars": 4, "expectancy": -2.0},
            {"hold_bars": 20, "expectancy": -0.5},
        ]
    )

    label, reason = classify_long_horizon(raw_overall=raw_overall, hold_ladder=hold_ladder)

    assert label == "keep as research-only"
    assert "realized expectancy stayed non-positive" in reason.lower()

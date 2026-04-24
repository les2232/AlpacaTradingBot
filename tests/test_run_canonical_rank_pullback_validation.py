import sys

import pandas as pd

from research.run_canonical_rank_pullback_validation import parse_args
from run_canonical_rank_pullback_validation import (
    _build_comparison_frame,
    _build_family_ranking_frame,
    _parse_floats,
    _parse_ints,
    _summarize_family_win_counts,
)


def test_parse_ints_and_floats() -> None:
    assert _parse_ints("10,5,10") == (5, 10)
    assert _parse_floats("0.3,0.2,0.3") == (0.2, 0.3)


def test_build_comparison_frame_for_grouped_summary() -> None:
    summary_df = pd.DataFrame(
        [
            {"window_idx": 1, "family": "baseline", "expectancy": 1.0, "profit_factor": 1.2, "trade_count": 10, "max_drawdown_pct": 5.0},
            {"window_idx": 1, "family": "pullback", "expectancy": 2.0, "profit_factor": 1.5, "trade_count": 8, "max_drawdown_pct": 4.0},
        ]
    )
    comparison_df = _build_comparison_frame(summary_df, index_col="window_idx")
    expectancy_row = comparison_df[comparison_df["metric"] == "expectancy"].iloc[0]
    assert expectancy_row["window_idx"] == 1
    assert expectancy_row["pullback_minus_baseline"] == 1.0


def test_build_family_ranking_frame_orders_by_expectancy_then_profit_factor() -> None:
    summary_df = pd.DataFrame(
        [
            {"family": "baseline", "expectancy": 1.0, "profit_factor": 1.1, "trade_count": 12},
            {"family": "ranking_only", "expectancy": 2.0, "profit_factor": 1.0, "trade_count": 9},
            {"family": "pullback", "expectancy": 2.0, "profit_factor": 1.4, "trade_count": 8},
        ]
    )
    ranked_df = _build_family_ranking_frame(summary_df)
    assert list(ranked_df["family"]) == ["pullback", "ranking_only", "baseline"]
    assert list(ranked_df["rank"]) == [1, 2, 3]


def test_summarize_family_win_counts_counts_fold_winners() -> None:
    ranked_df = pd.DataFrame(
        [
            {"window_idx": 1, "family": "pullback", "rank": 1},
            {"window_idx": 1, "family": "baseline", "rank": 2},
            {"window_idx": 2, "family": "ranking_only", "rank": 1},
            {"window_idx": 2, "family": "pullback", "rank": 2},
            {"window_idx": 3, "family": "pullback", "rank": 1},
        ]
    )
    win_counts_df = _summarize_family_win_counts(ranked_df, index_col="window_idx")
    records = {row["family"]: row["win_count"] for _, row in win_counts_df.iterrows()}
    assert records == {"pullback": 2, "ranking_only": 1}


def test_parse_args_accepts_early_exit_flag(monkeypatch) -> None:
    monkeypatch.setattr(sys, "argv", ["prog", "--use-early-no-follow-through-exit"])
    args = parse_args()
    assert args.use_early_no_follow_through_exit is True


def test_parse_args_accepts_recent_follow_through_filter_flag(monkeypatch) -> None:
    monkeypatch.setattr(sys, "argv", ["prog", "--use-recent-follow-through-filter"])
    args = parse_args()
    assert args.use_recent_follow_through_filter is True

from unittest.mock import patch

import pandas as pd

from run_backtest_experiments import (
    build_per_symbol_comparison,
    build_summary_dataframe,
)


def test_build_summary_dataframe_sorts_best_return_first() -> None:
    rows = [
        {"run": "baseline", "total_return_pct": -2.0, "sharpe_ratio": -0.2, "profit_factor": 0.97, "max_drawdown_pct": 8.8, "source_csv": "a.csv"},
        {"run": "improved", "total_return_pct": 1.5, "sharpe_ratio": 0.4, "profit_factor": 1.08, "max_drawdown_pct": 7.5, "source_csv": "b.csv"},
    ]

    df = build_summary_dataframe(rows)

    assert list(df["run"]) == ["improved", "baseline"]


def test_build_per_symbol_comparison_marks_best_and_worst_runs() -> None:
    baseline_df = pd.DataFrame(
        [
            {"symbol": "AAPL", "total_return_pct": 5.0},
            {"symbol": "MSFT", "total_return_pct": -3.0},
        ]
    )
    filtered_df = pd.DataFrame(
        [
            {"symbol": "AAPL", "total_return_pct": 8.0},
            {"symbol": "MSFT", "total_return_pct": -1.0},
        ]
    )

    rows = [
        {"run": "baseline", "source_csv": "baseline.csv"},
        {"run": "filtered", "source_csv": "filtered.csv"},
    ]

    with patch("run_backtest_experiments.Path.exists", return_value=True), patch(
        "run_backtest_experiments.pd.read_csv",
        side_effect=[baseline_df, filtered_df],
    ):
        comparison = build_per_symbol_comparison(rows)

    aapl_row = comparison[comparison["symbol"] == "AAPL"].iloc[0]
    msft_row = comparison[comparison["symbol"] == "MSFT"].iloc[0]

    assert aapl_row["best_run"] == "filtered"
    assert aapl_row["worst_run"] == "baseline"
    assert msft_row["best_run"] == "filtered"

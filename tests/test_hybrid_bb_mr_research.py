import pandas as pd

from run_hybrid_bb_mr_research import (
    _parse_bool_csv_values,
    _parse_csv_values,
    add_baseline_comparison_columns,
    add_usefulness_columns,
    build_hybrid_grid,
    build_ranked_hybrid_dataframe,
    flatten_hybrid_branch_stats,
)


def test_build_hybrid_grid_deduplicates_volume_multiplier_when_volume_confirm_is_off() -> None:
    grid = build_hybrid_grid(
        squeeze_quantiles=(0.2,),
        use_volume_confirm_values=(True, False),
        volume_mult_values=(1.0, 1.2, 1.5),
        slope_lookbacks=(1, 3),
    )

    assert len(grid) == 8
    off_specs = [spec for spec in grid if spec.bb_use_volume_confirm is False]
    assert len(off_specs) == 2
    assert all(spec.bb_volume_mult is None for spec in off_specs)


def test_parse_csv_helpers_support_extended_grid_inputs() -> None:
    assert _parse_csv_values("0.2,0.5,0.6", float) == (0.2, 0.5, 0.6)
    assert _parse_csv_values("1,3,5", int) == (1, 3, 5)
    assert _parse_bool_csv_values("true,false") == (True, False)


def test_flatten_hybrid_branch_stats_marks_no_bb_participation() -> None:
    stats = flatten_hybrid_branch_stats(
        {
            "hybrid_branch_stats": {
                "mean_reversion": {"total_trades": 7, "win_rate": 71.4, "realized_pnl": 5.0},
            }
        }
    )

    assert stats["mr_branch_trades"] == 7
    assert stats["bb_branch_trades"] == 0
    assert stats["bb_branch_trade_share_pct"] == 0.0
    assert stats["bb_branch_participation"] == "none"


def test_usefulness_score_rewards_hybrid_rows_with_bb_participation_and_better_pf() -> None:
    df = pd.DataFrame(
        [
            {
                "run_name": "mean_reversion_reference",
                "strategy_mode": "mean_reversion",
                "realized_pnl": 10.0,
                "profit_factor": 1.1,
                "sharpe_ratio": 0.8,
                "max_drawdown_pct": 5.0,
                "total_trades": 20,
                "bb_branch_trade_share_pct": 0.0,
                "bb_branch_trades": 0,
                "branch_balance_score": 0.0,
            },
            {
                "run_name": "hybrid_active",
                "strategy_mode": "hybrid_bb_mr",
                "realized_pnl": 15.0,
                "profit_factor": 1.3,
                "sharpe_ratio": 1.0,
                "max_drawdown_pct": 5.5,
                "total_trades": 18,
                "bb_branch_trade_share_pct": 20.0,
                "bb_branch_trades": 4,
                "branch_balance_score": 0.4,
            },
            {
                "run_name": "hybrid_dead_bb",
                "strategy_mode": "hybrid_bb_mr",
                "realized_pnl": 15.0,
                "profit_factor": 1.3,
                "sharpe_ratio": 1.0,
                "max_drawdown_pct": 5.5,
                "total_trades": 18,
                "bb_branch_trade_share_pct": 0.0,
                "bb_branch_trades": 0,
                "branch_balance_score": 0.0,
            },
        ]
    )

    scored = add_usefulness_columns(add_baseline_comparison_columns(df))
    ranked = build_ranked_hybrid_dataframe(scored)

    assert ranked.iloc[0]["run_name"] == "hybrid_active"
    assert ranked.iloc[0]["usefulness_score"] > ranked.iloc[1]["usefulness_score"]

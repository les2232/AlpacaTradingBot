import pandas as pd

from run_trend_pullback_exit_capture_clean import (
    build_exit_variants,
    classify_exit_capture,
    summarize_exit_variant,
    ExitVariant,
)


def test_build_exit_variants_includes_clean_baseline_and_hybrids() -> None:
    variants = build_exit_variants(
        baseline_hold_bars=20,
        tp_targets=(0.005, 0.0075, 0.01),
        sections={"baseline", "hybrid"},
    )
    names = {variant.name for variant in variants}
    assert {"fixed_20_next_open", "fixed_15_next_open", "fixed_25_next_open", "fixed_20_bar_close_control"} <= names
    assert "hybrid_tp_0.50pct_hold20_bar_close" in names
    assert "hybrid_tp_0.75pct_hold20_bar_close" in names
    assert "hybrid_tp_1.00pct_hold20_bar_close" in names


def test_summarize_exit_variant_handles_zero_trade_case() -> None:
    summary = summarize_exit_variant(
        variant=ExitVariant(
            name="fixed_20_next_open",
            section="baseline",
            exit_style="fixed_bars",
            hold_bars=20,
            take_profit_pct=0.0,
            research_exit_fill="next_open",
        ),
        backtest_result={"trades": [], "realized_pnl": 0.0, "expectancy": 0.0, "win_rate": 0.0, "profit_factor": 0.0, "max_drawdown_pct": 0.0},
        signal_count=201,
        signal_paths_df=pd.DataFrame(),
        slippage=0.05,
    )

    assert summary["signal_count"] == 201
    assert summary["trade_count"] == 0
    assert summary["avg_missed_opportunity_pct"] == 0.0
    assert summary["materially_below_best_frac"] == 0.0


def test_classify_exit_capture_requires_more_than_optimistic_fill() -> None:
    df = pd.DataFrame(
        [
            {
                "variant": "fixed_20_next_open",
                "section": "baseline",
                "research_exit_fill": "next_open",
                "expectancy": 1.2,
                "total_pnl": 60.0,
                "avg_missed_opportunity_pct": 1.3,
                "materially_below_best_frac": 96.0,
            },
            {
                "variant": "hybrid_tp_0.75pct_hold20_bar_close",
                "section": "hybrid",
                "research_exit_fill": "bar_close",
                "expectancy": 1.5,
                "total_pnl": 70.0,
                "avg_missed_opportunity_pct": 1.1,
                "materially_below_best_frac": 90.0,
            },
        ]
    )
    diagnosis, recommendation = classify_exit_capture(df)
    assert "optimistic research-only fill assumptions" in diagnosis
    assert recommendation == "keep as research-only"


def test_classify_exit_capture_promotes_material_improvement() -> None:
    df = pd.DataFrame(
        [
            {
                "variant": "fixed_20_next_open",
                "section": "baseline",
                "research_exit_fill": "next_open",
                "expectancy": 1.0,
                "total_pnl": 50.0,
                "avg_missed_opportunity_pct": 1.2,
                "materially_below_best_frac": 80.0,
            },
            {
                "variant": "hybrid_tp_0.75pct_hold20_bar_close",
                "section": "hybrid",
                "research_exit_fill": "bar_close",
                "expectancy": 2.0,
                "total_pnl": 90.0,
                "avg_missed_opportunity_pct": 0.7,
                "materially_below_best_frac": 60.0,
            },
        ]
    )
    diagnosis, recommendation = classify_exit_capture(df)
    assert "materially improved capture" in diagnosis
    assert recommendation == "continue with deeper out-of-sample validation"

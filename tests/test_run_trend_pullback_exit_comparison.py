import pandas as pd

from backtest_runner import (
    _resolve_bar_close_exit_fill_price,
    _trend_pullback_should_exit_on_close,
)
from run_trend_pullback_exit_comparison import (
    ExitVariant,
    build_exit_variants,
    diagnose_exit_results,
    summarize_exit_variant,
)


def test_build_exit_variants_includes_expected_baselines() -> None:
    variants = build_exit_variants(0.0025)
    names = {variant.name for variant in variants}
    assert {"fixed_3_next_open", "fixed_4_next_open", "fixed_3_bar_close", "fixed_4_bar_close"} <= names
    assert any(variant.exit_style == "take_profit" for variant in variants)
    assert any(variant.exit_style == "hybrid_tp_or_time" for variant in variants)


def test_trend_pullback_should_exit_on_close_only_for_research_mode() -> None:
    assert _trend_pullback_should_exit_on_close("trend_pullback", "bar_close")
    assert not _trend_pullback_should_exit_on_close("trend_pullback", "next_open")
    assert not _trend_pullback_should_exit_on_close("mean_reversion", "bar_close")


def test_resolve_bar_close_exit_fill_uses_target_hint() -> None:
    assert _resolve_bar_close_exit_fill_price(row_close=100.0, slippage=0.05, exit_price_hint=100.25) == 100.2
    assert _resolve_bar_close_exit_fill_price(row_close=100.0, slippage=0.05, exit_price_hint=None) == 99.95


def test_summarize_exit_variant_handles_zero_trades() -> None:
    summary = summarize_exit_variant(
        variant=ExitVariant(
            name="fixed_3_next_open",
            exit_style="fixed_bars",
            hold_bars=3,
            take_profit_pct=0.0,
            research_exit_fill="next_open",
        ),
        backtest_result={"trades": [], "realized_pnl": 0.0, "expectancy": 0.0, "win_rate": 0.0, "profit_factor": 0.0, "max_drawdown_pct": 0.0},
        signal_paths_df=pd.DataFrame(),
        slippage=0.05,
    )

    assert summary["trade_count"] == 0
    assert summary["avg_gave_back_pct"] == 0.0


def test_diagnose_exit_results_handles_no_positive_variants() -> None:
    diagnosis, recommendation = diagnose_exit_results(
        pd.DataFrame(
            [
                {"variant": "fixed_3_next_open", "total_pnl": -5.0, "expectancy": -0.1},
                {"variant": "fixed_4_next_open", "total_pnl": -2.0, "expectancy": -0.05},
            ]
        )
    )

    assert "least bad" in diagnosis
    assert recommendation == "keep as research-only because edge is still too small"

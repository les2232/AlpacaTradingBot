from types import SimpleNamespace

import pandas as pd

from run_edge_diagnostics import (
    DiagnosticRun,
    build_symbol_quality_ranking,
    compare_variant_parameters,
    summarize_fixed_horizon_metrics,
    summarize_variant_comparison,
)


def test_summarize_fixed_horizon_metrics_handles_empty_input() -> None:
    result = summarize_fixed_horizon_metrics(pd.DataFrame(), group_cols=["symbol"], horizons=(1, 2, 4))
    assert list(result.columns) == [
        "symbol",
        "horizon_bars",
        "sample_count",
        "avg_gross_return_pct",
        "median_gross_return_pct",
        "gross_win_rate_pct",
        "avg_net_expectancy_pct",
        "median_net_return_pct",
        "net_win_rate_pct",
        "net_profit_factor",
    ]
    assert result.empty


def test_summarize_fixed_horizon_metrics_computes_profit_factor_and_win_rate() -> None:
    signals = pd.DataFrame(
        [
            {"symbol": "AAA", "fwd_1b_gross_pct": 1.0, "fwd_1b_net_pct": 0.5},
            {"symbol": "AAA", "fwd_1b_gross_pct": -0.5, "fwd_1b_net_pct": -0.25},
            {"symbol": "AAA", "fwd_1b_gross_pct": 0.0, "fwd_1b_net_pct": 0.0},
        ]
    )
    result = summarize_fixed_horizon_metrics(signals, group_cols=["symbol"], horizons=(1,))
    row = result.iloc[0]
    assert row["sample_count"] == 3
    assert row["gross_win_rate_pct"] == (1 / 3) * 100
    assert row["avg_net_expectancy_pct"] == (0.5 - 0.25 + 0.0) / 3
    assert row["net_profit_factor"] == 2.0


def test_build_symbol_quality_ranking_merges_realized_metrics() -> None:
    fixed_horizon = pd.DataFrame(
        [
            {"symbol": "AAA", "horizon_bars": 1, "avg_net_expectancy_pct": 0.1, "avg_gross_return_pct": 0.2, "net_profit_factor": 1.5, "net_win_rate_pct": 60.0},
            {"symbol": "AAA", "horizon_bars": 2, "avg_net_expectancy_pct": 0.3, "avg_gross_return_pct": 0.4, "net_profit_factor": 1.8, "net_win_rate_pct": 65.0},
            {"symbol": "BBB", "horizon_bars": 1, "avg_net_expectancy_pct": -0.2, "avg_gross_return_pct": -0.1, "net_profit_factor": 0.7, "net_win_rate_pct": 40.0},
        ]
    )
    realized = pd.DataFrame(
        [
            {"symbol": "AAA", "realized_pnl": 10.0, "expectancy": 1.0, "trades": 5},
            {"symbol": "BBB", "realized_pnl": -5.0, "expectancy": -0.5, "trades": 4},
        ]
    )
    result = build_symbol_quality_ranking(fixed_horizon, realized)
    aaa = result[result["symbol"] == "AAA"].iloc[0]
    assert aaa["best_horizon_bars"] == 2
    assert aaa["best_net_expectancy_pct"] == 0.3
    assert aaa["realized_pnl"] == 10.0


def test_compare_variant_parameters_and_summary() -> None:
    live_context = SimpleNamespace(backtest_kwargs={"vwap_z_entry_threshold": 1.5, "min_atr_percentile": 20.0})
    explicit_context = SimpleNamespace(backtest_kwargs={"vwap_z_entry_threshold": 0.0, "min_atr_percentile": 0.0})
    runs = [
        DiagnosticRun(
            variant="live_effective",
            context=live_context,
            evaluations_df=pd.DataFrame(),
            signals_df=pd.DataFrame([{"fwd_1b_gross_pct": 0.1, "fwd_1b_net_pct": -0.1}]),
            backtest_result={"realized_pnl": -10.0, "expectancy": -1.0, "win_rate": 40.0, "total_trades": 3},
            realized_per_symbol_df=pd.DataFrame(),
        ),
        DiagnosticRun(
            variant="explicit_config",
            context=explicit_context,
            evaluations_df=pd.DataFrame(),
            signals_df=pd.DataFrame([{"fwd_1b_gross_pct": 0.2, "fwd_1b_net_pct": 0.05}]),
            backtest_result={"realized_pnl": 5.0, "expectancy": 0.5, "win_rate": 60.0, "total_trades": 3},
            realized_per_symbol_df=pd.DataFrame(),
        ),
    ]
    diff_df = compare_variant_parameters(runs)
    assert set(diff_df["field"]) == {"min_atr_percentile", "vwap_z_entry_threshold"}

    summary_df = summarize_variant_comparison(runs, horizons=(1,))
    explicit = summary_df[summary_df["variant"] == "explicit_config"].iloc[0]
    assert explicit["best_fixed_net_expectancy_pct"] == 0.05
    assert explicit["realized_pnl"] == 5.0

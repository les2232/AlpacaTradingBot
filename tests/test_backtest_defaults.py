"""Smoke tests for BacktestConfig and run_backtest default assumptions."""
from pathlib import Path

from backtest_runner import BacktestConfig, _summarize_hybrid_branch_stats, run_backtest
import inspect


def test_backtest_config_default_commission():
    """BacktestConfig should default to a non-zero commission."""
    cfg = BacktestConfig(dataset_path=Path("dummy"))
    assert cfg.commission == 0.01, (
        f"Expected default commission 0.01, got {cfg.commission}. "
        "Zero-cost defaults produce unrealistically optimistic backtest results."
    )


def test_backtest_config_default_slippage():
    """BacktestConfig should default to a non-zero slippage."""
    cfg = BacktestConfig(dataset_path=Path("dummy"))
    assert cfg.slippage == 0.05, (
        f"Expected default slippage 0.05, got {cfg.slippage}. "
        "Zero slippage defaults produce unrealistically optimistic backtest results."
    )


def test_run_backtest_default_commission():
    """run_backtest function signature should default to non-zero commission."""
    sig = inspect.signature(run_backtest)
    default = sig.parameters["commission"].default
    assert default == 0.01, f"run_backtest commission default should be 0.01, got {default}"


def test_run_backtest_default_slippage():
    """run_backtest function signature should default to non-zero slippage."""
    sig = inspect.signature(run_backtest)
    default = sig.parameters["slippage"].default
    assert default == 0.05, f"run_backtest slippage default should be 0.05, got {default}"


def test_backtest_config_default_sma_stop_pct():
    """BacktestConfig should expose sma_stop_pct with a disabled-by-default value."""
    cfg = BacktestConfig(dataset_path=Path("dummy"))
    assert cfg.sma_stop_pct == 0.0


def test_run_backtest_default_sma_stop_pct():
    """run_backtest should expose sma_stop_pct in its public signature."""
    sig = inspect.signature(run_backtest)
    default = sig.parameters["sma_stop_pct"].default
    assert default == 0.0, f"run_backtest sma_stop_pct default should be 0.0, got {default}"


def test_backtest_config_exposes_bollinger_squeeze_defaults():
    cfg = BacktestConfig(dataset_path=Path("dummy"))
    assert cfg.bb_period == 20
    assert cfg.bb_stddev_mult == 2.0
    assert cfg.bb_width_lookback == 100
    assert cfg.bb_squeeze_quantile == 0.20
    assert cfg.bb_slope_lookback == 3
    assert cfg.bb_use_volume_confirm is True
    assert cfg.bb_volume_mult == 1.2
    assert cfg.bb_breakout_buffer_pct == 0.0
    assert cfg.bb_min_mid_slope == 0.0
    assert cfg.bb_trend_filter is False
    assert cfg.bb_exit_mode == "middle_band"


def test_run_backtest_signature_exposes_bollinger_squeeze_defaults():
    sig = inspect.signature(run_backtest)
    assert sig.parameters["bb_period"].default == 20
    assert sig.parameters["bb_stddev_mult"].default == 2.0
    assert sig.parameters["bb_width_lookback"].default == 100
    assert sig.parameters["bb_squeeze_quantile"].default == 0.20
    assert sig.parameters["bb_slope_lookback"].default == 3
    assert sig.parameters["bb_use_volume_confirm"].default is True
    assert sig.parameters["bb_volume_mult"].default == 1.2
    assert sig.parameters["bb_breakout_buffer_pct"].default == 0.0
    assert sig.parameters["bb_min_mid_slope"].default == 0.0
    assert sig.parameters["bb_trend_filter"].default is False
    assert sig.parameters["bb_exit_mode"].default == "middle_band"


def test_backtest_config_exposes_trend_pullback_defaults():
    cfg = BacktestConfig(dataset_path=Path("dummy"))
    assert cfg.trend_pullback_min_adx == 20.0
    assert cfg.trend_pullback_min_slope == 0.0
    assert cfg.trend_pullback_entry_threshold == 0.0015
    assert cfg.trend_pullback_min_atr_percentile == 20.0
    assert cfg.trend_pullback_max_atr_percentile == 0.0
    assert cfg.trend_pullback_exit_style == "fixed_bars"
    assert cfg.trend_pullback_hold_bars == 4
    assert cfg.trend_pullback_take_profit_pct == 0.0
    assert cfg.trend_pullback_stop_pct == 0.0
    assert cfg.trend_pullback_research_exit_fill == "next_open"


def test_run_backtest_signature_exposes_trend_pullback_defaults():
    sig = inspect.signature(run_backtest)
    assert sig.parameters["trend_pullback_min_adx"].default == 20.0
    assert sig.parameters["trend_pullback_min_slope"].default == 0.0
    assert sig.parameters["trend_pullback_entry_threshold"].default == 0.0015
    assert sig.parameters["trend_pullback_min_atr_percentile"].default == 20.0
    assert sig.parameters["trend_pullback_max_atr_percentile"].default == 0.0
    assert sig.parameters["trend_pullback_exit_style"].default == "fixed_bars"
    assert sig.parameters["trend_pullback_hold_bars"].default == 4
    assert sig.parameters["trend_pullback_take_profit_pct"].default == 0.0
    assert sig.parameters["trend_pullback_stop_pct"].default == 0.0
    assert sig.parameters["trend_pullback_research_exit_fill"].default == "next_open"


def test_backtest_config_exposes_momentum_breakout_defaults():
    cfg = BacktestConfig(dataset_path=Path("dummy"))
    assert cfg.momentum_breakout_lookback_bars == 20
    assert cfg.momentum_breakout_entry_buffer_pct == 0.001
    assert cfg.momentum_breakout_min_adx == 20.0
    assert cfg.momentum_breakout_min_slope == 0.0
    assert cfg.momentum_breakout_min_atr_percentile == 20.0
    assert cfg.momentum_breakout_exit_style == "fixed_bars"
    assert cfg.momentum_breakout_hold_bars == 3
    assert cfg.momentum_breakout_stop_pct == 0.0
    assert cfg.momentum_breakout_take_profit_pct == 0.0


def test_run_backtest_signature_exposes_momentum_breakout_defaults():
    sig = inspect.signature(run_backtest)
    assert sig.parameters["momentum_breakout_lookback_bars"].default == 20
    assert sig.parameters["momentum_breakout_entry_buffer_pct"].default == 0.001
    assert sig.parameters["momentum_breakout_min_adx"].default == 20.0
    assert sig.parameters["momentum_breakout_min_slope"].default == 0.0
    assert sig.parameters["momentum_breakout_min_atr_percentile"].default == 20.0
    assert sig.parameters["momentum_breakout_exit_style"].default == "fixed_bars"
    assert sig.parameters["momentum_breakout_hold_bars"].default == 3
    assert sig.parameters["momentum_breakout_stop_pct"].default == 0.0
    assert sig.parameters["momentum_breakout_take_profit_pct"].default == 0.0


def test_backtest_config_exposes_volatility_expansion_defaults():
    cfg = BacktestConfig(dataset_path=Path("dummy"))
    assert cfg.volatility_expansion_lookback_bars == 20
    assert cfg.volatility_expansion_entry_buffer_pct == 0.001
    assert cfg.volatility_expansion_max_atr_percentile == 0.0
    assert cfg.volatility_expansion_trend_filter is False
    assert cfg.volatility_expansion_min_slope == 0.0
    assert cfg.volatility_expansion_use_volume_confirm is True
    assert cfg.volatility_expansion_exit_style == "fixed_bars"
    assert cfg.volatility_expansion_hold_bars == 4
    assert cfg.volatility_expansion_stop_pct == 0.0
    assert cfg.volatility_expansion_take_profit_pct == 0.0


def test_run_backtest_signature_exposes_volatility_expansion_defaults():
    sig = inspect.signature(run_backtest)
    assert sig.parameters["volatility_expansion_lookback_bars"].default == 20
    assert sig.parameters["volatility_expansion_entry_buffer_pct"].default == 0.001
    assert sig.parameters["volatility_expansion_max_atr_percentile"].default == 0.0
    assert sig.parameters["volatility_expansion_trend_filter"].default is False
    assert sig.parameters["volatility_expansion_min_slope"].default == 0.0
    assert sig.parameters["volatility_expansion_use_volume_confirm"].default is True
    assert sig.parameters["volatility_expansion_exit_style"].default == "fixed_bars"
    assert sig.parameters["volatility_expansion_hold_bars"].default == 4
    assert sig.parameters["volatility_expansion_stop_pct"].default == 0.0
    assert sig.parameters["volatility_expansion_take_profit_pct"].default == 0.0


def test_summarize_hybrid_branch_stats_classifies_completed_trades_by_origin_branch():
    stats = _summarize_hybrid_branch_stats([
        {"action": "BUY", "entry_branch": "bollinger_breakout"},
        {"action": "SELL", "entry_branch": "bollinger_breakout", "pnl": 25.0, "holding_bars": 3},
        {"action": "SELL", "entry_branch": "bollinger_breakout", "pnl": -10.0, "holding_bars": 2},
        {"action": "SELL", "entry_branch": "mean_reversion", "pnl": 5.0, "holding_bars": 1},
    ])

    assert stats["bollinger_breakout"]["total_trades"] == 2
    assert stats["bollinger_breakout"]["wins"] == 1
    assert stats["bollinger_breakout"]["losses"] == 1
    assert stats["bollinger_breakout"]["realized_pnl"] == 15.0
    assert stats["bollinger_breakout"]["win_rate"] == 50.0
    assert stats["bollinger_breakout"]["avg_pnl_per_trade"] == 7.5
    assert stats["bollinger_breakout"]["avg_hold_bars"] == 2.5
    assert stats["mean_reversion"]["total_trades"] == 1
    assert stats["mean_reversion"]["wins"] == 1
    assert stats["mean_reversion"]["losses"] == 0
    assert stats["mean_reversion"]["realized_pnl"] == 5.0

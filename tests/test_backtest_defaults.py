"""Smoke tests for BacktestConfig and run_backtest default assumptions."""
from pathlib import Path

from backtest_runner import BacktestConfig, run_backtest
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

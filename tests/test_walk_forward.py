"""Tests for walk-forward validation window generation and summary logic."""
import pytest
from backtest_runner import _generate_walk_forward_windows, _compute_wf_summary, _validate_wf_args


def _make_trading_days(n: int) -> list[str]:
    """Generate n fake trading day strings for testing (sequential dates, Mon-Fri skipping)."""
    from datetime import date, timedelta
    days = []
    d = date(2025, 1, 2)  # a Thursday
    while len(days) < n:
        if d.weekday() < 5:  # Mon-Fri
            days.append(str(d))
        d += timedelta(days=1)
    return days


def test_window_count_exact_fit():
    """With exactly train+test days, we get exactly one window."""
    days = _make_trading_days(40 + 15)
    windows = _generate_walk_forward_windows(days, train_days=40, test_days=15, step_days=15)
    assert len(windows) == 1


def test_window_count_two_steps():
    """With train+2*test days and step==test, we get exactly 2 windows."""
    days = _make_trading_days(40 + 15 + 15)
    windows = _generate_walk_forward_windows(days, train_days=40, test_days=15, step_days=15)
    assert len(windows) == 2


def test_empty_when_dataset_too_small():
    """Fewer days than train+test yields no windows."""
    days = _make_trading_days(10)
    windows = _generate_walk_forward_windows(days, train_days=40, test_days=15, step_days=15)
    assert windows == []


def test_window_indices_are_one_based():
    days = _make_trading_days(80)
    windows = _generate_walk_forward_windows(days, train_days=40, test_days=15, step_days=15)
    assert windows[0]["window_idx"] == 1
    assert windows[1]["window_idx"] == 2


def test_no_overlap_in_oos_periods():
    """OOS windows must not overlap each other (step >= test guarantees this)."""
    days = _make_trading_days(120)
    windows = _generate_walk_forward_windows(days, train_days=40, test_days=15, step_days=15)
    for i in range(len(windows) - 1):
        assert windows[i]["test_end"] < windows[i + 1]["test_start"], (
            f"OOS periods overlap between window {i+1} and {i+2}"
        )


def test_train_precedes_test():
    """IS period must end before OOS period starts for every window."""
    days = _make_trading_days(120)
    windows = _generate_walk_forward_windows(days, train_days=40, test_days=15, step_days=15)
    for w in windows:
        assert w["train_end"] < w["test_start"], (
            f"Window {w['window_idx']}: IS end {w['train_end']} >= OOS start {w['test_start']}"
        )


def test_train_window_contains_correct_day_count():
    """The train window should span exactly train_days trading days from the dataset."""
    days = _make_trading_days(120)
    train_days = 40
    windows = _generate_walk_forward_windows(days, train_days=train_days, test_days=15, step_days=15)
    w = windows[0]
    # train_start is days[0], train_end is days[train_days-1]
    train_start_idx = days.index(w["train_start"])
    train_end_idx = days.index(w["train_end"])
    assert train_end_idx - train_start_idx + 1 == train_days


def _make_oos_result(pnl: float, trades: int = 10, win_rate: float = 60.0) -> dict:
    winning = [pnl * 0.6 / max(1, trades)] * max(0, round(trades * win_rate / 100))
    losing = [-pnl * 0.4 / max(1, trades - len(winning))] * (trades - len(winning))
    return {
        "realized_pnl": pnl,
        "total_trades": trades,
        "win_rate": win_rate,
        "profit_factor": 1.5 if pnl > 0 else 0.8,
        "sharpe_ratio": 1.0 if pnl > 0 else -0.5,
        "total_return_pct": pnl / 10_000 * 100,
        "max_drawdown_pct": 2.0,
        "winning_pnls": winning,
        "losing_pnls": losing,
    }


def test_summary_profitable_window_count():
    window_results = [
        {"window_idx": 1, "is": _make_oos_result(50), "oos": _make_oos_result(100)},
        {"window_idx": 2, "is": _make_oos_result(50), "oos": _make_oos_result(-20)},
        {"window_idx": 3, "is": _make_oos_result(50), "oos": _make_oos_result(30)},
    ]
    summary = _compute_wf_summary(window_results)
    assert summary["total_windows"] == 3
    assert summary["profitable_windows"] == 2
    assert summary["losing_windows"] == 1


def test_summary_total_pnl():
    window_results = [
        {"window_idx": 1, "is": _make_oos_result(0), "oos": _make_oos_result(100)},
        {"window_idx": 2, "is": _make_oos_result(0), "oos": _make_oos_result(-20)},
    ]
    summary = _compute_wf_summary(window_results)
    assert abs(summary["total_oos_pnl"] - 80.0) < 0.01


def test_summary_best_worst_window():
    window_results = [
        {"window_idx": 1, "is": _make_oos_result(0), "oos": _make_oos_result(50)},
        {"window_idx": 2, "is": _make_oos_result(0), "oos": _make_oos_result(200)},
        {"window_idx": 3, "is": _make_oos_result(0), "oos": _make_oos_result(-30)},
    ]
    summary = _compute_wf_summary(window_results)
    assert summary["best_window_idx"] == 2
    assert summary["worst_window_idx"] == 3


# ---------------------------------------------------------------------------
# _validate_wf_args tests
# ---------------------------------------------------------------------------

def _valid_validate(**overrides):
    """Call _validate_wf_args with safe defaults, applying any overrides."""
    kwargs = dict(
        train_days=60, test_days=20, step_days=20,
        has_regime_specs=False, has_sweep_lists=False,
    )
    kwargs.update(overrides)
    _validate_wf_args(**kwargs)  # should not raise


def test_validate_accepts_valid_args():
    _valid_validate()  # step == test: valid


def test_validate_accepts_step_greater_than_test():
    _valid_validate(test_days=15, step_days=20)  # step > test: valid


def test_validate_rejects_overlapping_oos():
    with pytest.raises(ValueError, match="overlapping OOS windows"):
        _validate_wf_args(train_days=60, test_days=20, step_days=10,
                          has_regime_specs=False, has_sweep_lists=False)


def test_validate_overlap_error_names_both_flags():
    """Error message should reference both --wf-step-days and --wf-test-days values."""
    with pytest.raises(ValueError, match=r"wf-step-days \(10\).*wf-test-days \(20\)"):
        _validate_wf_args(train_days=60, test_days=20, step_days=10,
                          has_regime_specs=False, has_sweep_lists=False)


def test_validate_rejects_zero_train_days():
    with pytest.raises(ValueError, match="--wf-train-days"):
        _validate_wf_args(train_days=0, test_days=20, step_days=20,
                          has_regime_specs=False, has_sweep_lists=False)


def test_validate_rejects_negative_test_days():
    with pytest.raises(ValueError, match="--wf-test-days"):
        _validate_wf_args(train_days=60, test_days=-1, step_days=20,
                          has_regime_specs=False, has_sweep_lists=False)


def test_validate_rejects_zero_step_days():
    with pytest.raises(ValueError, match="--wf-step-days"):
        _validate_wf_args(train_days=60, test_days=20, step_days=0,
                          has_regime_specs=False, has_sweep_lists=False)


def test_validate_rejects_regime_conflict():
    with pytest.raises(ValueError, match="--regime"):
        _validate_wf_args(train_days=60, test_days=20, step_days=20,
                          has_regime_specs=True, has_sweep_lists=False)


def test_validate_rejects_sweep_list_conflict():
    with pytest.raises(ValueError, match="sweep"):
        _validate_wf_args(train_days=60, test_days=20, step_days=20,
                          has_regime_specs=False, has_sweep_lists=True)

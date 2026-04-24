import pandas as pd

from run_trade_path_diagnostics import (
    attach_trade_path_metrics,
    bucket_trade_shape,
    build_trade_shape_summary,
    pair_round_trip_trades,
    summarize_hold_delta,
    summarize_opportunity_capture,
)


def test_pair_round_trip_trades_pairs_buy_and_sell() -> None:
    trades = [
        {"symbol": "AMD", "side": "BUY", "price": 100.0, "shares": 10.0, "timestamp": "2026-01-02T14:30:00Z"},
        {"symbol": "AMD", "side": "SELL", "price": 101.0, "shares": 10.0, "timestamp": "2026-01-02T15:30:00Z", "pnl": 8.0, "holding_bars": 4},
    ]

    paired = pair_round_trip_trades(trades, configured_hold_bars=4)
    assert len(paired) == 1
    assert paired.iloc[0]["symbol"] == "AMD"
    assert paired.iloc[0]["exit_reason"] == "trend_pullback_fixed_bars_exit"
    assert paired.iloc[0]["realized_return_pct"] == 0.8


def test_attach_trade_path_metrics_joins_on_symbol_and_entry_ts() -> None:
    closed = pd.DataFrame(
        [
            {
                "symbol": "AMD",
                "entry_ts": pd.Timestamp("2026-01-02T14:30:00Z"),
                "realized_return_pct": 0.2,
            }
        ]
    )
    signals = pd.DataFrame(
        [
            {
                "symbol": "AMD",
                "entry_ts": pd.Timestamp("2026-01-02T14:30:00Z"),
                "best_net_exit_pct": 0.5,
            }
        ]
    )

    merged = attach_trade_path_metrics(closed, signals)
    assert merged.iloc[0]["best_net_exit_pct"] == 0.5


def test_summarize_hold_delta_handles_same_entry_set() -> None:
    signal_paths = pd.DataFrame(
        [
            {"net_exit_3b_return_pct": 0.2, "net_exit_4b_return_pct": -0.1, "best_exit_bar": 2},
            {"net_exit_3b_return_pct": -0.1, "net_exit_4b_return_pct": 0.3, "best_exit_bar": 4},
            {"net_exit_3b_return_pct": -0.2, "net_exit_4b_return_pct": -0.4, "best_exit_bar": 1},
        ]
    )

    summary = summarize_hold_delta(signal_paths, hold_a=3, hold_b=4)
    assert summary["sample_count"] == 3
    assert summary["profitable_3b_not_4b"] == 1
    assert summary["profitable_4b_not_3b"] == 1
    assert summary["avg_delta_4m3_pct"] < 0


def test_summarize_opportunity_capture_handles_empty_input() -> None:
    summary = summarize_opportunity_capture(pd.DataFrame())
    assert summary["trade_count"] == 0
    assert summary["avg_missed_opportunity_pct"] == 0.0


def test_summarize_opportunity_capture_reports_missed_edge() -> None:
    trade_paths = pd.DataFrame(
        [
            {
                "realized_return_pct": -0.1,
                "best_net_exit_pct": 0.4,
                "worst_net_exit_pct": -0.3,
                "drawdown_before_best_pct": -0.2,
            },
            {
                "realized_return_pct": 0.1,
                "best_net_exit_pct": 0.2,
                "worst_net_exit_pct": -0.1,
                "drawdown_before_best_pct": -0.05,
            },
        ]
    )

    summary = summarize_opportunity_capture(trade_paths)
    assert summary["trade_count"] == 2
    assert summary["avg_missed_opportunity_pct"] > 0
    assert summary["materially_worse_than_best_frac"] == 50.0


def test_bucket_trade_shape_covers_gave_back_case() -> None:
    shape = bucket_trade_shape(
        {
            "net_exit_1b_return_pct": 0.1,
            "best_net_exit_pct": 0.4,
            "realized_return_pct": -0.2,
            "mfe_pct": 0.5,
            "best_exit_bar": 2,
            "worst_adverse_bar": 3,
            "mae_pct": -0.3,
        }
    )
    assert shape == "gave_back_winner"


def test_build_trade_shape_summary_groups_rows() -> None:
    trade_paths = pd.DataFrame(
        [
            {
                "realized_return_pct": -0.2,
                "best_net_exit_pct": 0.4,
                "net_exit_1b_return_pct": 0.1,
                "mfe_pct": 0.5,
                "best_exit_bar": 2,
                "worst_adverse_bar": 2,
                "mae_pct": -0.3,
            },
            {
                "realized_return_pct": -0.4,
                "best_net_exit_pct": -0.1,
                "net_exit_1b_return_pct": -0.2,
                "mfe_pct": -0.05,
                "best_exit_bar": 0,
                "worst_adverse_bar": 1,
                "mae_pct": -0.6,
            },
        ]
    )

    summary = build_trade_shape_summary(trade_paths)
    assert set(summary["shape"]) == {"gave_back_winner", "never_worked"}

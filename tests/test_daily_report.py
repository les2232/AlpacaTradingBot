import json
from pathlib import Path
import shutil
import uuid

import daily_report


def test_load_logs_skips_malformed_lines_and_deduplicates() -> None:
    base_dir = Path.cwd() / f"test_daily_report_{uuid.uuid4().hex}"
    try:
        log_dir = base_dir / "logs" / "2026-04-16"
        log_dir.mkdir(parents=True)

        (log_dir / "bars.jsonl").write_text(
            "\n".join(
                [
                    json.dumps({"event": "bar.received", "trace": "A", "symbol": "AMD", "bar_age_s": 100.0, "stale": False}),
                    json.dumps({"event": "bar.received", "trace": "A", "symbol": "AMD", "bar_age_s": 100.0, "stale": False}),
                    "{bad json",
                ]
            ),
            encoding="utf-8",
        )
        (log_dir / "signals.jsonl").write_text(
            "\n".join(
                [
                    json.dumps({"event": "signal.evaluated", "trace": "A", "symbol": "AMD", "action": "BUY"}),
                    json.dumps({"event": "signal.evaluated", "trace": "A", "symbol": "AMD", "action": "BUY"}),
                    json.dumps({"event": "signal.evaluated", "trace": "B", "symbol": "MSFT", "action": "HOLD", "rejection": "no_signal"}),
                ]
            ),
            encoding="utf-8",
        )
        for name in ("risk", "execution", "positions"):
            (log_dir / f"{name}.jsonl").write_text("", encoding="utf-8")

        logs = daily_report.load_logs(log_dir)
        log_health = daily_report.compute_log_health(logs)
        signals = daily_report.compute_signal_behavior(logs["signals"])

        assert logs["bars"].attrs["malformed_line_count"] == 1
        assert logs["bars"].attrs["duplicate_row_count"] == 1
        assert log_health["malformed_lines"] == 1
        assert log_health["duplicate_rows"] == 2
        assert signals["total_evaluated"] == 2
        assert signals["buy_signals"] == 1
    finally:
        shutil.rmtree(base_dir, ignore_errors=True)


def test_detect_concerns_normalizes_partial_session_signal_rate() -> None:
    sig = {
        "total_evaluated": 255,
        "buy_signals": 7,
        "sell_signals": 2,
        "hold_count": 246,
        "by_symbol": {},
        "rejections": {},
        "rejection_rate": 246 / 255,
    }
    concerns = daily_report.detect_concerns(
        {"evaluated": 255, "avg_age_s": None, "stale_count": 0, "symbols": []},
        sig,
        {"submitted_buys": 0, "submitted_sells": 0, "avg_slippage_bps": None, "worst_slippage_bps": None, "partial_fills": 0, "avg_latency_ms": None},
        {"opened": 0, "closed": 0, "win_rate": None, "avg_hold_bars": None, "avg_pnl_usd": None, "exit_reasons": {}, "total_pnl_usd": None},
        {"malformed_lines": 0, "duplicate_rows": 0},
    )

    assert any("PARTIAL SESSION" in concern for concern in concerns)
    assert any("SIGNAL RATE LOW" in concern for concern in concerns)


def test_load_backtest_baseline_uses_live_config_and_research_artifact() -> None:
    base_dir = Path.cwd() / f"test_daily_report_baseline_{uuid.uuid4().hex}"
    try:
        (base_dir / "config").mkdir(parents=True)
        (base_dir / "results").mkdir(parents=True)
        (base_dir / "config" / "live_config.json").write_text(
            json.dumps(
                {
                    "runtime": {
                        "symbols": ["AMD", "MSFT", "TSLA"],
                        "bar_timeframe_minutes": 15,
                    }
                }
            ),
            encoding="utf-8",
        )
        (base_dir / "results" / "best_config_latest.json").write_text(
            json.dumps(
                {
                    "performance": {
                        "trades_per_day": 6.0,
                        "win_rate": 75.0,
                    }
                }
            ),
            encoding="utf-8",
        )

        baseline, source = daily_report.load_backtest_baseline(base_dir)

        assert baseline["symbols"] == 3
        assert baseline["bars_per_day"] == 26
        assert baseline["signal_rate_per_symbol_per_day"] == 2.0
        assert baseline["win_rate"] == 0.75
        assert source["mode"] == "live_config+research"
        assert source["research_metrics_used"] is True
        assert source["valid_for_comparison"] is True
    finally:
        shutil.rmtree(base_dir, ignore_errors=True)


def test_load_backtest_baseline_rejects_unapproved_or_mismatched_research() -> None:
    base_dir = Path.cwd() / f"test_daily_report_baseline_invalid_{uuid.uuid4().hex}"
    try:
        (base_dir / "config").mkdir(parents=True)
        (base_dir / "results").mkdir(parents=True)
        (base_dir / "config" / "live_config.json").write_text(
            json.dumps(
                {
                    "source": {
                        "approved": False,
                        "rejection_reasons": ["profit_factor 1.01 >= 1.2"],
                    },
                    "runtime": {
                        "symbols": ["AMD", "MSFT", "TSLA"],
                        "bar_timeframe_minutes": 15,
                        "strategy_mode": "mean_reversion",
                        "sma_bars": 15,
                        "entry_threshold_pct": 0.0015,
                    },
                }
            ),
            encoding="utf-8",
        )
        (base_dir / "results" / "best_config_latest.json").write_text(
            json.dumps(
                {
                    "approved": False,
                    "rejection_reasons": ["profit_factor 1.18 >= 1.2"],
                    "config": {
                        "strategy_mode": "mean_reversion",
                        "sma_bars": 20,
                        "entry_threshold_pct": 0.002,
                    },
                    "performance": {
                        "trades_per_day": 6.0,
                        "win_rate": 75.0,
                    },
                }
            ),
            encoding="utf-8",
        )

        baseline, source = daily_report.load_backtest_baseline(base_dir)

        assert baseline["symbols"] == 3
        assert baseline["bars_per_day"] == 26
        assert baseline["signal_rate_per_symbol_per_day"] == daily_report.DEFAULT_BACKTEST["signal_rate_per_symbol_per_day"]
        assert baseline["win_rate"] == daily_report.DEFAULT_BACKTEST["win_rate"]
        assert source["research_metrics_used"] is False
        assert source["valid_for_comparison"] is False
        assert source["live_config_approved"] is False
        assert source["research_approved"] is False
        assert source["research_matches_live_runtime"] is False
        assert any("live config is approved=false" in item for item in source["validation_errors"])
        assert any("research artifact is approved=false" in item for item in source["validation_errors"])
        assert any("research config does not match live runtime" in item for item in source["validation_errors"])
    finally:
        shutil.rmtree(base_dir, ignore_errors=True)


def test_load_backtest_baseline_parses_string_approval_flags() -> None:
    base_dir = Path.cwd() / f"test_daily_report_baseline_string_bool_{uuid.uuid4().hex}"
    try:
        (base_dir / "config").mkdir(parents=True)
        (base_dir / "results").mkdir(parents=True)
        (base_dir / "config" / "live_config.json").write_text(
            json.dumps(
                {
                    "source": {
                        "approved": "false",
                        "rejection_reasons": ["not approved yet"],
                    },
                    "runtime": {
                        "symbols": ["AMD"],
                        "bar_timeframe_minutes": 15,
                    },
                }
            ),
            encoding="utf-8",
        )
        (base_dir / "results" / "best_config_latest.json").write_text(
            json.dumps(
                {
                    "approved": "false",
                    "rejection_reasons": ["not approved yet"],
                    "config": {},
                    "performance": {
                        "trades_per_day": 2.0,
                        "win_rate": 80.0,
                    },
                }
            ),
            encoding="utf-8",
        )

        _, source = daily_report.load_backtest_baseline(base_dir)

        assert source["live_config_approved"] is False
        assert source["research_approved"] is False
        assert source["valid_for_comparison"] is False
    finally:
        shutil.rmtree(base_dir, ignore_errors=True)

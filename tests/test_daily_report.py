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
    finally:
        shutil.rmtree(base_dir, ignore_errors=True)

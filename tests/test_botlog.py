import json
import shutil
import uuid
from datetime import datetime
from pathlib import Path

from botlog import BotLogger


def test_botlogger_uses_distinct_file_handlers_per_log_root() -> None:
    base_dir = Path.cwd() / f"test_botlog_{uuid.uuid4().hex}"
    try:
        log_root_a = base_dir / "logs_a"
        log_root_b = base_dir / "logs_b"
        logger_a = BotLogger(log_root=log_root_a, session_id="session-a")
        logger_b = BotLogger(log_root=log_root_b, session_id="session-b")

        logger_a.signal(
            symbol="AMD",
            decision_ts="2026-04-17T14:30:00+00:00",
            bar_close=100.0,
            sma=101.0,
            trend_sma=102.0,
            atr_pct=1.0,
            atr_percentile=50.0,
            volume_ratio=1.2,
            action="BUY",
            holding=False,
            trend_filter_pass=True,
            atr_filter_pass=True,
            window_open=True,
            rejection=None,
        )
        logger_b.signal(
            symbol="MSFT",
            decision_ts="2026-04-17T14:45:00+00:00",
            bar_close=200.0,
            sma=199.0,
            trend_sma=198.0,
            atr_pct=1.1,
            atr_percentile=60.0,
            volume_ratio=1.3,
            action="SELL",
            holding=False,
            trend_filter_pass=True,
            atr_filter_pass=True,
            window_open=True,
            rejection=None,
        )

        date_str = datetime.now().strftime("%Y-%m-%d")
        signals_a = log_root_a / date_str / "signals.jsonl"
        signals_b = log_root_b / date_str / "signals.jsonl"

        rows_a = [json.loads(line) for line in signals_a.read_text(encoding="utf-8").splitlines() if line.strip()]
        rows_b = [json.loads(line) for line in signals_b.read_text(encoding="utf-8").splitlines() if line.strip()]

        assert [row["symbol"] for row in rows_a] == ["AMD"]
        assert [row["symbol"] for row in rows_b] == ["MSFT"]
        assert rows_a[0]["session_id"] == "session-a"
        assert rows_b[0]["session_id"] == "session-b"
        assert rows_a[0]["event"] == "signal.evaluated"
    finally:
        shutil.rmtree(base_dir, ignore_errors=True)


def test_botlogger_signal_persists_extra_explainability_fields() -> None:
    base_dir = Path.cwd() / f"test_botlog_{uuid.uuid4().hex}"
    try:
        logger = BotLogger(log_root=base_dir, session_id="session-a")
        logger.signal(
            symbol="AMD",
            decision_ts="2026-04-17T14:30:00+00:00",
            bar_close=100.0,
            sma=101.0,
            trend_sma=102.0,
            atr_pct=1.0,
            atr_percentile=50.0,
            volume_ratio=1.2,
            action="BUY",
            holding=False,
            trend_filter_pass=True,
            atr_filter_pass=True,
            window_open=True,
            rejection=None,
            extra_fields={
                "strategy_mode": "mean_reversion",
                "final_signal_reason": "mean_reversion_sma_entry",
                "decision_summary": "BUY in mean reversion mode. reason: mean reversion sma entry.",
            },
        )

        date_str = datetime.now().strftime("%Y-%m-%d")
        signals = base_dir / date_str / "signals.jsonl"
        rows = [json.loads(line) for line in signals.read_text(encoding="utf-8").splitlines() if line.strip()]

        assert rows[0]["strategy_mode"] == "mean_reversion"
        assert rows[0]["final_signal_reason"] == "mean_reversion_sma_entry"
        assert "mean reversion mode" in rows[0]["decision_summary"]
    finally:
        shutil.rmtree(base_dir, ignore_errors=True)


def test_botlogger_lifecycle_persists_process_stage() -> None:
    base_dir = Path.cwd() / f"test_botlog_{uuid.uuid4().hex}"
    try:
        logger = BotLogger(log_root=base_dir, session_id="session-a")
        logger.lifecycle("started", strategy_mode="mean_reversion", symbol_count=15)

        date_str = datetime.now().strftime("%Y-%m-%d")
        lifecycle = base_dir / date_str / "lifecycle.jsonl"
        rows = [json.loads(line) for line in lifecycle.read_text(encoding="utf-8").splitlines() if line.strip()]

        assert rows[0]["event"] == "process.lifecycle"
        assert rows[0]["stage"] == "started"
        assert rows[0]["strategy_mode"] == "mean_reversion"
        assert rows[0]["symbol_count"] == 15
    finally:
        shutil.rmtree(base_dir, ignore_errors=True)

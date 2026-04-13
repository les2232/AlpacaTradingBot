from __future__ import annotations

import json
from pathlib import Path
from uuid import uuid4

from dashboard_state import (
    PersistedBotSnapshot,
    PersistedCycleReport,
    PersistedStartupConfig,
    PersistedSymbolSnapshot,
    _build_session_warnings,
    _select_startup_config_for_session,
    load_dashboard_state,
)
from storage import BotStorage


def test_load_dashboard_state_reads_persisted_startup_config(monkeypatch) -> None:
    unique = uuid4().hex
    logs_dir = f"test_dashboard_logs_{unique}"
    db_path = f"test_dashboard_state_{unique}.db"

    monkeypatch.setenv("BOT_LOG_ROOT", logs_dir)
    try:
        BotStorage(Path(db_path))
        day_dir = Path(logs_dir) / "2026-04-12"
        day_dir.mkdir(parents=True, exist_ok=True)
        (day_dir / "startup_config.20260412T133000Z.json").write_text(
            json.dumps(
                {
                    "session_id": "session-a",
                    "started_at_utc": "2026-04-12T13:30:00+00:00",
                    "launch_mode": "live",
                    "execution_enabled": True,
                    "paper": True,
                    "account_mode": "paper",
                    "strategy_mode": "mean_reversion",
                    "bar_timeframe_minutes": 15,
                    "sma_bars": 20,
                    "symbols": ["AMD", "MSFT"],
                    "symbol_count": 2,
                    "runtime_config_path": "config/live_config.json",
                    "runtime_overrides": ["symbols", "strategy_mode"],
                    "max_usd_per_trade": 200.0,
                    "max_symbol_exposure_usd": 200.0,
                    "max_open_positions": 3,
                    "max_daily_loss_usd": 300.0,
                    "max_orders_per_minute": 6,
                    "max_price_deviation_bps": 75.0,
                    "max_live_price_age_seconds": 60,
                    "max_data_delay_seconds": 300,
                    "db_path": db_path,
                }
            ),
            encoding="utf-8",
        )

        state = load_dashboard_state()

        assert state.has_persisted_startup_config is True
        assert state.startup_config is not None
        assert state.startup_config.strategy_mode == "mean_reversion"
        assert state.startup_config.symbols == ["AMD", "MSFT"]
        assert state.symbol_state_status == "cleared_runtime_symbol_state"
        assert str(state.storage.db_path).endswith(db_path)
    finally:
        log_dir = Path(logs_dir)
        if log_dir.exists():
            for child in sorted(log_dir.rglob("*"), reverse=True):
                if child.is_file():
                    child.unlink()
                else:
                    child.rmdir()


def test_load_dashboard_state_reports_missing_startup_config(monkeypatch) -> None:
    unique = uuid4().hex
    logs_dir = f"test_dashboard_logs_{unique}"
    db_path = f"test_dashboard_state_{unique}.db"
    monkeypatch.setenv("BOT_LOG_ROOT", logs_dir)
    monkeypatch.setenv("BOT_DB_PATH", db_path)

    try:
        state = load_dashboard_state()

        assert state.has_persisted_startup_config is False
        assert state.startup_config is None
    finally:
        log_dir = Path(logs_dir)
        if log_dir.exists():
            for child in sorted(log_dir.rglob("*"), reverse=True):
                if child.is_file():
                    child.unlink()
                else:
                    child.rmdir()


def test_load_dashboard_state_ignores_launches_directory(monkeypatch) -> None:
    unique = uuid4().hex
    logs_dir = f"test_dashboard_logs_{unique}"
    db_path = f"test_dashboard_state_{unique}.db"
    monkeypatch.setenv("BOT_LOG_ROOT", logs_dir)

    try:
        BotStorage(Path(db_path))
        session_dir = Path(logs_dir) / "2026-04-13"
        session_dir.mkdir(parents=True, exist_ok=True)
        (session_dir / "startup_config.20260413T133000Z.json").write_text(
            json.dumps(
                {
                    "session_id": "session-qcom",
                    "started_at_utc": "2026-04-13T13:30:00+00:00",
                    "launch_mode": "live",
                    "execution_enabled": True,
                    "paper": True,
                    "account_mode": "paper",
                    "strategy_mode": "mean_reversion",
                    "bar_timeframe_minutes": 15,
                    "sma_bars": 20,
                    "symbols": ["QCOM"],
                    "symbol_count": 1,
                    "runtime_config_path": "config/live_config.json",
                    "runtime_overrides": [],
                    "max_usd_per_trade": 200.0,
                    "max_symbol_exposure_usd": 200.0,
                    "max_open_positions": 3,
                    "max_daily_loss_usd": 300.0,
                    "max_orders_per_minute": 6,
                    "max_price_deviation_bps": 75.0,
                    "max_live_price_age_seconds": 60,
                    "max_data_delay_seconds": 300,
                    "db_path": db_path,
                }
            ),
            encoding="utf-8",
        )
        launches_dir = Path(logs_dir) / "launches"
        launches_dir.mkdir(parents=True, exist_ok=True)

        state = load_dashboard_state()

        assert state.has_persisted_startup_config is True
        assert state.startup_config is not None
        assert state.startup_config.symbols == ["QCOM"]
    finally:
        log_dir = Path(logs_dir)
        if log_dir.exists():
            for child in sorted(log_dir.rglob("*"), reverse=True):
                if child.is_file():
                    child.unlink()
                else:
                    child.rmdir()


def test_load_dashboard_state_reports_unmatched_startup_artifact(monkeypatch) -> None:
    unique = uuid4().hex
    logs_dir = f"test_dashboard_logs_{unique}"
    db_path = f"test_dashboard_state_{unique}.db"
    monkeypatch.setenv("BOT_LOG_ROOT", logs_dir)
    monkeypatch.setenv("BOT_DB_PATH", db_path)

    try:
        storage = BotStorage(Path(db_path))
        day_dir = Path(logs_dir) / "2026-04-13"
        day_dir.mkdir(parents=True, exist_ok=True)
        (day_dir / "startup_config.20260413T143426Z.json").write_text(
            json.dumps(
                {
                    "session_id": "session-amd",
                    "started_at_utc": "2026-04-13T14:34:26+00:00",
                    "launch_mode": "live",
                    "execution_enabled": True,
                    "paper": True,
                    "account_mode": "paper",
                    "strategy_mode": "mean_reversion",
                    "bar_timeframe_minutes": 15,
                    "sma_bars": 20,
                    "symbols": ["AMD"],
                    "symbol_count": 1,
                    "runtime_config_path": "config/live_config.json",
                    "runtime_overrides": ["symbols"],
                    "max_usd_per_trade": 200.0,
                    "max_symbol_exposure_usd": 200.0,
                    "max_open_positions": 3,
                    "max_daily_loss_usd": 300.0,
                    "max_orders_per_minute": 6,
                    "max_price_deviation_bps": 75.0,
                    "max_live_price_age_seconds": 60,
                    "max_data_delay_seconds": 300,
                    "db_path": db_path,
                }
            ),
            encoding="utf-8",
        )
        snapshot = type("Snapshot", (), {})()
        snapshot.timestamp_utc = "2026-04-13T14:30:00+00:00"
        snapshot.cash = 1.0
        snapshot.buying_power = 1.0
        snapshot.equity = 1.0
        snapshot.last_equity = 1.0
        snapshot.daily_pnl = 0.0
        snapshot.kill_switch_triggered = False
        snapshot.positions = {}
        snapshot.symbols = [
            type("Row", (), {"symbol": "AMD", "price": 1.0, "sma": 1.0, "action": "HOLD", "holding": False, "quantity": 0.0, "market_value": 0.0, "ml_probability_up": None, "ml_confidence": None, "ml_training_rows": None, "holding_minutes": None, "error": None})(),
            type("Row", (), {"symbol": "TSLA", "price": 1.0, "sma": 1.0, "action": "HOLD", "holding": False, "quantity": 0.0, "market_value": 0.0, "ml_probability_up": None, "ml_confidence": None, "ml_training_rows": None, "holding_minutes": None, "error": None})(),
        ]
        storage.save_snapshot(snapshot, [], session_id="older-session", symbol_fingerprint="older")
        (day_dir / "risk.jsonl").write_text(
            json.dumps(
                {
                    "event": "cycle.summary",
                    "decision_ts": "2026-04-13T14:30:00+00:00",
                    "execute_orders": True,
                    "processed_bar": True,
                    "skip_reason": "signals_blocked_or_skipped",
                    "buy_signals": 1,
                    "sell_signals": 0,
                    "hold_signals": 1,
                    "error_signals": 0,
                    "orders_submitted": 0,
                    "ts": "2026-04-13T14:30:24+00:00",
                }
            ) + "\n",
            encoding="utf-8",
        )

        state = load_dashboard_state()

        assert state.startup_config is not None
        assert state.has_persisted_snapshot is False
        assert state.symbol_state_status == "stale_persisted_symbols_ignored"
        assert any("Persisted symbol state is from a different run and is being ignored." in message for message in state.session_warnings)
    finally:
        log_dir = Path(logs_dir)
        if log_dir.exists():
            for child in sorted(log_dir.rglob("*"), reverse=True):
                if child.is_file():
                    child.unlink()
                else:
                    child.rmdir()


def test_build_session_warnings_detects_newer_startup_and_symbol_mismatch() -> None:
    startup = PersistedStartupConfig(
        session_id="session-amd",
        started_at_utc="2026-04-13T14:34:26+00:00",
        launch_mode="live",
        execution_enabled=True,
        paper=True,
        account_mode="paper",
        strategy_mode="mean_reversion",
        bar_timeframe_minutes=15,
        sma_bars=20,
        symbols=["AMD"],
        symbol_count=1,
        runtime_config_path="config/live_config.json",
        runtime_overrides=("symbols",),
        max_usd_per_trade=200.0,
        max_symbol_exposure_usd=200.0,
        max_open_positions=3,
        max_daily_loss_usd=300.0,
        max_orders_per_minute=6,
        max_price_deviation_bps=75.0,
        max_live_price_age_seconds=60,
        max_data_delay_seconds=300,
        db_path="bot_history.db",
    )
    snapshot = PersistedBotSnapshot(
        timestamp_utc="2026-04-13T14:30:00+00:00",
        cash=1000.0,
        buying_power=1000.0,
        equity=1000.0,
        last_equity=999.0,
        daily_pnl=1.0,
        kill_switch_triggered=False,
        positions={},
        symbols=[
            PersistedSymbolSnapshot(
                symbol="AMD",
                price=1.0,
                sma=1.0,
                action="HOLD",
                holding=False,
                quantity=0.0,
                market_value=0.0,
            ),
            PersistedSymbolSnapshot(
                symbol="TSLA",
                price=1.0,
                sma=1.0,
                action="HOLD",
                holding=False,
                quantity=0.0,
                market_value=0.0,
            ),
        ],
    )
    cycle = PersistedCycleReport(
        decision_timestamp="2026-04-13T14:30:00+00:00",
        execute_orders=True,
        processed_bar=True,
        skip_reason="signals_blocked_or_skipped",
        buy_signals=1,
        sell_signals=0,
        hold_signals=1,
        error_signals=0,
        orders_submitted=0,
        observed_at_utc="2026-04-13T14:30:24+00:00",
    )

    warnings = _build_session_warnings(
        startup_config=startup,
        snapshot=snapshot,
        has_persisted_snapshot=True,
        last_cycle_report=cycle,
        ignored_snapshot_symbols=("AMD", "TSLA"),
    )

    assert any("Current config symbols" in message for message in warnings)
    assert any("Latest persisted snapshot symbols" in message for message in warnings)


def test_select_startup_config_prefers_matching_run() -> None:
    earlier = PersistedStartupConfig(
        session_id="session-earlier",
        started_at_utc="2026-04-13T14:00:00+00:00",
        launch_mode="live",
        execution_enabled=True,
        paper=True,
        account_mode="paper",
        strategy_mode="mean_reversion",
        bar_timeframe_minutes=15,
        sma_bars=20,
        symbols=["AMD", "TSLA"],
        symbol_count=2,
        runtime_config_path="config/live_config.json",
        runtime_overrides=("symbols",),
        max_usd_per_trade=200.0,
        max_symbol_exposure_usd=200.0,
        max_open_positions=3,
        max_daily_loss_usd=300.0,
        max_orders_per_minute=6,
        max_price_deviation_bps=75.0,
        max_live_price_age_seconds=60,
        max_data_delay_seconds=300,
        db_path="bot_history.db",
        artifact_path="logs/2026-04-13/startup_config.20260413T140000Z.json",
    )
    later = PersistedStartupConfig(
        session_id="session-later",
        started_at_utc="2026-04-13T14:34:26+00:00",
        launch_mode="live",
        execution_enabled=True,
        paper=True,
        account_mode="paper",
        strategy_mode="mean_reversion",
        bar_timeframe_minutes=15,
        sma_bars=20,
        symbols=["AMD"],
        symbol_count=1,
        runtime_config_path="config/live_config.json",
        runtime_overrides=("symbols",),
        max_usd_per_trade=200.0,
        max_symbol_exposure_usd=200.0,
        max_open_positions=3,
        max_daily_loss_usd=300.0,
        max_orders_per_minute=6,
        max_price_deviation_bps=75.0,
        max_live_price_age_seconds=60,
        max_data_delay_seconds=300,
        db_path="bot_history.db",
        artifact_path="logs/2026-04-13/startup_config.20260413T143426Z.json",
    )
    snapshot = PersistedBotSnapshot(
        timestamp_utc="2026-04-13T14:30:00+00:00",
        cash=1000.0,
        buying_power=1000.0,
        equity=1000.0,
        last_equity=999.0,
        daily_pnl=1.0,
        kill_switch_triggered=False,
        positions={},
        symbols=[
            PersistedSymbolSnapshot(
                symbol="AMD",
                price=1.0,
                sma=1.0,
                action="HOLD",
                holding=False,
                quantity=0.0,
                market_value=0.0,
            ),
            PersistedSymbolSnapshot(
                symbol="TSLA",
                price=1.0,
                sma=1.0,
                action="HOLD",
                holding=False,
                quantity=0.0,
                market_value=0.0,
            ),
        ],
    )
    cycle = PersistedCycleReport(
        decision_timestamp="2026-04-13T14:30:00+00:00",
        execute_orders=True,
        processed_bar=True,
        skip_reason="signals_blocked_or_skipped",
        buy_signals=1,
        sell_signals=0,
        hold_signals=1,
        error_signals=0,
        orders_submitted=0,
        observed_at_utc="2026-04-13T14:30:24+00:00",
    )

    selected = _select_startup_config_for_session(
        startup_configs=[earlier, later],
        snapshot=snapshot,
        has_persisted_snapshot=True,
        last_cycle_report=cycle,
    )

    assert selected is not None
    assert selected.artifact_path == later.artifact_path


def test_load_dashboard_state_ignores_snapshot_from_different_symbol_session(monkeypatch) -> None:
    unique = uuid4().hex
    logs_dir = f"test_dashboard_logs_{unique}"
    db_path = f"test_dashboard_state_{unique}.db"
    monkeypatch.setenv("BOT_LOG_ROOT", logs_dir)

    try:
        storage = BotStorage(Path(db_path))
        day_dir = Path(logs_dir) / "2026-04-13"
        day_dir.mkdir(parents=True, exist_ok=True)
        (day_dir / "startup_config.20260413T170000Z.json").write_text(
            json.dumps(
                {
                    "session_id": "session-current",
                    "started_at_utc": "2026-04-13T17:00:00+00:00",
                    "launch_mode": "live",
                    "execution_enabled": True,
                    "paper": True,
                    "account_mode": "paper",
                    "strategy_mode": "mean_reversion",
                    "bar_timeframe_minutes": 15,
                    "sma_bars": 20,
                    "symbols": ["AMD", "TSLA"],
                    "symbol_count": 2,
                    "runtime_config_path": "config/live_config.json",
                    "runtime_overrides": ["symbols"],
                    "max_usd_per_trade": 200.0,
                    "max_symbol_exposure_usd": 200.0,
                    "max_open_positions": 3,
                    "max_daily_loss_usd": 300.0,
                    "max_orders_per_minute": 6,
                    "max_price_deviation_bps": 75.0,
                    "max_live_price_age_seconds": 60,
                    "max_data_delay_seconds": 300,
                    "db_path": db_path,
                }
            ),
            encoding="utf-8",
        )
        stale_snapshot = type("Snapshot", (), {})()
        stale_snapshot.timestamp_utc = "2026-04-13T16:45:00+00:00"
        stale_snapshot.cash = 1.0
        stale_snapshot.buying_power = 1.0
        stale_snapshot.equity = 1.0
        stale_snapshot.last_equity = 1.0
        stale_snapshot.daily_pnl = 0.0
        stale_snapshot.kill_switch_triggered = False
        stale_snapshot.positions = {}
        stale_snapshot.symbols = [
            type("Row", (), {"symbol": "AMD", "price": 1.0, "sma": 1.0, "action": "HOLD", "holding": False, "quantity": 0.0, "market_value": 0.0, "ml_probability_up": None, "ml_confidence": None, "ml_training_rows": None, "holding_minutes": None, "error": None})(),
            type("Row", (), {"symbol": "QCOM", "price": 1.0, "sma": 1.0, "action": "HOLD", "holding": False, "quantity": 0.0, "market_value": 0.0, "ml_probability_up": None, "ml_confidence": None, "ml_training_rows": None, "holding_minutes": None, "error": None})(),
        ]
        storage.save_snapshot(stale_snapshot, [], session_id="session-stale", symbol_fingerprint="stale")

        state = load_dashboard_state()

        assert state.startup_config is not None
        assert state.startup_config.session_id == "session-current"
        assert state.has_persisted_snapshot is False
        assert state.snapshot.timestamp_utc == "No persisted snapshot yet"
        assert state.symbol_state_status == "stale_persisted_symbols_ignored"
        assert state.ignored_snapshot_symbols == ("AMD", "QCOM")
        assert any("Persisted symbol state is from a different run and is being ignored." in message for message in state.session_warnings)
    finally:
        log_dir = Path(logs_dir)
        if log_dir.exists():
            for child in sorted(log_dir.rglob("*"), reverse=True):
                if child.is_file():
                    child.unlink()
                else:
                    child.rmdir()

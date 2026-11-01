from __future__ import annotations

import json
import os
from pathlib import Path
from uuid import uuid4

import pytest

from dashboard_state import (
    PersistedBotSnapshot,
    PersistedCycleReport,
    PersistedStartupConfig,
    PersistedSymbolSnapshot,
    _build_session_warnings,
    _load_startup_configs,
    _payload_matches_session,
    _select_startup_config_for_session,
    _startup_config_from_payload,
    load_dashboard_state,
)
from storage import BotStorage


_TEST_ACCOUNT_FINGERPRINT = "a" * 64
_TEST_SCOPE_TIMESTAMP = "2026-04-13T13:00:00+00:00"


def _bind_paper_test_scope(storage: BotStorage) -> None:
    storage.bind_runtime_state_scope(
        environment="paper",
        account_fingerprint=_TEST_ACCOUNT_FINGERPRINT,
        provenance="offline-dashboard-test-fixture",
        history_cutoff_utc=_TEST_SCOPE_TIMESTAMP,
        bound_at_utc=_TEST_SCOPE_TIMESTAMP,
    )


def _cleanup_dashboard_test_artifacts(logs_dir: str | Path, db_path: str | Path) -> None:
    db_file = Path(db_path)
    try:
        if db_file.exists():
            db_file.unlink()
    except PermissionError:
        pass
    log_dir = Path(logs_dir)
    if log_dir.exists():
        for child in sorted(log_dir.rglob("*"), reverse=True):
            try:
                if child.is_file():
                    child.unlink()
                else:
                    child.rmdir()
            except OSError:
                pass


def _valid_startup_payload(
    db_path: Path,
    *,
    session_id: str = "session-valid",
    started_at_utc: str = "2026-04-24T18:17:22+00:00",
) -> dict[str, object]:
    return {
        "session_id": session_id,
        "started_at_utc": started_at_utc,
        "launch_mode": "live",
        "execution_enabled": True,
        "paper": True,
        "account_mode": "paper",
        "strategy_mode": "mean_reversion",
        "bar_timeframe_minutes": 15,
        "symbols": ["AMD", "MSFT"],
        "symbol_count": 2,
        "db_path": str(db_path.resolve()),
    }


def _minimal_persisted_startup_config(
    *,
    session_id: str = "session-current",
    symbols: list[str] | None = None,
    paper: bool = True,
) -> PersistedStartupConfig:
    selected_symbols = symbols or ["AMD"]
    return PersistedStartupConfig(
        session_id=session_id,
        started_at_utc="2026-08-17T14:00:00+00:00",
        launch_mode="live",
        execution_enabled=True,
        paper=paper,
        account_mode="paper" if paper else "live",
        strategy_mode="mean_reversion",
        bar_timeframe_minutes=15,
        sma_bars=20,
        symbols=selected_symbols,
        symbol_count=len(selected_symbols),
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


def _persisted_snapshot_with_symbols(symbols: list[str]) -> PersistedBotSnapshot:
    return PersistedBotSnapshot(
        timestamp_utc="2026-08-17T14:15:00+00:00",
        cash=1_000.0,
        buying_power=1_000.0,
        equity=1_000.0,
        last_equity=1_000.0,
        daily_pnl=0.0,
        kill_switch_triggered=False,
        positions={},
        symbols=[
            PersistedSymbolSnapshot(
                symbol=symbol,
                price=1.0,
                sma=1.0,
                action="HOLD",
                holding=False,
                quantity=0.0,
                market_value=0.0,
            )
            for symbol in symbols
        ],
    )


def test_load_dashboard_state_falls_back_from_invalid_latest_test_metadata(
    monkeypatch,
    tmp_path: Path,
) -> None:
    logs_dir = tmp_path / "logs"
    db_path = tmp_path / "bot.db"
    monkeypatch.setenv("BOT_LOG_ROOT", str(logs_dir))
    monkeypatch.setenv("BOT_DB_PATH", str(db_path))

    storage = BotStorage(db_path)
    _bind_paper_test_scope(storage)
    storage.save_snapshot(
        PersistedBotSnapshot(
            timestamp_utc="2026-04-24T19:45:00+00:00",
            cash=1000.0,
            buying_power=1000.0,
            equity=1000.0,
            last_equity=1000.0,
            daily_pnl=0.0,
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
                    symbol="MSFT",
                    price=1.0,
                    sma=1.0,
                    action="HOLD",
                    holding=False,
                    quantity=0.0,
                    market_value=0.0,
                ),
            ],
        ),
        [],
        session_id="session-valid",
        symbol_fingerprint="valid",
        environment="paper",
        account_fingerprint=_TEST_ACCOUNT_FINGERPRINT,
    )

    valid_day = logs_dir / "2026-04-24"
    valid_day.mkdir(parents=True)
    valid_path = valid_day / "startup_config.20260424T181722.246920Z.json"
    valid_path.write_text(
        json.dumps(_valid_startup_payload(db_path)),
        encoding="utf-8",
    )

    unpersisted_day = logs_dir / "2026-07-30"
    unpersisted_day.mkdir(parents=True)
    unpersisted_path = (
        unpersisted_day / "startup_config.20260731T004715.917400Z.json"
    )
    unpersisted_path.write_text(
        json.dumps(
            _valid_startup_payload(
                db_path,
                session_id="unpersisted-relaunch",
                started_at_utc="2026-07-31T00:47:15+00:00",
            )
        ),
        encoding="utf-8",
    )

    invalid_day = logs_dir / "2026-08-06"
    invalid_day.mkdir(parents=True)
    vanished_pytest_db = tmp_path / ".pytest_tmp_codex_cli_paths" / "test" / "bot.db"
    invalid_payload = _valid_startup_payload(
        vanished_pytest_db,
        session_id="live-pytest-artifact",
        started_at_utc="2026-08-07T01:12:43+00:00",
    )
    invalid_payload.update({"symbols": [], "symbol_count": 0})
    (invalid_day / "startup_config.20260807T011243.390360Z.json").write_text(
        json.dumps(invalid_payload),
        encoding="utf-8",
    )
    # A copied legacy alias must not rescue a day that already has versioned metadata.
    (invalid_day / "startup_config.json").write_text(
        json.dumps(_valid_startup_payload(db_path, session_id="legacy-alias")),
        encoding="utf-8",
    )

    configs = _load_startup_configs()
    state = load_dashboard_state()

    assert [config.session_id for config in configs] == [
        "session-valid",
        "unpersisted-relaunch",
    ]
    assert state.startup_config is not None
    assert state.startup_config.session_id == "session-valid"
    assert state.startup_config.artifact_path == str(valid_path)
    assert state.storage.db_path == db_path.resolve()


def test_load_dashboard_state_keeps_repaired_snapshot_when_startup_metadata_is_unmatched(
    monkeypatch,
    tmp_path: Path,
) -> None:
    logs_dir = tmp_path / "logs"
    db_path = tmp_path / "bot.db"
    monkeypatch.setenv("BOT_LOG_ROOT", str(logs_dir))
    monkeypatch.setenv("BOT_DB_PATH", str(db_path))

    source_session_id = "session-source"
    repair_session_id = "repair-20260817T010000Z-source-state"
    source_symbols = [
        PersistedSymbolSnapshot(
            symbol="AMD",
            price=180.0,
            sma=175.0,
            action="HOLD",
            holding=True,
            quantity=2.0,
            market_value=360.0,
        ),
        PersistedSymbolSnapshot(
            symbol="MSFT",
            price=510.0,
            sma=500.0,
            action="HOLD",
            holding=False,
            quantity=0.0,
            market_value=0.0,
        ),
    ]
    repaired_symbols = [
        PersistedSymbolSnapshot(
            symbol=item.symbol,
            price=item.price,
            sma=item.sma,
            action=item.action,
            holding=item.holding,
            quantity=item.quantity,
            market_value=item.market_value,
        )
        for item in source_symbols
    ]
    storage = BotStorage(db_path)
    storage.save_snapshot(
        PersistedBotSnapshot(
            timestamp_utc="2026-08-16T23:30:00+00:00",
            cash=1_000.0,
            buying_power=1_000.0,
            equity=1_360.0,
            last_equity=1_350.0,
            daily_pnl=10.0,
            kill_switch_triggered=False,
            positions={},
            symbols=source_symbols,
        ),
        [],
        session_id=source_session_id,
        symbol_fingerprint="source-universe",
    )
    # Outcome A repair rows deliberately copy the source run's balances,
    # fingerprint, and symbol rows under a new repair-* session identity.
    storage.save_snapshot(
        PersistedBotSnapshot(
            timestamp_utc="2026-08-17T01:00:00+00:00",
            cash=1_000.0,
            buying_power=1_000.0,
            equity=1_360.0,
            last_equity=1_350.0,
            daily_pnl=10.0,
            kill_switch_triggered=False,
            positions={},
            symbols=repaired_symbols,
        ),
        [],
        session_id=repair_session_id,
        symbol_fingerprint="source-universe",
    )

    day_dir = logs_dir / "2026-08-17"
    day_dir.mkdir(parents=True)
    source_path = day_dir / "startup_config.20260816T230000Z.json"
    source_path.write_text(
        json.dumps(
            _valid_startup_payload(
                db_path,
                session_id=source_session_id,
                started_at_utc="2026-08-16T23:00:00+00:00",
            )
        ),
        encoding="utf-8",
    )
    unrelated_payload = _valid_startup_payload(
        db_path,
        session_id="session-unpersisted-qcom",
        started_at_utc="2026-08-17T00:30:00+00:00",
    )
    unrelated_payload.update({"symbols": ["QCOM"], "symbol_count": 1})
    (day_dir / "startup_config.20260817T003000Z.json").write_text(
        json.dumps(unrelated_payload),
        encoding="utf-8",
    )

    state = load_dashboard_state()

    assert state.startup_config is None
    assert state.has_persisted_startup_config is False
    assert state.has_persisted_snapshot is True
    assert state.snapshot.timestamp_utc == "2026-08-17T01:00:00+00:00"
    assert state.snapshot.cash == 1_000.0
    assert [item.symbol for item in state.snapshot.symbols] == ["AMD", "MSFT"]
    assert state.drift_report is None
    assert state.symbol_state_status == "unmatched_runtime_metadata_snapshot_preserved"
    assert state.ignored_snapshot_symbols == ()
    assert any(
        "no matching validated startup metadata" in message
        for message in state.session_warnings
    )


def test_load_dashboard_state_warns_when_persisted_session_has_no_startup_artifacts(
    monkeypatch,
    tmp_path: Path,
) -> None:
    logs_dir = tmp_path / "logs"
    db_path = tmp_path / "bot.db"
    monkeypatch.setenv("BOT_LOG_ROOT", str(logs_dir))
    monkeypatch.setenv("BOT_DB_PATH", str(db_path))

    storage = BotStorage(db_path)
    storage.save_snapshot(
        PersistedBotSnapshot(
            timestamp_utc="2026-08-17T02:00:00+00:00",
            cash=900.0,
            buying_power=900.0,
            equity=1_080.0,
            last_equity=1_075.0,
            daily_pnl=5.0,
            kill_switch_triggered=False,
            positions={},
            symbols=[
                PersistedSymbolSnapshot(
                    symbol="AMD",
                    price=180.0,
                    sma=175.0,
                    action="HOLD",
                    holding=True,
                    quantity=1.0,
                    market_value=180.0,
                ),
            ],
        ),
        [],
        session_id="session-without-startup-artifact",
        symbol_fingerprint="amd-only",
    )

    state = load_dashboard_state()

    assert state.startup_config is None
    assert state.has_persisted_startup_config is False
    assert state.has_persisted_snapshot is True
    assert state.snapshot.timestamp_utc == "2026-08-17T02:00:00+00:00"
    assert [item.symbol for item in state.snapshot.symbols] == ["AMD"]
    assert state.symbol_state_status == "unmatched_runtime_metadata_snapshot_preserved"
    assert state.ignored_snapshot_symbols == ()
    assert any(
        "no matching validated startup metadata" in message
        for message in state.session_warnings
    )


def test_payload_matches_session_rejects_sessionless_event_for_authoritative_session() -> None:
    sessionless_event = {
        "event": "cycle.summary",
        "ts": "2026-08-17T15:05:00+00:00",
    }

    assert _payload_matches_session(
        sessionless_event,
        session_id="session-authoritative",
        started_at_utc="2026-08-17T15:00:00+00:00",
        ended_before_utc="2026-08-17T15:15:00+00:00",
    ) is False
    assert _payload_matches_session(
        sessionless_event,
        session_id=None,
        started_at_utc="2026-08-17T15:00:00+00:00",
        ended_before_utc="2026-08-17T15:15:00+00:00",
    ) is True


@pytest.mark.parametrize("persisted_session_id", [None, "   "], ids=["null", "blank"])
def test_load_dashboard_state_preserves_snapshot_when_run_session_id_is_missing(
    monkeypatch,
    tmp_path: Path,
    persisted_session_id: str | None,
) -> None:
    logs_dir = tmp_path / "logs"
    db_path = tmp_path / "bot.db"
    monkeypatch.setenv("BOT_LOG_ROOT", str(logs_dir))
    monkeypatch.setenv("BOT_DB_PATH", str(db_path))

    storage = BotStorage(db_path)
    storage.save_snapshot(
        _persisted_snapshot_with_symbols(["AMD"]),
        [],
        session_id=persisted_session_id,
        symbol_fingerprint="amd-only",
    )
    day_dir = logs_dir / "2026-08-17"
    day_dir.mkdir(parents=True)
    payload = _valid_startup_payload(
        db_path,
        session_id="session-artifact",
        started_at_utc="2026-08-17T14:00:00+00:00",
    )
    payload.update({"symbols": ["AMD"], "symbol_count": 1})
    (day_dir / "startup_config.20260817T140000Z.json").write_text(
        json.dumps(payload),
        encoding="utf-8",
    )

    state = load_dashboard_state()

    assert state.startup_config is None
    assert state.has_persisted_startup_config is False
    assert state.has_persisted_snapshot is True
    assert state.snapshot.timestamp_utc == "2026-08-17T14:15:00+00:00"
    assert [item.symbol for item in state.snapshot.symbols] == ["AMD"]
    assert state.symbol_state_status == "unmatched_runtime_metadata_snapshot_preserved"
    assert any(
        "no matching validated startup metadata" in message
        for message in state.session_warnings
    )


@pytest.mark.parametrize(
    "invalid_case",
    [
        "string_execution_boolean",
        "string_paper_boolean",
        "missing_session",
        "missing_database_path",
        "blank_strategy",
        "naive_timestamp",
        "malformed_timestamp",
        "nonpositive_timeframe",
        "empty_symbols",
        "partially_empty_symbols",
        "mismatched_symbol_count",
        "database_mismatch",
        "vanished_database",
    ],
)
def test_startup_config_validation_rejects_invalid_or_untrusted_metadata(
    invalid_case: str,
    tmp_path: Path,
) -> None:
    db_path = tmp_path / "authoritative.db"
    BotStorage(db_path)
    payload = _valid_startup_payload(db_path)

    if invalid_case == "string_execution_boolean":
        payload["execution_enabled"] = "true"
    elif invalid_case == "string_paper_boolean":
        payload["paper"] = "true"
    elif invalid_case == "missing_session":
        payload.pop("session_id")
    elif invalid_case == "missing_database_path":
        payload.pop("db_path")
    elif invalid_case == "blank_strategy":
        payload["strategy_mode"] = " "
    elif invalid_case == "naive_timestamp":
        payload["started_at_utc"] = "2026-04-24T18:17:22"
    elif invalid_case == "malformed_timestamp":
        payload["started_at_utc"] = "not-a-timestamp"
    elif invalid_case == "nonpositive_timeframe":
        payload["bar_timeframe_minutes"] = 0
    elif invalid_case == "empty_symbols":
        payload.update({"symbols": [], "symbol_count": 0})
    elif invalid_case == "partially_empty_symbols":
        payload.update({"symbols": ["AMD", " "], "symbol_count": 1})
    elif invalid_case == "mismatched_symbol_count":
        payload["symbol_count"] = 1
    elif invalid_case == "database_mismatch":
        other_db_path = tmp_path / "other.db"
        BotStorage(other_db_path)
        payload["db_path"] = str(other_db_path.resolve())
    elif invalid_case == "vanished_database":
        payload["db_path"] = str((tmp_path / "vanished.db").resolve())
    else:  # pragma: no cover - keeps additions to the parameter list explicit
        raise AssertionError(f"unhandled case: {invalid_case}")

    assert (
        _startup_config_from_payload(
            payload,
            authoritative_database_path=db_path.resolve(),
        )
        is None
    )


def test_load_dashboard_state_reads_persisted_startup_config(monkeypatch, tmp_path: Path) -> None:
    unique = uuid4().hex
    logs_dir = tmp_path / f"test_dashboard_logs_{unique}"
    db_path = tmp_path / f"test_dashboard_state_{unique}.db"

    monkeypatch.setenv("BOT_LOG_ROOT", str(logs_dir))
    monkeypatch.setenv("BOT_DB_PATH", str(db_path))
    try:
        BotStorage(db_path)
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
                    "db_path": str(db_path),
                }
            ),
            encoding="utf-8",
        )

        state = load_dashboard_state()

        assert state.has_persisted_startup_config is True
        assert state.startup_config is not None
        assert state.startup_config.strategy_mode == "mean_reversion"
        assert state.startup_config.symbols == ["AMD", "MSFT"]
        assert state.startup_config.mean_reversion_trend_filter is False
        assert state.startup_config.mean_reversion_stop_pct == 0.0
        assert state.symbol_state_status == "cleared_runtime_symbol_state"
        assert state.storage.db_path == db_path.resolve()
    finally:
        _cleanup_dashboard_test_artifacts(logs_dir, db_path)


def test_load_dashboard_state_reports_missing_database_without_creating_it(
    monkeypatch,
    tmp_path: Path,
) -> None:
    unique = uuid4().hex
    logs_dir = tmp_path / f"test_dashboard_logs_{unique}"
    db_path = tmp_path / f"test_dashboard_state_{unique}.db"
    monkeypatch.setenv("BOT_LOG_ROOT", str(logs_dir))
    monkeypatch.setenv("BOT_DB_PATH", str(db_path))

    try:
        with pytest.raises(FileNotFoundError, match="read-only database does not exist"):
            load_dashboard_state()
        assert not db_path.exists()
    finally:
        _cleanup_dashboard_test_artifacts(logs_dir, db_path)


def test_load_dashboard_state_ignores_launches_directory(monkeypatch, tmp_path: Path) -> None:
    unique = uuid4().hex
    logs_dir = tmp_path / f"test_dashboard_logs_{unique}"
    db_path = tmp_path / f"test_dashboard_state_{unique}.db"
    monkeypatch.setenv("BOT_LOG_ROOT", str(logs_dir))
    monkeypatch.setenv("BOT_DB_PATH", str(db_path))

    try:
        BotStorage(db_path)
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
                    "db_path": str(db_path),
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
        _cleanup_dashboard_test_artifacts(logs_dir, db_path)


def test_load_dashboard_state_reports_unmatched_startup_artifact(monkeypatch, tmp_path: Path) -> None:
    unique = uuid4().hex
    logs_dir = tmp_path / f"test_dashboard_logs_{unique}"
    db_path = tmp_path / f"test_dashboard_state_{unique}.db"
    monkeypatch.setenv("BOT_LOG_ROOT", str(logs_dir))
    monkeypatch.setenv("BOT_DB_PATH", str(db_path))

    try:
        storage = BotStorage(db_path)
        day_dir = Path(logs_dir) / "2026-04-13"
        day_dir.mkdir(parents=True, exist_ok=True)
        (day_dir / "startup_config.20260413T143426Z.json").write_text(
            json.dumps(
                    {
                        "session_id": "session-amd",
                        "started_at_utc": "2026-04-13T14:30:00+00:00",
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
                    "db_path": str(db_path),
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
                                "session_id": "session-amd",
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

        assert state.startup_config is None
        assert state.has_persisted_startup_config is False
        assert state.has_persisted_snapshot is True
        assert state.snapshot.timestamp_utc == "2026-04-13T14:30:00+00:00"
        assert [item.symbol for item in state.snapshot.symbols] == ["AMD", "TSLA"]
        assert state.last_cycle_report is None
        assert state.symbol_state_status == "unmatched_runtime_metadata_snapshot_preserved"
        assert state.ignored_snapshot_symbols == ()
        assert any(
            "no matching validated startup metadata" in message
            for message in state.session_warnings
        )
    finally:
        _cleanup_dashboard_test_artifacts(logs_dir, db_path)


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
    newer_startup = PersistedStartupConfig(
        session_id="session-newer",
        started_at_utc="2026-04-13T15:00:00+00:00",
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
        latest_startup_config=newer_startup,
        snapshot=snapshot,
        has_persisted_snapshot=True,
        last_cycle_report=cycle,
        ignored_snapshot_symbols=("AMD", "TSLA"),
    )

    assert any("newer TradeOS launch started at" in message for message in warnings)
    assert any("Current config symbols" in message for message in warnings)
    assert any("Latest persisted snapshot symbols" in message for message in warnings)


def test_select_startup_config_rejects_timestamp_inference_for_unidentified_run() -> None:
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

    assert selected is None


def test_select_startup_config_uses_cycle_timestamp_when_no_persisted_run_exists() -> None:
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
        artifact_path="logs/2026-04-13/startup_config.20260413T140000Z.json",
    )
    later = PersistedStartupConfig(
        session_id="session-failed-relaunch",
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
        ],
    )
    cycle = PersistedCycleReport(
        decision_timestamp="2026-04-13T14:30:00+00:00",
        execute_orders=True,
        processed_bar=True,
        skip_reason="signals_blocked_or_skipped",
        buy_signals=1,
        sell_signals=0,
        hold_signals=0,
        error_signals=0,
        orders_submitted=0,
        observed_at_utc="2026-04-13T14:30:24+00:00",
    )

    selected = _select_startup_config_for_session(
        startup_configs=[earlier, later],
        snapshot=snapshot,
        has_persisted_snapshot=False,
        last_cycle_report=cycle,
    )

    assert selected is not None
    assert selected.session_id == "session-earlier"


def test_select_startup_config_prefers_latest_run_session_id() -> None:
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
        session_id="session-current",
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
        resync_status="RESYNC_OK",
        gate_allows_entries=True,
        gate_allows_exits=True,
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
        ],
    )

    selected = _select_startup_config_for_session(
        startup_configs=[earlier, later],
        snapshot=snapshot,
        has_persisted_snapshot=True,
        last_cycle_report=None,
        latest_run_session_id="session-current",
        latest_run_environment="paper",
    )

    assert selected is not None
    assert selected.session_id == "session-current"


def test_select_startup_config_rejects_exact_session_environment_conflict() -> None:
    selected = _select_startup_config_for_session(
        startup_configs=[_minimal_persisted_startup_config(paper=True)],
        snapshot=_persisted_snapshot_with_symbols(["AMD"]),
        has_persisted_snapshot=True,
        last_cycle_report=None,
        latest_run_session_id="session-current",
        latest_run_environment="live",
    )

    assert selected is None


@pytest.mark.parametrize(
    "persisted_symbols",
    [["AMD", "AMD"], ["AMD", " "]],
    ids=["duplicate", "blank"],
)
def test_select_startup_config_rejects_malformed_exact_session_symbols(
    persisted_symbols: list[str],
) -> None:
    selected = _select_startup_config_for_session(
        startup_configs=[_minimal_persisted_startup_config()],
        snapshot=_persisted_snapshot_with_symbols(persisted_symbols),
        has_persisted_snapshot=True,
        last_cycle_report=None,
        latest_run_session_id="session-current",
        latest_run_environment="paper",
    )

    assert selected is None


def test_load_dashboard_state_keeps_snapshot_when_startup_session_does_not_match(
    monkeypatch,
    tmp_path: Path,
) -> None:
    unique = uuid4().hex
    logs_dir = tmp_path / f"test_dashboard_logs_{unique}"
    db_path = tmp_path / f"test_dashboard_state_{unique}.db"
    monkeypatch.setenv("BOT_LOG_ROOT", str(logs_dir))
    monkeypatch.setenv("BOT_DB_PATH", str(db_path))

    try:
        storage = BotStorage(db_path)
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
                    "db_path": str(db_path),
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

        assert state.startup_config is None
        assert state.has_persisted_startup_config is False
        assert state.has_persisted_snapshot is True
        assert state.snapshot.timestamp_utc == "2026-04-13T16:45:00+00:00"
        assert [item.symbol for item in state.snapshot.symbols] == ["AMD", "QCOM"]
        assert state.symbol_state_status == "unmatched_runtime_metadata_snapshot_preserved"
        assert state.ignored_snapshot_symbols == ()
        assert any(
            "no matching validated startup metadata" in message
            for message in state.session_warnings
        )
    finally:
        _cleanup_dashboard_test_artifacts(logs_dir, db_path)


def test_load_dashboard_state_includes_session_block_reason_counts(tmp_path: Path) -> None:
    temp_root = tmp_path
    logs_dir = temp_root / f"test_logs_{uuid4().hex}"
    day_dir = logs_dir / "2026-04-13"
    day_dir.mkdir(parents=True, exist_ok=True)
    db_path = temp_root / f"test_dashboard_state_{uuid4().hex}.db"
    storage = BotStorage(db_path)

    old_log_root = os.environ.get("BOT_LOG_ROOT")
    old_db_path = os.environ.get("BOT_DB_PATH")
    os.environ["BOT_LOG_ROOT"] = str(logs_dir)
    os.environ["BOT_DB_PATH"] = str(db_path)
    try:
        (day_dir / "startup_config.json").write_text(
            json.dumps(
                {
                    "session_id": "session-amd",
                    "started_at_utc": "2026-04-13T14:30:00+00:00",
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
                    "db_path": str(db_path),
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
        ]
        storage.save_snapshot(snapshot, [], session_id="session-amd", symbol_fingerprint="amd")
        (day_dir / "risk.jsonl").write_text(
            "\n".join(
                [
                    json.dumps(
                        {
                            "event": "cycle.summary",
                            "session_id": "session-amd",
                            "decision_ts": "2026-04-13T14:30:00+00:00",
                            "execute_orders": True,
                            "processed_bar": True,
                            "skip_reason": "signals_blocked_or_skipped",
                            "buy_signals": 2,
                            "sell_signals": 1,
                            "hold_signals": 3,
                            "error_signals": 0,
                            "orders_submitted": 0,
                            "ts": "2026-04-13T14:30:24+00:00",
                        }
                    ),
                    json.dumps(
                            {
                                "event": "risk.check",
                                "session_id": "session-amd",
                                "trace": "AMD_1776090600",
                            "symbol": "AMD",
                            "action": "BUY",
                            "allowed": False,
                            "block_reason": "max_open_positions_reached",
                            "detail": "portfolio already full",
                            "ts": "2026-04-13T14:30:25+00:00",
                        }
                    ),
                    json.dumps(
                            {
                                "event": "risk.check",
                                "session_id": "session-amd",
                                "trace": "TSLA_1776090600",
                            "symbol": "TSLA",
                            "action": "BUY",
                            "allowed": False,
                            "block_reason": "max_open_positions_reached",
                            "detail": "portfolio already full",
                            "ts": "2026-04-13T14:30:25+00:00",
                        }
                    ),
                    json.dumps(
                            {
                                "event": "risk.check",
                                "session_id": "session-amd",
                                "trace": "ABBV_1776090600",
                            "symbol": "ABBV",
                            "action": "SELL",
                            "allowed": False,
                            "block_reason": "stale_live_price",
                            "detail": "quote too old",
                            "ts": "2026-04-13T14:30:26+00:00",
                        }
                    ),
                ]
            ) + "\n",
            encoding="utf-8",
        )

        state = load_dashboard_state()

        assert state.session_block_reason_counts == (
            ("max_open_positions_reached", 2),
            ("stale_live_price", 1),
        )
        assert state.latest_cycle_risk_checks[0].detail is not None
    finally:
        if old_log_root is None:
            os.environ.pop("BOT_LOG_ROOT", None)
        else:
            os.environ["BOT_LOG_ROOT"] = old_log_root
        if old_db_path is None:
            os.environ.pop("BOT_DB_PATH", None)
        else:
            os.environ["BOT_DB_PATH"] = old_db_path
        _cleanup_dashboard_test_artifacts(logs_dir, db_path)


def test_load_dashboard_state_filters_day_logs_to_selected_session(monkeypatch, tmp_path: Path) -> None:
    temp_root = tmp_path
    logs_dir = temp_root / f"test_logs_{uuid4().hex}"
    day_dir = logs_dir / "2026-04-13"
    day_dir.mkdir(parents=True, exist_ok=True)
    db_path = temp_root / f"test_dashboard_state_{uuid4().hex}.db"
    storage = BotStorage(db_path)
    _bind_paper_test_scope(storage)

    old_log_root = os.environ.get("BOT_LOG_ROOT")
    old_db_path = os.environ.get("BOT_DB_PATH")
    os.environ["BOT_LOG_ROOT"] = str(logs_dir)
    os.environ["BOT_DB_PATH"] = str(db_path)
    try:
        (day_dir / "startup_config.20260413T140000Z.json").write_text(
            json.dumps(
                {
                    "session_id": "session-old",
                    "started_at_utc": "2026-04-13T14:00:00+00:00",
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
                    "db_path": str(db_path),
                }
            ),
            encoding="utf-8",
        )
        (day_dir / "startup_config.20260413T160000Z.json").write_text(
            json.dumps(
                {
                    "session_id": "session-current",
                    "started_at_utc": "2026-04-13T16:00:00+00:00",
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
                    "db_path": str(db_path),
                }
            ),
            encoding="utf-8",
        )

        snapshot = type("Snapshot", (), {})()
        snapshot.timestamp_utc = "2026-04-13T16:15:00+00:00"
        snapshot.cash = 1.0
        snapshot.buying_power = 1.0
        snapshot.equity = 1.0
        snapshot.last_equity = 1.0
        snapshot.daily_pnl = 0.0
        snapshot.kill_switch_triggered = False
        snapshot.positions = {}
        snapshot.symbols = [
            type("Row", (), {"symbol": "AMD", "price": 1.0, "sma": 1.0, "action": "HOLD", "holding": False, "quantity": 0.0, "market_value": 0.0, "ml_probability_up": None, "ml_confidence": None, "ml_training_rows": None, "holding_minutes": None, "error": None})(),
        ]
        storage.save_snapshot(
            snapshot,
            [],
            session_id="session-current",
            symbol_fingerprint="amd",
            environment="paper",
            account_fingerprint=_TEST_ACCOUNT_FINGERPRINT,
        )

        (day_dir / "risk.jsonl").write_text(
            "\n".join(
                [
                    json.dumps(
                        {
                            "event": "cycle.summary",
                            "session_id": "session-old",
                            "decision_ts": "2026-04-13T14:15:00+00:00",
                            "execute_orders": True,
                            "processed_bar": True,
                            "skip_reason": "signals_blocked_or_skipped",
                            "buy_signals": 1,
                            "sell_signals": 0,
                            "hold_signals": 1,
                            "error_signals": 0,
                            "orders_submitted": 0,
                            "ts": "2026-04-13T14:15:20+00:00",
                        }
                    ),
                    json.dumps(
                        {
                            "event": "cycle.summary",
                            "session_id": "session-current",
                            "decision_ts": "2026-04-13T16:15:00+00:00",
                            "execute_orders": True,
                            "processed_bar": True,
                            "skip_reason": "signals_blocked_or_skipped",
                            "buy_signals": 2,
                            "sell_signals": 0,
                            "hold_signals": 1,
                            "error_signals": 0,
                            "orders_submitted": 0,
                            "ts": "2026-04-13T16:15:20+00:00",
                        }
                    ),
                    json.dumps(
                        {
                            "event": "risk.check",
                            "session_id": "session-old",
                            "trace": "AMD_1776090600",
                            "symbol": "AMD",
                            "action": "BUY",
                            "allowed": False,
                            "block_reason": "max_open_positions_reached",
                            "detail": "old session block",
                            "ts": "2026-04-13T14:15:25+00:00",
                        }
                    ),
                    json.dumps(
                        {
                            "event": "risk.check",
                            "session_id": "session-current",
                            "trace": "AMD_1776097800",
                            "symbol": "AMD",
                            "action": "BUY",
                            "allowed": False,
                            "block_reason": "stale_live_price",
                            "detail": "current session block",
                            "ts": "2026-04-13T16:15:25+00:00",
                        }
                    ),
                ]
            ) + "\n",
            encoding="utf-8",
        )
        (day_dir / "signals.jsonl").write_text(
            "\n".join(
                [
                    json.dumps(
                        {
                            "event": "signal.evaluated",
                            "session_id": "session-old",
                            "symbol": "AMD",
                            "decision_ts": "2026-04-13T14:15:00+00:00",
                            "bar_close": 100.0,
                            "sma": 101.0,
                            "deviation_pct": -1.0,
                            "trend_sma": 102.0,
                            "above_trend_sma": False,
                            "atr_pct": 1.0,
                            "atr_percentile": 50.0,
                            "volume_ratio": 1.2,
                            "trend_filter": "reject",
                            "atr_filter": "pass",
                            "window_open": True,
                            "action": "HOLD",
                            "holding": False,
                            "rejection": "trend_filter",
                            "ts": "2026-04-13T14:15:10+00:00",
                        }
                    ),
                    json.dumps(
                        {
                            "event": "signal.evaluated",
                            "session_id": "session-current",
                            "symbol": "AMD",
                            "decision_ts": "2026-04-13T16:15:00+00:00",
                            "bar_close": 99.5,
                            "sma": 101.0,
                            "deviation_pct": -1.5,
                            "trend_sma": 100.0,
                            "above_trend_sma": False,
                            "atr_pct": 1.1,
                            "atr_percentile": 55.0,
                            "volume_ratio": 1.4,
                            "trend_filter": "pass",
                            "atr_filter": "reject",
                            "window_open": True,
                            "action": "HOLD",
                            "holding": False,
                            "rejection": "atr_filter",
                            "strategy_mode": "mean_reversion",
                            "final_signal_reason": "mean_reversion_no_signal",
                            "decision_summary": "HOLD in mean reversion mode. reason: mean reversion no signal.",
                            "entry_reference_price": 98.75,
                            "exit_reference_price": 101.0,
                            "vwap": 100.2,
                            "adx": 21.4,
                            "regime_state": "bearish",
                            "ts": "2026-04-13T16:15:10+00:00",
                        }
                    ),
                ]
            ) + "\n",
            encoding="utf-8",
        )

        state = load_dashboard_state()

        assert state.startup_config is not None
        assert state.startup_config.session_id == "session-current"
        assert state.last_cycle_report is not None
        assert state.last_cycle_report.decision_timestamp == "2026-04-13T16:15:00+00:00"
        assert state.session_block_reason_counts == (("stale_live_price", 1),)
        assert state.recent_near_misses[0].rejection == "atr_filter"
        assert state.latest_signal_rows[0].strategy_mode == "mean_reversion"
        assert state.latest_signal_rows[0].final_signal_reason == "mean_reversion_no_signal"
        assert state.latest_signal_rows[0].entry_reference_price == 98.75
        assert state.latest_signal_rows[0].regime_state == "bearish"
        assert state.drift_report is not None
        assert state.drift_report.live.total_evaluated == 1
        assert state.drift_report.live.stale_bar_count == 0
        assert all(item.rejection != "trend_filter" for item in state.recent_near_misses)
        assert all("old session block" not in item.narration for item in state.recent_narrations)
    finally:
        if old_log_root is None:
            os.environ.pop("BOT_LOG_ROOT", None)
        else:
            os.environ["BOT_LOG_ROOT"] = old_log_root
        if old_db_path is None:
            os.environ.pop("BOT_DB_PATH", None)
        else:
            os.environ["BOT_DB_PATH"] = old_db_path
        _cleanup_dashboard_test_artifacts(logs_dir, db_path)



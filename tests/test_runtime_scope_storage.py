from __future__ import annotations

import sqlite3
from pathlib import Path
from types import SimpleNamespace

import pytest

from storage import (
    BotStorage,
    bind_runtime_state_scope,
    ensure_runtime_scope_schema,
    get_runtime_state_binding,
)


PAPER_FINGERPRINT = "account-fingerprint-a"
OTHER_FINGERPRINT = "account-fingerprint-b"
BOUND_AT = "2026-08-05T14:00:00+00:00"


def _snapshot(*, timestamp: str, symbol: str, quantity: float) -> SimpleNamespace:
    position = SimpleNamespace(
        avg_entry_price=100.0,
        unrealized_pl=0.0,
    )
    symbol_state = SimpleNamespace(
        symbol=symbol,
        price=100.0,
        sma=99.0,
        ml_probability_up=None,
        ml_confidence=None,
        ml_training_rows=None,
        action="HOLD",
        holding=quantity != 0,
        holding_minutes=1.0,
        quantity=quantity,
        market_value=quantity * 100.0,
        error=None,
    )
    return SimpleNamespace(
        timestamp_utc=timestamp,
        cash=10_000.0,
        buying_power=20_000.0,
        equity=10_000.0,
        last_equity=10_000.0,
        daily_pnl=0.0,
        kill_switch_triggered=False,
        positions={symbol: position},
        symbols=[symbol_state],
    )


def _order(order_id: str, symbol: str) -> SimpleNamespace:
    return SimpleNamespace(
        order_id=order_id,
        client_order_id=f"client-{order_id}",
        submitted_at="2026-08-05T14:00:00+00:00",
        filled_at="2026-08-05T14:00:01+00:00",
        symbol=symbol,
        side="buy",
        status="filled",
        qty=1.0,
        filled_qty=1.0,
        filled_avg_price=100.0,
        notional=100.0,
    )


def test_runtime_scope_schema_helper_stays_inside_callers_transaction(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "legacy-schema.db"
    with sqlite3.connect(database_path) as connection:
        connection.executescript(
            """
            CREATE TABLE bot_runs (id INTEGER PRIMARY KEY AUTOINCREMENT);
            CREATE TABLE order_history (
                order_id TEXT PRIMARY KEY,
                observed_at_utc TEXT NOT NULL
            );
            CREATE TABLE order_intents (
                intent_id TEXT PRIMARY KEY,
                account_fingerprint TEXT NOT NULL,
                created_at_utc TEXT NOT NULL
            );
            """
        )
        connection.commit()

        connection.execute("BEGIN IMMEDIATE")
        ensure_runtime_scope_schema(connection)
        assert connection.in_transaction is True
        assert "environment" in {
            row[1] for row in connection.execute("PRAGMA table_info(bot_runs)")
        }
        connection.rollback()

        assert "environment" not in {
            row[1] for row in connection.execute("PRAGMA table_info(bot_runs)")
        }
        assert connection.execute(
            "SELECT name FROM sqlite_master WHERE name = 'runtime_state_bindings'"
        ).fetchone() is None

        connection.execute("BEGIN IMMEDIATE")
        ensure_runtime_scope_schema(connection)
        inserted = bind_runtime_state_scope(
            connection,
            environment="paper",
            account_fingerprint=PAPER_FINGERPRINT,
            provenance="offline-repair-test",
            history_cutoff_utc=BOUND_AT,
            bound_at_utc=BOUND_AT,
        )
        replayed = bind_runtime_state_scope(
            connection,
            environment="paper",
            account_fingerprint=PAPER_FINGERPRINT,
            provenance="offline-repair-test",
            history_cutoff_utc=BOUND_AT,
            bound_at_utc=BOUND_AT,
        )
        assert inserted is True
        assert replayed is False
        assert connection.in_transaction is True
        connection.commit()

        binding = get_runtime_state_binding(
            connection,
            environment="paper",
            account_fingerprint=PAPER_FINGERPRINT,
        )
        assert binding is not None
        assert binding["provenance"] == "offline-repair-test"


def test_scoped_readers_require_binding_and_never_fall_back_to_legacy_rows(
    tmp_path: Path,
) -> None:
    storage = BotStorage(tmp_path / "scoped-readers.db")
    storage.save_snapshot(
        _snapshot(
            timestamp="2026-08-05T13:00:00+00:00",
            symbol="COP",
            quantity=-8.0,
        ),
        [_order("legacy-order", "COP")],
        session_id="legacy",
        symbol_fingerprint="COP",
    )

    with pytest.raises(RuntimeError, match="scope is not bound"):
        storage.get_latest_run(
            environment="paper",
            account_fingerprint=PAPER_FINGERPRINT,
        )

    storage.bind_runtime_state_scope(
        environment="paper",
        account_fingerprint=PAPER_FINGERPRINT,
        provenance="offline-repair-test",
        history_cutoff_utc=BOUND_AT,
        bound_at_utc=BOUND_AT,
    )
    assert storage.get_latest_run(
        environment="paper",
        account_fingerprint=PAPER_FINGERPRINT,
    ) is None
    assert storage.get_latest_symbol_snapshot(
        environment="paper",
        account_fingerprint=PAPER_FINGERPRINT,
    ) == []
    assert storage.get_order_history(
        environment="paper",
        account_fingerprint=PAPER_FINGERPRINT,
    ) == []

    storage.save_snapshot(
        _snapshot(
            timestamp="2026-08-05T15:00:00+00:00",
            symbol="AMD",
            quantity=2.0,
        ),
        [_order("scoped-order", "AMD")],
        session_id="scoped",
        symbol_fingerprint="AMD",
        environment="paper",
        account_fingerprint=PAPER_FINGERPRINT,
    )
    latest = storage.get_latest_run(
        environment="paper",
        account_fingerprint=PAPER_FINGERPRINT,
    )
    assert latest is not None
    assert latest["environment"] == "paper"
    assert latest["account_fingerprint"] == PAPER_FINGERPRINT
    assert [
        row["symbol"]
        for row in storage.get_latest_symbol_snapshot(
            environment="paper",
            account_fingerprint=PAPER_FINGERPRINT,
        )
    ] == ["AMD"]
    assert [
        row["order_id"]
        for row in storage.get_order_history(
            environment="paper",
            account_fingerprint=PAPER_FINGERPRINT,
        )
    ] == ["scoped-order"]

    storage.bind_runtime_state_scope(
        environment="paper",
        account_fingerprint=OTHER_FINGERPRINT,
        provenance="different-account-test",
        history_cutoff_utc=BOUND_AT,
        bound_at_utc=BOUND_AT,
    )
    assert storage.get_latest_run(
        environment="paper",
        account_fingerprint=OTHER_FINGERPRINT,
    ) is None
    assert storage.get_order_history(
        environment="paper",
        account_fingerprint=OTHER_FINGERPRINT,
    ) == []

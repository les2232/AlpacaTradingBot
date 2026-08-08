from __future__ import annotations

import json
import hashlib
import sqlite3
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path

import pytest

from automation_safety import account_fingerprint
from local_state_repair import (
    OUTCOME_A,
    OUTCOME_B,
    OUTCOME_C,
    RepairSafetyError,
    database_state_sha256,
    execute_repair,
    format_repair_plan,
    raw_file_sha256,
)
from tradeos.brokers.base import (
    BrokerAccount,
    BrokerFillActivity,
    BrokerOrder,
    BrokerPosition,
)
from tradeos.execution_lock import ExecutionLock, inspect_lock


EXPECTED_ACCOUNT = "account-current-1234"
OTHER_ACCOUNT = "account-other-9999"
LOCAL_ORDER_SPECS = (
    ("order-1", "buy", "2026-04-13T17:15:32Z"),
    ("order-2", "sell", "2026-04-13T19:56:32Z"),
    ("order-3", "buy", "2026-04-21T13:45:55Z"),
    ("order-4", "sell", "2026-04-21T17:30:50Z"),
    ("order-5", "sell", "2026-04-21T17:30:59Z"),
)
COVER_SPEC = ("cover-1", "buy", "2026-04-29T13:34:21Z")


class FakeHistoricalBroker:
    def __init__(self) -> None:
        self.account = BrokerAccount(
            cash=100_000,
            buying_power=200_000,
            equity=100_000,
            last_equity=100_000,
            account_id=EXPECTED_ACCOUNT,
            account_number="PA1234",
            status="active",
            environment="paper",
        )
        self.positions: list[BrokerPosition] = []
        self.open_orders: list[BrokerOrder] = []
        specs = [*LOCAL_ORDER_SPECS, COVER_SPEC]
        self.orders = [_broker_order(*spec) for spec in specs]
        self.fills = [_fill(*spec) for spec in specs]

    def get_account(self) -> BrokerAccount:
        return self.account

    def list_positions(self) -> list[BrokerPosition]:
        return list(self.positions)

    def list_open_orders(self) -> list[BrokerOrder]:
        return list(self.open_orders)

    def list_orders_since(self, symbol: str, start: datetime) -> list[BrokerOrder]:
        assert symbol == "COP"
        assert start.tzinfo is not None
        return list(self.orders)

    def list_fill_activities(
        self, symbol: str, start: datetime
    ) -> list[BrokerFillActivity]:
        assert symbol == "COP"
        assert start.tzinfo is not None
        return list(self.fills)


def _broker_order(
    order_id: str,
    side: str,
    filled_at: str,
    *,
    status: str = "filled",
) -> BrokerOrder:
    return BrokerOrder(
        order_id=order_id,
        submitted_at=filled_at,
        filled_at=filled_at if status == "filled" else None,
        symbol="COP",
        side=side,
        status=status,
        qty=Decimal("8"),
        filled_qty=Decimal("8") if status == "filled" else Decimal("0"),
        filled_avg_price=120.0 if status == "filled" else None,
        notional=None,
        client_order_id=f"client-{order_id}",
    )


def _fill(order_id: str, side: str, timestamp: str) -> BrokerFillActivity:
    activity_side = "sell_short" if order_id == "order-5" else side
    return BrokerFillActivity(
        activity_id=f"activity-{order_id}",
        order_id=order_id,
        transaction_time=timestamp,
        symbol="COP",
        side=activity_side,
        qty=Decimal("8"),
        price=Decimal("120"),
        cum_qty=Decimal("8"),
        leaves_qty=Decimal("0"),
        fill_type="fill",
    )


def _make_database(
    tmp_path: Path,
    *,
    quantity: int = -8,
    source_environment: str | None = None,
    source_fingerprint: str | None = None,
    ambiguous_intent: bool = False,
) -> Path:
    path = tmp_path / "bot.db"
    connection = sqlite3.connect(path)
    connection.executescript(
        """
        CREATE TABLE bot_runs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp_utc TEXT NOT NULL,
            cash REAL NOT NULL,
            buying_power REAL NOT NULL,
            equity REAL NOT NULL,
            last_equity REAL NOT NULL,
            daily_pnl REAL NOT NULL,
            kill_switch_triggered INTEGER NOT NULL,
            session_id TEXT,
            symbol_fingerprint TEXT,
            environment TEXT,
            account_fingerprint TEXT
        );
        CREATE TABLE symbol_runs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id INTEGER NOT NULL,
            symbol TEXT NOT NULL,
            price REAL,
            sma REAL,
            ml_probability_up REAL,
            ml_confidence REAL,
            ml_training_rows INTEGER,
            action TEXT NOT NULL,
            holding INTEGER NOT NULL,
            holding_minutes REAL,
            quantity REAL NOT NULL,
            market_value REAL NOT NULL,
            avg_entry_price REAL,
            unrealized_pl REAL,
            error TEXT
        );
        CREATE TABLE order_history (
            order_id TEXT PRIMARY KEY,
            observed_at_utc TEXT NOT NULL,
            submitted_at TEXT,
            filled_at TEXT,
            symbol TEXT NOT NULL,
            side TEXT NOT NULL,
            status TEXT NOT NULL,
            qty REAL,
            filled_qty REAL,
            filled_avg_price REAL,
            notional REAL,
            session_id TEXT,
            client_order_id TEXT,
            environment TEXT,
            account_fingerprint TEXT
        );
        CREATE TABLE order_intents (
            intent_id TEXT PRIMARY KEY,
            client_order_id TEXT NOT NULL UNIQUE,
            account_fingerprint TEXT NOT NULL,
            environment TEXT NOT NULL DEFAULT 'unbound',
            decision_ts TEXT NOT NULL,
            symbol TEXT NOT NULL,
            side TEXT NOT NULL,
            intent_kind TEXT NOT NULL,
            requested_qty TEXT NOT NULL,
            final_qty TEXT NOT NULL,
            status TEXT NOT NULL,
            broker_order_id TEXT,
            broker_status TEXT,
            filled_qty TEXT NOT NULL DEFAULT '0',
            submission_attempts INTEGER NOT NULL DEFAULT 0,
            error_category TEXT,
            created_at_utc TEXT NOT NULL,
            updated_at_utc TEXT NOT NULL
        );
        """
    )
    connection.execute(
        """
        INSERT INTO bot_runs (
            id, timestamp_utc, cash, buying_power, equity, last_equity,
            daily_pnl, kill_switch_triggered, session_id, symbol_fingerprint,
            environment, account_fingerprint
        ) VALUES (1, '2026-04-24T19:45:00Z', 100000, 200000, 100000,
                  100000, 0, 0, 'legacy-session', 'symbols', ?, ?)
        """,
        (source_environment, source_fingerprint),
    )
    connection.execute(
        """
        INSERT INTO symbol_runs (
            run_id, symbol, price, sma, action, holding, holding_minutes,
            quantity, market_value, avg_entry_price, unrealized_pl, error
        ) VALUES (1, 'COP', NULL, NULL, 'SNAPSHOT_ONLY', ?, 105, ?, ?, 119.13, -16.76, NULL)
        """,
        (int(quantity != 0), quantity, quantity * 120),
    )
    for order_id, side, filled_at in LOCAL_ORDER_SPECS:
        connection.execute(
            """
            INSERT INTO order_history (
                order_id, observed_at_utc, submitted_at, filled_at, symbol,
                side, status, qty, filled_qty, filled_avg_price, notional,
                session_id, client_order_id, environment, account_fingerprint
            ) VALUES (?, '2026-04-24T19:45:00Z', ?, ?, 'COP', ?, 'filled',
                      8, 8, 120, NULL, 'legacy-session', NULL, ?, ?)
            """,
            (
                order_id,
                filled_at,
                filled_at,
                side,
                source_environment,
                source_fingerprint,
            ),
        )
    if ambiguous_intent:
        connection.execute(
            """
            INSERT INTO order_intents (
                intent_id, client_order_id, account_fingerprint, environment,
                decision_ts, symbol, side, intent_kind, requested_qty,
                final_qty, status, created_at_utc, updated_at_utc
            ) VALUES ('ambiguous', 'ambiguous-client', ?, 'paper',
                      '2026-04-21T17:30:00Z', 'COP', 'sell', 'long_exit',
                      '8', '8', 'ambiguous', '2026-04-21T17:30:00Z',
                      '2026-04-21T17:30:01Z')
            """,
            (account_fingerprint(EXPECTED_ACCOUNT),),
        )
    connection.commit()
    connection.close()
    return path


def _runtime_kwargs(tmp_path: Path) -> dict[str, object]:
    lock_path = tmp_path / "automation.lock"
    return {
        "lock_factory": lambda: ExecutionLock(lock_path),
        "lock_inspector": lambda: inspect_lock(lock_path),
        "backup_root": tmp_path / "backups",
    }


def _dry_run(
    database: Path,
    broker: FakeHistoricalBroker,
    tmp_path: Path,
    **overrides: object,
):
    arguments: dict[str, object] = {
        "broker": broker,
        "database_path": database,
        "configured_paper": True,
        "expected_account_id": EXPECTED_ACCOUNT,
        "symbol": "COP",
        **_runtime_kwargs(tmp_path),
    }
    arguments.update(overrides)
    return execute_repair(**arguments)


def _apply(database: Path, broker: FakeHistoricalBroker, tmp_path: Path, plan, **overrides: object):
    arguments: dict[str, object] = {
        "broker": broker,
        "database_path": database,
        "configured_paper": True,
        "expected_account_id": EXPECTED_ACCOUNT,
        "symbol": "COP",
        "apply": True,
        "repair_id": plan.repair_id,
        "approved_db_raw_sha256": plan.database_raw_file_sha256_before,
        "approved_db_state_sha256": plan.database_state_sha256_before,
        "approved_plan_sha256": plan.repair_plan_sha256,
        "approved_evidence_sha256": plan.evidence_sha256,
        **_runtime_kwargs(tmp_path),
    }
    arguments.update(overrides)
    return execute_repair(**arguments)


def test_matching_historical_cover_selects_outcome_a(tmp_path: Path) -> None:
    database = _make_database(tmp_path)
    plan = _dry_run(database, FakeHistoricalBroker(), tmp_path)
    assert plan.outcome == OUTCOME_A
    assert plan.evidence.closing_fill_confirmed
    assert plan.local_quantity == Decimal("-8")
    assert plan.broker_quantity == 0
    assert plan.application_allowed


def test_repair_digests_are_distinct_unambiguous_and_plan_is_stable(
    tmp_path: Path,
) -> None:
    database = _make_database(tmp_path)
    broker = FakeHistoricalBroker()
    first = _dry_run(database, broker, tmp_path)
    second = _dry_run(
        database,
        broker,
        tmp_path,
        repair_id=first.repair_id,
    )

    expected_raw = hashlib.sha256(database.read_bytes()).hexdigest()
    assert raw_file_sha256(database) == expected_raw
    assert first.database_raw_file_sha256_before == expected_raw
    assert first.database_state_sha256_before == database_state_sha256(database)
    assert first.database_state_sha256_before != expected_raw
    assert first.repair_plan_sha256 == second.repair_plan_sha256
    assert first.evidence_sha256 == second.evidence_sha256

    rendered = format_repair_plan(first)
    assert "Raw database file SHA-256 before repair:" in rendered
    assert "SQLite main+WAL state SHA-256 before repair:" in rendered
    assert "Canonical repair plan SHA-256:" in rendered
    assert "Broker evidence SHA-256:" in rendered
    assert "Backup raw file SHA-256: NOT_CREATED" in rendered
    assert "Raw database file SHA-256 after repair: NOT_APPLIED" in rendered
    assert "SQLite main+WAL state SHA-256 after repair: NOT_APPLIED" in rendered


def test_apply_rejects_unapproved_canonical_plan_digest(tmp_path: Path) -> None:
    database = _make_database(tmp_path)
    broker = FakeHistoricalBroker()
    plan = _dry_run(database, broker, tmp_path)

    with pytest.raises(RepairSafetyError, match="repair plan does not match"):
        _apply(
            database,
            broker,
            tmp_path,
            plan,
            approved_plan_sha256="0" * 64,
        )

    assert raw_file_sha256(database) == plan.database_raw_file_sha256_before


def test_cover_before_latest_local_snapshot_cannot_support_outcome_a(
    tmp_path: Path,
) -> None:
    database = _make_database(tmp_path)
    broker = FakeHistoricalBroker()
    earlier_cover = ("cover-1", "buy", "2026-04-24T18:00:00Z")
    broker.orders[-1] = _broker_order(*earlier_cover)
    broker.fills[-1] = _fill(*earlier_cover)
    plan = _dry_run(database, broker, tmp_path)
    assert plan.outcome == OUTCOME_C
    assert not plan.evidence.closing_fill_confirmed
    assert not plan.application_allowed


def test_no_matching_account_history_selects_outcome_b(tmp_path: Path) -> None:
    database = _make_database(tmp_path)
    broker = FakeHistoricalBroker()
    broker.orders = []
    broker.fills = []
    plan = _dry_run(database, broker, tmp_path)
    assert plan.outcome == OUTCOME_B
    assert plan.evidence.matched_local_order_count == 0
    assert plan.application_allowed


def test_local_state_bound_to_different_account_selects_b(tmp_path: Path) -> None:
    database = _make_database(
        tmp_path,
        source_environment="paper",
        source_fingerprint=account_fingerprint(OTHER_ACCOUNT),
    )
    broker = FakeHistoricalBroker()
    broker.orders = []
    broker.fills = []
    plan = _dry_run(database, broker, tmp_path)
    assert plan.outcome == OUTCOME_B
    assert plan.application_allowed


def test_live_local_state_while_current_environment_is_paper_selects_b(tmp_path: Path) -> None:
    database = _make_database(
        tmp_path,
        source_environment="live",
        source_fingerprint=account_fingerprint(OTHER_ACCOUNT),
    )
    broker = FakeHistoricalBroker()
    broker.orders = []
    broker.fills = []
    plan = _dry_run(database, broker, tmp_path)
    assert plan.outcome == OUTCOME_B


def test_explicit_scope_mismatch_cannot_be_reclassified_as_outcome_a(
    tmp_path: Path,
) -> None:
    database = _make_database(
        tmp_path,
        source_environment="live",
        source_fingerprint=account_fingerprint(OTHER_ACCOUNT),
    )
    plan = _dry_run(database, FakeHistoricalBroker(), tmp_path)
    assert plan.evidence.exact_local_history_match
    assert plan.outcome == OUTCOME_B


def test_outcome_b_initializes_clean_scope_without_copying_foreign_state(
    tmp_path: Path,
) -> None:
    foreign_fingerprint = account_fingerprint(OTHER_ACCOUNT)
    current_fingerprint = account_fingerprint(EXPECTED_ACCOUNT)
    database = _make_database(
        tmp_path,
        source_environment="live",
        source_fingerprint=foreign_fingerprint,
    )
    connection = sqlite3.connect(database)
    connection.execute(
        """
        UPDATE bot_runs
        SET cash = 1, buying_power = 2, equity = 3, last_equity = 4,
            daily_pnl = -999, kill_switch_triggered = 1,
            symbol_fingerprint = 'foreign-symbol-set'
        WHERE id = 1
        """
    )
    connection.execute(
        """
        INSERT INTO symbol_runs (
            run_id, symbol, price, sma, ml_probability_up, ml_confidence,
            ml_training_rows, action, holding, holding_minutes, quantity,
            market_value, avg_entry_price, unrealized_pl, error
        ) VALUES (1, 'AAPL', 999, 888, 0.9, 0.8, 777, 'STALE_SIGNAL',
                  0, 55, 0, 0, 123, 456, 'foreign-state')
        """
    )
    connection.commit()
    connection.close()

    broker = FakeHistoricalBroker()
    broker.account = BrokerAccount(
        cash=101_234,
        buying_power=202_468,
        equity=101_500,
        last_equity=100_000,
        account_id=EXPECTED_ACCOUNT,
        account_number="PA1234",
        status="active",
        environment="paper",
    )
    broker.orders = []
    broker.fills = []

    plan = _dry_run(database, broker, tmp_path)
    assert plan.outcome == OUTCOME_B
    assert plan.application_allowed
    bot_change, symbol_change, binding_change, _ = plan.proposed_changes
    assert bot_change["operation"] == "INSERT_INITIALIZED"
    assert bot_change["state_source"] == "authenticated_broker_account"
    assert bot_change["source_run_id"] is None
    assert bot_change["symbol_fingerprint"] is None
    assert symbol_change["operation"] == "INSERT_INITIALIZED_FLAT"
    assert symbol_change["row_count"] == 1
    assert symbol_change["source_run_id"] is None
    assert binding_change["history_cutoff_utc"] == plan.audit_timestamp_utc

    applied = _apply(database, broker, tmp_path, plan)
    connection = sqlite3.connect(database)
    source = connection.execute(
        """
        SELECT cash, buying_power, equity, last_equity, daily_pnl,
               kill_switch_triggered, symbol_fingerprint, environment,
               account_fingerprint
        FROM bot_runs WHERE id = 1
        """
    ).fetchone()
    initialized = connection.execute(
        """
        SELECT cash, buying_power, equity, last_equity, daily_pnl,
               kill_switch_triggered, symbol_fingerprint, environment,
               account_fingerprint
        FROM bot_runs WHERE id = ?
        """,
        (applied.repaired_run_id,),
    ).fetchone()
    initialized_symbols = connection.execute(
        """
        SELECT symbol, price, sma, ml_probability_up, ml_confidence,
               ml_training_rows, action, holding, holding_minutes, quantity,
               market_value, avg_entry_price, unrealized_pl, error
        FROM symbol_runs WHERE run_id = ? ORDER BY symbol
        """,
        (applied.repaired_run_id,),
    ).fetchall()
    binding = connection.execute(
        """
        SELECT provenance, history_cutoff_utc
        FROM runtime_state_bindings
        WHERE environment = 'paper' AND account_fingerprint = ?
        """,
        (current_fingerprint,),
    ).fetchone()
    connection.close()

    assert source == (
        1.0,
        2.0,
        3.0,
        4.0,
        -999.0,
        1,
        "foreign-symbol-set",
        "live",
        foreign_fingerprint,
    )
    assert initialized == (
        101_234.0,
        202_468.0,
        101_500.0,
        100_000.0,
        1_500.0,
        0,
        None,
        "paper",
        current_fingerprint,
    )
    assert initialized_symbols == [
        (
            "COP",
            None,
            None,
            None,
            None,
            None,
            "RECONCILED_FLAT",
            0,
            None,
            0.0,
            0.0,
            None,
            None,
            None,
        )
    ]
    provenance = json.loads(binding[0])
    assert provenance["source_state_copied"] is False
    assert provenance["state_source"] == "authenticated_broker_account"
    assert binding[1] == applied.audit_timestamp_utc


def test_scoped_local_history_does_not_mix_another_account(
    tmp_path: Path,
) -> None:
    expected_fingerprint = account_fingerprint(EXPECTED_ACCOUNT)
    database = _make_database(
        tmp_path,
        source_environment="paper",
        source_fingerprint=expected_fingerprint,
    )
    connection = sqlite3.connect(database)
    connection.execute(
        """
        INSERT INTO order_history (
            order_id, observed_at_utc, submitted_at, filled_at, symbol,
            side, status, qty, filled_qty, filled_avg_price, notional,
            session_id, client_order_id, environment, account_fingerprint
        ) VALUES ('other-account-order', '2026-04-22T00:00:00Z',
                  '2026-04-22T00:00:00Z', '2026-04-22T00:00:01Z',
                  'COP', 'buy', 'filled', 8, 8, 120, NULL,
                  'other-session', NULL, 'paper', ?)
        """,
        (account_fingerprint(OTHER_ACCOUNT),),
    )
    connection.commit()
    connection.close()
    plan = _dry_run(database, FakeHistoricalBroker(), tmp_path)
    assert plan.evidence.local_order_count == 5
    assert plan.outcome == OUTCOME_A


def test_open_cop_order_appearing_after_dry_run_blocks_apply(tmp_path: Path) -> None:
    database = _make_database(tmp_path)
    broker = FakeHistoricalBroker()
    plan = _dry_run(database, broker, tmp_path)
    broker.open_orders.append(
        _broker_order("new-open", "buy", "2026-08-06T00:00:00Z", status="new")
    )
    with pytest.raises(RepairSafetyError, match="repair plan does not match"):
        _apply(database, broker, tmp_path, plan)


def test_cop_position_appearing_after_dry_run_blocks_apply(tmp_path: Path) -> None:
    database = _make_database(tmp_path)
    broker = FakeHistoricalBroker()
    plan = _dry_run(database, broker, tmp_path)
    broker.positions.append(BrokerPosition("COP", Decimal("1"), 120.0))
    with pytest.raises(RepairSafetyError, match="repair plan does not match"):
        _apply(database, broker, tmp_path, plan)


def test_local_cop_change_after_dry_run_blocks_apply(tmp_path: Path) -> None:
    database = _make_database(tmp_path)
    broker = FakeHistoricalBroker()
    plan = _dry_run(database, broker, tmp_path)
    connection = sqlite3.connect(database)
    connection.execute("UPDATE symbol_runs SET quantity = -7 WHERE symbol = 'COP'")
    connection.commit()
    connection.close()
    with pytest.raises(RepairSafetyError):
        _apply(database, broker, tmp_path, plan)


def test_ambiguous_order_intent_blocks_repair(tmp_path: Path) -> None:
    database = _make_database(tmp_path, ambiguous_intent=True)
    plan = _dry_run(database, FakeHistoricalBroker(), tmp_path)
    assert not plan.application_allowed
    assert not dict(plan.checks)["no_ambiguous_order_intent"]


def test_missing_expected_account_id_fails_closed(tmp_path: Path) -> None:
    database = _make_database(tmp_path)
    with pytest.raises(RepairSafetyError, match="not configured"):
        _dry_run(
            database,
            FakeHistoricalBroker(),
            tmp_path,
            expected_account_id=None,
        )


def test_authenticated_account_mismatch_fails_closed(tmp_path: Path) -> None:
    database = _make_database(tmp_path)
    with pytest.raises(RepairSafetyError, match="does not match"):
        _dry_run(
            database,
            FakeHistoricalBroker(),
            tmp_path,
            expected_account_id=OTHER_ACCOUNT,
        )


def test_live_configuration_fails_before_broker_use(tmp_path: Path) -> None:
    database = _make_database(tmp_path)
    with pytest.raises(RepairSafetyError, match="paper environment"):
        _dry_run(
            database,
            FakeHistoricalBroker(),
            tmp_path,
            configured_paper=False,
        )


def test_database_backup_failure_prevents_mutation(tmp_path: Path) -> None:
    database = _make_database(tmp_path)
    broker = FakeHistoricalBroker()
    plan = _dry_run(database, broker, tmp_path)
    before = database_state_sha256(database)

    def fail_backup(source: Path, destination: Path) -> Path:
        raise OSError("simulated")

    with pytest.raises(RepairSafetyError, match="backup failed"):
        _apply(database, broker, tmp_path, plan, backup_function=fail_backup)
    assert database_state_sha256(database) == before


def test_transaction_failure_rolls_back_all_database_changes(tmp_path: Path) -> None:
    database = _make_database(tmp_path)
    broker = FakeHistoricalBroker()
    plan = _dry_run(database, broker, tmp_path)
    before = database_state_sha256(database)

    def fail_before_commit(connection: sqlite3.Connection) -> None:
        raise RuntimeError("simulated transaction failure")

    with pytest.raises(RepairSafetyError, match="rolled back"):
        _apply(database, broker, tmp_path, plan, before_commit=fail_before_commit)
    assert database_state_sha256(database) == before
    connection = sqlite3.connect(database)
    assert connection.execute("SELECT COUNT(*) FROM bot_runs").fetchone()[0] == 1
    connection.close()


def test_successful_repair_preserves_historical_rows(tmp_path: Path) -> None:
    database = _make_database(tmp_path)
    broker = FakeHistoricalBroker()
    plan = _dry_run(database, broker, tmp_path)
    result = _apply(database, broker, tmp_path, plan)
    assert result.applied and result.backup_created
    assert result.backup_raw_file_sha256 == raw_file_sha256(result.backup_path)
    assert result.database_raw_file_sha256_after == raw_file_sha256(database)
    assert result.database_state_sha256_after == database_state_sha256(database)
    assert result.database_raw_file_sha256_after != plan.database_raw_file_sha256_before
    backup_connection = sqlite3.connect(result.backup_path)
    assert backup_connection.execute(
        "SELECT quantity FROM symbol_runs WHERE run_id = 1 AND symbol = 'COP'"
    ).fetchone()[0] == -8
    assert backup_connection.execute(
        "SELECT COUNT(*) FROM sqlite_master "
        "WHERE type='table' AND name='local_state_reconciliation_events'"
    ).fetchone()[0] == 0
    backup_connection.close()
    connection = sqlite3.connect(database)
    assert connection.execute("SELECT COUNT(*) FROM order_history").fetchone()[0] == 5
    assert connection.execute(
        "SELECT quantity FROM symbol_runs WHERE run_id = 1 AND symbol = 'COP'"
    ).fetchone()[0] == -8
    assert connection.execute(
        "SELECT quantity FROM symbol_runs WHERE run_id = ? AND symbol = 'COP'",
        (plan.repaired_run_id,),
    ).fetchone()[0] == 0
    connection.close()


def test_successful_repair_writes_exact_immutable_audit_record(tmp_path: Path) -> None:
    database = _make_database(tmp_path)
    broker = FakeHistoricalBroker()
    plan = _dry_run(database, broker, tmp_path)
    _apply(database, broker, tmp_path, plan)
    connection = sqlite3.connect(database)
    row = connection.execute(
        "SELECT repair_id, outcome, previous_quantity, new_quantity, evidence_sha256 "
        "FROM local_state_reconciliation_events"
    ).fetchone()
    assert row == (plan.repair_id, "A", "-8", "0", plan.evidence_sha256)
    with pytest.raises(sqlite3.IntegrityError, match="immutable"):
        connection.execute(
            "UPDATE local_state_reconciliation_events SET new_quantity = '1'"
        )
    connection.close()


def test_dry_run_performs_no_database_mutation(tmp_path: Path) -> None:
    database = _make_database(tmp_path)
    before = database.read_bytes()
    before_hash = database_state_sha256(database)
    _dry_run(database, FakeHistoricalBroker(), tmp_path)
    assert database.read_bytes() == before
    assert database_state_sha256(database) == before_hash


def test_unbound_legacy_state_is_preserved_beside_new_bound_flat_state(tmp_path: Path) -> None:
    database = _make_database(tmp_path)
    broker = FakeHistoricalBroker()
    plan = _dry_run(database, broker, tmp_path)
    _apply(database, broker, tmp_path, plan)
    connection = sqlite3.connect(database)
    legacy = connection.execute(
        "SELECT quantity FROM symbol_runs WHERE run_id = 1 AND symbol = 'COP'"
    ).fetchone()[0]
    current = connection.execute(
        """
        SELECT s.quantity
        FROM symbol_runs s JOIN bot_runs b ON b.id = s.run_id
        WHERE b.environment = 'paper' AND b.account_fingerprint = ?
          AND s.symbol = 'COP'
        ORDER BY b.id DESC LIMIT 1
        """,
        (account_fingerprint(EXPECTED_ACCOUNT),),
    ).fetchone()[0]
    binding = connection.execute(
        "SELECT provenance FROM runtime_state_bindings WHERE environment='paper' "
        "AND account_fingerprint=?",
        (account_fingerprint(EXPECTED_ACCOUNT),),
    ).fetchone()
    connection.close()
    assert legacy == -8
    assert current == 0
    assert json.loads(binding[0])["repair_id"] == plan.repair_id


def test_normal_broker_local_agreement_remains_unchanged(tmp_path: Path) -> None:
    database = _make_database(tmp_path, quantity=0)
    before = database_state_sha256(database)
    plan = _dry_run(database, FakeHistoricalBroker(), tmp_path)
    assert not plan.application_allowed
    assert not dict(plan.checks)["local_cop_quantity_is_exactly_minus_8"]
    assert database_state_sha256(database) == before

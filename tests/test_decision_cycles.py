from __future__ import annotations

import sqlite3
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

import trading_bot
from storage import (
    DECISION_CYCLE_AMBIGUOUS,
    DECISION_CYCLE_CLAIMED,
    DECISION_CYCLE_COMPLETED,
    DECISION_CYCLE_FAILED,
    DECISION_CYCLE_RUNNING,
    BotStorage,
    normalize_decision_cycle_id,
)


CYCLE_TIME = datetime(2026, 4, 23, 15, 45, tzinfo=timezone.utc)
CLAIMED_AT = "2026-04-23T15:45:05+00:00"


def _claim(
    storage: BotStorage,
    decision_ts: datetime = CYCLE_TIME,
    *,
    owner_id: str = "owner-a",
):
    return storage.claim_decision_cycle(
        decision_ts,
        owner_id=owner_id,
        claimed_at_utc=CLAIMED_AT,
    )


def _make_cycle_bot(
    db_path: Path,
    *,
    session_id: str,
    decision_timestamp: datetime = CYCLE_TIME,
    symbols: tuple[str, ...] = ("AMD",),
    action: str = "HOLD",
    kill_switch_triggered: bool = False,
    snapshot_error: Exception | None = None,
) -> trading_bot.TradeOSBot:
    bot = trading_bot.TradeOSBot.__new__(trading_bot.TradeOSBot)
    positions: dict[str, trading_bot.Position] = {}
    if action == "SELL":
        positions["AMD"] = SimpleNamespace(
            symbol="AMD",
            qty=1.0,
            market_value=100.0,
            current_price=100.0,
            avg_entry_price=99.0,
        )
    symbol_snapshots = [
        trading_bot.SymbolSnapshot(
            symbol=symbol,
            price=100.0,
            sma=99.0,
            action=action,
            holding=symbol in positions,
            quantity=1.0 if symbol in positions else 0.0,
            market_value=100.0 if symbol in positions else 0.0,
        )
        for symbol in symbols
    ]
    snapshot = trading_bot.BotSnapshot(
        timestamp_utc=normalize_decision_cycle_id(decision_timestamp),
        cash=1_000.0,
        buying_power=1_000.0,
        equity=1_000.0,
        last_equity=1_000.0,
        daily_pnl=0.0,
        kill_switch_triggered=kill_switch_triggered,
        positions=positions,
        symbols=symbol_snapshots,
    )

    bot.config = SimpleNamespace(
        symbols=list(symbols),
        strategy_mode="mean_reversion",
        max_daily_loss_usd=300.0,
        max_orders_per_minute=5,
        max_open_positions=5,
        max_usd_per_trade=500.0,
        max_live_price_age_seconds=5,
    )
    bot.storage = BotStorage(db_path)
    bot.session_id = session_id
    bot._active_symbol_fingerprint = "AMD" if len(symbols) == 1 else "AMD|MSFT"
    bot._active_decision_cycle = None
    bot._last_processed_decision_timestamp = None
    bot._last_run_cycle_report = None
    bot._bars_cache = {}
    bot._hourly_regime_cache = {}
    bot._kill_switch_warned_pcts = set()
    bot._order_submission_ts = {}
    bot._order_submission_side = {}
    bot._order_exit_reason = {}
    bot._order_signal_price = {}
    bot._order_entry_branch = {}

    bot._et_now = lambda: decision_timestamp
    bot._utc_now = lambda: decision_timestamp + timedelta(seconds=5)
    bot.get_decision_timestamp = lambda: decision_timestamp
    bot._log_skip = Mock()
    bot._is_regular_hours = Mock(return_value=True)
    bot._validate_startup_market_data = Mock()
    bot._validate_cycle_bar_freshness = Mock(return_value={})
    bot.evaluate_symbol = Mock()

    def build_snapshot(
        *,
        decision_timestamp: datetime,
        evaluate_signals: bool,
    ) -> trading_bot.BotSnapshot:
        if evaluate_signals and snapshot_error is not None:
            raise snapshot_error
        if evaluate_signals:
            for symbol in symbols:
                bot.evaluate_symbol(
                    symbol,
                    positions.get(symbol),
                    decision_timestamp=decision_timestamp,
                )
        return snapshot

    bot.build_snapshot = Mock(side_effect=build_snapshot)
    bot.record_state = Mock()
    bot.get_startup_resync_result = lambda: SimpleNamespace(
        gate_allows_entries=True,
        gate_allows_exits=True,
        status=SimpleNamespace(value="RESYNC_OK"),
    )
    bot._is_market_open = Mock(return_value=True)
    bot._is_past_flatten_deadline = Mock(return_value=False)
    bot._is_in_entry_window = Mock(return_value=True)
    bot._resync_gate_for_action = Mock(return_value=(True, ""))
    bot.get_open_orders = Mock(return_value=[])
    bot._count_recent_orders = Mock(return_value=0)
    bot._active_symbol_event_blackout = Mock(return_value=None)
    bot.get_latest_price_with_age = Mock(return_value=(100.0, 0.0))
    bot._is_price_collar_breached = Mock(return_value=False)
    bot._is_symbol_exposure_exceeded = Mock(return_value=False)
    bot._log_risk_check = Mock()
    bot.place_market_buy = Mock()
    bot.place_market_sell = Mock()
    bot.broker = SimpleNamespace(submit_market_order=Mock())

    bot.decision_cycle_events = []
    bot.blog = SimpleNamespace(
        cycle_summary=Mock(),
        decision_cycle=lambda **fields: bot.decision_cycle_events.append(fields),
        execution_error=Mock(),
        kill_switch=Mock(),
    )
    return bot


def test_cycle_identity_is_canonical_timezone_aware_utc() -> None:
    eastern = timezone(timedelta(hours=-4))

    assert normalize_decision_cycle_id(
        datetime(2026, 4, 23, 11, 45, 0, 999, tzinfo=eastern)
    ) == "2026-04-23T15:45:00+00:00"

    with pytest.raises(ValueError, match="timezone-aware"):
        normalize_decision_cycle_id(datetime(2026, 4, 23, 15, 45))


def test_first_claim_and_duplicate_survive_storage_reopen(
    tmp_path: Path,
) -> None:
    db_path = tmp_path / "cycles.db"

    first = _claim(BotStorage(db_path), owner_id="owner-a")
    duplicate = _claim(BotStorage(db_path), owner_id="owner-b")

    assert first.claimed is True
    assert first.status == "claimed"
    assert duplicate.claimed is False
    assert duplicate.reason == "duplicate_cycle"
    assert duplicate.status == "claimed"
    assert duplicate.owner_id == "owner-a"


def test_two_owners_cannot_both_claim_same_cycle(tmp_path: Path) -> None:
    db_path = tmp_path / "race.db"
    owners = ("owner-a", "owner-b")
    storages = (BotStorage(db_path), BotStorage(db_path))
    barrier = threading.Barrier(2)

    def attempt(index: int) -> tuple[str, bool, str]:
        barrier.wait(timeout=5)
        result = _claim(storages[index], owner_id=owners[index])
        return owners[index], result.claimed, result.reason

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(attempt, range(2)))

    assert sum(1 for _, claimed, _ in results if claimed) == 1
    assert sorted(reason for _, _, reason in results) == [
        "cycle_claimed",
        "duplicate_cycle",
    ]
    persisted = BotStorage(db_path).get_decision_cycle(CYCLE_TIME)
    assert persisted is not None
    assert persisted["owner_id"] in owners


def test_two_owners_can_initialize_same_legacy_schema_concurrently(
    tmp_path: Path,
) -> None:
    db_path = tmp_path / "legacy_race.db"
    with sqlite3.connect(db_path) as connection:
        connection.execute(
            """
            CREATE TABLE processed_decisions (
                decision_ts TEXT PRIMARY KEY,
                claimed_at_utc TEXT NOT NULL
            )
            """
        )

    barrier = threading.Barrier(2)

    def initialize(_: int) -> BotStorage:
        barrier.wait(timeout=5)
        return BotStorage(db_path)

    with ThreadPoolExecutor(max_workers=2) as pool:
        storages = list(pool.map(initialize, range(2)))

    assert len(storages) == 2
    with sqlite3.connect(db_path) as connection:
        columns = {
            row[1]
            for row in connection.execute(
                "PRAGMA table_info(processed_decisions)"
            ).fetchall()
        }
    assert {
        "owner_id",
        "status",
        "updated_at_utc",
        "finished_at_utc",
        "reason",
    }.issubset(columns)


def test_newer_cycle_is_accepted_and_out_of_order_cycle_is_rejected(
    tmp_path: Path,
) -> None:
    storage = BotStorage(tmp_path / "ordering.db")
    newer = CYCLE_TIME + timedelta(minutes=15)
    oldest = CYCLE_TIME - timedelta(minutes=15)
    newest = CYCLE_TIME + timedelta(minutes=30)

    assert _claim(storage, newer).claimed is True

    out_of_order = _claim(storage, oldest, owner_id="owner-b")
    accepted_newest = _claim(storage, newest, owner_id="owner-c")

    assert out_of_order.claimed is False
    assert out_of_order.reason == "out_of_order_cycle"
    assert out_of_order.latest_decision_ts == normalize_decision_cycle_id(newer)
    assert storage.get_decision_cycle(oldest) is None
    assert accepted_newest.claimed is True


def test_lifecycle_transitions_are_owner_bound_and_terminal(
    tmp_path: Path,
) -> None:
    storage = BotStorage(tmp_path / "lifecycle.db")
    assert _claim(storage).claimed is True

    assert storage.transition_decision_cycle(
        CYCLE_TIME,
        owner_id="owner-b",
        status=DECISION_CYCLE_RUNNING,
        updated_at_utc="2026-04-23T15:45:06+00:00",
        reason="wrong_owner",
    ) is False
    assert storage.transition_decision_cycle(
        CYCLE_TIME,
        owner_id="owner-a",
        status=DECISION_CYCLE_RUNNING,
        updated_at_utc="2026-04-23T15:45:06+00:00",
        reason="normal_cycle_running",
    ) is True
    assert storage.transition_decision_cycle(
        CYCLE_TIME,
        owner_id="owner-a",
        status=DECISION_CYCLE_COMPLETED,
        updated_at_utc="2026-04-23T15:45:07+00:00",
        reason="execution_completed",
    ) is True
    assert storage.transition_decision_cycle(
        CYCLE_TIME,
        owner_id="owner-a",
        status=DECISION_CYCLE_FAILED,
        updated_at_utc="2026-04-23T15:45:08+00:00",
        reason="late_failure",
    ) is False

    persisted = storage.get_decision_cycle(CYCLE_TIME)
    assert persisted is not None
    assert persisted["status"] == DECISION_CYCLE_COMPLETED
    assert persisted["reason"] == "execution_completed"
    assert persisted["finished_at_utc"] == "2026-04-23T15:45:07+00:00"


def test_legacy_claim_migrates_to_fail_closed_ambiguous(
    tmp_path: Path,
) -> None:
    db_path = tmp_path / "legacy.db"
    with sqlite3.connect(db_path) as connection:
        connection.execute(
            """
            CREATE TABLE processed_decisions (
                decision_ts TEXT PRIMARY KEY,
                claimed_at_utc TEXT NOT NULL
            )
            """
        )
        connection.execute(
            """
            INSERT INTO processed_decisions (decision_ts, claimed_at_utc)
            VALUES (?, ?)
            """,
            (normalize_decision_cycle_id(CYCLE_TIME), CLAIMED_AT),
        )

    storage = BotStorage(db_path)
    persisted = storage.get_decision_cycle(CYCLE_TIME)
    duplicate = _claim(storage, owner_id="new-owner")

    assert persisted is not None
    assert persisted["owner_id"] == "legacy"
    assert persisted["status"] == DECISION_CYCLE_AMBIGUOUS
    assert persisted["reason"] == "legacy_claim_outcome_unknown"
    assert duplicate.claimed is False
    assert duplicate.status == DECISION_CYCLE_AMBIGUOUS


@pytest.mark.parametrize(
    "transition_to",
    [None, DECISION_CYCLE_RUNNING],
    ids=["claimed", "running"],
)
def test_orphaned_claim_is_fail_closed_after_reopen(
    tmp_path: Path,
    transition_to: str | None,
) -> None:
    db_path = tmp_path / f"orphan_{transition_to or 'claimed'}.db"
    storage = BotStorage(db_path)
    assert _claim(storage, owner_id="owner-a").claimed is True
    if transition_to is not None:
        assert storage.transition_decision_cycle(
            CYCLE_TIME,
            owner_id="owner-a",
            status=transition_to,
            updated_at_utc="2026-04-23T15:45:06+00:00",
            reason="normal_cycle_running",
        ) is True

    duplicate = _claim(BotStorage(db_path), owner_id="owner-b")

    assert duplicate.claimed is False
    assert duplicate.reason == "duplicate_cycle"
    assert duplicate.status == (transition_to or "claimed")
    assert duplicate.owner_id == "owner-a"

    restarted = _make_cycle_bot(
        db_path,
        session_id="owner-c",
        action="BUY",
    )
    restarted.run_once(execute_orders=True)
    restarted.evaluate_symbol.assert_not_called()
    restarted.place_market_buy.assert_not_called()
    restarted.place_market_sell.assert_not_called()
    restarted.broker.submit_market_order.assert_not_called()


def test_multi_symbol_cycle_runs_once_and_same_process_duplicate_is_blocked(
    tmp_path: Path,
) -> None:
    db_path = tmp_path / "multi_symbol.db"
    bot = _make_cycle_bot(
        db_path,
        session_id="owner-a",
        symbols=("AMD", "MSFT"),
    )

    bot.run_once(execute_orders=True)

    assert [
        call.args[0] for call in bot.evaluate_symbol.call_args_list
    ] == ["AMD", "MSFT"]
    persisted = bot.storage.get_decision_cycle(CYCLE_TIME)
    assert persisted is not None
    assert persisted["status"] == DECISION_CYCLE_COMPLETED
    assert [event["stage"] for event in bot.decision_cycle_events] == [
        "claimed",
        "completed",
    ]

    bot.run_once(execute_orders=True)

    assert [
        call.args[0] for call in bot.evaluate_symbol.call_args_list
    ] == ["AMD", "MSFT"]
    bot.place_market_buy.assert_not_called()
    bot.place_market_sell.assert_not_called()
    bot.broker.submit_market_order.assert_not_called()
    assert bot.decision_cycle_events[-1]["stage"] == "duplicate_skipped"
    assert bot.decision_cycle_events[-1]["reason"] == "duplicate_cycle"


def test_duplicate_after_restart_cannot_reach_strategy_or_orders(
    tmp_path: Path,
) -> None:
    db_path = tmp_path / "restart.db"
    owner = _make_cycle_bot(db_path, session_id="owner-a")
    owner.run_once(execute_orders=True)

    restarted = _make_cycle_bot(
        db_path,
        session_id="owner-b",
        action="BUY",
    )
    restarted.run_once(execute_orders=True)

    restarted.evaluate_symbol.assert_not_called()
    restarted.place_market_buy.assert_not_called()
    restarted.place_market_sell.assert_not_called()
    restarted.broker.submit_market_order.assert_not_called()
    assert restarted.decision_cycle_events == [
        {
            "stage": "duplicate_skipped",
            "decision_ts": normalize_decision_cycle_id(CYCLE_TIME),
            "owner_id": "owner-b",
            "reason": "duplicate_cycle",
            "status": DECISION_CYCLE_COMPLETED,
            "existing_owner_id": "owner-a",
            "latest_decision_ts": normalize_decision_cycle_id(CYCLE_TIME),
            "symbol_fingerprint": "AMD",
            "symbol_count": 1,
        }
    ]


def test_out_of_order_run_cannot_reach_strategy_or_orders(
    tmp_path: Path,
) -> None:
    db_path = tmp_path / "out_of_order_run.db"
    newer = CYCLE_TIME + timedelta(minutes=15)
    assert _claim(
        BotStorage(db_path),
        newer,
        owner_id="newer-owner",
    ).claimed is True
    older = _make_cycle_bot(
        db_path,
        session_id="older-owner",
        decision_timestamp=CYCLE_TIME,
        action="BUY",
    )

    older.run_once(execute_orders=True)

    older.evaluate_symbol.assert_not_called()
    older.place_market_buy.assert_not_called()
    older.place_market_sell.assert_not_called()
    older.broker.submit_market_order.assert_not_called()
    assert older.decision_cycle_events[-1]["reason"] == "out_of_order_cycle"


def test_exception_before_order_capable_phase_is_failed_and_not_retried(
    tmp_path: Path,
) -> None:
    db_path = tmp_path / "failed.db"
    failed = _make_cycle_bot(
        db_path,
        session_id="owner-a",
        snapshot_error=RuntimeError("strategy construction failed"),
    )

    with pytest.raises(RuntimeError, match="strategy construction failed"):
        failed.run_once(execute_orders=True)

    persisted = failed.storage.get_decision_cycle(CYCLE_TIME)
    assert persisted is not None
    assert persisted["status"] == DECISION_CYCLE_FAILED
    assert persisted["reason"] == "exception:RuntimeError:phase=running"
    assert [event["stage"] for event in failed.decision_cycle_events] == [
        "claimed",
        "failed",
    ]

    restarted = _make_cycle_bot(db_path, session_id="owner-b")
    restarted.run_once(execute_orders=True)
    restarted.evaluate_symbol.assert_not_called()
    restarted.place_market_buy.assert_not_called()
    restarted.place_market_sell.assert_not_called()
    restarted.broker.submit_market_order.assert_not_called()


def test_submission_exception_is_ambiguous_and_not_retried(
    tmp_path: Path,
) -> None:
    db_path = tmp_path / "ambiguous.db"
    ambiguous = _make_cycle_bot(
        db_path,
        session_id="owner-a",
        action="BUY",
    )
    del ambiguous.place_market_buy
    ambiguous.broker.submit_market_order.side_effect = RuntimeError(
        "mocked uncertain broker response"
    )

    with pytest.raises(RuntimeError, match="mocked uncertain broker response"):
        ambiguous.run_once(execute_orders=True)

    persisted = ambiguous.storage.get_decision_cycle(CYCLE_TIME)
    assert persisted is not None
    assert persisted["status"] == DECISION_CYCLE_AMBIGUOUS
    assert persisted["reason"] == (
        "exception:RuntimeError:phase=order_capable"
    )
    assert ambiguous.broker.submit_market_order.call_count == 1
    assert [event["stage"] for event in ambiguous.decision_cycle_events] == [
        "claimed",
        "ambiguous",
    ]

    restarted = _make_cycle_bot(
        db_path,
        session_id="owner-b",
        action="BUY",
    )
    restarted.run_once(execute_orders=True)
    restarted.evaluate_symbol.assert_not_called()
    restarted.place_market_buy.assert_not_called()
    restarted.place_market_sell.assert_not_called()
    restarted.broker.submit_market_order.assert_not_called()


def test_partial_multi_symbol_submission_failure_blocks_cycle_replay(
    tmp_path: Path,
) -> None:
    db_path = tmp_path / "partial_multi_symbol.db"
    bot = _make_cycle_bot(
        db_path,
        session_id="owner-a",
        symbols=("AMD", "MSFT"),
        action="BUY",
    )
    bot.place_market_buy.side_effect = [
        SimpleNamespace(order_id="first-order"),
        RuntimeError("mocked second submission uncertainty"),
    ]

    with pytest.raises(
        RuntimeError,
        match="mocked second submission uncertainty",
    ):
        bot.run_once(execute_orders=True)

    assert [call.args[0] for call in bot.place_market_buy.call_args_list] == [
        "AMD",
        "MSFT",
    ]
    persisted = bot.storage.get_decision_cycle(CYCLE_TIME)
    assert persisted is not None
    assert persisted["status"] == DECISION_CYCLE_AMBIGUOUS

    restarted = _make_cycle_bot(
        db_path,
        session_id="owner-b",
        symbols=("AMD", "MSFT"),
        action="BUY",
    )
    restarted.run_once(execute_orders=True)
    restarted.evaluate_symbol.assert_not_called()
    restarted.place_market_buy.assert_not_called()
    restarted.place_market_sell.assert_not_called()
    restarted.broker.submit_market_order.assert_not_called()


def test_eod_flatten_exit_remains_available_inside_claimed_cycle(
    tmp_path: Path,
) -> None:
    bot = _make_cycle_bot(
        tmp_path / "eod.db",
        session_id="owner-a",
        action="SELL",
    )
    bot._is_past_flatten_deadline.return_value = True
    bot.flatten_positions = Mock(return_value=[])

    bot.run_once(execute_orders=True)

    bot.flatten_positions.assert_called_once()
    bot.place_market_buy.assert_not_called()
    bot.place_market_sell.assert_not_called()
    persisted = bot.storage.get_decision_cycle(CYCLE_TIME)
    assert persisted is not None
    assert persisted["status"] == DECISION_CYCLE_COMPLETED
    assert persisted["reason"] == "eod_flatten_window"


def test_kill_switch_exit_remains_available_inside_claimed_cycle(
    tmp_path: Path,
) -> None:
    bot = _make_cycle_bot(
        tmp_path / "kill_switch.db",
        session_id="owner-a",
        action="SELL",
        kill_switch_triggered=True,
    )
    bot.flatten_positions = Mock(return_value=[])

    bot.run_once(execute_orders=True)

    bot.flatten_positions.assert_called_once()
    assert bot.flatten_positions.call_args.kwargs["exit_reason"] == "kill_switch"
    bot.place_market_buy.assert_not_called()
    bot.place_market_sell.assert_not_called()
    persisted = bot.storage.get_decision_cycle(CYCLE_TIME)
    assert persisted is not None
    assert persisted["status"] == DECISION_CYCLE_COMPLETED
    assert persisted["reason"] == "kill_switch_active"


@pytest.mark.parametrize(
    "prior_status",
    [
        DECISION_CYCLE_CLAIMED,
        DECISION_CYCLE_RUNNING,
        DECISION_CYCLE_COMPLETED,
        DECISION_CYCLE_FAILED,
        DECISION_CYCLE_AMBIGUOUS,
    ],
)
def test_eod_flatten_exit_remains_available_after_cycle_rejection(
    tmp_path: Path,
    prior_status: str,
) -> None:
    db_path = tmp_path / "duplicate_eod.db"
    storage = BotStorage(db_path)
    assert _claim(storage, owner_id="owner-a").claimed is True
    if prior_status != DECISION_CYCLE_CLAIMED:
        assert storage.transition_decision_cycle(
            CYCLE_TIME,
            owner_id="owner-a",
            status=prior_status,
            updated_at_utc="2026-04-23T15:45:06+00:00",
            reason=f"test_{prior_status}",
        ) is True
    restarted = _make_cycle_bot(
        db_path,
        session_id="owner-b",
        action="SELL",
    )
    restarted._is_past_flatten_deadline.return_value = True
    restarted.flatten_positions = Mock(return_value=[])

    restarted.run_once(execute_orders=True)

    restarted.evaluate_symbol.assert_not_called()
    restarted.flatten_positions.assert_called_once()
    assert (
        restarted.flatten_positions.call_args.kwargs["exit_reason"]
        == "eod_flatten"
    )
    restarted.place_market_buy.assert_not_called()
    restarted.place_market_sell.assert_not_called()
    restarted.broker.submit_market_order.assert_not_called()
    persisted = restarted.storage.get_decision_cycle(CYCLE_TIME)
    assert persisted is not None
    assert persisted["status"] == prior_status
    assert persisted["owner_id"] == "owner-a"


def test_kill_switch_exit_remains_available_after_cycle_rejection(
    tmp_path: Path,
) -> None:
    db_path = tmp_path / "duplicate_kill.db"
    storage = BotStorage(db_path)
    assert _claim(storage, owner_id="owner-a").claimed is True
    assert storage.transition_decision_cycle(
        CYCLE_TIME,
        owner_id="owner-a",
        status=DECISION_CYCLE_AMBIGUOUS,
        updated_at_utc="2026-04-23T15:45:06+00:00",
        reason="test_ambiguous",
    ) is True
    restarted = _make_cycle_bot(
        db_path,
        session_id="owner-b",
        action="SELL",
        kill_switch_triggered=True,
    )
    restarted.flatten_positions = Mock(return_value=[])

    restarted.run_once(execute_orders=True)

    restarted.evaluate_symbol.assert_not_called()
    restarted.flatten_positions.assert_called_once()
    assert (
        restarted.flatten_positions.call_args.kwargs["exit_reason"]
        == "kill_switch"
    )
    restarted.place_market_buy.assert_not_called()
    restarted.place_market_sell.assert_not_called()
    restarted.broker.submit_market_order.assert_not_called()
    persisted = restarted.storage.get_decision_cycle(CYCLE_TIME)
    assert persisted is not None
    assert persisted["status"] == DECISION_CYCLE_AMBIGUOUS
    assert persisted["owner_id"] == "owner-a"


@pytest.mark.parametrize(
    "blocked_gate",
    ["resync", "market_clock", "regular_hours"],
)
def test_rejected_cycle_forced_exit_respects_existing_exit_gates(
    tmp_path: Path,
    blocked_gate: str,
) -> None:
    db_path = tmp_path / f"blocked_{blocked_gate}.db"
    assert _claim(BotStorage(db_path), owner_id="owner-a").claimed is True
    restarted = _make_cycle_bot(
        db_path,
        session_id="owner-b",
        action="SELL",
    )
    restarted._is_past_flatten_deadline.return_value = True
    restarted.flatten_positions = Mock(return_value=[])
    if blocked_gate == "resync":
        restarted.get_startup_resync_result = lambda: SimpleNamespace(
            gate_allows_entries=False,
            gate_allows_exits=False,
            status=SimpleNamespace(value="RESYNC_LOCKED"),
        )
    elif blocked_gate == "market_clock":
        restarted._is_market_open.return_value = False
    else:
        restarted._is_regular_hours.return_value = False

    restarted.run_once(execute_orders=True)

    restarted.evaluate_symbol.assert_not_called()
    restarted.flatten_positions.assert_not_called()
    restarted.place_market_buy.assert_not_called()
    restarted.place_market_sell.assert_not_called()
    restarted.broker.submit_market_order.assert_not_called()
    assert restarted._last_run_cycle_report is not None
    assert restarted._last_run_cycle_report.skip_reason == "duplicate_bar"


def test_out_of_order_premarket_cycle_uses_current_time_for_eod_exit_gate(
    tmp_path: Path,
) -> None:
    db_path = tmp_path / "out_of_order_eod.db"
    rejected_cycle_time = datetime(
        2026,
        4,
        23,
        12,
        0,
        tzinfo=timezone.utc,
    )
    current_time = datetime(
        2026,
        4,
        23,
        19,
        56,
        tzinfo=timezone.utc,
    )
    assert _claim(
        BotStorage(db_path),
        CYCLE_TIME,
        owner_id="newer-owner",
    ).claimed is True
    restarted = _make_cycle_bot(
        db_path,
        session_id="older-owner",
        decision_timestamp=rejected_cycle_time,
        action="SELL",
    )
    restarted._et_now = lambda: current_time
    restarted._is_regular_hours = Mock(
        side_effect=lambda timestamp: (
            trading_bot.TradeOSBot._is_regular_hours(
                restarted,
                timestamp,
            )
        )
    )
    restarted._is_past_flatten_deadline.return_value = True
    restarted.flatten_positions = Mock(return_value=[])

    restarted.run_once(execute_orders=True)

    restarted.evaluate_symbol.assert_not_called()
    restarted.flatten_positions.assert_called_once()
    restarted._is_regular_hours.assert_called_once_with(current_time)
    restarted.place_market_buy.assert_not_called()
    restarted.place_market_sell.assert_not_called()
    restarted.broker.submit_market_order.assert_not_called()


def test_forced_exit_failure_is_ambiguous_after_accepted_cycle(
    tmp_path: Path,
) -> None:
    bot = _make_cycle_bot(
        tmp_path / "failed_eod.db",
        session_id="owner-a",
        action="SELL",
    )
    bot._is_past_flatten_deadline.return_value = True
    bot.place_market_sell.side_effect = RuntimeError(
        "mocked uncertain forced-exit response"
    )

    with pytest.raises(
        RuntimeError,
        match="forced_exit_submission_failed:1",
    ):
        bot.run_once(execute_orders=True)

    bot.place_market_sell.assert_called_once()
    persisted = bot.storage.get_decision_cycle(CYCLE_TIME)
    assert persisted is not None
    assert persisted["status"] == DECISION_CYCLE_AMBIGUOUS
    assert persisted["reason"] == (
        "exception:RuntimeError:phase=order_capable"
    )

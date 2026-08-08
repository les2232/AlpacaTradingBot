from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
import threading
from types import SimpleNamespace

import pytest

from automation_safety import (
    BLOCKED_AMBIGUOUS_SUBMISSION,
    BLOCKED_LIVE_CONFIGURATION,
    BLOCKED_RECONCILIATION_FAILURE,
    BLOCKED_STATE_MISMATCH,
    BLOCKED_UNMANAGED_ORDER,
    BLOCKED_UNMANAGED_POSITION,
    ENTRY_INTENT,
    LONG_EXIT_INTENT,
    SAFE_FOR_PAPER_AUTOMATION,
    SHORT_COVER_INTENT,
    AmbiguousSubmissionError,
    OrderRequest,
    SafetyReconciler,
    SubmissionGate,
    account_fingerprint,
    available_to_cover,
    available_to_sell,
    build_intent_identity,
    decimal_text,
    format_reconciliation,
)
from storage import BotStorage
from tradeos.brokers.base import BrokerAccount, BrokerOrder, BrokerPosition
from trading_bot import TradeOSBot


EXPECTED_ACCOUNT = "paper-account-1234"
EXPECTED_FINGERPRINT = account_fingerprint(EXPECTED_ACCOUNT)
TEST_SCOPE_BOUND_AT = "2026-08-05T14:00:00+00:00"


def _bind_test_scope(storage: BotStorage) -> None:
    storage.bind_runtime_state_scope(
        environment="paper",
        account_fingerprint=EXPECTED_FINGERPRINT,
        provenance="offline-test-fixture",
        history_cutoff_utc=TEST_SCOPE_BOUND_AT,
        bound_at_utc=TEST_SCOPE_BOUND_AT,
    )


def _account(*, environment: str = "paper", status: str = "active") -> BrokerAccount:
    return BrokerAccount(
        cash=10_000.0,
        buying_power=20_000.0,
        equity=10_000.0,
        last_equity=10_000.0,
        account_id=EXPECTED_ACCOUNT,
        account_number="PA1234",
        status=status,
        environment=environment,  # type: ignore[arg-type]
        trading_blocked=False,
        account_blocked=False,
        trade_suspended_by_user=False,
    )


def _order(
    *,
    order_id: str,
    client_order_id: str,
    symbol: str,
    side: str,
    qty: str,
    filled_qty: str = "0",
    status: str = "new",
) -> BrokerOrder:
    return BrokerOrder(
        order_id=order_id,
        client_order_id=client_order_id,
        submitted_at="2026-08-05T15:00:00+00:00",
        filled_at="2026-08-05T15:00:01+00:00" if status == "filled" else None,
        symbol=symbol,
        side=side,
        status=status,
        qty=Decimal(qty),
        filled_qty=Decimal(filled_qty),
        filled_avg_price=100.0 if Decimal(filled_qty) > 0 else None,
        notional=None,
    )


class FakeBroker:
    def __init__(self) -> None:
        self.account = _account()
        self.positions: list[BrokerPosition] = []
        self.open_orders: list[BrokerOrder] = []
        self.recent_orders: list[BrokerOrder] = []
        self.submissions: list[dict[str, object]] = []
        self.timeout_after_accept = False
        self.fail_account = False
        self.read_calls = {"account": 0, "positions": 0, "open": 0, "recent": 0, "lookup": 0}

    def get_account(self) -> BrokerAccount:
        self.read_calls["account"] += 1
        if self.fail_account:
            raise TimeoutError("offline fake timeout")
        return self.account

    def list_positions(self) -> list[BrokerPosition]:
        self.read_calls["positions"] += 1
        return list(self.positions)

    def list_open_orders(self) -> list[BrokerOrder]:
        self.read_calls["open"] += 1
        return list(self.open_orders)

    def list_recent_orders(self, limit: int = 10) -> list[BrokerOrder]:
        self.read_calls["recent"] += 1
        return list(self.recent_orders[:limit])

    def get_order_by_client_order_id(self, client_order_id: str) -> BrokerOrder | None:
        self.read_calls["lookup"] += 1
        return next(
            (
                item
                for item in [*self.open_orders, *self.recent_orders]
                if item.client_order_id == client_order_id
            ),
            None,
        )

    def submit_market_order(
        self,
        *,
        symbol: str,
        qty: object,
        side: str,
        client_order_id: str,
    ) -> BrokerOrder:
        self.submissions.append(
            {
                "symbol": symbol,
                "qty": Decimal(str(qty)),
                "side": side,
                "client_order_id": client_order_id,
            }
        )
        accepted = _order(
            order_id=f"broker-{len(self.submissions)}",
            client_order_id=client_order_id,
            symbol=symbol,
            side=side,
            qty=decimal_text(qty),
        )
        if self.timeout_after_accept:
            self.recent_orders.append(accepted)
            raise TimeoutError("accepted then response was lost")
        self.open_orders.append(accepted)
        self.recent_orders.append(accepted)
        return accepted


@pytest.fixture
def storage(tmp_path: Path) -> BotStorage:
    result = BotStorage(tmp_path / "automation_safety.db")
    _bind_test_scope(result)
    return result


def _save_local_position(
    storage: BotStorage,
    symbol: str,
    qty: str,
    *,
    scoped: bool = True,
) -> None:
    quantity = Decimal(qty)
    position = BrokerPosition(
        symbol=symbol,
        qty=quantity,
        market_value=float(quantity * Decimal("100")),
        avg_entry_price=100.0,
        current_price=100.0,
    )
    item = SimpleNamespace(
        symbol=symbol,
        price=100.0,
        sma=99.0,
        ml_probability_up=None,
        ml_confidence=None,
        ml_training_rows=None,
        action="HOLD",
        holding=True,
        holding_minutes=1.0,
        quantity=float(quantity),
        market_value=position.market_value,
        error=None,
    )
    snapshot = SimpleNamespace(
        timestamp_utc=datetime.now(timezone.utc).isoformat(),
        cash=10_000.0,
        buying_power=20_000.0,
        equity=10_000.0,
        last_equity=10_000.0,
        daily_pnl=0.0,
        kill_switch_triggered=False,
        positions={symbol: position},
        symbols=[item],
    )
    scope = (
        {
            "environment": "paper",
            "account_fingerprint": EXPECTED_FINGERPRINT,
        }
        if scoped
        else {}
    )
    storage.save_snapshot(
        snapshot,
        [],
        session_id="test",
        symbol_fingerprint=symbol,
        **scope,
    )


def _seed_intent(
    storage: BotStorage,
    *,
    symbol: str,
    side: str,
    qty: str,
    filled_qty: str,
    decision_id: str,
    kind: str,
    broker_status: str = "filled",
) -> tuple[str, str]:
    fingerprint = EXPECTED_FINGERPRINT
    intent_id, client_id = build_intent_identity(
        account_fingerprint_value=fingerprint,
        decision_id=decision_id,
        symbol=symbol,
        side=side,
        intent_kind=kind,
    )
    now = datetime.now(timezone.utc).isoformat()
    storage.reserve_order_intent(
        intent_id=intent_id,
        client_order_id=client_id,
        account_fingerprint=fingerprint,
        decision_ts=decision_id,
        symbol=symbol,
        side=side,
        intent_kind=kind,
        requested_qty=qty,
        final_qty=qty,
        created_at_utc=now,
        environment="paper",
    )
    assert storage.mark_order_intent_submitting(intent_id, updated_at_utc=now)
    storage.mark_order_intent_accepted(
        intent_id,
        broker_order_id=f"seed-{intent_id[:8]}",
        broker_status=broker_status,
        filled_qty=filled_qty,
        updated_at_utc=now,
    )
    return intent_id, client_id


def _seeded_broker_order_id(storage: BotStorage, client_order_id: str) -> str:
    intent = storage.get_order_intent_by_client_id(client_order_id)
    assert intent is not None
    broker_order_id = str(intent.get("broker_order_id") or "")
    assert broker_order_id
    return broker_order_id


def _managed_position(
    storage: BotStorage,
    broker: FakeBroker,
    *,
    symbol: str,
    qty: str,
) -> None:
    side = "buy" if Decimal(qty) > 0 else "sell"
    _seed_intent(
        storage,
        symbol=symbol,
        side=side,
        qty=decimal_text(abs(Decimal(qty))),
        filled_qty=decimal_text(abs(Decimal(qty))),
        decision_id=f"seed-{symbol}",
        kind=ENTRY_INTENT if side == "buy" else "managed_short_seed",
    )
    _save_local_position(storage, symbol, qty)
    broker.positions = [
        BrokerPosition(
            symbol=symbol,
            qty=Decimal(qty),
            market_value=float(Decimal(qty) * Decimal("100")),
            avg_entry_price=100.0,
            current_price=100.0,
        )
    ]


def _reconciler(storage: BotStorage, broker: FakeBroker, *, paper: bool = True) -> SafetyReconciler:
    return SafetyReconciler(
        broker=broker,
        storage=storage,
        configured_paper=paper,
        expected_account_id=EXPECTED_ACCOUNT,
        report_path=None,
    )


def _gate(storage: BotStorage, broker: FakeBroker, *, lock_held: bool = True) -> SubmissionGate:
    return SubmissionGate(
        broker=broker,  # type: ignore[arg-type]
        storage=storage,
        reconciler=_reconciler(storage, broker),
        submission_enabled=True,
        lock_checker=lambda: lock_held,
    )


def test_submission_policy_denies_before_lock_reconciliation_or_broker(
    storage: BotStorage,
) -> None:
    broker = FakeBroker()

    class UnexpectedReconciler:
        def reconcile(self):
            raise AssertionError("reconciliation must not run when submission is denied")

    gate = SubmissionGate(
        broker=broker,  # type: ignore[arg-type]
        storage=storage,
        reconciler=UnexpectedReconciler(),  # type: ignore[arg-type]
        lock_checker=lambda: (_ for _ in ()).throw(
            AssertionError("lock inspection must not run when submission is denied")
        ),
    )

    result = gate.submit(
        OrderRequest("AMD", "buy", "1", ENTRY_INTENT, "shadow-cycle")
    )

    assert result.allowed is False
    assert result.reason == "order_submission_disabled"
    assert broker.submissions == []
    assert storage.list_order_intents() == []


def test_quantity_formulas_reserve_pending_remainders() -> None:
    assert available_to_sell("10", "0") == Decimal("10")
    assert available_to_sell("10", "6") == Decimal("4")
    assert available_to_cover("-10", "0") == Decimal("10")
    assert available_to_cover("-10", "6") == Decimal("4")


def test_reconciliation_output_is_detailed_and_account_masked(storage: BotStorage) -> None:
    broker = FakeBroker()

    report = _reconciler(storage, broker).reconcile()
    rendered = format_reconciliation(report)

    assert EXPECTED_ACCOUNT not in rendered
    assert report.account_id_masked in rendered
    assert "Broker positions: []" in rendered
    assert "Open/pending orders: []" in rendered
    assert rendered.endswith(SAFE_FOR_PAPER_AUTOMATION)


def test_normal_sell_long_ten_submits_ten(storage: BotStorage) -> None:
    broker = FakeBroker()
    _managed_position(storage, broker, symbol="COP", qty="10")

    result = _gate(storage, broker).submit(
        OrderRequest("COP", "sell", "10", LONG_EXIT_INTENT, "bar-1")
    )

    assert result.allowed is True
    assert result.final_qty == Decimal("10")
    assert broker.submissions[0]["qty"] == Decimal("10")


def test_recent_only_pending_sell_caps_second_exit_to_four(storage: BotStorage) -> None:
    broker = FakeBroker()
    _managed_position(storage, broker, symbol="COP", qty="10")
    _, client_id = _seed_intent(
        storage,
        symbol="COP",
        side="sell",
        qty="6",
        filled_qty="0",
        decision_id="older-exit",
        kind=LONG_EXIT_INTENT,
        broker_status="new",
    )
    # Model a transient list_open_orders omission after broker acceptance.
    broker.recent_orders = [
        _order(
            order_id=_seeded_broker_order_id(storage, client_id),
            client_order_id=client_id,
            symbol="COP",
            side="sell",
            qty="6",
        )
    ]

    result = _gate(storage, broker).submit(
        OrderRequest("COP", "sell", "10", LONG_EXIT_INTENT, "bar-2")
    )

    assert result.allowed is True
    assert result.final_qty == Decimal("4")
    assert broker.submissions[0]["qty"] == Decimal("4")


def test_partial_fill_reserves_remaining_and_prevents_oversell(storage: BotStorage) -> None:
    broker = FakeBroker()
    _seed_intent(
        storage,
        symbol="COP",
        side="buy",
        qty="10",
        filled_qty="10",
        decision_id="entry",
        kind=ENTRY_INTENT,
    )
    _, client_id = _seed_intent(
        storage,
        symbol="COP",
        side="sell",
        qty="10",
        filled_qty="0",
        decision_id="exit",
        kind=LONG_EXIT_INTENT,
        broker_status="partially_filled",
    )
    partial = _order(
        order_id=_seeded_broker_order_id(storage, client_id),
        client_order_id=client_id,
        symbol="COP",
        side="sell",
        qty="10",
        filled_qty="4",
        status="partially_filled",
    )
    broker.open_orders = [partial]
    broker.recent_orders = [partial]
    broker.positions = [BrokerPosition("COP", Decimal("6"), 600.0)]
    _save_local_position(storage, "COP", "6")

    result = _gate(storage, broker).submit(
        OrderRequest("COP", "sell", "6", LONG_EXIT_INTENT, "new-exit")
    )

    assert result.allowed is False
    assert result.reason == "no_safely_available_long_quantity"
    assert broker.submissions == []


@pytest.mark.parametrize("broker_qty", [None, "-8"])
def test_flat_or_short_broker_blocks_ordinary_sell(
    storage: BotStorage,
    broker_qty: str | None,
) -> None:
    broker = FakeBroker()
    _save_local_position(storage, "COP", "8")
    if broker_qty is not None:
        broker.positions = [BrokerPosition("COP", Decimal(broker_qty), -800.0)]

    result = _gate(storage, broker).submit(
        OrderRequest("COP", "sell", "8", LONG_EXIT_INTENT, "stale-local-exit")
    )

    assert result.allowed is False
    assert broker.submissions == []


def test_explicit_cover_is_capped_to_current_short(storage: BotStorage) -> None:
    broker = FakeBroker()
    _managed_position(storage, broker, symbol="COP", qty="-10")

    result = _gate(storage, broker).submit(
        OrderRequest("COP", "buy", "20", SHORT_COVER_INTENT, "cover-1", True)
    )

    assert result.allowed is True
    assert result.final_qty == Decimal("10")
    assert broker.submissions[0]["qty"] == Decimal("10")


def test_cover_requires_explicit_action(storage: BotStorage) -> None:
    broker = FakeBroker()
    _managed_position(storage, broker, symbol="COP", qty="-10")

    result = _gate(storage, broker).submit(
        OrderRequest("COP", "buy", "10", SHORT_COVER_INTENT, "cover-1", False)
    )

    assert result.allowed is False
    assert result.reason == "cover_requires_explicit_cover_action"
    assert broker.submissions == []


def test_recent_only_pending_cover_caps_new_cover_to_four(storage: BotStorage) -> None:
    broker = FakeBroker()
    _managed_position(storage, broker, symbol="COP", qty="-10")
    _, client_id = _seed_intent(
        storage,
        symbol="COP",
        side="buy",
        qty="6",
        filled_qty="0",
        decision_id="older-cover",
        kind=SHORT_COVER_INTENT,
        broker_status="new",
    )
    broker.recent_orders = [
        _order(
            order_id=_seeded_broker_order_id(storage, client_id),
            client_order_id=client_id,
            symbol="COP",
            side="buy",
            qty="6",
        )
    ]

    result = _gate(storage, broker).submit(
        OrderRequest("COP", "buy", "10", SHORT_COVER_INTENT, "cover-2", True)
    )

    assert result.allowed is True
    assert result.final_qty == Decimal("4")


def test_same_logical_order_and_dashboard_rerun_submit_once(storage: BotStorage) -> None:
    broker = FakeBroker()
    gate = _gate(storage, broker)
    request = OrderRequest("AMD", "buy", "2", ENTRY_INTENT, "dashboard-action-1")

    first = gate.submit(request)
    second = gate.submit(request)

    assert first.allowed is True
    assert second.allowed is False
    assert len(broker.submissions) == 1


def test_concurrent_real_bot_sell_path_creates_one_intent_and_one_submission(
    storage: BotStorage,
) -> None:
    broker = FakeBroker()
    _managed_position(storage, broker, symbol="AMD", qty="8")

    def make_bot() -> TradeOSBot:
        bot = TradeOSBot.__new__(TradeOSBot)
        bot.submission_gate = _gate(storage, broker)
        bot._position_first_seen_utc = {}
        bot._order_submission_ts = {}
        bot._order_submission_side = {}
        bot._order_exit_reason = {}
        bot._order_signal_price = {}
        bot.blog = SimpleNamespace(order_submitted=lambda **fields: None)
        return bot

    barrier = threading.Barrier(3)
    results: list[BrokerOrder | None] = []
    errors: list[BaseException] = []
    position = BrokerPosition(
        symbol="AMD",
        qty=Decimal("8"),
        market_value=Decimal("800"),
        avg_entry_price=100.0,
        current_price=100.0,
    )

    def submit_from_real_bot_path() -> None:
        try:
            bot = make_bot()
            barrier.wait()
            results.append(
                bot.place_market_sell(
                    "AMD",
                    position,  # type: ignore[arg-type]
                    decision_ts="2026-08-06T15:00:00+00:00",
                )
            )
        except BaseException as exc:  # pragma: no cover - asserted below
            errors.append(exc)

    threads = [
        threading.Thread(target=submit_from_real_bot_path),
        threading.Thread(target=submit_from_real_bot_path),
    ]
    for thread in threads:
        thread.start()
    barrier.wait()
    for thread in threads:
        thread.join(timeout=5)

    assert not errors
    assert all(not thread.is_alive() for thread in threads)
    assert sum(result is not None for result in results) == 1
    assert len(broker.submissions) == 1
    intents = storage.list_order_intents()
    matching = [
        intent
        for intent in intents
        if intent["decision_ts"] == "2026-08-06T15:00:00+00:00"
    ]
    assert len(matching) == 1


def test_timeout_after_acceptance_is_ambiguous_and_never_retried(storage: BotStorage) -> None:
    broker = FakeBroker()
    broker.timeout_after_accept = True
    gate = _gate(storage, broker)
    request = OrderRequest("AMD", "buy", "2", ENTRY_INTENT, "timeout-decision")

    with pytest.raises(AmbiguousSubmissionError, match="reconciliation is required"):
        gate.submit(request)
    broker.timeout_after_accept = False
    second = gate.submit(request)

    assert second.allowed is False
    assert len(broker.submissions) == 1


def test_restart_resolves_submitting_intent_without_resubmit(storage: BotStorage) -> None:
    broker = FakeBroker()
    fingerprint = account_fingerprint(EXPECTED_ACCOUNT)
    intent_id, client_id = build_intent_identity(
        account_fingerprint_value=fingerprint,
        decision_id="restart-decision",
        symbol="AMD",
        side="buy",
        intent_kind=ENTRY_INTENT,
    )
    now = datetime.now(timezone.utc).isoformat()
    storage.reserve_order_intent(
        intent_id=intent_id,
        client_order_id=client_id,
        account_fingerprint=fingerprint,
        decision_ts="restart-decision",
        symbol="AMD",
        side="buy",
        intent_kind=ENTRY_INTENT,
        requested_qty="2",
        final_qty="2",
        created_at_utc=now,
        environment="paper",
    )
    assert storage.mark_order_intent_submitting(intent_id, updated_at_utc=now)
    broker.recent_orders = [
        _order(
            order_id="accepted-before-crash",
            client_order_id=client_id,
            symbol="AMD",
            side="buy",
            qty="2",
        )
    ]

    # Reopen the durable database to model a new Python process after the
    # broker accepted the order but before local acceptance was recorded.
    restarted_storage = BotStorage(storage.db_path)
    report = _reconciler(restarted_storage, broker).reconcile()
    result = _gate(restarted_storage, broker).submit(
        OrderRequest("AMD", "buy", "2", ENTRY_INTENT, "restart-decision")
    )
    different_decision = _gate(restarted_storage, broker).submit(
        OrderRequest("AMD", "buy", "2", ENTRY_INTENT, "later-dashboard-rerun")
    )

    assert report.result == SAFE_FOR_PAPER_AUTOMATION
    assert report.pending_buy_for("AMD") == Decimal("2")
    assert result.allowed is False
    assert different_decision.allowed is False
    assert different_decision.reason == "entry_requires_no_pending_symbol_orders"
    assert broker.submissions == []


def test_unresolved_submitting_intent_blocks_every_submission(storage: BotStorage) -> None:
    broker = FakeBroker()
    fingerprint = account_fingerprint(EXPECTED_ACCOUNT)
    intent_id, client_id = build_intent_identity(
        account_fingerprint_value=fingerprint,
        decision_id="missing-at-broker",
        symbol="AMD",
        side="buy",
        intent_kind=ENTRY_INTENT,
    )
    now = datetime.now(timezone.utc).isoformat()
    storage.reserve_order_intent(
        intent_id=intent_id,
        client_order_id=client_id,
        account_fingerprint=fingerprint,
        decision_ts="missing-at-broker",
        symbol="AMD",
        side="buy",
        intent_kind=ENTRY_INTENT,
        requested_qty="2",
        final_qty="2",
        created_at_utc=now,
        environment="paper",
    )
    assert storage.mark_order_intent_submitting(intent_id, updated_at_utc=now)

    report = _reconciler(storage, broker).reconcile()

    assert report.result == BLOCKED_AMBIGUOUS_SUBMISSION
    assert report.ready is False


def test_unmanaged_position_blocks_reconciliation(storage: BotStorage) -> None:
    broker = FakeBroker()
    broker.positions = [BrokerPosition("COP", Decimal("8"), 800.0)]

    report = _reconciler(storage, broker).reconcile()
    gate_result = _gate(storage, broker).submit(
        OrderRequest("AMD", "buy", "1", ENTRY_INTENT, "blocked-unmanaged-position")
    )

    assert report.result == BLOCKED_UNMANAGED_POSITION
    assert report.unmanaged_positions[0].symbol == "COP"
    assert gate_result.allowed is False
    assert broker.submissions == []


def test_unmanaged_order_blocks_reconciliation(storage: BotStorage) -> None:
    broker = FakeBroker()
    broker.open_orders = [
        _order(
            order_id="manual-order",
            client_order_id="manual-123",
            symbol="COP",
            side="sell",
            qty="8",
        )
    ]

    report = _reconciler(storage, broker).reconcile()
    gate_result = _gate(storage, broker).submit(
        OrderRequest("AMD", "buy", "1", ENTRY_INTENT, "blocked-unmanaged-order")
    )

    assert report.result == BLOCKED_UNMANAGED_ORDER
    assert report.unmanaged_orders[0].order_id == "manual-order"
    assert gate_result.allowed is False
    assert broker.submissions == []


def test_broker_local_state_mismatch_blocks(storage: BotStorage) -> None:
    broker = FakeBroker()
    _save_local_position(storage, "COP", "8")

    report = _reconciler(storage, broker).reconcile()

    assert report.result == BLOCKED_STATE_MISMATCH
    assert report.state_mismatches


def test_unbound_legacy_open_state_blocks_without_becoming_current(
    tmp_path: Path,
) -> None:
    storage = BotStorage(tmp_path / "unbound-legacy.db")
    _save_local_position(storage, "COP", "-8", scoped=False)

    report = _reconciler(storage, FakeBroker()).reconcile()

    assert report.result == BLOCKED_STATE_MISMATCH
    assert "unbound_legacy_open_state" in report.blocking_reasons
    assert "unbound_legacy_open_state:COP" in report.state_mismatches
    assert report.local_positions == ()
    assert storage.get_runtime_state_binding(
        environment="paper",
        account_fingerprint=EXPECTED_FINGERPRINT,
    ) is None


def test_explicitly_bound_account_never_reads_unbound_legacy_snapshot(
    tmp_path: Path,
) -> None:
    storage = BotStorage(tmp_path / "bound-after-repair.db")
    _save_local_position(storage, "COP", "-8", scoped=False)
    _bind_test_scope(storage)

    report = _reconciler(storage, FakeBroker()).reconcile()

    assert report.result == SAFE_FOR_PAPER_AUTOMATION
    assert report.local_positions == ()
    assert storage.get_latest_symbol_snapshot(
        environment="paper",
        account_fingerprint=EXPECTED_FINGERPRINT,
    ) == []


def test_scoped_local_broker_and_intent_state_agree(storage: BotStorage) -> None:
    broker = FakeBroker()
    _managed_position(storage, broker, symbol="COP", qty="8")

    report = _reconciler(storage, broker).reconcile()

    assert report.result == SAFE_FOR_PAPER_AUTOMATION
    assert report.local_positions == (("COP", Decimal("8")),)
    assert report.managed_positions[0].symbol == "COP"


def test_broker_api_unavailable_fails_closed(storage: BotStorage) -> None:
    broker = FakeBroker()
    broker.fail_account = True

    report = _reconciler(storage, broker).reconcile()

    assert report.result == BLOCKED_RECONCILIATION_FAILURE
    assert report.ready is False


def test_live_config_never_queries_broker(storage: BotStorage) -> None:
    broker = FakeBroker()
    broker.fail_account = True

    report = _reconciler(storage, broker, paper=False).reconcile()

    assert report.result == BLOCKED_LIVE_CONFIGURATION
    assert all(count == 0 for count in broker.read_calls.values())


def test_unknown_broker_environment_blocks(storage: BotStorage) -> None:
    broker = FakeBroker()
    broker.account = _account(environment="unknown")

    report = _reconciler(storage, broker).reconcile()

    assert report.result == BLOCKED_LIVE_CONFIGURATION


def test_configured_paper_but_broker_reports_live_blocks(storage: BotStorage) -> None:
    broker = FakeBroker()
    broker.account = _account(environment="live")

    report = _reconciler(storage, broker).reconcile()

    assert report.result == BLOCKED_LIVE_CONFIGURATION
    assert report.ready is False


def test_wrong_expected_account_id_blocks(storage: BotStorage) -> None:
    broker = FakeBroker()
    report = SafetyReconciler(
        broker=broker,  # type: ignore[arg-type]
        storage=storage,
        configured_paper=True,
        expected_account_id="different-paper-account",
        report_path=None,
    ).reconcile()

    assert report.result == BLOCKED_STATE_MISMATCH
    assert "broker_account_does_not_match_expected_account" in report.blocking_reasons


def test_missing_expected_account_binding_blocks(storage: BotStorage) -> None:
    broker = FakeBroker()
    report = SafetyReconciler(
        broker=broker,  # type: ignore[arg-type]
        storage=storage,
        configured_paper=True,
        expected_account_id="",
        report_path=None,
    ).reconcile()

    assert report.result == BLOCKED_STATE_MISMATCH
    assert "expected_account_id_not_configured" in report.blocking_reasons


def test_second_instance_without_lock_cannot_submit(storage: BotStorage) -> None:
    broker = FakeBroker()

    result = _gate(storage, broker, lock_held=False).submit(
        OrderRequest("AMD", "buy", "2", ENTRY_INTENT, "second-instance")
    )

    assert result.allowed is False
    assert result.reason == "trading_lock_not_held"
    assert broker.submissions == []


def test_valid_normal_paper_buy_remains_available_behind_gate(storage: BotStorage) -> None:
    broker = FakeBroker()

    result = _gate(storage, broker).submit(
        OrderRequest("AMD", "buy", "2", ENTRY_INTENT, "normal-paper-entry")
    )

    assert result.allowed is True
    assert result.order is not None
    assert result.client_order_id == broker.submissions[0]["client_order_id"]
    assert str(result.client_order_id).startswith("av1-")


def test_malformed_nonterminal_broker_order_fails_closed(storage: BotStorage) -> None:
    broker = FakeBroker()
    broker.open_orders = [
        _order(
            order_id="malformed-order",
            client_order_id="manual-malformed",
            symbol="COP",
            side="sell",
            qty="8",
            status="unknown_future_status",
        )
    ]

    report = _reconciler(storage, broker).reconcile()
    result = _gate(storage, broker).submit(
        OrderRequest("AMD", "buy", "1", ENTRY_INTENT, "malformed-response")
    )

    assert report.result == BLOCKED_RECONCILIATION_FAILURE
    assert result.allowed is False
    assert broker.submissions == []


@pytest.mark.parametrize(
    ("response_qty", "response_filled", "response_status"),
    [
        ("3", "0", "new"),
        ("2", "3", "new"),
        ("2", "0", "unknown_future_status"),
    ],
)
def test_malformed_broker_acceptance_is_ambiguous(
    storage: BotStorage,
    response_qty: str,
    response_filled: str,
    response_status: str,
) -> None:
    broker = FakeBroker()

    def malformed_submit(**kwargs: object) -> BrokerOrder:
        broker.submissions.append(dict(kwargs))
        return _order(
            order_id="malformed-acceptance",
            client_order_id=str(kwargs["client_order_id"]),
            symbol=str(kwargs["symbol"]),
            side=str(kwargs["side"]),
            qty=response_qty,
            filled_qty=response_filled,
            status=response_status,
        )

    broker.submit_market_order = malformed_submit  # type: ignore[method-assign]

    with pytest.raises(AmbiguousSubmissionError, match="acceptance could not be verified"):
        _gate(storage, broker).submit(
            OrderRequest("AMD", "buy", "2", ENTRY_INTENT, "malformed-acceptance")
        )

    assert len(broker.submissions) == 1

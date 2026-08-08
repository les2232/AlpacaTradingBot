from __future__ import annotations

import asyncio
import threading
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from types import SimpleNamespace

import pytest

from tradeos.brokers.alpaca_broker import _install_default_request_timeout, AlpacaBroker


def _price_only_broker() -> AlpacaBroker:
    broker = AlpacaBroker.__new__(AlpacaBroker)
    broker._price_lock = threading.Lock()
    broker._latest_prices = {}
    broker._latest_price_times = {}
    broker._latest_trade_times = {}
    broker._stream_error = None
    broker._start_price_stream = lambda: None
    return broker


def test_install_default_request_timeout_injects_default_timeout() -> None:
    calls: list[object] = []

    def _request(method: str, url: str, **kwargs: object) -> dict[str, object]:
        calls.append(kwargs.get("timeout"))
        return {"ok": True}

    client = SimpleNamespace(_session=SimpleNamespace(request=_request))

    _install_default_request_timeout(
        client,
        connect_timeout_seconds=5.0,
        read_timeout_seconds=20.0,
    )

    result = client._session.request("GET", "https://example.test")

    assert result == {"ok": True}
    assert calls == [(5.0, 20.0)]


def test_install_default_request_timeout_preserves_explicit_timeout() -> None:
    calls: list[object] = []

    def _request(method: str, url: str, **kwargs: object) -> dict[str, object]:
        calls.append(kwargs.get("timeout"))
        return {"ok": True}

    client = SimpleNamespace(_session=SimpleNamespace(request=_request))

    _install_default_request_timeout(
        client,
        connect_timeout_seconds=5.0,
        read_timeout_seconds=20.0,
    )

    client._session.request("GET", "https://example.test", timeout=(1.0, 2.0))

    assert calls == [(1.0, 2.0)]


def test_start_price_stream_restarts_after_stream_error(monkeypatch) -> None:
    started: list[str] = []

    class _FakeStream:
        def __init__(self, api_key: str, api_secret: str, feed: object) -> None:
            self.feed = feed

        def subscribe_trades(self, handler, *symbols: str) -> None:
            started.append(f"subscribed:{','.join(symbols)}")

        def run(self) -> None:
            started.append("run")

        def stop(self) -> None:
            started.append("stop")

    real_thread = threading.Thread

    class _ImmediateThread:
        def __init__(self, target, daemon: bool = False) -> None:
            self._target = target
            self._alive = False

        def start(self) -> None:
            self._alive = True
            try:
                self._target()
            finally:
                self._alive = False

        def is_alive(self) -> bool:
            return self._alive

    monkeypatch.setattr("tradeos.brokers.alpaca_broker.StockDataStream", _FakeStream)
    monkeypatch.setattr("tradeos.brokers.alpaca_broker.threading.Thread", _ImmediateThread)

    broker = AlpacaBroker.__new__(AlpacaBroker)
    broker._api_key = "key"
    broker._api_secret = "secret"
    broker._symbols = ["AMD"]
    broker._stream_enabled = True
    broker._stream_error = None
    broker._latest_prices = {}
    broker._latest_price_times = {}
    broker._latest_trade_times = {}
    broker._price_lock = threading.Lock()
    broker._data_stream = None
    broker._stream_thread = None
    broker._stream_restart_lock = threading.Lock()
    broker._preferred_price_stream_feed = lambda: "iex"
    broker._run_price_stream = lambda: started.append("worker")

    try:
        broker._start_price_stream()
        broker._stream_error = "WinError 121"
        broker._start_price_stream()
    finally:
        monkeypatch.setattr("tradeos.brokers.alpaca_broker.threading.Thread", real_thread)

    assert started == [
        "subscribed:AMD",
        "worker",
        "stop",
        "subscribed:AMD",
        "worker",
    ]


def test_stream_trade_without_aware_exchange_timestamp_is_not_cached() -> None:
    broker = _price_only_broker()

    asyncio.run(
        broker._handle_trade(
            SimpleNamespace(symbol="AMD", price=100.0, timestamp=None)
        )
    )
    asyncio.run(
        broker._handle_trade(
            SimpleNamespace(
                symbol="MSFT",
                price=200.0,
                timestamp=datetime(2026, 8, 6, 15, 0),
            )
        )
    )

    assert broker._latest_prices == {}


def test_latest_trade_lookup_fails_closed_without_exchange_timestamp() -> None:
    broker = _price_only_broker()
    broker._latest_trade_feeds = lambda: ["iex"]
    broker._data = SimpleNamespace(
        get_stock_latest_trade=lambda request: {
            "AMD": SimpleNamespace(price=100.0, timestamp=None)
        }
    )

    with pytest.raises(RuntimeError, match="timezone-aware timestamp"):
        broker.get_latest_price_with_age("AMD")


def test_price_feed_status_includes_configured_feed_names() -> None:
    broker = AlpacaBroker.__new__(AlpacaBroker)
    broker._stream_enabled = True
    broker._stream_error = None
    broker._latest_prices = {"AMD": 123.0}
    broker._latest_price_times = {}
    broker._latest_trade_times = {}
    broker._price_lock = threading.Lock()
    broker._stream_thread = object()
    broker._symbols = ["AMD", "MSFT"]
    broker._price_stream_feed_name = "iex"
    broker._latest_data_feed_name = "sip"

    status = broker.get_price_feed_status()

    assert "stream=iex" in status
    assert "latest=sip" in status


def test_get_bars_preserves_source_timestamp_for_cycle_validation() -> None:
    naive_timestamp = datetime(2026, 4, 22, 15, 30)
    source_bar = SimpleNamespace(
        timestamp=naive_timestamp,
        open=100.0,
        high=101.0,
        low=99.0,
        close=100.5,
        volume=1_000.0,
    )
    broker = AlpacaBroker.__new__(AlpacaBroker)
    broker._data = SimpleNamespace(
        get_stock_bars=lambda request: SimpleNamespace(data={"AMD": [source_bar]})
    )

    bars = broker.get_bars(
        "AMD",
        timeframe_minutes=15,
        start=datetime(2026, 4, 22, 14, 0, tzinfo=timezone.utc),
        end=datetime(2026, 4, 22, 15, 45, tzinfo=timezone.utc),
    )

    assert bars[0].timestamp is naive_timestamp


def test_get_account_exposes_identity_status_block_flags_and_environment() -> None:
    source_account = SimpleNamespace(
        id="account-uuid",
        account_number="PA123456",
        status="AccountStatus.ACTIVE",
        cash="1000.25",
        buying_power="2000.50",
        equity="1500.75",
        last_equity="1490.00",
        trading_blocked=False,
        transfers_blocked=True,
        account_blocked=False,
        trade_suspended_by_user=False,
    )
    broker = AlpacaBroker.__new__(AlpacaBroker)
    broker._paper = True
    broker._trading = SimpleNamespace(get_account=lambda: source_account)

    account = broker.get_account()

    assert account.account_id == "account-uuid"
    assert account.account_number == "PA123456"
    assert account.status == "active"
    assert account.environment == "paper"
    assert account.trading_blocked is False
    assert account.transfers_blocked is True
    assert account.account_blocked is False
    assert account.trade_suspended_by_user is False

    broker._paper = False
    assert broker.get_account().environment == "live"


def test_list_positions_preserves_fractional_quantity_as_decimal() -> None:
    source_position = SimpleNamespace(
        symbol="COP",
        qty="10.125000001",
        market_value="1000.00",
        avg_entry_price="98.50",
        current_price="99.25",
        unrealized_pl="7.50",
    )
    broker = AlpacaBroker.__new__(AlpacaBroker)
    broker._trading = SimpleNamespace(get_all_positions=lambda: [source_position])

    positions = broker.list_positions()

    assert positions[0].qty == Decimal("10.125000001")
    assert isinstance(positions[0].qty, Decimal)


def test_order_snapshot_preserves_client_order_id_and_decimal_quantities() -> None:
    broker = AlpacaBroker.__new__(AlpacaBroker)
    source_order = SimpleNamespace(
        id="order-uuid",
        client_order_id="automation-v1-intent-123",
        submitted_at="2026-08-05T12:00:00Z",
        filled_at=None,
        symbol="COP",
        side="OrderSide.SELL",
        status="OrderStatus.PARTIALLY_FILLED",
        qty="1.125000001",
        filled_qty="0.125000001",
        filled_avg_price="99.25",
        notional=None,
        canceled_at="2026-08-05T12:05:00Z",
        expired_at="2026-08-05T12:06:00Z",
        failed_at="2026-08-05T12:07:00Z",
        replaced_at="2026-08-05T12:04:00Z",
        replaced_by="replacement-order-uuid",
        replaces="original-order-uuid",
    )

    order = broker._to_order_snapshot(source_order)

    assert order.client_order_id == "automation-v1-intent-123"
    assert order.qty == Decimal("1.125000001")
    assert order.filled_qty == Decimal("0.125000001")
    assert order.side == "sell"
    assert order.status == "partially_filled"
    assert order.canceled_at == "2026-08-05T12:05:00Z"
    assert order.expired_at == "2026-08-05T12:06:00Z"
    assert order.failed_at == "2026-08-05T12:07:00Z"
    assert order.replaced_at == "2026-08-05T12:04:00Z"
    assert order.replaced_by == "replacement-order-uuid"
    assert order.replaces == "original-order-uuid"


def test_submit_market_order_forwards_deterministic_id_and_fractional_quantity() -> None:
    submitted_requests: list[object] = []

    def _submit_order(request: object) -> SimpleNamespace:
        submitted_requests.append(request)
        return SimpleNamespace(
            id="order-uuid",
            client_order_id=getattr(request, "client_order_id"),
            submitted_at="2026-08-05T12:00:00Z",
            filled_at=None,
            symbol=getattr(request, "symbol"),
            side=getattr(request, "side"),
            status="OrderStatus.NEW",
            qty=str(getattr(request, "qty")),
            filled_qty="0",
            filled_avg_price=None,
            notional=None,
        )

    broker = AlpacaBroker.__new__(AlpacaBroker)
    broker._trading = SimpleNamespace(submit_order=_submit_order)
    broker._submission_enabled = True

    result = broker.submit_market_order(
        symbol="COP",
        qty=Decimal("0.125000001"),
        side="sell",
        client_order_id="automation-v1-intent-123",
    )

    assert len(submitted_requests) == 1
    request = submitted_requests[0]
    assert getattr(request, "client_order_id") == "automation-v1-intent-123"
    assert Decimal(str(getattr(request, "qty"))) == Decimal("0.125000001")
    assert result.client_order_id == "automation-v1-intent-123"
    assert result.qty == Decimal("0.125000001")


def test_submit_market_order_rejects_missing_id_and_invalid_quantity_before_client_call() -> None:
    def _unexpected_submit(request: object) -> None:
        raise AssertionError("submit_order must not be called")

    broker = AlpacaBroker.__new__(AlpacaBroker)
    broker._trading = SimpleNamespace(submit_order=_unexpected_submit)
    broker._submission_enabled = True

    with pytest.raises(ValueError, match="client_order_id"):
        broker.submit_market_order(
            symbol="COP",
            qty=Decimal("1"),
            side="sell",
            client_order_id="   ",
        )

    with pytest.raises(ValueError, match="positive finite"):
        broker.submit_market_order(
            symbol="COP",
            qty=Decimal("NaN"),
            side="sell",
            client_order_id="automation-v1-intent-123",
        )

    with pytest.raises(ValueError, match="order side"):
        broker.submit_market_order(
            symbol="COP",
            qty=Decimal("1"),
            side="hold",  # type: ignore[arg-type]
            client_order_id="automation-v1-intent-123",
        )


def test_submit_market_order_denied_before_sdk_call() -> None:
    def _unexpected_submit(request: object) -> None:
        raise AssertionError("submit_order must not be called")

    broker = AlpacaBroker.__new__(AlpacaBroker)
    broker._trading = SimpleNamespace(submit_order=_unexpected_submit)
    broker._submission_enabled = False

    with pytest.raises(RuntimeError, match="submission is disabled"):
        broker.submit_market_order(
            symbol="COP",
            qty=Decimal("1"),
            side="sell",
            client_order_id="automation-v1-intent-123",
        )


def test_get_order_by_client_order_id_uses_read_only_lookup_and_retains_id() -> None:
    requested_ids: list[str] = []

    def _get_order_by_client_id(client_order_id: str) -> SimpleNamespace:
        requested_ids.append(client_order_id)
        return SimpleNamespace(
            id="order-uuid",
            client_order_id=client_order_id,
            submitted_at="2026-08-05T12:00:00Z",
            filled_at=None,
            symbol="COP",
            side="OrderSide.SELL",
            status="OrderStatus.ACCEPTED",
            qty="1.25",
            filled_qty="0",
            filled_avg_price=None,
            notional=None,
        )

    broker = AlpacaBroker.__new__(AlpacaBroker)
    broker._trading = SimpleNamespace(get_order_by_client_id=_get_order_by_client_id)

    order = broker.get_order_by_client_order_id("automation-v1-intent-123")

    assert requested_ids == ["automation-v1-intent-123"]
    assert order.client_order_id == "automation-v1-intent-123"
    assert order.qty == Decimal("1.25")


def _history_order_payload(
    index: int,
    submitted_at: datetime,
    *,
    order_id: str | None = None,
) -> dict[str, object]:
    return {
        "id": order_id or f"order-{index:03d}",
        "client_order_id": f"automation-v1-intent-{index:03d}",
        "submitted_at": submitted_at.isoformat().replace("+00:00", "Z"),
        "filled_at": None,
        "canceled_at": None,
        "expired_at": None,
        "failed_at": None,
        "replaced_at": None,
        "replaced_by": None,
        "replaces": None,
        "symbol": "COP",
        "side": "sell",
        "status": "filled",
        "qty": "0.125",
        "filled_qty": "0.125",
        "filled_avg_price": "99.25",
        "notional": None,
    }


def _fill_activity_payload(
    index: int,
    transaction_time: datetime,
    *,
    activity_id: str | None = None,
    symbol: str = "COP",
    side: str = "sell",
) -> dict[str, object]:
    return {
        "activity_type": "FILL",
        "id": activity_id or f"activity-{index:03d}",
        "order_id": f"order-{index:03d}",
        "transaction_time": transaction_time.isoformat().replace("+00:00", "Z"),
        "symbol": symbol,
        "side": side,
        "qty": "0.125",
        "price": "99.25",
        "cum_qty": "0.125",
        "leaves_qty": "0",
        "type": "fill",
    }


def test_list_orders_since_uses_supported_sdk_request_and_deduplicates() -> None:
    start = datetime(2026, 8, 5, 12, 0, tzinfo=timezone.utc)
    first_page = [
        _history_order_payload(0, start + timedelta(minutes=1)),
        _history_order_payload(0, start + timedelta(minutes=1)),
        _history_order_payload(500, start + timedelta(minutes=2)),
    ]
    typed_requests: list[object] = []

    class _FakeTrading:
        def get_orders(self, *, filter: object) -> list[dict[str, object]]:
            typed_requests.append(filter)
            return first_page

    broker = AlpacaBroker.__new__(AlpacaBroker)
    broker._trading = _FakeTrading()

    orders = broker.list_orders_since("cop", start)

    assert len(orders) == 2
    assert len({order.order_id for order in orders}) == 2
    assert orders[0].order_id == "order-000"
    assert orders[-1].order_id == "order-500"
    request = typed_requests[0]
    assert getattr(request, "status").value == "all"
    assert getattr(request, "limit") == 500
    assert getattr(request, "after") == start
    assert getattr(request, "direction").value == "asc"
    assert getattr(request, "symbols") == ["COP"]


def test_list_fill_activities_paginates_filters_symbol_and_deduplicates() -> None:
    start = datetime(2026, 8, 5, 12, 0, tzinfo=timezone.utc)
    first_page = [
        _fill_activity_payload(
            index,
            start + timedelta(minutes=1, microseconds=index),
            symbol="COP" if index in {0, 99} else "AAPL",
        )
        for index in range(100)
    ]
    continuation_page = [
        _fill_activity_payload(
            99,
            start + timedelta(minutes=1, microseconds=99),
        ),
        _fill_activity_payload(100, start + timedelta(minutes=2)),
    ]
    get_calls: list[tuple[str, dict[str, object]]] = []

    class _FakeTrading:
        def get(self, path: str, params: dict[str, object]) -> list[dict[str, object]]:
            get_calls.append((path, dict(params)))
            return first_page if len(get_calls) == 1 else continuation_page

    broker = AlpacaBroker.__new__(AlpacaBroker)
    broker._trading = _FakeTrading()

    activities = broker.list_fill_activities("COP", start)

    assert [activity.activity_id for activity in activities] == [
        "activity-000",
        "activity-099",
        "activity-100",
    ]
    assert all(activity.symbol == "COP" for activity in activities)
    assert activities[-1].qty == Decimal("0.125")
    assert activities[-1].price == Decimal("99.25")
    assert get_calls[0] == (
        "/account/activities/FILL",
        {
            "after": "2026-08-05T12:00:00Z",
            "direction": "asc",
            "page_size": 100,
        },
    )
    assert get_calls[1][0] == "/account/activities/FILL"
    assert get_calls[1][1]["page_token"] == "activity-099"


def test_list_fill_activities_accepts_sell_short_execution_side() -> None:
    start = datetime(2026, 8, 5, 12, 0, tzinfo=timezone.utc)
    broker = AlpacaBroker.__new__(AlpacaBroker)
    broker._trading = SimpleNamespace(
        get=lambda path, params: [
            _fill_activity_payload(1, start + timedelta(seconds=1), side="sell_short")
        ],
    )

    activities = broker.list_fill_activities("COP", start)

    assert len(activities) == 1
    assert activities[0].side == "sell_short"


def test_history_lookups_reject_naive_start_before_any_broker_call() -> None:
    class _UnexpectedTradingCall:
        def get_orders(self, *, filter: object) -> None:
            raise AssertionError("get_orders must not be called")

        def get(self, path: str, params: dict[str, object]) -> None:
            raise AssertionError("get must not be called")

    broker = AlpacaBroker.__new__(AlpacaBroker)
    broker._trading = _UnexpectedTradingCall()
    naive_start = datetime(2026, 8, 5, 12, 0)

    with pytest.raises(ValueError, match="timezone-aware"):
        broker.list_orders_since("COP", naive_start)
    with pytest.raises(ValueError, match="timezone-aware"):
        broker.list_fill_activities("COP", naive_start)


def test_list_orders_since_fails_closed_at_sdk_page_limit() -> None:
    start = datetime(2026, 8, 5, 12, 0, tzinfo=timezone.utc)
    repeated_page = [
        _history_order_payload(index, start + timedelta(minutes=1))
        for index in range(500)
    ]

    class _FakeTrading:
        def get_orders(self, *, filter: object) -> list[dict[str, object]]:
            return repeated_page

    broker = AlpacaBroker.__new__(AlpacaBroker)
    broker._trading = _FakeTrading()

    with pytest.raises(RuntimeError, match="500-record safety limit"):
        broker.list_orders_since("COP", start)


def test_list_fill_activities_fails_closed_on_malformed_page() -> None:
    broker = AlpacaBroker.__new__(AlpacaBroker)
    broker._trading = SimpleNamespace(
        get=lambda path, params: {"activities": []},
    )

    with pytest.raises(RuntimeError, match="Malformed Alpaca FILL activity"):
        broker.list_fill_activities(
            "COP",
            datetime(2026, 8, 5, 12, 0, tzinfo=timezone.utc),
        )


def test_list_fill_activities_fails_closed_on_a_repeated_page() -> None:
    start = datetime(2026, 8, 5, 12, 0, tzinfo=timezone.utc)
    repeated_page = [
        _fill_activity_payload(
            index,
            start + timedelta(minutes=1, microseconds=index),
        )
        for index in range(100)
    ]
    broker = AlpacaBroker.__new__(AlpacaBroker)
    broker._trading = SimpleNamespace(
        get=lambda path, params: repeated_page,
    )

    with pytest.raises(RuntimeError, match="repeated a FILL activity history page"):
        broker.list_fill_activities("COP", start)

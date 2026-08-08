from __future__ import annotations

import math
import os
import threading
import time
from collections.abc import Mapping
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from typing import Any, cast

from alpaca.common.enums import Sort
from alpaca.data.enums import DataFeed
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.live.stock import StockDataStream
from alpaca.data.requests import StockBarsRequest, StockLatestTradeRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
from alpaca.trading.client import TradingClient
from alpaca.trading.enums import OrderSide as AlpacaOrderSide
from alpaca.trading.enums import QueryOrderStatus, TimeInForce
from alpaca.trading.requests import GetOrdersRequest, MarketOrderRequest

from .base import (
    BrokerAccount,
    BrokerBar,
    BrokerClient,
    BrokerFillActivity,
    BrokerOrder,
    BrokerPosition,
    OrderSide,
    QuantityInput,
)


_PRICE_STREAM_FEED_BY_NAME = {
    "iex": DataFeed.IEX,
    "sip": DataFeed.SIP,
    "delayed_sip": DataFeed.DELAYED_SIP,
}

_DEFAULT_CONNECT_TIMEOUT_SECONDS = 5.0
_DEFAULT_READ_TIMEOUT_SECONDS = 20.0
_ORDER_HISTORY_PAGE_SIZE = 500
_FILL_ACTIVITY_PAGE_SIZE = 100


def _minutes_to_timeframe(minutes: int) -> TimeFrame:
    if minutes >= 60 and minutes % 60 == 0:
        return TimeFrame(minutes // 60, TimeFrameUnit.Hour)
    return TimeFrame(minutes, TimeFrameUnit.Minute)


def _normalize_enum_text(value: Any) -> str:
    text = str(value or "").strip()
    if "." in text:
        text = text.rsplit(".", 1)[-1]
    return text.lower()


def _optional_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _optional_bool(value: Any) -> bool | None:
    return None if value is None else bool(value)


def _record_field(record: Any, field_name: str) -> Any:
    if isinstance(record, Mapping):
        return record.get(field_name)
    return getattr(record, field_name, None)


def _decimal_or_none(value: Any) -> Decimal | None:
    if value is None:
        return None
    try:
        decimal_value = value if isinstance(value, Decimal) else Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError) as exc:
        raise ValueError(f"Invalid broker quantity: {value!r}") from exc
    if not decimal_value.is_finite():
        raise ValueError(f"Invalid broker quantity: {value!r}")
    return decimal_value


def _required_decimal(value: Any, *, field_name: str) -> Decimal:
    quantity = _decimal_or_none(value)
    if quantity is None:
        raise ValueError(f"Broker response is missing {field_name}.")
    return quantity


def _normalize_submit_quantity(value: QuantityInput) -> Decimal:
    if isinstance(value, bool):
        raise ValueError("Order quantity must be a positive finite number.")
    try:
        quantity = _decimal_or_none(value)
    except ValueError as exc:
        raise ValueError("Order quantity must be a positive finite number.") from exc
    if quantity is None or quantity <= 0:
        raise ValueError("Order quantity must be a positive finite number.")
    return quantity


def _normalize_client_order_id(value: str) -> str:
    normalized = str(value or "").strip()
    if not normalized:
        raise ValueError("A deterministic client_order_id is required.")
    if len(normalized) > 48:
        raise ValueError("client_order_id must be at most 48 characters.")
    return normalized


def _normalize_history_symbol(value: str) -> str:
    normalized = str(value or "").strip().upper()
    if not normalized or "," in normalized or any(character.isspace() for character in normalized):
        raise ValueError("A single broker symbol is required for history lookup.")
    return normalized


def _parse_aware_datetime(value: Any, *, field_name: str) -> datetime:
    if isinstance(value, datetime):
        parsed = value
    elif isinstance(value, str) and value.strip():
        text = value.strip()
        if text.endswith(("Z", "z")):
            text = f"{text[:-1]}+00:00"
        try:
            parsed = datetime.fromisoformat(text)
        except ValueError as exc:
            raise ValueError(f"Invalid {field_name}: {value!r}") from exc
    else:
        raise ValueError(f"Invalid {field_name}: {value!r}")

    try:
        offset = parsed.utcoffset()
    except (OverflowError, ValueError) as exc:
        raise ValueError(f"Invalid {field_name}: {value!r}") from exc
    if parsed.tzinfo is None or offset is None:
        raise ValueError(f"{field_name} must be timezone-aware.")
    return parsed.astimezone(timezone.utc)


def _normalize_history_start(value: datetime) -> datetime:
    if not isinstance(value, datetime):
        raise ValueError("History start must be a timezone-aware datetime.")
    return _parse_aware_datetime(value, field_name="history start")


def _utc_query_text(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _history_page(value: Any, *, history_name: str, maximum_size: int) -> list[Any]:
    if not isinstance(value, list) or len(value) > maximum_size:
        raise RuntimeError(f"Malformed Alpaca {history_name} history page.")
    return value


def _read_timeout_seconds(env_name: str, default: float) -> float:
    raw = os.getenv(env_name, "").strip()
    if not raw:
        return default
    try:
        value = float(raw)
    except ValueError:
        return default
    return value if value > 0 else default


def _normalize_data_feed_name(raw_value: str | None, *, default: str = "iex") -> str:
    normalized = str(raw_value or default).strip().lower()
    return normalized if normalized in _PRICE_STREAM_FEED_BY_NAME else default


def _install_default_request_timeout(
    client: Any,
    *,
    connect_timeout_seconds: float,
    read_timeout_seconds: float,
) -> None:
    session = getattr(client, "_session", None)
    request = getattr(session, "request", None)
    if session is None or request is None:
        return

    def _request_with_timeout(method: str, url: str, **kwargs: Any) -> Any:
        kwargs.setdefault("timeout", (connect_timeout_seconds, read_timeout_seconds))
        return request(method, url, **kwargs)

    session.request = _request_with_timeout


class AlpacaBroker(BrokerClient):
    def __init__(
        self,
        *,
        api_key: str,
        api_secret: str,
        paper: bool,
        symbols: list[str],
        submission_enabled: bool = False,
        price_stream_feed: str = "iex",
        latest_data_feed: str = "iex",
    ) -> None:
        self._api_key = api_key
        self._api_secret = api_secret
        self._paper = bool(paper)
        self._submission_enabled = submission_enabled is True
        self._symbols = list(symbols)
        self._trading = TradingClient(api_key, api_secret, paper=paper)
        self._data = StockHistoricalDataClient(api_key, api_secret)
        connect_timeout_seconds = _read_timeout_seconds(
            "ALPACA_HTTP_CONNECT_TIMEOUT_SECONDS",
            _DEFAULT_CONNECT_TIMEOUT_SECONDS,
        )
        read_timeout_seconds = _read_timeout_seconds(
            "ALPACA_HTTP_READ_TIMEOUT_SECONDS",
            _DEFAULT_READ_TIMEOUT_SECONDS,
        )
        _install_default_request_timeout(
            self._trading,
            connect_timeout_seconds=connect_timeout_seconds,
            read_timeout_seconds=read_timeout_seconds,
        )
        _install_default_request_timeout(
            self._data,
            connect_timeout_seconds=connect_timeout_seconds,
            read_timeout_seconds=read_timeout_seconds,
        )
        self._price_stream_feed_name = _normalize_data_feed_name(price_stream_feed)
        self._latest_data_feed_name = _normalize_data_feed_name(latest_data_feed)
        self._stream_enabled = os.getenv("ENABLE_PRICE_STREAM", "true").lower() != "false"
        self._stream_error: str | None = None
        self._latest_prices: dict[str, float] = {}
        self._latest_price_times: dict[str, float] = {}
        self._latest_trade_times: dict[str, float] = {}
        self._price_lock = threading.Lock()
        self._data_stream: StockDataStream | None = None
        self._stream_thread: threading.Thread | None = None
        self._stream_restart_lock = threading.Lock()

    def get_account(self) -> BrokerAccount:
        account = cast(Any, self._trading).get_account()
        return BrokerAccount(
            cash=float(account.cash),
            buying_power=float(account.buying_power),
            equity=float(account.equity),
            last_equity=float(account.last_equity),
            account_id=_optional_text(getattr(account, "id", None)),
            account_number=_optional_text(getattr(account, "account_number", None)),
            status=_optional_text(_normalize_enum_text(getattr(account, "status", None))),
            environment="paper" if self._paper else "live",
            trading_blocked=_optional_bool(getattr(account, "trading_blocked", None)),
            transfers_blocked=_optional_bool(getattr(account, "transfers_blocked", None)),
            account_blocked=_optional_bool(getattr(account, "account_blocked", None)),
            trade_suspended_by_user=_optional_bool(
                getattr(account, "trade_suspended_by_user", None)
            ),
        )

    def is_market_open(self) -> bool:
        try:
            clock = cast(Any, self._trading).get_clock()
        except Exception as exc:
            raise RuntimeError("Unable to fetch Alpaca market clock.") from exc
        return bool(getattr(clock, "is_open", False))

    def _preferred_price_stream_feed(self) -> DataFeed:
        return _PRICE_STREAM_FEED_BY_NAME.get(self._price_stream_feed_name, DataFeed.IEX)

    def _latest_trade_feeds(self) -> list[DataFeed]:
        preferred = _PRICE_STREAM_FEED_BY_NAME.get(self._latest_data_feed_name, DataFeed.IEX)
        feeds: list[DataFeed] = []
        for feed in (preferred, self._preferred_price_stream_feed(), DataFeed.IEX, DataFeed.SIP):
            if feed not in feeds:
                feeds.append(feed)
        return feeds

    def get_price_feed_status(self) -> str:
        with self._price_lock:
            active_symbols = len(self._latest_prices)

        if not self._stream_enabled:
            return "live stream disabled"
        if self._stream_error:
            return f"stream error: {self._stream_error}"
        if self._stream_thread is None:
            return f"stream idle ({self._price_stream_feed_name})"
        if active_symbols == 0:
            return f"stream connecting ({self._price_stream_feed_name})"
        return (
            f"live stream active for {active_symbols}/{len(self._symbols)} symbols "
            f"(stream={self._price_stream_feed_name}, latest={self._latest_data_feed_name})"
        )

    def _start_price_stream(self) -> None:
        if not self._stream_enabled:
            return
        with self._stream_restart_lock:
            if self._stream_thread is not None and self._stream_thread.is_alive() and self._stream_error is None:
                return
            self._reset_price_stream()
            try:
                stream_feed = self._preferred_price_stream_feed()
                self._data_stream = StockDataStream(self._api_key, self._api_secret, feed=stream_feed)
                self._data_stream.subscribe_trades(self._handle_trade, *self._symbols)
                self._stream_thread = threading.Thread(target=self._run_price_stream, daemon=True)
                self._stream_thread.start()
                self._stream_error = None
            except Exception as exc:
                self._stream_error = str(exc)
                self._data_stream = None
                self._stream_thread = None

    def _reset_price_stream(self) -> None:
        stream = self._data_stream
        self._data_stream = None
        self._stream_thread = None
        if stream is None:
            return
        for method_name in ("stop", "close"):
            method = getattr(stream, method_name, None)
            if callable(method):
                try:
                    method()
                except Exception:
                    pass

    def _run_price_stream(self) -> None:
        try:
            if self._data_stream is None:
                return
            self._data_stream.run()
        except Exception as exc:
            self._stream_error = str(exc)
        finally:
            self._stream_thread = None

    async def _handle_trade(self, trade: Any) -> None:
        symbol = str(getattr(trade, "symbol", ""))
        try:
            price = float(getattr(trade, "price", None))
        except (TypeError, ValueError):
            return
        trade_timestamp = getattr(trade, "timestamp", None)
        if (
            not symbol
            or not math.isfinite(price)
            or price <= 0
            or not isinstance(trade_timestamp, datetime)
            or trade_timestamp.tzinfo is None
        ):
            return
        normalized_trade_time = trade_timestamp.astimezone(timezone.utc)

        with self._price_lock:
            self._latest_prices[symbol] = price
            self._latest_price_times[symbol] = time.time()
            self._latest_trade_times[symbol] = normalized_trade_time.timestamp()
            self._stream_error = None

    def get_latest_price_with_age(self, symbol: str) -> tuple[float, float]:
        self._start_price_stream()

        with self._price_lock:
            cached_price = self._latest_prices.get(symbol)
            cached_at = self._latest_price_times.get(symbol)
            cached_trade_at = self._latest_trade_times.get(symbol)

        if cached_price is not None and cached_at is not None and (time.time() - cached_at) <= 15:
            if cached_trade_at is not None:
                return cached_price, max(0.0, time.time() - cached_trade_at)

        last_exc: Exception | None = None
        for feed in self._latest_trade_feeds():
            try:
                request = StockLatestTradeRequest(symbol_or_symbols=symbol, feed=feed)
                latest = cast(dict[str, Any], cast(Any, self._data).get_stock_latest_trade(request))
                trade = latest[symbol]
                price = float(trade.price)
                if not math.isfinite(price) or price <= 0:
                    raise RuntimeError(
                        f"Latest trade for {symbol} has an invalid price."
                    )
                timestamp = getattr(trade, "timestamp", None)
                if not isinstance(timestamp, datetime) or timestamp.tzinfo is None:
                    raise RuntimeError(
                        f"Latest trade for {symbol} has no timezone-aware timestamp."
                    )
                trade_timestamp = timestamp.astimezone(timezone.utc)
                age_seconds = max(
                    0.0,
                    (datetime.now(timezone.utc) - trade_timestamp).total_seconds(),
                )

                with self._price_lock:
                    self._latest_prices[symbol] = price
                    self._latest_price_times[symbol] = time.time()
                    self._latest_trade_times[symbol] = time.time() - age_seconds

                return price, age_seconds
            except Exception as exc:
                last_exc = exc

        if last_exc is not None:
            raise last_exc
        raise RuntimeError(f"Could not resolve latest trade for {symbol}")

    def list_positions(self) -> list[BrokerPosition]:
        positions = cast(list[Any], cast(Any, self._trading).get_all_positions())
        return [
            BrokerPosition(
                symbol=str(position.symbol),
                qty=_required_decimal(
                    getattr(position, "qty", None),
                    field_name="position quantity",
                ),
                market_value=float(position.market_value),
                avg_entry_price=float(position.avg_entry_price)
                if getattr(position, "avg_entry_price", None) is not None
                else None,
                current_price=float(position.current_price)
                if getattr(position, "current_price", None) is not None
                else None,
                unrealized_pl=float(position.unrealized_pl)
                if getattr(position, "unrealized_pl", None) is not None
                else None,
            )
            for position in positions
        ]

    def get_bars(
        self,
        symbol: str,
        *,
        timeframe_minutes: int,
        start: datetime,
        end: datetime,
    ) -> list[BrokerBar]:
        request = StockBarsRequest(
            symbol_or_symbols=[symbol],
            timeframe=_minutes_to_timeframe(timeframe_minutes),
            start=start,
            end=end,
            feed=DataFeed.IEX,
        )
        bars_response = cast(Any, self._data).get_stock_bars(request)
        bars = cast(list[Any], bars_response.data.get(symbol, []))
        normalized: list[BrokerBar] = []
        for bar in bars:
            # Preserve the source value for the context-rich TradeOS cycle
            # validator. It rejects missing, invalid, and naive timestamps
            # before any bar can reach strategy or order execution.
            timestamp = cast(datetime, getattr(bar, "timestamp", None))
            normalized.append(
                BrokerBar(
                    timestamp=timestamp,
                    open=float(bar.open),
                    high=float(bar.high),
                    low=float(bar.low),
                    close=float(bar.close),
                    volume=float(getattr(bar, "volume", 0.0) or 0.0),
                )
            )
        return normalized

    def _to_order_snapshot(self, order: Any) -> BrokerOrder:
        submitted_at_raw = _record_field(order, "submitted_at")
        filled_at_raw = _record_field(order, "filled_at")
        canceled_at_raw = _record_field(order, "canceled_at")
        expired_at_raw = _record_field(order, "expired_at")
        failed_at_raw = _record_field(order, "failed_at")
        replaced_at_raw = _record_field(order, "replaced_at")
        filled_avg_price_raw = _record_field(order, "filled_avg_price")
        notional_raw = _record_field(order, "notional")
        return BrokerOrder(
            order_id=_optional_text(_record_field(order, "id")) or "",
            submitted_at=str(submitted_at_raw) if submitted_at_raw is not None else None,
            filled_at=str(filled_at_raw) if filled_at_raw is not None else None,
            symbol=_optional_text(_record_field(order, "symbol")) or "",
            side=_normalize_enum_text(_record_field(order, "side")),
            status=_normalize_enum_text(_record_field(order, "status")),
            qty=_decimal_or_none(_record_field(order, "qty")),
            filled_qty=_decimal_or_none(_record_field(order, "filled_qty")),
            filled_avg_price=float(filled_avg_price_raw)
            if filled_avg_price_raw is not None
            else None,
            notional=float(notional_raw) if notional_raw is not None else None,
            client_order_id=_optional_text(_record_field(order, "client_order_id")),
            canceled_at=str(canceled_at_raw) if canceled_at_raw is not None else None,
            expired_at=str(expired_at_raw) if expired_at_raw is not None else None,
            failed_at=str(failed_at_raw) if failed_at_raw is not None else None,
            replaced_at=str(replaced_at_raw) if replaced_at_raw is not None else None,
            replaced_by=_optional_text(_record_field(order, "replaced_by")),
            replaces=_optional_text(_record_field(order, "replaces")),
        )

    def _to_fill_activity(self, activity: Any) -> tuple[BrokerFillActivity, datetime]:
        activity_type = _normalize_enum_text(_record_field(activity, "activity_type"))
        activity_id = _optional_text(_record_field(activity, "id"))
        order_id = _optional_text(_record_field(activity, "order_id"))
        symbol = _optional_text(_record_field(activity, "symbol"))
        side = _normalize_enum_text(_record_field(activity, "side"))
        transaction_time_raw = _record_field(activity, "transaction_time")

        if activity_type != "fill":
            raise ValueError("Activity is not a FILL activity.")
        if not activity_id or not order_id or not symbol:
            raise ValueError("FILL activity is missing an identifier or symbol.")
        # Alpaca's activity feed can distinguish a short-opening execution as
        # ``sell_short`` even though the corresponding order side is ``sell``.
        if side not in {"buy", "sell", "sell_short"}:
            raise ValueError("FILL activity has an invalid side.")

        transaction_time = _parse_aware_datetime(
            transaction_time_raw,
            field_name="FILL transaction_time",
        )
        qty = _required_decimal(_record_field(activity, "qty"), field_name="fill quantity")
        price = _required_decimal(_record_field(activity, "price"), field_name="fill price")
        cum_qty = _decimal_or_none(_record_field(activity, "cum_qty"))
        leaves_qty = _decimal_or_none(_record_field(activity, "leaves_qty"))
        if qty <= 0 or price <= 0:
            raise ValueError("FILL activity quantity and price must be positive.")
        if (cum_qty is not None and cum_qty < 0) or (
            leaves_qty is not None and leaves_qty < 0
        ):
            raise ValueError("FILL activity cumulative quantities cannot be negative.")

        return (
            BrokerFillActivity(
                activity_id=activity_id,
                order_id=order_id,
                transaction_time=str(transaction_time_raw),
                symbol=symbol,
                side=side,
                qty=qty,
                price=price,
                cum_qty=cum_qty,
                leaves_qty=leaves_qty,
                fill_type=_optional_text(
                    _normalize_enum_text(_record_field(activity, "type"))
                ),
            ),
            transaction_time,
        )

    def list_recent_orders(self, limit: int = 10) -> list[BrokerOrder]:
        request = GetOrdersRequest(status=QueryOrderStatus.ALL, limit=limit, nested=False)
        orders = cast(list[Any], cast(Any, self._trading).get_orders(filter=request))
        return [self._to_order_snapshot(order) for order in orders]

    def list_open_orders(self) -> list[BrokerOrder]:
        request = GetOrdersRequest(status=QueryOrderStatus.OPEN, nested=False)
        orders = cast(list[Any], cast(Any, self._trading).get_orders(filter=request))
        return [self._to_order_snapshot(order) for order in orders]

    def get_order_by_client_order_id(self, client_order_id: str) -> BrokerOrder:
        normalized_client_order_id = _normalize_client_order_id(client_order_id)
        order = cast(
            Any,
            cast(Any, self._trading).get_order_by_client_id(normalized_client_order_id),
        )
        return self._to_order_snapshot(order)

    def list_orders_since(self, symbol: str, start: datetime) -> list[BrokerOrder]:
        normalized_symbol = _normalize_history_symbol(symbol)
        start_utc = _normalize_history_start(start)
        request = GetOrdersRequest(
            status=QueryOrderStatus.ALL,
            limit=_ORDER_HISTORY_PAGE_SIZE,
            after=start_utc,
            direction=Sort.ASC,
            nested=False,
            symbols=[normalized_symbol],
        )
        raw_page = cast(Any, self._trading).get_orders(filter=request)
        page = _history_page(
            raw_page,
            history_name="order",
            maximum_size=_ORDER_HISTORY_PAGE_SIZE,
        )
        seen_order_ids: set[str] = set()
        normalized_orders: list[tuple[datetime, BrokerOrder]] = []

        page_times: list[datetime] = []
        try:
            for source_order in page:
                order = self._to_order_snapshot(source_order)
                if not order.order_id or not order.submitted_at:
                    raise ValueError("Order is missing its history identity fields.")
                if order.symbol.strip().upper() != normalized_symbol:
                    raise ValueError("Order page contains an unexpected symbol.")
                if order.side not in {"buy", "sell"} or not order.status:
                    raise ValueError("Order page contains malformed order state.")
                submitted_at = _parse_aware_datetime(
                    order.submitted_at,
                    field_name="order submitted_at",
                )
                page_times.append(submitted_at)
                if order.order_id in seen_order_ids:
                    continue
                seen_order_ids.add(order.order_id)
                if submitted_at >= start_utc:
                    normalized_orders.append((submitted_at, order))
        except (TypeError, ValueError, OverflowError) as exc:
            raise RuntimeError("Malformed Alpaca order history page.") from exc

        if page_times != sorted(page_times):
            raise RuntimeError("Alpaca order history page is out of order.")
        if len(page) == _ORDER_HISTORY_PAGE_SIZE:
            raise RuntimeError(
                "Alpaca order history reached the 500-record safety limit; "
                "the supported SDK request has no lossless continuation cursor."
            )

        normalized_orders.sort(key=lambda item: (item[0], item[1].order_id))
        return [order for _, order in normalized_orders]

    def list_fill_activities(
        self,
        symbol: str,
        start: datetime,
    ) -> list[BrokerFillActivity]:
        normalized_symbol = _normalize_history_symbol(symbol)
        start_utc = _normalize_history_start(start)
        page_token: str | None = None
        seen_page_tokens: set[str] = set()
        seen_page_signatures: set[tuple[str, ...]] = set()
        seen_activity_ids: set[str] = set()
        normalized_activities: list[tuple[datetime, BrokerFillActivity]] = []
        previous_page_last_time: datetime | None = None

        while True:
            params: dict[str, Any] = {
                "after": _utc_query_text(start_utc),
                "direction": Sort.ASC.value,
                "page_size": _FILL_ACTIVITY_PAGE_SIZE,
            }
            if page_token is not None:
                params["page_token"] = page_token
            raw_page = cast(Any, self._trading).get(
                "/account/activities/FILL",
                params,
            )
            page = _history_page(
                raw_page,
                history_name="FILL activity",
                maximum_size=_FILL_ACTIVITY_PAGE_SIZE,
            )
            if not page:
                break

            page_ids: list[str] = []
            page_times: list[datetime] = []
            page_activities: list[tuple[datetime, BrokerFillActivity]] = []
            try:
                for source_activity in page:
                    activity, transaction_time = self._to_fill_activity(source_activity)
                    page_ids.append(activity.activity_id)
                    page_times.append(transaction_time)
                    page_activities.append((transaction_time, activity))
            except (TypeError, ValueError, OverflowError) as exc:
                raise RuntimeError("Malformed Alpaca FILL activity history page.") from exc

            page_signature = tuple(page_ids)
            if page_signature in seen_page_signatures:
                raise RuntimeError("Alpaca repeated a FILL activity history page.")
            seen_page_signatures.add(page_signature)

            if page_times != sorted(page_times) or (
                previous_page_last_time is not None
                and page_times[0] < previous_page_last_time
            ):
                raise RuntimeError("Alpaca FILL activity history pages are out of order.")
            previous_page_last_time = page_times[-1]

            for transaction_time, activity in page_activities:
                if activity.activity_id in seen_activity_ids:
                    continue
                seen_activity_ids.add(activity.activity_id)
                if (
                    activity.symbol.strip().upper() == normalized_symbol
                    and transaction_time >= start_utc
                ):
                    normalized_activities.append((transaction_time, activity))

            if len(page) < _FILL_ACTIVITY_PAGE_SIZE:
                break

            next_page_token = page_ids[-1]
            if next_page_token in seen_page_tokens:
                raise RuntimeError("Alpaca repeated a FILL activity page token.")
            seen_page_tokens.add(next_page_token)
            page_token = next_page_token

        normalized_activities.sort(
            key=lambda item: (item[0], item[1].activity_id),
        )
        return [activity for _, activity in normalized_activities]

    def submit_market_order(
        self,
        *,
        symbol: str,
        qty: QuantityInput,
        side: OrderSide,
        client_order_id: str,
    ) -> BrokerOrder:
        if not self._submission_enabled:
            raise RuntimeError("broker order submission is disabled")
        normalized_qty = _normalize_submit_quantity(qty)
        normalized_client_order_id = _normalize_client_order_id(client_order_id)
        normalized_side = str(side).strip().lower()
        if normalized_side not in {"buy", "sell"}:
            raise ValueError("order side must be either 'buy' or 'sell'")
        order = cast(
            Any,
            self._trading.submit_order(
                MarketOrderRequest(
                    symbol=symbol,
                    qty=normalized_qty,
                    side=(
                        AlpacaOrderSide.BUY
                        if normalized_side == "buy"
                        else AlpacaOrderSide.SELL
                    ),
                    time_in_force=TimeInForce.DAY,
                    client_order_id=normalized_client_order_id,
                )
            ),
        )
        return self._to_order_snapshot(order)

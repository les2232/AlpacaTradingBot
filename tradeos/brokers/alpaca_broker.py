from __future__ import annotations

import threading
import time
from datetime import datetime, timezone
from typing import Any, cast
import os

from alpaca.data.enums import DataFeed
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.live.stock import StockDataStream
from alpaca.data.requests import StockBarsRequest, StockLatestTradeRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
from alpaca.trading.client import TradingClient
from alpaca.trading.enums import OrderSide as AlpacaOrderSide
from alpaca.trading.enums import QueryOrderStatus, TimeInForce
from alpaca.trading.requests import GetOrdersRequest, MarketOrderRequest

from .base import BrokerAccount, BrokerBar, BrokerClient, BrokerOrder, BrokerPosition, OrderSide


_PRICE_STREAM_FEED_BY_NAME = {
    "iex": DataFeed.IEX,
    "sip": DataFeed.SIP,
    "delayed_sip": DataFeed.DELAYED_SIP,
}


def _minutes_to_timeframe(minutes: int) -> TimeFrame:
    if minutes >= 60 and minutes % 60 == 0:
        return TimeFrame(minutes // 60, TimeFrameUnit.Hour)
    return TimeFrame(minutes, TimeFrameUnit.Minute)


def _normalize_enum_text(value: Any) -> str:
    text = str(value or "").strip()
    if "." in text:
        text = text.rsplit(".", 1)[-1]
    return text.lower()


class AlpacaBroker(BrokerClient):
    def __init__(
        self,
        *,
        api_key: str,
        api_secret: str,
        paper: bool,
        symbols: list[str],
    ) -> None:
        self._api_key = api_key
        self._api_secret = api_secret
        self._symbols = list(symbols)
        self._trading = TradingClient(api_key, api_secret, paper=paper)
        self._data = StockHistoricalDataClient(api_key, api_secret)
        self._stream_enabled = os.getenv("ENABLE_PRICE_STREAM", "true").lower() != "false"
        self._stream_error: str | None = None
        self._latest_prices: dict[str, float] = {}
        self._latest_price_times: dict[str, float] = {}
        self._latest_trade_times: dict[str, float] = {}
        self._price_lock = threading.Lock()
        self._data_stream: StockDataStream | None = None
        self._stream_thread: threading.Thread | None = None

    def get_account(self) -> BrokerAccount:
        account = cast(Any, self._trading).get_account()
        return BrokerAccount(
            cash=float(account.cash),
            buying_power=float(account.buying_power),
            equity=float(account.equity),
            last_equity=float(account.last_equity),
        )

    def is_market_open(self) -> bool:
        try:
            clock = cast(Any, self._trading).get_clock()
        except Exception as exc:
            raise RuntimeError("Unable to fetch Alpaca market clock.") from exc
        return bool(getattr(clock, "is_open", False))

    def _preferred_price_stream_feed(self) -> DataFeed:
        raw = os.getenv("PRICE_STREAM_FEED", "iex").strip().lower()
        return _PRICE_STREAM_FEED_BY_NAME.get(raw, DataFeed.IEX)

    def _latest_trade_feeds(self) -> list[DataFeed]:
        preferred = self._preferred_price_stream_feed()
        feeds: list[DataFeed] = []
        for feed in (preferred, DataFeed.IEX, DataFeed.SIP):
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
            return "stream idle"
        if active_symbols == 0:
            return "stream connecting"
        return f"live stream active for {active_symbols}/{len(self._symbols)} symbols"

    def _start_price_stream(self) -> None:
        if not self._stream_enabled or self._stream_thread is not None:
            return

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

    def _run_price_stream(self) -> None:
        try:
            if self._data_stream is None:
                return
            self._data_stream.run()
        except Exception as exc:
            self._stream_error = str(exc)

    async def _handle_trade(self, trade: Any) -> None:
        symbol = str(getattr(trade, "symbol", ""))
        price = getattr(trade, "price", None)
        if not symbol or price is None:
            return

        with self._price_lock:
            self._latest_prices[symbol] = float(price)
            self._latest_price_times[symbol] = time.time()
            trade_timestamp = getattr(trade, "timestamp", None)
            if isinstance(trade_timestamp, datetime):
                if trade_timestamp.tzinfo is None:
                    normalized_trade_time = trade_timestamp.replace(tzinfo=timezone.utc)
                else:
                    normalized_trade_time = trade_timestamp.astimezone(timezone.utc)
                self._latest_trade_times[symbol] = normalized_trade_time.timestamp()
            else:
                self._latest_trade_times[symbol] = time.time()
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
            return cached_price, time.time() - cached_at

        last_exc: Exception | None = None
        for feed in self._latest_trade_feeds():
            try:
                request = StockLatestTradeRequest(symbol_or_symbols=symbol, feed=feed)
                latest = cast(dict[str, Any], cast(Any, self._data).get_stock_latest_trade(request))
                trade = latest[symbol]
                price = float(trade.price)
                timestamp = getattr(trade, "timestamp", None)
                if isinstance(timestamp, datetime):
                    if timestamp.tzinfo is None:
                        trade_timestamp = timestamp.replace(tzinfo=timezone.utc)
                    else:
                        trade_timestamp = timestamp.astimezone(timezone.utc)
                    age_seconds = max(0.0, (datetime.now(timezone.utc) - trade_timestamp).total_seconds())
                else:
                    age_seconds = 0.0

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
                qty=float(position.qty),
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
            timestamp = cast(datetime, getattr(bar, "timestamp"))
            if timestamp.tzinfo is None:
                timestamp = timestamp.replace(tzinfo=timezone.utc)
            else:
                timestamp = timestamp.astimezone(timezone.utc)
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
        filled_at_raw = getattr(order, "filled_at", None)
        return BrokerOrder(
            order_id=str(getattr(order, "id", "")),
            submitted_at=str(getattr(order, "submitted_at", None)),
            filled_at=str(filled_at_raw) if filled_at_raw is not None else None,
            symbol=str(getattr(order, "symbol", "")),
            side=_normalize_enum_text(getattr(order, "side", "")),
            status=_normalize_enum_text(getattr(order, "status", "")),
            qty=float(order.qty) if getattr(order, "qty", None) else None,
            filled_qty=float(order.filled_qty) if getattr(order, "filled_qty", None) else None,
            filled_avg_price=float(order.filled_avg_price)
            if getattr(order, "filled_avg_price", None)
            else None,
            notional=float(order.notional) if getattr(order, "notional", None) else None,
        )

    def list_recent_orders(self, limit: int = 10) -> list[BrokerOrder]:
        request = GetOrdersRequest(status=QueryOrderStatus.ALL, limit=limit, nested=False)
        orders = cast(list[Any], cast(Any, self._trading).get_orders(filter=request))
        return [self._to_order_snapshot(order) for order in orders]

    def list_open_orders(self) -> list[BrokerOrder]:
        request = GetOrdersRequest(status=QueryOrderStatus.OPEN, nested=False)
        orders = cast(list[Any], cast(Any, self._trading).get_orders(filter=request))
        return [self._to_order_snapshot(order) for order in orders]

    def submit_market_order(
        self,
        *,
        symbol: str,
        qty: int,
        side: OrderSide,
    ) -> BrokerOrder:
        order = cast(
            Any,
            self._trading.submit_order(
                MarketOrderRequest(
                    symbol=symbol,
                    qty=qty,
                    side=AlpacaOrderSide.BUY if side == "buy" else AlpacaOrderSide.SELL,
                    time_in_force=TimeInForce.DAY,
                )
            ),
        )
        return self._to_order_snapshot(order)

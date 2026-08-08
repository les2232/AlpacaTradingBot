from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from typing import Literal


OrderSide = Literal["buy", "sell"]
BrokerEnvironment = Literal["paper", "live", "unknown"]
QuantityInput = Decimal | int | float | str


@dataclass(frozen=True)
class BrokerAccount:
    cash: float
    buying_power: float
    equity: float
    last_equity: float
    account_id: str | None = None
    account_number: str | None = None
    status: str | None = None
    environment: BrokerEnvironment = "unknown"
    trading_blocked: bool | None = None
    transfers_blocked: bool | None = None
    account_blocked: bool | None = None
    trade_suspended_by_user: bool | None = None


@dataclass(frozen=True)
class BrokerPosition:
    symbol: str
    qty: Decimal
    market_value: float
    avg_entry_price: float | None = None
    current_price: float | None = None
    unrealized_pl: float | None = None


@dataclass(frozen=True)
class BrokerOrder:
    order_id: str
    submitted_at: str | None
    filled_at: str | None
    symbol: str
    side: str
    status: str
    qty: Decimal | None
    filled_qty: Decimal | None
    filled_avg_price: float | None
    notional: float | None
    client_order_id: str | None = None
    canceled_at: str | None = None
    expired_at: str | None = None
    failed_at: str | None = None
    replaced_at: str | None = None
    replaced_by: str | None = None
    replaces: str | None = None


@dataclass(frozen=True)
class BrokerFillActivity:
    activity_id: str
    order_id: str
    transaction_time: str
    symbol: str
    side: str
    qty: Decimal
    price: Decimal
    cum_qty: Decimal | None = None
    leaves_qty: Decimal | None = None
    fill_type: str | None = None


@dataclass(frozen=True)
class BrokerBar:
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float


class BrokerClient(ABC):
    @abstractmethod
    def get_account(self) -> BrokerAccount: ...

    @abstractmethod
    def is_market_open(self) -> bool: ...

    @abstractmethod
    def get_price_feed_status(self) -> str: ...

    @abstractmethod
    def get_latest_price_with_age(self, symbol: str) -> tuple[float, float]: ...

    @abstractmethod
    def list_positions(self) -> list[BrokerPosition]: ...

    @abstractmethod
    def get_bars(
        self,
        symbol: str,
        *,
        timeframe_minutes: int,
        start: datetime,
        end: datetime,
    ) -> list[BrokerBar]: ...

    @abstractmethod
    def list_recent_orders(self, limit: int = 10) -> list[BrokerOrder]: ...

    @abstractmethod
    def list_open_orders(self) -> list[BrokerOrder]: ...

    @abstractmethod
    def get_order_by_client_order_id(self, client_order_id: str) -> BrokerOrder: ...

    @abstractmethod
    def list_orders_since(self, symbol: str, start: datetime) -> list[BrokerOrder]: ...

    @abstractmethod
    def list_fill_activities(
        self,
        symbol: str,
        start: datetime,
    ) -> list[BrokerFillActivity]: ...

    @abstractmethod
    def submit_market_order(
        self,
        *,
        symbol: str,
        qty: QuantityInput,
        side: OrderSide,
        client_order_id: str,
    ) -> BrokerOrder: ...

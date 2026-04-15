from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from typing import Literal


OrderSide = Literal["buy", "sell"]


@dataclass(frozen=True)
class BrokerAccount:
    cash: float
    buying_power: float
    equity: float
    last_equity: float


@dataclass(frozen=True)
class BrokerPosition:
    symbol: str
    qty: float
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
    qty: float | None
    filled_qty: float | None
    filled_avg_price: float | None
    notional: float | None


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
    def submit_market_order(
        self,
        *,
        symbol: str,
        qty: int,
        side: OrderSide,
    ) -> BrokerOrder: ...

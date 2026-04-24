from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime


@dataclass(frozen=True)
class Bar:
    symbol: str
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float


@dataclass(frozen=True)
class Quote:
    symbol: str
    price: float
    observed_at: datetime
    age_seconds: float = 0.0


@dataclass(frozen=True)
class AccountState:
    cash: float
    buying_power: float
    equity: float
    last_equity: float


@dataclass(frozen=True)
class PositionState:
    symbol: str
    qty: float
    avg_entry_price: float | None = None
    current_price: float | None = None

    @property
    def is_open(self) -> bool:
        return self.qty != 0

    @property
    def is_long(self) -> bool:
        return self.qty > 0

    @property
    def is_short(self) -> bool:
        return self.qty < 0

    @property
    def side(self) -> str:
        if self.qty > 0:
            return "long"
        if self.qty < 0:
            return "short"
        return "flat"


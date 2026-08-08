from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal, InvalidOperation


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
    qty: Decimal
    avg_entry_price: float | None = None
    current_price: float | None = None

    def __post_init__(self) -> None:
        if isinstance(self.qty, bool):
            raise ValueError("Position quantity must be a finite number.")
        try:
            quantity = self.qty if isinstance(self.qty, Decimal) else Decimal(str(self.qty))
        except (InvalidOperation, TypeError, ValueError) as exc:
            raise ValueError("Position quantity must be a finite number.") from exc
        if not quantity.is_finite():
            raise ValueError("Position quantity must be a finite number.")
        object.__setattr__(self, "qty", quantity)

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

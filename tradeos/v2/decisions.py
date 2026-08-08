from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation


def _decimal_quantity(value: Decimal | int | float | str, *, positive: bool) -> Decimal:
    if isinstance(value, bool):
        raise ValueError("Quantity must be a finite number.")
    try:
        quantity = value if isinstance(value, Decimal) else Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError) as exc:
        raise ValueError("Quantity must be a finite number.") from exc
    if not quantity.is_finite() or (positive and quantity <= 0):
        qualifier = "positive finite" if positive else "finite"
        raise ValueError(f"Quantity must be a {qualifier} number.")
    return quantity


@dataclass(frozen=True)
class StrategyDecision:
    action: str
    reason: str
    target_qty: Decimal = Decimal("0")
    metrics: dict[str, float | int | str | None] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "target_qty", _decimal_quantity(self.target_qty, positive=False))


@dataclass(frozen=True)
class OrderIntent:
    symbol: str
    side: str
    qty: Decimal
    reduce_only: bool
    reason: str

    def __post_init__(self) -> None:
        symbol = str(self.symbol or "").strip().upper()
        if not symbol:
            raise ValueError("Order intent symbol is required.")
        if self.side not in {"buy", "sell"}:
            raise ValueError("Order intent side must be 'buy' or 'sell'.")
        object.__setattr__(self, "symbol", symbol)
        object.__setattr__(self, "qty", _decimal_quantity(self.qty, positive=True))

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class StrategyDecision:
    action: str
    reason: str
    target_qty: int = 0
    metrics: dict[str, float | int | str | None] = field(default_factory=dict)


@dataclass(frozen=True)
class OrderIntent:
    symbol: str
    side: str
    qty: int
    reduce_only: bool
    reason: str


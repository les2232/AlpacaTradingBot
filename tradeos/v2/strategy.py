from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from .decisions import StrategyDecision
from .models import Bar, PositionState, Quote


@dataclass(frozen=True)
class StrategyContext:
    bar: Bar
    quote: Quote
    position: PositionState | None


class Strategy(Protocol):
    def evaluate(self, context: StrategyContext) -> StrategyDecision: ...


@dataclass(frozen=True)
class LongOnlyMeanReversionStrategy:
    entry_threshold_pct: float
    exit_threshold_pct: float = 0.0

    def evaluate(self, context: StrategyContext) -> StrategyDecision:
        anchor = context.bar.close
        quote_price = context.quote.price
        position = context.position
        if position is not None and position.is_short:
            return StrategyDecision(
                action="hold",
                reason="short_position_unsupported",
                metrics={"quote_price": quote_price, "bar_close": anchor},
            )

        if position is not None and position.is_long:
            if quote_price >= anchor * (1.0 + self.exit_threshold_pct):
                return StrategyDecision(
                    action="sell",
                    reason="reversion_exit",
                    target_qty=int(abs(position.qty)),
                    metrics={"quote_price": quote_price, "bar_close": anchor},
                )
            return StrategyDecision(
                action="hold",
                reason="long_waiting_for_exit",
                metrics={"quote_price": quote_price, "bar_close": anchor},
            )

        entry_price = anchor * (1.0 - self.entry_threshold_pct)
        if quote_price <= entry_price:
            return StrategyDecision(
                action="buy",
                reason="reversion_entry",
                target_qty=1,
                metrics={"quote_price": quote_price, "bar_close": anchor, "entry_price": entry_price},
            )
        return StrategyDecision(
            action="hold",
            reason="no_entry",
            metrics={"quote_price": quote_price, "bar_close": anchor, "entry_price": entry_price},
        )


from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from .config import EngineConfig
from .decisions import OrderIntent, StrategyDecision
from .events import EngineEvent
from .models import AccountState, Bar, PositionState, Quote
from .strategy import Strategy, StrategyContext


@dataclass(frozen=True)
class EngineResult:
    decision: StrategyDecision
    intents: tuple[OrderIntent, ...]
    events: tuple[EngineEvent, ...]


class TradingEngine:
    def __init__(self, config: EngineConfig, strategy: Strategy) -> None:
        self.config = config
        self.strategy = strategy

    def evaluate_symbol(
        self,
        *,
        now: datetime,
        bar: Bar,
        quote: Quote,
        position: PositionState | None,
        account: AccountState,
        open_positions: int,
    ) -> EngineResult:
        events: list[EngineEvent] = []
        intents: list[OrderIntent] = []

        if quote.age_seconds > self.config.risk.max_quote_age_seconds:
            events.append(
                EngineEvent(
                    event_type="risk.block",
                    symbol=bar.symbol,
                    message="quote too old",
                    details={"quote_age_seconds": quote.age_seconds},
                )
            )
            return EngineResult(
                decision=StrategyDecision(action="hold", reason="stale_quote"),
                intents=(),
                events=tuple(events),
            )

        if position is not None and position.is_short:
            events.append(
                EngineEvent(
                    event_type="engine.notice",
                    symbol=bar.symbol,
                    message="short position detected; strategy is long-only",
                    details={"qty": position.qty},
                )
            )
            return EngineResult(
                decision=StrategyDecision(action="hold", reason="short_position_unsupported"),
                intents=(),
                events=tuple(events),
            )

        entry_time = now.timetz().replace(tzinfo=None)
        if not (self.config.session.entry_start <= entry_time <= self.config.session.entry_end):
            events.append(
                EngineEvent(
                    event_type="risk.block",
                    symbol=bar.symbol,
                    message="outside entry window",
                    details={"entry_start": self.config.session.entry_start.isoformat(), "entry_end": self.config.session.entry_end.isoformat()},
                )
            )
            return EngineResult(
                decision=StrategyDecision(action="hold", reason="outside_entry_window"),
                intents=(),
                events=tuple(events),
            )

        decision = self.strategy.evaluate(
            StrategyContext(
                bar=bar,
                quote=quote,
                position=position,
            )
        )
        events.append(
            EngineEvent(
                event_type="strategy.decision",
                symbol=bar.symbol,
                message=decision.reason,
                details={"action": decision.action, **decision.metrics},
            )
        )

        if decision.action == "buy":
            if open_positions >= self.config.risk.max_positions:
                events.append(
                    EngineEvent(
                        event_type="risk.block",
                        symbol=bar.symbol,
                        message="max positions reached",
                        details={"open_positions": open_positions, "max_positions": self.config.risk.max_positions},
                    )
                )
                return EngineResult(
                    decision=StrategyDecision(action="hold", reason="max_positions_reached"),
                    intents=(),
                    events=tuple(events),
                )
            max_qty = int(self.config.risk.max_notional_per_trade // max(quote.price, 1e-9))
            max_affordable_qty = int(account.buying_power // max(quote.price, 1e-9))
            qty = min(max_qty, max_affordable_qty, max(0, decision.target_qty))
            if qty <= 0:
                events.append(
                    EngineEvent(
                        event_type="risk.block",
                        symbol=bar.symbol,
                        message="insufficient buying power for entry",
                        details={"buying_power": account.buying_power, "quote_price": quote.price},
                    )
                )
                return EngineResult(
                    decision=StrategyDecision(action="hold", reason="insufficient_buying_power"),
                    intents=(),
                    events=tuple(events),
                )
            intents.append(
                OrderIntent(
                    symbol=bar.symbol,
                    side="buy",
                    qty=qty,
                    reduce_only=False,
                    reason=decision.reason,
                )
            )
        elif decision.action == "sell" and position is not None and position.is_long:
            intents.append(
                OrderIntent(
                    symbol=bar.symbol,
                    side="sell",
                    qty=int(abs(position.qty)),
                    reduce_only=True,
                    reason=decision.reason,
                )
            )

        return EngineResult(decision=decision, intents=tuple(intents), events=tuple(events))

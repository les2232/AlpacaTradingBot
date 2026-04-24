from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from .engine import EngineResult, TradingEngine
from .events import EngineEvent
from .models import AccountState, Bar, PositionState, Quote


@dataclass(frozen=True)
class ReplayFrame:
    now: datetime
    bar: Bar
    quote: Quote
    account: AccountState


@dataclass(frozen=True)
class ReplayRecord:
    frame: ReplayFrame
    result: EngineResult
    position_after: PositionState | None
    open_positions_after: int


class ReplayRunner:
    def __init__(self, engine: TradingEngine) -> None:
        self.engine = engine

    def run(
        self,
        frames: list[ReplayFrame],
        *,
        initial_positions: dict[str, PositionState] | None = None,
    ) -> tuple[ReplayRecord, ...]:
        positions = dict(initial_positions or {})
        records: list[ReplayRecord] = []

        for frame in frames:
            symbol = frame.bar.symbol
            position = positions.get(symbol)
            open_positions = sum(1 for item in positions.values() if item.is_open)
            result = self.engine.evaluate_symbol(
                now=frame.now,
                bar=frame.bar,
                quote=frame.quote,
                position=position,
                account=frame.account,
                open_positions=open_positions,
            )
            position_after = self._apply_intents(position, result, frame.quote.price, symbol)
            if position_after is None or not position_after.is_open:
                positions.pop(symbol, None)
                position_after = None
            else:
                positions[symbol] = position_after
            records.append(
                ReplayRecord(
                    frame=frame,
                    result=result,
                    position_after=position_after,
                    open_positions_after=sum(1 for item in positions.values() if item.is_open),
                )
            )
        return tuple(records)

    def flatten_open_positions(
        self,
        positions: dict[str, PositionState],
        *,
        now: datetime,
        quote_by_symbol: dict[str, Quote],
    ) -> tuple[dict[str, PositionState], tuple[EngineEvent, ...]]:
        remaining = dict(positions)
        events: list[EngineEvent] = []
        for symbol, position in list(remaining.items()):
            if not position.is_open:
                remaining.pop(symbol, None)
                continue
            quote = quote_by_symbol[symbol]
            events.append(
                EngineEvent(
                    event_type="replay.flatten",
                    symbol=symbol,
                    message="flattened replay position",
                    ts=now,
                    details={"side": position.side, "qty": abs(position.qty), "price": quote.price},
                )
            )
            remaining.pop(symbol, None)
        return remaining, tuple(events)

    @staticmethod
    def _apply_intents(
        position: PositionState | None,
        result: EngineResult,
        fill_price: float,
        symbol: str,
    ) -> PositionState | None:
        current = position
        for intent in result.intents:
            signed_qty = intent.qty if intent.side == "buy" else -intent.qty
            if current is None or not current.is_open:
                current = PositionState(symbol=symbol, qty=signed_qty, avg_entry_price=fill_price, current_price=fill_price)
                continue

            new_qty = current.qty + signed_qty
            if current.qty == 0 or (current.qty > 0 and signed_qty > 0) or (current.qty < 0 and signed_qty < 0):
                total_abs_qty = abs(current.qty) + abs(signed_qty)
                avg_price = (
                    ((abs(current.qty) * (current.avg_entry_price or fill_price)) + (abs(signed_qty) * fill_price))
                    / max(total_abs_qty, 1)
                )
                current = PositionState(symbol=symbol, qty=new_qty, avg_entry_price=avg_price, current_price=fill_price)
                continue

            if new_qty == 0:
                current = None
                continue

            if current.qty > 0 > new_qty:
                current = PositionState(symbol=symbol, qty=new_qty, avg_entry_price=fill_price, current_price=fill_price)
            elif current.qty < 0 < new_qty:
                current = PositionState(symbol=symbol, qty=new_qty, avg_entry_price=fill_price, current_price=fill_price)
            else:
                current = PositionState(
                    symbol=symbol,
                    qty=new_qty,
                    avg_entry_price=current.avg_entry_price,
                    current_price=fill_price,
                )
        return current

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal
from math import isfinite

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

        bar_symbol = self._normalize_symbol(bar.symbol)
        quote_symbol = self._normalize_symbol(quote.symbol)
        position_symbol = self._normalize_symbol(position.symbol) if position is not None else None
        if quote_symbol != bar_symbol or (position_symbol is not None and position_symbol != bar_symbol):
            return self._blocked_result(
                reason="symbol_mismatch",
                symbol=bar_symbol or None,
                message="bar, quote, and position symbols must match",
                details={
                    "bar_symbol": bar_symbol,
                    "quote_symbol": quote_symbol,
                    "position_symbol": position_symbol,
                },
            )

        supported_symbols = {self._normalize_symbol(symbol) for symbol in self.config.symbols}
        if not bar_symbol or bar_symbol not in supported_symbols:
            return self._blocked_result(
                reason="unsupported_symbol",
                symbol=bar_symbol or None,
                message="symbol is not enabled in engine config",
                details={"configured_symbols": sorted(supported_symbols)},
            )

        now_utc = self._aware_utc(now)
        bar_timestamp_utc = self._aware_utc(bar.timestamp)
        quote_observed_at_utc = self._aware_utc(quote.observed_at)
        invalid_timestamp_fields = [
            field_name
            for field_name, parsed in (
                ("now", now_utc),
                ("bar.timestamp", bar_timestamp_utc),
                ("quote.observed_at", quote_observed_at_utc),
            )
            if parsed is None
        ]
        if invalid_timestamp_fields:
            return self._blocked_result(
                reason="invalid_timestamp",
                symbol=bar_symbol,
                message="market-data timestamps must be valid and timezone-aware",
                details={"fields": invalid_timestamp_fields},
            )
        assert now_utc is not None
        assert bar_timestamp_utc is not None
        assert quote_observed_at_utc is not None
        if bar_timestamp_utc > now_utc or quote_observed_at_utc > now_utc:
            future_fields = []
            if bar_timestamp_utc > now_utc:
                future_fields.append("bar.timestamp")
            if quote_observed_at_utc > now_utc:
                future_fields.append("quote.observed_at")
            return self._blocked_result(
                reason="invalid_timestamp",
                symbol=bar_symbol,
                message="market-data timestamps cannot be in the future",
                details={"fields": future_fields},
            )

        prices = {
            "quote.price": quote.price,
            "bar.open": bar.open,
            "bar.high": bar.high,
            "bar.low": bar.low,
            "bar.close": bar.close,
        }
        invalid_price_fields = [name for name, value in prices.items() if not self._is_positive_finite(value)]
        if invalid_price_fields:
            return self._blocked_result(
                reason="invalid_price",
                symbol=bar_symbol,
                message="quote and bar prices must be positive finite numbers",
                details={"fields": invalid_price_fields},
            )

        quote_age_seconds = (now_utc - quote_observed_at_utc).total_seconds()
        if not isfinite(quote_age_seconds) or quote_age_seconds < 0:
            return self._blocked_result(
                reason="invalid_timestamp",
                symbol=bar_symbol,
                message="quote observation time produced an invalid age",
                details={"quote_age_seconds": quote_age_seconds},
            )
        if quote_age_seconds > self.config.risk.max_quote_age_seconds:
            events.append(
                EngineEvent(
                    event_type="risk.block",
                    symbol=bar_symbol,
                    message="quote too old",
                    details={"quote_age_seconds": quote_age_seconds},
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
                    symbol=bar_symbol,
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
        outside_entry_window = not (
            self.config.session.entry_start <= entry_time <= self.config.session.entry_end
        )
        if outside_entry_window and (position is None or not position.is_long):
            events.append(
                EngineEvent(
                    event_type="risk.block",
                    symbol=bar_symbol,
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
                symbol=bar_symbol,
                message=decision.reason,
                details={"action": decision.action, **decision.metrics},
            )
        )

        if decision.action == "buy":
            if outside_entry_window:
                events.append(
                    EngineEvent(
                        event_type="risk.block",
                        symbol=bar_symbol,
                        message="outside entry window",
                        details={"entry_start": self.config.session.entry_start.isoformat(), "entry_end": self.config.session.entry_end.isoformat()},
                    )
                )
                return EngineResult(
                    decision=StrategyDecision(action="hold", reason="outside_entry_window"),
                    intents=(),
                    events=tuple(events),
                )
            if open_positions >= self.config.risk.max_positions:
                events.append(
                    EngineEvent(
                        event_type="risk.block",
                        symbol=bar_symbol,
                        message="max positions reached",
                        details={"open_positions": open_positions, "max_positions": self.config.risk.max_positions},
                    )
                )
                return EngineResult(
                    decision=StrategyDecision(action="hold", reason="max_positions_reached"),
                    intents=(),
                    events=tuple(events),
                )
            quote_price = float(quote.price)
            if not self._is_positive_finite(self.config.risk.max_notional_per_trade):
                return self._blocked_result(
                    reason="invalid_risk_config",
                    symbol=bar_symbol,
                    message="max notional per trade must be a positive finite number",
                    details={},
                )
            buying_power = float(account.buying_power) if self._is_positive_finite(account.buying_power) else 0.0
            max_qty = int(self.config.risk.max_notional_per_trade // quote_price)
            max_affordable_qty = int(buying_power // quote_price)
            qty = min(
                Decimal(max_qty),
                Decimal(max_affordable_qty),
                max(Decimal("0"), decision.target_qty),
            )
            if qty <= 0:
                events.append(
                    EngineEvent(
                        event_type="risk.block",
                        symbol=bar_symbol,
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
                    symbol=bar_symbol,
                    side="buy",
                    qty=qty,
                    reduce_only=False,
                    reason=decision.reason,
                )
            )
        elif decision.action == "sell" and position is not None and position.is_long:
            qty = abs(position.qty)
            intents.append(
                OrderIntent(
                    symbol=bar_symbol,
                    side="sell",
                    qty=qty,
                    reduce_only=True,
                    reason=decision.reason,
                )
            )

        return EngineResult(decision=decision, intents=tuple(intents), events=tuple(events))

    @staticmethod
    def _normalize_symbol(value: object) -> str:
        return str(value or "").strip().upper()

    @staticmethod
    def _aware_utc(value: object) -> datetime | None:
        if not isinstance(value, datetime) or value.tzinfo is None:
            return None
        try:
            if value.utcoffset() is None:
                return None
            return value.astimezone(timezone.utc)
        except (OverflowError, TypeError, ValueError):
            return None

    @staticmethod
    def _is_positive_finite(value: object) -> bool:
        if isinstance(value, bool):
            return False
        try:
            number = float(value)
        except (OverflowError, TypeError, ValueError):
            return False
        return isfinite(number) and number > 0

    @staticmethod
    def _blocked_result(
        *,
        reason: str,
        symbol: str | None,
        message: str,
        details: dict[str, object],
    ) -> EngineResult:
        return EngineResult(
            decision=StrategyDecision(action="hold", reason=reason),
            intents=(),
            events=(
                EngineEvent(
                    event_type="risk.block",
                    symbol=symbol,
                    message=message,
                    details=details,
                ),
            ),
        )

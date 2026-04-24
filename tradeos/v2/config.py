from __future__ import annotations

from dataclasses import dataclass
from datetime import time
from typing import Any


class StrategyConfigError(ValueError):
    pass


def _reject_unknown_keys(payload: dict[str, Any], allowed: set[str], *, context: str) -> None:
    unknown = sorted(key for key in payload if key not in allowed)
    if unknown:
        raise StrategyConfigError(f"Unknown {context} fields: {', '.join(unknown)}")


@dataclass(frozen=True)
class RiskConfig:
    max_positions: int
    max_notional_per_trade: float
    max_quote_age_seconds: int

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> "RiskConfig":
        allowed = {"max_positions", "max_notional_per_trade", "max_quote_age_seconds"}
        _reject_unknown_keys(payload, allowed, context="risk config")
        return cls(
            max_positions=int(payload["max_positions"]),
            max_notional_per_trade=float(payload["max_notional_per_trade"]),
            max_quote_age_seconds=int(payload["max_quote_age_seconds"]),
        )


@dataclass(frozen=True)
class MarketSessionConfig:
    entry_start: time
    entry_end: time

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> "MarketSessionConfig":
        allowed = {"entry_start", "entry_end"}
        _reject_unknown_keys(payload, allowed, context="session config")
        return cls(
            entry_start=time.fromisoformat(str(payload["entry_start"])),
            entry_end=time.fromisoformat(str(payload["entry_end"])),
        )


@dataclass(frozen=True)
class EngineConfig:
    symbols: tuple[str, ...]
    timeframe_minutes: int
    risk: RiskConfig
    session: MarketSessionConfig

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> "EngineConfig":
        allowed = {"symbols", "timeframe_minutes", "risk", "session"}
        _reject_unknown_keys(payload, allowed, context="engine config")
        return cls(
            symbols=tuple(str(symbol).strip().upper() for symbol in payload["symbols"]),
            timeframe_minutes=int(payload["timeframe_minutes"]),
            risk=RiskConfig.from_payload(dict(payload["risk"])),
            session=MarketSessionConfig.from_payload(dict(payload["session"])),
        )


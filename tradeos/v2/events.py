from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any


@dataclass(frozen=True)
class EngineEvent:
    event_type: str
    message: str
    ts: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    symbol: str | None = None
    details: dict[str, Any] = field(default_factory=dict)


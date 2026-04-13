from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4

import pytest

from storage import BotStorage
from trading_bot import AlpacaTradingBot, StaleMarketDataError


def test_storage_claim_decision_timestamp_is_global_and_single_use() -> None:
    db_path = Path.cwd() / f"test_decision_claim_{uuid4().hex}.db"
    storage = BotStorage(db_path)

    first_claim = storage.claim_decision_timestamp(
        "2026-04-10T18:00:00+00:00",
        "2026-04-10T18:00:05+00:00",
    )
    second_claim = storage.claim_decision_timestamp(
        "2026-04-10T18:00:00+00:00",
        "2026-04-10T18:00:06+00:00",
    )

    assert first_claim is True
    assert second_claim is False


@dataclass
class _DummyAccount:
    cash: float = 1000.0
    buying_power: float = 1000.0
    equity: float = 1000.0
    last_equity: float = 1000.0


@dataclass
class _DummyConfig:
    symbols: list[str]
    max_daily_loss_usd: float = 300.0
    bar_timeframe_minutes: int = 15


def test_build_snapshot_aborts_cycle_on_stale_market_data(monkeypatch: pytest.MonkeyPatch) -> None:
    bot = AlpacaTradingBot.__new__(AlpacaTradingBot)
    bot.config = _DummyConfig(symbols=["AAPL"])
    bot.get_account = lambda: _DummyAccount()
    bot.get_positions_by_symbol = lambda: {}
    bot._update_position_holding_state = lambda positions, observed_at_utc: None
    bot._position_holding_minutes = lambda symbol, now_utc: None

    def raise_stale(*args, **kwargs):
        raise StaleMarketDataError("stale completed bars")

    monkeypatch.setattr(bot, "evaluate_symbol", raise_stale)

    with pytest.raises(StaleMarketDataError):
        bot.build_snapshot()

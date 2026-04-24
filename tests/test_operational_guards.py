from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
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


def test_storage_claim_order_fill_is_global_and_single_use() -> None:
    db_path = Path.cwd() / f"test_order_fill_claim_{uuid4().hex}.db"
    storage = BotStorage(db_path)

    first_claim = storage.claim_order_fill(
        "order-123",
        5.0,
        "2026-04-10T18:00:05+00:00",
    )
    second_claim = storage.claim_order_fill(
        "order-123",
        5.0,
        "2026-04-10T18:00:06+00:00",
    )
    distinct_qty_claim = storage.claim_order_fill(
        "order-123",
        10.0,
        "2026-04-10T18:00:07+00:00",
    )

    assert first_claim is True
    assert second_claim is False
    assert distinct_qty_claim is True


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
    max_data_delay_seconds: int = 300


@dataclass
class _DummyBar:
    timestamp: datetime
    close: float = 100.0
    high: float = 101.0
    low: float = 99.0
    volume: float = 1000.0


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


def test_startup_guard_blocks_previous_session_bars() -> None:
    bot = AlpacaTradingBot.__new__(AlpacaTradingBot)
    bot.config = _DummyConfig(symbols=["AAPL", "MSFT"])
    bot.active_symbols = ["AAPL", "MSFT"]
    bot._startup_market_data_validated_for_et_date = None

    stale_bar = _DummyBar(timestamp=datetime(2026, 4, 15, 19, 30, tzinfo=timezone.utc))
    bot._get_intraday_bars = lambda symbol, bars_needed, decision_timestamp=None: [stale_bar]
    bot._latest_bar_close_time = lambda bars: bars[-1].timestamp + timedelta(minutes=15)

    with pytest.raises(StaleMarketDataError, match="startup market data not ready"):
        bot._validate_startup_market_data(datetime(2026, 4, 16, 13, 45, tzinfo=timezone.utc))


def test_startup_guard_accepts_same_session_bars() -> None:
    bot = AlpacaTradingBot.__new__(AlpacaTradingBot)
    bot.config = _DummyConfig(symbols=["AAPL"])
    bot.active_symbols = ["AAPL"]
    bot._startup_market_data_validated_for_et_date = None

    current_bar = _DummyBar(timestamp=datetime(2026, 4, 16, 13, 30, tzinfo=timezone.utc))
    bot._get_intraday_bars = lambda symbol, bars_needed, decision_timestamp=None: [current_bar]
    bot._latest_bar_close_time = lambda bars: bars[-1].timestamp + timedelta(minutes=15)

    bot._validate_startup_market_data(datetime(2026, 4, 16, 13, 45, tzinfo=timezone.utc))

    assert bot._startup_market_data_validated_for_et_date == "2026-04-16"

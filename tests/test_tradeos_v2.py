from datetime import datetime, time, timezone

import pytest

from tradeos.v2 import (
    AccountState,
    Bar,
    BrokerIntentExecutor,
    EngineConfig,
    LongOnlyMeanReversionStrategy,
    PositionState,
    Quote,
    ReplayFrame,
    ReplayRunner,
    TradingEngine,
)
from tradeos.v2.config import StrategyConfigError
from tradeos.brokers.base import BrokerOrder


def _engine() -> TradingEngine:
    config = EngineConfig.from_payload(
        {
            "symbols": ["AMD"],
            "timeframe_minutes": 15,
            "risk": {
                "max_positions": 2,
                "max_notional_per_trade": 1000.0,
                "max_quote_age_seconds": 30,
            },
            "session": {
                "entry_start": "09:45:00",
                "entry_end": "15:45:00",
            },
        }
    )
    return TradingEngine(
        config=config,
        strategy=LongOnlyMeanReversionStrategy(entry_threshold_pct=0.01),
    )


def test_engine_config_rejects_unknown_fields() -> None:
    with pytest.raises(StrategyConfigError):
        EngineConfig.from_payload(
            {
                "symbols": ["AMD"],
                "timeframe_minutes": 15,
                "risk": {
                    "max_positions": 2,
                    "max_notional_per_trade": 1000.0,
                    "max_quote_age_seconds": 30,
                    "extra": True,
                },
                "session": {
                    "entry_start": "09:45:00",
                    "entry_end": "15:45:00",
                },
            }
        )


def test_position_state_reports_long_short_and_flat() -> None:
    assert PositionState(symbol="AMD", qty=5).side == "long"
    assert PositionState(symbol="AMD", qty=-2).side == "short"
    assert PositionState(symbol="AMD", qty=0).side == "flat"


def test_engine_blocks_stale_quotes() -> None:
    result = _engine().evaluate_symbol(
        now=datetime(2026, 4, 22, 16, 0, tzinfo=timezone.utc),
        bar=Bar(symbol="AMD", timestamp=datetime(2026, 4, 22, 15, 45, tzinfo=timezone.utc), open=100, high=101, low=99, close=100, volume=1_000),
        quote=Quote(symbol="AMD", price=99.0, observed_at=datetime(2026, 4, 22, 16, 0, tzinfo=timezone.utc), age_seconds=45),
        position=None,
        account=AccountState(cash=10_000, buying_power=10_000, equity=10_000, last_equity=10_000),
        open_positions=0,
    )

    assert result.decision.reason == "stale_quote"
    assert result.intents == ()
    assert result.events[0].event_type == "risk.block"


def test_engine_suppresses_short_positions_for_long_only_strategy() -> None:
    result = _engine().evaluate_symbol(
        now=datetime(2026, 4, 22, 16, 0, tzinfo=timezone.utc),
        bar=Bar(symbol="AMD", timestamp=datetime(2026, 4, 22, 15, 45, tzinfo=timezone.utc), open=100, high=101, low=99, close=100, volume=1_000),
        quote=Quote(symbol="AMD", price=99.0, observed_at=datetime(2026, 4, 22, 16, 0, tzinfo=timezone.utc), age_seconds=1),
        position=PositionState(symbol="AMD", qty=-3, avg_entry_price=101.0),
        account=AccountState(cash=10_000, buying_power=10_000, equity=10_000, last_equity=10_000),
        open_positions=1,
    )

    assert result.decision.reason == "short_position_unsupported"
    assert result.intents == ()
    assert result.events[0].event_type == "engine.notice"


def test_engine_emits_buy_intent_when_entry_triggers() -> None:
    result = _engine().evaluate_symbol(
        now=datetime(2026, 4, 22, 16, 0, tzinfo=timezone.utc).replace(hour=15, minute=0),
        bar=Bar(symbol="AMD", timestamp=datetime(2026, 4, 22, 14, 45, tzinfo=timezone.utc), open=100, high=101, low=99, close=100, volume=1_000),
        quote=Quote(symbol="AMD", price=98.9, observed_at=datetime(2026, 4, 22, 15, 0, tzinfo=timezone.utc), age_seconds=1),
        position=None,
        account=AccountState(cash=10_000, buying_power=10_000, equity=10_000, last_equity=10_000),
        open_positions=0,
    )

    assert result.decision.action == "buy"
    assert result.intents == (
        result.intents[0],
    )
    assert result.intents[0].side == "buy"
    assert result.intents[0].reduce_only is False


def test_engine_emits_reduce_only_exit_for_long_position() -> None:
    result = _engine().evaluate_symbol(
        now=datetime(2026, 4, 22, 15, 0, tzinfo=timezone.utc),
        bar=Bar(symbol="AMD", timestamp=datetime(2026, 4, 22, 14, 45, tzinfo=timezone.utc), open=100, high=101, low=99, close=100, volume=1_000),
        quote=Quote(symbol="AMD", price=100.1, observed_at=datetime(2026, 4, 22, 15, 0, tzinfo=timezone.utc), age_seconds=1),
        position=PositionState(symbol="AMD", qty=4, avg_entry_price=99.0),
        account=AccountState(cash=10_000, buying_power=10_000, equity=10_000, last_equity=10_000),
        open_positions=1,
    )

    assert result.decision.action == "sell"
    assert result.intents[0].side == "sell"
    assert result.intents[0].reduce_only is True
    assert result.intents[0].qty == 4


def test_engine_blocks_entries_outside_session_window() -> None:
    result = _engine().evaluate_symbol(
        now=datetime(2026, 4, 22, 8, 0, tzinfo=timezone.utc).replace(hour=8, minute=0),
        bar=Bar(symbol="AMD", timestamp=datetime(2026, 4, 22, 7, 45, tzinfo=timezone.utc), open=100, high=101, low=99, close=100, volume=1_000),
        quote=Quote(symbol="AMD", price=98.9, observed_at=datetime(2026, 4, 22, 8, 0, tzinfo=timezone.utc), age_seconds=1),
        position=None,
        account=AccountState(cash=10_000, buying_power=10_000, equity=10_000, last_equity=10_000),
        open_positions=0,
    )

    assert result.decision.reason == "outside_entry_window"
    assert result.intents == ()


def test_replay_runner_applies_entry_and_exit_intents() -> None:
    runner = ReplayRunner(_engine())
    account = AccountState(cash=10_000, buying_power=10_000, equity=10_000, last_equity=10_000)
    records = runner.run(
        [
            ReplayFrame(
                now=datetime(2026, 4, 22, 15, 0, tzinfo=timezone.utc),
                bar=Bar(symbol="AMD", timestamp=datetime(2026, 4, 22, 14, 45, tzinfo=timezone.utc), open=100, high=101, low=99, close=100, volume=1_000),
                quote=Quote(symbol="AMD", price=98.9, observed_at=datetime(2026, 4, 22, 15, 0, tzinfo=timezone.utc), age_seconds=1),
                account=account,
            ),
            ReplayFrame(
                now=datetime(2026, 4, 22, 15, 15, tzinfo=timezone.utc),
                bar=Bar(symbol="AMD", timestamp=datetime(2026, 4, 22, 15, 0, tzinfo=timezone.utc), open=100, high=101, low=99, close=100, volume=1_000),
                quote=Quote(symbol="AMD", price=100.1, observed_at=datetime(2026, 4, 22, 15, 15, tzinfo=timezone.utc), age_seconds=1),
                account=account,
            ),
        ]
    )

    assert records[0].result.intents[0].side == "buy"
    assert records[0].position_after is not None
    assert records[0].position_after.is_long
    assert records[1].result.intents[0].side == "sell"
    assert records[1].position_after is None


def test_replay_runner_can_flatten_remaining_positions() -> None:
    runner = ReplayRunner(_engine())
    remaining, events = runner.flatten_open_positions(
        {"AMD": PositionState(symbol="AMD", qty=2, avg_entry_price=99.0)},
        now=datetime(2026, 4, 22, 15, 55, tzinfo=timezone.utc),
        quote_by_symbol={"AMD": Quote(symbol="AMD", price=100.0, observed_at=datetime(2026, 4, 22, 15, 55, tzinfo=timezone.utc), age_seconds=0)},
    )

    assert remaining == {}
    assert events[0].event_type == "replay.flatten"


class _FakeBroker:
    def __init__(self) -> None:
        self.submissions: list[tuple[str, int, str]] = []

    def submit_market_order(self, *, symbol: str, qty: int, side: str) -> BrokerOrder:
        self.submissions.append((symbol, qty, side))
        return BrokerOrder(
            order_id="order-1",
            submitted_at="2026-04-22T15:00:00+00:00",
            filled_at=None,
            symbol=symbol,
            side=side,
            status="new",
            qty=qty,
            filled_qty=0,
            filled_avg_price=None,
            notional=None,
        )


def test_broker_intent_executor_rejects_invalid_reduce_only_sell() -> None:
    broker = _FakeBroker()
    executor = BrokerIntentExecutor(broker)  # type: ignore[arg-type]
    results = executor.execute_intents(
        intents=(
            _engine().evaluate_symbol(
                now=datetime(2026, 4, 22, 15, 0, tzinfo=timezone.utc),
                bar=Bar(symbol="AMD", timestamp=datetime(2026, 4, 22, 14, 45, tzinfo=timezone.utc), open=100, high=101, low=99, close=100, volume=1_000),
                quote=Quote(symbol="AMD", price=100.1, observed_at=datetime(2026, 4, 22, 15, 0, tzinfo=timezone.utc), age_seconds=1),
                position=PositionState(symbol="AMD", qty=4, avg_entry_price=99.0),
                account=AccountState(cash=10_000, buying_power=10_000, equity=10_000, last_equity=10_000),
                open_positions=1,
            ).intents[0],
        ),
        positions={},
    )

    assert results[0].status == "rejected"
    assert broker.submissions == []


def test_broker_intent_executor_submits_valid_intent() -> None:
    broker = _FakeBroker()
    executor = BrokerIntentExecutor(broker)  # type: ignore[arg-type]
    intent = _engine().evaluate_symbol(
        now=datetime(2026, 4, 22, 15, 0, tzinfo=timezone.utc),
        bar=Bar(symbol="AMD", timestamp=datetime(2026, 4, 22, 14, 45, tzinfo=timezone.utc), open=100, high=101, low=99, close=100, volume=1_000),
        quote=Quote(symbol="AMD", price=98.9, observed_at=datetime(2026, 4, 22, 15, 0, tzinfo=timezone.utc), age_seconds=1),
        position=None,
        account=AccountState(cash=10_000, buying_power=10_000, equity=10_000, last_equity=10_000),
        open_positions=0,
    ).intents[0]

    results = executor.execute_intents(
        intents=(intent,),
        positions={},
    )

    assert results[0].status == "submitted"
    assert broker.submissions == [("AMD", 1, "buy")]

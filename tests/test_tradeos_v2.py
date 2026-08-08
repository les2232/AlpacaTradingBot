from dataclasses import replace
from datetime import datetime, timedelta, timezone
from decimal import Decimal

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


@pytest.mark.parametrize("reported_age", [float("nan"), -1_000.0])
def test_engine_derives_quote_age_and_blocks_stale_quotes(reported_age: float) -> None:
    now = datetime(2026, 4, 22, 16, 0, tzinfo=timezone.utc)
    result = _engine().evaluate_symbol(
        now=now,
        bar=Bar(symbol="AMD", timestamp=datetime(2026, 4, 22, 15, 45, tzinfo=timezone.utc), open=100, high=101, low=99, close=100, volume=1_000),
        quote=Quote(symbol="AMD", price=99.0, observed_at=now - timedelta(seconds=45), age_seconds=reported_age),
        position=None,
        account=AccountState(cash=10_000, buying_power=10_000, equity=10_000, last_equity=10_000),
        open_positions=0,
    )

    assert result.decision.reason == "stale_quote"
    assert result.intents == ()
    assert result.events[0].event_type == "risk.block"


def test_engine_ignores_negative_reported_age_for_fresh_quote() -> None:
    now = datetime(2026, 4, 22, 15, 0, tzinfo=timezone.utc)
    result = _engine().evaluate_symbol(
        now=now,
        bar=Bar(symbol="AMD", timestamp=now - timedelta(minutes=15), open=100, high=101, low=99, close=100, volume=1_000),
        quote=Quote(symbol="AMD", price=98.9, observed_at=now, age_seconds=-10_000),
        position=None,
        account=AccountState(cash=10_000, buying_power=10_000, equity=10_000, last_equity=10_000),
        open_positions=0,
    )

    assert result.decision.action == "buy"
    assert len(result.intents) == 1


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


def test_engine_allows_reduce_only_exit_outside_entry_window() -> None:
    now = datetime(2026, 4, 22, 8, 0, tzinfo=timezone.utc)
    result = _engine().evaluate_symbol(
        now=now,
        bar=Bar(symbol="AMD", timestamp=now - timedelta(minutes=15), open=100, high=101, low=99, close=100, volume=1_000),
        quote=Quote(symbol="AMD", price=100.1, observed_at=now, age_seconds=0),
        position=PositionState(symbol="AMD", qty=Decimal("0.375"), avg_entry_price=99.0),
        account=AccountState(cash=10_000, buying_power=10_000, equity=10_000, last_equity=10_000),
        open_positions=1,
    )

    assert result.decision.action == "sell"
    assert result.intents[0].reduce_only is True
    assert result.intents[0].qty == Decimal("0.375")


@pytest.mark.parametrize(
    ("quote_symbol", "position"),
    [
        ("MSFT", None),
        ("AMD", PositionState(symbol="MSFT", qty=1)),
    ],
)
def test_engine_rejects_mismatched_market_data_symbols(
    quote_symbol: str,
    position: PositionState | None,
) -> None:
    now = datetime(2026, 4, 22, 15, 0, tzinfo=timezone.utc)
    result = _engine().evaluate_symbol(
        now=now,
        bar=Bar(symbol="AMD", timestamp=now - timedelta(minutes=15), open=100, high=101, low=99, close=100, volume=1_000),
        quote=Quote(symbol=quote_symbol, price=98.9, observed_at=now),
        position=position,
        account=AccountState(cash=10_000, buying_power=10_000, equity=10_000, last_equity=10_000),
        open_positions=1 if position is not None else 0,
    )

    assert result.decision.reason == "symbol_mismatch"
    assert result.intents == ()
    assert result.events[0].event_type == "risk.block"


def test_engine_rejects_symbols_not_enabled_in_config() -> None:
    now = datetime(2026, 4, 22, 15, 0, tzinfo=timezone.utc)
    result = _engine().evaluate_symbol(
        now=now,
        bar=Bar(symbol="MSFT", timestamp=now - timedelta(minutes=15), open=100, high=101, low=99, close=100, volume=1_000),
        quote=Quote(symbol="MSFT", price=98.9, observed_at=now),
        position=None,
        account=AccountState(cash=10_000, buying_power=10_000, equity=10_000, last_equity=10_000),
        open_positions=0,
    )

    assert result.decision.reason == "unsupported_symbol"
    assert result.intents == ()


@pytest.mark.parametrize("timestamp_field", ["now", "bar", "quote"])
def test_engine_rejects_naive_timestamps(timestamp_field: str) -> None:
    now = datetime(2026, 4, 22, 15, 0, tzinfo=timezone.utc)
    bar = Bar(symbol="AMD", timestamp=now - timedelta(minutes=15), open=100, high=101, low=99, close=100, volume=1_000)
    quote = Quote(symbol="AMD", price=98.9, observed_at=now)
    evaluation_now = now
    if timestamp_field == "now":
        evaluation_now = now.replace(tzinfo=None)
    elif timestamp_field == "bar":
        bar = replace(bar, timestamp=bar.timestamp.replace(tzinfo=None))
    else:
        quote = replace(quote, observed_at=quote.observed_at.replace(tzinfo=None))

    result = _engine().evaluate_symbol(
        now=evaluation_now,
        bar=bar,
        quote=quote,
        position=None,
        account=AccountState(cash=10_000, buying_power=10_000, equity=10_000, last_equity=10_000),
        open_positions=0,
    )

    assert result.decision.reason == "invalid_timestamp"
    assert result.intents == ()


@pytest.mark.parametrize("timestamp_field", ["bar", "quote"])
def test_engine_rejects_future_market_data_timestamps(timestamp_field: str) -> None:
    now = datetime(2026, 4, 22, 15, 0, tzinfo=timezone.utc)
    bar = Bar(symbol="AMD", timestamp=now - timedelta(minutes=15), open=100, high=101, low=99, close=100, volume=1_000)
    quote = Quote(symbol="AMD", price=98.9, observed_at=now)
    if timestamp_field == "bar":
        bar = replace(bar, timestamp=now + timedelta(seconds=1))
    else:
        quote = replace(quote, observed_at=now + timedelta(seconds=1))

    result = _engine().evaluate_symbol(
        now=now,
        bar=bar,
        quote=quote,
        position=None,
        account=AccountState(cash=10_000, buying_power=10_000, equity=10_000, last_equity=10_000),
        open_positions=0,
    )

    assert result.decision.reason == "invalid_timestamp"
    assert result.intents == ()


@pytest.mark.parametrize(
    ("price_field", "invalid_value"),
    [
        ("quote.price", 0.0),
        ("quote.price", float("nan")),
        ("bar.open", -1.0),
        ("bar.high", float("inf")),
        ("bar.low", 0.0),
        ("bar.close", float("-inf")),
    ],
)
def test_engine_rejects_nonpositive_or_nonfinite_prices(
    price_field: str,
    invalid_value: float,
) -> None:
    now = datetime(2026, 4, 22, 15, 0, tzinfo=timezone.utc)
    bar = Bar(symbol="AMD", timestamp=now - timedelta(minutes=15), open=100, high=101, low=99, close=100, volume=1_000)
    quote = Quote(symbol="AMD", price=98.9, observed_at=now)
    target, field_name = price_field.split(".")
    if target == "bar":
        bar = replace(bar, **{field_name: invalid_value})
    else:
        quote = replace(quote, **{field_name: invalid_value})

    result = _engine().evaluate_symbol(
        now=now,
        bar=bar,
        quote=quote,
        position=None,
        account=AccountState(cash=10_000, buying_power=10_000, equity=10_000, last_equity=10_000),
        open_positions=0,
    )

    assert result.decision.reason == "invalid_price"
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


def test_replay_runner_preserves_fractional_decimal_exit_quantity() -> None:
    now = datetime(2026, 4, 22, 15, 0, tzinfo=timezone.utc)
    records = ReplayRunner(_engine()).run(
        [
            ReplayFrame(
                now=now,
                bar=Bar(symbol="AMD", timestamp=now - timedelta(minutes=15), open=100, high=101, low=99, close=100, volume=1_000),
                quote=Quote(symbol="AMD", price=100.1, observed_at=now),
                account=AccountState(cash=10_000, buying_power=10_000, equity=10_000, last_equity=10_000),
            )
        ],
        initial_positions={
            "AMD": PositionState(symbol="AMD", qty=Decimal("0.375"), avg_entry_price=99.0)
        },
    )

    assert records[0].result.intents[0].qty == Decimal("0.375")
    assert records[0].position_after is None


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
        self.submissions: list[tuple[str, Decimal, str, str]] = []

    def submit_market_order(
        self,
        *,
        symbol: str,
        qty: Decimal,
        side: str,
        client_order_id: str,
    ) -> BrokerOrder:
        self.submissions.append((symbol, qty, side, client_order_id))
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


def test_broker_intent_executor_never_submits_duplicate_reduce_only_exits() -> None:
    broker = _FakeBroker()
    executor = BrokerIntentExecutor(broker)  # type: ignore[arg-type]
    position = PositionState(symbol="AMD", qty=Decimal("0.375"), avg_entry_price=99.0)
    intent = _engine().evaluate_symbol(
        now=datetime(2026, 4, 22, 15, 0, tzinfo=timezone.utc),
        bar=Bar(symbol="AMD", timestamp=datetime(2026, 4, 22, 14, 45, tzinfo=timezone.utc), open=100, high=101, low=99, close=100, volume=1_000),
        quote=Quote(symbol="AMD", price=100.1, observed_at=datetime(2026, 4, 22, 15, 0, tzinfo=timezone.utc)),
        position=position,
        account=AccountState(cash=10_000, buying_power=10_000, equity=10_000, last_equity=10_000),
        open_positions=1,
    ).intents[0]

    results = executor.execute_intents(
        intents=(intent, intent),
        positions={"AMD": position},
    )

    assert [result.reason for result in results] == [
        "v2_broker_submission_disabled",
        "v2_broker_submission_disabled",
    ]
    assert broker.submissions == []


def test_broker_intent_executor_fails_closed_without_submission_gate() -> None:
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
        intents=(intent, intent),
        positions={},
    )

    assert [result.status for result in results] == ["rejected", "rejected"]
    assert [result.reason for result in results] == [
        "v2_broker_submission_disabled",
        "v2_broker_submission_disabled",
    ]
    assert broker.submissions == []

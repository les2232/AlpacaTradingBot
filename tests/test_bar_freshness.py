from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

import trading_bot
from tradeos.brokers import BrokerBar


DECISION_TIME = datetime(2026, 4, 22, 15, 45, tzinfo=timezone.utc)
TIMEFRAME_MINUTES = 15
MAX_DATA_DELAY_SECONDS = 300


def _validate(
    raw_timestamp: object,
    *,
    reference_time: datetime = DECISION_TIME,
) -> trading_bot.BarFreshnessValidation:
    return trading_bot.validate_bar_timestamp(
        raw_timestamp,
        symbol="AMD",
        decision_timestamp=DECISION_TIME,
        reference_time=reference_time,
        timeframe_minutes=TIMEFRAME_MINUTES,
        max_data_delay_seconds=MAX_DATA_DELAY_SECONDS,
    )


def _fresh_bar() -> SimpleNamespace:
    return SimpleNamespace(
        timestamp=DECISION_TIME - timedelta(minutes=TIMEFRAME_MINUTES),
    )


def _rejected_bar(raw_timestamp: object) -> SimpleNamespace:
    return SimpleNamespace(timestamp=raw_timestamp)


def _make_validation_bot(bar: object) -> tuple[trading_bot.TradeOSBot, Mock]:
    bot = trading_bot.AlpacaTradingBot.__new__(trading_bot.AlpacaTradingBot)
    strategy_execute = Mock(name="strategy_execute")
    strategy = SimpleNamespace(
        config=SimpleNamespace(),
        decide_action_details=strategy_execute,
    )
    bot.config = SimpleNamespace(
        symbols=["AMD"],
        strategy_mode="mean_reversion",
        bar_timeframe_minutes=TIMEFRAME_MINUTES,
        max_data_delay_seconds=MAX_DATA_DELAY_SECONDS,
        max_daily_loss_usd=300.0,
    )
    bot._bars_cache = {}
    bot._hourly_regime_cache = {}
    bot._et_now = lambda: DECISION_TIME
    bot._utc_now = lambda: DECISION_TIME
    bot.get_decision_timestamp = lambda: DECISION_TIME
    bot._should_process_decision_timestamp = lambda timestamp: True
    bot._claim_global_decision_execution = lambda timestamp: True
    bot._is_regular_hours = lambda timestamp: True
    bot._validate_startup_market_data = lambda timestamp: None
    bot.strategy = strategy
    bot._strategy_for_symbol = Mock(return_value=strategy)
    bot._bars_needed_for_strategy = lambda selected_strategy: 1
    bot._get_intraday_bars = (
        lambda symbol, bars_needed, decision_timestamp=None: [bar]
    )
    bot.build_snapshot = Mock(
        side_effect=AssertionError(
            "strategy snapshot construction must not run for a rejected bar"
        )
    )
    bot.blog = SimpleNamespace(
        cycle_summary=Mock(),
        bar_timestamp_rejected=Mock(),
    )
    bot.broker = SimpleNamespace(submit_market_order=Mock())
    return bot, strategy_execute


def test_fresh_completed_aware_bar_is_accepted() -> None:
    validation = _validate(
        DECISION_TIME - timedelta(minutes=TIMEFRAME_MINUTES),
    )

    assert validation.accepted is True
    assert validation.status is trading_bot.BarFreshnessStatus.FRESH
    assert validation.bar_close_utc == DECISION_TIME
    assert validation.age_seconds == 0.0


def test_aware_non_utc_bar_is_normalized_and_accepted() -> None:
    eastern = timezone(timedelta(hours=-4))
    validation = _validate(datetime(2026, 4, 22, 11, 30, tzinfo=eastern))

    assert validation.accepted is True
    assert validation.bar_start_utc == datetime(
        2026,
        4,
        22,
        15,
        30,
        tzinfo=timezone.utc,
    )


def test_bar_at_exact_staleness_limit_is_accepted() -> None:
    validation = _validate(
        DECISION_TIME
        - timedelta(
            minutes=TIMEFRAME_MINUTES,
            seconds=MAX_DATA_DELAY_SECONDS,
        )
    )

    assert validation.accepted is True
    assert validation.age_seconds == MAX_DATA_DELAY_SECONDS


@pytest.mark.parametrize(
    ("raw_timestamp", "expected_status"),
    [
        (None, trading_bot.BarFreshnessStatus.MISSING_TIMESTAMP),
        ("not-a-timestamp", trading_bot.BarFreshnessStatus.INVALID_TIMESTAMP),
        (
            datetime(2026, 4, 22, 15, 30),
            trading_bot.BarFreshnessStatus.TIMEZONE_NAIVE_TIMESTAMP,
        ),
        (
            DECISION_TIME - timedelta(minutes=5),
            trading_bot.BarFreshnessStatus.FUTURE_OR_IN_PROGRESS_TIMESTAMP,
        ),
        (
            DECISION_TIME
            - timedelta(
                minutes=TIMEFRAME_MINUTES,
                seconds=MAX_DATA_DELAY_SECONDS + 1,
            ),
            trading_bot.BarFreshnessStatus.STALE_TIMESTAMP,
        ),
    ],
    ids=["missing", "invalid", "naive", "future_or_in_progress", "stale"],
)
def test_unsafe_bar_timestamp_has_structured_rejection(
    raw_timestamp: object,
    expected_status: trading_bot.BarFreshnessStatus,
) -> None:
    validation = _validate(raw_timestamp)

    assert validation.accepted is False
    assert validation.status is expected_status
    assert validation.symbol == "AMD"
    assert validation.decision_timestamp_utc == DECISION_TIME
    assert validation.reference_time_utc == DECISION_TIME
    assert validation.detail


@pytest.mark.parametrize(
    ("raw_timestamp", "expected_status"),
    [
        (None, trading_bot.BarFreshnessStatus.MISSING_TIMESTAMP),
        ("not-a-timestamp", trading_bot.BarFreshnessStatus.INVALID_TIMESTAMP),
        (
            datetime(2026, 4, 22, 15, 30),
            trading_bot.BarFreshnessStatus.TIMEZONE_NAIVE_TIMESTAMP,
        ),
        (
            DECISION_TIME - timedelta(minutes=5),
            trading_bot.BarFreshnessStatus.FUTURE_OR_IN_PROGRESS_TIMESTAMP,
        ),
        (
            DECISION_TIME
            - timedelta(
                minutes=TIMEFRAME_MINUTES,
                seconds=MAX_DATA_DELAY_SECONDS + 1,
            ),
            trading_bot.BarFreshnessStatus.STALE_TIMESTAMP,
        ),
    ],
    ids=["missing", "invalid", "naive", "future_or_in_progress", "stale"],
)
def test_run_once_rejects_before_strategy_or_broker_submission(
    raw_timestamp: object,
    expected_status: trading_bot.BarFreshnessStatus,
) -> None:
    bot, strategy_execute = _make_validation_bot(
        _rejected_bar(raw_timestamp),
    )

    with pytest.raises(trading_bot.UnsafeBarTimestampError) as exc_info:
        bot.run_once(execute_orders=True)

    assert exc_info.value.validation.status is expected_status
    strategy_execute.assert_not_called()
    bot._strategy_for_symbol.assert_not_called()
    bot.build_snapshot.assert_not_called()
    bot.broker.submit_market_order.assert_not_called()
    rejection = bot.blog.bar_timestamp_rejected.call_args.kwargs
    assert rejection["rejection_code"] == expected_status.value
    cycle_summary = bot.blog.cycle_summary.call_args.kwargs
    assert cycle_summary["skip_reason"] == (
        f"bar_timestamp_rejected:{expected_status.value}"
    )
    assert cycle_summary["orders_submitted"] == 0


def test_cycle_freshness_preflight_accepts_without_running_strategy() -> None:
    bot, strategy_execute = _make_validation_bot(_fresh_bar())

    validations = bot._validate_cycle_bar_freshness(
        decision_timestamp=DECISION_TIME,
        reference_time=DECISION_TIME,
    )

    assert validations["AMD"].status is trading_bot.BarFreshnessStatus.FRESH
    strategy_execute.assert_not_called()
    bot._strategy_for_symbol.assert_not_called()
    bot.blog.bar_timestamp_rejected.assert_not_called()


@pytest.mark.parametrize(
    ("raw_timestamp", "expected_status"),
    [
        (None, trading_bot.BarFreshnessStatus.MISSING_TIMESTAMP),
        ("not-a-timestamp", trading_bot.BarFreshnessStatus.INVALID_TIMESTAMP),
        (
            datetime(2026, 4, 22, 15, 30),
            trading_bot.BarFreshnessStatus.TIMEZONE_NAIVE_TIMESTAMP,
        ),
        (
            DECISION_TIME - timedelta(minutes=5),
            trading_bot.BarFreshnessStatus.FUTURE_OR_IN_PROGRESS_TIMESTAMP,
        ),
    ],
    ids=["missing", "invalid", "naive", "future_or_in_progress"],
)
def test_bar_retrieval_rejects_unsafe_timestamp_before_caching(
    raw_timestamp: object,
    expected_status: trading_bot.BarFreshnessStatus,
) -> None:
    bot, _ = _make_validation_bot(_fresh_bar())
    source_bar = BrokerBar(
        timestamp=raw_timestamp,  # type: ignore[arg-type]
        open=100.0,
        high=101.0,
        low=99.0,
        close=100.5,
        volume=1_000.0,
    )
    bot.broker.get_bars = Mock(return_value=[source_bar])

    with pytest.raises(trading_bot.UnsafeBarTimestampError) as exc_info:
        bot._get_bars(
            "AMD",
            bars_needed=1,
            timeframe_minutes=TIMEFRAME_MINUTES,
            decision_timestamp=DECISION_TIME,
        )

    assert exc_info.value.validation.status is expected_status
    assert bot._bars_cache == {}


def test_run_once_revalidates_freshness_before_submission() -> None:
    bot, _ = _make_validation_bot(_fresh_bar())
    reference_times = iter(
        [
            DECISION_TIME,
            DECISION_TIME + timedelta(seconds=MAX_DATA_DELAY_SECONDS + 1),
        ]
    )
    bot._utc_now = lambda: next(reference_times)
    bot.build_snapshot = Mock(
        return_value=trading_bot.BotSnapshot(
            timestamp_utc=DECISION_TIME.isoformat(),
            cash=1_000.0,
            buying_power=1_000.0,
            equity=1_000.0,
            last_equity=1_000.0,
            daily_pnl=0.0,
            kill_switch_triggered=False,
            positions={},
            symbols=[
                trading_bot.SymbolSnapshot(
                    symbol="AMD",
                    price=100.0,
                    sma=99.0,
                    action="BUY",
                    holding=False,
                    quantity=0.0,
                    market_value=0.0,
                )
            ],
        )
    )
    bot.get_startup_resync_result = lambda: SimpleNamespace(
        gate_allows_exits=True,
        status=SimpleNamespace(value="RESYNC_OK"),
    )
    bot._is_market_open = lambda: True
    bot._is_past_flatten_deadline = lambda now_et=None: False
    bot._kill_switch_warned_pcts = set()

    with pytest.raises(trading_bot.UnsafeBarTimestampError) as exc_info:
        bot.run_once(execute_orders=True)

    assert (
        exc_info.value.validation.status
        is trading_bot.BarFreshnessStatus.STALE_TIMESTAMP
    )
    bot.broker.submit_market_order.assert_not_called()
    cycle_summary = bot.blog.cycle_summary.call_args.kwargs
    assert cycle_summary["orders_submitted"] == 0
    assert cycle_summary["skip_reason"] == (
        "bar_timestamp_rejected:stale_timestamp"
    )


def test_run_once_revalidates_after_live_risk_work_before_buy_call() -> None:
    bot, _ = _make_validation_bot(_fresh_bar())
    reference_times = iter(
        [
            DECISION_TIME,
            DECISION_TIME,
            DECISION_TIME + timedelta(seconds=MAX_DATA_DELAY_SECONDS + 1),
        ]
    )
    bot._utc_now = lambda: next(reference_times)
    bot.build_snapshot = Mock(
        return_value=trading_bot.BotSnapshot(
            timestamp_utc=DECISION_TIME.isoformat(),
            cash=1_000.0,
            buying_power=1_000.0,
            equity=1_000.0,
            last_equity=1_000.0,
            daily_pnl=0.0,
            kill_switch_triggered=False,
            positions={},
            symbols=[
                trading_bot.SymbolSnapshot(
                    symbol="AMD",
                    price=100.0,
                    sma=99.0,
                    action="BUY",
                    holding=False,
                    quantity=0.0,
                    market_value=0.0,
                )
            ],
        )
    )
    bot.get_startup_resync_result = lambda: SimpleNamespace(
        gate_allows_entries=True,
        gate_allows_exits=True,
        status=SimpleNamespace(value="RESYNC_OK"),
    )
    bot._is_market_open = lambda: True
    bot._is_past_flatten_deadline = lambda now_et=None: False
    bot._is_in_entry_window = lambda now_et=None: True
    bot._resync_gate_for_action = lambda **kwargs: (True, "")
    bot.get_open_orders = lambda: []
    bot._count_recent_orders = lambda window_seconds: 0
    bot._active_symbol_event_blackout = lambda symbol, timestamp: None
    bot.get_latest_price_with_age = lambda symbol: (100.0, 0.0)
    bot._is_price_collar_breached = lambda signal_price, live_price: False
    bot._is_symbol_exposure_exceeded = (
        lambda symbol, live_price, positions: False
    )
    bot._log_risk_check = Mock()
    bot._kill_switch_warned_pcts = set()
    bot.config.max_orders_per_minute = 5
    bot.config.max_open_positions = 5
    bot.config.max_usd_per_trade = 500.0
    bot.config.max_live_price_age_seconds = 5
    bot.place_market_buy = Mock(
        side_effect=AssertionError("stale bar must block the buy call")
    )

    with pytest.raises(trading_bot.UnsafeBarTimestampError) as exc_info:
        bot.run_once(execute_orders=True)

    assert (
        exc_info.value.validation.status
        is trading_bot.BarFreshnessStatus.STALE_TIMESTAMP
    )
    bot.place_market_buy.assert_not_called()
    bot.broker.submit_market_order.assert_not_called()
    cycle_summary = bot.blog.cycle_summary.call_args.kwargs
    assert cycle_summary["orders_submitted"] == 0
    assert cycle_summary["skip_reason"] == (
        "bar_timestamp_rejected:stale_timestamp"
    )


def test_run_once_accepts_fresh_bar_and_reaches_preview_snapshot() -> None:
    bot, strategy_execute = _make_validation_bot(_fresh_bar())
    snapshot = trading_bot.BotSnapshot(
        timestamp_utc=DECISION_TIME.isoformat(),
        cash=1_000.0,
        buying_power=1_000.0,
        equity=1_000.0,
        last_equity=1_000.0,
        daily_pnl=0.0,
        kill_switch_triggered=False,
        positions={},
        symbols=[
            trading_bot.SymbolSnapshot(
                symbol="AMD",
                price=100.0,
                sma=99.0,
                action="HOLD",
                holding=False,
                quantity=0.0,
                market_value=0.0,
            )
        ],
    )
    bot.build_snapshot = Mock(return_value=snapshot)
    bot.get_startup_resync_result = lambda: SimpleNamespace(
        gate_allows_entries=True,
        gate_allows_exits=True,
        status=SimpleNamespace(value="RESYNC_OK"),
    )
    bot._kill_switch_warned_pcts = set()
    bot.record_state = Mock()

    result = bot.run_once(execute_orders=False)

    assert result is snapshot
    assert bot.build_snapshot.call_args.kwargs["evaluate_signals"] is True
    strategy_execute.assert_not_called()
    bot.broker.submit_market_order.assert_not_called()
    bot.record_state.assert_called_once_with(snapshot)
    cycle_summary = bot.blog.cycle_summary.call_args.kwargs
    assert cycle_summary["processed_bar"] is True
    assert cycle_summary["skip_reason"] == "preview_only"


def test_cached_aware_bar_is_normalized_and_revalidated_against_later_time() -> None:
    bot, _ = _make_validation_bot(_fresh_bar())
    del bot._get_intraday_bars
    eastern = timezone(timedelta(hours=-4))
    source_bar = BrokerBar(
        timestamp=datetime(2026, 4, 22, 11, 30, tzinfo=eastern),
        open=100.0,
        high=101.0,
        low=99.0,
        close=100.5,
        volume=1_000.0,
    )
    bot.broker.get_bars = Mock(return_value=[source_bar])

    validations = bot._validate_cycle_bar_freshness(
        decision_timestamp=DECISION_TIME,
        reference_time=DECISION_TIME,
    )

    assert validations["AMD"].status is trading_bot.BarFreshnessStatus.FRESH
    cached_bar = next(iter(bot._bars_cache.values()))[0]
    assert cached_bar.timestamp == datetime(
        2026,
        4,
        22,
        15,
        30,
        tzinfo=timezone.utc,
    )

    with pytest.raises(trading_bot.UnsafeBarTimestampError) as exc_info:
        bot._validate_cycle_bar_freshness(
            decision_timestamp=DECISION_TIME,
            reference_time=(
                DECISION_TIME + timedelta(seconds=MAX_DATA_DELAY_SECONDS + 1)
            ),
        )

    assert (
        exc_info.value.validation.status
        is trading_bot.BarFreshnessStatus.STALE_TIMESTAMP
    )
    bot.broker.get_bars.assert_called_once()

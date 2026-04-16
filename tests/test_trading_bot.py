from datetime import datetime, timezone
from types import SimpleNamespace

import trading_bot


def test_log_fills_skips_orders_filled_before_current_session() -> None:
    bot = trading_bot.AlpacaTradingBot.__new__(trading_bot.AlpacaTradingBot)
    bot._session_started_at = datetime(2026, 4, 13, 18, 0, tzinfo=timezone.utc)
    bot._logged_fills = set()
    bot._order_submission_ts = {}
    bot._order_signal_price = {}
    bot._order_submission_side = {}
    bot._order_exit_reason = {}
    bot._position_entry_price = {}
    bot._position_entry_ts = {}
    bot._position_qty = {}
    bot.config = SimpleNamespace(strategy_mode="mean_reversion", bar_timeframe_minutes=15)
    bot.blog = SimpleNamespace(
        order_partial_fill=lambda **kwargs: None,
        order_filled=lambda **kwargs: (_ for _ in ()).throw(AssertionError("historical fill should be ignored")),
        position_opened=lambda **kwargs: (_ for _ in ()).throw(AssertionError("historical fill should be ignored")),
        position_closed=lambda **kwargs: (_ for _ in ()).throw(AssertionError("historical fill should be ignored")),
    )
    bot.get_decision_timestamp = lambda: datetime(2026, 4, 13, 18, 15, tzinfo=timezone.utc)

    old_order = trading_bot.OrderSnapshot(
        order_id="historical-order",
        submitted_at="2026-04-10T18:07:55.326806+00:00",
        filled_at="2026-04-10T18:07:56.579333+00:00",
        symbol="QCOM",
        side="buy",
        status="filled",
        qty=1.0,
        filled_qty=1.0,
        filled_avg_price=128.62,
        notional=128.62,
    )

    bot._log_fills_from_orders([old_order])

    assert bot._logged_fills == set()


def test_log_fills_records_current_session_fill_once() -> None:
    bot = trading_bot.AlpacaTradingBot.__new__(trading_bot.AlpacaTradingBot)
    bot._session_started_at = datetime(2026, 4, 13, 18, 0, tzinfo=timezone.utc)
    bot._logged_fills = set()
    bot._order_submission_ts = {"live-order": "2026-04-13T18:15:00+00:00"}
    bot._order_signal_price = {"live-order": 244.425}
    bot._order_submission_side = {}
    bot._order_exit_reason = {}
    bot._position_entry_price = {}
    bot._position_entry_ts = {}
    bot._position_qty = {}
    bot.config = SimpleNamespace(strategy_mode="mean_reversion", bar_timeframe_minutes=15)
    recorded: list[tuple[str, dict[str, object]]] = []
    bot.blog = SimpleNamespace(
        order_partial_fill=lambda **kwargs: recorded.append(("partial", kwargs)),
        order_filled=lambda **kwargs: recorded.append(("filled", kwargs)),
        position_opened=lambda **kwargs: recorded.append(("opened", kwargs)),
        position_closed=lambda **kwargs: recorded.append(("closed", kwargs)),
    )
    bot.get_decision_timestamp = lambda: datetime.now(timezone.utc)
    bot._position_holding_minutes = lambda symbol, now: 15.0

    live_order = trading_bot.OrderSnapshot(
        order_id="live-order",
        submitted_at="2026-04-13T18:15:31.467459+00:00",
        filled_at="2026-04-13T18:15:32.289825+00:00",
        symbol="AMD",
        side="buy",
        status="filled",
        qty=4.0,
        filled_qty=4.0,
        filled_avg_price=243.8,
        notional=975.2,
    )

    bot._log_fills_from_orders([live_order, live_order])

    assert [event for event, _ in recorded] == ["filled", "opened"]
    assert bot._logged_fills == {("live-order", 4.0)}
    assert bot._position_entry_price["AMD"] == 243.8
    assert bot._position_qty["AMD"] == 4.0


def test_parse_iso_timestamp_handles_z_suffix() -> None:
    parsed = trading_bot._parse_iso_timestamp("2026-04-13T18:15:32.289825Z")

    assert parsed == datetime(2026, 4, 13, 18, 15, 32, 289825, tzinfo=timezone.utc)


def test_load_config_details_loads_dotenv_before_base_config(monkeypatch) -> None:
    monkeypatch.delenv("MAX_USD_PER_TRADE", raising=False)
    monkeypatch.delenv("BOT_SYMBOLS", raising=False)

    def fake_load_dotenv(path):
        monkeypatch.setenv("BOT_SYMBOLS", "AMD")
        monkeypatch.setenv("MAX_USD_PER_TRADE", "1000")
        return True

    monkeypatch.setattr(trading_bot, "load_dotenv", fake_load_dotenv)
    monkeypatch.setattr(trading_bot, "_load_runtime_config_payload", lambda: (None, None))

    details = trading_bot.load_config_details()

    assert details.config.max_usd_per_trade == 1000.0


def test_determine_signal_rejection_ignores_inactive_trend_filter() -> None:
    rejection = trading_bot._determine_signal_rejection(
        action="HOLD",
        holding=False,
        trend_filter_active=False,
        trend_pass=False,
        atr_pass=True,
    )

    assert rejection == "no_signal"


def test_determine_signal_rejection_prioritizes_holding_state() -> None:
    rejection = trading_bot._determine_signal_rejection(
        action="HOLD",
        holding=True,
        trend_filter_active=True,
        trend_pass=False,
        atr_pass=False,
    )

    assert rejection == "holding_no_exit"

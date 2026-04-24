from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from pathlib import Path
from uuid import uuid4
import json
import os

import pytest
import trading_bot
from storage import BotStorage


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
    bot._position_entry_branch = {}
    bot._order_entry_branch = {}
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
    bot._position_entry_branch = {}
    bot._order_entry_branch = {"live-order": "bollinger_breakout"}
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
    assert bot._position_entry_branch["AMD"] == "bollinger_breakout"


def test_log_fills_skips_duplicate_sell_fill_after_restart() -> None:
    db_path = Path.cwd() / f"test_botlog_{uuid4().hex}.db"
    storage = BotStorage(db_path)
    claimed = storage.claim_order_fill(
        "sell-order",
        8.0,
        "2026-04-21T17:31:04+00:00",
    )
    assert claimed is True

    bot = trading_bot.AlpacaTradingBot.__new__(trading_bot.AlpacaTradingBot)
    bot._session_started_at = datetime(2026, 4, 21, 17, 20, tzinfo=timezone.utc)
    bot._logged_fills = set()
    bot._order_submission_ts = {"sell-order": "2026-04-21T17:15:00+00:00"}
    bot._order_signal_price = {"sell-order": 123.35}
    bot._order_submission_side = {}
    bot._order_exit_reason = {"sell-order": "sell_signal"}
    bot._position_entry_price = {"COP": 123.35}
    bot._position_entry_ts = {"COP": "2026-04-21T17:15:00+00:00"}
    bot._position_qty = {"COP": 8.0}
    bot._position_entry_branch = {}
    bot._order_entry_branch = {}
    bot.storage = storage
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

    sell_order = trading_bot.OrderSnapshot(
        order_id="sell-order",
        submitted_at="2026-04-21T17:15:31.000000+00:00",
        filled_at="2026-04-21T17:31:04.000000+00:00",
        symbol="COP",
        side="sell",
        status="filled",
        qty=8.0,
        filled_qty=8.0,
        filled_avg_price=119.15,
        notional=953.2,
    )

    bot._log_fills_from_orders([sell_order])

    assert recorded == []
    assert bot._logged_fills == {("sell-order", 8.0)}


def test_run_once_duplicate_global_claim_sets_cycle_report_without_hold_reasons() -> None:
    bot = trading_bot.AlpacaTradingBot.__new__(trading_bot.AlpacaTradingBot)
    decision_timestamp = datetime(2026, 4, 20, 14, 45, tzinfo=timezone.utc)
    snapshot = trading_bot.BotSnapshot(
        timestamp_utc=decision_timestamp.isoformat(),
        cash=0.0,
        buying_power=0.0,
        equity=0.0,
        last_equity=0.0,
        daily_pnl=0.0,
        kill_switch_triggered=False,
        positions={},
        symbols=[],
    )
    recorded_states: list[trading_bot.BotSnapshot] = []
    cycle_summaries: list[dict[str, object]] = []

    bot._bars_cache = {}
    bot._hourly_regime_cache = {}
    bot._et_now = lambda: decision_timestamp
    bot.get_decision_timestamp = lambda: decision_timestamp
    bot._should_process_decision_timestamp = lambda ts: True
    bot._claim_global_decision_execution = lambda ts: False
    bot.build_snapshot = lambda decision_timestamp, evaluate_signals: snapshot
    bot.record_state = lambda state: recorded_states.append(state)
    bot.blog = SimpleNamespace(cycle_summary=lambda **kwargs: cycle_summaries.append(kwargs))

    result = bot.run_once(execute_orders=True)

    assert result is snapshot
    assert recorded_states == [snapshot]
    assert cycle_summaries == [
        {
            "decision_ts": decision_timestamp.isoformat(),
            "execute_orders": True,
            "processed_bar": False,
            "skip_reason": "duplicate_bar",
            "buy_signals": 0,
            "sell_signals": 0,
            "hold_signals": 0,
            "error_signals": 0,
            "orders_submitted": 0,
        }
    ]
    assert bot._last_run_cycle_report == trading_bot.RunCycleReport(
        decision_timestamp=decision_timestamp.isoformat(),
        execute_orders=True,
        processed_bar=False,
        skip_reason="duplicate_bar",
        buy_signals=0,
        sell_signals=0,
        hold_signals=0,
        error_signals=0,
        orders_submitted=0,
    )


def test_update_position_holding_state_restores_open_position_from_logs() -> None:
    bot = trading_bot.AlpacaTradingBot.__new__(trading_bot.AlpacaTradingBot)
    bot._position_entry_price = {}
    bot._position_entry_ts = {}
    bot._position_qty = {}
    bot._position_entry_branch = {}
    bot._position_first_seen_utc = {}
    bot._breakout_stored_stop = {}
    bot.blog = SimpleNamespace(log_root="logs")

    original_loader = trading_bot._load_open_position_state_from_logs
    trading_bot._load_open_position_state_from_logs = lambda log_root: {
        "BAC": {
            "entry_price": 53.81,
            "qty": 18.0,
            "decision_ts": "2026-04-16T16:30:00+00:00",
            "entry_branch": "mean_reversion",
        }
    }
    try:
        positions = {
            "BAC": trading_bot.Position(
                symbol="BAC",
                qty=18.0,
                market_value=975.0,
                avg_entry_price=54.15,
                current_price=54.15,
            )
        }
        observed_at = datetime(2026, 4, 17, 13, 45, tzinfo=timezone.utc)

        bot._update_position_holding_state(positions, observed_at)
    finally:
        trading_bot._load_open_position_state_from_logs = original_loader

    assert bot._position_entry_price["BAC"] == 53.81
    assert bot._position_qty["BAC"] == 18.0
    assert bot._position_entry_ts["BAC"] == "2026-04-16T16:30:00+00:00"
    assert bot._position_entry_branch["BAC"] == "mean_reversion"
    assert bot._position_first_seen_utc["BAC"] == datetime(2026, 4, 16, 16, 30, tzinfo=timezone.utc)


def test_parse_iso_timestamp_handles_z_suffix() -> None:
    parsed = trading_bot._parse_iso_timestamp("2026-04-13T18:15:32.289825Z")

    assert parsed == datetime(2026, 4, 13, 18, 15, 32, 289825, tzinfo=timezone.utc)


def test_apply_runtime_config_filters_excluded_symbols_and_parses_blackouts() -> None:
    base = trading_bot.BotConfig(
        symbols=["ABBV", "AMD", "HON"],
        max_usd_per_trade=1000.0,
        max_symbol_exposure_usd=1000.0,
        max_open_positions=2,
        max_daily_loss_usd=500.0,
        sma_bars=15,
        bar_timeframe_minutes=15,
    )

    config, changed = trading_bot._apply_runtime_config(
        base,
        runtime_path=trading_bot.Path("config/live_config.json"),
        runtime={
            "excluded_symbols": ["HON", "ABBV"],
            "symbol_event_blackouts": [
                {
                    "symbol": "AMD",
                    "start_utc": "2026-04-21T13:30:00Z",
                    "end_utc": "2026-04-21T20:00:00Z",
                    "reason": "earnings_blackout",
                }
            ],
        },
    )

    assert config.symbols == ["AMD"]
    assert config.excluded_symbols == ("ABBV", "HON")
    assert len(config.symbol_event_blackouts) == 1
    assert config.symbol_event_blackouts[0].symbol == "AMD"
    assert config.symbol_event_blackouts[0].reason == "earnings_blackout"
    assert "excluded_symbols" in changed
    assert "symbol_event_blackouts" in changed


def test_apply_runtime_config_parses_data_parity_block() -> None:
    base = trading_bot.BotConfig(
        symbols=["AMD", "MSFT"],
        max_usd_per_trade=1000.0,
        max_symbol_exposure_usd=1000.0,
        max_open_positions=2,
        max_daily_loss_usd=500.0,
        sma_bars=15,
        bar_timeframe_minutes=15,
    )

    config, changed = trading_bot._apply_runtime_config(
        base,
        runtime_path=trading_bot.Path("config/live_config.json"),
        runtime={
            "data_parity": {
                "historical_feed": "sip",
                "live_feed": "iex",
                "latest_bar_feed": "sip",
                "bar_build_mode": "historical_preaggregated",
                "apply_updated_bars": False,
                "post_bar_reconcile_poll": False,
                "block_trading_until_resync": True,
                "assert_feed_on_startup": True,
                "log_bar_components": False,
            }
        },
    )

    assert config.historical_feed == "sip"
    assert config.live_feed == "iex"
    assert config.latest_bar_feed == "sip"
    assert config.bar_build_mode == "historical_preaggregated"
    assert config.apply_updated_bars is False
    assert config.post_bar_reconcile_poll is False
    assert config.block_trading_until_resync is True
    assert config.assert_feed_on_startup is True
    assert config.log_bar_components is False
    assert "historical_feed" in changed
    assert "bar_build_mode" in changed


def test_active_symbol_event_blackout_matches_window() -> None:
    bot = trading_bot.AlpacaTradingBot.__new__(trading_bot.AlpacaTradingBot)
    bot.config = SimpleNamespace(
        symbol_event_blackouts=(
            trading_bot.SymbolEventBlackout(
                symbol="AMD",
                start_utc="2026-04-21T13:30:00Z",
                end_utc="2026-04-21T20:00:00Z",
                reason="earnings_blackout",
            ),
        )
    )

    active = bot._active_symbol_event_blackout("AMD", datetime(2026, 4, 21, 15, 0, tzinfo=timezone.utc))
    inactive = bot._active_symbol_event_blackout("AMD", datetime(2026, 4, 22, 15, 0, tzinfo=timezone.utc))

    assert active is not None
    assert active.reason == "earnings_blackout"
    assert inactive is None


def _build_resync_bot(*, latest_run: dict | None, latest_symbols: list[dict], order_history: list[dict], lifecycle_events: list[tuple[str, dict]]) -> trading_bot.AlpacaTradingBot:
    bot = trading_bot.AlpacaTradingBot.__new__(trading_bot.AlpacaTradingBot)
    bot.config = SimpleNamespace(block_trading_until_resync=True)
    bot.session_id = "session-test"
    bot.storage = SimpleNamespace(
        get_latest_run=lambda: latest_run,
        get_latest_symbol_snapshot=lambda: latest_symbols,
        get_order_history=lambda limit=200: order_history,
    )
    bot.blog = SimpleNamespace(
        lifecycle=lambda stage, **fields: lifecycle_events.append((stage, fields)),
    )
    bot._startup_resync_result = trading_bot.ResyncResult(
        status=trading_bot.ResyncStatus.LOCKED,
        started_at_utc="2026-04-22T13:00:00+00:00",
        completed_at_utc=None,
        gate_allows_entries=False,
        gate_allows_exits=False,
    )
    bot._position_entry_price = {}
    bot._position_entry_ts = {}
    bot._position_qty = {}
    bot._position_entry_branch = {}
    bot._position_first_seen_utc = {}
    bot._order_submission_ts = {}
    bot._order_submission_side = {}
    bot._order_exit_reason = {}
    bot._logged_fills = set()
    bot.data_parity_summary = lambda: {"historical_feed": "iex", "live_feed": "iex", "latest_bar_feed": "iex"}
    return bot


def test_perform_startup_resync_recovers_broker_position_missing_local() -> None:
    lifecycle_events: list[tuple[str, dict]] = []
    bot = _build_resync_bot(latest_run=None, latest_symbols=[], order_history=[], lifecycle_events=lifecycle_events)
    bot.get_positions_by_symbol = lambda: {
        "AMD": SimpleNamespace(symbol="AMD", qty=5.0, market_value=1000.0, avg_entry_price=200.0, current_price=205.0)
    }
    bot.get_open_orders = lambda: []
    bot.get_recent_orders = lambda limit=50: []

    result = bot.perform_startup_resync()

    assert result.status == trading_bot.ResyncStatus.OK
    assert result.gate_allows_entries is True
    assert result.gate_allows_exits is True
    assert "RESYNC_BROKER_POSITION_LOCAL_MISSING" in result.reason_codes
    assert result.positions_recovered[0]["symbol"] == "AMD"
    assert bot._position_qty["AMD"] == 5.0
    assert lifecycle_events[0][0] == "resync.started"
    assert lifecycle_events[-1][0] == "resync.completed"


def test_perform_startup_resync_marks_lookback_insufficient_as_degraded() -> None:
    lifecycle_events: list[tuple[str, dict]] = []
    current_time = datetime(2026, 4, 22, 13, 0, tzinfo=timezone.utc)
    bot = _build_resync_bot(
        latest_run={"timestamp_utc": "2026-04-22T10:00:00+00:00"},
        latest_symbols=[],
        order_history=[],
        lifecycle_events=lifecycle_events,
    )
    bot.get_positions_by_symbol = lambda: {
        "MSFT": SimpleNamespace(symbol="MSFT", qty=3.0, market_value=900.0, avg_entry_price=300.0, current_price=305.0)
    }
    bot.get_open_orders = lambda: []
    bot.get_recent_orders = lambda limit=50: []

    result = bot.perform_startup_resync(lookback_minutes=60, current_time_utc=current_time)

    assert result.status == trading_bot.ResyncStatus.DEGRADED
    assert result.gate_allows_entries is False
    assert result.gate_allows_exits is True
    assert "RESYNC_LOOKBACK_INSUFFICIENT" in result.reason_codes


def test_perform_startup_resync_allows_overnight_carry_restart_without_open_orders() -> None:
    lifecycle_events: list[tuple[str, dict]] = []
    current_time = datetime(2026, 4, 23, 13, 0, tzinfo=timezone.utc)
    bot = _build_resync_bot(
        latest_run={"timestamp_utc": "2026-04-22T20:15:00+00:00"},
        latest_symbols=[],
        order_history=[],
        lifecycle_events=lifecycle_events,
    )
    bot.get_positions_by_symbol = lambda: {
        "COP": SimpleNamespace(symbol="COP", qty=-8.0, market_value=-960.0, avg_entry_price=119.13, current_price=124.15)
    }
    bot.get_open_orders = lambda: []
    bot.get_recent_orders = lambda limit=50: []

    result = bot.perform_startup_resync(lookback_minutes=240, current_time_utc=current_time)

    assert result.status == trading_bot.ResyncStatus.OK
    assert result.gate_allows_entries is True
    assert result.gate_allows_exits is True
    assert "RESYNC_BROKER_POSITION_LOCAL_MISSING" in result.reason_codes
    assert "RESYNC_LOOKBACK_INSUFFICIENT" not in result.reason_codes


def test_perform_startup_resync_is_idempotent() -> None:
    lifecycle_events: list[tuple[str, dict]] = []
    bot = _build_resync_bot(latest_run=None, latest_symbols=[], order_history=[], lifecycle_events=lifecycle_events)
    bot.get_positions_by_symbol = lambda: {}
    bot.get_open_orders = lambda: []
    bot.get_recent_orders = lambda limit=50: []

    first = bot.perform_startup_resync()
    second = bot.perform_startup_resync()

    assert first == second
    assert [stage for stage, _ in lifecycle_events].count("resync.started") == 1
    assert [stage for stage, _ in lifecycle_events].count("resync.completed") == 1


def test_perform_startup_resync_marks_broker_query_failure_as_failed() -> None:
    lifecycle_events: list[tuple[str, dict]] = []
    bot = _build_resync_bot(latest_run=None, latest_symbols=[], order_history=[], lifecycle_events=lifecycle_events)
    bot.get_positions_by_symbol = lambda: (_ for _ in ()).throw(RuntimeError("broker unavailable"))
    bot.get_open_orders = lambda: []
    bot.get_recent_orders = lambda limit=50: []

    result = bot.perform_startup_resync()

    assert result.status == trading_bot.ResyncStatus.FAILED
    assert result.gate_allows_entries is False


def test_perform_startup_preflight_logs_ready() -> None:
    lifecycle_events: list[tuple[str, dict]] = []
    bot = trading_bot.AlpacaTradingBot.__new__(trading_bot.AlpacaTradingBot)
    bot.blog = SimpleNamespace(lifecycle=lambda stage, **fields: lifecycle_events.append((stage, fields)))
    bot.get_account = lambda: SimpleNamespace(cash=1000.0, buying_power=1500.0, equity=1100.0)
    bot.get_positions_by_symbol = lambda: {
        "AMD": SimpleNamespace(symbol="AMD", qty=1.0, market_value=100.0, avg_entry_price=100.0, current_price=100.0)
    }
    bot._is_market_open = lambda: True
    bot.get_price_feed_status = lambda: "live stream active"

    summary = bot.perform_startup_preflight(execute_orders=True)

    assert summary == {
        "execute_orders": True,
        "cash": 1000.0,
        "buying_power": 1500.0,
        "equity": 1100.0,
        "position_count": 1,
        "market_open": True,
        "feed_status": "live stream active",
    }
    assert lifecycle_events == [
        (
            "startup.ready",
            {
                "execute_orders": True,
                "cash": 1000.0,
                "buying_power": 1500.0,
                "equity": 1100.0,
                "position_count": 1,
                "market_open": True,
                "feed_status": "live stream active",
            },
        )
    ]


def test_main_logs_startup_failed_when_preflight_raises(monkeypatch) -> None:
    lifecycle_events: list[tuple[str, dict]] = []
    fake_bot = SimpleNamespace(
        perform_startup_resync=lambda: None,
        perform_startup_preflight=lambda *, execute_orders: (_ for _ in ()).throw(RuntimeError("broker unavailable")),
        blog=SimpleNamespace(lifecycle=lambda stage, **fields: lifecycle_events.append((stage, fields))),
        build_snapshot=lambda evaluate_signals=False: SimpleNamespace(),
        record_state=lambda snapshot: None,
    )

    monkeypatch.setattr(trading_bot, "load_dotenv", lambda path: None)
    monkeypatch.setattr(
        trading_bot,
        "TradeOSBot",
        lambda config, session_id=None: fake_bot,
    )

    with pytest.raises(RuntimeError, match="broker unavailable"):
        trading_bot.main(
            config=trading_bot.BotConfig(
                symbols=["AMD"],
                max_usd_per_trade=200.0,
                max_symbol_exposure_usd=200.0,
                max_open_positions=3,
                max_daily_loss_usd=300.0,
                sma_bars=20,
                bar_timeframe_minutes=15,
            ),
            session_id="test-session",
        )

    assert lifecycle_events == [
        (
            "startup.failed",
            {
                "reason": "startup_preflight_error",
                "execute_orders": True,
                "error": "broker unavailable",
            },
        )
    ]


def test_perform_startup_resync_recovers_recent_fill() -> None:
    lifecycle_events: list[tuple[str, dict]] = []
    current_time = datetime(2026, 4, 22, 13, 30, tzinfo=timezone.utc)
    bot = _build_resync_bot(latest_run=None, latest_symbols=[], order_history=[], lifecycle_events=lifecycle_events)
    bot.get_positions_by_symbol = lambda: {}
    bot.get_open_orders = lambda: []
    bot.get_recent_orders = lambda limit=50: [
        trading_bot.OrderSnapshot(
            order_id="fill-1",
            submitted_at="2026-04-22T12:55:00+00:00",
            filled_at="2026-04-22T12:56:00+00:00",
            symbol="AMD",
            side="buy",
            status="filled",
            qty=2.0,
            filled_qty=2.0,
            filled_avg_price=101.5,
            notional=203.0,
        )
    ]

    result = bot.perform_startup_resync(lookback_minutes=240, current_time_utc=current_time)

    assert result.status == trading_bot.ResyncStatus.OK
    assert "RESYNC_RECENT_FILL_RECOVERED" in result.reason_codes
    assert result.recent_fills_recovered[0]["order_id"] == "fill-1"
    assert ("fill-1", 2.0) in bot._logged_fills


def test_update_startup_artifact_with_resync_writes_gate_state() -> None:
    artifact_dir = Path.cwd() / f"test_resync_artifact_{uuid4().hex}"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    artifact_path = artifact_dir / "startup_config.20260422T130000Z.json"
    artifact_path.write_text(json.dumps({"session_id": "session-test"}), encoding="utf-8")

    bot = _build_resync_bot(latest_run=None, latest_symbols=[], order_history=[], lifecycle_events=[])
    result = trading_bot.ResyncResult(
        status=trading_bot.ResyncStatus.DEGRADED,
        started_at_utc="2026-04-22T13:00:00+00:00",
        completed_at_utc="2026-04-22T13:00:05+00:00",
        reason_codes=("RESYNC_LOOKBACK_INSUFFICIENT",),
        gate_allows_entries=False,
        gate_allows_exits=True,
    )

    previous = os.environ.get(trading_bot.STARTUP_ARTIFACT_PATH_ENV)
    os.environ[trading_bot.STARTUP_ARTIFACT_PATH_ENV] = str(artifact_path)
    try:
        bot._update_startup_artifact_with_resync(result)
        payload = json.loads(artifact_path.read_text(encoding="utf-8"))
        latest_payload = json.loads((artifact_dir / "startup_config.json").read_text(encoding="utf-8"))
        assert payload["resync_status"] == "RESYNC_DEGRADED"
        assert payload["gate_allows_entries"] is False
        assert payload["gate_allows_exits"] is True
        assert payload["resync_reason_codes"] == ["RESYNC_LOOKBACK_INSUFFICIENT"]
        assert latest_payload == payload
    finally:
        if previous is None:
            os.environ.pop(trading_bot.STARTUP_ARTIFACT_PATH_ENV, None)
        else:
            os.environ[trading_bot.STARTUP_ARTIFACT_PATH_ENV] = previous
        for child in sorted(artifact_dir.rglob("*"), reverse=True):
            if child.is_file():
                child.unlink()
            else:
                child.rmdir()


def test_evaluate_symbol_does_not_log_trend_filter_reject_when_mr_trend_filter_disabled() -> None:
    bot = trading_bot.AlpacaTradingBot.__new__(trading_bot.AlpacaTradingBot)
    bot.config = trading_bot.BotConfig(
        symbols=["C"],
        max_usd_per_trade=1000.0,
        max_symbol_exposure_usd=1000.0,
        max_open_positions=5,
        max_daily_loss_usd=300.0,
        sma_bars=15,
        bar_timeframe_minutes=15,
        strategy_mode="mean_reversion",
        mean_reversion_trend_filter=False,
    )
    bot._bars_cache = {}
    bot._hourly_regime_cache = {}
    bot._position_entry_branch = {}
    bot._position_first_seen_utc = {}
    bot._ml_disabled_reason = None

    captured: list[dict[str, object]] = []
    bot.blog = SimpleNamespace(
        bar_received=lambda **kwargs: None,
        signal=lambda **kwargs: captured.append(kwargs),
    )
    strategy = trading_bot.Strategy(
        trading_bot.StrategyConfig(
            strategy_mode=trading_bot.STRATEGY_MODE_MEAN_REVERSION,
            mean_reversion_trend_filter=False,
        )
    )
    strategy.decide = lambda **kwargs: SimpleNamespace(
        action="BUY",
        reason="mean_reversion_sma_entry",
        hybrid_branch=None,
        mr_signal=None,
        hybrid_branch_active=None,
        hybrid_entry_branch=None,
        hybrid_regime_branch=None,
    )
    bot._strategy_for_symbol = lambda symbol: strategy
    bot._get_hourly_regime = lambda symbol, decision_timestamp: True

    start = datetime(2026, 4, 22, 1, 0, tzinfo=timezone.utc)
    closes = [110.0] * 54 + [100.0]
    bot._get_intraday_bars = lambda symbol, bars_needed, decision_timestamp=None: [
        trading_bot.BrokerBar(
            timestamp=start + timedelta(minutes=15 * idx),
            open=close,
            high=close + 1.0,
            low=close - 1.0,
            close=close,
            volume=1000.0,
        )
        for idx, close in enumerate(closes)
    ]

    evaluation = bot.evaluate_symbol(
        "C",
        position=None,
        decision_timestamp=datetime(2026, 4, 22, 14, 45, tzinfo=timezone.utc),
    )

    assert evaluation.action == "BUY"
    assert len(captured) == 1
    assert captured[0]["trend_filter_pass"] is None
    assert captured[0]["extra_fields"]["trend_filter_active"] is False


def test_resync_gate_allows_only_exits_in_degraded_mode() -> None:
    bot = _build_resync_bot(latest_run=None, latest_symbols=[], order_history=[], lifecycle_events=[])
    bot._startup_resync_result = trading_bot.ResyncResult(
        status=trading_bot.ResyncStatus.DEGRADED,
        started_at_utc="2026-04-22T13:00:00+00:00",
        completed_at_utc="2026-04-22T13:00:05+00:00",
        gate_allows_entries=False,
        gate_allows_exits=True,
    )
    positions = {"MSFT": SimpleNamespace(symbol="MSFT", qty=2.0)}

    buy_allowed, buy_reason = bot._resync_gate_for_action(action="BUY", symbol="AMD", positions=positions)
    sell_allowed, sell_reason = bot._resync_gate_for_action(action="SELL", symbol="MSFT", positions=positions)
    stray_sell_allowed, stray_sell_reason = bot._resync_gate_for_action(action="SELL", symbol="AMD", positions=positions)

    assert buy_allowed is False
    assert buy_reason == "resync_entries_disabled"
    assert sell_allowed is True
    assert sell_reason is None
    assert stray_sell_allowed is False
    assert stray_sell_reason == "resync_exit_requires_broker_position"


def test_load_config_details_loads_dotenv_before_base_config(monkeypatch) -> None:
    monkeypatch.delenv("MAX_USD_PER_TRADE", raising=False)
    monkeypatch.delenv("BOT_SYMBOLS", raising=False)

    def fake_load_dotenv(path):
        monkeypatch.setenv("BOT_SYMBOLS", "AMD")
        monkeypatch.setenv("MAX_USD_PER_TRADE", "1000")
        return True

    monkeypatch.setattr(trading_bot, "load_dotenv", fake_load_dotenv)
    monkeypatch.setattr(trading_bot, "_load_runtime_config_payload", lambda: (None, None, None))

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


def test_trading_bot_imports_sma_strategy_constant() -> None:
    assert trading_bot.STRATEGY_MODE_SMA == "sma"


def test_apply_runtime_config_accepts_trend_pullback_fields() -> None:
    base = trading_bot.BotConfig(
        symbols=["AMD"],
        max_usd_per_trade=1000.0,
        max_symbol_exposure_usd=1000.0,
        max_open_positions=2,
        max_daily_loss_usd=500.0,
        sma_bars=15,
        bar_timeframe_minutes=15,
    )

    config, changed = trading_bot._apply_runtime_config(
        base,
        runtime_path=trading_bot.Path("config/live_config.json"),
        runtime={
            "strategy_mode": "trend_pullback",
            "trend_pullback_min_adx": 27.0,
            "trend_pullback_min_slope": 0.05,
            "trend_pullback_entry_threshold": 0.002,
            "trend_pullback_min_atr_percentile": 30.0,
            "trend_pullback_max_atr_percentile": 80.0,
            "trend_pullback_exit_style": "hybrid_tp_or_time",
            "trend_pullback_hold_bars": 4,
            "trend_pullback_take_profit_pct": 0.0025,
            "trend_pullback_stop_pct": 0.01,
        },
    )

    assert config.strategy_mode == "trend_pullback"
    assert config.trend_pullback_min_adx == 27.0
    assert config.trend_pullback_min_slope == 0.05
    assert config.trend_pullback_entry_threshold == 0.002
    assert config.trend_pullback_min_atr_percentile == 30.0
    assert config.trend_pullback_max_atr_percentile == 80.0
    assert config.trend_pullback_exit_style == "hybrid_tp_or_time"
    assert config.trend_pullback_hold_bars == 4
    assert config.trend_pullback_take_profit_pct == 0.0025
    assert config.trend_pullback_stop_pct == 0.01
    assert "strategy_mode" in changed


def test_apply_runtime_config_accepts_mean_reversion_stop_pct() -> None:
    base = trading_bot.BotConfig(
        symbols=["AMD"],
        max_usd_per_trade=1000.0,
        max_symbol_exposure_usd=1000.0,
        max_open_positions=2,
        max_daily_loss_usd=500.0,
        sma_bars=15,
        bar_timeframe_minutes=15,
    )

    config, changed = trading_bot._apply_runtime_config(
        base,
        runtime_path=trading_bot.Path("config/live_config.json"),
        runtime={
            "strategy_mode": "mean_reversion",
            "mean_reversion_stop_pct": 0.01,
        },
    )

    assert config.strategy_mode == "mean_reversion"
    assert config.mean_reversion_stop_pct == 0.01
    assert "mean_reversion_stop_pct" in changed


def test_apply_runtime_config_accepts_live_safety_controls() -> None:
    base = trading_bot.BotConfig(
        symbols=["AMD"],
        max_usd_per_trade=1000.0,
        max_symbol_exposure_usd=1000.0,
        max_open_positions=2,
        max_daily_loss_usd=500.0,
        sma_bars=15,
        bar_timeframe_minutes=15,
    )

    config, changed = trading_bot._apply_runtime_config(
        base,
        runtime_path=trading_bot.Path("config/live_config.json"),
        runtime={
            "ml_lookback_bars": 400,
            "breakout_max_stop_pct": 0.01,
            "sma_stop_pct": 0.02,
            "max_orders_per_minute": 2,
            "max_price_deviation_bps": 10.0,
            "max_data_delay_seconds": 30,
            "max_live_price_age_seconds": 5,
        },
    )

    assert config.ml_lookback_bars == 400
    assert config.breakout_max_stop_pct == 0.01
    assert config.sma_stop_pct == 0.02
    assert config.max_orders_per_minute == 2
    assert config.max_price_deviation_bps == 10.0
    assert config.max_data_delay_seconds == 30
    assert config.max_live_price_age_seconds == 5
    assert "ml_lookback_bars" in changed
    assert "breakout_max_stop_pct" in changed
    assert "sma_stop_pct" in changed
    assert "max_orders_per_minute" in changed
    assert "max_price_deviation_bps" in changed
    assert "max_data_delay_seconds" in changed
    assert "max_live_price_age_seconds" in changed


def test_update_position_holding_state_restores_recovered_short_position() -> None:
    bot = trading_bot.AlpacaTradingBot.__new__(trading_bot.AlpacaTradingBot)
    bot._position_entry_price = {}
    bot._position_entry_ts = {}
    bot._position_qty = {}
    bot._position_entry_branch = {}
    bot._position_first_seen_utc = {}
    bot._breakout_stored_stop = {}
    bot.blog = SimpleNamespace(log_root="logs")

    original_loader = trading_bot._load_open_position_state_from_logs
    trading_bot._load_open_position_state_from_logs = lambda log_root: {
        "COP": {
            "entry_price": 119.13,
            "qty": -8.0,
            "decision_ts": "2026-04-22T14:15:00+00:00",
            "entry_branch": None,
        }
    }
    try:
        positions = {
            "COP": trading_bot.Position(
                symbol="COP",
                qty=-8.0,
                market_value=-980.4,
                avg_entry_price=119.13,
                current_price=122.55,
            )
        }
        observed_at = datetime(2026, 4, 22, 16, 7, tzinfo=timezone.utc)

        bot._update_position_holding_state(positions, observed_at)
    finally:
        trading_bot._load_open_position_state_from_logs = original_loader

    assert bot._position_entry_price["COP"] == 119.13
    assert bot._position_qty["COP"] == -8.0
    assert bot._position_entry_ts["COP"] == "2026-04-22T14:15:00+00:00"


def test_log_fills_treats_buy_against_recovered_short_as_cover_not_new_long() -> None:
    bot = trading_bot.AlpacaTradingBot.__new__(trading_bot.AlpacaTradingBot)
    bot._session_started_at = datetime(2026, 4, 22, 16, 0, tzinfo=timezone.utc)
    bot._logged_fills = set()
    bot._order_submission_ts = {"cover-order": "2026-04-22T16:10:00+00:00"}
    bot._order_signal_price = {}
    bot._order_submission_side = {}
    bot._order_exit_reason = {"cover-order": "eod_flatten"}
    bot._position_entry_price = {"COP": 119.13}
    bot._position_entry_ts = {"COP": "2026-04-22T14:15:00+00:00"}
    bot._position_qty = {"COP": -8.0}
    bot._position_entry_branch = {}
    bot._order_entry_branch = {}
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

    cover_order = trading_bot.OrderSnapshot(
        order_id="cover-order",
        submitted_at="2026-04-22T16:10:10+00:00",
        filled_at="2026-04-22T16:10:11+00:00",
        symbol="COP",
        side="buy",
        status="filled",
        qty=8.0,
        filled_qty=8.0,
        filled_avg_price=122.7,
        notional=981.6,
    )

    bot._log_fills_from_orders([cover_order])

    assert [event for event, _ in recorded] == ["filled"]
    assert "COP" not in bot._position_qty
    assert "COP" not in bot._position_entry_price


def test_apply_runtime_config_accepts_momentum_breakout_fields() -> None:
    base = trading_bot.BotConfig(
        symbols=["AMD"],
        max_usd_per_trade=1000.0,
        max_symbol_exposure_usd=1000.0,
        max_open_positions=2,
        max_daily_loss_usd=500.0,
        sma_bars=15,
        bar_timeframe_minutes=15,
    )

    config, changed = trading_bot._apply_runtime_config(
        base,
        runtime_path=trading_bot.Path("config/momentum_breakout.example.json"),
        runtime={
            "strategy_mode": "momentum_breakout",
            "momentum_breakout_lookback_bars": 30,
            "momentum_breakout_entry_buffer_pct": 0.002,
            "momentum_breakout_min_adx": 25.0,
            "momentum_breakout_min_slope": 0.05,
            "momentum_breakout_min_atr_percentile": 35.0,
            "momentum_breakout_exit_style": "fixed_bars",
            "momentum_breakout_hold_bars": 4,
            "momentum_breakout_stop_pct": 0.01,
            "momentum_breakout_take_profit_pct": 0.03,
        },
    )

    assert config.strategy_mode == "momentum_breakout"
    assert config.momentum_breakout_lookback_bars == 30
    assert config.momentum_breakout_entry_buffer_pct == 0.002
    assert config.momentum_breakout_min_adx == 25.0
    assert config.momentum_breakout_min_slope == 0.05
    assert config.momentum_breakout_min_atr_percentile == 35.0
    assert config.momentum_breakout_exit_style == "fixed_bars"
    assert config.momentum_breakout_hold_bars == 4
    assert config.momentum_breakout_stop_pct == 0.01
    assert config.momentum_breakout_take_profit_pct == 0.03
    assert "strategy_mode" in changed


def test_apply_runtime_config_accepts_volatility_expansion_fields() -> None:
    base = trading_bot.BotConfig(
        symbols=["AMD"],
        max_usd_per_trade=1000.0,
        max_symbol_exposure_usd=1000.0,
        max_open_positions=2,
        max_daily_loss_usd=500.0,
        sma_bars=15,
        bar_timeframe_minutes=15,
    )

    config, changed = trading_bot._apply_runtime_config(
        base,
        runtime_path=trading_bot.Path("config/volatility_expansion.example.json"),
        runtime={
            "strategy_mode": "volatility_expansion",
            "volatility_expansion_lookback_bars": 25,
            "volatility_expansion_entry_buffer_pct": 0.002,
            "volatility_expansion_max_atr_percentile": 30.0,
            "volatility_expansion_trend_filter": True,
            "volatility_expansion_min_slope": 0.05,
            "volatility_expansion_use_volume_confirm": False,
            "volatility_expansion_exit_style": "fixed_bars",
            "volatility_expansion_hold_bars": 5,
            "volatility_expansion_stop_pct": 0.01,
            "volatility_expansion_take_profit_pct": 0.03,
        },
    )

    assert config.strategy_mode == "volatility_expansion"
    assert config.volatility_expansion_lookback_bars == 25
    assert config.volatility_expansion_entry_buffer_pct == 0.002
    assert config.volatility_expansion_max_atr_percentile == 30.0
    assert config.volatility_expansion_trend_filter is True
    assert config.volatility_expansion_min_slope == 0.05
    assert config.volatility_expansion_use_volume_confirm is False
    assert config.volatility_expansion_exit_style == "fixed_bars"
    assert config.volatility_expansion_hold_bars == 5
    assert config.volatility_expansion_stop_pct == 0.01
    assert config.volatility_expansion_take_profit_pct == 0.03
    assert "strategy_mode" in changed

from datetime import datetime, timezone
from types import SimpleNamespace

import dashboard
from drift_monitor import BaselineProfile, DriftAlert, DriftReport, LiveMetrics
from dashboard_state import PersistedSignalDecision
from dashboard_state import DrilldownEvent


def test_compact_time_str_renders_safe_multiline_timestamp_html() -> None:
    formatted = dashboard._compact_time_str("2026-04-20T15:00:00+00:00")

    assert "<br><span" in formatted
    assert "11:00 AM ET" in formatted
    assert "9:00 AM local" in formatted


def test_format_datetime_pretty_escapes_unparseable_values() -> None:
    formatted = dashboard._format_datetime_pretty("<script>alert(1)</script>")

    assert "<script>" not in formatted
    assert "&lt;script&gt;" in formatted


def test_recent_activity_feed_keeps_timestamp_markup() -> None:
    item = SimpleNamespace(
        level="info",
        symbol="AMD",
        observed_at_utc="2026-04-20T15:00:00+00:00",
        title="Decision cycle recorded",
        body="No actionable signals.",
    )

    html = dashboard._recent_activity_feed_html([item])

    assert "<div class='activity-time'>11:00 AM ET<br><span" in html
    assert "&lt;span" not in html


def test_strategy_mode_uses_ml_only_for_ml_modes() -> None:
    assert dashboard._strategy_mode_uses_ml("ml") is True
    assert dashboard._strategy_mode_uses_ml("hybrid") is True
    assert dashboard._strategy_mode_uses_ml("mean_reversion") is False
    assert dashboard._strategy_mode_uses_ml("breakout") is False
    assert dashboard._strategy_mode_uses_ml(None) is False


def test_decision_logic_html_surfaces_saved_explanation_context() -> None:
    row = PersistedSignalDecision(
        observed_at_utc="2026-04-17T14:30:10+00:00",
        decision_timestamp="2026-04-17T14:30:00+00:00",
        symbol="AMD",
        action="BUY",
        price=99.5,
        sma=101.0,
        deviation_pct=-1.5,
        trend_sma=100.0,
        above_trend_sma=False,
        atr_pct=0.011,
        atr_percentile=55.0,
        volume_ratio=1.4,
        strategy_mode="mean_reversion",
        final_signal_reason="mean_reversion_sma_entry",
        decision_summary="BUY in mean reversion mode. reason: mean reversion sma entry.",
        entry_reference_price=99.0,
        exit_reference_price=101.0,
        stop_reference_price=97.0,
        vwap=100.2,
        adx=21.4,
        regime_state="bearish",
        rejection=None,
        trend_filter="pass",
        atr_filter="pass",
        window_open=True,
        holding=False,
    )

    html = dashboard._decision_logic_html(row)

    assert "Decision core" in html
    assert "mean reversion sma entry" in html
    assert "BUY in mean reversion mode" in html
    assert "Reference levels" in html


def test_drift_summary_html_renders_operator_alerts() -> None:
    report = DriftReport(
        baseline=BaselineProfile(
            source_label="live_config+research",
            valid_for_comparison=True,
            validation_errors=(),
            strategy_mode="mean_reversion",
            symbols=2,
            bars_per_day=26,
            buy_signal_rate_per_symbol_per_day=2.0,
            sell_signal_rate_per_symbol_per_day=2.0,
            rejection_rate=0.7,
            rejection_breakdown=(),
            hybrid_branch_participation=(),
            avg_bar_age_s=120.0,
            stale_bar_rate=0.0,
            trade_count_per_day=4.0,
            win_rate=0.6,
        ),
        live=LiveMetrics(
            window_label="current session",
            total_evaluated=30,
            buy_signals=2,
            sell_signals=1,
            hold_count=27,
            rejection_rate=0.9,
            rejection_breakdown=(("trend_filter", 8),),
            normalized_rejection_breakdown=(("trend_filter", 1.0),),
            hybrid_branch_participation=(),
            hybrid_branch_counts=(),
            avg_bar_age_s=310.0,
            stale_bar_count=3,
            stale_bar_rate=0.1,
            buy_signal_rate_per_symbol_per_day=0.8,
            sell_signal_rate_per_symbol_per_day=0.4,
            blocked_buy_signals=2,
            blocked_sell_signals=0,
            allowed_buy_checks=0,
            allowed_sell_checks=0,
            closed_trades=0,
            win_rate=None,
        ),
        alerts=(
            DriftAlert(
                key="buy_signal_rate_low",
                severity="warn",
                summary="Buy signal frequency is 60% below the validated profile.",
                observed="0.80 per symbol/day",
                expected="2.00 per symbol/day",
                why_it_matters="Fewer entries than expected usually means live filters or data quality are suppressing the tested edge.",
            ),
        ),
    )

    html = dashboard._drift_summary_html(report)

    assert "Behavior Drift" in html
    assert "60% below the validated profile" in html
    assert "Observed: 0.80 per symbol/day" in html


def test_last_cycle_summary_surfaces_evaluation_errors() -> None:
    report = SimpleNamespace(
        processed_bar=True,
        buy_signals=0,
        sell_signals=0,
        hold_signals=0,
        error_signals=15,
        orders_submitted=0,
        decision_timestamp="2026-04-20T19:15:00+00:00",
        skip_reason="no_actionable_signals",
    )

    value, note, tone = dashboard._last_cycle_summary(report, [], [])

    assert value == "Evaluation errors"
    assert note == "15 evaluated · 15 errors"
    assert tone == "err"


def test_next_cycle_text_reports_closed_after_flatten_deadline(monkeypatch) -> None:
    monkeypatch.setattr(
        dashboard,
        "_is_past_session_flatten_deadline",
        lambda now_utc=None: True,
    )

    value, note = dashboard._next_cycle_text(
        SimpleNamespace(
            startup_config=None,
            last_cycle_report=None,
            has_persisted_snapshot=False,
        )
    )

    assert value == "Closed"
    assert "3:55 PM ET flatten deadline" in note


def test_next_cycle_text_reports_missing_live_bot_heartbeat(monkeypatch) -> None:
    monkeypatch.setattr(
        dashboard,
        "_is_past_session_flatten_deadline",
        lambda now_utc=None: False,
    )
    monkeypatch.setattr(
        dashboard,
        "_live_bot_process_health",
        lambda: ("missing", "No active TradeOS live process was detected."),
    )
    monkeypatch.setattr(
        dashboard,
        "datetime",
        type(
            "FrozenDateTime",
            (),
            {
                "now": staticmethod(lambda tz=None: datetime(2026, 4, 21, 17, 0, tzinfo=timezone.utc)),
                "fromisoformat": staticmethod(datetime.fromisoformat),
            },
        ),
    )

    value, note = dashboard._next_cycle_text(
        SimpleNamespace(
            startup_config=SimpleNamespace(bar_timeframe_minutes=15),
            last_cycle_report=SimpleNamespace(decision_timestamp="2026-04-21T16:15:00+00:00"),
            has_persisted_snapshot=False,
            snapshot=SimpleNamespace(timestamp_utc=None),
        )
    )

    assert value == "No live bot heartbeat"
    assert "No active TradeOS live process was detected." in note


def test_resync_status_summary_reports_ok_state() -> None:
    value, note, tone = dashboard._resync_status_summary(
        SimpleNamespace(startup_config=SimpleNamespace(resync_status="RESYNC_OK"))
    )

    assert value == "Resync OK"
    assert "allowed to trade normally" in note
    assert tone == "good"


def test_status_overview_reports_degraded_resync(monkeypatch) -> None:
    monkeypatch.setattr(
        dashboard,
        "_is_past_session_flatten_deadline",
        lambda now_utc=None: False,
    )

    title, tone, copy = dashboard._status_overview(
        SimpleNamespace(
            startup_config=SimpleNamespace(resync_status="RESYNC_DEGRADED"),
            last_cycle_report=SimpleNamespace(processed_bar=True, skip_reason=""),
            has_persisted_snapshot=True,
            snapshot=SimpleNamespace(timestamp_utc="2026-04-22T14:45:00+00:00"),
            feed_status="persisted bar fresh (0s late)",
        )
    )

    assert title == "Degraded"
    assert tone == "warn"
    assert "exits-only" in copy


def test_build_operator_header_summary_includes_runtime_and_resync_cues(monkeypatch) -> None:
    monkeypatch.setattr(
        dashboard,
        "_process_health_summary",
        lambda: ("Process live", "Live bot process detected (pid=15592).", "good"),
    )
    monkeypatch.setattr(
        dashboard,
        "_resync_status_summary",
        lambda state: ("Resync OK", "Broker state was recovered.", "good"),
    )
    monkeypatch.setattr(
        dashboard,
        "_next_cycle_text",
        lambda state: ("11:00:00 AM ET", "About 12m until the next scheduled bar close."),
    )
    monkeypatch.setattr(
        dashboard,
        "_status_overview",
        lambda state: ("Running", "primary", "Recent session data is available."),
    )
    monkeypatch.setattr(
        dashboard,
        "_operator_data_freshness_status",
        lambda state, snapshot: (
            "2m ago",
            "Apr 22, 2026 · 10:45 AM ET",
            "good",
            "persisted bar fresh (0s late)",
            "2m ago",
        ),
    )

    summary = dashboard._build_operator_header_summary(
        SimpleNamespace(
            snapshot=SimpleNamespace(kill_switch_triggered=False),
            session_warnings=(),
            feed_status="persisted bar fresh (0s late)",
            startup_config=SimpleNamespace(
                session_id="live-123",
                strategy_mode="mean_reversion",
                execution_enabled=True,
                launch_mode="live",
                paper=True,
                runtime_config_approved=True,
                runtime_config_rejection_reasons=(),
            ),
            last_cycle_report=None,
            latest_signal_rows=(),
            latest_cycle_risk_checks=(),
        ),
        SimpleNamespace(
            timestamp_utc="2026-04-22T14:45:00+00:00",
            positions={},
            equity=100000.0,
            daily_pnl=10.0,
        ),
        [],
    )

    assert ("RESYNC OK", "good") in summary["chips"]
    assert ("PROCESS LIVE", "good") in summary["chips"]
    assert any(card[0] == "Runtime" and card[1] == "Process live" for card in summary["status_cards"])
    assert any(card[0] == "Resync" and card[1] == "Resync OK" for card in summary["status_cards"])


def test_status_overview_reports_closed_after_flatten_deadline(monkeypatch) -> None:
    monkeypatch.setattr(
        dashboard,
        "_is_past_session_flatten_deadline",
        lambda now_utc=None: True,
    )
    state = SimpleNamespace(
        snapshot=SimpleNamespace(timestamp_utc="2026-04-20T19:45:00+00:00"),
        has_persisted_snapshot=True,
    )

    title, tone, copy = dashboard._status_overview(state)

    assert title == "Closed"
    assert tone == "primary"
    assert "3:55 PM ET flatten deadline" in copy


def test_is_past_session_flatten_deadline_uses_eastern_time() -> None:
    before_deadline = datetime(2026, 4, 20, 19, 54, tzinfo=timezone.utc)
    after_deadline = datetime(2026, 4, 20, 19, 55, tzinfo=timezone.utc)

    assert dashboard._is_past_session_flatten_deadline(before_deadline) is False
    assert dashboard._is_past_session_flatten_deadline(after_deadline) is True


def test_pretty_reason_formats_compound_rejections() -> None:
    assert dashboard._pretty_reason("trend_filter|atr_filter") == (
        "filtered by trend filter + filtered by ATR filter"
    )


def test_drilldown_panel_keeps_timestamp_markup() -> None:
    event = DrilldownEvent(
        timestamp_utc="2026-04-21T16:15:00+00:00",
        symbol="XOM",
        event_type="signal.evaluated",
        action="HOLD",
        strategy_mode="mean_reversion",
        allowed=None,
        block_reason=None,
        rejection="trend_filter",
        trend_filter="reject",
        atr_filter="pass",
        above_trend_sma=False,
        deviation_pct=-1.15,
        atr_pct=0.0036,
        atr_percentile=16.0,
        ml_prob=None,
        pnl_usd=None,
        exit_reason=None,
        bar_close=None,
        sma=None,
        volume_ratio=1.08,
        window_open=True,
        holding=False,
    )

    html = dashboard._drilldown_panel_html(event)

    assert "<br><span" in html
    assert "&lt;span" not in html

from __future__ import annotations

import json
import shutil
import uuid
from pathlib import Path

from drift_monitor import build_drift_report, compute_live_metrics, evaluate_drift, load_baseline_profile


def test_load_baseline_profile_uses_optional_drift_overlay() -> None:
    base_dir = Path.cwd() / f"test_drift_profile_{uuid.uuid4().hex}"
    try:
        (base_dir / "config").mkdir(parents=True)
        (base_dir / "results").mkdir(parents=True)
        (base_dir / "config" / "live_config.json").write_text(
            json.dumps(
                {
                    "runtime": {
                        "symbols": ["AMD", "MSFT"],
                        "bar_timeframe_minutes": 15,
                        "strategy_mode": "hybrid_bb_mr",
                    }
                }
            ),
            encoding="utf-8",
        )
        (base_dir / "results" / "best_config_latest.json").write_text(
            json.dumps(
                {
                    "approved": True,
                    "config": {"strategy_mode": "hybrid_bb_mr"},
                    "performance": {"trades_per_day": 4.0, "win_rate": 60.0},
                }
            ),
            encoding="utf-8",
        )
        (base_dir / "results" / "drift_profile.json").write_text(
            json.dumps(
                {
                    "metrics": {
                        "rejection_breakdown": {"no_signal": 0.7, "trend_filter": 0.2, "atr_filter": 0.1},
                        "hybrid_branch_participation": {"mean_reversion": 0.65, "bollinger_breakout": 0.35},
                        "avg_bar_age_s": 120.0,
                        "stale_bar_rate": 0.01,
                    }
                }
            ),
            encoding="utf-8",
        )

        profile = load_baseline_profile(base_dir)

        assert profile.strategy_mode == "hybrid_bb_mr"
        assert dict(profile.rejection_breakdown)["no_signal"] == 0.7
        assert dict(profile.hybrid_branch_participation)["bollinger_breakout"] == 0.35
        assert profile.avg_bar_age_s == 120.0
        assert profile.stale_bar_rate == 0.01
    finally:
        shutil.rmtree(base_dir, ignore_errors=True)


def test_compute_live_metrics_aggregates_signal_branch_and_bar_quality() -> None:
    profile = load_baseline_profile()
    live = compute_live_metrics(
        signal_events=[
            {"event": "signal.evaluated", "action": "BUY", "symbol": "AMD", "hybrid_entry_branch": "mean_reversion"},
            {"event": "signal.evaluated", "action": "BUY", "symbol": "MSFT", "hybrid_entry_branch": "bollinger_breakout"},
            {"event": "signal.evaluated", "action": "HOLD", "symbol": "AMD", "rejection": "trend_filter|atr_filter"},
            {"event": "signal.evaluated", "action": "SELL", "symbol": "AMD"},
        ],
        bar_events=[
            {"event": "bar.received", "symbol": "AMD", "bar_age_s": 120.0, "stale": False},
            {"event": "bar.received", "symbol": "MSFT", "bar_age_s": 320.0, "stale": True},
        ],
        risk_events=[
            {"event": "risk.check", "action": "BUY", "allowed": False},
            {"event": "risk.check", "action": "BUY", "allowed": True},
        ],
        position_events=[
            {"event": "position.closed", "winner": True},
            {"event": "position.closed", "winner": False},
        ],
        baseline=profile,
    )

    assert live.total_evaluated == 4
    assert live.buy_signals == 2
    assert live.sell_signals == 1
    assert live.hold_count == 1
    assert dict(live.rejection_breakdown)["trend_filter"] == 1
    assert dict(live.rejection_breakdown)["atr_filter"] == 1
    assert dict(live.hybrid_branch_counts)["mean_reversion"] == 1
    assert dict(live.hybrid_branch_counts)["bollinger_breakout"] == 1
    assert live.stale_bar_count == 1
    assert live.avg_bar_age_s == 220.0
    assert live.blocked_buy_signals == 1
    assert live.allowed_buy_checks == 1
    assert live.win_rate == 0.5


def test_evaluate_drift_emits_alerts_for_clear_breaches() -> None:
    base_dir = Path.cwd() / f"test_drift_alerts_{uuid.uuid4().hex}"
    try:
        (base_dir / "config").mkdir(parents=True)
        (base_dir / "results").mkdir(parents=True)
        (base_dir / "config" / "live_config.json").write_text(
            json.dumps(
                {
                    "runtime": {
                        "symbols": ["AMD", "MSFT"],
                        "bar_timeframe_minutes": 15,
                        "strategy_mode": "hybrid_bb_mr",
                    }
                }
            ),
            encoding="utf-8",
        )
        (base_dir / "results" / "best_config_latest.json").write_text(
            json.dumps(
                {
                    "approved": True,
                    "config": {"strategy_mode": "hybrid_bb_mr"},
                    "performance": {"trades_per_day": 16.0, "win_rate": 65.0},
                }
            ),
            encoding="utf-8",
        )
        (base_dir / "results" / "drift_profile.json").write_text(
            json.dumps(
                {
                    "metrics": {
                        "rejection_breakdown": {"no_signal": 0.6, "trend_filter": 0.2, "atr_filter": 0.2},
                        "hybrid_branch_participation": {"mean_reversion": 0.5, "bollinger_breakout": 0.5},
                        "avg_bar_age_s": 120.0,
                        "stale_bar_rate": 0.0,
                    }
                }
            ),
            encoding="utf-8",
        )

        signal_events = [
            {"event": "signal.evaluated", "action": "BUY", "symbol": "AMD", "hybrid_entry_branch": "mean_reversion"},
            {"event": "signal.evaluated", "action": "BUY", "symbol": "MSFT", "hybrid_entry_branch": "mean_reversion"},
            {"event": "signal.evaluated", "action": "BUY", "symbol": "AMD", "hybrid_entry_branch": "mean_reversion"},
            {"event": "signal.evaluated", "action": "BUY", "symbol": "MSFT", "hybrid_entry_branch": "mean_reversion"},
        ] + [
            {
                "event": "signal.evaluated",
                "action": "HOLD",
                "symbol": "AMD" if idx % 2 == 0 else "MSFT",
                "rejection": "trend_filter",
            }
            for idx in range(22)
        ]

        report = build_drift_report(
            signal_events=signal_events,
            bar_events=[
                {"event": "bar.received", "symbol": "AMD", "bar_age_s": 310.0, "stale": True},
                {"event": "bar.received", "symbol": "MSFT", "bar_age_s": 290.0, "stale": True},
            ],
            risk_events=[],
            position_events=[],
            repo_root=base_dir,
        )

        keys = {alert.key for alert in report.alerts}

        assert "buy_signal_rate_low" in keys
        assert "rejection_rate_high" in keys
        assert "rejection_mix_trend_filter" in keys
        assert "hybrid_branch_bollinger_breakout_missing" in keys
        assert "avg_bar_age_high" in keys
        assert "stale_bar_rate_high" in keys
    finally:
        shutil.rmtree(base_dir, ignore_errors=True)


def test_evaluate_drift_returns_no_alerts_when_within_profile() -> None:
    base_dir = Path.cwd() / f"test_drift_no_alerts_{uuid.uuid4().hex}"
    try:
        (base_dir / "config").mkdir(parents=True)
        (base_dir / "results").mkdir(parents=True)
        (base_dir / "config" / "live_config.json").write_text(
            json.dumps(
                {
                    "runtime": {
                        "symbols": ["AMD", "MSFT"],
                        "bar_timeframe_minutes": 15,
                        "strategy_mode": "hybrid_bb_mr",
                    }
                }
            ),
            encoding="utf-8",
        )
        (base_dir / "results" / "best_config_latest.json").write_text(
            json.dumps(
                {
                    "approved": True,
                    "config": {"strategy_mode": "hybrid_bb_mr"},
                    "performance": {"trades_per_day": 26.0, "win_rate": 65.0},
                }
            ),
            encoding="utf-8",
        )
        (base_dir / "results" / "drift_profile.json").write_text(
            json.dumps(
                {
                    "metrics": {
                        "rejection_breakdown": {"no_signal": 0.75, "trend_filter": 0.15, "atr_filter": 0.1},
                        "hybrid_branch_participation": {"mean_reversion": 0.6, "bollinger_breakout": 0.4},
                        "avg_bar_age_s": 180.0,
                        "stale_bar_rate": 0.02,
                    }
                }
            ),
            encoding="utf-8",
        )

        report = build_drift_report(
            signal_events=[
                {"event": "signal.evaluated", "action": "BUY", "symbol": "AMD", "hybrid_entry_branch": "mean_reversion"},
                {"event": "signal.evaluated", "action": "BUY", "symbol": "MSFT", "hybrid_entry_branch": "bollinger_breakout"},
                {"event": "signal.evaluated", "action": "BUY", "symbol": "AMD", "hybrid_entry_branch": "mean_reversion"},
                {"event": "signal.evaluated", "action": "BUY", "symbol": "MSFT", "hybrid_entry_branch": "bollinger_breakout"},
                {"event": "signal.evaluated", "action": "HOLD", "symbol": "AMD", "rejection": "no_signal"},
                {"event": "signal.evaluated", "action": "HOLD", "symbol": "MSFT", "rejection": "no_signal"},
                {"event": "signal.evaluated", "action": "HOLD", "symbol": "AMD", "rejection": "trend_filter"},
                {"event": "signal.evaluated", "action": "HOLD", "symbol": "MSFT", "rejection": "atr_filter"},
            ],
            bar_events=[
                {"event": "bar.received", "symbol": "AMD", "bar_age_s": 170.0, "stale": False},
                {"event": "bar.received", "symbol": "MSFT", "bar_age_s": 175.0, "stale": False},
            ],
            risk_events=[],
            position_events=[],
            repo_root=base_dir,
        )

        assert report.alerts == ()
        assert report.within_expected_profile is True
    finally:
        shutil.rmtree(base_dir, ignore_errors=True)

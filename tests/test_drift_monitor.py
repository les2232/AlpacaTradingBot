from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import drift_monitor
import pytest
from daily_report import PROMOTED_RUNTIME_FIELDS
from drift_monitor import build_drift_report, compute_live_metrics, evaluate_drift, load_baseline_profile
from tradeos.research_bundle import generation_directory_component
from tradeos.strategy_contract import (
    CanonicalResolvedPromotedStrategy,
    RUNTIME_FIXED_STRATEGY_ASSUMPTIONS,
)
from trading_bot import BotConfig


_BASELINE_BOT_CONFIG = BotConfig(
    symbols=["AMD", "MSFT"],
    max_usd_per_trade=200.0,
    max_symbol_exposure_usd=200.0,
    max_open_positions=2,
    max_daily_loss_usd=300.0,
    sma_bars=20,
    bar_timeframe_minutes=15,
    strategy_mode="hybrid_bb_mr",
    entry_threshold_pct=0.0015,
    mean_reversion_max_atr_percentile=80.0,
    mean_reversion_stop_pct=0.01,
)
APPROVED_PROMOTED_CONFIG: dict[str, object] = {
    key: getattr(_BASELINE_BOT_CONFIG, key) for key in PROMOTED_RUNTIME_FIELDS
}
APPROVED_PROMOTED_CONFIG["symbol_strategy_modes"] = {
    symbol: _BASELINE_BOT_CONFIG.strategy_mode
    for symbol in _BASELINE_BOT_CONFIG.symbols
}
APPROVED_CANONICAL_CONTRACT = (
    CanonicalResolvedPromotedStrategy.from_research_resolution(
        _BASELINE_BOT_CONFIG.symbols,
        APPROVED_PROMOTED_CONFIG,
        dict(RUNTIME_FIXED_STRATEGY_ASSUMPTIONS),
    ).to_canonical_dict()
)


_GENERATION_SHA256 = "1" * 64
_BUNDLE_SHA256 = "2" * 64
_BUNDLE_PATH = "results/promotions/promotion_bundle.test.json"


def _install_verified_pointer_baseline(
    monkeypatch: pytest.MonkeyPatch,
    base_dir: Path,
    *,
    trades_per_day: float,
    win_rate: float,
    metrics_overrides: dict[str, object] | None = None,
    overlay_overrides: dict[str, object] | None = None,
) -> tuple[Path, dict[str, object], dict[str, object]]:
    baseline = {
        "symbols": 2,
        "bars_per_day": 26,
        "signal_rate_per_symbol_per_day": trades_per_day / 2,
        "win_rate": win_rate / 100.0 if win_rate > 1.0 else win_rate,
        "rejection_rate": float(drift_monitor.DEFAULT_BACKTEST["rejection_rate"]),
    }
    source: dict[str, object] = {
        "mode": "live_config+research",
        "valid_for_comparison": True,
        "validation_errors": [],
        "research_generation_sha256": _GENERATION_SHA256,
        "promotion_bundle_sha256": _BUNDLE_SHA256,
    }
    monkeypatch.setattr(
        drift_monitor,
        "load_backtest_baseline",
        lambda _root: (dict(baseline), dict(source)),
    )

    metrics: dict[str, object] = {
        "buy_signal_rate_per_symbol_per_day": trades_per_day / 2,
        "sell_signal_rate_per_symbol_per_day": trades_per_day / 2,
        "rejection_rate": float(drift_monitor.DEFAULT_BACKTEST["rejection_rate"]),
        "rejection_breakdown": {
            "no_signal": 0.7,
            "trend_filter": 0.2,
            "atr_filter": 0.1,
        },
        "hybrid_branch_participation": {
            "mean_reversion": 0.65,
            "bollinger_breakout": 0.35,
        },
        "avg_bar_age_s": 120.0,
        "stale_bar_rate": 0.01,
    }
    metrics.update(metrics_overrides or {})
    overlay: dict[str, object] = {
        "generated_at": "2026-08-20T12:00:00Z",
        "approved": True,
        "source_artifacts": {},
        "context": {
            "strategy_mode": "hybrid_bb_mr",
            "symbols": ["AMD", "MSFT"],
            "symbol_count": 2,
            "timeframe": "15Min",
            "bar_timeframe_minutes": 15,
            "stable": True,
            "market_regime": "sideways",
        },
        "performance": {
            "trades_per_day": trades_per_day,
            "win_rate": win_rate,
            "profit_factor": 1.4,
            "sharpe_ratio": 1.9,
            "realized_pnl": 300.0,
            "combined_score": 2.0,
        },
        "metrics": metrics,
        "metric_sources": {},
        "research_generation_sha256": _GENERATION_SHA256,
        "bundle_path": _BUNDLE_PATH,
        "bundle_sha256": _BUNDLE_SHA256,
    }
    overlay.update(overlay_overrides or {})
    overlay_path = (
        base_dir
        / "results"
        / "promotions"
        / "generations"
        / generation_directory_component(_GENERATION_SHA256)
        / "drift_profile.json"
    )
    overlay_path.parent.mkdir(parents=True)
    raw = json.dumps(overlay, allow_nan=True).encode("utf-8")
    overlay_path.write_bytes(raw)
    pointer_payload = {
        "research_generation_sha256": _GENERATION_SHA256,
        "bundle": {
            "path": _BUNDLE_PATH,
            "file_sha256": "3" * 64,
            "bundle_sha256": _BUNDLE_SHA256,
        },
        "artifacts": {
            "drift_profile": {
                "path": overlay_path.relative_to(base_dir).as_posix(),
                "sha256": hashlib.sha256(raw).hexdigest(),
            }
        },
    }
    monkeypatch.setattr(
        drift_monitor,
        "load_verified_promotion_pointer",
        lambda _path, *, repo_root: SimpleNamespace(payload=pointer_payload),
    )
    return overlay_path, pointer_payload, source


def _rewrite_bound_overlay(
    overlay_path: Path,
    pointer_payload: dict[str, object],
    payload: dict[str, object],
) -> None:
    raw = json.dumps(payload, allow_nan=True).encode("utf-8")
    overlay_path.write_bytes(raw)
    artifacts = pointer_payload["artifacts"]
    assert isinstance(artifacts, dict)
    reference = artifacts["drift_profile"]
    assert isinstance(reference, dict)
    reference["sha256"] = hashlib.sha256(raw).hexdigest()


def test_load_baseline_profile_uses_pointer_bound_drift_overlay(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _install_verified_pointer_baseline(
        monkeypatch,
        tmp_path,
        trades_per_day=4.0,
        win_rate=60.0,
    )

    profile = load_baseline_profile(tmp_path)

    assert profile.valid_for_comparison is True
    assert profile.source_label == "live_config+research+pointer_drift_profile"
    assert profile.strategy_mode == "hybrid_bb_mr"
    assert profile.trade_count_per_day == 4.0
    assert profile.win_rate == 0.6
    assert dict(profile.rejection_breakdown)["no_signal"] == 0.7
    assert dict(profile.hybrid_branch_participation)["bollinger_breakout"] == 0.35
    assert profile.avg_bar_age_s == 120.0
    assert profile.stale_bar_rate == 0.01


def test_mutable_drift_best_and_decision_aliases_are_never_consumed(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _install_verified_pointer_baseline(
        monkeypatch,
        tmp_path,
        trades_per_day=4.0,
        win_rate=60.0,
    )
    results = tmp_path / "results"
    for name, payload in {
        "drift_profile.json": {
            "approved": True,
            "context": {"strategy_mode": "malicious_alias"},
            "metrics": {"avg_bar_age_s": 0.0},
        },
        "best_config_latest.json": {
            "config": {"strategy_mode": "malicious_alias"},
            "performance": {"trades_per_day": 9999.0, "win_rate": 100.0},
        },
        "trade_decision.json": {
            "selected_config": {"strategy_mode": "malicious_alias"},
        },
    }.items():
        (results / name).write_text(json.dumps(payload), encoding="utf-8")

    profile = load_baseline_profile(tmp_path)

    assert profile.valid_for_comparison is True
    assert profile.strategy_mode == "hybrid_bb_mr"
    assert profile.trade_count_per_day == 4.0
    assert profile.win_rate == 0.6
    assert profile.avg_bar_age_s == 120.0


def test_pointer_bound_drift_mutation_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    overlay_path, _, _ = _install_verified_pointer_baseline(
        monkeypatch,
        tmp_path,
        trades_per_day=4.0,
        win_rate=60.0,
    )
    overlay_path.write_text('{"approved":true}', encoding="utf-8")

    profile = load_baseline_profile(tmp_path)

    assert profile.valid_for_comparison is False
    assert profile.strategy_mode is None
    assert profile.hybrid_branch_participation == ()
    assert any("bytes do not match" in error for error in profile.validation_errors)


@pytest.mark.parametrize(
    ("overlay_field", "overlay_value", "metric_field", "metric_value"),
    [
        ("approved", False, None, None),
        ("research_generation_sha256", "4" * 64, None, None),
        (None, None, "avg_bar_age_s", True),
        (None, None, "stale_bar_rate", float("nan")),
        (None, None, "hybrid_branch_participation", None),
    ],
)
def test_invalid_pointer_bound_drift_context_or_numbers_fail_closed(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    overlay_field: str | None,
    overlay_value: object,
    metric_field: str | None,
    metric_value: object,
) -> None:
    overlay_path, pointer_payload, _ = _install_verified_pointer_baseline(
        monkeypatch,
        tmp_path,
        trades_per_day=4.0,
        win_rate=60.0,
    )
    overlay = json.loads(overlay_path.read_text(encoding="utf-8"))
    if overlay_field is not None:
        overlay[overlay_field] = overlay_value
    if metric_field is not None:
        metrics = overlay["metrics"]
        assert isinstance(metrics, dict)
        metrics[metric_field] = metric_value
    _rewrite_bound_overlay(overlay_path, pointer_payload, overlay)

    profile = load_baseline_profile(tmp_path)

    assert profile.valid_for_comparison is False
    assert profile.source_label == "live_config+research"
    assert profile.strategy_mode is None
    assert profile.hybrid_branch_participation == ()
    assert any("pointer-bound drift profile" in error for error in profile.validation_errors)


def test_truthy_nonboolean_daily_validity_never_enables_drift(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(
        drift_monitor,
        "load_backtest_baseline",
        lambda _root: (
            dict(drift_monitor.DEFAULT_BACKTEST),
            {
                "mode": "untrusted",
                "valid_for_comparison": "true",
                "validation_errors": [],
            },
        ),
    )
    monkeypatch.setattr(
        drift_monitor,
        "load_verified_promotion_pointer",
        lambda *args, **kwargs: pytest.fail("invalid daily baseline reached pointer loader"),
    )

    profile = load_baseline_profile(tmp_path)

    assert profile.valid_for_comparison is False
    assert profile.strategy_mode is None


def test_compute_live_metrics_aggregates_signal_branch_and_bar_quality(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _install_verified_pointer_baseline(
        monkeypatch,
        tmp_path,
        trades_per_day=4.0,
        win_rate=60.0,
    )
    profile = load_baseline_profile(tmp_path)
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


def test_evaluate_drift_emits_alerts_for_clear_breaches(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _install_verified_pointer_baseline(
        monkeypatch,
        tmp_path,
        trades_per_day=16.0,
        win_rate=65.0,
        metrics_overrides={
            "rejection_breakdown": {
                "no_signal": 0.6,
                "trend_filter": 0.2,
                "atr_filter": 0.2,
            },
            "hybrid_branch_participation": {
                "mean_reversion": 0.5,
                "bollinger_breakout": 0.5,
            },
            "avg_bar_age_s": 120.0,
            "stale_bar_rate": 0.0,
        },
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
            repo_root=tmp_path,
        )

    keys = {alert.key for alert in report.alerts}

    assert "buy_signal_rate_low" in keys
    assert "rejection_rate_high" in keys
    assert "rejection_mix_trend_filter" in keys
    assert "hybrid_branch_bollinger_breakout_missing" in keys
    assert "avg_bar_age_high" in keys
    assert "stale_bar_rate_high" in keys


def test_evaluate_drift_returns_no_alerts_when_within_profile(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _install_verified_pointer_baseline(
        monkeypatch,
        tmp_path,
        trades_per_day=26.0,
        win_rate=65.0,
        metrics_overrides={
            "rejection_breakdown": {
                "no_signal": 0.75,
                "trend_filter": 0.15,
                "atr_filter": 0.1,
            },
            "hybrid_branch_participation": {
                "mean_reversion": 0.6,
                "bollinger_breakout": 0.4,
            },
            "avg_bar_age_s": 180.0,
            "stale_bar_rate": 0.02,
        },
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
            repo_root=tmp_path,
        )

    assert report.alerts == ()
    assert report.within_expected_profile is True

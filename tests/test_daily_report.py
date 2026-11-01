import hashlib
import json
import math
from dataclasses import fields
from pathlib import Path
import shutil
from types import SimpleNamespace
import uuid

import daily_report
import pytest
from strategy import StrategyConfig
from tradeos.strategy_contract import (
    CanonicalResolvedPromotedStrategy,
    RUNTIME_FIXED_STRATEGY_ASSUMPTIONS,
)
from trading_bot import BotConfig


_TEST_RESEARCH_GENERATION_SHA256 = "1" * 64
_TEST_PROMOTION_BUNDLE_SHA256 = "2" * 64
_TEST_RESEARCH_SOURCE_CONTENT_SHA256 = "3" * 64

_TEST_DATA_PARITY: dict[str, object] = {
    "historical_feed": "iex",
    "live_feed": "iex",
    "latest_bar_feed": "iex",
    "bar_build_mode": "stream_minute_aggregate",
    "apply_updated_bars": True,
    "post_bar_reconcile_poll": True,
    "block_trading_until_resync": True,
    "assert_feed_on_startup": True,
    "log_bar_components": True,
}


def _complete_runtime(**overrides: object) -> dict[str, object]:
    config = BotConfig(
        symbols=["AMD", "MSFT", "TSLA"],
        max_usd_per_trade=200.0,
        max_symbol_exposure_usd=200.0,
        max_open_positions=3,
        max_daily_loss_usd=300.0,
        sma_bars=20,
        bar_timeframe_minutes=15,
        strategy_mode="mean_reversion",
    )
    runtime: dict[str, object] = {
        key: getattr(config, key) for key in daily_report.PROMOTED_RUNTIME_FIELDS
    }
    runtime.update(
        {
            "symbols": ["AMD", "MSFT", "TSLA"],
            "bar_timeframe_minutes": 15,
            "data_parity": dict(_TEST_DATA_PARITY),
            **{
                key: getattr(config, key)
                for key in daily_report.RISK_EXECUTION_CONTEXT_FIELDS
            },
        }
    )
    runtime.update(overrides)
    if "symbol_strategy_modes" not in overrides:
        runtime["symbol_strategy_modes"] = {
            symbol: runtime["strategy_mode"] for symbol in runtime["symbols"]
        }
    return runtime


def _research_config(runtime: dict[str, object]) -> dict[str, object]:
    return {
        "symbols": runtime["symbols"],
        **{key: runtime[key] for key in daily_report.PROMOTED_RUNTIME_FIELDS},
    }


def _canonical_research_contract(
    config: dict[str, object],
) -> dict[str, object]:
    return CanonicalResolvedPromotedStrategy.from_research_resolution(
        config["symbols"],
        {
            key: config[key]
            for key in daily_report.PROMOTED_RUNTIME_FIELDS
        },
        dict(RUNTIME_FIXED_STRATEGY_ASSUMPTIONS),
    ).to_canonical_dict()


def _mock_committed_promotion_pointer(
    monkeypatch: object,
    base_dir: Path,
) -> dict[str, object]:
    live_path = base_dir / "config" / "live_config.json"
    research_path = base_dir / "results" / "best_config_latest.json"
    research_payload = json.loads(research_path.read_text(encoding="utf-8"))
    contract = CanonicalResolvedPromotedStrategy.from_payload(
        research_payload["canonical_strategy_contract"]
    )
    bundle_path = base_dir / "results" / "promotions" / "test_bundle.json"
    bundle_path.parent.mkdir(parents=True, exist_ok=True)
    bundle_path.write_text(
        json.dumps(
            {
                "bundle_sha256": _TEST_PROMOTION_BUNDLE_SHA256,
                "source_identity": {
                    "content_sha256": _TEST_RESEARCH_SOURCE_CONTENT_SHA256,
                },
                "strategy": {
                    "canonical_json": contract.canonical_json(),
                    "canonical_sha256": contract.sha256(),
                },
            }
        ),
        encoding="utf-8",
    )
    payload: dict[str, object] = {
        "research_generation_sha256": _TEST_RESEARCH_GENERATION_SHA256,
        "bundle": {
            "path": "results/promotions/test_bundle.json",
            "file_sha256": hashlib.sha256(bundle_path.read_bytes()).hexdigest(),
            "bundle_sha256": _TEST_PROMOTION_BUNDLE_SHA256,
        },
        "artifacts": {
            "live_config": {
                "path": "config/live_config.json",
                "sha256": hashlib.sha256(live_path.read_bytes()).hexdigest(),
            },
            "best_config_latest": {
                "path": "results/best_config_latest.json",
                "sha256": hashlib.sha256(research_path.read_bytes()).hexdigest(),
            },
        },
    }
    monkeypatch.setattr(  # type: ignore[attr-defined]
        daily_report,
        "load_verified_promotion_pointer",
        lambda _pointer_path, *, repo_root: SimpleNamespace(payload=payload),
    )
    return payload


def _write_approved_baseline_artifacts(base_dir: Path) -> None:
    (base_dir / "config").mkdir(parents=True)
    (base_dir / "results").mkdir(parents=True)
    runtime = _complete_runtime(
        symbol_strategy_modes={
            "AMD": "mean_reversion",
            "MSFT": "mean_reversion",
            "TSLA": "mean_reversion",
        }
    )
    (base_dir / "config" / "live_config.json").write_text(
        json.dumps(
            {
                "source": {
                    "approved": True,
                    "rejection_reasons": [],
                    "research_generation_sha256": _TEST_RESEARCH_GENERATION_SHA256,
                },
                "runtime": runtime,
            }
        ),
        encoding="utf-8",
    )
    research_config = _research_config(runtime)
    (base_dir / "results" / "best_config_latest.json").write_text(
        json.dumps(
            {
                "approved": True,
                "rejection_reasons": [],
                "config": research_config,
                "canonical_strategy_contract": _canonical_research_contract(
                    research_config
                ),
                "performance": {"trades_per_day": 6.0, "win_rate": 75.0},
            }
        ),
        encoding="utf-8",
    )


def test_promoted_runtime_fields_cover_the_live_execution_strategy_surface() -> None:
    bot_fields = {field.name for field in fields(BotConfig)}
    strategy_fields = {field.name for field in fields(StrategyConfig)}
    expected = (strategy_fields & bot_fields) | {
        "sma_bars",
        "symbol_strategy_modes",
        "bar_timeframe_minutes",
    }

    assert set(daily_report.PROMOTED_RUNTIME_FIELDS) == expected
    assert {
        "breakout_max_stop_pct",
        "sma_stop_pct",
        "vwap_z_entry_threshold",
        "vwap_z_stop_atr_multiple",
        "trend_pullback_entry_threshold",
        "momentum_breakout_entry_buffer_pct",
        "volatility_expansion_entry_buffer_pct",
        "bb_period",
        "bb_exit_mode",
        "symbol_strategy_modes",
    } <= expected
    assert {
        "max_usd_per_trade",
        "max_daily_loss_usd",
        "historical_feed",
        "live_feed",
        "bar_build_mode",
        "max_orders_per_minute",
        "broker_backend",
        "ml_lookback_bars",
    }.isdisjoint(expected)


def test_load_logs_skips_malformed_lines_and_deduplicates() -> None:
    base_dir = Path.cwd() / f"test_daily_report_{uuid.uuid4().hex}"
    try:
        log_dir = base_dir / "logs" / "2026-04-16"
        log_dir.mkdir(parents=True)

        (log_dir / "bars.jsonl").write_text(
            "\n".join(
                [
                    json.dumps({"event": "bar.received", "trace": "A", "symbol": "AMD", "bar_age_s": 100.0, "stale": False}),
                    json.dumps({"event": "bar.received", "trace": "A", "symbol": "AMD", "bar_age_s": 100.0, "stale": False}),
                    "{bad json",
                ]
            ),
            encoding="utf-8",
        )
        (log_dir / "signals.jsonl").write_text(
            "\n".join(
                [
                    json.dumps({"event": "signal.evaluated", "trace": "A", "symbol": "AMD", "action": "BUY"}),
                    json.dumps({"event": "signal.evaluated", "trace": "A", "symbol": "AMD", "action": "BUY"}),
                    json.dumps({"event": "signal.evaluated", "trace": "B", "symbol": "MSFT", "action": "HOLD", "rejection": "no_signal"}),
                ]
            ),
            encoding="utf-8",
        )
        (log_dir / "execution.jsonl").write_text(
            "\n".join(
                [
                    json.dumps({"event": "order.partial_fill", "trace": "A", "order_id": "order-1", "filled_qty": 1.0}),
                    json.dumps({"event": "order.partial_fill", "trace": "A", "order_id": "order-1", "filled_qty": 2.0}),
                    json.dumps({"event": "order.partial_fill", "trace": "A", "order_id": "order-1", "filled_qty": 2.0}),
                ]
            ),
            encoding="utf-8",
        )
        for name in ("risk", "positions"):
            (log_dir / f"{name}.jsonl").write_text("", encoding="utf-8")

        logs = daily_report.load_logs(log_dir)
        log_health = daily_report.compute_log_health(logs)
        signals = daily_report.compute_signal_behavior(logs["signals"])

        assert logs["bars"].attrs["malformed_line_count"] == 1
        assert logs["bars"].attrs["duplicate_row_count"] == 1
        assert log_health["malformed_lines"] == 1
        assert log_health["duplicate_rows"] == 3
        assert signals["total_evaluated"] == 2
        assert signals["buy_signals"] == 1
        assert logs["execution"]["filled_qty"].tolist() == [1.0, 2.0]
    finally:
        shutil.rmtree(base_dir, ignore_errors=True)


_VALID_BASELINE: dict[str, object] = {
    "symbols": 15,
    "bars_per_day": 26,
    "signal_rate_per_symbol_per_day": 2.1,
    "win_rate": 0.65,
    "avg_hold_bars": 4.2,
    "rejection_rate": 0.72,
    "avg_slippage_bps": 5.0,
}
_INVALID_BASELINE_REASONS = [
    "promotion pointer results/promotion_current.json is missing",
    "research config is not approved",
]


@pytest.fixture
def valid_baseline(monkeypatch: pytest.MonkeyPatch) -> dict[str, object]:
    baseline = dict(_VALID_BASELINE)
    monkeypatch.setattr(daily_report, "BACKTEST", baseline)
    monkeypatch.setattr(
        daily_report,
        "BACKTEST_SOURCE",
        {"mode": "live_config+research", "validation_errors": [], "valid_for_comparison": True},
    )
    return baseline


@pytest.fixture
def invalid_baseline(monkeypatch: pytest.MonkeyPatch) -> dict[str, object]:
    # BACKTEST still holds the DEFAULT_BACKTEST fallback numbers, exactly as
    # load_backtest_baseline() returns them when validation fails.
    baseline = dict(daily_report.DEFAULT_BACKTEST)
    monkeypatch.setattr(daily_report, "BACKTEST", baseline)
    monkeypatch.setattr(
        daily_report,
        "BACKTEST_SOURCE",
        {
            "mode": "defaults",
            "validation_errors": list(_INVALID_BASELINE_REASONS),
            "valid_for_comparison": False,
        },
    )
    return baseline


def _partial_session_sig() -> dict:
    return {
        "total_evaluated": 255,
        "buy_signals": 7,
        "sell_signals": 2,
        "hold_count": 246,
        "by_symbol": {"AMD": 4, "MSFT": 3},
        "rejections": {"no_signal": 246},
        "rejection_rate": 246 / 255,
    }


def _quiet_inputs() -> tuple[dict, dict, dict, dict, dict]:
    return (
        {"evaluated": 255, "avg_age_s": None, "stale_count": 0, "symbols": []},
        _partial_session_sig(),
        {"submitted_buys": 0, "submitted_sells": 0, "avg_slippage_bps": None, "worst_slippage_bps": None, "partial_fills": 0, "avg_latency_ms": None},
        {"opened": 0, "closed": 0, "win_rate": None, "avg_hold_bars": None, "avg_pnl_usd": None, "exit_reasons": {}, "total_pnl_usd": None},
        {"malformed_lines": 0, "duplicate_rows": 0},
    )


def _noisy_inputs() -> tuple[dict, dict, dict, dict, dict]:
    """Inputs that breach every baseline-independent (absolute) threshold."""
    return (
        {"evaluated": 255, "avg_age_s": 300.0, "stale_count": 3, "symbols": []},
        _partial_session_sig(),
        {"submitted_buys": 6, "submitted_sells": 6, "avg_slippage_bps": 20.0, "worst_slippage_bps": 55.0, "partial_fills": 1, "avg_latency_ms": 120.0},
        {"opened": 6, "closed": 6, "win_rate": 1 / 3, "avg_hold_bars": 3.0, "avg_pnl_usd": -1.5, "exit_reasons": {"eod_flatten": 4, "sma_exit": 2}, "total_pnl_usd": -9.0},
        {"malformed_lines": 2, "duplicate_rows": 1},
    )


def test_detect_concerns_normalizes_partial_session_signal_rate(valid_baseline: dict) -> None:
    concerns = daily_report.detect_concerns(*_quiet_inputs())

    assert any("PARTIAL SESSION" in concern for concern in concerns)
    assert any("SIGNAL RATE LOW" in concern for concern in concerns)
    assert not any("BASELINE INVALID" in concern for concern in concerns)


def test_detect_concerns_flags_high_signal_rate_against_valid_baseline(valid_baseline: dict) -> None:
    bar_q, sig, ex, pos, log_health = _quiet_inputs()
    sig = {**sig, "total_evaluated": 390, "buy_signals": 150}  # full session, 10/sym/day

    concerns = daily_report.detect_concerns(bar_q, sig, ex, pos, log_health)

    assert any("SIGNAL RATE HIGH" in concern for concern in concerns)
    assert not any("PARTIAL SESSION" in concern for concern in concerns)


def test_baseline_comparison_status_reflects_baseline_validity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(daily_report, "BACKTEST_SOURCE", {"valid_for_comparison": True, "validation_errors": []})
    assert daily_report.baseline_comparison_status() == {"available": True, "status": "available", "reasons": []}

    for source in (
        {"valid_for_comparison": False, "validation_errors": []},
        {"valid_for_comparison": "true", "validation_errors": None},  # truthy but not True
        {},
    ):
        monkeypatch.setattr(daily_report, "BACKTEST_SOURCE", source)
        assert daily_report.baseline_comparison_status() == {
            "available": False,
            "status": "unavailable",
            "reasons": ["baseline validation failed"],
        }


def test_detect_concerns_suppresses_backtest_relative_checks_when_baseline_invalid(
    invalid_baseline: dict,
) -> None:
    concerns = daily_report.detect_concerns(*_quiet_inputs())

    assert len(concerns) == 1
    assert concerns[0].startswith("BASELINE INVALID: ")
    assert "comparison unavailable" in concerns[0]
    for reason in _INVALID_BASELINE_REASONS:
        assert reason in concerns[0]
    assert not any("SIGNAL RATE" in concern for concern in concerns)
    assert not any("PARTIAL SESSION" in concern for concern in concerns)
    assert not any("vs backtest" in concern for concern in concerns)


def test_detect_concerns_keeps_absolute_threshold_checks_when_baseline_invalid(
    invalid_baseline: dict,
) -> None:
    concerns = daily_report.detect_concerns(*_noisy_inputs())
    prefixes = [concern.split(":", 1)[0] for concern in concerns]

    assert prefixes == [
        "BASELINE INVALID",
        "SLIPPAGE HIGH",
        "SLIPPAGE SPIKE",
        "WIN RATE LOW",
        "EOD EXITS HIGH",
        "STALE BARS",
        "BAR LATENCY",
        "PARTIAL FILLS",
        "LOG PARSE WARNINGS",
        "RESTART DUPLICATES",
    ]


def _report_args(inputs: tuple[dict, dict, dict, dict, dict]) -> tuple:
    bar_q, sig, ex, pos, log_health = inputs
    risk = {"kill_switch_triggered": False, "buy_blocks_by_reason": {}, "sell_blocks_by_reason": {}}
    concerns = daily_report.detect_concerns(bar_q, sig, ex, pos, log_health)
    return ("2026-04-16", bar_q, sig, ex, risk, pos, concerns, log_health)


def test_build_json_output_nulls_expectations_when_baseline_invalid(invalid_baseline: dict) -> None:
    output = daily_report.build_json_output(*_report_args(_noisy_inputs()))

    assert output["comparison"] == {
        "available": False,
        "status": "unavailable",
        "reasons": _INVALID_BASELINE_REASONS,
    }
    assert set(output["backtest"]) == set(daily_report.DEFAULT_BACKTEST)
    assert all(value is None for value in output["backtest"].values())
    assert not any("SIGNAL RATE" in concern for concern in output["concerns"])
    # Observed metrics are still reported.
    assert output["execution"]["avg_slippage_bps"] == 20.0
    assert output["signals"]["buy_signals"] == 7
    # The report must stay JSON-serialisable for --json.
    payload = json.loads(json.dumps(output, default=str))
    assert payload["backtest"]["signal_rate_per_symbol_per_day"] is None
    assert payload["comparison"]["available"] is False


def test_build_json_output_reports_expectations_when_baseline_valid(valid_baseline: dict) -> None:
    output = daily_report.build_json_output(*_report_args(_quiet_inputs()))

    assert output["comparison"] == {"available": True, "status": "available", "reasons": []}
    assert output["backtest"] == valid_baseline
    assert any("SIGNAL RATE LOW" in concern for concern in output["concerns"])


def test_print_report_omits_backtest_expectations_when_baseline_invalid(
    invalid_baseline: dict, capsys: pytest.CaptureFixture[str]
) -> None:
    daily_report.print_report(*_report_args(_noisy_inputs()))
    out = capsys.readouterr().out

    assert "Comparison unavailable" in out
    for reason in _INVALID_BASELINE_REASONS:
        assert reason in out
    for fallback_marker in ("expect ~", "(backtest", "bt=", "/sym/day", "Session coverage", "Delta"):
        assert fallback_marker not in out
    # Absolute-threshold flags and observed values survive.
    assert "Avg slippage (bps)" in out and "20.0" in out
    assert "SLIPPAGE HIGH" in out


def test_print_report_shows_backtest_comparison_when_baseline_valid(
    valid_baseline: dict, capsys: pytest.CaptureFixture[str]
) -> None:
    daily_report.print_report(*_report_args(_quiet_inputs()))
    out = capsys.readouterr().out

    assert "(expect ~390 = 15 syms × 26 bars)" in out
    assert "bt=2.1" in out
    assert "Session coverage" in out
    assert "Backtest" in out and "BUY signals/sym/day" in out
    assert "Comparison unavailable" not in out


def test_load_backtest_baseline_uses_pointer_bound_live_and_research_artifacts(
    monkeypatch,
) -> None:
    base_dir = Path.cwd() / f"test_daily_report_baseline_{uuid.uuid4().hex}"
    try:
        (base_dir / "config").mkdir(parents=True)
        (base_dir / "results").mkdir(parents=True)
        runtime = _complete_runtime(
            symbol_strategy_modes={
                "AMD": "mean_reversion",
                "MSFT": "mean_reversion",
                "TSLA": "mean_reversion",
            }
        )
        (base_dir / "config" / "live_config.json").write_text(
            json.dumps(
                {
                        "source": {
                            "approved": True,
                            "rejection_reasons": [],
                            "research_generation_sha256": (
                                _TEST_RESEARCH_GENERATION_SHA256
                            ),
                        },
                    "runtime": runtime,
                }
            ),
            encoding="utf-8",
        )
        (base_dir / "results" / "best_config_latest.json").write_text(
            json.dumps(
                {
                    "approved": True,
                    "rejection_reasons": [],
                    "config": _research_config(runtime),
                    "canonical_strategy_contract": (
                        _canonical_research_contract(_research_config(runtime))
                    ),
                    "performance": {
                        "trades_per_day": 6.0,
                        "win_rate": 75.0,
                    }
                }
            ),
            encoding="utf-8",
        )
        pointer = _mock_committed_promotion_pointer(monkeypatch, base_dir)

        baseline, source = daily_report.load_backtest_baseline(base_dir)

        assert baseline["symbols"] == 3
        assert baseline["bars_per_day"] == 26
        assert baseline["signal_rate_per_symbol_per_day"] == 2.0
        assert baseline["win_rate"] == 0.75
        assert source["mode"] == "live_config+research"
        assert source["promotion_pointer_valid"] is True
        assert source["research_generation_sha256"] == pointer[
            "research_generation_sha256"
        ]
        assert source["research_source_content_sha256"] == (
            _TEST_RESEARCH_SOURCE_CONTENT_SHA256
        )
        assert source["execution_context_matches_pointer_live"] is True
        assert source["execution_context_validation_errors"] == []
        assert source["research_metrics_used"] is True
        assert source["valid_for_comparison"] is True
    finally:
        shutil.rmtree(base_dir, ignore_errors=True)


@pytest.mark.parametrize(
    "source_content_sha256",
    [None, "3" * 63, "G" * 64, "z" * 64, 7],
)
def test_load_backtest_baseline_rejects_missing_or_malformed_source_attribution(
    monkeypatch,
    tmp_path: Path,
    source_content_sha256: object,
) -> None:
    _write_approved_baseline_artifacts(tmp_path)
    pointer = _mock_committed_promotion_pointer(monkeypatch, tmp_path)
    bundle_reference = pointer["bundle"]
    assert isinstance(bundle_reference, dict)
    bundle_path = tmp_path / str(bundle_reference["path"])
    bundle_payload = json.loads(bundle_path.read_text(encoding="utf-8"))
    source_identity = bundle_payload["source_identity"]
    assert isinstance(source_identity, dict)
    if source_content_sha256 is None:
        source_identity.pop("content_sha256")
    else:
        source_identity["content_sha256"] = source_content_sha256
    bundle_path.write_text(json.dumps(bundle_payload), encoding="utf-8")
    bundle_reference["file_sha256"] = hashlib.sha256(
        bundle_path.read_bytes()
    ).hexdigest()

    _, source = daily_report.load_backtest_baseline(tmp_path)

    assert source["promotion_pointer_valid"] is False
    assert source["valid_for_comparison"] is False
    assert source["research_source_content_sha256"] is None
    assert any(
        "research source identity is missing or invalid" in error
        for error in source["validation_errors"]
    )


def test_load_backtest_baseline_keeps_legacy_matching_aliases_red_without_pointer() -> None:
    base_dir = Path.cwd() / f"test_daily_report_no_pointer_{uuid.uuid4().hex}"
    try:
        _write_approved_baseline_artifacts(base_dir)

        baseline, source = daily_report.load_backtest_baseline(base_dir)

        assert baseline["symbols"] == 3
        assert baseline["signal_rate_per_symbol_per_day"] == daily_report.DEFAULT_BACKTEST[
            "signal_rate_per_symbol_per_day"
        ]
        assert source["promotion_pointer_valid"] is False
        assert source["research_metrics_used"] is False
        assert source["valid_for_comparison"] is False
        assert any(
            "promotion pointer is missing or invalid" in error
            for error in source["validation_errors"]
        )
    finally:
        shutil.rmtree(base_dir, ignore_errors=True)


def test_load_backtest_baseline_rejects_artifact_changed_after_pointer_validation(
    monkeypatch,
) -> None:
    base_dir = Path.cwd() / f"test_daily_report_mixed_generation_{uuid.uuid4().hex}"
    try:
        _write_approved_baseline_artifacts(base_dir)
        _mock_committed_promotion_pointer(monkeypatch, base_dir)
        research_path = base_dir / "results" / "best_config_latest.json"
        research_path.write_bytes(research_path.read_bytes() + b"\n")

        baseline, source = daily_report.load_backtest_baseline(base_dir)

        assert baseline["symbols"] == 3
        assert baseline["signal_rate_per_symbol_per_day"] == daily_report.DEFAULT_BACKTEST[
            "signal_rate_per_symbol_per_day"
        ]
        assert source["promotion_pointer_valid"] is False
        assert source["research_metrics_used"] is False
        assert source["valid_for_comparison"] is False
        assert any(
            "promotion artifact bytes do not match the commit pointer" in error
            for error in source["validation_errors"]
        )
    finally:
        shutil.rmtree(base_dir, ignore_errors=True)


def test_load_backtest_baseline_allows_explicit_runtime_matching_bundle_and_generation(
    monkeypatch,
) -> None:
    base_dir = Path.cwd() / f"test_daily_report_explicit_match_{uuid.uuid4().hex}"
    try:
        _write_approved_baseline_artifacts(base_dir)
        _mock_committed_promotion_pointer(monkeypatch, base_dir)
        explicit_path = base_dir / "config" / "paper_shadow_runtime.json"
        explicit_path.write_bytes(
            (base_dir / "config" / "live_config.json").read_bytes()
        )
        effective_config = BotConfig(
            symbols=["AMD", "MSFT", "TSLA"],
            max_usd_per_trade=200.0,
            max_symbol_exposure_usd=200.0,
            max_open_positions=3,
            max_daily_loss_usd=300.0,
            sma_bars=20,
            bar_timeframe_minutes=15,
            strategy_mode="mean_reversion",
        )

        baseline, source = daily_report.load_backtest_baseline(
            base_dir,
            explicit_path,
            expected_runtime_config_sha256=hashlib.sha256(
                explicit_path.read_bytes()
            ).hexdigest(),
            effective_runtime_config=effective_config,
            effective_canonical_strategy_sha256=(
                CanonicalResolvedPromotedStrategy.from_bot_config(
                    effective_config
                ).sha256()
            ),
        )

        assert baseline["signal_rate_per_symbol_per_day"] == 2.0
        assert baseline["win_rate"] == 0.75
        assert source["live_config_path"] == str(explicit_path.resolve())
        assert source["promotion_pointer_valid"] is True
        assert source["execution_context_matches_pointer_live"] is True
        assert source["execution_context_validation_errors"] == []
        assert source["effective_runtime_binding_valid"] is True
        assert source["effective_runtime_binding_errors"] == []
        assert source["research_metrics_used"] is True
        assert source["valid_for_comparison"] is True
    finally:
        shutil.rmtree(base_dir, ignore_errors=True)


def test_effective_runtime_binding_rejects_semantic_match_after_file_byte_swap(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _write_approved_baseline_artifacts(tmp_path)
    _mock_committed_promotion_pointer(monkeypatch, tmp_path)
    pointer_live_path = tmp_path / "config" / "live_config.json"
    explicit_path = tmp_path / "config" / "paper_shadow_config.json"
    pointer_bytes = pointer_live_path.read_bytes()

    # Simulate the first config read: the extra newline is valid JSON and has
    # identical semantics, but its exact byte identity is distinct.
    first_read_bytes = pointer_bytes + b"\n"
    explicit_path.write_bytes(first_read_bytes)
    first_read_sha256 = hashlib.sha256(first_read_bytes).hexdigest()
    effective_config = BotConfig(
        symbols=["AMD", "MSFT", "TSLA"],
        max_usd_per_trade=200.0,
        max_symbol_exposure_usd=200.0,
        max_open_positions=3,
        max_daily_loss_usd=300.0,
        sma_bars=20,
        bar_timeframe_minutes=15,
        strategy_mode="mean_reversion",
    )
    canonical_sha256 = CanonicalResolvedPromotedStrategy.from_bot_config(
        effective_config
    ).sha256()

    # Swap the path to different bytes with the same validated payload before
    # daily baseline validation performs its one and only selected-file read.
    explicit_path.write_bytes(pointer_bytes)
    _, source = daily_report.load_backtest_baseline(
        tmp_path,
        explicit_path,
        expected_runtime_config_sha256=first_read_sha256,
        effective_runtime_config=effective_config,
        effective_canonical_strategy_sha256=canonical_sha256,
    )

    assert source["selected_runtime_config_sha256"] == hashlib.sha256(
        pointer_bytes
    ).hexdigest()
    assert source["effective_runtime_binding_valid"] is False
    assert any(
        "runtime config bytes changed after the effective BotConfig was resolved"
        in error
        for error in source["effective_runtime_binding_errors"]
    )
    assert source["research_metrics_used"] is False
    assert source["valid_for_comparison"] is False


@pytest.mark.parametrize(
    "mutation",
    [
        "missing_block",
        "missing_field",
        "malformed_bool",
        "divergent_feed",
        "unexpected_field",
        "overlapping_exclusion",
        "event_blackout",
    ],
)
def test_load_backtest_baseline_requires_exact_pointer_live_execution_context(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    mutation: str,
) -> None:
    _write_approved_baseline_artifacts(tmp_path)
    _mock_committed_promotion_pointer(monkeypatch, tmp_path)
    payload = json.loads(
        (tmp_path / "config" / "live_config.json").read_text(encoding="utf-8")
    )
    context = payload["runtime"]["data_parity"]
    assert isinstance(context, dict)
    if mutation == "missing_block":
        payload["runtime"].pop("data_parity")
    elif mutation == "missing_field":
        context.pop("log_bar_components")
    elif mutation == "malformed_bool":
        context["apply_updated_bars"] = "true"
    elif mutation == "divergent_feed":
        context["live_feed"] = "sip"
    elif mutation == "unexpected_field":
        context["future_execution_switch"] = False
    elif mutation == "overlapping_exclusion":
        payload["runtime"]["excluded_symbols"] = ["AMD"]
    else:
        payload["runtime"]["symbol_event_blackouts"] = [
            {
                "symbol": "AMD",
                "start_utc": "2026-08-20T13:30:00Z",
                "end_utc": "2026-08-20T20:00:00Z",
                "reason": "earnings_blackout",
            }
        ]
    explicit_path = tmp_path / "config" / "paper_shadow_runtime.json"
    explicit_path.write_text(json.dumps(payload), encoding="utf-8")

    baseline, source = daily_report.load_backtest_baseline(
        tmp_path,
        explicit_path,
    )

    assert baseline["signal_rate_per_symbol_per_day"] == daily_report.DEFAULT_BACKTEST[
        "signal_rate_per_symbol_per_day"
    ]
    assert source["promotion_pointer_valid"] is True
    assert source["execution_context_matches_pointer_live"] is False
    assert source["execution_context_validation_errors"]
    assert source["research_metrics_used"] is False
    assert source["valid_for_comparison"] is False
    assert any(
        "runtime execution context is missing or invalid" in error
        for error in source["validation_errors"]
    )


@pytest.mark.parametrize(
    ("field", "divergent_value"),
    [
        ("max_usd_per_trade", 201.0),
        ("max_symbol_exposure_usd", 201.0),
        ("max_open_positions", 4),
        ("max_daily_loss_usd", 301.0),
        ("max_orders_per_minute", 7),
        ("max_price_deviation_bps", 76.0),
        ("max_data_delay_seconds", 301),
        ("max_live_price_age_seconds", 61),
    ],
)
def test_load_backtest_baseline_rejects_divergent_execution_limit(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    field: str,
    divergent_value: object,
) -> None:
    _write_approved_baseline_artifacts(tmp_path)
    _mock_committed_promotion_pointer(monkeypatch, tmp_path)
    payload = json.loads(
        (tmp_path / "config" / "live_config.json").read_text(encoding="utf-8")
    )
    payload["runtime"][field] = divergent_value
    explicit_path = tmp_path / "config" / "paper_shadow_runtime.json"
    explicit_path.write_text(json.dumps(payload), encoding="utf-8")

    baseline, source = daily_report.load_backtest_baseline(
        tmp_path,
        explicit_path,
    )

    assert baseline["signal_rate_per_symbol_per_day"] == daily_report.DEFAULT_BACKTEST[
        "signal_rate_per_symbol_per_day"
    ]
    assert source["promotion_pointer_valid"] is True
    assert source["execution_context_matches_pointer_live"] is False
    assert source["research_metrics_used"] is False
    assert source["valid_for_comparison"] is False
    assert any(
        field in error
        for error in source["execution_context_validation_errors"]
    )


@pytest.mark.parametrize(
    ("field", "malformed_value"),
    [
        ("max_usd_per_trade", 200),
        ("max_symbol_exposure_usd", 0.0),
        ("max_open_positions", True),
        ("max_daily_loss_usd", -1.0),
        ("max_orders_per_minute", 0),
        ("max_price_deviation_bps", -0.1),
        ("max_data_delay_seconds", -1),
        ("max_live_price_age_seconds", -1),
    ],
)
def test_load_backtest_baseline_rejects_malformed_execution_limit(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    field: str,
    malformed_value: object,
) -> None:
    _write_approved_baseline_artifacts(tmp_path)
    _mock_committed_promotion_pointer(monkeypatch, tmp_path)
    payload = json.loads(
        (tmp_path / "config" / "live_config.json").read_text(encoding="utf-8")
    )
    payload["runtime"][field] = malformed_value
    explicit_path = tmp_path / "config" / "paper_shadow_runtime.json"
    explicit_path.write_text(json.dumps(payload), encoding="utf-8")

    _, source = daily_report.load_backtest_baseline(tmp_path, explicit_path)

    assert source["promotion_pointer_valid"] is True
    assert source["execution_context_matches_pointer_live"] is False
    assert source["research_metrics_used"] is False
    assert source["valid_for_comparison"] is False
    assert any(
        field in error
        for error in source["execution_context_validation_errors"]
    )


def test_load_backtest_baseline_rejects_missing_execution_limit(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _write_approved_baseline_artifacts(tmp_path)
    _mock_committed_promotion_pointer(monkeypatch, tmp_path)
    payload = json.loads(
        (tmp_path / "config" / "live_config.json").read_text(encoding="utf-8")
    )
    payload["runtime"].pop("max_data_delay_seconds")
    explicit_path = tmp_path / "config" / "paper_shadow_runtime.json"
    explicit_path.write_text(json.dumps(payload), encoding="utf-8")

    _, source = daily_report.load_backtest_baseline(tmp_path, explicit_path)

    assert source["execution_context_matches_pointer_live"] is False
    assert source["research_metrics_used"] is False
    assert source["valid_for_comparison"] is False
    assert any(
        "max_data_delay_seconds" in error
        for error in source["execution_context_validation_errors"]
    )


def test_load_backtest_baseline_rejects_malformed_pointer_live_execution_context(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _write_approved_baseline_artifacts(tmp_path)
    live_path = tmp_path / "config" / "live_config.json"
    payload = json.loads(live_path.read_text(encoding="utf-8"))
    payload["runtime"]["data_parity"]["post_bar_reconcile_poll"] = 1
    live_path.write_text(json.dumps(payload), encoding="utf-8")
    _mock_committed_promotion_pointer(monkeypatch, tmp_path)

    _, source = daily_report.load_backtest_baseline(tmp_path)

    assert source["promotion_pointer_valid"] is True
    assert source["execution_context_matches_pointer_live"] is False
    assert source["research_metrics_used"] is False
    assert source["valid_for_comparison"] is False
    assert any(
        "pointer live config runtime data_parity.post_bar_reconcile_poll"
        in error
        for error in source["execution_context_validation_errors"]
    )


def test_load_backtest_baseline_rejects_malformed_pointer_live_execution_limit(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _write_approved_baseline_artifacts(tmp_path)
    live_path = tmp_path / "config" / "live_config.json"
    payload = json.loads(live_path.read_text(encoding="utf-8"))
    payload["runtime"]["max_orders_per_minute"] = False
    live_path.write_text(json.dumps(payload), encoding="utf-8")
    _mock_committed_promotion_pointer(monkeypatch, tmp_path)

    _, source = daily_report.load_backtest_baseline(tmp_path)

    assert source["promotion_pointer_valid"] is True
    assert source["execution_context_matches_pointer_live"] is False
    assert source["research_metrics_used"] is False
    assert source["valid_for_comparison"] is False
    assert any(
        "pointer live config runtime risk execution context" in error
        and "max_orders_per_minute" in error
        for error in source["execution_context_validation_errors"]
    )


def test_load_backtest_baseline_rejects_explicit_runtime_canonical_mismatch(
    monkeypatch,
) -> None:
    base_dir = Path.cwd() / f"test_daily_report_explicit_contract_{uuid.uuid4().hex}"
    try:
        _write_approved_baseline_artifacts(base_dir)
        _mock_committed_promotion_pointer(monkeypatch, base_dir)
        payload = json.loads(
            (base_dir / "config" / "live_config.json").read_text(encoding="utf-8")
        )
        payload["runtime"]["sma_bars"] = 21
        explicit_path = base_dir / "config" / "paper_shadow_runtime.json"
        explicit_path.write_text(json.dumps(payload), encoding="utf-8")

        baseline, source = daily_report.load_backtest_baseline(
            base_dir,
            explicit_path,
        )

        assert baseline["signal_rate_per_symbol_per_day"] == daily_report.DEFAULT_BACKTEST[
            "signal_rate_per_symbol_per_day"
        ]
        assert source["promotion_pointer_valid"] is True
        assert source["research_metrics_used"] is False
        assert source["valid_for_comparison"] is False
        assert (
            "live runtime canonical contract does not match the committed promotion bundle"
            in source["validation_errors"]
        )
    finally:
        shutil.rmtree(base_dir, ignore_errors=True)


@pytest.mark.parametrize(
    ("generation", "expected_error"),
    [
        (None, "live config research generation is missing or invalid"),
        (
            "3" * 64,
            "live config research generation does not match the committed promotion",
        ),
    ],
)
def test_load_backtest_baseline_rejects_explicit_runtime_without_exact_generation(
    monkeypatch,
    generation: str | None,
    expected_error: str,
) -> None:
    base_dir = Path.cwd() / f"test_daily_report_explicit_generation_{uuid.uuid4().hex}"
    try:
        _write_approved_baseline_artifacts(base_dir)
        _mock_committed_promotion_pointer(monkeypatch, base_dir)
        payload = json.loads(
            (base_dir / "config" / "live_config.json").read_text(encoding="utf-8")
        )
        if generation is None:
            payload["source"].pop("research_generation_sha256")
        else:
            payload["source"]["research_generation_sha256"] = generation
        explicit_path = base_dir / "config" / "paper_shadow_runtime.json"
        explicit_path.write_text(json.dumps(payload), encoding="utf-8")

        baseline, source = daily_report.load_backtest_baseline(
            base_dir,
            explicit_path,
        )

        assert baseline["signal_rate_per_symbol_per_day"] == daily_report.DEFAULT_BACKTEST[
            "signal_rate_per_symbol_per_day"
        ]
        assert source["promotion_pointer_valid"] is True
        assert source["research_metrics_used"] is False
        assert source["valid_for_comparison"] is False
        assert expected_error in source["validation_errors"]
    finally:
        shutil.rmtree(base_dir, ignore_errors=True)


def test_complete_legacy_research_config_without_canonical_evidence_stays_red() -> None:
    base_dir = Path.cwd() / f"test_daily_report_legacy_{uuid.uuid4().hex}"
    try:
        (base_dir / "config").mkdir(parents=True)
        (base_dir / "results").mkdir(parents=True)
        runtime = _complete_runtime()
        (base_dir / "config" / "live_config.json").write_text(
            json.dumps(
                {
                    "source": {"approved": True, "rejection_reasons": []},
                    "runtime": runtime,
                }
            ),
            encoding="utf-8",
        )
        (base_dir / "results" / "best_config_latest.json").write_text(
            json.dumps(
                {
                    "approved": True,
                    "rejection_reasons": [],
                    "config": _research_config(runtime),
                    "performance": {"trades_per_day": 6.0, "win_rate": 75.0},
                }
            ),
            encoding="utf-8",
        )

        _, source = daily_report.load_backtest_baseline(base_dir)

        assert source["valid_for_comparison"] is False
        assert source["research_metrics_used"] is False
        assert source["research_matches_live_runtime"] is None
        assert any(
            "research canonical strategy contract is missing or invalid" in error
            for error in source["validation_errors"]
        )
    finally:
        shutil.rmtree(base_dir, ignore_errors=True)


def test_load_backtest_baseline_rejects_unapproved_or_mismatched_research() -> None:
    base_dir = Path.cwd() / f"test_daily_report_baseline_invalid_{uuid.uuid4().hex}"
    try:
        (base_dir / "config").mkdir(parents=True)
        (base_dir / "results").mkdir(parents=True)
        runtime = _complete_runtime(
            sma_bars=15,
            entry_threshold_pct=0.0015,
            bb_period=25,
        )
        (base_dir / "config" / "live_config.json").write_text(
            json.dumps(
                {
                    "source": {
                        "approved": False,
                        "rejection_reasons": ["profit_factor 1.01 >= 1.2"],
                    },
                    "runtime": runtime,
                }
            ),
            encoding="utf-8",
        )
        (base_dir / "results" / "best_config_latest.json").write_text(
            json.dumps(
                {
                    "approved": False,
                    "rejection_reasons": ["profit_factor 1.18 >= 1.2"],
                    "config": _research_config(_complete_runtime()),
                    "canonical_strategy_contract": _canonical_research_contract(
                        _research_config(_complete_runtime())
                    ),
                    "performance": {
                        "trades_per_day": 6.0,
                        "win_rate": 75.0,
                    },
                }
            ),
            encoding="utf-8",
        )

        baseline, source = daily_report.load_backtest_baseline(base_dir)

        assert baseline["symbols"] == 3
        assert baseline["bars_per_day"] == 26
        assert baseline["signal_rate_per_symbol_per_day"] == daily_report.DEFAULT_BACKTEST["signal_rate_per_symbol_per_day"]
        assert baseline["win_rate"] == daily_report.DEFAULT_BACKTEST["win_rate"]
        assert source["research_metrics_used"] is False
        assert source["valid_for_comparison"] is False
        assert source["live_config_approved"] is False
        assert source["research_approved"] is False
        assert source["research_matches_live_runtime"] is False
        assert any("live config is approved=false" in item for item in source["validation_errors"])
        assert any("research artifact is approved=false" in item for item in source["validation_errors"])
        assert any("research config does not match live runtime" in item for item in source["validation_errors"])
    finally:
        shutil.rmtree(base_dir, ignore_errors=True)


def test_load_backtest_baseline_rejects_string_approval_flags() -> None:
    base_dir = Path.cwd() / f"test_daily_report_baseline_string_bool_{uuid.uuid4().hex}"
    try:
        (base_dir / "config").mkdir(parents=True)
        (base_dir / "results").mkdir(parents=True)
        (base_dir / "config" / "live_config.json").write_text(
            json.dumps(
                {
                    "source": {
                        "approved": "false",
                        "rejection_reasons": ["not approved yet"],
                    },
                    "runtime": _complete_runtime(symbols=["AMD"]),
                }
            ),
            encoding="utf-8",
        )
        (base_dir / "results" / "best_config_latest.json").write_text(
            json.dumps(
                {
                    "approved": "false",
                    "rejection_reasons": ["not approved yet"],
                    "config": _research_config(_complete_runtime(symbols=["AMD"])),
                    "canonical_strategy_contract": _canonical_research_contract(
                        _research_config(_complete_runtime(symbols=["AMD"]))
                    ),
                    "performance": {
                        "trades_per_day": 2.0,
                        "win_rate": 80.0,
                    },
                }
            ),
            encoding="utf-8",
        )

        _, source = daily_report.load_backtest_baseline(base_dir)

        assert source["live_config_approved"] is None
        assert source["research_approved"] is None
        assert source["valid_for_comparison"] is False
        assert any("approval metadata is missing or invalid" in item for item in source["validation_errors"])
    finally:
        shutil.rmtree(base_dir, ignore_errors=True)


def test_load_backtest_baseline_requires_positive_approval_metadata() -> None:
    base_dir = Path.cwd() / f"test_daily_report_baseline_missing_approval_{uuid.uuid4().hex}"
    try:
        (base_dir / "config").mkdir(parents=True)
        (base_dir / "results").mkdir(parents=True)
        runtime = _complete_runtime()
        (base_dir / "config" / "live_config.json").write_text(
            json.dumps({"runtime": runtime}),
            encoding="utf-8",
        )
        (base_dir / "results" / "best_config_latest.json").write_text(
            json.dumps(
                {
                    "config": _research_config(runtime),
                    "canonical_strategy_contract": (
                        _canonical_research_contract(_research_config(runtime))
                    ),
                    "performance": {"trades_per_day": 6.0, "win_rate": 0.75},
                }
            ),
            encoding="utf-8",
        )

        _, source = daily_report.load_backtest_baseline(base_dir)

        assert source["valid_for_comparison"] is False
        assert source["live_config_approved"] is None
        assert source["research_approved"] is None
    finally:
        shutil.rmtree(base_dir, ignore_errors=True)


def test_load_backtest_baseline_rejects_partial_promoted_config() -> None:
    base_dir = Path.cwd() / f"test_daily_report_baseline_partial_{uuid.uuid4().hex}"
    try:
        (base_dir / "config").mkdir(parents=True)
        (base_dir / "results").mkdir(parents=True)
        runtime = _complete_runtime()
        partial = _research_config(runtime)
        partial.pop("bb_period")
        (base_dir / "config" / "live_config.json").write_text(
            json.dumps(
                {
                    "source": {"approved": True, "rejection_reasons": []},
                    "runtime": runtime,
                }
            ),
            encoding="utf-8",
        )
        (base_dir / "results" / "best_config_latest.json").write_text(
            json.dumps(
                {
                    "approved": True,
                    "rejection_reasons": [],
                    "config": partial,
                    "performance": {"trades_per_day": 6.0, "win_rate": 0.75},
                }
            ),
            encoding="utf-8",
        )

        _, source = daily_report.load_backtest_baseline(base_dir)

        assert source["valid_for_comparison"] is False
        assert source["research_matches_live_runtime"] is None
        assert any(
            "research baseline is missing promoted fields (bb_period)" in item
            for item in source["validation_errors"]
        )
    finally:
        shutil.rmtree(base_dir, ignore_errors=True)


def test_load_backtest_baseline_rejects_invalid_performance_numeric_types_and_domains() -> None:
    base_dir = Path.cwd() / f"test_daily_report_baseline_metrics_{uuid.uuid4().hex}"
    try:
        (base_dir / "config").mkdir(parents=True)
        (base_dir / "results").mkdir(parents=True)
        runtime = _complete_runtime()
        (base_dir / "config" / "live_config.json").write_text(
            json.dumps(
                {
                    "source": {"approved": True, "rejection_reasons": []},
                    "runtime": runtime,
                }
            ),
            encoding="utf-8",
        )

        invalid_values = (
            ("trades_per_day", True),
            ("trades_per_day", "6.0"),
            ("trades_per_day", float("nan")),
            ("trades_per_day", float("inf")),
            ("trades_per_day", 10**1000),
            ("trades_per_day", -0.01),
            ("win_rate", False),
            ("win_rate", "75.0"),
            ("win_rate", float("nan")),
            ("win_rate", float("-inf")),
            ("win_rate", 10**1000),
            ("win_rate", -0.01),
            ("win_rate", 100.01),
        )
        for field_name, invalid_value in invalid_values:
            performance: dict[str, object] = {
                "trades_per_day": 6.0,
                "win_rate": 75.0,
            }
            performance[field_name] = invalid_value
            (base_dir / "results" / "best_config_latest.json").write_text(
                json.dumps(
                    {
                        "approved": True,
                        "rejection_reasons": [],
                        "config": _research_config(runtime),
                        "canonical_strategy_contract": (
                            _canonical_research_contract(_research_config(runtime))
                        ),
                        "performance": performance,
                    }
                ),
                encoding="utf-8",
            )

            baseline, source = daily_report.load_backtest_baseline(base_dir)

            assert source["valid_for_comparison"] is False
            assert source["research_metrics_used"] is False
            assert baseline["win_rate"] == daily_report.DEFAULT_BACKTEST["win_rate"]
            if isinstance(invalid_value, float) and not math.isfinite(invalid_value):
                assert "research baseline artifact is missing or invalid" in source[
                    "validation_errors"
                ]
            else:
                assert any(
                    f"research {field_name} is missing or invalid" in error
                    for error in source["validation_errors"]
                )
    finally:
        shutil.rmtree(base_dir, ignore_errors=True)


def test_load_backtest_baseline_requires_the_exact_research_symbol_universe() -> None:
    base_dir = Path.cwd() / f"test_daily_report_baseline_symbols_{uuid.uuid4().hex}"
    try:
        (base_dir / "config").mkdir(parents=True)
        (base_dir / "results").mkdir(parents=True)
        runtime = _complete_runtime()
        (base_dir / "config" / "live_config.json").write_text(
            json.dumps(
                {
                    "source": {"approved": True, "rejection_reasons": []},
                    "runtime": runtime,
                }
            ),
            encoding="utf-8",
        )

        research_config = _research_config(runtime)
        research_config["symbols"] = ["NVDA", "MSFT", "TSLA"]
        research_config["symbol_strategy_modes"] = {
            "NVDA": "mean_reversion",
            "MSFT": "mean_reversion",
            "TSLA": "mean_reversion",
        }
        research_path = base_dir / "results" / "best_config_latest.json"
        research_path.write_text(
            json.dumps(
                {
                    "approved": True,
                    "rejection_reasons": [],
                    "config": research_config,
                    "canonical_strategy_contract": (
                        _canonical_research_contract(research_config)
                    ),
                    "performance": {"trades_per_day": 6.0, "win_rate": 75.0},
                }
            ),
            encoding="utf-8",
        )

        _, divergent_source = daily_report.load_backtest_baseline(base_dir)

        assert divergent_source["valid_for_comparison"] is False
        assert divergent_source["research_matches_live_symbols"] is False
        assert any(
            "research symbols do not match live runtime" in error
            for error in divergent_source["validation_errors"]
        )

        research_config.pop("symbols")
        research_path.write_text(
            json.dumps(
                {
                    "approved": True,
                    "rejection_reasons": [],
                    "config": research_config,
                    "performance": {"trades_per_day": 6.0, "win_rate": 75.0},
                }
            ),
            encoding="utf-8",
        )

        _, missing_source = daily_report.load_backtest_baseline(base_dir)

        assert missing_source["valid_for_comparison"] is False
        assert missing_source["research_matches_live_symbols"] is None
        assert "research baseline symbols are missing or invalid" in missing_source[
            "validation_errors"
        ]
    finally:
        shutil.rmtree(base_dir, ignore_errors=True)


def test_load_backtest_baseline_rejects_runtime_config_outside_repository() -> None:
    base_dir = Path.cwd() / f"test_daily_report_baseline_root_{uuid.uuid4().hex}"
    outside_config = Path.cwd() / f"test_outside_runtime_{uuid.uuid4().hex}.json"
    try:
        base_dir.mkdir()
        outside_config.write_text("{}", encoding="utf-8")

        _, source = daily_report.load_backtest_baseline(
            base_dir,
            runtime_config_path=outside_config,
        )

        assert source["valid_for_comparison"] is False
        assert source["live_config_path"] is None
        assert "runtime config path is invalid or outside the repository" in source[
            "validation_errors"
        ]
    finally:
        shutil.rmtree(base_dir, ignore_errors=True)
        outside_config.unlink(missing_ok=True)

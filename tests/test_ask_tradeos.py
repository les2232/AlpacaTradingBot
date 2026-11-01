"""Focused tests for Ask TradeOS (tradeos.ask + dashboard_ask).

All model calls use mocked Responses clients; nothing contacts the network,
submits orders, or writes runtime state.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import dotenv
import httpx2 as httpx
import openai
import pytest

from drift_monitor import BaselineProfile
from tradeos.ask import (
    QUESTION_COMPARE,
    QUESTION_EXPLAIN,
    QUESTION_SYMBOL,
    AskSettings,
    ask,
    build_evidence_package,
    calculated_answer,
    discover_sessions,
    load_ask_settings,
)
from tradeos.ask.analyst import (
    ANSWER_SCHEMA,
    ModelOutputError,
    assert_no_secrets,
    build_request,
    extract_output_text,
    validate_model_output,
)
from tradeos.ask.comparison import list_comparison_targets

DAY = "2026-10-15"
LEGACY_DAY = "2026-10-14"
START = datetime(2026, 10, 15, 13, 45, tzinfo=timezone.utc)  # 09:45 ET
SHA_STRATEGY = "a" * 64
SHA_BUNDLE = "b" * 64
SHA_GENERATION = "c" * 64
SECRET_FINGERPRINT = "acct-fp-SECRET-123"
API_KEY = "sk-test-0123456789abcdefghijklmnop"


# ---------------------------------------------------------------------------
# Fixture construction
# ---------------------------------------------------------------------------

def _decision(k: int) -> str:
    return (START + timedelta(minutes=15 * k)).isoformat()


def _ts(k: int, seconds: int = 20) -> str:
    return (START + timedelta(minutes=15 * k, seconds=seconds)).isoformat()


def _signal(symbol: str, k: int, action: str, rejection: str | None, session_id: str | None) -> dict[str, Any]:
    payload = {
        "trace": f"{symbol}_{k}",
        "symbol": symbol,
        "decision_ts": _decision(k),
        "bar_close": 100.0 + k,
        "sma": 101.0,
        "deviation_pct": round((100.0 + k - 101.0) / 101.0 * 100, 4),
        "trend_filter": "reject" if rejection == "trend_filter" else "pass",
        "atr_filter": "reject" if rejection == "atr_filter" else "pass",
        "window_open": True,
        "action": action,
        "holding": rejection == "holding",
        "rejection": rejection,
        "event": "signal.evaluated",
        "ts": _ts(k),
    }
    if session_id:
        payload["session_id"] = session_id
    return payload


def _write_jsonl(path: Path, rows: list[dict[str, Any]], extra_lines: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [json.dumps(row) for row in rows] + list(extra_lines or [])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _startup_artifact(session_id: str, **overrides: Any) -> dict[str, Any]:
    payload = {
        "artifact_type": "startup_config",
        "session_id": session_id,
        "started_at_utc": "2026-10-15T13:30:00+00:00",
        "launch_mode": "paper-shadow",
        "execution_enabled": False,
        "paper": True,
        "account_mode": "paper",
        "strategy_mode": "mean_reversion",
        "bar_timeframe_minutes": 15,
        "sma_bars": 20,
        "symbols": ["AAA", "BBB"],
        "symbol_count": 2,
        "db_path": "/home/operator/private/bot_history.db",
        "runtime_config_path": "/home/operator/private/config/live_config.json",
        "expected_account_fingerprint": SECRET_FINGERPRINT,
        "baseline_valid_for_comparison": True,
        "baseline_validation_errors": [],
        "canonical_strategy_sha256": SHA_STRATEGY,
        "promotion_bundle_sha256": SHA_BUNDLE,
        "research_generation_sha256": SHA_GENERATION,
        "max_usd_per_trade": 200.0,
        "max_open_positions": 3,
        "max_daily_loss_usd": 300.0,
    }
    payload.update(overrides)
    return payload


def build_fixture(root: Path, *, artifact_overrides: dict[str, Any] | None = None, db_env: str | None = "paper") -> dict[str, Path]:
    logs = root / "logs"
    day = logs / DAY
    a, b = "sess-A", "sess-B"
    signals: list[dict[str, Any]] = []
    bars: list[dict[str, Any]] = []
    for k in range(25):
        if k == 2:
            aaa = ("BUY", None)
        elif 3 <= k <= 5:
            aaa = ("HOLD", "holding")
        elif k == 6:
            aaa = ("SELL", None)
        else:
            aaa = ("HOLD", "no_signal")
        signals.append(_signal("AAA", k, aaa[0], aaa[1], a))
        signals.append(_signal("BBB", k, "HOLD", "trend_filter", a))
        for sym in ("AAA", "BBB"):
            bars.append({"trace": f"{sym}_{k}", "symbol": sym, "bar_close": 100.0, "bar_age_s": 5.0, "stale": False,
                         "event": "bar.received", "ts": _ts(k, 5), "session_id": a})
    for k in range(4):  # sess-B: partial, different behaviour for AAA
        signals.append(_signal("AAA", k, "HOLD", "atr_filter", b))
        signals.append(_signal("CCC", k, "HOLD", "no_signal", b))
    for k in range(10, 15):  # legacy rows without a session id, same day
        signals.append(_signal("AAA", k, "BUY", None, None))
    _write_jsonl(day / "signals.jsonl", signals)
    _write_jsonl(day / "bars.jsonl", bars)
    _write_jsonl(
        day / "risk.jsonl",
        [
            {"trace": "AAA_2", "symbol": "AAA", "action": "BUY", "allowed": True, "block_reason": None,
             "open_positions": 0, "max_positions": 3, "daily_pnl": 0.0, "event": "risk.check", "ts": _ts(2, 30), "session_id": a},
            {"trace": "AAA_6", "symbol": "AAA", "action": "SELL", "allowed": True, "block_reason": None,
             "open_positions": 1, "max_positions": 3, "daily_pnl": 0.0, "event": "risk.check", "ts": _ts(6, 30), "session_id": a},
            {"trace": "AAA_11", "symbol": "AAA", "action": "BUY", "allowed": False, "block_reason": "max_positions",
             "event": "risk.check", "ts": _ts(11, 30)},
        ],
    )
    _write_jsonl(
        day / "execution.jsonl",
        [
            {"trace": "AAA_2", "symbol": "AAA", "side": "buy", "qty": 2, "live_price": 102.0, "notional": 204.0,
             "order_id": "ord-1", "event": "order.submitted", "ts": _ts(2, 40), "session_id": a},
            {"trace": "AAA_2", "symbol": "AAA", "side": "buy", "fill_price": 102.05, "fill_qty": 2, "slippage_bps": 4.9,
             "submit_to_fill_ms": 300, "order_id": "ord-1", "event": "order.filled", "ts": _ts(2, 50), "session_id": a},
            {"trace": "AAA_6", "symbol": "AAA", "side": "sell", "qty": 2, "live_price": 106.0, "notional": 212.0,
             "order_id": "ord-2", "event": "order.submitted", "ts": _ts(6, 40), "session_id": a},
            {"trace": "AAA_6", "symbol": "AAA", "side": "sell", "fill_price": 105.9, "fill_qty": 2, "slippage_bps": 9.4,
             "submit_to_fill_ms": 280, "order_id": "ord-2", "event": "order.filled", "ts": _ts(6, 50), "session_id": a},
        ],
    )
    _write_jsonl(
        day / "positions.jsonl",
        [
            {"trace": "AAA_2", "symbol": "AAA", "entry_price": 102.05, "qty": 2, "notional": 204.1,
             "strategy_mode": "mean_reversion", "event": "position.opened", "ts": _ts(2, 55), "session_id": a},
            {"trace": "AAA_2", "symbol": "AAA", "entry_price": 102.05, "exit_price": 105.9, "qty": 2, "pnl_usd": 7.7,
             "pnl_pct": 3.77, "winner": True, "holding_bars": 4, "holding_minutes": 60.0, "exit_reason": "sma_recovery",
             "event": "position.closed", "ts": _ts(6, 55), "session_id": a},
        ],
    )
    _write_jsonl(
        day / "lifecycle.jsonl",
        [{"stage": "startup.ready", "cash": 12345.67, "equity": 12345.67, "buying_power": 24691.34,
          "execute_orders": False, "event": "process.lifecycle", "ts": "2026-10-15T13:31:00+00:00", "session_id": a}],
    )
    artifact = _startup_artifact(a, **(artifact_overrides or {}))
    (day / "startup_config.2026-10-15T13-30-00Z.json").write_text(json.dumps(artifact), encoding="utf-8")

    _write_jsonl(
        logs / LEGACY_DAY / "signals.jsonl",
        [_signal("AAA", k, "HOLD", "no_signal", None) for k in range(3)],
        extra_lines=["{not json"],
    )

    db = root / "bot_history.db"
    with sqlite3.connect(db) as connection:
        connection.execute(
            "CREATE TABLE bot_runs (id INTEGER PRIMARY KEY AUTOINCREMENT, session_id TEXT, environment TEXT, "
            "account_fingerprint TEXT, timestamp_utc TEXT NOT NULL)"
        )
        for k in (0, 10, 24):
            connection.execute(
                "INSERT INTO bot_runs (session_id, environment, account_fingerprint, timestamp_utc) VALUES (?, ?, ?, ?)",
                (a, db_env, SECRET_FINGERPRINT, _ts(k)),
            )
        connection.execute(
            "INSERT INTO bot_runs (session_id, environment, timestamp_utc) VALUES (NULL, NULL, ?)", (_ts(12),)
        )
    return {"logs": logs, "db": db, "root": root}


@pytest.fixture()
def fx(tmp_path: Path) -> dict[str, Path]:
    return build_fixture(tmp_path)


def _session(fx: dict[str, Path], key: str):
    sessions = {s.key: s for s in discover_sessions(fx["logs"], fx["db"])}
    return sessions[key]


def _package(fx, key, question, **kwargs):
    return build_evidence_package(
        _session(fx, key), question, log_root=fx["logs"], repo_root=fx["root"], db_path=fx["db"], **kwargs
    )


def _record_at(fx: dict[str, Path], source: str) -> dict[str, Any]:
    path, _, line = source.rpartition(":L")
    lines = (fx["logs"].parent / path).read_text(encoding="utf-8").splitlines()
    return json.loads(lines[int(line) - 1])


def _valid_loader(_root: Path):
    return (
        {"symbols": 2, "bars_per_day": 26, "signal_rate_per_symbol_per_day": 0.5, "win_rate": 0.6,
         "avg_hold_bars": 3.0, "rejection_rate": 0.9, "avg_slippage_bps": 5.0},
        {"valid_for_comparison": True, "validation_errors": [], "promotion_bundle_sha256": SHA_BUNDLE,
         "research_generation_sha256": SHA_GENERATION},
    )


def _valid_profile(_root: Path) -> BaselineProfile:
    return BaselineProfile(
        source_label="test", valid_for_comparison=True, validation_errors=(), strategy_mode="mean_reversion",
        symbols=2, bars_per_day=26, buy_signal_rate_per_symbol_per_day=0.5, sell_signal_rate_per_symbol_per_day=0.5,
        rejection_rate=0.9, rejection_breakdown=(), hybrid_branch_participation=(), avg_bar_age_s=240.0,
        stale_bar_rate=0.0, trade_count_per_day=1.0, win_rate=0.6,
    )


class FakeClient:
    def __init__(self, response: Any = None, exc: Exception | None = None) -> None:
        self.response, self.exc, self.calls = response, exc, []

    def create(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        if self.exc is not None:
            raise self.exc
        return self.response


def _response(payload: Any, *, status: str = "completed") -> dict[str, Any]:
    text = payload if isinstance(payload, str) else json.dumps(payload)
    return {"id": "resp_test", "status": status, "output_text": text,
            "output": [{"type": "message", "content": [{"type": "output_text", "text": text}]}]}


SETTINGS = AskSettings(model="gpt-6.1-sol", timeout_seconds=5.0, api_key=API_KEY, api_key_source="test")
NO_KEY = AskSettings(model="gpt-6.1-sol", timeout_seconds=5.0, api_key=None)


# ---------------------------------------------------------------------------
# Session catalog and isolation
# ---------------------------------------------------------------------------

def test_catalog_separates_bound_sessions_from_unbound_log_days(fx) -> None:
    sessions = {s.key: s for s in discover_sessions(fx["logs"], fx["db"])}
    assert {"session:sess-A", "session:sess-B", f"day:{DAY}", f"day:{LEGACY_DAY}"} <= set(sessions)
    assert sessions["session:sess-A"].kind == "bound"
    assert sessions["session:sess-A"].db_run_count == 3
    assert sessions[f"day:{DAY}"].kind == "unbound_log_day"
    assert "unbound" in sessions[f"day:{DAY}"].label


def test_session_scope_excludes_other_sessions_and_records_without_session_id(fx) -> None:
    pkg = _package(fx, "session:sess-A", QUESTION_EXPLAIN)
    signals = pkg.metrics["session"]["signals"]
    assert signals["evaluations"] == 50
    assert signals["buy"] == 1 and signals["sell"] == 1
    streams = pkg.data_coverage["streams"]
    assert streams["signals"]["excluded_without_session_id"] == 5
    assert streams["signals"]["excluded_other_sessions"] == 8
    assert any("lack a session id" in text for text in pkg.limitations)
    for item in pkg.items:
        if item.kind == "event":
            assert _record_at(fx, item.source).get("session_id") == "sess-A"


def test_symbol_question_is_isolated_to_symbol_and_session(fx) -> None:
    pkg = _package(fx, "session:sess-A", QUESTION_SYMBOL, symbol="aaa")
    metrics = pkg.metrics["symbol"]
    assert pkg.symbol == "AAA"
    assert metrics["evaluations"] == 25
    assert metrics["actions"] == {"HOLD": 23, "BUY": 1, "SELL": 1}
    assert "atr_filter" not in metrics["hold_reasons"]  # sess-B behaviour
    assert metrics["risk_blocks_by_action_and_reason"] == {}  # legacy block excluded
    assert metrics["fill_events"] == 2 and metrics["realized_pnl_usd"] == 7.7
    events = [item for item in pkg.items if item.kind == "event"]
    assert events
    for item in events:
        record = _record_at(fx, item.source)
        assert record["symbol"] == "AAA" and record["session_id"] == "sess-A"


def test_symbol_without_records_is_disclosed(fx) -> None:
    pkg = _package(fx, "session:sess-A", QUESTION_SYMBOL, symbol="CCC")
    assert pkg.metrics["symbol"]["evaluations"] == 0
    assert any("No records for CCC" in text for text in pkg.limitations)
    assert any("not in the session's configured symbol list" in text for text in pkg.limitations)


def test_symbol_question_requires_symbol(fx) -> None:
    with pytest.raises(ValueError):
        _package(fx, "session:sess-A", QUESTION_SYMBOL)


# ---------------------------------------------------------------------------
# Historical configuration and ambiguous records
# ---------------------------------------------------------------------------

def test_bound_session_uses_its_own_startup_artifact(fx) -> None:
    pkg = _package(fx, "session:sess-A", QUESTION_EXPLAIN)
    assert pkg.configuration["status"] == "recorded_at_launch"
    assert pkg.configuration["values"]["symbols"] == ["AAA", "BBB"]
    assert pkg.session["environment"]["recorded_at_launch"] == "paper"


def test_session_without_artifact_never_borrows_current_configuration(fx) -> None:
    pkg = _package(fx, "session:sess-B", QUESTION_EXPLAIN)
    assert pkg.configuration["status"] == "not_recorded"
    assert "values" not in pkg.configuration
    assert pkg.session["symbols_configured"] is None
    assert "live_config.json is not used" in pkg.configuration["note"]


def test_unbound_log_day_is_not_attributable(fx) -> None:
    pkg = _package(fx, f"day:{DAY}", QUESTION_EXPLAIN)
    assert pkg.configuration["status"] == "not_attributable"
    assert pkg.metrics["session"]["signals"]["evaluations"] == 5  # only rows without a session id
    assert pkg.data_coverage["streams"]["signals"]["excluded_other_sessions"] == 58
    assert "not a single session" in pkg.session["scope"]
    assert pkg.comparison_status == "unavailable"


def test_conflicting_startup_artifacts_make_configuration_ambiguous(fx) -> None:
    extra = _startup_artifact("sess-A", strategy_mode="hybrid")
    (fx["logs"] / DAY / "startup_config.2026-10-15T13-31-00Z.json").write_text(json.dumps(extra), encoding="utf-8")
    pkg = _package(fx, "session:sess-A", QUESTION_COMPARE)
    assert pkg.configuration["status"] == "conflicting"
    assert pkg.comparison_status == "unavailable"


def test_missing_environment_identifier_blocks_comparison(tmp_path: Path) -> None:
    fx = build_fixture(tmp_path, db_env=None)
    pkg = _package(fx, "session:sess-A", QUESTION_COMPARE, baseline_loader=_valid_loader, drift_profile_loader=_valid_profile)
    assert pkg.comparison_status == "unavailable"
    assert any("lack an environment identifier" in reason for reason in pkg.comparison["reasons"])


# ---------------------------------------------------------------------------
# Baseline validity and comparison gating
# ---------------------------------------------------------------------------

def test_invalid_saved_baseline_returns_comparison_unavailable_without_expected_values(fx) -> None:
    pkg = _package(fx, "session:sess-A", QUESTION_COMPARE)  # real loader; fixture has no promotion pointer
    assert pkg.comparison_status == "unavailable"
    assert any("fails current validation" in reason for reason in pkg.comparison["reasons"])
    assert "metric_comparisons" not in pkg.comparison
    assert "underperformance" in pkg.comparison["suppressed_conclusions"]
    body = json.dumps(pkg.to_model_payload())
    for fallback_key in ("signal_rate_per_symbol_per_day", "avg_slippage_bps\": 5.0", "rejection_rate"):
        assert fallback_key not in body
    # Observed metrics are still reported.
    assert pkg.metrics["session"]["signals"]["evaluations"] == 50


def test_baseline_identity_mismatch_blocks_comparison(fx) -> None:
    def other_bundle(_root):
        base, source = _valid_loader(_root)
        source["promotion_bundle_sha256"] = "d" * 64
        return base, source

    pkg = _package(fx, "session:sess-A", QUESTION_COMPARE, baseline_loader=other_bundle, drift_profile_loader=_valid_profile)
    checks = {c["check"]: c["passed"] for c in pkg.comparison["checks"]}
    assert checks["baseline_valid_now"] is True
    assert checks["baseline_identity_matches_session"] is False
    assert pkg.comparison_status == "unavailable"


def test_baseline_invalid_at_launch_blocks_comparison(tmp_path: Path) -> None:
    fx = build_fixture(
        tmp_path,
        artifact_overrides={"baseline_valid_for_comparison": False, "baseline_validation_errors": ["pointer missing"]},
    )
    pkg = _package(fx, "session:sess-A", QUESTION_COMPARE, baseline_loader=_valid_loader, drift_profile_loader=_valid_profile)
    assert pkg.comparison_status == "unavailable"
    assert any("pointer missing" in reason for reason in pkg.comparison["reasons"])


def test_fully_valid_comparison_is_available_and_gates_small_samples(fx) -> None:
    pkg = _package(fx, "session:sess-A", QUESTION_COMPARE, baseline_loader=_valid_loader, drift_profile_loader=_valid_profile)
    assert pkg.comparison_status == "available", pkg.comparison["reasons"]
    rows = {row["metric"]: row for row in pkg.comparison["metric_comparisons"]}
    assert rows["hold_rate"]["available"] is True
    assert rows["hold_rate"]["expected"] == 0.9
    assert rows["win_rate"]["available"] is False  # one closed trade
    assert rows["win_rate"]["expected"] is None


def test_invalid_drift_profile_blocks_comparison(fx) -> None:
    def invalid_profile(root):
        from dataclasses import replace

        return replace(_valid_profile(root), valid_for_comparison=False, validation_errors=("overlay missing",))

    pkg = _package(fx, "session:sess-A", QUESTION_COMPARE, baseline_loader=_valid_loader, drift_profile_loader=invalid_profile)
    assert pkg.comparison_status == "unavailable"
    assert "metric_comparisons" not in pkg.comparison


def test_experiment_targets_without_canonical_identity_are_unavailable(fx) -> None:
    log = fx["root"] / "results" / "experiment_log.jsonl"
    log.parent.mkdir()
    log.write_text(json.dumps({"run_id": "run-1", "strategy_name": "sma", "symbols": ["AAA"], "metrics": {"win_rate": 0.7}}) + "\n")
    targets = dict(list_comparison_targets(log))
    assert "experiment:run-1" in targets
    pkg = _package(fx, "session:sess-A", QUESTION_COMPARE, comparison_target="experiment:run-1", experiment_log_path=log)
    assert pkg.comparison_status == "unavailable"
    joined = " ".join(pkg.comparison["reasons"])
    assert "no matching canonical strategy identity" in joined and "strategy sma vs session mean_reversion" in joined


def test_missing_experiment_log_lists_only_promoted_baseline(tmp_path: Path) -> None:
    assert [key for key, _ in list_comparison_targets(tmp_path / "missing.jsonl")] == ["promoted_baseline"]


# ---------------------------------------------------------------------------
# Partial sessions, insufficient samples, missing and malformed logs
# ---------------------------------------------------------------------------

def test_partial_session_is_flagged_and_blocks_comparison(fx) -> None:
    pkg = _package(fx, "session:sess-B", QUESTION_COMPARE, baseline_loader=_valid_loader, drift_profile_loader=_valid_profile)
    decisions = pkg.data_coverage["decisions"]
    assert decisions["distinct_decisions_in_entry_window"] == 4
    assert decisions["partial_session"] is True
    assert decisions["timeframe_source"] == "inferred from decision spacing"
    assert any(text.startswith("Partial session") for text in pkg.limitations)
    checks = {c["check"]: c["passed"] for c in pkg.comparison["checks"]}
    assert checks["sufficient_session_sample"] is False


def test_full_session_coverage_and_small_trade_sample(fx) -> None:
    pkg = _package(fx, "session:sess-A", QUESTION_EXPLAIN)
    assert pkg.data_coverage["decisions"]["coverage_ratio"] == 1.0
    assert pkg.data_coverage["decisions"]["partial_session"] is False
    assert pkg.metrics["session"]["positions"]["win_rate_sample_sufficient"] is False
    assert any("win rate needs at least 5" in text for text in pkg.limitations)


def test_missing_streams_and_malformed_lines_are_disclosed(fx) -> None:
    pkg = _package(fx, f"day:{LEGACY_DAY}", QUESTION_EXPLAIN)
    assert any("Missing log stream(s): execution, positions, risk, bars" in text for text in pkg.limitations)
    assert pkg.data_coverage["streams"]["signals"]["malformed_lines_in_files"] == 1
    assert any("malformed" in text for text in pkg.limitations)


def test_empty_log_root_has_no_sessions(tmp_path: Path) -> None:
    assert discover_sessions(tmp_path / "logs") == []


def test_db_only_session_without_logs_reports_missing_records(fx) -> None:
    with sqlite3.connect(fx["db"]) as connection:
        connection.execute(
            "INSERT INTO bot_runs (session_id, environment, timestamp_utc) VALUES ('sess-db', 'paper', '2026-10-13T14:00:00+00:00')"
        )
    pkg = _package(fx, "session:sess-db", QUESTION_EXPLAIN)
    assert any("No log records were found" in text for text in pkg.limitations)
    assert pkg.metrics["session"]["signals"]["evaluations"] == 0


def test_exact_duplicate_records_are_removed(fx) -> None:
    path = fx["logs"] / DAY / "positions.jsonl"
    lines = path.read_text().splitlines()
    path.write_text("\n".join(lines + [lines[-1]]) + "\n")
    pkg = _package(fx, "session:sess-A", QUESTION_EXPLAIN)
    assert pkg.data_coverage["exact_duplicate_records_removed"] == 1
    assert pkg.metrics["session"]["positions"]["close_events"] == 1


# ---------------------------------------------------------------------------
# Data minimisation and request construction
# ---------------------------------------------------------------------------

def test_request_contains_only_needed_evidence_and_no_identifiers(fx) -> None:
    pkg = _package(fx, "session:sess-A", QUESTION_EXPLAIN)
    request = build_request(pkg, SETTINGS)
    body = json.dumps(request)
    assert request["model"] == "gpt-6.1-sol"
    assert request["store"] is False
    assert request["text"]["format"]["type"] == "json_schema"
    assert request["text"]["format"]["strict"] is True
    assert request["text"]["format"]["schema"] == ANSWER_SCHEMA
    for forbidden in (API_KEY, SECRET_FINGERPRINT, "ord-1", "/home/operator", "12345.67", "order_id"):
        assert forbidden not in body


def test_secret_like_values_block_the_request(fx) -> None:
    risk = fx["logs"] / DAY / "risk.jsonl"
    with risk.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"symbol": "AAA", "action": "BUY", "error": "auth failed for sk-abcdefghijklmnopqrstuvwx",
                                 "event": "execution.error", "ts": _ts(8), "session_id": "sess-A"}) + "\n")
    pkg = _package(fx, "session:sess-A", QUESTION_EXPLAIN)
    client = FakeClient(_response({}))
    result = ask(pkg, SETTINGS, client=client)
    assert client.calls == []
    assert result.answer.mode == "calculated"
    assert "credential-like" in (result.error or "")
    with pytest.raises(RuntimeError):
        assert_no_secrets(f'{{"x": "{API_KEY}"}}', SETTINGS)


# ---------------------------------------------------------------------------
# Model output validation and citations
# ---------------------------------------------------------------------------

def _good_output(pkg, **overrides) -> dict[str, Any]:
    payload = {
        "answer": "AAA was bought once and sold once [M1].",
        "findings": [
            {"statement": "One BUY and one SELL signal were logged.", "kind": "observation", "citations": ["M1"]},
            {"statement": "The exit likely followed SMA recovery.", "kind": "hypothesis", "citations": ["E1", "M4"]},
        ],
        "missing_evidence": ["Win rate is not interpretable from one trade."],
        "comparison_status": "not_applicable",
    }
    payload.update(overrides)
    return payload


def test_valid_model_output_is_accepted_with_resolving_citations(fx) -> None:
    pkg = _package(fx, "session:sess-A", QUESTION_EXPLAIN)
    client = FakeClient(_response(_good_output(pkg)))
    result = ask(pkg, SETTINGS, client=client)
    assert result.error is None
    assert result.answer.mode == "model" and result.answer.model == "gpt-6.1-sol"
    assert result.answer.response_id == "resp_test"
    assert [f.kind for f in result.answer.findings] == ["observation", "hypothesis"]
    assert set(result.answer.cited_ids) <= pkg.item_ids
    assert client.calls[0]["timeout"] == 5.0


def test_findings_with_unknown_or_missing_citations_are_dropped(fx) -> None:
    pkg = _package(fx, "session:sess-A", QUESTION_EXPLAIN)
    output = _good_output(
        pkg,
        answer="Summary [M1] with a fabricated reference [Z99].",
        findings=[
            {"statement": "Cited correctly.", "kind": "observation", "citations": ["M1"]},
            {"statement": "Cites a fake record.", "kind": "observation", "citations": ["E999"]},
            {"statement": "Cites nothing.", "kind": "hypothesis", "citations": []},
        ],
    )
    answer = validate_model_output(json.dumps(output), pkg)
    assert [f.statement for f in answer.findings] == ["Cited correctly."]
    assert "[Z99]" not in answer.answer and "[M1]" in answer.answer
    assert len(answer.warnings) == 3
    assert set(answer.cited_ids) <= pkg.item_ids


def test_comparative_claims_are_suppressed_when_comparison_unavailable(fx) -> None:
    pkg = _package(fx, "session:sess-A", QUESTION_COMPARE)
    output = _good_output(
        pkg,
        answer="The session underperformed the backtest. Comparison unavailable, so drift cannot be assessed.",
        findings=[
            {"statement": "Win rate was below the backtest expectation.", "kind": "observation", "citations": ["M1"]},
            {"statement": "Underperformance cannot be established because the baseline is invalid.", "kind": "observation", "citations": ["CMP1"]},
        ],
        comparison_status="unavailable",
    )
    answer = validate_model_output(json.dumps(output), pkg)
    assert "underperformed the backtest" not in answer.answer
    assert answer.answer.startswith("Comparison unavailable")
    assert [f.citations for f in answer.findings] == [("CMP1",)]
    assert any("comparative" in w for w in answer.warnings)


def test_model_claiming_available_comparison_falls_back(fx) -> None:
    pkg = _package(fx, "session:sess-A", QUESTION_COMPARE)
    result = ask(pkg, SETTINGS, client=FakeClient(_response(_good_output(pkg, comparison_status="available"))))
    assert result.answer.mode == "calculated"
    assert "evidence says 'unavailable'" in (result.error or "")
    assert result.answer.answer.startswith("Comparison unavailable")


@pytest.mark.parametrize(
    "raw, message",
    [
        ("not json at all", "not valid JSON"),
        (json.dumps(["a"]), "not a JSON object"),
        (json.dumps({"answer": "x", "findings": []}), "do not match the schema"),
        (json.dumps({"answer": "", "findings": [], "missing_evidence": [], "comparison_status": "not_applicable"}), "empty"),
        (json.dumps({"answer": "x", "findings": "y", "missing_evidence": [], "comparison_status": "not_applicable"}), "not a list"),
        (json.dumps({"answer": "x", "findings": [], "missing_evidence": [], "comparison_status": "maybe"}), "invalid"),
    ],
)
def test_malformed_model_output_falls_back_to_calculated(fx, raw: str, message: str) -> None:
    pkg = _package(fx, "session:sess-A", QUESTION_EXPLAIN)
    with pytest.raises(ModelOutputError, match=message):
        validate_model_output(raw, pkg)
    result = ask(pkg, SETTINGS, client=FakeClient(_response(raw)))
    assert result.answer.mode == "calculated" and result.error


def test_refusal_and_incomplete_responses_are_reported(fx) -> None:
    refusal = {"status": "completed", "output": [{"type": "message", "content": [{"type": "refusal", "refusal": "no"}]}]}
    with pytest.raises(ModelOutputError, match="declined"):
        extract_output_text(refusal)
    incomplete = {"status": "incomplete", "incomplete_details": {"reason": "max_output_tokens"}, "output": []}
    with pytest.raises(ModelOutputError, match="max_output_tokens"):
        extract_output_text(incomplete)
    pkg = _package(fx, "session:sess-A", QUESTION_EXPLAIN)
    result = ask(pkg, SETTINGS, client=FakeClient(incomplete))
    assert result.answer.mode == "calculated" and "incomplete" in (result.error or "")


def test_calculated_answer_citations_resolve(fx) -> None:
    for question, symbol in ((QUESTION_EXPLAIN, None), (QUESTION_SYMBOL, "AAA"), (QUESTION_COMPARE, None)):
        pkg = _package(fx, "session:sess-A", question, symbol=symbol)
        answer = calculated_answer(pkg)
        assert answer.mode == "calculated" and answer.findings
        assert set(answer.cited_ids) <= pkg.item_ids
        assert all(f.kind == "observation" for f in answer.findings)


# ---------------------------------------------------------------------------
# Credentials and API errors
# ---------------------------------------------------------------------------

def test_missing_credentials_use_calculated_diagnostics_without_network(fx, monkeypatch) -> None:
    def fail(*_a, **_k):
        raise AssertionError("no client may be created without credentials")

    monkeypatch.setattr("tradeos.ask.analyst.default_client", fail)
    pkg = _package(fx, "session:sess-A", QUESTION_EXPLAIN)
    result = ask(pkg, NO_KEY)
    assert result.answer.mode == "calculated"
    assert result.error is None
    assert result.request_body is not None


def _request() -> httpx.Request:
    return httpx.Request("POST", "https://api.openai.com/v1/responses")


@pytest.mark.parametrize(
    "exc, expected",
    [
        (openai.AuthenticationError(f"Incorrect API key provided: {API_KEY}", response=httpx.Response(401, request=_request()), body=None), "authentication failed"),
        (openai.RateLimitError("slow down", response=httpx.Response(429, request=_request()), body=None), "rate limit"),
        (openai.NotFoundError("no model", response=httpx.Response(404, request=_request()), body=None), "was not found"),
        (openai.InternalServerError("boom", response=httpx.Response(500, request=_request()), body=None), "HTTP 500"),
        (openai.APITimeoutError(request=_request()), "timed out"),
        (openai.APIConnectionError(request=_request()), "network error"),
        (RuntimeError(f"unexpected {API_KEY}"), "RuntimeError"),
    ],
)
def test_api_errors_fall_back_and_never_echo_the_key(fx, exc: Exception, expected: str) -> None:
    pkg = _package(fx, "session:sess-A", QUESTION_EXPLAIN)
    result = ask(pkg, SETTINGS, client=FakeClient(exc=exc))
    assert result.answer.mode == "calculated"
    assert expected in (result.error or "")
    assert API_KEY not in (result.error or "")


def test_settings_precedence_and_defaults(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(dotenv, "dotenv_values", lambda *a, **k: {"OPENAI_API_KEY": "from-file", "ALPACA_API_SECRET": "x"})
    settings = load_ask_settings(environment={})
    assert settings.api_key == "from-file" and settings.api_key_source == ".env"
    assert settings.model == "gpt-6.1-sol"
    assert "from-file" not in settings.describe() and "from-file" not in repr(settings)
    settings = load_ask_settings(environment={"OPENAI_API_KEY": "from-env", "TRADEOS_ASK_MODEL": "other-model",
                                              "TRADEOS_ASK_TIMEOUT_SECONDS": "12"})
    assert (settings.api_key, settings.api_key_source, settings.model, settings.timeout_seconds) == (
        "from-env", "environment", "other-model", 12.0)
    monkeypatch.setattr(dotenv, "dotenv_values", lambda *a, **k: {})
    settings = load_ask_settings(environment={"OPENAI_API_KEY": "  "})
    assert settings.api_key is None and not settings.model_configured
    assert "OPENAI_API_KEY is not set" in settings.describe()


# ---------------------------------------------------------------------------
# Dashboard
# ---------------------------------------------------------------------------

def _tab_app(mode: str) -> None:
    import json as _json

    from dashboard_ask import render_ask_tradeos_tab
    from tradeos.ask.settings import AskSettings as _Settings

    class _Client:
        def create(self, **kwargs):
            if mode == "error":
                raise RuntimeError("down")
            return {"id": "resp_ui", "status": "completed", "output_text": _json.dumps({
                "answer": "Session sess-A logged one round trip.",
                "findings": [{"statement": "One BUY and one SELL were logged.", "kind": "observation", "citations": ["M1"]}],
                "missing_evidence": ["Win rate needs more trades."],
                "comparison_status": "not_applicable",
            })}

    if mode == "nokey":
        render_ask_tradeos_tab(settings_loader=lambda: _Settings(model="gpt-6.1-sol", timeout_seconds=5.0))
    else:
        render_ask_tradeos_tab(
            settings_loader=lambda: _Settings(model="gpt-6.1-sol", timeout_seconds=5.0, api_key="test-key-not-real-000000", api_key_source="test"),
            client=_Client(),
        )


@pytest.fixture()
def ui_env(fx, monkeypatch):
    monkeypatch.setenv("BOT_LOG_ROOT", str(fx["logs"]))
    monkeypatch.setenv("BOT_DB_PATH", str(fx["db"]))
    return fx


def _run_tab(mode: str):
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_function(_tab_app, kwargs={"mode": mode}, default_timeout=30)
    at.run()
    at.selectbox(key="ask_session").set_value("session:sess-A").run()
    return at


def test_dashboard_tab_renders_model_answer_with_evidence(ui_env) -> None:
    at = _run_tab("model")
    assert not at.exception
    at.button(key="ask_submit").click().run()
    assert not at.exception
    assert any("Model explanation · gpt-6.1-sol" in s.value for s in at.success)
    assert any("One BUY and one SELL" in m.value for m in at.markdown)
    labels = [e.label for e in at.expander]
    assert "Supporting evidence (1 cited)" in labels
    assert any(label.startswith("All evidence in the request") for label in labels)


def test_dashboard_tab_without_key_shows_calculated_diagnostics(ui_env) -> None:
    at = _run_tab("nokey")
    assert any("OPENAI_API_KEY" in i.value for i in at.info)
    assert at.button(key="ask_submit").label == "Show calculated diagnostics"
    at.radio(key="ask_question").set_value(QUESTION_COMPARE).run()
    at.button(key="ask_submit").click().run()
    assert not at.exception
    assert any("Calculated diagnostics" in i.value for i in at.info)
    assert any("Comparison unavailable" in w.value for w in at.warning)


def test_dashboard_tab_shows_error_state_and_fallback(ui_env) -> None:
    at = _run_tab("error")
    at.radio(key="ask_question").set_value(QUESTION_SYMBOL).run()
    at.selectbox(key="ask_symbol").set_value("AAA").run()
    at.button(key="ask_submit").click().run()
    assert not at.exception
    assert any("Model explanation unavailable" in e.value for e in at.error)
    assert any("Calculated diagnostics" in i.value for i in at.info)


def test_full_dashboard_exposes_ask_tab_even_when_state_fails(ui_env, tmp_path: Path, monkeypatch) -> None:
    from streamlit.testing.v1 import AppTest

    monkeypatch.setenv("BOT_DB_PATH", str(tmp_path / "missing.db"))
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    at = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "dashboard.py"), default_timeout=60)
    at.run()
    assert any(b.label == "Show calculated diagnostics" for b in at.button)


def test_dashboard_sources_have_no_double_encoded_text() -> None:
    """UTF-8 text re-saved as cp1252 shows up as 'â€”' etc. in the operator UI."""

    root = Path(__file__).resolve().parents[1]
    for name in ("dashboard.py", "dashboard_ask.py", *(str(p.relative_to(root)) for p in (root / "tradeos" / "ask").glob("*.py"))):
        text = (root / name).read_text(encoding="utf-8")
        for marker in ("â€", "â”", "â†", "â‰", "Ã"):
            assert marker not in text, f"double-encoded text {marker!r} in {name}"

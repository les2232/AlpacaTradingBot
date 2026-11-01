"""Assemble a structured, citable evidence package for one Ask TradeOS question.

All numbers are calculated here in Python.  The model only explains them.
Every fact the model may cite is an ``EvidenceItem`` with a stable ID and a
``source`` that points at a log line, startup artifact, database row, or the
Python function that produced the figure.

Isolation rules:
* A bound session includes only records whose ``session_id`` equals the
  selected session.  Records on the same log day without a session id are
  counted as ambiguous and excluded.  Records from other sessions are counted
  and excluded.
* An unbound log day includes only records without a session id on that day,
  is labelled as not attributable to a single session, and never borrows a
  configuration.
* A symbol question includes only records whose ``symbol`` equals the selected
  symbol (exact, case-insensitive) within the session scope.
* Today's ``config/live_config.json`` is never used as a historical session's
  configuration.  Configuration comes only from that session's own startup
  artifact.
"""

from __future__ import annotations

import json
import math
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, time, timedelta
from pathlib import Path
from statistics import median
from typing import Any, Callable, Iterable, Mapping

import pandas as pd

from tradeos.ask.records import (
    EXPECTED_STREAMS,
    STREAMS,
    LogRecord,
    SessionRef,
    StartupArtifact,
    _ET,
    is_sha256,
    load_day,
    load_db_sessions,
    load_startup_artifacts,
    parse_utc,
)

QUESTION_EXPLAIN = "explain_session"
QUESTION_SYMBOL = "symbol_decision"
QUESTION_COMPARE = "backtest_comparison"
QUESTIONS: dict[str, str] = {
    QUESTION_EXPLAIN: "Explain this session.",
    QUESTION_SYMBOL: "Why did this symbol buy, sell, or hold?",
    QUESTION_COMPARE: "Can this session validly be compared with this backtest?",
}
PACKAGE_VERSION = 1
MIN_CLOSED_TRADES_FOR_WIN_RATE = 5
PARTIAL_SESSION_COVERAGE = 0.9
ENTRY_WINDOW_START = time(9, 45)
ENTRY_WINDOW_END = time(15, 45)
MAX_SESSION_EVENTS = 60
MAX_SYMBOL_EVENTS = 80
_METRIC_SOURCES = {
    "signals": "signal_behavior",
    "execution": "execution",
    "risk": "risk",
    "positions": "positions",
    "bars": "bar_quality",
}

_CONFIG_FIELDS: tuple[str, ...] = (
    "launch_mode",
    "execution_enabled",
    "paper",
    "strategy_mode",
    "bar_timeframe_minutes",
    "sma_bars",
    "symbols",
    "symbol_count",
    "runtime_config_approved",
    "runtime_config_rejection_reasons",
    "baseline_valid_for_comparison",
    "baseline_validation_errors",
    "canonical_strategy_sha256",
    "promotion_bundle_sha256",
    "research_generation_sha256",
    "max_usd_per_trade",
    "max_open_positions",
    "max_daily_loss_usd",
    "mean_reversion_exit_style",
    "mean_reversion_max_atr_percentile",
    "mean_reversion_trend_filter",
    "mean_reversion_trend_slope_filter",
    "mean_reversion_stop_pct",
    "sma_stop_pct",
    "historical_feed",
    "live_feed",
    "bar_build_mode",
)

# Allowlisted event fields sent to the model.  Order ids, account fingerprints,
# balances, file paths, and anything not listed are dropped.
_EVENT_FIELDS: dict[str, tuple[str, ...]] = {
    "signal.evaluated": (
        "symbol", "decision_ts", "action", "rejection", "holding", "bar_close", "sma",
        "deviation_pct", "trend_sma", "above_trend_sma", "atr_pct", "atr_percentile",
        "volume_ratio", "ml_prob", "trend_filter", "atr_filter", "window_open",
    ),
    "risk.check": (
        "symbol", "action", "allowed", "block_reason", "open_positions", "max_positions",
        "daily_pnl", "kill_switch_pct_consumed", "price_deviation_bps", "live_price_age_s",
        "in_entry_window", "detail",
    ),
    "order.submitted": ("symbol", "side", "qty", "live_price", "signal_bar_close", "notional"),
    "order.filled": ("symbol", "side", "fill_price", "fill_qty", "signal_bar_close", "slippage_bps", "submit_to_fill_ms"),
    "order.partial_fill": ("symbol", "fill_price", "filled_qty", "requested_qty", "fill_rate"),
    "order.rejected": ("symbol", "side", "reason"),
    "position.opened": ("symbol", "entry_price", "qty", "notional", "strategy_mode", "entry_branch"),
    "position.closed": (
        "symbol", "entry_price", "exit_price", "qty", "pnl_usd", "pnl_pct", "winner",
        "holding_bars", "holding_minutes", "exit_reason", "mae_pct", "mfe_pct",
    ),
    "kill_switch.triggered": ("daily_pnl", "daily_limit", "pct_consumed", "reason"),
    "execution.error": ("symbol", "action", "error"),
    "cycle.summary": (
        "decision_ts", "execute_orders", "processed_bar", "skip_reason", "buy_signals",
        "sell_signals", "hold_signals", "error_signals", "orders_submitted",
    ),
    "bar.timestamp_rejected": ("symbol", "decision_ts", "rejection_code", "timestamp_type", "age_seconds"),
    "process.lifecycle": ("stage", "result", "ready", "blocking_reasons", "execute_orders", "market_open", "feed_status", "status", "reason_codes"),
}


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class EvidenceItem:
    id: str
    kind: str  # metric | event | coverage | configuration | comparison | check | limitation
    title: str
    data: Mapping[str, Any]
    source: str

    def to_payload(self) -> dict[str, Any]:
        return {"id": self.id, "kind": self.kind, "title": self.title, "data": dict(self.data), "source": self.source}


@dataclass
class EvidencePackage:
    question_type: str
    question_text: str
    symbol: str | None
    session: dict[str, Any]
    configuration: dict[str, Any]
    metrics: dict[str, Any]
    data_coverage: dict[str, Any]
    comparison: dict[str, Any]
    limitations: list[str]
    items: list[EvidenceItem] = field(default_factory=list)

    @property
    def item_ids(self) -> set[str]:
        return {item.id for item in self.items}

    def item(self, item_id: str) -> EvidenceItem | None:
        return next((item for item in self.items if item.id == item_id), None)

    @property
    def comparison_status(self) -> str:
        return str(self.comparison.get("status", "unavailable"))

    def to_model_payload(self) -> dict[str, Any]:
        """The exact, JSON-safe structure sent to the model."""

        return _json_safe(
            {
                "package_version": PACKAGE_VERSION,
                "question": {"type": self.question_type, "text": self.question_text, "symbol": self.symbol},
                "session": self.session,
                "configuration": self.configuration,
                "metrics": self.metrics,
                "data_coverage": self.data_coverage,
                "comparison": self.comparison,
                "limitations": self.limitations,
                "evidence": [item.to_payload() for item in self.items],
            }
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(v) for v in value]
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return value
    if isinstance(value, int):
        return int(value)
    if isinstance(value, float):
        return None if not math.isfinite(value) else round(value, 6)
    if hasattr(value, "item"):  # numpy scalar
        try:
            return _json_safe(value.item())
        except (TypeError, ValueError):
            return str(value)
    return str(value)


def _clip_text(value: Any, limit: int = 240) -> Any:
    if isinstance(value, str) and len(value) > limit:
        return value[: limit - 1] + "…"
    return value


def _event_fields(record: LogRecord) -> dict[str, Any]:
    allowed = _EVENT_FIELDS.get(record.event, ("symbol",))
    data = {key: _clip_text(record.payload[key]) for key in allowed if key in record.payload}
    data["event"] = record.event
    if record.payload.get("ts"):
        data["ts"] = record.payload.get("ts")
    return _json_safe(data)


def _fingerprint(payload: Mapping[str, Any]) -> str:
    return json.dumps(payload, sort_keys=True, default=str, separators=(",", ":"))


def _dedupe(records: Iterable[LogRecord]) -> tuple[list[LogRecord], int]:
    seen: set[tuple[str, str]] = set()
    kept: list[LogRecord] = []
    duplicates = 0
    for record in records:
        key = (record.stream, _fingerprint(record.payload))
        if key in seen:
            duplicates += 1
            continue
        seen.add(key)
        kept.append(record)
    return kept, duplicates


def _frame(records: Iterable[LogRecord]) -> pd.DataFrame:
    rows = [dict(record.payload) for record in records]
    return pd.DataFrame(rows) if rows else pd.DataFrame()


def _fmt_et(value: str | None) -> str | None:
    parsed = parse_utc(value)
    if parsed is None:
        return None
    return parsed.astimezone(_ET).strftime("%Y-%m-%d %H:%M ET")


def _reason_label(code: Any, table: str) -> str:
    try:
        import avatar

        labels = getattr(avatar, table, {})
    except Exception:  # pragma: no cover - avatar is optional for labels
        labels = {}
    text = str(code) if code not in (None, "") else "none"
    label = labels.get(text)
    return f"{text} ({label.rstrip('.')})" if label else text


class _Ids:
    def __init__(self) -> None:
        self._counts: Counter[str] = Counter()

    def next(self, prefix: str) -> str:
        self._counts[prefix] += 1
        return f"{prefix}{self._counts[prefix]}"


# ---------------------------------------------------------------------------
# Configuration identity
# ---------------------------------------------------------------------------

def _resolve_configuration(
    session: SessionRef, artifacts: list[StartupArtifact]
) -> tuple[dict[str, Any], StartupArtifact | None]:
    note = (
        "The current config/live_config.json is not used for historical sessions; only "
        "the session's own startup artifact can establish its configuration."
    )
    if session.kind != "bound":
        return (
            {
                "status": "not_attributable",
                "detail": "Records without a session id cannot be bound to any launch configuration.",
                "note": note,
                "source": None,
            },
            None,
        )
    mine = [a for a in artifacts if a.session_id == session.session_id]
    valid = [a for a in mine if a.valid]
    if not mine:
        return (
            {
                "status": "not_recorded",
                "detail": "No startup artifact was found for this session id.",
                "note": note,
                "source": None,
            },
            None,
        )
    if not valid:
        errors = sorted({e for a in mine for e in a.errors})
        return (
            {
                "status": "invalid",
                "detail": "Startup artifact(s) for this session failed validation: " + "; ".join(errors),
                "note": note,
                "source": ", ".join(a.path for a in mine),
            },
            None,
        )
    distinct = {_fingerprint({k: a.payload.get(k) for k in _CONFIG_FIELDS}): a for a in valid}
    if len(distinct) > 1:
        return (
            {
                "status": "conflicting",
                "detail": f"{len(distinct)} startup artifacts for this session disagree; configuration is ambiguous.",
                "note": note,
                "source": ", ".join(a.path for a in valid),
            },
            None,
        )
    chosen = valid[0]
    values = {k: chosen.payload.get(k) for k in _CONFIG_FIELDS if k in chosen.payload}
    values["environment"] = "paper" if chosen.payload.get("paper") is True else "live"
    values["started_at_utc"] = chosen.payload.get("started_at_utc")
    return (
        {
            "status": "recorded_at_launch",
            "detail": "Configuration recorded by this session's own startup artifact.",
            "note": note,
            "source": chosen.path,
            "values": _json_safe(values),
        },
        chosen,
    )


# ---------------------------------------------------------------------------
# Coverage / partial sessions
# ---------------------------------------------------------------------------

def _decision_slots(timeframe: int) -> int:
    start = datetime.combine(datetime(2000, 1, 3).date(), ENTRY_WINDOW_START)
    end = datetime.combine(start.date(), ENTRY_WINDOW_END)
    return int((end - start) / timedelta(minutes=timeframe)) + 1


def _decision_coverage(signal_records: list[LogRecord], timeframe: int | None, timeframe_source: str) -> dict[str, Any]:
    decisions: dict[str, set[datetime]] = {}
    for record in signal_records:
        parsed = parse_utc(record.payload.get("decision_ts"))
        if parsed is None:
            continue
        local = parsed.astimezone(_ET)
        if ENTRY_WINDOW_START <= local.time() <= ENTRY_WINDOW_END:
            decisions.setdefault(local.date().isoformat(), set()).add(parsed)
    all_points = sorted(p for points in decisions.values() for p in points)
    inferred = None
    if timeframe is None and len(all_points) >= 3:
        gaps = [(b - a).total_seconds() / 60 for a, b in zip(all_points, all_points[1:]) if b > a]
        gaps = [g for g in gaps if g <= 60]
        if gaps:
            inferred = int(round(median(gaps)))
            timeframe, timeframe_source = inferred, "inferred from decision spacing"
    distinct = sum(len(points) for points in decisions.values())
    per_day = []
    expected_total = None
    if timeframe and timeframe > 0:
        slots = _decision_slots(timeframe)
        expected_total = slots * max(len(decisions), 1)
        for day, points in sorted(decisions.items()):
            ordered = sorted(points)
            per_day.append(
                {
                    "day": day,
                    "distinct_decisions": len(points),
                    "expected_decisions": slots,
                    "first_decision": ordered[0].astimezone(_ET).strftime("%H:%M ET"),
                    "last_decision": ordered[-1].astimezone(_ET).strftime("%H:%M ET"),
                }
            )
    coverage = (distinct / expected_total) if expected_total else None
    return {
        "distinct_decisions_in_entry_window": distinct,
        "expected_decisions": expected_total,
        "coverage_ratio": round(coverage, 4) if coverage is not None else None,
        "partial_session": (coverage is not None and coverage < PARTIAL_SESSION_COVERAGE) or distinct == 0,
        "timeframe_minutes": timeframe,
        "timeframe_source": timeframe_source if timeframe else "unknown",
        "entry_window": "09:45-15:45 ET",
        "per_day": per_day,
    }


# ---------------------------------------------------------------------------
# Metrics (delegating to daily_report where an implementation exists)
# ---------------------------------------------------------------------------

def _session_metrics(records: list[LogRecord]) -> dict[str, Any]:
    import daily_report

    by_stream = {stream: [r for r in records if r.stream == stream] for stream in STREAMS}
    signals = daily_report.compute_signal_behavior(_frame(by_stream["signals"]))
    execution = daily_report.compute_execution(_frame(by_stream["execution"]))
    risk = daily_report.compute_risk(_frame(by_stream["risk"]))
    positions = daily_report.compute_positions(_frame(by_stream["positions"]))
    bars = daily_report.compute_bar_quality(_frame(by_stream["bars"]))

    closes = [r for r in by_stream["positions"] if r.event == "position.closed"]
    unique_close_traces = {str(r.payload.get("trace")) for r in closes if r.payload.get("trace")}
    repeated_evals = Counter(
        (r.symbol, r.payload.get("decision_ts"))
        for r in by_stream["signals"]
        if r.event == "signal.evaluated"
    )
    repeated = sum(1 for count in repeated_evals.values() if count > 1)
    closed = int(positions.get("closed") or 0)
    unique_closed = len(unique_close_traces) if unique_close_traces else closed
    win_rate_sufficient = unique_closed >= MIN_CLOSED_TRADES_FOR_WIN_RATE
    fills = [r for r in by_stream["execution"] if r.event == "order.filled"]
    fill_orders = {str(r.payload.get("order_id")) for r in fills if r.payload.get("order_id")}
    return _json_safe(
        {
            "signals": {
                "evaluations": signals["total_evaluated"],
                "buy": signals["buy_signals"],
                "sell": signals["sell_signals"],
                "hold": signals["hold_count"],
                "hold_rate": signals["rejection_rate"],
                "hold_reasons": signals["rejections"],
                "buy_signals_by_symbol": signals["by_symbol"],
                "symbol_decisions_evaluated_more_than_once": repeated,
            },
            "execution": {
                "orders_submitted_buy": execution["submitted_buys"],
                "orders_submitted_sell": execution["submitted_sells"],
                "avg_slippage_bps": execution["avg_slippage_bps"],
                "worst_slippage_bps": execution["worst_slippage_bps"],
                "partial_fills": execution["partial_fills"],
                "avg_submit_to_fill_ms": execution["avg_latency_ms"],
                "order_rejections": sum(1 for r in by_stream["execution"] if r.event == "order.rejected"),
                "fill_events": len(fills),
                "fill_events_unique_orders": len(fill_orders) if fill_orders else None,
                "possible_duplicate_fill_events": max(len(fills) - len(fill_orders), 0) if fill_orders else 0,
            },
            "risk": {
                "kill_switch_triggered": risk["kill_switch_triggered"],
                "buy_blocks_by_reason": risk["buy_blocks_by_reason"],
                "sell_blocks_by_reason": risk["sell_blocks_by_reason"],
                "execution_errors": sum(1 for r in by_stream["risk"] if r.event == "execution.error"),
            },
            "positions": {
                "opened": positions["opened"],
                "close_events": closed,
                "closed_unique_traces": len(unique_close_traces),
                "possible_duplicate_close_events": max(closed - len(unique_close_traces), 0) if unique_close_traces else 0,
                "total_pnl_usd": positions["total_pnl_usd"],
                "avg_pnl_usd": positions["avg_pnl_usd"],
                "win_rate": positions["win_rate"],
                "win_rate_sample_sufficient": win_rate_sufficient,
                "min_closed_trades_for_win_rate": MIN_CLOSED_TRADES_FOR_WIN_RATE,
                "avg_hold_bars": positions["avg_hold_bars"],
                "exit_reasons": positions["exit_reasons"],
            },
            "bars": {
                "received": bars["evaluated"],
                "stale_count": bars["stale_count"],
                "avg_age_s": bars["avg_age_s"],
            },
            "calculated_by": "daily_report.compute_* on session-scoped records",
        }
    )


def _symbol_metrics(records: list[LogRecord], symbol: str) -> dict[str, Any]:
    signals = [r for r in records if r.event == "signal.evaluated"]
    actions = Counter(str(r.payload.get("action", "")).upper() or "UNKNOWN" for r in signals)
    rejections = Counter(str(r.payload.get("rejection")) for r in signals if r.payload.get("rejection"))
    trend = Counter(str(r.payload.get("trend_filter")) for r in signals if r.payload.get("trend_filter"))
    atr = Counter(str(r.payload.get("atr_filter")) for r in signals if r.payload.get("atr_filter"))
    window_closed = sum(1 for r in signals if r.payload.get("window_open") is False)
    checks = [r for r in records if r.event == "risk.check"]
    blocks = Counter(
        f"{r.payload.get('action')}:{r.payload.get('block_reason')}"
        for r in checks
        if r.payload.get("allowed") is False
    )
    allowed = Counter(str(r.payload.get("action")) for r in checks if r.payload.get("allowed") is True)
    submitted = Counter(str(r.payload.get("side")) for r in records if r.event == "order.submitted")
    fills = [r for r in records if r.event == "order.filled"]
    fill_orders = {str(r.payload.get("order_id")) for r in fills if r.payload.get("order_id")}
    closes = [r for r in records if r.event == "position.closed"]
    close_traces = {str(r.payload.get("trace")) for r in closes if r.payload.get("trace")}
    pnl_values = [float(r.payload["pnl_usd"]) for r in closes if isinstance(r.payload.get("pnl_usd"), (int, float))]
    ordered = sorted(signals, key=lambda r: str(r.payload.get("decision_ts", "")))
    deviations = [r.payload.get("deviation_pct") for r in signals if isinstance(r.payload.get("deviation_pct"), (int, float))]
    return _json_safe(
        {
            "symbol": symbol,
            "evaluations": len(signals),
            "actions": dict(actions),
            "hold_reasons": dict(rejections),
            "trend_filter": dict(trend),
            "atr_filter": dict(atr),
            "evaluations_with_entry_window_closed": window_closed,
            "risk_checks_allowed_by_action": dict(allowed),
            "risk_blocks_by_action_and_reason": dict(blocks),
            "orders_submitted_by_side": dict(submitted),
            "fill_events": len(fills),
            "fill_events_unique_orders": len(fill_orders) if fill_orders else None,
            "positions_opened": sum(1 for r in records if r.event == "position.opened"),
            "position_close_events": len(closes),
            "position_closes_unique_traces": len(close_traces) if close_traces else None,
            "realized_pnl_usd": round(sum(pnl_values), 4) if pnl_values else None,
            "winning_closes": sum(1 for r in closes if r.payload.get("winner") is True),
            "deviation_pct_min": min(deviations) if deviations else None,
            "deviation_pct_max": max(deviations) if deviations else None,
            "first_decision_ts": ordered[0].payload.get("decision_ts") if ordered else None,
            "last_decision_ts": ordered[-1].payload.get("decision_ts") if ordered else None,
            "last_action": str(ordered[-1].payload.get("action")) if ordered else None,
            "last_rejection": ordered[-1].payload.get("rejection") if ordered else None,
            "calculated_by": "tradeos.ask.evidence._symbol_metrics",
        }
    )


# ---------------------------------------------------------------------------
# Event selection
# ---------------------------------------------------------------------------

def _event_title(record: LogRecord) -> str:
    p = record.payload
    sym = record.symbol or ""
    when = _fmt_et(p.get("decision_ts")) or _fmt_et(p.get("ts")) or "unknown time"
    event = record.event
    if event == "signal.evaluated":
        action = str(p.get("action", "?")).upper()
        reason = f"; hold reason {_reason_label(p.get('rejection'), '_REJECTION_LABELS')}" if action == "HOLD" else ""
        dev = p.get("deviation_pct")
        dev_text = f"; deviation {dev}% vs SMA" if isinstance(dev, (int, float)) else ""
        return f"{sym} signal {action} for decision {when}{reason}{dev_text}"
    if event == "risk.check":
        verdict = "allowed" if p.get("allowed") else f"blocked by {_reason_label(p.get('block_reason'), '_BLOCK_REASON_LABELS')}"
        return f"{sym} risk check for {p.get('action')} {verdict} at {when}"
    if event == "order.submitted":
        return f"{sym} {p.get('side')} order submitted for {p.get('qty')} shares at {when}"
    if event == "order.filled":
        return f"{sym} {p.get('side')} order filled at {p.get('fill_price')} (slippage {p.get('slippage_bps')} bps) at {when}"
    if event == "position.closed":
        return (
            f"{sym} position closed via {_reason_label(p.get('exit_reason'), '_EXIT_REASON_LABELS')}, "
            f"P&L {p.get('pnl_usd')} USD at {when}"
        )
    if event == "position.opened":
        return f"{sym} position opened at {p.get('entry_price')} at {when}"
    if event == "kill_switch.triggered":
        return f"Kill switch triggered (daily P&L {p.get('daily_pnl')} vs limit {p.get('daily_limit')}) at {when}"
    if event == "cycle.summary":
        return f"Cycle {when}: processed={p.get('processed_bar')} skip={p.get('skip_reason')} orders={p.get('orders_submitted')}"
    return f"{event} {sym} at {when}".strip()


def _select_session_events(records: list[LogRecord]) -> tuple[list[LogRecord], int]:
    priority = (
        "kill_switch.triggered", "execution.error", "order.rejected", "order.submitted",
        "order.filled", "order.partial_fill", "position.opened", "position.closed",
        "bar.timestamp_rejected",
    )
    picked = [r for r in records if r.event in priority]
    lifecycle = [r for r in records if r.event == "process.lifecycle"]
    picked.extend(lifecycle[:10])
    picked.sort(key=lambda r: (r.payload.get("ts") or "", r.source))
    total = len(picked)
    return picked[:MAX_SESSION_EVENTS], max(total - MAX_SESSION_EVENTS, 0)


def _select_symbol_events(records: list[LogRecord]) -> tuple[list[LogRecord], int]:
    signals = sorted(
        (r for r in records if r.event == "signal.evaluated"),
        key=lambda r: (str(r.payload.get("decision_ts", "")), r.source),
    )
    others = [r for r in records if r.event != "signal.evaluated" and r.event in _EVENT_FIELDS]
    if len(signals) <= 40:
        chosen_signals = signals
    else:
        chosen_signals = []
        previous_key = None
        for index, record in enumerate(signals):
            action = str(record.payload.get("action", "")).upper()
            key = (action, record.payload.get("rejection"))
            if action != "HOLD" or key != previous_key or index >= len(signals) - 5:
                chosen_signals.append(record)
            previous_key = key
    picked = sorted(chosen_signals + others, key=lambda r: (r.payload.get("ts") or "", r.source))
    total_available = len(signals) + len(others)
    if len(picked) > MAX_SYMBOL_EVENTS:
        non_hold = [r for r in picked if not (r.event == "signal.evaluated" and str(r.payload.get("action", "")).upper() == "HOLD")]
        holds = [r for r in picked if r not in non_hold]
        room = max(MAX_SYMBOL_EVENTS - len(non_hold), 0)
        picked = sorted(non_hold[:MAX_SYMBOL_EVENTS] + holds[-room:] if room else non_hold[:MAX_SYMBOL_EVENTS],
                        key=lambda r: (r.payload.get("ts") or "", r.source))
    return picked, max(total_available - len(picked), 0)


# ---------------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------------

BaselineLoader = Callable[[Path], tuple[dict[str, Any], dict[str, Any]]]


def build_evidence_package(
    session: SessionRef,
    question_type: str,
    *,
    log_root: Path,
    repo_root: Path,
    db_path: Path | None = None,
    symbol: str | None = None,
    comparison_target: str = "promoted_baseline",
    baseline_loader: BaselineLoader | None = None,
    drift_profile_loader: Callable[[Path], Any] | None = None,
    experiment_log_path: Path | None = None,
) -> EvidencePackage:
    from tradeos.ask import comparison as comparison_module

    if question_type not in QUESTIONS:
        raise ValueError(f"unsupported question type: {question_type}")
    if question_type == QUESTION_SYMBOL:
        symbol = (symbol or "").strip().upper()
        if not symbol:
            raise ValueError("a symbol is required for the symbol question")
    else:
        symbol = None

    ids = _Ids()
    items: list[EvidenceItem] = []
    limitations: list[str] = []

    def limit(text: str, source: str = "computed:tradeos.ask.evidence") -> None:
        limitations.append(text)
        items.append(EvidenceItem(ids.next("L"), "limitation", text, {}, source))

    # ---- scope the records -------------------------------------------------
    in_scope: list[LogRecord] = []
    excluded_ambiguous: Counter[str] = Counter()
    excluded_other: Counter[str] = Counter()
    file_stats: dict[str, dict[str, Any]] = {s: {"files_present": 0, "files_missing": 0, "malformed_lines": 0, "paths": []} for s in STREAMS}
    for day in session.days:
        day_log = load_day(log_root, day)
        for f in day_log.files:
            stats = file_stats[f.stream]
            stats["files_present" if f.present else "files_missing"] += 1
            stats["malformed_lines"] += f.malformed_lines
            if f.present:
                stats["paths"].append(f.path)
        for record in day_log.records:
            sid = record.session_id
            if session.kind == "bound":
                if sid == session.session_id:
                    in_scope.append(record)
                elif sid is None:
                    excluded_ambiguous[record.stream] += 1
                else:
                    excluded_other[record.stream] += 1
            else:
                if sid is None:
                    in_scope.append(record)
                else:
                    excluded_other[record.stream] += 1
    in_scope, duplicates_removed = _dedupe(in_scope)
    duplicate_by_stream = duplicates_removed  # total; per-stream detail not needed
    scoped = [r for r in in_scope if r.symbol == symbol] if symbol else in_scope

    # ---- configuration ------------------------------------------------------
    artifacts = load_startup_artifacts(log_root)
    configuration, artifact = _resolve_configuration(session, artifacts)
    cfg_id = ids.next("CFG")
    items.append(
        EvidenceItem(
            cfg_id,
            "configuration",
            f"Session configuration status: {configuration['status']}",
            {k: v for k, v in configuration.items() if k != "source"},  # copy
            configuration.get("source") or "computed:tradeos.ask.evidence._resolve_configuration",
        )
    )
    configuration["evidence_id"] = cfg_id
    cfg_values = configuration.get("values") or {}

    # ---- environment identifiers -------------------------------------------
    db_envs: tuple[Any, ...] = ()
    db_runs = 0
    if session.kind == "bound":
        for summary in load_db_sessions(db_path):
            if summary.session_id == session.session_id:
                db_envs, db_runs = summary.environments, summary.run_count
    environment = {
        "recorded_at_launch": cfg_values.get("environment"),
        "database_environments": list(db_envs),
        "database_runs": db_runs,
        "database_runs_missing_environment": None in db_envs,
        "consistent": None,
    }
    known_db_envs = {str(e).strip().lower() for e in db_envs if isinstance(e, str) and e.strip()}
    if environment["recorded_at_launch"] and known_db_envs:
        environment["consistent"] = known_db_envs == {environment["recorded_at_launch"]}
    elif known_db_envs and len(known_db_envs) > 1:
        environment["consistent"] = False

    # ---- session descriptor -------------------------------------------------
    stamps = sorted(p for r in in_scope if (p := r.ts) is not None)
    observed_symbols = sorted({r.symbol for r in in_scope if r.symbol})
    session_info = _json_safe(
        {
            "key": session.key,
            "kind": session.kind,
            "session_id": session.session_id,
            "run_type": session.run_type if session.kind == "bound" else "unknown (no session id)",
            "run_type_source": "session id prefix assigned by the tradeos CLI" if session.kind == "bound" else None,
            "scope": (
                "single session (records carrying this session id only)"
                if session.kind == "bound"
                else "entire log day without session ids; may combine several processes and is not a single session"
            ),
            "log_days": list(session.days),
            "time_range_utc": {
                "start": stamps[0].isoformat() if stamps else None,
                "end": stamps[-1].isoformat() if stamps else None,
            },
            "time_range_et": {
                "start": _fmt_et(stamps[0].isoformat()) if stamps else None,
                "end": _fmt_et(stamps[-1].isoformat()) if stamps else None,
            },
            "symbols_observed": observed_symbols,
            "symbols_configured": cfg_values.get("symbols"),
            "environment": environment,
        }
    )

    # ---- coverage -----------------------------------------------------------
    stream_counts = Counter(r.stream for r in in_scope)
    streams = {}
    for stream in STREAMS:
        stats = file_stats[stream]
        streams[stream] = {
            "records_in_scope": stream_counts.get(stream, 0),
            "excluded_without_session_id": excluded_ambiguous.get(stream, 0),
            "excluded_other_sessions": excluded_other.get(stream, 0),
            "malformed_lines_in_files": stats["malformed_lines"],
            "files_present": stats["files_present"],
            "files_missing": stats["files_missing"],
        }
    timeframe = cfg_values.get("bar_timeframe_minutes") if isinstance(cfg_values.get("bar_timeframe_minutes"), int) else None
    decisions = _decision_coverage(
        [r for r in in_scope if r.stream == "signals" and r.event == "signal.evaluated"],
        timeframe,
        "startup artifact" if timeframe else "unknown",
    )
    coverage = _json_safe(
        {
            "streams": streams,
            "exact_duplicate_records_removed": duplicate_by_stream,
            "decisions": decisions,
            "database_runs_for_session": db_runs,
        }
    )
    cov_id = ids.next("C")
    items.append(
        EvidenceItem(
            cov_id,
            "coverage",
            "Data coverage and exclusions for the selected scope",
            dict(coverage),
            "computed:tradeos.ask.evidence (files: "
            + ", ".join(sorted({p for s in file_stats.values() for p in s["paths"]})[:8])
            + ")",
        )
    )
    coverage["evidence_id"] = cov_id

    # ---- limitations --------------------------------------------------------
    if session.kind != "bound":
        limit(
            "Records on this log day carry no session id, so they cannot be attributed to one session or to a launch "
            "configuration. Figures describe the whole log day."
        )
    if session.kind == "bound" and session.run_type.endswith("no trading"):
        limit(
            f"This session id belongs to a {session.run_type} run, not a trading session; trading decisions, "
            "orders, and fills are not expected in it."
        )
    total_ambiguous = sum(excluded_ambiguous.values())
    if total_ambiguous:
        limit(f"{total_ambiguous} record(s) on the session's log day(s) lack a session id and were excluded from session conclusions.")
    if sum(excluded_other.values()):
        limit(f"{sum(excluded_other.values())} record(s) belonging to other sessions were excluded.")
    if configuration["status"] != "recorded_at_launch":
        limit(f"Session configuration cannot be established ({configuration['status']}): {configuration['detail']}")
    missing_streams = [s for s in EXPECTED_STREAMS if file_stats[s]["files_present"] == 0]
    if missing_streams:
        limit("Missing log stream(s): " + ", ".join(missing_streams) + ". Conclusions depending on them are unavailable.")
    if not in_scope:
        limit("No log records were found for this scope.")
    malformed = sum(s["malformed_lines"] for s in file_stats.values())
    if malformed:
        limit(f"{malformed} malformed log line(s) could not be parsed.")
    if duplicates_removed:
        limit(f"{duplicates_removed} exact duplicate record(s) were removed before calculation.")
    if decisions["partial_session"]:
        limit(
            "Partial session: "
            f"{decisions['distinct_decisions_in_entry_window']} distinct decisions observed versus "
            f"{decisions['expected_decisions'] if decisions['expected_decisions'] is not None else 'an unknown number'} expected "
            f"in the 09:45-15:45 ET window (timeframe {decisions['timeframe_source']})."
        )
    if environment["database_runs_missing_environment"]:
        limit("Some persisted database runs for this session have no environment identifier.")
    if environment["consistent"] is False:
        limit("Environment identifiers for this session disagree (paper vs live); environment cannot be established.")
    if session.kind == "bound" and db_path is None:
        limit("No runtime history database was available; persisted account snapshots were not checked.")

    # ---- metrics ------------------------------------------------------------
    metrics: dict[str, Any] = {}
    session_metrics = _session_metrics(in_scope)
    if question_type == QUESTION_SYMBOL:
        assert symbol is not None
        metrics["symbol"] = _symbol_metrics(scoped, symbol)
        if not scoped:
            limit(f"No records for {symbol} exist in this scope; its decisions cannot be explained.")
        elif metrics["symbol"]["evaluations"] == 0:
            limit(f"No signal evaluations for {symbol} were logged in this scope.")
        if cfg_values.get("symbols") and symbol not in [str(s).upper() for s in cfg_values["symbols"]]:
            limit(f"{symbol} is not in the session's configured symbol list.")
        mid = ids.next("M")
        items.append(EvidenceItem(mid, "metric", f"{symbol} decision counts", dict(metrics["symbol"]), "computed:tradeos.ask.evidence._symbol_metrics"))
        metrics["symbol"]["evidence_id"] = mid
    else:
        metrics["session"] = session_metrics
        for section in ("signals", "execution", "risk", "positions", "bars"):
            mid = ids.next("M")
            items.append(
                EvidenceItem(
                    mid,
                    "metric",
                    f"Session {section} metrics",
                    dict(session_metrics[section]),
                    f"computed:daily_report.compute_{_METRIC_SOURCES[section]}",
                )
            )
            session_metrics[section]["evidence_id"] = mid
        pos = session_metrics["positions"]
        if not pos["win_rate_sample_sufficient"]:
            limit(
                f"Only {pos['closed_unique_traces'] or pos['close_events']} closed trade(s); win rate needs at least "
                f"{MIN_CLOSED_TRADES_FOR_WIN_RATE} to be interpreted."
            )
        if pos["possible_duplicate_close_events"]:
            limit(f"{pos['possible_duplicate_close_events']} position-close event(s) repeat an earlier trace; P&L totals may double count.")
        if session_metrics["execution"]["possible_duplicate_fill_events"]:
            limit(
                f"{session_metrics['execution']['possible_duplicate_fill_events']} fill event(s) repeat an earlier "
                "broker order; fill counts may double count."
            )
        if session_metrics["signals"]["symbol_decisions_evaluated_more_than_once"]:
            limit(
                f"{session_metrics['signals']['symbol_decisions_evaluated_more_than_once']} symbol/decision pair(s) were "
                "evaluated more than once (restarts or overlapping processes)."
            )

    # ---- comparison ---------------------------------------------------------
    comparison = comparison_module.evaluate_comparison(
        session=session,
        configuration=configuration,
        artifact=artifact,
        environment=environment,
        decisions=decisions,
        session_metrics=session_metrics,
        target=comparison_target,
        repo_root=repo_root,
        in_scope_records=in_scope,
        baseline_loader=baseline_loader,
        drift_profile_loader=drift_profile_loader,
        experiment_log_path=experiment_log_path,
    )
    cmp_id = ids.next("CMP")
    comparison_payload = comparison.to_payload(include_details=question_type == QUESTION_COMPARE)
    items.append(
        EvidenceItem(
            cmp_id,
            "comparison",
            f"Backtest comparison {comparison.status}" + (f": {'; '.join(comparison.reasons[:3])}" if comparison.reasons else ""),
            dict(comparison_payload),
            "computed:tradeos.ask.comparison.evaluate_comparison",
        )
    )
    comparison_payload["evidence_id"] = cmp_id
    if question_type == QUESTION_COMPARE:
        for check in comparison.checks:
            items.append(
                EvidenceItem(
                    ids.next("CHK"),
                    "check",
                    f"{'PASS' if check.passed else 'FAIL'}: {check.label}",
                    {"check": check.key, "passed": check.passed, "detail": check.detail},
                    check.source,
                )
            )
        for metric in comparison.metric_comparisons:
            items.append(
                EvidenceItem(
                    ids.next("CM"),
                    "comparison",
                    f"{metric['metric']}: observed {metric['observed']} vs expected {metric['expected']}",
                    metric,
                    metric.get("source", "computed:tradeos.ask.comparison"),
                )
            )

    # ---- events ------------------------------------------------------------
    if question_type == QUESTION_SYMBOL:
        events, omitted = _select_symbol_events(scoped)
    elif question_type == QUESTION_EXPLAIN:
        events, omitted = _select_session_events(in_scope)
    else:
        events, omitted = [], 0
    for record in events:
        items.append(EvidenceItem(ids.next("E"), "event", _event_title(record), _event_fields(record), record.source))
    if omitted:
        limit(f"{omitted} lower-priority event(s) were summarised in metrics but not listed individually.")

    return EvidencePackage(
        question_type=question_type,
        question_text=QUESTIONS[question_type],
        symbol=symbol,
        session=session_info,
        configuration=_json_safe(configuration),
        metrics=_json_safe(metrics),
        data_coverage=coverage,
        comparison=comparison_payload,
        limitations=limitations,
        items=items,
    )

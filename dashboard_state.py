from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from pathlib import Path
from types import SimpleNamespace

from avatar import generate_near_miss_narration, generate_trade_narration
from drift_monitor import BaselineProfile, DriftAlert, DriftReport, LiveMetrics, build_drift_report
from storage import BotStorage
from symbol_state import format_symbol_list, normalize_symbols, symbols_match
from tradeos.runtime_paths import (
    DATABASE_PATH_ENV,
    resolve_database_path,
    resolve_explicit_database_path,
    resolve_log_root,
)


_UTC_MIN = datetime.min.replace(tzinfo=timezone.utc)
_ACCOUNT_FINGERPRINT_RE = re.compile(r"^[0-9a-f]{64}$")


@dataclass(frozen=True)
class PersistedSymbolSnapshot:
    symbol: str
    price: float | None
    sma: float | None
    action: str
    holding: bool
    quantity: float
    market_value: float
    ml_probability_up: float | None = None
    ml_confidence: float | None = None
    ml_training_rows: int | None = None
    ml_buy_threshold: float | None = None
    ml_sell_threshold: float | None = None
    ml_model_name: str | None = None
    holding_minutes: float | None = None
    error: str | None = None


@dataclass(frozen=True)
class PersistedOrderSnapshot:
    order_id: str
    observed_at_utc: str | None
    submitted_at: str | None
    filled_at: str | None
    symbol: str
    side: str
    status: str
    qty: float | None
    filled_qty: float | None
    filled_avg_price: float | None
    notional: float | None


@dataclass(frozen=True)
class PersistedCycleReport:
    decision_timestamp: str
    execute_orders: bool
    processed_bar: bool
    skip_reason: str
    buy_signals: int
    sell_signals: int
    hold_signals: int
    error_signals: int
    orders_submitted: int
    observed_at_utc: str | None = None


@dataclass(frozen=True)
class PersistedExecutionActivity:
    observed_at_utc: str | None
    decision_timestamp: str | None
    symbol: str
    side: str
    qty: float | None
    price: float | None
    reason: str | None = None
    source_event: str | None = None
    allowed: bool | None = None


@dataclass(frozen=True)
class PersistedRiskCheck:
    observed_at_utc: str | None
    decision_timestamp: str | None
    symbol: str
    action: str
    allowed: bool
    block_reason: str | None = None
    detail: str | None = None


@dataclass(frozen=True)
class PersistedSignalDecision:
    observed_at_utc: str | None
    decision_timestamp: str | None
    symbol: str
    action: str
    price: float | None
    sma: float | None
    deviation_pct: float | None
    trend_sma: float | None
    above_trend_sma: bool | None
    atr_pct: float | None
    atr_percentile: float | None
    volume_ratio: float | None
    strategy_mode: str | None = None
    final_signal_reason: str | None = None
    decision_summary: str | None = None
    hybrid_branch_active: str | None = None
    hybrid_entry_branch: str | None = None
    hybrid_regime_branch: str | None = None
    mr_signal: str | None = None
    trend_sma_slope: float | None = None
    entry_reference_price: float | None = None
    exit_reference_price: float | None = None
    stop_reference_price: float | None = None
    vwap: float | None = None
    adx: float | None = None
    regime_state: str | None = None
    rejection: str | None = None
    trend_filter: str | None = None
    atr_filter: str | None = None
    window_open: bool | None = None
    holding: bool | None = None


@dataclass(frozen=True)
class PersistedBotSnapshot:
    timestamp_utc: str
    cash: float
    buying_power: float
    equity: float
    last_equity: float
    daily_pnl: float
    kill_switch_triggered: bool
    positions: dict[str, object] = field(default_factory=dict)
    symbols: list[PersistedSymbolSnapshot] = field(default_factory=list)


@dataclass(frozen=True)
class PersistedStartupConfig:
    session_id: str | None
    started_at_utc: str | None
    launch_mode: str
    execution_enabled: bool
    paper: bool
    account_mode: str
    strategy_mode: str
    bar_timeframe_minutes: int
    sma_bars: int
    symbols: list[str]
    symbol_count: int
    runtime_config_path: str | None
    runtime_overrides: tuple[str, ...]
    max_usd_per_trade: float
    max_symbol_exposure_usd: float
    max_open_positions: int
    max_daily_loss_usd: float
    max_orders_per_minute: int
    max_price_deviation_bps: float
    max_live_price_age_seconds: int
    max_data_delay_seconds: int
    db_path: str
    ml_lookback_bars: int = 0
    breakout_max_stop_pct: float = 0.0
    sma_stop_pct: float = 0.0
    mean_reversion_exit_style: str = ""
    mean_reversion_max_atr_percentile: float = 0.0
    mean_reversion_trend_filter: bool = False
    mean_reversion_trend_slope_filter: bool = False
    mean_reversion_stop_pct: float = 0.0
    historical_feed: str = ""
    live_feed: str = ""
    latest_bar_feed: str = ""
    bar_build_mode: str = ""
    apply_updated_bars: bool = False
    post_bar_reconcile_poll: bool = False
    block_trading_until_resync: bool = False
    assert_feed_on_startup: bool = False
    log_bar_components: bool = False
    resync_status: str | None = None
    resync_started_at_utc: str | None = None
    resync_completed_at_utc: str | None = None
    resync_reason_codes: tuple[str, ...] = ()
    gate_allows_entries: bool | None = None
    gate_allows_exits: bool | None = None
    runtime_config_approved: bool | None = None
    runtime_config_rejection_reasons: tuple[str, ...] = ()
    baseline_valid_for_comparison: bool | None = None
    baseline_validation_errors: tuple[str, ...] = ()
    artifact_path: str | None = None


@dataclass(frozen=True)
class PersistedNarration:
    timestamp_utc: str | None
    symbol: str
    narration: str
    event_type: str  # "signal" | "risk_block" | "exit"


@dataclass(frozen=True)
class PersistedNearMiss:
    timestamp_utc: str | None
    symbol: str
    narration: str
    rejection: str | None  # "trend_filter" | "atr_filter" | other


@dataclass(frozen=True)
class DrilldownEvent:
    """Normalised view of one decision event for the drilldown panel."""
    timestamp_utc: str | None
    symbol: str
    event_type: str          # "signal.evaluated" | "risk.check" | "position.closed"
    action: str              # "BUY" | "SELL" | "HOLD" | ""
    strategy_mode: str | None
    allowed: bool | None     # risk.check only
    block_reason: str | None
    rejection: str | None    # signal.evaluated rejection key
    trend_filter: str | None
    atr_filter: str | None
    above_trend_sma: bool | None
    deviation_pct: float | None
    atr_pct: float | None
    atr_percentile: float | None
    ml_prob: float | None
    pnl_usd: float | None    # position.closed only
    exit_reason: str | None  # position.closed only
    bar_close: float | None
    sma: float | None
    volume_ratio: float | None
    window_open: bool | None
    holding: bool | None


@dataclass(frozen=True)
class PersistedActivityItem:
    observed_at_utc: str | None
    event_type: str
    title: str
    body: str
    level: str = "info"
    symbol: str | None = None
    decision_timestamp: str | None = None


@dataclass(frozen=True)
class DashboardState:
    startup_config: PersistedStartupConfig | None
    storage: BotStorage
    snapshot: PersistedBotSnapshot
    recent_orders: list[PersistedOrderSnapshot]
    last_cycle_report: PersistedCycleReport | None
    feed_status: str
    has_persisted_snapshot: bool
    has_persisted_startup_config: bool
    session_warnings: tuple[str, ...] = ()
    symbol_state_status: str = "unknown"
    ignored_snapshot_symbols: tuple[str, ...] = ()
    latest_signals: dict = field(default_factory=dict)
    recent_execution_activity: tuple[PersistedExecutionActivity, ...] = ()
    latest_signal_rows: tuple[PersistedSignalDecision, ...] = ()
    latest_cycle_risk_checks: tuple[PersistedRiskCheck, ...] = ()
    session_block_reason_counts: tuple[tuple[str, int], ...] = ()
    session_first_prices: dict[str, float] = field(default_factory=dict)
    recent_narrations: tuple[PersistedNarration, ...] = ()
    recent_near_misses: tuple[PersistedNearMiss, ...] = ()
    drilldown_candidates: tuple[DrilldownEvent, ...] = ()
    recent_activity: tuple[PersistedActivityItem, ...] = ()
    drift_report: DriftReport | None = None


@dataclass(frozen=True)
class BrokerSafetyPanelState:
    """Read-only, credential-free automation safety status for the dashboard."""

    environment: str
    account_id_masked: str
    lock_status: str
    reconciliation_time: str
    managed_position_count: int | None
    unmanaged_position_count: int | None
    managed_order_count: int | None
    unmanaged_order_count: int | None
    mismatch_status: str
    automation_readiness: str
    blocking_reasons: tuple[str, ...]
    result: str
    stale: bool


def _log_root() -> Path:
    return resolve_log_root()


def _is_session_log_dir(path: Path) -> bool:
    if not path.is_dir():
        return False
    try:
        datetime.strptime(path.name, "%Y-%m-%d")
    except ValueError:
        return False
    return True


def _latest_log_dir() -> Path | None:
    log_root = _log_root()
    if not log_root.exists():
        return None
    day_dirs = sorted((path for path in log_root.iterdir() if _is_session_log_dir(path)), key=lambda path: path.name)
    return day_dirs[-1] if day_dirs else None


def _startup_config_from_payload(
    payload: dict[str, object],
    artifact_path: Path | None = None,
    *,
    authoritative_database_path: Path | None = None,
) -> PersistedStartupConfig | None:
    if not isinstance(payload, dict):
        return None

    def required_text(field_name: str) -> str:
        value = payload.get(field_name)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{field_name} must be a nonempty string")
        return value.strip()

    def optional_bool(field_name: str, default: bool = False) -> bool:
        if field_name not in payload or payload[field_name] is None:
            return default
        value = payload[field_name]
        if type(value) is not bool:
            raise ValueError(f"{field_name} must be a boolean")
        return value

    def optional_bool_or_none(field_name: str) -> bool | None:
        if field_name not in payload or payload[field_name] is None:
            return None
        value = payload[field_name]
        if type(value) is not bool:
            raise ValueError(f"{field_name} must be a boolean or null")
        return value

    def optional_text_list(field_name: str) -> tuple[str, ...]:
        value = payload.get(field_name, [])
        if value is None:
            return ()
        if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
            raise ValueError(f"{field_name} must be a list of strings")
        return tuple(item.strip() for item in value if item.strip())

    try:
        session_id = required_text("session_id")
        started_at_utc = required_text("started_at_utc")
        if _parse_iso_utc(started_at_utc) is None:
            raise ValueError("started_at_utc must be a timezone-aware ISO timestamp")
        launch_mode = required_text("launch_mode")
        account_mode = required_text("account_mode")
        strategy_mode = required_text("strategy_mode")

        execution_enabled = payload.get("execution_enabled")
        paper = payload.get("paper")
        if type(execution_enabled) is not bool or type(paper) is not bool:
            raise ValueError("execution_enabled and paper must be booleans")
        normalized_account_mode = account_mode.lower()
        if paper and normalized_account_mode != "paper":
            raise ValueError("paper metadata must use account_mode=paper")
        if not paper and normalized_account_mode not in {"live", "live-account"}:
            raise ValueError("non-paper metadata must identify a live account mode")

        timeframe = payload.get("bar_timeframe_minutes")
        if type(timeframe) is not int or timeframe <= 0:
            raise ValueError("bar_timeframe_minutes must be a positive integer")

        raw_symbols = payload.get("symbols")
        if not isinstance(raw_symbols, list) or any(
            not isinstance(item, str) for item in raw_symbols
        ):
            raise ValueError("symbols must be a list of strings")
        symbols = normalize_symbols(raw_symbols)
        symbol_count = payload.get("symbol_count")
        if (
            not symbols
            or len(symbols) != len(raw_symbols)
            or type(symbol_count) is not int
            or symbol_count != len(symbols)
        ):
            raise ValueError("symbols must be nonempty and match symbol_count")

        raw_db_path = required_text("db_path")
        database_path = resolve_database_path(raw_db_path)
        expected_database_path = authoritative_database_path
        if expected_database_path is None:
            expected_database_path = (
                resolve_explicit_database_path() or resolve_database_path()
            )
        if (
            os.path.normcase(str(database_path))
            != os.path.normcase(str(expected_database_path))
            or not database_path.is_file()
        ):
            raise ValueError("db_path does not match the existing database authority")

        return PersistedStartupConfig(
            session_id=session_id,
            started_at_utc=started_at_utc,
            launch_mode=launch_mode,
            execution_enabled=execution_enabled,
            paper=paper,
            account_mode=account_mode,
            strategy_mode=strategy_mode,
            bar_timeframe_minutes=timeframe,
            sma_bars=int(payload.get("sma_bars", 0) or 0),
            symbols=symbols,
            symbol_count=symbol_count,
            runtime_config_path=str(payload.get("runtime_config_path", "") or "").strip() or None,
            runtime_overrides=optional_text_list("runtime_overrides"),
            max_usd_per_trade=float(payload.get("max_usd_per_trade", 0.0) or 0.0),
            max_symbol_exposure_usd=float(payload.get("max_symbol_exposure_usd", 0.0) or 0.0),
            max_open_positions=int(payload.get("max_open_positions", 0) or 0),
            max_daily_loss_usd=float(payload.get("max_daily_loss_usd", 0.0) or 0.0),
            max_orders_per_minute=int(payload.get("max_orders_per_minute", 0) or 0),
            max_price_deviation_bps=float(payload.get("max_price_deviation_bps", 0.0) or 0.0),
            max_live_price_age_seconds=int(payload.get("max_live_price_age_seconds", 0) or 0),
            max_data_delay_seconds=int(payload.get("max_data_delay_seconds", 0) or 0),
            ml_lookback_bars=int(payload.get("ml_lookback_bars", 0) or 0),
            breakout_max_stop_pct=float(payload.get("breakout_max_stop_pct", 0.0) or 0.0),
            sma_stop_pct=float(payload.get("sma_stop_pct", 0.0) or 0.0),
            mean_reversion_exit_style=str(payload.get("mean_reversion_exit_style", "") or ""),
            mean_reversion_max_atr_percentile=float(payload.get("mean_reversion_max_atr_percentile", 0.0) or 0.0),
            mean_reversion_trend_filter=optional_bool("mean_reversion_trend_filter"),
            mean_reversion_trend_slope_filter=optional_bool("mean_reversion_trend_slope_filter"),
            mean_reversion_stop_pct=float(payload.get("mean_reversion_stop_pct", 0.0) or 0.0),
            db_path=str(database_path),
            historical_feed=str(payload.get("historical_feed", "") or ""),
            live_feed=str(payload.get("live_feed", "") or ""),
            latest_bar_feed=str(payload.get("latest_bar_feed", "") or ""),
            bar_build_mode=str(payload.get("bar_build_mode", "") or ""),
            apply_updated_bars=optional_bool("apply_updated_bars"),
            post_bar_reconcile_poll=optional_bool("post_bar_reconcile_poll"),
            block_trading_until_resync=optional_bool("block_trading_until_resync"),
            assert_feed_on_startup=optional_bool("assert_feed_on_startup"),
            log_bar_components=optional_bool("log_bar_components"),
            resync_status=str(payload.get("resync_status", "") or "") or None,
            resync_started_at_utc=str(payload.get("resync_started_at_utc", "") or "") or None,
            resync_completed_at_utc=str(payload.get("resync_completed_at_utc", "") or "") or None,
            resync_reason_codes=optional_text_list("resync_reason_codes"),
            gate_allows_entries=optional_bool_or_none("gate_allows_entries"),
            gate_allows_exits=optional_bool_or_none("gate_allows_exits"),
            runtime_config_approved=optional_bool_or_none("runtime_config_approved"),
            runtime_config_rejection_reasons=optional_text_list(
                "runtime_config_rejection_reasons"
            ),
            baseline_valid_for_comparison=optional_bool_or_none(
                "baseline_valid_for_comparison"
            ),
            baseline_validation_errors=optional_text_list(
                "baseline_validation_errors"
            ),
            artifact_path=str(artifact_path) if artifact_path is not None else None,
        )
    except (OSError, TypeError, ValueError, OverflowError):
        return None


def _load_latest_jsonl_event(path: Path, event_name: str | None = None) -> dict[str, object] | None:
    if not path.exists():
        return None
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return None
    for raw_line in reversed(lines):
        if not raw_line.strip():
            continue
        try:
            payload = json.loads(raw_line)
        except json.JSONDecodeError:
            continue
        if not isinstance(payload, dict):
            continue
        if event_name is None or payload.get("event") == event_name:
            return payload
    return None


def _next_startup_started_at(
    startup_configs: list[PersistedStartupConfig],
    startup_config: PersistedStartupConfig | None,
) -> str | None:
    if startup_config is None:
        return None
    for index, candidate in enumerate(startup_configs):
        if candidate == startup_config and index + 1 < len(startup_configs):
            return startup_configs[index + 1].started_at_utc
    return None


def _payload_matches_session(
    payload: dict[str, object],
    *,
    session_id: str | None = None,
    started_at_utc: str | None = None,
    ended_before_utc: str | None = None,
) -> bool:
    if session_id:
        payload_session_value = payload.get("session_id")
        if not isinstance(payload_session_value, str):
            return False
        payload_session_id = payload_session_value.strip()
        if not payload_session_id:
            return False
        return payload_session_id == session_id

    payload_ts = (
        str(payload.get("ts", "")) or None
        or str(payload.get("decision_ts", "")) or None
    )
    payload_dt = _parse_iso_utc(payload_ts)
    start_dt = _parse_iso_utc(started_at_utc)
    end_dt = _parse_iso_utc(ended_before_utc)
    if payload_dt is None:
        return session_id is None and started_at_utc is None and ended_before_utc is None
    if start_dt is not None and payload_dt < start_dt:
        return False
    if end_dt is not None and payload_dt >= end_dt:
        return False
    return True


def _load_jsonl_events(
    path: Path,
    event_names: set[str] | None = None,
    limit: int | None = None,
    *,
    session_id: str | None = None,
    started_at_utc: str | None = None,
    ended_before_utc: str | None = None,
) -> list[dict[str, object]]:
    if not path.exists():
        return []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return []

    results: list[dict[str, object]] = []
    iterable = reversed(lines) if limit is not None else lines
    for raw_line in iterable:
        if not raw_line.strip():
            continue
        try:
            payload = json.loads(raw_line)
        except json.JSONDecodeError:
            continue
        if not isinstance(payload, dict):
            continue
        if event_names is not None and str(payload.get("event", "")) not in event_names:
            continue
        if not _payload_matches_session(
            payload,
            session_id=session_id,
            started_at_utc=started_at_utc,
            ended_before_utc=ended_before_utc,
        ):
            continue
        results.append(payload)
        if limit is not None and len(results) >= limit:
            break
    if limit is not None:
        results.reverse()
    return results


def _load_latest_cycle_report(
    *,
    session_id: str | None = None,
    started_at_utc: str | None = None,
    ended_before_utc: str | None = None,
) -> PersistedCycleReport | None:
    log_dir = _latest_log_dir()
    if log_dir is None:
        return None
    rows = _load_jsonl_events(
        log_dir / "risk.jsonl",
        {"cycle.summary"},
        limit=1,
        session_id=session_id,
        started_at_utc=started_at_utc,
        ended_before_utc=ended_before_utc,
    )
    payload = rows[-1] if rows else None
    if payload is None:
        return None
    return PersistedCycleReport(
        decision_timestamp=str(payload.get("decision_ts", "")),
        execute_orders=bool(payload.get("execute_orders", False)),
        processed_bar=bool(payload.get("processed_bar", False)),
        skip_reason=str(payload.get("skip_reason", "")),
        buy_signals=int(payload.get("buy_signals", 0) or 0),
        sell_signals=int(payload.get("sell_signals", 0) or 0),
        hold_signals=int(payload.get("hold_signals", 0) or 0),
        error_signals=int(payload.get("error_signals", 0) or 0),
        orders_submitted=int(payload.get("orders_submitted", 0) or 0),
        observed_at_utc=str(payload.get("ts", "")) or None,
    )


def _load_feed_status(
    *,
    session_id: str | None = None,
    started_at_utc: str | None = None,
    ended_before_utc: str | None = None,
) -> str:
    log_dir = _latest_log_dir()
    if log_dir is None:
        return "no persisted logs yet"
    stale_rows = _load_jsonl_events(
        log_dir / "bars.jsonl",
        {"bar.stale_warning"},
        limit=1,
        session_id=session_id,
        started_at_utc=started_at_utc,
        ended_before_utc=ended_before_utc,
    )
    received_rows = _load_jsonl_events(
        log_dir / "bars.jsonl",
        {"bar.received"},
        limit=1,
        session_id=session_id,
        started_at_utc=started_at_utc,
        ended_before_utc=ended_before_utc,
    )
    stale_warning = stale_rows[-1] if stale_rows else None
    bar_received = received_rows[-1] if received_rows else None
    # Use whichever event is more recent so a pre-market stale_warning doesn't
    # persist as the status after fresh bars arrive during the session.
    stale_ts = str(stale_warning.get("ts", "")) if stale_warning else ""
    received_ts = str(bar_received.get("ts", "")) if bar_received else ""
    if stale_warning is not None and stale_ts >= received_ts:
        age_s = stale_warning.get("bar_age_s")
        if age_s is not None:
            return f"persisted bar stale ({float(age_s):.0f}s late)"
        return "persisted bar stale"
    if bar_received is not None:
        age_s = bar_received.get("bar_age_s")
        if age_s is not None:
            return f"persisted bar fresh ({float(age_s):.0f}s late)"
        return "persisted bar received"
    return "no persisted bar logs yet"


def _load_startup_configs() -> list[PersistedStartupConfig]:
    log_root = _log_root()
    if not log_root.exists():
        return []
    authoritative_database_path = (
        resolve_explicit_database_path() or resolve_database_path()
    )
    startup_configs: list[PersistedStartupConfig] = []
    try:
        day_dirs = sorted(
            (path for path in log_root.iterdir() if _is_session_log_dir(path)),
            key=lambda path: path.name,
        )
    except OSError:
        return []
    for log_dir in day_dirs:
        try:
            startup_paths = sorted(
                log_dir.glob("startup_config.*.json"), key=lambda path: path.name
            )
        except OSError:
            continue
        if not startup_paths:
            legacy_path = log_dir / "startup_config.json"
            if legacy_path.is_file():
                startup_paths = [legacy_path]
        for path in startup_paths:
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, UnicodeError, json.JSONDecodeError):
                continue
            config = _startup_config_from_payload(
                payload,
                artifact_path=path,
                authoritative_database_path=authoritative_database_path,
            )
            if config is not None:
                startup_configs.append(config)
    startup_configs.sort(
        key=lambda item: (
            _parse_iso_utc(item.started_at_utc) or _UTC_MIN,
            item.artifact_path or "",
        )
    )
    return startup_configs


def _empty_snapshot() -> PersistedBotSnapshot:
    return PersistedBotSnapshot(
        timestamp_utc="No persisted snapshot yet",
        cash=0.0,
        buying_power=0.0,
        equity=0.0,
        last_equity=0.0,
        daily_pnl=0.0,
        kill_switch_triggered=False,
        positions={},
        symbols=[],
    )


def _to_float(value: object) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _parse_iso_utc(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        if parsed.tzinfo is None or parsed.utcoffset() is None:
            return None
        return parsed.astimezone(timezone.utc)
    except (OverflowError, ValueError):
        return None


def _strict_decimal(value: object, *, nonnegative: bool = False) -> Decimal | None:
    if type(value) is not str or not value.strip():
        return None
    try:
        parsed = Decimal(value)
    except InvalidOperation:
        return None
    if not parsed.is_finite() or (nonnegative and parsed < 0):
        return None
    return parsed


def _strict_quantity_map(
    value: object,
    *,
    nonnegative: bool,
) -> dict[str, Decimal] | None:
    if type(value) is not dict:
        return None
    normalized: dict[str, Decimal] = {}
    for raw_symbol, raw_quantity in value.items():
        if type(raw_symbol) is not str:
            return None
        symbol = raw_symbol.strip().upper()
        quantity = _strict_decimal(raw_quantity, nonnegative=nonnegative)
        if not symbol or symbol != raw_symbol or quantity is None:
            return None
        normalized[symbol] = quantity
    return normalized


def _valid_position_records(value: object) -> bool:
    required = {"symbol", "qty", "managed", "reason"}
    if type(value) is not list:
        return False
    seen: set[str] = set()
    for item in value:
        if type(item) is not dict or set(item) != required:
            return False
        symbol = item.get("symbol")
        reason = item.get("reason")
        if (
            type(symbol) is not str
            or not symbol
            or symbol != symbol.strip().upper()
            or symbol in seen
            or type(item.get("managed")) is not bool
            or type(reason) is not str
            or not reason.strip()
            or _strict_decimal(item.get("qty")) is None
        ):
            return False
        seen.add(symbol)
    return True


def _valid_order_records(value: object) -> bool:
    required = {
        "order_id",
        "client_order_id",
        "symbol",
        "side",
        "status",
        "qty",
        "filled_qty",
        "remaining_qty",
        "managed",
        "reason",
    }
    if type(value) is not list:
        return False
    seen: set[str] = set()
    for item in value:
        if type(item) is not dict or set(item) != required:
            return False
        order_id = item.get("order_id")
        symbol = item.get("symbol")
        quantities = {
            key: _strict_decimal(item.get(key), nonnegative=True)
            for key in ("qty", "filled_qty", "remaining_qty")
        }
        if (
            type(order_id) is not str
            or not order_id.strip()
            or order_id in seen
            or type(item.get("client_order_id")) is not str
            or type(symbol) is not str
            or not symbol
            or symbol != symbol.strip().upper()
            or str(item.get("side", "")).lower() not in {"buy", "sell"}
            or type(item.get("status")) is not str
            or not str(item.get("status")).strip()
            or type(item.get("managed")) is not bool
            or type(item.get("reason")) is not str
            or not str(item.get("reason")).strip()
            or any(quantity is None for quantity in quantities.values())
        ):
            return False
        if quantities["filled_qty"] + quantities["remaining_qty"] != quantities["qty"]:
            return False
        seen.add(order_id)
    return True


def _canonical_record_multiset(value: list[object]) -> tuple[str, ...]:
    return tuple(
        sorted(json.dumps(item, sort_keys=True, separators=(",", ":")) for item in value)
    )


def _load_paper_shadow_account_fingerprint(config_path: Path) -> str:
    """Load the non-secret expected account binding from the named profile."""

    try:
        payload = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError("paper-shadow config is missing or malformed") from exc
    if type(payload) is not dict:
        raise ValueError("paper-shadow config must be a JSON object")
    launch = payload.get("launch")
    if type(launch) is not dict:
        raise ValueError("paper-shadow config launch metadata is missing or invalid")
    expected_launch_values = {
        "profile": "paper_shadow",
        "broker_environment": "paper",
        "submission_policy": "deny",
        "dashboard_mode": "read_only",
    }
    for field_name, expected_value in expected_launch_values.items():
        value = launch.get(field_name)
        if type(value) is not str or value != expected_value:
            raise ValueError(
                f"paper-shadow config launch field {field_name} is invalid"
            )
    fingerprint = launch.get("expected_account_fingerprint")
    if type(fingerprint) is not str or not _ACCOUNT_FINGERPRINT_RE.fullmatch(
        fingerprint
    ):
        raise ValueError("paper-shadow expected account fingerprint is invalid")
    return fingerprint


def _build_broker_safety_panel_state(
    payload: dict[str, object] | None,
    lock_status: object | None,
    *,
    now_utc: datetime | None = None,
    max_age_seconds: float = 300.0,
    artifact_failure_reason: str | None = None,
    repository_readiness_reasons: tuple[str, ...] = (),
    expected_account_fingerprint: str | None = None,
) -> BrokerSafetyPanelState:
    """Project persisted reconciliation and lock data into a fail-closed view."""

    observed_lock_state = getattr(getattr(lock_status, "state", None), "value", None)
    if observed_lock_state not in {"held", "unlocked", "unknown"}:
        observed_lock_state = "unknown"
    if observed_lock_state == "held":
        if bool(getattr(lock_status, "held_by_current_process", False)):
            lock_label = "held by this process"
        elif bool(getattr(lock_status, "metadata_valid", False)):
            lock_label = "held by another process"
        else:
            lock_label = "held (holder metadata unavailable)"
    else:
        lock_label = str(observed_lock_state)

    lock_reasons = ["trading_lock_status_unknown"] if observed_lock_state == "unknown" else []
    account_binding_reasons: list[str] = []
    if (
        type(expected_account_fingerprint) is not str
        or not _ACCOUNT_FINGERPRINT_RE.fullmatch(expected_account_fingerprint)
    ):
        account_binding_reasons.append(
            "expected_account_fingerprint_missing_or_invalid"
        )
    external_reasons = [
        *lock_reasons,
        *repository_readiness_reasons,
        *account_binding_reasons,
    ]

    def blocked_unknown(reason: str) -> BrokerSafetyPanelState:
        reasons = tuple(dict.fromkeys([reason, *external_reasons]))
        return BrokerSafetyPanelState(
            environment="unknown",
            account_id_masked="unknown",
            lock_status=lock_label,
            reconciliation_time="unknown",
            managed_position_count=None,
            unmanaged_position_count=None,
            managed_order_count=None,
            unmanaged_order_count=None,
            mismatch_status="unknown",
            automation_readiness="blocked",
            blocking_reasons=reasons,
            result="BLOCKED_RECONCILIATION_FAILURE",
            stale=True,
        )

    if payload is None:
        return blocked_unknown(
            artifact_failure_reason or "reconciliation_artifact_missing_or_malformed"
        )

    required_text = (
        "environment",
        "account_id_masked",
        "account_fingerprint",
        "account_status",
        "checked_at_utc",
        "automation_readiness",
        "result",
    )
    required_lists = (
        "managed_positions",
        "unmanaged_positions",
        "managed_orders",
        "unmanaged_orders",
        "current_broker_positions",
        "current_open_pending_orders",
        "ambiguous_intents",
        "state_mismatches",
        "blocking_reasons",
    )
    required_dicts = ("local_positions", "pending_buy_qty", "pending_sell_qty")
    malformed = any(
        not isinstance(payload.get(key), str) or not str(payload.get(key)).strip()
        for key in required_text
    ) or any(type(payload.get(key)) is not list for key in required_lists)
    malformed = malformed or any(type(payload.get(key)) is not dict for key in required_dicts)
    malformed = malformed or type(payload.get("reconciliation_succeeded")) is not bool
    raw_reasons = payload.get("blocking_reasons")
    raw_mismatches = payload.get("state_mismatches")
    if isinstance(raw_reasons, list) and any(
        not isinstance(item, str) or not item.strip() for item in raw_reasons
    ):
        malformed = True
    if isinstance(raw_mismatches, list) and any(not isinstance(item, str) for item in raw_mismatches):
        malformed = True
    raw_ambiguous_intents = payload.get("ambiguous_intents")
    if isinstance(raw_ambiguous_intents, list) and any(
        type(item) is not str or not item.strip() for item in raw_ambiguous_intents
    ):
        malformed = True
    current_positions = payload.get("current_broker_positions")
    managed_positions = payload.get("managed_positions")
    unmanaged_positions = payload.get("unmanaged_positions")
    current_orders = payload.get("current_open_pending_orders")
    managed_orders = payload.get("managed_orders")
    unmanaged_orders = payload.get("unmanaged_orders")
    if not all(
        _valid_position_records(value)
        for value in (current_positions, managed_positions, unmanaged_positions)
    ):
        malformed = True
    if not all(
        _valid_order_records(value)
        for value in (current_orders, managed_orders, unmanaged_orders)
    ):
        malformed = True
    local_positions = _strict_quantity_map(payload.get("local_positions"), nonnegative=False)
    pending_buy_qty = _strict_quantity_map(payload.get("pending_buy_qty"), nonnegative=True)
    pending_sell_qty = _strict_quantity_map(payload.get("pending_sell_qty"), nonnegative=True)
    if local_positions is None or pending_buy_qty is None or pending_sell_qty is None:
        malformed = True
    if malformed:
        return blocked_unknown("reconciliation_artifact_malformed")

    checked_at = _parse_iso_utc(str(payload["checked_at_utc"]))
    if checked_at is None or checked_at.tzinfo is None:
        return blocked_unknown("reconciliation_timestamp_missing_or_invalid")

    current_time = now_utc or datetime.now(timezone.utc)
    if current_time.tzinfo is None:
        current_time = current_time.replace(tzinfo=timezone.utc)
    age_seconds = (current_time.astimezone(timezone.utc) - checked_at.astimezone(timezone.utc)).total_seconds()

    environment = str(payload["environment"]).strip().lower()
    account_id_masked = str(payload["account_id_masked"]).strip()
    account_fingerprint = str(payload["account_fingerprint"]).strip()
    account_status = str(payload["account_status"]).strip().lower()
    readiness = str(payload["automation_readiness"]).strip().lower()
    result = str(payload["result"]).strip()
    managed_positions = payload["managed_positions"]
    unmanaged_positions = payload["unmanaged_positions"]
    managed_orders = payload["managed_orders"]
    unmanaged_orders = payload["unmanaged_orders"]
    state_mismatches = payload["state_mismatches"]

    reasons = [str(item).strip() for item in raw_reasons]
    reasons.extend(external_reasons)
    consistency_failed = False
    if _canonical_record_multiset(current_positions) != _canonical_record_multiset(
        [*managed_positions, *unmanaged_positions]
    ):
        consistency_failed = True
    if _canonical_record_multiset(current_orders) != _canonical_record_multiset(
        [*managed_orders, *unmanaged_orders]
    ):
        consistency_failed = True
    if any(item["managed"] is not True for item in managed_positions) or any(
        item["managed"] is not False for item in unmanaged_positions
    ):
        consistency_failed = True
    if any(item["managed"] is not True for item in managed_orders) or any(
        item["managed"] is not False for item in unmanaged_orders
    ):
        consistency_failed = True

    broker_quantities = {
        str(item["symbol"]): _strict_decimal(item["qty"])
        for item in current_positions
    }
    derived_buy: dict[str, Decimal] = {}
    derived_sell: dict[str, Decimal] = {}
    for item in current_orders:
        target = derived_buy if str(item["side"]).lower() == "buy" else derived_sell
        symbol = str(item["symbol"])
        target[symbol] = target.get(symbol, Decimal("0")) + Decimal(
            str(item["remaining_qty"])
        )
    normalized_pending_buy = {
        key: value for key, value in pending_buy_qty.items() if value != 0
    }
    normalized_pending_sell = {
        key: value for key, value in pending_sell_qty.items() if value != 0
    }
    if derived_buy != normalized_pending_buy or derived_sell != normalized_pending_sell:
        consistency_failed = True
    derived_state_mismatch = any(
        (broker_quantities.get(symbol) or Decimal("0"))
        != local_positions.get(symbol, Decimal("0"))
        for symbol in set(broker_quantities) | set(local_positions)
    )
    if derived_state_mismatch and not state_mismatches:
        consistency_failed = True
    if consistency_failed:
        reasons.append("reconciliation_metadata_inconsistent")
    stale = age_seconds > max_age_seconds
    if age_seconds < -30.0:
        stale = True
        reasons.append("reconciliation_timestamp_in_future")
    elif stale:
        reasons.append("reconciliation_stale_over_5_minutes")
    if environment != "paper":
        reasons.append("environment_not_positively_paper")
    if account_id_masked.lower() in {"", "unknown", "none", "n/a"}:
        reasons.append("account_identity_unknown")
    if not _ACCOUNT_FINGERPRINT_RE.fullmatch(account_fingerprint):
        reasons.append("reconciliation_account_fingerprint_invalid")
    elif (
        type(expected_account_fingerprint) is str
        and _ACCOUNT_FINGERPRINT_RE.fullmatch(expected_account_fingerprint)
        and account_fingerprint != expected_account_fingerprint
    ):
        reasons.append("reconciliation_account_fingerprint_mismatch")
    if account_status != "active":
        reasons.append("broker_account_not_active")
    if unmanaged_positions:
        reasons.append("unmanaged_positions_present")
    if unmanaged_orders:
        reasons.append("unmanaged_orders_present")
    if state_mismatches:
        reasons.append("broker_local_state_mismatch")
    if payload["ambiguous_intents"]:
        reasons.append("ambiguous_intents_present")
    if readiness != "ready":
        reasons.append("persisted_reconciliation_not_ready")
    if result != "SAFE_FOR_PAPER_AUTOMATION":
        reasons.append("persisted_result_not_safe_for_paper_automation")
    if payload.get("reconciliation_succeeded") is not True:
        reasons.append("reconciliation_not_successful")

    unique_reasons = tuple(dict.fromkeys(reasons))
    positively_ready = not unique_reasons
    return BrokerSafetyPanelState(
        environment=environment if environment in {"paper", "live"} else "unknown",
        account_id_masked=account_id_masked,
        lock_status=lock_label,
        reconciliation_time=checked_at.astimezone(timezone.utc).isoformat(),
        managed_position_count=len(managed_positions),
        unmanaged_position_count=len(unmanaged_positions),
        managed_order_count=len(managed_orders),
        unmanaged_order_count=len(unmanaged_orders),
        mismatch_status="clear" if not state_mismatches else f"mismatch ({len(state_mismatches)})",
        automation_readiness="ready" if positively_ready else "blocked",
        blocking_reasons=unique_reasons,
        result=result if positively_ready else result or "BLOCKED_RECONCILIATION_FAILURE",
        stale=stale,
    )


def load_broker_safety_panel_state(
    reconciliation_path: Path | None = None,
    *,
    now_utc: datetime | None = None,
    max_age_seconds: float = 300.0,
    readiness_config_path: Path | None = None,
    readiness_evidence_dir: Path | None = None,
) -> BrokerSafetyPanelState:
    """Load only persisted/redacted safety state; never create a broker client."""

    payload: dict[str, object] | None = None
    artifact_failure_reason: str | None = None
    try:
        from automation_safety import DEFAULT_RECONCILIATION_PATH, load_redacted_reconciliation

        artifact_path = reconciliation_path or DEFAULT_RECONCILIATION_PATH
        payload = load_redacted_reconciliation(artifact_path)
    except Exception:
        artifact_failure_reason = "reconciliation_artifact_read_failure"

    try:
        from tradeos.execution_lock import inspect_lock

        lock_status = inspect_lock()
    except Exception:
        lock_status = None

    repository_readiness_reasons: tuple[str, ...] = ()
    expected_account_fingerprint: str | None = None
    try:
        from tradeos.readiness import DEFAULT_EVIDENCE_DIR, select_latest_valid_evidence

        selected_config = readiness_config_path or (
            Path(__file__).resolve().parent / "config" / "paper_shadow_config.json"
        )
        expected_account_fingerprint = _load_paper_shadow_account_fingerprint(
            selected_config
        )
        selection = select_latest_valid_evidence(
            readiness_evidence_dir or DEFAULT_EVIDENCE_DIR,
            config_path=selected_config,
            repo_root=Path(__file__).resolve().parent,
            now_utc=now_utc,
        )
        if selection.evidence is None:
            repository_readiness_reasons = (
                "repository_readiness_evidence_missing_or_invalid",
            )
            if selection.rejected:
                repository_readiness_reasons += (
                    f"repository_readiness_rejected:{selection.rejected[0][1]}",
                )
    except Exception:
        repository_readiness_reasons = ("repository_readiness_check_failed",)

    return _build_broker_safety_panel_state(
        payload,
        lock_status,
        now_utc=now_utc,
        max_age_seconds=max_age_seconds,
        artifact_failure_reason=artifact_failure_reason,
        repository_readiness_reasons=repository_readiness_reasons,
        expected_account_fingerprint=expected_account_fingerprint,
    )


def _decision_timestamp_from_trace(trace: object) -> str | None:
    raw = str(trace or "")
    if "_" not in raw:
        return None
    try:
        epoch = int(raw.rsplit("_", 1)[-1])
        return datetime.fromtimestamp(epoch, tz=timezone.utc).isoformat()
    except (TypeError, ValueError, OSError):
        return None


def _build_session_warnings(
    startup_config: PersistedStartupConfig | None,
    latest_startup_config: PersistedStartupConfig | None,
    snapshot: PersistedBotSnapshot,
    has_persisted_snapshot: bool,
    last_cycle_report: PersistedCycleReport | None,
    ignored_snapshot_symbols: tuple[str, ...] = (),
) -> tuple[str, ...]:
    warnings: list[str] = []

    if startup_config is None:
        if has_persisted_snapshot:
            return (
                "The latest persisted database session has no matching validated "
                "startup metadata. The dashboard is showing that session's saved "
                "state without applying unrelated startup metadata.",
            )
        return ()
    latest_started_at = _parse_iso_utc(latest_startup_config.started_at_utc) if latest_startup_config is not None else None
    selected_started_at = _parse_iso_utc(startup_config.started_at_utc)
    if (
        latest_startup_config is not None
        and latest_startup_config != startup_config
        and latest_started_at is not None
        and selected_started_at is not None
        and latest_started_at > selected_started_at
    ):
        launch_time = latest_started_at.astimezone().strftime("%b %d, %Y %I:%M %p %Z")
        warnings.append(
            "A newer TradeOS launch started at "
            f"{launch_time}, but it has not saved session data yet. "
            "The dashboard is showing the most recent persisted session instead."
        )
    if ignored_snapshot_symbols:
        warnings.append(
            "Current config symbols: "
            f"[{format_symbol_list(startup_config.symbols)}]. "
            "Latest persisted snapshot symbols: "
            f"[{format_symbol_list(ignored_snapshot_symbols)}]. "
            "Persisted symbol state is from a different run and is being ignored."
        )
    if startup_config.runtime_config_approved is False:
        reasons = "; ".join(startup_config.runtime_config_rejection_reasons)
        detail = (
            f" Current runtime approval notes: {reasons}."
            if reasons else
            " Current runtime approval notes are unavailable."
        )
        warnings.append(
            "This session is using a runtime config that is marked unapproved for live trading."
            + detail
        )

    return tuple(warnings)


def _select_startup_config_for_session(
    startup_configs: list[PersistedStartupConfig],
    snapshot: PersistedBotSnapshot,
    has_persisted_snapshot: bool,
    last_cycle_report: PersistedCycleReport | None,
    latest_run_session_id: str | None = None,
    latest_run_environment: object | None = None,
) -> PersistedStartupConfig | None:
    if not startup_configs:
        return None
    if latest_run_session_id:
        matching = [
            config
            for config in startup_configs
            if (config.session_id or "") == latest_run_session_id
        ]
        if len(matching) == 1:
            selected = matching[0]
            if has_persisted_snapshot:
                if not isinstance(latest_run_environment, str):
                    return None
                persisted_environment = latest_run_environment.strip().lower()
                config_environment = "paper" if selected.paper else "live"
                if persisted_environment != config_environment:
                    return None
                raw_persisted_symbols = [item.symbol for item in snapshot.symbols]
                if any(
                    not isinstance(symbol, str) or not symbol.strip()
                    for symbol in raw_persisted_symbols
                ):
                    return None
                persisted_symbols = [
                    symbol.strip().upper() for symbol in raw_persisted_symbols
                ]
                if (
                    len(persisted_symbols) != selected.symbol_count
                    or len(set(persisted_symbols)) != len(persisted_symbols)
                    or not symbols_match(selected.symbols, persisted_symbols)
                ):
                    return None
            return selected
        # A persisted session identifier is authoritative. Timestamp proximity is
        # not provenance: falling through here can bind a later failed/test launch
        # to an unrelated database run and hide the actual saved state.
        return None
    if has_persisted_snapshot:
        # A persisted run without a usable session identifier is also unbound.
        # Do not infer provenance from timestamps for legacy or partial rows.
        return None
    target_timestamps: list[datetime] = []
    if has_persisted_snapshot:
        snapshot_dt = _parse_iso_utc(snapshot.timestamp_utc)
        if snapshot_dt is not None:
            target_timestamps.append(snapshot_dt)
    if last_cycle_report is not None:
        cycle_dt = _parse_iso_utc(last_cycle_report.decision_timestamp) or _parse_iso_utc(
            last_cycle_report.observed_at_utc
        )
        if cycle_dt is not None:
            target_timestamps.append(cycle_dt)

    def _config_windows() -> list[tuple[PersistedStartupConfig, datetime | None, datetime | None]]:
        windows: list[tuple[PersistedStartupConfig, datetime | None, datetime | None]] = []
        for index, config in enumerate(startup_configs):
            start_dt = _parse_iso_utc(config.started_at_utc)
            next_dt = (
                _parse_iso_utc(startup_configs[index + 1].started_at_utc)
                if index + 1 < len(startup_configs)
                else None
            )
            windows.append((config, start_dt, next_dt))
        return windows

    config_windows = _config_windows()
    for target_dt in target_timestamps:
        for config, start_dt, next_dt in config_windows:
            if start_dt is not None and target_dt < start_dt:
                continue
            if next_dt is not None and target_dt >= next_dt:
                continue
            return config

    if target_timestamps:
        latest_target = max(target_timestamps)
        eligible = [
            config
            for config, start_dt, _ in config_windows
            if start_dt is not None and start_dt <= latest_target
        ]
        if eligible:
            return max(eligible, key=lambda item: _parse_iso_utc(item.started_at_utc) or _UTC_MIN)
    return max(startup_configs, key=lambda item: _parse_iso_utc(item.started_at_utc) or _UTC_MIN)

def _build_snapshot(storage: BotStorage, session_id: str | None = None) -> tuple[PersistedBotSnapshot, bool]:
    latest_run = storage.get_latest_run(session_id=session_id)
    if latest_run is None:
        return _empty_snapshot(), False

    symbol_rows = storage.get_latest_symbol_snapshot(session_id=session_id)
    symbols: list[PersistedSymbolSnapshot] = []
    positions: dict[str, object] = {}
    for row in symbol_rows:
        item = PersistedSymbolSnapshot(
            symbol=str(row.get("symbol", "")),
            price=_to_float(row.get("price")),
            sma=_to_float(row.get("sma")),
            action=str(row.get("action", "SNAPSHOT_ONLY")),
            holding=bool(row.get("holding", 0)),
            quantity=float(row.get("quantity", 0.0) or 0.0),
            market_value=float(row.get("market_value", 0.0) or 0.0),
            ml_probability_up=_to_float(row.get("ml_probability_up")),
            ml_confidence=_to_float(row.get("ml_confidence")),
            ml_training_rows=int(row["ml_training_rows"]) if row.get("ml_training_rows") is not None else None,
            holding_minutes=_to_float(row.get("holding_minutes")),
            error=str(row.get("error")) if row.get("error") else None,
        )
        symbols.append(item)
        if item.holding or item.quantity > 0:
            positions[item.symbol] = SimpleNamespace(
                qty=item.quantity,
                market_value=item.market_value,
                avg_entry_price=_to_float(row.get("avg_entry_price")),
                unrealized_pl=_to_float(row.get("unrealized_pl")),
            )

    snapshot = PersistedBotSnapshot(
        timestamp_utc=str(latest_run.get("timestamp_utc", "")),
        cash=float(latest_run.get("cash", 0.0) or 0.0),
        buying_power=float(latest_run.get("buying_power", 0.0) or 0.0),
        equity=float(latest_run.get("equity", 0.0) or 0.0),
        last_equity=float(latest_run.get("last_equity", 0.0) or 0.0),
        daily_pnl=float(latest_run.get("daily_pnl", 0.0) or 0.0),
        kill_switch_triggered=bool(latest_run.get("kill_switch_triggered", 0)),
        positions=positions,
        symbols=symbols,
    )
    return snapshot, True


def _load_recent_orders(storage: BotStorage, session_id: str | None = None) -> list[PersistedOrderSnapshot]:
    rows = storage.get_order_history(limit=50, session_id=session_id)
    return [
        PersistedOrderSnapshot(
            order_id=str(row.get("order_id", "")),
            observed_at_utc=str(row.get("observed_at_utc", "")) or None,
            submitted_at=str(row.get("submitted_at", "")) or None,
            filled_at=str(row.get("filled_at", "")) or None,
            symbol=str(row.get("symbol", "")),
            side=str(row.get("side", "")),
            status=str(row.get("status", "")),
            qty=_to_float(row.get("qty")),
            filled_qty=_to_float(row.get("filled_qty")),
            filled_avg_price=_to_float(row.get("filled_avg_price")),
            notional=_to_float(row.get("notional")),
        )
        for row in rows
    ]


def _execution_reason(payload: dict[str, object]) -> str | None:
    event = str(payload.get("event", ""))
    if event == "order.submitted":
        return "Order submitted"
    if event == "order.filled":
        return "Order filled"
    if event == "order.partial_fill":
        fill_rate = _to_float(payload.get("fill_rate"))
        return f"Partial fill {fill_rate:.0%}" if fill_rate is not None else "Partial fill"
    if event == "order.rejected":
        reason = str(payload.get("reason", "")).replace("_", " ").strip()
        return reason or "Order rejected"
    if event == "risk.check":
        block_reason = payload.get("block_reason")
        if block_reason:
            return f"blocked by {str(block_reason).replace('_', ' ')}"
        return "Allowed by risk checks"
    if event == "signal.evaluated":
        rejection = payload.get("rejection")
        if rejection:
            rejection_key = str(rejection).strip().lower()
            if rejection_key == "trend_filter":
                return "filtered by trend filter"
            if rejection_key == "atr_filter":
                return "filtered by ATR filter"
            if rejection_key == "no_signal":
                return "no actionable signal"
            return str(rejection).replace("_", " ")
        action = str(payload.get("action", "")).upper()
        if action == "HOLD":
            return "no actionable signal"
        return "Actionable signal"
    return None


def _load_recent_execution_activity(
    last_cycle_report: PersistedCycleReport | None,
    session_id: str | None = None,
    started_at_utc: str | None = None,
    ended_before_utc: str | None = None,
    limit: int = 5,
) -> tuple[PersistedExecutionActivity, ...]:
    log_dir = _latest_log_dir()
    if log_dir is None:
        return ()
    latest_cycle_ts = last_cycle_report.decision_timestamp if last_cycle_report is not None else None

    events: list[dict[str, object]] = []
    events.extend(_load_jsonl_events(log_dir / "execution.jsonl", {"order.submitted", "order.filled", "order.partial_fill", "order.rejected"}, limit=20, session_id=session_id, started_at_utc=started_at_utc, ended_before_utc=ended_before_utc))
    events.extend(_load_jsonl_events(log_dir / "risk.jsonl", {"risk.check"}, limit=20, session_id=session_id, started_at_utc=started_at_utc, ended_before_utc=ended_before_utc))
    events.extend(_load_jsonl_events(log_dir / "signals.jsonl", {"signal.evaluated"}, limit=40, session_id=session_id, started_at_utc=started_at_utc, ended_before_utc=ended_before_utc))

    normalized: list[tuple[datetime, PersistedExecutionActivity]] = []
    for payload in events:
        event = str(payload.get("event", ""))
        side = ""
        qty = None
        price = None
        decision_timestamp = str(payload.get("decision_ts", "")) or None
        if decision_timestamp is None:
            decision_timestamp = _decision_timestamp_from_trace(payload.get("trace"))
        allowed = None

        if event.startswith("order."):
            side = str(payload.get("side", "")).upper()
            qty = _to_float(payload.get("fill_qty"))
            if qty is None:
                qty = _to_float(payload.get("filled_qty"))
            if qty is None:
                qty = _to_float(payload.get("requested_qty"))
            if qty is None:
                qty = _to_float(payload.get("qty"))
            price = _to_float(payload.get("fill_price"))
            if price is None:
                price = _to_float(payload.get("live_price"))
        elif event == "risk.check":
            side = str(payload.get("action", "")).upper()
            qty = None
            price = _to_float(payload.get("live_price"))
            allowed = bool(payload.get("allowed"))
        elif event == "signal.evaluated":
            side = str(payload.get("action", "")).upper()
            qty = None
            price = _to_float(payload.get("bar_close"))
            if side != "HOLD":
                continue
        if not side:
            continue

        observed_at = str(payload.get("ts", "")) or None
        ts = _parse_iso_utc(observed_at)
        if ts is None:
            continue

        priority = 4
        if latest_cycle_ts and decision_timestamp == latest_cycle_ts:
            priority = 2
            if event == "risk.check" and allowed is False and side in {"BUY", "SELL"}:
                priority = 0
            elif event == "risk.check" and allowed is True and side in {"BUY", "SELL"}:
                priority = 1
            elif event.startswith("order.") and side in {"BUY", "SELL"}:
                priority = 1
            elif event == "signal.evaluated" and side == "HOLD":
                priority = 3
        normalized.append(
            (
                (priority, -ts.timestamp()),
                PersistedExecutionActivity(
                    observed_at_utc=observed_at,
                    decision_timestamp=decision_timestamp,
                    symbol=str(payload.get("symbol", "")),
                    side=side,
                    qty=qty,
                    price=price,
                    reason=_execution_reason(payload),
                    source_event=event,
                    allowed=allowed,
                ),
            )
        )

    normalized.sort(key=lambda item: item[0])
    deduped: list[PersistedExecutionActivity] = []
    seen: set[tuple[str | None, str, str, str | None, str | None]] = set()
    for _, item in normalized:
        key = (item.decision_timestamp, item.symbol, item.side, item.source_event, item.reason)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(item)
        if len(deduped) >= limit:
            break
    return tuple(deduped)


def _load_latest_cycle_risk_checks(
    last_cycle_report: PersistedCycleReport | None,
    session_id: str | None = None,
    started_at_utc: str | None = None,
    ended_before_utc: str | None = None,
) -> tuple[PersistedRiskCheck, ...]:
    if last_cycle_report is None:
        return ()
    log_dir = _latest_log_dir()
    if log_dir is None:
        return ()
    rows = _load_jsonl_events(
        log_dir / "risk.jsonl",
        {"risk.check"},
        session_id=session_id,
        started_at_utc=started_at_utc,
        ended_before_utc=ended_before_utc,
    )
    target = last_cycle_report.decision_timestamp
    checks: list[PersistedRiskCheck] = []
    for payload in rows:
        decision_timestamp = _decision_timestamp_from_trace(payload.get("trace"))
        if decision_timestamp != target:
            continue
        action = str(payload.get("action", "")).upper()
        if action not in {"BUY", "SELL"}:
            continue
        checks.append(
            PersistedRiskCheck(
                observed_at_utc=str(payload.get("ts", "")) or None,
                decision_timestamp=decision_timestamp,
                symbol=str(payload.get("symbol", "")),
                action=action,
                allowed=bool(payload.get("allowed")),
                block_reason=str(payload.get("block_reason", "")) or None,
                detail=str(payload.get("detail", "")) or None,
            )
        )
    return tuple(checks)


def _load_session_block_reason_counts(
    *,
    session_id: str | None = None,
    started_at_utc: str | None = None,
    ended_before_utc: str | None = None,
) -> tuple[tuple[str, int], ...]:
    log_dir = _latest_log_dir()
    if log_dir is None:
        return ()
    counts: dict[str, int] = {}
    for payload in _load_jsonl_events(
        log_dir / "risk.jsonl",
        {"risk.check"},
        session_id=session_id,
        started_at_utc=started_at_utc,
        ended_before_utc=ended_before_utc,
    ):
        action = str(payload.get("action", "")).upper()
        if action not in {"BUY", "SELL"}:
            continue
        if bool(payload.get("allowed")):
            continue
        reason = str(payload.get("block_reason", "")) or "unknown_block"
        counts[reason] = counts.get(reason, 0) + 1
    return tuple(sorted(counts.items(), key=lambda item: (-item[1], item[0])))


def _load_latest_signal_rows(
    *,
    session_id: str | None = None,
    started_at_utc: str | None = None,
    ended_before_utc: str | None = None,
) -> tuple[PersistedSignalDecision, ...]:
    log_dir = _latest_log_dir()
    if log_dir is None:
        return ()
    rows = _load_jsonl_events(
        log_dir / "signals.jsonl",
        {"signal.evaluated"},
        session_id=session_id,
        started_at_utc=started_at_utc,
        ended_before_utc=ended_before_utc,
    )
    if not rows:
        return ()

    latest_by_symbol: dict[str, PersistedSignalDecision] = {}
    latest_ts_by_symbol: dict[str, datetime] = {}
    for payload in rows:
        symbol = str(payload.get("symbol", ""))
        observed_at = str(payload.get("ts", "")) or None
        ts = _parse_iso_utc(observed_at)
        if not symbol or ts is None:
            continue
        if symbol in latest_ts_by_symbol and ts <= latest_ts_by_symbol[symbol]:
            continue
        latest_ts_by_symbol[symbol] = ts
        latest_by_symbol[symbol] = PersistedSignalDecision(
            observed_at_utc=observed_at,
            decision_timestamp=str(payload.get("decision_ts", "")) or None,
            symbol=symbol,
            action=str(payload.get("action", "")),
            price=_to_float(payload.get("bar_close")),
            sma=_to_float(payload.get("sma")),
            deviation_pct=_to_float(payload.get("deviation_pct")),
            trend_sma=_to_float(payload.get("trend_sma")),
            above_trend_sma=payload.get("above_trend_sma"),
            atr_pct=_to_float(payload.get("atr_pct")),
            atr_percentile=_to_float(payload.get("atr_percentile")),
            volume_ratio=_to_float(payload.get("volume_ratio")),
            strategy_mode=str(payload.get("strategy_mode", "")) or None,
            final_signal_reason=str(payload.get("final_signal_reason", "")) or None,
            decision_summary=str(payload.get("decision_summary", "")) or None,
            hybrid_branch_active=str(payload.get("hybrid_branch_active", "")) or None,
            hybrid_entry_branch=str(payload.get("hybrid_entry_branch", "")) or None,
            hybrid_regime_branch=str(payload.get("hybrid_regime_branch", "")) or None,
            mr_signal=str(payload.get("mr_signal", "")) or None,
            trend_sma_slope=_to_float(payload.get("trend_sma_slope")),
            entry_reference_price=_to_float(payload.get("entry_reference_price")),
            exit_reference_price=_to_float(payload.get("exit_reference_price")),
            stop_reference_price=_to_float(payload.get("stop_reference_price")),
            vwap=_to_float(payload.get("vwap")),
            adx=_to_float(payload.get("adx")),
            regime_state=str(payload.get("regime_state", "")) or None,
            rejection=str(payload.get("rejection", "")) or None,
            trend_filter=str(payload.get("trend_filter", "")) or None,
            atr_filter=str(payload.get("atr_filter", "")) or None,
            window_open=payload.get("window_open"),
            holding=payload.get("holding"),
        )
    return tuple(sorted(latest_by_symbol.values(), key=lambda item: item.symbol))


def _load_recent_narrations(
    session_id: str | None = None,
    started_at_utc: str | None = None,
    ended_before_utc: str | None = None,
    limit: int = 10,
) -> tuple[PersistedNarration, ...]:
    """
    Load and narrate the most recent actionable events from today's JSONL logs.

    Narrates:
        - BUY / SELL signals that cleared all filters (signals.jsonl)
        - Risk checks that blocked a trade (risk.jsonl)
        - Position exits (positions.jsonl)

    HOLD signals are excluded — they are too frequent and low-value for the panel.
    """
    log_dir = _latest_log_dir()
    if log_dir is None:
        return ()

    raw_events: list[dict] = []

    # Fetch recent signal events; filter to only actionable BUY/SELL decisions
    for payload in _load_jsonl_events(log_dir / "signals.jsonl", {"signal.evaluated"}, limit=60, session_id=session_id, started_at_utc=started_at_utc, ended_before_utc=ended_before_utc):
        if str(payload.get("action", "")).upper() in {"BUY", "SELL"}:
            raw_events.append(payload)

    # Fetch risk blocks only
    for payload in _load_jsonl_events(log_dir / "risk.jsonl", {"risk.check"}, limit=40, session_id=session_id, started_at_utc=started_at_utc, ended_before_utc=ended_before_utc):
        if payload.get("allowed") is False:
            raw_events.append(payload)

    # Fetch position closes
    raw_events.extend(
        _load_jsonl_events(log_dir / "positions.jsonl", {"position.closed"}, limit=20, session_id=session_id, started_at_utc=started_at_utc, ended_before_utc=ended_before_utc)
    )

    # Sort newest first
    def _ts_key(p: dict) -> datetime:
        return _parse_iso_utc(str(p.get("ts", ""))) or datetime.min.replace(tzinfo=timezone.utc)

    raw_events.sort(key=_ts_key, reverse=True)

    results: list[PersistedNarration] = []
    seen: set[tuple[str | None, str, str]] = set()

    for payload in raw_events:
        event_type = str(payload.get("event", ""))
        symbol = str(payload.get("symbol", "")).upper()
        ts_str = str(payload.get("ts", "")) or None

        dedup_key = (ts_str, symbol, event_type)
        if dedup_key in seen:
            continue
        seen.add(dedup_key)

        narration = generate_trade_narration(payload)

        if event_type == "signal.evaluated":
            kind = "signal"
        elif event_type == "risk.check":
            kind = "risk_block"
        elif event_type == "position.closed":
            kind = "exit"
        else:
            kind = "other"

        results.append(PersistedNarration(
            timestamp_utc=ts_str,
            symbol=symbol,
            narration=narration,
            event_type=kind,
        ))

        if len(results) >= limit:
            break

    return tuple(results)


def _qualifies_as_near_miss(event: dict) -> bool:
    """
    Return True if a signal.evaluated event represents a genuine near-miss:
    a setup where the bot evaluated an entry but a specific filter rejected it.

    Excludes plain "no_signal" HOLD events — those have no directional intent
    and are not useful to surface as near-misses.
    """
    if str(event.get("action", "")).upper() != "HOLD":
        return False
    rejection = str(event.get("rejection") or "").lower().strip()
    trend_filter = str(event.get("trend_filter") or "").lower().strip()
    atr_filter = str(event.get("atr_filter") or "").lower().strip()
    return (
        rejection in {"trend_filter", "atr_filter"}
        or trend_filter == "reject"
        or atr_filter == "reject"
    )


def _load_recent_near_misses(
    session_id: str | None = None,
    started_at_utc: str | None = None,
    ended_before_utc: str | None = None,
    limit: int = 5,
) -> tuple[PersistedNearMiss, ...]:
    """
    Load the most recent near-miss signal events from today's signals.jsonl.

    A near-miss is a HOLD evaluation where trend_filter or atr_filter
    explicitly rejected a potential entry — distinct from a plain "no signal"
    hold where there was no directional setup at all.

    Deduplicates by (decision_ts, symbol) so each symbol appears at most once
    per bar cycle in the output.
    """
    log_dir = _latest_log_dir()
    if log_dir is None:
        return ()

    raw: list[dict] = []
    for payload in _load_jsonl_events(log_dir / "signals.jsonl", {"signal.evaluated"}, limit=120, session_id=session_id, started_at_utc=started_at_utc, ended_before_utc=ended_before_utc):
        if _qualifies_as_near_miss(payload):
            raw.append(payload)

    # Sort newest first
    def _ts_key(p: dict) -> datetime:
        return _parse_iso_utc(str(p.get("ts", ""))) or datetime.min.replace(tzinfo=timezone.utc)

    raw.sort(key=_ts_key, reverse=True)

    results: list[PersistedNearMiss] = []
    # Deduplicate: one entry per (decision_ts, symbol) — keep the newest occurrence
    seen: set[tuple[str | None, str]] = set()

    for payload in raw:
        symbol = str(payload.get("symbol", "")).upper()
        decision_ts = str(payload.get("decision_ts", "")) or None
        ts_str = str(payload.get("ts", "")) or None

        dedup_key = (decision_ts, symbol)
        if dedup_key in seen:
            continue
        seen.add(dedup_key)

        rejection = str(payload.get("rejection") or "").lower().strip() or None
        narration = generate_near_miss_narration(payload)

        results.append(PersistedNearMiss(
            timestamp_utc=ts_str,
            symbol=symbol,
            narration=narration,
            rejection=rejection,
        ))

        if len(results) >= limit:
            break

    return tuple(results)


def _load_drilldown_candidates(
    strategy_mode: str | None,
    session_id: str | None = None,
    started_at_utc: str | None = None,
    ended_before_utc: str | None = None,
    limit: int = 40,
) -> tuple[DrilldownEvent, ...]:
    """
    Load recent decision events from today's JSONL logs and normalise them
    into DrilldownEvent instances for the drilldown inspector panel.

    Covers signal.evaluated, risk.check, and position.closed events.
    Returns newest-first, deduplicated by (ts, symbol, event_type).
    """
    log_dir = _latest_log_dir()
    if log_dir is None:
        return ()

    raw: list[dict] = []
    raw.extend(_load_jsonl_events(log_dir / "signals.jsonl", {"signal.evaluated"}, limit=60, session_id=session_id, started_at_utc=started_at_utc, ended_before_utc=ended_before_utc))
    raw.extend(_load_jsonl_events(log_dir / "risk.jsonl", {"risk.check"}, limit=40, session_id=session_id, started_at_utc=started_at_utc, ended_before_utc=ended_before_utc))
    raw.extend(_load_jsonl_events(log_dir / "positions.jsonl", {"position.closed"}, limit=20, session_id=session_id, started_at_utc=started_at_utc, ended_before_utc=ended_before_utc))

    def _ts_key(p: dict) -> datetime:
        return _parse_iso_utc(str(p.get("ts", ""))) or datetime.min.replace(tzinfo=timezone.utc)

    raw.sort(key=_ts_key, reverse=True)

    results: list[DrilldownEvent] = []
    seen: set[tuple[str, str, str]] = set()

    for payload in raw:
        event_type = str(payload.get("event", ""))
        symbol = str(payload.get("symbol", "")).upper()
        ts_str = str(payload.get("ts", "")) or None

        dedup_key = (ts_str or "", symbol, event_type)
        if dedup_key in seen:
            continue
        seen.add(dedup_key)

        action = str(payload.get("action", "")).upper()
        allowed: bool | None = None
        if event_type == "risk.check":
            raw_allowed = payload.get("allowed")
            if raw_allowed is not None:
                allowed = bool(raw_allowed)

        results.append(DrilldownEvent(
            timestamp_utc=ts_str,
            symbol=symbol,
            event_type=event_type,
            action=action,
            strategy_mode=strategy_mode,
            allowed=allowed,
            block_reason=str(payload.get("block_reason", "")) or None,
            rejection=str(payload.get("rejection", "")) or None,
            trend_filter=str(payload.get("trend_filter", "")) or None,
            atr_filter=str(payload.get("atr_filter", "")) or None,
            above_trend_sma=payload.get("above_trend_sma"),
            deviation_pct=_to_float(payload.get("deviation_pct")),
            atr_pct=_to_float(payload.get("atr_pct")),
            atr_percentile=_to_float(payload.get("atr_percentile")),
            ml_prob=_to_float(payload.get("ml_prob")),
            pnl_usd=_to_float(payload.get("pnl_usd")),
            exit_reason=str(payload.get("exit_reason", "")) or None,
            bar_close=_to_float(payload.get("bar_close")),
            sma=_to_float(payload.get("sma")),
            volume_ratio=_to_float(payload.get("volume_ratio")),
            window_open=payload.get("window_open"),
            holding=payload.get("holding"),
        ))

        if len(results) >= limit:
            break

    return tuple(results)


def _pretty_activity_text(value: object | None, fallback: str = "unknown") -> str:
    text = str(value or "").strip().replace("_", " ")
    return text if text else fallback


def _load_recent_activity(
    session_id: str | None = None,
    started_at_utc: str | None = None,
    ended_before_utc: str | None = None,
    limit: int = 12,
) -> tuple[PersistedActivityItem, ...]:
    log_dir = _latest_log_dir()
    if log_dir is None:
        return ()

    event_specs = {
        log_dir / "execution.jsonl": {"order.submitted", "order.filled", "order.partial_fill", "order.rejected"},
        log_dir / "risk.jsonl": {"risk.check", "cycle.summary", "execution.error", "kill_switch.warning", "kill_switch.triggered"},
        log_dir / "signals.jsonl": {"signal.evaluated"},
        log_dir / "bars.jsonl": {"bar.stale_warning"},
        log_dir / "positions.jsonl": {"position.opened", "position.closed"},
    }

    raw_events: list[dict[str, object]] = []
    for path, names in event_specs.items():
        raw_events.extend(
            _load_jsonl_events(
                path,
                names,
                limit=40,
                session_id=session_id,
                started_at_utc=started_at_utc,
                ended_before_utc=ended_before_utc,
            )
        )

    def _ts_key(payload: dict[str, object]) -> datetime:
        return _parse_iso_utc(str(payload.get("ts", ""))) or datetime.min.replace(tzinfo=timezone.utc)

    raw_events.sort(key=_ts_key, reverse=True)

    items: list[PersistedActivityItem] = []
    seen: set[tuple[str | None, str, str | None, str | None]] = set()
    for payload in raw_events:
        event = str(payload.get("event", ""))
        observed_at = str(payload.get("ts", "")) or None
        symbol = str(payload.get("symbol", "")).upper() or None
        decision_timestamp = (
            str(payload.get("decision_ts", "")) or None
            or _decision_timestamp_from_trace(payload.get("trace"))
        )
        dedupe_key = (observed_at, event, symbol, decision_timestamp)
        if dedupe_key in seen:
            continue
        seen.add(dedupe_key)

        title = ""
        body = ""
        level = "info"

        if event == "cycle.summary":
            processed = bool(payload.get("processed_bar"))
            skip_reason = _pretty_activity_text(payload.get("skip_reason"))
            buy_count = int(payload.get("buy_signals", 0) or 0)
            sell_count = int(payload.get("sell_signals", 0) or 0)
            hold_count = int(payload.get("hold_signals", 0) or 0)
            order_count = int(payload.get("orders_submitted", 0) or 0)
            title = "Decision cycle recorded" if processed else "Cycle skipped"
            body = (
                f"{buy_count} buys, {sell_count} sells, {hold_count} holds, {order_count} orders. "
                f"Outcome: {skip_reason}."
            )
            level = "warn" if not processed or order_count == 0 and (buy_count or sell_count) else "info"
        elif event == "risk.check":
            action = str(payload.get("action", "")).upper() or "CHECK"
            allowed = bool(payload.get("allowed"))
            if allowed:
                title = f"{action} cleared risk checks"
                body = f"{symbol or 'Signal'} passed the current live safety gates."
                level = "info"
            else:
                block_reason = _pretty_activity_text(payload.get("block_reason"))
                detail = str(payload.get("detail", "")).strip()
                title = f"{action} blocked"
                body = f"{symbol or 'Signal'} was blocked by {block_reason}."
                if detail:
                    body = f"{body} {detail}"
                level = "warn"
        elif event == "signal.evaluated":
            action = str(payload.get("action", "")).upper() or "HOLD"
            rejection = str(payload.get("rejection", "")).strip()
            if action in {"BUY", "SELL"}:
                title = f"{action} signal detected"
                body = f"{symbol or 'Symbol'} produced an actionable {action.lower()} signal on the latest saved bar."
                level = "info"
            elif rejection:
                title = "Setup rejected"
                body = f"{symbol or 'Symbol'} was evaluated but held because {_pretty_activity_text(rejection)}."
                level = "warn"
            else:
                title = "No actionable signal"
                body = f"{symbol or 'Symbol'} was evaluated and no trade was taken."
                level = "info"
        elif event == "order.submitted":
            side = str(payload.get("side", "")).upper() or "ORDER"
            qty = payload.get("qty")
            title = f"{side} order submitted"
            body = f"{symbol or 'Order'} submitted"
            if qty is not None:
                body = f"{body} for {float(qty):.2f} shares."
            else:
                body = f"{body}."
            level = "info"
        elif event == "order.filled":
            side = str(payload.get("side", "")).upper() or "ORDER"
            fill_price = _to_float(payload.get("fill_price"))
            title = f"{side} order filled"
            body = f"{symbol or 'Order'} filled"
            if fill_price is not None:
                body = f"{body} at ${fill_price:,.2f}."
            else:
                body = f"{body}."
            level = "info"
        elif event == "order.partial_fill":
            title = "Partial fill"
            body = f"{symbol or 'Order'} filled partially and may still be working."
            level = "warn"
        elif event == "order.rejected":
            reason = _pretty_activity_text(payload.get("reason"))
            title = "Order rejected"
            body = f"{symbol or 'Order'} was rejected: {reason}."
            level = "err"
        elif event == "position.opened":
            title = "Position opened"
            body = f"{symbol or 'Position'} is now open."
            level = "info"
        elif event == "position.closed":
            exit_reason = _pretty_activity_text(payload.get("exit_reason"))
            pnl = _to_float(payload.get("pnl_usd"))
            body = f"{symbol or 'Position'} closed via {exit_reason}."
            if pnl is not None:
                body = f"{body} Realized PnL: ${pnl:,.2f}."
            title = "Position closed"
            level = "info"
        elif event == "bar.stale_warning":
            age_s = _to_float(payload.get("bar_age_s"))
            title = "Bar data arrived late"
            body = (
                f"{symbol or 'A symbol'} latest completed bar was stale"
                + (f" by {age_s:.0f}s." if age_s is not None else ".")
            )
            level = "warn"
        elif event == "execution.error":
            action = str(payload.get("action", "")).upper() or "EXECUTION"
            error = str(payload.get("error", "")).strip() or "unknown error"
            title = f"{action} execution error"
            body = f"{symbol or 'Execution'} hit an error: {error}."
            level = "err"
        elif event in {"kill_switch.warning", "kill_switch.triggered"}:
            reason = _pretty_activity_text(payload.get("reason"))
            title = "Kill switch warning" if event == "kill_switch.warning" else "Kill switch triggered"
            body = f"Risk control update: {reason}."
            level = "warn" if event == "kill_switch.warning" else "err"

        if not title or not body:
            continue

        items.append(
            PersistedActivityItem(
                observed_at_utc=observed_at,
                event_type=event,
                title=title,
                body=body,
                level=level,
                symbol=symbol,
                decision_timestamp=decision_timestamp,
            )
        )
        if len(items) >= limit:
            break

    return tuple(items)


def _resolve_dashboard_database_path(
    latest_startup_config: PersistedStartupConfig | None,
) -> Path:
    configured_path = resolve_explicit_database_path()

    if latest_startup_config is None:
        return configured_path or resolve_database_path()

    artifact_path = resolve_database_path(latest_startup_config.db_path)
    authorities_conflict = (
        configured_path is not None
        and os.path.normcase(str(configured_path))
        != os.path.normcase(str(artifact_path))
    )
    if authorities_conflict:
        raise RuntimeError(
            "Conflicting database authorities: startup metadata resolves to "
            f"{artifact_path}, but {DATABASE_PATH_ENV} resolves to {configured_path}. "
            "Refusing to initialize dashboard storage."
        )
    return artifact_path


def load_dashboard_state() -> DashboardState:
    startup_configs = _load_startup_configs()
    latest_startup_config = startup_configs[-1] if startup_configs else None
    db_path = _resolve_dashboard_database_path(latest_startup_config)
    storage = BotStorage(db_path, read_only=True)
    latest_run = storage.get_latest_run()
    latest_snapshot, has_latest_snapshot = _build_snapshot(storage)
    latest_run_session_value = (
        latest_run.get("session_id") if latest_run is not None else None
    )
    latest_run_session_id = (
        latest_run_session_value.strip()
        if isinstance(latest_run_session_value, str)
        and latest_run_session_value.strip()
        else None
    )
    latest_run_environment = (
        latest_run.get("environment") if latest_run is not None else None
    )
    startup_config = _select_startup_config_for_session(
        startup_configs=startup_configs,
        snapshot=latest_snapshot,
        has_persisted_snapshot=has_latest_snapshot,
        last_cycle_report=None,
        latest_run_session_id=latest_run_session_id,
        latest_run_environment=latest_run_environment,
    )
    session_id = (
        startup_config.session_id
        if startup_config is not None
        else latest_run_session_id
    )
    session_started_at = (
        startup_config.started_at_utc
        if startup_config is not None
        else (
            str(latest_run.get("timestamp_utc", "")) or None
            if latest_run is not None
            else None
        )
    )
    session_ended_before = _next_startup_started_at(startup_configs, startup_config)
    has_session_scope = session_id is not None
    if has_session_scope:
        last_cycle_report = _load_latest_cycle_report(
            session_id=session_id,
            started_at_utc=session_started_at,
            ended_before_utc=session_ended_before,
        )
        feed_status = _load_feed_status(
            session_id=session_id,
            started_at_utc=session_started_at,
            ended_before_utc=session_ended_before,
        )
    else:
        last_cycle_report = None
        feed_status = "session metadata unavailable"
    snapshot, has_persisted_snapshot = _build_snapshot(storage, session_id=session_id)
    recent_orders = (
        _load_recent_orders(storage, session_id=session_id)
        if has_session_scope
        else []
    )
    ignored_snapshot_symbols: tuple[str, ...] = ()
    symbol_state_status = "no_runtime_config"
    if (
        startup_config is None
        and has_persisted_snapshot
    ):
        symbol_state_status = "unmatched_runtime_metadata_snapshot_preserved"
    if startup_config is not None:
        symbol_state_status = "cleared_runtime_symbol_state"
        latest_symbols = tuple(normalize_symbols(item.symbol for item in latest_snapshot.symbols))
        if latest_symbols and not symbols_match(startup_config.symbols, latest_symbols):
            ignored_snapshot_symbols = latest_symbols
            if not has_persisted_snapshot:
                symbol_state_status = "stale_persisted_symbols_ignored"
        elif has_persisted_snapshot:
            symbol_state_status = "current_runtime_symbols"
    session_warnings = _build_session_warnings(
        startup_config=startup_config,
        latest_startup_config=latest_startup_config,
        snapshot=snapshot,
        has_persisted_snapshot=has_persisted_snapshot,
        last_cycle_report=last_cycle_report,
        ignored_snapshot_symbols=ignored_snapshot_symbols,
    )
    _strategy_mode = startup_config.strategy_mode if startup_config is not None else None
    if has_session_scope:
        latest_signal_rows = _load_latest_signal_rows(
            session_id=session_id,
            started_at_utc=session_started_at,
            ended_before_utc=session_ended_before,
        )
        latest_cycle_risk_checks = _load_latest_cycle_risk_checks(
            last_cycle_report,
            session_id=session_id,
            started_at_utc=session_started_at,
            ended_before_utc=session_ended_before,
        )
        session_block_reason_counts = _load_session_block_reason_counts(
            session_id=session_id,
            started_at_utc=session_started_at,
            ended_before_utc=session_ended_before,
        )
        session_first_prices = storage.get_session_first_prices(
            session_id=session_id
        )
        recent_narrations = _load_recent_narrations(
            session_id=session_id,
            started_at_utc=session_started_at,
            ended_before_utc=session_ended_before,
        )
        recent_near_misses = _load_recent_near_misses(
            session_id=session_id,
            started_at_utc=session_started_at,
            ended_before_utc=session_ended_before,
        )
        drilldown_candidates = _load_drilldown_candidates(
            _strategy_mode,
            session_id=session_id,
            started_at_utc=session_started_at,
            ended_before_utc=session_ended_before,
        )
        recent_activity = _load_recent_activity(
            session_id=session_id,
            started_at_utc=session_started_at,
            ended_before_utc=session_ended_before,
        )
    else:
        latest_signal_rows = ()
        latest_cycle_risk_checks = ()
        session_block_reason_counts = ()
        session_first_prices = {}
        recent_narrations = ()
        recent_near_misses = ()
        drilldown_candidates = ()
        recent_activity = ()
    log_dir = _latest_log_dir()
    drift_report = None
    if log_dir is not None and has_session_scope:
        signal_events = list(
            _load_jsonl_events(
                log_dir / "signals.jsonl",
                {"signal.evaluated"},
                session_id=session_id,
                started_at_utc=session_started_at,
                ended_before_utc=session_ended_before,
            )
        )
        bar_events = list(
            _load_jsonl_events(
                log_dir / "bars.jsonl",
                {"bar.received", "bar.stale_warning"},
                session_id=session_id,
                started_at_utc=session_started_at,
                ended_before_utc=session_ended_before,
            )
        )
        risk_events = list(
            _load_jsonl_events(
                log_dir / "risk.jsonl",
                {"risk.check"},
                session_id=session_id,
                started_at_utc=session_started_at,
                ended_before_utc=session_ended_before,
            )
        )
        position_events = list(
            _load_jsonl_events(
                log_dir / "positions.jsonl",
                {"position.closed"},
                session_id=session_id,
                started_at_utc=session_started_at,
                ended_before_utc=session_ended_before,
            )
        )
        if signal_events or bar_events or risk_events or position_events:
            drift_report = build_drift_report(
                signal_events=signal_events,
                bar_events=bar_events,
                risk_events=risk_events,
                position_events=position_events,
            )
    return DashboardState(
        startup_config=startup_config,
        storage=storage,
        snapshot=snapshot,
        recent_orders=recent_orders,
        last_cycle_report=last_cycle_report,
        feed_status=feed_status,
        has_persisted_snapshot=has_persisted_snapshot,
        has_persisted_startup_config=startup_config is not None,
        session_warnings=session_warnings,
        symbol_state_status=symbol_state_status,
        ignored_snapshot_symbols=ignored_snapshot_symbols,
        latest_signals={item.symbol: item for item in latest_signal_rows},
        recent_execution_activity=_load_recent_execution_activity(last_cycle_report),
        latest_signal_rows=latest_signal_rows,
        latest_cycle_risk_checks=latest_cycle_risk_checks,
        session_block_reason_counts=session_block_reason_counts,
        session_first_prices=session_first_prices,
        recent_narrations=recent_narrations,
        recent_near_misses=recent_near_misses,
        drilldown_candidates=drilldown_candidates,
        recent_activity=recent_activity,
        drift_report=drift_report,
    )

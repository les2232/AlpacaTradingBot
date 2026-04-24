from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

from avatar import generate_near_miss_narration, generate_trade_narration
from drift_monitor import BaselineProfile, DriftAlert, DriftReport, LiveMetrics, build_drift_report
from storage import BotStorage
from symbol_state import format_symbol_list, normalize_symbols, symbols_match


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


def _log_root() -> Path:
    return Path(os.getenv("BOT_LOG_ROOT", "logs"))


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


def _startup_config_from_payload(payload: dict[str, object], artifact_path: Path | None = None) -> PersistedStartupConfig | None:
    if not isinstance(payload, dict):
        return None
    symbols = normalize_symbols(payload.get("symbols", []))
    return PersistedStartupConfig(
        session_id=str(payload.get("session_id", "")) or None,
        started_at_utc=str(payload.get("started_at_utc", "")) or None,
        launch_mode=str(payload.get("launch_mode", "")),
        execution_enabled=bool(payload.get("execution_enabled", False)),
        paper=bool(payload.get("paper", False)),
        account_mode=str(payload.get("account_mode", "")),
        strategy_mode=str(payload.get("strategy_mode", "")),
        bar_timeframe_minutes=int(payload.get("bar_timeframe_minutes", 0) or 0),
        sma_bars=int(payload.get("sma_bars", 0) or 0),
        symbols=symbols,
        symbol_count=int(payload.get("symbol_count", len(symbols)) or len(symbols)),
        runtime_config_path=str(payload.get("runtime_config_path", "")) or None,
        runtime_overrides=tuple(str(item) for item in payload.get("runtime_overrides", []) if str(item)),
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
        mean_reversion_trend_filter=bool(payload.get("mean_reversion_trend_filter", False)),
        mean_reversion_trend_slope_filter=bool(payload.get("mean_reversion_trend_slope_filter", False)),
        mean_reversion_stop_pct=float(payload.get("mean_reversion_stop_pct", 0.0) or 0.0),
        db_path=str(payload.get("db_path", "bot_history.db") or "bot_history.db"),
        historical_feed=str(payload.get("historical_feed", "") or ""),
        live_feed=str(payload.get("live_feed", "") or ""),
        latest_bar_feed=str(payload.get("latest_bar_feed", "") or ""),
        bar_build_mode=str(payload.get("bar_build_mode", "") or ""),
        apply_updated_bars=bool(payload.get("apply_updated_bars", False)),
        post_bar_reconcile_poll=bool(payload.get("post_bar_reconcile_poll", False)),
        block_trading_until_resync=bool(payload.get("block_trading_until_resync", False)),
        assert_feed_on_startup=bool(payload.get("assert_feed_on_startup", False)),
        log_bar_components=bool(payload.get("log_bar_components", False)),
        resync_status=str(payload.get("resync_status", "")) or None,
        resync_started_at_utc=str(payload.get("resync_started_at_utc", "")) or None,
        resync_completed_at_utc=str(payload.get("resync_completed_at_utc", "")) or None,
        resync_reason_codes=tuple(
            str(item) for item in payload.get("resync_reason_codes", []) if str(item)
        ),
        gate_allows_entries=payload.get("gate_allows_entries")
        if isinstance(payload.get("gate_allows_entries"), bool)
        else None,
        gate_allows_exits=payload.get("gate_allows_exits")
        if isinstance(payload.get("gate_allows_exits"), bool)
        else None,
        runtime_config_approved=payload.get("runtime_config_approved")
        if isinstance(payload.get("runtime_config_approved"), bool)
        else None,
        runtime_config_rejection_reasons=tuple(
            str(item) for item in payload.get("runtime_config_rejection_reasons", []) if str(item)
        ),
        baseline_valid_for_comparison=payload.get("baseline_valid_for_comparison")
        if isinstance(payload.get("baseline_valid_for_comparison"), bool)
        else None,
        baseline_validation_errors=tuple(
            str(item) for item in payload.get("baseline_validation_errors", []) if str(item)
        ),
        artifact_path=str(artifact_path) if artifact_path is not None else None,
    )


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
        payload_session_id = str(payload.get("session_id", "")) or None
        if payload_session_id:
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
    log_dir = _latest_log_dir()
    if log_dir is None:
        return []
    startup_paths = sorted(log_dir.glob("startup_config.*.json"), key=lambda path: path.name)
    if not startup_paths:
        legacy_path = log_dir / "startup_config.json"
        if legacy_path.exists():
            startup_paths = [legacy_path]
    startup_configs: list[PersistedStartupConfig] = []
    for path in startup_paths:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        config = _startup_config_from_payload(payload, artifact_path=path)
        if config is not None:
            startup_configs.append(config)
    startup_configs.sort(key=lambda item: _parse_iso_utc(item.started_at_utc) or datetime.min)
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
        return datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None


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
) -> PersistedStartupConfig | None:
    if not startup_configs:
        return None
    if latest_run_session_id:
        matching = [
            config
            for config in startup_configs
            if (config.session_id or "") == latest_run_session_id
        ]
        if matching:
            return max(matching, key=lambda item: _parse_iso_utc(item.started_at_utc) or datetime.min)
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
            return max(eligible, key=lambda item: _parse_iso_utc(item.started_at_utc) or datetime.min)
    return max(startup_configs, key=lambda item: _parse_iso_utc(item.started_at_utc) or datetime.min)

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


def load_dashboard_state() -> DashboardState:
    startup_configs = _load_startup_configs()
    latest_startup_config = startup_configs[-1] if startup_configs else None
    db_path = Path(latest_startup_config.db_path if latest_startup_config is not None else os.getenv("BOT_DB_PATH", "bot_history.db"))
    storage = BotStorage(db_path)
    latest_run = storage.get_latest_run()
    latest_snapshot, has_latest_snapshot = _build_snapshot(storage)
    startup_config = _select_startup_config_for_session(
        startup_configs=startup_configs,
        snapshot=latest_snapshot,
        has_persisted_snapshot=has_latest_snapshot,
        last_cycle_report=None,
        latest_run_session_id=str(latest_run.get("session_id", "")) or None if latest_run is not None else None,
    )
    session_id = startup_config.session_id if startup_config is not None else None
    session_started_at = startup_config.started_at_utc if startup_config is not None else None
    session_ended_before = _next_startup_started_at(startup_configs, startup_config)
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
    session_id = startup_config.session_id if startup_config is not None else None
    snapshot, has_persisted_snapshot = _build_snapshot(storage, session_id=session_id)
    recent_orders = _load_recent_orders(storage, session_id=session_id)
    ignored_snapshot_symbols: tuple[str, ...] = ()
    symbol_state_status = "no_runtime_config"
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
    session_first_prices = storage.get_session_first_prices(session_id=session_id)
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
    _strategy_mode = startup_config.strategy_mode if startup_config is not None else None
    drilldown_candidates = _load_drilldown_candidates(
        _strategy_mode,
        session_id=session_id,
        started_at_utc=session_started_at,
        ended_before_utc=session_ended_before,
    )
    log_dir = _latest_log_dir()
    drift_report = None
    if log_dir is not None:
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
        recent_activity=_load_recent_activity(
            session_id=session_id,
            started_at_utc=session_started_at,
            ended_before_utc=session_ended_before,
        ),
        drift_report=drift_report,
    )

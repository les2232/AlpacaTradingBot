import logging
import os
import math
import threading
import time
import json
from dataclasses import dataclass, replace
from datetime import datetime, timedelta, timezone
from datetime import time as dt_time
from enum import Enum
from pathlib import Path
from typing import Any
import pandas as pd
import pytz
from dotenv import load_dotenv

try:
    from ml.predict import to_ml_signal as offline_to_ml_signal
except Exception as exc:
    offline_to_ml_signal = None
    _ML_PREDICT_IMPORT_ERROR: Exception | None = exc
else:
    _ML_PREDICT_IMPORT_ERROR = None

from botlog import BotLogger
from storage import BotStorage
from tradeos.brokers import AlpacaBroker, BrokerBar, BrokerOrder, BrokerPosition
from symbol_state import (
    format_symbol_list,
    normalize_symbols,
    symbol_fingerprint,
    symbols_match,
)
from strategy import (
    BOLLINGER_EXIT_CHOICES,
    BOLLINGER_EXIT_MIDDLE_BAND,
    BREAKOUT_EXIT_CHOICES,
    BREAKOUT_EXIT_TARGET_1X_STOP_LOW,
    MEAN_REVERSION_EXIT_CHOICES,
    MEAN_REVERSION_EXIT_SMA,
    ORB_FILTER_CHOICES,
    ORB_FILTER_NONE,
    STRATEGY_MODE_BOLLINGER_SQUEEZE,
    STRATEGY_MODE_BREAKOUT,
    STRATEGY_MODE_CHOICES,
    STRATEGY_MODE_HYBRID,
    STRATEGY_MODE_HYBRID_BB_MR,
    STRATEGY_MODE_MEAN_REVERSION,
    STRATEGY_MODE_ML,
    STRATEGY_MODE_ORB,
    STRATEGY_MODE_SMA,
    STRATEGY_MODE_MOMENTUM_BREAKOUT,
    STRATEGY_MODE_TREND_PULLBACK,
    STRATEGY_MODE_VOLATILITY_EXPANSION,
    MOMENTUM_BREAKOUT_EXIT_CHOICES,
    MOMENTUM_BREAKOUT_EXIT_FIXED_BARS,
    TREND_PULLBACK_EXIT_CHOICES,
    VOLATILITY_EXPANSION_EXIT_CHOICES,
    VOLATILITY_EXPANSION_EXIT_FIXED_BARS,
    Strategy,
    StrategyConfig,
    MlSignal,
    THRESHOLD_MODE_CHOICES,
    THRESHOLD_MODE_STATIC_PCT,
    calculate_bollinger_squeeze_features,
    calculate_opening_range_series,
    calculate_atr_pct_values,
    calculate_atr_percentile_series,
    calculate_adx_series,
    calculate_vwap_series,
    get_capped_breakout_stop_price,
    REGIME_SMA_PERIOD,
    REGIME_TIMEFRAME_MINUTES,
    estimate_atr_percentile_lookback_bars,
    estimate_bollinger_lookback_bars,
    estimate_regime_lookback_bars,
    is_entry_window_open,
    normalize_bollinger_exit_mode,
    normalize_breakout_exit_style,
    normalize_mean_reversion_exit_style,
    normalize_momentum_breakout_exit_style,
    normalize_trend_pullback_exit_style,
    normalize_volatility_expansion_exit_style,
    normalize_orb_filter_mode,
    normalize_strategy_mode,
    normalize_threshold_mode,
    normalize_time_window_mode,
    TIME_WINDOW_CHOICES,
    TIME_WINDOW_FULL_DAY,
)

logger = logging.getLogger(__name__)

Position = BrokerPosition
DATA_FEED_CHOICES = ("iex", "sip", "delayed_sip")
BAR_BUILD_MODE_CHOICES = ("stream_minute_aggregate", "historical_preaggregated")


class StaleMarketDataError(RuntimeError):
    """Raised when the most recent completed bar is too old to trust."""


@dataclass(frozen=True)
class SymbolEventBlackout:
    symbol: str
    start_utc: str
    end_utc: str
    reason: str = "event_blackout"


@dataclass(frozen=True)
class BotConfig:
    symbols: list[str]
    max_usd_per_trade: float
    max_symbol_exposure_usd: float
    max_open_positions: int
    max_daily_loss_usd: float
    sma_bars: int
    bar_timeframe_minutes: int
    excluded_symbols: tuple[str, ...] = ()
    symbol_event_blackouts: tuple[SymbolEventBlackout, ...] = ()
    historical_feed: str = "iex"
    live_feed: str = "iex"
    latest_bar_feed: str = "iex"
    bar_build_mode: str = "stream_minute_aggregate"
    apply_updated_bars: bool = True
    post_bar_reconcile_poll: bool = True
    block_trading_until_resync: bool = True
    assert_feed_on_startup: bool = True
    log_bar_components: bool = True
    paper: bool = True
    strategy_mode: str = "hybrid"
    symbol_strategy_modes: dict[str, str] | None = None
    ml_lookback_bars: int = 320
    ml_probability_buy: float = 0.55
    ml_probability_sell: float = 0.45
    entry_threshold_pct: float = 0.001
    threshold_mode: str = THRESHOLD_MODE_STATIC_PCT
    atr_multiple: float = 1.0
    atr_percentile_threshold: float = 0.0
    time_window_mode: str = TIME_WINDOW_FULL_DAY
    regime_filter_enabled: bool = False
    orb_filter_mode: str = ORB_FILTER_NONE
    breakout_exit_style: str = BREAKOUT_EXIT_TARGET_1X_STOP_LOW
    breakout_tight_stop_fraction: float = 0.5
    breakout_max_stop_pct: float = 0.03
    sma_stop_pct: float = 0.0
    mean_reversion_exit_style: str = MEAN_REVERSION_EXIT_SMA
    mean_reversion_max_atr_percentile: float = 0.0
    mean_reversion_trend_filter: bool = False
    mean_reversion_trend_slope_filter: bool = False
    mean_reversion_stop_pct: float = 0.0
    vwap_z_entry_threshold: float = 1.5
    vwap_z_stop_atr_multiple: float = 2.0
    min_atr_percentile: float = 20.0
    max_adx_threshold: float = 25.0
    trend_pullback_min_adx: float = 20.0
    trend_pullback_min_slope: float = 0.0
    trend_pullback_entry_threshold: float = 0.0015
    trend_pullback_min_atr_percentile: float = 20.0
    trend_pullback_max_atr_percentile: float = 0.0
    trend_pullback_exit_style: str = "fixed_bars"
    trend_pullback_hold_bars: int = 4
    trend_pullback_take_profit_pct: float = 0.0
    trend_pullback_stop_pct: float = 0.0
    momentum_breakout_lookback_bars: int = 20
    momentum_breakout_entry_buffer_pct: float = 0.001
    momentum_breakout_min_adx: float = 20.0
    momentum_breakout_min_slope: float = 0.0
    momentum_breakout_min_atr_percentile: float = 20.0
    momentum_breakout_exit_style: str = MOMENTUM_BREAKOUT_EXIT_FIXED_BARS
    momentum_breakout_hold_bars: int = 3
    momentum_breakout_stop_pct: float = 0.0
    momentum_breakout_take_profit_pct: float = 0.0
    volatility_expansion_lookback_bars: int = 20
    volatility_expansion_entry_buffer_pct: float = 0.001
    volatility_expansion_max_atr_percentile: float = 0.0
    volatility_expansion_trend_filter: bool = False
    volatility_expansion_min_slope: float = 0.0
    volatility_expansion_use_volume_confirm: bool = True
    volatility_expansion_exit_style: str = VOLATILITY_EXPANSION_EXIT_FIXED_BARS
    volatility_expansion_hold_bars: int = 4
    volatility_expansion_stop_pct: float = 0.0
    volatility_expansion_take_profit_pct: float = 0.0
    bb_period: int = 20
    bb_stddev_mult: float = 2.0
    bb_width_lookback: int = 100
    bb_squeeze_quantile: float = 0.20
    bb_slope_lookback: int = 3
    bb_use_volume_confirm: bool = True
    bb_volume_mult: float = 1.2
    bb_breakout_buffer_pct: float = 0.0
    bb_min_mid_slope: float = 0.0
    bb_trend_filter: bool = False
    bb_exit_mode: str = BOLLINGER_EXIT_MIDDLE_BAND
    max_orders_per_minute: int = 6
    max_price_deviation_bps: float = 75.0
    max_data_delay_seconds: int = 300
    max_live_price_age_seconds: int = 60
    broker_backend: str = "alpaca"


@dataclass(frozen=True)
class RuntimeConfigDetails:
    config: BotConfig
    runtime_config_path: str | None
    overridden_fields: tuple[str, ...]
    runtime_config_approved: bool | None
    runtime_config_rejection_reasons: tuple[str, ...]
    baseline_valid_for_comparison: bool | None = None
    baseline_validation_errors: tuple[str, ...] = ()


@dataclass(frozen=True)
class SymbolSnapshot:
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
    hold_reason: str | None = None  # why action is HOLD: "trend_filter", "atr_filter", "no_signal", "holding_no_exit"
    signal_pct: float | None = None   # normalized 0–1 signal strength for non-ML modes
    signal_label: str | None = None   # e.g. "Distance to Entry" or "Recovery Progress"
    hybrid_branch_active: str | None = None
    hybrid_entry_branch: str | None = None
    hybrid_regime_branch: str | None = None
    final_signal_reason: str | None = None


@dataclass(frozen=True)
class SymbolEvaluation:
    price: float
    sma: float
    ml_signal: MlSignal
    action: str
    latest_bar_close_utc: str
    hold_reason: str | None = None  # populated when action == "HOLD"; e.g. "trend_filter", "atr_filter", "no_signal"
    signal_pct: float | None = None
    signal_label: str | None = None
    hybrid_branch_active: str | None = None
    hybrid_entry_branch: str | None = None
    hybrid_regime_branch: str | None = None
    final_signal_reason: str | None = None


@dataclass(frozen=True)
class BotSnapshot:
    timestamp_utc: str
    cash: float
    buying_power: float
    equity: float
    last_equity: float
    daily_pnl: float
    kill_switch_triggered: bool
    positions: dict[str, Position]
    symbols: list[SymbolSnapshot]


OrderSnapshot = BrokerOrder


@dataclass(frozen=True)
class ExecutionPreview:
    symbol: str
    action: str
    status: str
    reason: str | None = None
    detail: str | None = None
    live_price: float | None = None
    signal_price: float | None = None
    price_deviation_bps: float | None = None
    live_price_age_s: float | None = None


@dataclass(frozen=True)
class RunCycleReport:
    decision_timestamp: str
    execute_orders: bool
    processed_bar: bool
    skip_reason: str
    buy_signals: int
    sell_signals: int
    hold_signals: int
    error_signals: int
    orders_submitted: int


class ResyncStatus(str, Enum):
    LOCKED = "RESYNC_LOCKED"
    OK = "RESYNC_OK"
    DEGRADED = "RESYNC_DEGRADED"
    FAILED = "RESYNC_FAILED"


@dataclass(frozen=True)
class ResyncDiscrepancy:
    symbol: str | None
    kind: str
    detail: str


@dataclass(frozen=True)
class ResyncResult:
    status: ResyncStatus
    started_at_utc: str
    completed_at_utc: str | None
    positions_recovered: tuple[dict[str, Any], ...] = ()
    open_orders_recovered: tuple[dict[str, Any], ...] = ()
    recent_fills_recovered: tuple[dict[str, Any], ...] = ()
    discrepancies: tuple[ResyncDiscrepancy, ...] = ()
    reason_codes: tuple[str, ...] = ()
    gate_allows_entries: bool = False
    gate_allows_exits: bool = False


_ET = pytz.timezone("America/New_York")
_SESSION_ENTRY_START = dt_time(9, 45)   # no new entries before this
_SESSION_ENTRY_END   = dt_time(15, 45)  # no new entries after this
_SESSION_FLATTEN_AT  = dt_time(15, 55)  # forced EOD flatten deadline
DEFAULT_RUNTIME_CONFIG_PATH = Path("config") / "live_config.json"
STARTUP_ARTIFACT_PATH_ENV = "TRADEOS_STARTUP_ARTIFACT_PATH"


def _safe_float(value: str | None, default: float) -> float:
    try:
        return float(value) if value is not None else default
    except (TypeError, ValueError):
        return default


def _optional_float(value: Any) -> float | None:
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _float_or_default(value: Any, default: float = 0.0) -> float:
    converted = _optional_float(value)
    return converted if converted is not None else default


def _normalize_enum_text(value: Any) -> str:
    text = str(value or "").strip()
    if "." in text:
        text = text.rsplit(".", 1)[-1]
    return text.lower()


def _parse_iso_timestamp(raw_value: str | None) -> datetime | None:
    if not raw_value or raw_value == "None":
        return None
    normalized = raw_value
    if normalized.endswith("Z"):
        normalized = normalized[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def normalize_data_feed_name(raw_value: Any) -> str:
    normalized = str(raw_value or "").strip().lower()
    if normalized not in DATA_FEED_CHOICES:
        raise RuntimeError(
            f"Unsupported data feed {raw_value!r}. Choose from {', '.join(DATA_FEED_CHOICES)}."
        )
    return normalized


def normalize_bar_build_mode(raw_value: Any) -> str:
    normalized = str(raw_value or "").strip().lower()
    if normalized not in BAR_BUILD_MODE_CHOICES:
        raise RuntimeError(
            f"Unsupported bar build mode {raw_value!r}. Choose from {', '.join(BAR_BUILD_MODE_CHOICES)}."
        )
    return normalized


def _normalize_runtime_symbol_tuple(raw_value: Any, field_name: str) -> tuple[str, ...]:
    if raw_value is None:
        return ()
    if isinstance(raw_value, str):
        values = [item.strip() for item in raw_value.split(",") if item.strip()]
    elif isinstance(raw_value, list):
        values = [str(item).strip() for item in raw_value if str(item).strip()]
    else:
        raise RuntimeError(
            f"Runtime config field {field_name!r} must be a string or array of ticker symbols."
        )
    return tuple(sorted(normalize_symbols(values)))


def _coerce_runtime_symbol_event_blackouts(runtime: dict[str, Any], key: str) -> tuple[SymbolEventBlackout, ...]:
    raw_value = runtime.get(key)
    if raw_value is None:
        return ()
    if not isinstance(raw_value, list):
        raise RuntimeError(f"Runtime config field {key!r} must be an array of blackout objects.")

    blackouts: list[SymbolEventBlackout] = []
    for index, item in enumerate(raw_value):
        if not isinstance(item, dict):
            raise RuntimeError(f"Runtime config field {key!r}[{index}] must be an object.")
        symbol = str(item.get("symbol", "") or "").strip().upper()
        start_utc = str(item.get("start_utc", "") or "").strip()
        end_utc = str(item.get("end_utc", "") or "").strip()
        reason = str(item.get("reason", "event_blackout") or "event_blackout").strip()
        if not symbol:
            raise RuntimeError(f"Runtime config field {key!r}[{index}] is missing symbol.")
        if not start_utc or _parse_iso_timestamp(start_utc) is None:
            raise RuntimeError(f"Runtime config field {key!r}[{index}] has invalid start_utc.")
        if not end_utc or _parse_iso_timestamp(end_utc) is None:
            raise RuntimeError(f"Runtime config field {key!r}[{index}] has invalid end_utc.")
        start_ts = _parse_iso_timestamp(start_utc)
        end_ts = _parse_iso_timestamp(end_utc)
        if start_ts is None or end_ts is None or start_ts > end_ts:
            raise RuntimeError(f"Runtime config field {key!r}[{index}] must satisfy start_utc <= end_utc.")
        blackouts.append(
            SymbolEventBlackout(
                symbol=symbol,
                start_utc=start_utc,
                end_utc=end_utc,
                reason=reason or "event_blackout",
            )
        )
    return tuple(blackouts)


def _load_open_position_state_from_logs(log_root: str | Path = "logs") -> dict[str, dict[str, Any]]:
    open_positions: dict[str, dict[str, Any]] = {}
    log_path = Path(log_root)
    if not log_path.exists():
        return open_positions

    position_logs = sorted(log_path.glob("*/positions.jsonl"))
    for positions_file in position_logs:
        try:
            lines = positions_file.read_text(encoding="utf-8").splitlines()
        except OSError:
            continue
        for line in lines:
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            event = payload.get("event")
            symbol = str(payload.get("symbol") or "").strip().upper()
            if not symbol:
                continue
            if event == "position.opened":
                open_positions[symbol] = {
                    "entry_price": _optional_float(payload.get("entry_price")),
                    "qty": _optional_float(payload.get("qty")),
                    "decision_ts": payload.get("decision_ts"),
                    "entry_branch": payload.get("entry_branch"),
                }
            elif event == "position.closed":
                open_positions.pop(symbol, None)
    return open_positions


def _position_qty_value(position: Position | None) -> float:
    if position is None:
        return 0.0
    qty = getattr(position, "qty", 0.0)
    try:
        return float(qty)
    except (TypeError, ValueError):
        return 0.0


def _has_long_position(position: Position | None) -> bool:
    return _position_qty_value(position) > 0


def _has_short_position(position: Position | None) -> bool:
    return _position_qty_value(position) < 0


def _has_open_position(position: Position | None) -> bool:
    return _position_qty_value(position) != 0


def _parse_symbol_strategy_map(raw_value: str | None) -> dict[str, str]:
    if not raw_value:
        return {}
    parsed: dict[str, str] = {}
    for item in raw_value.split(","):
        if not item.strip():
            continue
        symbol, sep, mode = item.partition(":")
        if not sep:
            raise RuntimeError(
                "SYMBOL_STRATEGY_MAP must use SYMBOL:MODE pairs, e.g. AAPL:sma,MSFT:breakout"
            )
        normalized_mode = normalize_strategy_mode(mode)
        if normalized_mode not in STRATEGY_MODE_CHOICES:
            raise RuntimeError(
                f"Unsupported strategy mode {mode!r} in SYMBOL_STRATEGY_MAP. "
                f"Choose from {', '.join(STRATEGY_MODE_CHOICES)}."
            )
        parsed[symbol.strip().upper()] = normalized_mode
    return parsed


def _determine_signal_rejection(
    *,
    action: str,
    holding: bool,
    trend_filter_active: bool,
    trend_pass: bool | None,
    atr_pass: bool | None,
) -> str | None:
    if action in ("BUY", "SELL"):
        return None
    if holding:
        return "holding_no_exit"

    failing: list[str] = []
    if trend_filter_active and trend_pass is False:
        failing.append("trend_filter")
    if atr_pass is False:
        failing.append("atr_filter")
    if failing:
        return "|".join(failing)
    return "no_signal"


def _format_reason_label(reason: str | None) -> str:
    if not reason:
        return "decision context unavailable"
    return reason.replace("_", " ")


def _build_signal_explanation(
    *,
    strategy_mode: str,
    action: str,
    reason: str | None,
    price: float,
    sma: float,
    holding: bool,
    time_window_open: bool,
    atr_percentile: float | None,
    trend_sma: float | None,
    bullish_regime: bool | None,
    entry_reference_price: float | None,
    exit_reference_price: float | None,
    stop_reference_price: float | None,
    vwap: float | None,
    adx: float | None,
    hybrid_branch_active: str | None = None,
) -> str:
    parts: list[str] = []
    mode_label = strategy_mode.replace("_", " ")
    parts.append(f"{action} in {mode_label} mode")
    if reason:
        parts.append(f"reason: {_format_reason_label(reason)}")

    if entry_reference_price is not None and action != "SELL":
        relation = "above" if price >= entry_reference_price else "below"
        move_pct = abs(price - entry_reference_price) / max(entry_reference_price, 1e-9) * 100.0
        parts.append(f"price {relation} entry trigger by {move_pct:.2f}%")
    elif sma > 0:
        move_pct = abs(price - sma) / sma * 100.0
        relation = "above" if price >= sma else "below"
        parts.append(f"price {relation} SMA by {move_pct:.2f}%")

    if exit_reference_price is not None and holding:
        relation = "above" if price >= exit_reference_price else "below"
        parts.append(f"price {relation} exit reference")

    if stop_reference_price is not None and holding:
        cushion_pct = (price - stop_reference_price) / max(stop_reference_price, 1e-9) * 100.0
        parts.append(f"stop cushion {cushion_pct:.2f}%")

    if atr_percentile is not None:
        parts.append(f"ATR percentile {atr_percentile:.1f}")
    if bullish_regime is True:
        parts.append("regime bullish")
    elif bullish_regime is False:
        parts.append("regime bearish")
    if trend_sma is not None:
        parts.append("above trend SMA" if price >= trend_sma else "below trend SMA")
    if not time_window_open and not holding:
        parts.append("entry window closed")
    if vwap is not None:
        parts.append("below VWAP" if price < vwap else "above VWAP")
    if adx is not None:
        parts.append(f"ADX {adx:.1f}")
    if hybrid_branch_active:
        parts.append(f"branch {hybrid_branch_active.replace('_', ' ')}")
    return ". ".join(parts) + "."


def _normalize_runtime_symbols(raw_symbols: Any) -> list[str]:
    if not isinstance(raw_symbols, list):
        raise RuntimeError("Runtime config field 'symbols' must be a list of ticker strings.")
    symbols = normalize_symbols(raw_symbols)
    if not symbols:
        raise RuntimeError("Runtime config field 'symbols' must contain at least one ticker.")
    return symbols


def _coerce_runtime_float(runtime: dict[str, Any], key: str) -> float:
    value = runtime.get(key)
    if value is None:
        raise RuntimeError(f"Runtime config is missing required numeric field: {key}")
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise RuntimeError(f"Runtime config field {key!r} must be numeric, got {value!r}.") from exc


def _coerce_runtime_int(runtime: dict[str, Any], key: str) -> int:
    value = runtime.get(key)
    if value is None:
        raise RuntimeError(f"Runtime config is missing required integer field: {key}")
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise RuntimeError(f"Runtime config field {key!r} must be an integer, got {value!r}.") from exc


def _coerce_runtime_bool(runtime: dict[str, Any], key: str) -> bool:
    value = runtime.get(key)
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "1", "yes"}:
            return True
        if lowered in {"false", "0", "no"}:
            return False
    raise RuntimeError(f"Runtime config field {key!r} must be a boolean, got {value!r}.")


def _load_runtime_config_payload() -> tuple[Path | None, dict[str, Any] | None, dict[str, Any] | None]:
    runtime_path = Path(os.getenv("BOT_RUNTIME_CONFIG_PATH", str(DEFAULT_RUNTIME_CONFIG_PATH)))
    if not runtime_path.exists():
        return None, None, None

    try:
        payload = json.loads(runtime_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Runtime config is not valid JSON: {runtime_path}") from exc

    if not isinstance(payload, dict):
        raise RuntimeError(f"Runtime config must contain a JSON object at top level: {runtime_path}")

    runtime = payload.get("runtime", payload)
    if not isinstance(runtime, dict):
        raise RuntimeError(f"Runtime config field 'runtime' must be a JSON object: {runtime_path}")

    return runtime_path, runtime, payload


def _apply_runtime_config(
    base_config: BotConfig,
    runtime_path: Path | None,
    runtime: dict[str, Any] | None,
) -> tuple[BotConfig, tuple[str, ...]]:
    if runtime_path is None or runtime is None:
        return base_config, ()

    overrides: dict[str, Any] = {}
    data_parity = runtime.get("data_parity")
    if data_parity is not None and not isinstance(data_parity, dict):
        raise RuntimeError("Runtime config field 'data_parity' must be an object when present.")
    data_parity = data_parity or {}
    if "symbols" in runtime:
        overrides["symbols"] = _normalize_runtime_symbols(runtime["symbols"])
    if "excluded_symbols" in runtime:
        overrides["excluded_symbols"] = _normalize_runtime_symbol_tuple(runtime.get("excluded_symbols"), "excluded_symbols")
    if "symbol_event_blackouts" in runtime:
        overrides["symbol_event_blackouts"] = _coerce_runtime_symbol_event_blackouts(runtime, "symbol_event_blackouts")
    if "historical_feed" in runtime or "historical_feed" in data_parity:
        overrides["historical_feed"] = normalize_data_feed_name(
            data_parity.get("historical_feed", runtime.get("historical_feed"))
        )
    if "live_feed" in runtime or "live_feed" in data_parity:
        overrides["live_feed"] = normalize_data_feed_name(
            data_parity.get("live_feed", runtime.get("live_feed"))
        )
    if "latest_bar_feed" in runtime or "latest_bar_feed" in data_parity:
        overrides["latest_bar_feed"] = normalize_data_feed_name(
            data_parity.get("latest_bar_feed", runtime.get("latest_bar_feed"))
        )
    if "bar_build_mode" in runtime or "bar_build_mode" in data_parity:
        overrides["bar_build_mode"] = normalize_bar_build_mode(
            data_parity.get("bar_build_mode", runtime.get("bar_build_mode"))
        )
    for key in (
        "apply_updated_bars",
        "post_bar_reconcile_poll",
        "block_trading_until_resync",
        "assert_feed_on_startup",
        "log_bar_components",
    ):
        if key in runtime or key in data_parity:
            overrides[key] = _coerce_runtime_bool({key: data_parity.get(key, runtime.get(key))}, key)
    if "bar_timeframe_minutes" in runtime:
        overrides["bar_timeframe_minutes"] = _coerce_runtime_int(runtime, "bar_timeframe_minutes")
    if "strategy_mode" in runtime:
        overrides["strategy_mode"] = normalize_strategy_mode(str(runtime["strategy_mode"]))
    if "sma_bars" in runtime and runtime["sma_bars"] is not None:
        overrides["sma_bars"] = _coerce_runtime_int(runtime, "sma_bars")
    if "ml_probability_buy" in runtime and runtime["ml_probability_buy"] is not None:
        overrides["ml_probability_buy"] = _coerce_runtime_float(runtime, "ml_probability_buy")
    if "ml_probability_sell" in runtime and runtime["ml_probability_sell"] is not None:
        overrides["ml_probability_sell"] = _coerce_runtime_float(runtime, "ml_probability_sell")
    if "entry_threshold_pct" in runtime and runtime["entry_threshold_pct"] is not None:
        overrides["entry_threshold_pct"] = _coerce_runtime_float(runtime, "entry_threshold_pct")
    if "threshold_mode" in runtime:
        overrides["threshold_mode"] = normalize_threshold_mode(str(runtime["threshold_mode"]))
    if "atr_multiple" in runtime and runtime["atr_multiple"] is not None:
        overrides["atr_multiple"] = _coerce_runtime_float(runtime, "atr_multiple")
    if "atr_percentile_threshold" in runtime and runtime["atr_percentile_threshold"] is not None:
        overrides["atr_percentile_threshold"] = _coerce_runtime_float(runtime, "atr_percentile_threshold")
    if "time_window_mode" in runtime:
        overrides["time_window_mode"] = normalize_time_window_mode(str(runtime["time_window_mode"]))
    if "regime_filter_enabled" in runtime:
        overrides["regime_filter_enabled"] = _coerce_runtime_bool(runtime, "regime_filter_enabled")
    if "orb_filter_mode" in runtime:
        overrides["orb_filter_mode"] = normalize_orb_filter_mode(str(runtime["orb_filter_mode"]))
    if "breakout_exit_style" in runtime:
        overrides["breakout_exit_style"] = normalize_breakout_exit_style(str(runtime["breakout_exit_style"]))
    if "breakout_tight_stop_fraction" in runtime and runtime["breakout_tight_stop_fraction"] is not None:
        overrides["breakout_tight_stop_fraction"] = _coerce_runtime_float(runtime, "breakout_tight_stop_fraction")
    if "breakout_max_stop_pct" in runtime and runtime["breakout_max_stop_pct"] is not None:
        overrides["breakout_max_stop_pct"] = _coerce_runtime_float(runtime, "breakout_max_stop_pct")
    if "sma_stop_pct" in runtime and runtime["sma_stop_pct"] is not None:
        overrides["sma_stop_pct"] = _coerce_runtime_float(runtime, "sma_stop_pct")
    if "mean_reversion_exit_style" in runtime:
        overrides["mean_reversion_exit_style"] = normalize_mean_reversion_exit_style(
            str(runtime["mean_reversion_exit_style"])
        )
    if "mean_reversion_max_atr_percentile" in runtime and runtime["mean_reversion_max_atr_percentile"] is not None:
        overrides["mean_reversion_max_atr_percentile"] = _coerce_runtime_float(
            runtime,
            "mean_reversion_max_atr_percentile",
        )
    if "mean_reversion_trend_filter" in runtime:
        overrides["mean_reversion_trend_filter"] = _coerce_runtime_bool(runtime, "mean_reversion_trend_filter")
    if "mean_reversion_trend_slope_filter" in runtime:
        overrides["mean_reversion_trend_slope_filter"] = _coerce_runtime_bool(runtime, "mean_reversion_trend_slope_filter")
    if "mean_reversion_stop_pct" in runtime and runtime["mean_reversion_stop_pct"] is not None:
        overrides["mean_reversion_stop_pct"] = _coerce_runtime_float(runtime, "mean_reversion_stop_pct")
    if "vwap_z_entry_threshold" in runtime and runtime["vwap_z_entry_threshold"] is not None:
        overrides["vwap_z_entry_threshold"] = _coerce_runtime_float(runtime, "vwap_z_entry_threshold")
    if "vwap_z_stop_atr_multiple" in runtime and runtime["vwap_z_stop_atr_multiple"] is not None:
        overrides["vwap_z_stop_atr_multiple"] = _coerce_runtime_float(runtime, "vwap_z_stop_atr_multiple")
    if "min_atr_percentile" in runtime and runtime["min_atr_percentile"] is not None:
        overrides["min_atr_percentile"] = _coerce_runtime_float(runtime, "min_atr_percentile")
    if "max_adx_threshold" in runtime and runtime["max_adx_threshold"] is not None:
        overrides["max_adx_threshold"] = _coerce_runtime_float(runtime, "max_adx_threshold")
    if "trend_pullback_min_adx" in runtime and runtime["trend_pullback_min_adx"] is not None:
        overrides["trend_pullback_min_adx"] = _coerce_runtime_float(runtime, "trend_pullback_min_adx")
    if "trend_pullback_min_slope" in runtime and runtime["trend_pullback_min_slope"] is not None:
        overrides["trend_pullback_min_slope"] = _coerce_runtime_float(runtime, "trend_pullback_min_slope")
    if "trend_pullback_entry_threshold" in runtime and runtime["trend_pullback_entry_threshold"] is not None:
        overrides["trend_pullback_entry_threshold"] = _coerce_runtime_float(runtime, "trend_pullback_entry_threshold")
    if "trend_pullback_min_atr_percentile" in runtime and runtime["trend_pullback_min_atr_percentile"] is not None:
        overrides["trend_pullback_min_atr_percentile"] = _coerce_runtime_float(runtime, "trend_pullback_min_atr_percentile")
    if "trend_pullback_max_atr_percentile" in runtime and runtime["trend_pullback_max_atr_percentile"] is not None:
        overrides["trend_pullback_max_atr_percentile"] = _coerce_runtime_float(runtime, "trend_pullback_max_atr_percentile")
    if "trend_pullback_exit_style" in runtime:
        overrides["trend_pullback_exit_style"] = normalize_trend_pullback_exit_style(str(runtime["trend_pullback_exit_style"]))
    if "trend_pullback_hold_bars" in runtime and runtime["trend_pullback_hold_bars"] is not None:
        overrides["trend_pullback_hold_bars"] = _coerce_runtime_int(runtime, "trend_pullback_hold_bars")
    if "trend_pullback_take_profit_pct" in runtime and runtime["trend_pullback_take_profit_pct"] is not None:
        overrides["trend_pullback_take_profit_pct"] = _coerce_runtime_float(runtime, "trend_pullback_take_profit_pct")
    if "trend_pullback_stop_pct" in runtime and runtime["trend_pullback_stop_pct"] is not None:
        overrides["trend_pullback_stop_pct"] = _coerce_runtime_float(runtime, "trend_pullback_stop_pct")
    if "momentum_breakout_lookback_bars" in runtime and runtime["momentum_breakout_lookback_bars"] is not None:
        overrides["momentum_breakout_lookback_bars"] = _coerce_runtime_int(runtime, "momentum_breakout_lookback_bars")
    if "momentum_breakout_entry_buffer_pct" in runtime and runtime["momentum_breakout_entry_buffer_pct"] is not None:
        overrides["momentum_breakout_entry_buffer_pct"] = _coerce_runtime_float(runtime, "momentum_breakout_entry_buffer_pct")
    if "momentum_breakout_min_adx" in runtime and runtime["momentum_breakout_min_adx"] is not None:
        overrides["momentum_breakout_min_adx"] = _coerce_runtime_float(runtime, "momentum_breakout_min_adx")
    if "momentum_breakout_min_slope" in runtime and runtime["momentum_breakout_min_slope"] is not None:
        overrides["momentum_breakout_min_slope"] = _coerce_runtime_float(runtime, "momentum_breakout_min_slope")
    if "momentum_breakout_min_atr_percentile" in runtime and runtime["momentum_breakout_min_atr_percentile"] is not None:
        overrides["momentum_breakout_min_atr_percentile"] = _coerce_runtime_float(runtime, "momentum_breakout_min_atr_percentile")
    if "momentum_breakout_exit_style" in runtime:
        overrides["momentum_breakout_exit_style"] = normalize_momentum_breakout_exit_style(str(runtime["momentum_breakout_exit_style"]))
    if "momentum_breakout_hold_bars" in runtime and runtime["momentum_breakout_hold_bars"] is not None:
        overrides["momentum_breakout_hold_bars"] = _coerce_runtime_int(runtime, "momentum_breakout_hold_bars")
    if "momentum_breakout_stop_pct" in runtime and runtime["momentum_breakout_stop_pct"] is not None:
        overrides["momentum_breakout_stop_pct"] = _coerce_runtime_float(runtime, "momentum_breakout_stop_pct")
    if "momentum_breakout_take_profit_pct" in runtime and runtime["momentum_breakout_take_profit_pct"] is not None:
        overrides["momentum_breakout_take_profit_pct"] = _coerce_runtime_float(runtime, "momentum_breakout_take_profit_pct")
    if "volatility_expansion_lookback_bars" in runtime and runtime["volatility_expansion_lookback_bars"] is not None:
        overrides["volatility_expansion_lookback_bars"] = _coerce_runtime_int(runtime, "volatility_expansion_lookback_bars")
    if "volatility_expansion_entry_buffer_pct" in runtime and runtime["volatility_expansion_entry_buffer_pct"] is not None:
        overrides["volatility_expansion_entry_buffer_pct"] = _coerce_runtime_float(runtime, "volatility_expansion_entry_buffer_pct")
    if "volatility_expansion_max_atr_percentile" in runtime and runtime["volatility_expansion_max_atr_percentile"] is not None:
        overrides["volatility_expansion_max_atr_percentile"] = _coerce_runtime_float(runtime, "volatility_expansion_max_atr_percentile")
    if "volatility_expansion_trend_filter" in runtime:
        overrides["volatility_expansion_trend_filter"] = _coerce_runtime_bool(runtime, "volatility_expansion_trend_filter")
    if "volatility_expansion_min_slope" in runtime and runtime["volatility_expansion_min_slope"] is not None:
        overrides["volatility_expansion_min_slope"] = _coerce_runtime_float(runtime, "volatility_expansion_min_slope")
    if "volatility_expansion_use_volume_confirm" in runtime:
        overrides["volatility_expansion_use_volume_confirm"] = _coerce_runtime_bool(runtime, "volatility_expansion_use_volume_confirm")
    if "volatility_expansion_exit_style" in runtime:
        overrides["volatility_expansion_exit_style"] = normalize_volatility_expansion_exit_style(str(runtime["volatility_expansion_exit_style"]))
    if "volatility_expansion_hold_bars" in runtime and runtime["volatility_expansion_hold_bars"] is not None:
        overrides["volatility_expansion_hold_bars"] = _coerce_runtime_int(runtime, "volatility_expansion_hold_bars")
    if "volatility_expansion_stop_pct" in runtime and runtime["volatility_expansion_stop_pct"] is not None:
        overrides["volatility_expansion_stop_pct"] = _coerce_runtime_float(runtime, "volatility_expansion_stop_pct")
    if "volatility_expansion_take_profit_pct" in runtime and runtime["volatility_expansion_take_profit_pct"] is not None:
        overrides["volatility_expansion_take_profit_pct"] = _coerce_runtime_float(runtime, "volatility_expansion_take_profit_pct")
    if "bb_period" in runtime and runtime["bb_period"] is not None:
        overrides["bb_period"] = _coerce_runtime_int(runtime, "bb_period")
    if "bb_stddev_mult" in runtime and runtime["bb_stddev_mult"] is not None:
        overrides["bb_stddev_mult"] = _coerce_runtime_float(runtime, "bb_stddev_mult")
    if "bb_width_lookback" in runtime and runtime["bb_width_lookback"] is not None:
        overrides["bb_width_lookback"] = _coerce_runtime_int(runtime, "bb_width_lookback")
    if "bb_squeeze_quantile" in runtime and runtime["bb_squeeze_quantile"] is not None:
        overrides["bb_squeeze_quantile"] = _coerce_runtime_float(runtime, "bb_squeeze_quantile")
    if "bb_slope_lookback" in runtime and runtime["bb_slope_lookback"] is not None:
        overrides["bb_slope_lookback"] = _coerce_runtime_int(runtime, "bb_slope_lookback")
    if "bb_use_volume_confirm" in runtime:
        overrides["bb_use_volume_confirm"] = _coerce_runtime_bool(runtime, "bb_use_volume_confirm")
    if "bb_volume_mult" in runtime and runtime["bb_volume_mult"] is not None:
        overrides["bb_volume_mult"] = _coerce_runtime_float(runtime, "bb_volume_mult")
    if "bb_breakout_buffer_pct" in runtime and runtime["bb_breakout_buffer_pct"] is not None:
        overrides["bb_breakout_buffer_pct"] = _coerce_runtime_float(runtime, "bb_breakout_buffer_pct")
    if "bb_min_mid_slope" in runtime and runtime["bb_min_mid_slope"] is not None:
        overrides["bb_min_mid_slope"] = _coerce_runtime_float(runtime, "bb_min_mid_slope")
    if "bb_trend_filter" in runtime:
        overrides["bb_trend_filter"] = _coerce_runtime_bool(runtime, "bb_trend_filter")
    if "bb_exit_mode" in runtime:
        overrides["bb_exit_mode"] = normalize_bollinger_exit_mode(str(runtime["bb_exit_mode"]))
    if "ml_lookback_bars" in runtime and runtime["ml_lookback_bars"] is not None:
        overrides["ml_lookback_bars"] = _coerce_runtime_int(runtime, "ml_lookback_bars")
    if "max_orders_per_minute" in runtime and runtime["max_orders_per_minute"] is not None:
        overrides["max_orders_per_minute"] = _coerce_runtime_int(runtime, "max_orders_per_minute")
    if "max_price_deviation_bps" in runtime and runtime["max_price_deviation_bps"] is not None:
        overrides["max_price_deviation_bps"] = _coerce_runtime_float(runtime, "max_price_deviation_bps")
    if "max_data_delay_seconds" in runtime and runtime["max_data_delay_seconds"] is not None:
        overrides["max_data_delay_seconds"] = _coerce_runtime_int(runtime, "max_data_delay_seconds")
    if "max_live_price_age_seconds" in runtime and runtime["max_live_price_age_seconds"] is not None:
        overrides["max_live_price_age_seconds"] = _coerce_runtime_int(runtime, "max_live_price_age_seconds")
    if "symbol_strategy_modes" in runtime:
        if not isinstance(runtime["symbol_strategy_modes"], dict):
            raise RuntimeError("Runtime config field 'symbol_strategy_modes' must be an object mapping SYMBOL to mode.")
        overrides["symbol_strategy_modes"] = {
            str(symbol).strip().upper(): normalize_strategy_mode(str(mode))
            for symbol, mode in runtime["symbol_strategy_modes"].items()
            if str(symbol).strip()
        }

    config = replace(base_config, **overrides)
    if config.excluded_symbols:
        excluded = set(config.excluded_symbols)
        filtered_symbols = [symbol for symbol in config.symbols if symbol not in excluded]
        filtered_strategy_modes = {
            symbol: mode
            for symbol, mode in (config.symbol_strategy_modes or {}).items()
            if symbol in filtered_symbols
        }
        if not filtered_symbols:
            raise RuntimeError("Runtime config excluded every symbol; at least one active ticker is required.")
        config = replace(
            config,
            symbols=filtered_symbols,
            symbol_strategy_modes=filtered_strategy_modes or None,
        )
    runtime_symbols = ", ".join(config.symbols)
    changed_fields = sorted(
        field_name
        for field_name in overrides
        if getattr(base_config, field_name) != getattr(config, field_name)
    )
    logger.info("Loaded runtime config from %s", runtime_path)
    print(f"Runtime config loaded from {runtime_path}")
    print(f"Runtime config symbols: {runtime_symbols}")
    if changed_fields:
        changed_text = ", ".join(changed_fields)
        print(
            "Runtime config overrides .env for: "
            f"{changed_text}"
        )
    return config, tuple(changed_fields)


def _load_base_config() -> BotConfig:
    symbols_raw = os.getenv("BOT_SYMBOLS", "AAPL,MSFT,NVDA")
    symbols = normalize_symbols(symbols_raw.split(","))
    if not symbols:
        raise RuntimeError("BOT_SYMBOLS must contain at least one ticker.")

    sma_bars_raw = os.getenv("SMA_BARS") or os.getenv("SMA_DAYS", "20")
    bar_timeframe_minutes = int(os.getenv("BAR_TIMEFRAME_MINUTES", "15"))

    return BotConfig(
        symbols=symbols,
        max_usd_per_trade=float(os.getenv("MAX_USD_PER_TRADE", "200")),
        max_symbol_exposure_usd=float(
            os.getenv("MAX_SYMBOL_EXPOSURE_USD", os.getenv("MAX_USD_PER_TRADE", "200"))
        ),
        max_open_positions=int(os.getenv("MAX_OPEN_POSITIONS", "3")),
        max_daily_loss_usd=float(os.getenv("MAX_DAILY_LOSS_USD", "300")),
        sma_bars=int(sma_bars_raw),
        bar_timeframe_minutes=bar_timeframe_minutes,
        excluded_symbols=(),
        symbol_event_blackouts=(),
        historical_feed=normalize_data_feed_name(os.getenv("HISTORICAL_FEED", "iex")),
        live_feed=normalize_data_feed_name(os.getenv("LIVE_FEED", os.getenv("PRICE_STREAM_FEED", "iex"))),
        latest_bar_feed=normalize_data_feed_name(os.getenv("LATEST_BAR_FEED", "iex")),
        bar_build_mode=normalize_bar_build_mode(os.getenv("BAR_BUILD_MODE", "stream_minute_aggregate")),
        apply_updated_bars=os.getenv("APPLY_UPDATED_BARS", "true").lower() != "false",
        post_bar_reconcile_poll=os.getenv("POST_BAR_RECONCILE_POLL", "true").lower() != "false",
        block_trading_until_resync=os.getenv("BLOCK_TRADING_UNTIL_RESYNC", "true").lower() != "false",
        assert_feed_on_startup=os.getenv("ASSERT_FEED_ON_STARTUP", "true").lower() != "false",
        log_bar_components=os.getenv("LOG_BAR_COMPONENTS", "true").lower() != "false",
        paper=os.getenv("ALPACA_PAPER", "true").lower() != "false",
        strategy_mode=normalize_strategy_mode(os.getenv("STRATEGY_MODE", "hybrid")),
        symbol_strategy_modes=_parse_symbol_strategy_map(os.getenv("SYMBOL_STRATEGY_MAP")),
        ml_lookback_bars=int(os.getenv("ML_LOOKBACK_BARS", "320")),
        ml_probability_buy=_safe_float(os.getenv("ML_PROBABILITY_BUY"), 0.55),
        ml_probability_sell=_safe_float(os.getenv("ML_PROBABILITY_SELL"), 0.45),
        entry_threshold_pct=_safe_float(os.getenv("ENTRY_THRESHOLD_PCT"), 0.001),
        threshold_mode=normalize_threshold_mode(
            os.getenv("THRESHOLD_MODE", THRESHOLD_MODE_STATIC_PCT)
        ),
        atr_multiple=_safe_float(
            os.getenv("ATR_MULTIPLE", os.getenv("ATR_THRESHOLD_MULTIPLIER")),
            1.0,
        ),
        atr_percentile_threshold=_safe_float(os.getenv("ATR_PERCENTILE_THRESHOLD"), 0.0),
        time_window_mode=normalize_time_window_mode(
            os.getenv("TIME_WINDOW_MODE", TIME_WINDOW_FULL_DAY)
        ),
        regime_filter_enabled=os.getenv("REGIME_FILTER_ENABLED", "false").lower() == "true",
        orb_filter_mode=normalize_orb_filter_mode(os.getenv("ORB_FILTER_MODE", ORB_FILTER_NONE)),
        breakout_exit_style=normalize_breakout_exit_style(
            os.getenv("BREAKOUT_EXIT_STYLE", BREAKOUT_EXIT_TARGET_1X_STOP_LOW)
        ),
        breakout_tight_stop_fraction=_safe_float(os.getenv("BREAKOUT_TIGHT_STOP_FRACTION"), 0.5),
        breakout_max_stop_pct=_safe_float(os.getenv("BREAKOUT_MAX_STOP_PCT"), 0.03),
        sma_stop_pct=_safe_float(os.getenv("SMA_STOP_PCT"), 0.0),
        mean_reversion_exit_style=normalize_mean_reversion_exit_style(
            os.getenv("MEAN_REVERSION_EXIT_STYLE", MEAN_REVERSION_EXIT_SMA)
        ),
        mean_reversion_max_atr_percentile=_safe_float(os.getenv("MEAN_REVERSION_MAX_ATR_PERCENTILE"), 0.0),
        mean_reversion_trend_filter=os.getenv("MEAN_REVERSION_TREND_FILTER", "false").lower() == "true",
        mean_reversion_trend_slope_filter=os.getenv("MEAN_REVERSION_TREND_SLOPE_FILTER", "false").lower() == "true",
        mean_reversion_stop_pct=_safe_float(os.getenv("MEAN_REVERSION_STOP_PCT"), 0.0),
        vwap_z_entry_threshold=_safe_float(os.getenv("VWAP_Z_ENTRY_THRESHOLD"), 1.5),
        vwap_z_stop_atr_multiple=_safe_float(os.getenv("VWAP_Z_STOP_ATR_MULTIPLE"), 2.0),
        min_atr_percentile=_safe_float(os.getenv("MIN_ATR_PERCENTILE"), 20.0),
        max_adx_threshold=_safe_float(os.getenv("MAX_ADX_THRESHOLD"), 25.0),
        trend_pullback_min_adx=_safe_float(os.getenv("TREND_PULLBACK_MIN_ADX"), 20.0),
        trend_pullback_min_slope=_safe_float(os.getenv("TREND_PULLBACK_MIN_SLOPE"), 0.0),
        trend_pullback_entry_threshold=_safe_float(os.getenv("TREND_PULLBACK_ENTRY_THRESHOLD"), 0.0015),
        trend_pullback_min_atr_percentile=_safe_float(os.getenv("TREND_PULLBACK_MIN_ATR_PERCENTILE"), 20.0),
        trend_pullback_max_atr_percentile=_safe_float(os.getenv("TREND_PULLBACK_MAX_ATR_PERCENTILE"), 0.0),
        trend_pullback_exit_style=normalize_trend_pullback_exit_style(
            os.getenv("TREND_PULLBACK_EXIT_STYLE", "fixed_bars")
        ),
        trend_pullback_hold_bars=int(os.getenv("TREND_PULLBACK_HOLD_BARS", "4")),
        trend_pullback_take_profit_pct=_safe_float(os.getenv("TREND_PULLBACK_TAKE_PROFIT_PCT"), 0.0),
        trend_pullback_stop_pct=_safe_float(os.getenv("TREND_PULLBACK_STOP_PCT"), 0.0),
        momentum_breakout_lookback_bars=int(os.getenv("MOMENTUM_BREAKOUT_LOOKBACK_BARS", "20")),
        momentum_breakout_entry_buffer_pct=_safe_float(os.getenv("MOMENTUM_BREAKOUT_ENTRY_BUFFER_PCT"), 0.001),
        momentum_breakout_min_adx=_safe_float(os.getenv("MOMENTUM_BREAKOUT_MIN_ADX"), 20.0),
        momentum_breakout_min_slope=_safe_float(os.getenv("MOMENTUM_BREAKOUT_MIN_SLOPE"), 0.0),
        momentum_breakout_min_atr_percentile=_safe_float(os.getenv("MOMENTUM_BREAKOUT_MIN_ATR_PERCENTILE"), 20.0),
        momentum_breakout_exit_style=normalize_momentum_breakout_exit_style(
            os.getenv("MOMENTUM_BREAKOUT_EXIT_STYLE", MOMENTUM_BREAKOUT_EXIT_FIXED_BARS)
        ),
        momentum_breakout_hold_bars=int(os.getenv("MOMENTUM_BREAKOUT_HOLD_BARS", "3")),
        momentum_breakout_stop_pct=_safe_float(os.getenv("MOMENTUM_BREAKOUT_STOP_PCT"), 0.0),
        momentum_breakout_take_profit_pct=_safe_float(os.getenv("MOMENTUM_BREAKOUT_TAKE_PROFIT_PCT"), 0.0),
        volatility_expansion_lookback_bars=int(os.getenv("VOLATILITY_EXPANSION_LOOKBACK_BARS", "20")),
        volatility_expansion_entry_buffer_pct=_safe_float(os.getenv("VOLATILITY_EXPANSION_ENTRY_BUFFER_PCT"), 0.001),
        volatility_expansion_max_atr_percentile=_safe_float(os.getenv("VOLATILITY_EXPANSION_MAX_ATR_PERCENTILE"), 0.0),
        volatility_expansion_trend_filter=os.getenv("VOLATILITY_EXPANSION_TREND_FILTER", "false").lower() == "true",
        volatility_expansion_min_slope=_safe_float(os.getenv("VOLATILITY_EXPANSION_MIN_SLOPE"), 0.0),
        volatility_expansion_use_volume_confirm=os.getenv("VOLATILITY_EXPANSION_USE_VOLUME_CONFIRM", "true").lower() != "false",
        volatility_expansion_exit_style=normalize_volatility_expansion_exit_style(
            os.getenv("VOLATILITY_EXPANSION_EXIT_STYLE", VOLATILITY_EXPANSION_EXIT_FIXED_BARS)
        ),
        volatility_expansion_hold_bars=int(os.getenv("VOLATILITY_EXPANSION_HOLD_BARS", "4")),
        volatility_expansion_stop_pct=_safe_float(os.getenv("VOLATILITY_EXPANSION_STOP_PCT"), 0.0),
        volatility_expansion_take_profit_pct=_safe_float(os.getenv("VOLATILITY_EXPANSION_TAKE_PROFIT_PCT"), 0.0),
        bb_period=int(os.getenv("BB_PERIOD", "20")),
        bb_stddev_mult=_safe_float(os.getenv("BB_STDDEV_MULT"), 2.0),
        bb_width_lookback=int(os.getenv("BB_WIDTH_LOOKBACK", "100")),
        bb_squeeze_quantile=_safe_float(os.getenv("BB_SQUEEZE_QUANTILE"), 0.20),
        bb_slope_lookback=int(os.getenv("BB_SLOPE_LOOKBACK", "3")),
        bb_use_volume_confirm=os.getenv("BB_USE_VOLUME_CONFIRM", "true").lower() == "true",
        bb_volume_mult=_safe_float(os.getenv("BB_VOLUME_MULT"), 1.2),
        bb_breakout_buffer_pct=_safe_float(os.getenv("BB_BREAKOUT_BUFFER_PCT"), 0.0),
        bb_min_mid_slope=_safe_float(os.getenv("BB_MIN_MID_SLOPE"), 0.0),
        bb_trend_filter=os.getenv("BB_TREND_FILTER", "false").lower() == "true",
        bb_exit_mode=normalize_bollinger_exit_mode(os.getenv("BB_EXIT_MODE", BOLLINGER_EXIT_MIDDLE_BAND)),
        max_orders_per_minute=int(os.getenv("MAX_ORDERS_PER_MINUTE", "6")),
        max_price_deviation_bps=_safe_float(os.getenv("MAX_PRICE_DEVIATION_BPS"), 75.0),
        max_data_delay_seconds=int(os.getenv("MAX_DATA_DELAY_SECONDS", "300")),
        max_live_price_age_seconds=int(os.getenv("MAX_LIVE_PRICE_AGE_SECONDS", "60")),
        broker_backend=os.getenv("BROKER_BACKEND", "alpaca").strip().lower() or "alpaca",
    )


def load_config_details() -> RuntimeConfigDetails:
    load_dotenv(Path.cwd() / ".env")
    base_config = _load_base_config()
    runtime_path, runtime, payload = _load_runtime_config_payload()
    config, changed_fields = _apply_runtime_config(base_config, runtime_path, runtime)
    runtime_config_approved: bool | None = None
    runtime_config_rejection_reasons: tuple[str, ...] = ()
    if isinstance(payload, dict):
        source = payload.get("source")
        if isinstance(source, dict):
            approved = source.get("approved")
            if isinstance(approved, bool):
                runtime_config_approved = approved
            rejection_reasons = source.get("rejection_reasons")
            if isinstance(rejection_reasons, list):
                runtime_config_rejection_reasons = tuple(
                    str(reason) for reason in rejection_reasons if str(reason).strip()
                )
    return RuntimeConfigDetails(
        config=config,
        runtime_config_path=str(runtime_path) if runtime_path is not None else None,
        overridden_fields=changed_fields,
        runtime_config_approved=runtime_config_approved,
        runtime_config_rejection_reasons=runtime_config_rejection_reasons,
    )


def load_config() -> BotConfig:
    # Live execution config is intentionally sourced from environment variables
    # and normalized into BotConfig here. Offline tools use their own CLI args.
    return load_config_details().config


class TradeOSBot:
    def __init__(self, config: BotConfig, session_id: str | None = None) -> None:
        load_dotenv(Path.cwd() / ".env")
        self._session_started_at = datetime.now(timezone.utc)

        self.config = config
        self.broker = self._build_broker()
        db_path = Path(os.getenv("BOT_DB_PATH", "bot_history.db"))
        self.storage = BotStorage(db_path)
        self.session_id = session_id or f"session-{int(datetime.now(timezone.utc).timestamp())}"
        self.blog = BotLogger(log_root="logs", session_id=self.session_id)
        self.active_symbols = normalize_symbols(config.symbols)
        self._active_symbol_fingerprint = symbol_fingerprint(self.active_symbols)
        # Fill-tracking state (keyed by order_id)
        self._order_submission_ts: dict[str, str] = {}    # order_id → decision_ts at submit time
        self._order_submission_side: dict[str, str] = {}  # order_id → "buy" | "sell"
        self._order_exit_reason: dict[str, str] = {}      # order_id → exit reason for sell orders
        self._logged_fills: set[tuple[str, float]] = set() # (order_id, filled_qty) — prevents duplicates
        self._kill_switch_warned_pcts: set[int] = set()   # tracks which % thresholds already logged (50, 75)
        # Position lifecycle state (keyed by symbol)
        self._position_entry_price: dict[str, float] = {} # symbol → fill price at entry
        self._position_entry_ts: dict[str, str] = {}      # symbol → decision_ts at entry
        self._position_qty: dict[str, float] = {}         # symbol → qty at entry
        self._position_entry_branch: dict[str, str] = {}  # symbol → branch that opened the position
        self._ml_disabled_reason: str | None = None
        self._order_signal_price: dict[str, float] = {}
        self._order_entry_branch: dict[str, str] = {}     # order_id → branch intended at submit time
        self._last_processed_decision_timestamp: datetime | None = None
        self._position_first_seen_utc: dict[str, datetime] = {}
        self._bars_cache: dict[tuple[str, int, int, str], list[BrokerBar]] = {}
        self._hourly_regime_cache: dict[tuple[str, str], bool | None] = {}
        self._strategy_cache: dict[str, Strategy] = {}
        self._last_run_cycle_report: RunCycleReport | None = None
        self._startup_resync_result = ResyncResult(
            status=ResyncStatus.LOCKED,
            started_at_utc=datetime.now(timezone.utc).isoformat(),
            completed_at_utc=None,
            gate_allows_entries=False,
            gate_allows_exits=False,
        )
        self._startup_market_data_validated_for_et_date: str | None = None
        self._validate_persisted_symbol_state()
        self.strategy = Strategy(
            StrategyConfig(
                strategy_mode=config.strategy_mode,
                ml_probability_buy=config.ml_probability_buy,
                ml_probability_sell=config.ml_probability_sell,
                entry_threshold_pct=config.entry_threshold_pct,
                threshold_mode=config.threshold_mode,
                atr_multiple=config.atr_multiple,
                atr_percentile_threshold=config.atr_percentile_threshold,
                time_window_mode=config.time_window_mode,
                regime_filter_enabled=config.regime_filter_enabled,
                orb_filter_mode=config.orb_filter_mode,
                breakout_exit_style=config.breakout_exit_style,
                breakout_tight_stop_fraction=config.breakout_tight_stop_fraction,
                breakout_max_stop_pct=config.breakout_max_stop_pct,
                sma_stop_pct=config.sma_stop_pct,
                mean_reversion_exit_style=config.mean_reversion_exit_style,
                mean_reversion_max_atr_percentile=config.mean_reversion_max_atr_percentile,
                mean_reversion_trend_filter=config.mean_reversion_trend_filter,
                mean_reversion_trend_slope_filter=config.mean_reversion_trend_slope_filter,
                mean_reversion_stop_pct=config.mean_reversion_stop_pct,
                vwap_z_entry_threshold=config.vwap_z_entry_threshold,
                vwap_z_stop_atr_multiple=config.vwap_z_stop_atr_multiple,
                min_atr_percentile=config.min_atr_percentile,
                max_adx_threshold=config.max_adx_threshold,
                trend_pullback_min_adx=config.trend_pullback_min_adx,
                trend_pullback_min_slope=config.trend_pullback_min_slope,
                trend_pullback_entry_threshold=config.trend_pullback_entry_threshold,
                trend_pullback_min_atr_percentile=config.trend_pullback_min_atr_percentile,
                trend_pullback_max_atr_percentile=config.trend_pullback_max_atr_percentile,
                trend_pullback_exit_style=config.trend_pullback_exit_style,
                trend_pullback_hold_bars=config.trend_pullback_hold_bars,
                trend_pullback_take_profit_pct=config.trend_pullback_take_profit_pct,
                trend_pullback_stop_pct=config.trend_pullback_stop_pct,
                momentum_breakout_lookback_bars=config.momentum_breakout_lookback_bars,
                momentum_breakout_entry_buffer_pct=config.momentum_breakout_entry_buffer_pct,
                momentum_breakout_min_adx=config.momentum_breakout_min_adx,
                momentum_breakout_min_slope=config.momentum_breakout_min_slope,
                momentum_breakout_min_atr_percentile=config.momentum_breakout_min_atr_percentile,
                momentum_breakout_exit_style=config.momentum_breakout_exit_style,
                momentum_breakout_hold_bars=config.momentum_breakout_hold_bars,
                momentum_breakout_stop_pct=config.momentum_breakout_stop_pct,
                momentum_breakout_take_profit_pct=config.momentum_breakout_take_profit_pct,
                volatility_expansion_lookback_bars=config.volatility_expansion_lookback_bars,
                volatility_expansion_entry_buffer_pct=config.volatility_expansion_entry_buffer_pct,
                volatility_expansion_max_atr_percentile=config.volatility_expansion_max_atr_percentile,
                volatility_expansion_trend_filter=config.volatility_expansion_trend_filter,
                volatility_expansion_min_slope=config.volatility_expansion_min_slope,
                volatility_expansion_use_volume_confirm=config.volatility_expansion_use_volume_confirm,
                volatility_expansion_exit_style=config.volatility_expansion_exit_style,
                volatility_expansion_hold_bars=config.volatility_expansion_hold_bars,
                volatility_expansion_stop_pct=config.volatility_expansion_stop_pct,
                volatility_expansion_take_profit_pct=config.volatility_expansion_take_profit_pct,
                bb_period=config.bb_period,
                bb_stddev_mult=config.bb_stddev_mult,
                bb_width_lookback=config.bb_width_lookback,
                bb_squeeze_quantile=config.bb_squeeze_quantile,
                bb_slope_lookback=config.bb_slope_lookback,
                bb_use_volume_confirm=config.bb_use_volume_confirm,
                bb_volume_mult=config.bb_volume_mult,
                bb_breakout_buffer_pct=config.bb_breakout_buffer_pct,
                bb_min_mid_slope=config.bb_min_mid_slope,
                bb_trend_filter=config.bb_trend_filter,
                bb_exit_mode=config.bb_exit_mode,
            )
        )
        self._symbol_strategy_modes = config.symbol_strategy_modes or {}
        self._breakout_stored_stop: dict[str, float] = {}
        invalid_symbol_modes = [
            mode for mode in self._symbol_strategy_modes.values()
            if mode not in STRATEGY_MODE_CHOICES
        ]
        if invalid_symbol_modes:
            raise RuntimeError(
                f"Unsupported symbol strategy mode(s): {', '.join(sorted(set(invalid_symbol_modes)))}. "
                f"Choose from {', '.join(STRATEGY_MODE_CHOICES)}."
            )
        if config.strategy_mode not in STRATEGY_MODE_CHOICES:
            raise RuntimeError(
                f"Unsupported STRATEGY_MODE={config.strategy_mode}. "
                f"Choose from {', '.join(STRATEGY_MODE_CHOICES)}."
            )
        if config.threshold_mode not in THRESHOLD_MODE_CHOICES:
            raise RuntimeError(
                f"Unsupported THRESHOLD_MODE={config.threshold_mode}. "
                f"Choose from {', '.join(THRESHOLD_MODE_CHOICES)}."
            )
        if config.time_window_mode not in TIME_WINDOW_CHOICES:
            raise RuntimeError(
                f"Unsupported TIME_WINDOW_MODE={config.time_window_mode}. "
                f"Choose from {', '.join(TIME_WINDOW_CHOICES)}."
            )
        if config.orb_filter_mode not in ORB_FILTER_CHOICES:
            raise RuntimeError(
                f"Unsupported ORB_FILTER_MODE={config.orb_filter_mode}. "
                f"Choose from {', '.join(ORB_FILTER_CHOICES)}."
            )
        if config.breakout_exit_style not in BREAKOUT_EXIT_CHOICES:
            raise RuntimeError(
                f"Unsupported BREAKOUT_EXIT_STYLE={config.breakout_exit_style}. "
                f"Choose from {', '.join(BREAKOUT_EXIT_CHOICES)}."
            )
        if config.mean_reversion_exit_style not in MEAN_REVERSION_EXIT_CHOICES:
            raise RuntimeError(
                f"Unsupported MEAN_REVERSION_EXIT_STYLE={config.mean_reversion_exit_style}. "
                f"Choose from {', '.join(MEAN_REVERSION_EXIT_CHOICES)}."
            )
        if normalize_trend_pullback_exit_style(config.trend_pullback_exit_style) not in TREND_PULLBACK_EXIT_CHOICES:
            raise RuntimeError(
                f"Unsupported TREND_PULLBACK_EXIT_STYLE={config.trend_pullback_exit_style}. "
                f"Choose from {', '.join(TREND_PULLBACK_EXIT_CHOICES)}."
            )
        if normalize_momentum_breakout_exit_style(config.momentum_breakout_exit_style) not in MOMENTUM_BREAKOUT_EXIT_CHOICES:
            raise RuntimeError(
                f"Unsupported MOMENTUM_BREAKOUT_EXIT_STYLE={config.momentum_breakout_exit_style}. "
                f"Choose from {', '.join(MOMENTUM_BREAKOUT_EXIT_CHOICES)}."
            )
        if normalize_volatility_expansion_exit_style(config.volatility_expansion_exit_style) not in VOLATILITY_EXPANSION_EXIT_CHOICES:
            raise RuntimeError(
                f"Unsupported VOLATILITY_EXPANSION_EXIT_STYLE={config.volatility_expansion_exit_style}. "
                f"Choose from {', '.join(VOLATILITY_EXPANSION_EXIT_CHOICES)}."
            )
        if config.bb_exit_mode not in BOLLINGER_EXIT_CHOICES:
            raise RuntimeError(
                f"Unsupported BB_EXIT_MODE={config.bb_exit_mode}. "
                f"Choose from {', '.join(BOLLINGER_EXIT_CHOICES)}."
            )
        if offline_to_ml_signal is None and any(
            mode in {STRATEGY_MODE_ML, STRATEGY_MODE_HYBRID}
            for mode in {config.strategy_mode, *self._symbol_strategy_modes.values()}
        ):
            self._disable_ml_trading("ml.predict import failed", _ML_PREDICT_IMPORT_ERROR)

    def _build_broker(self) -> AlpacaBroker:
        backend = (self.config.broker_backend or "alpaca").strip().lower()
        if backend != "alpaca":
            raise RuntimeError(
                f"Unsupported BROKER_BACKEND={backend!r}. TradeOS currently ships with the Alpaca adapter only."
            )

        api_key = os.getenv("ALPACA_API_KEY")
        api_secret = os.getenv("ALPACA_API_SECRET")
        if not api_key or not api_secret:
            raise RuntimeError(
                "Missing Alpaca credentials. Set ALPACA_API_KEY and ALPACA_API_SECRET in .env."
            )
        return AlpacaBroker(
            api_key=api_key,
            api_secret=api_secret,
            paper=self.config.paper,
            symbols=self.config.symbols,
            price_stream_feed=self.config.live_feed,
            latest_data_feed=self.config.latest_bar_feed,
        )

    def data_parity_summary(self) -> dict[str, Any]:
        return {
            "historical_feed": self.config.historical_feed,
            "live_feed": self.config.live_feed,
            "latest_bar_feed": self.config.latest_bar_feed,
            "bar_build_mode": self.config.bar_build_mode,
            "apply_updated_bars": self.config.apply_updated_bars,
            "post_bar_reconcile_poll": self.config.post_bar_reconcile_poll,
            "block_trading_until_resync": self.config.block_trading_until_resync,
            "assert_feed_on_startup": self.config.assert_feed_on_startup,
            "log_bar_components": self.config.log_bar_components,
        }

    def get_startup_resync_result(self) -> ResyncResult:
        return self._startup_resync_result

    def _set_startup_resync_result(self, result: ResyncResult) -> ResyncResult:
        self._startup_resync_result = result
        self._update_startup_artifact_with_resync(result)
        return result

    def _update_startup_artifact_with_resync(self, result: ResyncResult) -> None:
        artifact_path_raw = os.getenv(STARTUP_ARTIFACT_PATH_ENV, "").strip()
        if not artifact_path_raw:
            return
        artifact_path = Path(artifact_path_raw)
        if not artifact_path.exists():
            return
        try:
            payload = json.loads(artifact_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return
        if not isinstance(payload, dict):
            return
        payload["resync_status"] = result.status.value
        payload["resync_started_at_utc"] = result.started_at_utc
        payload["resync_completed_at_utc"] = result.completed_at_utc
        payload["resync_reason_codes"] = list(result.reason_codes)
        payload["resync_positions_recovered"] = list(result.positions_recovered)
        payload["resync_open_orders_recovered"] = list(result.open_orders_recovered)
        payload["resync_recent_fills_recovered"] = list(result.recent_fills_recovered)
        payload["resync_discrepancies"] = [
            {
                "symbol": item.symbol,
                "kind": item.kind,
                "detail": item.detail,
            }
            for item in result.discrepancies
        ]
        payload["gate_allows_entries"] = result.gate_allows_entries
        payload["gate_allows_exits"] = result.gate_allows_exits
        artifact_text = json.dumps(payload, indent=2)
        artifact_path.write_text(artifact_text, encoding="utf-8")
        latest_path = artifact_path.parent / "startup_config.json"
        latest_path.write_text(artifact_text, encoding="utf-8")

    def _local_resync_context(self) -> tuple[dict[str, Any] | None, dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
        latest_run = self.storage.get_latest_run()
        latest_symbols = {
            str(row.get("symbol", "")).strip().upper(): row
            for row in self.storage.get_latest_symbol_snapshot()
            if str(row.get("symbol", "")).strip()
        }
        order_history = {
            str(row.get("order_id", "")).strip(): row
            for row in self.storage.get_order_history(limit=200)
            if str(row.get("order_id", "")).strip()
        }
        return latest_run, latest_symbols, order_history

    def _resync_gate_for_action(
        self,
        *,
        action: str,
        symbol: str,
        positions: dict[str, Position],
    ) -> tuple[bool, str | None]:
        result = self._startup_resync_result
        if action == "BUY":
            if result.gate_allows_entries:
                return True, None
            if result.status == ResyncStatus.LOCKED:
                return False, "resync_locked"
            if result.status == ResyncStatus.FAILED:
                return False, "resync_failed"
            return False, "resync_entries_disabled"
        if action == "SELL":
            if not result.gate_allows_exits:
                return False, "resync_exits_disabled"
            if result.status == ResyncStatus.DEGRADED and symbol not in positions:
                return False, "resync_exit_requires_broker_position"
        return True, None

    def perform_startup_resync(
        self,
        *,
        recent_orders_limit: int = 50,
        lookback_minutes: int = 240,
        current_time_utc: datetime | None = None,
    ) -> ResyncResult:
        current = self._startup_resync_result
        if current.status != ResyncStatus.LOCKED and current.completed_at_utc is not None:
            return current

        now_utc = current_time_utc or datetime.now(timezone.utc)
        started_at = now_utc.isoformat()
        lookback_cutoff = now_utc - timedelta(minutes=lookback_minutes)
        self.blog.lifecycle(
            "resync.started",
            lookback_minutes=lookback_minutes,
            recent_orders_limit=recent_orders_limit,
            data_parity=self.data_parity_summary(),
        )

        positions_recovered: list[dict[str, Any]] = []
        open_orders_recovered: list[dict[str, Any]] = []
        recent_fills_recovered: list[dict[str, Any]] = []
        discrepancies: list[ResyncDiscrepancy] = []
        reason_codes: list[str] = []

        try:
            broker_positions = self.get_positions_by_symbol()
            broker_open_orders = self.get_open_orders()
            recent_orders = self.get_recent_orders(limit=recent_orders_limit)
        except Exception as exc:
            result = ResyncResult(
                status=ResyncStatus.FAILED,
                started_at_utc=started_at,
                completed_at_utc=datetime.now(timezone.utc).isoformat(),
                discrepancies=(ResyncDiscrepancy(symbol=None, kind="broker_query_failed", detail=str(exc)),),
                reason_codes=("RESYNC_UNRESOLVED_ORDER_STATE", "RESYNC_UNRESOLVED_POSITION_STATE"),
                gate_allows_entries=False,
                gate_allows_exits=False,
            )
            self.blog.lifecycle("resync.failed", status=result.status.value, reason_codes=list(result.reason_codes))
            return self._set_startup_resync_result(result)

        latest_run, local_symbols, local_orders = self._local_resync_context()
        latest_run_ts = _parse_iso_timestamp(str(latest_run.get("timestamp_utc", "") or "")) if latest_run else None
        overnight_carry_restart = (
            latest_run_ts is not None
            and latest_run_ts.astimezone(_ET).date() < now_utc.astimezone(_ET).date()
        )
        if (broker_positions or broker_open_orders) and latest_run_ts is not None and latest_run_ts < lookback_cutoff:
            # Overnight restarts with carried positions are expected; keep the
            # stricter degraded gate for stale intraday restarts or any case
            # where open orders still need reconciliation.
            if broker_open_orders or not overnight_carry_restart:
                reason_codes.append("RESYNC_LOOKBACK_INSUFFICIENT")
                discrepancies.append(
                    ResyncDiscrepancy(
                        symbol=None,
                        kind="lookback_insufficient",
                        detail="local persisted state predates the recovery lookback window",
                    )
                )

        for symbol, position in broker_positions.items():
            local_row = local_symbols.get(symbol)
            local_holding = bool(local_row.get("holding")) if local_row is not None else False
            local_qty = _optional_float(local_row.get("quantity")) if local_row is not None else None
            if not local_holding or (local_qty or 0.0) <= 0:
                reason_codes.append("RESYNC_BROKER_POSITION_LOCAL_MISSING")
                recovered = {
                    "symbol": symbol,
                    "qty": _position_qty_value(position),
                    "avg_entry_price": _optional_float(getattr(position, "avg_entry_price", None)),
                    "current_price": _optional_float(getattr(position, "current_price", None)),
                }
                positions_recovered.append(recovered)
                entry_price = recovered["avg_entry_price"] or (
                    _optional_float(local_row.get("avg_entry_price")) if local_row is not None else None
                )
                if entry_price is not None:
                    self._position_entry_price[symbol] = entry_price
                self._position_qty[symbol] = recovered["qty"]
                if latest_run is not None and latest_run.get("timestamp_utc"):
                    self._position_entry_ts.setdefault(symbol, str(latest_run["timestamp_utc"]))

        if not broker_positions and not any(bool(row.get("holding")) for row in local_symbols.values()):
            reason_codes.append("RESYNC_NO_BROKER_POSITION_LOCAL_FLAT")

        for symbol, row in local_symbols.items():
            local_holding = bool(row.get("holding"))
            local_qty = _optional_float(row.get("quantity")) or 0.0
            if local_holding and local_qty > 0 and symbol not in broker_positions:
                reason_codes.append("RESYNC_SYMBOL_STATE_RESET")
                discrepancies.append(
                    ResyncDiscrepancy(
                        symbol=symbol,
                        kind="local_position_reset",
                        detail="local position state was cleared because the broker reported no open position",
                    )
                )
                self._position_entry_price.pop(symbol, None)
                self._position_entry_ts.pop(symbol, None)
                self._position_qty.pop(symbol, None)
                self._position_entry_branch.pop(symbol, None)
                self._position_first_seen_utc.pop(symbol, None)

        for order in broker_open_orders:
            if not order.order_id or not order.symbol:
                reason_codes.append("RESYNC_UNRESOLVED_ORDER_STATE")
                discrepancies.append(
                    ResyncDiscrepancy(
                        symbol=order.symbol or None,
                        kind="open_order_missing_identity",
                        detail="broker open order missing order_id or symbol",
                    )
                )
                continue
            local_order = local_orders.get(order.order_id)
            if local_order is None or str(local_order.get("status", "")).lower() != str(order.status).lower():
                reason_codes.append("RESYNC_OPEN_ORDER_RECOVERED")
                open_orders_recovered.append(
                    {
                        "order_id": order.order_id,
                        "symbol": order.symbol,
                        "side": order.side,
                        "status": order.status,
                        "qty": order.qty,
                        "filled_qty": order.filled_qty,
                    }
                )
            if order.submitted_at:
                self._order_submission_ts.setdefault(order.order_id, order.submitted_at)
            if order.side:
                self._order_submission_side.setdefault(order.order_id, order.side.lower())
                if order.side.lower() == "sell":
                    self._order_exit_reason.setdefault(order.order_id, "recovered_open_order")

        for order in recent_orders:
            order_id = str(order.order_id or "").strip()
            filled_qty = _optional_float(order.filled_qty) or 0.0
            if not order_id or filled_qty <= 0:
                continue
            filled_at = _parse_iso_timestamp(order.filled_at) or _parse_iso_timestamp(order.submitted_at)
            if filled_at is None or filled_at < lookback_cutoff:
                if order_id not in local_orders:
                    reason_codes.append("RESYNC_LOOKBACK_INSUFFICIENT")
                continue
            local_order = local_orders.get(order_id)
            local_filled_qty = _optional_float(local_order.get("filled_qty")) if local_order is not None else None
            if local_order is None or (local_filled_qty or 0.0) < filled_qty:
                reason_codes.append("RESYNC_RECENT_FILL_RECOVERED")
                recent_fills_recovered.append(
                    {
                        "order_id": order_id,
                        "symbol": order.symbol,
                        "side": order.side,
                        "filled_qty": filled_qty,
                        "filled_at": order.filled_at,
                    }
                )
                self._logged_fills.add((order_id, round(filled_qty, 6)))

        unresolved_position = any(item.kind == "lookback_insufficient" for item in discrepancies)
        unresolved_order = any(item.kind == "open_order_missing_identity" for item in discrepancies)
        if unresolved_order:
            reason_codes.append("RESYNC_UNRESOLVED_ORDER_STATE")
        if unresolved_position:
            reason_codes.append("RESYNC_UNRESOLVED_POSITION_STATE")

        status = ResyncStatus.OK
        gate_allows_entries = True
        gate_allows_exits = True
        if unresolved_order:
            status = ResyncStatus.FAILED
            gate_allows_entries = False
            gate_allows_exits = False
        elif unresolved_position or "RESYNC_LOOKBACK_INSUFFICIENT" in reason_codes:
            status = ResyncStatus.DEGRADED
            gate_allows_entries = False
            gate_allows_exits = True

        result = ResyncResult(
            status=status,
            started_at_utc=started_at,
            completed_at_utc=(current_time_utc or datetime.now(timezone.utc)).isoformat(),
            positions_recovered=tuple(positions_recovered),
            open_orders_recovered=tuple(open_orders_recovered),
            recent_fills_recovered=tuple(recent_fills_recovered),
            discrepancies=tuple(discrepancies),
            reason_codes=tuple(sorted(set(reason_codes))),
            gate_allows_entries=gate_allows_entries,
            gate_allows_exits=gate_allows_exits,
        )
        self.blog.lifecycle(
            "resync.completed" if status != ResyncStatus.FAILED else "resync.failed",
            status=result.status.value,
            reason_codes=list(result.reason_codes),
            gate_allows_entries=result.gate_allows_entries,
            gate_allows_exits=result.gate_allows_exits,
        )
        return self._set_startup_resync_result(result)

    def _validate_persisted_symbol_state(self) -> None:
        latest_run = self.storage.get_latest_run()
        if latest_run is None:
            return
        persisted_symbols = normalize_symbols(self.storage.get_latest_snapshot_symbols())
        if not persisted_symbols or symbols_match(self.active_symbols, persisted_symbols):
            return
        latest_ts = str(latest_run.get("timestamp_utc", "") or "")
        current_text = format_symbol_list(self.active_symbols)
        persisted_text = format_symbol_list(persisted_symbols)
        message = (
            "Persisted symbol state belongs to a different run and will be ignored for this session. "
            f"current=[{current_text}] persisted=[{persisted_text}] "
            f"persisted_snapshot_ts={latest_ts or 'unknown'} session_id={self.session_id}"
        )
        logger.warning(message)
        print(f"[STATE] {message}")
        self.blog.symbol_state_mismatch(
            current_symbols=self.active_symbols,
            persisted_symbols=persisted_symbols,
            persisted_snapshot_ts=latest_ts or None,
            action="ignore_for_current_session",
            session_id=self.session_id,
        )

    def _disable_ml_trading(self, reason: str, exc: Exception | None = None) -> None:
        if self._ml_disabled_reason is not None:
            return
        self._ml_disabled_reason = reason
        if exc is not None:
            logger.error("ML trading disabled: %s (%s)", reason, exc)
        else:
            logger.error("ML trading disabled: %s", reason)

    def _strategy_for_symbol(self, symbol: str) -> Strategy:
        cached = self._strategy_cache.get(symbol)
        if cached is not None:
            return cached
        strategy_mode = self._symbol_strategy_modes.get(symbol, self.config.strategy_mode)
        strategy = Strategy(
            StrategyConfig(
                strategy_mode=strategy_mode,
                ml_probability_buy=self.config.ml_probability_buy,
                ml_probability_sell=self.config.ml_probability_sell,
                entry_threshold_pct=self.config.entry_threshold_pct,
                threshold_mode=self.config.threshold_mode,
                atr_multiple=self.config.atr_multiple,
                atr_percentile_threshold=self.config.atr_percentile_threshold,
                time_window_mode=self.config.time_window_mode,
                regime_filter_enabled=self.config.regime_filter_enabled,
                orb_filter_mode=self.config.orb_filter_mode,
                breakout_exit_style=self.config.breakout_exit_style,
                breakout_tight_stop_fraction=self.config.breakout_tight_stop_fraction,
                breakout_max_stop_pct=self.config.breakout_max_stop_pct,
                sma_stop_pct=self.config.sma_stop_pct,
                mean_reversion_exit_style=self.config.mean_reversion_exit_style,
                mean_reversion_max_atr_percentile=self.config.mean_reversion_max_atr_percentile,
                mean_reversion_trend_filter=self.config.mean_reversion_trend_filter,
                mean_reversion_trend_slope_filter=self.config.mean_reversion_trend_slope_filter,
                mean_reversion_stop_pct=self.config.mean_reversion_stop_pct,
                vwap_z_entry_threshold=self.config.vwap_z_entry_threshold,
                vwap_z_stop_atr_multiple=self.config.vwap_z_stop_atr_multiple,
                min_atr_percentile=self.config.min_atr_percentile,
                max_adx_threshold=self.config.max_adx_threshold,
                trend_pullback_min_adx=self.config.trend_pullback_min_adx,
                trend_pullback_min_slope=self.config.trend_pullback_min_slope,
                trend_pullback_entry_threshold=self.config.trend_pullback_entry_threshold,
                trend_pullback_min_atr_percentile=self.config.trend_pullback_min_atr_percentile,
                trend_pullback_max_atr_percentile=self.config.trend_pullback_max_atr_percentile,
                trend_pullback_exit_style=self.config.trend_pullback_exit_style,
                trend_pullback_hold_bars=self.config.trend_pullback_hold_bars,
                trend_pullback_take_profit_pct=self.config.trend_pullback_take_profit_pct,
                trend_pullback_stop_pct=self.config.trend_pullback_stop_pct,
                momentum_breakout_lookback_bars=self.config.momentum_breakout_lookback_bars,
                momentum_breakout_entry_buffer_pct=self.config.momentum_breakout_entry_buffer_pct,
                momentum_breakout_min_adx=self.config.momentum_breakout_min_adx,
                momentum_breakout_min_slope=self.config.momentum_breakout_min_slope,
                momentum_breakout_min_atr_percentile=self.config.momentum_breakout_min_atr_percentile,
                momentum_breakout_exit_style=self.config.momentum_breakout_exit_style,
                momentum_breakout_hold_bars=self.config.momentum_breakout_hold_bars,
                momentum_breakout_stop_pct=self.config.momentum_breakout_stop_pct,
                momentum_breakout_take_profit_pct=self.config.momentum_breakout_take_profit_pct,
                volatility_expansion_lookback_bars=self.config.volatility_expansion_lookback_bars,
                volatility_expansion_entry_buffer_pct=self.config.volatility_expansion_entry_buffer_pct,
                volatility_expansion_max_atr_percentile=self.config.volatility_expansion_max_atr_percentile,
                volatility_expansion_trend_filter=self.config.volatility_expansion_trend_filter,
                volatility_expansion_min_slope=self.config.volatility_expansion_min_slope,
                volatility_expansion_use_volume_confirm=self.config.volatility_expansion_use_volume_confirm,
                volatility_expansion_exit_style=self.config.volatility_expansion_exit_style,
                volatility_expansion_hold_bars=self.config.volatility_expansion_hold_bars,
                volatility_expansion_stop_pct=self.config.volatility_expansion_stop_pct,
                volatility_expansion_take_profit_pct=self.config.volatility_expansion_take_profit_pct,
                bb_period=self.config.bb_period,
                bb_stddev_mult=self.config.bb_stddev_mult,
                bb_width_lookback=self.config.bb_width_lookback,
                bb_squeeze_quantile=self.config.bb_squeeze_quantile,
                bb_slope_lookback=self.config.bb_slope_lookback,
                bb_use_volume_confirm=self.config.bb_use_volume_confirm,
                bb_volume_mult=self.config.bb_volume_mult,
                bb_breakout_buffer_pct=self.config.bb_breakout_buffer_pct,
                bb_min_mid_slope=self.config.bb_min_mid_slope,
                bb_trend_filter=self.config.bb_trend_filter,
                bb_exit_mode=self.config.bb_exit_mode,
            )
        )
        self._strategy_cache[symbol] = strategy
        return strategy

    def get_account(self) -> Any:
        return self.broker.get_account()

    def perform_startup_preflight(self, *, execute_orders: bool) -> dict[str, object]:
        account = self.get_account()
        positions = self.get_positions_by_symbol()
        market_open: bool | None = None
        if execute_orders:
            market_open = self._is_market_open()
        summary = {
            "execute_orders": execute_orders,
            "cash": float(account.cash),
            "buying_power": float(account.buying_power),
            "equity": float(account.equity),
            "position_count": len(positions),
            "market_open": market_open,
            "feed_status": self.get_price_feed_status(),
        }
        self.blog.lifecycle("startup.ready", **summary)
        return summary

    def _log_skip(self, reason: str, detail: str) -> None:
        logger.info("%s %s", reason, detail)
        print(f"{reason}: {detail}")

    def _position_holding_minutes(self, symbol: str, now_utc: datetime) -> float | None:
        first_seen = self._position_first_seen_utc.get(symbol)
        if first_seen is None:
            return None
        return max(0.0, (now_utc - first_seen).total_seconds() / 60.0)

    def _hydrate_open_position_state(self, positions: dict[str, Position]) -> None:
        persisted_positions = _load_open_position_state_from_logs(getattr(self.blog, "log_root", "logs"))
        for symbol, position in positions.items():
            if not _has_open_position(position):
                continue
            persisted = persisted_positions.get(symbol)
            if symbol not in self._position_entry_price:
                persisted_entry_price = _optional_float(
                    persisted.get("entry_price") if persisted is not None else None
                )
                broker_entry_price = _optional_float(getattr(position, "avg_entry_price", None))
                entry_price = persisted_entry_price if persisted_entry_price is not None else broker_entry_price
                if entry_price is not None:
                    self._position_entry_price[symbol] = entry_price
            if symbol not in self._position_qty:
                persisted_qty = _optional_float(persisted.get("qty") if persisted is not None else None)
                quantity = persisted_qty if persisted_qty is not None else _position_qty_value(position)
                if quantity != 0:
                    self._position_qty[symbol] = quantity
            if symbol not in self._position_entry_ts:
                persisted_decision_ts = persisted.get("decision_ts") if persisted is not None else None
                if isinstance(persisted_decision_ts, str) and persisted_decision_ts:
                    self._position_entry_ts[symbol] = persisted_decision_ts
            if symbol not in self._position_entry_branch:
                persisted_entry_branch = persisted.get("entry_branch") if persisted is not None else None
                if isinstance(persisted_entry_branch, str) and persisted_entry_branch:
                    self._position_entry_branch[symbol] = persisted_entry_branch

    def _update_position_holding_state(
        self,
        positions: dict[str, Position],
        observed_at_utc: datetime,
    ) -> None:
        self._hydrate_open_position_state(positions)
        for symbol in positions:
            entry_timestamp = _parse_iso_timestamp(self._position_entry_ts.get(symbol))
            if entry_timestamp is not None:
                self._position_first_seen_utc.setdefault(symbol, entry_timestamp)
            else:
                self._position_first_seen_utc.setdefault(symbol, observed_at_utc)
        for symbol in list(self._position_first_seen_utc):
            if symbol not in positions:
                del self._position_first_seen_utc[symbol]
                self._breakout_stored_stop.pop(symbol, None)

    def _should_process_decision_timestamp(self, timestamp: datetime) -> bool:
        last_processed = self._last_processed_decision_timestamp
        if last_processed is not None and timestamp <= last_processed:
            self._log_skip(
                "SKIP_DUPLICATE_BAR",
                f"decision_timestamp={timestamp.isoformat()} last_processed={last_processed.isoformat()}",
            )
            return False
        self._last_processed_decision_timestamp = timestamp
        return True

    def _claim_global_decision_execution(self, timestamp: datetime) -> bool:
        decision_ts = timestamp.isoformat()
        claimed = self.storage.claim_decision_timestamp(
            decision_ts,
            datetime.now(timezone.utc).isoformat(),
        )
        if not claimed:
            self._log_skip(
                "SKIP_DUPLICATE_BAR_GLOBAL",
                f"decision_timestamp={decision_ts}",
            )
        return claimed

    def _is_regular_hours(self, timestamp: datetime | pd.Timestamp) -> bool:
        stamp = pd.Timestamp(timestamp)
        if stamp.tzinfo is None:
            stamp = stamp.tz_localize("UTC")
        time_et = stamp.tz_convert("America/New_York").time()
        return dt_time(9, 30) <= time_et < dt_time(16, 0)

    def _validate_startup_market_data(self, decision_timestamp: datetime) -> None:
        session_date_et = decision_timestamp.astimezone(_ET).date().isoformat()
        if self._startup_market_data_validated_for_et_date == session_date_et:
            return

        blocking_reasons: list[str] = []
        for symbol in self.active_symbols:
            intraday_bars = self._get_intraday_bars(
                symbol,
                2,
                decision_timestamp=decision_timestamp,
            )
            if not intraday_bars:
                blocking_reasons.append(f"{symbol}:no_completed_bar")
                continue
            latest_bar_close = self._latest_bar_close_time(intraday_bars)
            latest_bar_date_et = latest_bar_close.astimezone(_ET).date()
            if latest_bar_date_et != decision_timestamp.astimezone(_ET).date():
                blocking_reasons.append(
                    f"{symbol}:previous_session_bar:{latest_bar_close.isoformat()}"
                )
                continue
            bar_delay_seconds = max(0.0, (decision_timestamp - latest_bar_close).total_seconds())
            if bar_delay_seconds > self.config.max_data_delay_seconds:
                blocking_reasons.append(
                    f"{symbol}:bar_age={bar_delay_seconds:.0f}s"
                )

        if blocking_reasons:
            sample = ", ".join(blocking_reasons[:5])
            remaining = len(blocking_reasons) - min(len(blocking_reasons), 5)
            if remaining > 0:
                sample = f"{sample}, +{remaining} more"
            raise StaleMarketDataError(
                "startup market data not ready; refusing to trade until every symbol has "
                f"a same-session completed bar ({sample})"
            )

        self._startup_market_data_validated_for_et_date = session_date_et

    def _log_risk_check(
        self,
        *,
        symbol: str,
        decision_ts: str,
        action: str,
        allowed: bool,
        block_reason: str | None,
        snapshot: BotSnapshot,
        open_positions: int,
        live_price: float | None = None,
        signal_price: float | None = None,
        price_deviation_bps: float | None = None,
        live_price_age_s: float | None = None,
        detail: str | None = None,
        in_entry_window: bool | None = None,
        remaining_buying_power: float | None = None,
        trade_budget: float | None = None,
        recent_order_count: int | None = None,
    ) -> None:
        self.blog.risk_check(
            symbol=symbol,
            decision_ts=decision_ts,
            action=action,
            allowed=allowed,
            block_reason=block_reason,
            open_positions=open_positions,
            max_positions=self.config.max_open_positions,
            daily_pnl=snapshot.daily_pnl,
            daily_limit=self.config.max_daily_loss_usd,
            live_price=live_price,
            signal_price=signal_price,
            price_deviation_bps=price_deviation_bps,
            live_price_age_s=live_price_age_s,
            detail=detail,
            in_entry_window=in_entry_window,
            remaining_buying_power=remaining_buying_power,
            trade_budget=trade_budget,
            recent_order_count=recent_order_count,
            max_orders_per_minute=self.config.max_orders_per_minute,
        )

    def _is_market_open(self) -> bool:
        return self.broker.is_market_open()

    def get_price_feed_status(self) -> str:
        return self.broker.get_price_feed_status()

    def get_latest_price_with_age(self, symbol: str) -> tuple[float, float]:
        return self.broker.get_latest_price_with_age(symbol)

    def get_latest_price(self, symbol: str) -> float:
        price, _ = self.get_latest_price_with_age(symbol)
        return price

    def get_positions_by_symbol(self) -> dict[str, Position]:
        positions = self.broker.list_positions()
        return {position.symbol: position for position in positions}

    def _bar_interval(self) -> timedelta:
        return timedelta(minutes=self.config.bar_timeframe_minutes)

    def get_decision_timestamp(self, now: datetime | None = None) -> datetime:
        current_time = now or datetime.now(timezone.utc)
        bar_seconds = int(self._bar_interval().total_seconds())
        decision_unix = int(current_time.timestamp()) // bar_seconds * bar_seconds
        return datetime.fromtimestamp(decision_unix, tz=timezone.utc)

    def _get_bar_start_time(self, bar: BrokerBar) -> datetime:
        return bar.timestamp

    def _get_bars(
        self,
        symbol: str,
        bars_needed: int,
        timeframe_minutes: int,
        decision_timestamp: datetime | None = None,
    ) -> list[BrokerBar]:
        if bars_needed <= 0:
            raise RuntimeError("bars_needed must be greater than zero.")

        aligned_decision_timestamp = decision_timestamp or self.get_decision_timestamp()
        cache_key = (symbol, timeframe_minutes, bars_needed, aligned_decision_timestamp.isoformat())
        cached_bars = self._bars_cache.get(cache_key)
        if cached_bars is not None:
            return cached_bars

        trading_minutes_per_day = 390
        trading_days_needed = max(3, math.ceil((bars_needed * timeframe_minutes) / trading_minutes_per_day))
        start = aligned_decision_timestamp - timedelta(days=trading_days_needed * 6)
        request_end = aligned_decision_timestamp + timedelta(minutes=timeframe_minutes)
        bars = self.broker.get_bars(
            symbol,
            timeframe_minutes=timeframe_minutes,
            start=start,
            end=request_end,
        )
        completed_bars = [
            bar
            for bar in bars
            if (self._get_bar_start_time(bar) + timedelta(minutes=timeframe_minutes)) <= aligned_decision_timestamp
        ]
        result = completed_bars[-bars_needed:]
        self._bars_cache[cache_key] = result
        return result

    def _get_intraday_bars(
        self,
        symbol: str,
        bars_needed: int,
        decision_timestamp: datetime | None = None,
    ) -> list[BrokerBar]:
        return self._get_bars(
            symbol,
            bars_needed,
            self.config.bar_timeframe_minutes,
            decision_timestamp=decision_timestamp,
        )

    def _get_hourly_regime(self, symbol: str, decision_timestamp: datetime) -> bool | None:
        cache_key = (symbol, decision_timestamp.isoformat())
        cached = self._hourly_regime_cache.get(cache_key)
        if cached is not None or cache_key in self._hourly_regime_cache:
            return cached

        hourly_bars = self._get_bars(
            symbol,
            REGIME_SMA_PERIOD + 1,
            REGIME_TIMEFRAME_MINUTES,
            decision_timestamp=decision_timestamp,
        )
        if len(hourly_bars) < REGIME_SMA_PERIOD:
            self._hourly_regime_cache[cache_key] = None
            return None

        hourly_closes = [float(bar.close) for bar in hourly_bars]
        sma_50 = sum(hourly_closes[-REGIME_SMA_PERIOD:]) / REGIME_SMA_PERIOD
        bullish = hourly_closes[-1] > sma_50
        self._hourly_regime_cache[cache_key] = bullish
        return bullish

    def _latest_bar_close_time(self, bars: list[BrokerBar]) -> datetime:
        if not bars:
            raise RuntimeError("No completed bars available.")
        return self._get_bar_start_time(bars[-1]) + self._bar_interval()

    def get_sma(self, symbol: str, bars: int, decision_timestamp: datetime | None = None) -> float:
        if bars <= 0:
            raise RuntimeError("SMA_BARS must be greater than zero.")

        intraday_bars = self._get_intraday_bars(symbol, bars, decision_timestamp=decision_timestamp)
        closes = [float(bar.close) for bar in intraday_bars][-bars:]
        if len(closes) < max(5, bars // 2):
            raise RuntimeError(f"Not enough intraday bars for {symbol}: got {len(closes)}")

        return sum(closes) / len(closes)

    def _mean(self, values: list[float]) -> float:
        return sum(values) / len(values)

    def _stddev(self, values: list[float], mean_value: float) -> float:
        variance = sum((value - mean_value) ** 2 for value in values) / max(1, len(values))
        return math.sqrt(max(variance, 1e-12))

    def _build_feature_vector(self, closes: list[float], volumes: list[float], index: int) -> list[float]:
        price = closes[index]
        ret_1 = (price / closes[index - 1]) - 1.0
        ret_3 = (price / closes[index - 3]) - 1.0
        ret_5 = (price / closes[index - 5]) - 1.0
        sma_10 = self._mean(closes[index - 9 : index + 1])
        sma_20 = self._mean(closes[index - 19 : index + 1])
        vol_returns = [
            (closes[j] / closes[j - 1]) - 1.0
            for j in range(index - 9, index + 1)
            if j - 1 >= 0
        ]
        avg_volume_10 = self._mean(volumes[index - 9 : index + 1])
        return [
            ret_1,
            ret_3,
            ret_5,
            (price / sma_10) - 1.0,
            (price / sma_20) - 1.0,
            self._stddev(vol_returns, self._mean(vol_returns)),
            (volumes[index] / max(avg_volume_10, 1.0)) - 1.0,
        ]

    def get_ml_signal(self, symbol: str, decision_timestamp: datetime | None = None) -> MlSignal:
        if self._ml_disabled_reason is not None:
            raise RuntimeError(f"ML trading disabled: {self._ml_disabled_reason}")
        if offline_to_ml_signal is None:
            self._disable_ml_trading("ml.predict import failed", _ML_PREDICT_IMPORT_ERROR)
            raise RuntimeError(f"ML trading disabled: {self._ml_disabled_reason}")

        bars_needed = max(self.config.sma_bars, 25)
        aligned_decision_timestamp = decision_timestamp or self.get_decision_timestamp()
        intraday_bars = self._get_intraday_bars(
            symbol,
            bars_needed,
            decision_timestamp=aligned_decision_timestamp,
        )
        closes = [float(bar.close) for bar in intraday_bars]
        volumes = [float(getattr(bar, "volume", 0.0) or 0.0) for bar in intraday_bars]
        latest_index = len(closes) - 1
        if latest_index < 19:
            raise RuntimeError(f"Not enough bars to score ML signal for {symbol}: got {len(closes)}")

        features = self._build_feature_vector(closes, volumes, latest_index)
        try:
            ml_signal = offline_to_ml_signal(
                features,
                buy_threshold=self.config.ml_probability_buy,
                sell_threshold=self.config.ml_probability_sell,
            )
        except Exception as exc:
            self._disable_ml_trading("offline model load or inference failed", exc)
            raise RuntimeError(f"ML trading disabled: {self._ml_disabled_reason}") from exc

        logger.debug(
            "ML inference symbol=%s prob=%.3f buy_thr=%.3f sell_thr=%.3f model=%s",
            symbol,
            ml_signal.probability_up,
            ml_signal.buy_threshold,
            ml_signal.sell_threshold,
            ml_signal.model_name,
        )
        return ml_signal

    def evaluate_symbol(
        self,
        symbol: str,
        position: Position | None,
        decision_timestamp: datetime | None = None,
    ) -> SymbolEvaluation:
        aligned_decision_timestamp = decision_timestamp or self.get_decision_timestamp()
        holding = _has_long_position(position)
        short_position = _has_short_position(position)
        strategy = self._strategy_for_symbol(symbol)
        effective_strategy_mode = strategy.config.strategy_mode
        bars_needed = max(
            self.config.sma_bars,
            25,
            50,  # minimum for trend_sma (50-bar SMA used by mean_reversion_trend_filter)
            55,  # minimum to compute the 50-bar slope used by trend-aware strategies
            strategy.config.momentum_breakout_lookback_bars + 1,
            strategy.config.volatility_expansion_lookback_bars + 1,
            estimate_atr_percentile_lookback_bars(self.config.bar_timeframe_minutes),
            estimate_bollinger_lookback_bars(
                strategy.config.bb_period,
                strategy.config.bb_width_lookback,
                strategy.config.bb_slope_lookback,
            ),
            estimate_regime_lookback_bars(self.config.bar_timeframe_minutes),
        )
        intraday_bars = self._get_intraday_bars(
            symbol,
            bars_needed,
            decision_timestamp=aligned_decision_timestamp,
        )
        closes = [float(bar.close) for bar in intraday_bars]
        highs = [float(bar.high) for bar in intraday_bars]
        lows = [float(bar.low) for bar in intraday_bars]
        timestamps = [pd.Timestamp(getattr(bar, "timestamp")) for bar in intraday_bars]
        if effective_strategy_mode in (STRATEGY_MODE_BREAKOUT, STRATEGY_MODE_ORB):
            min_history_bars = 2
        elif effective_strategy_mode == STRATEGY_MODE_MOMENTUM_BREAKOUT:
            min_history_bars = max(
                55,
                self.config.sma_bars,
                strategy.config.momentum_breakout_lookback_bars + 1,
            )
        elif effective_strategy_mode == STRATEGY_MODE_VOLATILITY_EXPANSION:
            min_history_bars = max(
                55,
                self.config.sma_bars,
                strategy.config.volatility_expansion_lookback_bars + 1,
                estimate_bollinger_lookback_bars(
                    strategy.config.bb_period,
                    strategy.config.bb_width_lookback,
                    strategy.config.bb_slope_lookback,
                ),
            )
        elif effective_strategy_mode in (STRATEGY_MODE_BOLLINGER_SQUEEZE, STRATEGY_MODE_HYBRID_BB_MR):
            min_history_bars = estimate_bollinger_lookback_bars(
                strategy.config.bb_period,
                strategy.config.bb_width_lookback,
                strategy.config.bb_slope_lookback,
            )
        else:
            min_history_bars = max(20, self.config.sma_bars)
        if len(closes) < min_history_bars:
            raise RuntimeError(
                f"Not enough completed bars for {symbol} at {aligned_decision_timestamp.isoformat()}: got {len(closes)}"
            )
        latest_bar_close = self._latest_bar_close_time(intraday_bars)
        bar_delay_seconds = max(0.0, (aligned_decision_timestamp - latest_bar_close).total_seconds())
        self.blog.bar_received(
            symbol=symbol,
            bar_close=closes[-1],
            bar_volume=float(getattr(intraday_bars[-1], "volume", 0) or 0),
            bar_ts=self._get_bar_start_time(intraday_bars[-1]).isoformat(),
            bar_age_s=bar_delay_seconds,
            decision_ts=aligned_decision_timestamp.isoformat(),
        )
        if bar_delay_seconds > self.config.max_data_delay_seconds:
            raise StaleMarketDataError(
                f"Stale completed bars for {symbol}: latest close {latest_bar_close.isoformat()}"
            )

        price = closes[-1]
        sma = (
            sum(closes[-self.config.sma_bars :]) / self.config.sma_bars
            if len(closes) >= self.config.sma_bars
            else price
        )
        trend_sma = sum(closes[-50:]) / 50 if len(closes) >= 50 else None
        ml_signal = (
            self.get_ml_signal(symbol, decision_timestamp=aligned_decision_timestamp)
            if effective_strategy_mode in (STRATEGY_MODE_ML, STRATEGY_MODE_HYBRID)
            else MlSignal(
                probability_up=0.5,
                confidence=0.0,
                training_rows=0,
                model_age_seconds=0.0,
                feature_names=(),
                buy_threshold=self.config.ml_probability_buy,
                sell_threshold=self.config.ml_probability_sell,
                validation_rows=0,
                model_name="dummy",
            )
        )
        atr_pct_values = calculate_atr_pct_values(highs, lows, closes)
        atr_percentiles = calculate_atr_percentile_series(timestamps, highs, lows, closes)
        opening_range_highs, opening_range_lows = calculate_opening_range_series(timestamps, highs, lows)
        volumes = [float(getattr(bar, "volume", 0.0) or 0.0) for bar in intraday_bars]
        vwap_values = calculate_vwap_series(timestamps, highs, lows, closes, volumes)
        adx_values = calculate_adx_series(highs, lows, closes)
        bb_features = calculate_bollinger_squeeze_features(
            closes,
            volumes,
            period=strategy.config.bb_period,
            stddev_mult=strategy.config.bb_stddev_mult,
            width_lookback=strategy.config.bb_width_lookback,
            squeeze_quantile=strategy.config.bb_squeeze_quantile,
            slope_lookback=strategy.config.bb_slope_lookback,
            use_volume_confirm=strategy.config.bb_use_volume_confirm,
            volume_mult=strategy.config.bb_volume_mult,
        )
        recent_volumes = volumes[-20:]
        avg_volume = (sum(recent_volumes) / len(recent_volumes)) if recent_volumes else None
        current_volume = float(getattr(intraday_bars[-1], "volume", 0.0) or 0.0)
        volume_ratio = (current_volume / avg_volume) if avg_volume and avg_volume > 0 else None
        recent_atr_pct_values = [value for value in atr_pct_values[-20:] if value is not None]
        avg_atr_pct = (
            sum(recent_atr_pct_values) / len(recent_atr_pct_values)
            if recent_atr_pct_values
            else None
        )
        current_atr_pct = atr_pct_values[-1] if atr_pct_values else None
        volatility_ratio = (
            current_atr_pct / avg_atr_pct
            if current_atr_pct is not None and avg_atr_pct is not None and avg_atr_pct > 0
            else None
        )
        bullish_regime = self._get_hourly_regime(symbol, aligned_decision_timestamp)

        # Compute and cache the capped breakout stop at entry time (BREAKOUT mode only).
        # If we're now holding and have no stored stop yet, compute it now.
        or_low = opening_range_lows[-1] if opening_range_lows else None
        if (
            holding
            and effective_strategy_mode == STRATEGY_MODE_BREAKOUT
            and symbol not in self._breakout_stored_stop
            and or_low is not None
            and position is not None
        ):
            entry = _float_or_default(getattr(position, "avg_entry_price", None))
            capped_stop = get_capped_breakout_stop_price(entry, or_low, strategy.config.breakout_max_stop_pct)
            self._breakout_stored_stop[symbol] = capped_stop
            logger.info(
                "breakout entry symbol=%s entry=%.4f or_low=%.4f capped_stop=%.4f dist_pct=%.2f%%",
                symbol, entry, or_low, capped_stop, (entry - capped_stop) / entry * 100,
            )

        effective_stop_price = self._breakout_stored_stop.get(symbol) if holding else None

        _window_open = self._is_in_entry_window(aligned_decision_timestamp.astimezone(_ET))
        _atr_pct_now = atr_pct_values[-1] if atr_pct_values else None
        _atr_pct_now_val = atr_percentiles[-1]
        current_vwap = vwap_values[-1] if vwap_values else None
        current_adx = adx_values[-1] if adx_values else None
        trend_pullback_bars_held = 0
        momentum_breakout_bars_held = 0
        volatility_expansion_bars_held = 0
        holding_minutes = self._position_holding_minutes(symbol, aligned_decision_timestamp)
        if holding and holding_minutes is not None and self.config.bar_timeframe_minutes > 0:
            trend_pullback_bars_held = max(
                1,
                math.ceil(holding_minutes / self.config.bar_timeframe_minutes),
            )
            momentum_breakout_bars_held = trend_pullback_bars_held
            volatility_expansion_bars_held = trend_pullback_bars_held
        current_trend_slope = (
            trend_sma - (sum(closes[-51:-1]) / 50)
            if trend_sma is not None and len(closes) >= 51
            else None
        )
        recent_breakout_high = (
            max(highs[-(max(strategy.config.momentum_breakout_lookback_bars, strategy.config.volatility_expansion_lookback_bars) + 1):-1])
            if len(highs) >= max(strategy.config.momentum_breakout_lookback_bars, strategy.config.volatility_expansion_lookback_bars) + 1
            else None
        )
        if short_position:
            explanation_summary = (
                "HOLD because the broker reports a short position. "
                "Long-only strategy evaluation is suppressed until the short is flattened."
            )
            self.blog.signal(
                symbol=symbol,
                decision_ts=aligned_decision_timestamp.isoformat(),
                bar_close=price,
                sma=sma,
                trend_sma=trend_sma,
                atr_pct=_atr_pct_now,
                atr_percentile=_atr_pct_now_val,
                volume_ratio=volume_ratio,
                action="HOLD",
                holding=True,
                trend_filter_pass=None,
                atr_filter_pass=None,
                window_open=_window_open,
                rejection="short_position",
                ml_prob=ml_signal.probability_up if ml_signal.probability_up != 0.5 else None,
                extra_fields={
                    "strategy_mode": effective_strategy_mode,
                    "final_signal_reason": "short_position_unsupported",
                    "decision_summary": explanation_summary,
                    "entry_reference_price": None,
                    "exit_reference_price": None,
                    "stop_reference_price": None,
                    "vwap": round(current_vwap, 4) if current_vwap is not None else None,
                    "adx": round(current_adx, 4) if current_adx is not None else None,
                    "regime_state": (
                        "bullish" if bullish_regime is True else ("bearish" if bullish_regime is False else "unknown")
                    ),
                },
            )
            return SymbolEvaluation(
                price=price,
                sma=sma,
                ml_signal=ml_signal,
                action="HOLD",
                latest_bar_close_utc=latest_bar_close.isoformat(),
                hold_reason="short_position",
                final_signal_reason="short_position_unsupported",
            )
        decision_details = strategy.decide_action_details(
            price,
            sma,
            ml_signal,
            holding,
            _atr_pct_now,
            _atr_pct_now_val,
            time_window_open=_window_open,
            bullish_regime=bullish_regime,
            opening_range_high=opening_range_highs[-1] if opening_range_highs else None,
            opening_range_low=or_low,
            position_entry_price=_optional_float(getattr(position, "avg_entry_price", None)) if holding and position is not None else None,
            volume_ratio=volume_ratio,
            volatility_ratio=volatility_ratio,
            effective_stop_price=effective_stop_price,
            trend_sma=trend_sma,
            trend_sma_slope=current_trend_slope,
            vwap=current_vwap,
            adx=current_adx,
            bb_middle=bb_features["middle"][-1] if bb_features["middle"] else None,
            bb_upper=bb_features["upper"][-1] if bb_features["upper"] else None,
            bb_lower=bb_features["lower"][-1] if bb_features["lower"] else None,
            bb_prev_squeeze=bb_features["squeeze"][-2] if len(bb_features["squeeze"]) >= 2 else None,
            bb_mid_slope=bb_features["mid_slope"][-1] if bb_features["mid_slope"] else None,
            bb_bias=bb_features["bias"][-1] if bb_features["bias"] else None,
            bb_breakout_up=bb_features["breakout_up"][-1] if bb_features["breakout_up"] else None,
            bb_breakout_down=bb_features["breakout_down"][-1] if bb_features["breakout_down"] else None,
            bb_volume_confirm=bb_features["volume_confirm"][-1] if bb_features["volume_confirm"] else None,
            hybrid_entry_branch=self._position_entry_branch.get(symbol) if holding else None,
            trend_pullback_bars_held=trend_pullback_bars_held,
            recent_breakout_high=recent_breakout_high,
            momentum_breakout_bars_held=momentum_breakout_bars_held,
            volatility_expansion_bars_held=volatility_expansion_bars_held,
        )
        action = decision_details.action
        # --- structured signal log ---
        _trend_pass = (price >= trend_sma) if trend_sma is not None else None
        _atr_pass = (
            (_atr_pct_now_val <= self.config.mean_reversion_max_atr_percentile)
            if (_atr_pct_now_val is not None and self.config.mean_reversion_max_atr_percentile > 0)
            else None
        )
        _trend_filter_active = (
            strategy.config.strategy_mode != STRATEGY_MODE_MEAN_REVERSION
            or strategy.config.mean_reversion_trend_filter
        )
        _rejection = _determine_signal_rejection(
            action=action,
            holding=holding,
            trend_filter_active=_trend_filter_active,
            trend_pass=_trend_pass,
            atr_pass=_atr_pass if not holding else None,
        )
        entry_reference_price: float | None = None
        exit_reference_price: float | None = None
        stop_reference_price: float | None = None
        if effective_strategy_mode == STRATEGY_MODE_SMA:
            entry_reference_price = strategy._entry_threshold_price(sma, _atr_pct_now)
            exit_reference_price = sma if holding else None
            if holding and self.config.sma_stop_pct > 0 and position is not None:
                stop_reference_price = _optional_float(getattr(position, "avg_entry_price", None))
                if stop_reference_price is not None:
                    stop_reference_price *= (1.0 - self.config.sma_stop_pct)
        elif effective_strategy_mode == STRATEGY_MODE_MEAN_REVERSION:
            entry_reference_price = (
                current_vwap - (strategy.config.vwap_z_entry_threshold * (_atr_pct_now * price))
                if current_vwap is not None and _atr_pct_now is not None
                else strategy._reversion_threshold_price(sma, _atr_pct_now)
            )
            exit_reference_price = current_vwap if current_vwap is not None else sma
            if holding and self.config.mean_reversion_stop_pct > 0 and position is not None:
                stop_reference_price = _optional_float(getattr(position, "avg_entry_price", None))
                if stop_reference_price is not None:
                    stop_reference_price *= (1.0 - self.config.mean_reversion_stop_pct)
        elif effective_strategy_mode == STRATEGY_MODE_TREND_PULLBACK:
            entry_reference_price = strategy._trend_pullback_entry_price(sma)
            if holding and self.config.trend_pullback_stop_pct > 0 and position is not None:
                stop_reference_price = _optional_float(getattr(position, "avg_entry_price", None))
                if stop_reference_price is not None:
                    stop_reference_price *= (1.0 - self.config.trend_pullback_stop_pct)
        elif effective_strategy_mode in (STRATEGY_MODE_BREAKOUT, STRATEGY_MODE_ORB):
            entry_reference_price = opening_range_highs[-1] if opening_range_highs else None
            stop_reference_price = effective_stop_price
        elif effective_strategy_mode == STRATEGY_MODE_BOLLINGER_SQUEEZE:
            entry_reference_price = bb_features["upper"][-1] if bb_features["upper"] else None
            exit_reference_price = bb_features["middle"][-1] if bb_features["middle"] else None
        elif effective_strategy_mode == STRATEGY_MODE_HYBRID_BB_MR:
            if decision_details.hybrid_branch_active == "bollinger_breakout":
                entry_reference_price = bb_features["upper"][-1] if bb_features["upper"] else None
                exit_reference_price = bb_features["middle"][-1] if bb_features["middle"] else None
            else:
                entry_reference_price = strategy._reversion_threshold_price(sma, _atr_pct_now)
                exit_reference_price = current_vwap if current_vwap is not None else sma
        explanation_summary = _build_signal_explanation(
            strategy_mode=effective_strategy_mode,
            action=action,
            reason=decision_details.reason,
            price=price,
            sma=sma,
            holding=holding,
            time_window_open=_window_open,
            atr_percentile=_atr_pct_now_val,
            trend_sma=trend_sma,
            bullish_regime=bullish_regime,
            entry_reference_price=entry_reference_price,
            exit_reference_price=exit_reference_price,
            stop_reference_price=stop_reference_price,
            vwap=current_vwap,
            adx=current_adx,
            hybrid_branch_active=decision_details.hybrid_branch_active,
        )
        self.blog.signal(
            symbol=symbol,
            decision_ts=aligned_decision_timestamp.isoformat(),
            bar_close=price,
            sma=sma,
            trend_sma=trend_sma,
            atr_pct=_atr_pct_now,
            atr_percentile=_atr_pct_now_val,
            volume_ratio=volume_ratio,
            action=action,
            holding=holding,
            trend_filter_pass=_trend_pass if _trend_filter_active else None,
            atr_filter_pass=_atr_pass,
            window_open=_window_open,
            rejection=_rejection,
            ml_prob=ml_signal.probability_up if ml_signal.probability_up != 0.5 else None,
            extra_fields={
                "bb_mid": round(bb_features["middle"][-1], 4) if bb_features["middle"] and bb_features["middle"][-1] is not None else None,
                "bb_upper": round(bb_features["upper"][-1], 4) if bb_features["upper"] and bb_features["upper"][-1] is not None else None,
                "bb_lower": round(bb_features["lower"][-1], 4) if bb_features["lower"] and bb_features["lower"][-1] is not None else None,
                "bb_width": round(bb_features["width"][-1], 6) if bb_features["width"] and bb_features["width"][-1] is not None else None,
                "bb_squeeze": bb_features["squeeze"][-1] if bb_features["squeeze"] else None,
                "bb_bias": bb_features["bias"][-1] if bb_features["bias"] else None,
                "bb_mid_slope": round(bb_features["mid_slope"][-1], 6) if bb_features["mid_slope"] and bb_features["mid_slope"][-1] is not None else None,
                "bb_breakout_up": bb_features["breakout_up"][-1] if bb_features["breakout_up"] else None,
                "bb_breakout_down": bb_features["breakout_down"][-1] if bb_features["breakout_down"] else None,
                "bb_volume_confirm": bb_features["volume_confirm"][-1] if bb_features["volume_confirm"] else None,
                "hybrid_branch": decision_details.hybrid_branch,
                "hybrid_branch_active": decision_details.hybrid_branch_active,
                "hybrid_entry_branch": decision_details.hybrid_entry_branch,
                "hybrid_regime_branch": decision_details.hybrid_regime_branch,
                "mr_signal": decision_details.mr_signal,
                "strategy_mode": effective_strategy_mode,
                "final_signal_reason": decision_details.reason,
                "decision_summary": explanation_summary,
                "trend_filter_active": _trend_filter_active,
                "trend_sma_slope": round(current_trend_slope, 6) if current_trend_slope is not None else None,
                "entry_reference_price": round(entry_reference_price, 4) if entry_reference_price is not None else None,
                "exit_reference_price": round(exit_reference_price, 4) if exit_reference_price is not None else None,
                "stop_reference_price": round(stop_reference_price, 4) if stop_reference_price is not None else None,
                "vwap": round(current_vwap, 4) if current_vwap is not None else None,
                "adx": round(current_adx, 4) if current_adx is not None else None,
                "regime_state": (
                    "bullish" if bullish_regime is True else ("bearish" if bullish_regime is False else "unknown")
                ),
            },
        )
        return SymbolEvaluation(
            price=price,
            sma=sma,
            ml_signal=ml_signal,
            action=action,
            latest_bar_close_utc=latest_bar_close.isoformat(),
            hold_reason=_rejection,
            hybrid_branch_active=decision_details.hybrid_branch_active,
            hybrid_entry_branch=decision_details.hybrid_entry_branch,
            hybrid_regime_branch=decision_details.hybrid_regime_branch,
            final_signal_reason=decision_details.reason,
        )

    def decide(
        self,
        symbol: str,
        positions: dict[str, Position],
        decision_timestamp: datetime | None = None,
    ) -> str:
        return self.evaluate_symbol(symbol, positions.get(symbol), decision_timestamp=decision_timestamp).action

    def daily_pnl(self) -> float:
        account = self.get_account()
        return float(account.equity) - float(account.last_equity)

    def kill_switch_triggered(self) -> bool:
        pnl = self.daily_pnl()
        print(f"Daily PnL: {pnl:.2f}")
        return pnl <= -self.config.max_daily_loss_usd

    def build_snapshot(
        self,
        decision_timestamp: datetime | None = None,
        evaluate_signals: bool = True,
    ) -> BotSnapshot:
        account = self.get_account()
        positions = self.get_positions_by_symbol()
        aligned_decision_timestamp = decision_timestamp or self.get_decision_timestamp()
        self._update_position_holding_state(positions, aligned_decision_timestamp)
        symbols: list[SymbolSnapshot] = []

        for symbol in self.config.symbols:
            position = positions.get(symbol)
            quantity = _position_qty_value(position)
            holding = quantity != 0
            market_value = _float_or_default(getattr(position, "market_value", None)) if position is not None else 0.0
            holding_minutes = self._position_holding_minutes(symbol, aligned_decision_timestamp)
            try:
                if not evaluate_signals:
                    symbols.append(
                        SymbolSnapshot(
                            symbol=symbol,
                            price=None,
                            sma=None,
                            action="SNAPSHOT_ONLY",
                            holding=holding,
                            quantity=quantity,
                            market_value=market_value,
                            holding_minutes=holding_minutes,
                        )
                    )
                    continue
                evaluation = self.evaluate_symbol(
                    symbol,
                    position if quantity != 0 else None,
                    decision_timestamp=aligned_decision_timestamp,
                )

                symbols.append(
                    SymbolSnapshot(
                        symbol=symbol,
                        price=evaluation.price,
                        sma=evaluation.sma,
                        action=evaluation.action,
                        holding=holding,
                        quantity=quantity,
                        market_value=market_value,
                        ml_probability_up=evaluation.ml_signal.probability_up,
                        ml_confidence=evaluation.ml_signal.confidence,
                        ml_training_rows=evaluation.ml_signal.training_rows,
                        ml_buy_threshold=evaluation.ml_signal.buy_threshold,
                        ml_sell_threshold=evaluation.ml_signal.sell_threshold,
                        ml_model_name=evaluation.ml_signal.model_name,
                        holding_minutes=holding_minutes,
                        hold_reason=evaluation.hold_reason,
                        hybrid_branch_active=evaluation.hybrid_branch_active,
                        hybrid_entry_branch=evaluation.hybrid_entry_branch,
                        hybrid_regime_branch=evaluation.hybrid_regime_branch,
                        final_signal_reason=evaluation.final_signal_reason,
                    )
                )
            except StaleMarketDataError:
                raise
            except Exception as exc:
                symbols.append(
                    SymbolSnapshot(
                        symbol=symbol,
                        price=None,
                        sma=None,
                        action="ERROR",
                        holding=holding,
                        quantity=quantity,
                        market_value=market_value,
                        holding_minutes=holding_minutes,
                        error=str(exc),
                    )
                )

        daily_pnl = float(account.equity) - float(account.last_equity)
        return BotSnapshot(
            timestamp_utc=aligned_decision_timestamp.isoformat(),
            cash=float(account.cash),
            buying_power=float(account.buying_power),
            equity=float(account.equity),
            last_equity=float(account.last_equity),
            daily_pnl=daily_pnl,
            kill_switch_triggered=daily_pnl <= -self.config.max_daily_loss_usd,
            positions=positions,
            symbols=symbols,
        )

    def get_recent_orders(self, limit: int = 10) -> list[OrderSnapshot]:
        return self.broker.list_recent_orders(limit=limit)

    def get_open_orders(self) -> list[OrderSnapshot]:
        return self.broker.list_open_orders()

    def _count_recent_orders(self, window_seconds: int = 60) -> int:
        orders = self.get_recent_orders(limit=100)
        now = datetime.now(timezone.utc)
        count = 0
        for order in orders:
            submitted_at = _parse_iso_timestamp(order.submitted_at)
            if submitted_at is None:
                continue
            if (now - submitted_at).total_seconds() <= window_seconds:
                count += 1
        return count

    def _is_price_collar_breached(self, decision_price: float, live_price: float) -> bool:
        if decision_price <= 0:
            return True
        deviation_bps = abs((live_price / decision_price) - 1.0) * 10000.0
        return deviation_bps > self.config.max_price_deviation_bps

    def _is_symbol_exposure_exceeded(
        self,
        symbol: str,
        live_price: float,
        positions: dict[str, Position],
    ) -> bool:
        existing_position_value = 0.0
        if symbol in positions and getattr(positions[symbol], "market_value", None) is not None:
            existing_position_value = abs(_float_or_default(getattr(positions[symbol], "market_value", None)))
        proposed_qty = int(self.config.max_usd_per_trade // live_price)
        proposed_value = proposed_qty * live_price
        return (existing_position_value + proposed_value) > self.config.max_symbol_exposure_usd

    def get_last_run_cycle_report(self) -> RunCycleReport | None:
        return self._last_run_cycle_report

    def preview_execution(self, snapshot: BotSnapshot) -> list[ExecutionPreview]:
        try:
            snapshot_ts = datetime.fromisoformat(snapshot.timestamp_utc)
        except ValueError:
            snapshot_ts = self.get_decision_timestamp()
        if snapshot_ts.tzinfo is None:
            snapshot_ts = snapshot_ts.replace(tzinfo=timezone.utc)

        now_et = self._et_now()
        in_entry_window = self._is_in_entry_window(now_et)
        open_positions = len(snapshot.positions)
        remaining_buying_power = snapshot.buying_power
        previews: list[ExecutionPreview] = []

        try:
            open_orders = self.get_open_orders()
        except Exception:
            open_orders = []
        open_order_symbols = {
            str(getattr(order, "symbol", ""))
            for order in open_orders
            if getattr(order, "symbol", None)
        }
        recent_order_count = self._count_recent_orders(window_seconds=60)

        for item in snapshot.symbols:
            symbol = item.symbol
            action = item.action

            if item.error:
                previews.append(ExecutionPreview(symbol=symbol, action=action, status="ERROR", reason=item.error))
                continue
            if action not in ("BUY", "SELL"):
                previews.append(ExecutionPreview(symbol=symbol, action=action, status="NO_SIGNAL"))
                continue
            allowed_by_resync, resync_reason = self._resync_gate_for_action(
                action=action,
                symbol=symbol,
                positions=snapshot.positions,
            )
            if not allowed_by_resync:
                previews.append(
                    ExecutionPreview(symbol=symbol, action=action, status="BLOCKED", reason=resync_reason)
                )
                continue
            if symbol in open_order_symbols:
                previews.append(
                    ExecutionPreview(symbol=symbol, action=action, status="BLOCKED", reason="open_order_in_flight")
                )
                continue
            if recent_order_count >= self.config.max_orders_per_minute:
                previews.append(
                    ExecutionPreview(symbol=symbol, action=action, status="BLOCKED", reason="max_orders_per_minute")
                )
                continue

            if action == "BUY":
                if snapshot.kill_switch_triggered:
                    previews.append(
                        ExecutionPreview(symbol=symbol, action=action, status="BLOCKED", reason="kill_switch_active")
                    )
                    continue
                if not in_entry_window:
                    previews.append(
                        ExecutionPreview(symbol=symbol, action=action, status="BLOCKED", reason="outside_entry_window")
                    )
                    continue
                if symbol in snapshot.positions:
                    previews.append(
                        ExecutionPreview(symbol=symbol, action=action, status="BLOCKED", reason="already_holding")
                    )
                    continue
                if open_positions >= self.config.max_open_positions:
                    previews.append(
                        ExecutionPreview(
                            symbol=symbol,
                            action=action,
                            status="BLOCKED",
                            reason="max_open_positions_reached",
                        )
                    )
                    continue
                try:
                    live_price, live_price_age = self.get_latest_price_with_age(symbol)
                except Exception as exc:
                    previews.append(
                        ExecutionPreview(
                            symbol=symbol,
                            action=action,
                            status="ERROR",
                            reason="live_price_unavailable",
                            detail=str(exc),
                        )
                    )
                    continue
                signal_price = item.price or 0.0
                deviation_bps = abs((live_price / signal_price) - 1.0) * 10_000 if signal_price > 0 else None
                if live_price_age > self.config.max_live_price_age_seconds:
                    previews.append(
                        ExecutionPreview(
                            symbol=symbol,
                            action=action,
                            status="BLOCKED",
                            reason="stale_live_price",
                            live_price=live_price,
                            signal_price=signal_price or None,
                            price_deviation_bps=deviation_bps,
                            live_price_age_s=live_price_age,
                        )
                    )
                    continue
                if self._is_price_collar_breached(signal_price, live_price):
                    previews.append(
                        ExecutionPreview(
                            symbol=symbol,
                            action=action,
                            status="BLOCKED",
                            reason="price_collar_breached",
                            live_price=live_price,
                            signal_price=signal_price or None,
                            price_deviation_bps=deviation_bps,
                            live_price_age_s=live_price_age,
                        )
                    )
                    continue
                if self._is_symbol_exposure_exceeded(symbol, live_price, snapshot.positions):
                    previews.append(
                        ExecutionPreview(
                            symbol=symbol,
                            action=action,
                            status="BLOCKED",
                            reason="symbol_exposure_exceeded",
                            live_price=live_price,
                            signal_price=signal_price or None,
                            price_deviation_bps=deviation_bps,
                            live_price_age_s=live_price_age,
                        )
                    )
                    continue
                trade_budget = min(self.config.max_usd_per_trade, remaining_buying_power)
                if trade_budget < live_price:
                    previews.append(
                        ExecutionPreview(
                            symbol=symbol,
                            action=action,
                            status="BLOCKED",
                            reason="insufficient_buying_power",
                            live_price=live_price,
                            signal_price=signal_price or None,
                            price_deviation_bps=deviation_bps,
                            live_price_age_s=live_price_age,
                        )
                    )
                    continue
                previews.append(
                    ExecutionPreview(
                        symbol=symbol,
                        action=action,
                        status="READY",
                        live_price=live_price,
                        signal_price=signal_price or None,
                        price_deviation_bps=deviation_bps,
                        live_price_age_s=live_price_age,
                    )
                )
                recent_order_count += 1
                open_positions += 1
                estimated_cost = int(self.config.max_usd_per_trade // live_price) * live_price
                remaining_buying_power = max(0.0, remaining_buying_power - estimated_cost)
                continue

            if symbol not in snapshot.positions:
                previews.append(
                    ExecutionPreview(symbol=symbol, action=action, status="BLOCKED", reason="not_holding")
                )
                continue
            try:
                live_price, live_price_age = self.get_latest_price_with_age(symbol)
            except Exception as exc:
                previews.append(
                    ExecutionPreview(
                        symbol=symbol,
                        action=action,
                        status="ERROR",
                        reason="live_price_unavailable",
                        detail=str(exc),
                    )
                )
                continue
            signal_price = item.price or 0.0
            deviation_bps = abs((live_price / signal_price) - 1.0) * 10_000 if signal_price > 0 else None
            if live_price_age > self.config.max_live_price_age_seconds:
                previews.append(
                    ExecutionPreview(
                        symbol=symbol,
                        action=action,
                        status="BLOCKED",
                        reason="stale_live_price",
                        live_price=live_price,
                        signal_price=signal_price or None,
                        price_deviation_bps=deviation_bps,
                        live_price_age_s=live_price_age,
                    )
                )
                continue
            if self._is_price_collar_breached(signal_price, live_price):
                previews.append(
                    ExecutionPreview(
                        symbol=symbol,
                        action=action,
                        status="BLOCKED",
                        reason="price_collar_breached",
                        live_price=live_price,
                        signal_price=signal_price or None,
                        price_deviation_bps=deviation_bps,
                        live_price_age_s=live_price_age,
                    )
                )
                continue
            previews.append(
                ExecutionPreview(
                    symbol=symbol,
                    action=action,
                    status="READY",
                    live_price=live_price,
                    signal_price=signal_price or None,
                    price_deviation_bps=deviation_bps,
                    live_price_age_s=live_price_age,
                )
            )
            recent_order_count += 1

        return previews

    def flatten_positions(self, positions: dict[str, Position], open_order_symbols: set[str], exit_reason: str = "eod_flatten") -> None:
        for symbol, position in positions.items():
            if symbol in open_order_symbols:
                print(f"Skip flatten {symbol}: existing open order in flight")
                continue
            try:
                qty = int(float(position.qty))
                if qty > 0:
                    self.place_market_sell(symbol, position, exit_reason=exit_reason)
                elif qty < 0:
                    # Short position — buy to cover
                    cover_qty = abs(qty)
                    order = self.broker.submit_market_order(symbol=symbol, qty=cover_qty, side="buy")
                    print(f"Submitted BUY-TO-COVER {symbol} qty={cover_qty} (flatten)")
                    order_id = order.order_id
                    dec_ts = self.get_decision_timestamp().isoformat()
                    if order_id:
                        self._order_submission_ts[order_id] = dec_ts
                        self._order_submission_side[order_id] = "buy"
                        self._order_exit_reason[order_id] = exit_reason
                    self.blog.order_submitted(
                        symbol=symbol,
                        decision_ts=dec_ts,
                        side="buy",
                        qty=cover_qty,
                        live_price=_float_or_default(getattr(position, "current_price", None) or getattr(position, "avg_entry_price", None)),
                        order_id=order_id,
                        signal_bar_close=None,
                    )
                else:
                    print(f"Skip flatten {symbol}: qty is 0")
            except Exception as exc:
                print(f"Flatten {symbol} ERROR: {exc}")

    def record_state(self, snapshot: BotSnapshot, orders_limit: int = 50) -> list[OrderSnapshot]:
        orders = self.get_recent_orders(limit=orders_limit)
        self.storage.save_snapshot(
            snapshot,
            orders,
            session_id=self.session_id,
            symbol_fingerprint=self._active_symbol_fingerprint,
        )
        self._log_fills_from_orders(orders)
        return orders

    def _log_fills_from_orders(self, orders: list[OrderSnapshot]) -> None:
        """
        Emit fill log events for any new fill states not yet logged.

        Cache key is (order_id, filled_qty) so that a partial fill and the
        subsequent final fill are each logged exactly once.
        """
        for order in orders:
            status = _normalize_enum_text(order.status)
            side = _normalize_enum_text(order.side)
            if status not in ("filled", "partially_filled"):
                continue
            if order.filled_avg_price is None or order.filled_qty is None:
                continue
            fill_time = _parse_iso_timestamp(order.filled_at) or _parse_iso_timestamp(order.submitted_at)
            if fill_time is not None and fill_time < self._session_started_at:
                continue

            fill_qty = float(order.filled_qty)
            cache_key = (order.order_id, round(fill_qty, 6))
            if cache_key in self._logged_fills:
                continue
            storage = getattr(self, "storage", None)
            if order.order_id and storage is not None:
                claimed = storage.claim_order_fill(
                    order.order_id,
                    fill_qty,
                    datetime.now(timezone.utc).isoformat(),
                )
                if not claimed:
                    self._logged_fills.add(cache_key)
                    continue
            self._logged_fills.add(cache_key)

            # Use the decision_ts from submission time, not the current bar.
            dec_ts = self._order_submission_ts.get(
                order.order_id, self.get_decision_timestamp().isoformat()
            )
            signal_price = self._order_signal_price.get(order.order_id)
            fill_price = float(order.filled_avg_price)
            requested_qty = float(order.qty or fill_qty)

            # Partial fill — always log it, even if the order later completes.
            if status == "partially_filled" or fill_qty < requested_qty:
                self.blog.order_partial_fill(
                    symbol=order.symbol,
                    decision_ts=dec_ts,
                    fill_price=fill_price,
                    filled_qty=fill_qty,
                    requested_qty=requested_qty,
                    order_id=order.order_id,
                )

            # Final fill — log order.filled and update position lifecycle.
            if status == "filled":
                self.blog.order_filled(
                    symbol=order.symbol,
                    decision_ts=dec_ts,
                    side=side,
                    fill_price=fill_price,
                    fill_qty=fill_qty,
                    order_id=order.order_id,
                    submitted_at=order.submitted_at or dec_ts,
                    filled_at=order.filled_at or order.submitted_at or dec_ts,
                    signal_bar_close=signal_price,
                )

                if side == "buy":
                    existing_qty = _optional_float(self._position_qty.get(order.symbol)) or 0.0
                    if existing_qty < 0:
                        remaining_qty = existing_qty + fill_qty
                        if remaining_qty < 0:
                            self._position_qty[order.symbol] = remaining_qty
                        else:
                            self._position_entry_price.pop(order.symbol, None)
                            self._position_entry_ts.pop(order.symbol, None)
                            self._position_qty.pop(order.symbol, None)
                            self._position_entry_branch.pop(order.symbol, None)
                        self._order_submission_ts.pop(order.order_id, None)
                        self._order_submission_side.pop(order.order_id, None)
                        self._order_exit_reason.pop(order.order_id, None)
                        self._order_signal_price.pop(order.order_id, None)
                        self._order_entry_branch.pop(order.order_id, None)
                        continue
                    # Record entry state so we can compute PnL when the position closes.
                    # Only log position.opened on the first fill — partial fills must not
                    # inflate the count.
                    _is_first_fill = order.symbol not in self._position_entry_price
                    self._position_entry_price[order.symbol] = fill_price
                    self._position_entry_ts[order.symbol] = dec_ts
                    self._position_qty[order.symbol] = fill_qty
                    entry_branch = self._order_entry_branch.get(order.order_id)
                    if entry_branch:
                        self._position_entry_branch[order.symbol] = entry_branch
                    if _is_first_fill:
                        self.blog.position_opened(
                            symbol=order.symbol,
                            decision_ts=dec_ts,
                            entry_price=fill_price,
                            qty=fill_qty,
                            strategy_mode=self.config.strategy_mode,
                            entry_branch=entry_branch,
                        )

                elif side == "sell":
                    entry_price = self._position_entry_price.pop(order.symbol, fill_price)
                    self._position_entry_ts.pop(order.symbol, None)
                    self._position_qty.pop(order.symbol, None)
                    self._position_entry_branch.pop(order.symbol, None)
                    holding_minutes = (
                        self._position_holding_minutes(order.symbol, datetime.now(timezone.utc)) or 0.0
                    )
                    holding_bars = max(1, round(holding_minutes / self.config.bar_timeframe_minutes))
                    exit_reason = self._order_exit_reason.get(order.order_id, "sell_signal")
                    self.blog.position_closed(
                        symbol=order.symbol,
                        decision_ts=dec_ts,
                        entry_price=entry_price,
                        exit_price=fill_price,
                        qty=fill_qty,
                        holding_bars=holding_bars,
                        holding_minutes=holding_minutes,
                        exit_reason=exit_reason,
                    )
                self._order_submission_ts.pop(order.order_id, None)
                self._order_submission_side.pop(order.order_id, None)
                self._order_exit_reason.pop(order.order_id, None)
                self._order_signal_price.pop(order.order_id, None)
                self._order_entry_branch.pop(order.order_id, None)

    def capture_state(
        self,
        orders_limit: int = 20,
        evaluate_signals: bool = True,
    ) -> tuple[BotSnapshot, list[OrderSnapshot]]:
        snapshot = self.build_snapshot(evaluate_signals=evaluate_signals)
        orders = self.record_state(snapshot, orders_limit=orders_limit)
        return snapshot, orders

    def place_market_buy(
        self,
        symbol: str,
        buying_power_available: float | None = None,
        price: float | None = None,
        decision_ts: str | None = None,
        signal_price: float | None = None,
        entry_branch: str | None = None,
    ) -> OrderSnapshot | None:
        execution_price = price if price is not None else self.get_latest_price(symbol)
        available_buying_power = buying_power_available
        if available_buying_power is None:
            account = self.get_account()
            available_buying_power = float(account.buying_power)

        trade_budget = min(self.config.max_usd_per_trade, available_buying_power)
        if trade_budget < execution_price:
            print(
                f"Skip {symbol}: available buying power ${available_buying_power:.2f} "
                f"cannot fund one share at {execution_price:.2f}"
            )
            return None

        qty = int(trade_budget // execution_price)
        if qty <= 0:
            print(
                f"Skip {symbol}: price {execution_price:.2f} is above ${self.config.max_usd_per_trade:.2f}"
            )
            return None

        order = self.broker.submit_market_order(symbol=symbol, qty=qty, side="buy")
        print(f"Submitted BUY {symbol} qty={qty} approx=${qty * execution_price:.2f}")
        order_id = order.order_id
        dec_ts = decision_ts or self.get_decision_timestamp().isoformat()
        if order_id:
            self._order_submission_ts[order_id] = dec_ts
            self._order_submission_side[order_id] = "buy"
            if signal_price is not None:
                self._order_signal_price[order_id] = signal_price
            if entry_branch:
                self._order_entry_branch[order_id] = entry_branch
        self.blog.order_submitted(
            symbol=symbol,
            decision_ts=dec_ts,
            side="buy",
            qty=qty,
            live_price=execution_price,
            order_id=order_id,
            signal_bar_close=signal_price,
        )
        return order

    def place_market_sell(
        self,
        symbol: str,
        position: Position,
        exit_reason: str = "sell_signal",
        live_price: float | None = None,
        decision_ts: str | None = None,
        signal_price: float | None = None,
    ) -> OrderSnapshot | None:
        qty = int(float(position.qty))
        if qty <= 0:
            print(f"Skip SELL {symbol}: non-positive quantity {position.qty}")
            return None

        holding_minutes = self._position_holding_minutes(symbol, datetime.now(timezone.utc))

        order = self.broker.submit_market_order(symbol=symbol, qty=qty, side="sell")
        holding_text = f" holding_minutes={holding_minutes:.1f}" if holding_minutes is not None else ""
        print(f"Submitted SELL {symbol} qty={qty}{holding_text}")
        order_id = order.order_id
        dec_ts = decision_ts or self.get_decision_timestamp().isoformat()
        if order_id:
            self._order_submission_ts[order_id] = dec_ts
            self._order_submission_side[order_id] = "sell"
            self._order_exit_reason[order_id] = exit_reason
            if signal_price is not None:
                self._order_signal_price[order_id] = signal_price
        self.blog.order_submitted(
            symbol=symbol,
            decision_ts=dec_ts,
            side="sell",
            qty=qty,
            live_price=_float_or_default(live_price if live_price is not None else (getattr(position, "current_price", None) or getattr(position, "avg_entry_price", None))),
            order_id=order_id,
            signal_bar_close=signal_price,
        )
        return order

    def _seconds_until_next_bar(self) -> float:
        now = datetime.now(timezone.utc)
        bar_seconds = int(self._bar_interval().total_seconds())
        last_bar_unix = int(now.timestamp()) // bar_seconds * bar_seconds
        next_bar_unix = last_bar_unix + bar_seconds
        return max(0.0, next_bar_unix - now.timestamp())

    def _et_now(self) -> datetime:
        return datetime.now(_ET)

    def _is_in_entry_window(self, now_et: datetime | None = None) -> bool:
        timestamp = now_et or self._et_now()
        return is_entry_window_open(pd.Timestamp(timestamp), self.config.time_window_mode)

    def _active_symbol_event_blackout(self, symbol: str, timestamp_utc: datetime) -> SymbolEventBlackout | None:
        for blackout in self.config.symbol_event_blackouts:
            if blackout.symbol != symbol:
                continue
            start_ts = _parse_iso_timestamp(blackout.start_utc)
            end_ts = _parse_iso_timestamp(blackout.end_utc)
            if start_ts is None or end_ts is None:
                continue
            if start_ts <= timestamp_utc <= end_ts:
                return blackout
        return None

    def _is_past_flatten_deadline(self, now_et: datetime | None = None) -> bool:
        t = (now_et or self._et_now()).time()
        return t >= _SESSION_FLATTEN_AT

    def run_once(self, execute_orders: bool = True, force_process: bool = False) -> BotSnapshot:
        print("\n=== BOT TICK ===")
        self._bars_cache.clear()
        self._hourly_regime_cache.clear()
        now_et = self._et_now()
        decision_timestamp = self.get_decision_timestamp()
        should_process = True if force_process else self._should_process_decision_timestamp(decision_timestamp)
        buy_signals = 0
        sell_signals = 0
        hold_signals = 0
        error_signals = 0
        orders_submitted = 0
        _hold_reasons: dict[str, int] = {}

        def _set_cycle_report(processed_bar: bool, skip_reason: str) -> None:
            self._last_run_cycle_report = RunCycleReport(
                decision_timestamp=decision_timestamp.isoformat(),
                execute_orders=execute_orders,
                processed_bar=processed_bar,
                skip_reason=skip_reason,
                buy_signals=buy_signals,
                sell_signals=sell_signals,
                hold_signals=hold_signals,
                error_signals=error_signals,
                orders_submitted=orders_submitted,
            )
            self.blog.cycle_summary(
                decision_ts=decision_timestamp.isoformat(),
                execute_orders=execute_orders,
                processed_bar=processed_bar,
                skip_reason=skip_reason,
                buy_signals=buy_signals,
                sell_signals=sell_signals,
                hold_signals=hold_signals,
                error_signals=error_signals,
                orders_submitted=orders_submitted,
            )
            _hold_detail = (
                " [" + " ".join(f"{k}={v}" for k, v in sorted(_hold_reasons.items())) + "]"
                if _hold_reasons else ""
            )
            print(
                f"[CYCLE RESULT] processed_bar={processed_bar} reason={skip_reason} "
                f"buy={buy_signals} sell={sell_signals} hold={hold_signals}{_hold_detail} "
                f"errors={error_signals} orders_submitted={orders_submitted}"
            )

        if execute_orders and should_process and not self._claim_global_decision_execution(decision_timestamp):
            snapshot = self.build_snapshot(
                decision_timestamp=decision_timestamp,
                evaluate_signals=False,
            )
            print("[CYCLE] Skipping globally claimed bar before execution branch")
            _set_cycle_report(processed_bar=False, skip_reason="duplicate_bar")
            self.record_state(snapshot)
            return snapshot

        try:
            if execute_orders and should_process and self._is_regular_hours(decision_timestamp):
                self._validate_startup_market_data(decision_timestamp)
            snapshot = self.build_snapshot(
                decision_timestamp=decision_timestamp,
                evaluate_signals=should_process,
            )
        except StaleMarketDataError as exc:
            print(f"[CYCLE] stale_market_data: {exc}")
            _set_cycle_report(processed_bar=False, skip_reason="stale_market_data")
            raise

        print(f"Connected. Cash: {snapshot.cash} Buying power: {snapshot.buying_power}")
        print(f"Daily PnL: {snapshot.daily_pnl:.2f}")
        print(f"Strategy mode: {self.config.strategy_mode}")
        print(
            f"[CYCLE] decision_ts={decision_timestamp.isoformat()} "
            f"execute_orders={execute_orders} should_process={should_process} "
            f"force_process={force_process}"
        )
        buy_signals = sum(1 for item in snapshot.symbols if item.action == "BUY")
        sell_signals = sum(1 for item in snapshot.symbols if item.action == "SELL")
        hold_signals = sum(1 for item in snapshot.symbols if item.action == "HOLD")
        error_signals = sum(1 for item in snapshot.symbols if item.action == "ERROR")
        for _item in snapshot.symbols:
            if _item.action == "HOLD" and _item.hold_reason:
                _hold_reasons[_item.hold_reason] = _hold_reasons.get(_item.hold_reason, 0) + 1

        if not should_process:
            print("[CYCLE] Skipping duplicate bar before execution branch")
            _set_cycle_report(processed_bar=False, skip_reason="duplicate_bar")
            self.record_state(snapshot)
            return snapshot

        resync_result = self.get_startup_resync_result()
        if execute_orders and not resync_result.gate_allows_exits:
            self._log_skip(
                "SKIP_RESYNC_GATE",
                f"status={resync_result.status.value} entries={resync_result.gate_allows_entries} exits={resync_result.gate_allows_exits}",
            )
            _set_cycle_report(processed_bar=True, skip_reason=resync_result.status.value.lower())
            self.record_state(snapshot)
            return snapshot

        if execute_orders:
            try:
                market_open = self._is_market_open()
            except Exception as exc:
                self._log_skip("SKIP_MARKET_CLOSED", str(exc))
                _set_cycle_report(processed_bar=True, skip_reason="market_clock_error")
                self.record_state(snapshot)
                return snapshot
            if not market_open:
                self._log_skip("SKIP_MARKET_CLOSED", "Alpaca market clock reports closed")
                _set_cycle_report(processed_bar=True, skip_reason="market_closed")
                self.record_state(snapshot)
                return snapshot
            if not self._is_regular_hours(decision_timestamp):
                self._log_skip(
                    "SKIP_OUTSIDE_REGULAR_HOURS",
                    f"decision_timestamp={decision_timestamp.astimezone(_ET).strftime('%H:%M:%S')} ET",
                )
                _set_cycle_report(processed_bar=True, skip_reason="outside_regular_hours")
                self.record_state(snapshot)
                return snapshot

        if execute_orders and self._is_past_flatten_deadline(now_et):
            self._log_skip(
                "SKIP_EOD_FLATTEN_WINDOW",
                f"{now_et.strftime('%H:%M:%S')} ET >= 15:55, closing all positions",
            )
            open_orders = self.get_open_orders()
            open_order_symbols = {
                str(getattr(order, "symbol", ""))
                for order in open_orders
                if getattr(order, "symbol", None)
            }
            self.flatten_positions(snapshot.positions, open_order_symbols)
            snapshot = self.build_snapshot(decision_timestamp=decision_timestamp, evaluate_signals=False)
            _set_cycle_report(processed_bar=True, skip_reason="eod_flatten_window")
            self.record_state(snapshot)
            return snapshot

        _ks_pct = (
            round(-snapshot.daily_pnl / self.config.max_daily_loss_usd * 100, 1)
            if self.config.max_daily_loss_usd > 0 else 0.0
        )
        for _threshold in (50, 75):
            if _ks_pct >= _threshold and _threshold not in self._kill_switch_warned_pcts:
                self._kill_switch_warned_pcts.add(_threshold)
                self.blog.kill_switch(
                    daily_pnl=snapshot.daily_pnl,
                    daily_limit=self.config.max_daily_loss_usd,
                    trigger=False,
                    reason=f"warning_{_threshold}pct",
                )
        if snapshot.kill_switch_triggered:
            print("Kill switch triggered.")
            self.blog.kill_switch(
                daily_pnl=snapshot.daily_pnl,
                daily_limit=self.config.max_daily_loss_usd,
                trigger=True,
            )
            if execute_orders:
                open_orders = self.get_open_orders()
                open_order_symbols = {
                    str(getattr(order, "symbol", ""))
                    for order in open_orders
                    if getattr(order, "symbol", None)
                }
                self.flatten_positions(snapshot.positions, open_order_symbols, exit_reason="kill_switch")
                snapshot = self.build_snapshot(decision_timestamp=decision_timestamp, evaluate_signals=False)
            _set_cycle_report(processed_bar=True, skip_reason="kill_switch_active")
            self.record_state(snapshot)
            return snapshot

        if not execute_orders:
            print("[CYCLE] Preview-only mode; execution branch will not submit orders")
            for item in snapshot.symbols:
                suffix = (
                    f" ml_up={item.ml_probability_up:.3f}"
                    if item.ml_probability_up is not None and item.ml_model_name != "dummy"
                    else ""
                )
                error_suffix = f" ERROR: {item.error}" if item.error else ""
                print(f"{item.symbol} -> {item.action}{suffix}{error_suffix}")
            _set_cycle_report(processed_bar=True, skip_reason="preview_only")
            self.record_state(snapshot)
            return snapshot

        print(
            f"[CYCLE] Entering live execution branch buy_signals={buy_signals} "
            f"sell_signals={sell_signals} open_positions={len(snapshot.positions)}"
        )
        in_entry_window = self._is_in_entry_window(now_et)
        if not in_entry_window:
            print(f"Outside entry window ({now_et.strftime('%H:%M:%S')} ET): new entries suppressed, exits still active")

        positions = snapshot.positions.copy()
        open_positions = sum(1 for position in positions.values() if _has_open_position(position))
        remaining_buying_power = snapshot.buying_power
        open_orders = self.get_open_orders()
        open_order_symbols = {
            str(getattr(order, "symbol", ""))
            for order in open_orders
            if getattr(order, "symbol", None)
        }
        recent_order_count = self._count_recent_orders(window_seconds=60)

        for item in snapshot.symbols:
            if self._is_past_flatten_deadline():
                self._log_skip(
                    "SKIP_EOD_FLATTEN_WINDOW",
                    f"mid-cycle at {self._et_now().strftime('%H:%M:%S')} ET",
                )
                self.flatten_positions(snapshot.positions, open_order_symbols)
                break

            symbol = item.symbol
            try:
                action = item.action
                if item.error:
                    print(f"{symbol} ERROR: {item.error}")
                    continue

                ml_text = (
                    f" ml_up={item.ml_probability_up:.3f} conf={item.ml_confidence:.3f}"
                    if item.ml_probability_up is not None
                    and item.ml_confidence is not None
                    and item.ml_model_name != "dummy"
                    else ""
                )
                holding_text = (
                    f" holding_minutes={item.holding_minutes:.1f}"
                    if item.holding_minutes is not None
                    else ""
                )
                print(f"{symbol} -> {action}{ml_text}{holding_text}")

                allowed_by_resync, resync_reason = self._resync_gate_for_action(
                    action=action,
                    symbol=symbol,
                    positions=positions,
                )
                if not allowed_by_resync:
                    self._log_risk_check(
                        symbol=symbol,
                        decision_ts=decision_timestamp.isoformat(),
                        action=action,
                        allowed=False,
                        block_reason=resync_reason,
                        snapshot=snapshot,
                        open_positions=open_positions,
                        signal_price=item.price or None,
                        detail=f"startup resync gate blocked action under status={resync_result.status.value}",
                        in_entry_window=in_entry_window,
                        remaining_buying_power=remaining_buying_power,
                        recent_order_count=recent_order_count,
                    )
                    continue

                if symbol in open_order_symbols:
                    print(f"Skip {symbol}: existing open order in flight")
                    self._log_risk_check(
                        symbol=symbol,
                        decision_ts=decision_timestamp.isoformat(),
                        action=action,
                        allowed=False,
                        block_reason="open_order_in_flight",
                        snapshot=snapshot,
                        open_positions=open_positions,
                        signal_price=item.price or None,
                        detail="symbol already has an open order working at the broker",
                        in_entry_window=in_entry_window,
                        remaining_buying_power=remaining_buying_power,
                        recent_order_count=recent_order_count,
                    )
                    continue

                if recent_order_count >= self.config.max_orders_per_minute:
                    print("Skip new orders: max order rate reached")
                    self._log_risk_check(
                        symbol=symbol,
                        decision_ts=decision_timestamp.isoformat(),
                        action=action,
                        allowed=False,
                        block_reason="max_orders_per_minute",
                        snapshot=snapshot,
                        open_positions=open_positions,
                        signal_price=item.price or None,
                        detail="rate limiter reached before this signal could be submitted",
                        in_entry_window=in_entry_window,
                        remaining_buying_power=remaining_buying_power,
                        recent_order_count=recent_order_count,
                    )
                    break

                if action == "BUY":
                    _dec_ts = decision_timestamp.isoformat() if decision_timestamp else ""
                    active_blackout = self._active_symbol_event_blackout(symbol, decision_timestamp)
                    if active_blackout is not None:
                        print(f"Skip {symbol} BUY: event blackout active ({active_blackout.reason})")
                        self._log_risk_check(
                            symbol=symbol,
                            decision_ts=_dec_ts,
                            action="BUY",
                            allowed=False,
                            block_reason="event_blackout",
                            snapshot=snapshot,
                            open_positions=open_positions,
                            signal_price=item.price or None,
                            detail=(
                                f"blackout active from {active_blackout.start_utc} to "
                                f"{active_blackout.end_utc} ({active_blackout.reason})"
                            ),
                            in_entry_window=in_entry_window,
                            remaining_buying_power=remaining_buying_power,
                            recent_order_count=recent_order_count,
                        )
                        continue
                    if not in_entry_window:
                        print(f"Skip {symbol} BUY: outside trading window ({now_et.strftime('%H:%M:%S')} ET)")
                        self._log_risk_check(
                            symbol=symbol,
                            decision_ts=_dec_ts,
                            action="BUY",
                            allowed=False,
                            block_reason="outside_entry_window",
                            snapshot=snapshot,
                            open_positions=open_positions,
                            signal_price=item.price or None,
                            detail=f"entry window closed at {now_et.strftime('%H:%M:%S')} ET",
                            in_entry_window=in_entry_window,
                            remaining_buying_power=remaining_buying_power,
                            recent_order_count=recent_order_count,
                        )
                        continue
                    if symbol in positions:
                        print(f"Already holding {symbol}")
                        self._log_risk_check(
                            symbol=symbol,
                            decision_ts=_dec_ts,
                            action="BUY",
                            allowed=False,
                            block_reason="already_holding",
                            snapshot=snapshot,
                            open_positions=open_positions,
                            signal_price=item.price or None,
                            detail="buy signal suppressed because the symbol already has a long position",
                            in_entry_window=in_entry_window,
                            remaining_buying_power=remaining_buying_power,
                            recent_order_count=recent_order_count,
                        )
                        continue
                    if open_positions >= self.config.max_open_positions:
                        print("Max positions reached")
                        self._log_risk_check(
                            symbol=symbol,
                            decision_ts=_dec_ts,
                            action="BUY",
                            allowed=False,
                            block_reason="max_open_positions_reached",
                            snapshot=snapshot,
                            open_positions=open_positions,
                            signal_price=item.price or None,
                            detail=(
                                f"portfolio already has {open_positions} open positions "
                                f"with limit {self.config.max_open_positions}"
                            ),
                            in_entry_window=in_entry_window,
                            remaining_buying_power=remaining_buying_power,
                            recent_order_count=recent_order_count,
                        )
                        continue
                    live_price, live_price_age = self.get_latest_price_with_age(symbol)
                    _signal_price = item.price or 0.0
                    _dev_bps = abs((live_price / _signal_price) - 1.0) * 10_000 if _signal_price > 0 else None
                    trade_budget = min(self.config.max_usd_per_trade, remaining_buying_power)
                    if live_price_age > self.config.max_live_price_age_seconds:
                        print(f"Skip {symbol}: stale live price age {live_price_age:.1f}s")
                        self._log_risk_check(
                            symbol=symbol,
                            decision_ts=_dec_ts,
                            action="BUY",
                            allowed=False,
                            block_reason="stale_live_price",
                            snapshot=snapshot,
                            open_positions=open_positions,
                            live_price=live_price,
                            signal_price=_signal_price or None,
                            price_deviation_bps=_dev_bps,
                            live_price_age_s=round(live_price_age, 1),
                            detail=(
                                f"latest trade quote age {live_price_age:.1f}s exceeds "
                                f"{self.config.max_live_price_age_seconds}s"
                            ),
                            in_entry_window=in_entry_window,
                            remaining_buying_power=remaining_buying_power,
                            trade_budget=trade_budget,
                            recent_order_count=recent_order_count,
                        )
                        continue
                    if self._is_price_collar_breached(_signal_price, live_price):
                        print(
                            f"Skip {symbol}: live price {live_price:.2f} breaches collar vs decision price {_signal_price:.2f}"
                        )
                        self._log_risk_check(
                            symbol=symbol,
                            decision_ts=_dec_ts,
                            action="BUY",
                            allowed=False,
                            block_reason="price_collar_breached",
                            snapshot=snapshot,
                            open_positions=open_positions,
                            live_price=live_price,
                            signal_price=_signal_price,
                            price_deviation_bps=_dev_bps,
                            live_price_age_s=live_price_age,
                            detail="live price moved outside the allowed collar versus the signal bar close",
                            in_entry_window=in_entry_window,
                            remaining_buying_power=remaining_buying_power,
                            trade_budget=trade_budget,
                            recent_order_count=recent_order_count,
                        )
                        continue
                    if self._is_symbol_exposure_exceeded(symbol, live_price, positions):
                        print(f"Skip {symbol}: max symbol exposure would be exceeded")
                        self._log_risk_check(
                            symbol=symbol,
                            decision_ts=_dec_ts,
                            action="BUY",
                            allowed=False,
                            block_reason="symbol_exposure_exceeded",
                            snapshot=snapshot,
                            open_positions=open_positions,
                            live_price=live_price,
                            signal_price=_signal_price or None,
                            price_deviation_bps=_dev_bps,
                            live_price_age_s=live_price_age,
                            detail="buy would exceed the configured single-symbol exposure cap",
                            in_entry_window=in_entry_window,
                            remaining_buying_power=remaining_buying_power,
                            trade_budget=trade_budget,
                            recent_order_count=recent_order_count,
                        )
                        continue
                    if trade_budget < live_price:
                        print(
                            f"Skip {symbol}: trade budget ${trade_budget:.2f} "
                            f"cannot fund one share at {live_price:.2f}"
                        )
                        self._log_risk_check(
                            symbol=symbol,
                            decision_ts=_dec_ts,
                            action="BUY",
                            allowed=False,
                            block_reason="insufficient_buying_power",
                            snapshot=snapshot,
                            open_positions=open_positions,
                            live_price=live_price,
                            signal_price=_signal_price,
                            price_deviation_bps=_dev_bps,
                            live_price_age_s=live_price_age,
                            detail="remaining buying power cannot fund the minimum one-share order",
                            in_entry_window=in_entry_window,
                            remaining_buying_power=remaining_buying_power,
                            trade_budget=trade_budget,
                            recent_order_count=recent_order_count,
                        )
                        continue
                    # All checks passed
                    self._log_risk_check(
                        symbol=symbol,
                        decision_ts=_dec_ts,
                        action="BUY",
                        allowed=True,
                        block_reason=None,
                        snapshot=snapshot,
                        open_positions=open_positions,
                        live_price=live_price,
                        signal_price=_signal_price,
                        price_deviation_bps=_dev_bps,
                        live_price_age_s=live_price_age,
                        detail="all buy-side live risk checks passed",
                        in_entry_window=in_entry_window,
                        remaining_buying_power=remaining_buying_power,
                        trade_budget=trade_budget,
                        recent_order_count=recent_order_count,
                    )
                    order = self.place_market_buy(
                        symbol,
                        buying_power_available=remaining_buying_power,
                        price=live_price,
                        decision_ts=_dec_ts,
                        signal_price=_signal_price,
                        entry_branch=item.hybrid_entry_branch,
                    )
                    if order is not None:
                        orders_submitted += 1
                        open_positions += 1
                        recent_order_count += 1
                        estimated_cost = int(self.config.max_usd_per_trade // live_price) * live_price
                        remaining_buying_power = max(0.0, remaining_buying_power - estimated_cost)

                elif action == "SELL" and symbol in positions:
                    _dec_ts = decision_timestamp.isoformat() if decision_timestamp else ""
                    live_price, live_price_age = self.get_latest_price_with_age(symbol)
                    _signal_price = item.price or 0.0
                    _dev_bps = abs((live_price / _signal_price) - 1.0) * 10_000 if _signal_price > 0 else None
                    if live_price_age > self.config.max_live_price_age_seconds:
                        print(f"Skip {symbol}: stale live price age {live_price_age:.1f}s")
                        self._log_risk_check(
                            symbol=symbol,
                            decision_ts=_dec_ts,
                            action="SELL",
                            allowed=False,
                            block_reason="stale_live_price",
                            snapshot=snapshot,
                            open_positions=open_positions,
                            live_price=live_price,
                            signal_price=_signal_price or None,
                            price_deviation_bps=_dev_bps,
                            live_price_age_s=round(live_price_age, 1),
                            detail=(
                                f"latest trade quote age {live_price_age:.1f}s exceeds "
                                f"{self.config.max_live_price_age_seconds}s"
                            ),
                            in_entry_window=in_entry_window,
                            remaining_buying_power=remaining_buying_power,
                            recent_order_count=recent_order_count,
                        )
                        continue
                    if self._is_price_collar_breached(_signal_price, live_price):
                        print(
                            f"Skip {symbol}: live price {live_price:.2f} breaches collar vs decision price {item.price:.2f}"
                        )
                        self._log_risk_check(
                            symbol=symbol,
                            decision_ts=_dec_ts,
                            action="SELL",
                            allowed=False,
                            block_reason="price_collar_breached",
                            snapshot=snapshot,
                            open_positions=open_positions,
                            live_price=live_price,
                            signal_price=_signal_price,
                            price_deviation_bps=_dev_bps,
                            live_price_age_s=live_price_age,
                            detail="live price moved outside the allowed collar versus the signal bar close",
                            in_entry_window=in_entry_window,
                            remaining_buying_power=remaining_buying_power,
                            recent_order_count=recent_order_count,
                        )
                        continue
                    self._log_risk_check(
                        symbol=symbol,
                        decision_ts=_dec_ts,
                        action="SELL",
                        allowed=True,
                        block_reason=None,
                        snapshot=snapshot,
                        open_positions=open_positions,
                        live_price=live_price,
                        signal_price=_signal_price,
                        price_deviation_bps=_dev_bps,
                        live_price_age_s=live_price_age,
                        detail="all sell-side live risk checks passed",
                        in_entry_window=in_entry_window,
                        remaining_buying_power=remaining_buying_power,
                        recent_order_count=recent_order_count,
                    )
                    order = self.place_market_sell(
                        symbol,
                        positions[symbol],
                        live_price=live_price,
                        decision_ts=_dec_ts,
                        signal_price=_signal_price,
                    )
                    if order is not None:
                        orders_submitted += 1
                        open_positions = max(0, open_positions - 1)
                        recent_order_count += 1
                elif action == "SELL":
                    print(f"Skip {symbol} SELL: not currently holding")
                    self._log_risk_check(
                        symbol=symbol,
                        decision_ts=decision_timestamp.isoformat(),
                        action="SELL",
                        allowed=False,
                        block_reason="not_holding",
                        snapshot=snapshot,
                        open_positions=open_positions,
                        signal_price=item.price or None,
                        detail="sell signal ignored because there is no live position to exit",
                        in_entry_window=in_entry_window,
                        remaining_buying_power=remaining_buying_power,
                        recent_order_count=recent_order_count,
                    )

            except Exception as exc:
                import traceback
                print(f"{symbol} ERROR: {exc}")
                print(f"[DEBUG TRACEBACK] {traceback.format_exc()}")
                try:
                    self.blog.execution_error(
                        symbol=symbol,
                        decision_ts=decision_timestamp.isoformat(),
                        action=action if "action" in locals() else "UNKNOWN",
                        error=str(exc),
                    )
                except Exception:
                    pass

        final_reason = "execution_completed"
        if orders_submitted == 0:
            if buy_signals > 0 or sell_signals > 0:
                print("[CYCLE] Execution branch completed with signals present but zero submitted orders")
                final_reason = "signals_blocked_or_skipped"
            else:
                print("[CYCLE] Execution branch completed with no actionable BUY/SELL signals")
                final_reason = "no_actionable_signals"
        _set_cycle_report(processed_bar=True, skip_reason=final_reason)
        # Persist a post-execution snapshot without re-evaluating signals.
        # Re-running signal evaluation here duplicates signal logs for the same bar.
        snapshot = self.build_snapshot(decision_timestamp=decision_timestamp, evaluate_signals=False)
        self.record_state(snapshot)
        return snapshot


AlpacaTradingBot = TradeOSBot


def main(config: BotConfig | None = None, session_id: str | None = None) -> None:
    load_dotenv(Path.cwd() / ".env")
    if config is None:
        config = load_config()
    bot = TradeOSBot(config, session_id=session_id)
    bot.perform_startup_resync()
    execute_orders = os.getenv("EXECUTE_ORDERS", "true").lower() != "false"
    try:
        bot.perform_startup_preflight(execute_orders=execute_orders)
    except Exception as exc:
        bot.blog.lifecycle(
            "startup.failed",
            reason="startup_preflight_error",
            execute_orders=execute_orders,
            error=str(exc),
        )
        raise
    bar_interval_seconds = config.bar_timeframe_minutes * 60
    shutdown_event = threading.Event()
    exit_reason = "unknown"
    bot.blog.lifecycle(
        "started",
        execute_orders=execute_orders,
        strategy_mode=config.strategy_mode,
        timeframe_minutes=config.bar_timeframe_minutes,
        symbol_count=len(config.symbols),
        data_parity=bot.data_parity_summary(),
    )

    def _eod_flatten_worker() -> None:
        now_et = bot._et_now()
        target_et = now_et.replace(hour=15, minute=55, second=0, microsecond=0)
        delay = (target_et - now_et).total_seconds()
        if delay <= 0:
            print("EOD flatten thread: already past 15:55 ET, signalling shutdown")
            shutdown_event.set()
            return
        print(f"EOD flatten thread: scheduled in {delay:.0f}s at 15:55 ET")
        # Wait until 15:55, but wake up early if the main loop shuts down first
        shutdown_event.wait(timeout=delay)
        if shutdown_event.is_set():
            # Kill switch or other early exit already handled the flatten
            return
        print(f"EOD flatten thread firing at {bot._et_now().strftime('%H:%M:%S')} ET")
        if execute_orders:
            try:
                positions = bot.get_positions_by_symbol()
                open_orders = bot.get_open_orders()
                open_order_symbols = {
                    str(getattr(order, "symbol", ""))
                    for order in open_orders
                    if getattr(order, "symbol", None)
                }
                bot.flatten_positions(positions, open_order_symbols)
            except Exception as exc:
                print(f"EOD flatten thread ERROR: {exc}")
        else:
            print("EOD flatten thread: execute_orders=False, skipping flatten")
        shutdown_event.set()

    # Keep this thread even though run_once() also enforces the flatten window.
    # The duplication is intentional: run_once() protects the decision path,
    # while the thread is a wall-clock fail-safe if the main loop drifts.
    flatten_thread = threading.Thread(target=_eod_flatten_worker, name="eod-flatten", daemon=True)
    flatten_thread.start()

    try:
        wait_seconds = bot._seconds_until_next_bar()
        if wait_seconds > 1:
            next_bar_utc = datetime.now(timezone.utc) + timedelta(seconds=wait_seconds)
            print(f"Aligning to next bar boundary: waiting {wait_seconds:.0f}s until {next_bar_utc.strftime('%H:%M:%S')} UTC")
            shutdown_event.wait(timeout=wait_seconds)
            if shutdown_event.is_set():
                exit_reason = "startup_wait_interrupted"
                print("Bot loop exited.")
                return

        while not shutdown_event.is_set():
            if bot._is_past_flatten_deadline():
                exit_reason = "past_flatten_deadline"
                print("Main loop: past 15:55 ET, exiting without new tick")
                break

            tick_start = time.time()
            try:
                snapshot = bot.run_once(execute_orders=execute_orders)
                if snapshot.kill_switch_triggered:
                    exit_reason = "kill_switch_triggered"
                    print("Main loop: kill switch triggered, exiting")
                    shutdown_event.set()
                    break
            except Exception as exc:
                print(f"run_once ERROR: {exc}")

            if shutdown_event.is_set():
                if exit_reason == "unknown":
                    exit_reason = "shutdown_event_set"
                break

            elapsed = time.time() - tick_start
            sleep_seconds = max(0.0, bar_interval_seconds - elapsed)
            print(f"Next tick in {sleep_seconds:.0f}s")
            # Wait for the next bar, but wake immediately if flatten thread fires
            shutdown_event.wait(timeout=sleep_seconds)

        if exit_reason == "unknown":
            exit_reason = "loop_completed"
        print("Bot loop exited.")
    finally:
        try:
            final_snapshot = bot.build_snapshot(evaluate_signals=False)
            bot.record_state(final_snapshot)
        except Exception as exc:
            print(f"Final state capture ERROR: {exc}")
        bot.blog.lifecycle(
            "exiting",
            reason=exit_reason,
            execute_orders=execute_orders,
        )


if __name__ == "__main__":
    from alpaca_trading_bot.cli import _run_live
    raise SystemExit(_run_live(preview=False))

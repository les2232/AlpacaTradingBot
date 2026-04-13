import logging
import os
import math
import threading
import time
import json
from dataclasses import dataclass, replace
from datetime import datetime, timedelta, timezone
from datetime import time as dt_time
from pathlib import Path
from typing import Any, cast

from alpaca.data.enums import DataFeed
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.live.stock import StockDataStream
from alpaca.data.requests import StockBarsRequest, StockLatestTradeRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
from alpaca.trading.client import TradingClient
from alpaca.trading.enums import OrderSide, QueryOrderStatus, TimeInForce
from alpaca.trading.models import Position
from alpaca.trading.requests import GetOrdersRequest, MarketOrderRequest
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
from symbol_state import (
    format_symbol_list,
    normalize_symbols,
    symbol_fingerprint,
    symbols_match,
)
from strategy import (
    BREAKOUT_EXIT_CHOICES,
    BREAKOUT_EXIT_TARGET_1X_STOP_LOW,
    MEAN_REVERSION_EXIT_CHOICES,
    MEAN_REVERSION_EXIT_SMA,
    ORB_FILTER_CHOICES,
    ORB_FILTER_NONE,
    STRATEGY_MODE_BREAKOUT,
    STRATEGY_MODE_CHOICES,
    STRATEGY_MODE_HYBRID,
    STRATEGY_MODE_MEAN_REVERSION,
    STRATEGY_MODE_ML,
    STRATEGY_MODE_ORB,
    Strategy,
    StrategyConfig,
    MlSignal,
    THRESHOLD_MODE_CHOICES,
    THRESHOLD_MODE_STATIC_PCT,
    calculate_opening_range_series,
    calculate_atr_pct_values,
    calculate_atr_percentile_series,
    get_capped_breakout_stop_price,
    REGIME_SMA_PERIOD,
    REGIME_TIMEFRAME_MINUTES,
    estimate_atr_percentile_lookback_bars,
    estimate_regime_lookback_bars,
    is_entry_window_open,
    normalize_breakout_exit_style,
    normalize_mean_reversion_exit_style,
    normalize_orb_filter_mode,
    normalize_strategy_mode,
    normalize_threshold_mode,
    normalize_time_window_mode,
    TIME_WINDOW_CHOICES,
    TIME_WINDOW_FULL_DAY,
)

logger = logging.getLogger(__name__)


class StaleMarketDataError(RuntimeError):
    """Raised when the most recent completed bar is too old to trust."""


@dataclass(frozen=True)
class BotConfig:
    symbols: list[str]
    max_usd_per_trade: float
    max_symbol_exposure_usd: float
    max_open_positions: int
    max_daily_loss_usd: float
    sma_bars: int
    bar_timeframe_minutes: int
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
    mean_reversion_stop_pct: float = 0.0
    max_orders_per_minute: int = 6
    max_price_deviation_bps: float = 75.0
    max_data_delay_seconds: int = 300
    max_live_price_age_seconds: int = 60


@dataclass(frozen=True)
class RuntimeConfigDetails:
    config: BotConfig
    runtime_config_path: str | None
    overridden_fields: tuple[str, ...]


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


@dataclass(frozen=True)
class SymbolEvaluation:
    price: float
    sma: float
    ml_signal: MlSignal
    action: str
    latest_bar_close_utc: str
    hold_reason: str | None = None  # populated when action == "HOLD"; e.g. "trend_filter", "atr_filter", "no_signal"


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


@dataclass(frozen=True)
class OrderSnapshot:
    order_id: str
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


_ET = pytz.timezone("America/New_York")
_SESSION_ENTRY_START = dt_time(9, 45)   # no new entries before this
_SESSION_ENTRY_END   = dt_time(15, 45)  # no new entries after this
_SESSION_FLATTEN_AT  = dt_time(15, 55)  # forced EOD flatten deadline
DEFAULT_RUNTIME_CONFIG_PATH = Path("config") / "live_config.json"
_PRICE_STREAM_FEED_BY_NAME = {
    "iex": DataFeed.IEX,
    "sip": DataFeed.SIP,
    "delayed_sip": DataFeed.DELAYED_SIP,
}


def _minutes_to_timeframe(minutes: int) -> TimeFrame:
    if minutes >= 60 and minutes % 60 == 0:
        return TimeFrame(minutes // 60, TimeFrameUnit.Hour)
    return TimeFrame(minutes, TimeFrameUnit.Minute)


def _safe_float(value: str | None, default: float) -> float:
    try:
        return float(value) if value is not None else default
    except (TypeError, ValueError):
        return default


def _normalize_enum_text(value: Any) -> str:
    text = str(value or "").strip()
    if "." in text:
        text = text.rsplit(".", 1)[-1]
    return text.lower()


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


def _load_runtime_config_payload() -> tuple[Path | None, dict[str, Any] | None]:
    runtime_path = Path(os.getenv("BOT_RUNTIME_CONFIG_PATH", str(DEFAULT_RUNTIME_CONFIG_PATH)))
    if not runtime_path.exists():
        return None, None

    try:
        payload = json.loads(runtime_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Runtime config is not valid JSON: {runtime_path}") from exc

    if not isinstance(payload, dict):
        raise RuntimeError(f"Runtime config must contain a JSON object at top level: {runtime_path}")

    runtime = payload.get("runtime", payload)
    if not isinstance(runtime, dict):
        raise RuntimeError(f"Runtime config field 'runtime' must be a JSON object: {runtime_path}")

    return runtime_path, runtime


def _apply_runtime_config(
    base_config: BotConfig,
    runtime_path: Path | None,
    runtime: dict[str, Any] | None,
) -> tuple[BotConfig, tuple[str, ...]]:
    if runtime_path is None or runtime is None:
        return base_config, ()

    overrides: dict[str, Any] = {}
    if "symbols" in runtime:
        overrides["symbols"] = _normalize_runtime_symbols(runtime["symbols"])
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
    if "symbol_strategy_modes" in runtime:
        if not isinstance(runtime["symbol_strategy_modes"], dict):
            raise RuntimeError("Runtime config field 'symbol_strategy_modes' must be an object mapping SYMBOL to mode.")
        overrides["symbol_strategy_modes"] = {
            str(symbol).strip().upper(): normalize_strategy_mode(str(mode))
            for symbol, mode in runtime["symbol_strategy_modes"].items()
            if str(symbol).strip()
        }

    config = replace(base_config, **overrides)
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
        mean_reversion_stop_pct=_safe_float(os.getenv("MEAN_REVERSION_STOP_PCT"), 0.0),
        max_orders_per_minute=int(os.getenv("MAX_ORDERS_PER_MINUTE", "6")),
        max_price_deviation_bps=_safe_float(os.getenv("MAX_PRICE_DEVIATION_BPS"), 75.0),
        max_data_delay_seconds=int(os.getenv("MAX_DATA_DELAY_SECONDS", "300")),
        max_live_price_age_seconds=int(os.getenv("MAX_LIVE_PRICE_AGE_SECONDS", "60")),
    )


def load_config_details() -> RuntimeConfigDetails:
    base_config = _load_base_config()
    runtime_path, runtime = _load_runtime_config_payload()
    config, changed_fields = _apply_runtime_config(base_config, runtime_path, runtime)
    return RuntimeConfigDetails(
        config=config,
        runtime_config_path=str(runtime_path) if runtime_path is not None else None,
        overridden_fields=changed_fields,
    )


def load_config() -> BotConfig:
    # Live execution config is intentionally sourced from environment variables
    # and normalized into BotConfig here. Offline tools use their own CLI args.
    return load_config_details().config


class AlpacaTradingBot:
    def __init__(self, config: BotConfig, session_id: str | None = None) -> None:
        load_dotenv(Path.cwd() / ".env")

        api_key = os.getenv("ALPACA_API_KEY")
        api_secret = os.getenv("ALPACA_API_SECRET")
        if not api_key or not api_secret:
            raise RuntimeError(
                "Missing Alpaca credentials. Set ALPACA_API_KEY and ALPACA_API_SECRET in .env."
            )

        self.config = config
        self.trading = TradingClient(api_key, api_secret, paper=config.paper)
        self.data = StockHistoricalDataClient(api_key, api_secret)
        self._api_key = api_key
        self._api_secret = api_secret
        db_path = Path(os.getenv("BOT_DB_PATH", "bot_history.db"))
        self.storage = BotStorage(db_path)
        self.blog = BotLogger(log_root="logs")
        self.session_id = session_id or f"session-{int(datetime.now(timezone.utc).timestamp())}"
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
        self._latest_prices: dict[str, float] = {}
        self._latest_price_times: dict[str, float] = {}
        self._latest_trade_times: dict[str, float] = {}
        self._price_lock = threading.Lock()
        self._stream_enabled = os.getenv("ENABLE_PRICE_STREAM", "true").lower() != "false"
        self._stream_error: str | None = None
        self.data_stream: StockDataStream | None = None
        self._stream_thread: threading.Thread | None = None
        self._ml_disabled_reason: str | None = None
        self._order_signal_price: dict[str, float] = {}
        self._last_processed_decision_timestamp: datetime | None = None
        self._position_first_seen_utc: dict[str, datetime] = {}
        self._bars_cache: dict[tuple[str, int, int, str], list[Any]] = {}
        self._hourly_regime_cache: dict[tuple[str, str], bool | None] = {}
        self._strategy_cache: dict[str, Strategy] = {}
        self._last_run_cycle_report: RunCycleReport | None = None
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
                mean_reversion_stop_pct=config.mean_reversion_stop_pct,
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
        if offline_to_ml_signal is None and any(
            mode in {STRATEGY_MODE_ML, STRATEGY_MODE_HYBRID}
            for mode in {config.strategy_mode, *self._symbol_strategy_modes.values()}
        ):
            self._disable_ml_trading("ml.predict import failed", _ML_PREDICT_IMPORT_ERROR)

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
            )
        )
        self._strategy_cache[symbol] = strategy
        return strategy

    def get_account(self) -> Any:
        return cast(Any, self.trading).get_account()

    def _log_skip(self, reason: str, detail: str) -> None:
        logger.info("%s %s", reason, detail)
        print(f"{reason}: {detail}")

    def _preferred_price_stream_feed(self) -> DataFeed:
        raw = os.getenv("PRICE_STREAM_FEED", "iex").strip().lower()
        return _PRICE_STREAM_FEED_BY_NAME.get(raw, DataFeed.IEX)

    def _latest_trade_feeds(self) -> list[DataFeed]:
        preferred = self._preferred_price_stream_feed()
        feeds: list[DataFeed] = []
        for feed in (preferred, DataFeed.IEX, DataFeed.SIP):
            if feed not in feeds:
                feeds.append(feed)
        return feeds

    def _position_holding_minutes(self, symbol: str, now_utc: datetime) -> float | None:
        first_seen = self._position_first_seen_utc.get(symbol)
        if first_seen is None:
            return None
        return max(0.0, (now_utc - first_seen).total_seconds() / 60.0)

    def _update_position_holding_state(
        self,
        positions: dict[str, Position],
        observed_at_utc: datetime,
    ) -> None:
        for symbol in positions:
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

    def _is_market_open(self) -> bool:
        try:
            clock = cast(Any, self.trading).get_clock()
        except Exception as exc:
            raise RuntimeError("Unable to fetch Alpaca market clock.") from exc
        return bool(getattr(clock, "is_open", False))

    def get_price_feed_status(self) -> str:
        with self._price_lock:
            active_symbols = len(self._latest_prices)

        if not self._stream_enabled:
            return "live stream disabled"
        if self._stream_error:
            return f"stream error: {self._stream_error}"
        if self._stream_thread is None:
            return "stream idle"
        if active_symbols == 0:
            return "stream connecting"
        return f"live stream active for {active_symbols}/{len(self.config.symbols)} symbols"

    def _start_price_stream(self) -> None:
        if not self._stream_enabled or self._stream_thread is not None:
            return

        try:
            stream_feed = self._preferred_price_stream_feed()
            self.data_stream = StockDataStream(self._api_key, self._api_secret, feed=stream_feed)
            self.data_stream.subscribe_trades(self._handle_trade, *self.config.symbols)
            self._stream_thread = threading.Thread(target=self._run_price_stream, daemon=True)
            self._stream_thread.start()
            self._stream_error = None
        except Exception as exc:
            self._stream_error = str(exc)
            self.data_stream = None
            self._stream_thread = None

    def _run_price_stream(self) -> None:
        try:
            if self.data_stream is None:
                return
            self.data_stream.run()
        except Exception as exc:
            self._stream_error = str(exc)

    async def _handle_trade(self, trade: Any) -> None:
        symbol = str(getattr(trade, "symbol", ""))
        price = getattr(trade, "price", None)
        if not symbol or price is None:
            return

        with self._price_lock:
            self._latest_prices[symbol] = float(price)
            self._latest_price_times[symbol] = time.time()
            trade_timestamp = getattr(trade, "timestamp", None)
            if isinstance(trade_timestamp, datetime):
                if trade_timestamp.tzinfo is None:
                    normalized_trade_time = trade_timestamp.replace(tzinfo=timezone.utc)
                else:
                    normalized_trade_time = trade_timestamp.astimezone(timezone.utc)
                self._latest_trade_times[symbol] = normalized_trade_time.timestamp()
            else:
                self._latest_trade_times[symbol] = time.time()
            self._stream_error = None

    def get_latest_price_with_age(self, symbol: str) -> tuple[float, float]:
        self._start_price_stream()

        with self._price_lock:
            cached_price = self._latest_prices.get(symbol)
            cached_at = self._latest_price_times.get(symbol)
            cached_trade_at = self._latest_trade_times.get(symbol)

        if cached_price is not None and cached_at is not None and (time.time() - cached_at) <= 15:
            if cached_trade_at is not None:
                return cached_price, max(0.0, time.time() - cached_trade_at)
            return cached_price, time.time() - cached_at

        last_exc: Exception | None = None
        for feed in self._latest_trade_feeds():
            try:
                request = StockLatestTradeRequest(symbol_or_symbols=symbol, feed=feed)
                latest = cast(dict[str, Any], cast(Any, self.data).get_stock_latest_trade(request))
                trade = latest[symbol]
                price = float(trade.price)
                timestamp = getattr(trade, "timestamp", None)
                if isinstance(timestamp, datetime):
                    if timestamp.tzinfo is None:
                        trade_timestamp = timestamp.replace(tzinfo=timezone.utc)
                    else:
                        trade_timestamp = timestamp.astimezone(timezone.utc)
                    age_seconds = max(0.0, (datetime.now(timezone.utc) - trade_timestamp).total_seconds())
                else:
                    age_seconds = 0.0

                with self._price_lock:
                    self._latest_prices[symbol] = price
                    self._latest_price_times[symbol] = time.time()
                    self._latest_trade_times[symbol] = time.time() - age_seconds

                return price, age_seconds
            except Exception as exc:
                last_exc = exc

        if last_exc is not None:
            raise last_exc
        raise RuntimeError(f"Could not resolve latest trade for {symbol}")

    def get_latest_price(self, symbol: str) -> float:
        price, _ = self.get_latest_price_with_age(symbol)
        return price

    def get_positions_by_symbol(self) -> dict[str, Position]:
        positions = cast(list[Position], cast(Any, self.trading).get_all_positions())
        return {position.symbol: position for position in positions}

    def _bar_interval(self) -> timedelta:
        return timedelta(minutes=self.config.bar_timeframe_minutes)

    def get_decision_timestamp(self, now: datetime | None = None) -> datetime:
        current_time = now or datetime.now(timezone.utc)
        bar_seconds = int(self._bar_interval().total_seconds())
        decision_unix = int(current_time.timestamp()) // bar_seconds * bar_seconds
        return datetime.fromtimestamp(decision_unix, tz=timezone.utc)

    def _get_bar_start_time(self, bar: Any) -> datetime:
        raw_timestamp = getattr(bar, "timestamp", None)
        if not isinstance(raw_timestamp, datetime):
            raise RuntimeError("Bar is missing a timestamp.")
        if raw_timestamp.tzinfo is None:
            return raw_timestamp.replace(tzinfo=timezone.utc)
        return raw_timestamp.astimezone(timezone.utc)

    def _get_bars(
        self,
        symbol: str,
        bars_needed: int,
        timeframe_minutes: int,
        decision_timestamp: datetime | None = None,
    ) -> list[Any]:
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
        tf = _minutes_to_timeframe(timeframe_minutes)
        request = StockBarsRequest(
            symbol_or_symbols=[symbol],
            timeframe=tf,
            start=start,
            end=request_end,
            feed=DataFeed.IEX,
        )

        bars_response = cast(Any, self.data).get_stock_bars(request)
        bars = cast(list[Any], bars_response.data.get(symbol, []))
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
    ) -> list[Any]:
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

    def _latest_bar_close_time(self, bars: list[Any]) -> datetime:
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
        strategy = self._strategy_for_symbol(symbol)
        effective_strategy_mode = strategy.config.strategy_mode
        bars_needed = max(
            self.config.sma_bars,
            25,
            50,  # minimum for trend_sma (50-bar SMA used by mean_reversion_trend_filter)
            estimate_atr_percentile_lookback_bars(self.config.bar_timeframe_minutes),
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
        min_history_bars = 2 if effective_strategy_mode in (STRATEGY_MODE_BREAKOUT, STRATEGY_MODE_ORB) else max(20, self.config.sma_bars)
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
        recent_volumes = [float(getattr(bar, "volume", 0.0) or 0.0) for bar in intraday_bars][-20:]
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
            entry = float(position.avg_entry_price)
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
        action = strategy.decide_action(
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
            position_entry_price=float(position.avg_entry_price) if holding and position is not None else None,
            volume_ratio=volume_ratio,
            volatility_ratio=volatility_ratio,
            effective_stop_price=effective_stop_price,
            trend_sma=trend_sma,
        )
        # --- structured signal log ---
        _trend_pass = (price >= trend_sma) if trend_sma is not None else None
        _atr_pass = (
            (_atr_pct_now_val <= self.config.mean_reversion_max_atr_percentile)
            if (_atr_pct_now_val is not None and self.config.mean_reversion_max_atr_percentile > 0)
            else None
        )
        if action in ("BUY", "SELL"):
            _rejection = None
        else:
            _failing = []
            _trend_filter_active = (
                strategy.config.strategy_mode != STRATEGY_MODE_MEAN_REVERSION
                or self.config.mean_reversion_trend_filter
            )
            if _trend_pass is False and _trend_filter_active:
                _failing.append("trend_filter")
            if _atr_pass is False:
                _failing.append("atr_filter")
            if _failing:
                _rejection = "|".join(_failing)
            elif holding:
                _rejection = "holding_no_exit"   # position open, awaiting exit signal
            else:
                _rejection = "no_signal"          # no entry condition met
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
            trend_filter_pass=_trend_pass,
            atr_filter_pass=_atr_pass,
            window_open=_window_open,
            rejection=_rejection,
            ml_prob=ml_signal.probability_up if ml_signal.probability_up != 0.5 else None,
        )
        return SymbolEvaluation(
            price=price,
            sma=sma,
            ml_signal=ml_signal,
            action=action,
            latest_bar_close_utc=latest_bar_close.isoformat(),
            hold_reason=_rejection,
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
            holding = quantity > 0
            market_value = float(position.market_value) if position is not None else 0.0
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
                    position if holding else None,
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
        request = GetOrdersRequest(status=QueryOrderStatus.ALL, limit=limit, nested=False)
        orders = cast(list[Any], cast(Any, self.trading).get_orders(filter=request))
        snapshots: list[OrderSnapshot] = []

        for order in orders:
            _filled_at_raw = getattr(order, "filled_at", None)
            _filled_at_str = str(_filled_at_raw) if _filled_at_raw is not None else None
            snapshots.append(
                OrderSnapshot(
                    order_id=str(getattr(order, "id", "")),
                    submitted_at=str(getattr(order, "submitted_at", None)),
                    filled_at=_filled_at_str,
                    symbol=str(getattr(order, "symbol", "")),
                    side=_normalize_enum_text(getattr(order, "side", "")),
                    status=_normalize_enum_text(getattr(order, "status", "")),
                    qty=float(order.qty) if getattr(order, "qty", None) else None,
                    filled_qty=float(order.filled_qty) if getattr(order, "filled_qty", None) else None,
                    filled_avg_price=(
                        float(order.filled_avg_price)
                        if getattr(order, "filled_avg_price", None)
                        else None
                    ),
                    notional=float(order.notional) if getattr(order, "notional", None) else None,
                )
            )

        return snapshots

    def get_open_orders(self) -> list[Any]:
        request = GetOrdersRequest(status=QueryOrderStatus.OPEN, nested=False)
        return cast(list[Any], cast(Any, self.trading).get_orders(filter=request))

    def _count_recent_orders(self, window_seconds: int = 60) -> int:
        orders = self.get_recent_orders(limit=100)
        now = datetime.now(timezone.utc)
        count = 0
        for order in orders:
            if not order.submitted_at or order.submitted_at == "None":
                continue
            raw_timestamp = order.submitted_at
            if raw_timestamp.endswith("Z"):
                raw_timestamp = raw_timestamp[:-1] + "+00:00"
            try:
                submitted_at = datetime.fromisoformat(raw_timestamp)
            except ValueError:
                continue
            if submitted_at.tzinfo is None:
                submitted_at = submitted_at.replace(tzinfo=timezone.utc)
            else:
                submitted_at = submitted_at.astimezone(timezone.utc)
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
            existing_position_value = abs(float(positions[symbol].market_value))
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
                self.place_market_sell(symbol, position, exit_reason=exit_reason)
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

            fill_qty = float(order.filled_qty)
            cache_key = (order.order_id, round(fill_qty, 6))
            if cache_key in self._logged_fills:
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
                    # Record entry state so we can compute PnL when the position closes.
                    # Only log position.opened on the first fill — partial fills must not
                    # inflate the count.
                    _is_first_fill = order.symbol not in self._position_entry_price
                    self._position_entry_price[order.symbol] = fill_price
                    self._position_entry_ts[order.symbol] = dec_ts
                    self._position_qty[order.symbol] = fill_qty
                    if _is_first_fill:
                        self.blog.position_opened(
                            symbol=order.symbol,
                            decision_ts=dec_ts,
                            entry_price=fill_price,
                            qty=fill_qty,
                            strategy_mode=self.config.strategy_mode,
                        )

                elif side == "sell":
                    entry_price = self._position_entry_price.pop(order.symbol, fill_price)
                    self._position_entry_ts.pop(order.symbol, None)
                    self._position_qty.pop(order.symbol, None)
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
    ) -> Any | None:
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

        order = cast(
            Any,
            self.trading.submit_order(
                MarketOrderRequest(
                    symbol=symbol,
                    qty=qty,
                    side=OrderSide.BUY,
                    time_in_force=TimeInForce.DAY,
                )
            ),
        )
        print(f"Submitted BUY {symbol} qty={qty} approx=${qty * execution_price:.2f}")
        order_id = str(getattr(order, "id", ""))
        dec_ts = decision_ts or self.get_decision_timestamp().isoformat()
        if order_id:
            self._order_submission_ts[order_id] = dec_ts
            self._order_submission_side[order_id] = "buy"
            if signal_price is not None:
                self._order_signal_price[order_id] = signal_price
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
    ) -> Any | None:
        qty = int(float(position.qty))
        if qty <= 0:
            print(f"Skip SELL {symbol}: non-positive quantity {position.qty}")
            return None

        holding_minutes = self._position_holding_minutes(symbol, datetime.now(timezone.utc))

        order = cast(
            Any,
            self.trading.submit_order(
                MarketOrderRequest(
                    symbol=symbol,
                    qty=qty,
                    side=OrderSide.SELL,
                    time_in_force=TimeInForce.DAY,
                )
            ),
        )
        holding_text = f" holding_minutes={holding_minutes:.1f}" if holding_minutes is not None else ""
        print(f"Submitted SELL {symbol} qty={qty}{holding_text}")
        order_id = str(getattr(order, "id", ""))
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
            live_price=float(live_price if live_price is not None else (position.current_price or position.avg_entry_price)),
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
        _hold_reasons: dict[str, int] = {}
        for _item in snapshot.symbols:
            if _item.action == "HOLD" and _item.hold_reason:
                _hold_reasons[_item.hold_reason] = _hold_reasons.get(_item.hold_reason, 0) + 1

        if not should_process:
            print("[CYCLE] Skipping duplicate bar before execution branch")
            _set_cycle_report(processed_bar=False, skip_reason="duplicate_bar")
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
        open_positions = sum(1 for position in positions.values() if _has_long_position(position))
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

                if symbol in open_order_symbols:
                    print(f"Skip {symbol}: existing open order in flight")
                    continue

                if recent_order_count >= self.config.max_orders_per_minute:
                    print("Skip new orders: max order rate reached")
                    break

                if action == "BUY":
                    _dec_ts = decision_timestamp.isoformat() if decision_timestamp else ""
                    if not in_entry_window:
                        print(f"Skip {symbol} BUY: outside trading window ({now_et.strftime('%H:%M:%S')} ET)")
                        self.blog.risk_check(symbol=symbol, decision_ts=_dec_ts, action="BUY",
                            allowed=False, block_reason="outside_entry_window",
                            open_positions=open_positions, max_positions=self.config.max_open_positions,
                            daily_pnl=snapshot.daily_pnl, daily_limit=self.config.max_daily_loss_usd)
                        continue
                    if symbol in positions:
                        print(f"Already holding {symbol}")
                        self.blog.risk_check(symbol=symbol, decision_ts=_dec_ts, action="BUY",
                            allowed=False, block_reason="already_holding",
                            open_positions=open_positions, max_positions=self.config.max_open_positions,
                            daily_pnl=snapshot.daily_pnl, daily_limit=self.config.max_daily_loss_usd)
                        continue
                    if open_positions >= self.config.max_open_positions:
                        print("Max positions reached")
                        self.blog.risk_check(symbol=symbol, decision_ts=_dec_ts, action="BUY",
                            allowed=False, block_reason="max_open_positions_reached",
                            open_positions=open_positions, max_positions=self.config.max_open_positions,
                            daily_pnl=snapshot.daily_pnl, daily_limit=self.config.max_daily_loss_usd)
                        continue
                    live_price, live_price_age = self.get_latest_price_with_age(symbol)
                    if live_price_age > self.config.max_live_price_age_seconds:
                        print(f"Skip {symbol}: stale live price age {live_price_age:.1f}s")
                        self.blog.risk_check(symbol=symbol, decision_ts=_dec_ts, action="BUY",
                            allowed=False, block_reason="stale_live_price",
                            open_positions=open_positions, max_positions=self.config.max_open_positions,
                            daily_pnl=snapshot.daily_pnl, daily_limit=self.config.max_daily_loss_usd,
                            live_price_age_s=round(live_price_age, 1))
                        continue
                    _signal_price = item.price or 0.0
                    _dev_bps = abs((live_price / _signal_price) - 1.0) * 10_000 if _signal_price > 0 else None
                    if self._is_price_collar_breached(_signal_price, live_price):
                        print(
                            f"Skip {symbol}: live price {live_price:.2f} breaches collar vs decision price {_signal_price:.2f}"
                        )
                        self.blog.risk_check(symbol=symbol, decision_ts=_dec_ts, action="BUY",
                            allowed=False, block_reason="price_collar_breached",
                            open_positions=open_positions, max_positions=self.config.max_open_positions,
                            daily_pnl=snapshot.daily_pnl, daily_limit=self.config.max_daily_loss_usd,
                            live_price=live_price, signal_price=_signal_price, price_deviation_bps=_dev_bps)
                        continue
                    if self._is_symbol_exposure_exceeded(symbol, live_price, positions):
                        print(f"Skip {symbol}: max symbol exposure would be exceeded")
                        self.blog.risk_check(symbol=symbol, decision_ts=_dec_ts, action="BUY",
                            allowed=False, block_reason="symbol_exposure_exceeded",
                            open_positions=open_positions, max_positions=self.config.max_open_positions,
                            daily_pnl=snapshot.daily_pnl, daily_limit=self.config.max_daily_loss_usd)
                        continue
                    trade_budget = min(self.config.max_usd_per_trade, remaining_buying_power)
                    if trade_budget < live_price:
                        print(
                            f"Skip {symbol}: trade budget ${trade_budget:.2f} "
                            f"cannot fund one share at {live_price:.2f}"
                        )
                        self.blog.risk_check(symbol=symbol, decision_ts=_dec_ts, action="BUY",
                            allowed=False, block_reason="insufficient_buying_power",
                            open_positions=open_positions, max_positions=self.config.max_open_positions,
                            daily_pnl=snapshot.daily_pnl, daily_limit=self.config.max_daily_loss_usd,
                            live_price=live_price, signal_price=_signal_price, price_deviation_bps=_dev_bps)
                        continue
                    # All checks passed
                    self.blog.risk_check(symbol=symbol, decision_ts=_dec_ts, action="BUY",
                        allowed=True, block_reason=None,
                        open_positions=open_positions, max_positions=self.config.max_open_positions,
                        daily_pnl=snapshot.daily_pnl, daily_limit=self.config.max_daily_loss_usd,
                        live_price=live_price, signal_price=_signal_price, price_deviation_bps=_dev_bps)
                    order = self.place_market_buy(
                        symbol,
                        buying_power_available=remaining_buying_power,
                        price=live_price,
                        decision_ts=_dec_ts,
                        signal_price=_signal_price,
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
                    if live_price_age > self.config.max_live_price_age_seconds:
                        print(f"Skip {symbol}: stale live price age {live_price_age:.1f}s")
                        self.blog.risk_check(symbol=symbol, decision_ts=_dec_ts, action="SELL",
                            allowed=False, block_reason="stale_live_price",
                            open_positions=open_positions, max_positions=self.config.max_open_positions,
                            daily_pnl=snapshot.daily_pnl, daily_limit=self.config.max_daily_loss_usd,
                            live_price_age_s=round(live_price_age, 1))
                        continue
                    _signal_price = item.price or 0.0
                    _dev_bps = abs((live_price / _signal_price) - 1.0) * 10_000 if _signal_price > 0 else None
                    if self._is_price_collar_breached(_signal_price, live_price):
                        print(
                            f"Skip {symbol}: live price {live_price:.2f} breaches collar vs decision price {item.price:.2f}"
                        )
                        self.blog.risk_check(symbol=symbol, decision_ts=_dec_ts, action="SELL",
                            allowed=False, block_reason="price_collar_breached",
                            open_positions=open_positions, max_positions=self.config.max_open_positions,
                            daily_pnl=snapshot.daily_pnl, daily_limit=self.config.max_daily_loss_usd,
                            live_price=live_price, signal_price=_signal_price, price_deviation_bps=_dev_bps)
                        continue
                    self.blog.risk_check(symbol=symbol, decision_ts=_dec_ts, action="SELL",
                        allowed=True, block_reason=None,
                        open_positions=open_positions, max_positions=self.config.max_open_positions,
                        daily_pnl=snapshot.daily_pnl, daily_limit=self.config.max_daily_loss_usd,
                        live_price=live_price, signal_price=_signal_price, price_deviation_bps=_dev_bps)
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
        snapshot = self.build_snapshot(decision_timestamp=decision_timestamp)
        self.record_state(snapshot)
        return snapshot


def main(config: BotConfig | None = None, session_id: str | None = None) -> None:
    load_dotenv(Path.cwd() / ".env")
    if config is None:
        config = load_config()
    bot = AlpacaTradingBot(config, session_id=session_id)
    execute_orders = os.getenv("EXECUTE_ORDERS", "true").lower() != "false"
    bar_interval_seconds = config.bar_timeframe_minutes * 60
    shutdown_event = threading.Event()

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

    wait_seconds = bot._seconds_until_next_bar()
    if wait_seconds > 1:
        next_bar_utc = datetime.now(timezone.utc) + timedelta(seconds=wait_seconds)
        print(f"Aligning to next bar boundary: waiting {wait_seconds:.0f}s until {next_bar_utc.strftime('%H:%M:%S')} UTC")
        shutdown_event.wait(timeout=wait_seconds)
        if shutdown_event.is_set():
            print("Bot loop exited.")
            return

    while not shutdown_event.is_set():
        if bot._is_past_flatten_deadline():
            print("Main loop: past 15:55 ET, exiting without new tick")
            break

        tick_start = time.time()
        try:
            snapshot = bot.run_once(execute_orders=execute_orders)
            if snapshot.kill_switch_triggered:
                print("Main loop: kill switch triggered, exiting")
                shutdown_event.set()
                break
        except Exception as exc:
            print(f"run_once ERROR: {exc}")

        if shutdown_event.is_set():
            break

        elapsed = time.time() - tick_start
        sleep_seconds = max(0.0, bar_interval_seconds - elapsed)
        print(f"Next tick in {sleep_seconds:.0f}s")
        # Wait for the next bar, but wake immediately if flatten thread fires
        shutdown_event.wait(timeout=sleep_seconds)

    print("Bot loop exited.")


if __name__ == "__main__":
    main()

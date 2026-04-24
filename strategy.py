from dataclasses import dataclass
from datetime import time as dt_time
import logging

import pandas as pd

logger = logging.getLogger(__name__)

ATR_PERIOD = 14
ATR_PERCENTILE_LOOKBACK_DAYS = 20
ADX_PERIOD = 14
REGIME_SMA_PERIOD = 50
REGIME_TIMEFRAME_MINUTES = 60
ML_BUY_THRESHOLD = 0.55
ML_SELL_THRESHOLD = 0.45
STRATEGY_MODE_SMA = "sma"
STRATEGY_MODE_ML = "ml"
STRATEGY_MODE_HYBRID = "hybrid"
STRATEGY_MODE_BREAKOUT = "breakout"
STRATEGY_MODE_MEAN_REVERSION = "mean_reversion"
STRATEGY_MODE_TREND_PULLBACK = "trend_pullback"
STRATEGY_MODE_MOMENTUM_BREAKOUT = "momentum_breakout"
STRATEGY_MODE_VOLATILITY_EXPANSION = "volatility_expansion"
STRATEGY_MODE_ORB = "orb"
STRATEGY_MODE_WICK_FADE = "wick_fade"
STRATEGY_MODE_BOLLINGER_SQUEEZE = "bollinger_squeeze"
STRATEGY_MODE_HYBRID_BB_MR = "hybrid_bb_mr"
STRATEGY_MODE_CHOICES = (
    STRATEGY_MODE_SMA,
    STRATEGY_MODE_ML,
    STRATEGY_MODE_HYBRID,
    STRATEGY_MODE_BREAKOUT,
    STRATEGY_MODE_MEAN_REVERSION,
    STRATEGY_MODE_TREND_PULLBACK,
    STRATEGY_MODE_MOMENTUM_BREAKOUT,
    STRATEGY_MODE_VOLATILITY_EXPANSION,
    STRATEGY_MODE_ORB,
    STRATEGY_MODE_WICK_FADE,
    STRATEGY_MODE_BOLLINGER_SQUEEZE,
    STRATEGY_MODE_HYBRID_BB_MR,
)
ORB_FILTER_NONE = "none"
ORB_FILTER_VOLUME_OR_VOLATILITY = "volume_or_volatility"
ORB_FILTER_CHOICES = (
    ORB_FILTER_NONE,
    ORB_FILTER_VOLUME_OR_VOLATILITY,
)
BREAKOUT_EXIT_TARGET_1X_STOP_LOW = "target_1x_stop_low"
BREAKOUT_EXIT_TARGET_1_5X_STOP_LOW = "target_1_5x_stop_low"
BREAKOUT_EXIT_EOD_ONLY = "eod_only"
BREAKOUT_EXIT_TRAILING_HALF_RANGE = "trailing_stop_half_range"
BREAKOUT_EXIT_TRAILING_FULL_RANGE = "trailing_stop_full_range"
BREAKOUT_EXIT_TARGET_1X_TIGHT_STOP = "target_1x_tight_stop"
BREAKOUT_EXIT_CHOICES = (
    BREAKOUT_EXIT_TARGET_1X_STOP_LOW,
    BREAKOUT_EXIT_TARGET_1_5X_STOP_LOW,
    BREAKOUT_EXIT_EOD_ONLY,
    BREAKOUT_EXIT_TRAILING_HALF_RANGE,
    BREAKOUT_EXIT_TRAILING_FULL_RANGE,
    BREAKOUT_EXIT_TARGET_1X_TIGHT_STOP,
)
MEAN_REVERSION_EXIT_SMA = "sma"
MEAN_REVERSION_EXIT_MIDPOINT = "midpoint"
MEAN_REVERSION_EXIT_EOD = "eod"
MEAN_REVERSION_EXIT_CHOICES = (
    MEAN_REVERSION_EXIT_SMA,
    MEAN_REVERSION_EXIT_MIDPOINT,
    MEAN_REVERSION_EXIT_EOD,
)
TREND_PULLBACK_EXIT_FIXED_BARS = "fixed_bars"
TREND_PULLBACK_EXIT_TAKE_PROFIT = "take_profit"
TREND_PULLBACK_EXIT_HYBRID_TP_OR_TIME = "hybrid_tp_or_time"
TREND_PULLBACK_EXIT_CHOICES = (
    TREND_PULLBACK_EXIT_FIXED_BARS,
    TREND_PULLBACK_EXIT_TAKE_PROFIT,
    TREND_PULLBACK_EXIT_HYBRID_TP_OR_TIME,
)
MOMENTUM_BREAKOUT_EXIT_FIXED_BARS = "fixed_bars"
MOMENTUM_BREAKOUT_EXIT_CHOICES = (
    MOMENTUM_BREAKOUT_EXIT_FIXED_BARS,
)
VOLATILITY_EXPANSION_EXIT_FIXED_BARS = "fixed_bars"
VOLATILITY_EXPANSION_EXIT_CHOICES = (
    VOLATILITY_EXPANSION_EXIT_FIXED_BARS,
)
BOLLINGER_EXIT_MIDDLE_BAND = "middle_band"
BOLLINGER_EXIT_CHOICES = (
    BOLLINGER_EXIT_MIDDLE_BAND,
)
THRESHOLD_MODE_STATIC_PCT = "static_pct"
THRESHOLD_MODE_ATR_MULTIPLE = "atr_multiple"
THRESHOLD_MODE_CHOICES = (
    THRESHOLD_MODE_STATIC_PCT,
    THRESHOLD_MODE_ATR_MULTIPLE,
)
THRESHOLD_MODE_ALIASES = {
    "static": THRESHOLD_MODE_STATIC_PCT,
    "dynamic": THRESHOLD_MODE_ATR_MULTIPLE,
}
TIME_WINDOW_FULL_DAY = "full_day"
TIME_WINDOW_MORNING_ONLY = "morning_only"
TIME_WINDOW_AFTERNOON_ONLY = "afternoon_only"
TIME_WINDOW_COMBINED_WINDOWS = "combined_windows"
TIME_WINDOW_CHOICES = (
    TIME_WINDOW_FULL_DAY,
    TIME_WINDOW_MORNING_ONLY,
    TIME_WINDOW_AFTERNOON_ONLY,
    TIME_WINDOW_COMBINED_WINDOWS,
)
TIME_WINDOW_ALIASES = {
    "morning": TIME_WINDOW_MORNING_ONLY,
    "afternoon": TIME_WINDOW_AFTERNOON_ONLY,
    "combined": TIME_WINDOW_COMBINED_WINDOWS,
}
_MORNING_START = dt_time(9, 45)
_MORNING_END = dt_time(11, 30)
_AFTERNOON_START = dt_time(13, 30)
_AFTERNOON_END = dt_time(15, 45)


def estimate_atr_percentile_lookback_bars(bar_timeframe_minutes: int) -> int:
    bars_per_day = max(1, int((6.5 * 60) / bar_timeframe_minutes))
    return ATR_PERIOD + (ATR_PERCENTILE_LOOKBACK_DAYS * bars_per_day)


def estimate_regime_lookback_bars(bar_timeframe_minutes: int) -> int:
    bars_per_hour = max(1, REGIME_TIMEFRAME_MINUTES // max(1, bar_timeframe_minutes))
    return REGIME_SMA_PERIOD * bars_per_hour


def estimate_bollinger_lookback_bars(
    bb_period: int,
    bb_width_lookback: int,
    bb_slope_lookback: int,
) -> int:
    width_ready = max(1, bb_period) + max(1, bb_width_lookback)
    slope_ready = max(1, bb_period) + max(1, bb_slope_lookback)
    return max(width_ready, slope_ready)


def calculate_atr_values(
    highs: list[float],
    lows: list[float],
    closes: list[float],
    period: int = ATR_PERIOD,
) -> list[float | None]:
    if not highs or len(highs) != len(lows) or len(highs) != len(closes):
        return []

    true_ranges: list[float] = []
    for idx, (high, low) in enumerate(zip(highs, lows)):
        if idx == 0:
            true_ranges.append(high - low)
            continue
        prev_close = closes[idx - 1]
        true_ranges.append(max(high - low, abs(high - prev_close), abs(low - prev_close)))

    atr_values: list[float | None] = [None] * len(closes)
    if len(true_ranges) < period:
        return atr_values

    first_atr = sum(true_ranges[:period]) / period
    atr_values[period - 1] = first_atr
    prev_atr = first_atr
    for idx in range(period, len(true_ranges)):
        prev_atr = ((prev_atr * (period - 1)) + true_ranges[idx]) / period
        atr_values[idx] = prev_atr
    return atr_values


def calculate_mean_reversion_signal(
    current_price: float,
    sma: float,
    entry_threshold: float,
    is_holding: bool,
    exit_target: float,
) -> tuple[float, str]:
    """Return (signal_pct, label) for mean reversion mode.

    Pre-entry: 0.0 at/above SMA, 1.0 at/below entry threshold ("Distance to Entry").
    Post-entry: 1.0 at entry threshold, 0.0 at exit target ("Recovery Progress").
    """
    if not is_holding:
        denom = sma - entry_threshold
        if denom <= 0:
            return 0.0, "Distance to Entry"
        return max(0.0, min(1.0, (sma - current_price) / denom)), "Distance to Entry"
    else:
        denom = exit_target - entry_threshold
        if denom <= 0:
            return 1.0, "Recovery Progress"
        return max(0.0, min(1.0, (exit_target - current_price) / denom)), "Recovery Progress"


def calculate_atr_pct_values(
    highs: list[float],
    lows: list[float],
    closes: list[float],
    period: int = ATR_PERIOD,
) -> list[float | None]:
    atr_values = calculate_atr_values(highs, lows, closes, period=period)
    atr_pct_values: list[float | None] = [None] * len(closes)
    for idx, atr_value in enumerate(atr_values):
        close = closes[idx]
        if atr_value is None or close <= 0:
            continue
        atr_pct_values[idx] = atr_value / close
    return atr_pct_values


def calculate_atr_percentile_series(
    timestamps: list[pd.Timestamp],
    highs: list[float],
    lows: list[float],
    closes: list[float],
    period: int = ATR_PERIOD,
    lookback_days: int = ATR_PERCENTILE_LOOKBACK_DAYS,
) -> list[float | None]:
    atr_values = calculate_atr_values(highs, lows, closes, period=period)
    percentile_values: list[float | None] = [None] * len(closes)
    if not atr_values:
        return percentile_values

    trading_dates: list = []
    for ts in timestamps:
        stamp = pd.Timestamp(ts)
        if stamp.tzinfo is None:
            stamp = stamp.tz_localize("UTC")
        trading_dates.append(stamp.tz_convert("America/New_York").date())

    for idx, current_atr in enumerate(atr_values):
        if current_atr is None:
            continue

        window_atrs: list[float] = []
        seen_days: list = []
        for back_idx in range(idx, -1, -1):
            trading_day = trading_dates[back_idx]
            if not seen_days or trading_day != seen_days[-1]:
                seen_days.append(trading_day)
                if len(seen_days) > lookback_days:
                    break
            atr_value = atr_values[back_idx]
            if atr_value is not None:
                window_atrs.append(atr_value)

        if window_atrs:
            percentile_values[idx] = 100.0 * sum(value <= current_atr for value in window_atrs) / len(window_atrs)

    return percentile_values


def calculate_hourly_regime_series(
    timestamps: list[pd.Timestamp],
    closes: list[float],
    source_bar_minutes: int,
    sma_period: int = REGIME_SMA_PERIOD,
) -> list[bool | None]:
    if not timestamps or len(timestamps) != len(closes):
        return []

    intraday = pd.DataFrame({
        "timestamp": [pd.Timestamp(ts) for ts in timestamps],
        "close": closes,
    }).sort_values("timestamp")
    if intraday["timestamp"].dt.tz is None:
        intraday["timestamp"] = intraday["timestamp"].dt.tz_localize("UTC")

    hourly = (
        intraday.assign(hour_start=intraday["timestamp"].dt.floor("1h"))
        .groupby("hour_start", as_index=False)
        .agg(close=("close", "last"))
        .sort_values("hour_start")
    )
    hourly["sma_50"] = hourly["close"].rolling(sma_period).mean()
    hourly["bullish"] = hourly["close"] > hourly["sma_50"]
    hourly["available_at"] = hourly["hour_start"] + pd.Timedelta(hours=1)

    intraday["decision_time"] = intraday["timestamp"] + pd.Timedelta(minutes=source_bar_minutes)
    merged = pd.merge_asof(
        intraday[["decision_time"]].sort_values("decision_time"),
        hourly[["available_at", "bullish"]].sort_values("available_at"),
        left_on="decision_time",
        right_on="available_at",
        direction="backward",
    )
    return [None if pd.isna(value) else bool(value) for value in merged["bullish"]]


def calculate_vwap_series(
    timestamps: list[pd.Timestamp],
    highs: list[float],
    lows: list[float],
    closes: list[float],
    volumes: list[float],
) -> list[float | None]:
    """Compute intraday session VWAP for each bar, resetting at each trading day.

    VWAP = Σ(typical_price_i × volume_i) / Σ(volume_i)
    where typical_price = (high + low + close) / 3

    Design rules that match the rest of this module's series functions:
    - Returns a parallel list of the same length as the inputs.
    - Accumulators (cum_pv, cum_v) reset at the first bar of each new calendar
      day in the US/Eastern timezone, so overnight carry-over never occurs.
    - The first bar of each session always returns None.  A single-bar VWAP is
      just the typical price of that bar — it is not a meaningful intraday anchor
      and would produce spurious z-scores on the open bar.
    - Returns None for any bar where cumulative session volume is still zero.
    - O(n) single pass; no nested loops or pandas groupby.

    Live usage (batch re-compute each tick, matching existing bot pattern):
        vwap_values = calculate_vwap_series(timestamps, highs, lows, closes, volumes)
        current_vwap = vwap_values[-1]   # None if first bar of session

    Incremental streaming state (one bar at a time):
        Maintain (cum_pv, cum_v, current_day) across calls. On each new bar:
          1. If trading_day != current_day: reset cum_pv=0, cum_v=0, is_first=True
          2. Accumulate: cum_pv += tp * vol; cum_v += vol
          3. If is_first: return None (and clear is_first flag)
          4. Else: return cum_pv / cum_v if cum_v > 0 else None

    Args:
        timestamps: bar timestamps; timezone-aware or naive UTC.
        highs:      per-bar high prices.
        lows:       per-bar low prices.
        closes:     per-bar close prices.
        volumes:    per-bar traded volumes; zero or negative treated as zero.

    Returns:
        Parallel list of VWAP floats; None where not yet computable.
    """
    n = len(timestamps)
    if not timestamps or not (n == len(highs) == len(lows) == len(closes) == len(volumes)):
        return []

    vwap_values: list[float | None] = [None] * n
    cum_pv: float = 0.0
    cum_v: float = 0.0
    current_day = None
    is_first_bar_of_session: bool = False

    for idx in range(n):
        stamp = pd.Timestamp(timestamps[idx])
        if stamp.tzinfo is None:
            stamp = stamp.tz_localize("UTC")
        trading_day = stamp.tz_convert("America/New_York").date()

        # New session: reset accumulators before touching this bar's data.
        if trading_day != current_day:
            current_day = trading_day
            cum_pv = 0.0
            cum_v = 0.0
            is_first_bar_of_session = True

        tp = (highs[idx] + lows[idx] + closes[idx]) / 3.0
        vol = max(0.0, float(volumes[idx]))
        cum_pv += tp * vol
        cum_v += vol

        # Emit None for the opening bar; it seeds the accumulator for bar 2+.
        if is_first_bar_of_session:
            is_first_bar_of_session = False
            continue

        if cum_v > 0.0:
            vwap_values[idx] = cum_pv / cum_v

    return vwap_values


def calculate_bollinger_squeeze_features(
    closes: list[float],
    volumes: list[float] | None = None,
    *,
    period: int = 20,
    stddev_mult: float = 2.0,
    width_lookback: int = 100,
    squeeze_quantile: float = 0.20,
    slope_lookback: int = 3,
    use_volume_confirm: bool = True,
    volume_mult: float = 1.2,
) -> dict[str, list[float | bool | str | None]]:
    n = len(closes)
    if n == 0:
        return {
            "middle": [],
            "upper": [],
            "lower": [],
            "width": [],
            "squeeze_threshold": [],
            "squeeze": [],
            "mid_slope": [],
            "bias": [],
            "breakout_up": [],
            "breakout_down": [],
            "volume_confirm": [],
        }

    close_series = pd.Series(closes, dtype="float64")
    middle = close_series.rolling(window=period, min_periods=period).mean()
    std = close_series.rolling(window=period, min_periods=period).std(ddof=0)
    upper = middle + (stddev_mult * std)
    lower = middle - (stddev_mult * std)
    width = (upper - lower) / middle.where(middle != 0)
    squeeze_threshold = width.rolling(window=width_lookback, min_periods=width_lookback).quantile(squeeze_quantile)
    squeeze = (width <= squeeze_threshold) & width.notna() & squeeze_threshold.notna()
    mid_slope = middle - middle.shift(slope_lookback)

    bias: list[str] = []
    breakout_up: list[bool | None] = []
    breakout_down: list[bool | None] = []
    for idx in range(n):
        close = closes[idx]
        mid = middle.iloc[idx]
        slope = mid_slope.iloc[idx]
        up = upper.iloc[idx]
        down = lower.iloc[idx]
        if pd.notna(mid) and pd.notna(slope) and close > mid and slope > 0:
            bias.append("bullish")
        elif pd.notna(mid) and pd.notna(slope) and close < mid and slope < 0:
            bias.append("bearish")
        else:
            bias.append("neutral")
        breakout_up.append(bool(close > up) if pd.notna(up) else None)
        breakout_down.append(bool(close < down) if pd.notna(down) else None)

    if use_volume_confirm and volumes is not None and len(volumes) == n:
        volume_series = pd.Series(volumes, dtype="float64")
        avg_volume = volume_series.rolling(window=period, min_periods=period).mean()
        volume_confirm_series = (volume_series >= (avg_volume * volume_mult)) & avg_volume.notna()
        volume_confirm: list[bool | None] = [
            bool(value) if pd.notna(avg_volume.iloc[idx]) else None
            for idx, value in enumerate(volume_confirm_series.tolist())
        ]
    else:
        volume_confirm = [True] * n

    return {
        "middle": middle.tolist(),
        "upper": upper.tolist(),
        "lower": lower.tolist(),
        "width": width.tolist(),
        "squeeze_threshold": squeeze_threshold.tolist(),
        "squeeze": [bool(value) for value in squeeze.tolist()],
        "mid_slope": mid_slope.tolist(),
        "bias": bias,
        "breakout_up": breakout_up,
        "breakout_down": breakout_down,
        "volume_confirm": volume_confirm,
    }


def calculate_adx_series(
    highs: list[float],
    lows: list[float],
    closes: list[float],
    period: int = ADX_PERIOD,
) -> list[float | None]:
    """Wilder's Average Directional Index (ADX) for each bar.

    Algorithm:
        1. True Range (TR), +DM, -DM for every bar.
        2. Wilder-smooth each over `period` bars (same EMA formula as ATR).
        3. +DI = 100 * smoothed_+DM / smoothed_TR
           -DI = 100 * smoothed_-DM / smoothed_TR
        4. DX  = 100 * |+DI - -DI| / (+DI + -DI)
        5. ADX = Wilder-smooth DX over `period` bars.

    The first valid ADX value appears at index 2*period - 2 (27 bars for period=14).
    Earlier bars return None.  None is also returned when smoothed TR is zero.

    Consistent with calculate_atr_values: uses simple average for the first
    smoothing window, then Wilder's EMA ((prev*(n-1) + x) / n) thereafter.
    The DI ratio is invariant to this normalisation choice.
    """
    n = len(highs)
    if not highs or not (n == len(lows) == len(closes)):
        return []

    result: list[float | None] = [None] * n
    if n < period * 2 - 1:
        return result

    # --- Step 1: raw TR, +DM, -DM per bar ---
    tr_raw:      list[float] = [highs[0] - lows[0]]
    plus_dm_raw: list[float] = [0.0]
    minus_dm_raw: list[float] = [0.0]

    for i in range(1, n):
        up_move   = highs[i]   - highs[i - 1]
        down_move = lows[i - 1] - lows[i]
        tr_raw.append(max(
            highs[i] - lows[i],
            abs(highs[i]  - closes[i - 1]),
            abs(lows[i]   - closes[i - 1]),
        ))
        plus_dm_raw.append(up_move   if up_move > down_move and up_move > 0   else 0.0)
        minus_dm_raw.append(down_move if down_move > up_move and down_move > 0 else 0.0)

    # --- Step 2: first Wilder window (simple average, matching calculate_atr_values) ---
    tr_s   = sum(tr_raw[:period])   / period
    pdm_s  = sum(plus_dm_raw[:period])  / period
    mdm_s  = sum(minus_dm_raw[:period]) / period

    # --- Step 3: DX series ---
    dx_values: list[float | None] = [None] * n

    def _dx(tr_s: float, pdm_s: float, mdm_s: float) -> float | None:
        if tr_s <= 0:
            return None
        plus_di  = 100.0 * pdm_s / tr_s
        minus_di = 100.0 * mdm_s / tr_s
        di_sum   = plus_di + minus_di
        return 100.0 * abs(plus_di - minus_di) / di_sum if di_sum > 0 else 0.0

    dx_values[period - 1] = _dx(tr_s, pdm_s, mdm_s)
    for i in range(period, n):
        tr_s  = (tr_s  * (period - 1) + tr_raw[i])       / period
        pdm_s = (pdm_s * (period - 1) + plus_dm_raw[i])  / period
        mdm_s = (mdm_s * (period - 1) + minus_dm_raw[i]) / period
        dx_values[i] = _dx(tr_s, pdm_s, mdm_s)

    # --- Step 4: ADX = Wilder-smooth DX ---
    first_adx_idx = 2 * period - 2
    first_dx_window = [dx_values[i] for i in range(period - 1, first_adx_idx + 1)
                       if dx_values[i] is not None]
    if len(first_dx_window) < period:
        return result

    adx: float | None = sum(first_dx_window) / period
    result[first_adx_idx] = adx

    for i in range(first_adx_idx + 1, n):
        dx = dx_values[i]
        if dx is None or adx is None:
            adx = None
        else:
            adx = (adx * (period - 1) + dx) / period
        result[i] = adx

    return result


def get_capped_breakout_stop_price(
    entry_price: float,
    opening_range_low: float,
    breakout_max_stop_pct: float,
) -> float:
    """Return the effective breakout stop price.

    The stop is OR low, raised to a floor of entry_price * (1 - breakout_max_stop_pct)
    so that a very wide opening range cannot produce an outsized loss.

    Args:
        entry_price: actual fill price at entry.
        opening_range_low: low of the opening range for this day.
        breakout_max_stop_pct: maximum tolerated loss as a fraction (e.g. 0.03 = 3%).

    Returns:
        max(opening_range_low, entry_price * (1 - breakout_max_stop_pct))
    """
    max_stop_floor = entry_price * (1.0 - breakout_max_stop_pct)
    return max(opening_range_low, max_stop_floor)


def calculate_opening_range_series(
    timestamps: list[pd.Timestamp],
    highs: list[float],
    lows: list[float],
    opening_range_minutes: int = 30,
) -> tuple[list[float | None], list[float | None]]:
    opening_highs: list[float | None] = [None] * len(timestamps)
    opening_lows: list[float | None] = [None] * len(timestamps)

    current_day = None
    day_highs: list[float] = []
    day_lows: list[float] = []
    range_ready = False
    bars_needed = max(1, opening_range_minutes // 15)
    # Compute OR window end as market-open (9:30 ET) + opening_range_minutes.
    _market_open_minutes = 9 * 60 + 30
    _or_end_minutes = _market_open_minutes + opening_range_minutes
    _or_end_time = dt_time(_or_end_minutes // 60, _or_end_minutes % 60)

    for idx, timestamp in enumerate(timestamps):
        stamp = pd.Timestamp(timestamp)
        if stamp.tzinfo is None:
            stamp = stamp.tz_localize("UTC")
        stamp_et = stamp.tz_convert("America/New_York")
        trading_day = stamp_et.date()
        if trading_day != current_day:
            current_day = trading_day
            day_highs = []
            day_lows = []
            range_ready = False

        if stamp_et.time() < _or_end_time:
            day_highs.append(highs[idx])
            day_lows.append(lows[idx])
            if len(day_highs) >= bars_needed:
                range_ready = True

        if range_ready:
            opening_highs[idx] = max(day_highs)
            opening_lows[idx] = min(day_lows)

    return opening_highs, opening_lows


def normalize_time_window_mode(time_window_mode: str) -> str:
    normalized_mode = time_window_mode.strip().lower()
    return TIME_WINDOW_ALIASES.get(normalized_mode, normalized_mode)


def normalize_threshold_mode(threshold_mode: str) -> str:
    normalized_mode = threshold_mode.strip().lower()
    return THRESHOLD_MODE_ALIASES.get(normalized_mode, normalized_mode)


def normalize_strategy_mode(strategy_mode: str) -> str:
    return strategy_mode.strip().lower()


def normalize_orb_filter_mode(filter_mode: str) -> str:
    return filter_mode.strip().lower()


def normalize_breakout_exit_style(exit_style: str) -> str:
    return exit_style.strip().lower()


def normalize_mean_reversion_exit_style(exit_style: str) -> str:
    return exit_style.strip().lower()


def normalize_trend_pullback_exit_style(exit_style: str) -> str:
    return exit_style.strip().lower()


def normalize_momentum_breakout_exit_style(exit_style: str) -> str:
    return exit_style.strip().lower()


def normalize_volatility_expansion_exit_style(exit_style: str) -> str:
    return exit_style.strip().lower()


def normalize_bollinger_exit_mode(exit_mode: str) -> str:
    return exit_mode.strip().lower()


def strategy_requires_adx(config: "StrategyConfig") -> bool:
    mode = normalize_strategy_mode(config.strategy_mode)
    return bool(
        config.max_adx_threshold > 0
        or (mode == STRATEGY_MODE_TREND_PULLBACK and config.trend_pullback_min_adx > 0)
        or (mode == STRATEGY_MODE_MOMENTUM_BREAKOUT and config.momentum_breakout_min_adx > 0)
    )


def is_entry_window_open(timestamp: pd.Timestamp, time_window_mode: str) -> bool:
    stamp = pd.Timestamp(timestamp)
    if stamp.tzinfo is None:
        stamp = stamp.tz_localize("UTC")
    time_et = stamp.tz_convert("America/New_York").time()
    normalized_mode = normalize_time_window_mode(time_window_mode)

    if normalized_mode == TIME_WINDOW_FULL_DAY:
        return _MORNING_START <= time_et <= _AFTERNOON_END
    if normalized_mode == TIME_WINDOW_MORNING_ONLY:
        return _MORNING_START <= time_et <= _MORNING_END
    if normalized_mode == TIME_WINDOW_AFTERNOON_ONLY:
        return _AFTERNOON_START <= time_et <= _AFTERNOON_END
    if normalized_mode == TIME_WINDOW_COMBINED_WINDOWS:
        return (
            _MORNING_START <= time_et <= _MORNING_END
            or _AFTERNOON_START <= time_et <= _AFTERNOON_END
        )
    raise ValueError(f"Unsupported time window mode: {time_window_mode}")


def is_time_window_open(time_window_mode: str, time_et: dt_time) -> bool:
    # Backward-compatible wrapper for older call sites that already converted to ET.
    reference_timestamp = pd.Timestamp(
        f"2024-01-02 {time_et.isoformat()}",
        tz="America/New_York",
    )
    return is_entry_window_open(reference_timestamp, time_window_mode)


@dataclass(frozen=True)
class StrategyConfig:
    strategy_mode: str
    ml_probability_buy: float = ML_BUY_THRESHOLD
    ml_probability_sell: float = ML_SELL_THRESHOLD
    entry_threshold_pct: float = 0.001
    threshold_mode: str = THRESHOLD_MODE_STATIC_PCT
    atr_multiple: float = 1.0
    atr_percentile_threshold: float = 0.0
    time_window_mode: str = TIME_WINDOW_FULL_DAY
    regime_filter_enabled: bool = False
    orb_target_multiple: float = 1.0
    orb_filter_mode: str = ORB_FILTER_NONE
    # ORB mode parameters
    orb_entry_buffer_pct: float = 0.0015   # 0.15% above OR high required before entry
    orb_min_or_size_pct: float = 0.003     # skip if OR is narrower than 0.3% (flat open)
    orb_max_or_size_pct: float = 0.03      # skip if OR is wider than 3% (binary event)
    orb_hard_stop_pct: float = 0.015       # exit if price drops 1.5% below entry
    breakout_exit_style: str = BREAKOUT_EXIT_TARGET_1X_STOP_LOW
    breakout_tight_stop_fraction: float = 0.5
    breakout_max_stop_pct: float = 0.03    # cap per-trade stop at 3% of entry price
    breakout_gap_pct_min: float = 0.0     # 0 = off; require gap-up >= N% (e.g. 0.003 = 0.3%)
    breakout_or_range_pct_min: float = 0.0  # 0 = off; require OR width >= N% of OR low
    mean_reversion_exit_style: str = MEAN_REVERSION_EXIT_SMA
    mean_reversion_max_atr_percentile: float = 0.0
    mean_reversion_stop_pct: float = 0.0   # 0 = disabled; e.g. 0.02 = exit if price falls 2% below entry
    sma_stop_pct: float = 0.0              # 0 = disabled; e.g. 0.02 = exit if price falls 2% below entry
    mean_reversion_trend_filter: bool = False       # when True, skip entries where price < 50-bar SMA
    mean_reversion_trend_slope_filter: bool = False # when True, skip entries where SMA_50 slope < 0
    vwap_z_entry_threshold: float = 1.5  # |z| required to trigger VWAP Z-score entry
    vwap_z_stop_atr_multiple: float = 2.0  # long stop = entry_price - (mult * ATR)
    min_atr_percentile: float = 20.0  # VWAP MR entry: skip if atr_percentile < this; 0 = disabled
    max_adx_threshold: float = 25.0  # VWAP MR entry: skip if adx >= this (trending); 0 = disabled
    trend_pullback_min_adx: float = 20.0
    trend_pullback_min_slope: float = 0.0
    trend_pullback_entry_threshold: float = 0.0015
    trend_pullback_min_atr_percentile: float = 20.0
    trend_pullback_max_atr_percentile: float = 0.0
    trend_pullback_exit_style: str = TREND_PULLBACK_EXIT_FIXED_BARS
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
    # Wick-fade parameters
    wick_fade_min_lower_wick_ratio: float = 0.4   # lower wick / total range must be >= this
    wick_fade_min_close_position: float = 0.5     # (close - low) / range must be >= this (close in upper half)
    wick_fade_min_range_pct: float = 0.003        # bar range / close must be >= this (dead tape filter)
    wick_fade_stop_atr_multiple: float = 1.5      # stop = entry - N * ATR
    wick_fade_target_atr_multiple: float = 1.0    # target = entry + N * ATR
    wick_fade_max_hold_bars: int = 4              # force exit after N bars (0 = disabled)
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


@dataclass(frozen=True)
class MlSignal:
    probability_up: float
    confidence: float
    training_rows: int
    model_age_seconds: float
    feature_names: tuple[str, ...]
    buy_threshold: float
    sell_threshold: float
    validation_rows: int
    model_name: str


@dataclass(frozen=True)
class DecisionDetails:
    action: str
    reason: str | None = None
    exit_price: float | None = None
    hybrid_branch: str | None = None
    mr_signal: str | None = None
    hybrid_branch_active: str | None = None
    hybrid_entry_branch: str | None = None
    hybrid_regime_branch: str | None = None


class Strategy:
    def __init__(self, config: StrategyConfig) -> None:
        self.config = config

    def _log_ml_decision(
        self,
        mode: str,
        ml_signal: MlSignal,
        action: str,
        reason: str,
        *,
        level: int = logging.INFO,
    ) -> None:
        logger.log(
            level,
            "ML decision mode=%s prob=%.3f buy_thr=%.3f sell_thr=%.3f action=%s reason=%s",
            mode,
            ml_signal.probability_up,
            ml_signal.buy_threshold,
            ml_signal.sell_threshold,
            action,
            reason,
        )

    def _entry_allowed(self, atr_percentile: float | None) -> bool:
        threshold = self.config.atr_percentile_threshold
        if threshold <= 0:
            return True
        return atr_percentile is not None and atr_percentile >= threshold

    def _regime_allows_entry(self, bullish_regime: bool | None) -> bool:
        if not self.config.regime_filter_enabled:
            return True
        return bullish_regime is True

    def _entry_threshold_price(self, sma: float, atr_pct: float | None) -> float | None:
        if self.config.threshold_mode == THRESHOLD_MODE_ATR_MULTIPLE:
            if atr_pct is None:
                return None
            return sma * (1 + (self.config.atr_multiple * atr_pct))
        return sma * (1 + self.config.entry_threshold_pct)

    def _reversion_threshold_price(self, sma: float, atr_pct: float | None) -> float | None:
        if self.config.threshold_mode == THRESHOLD_MODE_ATR_MULTIPLE:
            if atr_pct is None:
                return None
            return sma * (1 - (self.config.atr_multiple * atr_pct))
        return sma * (1 - self.config.entry_threshold_pct)

    def _orb_filter_allows_entry(
        self,
        volume_ratio: float | None,
        volatility_ratio: float | None,
    ) -> bool:
        if self.config.orb_filter_mode == ORB_FILTER_NONE:
            return True
        if self.config.orb_filter_mode == ORB_FILTER_VOLUME_OR_VOLATILITY:
            return (volume_ratio is not None and volume_ratio > 1.0) or (
                volatility_ratio is not None and volatility_ratio > 1.0
            )
        raise ValueError(f"Unsupported ORB filter mode: {self.config.orb_filter_mode}")

    def _breakout_gap_allows_entry(self, gap_pct: float | None) -> bool:
        threshold = self.config.breakout_gap_pct_min
        if threshold <= 0:
            return True
        return gap_pct is not None and gap_pct >= threshold

    def _breakout_or_range_allows_entry(
        self,
        opening_range_high: float | None,
        opening_range_low: float | None,
    ) -> bool:
        threshold = self.config.breakout_or_range_pct_min
        if threshold <= 0:
            return True
        if opening_range_high is None or opening_range_low is None or opening_range_low <= 0:
            return False
        return (opening_range_high - opening_range_low) / opening_range_low >= threshold

    def _mean_reversion_entry_allowed(self, atr_percentile: float | None) -> bool:
        max_threshold = self.config.mean_reversion_max_atr_percentile
        if max_threshold <= 0:
            return True
        return atr_percentile is not None and atr_percentile <= max_threshold

    def _bollinger_volume_confirmed(self, volume_confirm: bool | None) -> bool:
        if not self.config.bb_use_volume_confirm:
            return True
        return volume_confirm is True

    def _bollinger_trend_filter_ok(self, price: float, trend_sma: float | None) -> bool:
        return not self.config.bb_trend_filter or trend_sma is None or price >= trend_sma

    def _trend_pullback_min_atr_allowed(self, atr_percentile: float | None) -> bool:
        threshold = self.config.trend_pullback_min_atr_percentile
        if threshold <= 0:
            return True
        return atr_percentile is not None and atr_percentile >= threshold

    def _trend_pullback_max_atr_allowed(self, atr_percentile: float | None) -> bool:
        threshold = self.config.trend_pullback_max_atr_percentile
        if threshold <= 0:
            return True
        return atr_percentile is not None and atr_percentile <= threshold

    def _trend_pullback_adx_allowed(self, adx: float | None) -> bool:
        threshold = self.config.trend_pullback_min_adx
        if threshold <= 0:
            return True
        return adx is not None and adx >= threshold

    def _trend_pullback_entry_price(self, sma: float) -> float:
        threshold = max(0.0, self.config.trend_pullback_entry_threshold)
        return sma * (1.0 - threshold)

    def _trend_pullback_take_profit_price(self, position_entry_price: float | None) -> float | None:
        take_profit_pct = self.config.trend_pullback_take_profit_pct
        if position_entry_price is None or take_profit_pct <= 0:
            return None
        return position_entry_price * (1.0 + take_profit_pct)

    def _momentum_breakout_take_profit_price(self, position_entry_price: float | None) -> float | None:
        take_profit_pct = self.config.momentum_breakout_take_profit_pct
        if position_entry_price is None or take_profit_pct <= 0:
            return None
        return position_entry_price * (1.0 + take_profit_pct)

    def _momentum_breakout_min_atr_allowed(self, atr_percentile: float | None) -> bool:
        threshold = self.config.momentum_breakout_min_atr_percentile
        if threshold <= 0:
            return True
        return atr_percentile is not None and atr_percentile >= threshold

    def _momentum_breakout_adx_allowed(self, adx: float | None) -> bool:
        threshold = self.config.momentum_breakout_min_adx
        if threshold <= 0:
            return True
        return adx is not None and adx >= threshold

    def _volatility_expansion_take_profit_price(self, position_entry_price: float | None) -> float | None:
        take_profit_pct = self.config.volatility_expansion_take_profit_pct
        if position_entry_price is None or take_profit_pct <= 0:
            return None
        return position_entry_price * (1.0 + take_profit_pct)

    def _volatility_expansion_max_atr_allowed(self, atr_percentile: float | None) -> bool:
        threshold = self.config.volatility_expansion_max_atr_percentile
        if threshold <= 0:
            return True
        return atr_percentile is not None and atr_percentile <= threshold

    def _decide_mean_reversion_action(
        self,
        *,
        price: float,
        sma: float,
        holding: bool,
        atr_pct: float | None,
        atr_percentile: float | None,
        time_window_open: bool,
        bullish_regime: bool | None,
        position_entry_price: float | None,
        mean_reversion_target_price: float | None,
        trend_sma: float | None,
        trend_sma_slope: float | None,
        vwap: float | None,
        adx: float | None,
    ) -> DecisionDetails:
        if vwap is not None and atr_pct is not None:
            atr = atr_pct * price
            if atr <= 0:
                return DecisionDetails(action="HOLD", reason="mean_reversion_degenerate_atr")
            z = (price - vwap) / atr

            if not holding:
                if (
                    z <= -self.config.vwap_z_entry_threshold
                    and time_window_open
                    and self._entry_allowed(atr_percentile)
                    and self._vwap_mr_min_atr_allowed(atr_percentile)
                    and self._vwap_mr_adx_allowed(adx)
                    and self._mean_reversion_entry_allowed(atr_percentile)
                    and self._regime_allows_entry(bullish_regime)
                ):
                    return DecisionDetails(action="BUY", reason="mean_reversion_vwap_entry")
                return DecisionDetails(action="HOLD", reason="mean_reversion_vwap_no_entry")

            if position_entry_price is not None:
                stop = position_entry_price - self.config.vwap_z_stop_atr_multiple * atr
                if price <= stop:
                    return DecisionDetails(action="SELL", reason="mean_reversion_vwap_stop")
            if price >= vwap:
                return DecisionDetails(action="SELL", reason="mean_reversion_vwap_exit")
            return DecisionDetails(action="HOLD", reason="mean_reversion_vwap_hold")

        reversion_entry_price = self._reversion_threshold_price(sma, atr_pct)
        trend_filter_ok = (
            not self.config.mean_reversion_trend_filter
            or trend_sma is None
            or price >= trend_sma
        )
        slope_filter_ok = (
            not self.config.mean_reversion_trend_slope_filter
            or trend_sma_slope is None
            or trend_sma_slope >= 0
        )
        if (
            reversion_entry_price is not None
            and price < reversion_entry_price
            and not holding
            and self._entry_allowed(atr_percentile)
            and self._mean_reversion_entry_allowed(atr_percentile)
            and time_window_open
            and self._regime_allows_entry(bullish_regime)
            and trend_filter_ok
            and slope_filter_ok
        ):
            return DecisionDetails(action="BUY", reason="mean_reversion_sma_entry")
        if holding:
            if (
                self.config.mean_reversion_stop_pct > 0
                and position_entry_price is not None
                and price <= position_entry_price * (1.0 - self.config.mean_reversion_stop_pct)
            ):
                return DecisionDetails(action="SELL", reason="mean_reversion_stop")
            mean_reversion_exit_style = normalize_mean_reversion_exit_style(self.config.mean_reversion_exit_style)
            if mean_reversion_exit_style == MEAN_REVERSION_EXIT_SMA and price >= sma:
                return DecisionDetails(action="SELL", reason="mean_reversion_sma_exit")
            if (
                mean_reversion_exit_style == MEAN_REVERSION_EXIT_MIDPOINT
                and mean_reversion_target_price is not None
                and price >= mean_reversion_target_price
            ):
                return DecisionDetails(action="SELL", reason="mean_reversion_midpoint_exit")
            if mean_reversion_exit_style == MEAN_REVERSION_EXIT_EOD:
                return DecisionDetails(action="HOLD", reason="mean_reversion_eod_hold")
            if mean_reversion_exit_style not in MEAN_REVERSION_EXIT_CHOICES:
                raise ValueError(
                    f"Unsupported mean reversion exit style: {self.config.mean_reversion_exit_style}"
                )
        return DecisionDetails(action="HOLD", reason="mean_reversion_no_signal")

    def _decide_bollinger_squeeze_action(
        self,
        *,
        price: float,
        holding: bool,
        atr_percentile: float | None,
        time_window_open: bool,
        bullish_regime: bool | None,
        bb_middle: float | None,
        bb_upper: float | None,
        bb_lower: float | None,
        bb_prev_squeeze: bool | None,
        bb_mid_slope: float | None,
        bb_bias: str | None,
        bb_breakout_up: bool | None,
        bb_breakout_down: bool | None,
        bb_volume_confirm: bool | None,
        trend_sma: float | None,
    ) -> DecisionDetails:
        if holding:
            if normalize_bollinger_exit_mode(self.config.bb_exit_mode) == BOLLINGER_EXIT_MIDDLE_BAND:
                if bb_middle is not None and price < bb_middle:
                    return DecisionDetails(action="SELL", reason="bollinger_middle_band_exit")
            elif self.config.bb_exit_mode not in BOLLINGER_EXIT_CHOICES:
                raise ValueError(f"Unsupported Bollinger exit mode: {self.config.bb_exit_mode}")
            return DecisionDetails(action="HOLD", reason="bollinger_hold")

        min_mid_slope = max(0.0, self.config.bb_min_mid_slope)
        bullish_bias = (
            bb_bias == "bullish"
            and bb_mid_slope is not None
            and bb_mid_slope >= min_mid_slope
        )
        bearish_bias = (
            bb_bias == "bearish"
            and bb_mid_slope is not None
            and bb_mid_slope <= -min_mid_slope
        )
        breakout_buffer = max(0.0, self.config.bb_breakout_buffer_pct)
        long_breakout = (
            bb_breakout_up is True
            and bb_upper is not None
            and price > (bb_upper * (1.0 + breakout_buffer))
        )
        short_breakout = (
            bb_breakout_down is True
            and bb_lower is not None
            and price < (bb_lower * (1.0 - breakout_buffer))
        )

        if (
            bb_prev_squeeze is True
            and bullish_bias
            and long_breakout
            and self._bollinger_volume_confirmed(bb_volume_confirm)
            and self._bollinger_trend_filter_ok(price, trend_sma)
            and self._entry_allowed(atr_percentile)
            and time_window_open
            and self._regime_allows_entry(bullish_regime)
        ):
            return DecisionDetails(action="BUY", reason="bollinger_breakout_long")
        if (
            bb_prev_squeeze is True
            and bearish_bias
            and short_breakout
            and self._bollinger_volume_confirmed(bb_volume_confirm)
            and self._entry_allowed(atr_percentile)
            and time_window_open
        ):
            return DecisionDetails(action="HOLD", reason="bollinger_short_signal_blocked")
        return DecisionDetails(action="HOLD", reason="bollinger_no_signal")

    def _decide_trend_pullback_action(
        self,
        *,
        price: float,
        sma: float,
        holding: bool,
        atr_percentile: float | None,
        time_window_open: bool,
        bullish_regime: bool | None,
        position_entry_price: float | None,
        trend_sma: float | None,
        trend_sma_slope: float | None,
        adx: float | None,
        bar_high: float | None,
        trend_pullback_bars_held: int,
    ) -> DecisionDetails:
        if holding:
            if (
                self.config.trend_pullback_stop_pct > 0
                and position_entry_price is not None
                and price <= position_entry_price * (1.0 - self.config.trend_pullback_stop_pct)
            ):
                return DecisionDetails(action="SELL", reason="trend_pullback_stop")
            exit_style = normalize_trend_pullback_exit_style(self.config.trend_pullback_exit_style)
            take_profit_price = self._trend_pullback_take_profit_price(position_entry_price)
            if exit_style == TREND_PULLBACK_EXIT_FIXED_BARS:
                if (
                    self.config.trend_pullback_hold_bars > 0
                    and trend_pullback_bars_held >= self.config.trend_pullback_hold_bars
                ):
                    return DecisionDetails(action="SELL", reason="trend_pullback_fixed_bars_exit")
                return DecisionDetails(action="HOLD", reason="trend_pullback_hold")
            if exit_style == TREND_PULLBACK_EXIT_TAKE_PROFIT:
                if take_profit_price is None:
                    raise ValueError("trend_pullback_take_profit_pct must be > 0 for take_profit exit style.")
                if bar_high is not None and bar_high >= take_profit_price:
                    return DecisionDetails(
                        action="SELL",
                        reason="trend_pullback_take_profit_exit",
                        exit_price=take_profit_price,
                    )
                return DecisionDetails(action="HOLD", reason="trend_pullback_take_profit_hold")
            if exit_style == TREND_PULLBACK_EXIT_HYBRID_TP_OR_TIME:
                if take_profit_price is None:
                    raise ValueError("trend_pullback_take_profit_pct must be > 0 for hybrid_tp_or_time exit style.")
                if bar_high is not None and bar_high >= take_profit_price:
                    return DecisionDetails(
                        action="SELL",
                        reason="trend_pullback_take_profit_exit",
                        exit_price=take_profit_price,
                    )
                if (
                    self.config.trend_pullback_hold_bars > 0
                    and trend_pullback_bars_held >= self.config.trend_pullback_hold_bars
                ):
                    return DecisionDetails(action="SELL", reason="trend_pullback_fixed_bars_exit")
                return DecisionDetails(action="HOLD", reason="trend_pullback_hybrid_hold")
            raise ValueError(
                f"Unsupported trend pullback exit style: {self.config.trend_pullback_exit_style}"
            )

        trend_ok = (
            trend_sma is not None
            and price >= trend_sma
            and trend_sma_slope is not None
            and trend_sma_slope >= self.config.trend_pullback_min_slope
        )
        if (
            trend_ok
            and price <= self._trend_pullback_entry_price(sma)
            and self._entry_allowed(atr_percentile)
            and self._trend_pullback_min_atr_allowed(atr_percentile)
            and self._trend_pullback_max_atr_allowed(atr_percentile)
            and self._trend_pullback_adx_allowed(adx)
            and time_window_open
            and self._regime_allows_entry(bullish_regime)
        ):
            return DecisionDetails(action="BUY", reason="trend_pullback_entry")
        return DecisionDetails(action="HOLD", reason="trend_pullback_no_signal")

    def _decide_momentum_breakout_action(
        self,
        *,
        price: float,
        holding: bool,
        atr_percentile: float | None,
        time_window_open: bool,
        bullish_regime: bool | None,
        position_entry_price: float | None,
        trend_sma: float | None,
        trend_sma_slope: float | None,
        adx: float | None,
        bar_high: float | None,
        bar_low: float | None,
        recent_breakout_high: float | None,
        momentum_breakout_bars_held: int,
    ) -> DecisionDetails:
        if holding:
            if (
                self.config.momentum_breakout_stop_pct > 0
                and position_entry_price is not None
                and bar_low is not None
                and bar_low <= position_entry_price * (1.0 - self.config.momentum_breakout_stop_pct)
            ):
                return DecisionDetails(action="SELL", reason="momentum_breakout_stop")
            take_profit_price = self._momentum_breakout_take_profit_price(position_entry_price)
            if take_profit_price is not None and bar_high is not None and bar_high >= take_profit_price:
                return DecisionDetails(
                    action="SELL",
                    reason="momentum_breakout_take_profit_exit",
                    exit_price=take_profit_price,
                )
            exit_style = normalize_momentum_breakout_exit_style(self.config.momentum_breakout_exit_style)
            if exit_style == MOMENTUM_BREAKOUT_EXIT_FIXED_BARS:
                if (
                    self.config.momentum_breakout_hold_bars > 0
                    and momentum_breakout_bars_held >= self.config.momentum_breakout_hold_bars
                ):
                    return DecisionDetails(action="SELL", reason="momentum_breakout_fixed_bars_exit")
                return DecisionDetails(action="HOLD", reason="momentum_breakout_hold")
            raise ValueError(
                f"Unsupported momentum breakout exit style: {self.config.momentum_breakout_exit_style}"
            )

        trend_ok = (
            trend_sma is not None
            and price >= trend_sma
            and trend_sma_slope is not None
            and trend_sma_slope >= self.config.momentum_breakout_min_slope
        )
        breakout_trigger_price = (
            recent_breakout_high * (1.0 + max(0.0, self.config.momentum_breakout_entry_buffer_pct))
            if recent_breakout_high is not None
            else None
        )
        if (
            trend_ok
            and breakout_trigger_price is not None
            and price >= breakout_trigger_price
            and self._entry_allowed(atr_percentile)
            and self._momentum_breakout_min_atr_allowed(atr_percentile)
            and self._momentum_breakout_adx_allowed(adx)
            and time_window_open
            and self._regime_allows_entry(bullish_regime)
        ):
            return DecisionDetails(action="BUY", reason="momentum_breakout_entry")
        return DecisionDetails(action="HOLD", reason="momentum_breakout_no_signal")

    def _decide_volatility_expansion_action(
        self,
        *,
        price: float,
        holding: bool,
        atr_percentile: float | None,
        time_window_open: bool,
        bullish_regime: bool | None,
        position_entry_price: float | None,
        trend_sma: float | None,
        trend_sma_slope: float | None,
        bar_high: float | None,
        bar_low: float | None,
        recent_breakout_high: float | None,
        bb_prev_squeeze: bool | None,
        bb_breakout_up: bool | None,
        bb_volume_confirm: bool | None,
        volatility_expansion_bars_held: int,
    ) -> DecisionDetails:
        if holding:
            if (
                self.config.volatility_expansion_stop_pct > 0
                and position_entry_price is not None
                and bar_low is not None
                and bar_low <= position_entry_price * (1.0 - self.config.volatility_expansion_stop_pct)
            ):
                return DecisionDetails(action="SELL", reason="volatility_expansion_stop")
            take_profit_price = self._volatility_expansion_take_profit_price(position_entry_price)
            if take_profit_price is not None and bar_high is not None and bar_high >= take_profit_price:
                return DecisionDetails(
                    action="SELL",
                    reason="volatility_expansion_take_profit_exit",
                    exit_price=take_profit_price,
                )
            exit_style = normalize_volatility_expansion_exit_style(self.config.volatility_expansion_exit_style)
            if exit_style == VOLATILITY_EXPANSION_EXIT_FIXED_BARS:
                if (
                    self.config.volatility_expansion_hold_bars > 0
                    and volatility_expansion_bars_held >= self.config.volatility_expansion_hold_bars
                ):
                    return DecisionDetails(action="SELL", reason="volatility_expansion_fixed_bars_exit")
                return DecisionDetails(action="HOLD", reason="volatility_expansion_hold")
            raise ValueError(
                f"Unsupported volatility expansion exit style: {self.config.volatility_expansion_exit_style}"
            )

        trend_ok = True
        if self.config.volatility_expansion_trend_filter:
            trend_ok = (
                trend_sma is not None
                and price >= trend_sma
                and trend_sma_slope is not None
                and trend_sma_slope >= self.config.volatility_expansion_min_slope
            )

        breakout_trigger_price = (
            recent_breakout_high * (1.0 + max(0.0, self.config.volatility_expansion_entry_buffer_pct))
            if recent_breakout_high is not None
            else None
        )
        volume_ok = not self.config.volatility_expansion_use_volume_confirm or bb_volume_confirm is True
        if (
            bb_prev_squeeze is True
            and bb_breakout_up is True
            and breakout_trigger_price is not None
            and price >= breakout_trigger_price
            and trend_ok
            and volume_ok
            and self._entry_allowed(atr_percentile)
            and self._volatility_expansion_max_atr_allowed(atr_percentile)
            and time_window_open
            and self._regime_allows_entry(bullish_regime)
        ):
            return DecisionDetails(action="BUY", reason="volatility_expansion_entry")
        return DecisionDetails(action="HOLD", reason="volatility_expansion_no_signal")

    def _decide_hybrid_bb_mr_action(
        self,
        *,
        price: float,
        sma: float,
        holding: bool,
        atr_pct: float | None,
        atr_percentile: float | None,
        time_window_open: bool,
        bullish_regime: bool | None,
        position_entry_price: float | None,
        mean_reversion_target_price: float | None,
        trend_sma: float | None,
        trend_sma_slope: float | None,
        vwap: float | None,
        adx: float | None,
        bb_middle: float | None,
        bb_upper: float | None,
        bb_lower: float | None,
        bb_prev_squeeze: bool | None,
        bb_mid_slope: float | None,
        bb_bias: str | None,
        bb_breakout_up: bool | None,
        bb_breakout_down: bool | None,
        bb_volume_confirm: bool | None,
        hybrid_entry_branch: str | None = None,
    ) -> DecisionDetails:
        mr_details = self._decide_mean_reversion_action(
            price=price,
            sma=sma,
            holding=holding,
            atr_pct=atr_pct,
            atr_percentile=atr_percentile,
            time_window_open=time_window_open,
            bullish_regime=bullish_regime,
            position_entry_price=position_entry_price,
            mean_reversion_target_price=mean_reversion_target_price,
            trend_sma=trend_sma,
            trend_sma_slope=trend_sma_slope,
            vwap=vwap,
            adx=adx,
        )
        regime_branch = "bollinger_breakout" if bb_prev_squeeze is True else "mean_reversion"
        active_branch = hybrid_entry_branch if holding and hybrid_entry_branch else regime_branch
        if active_branch == "bollinger_breakout":
            selected = self._decide_bollinger_squeeze_action(
                price=price,
                holding=holding,
                atr_percentile=atr_percentile,
                time_window_open=time_window_open,
                bullish_regime=bullish_regime,
                bb_middle=bb_middle,
                bb_upper=bb_upper,
                bb_lower=bb_lower,
                bb_prev_squeeze=bb_prev_squeeze,
                bb_mid_slope=bb_mid_slope,
                bb_bias=bb_bias,
                bb_breakout_up=bb_breakout_up,
                bb_breakout_down=bb_breakout_down,
                bb_volume_confirm=bb_volume_confirm,
                trend_sma=trend_sma,
            )
        else:
            selected = mr_details
        entry_branch = hybrid_entry_branch
        if not holding and selected.action == "BUY":
            entry_branch = active_branch
        return DecisionDetails(
            action=selected.action,
            reason=selected.reason,
            hybrid_branch=active_branch,
            mr_signal=mr_details.action,
            hybrid_branch_active=active_branch,
            hybrid_entry_branch=entry_branch,
            hybrid_regime_branch=regime_branch,
        )

    def decide_action_details(
        self,
        price: float,
        sma: float,
        ml_signal: MlSignal,
        holding: bool,
        atr_pct: float | None = None,
        atr_percentile: float | None = None,
        time_window_open: bool = True,
        bullish_regime: bool | None = None,
        opening_range_high: float | None = None,
        opening_range_low: float | None = None,
        position_entry_price: float | None = None,
        volume_ratio: float | None = None,
        volatility_ratio: float | None = None,
        trailing_stop_price: float | None = None,
        mean_reversion_target_price: float | None = None,
        breakout_already_taken: bool = False,
        effective_stop_price: float | None = None,
        gap_pct: float | None = None,
        trend_sma: float | None = None,
        trend_sma_slope: float | None = None,
        vwap: float | None = None,
        adx: float | None = None,
        bar_high: float | None = None,
        bar_low: float | None = None,
        bar_open: float | None = None,
        wick_fade_stop: float = 0.0,
        wick_fade_target: float = 0.0,
        wick_fade_bars_held: int = 0,
        trend_pullback_bars_held: int = 0,
        recent_breakout_high: float | None = None,
        momentum_breakout_bars_held: int = 0,
        volatility_expansion_bars_held: int = 0,
        bb_middle: float | None = None,
        bb_upper: float | None = None,
        bb_lower: float | None = None,
        bb_prev_squeeze: bool | None = None,
        bb_mid_slope: float | None = None,
        bb_bias: str | None = None,
        bb_breakout_up: bool | None = None,
        bb_breakout_down: bool | None = None,
        bb_volume_confirm: bool | None = None,
        hybrid_entry_branch: str | None = None,
    ) -> DecisionDetails:
        mode = normalize_strategy_mode(self.config.strategy_mode)

        if mode == STRATEGY_MODE_HYBRID_BB_MR:
            return self._decide_hybrid_bb_mr_action(
                price=price,
                sma=sma,
                holding=holding,
                atr_pct=atr_pct,
                atr_percentile=atr_percentile,
                time_window_open=time_window_open,
                bullish_regime=bullish_regime,
                position_entry_price=position_entry_price,
                mean_reversion_target_price=mean_reversion_target_price,
                trend_sma=trend_sma,
                trend_sma_slope=trend_sma_slope,
                vwap=vwap,
                adx=adx,
                bb_middle=bb_middle,
                bb_upper=bb_upper,
                bb_lower=bb_lower,
                bb_prev_squeeze=bb_prev_squeeze,
                bb_mid_slope=bb_mid_slope,
                bb_bias=bb_bias,
                bb_breakout_up=bb_breakout_up,
                bb_breakout_down=bb_breakout_down,
                bb_volume_confirm=bb_volume_confirm,
                hybrid_entry_branch=hybrid_entry_branch,
            )
        if mode == STRATEGY_MODE_TREND_PULLBACK:
            return self._decide_trend_pullback_action(
                price=price,
                sma=sma,
                holding=holding,
                atr_percentile=atr_percentile,
                time_window_open=time_window_open,
                bullish_regime=bullish_regime,
                position_entry_price=position_entry_price,
                trend_sma=trend_sma,
                trend_sma_slope=trend_sma_slope,
                adx=adx,
                bar_high=bar_high,
                trend_pullback_bars_held=trend_pullback_bars_held,
            )
        if mode == STRATEGY_MODE_MOMENTUM_BREAKOUT:
            return self._decide_momentum_breakout_action(
                price=price,
                holding=holding,
                atr_percentile=atr_percentile,
                time_window_open=time_window_open,
                bullish_regime=bullish_regime,
                position_entry_price=position_entry_price,
                trend_sma=trend_sma,
                trend_sma_slope=trend_sma_slope,
                adx=adx,
                bar_high=bar_high,
                bar_low=bar_low,
                recent_breakout_high=recent_breakout_high,
                momentum_breakout_bars_held=momentum_breakout_bars_held,
            )
        if mode == STRATEGY_MODE_VOLATILITY_EXPANSION:
            return self._decide_volatility_expansion_action(
                price=price,
                holding=holding,
                atr_percentile=atr_percentile,
                time_window_open=time_window_open,
                bullish_regime=bullish_regime,
                position_entry_price=position_entry_price,
                trend_sma=trend_sma,
                trend_sma_slope=trend_sma_slope,
                bar_high=bar_high,
                bar_low=bar_low,
                recent_breakout_high=recent_breakout_high,
                bb_prev_squeeze=bb_prev_squeeze,
                bb_breakout_up=bb_breakout_up,
                bb_volume_confirm=bb_volume_confirm,
                volatility_expansion_bars_held=volatility_expansion_bars_held,
            )
        return DecisionDetails(
            action=self.decide_action(
                price=price,
                sma=sma,
                ml_signal=ml_signal,
                holding=holding,
                atr_pct=atr_pct,
                atr_percentile=atr_percentile,
                time_window_open=time_window_open,
                bullish_regime=bullish_regime,
                opening_range_high=opening_range_high,
                opening_range_low=opening_range_low,
                position_entry_price=position_entry_price,
                volume_ratio=volume_ratio,
                volatility_ratio=volatility_ratio,
                trailing_stop_price=trailing_stop_price,
                mean_reversion_target_price=mean_reversion_target_price,
                breakout_already_taken=breakout_already_taken,
                effective_stop_price=effective_stop_price,
                gap_pct=gap_pct,
                trend_sma=trend_sma,
                trend_sma_slope=trend_sma_slope,
                vwap=vwap,
                adx=adx,
                bar_high=bar_high,
                bar_low=bar_low,
                bar_open=bar_open,
                wick_fade_stop=wick_fade_stop,
                wick_fade_target=wick_fade_target,
                wick_fade_bars_held=wick_fade_bars_held,
                trend_pullback_bars_held=trend_pullback_bars_held,
                recent_breakout_high=recent_breakout_high,
                momentum_breakout_bars_held=momentum_breakout_bars_held,
                volatility_expansion_bars_held=volatility_expansion_bars_held,
                bb_middle=bb_middle,
                bb_upper=bb_upper,
                bb_lower=bb_lower,
                bb_prev_squeeze=bb_prev_squeeze,
                bb_mid_slope=bb_mid_slope,
                bb_bias=bb_bias,
                bb_breakout_up=bb_breakout_up,
                bb_breakout_down=bb_breakout_down,
                bb_volume_confirm=bb_volume_confirm,
                hybrid_entry_branch=hybrid_entry_branch,
            )
        )

    def _vwap_mr_min_atr_allowed(self, atr_percentile: float | None) -> bool:
        """Block VWAP MR entries during low-volatility regimes.

        Returns False only when atr_percentile is a known value strictly below
        the configured minimum.  None is treated as unknown → fail open (True),
        so bars with missing percentile data are never silently skipped.
        """
        threshold = self.config.min_atr_percentile
        if threshold <= 0:
            return True
        if atr_percentile is None:
            return True  # fail open — never block on missing data
        return atr_percentile >= threshold

    def _vwap_mr_adx_allowed(self, adx: float | None) -> bool:
        """Block VWAP MR entries during strong directional trends.

        Returns False only when adx is a known value >= the configured maximum.
        None is treated as unknown → fail open (True), consistent with the
        ATR percentile filter convention.
        """
        threshold = self.config.max_adx_threshold
        if threshold <= 0:
            return True
        if adx is None:
            return True  # fail open — never block on missing data
        return adx < threshold

    def decide_action(
        self,
        price: float,
        sma: float,
        ml_signal: MlSignal,
        holding: bool,
        atr_pct: float | None = None,
        atr_percentile: float | None = None,
        time_window_open: bool = True,
        bullish_regime: bool | None = None,
        opening_range_high: float | None = None,
        opening_range_low: float | None = None,
        position_entry_price: float | None = None,
        volume_ratio: float | None = None,
        volatility_ratio: float | None = None,
        trailing_stop_price: float | None = None,
        mean_reversion_target_price: float | None = None,
        breakout_already_taken: bool = False,
        effective_stop_price: float | None = None,
        gap_pct: float | None = None,
        trend_sma: float | None = None,
        trend_sma_slope: float | None = None,
        vwap: float | None = None,
        adx: float | None = None,
        bar_high: float | None = None,
        bar_low: float | None = None,
        bar_open: float | None = None,
        wick_fade_stop: float = 0.0,
        wick_fade_target: float = 0.0,
        wick_fade_bars_held: int = 0,
        trend_pullback_bars_held: int = 0,
        recent_breakout_high: float | None = None,
        momentum_breakout_bars_held: int = 0,
        volatility_expansion_bars_held: int = 0,
        bb_middle: float | None = None,
        bb_upper: float | None = None,
        bb_lower: float | None = None,
        bb_prev_squeeze: bool | None = None,
        bb_mid_slope: float | None = None,
        bb_bias: str | None = None,
        bb_breakout_up: bool | None = None,
        bb_breakout_down: bool | None = None,
        bb_volume_confirm: bool | None = None,
        hybrid_entry_branch: str | None = None,
    ) -> str:
        mode = normalize_strategy_mode(self.config.strategy_mode)

        if mode == STRATEGY_MODE_SMA:
            threshold_price = self._entry_threshold_price(sma, atr_pct)
            if (
                threshold_price is not None
                and
                price > threshold_price
                and not holding
                and self._entry_allowed(atr_percentile)
                and time_window_open
                and self._regime_allows_entry(bullish_regime)
            ):
                return "BUY"
            if holding:
                stop_triggered = (
                    self.config.sma_stop_pct > 0
                    and position_entry_price is not None
                    and price <= position_entry_price * (1.0 - self.config.sma_stop_pct)
                )
                if price < sma or stop_triggered:
                    return "SELL"
            return "HOLD"

        if mode == STRATEGY_MODE_ML:
            if (
                ml_signal.probability_up >= ml_signal.buy_threshold
                and not holding
                and self._entry_allowed(atr_percentile)
                and time_window_open
                and self._regime_allows_entry(bullish_regime)
            ):
                self._log_ml_decision(mode, ml_signal, "BUY", "ml_only_threshold_passed")
                return "BUY"
            if ml_signal.probability_up <= ml_signal.sell_threshold and holding:
                self._log_ml_decision(mode, ml_signal, "SELL", "ml_only_sell_threshold_passed")
                return "SELL"
            reason = "holding_without_sell_signal" if holding else "ml_buy_threshold_not_met"
            if not holding and not self._entry_allowed(atr_percentile):
                reason = "atr_percentile_blocked"
            elif not holding and not time_window_open:
                reason = "time_window_blocked"
            elif not holding and not self._regime_allows_entry(bullish_regime):
                reason = "regime_filter_blocked"
            self._log_ml_decision(mode, ml_signal, "HOLD", reason, level=logging.DEBUG)
            return "HOLD"

        if mode == STRATEGY_MODE_BREAKOUT:
            orb_range = None
            if opening_range_high is not None and opening_range_low is not None:
                orb_range = opening_range_high - opening_range_low
            if (
                opening_range_high is not None
                and orb_range is not None
                and orb_range > 0
                and price > opening_range_high
                and not holding
                and self._entry_allowed(atr_percentile)
                and time_window_open
                and self._regime_allows_entry(bullish_regime)
                and self._orb_filter_allows_entry(volume_ratio, volatility_ratio)
                and self._breakout_gap_allows_entry(gap_pct)
                and self._breakout_or_range_allows_entry(opening_range_high, opening_range_low)
            ):
                return "BUY"
            if holding and position_entry_price is not None and orb_range is not None and orb_range > 0:
                breakout_exit_style = normalize_breakout_exit_style(self.config.breakout_exit_style)
                target_multiple = 1.0
                stop_price = None
                if breakout_exit_style == BREAKOUT_EXIT_TARGET_1_5X_STOP_LOW:
                    target_multiple = 1.5
                    stop_price = effective_stop_price if effective_stop_price is not None else opening_range_low
                elif breakout_exit_style == BREAKOUT_EXIT_TARGET_1X_STOP_LOW:
                    stop_price = effective_stop_price if effective_stop_price is not None else opening_range_low
                elif breakout_exit_style == BREAKOUT_EXIT_TARGET_1X_TIGHT_STOP:
                    stop_price = position_entry_price - (self.config.breakout_tight_stop_fraction * orb_range)
                elif breakout_exit_style == BREAKOUT_EXIT_EOD_ONLY:
                    return "HOLD"
                elif breakout_exit_style in {
                    BREAKOUT_EXIT_TRAILING_HALF_RANGE,
                    BREAKOUT_EXIT_TRAILING_FULL_RANGE,
                }:
                    if trailing_stop_price is not None and price <= trailing_stop_price:
                        return "SELL"
                    return "HOLD"
                else:
                    raise ValueError(f"Unsupported breakout exit style: {self.config.breakout_exit_style}")

                target_price = position_entry_price + (target_multiple * orb_range)
                if price >= target_price or (stop_price is not None and price <= stop_price):
                    return "SELL"
            return "HOLD"

        if mode == STRATEGY_MODE_MOMENTUM_BREAKOUT:
            return self._decide_momentum_breakout_action(
                price=price,
                holding=holding,
                atr_percentile=atr_percentile,
                time_window_open=time_window_open,
                bullish_regime=bullish_regime,
                position_entry_price=position_entry_price,
                trend_sma=trend_sma,
                trend_sma_slope=trend_sma_slope,
                adx=adx,
                bar_high=bar_high,
                bar_low=bar_low,
                recent_breakout_high=recent_breakout_high,
                momentum_breakout_bars_held=momentum_breakout_bars_held,
            ).action

        if mode == STRATEGY_MODE_VOLATILITY_EXPANSION:
            return self._decide_volatility_expansion_action(
                price=price,
                holding=holding,
                atr_percentile=atr_percentile,
                time_window_open=time_window_open,
                bullish_regime=bullish_regime,
                position_entry_price=position_entry_price,
                trend_sma=trend_sma,
                trend_sma_slope=trend_sma_slope,
                bar_high=bar_high,
                bar_low=bar_low,
                recent_breakout_high=recent_breakout_high,
                bb_prev_squeeze=bb_prev_squeeze,
                bb_breakout_up=bb_breakout_up,
                bb_volume_confirm=bb_volume_confirm,
                volatility_expansion_bars_held=volatility_expansion_bars_held,
            ).action

        if mode == STRATEGY_MODE_MEAN_REVERSION:
            return self._decide_mean_reversion_action(
                price=price,
                sma=sma,
                holding=holding,
                atr_pct=atr_pct,
                atr_percentile=atr_percentile,
                time_window_open=time_window_open,
                bullish_regime=bullish_regime,
                position_entry_price=position_entry_price,
                mean_reversion_target_price=mean_reversion_target_price,
                trend_sma=trend_sma,
                trend_sma_slope=trend_sma_slope,
                vwap=vwap,
                adx=adx,
            ).action

        if mode == STRATEGY_MODE_TREND_PULLBACK:
            return self._decide_trend_pullback_action(
                price=price,
                sma=sma,
                holding=holding,
                atr_percentile=atr_percentile,
                time_window_open=time_window_open,
                bullish_regime=bullish_regime,
                position_entry_price=position_entry_price,
                trend_sma=trend_sma,
                trend_sma_slope=trend_sma_slope,
                adx=adx,
                bar_high=bar_high,
                trend_pullback_bars_held=trend_pullback_bars_held,
            ).action

        if mode == STRATEGY_MODE_ORB:
            if opening_range_high is None or opening_range_low is None:
                return "HOLD"

            orb_range = opening_range_high - opening_range_low
            orb_range_pct = orb_range / opening_range_low if opening_range_low > 0 else 0.0
            or_size_ok = self.config.orb_min_or_size_pct <= orb_range_pct <= self.config.orb_max_or_size_pct

            if not holding and not breakout_already_taken and or_size_ok:
                entry_level = opening_range_high * (1.0 + self.config.orb_entry_buffer_pct)
                if (
                    price > entry_level
                    and time_window_open
                    and self._regime_allows_entry(bullish_regime)
                    and self._orb_filter_allows_entry(volume_ratio, volatility_ratio)
                ):
                    return "BUY"

            if holding and position_entry_price is not None:
                hard_stop = position_entry_price * (1.0 - self.config.orb_hard_stop_pct)
                if price <= hard_stop:
                    return "SELL"

            return "HOLD"

        if mode == STRATEGY_MODE_WICK_FADE:
            if holding:
                # Exits run on every held bar regardless of bar geometry.
                if wick_fade_stop > 0 and price <= wick_fade_stop:
                    return "SELL"
                if wick_fade_target > 0 and price >= wick_fade_target:
                    return "SELL"
                if self.config.wick_fade_max_hold_bars > 0 and wick_fade_bars_held >= self.config.wick_fade_max_hold_bars:
                    return "SELL"
                return "HOLD"

            # Entry: requires valid OHLC and a wick-rejection pattern.
            if bar_high is None or bar_low is None or bar_open is None:
                return "HOLD"
            bar_range = bar_high - bar_low
            if bar_range <= 0:
                return "HOLD"
            bar_range_pct = bar_range / max(price, 1e-9)
            if bar_range_pct < self.config.wick_fade_min_range_pct:
                return "HOLD"
            body_bottom = min(bar_open, price)
            lower_wick = body_bottom - bar_low
            lower_wick_ratio = lower_wick / bar_range
            close_position = (price - bar_low) / bar_range
            if (
                lower_wick_ratio >= self.config.wick_fade_min_lower_wick_ratio
                and close_position >= self.config.wick_fade_min_close_position
                and time_window_open
                and self._entry_allowed(atr_percentile)
                and self._regime_allows_entry(bullish_regime)
            ):
                return "BUY"
            return "HOLD"

        if mode == STRATEGY_MODE_BOLLINGER_SQUEEZE:
            return self._decide_bollinger_squeeze_action(
                price=price,
                holding=holding,
                atr_percentile=atr_percentile,
                time_window_open=time_window_open,
                bullish_regime=bullish_regime,
                bb_middle=bb_middle,
                bb_upper=bb_upper,
                bb_lower=bb_lower,
                bb_prev_squeeze=bb_prev_squeeze,
                bb_mid_slope=bb_mid_slope,
                bb_bias=bb_bias,
                bb_breakout_up=bb_breakout_up,
                bb_breakout_down=bb_breakout_down,
                bb_volume_confirm=bb_volume_confirm,
                trend_sma=trend_sma,
            ).action

        if mode == STRATEGY_MODE_HYBRID_BB_MR:
            return self._decide_hybrid_bb_mr_action(
                price=price,
                sma=sma,
                holding=holding,
                atr_pct=atr_pct,
                atr_percentile=atr_percentile,
                time_window_open=time_window_open,
                bullish_regime=bullish_regime,
                position_entry_price=position_entry_price,
                mean_reversion_target_price=mean_reversion_target_price,
                trend_sma=trend_sma,
                trend_sma_slope=trend_sma_slope,
                vwap=vwap,
                adx=adx,
                bb_middle=bb_middle,
                bb_upper=bb_upper,
                bb_lower=bb_lower,
                bb_prev_squeeze=bb_prev_squeeze,
                bb_mid_slope=bb_mid_slope,
                bb_bias=bb_bias,
                bb_breakout_up=bb_breakout_up,
                bb_breakout_down=bb_breakout_down,
                bb_volume_confirm=bb_volume_confirm,
            ).action

        # hybrid mode: require SMA trend confirmation for buys, allow either risk signal for sells
        if (
            price > sma
            and ml_signal.probability_up >= ml_signal.buy_threshold
            and not holding
            and self._entry_allowed(atr_percentile)
            and time_window_open
            and self._regime_allows_entry(bullish_regime)
        ):
            self._log_ml_decision(mode, ml_signal, "BUY", "hybrid_ml_and_sma_confirmed")
            return "BUY"
        if holding and (
            price < sma or ml_signal.probability_up <= ml_signal.sell_threshold
        ):
            exit_reason = "hybrid_sma_exit" if price < sma else "hybrid_ml_exit"
            self._log_ml_decision(mode, ml_signal, "SELL", exit_reason)
            return "SELL"
        reason = "holding_without_exit_signal" if holding else "hybrid_confirmation_missing"
        if not holding and price <= sma:
            reason = "hybrid_sma_not_bullish"
        elif not holding and ml_signal.probability_up < ml_signal.buy_threshold:
            reason = "hybrid_ml_threshold_not_met"
        elif not holding and not self._entry_allowed(atr_percentile):
            reason = "atr_percentile_blocked"
        elif not holding and not time_window_open:
            reason = "time_window_blocked"
        elif not holding and not self._regime_allows_entry(bullish_regime):
            reason = "regime_filter_blocked"
        self._log_ml_decision(mode, ml_signal, "HOLD", reason, level=logging.DEBUG)
        return "HOLD"

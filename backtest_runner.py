import argparse
import json
import logging
import math
import pickle
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Any

import pandas as pd

from strategy import (
    BREAKOUT_EXIT_CHOICES,
    BREAKOUT_EXIT_EOD_ONLY,
    BREAKOUT_EXIT_TARGET_1X_STOP_LOW,
    BREAKOUT_EXIT_TARGET_1_5X_STOP_LOW,
    BREAKOUT_EXIT_TARGET_1X_TIGHT_STOP,
    BREAKOUT_EXIT_TRAILING_FULL_RANGE,
    BREAKOUT_EXIT_TRAILING_HALF_RANGE,
    MEAN_REVERSION_EXIT_CHOICES,
    MEAN_REVERSION_EXIT_EOD,
    MEAN_REVERSION_EXIT_SMA,
    MlSignal,
    ORB_FILTER_CHOICES,
    ORB_FILTER_NONE,
    REGIME_SMA_PERIOD,
    STRATEGY_MODE_BREAKOUT,
    STRATEGY_MODE_CHOICES,
    STRATEGY_MODE_HYBRID,
    STRATEGY_MODE_MEAN_REVERSION,
    STRATEGY_MODE_ML,
    STRATEGY_MODE_ORB,
    STRATEGY_MODE_SMA,
    Strategy,
    StrategyConfig,
    THRESHOLD_MODE_CHOICES,
    THRESHOLD_MODE_ATR_MULTIPLE,
    THRESHOLD_MODE_STATIC_PCT,
    TIME_WINDOW_CHOICES,
    TIME_WINDOW_FULL_DAY,
    calculate_atr_pct_values,
    calculate_atr_percentile_series,
    calculate_hourly_regime_series,
    calculate_opening_range_series,
    get_capped_breakout_stop_price,
    is_entry_window_open,
    normalize_strategy_mode,
    normalize_threshold_mode,
    normalize_time_window_mode,
    normalize_orb_filter_mode,
    normalize_breakout_exit_style,
    normalize_mean_reversion_exit_style,
)

logger = logging.getLogger(__name__)
_LOGGED_DATASET_SYMBOL_MESSAGES: set[tuple[str, str, tuple[str, ...]]] = set()


DEFAULT_SMA_BARS = 20
DEFAULT_ENTRY_THRESHOLD_PCT = 0.001
DEFAULT_ATR_MULTIPLE = 1.0
DEFAULT_DISABLED_ATR_PERCENTILE_THRESHOLD = 0.0
DEFAULT_STARTING_CAPITAL = 10000.0
DEFAULT_POSITION_SIZE = 1000.0
DEFAULT_ML_LOOKBACK_BARS = 320
DEFAULT_ML_RETRAIN_EVERY_BARS = 15
DEFAULT_ML_PROBABILITY_BUY = 0.55
DEFAULT_ML_PROBABILITY_SELL = 0.45
OFFLINE_MODEL_PATH = Path(__file__).resolve().parent / "ml" / "models" / "logistic_latest.pkl"


@dataclass(frozen=True)
class BacktestConfig:
    dataset_path: Path
    symbols: list[str] | None = None
    sma_bars: int = DEFAULT_SMA_BARS
    commission: float = 0.01   # $0.01 flat per order side (~$0.005/share for ~2 shares)
    slippage: float = 0.05     # $0.05/share per side (~0.025-0.1% at typical prices)
    entry_threshold_pct: float = DEFAULT_ENTRY_THRESHOLD_PCT
    threshold_mode: str = THRESHOLD_MODE_STATIC_PCT
    atr_multiple: float = DEFAULT_ATR_MULTIPLE
    atr_percentile_threshold: float = DEFAULT_DISABLED_ATR_PERCENTILE_THRESHOLD
    time_window_mode: str = TIME_WINDOW_FULL_DAY
    regime_filter_enabled: bool = False
    orb_filter_mode: str = ORB_FILTER_NONE
    breakout_exit_style: str = BREAKOUT_EXIT_TARGET_1X_STOP_LOW
    breakout_tight_stop_fraction: float = 0.5
    mean_reversion_exit_style: str = MEAN_REVERSION_EXIT_SMA
    mean_reversion_max_atr_percentile: float = 0.0
    starting_capital: float = DEFAULT_STARTING_CAPITAL
    position_size: float = DEFAULT_POSITION_SIZE
    start_date: str | None = None
    end_date: str | None = None
    strategy_mode: str = STRATEGY_MODE_HYBRID
    symbol_strategy_modes: dict[str, str] | None = None
    ml_lookback_bars: int = DEFAULT_ML_LOOKBACK_BARS
    ml_retrain_every_bars: int = DEFAULT_ML_RETRAIN_EVERY_BARS
    ml_probability_buy: float = DEFAULT_ML_PROBABILITY_BUY
    ml_probability_sell: float = DEFAULT_ML_PROBABILITY_SELL


@dataclass(frozen=True)
class PreparedSymbolData:
    symbol: str
    timestamps: list[pd.Timestamp]
    timestamp_strings: list[str]
    opens: list[float]
    highs: list[float]
    lows: list[float]
    closes: list[float]
    volumes: list[float]
    day_keys: list[pd.Timestamp]
    is_eod: list[bool]
    atr_pct: list[float | None]
    atr_percentile: list[float | None]
    bullish_regime: list[bool | None]
    opening_range_high: list[float | None]
    opening_range_low: list[float | None]


# ---------------------------------------------------------------------------
# ML helpers — mirrors trading_bot.py so the backtest uses the same feature
# engineering and training pipeline as the live bot.
# ---------------------------------------------------------------------------

@dataclass
class _BacktestInputs:
    """Normalized config and loaded data produced by _prepare_backtest_inputs."""
    df: pd.DataFrame
    manifest: dict
    symbols: list[str]
    symbol_strategies: dict[str, "Strategy"]
    dummy_ml: "MlSignal"
    strategy_mode: str
    time_window_mode: str
    threshold_mode: str
    orb_filter_mode: str
    breakout_exit_style: str
    breakout_tight_stop_fraction: float
    breakout_max_stop_pct: float
    breakout_gap_pct_min: float
    breakout_or_range_pct_min: float
    mean_reversion_exit_style: str
    mean_reversion_max_atr_percentile: float
    mean_reversion_stop_pct: float
    mean_reversion_trend_filter: bool
    mean_reversion_trend_slope_filter: bool
    regime_filter_enabled: bool
    atr_multiple: float
    atr_percentile_threshold: float
    entry_threshold_pct: float


@dataclass
class _SimState:
    """All precomputed arrays and mutable simulation state for one backtest run."""
    # Precomputed per-symbol arrays (read-only after initialization)
    symbols_dfs: dict[str, pd.DataFrame]
    close_arrs: dict[str, list[float]]
    volume_arrs: dict[str, list[float]]
    timestamp_str_arrs: dict[str, list[str]]
    day_key_arrs: dict[str, list[pd.Timestamp]]
    is_eod_arrs: dict[str, list[bool]]
    atr_pct_arrs: dict[str, list[float | None]]
    atr_percentile_arrs: dict[str, list[float | None]]
    bullish_regime_arrs: dict[str, list[bool | None]]
    opening_range_high_arrs: dict[str, list[float | None]]
    opening_range_low_arrs: dict[str, list[float | None]]
    # Mutable position / trade tracking
    pointers: dict[str, int]
    position: dict[str, bool]
    entry_price: dict[str, float]
    entry_cost: dict[str, float]
    shares_held: dict[str, float]
    latest_mark_price: dict[str, float]
    signal_reference_price: dict[str, float]
    breakout_trailing_high: dict[str, float]
    breakout_range_at_entry: dict[str, float]
    breakout_stored_stop: dict[str, float]
    mean_reversion_target_price: dict[str, float]
    pending_buys: dict[str, tuple[float, pd.Timestamp]]
    pending_sells: set[str]
    # ORB per-day state (one trade per symbol per day)
    orb_entry_taken: dict[str, bool]
    orb_last_day: dict[str, pd.Timestamp | None]
    breakout_day_gap_pct: dict[str, float]
    # ML state
    ml_signals: dict[str, list[MlSignal | None]]
    # Accounting (mutated by _execute_buy / _execute_sell)
    results: dict[str, Any]
    symbol_stats: dict[str, dict[str, Any]]
    trades: list[dict[str, Any]]


def _mean(values: list[float]) -> float:
    return sum(values) / len(values)


def _stddev(values: list[float], mean_val: float) -> float:
    variance = sum((v - mean_val) ** 2 for v in values) / max(1, len(values))
    return math.sqrt(max(variance, 1e-12))


def _build_feature_vector(closes: list[float], volumes: list[float], idx: int) -> list[float]:
    """Identical feature set to trading_bot.py._build_feature_vector."""
    price = closes[idx]
    ret_1 = (price / closes[idx - 1]) - 1.0
    ret_3 = (price / closes[idx - 3]) - 1.0
    ret_5 = (price / closes[idx - 5]) - 1.0
    sma_10 = _mean(closes[idx - 9: idx + 1])
    sma_20 = _mean(closes[idx - 19: idx + 1])
    vol_returns = [
        (closes[j] / closes[j - 1]) - 1.0
        for j in range(idx - 9, idx + 1)
        if j - 1 >= 0
    ]
    avg_vol = _mean(vol_returns)
    avg_volume_10 = _mean(volumes[idx - 9: idx + 1])
    return [
        ret_1, ret_3, ret_5,
        (price / sma_10) - 1.0,
        (price / sma_20) - 1.0,
        _stddev(vol_returns, avg_vol),
        (volumes[idx] / max(avg_volume_10, 1.0)) - 1.0,
    ]


def _load_offline_model_payload(model_path: Path = OFFLINE_MODEL_PATH) -> dict[str, Any]:
    if not model_path.exists():
        raise RuntimeError(
            f"Offline ML model not found at {model_path}. "
            "Train it first so ML and HYBRID backtests can use the saved model."
        )
    with open(model_path, "rb") as fh:
        payload = pickle.load(fh)
    if not isinstance(payload, dict) or "model" not in payload or "feature_cols" not in payload:
        raise RuntimeError("Offline ML model payload is invalid. Expected keys: model, feature_cols.")
    return payload


def _model_age_seconds(payload: dict[str, Any]) -> float:
    metadata = payload.get("metadata", {})
    trained_at = metadata.get("trained_at")
    if not trained_at:
        return 0.0
    try:
        trained_at_ts = datetime.fromisoformat(str(trained_at).replace("Z", "+00:00"))
        return max(0.0, (datetime.now(timezone.utc) - trained_at_ts).total_seconds())
    except Exception:
        return 0.0


def _precompute_ml_signals(
    state: _SimState,
    symbol_strategies: dict[str, Strategy],
    payload: dict[str, Any] | None,
    ml_probability_buy: float,
    ml_probability_sell: float,
) -> dict[str, list[MlSignal | None]]:
    precomputed: dict[str, list[MlSignal | None]] = {}
    if payload is None:
        for symbol, closes in state.close_arrs.items():
            precomputed[symbol] = [None] * len(closes)
        return precomputed

    estimator = payload["model"]
    feature_cols = list(payload["feature_cols"])
    expected_feature_count = 7
    if len(feature_cols) != expected_feature_count:
        raise RuntimeError(
            f"Offline ML model expects {len(feature_cols)} features, but the backtest live-compatible "
            f"pipeline builds {expected_feature_count} features."
        )
    model_age_seconds = _model_age_seconds(payload)
    metadata = payload.get("metadata", {})
    training_rows = int(metadata.get("train_rows", 0) or 0)
    model_name = str(metadata.get("model_name", "logistic"))

    for symbol, closes in state.close_arrs.items():
        symbol_signals: list[MlSignal | None] = [None] * len(closes)
        effective_strategy_mode = symbol_strategies[symbol].config.strategy_mode
        if effective_strategy_mode not in (STRATEGY_MODE_ML, STRATEGY_MODE_HYBRID):
            precomputed[symbol] = symbol_signals
            continue

        volumes = state.volume_arrs[symbol]
        feature_rows: list[list[float]] = []
        indices: list[int] = []
        for idx in range(19, len(closes)):
            feature_rows.append(_build_feature_vector(closes, volumes, idx))
            indices.append(idx)

        if not feature_rows:
            precomputed[symbol] = symbol_signals
            continue

        batch_features = pd.DataFrame(feature_rows, columns=feature_cols)
        probabilities = estimator.predict_proba(batch_features)[:, 1]
        for idx, prob in zip(indices, probabilities):
            probability = float(prob)
            symbol_signals[idx] = MlSignal(
                probability_up=probability,
                confidence=abs(probability - 0.5) * 2.0,
                training_rows=training_rows,
                model_age_seconds=model_age_seconds,
                feature_names=tuple(feature_cols),
                buy_threshold=ml_probability_buy,
                sell_threshold=ml_probability_sell,
                validation_rows=0,
                model_name=model_name,
            )
        precomputed[symbol] = symbol_signals

    return precomputed


def _summarize_ml_signals(
    ml_signals: dict[str, list[MlSignal | None]],
    symbol_strategies: dict[str, Strategy],
) -> dict[str, dict[str, Any]]:
    summary: dict[str, dict[str, Any]] = {}
    for symbol, signals in ml_signals.items():
        mode = symbol_strategies[symbol].config.strategy_mode
        if mode not in (STRATEGY_MODE_ML, STRATEGY_MODE_HYBRID):
            continue
        valid_signals = [signal for signal in signals if signal is not None]
        if not valid_signals:
            summary[symbol] = {
                "count": 0,
                "min_prob": None,
                "max_prob": None,
                "mean_prob": None,
                "buy_threshold": None,
                "sell_threshold": None,
                "above_buy_count": 0,
                "below_sell_count": 0,
            }
            continue

        probabilities = [signal.probability_up for signal in valid_signals]
        buy_threshold = valid_signals[0].buy_threshold
        sell_threshold = valid_signals[0].sell_threshold
        summary[symbol] = {
            "count": len(valid_signals),
            "min_prob": min(probabilities),
            "max_prob": max(probabilities),
            "mean_prob": _mean(probabilities),
            "buy_threshold": buy_threshold,
            "sell_threshold": sell_threshold,
            "above_buy_count": sum(probability >= buy_threshold for probability in probabilities),
            "below_sell_count": sum(probability <= sell_threshold for probability in probabilities),
        }
    return summary


# ---------------------------------------------------------------------------
# Dataset I/O
# ---------------------------------------------------------------------------

def load_dataset(dataset_path: Path) -> tuple[pd.DataFrame, dict]:
    bars_path = dataset_path / "bars.parquet"
    manifest_path = dataset_path / "manifest.json"
    df = pd.read_parquet(bars_path)
    if manifest_path.exists():
        with open(manifest_path) as f:
            manifest = json.load(f)
    else:
        manifest = {}
        logger.warning("Dataset manifest not found at %s; symbol discovery will fall back to dataset content.", manifest_path)
    return df, manifest


def _normalize_symbol_list(symbols: list[str] | None) -> list[str]:
    if symbols is None:
        return []

    normalized: list[str] = []
    seen: set[str] = set()
    for raw_symbol in symbols:
        symbol = str(raw_symbol).strip().upper()
        if not symbol or symbol in seen:
            continue
        seen.add(symbol)
        normalized.append(symbol)
    return normalized


def _discover_dataset_symbols(
    manifest: dict,
    df: pd.DataFrame,
    cli_symbols: list[str] | None,
) -> tuple[list[str], str]:
    available_symbols = _normalize_symbol_list(df["symbol"].dropna().astype(str).tolist())
    if not available_symbols:
        raise ValueError("Dataset contains no symbol rows after the requested date filtering.")

    requested_symbols = _normalize_symbol_list(cli_symbols)
    if requested_symbols:
        return requested_symbols, "cli_override"

    manifest_symbols = _normalize_symbol_list(manifest.get("symbols"))
    if manifest_symbols:
        return manifest_symbols, "dataset_metadata"

    return available_symbols, "dataset_content"


def _load_filtered_dataset(
    dataset_path: Path,
    symbols: list[str] | None,
    start_date: str | None,
    end_date: str | None,
) -> tuple[pd.DataFrame, dict, list[str]]:
    df, manifest = load_dataset(dataset_path)
    if "timestamp" not in df.columns:
        raise ValueError("Dataset must contain a timestamp column.")
    if str(df["timestamp"].dtype) != "datetime64[ns, UTC]":
        raise ValueError(f"Expected UTC timestamps, got {df['timestamp'].dtype}")

    df = df.sort_values(["symbol", "timestamp"]).reset_index(drop=True)

    start_ts = pd.Timestamp(start_date, tz="UTC") if start_date else None
    end_ts = (
        pd.Timestamp(end_date, tz="UTC") + pd.Timedelta(days=1) - pd.Timedelta(microseconds=1)
        if end_date
        else None
    )
    if start_ts is not None and end_ts is not None and start_ts > end_ts:
        raise ValueError("start_date must be <= end_date")
    if start_ts is not None:
        df = df[df["timestamp"] >= start_ts]
    if end_ts is not None:
        df = df[df["timestamp"] <= end_ts]

    requested_symbols, symbol_source = _discover_dataset_symbols(manifest, df, symbols)
    available_symbols = _normalize_symbol_list(df["symbol"].dropna().astype(str).tolist())
    missing_symbols = [symbol for symbol in requested_symbols if symbol not in available_symbols]
    active_symbols = [symbol for symbol in requested_symbols if symbol in available_symbols]

    if missing_symbols and symbol_source == "cli_override":
        raise ValueError(
            "CLI symbol override contains symbols that are unavailable in the dataset/date range: "
            f"{', '.join(missing_symbols)}. Available symbols: {', '.join(available_symbols)}"
        )
    if not active_symbols:
        raise ValueError(
            "No symbols remain after filtering the dataset. "
            f"Source={symbol_source}; requested={requested_symbols}; available={available_symbols}"
        )

    if missing_symbols:
        logger.warning(
            "Some %s symbols were unavailable after dataset/date filtering and will be skipped: %s",
            symbol_source,
            ", ".join(missing_symbols),
        )
    message_key = (str(dataset_path), symbol_source, tuple(active_symbols))
    if message_key not in _LOGGED_DATASET_SYMBOL_MESSAGES:
        logger.info(
            "Loaded dataset symbols source=%s count=%d symbols=%s",
            symbol_source,
            len(active_symbols),
            ", ".join(active_symbols),
        )
        print(
            f"[dataset] symbol source={symbol_source} count={len(active_symbols)} symbols={', '.join(active_symbols)}"
        )
        if missing_symbols:
            print(
                f"[dataset] symbol source={symbol_source} skipped_unavailable={', '.join(missing_symbols)}"
            )
        _LOGGED_DATASET_SYMBOL_MESSAGES.add(message_key)
    df = df[df["symbol"].isin(active_symbols)].copy()
    return df, manifest, active_symbols


def _prepare_symbol_data(symbol: str, sdf: pd.DataFrame) -> PreparedSymbolData:
    sdf = sdf.sort_values("timestamp").reset_index(drop=True).copy()
    timestamps = [pd.Timestamp(ts) for ts in sdf["timestamp"].tolist()]
    timestamps_et = [ts.tz_convert("America/New_York") for ts in timestamps]
    day_keys = [ts.normalize() for ts in timestamps_et]
    is_eod = [
        idx + 1 >= len(day_keys) or day_keys[idx + 1] != day_keys[idx]
        for idx in range(len(day_keys))
    ]
    highs = sdf["high"].tolist()
    lows = sdf["low"].tolist()
    closes = sdf["close"].tolist()
    opening_range_high, opening_range_low = calculate_opening_range_series(timestamps, highs, lows)
    return PreparedSymbolData(
        symbol=symbol,
        timestamps=timestamps,
        timestamp_strings=[str(ts) for ts in timestamps],
        opens=sdf["open"].tolist(),
        highs=highs,
        lows=lows,
        closes=closes,
        volumes=sdf["volume"].tolist(),
        day_keys=day_keys,
        is_eod=is_eod,
        atr_pct=calculate_atr_pct_values(highs, lows, closes),
        atr_percentile=calculate_atr_percentile_series(timestamps, highs, lows, closes),
        bullish_regime=calculate_hourly_regime_series(
            timestamps,
            closes,
            source_bar_minutes=15,
            sma_period=REGIME_SMA_PERIOD,
        ),
        opening_range_high=opening_range_high,
        opening_range_low=opening_range_low,
    )


def _prepare_symbol_data_map(df: pd.DataFrame, symbols: list[str]) -> dict[str, PreparedSymbolData]:
    prepared: dict[str, PreparedSymbolData] = {}
    for symbol in symbols:
        sdf = df[df["symbol"] == symbol]
        if sdf.empty:
            raise ValueError(f"Dataset has no rows for symbol {symbol}")
        prepared[symbol] = _prepare_symbol_data(symbol, sdf)
    return prepared


def _build_performance_summary(
    starting_capital: float,
    final_equity: float,
    max_drawdown_pct: float,
    total_trades: int,
    winning_pnls: list[float],
    losing_pnls: list[float],
    realized_pnl: float,
    final_cash: float,
    position_size: float,
    daily_equity_by_day: dict[pd.Timestamp, float],
) -> dict[str, float | int]:
    closed = winning_pnls + losing_pnls
    ordered_days = sorted(daily_equity_by_day.items(), key=lambda item: item[0])
    daily_pnls: list[float] = []
    daily_returns: list[float] = []
    prev_equity = starting_capital
    for _, day_equity in ordered_days:
        daily_pnl = day_equity - prev_equity
        daily_pnls.append(daily_pnl)
        daily_returns.append((day_equity / prev_equity) - 1.0 if prev_equity > 0 else 0.0)
        prev_equity = day_equity

    pnl_stability = _stddev(daily_pnls, _mean(daily_pnls)) if len(daily_pnls) >= 2 else 0.0
    gross_profit = sum(pnl for pnl in winning_pnls if pnl > 0)
    gross_loss = abs(sum(pnl for pnl in losing_pnls if pnl < 0))
    expectancy = (realized_pnl / len(closed)) if closed else 0.0
    max_loss_trade = min(closed) if closed else 0.0
    if len(daily_returns) >= 2:
        avg_daily_return = _mean(daily_returns)
        daily_return_std = _stddev(daily_returns, avg_daily_return)
        sharpe_ratio = (avg_daily_return / daily_return_std) * math.sqrt(252) if daily_return_std > 0 else 0.0
    else:
        sharpe_ratio = 0.0

    return {
        "starting_capital": starting_capital,
        "final_equity": final_equity,
        "total_return_pct": ((final_equity - starting_capital) / starting_capital * 100) if starting_capital > 0 else 0.0,
        "max_drawdown_pct": max_drawdown_pct,
        "total_trades": total_trades,
        "trades_per_day": total_trades / len(ordered_days) if ordered_days else 0.0,
        "win_rate": len(winning_pnls) / max(1, len(closed)) * 100,
        "avg_winning_trade": sum(winning_pnls) / len(winning_pnls) if winning_pnls else 0.0,
        "avg_losing_trade": sum(losing_pnls) / len(losing_pnls) if losing_pnls else 0.0,
        "expectancy": expectancy,
        "max_loss_trade": max_loss_trade,
        "avg_return_per_trade_pct": (
            realized_pnl / (len(closed) * position_size) * 100
            if closed and position_size > 0
            else 0.0
        ),
        "avg_move_captured_pct": (
            realized_pnl / (len(closed) * position_size) * 100
            if closed and position_size > 0
            else 0.0
        ),
        "profit_factor": (gross_profit / gross_loss) if gross_loss > 0 else (float("inf") if gross_profit > 0 else 0.0),
        "sharpe_ratio": sharpe_ratio,
        "pnl_stability": pnl_stability,
        "realized_pnl": realized_pnl,
        "final_cash": final_cash,
    }


def _update_mark_to_market(
    symbols: list[str],
    position: dict[str, bool],
    shares_held: dict[str, float],
    latest_mark_price: dict[str, float],
    cash: float,
    peak_equity: float,
    max_drawdown_pct: float,
    symbol_stats: dict[str, dict[str, Any]],
) -> tuple[float, float, float]:
    open_positions_value = 0.0
    for symbol in symbols:
        market_value = shares_held[symbol] * latest_mark_price[symbol] if position[symbol] else 0.0
        open_positions_value += market_value
        symbol_stats[symbol]["open_positions_value"] = market_value
        symbol_equity = symbol_stats[symbol]["cash"] + market_value
        symbol_stats[symbol]["peak_equity"] = max(symbol_stats[symbol]["peak_equity"], symbol_equity)
        if symbol_stats[symbol]["peak_equity"] > 0:
            symbol_stats[symbol]["max_drawdown_pct"] = max(
                symbol_stats[symbol]["max_drawdown_pct"],
                (symbol_stats[symbol]["peak_equity"] - symbol_equity) / symbol_stats[symbol]["peak_equity"] * 100,
            )

    current_equity = cash + open_positions_value
    peak_equity = max(peak_equity, current_equity)
    if peak_equity > 0:
        max_drawdown_pct = max(max_drawdown_pct, (peak_equity - current_equity) / peak_equity * 100)
    return open_positions_value, peak_equity, max_drawdown_pct


def _execute_sell(
    *,
    results: dict[str, Any],
    symbol_stats: dict[str, dict[str, Any]],
    trades: list[dict[str, Any]],
    symbol: str,
    fill_price: float,
    timestamp_str: str,
    shares_held: dict[str, float],
    entry_price: dict[str, float],
    entry_cost: dict[str, float],
    breakout_trailing_high: dict[str, float],
    breakout_range_at_entry: dict[str, float],
    breakout_stored_stop: dict[str, float],
    mean_reversion_target_price: dict[str, float],
    latest_mark_price: dict[str, float],
    position: dict[str, bool],
    commission: float,
    cash: float,
    extra_trade_fields: dict[str, Any] | None = None,
) -> float:
    exit_proceeds = shares_held[symbol] * fill_price
    pnl = exit_proceeds - commission - entry_cost[symbol]
    results["realized_pnl"] += pnl
    (results["winning_pnls"] if pnl > 0 else results["losing_pnls"]).append(pnl)
    symbol_stat = symbol_stats[symbol]
    symbol_stat["realized_pnl"] += pnl
    (symbol_stat["winning_pnls"] if pnl > 0 else symbol_stat["losing_pnls"]).append(pnl)
    cash += exit_proceeds - commission
    symbol_stat["cash"] += exit_proceeds - commission
    position[symbol] = False
    latest_mark_price[symbol] = fill_price
    trade_record = {
        "symbol": symbol,
        "side": "SELL",
        "price": fill_price,
        "shares": shares_held[symbol],
        "timestamp": timestamp_str,
        "pnl": pnl,
    }
    if extra_trade_fields:
        trade_record.update(extra_trade_fields)
    trades.append(trade_record)
    shares_held[symbol] = 0.0
    entry_price[symbol] = 0.0
    entry_cost[symbol] = 0.0
    breakout_trailing_high[symbol] = 0.0
    breakout_range_at_entry[symbol] = 0.0
    breakout_stored_stop[symbol] = 0.0
    mean_reversion_target_price[symbol] = 0.0
    results["symbol_trade_counts"][symbol] += 1
    results["total_trades"] += 1
    symbol_stat["total_trades"] += 1
    return cash


def _execute_buy(
    *,
    results: dict[str, Any],
    symbol_stats: dict[str, dict[str, Any]],
    trades: list[dict[str, Any]],
    symbol: str,
    fill_price: float,
    timestamp_str: str,
    position_size: float,
    commission: float,
    cash: float,
    shares_held: dict[str, float],
    entry_price: dict[str, float],
    entry_cost: dict[str, float],
    latest_mark_price: dict[str, float],
    position: dict[str, bool],
) -> float:
    total_cash_needed = position_size + commission
    if cash < total_cash_needed:
        return cash

    shares_held[symbol] = position_size / fill_price
    entry_price[symbol] = fill_price
    entry_cost[symbol] = position_size + commission
    symbol_stat = symbol_stats[symbol]
    cash -= total_cash_needed
    symbol_stat["cash"] -= total_cash_needed
    position[symbol] = True
    latest_mark_price[symbol] = fill_price
    trades.append({
        "symbol": symbol,
        "side": "BUY",
        "price": fill_price,
        "shares": shares_held[symbol],
        "timestamp": timestamp_str,
    })
    results["symbol_trade_counts"][symbol] += 1
    results["total_trades"] += 1
    symbol_stat["total_trades"] += 1
    return cash


def _parse_symbol_strategy_map(raw_value: str | None) -> dict[str, str]:
    if not raw_value:
        return {}
    parsed: dict[str, str] = {}
    for item in raw_value.split(","):
        if not item.strip():
            continue
        symbol, sep, mode = item.partition(":")
        if not sep:
            raise ValueError(
                "Use SYMBOL:MODE pairs for symbol strategy maps, e.g. AAPL:sma,MSFT:breakout"
            )
        normalized_mode = normalize_strategy_mode(mode)
        if normalized_mode not in STRATEGY_MODE_CHOICES:
            raise ValueError(
                f"Unsupported strategy mode {mode!r}. Choose from {', '.join(STRATEGY_MODE_CHOICES)}."
            )
        parsed[symbol.strip().upper()] = normalized_mode
    return parsed


def _signal_priority(
    strategy_mode: str,
    price: float,
    sma: float,
    opening_range_high: float | None,
    opening_range_low: float | None,
) -> float:
    if strategy_mode == STRATEGY_MODE_BREAKOUT and opening_range_high is not None and opening_range_high > 0:
        return (price / opening_range_high) - 1.0
    if strategy_mode == STRATEGY_MODE_MEAN_REVERSION and sma > 0:
        return (sma - price) / sma
    if sma > 0:
        return (price - sma) / sma
    return 0.0


def _is_end_of_trading_day(timestamps: list[pd.Timestamp], idx: int) -> bool:
    current_ts = pd.Timestamp(timestamps[idx])
    if current_ts.tzinfo is None:
        current_ts = current_ts.tz_localize("UTC")
    current_day = current_ts.tz_convert("America/New_York").date()
    if idx + 1 >= len(timestamps):
        return True
    next_ts = pd.Timestamp(timestamps[idx + 1])
    if next_ts.tzinfo is None:
        next_ts = next_ts.tz_localize("UTC")
    next_day = next_ts.tz_convert("America/New_York").date()
    return next_day != current_day


# ---------------------------------------------------------------------------
# Backtest preparation helpers
# ---------------------------------------------------------------------------

def _prepare_backtest_inputs(
    dataset_path: Path,
    symbols: list[str] | None,
    start_date: str | None,
    end_date: str | None,
    strategy_mode: str,
    symbol_strategy_modes: dict[str, str] | None,
    time_window_mode: str,
    threshold_mode: str,
    orb_filter_mode: str,
    breakout_exit_style: str,
    breakout_tight_stop_fraction: float,
    breakout_max_stop_pct: float,
    breakout_gap_pct_min: float,
    breakout_or_range_pct_min: float,
    mean_reversion_exit_style: str,
    mean_reversion_max_atr_percentile: float,
    mean_reversion_stop_pct: float,
    mean_reversion_trend_filter: bool,
    mean_reversion_trend_slope_filter: bool,
    ml_probability_buy: float,
    ml_probability_sell: float,
    entry_threshold_pct: float,
    atr_multiple: float,
    atr_percentile_threshold: float,
    regime_filter_enabled: bool,
) -> _BacktestInputs:
    """Normalize all mode strings, load the dataset, validate symbol strategy modes, and build Strategy objects."""
    time_window_mode = normalize_time_window_mode(time_window_mode)
    strategy_mode = normalize_strategy_mode(strategy_mode)
    threshold_mode = normalize_threshold_mode(threshold_mode)
    orb_filter_mode = normalize_orb_filter_mode(orb_filter_mode)
    breakout_exit_style = normalize_breakout_exit_style(breakout_exit_style)
    mean_reversion_exit_style = normalize_mean_reversion_exit_style(mean_reversion_exit_style)

    df, manifest, resolved_symbols = _load_filtered_dataset(dataset_path, symbols, start_date, end_date)

    normalized_symbol_modes = {
        symbol: normalize_strategy_mode(mode)
        for symbol, mode in (symbol_strategy_modes or {}).items()
    }
    invalid_modes = [mode for mode in normalized_symbol_modes.values() if mode not in STRATEGY_MODE_CHOICES]
    if invalid_modes:
        raise ValueError(
            f"Unsupported strategy mode(s): {', '.join(sorted(set(invalid_modes)))}. "
            f"Choose from {', '.join(STRATEGY_MODE_CHOICES)}."
        )

    symbol_strategies = {
        symbol: Strategy(StrategyConfig(
            strategy_mode=normalized_symbol_modes.get(symbol, strategy_mode),
            ml_probability_buy=ml_probability_buy,
            ml_probability_sell=ml_probability_sell,
            entry_threshold_pct=entry_threshold_pct,
            threshold_mode=threshold_mode,
            atr_multiple=atr_multiple,
            atr_percentile_threshold=atr_percentile_threshold,
            time_window_mode=time_window_mode,
            regime_filter_enabled=regime_filter_enabled,
            orb_filter_mode=orb_filter_mode,
            breakout_exit_style=breakout_exit_style,
            breakout_tight_stop_fraction=breakout_tight_stop_fraction,
            breakout_max_stop_pct=breakout_max_stop_pct,
            breakout_gap_pct_min=breakout_gap_pct_min,
            breakout_or_range_pct_min=breakout_or_range_pct_min,
            mean_reversion_exit_style=mean_reversion_exit_style,
            mean_reversion_max_atr_percentile=mean_reversion_max_atr_percentile,
            mean_reversion_stop_pct=mean_reversion_stop_pct,
            mean_reversion_trend_filter=mean_reversion_trend_filter,
            mean_reversion_trend_slope_filter=mean_reversion_trend_slope_filter,
        ))
        for symbol in resolved_symbols
    }

    dummy_ml = MlSignal(
        probability_up=0.5, confidence=0.0, training_rows=0,
        model_age_seconds=0.0, feature_names=(),
        buy_threshold=ml_probability_buy, sell_threshold=ml_probability_sell,
        validation_rows=0, model_name="dummy",
    )

    return _BacktestInputs(
        df=df,
        manifest=manifest,
        symbols=resolved_symbols,
        symbol_strategies=symbol_strategies,
        dummy_ml=dummy_ml,
        strategy_mode=strategy_mode,
        time_window_mode=time_window_mode,
        threshold_mode=threshold_mode,
        orb_filter_mode=orb_filter_mode,
        breakout_exit_style=breakout_exit_style,
        breakout_tight_stop_fraction=breakout_tight_stop_fraction,
        breakout_max_stop_pct=breakout_max_stop_pct,
        breakout_gap_pct_min=breakout_gap_pct_min,
        breakout_or_range_pct_min=breakout_or_range_pct_min,
        mean_reversion_exit_style=mean_reversion_exit_style,
        mean_reversion_max_atr_percentile=mean_reversion_max_atr_percentile,
        mean_reversion_stop_pct=mean_reversion_stop_pct,
        mean_reversion_trend_filter=mean_reversion_trend_filter,
        mean_reversion_trend_slope_filter=mean_reversion_trend_slope_filter,
        regime_filter_enabled=regime_filter_enabled,
        atr_multiple=atr_multiple,
        atr_percentile_threshold=atr_percentile_threshold,
        entry_threshold_pct=entry_threshold_pct,
    )


def _initialize_simulation_state(
    inputs: _BacktestInputs,
    starting_capital: float,
    position_size: float,
) -> _SimState:
    """Initialize the results dict, symbol stats, precomputed arrays, and all mutable simulation dicts."""
    symbols = inputs.symbols
    df = inputs.df
    per_symbol_starting_capital = starting_capital / max(1, len(symbols))

    results: dict[str, Any] = {
        "strategy_mode": inputs.strategy_mode,
        "symbol_strategy_modes": {s: inputs.symbol_strategies[s].config.strategy_mode for s in symbols},
        "symbols": list(symbols),
        "time_window_mode": inputs.time_window_mode,
        "regime_filter_enabled": inputs.regime_filter_enabled,
        "orb_filter_mode": inputs.orb_filter_mode,
        "breakout_exit_style": inputs.breakout_exit_style,
        "breakout_tight_stop_fraction": inputs.breakout_tight_stop_fraction,
        "mean_reversion_exit_style": inputs.mean_reversion_exit_style,
        "mean_reversion_max_atr_percentile": inputs.mean_reversion_max_atr_percentile,
        "threshold_mode": inputs.threshold_mode,
        "sma_bars": None,
        "entry_threshold_pct": inputs.entry_threshold_pct,
        "ml_probability_buy": inputs.dummy_ml.buy_threshold,
        "ml_probability_sell": inputs.dummy_ml.sell_threshold,
        "total_trades": 0,
        "realized_pnl": 0.0,
        "starting_capital": starting_capital,
        "final_equity": starting_capital,
        "total_return_pct": 0.0,
        "symbol_trade_counts": {symbol: 0 for symbol in symbols},
        "final_positions": {},
        "trades": [],
        "winning_pnls": [],
        "losing_pnls": [],
        "max_drawdown_pct": 0.0,
        "position_size": position_size,
        "atr_multiple": inputs.atr_multiple,
        "atr_percentile_threshold": inputs.atr_percentile_threshold,
        "final_cash": starting_capital,
        "skipped_trades": 0,
        "max_concurrent_positions": 0,
        "total_buy_candidates": 0,
        "competing_timestamps": 0,
        "per_symbol": {},
    }
    symbol_stats: dict[str, dict[str, Any]] = {
        symbol: {
            "realized_pnl": 0.0,
            "winning_pnls": [],
            "losing_pnls": [],
            "cash": per_symbol_starting_capital,
            "open_positions_value": 0.0,
            "peak_equity": per_symbol_starting_capital,
            "max_drawdown_pct": 0.0,
            "total_trades": 0,
        }
        for symbol in symbols
    }

    # Precompute per-symbol arrays — avoids repeated pandas overhead in the hot loop
    symbols_dfs: dict[str, pd.DataFrame] = {}
    close_arrs: dict[str, list[float]] = {}
    volume_arrs: dict[str, list[float]] = {}
    timestamp_str_arrs: dict[str, list[str]] = {}
    day_key_arrs: dict[str, list[pd.Timestamp]] = {}
    is_eod_arrs: dict[str, list[bool]] = {}
    atr_pct_arrs: dict[str, list[float | None]] = {}
    atr_percentile_arrs: dict[str, list[float | None]] = {}
    bullish_regime_arrs: dict[str, list[bool | None]] = {}
    opening_range_high_arrs: dict[str, list[float | None]] = {}
    opening_range_low_arrs: dict[str, list[float | None]] = {}

    for symbol in symbols:
        sdf = df[df["symbol"] == symbol].sort_values("timestamp").reset_index(drop=True)
        symbols_dfs[symbol] = sdf
        close_arrs[symbol] = sdf["close"].tolist()
        volume_arrs[symbol] = sdf["volume"].tolist()
        high_arrs_sym = sdf["high"].tolist()
        low_arrs_sym = sdf["low"].tolist()
        timestamp_arrs_sym = [pd.Timestamp(ts) for ts in sdf["timestamp"].tolist()]
        timestamp_str_arrs[symbol] = [str(ts) for ts in timestamp_arrs_sym]
        day_key_arrs[symbol] = [ts.tz_convert("America/New_York").normalize() for ts in timestamp_arrs_sym]
        is_eod_arrs[symbol] = [
            idx + 1 >= len(day_key_arrs[symbol]) or day_key_arrs[symbol][idx + 1] != day_key_arrs[symbol][idx]
            for idx in range(len(day_key_arrs[symbol]))
        ]
        atr_pct_arrs[symbol] = calculate_atr_pct_values(high_arrs_sym, low_arrs_sym, close_arrs[symbol])
        atr_percentile_arrs[symbol] = calculate_atr_percentile_series(
            timestamp_arrs_sym, high_arrs_sym, low_arrs_sym, close_arrs[symbol],
        )
        bullish_regime_arrs[symbol] = calculate_hourly_regime_series(
            timestamp_arrs_sym, close_arrs[symbol], source_bar_minutes=15, sma_period=REGIME_SMA_PERIOD,
        )
        opening_range_high_arrs[symbol], opening_range_low_arrs[symbol] = calculate_opening_range_series(
            timestamp_arrs_sym, high_arrs_sym, low_arrs_sym,
        )

    return _SimState(
        symbols_dfs=symbols_dfs,
        close_arrs=close_arrs,
        volume_arrs=volume_arrs,
        timestamp_str_arrs=timestamp_str_arrs,
        day_key_arrs=day_key_arrs,
        is_eod_arrs=is_eod_arrs,
        atr_pct_arrs=atr_pct_arrs,
        atr_percentile_arrs=atr_percentile_arrs,
        bullish_regime_arrs=bullish_regime_arrs,
        opening_range_high_arrs=opening_range_high_arrs,
        opening_range_low_arrs=opening_range_low_arrs,
        pointers={s: 0 for s in symbols},
        position={s: False for s in symbols},
        entry_price={s: 0.0 for s in symbols},
        entry_cost={s: 0.0 for s in symbols},
        shares_held={s: 0.0 for s in symbols},
        latest_mark_price={s: 0.0 for s in symbols},
        signal_reference_price={s: 0.0 for s in symbols},
        breakout_trailing_high={s: 0.0 for s in symbols},
        breakout_range_at_entry={s: 0.0 for s in symbols},
        breakout_stored_stop={s: 0.0 for s in symbols},
        mean_reversion_target_price={s: 0.0 for s in symbols},
        pending_buys={},
        pending_sells=set(),
        orb_entry_taken={s: False for s in symbols},
        orb_last_day={s: None for s in symbols},
        breakout_day_gap_pct={s: 0.0 for s in symbols},
        ml_signals={s: [] for s in symbols},
        results=results,
        symbol_stats=symbol_stats,
        trades=[],
    )


def _compute_ml_signal(
    symbol: str,
    p: int,
    effective_strategy_mode: str,
    ml_signals: dict[str, list[MlSignal | None]],
    dummy_ml: MlSignal,
) -> MlSignal:
    """Return a precomputed ML signal for the current bar."""
    if effective_strategy_mode not in (STRATEGY_MODE_HYBRID, STRATEGY_MODE_ML):
        return dummy_ml

    symbol_signals = ml_signals.get(symbol)
    if symbol_signals is None or p >= len(symbol_signals):
        return dummy_ml
    return symbol_signals[p] or dummy_ml


def _print_strategy_mode_comparison(results_list: list[dict]) -> None:
    if not results_list:
        return

    comparison_modes = [STRATEGY_MODE_SMA, STRATEGY_MODE_ML, STRATEGY_MODE_HYBRID]
    mode_rows = [
        result for result in results_list
        if result.get("strategy_mode") in comparison_modes
    ]
    if not mode_rows:
        return

    best_by_mode: dict[str, dict] = {}
    for mode in comparison_modes:
        matches = [result for result in mode_rows if result.get("strategy_mode") == mode]
        if matches:
            best_by_mode[mode] = max(matches, key=lambda row: row["total_return_pct"])

    if len(best_by_mode) < 2:
        return

    print("\nStrategy Mode Comparison:")
    print(
        f"{'Mode':<10}  {'Return':>9}  {'Win%':>6}  {'PF':>6}  {'Trades':>8}"
    )
    print("-" * 49)
    for mode in comparison_modes:
        row = best_by_mode.get(mode)
        if row is None:
            continue
        print(
            f"{mode:<10}  "
            f"{row['total_return_pct']:>8.2f}%  "
            f"{row['win_rate']:>5.1f}%  "
            f"{row['profit_factor']:>6.2f}  "
            f"{row['total_trades']:>8}"
        )


def _handle_pending_orders_at_time(
    rows_at_time: dict[str, Any],
    state: _SimState,
    slippage: float,
    commission: float,
    position_size: float,
    sma_bars: int,
    symbol_strategies: dict[str, Strategy],
    cash: float,
) -> tuple[float, int]:
    """Execute any pending sell and buy orders at the open of the current bar. Returns (cash, skipped_buy_count)."""
    skipped = 0

    sell_actions = [(symbol, rows_at_time[symbol]) for symbol in rows_at_time if symbol in state.pending_sells]
    for symbol, row in sell_actions:
        fill_price = float(row["open"]) - slippage
        cash = _execute_sell(
            results=state.results,
            symbol_stats=state.symbol_stats,
            trades=state.trades,
            symbol=symbol,
            fill_price=fill_price,
            timestamp_str=state.timestamp_str_arrs[symbol][state.pointers[symbol]],
            shares_held=state.shares_held,
            entry_price=state.entry_price,
            entry_cost=state.entry_cost,
            breakout_trailing_high=state.breakout_trailing_high,
            breakout_range_at_entry=state.breakout_range_at_entry,
            breakout_stored_stop=state.breakout_stored_stop,
            mean_reversion_target_price=state.mean_reversion_target_price,
            latest_mark_price=state.latest_mark_price,
            position=state.position,
            commission=commission,
            cash=cash,
        )
        state.pending_sells.discard(symbol)

    buy_candidates = [
        (state.pending_buys[symbol][0], symbol, rows_at_time[symbol])
        for symbol in rows_at_time
        if symbol in state.pending_buys
    ]
    buy_candidates.sort(reverse=True)
    state.results["total_buy_candidates"] += len(buy_candidates)
    if len(buy_candidates) > 1:
        state.results["competing_timestamps"] += 1

    for _, symbol, row in buy_candidates:
        fill_price = float(row["open"]) + slippage
        prior_cash = cash
        cash = _execute_buy(
            results=state.results,
            symbol_stats=state.symbol_stats,
            trades=state.trades,
            symbol=symbol,
            fill_price=fill_price,
            timestamp_str=state.timestamp_str_arrs[symbol][state.pointers[symbol]],
            position_size=position_size,
            commission=commission,
            cash=cash,
            shares_held=state.shares_held,
            entry_price=state.entry_price,
            entry_cost=state.entry_cost,
            latest_mark_price=state.latest_mark_price,
            position=state.position,
        )
        if math.isclose(cash, prior_cash):
            skipped += 1
            state.pending_buys.pop(symbol, None)
            continue
        state.signal_reference_price[symbol] = fill_price
        p = state.pointers[symbol]
        symbol_strategy = symbol_strategies[symbol]
        if symbol_strategy.config.strategy_mode == STRATEGY_MODE_BREAKOUT:
            state.breakout_trailing_high[symbol] = fill_price
            or_low = state.opening_range_low_arrs[symbol][p] or fill_price
            state.breakout_range_at_entry[symbol] = max(
                0.0,
                (state.opening_range_high_arrs[symbol][p] or fill_price) - or_low,
            )
            state.breakout_stored_stop[symbol] = get_capped_breakout_stop_price(
                fill_price,
                or_low,
                symbol_strategy.config.breakout_max_stop_pct,
            )
            logger.info(
                "breakout entry symbol=%s entry=%.4f or_low=%.4f capped_stop=%.4f dist_pct=%.2f%%",
                symbol, fill_price, or_low, state.breakout_stored_stop[symbol],
                (fill_price - state.breakout_stored_stop[symbol]) / fill_price * 100,
            )
        elif symbol_strategy.config.strategy_mode == STRATEGY_MODE_MEAN_REVERSION:
            current_sma = (
                _mean(state.close_arrs[symbol][p - sma_bars + 1: p + 1])
                if p >= sma_bars - 1
                else fill_price
            )
            state.mean_reversion_target_price[symbol] = (fill_price + current_sma) / 2.0
        elif symbol_strategy.config.strategy_mode == STRATEGY_MODE_ORB:
            state.orb_entry_taken[symbol] = True
        state.results["max_concurrent_positions"] = max(
            state.results["max_concurrent_positions"], sum(state.position.values())
        )
        state.pending_buys.pop(symbol, None)

    return cash, skipped


def _process_bar(
    symbol: str,
    row: Any,
    p: int,
    state: _SimState,
    sma_bars: int,
    time_window_mode: str,
    slippage: float,
    commission: float,
    ml_lookback_bars: int,
    ml_retrain_every_bars: int,
    ml_probability_buy: float,
    ml_probability_sell: float,
    symbol_strategy: Strategy,
    dummy_ml: MlSignal,
    cash: float,
) -> float:
    """Evaluate one completed bar for a single symbol: compute indicators, get a signal, update pending queues,
    and execute any EOD force-close. Returns updated cash."""
    closes = state.close_arrs[symbol]
    volumes = state.volume_arrs[symbol]
    effective_strategy_mode = symbol_strategy.config.strategy_mode
    state.latest_mark_price[symbol] = float(row["close"])

    min_history_bars = 2 if effective_strategy_mode in (STRATEGY_MODE_BREAKOUT, STRATEGY_MODE_ORB) else sma_bars
    if p < min_history_bars - 1:
        return cash

    # Reset ORB one-trade-per-day flag at the start of each new trading day.
    current_day = state.day_key_arrs[symbol][p]
    if current_day != state.orb_last_day[symbol]:
        state.orb_entry_taken[symbol] = False
        state.orb_last_day[symbol] = current_day
        if p > 0:
            prev_close = closes[p - 1]
            state.breakout_day_gap_pct[symbol] = (
                (float(row["open"]) - prev_close) / prev_close if prev_close > 0 else 0.0
            )
        else:
            state.breakout_day_gap_pct[symbol] = 0.0

    sma = _mean(closes[p - sma_bars + 1: p + 1]) if p >= sma_bars - 1 else closes[p]
    trend_sma = _mean(closes[p - 49: p + 1]) if p >= 49 else None
    # Slope = SMA_50 now minus SMA_50 five bars ago (75 min on 15-min bars).
    # Positive means the 50-bar average is rising; negative means it's falling.
    trend_sma_slope = (
        trend_sma - _mean(closes[p - 54: p - 4])
        if p >= 54 and trend_sma is not None
        else None
    )
    price = closes[p]

    if state.position[symbol] and effective_strategy_mode == STRATEGY_MODE_BREAKOUT:
        state.breakout_trailing_high[symbol] = max(state.breakout_trailing_high[symbol], price)

    recent_volumes = volumes[max(0, p - 19): p + 1]
    avg_volume = _mean(recent_volumes) if recent_volumes else 0.0
    volume_ratio = (volumes[p] / avg_volume) if avg_volume > 0 else None

    recent_atr_pct_values = [v for v in state.atr_pct_arrs[symbol][max(0, p - 19): p + 1] if v is not None]
    avg_atr_pct = _mean(recent_atr_pct_values) if recent_atr_pct_values else None
    atr_pct = state.atr_pct_arrs[symbol][p]
    volatility_ratio = (
        atr_pct / avg_atr_pct
        if atr_pct is not None and avg_atr_pct is not None and avg_atr_pct > 0
        else None
    )

    ml_signal = _compute_ml_signal(
        symbol,
        p,
        effective_strategy_mode,
        state.ml_signals,
        dummy_ml,
    )

    next_entry_timestamp = (
        pd.Timestamp(state.symbols_dfs[symbol].iloc[p + 1]["timestamp"])
        if p + 1 < len(state.symbols_dfs[symbol])
        else None
    )

    exit_style = symbol_strategy.config.breakout_exit_style
    trailing_stop_price: float | None = None
    if state.position[symbol]:
        if exit_style == BREAKOUT_EXIT_TRAILING_HALF_RANGE:
            trailing_stop_price = state.breakout_trailing_high[symbol] - (0.5 * state.breakout_range_at_entry[symbol])
        elif exit_style == BREAKOUT_EXIT_TRAILING_FULL_RANGE:
            trailing_stop_price = state.breakout_trailing_high[symbol] - state.breakout_range_at_entry[symbol]

    action = symbol_strategy.decide_action(
        price,
        sma,
        ml_signal,
        state.position[symbol],
        atr_pct,
        state.atr_percentile_arrs[symbol][p],
        time_window_open=(
            True if next_entry_timestamp is None
            else is_entry_window_open(next_entry_timestamp, time_window_mode)
        ),
        bullish_regime=state.bullish_regime_arrs[symbol][p],
        opening_range_high=state.opening_range_high_arrs[symbol][p],
        opening_range_low=state.opening_range_low_arrs[symbol][p],
        position_entry_price=state.entry_price[symbol] if state.position[symbol] else None,
        volume_ratio=volume_ratio,
        volatility_ratio=volatility_ratio,
        trailing_stop_price=trailing_stop_price,
        mean_reversion_target_price=state.mean_reversion_target_price[symbol] if state.position[symbol] else None,
        breakout_already_taken=state.orb_entry_taken[symbol],
        effective_stop_price=state.breakout_stored_stop[symbol] if state.position[symbol] and state.breakout_stored_stop[symbol] > 0 else None,
        gap_pct=state.breakout_day_gap_pct[symbol],
        trend_sma=trend_sma,
        trend_sma_slope=trend_sma_slope,
    )

    has_next_bar = p + 1 < len(state.symbols_dfs[symbol])
    if has_next_bar and action == "BUY" and not state.position[symbol]:
        state.pending_buys[symbol] = (
            _signal_priority(
                effective_strategy_mode, price, sma,
                state.opening_range_high_arrs[symbol][p],
                state.opening_range_low_arrs[symbol][p],
            ),
            row["timestamp"],
        )
    elif has_next_bar and action == "SELL" and state.position[symbol]:
        state.pending_sells.add(symbol)

    if state.position[symbol] and state.is_eod_arrs[symbol][p]:
        if (
            effective_strategy_mode == STRATEGY_MODE_BREAKOUT
            and symbol_strategy.config.breakout_exit_style == BREAKOUT_EXIT_EOD_ONLY
        ) or (
            effective_strategy_mode == STRATEGY_MODE_MEAN_REVERSION
            and symbol_strategy.config.mean_reversion_exit_style == MEAN_REVERSION_EXIT_EOD
        ) or effective_strategy_mode == STRATEGY_MODE_ORB:
            fill_price = float(row["close"]) - slippage
            cash = _execute_sell(
                results=state.results,
                symbol_stats=state.symbol_stats,
                trades=state.trades,
                symbol=symbol,
                fill_price=fill_price,
                timestamp_str=state.timestamp_str_arrs[symbol][p],
                shares_held=state.shares_held,
                entry_price=state.entry_price,
                entry_cost=state.entry_cost,
                breakout_trailing_high=state.breakout_trailing_high,
                breakout_range_at_entry=state.breakout_range_at_entry,
                breakout_stored_stop=state.breakout_stored_stop,
                mean_reversion_target_price=state.mean_reversion_target_price,
                latest_mark_price=state.latest_mark_price,
                position=state.position,
                commission=commission,
                cash=cash,
                extra_trade_fields={"eod_exit": True},
            )
            state.pending_sells.discard(symbol)

    return cash


def _finalize_simulation(
    symbols: list[str],
    state: _SimState,
    slippage: float,
    commission: float,
    cash: float,
    peak_equity: float,
    max_drawdown_pct: float,
    portfolio_equity_by_day: dict[pd.Timestamp, float],
    per_symbol_equity_by_day: dict[str, dict[pd.Timestamp, float]],
) -> tuple[float, float, float, float]:
    """Force-close any remaining open positions at the last bar and run a final mark-to-market pass.
    Returns (cash, open_positions_value, peak_equity, max_drawdown_pct)."""
    for symbol in symbols:
        if state.position[symbol]:
            last_row = state.symbols_dfs[symbol].iloc[-1]
            fill_price = float(last_row["close"]) - slippage
            cash = _execute_sell(
                results=state.results,
                symbol_stats=state.symbol_stats,
                trades=state.trades,
                symbol=symbol,
                fill_price=fill_price,
                timestamp_str=state.timestamp_str_arrs[symbol][-1],
                shares_held=state.shares_held,
                entry_price=state.entry_price,
                entry_cost=state.entry_cost,
                breakout_trailing_high=state.breakout_trailing_high,
                breakout_range_at_entry=state.breakout_range_at_entry,
                breakout_stored_stop=state.breakout_stored_stop,
                mean_reversion_target_price=state.mean_reversion_target_price,
                latest_mark_price=state.latest_mark_price,
                position=state.position,
                commission=commission,
                cash=cash,
                extra_trade_fields={"forced_close": True},
            )

    open_positions_value, peak_equity, max_drawdown_pct = _update_mark_to_market(
        symbols, state.position, state.shares_held, state.latest_mark_price,
        cash, peak_equity, max_drawdown_pct, state.symbol_stats,
    )

    if any(state.day_key_arrs[s] for s in symbols):
        final_day_key = max(max(dk) for dk in state.day_key_arrs.values())
        portfolio_equity_by_day[final_day_key] = cash + open_positions_value
        for symbol in symbols:
            per_symbol_equity_by_day[symbol][final_day_key] = (
                state.symbol_stats[symbol]["cash"] + state.symbol_stats[symbol]["open_positions_value"]
            )

    return cash, open_positions_value, peak_equity, max_drawdown_pct


def _build_final_results(
    inputs: _BacktestInputs,
    state: _SimState,
    starting_capital: float,
    position_size: float,
    per_symbol_starting_capital: float,
    cash: float,
    open_positions_value: float,
    max_drawdown_pct: float,
    skipped_trades: int,
    portfolio_equity_by_day: dict[pd.Timestamp, float],
    per_symbol_equity_by_day: dict[str, dict[pd.Timestamp, float]],
) -> dict:
    """Assemble the final results dict with portfolio and per-symbol performance summaries."""
    results = state.results
    results["trades"].extend(state.trades)
    results.update(
        _build_performance_summary(
            starting_capital=starting_capital,
            final_equity=cash + open_positions_value,
            max_drawdown_pct=max_drawdown_pct,
            total_trades=results["total_trades"],
            winning_pnls=results["winning_pnls"],
            losing_pnls=results["losing_pnls"],
            realized_pnl=results["realized_pnl"],
            final_cash=cash,
            position_size=position_size,
            daily_equity_by_day=portfolio_equity_by_day,
        )
    )
    results["skipped_trades"] = skipped_trades
    results["per_symbol"] = {
        symbol: {
            "symbol": symbol,
            **_build_performance_summary(
                starting_capital=per_symbol_starting_capital,
                final_equity=stats["cash"] + stats["open_positions_value"],
                max_drawdown_pct=stats["max_drawdown_pct"],
                total_trades=stats["total_trades"],
                winning_pnls=stats["winning_pnls"],
                losing_pnls=stats["losing_pnls"],
                realized_pnl=stats["realized_pnl"],
                final_cash=stats["cash"],
                position_size=position_size,
                daily_equity_by_day=per_symbol_equity_by_day[symbol],
            ),
        }
        for symbol, stats in state.symbol_stats.items()
    }
    total_realized_pnl = results["realized_pnl"]
    for symbol_result in results["per_symbol"].values():
        symbol_result["symbol_contribution_pct"] = (
            symbol_result["realized_pnl"] / total_realized_pnl * 100
            if abs(total_realized_pnl) > 1e-12
            else 0.0
        )
    return results


# ---------------------------------------------------------------------------
# Core backtest loop
# ---------------------------------------------------------------------------

def run_backtest(
    dataset_path: Path,
    symbols: list[str] | None = None,
    sma_bars: int = DEFAULT_SMA_BARS,
    commission: float = 0.01,
    slippage: float = 0.05,
    entry_threshold_pct: float = DEFAULT_ENTRY_THRESHOLD_PCT,
    threshold_mode: str = THRESHOLD_MODE_STATIC_PCT,
    atr_multiple: float = DEFAULT_ATR_MULTIPLE,
    atr_percentile_threshold: float = DEFAULT_DISABLED_ATR_PERCENTILE_THRESHOLD,
    time_window_mode: str = TIME_WINDOW_FULL_DAY,
    regime_filter_enabled: bool = False,
    orb_filter_mode: str = ORB_FILTER_NONE,
    breakout_exit_style: str = BREAKOUT_EXIT_TARGET_1X_STOP_LOW,
    breakout_tight_stop_fraction: float = 0.5,
    breakout_max_stop_pct: float = 0.03,
    breakout_gap_pct_min: float = 0.0,
    breakout_or_range_pct_min: float = 0.0,
    mean_reversion_exit_style: str = MEAN_REVERSION_EXIT_SMA,
    mean_reversion_max_atr_percentile: float = 0.0,
    mean_reversion_stop_pct: float = 0.0,
    mean_reversion_trend_filter: bool = False,
    mean_reversion_trend_slope_filter: bool = False,
    starting_capital: float = DEFAULT_STARTING_CAPITAL,
    position_size: float = DEFAULT_POSITION_SIZE,
    start_date: str | None = None,
    end_date: str | None = None,
    strategy_mode: str = STRATEGY_MODE_HYBRID,
    symbol_strategy_modes: dict[str, str] | None = None,
    ml_lookback_bars: int = DEFAULT_ML_LOOKBACK_BARS,
    ml_retrain_every_bars: int = DEFAULT_ML_RETRAIN_EVERY_BARS,
    ml_probability_buy: float = DEFAULT_ML_PROBABILITY_BUY,
    ml_probability_sell: float = DEFAULT_ML_PROBABILITY_SELL,
) -> dict:
    total_start = perf_counter()

    load_start = perf_counter()
    inputs = _prepare_backtest_inputs(
        dataset_path=dataset_path,
        symbols=symbols,
        start_date=start_date,
        end_date=end_date,
        strategy_mode=strategy_mode,
        symbol_strategy_modes=symbol_strategy_modes,
        time_window_mode=time_window_mode,
        threshold_mode=threshold_mode,
        orb_filter_mode=orb_filter_mode,
        breakout_exit_style=breakout_exit_style,
        breakout_tight_stop_fraction=breakout_tight_stop_fraction,
        breakout_max_stop_pct=breakout_max_stop_pct,
        breakout_gap_pct_min=breakout_gap_pct_min,
        breakout_or_range_pct_min=breakout_or_range_pct_min,
        mean_reversion_exit_style=mean_reversion_exit_style,
        mean_reversion_max_atr_percentile=mean_reversion_max_atr_percentile,
        mean_reversion_stop_pct=mean_reversion_stop_pct,
        mean_reversion_trend_filter=mean_reversion_trend_filter,
        mean_reversion_trend_slope_filter=mean_reversion_trend_slope_filter,
        ml_probability_buy=ml_probability_buy,
        ml_probability_sell=ml_probability_sell,
        entry_threshold_pct=entry_threshold_pct,
        atr_multiple=atr_multiple,
        atr_percentile_threshold=atr_percentile_threshold,
        regime_filter_enabled=regime_filter_enabled,
    )
    load_seconds = perf_counter() - load_start

    prep_start = perf_counter()
    state = _initialize_simulation_state(inputs, starting_capital, position_size)
    prep_seconds = perf_counter() - prep_start

    uses_ml = any(
        strategy.config.strategy_mode in (STRATEGY_MODE_ML, STRATEGY_MODE_HYBRID)
        for strategy in inputs.symbol_strategies.values()
    )
    model_load_start = perf_counter()
    offline_model_payload = _load_offline_model_payload() if uses_ml else None
    model_load_seconds = perf_counter() - model_load_start

    ml_precompute_start = perf_counter()
    state.ml_signals = _precompute_ml_signals(
        state,
        inputs.symbol_strategies,
        offline_model_payload,
        ml_probability_buy,
        ml_probability_sell,
    )
    ml_precompute_seconds = perf_counter() - ml_precompute_start
    ml_signal_summary = _summarize_ml_signals(state.ml_signals, inputs.symbol_strategies)
    for symbol, summary in ml_signal_summary.items():
        logger.info(
            "ML precompute symbol=%s count=%s min=%.3f max=%.3f mean=%.3f buy_thr=%.3f sell_thr=%.3f above_buy=%s below_sell=%s",
            symbol,
            summary["count"],
            summary["min_prob"] if summary["min_prob"] is not None else float("nan"),
            summary["max_prob"] if summary["max_prob"] is not None else float("nan"),
            summary["mean_prob"] if summary["mean_prob"] is not None else float("nan"),
            summary["buy_threshold"] if summary["buy_threshold"] is not None else float("nan"),
            summary["sell_threshold"] if summary["sell_threshold"] is not None else float("nan"),
            summary["above_buy_count"],
            summary["below_sell_count"],
        )

    cash = starting_capital
    skipped_trades = 0
    open_positions_value = 0.0
    peak_equity = starting_capital
    max_drawdown_pct = 0.0
    per_symbol_starting_capital = starting_capital / max(1, len(inputs.symbols))
    portfolio_equity_by_day: dict[pd.Timestamp, float] = {}
    per_symbol_equity_by_day: dict[str, dict[pd.Timestamp, float]] = {s: {} for s in inputs.symbols}

    simulate_start = perf_counter()
    while any(state.pointers[s] < len(state.symbols_dfs[s]) for s in inputs.symbols):
        current_rows = {
            s: state.symbols_dfs[s].iloc[state.pointers[s]]
            for s in inputs.symbols
            if state.pointers[s] < len(state.symbols_dfs[s])
        }
        next_times = [row["timestamp"] for row in current_rows.values()]
        if not next_times:
            break
        current_time = min(next_times)
        rows_at_time = {
            symbol: row
            for symbol, row in current_rows.items()
            if row["timestamp"] == current_time
        }

        # Signals are generated from the prior completed bar and execute on this bar's open.
        cash, skipped_at_time = _handle_pending_orders_at_time(
            rows_at_time, state, slippage, commission, position_size, sma_bars,
            inputs.symbol_strategies, cash,
        )
        skipped_trades += skipped_at_time

        for symbol in inputs.symbols:
            row = rows_at_time.get(symbol)
            if row is None:
                continue
            p = state.pointers[symbol]
            cash = _process_bar(
                symbol=symbol,
                row=row,
                p=p,
                state=state,
                sma_bars=sma_bars,
                time_window_mode=inputs.time_window_mode,
                slippage=slippage,
                commission=commission,
                ml_lookback_bars=ml_lookback_bars,
                ml_retrain_every_bars=ml_retrain_every_bars,
                ml_probability_buy=ml_probability_buy,
                ml_probability_sell=ml_probability_sell,
                symbol_strategy=inputs.symbol_strategies[symbol],
                dummy_ml=inputs.dummy_ml,
                cash=cash,
            )
            state.pointers[symbol] += 1

        open_positions_value, peak_equity, max_drawdown_pct = _update_mark_to_market(
            inputs.symbols, state.position, state.shares_held, state.latest_mark_price,
            cash, peak_equity, max_drawdown_pct, state.symbol_stats,
        )
        sample_symbol = next(iter(rows_at_time))
        day_key = state.day_key_arrs[sample_symbol][rows_at_time[sample_symbol].name]
        portfolio_equity_by_day[day_key] = cash + open_positions_value
        for symbol in inputs.symbols:
            per_symbol_equity_by_day[symbol][day_key] = (
                state.symbol_stats[symbol]["cash"] + state.symbol_stats[symbol]["open_positions_value"]
            )
    simulate_seconds = perf_counter() - simulate_start

    finalize_start = perf_counter()
    cash, open_positions_value, peak_equity, max_drawdown_pct = _finalize_simulation(
        inputs.symbols, state, slippage, commission, cash,
        peak_equity, max_drawdown_pct,
        portfolio_equity_by_day, per_symbol_equity_by_day,
    )

    results = _build_final_results(
        inputs=inputs,
        state=state,
        starting_capital=starting_capital,
        position_size=position_size,
        per_symbol_starting_capital=per_symbol_starting_capital,
        cash=cash,
        open_positions_value=open_positions_value,
        max_drawdown_pct=max_drawdown_pct,
        skipped_trades=skipped_trades,
        portfolio_equity_by_day=portfolio_equity_by_day,
        per_symbol_equity_by_day=per_symbol_equity_by_day,
    )
    results["timing_seconds"] = {
        "load": load_seconds,
        "precompute": prep_seconds,
        "model_load": model_load_seconds,
        "ml_precompute": ml_precompute_seconds,
        "simulate": simulate_seconds,
        "finalize": perf_counter() - finalize_start,
        "total": perf_counter() - total_start,
    }
    results["ml_signal_summary"] = ml_signal_summary
    timing = results["timing_seconds"]
    print(
        "[timing] "
        f"load={timing['load']:.3f}s "
        f"precompute={timing['precompute']:.3f}s "
        f"model_load={timing['model_load']:.3f}s "
        f"ml_precompute={timing['ml_precompute']:.3f}s "
        f"simulate={timing['simulate']:.3f}s "
        f"finalize={timing['finalize']:.3f}s "
        f"total={timing['total']:.3f}s"
    )
    results["sma_bars"] = sma_bars
    return results


# ---------------------------------------------------------------------------
# CLI helpers
# ---------------------------------------------------------------------------

def extract_clean_columns(result_dict: dict) -> dict:
    clean_cols = [
        "regime", "start_date", "end_date",
        "strategy_mode", "time_window_mode", "regime_filter_enabled", "threshold_mode",
        "entry_threshold_pct", "atr_multiple", "atr_percentile_threshold", "sma_bars", "orb_filter_mode",
        "breakout_exit_style", "breakout_tight_stop_fraction", "mean_reversion_exit_style", "mean_reversion_max_atr_percentile",
        "ml_probability_buy", "ml_probability_sell",
        "total_return_pct", "max_drawdown_pct",
        "total_trades", "total_buy_candidates", "skipped_trades",
        "competing_timestamps", "max_concurrent_positions",
        "win_rate", "avg_winning_trade", "avg_losing_trade", "expectancy", "max_loss_trade",
        "avg_return_per_trade_pct", "avg_move_captured_pct", "profit_factor",
        "sharpe_ratio", "trades_per_day", "pnl_stability",
        "realized_pnl", "final_equity", "final_cash",
        "starting_capital", "position_size",
        "timing_load_seconds", "timing_precompute_seconds", "timing_model_load_seconds",
        "timing_ml_precompute_seconds", "timing_simulate_seconds", "timing_finalize_seconds", "timing_total_seconds",
    ]
    cleaned = {k: result_dict[k] for k in clean_cols if k in result_dict}
    timing = result_dict.get("timing_seconds")
    if timing:
        cleaned.update({
            "timing_load_seconds": timing.get("load"),
            "timing_precompute_seconds": timing.get("precompute"),
            "timing_model_load_seconds": timing.get("model_load"),
            "timing_ml_precompute_seconds": timing.get("ml_precompute"),
            "timing_simulate_seconds": timing.get("simulate"),
            "timing_finalize_seconds": timing.get("finalize"),
            "timing_total_seconds": timing.get("total"),
        })
    return cleaned


def _result_grid_dataframe(results_list: list[dict]) -> pd.DataFrame:
    return pd.DataFrame([extract_clean_columns(result) for result in results_list])


def _build_per_symbol_rows(result_dict: dict) -> list[dict]:
    base_cols = {
        "regime": result_dict.get("regime"),
        "start_date": result_dict.get("start_date"),
        "end_date": result_dict.get("end_date"),
        "strategy_mode": result_dict.get("strategy_mode"),
        "time_window_mode": result_dict.get("time_window_mode"),
        "regime_filter_enabled": result_dict.get("regime_filter_enabled"),
        "threshold_mode": result_dict.get("threshold_mode"),
        "sma_bars": result_dict.get("sma_bars"),
        "entry_threshold_pct": result_dict.get("entry_threshold_pct"),
        "atr_multiple": result_dict.get("atr_multiple"),
        "atr_percentile_threshold": result_dict.get("atr_percentile_threshold"),
        "orb_filter_mode": result_dict.get("orb_filter_mode"),
        "breakout_exit_style": result_dict.get("breakout_exit_style"),
        "breakout_tight_stop_fraction": result_dict.get("breakout_tight_stop_fraction"),
        "mean_reversion_exit_style": result_dict.get("mean_reversion_exit_style"),
        "mean_reversion_max_atr_percentile": result_dict.get("mean_reversion_max_atr_percentile"),
    }
    rows = [{
        **base_cols,
        "scope": "portfolio",
        "symbol": "COMBINED",
        "realized_pnl": result_dict["realized_pnl"],
        "total_return_pct": result_dict["total_return_pct"],
        "max_drawdown_pct": result_dict["max_drawdown_pct"],
        "win_rate": result_dict["win_rate"],
        "avg_winning_trade": result_dict["avg_winning_trade"],
        "avg_losing_trade": result_dict["avg_losing_trade"],
        "expectancy": result_dict["expectancy"],
        "max_loss_trade": result_dict["max_loss_trade"],
        "avg_return_per_trade_pct": result_dict["avg_return_per_trade_pct"],
        "avg_move_captured_pct": result_dict["avg_move_captured_pct"],
        "profit_factor": result_dict["profit_factor"],
        "sharpe_ratio": result_dict["sharpe_ratio"],
        "trades_per_day": result_dict["trades_per_day"],
        "pnl_stability": result_dict["pnl_stability"],
        "total_trades": result_dict["total_trades"],
    }]
    for symbol, symbol_result in result_dict.get("per_symbol", {}).items():
        rows.append({
            **base_cols,
            "scope": "symbol",
            "symbol": symbol,
            "symbol_strategy_mode": result_dict.get("symbol_strategy_modes", {}).get(symbol, result_dict.get("strategy_mode")),
            "realized_pnl": symbol_result["realized_pnl"],
            "total_return_pct": symbol_result["total_return_pct"],
            "max_drawdown_pct": symbol_result["max_drawdown_pct"],
            "win_rate": symbol_result["win_rate"],
            "avg_winning_trade": symbol_result["avg_winning_trade"],
            "avg_losing_trade": symbol_result["avg_losing_trade"],
            "expectancy": symbol_result["expectancy"],
            "max_loss_trade": symbol_result["max_loss_trade"],
            "avg_return_per_trade_pct": symbol_result["avg_return_per_trade_pct"],
            "avg_move_captured_pct": symbol_result["avg_move_captured_pct"],
            "profit_factor": symbol_result["profit_factor"],
            "sharpe_ratio": symbol_result["sharpe_ratio"],
            "trades_per_day": symbol_result["trades_per_day"],
            "pnl_stability": symbol_result["pnl_stability"],
            "total_trades": symbol_result["total_trades"],
            "symbol_contribution_pct": symbol_result.get("symbol_contribution_pct", 0.0),
        })
    return rows


def _per_symbol_dataframe(results_list: list[dict]) -> pd.DataFrame:
    rows: list[dict] = []
    for result in results_list:
        rows.extend(_build_per_symbol_rows(result))
    return pd.DataFrame(rows)


def _write_per_symbol_csv(results_list: list[dict], output_csv: str) -> Path:
    output_path = Path(output_csv)
    suffix = output_path.suffix or ".csv"
    per_symbol_path = output_path.with_name(f"{output_path.stem}_per_symbol{suffix}")
    _per_symbol_dataframe(results_list).to_csv(per_symbol_path, index=False)
    return per_symbol_path


def _build_robustness_dataframe(results_list: list[dict]) -> pd.DataFrame:
    per_symbol_df = _per_symbol_dataframe(results_list)
    if per_symbol_df.empty:
        return pd.DataFrame()

    symbol_rows = per_symbol_df[per_symbol_df["scope"] == "symbol"].copy()
    if symbol_rows.empty:
        return pd.DataFrame()

    grouping_cols = [
        "strategy_mode",
        "threshold_mode",
        "entry_threshold_pct",
        "atr_multiple",
        "atr_percentile_threshold",
        "sma_bars",
        "regime_filter_enabled",
    ]
    total_symbols = symbol_rows["symbol"].nunique()
    total_windows = symbol_rows["time_window_mode"].nunique()

    robust_rows: list[dict[str, Any]] = []
    for config_key, config_rows in symbol_rows.groupby(grouping_cols, dropna=False):
        config_df = config_rows.copy()
        symbol_coverage = config_df["symbol"].nunique()
        window_coverage = config_df["time_window_mode"].nunique()
        if symbol_coverage < total_symbols or window_coverage < total_windows:
            continue

        robust_rows.append({
            "strategy_mode": config_key[0],
            "threshold_mode": config_key[1],
            "entry_threshold_pct": config_key[2],
            "atr_multiple": config_key[3],
            "atr_percentile_threshold": config_key[4],
            "sma_bars": config_key[5],
            "regime_filter_enabled": config_key[6],
            "coverage_symbols": symbol_coverage,
            "coverage_time_windows": window_coverage,
            "avg_net_pnl": config_df["realized_pnl"].mean(),
            "avg_sharpe": config_df["sharpe_ratio"].mean(),
            "avg_win_rate": config_df["win_rate"].mean(),
            "avg_profit_factor": config_df["profit_factor"].mean(),
            "avg_trades_per_day": config_df["trades_per_day"].mean(),
            "avg_return_per_trade_pct": config_df["avg_return_per_trade_pct"].mean(),
            "avg_pnl_stability": config_df["pnl_stability"].mean(),
            "worst_max_drawdown_pct": config_df["max_drawdown_pct"].max(),
            "net_pnl_std": config_df["realized_pnl"].std(ddof=0) if len(config_df) > 1 else 0.0,
        })

    if not robust_rows:
        return pd.DataFrame()

    robust_df = pd.DataFrame(robust_rows)
    robust_df["robustness_score"] = (
        robust_df["avg_net_pnl"]
        + (100.0 * robust_df["avg_sharpe"])
        - (5.0 * robust_df["worst_max_drawdown_pct"])
        - robust_df["net_pnl_std"]
    )
    robust_df = robust_df.sort_values(
        ["robustness_score", "avg_net_pnl", "avg_sharpe", "worst_max_drawdown_pct"],
        ascending=[False, False, False, True],
    ).reset_index(drop=True)
    return robust_df


def _write_robust_configs_csv(results_list: list[dict], output_csv: str, top_n: int = 10) -> Path | None:
    robust_df = _build_robustness_dataframe(results_list)
    if robust_df.empty:
        return None

    output_path = Path(output_csv)
    suffix = output_path.suffix or ".csv"
    robust_path = output_path.with_name(f"{output_path.stem}_robust_top{top_n}{suffix}")
    robust_df.head(top_n).to_csv(robust_path, index=False)
    return robust_path


def _winner_threshold_value(row: pd.Series) -> float:
    if row["threshold_mode"] == THRESHOLD_MODE_ATR_MULTIPLE:
        return float(row["atr_multiple"])
    return float(row["entry_threshold_pct"])


def _winner_threshold_label(row: pd.Series) -> str:
    if row["threshold_mode"] == THRESHOLD_MODE_ATR_MULTIPLE:
        return f"{float(row['atr_multiple']):.2f}"
    return f"{float(row['entry_threshold_pct']):.4f}"


def _winner_config_key(row: pd.Series) -> tuple[Any, ...]:
    return (
        int(row["sma_bars"]),
        str(row["time_window_mode"]),
        str(row["threshold_mode"]),
        _winner_threshold_value(row),
        float(row["atr_percentile_threshold"]),
        bool(row["regime_filter_enabled"]),
    )


def _build_winner_by_symbol_dataframe(results_list: list[dict]) -> pd.DataFrame:
    per_symbol_df = _per_symbol_dataframe(results_list)
    if per_symbol_df.empty:
        return pd.DataFrame()

    symbol_rows = per_symbol_df[per_symbol_df["scope"] == "symbol"].copy()
    if symbol_rows.empty:
        return pd.DataFrame()

    metric_specs = [
        ("net_pnl", "realized_pnl", False),
        ("sharpe", "sharpe_ratio", False),
        ("max_drawdown", "max_drawdown_pct", True),
    ]

    winner_rows: list[dict[str, Any]] = []
    for symbol, group in symbol_rows.groupby("symbol", sort=True):
        for metric_name, metric_col, ascending in metric_specs:
            ranked = group.sort_values(
                [metric_col, "profit_factor", "win_rate", "trades_per_day"],
                ascending=[ascending, False, False, True],
            )
            winner = ranked.iloc[0]
            winner_rows.append({
                "symbol": symbol,
                "winner_metric": metric_name,
                "metric_value": float(winner[metric_col]),
                "sma_bars": int(winner["sma_bars"]),
                "time_window_mode": winner["time_window_mode"],
                "threshold_mode": winner["threshold_mode"],
                "threshold_value": _winner_threshold_value(winner),
                "threshold_label": _winner_threshold_label(winner),
                "trades_per_day": float(winner["trades_per_day"]),
                "win_rate": float(winner["win_rate"]),
                "profit_factor": float(winner["profit_factor"]),
                "realized_pnl": float(winner["realized_pnl"]),
                "sharpe_ratio": float(winner["sharpe_ratio"]),
                "max_drawdown_pct": float(winner["max_drawdown_pct"]),
                "expectancy": float(winner["expectancy"]),
                "max_loss_trade": float(winner["max_loss_trade"]),
                "avg_winning_trade": float(winner["avg_winning_trade"]),
                "avg_losing_trade": float(winner["avg_losing_trade"]),
                "avg_return_per_trade_pct": float(winner["avg_return_per_trade_pct"]),
                "avg_move_captured_pct": float(winner["avg_move_captured_pct"]),
                "pnl_stability": float(winner["pnl_stability"]),
                "atr_percentile_threshold": float(winner["atr_percentile_threshold"]),
                "regime_filter_enabled": bool(winner["regime_filter_enabled"]),
                "regime": winner.get("regime"),
                "start_date": winner.get("start_date"),
                "end_date": winner.get("end_date"),
                "config_key": str(_winner_config_key(winner)),
            })

    return pd.DataFrame(winner_rows)


def _write_winner_by_symbol_csv(results_list: list[dict], output_csv: str) -> Path | None:
    winners_df = _build_winner_by_symbol_dataframe(results_list)
    if winners_df.empty:
        return None

    output_path = Path(output_csv)
    suffix = output_path.suffix or ".csv"
    winners_path = output_path.with_name(f"{output_path.stem}_winner_by_symbol{suffix}")
    winners_df.to_csv(winners_path, index=False)
    return winners_path


def _format_config_label(result: dict) -> str:
    threshold_label = (
        f"k={result['atr_multiple']:.2f}"
        if result["threshold_mode"] == THRESHOLD_MODE_ATR_MULTIPLE
        else f"threshold={result['entry_threshold_pct']:.4f}"
    )
    parts = [
        f"strategy={result['strategy_mode']}",
        f"SMA={result['sma_bars']}",
        f"mode={result['threshold_mode']}",
        threshold_label,
        f"atr={result['atr_percentile_threshold']:.0f}",
        f"window={result['time_window_mode']}",
        f"regime={'on' if result['regime_filter_enabled'] else 'off'}",
    ]
    if result["strategy_mode"] == STRATEGY_MODE_BREAKOUT:
        parts.append(f"breakout_exit={result.get('breakout_exit_style')}")
    if result["strategy_mode"] == STRATEGY_MODE_MEAN_REVERSION:
        parts.append(f"mr_exit={result.get('mean_reversion_exit_style')}")
        parts.append(f"mr_atr_max={result.get('mean_reversion_max_atr_percentile', 0):.0f}")
    if result.get("regime"):
        parts.append(f"regime={result['regime']}")
    return ", ".join(parts)


def _group_results_by_config(results_list: list[dict]) -> dict[tuple[str, int, str, float, float, float, str, bool], list[dict]]:
    grouped: dict[tuple[str, int, str, float, float, float, str, bool], list[dict]] = {}
    for result in results_list:
        key = (
            result["strategy_mode"],
            result["sma_bars"],
            result["threshold_mode"],
            result["entry_threshold_pct"],
            result["atr_multiple"],
            result["atr_percentile_threshold"],
            result["time_window_mode"],
            result["regime_filter_enabled"],
        )
        grouped.setdefault(key, []).append(result)
    return grouped


def _print_post_run_summary(results_list: list[dict]) -> None:
    if not results_list:
        return

    print("\nPost-Run Summary:")

    best_return = max(results_list, key=lambda r: r["total_return_pct"])
    print(
        f"Best Return:            {_format_config_label(best_return)} "
        f"({best_return['total_return_pct']:.2f}%)"
    )

    best_drawdown = min(results_list, key=lambda r: r["max_drawdown_pct"])
    print(
        f"Lowest Max Drawdown:    {_format_config_label(best_drawdown)} "
        f"({best_drawdown['max_drawdown_pct']:.1f}%)"
    )

    eligible_win_rate = [r for r in results_list if r["total_trades"] >= 10]
    if eligible_win_rate:
        best_win_rate = max(eligible_win_rate, key=lambda r: r["win_rate"])
        print(
            f"Best Win Rate:          {_format_config_label(best_win_rate)} "
            f"({best_win_rate['win_rate']:.1f}% with {best_win_rate['total_trades']} trades)"
        )
    else:
        print("Best Win Rate:          No config met the 10-trade minimum")

    regime_names = {result.get("regime") for result in results_list if result.get("regime")}
    if len(regime_names) > 1:
        consistent_candidates: list[tuple[float, int, float, list[dict]]] = []
        for _, config_results in _group_results_by_config(results_list).items():
            config_regimes = {result.get("regime") for result in config_results}
            if config_regimes == regime_names:
                avg_return = sum(result["total_return_pct"] for result in config_results) / len(config_results)
                total_trades = sum(result["total_trades"] for result in config_results)
                avg_drawdown = sum(result["max_drawdown_pct"] for result in config_results) / len(config_results)
                consistent_candidates.append((avg_return, total_trades, avg_drawdown, config_results))

        if consistent_candidates:
            _, _, _, best_consistent_results = max(
                consistent_candidates,
                key=lambda item: (item[0], item[1], -item[2]),
            )
            sample = best_consistent_results[0]
            avg_return = sum(result["total_return_pct"] for result in best_consistent_results) / len(best_consistent_results)
            print(
                f"Most Consistent:        SMA={sample['sma_bars']}, "
                f"mode={sample['threshold_mode']}, "
                f"{'k=' + format(sample['atr_multiple'], '.2f') if sample['threshold_mode'] == THRESHOLD_MODE_ATR_MULTIPLE else 'threshold=' + format(sample['entry_threshold_pct'], '.4f')}, "
                f"atr={sample['atr_percentile_threshold']:.0f}, "
                f"window={sample['time_window_mode']} "
                f"(avg return {avg_return:.2f}% across {len(best_consistent_results)} regimes)"
            )
        else:
            print("Most Consistent:        No config appeared in all regimes")

    symbol_returns: dict[str, list[float]] = {}
    for result in results_list:
        for symbol, symbol_result in result.get("per_symbol", {}).items():
            symbol_returns.setdefault(symbol, []).append(symbol_result["total_return_pct"])

    if symbol_returns:
        avg_symbol_returns = {
            symbol: sum(returns) / len(returns)
            for symbol, returns in symbol_returns.items()
        }
        worst_symbol, worst_avg_return = min(avg_symbol_returns.items(), key=lambda item: item[1])
        best_symbol, best_avg_return = max(avg_symbol_returns.items(), key=lambda item: item[1])
        print(f"Worst Symbol Overall:   {worst_symbol} ({worst_avg_return:.2f}% avg return)")
        print(f"Best Symbol Overall:    {best_symbol} ({best_avg_return:.2f}% avg return)")


def _print_grouped_result_tables(results_list: list[dict]) -> None:
    per_symbol_df = _per_symbol_dataframe(results_list)
    if per_symbol_df.empty:
        return

    symbol_rows = per_symbol_df[per_symbol_df["scope"] == "symbol"].copy()
    if symbol_rows.empty:
        return

    print("\nGrouped Results:")
    for (symbol, symbol_strategy_mode, time_window_mode, threshold_mode), group in symbol_rows.groupby(
        ["symbol", "symbol_strategy_mode", "time_window_mode", "threshold_mode"],
        sort=True,
        dropna=False,
    ):
        print(
            f"\n{symbol} | strategy={symbol_strategy_mode} | window={time_window_mode} | threshold_mode={threshold_mode}"
        )
        print(
            f"{'SMA':>5}  {'Param':>8}  {'NetPnL':>9}  {'Sharpe':>7}  {'Max DD':>7}  "
            f"{'Win%':>6}  {'PF':>6}  {'Trd/Day':>7}  {'AvgRet':>8}  {'Stability':>10}"
        )
        print("-" * 96)
        sorted_group = group.sort_values("realized_pnl", ascending=False)
        for _, row in sorted_group.iterrows():
            threshold_param = (
                f"{row['atr_multiple']:.2f}"
                if row["threshold_mode"] == THRESHOLD_MODE_ATR_MULTIPLE
                else f"{row['entry_threshold_pct']:.4f}"
            )
            print(
                f"{int(row['sma_bars']):>5}  "
                f"{threshold_param:>8}  "
                f"${row['realized_pnl']:>8.2f}  "
                f"{row['sharpe_ratio']:>7.2f}  "
                f"{row['max_drawdown_pct']:>6.1f}%  "
                f"{row['win_rate']:>5.1f}%  "
                f"{row['profit_factor']:>6.2f}  "
                f"{row['trades_per_day']:>7.2f}  "
                f"{row['avg_return_per_trade_pct']:>7.2f}%  "
                f"${row['pnl_stability']:>9.2f}"
            )


def _print_trade_risk_summary(results_list: list[dict]) -> None:
    per_symbol_df = _per_symbol_dataframe(results_list)
    if per_symbol_df.empty:
        return

    symbol_rows = per_symbol_df[per_symbol_df["scope"] == "symbol"].copy()
    if symbol_rows.empty:
        return

    print("\nTrade Risk Summary:")
    print(
        f"{'Symbol':<6}  {'Strategy':<15}  {'Window':<16}  {'Expect':>9}  {'AvgWin':>9}  "
        f"{'AvgLoss':>9}  {'MaxLoss':>9}  {'PF':>6}  {'Contrib':>8}"
    )
    print("-" * 101)
    best_rows = symbol_rows.sort_values(
        ["realized_pnl", "sharpe_ratio", "profit_factor"],
        ascending=[False, False, False],
    ).groupby("symbol", sort=True).head(1)
    for _, row in best_rows.iterrows():
        print(
            f"{row['symbol']:<6}  "
            f"{row['symbol_strategy_mode']:<15}  "
            f"{row['time_window_mode']:<16}  "
            f"${row['expectancy']:>8.2f}  "
            f"${row['avg_winning_trade']:>8.2f}  "
            f"${row['avg_losing_trade']:>8.2f}  "
            f"${row['max_loss_trade']:>8.2f}  "
            f"{row['profit_factor']:>6.2f}  "
            f"{row.get('symbol_contribution_pct', 0.0):>7.1f}%"
        )


def _print_metric_summary_tables(results_list: list[dict]) -> None:
    if not results_list:
        return

    summary_specs = [
        ("Best Net PnL", "realized_pnl", False, "${value:.2f}"),
        ("Best Sharpe", "sharpe_ratio", False, "{value:.2f}"),
        ("Lowest Max Drawdown", "max_drawdown_pct", True, "{value:.1f}%"),
        ("Lowest Trades / Day", "trades_per_day", True, "{value:.2f}"),
    ]

    print("\nSummary Tables:")
    for title, metric, ascending, value_fmt in summary_specs:
        ranked = sorted(results_list, key=lambda row: row[metric], reverse=not ascending)[:5]
        print(f"\n{title}:")
        print(
            f"{'Rank':>4}  {'Window':<16}  {'Mode':<12}  {'Param':>8}  {'Symbols':>7}  {'Value':>12}"
        )
        print("-" * 72)
        for idx, result in enumerate(ranked, start=1):
            threshold_param = (
                f"{result['atr_multiple']:.2f}"
                if result["threshold_mode"] == THRESHOLD_MODE_ATR_MULTIPLE
                else f"{result['entry_threshold_pct']:.4f}"
            )
            per_symbol_count = len(result.get("per_symbol", {}))
            print(
                f"{idx:>4}  "
                f"{result['time_window_mode']:<16}  "
                f"{result['threshold_mode']:<12}  "
                f"{threshold_param:>8}  "
                f"{per_symbol_count:>7}  "
                f"{value_fmt.format(value=result[metric]):>12}"
            )


def _print_robustness_summary(results_list: list[dict], top_n: int = 10) -> None:
    robust_df = _build_robustness_dataframe(results_list)
    if robust_df.empty:
        print("\nRobustness Analysis:\nNo configuration covered all symbols and time windows.")
        return

    print("\nRobustness Analysis:")
    print(
        f"{'Rank':>4}  {'Mode':<12}  {'Param':>8}  {'AvgPnL':>9}  {'AvgSharpe':>10}  "
        f"{'WorstDD':>8}  {'PnL Std':>9}  {'AvgT/Day':>9}"
    )
    print("-" * 84)
    for idx, (_, row) in enumerate(robust_df.head(top_n).iterrows(), start=1):
        threshold_param = (
            f"{row['atr_multiple']:.2f}"
            if row["threshold_mode"] == THRESHOLD_MODE_ATR_MULTIPLE
            else f"{row['entry_threshold_pct']:.4f}"
        )
        print(
            f"{idx:>4}  "
            f"{row['threshold_mode']:<12}  "
            f"{threshold_param:>8}  "
            f"${row['avg_net_pnl']:>8.2f}  "
            f"{row['avg_sharpe']:>10.2f}  "
            f"{row['worst_max_drawdown_pct']:>7.1f}%  "
            f"${row['net_pnl_std']:>8.2f}  "
            f"{row['avg_trades_per_day']:>9.2f}"
        )


def _print_winner_by_symbol_summary(results_list: list[dict]) -> None:
    winners_df = _build_winner_by_symbol_dataframe(results_list)
    if winners_df.empty:
        return

    metric_labels = {
        "net_pnl": "Best Net PnL",
        "sharpe": "Best Sharpe",
        "max_drawdown": "Lowest Max DD",
    }

    print("\nWinner By Symbol:")
    for metric_name in ("net_pnl", "sharpe", "max_drawdown"):
        metric_winners = winners_df[winners_df["winner_metric"] == metric_name].copy()
        if metric_winners.empty:
            continue

        print(f"\n{metric_labels[metric_name]}:")
        print(
            f"{'Symbol':<6}  {'Window':<16}  {'Mode':<12}  {'Value':>10}  "
            f"{'Thresh':>8}  {'Trd/Day':>7}  {'Win%':>6}  {'PF':>6}"
        )
        print("-" * 84)
        for _, row in metric_winners.iterrows():
            value_str = (
                f"{row['metric_value']:.1f}%"
                if metric_name == "max_drawdown"
                else f"{row['metric_value']:.2f}"
            )
            if metric_name == "net_pnl":
                value_str = f"${row['metric_value']:.2f}"
            print(
                f"{row['symbol']:<6}  "
                f"{row['time_window_mode']:<16}  "
                f"{row['threshold_mode']:<12}  "
                f"{value_str:>10}  "
                f"{row['threshold_label']:>8}  "
                f"{row['trades_per_day']:>7.2f}  "
                f"{row['win_rate']:>5.1f}%  "
                f"{row['profit_factor']:>6.2f}"
            )

    pnl_winners = winners_df[winners_df["winner_metric"] == "net_pnl"].copy()
    if pnl_winners.empty:
        return

    print("\nThreshold Winner Comparison:")
    mode_counts = pnl_winners["threshold_mode"].value_counts()
    print(f"{'Symbol':<6}  {'Winning Mode':<12}  {'Window':<16}  {'Thresh':>8}")
    print("-" * 52)
    for _, row in pnl_winners.iterrows():
        print(
            f"{row['symbol']:<6}  "
            f"{row['threshold_mode']:<12}  "
            f"{row['time_window_mode']:<16}  "
            f"{row['threshold_label']:>8}"
        )

    unique_configs = pnl_winners["config_key"].nunique()
    if unique_configs == 1:
        print("Config Note:            The same net-PnL winner appears across all symbols.")
    else:
        print("Config Note:            Net-PnL winners differ by symbol; the setup remains symbol-specific.")

    if len(mode_counts) == 1:
        winning_mode = mode_counts.index[0]
        print(f"Threshold Note:         All symbol winners use {winning_mode}.")
    else:
        mode_summary = ", ".join(f"{mode}={count}" for mode, count in mode_counts.items())
        print(f"Threshold Note:         Winner mix by symbol -> {mode_summary}.")


def _parse_regime_specs(regime_specs: list[str] | None, start_date: str | None, end_date: str | None) -> list[dict[str, str | None]]:
    if regime_specs:
        regimes = []
        for spec in regime_specs:
            parts = [part.strip() for part in spec.split(":")]
            if len(parts) != 3 or not all(parts):
                raise ValueError(
                    "Invalid --regime format. Use NAME:YYYY-MM-DD:YYYY-MM-DD, "
                    "for example Bull:2020-04-01:2021-12-31"
                )
            name, regime_start, regime_end = parts
            regimes.append({"regime": name, "start_date": regime_start, "end_date": regime_end})
        return regimes
    return [{"regime": "Custom", "start_date": start_date, "end_date": end_date}]


def _parse_csv_list(raw_value: str, cast: type[int] | type[float]) -> list[int] | list[float]:
    values = [item.strip() for item in raw_value.split(",") if item.strip()]
    if not values:
        raise ValueError("Sweep list cannot be empty.")
    return [cast(item) for item in values]


def _parse_str_list(raw_value: str) -> list[str]:
    values = [item.strip().lower() for item in raw_value.split(",") if item.strip()]
    if not values:
        raise ValueError("Sweep list cannot be empty.")
    return values


def _parse_regime_filter_values(raw_value: str) -> list[bool]:
    parsed: list[bool] = []
    for item in _parse_str_list(raw_value):
        if item in {"on", "true", "enabled", "enable", "1"}:
            parsed.append(True)
        elif item in {"off", "false", "disabled", "disable", "0"}:
            parsed.append(False)
        else:
            raise ValueError("Use regime filter values of on/off")
    return parsed


def _parse_strategy_mode_list(raw_value: str) -> list[str]:
    modes = [normalize_strategy_mode(value) for value in _parse_str_list(raw_value)]
    invalid_modes = [mode for mode in modes if mode not in STRATEGY_MODE_CHOICES]
    if invalid_modes:
        raise ValueError(
            f"Unsupported strategy mode(s): {', '.join(invalid_modes)}. "
            f"Choose from {', '.join(STRATEGY_MODE_CHOICES)}."
        )
    return modes


def _parse_breakout_exit_style_list(raw_value: str) -> list[str]:
    styles = [normalize_breakout_exit_style(value) for value in _parse_str_list(raw_value)]
    invalid_styles = [style for style in styles if style not in BREAKOUT_EXIT_CHOICES]
    if invalid_styles:
        raise ValueError(
            f"Unsupported breakout exit style(s): {', '.join(invalid_styles)}. "
            f"Choose from {', '.join(BREAKOUT_EXIT_CHOICES)}."
        )
    return styles


def _parse_mean_reversion_exit_style_list(raw_value: str) -> list[str]:
    styles = [normalize_mean_reversion_exit_style(value) for value in _parse_str_list(raw_value)]
    invalid_styles = [style for style in styles if style not in MEAN_REVERSION_EXIT_CHOICES]
    if invalid_styles:
        raise ValueError(
            f"Unsupported mean reversion exit style(s): {', '.join(invalid_styles)}. "
            f"Choose from {', '.join(MEAN_REVERSION_EXIT_CHOICES)}."
        )
    return styles


def _print_sweep_results(results_list: list[dict], strategy_mode: str, show_regime: bool = False) -> None:
    print(f"\nSweep Results (sorted by Total Return)")
    regime_header = f"{'Regime':<12}  " if show_regime else ""
    print(
        f"{regime_header}{'Strategy':<14}  {'Window':<16}  {'Mode':<12}  {'Param':>8}  {'Trades':>7}  {'Win%':>6}  "
        f"{'AvgRet':>8}  {'AvgMove':>8}  {'Max DD':>7}  {'Sharpe':>7}  {'PF':>6}"
    )
    print("-" * (146 if show_regime else 132))
    for res in results_list:
        regime_prefix = f"{res.get('regime', ''):<12}  " if show_regime else ""
        threshold_param = (
            f"{res['atr_multiple']:.2f}"
            if res["threshold_mode"] == THRESHOLD_MODE_ATR_MULTIPLE
            else f"{res['entry_threshold_pct']:.4f}"
        )
        print(
            f"{regime_prefix}"
            f"{res['strategy_mode']:<14}  "
            f"{res['time_window_mode']:<16}  "
            f"{res['threshold_mode']:<12}  "
            f"{threshold_param:>8}  "
            f"{res['total_trades']:>7}  "
            f"{res['win_rate']:>5.1f}%  "
            f"{res['avg_return_per_trade_pct']:>7.2f}%  "
            f"{res['avg_move_captured_pct']:>7.2f}%  "
            f"{res['max_drawdown_pct']:>6.1f}%  "
            f"{res['sharpe_ratio']:>7.2f}  "
            f"{res['profit_factor']:>6.2f}"
        )


def _print_single_result(results: dict) -> None:
    print(f"Regime:                 {results.get('regime', 'Custom')}")
    print(f"Date Range:             {results.get('start_date') or 'start'} to {results.get('end_date') or 'end'}")
    print("Combined Portfolio:")
    print(f"Strategy Mode:          {results['strategy_mode']}")
    if results.get("symbol_strategy_modes"):
        symbol_strategy_text = ", ".join(
            f"{symbol}:{mode}" for symbol, mode in sorted(results["symbol_strategy_modes"].items())
        )
        print(f"Symbol Strategies:      {symbol_strategy_text}")
    print(f"Threshold Mode:         {results['threshold_mode']}")
    if results["threshold_mode"] == THRESHOLD_MODE_ATR_MULTIPLE:
        print(f"ATR Multiple:           {results['atr_multiple']:.2f}")
    else:
        print(f"Static Threshold:       {results['entry_threshold_pct']:.4f}")
    print(f"Time Window Mode:       {results['time_window_mode']}")
    print(f"Regime Filter:          {'enabled' if results['regime_filter_enabled'] else 'disabled'}")
    print(f"Position Size:          ${results['position_size']:.2f}")
    print(f"ATR Percentile Filter:  {results['atr_percentile_threshold']:.0f}")
    print(f"Starting Capital:       ${results['starting_capital']:.2f}")
    print(f"Final Cash:             ${results['final_cash']:.2f}")
    print(f"Final Equity:           ${results['final_equity']:.2f}")
    print(f"Net PnL:                ${results['realized_pnl']:.2f}")
    print(f"Total Return:           {results['total_return_pct']:.2f}%")
    print(f"Max Drawdown:           {results['max_drawdown_pct']:.1f}%")
    print(f"Total Trades:           {results['total_trades']}")
    print(f"Skipped Trades:         {results['skipped_trades']}")
    print(f"Max Concurrent Pos:     {results['max_concurrent_positions']}")
    print(f"Total BUY Candidates:   {results['total_buy_candidates']}")
    print(f"Competing Timestamps:   {results['competing_timestamps']}")
    print(f"Sharpe Ratio:           {results['sharpe_ratio']:.2f}")
    print(f"Win Rate:               {results['win_rate']:.1f}%")
    print(f"Profit Factor:          {results['profit_factor']:.2f}")
    print(f"Trades / Day:           {results['trades_per_day']:.2f}")
    print(f"PnL Stability:          ${results['pnl_stability']:.2f}")
    print(f"Expectancy:             ${results['expectancy']:.2f}")
    print(f"Max Loss / Trade:       ${results['max_loss_trade']:.2f}")
    print(f"Avg Move Captured:      {results['avg_move_captured_pct']:.2f}%")
    print(f"Average Winning Trade:  ${results['avg_winning_trade']:.2f}")
    print(f"Average Losing Trade:   ${results['avg_losing_trade']:.2f}")
    if results.get("ml_signal_summary"):
        print("ML Signal Summary:")
        for symbol, summary in results["ml_signal_summary"].items():
            min_prob = summary["min_prob"]
            max_prob = summary["max_prob"]
            mean_prob = summary["mean_prob"]
            buy_threshold = summary["buy_threshold"]
            sell_threshold = summary["sell_threshold"]
            min_text = f"{min_prob:.3f}" if min_prob is not None else "n/a"
            max_text = f"{max_prob:.3f}" if max_prob is not None else "n/a"
            mean_text = f"{mean_prob:.3f}" if mean_prob is not None else "n/a"
            buy_text = f"{buy_threshold:.3f}" if buy_threshold is not None else "n/a"
            sell_text = f"{sell_threshold:.3f}" if sell_threshold is not None else "n/a"
            print(
                f"  {symbol}: "
                f"count={summary['count']} "
                f"min={min_text} max={max_text} mean={mean_text} "
                f"buy_thr={buy_text} sell_thr={sell_text} "
                f"above_buy={summary['above_buy_count']} below_sell={summary['below_sell_count']}"
            )
    if results.get("timing_seconds"):
        timing = results["timing_seconds"]
        print(
            "Timing (s):            "
            f"load={timing['load']:.3f}, prep={timing['precompute']:.3f}, "
            f"model_load={timing.get('model_load', 0.0):.3f}, "
            f"ml_precompute={timing.get('ml_precompute', 0.0):.3f}, "
            f"simulate={timing['simulate']:.3f}, finalize={timing['finalize']:.3f}"
        )
    print("Symbol Trade Counts:")
    for symbol, count in results["symbol_trade_counts"].items():
        print(f"  {symbol}: {count}")
    print("Final Positions: All flat (forced close at end)")
    print("\nPer-Symbol Results:")
    for symbol, symbol_result in results.get("per_symbol", {}).items():
        print(f"\n{symbol}:")
        print(f"  Strategy:             {results.get('symbol_strategy_modes', {}).get(symbol, results['strategy_mode'])}")
        print(f"  Return:               {symbol_result['total_return_pct']:.2f}%")
        print(f"  Max Drawdown:         {symbol_result['max_drawdown_pct']:.1f}%")
        print(f"  Win Rate:             {symbol_result['win_rate']:.1f}%")
        print(f"  PnL Contribution:     {symbol_result.get('symbol_contribution_pct', 0.0):.1f}%")
        print(f"  Expectancy:           ${symbol_result['expectancy']:.2f}")
        print(f"  Max Loss / Trade:     ${symbol_result['max_loss_trade']:.2f}")
        print(f"  Avg Return / Trade:   {symbol_result['avg_return_per_trade_pct']:.2f}%")
        print(f"  Average Winning Trade:${symbol_result['avg_winning_trade']:.2f}")
        print(f"  Average Losing Trade: ${symbol_result['avg_losing_trade']:.2f}")
        print(f"  Total Trades:         {symbol_result['total_trades']}")


def _print_regime_results(results_list: list[dict]) -> None:
    for idx, result in enumerate(results_list):
        if idx > 0:
            print("\n" + "=" * 72)
        _print_single_result(result)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a backtest on a historical dataset.")
    parser.add_argument("--dataset", required=True, help="Path to dataset directory")
    parser.add_argument(
        "--symbols",
        nargs="*",
        help="Optional symbol override for debugging. Default: discover from manifest metadata, then dataset content.",
    )
    parser.add_argument("--strategy-mode", default=STRATEGY_MODE_HYBRID,
                        help=f"Strategy mode (default: {STRATEGY_MODE_HYBRID})")
    parser.add_argument("--strategy-mode-list",
                        help="Comma-separated strategy modes to compare, e.g. sma,breakout,mean_reversion")
    parser.add_argument("--symbol-strategy-map",
                        help="Per-symbol strategy map, e.g. AAPL:sma,MSFT:breakout,NVDA:mean_reversion")
    parser.add_argument("--sma-bars", type=int, default=DEFAULT_SMA_BARS)
    parser.add_argument("--sma-bars-list", help="Comma-separated SMA values to sweep, e.g. 10,20,30,50")
    parser.add_argument("--threshold-mode", default=THRESHOLD_MODE_STATIC_PCT)
    parser.add_argument("--threshold-mode-list", help="Comma-separated threshold modes to compare: static_pct,atr_multiple")
    parser.add_argument("--entry-threshold-pct", type=float, default=DEFAULT_ENTRY_THRESHOLD_PCT)
    parser.add_argument("--entry-threshold-pct-list",
                        help="Comma-separated entry thresholds to sweep")
    parser.add_argument("--atr-multiple", "--atr-threshold-multiplier", dest="atr_multiple", type=float, default=DEFAULT_ATR_MULTIPLE)
    parser.add_argument("--atr-multiple-list", "--atr-threshold-multiplier-list", dest="atr_multiple_list",
                        help="Comma-separated ATR multiples to sweep, e.g. 0.5,1.0,1.5,2.0")
    parser.add_argument("--atr-percentile-threshold", type=float, default=DEFAULT_DISABLED_ATR_PERCENTILE_THRESHOLD)
    parser.add_argument("--atr-percentile-threshold-list",
                        help="Comma-separated ATR percentile thresholds to sweep, e.g. 40,50,60,70")
    parser.add_argument("--time-window-mode", default=TIME_WINDOW_FULL_DAY)
    parser.add_argument(
        "--time-window-mode-list",
        help="Comma-separated time window modes to compare: full_day,morning_only,afternoon_only,combined_windows",
    )
    parser.add_argument("--regime-filter-enabled", action="store_true")
    parser.add_argument("--regime-filter-list", help="Comma-separated regime filter modes: off,on")
    parser.add_argument("--orb-filter-mode", default=ORB_FILTER_NONE,
                        help="ORB entry filter: none or volume_or_volatility")
    parser.add_argument("--breakout-exit-style", default=BREAKOUT_EXIT_TARGET_1X_STOP_LOW,
                        help="Breakout exit style")
    parser.add_argument("--breakout-exit-style-list",
                        help="Comma-separated breakout exit styles to compare")
    parser.add_argument("--breakout-tight-stop-fraction", type=float, default=0.5)
    parser.add_argument("--breakout-max-stop-pct", type=float, default=0.03,
                        help="Cap breakout stop as fraction of entry price (default: 0.03 = 3%%)")
    parser.add_argument("--breakout-gap-pct-min", type=float, default=0.0,
                        help="Min gap-up %% required for breakout entry (default: 0 = off)")
    parser.add_argument("--breakout-or-range-pct-min", type=float, default=0.0,
                        help="Min OR range %% of OR low required for breakout entry (default: 0 = off)")
    parser.add_argument("--mean-reversion-exit-style", default=MEAN_REVERSION_EXIT_SMA,
                        help="Mean reversion exit style")
    parser.add_argument("--mean-reversion-exit-style-list",
                        help="Comma-separated mean reversion exit styles to compare")
    parser.add_argument("--mean-reversion-max-atr-percentile", type=float, default=0.0)
    parser.add_argument("--mean-reversion-stop-pct", type=float, default=0.0,
                        help="Exit if price falls N%% below entry (default: 0 = disabled, e.g. 0.02 = 2%%)")
    parser.add_argument("--mean-reversion-trend-filter", action="store_true", default=False,
                        help="Only enter mean reversion trades when price >= 50-bar SMA (uptrend filter)")
    parser.add_argument("--mean-reversion-trend-slope-filter", action="store_true", default=False,
                        help="Only enter mean reversion trades when SMA_50 slope >= 0 (rising trend filter)")
    parser.add_argument("--mean-reversion-max-atr-percentile-list",
                        help="Comma-separated max ATR percentile values for mean reversion entry")
    parser.add_argument("--ml-lookback-bars", type=int, default=DEFAULT_ML_LOOKBACK_BARS)
    parser.add_argument("--ml-retrain-every-bars", type=int, default=DEFAULT_ML_RETRAIN_EVERY_BARS)
    parser.add_argument("--ml-probability-buy", type=float, default=DEFAULT_ML_PROBABILITY_BUY)
    parser.add_argument("--ml-probability-sell", type=float, default=DEFAULT_ML_PROBABILITY_SELL)
    parser.add_argument("--commission-per-order", type=float, default=0.01)
    parser.add_argument("--slippage-per-share", type=float, default=0.05)
    parser.add_argument("--starting-capital", type=float, default=DEFAULT_STARTING_CAPITAL)
    parser.add_argument("--position-size", type=float, default=DEFAULT_POSITION_SIZE)
    parser.add_argument("--output-csv", help="CSV output path for sweep results")
    parser.add_argument("--start-date")
    parser.add_argument("--end-date")
    parser.add_argument(
        "--regime",
        action="append",
        help="Named regime in NAME:YYYY-MM-DD:YYYY-MM-DD format. Repeat for multiple ranges.",
    )

    args = parser.parse_args()
    args.time_window_mode = normalize_time_window_mode(args.time_window_mode)
    args.threshold_mode = normalize_threshold_mode(args.threshold_mode)
    args.strategy_mode = normalize_strategy_mode(args.strategy_mode)
    args.orb_filter_mode = normalize_orb_filter_mode(args.orb_filter_mode)
    args.breakout_exit_style = normalize_breakout_exit_style(args.breakout_exit_style)
    args.mean_reversion_exit_style = normalize_mean_reversion_exit_style(args.mean_reversion_exit_style)
    if args.orb_filter_mode not in ORB_FILTER_CHOICES:
        raise ValueError(
            f"Unsupported orb filter mode: {args.orb_filter_mode}. "
            f"Choose from {', '.join(ORB_FILTER_CHOICES)}."
        )
    if args.breakout_exit_style not in BREAKOUT_EXIT_CHOICES:
        raise ValueError(
            f"Unsupported breakout exit style: {args.breakout_exit_style}. "
            f"Choose from {', '.join(BREAKOUT_EXIT_CHOICES)}."
        )
    if args.mean_reversion_exit_style not in MEAN_REVERSION_EXIT_CHOICES:
        raise ValueError(
            f"Unsupported mean reversion exit style: {args.mean_reversion_exit_style}. "
            f"Choose from {', '.join(MEAN_REVERSION_EXIT_CHOICES)}."
        )
    strategy_modes = (
        _parse_strategy_mode_list(args.strategy_mode_list)
        if args.strategy_mode_list
        else [args.strategy_mode]
    )
    invalid_direct_modes = [mode for mode in strategy_modes if mode not in STRATEGY_MODE_CHOICES]
    if invalid_direct_modes:
        raise ValueError(
            f"Unsupported strategy mode(s): {', '.join(invalid_direct_modes)}. "
            f"Choose from {', '.join(STRATEGY_MODE_CHOICES)}."
        )
    symbol_strategy_map = _parse_symbol_strategy_map(args.symbol_strategy_map)
    dataset_path = Path(args.dataset)
    sma_values = _parse_csv_list(args.sma_bars_list, int) if args.sma_bars_list else [args.sma_bars]
    threshold_modes = (
        _parse_str_list(args.threshold_mode_list)
        if args.threshold_mode_list
        else [args.threshold_mode]
    )
    threshold_modes = [normalize_threshold_mode(mode) for mode in threshold_modes]
    invalid_threshold_modes = [mode for mode in threshold_modes if mode not in THRESHOLD_MODE_CHOICES]
    if invalid_threshold_modes:
        raise ValueError(
            f"Unsupported threshold mode(s): {', '.join(invalid_threshold_modes)}. "
            f"Choose from {', '.join(THRESHOLD_MODE_CHOICES)}."
        )
    static_threshold_values = (
        _parse_csv_list(args.entry_threshold_pct_list, float)
        if args.entry_threshold_pct_list
        else [args.entry_threshold_pct]
    )
    dynamic_threshold_values = (
        _parse_csv_list(args.atr_multiple_list, float)
        if args.atr_multiple_list
        else [args.atr_multiple]
    )
    atr_threshold_values = (
        _parse_csv_list(args.atr_percentile_threshold_list, float)
        if args.atr_percentile_threshold_list
        else [args.atr_percentile_threshold]
    )
    breakout_exit_styles = (
        _parse_breakout_exit_style_list(args.breakout_exit_style_list)
        if args.breakout_exit_style_list
        else [args.breakout_exit_style]
    )
    mean_reversion_exit_styles = (
        _parse_mean_reversion_exit_style_list(args.mean_reversion_exit_style_list)
        if args.mean_reversion_exit_style_list
        else [args.mean_reversion_exit_style]
    )
    mean_reversion_atr_max_values = (
        _parse_csv_list(args.mean_reversion_max_atr_percentile_list, float)
        if args.mean_reversion_max_atr_percentile_list
        else [args.mean_reversion_max_atr_percentile]
    )
    time_window_modes = (
        _parse_str_list(args.time_window_mode_list)
        if args.time_window_mode_list
        else [args.time_window_mode]
    )
    time_window_modes = [normalize_time_window_mode(mode) for mode in time_window_modes]
    regime_filter_values = (
        _parse_regime_filter_values(args.regime_filter_list)
        if args.regime_filter_list
        else [args.regime_filter_enabled]
    )
    invalid_window_modes = [mode for mode in time_window_modes if mode not in TIME_WINDOW_CHOICES]
    if invalid_window_modes:
        raise ValueError(
            f"Unsupported time window mode(s): {', '.join(invalid_window_modes)}. "
            f"Choose from {', '.join(TIME_WINDOW_CHOICES)}."
        )
    threshold_sensitive_modes = set(strategy_modes) | set(symbol_strategy_map.values())
    if any(mode not in {STRATEGY_MODE_SMA, STRATEGY_MODE_MEAN_REVERSION} for mode in threshold_sensitive_modes) and (
        args.entry_threshold_pct_list
        or args.atr_multiple_list
        or args.threshold_mode_list
        or args.threshold_mode == THRESHOLD_MODE_ATR_MULTIPLE
    ):
        raise ValueError("threshold sweeps are only supported for sma and mean_reversion strategy modes")
    regimes = _parse_regime_specs(args.regime, args.start_date, args.end_date)

    common_kwargs = dict(
        commission=args.commission_per_order,
        slippage=args.slippage_per_share,
        starting_capital=args.starting_capital,
        position_size=args.position_size,
        strategy_mode=args.strategy_mode,
        orb_filter_mode=args.orb_filter_mode,
        breakout_exit_style=args.breakout_exit_style,
        breakout_tight_stop_fraction=args.breakout_tight_stop_fraction,
        breakout_max_stop_pct=args.breakout_max_stop_pct,
        breakout_gap_pct_min=args.breakout_gap_pct_min,
        breakout_or_range_pct_min=args.breakout_or_range_pct_min,
        mean_reversion_exit_style=args.mean_reversion_exit_style,
        mean_reversion_max_atr_percentile=args.mean_reversion_max_atr_percentile,
        mean_reversion_stop_pct=args.mean_reversion_stop_pct,
        mean_reversion_trend_filter=args.mean_reversion_trend_filter,
        mean_reversion_trend_slope_filter=args.mean_reversion_trend_slope_filter,
        ml_lookback_bars=args.ml_lookback_bars,
        ml_retrain_every_bars=args.ml_retrain_every_bars,
        ml_probability_buy=args.ml_probability_buy,
        ml_probability_sell=args.ml_probability_sell,
    )

    if (
        args.sma_bars_list
        or args.strategy_mode_list
        or args.threshold_mode_list
        or args.entry_threshold_pct_list
        or args.atr_multiple_list
        or args.atr_percentile_threshold_list
        or args.time_window_mode_list
        or args.regime_filter_list
        or args.regime
    ):
        results_list = []
        for regime in regimes:
            for regime_filter_enabled in regime_filter_values:
                for time_window_mode in time_window_modes:
                    for selected_strategy_mode in strategy_modes:
                        active_symbol_strategy_map = symbol_strategy_map or {}
                        effective_strategy_mode = selected_strategy_mode
                        active_modes = set(active_symbol_strategy_map.values()) | {effective_strategy_mode}
                        active_breakout_exit_styles = (
                            breakout_exit_styles
                            if STRATEGY_MODE_BREAKOUT in active_modes
                            else [args.breakout_exit_style]
                        )
                        active_mean_reversion_exit_styles = (
                            mean_reversion_exit_styles
                            if STRATEGY_MODE_MEAN_REVERSION in active_modes
                            else [args.mean_reversion_exit_style]
                        )
                        active_mean_reversion_atr_max_values = (
                            mean_reversion_atr_max_values
                            if STRATEGY_MODE_MEAN_REVERSION in active_modes
                            else [args.mean_reversion_max_atr_percentile]
                        )
                        for sma in sma_values:
                            for threshold_mode in threshold_modes:
                                threshold_param_values = (
                                    dynamic_threshold_values
                                    if threshold_mode == THRESHOLD_MODE_ATR_MULTIPLE
                                    else static_threshold_values
                                )
                                for breakout_exit_style in active_breakout_exit_styles:
                                    for mean_reversion_exit_style in active_mean_reversion_exit_styles:
                                        for mean_reversion_atr_max in active_mean_reversion_atr_max_values:
                                            for threshold_param in threshold_param_values:
                                                for atr_threshold in atr_threshold_values:
                                                    threshold_desc = (
                                                        f"k={threshold_param}"
                                                        if threshold_mode == THRESHOLD_MODE_ATR_MULTIPLE
                                                        else f"threshold={threshold_param}"
                                                    )
                                                    print(
                                                        f"Running regime={regime['regime']}, strategy={effective_strategy_mode}, "
                                                        f"regime_filter={'on' if regime_filter_enabled else 'off'}, "
                                                        f"window={time_window_mode}, SMA={sma}, threshold_mode={threshold_mode}, "
                                                        f"{threshold_desc}, atr={atr_threshold}, "
                                                        f"breakout_exit={breakout_exit_style}, mr_exit={mean_reversion_exit_style}, "
                                                        f"mr_atr_max={mean_reversion_atr_max}...",
                                                        flush=True,
                                                    )
                                                    result = run_backtest(
                                                        dataset_path,
                                                        args.symbols,
                                                        sma,
                                                        entry_threshold_pct=(
                                                            threshold_param if threshold_mode == THRESHOLD_MODE_STATIC_PCT else args.entry_threshold_pct
                                                        ),
                                                        threshold_mode=threshold_mode,
                                                        atr_multiple=(
                                                            threshold_param if threshold_mode == THRESHOLD_MODE_ATR_MULTIPLE else args.atr_multiple
                                                        ),
                                                        atr_percentile_threshold=atr_threshold,
                                                        time_window_mode=time_window_mode,
                                                        regime_filter_enabled=regime_filter_enabled,
                                                        start_date=regime["start_date"],
                                                        end_date=regime["end_date"],
                                                        strategy_mode=effective_strategy_mode,
                                                        symbol_strategy_modes=active_symbol_strategy_map,
                                                        breakout_exit_style=breakout_exit_style,
                                                        breakout_tight_stop_fraction=args.breakout_tight_stop_fraction,
                                                        mean_reversion_exit_style=mean_reversion_exit_style,
                                                        mean_reversion_max_atr_percentile=mean_reversion_atr_max,
                                                        **{k: v for k, v in common_kwargs.items() if k not in {
                                                            "strategy_mode",
                                                            "breakout_exit_style",
                                                            "breakout_tight_stop_fraction",
                                                            "mean_reversion_exit_style",
                                                            "mean_reversion_max_atr_percentile",
                                                        }},
                                                    )
                                                    result["regime"] = regime["regime"]
                                                    result["start_date"] = regime["start_date"]
                                                    result["end_date"] = regime["end_date"]
                                                    result["regime_filter_enabled"] = regime_filter_enabled
                                                    result["time_window_mode"] = time_window_mode
                                                    result["strategy_mode"] = effective_strategy_mode
                                                    result["threshold_mode"] = threshold_mode
                                                    result["sma_bars"] = sma
                                                    result["orb_filter_mode"] = args.orb_filter_mode
                                                    result["breakout_exit_style"] = breakout_exit_style
                                                    result["breakout_tight_stop_fraction"] = args.breakout_tight_stop_fraction
                                                    result["mean_reversion_exit_style"] = mean_reversion_exit_style
                                                    result["mean_reversion_max_atr_percentile"] = mean_reversion_atr_max
                                                    result["entry_threshold_pct"] = (
                                                        threshold_param if threshold_mode == THRESHOLD_MODE_STATIC_PCT else args.entry_threshold_pct
                                                    )
                                                    result["atr_multiple"] = (
                                                        threshold_param if threshold_mode == THRESHOLD_MODE_ATR_MULTIPLE else args.atr_multiple
                                                    )
                                                    result["atr_percentile_threshold"] = atr_threshold
                                                    results_list.append(result)

        results_list.sort(key=lambda x: x["total_return_pct"], reverse=True)
        if (
            args.sma_bars_list
            or args.strategy_mode_list
            or args.threshold_mode_list
            or args.entry_threshold_pct_list
            or args.atr_multiple_list
            or args.atr_percentile_threshold_list
            or args.time_window_mode_list
            or args.regime_filter_list
        ):
            _print_sweep_results(results_list, args.strategy_mode, show_regime=len(regimes) > 1)
            _print_strategy_mode_comparison(results_list)
            _print_grouped_result_tables(results_list)
            _print_metric_summary_tables(results_list)
            _print_trade_risk_summary(results_list)
            _print_robustness_summary(results_list)
            _print_winner_by_symbol_summary(results_list)
            _print_post_run_summary(results_list)
        else:
            _print_regime_results(results_list)

        if args.output_csv:
            _result_grid_dataframe(results_list).to_csv(args.output_csv, index=False)
            print(f"\nResults written to {args.output_csv}")
            per_symbol_csv = _write_per_symbol_csv(results_list, args.output_csv)
            print(f"Per-symbol results written to {per_symbol_csv}")
            robust_csv = _write_robust_configs_csv(results_list, args.output_csv)
            if robust_csv is not None:
                print(f"Robust configurations written to {robust_csv}")
            winner_csv = _write_winner_by_symbol_csv(results_list, args.output_csv)
            if winner_csv is not None:
                print(f"Winner-by-symbol results written to {winner_csv}")

    else:
        print("Running backtest...", flush=True)
        results = run_backtest(dataset_path, args.symbols, args.sma_bars,
                               entry_threshold_pct=args.entry_threshold_pct,
                               threshold_mode=args.threshold_mode,
                               atr_multiple=args.atr_multiple,
                               atr_percentile_threshold=args.atr_percentile_threshold,
                               time_window_mode=args.time_window_mode,
                               regime_filter_enabled=args.regime_filter_enabled,
                               symbol_strategy_modes=symbol_strategy_map,
                               **common_kwargs)
        results["regime"] = regimes[0]["regime"]
        results["start_date"] = regimes[0]["start_date"]
        results["end_date"] = regimes[0]["end_date"]
        results["regime_filter_enabled"] = args.regime_filter_enabled
        results["time_window_mode"] = args.time_window_mode
        results["threshold_mode"] = args.threshold_mode
        results["sma_bars"] = args.sma_bars
        results["orb_filter_mode"] = args.orb_filter_mode
        results["breakout_exit_style"] = args.breakout_exit_style
        results["breakout_tight_stop_fraction"] = args.breakout_tight_stop_fraction
        results["mean_reversion_exit_style"] = args.mean_reversion_exit_style
        results["mean_reversion_max_atr_percentile"] = args.mean_reversion_max_atr_percentile
        results["entry_threshold_pct"] = args.entry_threshold_pct
        results["atr_multiple"] = args.atr_multiple
        results["atr_percentile_threshold"] = args.atr_percentile_threshold
        print("\nBacktest Results:")
        _print_single_result(results)
        if args.output_csv:
            _result_grid_dataframe([results]).to_csv(args.output_csv, index=False)
            print(f"\nResults written to {args.output_csv}")
            per_symbol_csv = _write_per_symbol_csv([results], args.output_csv)
            print(f"Per-symbol results written to {per_symbol_csv}")
            robust_csv = _write_robust_configs_csv([results], args.output_csv)
            if robust_csv is not None:
                print(f"Robust configurations written to {robust_csv}")
            winner_csv = _write_winner_by_symbol_csv([results], args.output_csv)
            if winner_csv is not None:
                print(f"Winner-by-symbol results written to {winner_csv}")


if __name__ == "__main__":
    main()

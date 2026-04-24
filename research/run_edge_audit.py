from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from backtest_runner import (
    DEFAULT_POSITION_SIZE,
    DEFAULT_SMA_BARS,
    _initialize_simulation_state,
    _load_offline_model_payload,
    _mean,
    _prepare_backtest_inputs,
    _precompute_ml_signals,
    run_backtest,
)
from strategy import (
    STRATEGY_MODE_BOLLINGER_SQUEEZE,
    STRATEGY_MODE_BREAKOUT,
    STRATEGY_MODE_HYBRID,
    STRATEGY_MODE_HYBRID_BB_MR,
    STRATEGY_MODE_MEAN_REVERSION,
    STRATEGY_MODE_MOMENTUM_BREAKOUT,
    STRATEGY_MODE_ML,
    STRATEGY_MODE_ORB,
    STRATEGY_MODE_SMA,
    STRATEGY_MODE_TREND_PULLBACK,
    STRATEGY_MODE_VOLATILITY_EXPANSION,
    STRATEGY_MODE_WICK_FADE,
    StrategyConfig,
    is_entry_window_open,
    normalize_bollinger_exit_mode,
    normalize_breakout_exit_style,
    normalize_mean_reversion_exit_style,
    normalize_momentum_breakout_exit_style,
    normalize_volatility_expansion_exit_style,
    normalize_orb_filter_mode,
    normalize_strategy_mode,
    normalize_threshold_mode,
    normalize_time_window_mode,
    strategy_requires_adx,
)

DEFAULT_CONFIG_PATH = Path("config") / "live_config.json"
DEFAULT_COMMISSION = 0.01
DEFAULT_SLIPPAGE = 0.05
VARIANT_LIVE_EFFECTIVE = "live_effective"
VARIANT_EXPLICIT_CONFIG = "explicit_config"
VARIANT_CHOICES = (VARIANT_LIVE_EFFECTIVE, VARIANT_EXPLICIT_CONFIG)


@dataclass(frozen=True)
class AuditContext:
    dataset_path: Path
    config_path: Path
    runtime: dict[str, Any]
    backtest_kwargs: dict[str, Any]
    symbols: list[str] | None
    output_dir: Path | None
    source_dataset: str | None
    live_runtime_missing_fields: tuple[str, ...]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run a research-focused edge audit for the current strategy. "
            "Defaults to config/live_config.json and, when available, its source.dataset."
        )
    )
    parser.add_argument("--dataset", help="Path to dataset directory. Defaults to config.live_config source.dataset.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH), help="Path to runtime config JSON.")
    parser.add_argument(
        "--symbols",
        nargs="*",
        help="Optional symbol override. Default: runtime symbols, then dataset symbols.",
    )
    parser.add_argument("--start-date", help="Optional UTC date filter (YYYY-MM-DD).")
    parser.add_argument("--end-date", help="Optional UTC date filter (YYYY-MM-DD).")
    parser.add_argument(
        "--horizons",
        default="1,2,4,8",
        help="Comma-separated forward-return horizons in bars (default: 1,2,4,8).",
    )
    parser.add_argument(
        "--commission-per-order",
        type=float,
        default=DEFAULT_COMMISSION,
        help="Hypothetical commission per order side for net-return analysis (default: 0.01).",
    )
    parser.add_argument(
        "--slippage-per-share",
        type=float,
        default=DEFAULT_SLIPPAGE,
        help="Hypothetical slippage per share for net-return analysis (default: 0.05).",
    )
    parser.add_argument(
        "--position-size",
        type=float,
        default=DEFAULT_POSITION_SIZE,
        help="Position size used for net-return math (default: 1000).",
    )
    parser.add_argument(
        "--output-dir",
        help="Optional directory to save CSV/JSON summaries. Default: print-only.",
    )
    return parser.parse_args()


def _parse_horizons(raw: str) -> tuple[int, ...]:
    values = tuple(int(part.strip()) for part in raw.split(",") if part.strip())
    if not values or any(value <= 0 for value in values):
        raise ValueError("Horizons must be a comma-separated list of positive integers.")
    return tuple(sorted(set(values)))


def _load_runtime_payload(config_path: Path) -> dict[str, Any]:
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"Runtime config must be a JSON object: {config_path}")
    runtime = payload.get("runtime", payload)
    if not isinstance(runtime, dict):
        raise RuntimeError(f"Runtime config field 'runtime' must be a JSON object: {config_path}")
    source = payload.get("source")
    return {
        "payload": payload,
        "runtime": runtime,
        "source": source if isinstance(source, dict) else {},
    }


def _runtime_base_defaults(strategy_mode: str) -> StrategyConfig:
    return StrategyConfig(strategy_mode=normalize_strategy_mode(strategy_mode))


def build_audit_context(
    *,
    config_path: Path,
    runtime: dict[str, Any],
    source_dataset: str | None,
    dataset_override: str | None,
    symbols_override: list[str] | None,
    start_date: str | None,
    end_date: str | None,
    commission_per_order: float,
    slippage_per_share: float,
    position_size: float,
    output_dir: str | None,
    variant: str = VARIANT_LIVE_EFFECTIVE,
) -> AuditContext:
    if variant not in VARIANT_CHOICES:
        raise RuntimeError(f"Unsupported audit variant: {variant}")

    dataset_path = Path(dataset_override) if dataset_override else (Path(source_dataset) if source_dataset else None)
    if dataset_path is None:
        raise RuntimeError(
            "Dataset path not provided and config source.dataset is missing. "
            "Pass --dataset explicitly."
        )
    if not dataset_path.exists() or not dataset_path.is_dir():
        raise RuntimeError(f"Dataset path does not exist or is not a directory: {dataset_path}")

    runtime_mode = normalize_strategy_mode(str(runtime.get("strategy_mode", STRATEGY_MODE_HYBRID)))
    base = _runtime_base_defaults(runtime_mode)
    runtime_symbols = runtime.get("symbols")
    symbols = symbols_override if symbols_override else runtime_symbols
    if symbols is not None and not isinstance(symbols, list):
        raise RuntimeError("Runtime symbols must be a JSON list when present.")

    symbol_strategy_modes = runtime.get("symbol_strategy_modes")
    if symbol_strategy_modes is not None and not isinstance(symbol_strategy_modes, dict):
        raise RuntimeError("runtime.symbol_strategy_modes must be an object mapping symbol to strategy mode.")
    normalized_symbol_modes = {
        str(symbol).strip().upper(): normalize_strategy_mode(str(mode))
        for symbol, mode in (symbol_strategy_modes or {}).items()
        if str(symbol).strip()
    }

    missing_live_fields = tuple(
        field
        for field in (
            "vwap_z_entry_threshold",
            "vwap_z_stop_atr_multiple",
            "min_atr_percentile",
            "max_adx_threshold",
            "mean_reversion_trend_slope_filter",
            "trend_pullback_min_adx",
            "trend_pullback_min_slope",
            "trend_pullback_entry_threshold",
            "trend_pullback_min_atr_percentile",
            "trend_pullback_max_atr_percentile",
            "trend_pullback_exit_style",
            "trend_pullback_hold_bars",
            "trend_pullback_take_profit_pct",
            "trend_pullback_stop_pct",
            "trend_pullback_research_exit_fill",
            "momentum_breakout_lookback_bars",
            "momentum_breakout_entry_buffer_pct",
            "momentum_breakout_min_adx",
            "momentum_breakout_min_slope",
            "momentum_breakout_min_atr_percentile",
            "momentum_breakout_exit_style",
            "momentum_breakout_hold_bars",
            "momentum_breakout_stop_pct",
            "momentum_breakout_take_profit_pct",
            "volatility_expansion_lookback_bars",
            "volatility_expansion_entry_buffer_pct",
            "volatility_expansion_max_atr_percentile",
            "volatility_expansion_trend_filter",
            "volatility_expansion_min_slope",
            "volatility_expansion_use_volume_confirm",
            "volatility_expansion_exit_style",
            "volatility_expansion_hold_bars",
            "volatility_expansion_stop_pct",
            "volatility_expansion_take_profit_pct",
        )
        if field not in runtime
    )

    if variant == VARIANT_LIVE_EFFECTIVE:
        variant_defaults = {
            "mean_reversion_trend_slope_filter": base.mean_reversion_trend_slope_filter,
            "vwap_z_entry_threshold": base.vwap_z_entry_threshold,
            "vwap_z_stop_atr_multiple": base.vwap_z_stop_atr_multiple,
            "min_atr_percentile": base.min_atr_percentile,
            "max_adx_threshold": base.max_adx_threshold,
            "trend_pullback_min_adx": base.trend_pullback_min_adx,
            "trend_pullback_min_slope": base.trend_pullback_min_slope,
            "trend_pullback_entry_threshold": base.trend_pullback_entry_threshold,
            "trend_pullback_min_atr_percentile": base.trend_pullback_min_atr_percentile,
            "trend_pullback_max_atr_percentile": base.trend_pullback_max_atr_percentile,
            "trend_pullback_exit_style": base.trend_pullback_exit_style,
            "trend_pullback_hold_bars": base.trend_pullback_hold_bars,
            "trend_pullback_take_profit_pct": base.trend_pullback_take_profit_pct,
            "trend_pullback_stop_pct": base.trend_pullback_stop_pct,
            "trend_pullback_research_exit_fill": "next_open",
            "momentum_breakout_lookback_bars": base.momentum_breakout_lookback_bars,
            "momentum_breakout_entry_buffer_pct": base.momentum_breakout_entry_buffer_pct,
            "momentum_breakout_min_adx": base.momentum_breakout_min_adx,
            "momentum_breakout_min_slope": base.momentum_breakout_min_slope,
            "momentum_breakout_min_atr_percentile": base.momentum_breakout_min_atr_percentile,
            "momentum_breakout_exit_style": base.momentum_breakout_exit_style,
            "momentum_breakout_hold_bars": base.momentum_breakout_hold_bars,
            "momentum_breakout_stop_pct": base.momentum_breakout_stop_pct,
            "momentum_breakout_take_profit_pct": base.momentum_breakout_take_profit_pct,
            "volatility_expansion_lookback_bars": base.volatility_expansion_lookback_bars,
            "volatility_expansion_entry_buffer_pct": base.volatility_expansion_entry_buffer_pct,
            "volatility_expansion_max_atr_percentile": base.volatility_expansion_max_atr_percentile,
            "volatility_expansion_trend_filter": base.volatility_expansion_trend_filter,
            "volatility_expansion_min_slope": base.volatility_expansion_min_slope,
            "volatility_expansion_use_volume_confirm": base.volatility_expansion_use_volume_confirm,
            "volatility_expansion_exit_style": base.volatility_expansion_exit_style,
            "volatility_expansion_hold_bars": base.volatility_expansion_hold_bars,
            "volatility_expansion_stop_pct": base.volatility_expansion_stop_pct,
            "volatility_expansion_take_profit_pct": base.volatility_expansion_take_profit_pct,
        }
    else:
        variant_defaults = {
            "mean_reversion_trend_slope_filter": False,
            "vwap_z_entry_threshold": 0.0,
            "vwap_z_stop_atr_multiple": 2.0,
            "min_atr_percentile": 0.0,
            "max_adx_threshold": 0.0,
            "trend_pullback_min_adx": base.trend_pullback_min_adx,
            "trend_pullback_min_slope": base.trend_pullback_min_slope,
            "trend_pullback_entry_threshold": base.trend_pullback_entry_threshold,
            "trend_pullback_min_atr_percentile": base.trend_pullback_min_atr_percentile,
            "trend_pullback_max_atr_percentile": base.trend_pullback_max_atr_percentile,
            "trend_pullback_exit_style": base.trend_pullback_exit_style,
            "trend_pullback_hold_bars": base.trend_pullback_hold_bars,
            "trend_pullback_take_profit_pct": base.trend_pullback_take_profit_pct,
            "trend_pullback_stop_pct": base.trend_pullback_stop_pct,
            "trend_pullback_research_exit_fill": "next_open",
            "momentum_breakout_lookback_bars": base.momentum_breakout_lookback_bars,
            "momentum_breakout_entry_buffer_pct": base.momentum_breakout_entry_buffer_pct,
            "momentum_breakout_min_adx": base.momentum_breakout_min_adx,
            "momentum_breakout_min_slope": base.momentum_breakout_min_slope,
            "momentum_breakout_min_atr_percentile": base.momentum_breakout_min_atr_percentile,
            "momentum_breakout_exit_style": base.momentum_breakout_exit_style,
            "momentum_breakout_hold_bars": base.momentum_breakout_hold_bars,
            "momentum_breakout_stop_pct": base.momentum_breakout_stop_pct,
            "momentum_breakout_take_profit_pct": base.momentum_breakout_take_profit_pct,
            "volatility_expansion_lookback_bars": base.volatility_expansion_lookback_bars,
            "volatility_expansion_entry_buffer_pct": base.volatility_expansion_entry_buffer_pct,
            "volatility_expansion_max_atr_percentile": base.volatility_expansion_max_atr_percentile,
            "volatility_expansion_trend_filter": base.volatility_expansion_trend_filter,
            "volatility_expansion_min_slope": base.volatility_expansion_min_slope,
            "volatility_expansion_use_volume_confirm": base.volatility_expansion_use_volume_confirm,
            "volatility_expansion_exit_style": base.volatility_expansion_exit_style,
            "volatility_expansion_hold_bars": base.volatility_expansion_hold_bars,
            "volatility_expansion_stop_pct": base.volatility_expansion_stop_pct,
            "volatility_expansion_take_profit_pct": base.volatility_expansion_take_profit_pct,
        }

    backtest_kwargs = {
        "symbols": symbols,
        "start_date": start_date,
        "end_date": end_date,
        "strategy_mode": runtime_mode,
        "symbol_strategy_modes": normalized_symbol_modes or None,
        "sma_bars": int(runtime.get("sma_bars", DEFAULT_SMA_BARS) or DEFAULT_SMA_BARS),
        "commission": float(commission_per_order),
        "slippage": float(slippage_per_share),
        "position_size": float(position_size),
        "entry_threshold_pct": float(runtime.get("entry_threshold_pct", base.entry_threshold_pct) or base.entry_threshold_pct),
        "threshold_mode": normalize_threshold_mode(str(runtime.get("threshold_mode", base.threshold_mode))),
        "atr_multiple": float(runtime.get("atr_multiple", base.atr_multiple) or base.atr_multiple),
        "atr_percentile_threshold": float(
            runtime.get("atr_percentile_threshold", base.atr_percentile_threshold) or base.atr_percentile_threshold
        ),
        "time_window_mode": normalize_time_window_mode(str(runtime.get("time_window_mode", base.time_window_mode))),
        "regime_filter_enabled": bool(runtime.get("regime_filter_enabled", base.regime_filter_enabled)),
        "orb_filter_mode": normalize_orb_filter_mode(str(runtime.get("orb_filter_mode", base.orb_filter_mode))),
        "breakout_exit_style": normalize_breakout_exit_style(
            str(runtime.get("breakout_exit_style", base.breakout_exit_style))
        ),
        "breakout_tight_stop_fraction": float(
            runtime.get("breakout_tight_stop_fraction", base.breakout_tight_stop_fraction)
            or base.breakout_tight_stop_fraction
        ),
        "breakout_max_stop_pct": float(runtime.get("breakout_max_stop_pct", base.breakout_max_stop_pct) or base.breakout_max_stop_pct),
        "breakout_gap_pct_min": float(runtime.get("breakout_gap_pct_min", base.breakout_gap_pct_min) or base.breakout_gap_pct_min),
        "breakout_or_range_pct_min": float(
            runtime.get("breakout_or_range_pct_min", base.breakout_or_range_pct_min) or base.breakout_or_range_pct_min
        ),
        "mean_reversion_exit_style": normalize_mean_reversion_exit_style(
            str(runtime.get("mean_reversion_exit_style", base.mean_reversion_exit_style))
        ),
        "mean_reversion_max_atr_percentile": float(
            runtime.get("mean_reversion_max_atr_percentile", base.mean_reversion_max_atr_percentile)
            or base.mean_reversion_max_atr_percentile
        ),
        "mean_reversion_stop_pct": float(
            runtime.get("mean_reversion_stop_pct", base.mean_reversion_stop_pct) or base.mean_reversion_stop_pct
        ),
        "sma_stop_pct": float(runtime.get("sma_stop_pct", base.sma_stop_pct) or base.sma_stop_pct),
        "mean_reversion_trend_filter": bool(
            runtime.get("mean_reversion_trend_filter", base.mean_reversion_trend_filter)
        ),
        "mean_reversion_trend_slope_filter": bool(
            runtime.get("mean_reversion_trend_slope_filter", variant_defaults["mean_reversion_trend_slope_filter"])
        ),
        "bb_period": int(runtime.get("bb_period", base.bb_period) or base.bb_period),
        "bb_stddev_mult": float(runtime.get("bb_stddev_mult", base.bb_stddev_mult) or base.bb_stddev_mult),
        "bb_width_lookback": int(runtime.get("bb_width_lookback", base.bb_width_lookback) or base.bb_width_lookback),
        "bb_squeeze_quantile": float(
            runtime.get("bb_squeeze_quantile", base.bb_squeeze_quantile) or base.bb_squeeze_quantile
        ),
        "bb_slope_lookback": int(runtime.get("bb_slope_lookback", base.bb_slope_lookback) or base.bb_slope_lookback),
        "bb_use_volume_confirm": bool(runtime.get("bb_use_volume_confirm", base.bb_use_volume_confirm)),
        "bb_volume_mult": float(runtime.get("bb_volume_mult", base.bb_volume_mult) or base.bb_volume_mult),
        "bb_breakout_buffer_pct": float(
            runtime.get("bb_breakout_buffer_pct", base.bb_breakout_buffer_pct) or base.bb_breakout_buffer_pct
        ),
        "bb_min_mid_slope": float(runtime.get("bb_min_mid_slope", base.bb_min_mid_slope) or base.bb_min_mid_slope),
        "bb_trend_filter": bool(runtime.get("bb_trend_filter", base.bb_trend_filter)),
        "bb_exit_mode": normalize_bollinger_exit_mode(str(runtime.get("bb_exit_mode", base.bb_exit_mode))),
        "ml_probability_buy": float(runtime.get("ml_probability_buy", base.ml_probability_buy) or base.ml_probability_buy),
        "ml_probability_sell": float(runtime.get("ml_probability_sell", base.ml_probability_sell) or base.ml_probability_sell),
        "vwap_z_entry_threshold": float(
            runtime.get("vwap_z_entry_threshold", variant_defaults["vwap_z_entry_threshold"])
            or variant_defaults["vwap_z_entry_threshold"]
        ),
        "vwap_z_stop_atr_multiple": float(
            runtime.get("vwap_z_stop_atr_multiple", variant_defaults["vwap_z_stop_atr_multiple"])
            or variant_defaults["vwap_z_stop_atr_multiple"]
        ),
        "min_atr_percentile": float(
            runtime.get("min_atr_percentile", variant_defaults["min_atr_percentile"])
            or variant_defaults["min_atr_percentile"]
        ),
        "max_adx_threshold": float(
            runtime.get("max_adx_threshold", variant_defaults["max_adx_threshold"])
            or variant_defaults["max_adx_threshold"]
        ),
        "trend_pullback_min_adx": float(
            runtime.get("trend_pullback_min_adx", variant_defaults["trend_pullback_min_adx"])
            or variant_defaults["trend_pullback_min_adx"]
        ),
        "trend_pullback_min_slope": float(
            runtime.get("trend_pullback_min_slope", variant_defaults["trend_pullback_min_slope"])
            or variant_defaults["trend_pullback_min_slope"]
        ),
        "trend_pullback_entry_threshold": float(
            runtime.get("trend_pullback_entry_threshold", variant_defaults["trend_pullback_entry_threshold"])
            or variant_defaults["trend_pullback_entry_threshold"]
        ),
        "trend_pullback_min_atr_percentile": float(
            runtime.get("trend_pullback_min_atr_percentile", variant_defaults["trend_pullback_min_atr_percentile"])
            or variant_defaults["trend_pullback_min_atr_percentile"]
        ),
        "trend_pullback_max_atr_percentile": float(
            runtime.get("trend_pullback_max_atr_percentile", variant_defaults["trend_pullback_max_atr_percentile"])
            or variant_defaults["trend_pullback_max_atr_percentile"]
        ),
        "trend_pullback_exit_style": str(
            runtime.get("trend_pullback_exit_style", variant_defaults["trend_pullback_exit_style"])
        ),
        "trend_pullback_hold_bars": int(
            runtime.get("trend_pullback_hold_bars", variant_defaults["trend_pullback_hold_bars"])
        ),
        "trend_pullback_take_profit_pct": float(
            runtime.get("trend_pullback_take_profit_pct", variant_defaults["trend_pullback_take_profit_pct"])
            or variant_defaults["trend_pullback_take_profit_pct"]
        ),
        "trend_pullback_stop_pct": float(
            runtime.get("trend_pullback_stop_pct", variant_defaults["trend_pullback_stop_pct"])
            or variant_defaults["trend_pullback_stop_pct"]
        ),
        "trend_pullback_research_exit_fill": str(
            runtime.get("trend_pullback_research_exit_fill", variant_defaults["trend_pullback_research_exit_fill"])
        ),
        "momentum_breakout_lookback_bars": int(
            runtime.get("momentum_breakout_lookback_bars", variant_defaults["momentum_breakout_lookback_bars"])
        ),
        "momentum_breakout_entry_buffer_pct": float(
            runtime.get("momentum_breakout_entry_buffer_pct", variant_defaults["momentum_breakout_entry_buffer_pct"])
            or variant_defaults["momentum_breakout_entry_buffer_pct"]
        ),
        "momentum_breakout_min_adx": float(
            runtime.get("momentum_breakout_min_adx", variant_defaults["momentum_breakout_min_adx"])
            or variant_defaults["momentum_breakout_min_adx"]
        ),
        "momentum_breakout_min_slope": float(
            runtime.get("momentum_breakout_min_slope", variant_defaults["momentum_breakout_min_slope"])
            or variant_defaults["momentum_breakout_min_slope"]
        ),
        "momentum_breakout_min_atr_percentile": float(
            runtime.get("momentum_breakout_min_atr_percentile", variant_defaults["momentum_breakout_min_atr_percentile"])
            or variant_defaults["momentum_breakout_min_atr_percentile"]
        ),
        "momentum_breakout_exit_style": normalize_momentum_breakout_exit_style(
            str(runtime.get("momentum_breakout_exit_style", variant_defaults["momentum_breakout_exit_style"]))
        ),
        "momentum_breakout_hold_bars": int(
            runtime.get("momentum_breakout_hold_bars", variant_defaults["momentum_breakout_hold_bars"])
        ),
        "momentum_breakout_stop_pct": float(
            runtime.get("momentum_breakout_stop_pct", variant_defaults["momentum_breakout_stop_pct"])
            or variant_defaults["momentum_breakout_stop_pct"]
        ),
        "momentum_breakout_take_profit_pct": float(
            runtime.get("momentum_breakout_take_profit_pct", variant_defaults["momentum_breakout_take_profit_pct"])
            or variant_defaults["momentum_breakout_take_profit_pct"]
        ),
        "volatility_expansion_lookback_bars": int(
            runtime.get("volatility_expansion_lookback_bars", variant_defaults["volatility_expansion_lookback_bars"])
        ),
        "volatility_expansion_entry_buffer_pct": float(
            runtime.get("volatility_expansion_entry_buffer_pct", variant_defaults["volatility_expansion_entry_buffer_pct"])
            or variant_defaults["volatility_expansion_entry_buffer_pct"]
        ),
        "volatility_expansion_max_atr_percentile": float(
            runtime.get("volatility_expansion_max_atr_percentile", variant_defaults["volatility_expansion_max_atr_percentile"])
            or variant_defaults["volatility_expansion_max_atr_percentile"]
        ),
        "volatility_expansion_trend_filter": bool(
            runtime.get("volatility_expansion_trend_filter", variant_defaults["volatility_expansion_trend_filter"])
        ),
        "volatility_expansion_min_slope": float(
            runtime.get("volatility_expansion_min_slope", variant_defaults["volatility_expansion_min_slope"])
            or variant_defaults["volatility_expansion_min_slope"]
        ),
        "volatility_expansion_use_volume_confirm": bool(
            runtime.get("volatility_expansion_use_volume_confirm", variant_defaults["volatility_expansion_use_volume_confirm"])
        ),
        "volatility_expansion_exit_style": normalize_volatility_expansion_exit_style(
            str(runtime.get("volatility_expansion_exit_style", variant_defaults["volatility_expansion_exit_style"]))
        ),
        "volatility_expansion_hold_bars": int(
            runtime.get("volatility_expansion_hold_bars", variant_defaults["volatility_expansion_hold_bars"])
        ),
        "volatility_expansion_stop_pct": float(
            runtime.get("volatility_expansion_stop_pct", variant_defaults["volatility_expansion_stop_pct"])
            or variant_defaults["volatility_expansion_stop_pct"]
        ),
        "volatility_expansion_take_profit_pct": float(
            runtime.get("volatility_expansion_take_profit_pct", variant_defaults["volatility_expansion_take_profit_pct"])
            or variant_defaults["volatility_expansion_take_profit_pct"]
        ),
    }

    return AuditContext(
        dataset_path=dataset_path,
        config_path=config_path,
        runtime=runtime,
        backtest_kwargs=backtest_kwargs,
        symbols=symbols,
        output_dir=Path(output_dir) if output_dir else None,
        source_dataset=source_dataset,
        live_runtime_missing_fields=missing_live_fields,
    )


def _build_audit_context(args: argparse.Namespace) -> AuditContext:
    config_path = Path(args.config)
    if not config_path.exists():
        raise RuntimeError(f"Config path does not exist: {config_path}")

    payload = _load_runtime_payload(config_path)
    return build_audit_context(
        config_path=config_path,
        runtime=dict(payload["runtime"]),
        source_dataset=payload["source"].get("dataset") if isinstance(payload["source"].get("dataset"), str) else None,
        dataset_override=args.dataset,
        symbols_override=args.symbols,
        start_date=args.start_date,
        end_date=args.end_date,
        commission_per_order=args.commission_per_order,
        slippage_per_share=args.slippage_per_share,
        position_size=args.position_size,
        output_dir=args.output_dir,
        variant=VARIANT_LIVE_EFFECTIVE,
    )


def _time_bucket(ts: pd.Timestamp) -> str:
    ts_et = pd.Timestamp(ts).tz_convert("America/New_York")
    minutes = ts_et.hour * 60 + ts_et.minute
    if minutes < 10 * 60 + 30:
        return "open_30m"
    if minutes < 11 * 60 + 30:
        return "morning"
    if minutes < 13 * 60 + 30:
        return "midday"
    return "afternoon"


def classify_volatility_regime(atr_percentile: float | None) -> str:
    if atr_percentile is None:
        return "unknown"
    if atr_percentile < 33.33:
        return "low_vol"
    if atr_percentile < 66.67:
        return "mid_vol"
    return "high_vol"


def classify_trend_proxy(adx: float | None) -> str:
    if adx is None:
        return "unknown"
    if adx >= 25.0:
        return "trend"
    if adx < 20.0:
        return "range"
    return "mixed"


def _regime_label(value: bool | None) -> str:
    if value is True:
        return "bullish"
    if value is False:
        return "bearish"
    return "unknown"


def _uses_ml(strategy_modes: Iterable[str]) -> bool:
    return any(mode in {STRATEGY_MODE_ML, STRATEGY_MODE_HYBRID} for mode in strategy_modes)


def _min_history_bars(strategy: StrategyConfig, sma_bars: int) -> int:
    if strategy.strategy_mode in (STRATEGY_MODE_BREAKOUT, STRATEGY_MODE_ORB):
        return 2
    if strategy.strategy_mode == STRATEGY_MODE_MOMENTUM_BREAKOUT:
        return max(sma_bars, 55, strategy.momentum_breakout_lookback_bars + 1)
    if strategy.strategy_mode == STRATEGY_MODE_VOLATILITY_EXPANSION:
        return max(
            sma_bars,
            55,
            strategy.volatility_expansion_lookback_bars + 1,
            strategy.bb_period + strategy.bb_width_lookback,
            strategy.bb_period + strategy.bb_slope_lookback,
        )
    if strategy.strategy_mode in (STRATEGY_MODE_BOLLINGER_SQUEEZE, STRATEGY_MODE_HYBRID_BB_MR):
        return max(
            strategy.bb_period + strategy.bb_width_lookback,
            strategy.bb_period + strategy.bb_slope_lookback,
        )
    return sma_bars


def _signal_priority(
    strategy_mode: str,
    price: float,
    sma: float,
    opening_range_high: float | None,
    opening_range_low: float | None,
) -> float:
    if strategy_mode == STRATEGY_MODE_BREAKOUT and opening_range_high is not None and opening_range_low is not None:
        orb_range = max(0.0, opening_range_high - opening_range_low)
        return (price - opening_range_high) / max(orb_range, 1e-9)
    if strategy_mode in (STRATEGY_MODE_MEAN_REVERSION, STRATEGY_MODE_HYBRID_BB_MR):
        return max(0.0, (sma - price) / max(abs(sma), 1e-9))
    return 0.0


def compute_forward_return_pcts(
    *,
    entry_open: float,
    entry_fill: float,
    future_close: float,
    slippage: float,
    commission: float,
    position_size: float,
) -> tuple[float, float]:
    gross_return_pct = ((future_close - entry_open) / entry_open * 100.0) if entry_open > 0 else 0.0
    if position_size <= 0 or entry_fill <= 0:
        return gross_return_pct, 0.0
    shares = position_size / entry_fill
    exit_proceeds = shares * max(future_close - slippage, 0.0)
    pnl = exit_proceeds - commission - (position_size + commission)
    net_return_pct = pnl / position_size * 100.0
    return gross_return_pct, net_return_pct


def _prepare_inputs_and_state(context: AuditContext) -> tuple[Any, Any]:
    inputs = _prepare_backtest_inputs(
        dataset_path=context.dataset_path,
        symbols=context.backtest_kwargs["symbols"],
        start_date=context.backtest_kwargs["start_date"],
        end_date=context.backtest_kwargs["end_date"],
        strategy_mode=context.backtest_kwargs["strategy_mode"],
        symbol_strategy_modes=context.backtest_kwargs["symbol_strategy_modes"],
        time_window_mode=context.backtest_kwargs["time_window_mode"],
        threshold_mode=context.backtest_kwargs["threshold_mode"],
        orb_filter_mode=context.backtest_kwargs["orb_filter_mode"],
        breakout_exit_style=context.backtest_kwargs["breakout_exit_style"],
        breakout_tight_stop_fraction=context.backtest_kwargs["breakout_tight_stop_fraction"],
        breakout_max_stop_pct=context.backtest_kwargs["breakout_max_stop_pct"],
        breakout_gap_pct_min=context.backtest_kwargs["breakout_gap_pct_min"],
        breakout_or_range_pct_min=context.backtest_kwargs["breakout_or_range_pct_min"],
        mean_reversion_exit_style=context.backtest_kwargs["mean_reversion_exit_style"],
        mean_reversion_max_atr_percentile=context.backtest_kwargs["mean_reversion_max_atr_percentile"],
        mean_reversion_stop_pct=context.backtest_kwargs["mean_reversion_stop_pct"],
        sma_stop_pct=context.backtest_kwargs["sma_stop_pct"],
        mean_reversion_trend_filter=context.backtest_kwargs["mean_reversion_trend_filter"],
        mean_reversion_trend_slope_filter=context.backtest_kwargs["mean_reversion_trend_slope_filter"],
        ml_probability_buy=context.backtest_kwargs["ml_probability_buy"],
        ml_probability_sell=context.backtest_kwargs["ml_probability_sell"],
        entry_threshold_pct=context.backtest_kwargs["entry_threshold_pct"],
        atr_multiple=context.backtest_kwargs["atr_multiple"],
        atr_percentile_threshold=context.backtest_kwargs["atr_percentile_threshold"],
        regime_filter_enabled=context.backtest_kwargs["regime_filter_enabled"],
        bb_period=context.backtest_kwargs["bb_period"],
        bb_stddev_mult=context.backtest_kwargs["bb_stddev_mult"],
        bb_width_lookback=context.backtest_kwargs["bb_width_lookback"],
        bb_squeeze_quantile=context.backtest_kwargs["bb_squeeze_quantile"],
        bb_slope_lookback=context.backtest_kwargs["bb_slope_lookback"],
        bb_use_volume_confirm=context.backtest_kwargs["bb_use_volume_confirm"],
        bb_volume_mult=context.backtest_kwargs["bb_volume_mult"],
        bb_breakout_buffer_pct=context.backtest_kwargs["bb_breakout_buffer_pct"],
        bb_min_mid_slope=context.backtest_kwargs["bb_min_mid_slope"],
        bb_trend_filter=context.backtest_kwargs["bb_trend_filter"],
        bb_exit_mode=context.backtest_kwargs["bb_exit_mode"],
        vwap_z_entry_threshold=context.backtest_kwargs["vwap_z_entry_threshold"],
        vwap_z_stop_atr_multiple=context.backtest_kwargs["vwap_z_stop_atr_multiple"],
        min_atr_percentile=context.backtest_kwargs["min_atr_percentile"],
        max_adx_threshold=context.backtest_kwargs["max_adx_threshold"],
        trend_pullback_min_adx=context.backtest_kwargs["trend_pullback_min_adx"],
        trend_pullback_min_slope=context.backtest_kwargs["trend_pullback_min_slope"],
        trend_pullback_entry_threshold=context.backtest_kwargs["trend_pullback_entry_threshold"],
        trend_pullback_min_atr_percentile=context.backtest_kwargs["trend_pullback_min_atr_percentile"],
        trend_pullback_max_atr_percentile=context.backtest_kwargs["trend_pullback_max_atr_percentile"],
        trend_pullback_exit_style=context.backtest_kwargs["trend_pullback_exit_style"],
        trend_pullback_hold_bars=context.backtest_kwargs["trend_pullback_hold_bars"],
        trend_pullback_take_profit_pct=context.backtest_kwargs["trend_pullback_take_profit_pct"],
        trend_pullback_stop_pct=context.backtest_kwargs["trend_pullback_stop_pct"],
        trend_pullback_research_exit_fill=context.backtest_kwargs["trend_pullback_research_exit_fill"],
        momentum_breakout_lookback_bars=context.backtest_kwargs["momentum_breakout_lookback_bars"],
        momentum_breakout_entry_buffer_pct=context.backtest_kwargs["momentum_breakout_entry_buffer_pct"],
        momentum_breakout_min_adx=context.backtest_kwargs["momentum_breakout_min_adx"],
        momentum_breakout_min_slope=context.backtest_kwargs["momentum_breakout_min_slope"],
        momentum_breakout_min_atr_percentile=context.backtest_kwargs["momentum_breakout_min_atr_percentile"],
        momentum_breakout_exit_style=context.backtest_kwargs["momentum_breakout_exit_style"],
        momentum_breakout_hold_bars=context.backtest_kwargs["momentum_breakout_hold_bars"],
        momentum_breakout_stop_pct=context.backtest_kwargs["momentum_breakout_stop_pct"],
        momentum_breakout_take_profit_pct=context.backtest_kwargs["momentum_breakout_take_profit_pct"],
        volatility_expansion_lookback_bars=context.backtest_kwargs["volatility_expansion_lookback_bars"],
        volatility_expansion_entry_buffer_pct=context.backtest_kwargs["volatility_expansion_entry_buffer_pct"],
        volatility_expansion_max_atr_percentile=context.backtest_kwargs["volatility_expansion_max_atr_percentile"],
        volatility_expansion_trend_filter=context.backtest_kwargs["volatility_expansion_trend_filter"],
        volatility_expansion_min_slope=context.backtest_kwargs["volatility_expansion_min_slope"],
        volatility_expansion_use_volume_confirm=context.backtest_kwargs["volatility_expansion_use_volume_confirm"],
        volatility_expansion_exit_style=context.backtest_kwargs["volatility_expansion_exit_style"],
        volatility_expansion_hold_bars=context.backtest_kwargs["volatility_expansion_hold_bars"],
        volatility_expansion_stop_pct=context.backtest_kwargs["volatility_expansion_stop_pct"],
        volatility_expansion_take_profit_pct=context.backtest_kwargs["volatility_expansion_take_profit_pct"],
    )
    state = _initialize_simulation_state(inputs, starting_capital=10000.0, position_size=context.backtest_kwargs["position_size"])
    strategy_modes = [strategy.config.strategy_mode for strategy in inputs.symbol_strategies.values()]
    if _uses_ml(strategy_modes):
        payload = _load_offline_model_payload()
        state.ml_signals = _precompute_ml_signals(
            state,
            inputs.symbol_strategies,
            payload,
            context.backtest_kwargs["ml_probability_buy"],
            context.backtest_kwargs["ml_probability_sell"],
        )
    else:
        state.ml_signals = {symbol: [inputs.dummy_ml] * len(state.close_arrs[symbol]) for symbol in inputs.symbols}
    return inputs, state


def _infer_entry_reason(
    *,
    strategy: Any,
    price: float,
    sma: float,
    ml_signal: Any,
    atr_pct: float | None,
    atr_percentile: float | None,
    time_window_open: bool,
    bullish_regime: bool | None,
    opening_range_high: float | None,
    opening_range_low: float | None,
    volume_ratio: float | None,
    volatility_ratio: float | None,
    gap_pct: float | None,
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
    bar_high: float | None,
    bar_low: float | None,
    bar_open: float | None,
) -> str:
    config = strategy.config
    mode = normalize_strategy_mode(config.strategy_mode)

    if mode == STRATEGY_MODE_MEAN_REVERSION:
        if vwap is not None and atr_pct is not None:
            atr = atr_pct * price
            if atr <= 0:
                return "mean_reversion_degenerate_atr"
            z = (price - vwap) / atr
            if z > -config.vwap_z_entry_threshold:
                return "mean_reversion_vwap_z_not_deep_enough"
            if not time_window_open:
                return "time_window_blocked"
            if not strategy._entry_allowed(atr_percentile):
                return "atr_percentile_blocked"
            if not strategy._vwap_mr_min_atr_allowed(atr_percentile):
                return "mean_reversion_vwap_min_atr_blocked"
            if not strategy._vwap_mr_adx_allowed(adx):
                return "mean_reversion_vwap_adx_blocked"
            if not strategy._mean_reversion_entry_allowed(atr_percentile):
                return "mean_reversion_atr_cap_blocked"
            if not strategy._regime_allows_entry(bullish_regime):
                return "regime_filter_blocked"
            return "mean_reversion_vwap_entry"

        reversion_entry_price = strategy._reversion_threshold_price(sma, atr_pct)
        if reversion_entry_price is None:
            return "mean_reversion_threshold_unavailable"
        if price >= reversion_entry_price:
            return "mean_reversion_price_not_below_threshold"
        if not strategy._entry_allowed(atr_percentile):
            return "atr_percentile_blocked"
        if not strategy._mean_reversion_entry_allowed(atr_percentile):
            return "mean_reversion_atr_cap_blocked"
        if not time_window_open:
            return "time_window_blocked"
        if not strategy._regime_allows_entry(bullish_regime):
            return "regime_filter_blocked"
        if config.mean_reversion_trend_filter and trend_sma is not None and price < trend_sma:
            return "mean_reversion_trend_filter_blocked"
        if config.mean_reversion_trend_slope_filter and trend_sma_slope is not None and trend_sma_slope < 0:
            return "mean_reversion_trend_slope_filter_blocked"
        return "mean_reversion_sma_entry"

    if mode == STRATEGY_MODE_BREAKOUT:
        orb_range = (
            opening_range_high - opening_range_low
            if opening_range_high is not None and opening_range_low is not None
            else None
        )
        if opening_range_high is None or orb_range is None or orb_range <= 0:
            return "breakout_or_not_ready"
        if price <= opening_range_high:
            return "breakout_not_above_or_high"
        if not strategy._entry_allowed(atr_percentile):
            return "atr_percentile_blocked"
        if not time_window_open:
            return "time_window_blocked"
        if not strategy._regime_allows_entry(bullish_regime):
            return "regime_filter_blocked"
        if not strategy._orb_filter_allows_entry(volume_ratio, volatility_ratio):
            return "orb_filter_blocked"
        if not strategy._breakout_gap_allows_entry(gap_pct):
            return "breakout_gap_filter_blocked"
        if not strategy._breakout_or_range_allows_entry(opening_range_high, opening_range_low):
            return "breakout_or_range_filter_blocked"
        return "breakout_entry"

    if mode == STRATEGY_MODE_SMA:
        threshold_price = strategy._entry_threshold_price(sma, atr_pct)
        if threshold_price is None:
            return "threshold_unavailable"
        if price <= threshold_price:
            return "sma_not_above_threshold"
        if not strategy._entry_allowed(atr_percentile):
            return "atr_percentile_blocked"
        if not time_window_open:
            return "time_window_blocked"
        if not strategy._regime_allows_entry(bullish_regime):
            return "regime_filter_blocked"
        return "sma_entry"

    if mode == STRATEGY_MODE_ML:
        if ml_signal.probability_up < ml_signal.buy_threshold:
            return "ml_buy_threshold_not_met"
        if not strategy._entry_allowed(atr_percentile):
            return "atr_percentile_blocked"
        if not time_window_open:
            return "time_window_blocked"
        if not strategy._regime_allows_entry(bullish_regime):
            return "regime_filter_blocked"
        return "ml_only_threshold_passed"

    if mode == STRATEGY_MODE_HYBRID:
        if price <= sma:
            return "hybrid_sma_not_bullish"
        if ml_signal.probability_up < ml_signal.buy_threshold:
            return "hybrid_ml_threshold_not_met"
        if not strategy._entry_allowed(atr_percentile):
            return "atr_percentile_blocked"
        if not time_window_open:
            return "time_window_blocked"
        if not strategy._regime_allows_entry(bullish_regime):
            return "regime_filter_blocked"
        return "hybrid_ml_and_sma_confirmed"

    if mode == STRATEGY_MODE_BOLLINGER_SQUEEZE:
        if bb_prev_squeeze is not True:
            return "bollinger_not_after_squeeze"
        min_mid_slope = max(0.0, config.bb_min_mid_slope)
        bullish_bias = bb_bias == "bullish" and bb_mid_slope is not None and bb_mid_slope >= min_mid_slope
        if not bullish_bias:
            if bb_bias == "bearish" and bb_breakout_down is True:
                return "bollinger_short_signal_blocked"
            return "bollinger_bias_not_bullish"
        breakout_buffer = max(0.0, config.bb_breakout_buffer_pct)
        if bb_breakout_up is not True or bb_upper is None or price <= (bb_upper * (1.0 + breakout_buffer)):
            return "bollinger_breakout_not_confirmed"
        if not strategy._bollinger_volume_confirmed(bb_volume_confirm):
            return "bollinger_volume_filter_blocked"
        if not strategy._bollinger_trend_filter_ok(price, trend_sma):
            return "bollinger_trend_filter_blocked"
        if not strategy._entry_allowed(atr_percentile):
            return "atr_percentile_blocked"
        if not time_window_open:
            return "time_window_blocked"
        if not strategy._regime_allows_entry(bullish_regime):
            return "regime_filter_blocked"
        return "bollinger_breakout_long"

    if mode == STRATEGY_MODE_HYBRID_BB_MR:
        active_branch = "bollinger_breakout" if bb_prev_squeeze is True else "mean_reversion"
        if active_branch == "bollinger_breakout":
            min_mid_slope = max(0.0, config.bb_min_mid_slope)
            bullish_bias = bb_bias == "bullish" and bb_mid_slope is not None and bb_mid_slope >= min_mid_slope
            if not bullish_bias:
                if bb_bias == "bearish" and bb_breakout_down is True:
                    return "bollinger_short_signal_blocked"
                return "bollinger_bias_not_bullish"
            breakout_buffer = max(0.0, config.bb_breakout_buffer_pct)
            if bb_breakout_up is not True or bb_upper is None or price <= (bb_upper * (1.0 + breakout_buffer)):
                return "bollinger_breakout_not_confirmed"
            if not strategy._bollinger_volume_confirmed(bb_volume_confirm):
                return "bollinger_volume_filter_blocked"
            if not strategy._bollinger_trend_filter_ok(price, trend_sma):
                return "bollinger_trend_filter_blocked"
            if not strategy._entry_allowed(atr_percentile):
                return "atr_percentile_blocked"
            if not time_window_open:
                return "time_window_blocked"
            if not strategy._regime_allows_entry(bullish_regime):
                return "regime_filter_blocked"
            return "bollinger_breakout_long"

        if vwap is not None and atr_pct is not None:
            atr = atr_pct * price
            if atr <= 0:
                return "mean_reversion_degenerate_atr"
            z = (price - vwap) / atr
            if z > -config.vwap_z_entry_threshold:
                return "mean_reversion_vwap_z_not_deep_enough"
            if not time_window_open:
                return "time_window_blocked"
            if not strategy._entry_allowed(atr_percentile):
                return "atr_percentile_blocked"
            if not strategy._vwap_mr_min_atr_allowed(atr_percentile):
                return "mean_reversion_vwap_min_atr_blocked"
            if not strategy._vwap_mr_adx_allowed(adx):
                return "mean_reversion_vwap_adx_blocked"
            if not strategy._mean_reversion_entry_allowed(atr_percentile):
                return "mean_reversion_atr_cap_blocked"
            if not strategy._regime_allows_entry(bullish_regime):
                return "regime_filter_blocked"
            return "mean_reversion_vwap_entry"

        reversion_entry_price = strategy._reversion_threshold_price(sma, atr_pct)
        if reversion_entry_price is None:
            return "mean_reversion_threshold_unavailable"
        if price >= reversion_entry_price:
            return "mean_reversion_price_not_below_threshold"
        if not strategy._entry_allowed(atr_percentile):
            return "atr_percentile_blocked"
        if not strategy._mean_reversion_entry_allowed(atr_percentile):
            return "mean_reversion_atr_cap_blocked"
        if not time_window_open:
            return "time_window_blocked"
        if not strategy._regime_allows_entry(bullish_regime):
            return "regime_filter_blocked"
        if config.mean_reversion_trend_filter and trend_sma is not None and price < trend_sma:
            return "mean_reversion_trend_filter_blocked"
        if config.mean_reversion_trend_slope_filter and trend_sma_slope is not None and trend_sma_slope < 0:
            return "mean_reversion_trend_slope_filter_blocked"
        return "mean_reversion_sma_entry"

    if mode == STRATEGY_MODE_ORB:
        if opening_range_high is None or opening_range_low is None:
            return "orb_range_not_ready"
        orb_range = opening_range_high - opening_range_low
        orb_range_pct = orb_range / opening_range_low if opening_range_low > 0 else 0.0
        if not (config.orb_min_or_size_pct <= orb_range_pct <= config.orb_max_or_size_pct):
            return "orb_range_size_filter_blocked"
        entry_level = opening_range_high * (1.0 + config.orb_entry_buffer_pct)
        if price <= entry_level:
            return "orb_not_above_entry_buffer"
        if not time_window_open:
            return "time_window_blocked"
        if not strategy._regime_allows_entry(bullish_regime):
            return "regime_filter_blocked"
        if not strategy._orb_filter_allows_entry(volume_ratio, volatility_ratio):
            return "orb_filter_blocked"
        return "orb_entry"

    if mode == STRATEGY_MODE_WICK_FADE:
        if bar_high is None or bar_low is None or bar_open is None:
            return "wick_fade_missing_ohlc"
        bar_range = bar_high - bar_low
        if bar_range <= 0:
            return "wick_fade_zero_range"
        bar_range_pct = bar_range / max(price, 1e-9)
        if bar_range_pct < config.wick_fade_min_range_pct:
            return "wick_fade_range_filter_blocked"
        body_bottom = min(bar_open, price)
        lower_wick_ratio = (body_bottom - bar_low) / bar_range
        close_position = (price - bar_low) / bar_range
        if lower_wick_ratio < config.wick_fade_min_lower_wick_ratio:
            return "wick_fade_lower_wick_filter_blocked"
        if close_position < config.wick_fade_min_close_position:
            return "wick_fade_close_position_blocked"
        if not time_window_open:
            return "time_window_blocked"
        if not strategy._entry_allowed(atr_percentile):
            return "atr_percentile_blocked"
        if not strategy._regime_allows_entry(bullish_regime):
            return "regime_filter_blocked"
        return "wick_fade_entry"

    return "unclassified_hold"


def evaluate_raw_entry_signals(
    *,
    inputs: Any,
    state: Any,
    sma_bars: int,
    time_window_mode: str,
    slippage: float,
    commission: float,
    position_size: float,
    horizons: tuple[int, ...],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    evaluations: list[dict[str, Any]] = []
    events: list[dict[str, Any]] = []

    for symbol in inputs.symbols:
        strategy = inputs.symbol_strategies[symbol]
        config = strategy.config
        closes = state.close_arrs[symbol]
        opens = state.symbols_dfs[symbol]["open"].tolist()
        volumes = state.volume_arrs[symbol]
        timestamps = state.symbols_dfs[symbol]["timestamp"].tolist()
        min_history = _min_history_bars(config, sma_bars)
        breakout_day_gap_pct = 0.0
        last_day = None

        for p in range(len(closes) - 1):
            if p < min_history - 1:
                continue

            current_day = state.day_key_arrs[symbol][p]
            if current_day != last_day:
                last_day = current_day
                if p > 0:
                    prev_close = closes[p - 1]
                    breakout_day_gap_pct = ((opens[p] - prev_close) / prev_close) if prev_close > 0 else 0.0
                else:
                    breakout_day_gap_pct = 0.0

            price = closes[p]
            sma = _mean(closes[p - sma_bars + 1: p + 1]) if p >= sma_bars - 1 else closes[p]
            trend_sma = _mean(closes[p - 49: p + 1]) if p >= 49 else None
            trend_sma_slope = (
                trend_sma - _mean(closes[p - 54: p - 4])
                if p >= 54 and trend_sma is not None
                else None
            )
            recent_volumes = volumes[max(0, p - 19): p + 1]
            avg_volume = _mean(recent_volumes) if recent_volumes else 0.0
            volume_ratio = (volumes[p] / avg_volume) if avg_volume > 0 else None
            atr_pct = state.atr_pct_arrs[symbol][p]
            recent_atr = [value for value in state.atr_pct_arrs[symbol][max(0, p - 19): p + 1] if value is not None]
            avg_atr_pct = _mean(recent_atr) if recent_atr else None
            volatility_ratio = (
                atr_pct / avg_atr_pct
                if atr_pct is not None and avg_atr_pct is not None and avg_atr_pct > 0
                else None
            )
            breakout_lookback = 0
            if config.strategy_mode == STRATEGY_MODE_MOMENTUM_BREAKOUT:
                breakout_lookback = config.momentum_breakout_lookback_bars
            elif config.strategy_mode == STRATEGY_MODE_VOLATILITY_EXPANSION:
                breakout_lookback = config.volatility_expansion_lookback_bars
            recent_breakout_high = (
                max(state.high_arrs[symbol][p - breakout_lookback: p])
                if breakout_lookback > 0 and p >= breakout_lookback
                else None
            )
            next_entry_timestamp = pd.Timestamp(timestamps[p + 1])
            ml_signal = state.ml_signals[symbol][p] if state.ml_signals[symbol][p] is not None else inputs.dummy_ml
            details = strategy.decide_action_details(
                price,
                sma,
                ml_signal,
                False,
                atr_pct,
                state.atr_percentile_arrs[symbol][p],
                time_window_open=is_entry_window_open(next_entry_timestamp, time_window_mode),
                bullish_regime=state.bullish_regime_arrs[symbol][p],
                opening_range_high=state.opening_range_high_arrs[symbol][p],
                opening_range_low=state.opening_range_low_arrs[symbol][p],
                position_entry_price=None,
                volume_ratio=volume_ratio,
                volatility_ratio=volatility_ratio,
                trailing_stop_price=None,
                mean_reversion_target_price=None,
                breakout_already_taken=False,
                effective_stop_price=None,
                gap_pct=breakout_day_gap_pct,
                trend_sma=trend_sma,
                trend_sma_slope=trend_sma_slope,
                vwap=state.vwap_arrs[symbol][p] if config.vwap_z_entry_threshold > 0 else None,
                adx=state.adx_arrs[symbol][p] if strategy_requires_adx(config) else None,
                bb_middle=state.bb_middle_arrs[symbol][p],
                bb_upper=state.bb_upper_arrs[symbol][p],
                bb_lower=state.bb_lower_arrs[symbol][p],
                bb_prev_squeeze=state.bb_squeeze_arrs[symbol][p - 1] if p > 0 else None,
                bb_mid_slope=state.bb_mid_slope_arrs[symbol][p],
                bb_bias=state.bb_bias_arrs[symbol][p],
                bb_breakout_up=state.bb_breakout_up_arrs[symbol][p],
                bb_breakout_down=state.bb_breakout_down_arrs[symbol][p],
                bb_volume_confirm=state.bb_volume_confirm_arrs[symbol][p],
                hybrid_entry_branch=None,
                bar_high=state.high_arrs[symbol][p],
                bar_low=state.low_arrs[symbol][p],
                bar_open=opens[p],
                wick_fade_stop=0.0,
                wick_fade_target=0.0,
                wick_fade_bars_held=0,
                recent_breakout_high=recent_breakout_high,
                volatility_expansion_bars_held=0,
            )
            resolved_reason = details.reason or _infer_entry_reason(
                strategy=strategy,
                price=price,
                sma=sma,
                ml_signal=ml_signal,
                atr_pct=atr_pct,
                atr_percentile=state.atr_percentile_arrs[symbol][p],
                time_window_open=is_entry_window_open(next_entry_timestamp, time_window_mode),
                bullish_regime=state.bullish_regime_arrs[symbol][p],
                opening_range_high=state.opening_range_high_arrs[symbol][p],
                opening_range_low=state.opening_range_low_arrs[symbol][p],
                volume_ratio=volume_ratio,
                volatility_ratio=volatility_ratio,
                gap_pct=breakout_day_gap_pct,
                trend_sma=trend_sma,
                trend_sma_slope=trend_sma_slope,
                vwap=state.vwap_arrs[symbol][p] if config.vwap_z_entry_threshold > 0 else None,
                adx=state.adx_arrs[symbol][p] if strategy_requires_adx(config) else None,
                bb_middle=state.bb_middle_arrs[symbol][p],
                bb_upper=state.bb_upper_arrs[symbol][p],
                bb_lower=state.bb_lower_arrs[symbol][p],
                bb_prev_squeeze=state.bb_squeeze_arrs[symbol][p - 1] if p > 0 else None,
                bb_mid_slope=state.bb_mid_slope_arrs[symbol][p],
                bb_bias=state.bb_bias_arrs[symbol][p],
                bb_breakout_up=state.bb_breakout_up_arrs[symbol][p],
                bb_breakout_down=state.bb_breakout_down_arrs[symbol][p],
                bb_volume_confirm=state.bb_volume_confirm_arrs[symbol][p],
                bar_high=state.high_arrs[symbol][p],
                bar_low=state.low_arrs[symbol][p],
                bar_open=opens[p],
            )

            signal_ts = pd.Timestamp(timestamps[p]).tz_convert("America/New_York")
            record = {
                "symbol": symbol,
                "strategy_mode": config.strategy_mode,
                "signal_ts": signal_ts,
                "entry_ts": next_entry_timestamp.tz_convert("America/New_York"),
                "signal_hour": signal_ts.strftime("%H:00"),
                "time_bucket": _time_bucket(signal_ts),
                "month": signal_ts.strftime("%Y-%m"),
                "date": signal_ts.strftime("%Y-%m-%d"),
                "action": details.action,
                "reason": resolved_reason,
                "hybrid_branch_active": details.hybrid_branch_active,
                "hybrid_regime_branch": details.hybrid_regime_branch,
                "mr_signal": details.mr_signal,
                "price": price,
                "sma": sma,
                "trend_sma": trend_sma,
                "trend_sma_slope": trend_sma_slope,
                "atr_pct": atr_pct,
                "atr_percentile": state.atr_percentile_arrs[symbol][p],
                "volatility_regime": classify_volatility_regime(state.atr_percentile_arrs[symbol][p]),
                "adx": state.adx_arrs[symbol][p],
                "trend_proxy": classify_trend_proxy(state.adx_arrs[symbol][p]),
                "bullish_regime": _regime_label(state.bullish_regime_arrs[symbol][p]),
                "priority": _signal_priority(
                    config.strategy_mode,
                    price,
                    sma,
                    state.opening_range_high_arrs[symbol][p],
                    state.opening_range_low_arrs[symbol][p],
                ),
            }
            evaluations.append(record)

            if details.action != "BUY":
                continue

            entry_open = float(opens[p + 1])
            entry_fill = entry_open + slippage
            event = dict(record)
            event["entry_open"] = entry_open
            event["entry_fill"] = entry_fill
            for horizon in horizons:
                target_index = p + horizon
                if target_index >= len(closes):
                    event[f"fwd_{horizon}b_gross_pct"] = None
                    event[f"fwd_{horizon}b_net_pct"] = None
                    continue
                gross_pct, net_pct = compute_forward_return_pcts(
                    entry_open=entry_open,
                    entry_fill=entry_fill,
                    future_close=closes[target_index],
                    slippage=slippage,
                    commission=commission,
                    position_size=position_size,
                )
                event[f"fwd_{horizon}b_gross_pct"] = gross_pct
                event[f"fwd_{horizon}b_net_pct"] = net_pct
            events.append(event)

    return pd.DataFrame(evaluations), pd.DataFrame(events)


def summarize_forward_returns(
    signals_df: pd.DataFrame,
    *,
    group_cols: list[str],
    horizons: tuple[int, ...],
) -> pd.DataFrame:
    base_cols = group_cols + [
        "horizon_bars",
        "sample_count",
        "avg_gross_return_pct",
        "median_gross_return_pct",
        "gross_win_rate_pct",
        "avg_net_return_pct",
        "median_net_return_pct",
        "net_win_rate_pct",
    ]
    if signals_df.empty:
        return pd.DataFrame(columns=base_cols)

    rows: list[dict[str, Any]] = []
    grouped = [((), signals_df)] if not group_cols else signals_df.groupby(group_cols, dropna=False, sort=True)
    for group_key, group_df in grouped:
        key_values = group_key if isinstance(group_key, tuple) else (group_key,)
        key_map = {col: key_values[idx] for idx, col in enumerate(group_cols)}
        for horizon in horizons:
            gross_col = f"fwd_{horizon}b_gross_pct"
            net_col = f"fwd_{horizon}b_net_pct"
            gross = pd.to_numeric(group_df[gross_col], errors="coerce").dropna()
            net = pd.to_numeric(group_df[net_col], errors="coerce").dropna()
            rows.append({
                **key_map,
                "horizon_bars": horizon,
                "sample_count": int(len(gross)),
                "avg_gross_return_pct": float(gross.mean()) if not gross.empty else 0.0,
                "median_gross_return_pct": float(gross.median()) if not gross.empty else 0.0,
                "gross_win_rate_pct": float((gross > 0).mean() * 100.0) if not gross.empty else 0.0,
                "avg_net_return_pct": float(net.mean()) if not net.empty else 0.0,
                "median_net_return_pct": float(net.median()) if not net.empty else 0.0,
                "net_win_rate_pct": float((net > 0).mean() * 100.0) if not net.empty else 0.0,
            })
    return pd.DataFrame(rows)


def _frame_text(df: pd.DataFrame, *, max_rows: int | None = None) -> str:
    if df.empty:
        return "(none)"
    frame = df.head(max_rows) if max_rows is not None else df
    return frame.to_string(index=False)


def _top_counts(df: pd.DataFrame, column: str, *, max_rows: int = 20) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=[column, "count"])
    return (
        df.groupby(column, dropna=False)
        .size()
        .reset_index(name="count")
        .sort_values(["count", column], ascending=[False, True])
        .head(max_rows)
    )


def _signal_frequency_by_symbol(signals_df: pd.DataFrame, total_dates: int) -> pd.DataFrame:
    columns = ["symbol", "signals", "signals_per_day"]
    if signals_df.empty:
        return pd.DataFrame(columns=columns)
    grouped = signals_df.groupby("symbol").size().reset_index(name="signals")
    grouped["signals_per_day"] = grouped["signals"] / max(total_dates, 1)
    return grouped.sort_values(["signals", "symbol"], ascending=[False, True])


def _backtest_exit_summary(result: dict[str, Any]) -> pd.DataFrame:
    rows = [
        {
            "symbol": symbol,
            "realized_pnl": metrics.get("realized_pnl", 0.0),
            "expectancy": metrics.get("expectancy", 0.0),
            "win_rate": metrics.get("win_rate", 0.0),
            "trades": metrics.get("total_trades", 0),
            "avg_return_per_trade_pct": metrics.get("avg_return_per_trade_pct", 0.0),
        }
        for symbol, metrics in result.get("per_symbol", {}).items()
    ]
    return pd.DataFrame(rows).sort_values(["realized_pnl", "symbol"], ascending=[False, True]) if rows else pd.DataFrame(
        columns=["symbol", "realized_pnl", "expectancy", "win_rate", "trades", "avg_return_per_trade_pct"]
    )


def _hypothesis_summary(context: AuditContext) -> list[str]:
    mode = context.backtest_kwargs["strategy_mode"]
    sma_bars = context.backtest_kwargs["sma_bars"]
    threshold = context.backtest_kwargs["entry_threshold_pct"]
    threshold_mode = context.backtest_kwargs["threshold_mode"]
    mr_exit = context.backtest_kwargs["mean_reversion_exit_style"]
    mr_atr_cap = context.backtest_kwargs["mean_reversion_max_atr_percentile"]
    vwap_z = context.backtest_kwargs["vwap_z_entry_threshold"]
    lines = [f"Active/default strategy mode: {mode}"]
    if context.backtest_kwargs["symbol_strategy_modes"]:
        override_text = ", ".join(
            f"{symbol}:{value}" for symbol, value in sorted(context.backtest_kwargs["symbol_strategy_modes"].items())
        )
        lines.append(f"Per-symbol overrides: {override_text}")
    if mode == STRATEGY_MODE_MEAN_REVERSION:
        if vwap_z > 0:
            lines.append(
                "Hypothesis: buy oversold intraday pullbacks when price is materially below VWAP by a z-score threshold, "
                "then exit on reversion back toward VWAP."
            )
        else:
            threshold_text = (
                f"{context.backtest_kwargs['atr_multiple']:.2f} ATR below the {sma_bars}-bar SMA"
                if threshold_mode == "atr_multiple"
                else f"{threshold * 100:.2f}% below the {sma_bars}-bar SMA"
            )
            lines.append(
                "Hypothesis: buy short-term pullbacks inside a broader intraday mean-reversion process and exit when "
                f"price snaps back to the mean. Current trigger: roughly {threshold_text}; exit style: {mr_exit}."
            )
        if mr_atr_cap > 0:
            lines.append(f"Entry filter: skip the highest-volatility bars above the {mr_atr_cap:.1f} ATR percentile.")
    elif mode == STRATEGY_MODE_BREAKOUT:
        lines.append("Hypothesis: buy opening-range breakouts that continue intraday.")
    elif mode == STRATEGY_MODE_BOLLINGER_SQUEEZE:
        lines.append("Hypothesis: buy post-squeeze Bollinger expansion when trend, slope, and volume align.")
    elif mode == STRATEGY_MODE_HYBRID_BB_MR:
        lines.append("Hypothesis: trade Bollinger breakouts after squeezes, otherwise fall back to mean reversion.")
    elif mode == STRATEGY_MODE_SMA:
        lines.append("Hypothesis: buy upside trend continuation once price clears the SMA threshold.")
    elif mode == STRATEGY_MODE_ML:
        lines.append("Hypothesis: use the offline logistic model probability as the primary directional edge.")
    elif mode == STRATEGY_MODE_ORB:
        lines.append("Hypothesis: buy opening-range breakout continuation with one trade per symbol per day.")
    elif mode == STRATEGY_MODE_WICK_FADE:
        lines.append("Hypothesis: fade intrabar downside rejection when a large lower wick signals reversal demand.")
    return lines


def _print_strategy_summary(context: AuditContext) -> None:
    print("\n=== Strategy Summary ===")
    print(f"Config path: {context.config_path}")
    print(f"Dataset:     {context.dataset_path}")
    if context.source_dataset:
        print(f"Config source.dataset: {context.source_dataset}")
    print(f"Symbols:     {', '.join(context.symbols) if context.symbols else 'dataset default'}")
    for line in _hypothesis_summary(context):
        print(f"- {line}")
    print("- Runtime/backtest knobs:")
    for key in (
        "sma_bars",
        "strategy_mode",
        "entry_threshold_pct",
        "threshold_mode",
        "time_window_mode",
        "regime_filter_enabled",
        "atr_percentile_threshold",
        "mean_reversion_exit_style",
        "mean_reversion_max_atr_percentile",
        "mean_reversion_stop_pct",
        "mean_reversion_trend_filter",
        "mean_reversion_trend_slope_filter",
        "vwap_z_entry_threshold",
        "vwap_z_stop_atr_multiple",
        "min_atr_percentile",
        "max_adx_threshold",
        "bb_exit_mode",
        "bb_trend_filter",
    ):
        print(f"  {key}: {context.backtest_kwargs[key]}")
    if context.live_runtime_missing_fields:
        print(
            "- Live/runtime divergence warning: config does not specify "
            f"{', '.join(context.live_runtime_missing_fields)}. "
            "The live bot constructor omits these fields, so live may inherit StrategyConfig defaults while a plain "
            "backtest would not unless they are passed explicitly. This audit applies the live-effective defaults."
        )


def _print_signal_generation_summary(evaluations_df: pd.DataFrame, signals_df: pd.DataFrame) -> None:
    print("\n=== Signal Generation ===")
    print(f"Bars evaluated: {len(evaluations_df)}")
    print(f"Signals fired:  {len(signals_df)}")
    print(f"Long signals:   {len(signals_df)}")
    print(f"Short signals:  0 (current code does not execute short entries)")
    print("\nRejections by reason:")
    print(_frame_text(_top_counts(evaluations_df[evaluations_df["action"] != "BUY"], "reason")))
    unique_days = signals_df["date"].nunique() if not signals_df.empty else evaluations_df["date"].nunique()
    print("\nSignal frequency by symbol:")
    print(_frame_text(_signal_frequency_by_symbol(signals_df, unique_days)))
    print("\nSignal frequency by hour:")
    print(_frame_text(_top_counts(signals_df, "signal_hour")))
    print("\nSignal frequency by time bucket:")
    print(_frame_text(_top_counts(signals_df, "time_bucket")))


def _print_forward_return_sections(signals_df: pd.DataFrame, horizons: tuple[int, ...]) -> dict[str, pd.DataFrame]:
    outputs = {
        "overall": summarize_forward_returns(signals_df, group_cols=[], horizons=horizons),
        "by_symbol": summarize_forward_returns(signals_df, group_cols=["symbol"], horizons=horizons),
        "by_month": summarize_forward_returns(signals_df, group_cols=["month"], horizons=horizons),
        "by_volatility": summarize_forward_returns(signals_df, group_cols=["volatility_regime"], horizons=horizons),
        "by_trend_proxy": summarize_forward_returns(signals_df, group_cols=["trend_proxy"], horizons=horizons),
        "by_time_bucket": summarize_forward_returns(signals_df, group_cols=["time_bucket"], horizons=horizons),
    }
    print("\n=== Forward Returns After Entry ===")
    print("\nOverall:")
    print(_frame_text(outputs["overall"]))
    print("\nBy symbol:")
    print(_frame_text(outputs["by_symbol"], max_rows=50))
    print("\nBy month:")
    print(_frame_text(outputs["by_month"], max_rows=24))
    print("\nBy volatility regime:")
    print(_frame_text(outputs["by_volatility"], max_rows=24))
    print("\nBy trend/range proxy:")
    print(_frame_text(outputs["by_trend_proxy"], max_rows=24))
    print("\nBy time bucket:")
    print(_frame_text(outputs["by_time_bucket"], max_rows=24))
    return outputs


def _print_exit_diagnostics(backtest_result: dict[str, Any]) -> pd.DataFrame:
    per_symbol = _backtest_exit_summary(backtest_result)
    print("\n=== Current Exit Diagnostics ===")
    print(
        "Raw signal quality above uses independent entry events. The table below is the realized backtest outcome "
        "under current exit rules, holding constraints, next-bar execution, slippage, and commission."
    )
    print(
        f"Overall realized expectancy: {backtest_result.get('expectancy', 0.0):.4f} | "
        f"win_rate: {backtest_result.get('win_rate', 0.0):.2f}% | "
        f"total_trades: {backtest_result.get('total_trades', 0)} | "
        f"realized_pnl: {backtest_result.get('realized_pnl', 0.0):.2f}"
    )
    print("\nPer symbol realized summary:")
    print(_frame_text(per_symbol, max_rows=50))
    branch_stats = backtest_result.get("hybrid_branch_stats") or {}
    if branch_stats:
        branch_df = pd.DataFrame(
            [{"branch": branch, **stats} for branch, stats in branch_stats.items()]
        ).sort_values("realized_pnl", ascending=False)
        print("\nHybrid branch realized summary:")
        print(_frame_text(branch_df))
    return per_symbol


def _save_outputs(
    output_dir: Path,
    *,
    context: AuditContext,
    evaluations_df: pd.DataFrame,
    signals_df: pd.DataFrame,
    forward_tables: dict[str, pd.DataFrame],
    backtest_result: dict[str, Any],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "edge_audit_context.json").write_text(
        json.dumps(
            {
                "dataset": str(context.dataset_path),
                "config": str(context.config_path),
                "backtest_kwargs": {
                    key: (str(value) if isinstance(value, Path) else value)
                    for key, value in context.backtest_kwargs.items()
                },
                "live_runtime_missing_fields": list(context.live_runtime_missing_fields),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    evaluations_df.to_csv(output_dir / "entry_evaluations.csv", index=False)
    signals_df.to_csv(output_dir / "entry_signals.csv", index=False)
    for name, table in forward_tables.items():
        table.to_csv(output_dir / f"forward_returns_{name}.csv", index=False)
    pd.DataFrame(backtest_result.get("trades", [])).to_csv(output_dir / "realized_trades.csv", index=False)
    pd.DataFrame(
        [{"symbol": symbol, **metrics} for symbol, metrics in backtest_result.get("per_symbol", {}).items()]
    ).to_csv(output_dir / "realized_per_symbol.csv", index=False)


def main() -> None:
    args = parse_args()
    horizons = _parse_horizons(args.horizons)
    context = _build_audit_context(args)
    _print_strategy_summary(context)

    inputs, state = _prepare_inputs_and_state(context)
    evaluations_df, signals_df = evaluate_raw_entry_signals(
        inputs=inputs,
        state=state,
        sma_bars=context.backtest_kwargs["sma_bars"],
        time_window_mode=context.backtest_kwargs["time_window_mode"],
        slippage=context.backtest_kwargs["slippage"],
        commission=context.backtest_kwargs["commission"],
        position_size=context.backtest_kwargs["position_size"],
        horizons=horizons,
    )
    _print_signal_generation_summary(evaluations_df, signals_df)
    forward_tables = _print_forward_return_sections(signals_df, horizons)

    backtest_result = run_backtest(
        context.dataset_path,
        **context.backtest_kwargs,
    )
    _print_exit_diagnostics(backtest_result)

    if context.output_dir is not None:
        _save_outputs(
            context.output_dir,
            context=context,
            evaluations_df=evaluations_df,
            signals_df=signals_df,
            forward_tables=forward_tables,
            backtest_result=backtest_result,
        )
        print(f"\nSaved audit outputs to: {context.output_dir}")


if __name__ == "__main__":
    main()

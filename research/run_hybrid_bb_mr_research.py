from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from backtest_runner import run_backtest


PROJECT_ROOT = Path(__file__).resolve().parent
RESULTS_ROOT = PROJECT_ROOT / "results" / "experiments"
LIVE_CONFIG_PATH = PROJECT_ROOT / "config" / "live_config.json"

DEFAULT_BB_SQUEEZE_QUANTILES = (0.20, 0.30, 0.40)
DEFAULT_BB_USE_VOLUME_CONFIRM = (True, False)
DEFAULT_BB_VOLUME_MULTS = (1.0, 1.2, 1.5)
DEFAULT_BB_SLOPE_LOOKBACKS = (1, 3, 5)


@dataclass(frozen=True)
class ResearchRunSpec:
    name: str
    strategy_mode: str
    bb_squeeze_quantile: float | None = None
    bb_use_volume_confirm: bool | None = None
    bb_volume_mult: float | None = None
    bb_slope_lookback: int | None = None
    bb_breakout_buffer_pct: float | None = None
    bb_min_mid_slope: float | None = None
    bb_trend_filter: bool | None = None
    is_live_reference: bool = False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a focused hybrid_bb_mr research sweep with BB-branch participation reporting."
    )
    parser.add_argument("--dataset", required=True, help="Path to dataset directory.")
    parser.add_argument("--symbols", nargs="*", help="Optional symbol override.")
    parser.add_argument("--start-date", help="Optional inclusive start date (YYYY-MM-DD).")
    parser.add_argument("--end-date", help="Optional inclusive end date (YYYY-MM-DD).")
    parser.add_argument("--output-dir", help="Optional output folder. Default: timestamped folder in results/experiments.")
    parser.add_argument("--starting-capital", type=float, default=10000.0)
    parser.add_argument("--position-size", type=float, default=1000.0)
    parser.add_argument("--top-n", type=int, default=10, help="Leaderboard rows to export and print.")
    parser.add_argument(
        "--bb-squeeze-quantiles",
        default="0.20,0.30,0.40",
        help="Comma-separated squeeze quantiles for the hybrid grid, e.g. 0.20,0.30,0.40,0.50,0.60",
    )
    parser.add_argument(
        "--bb-use-volume-confirm-options",
        default="true,false",
        help="Comma-separated bool values for BB volume confirmation, e.g. true,false",
    )
    parser.add_argument(
        "--bb-volume-mults",
        default="1.0,1.2,1.5",
        help="Comma-separated BB volume multipliers for runs where volume confirmation is enabled.",
    )
    parser.add_argument(
        "--bb-slope-lookbacks",
        default="1,3,5",
        help="Comma-separated BB middle-band slope lookbacks, e.g. 1,3,5",
    )
    parser.add_argument(
        "--bb-breakout-buffer-pcts",
        default="0.0",
        help="Comma-separated BB breakout buffer percentages, e.g. 0.0,0.001,0.002",
    )
    parser.add_argument(
        "--bb-min-mid-slopes",
        default="0.0",
        help="Comma-separated minimum BB middle-band slopes, e.g. 0.0,0.01,0.02",
    )
    parser.add_argument(
        "--bb-trend-filter-options",
        default="false",
        help="Comma-separated bool values for BB trend filter, e.g. false,true",
    )
    return parser.parse_args()


def _timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def _sanitize_name(value: str) -> str:
    allowed = []
    for ch in value:
        allowed.append(ch if ch.isalnum() or ch in ("-", "_") else "_")
    return "".join(allowed).strip("_") or "hybrid_bb_mr_research"


def build_output_dir(output_dir: str | None) -> Path:
    if output_dir:
        return _ensure_dir(Path(output_dir))
    return _ensure_dir(RESULTS_ROOT / f"hybrid_bb_mr_research_{_timestamp()}")


def load_live_runtime_defaults(config_path: Path = LIVE_CONFIG_PATH) -> dict[str, Any]:
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    runtime = payload.get("runtime")
    if not isinstance(runtime, dict):
        raise RuntimeError(f"Live config is missing runtime defaults: {config_path}")
    return {
        "sma_bars": int(runtime.get("sma_bars", 20) or 20),
        "entry_threshold_pct": float(runtime.get("entry_threshold_pct", 0.001) or 0.001),
        "threshold_mode": str(runtime.get("threshold_mode", "static_pct") or "static_pct"),
        "atr_multiple": float(runtime.get("atr_multiple", 1.0) or 1.0),
        "atr_percentile_threshold": float(runtime.get("atr_percentile_threshold", 0.0) or 0.0),
        "time_window_mode": str(runtime.get("time_window_mode", "full_day") or "full_day"),
        "regime_filter_enabled": bool(runtime.get("regime_filter_enabled", False)),
        "mean_reversion_exit_style": str(runtime.get("mean_reversion_exit_style", "sma") or "sma"),
        "mean_reversion_max_atr_percentile": float(
            runtime.get("mean_reversion_max_atr_percentile", 0.0) or 0.0
        ),
        "mean_reversion_stop_pct": float(runtime.get("mean_reversion_stop_pct", 0.0) or 0.0),
        "mean_reversion_trend_filter": bool(runtime.get("mean_reversion_trend_filter", False)),
        "mean_reversion_trend_slope_filter": bool(runtime.get("mean_reversion_trend_slope_filter", False)),
        "bb_period": int(runtime.get("bb_period", 20) or 20),
        "bb_stddev_mult": float(runtime.get("bb_stddev_mult", 2.0) or 2.0),
        "bb_width_lookback": int(runtime.get("bb_width_lookback", 100) or 100),
        "bb_squeeze_quantile": float(runtime.get("bb_squeeze_quantile", 0.20) or 0.20),
        "bb_slope_lookback": int(runtime.get("bb_slope_lookback", 3) or 3),
        "bb_use_volume_confirm": bool(runtime.get("bb_use_volume_confirm", True)),
        "bb_volume_mult": float(runtime.get("bb_volume_mult", 1.2) or 1.2),
        "bb_breakout_buffer_pct": float(runtime.get("bb_breakout_buffer_pct", 0.0) or 0.0),
        "bb_min_mid_slope": float(runtime.get("bb_min_mid_slope", 0.0) or 0.0),
        "bb_trend_filter": bool(runtime.get("bb_trend_filter", False)),
        "bb_exit_mode": str(runtime.get("bb_exit_mode", "middle_band") or "middle_band"),
    }


def _parse_csv_values(raw: str, cast: type[float] | type[int]) -> tuple[float, ...] | tuple[int, ...]:
    values = [item.strip() for item in raw.split(",") if item.strip()]
    return tuple(cast(item) for item in values)


def _parse_bool_csv_values(raw: str) -> tuple[bool, ...]:
    normalized: list[bool] = []
    for item in raw.split(","):
        value = item.strip().lower()
        if not value:
            continue
        if value in {"true", "1", "yes", "on"}:
            normalized.append(True)
        elif value in {"false", "0", "no", "off"}:
            normalized.append(False)
        else:
            raise ValueError(f"Unsupported boolean list value: {item}")
    return tuple(normalized)


def build_hybrid_grid(
    squeeze_quantiles: tuple[float, ...] = DEFAULT_BB_SQUEEZE_QUANTILES,
    use_volume_confirm_values: tuple[bool, ...] = DEFAULT_BB_USE_VOLUME_CONFIRM,
    volume_mult_values: tuple[float, ...] = DEFAULT_BB_VOLUME_MULTS,
    slope_lookbacks: tuple[int, ...] = DEFAULT_BB_SLOPE_LOOKBACKS,
    breakout_buffer_pcts: tuple[float, ...] = (0.0,),
    min_mid_slopes: tuple[float, ...] = (0.0,),
    trend_filter_values: tuple[bool, ...] = (False,),
) -> list[ResearchRunSpec]:
    grid: list[ResearchRunSpec] = []
    for squeeze_quantile in squeeze_quantiles:
        for use_volume_confirm in use_volume_confirm_values:
            effective_volume_mults = volume_mult_values if use_volume_confirm else (None,)
            for volume_mult in effective_volume_mults:
                for slope_lookback in slope_lookbacks:
                    for breakout_buffer_pct in breakout_buffer_pcts:
                        for min_mid_slope in min_mid_slopes:
                            for trend_filter in trend_filter_values:
                                grid.append(
                                    ResearchRunSpec(
                                        name=(
                                            f"hybrid_q{int(squeeze_quantile * 100):02d}_"
                                            f"vol{'on' if use_volume_confirm else 'off'}_"
                                            f"vm{volume_mult if volume_mult is not None else 'na'}_"
                                            f"slope{slope_lookback}_"
                                            f"buf{breakout_buffer_pct}_"
                                            f"minslope{min_mid_slope}_"
                                            f"trend{'on' if trend_filter else 'off'}"
                                        ),
                                        strategy_mode="hybrid_bb_mr",
                                        bb_squeeze_quantile=squeeze_quantile,
                                        bb_use_volume_confirm=use_volume_confirm,
                                        bb_volume_mult=volume_mult,
                                        bb_slope_lookback=slope_lookback,
                                        bb_breakout_buffer_pct=breakout_buffer_pct,
                                        bb_min_mid_slope=min_mid_slope,
                                        bb_trend_filter=trend_filter,
                                    )
                                )
    return grid


def build_research_specs(
    live_defaults: dict[str, Any],
    squeeze_quantiles: tuple[float, ...] = DEFAULT_BB_SQUEEZE_QUANTILES,
    use_volume_confirm_values: tuple[bool, ...] = DEFAULT_BB_USE_VOLUME_CONFIRM,
    volume_mult_values: tuple[float, ...] = DEFAULT_BB_VOLUME_MULTS,
    slope_lookbacks: tuple[int, ...] = DEFAULT_BB_SLOPE_LOOKBACKS,
    breakout_buffer_pcts: tuple[float, ...] = (0.0,),
    min_mid_slopes: tuple[float, ...] = (0.0,),
    trend_filter_values: tuple[bool, ...] = (False,),
) -> list[ResearchRunSpec]:
    specs = [
        ResearchRunSpec(
            name="mean_reversion_reference",
            strategy_mode="mean_reversion",
            is_live_reference=True,
        ),
        ResearchRunSpec(
            name="bollinger_squeeze_reference",
            strategy_mode="bollinger_squeeze",
            bb_squeeze_quantile=live_defaults["bb_squeeze_quantile"],
            bb_use_volume_confirm=live_defaults["bb_use_volume_confirm"],
            bb_volume_mult=live_defaults["bb_volume_mult"],
            bb_slope_lookback=live_defaults["bb_slope_lookback"],
            bb_breakout_buffer_pct=live_defaults["bb_breakout_buffer_pct"],
            bb_min_mid_slope=live_defaults["bb_min_mid_slope"],
            bb_trend_filter=live_defaults["bb_trend_filter"],
            is_live_reference=True,
        ),
    ]
    for spec in build_hybrid_grid(
        squeeze_quantiles=squeeze_quantiles,
        use_volume_confirm_values=use_volume_confirm_values,
        volume_mult_values=volume_mult_values,
        slope_lookbacks=slope_lookbacks,
        breakout_buffer_pcts=breakout_buffer_pcts,
        min_mid_slopes=min_mid_slopes,
        trend_filter_values=trend_filter_values,
    ):
        specs.append(
            ResearchRunSpec(
                **{
                    **spec.__dict__,
                    "is_live_reference": (
                        spec.bb_squeeze_quantile == live_defaults["bb_squeeze_quantile"]
                        and spec.bb_use_volume_confirm == live_defaults["bb_use_volume_confirm"]
                        and spec.bb_volume_mult == live_defaults["bb_volume_mult"]
                        and spec.bb_slope_lookback == live_defaults["bb_slope_lookback"]
                        and spec.bb_breakout_buffer_pct == live_defaults["bb_breakout_buffer_pct"]
                        and spec.bb_min_mid_slope == live_defaults["bb_min_mid_slope"]
                        and spec.bb_trend_filter == live_defaults["bb_trend_filter"]
                    ),
                }
            )
        )
    return specs


def flatten_hybrid_branch_stats(result: dict[str, Any]) -> dict[str, Any]:
    branch_stats = result.get("hybrid_branch_stats", {}) or {}
    mr_stats = branch_stats.get("mean_reversion", {})
    bb_stats = branch_stats.get("bollinger_breakout", {})
    mr_trades = int(mr_stats.get("total_trades", 0) or 0)
    bb_trades = int(bb_stats.get("total_trades", 0) or 0)
    total_branch_trades = mr_trades + bb_trades
    bb_trade_share_pct = (bb_trades / total_branch_trades * 100.0) if total_branch_trades else 0.0
    mr_trade_share_pct = (mr_trades / total_branch_trades * 100.0) if total_branch_trades else 0.0
    branch_balance_score = (
        max(0.0, 1.0 - abs(bb_trade_share_pct - 50.0) / 50.0)
        if total_branch_trades
        else 0.0
    )
    if bb_trades == 0:
        bb_participation = "none"
    elif bb_trades < 3 or bb_trade_share_pct < 10.0:
        bb_participation = "rare"
    elif bb_trade_share_pct < 25.0:
        bb_participation = "active"
    else:
        bb_participation = "meaningful"
    return {
        "branch_total_trades": total_branch_trades,
        "mr_branch_trades": mr_trades,
        "bb_branch_trades": bb_trades,
        "bb_branch_triggered": bb_trades > 0,
        "mr_branch_trade_share_pct": mr_trade_share_pct,
        "bb_branch_trade_share_pct": bb_trade_share_pct,
        "branch_balance_score": branch_balance_score,
        "bb_branch_participation": bb_participation,
        "bb_branch_dead_weight": bb_trades == 0,
        "bb_branch_positive_contribution": (bb_stats.get("realized_pnl", 0.0) or 0.0) > 0 and bb_trades > 0,
        "mr_branch_win_rate": mr_stats.get("win_rate"),
        "bb_branch_win_rate": bb_stats.get("win_rate"),
        "mr_branch_pnl": mr_stats.get("realized_pnl"),
        "bb_branch_pnl": bb_stats.get("realized_pnl"),
        "mr_branch_avg_pnl_per_trade": mr_stats.get("avg_pnl_per_trade"),
        "bb_branch_avg_pnl_per_trade": bb_stats.get("avg_pnl_per_trade"),
        "mr_branch_avg_hold_bars": mr_stats.get("avg_hold_bars"),
        "bb_branch_avg_hold_bars": bb_stats.get("avg_hold_bars"),
        "mr_branch_avg_winning_trade": mr_stats.get("avg_winning_trade"),
        "bb_branch_avg_winning_trade": bb_stats.get("avg_winning_trade"),
        "mr_branch_avg_losing_trade": mr_stats.get("avg_losing_trade"),
        "bb_branch_avg_losing_trade": bb_stats.get("avg_losing_trade"),
    }


def result_to_row(spec: ResearchRunSpec, result: dict[str, Any], dataset: Path, symbols: list[str] | None) -> dict[str, Any]:
    row = {
        "run_name": spec.name,
        "strategy_mode": spec.strategy_mode,
        "dataset": str(dataset),
        "symbols": ",".join(symbols) if symbols else "",
        "start_date": result.get("start_date"),
        "end_date": result.get("end_date"),
        "is_live_reference": spec.is_live_reference,
        "bb_squeeze_quantile": spec.bb_squeeze_quantile,
        "bb_use_volume_confirm": spec.bb_use_volume_confirm,
        "bb_volume_mult": spec.bb_volume_mult,
        "bb_slope_lookback": spec.bb_slope_lookback,
        "bb_breakout_buffer_pct": spec.bb_breakout_buffer_pct,
        "bb_min_mid_slope": spec.bb_min_mid_slope,
        "bb_trend_filter": spec.bb_trend_filter,
        "total_return_pct": result.get("total_return_pct"),
        "realized_pnl": result.get("realized_pnl"),
        "profit_factor": result.get("profit_factor"),
        "sharpe_ratio": result.get("sharpe_ratio"),
        "max_drawdown_pct": result.get("max_drawdown_pct"),
        "total_trades": result.get("total_trades"),
        "win_rate": result.get("win_rate"),
        "expectancy": result.get("expectancy"),
        "avg_winning_trade": result.get("avg_winning_trade"),
        "avg_losing_trade": result.get("avg_losing_trade"),
        "trades_per_day": result.get("trades_per_day"),
        "time_window_mode": result.get("time_window_mode"),
        "threshold_mode": result.get("threshold_mode"),
        "sma_bars": result.get("sma_bars"),
        "entry_threshold_pct": result.get("entry_threshold_pct"),
        "mean_reversion_exit_style": result.get("mean_reversion_exit_style"),
        "mean_reversion_max_atr_percentile": result.get("mean_reversion_max_atr_percentile"),
    }
    row.update(flatten_hybrid_branch_stats(result))
    return row


def add_baseline_comparison_columns(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    mr_baseline = df[df["strategy_mode"] == "mean_reversion"].iloc[0]
    df = df.copy()
    df["baseline_mr_realized_pnl"] = mr_baseline["realized_pnl"]
    df["baseline_mr_profit_factor"] = mr_baseline["profit_factor"]
    df["baseline_mr_sharpe_ratio"] = mr_baseline["sharpe_ratio"]
    df["baseline_mr_max_drawdown_pct"] = mr_baseline["max_drawdown_pct"]
    df["delta_vs_mr_realized_pnl"] = df["realized_pnl"] - mr_baseline["realized_pnl"]
    df["delta_vs_mr_profit_factor"] = df["profit_factor"] - mr_baseline["profit_factor"]
    df["delta_vs_mr_sharpe_ratio"] = df["sharpe_ratio"] - mr_baseline["sharpe_ratio"]
    df["delta_vs_mr_max_drawdown_pct"] = df["max_drawdown_pct"] - mr_baseline["max_drawdown_pct"]
    return df


def add_usefulness_columns(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    df = df.copy()
    if "bb_branch_triggered" not in df.columns:
        df["bb_branch_triggered"] = df["bb_branch_trades"].fillna(0) > 0
    if "bb_branch_dead_weight" not in df.columns:
        df["bb_branch_dead_weight"] = df["bb_branch_trades"].fillna(0) == 0
    if "bb_branch_positive_contribution" not in df.columns:
        bb_branch_pnl = df["bb_branch_pnl"] if "bb_branch_pnl" in df.columns else pd.Series(0.0, index=df.index)
        df["bb_branch_positive_contribution"] = (
            (df["bb_branch_trades"].fillna(0) > 0) & (bb_branch_pnl.fillna(0.0) > 0)
        )
    hybrid_mask = df["strategy_mode"] == "hybrid_bb_mr"
    df["score_branch_participation"] = 0.0
    df["score_branch_balance"] = 0.0
    df["score_delta_profit_factor"] = 0.0
    df["score_delta_sharpe"] = 0.0
    df["score_delta_pnl"] = 0.0
    df["score_drawdown_penalty"] = 0.0
    df["score_trade_count_penalty"] = 0.0
    df.loc[hybrid_mask, "score_branch_participation"] = (
        df.loc[hybrid_mask, "bb_branch_trade_share_pct"].clip(upper=30.0) * 0.5
        - (df.loc[hybrid_mask, "bb_branch_trades"] == 0).astype(float) * 40.0
    )
    df.loc[hybrid_mask, "score_branch_balance"] = df.loc[hybrid_mask, "branch_balance_score"] * 10.0
    df.loc[hybrid_mask, "score_delta_profit_factor"] = df.loc[hybrid_mask, "delta_vs_mr_profit_factor"] * 20.0
    df.loc[hybrid_mask, "score_delta_sharpe"] = df.loc[hybrid_mask, "delta_vs_mr_sharpe_ratio"] * 10.0
    df.loc[hybrid_mask, "score_delta_pnl"] = df.loc[hybrid_mask, "delta_vs_mr_realized_pnl"] * 0.5
    df.loc[hybrid_mask, "score_drawdown_penalty"] = (
        df.loc[hybrid_mask, "delta_vs_mr_max_drawdown_pct"].clip(lower=0.0) * 2.0
    )
    df.loc[hybrid_mask, "score_trade_count_penalty"] = (
        (df.loc[hybrid_mask, "total_trades"] < 5).astype(float) * 10.0
    )
    df["usefulness_score"] = (
        df["score_branch_participation"]
        + df["score_branch_balance"]
        + df["score_delta_profit_factor"]
        + df["score_delta_sharpe"]
        + df["score_delta_pnl"]
        - df["score_drawdown_penalty"]
        - df["score_trade_count_penalty"]
    )
    return df


def build_ranked_hybrid_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    hybrid_df = df[df["strategy_mode"] == "hybrid_bb_mr"].copy()
    if hybrid_df.empty:
        return hybrid_df
    return hybrid_df.sort_values(
        by=[
            "bb_branch_triggered",
            "usefulness_score",
            "bb_branch_trades",
            "delta_vs_mr_profit_factor",
            "delta_vs_mr_realized_pnl",
            "max_drawdown_pct",
        ],
        ascending=[False, False, False, False, False, True],
        na_position="last",
    ).reset_index(drop=True)


def run_research_sweep(
    dataset: Path,
    symbols: list[str] | None,
    start_date: str | None,
    end_date: str | None,
    starting_capital: float,
    position_size: float,
    squeeze_quantiles: tuple[float, ...],
    use_volume_confirm_values: tuple[bool, ...],
    volume_mult_values: tuple[float, ...],
    slope_lookbacks: tuple[int, ...],
    breakout_buffer_pcts: tuple[float, ...],
    min_mid_slopes: tuple[float, ...],
    trend_filter_values: tuple[bool, ...],
) -> pd.DataFrame:
    live_defaults = load_live_runtime_defaults()
    common_kwargs = dict(
        sma_bars=live_defaults["sma_bars"],
        entry_threshold_pct=live_defaults["entry_threshold_pct"],
        threshold_mode=live_defaults["threshold_mode"],
        atr_multiple=live_defaults["atr_multiple"],
        atr_percentile_threshold=live_defaults["atr_percentile_threshold"],
        time_window_mode=live_defaults["time_window_mode"],
        regime_filter_enabled=live_defaults["regime_filter_enabled"],
        mean_reversion_exit_style=live_defaults["mean_reversion_exit_style"],
        mean_reversion_max_atr_percentile=live_defaults["mean_reversion_max_atr_percentile"],
        mean_reversion_stop_pct=live_defaults["mean_reversion_stop_pct"],
        mean_reversion_trend_filter=live_defaults["mean_reversion_trend_filter"],
        mean_reversion_trend_slope_filter=live_defaults["mean_reversion_trend_slope_filter"],
        bb_period=live_defaults["bb_period"],
        bb_stddev_mult=live_defaults["bb_stddev_mult"],
        bb_width_lookback=live_defaults["bb_width_lookback"],
        bb_exit_mode=live_defaults["bb_exit_mode"],
        start_date=start_date,
        end_date=end_date,
        starting_capital=starting_capital,
        position_size=position_size,
    )
    rows: list[dict[str, Any]] = []
    for spec in build_research_specs(
        live_defaults,
        squeeze_quantiles=squeeze_quantiles,
        use_volume_confirm_values=use_volume_confirm_values,
        volume_mult_values=volume_mult_values,
        slope_lookbacks=slope_lookbacks,
        breakout_buffer_pcts=breakout_buffer_pcts,
        min_mid_slopes=min_mid_slopes,
        trend_filter_values=trend_filter_values,
    ):
        print(
            f"Running {spec.name} strategy={spec.strategy_mode} "
            f"q={spec.bb_squeeze_quantile} vol_confirm={spec.bb_use_volume_confirm} "
            f"vol_mult={spec.bb_volume_mult} slope={spec.bb_slope_lookback}",
            flush=True,
        )
        result = run_backtest(
            dataset_path=dataset,
            symbols=symbols,
            strategy_mode=spec.strategy_mode,
            bb_squeeze_quantile=(
                spec.bb_squeeze_quantile
                if spec.bb_squeeze_quantile is not None
                else live_defaults["bb_squeeze_quantile"]
            ),
            bb_use_volume_confirm=(
                spec.bb_use_volume_confirm
                if spec.bb_use_volume_confirm is not None
                else live_defaults["bb_use_volume_confirm"]
            ),
            bb_volume_mult=(
                spec.bb_volume_mult
                if spec.bb_volume_mult is not None
                else live_defaults["bb_volume_mult"]
            ),
            bb_slope_lookback=(
                spec.bb_slope_lookback
                if spec.bb_slope_lookback is not None
                else live_defaults["bb_slope_lookback"]
            ),
            bb_breakout_buffer_pct=(
                spec.bb_breakout_buffer_pct
                if spec.bb_breakout_buffer_pct is not None
                else live_defaults["bb_breakout_buffer_pct"]
            ),
            bb_min_mid_slope=(
                spec.bb_min_mid_slope
                if spec.bb_min_mid_slope is not None
                else live_defaults["bb_min_mid_slope"]
            ),
            bb_trend_filter=(
                spec.bb_trend_filter
                if spec.bb_trend_filter is not None
                else live_defaults["bb_trend_filter"]
            ),
            **common_kwargs,
        )
        result["start_date"] = start_date
        result["end_date"] = end_date
        rows.append(result_to_row(spec, result, dataset, symbols))

    df = pd.DataFrame(rows)
    df = add_baseline_comparison_columns(df)
    df = add_usefulness_columns(df)
    return df


def write_outputs(output_dir: Path, all_runs_df: pd.DataFrame, ranked_df: pd.DataFrame, top_n: int) -> tuple[Path, Path]:
    runs_csv = output_dir / "hybrid_bb_mr_research_runs.csv"
    ranked_csv = output_dir / "hybrid_bb_mr_research_top.csv"
    all_runs_df.to_csv(runs_csv, index=False)
    ranked_df.head(top_n).to_csv(ranked_csv, index=False)
    return runs_csv, ranked_csv


def print_summary(all_runs_df: pd.DataFrame, ranked_df: pd.DataFrame, top_n: int) -> None:
    baseline_mr = all_runs_df[all_runs_df["strategy_mode"] == "mean_reversion"].iloc[0]
    baseline_bb = all_runs_df[all_runs_df["strategy_mode"] == "bollinger_squeeze"].iloc[0]
    print("\nBaseline Comparison:")
    print(
        pd.DataFrame([
            baseline_mr[["run_name", "strategy_mode", "realized_pnl", "profit_factor", "sharpe_ratio", "max_drawdown_pct", "total_trades"]],
            baseline_bb[["run_name", "strategy_mode", "realized_pnl", "profit_factor", "sharpe_ratio", "max_drawdown_pct", "total_trades"]],
        ]).to_string(index=False)
    )
    if ranked_df.empty:
        print("\nNo hybrid rows were produced.")
        return
    triggered_count = int(ranked_df["bb_branch_triggered"].fillna(False).sum())
    dead_weight_count = int(ranked_df["bb_branch_dead_weight"].fillna(False).sum())
    positive_bb_count = int(ranked_df["bb_branch_positive_contribution"].fillna(False).sum())
    print("\nHybrid Branch Participation Summary:")
    print(f"  Hybrid combinations tested: {len(ranked_df)}")
    print(f"  Combinations with BB trades: {triggered_count}")
    print(f"  Combinations with zero BB trades: {dead_weight_count}")
    print(f"  Combinations with positive BB branch PnL: {positive_bb_count}")
    summary_cols = [
        "run_name",
        "bb_branch_triggered",
        "usefulness_score",
        "realized_pnl",
        "profit_factor",
        "sharpe_ratio",
        "max_drawdown_pct",
        "total_trades",
        "bb_branch_trades",
        "bb_branch_trade_share_pct",
        "bb_branch_participation",
        "delta_vs_mr_realized_pnl",
        "delta_vs_mr_profit_factor",
    ]
    print(f"\nTop {min(top_n, len(ranked_df))} Hybrid Rows:")
    print(ranked_df[summary_cols].head(top_n).to_string(index=False))


def main() -> None:
    args = parse_args()
    dataset = Path(args.dataset)
    output_dir = build_output_dir(args.output_dir)
    squeeze_quantiles = _parse_csv_values(args.bb_squeeze_quantiles, float)
    use_volume_confirm_values = _parse_bool_csv_values(args.bb_use_volume_confirm_options)
    volume_mult_values = _parse_csv_values(args.bb_volume_mults, float)
    slope_lookbacks = _parse_csv_values(args.bb_slope_lookbacks, int)
    breakout_buffer_pcts = _parse_csv_values(args.bb_breakout_buffer_pcts, float)
    min_mid_slopes = _parse_csv_values(args.bb_min_mid_slopes, float)
    trend_filter_values = _parse_bool_csv_values(args.bb_trend_filter_options)
    all_runs_df = run_research_sweep(
        dataset=dataset,
        symbols=args.symbols,
        start_date=args.start_date,
        end_date=args.end_date,
        starting_capital=args.starting_capital,
        position_size=args.position_size,
        squeeze_quantiles=squeeze_quantiles,
        use_volume_confirm_values=use_volume_confirm_values,
        volume_mult_values=volume_mult_values,
        slope_lookbacks=slope_lookbacks,
        breakout_buffer_pcts=breakout_buffer_pcts,
        min_mid_slopes=min_mid_slopes,
        trend_filter_values=trend_filter_values,
    )
    ranked_df = build_ranked_hybrid_dataframe(all_runs_df)
    runs_csv, ranked_csv = write_outputs(output_dir, all_runs_df, ranked_df, args.top_n)
    print_summary(all_runs_df, ranked_df, args.top_n)
    print("\nSaved:")
    print(f"  - {runs_csv}")
    print(f"  - {ranked_csv}")


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from research.experiment_log import log_experiment_run
from run_edge_audit import _frame_text
from run_long_horizon_validation import _parse_horizon_list, _run_long_horizon_scenario
from run_trend_pullback_robustness_clean import compute_symbol_concentration
from run_volatility_expansion_validation import _load_config_runtime, summarize_realized_by_symbol
from strategy import STRATEGY_MODE_TREND_PULLBACK


DEFAULT_CONFIG_PATH = Path("config") / "trend_pullback.example.json"
DEFAULT_SYMBOLS = ("AMD", "JPM", "HON", "C")
DEFAULT_FORWARD_HORIZONS = (4, 6, 8, 10, 15, 20, 30)
DEFAULT_HOLD_BARS = 20


@dataclass(frozen=True)
class FrozenBaseline:
    strategy_mode: str
    symbols: tuple[str, ...]
    hold_bars: int
    runtime: dict[str, Any]
    forward_horizons: tuple[int, ...]
    dataset_path: str


@dataclass(frozen=True)
class WindowSpec:
    label: str
    section: str
    start_date: str
    end_date: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run strict out-of-sample validation for the frozen trend_pullback baseline."
    )
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH), help="Frozen trend_pullback config JSON.")
    parser.add_argument("--dataset", help="Clean SIP regular-session dataset directory.")
    parser.add_argument("--symbols", nargs="*", help="Optional baseline symbol override. Defaults to frozen research subset.")
    parser.add_argument("--hold-bars", type=int, default=DEFAULT_HOLD_BARS, help="Frozen hold horizon. Default: 20.")
    parser.add_argument("--forward-horizons", default="4,6,8,10,15,20,30")
    parser.add_argument("--commission-per-order", type=float, default=0.01)
    parser.add_argument("--slippage-per-share", type=float, default=0.05)
    parser.add_argument("--position-size", type=float, default=1000.0)
    parser.add_argument("--sections", default="all", help="Comma-separated subset: all,months,segments.")
    parser.add_argument("--output-dir", help="Optional directory for CSV/JSON artifacts.")
    return parser.parse_args()


def _normalize_symbol_list(symbols: list[str] | tuple[str, ...]) -> tuple[str, ...]:
    normalized: list[str] = []
    seen: set[str] = set()
    for raw_symbol in symbols:
        symbol = str(raw_symbol).strip().upper()
        if not symbol or symbol in seen:
            continue
        normalized.append(symbol)
        seen.add(symbol)
    return tuple(normalized)


def infer_clean_sip_dataset(datasets_root: Path) -> Path:
    candidates: list[tuple[pd.Timestamp, Path]] = []
    for manifest_path in datasets_root.glob("**/manifest.json"):
        try:
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if payload.get("feed") != "sip":
            continue
        if payload.get("session_filter") != "regular":
            continue
        if payload.get("dataset_variant") != "sip_clean_regular_session":
            continue
        created = pd.Timestamp(payload.get("created_at_utc")) if payload.get("created_at_utc") else pd.Timestamp.min.tz_localize("UTC")
        candidates.append((created, manifest_path.parent))
    if not candidates:
        raise RuntimeError("No clean SIP regular-session dataset found. Pass --dataset explicitly.")
    candidates.sort(key=lambda item: item[0], reverse=True)
    return candidates[0][1]


def build_frozen_baseline(
    *,
    config_path: Path,
    dataset: str | None,
    symbols_override: list[str] | None,
    hold_bars: int,
    forward_horizons: tuple[int, ...],
) -> FrozenBaseline:
    runtime, source_dataset = _load_config_runtime(config_path)
    runtime["strategy_mode"] = STRATEGY_MODE_TREND_PULLBACK
    symbols = _normalize_symbol_list(symbols_override or runtime.get("symbols") or DEFAULT_SYMBOLS)
    runtime["symbols"] = list(symbols)
    runtime["trend_pullback_hold_bars"] = int(hold_bars)
    dataset_path = str(Path(dataset)) if dataset else (
        str(Path(source_dataset)) if source_dataset else str(infer_clean_sip_dataset(Path("datasets")))
    )
    return FrozenBaseline(
        strategy_mode=STRATEGY_MODE_TREND_PULLBACK,
        symbols=symbols,
        hold_bars=int(hold_bars),
        runtime=runtime,
        forward_horizons=forward_horizons,
        dataset_path=dataset_path,
    )


def build_month_window_specs(evaluations_df: pd.DataFrame) -> list[WindowSpec]:
    if evaluations_df.empty or "month" not in evaluations_df.columns:
        return []
    specs: list[WindowSpec] = []
    for month in sorted(str(value) for value in evaluations_df["month"].dropna().unique()):
        period = pd.Period(month, freq="M")
        specs.append(
            WindowSpec(
                label=f"month:{month}",
                section="months",
                start_date=period.start_time.strftime("%Y-%m-%d"),
                end_date=period.end_time.strftime("%Y-%m-%d"),
            )
        )
    return specs


def build_contiguous_window_specs(evaluations_df: pd.DataFrame) -> list[WindowSpec]:
    if evaluations_df.empty or "date" not in evaluations_df.columns:
        return []
    unique_dates = sorted(pd.to_datetime(evaluations_df["date"], utc=True, errors="coerce").dropna().dt.strftime("%Y-%m-%d").unique())
    if len(unique_dates) < 3:
        return []
    windows: list[WindowSpec] = []
    chunks = pd.Series(unique_dates).groupby(pd.cut(range(len(unique_dates)), bins=3, labels=False)).agg(list)
    labels = ("segment:early", "segment:middle", "segment:late")
    for idx, dates in enumerate(chunks):
        if not dates:
            continue
        windows.append(
            WindowSpec(
                label=labels[idx],
                section="segments",
                start_date=str(dates[0]),
                end_date=str(dates[-1]),
            )
        )
    return windows


def _best_raw_summary(signals_df: pd.DataFrame, horizons: tuple[int, ...]) -> tuple[int, float]:
    rows: list[tuple[int, float]] = []
    for horizon in horizons:
        column = f"fwd_{horizon}b_net_pct"
        if column not in signals_df.columns:
            continue
        series = pd.to_numeric(signals_df[column], errors="coerce").dropna()
        if series.empty:
            continue
        rows.append((horizon, float(series.mean())))
    if not rows:
        return 0, 0.0
    return max(rows, key=lambda item: (item[1], -item[0]))


def run_window_validation(
    *,
    baseline: FrozenBaseline,
    config_path: Path,
    window: WindowSpec,
    commission_per_order: float,
    slippage_per_share: float,
    position_size: float,
) -> dict[str, Any]:
    run = _run_long_horizon_scenario(
        config_path=config_path,
        runtime=baseline.runtime,
        source_dataset=baseline.dataset_path,
        dataset_override=baseline.dataset_path,
        symbols=baseline.symbols,
        start_date=window.start_date,
        end_date=window.end_date,
        commission_per_order=commission_per_order,
        slippage_per_share=slippage_per_share,
        position_size=position_size,
        output_dir=None,
        variant_name=window.label,
        horizons=baseline.forward_horizons,
    )
    best_horizon, best_raw = _best_raw_summary(run.signals_df, baseline.forward_horizons)
    per_symbol_df = summarize_realized_by_symbol(run.backtest_result)
    concentration = compute_symbol_concentration(per_symbol_df)
    return {
        "section": window.section,
        "period_label": window.label,
        "start_date": window.start_date,
        "end_date": window.end_date,
        "signal_count": int(len(run.signals_df)),
        "trade_count": int(len(run.closed_trades_df)),
        "win_rate": float(run.backtest_result.get("win_rate", 0.0)),
        "expectancy": float(run.backtest_result.get("expectancy", 0.0)),
        "pnl": float(run.backtest_result.get("realized_pnl", 0.0)),
        "profit_factor": float(run.backtest_result.get("profit_factor", 0.0)),
        "avg_hold_bars": float(pd.to_numeric(run.closed_trades_df.get("holding_bars"), errors="coerce").dropna().mean()) if not run.closed_trades_df.empty and "holding_bars" in run.closed_trades_df.columns else 0.0,
        "best_raw_horizon_bars": best_horizon,
        "best_raw_net_expectancy_pct": best_raw,
        "top_symbol": concentration.get("top_symbol"),
        "top_positive_share": float(concentration.get("top_positive_share", 0.0)),
    }


def interpret_oos_results(summary_df: pd.DataFrame) -> tuple[str, str, dict[str, Any]]:
    if summary_df.empty:
        return "reject", "No OOS windows were generated.", {"zero_trade_periods": [], "negative_expectancy_periods": []}

    zero_trade_periods = summary_df.loc[summary_df["trade_count"] <= 0, "period_label"].tolist()
    negative_expectancy_periods = summary_df.loc[summary_df["expectancy"] <= 0, "period_label"].tolist()
    positive_period_ratio = float((summary_df["expectancy"] > 0).mean())
    positive_trade_ratio = float((summary_df["trade_count"] > 0).mean())
    pnl_range = float(summary_df["pnl"].max() - summary_df["pnl"].min()) if not summary_df.empty else 0.0
    max_top_share = float(summary_df["top_positive_share"].fillna(0.0).max()) if "top_positive_share" in summary_df.columns else 0.0

    diagnostics = {
        "zero_trade_periods": zero_trade_periods,
        "negative_expectancy_periods": negative_expectancy_periods,
        "positive_period_ratio": positive_period_ratio,
        "positive_trade_ratio": positive_trade_ratio,
        "pnl_range": pnl_range,
        "max_top_positive_share": max_top_share,
    }

    if positive_trade_ratio < 0.67 or positive_period_ratio < 0.34:
        return "demote", "The frozen baseline collapses across too many OOS windows to justify deeper confidence.", diagnostics
    if positive_period_ratio < 0.5 or len(negative_expectancy_periods) >= max(2, len(summary_df) // 2):
        return "research-only", "The frozen baseline is inconsistent across OOS windows and remains too fragile to trust.", diagnostics
    if max_top_share > 0.7:
        return "research-only", "OOS behavior stays too concentrated in a single symbol within at least one window.", diagnostics
    if pnl_range > 150.0:
        return "promising but narrow", "The baseline remains positive more often than not, but instability across windows is still large.", diagnostics
    return "stable enough for deeper research", "The frozen baseline stays positive across most OOS windows without obvious collapse.", diagnostics


def _save_outputs(*, output_dir: Path, baseline: FrozenBaseline, summary_df: pd.DataFrame, interpretation: tuple[str, str, dict[str, Any]], windows: list[WindowSpec]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(output_dir / "trend_pullback_oos_summary.csv", index=False)
    payload = {
        "baseline": {
            "strategy_mode": baseline.strategy_mode,
            "symbols": list(baseline.symbols),
            "hold_bars": baseline.hold_bars,
            "forward_horizons": list(baseline.forward_horizons),
            "dataset_path": baseline.dataset_path,
            "runtime": baseline.runtime,
        },
        "windows": [asdict(window) for window in windows],
        "classification": interpretation[0],
        "reason": interpretation[1],
        "diagnostics": interpretation[2],
    }
    (output_dir / "trend_pullback_oos_summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main() -> None:
    args = parse_args()
    config_path = Path(args.config)
    forward_horizons = _parse_horizon_list(args.forward_horizons)
    baseline = build_frozen_baseline(
        config_path=config_path,
        dataset=args.dataset,
        symbols_override=args.symbols,
        hold_bars=int(args.hold_bars),
        forward_horizons=forward_horizons,
    )

    baseline_run = _run_long_horizon_scenario(
        config_path=config_path,
        runtime=baseline.runtime,
        source_dataset=baseline.dataset_path,
        dataset_override=baseline.dataset_path,
        symbols=baseline.symbols,
        start_date=None,
        end_date=None,
        commission_per_order=args.commission_per_order,
        slippage_per_share=args.slippage_per_share,
        position_size=args.position_size,
        output_dir=None,
        variant_name="frozen_baseline_probe",
        horizons=baseline.forward_horizons,
    )

    windows: list[WindowSpec] = []
    selected_sections = args.sections.strip().lower()
    if selected_sections == "all" or "months" in selected_sections.split(","):
        windows.extend(build_month_window_specs(baseline_run.evaluations_df))
    if selected_sections == "all" or "segments" in selected_sections.split(","):
        windows.extend(build_contiguous_window_specs(baseline_run.evaluations_df))

    deduped_windows: list[WindowSpec] = []
    seen: set[tuple[str, str, str]] = set()
    for window in windows:
        key = (window.label, window.start_date, window.end_date)
        if key in seen:
            continue
        seen.add(key)
        deduped_windows.append(window)
    windows = deduped_windows

    rows = [
        run_window_validation(
            baseline=baseline,
            config_path=config_path,
            window=window,
            commission_per_order=args.commission_per_order,
            slippage_per_share=args.slippage_per_share,
            position_size=args.position_size,
        )
        for window in windows
    ]
    summary_df = pd.DataFrame(rows).sort_values(["start_date", "period_label"]) if rows else pd.DataFrame()
    interpretation = interpret_oos_results(summary_df)

    print("\n=== Trend Pullback OOS Validation ===")
    print(f"Dataset: {baseline.dataset_path}")
    print(f"Symbols: {', '.join(baseline.symbols)}")
    print(f"Hold:    {baseline.hold_bars} bars")
    print(f"Windows: {len(windows)}")
    print("\nOOS summary:")
    print(_frame_text(summary_df, max_rows=24))
    print("\nInterpretation:")
    print(_frame_text(pd.DataFrame([{"classification": interpretation[0], "reason": interpretation[1], **interpretation[2]}]), max_rows=5))

    if args.output_dir:
        _save_outputs(output_dir=Path(args.output_dir), baseline=baseline, summary_df=summary_df, interpretation=interpretation, windows=windows)
    log_experiment_run(
        run_type="oos_validation",
        script_path=__file__,
        strategy_name=baseline.strategy_mode,
        dataset_path=baseline.dataset_path,
        symbols=baseline.symbols,
        params={
            "config": args.config,
            "hold_bars": args.hold_bars,
            "forward_horizons": list(forward_horizons),
            "sections": args.sections,
            "commission_per_order": args.commission_per_order,
            "slippage_per_share": args.slippage_per_share,
            "position_size": args.position_size,
        },
        metrics={
            "trade_count": float(summary_df["trade_count"].sum()) if "trade_count" in summary_df.columns else None,
            "expectancy": float(summary_df["expectancy"].mean()) if "expectancy" in summary_df.columns and not summary_df.empty else None,
            "realized_pnl": float(summary_df["pnl"].sum()) if "pnl" in summary_df.columns else None,
        },
        output_path=args.output_dir,
        summary_path=(Path(args.output_dir) / "trend_pullback_oos_summary.json") if args.output_dir else None,
        extra_fields={
            "classification": {"label": interpretation[0], "reason": interpretation[1]},
            "diagnostics": interpretation[2],
            "window_count": len(windows),
        },
    )


if __name__ == "__main__":
    main()

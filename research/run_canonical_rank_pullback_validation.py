from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from research.canonical_rank_pullback import (
    DEFAULT_COMMISSION_PER_ORDER,
    DEFAULT_LIVE_CONFIG_PATH,
    DEFAULT_MAX_POSITIONS,
    DEFAULT_POSITION_SIZE,
    DEFAULT_SLIPPAGE_PER_SHARE,
    DEFAULT_STEP_DAYS,
    DEFAULT_TEST_DAYS,
    DEFAULT_TRAIN_DAYS,
    SCORE_MODE_CHOICES,
    SCORE_MODE_RETURN_20,
    evaluate_strategy_success,
    generate_strategy_variants,
    load_research_panel,
    normalize_rank_lookbacks_for_score_mode,
    print_table,
    resolve_dataset_and_symbols,
    run_walk_forward_validation,
    summarize_trades_by_symbol,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run canonical immediate-entry vs pullback validation with narrow walk-forward selection."
    )
    parser.add_argument("--dataset", help="Dataset directory. Defaults to config/live_config source.dataset or latest clean SIP dataset.")
    parser.add_argument("--config", default=str(DEFAULT_LIVE_CONFIG_PATH), help="Runtime config used to resolve the research universe.")
    parser.add_argument("--symbols", nargs="*", help="Optional universe override.")
    parser.add_argument("--score-mode", default=SCORE_MODE_RETURN_20, choices=SCORE_MODE_CHOICES, help="Ranking score definition.")
    parser.add_argument("--rank-lookbacks", default="20", help="Comma-separated rank lookbacks.")
    parser.add_argument("--eligible-percents", default="0.20,0.30", help="Comma-separated top-percent cutoffs.")
    parser.add_argument("--pullback-depths", default="0.5,1.0", help="Comma-separated ATR pullback depths.")
    parser.add_argument("--hold-bars", default="5,10", help="Comma-separated fixed holding periods.")
    parser.add_argument("--use-early-no-follow-through-exit", action="store_true", help="Apply the optional early no-follow-through exit to pullback variants only.")
    parser.add_argument("--use-recent-follow-through-filter", action="store_true", help="Apply the optional recent top-bucket follow-through filter to pullback variants only.")
    parser.add_argument("--train-days", type=int, default=DEFAULT_TRAIN_DAYS)
    parser.add_argument("--test-days", type=int, default=DEFAULT_TEST_DAYS)
    parser.add_argument("--step-days", type=int, default=DEFAULT_STEP_DAYS)
    parser.add_argument("--max-positions", type=int, default=DEFAULT_MAX_POSITIONS)
    parser.add_argument("--position-size", type=float, default=DEFAULT_POSITION_SIZE)
    parser.add_argument("--commission-per-order", type=float, default=DEFAULT_COMMISSION_PER_ORDER)
    parser.add_argument("--slippage-per-share", type=float, default=DEFAULT_SLIPPAGE_PER_SHARE)
    parser.add_argument("--output-dir", help="Optional output directory for CSV/JSON artifacts.")
    return parser.parse_args()


def _parse_ints(raw: str) -> tuple[int, ...]:
    values = tuple(sorted({int(part.strip()) for part in raw.split(",") if part.strip()}))
    if not values:
        raise ValueError("Expected at least one integer value.")
    return values


def _parse_floats(raw: str) -> tuple[float, ...]:
    values = tuple(sorted({float(part.strip()) for part in raw.split(",") if part.strip()}))
    if not values:
        raise ValueError("Expected at least one numeric value.")
    return values


def _build_comparison_frame(summary_df: pd.DataFrame, *, index_col: str | None = None) -> pd.DataFrame:
    if summary_df.empty:
        columns = [index_col, "metric", "baseline", "pullback", "pullback_minus_baseline"] if index_col else [
            "metric",
            "baseline",
            "pullback",
            "pullback_minus_baseline",
        ]
        return pd.DataFrame(columns=columns)

    def _rows_for_lookup(lookup: dict[str, dict[str, object]], group_value: object | None = None) -> list[dict[str, object]]:
        baseline = lookup.get("baseline", {})
        pullback = lookup.get("pullback", {})
        rows = [
            {
                "metric": "expectancy",
                "baseline": baseline.get("expectancy", 0.0),
                "pullback": pullback.get("expectancy", 0.0),
                "pullback_minus_baseline": float(pullback.get("expectancy", 0.0)) - float(baseline.get("expectancy", 0.0)),
            },
            {
                "metric": "profit_factor",
                "baseline": baseline.get("profit_factor", 0.0),
                "pullback": pullback.get("profit_factor", 0.0),
                "pullback_minus_baseline": float(pullback.get("profit_factor", 0.0)) - float(baseline.get("profit_factor", 0.0)),
            },
            {
                "metric": "trade_count",
                "baseline": baseline.get("trade_count", 0),
                "pullback": pullback.get("trade_count", 0),
                "pullback_minus_baseline": float(pullback.get("trade_count", 0.0)) - float(baseline.get("trade_count", 0.0)),
            },
            {
                "metric": "max_drawdown_pct",
                "baseline": baseline.get("max_drawdown_pct", 0.0),
                "pullback": pullback.get("max_drawdown_pct", 0.0),
                "pullback_minus_baseline": float(pullback.get("max_drawdown_pct", 0.0)) - float(baseline.get("max_drawdown_pct", 0.0)),
            },
        ]
        if index_col is not None:
            for row in rows:
                row[index_col] = group_value
        return rows

    if index_col is None:
        lookup = summary_df.set_index("family").to_dict("index")
        return pd.DataFrame(_rows_for_lookup(lookup))

    rows: list[dict[str, object]] = []
    for group_value, group in summary_df.groupby(index_col, sort=True):
        lookup = group.set_index("family").to_dict("index")
        rows.extend(_rows_for_lookup(lookup, group_value=group_value))
    return pd.DataFrame(rows)


def _build_family_ranking_frame(summary_df: pd.DataFrame, *, index_col: str | None = None) -> pd.DataFrame:
    if summary_df.empty:
        columns = [index_col, "family", "expectancy", "profit_factor", "trade_count", "max_drawdown_pct", "rank"] if index_col else [
            "family",
            "expectancy",
            "profit_factor",
            "trade_count",
            "max_drawdown_pct",
            "rank",
        ]
        return pd.DataFrame(columns=columns)

    frames: list[pd.DataFrame] = []
    if index_col is None:
        ranked = summary_df.sort_values(
            ["expectancy", "profit_factor", "trade_count", "family"],
            ascending=[False, False, False, True],
        ).reset_index(drop=True)
        ranked["rank"] = range(1, len(ranked) + 1)
        return ranked

    for group_value, group in summary_df.groupby(index_col, sort=True):
        ranked = group.sort_values(
            ["expectancy", "profit_factor", "trade_count", "family"],
            ascending=[False, False, False, True],
        ).reset_index(drop=True)
        ranked["rank"] = range(1, len(ranked) + 1)
        ranked[index_col] = group_value
        frames.append(ranked)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _summarize_family_win_counts(ranked_family_df: pd.DataFrame, *, index_col: str) -> pd.DataFrame:
    if ranked_family_df.empty or index_col not in ranked_family_df.columns:
        return pd.DataFrame(columns=["family", "win_count"])
    winners = ranked_family_df[ranked_family_df["rank"] == 1]
    return (
        winners.groupby("family", as_index=False)
        .size()
        .rename(columns={"size": "win_count"})
        .sort_values(["win_count", "family"], ascending=[False, True])
        .reset_index(drop=True)
    )


def _summarize_selected_trades_by_group(
    trades_df: pd.DataFrame,
    *,
    group_col: str,
) -> pd.DataFrame:
    if trades_df.empty or group_col not in trades_df.columns:
        return pd.DataFrame(
            columns=[
                group_col,
                "family",
                "score_mode",
                "trade_count",
                "expectancy",
                "win_rate",
                "profit_factor",
                "avg_hold_bars",
                "total_pnl",
                "total_return_pct",
                "max_drawdown_pct",
            ]
        )

    rows: list[dict[str, object]] = []
    for (group_value, family), group in trades_df.groupby([group_col, "family"], sort=True):
        family_variant = group["variant_id"].iloc[0] if "variant_id" in group.columns and not group.empty else f"selected_{family}"
        summary = {
            group_col: group_value,
            "family": family,
            "score_mode": str(group["score_mode"].iloc[0]) if "score_mode" in group.columns and not group.empty else None,
            "variant_id": family_variant,
        }
        ordered_group = group.sort_values("exit_ts") if "exit_ts" in group.columns else group
        path = ordered_group.copy()
        path["equity"] = path["realized_pnl"].cumsum()
        running_max = path["equity"].cummax()
        drawdown = path["equity"] - running_max
        capital_base = float(DEFAULT_POSITION_SIZE * max(1, DEFAULT_MAX_POSITIONS))
        max_drawdown_pct = float((drawdown.min() / capital_base) * 100.0 if capital_base > 0 else 0.0)
        summary.update(
            {
                "trade_count": int(len(group)),
                "expectancy": float(group["realized_pnl"].mean()),
                "win_rate": float((group["realized_pnl"] > 0).mean() * 100.0),
                "profit_factor": float(
                    group.loc[group["realized_pnl"] > 0, "realized_pnl"].sum()
                    / abs(group.loc[group["realized_pnl"] < 0, "realized_pnl"].sum())
                )
                if (group["realized_pnl"] < 0).any()
                else (999.0 if (group["realized_pnl"] > 0).any() else 0.0),
                "avg_hold_bars": float(group["hold_bars"].mean()) if "hold_bars" in group.columns else 0.0,
                "total_pnl": float(group["realized_pnl"].sum()),
                "total_return_pct": float(group["return_pct"].sum()) if "return_pct" in group.columns else 0.0,
                "max_drawdown_pct": abs(max_drawdown_pct),
            }
        )
        rows.append(summary)
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    config_path = Path(args.config)
    rank_lookbacks = normalize_rank_lookbacks_for_score_mode(args.score_mode, _parse_ints(args.rank_lookbacks))
    eligible_percents = _parse_floats(args.eligible_percents)
    pullback_depths = _parse_floats(args.pullback_depths)
    hold_bars_list = _parse_ints(args.hold_bars)
    variants = generate_strategy_variants(
        score_mode=args.score_mode,
        rank_lookbacks=rank_lookbacks,
        eligible_percents=eligible_percents,
        hold_bars_list=hold_bars_list,
        pullback_depths=pullback_depths,
        use_early_no_follow_through_exit=bool(args.use_early_no_follow_through_exit),
        use_recent_follow_through_filter=bool(args.use_recent_follow_through_filter),
    )

    dataset_path, symbols, manifest = resolve_dataset_and_symbols(
        dataset=args.dataset,
        config_path=config_path,
        symbols=args.symbols,
    )
    panel_df, _ = load_research_panel(dataset_path=dataset_path, symbols=symbols)
    validation_outputs = run_walk_forward_validation(
        panel_df,
        variants=variants,
        train_days=int(args.train_days),
        test_days=int(args.test_days),
        step_days=int(args.step_days),
        max_positions=int(args.max_positions),
        position_size=float(args.position_size),
        commission_per_order=float(args.commission_per_order),
        slippage_per_share=float(args.slippage_per_share),
    )

    stitched_selected_summary_df = validation_outputs["stitched_selected_summary"]
    stitched_selected_trades_df = validation_outputs["stitched_selected_trades"]
    per_symbol_frames: list[pd.DataFrame] = []
    if not stitched_selected_trades_df.empty:
        for family, group in stitched_selected_trades_df.groupby("family", sort=True):
            family_summary = summarize_trades_by_symbol(group)
            family_summary.insert(0, "family", family)
            per_symbol_frames.append(family_summary)
    per_symbol_df = pd.concat(per_symbol_frames, ignore_index=True) if per_symbol_frames else pd.DataFrame()
    success = evaluate_strategy_success(stitched_selected_summary_df, stitched_selected_trades_df)

    if not stitched_selected_trades_df.empty and "exit_ts" in stitched_selected_trades_df.columns:
        stitched_selected_trades_df = stitched_selected_trades_df.copy()
        stitched_selected_trades_df["exit_ts"] = pd.to_datetime(stitched_selected_trades_df["exit_ts"], utc=True)
        stitched_selected_trades_df["exit_month"] = stitched_selected_trades_df["exit_ts"].dt.tz_convert("America/New_York").dt.strftime("%Y-%m")
    comparison_df = _build_comparison_frame(stitched_selected_summary_df)
    stitched_family_ranking_df = _build_family_ranking_frame(stitched_selected_summary_df)
    per_fold_selected_summary_df = _summarize_selected_trades_by_group(
        stitched_selected_trades_df,
        group_col="window_idx",
    )
    per_fold_comparison_df = _build_comparison_frame(per_fold_selected_summary_df, index_col="window_idx")
    per_fold_family_ranking_df = _build_family_ranking_frame(per_fold_selected_summary_df, index_col="window_idx")
    family_win_counts_df = _summarize_family_win_counts(per_fold_family_ranking_df, index_col="window_idx")
    time_slice_selected_summary_df = _summarize_selected_trades_by_group(
        stitched_selected_trades_df,
        group_col="exit_month",
    )
    time_slice_comparison_df = _build_comparison_frame(time_slice_selected_summary_df, index_col="exit_month")

    diagnostics = {
        "dataset_path": str(dataset_path),
        "symbol_count": len(symbols),
        "symbols": symbols,
        "timeframe": manifest.get("timeframe"),
        "feed": manifest.get("feed"),
        "score_mode": args.score_mode,
        "tested_variant_count": len(variants),
        "fold_count": int(validation_outputs["windows"]["window_idx"].nunique()) if not validation_outputs["windows"].empty else 0,
        "rank_lookbacks": list(rank_lookbacks),
        "eligible_percents": list(eligible_percents),
        "pullback_depths": list(pullback_depths),
        "hold_bars": list(hold_bars_list),
        "use_early_no_follow_through_exit": bool(args.use_early_no_follow_through_exit),
        "use_recent_follow_through_filter": bool(args.use_recent_follow_through_filter),
        "success_criteria": success,
    }

    print("\n=== Canonical Rank Pullback Validation ===")
    print(f"Dataset: {dataset_path}")
    print(f"Symbols: {', '.join(symbols)}")
    print(f"Score mode: {args.score_mode}")
    print(f"Variants tried: {len(variants)}")
    print(f"Walk-forward folds: {diagnostics['fold_count']}")
    print_table("Stitched Selected OOS Summary", stitched_selected_summary_df, max_rows=8)
    print_table("Stitched Family Ranking", stitched_family_ranking_df, max_rows=8)
    print_table("Baseline vs Pullback", comparison_df, max_rows=8)
    print_table("Per-Fold Selected OOS", per_fold_selected_summary_df, max_rows=16)
    print_table("Per-Fold Family Ranking", per_fold_family_ranking_df, max_rows=16)
    print_table("Family Win Counts", family_win_counts_df, max_rows=8)
    print_table("Monthly Selected OOS", time_slice_selected_summary_df, max_rows=16)
    print_table("Fold Winners", validation_outputs["fold_winners"], max_rows=24)
    print_table("Stitched OOS by Variant", validation_outputs["stitched_variant_summary"], max_rows=32)
    print_table("Per-Symbol Breakdown", per_symbol_df, max_rows=32)
    print_table("Success Criteria", pd.DataFrame([success]))

    if args.output_dir:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        validation_outputs["windows"].to_csv(output_dir / "walk_forward_windows.csv", index=False)
        validation_outputs["variant_results"].to_csv(output_dir / "variant_results.csv", index=False)
        validation_outputs["fold_winners"].to_csv(output_dir / "fold_winners.csv", index=False)
        validation_outputs["stitched_selected_trades"].to_csv(output_dir / "stitched_selected_trades.csv", index=False)
        validation_outputs["stitched_selected_summary"].to_csv(output_dir / "stitched_selected_summary.csv", index=False)
        validation_outputs["stitched_variant_summary"].to_csv(output_dir / "stitched_variant_summary.csv", index=False)
        stitched_family_ranking_df.to_csv(output_dir / "stitched_family_ranking.csv", index=False)
        comparison_df.to_csv(output_dir / "baseline_vs_pullback.csv", index=False)
        per_fold_selected_summary_df.to_csv(output_dir / "per_fold_selected_summary.csv", index=False)
        per_fold_comparison_df.to_csv(output_dir / "baseline_vs_pullback_by_fold.csv", index=False)
        per_fold_family_ranking_df.to_csv(output_dir / "per_fold_family_ranking.csv", index=False)
        family_win_counts_df.to_csv(output_dir / "family_win_counts.csv", index=False)
        time_slice_selected_summary_df.to_csv(output_dir / "time_slice_selected_summary.csv", index=False)
        time_slice_comparison_df.to_csv(output_dir / "baseline_vs_pullback_by_time_slice.csv", index=False)
        per_symbol_df.to_csv(output_dir / "per_symbol_breakdown.csv", index=False)
        payload = {
            "diagnostics": diagnostics,
            "promotion_discipline": {
                "selection_rule": "Best in-sample variant is selected separately inside each fold and family; stitched OOS behavior is reported for comparison.",
                "selected_from_single_is_run": False,
            },
        }
        (output_dir / "validation_summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from research.canonical_rank_pullback import (
    DEFAULT_AUDIT_HORIZONS,
    DEFAULT_LIVE_CONFIG_PATH,
    SCORE_MODE_CHOICES,
    SCORE_MODE_RETURN_20,
    DEFAULT_STEP_DAYS,
    DEFAULT_TEST_DAYS,
    DEFAULT_TRAIN_DAYS,
    assign_cross_sectional_ranks,
    build_spread_observations,
    build_walk_forward_windows,
    evaluate_signal_success,
    load_research_panel,
    prepare_panel_features,
    print_table,
    resolve_dataset_and_symbols,
    summarize_bucket_forward_returns,
    summarize_fold_spreads,
    summarize_spread,
    summarize_symbol_concentration,
    summarize_time_slices,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the canonical cross-sectional rank audit for the TradeOS research universe."
    )
    parser.add_argument("--dataset", help="Dataset directory. Defaults to config/live_config source.dataset or latest clean SIP dataset.")
    parser.add_argument("--config", default=str(DEFAULT_LIVE_CONFIG_PATH), help="Runtime config used to resolve the research universe.")
    parser.add_argument("--symbols", nargs="*", help="Optional universe override.")
    parser.add_argument("--score-mode", default=SCORE_MODE_RETURN_20, choices=SCORE_MODE_CHOICES, help="Ranking score definition.")
    parser.add_argument("--lookback-bars", type=int, default=20, help="Rolling return lookback for the rank score.")
    parser.add_argument("--eligible-percent", type=float, default=0.20, help="Top-percent cutoff carried into the canonical tradable phase.")
    parser.add_argument("--horizons", default="5,10,20", help="Forward-return horizons in bars.")
    parser.add_argument("--train-days", type=int, default=DEFAULT_TRAIN_DAYS)
    parser.add_argument("--test-days", type=int, default=DEFAULT_TEST_DAYS)
    parser.add_argument("--step-days", type=int, default=DEFAULT_STEP_DAYS)
    parser.add_argument("--output-dir", help="Optional output directory for CSV/JSON artifacts.")
    return parser.parse_args()


def _parse_horizons(raw: str) -> tuple[int, ...]:
    values = tuple(sorted({int(part.strip()) for part in raw.split(",") if part.strip()}))
    if not values or any(value <= 0 for value in values):
        raise ValueError("Horizons must be a comma-separated list of positive integers.")
    return values


def main() -> None:
    args = parse_args()
    config_path = Path(args.config)
    horizons = _parse_horizons(args.horizons) if args.horizons else DEFAULT_AUDIT_HORIZONS
    dataset_path, symbols, _ = resolve_dataset_and_symbols(
        dataset=args.dataset,
        config_path=config_path,
        symbols=args.symbols,
    )
    panel_df, manifest = load_research_panel(dataset_path=dataset_path, symbols=symbols)
    prepared_df = prepare_panel_features(
        panel_df,
        rank_lookback_bars=int(args.lookback_bars),
        audit_horizons=horizons,
        score_mode=args.score_mode,
    )
    ranked_df, rank_diagnostics = assign_cross_sectional_ranks(
        prepared_df,
        eligible_percent=float(args.eligible_percent),
    )
    bucket_summary_df = summarize_bucket_forward_returns(ranked_df, horizons=horizons)
    spread_df = build_spread_observations(ranked_df, horizons=horizons)
    spread_summary_df = summarize_spread(spread_df)
    full_bucket_summary_df = pd.concat([bucket_summary_df, spread_summary_df], ignore_index=True)
    time_slice_df = summarize_time_slices(ranked_df, spread_df, horizons=horizons)
    symbol_contribution_df = summarize_symbol_concentration(ranked_df, horizons=horizons)
    windows = build_walk_forward_windows(
        ranked_df,
        train_days=int(args.train_days),
        test_days=int(args.test_days),
        step_days=int(args.step_days),
    )
    fold_spread_df = summarize_fold_spreads(ranked_df, horizons=horizons, windows=windows)
    interpretation = evaluate_signal_success(
        full_bucket_summary_df,
        spread_summary_df,
        time_slice_df,
        symbol_contribution_df,
    )

    for df in (full_bucket_summary_df, time_slice_df, symbol_contribution_df, fold_spread_df):
        if not df.empty:
            df.insert(0, "score_mode", args.score_mode)

    diagnostics = {
        "dataset_path": str(dataset_path),
        "symbol_count": len(symbols),
        "symbols": symbols,
        "timeframe": manifest.get("timeframe"),
        "feed": manifest.get("feed"),
        "session_filter": manifest.get("session_filter"),
        "score_mode": args.score_mode,
        "rank_lookback_bars": int(args.lookback_bars),
        "eligible_percent": float(args.eligible_percent),
        "horizons": list(horizons),
        **rank_diagnostics,
        "observation_count": int(len(ranked_df)),
        "spread_observation_count": int(len(spread_df)),
        "walk_forward_window_count": len(windows),
        "success_criteria": interpretation,
    }

    print("\n=== Canonical Cross-Sectional Rank Audit ===")
    print(f"Dataset:  {dataset_path}")
    print(f"Symbols:  {', '.join(symbols)}")
    print(f"Score:    {args.score_mode}")
    print(f"Lookback: {args.lookback_bars} bars")
    print(f"Horizons: {', '.join(str(value) for value in horizons)}")
    print(f"Ranked timestamps: {rank_diagnostics['timestamps_ranked']} / {rank_diagnostics['timestamps_total']}")
    print_table("Bucket Summary", full_bucket_summary_df, max_rows=24)
    print_table("Monthly Slices", time_slice_df, max_rows=36)
    print_table("Symbol Concentration", symbol_contribution_df, max_rows=36)
    print_table("Fold Spread Summary", fold_spread_df, max_rows=24)
    print_table("Interpretation", pd.DataFrame([interpretation]))

    if args.output_dir:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        full_bucket_summary_df.to_csv(output_dir / "bucket_summary.csv", index=False)
        time_slice_df.to_csv(output_dir / "time_slice_summary.csv", index=False)
        symbol_contribution_df.to_csv(output_dir / "symbol_contribution.csv", index=False)
        fold_spread_df.to_csv(output_dir / "fold_spread_summary.csv", index=False)
        ranked_df.to_csv(output_dir / "ranked_observations.csv", index=False)
        pd.DataFrame([vars(window) for window in windows]).to_csv(output_dir / "walk_forward_windows.csv", index=False)
        (output_dir / "diagnostics.json").write_text(json.dumps(diagnostics, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()

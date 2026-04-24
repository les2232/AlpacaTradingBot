from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from backtest_runner import load_dataset
from run_edge_audit import _frame_text


DEFAULT_LOOKBACK_BARS = 20
DEFAULT_HORIZONS = (5, 10, 20)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a first-pass cross-sectional relative-strength edge audit on a clean SIP regular-session dataset."
    )
    parser.add_argument("--dataset", help="Dataset directory. Defaults to the latest clean SIP regular-session dataset in datasets/.")
    parser.add_argument("--lookback-bars", type=int, default=DEFAULT_LOOKBACK_BARS, help="Rolling return lookback in bars.")
    parser.add_argument("--horizons", default="5,10,20", help="Comma-separated forward-return horizons in bars.")
    parser.add_argument("--output-dir", help="Optional directory to save CSV/JSON outputs.")
    return parser.parse_args()


def _parse_horizons(raw: str) -> tuple[int, ...]:
    values = tuple(sorted({int(part.strip()) for part in raw.split(",") if part.strip()}))
    if not values or any(value <= 0 for value in values):
        raise ValueError("Horizons must be a comma-separated list of positive integers.")
    return values


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


def prepare_cross_sectional_frame(
    df: pd.DataFrame,
    *,
    lookback_bars: int,
    horizons: tuple[int, ...],
) -> tuple[pd.DataFrame, dict[str, int]]:
    if df.empty:
        return pd.DataFrame(), {
            "total_rows": 0,
            "rows_after_symbol_sort": 0,
            "rows_with_lookback": 0,
            "rows_with_all_horizons": 0,
        }

    prepared = df.copy()
    prepared["symbol"] = prepared["symbol"].astype(str).str.upper()
    prepared["timestamp"] = pd.to_datetime(prepared["timestamp"], utc=True)
    prepared = prepared.sort_values(["symbol", "timestamp"]).reset_index(drop=True)

    prepared["lookback_return"] = prepared.groupby("symbol", sort=False)["close"].transform(
        lambda series: series / series.shift(lookback_bars) - 1.0
    )
    for horizon in horizons:
        prepared[f"fwd_{horizon}b_return_pct"] = prepared.groupby("symbol", sort=False)["close"].transform(
            lambda series, horizon=horizon: (series.shift(-horizon) / series - 1.0) * 100.0
        )
    timestamps_et = prepared["timestamp"].dt.tz_convert("America/New_York")
    prepared["month"] = timestamps_et.dt.strftime("%Y-%m")
    prepared["date"] = timestamps_et.dt.strftime("%Y-%m-%d")
    prepared["signal_hour"] = timestamps_et.dt.hour
    prepared["time_bucket"] = timestamps_et.apply(classify_time_bucket)

    diagnostics = {
        "total_rows": int(len(df)),
        "rows_after_symbol_sort": int(len(prepared)),
        "rows_with_lookback": int(prepared["lookback_return"].notna().sum()),
        "rows_with_all_horizons": int(prepared[[f"fwd_{horizon}b_return_pct" for horizon in horizons]].notna().all(axis=1).sum()),
    }
    return prepared, diagnostics


def classify_time_bucket(ts: pd.Timestamp) -> str:
    minutes = ts.hour * 60 + ts.minute
    if minutes < (10 * 60):
        return "open_30m"
    if minutes < (12 * 60):
        return "morning"
    if minutes < (14 * 60):
        return "midday"
    return "afternoon"


def assign_rank_buckets(prepared_df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    columns = [
        "timestamp",
        "symbol",
        "lookback_return",
        "bucket",
        "bucket_rank",
        "bucket_size",
        "month",
        "date",
        "time_bucket",
    ] + [col for col in prepared_df.columns if col.startswith("fwd_") and col.endswith("b_return_pct")]
    if prepared_df.empty:
        return pd.DataFrame(columns=columns), {"timestamps_total": 0, "timestamps_ranked": 0, "timestamps_dropped_lt3": 0}

    rows: list[dict[str, Any]] = []
    timestamps_total = 0
    timestamps_ranked = 0
    timestamps_dropped_lt3 = 0

    for timestamp, group in prepared_df.groupby("timestamp", sort=True):
        timestamps_total += 1
        valid = group[group["lookback_return"].notna()].copy()
        if len(valid) < 3:
            timestamps_dropped_lt3 += 1
            continue
        valid = valid.sort_values(["lookback_return", "symbol"], ascending=[False, True]).reset_index(drop=True)
        n = len(valid)
        top_n = max(1, n // 3)
        bottom_n = max(1, n // 3)
        timestamps_ranked += 1
        for idx, (_, row) in enumerate(valid.iterrows()):
            if idx < top_n:
                bucket = "top"
            elif idx >= n - bottom_n:
                bucket = "bottom"
            else:
                bucket = "middle"
            rows.append(
                {
                    "timestamp": timestamp,
                    "symbol": row["symbol"],
                    "lookback_return": float(row["lookback_return"]),
                    "bucket": bucket,
                    "bucket_rank": idx + 1,
                    "bucket_size": n,
                    "month": row["month"],
                    "date": row["date"],
                    "time_bucket": row["time_bucket"],
                    **{col: row[col] for col in valid.columns if col.startswith("fwd_") and col.endswith("b_return_pct")},
                }
            )

    diagnostics = {
        "timestamps_total": timestamps_total,
        "timestamps_ranked": timestamps_ranked,
        "timestamps_dropped_lt3": timestamps_dropped_lt3,
    }
    return pd.DataFrame(rows, columns=columns), diagnostics


def _return_summary(series: pd.Series) -> dict[str, float]:
    clean = pd.to_numeric(series, errors="coerce").dropna()
    if clean.empty:
        return {
            "observation_count": 0,
            "mean_return_pct": 0.0,
            "median_return_pct": 0.0,
            "std_return_pct": 0.0,
            "hit_rate_pct": 0.0,
        }
    return {
        "observation_count": int(len(clean)),
        "mean_return_pct": float(clean.mean()),
        "median_return_pct": float(clean.median()),
        "std_return_pct": float(clean.std(ddof=0)),
        "hit_rate_pct": float((clean > 0).mean() * 100.0),
    }


def summarize_buckets(observations_df: pd.DataFrame, *, horizons: tuple[int, ...]) -> pd.DataFrame:
    columns = [
        "bucket",
        "horizon_bars",
        "observation_count",
        "mean_return_pct",
        "median_return_pct",
        "std_return_pct",
        "hit_rate_pct",
    ]
    if observations_df.empty:
        return pd.DataFrame(columns=columns)

    rows: list[dict[str, Any]] = []
    for bucket, bucket_df in observations_df.groupby("bucket", sort=True):
        for horizon in horizons:
            stats = _return_summary(bucket_df[f"fwd_{horizon}b_return_pct"])
            rows.append({"bucket": bucket, "horizon_bars": horizon, **stats})
    return pd.DataFrame(rows).sort_values(["horizon_bars", "bucket"])


def build_spread_observations(observations_df: pd.DataFrame, *, horizons: tuple[int, ...]) -> pd.DataFrame:
    columns = ["timestamp", "month", "date", "time_bucket", "horizon_bars", "spread_return_pct"]
    if observations_df.empty:
        return pd.DataFrame(columns=columns)

    rows: list[dict[str, Any]] = []
    for timestamp, group in observations_df.groupby("timestamp", sort=True):
        top = group[group["bucket"] == "top"]
        bottom = group[group["bucket"] == "bottom"]
        if top.empty or bottom.empty:
            continue
        for horizon in horizons:
            top_returns = pd.to_numeric(top[f"fwd_{horizon}b_return_pct"], errors="coerce").dropna()
            bottom_returns = pd.to_numeric(bottom[f"fwd_{horizon}b_return_pct"], errors="coerce").dropna()
            if top_returns.empty or bottom_returns.empty:
                continue
            rows.append(
                {
                    "timestamp": timestamp,
                    "month": str(group.iloc[0]["month"]),
                    "date": str(group.iloc[0]["date"]),
                    "time_bucket": str(group.iloc[0]["time_bucket"]),
                    "horizon_bars": horizon,
                    "spread_return_pct": float(top_returns.mean() - bottom_returns.mean()),
                }
            )
    return pd.DataFrame(rows, columns=columns)


def summarize_spread(spread_df: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "bucket",
        "horizon_bars",
        "observation_count",
        "mean_return_pct",
        "median_return_pct",
        "std_return_pct",
        "hit_rate_pct",
    ]
    if spread_df.empty:
        return pd.DataFrame(columns=columns)

    rows = []
    for horizon, group in spread_df.groupby("horizon_bars", sort=True):
        stats = _return_summary(group["spread_return_pct"])
        rows.append({"bucket": "top_minus_bottom", "horizon_bars": int(horizon), **stats})
    return pd.DataFrame(rows, columns=columns).sort_values("horizon_bars")


def summarize_time_slices(
    observations_df: pd.DataFrame,
    spread_df: pd.DataFrame,
    *,
    horizons: tuple[int, ...],
) -> pd.DataFrame:
    columns = [
        "period",
        "bucket",
        "horizon_bars",
        "observation_count",
        "mean_return_pct",
        "median_return_pct",
        "std_return_pct",
        "hit_rate_pct",
    ]
    if observations_df.empty:
        return pd.DataFrame(columns=columns)

    rows: list[dict[str, Any]] = []
    for period, group in observations_df.groupby("month", sort=True):
        for bucket, bucket_df in group.groupby("bucket", sort=True):
            for horizon in horizons:
                stats = _return_summary(bucket_df[f"fwd_{horizon}b_return_pct"])
                rows.append({"period": period, "bucket": bucket, "horizon_bars": horizon, **stats})
    if not spread_df.empty:
        for (period, horizon), group in spread_df.groupby(["month", "horizon_bars"], sort=True):
            stats = _return_summary(group["spread_return_pct"])
            rows.append({"period": period, "bucket": "top_minus_bottom", "horizon_bars": int(horizon), **stats})
    return pd.DataFrame(rows, columns=columns).sort_values(["period", "horizon_bars", "bucket"])


def summarize_symbol_contribution(observations_df: pd.DataFrame, *, horizons: tuple[int, ...]) -> pd.DataFrame:
    columns = [
        "symbol",
        "bucket",
        "membership_count",
        "membership_share_pct",
    ] + [f"avg_{horizon}b_return_pct" for horizon in horizons]
    if observations_df.empty:
        return pd.DataFrame(columns=columns)

    total_by_bucket = observations_df.groupby("bucket", sort=True).size().to_dict()
    rows: list[dict[str, Any]] = []
    for (symbol, bucket), group in observations_df.groupby(["symbol", "bucket"], sort=True):
        record: dict[str, Any] = {
            "symbol": symbol,
            "bucket": bucket,
            "membership_count": int(len(group)),
            "membership_share_pct": float(len(group) / max(1, total_by_bucket.get(bucket, 0)) * 100.0),
        }
        for horizon in horizons:
            record[f"avg_{horizon}b_return_pct"] = float(pd.to_numeric(group[f"fwd_{horizon}b_return_pct"], errors="coerce").dropna().mean()) if not group.empty else 0.0
        rows.append(record)
    return pd.DataFrame(rows, columns=columns).sort_values(["bucket", "membership_count", "symbol"], ascending=[True, False, True])


def interpret_results(
    bucket_summary_df: pd.DataFrame,
    spread_summary_df: pd.DataFrame,
    time_slice_df: pd.DataFrame,
    symbol_contribution_df: pd.DataFrame,
) -> dict[str, Any]:
    overall_top = bucket_summary_df[bucket_summary_df["bucket"] == "top"]
    overall_bottom = bucket_summary_df[bucket_summary_df["bucket"] == "bottom"]
    spread_positive_ratio = float((spread_summary_df["mean_return_pct"] > 0).mean()) if not spread_summary_df.empty else 0.0
    top_beats_bottom_horizons = []
    for horizon in sorted(bucket_summary_df["horizon_bars"].unique()) if not bucket_summary_df.empty else []:
        top_row = overall_top[overall_top["horizon_bars"] == horizon]
        bottom_row = overall_bottom[overall_bottom["horizon_bars"] == horizon]
        if top_row.empty or bottom_row.empty:
            continue
        if float(top_row.iloc[0]["mean_return_pct"]) > float(bottom_row.iloc[0]["mean_return_pct"]):
            top_beats_bottom_horizons.append(int(horizon))

    best_spread_row = (
        spread_summary_df.sort_values(["mean_return_pct", "horizon_bars"], ascending=[False, True]).iloc[0].to_dict()
        if not spread_summary_df.empty
        else {}
    )
    spread_time_positive_ratio = 0.0
    if not time_slice_df.empty:
        spread_time = time_slice_df[time_slice_df["bucket"] == "top_minus_bottom"]
        if not spread_time.empty:
            spread_time_positive_ratio = float((spread_time["mean_return_pct"] > 0).mean())

    top_contrib = symbol_contribution_df[symbol_contribution_df["bucket"] == "top"]
    concentration = {}
    if not top_contrib.empty:
        top_contrib = top_contrib.sort_values(["membership_count", "symbol"], ascending=[False, True]).reset_index(drop=True)
        concentration = {
            "top_bucket_most_frequent_symbol": str(top_contrib.iloc[0]["symbol"]),
            "top_bucket_membership_share_pct": float(top_contrib.iloc[0]["membership_share_pct"]),
        }

    return {
        "top_beats_bottom_horizons": top_beats_bottom_horizons,
        "spread_positive_horizon_ratio": spread_positive_ratio,
        "spread_positive_time_slice_ratio": spread_time_positive_ratio,
        "best_spread_horizon_bars": int(best_spread_row.get("horizon_bars", 0) or 0),
        "best_spread_mean_return_pct": float(best_spread_row.get("mean_return_pct", 0.0) or 0.0),
        **concentration,
    }


def _save_outputs(
    *,
    output_dir: Path,
    bucket_summary_df: pd.DataFrame,
    time_slice_df: pd.DataFrame,
    symbol_contribution_df: pd.DataFrame,
    diagnostics: dict[str, Any],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    bucket_summary_df.to_csv(output_dir / "bucket_summary.csv", index=False)
    time_slice_df.to_csv(output_dir / "time_slice_summary.csv", index=False)
    symbol_contribution_df.to_csv(output_dir / "symbol_contribution.csv", index=False)
    (output_dir / "diagnostics.json").write_text(json.dumps(diagnostics, indent=2), encoding="utf-8")


def main() -> None:
    args = parse_args()
    dataset_path = Path(args.dataset) if args.dataset else infer_clean_sip_dataset(Path("datasets"))
    lookback_bars = int(args.lookback_bars)
    horizons = _parse_horizons(args.horizons)
    df, manifest = load_dataset(dataset_path)

    prepared_df, prep_diag = prepare_cross_sectional_frame(df, lookback_bars=lookback_bars, horizons=horizons)
    observations_df, bucket_diag = assign_rank_buckets(prepared_df)
    spread_df = build_spread_observations(observations_df, horizons=horizons)
    spread_summary_df = summarize_spread(spread_df)
    bucket_summary_df = pd.concat(
        [
            summarize_buckets(observations_df, horizons=horizons),
            spread_summary_df,
        ],
        ignore_index=True,
    )
    time_slice_df = summarize_time_slices(observations_df, spread_df, horizons=horizons)
    symbol_contribution_df = summarize_symbol_contribution(observations_df, horizons=horizons)
    interpretation = interpret_results(bucket_summary_df, spread_summary_df, time_slice_df, symbol_contribution_df)

    diagnostics = {
        "dataset_path": str(dataset_path),
        "timeframe": manifest.get("timeframe"),
        "feed": manifest.get("feed"),
        "session_filter": manifest.get("session_filter"),
        "symbol_count": int(df["symbol"].nunique()) if not df.empty else 0,
        "symbols": sorted(df["symbol"].astype(str).str.upper().unique().tolist()) if not df.empty else [],
        "lookback_bars": lookback_bars,
        "horizons": list(horizons),
        **prep_diag,
        **bucket_diag,
        "symbol_observations_used": int(len(observations_df)),
        "spread_observations_used": int(len(spread_df)),
        "dropped_rows_for_lookback": int(len(prepared_df) - prep_diag["rows_with_lookback"]),
        "dropped_rows_for_max_horizon": int(prep_diag["rows_with_lookback"] - prep_diag["rows_with_all_horizons"]),
        "interpretation": interpretation,
    }

    print("\n=== Cross-Sectional Relative Strength Edge Audit ===")
    print(f"Dataset:  {dataset_path}")
    print(f"Lookback: {lookback_bars} bars")
    print(f"Horizons: {', '.join(str(value) for value in horizons)}")
    print(f"Symbols:  {diagnostics['symbol_count']}")
    print(f"Ranked timestamps: {bucket_diag['timestamps_ranked']} / {bucket_diag['timestamps_total']}")
    print(f"Symbol observations used: {len(observations_df)}")
    print(f"Spread observations used: {len(spread_df)}")
    print("\nBucket summary:")
    print(_frame_text(bucket_summary_df, max_rows=24))
    print("\nTime-slice summary:")
    print(_frame_text(time_slice_df, max_rows=36))
    print("\nPer-symbol contribution:")
    print(_frame_text(symbol_contribution_df, max_rows=36))
    print("\nInterpretation:")
    print(_frame_text(pd.DataFrame([interpretation]), max_rows=5))

    if args.output_dir:
        _save_outputs(
            output_dir=Path(args.output_dir),
            bucket_summary_df=bucket_summary_df,
            time_slice_df=time_slice_df,
            symbol_contribution_df=symbol_contribution_df,
            diagnostics=diagnostics,
        )


if __name__ == "__main__":
    main()

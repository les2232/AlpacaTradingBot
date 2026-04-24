from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from backtest_runner import load_dataset
from inspect_dataset_and_run import _frame_text


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Investigate missing intraday bars and separate feed gaps from shared-alignment effects."
    )
    parser.add_argument("--dataset", required=True, help="Primary dataset directory.")
    parser.add_argument(
        "--timestamps",
        required=True,
        help="Comma-separated ET timestamps like 2026-02-23T13:30:00,2026-02-27T12:00:00.",
    )
    parser.add_argument("--raw-dataset", help="Optional non-shared/raw dataset directory for the same symbol universe.")
    parser.add_argument("--compare-dataset", help="Optional comparison dataset, e.g. another feed snapshot.")
    parser.add_argument("--symbols", nargs="*", help="Optional symbol subset.")
    parser.add_argument("--output-dir", help="Optional directory for CSV/JSON artifacts.")
    return parser.parse_args()


def _normalize_symbols(symbols: list[str] | None) -> list[str]:
    if not symbols:
        return []
    normalized: list[str] = []
    seen: set[str] = set()
    for raw_symbol in symbols:
        symbol = str(raw_symbol).strip().upper()
        if not symbol or symbol in seen:
            continue
        normalized.append(symbol)
        seen.add(symbol)
    return normalized


def _parse_et_timestamps(raw: str) -> list[pd.Timestamp]:
    timestamps: list[pd.Timestamp] = []
    for chunk in raw.split(","):
        value = chunk.strip()
        if not value:
            continue
        timestamps.append(pd.Timestamp(value, tz="America/New_York"))
    if not timestamps:
        raise ValueError("At least one ET timestamp is required.")
    return timestamps


def _load_dataset_frame(dataset_path: Path, symbols: list[str] | None) -> tuple[pd.DataFrame, dict[str, Any]]:
    df, manifest = load_dataset(dataset_path)
    manifest = manifest if isinstance(manifest, dict) else {}
    if symbols:
        df = df[df["symbol"].isin(symbols)].copy()
    if df.empty:
        return df, manifest
    df = df.sort_values(["symbol", "timestamp"]).reset_index(drop=True).copy()
    df["ts_et"] = df["timestamp"].dt.tz_convert("America/New_York")
    return df, manifest


def per_symbol_bar_presence(df: pd.DataFrame, symbols: list[str], timestamps_et: list[pd.Timestamp]) -> pd.DataFrame:
    columns = [
        "timestamp_et",
        "symbol",
        "present",
        "open",
        "high",
        "low",
        "close",
        "volume",
    ]
    if df.empty:
        return pd.DataFrame(columns=columns)
    rows: list[dict[str, Any]] = []
    for ts_et in timestamps_et:
        matches = df[df["ts_et"] == ts_et]
        lookup = {str(row["symbol"]).upper(): row for _, row in matches.iterrows()}
        for symbol in symbols:
            row = lookup.get(symbol)
            rows.append(
                {
                    "timestamp_et": ts_et.strftime("%Y-%m-%d %H:%M"),
                    "symbol": symbol,
                    "present": row is not None,
                    "open": None if row is None else float(row["open"]),
                    "high": None if row is None else float(row["high"]),
                    "low": None if row is None else float(row["low"]),
                    "close": None if row is None else float(row["close"]),
                    "volume": None if row is None else int(row["volume"]),
                }
            )
    return pd.DataFrame(rows, columns=columns)


def summarize_timestamp_coverage(presence_df: pd.DataFrame) -> pd.DataFrame:
    columns = ["timestamp_et", "symbols_present", "symbols_missing", "coverage_pct"]
    if presence_df.empty:
        return pd.DataFrame(columns=columns)
    rows = []
    for ts, group in presence_df.groupby("timestamp_et", sort=True):
        present = int(group["present"].sum())
        total = int(len(group))
        rows.append(
            {
                "timestamp_et": ts,
                "symbols_present": present,
                "symbols_missing": total - present,
                "coverage_pct": (present / total * 100.0) if total else 0.0,
            }
        )
    return pd.DataFrame(rows, columns=columns)


def summarize_missing_symbol_patterns(presence_df: pd.DataFrame) -> pd.DataFrame:
    columns = ["symbol", "missing_timestamps", "missing_count"]
    if presence_df.empty:
        return pd.DataFrame(columns=columns)
    missing = presence_df[~presence_df["present"]].copy()
    if missing.empty:
        return pd.DataFrame(columns=columns)
    rows = []
    for symbol, group in missing.groupby("symbol", sort=True):
        timestamps = sorted(group["timestamp_et"].astype(str).tolist())
        rows.append(
            {
                "symbol": symbol,
                "missing_timestamps": ", ".join(timestamps),
                "missing_count": len(timestamps),
            }
        )
    return pd.DataFrame(rows, columns=columns).sort_values(["missing_count", "symbol"], ascending=[False, True])


def compare_dataset_coverage(
    *,
    base_presence: pd.DataFrame,
    compare_presence: pd.DataFrame,
    compare_label: str,
    covered_timestamps: set[str] | None = None,
) -> pd.DataFrame:
    columns = ["timestamp_et", "base_present", f"{compare_label}_present", "delta_present", "compare_covered"]
    if base_presence.empty or compare_presence.empty:
        return pd.DataFrame(columns=columns)
    base_summary = summarize_timestamp_coverage(base_presence).rename(columns={"symbols_present": "base_present"})
    compare_summary = summarize_timestamp_coverage(compare_presence).rename(columns={"symbols_present": f"{compare_label}_present"})
    merged = base_summary[["timestamp_et", "base_present"]].merge(
        compare_summary[["timestamp_et", f"{compare_label}_present"]],
        on="timestamp_et",
        how="outer",
    ).fillna(0)
    merged["base_present"] = merged["base_present"].astype(int)
    merged[f"{compare_label}_present"] = merged[f"{compare_label}_present"].astype(int)
    merged["delta_present"] = merged[f"{compare_label}_present"] - merged["base_present"]
    covered = covered_timestamps or set()
    merged["compare_covered"] = merged["timestamp_et"].isin(covered)
    return merged[columns].sort_values("timestamp_et")


def simulate_alignment_impact(df: pd.DataFrame, symbols: list[str]) -> dict[str, Any]:
    if df.empty:
        return {"timestamp_count": 0, "shared_timestamp_count": 0, "dropped_timestamps": pd.DataFrame(columns=["timestamp_et", "symbols_present", "missing_symbols"])}
    timestamp_counts = df.groupby("timestamp")["symbol"].nunique().reset_index(name="symbols_present")
    timestamp_counts["timestamp_et"] = pd.to_datetime(timestamp_counts["timestamp"], utc=True).dt.tz_convert("America/New_York")
    timestamp_counts["missing_symbols"] = len(symbols) - timestamp_counts["symbols_present"]
    dropped = timestamp_counts[timestamp_counts["symbols_present"] < len(symbols)].copy()
    return {
        "timestamp_count": int(timestamp_counts["timestamp"].nunique()),
        "shared_timestamp_count": int((timestamp_counts["symbols_present"] == len(symbols)).sum()),
        "dropped_timestamps": dropped[["timestamp_et", "symbols_present", "missing_symbols"]].sort_values("timestamp_et"),
    }


def classify_issue(
    *,
    raw_presence: pd.DataFrame,
    compare_presence: pd.DataFrame,
) -> tuple[str, str]:
    if not raw_presence.empty:
        coverage = summarize_timestamp_coverage(raw_presence)
        if not coverage.empty and (coverage["symbols_present"] == 0).all():
            return "feed_quality_issue", "The raw dataset has no bars for the affected timestamps, so the issue is upstream of alignment."
        if not coverage.empty and (coverage["symbols_missing"] > 0).any():
            if not compare_presence.empty:
                compare_cov = summarize_timestamp_coverage(compare_presence)
                merged = coverage.merge(compare_cov, on="timestamp_et", suffixes=("_raw", "_compare"), how="left").fillna(0)
                if (merged["symbols_present_compare"] > merged["symbols_present_raw"]).any():
                    return "mixed", "Some symbols are missing in the raw target feed while the comparison dataset has better coverage, so alignment amplifies a feed-quality issue."
            return "alignment_artifact", "Some symbols are present and some are missing in the raw dataset, so shared alignment is removing valid bars for the symbols that do have data."
        return "alignment_artifact", "The raw dataset has full coverage at the affected timestamps, so the loss is caused by the aligned dataset construction."

    if not compare_presence.empty:
        compare_cov = summarize_timestamp_coverage(compare_presence)
        if not compare_cov.empty and (compare_cov["symbols_present"] > 0).any():
            return "feed_quality_issue", "The comparison dataset retains bars at these timestamps while the aligned target dataset does not, which points to feed-specific availability problems that alignment then propagates."
    return "mixed", "The aligned dataset is missing the timestamps, but there is not enough raw local evidence to separate feed loss from alignment loss conclusively."


def print_report(
    *,
    dataset_path: Path,
    dataset_manifest: dict[str, Any],
    base_presence: pd.DataFrame,
    raw_presence: pd.DataFrame,
    compare_presence: pd.DataFrame,
    compare_covered_timestamps: set[str],
    raw_label: str | None,
    compare_label: str | None,
    alignment_impact: dict[str, Any],
    classification: tuple[str, str],
) -> None:
    print("\nMISSING BAR INVESTIGATION")
    print("------------------------")
    print(f"Dataset: {dataset_path}")
    print(f"Feed: {dataset_manifest.get('feed', 'unknown')}")
    print(f"Align mode: {dataset_manifest.get('align_mode', 'unknown')}")
    print("\nAligned dataset coverage:")
    print(_frame_text(summarize_timestamp_coverage(base_presence), max_rows=32))
    if not base_presence.empty:
        for ts, group in base_presence.groupby("timestamp_et", sort=True):
            print(f"\nTimestamp: {ts} ET")
            display = group[["symbol", "present"]].copy()
            display["present"] = display["present"].map({True: "YES", False: "NO"})
            print(_frame_text(display, max_rows=64))

    if not raw_presence.empty and raw_label:
        print(f"\n{raw_label} per-symbol coverage:")
        print(_frame_text(summarize_timestamp_coverage(raw_presence), max_rows=32))
        for ts, group in raw_presence.groupby("timestamp_et", sort=True):
            print(f"\n{raw_label} @ {ts} ET")
            display = group[["symbol", "present", "open", "high", "low", "close", "volume"]].copy()
            display["present"] = display["present"].map({True: "YES", False: "NO"})
            print(_frame_text(display, max_rows=64))
        print("\nMissing-symbol pattern:")
        print(_frame_text(summarize_missing_symbol_patterns(raw_presence), max_rows=64))

    if not compare_presence.empty and compare_label:
        print(f"\n{compare_label} comparison summary:")
        print(
            _frame_text(
                compare_dataset_coverage(
                    base_presence=base_presence,
                    compare_presence=compare_presence,
                    compare_label=compare_label,
                    covered_timestamps=compare_covered_timestamps,
                ),
                max_rows=32,
            )
        )

    print("\nAlignment impact:")
    if alignment_impact.get("not_computable"):
        print(alignment_impact["not_computable"])
    else:
        print(
            f"timestamps_total={alignment_impact['timestamp_count']} | "
            f"timestamps_shared={alignment_impact['shared_timestamp_count']} | "
            f"timestamps_dropped={len(alignment_impact['dropped_timestamps'])}"
        )
    if (
        not alignment_impact.get("not_computable")
        and isinstance(alignment_impact["dropped_timestamps"], pd.DataFrame)
        and not alignment_impact["dropped_timestamps"].empty
    ):
        display = alignment_impact["dropped_timestamps"].copy()
        display["timestamp_et"] = pd.to_datetime(display["timestamp_et"]).dt.strftime("%Y-%m-%d %H:%M")
        print(_frame_text(display, max_rows=32))

    print(f"\nClassification: {classification[0]}")
    print(classification[1])


def save_outputs(
    output_dir: Path,
    *,
    base_presence: pd.DataFrame,
    raw_presence: pd.DataFrame,
    compare_presence: pd.DataFrame,
    alignment_impact: dict[str, Any],
    classification: tuple[str, str],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    base_presence.to_csv(output_dir / "aligned_presence.csv", index=False)
    summarize_timestamp_coverage(base_presence).to_csv(output_dir / "aligned_coverage.csv", index=False)
    if not raw_presence.empty:
        raw_presence.to_csv(output_dir / "raw_presence.csv", index=False)
        summarize_timestamp_coverage(raw_presence).to_csv(output_dir / "raw_coverage.csv", index=False)
        summarize_missing_symbol_patterns(raw_presence).to_csv(output_dir / "raw_missing_symbol_patterns.csv", index=False)
    if not compare_presence.empty:
        compare_presence.to_csv(output_dir / "compare_presence.csv", index=False)
        summarize_timestamp_coverage(compare_presence).to_csv(output_dir / "compare_coverage.csv", index=False)
    dropped = alignment_impact["dropped_timestamps"]
    if isinstance(dropped, pd.DataFrame):
        dropped.to_csv(output_dir / "alignment_dropped_timestamps.csv", index=False)
    payload = {"classification": classification[0], "reason": classification[1]}
    (output_dir / "summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main() -> None:
    args = parse_args()
    dataset_path = Path(args.dataset)
    timestamps_et = _parse_et_timestamps(args.timestamps)
    cli_symbols = _normalize_symbols(args.symbols)

    base_df, base_manifest = _load_dataset_frame(dataset_path, cli_symbols or None)
    if base_df.empty:
        raise RuntimeError("Primary dataset is empty after the requested symbol filter.")
    symbols = cli_symbols or sorted(base_df["symbol"].astype(str).str.upper().unique().tolist())
    base_presence = per_symbol_bar_presence(base_df, symbols, timestamps_et)

    raw_presence = pd.DataFrame()
    raw_label = None
    if args.raw_dataset:
        raw_dataset_path = Path(args.raw_dataset)
        raw_df, _ = _load_dataset_frame(raw_dataset_path, symbols)
        raw_presence = per_symbol_bar_presence(raw_df, symbols, timestamps_et)
        raw_label = f"Raw dataset ({raw_dataset_path.name})"

    compare_presence = pd.DataFrame()
    compare_label = None
    compare_covered_timestamps: set[str] = set()
    if args.compare_dataset:
        compare_dataset_path = Path(args.compare_dataset)
        compare_df, compare_manifest = _load_dataset_frame(compare_dataset_path, None)
        overlap_symbols = sorted(set(symbols) & set(compare_df["symbol"].astype(str).str.upper().unique().tolist()))
        compare_df = compare_df[compare_df["symbol"].isin(overlap_symbols)].copy()
        compare_presence = per_symbol_bar_presence(compare_df, overlap_symbols, timestamps_et)
        compare_label = str(compare_manifest.get("feed", compare_dataset_path.name))
        compare_covered_timestamps = set(compare_df["ts_et"].dt.strftime("%Y-%m-%d %H:%M").unique().tolist()) if not compare_df.empty else set()

    if args.raw_dataset:
        raw_df, _ = _load_dataset_frame(Path(args.raw_dataset), symbols)
        alignment_impact = simulate_alignment_impact(raw_df, symbols)
    else:
        alignment_impact = {
            "timestamp_count": 0,
            "shared_timestamp_count": 0,
            "dropped_timestamps": pd.DataFrame(columns=["timestamp_et", "symbols_present", "missing_symbols"]),
            "not_computable": "No raw/non-shared target-feed dataset was provided, so shared-alignment drop counts for the current IEX snapshot cannot be computed directly.",
        }
    classification = classify_issue(raw_presence=raw_presence, compare_presence=compare_presence)
    print_report(
        dataset_path=dataset_path,
        dataset_manifest=base_manifest,
        base_presence=base_presence,
        raw_presence=raw_presence,
        compare_presence=compare_presence,
        compare_covered_timestamps=compare_covered_timestamps,
        raw_label=raw_label,
        compare_label=compare_label,
        alignment_impact=alignment_impact,
        classification=classification,
    )

    if args.output_dir:
        save_outputs(
            Path(args.output_dir),
            base_presence=base_presence,
            raw_presence=raw_presence,
            compare_presence=compare_presence,
            alignment_impact=alignment_impact,
            classification=classification,
        )


if __name__ == "__main__":
    main()

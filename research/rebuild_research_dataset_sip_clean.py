from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pandas as pd

from backtest_runner import load_dataset
from rebuild_research_dataset_sip import (
    build_snapshotter_command,
    extract_rebuild_parameters,
    parse_snapshotter_output,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a cleaner SIP research dataset by restricting a SIP rebuild to regular session bars."
    )
    parser.add_argument("--reference-dataset", required=True, help="Reference dataset directory.")
    parser.add_argument("--shared-sip-dataset", help="Optional existing shared SIP dataset to reuse.")
    parser.add_argument("--output-dir", default="datasets", help="Dataset output root.")
    parser.add_argument("--session-filter", default="regular", choices=["regular"], help="Session filter mode.")
    parser.add_argument("--print-command", action="store_true", help="Print the rebuild command without executing it.")
    return parser.parse_args()


def filter_regular_session(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df.copy()
    session_df = df.copy()
    timestamps_et = session_df["timestamp"].dt.tz_convert("America/New_York")
    mask = (timestamps_et.dt.time >= pd.Timestamp("09:30").time()) & (timestamps_et.dt.time <= pd.Timestamp("15:45").time())
    return session_df.loc[mask].sort_values(["symbol", "timestamp"]).reset_index(drop=True)


def build_clean_dataset_dir(shared_sip_dataset: Path, output_root: Path) -> Path:
    return output_root / f"{shared_sip_dataset.name}__regular"


def build_clean_manifest(shared_manifest: dict[str, Any], clean_df: pd.DataFrame, shared_sip_dataset: Path) -> dict[str, Any]:
    symbol_count = int(clean_df["symbol"].nunique()) if not clean_df.empty else 0
    aligned_timestamps = int(clean_df["timestamp"].nunique()) if not clean_df.empty else 0
    per_symbol_counts = {
        str(symbol): int(count)
        for symbol, count in clean_df.groupby("symbol").size().to_dict().items()
    } if not clean_df.empty else {}
    manifest = dict(shared_manifest)
    raw_row_count = int(manifest.get("raw_row_count", len(clean_df)))
    manifest["dataset_id"] = f"{shared_sip_dataset.name}__regular"
    manifest["row_count"] = int(len(clean_df))
    manifest["aligned_row_count"] = int(len(clean_df))
    manifest["symbol_count"] = symbol_count
    manifest["aligned_timestamp_count"] = aligned_timestamps
    manifest["percent_rows_retained"] = ((len(clean_df) / raw_row_count) * 100.0) if raw_row_count else 0.0
    manifest["per_symbol_row_counts_after_alignment"] = per_symbol_counts
    before_counts = manifest.get("per_symbol_row_counts_before_alignment", {})
    manifest["symbol_retention_report"] = [
        {
            "symbol": str(symbol),
            "rows_before_alignment": int(before_counts.get(symbol, count)),
            "rows_after_alignment": int(count),
            "percent_retained": ((int(count) / int(before_counts.get(symbol, count))) * 100.0)
            if int(before_counts.get(symbol, count))
            else 0.0,
        }
        for symbol, count in sorted(per_symbol_counts.items())
    ]
    manifest["session_filter"] = "regular"
    manifest["parent_dataset"] = str(shared_sip_dataset)
    manifest["dataset_variant"] = "sip_clean_regular_session"
    manifest["feed_warning"] = None
    manifest["notes"] = "Regular-session-only SIP research dataset derived from a shared SIP rebuild."
    return manifest


def ensure_shared_sip_dataset(reference_dataset: Path, shared_sip_dataset: str | None, output_dir: str, print_command: bool) -> Path:
    if shared_sip_dataset:
        return Path(shared_sip_dataset)
    params = extract_rebuild_parameters(reference_dataset)
    command = build_snapshotter_command(params, feed="sip", output_dir=output_dir, align_mode="shared")
    print("Shared SIP rebuild command:")
    print(" ".join(command))
    if print_command:
        raise SystemExit(0)
    completed = subprocess.run(command, check=True, capture_output=True, text=True)
    if completed.stdout:
        print(completed.stdout.strip())
    if completed.stderr:
        print(completed.stderr.strip(), file=sys.stderr)
    parsed = parse_snapshotter_output(completed.stdout)
    manifest_path = parsed.get("manifest")
    if not manifest_path:
        raise RuntimeError("Unable to determine rebuilt shared SIP dataset path from snapshotter output.")
    return Path(manifest_path).parent


def main() -> None:
    args = parse_args()
    reference_dataset = Path(args.reference_dataset)
    output_root = Path(args.output_dir)
    shared_sip_dataset = ensure_shared_sip_dataset(
        reference_dataset=reference_dataset,
        shared_sip_dataset=args.shared_sip_dataset,
        output_dir=args.output_dir,
        print_command=args.print_command,
    )
    shared_df, shared_manifest = load_dataset(shared_sip_dataset)
    clean_df = filter_regular_session(shared_df)
    clean_dir = build_clean_dataset_dir(shared_sip_dataset, output_root)
    clean_dir.mkdir(parents=True, exist_ok=True)
    clean_df.to_parquet(clean_dir / "bars.parquet", index=False)
    clean_manifest = build_clean_manifest(shared_manifest if isinstance(shared_manifest, dict) else {}, clean_df, shared_sip_dataset)
    (clean_dir / "manifest.json").write_text(json.dumps(clean_manifest, indent=2), encoding="utf-8")
    print(f"shared_sip_dataset={shared_sip_dataset}")
    print(f"clean_sip_dataset={clean_dir}")
    print(f"rows={len(clean_df)}")
    print(f"timestamps={clean_df['timestamp'].nunique() if not clean_df.empty else 0}")


if __name__ == "__main__":
    main()

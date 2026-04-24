from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rebuild a research dataset from an existing reference manifest using SIP data."
    )
    parser.add_argument("--reference-dataset", required=True, help="Existing dataset directory with manifest.json.")
    parser.add_argument("--output-dir", default="datasets", help="Dataset output root.")
    parser.add_argument("--feed", default="sip", choices=["sip"], help="Target feed. Default: sip.")
    parser.add_argument(
        "--align-mode",
        help="Optional align-mode override. Defaults to the reference manifest align_mode.",
    )
    parser.add_argument(
        "--print-command",
        action="store_true",
        help="Print the dataset_snapshotter command without executing it.",
    )
    return parser.parse_args()


def extract_rebuild_parameters(reference_dataset: Path) -> dict[str, Any]:
    manifest_path = reference_dataset / "manifest.json"
    if not manifest_path.exists():
        raise RuntimeError(f"Reference dataset manifest not found: {manifest_path}")
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    symbols = payload.get("symbols")
    if not isinstance(symbols, list) or not symbols:
        raise RuntimeError("Reference manifest is missing symbols.")
    timeframe = payload.get("timeframe")
    start_utc = payload.get("start_utc")
    end_utc = payload.get("end_utc")
    if not all(isinstance(value, str) and value for value in (timeframe, start_utc, end_utc)):
        raise RuntimeError("Reference manifest is missing timeframe/start_utc/end_utc.")
    return {
        "reference_dataset": str(reference_dataset),
        "symbols": [str(symbol).strip().upper() for symbol in symbols if str(symbol).strip()],
        "timeframe": str(timeframe),
        "start_utc": str(start_utc),
        "end_utc": str(end_utc),
        "adjustment": str(payload.get("adjustment") or "raw"),
        "align_mode": str(payload.get("align_mode") or "shared"),
        "reference_feed": str(payload.get("feed") or "unknown"),
    }


def build_snapshotter_command(params: dict[str, Any], *, feed: str, output_dir: str, align_mode: str | None) -> list[str]:
    final_align_mode = align_mode or str(params["align_mode"])
    return [
        sys.executable,
        "dataset_snapshotter.py",
        "--symbols",
        *list(params["symbols"]),
        "--start",
        str(params["start_utc"]),
        "--end",
        str(params["end_utc"]),
        "--timeframe",
        str(params["timeframe"]),
        "--feed",
        str(feed),
        "--adjustment",
        str(params["adjustment"]),
        "--align-mode",
        final_align_mode,
        "--output-dir",
        str(output_dir),
    ]


def parse_snapshotter_output(stdout: str) -> dict[str, str]:
    parsed: dict[str, str] = {}
    for line in stdout.splitlines():
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        if key in {"dataset_id", "parquet", "manifest"}:
            parsed[key] = value.strip()
    return parsed


def main() -> None:
    args = parse_args()
    reference_dataset = Path(args.reference_dataset)
    params = extract_rebuild_parameters(reference_dataset)
    command = build_snapshotter_command(
        params,
        feed=args.feed,
        output_dir=args.output_dir,
        align_mode=args.align_mode,
    )
    print("Reference dataset:")
    print(reference_dataset)
    print(f"Reference feed: {params['reference_feed']}")
    print(f"Target feed:    {args.feed}")
    print("Command:")
    print(" ".join(command))
    if args.print_command:
        return

    completed = subprocess.run(
        command,
        check=True,
        capture_output=True,
        text=True,
    )
    if completed.stdout:
        print(completed.stdout.strip())
    if completed.stderr:
        print(completed.stderr.strip(), file=sys.stderr)
    parsed = parse_snapshotter_output(completed.stdout)
    if parsed.get("manifest"):
        print(f"sip_manifest={parsed['manifest']}")
    if parsed.get("parquet"):
        print(f"sip_parquet={parsed['parquet']}")


if __name__ == "__main__":
    main()

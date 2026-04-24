from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from backtest_runner import load_dataset
from inspect_dataset_and_run import _frame_text, _infer_timeframe_from_df, _infer_timeframe_from_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit dataset timestamp spacing and distinguish benign session structure from harmful intraday gaps."
    )
    parser.add_argument("--dataset", required=True, help="Dataset directory.")
    parser.add_argument("--symbols", nargs="*", help="Optional symbol subset.")
    parser.add_argument("--expected-timeframe", help="Optional timeframe override, e.g. 15Min.")
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


def _timeframe_minutes(timeframe_label: str | None) -> int | None:
    if not timeframe_label:
        return None
    label = str(timeframe_label).strip()
    if label.endswith("Min"):
        try:
            return int(label[:-3])
        except ValueError:
            return None
    mapping = {"1H": 60, "4H": 240, "1D": 1440}
    return mapping.get(label)


def _load_spacing_frame(dataset_path: Path, symbols: list[str] | None) -> tuple[pd.DataFrame, dict[str, Any], str | None]:
    df, manifest = load_dataset(dataset_path)
    manifest = manifest if isinstance(manifest, dict) else {}
    if symbols:
        df = df[df["symbol"].isin(symbols)].copy()
    if df.empty:
        return df, manifest, None
    df = df.sort_values(["symbol", "timestamp"]).reset_index(drop=True).copy()
    df["ts_et"] = df["timestamp"].dt.tz_convert("America/New_York")
    df["day"] = df["ts_et"].dt.normalize()
    df["time_str"] = df["ts_et"].dt.strftime("%H:%M")
    timeframe = (
        str(manifest.get("timeframe"))
        if isinstance(manifest.get("timeframe"), str)
        else _infer_timeframe_from_path(dataset_path) or _infer_timeframe_from_df(df)
    )
    return df, manifest, timeframe


def infer_session_template(global_grid: pd.DataFrame, expected_minutes: int) -> dict[str, Any]:
    columns = ["day", "start_time", "end_time", "bars"]
    if global_grid.empty:
        return {"session_table": pd.DataFrame(columns=columns), "typical_start": None, "typical_end": None, "typical_bars": 0}
    rows: list[dict[str, Any]] = []
    for day, group in global_grid.groupby("day", sort=True):
        actual = list(group["ts_et"].sort_values())
        rows.append(
            {
                "day": day,
                "start_time": actual[0].strftime("%H:%M"),
                "end_time": actual[-1].strftime("%H:%M"),
                "bars": len(actual),
            }
        )
    table = pd.DataFrame(rows, columns=columns)
    typical_start = table["start_time"].mode().iloc[0] if not table.empty else None
    typical_end = table["end_time"].mode().iloc[0] if not table.empty else None
    typical_bars = 0
    if typical_start and typical_end and expected_minutes:
        typical_bars = len(
            pd.date_range(
                pd.Timestamp(f"2026-01-01 {typical_start}", tz="America/New_York"),
                pd.Timestamp(f"2026-01-01 {typical_end}", tz="America/New_York"),
                freq=f"{expected_minutes}min",
            )
        )
    return {
        "session_table": table,
        "typical_start": typical_start,
        "typical_end": typical_end,
        "typical_bars": typical_bars,
    }


def classify_day_spacing(
    *,
    actual_times: list[pd.Timestamp],
    typical_start: str | None,
    typical_end: str | None,
    expected_minutes: int,
) -> tuple[str, list[str]]:
    if not actual_times or typical_start is None or typical_end is None or expected_minutes <= 0:
        return "unknown", []
    day = actual_times[0].normalize()
    expected = pd.date_range(
        day + pd.Timedelta(hours=int(typical_start.split(":")[0]), minutes=int(typical_start.split(":")[1])),
        day + pd.Timedelta(hours=int(typical_end.split(":")[0]), minutes=int(typical_end.split(":")[1])),
        freq=f"{expected_minutes}min",
        tz="America/New_York",
    )
    actual_set = {ts.strftime("%H:%M") for ts in actual_times}
    missing = [ts.strftime("%H:%M") for ts in expected if ts.strftime("%H:%M") not in actual_set]
    actual_start = actual_times[0].strftime("%H:%M")
    actual_end = actual_times[-1].strftime("%H:%M")
    if not missing:
        return "regular", []
    if actual_start != typical_start or actual_end != typical_end:
        return "partial_or_shifted_session", missing
    return "within_session_missing", missing


def audit_spacing(dataset_path: Path, symbols: list[str] | None, expected_timeframe: str | None) -> dict[str, Any]:
    df, manifest, detected_timeframe = _load_spacing_frame(dataset_path, symbols)
    timeframe = expected_timeframe or detected_timeframe
    expected_minutes = _timeframe_minutes(timeframe)
    if df.empty:
        return {
            "dataset_path": dataset_path,
            "timeframe": timeframe,
            "symbol_count": 0,
            "summary": pd.DataFrame(),
            "global_day_audit": pd.DataFrame(),
            "symbol_gap_summary": pd.DataFrame(),
            "shared_timestamp_issues": pd.DataFrame(),
            "classification": ("not ready", "Dataset is empty after the requested symbol filter."),
        }

    symbol_count = int(df["symbol"].nunique())
    global_grid = df[["timestamp", "ts_et", "day", "time_str"]].drop_duplicates().sort_values("timestamp").reset_index(drop=True)
    session_template = infer_session_template(global_grid, expected_minutes or 0)
    typical_start = session_template["typical_start"]
    typical_end = session_template["typical_end"]
    typical_bars = int(session_template["typical_bars"])

    day_rows: list[dict[str, Any]] = []
    for day, group in global_grid.groupby("day", sort=True):
        actual_times = list(group["ts_et"].sort_values())
        classification, missing = classify_day_spacing(
            actual_times=actual_times,
            typical_start=typical_start,
            typical_end=typical_end,
            expected_minutes=expected_minutes or 0,
        )
        diffs = pd.Series(actual_times).diff().dropna().dt.total_seconds().div(60) if len(actual_times) > 1 else pd.Series(dtype=float)
        day_rows.append(
            {
                "day": pd.Timestamp(day).strftime("%Y-%m-%d"),
                "bars": len(actual_times),
                "start_time": actual_times[0].strftime("%H:%M"),
                "end_time": actual_times[-1].strftime("%H:%M"),
                "classification": classification,
                "missing_intervals": ", ".join(missing),
                "odd_diff_count": int((diffs.round() != (expected_minutes or 0)).sum()) if expected_minutes else 0,
            }
        )
    global_day_audit = pd.DataFrame(day_rows).sort_values("day") if day_rows else pd.DataFrame(
        columns=["day", "bars", "start_time", "end_time", "classification", "missing_intervals", "odd_diff_count"]
    )

    timestamp_symbol_counts = df.groupby("timestamp")["symbol"].nunique().reset_index(name="symbols_present")
    shared_timestamp_issues = timestamp_symbol_counts[timestamp_symbol_counts["symbols_present"] != symbol_count].copy()
    if not shared_timestamp_issues.empty:
        shared_timestamp_issues["timestamp_et"] = pd.to_datetime(shared_timestamp_issues["timestamp"], utc=True).dt.tz_convert("America/New_York")
        shared_timestamp_issues["missing_symbols"] = symbol_count - shared_timestamp_issues["symbols_present"]

    symbol_rows: list[dict[str, Any]] = []
    global_day_lookup = {
        row["day"]: set(str(row["missing_intervals"]).split(", ")) if row["missing_intervals"] else set()
        for row in global_day_audit.to_dict("records")
    }
    for symbol, group in df.groupby("symbol", sort=True):
        symbol_day_times = group.groupby("day")["time_str"].agg(list)
        harmful_days = 0
        missing_slots = 0
        for day, times in symbol_day_times.items():
            row = global_day_audit[global_day_audit["day"] == pd.Timestamp(day).strftime("%Y-%m-%d")]
            if row.empty:
                continue
            if row.iloc[0]["classification"] == "within_session_missing":
                harmful_days += 1
                missing_slots += len([slot for slot in str(row.iloc[0]["missing_intervals"]).split(", ") if slot])
        symbol_rows.append(
            {
                "symbol": symbol,
                "bars": int(len(group)),
                "trading_days": int(group["day"].nunique()),
                "within_session_missing_days": harmful_days,
                "missing_slots_from_typical_session": missing_slots,
            }
        )
    symbol_gap_summary = pd.DataFrame(symbol_rows).sort_values(["within_session_missing_days", "symbol"], ascending=[False, True])

    total_days = int(global_day_audit["day"].nunique()) if not global_day_audit.empty else 0
    harmful_days = int((global_day_audit["classification"] == "within_session_missing").sum()) if not global_day_audit.empty else 0
    partial_days = int((global_day_audit["classification"] == "partial_or_shifted_session").sum()) if not global_day_audit.empty else 0
    total_missing_slots = 0
    if not global_day_audit.empty:
        total_missing_slots = sum(
            len([slot for slot in str(value).split(", ") if slot])
            for value in global_day_audit["missing_intervals"]
        )

    if harmful_days == 0 and partial_days == 0 and shared_timestamp_issues.empty:
        classification = ("benign", "The dataset uses a consistent intraday timestamp grid; no within-session spacing problems were detected.")
    elif harmful_days == 0 and shared_timestamp_issues.empty:
        classification = (
            "partly benign",
            "Spacing irregularities are explained by session-shape differences rather than missing intraday bars. The warning logic should treat partial sessions separately.",
        )
    else:
        classification = (
            "real integrity issue",
            f"The dataset has {total_missing_slots} missing within-session 15-minute slots across {harmful_days} trading days; this is not explained by overnight or weekend gaps.",
        )

    summary = pd.DataFrame(
        [
            {
                "dataset": str(dataset_path),
                "timeframe": timeframe or "unknown",
                "symbols": symbol_count,
                "trading_days": total_days,
                "shared_timestamp_grid": bool(shared_timestamp_issues.empty),
                "typical_session": f"{typical_start}-{typical_end}" if typical_start and typical_end else "unknown",
                "typical_bars_per_day": typical_bars,
                "within_session_missing_days": harmful_days,
                "partial_or_shifted_days": partial_days,
                "total_missing_slots": total_missing_slots,
            }
        ]
    )

    return {
        "dataset_path": dataset_path,
        "manifest": manifest,
        "timeframe": timeframe,
        "summary": summary,
        "global_day_audit": global_day_audit,
        "symbol_gap_summary": symbol_gap_summary,
        "shared_timestamp_issues": shared_timestamp_issues,
        "classification": classification,
    }


def print_audit(result: dict[str, Any]) -> None:
    print("\nDATASET SPACING AUDIT")
    print("---------------------")
    print(_frame_text(result["summary"]))
    print("\nDay-level spacing:")
    print(_frame_text(result["global_day_audit"], max_rows=64))
    print("\nPer-symbol summary:")
    print(_frame_text(result["symbol_gap_summary"], max_rows=64))
    issues = result["shared_timestamp_issues"]
    if isinstance(issues, pd.DataFrame) and not issues.empty:
        print("\nShared timestamp issues:")
        display = issues.copy()
        if "timestamp_et" in display.columns:
            display["timestamp_et"] = pd.to_datetime(display["timestamp_et"]).dt.strftime("%Y-%m-%d %H:%M")
        print(_frame_text(display, max_rows=32))
    print(f"\nConclusion: {result['classification'][0]}")
    print(result["classification"][1])


def save_outputs(result: dict[str, Any], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    result["summary"].to_csv(output_dir / "spacing_summary.csv", index=False)
    result["global_day_audit"].to_csv(output_dir / "global_day_audit.csv", index=False)
    result["symbol_gap_summary"].to_csv(output_dir / "symbol_gap_summary.csv", index=False)
    issues = result["shared_timestamp_issues"]
    if isinstance(issues, pd.DataFrame):
        issues.to_csv(output_dir / "shared_timestamp_issues.csv", index=False)
    payload = {
        "dataset": str(result["dataset_path"]),
        "timeframe": result["timeframe"],
        "classification": result["classification"][0],
        "reason": result["classification"][1],
    }
    (output_dir / "spacing_summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main() -> None:
    args = parse_args()
    dataset_path = Path(args.dataset)
    symbols = _normalize_symbols(args.symbols)
    result = audit_spacing(dataset_path, symbols or None, args.expected_timeframe)
    print_audit(result)
    if args.output_dir:
        save_outputs(result, Path(args.output_dir))


if __name__ == "__main__":
    main()

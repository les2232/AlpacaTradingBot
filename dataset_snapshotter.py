import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
from alpaca.data.enums import Adjustment, DataFeed
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
from dotenv import load_dotenv


DATASET_ROOT = Path("datasets")
CODE_HASH_FILES = (
    "dataset_snapshotter.py",
    "trading_bot.py",
    "storage.py",
    "dashboard.py",
    "requirements.txt",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Download historical Alpaca bars for a symbol set and store a versioned offline "
            "Parquet snapshot plus manifest metadata."
        )
    )
    parser.add_argument("--symbols", nargs="+", required=True, help="Ticker symbols, e.g. AAPL MSFT NVDA")
    parser.add_argument("--start", required=True, help="Inclusive UTC start date or timestamp.")
    parser.add_argument("--end", required=True, help="Exclusive UTC end date or timestamp.")
    parser.add_argument(
        "--timeframe",
        default="15Min",
        help="Bar timeframe such as 1Min, 5Min, 15Min, 1Hour, or 1Day.",
    )
    parser.add_argument(
        "--feed",
        default="iex",
        choices=["iex", "sip", "otc"],
        help="Alpaca data feed.",
    )
    parser.add_argument(
        "--adjustment",
        default="raw",
        choices=["raw", "split", "all"],
        help="Corporate action adjustment mode.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DATASET_ROOT),
        help="Root directory for versioned dataset snapshots.",
    )
    parser.add_argument(
        "--align-mode",
        default="shared",
        choices=["shared", "none"],
        help="Alignment mode: shared keeps only timestamps present for every symbol; none keeps all rows.",
    )
    return parser.parse_args()


def parse_datetime(value: str) -> datetime:
    normalized = value.strip()
    if normalized.endswith("Z"):
        normalized = normalized[:-1] + "+00:00"
    parsed = datetime.fromisoformat(normalized)
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def parse_timeframe(value: str) -> TimeFrame:
    normalized = value.strip().lower()
    # Longest suffixes first so "minutes" is matched before "min", etc.
    units = [
        ("minutes", TimeFrameUnit.Minute),
        ("minute", TimeFrameUnit.Minute),
        ("mins", TimeFrameUnit.Minute),
        ("min", TimeFrameUnit.Minute),
        ("hours", TimeFrameUnit.Hour),
        ("hour", TimeFrameUnit.Hour),
        ("days", TimeFrameUnit.Day),
        ("day", TimeFrameUnit.Day),
    ]

    for suffix, unit in units:
        if normalized.endswith(suffix):
            amount_text = normalized[: -len(suffix)].strip()
            if not amount_text.isdigit():
                continue
            amount = int(amount_text)
            if amount <= 0:
                continue
            return TimeFrame(amount=amount, unit=unit)  # type: ignore[call-arg]

    raise ValueError(f"Unsupported timeframe: {value}")


def parse_feed(value: str) -> DataFeed:
    return {
        "iex": DataFeed.IEX,
        "sip": DataFeed.SIP,
        "otc": DataFeed.OTC,
    }[value.lower()]


def parse_adjustment(value: str) -> Adjustment:
    return {
        "raw": Adjustment.RAW,
        "split": Adjustment.SPLIT,
        "all": Adjustment.ALL,
    }[value.lower()]


def compute_code_hash(repo_root: Path) -> str:
    digest = hashlib.sha256()
    for relative_path in CODE_HASH_FILES:
        path = repo_root / relative_path
        digest.update(relative_path.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()[:12]


def build_dataset_id(
    symbols: list[str],
    timeframe_text: str,
    start: datetime,
    end: datetime,
    feed: str,
    code_hash: str,
) -> str:
    sorted_syms = sorted(symbols)
    symbols_key = "-".join(sorted_syms)
    # Windows MAX_PATH is 260 chars; hash the symbol list when it would exceed safe limits.
    if len(symbols_key) > 100:
        sym_hash = hashlib.sha256(symbols_key.encode()).hexdigest()[:8]
        symbols_key = f"{len(sorted_syms)}syms_{sym_hash}"
    timeframe_key = timeframe_text.replace(" ", "")
    start_key = start.strftime("%Y%m%dT%H%M%SZ")
    end_key = end.strftime("%Y%m%dT%H%M%SZ")
    return f"{symbols_key}__{timeframe_key}__{start_key}__{end_key}__{feed}__{code_hash}"


def fetch_bars(
    client: StockHistoricalDataClient,
    symbols: list[str],
    timeframe: TimeFrame,
    start: datetime,
    end: datetime,
    feed: DataFeed,
    adjustment: Adjustment,
) -> pd.DataFrame:
    request = StockBarsRequest(
        symbol_or_symbols=symbols,
        timeframe=timeframe,
        start=start,
        end=end,
        feed=feed,
        adjustment=adjustment,
    )
    response = client.get_stock_bars(request)
    rows: list[dict[str, Any]] = []
    for symbol in symbols:
        symbol_bars = getattr(response, "data", {}).get(symbol, [])
        for bar in symbol_bars:
            timestamp = getattr(bar, "timestamp", None)
            rows.append(
                {
                    "symbol": symbol,
                    "timestamp": pd.to_datetime(timestamp, utc=True) if timestamp is not None else None,
                    "open": float(getattr(bar, "open", 0.0)),
                    "high": float(getattr(bar, "high", 0.0)),
                    "low": float(getattr(bar, "low", 0.0)),
                    "close": float(getattr(bar, "close", 0.0)),
                    "volume": int(getattr(bar, "volume", 0) or 0),
                    "trade_count": int(getattr(bar, "trade_count", 0) or 0),
                    "vwap": (
                        float(getattr(bar, "vwap", 0.0))
                        if getattr(bar, "vwap", None) is not None
                        else None
                    ),
                }
            )

    if not rows:
        return pd.DataFrame(
            columns=[
                "symbol",
                "timestamp",
                "open",
                "high",
                "low",
                "close",
                "volume",
                "trade_count",
                "vwap",
            ]
        )

    dataset = pd.DataFrame(rows)
    dataset = dataset.sort_values(["symbol", "timestamp"]).reset_index(drop=True)
    return dataset


def align_symbols_on_shared_timestamps(dataset: pd.DataFrame, symbols: list[str]) -> pd.DataFrame:
    if dataset.empty:
        return dataset

    timestamp_counts = (
        dataset.dropna(subset=["timestamp"])
        .groupby("timestamp")["symbol"]
        .nunique()
    )
    shared_timestamps = timestamp_counts[timestamp_counts == len(symbols)].index
    aligned = dataset[dataset["timestamp"].isin(shared_timestamps)].copy()
    return aligned.sort_values(["symbol", "timestamp"]).reset_index(drop=True)


def compute_symbol_row_counts(dataset: pd.DataFrame, symbols: list[str]) -> dict[str, int]:
    if dataset.empty:
        return {symbol: 0 for symbol in symbols}

    counts = dataset.groupby("symbol").size().to_dict()
    return {symbol: int(counts.get(symbol, 0)) for symbol in symbols}


def build_symbol_retention_report(
    symbols: list[str],
    before_counts: dict[str, int],
    after_counts: dict[str, int],
) -> list[dict[str, Any]]:
    report: list[dict[str, Any]] = []
    for symbol in symbols:
        before = int(before_counts.get(symbol, 0))
        after = int(after_counts.get(symbol, 0))
        percent_retained = (after / before * 100.0) if before > 0 else 0.0
        report.append(
            {
                "symbol": symbol,
                "rows_before_alignment": before,
                "rows_after_alignment": after,
                "percent_retained": percent_retained,
            }
        )
    report.sort(key=lambda row: (row["percent_retained"], row["rows_after_alignment"], row["symbol"]))
    return report


def write_dataset(
    dataset: pd.DataFrame,
    manifest: dict[str, Any],
    output_root: Path,
    dataset_id: str,
) -> tuple[Path, Path]:
    dataset_dir = output_root / dataset_id
    dataset_dir.mkdir(parents=True, exist_ok=True)

    parquet_path = dataset_dir / "bars.parquet"
    manifest_path = dataset_dir / "manifest.json"

    dataset.to_parquet(parquet_path, index=False)
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return parquet_path, manifest_path


def main() -> None:
    load_dotenv(Path.cwd() / ".env")

    api_key = os.getenv("ALPACA_API_KEY")
    api_secret = os.getenv("ALPACA_API_SECRET")
    if not api_key or not api_secret:
        raise RuntimeError("Missing Alpaca credentials. Set ALPACA_API_KEY and ALPACA_API_SECRET in .env.")

    args = parse_args()
    repo_root = Path.cwd()
    output_root = Path(args.output_dir)

    symbols = sorted({symbol.strip().upper() for symbol in args.symbols if symbol.strip()})
    if not symbols:
        raise RuntimeError("At least one symbol is required.")

    start = parse_datetime(args.start)
    end = parse_datetime(args.end)
    if end <= start:
        raise RuntimeError("End time must be later than start time.")

    timeframe = parse_timeframe(args.timeframe)
    feed = parse_feed(args.feed)
    adjustment = parse_adjustment(args.adjustment)
    code_hash = compute_code_hash(repo_root)
    dataset_id = build_dataset_id(symbols, args.timeframe, start, end, args.feed.lower(), code_hash)

    client = StockHistoricalDataClient(api_key, api_secret)
    print(f"requested_feed={args.feed.lower()}")
    try:
        dataset = fetch_bars(client, symbols, timeframe, start, end, feed, adjustment)
    except Exception as exc:
        if args.feed.lower() == "sip":
            raise RuntimeError(
                "SIP data request failed. This usually means the account does not have SIP market-data "
                "permissions or subscription access enabled. Try `--feed iex` or enable SIP access in Alpaca."
            ) from exc
        raise
    raw_row_count = int(len(dataset))
    raw_symbol_row_counts = compute_symbol_row_counts(dataset, symbols)
    if args.align_mode == "shared":
        dataset = align_symbols_on_shared_timestamps(dataset, symbols)
    elif args.align_mode != "none":
        raise RuntimeError(f"Unsupported align mode: {args.align_mode}")

    aligned_row_count = int(len(dataset))
    aligned_timestamp_count = int(dataset["timestamp"].nunique()) if not dataset.empty else 0
    percent_rows_retained = (
        (aligned_row_count / raw_row_count) * 100.0
        if raw_row_count > 0
        else 0.0
    )
    aligned_symbol_row_counts = compute_symbol_row_counts(dataset, symbols)
    symbol_retention_report = build_symbol_retention_report(
        symbols,
        raw_symbol_row_counts,
        aligned_symbol_row_counts,
    )

    manifest = {
        "dataset_id": dataset_id,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "symbols": symbols,
        "timeframe": args.timeframe,
        "start_utc": start.isoformat(),
        "end_utc": end.isoformat(),
        "feed": args.feed.lower(),
        "feed_warning": (
            "IEX is single-exchange data and can distort volume-sensitive intraday research."
            if args.feed.lower() == "iex"
            else None
        ),
        "adjustment": args.adjustment.lower(),
        "align_mode": args.align_mode,
        "code_hash": code_hash,
        "row_count": aligned_row_count,
        "raw_row_count": raw_row_count,
        "aligned_row_count": aligned_row_count,
        "symbol_count": int(len(symbols)),
        "aligned_timestamp_count": aligned_timestamp_count,
        "percent_rows_retained": percent_rows_retained,
        "per_symbol_row_counts_before_alignment": raw_symbol_row_counts,
        "per_symbol_row_counts_after_alignment": aligned_symbol_row_counts,
        "symbol_retention_report": symbol_retention_report,
        "columns": list(dataset.columns),
        "source": "alpaca_stock_bars",
        "code_hash_files": list(CODE_HASH_FILES),
    }

    parquet_path, manifest_path = write_dataset(dataset, manifest, output_root, dataset_id)
    print(f"dataset_id={dataset_id}")
    print(f"raw_rows={raw_row_count}")
    print(f"aligned_rows={aligned_row_count}")
    print(f"percent_retained={percent_rows_retained:.2f}%")
    print(f"aligned_timestamps={aligned_timestamp_count}")
    print("symbol_retention_report:")
    print(f"{'symbol':<8}  {'before':>8}  {'after':>8}  {'retained':>9}")
    for row in symbol_retention_report:
        print(
            f"{row['symbol']:<8}  "
            f"{row['rows_before_alignment']:>8}  "
            f"{row['rows_after_alignment']:>8}  "
            f"{row['percent_retained']:>8.2f}%"
        )
    if percent_rows_retained < 90.0:
        print("warning=alignment retained less than 90% of rows")
        if args.align_mode == "shared":
            print("worst_symbols_for_shared_alignment:")
            for row in symbol_retention_report[:3]:
                print(
                    f"{row['symbol']}: before={row['rows_before_alignment']}, "
                    f"after={row['rows_after_alignment']}, retained={row['percent_retained']:.2f}%"
                )
    print(f"parquet={parquet_path}")
    print(f"manifest={manifest_path}")


if __name__ == "__main__":
    main()

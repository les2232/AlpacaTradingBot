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
    symbols_key = "-".join(sorted(symbols))
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
    dataset = fetch_bars(client, symbols, timeframe, start, end, feed, adjustment)

    manifest = {
        "dataset_id": dataset_id,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "symbols": symbols,
        "timeframe": args.timeframe,
        "start_utc": start.isoformat(),
        "end_utc": end.isoformat(),
        "feed": args.feed.lower(),
        "adjustment": args.adjustment.lower(),
        "code_hash": code_hash,
        "row_count": int(len(dataset)),
        "symbol_count": int(len(symbols)),
        "columns": list(dataset.columns),
        "source": "alpaca_stock_bars",
        "code_hash_files": list(CODE_HASH_FILES),
    }

    parquet_path, manifest_path = write_dataset(dataset, manifest, output_root, dataset_id)
    print(f"dataset_id={dataset_id}")
    print(f"rows={len(dataset)}")
    print(f"parquet={parquet_path}")
    print(f"manifest={manifest_path}")


if __name__ == "__main__":
    main()
"""
ml/feature_pipeline.py
----------------------
Convert raw bar data into a feature DataFrame for offline ML training.

Design contract
---------------
The core 7 features (ret_1, ret_3, ret_5, dist_sma_10, dist_sma_20,
rolling_vol, vol_ratio) are computed identically to _build_feature_vector
in backtest_runner.py.  A model trained on this output is therefore
directly usable by the live inference path (predict.py → MlSignal).

Additional features (ret_2, ret_4, time-of-day) are appended after the
core set and are clearly separated by CORE_FEATURE_COLS / EXTRA_FEATURE_COLS.

Input requirements
------------------
DataFrame with columns: symbol, timestamp, open, high, low, close, volume
Timestamps must be timezone-aware (UTC preferred).

Output
------
DataFrame with all original columns + feature columns + 'target'.
Rows with any NaN in features or target are dropped (warm-up + label horizon).
"""

import pandas as pd

# ---------------------------------------------------------------------------
# Config — tune these without touching any other code
# ---------------------------------------------------------------------------

# Rolling window periods — must match backtest_runner._build_feature_vector.
SMA_SHORT_PERIOD = 10   # bars for dist_sma_10 and rolling_vol
SMA_LONG_PERIOD  = 20   # bars for dist_sma_20
VOL_PERIOD       = 10   # bars for rolling return std (volatility)

# Forward-return label: target = 1 if close rises > threshold over next N bars.
LABEL_HORIZON    = 4     # bars ahead
LABEL_THRESHOLD  = 0.003  # 0.3 %

# ---------------------------------------------------------------------------
# Feature name lists (used externally by train.py / predict.py)
# ---------------------------------------------------------------------------

# These 7 features match backtest_runner._build_feature_vector exactly.
# Any model trained on them is a drop-in for the live loop.
CORE_FEATURE_COLS: list[str] = [
    "ret_1",        # 1-bar return
    "ret_3",        # 3-bar return
    "ret_5",        # 5-bar return
    "dist_sma_10",  # (close / sma_10) - 1
    "dist_sma_20",  # (close / sma_20) - 1
    "rolling_vol",  # rolling std of 1-bar returns over VOL_PERIOD
    "vol_ratio",    # (volume / avg_volume_10) - 1
]

# Extra features not in the live loop; useful for offline analysis.
EXTRA_FEATURE_COLS: list[str] = [
    "ret_2",        # 2-bar return
    "ret_4",        # 4-bar return
    "bar_hour",     # UTC hour of bar timestamp
    "bar_minute",   # minute of bar timestamp (0, 15, 30, 45 for 15-Min bars)
    "minutes_utc",  # bar_hour * 60 + bar_minute — ordinal position in the day
]

ALL_FEATURE_COLS: list[str] = CORE_FEATURE_COLS + EXTRA_FEATURE_COLS

LABEL_COL = "target"


# ---------------------------------------------------------------------------
# Internal: per-symbol feature helpers
# ---------------------------------------------------------------------------

def _core_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute the 7 core features, matching backtest_runner._build_feature_vector.

    Minimum warm-up: SMA_LONG_PERIOD (20) bars before the first valid row.
    """
    close  = df["close"]
    volume = df["volume"]

    # Returns
    df["ret_1"] = close / close.shift(1) - 1.0
    df["ret_3"] = close / close.shift(3) - 1.0
    df["ret_5"] = close / close.shift(5) - 1.0

    # SMA distances
    sma_10 = close.rolling(SMA_SHORT_PERIOD).mean()
    sma_20 = close.rolling(SMA_LONG_PERIOD).mean()
    df["dist_sma_10"] = close / sma_10 - 1.0
    df["dist_sma_20"] = close / sma_20 - 1.0

    # Rolling volatility: std of 1-bar returns over VOL_PERIOD bars
    df["rolling_vol"] = (close / close.shift(1) - 1.0).rolling(VOL_PERIOD).std()

    # Volume ratio vs. short-period average
    avg_vol_10 = volume.rolling(SMA_SHORT_PERIOD).mean()
    df["vol_ratio"] = volume / avg_vol_10.clip(lower=1.0) - 1.0

    return df


def _extra_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute extra features not present in the live loop."""
    close = df["close"]

    df["ret_2"] = close / close.shift(2) - 1.0
    df["ret_4"] = close / close.shift(4) - 1.0

    # Time-of-day from UTC timestamp.
    # NYSE regular session: 14:30–21:00 UTC.
    ts = pd.to_datetime(df["timestamp"]).dt
    df["bar_hour"]    = ts.hour
    df["bar_minute"]  = ts.minute
    df["minutes_utc"] = ts.hour * 60 + ts.minute

    return df


def _add_label(df: pd.DataFrame) -> pd.DataFrame:
    """
    Binary label: 1 if the close rises > LABEL_THRESHOLD over the next
    LABEL_HORIZON bars.  The final LABEL_HORIZON rows cannot have a valid
    label and are left as NaN so they get dropped by the caller.
    """
    future_return = df["close"].shift(-LABEL_HORIZON) / df["close"] - 1.0
    df[LABEL_COL] = (future_return > LABEL_THRESHOLD).astype(float)
    # Rows where future close doesn't exist → NaN → dropped downstream.
    df.loc[df.index[-LABEL_HORIZON:], LABEL_COL] = float("nan")
    return df


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_features(
    df: pd.DataFrame,
    include_label: bool = True,
) -> pd.DataFrame:
    """
    Compute all features (and optionally the label) for a single-symbol DataFrame.

    Parameters
    ----------
    df : DataFrame
        Must contain: timestamp, close, volume.
        Rows must be in chronological order; the function sorts them if not.
    include_label : bool
        If True, compute and attach the binary 'target' column.

    Returns
    -------
    DataFrame
        Original columns + ALL_FEATURE_COLS [+ LABEL_COL].
        Rows with NaN in any feature or the label are dropped (rolling warm-up
        and the final LABEL_HORIZON rows when include_label=True).
    """
    required = {"timestamp", "close", "volume"}
    missing  = required - set(df.columns)
    if missing:
        raise ValueError(f"build_features: missing required columns {missing}")

    df = df.copy().sort_values("timestamp").reset_index(drop=True)
    df = _core_features(df)
    df = _extra_features(df)

    drop_subset = list(ALL_FEATURE_COLS)
    if include_label:
        df = _add_label(df)
        drop_subset.append(LABEL_COL)

    df = df.dropna(subset=drop_subset).reset_index(drop=True)
    return df


def build_features_multi_symbol(
    df: pd.DataFrame,
    include_label: bool = True,
) -> pd.DataFrame:
    """
    Apply build_features to every symbol in df and return a combined DataFrame.

    Parameters
    ----------
    df : DataFrame
        Must have a 'symbol' column plus all columns required by build_features.

    Returns
    -------
    DataFrame sorted by (symbol, timestamp).

    Notes
    -----
    Features are computed independently per symbol so rolling windows never
    mix data across tickers.
    """
    if "symbol" not in df.columns:
        raise ValueError("build_features_multi_symbol: 'symbol' column is required")

    parts = []
    for symbol, group in df.groupby("symbol", sort=True):
        processed = build_features(group, include_label=include_label)
        parts.append(processed)

    if not parts:
        raise ValueError("build_features_multi_symbol: no symbols found in DataFrame")

    combined = pd.concat(parts, ignore_index=True)
    return combined.sort_values(["symbol", "timestamp"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Quick self-test  (python -m ml.feature_pipeline)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from pathlib import Path

    datasets_dir = Path(__file__).resolve().parent.parent / "datasets"
    if not datasets_dir.is_dir():
        raise SystemExit("No datasets/ directory found. Run dataset_snapshotter.py first.")

    dataset_path = max(datasets_dir.iterdir(), key=lambda p: p.stat().st_mtime)
    print(f"Using dataset: {dataset_path.name}")

    raw = pd.read_parquet(dataset_path / "bars.parquet")
    print(f"Raw bars: {len(raw)} rows, {raw['symbol'].nunique()} symbols")

    featured = build_features_multi_symbol(raw, include_label=True)
    print(f"Feature rows: {len(featured)} (after warm-up + label trim)")
    print(f"Columns: {list(featured.columns)}")
    print()

    for sym, grp in featured.groupby("symbol"):
        pos  = int(grp[LABEL_COL].sum())
        neg  = int((grp[LABEL_COL] == 0).sum())
        rate = pos / len(grp) * 100
        print(f"  {sym:>6}  rows={len(grp):>5}  target=1: {pos:>4} ({rate:.1f}%)  target=0: {neg:>4}")

    print()
    print("Core feature sample (first row):")
    print(featured[CORE_FEATURE_COLS].iloc[0].to_string())
    print()
    print("Extra feature sample (first row):")
    print(featured[EXTRA_FEATURE_COLS].iloc[0].to_string())

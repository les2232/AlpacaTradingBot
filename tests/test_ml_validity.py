"""Regression tests for offline ML feature/split validity."""

import pandas as pd
import pytest

from backtest_runner import _build_feature_vector
from ml.feature_pipeline import CORE_FEATURE_COLS, LABEL_HORIZON, build_features
from ml.train import _split_per_symbol


@pytest.fixture
def feature_parity_bars() -> tuple[pd.DataFrame, list[float], list[float]]:
    """Bars containing both a zero-volatility window and varying returns."""
    closes = [100.0] * 20 + [100.5, 100.25, 101.0, 100.75, 101.5, 101.25, 102.0]
    volumes = [1_000.0 + (idx % 5) * 125.0 for idx in range(len(closes))]
    bars = pd.DataFrame(
        {
            "bar_index": range(len(closes)),
            "timestamp": pd.date_range("2026-01-02 14:30", periods=len(closes), freq="15min", tz="UTC"),
            "close": closes,
            "volume": volumes,
        }
    )
    return bars, closes, volumes


def test_offline_core_features_match_backtest_vector(feature_parity_bars) -> None:
    bars, closes, volumes = feature_parity_bars

    offline = build_features(bars, include_label=False)

    for row in offline.itertuples(index=False):
        idx = row.bar_index
        expected = _build_feature_vector(closes, volumes, idx)
        actual = [getattr(row, column) for column in CORE_FEATURE_COLS]
        assert actual == pytest.approx(expected, rel=1e-12, abs=1e-12)

    # The first valid row has ten zero returns.  Both paths floor variance at
    # 1e-12, so volatility is 1e-6 rather than zero or NaN.
    assert offline.iloc[0]["rolling_vol"] == pytest.approx(1e-6)


def test_split_uses_one_global_cutoff_and_purges_each_symbol() -> None:
    timestamps = pd.date_range("2026-01-02 14:30", periods=12, freq="15min", tz="UTC")
    rows = [
        {"symbol": "AAPL", "timestamp": timestamp, "row_number": idx}
        for idx, timestamp in enumerate(timestamps)
    ]
    # A shorter history catches regressions to the former per-symbol row split:
    # it must still use AAPL's global timestamp cutoff, not its own midpoint.
    rows.extend(
        {"symbol": "MSFT", "timestamp": timestamp, "row_number": idx}
        for idx, timestamp in enumerate(timestamps[:8])
    )
    featured = pd.DataFrame(rows)

    train, test = _split_per_symbol(featured, train_fraction=0.5)

    cutoff = timestamps[5]
    assert train.attrs["chronological_cutoff"] == cutoff
    assert test.attrs["chronological_cutoff"] == cutoff
    assert train.attrs["label_horizon_purged_per_symbol"] == LABEL_HORIZON
    assert train.attrs["purged_rows"] == 2 * LABEL_HORIZON
    assert (pd.to_datetime(test["timestamp"], utc=True) > cutoff).all()

    for symbol in ("AAPL", "MSFT"):
        symbol_train = train.loc[train["symbol"] == symbol]
        symbol_test = test.loc[test["symbol"] == symbol]
        assert symbol_train["row_number"].tolist() == [0, 1]
        assert symbol_test["row_number"].min() == 6

"""Tests for strategy.py — focused on the opening range calculation fix."""
import pandas as pd
import pytest

from strategy import (
    MlSignal,
    STRATEGY_MODE_SMA,
    Strategy,
    StrategyConfig,
    calculate_opening_range_series,
)


def _make_15min_timestamps(date_str: str, n_bars: int) -> list[pd.Timestamp]:
    """Return n_bars consecutive 15-min bar timestamps starting at 9:30 ET on date_str."""
    start = pd.Timestamp(f"{date_str} 09:30:00", tz="America/New_York")
    return [start + pd.Timedelta(minutes=15 * i) for i in range(n_bars)]


def _constant_bars(n: int, high: float = 101.0, low: float = 99.0):
    """Return (highs, lows) lists of length n with constant values."""
    return [high] * n, [low] * n


# ---------------------------------------------------------------------------
# Core correctness: how many bars end up inside each OR window
# ---------------------------------------------------------------------------

class TestOpeningRangeBarsIncluded:
    """Verify that opening_range_minutes correctly controls which bars form the OR."""

    def test_15min_or_uses_exactly_one_bar(self):
        """A 15-min OR should include only the 9:30 bar."""
        timestamps = _make_15min_timestamps("2024-01-02", 6)
        highs = [100, 102, 103, 104, 105, 106]
        lows  = [ 99,  98,  97,  96,  95,  94]

        or_highs, or_lows = calculate_opening_range_series(
            timestamps, highs, lows, opening_range_minutes=15
        )

        # Range should be ready from the second bar onward (bar 0 is in window,
        # range_ready fires, subsequent bars carry OR forward).
        # OR high/low must equal the 9:30 bar only (high=100, low=99).
        ready = [h for h in or_highs if h is not None]
        assert all(h == 100.0 for h in ready), f"OR high should be 100, got {ready}"
        ready_lows = [l for l in or_lows if l is not None]
        assert all(l == 99.0 for l in ready_lows), f"OR low should be 99, got {ready_lows}"

    def test_30min_or_uses_two_bars(self):
        """A 30-min OR should include the 9:30 and 9:45 bars."""
        timestamps = _make_15min_timestamps("2024-01-02", 6)
        highs = [100, 102, 99, 99, 99, 99]   # highest in first two bars
        lows  = [ 98,  99, 100, 100, 100, 100]  # lowest in first two bars

        or_highs, or_lows = calculate_opening_range_series(
            timestamps, highs, lows, opening_range_minutes=30
        )

        # OR becomes ready at bar index 1 (second bar, 9:45).
        # From that index forward, OR high = max(100, 102) = 102, low = min(98, 99) = 98.
        assert or_highs[0] is None, "OR not ready before enough bars are collected"
        ready_highs = [h for h in or_highs if h is not None]
        ready_lows  = [l for l in or_lows  if l is not None]
        assert all(h == 102.0 for h in ready_highs), f"OR high should be 102, got {ready_highs}"
        assert all(l == 98.0  for l in ready_lows),  f"OR low should be 98, got {ready_lows}"

    def test_60min_or_uses_four_bars(self):
        """A 60-min OR should include the 9:30, 9:45, 10:00, and 10:15 bars.

        This was broken before the fix — with `<= dt_time(9, 45)` the 60-min OR
        captured at most 2 bars and never met bars_needed=4, so range_ready was
        always False and no OR values were emitted.
        """
        timestamps = _make_15min_timestamps("2024-01-02", 8)
        highs = [100, 101, 102, 103, 99, 99, 99, 99]  # 4-bar window high = 103
        lows  = [ 99,  98,  97,  96, 100, 100, 100, 100]  # 4-bar window low = 96

        or_highs, or_lows = calculate_opening_range_series(
            timestamps, highs, lows, opening_range_minutes=60
        )

        # Before the fix, all entries would be None because range_ready never fired.
        ready_highs = [h for h in or_highs if h is not None]
        ready_lows  = [l for l in or_lows  if l is not None]
        assert len(ready_highs) > 0, "60-min OR should produce values (was broken before fix)"
        assert all(h == 103.0 for h in ready_highs), f"OR high should be 103, got {ready_highs}"
        assert all(l == 96.0  for l in ready_lows),  f"OR low should be 96, got {ready_lows}"

    def test_range_ready_starts_at_correct_index(self):
        """OR values should be None before enough bars are collected."""
        timestamps = _make_15min_timestamps("2024-01-02", 6)
        highs, lows = _constant_bars(6)

        or_highs, _ = calculate_opening_range_series(
            timestamps, highs, lows, opening_range_minutes=30
        )

        # bars_needed = 2; first bar (index 0) alone is not enough → None at index 0
        assert or_highs[0] is None
        # At index 1 (9:45 bar), both bars are in window → range ready
        assert or_highs[1] is not None

    def test_60min_range_ready_at_index_3(self):
        """60-min OR should be None for the first 3 bars, ready at bar index 3."""
        timestamps = _make_15min_timestamps("2024-01-02", 6)
        highs, lows = _constant_bars(6)

        or_highs, _ = calculate_opening_range_series(
            timestamps, highs, lows, opening_range_minutes=60
        )

        assert or_highs[0] is None
        assert or_highs[1] is None
        assert or_highs[2] is None
        assert or_highs[3] is not None


# ---------------------------------------------------------------------------
# Day boundary: OR resets each trading day
# ---------------------------------------------------------------------------

class TestOpeningRangeDayBoundary:
    def test_or_resets_across_days(self):
        """Each trading day should have its own independent OR."""
        day1 = _make_15min_timestamps("2024-01-02", 4)
        day2 = _make_15min_timestamps("2024-01-03", 4)
        timestamps = day1 + day2

        # Day 1: highs 110/111, lows 109/108  →  OR high=111, low=108
        # Day 2: highs 200/201, lows 199/198  →  OR high=201, low=198
        highs = [110, 111, 109, 109,  200, 201, 199, 199]
        lows  = [109, 108, 108, 108,  199, 198, 198, 198]

        or_highs, or_lows = calculate_opening_range_series(
            timestamps, highs, lows, opening_range_minutes=30
        )

        # Day 1: bar 0 → None, bars 1–3 → 111/108
        assert or_highs[0] is None
        assert or_highs[1] == 111.0
        assert or_lows[1]  == 108.0

        # Day 2: bar 4 → None, bars 5–7 → 201/198
        assert or_highs[4] is None
        assert or_highs[5] == 201.0
        assert or_lows[5]  == 198.0


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestOpeningRangeEdgeCases:
    def test_empty_input(self):
        result_h, result_l = calculate_opening_range_series([], [], [])
        assert result_h == []
        assert result_l == []

    def test_single_bar_30min_or_not_ready(self):
        """With only 1 bar and a 30-min OR, bars_needed=2 is never met."""
        ts = _make_15min_timestamps("2024-01-02", 1)
        highs, lows = _constant_bars(1)
        or_highs, or_lows = calculate_opening_range_series(ts, highs, lows, opening_range_minutes=30)
        assert or_highs == [None]
        assert or_lows  == [None]

    def test_bars_after_or_window_do_not_extend_or(self):
        """Bars outside the OR window must not change the OR high/low."""
        timestamps = _make_15min_timestamps("2024-01-02", 8)
        # Spike in bars 4–7 — should not affect OR
        highs = [100, 101, 100, 100,  999, 999, 999, 999]
        lows  = [ 99,  98,  99,  99,    1,   1,   1,   1]

        or_highs, or_lows = calculate_opening_range_series(
            timestamps, highs, lows, opening_range_minutes=30
        )

        ready_highs = [h for h in or_highs if h is not None]
        ready_lows  = [l for l in or_lows  if l is not None]
        assert all(h == 101.0 for h in ready_highs)
        assert all(l == 98.0  for l in ready_lows)


def _ml_signal() -> MlSignal:
    return MlSignal(
        probability_up=0.5,
        confidence=0.0,
        training_rows=0,
        model_age_seconds=0.0,
        feature_names=(),
        buy_threshold=0.55,
        sell_threshold=0.45,
        validation_rows=0,
        model_name="test",
    )


class TestSmaStopPct:
    def test_sma_mode_sells_on_stop_even_above_sma(self):
        strategy = Strategy(
            StrategyConfig(
                strategy_mode=STRATEGY_MODE_SMA,
                sma_stop_pct=0.02,
            )
        )

        action = strategy.decide_action(
            price=97.5,
            sma=95.0,
            ml_signal=_ml_signal(),
            holding=True,
            position_entry_price=100.0,
        )

        assert action == "SELL"

    def test_sma_mode_does_not_use_stop_when_disabled(self):
        strategy = Strategy(
            StrategyConfig(
                strategy_mode=STRATEGY_MODE_SMA,
                sma_stop_pct=0.0,
            )
        )

        action = strategy.decide_action(
            price=97.5,
            sma=95.0,
            ml_signal=_ml_signal(),
            holding=True,
            position_entry_price=100.0,
        )

        assert action == "HOLD"

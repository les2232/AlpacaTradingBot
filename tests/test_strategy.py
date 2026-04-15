"""Tests for strategy.py — focused on the opening range calculation fix."""
import pandas as pd
import pytest

from strategy import (
    ADX_PERIOD,
    MlSignal,
    STRATEGY_MODE_MEAN_REVERSION,
    STRATEGY_MODE_SMA,
    Strategy,
    StrategyConfig,
    calculate_adx_series,
    calculate_opening_range_series,
    calculate_vwap_series,
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


# ---------------------------------------------------------------------------
# calculate_vwap_series
# ---------------------------------------------------------------------------

def _make_vwap_inputs(
    timestamps: list[pd.Timestamp],
    highs: list[float],
    lows: list[float],
    closes: list[float],
    volumes: list[float],
):
    """Thin wrapper so call sites stay readable."""
    return calculate_vwap_series(timestamps, highs, lows, closes, volumes)


class TestVwapBasicCorrectness:
    def test_two_bar_weighted_average(self):
        """VWAP at bar 1 must be the volume-weighted average of both bars."""
        ts = _make_15min_timestamps("2024-01-02", 3)
        # Bar 0: tp=(102+98+100)/3=100, vol=200  → seed only (returns None)
        # Bar 1: tp=(104+96+100)/3=100, vol=100  → VWAP = (100*200 + 100*100)/(200+100) = 100
        # Bar 2: tp=(110+90+100)/3=100, vol=300  → VWAP = (20000+10000+30000)/600 = 100
        result = _make_vwap_inputs(
            ts,
            highs=[102.0, 104.0, 110.0],
            lows=[98.0, 96.0, 90.0],
            closes=[100.0, 100.0, 100.0],
            volumes=[200.0, 100.0, 300.0],
        )
        assert result[0] is None
        assert result[1] == pytest.approx(100.0)
        assert result[2] == pytest.approx(100.0)

    def test_unequal_volumes_shift_vwap(self):
        """Heavy volume on a high bar should pull VWAP up."""
        ts = _make_15min_timestamps("2024-01-02", 3)
        # Bar 0 (seed): tp=100, vol=10
        # Bar 1: tp=110, vol=90  → VWAP = (100*10 + 110*90) / 100 = (1000+9900)/100 = 109
        result = _make_vwap_inputs(
            ts,
            highs=[101.0, 111.0, 111.0],
            lows=[99.0, 109.0, 109.0],
            closes=[100.0, 110.0, 110.0],
            volumes=[10.0, 90.0, 0.0],
        )
        assert result[0] is None
        assert result[1] == pytest.approx(109.0)

    def test_vwap_is_cumulative_not_rolling(self):
        """VWAP must accumulate from session start, not just the last two bars."""
        ts = _make_15min_timestamps("2024-01-02", 4)
        # All bars: tp=100, equal volume → VWAP stays 100 throughout.
        result = _make_vwap_inputs(
            ts,
            highs=[101.0] * 4,
            lows=[99.0] * 4,
            closes=[100.0] * 4,
            volumes=[100.0] * 4,
        )
        assert result[0] is None
        for val in result[1:]:
            assert val == pytest.approx(100.0)


class TestVwapSessionBoundary:
    def test_first_bar_of_every_session_is_none(self):
        """Index 0 of each trading day must return None regardless of volume."""
        day1 = _make_15min_timestamps("2024-01-02", 3)
        day2 = _make_15min_timestamps("2024-01-03", 3)
        ts = day1 + day2
        vals = _make_vwap_inputs(
            ts,
            highs=[101.0] * 6,
            lows=[99.0] * 6,
            closes=[100.0] * 6,
            volumes=[500.0] * 6,
        )
        # First bar of each day must be None.
        assert vals[0] is None, "first bar of day 1 should be None"
        assert vals[3] is None, "first bar of day 2 should be None"
        # Non-first bars must have a value.
        assert vals[1] is not None
        assert vals[4] is not None

    def test_resets_between_days(self):
        """Day 2 VWAP must ignore day 1 volume entirely."""
        day1 = _make_15min_timestamps("2024-01-02", 2)
        day2 = _make_15min_timestamps("2024-01-03", 2)
        ts = day1 + day2

        # Day 1: tp=200 with huge vol → would dominate if carried over.
        # Day 2: tp=100, vol=50 each → VWAP at bar[3] = 100 exactly.
        result = _make_vwap_inputs(
            ts,
            highs=[201.0, 201.0, 101.0, 101.0],
            lows=[199.0, 199.0,  99.0,  99.0],
            closes=[200.0, 200.0, 100.0, 100.0],
            volumes=[10000.0, 10000.0, 50.0, 50.0],
        )
        # bar[3] is the second bar of day 2 → should be 100, not contaminated by day 1.
        assert result[3] == pytest.approx(100.0), (
            f"Day 2 VWAP should be 100, got {result[3]} — accumulator may not have reset"
        )


class TestVwapZeroVolume:
    def test_zero_volume_bars_return_none(self):
        """If all bars so far have zero volume, VWAP must be None (no /0)."""
        ts = _make_15min_timestamps("2024-01-02", 3)
        result = _make_vwap_inputs(
            ts,
            highs=[101.0, 101.0, 101.0],
            lows=[99.0, 99.0, 99.0],
            closes=[100.0, 100.0, 100.0],
            volumes=[0.0, 0.0, 0.0],
        )
        assert all(v is None for v in result), f"all-zero-volume session should produce all None: {result}"

    def test_zero_volume_first_bar_then_volume(self):
        """A zero-volume seed bar followed by a volume bar must still yield VWAP."""
        ts = _make_15min_timestamps("2024-01-02", 3)
        # Bar 0: vol=0 (seed, None)
        # Bar 1: vol=0 still (no accumulated volume → None)
        # Bar 2: vol=100 → VWAP computable
        result = _make_vwap_inputs(
            ts,
            highs=[101.0, 101.0, 101.0],
            lows=[99.0, 99.0, 99.0],
            closes=[100.0, 100.0, 100.0],
            volumes=[0.0, 0.0, 100.0],
        )
        assert result[0] is None
        assert result[1] is None   # cum_v still 0
        assert result[2] == pytest.approx(100.0)

    def test_negative_volume_treated_as_zero(self):
        """Negative volume (bad data) must not cause negative cum_v or exceptions."""
        ts = _make_15min_timestamps("2024-01-02", 3)
        result = _make_vwap_inputs(
            ts,
            highs=[101.0, 101.0, 101.0],
            lows=[99.0, 99.0, 99.0],
            closes=[100.0, 100.0, 100.0],
            volumes=[100.0, -999.0, 100.0],
        )
        # Bar 0: seed → None
        # Bar 1: cum_v = 100 (negative ignored) → VWAP = 100
        # Bar 2: cum_v = 200 → VWAP = 100
        assert result[0] is None
        assert result[1] == pytest.approx(100.0)
        assert result[2] == pytest.approx(100.0)


class TestVwapEdgeCases:
    def test_empty_inputs_return_empty(self):
        assert calculate_vwap_series([], [], [], [], []) == []

    def test_mismatched_lengths_return_empty(self):
        ts = _make_15min_timestamps("2024-01-02", 3)
        result = calculate_vwap_series(ts, [101.0] * 3, [99.0] * 3, [100.0] * 3, [100.0] * 2)
        assert result == []

    def test_single_bar_returns_none(self):
        ts = _make_15min_timestamps("2024-01-02", 1)
        result = _make_vwap_inputs(ts, [101.0], [99.0], [100.0], [500.0])
        assert result == [None]

    def test_output_length_matches_input(self):
        n = 26  # full trading day at 15-min bars
        ts = _make_15min_timestamps("2024-01-02", n)
        result = _make_vwap_inputs(
            ts,
            highs=[101.0] * n,
            lows=[99.0] * n,
            closes=[100.0] * n,
            volumes=[100.0] * n,
        )
        assert len(result) == n

    def test_naive_utc_timestamps_accepted(self):
        """Naive (tz-unaware) timestamps should be interpreted as UTC without raising."""
        ts = [pd.Timestamp("2024-01-02 14:30:00")] + [
            pd.Timestamp(f"2024-01-02 {h}:00:00") for h in range(15, 21)
        ]
        highs = [101.0] * 7
        lows = [99.0] * 7
        closes = [100.0] * 7
        volumes = [100.0] * 7
        result = calculate_vwap_series(ts, highs, lows, closes, volumes)
        assert len(result) == 7
        # Should not raise; values may be None or float depending on session grouping.


# ---------------------------------------------------------------------------
# MeanReversionVwapZScore — VWAP Z-score path in STRATEGY_MODE_MEAN_REVERSION
# ---------------------------------------------------------------------------

def _mr_strategy(
    vwap_z_entry_threshold: float = 1.5,
    vwap_z_stop_atr_multiple: float = 2.0,
    min_atr_percentile: float = 0.0,   # 0 = disabled so non-filter tests are unaffected
    max_adx_threshold: float = 0.0,    # 0 = disabled so non-filter tests are unaffected
) -> Strategy:
    return Strategy(
        StrategyConfig(
            strategy_mode=STRATEGY_MODE_MEAN_REVERSION,
            vwap_z_entry_threshold=vwap_z_entry_threshold,
            vwap_z_stop_atr_multiple=vwap_z_stop_atr_multiple,
            min_atr_percentile=min_atr_percentile,
            max_adx_threshold=max_adx_threshold,
        )
    )


def _decide(
    strategy: Strategy,
    price: float,
    vwap: float | None,
    atr_pct: float | None,
    holding: bool = False,
    position_entry_price: float | None = None,
    atr_percentile: float | None = None,
    adx: float | None = None,
) -> str:
    """Thin wrapper — only the VWAP / ATR inputs vary across these tests."""
    return strategy.decide_action(
        price=price,
        sma=price,           # SMA unused when VWAP path fires
        ml_signal=_ml_signal(),
        holding=holding,
        atr_pct=atr_pct,
        atr_percentile=atr_percentile,
        position_entry_price=position_entry_price,
        vwap=vwap,
        adx=adx,
    )


class TestMrVwapZscoreEntry:
    def test_buys_when_z_below_negative_threshold(self):
        """z = (price - vwap) / atr = (90 - 100) / 5 = -2.0 → below -1.5 → BUY."""
        assert _decide(_mr_strategy(), price=90.0, vwap=100.0, atr_pct=0.05) == "BUY"
        # atr = atr_pct * price = 0.05 * 90 = 4.5; z = (90-100)/4.5 ≈ -2.22 → BUY

    def test_holds_when_z_inside_threshold(self):
        """z near zero (price ≈ vwap) — no entry."""
        # z = (99 - 100) / (0.05 * 99) ≈ -0.20 → inside threshold
        assert _decide(_mr_strategy(), price=99.0, vwap=100.0, atr_pct=0.05) == "HOLD"

    def test_holds_when_z_barely_misses_threshold(self):
        """z just above -1.5 must not trigger entry."""
        # Target z = -1.49 → price = vwap + z * atr = 100 + (-1.49 * 5) = 92.55
        # Use atr_pct=0.05, so atr = 0.05 * 92.55 ≈ 4.63; z ≈ (92.55-100)/4.63 ≈ -1.61
        # Let's engineer exactly: vwap=100, want z=-1.4 → need (price-100)/(atr_pct*price)=-1.4
        # price/(1 + 1.4*atr_pct) = 100/(1+0.07) ≈ 93.46
        # Verify: atr=0.05*93.46=4.673; z=(93.46-100)/4.673=-1.40 → inside → HOLD
        assert _decide(_mr_strategy(), price=93.46, vwap=100.0, atr_pct=0.05) == "HOLD"

    def test_no_entry_when_already_holding(self):
        """Must not enter a second position while holding."""
        result = _decide(
            _mr_strategy(), price=85.0, vwap=100.0, atr_pct=0.05,
            holding=True, position_entry_price=88.0,
        )
        assert result != "BUY"

    def test_no_entry_when_vwap_is_none(self):
        """None VWAP falls through to SMA path; plain SMA HOLD when price ≈ sma."""
        # atr_pct provided but vwap=None → SMA fallback with price=sma → no entry
        result = _decide(_mr_strategy(), price=100.0, vwap=None, atr_pct=0.05)
        assert result == "HOLD"

    def test_no_entry_when_atr_pct_is_none(self):
        """None ATR falls through to SMA fallback (atr_pct=None, vwap=None path)."""
        result = _decide(_mr_strategy(), price=90.0, vwap=100.0, atr_pct=None)
        assert result == "HOLD"


class TestMrVwapZscoreExit:
    def test_sells_at_vwap_profit_target(self):
        """Price returning to VWAP should trigger SELL."""
        result = _decide(
            _mr_strategy(), price=100.0, vwap=100.0, atr_pct=0.05,
            holding=True, position_entry_price=90.0,
        )
        assert result == "SELL"

    def test_sells_above_vwap(self):
        """Price above VWAP also triggers exit (>= vwap)."""
        result = _decide(
            _mr_strategy(), price=101.0, vwap=100.0, atr_pct=0.05,
            holding=True, position_entry_price=90.0,
        )
        assert result == "SELL"

    def test_holds_while_below_vwap_and_above_stop(self):
        """Position still below VWAP but not stopped out → HOLD."""
        # entry=95, price=93, vwap=100; atr=0.05*93=4.65; stop=95-2*4.65=85.7
        result = _decide(
            _mr_strategy(), price=93.0, vwap=100.0, atr_pct=0.05,
            holding=True, position_entry_price=95.0,
        )
        assert result == "HOLD"

    def test_stop_loss_fires_before_vwap_check(self):
        """Stop takes priority — price collapsed well below entry.

        atr = atr_pct * current_price (not entry price).
        At price=85: atr = 0.05 * 85 = 4.25; stop = 95 - 2*4.25 = 86.5
        85 < 86.5 → SELL, even though price is still below vwap=100.
        """
        result = _decide(
            _mr_strategy(), price=85.0, vwap=100.0, atr_pct=0.05,
            holding=True, position_entry_price=95.0,
        )
        assert result == "SELL"

    def test_stop_respects_atr_multiple_config(self):
        """A wider stop_multiple allows more room before stopping out."""
        strategy_tight = _mr_strategy(vwap_z_stop_atr_multiple=1.0)
        strategy_wide = _mr_strategy(vwap_z_stop_atr_multiple=3.0)
        # entry=95, atr=0.05*91=4.55
        # tight stop: 95 - 1*4.55 = 90.45 → price=91 > 90.45 → HOLD
        # wide stop:  95 - 3*4.55 = 81.35 → price=91 > 81.35 → HOLD
        # Move price to 91: tight fires at 90.45 → HOLD (still above)
        # Force tight to fire: price=90
        # atr = 0.05 * 90 = 4.5; tight stop = 95 - 4.5 = 90.5 → 90 < 90.5 → SELL
        # wide  stop = 95 - 3*4.5 = 81.5 → 90 > 81.5 → HOLD
        result_tight = _decide(
            strategy_tight, price=90.0, vwap=100.0, atr_pct=0.05,
            holding=True, position_entry_price=95.0,
        )
        result_wide = _decide(
            strategy_wide, price=90.0, vwap=100.0, atr_pct=0.05,
            holding=True, position_entry_price=95.0,
        )
        assert result_tight == "SELL"
        assert result_wide == "HOLD"

    def test_degenerate_zero_atr_returns_hold(self):
        """ATR of zero (flat tape) must not cause ZeroDivisionError."""
        # atr = atr_pct * price = 0.0 * 90 = 0 → guard fires → HOLD
        result = _decide(
            _mr_strategy(), price=90.0, vwap=100.0, atr_pct=0.0,
            holding=False,
        )
        assert result == "HOLD"


class TestMrVwapAtrPercentileFilter:
    """ATR percentile minimum filter on VWAP MR entries."""

    def test_entry_blocked_below_min_percentile(self):
        """Known low ATR percentile below threshold must block entry."""
        # z = (90-100)/(0.05*90) = -2.22 → strong signal, but atr_pct=8 < min=30 → HOLD
        s = _mr_strategy(min_atr_percentile=30.0)
        assert _decide(s, price=90.0, vwap=100.0, atr_pct=0.05, atr_percentile=8.0) == "HOLD"

    def test_entry_allowed_at_exact_min_percentile(self):
        """atr_percentile == min_atr_percentile is >= threshold → entry fires."""
        s = _mr_strategy(min_atr_percentile=30.0)
        assert _decide(s, price=90.0, vwap=100.0, atr_pct=0.05, atr_percentile=30.0) == "BUY"

    def test_entry_allowed_above_min_percentile(self):
        """atr_percentile well above threshold → normal entry."""
        s = _mr_strategy(min_atr_percentile=30.0)
        assert _decide(s, price=90.0, vwap=100.0, atr_pct=0.05, atr_percentile=65.0) == "BUY"

    def test_fail_open_when_percentile_is_none(self):
        """None atr_percentile must not block entry — filter always fails open."""
        s = _mr_strategy(min_atr_percentile=50.0)
        assert _decide(s, price=90.0, vwap=100.0, atr_pct=0.05, atr_percentile=None) == "BUY"

    def test_disabled_when_threshold_is_zero(self):
        """min_atr_percentile=0 disables the filter; any percentile value is accepted."""
        s = _mr_strategy(min_atr_percentile=0.0)
        assert _decide(s, price=90.0, vwap=100.0, atr_pct=0.05, atr_percentile=2.0) == "BUY"

    def test_filter_does_not_block_vwap_exit(self):
        """Low ATR percentile must not prevent exit when price returns to VWAP."""
        s = _mr_strategy(min_atr_percentile=50.0)
        result = _decide(
            s, price=100.0, vwap=100.0, atr_pct=0.05,
            holding=True, position_entry_price=90.0, atr_percentile=5.0,
        )
        assert result == "SELL"

    def test_filter_does_not_block_stop_loss(self):
        """Stop loss must fire regardless of ATR percentile.

        entry=95, atr=0.05*85=4.25, stop=95-(1.0*4.25)=90.75; price=85 < 90.75 → SELL
        """
        s = _mr_strategy(min_atr_percentile=50.0, vwap_z_stop_atr_multiple=1.0)
        result = _decide(
            s, price=85.0, vwap=100.0, atr_pct=0.05,
            holding=True, position_entry_price=95.0, atr_percentile=3.0,
        )
        assert result == "SELL"


class TestMrSmaFallback:
    """Verify the original SMA-based path still fires unchanged when vwap=None."""

    def test_sma_path_buys_below_threshold(self):
        """With vwap=None and price below SMA threshold, original BUY fires."""
        strategy = Strategy(
            StrategyConfig(
                strategy_mode=STRATEGY_MODE_MEAN_REVERSION,
                entry_threshold_pct=0.02,   # entry when price < sma * (1 - 0.02)
            )
        )
        # price=97, sma=100, threshold=100*(1-0.02)=98 → price < threshold → BUY
        result = strategy.decide_action(
            price=97.0, sma=100.0, ml_signal=_ml_signal(), holding=False, vwap=None,
        )
        assert result == "BUY"

    def test_sma_path_sells_at_sma(self):
        """With vwap=None and price >= sma, original SELL fires while holding."""
        strategy = Strategy(
            StrategyConfig(
                strategy_mode=STRATEGY_MODE_MEAN_REVERSION,
                entry_threshold_pct=0.02,
            )
        )
        result = strategy.decide_action(
            price=101.0, sma=100.0, ml_signal=_ml_signal(),
            holding=True, position_entry_price=95.0, vwap=None,
        )
        assert result == "SELL"


# ---------------------------------------------------------------------------
# ADX series computation
# ---------------------------------------------------------------------------

def _trending_bars(n: int, start: float = 100.0, step: float = 0.5):
    """Return (highs, lows, closes) for a steadily trending up sequence."""
    closes = [start + step * i for i in range(n)]
    highs  = [c + 0.5 for c in closes]
    lows   = [c - 0.5 for c in closes]
    return highs, lows, closes


class TestAdxSeries:
    def test_returns_empty_on_empty_input(self):
        assert calculate_adx_series([], [], []) == []

    def test_returns_all_none_for_insufficient_bars(self):
        """Fewer than 2*period - 1 bars: all None."""
        n = 2 * ADX_PERIOD - 2   # one short of the minimum
        highs, lows, closes = _trending_bars(n)
        result = calculate_adx_series(highs, lows, closes)
        assert len(result) == n
        assert all(v is None for v in result)

    def test_first_valid_adx_at_correct_index(self):
        """First non-None ADX appears at index 2*period - 2."""
        n = 2 * ADX_PERIOD + 5
        highs, lows, closes = _trending_bars(n)
        result = calculate_adx_series(highs, lows, closes)
        first_valid = next((i for i, v in enumerate(result) if v is not None), None)
        assert first_valid == 2 * ADX_PERIOD - 2

    def test_output_length_matches_input(self):
        n = 60
        highs, lows, closes = _trending_bars(n)
        result = calculate_adx_series(highs, lows, closes)
        assert len(result) == n

    def test_strong_trend_produces_high_adx(self):
        """A sustained unidirectional move should produce ADX > 25."""
        n = 60
        highs, lows, closes = _trending_bars(n, step=1.0)
        result = calculate_adx_series(highs, lows, closes)
        valid = [v for v in result if v is not None]
        assert valid, "Expected at least one valid ADX value"
        # After enough bars the ADX should be firmly in trending territory
        assert max(valid) > 25.0

    def test_flat_market_produces_low_adx(self):
        """Sideways price action should produce ADX < 25."""
        import math, random
        random.seed(42)
        n = 60
        closes = [100.0 + math.sin(i * 0.3) for i in range(n)]
        highs  = [c + 0.2 for c in closes]
        lows   = [c - 0.2 for c in closes]
        result = calculate_adx_series(highs, lows, closes)
        valid = [v for v in result if v is not None]
        assert valid
        # Oscillating prices should settle to low ADX
        assert min(valid) < 25.0

    def test_mismatched_lengths_return_empty(self):
        assert calculate_adx_series([100.0, 101.0], [99.0], [100.0]) == []


# ---------------------------------------------------------------------------
# ADX entry filter in VWAP MR
# ---------------------------------------------------------------------------

class TestMrVwapAdxFilter:
    """ADX maximum threshold filter on VWAP MR entries."""

    def test_entry_blocked_when_adx_above_threshold(self):
        """High ADX (trending market) must block entry even on strong z-score."""
        s = _mr_strategy(max_adx_threshold=25.0)
        # z = (90-100)/(0.05*90) = -2.22 → strong signal, but ADX=35 >= 25 → HOLD
        assert _decide(s, price=90.0, vwap=100.0, atr_pct=0.05, adx=35.0) == "HOLD"

    def test_entry_allowed_when_adx_below_threshold(self):
        """Low ADX (ranging market) permits entry when z is sufficiently negative."""
        s = _mr_strategy(max_adx_threshold=25.0)
        assert _decide(s, price=90.0, vwap=100.0, atr_pct=0.05, adx=18.0) == "BUY"

    def test_entry_blocked_at_exact_threshold(self):
        """adx == threshold is NOT below threshold → block entry."""
        s = _mr_strategy(max_adx_threshold=25.0)
        assert _decide(s, price=90.0, vwap=100.0, atr_pct=0.05, adx=25.0) == "HOLD"

    def test_entry_allowed_just_below_threshold(self):
        """adx one tick below threshold → allow entry."""
        s = _mr_strategy(max_adx_threshold=25.0)
        assert _decide(s, price=90.0, vwap=100.0, atr_pct=0.05, adx=24.99) == "BUY"

    def test_fail_open_when_adx_is_none(self):
        """None ADX (insufficient warmup) must not block entry — filter fails open."""
        s = _mr_strategy(max_adx_threshold=25.0)
        assert _decide(s, price=90.0, vwap=100.0, atr_pct=0.05, adx=None) == "BUY"

    def test_disabled_when_threshold_is_zero(self):
        """max_adx_threshold=0 disables the filter; any ADX value is accepted."""
        s = _mr_strategy(max_adx_threshold=0.0)
        assert _decide(s, price=90.0, vwap=100.0, atr_pct=0.05, adx=80.0) == "BUY"

    def test_filter_does_not_block_vwap_exit(self):
        """High ADX must not prevent exit when price returns to VWAP."""
        s = _mr_strategy(max_adx_threshold=25.0)
        result = _decide(
            s, price=100.0, vwap=100.0, atr_pct=0.05,
            holding=True, position_entry_price=90.0, adx=50.0,
        )
        assert result == "SELL"

    def test_filter_does_not_block_stop_loss(self):
        """ADX filter on entry must not interfere with stop loss on exit.

        entry=95, atr=0.05*85=4.25, stop=95-(1.0*4.25)=90.75; price=85 < 90.75 → SELL
        """
        s = _mr_strategy(max_adx_threshold=25.0, vwap_z_stop_atr_multiple=1.0)
        result = _decide(
            s, price=85.0, vwap=100.0, atr_pct=0.05,
            holding=True, position_entry_price=95.0, adx=60.0,
        )
        assert result == "SELL"

    def test_both_filters_must_pass_for_entry(self):
        """Entry requires BOTH ATR percentile AND ADX conditions to hold."""
        s = _mr_strategy(min_atr_percentile=30.0, max_adx_threshold=25.0)
        # ADX ok, ATR percentile too low → HOLD
        assert _decide(
            s, price=90.0, vwap=100.0, atr_pct=0.05, atr_percentile=10.0, adx=15.0
        ) == "HOLD"
        # ATR percentile ok, ADX too high → HOLD
        assert _decide(
            s, price=90.0, vwap=100.0, atr_pct=0.05, atr_percentile=60.0, adx=35.0
        ) == "HOLD"
        # Both ok → BUY
        assert _decide(
            s, price=90.0, vwap=100.0, atr_pct=0.05, atr_percentile=60.0, adx=15.0
        ) == "BUY"

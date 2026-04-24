"""Tests for strategy.py — focused on the opening range calculation fix."""
import pandas as pd
import pytest

from strategy import (
    ADX_PERIOD,
    BOLLINGER_EXIT_MIDDLE_BAND,
    MlSignal,
    STRATEGY_MODE_BOLLINGER_SQUEEZE,
    STRATEGY_MODE_HYBRID_BB_MR,
    STRATEGY_MODE_MEAN_REVERSION,
    STRATEGY_MODE_MOMENTUM_BREAKOUT,
    STRATEGY_MODE_SMA,
    STRATEGY_MODE_TREND_PULLBACK,
    STRATEGY_MODE_VOLATILITY_EXPANSION,
    STRATEGY_MODE_WICK_FADE,
    Strategy,
    StrategyConfig,
    calculate_adx_series,
    calculate_bollinger_squeeze_features,
    calculate_opening_range_series,
    calculate_vwap_series,
    strategy_requires_adx,
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


class TestMeanReversionRiskControls:
    def test_mean_reversion_sells_on_stop_even_before_reversion_completes(self):
        strategy = Strategy(
            StrategyConfig(
                strategy_mode=STRATEGY_MODE_MEAN_REVERSION,
                mean_reversion_stop_pct=0.01,
            )
        )

        action = strategy.decide_action(
            price=98.9,
            sma=101.0,
            ml_signal=_ml_signal(),
            holding=True,
            position_entry_price=100.0,
        )

        assert action == "SELL"


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


# ---------------------------------------------------------------------------
# Wick-fade strategy tests
# ---------------------------------------------------------------------------

_DUMMY_ML = MlSignal(
    probability_up=0.5, confidence=0.0, training_rows=0,
    model_age_seconds=0.0, feature_names=(),
    buy_threshold=0.55, sell_threshold=0.45,
    validation_rows=0, model_name="dummy",
)


def _wf_strategy(**kwargs) -> Strategy:
    defaults = dict(
        strategy_mode=STRATEGY_MODE_WICK_FADE,
        wick_fade_min_lower_wick_ratio=0.4,
        wick_fade_min_close_position=0.5,
        wick_fade_min_range_pct=0.003,
        wick_fade_stop_atr_multiple=1.5,
        wick_fade_target_atr_multiple=1.0,
        wick_fade_max_hold_bars=4,
    )
    defaults.update(kwargs)
    return Strategy(StrategyConfig(**defaults))


def _wf_decide(
    s: Strategy,
    *,
    close: float,
    high: float,
    low: float,
    open_: float,
    holding: bool = False,
    wick_fade_stop: float = 0.0,
    wick_fade_target: float = 0.0,
    wick_fade_bars_held: int = 0,
    time_window_open: bool = True,
    atr_percentile: float | None = None,
    bullish_regime: bool | None = None,
) -> str:
    return s.decide_action(
        close, sma=close, ml_signal=_DUMMY_ML, holding=holding,
        time_window_open=time_window_open,
        atr_percentile=atr_percentile,
        bullish_regime=bullish_regime,
        bar_high=high, bar_low=low, bar_open=open_,
        wick_fade_stop=wick_fade_stop,
        wick_fade_target=wick_fade_target,
        wick_fade_bars_held=wick_fade_bars_held,
    )


def _tp_strategy(**kwargs) -> Strategy:
    defaults = dict(
        strategy_mode=STRATEGY_MODE_TREND_PULLBACK,
        trend_pullback_min_adx=20.0,
        trend_pullback_min_slope=0.0,
        trend_pullback_entry_threshold=0.01,
        trend_pullback_min_atr_percentile=20.0,
        trend_pullback_max_atr_percentile=0.0,
        trend_pullback_exit_style="fixed_bars",
        trend_pullback_hold_bars=4,
        trend_pullback_stop_pct=0.0,
    )
    defaults.update(kwargs)
    return Strategy(StrategyConfig(**defaults))


def _tp_decide(
    strategy: Strategy,
    *,
    price: float,
    sma: float = 100.0,
    holding: bool = False,
    atr_percentile: float | None = 40.0,
    time_window_open: bool = True,
    bullish_regime: bool | None = None,
    position_entry_price: float | None = None,
    trend_sma: float | None = 95.0,
    trend_sma_slope: float | None = 0.2,
    adx: float | None = 25.0,
    bar_high: float | None = None,
    trend_pullback_bars_held: int = 0,
) -> str:
    return strategy.decide_action(
        price=price,
        sma=sma,
        ml_signal=_ml_signal(),
        holding=holding,
        atr_percentile=atr_percentile,
        time_window_open=time_window_open,
        bullish_regime=bullish_regime,
        position_entry_price=position_entry_price,
        trend_sma=trend_sma,
        trend_sma_slope=trend_sma_slope,
        adx=adx,
        bar_high=bar_high,
        trend_pullback_bars_held=trend_pullback_bars_held,
    )


def _mb_strategy(**kwargs) -> Strategy:
    defaults = dict(
        strategy_mode=STRATEGY_MODE_MOMENTUM_BREAKOUT,
        momentum_breakout_lookback_bars=20,
        momentum_breakout_entry_buffer_pct=0.001,
        momentum_breakout_min_adx=20.0,
        momentum_breakout_min_slope=0.0,
        momentum_breakout_min_atr_percentile=20.0,
        momentum_breakout_exit_style="fixed_bars",
        momentum_breakout_hold_bars=3,
        momentum_breakout_stop_pct=0.0,
        momentum_breakout_take_profit_pct=0.0,
    )
    defaults.update(kwargs)
    return Strategy(StrategyConfig(**defaults))


def _mb_decide(
    strategy: Strategy,
    *,
    price: float,
    holding: bool = False,
    atr_percentile: float | None = 40.0,
    time_window_open: bool = True,
    bullish_regime: bool | None = None,
    position_entry_price: float | None = None,
    trend_sma: float | None = 100.0,
    trend_sma_slope: float | None = 0.2,
    adx: float | None = 25.0,
    bar_high: float | None = None,
    bar_low: float | None = None,
    recent_breakout_high: float | None = 99.0,
    momentum_breakout_bars_held: int = 0,
) -> str:
    return strategy.decide_action(
        price=price,
        sma=100.0,
        ml_signal=_ml_signal(),
        holding=holding,
        atr_percentile=atr_percentile,
        time_window_open=time_window_open,
        bullish_regime=bullish_regime,
        position_entry_price=position_entry_price,
        trend_sma=trend_sma,
        trend_sma_slope=trend_sma_slope,
        adx=adx,
        bar_high=bar_high,
        bar_low=bar_low,
        recent_breakout_high=recent_breakout_high,
        momentum_breakout_bars_held=momentum_breakout_bars_held,
    )


def _ve_strategy(**kwargs) -> Strategy:
    defaults = dict(
        strategy_mode=STRATEGY_MODE_VOLATILITY_EXPANSION,
        volatility_expansion_lookback_bars=20,
        volatility_expansion_entry_buffer_pct=0.001,
        volatility_expansion_max_atr_percentile=35.0,
        volatility_expansion_trend_filter=False,
        volatility_expansion_min_slope=0.0,
        volatility_expansion_use_volume_confirm=True,
        volatility_expansion_exit_style="fixed_bars",
        volatility_expansion_hold_bars=4,
        volatility_expansion_stop_pct=0.0,
        volatility_expansion_take_profit_pct=0.0,
    )
    defaults.update(kwargs)
    return Strategy(StrategyConfig(**defaults))


def _ve_decide(
    strategy: Strategy,
    *,
    price: float,
    holding: bool = False,
    atr_percentile: float | None = 20.0,
    time_window_open: bool = True,
    bullish_regime: bool | None = None,
    position_entry_price: float | None = None,
    trend_sma: float | None = 100.0,
    trend_sma_slope: float | None = 0.2,
    bar_high: float | None = None,
    bar_low: float | None = None,
    recent_breakout_high: float | None = 99.0,
    bb_prev_squeeze: bool | None = True,
    bb_breakout_up: bool | None = True,
    bb_volume_confirm: bool | None = True,
    volatility_expansion_bars_held: int = 0,
) -> str:
    return strategy.decide_action(
        price=price,
        sma=100.0,
        ml_signal=_ml_signal(),
        holding=holding,
        atr_percentile=atr_percentile,
        time_window_open=time_window_open,
        bullish_regime=bullish_regime,
        position_entry_price=position_entry_price,
        trend_sma=trend_sma,
        trend_sma_slope=trend_sma_slope,
        bar_high=bar_high,
        bar_low=bar_low,
        recent_breakout_high=recent_breakout_high,
        bb_prev_squeeze=bb_prev_squeeze,
        bb_breakout_up=bb_breakout_up,
        bb_volume_confirm=bb_volume_confirm,
        volatility_expansion_bars_held=volatility_expansion_bars_held,
    )


def test_strategy_requires_adx_for_trend_pullback_min_adx() -> None:
    assert strategy_requires_adx(
        StrategyConfig(strategy_mode=STRATEGY_MODE_TREND_PULLBACK, trend_pullback_min_adx=20.0, max_adx_threshold=0.0)
    )
    assert not strategy_requires_adx(
        StrategyConfig(strategy_mode=STRATEGY_MODE_TREND_PULLBACK, trend_pullback_min_adx=0.0, max_adx_threshold=0.0)
    )


def test_strategy_requires_adx_for_momentum_breakout_min_adx() -> None:
    assert strategy_requires_adx(
        StrategyConfig(strategy_mode=STRATEGY_MODE_MOMENTUM_BREAKOUT, momentum_breakout_min_adx=20.0, max_adx_threshold=0.0)
    )
    assert not strategy_requires_adx(
        StrategyConfig(strategy_mode=STRATEGY_MODE_MOMENTUM_BREAKOUT, momentum_breakout_min_adx=0.0, max_adx_threshold=0.0)
    )


class TestWickFadeEntry:
    def test_strong_lower_wick_triggers_buy(self):
        """Classic rejection candle: large lower wick, close near high."""
        s = _wf_strategy()
        # Bar: low=95, open=99, close=100, high=101 → range=6
        # lower_wick = min(99,100) - 95 = 4  →  ratio = 4/6 = 0.667 >= 0.4 ✓
        # close_position = (100-95)/6 = 0.833 >= 0.5 ✓
        assert _wf_decide(s, close=100.0, high=101.0, low=95.0, open_=99.0) == "BUY"

    def test_no_lower_wick_no_buy(self):
        """Bearish engulfing — close at low, no lower wick."""
        s = _wf_strategy()
        # Bar: low=95, open=101, close=95, high=101 → range=6
        # lower_wick = min(101,95) - 95 = 0  →  ratio = 0 < 0.4
        assert _wf_decide(s, close=95.0, high=101.0, low=95.0, open_=101.0) == "HOLD"

    def test_close_in_lower_half_no_buy(self):
        """Wick present but close is in lower half of range — no rejection confirmed."""
        s = _wf_strategy()
        # Bar: low=95, open=99, close=97, high=101 → range=6
        # lower_wick = min(99,97) - 95 = 2  →  ratio = 2/6 = 0.333 < 0.4
        assert _wf_decide(s, close=97.0, high=101.0, low=95.0, open_=99.0) == "HOLD"

    def test_wick_ratio_threshold_boundary(self):
        """Exactly at min_lower_wick_ratio should trigger; just below should not."""
        s = _wf_strategy(wick_fade_min_lower_wick_ratio=0.4)
        # range=10; wick=4 → ratio=0.4 exactly (meets threshold)
        # close_position = (100-90)/10 = 1.0
        assert _wf_decide(s, close=100.0, high=100.0, low=90.0, open_=94.0) == "BUY"
        # wick=3 → ratio=0.3 < 0.4
        assert _wf_decide(s, close=100.0, high=100.0, low=90.0, open_=93.0) == "HOLD"

    def test_dead_tape_filter(self):
        """Bar range below min_range_pct filter → HOLD even if wick looks good."""
        s = _wf_strategy(wick_fade_min_range_pct=0.005)
        # price=100, range=0.4 → range_pct=0.004 < 0.005
        assert _wf_decide(s, close=100.0, high=100.2, low=99.8, open_=100.1) == "HOLD"

    def test_zero_range_bar_no_buy(self):
        """Doji / zero-range bar must not crash and must return HOLD."""
        s = _wf_strategy()
        assert _wf_decide(s, close=100.0, high=100.0, low=100.0, open_=100.0) == "HOLD"

    def test_missing_bar_data_no_buy(self):
        """None bar_high/low/open falls through to HOLD safely."""
        s = _wf_strategy()
        result = s.decide_action(
            100.0, sma=100.0, ml_signal=_DUMMY_ML, holding=False,
            bar_high=None, bar_low=None, bar_open=None,
        )
        assert result == "HOLD"

    def test_time_window_closed_blocks_entry(self):
        s = _wf_strategy()
        assert _wf_decide(
            s, close=100.0, high=101.0, low=95.0, open_=99.0,
            time_window_open=False,
        ) == "HOLD"

    def test_regime_filter_blocks_non_bullish_entry(self):
        s = _wf_strategy(regime_filter_enabled=True)
        assert _wf_decide(
            s, close=100.0, high=101.0, low=95.0, open_=99.0,
            bullish_regime=False,
        ) == "HOLD"

    def test_regime_filter_allows_bullish_entry(self):
        s = _wf_strategy(regime_filter_enabled=True)
        assert _wf_decide(
            s, close=100.0, high=101.0, low=95.0, open_=99.0,
            bullish_regime=True,
        ) == "BUY"

    def test_already_holding_no_second_entry(self):
        """No new BUY while already holding (stop not yet hit)."""
        s = _wf_strategy(wick_fade_stop_atr_multiple=1.5)
        assert _wf_decide(
            s, close=100.0, high=101.0, low=95.0, open_=99.0,
            holding=True, wick_fade_stop=97.0, wick_fade_target=103.0,
        ) == "HOLD"


class TestWickFadeExit:
    def test_stop_hit_triggers_sell(self):
        s = _wf_strategy()
        # Price drops to stop
        assert _wf_decide(
            s, close=97.0, high=98.0, low=96.5, open_=98.0,
            holding=True, wick_fade_stop=97.5, wick_fade_target=103.0,
        ) == "SELL"

    def test_target_hit_triggers_sell(self):
        s = _wf_strategy()
        assert _wf_decide(
            s, close=103.0, high=103.5, low=101.0, open_=101.5,
            holding=True, wick_fade_stop=97.0, wick_fade_target=103.0,
        ) == "SELL"

    def test_price_between_stop_and_target_holds(self):
        s = _wf_strategy(wick_fade_max_hold_bars=10)
        assert _wf_decide(
            s, close=100.5, high=101.0, low=100.0, open_=100.2,
            holding=True, wick_fade_stop=97.0, wick_fade_target=103.0,
            wick_fade_bars_held=3,
        ) == "HOLD"

    def test_max_hold_bars_timeout(self):
        """After max_hold_bars bars, force exit regardless of price."""
        s = _wf_strategy(wick_fade_max_hold_bars=4)
        # 4 bars held — should sell
        assert _wf_decide(
            s, close=101.0, high=101.5, low=100.5, open_=101.0,
            holding=True, wick_fade_stop=97.0, wick_fade_target=103.0,
            wick_fade_bars_held=4,
        ) == "SELL"

    def test_below_max_hold_bars_no_timeout(self):
        s = _wf_strategy(wick_fade_max_hold_bars=4)
        assert _wf_decide(
            s, close=101.0, high=101.5, low=100.5, open_=101.0,
            holding=True, wick_fade_stop=97.0, wick_fade_target=103.0,
            wick_fade_bars_held=3,
        ) == "HOLD"

    def test_max_hold_bars_zero_disables_timeout(self):
        """max_hold_bars=0 means never timeout."""
        s = _wf_strategy(wick_fade_max_hold_bars=0)
        assert _wf_decide(
            s, close=101.0, high=101.5, low=100.5, open_=101.0,
            holding=True, wick_fade_stop=97.0, wick_fade_target=103.0,
            wick_fade_bars_held=999,
        ) == "HOLD"

    def test_stop_takes_priority_over_target(self):
        """If somehow price is both at/below stop AND at/above target, stop wins."""
        s = _wf_strategy()
        # Degenerate config where stop > target (shouldn't happen in practice)
        assert _wf_decide(
            s, close=100.0, high=100.0, low=100.0, open_=100.0,
            holding=True, wick_fade_stop=100.0, wick_fade_target=99.0,
        ) == "SELL"


class TestTrendPullbackStrategy:
    def test_buys_during_pullback_in_confirmed_trend(self):
        strategy = _tp_strategy(trend_pullback_entry_threshold=0.01)

        action = _tp_decide(
            strategy,
            price=98.9,
            sma=100.0,
            trend_sma=96.0,
            trend_sma_slope=0.3,
            adx=28.0,
        )

        assert action == "BUY"

    def test_rejects_when_trend_reference_missing(self):
        strategy = _tp_strategy()
        assert _tp_decide(strategy, price=98.9, trend_sma=None) == "HOLD"

    def test_rejects_when_slope_is_negative(self):
        strategy = _tp_strategy(trend_pullback_min_slope=0.0)
        assert _tp_decide(strategy, price=98.9, trend_sma_slope=-0.01) == "HOLD"

    def test_rejects_when_price_loses_trend_reference(self):
        strategy = _tp_strategy()
        assert _tp_decide(strategy, price=94.0, trend_sma=95.0, sma=100.0) == "HOLD"

    def test_rejects_when_adx_below_threshold(self):
        strategy = _tp_strategy(trend_pullback_min_adx=25.0)
        assert _tp_decide(strategy, price=98.9, adx=20.0) == "HOLD"

    def test_rejects_when_atr_percentile_outside_bounds(self):
        strategy = _tp_strategy(trend_pullback_min_atr_percentile=30.0, trend_pullback_max_atr_percentile=60.0)
        assert _tp_decide(strategy, price=98.9, atr_percentile=10.0) == "HOLD"
        assert _tp_decide(strategy, price=98.9, atr_percentile=75.0) == "HOLD"

    def test_fixed_bar_exit_fires_at_limit(self):
        strategy = _tp_strategy(trend_pullback_hold_bars=4)
        assert _tp_decide(
            strategy,
            price=101.0,
            holding=True,
            position_entry_price=99.0,
            trend_pullback_bars_held=4,
        ) == "SELL"

    def test_holds_before_fixed_bar_exit_limit(self):
        strategy = _tp_strategy(trend_pullback_hold_bars=4)
        assert _tp_decide(
            strategy,
            price=101.0,
            holding=True,
            position_entry_price=99.0,
            trend_pullback_bars_held=3,
        ) == "HOLD"

    def test_take_profit_exit_fires_when_bar_high_hits_target(self):
        strategy = _tp_strategy(
            trend_pullback_exit_style="take_profit",
            trend_pullback_take_profit_pct=0.0025,
        )
        assert _tp_decide(
            strategy,
            price=100.1,
            holding=True,
            position_entry_price=100.0,
            bar_high=100.3,
            trend_pullback_bars_held=1,
        ) == "SELL"

    def test_take_profit_holds_when_target_not_hit(self):
        strategy = _tp_strategy(
            trend_pullback_exit_style="take_profit",
            trend_pullback_take_profit_pct=0.0025,
        )
        assert _tp_decide(
            strategy,
            price=100.1,
            holding=True,
            position_entry_price=100.0,
            bar_high=100.2,
            trend_pullback_bars_held=2,
        ) == "HOLD"

    def test_hybrid_take_profit_beats_time_exit(self):
        strategy = _tp_strategy(
            trend_pullback_exit_style="hybrid_tp_or_time",
            trend_pullback_hold_bars=3,
            trend_pullback_take_profit_pct=0.0025,
        )
        assert _tp_decide(
            strategy,
            price=100.05,
            holding=True,
            position_entry_price=100.0,
            bar_high=100.4,
            trend_pullback_bars_held=3,
        ) == "SELL"

    def test_hybrid_time_exit_fires_when_target_missed(self):
        strategy = _tp_strategy(
            trend_pullback_exit_style="hybrid_tp_or_time",
            trend_pullback_hold_bars=3,
            trend_pullback_take_profit_pct=0.0025,
        )
        assert _tp_decide(
            strategy,
            price=100.0,
            holding=True,
            position_entry_price=100.0,
            bar_high=100.2,
            trend_pullback_bars_held=3,
        ) == "SELL"

    def test_optional_stop_pct_exits_early(self):
        strategy = _tp_strategy(trend_pullback_stop_pct=0.02)
        assert _tp_decide(
            strategy,
            price=97.5,
            holding=True,
            position_entry_price=100.0,
            trend_pullback_bars_held=1,
        ) == "SELL"


class TestMomentumBreakoutStrategy:
    def test_buys_on_breakout_in_confirmed_trend(self):
        strategy = _mb_strategy(
            momentum_breakout_lookback_bars=20,
            momentum_breakout_entry_buffer_pct=0.001,
        )

        action = _mb_decide(
            strategy,
            price=100.2,
            trend_sma=98.0,
            trend_sma_slope=0.25,
            adx=28.0,
            recent_breakout_high=100.0,
        )

        assert action == "BUY"

    def test_rejects_without_trend_confirmation(self):
        strategy = _mb_strategy()
        assert _mb_decide(strategy, price=100.2, trend_sma=101.0) == "HOLD"
        assert _mb_decide(strategy, price=100.2, trend_sma_slope=-0.01) == "HOLD"

    def test_rejects_without_breakout_trigger(self):
        strategy = _mb_strategy(momentum_breakout_entry_buffer_pct=0.001)
        assert _mb_decide(strategy, price=100.05, recent_breakout_high=100.0) == "HOLD"

    def test_rejects_when_filters_fail(self):
        strategy = _mb_strategy(momentum_breakout_min_adx=25.0, momentum_breakout_min_atr_percentile=30.0)
        assert _mb_decide(strategy, price=100.2, adx=20.0, recent_breakout_high=100.0) == "HOLD"
        assert _mb_decide(strategy, price=100.2, atr_percentile=10.0, recent_breakout_high=100.0) == "HOLD"

    def test_fixed_bar_exit_fires_at_limit(self):
        strategy = _mb_strategy(momentum_breakout_hold_bars=3)
        assert _mb_decide(
            strategy,
            price=101.0,
            holding=True,
            position_entry_price=100.0,
            momentum_breakout_bars_held=3,
        ) == "SELL"

    def test_stop_and_take_profit_are_optional_overlays(self):
        stop_strategy = _mb_strategy(momentum_breakout_stop_pct=0.01)
        assert _mb_decide(
            stop_strategy,
            price=100.0,
            holding=True,
            position_entry_price=100.0,
            bar_low=98.9,
            momentum_breakout_bars_held=1,
        ) == "SELL"

        tp_strategy = _mb_strategy(momentum_breakout_take_profit_pct=0.01)
        assert _mb_decide(
            tp_strategy,
            price=100.4,
            holding=True,
            position_entry_price=100.0,
            bar_high=101.1,
            momentum_breakout_bars_held=1,
        ) == "SELL"


class TestVolatilityExpansionStrategy:
    def test_buys_after_squeeze_breakout(self):
        strategy = _ve_strategy()
        assert _ve_decide(strategy, price=99.2, recent_breakout_high=99.0, atr_percentile=20.0) == "BUY"

    def test_rejects_without_compression_setup(self):
        strategy = _ve_strategy()
        assert _ve_decide(strategy, price=99.2, bb_prev_squeeze=False) == "HOLD"

    def test_rejects_without_breakout_confirmation(self):
        strategy = _ve_strategy()
        assert _ve_decide(strategy, price=99.2, bb_breakout_up=False) == "HOLD"

    def test_trend_filter_blocks_weak_context(self):
        strategy = _ve_strategy(volatility_expansion_trend_filter=True, volatility_expansion_min_slope=0.05)
        assert _ve_decide(strategy, price=99.2, trend_sma=100.0, trend_sma_slope=0.0) == "HOLD"

    def test_volume_confirm_can_be_disabled(self):
        strategy = _ve_strategy(volatility_expansion_use_volume_confirm=False)
        assert _ve_decide(strategy, price=99.2, bb_volume_confirm=False) == "BUY"

    def test_max_atr_percentile_blocks_non_compression_regime(self):
        strategy = _ve_strategy(volatility_expansion_max_atr_percentile=25.0)
        assert _ve_decide(strategy, price=99.2, atr_percentile=40.0) == "HOLD"

    def test_fixed_bar_exit_fires_at_limit(self):
        strategy = _ve_strategy(volatility_expansion_hold_bars=4)
        assert _ve_decide(
            strategy,
            price=101.0,
            holding=True,
            position_entry_price=100.0,
            volatility_expansion_bars_held=4,
        ) == "SELL"

    def test_stop_and_take_profit_are_optional_overlays(self):
        stop_strategy = _ve_strategy(volatility_expansion_stop_pct=0.02)
        assert _ve_decide(
            stop_strategy,
            price=100.0,
            holding=True,
            position_entry_price=100.0,
            bar_low=97.5,
            volatility_expansion_bars_held=1,
        ) == "SELL"

        tp_strategy = _ve_strategy(volatility_expansion_take_profit_pct=0.01)
        assert _ve_decide(
            tp_strategy,
            price=100.4,
            holding=True,
            position_entry_price=100.0,
            bar_high=101.1,
            volatility_expansion_bars_held=1,
        ) == "SELL"


class TestBollingerSqueezeFeatures:
    def test_returns_parallel_series(self):
        closes = [100.0 + (i * 0.1) for i in range(160)]
        volumes = [1000.0] * 160

        features = calculate_bollinger_squeeze_features(
            closes,
            volumes,
            period=20,
            width_lookback=100,
            slope_lookback=3,
        )

        assert len(features["middle"]) == len(closes)
        assert len(features["squeeze"]) == len(closes)
        assert len(features["bias"]) == len(closes)

    def test_marks_bullish_breakout_after_squeeze(self):
        closes = [100.0] * 123 + [104.0]
        volumes = [1000.0] * 123 + [2500.0]

        features = calculate_bollinger_squeeze_features(
            closes,
            volumes,
            period=20,
            width_lookback=100,
            squeeze_quantile=0.2,
            slope_lookback=3,
            use_volume_confirm=True,
            volume_mult=1.2,
        )

        assert features["squeeze"][-2] is True
        assert features["bias"][-1] == "bullish"
        assert features["breakout_up"][-1] is True
        assert features["volume_confirm"][-1] is True


def _bb_strategy(**kwargs) -> Strategy:
    defaults = dict(
        strategy_mode=STRATEGY_MODE_BOLLINGER_SQUEEZE,
        bb_exit_mode=BOLLINGER_EXIT_MIDDLE_BAND,
    )
    defaults.update(kwargs)
    return Strategy(StrategyConfig(**defaults))


class TestBollingerSqueezeStrategy:
    def test_buys_on_confirmed_bullish_breakout(self):
        strategy = _bb_strategy(bb_use_volume_confirm=True, bb_volume_mult=1.2)

        action = strategy.decide_action(
            price=104.0,
            sma=101.0,
            ml_signal=_ml_signal(),
            holding=False,
            time_window_open=True,
            bb_middle=101.0,
            bb_upper=103.0,
            bb_lower=99.0,
            bb_prev_squeeze=True,
            bb_mid_slope=0.8,
            bb_bias="bullish",
            bb_breakout_up=True,
            bb_breakout_down=False,
            bb_volume_confirm=True,
        )

        assert action == "BUY"

    def test_rejects_breakout_without_prior_squeeze(self):
        strategy = _bb_strategy()

        action = strategy.decide_action(
            price=104.0,
            sma=101.0,
            ml_signal=_ml_signal(),
            holding=False,
            bb_middle=101.0,
            bb_upper=103.0,
            bb_lower=99.0,
            bb_prev_squeeze=False,
            bb_mid_slope=0.8,
            bb_bias="bullish",
            bb_breakout_up=True,
            bb_breakout_down=False,
            bb_volume_confirm=True,
        )

        assert action == "HOLD"

    def test_rejects_when_volume_confirm_enabled_and_missing(self):
        strategy = _bb_strategy(bb_use_volume_confirm=True)

        action = strategy.decide_action(
            price=104.0,
            sma=101.0,
            ml_signal=_ml_signal(),
            holding=False,
            bb_middle=101.0,
            bb_upper=103.0,
            bb_lower=99.0,
            bb_prev_squeeze=True,
            bb_mid_slope=0.8,
            bb_bias="bullish",
            bb_breakout_up=True,
            bb_breakout_down=False,
            bb_volume_confirm=False,
        )

        assert action == "HOLD"

    def test_sells_on_middle_band_exit(self):
        strategy = _bb_strategy()

        action = strategy.decide_action(
            price=99.0,
            sma=100.0,
            ml_signal=_ml_signal(),
            holding=True,
            bb_middle=100.0,
        )

        assert action == "SELL"

    def test_requires_breakout_buffer_when_configured(self):
        strategy = _bb_strategy(bb_breakout_buffer_pct=0.01)

        action = strategy.decide_action(
            price=103.5,
            sma=101.0,
            ml_signal=_ml_signal(),
            holding=False,
            bb_middle=101.0,
            bb_upper=103.0,
            bb_lower=99.0,
            bb_prev_squeeze=True,
            bb_mid_slope=0.8,
            bb_bias="bullish",
            bb_breakout_up=True,
            bb_breakout_down=False,
            bb_volume_confirm=True,
            trend_sma=100.0,
        )

        assert action == "HOLD"

    def test_requires_trend_filter_when_enabled(self):
        strategy = _bb_strategy(bb_trend_filter=True)

        action = strategy.decide_action(
            price=104.0,
            sma=101.0,
            ml_signal=_ml_signal(),
            holding=False,
            bb_middle=101.0,
            bb_upper=103.0,
            bb_lower=99.0,
            bb_prev_squeeze=True,
            bb_mid_slope=0.8,
            bb_bias="bullish",
            bb_breakout_up=True,
            bb_breakout_down=False,
            bb_volume_confirm=True,
            trend_sma=105.0,
        )

        assert action == "HOLD"


def _hybrid_bb_mr_strategy(**kwargs) -> Strategy:
    defaults = dict(
        strategy_mode=STRATEGY_MODE_HYBRID_BB_MR,
        bb_exit_mode=BOLLINGER_EXIT_MIDDLE_BAND,
        entry_threshold_pct=0.02,
    )
    defaults.update(kwargs)
    return Strategy(StrategyConfig(**defaults))


class TestHybridBbMrStrategy:
    def test_selects_bollinger_branch_when_previous_bar_was_squeeze(self):
        strategy = _hybrid_bb_mr_strategy()

        details = strategy.decide_action_details(
            price=104.0,
            sma=101.0,
            ml_signal=_ml_signal(),
            holding=False,
            bb_middle=101.0,
            bb_upper=103.0,
            bb_lower=99.0,
            bb_prev_squeeze=True,
            bb_mid_slope=0.5,
            bb_bias="bullish",
            bb_breakout_up=True,
            bb_breakout_down=False,
            bb_volume_confirm=True,
        )

        assert details.hybrid_branch == "bollinger_breakout"
        assert details.action == "BUY"
        assert details.reason == "bollinger_breakout_long"

    def test_falls_back_to_mean_reversion_when_no_squeeze(self):
        strategy = _hybrid_bb_mr_strategy(entry_threshold_pct=0.02)

        details = strategy.decide_action_details(
            price=97.0,
            sma=100.0,
            ml_signal=_ml_signal(),
            holding=False,
            atr_pct=None,
            bb_middle=99.0,
            bb_upper=101.0,
            bb_lower=97.0,
            bb_prev_squeeze=False,
            bb_mid_slope=0.2,
            bb_bias="bullish",
            bb_breakout_up=False,
            bb_breakout_down=False,
            bb_volume_confirm=True,
        )

        assert details.hybrid_branch == "mean_reversion"
        assert details.mr_signal == "BUY"
        assert details.action == "BUY"
        assert details.reason == "mean_reversion_sma_entry"

    def test_uses_previous_squeeze_only_without_lookahead(self):
        strategy = _hybrid_bb_mr_strategy(entry_threshold_pct=0.02)

        details = strategy.decide_action_details(
            price=97.0,
            sma=100.0,
            ml_signal=_ml_signal(),
            holding=False,
            bb_middle=99.0,
            bb_upper=101.0,
            bb_lower=97.0,
            bb_prev_squeeze=False,
            bb_mid_slope=0.5,
            bb_bias="bullish",
            bb_breakout_up=True,
            bb_breakout_down=False,
            bb_volume_confirm=True,
        )

        assert details.hybrid_branch == "mean_reversion"
        assert details.action == "BUY"
        assert details.reason == "mean_reversion_sma_entry"

    def test_bollinger_branch_blocks_entry_without_breakout(self):
        strategy = _hybrid_bb_mr_strategy()

        details = strategy.decide_action_details(
            price=102.0,
            sma=100.0,
            ml_signal=_ml_signal(),
            holding=False,
            bb_middle=100.0,
            bb_upper=103.0,
            bb_lower=97.0,
            bb_prev_squeeze=True,
            bb_mid_slope=0.5,
            bb_bias="bullish",
            bb_breakout_up=False,
            bb_breakout_down=False,
            bb_volume_confirm=True,
        )

        assert details.hybrid_branch == "bollinger_breakout"
        assert details.action == "HOLD"
        assert details.reason == "bollinger_no_signal"

    def test_persists_bollinger_branch_while_position_is_open(self):
        strategy = _hybrid_bb_mr_strategy()

        details = strategy.decide_action_details(
            price=99.0,
            sma=101.0,
            ml_signal=_ml_signal(),
            holding=True,
            bb_middle=100.0,
            bb_upper=104.0,
            bb_lower=96.0,
            bb_prev_squeeze=False,
            bb_mid_slope=-0.5,
            bb_bias="bearish",
            bb_breakout_up=False,
            bb_breakout_down=False,
            bb_volume_confirm=True,
            hybrid_entry_branch="bollinger_breakout",
        )

        assert details.hybrid_branch_active == "bollinger_breakout"
        assert details.hybrid_regime_branch == "mean_reversion"
        assert details.hybrid_entry_branch == "bollinger_breakout"
        assert details.action == "SELL"
        assert details.reason == "bollinger_middle_band_exit"

    def test_persists_mean_reversion_branch_while_position_is_open(self):
        strategy = _hybrid_bb_mr_strategy(entry_threshold_pct=0.02)

        details = strategy.decide_action_details(
            price=100.0,
            sma=99.0,
            ml_signal=_ml_signal(),
            holding=True,
            bb_middle=98.0,
            bb_upper=99.0,
            bb_lower=95.0,
            bb_prev_squeeze=True,
            bb_mid_slope=0.4,
            bb_bias="bullish",
            bb_breakout_up=True,
            bb_breakout_down=False,
            bb_volume_confirm=True,
            hybrid_entry_branch="mean_reversion",
        )

        assert details.hybrid_branch_active == "mean_reversion"
        assert details.hybrid_regime_branch == "bollinger_breakout"
        assert details.hybrid_entry_branch == "mean_reversion"
        assert details.action == "SELL"
        assert details.reason == "mean_reversion_sma_exit"

    def test_clears_entry_branch_after_exit_and_reselects_from_regime(self):
        strategy = _hybrid_bb_mr_strategy(entry_threshold_pct=0.02)

        exit_details = strategy.decide_action_details(
            price=100.0,
            sma=99.0,
            ml_signal=_ml_signal(),
            holding=True,
            bb_middle=98.0,
            bb_upper=99.0,
            bb_lower=95.0,
            bb_prev_squeeze=True,
            bb_mid_slope=0.4,
            bb_bias="bullish",
            bb_breakout_up=True,
            bb_breakout_down=False,
            bb_volume_confirm=True,
            hybrid_entry_branch="mean_reversion",
        )
        next_entry_details = strategy.decide_action_details(
            price=104.0,
            sma=101.0,
            ml_signal=_ml_signal(),
            holding=False,
            bb_middle=101.0,
            bb_upper=103.0,
            bb_lower=99.0,
            bb_prev_squeeze=True,
            bb_mid_slope=0.5,
            bb_bias="bullish",
            bb_breakout_up=True,
            bb_breakout_down=False,
            bb_volume_confirm=True,
            hybrid_entry_branch=None,
        )

        assert exit_details.action == "SELL"
        assert next_entry_details.hybrid_branch_active == "bollinger_breakout"
        assert next_entry_details.hybrid_entry_branch == "bollinger_breakout"
        assert next_entry_details.action == "BUY"

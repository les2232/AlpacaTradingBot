from run_feed_comparison_validation import classify_material_difference, FeedStrategySummary


def _summary(*, strategy: str, feed: str, raw: float, expectancy: float, label: str, signals: int = 10) -> FeedStrategySummary:
    return FeedStrategySummary(
        strategy_mode=strategy,
        feed_label=feed,
        signal_count=signals,
        trade_count=signals,
        realized_pnl=expectancy * signals,
        expectancy=expectancy,
        win_rate=50.0,
        profit_factor=1.0,
        best_raw_horizon_bars=10,
        best_raw_net_expectancy_pct=raw,
        classification_label=label,
        classification_reason="x",
        spacing_label="benign",
    )


def test_classify_material_difference_detects_label_change() -> None:
    iex = _summary(strategy="trend_pullback", feed="iex", raw=0.05, expectancy=-0.1, label="weak")
    sip = _summary(strategy="trend_pullback", feed="sip", raw=0.12, expectancy=0.2, label="promising but narrow")

    label, _ = classify_material_difference(iex, sip)

    assert label == "material difference"


def test_classify_material_difference_detects_small_shift() -> None:
    iex = _summary(strategy="volatility_expansion", feed="iex", raw=-0.40, expectancy=-2.0, label="reject", signals=8)
    sip = _summary(strategy="volatility_expansion", feed="sip", raw=-0.36, expectancy=-1.8, label="reject", signals=9)

    label, _ = classify_material_difference(iex, sip)

    assert label == "mostly unchanged"

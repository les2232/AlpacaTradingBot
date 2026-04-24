from run_feed_comparison_validation import FeedStrategySummary
from run_trend_pullback_clean_dataset_comparison import classify_clean_dataset_materiality


def _summary(expectancy: float, raw: float, label: str) -> FeedStrategySummary:
    return FeedStrategySummary(
        strategy_mode="trend_pullback",
        feed_label="x",
        signal_count=10,
        trade_count=10,
        realized_pnl=expectancy * 10,
        expectancy=expectancy,
        win_rate=50.0,
        profit_factor=1.0,
        best_raw_horizon_bars=20,
        best_raw_net_expectancy_pct=raw,
        classification_label=label,
        classification_reason="x",
        spacing_label="benign",
    )


def test_classify_clean_dataset_materiality_detects_same_conclusion_cleaner() -> None:
    iex = _summary(1.1, 0.21, "promising but narrow")
    sip_shared = _summary(1.2, 0.22, "promising but narrow")
    sip_clean = _summary(1.12, 0.23, "promising but narrow")

    label, _ = classify_clean_dataset_materiality(iex, sip_shared, sip_clean)

    assert label == "mostly cleaner, same conclusion"

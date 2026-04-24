import pandas as pd

from investigate_missing_bars import (
    classify_issue,
    compare_dataset_coverage,
    per_symbol_bar_presence,
    summarize_missing_symbol_patterns,
    summarize_timestamp_coverage,
)


def test_per_symbol_bar_presence_marks_present_and_missing() -> None:
    df = pd.DataFrame(
        [
            {
                "symbol": "AMD",
                "timestamp": pd.Timestamp("2026-02-23T18:30:00Z"),
                "open": 1.0,
                "high": 1.1,
                "low": 0.9,
                "close": 1.05,
                "volume": 10,
                "ts_et": pd.Timestamp("2026-02-23 13:30", tz="America/New_York"),
            }
        ]
    )

    result = per_symbol_bar_presence(df, ["AMD", "JPM"], [pd.Timestamp("2026-02-23 13:30", tz="America/New_York")])

    assert list(result["present"]) == [True, False]
    assert result.iloc[0]["open"] == 1.0


def test_summarize_timestamp_coverage_counts_presence() -> None:
    df = pd.DataFrame(
        [
            {"timestamp_et": "2026-02-23 13:30", "symbol": "AMD", "present": True},
            {"timestamp_et": "2026-02-23 13:30", "symbol": "JPM", "present": False},
        ]
    )

    summary = summarize_timestamp_coverage(df)

    assert summary.iloc[0]["symbols_present"] == 1
    assert summary.iloc[0]["symbols_missing"] == 1


def test_summarize_missing_symbol_patterns_groups_repeated_missing_symbols() -> None:
    df = pd.DataFrame(
        [
            {"timestamp_et": "2026-02-23 13:30", "symbol": "AMD", "present": False},
            {"timestamp_et": "2026-02-27 12:00", "symbol": "AMD", "present": False},
            {"timestamp_et": "2026-02-23 13:30", "symbol": "JPM", "present": False},
        ]
    )

    summary = summarize_missing_symbol_patterns(df)

    assert summary.iloc[0]["symbol"] == "AMD"
    assert summary.iloc[0]["missing_count"] == 2


def test_compare_dataset_coverage_computes_delta() -> None:
    base = pd.DataFrame(
        [
            {"timestamp_et": "2026-02-23 13:30", "symbol": "AMD", "present": False},
            {"timestamp_et": "2026-02-23 13:30", "symbol": "JPM", "present": False},
        ]
    )
    compare = pd.DataFrame(
        [
            {"timestamp_et": "2026-02-23 13:30", "symbol": "AMD", "present": True},
            {"timestamp_et": "2026-02-23 13:30", "symbol": "JPM", "present": True},
        ]
    )

    summary = compare_dataset_coverage(base_presence=base, compare_presence=compare, compare_label="sip")

    assert summary.iloc[0]["delta_present"] == 2


def test_classify_issue_alignment_artifact_when_raw_has_partial_coverage() -> None:
    raw = pd.DataFrame(
        [
            {"timestamp_et": "2026-02-23 13:30", "symbol": "AMD", "present": True},
            {"timestamp_et": "2026-02-23 13:30", "symbol": "JPM", "present": False},
        ]
    )

    label, _ = classify_issue(raw_presence=raw, compare_presence=pd.DataFrame())

    assert label == "alignment_artifact"


def test_classify_issue_feed_quality_when_compare_has_more_coverage_and_no_raw() -> None:
    compare = pd.DataFrame(
        [
            {"timestamp_et": "2026-02-23 13:30", "symbol": "AMD", "present": True},
            {"timestamp_et": "2026-02-23 13:30", "symbol": "JPM", "present": True},
        ]
    )

    label, _ = classify_issue(raw_presence=pd.DataFrame(), compare_presence=compare)

    assert label == "feed_quality_issue"

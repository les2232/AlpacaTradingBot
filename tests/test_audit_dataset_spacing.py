import pandas as pd

from audit_dataset_spacing import (
    classify_day_spacing,
    infer_session_template,
)


def test_classify_day_spacing_regular_session() -> None:
    day = pd.Timestamp("2026-01-05", tz="America/New_York")
    actual = list(pd.date_range(day + pd.Timedelta(hours=9, minutes=30), day + pd.Timedelta(hours=10, minutes=15), freq="15min", tz="America/New_York"))

    classification, missing = classify_day_spacing(
        actual_times=actual,
        typical_start="09:30",
        typical_end="10:15",
        expected_minutes=15,
    )

    assert classification == "regular"
    assert missing == []


def test_classify_day_spacing_detects_within_session_missing() -> None:
    day = pd.Timestamp("2026-01-05", tz="America/New_York")
    actual = [
        pd.Timestamp("2026-01-05 09:30", tz="America/New_York"),
        pd.Timestamp("2026-01-05 09:45", tz="America/New_York"),
        pd.Timestamp("2026-01-05 10:15", tz="America/New_York"),
    ]

    classification, missing = classify_day_spacing(
        actual_times=actual,
        typical_start="09:30",
        typical_end="10:15",
        expected_minutes=15,
    )

    assert classification == "within_session_missing"
    assert missing == ["10:00"]


def test_classify_day_spacing_treats_shorter_session_separately() -> None:
    day = pd.Timestamp("2026-01-05", tz="America/New_York")
    actual = [
        pd.Timestamp("2026-01-05 09:30", tz="America/New_York"),
        pd.Timestamp("2026-01-05 09:45", tz="America/New_York"),
        pd.Timestamp("2026-01-05 10:00", tz="America/New_York"),
    ]

    classification, missing = classify_day_spacing(
        actual_times=actual,
        typical_start="09:30",
        typical_end="10:15",
        expected_minutes=15,
    )

    assert classification == "partial_or_shifted_session"
    assert missing == ["10:15"]


def test_infer_session_template_uses_modal_start_end() -> None:
    grid = pd.DataFrame(
        [
            {"day": pd.Timestamp("2026-01-05", tz="America/New_York"), "ts_et": pd.Timestamp("2026-01-05 09:30", tz="America/New_York")},
            {"day": pd.Timestamp("2026-01-05", tz="America/New_York"), "ts_et": pd.Timestamp("2026-01-05 09:45", tz="America/New_York")},
            {"day": pd.Timestamp("2026-01-05", tz="America/New_York"), "ts_et": pd.Timestamp("2026-01-05 10:00", tz="America/New_York")},
            {"day": pd.Timestamp("2026-01-06", tz="America/New_York"), "ts_et": pd.Timestamp("2026-01-06 09:30", tz="America/New_York")},
            {"day": pd.Timestamp("2026-01-06", tz="America/New_York"), "ts_et": pd.Timestamp("2026-01-06 09:45", tz="America/New_York")},
            {"day": pd.Timestamp("2026-01-06", tz="America/New_York"), "ts_et": pd.Timestamp("2026-01-06 10:00", tz="America/New_York")},
        ]
    )

    result = infer_session_template(grid, 15)

    assert result["typical_start"] == "09:30"
    assert result["typical_end"] == "10:00"
    assert result["typical_bars"] == 3


def test_infer_session_template_handles_empty_input() -> None:
    result = infer_session_template(pd.DataFrame(columns=["day", "ts_et"]), 15)

    assert result["typical_start"] is None
    assert result["typical_bars"] == 0

from pathlib import Path
import json
import pandas as pd

from rebuild_research_dataset_sip_clean import build_clean_manifest, filter_regular_session


def test_filter_regular_session_keeps_only_regular_bars() -> None:
    df = pd.DataFrame(
        [
            {"symbol": "AMD", "timestamp": pd.Timestamp("2026-01-14T13:15:00Z")},
            {"symbol": "AMD", "timestamp": pd.Timestamp("2026-01-14T14:30:00Z")},
            {"symbol": "AMD", "timestamp": pd.Timestamp("2026-01-14T20:45:00Z")},
        ]
    )

    filtered = filter_regular_session(df)

    assert list(filtered["timestamp"]) == [pd.Timestamp("2026-01-14T14:30:00Z"), pd.Timestamp("2026-01-14T20:45:00Z")]


def test_build_clean_manifest_marks_regular_session_variant() -> None:
    manifest = build_clean_manifest(
        {"feed": "sip", "align_mode": "shared"},
        pd.DataFrame(
            [
                {"symbol": "AMD", "timestamp": pd.Timestamp("2026-01-14T14:30:00Z")},
                {"symbol": "JPM", "timestamp": pd.Timestamp("2026-01-14T14:30:00Z")},
            ]
        ),
        Path("datasets/sample"),
    )

    assert manifest["session_filter"] == "regular"
    assert manifest["dataset_variant"] == "sip_clean_regular_session"


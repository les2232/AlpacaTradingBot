from pathlib import Path
import json
import uuid

from rebuild_research_dataset_sip import (
    build_snapshotter_command,
    extract_rebuild_parameters,
    parse_snapshotter_output,
)


def _make_workspace_tmpdir() -> Path:
    base = Path("tmp_test_artifacts")
    base.mkdir(exist_ok=True)
    path = (base / f"rebuild_sip_{uuid.uuid4().hex}").resolve()
    path.mkdir(parents=True, exist_ok=False)
    return path


def test_extract_rebuild_parameters_reads_manifest() -> None:
    tmp_path = _make_workspace_tmpdir()
    dataset_dir = tmp_path / "sample"
    dataset_dir.mkdir()
    (dataset_dir / "manifest.json").write_text(
        json.dumps(
            {
                "symbols": ["AMD", "JPM"],
                "timeframe": "15Min",
                "start_utc": "2026-01-14T00:00:00+00:00",
                "end_utc": "2026-04-14T00:00:00+00:00",
                "adjustment": "raw",
                "align_mode": "shared",
                "feed": "iex",
            }
        ),
        encoding="utf-8",
    )

    params = extract_rebuild_parameters(dataset_dir)

    assert params["symbols"] == ["AMD", "JPM"]
    assert params["timeframe"] == "15Min"
    assert params["reference_feed"] == "iex"


def test_build_snapshotter_command_preserves_reference_shape() -> None:
    command = build_snapshotter_command(
        {
            "symbols": ["AMD", "JPM"],
            "timeframe": "15Min",
            "start_utc": "2026-01-14T00:00:00+00:00",
            "end_utc": "2026-04-14T00:00:00+00:00",
            "adjustment": "raw",
            "align_mode": "shared",
        },
        feed="sip",
        output_dir="datasets",
        align_mode=None,
    )

    assert "--feed" in command
    assert "sip" in command
    assert "--align-mode" in command
    assert "shared" in command


def test_parse_snapshotter_output_extracts_paths() -> None:
    parsed = parse_snapshotter_output(
        "dataset_id=abc\nparquet=C:\\foo\\bars.parquet\nmanifest=C:\\foo\\manifest.json\n"
    )

    assert parsed["dataset_id"] == "abc"
    assert parsed["manifest"].endswith("manifest.json")

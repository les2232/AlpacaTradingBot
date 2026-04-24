from __future__ import annotations

import json
import shutil
import uuid
from pathlib import Path
from unittest.mock import patch

import dashboard
from research.experiment_log import load_experiment_log, load_experiment_log_frame, log_experiment_run


def _make_dataset(base_dir: Path, name: str = "sample_dataset") -> Path:
    dataset_dir = base_dir / name
    dataset_dir.mkdir(parents=True)
    (dataset_dir / "manifest.json").write_text(
        json.dumps(
            {
                "symbols": ["AMD", "MSFT"],
                "timeframe": "15Min",
                "feed": "iex",
                "start": "2026-01-01T00:00:00Z",
                "end": "2026-02-01T00:00:00Z",
            }
        ),
        encoding="utf-8",
    )
    return dataset_dir


def test_log_experiment_run_appends_and_marks_improved_vs_prior() -> None:
    base_dir = Path.cwd() / f"test_experiment_log_{uuid.uuid4().hex}"
    try:
        log_path = base_dir / "results" / "experiment_log.jsonl"
        dataset_dir = _make_dataset(base_dir)

        with patch("research.experiment_log.collect_git_context", return_value={"commit": "abc123", "branch": "main", "is_dirty": False}):
            first = log_experiment_run(
                run_type="strategy_validation",
                script_path="research/run_validation.py",
                strategy_name="momentum_breakout",
                dataset_path=dataset_dir,
                symbols=["AMD", "MSFT"],
                params={"hold_bars": 3},
                metrics={
                    "total_return_pct": 4.0,
                    "profit_factor": 1.1,
                    "sharpe": 0.8,
                    "win_rate": 51.0,
                    "max_drawdown_pct": 8.0,
                    "trade_count": 18,
                },
                output_path=base_dir / "results" / "momentum_breakout_validation",
                log_path=log_path,
            )
            second = log_experiment_run(
                run_type="strategy_validation",
                script_path="research/run_validation.py",
                strategy_name="momentum_breakout",
                dataset_path=dataset_dir,
                symbols=["AMD", "MSFT"],
                params={"hold_bars": 4},
                metrics={
                    "total_return_pct": 7.0,
                    "profit_factor": 1.4,
                    "sharpe": 1.2,
                    "win_rate": 58.0,
                    "max_drawdown_pct": 6.5,
                    "trade_count": 20,
                },
                output_path=base_dir / "results" / "momentum_breakout_validation_v2",
                log_path=log_path,
            )

        assert first["auto_summary"]["label"] == "insufficient evidence"
        assert second["auto_summary"]["label"] == "improved vs prior"

        entries = load_experiment_log(log_path)
        assert len(entries) == 2
        frame = load_experiment_log_frame(log_path)
        assert set(frame["summary_label"]) == {"improved vs prior", "insufficient evidence"}
        assert frame.iloc[0]["dataset_name"] == dataset_dir.name
    finally:
        shutil.rmtree(base_dir, ignore_errors=True)


def test_log_experiment_run_marks_trade_count_collapsed() -> None:
    base_dir = Path.cwd() / f"test_experiment_log_{uuid.uuid4().hex}"
    try:
        log_path = base_dir / "results" / "experiment_log.jsonl"
        dataset_dir = _make_dataset(base_dir)

        with patch("research.experiment_log.collect_git_context", return_value={"commit": "abc123", "branch": "main", "is_dirty": False}):
            log_experiment_run(
                run_type="strategy_validation",
                script_path="research/run_validation.py",
                strategy_name="trend_pullback",
                dataset_path=dataset_dir,
                metrics={"trade_count": 40, "profit_factor": 1.2, "max_drawdown_pct": 9.0},
                log_path=log_path,
            )
            collapsed = log_experiment_run(
                run_type="strategy_validation",
                script_path="research/run_validation.py",
                strategy_name="trend_pullback",
                dataset_path=dataset_dir,
                metrics={"trade_count": 12, "profit_factor": 1.25, "max_drawdown_pct": 8.5},
                log_path=log_path,
            )

        assert collapsed["auto_summary"]["label"] == "trade count collapsed"
        assert collapsed["auto_summary"]["compared_run_id"] is not None
    finally:
        shutil.rmtree(base_dir, ignore_errors=True)


def test_dashboard_experiment_tables_are_shaped_for_display() -> None:
    frame = load_experiment_log_frame()
    if frame.empty:
        frame = dashboard.load_experiment_log_frame(
            Path.cwd() / "results" / "does_not_exist.jsonl"
        )
    sample = frame
    if sample.empty:
        sample = dashboard.pd.DataFrame(
            [
                {
                    "timestamp": dashboard.pd.Timestamp("2026-04-18T18:00:00Z"),
                    "strategy_name": "trend_pullback",
                    "run_type": "oos_validation",
                    "summary_label": "improved vs prior",
                    "total_return_pct": 6.0,
                    "profit_factor": 1.3,
                    "sharpe": 1.1,
                    "max_drawdown_pct": 5.5,
                    "trade_count": 24,
                    "dataset_name": "dataset_a",
                },
                {
                    "timestamp": dashboard.pd.Timestamp("2026-04-17T18:00:00Z"),
                    "strategy_name": "trend_pullback",
                    "run_type": "oos_validation",
                    "summary_label": "worse than prior",
                    "total_return_pct": 2.0,
                    "profit_factor": 1.0,
                    "sharpe": 0.6,
                    "max_drawdown_pct": 7.0,
                    "trade_count": 28,
                    "dataset_name": "dataset_b",
                },
            ]
        )

    recent = dashboard._build_recent_experiment_table(sample, limit=2)
    top = dashboard._build_top_experiment_table(sample, "total_return_pct", limit=2)

    assert list(recent.columns) == ["Time", "Strategy", "Run", "Summary", "Return %", "PF", "Sharpe", "Max DD %", "Trades", "Dataset"]
    assert top.iloc[0]["Strategy"] == "trend_pullback"
    assert float(top.iloc[0]["Metric"]) >= float(top.iloc[1]["Metric"])

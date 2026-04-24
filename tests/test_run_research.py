from __future__ import annotations

import json
import shutil
import uuid
from pathlib import Path

import pandas as pd

import run_research
from drift_monitor import load_baseline_profile


def _make_dataset(base_dir: Path, *, symbols: list[str], timeframe: str = "15Min") -> Path:
    dataset_dir = base_dir / "datasets" / f"sample_{uuid.uuid4().hex}"
    dataset_dir.mkdir(parents=True)
    (dataset_dir / "manifest.json").write_text(
        json.dumps(
            {
                "symbols": symbols,
                "timeframe": timeframe,
            }
        ),
        encoding="utf-8",
    )
    pd.DataFrame({"symbol": symbols}).to_parquet(dataset_dir / "bars.parquet", index=False)
    return dataset_dir


def test_build_drift_profile_payload_includes_operator_facing_metrics() -> None:
    base_dir = Path.cwd() / f"test_run_research_drift_{uuid.uuid4().hex}"
    try:
        dataset_dir = _make_dataset(base_dir, symbols=["AMD", "MSFT", "TSLA"])
        payload = run_research.build_drift_profile_payload(
            best_row={
                "strategy_mode": "mean_reversion",
                "trades_per_day": 6.0,
                "win_rate": 75.0,
                "profit_factor": 1.4,
                "sharpe_ratio": 1.9,
                "realized_pnl": 300.0,
                "combined_score": 2.0,
            },
            dataset_path=dataset_dir,
            approved=True,
            stability={"stable": True},
            regime={"regime": "sideways"},
        )

        assert payload["approved"] is True
        assert payload["context"]["strategy_mode"] == "mean_reversion"
        assert payload["context"]["symbol_count"] == 3
        assert payload["metrics"]["buy_signal_rate_per_symbol_per_day"] == 2.0
        assert payload["metrics"]["sell_signal_rate_per_symbol_per_day"] == 2.0
        assert payload["metrics"]["rejection_rate"] == run_research.DEFAULT_BACKTEST["rejection_rate"]
        assert payload["metrics"]["rejection_breakdown"] is None
        assert payload["metrics"]["hybrid_branch_participation"] is None
        assert payload["metrics"]["avg_bar_age_s"] == run_research.WARN["avg_bar_age_s"]
        assert payload["metrics"]["stale_bar_rate"] == 0.0
    finally:
        shutil.rmtree(base_dir, ignore_errors=True)


def test_build_drift_profile_payload_includes_hybrid_branch_participation() -> None:
    base_dir = Path.cwd() / f"test_run_research_drift_hybrid_{uuid.uuid4().hex}"
    try:
        dataset_dir = _make_dataset(base_dir, symbols=["AMD", "MSFT"])
        payload = run_research.build_drift_profile_payload(
            best_row={
                "strategy_mode": "hybrid_bb_mr",
                "trades_per_day": 8.0,
                "mr_branch_trade_share_pct": 62.5,
                "bb_branch_trade_share_pct": 37.5,
            },
            dataset_path=dataset_dir,
            approved=True,
            stability={"stable": True},
            regime={"regime": "sideways"},
        )

        assert payload["metrics"]["hybrid_branch_participation"] == {
            "mean_reversion": 0.625,
            "bollinger_breakout": 0.375,
        }
    finally:
        shutil.rmtree(base_dir, ignore_errors=True)


def test_save_drift_profile_writes_stable_json_shape_and_loader_can_read_it() -> None:
    base_dir = Path.cwd() / f"test_run_research_drift_write_{uuid.uuid4().hex}"
    try:
        (base_dir / "config").mkdir(parents=True)
        (base_dir / "results").mkdir(parents=True)
        dataset_dir = _make_dataset(base_dir, symbols=["AMD", "MSFT"])

        original_best = run_research.BEST_CONFIG_PATH
        original_trade = run_research.TRADE_DECISION_PATH
        original_stability = run_research.STABILITY_REPORT_PATH
        original_live = run_research.LIVE_CONFIG_PATH
        try:
            run_research.BEST_CONFIG_PATH = base_dir / "results" / "best_config_latest.json"
            run_research.TRADE_DECISION_PATH = base_dir / "results" / "trade_decision.json"
            run_research.STABILITY_REPORT_PATH = base_dir / "results" / "stability_report.json"
            run_research.LIVE_CONFIG_PATH = base_dir / "config" / "live_config.json"

            (run_research.BEST_CONFIG_PATH).write_text(
                json.dumps(
                    {
                        "approved": True,
                        "config": {"strategy_mode": "hybrid_bb_mr"},
                        "performance": {"trades_per_day": 8.0, "win_rate": 65.0},
                    }
                ),
                encoding="utf-8",
            )
            (run_research.TRADE_DECISION_PATH).write_text(json.dumps({}), encoding="utf-8")
            (run_research.STABILITY_REPORT_PATH).write_text(json.dumps({}), encoding="utf-8")
            (run_research.LIVE_CONFIG_PATH).write_text(
                json.dumps(
                    {
                        "runtime": {
                            "symbols": ["AMD", "MSFT"],
                            "bar_timeframe_minutes": 15,
                            "strategy_mode": "hybrid_bb_mr",
                        }
                    }
                ),
                encoding="utf-8",
            )

            output_path = base_dir / "results" / "drift_profile.json"
            run_research.step_save_drift_profile(
                {
                    "strategy_mode": "hybrid_bb_mr",
                    "trades_per_day": 8.0,
                    "win_rate": 65.0,
                    "profit_factor": 1.3,
                    "mr_branch_trade_share_pct": 70.0,
                    "bb_branch_trade_share_pct": 30.0,
                },
                dataset_dir,
                True,
                {"stable": True},
                {"regime": "sideways"},
                output_path,
            )

            payload = json.loads(output_path.read_text(encoding="utf-8"))
            assert sorted(payload.keys()) == [
                "approved",
                "context",
                "generated_at",
                "metric_sources",
                "metrics",
                "performance",
                "source_artifacts",
            ]
            assert payload["context"]["bar_timeframe_minutes"] == 15
            assert payload["metrics"]["hybrid_branch_participation"]["bollinger_breakout"] == 0.3

            profile = load_baseline_profile(base_dir)
            assert profile.strategy_mode == "hybrid_bb_mr"
            assert profile.buy_signal_rate_per_symbol_per_day == 4.0
            assert dict(profile.hybrid_branch_participation)["bollinger_breakout"] == 0.3
        finally:
            run_research.BEST_CONFIG_PATH = original_best
            run_research.TRADE_DECISION_PATH = original_trade
            run_research.STABILITY_REPORT_PATH = original_stability
            run_research.LIVE_CONFIG_PATH = original_live
    finally:
        shutil.rmtree(base_dir, ignore_errors=True)

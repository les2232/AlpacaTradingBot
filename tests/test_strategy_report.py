from __future__ import annotations

import json
import shutil
import uuid
from pathlib import Path

import strategy_report


def test_build_strategy_status_surfaces_live_runtime_fields_and_mismatch() -> None:
    base_dir = Path.cwd() / f"test_strategy_report_{uuid.uuid4().hex}"
    try:
        (base_dir / "config").mkdir(parents=True)
        (base_dir / "results").mkdir(parents=True)
        (base_dir / "config" / "live_config.json").write_text(
            json.dumps(
                {
                    "runtime": {
                        "symbols": ["AMD", "MSFT"],
                        "bar_timeframe_minutes": 15,
                        "strategy_mode": "mean_reversion",
                        "sma_bars": 15,
                        "entry_threshold_pct": 0.0015,
                        "mean_reversion_exit_style": "sma",
                        "mean_reversion_max_atr_percentile": 80.0,
                        "mean_reversion_trend_filter": True,
                        "mean_reversion_trend_slope_filter": True,
                        "mean_reversion_stop_pct": 0.01,
                        "regime_filter_enabled": False,
                    }
                }
            ),
            encoding="utf-8",
        )
        (base_dir / "results" / "best_config_latest.json").write_text(
            json.dumps(
                {
                    "approved": True,
                    "config": {
                        "strategy_mode": "mean_reversion",
                        "sma_bars": 15,
                        "entry_threshold_pct": 0.0015,
                        "mean_reversion_exit_style": "sma",
                        "mean_reversion_max_atr_percentile": 80.0,
                        "mean_reversion_trend_filter": False,
                        "mean_reversion_trend_slope_filter": False,
                        "mean_reversion_stop_pct": 0.0,
                    },
                    "performance": {
                        "profit_factor": 1.18,
                        "win_rate": 68.7,
                        "trades_per_day": 29.9,
                        "max_drawdown_pct": 3.8,
                    },
                }
            ),
            encoding="utf-8",
        )

        content = strategy_report.build_strategy_status(base_dir)

        assert "| Trend filter | enabled |" in content
        assert "| Trend slope filter | enabled |" in content
        assert "`mean_reversion_trend_filter` live=True research=False" in content
        assert "`mean_reversion_trend_slope_filter` live=True research=False" in content
    finally:
        shutil.rmtree(base_dir, ignore_errors=True)


def test_build_bot_reevaluation_returns_no_go_for_negative_edge_and_drift() -> None:
    base_dir = Path.cwd() / f"test_strategy_report_reeval_{uuid.uuid4().hex}"
    try:
        (base_dir / "config").mkdir(parents=True)
        (base_dir / "results" / "edge_diagnostics_current" / "live_effective").mkdir(parents=True)
        (base_dir / "results" / "trend_pullback_oos_validation").mkdir(parents=True)
        (base_dir / "results" / "volatility_expansion_validation").mkdir(parents=True)

        (base_dir / "config" / "live_config.json").write_text(
            json.dumps(
                {
                    "runtime": {
                        "strategy_mode": "mean_reversion",
                        "symbols": ["AMD", "MSFT"],
                        "mean_reversion_trend_filter": True,
                        "mean_reversion_trend_slope_filter": True,
                        "mean_reversion_stop_pct": 0.01,
                    }
                }
            ),
            encoding="utf-8",
        )
        (base_dir / "results" / "best_config_latest.json").write_text(
            json.dumps(
                {
                    "approved": True,
                    "config": {
                        "strategy_mode": "mean_reversion",
                        "mean_reversion_trend_filter": False,
                        "mean_reversion_trend_slope_filter": False,
                        "mean_reversion_stop_pct": 0.0,
                    },
                    "performance": {
                        "total_return_pct": 5.4,
                        "profit_factor": 1.18,
                    },
                }
            ),
            encoding="utf-8",
        )
        (base_dir / "results" / "week_full_live_config_2026-04-14_2026-04-21.csv").write_text(
            "\n".join(
                [
                    "total_return_pct,profit_factor",
                    "0.4640374982249704,1.041010865103201",
                ]
            ),
            encoding="utf-8",
        )
        (base_dir / "results" / "edge_diagnostics_current" / "live_effective" / "overall.csv").write_text(
            "\n".join(
                [
                    "horizon_bars,avg_net_expectancy_pct,net_profit_factor",
                    "1,-0.054,0.68",
                    "2,-0.044,0.80",
                    "4,-0.057,0.81",
                ]
            ),
            encoding="utf-8",
        )
        (base_dir / "results" / "edge_diagnostics_current" / "live_effective" / "by_symbol.csv").write_text(
            "\n".join(
                [
                    "symbol,horizon_bars,avg_net_expectancy_pct",
                    "AMD,1,-0.12",
                    "MSFT,1,0.07",
                ]
            ),
            encoding="utf-8",
        )
        (base_dir / "results" / "trend_pullback_oos_validation" / "trend_pullback_oos_summary.json").write_text(
            json.dumps(
                {
                    "classification": "research-only",
                    "reason": "Still fragile across windows.",
                }
            ),
            encoding="utf-8",
        )
        (base_dir / "results" / "volatility_expansion_validation" / "strategy_comparison.csv").write_text(
            "\n".join(
                [
                    "strategy_mode,realized_pnl,expectancy,profit_factor",
                    "trend_pullback,-63.9,-0.85,0.76",
                ]
            ),
            encoding="utf-8",
        )

        content = strategy_report.build_bot_reevaluation(base_dir)

        assert "- `NO-GO`" in content
        assert "live-effective edge diagnostics are negative after costs across all measured short horizons" in content
        assert "promoted research config does not match the current live runtime" in content
    finally:
        shutil.rmtree(base_dir, ignore_errors=True)

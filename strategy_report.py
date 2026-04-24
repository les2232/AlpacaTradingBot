#!/usr/bin/env python3
"""
Generate operator-facing markdown reports from canonical research artifacts.
"""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parent
RESULTS_DIR = REPO_ROOT / "results"
CONFIG_DIR = REPO_ROOT / "config"

BEST_CONFIG_PATH = RESULTS_DIR / "best_config_latest.json"
LIVE_CONFIG_PATH = CONFIG_DIR / "live_config.json"
STRATEGY_STATUS_PATH = RESULTS_DIR / "strategy_status.md"
BOT_REEVALUATION_PATH = RESULTS_DIR / "bot_reevaluation.md"
EXECUTION_CHECKLIST_PATH = RESULTS_DIR / "execution_checklist.md"


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _load_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            return list(csv.DictReader(handle))
    except OSError:
        return []


def _fmt_pct(value: Any, *, scale: float = 1.0, digits: int = 2, fallback: str = "n/a") -> str:
    try:
        numeric = float(value) * scale
    except (TypeError, ValueError):
        return fallback
    return f"{numeric:.{digits}f}%"


def _fmt_num(value: Any, digits: int = 2, fallback: str = "n/a") -> str:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return fallback
    return f"{numeric:.{digits}f}"


def _fmt_bool(value: Any) -> str:
    if value is True:
        return "enabled"
    if value is False:
        return "disabled"
    return "n/a"


def _normalize_runtime_for_comparison(runtime: dict[str, Any]) -> dict[str, Any]:
    normalized: dict[str, Any] = {}
    for key in (
        "strategy_mode",
        "sma_bars",
        "entry_threshold_pct",
        "ml_probability_buy",
        "ml_probability_sell",
        "threshold_mode",
        "atr_multiple",
        "atr_percentile_threshold",
        "time_window_mode",
        "regime_filter_enabled",
        "orb_filter_mode",
        "breakout_exit_style",
        "breakout_tight_stop_fraction",
        "mean_reversion_exit_style",
        "mean_reversion_max_atr_percentile",
        "mean_reversion_trend_filter",
        "mean_reversion_trend_slope_filter",
        "mean_reversion_stop_pct",
    ):
        if key in runtime:
            normalized[key] = runtime[key]
    return normalized


def _normalize_research_for_comparison(config: dict[str, Any]) -> dict[str, Any]:
    return _normalize_runtime_for_comparison(config)


def _summarize_config_mismatches(live_runtime: dict[str, Any], research_config: dict[str, Any]) -> list[str]:
    live_cmp = _normalize_runtime_for_comparison(live_runtime)
    research_cmp = _normalize_research_for_comparison(research_config)
    mismatches: list[str] = []
    for key, research_value in research_cmp.items():
        if key not in live_cmp:
            continue
        if live_cmp[key] != research_value:
            mismatches.append(f"`{key}` live={live_cmp[key]!r} research={research_value!r}")
    extra_runtime = [
        key
        for key in ("mean_reversion_trend_filter", "mean_reversion_trend_slope_filter", "mean_reversion_stop_pct")
        if key in live_cmp and key not in research_cmp and live_cmp[key] not in (False, 0, 0.0, None)
    ]
    for key in extra_runtime:
        mismatches.append(f"`{key}` live-only={live_cmp[key]!r} (missing from promoted research config)")
    return mismatches


def _top_symbol_rows(rows: list[dict[str, str]], *, horizon_bars: str = "1", limit: int = 5) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    filtered = [row for row in rows if row.get("horizon_bars") == horizon_bars]
    sorted_rows = sorted(
        filtered,
        key=lambda row: float(row.get("avg_net_expectancy_pct", "nan")),
    )
    worst = sorted_rows[:limit]
    best = list(reversed(sorted_rows[-limit:]))
    return best, worst


def build_strategy_status(repo_root: Path = REPO_ROOT) -> str:
    live_config = _load_json(repo_root / "config" / "live_config.json")
    best_config = _load_json(repo_root / "results" / "best_config_latest.json")
    runtime = live_config.get("runtime") if isinstance(live_config.get("runtime"), dict) else {}
    research_config = best_config.get("config") if isinstance(best_config.get("config"), dict) else {}
    performance = best_config.get("performance") if isinstance(best_config.get("performance"), dict) else {}
    mismatches = _summarize_config_mismatches(runtime, research_config)

    symbols = runtime.get("symbols") or []
    symbol_text = ", ".join(str(symbol) for symbol in symbols) if symbols else "n/a"
    lines = [
        "# Strategy Status",
        "",
        f"_Last generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}_",
        "",
        "## Current Live Strategy",
        "",
        "| Field | Value |",
        "|---|---|",
        f"| Mode | `{runtime.get('strategy_mode', 'n/a')}` |",
        f"| Symbols | {len(symbols)} ({symbol_text}) |",
        f"| Bar timeframe | {runtime.get('bar_timeframe_minutes', 'n/a')} minutes |",
        f"| SMA bars | `{runtime.get('sma_bars', 'n/a')}` |",
        f"| Entry threshold | {_fmt_pct(runtime.get('entry_threshold_pct'), scale=100.0, digits=2)} pullback below SMA |",
        f"| Mean-reversion exit | `{runtime.get('mean_reversion_exit_style', 'n/a')}` |",
        f"| Mean-reversion ATR cap | {_fmt_num(runtime.get('mean_reversion_max_atr_percentile'), digits=1)} |",
        f"| Trend filter | {_fmt_bool(runtime.get('mean_reversion_trend_filter'))} |",
        f"| Trend slope filter | {_fmt_bool(runtime.get('mean_reversion_trend_slope_filter'))} |",
        f"| Mean-reversion stop | {_fmt_pct(runtime.get('mean_reversion_stop_pct'), scale=100.0, digits=2)} |",
        f"| Regime filter | {_fmt_bool(runtime.get('regime_filter_enabled'))} |",
        f"| Config source | `config/live_config.json` |",
        "",
        "## Promoted Research Snapshot",
        "",
        f"- `results/best_config_latest.json` approval: `{best_config.get('approved', 'n/a')}`",
        f"- Profit factor: `{_fmt_num(performance.get('profit_factor'), digits=3)}`",
        f"- Win rate: `{_fmt_pct(performance.get('win_rate'), digits=1)}`",
        f"- Trades/day: `{_fmt_num(performance.get('trades_per_day'), digits=2)}`",
        f"- Max drawdown: `{_fmt_pct(performance.get('max_drawdown_pct'), digits=2)}`",
        "",
        "## Config Alignment",
        "",
    ]

    if mismatches:
        lines.append("- Status: `MISMATCHED`")
        lines.extend(f"- {item}" for item in mismatches)
    else:
        lines.append("- Status: `aligned`")
        lines.append("- Live runtime and promoted research config match on tracked fields.")

    lines.extend(
        [
            "",
            "## Caveats",
            "",
            "- This file is generated from `config/live_config.json` and `results/best_config_latest.json`.",
            "- If this status disagrees with recent diagnostics, trust the diagnostics and reevaluate before trading live capital.",
            "",
        ]
    )
    return "\n".join(lines)


def build_bot_reevaluation(repo_root: Path = REPO_ROOT) -> str:
    live_config = _load_json(repo_root / "config" / "live_config.json")
    best_config = _load_json(repo_root / "results" / "best_config_latest.json")
    runtime = live_config.get("runtime") if isinstance(live_config.get("runtime"), dict) else {}
    research_config = best_config.get("config") if isinstance(best_config.get("config"), dict) else {}
    performance = best_config.get("performance") if isinstance(best_config.get("performance"), dict) else {}

    edge_overall_rows = _load_csv_rows(repo_root / "results" / "edge_diagnostics_current" / "live_effective" / "overall.csv")
    week_rows = _load_csv_rows(repo_root / "results" / "week_full_live_config_2026-04-14_2026-04-21.csv")
    symbol_rows = _load_csv_rows(repo_root / "results" / "edge_diagnostics_current" / "live_effective" / "by_symbol.csv")
    trend_pullback_summary = _load_json(repo_root / "results" / "trend_pullback_oos_validation" / "trend_pullback_oos_summary.json")
    strategy_comparison_rows = _load_csv_rows(repo_root / "results" / "volatility_expansion_validation" / "strategy_comparison.csv")

    mismatches = _summarize_config_mismatches(runtime, research_config)
    negative_horizons = [
        row for row in edge_overall_rows
        if float(row.get("avg_net_expectancy_pct", "0")) < 0
    ]
    live_effective_negative = len(negative_horizons) == len(edge_overall_rows) and len(edge_overall_rows) > 0

    week_row = week_rows[0] if week_rows else {}
    week_profit_factor = float(week_row.get("profit_factor", "nan")) if week_row else None
    week_return_pct = float(week_row.get("total_return_pct", "nan")) if week_row else None

    verdict = "NO-GO"
    reasons: list[str] = []
    if mismatches:
        reasons.append("promoted research config does not match the current live runtime")
    if live_effective_negative:
        reasons.append("live-effective edge diagnostics are negative after costs across all measured short horizons")
    if week_profit_factor is not None and week_profit_factor <= 1.05:
        reasons.append("recent one-week forward validation is too close to break-even to trust with slippage and regime drift")
    if not reasons:
        verdict = "HOLD / NEED MORE DATA"
        reasons.append("no single hard fail triggered, but the evidence is not strong enough for fresh production confidence")

    best_symbols, worst_symbols = _top_symbol_rows(symbol_rows, horizon_bars="1", limit=5)
    comparison_lookup = {row.get("strategy_mode"): row for row in strategy_comparison_rows}
    trend_pullback_row = comparison_lookup.get("trend_pullback", {})

    lines = [
        "# Bot Reevaluation",
        "",
        f"_Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}_",
        "",
        "## Verdict",
        "",
        f"- `{verdict}`",
    ]
    lines.extend(f"- {reason}" for reason in reasons)
    lines.extend(
        [
            "",
            "## Current Mean-Reversion Evidence",
            "",
            f"- Promoted research window return: `{_fmt_pct(performance.get('total_return_pct'), digits=2)}` with PF `{_fmt_num(performance.get('profit_factor'), digits=3)}`",
            f"- Recent week (`2026-04-14` to `2026-04-21`) return: `{_fmt_pct(week_return_pct, digits=2)}` with PF `{_fmt_num(week_profit_factor, digits=3)}`",
        ]
    )
    if edge_overall_rows:
        for row in edge_overall_rows:
            lines.append(
                "- Live-effective horizon "
                f"`{row.get('horizon_bars')}` bars: net expectancy `{_fmt_pct(row.get('avg_net_expectancy_pct'), digits=3)}`"
                f", PF `{_fmt_num(row.get('net_profit_factor'), digits=3)}`"
            )

    lines.extend(
        [
            "",
            "## Runtime Trust Check",
            "",
        ]
    )
    if mismatches:
        lines.extend(f"- {item}" for item in mismatches)
    else:
        lines.append("- No tracked config drift detected between live runtime and promoted research config.")

    lines.extend(
        [
            "",
            "## Symbol Pressure",
            "",
            "- Worst 1-bar live-effective net expectancy names:",
        ]
    )
    lines.extend(
        f"  - {row.get('symbol')}: `{_fmt_pct(row.get('avg_net_expectancy_pct'), digits=3)}`"
        for row in worst_symbols
    )
    lines.append("- Best 1-bar live-effective net expectancy names:")
    lines.extend(
        f"  - {row.get('symbol')}: `{_fmt_pct(row.get('avg_net_expectancy_pct'), digits=3)}`"
        for row in best_symbols
    )

    lines.extend(
        [
            "",
            "## Candidate Replacement Check",
            "",
            f"- `trend_pullback_oos_validation` classification: `{trend_pullback_summary.get('classification', 'n/a')}`",
            f"- `trend_pullback_oos_validation` reason: {trend_pullback_summary.get('reason', 'n/a')}",
            f"- Recent `trend_pullback` live mismatch comparison: expectancy `{_fmt_num(trend_pullback_row.get('expectancy'), digits=3)}`,"
            f" PF `{_fmt_num(trend_pullback_row.get('profit_factor'), digits=3)}`, realized PnL `{_fmt_num(trend_pullback_row.get('realized_pnl'), digits=2)}`",
            "- Conclusion: the newer candidate lane is more interesting than the current mean-reversion lane, but it is still research-only rather than production-ready.",
            "",
            "## Recommended Next Step",
            "",
            "- Pause or heavily throttle the current live mean-reversion bot.",
            "- Refresh research artifacts using the exact live runtime fields now present in `config/live_config.json`.",
            "- Re-test with symbol pruning and stricter production promotion criteria before re-enabling normal size.",
            "",
        ]
    )
    return "\n".join(lines)


def build_execution_checklist(repo_root: Path = REPO_ROOT) -> str:
    live_config = _load_json(repo_root / "config" / "live_config.json")
    runtime = live_config.get("runtime") if isinstance(live_config.get("runtime"), dict) else {}
    return "\n".join(
        [
            "# Execution Checklist",
            "",
            f"_Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}_",
            "",
            "## Immediate",
            "",
            f"- Pause or heavily throttle the current `{runtime.get('strategy_mode', 'unknown')}` live lane.",
            "- Do not allow live startup when `config/live_config.json` and `results/best_config_latest.json` fail baseline validation.",
            "- Use `results/bot_reevaluation.md` as the current go/no-go summary.",
            "- Treat the current loser cluster as first-pass prune candidates: `BAC`, `WFC`, `AMD`, `TSLA`, `COP`.",
            "",
            "## This Week",
            "",
            "- Revalidate the exact live runtime fields now present in `config/live_config.json`.",
            "- Re-run recent-window evaluation with costs included and reject near-break-even configs.",
            "- Test symbol pruning before larger rule changes.",
            "- Test exit changes separately from entry changes.",
            "- Keep replacement candidates in research-only status unless they hold up across recent windows.",
            "",
            "## Only If Needed",
            "",
            "- Consider a broader strategy overhaul only if cleaned-up config parity still leaves backtest/live divergence.",
            "- Consider an architecture overhaul only if multiple strategy families remain fragile after the current controls cleanup.",
            "",
        ]
    )


def write_reports(repo_root: Path = REPO_ROOT) -> list[Path]:
    outputs = [
        (repo_root / "results" / "strategy_status.md", build_strategy_status(repo_root)),
        (repo_root / "results" / "bot_reevaluation.md", build_bot_reevaluation(repo_root)),
        (repo_root / "results" / "execution_checklist.md", build_execution_checklist(repo_root)),
    ]
    written: list[Path] = []
    for path, content in outputs:
        path.write_text(content + "\n", encoding="utf-8")
        written.append(path)
    return written


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate operator-facing strategy markdown reports.")
    parser.add_argument("--write", action="store_true", help="Write markdown reports to the results directory.")
    args = parser.parse_args()

    if args.write:
        written = write_reports(REPO_ROOT)
        for path in written:
            print(path)
        return

    print(build_bot_reevaluation(REPO_ROOT))


if __name__ == "__main__":
    main()

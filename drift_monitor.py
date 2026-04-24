from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from daily_report import DEFAULT_BACKTEST, WARN, load_backtest_baseline


DEFAULT_RESEARCH_DRIFT_PROFILE_PATH = Path("results") / "drift_profile.json"


@dataclass(frozen=True)
class BaselineProfile:
    source_label: str
    valid_for_comparison: bool
    validation_errors: tuple[str, ...]
    strategy_mode: str | None
    symbols: int | None
    bars_per_day: int | None
    buy_signal_rate_per_symbol_per_day: float | None
    sell_signal_rate_per_symbol_per_day: float | None
    rejection_rate: float | None
    rejection_breakdown: tuple[tuple[str, float], ...]
    hybrid_branch_participation: tuple[tuple[str, float], ...]
    avg_bar_age_s: float | None
    stale_bar_rate: float | None
    trade_count_per_day: float | None
    win_rate: float | None


@dataclass(frozen=True)
class LiveMetrics:
    window_label: str
    total_evaluated: int
    buy_signals: int
    sell_signals: int
    hold_count: int
    rejection_rate: float | None
    rejection_breakdown: tuple[tuple[str, int], ...]
    normalized_rejection_breakdown: tuple[tuple[str, float], ...]
    hybrid_branch_participation: tuple[tuple[str, float], ...]
    hybrid_branch_counts: tuple[tuple[str, int], ...]
    avg_bar_age_s: float | None
    stale_bar_count: int
    stale_bar_rate: float | None
    buy_signal_rate_per_symbol_per_day: float | None
    sell_signal_rate_per_symbol_per_day: float | None
    blocked_buy_signals: int
    blocked_sell_signals: int
    allowed_buy_checks: int
    allowed_sell_checks: int
    closed_trades: int
    win_rate: float | None


@dataclass(frozen=True)
class DriftAlert:
    key: str
    severity: str
    summary: str
    observed: str
    expected: str
    why_it_matters: str


@dataclass(frozen=True)
class DriftReport:
    baseline: BaselineProfile
    live: LiveMetrics
    alerts: tuple[DriftAlert, ...]

    @property
    def within_expected_profile(self) -> bool:
        return len(self.alerts) == 0 and self.baseline.valid_for_comparison


def _safe_float(value: Any) -> float | None:
    try:
        if value is None or value == "":
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _safe_int(value: Any) -> int | None:
    try:
        if value is None or value == "":
            return None
        return int(value)
    except (TypeError, ValueError):
        return None


def _load_json_file(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def _normalize_mix(value: Any) -> tuple[tuple[str, float], ...]:
    if not isinstance(value, dict):
        return ()
    pairs: list[tuple[str, float]] = []
    for key, raw in value.items():
        numeric = _safe_float(raw)
        if not key or numeric is None:
            continue
        if numeric > 1.0:
            numeric = numeric / 100.0
        pairs.append((str(key), numeric))
    return tuple(sorted(pairs, key=lambda item: (-item[1], item[0])))


def _find_latest_hybrid_research_profile(results_dir: Path) -> tuple[tuple[tuple[str, float], ...], str | None]:
    candidates = sorted(
        results_dir.rglob("hybrid_bb_mr_research_top.csv"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    for candidate in candidates:
        try:
            with candidate.open("r", encoding="utf-8", newline="") as handle:
                reader = csv.DictReader(handle)
                row = next(reader, None)
        except OSError:
            continue
        if not isinstance(row, dict):
            continue
        mr_share = _safe_float(row.get("mr_branch_trade_share_pct"))
        bb_share = _safe_float(row.get("bb_branch_trade_share_pct"))
        mix: list[tuple[str, float]] = []
        if mr_share is not None:
            mix.append(("mean_reversion", mr_share / 100.0 if mr_share > 1.0 else mr_share))
        if bb_share is not None:
            mix.append(("bollinger_breakout", bb_share / 100.0 if bb_share > 1.0 else bb_share))
        if mix:
            return tuple(sorted(mix, key=lambda item: (-item[1], item[0]))), str(candidate)
    return (), None


def load_baseline_profile(repo_root: Path | None = None) -> BaselineProfile:
    root = repo_root or Path(__file__).resolve().parent
    baseline, source = load_backtest_baseline(root)
    best_config = _load_json_file(root / "results" / "best_config_latest.json") or {}
    trade_decision = _load_json_file(root / "results" / "trade_decision.json") or {}
    overlay = _load_json_file(root / DEFAULT_RESEARCH_DRIFT_PROFILE_PATH)

    strategy_mode = None
    if isinstance(best_config.get("config"), dict):
        strategy_mode = str(best_config["config"].get("strategy_mode", "")) or None
    if strategy_mode is None and isinstance(trade_decision.get("selected_config"), dict):
        strategy_mode = str(trade_decision["selected_config"].get("strategy_mode", "")) or None

    rejection_breakdown: tuple[tuple[str, float], ...] = ()
    hybrid_branch_participation: tuple[tuple[str, float], ...] = ()
    avg_bar_age_s = float(WARN["avg_bar_age_s"])
    stale_bar_rate = 0.0
    buy_rate = _safe_float(baseline.get("signal_rate_per_symbol_per_day"))
    sell_rate = buy_rate
    trade_count_per_day = None
    win_rate = _safe_float(baseline.get("win_rate"))

    if isinstance(overlay, dict):
        metrics = overlay.get("metrics")
        context = overlay.get("context")
        performance = overlay.get("performance")
        approved = overlay.get("approved")
        if isinstance(context, dict):
            strategy_mode = str(context.get("strategy_mode", "")) or strategy_mode
            overlay_symbols = _safe_int(context.get("symbol_count"))
            overlay_bars_per_day = None
            timeframe = _safe_int(context.get("bar_timeframe_minutes"))
            if timeframe is not None and timeframe > 0:
                overlay_bars_per_day = max(1, int((6.5 * 60) / timeframe))
            if overlay_symbols is not None:
                baseline["symbols"] = overlay_symbols
            if overlay_bars_per_day is not None:
                baseline["bars_per_day"] = overlay_bars_per_day
        if isinstance(performance, dict):
            trade_count_per_day = _safe_float(performance.get("trades_per_day"))
            overlay_win_rate = _safe_float(performance.get("win_rate"))
            if overlay_win_rate is not None:
                win_rate = overlay_win_rate / 100.0 if overlay_win_rate > 1.0 else overlay_win_rate
        if approved is False:
            source["valid_for_comparison"] = False
            source["validation_errors"] = list(source.get("validation_errors", [])) + [
                "drift profile is approved=false"
            ]
        if isinstance(metrics, dict):
            overlay_buy_rate = _safe_float(metrics.get("buy_signal_rate_per_symbol_per_day"))
            overlay_sell_rate = _safe_float(metrics.get("sell_signal_rate_per_symbol_per_day"))
            overlay_rejection_rate = _safe_float(metrics.get("rejection_rate"))
            if overlay_buy_rate is not None:
                buy_rate = overlay_buy_rate
            if overlay_sell_rate is not None:
                sell_rate = overlay_sell_rate
            if overlay_rejection_rate is not None:
                baseline["rejection_rate"] = overlay_rejection_rate
            rejection_breakdown = _normalize_mix(metrics.get("rejection_breakdown"))
            hybrid_branch_participation = _normalize_mix(metrics.get("hybrid_branch_participation"))
            avg_bar_age_s = _safe_float(metrics.get("avg_bar_age_s")) or avg_bar_age_s
            stale_bar_rate = _safe_float(metrics.get("stale_bar_rate"))
            if stale_bar_rate is None:
                stale_bar_rate = 0.0

    if strategy_mode == "hybrid_bb_mr" and not hybrid_branch_participation:
        hybrid_branch_participation, branch_source = _find_latest_hybrid_research_profile(root / "results")
        if branch_source:
            source["branch_profile_path"] = branch_source

    if isinstance(best_config.get("performance"), dict):
        trade_count_per_day = _safe_float(best_config["performance"].get("trades_per_day"))
        win_rate = _safe_float(best_config["performance"].get("win_rate"))
        if win_rate is not None and win_rate > 1.0:
            win_rate = win_rate / 100.0

    source_parts = [str(source.get("mode", "defaults"))]
    if overlay is not None:
        source_parts.append("drift_profile")
    if strategy_mode == "hybrid_bb_mr" and hybrid_branch_participation:
        source_parts.append("hybrid_branch_research")

    return BaselineProfile(
        source_label="+".join(source_parts),
        valid_for_comparison=bool(source.get("valid_for_comparison", True)),
        validation_errors=tuple(str(item) for item in source.get("validation_errors", []) if str(item)),
        strategy_mode=strategy_mode,
        symbols=_safe_int(baseline.get("symbols")),
        bars_per_day=_safe_int(baseline.get("bars_per_day")),
        buy_signal_rate_per_symbol_per_day=buy_rate,
        sell_signal_rate_per_symbol_per_day=sell_rate,
        rejection_rate=_safe_float(baseline.get("rejection_rate")),
        rejection_breakdown=rejection_breakdown,
        hybrid_branch_participation=hybrid_branch_participation,
        avg_bar_age_s=avg_bar_age_s,
        stale_bar_rate=stale_bar_rate,
        trade_count_per_day=trade_count_per_day,
        win_rate=win_rate,
    )


def _normalized_signal_rate_per_symbol_per_day(
    *,
    count: int,
    total_evaluated: int,
    symbols: int | None,
    bars_per_day: int | None,
) -> float | None:
    if total_evaluated <= 0 or not symbols or symbols <= 0 or not bars_per_day or bars_per_day <= 0:
        return None
    bars_per_symbol_seen = total_evaluated / symbols
    if bars_per_symbol_seen <= 0:
        return None
    return (count / symbols) * (bars_per_day / bars_per_symbol_seen)


def _split_rejection_reason(reason: str | None) -> list[str]:
    if not reason:
        return []
    return [item.strip() for item in str(reason).split("|") if item.strip()]


def compute_live_metrics(
    *,
    signal_events: list[dict[str, Any]],
    bar_events: list[dict[str, Any]],
    risk_events: list[dict[str, Any]],
    position_events: list[dict[str, Any]] | None = None,
    baseline: BaselineProfile | None = None,
    window_label: str = "current session",
) -> LiveMetrics:
    position_events = position_events or []
    signal_rows = [payload for payload in signal_events if str(payload.get("event", "")) == "signal.evaluated"]
    bar_rows = [payload for payload in bar_events if str(payload.get("event", "")) == "bar.received"]
    risk_rows = [payload for payload in risk_events if str(payload.get("event", "")) == "risk.check"]
    closed_rows = [payload for payload in position_events if str(payload.get("event", "")) == "position.closed"]

    buy_signals = [row for row in signal_rows if str(row.get("action", "")).upper() == "BUY"]
    sell_signals = [row for row in signal_rows if str(row.get("action", "")).upper() == "SELL"]
    hold_rows = [row for row in signal_rows if str(row.get("action", "")).upper() == "HOLD"]

    rejection_counts: dict[str, int] = {}
    for row in hold_rows:
        split = _split_rejection_reason(row.get("rejection"))
        if not split:
            split = ["no_signal"]
        for reason in split:
            rejection_counts[reason] = rejection_counts.get(reason, 0) + 1
    total_rejections = sum(rejection_counts.values())
    normalized_rejection_breakdown = tuple(
        sorted(
            (
                (reason, count / total_rejections)
                for reason, count in rejection_counts.items()
                if total_rejections > 0
            ),
            key=lambda item: (-item[1], item[0]),
        )
    )

    branch_counts: dict[str, int] = {}
    for row in buy_signals:
        branch = str(
            row.get("hybrid_entry_branch")
            or row.get("hybrid_branch_active")
            or row.get("hybrid_branch")
            or ""
        ).strip()
        if not branch:
            continue
        branch_counts[branch] = branch_counts.get(branch, 0) + 1
    total_branch_count = sum(branch_counts.values())
    branch_participation = tuple(
        sorted(
            (
                (branch, count / total_branch_count)
                for branch, count in branch_counts.items()
                if total_branch_count > 0
            ),
            key=lambda item: (-item[1], item[0]),
        )
    )

    avg_bar_age_s = None
    stale_bar_count = 0
    stale_bar_rate = None
    if bar_rows:
        ages = [_safe_float(row.get("bar_age_s")) for row in bar_rows]
        clean_ages = [age for age in ages if age is not None]
        if clean_ages:
            avg_bar_age_s = sum(clean_ages) / len(clean_ages)
        stale_bar_count = sum(bool(row.get("stale")) for row in bar_rows)
        stale_bar_rate = stale_bar_count / len(bar_rows)

    blocked_buy = 0
    blocked_sell = 0
    allowed_buy = 0
    allowed_sell = 0
    for row in risk_rows:
        action = str(row.get("action", "")).upper()
        allowed = bool(row.get("allowed"))
        if action == "BUY":
            if allowed:
                allowed_buy += 1
            else:
                blocked_buy += 1
        elif action == "SELL":
            if allowed:
                allowed_sell += 1
            else:
                blocked_sell += 1

    win_rate = None
    if closed_rows:
        winners = 0
        counted = 0
        for row in closed_rows:
            winner = row.get("winner")
            if isinstance(winner, bool):
                winners += int(winner)
                counted += 1
                continue
            pnl = _safe_float(row.get("pnl_usd"))
            if pnl is not None:
                winners += int(pnl > 0)
                counted += 1
        if counted > 0:
            win_rate = winners / counted

    symbols = baseline.symbols if baseline is not None else None
    bars_per_day = baseline.bars_per_day if baseline is not None else None
    total_evaluated = len(signal_rows)
    buy_rate = _normalized_signal_rate_per_symbol_per_day(
        count=len(buy_signals),
        total_evaluated=total_evaluated,
        symbols=symbols,
        bars_per_day=bars_per_day,
    )
    sell_rate = _normalized_signal_rate_per_symbol_per_day(
        count=len(sell_signals),
        total_evaluated=total_evaluated,
        symbols=symbols,
        bars_per_day=bars_per_day,
    )

    return LiveMetrics(
        window_label=window_label,
        total_evaluated=total_evaluated,
        buy_signals=len(buy_signals),
        sell_signals=len(sell_signals),
        hold_count=len(hold_rows),
        rejection_rate=(len(hold_rows) / total_evaluated) if total_evaluated > 0 else None,
        rejection_breakdown=tuple(sorted(rejection_counts.items(), key=lambda item: (-item[1], item[0]))),
        normalized_rejection_breakdown=normalized_rejection_breakdown,
        hybrid_branch_participation=branch_participation,
        hybrid_branch_counts=tuple(sorted(branch_counts.items(), key=lambda item: (-item[1], item[0]))),
        avg_bar_age_s=avg_bar_age_s,
        stale_bar_count=stale_bar_count,
        stale_bar_rate=stale_bar_rate,
        buy_signal_rate_per_symbol_per_day=buy_rate,
        sell_signal_rate_per_symbol_per_day=sell_rate,
        blocked_buy_signals=blocked_buy,
        blocked_sell_signals=blocked_sell,
        allowed_buy_checks=allowed_buy,
        allowed_sell_checks=allowed_sell,
        closed_trades=len(closed_rows),
        win_rate=win_rate,
    )


def _mix_to_dict(items: tuple[tuple[str, float], ...]) -> dict[str, float]:
    return {key: value for key, value in items}


def _severity_for_ratio(delta_ratio: float) -> str:
    return "err" if delta_ratio >= 0.6 else "warn"


def evaluate_drift(baseline: BaselineProfile, live: LiveMetrics) -> tuple[DriftAlert, ...]:
    alerts: list[DriftAlert] = []

    if not baseline.valid_for_comparison:
        detail = "; ".join(baseline.validation_errors) or "baseline validation failed"
        alerts.append(
            DriftAlert(
                key="baseline_invalid",
                severity="warn",
                summary="Validated research profile is not clean enough for drift comparison.",
                observed=detail,
                expected="Approved, runtime-aligned baseline",
                why_it_matters="Without a trustworthy baseline, drift alerts become much less reliable for live operations.",
            )
        )
        return tuple(alerts)

    if (
        baseline.buy_signal_rate_per_symbol_per_day is not None
        and live.buy_signal_rate_per_symbol_per_day is not None
        and baseline.buy_signal_rate_per_symbol_per_day > 0
        and live.total_evaluated >= max((baseline.symbols or 1) * 4, 10)
    ):
        ratio = live.buy_signal_rate_per_symbol_per_day / baseline.buy_signal_rate_per_symbol_per_day
        if ratio < 0.65:
            alerts.append(
                DriftAlert(
                    key="buy_signal_rate_low",
                    severity=_severity_for_ratio(abs(1.0 - ratio)),
                    summary=f"Buy signal frequency is {(1.0 - ratio):.0%} below the validated profile.",
                    observed=f"{live.buy_signal_rate_per_symbol_per_day:.2f} per symbol/day",
                    expected=f"{baseline.buy_signal_rate_per_symbol_per_day:.2f} per symbol/day",
                    why_it_matters="Fewer entries than expected usually means live filters, market data, or market regime are suppressing the tested edge.",
                )
            )
        elif ratio > 1.5:
            alerts.append(
                DriftAlert(
                    key="buy_signal_rate_high",
                    severity=_severity_for_ratio(abs(ratio - 1.0)),
                    summary=f"Buy signal frequency is {(ratio - 1.0):.0%} above the validated profile.",
                    observed=f"{live.buy_signal_rate_per_symbol_per_day:.2f} per symbol/day",
                    expected=f"{baseline.buy_signal_rate_per_symbol_per_day:.2f} per symbol/day",
                    why_it_matters="Too many live signals can mean filters are no longer constraining trades the way research assumed.",
                )
            )

    if (
        baseline.rejection_rate is not None
        and live.rejection_rate is not None
        and live.total_evaluated >= 10
    ):
        delta = live.rejection_rate - baseline.rejection_rate
        if delta >= 0.12:
            alerts.append(
                DriftAlert(
                    key="rejection_rate_high",
                    severity="err" if delta >= 0.2 else "warn",
                    summary="Rejection rate materially exceeds the validated profile.",
                    observed=f"{live.rejection_rate:.0%}",
                    expected=f"{baseline.rejection_rate:.0%}",
                    why_it_matters="When the bot is rejecting far more setups than research, live behavior can drift away from the tested strategy profile even if the code is still running.",
                )
            )

    baseline_rejection_mix = _mix_to_dict(baseline.rejection_breakdown)
    live_rejection_mix = _mix_to_dict(live.normalized_rejection_breakdown)
    if baseline_rejection_mix and live_rejection_mix and live.hold_count >= 8:
        for reason, observed_share in live_rejection_mix.items():
            expected_share = baseline_rejection_mix.get(reason, 0.0)
            if observed_share >= expected_share + 0.2 and observed_share >= 0.25:
                alerts.append(
                    DriftAlert(
                        key=f"rejection_mix_{reason}",
                        severity="warn",
                        summary=f"{reason.replace('_', ' ').title()} is blocking far more often than the validated profile.",
                        observed=f"{observed_share:.0%} of rejections",
                        expected=f"{expected_share:.0%} of rejections",
                        why_it_matters="A shifted rejection mix usually points to a concrete operational or market-structure change, not just random noise.",
                    )
                )

    baseline_branch_mix = _mix_to_dict(baseline.hybrid_branch_participation)
    live_branch_mix = _mix_to_dict(live.hybrid_branch_participation)
    if baseline_branch_mix and live.buy_signals >= 4:
        for branch, expected_share in baseline_branch_mix.items():
            observed_share = live_branch_mix.get(branch, 0.0)
            if expected_share >= 0.15 and observed_share <= max(0.05, expected_share * 0.25):
                alerts.append(
                    DriftAlert(
                        key=f"hybrid_branch_{branch}_missing",
                        severity="warn",
                        summary=f"Hybrid branch mix is skewed: {branch.replace('_', ' ')} participation is near zero.",
                        observed=f"{observed_share:.0%} of branch-tagged buy signals",
                        expected=f"{expected_share:.0%} of branch-tagged buy signals",
                        why_it_matters="If one hybrid branch disappears in live trading, the bot may no longer be expressing the same opportunity mix that research validated.",
                    )
                )

    if live.avg_bar_age_s is not None and baseline.avg_bar_age_s is not None:
        if live.avg_bar_age_s > max(baseline.avg_bar_age_s * 1.5, baseline.avg_bar_age_s + 60.0):
            alerts.append(
                DriftAlert(
                    key="avg_bar_age_high",
                    severity="err" if live.avg_bar_age_s > baseline.avg_bar_age_s + 180.0 else "warn",
                    summary="Data freshness drift: average bar age is materially worse than baseline tolerance.",
                    observed=f"{live.avg_bar_age_s:.0f}s",
                    expected=f"{baseline.avg_bar_age_s:.0f}s or better",
                    why_it_matters="Older bars change both signal timing and trade quality, so this is an operator risk rather than just a feed-quality stat.",
                )
            )

    if live.stale_bar_rate is not None and baseline.stale_bar_rate is not None and len(live.hybrid_branch_counts) + live.total_evaluated > 0:
        if live.stale_bar_rate > max(baseline.stale_bar_rate + 0.03, 0.05):
            alerts.append(
                DriftAlert(
                    key="stale_bar_rate_high",
                    severity="err" if live.stale_bar_rate >= 0.1 else "warn",
                    summary="Stale bar rate is elevated versus the validated operating profile.",
                    observed=f"{live.stale_bar_rate:.0%} of received bars",
                    expected=f"{baseline.stale_bar_rate:.0%} of received bars",
                    why_it_matters="A rising stale-bar rate can suppress signals, distort branch mix, and make live behavior incomparable to research.",
                )
            )

    return tuple(alerts)


def build_drift_report(
    *,
    signal_events: list[dict[str, Any]],
    bar_events: list[dict[str, Any]],
    risk_events: list[dict[str, Any]],
    position_events: list[dict[str, Any]] | None = None,
    repo_root: Path | None = None,
    baseline: BaselineProfile | None = None,
    window_label: str = "current session",
) -> DriftReport:
    profile = baseline or load_baseline_profile(repo_root)
    live = compute_live_metrics(
        signal_events=signal_events,
        bar_events=bar_events,
        risk_events=risk_events,
        position_events=position_events,
        baseline=profile,
        window_label=window_label,
    )
    return DriftReport(
        baseline=profile,
        live=live,
        alerts=evaluate_drift(profile, live),
    )

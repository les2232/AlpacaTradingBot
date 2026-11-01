from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

from daily_report import WARN, load_backtest_baseline
from tradeos.research_bundle import DRIFT_PROFILE_POLICY_V1, load_verified_promotion_pointer


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
        if value is None or value == "" or isinstance(value, bool):
            return None
        numeric = float(value)
        return numeric if math.isfinite(numeric) else None
    except (OverflowError, TypeError, ValueError):
        return None


def _safe_int(value: Any) -> int | None:
    try:
        if value is None or value == "":
            return None
        return int(value)
    except (TypeError, ValueError):
        return None


def _strict_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    for key, value in pairs:
        if key in payload:
            raise ValueError(f"duplicate JSON field {key!r}")
        payload[key] = value
    return payload


def _reject_nonfinite_json_number(token: str) -> None:
    raise ValueError(f"non-finite JSON number {token!r}")


def _canonical_sha256(value: object) -> str | None:
    if (
        type(value) is not str
        or len(value) != 64
        or value != value.lower()
        or any(character not in "0123456789abcdef" for character in value)
    ):
        return None
    return value


def _strict_finite_number(value: object, *, field: str) -> float:
    if type(value) not in {int, float}:
        raise ValueError(f"{field} must be a JSON number")
    try:
        numeric = float(value)
    except (OverflowError, ValueError) as exc:
        raise ValueError(f"{field} must be finite") from exc
    if not math.isfinite(numeric):
        raise ValueError(f"{field} must be finite")
    return numeric


def _load_pointer_bound_json(
    repo_root: Path,
    reference: object,
) -> dict[str, Any]:
    if not isinstance(reference, dict) or set(reference) != {"path", "sha256"}:
        raise ValueError("drift projection reference fields are not exact")
    relative_path = reference["path"]
    expected_sha256 = _canonical_sha256(reference["sha256"])
    if type(relative_path) is not str or not relative_path:
        raise ValueError("drift projection path is missing or invalid")
    pure_path = PurePosixPath(relative_path)
    if (
        pure_path.is_absolute()
        or ".." in pure_path.parts
        or pure_path.as_posix() != relative_path
    ):
        raise ValueError("drift projection path is not safe repo-relative POSIX text")
    if expected_sha256 is None:
        raise ValueError("drift projection SHA-256 is missing or invalid")

    root = Path(repo_root).resolve(strict=True)
    try:
        path = (root / pure_path).resolve(strict=True)
        path.relative_to(root)
    except (OSError, ValueError) as exc:
        raise ValueError("drift projection path escapes the repository") from exc
    if not path.is_file():
        raise ValueError("drift projection target is not a file")
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected_sha256:
        raise ValueError("drift projection bytes do not match the commit pointer")
    try:
        payload = json.loads(
            raw.decode("utf-8", errors="strict"),
            object_pairs_hook=_strict_json_object,
            parse_constant=_reject_nonfinite_json_number,
        )
    except (UnicodeError, ValueError) as exc:
        raise ValueError("drift projection is not strict UTF-8 JSON") from exc
    if not isinstance(payload, dict):
        raise ValueError("drift projection must contain an object")
    return payload


def _validated_mix(value: object, *, field: str) -> tuple[tuple[str, float], ...]:
    if value is None:
        return ()
    if not isinstance(value, dict):
        raise ValueError(f"{field} must be an object or null")
    normalized: list[tuple[str, float]] = []
    for key, raw in value.items():
        if type(key) is not str or not key:
            raise ValueError(f"{field} contains an invalid entry")
        numeric = _strict_finite_number(raw, field=f"{field}.{key}")
        ratio = numeric / 100.0 if numeric > 1.0 else numeric
        if not 0.0 <= ratio <= 1.0:
            raise ValueError(f"{field} contains an out-of-range ratio")
        normalized.append((key, ratio))
    return tuple(sorted(normalized, key=lambda item: (-item[1], item[0])))


def _load_verified_drift_overlay(
    repo_root: Path,
    baseline: dict[str, Any],
    source: dict[str, Any],
) -> dict[str, Any]:
    generation_sha256 = _canonical_sha256(source.get("research_generation_sha256"))
    bundle_sha256 = _canonical_sha256(source.get("promotion_bundle_sha256"))
    if source.get("valid_for_comparison") is not True:
        raise ValueError("daily baseline is not valid for comparison")
    if generation_sha256 is None or bundle_sha256 is None:
        raise ValueError("daily baseline promotion identities are missing or invalid")

    root = Path(repo_root).resolve(strict=True)
    pointer = load_verified_promotion_pointer(
        root / "results" / "promotion_current.json",
        repo_root=root,
    )
    pointer_payload = pointer.payload
    bundle_reference = pointer_payload.get("bundle")
    artifacts = pointer_payload.get("artifacts")
    if not isinstance(bundle_reference, dict) or not isinstance(artifacts, dict):
        raise ValueError("promotion pointer references are missing or invalid")
    if (
        pointer_payload.get("research_generation_sha256") != generation_sha256
        or bundle_reference.get("bundle_sha256") != bundle_sha256
    ):
        raise ValueError("daily baseline and promotion pointer identities disagree")

    overlay = _load_pointer_bound_json(root, artifacts.get("drift_profile"))
    if (
        overlay.get("research_generation_sha256") != generation_sha256
        or overlay.get("bundle_sha256") != bundle_sha256
        or overlay.get("bundle_path") != bundle_reference.get("path")
        or overlay.get("approved") is not True
    ):
        raise ValueError("drift projection promotion identity or approval is invalid")

    context = overlay.get("context")
    performance = overlay.get("performance")
    metrics = overlay.get("metrics")
    if not isinstance(context, dict) or not isinstance(performance, dict) or not isinstance(metrics, dict):
        raise ValueError("drift projection context, performance, or metrics is invalid")
    strategy_mode = context.get("strategy_mode")
    symbol_count = context.get("symbol_count")
    timeframe = context.get("bar_timeframe_minutes")
    if (
        type(strategy_mode) is not str
        or not strategy_mode
        or type(symbol_count) is not int
        or symbol_count <= 0
        or type(timeframe) is not int
        or timeframe <= 0
        or context.get("stable") is not True
        or type(context.get("market_regime")) is not str
        or not context.get("market_regime")
    ):
        raise ValueError("drift projection approval context is invalid")
    expected_symbols = _safe_int(baseline.get("symbols"))
    expected_bars_per_day = _safe_int(baseline.get("bars_per_day"))
    if (
        symbol_count != expected_symbols
        or max(1, int((6.5 * 60) / timeframe)) != expected_bars_per_day
    ):
        raise ValueError("drift projection context contradicts the daily baseline")

    trades_per_day = _strict_finite_number(
        performance.get("trades_per_day"),
        field="performance.trades_per_day",
    )
    overlay_win_rate = _strict_finite_number(
        performance.get("win_rate"),
        field="performance.win_rate",
    )
    baseline_buy_rate = _safe_float(baseline.get("signal_rate_per_symbol_per_day"))
    baseline_win_rate = _safe_float(baseline.get("win_rate"))
    if (
        trades_per_day < 0
        or baseline_buy_rate is None
        or baseline_win_rate is None
        or trades_per_day / symbol_count != baseline_buy_rate
        or (overlay_win_rate / 100.0 if overlay_win_rate > 1.0 else overlay_win_rate)
        != baseline_win_rate
    ):
        raise ValueError("drift projection performance contradicts the daily baseline")

    overlay_buy_rate = _strict_finite_number(
        metrics.get("buy_signal_rate_per_symbol_per_day"),
        field="metrics.buy_signal_rate_per_symbol_per_day",
    )
    overlay_sell_rate = _strict_finite_number(
        metrics.get("sell_signal_rate_per_symbol_per_day"),
        field="metrics.sell_signal_rate_per_symbol_per_day",
    )
    overlay_rejection_rate = _strict_finite_number(
        metrics.get("rejection_rate"),
        field="metrics.rejection_rate",
    )
    if (
        overlay_buy_rate != baseline_buy_rate
        or overlay_sell_rate != baseline_buy_rate
    ):
        raise ValueError("drift projection metrics contradict the daily baseline")
    # The daily baseline has no research-backed rejection rate (its BACKTEST
    # value is a DEFAULT_BACKTEST placeholder), so the bound drift profile is
    # pinned to the versioned policy its producer writes instead.
    if overlay_rejection_rate != DRIFT_PROFILE_POLICY_V1.rejection_rate:
        raise ValueError("drift projection rejection rate does not match drift policy v1")
    avg_bar_age_s = _strict_finite_number(
        metrics.get("avg_bar_age_s"),
        field="metrics.avg_bar_age_s",
    )
    stale_bar_rate = _strict_finite_number(
        metrics.get("stale_bar_rate"),
        field="metrics.stale_bar_rate",
    )
    if (
        avg_bar_age_s < 0
        or not 0.0 <= stale_bar_rate <= 1.0
    ):
        raise ValueError("drift projection bar-quality metrics are invalid")
    _validated_mix(metrics.get("rejection_breakdown"), field="rejection_breakdown")
    hybrid_participation = _validated_mix(
        metrics.get("hybrid_branch_participation"),
        field="hybrid_branch_participation",
    )
    if strategy_mode == "hybrid_bb_mr" and not hybrid_participation:
        raise ValueError(
            "hybrid drift projection lacks bound branch participation evidence"
        )
    for key, value in performance.items():
        if value is not None:
            _strict_finite_number(value, field=f"performance.{key}")
    return overlay


def load_baseline_profile(repo_root: Path | None = None) -> BaselineProfile:
    root = (repo_root or Path(__file__).resolve().parent).resolve(strict=True)
    baseline, source = load_backtest_baseline(root)
    validation_errors = [
        str(item) for item in source.get("validation_errors", []) if str(item)
    ]
    overlay: dict[str, Any] | None = None
    if source.get("valid_for_comparison") is True:
        try:
            overlay = _load_verified_drift_overlay(root, baseline, source)
        except (OSError, RuntimeError, TypeError, UnicodeError, ValueError) as exc:
            source["valid_for_comparison"] = False
            validation_errors.append(
                f"pointer-bound drift profile is missing or invalid ({exc})"
            )

    strategy_mode: str | None = None

    rejection_breakdown: tuple[tuple[str, float], ...] = ()
    hybrid_branch_participation: tuple[tuple[str, float], ...] = ()
    avg_bar_age_s = float(WARN["avg_bar_age_s"])
    stale_bar_rate = 0.0
    buy_rate = _safe_float(baseline.get("signal_rate_per_symbol_per_day"))
    sell_rate = buy_rate
    symbol_count = _safe_int(baseline.get("symbols"))
    trade_count_per_day = (
        buy_rate * symbol_count
        if buy_rate is not None and symbol_count is not None and symbol_count > 0
        else None
    )
    win_rate = _safe_float(baseline.get("win_rate"))

    if isinstance(overlay, dict):
        metrics = overlay.get("metrics")
        context = overlay.get("context")
        if isinstance(context, dict):
            strategy_mode = str(context["strategy_mode"])
        if isinstance(metrics, dict):
            sell_rate = _safe_float(metrics["sell_signal_rate_per_symbol_per_day"])
            rejection_breakdown = _validated_mix(
                metrics.get("rejection_breakdown"),
                field="rejection_breakdown",
            )
            hybrid_branch_participation = _validated_mix(
                metrics.get("hybrid_branch_participation"),
                field="hybrid_branch_participation",
            )
            avg_bar_age_s = _safe_float(metrics["avg_bar_age_s"])
            stale_bar_rate = _safe_float(metrics["stale_bar_rate"])

    source_parts = [str(source.get("mode", "defaults"))]
    if overlay is not None:
        source_parts.append("pointer_drift_profile")

    if source.get("valid_for_comparison") is not True or not isinstance(overlay, dict):
        # Fail closed: an invalid baseline's numbers are DEFAULT_BACKTEST
        # placeholders (or unvalidated live/research values).  Carry none of
        # them, so neither drift expectations nor live per-symbol/day rates
        # normalized by placeholder symbols/bars_per_day can reach the panel.
        return BaselineProfile(
            source_label="+".join(source_parts),
            valid_for_comparison=False,
            validation_errors=tuple(validation_errors),
            strategy_mode=None,
            symbols=None,
            bars_per_day=None,
            buy_signal_rate_per_symbol_per_day=None,
            sell_signal_rate_per_symbol_per_day=None,
            rejection_rate=None,
            rejection_breakdown=(),
            hybrid_branch_participation=(),
            avg_bar_age_s=None,
            stale_bar_rate=None,
            trade_count_per_day=None,
            win_rate=None,
        )

    return BaselineProfile(
        source_label="+".join(source_parts),
        valid_for_comparison=source.get("valid_for_comparison") is True,
        validation_errors=tuple(validation_errors),
        strategy_mode=strategy_mode,
        symbols=_safe_int(baseline.get("symbols")),
        bars_per_day=_safe_int(baseline.get("bars_per_day")),
        buy_signal_rate_per_symbol_per_day=buy_rate,
        sell_signal_rate_per_symbol_per_day=sell_rate,
        rejection_rate=_safe_float(overlay["metrics"]["rejection_rate"]),
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
                    # baseline.rejection_rate is the drift policy v1 threshold
                    # bound into the promotion, not a research measurement, so
                    # the alert must not present it as the validated profile.
                    summary="Rejection rate materially exceeds the drift policy threshold.",
                    observed=f"{live.rejection_rate:.0%}",
                    expected=f"{baseline.rejection_rate:.0%} or lower (drift policy threshold, not research-measured)",
                    why_it_matters="The bot is rejecting far more setups than the operator drift policy allows. This threshold is a fixed policy value rather than a research result, so treat it as an operational check on filters and data, not as proof the strategy has drifted from research.",
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

"""Decide whether a historical session can validly be compared with a backtest.

The rule is fail-closed: a comparison is ``available`` only when every check
passes.  Otherwise the result is ``unavailable`` with the failed checks as
reasons, no expected (backtest) values are emitted, and conclusions that depend
on a comparison (underperformance, drift, improvement) are suppressed.  The
hard-coded ``daily_report.DEFAULT_BACKTEST`` fallback is never used as a
comparison target.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable

from tradeos.ask.records import LogRecord, SessionRef, StartupArtifact, is_sha256

PROMOTED_BASELINE = "promoted_baseline"
SUPPRESSED_CONCLUSIONS = ("underperformance", "outperformance", "drift", "improvement", "deterioration")
MIN_COVERAGE_FOR_COMPARISON = 0.5


@dataclass(frozen=True)
class ComparisonCheck:
    key: str
    label: str
    passed: bool
    detail: str
    source: str


@dataclass
class ComparisonResult:
    target: dict[str, Any]
    checks: list[ComparisonCheck]
    metric_comparisons: list[dict[str, Any]] = field(default_factory=list)
    drift_alerts: list[dict[str, Any]] = field(default_factory=list)

    @property
    def status(self) -> str:
        return "available" if self.checks and all(c.passed for c in self.checks) else "unavailable"

    @property
    def reasons(self) -> list[str]:
        return [c.detail for c in self.checks if not c.passed]

    def to_payload(self, *, include_details: bool) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "status": self.status,
            "target": self.target,
            "reasons": self.reasons,
        }
        if self.status != "available":
            payload["suppressed_conclusions"] = list(SUPPRESSED_CONCLUSIONS)
            payload["instruction"] = (
                "Comparison unavailable. Do not state or imply underperformance, outperformance, "
                "drift, improvement, or deterioration relative to any backtest. Observed session "
                "metrics may still be reported as observations."
            )
        if include_details:
            payload["checks"] = [
                {"check": c.key, "label": c.label, "passed": c.passed, "detail": c.detail} for c in self.checks
            ]
        if self.status == "available":
            payload["metric_comparisons"] = self.metric_comparisons
            payload["drift_alerts"] = self.drift_alerts
        return payload


def list_comparison_targets(experiment_log_path: Path | None) -> list[tuple[str, str]]:
    targets = [(PROMOTED_BASELINE, "Promoted backtest baseline (results/promotion_current.json)")]
    for entry in _load_experiments(experiment_log_path):
        run_id = entry.get("run_id")
        if isinstance(run_id, str) and run_id:
            label = f"Experiment {run_id} · {entry.get('strategy_name') or 'unknown strategy'}"
            targets.append((f"experiment:{run_id}", label))
    return targets


def _load_experiments(path: Path | None) -> list[dict[str, Any]]:
    if path is None or not path.is_file():
        return []
    try:
        from research.experiment_log import load_experiment_log

        return [e for e in load_experiment_log(path) if isinstance(e, dict)]
    except Exception:
        return []


def _default_baseline_loader(repo_root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    from daily_report import load_backtest_baseline

    return load_backtest_baseline(repo_root)


def _default_drift_profile_loader(repo_root: Path) -> Any:
    from drift_monitor import load_baseline_profile

    return load_baseline_profile(repo_root)


def _common_checks(
    *,
    session: SessionRef,
    configuration: dict[str, Any],
    artifact: StartupArtifact | None,
    environment: dict[str, Any],
    decisions: dict[str, Any],
) -> list[ComparisonCheck]:
    artifact_source = artifact.path if artifact else "computed:tradeos.ask.evidence._resolve_configuration"
    checks = [
        ComparisonCheck(
            "session_bound",
            "Records are bound to one session id",
            session.kind == "bound",
            "Session is identified by an explicit session id."
            if session.kind == "bound"
            else "Records have no session id, so they cannot be attributed to one session.",
            "computed:tradeos.ask.records.discover_sessions",
        ),
        ComparisonCheck(
            "configuration_recorded",
            "Session configuration recorded at launch",
            configuration.get("status") == "recorded_at_launch",
            "Startup artifact records the session's configuration."
            if configuration.get("status") == "recorded_at_launch"
            else f"Session configuration cannot be established ({configuration.get('status')}).",
            artifact_source,
        ),
    ]
    payload = artifact.payload if artifact else {}
    sha = payload.get("canonical_strategy_sha256")
    checks.append(
        ComparisonCheck(
            "strategy_identity_recorded",
            "Canonical strategy identity recorded for the session",
            is_sha256(sha),
            "Session recorded a canonical strategy hash."
            if is_sha256(sha)
            else "Session has no valid canonical strategy hash, so its strategy identity cannot be matched.",
            artifact_source,
        )
    )
    env_ok = (
        environment.get("recorded_at_launch") in {"paper", "live"}
        and environment.get("consistent") is not False
        and not environment.get("database_runs_missing_environment")
    )
    if env_ok:
        env_detail = f"Environment recorded as {environment.get('recorded_at_launch')}."
    elif environment.get("consistent") is False:
        env_detail = "Environment identifiers disagree between the startup artifact and persisted runs."
    elif environment.get("database_runs_missing_environment"):
        env_detail = "Persisted runs for this session lack an environment identifier."
    else:
        env_detail = "Session environment (paper/live) was not recorded."
    checks.append(ComparisonCheck("environment_identified", "Session environment identified", env_ok, env_detail, artifact_source))

    coverage = decisions.get("coverage_ratio")
    distinct = decisions.get("distinct_decisions_in_entry_window") or 0
    sample_ok = distinct > 0 and coverage is not None and coverage >= MIN_COVERAGE_FOR_COMPARISON
    checks.append(
        ComparisonCheck(
            "sufficient_session_sample",
            f"At least {MIN_COVERAGE_FOR_COMPARISON:.0%} of expected decisions observed",
            sample_ok,
            f"Observed {distinct} of {decisions.get('expected_decisions')} expected decisions."
            if sample_ok
            else (
                f"Insufficient session sample: {distinct} distinct decisions observed versus "
                f"{decisions.get('expected_decisions') if decisions.get('expected_decisions') is not None else 'an unknown number'} expected."
            ),
            "computed:tradeos.ask.evidence._decision_coverage",
        )
    )
    return checks


def _baseline_checks(
    *,
    artifact: StartupArtifact | None,
    repo_root: Path,
    baseline_loader: Callable[[Path], tuple[dict[str, Any], dict[str, Any]]],
) -> tuple[list[ComparisonCheck], dict[str, Any] | None, dict[str, Any]]:
    payload = artifact.payload if artifact else {}
    source_ref = artifact.path if artifact else "computed:tradeos.ask.evidence._resolve_configuration"
    checks: list[ComparisonCheck] = []
    launch_valid = payload.get("baseline_valid_for_comparison") is True
    launch_errors = [str(e) for e in payload.get("baseline_validation_errors") or [] if str(e).strip()]
    checks.append(
        ComparisonCheck(
            "baseline_valid_at_launch",
            "Baseline was valid for this session when it launched",
            launch_valid,
            "Startup artifact marks the baseline valid for comparison."
            if launch_valid
            else "Baseline was not valid for this session at launch"
            + (f": {'; '.join(launch_errors[:4])}" if launch_errors else " (validity not recorded)."),
            source_ref,
        )
    )
    try:
        base, source = baseline_loader(repo_root)
    except Exception as exc:  # loader failure is itself a reason
        base, source = None, {"valid_for_comparison": False, "validation_errors": [f"baseline could not be loaded ({type(exc).__name__})"]}
    current_valid = isinstance(source, dict) and source.get("valid_for_comparison") is True
    current_errors = [str(e) for e in (source or {}).get("validation_errors") or [] if str(e).strip()]
    checks.append(
        ComparisonCheck(
            "baseline_valid_now",
            "Saved backtest baseline passes current validation",
            current_valid,
            "Saved baseline passes validation."
            if current_valid
            else "Saved backtest baseline fails current validation: " + "; ".join(
                (e if len(e) <= 160 else e[:157] + "...") for e in current_errors[:5]
            ),
            "computed:daily_report.load_backtest_baseline (results/promotion_current.json)",
        )
    )
    identity_pairs = [
        (payload.get("promotion_bundle_sha256"), (source or {}).get("promotion_bundle_sha256")),
        (payload.get("research_generation_sha256"), (source or {}).get("research_generation_sha256")),
    ]
    identity_ok = all(is_sha256(a) and a == b for a, b in identity_pairs)
    checks.append(
        ComparisonCheck(
            "baseline_identity_matches_session",
            "Baseline identity matches the one the session launched against",
            identity_ok,
            "Session launched against the same promotion bundle and research generation."
            if identity_ok
            else "The session's recorded baseline identity is missing or differs from the current saved baseline.",
            source_ref,
        )
    )
    return checks, (base if current_valid else None), (source or {})


def _rate(count: int, evaluations: int, symbols: int, bars_per_day: int) -> float | None:
    if evaluations <= 0 or symbols <= 0 or bars_per_day <= 0:
        return None
    bars_seen = evaluations / symbols
    return (count / symbols) * (bars_per_day / bars_seen) if bars_seen > 0 else None


def _metric_comparisons(base: dict[str, Any], metrics: dict[str, Any], symbol_count: int) -> list[dict[str, Any]]:
    from daily_report import WARN

    rows: list[dict[str, Any]] = []
    signals = metrics.get("signals", {})
    positions = metrics.get("positions", {})
    execution = metrics.get("execution", {})
    bars_per_day = int(base.get("bars_per_day") or 0)

    def add(metric: str, observed: Any, expected: Any, *, sample_ok: bool = True, note: str = "") -> None:
        available = observed is not None and expected is not None and sample_ok
        delta = round(float(observed) - float(expected), 6) if available else None
        rows.append(
            {
                "metric": metric,
                "observed": observed,
                "expected": expected if available else None,
                "delta": delta,
                "available": available,
                "note": note or ("" if available else "insufficient sample or missing value"),
                "source": "computed:tradeos.ask.comparison._metric_comparisons",
            }
        )

    buy_rate = _rate(int(signals.get("buy") or 0), int(signals.get("evaluations") or 0), symbol_count, bars_per_day)
    add("buy_signal_rate_per_symbol_per_day", round(buy_rate, 4) if buy_rate is not None else None, base.get("signal_rate_per_symbol_per_day"))
    add("hold_rate", signals.get("hold_rate"), base.get("rejection_rate"))
    closed = int(positions.get("closed_unique_traces") or 0)
    win_ok = closed >= int(WARN["min_trades_for_win_rate"])
    add("win_rate", positions.get("win_rate"), base.get("win_rate"), sample_ok=win_ok,
        note="" if win_ok else f"only {closed} closed trade(s); need {WARN['min_trades_for_win_rate']}")
    add("avg_hold_bars", positions.get("avg_hold_bars"), base.get("avg_hold_bars"), sample_ok=closed > 0)
    add("avg_slippage_bps", execution.get("avg_slippage_bps"), base.get("avg_slippage_bps"))
    return rows


def _drift_alerts(
    profile_loader: Callable[[Path], Any], repo_root: Path, records: Iterable[LogRecord]
) -> tuple[list[dict[str, Any]], ComparisonCheck]:
    try:
        from drift_monitor import build_drift_report

        profile = profile_loader(repo_root)
        if not getattr(profile, "valid_for_comparison", False):
            errors = "; ".join(list(getattr(profile, "validation_errors", ()))[:3])
            return [], ComparisonCheck(
                "drift_profile_valid", "Pointer-bound drift profile valid", False,
                "Drift profile is not valid for comparison" + (f": {errors}" if errors else "."),
                "computed:drift_monitor.load_baseline_profile",
            )
        by_stream: dict[str, list[dict[str, Any]]] = {"signals": [], "bars": [], "risk": [], "positions": []}
        for record in records:
            if record.stream in by_stream:
                by_stream[record.stream].append(dict(record.payload))
        report = build_drift_report(
            signal_events=by_stream["signals"],
            bar_events=by_stream["bars"],
            risk_events=by_stream["risk"],
            position_events=by_stream["positions"],
            baseline=profile,
            window_label="selected session",
        )
    except Exception as exc:
        return [], ComparisonCheck(
            "drift_profile_valid", "Pointer-bound drift profile valid", False,
            f"Drift profile could not be evaluated ({type(exc).__name__}).",
            "computed:drift_monitor.build_drift_report",
        )
    alerts = [
        {"key": a.key, "severity": a.severity, "summary": a.summary, "observed": a.observed, "expected": a.expected}
        for a in report.alerts
    ]
    return alerts, ComparisonCheck(
        "drift_profile_valid", "Pointer-bound drift profile valid", True,
        "Drift profile is valid; alerts computed by drift_monitor.", "computed:drift_monitor.build_drift_report",
    )


def evaluate_comparison(
    *,
    session: SessionRef,
    configuration: dict[str, Any],
    artifact: StartupArtifact | None,
    environment: dict[str, Any],
    decisions: dict[str, Any],
    session_metrics: dict[str, Any],
    target: str,
    repo_root: Path,
    in_scope_records: list[LogRecord],
    baseline_loader: Callable[[Path], tuple[dict[str, Any], dict[str, Any]]] | None = None,
    drift_profile_loader: Callable[[Path], Any] | None = None,
    experiment_log_path: Path | None = None,
) -> ComparisonResult:
    checks = _common_checks(
        session=session, configuration=configuration, artifact=artifact, environment=environment, decisions=decisions
    )
    cfg_values = configuration.get("values") or {}
    symbol_count = int(cfg_values.get("symbol_count") or len(cfg_values.get("symbols") or []) or 0)

    if target == PROMOTED_BASELINE:
        target_info = {"key": target, "label": "Promoted backtest baseline", "source": "results/promotion_current.json"}
        baseline_checks, base, _source = _baseline_checks(
            artifact=artifact, repo_root=repo_root, baseline_loader=baseline_loader or _default_baseline_loader
        )
        checks.extend(baseline_checks)
        result = ComparisonResult(target=target_info, checks=checks)
        if all(c.passed for c in checks) and base is not None:
            alerts, drift_check = _drift_alerts(
                drift_profile_loader or _default_drift_profile_loader, repo_root, in_scope_records
            )
            result.checks.append(drift_check)
            if drift_check.passed:
                result.metric_comparisons = _metric_comparisons(base, session_metrics, symbol_count)
                result.drift_alerts = alerts
        return result

    if target.startswith("experiment:"):
        run_id = target.split(":", 1)[1]
        entries = {e.get("run_id"): e for e in _load_experiments(experiment_log_path)}
        entry = entries.get(run_id)
        log_ref = "results/experiment_log.jsonl"
        target_info = {"key": target, "label": f"Experiment {run_id}", "source": log_ref}
        checks.append(
            ComparisonCheck(
                "experiment_found", "Experiment entry exists", entry is not None,
                "Experiment entry found." if entry else f"Experiment {run_id} was not found in {log_ref}.", log_ref,
            )
        )
        session_sha = (artifact.payload if artifact else {}).get("canonical_strategy_sha256")
        entry_sha = (entry or {}).get("canonical_strategy_sha256")
        identity_ok = is_sha256(entry_sha) and entry_sha == session_sha
        detail = "Experiment canonical strategy identity matches the session."
        if not identity_ok:
            mismatches = []
            if entry:
                if cfg_values.get("strategy_mode") and entry.get("strategy_name") not in (None, cfg_values.get("strategy_mode")):
                    mismatches.append(f"strategy {entry.get('strategy_name')} vs session {cfg_values.get('strategy_mode')}")
                entry_symbols = sorted(str(s).upper() for s in entry.get("symbols") or [])
                session_symbols = sorted(str(s).upper() for s in cfg_values.get("symbols") or [])
                if entry_symbols and session_symbols and entry_symbols != session_symbols:
                    mismatches.append("symbol universe differs")
            detail = (
                "Experiment entry carries no matching canonical strategy identity, so it cannot be shown to use "
                "the session's configuration" + (f" ({'; '.join(mismatches)})" if mismatches else "") + "."
            )
        checks.append(ComparisonCheck("experiment_identity_matches_session", "Experiment strategy identity matches session", identity_ok, detail, log_ref))
        result = ComparisonResult(target=target_info, checks=checks)
        if result.status == "available" and entry:
            metrics = entry.get("metrics") or {}
            positions = session_metrics.get("positions", {})
            from daily_report import WARN

            closed = int(positions.get("closed_unique_traces") or 0)
            ok = closed >= int(WARN["min_trades_for_win_rate"])
            observed = positions.get("win_rate")
            expected = metrics.get("win_rate")
            result.metric_comparisons = [
                {
                    "metric": "win_rate",
                    "observed": observed,
                    "expected": expected if ok else None,
                    "delta": round(observed - expected, 6) if ok and observed is not None and expected is not None else None,
                    "available": bool(ok and observed is not None and expected is not None),
                    "note": "" if ok else f"only {closed} closed trade(s)",
                    "source": log_ref,
                }
            ]
        return result

    checks.append(
        ComparisonCheck("target_supported", "Comparison target recognised", False, f"Unknown comparison target {target!r}.", "computed:tradeos.ask.comparison")
    )
    return ComparisonResult(target={"key": target, "label": target, "source": None}, checks=checks)

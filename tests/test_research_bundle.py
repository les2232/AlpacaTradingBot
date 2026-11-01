from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace

import pandas as pd
import pytest

import run_research
import tradeos.research_bundle as research_bundle_module
from dataset_integrity import file_sha256, finalize_manifest
from trading_bot import BotConfig
from tradeos.research_bundle import (
    LatestResearchState,
    PromotionPointer,
    ResearchBundle,
    ResearchBundleValidationError,
    ResearchPublicationError,
    RUNTIME_ENVIRONMENT_DISTRIBUTIONS,
    assert_source_identity_unchanged,
    build_backtest_selection_evidence,
    build_live_config_payload,
    capture_dataset_checkpoint,
    capture_dataset_provenance,
    capture_runtime_environment,
    capture_source_identity,
    evaluate_approval_policy,
    evaluate_stability_policy,
    generation_directory_component,
    load_verified_latest_research_state,
    load_verified_promotion_pointer,
    normalize_metrics,
    publish_files_atomically,
    write_immutable_generation,
)
from tradeos.strategy_contract import (
    PROMOTED_STRATEGY_FIELDS,
    CanonicalResolvedPromotedStrategy,
)


GENERATED_AT = "2026-08-20T12:00:00Z"


def _sha256(contents: bytes) -> str:
    return hashlib.sha256(contents).hexdigest()


def _rehash_runtime_environment(payload: dict[str, object]) -> None:
    body = {
        key: value
        for key, value in payload.items()
        if key != "environment_sha256"
    }
    payload["environment_sha256"] = _sha256(
        json.dumps(
            body,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
        ).encode("utf-8")
    )


def _git(repo: Path, *args: str) -> None:
    completed = subprocess.run(
        ["git", *args],
        cwd=repo,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if completed.returncode != 0:
        raise AssertionError(completed.stderr.decode("utf-8", errors="replace"))


def _make_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "research_source.py").write_text("VALUE = 1\n", encoding="utf-8")
    _git(repo, "init", "-q")
    _git(repo, "add", "research_source.py")
    _git(
        repo,
        "-c",
        "user.name=TradeOS Test",
        "-c",
        "user.email=tradeos@example.invalid",
        "commit",
        "-q",
        "-m",
        "fixture",
    )
    return repo


def _canonical_strategy() -> CanonicalResolvedPromotedStrategy:
    resolved = run_research._resolve_research_inputs()
    symbols = list(resolved.symbols)
    config = BotConfig(
        symbols=symbols,
        max_usd_per_trade=1_000.0,
        max_symbol_exposure_usd=1_000.0,
        max_open_positions=3,
        max_daily_loss_usd=500.0,
        sma_bars=20,
        bar_timeframe_minutes=resolved.bar_timeframe_minutes,
        strategy_mode="mean_reversion",
    )
    return CanonicalResolvedPromotedStrategy.from_bot_config(config)


def _unused_model_evidence() -> tuple[str, str]:
    payload = {
        "schema_name": "tradeos.backtest_model_execution_evidence",
        "schema_version": 1,
        "used": False,
        "model_path": None,
        "model_sha256": None,
        "sidecar_path": None,
        "sidecar_declared_sha256": None,
        "sidecar_sha256": None,
        "feature_columns": [],
        "training_metadata": {},
        "training_dataset_identity": None,
    }
    encoded = json.dumps(
        payload,
        sort_keys=True,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
    )
    return encoded, _sha256(encoded.encode("utf-8"))


def _best_row(contract: CanonicalResolvedPromotedStrategy) -> dict[str, object]:
    model_json, model_sha = _unused_model_evidence()
    return {
        "symbols": ",".join(contract.symbols),
        **contract.promoted_values,
        "total_return_pct": 3.0,
        "profit_factor": 1.5,
        "max_drawdown_pct": 2.0,
        "sharpe_ratio": 1.1,
        "win_rate": 60.0,
        "trades_per_day": 1.0,
        "realized_pnl": 100.0,
        "combined_score": 1.0,
        "total_trades": 30.0,
        "expectancy": 0.25,
        "avg_return_per_trade_pct": 0.1,
        "canonical_strategy_json": contract.canonical_json(),
        "canonical_strategy_sha256": contract.sha256(),
        "model_execution_evidence_json": model_json,
        "model_execution_evidence_sha256": model_sha,
    }


def _write_dataset(
    repo: Path,
    *,
    lookback_days: int,
    contract: CanonicalResolvedPromotedStrategy,
    versioned: bool,
) -> Path:
    dataset = repo / "datasets" / f"snapshot_{lookback_days}d"
    dataset.mkdir(parents=True)
    frame = pd.DataFrame(
        {
            "symbol": list(contract.symbols),
            "timestamp": pd.to_datetime(
                [GENERATED_AT] * len(contract.symbols),
                utc=True,
            ),
            "close": [100.0 + index for index in range(len(contract.symbols))],
        }
    )
    bars_path = dataset / "bars.parquet"
    frame.to_parquet(bars_path, index=False)
    manifest = {
        "schema_version": 1,
        "dataset_id": dataset.name,
        "created_at_utc": GENERATED_AT,
        "code_hash": "a" * 64,
        "symbols": list(contract.symbols),
        "timeframe": f"{contract.promoted_values['bar_timeframe_minutes']}Min",
        "start_utc": (
            datetime.fromisoformat(GENERATED_AT.replace("Z", "+00:00"))
            - pd.Timedelta(days=lookback_days)
        ).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "end_utc": GENERATED_AT,
        "feed": run_research.SNAPSHOT_FEED.lower(),
        "adjustment": run_research.SNAPSHOT_ADJUSTMENT.lower(),
        "align_mode": run_research.SNAPSHOT_ALIGN_MODE,
        "row_count": len(frame),
    }
    if versioned:
        manifest = finalize_manifest(manifest, bars_sha256=file_sha256(bars_path))
    (dataset / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return dataset


def _targets(repo: Path) -> run_research.ResearchPublicationTargets:
    return run_research.ResearchPublicationTargets(
        repo_root=repo,
        bundle_directory=repo / "results" / "promotions",
        generation_directory=repo / "results" / "promotions" / "generations",
        pointer_path=repo / "results" / "promotion_current.json",
        latest_state_path=repo / "results" / "research_latest.json",
        best_config_path=repo / "results" / "best_config_latest.json",
        stability_report_path=repo / "results" / "stability_report.json",
        trade_decision_path=repo / "results" / "trade_decision.json",
        drift_profile_path=repo / "results" / "drift_profile.json",
        live_config_path=repo / "config" / "live_config.json",
    )


def _prepare(tmp_path: Path, *, versioned: bool = True) -> dict[str, object]:
    repo = _make_repo(tmp_path)
    contract = _canonical_strategy()
    targets = _targets(repo)
    targets.live_config_path.parent.mkdir(parents=True)
    targets.live_config_path.write_text(
        json.dumps(
            {
                "saved_at": "2000-01-01T00:00:00Z",
                "runtime": {
                    "symbols": list(contract.symbols),
                    **contract.promoted_values,
                    "excluded_symbols": ["NOT_ACTIVE"],
                    "private_runtime_section": {"keep": True},
                },
                "source": {
                    "approved": False,
                    "rejection_reasons": ["old rejection"],
                    "dataset": "datasets/old",
                    "private_source_field": "keep",
                },
                "private_top_level": {"keep": True},
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    source_identity = capture_source_identity(repo)
    resolved_inputs = run_research._resolve_research_inputs(targets.live_config_path)
    window_results: list[dict[str, object]] = []
    stability_windows: list[dict[str, object]] = []
    for lookback_days in run_research.VALIDATION_WINDOWS:
        dataset = _write_dataset(
            repo,
            lookback_days=lookback_days,
            contract=contract,
            versioned=versioned,
        )
        checkpoint = capture_dataset_checkpoint(dataset)
        provenance = capture_dataset_provenance(dataset, contract)
        sweep_csv = repo / "results" / f"sweep_{lookback_days}.csv"
        leaderboard_csv = repo / "results" / f"leaderboard_{lookback_days}.csv"
        sweep_csv.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame([_best_row(contract)]).to_csv(sweep_csv, index=False)
        row = run_research.step_rank_results(sweep_csv, leaderboard_csv)
        window_results.append(
            {
                "lookback_days": lookback_days,
                "dataset": dataset,
                "sweep_csv": sweep_csv,
                "leaderboard_csv": leaderboard_csv,
                "best_row": row,
                "dataset_checkpoint": checkpoint,
                "dataset_provenance": provenance,
            }
        )
        stability_windows.append(
            {
                "lookback_days": lookback_days,
                "canonical_strategy_sha256": contract.sha256(),
                "dataset_provenance_sha256": provenance["provenance_sha256"],
                "model_execution_evidence_json": row[
                    "model_execution_evidence_json"
                ],
                "model_execution_evidence_sha256": row[
                    "model_execution_evidence_sha256"
                ],
                "selection_evidence": row["research_selection_evidence"],
                "metrics": run_research._selected_metrics(row),
            }
        )
    stability = evaluate_stability_policy(
        stability_windows,
        run_research._stability_policy(),
    )
    metrics = run_research._selected_metrics(window_results[-1]["best_row"])
    approval = evaluate_approval_policy(metrics, run_research._approval_policy())
    assert approval.approved and stability["stable"] is True
    regime = {
        "policy": run_research._regime_policy().to_payload(),
        "regime": "sideways",
        "disabled_by_regime": False,
        "latest_bar_utc": GENERATED_AT,
        "avg_volatility": 0.001,
        "avg_sma_slope_pct": 0.0,
        "per_symbol": {
            symbol: {
                "regime": "sideways",
                "current_volatility": 0.001,
                "sma_slope_pct": 0.0,
            }
            for symbol in contract.symbols
        },
    }
    decision = {
        "should_trade": True,
        "approved": True,
        "stable": True,
        "disabled_by_regime": False,
        "market_regime": "sideways",
        "canonical_strategy_sha256": contract.sha256(),
    }
    return {
        "repo": repo,
        "contract": contract,
        "targets": targets,
        "source_identity": source_identity,
        "window_results": window_results,
        "stability": stability,
        "regime": regime,
        "decision": decision,
        "research_config_inputs": run_research._research_config_inputs(
            SimpleNamespace(dataset=None),
            resolved_inputs=resolved_inputs,
            research_anchor_utc=GENERATED_AT,
            window_ranges=[
                {
                    "lookback_days": days,
                    "start_utc": (
                        datetime.fromisoformat(GENERATED_AT.replace("Z", "+00:00"))
                        - pd.Timedelta(days=days)
                    ).strftime("%Y-%m-%dT%H:%M:%SZ"),
                    "end_utc": GENERATED_AT,
                }
                for days in run_research.VALIDATION_WINDOWS
            ],
        ),
    }


def _publish(
    prepared: dict[str, object],
    *,
    generated_at: str = GENERATED_AT,
    regime: dict[str, object] | None = None,
    decision: dict[str, object] | None = None,
) -> run_research.ResearchPublicationResult:
    return run_research.finalize_research_bundle_and_publish(
        window_results=prepared["window_results"],
        approved=True,
        rejection_reasons=[],
        stability=prepared["stability"],
        regime=regime or prepared["regime"],
        final_decision=decision or prepared["decision"],
        source_identity=prepared["source_identity"],
        research_config_inputs=prepared["research_config_inputs"],
        targets=prepared["targets"],
        generated_at_utc=generated_at,
    )


def _rebuild_single_candidate_selection(window: dict[str, object]) -> None:
    row = window["best_row"]
    assert isinstance(row, dict)
    sweep_csv = window["sweep_csv"]
    leaderboard_csv = window["leaderboard_csv"]
    assert isinstance(sweep_csv, Path) and isinstance(leaderboard_csv, Path)
    sweep_row = {
        key: value
        for key, value in row.items()
        if key
        not in {
            "source_row_index",
            "rank_return",
            "rank_pf",
            "rank_drawdown",
            "research_selection_evidence",
        }
    }
    pd.DataFrame([sweep_row]).to_csv(sweep_csv, index=False)
    window["best_row"] = run_research.step_rank_results(
        sweep_csv,
        leaderboard_csv,
    )


def _current_publication_bytes(
    targets: run_research.ResearchPublicationTargets,
) -> dict[Path, bytes]:
    paths = (
        targets.pointer_path,
        targets.latest_state_path,
        targets.best_config_path,
        targets.live_config_path,
        targets.stability_report_path,
        targets.trade_decision_path,
        targets.drift_profile_path,
    )
    return {path: path.read_bytes() for path in paths}


def test_risk_execution_context_uses_runtime_then_process_then_dotenv_defaults(
    tmp_path: Path,
) -> None:
    live_path = tmp_path / "live.json"
    live_path.write_text(
        json.dumps(
            {
                "runtime": {
                    "symbols": ["TSLA", "AMD"],
                    "bar_timeframe_minutes": 15,
                    "strategy_mode": "mean_reversion",
                    "max_usd_per_trade": "321.5",
                    "max_open_positions": 7,
                }
            }
        ),
        encoding="utf-8",
    )
    dotenv_path = tmp_path / ".env"
    dotenv_path.write_text(
        "\n".join(
            (
                "MAX_USD_PER_TRADE=111.5",
                "MAX_SYMBOL_EXPOSURE_USD=222.5",
                "MAX_OPEN_POSITIONS=4",
                "MAX_DAILY_LOSS_USD=333.5",
                "MAX_ORDERS_PER_MINUTE=8",
                "MAX_PRICE_DEVIATION_BPS=44.5",
                "MAX_DATA_DELAY_SECONDS=250",
                "MAX_LIVE_PRICE_AGE_SECONDS=50",
            )
        ),
        encoding="utf-8",
    )
    resolved = run_research._load_live_runtime_defaults(
        live_path,
        dotenv_path=dotenv_path,
        process_environment={"MAX_DAILY_LOSS_USD": "444.5"},
    )
    context = resolved["risk_execution_context"]
    assert context == {
        "max_usd_per_trade": 321.5,
        "max_symbol_exposure_usd": 222.5,
        "max_open_positions": 7,
        "max_daily_loss_usd": 444.5,
        "max_orders_per_minute": 8,
        "max_price_deviation_bps": 44.5,
        "max_data_delay_seconds": 250,
        "max_live_price_age_seconds": 50,
    }
    assert list(resolved["symbols"]) == ["TSLA", "AMD"]

    fallback = research_bundle_module.resolve_risk_execution_context(
        {},
        {"MAX_USD_PER_TRADE": "123.5"},
    )
    assert fallback["max_symbol_exposure_usd"] == fallback["max_usd_per_trade"]
    defaults = research_bundle_module.resolve_risk_execution_context({}, {})
    assert defaults == {
        "max_usd_per_trade": 200.0,
        "max_symbol_exposure_usd": 200.0,
        "max_open_positions": 3,
        "max_daily_loss_usd": 300.0,
        "max_orders_per_minute": 6,
        "max_price_deviation_bps": 75.0,
        "max_data_delay_seconds": 300,
        "max_live_price_age_seconds": 60,
    }

    with pytest.raises(ResearchBundleValidationError):
        research_bundle_module.resolve_risk_execution_context(
            {"max_open_positions": 0},
            {},
        )


def test_risk_execution_context_changes_config_and_generation_identity(
    tmp_path: Path,
) -> None:
    prepared = _prepare(tmp_path)
    first = _publish(prepared)
    targets = prepared["targets"]
    assert isinstance(targets, run_research.ResearchPublicationTargets)
    live = json.loads(targets.live_config_path.read_text(encoding="utf-8"))
    live["runtime"]["max_orders_per_minute"] += 1
    targets.live_config_path.write_text(json.dumps(live), encoding="utf-8")
    changed_inputs = run_research._resolve_research_inputs(targets.live_config_path)
    changed_config = deepcopy(prepared["research_config_inputs"])
    changed_config["resolved_live_inputs"] = changed_inputs.to_payload()
    prepared["research_config_inputs"] = changed_config

    second = _publish(prepared)
    assert second.bundle_sha256 != first.bundle_sha256
    assert second.research_generation_sha256 != first.research_generation_sha256


def test_eligible_publication_is_pointer_and_latest_state_verified(tmp_path: Path) -> None:
    prepared = _prepare(tmp_path)
    result = _publish(prepared)
    repo = prepared["repo"]
    targets = prepared["targets"]
    assert isinstance(repo, Path)
    assert isinstance(targets, run_research.ResearchPublicationTargets)
    assert result.published is True

    pointer = load_verified_promotion_pointer(targets.pointer_path, repo_root=repo)
    latest = load_verified_latest_research_state(
        targets.latest_state_path,
        repo_root=repo,
    )
    assert latest.payload["eligible"] is True
    assert latest.payload["research_generation_sha256"] == result.research_generation_sha256
    generation_component = generation_directory_component(
        result.research_generation_sha256
    )
    for artifact in pointer.payload["artifacts"].values():
        assert f"/generations/{generation_component}/" in artifact["path"]

    live = json.loads(targets.live_config_path.read_text(encoding="utf-8"))
    contract = prepared["contract"]
    assert isinstance(contract, CanonicalResolvedPromotedStrategy)
    assert live["private_top_level"] == {"keep": True}
    assert live["runtime"]["private_runtime_section"] == {"keep": True}
    assert live["source"]["private_source_field"] == "keep"
    assert live["source"]["rejection_reasons"] == []
    assert live["saved_at"] == GENERATED_AT
    assert live["runtime"]["symbols"] == list(contract.symbols)
    assert live["runtime"]["excluded_symbols"] == ["NOT_ACTIVE"]
    bound_risk_context = prepared["research_config_inputs"][
        "resolved_live_inputs"
    ]["risk_execution_context"]
    assert {
        field: live["runtime"][field]
        for field in research_bundle_module.RISK_EXECUTION_CONTEXT_FIELDS
    } == bound_risk_context

    # Compatibility aliases are not authoritative; changing them cannot alter
    # a committed immutable generation.
    for alias in (
        targets.best_config_path,
        targets.live_config_path,
        targets.stability_report_path,
        targets.trade_decision_path,
        targets.drift_profile_path,
    ):
        alias.write_bytes(b"{}\n")
    load_verified_promotion_pointer(targets.pointer_path, repo_root=repo)

    # Executable source drift invalidates even internally consistent old bytes.
    (repo / "research_source.py").write_text("VALUE = 2\n", encoding="utf-8")
    with pytest.raises(ResearchBundleValidationError, match="source|identity"):
        load_verified_promotion_pointer(targets.pointer_path, repo_root=repo)


def test_ineligible_latest_state_revokes_old_promotion_without_alias_changes(
    tmp_path: Path,
) -> None:
    prepared = _prepare(tmp_path)
    first = _publish(prepared)
    targets = prepared["targets"]
    repo = prepared["repo"]
    assert isinstance(targets, run_research.ResearchPublicationTargets)
    assert isinstance(repo, Path)
    protected_paths = (
        targets.pointer_path,
        targets.best_config_path,
        targets.live_config_path,
        targets.stability_report_path,
        targets.trade_decision_path,
        targets.drift_profile_path,
    )
    originals = {path: path.read_bytes() for path in protected_paths}
    old_state = targets.latest_state_path.read_bytes()

    contract = prepared["contract"]
    assert isinstance(contract, CanonicalResolvedPromotedStrategy)
    regime = {
        "policy": run_research._regime_policy().to_payload(),
        "regime": "high_volatility",
        "disabled_by_regime": True,
        "latest_bar_utc": "2026-08-21T12:00:00Z",
        "avg_volatility": 0.004,
        "avg_sma_slope_pct": 0.0,
        "per_symbol": {
            symbol: {
                "regime": "high_volatility",
                "current_volatility": 0.004,
                "sma_slope_pct": 0.0,
            }
            for symbol in contract.symbols
        },
    }
    decision = {
        "should_trade": False,
        "approved": True,
        "stable": True,
        "disabled_by_regime": True,
        "market_regime": "high_volatility",
        "canonical_strategy_sha256": contract.sha256(),
    }
    second = _publish(
        prepared,
        generated_at="2026-08-21T12:00:00Z",
        regime=regime,
        decision=decision,
    )
    assert first.research_generation_sha256 != second.research_generation_sha256
    assert second.published is False
    assert targets.latest_state_path.read_bytes() != old_state
    assert all(path.read_bytes() == originals[path] for path in protected_paths)
    latest = load_verified_latest_research_state(
        targets.latest_state_path,
        repo_root=repo,
    )
    assert latest.payload["eligible"] is False
    with pytest.raises(ResearchBundleValidationError):
        load_verified_promotion_pointer(targets.pointer_path, repo_root=repo)


def test_stability_failure_advances_only_ineligible_latest_state(tmp_path: Path) -> None:
    prepared = _prepare(tmp_path)
    _publish(prepared)
    targets = prepared["targets"]
    assert isinstance(targets, run_research.ResearchPublicationTargets)
    originals = _current_publication_bytes(targets)

    first_window = prepared["window_results"][0]
    assert isinstance(first_window, dict)
    first_row = first_window["best_row"]
    assert isinstance(first_row, dict)
    first_row["profit_factor"] = 1.0
    _rebuild_single_candidate_selection(first_window)
    stability = run_research.step_evaluate_stability(prepared["window_results"])
    assert stability["stable"] is False
    contract = prepared["contract"]
    assert isinstance(contract, CanonicalResolvedPromotedStrategy)
    decision = {
        "should_trade": False,
        "approved": True,
        "stable": False,
        "disabled_by_regime": False,
        "market_regime": "sideways",
        "canonical_strategy_sha256": contract.sha256(),
    }
    result = run_research.finalize_research_bundle_and_publish(
        window_results=prepared["window_results"],
        approved=True,
        rejection_reasons=[],
        stability=stability,
        regime=prepared["regime"],
        final_decision=decision,
        source_identity=prepared["source_identity"],
        research_config_inputs=prepared["research_config_inputs"],
        targets=targets,
        generated_at_utc="2026-08-20T12:00:01Z",
    )
    assert result.published is False
    assert targets.latest_state_path.read_bytes() != originals[targets.latest_state_path]
    for path, contents in originals.items():
        if path != targets.latest_state_path:
            assert path.read_bytes() == contents


def test_contradictory_final_decision_is_rejected_before_current_state_change(
    tmp_path: Path,
) -> None:
    prepared = _prepare(tmp_path)
    _publish(prepared)
    targets = prepared["targets"]
    assert isinstance(targets, run_research.ResearchPublicationTargets)
    originals = _current_publication_bytes(targets)
    contradictory = deepcopy(prepared["decision"])
    contradictory["should_trade"] = False
    with pytest.raises(ResearchBundleValidationError, match="Final decision"):
        _publish(
            prepared,
            generated_at="2026-08-20T12:00:01Z",
            decision=contradictory,
        )
    assert all(path.read_bytes() == contents for path, contents in originals.items())


def test_contradictory_approval_and_metrics_are_rejected_without_state_change(
    tmp_path: Path,
) -> None:
    prepared = _prepare(tmp_path)
    _publish(prepared)
    targets = prepared["targets"]
    assert isinstance(targets, run_research.ResearchPublicationTargets)
    originals = _current_publication_bytes(targets)
    primary = prepared["window_results"][-1]
    assert isinstance(primary, dict)
    row = primary["best_row"]
    assert isinstance(row, dict)
    row["profit_factor"] = 1.188961
    _rebuild_single_candidate_selection(primary)
    with pytest.raises(
        ResearchBundleValidationError,
        match="approval contradicts bound metrics",
    ):
        _publish(prepared, generated_at="2026-08-20T12:00:01Z")
    assert all(path.read_bytes() == contents for path, contents in originals.items())


def test_dataset_provenance_change_changes_bundle_and_generation_identity(
    tmp_path: Path,
) -> None:
    prepared = _prepare(tmp_path)
    first = _publish(prepared)
    changed_window = prepared["window_results"][0]
    assert isinstance(changed_window, dict)
    dataset = changed_window["dataset"]
    assert isinstance(dataset, Path)
    manifest_path = dataset / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["created_at_utc"] = "2026-08-20T11:59:59Z"
    manifest = finalize_manifest(
        manifest,
        bars_sha256=file_sha256(dataset / "bars.parquet"),
    )
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    contract = prepared["contract"]
    assert isinstance(contract, CanonicalResolvedPromotedStrategy)
    changed_window["dataset_checkpoint"] = capture_dataset_checkpoint(dataset)
    changed_window["dataset_provenance"] = capture_dataset_provenance(
        dataset,
        contract,
    )
    prepared["stability"] = run_research.step_evaluate_stability(
        prepared["window_results"]
    )
    second = _publish(prepared)
    assert second.bundle_sha256 != first.bundle_sha256
    assert second.research_generation_sha256 != first.research_generation_sha256


def test_runtime_environment_is_exact_and_changes_generation_and_pointer_validity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prepared = _prepare(tmp_path)
    first = _publish(prepared)
    targets = prepared["targets"]
    repo = prepared["repo"]
    assert isinstance(targets, run_research.ResearchPublicationTargets)
    assert isinstance(repo, Path)

    source_identity = prepared["source_identity"]
    assert isinstance(source_identity, dict)
    original_environment = source_identity["runtime_environment"]
    assert isinstance(original_environment, dict)
    distributions = original_environment["distributions"]
    assert isinstance(distributions, list)
    assert [entry["name"] for entry in distributions] == list(
        RUNTIME_ENVIRONMENT_DISTRIBUTIONS
    )

    changed_environment = deepcopy(original_environment)
    changed_distributions = changed_environment["distributions"]
    assert isinstance(changed_distributions, list)
    changed_distributions[0]["version"] += ".changed"
    _rehash_runtime_environment(changed_environment)
    monkeypatch.setattr(
        research_bundle_module,
        "capture_runtime_environment",
        lambda: deepcopy(changed_environment),
    )

    with pytest.raises(ResearchBundleValidationError, match="source|identity"):
        load_verified_promotion_pointer(targets.pointer_path, repo_root=repo)

    changed_source = deepcopy(source_identity)
    changed_source["runtime_environment"] = changed_environment
    prepared["source_identity"] = changed_source
    second = _publish(prepared)
    assert second.bundle_sha256 != first.bundle_sha256
    assert second.research_generation_sha256 != first.research_generation_sha256


def test_runtime_environment_missing_fields_and_distributions_fail_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo = _make_repo(tmp_path)
    identity = capture_source_identity(repo)
    missing_field = deepcopy(identity)
    del missing_field["runtime_environment"]["python"]["cache_tag"]
    with pytest.raises(ResearchBundleValidationError, match="Python identity"):
        assert_source_identity_unchanged(missing_field, identity)

    boolean_version = deepcopy(identity)
    boolean_version["runtime_environment"]["schema_version"] = True
    with pytest.raises(ResearchBundleValidationError, match="schema"):
        assert_source_identity_unchanged(boolean_version, identity)

    original_version = research_bundle_module.importlib_metadata.version

    def missing_distribution(name: str) -> str:
        if name == "numpy":
            raise research_bundle_module.importlib_metadata.PackageNotFoundError(name)
        return original_version(name)

    monkeypatch.setattr(
        research_bundle_module.importlib_metadata,
        "version",
        missing_distribution,
    )
    with pytest.raises(ResearchBundleValidationError, match="numpy"):
        capture_runtime_environment()


@pytest.mark.parametrize(
    "projection",
    ["best", "live", "live_risk", "stability", "trade", "drift"],
)
def test_projection_contradiction_fails_preflight_without_current_state_changes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    projection: str,
) -> None:
    prepared = _prepare(tmp_path)
    _publish(prepared)
    targets = prepared["targets"]
    assert isinstance(targets, run_research.ResearchPublicationTargets)
    protected_paths = (
        targets.pointer_path,
        targets.latest_state_path,
        targets.best_config_path,
        targets.live_config_path,
        targets.stability_report_path,
        targets.trade_decision_path,
        targets.drift_profile_path,
    )
    originals = {path: path.read_bytes() for path in protected_paths}
    original_serializer = run_research.json_file_bytes

    def contradictory_serializer(payload: dict[str, object]) -> bytes:
        candidate = deepcopy(payload)
        if projection == "best" and candidate.get("schema_name") == (
            "tradeos.promoted_best_config_projection"
        ):
            candidate["performance"]["profit_factor"] = 999.0
        if projection == "live" and "runtime" in candidate and "source" in candidate:
            candidate["source"]["canonical_strategy_sha256"] = "0" * 64
        if projection == "live_risk" and "runtime" in candidate and "source" in candidate:
            candidate["runtime"]["max_open_positions"] += 1
        if (
            projection == "stability"
            and candidate.get("schema_name")
            == "tradeos.promoted_stability_projection"
        ):
            candidate["validation_windows"][0] = 999
        if (
            projection == "trade"
            and candidate.get("schema_name")
            == "tradeos.promoted_trade_decision_projection"
        ):
            candidate["should_trade"] = False
        if (
            projection == "drift"
            and candidate.get("schema_name")
            == "tradeos.promoted_drift_profile_projection"
        ):
            candidate["metrics"]["avg_bar_age_s"] = 999_999.0
        return original_serializer(candidate)

    monkeypatch.setattr(run_research, "json_file_bytes", contradictory_serializer)
    with pytest.raises(ResearchBundleValidationError):
        _publish(prepared, generated_at="2026-08-20T12:00:01Z")
    assert all(path.read_bytes() == originals[path] for path in protected_paths)


@pytest.mark.parametrize("replace_index", range(7))
def test_publication_failure_restores_every_target(
    tmp_path: Path,
    replace_index: int,
) -> None:
    targets = [tmp_path / f"artifact_{index}.json" for index in range(6)]
    latest = tmp_path / "research_latest.json"
    all_targets = [*targets, latest]
    originals = {}
    for index, path in enumerate(all_targets):
        contents = f"old-{index}".encode("ascii")
        path.write_bytes(contents)
        originals[path] = contents

    def fail_after_replace(stage: str, index: int, _target: Path) -> None:
        if stage == "after_replace" and index == replace_index:
            raise RuntimeError("injected replacement failure")

    with pytest.raises(ResearchPublicationError):
        publish_files_atomically(
            {
                path: f"new-{index}".encode("ascii")
                for index, path in enumerate(targets)
            },
            commit_pointer=(latest, b"new-latest"),
            failure_injector=fail_after_replace,
        )
    assert all(path.read_bytes() == originals[path] for path in all_targets)


def test_publication_rollback_never_clobbers_a_later_external_write(
    tmp_path: Path,
) -> None:
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"
    first.write_bytes(b"old-first")
    second.write_bytes(b"old-second")

    def replace_then_external_write(stage: str, index: int, target: Path) -> None:
        if stage == "after_replace" and index == 0:
            target.write_bytes(b"external-writer")
            raise RuntimeError("injected failure after external writer")

    with pytest.raises(ResearchPublicationError, match="rollback was incomplete"):
        publish_files_atomically(
            {first: b"new-first", second: b"new-second"},
            failure_injector=replace_then_external_write,
        )

    assert first.read_bytes() == b"external-writer"
    assert second.read_bytes() == b"old-second"


def test_legacy_dataset_hashes_cannot_form_eligible_bundle(tmp_path: Path) -> None:
    prepared = _prepare(tmp_path, versioned=False)
    targets = prepared["targets"]
    assert isinstance(targets, run_research.ResearchPublicationTargets)
    original_live = targets.live_config_path.read_bytes()
    with pytest.raises(
        ResearchBundleValidationError,
        match="legacy|versioned dataset|verified declared",
    ):
        _publish(prepared)
    assert targets.live_config_path.read_bytes() == original_live
    assert not targets.pointer_path.exists()
    assert not targets.latest_state_path.exists()


def test_impossible_metric_and_regime_domains_fail_with_validation_error(
    tmp_path: Path,
) -> None:
    with pytest.raises(ResearchBundleValidationError):
        normalize_metrics(
            {
                "profit_factor": 10**10_000,
                "max_drawdown_pct": 1.0,
                "total_return_pct": 1.0,
                "trades_per_day": 1.0,
            }
        )
    with pytest.raises(ResearchBundleValidationError):
        normalize_metrics(
            {
                "profit_factor": 1.2,
                "max_drawdown_pct": -1.0,
                "total_return_pct": 1.0,
                "trades_per_day": 1.0,
            }
        )

    prepared = _prepare(tmp_path)
    result = _publish(prepared)
    payload = json.loads(result.bundle_path.read_text(encoding="utf-8"))
    symbol = next(iter(payload["regime"]["per_symbol"]))
    payload["regime"]["per_symbol"][symbol]["current_volatility"] = 10**10_000
    with pytest.raises(ResearchBundleValidationError):
        ResearchBundle.from_payload(payload)


def test_nonbest_selection_and_unsafe_generation_names_are_rejected(
    tmp_path: Path,
) -> None:
    common = {
        "canonical_strategy_sha256": "1" * 64,
        "model_execution_evidence_sha256": "2" * 64,
    }
    with pytest.raises(ResearchBundleValidationError, match="not the deterministic"):
        build_backtest_selection_evidence(
            sweep_csv_sha256="3" * 64,
            leaderboard_csv_sha256="4" * 64,
            selected_source_row_index=0,
            candidates=[
                {
                    "source_row_index": 0,
                    "total_return_pct": 1.0,
                    "profit_factor": 1.0,
                    "max_drawdown_pct": 5.0,
                    **common,
                },
                {
                    "source_row_index": 1,
                    "total_return_pct": 2.0,
                    "profit_factor": 2.0,
                    "max_drawdown_pct": 1.0,
                    **common,
                },
            ],
        )
    for unsafe_name in ("..\\escape.json", "C:\\outside.json", "../escape.json"):
        with pytest.raises((ResearchBundleValidationError, ResearchPublicationError)):
            write_immutable_generation(
                "a" * 64,
                {unsafe_name: b"{}\n"},
                tmp_path / "generations",
            )


def test_overlapping_exclusions_and_boolean_schema_versions_fail_closed(
    tmp_path: Path,
) -> None:
    contract = _canonical_strategy()
    canonical = run_research.BacktestCanonicalStrategy(
        canonical_json=contract.canonical_json(),
        sha256=contract.sha256(),
        contract=contract,
    )
    with pytest.raises(ResearchBundleValidationError, match="exclusions"):
        build_live_config_payload(
            {
                "runtime": {"excluded_symbols": [contract.symbols[0]]},
                "source": {},
            },
            canonical,
            risk_execution_context=(
                research_bundle_module.resolve_risk_execution_context({}, {})
            ),
            source_updates={},
        )

    prepared = _prepare(tmp_path)
    _publish(prepared)
    targets = prepared["targets"]
    assert isinstance(targets, run_research.ResearchPublicationTargets)
    pointer_payload = json.loads(targets.pointer_path.read_text(encoding="utf-8"))
    pointer_payload["schema_version"] = True
    with pytest.raises(ResearchBundleValidationError):
        PromotionPointer.from_payload(pointer_payload)
    state_payload = json.loads(targets.latest_state_path.read_text(encoding="utf-8"))
    state_payload["schema_version"] = True
    with pytest.raises(ResearchBundleValidationError):
        LatestResearchState.from_payload(state_payload)


def test_deprecated_approval_only_writers_never_touch_targets(tmp_path: Path) -> None:
    assert run_research.CONFIG_COLS == list(PROMOTED_STRATEGY_FIELDS)
    target = tmp_path / "must_not_exist.json"
    with pytest.raises(ResearchPublicationError):
        run_research.step_save_best_config({}, tmp_path, target, True, [])
    with pytest.raises(ResearchPublicationError):
        run_research.step_write_live_config({}, tmp_path, target, True, [])
    assert not target.exists()

from __future__ import annotations

from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace

import pandas as pd
import pytest

import run_research
from backtest_runner import (
    build_backtest_canonical_strategy_evidence,
    extract_clean_columns,
)
from dataset_integrity import file_sha256, finalize_manifest
from trading_bot import BotConfig
from tradeos.research_bundle import (
    ResearchBundle,
    ResearchBundleValidationError,
    capture_dataset_checkpoint,
    capture_dataset_provenance,
    capture_source_identity,
    load_verified_latest_research_state,
    load_verified_promotion_pointer,
    validate_backtest_selection_artifacts,
)
from tradeos.strategy_contract import (
    PROMOTED_STRATEGY_FIELDS,
    RUNTIME_FIXED_STRATEGY_ASSUMPTIONS,
    CanonicalResolvedPromotedStrategy,
)


PAPER_SYMBOLS = [
    "ABBV",
    "AMD",
    "BAC",
    "C",
    "COP",
    "GOOGL",
    "HD",
    "HON",
    "JPM",
    "LLY",
    "MSFT",
    "QCOM",
    "TSLA",
    "WFC",
    "XOM",
]
GENERATED_AT = "2026-08-20T12:00:00Z"
SNAPSHOT_END = datetime(2026, 8, 20, tzinfo=timezone.utc)
_SELECTOR_FIELDS = {
    "strategy_mode",
    "sma_bars",
    "symbol_strategy_modes",
    "bar_timeframe_minutes",
}


def _git(repo: Path, *arguments: str) -> None:
    completed = subprocess.run(
        ["git", *arguments],
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


def _producer_evidence(*, entry_threshold_pct: float) -> dict[str, object]:
    config = BotConfig(
        symbols=list(PAPER_SYMBOLS),
        max_usd_per_trade=1_000.0,
        max_symbol_exposure_usd=1_000.0,
        max_open_positions=3,
        max_daily_loss_usd=500.0,
        sma_bars=15,
        bar_timeframe_minutes=15,
        strategy_mode="mean_reversion",
        entry_threshold_pct=entry_threshold_pct,
    )
    runtime_contract = CanonicalResolvedPromotedStrategy.from_bot_config(config)
    promoted = runtime_contract.promoted_values
    symbol_modes = promoted.pop("symbol_strategy_modes")
    strategy_mode = promoted.pop("strategy_mode")
    sma_bars = promoted.pop("sma_bars")
    timeframe = promoted.pop("bar_timeframe_minutes")
    assert set(promoted) == set(PROMOTED_STRATEGY_FIELDS) - _SELECTOR_FIELDS
    evidence = build_backtest_canonical_strategy_evidence(
        symbols=PAPER_SYMBOLS,
        bar_timeframe_minutes=timeframe,
        sma_bars=sma_bars,
        strategy_mode=strategy_mode,
        symbol_strategy_modes=symbol_modes,
        strategy_values=promoted,
        runtime_fixed_strategy_values=dict(RUNTIME_FIXED_STRATEGY_ASSUMPTIONS),
    )
    produced_contract = evidence["contract"]
    assert isinstance(produced_contract, CanonicalResolvedPromotedStrategy)
    assert produced_contract.canonical_bytes() == runtime_contract.canonical_bytes()
    return evidence


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
    canonical = json.dumps(
        payload,
        sort_keys=True,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
    )
    return canonical, hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _backtest_row(
    evidence: dict[str, object],
    *,
    total_return_pct: float,
    profit_factor: float,
    max_drawdown_pct: float,
) -> dict[str, object]:
    contract = evidence["contract"]
    assert isinstance(contract, CanonicalResolvedPromotedStrategy)
    model_json, model_sha256 = _unused_model_evidence()
    raw = {
        "regime": "Custom",
        "start_date": None,
        "end_date": None,
        "symbols": list(contract.symbols),
        "runtime_fixed_strategy_values": dict(
            evidence["runtime_fixed_strategy_values"]  # type: ignore[arg-type]
        ),
        **contract.promoted_values,
        "canonical_strategy_json": evidence["canonical_strategy_json"],
        "canonical_strategy_sha256": evidence["canonical_strategy_sha256"],
        "model_execution_evidence_json": model_json,
        "model_execution_evidence_sha256": model_sha256,
        "total_return_pct": total_return_pct,
        "profit_factor": profit_factor,
        "max_drawdown_pct": max_drawdown_pct,
        "sharpe_ratio": 1.1,
        "win_rate": 60.0,
        "trades_per_day": 1.0,
        "realized_pnl": 100.0,
        "total_trades": 30,
        "expectancy": 0.25,
        "avg_return_per_trade_pct": 0.1,
    }
    return extract_clean_columns(raw)


def _write_dataset(
    repo: Path,
    *,
    lookback_days: int,
    contract: CanonicalResolvedPromotedStrategy,
) -> Path:
    dataset = repo / "datasets" / f"snapshot_{lookback_days}d"
    dataset.mkdir(parents=True)
    frame = pd.DataFrame(
        {
            "symbol": list(contract.symbols),
            "timestamp": pd.to_datetime(
                [SNAPSHOT_END] * len(contract.symbols),
                utc=True,
            ),
            "open": [100.0 + index for index in range(len(contract.symbols))],
            "high": [101.0 + index for index in range(len(contract.symbols))],
            "low": [99.0 + index for index in range(len(contract.symbols))],
            "close": [100.5 + index for index in range(len(contract.symbols))],
            "volume": [1_000 + index for index in range(len(contract.symbols))],
        }
    )
    bars_path = dataset / "bars.parquet"
    frame.to_parquet(bars_path, index=False)
    start = SNAPSHOT_END - timedelta(days=lookback_days)
    manifest = finalize_manifest(
        {
            "schema_version": 1,
            "dataset_id": dataset.name,
            "created_at_utc": GENERATED_AT,
            "code_hash": "a" * 12,
            "symbols": list(contract.symbols),
            "timeframe": "15Min",
            "start_utc": start.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "end_utc": SNAPSHOT_END.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "feed": "iex",
            "adjustment": "raw",
            "align_mode": "shared",
            "row_count": len(frame),
        },
        bars_sha256=file_sha256(bars_path),
    )
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


def test_full_roundtrip(tmp_path: Path) -> None:
    repo = _make_repo(tmp_path)
    targets = _targets(repo)
    targets.live_config_path.parent.mkdir(parents=True)
    targets.live_config_path.write_text(
        json.dumps(
            {
                "saved_at": "2000-01-01T00:00:00Z",
                "runtime": {
                    "symbols": list(PAPER_SYMBOLS),
                    "excluded_symbols": [],
                    "bar_timeframe_minutes": 15,
                    "strategy_mode": "mean_reversion",
                    "data_parity": {"historical_feed": "iex"},
                },
                "source": {"approved": False, "rejection_reasons": ["old"]},
            }
        ),
        encoding="utf-8",
    )
    source_identity = capture_source_identity(repo)
    winner_evidence = _producer_evidence(entry_threshold_pct=0.0015)
    loser_evidence = _producer_evidence(entry_threshold_pct=0.003)
    winner_contract = winner_evidence["contract"]
    assert isinstance(winner_contract, CanonicalResolvedPromotedStrategy)

    window_results: list[dict[str, object]] = []
    for lookback_days in run_research.VALIDATION_WINDOWS:
        dataset = _write_dataset(
            repo,
            lookback_days=lookback_days,
            contract=winner_contract,
        )
        checkpoint = capture_dataset_checkpoint(dataset)
        sweep = repo / "results" / f"sweep_{lookback_days}.csv"
        leaderboard = repo / "results" / f"leaderboard_{lookback_days}.csv"
        sweep.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(
            [
                _backtest_row(
                    loser_evidence,
                    total_return_pct=1.0,
                    profit_factor=1.1,
                    max_drawdown_pct=5.0,
                ),
                _backtest_row(
                    winner_evidence,
                    total_return_pct=3.0,
                    profit_factor=1.5,
                    max_drawdown_pct=2.0,
                ),
            ]
        ).to_csv(sweep, index=False)
        best_row = run_research.step_rank_results(sweep, leaderboard)
        assert best_row["canonical_strategy_sha256"] == winner_contract.sha256()
        provenance = capture_dataset_provenance(dataset, winner_contract)
        window_results.append(
            {
                "lookback_days": lookback_days,
                "dataset": dataset,
                "sweep_csv": sweep,
                "leaderboard_csv": leaderboard,
                "best_row": best_row,
                "dataset_checkpoint": checkpoint,
                "dataset_provenance": provenance,
            }
        )

    stability = run_research.step_evaluate_stability(window_results)
    approved, rejection_reasons = run_research.step_approve_config(
        window_results[-1]["best_row"]  # type: ignore[arg-type]
    )
    assert approved is True
    assert rejection_reasons == []
    regime = {
        "policy": run_research._regime_policy().to_payload(),
        "regime": "sideways",
        "disabled_by_regime": False,
        "latest_bar_utc": SNAPSHOT_END.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "avg_volatility": 0.001,
        "avg_sma_slope_pct": 0.0,
        "per_symbol": {
            symbol: {
                "regime": "sideways",
                "current_volatility": 0.001,
                "sma_slope_pct": 0.0,
            }
            for symbol in winner_contract.symbols
        },
    }
    decision = run_research.step_make_trade_decision(
        approved,
        rejection_reasons,
        stability,
        regime,
        window_results[-1]["best_row"],  # type: ignore[arg-type]
    )
    anchor_utc = SNAPSHOT_END.strftime("%Y-%m-%dT%H:%M:%SZ")
    resolved_inputs = run_research._resolve_research_inputs(
        targets.live_config_path
    )
    assert list(resolved_inputs.symbols) == PAPER_SYMBOLS
    result = run_research.finalize_research_bundle_and_publish(
        window_results=window_results,  # type: ignore[arg-type]
        approved=approved,
        rejection_reasons=rejection_reasons,
        stability=stability,
        regime=regime,
        final_decision=decision,
        source_identity=source_identity,
        research_config_inputs=run_research._research_config_inputs(
            SimpleNamespace(dataset=None),
            resolved_inputs=resolved_inputs,
            research_anchor_utc=anchor_utc,
            window_ranges=[
                {
                    "lookback_days": lookback_days,
                    "start_utc": (
                        SNAPSHOT_END - timedelta(days=lookback_days)
                    ).strftime("%Y-%m-%dT%H:%M:%SZ"),
                    "end_utc": anchor_utc,
                }
                for lookback_days in run_research.VALIDATION_WINDOWS
            ],
        ),
        targets=targets,
        generated_at_utc=GENERATED_AT,
    )
    assert result.published is True
    assert len(winner_contract.symbols) == 15

    pointer = load_verified_promotion_pointer(targets.pointer_path, repo_root=repo)
    latest = load_verified_latest_research_state(
        targets.latest_state_path,
        repo_root=repo,
    )
    bundle = ResearchBundle.from_json(result.bundle_path.read_text(encoding="utf-8"))
    assert latest.payload["eligible"] is True
    assert bundle.payload["strategy"]["canonical_sha256"] == winner_contract.sha256()
    assert pointer.payload["research_generation_sha256"] == (
        result.research_generation_sha256
    )

    artifacts = pointer.payload["artifacts"]
    best_path = repo / artifacts["best_config_latest"]["path"]
    live_path = repo / artifacts["live_config"]["path"]
    best = json.loads(best_path.read_text(encoding="utf-8"))
    live = json.loads(live_path.read_text(encoding="utf-8"))
    live_contract = CanonicalResolvedPromotedStrategy.from_mapping(
        live["runtime"]["symbols"],
        {
            field: live["runtime"][field]
            for field in PROMOTED_STRATEGY_FIELDS
        },
    )
    assert best["canonical_strategy_sha256"] == winner_contract.sha256()
    assert live_contract.canonical_bytes() == winner_contract.canonical_bytes()
    assert live["runtime"]["symbols"] == PAPER_SYMBOLS


def test_rank_rejects_tampered_digest(tmp_path: Path) -> None:
    evidence = _producer_evidence(entry_threshold_pct=0.0015)
    row = _backtest_row(
        evidence,
        total_return_pct=3.0,
        profit_factor=1.5,
        max_drawdown_pct=2.0,
    )
    # Keep the wrong digest string-typed through pandas CSV inference so this
    # exercises digest/content contradiction, not merely a malformed type.
    row["canonical_strategy_sha256"] = "f" * 64
    sweep = tmp_path / "sweep.csv"
    leaderboard = tmp_path / "leaderboard.csv"
    pd.DataFrame([row]).to_csv(sweep, index=False)
    with pytest.raises(ResearchBundleValidationError, match="hash"):
        run_research.step_rank_results(sweep, leaderboard)


def test_selection_evidence_rejects_sweep_changed_after_ranking(
    tmp_path: Path,
) -> None:
    evidence = _producer_evidence(entry_threshold_pct=0.0015)
    row = _backtest_row(
        evidence,
        total_return_pct=3.0,
        profit_factor=1.5,
        max_drawdown_pct=2.0,
    )
    sweep = tmp_path / "sweep.csv"
    leaderboard = tmp_path / "leaderboard.csv"
    pd.DataFrame([row]).to_csv(sweep, index=False)
    best_row = run_research.step_rank_results(sweep, leaderboard)
    selection = best_row["research_selection_evidence"]
    assert isinstance(selection, dict)

    changed = pd.read_csv(sweep)
    changed.loc[0, "total_return_pct"] = 4.0
    changed.to_csv(sweep, index=False)
    with pytest.raises(
        ResearchBundleValidationError,
        match="hashes do not match exact CSV bytes",
    ):
        validate_backtest_selection_artifacts(
            selection,
            sweep_csv_bytes=sweep.read_bytes(),
            leaderboard_csv_bytes=leaderboard.read_bytes(),
        )

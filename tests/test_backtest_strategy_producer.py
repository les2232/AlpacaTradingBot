from __future__ import annotations

import hashlib
import inspect
import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

import backtest_runner
from dataset_integrity import file_sha256, finalize_manifest
from backtest_runner import (
    DEFAULT_MAX_ADX_THRESHOLD,
    DEFAULT_MIN_ATR_PERCENTILE,
    DEFAULT_VWAP_Z_ENTRY_THRESHOLD,
    BacktestConfig,
    _canonical_strategy_evidence_from_inputs,
    _prepare_backtest_inputs,
    _result_grid_dataframe,
    build_backtest_canonical_strategy_evidence,
    extract_clean_columns,
    run_backtest,
)
from ml.model_artifact import load_model_payload, save_model_payload
from ml.predict import MODEL_PATH
from strategy import Strategy, StrategyConfig
from tradeos.strategy_contract import (
    PROMOTED_STRATEGY_FIELDS,
    RUNTIME_FIXED_STRATEGY_ASSUMPTIONS,
    CanonicalResolvedPromotedStrategy,
    ContractValidationError,
)


_RESOLUTION_SELECTORS = {
    "strategy_mode",
    "sma_bars",
    "symbol_strategy_modes",
    "bar_timeframe_minutes",
}


def _strategy_values(config: StrategyConfig | None = None) -> dict[str, object]:
    resolved = config or StrategyConfig(strategy_mode="hybrid")
    return {
        name: getattr(resolved, name)
        for name in PROMOTED_STRATEGY_FIELDS
        if name not in _RESOLUTION_SELECTORS
    }


def _runtime_fixed_values(
    config: StrategyConfig | None = None,
) -> dict[str, object]:
    resolved = config or StrategyConfig(strategy_mode="hybrid")
    return {
        name: getattr(resolved, name)
        for name, _expected in RUNTIME_FIXED_STRATEGY_ASSUMPTIONS
    }


def _evidence(
    *,
    symbols: list[str] | None = None,
    modes: dict[str, str] | None = None,
    strategy_mode: str = "hybrid",
) -> dict[str, object]:
    resolved_symbols = symbols or ["AMD", "MSFT"]
    resolved_modes = modes or {symbol: strategy_mode for symbol in resolved_symbols}
    return build_backtest_canonical_strategy_evidence(
        symbols=resolved_symbols,
        bar_timeframe_minutes=15,
        sma_bars=20,
        strategy_mode=strategy_mode,
        symbol_strategy_modes=resolved_modes,
        strategy_values=_strategy_values(),
        runtime_fixed_strategy_values=_runtime_fixed_values(),
    )


def _model_execution_evidence(*, used: bool) -> tuple[str, str]:
    if used:
        payload: dict[str, object] = {
            "schema_name": backtest_runner.MODEL_EXECUTION_EVIDENCE_SCHEMA_NAME,
            "schema_version": backtest_runner.MODEL_EXECUTION_EVIDENCE_SCHEMA_VERSION,
            "used": True,
            "model_path": "ml/models/test_model.pkl",
            "model_sha256": "1" * 64,
            "sidecar_path": "ml/models/test_model.pkl.sha256",
            "sidecar_declared_sha256": "1" * 64,
            "sidecar_sha256": "2" * 64,
            "feature_columns": ["ret_1", "ret_3"],
            "training_metadata": {
                "dataset": "datasets/training-v1",
                "trained_at": "2026-08-20T00:00:00Z",
            },
            "training_dataset_identity": {
                "dataset_reference": "datasets/training-v1",
                "dataset_id": "training-v1",
                "manifest_sha256": "3" * 64,
                "bars_sha256": "4" * 64,
                "snapshot_sha256": "5" * 64,
            },
        }
    else:
        payload = {
            "schema_name": backtest_runner.MODEL_EXECUTION_EVIDENCE_SCHEMA_NAME,
            "schema_version": backtest_runner.MODEL_EXECUTION_EVIDENCE_SCHEMA_VERSION,
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
    canonical_json = json.dumps(
        payload,
        sort_keys=True,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
    )
    return canonical_json, hashlib.sha256(canonical_json.encode("utf-8")).hexdigest()


def _serializable_result(evidence: dict[str, object]) -> dict[str, object]:
    promoted_values = evidence["promoted_values"]
    assert isinstance(promoted_values, dict)
    modes = promoted_values["symbol_strategy_modes"]
    assert isinstance(modes, dict)
    model_json, model_sha256 = _model_execution_evidence(
        used=any(mode in {"ml", "hybrid"} for mode in modes.values())
    )
    return {
        "regime": "all",
        **promoted_values,
        "symbols": evidence["symbols"],
        "runtime_fixed_strategy_values": evidence[
            "runtime_fixed_strategy_values"
        ],
        "canonical_strategy_json": evidence["canonical_strategy_json"],
        "canonical_strategy_sha256": evidence["canonical_strategy_sha256"],
        "model_execution_evidence_json": model_json,
        "model_execution_evidence_sha256": model_sha256,
        "total_return_pct": 1.25,
    }


def _write_versioned_training_dataset(repo_root: Path) -> Path:
    dataset_path = repo_root / "datasets" / "training-v1"
    dataset_path.mkdir(parents=True)
    bars_path = dataset_path / "bars.parquet"
    bars_path.write_bytes(b"versioned-training-bars")
    manifest = finalize_manifest(
        {
            "dataset_id": dataset_path.name,
            "symbols": ["AMD"],
            "timeframe": "15Min",
        },
        bars_sha256=file_sha256(bars_path),
    )
    (dataset_path / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )
    return dataset_path


def test_pure_producer_returns_complete_contract_envelope_and_digest() -> None:
    evidence = _evidence()

    assert set(evidence) == {
        "contract",
        "symbols",
        "promoted_values",
        "symbol_strategy_modes",
        "runtime_fixed_strategy_values",
        "canonical_strategy_json",
        "canonical_strategy_sha256",
    }
    contract = evidence["contract"]
    assert isinstance(contract, CanonicalResolvedPromotedStrategy)
    assert tuple(evidence["promoted_values"]) == PROMOTED_STRATEGY_FIELDS
    assert evidence["runtime_fixed_strategy_values"] == dict(
        RUNTIME_FIXED_STRATEGY_ASSUMPTIONS
    )

    canonical_json = evidence["canonical_strategy_json"]
    assert isinstance(canonical_json, str)
    assert json.loads(canonical_json) == contract.to_canonical_dict()
    assert canonical_json == contract.canonical_json()
    assert evidence["canonical_strategy_sha256"] == hashlib.sha256(
        canonical_json.encode("utf-8")
    ).hexdigest()
    assert evidence["canonical_strategy_sha256"] == contract.sha256()


def test_pure_producer_preserves_order_and_requires_full_effective_mode_map() -> None:
    symbols = ["MSFT", "AMD", "AAPL"]
    modes = {"MSFT": "sma", "AMD": "hybrid", "AAPL": "breakout"}

    evidence = _evidence(symbols=symbols, modes=modes)

    assert evidence["symbols"] == symbols
    assert evidence["symbol_strategy_modes"] == modes
    payload = json.loads(evidence["canonical_strategy_json"])
    assert payload["symbols"] == symbols
    assert payload["promoted_values"]["symbol_strategy_modes"] == modes


def test_pure_producer_handles_synthetic_fifteen_symbol_universe_without_io() -> None:
    symbols = [
        "AAPL",
        "MSFT",
        "NVDA",
        "AMZN",
        "GOOGL",
        "META",
        "TSLA",
        "AMD",
        "NFLX",
        "AVGO",
        "COST",
        "ADBE",
        "CRM",
        "QCOM",
        "INTC",
    ]
    modes = {
        symbol: ("sma" if index % 2 else "hybrid")
        for index, symbol in enumerate(symbols)
    }

    evidence = _evidence(symbols=symbols, modes=modes)

    assert evidence["symbols"] == symbols
    assert evidence["symbol_strategy_modes"] == modes
    assert len(evidence["promoted_values"]) == 65


def test_pure_producer_is_deterministic_and_does_not_mutate_inputs() -> None:
    symbols = ["AMD", "MSFT"]
    modes = {"AMD": "hybrid", "MSFT": "sma"}
    values = _strategy_values()
    fixed = _runtime_fixed_values()
    original_values = values.copy()
    original_fixed = fixed.copy()

    first = build_backtest_canonical_strategy_evidence(
        symbols=symbols,
        bar_timeframe_minutes=15,
        sma_bars=20,
        strategy_mode="hybrid",
        symbol_strategy_modes=modes,
        strategy_values=values,
        runtime_fixed_strategy_values=fixed,
    )
    second = build_backtest_canonical_strategy_evidence(
        symbols=symbols,
        bar_timeframe_minutes=15,
        sma_bars=20,
        strategy_mode="hybrid",
        symbol_strategy_modes=modes,
        strategy_values=values,
        runtime_fixed_strategy_values=fixed,
    )

    assert values == original_values
    assert fixed == original_fixed
    assert modes == {"AMD": "hybrid", "MSFT": "sma"}
    assert first["canonical_strategy_json"] == second["canonical_strategy_json"]
    assert first["canonical_strategy_sha256"] == second[
        "canonical_strategy_sha256"
    ]


def test_pure_producer_rejects_missing_promoted_value() -> None:
    values = _strategy_values()
    values.pop("mean_reversion_stop_pct")

    with pytest.raises(ContractValidationError, match="missing=.*mean_reversion_stop_pct"):
        build_backtest_canonical_strategy_evidence(
            symbols=["AMD"],
            bar_timeframe_minutes=15,
            sma_bars=20,
            strategy_mode="hybrid",
            symbol_strategy_modes={"AMD": "hybrid"},
            strategy_values=values,
            runtime_fixed_strategy_values=_runtime_fixed_values(),
        )


def test_pure_producer_rejects_extra_promoted_value() -> None:
    values = _strategy_values()
    values["unreviewed_parameter"] = 1.0

    with pytest.raises(ContractValidationError, match="unexpected=.*unreviewed_parameter"):
        build_backtest_canonical_strategy_evidence(
            symbols=["AMD"],
            bar_timeframe_minutes=15,
            sma_bars=20,
            strategy_mode="hybrid",
            symbol_strategy_modes={"AMD": "hybrid"},
            strategy_values=values,
            runtime_fixed_strategy_values=_runtime_fixed_values(),
        )


def test_pure_producer_rejects_duplicate_selector_inputs() -> None:
    values = _strategy_values()
    values["sma_bars"] = 99

    with pytest.raises(ContractValidationError, match="duplicate resolved selectors"):
        build_backtest_canonical_strategy_evidence(
            symbols=["AMD"],
            bar_timeframe_minutes=15,
            sma_bars=20,
            strategy_mode="hybrid",
            symbol_strategy_modes={"AMD": "hybrid"},
            strategy_values=values,
            runtime_fixed_strategy_values=_runtime_fixed_values(),
        )


def test_pure_producer_rejects_sparse_symbol_modes() -> None:
    with pytest.raises(ContractValidationError, match="exactly the active symbol universe"):
        _evidence(
            symbols=["AMD", "MSFT"],
            modes={"AMD": "hybrid"},
        )


def test_pure_producer_rejects_runtime_fixed_value_mismatch() -> None:
    fixed = _runtime_fixed_values()
    fixed["breakout_gap_pct_min"] = 0.01

    with pytest.raises(ContractValidationError, match="breakout_gap_pct_min"):
        build_backtest_canonical_strategy_evidence(
            symbols=["AMD"],
            bar_timeframe_minutes=15,
            sma_bars=20,
            strategy_mode="hybrid",
            symbol_strategy_modes={"AMD": "hybrid"},
            strategy_values=_strategy_values(),
            runtime_fixed_strategy_values=fixed,
        )


def test_actual_strategy_adapter_reads_resolved_modes_values_and_timeframe() -> None:
    symbols = ["MSFT", "AMD"]
    inputs = SimpleNamespace(
        symbols=symbols,
        manifest={"timeframe": "15Min"},
        strategy_mode="hybrid",
        symbol_strategies={
            "MSFT": Strategy(StrategyConfig(strategy_mode="sma")),
            "AMD": Strategy(StrategyConfig(strategy_mode="hybrid")),
        },
    )

    evidence = _canonical_strategy_evidence_from_inputs(inputs, sma_bars=21)

    assert evidence["symbols"] == symbols
    promoted = evidence["promoted_values"]
    assert promoted["bar_timeframe_minutes"] == 15
    assert promoted["sma_bars"] == 21
    assert promoted["symbol_strategy_modes"] == {
        "MSFT": "sma",
        "AMD": "hybrid",
    }


def test_non_fifteen_minute_preprocessing_and_contract_use_same_manifest_cadence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    timestamps = list(pd.date_range("2026-01-05T14:30:00Z", periods=8, freq="5min"))
    frame = pd.DataFrame(
        {
            "symbol": ["AMD"] * len(timestamps),
            "timestamp": timestamps,
            "open": [100.0 + index for index in range(len(timestamps))],
            "high": [100.5 + index for index in range(len(timestamps))],
            "low": [99.5 + index for index in range(len(timestamps))],
            "close": [100.25 + index for index in range(len(timestamps))],
            "volume": [1_000.0 + index for index in range(len(timestamps))],
        }
    )
    observed_source_minutes: list[int] = []

    def fake_hourly_regime(
        raw_timestamps: list[pd.Timestamp],
        closes: list[float],
        *,
        source_bar_minutes: int,
        sma_period: int,
    ) -> list[None]:
        assert len(raw_timestamps) == len(closes)
        assert sma_period == backtest_runner.REGIME_SMA_PERIOD
        observed_source_minutes.append(source_bar_minutes)
        return [None] * len(closes)

    monkeypatch.setattr(
        backtest_runner,
        "calculate_hourly_regime_series",
        fake_hourly_regime,
    )
    config = StrategyConfig(strategy_mode="sma")
    input_values = {
        name: getattr(config, name)
        for name in backtest_runner._BacktestInputs.__dataclass_fields__
        if hasattr(config, name)
    }
    input_values.update(
        {
            "df": frame,
            "manifest": {"timeframe": "5Min"},
            "symbols": ["AMD"],
            "symbol_strategies": {"AMD": Strategy(config)},
            "dummy_ml": SimpleNamespace(buy_threshold=0.55, sell_threshold=0.45),
            "strategy_mode": "sma",
            "trend_pullback_research_exit_fill": "next_open",
        }
    )
    inputs = backtest_runner._BacktestInputs(**input_values)

    state = backtest_runner._initialize_simulation_state(
        inputs,
        starting_capital=10_000.0,
        position_size=1_000.0,
    )
    evidence = _canonical_strategy_evidence_from_inputs(inputs, sma_bars=20)

    assert observed_source_minutes == [5]
    assert state.bullish_regime_arrs["AMD"] == [None] * len(timestamps)
    assert evidence["promoted_values"]["bar_timeframe_minutes"] == 5


def test_actual_strategy_adapter_rejects_non_live_fixed_override() -> None:
    inputs = SimpleNamespace(
        symbols=["AMD"],
        manifest={"timeframe": "15Min"},
        strategy_mode="hybrid",
        symbol_strategies={
            "AMD": Strategy(
                StrategyConfig(
                    strategy_mode="hybrid",
                    breakout_gap_pct_min=0.01,
                )
            )
        },
    )

    with pytest.raises(ContractValidationError, match="breakout_gap_pct_min"):
        _canonical_strategy_evidence_from_inputs(inputs, sma_bars=20)


def test_actual_strategy_adapter_rejects_cross_symbol_parameter_disagreement() -> None:
    inputs = SimpleNamespace(
        symbols=["AMD", "MSFT"],
        manifest={"timeframe": "15Min"},
        strategy_mode="hybrid",
        symbol_strategies={
            "AMD": Strategy(StrategyConfig(strategy_mode="hybrid")),
            "MSFT": Strategy(
                StrategyConfig(strategy_mode="sma", mean_reversion_stop_pct=0.02)
            ),
        },
    )

    with pytest.raises(
        ContractValidationError,
        match="disagree.*mean_reversion_stop_pct",
    ):
        _canonical_strategy_evidence_from_inputs(inputs, sma_bars=20)


def test_ml_loader_binds_exact_deserialized_model_and_training_dataset_bytes(
    tmp_path: Path,
) -> None:
    dataset_path = _write_versioned_training_dataset(tmp_path)
    source_payload = load_model_payload(MODEL_PATH)
    model_payload = dict(source_payload)
    model_payload["metadata"] = {
        **source_payload["metadata"],
        "dataset": str(dataset_path),
    }
    model_path = tmp_path / "ml" / "models" / "test_model.pkl"
    save_model_payload(model_payload, model_path)

    loaded = backtest_runner._load_offline_model_payload(
        model_path,
        repo_root=tmp_path,
    )
    evidence_json = loaded.execution_evidence.canonical_json
    evidence = json.loads(evidence_json)
    sidecar_path = model_path.with_suffix(".pkl.sha256")

    assert loaded.payload["feature_cols"] == source_payload["feature_cols"]
    assert evidence["used"] is True
    assert evidence["model_path"] == "ml/models/test_model.pkl"
    assert evidence["model_sha256"] == hashlib.sha256(
        model_path.read_bytes()
    ).hexdigest()
    assert evidence["sidecar_path"] == "ml/models/test_model.pkl.sha256"
    assert evidence["sidecar_declared_sha256"] == evidence["model_sha256"]
    assert evidence["sidecar_sha256"] == hashlib.sha256(
        sidecar_path.read_bytes()
    ).hexdigest()
    assert evidence["feature_columns"] == source_payload["feature_cols"]
    assert evidence["training_metadata"]["dataset"] == "datasets/training-v1"
    assert evidence["training_dataset_identity"] == {
        "dataset_reference": "datasets/training-v1",
        "dataset_id": "training-v1",
        "manifest_sha256": hashlib.sha256(
            (dataset_path / "manifest.json").read_bytes()
        ).hexdigest(),
        "bars_sha256": file_sha256(dataset_path / "bars.parquet"),
        "snapshot_sha256": json.loads(
            (dataset_path / "manifest.json").read_text(encoding="utf-8")
        )["snapshot_sha256"],
    }
    assert evidence_json == json.dumps(
        evidence,
        sort_keys=True,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
    )
    assert loaded.execution_evidence.sha256 == hashlib.sha256(
        evidence_json.encode("utf-8")
    ).hexdigest()


def test_ml_loader_rejects_legacy_training_dataset_without_declared_hashes(
    tmp_path: Path,
) -> None:
    dataset_path = tmp_path / "datasets" / "legacy-training"
    dataset_path.mkdir(parents=True)
    (dataset_path / "bars.parquet").write_bytes(b"legacy-bars")
    (dataset_path / "manifest.json").write_text(
        json.dumps({"dataset_id": dataset_path.name}),
        encoding="utf-8",
    )
    source_payload = load_model_payload(MODEL_PATH)
    model_payload = dict(source_payload)
    model_payload["metadata"] = {
        **source_payload["metadata"],
        "dataset": str(dataset_path),
    }
    model_path = tmp_path / "ml" / "models" / "legacy_model.pkl"
    save_model_payload(model_payload, model_path)

    with pytest.raises(RuntimeError, match="bars_sha256"):
        backtest_runner._load_offline_model_payload(
            model_path,
            repo_root=tmp_path,
        )


def test_ml_loader_requires_hash_sidecar_for_promotable_execution(
    tmp_path: Path,
) -> None:
    dataset_path = _write_versioned_training_dataset(tmp_path)
    source_payload = load_model_payload(MODEL_PATH)
    model_payload = dict(source_payload)
    model_payload["metadata"] = {
        **source_payload["metadata"],
        "dataset": str(dataset_path),
    }
    model_path = tmp_path / "ml" / "models" / "sidecarless.pkl"
    save_model_payload(model_payload, model_path)
    model_path.with_suffix(".pkl.sha256").unlink()

    with pytest.raises(RuntimeError, match="require a verified model SHA-256 sidecar"):
        backtest_runner._load_offline_model_payload(
            model_path,
            repo_root=tmp_path,
        )


def test_mean_reversion_projection_records_model_explicitly_unused() -> None:
    evidence = _evidence(
        strategy_mode="mean_reversion",
        modes={"AMD": "mean_reversion", "MSFT": "mean_reversion"},
    )
    cleaned = extract_clean_columns(_serializable_result(evidence))
    payload = json.loads(cleaned["model_execution_evidence_json"])

    assert payload == {
        "schema_name": backtest_runner.MODEL_EXECUTION_EVIDENCE_SCHEMA_NAME,
        "schema_version": backtest_runner.MODEL_EXECUTION_EVIDENCE_SCHEMA_VERSION,
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
    assert cleaned["model_execution_evidence_sha256"] == hashlib.sha256(
        cleaned["model_execution_evidence_json"].encode("utf-8")
    ).hexdigest()


def test_clean_column_projection_retains_all_canonical_evidence() -> None:
    evidence = _evidence()
    cleaned = extract_clean_columns(_serializable_result(evidence))

    assert tuple(name for name in PROMOTED_STRATEGY_FIELDS if name in cleaned) == (
        PROMOTED_STRATEGY_FIELDS
    )
    assert cleaned["symbols"] == ["AMD", "MSFT"]
    assert cleaned["symbol_strategy_modes"] == {
        "AMD": "hybrid",
        "MSFT": "hybrid",
    }
    assert cleaned["runtime_fixed_strategy_values"] == dict(
        RUNTIME_FIXED_STRATEGY_ASSUMPTIONS
    )
    assert cleaned["canonical_strategy_json"] == evidence[
        "canonical_strategy_json"
    ]
    assert cleaned["canonical_strategy_sha256"] == evidence[
        "canonical_strategy_sha256"
    ]
    assert json.loads(cleaned["model_execution_evidence_json"])["used"] is True
    assert cleaned["model_execution_evidence_sha256"] == hashlib.sha256(
        cleaned["model_execution_evidence_json"].encode("utf-8")
    ).hexdigest()


def test_clean_column_projection_keeps_three_historically_omitted_fields() -> None:
    evidence = _evidence()
    cleaned = extract_clean_columns(_serializable_result(evidence))

    assert cleaned["mean_reversion_trend_filter"] is False
    assert cleaned["mean_reversion_trend_slope_filter"] is False
    assert cleaned["mean_reversion_stop_pct"] == 0.0


def test_clean_column_projection_rejects_missing_canonical_evidence() -> None:
    result = _serializable_result(_evidence())
    result.pop("canonical_strategy_json")

    with pytest.raises(ContractValidationError, match="missing canonical strategy"):
        extract_clean_columns(result)


def test_clean_column_projection_rejects_digest_mismatch() -> None:
    result = _serializable_result(_evidence())
    result["canonical_strategy_sha256"] = "0" * 64

    with pytest.raises(ContractValidationError, match="does not match"):
        extract_clean_columns(result)


def test_clean_column_projection_rejects_model_evidence_digest_mismatch() -> None:
    result = _serializable_result(_evidence())
    result["model_execution_evidence_sha256"] = "0" * 64

    with pytest.raises(ContractValidationError, match="does not match its exact JSON"):
        extract_clean_columns(result)


def test_clean_column_projection_rejects_unused_model_for_hybrid_modes() -> None:
    result = _serializable_result(_evidence())
    model_json, model_sha256 = _model_execution_evidence(used=False)
    result["model_execution_evidence_json"] = model_json
    result["model_execution_evidence_sha256"] = model_sha256

    with pytest.raises(ContractValidationError, match="contradicts the resolved symbol modes"):
        extract_clean_columns(result)


def test_clean_column_projection_rejects_windows_absolute_model_path() -> None:
    result = _serializable_result(_evidence())
    payload = json.loads(result["model_execution_evidence_json"])
    payload["model_path"] = "C:/outside/test_model.pkl"
    payload["sidecar_path"] = "C:/outside/test_model.pkl.sha256"
    canonical_json = json.dumps(
        payload,
        sort_keys=True,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
    )
    result["model_execution_evidence_json"] = canonical_json
    result["model_execution_evidence_sha256"] = hashlib.sha256(
        canonical_json.encode("utf-8")
    ).hexdigest()

    with pytest.raises(ContractValidationError, match="safe repo-relative path"):
        extract_clean_columns(result)


def test_clean_column_projection_rejects_flat_field_mismatch() -> None:
    result = _serializable_result(_evidence())
    result["mean_reversion_stop_pct"] = 0.02

    with pytest.raises(
        ContractValidationError,
        match="mean_reversion_stop_pct.*does not match",
    ):
        extract_clean_columns(result)


def test_clean_column_projection_rejects_symbol_order_mismatch() -> None:
    result = _serializable_result(_evidence())
    result["symbols"] = ["MSFT", "AMD"]

    with pytest.raises(ContractValidationError, match="ordered universe"):
        extract_clean_columns(result)


def test_result_grid_retains_lossless_strategy_columns_for_every_row() -> None:
    first = _serializable_result(_evidence())
    second_evidence = _evidence(
        symbols=["MSFT", "AMD"],
        modes={"MSFT": "sma", "AMD": "hybrid"},
    )
    second = _serializable_result(second_evidence)

    grid = _result_grid_dataframe([first, second])

    assert len(grid) == 2
    assert set(PROMOTED_STRATEGY_FIELDS).issubset(grid.columns)
    assert {
        "symbols",
        "runtime_fixed_strategy_values",
        "canonical_strategy_json",
        "canonical_strategy_sha256",
        "model_execution_evidence_json",
        "model_execution_evidence_sha256",
    }.issubset(grid.columns)
    assert grid.loc[1, "symbols"] == ["MSFT", "AMD"]
    assert grid.loc[1, "canonical_strategy_json"] == second_evidence[
        "canonical_strategy_json"
    ]
    assert grid.loc[1, "canonical_strategy_sha256"] == second_evidence[
        "canonical_strategy_sha256"
    ]


def test_backtest_defaults_match_authoritative_live_filter_defaults() -> None:
    # These defaults were historically 0.0 (disabled) in research.  They now
    # match the live StrategyConfig/BotConfig resolution exactly.
    expected = {
        "vwap_z_entry_threshold": 1.5,
        "min_atr_percentile": 20.0,
        "max_adx_threshold": 25.0,
    }
    assert {
        "vwap_z_entry_threshold": DEFAULT_VWAP_Z_ENTRY_THRESHOLD,
        "min_atr_percentile": DEFAULT_MIN_ATR_PERCENTILE,
        "max_adx_threshold": DEFAULT_MAX_ADX_THRESHOLD,
    } == expected

    config = BacktestConfig(dataset_path=Path("unused.csv"))
    run_signature = inspect.signature(run_backtest)
    prepare_signature = inspect.signature(_prepare_backtest_inputs)
    strategy = StrategyConfig(strategy_mode="hybrid")
    for name, value in expected.items():
        assert getattr(config, name) == value
        assert getattr(strategy, name) == value
        assert run_signature.parameters[name].default == value
        assert prepare_signature.parameters[name].default == value

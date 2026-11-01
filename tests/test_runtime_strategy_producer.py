import json
from pathlib import Path

import pytest

from alpaca_trading_bot import cli
from tradeos.strategy_contract import (
    CanonicalResolvedPromotedStrategy,
    ContractValidationError,
)
import trading_bot


def _config(*, symbols: list[str] | None = None) -> trading_bot.BotConfig:
    return trading_bot.BotConfig(
        symbols=list(symbols or ["MSFT", "AMD"]),
        max_usd_per_trade=200.0,
        max_symbol_exposure_usd=400.0,
        max_open_positions=3,
        max_daily_loss_usd=300.0,
        sma_bars=20,
        bar_timeframe_minutes=15,
        strategy_mode="mean_reversion",
        symbol_strategy_modes={"AMD": "sma"},
    )


def _details(config: trading_bot.BotConfig) -> trading_bot.RuntimeConfigDetails:
    return trading_bot.RuntimeConfigDetails(
        config=config,
        runtime_config_path="C:/private/runtime-profile.json",
        overridden_fields=("symbols", "strategy_mode"),
        runtime_config_approved=True,
        runtime_config_rejection_reasons=(),
        runtime_config_sha256="e" * 64,
        baseline_valid_for_comparison=True,
        baseline_validation_errors=(),
        execution_context_matches_pointer_live=True,
        execution_context_validation_errors=(),
        effective_runtime_binding_valid=True,
        effective_runtime_binding_errors=(),
        research_generation_sha256="b" * 64,
        promotion_bundle_sha256="c" * 64,
        research_source_content_sha256="d" * 64,
        launch_profile="paper_shadow",
        broker_environment="paper",
        submission_policy="deny",
        expected_account_fingerprint="a" * 64,
        dashboard_mode="read_only",
    )


def test_runtime_canonical_strategy_evidence_is_pure_complete_and_round_trips() -> None:
    config = _config()

    evidence = trading_bot.build_runtime_canonical_strategy_evidence(config)

    assert set(evidence) == {
        "contract",
        "canonical_strategy_contract",
        "canonical_strategy_json",
        "canonical_strategy_sha256",
        "symbols",
        "symbol_strategy_modes",
    }
    contract = evidence["contract"]
    assert isinstance(contract, CanonicalResolvedPromotedStrategy)
    assert evidence["symbols"] == ["MSFT", "AMD"]
    assert evidence["symbol_strategy_modes"] == {
        "MSFT": "mean_reversion",
        "AMD": "sma",
    }
    assert CanonicalResolvedPromotedStrategy.from_payload(
        evidence["canonical_strategy_contract"]  # type: ignore[arg-type]
    ).sha256() == evidence["canonical_strategy_sha256"]
    assert CanonicalResolvedPromotedStrategy.from_json(
        evidence["canonical_strategy_json"]  # type: ignore[arg-type]
    ).sha256() == evidence["canonical_strategy_sha256"]
    assert "runtime-profile" not in str(evidence["canonical_strategy_json"])


def test_startup_writer_persists_only_validated_strategy_identity_and_is_immutable(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    config = _config()
    evidence = trading_bot.build_runtime_canonical_strategy_evidence(config)
    monkeypatch.setattr(cli, "LOG_ROOT", tmp_path / "logs")
    monkeypatch.setattr(cli, "_utcnow_iso", lambda: "2026-08-20T12:34:56+00:00")

    artifact_path = cli._persist_startup_config(
        _details(config),
        preview=True,
        session_id="paper-shadow-session-test",
        strategy_evidence=evidence,
        database_path=tmp_path / "authority.db",
        launch_mode="paper-shadow",
        execution_enabled=False,
    )
    source_bytes = artifact_path.read_bytes()
    payload = json.loads(source_bytes.decode("utf-8"))

    assert payload["artifact_schema_name"] == cli.STARTUP_CONFIG_SCHEMA_NAME
    assert payload["artifact_schema_version"] == cli.STARTUP_CONFIG_SCHEMA_VERSION
    assert payload["canonical_strategy_contract"] == evidence["canonical_strategy_contract"]
    assert payload["canonical_strategy_sha256"] == evidence["canonical_strategy_sha256"]
    assert payload["runtime_config_path"] == "C:/private/runtime-profile.json"
    assert payload["runtime_config_sha256"] == "e" * 64
    assert payload["research_generation_sha256"] == "b" * 64
    assert payload["promotion_bundle_sha256"] == "c" * 64
    assert payload["research_source_content_sha256"] == "d" * 64
    assert payload["execution_context_matches_pointer_live"] is True
    assert payload["execution_context_validation_errors"] == []
    assert payload["effective_runtime_binding_valid"] is True
    assert payload["effective_runtime_binding_errors"] == []
    assert "private" not in evidence["canonical_strategy_json"]  # type: ignore[operator]
    assert json.loads((artifact_path.parent / "startup_config.json").read_text("utf-8")) == payload

    with pytest.raises(FileExistsError):
        cli._persist_startup_config(
            _details(config),
            preview=False,
            session_id="different-session",
            strategy_evidence=evidence,
            database_path=tmp_path / "different-authority.db",
        )
    assert artifact_path.read_bytes() == source_bytes


def test_startup_writer_rejects_evidence_from_a_different_effective_config(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    config = _config()
    evidence = trading_bot.build_runtime_canonical_strategy_evidence(config)
    config.symbols.append("QCOM")
    monkeypatch.setattr(cli, "LOG_ROOT", tmp_path / "logs")

    with pytest.raises(ContractValidationError, match="does not match"):
        cli._persist_startup_config(
            _details(config),
            preview=True,
            session_id="mismatched-config",
            strategy_evidence=evidence,
            database_path=tmp_path / "authority.db",
        )
    assert not (tmp_path / "logs").exists()


def test_bot_rejects_invalid_contract_before_broker_or_storage_construction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config()
    object.__setattr__(config, "sma_bars", 0)
    calls: list[str] = []
    monkeypatch.setattr(
        trading_bot.TradeOSBot,
        "_build_broker",
        lambda self: calls.append("broker"),
    )
    monkeypatch.setattr(
        trading_bot,
        "BotStorage",
        lambda *args, **kwargs: calls.append("storage"),
    )
    monkeypatch.setattr(trading_bot, "load_dotenv", lambda *args, **kwargs: None)

    with pytest.raises(ContractValidationError, match="sma_bars"):
        trading_bot.TradeOSBot(config)

    assert calls == []


def test_resync_finalization_rejects_cross_session_startup_artifact(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    config = _config()
    evidence = trading_bot.build_runtime_canonical_strategy_evidence(config)
    monkeypatch.setattr(cli, "LOG_ROOT", tmp_path / "logs")
    artifact_path = cli._persist_startup_config(
        _details(config),
        preview=True,
        session_id="session-a",
        strategy_evidence=evidence,
        database_path=tmp_path / "authority.db",
    )
    source_bytes = artifact_path.read_bytes()
    alias_bytes = (artifact_path.parent / "startup_config.json").read_bytes()
    bot = trading_bot.TradeOSBot.__new__(trading_bot.TradeOSBot)
    bot.session_id = "session-b"
    result = trading_bot.ResyncResult(
        status=trading_bot.ResyncStatus.DEGRADED,
        started_at_utc="2026-08-20T12:00:00+00:00",
        completed_at_utc="2026-08-20T12:00:01+00:00",
        reason_codes=("TEST",),
    )
    monkeypatch.setenv(trading_bot.STARTUP_ARTIFACT_PATH_ENV, str(artifact_path))

    assert bot._update_startup_artifact_with_resync(result) is None
    assert artifact_path.read_bytes() == source_bytes
    assert (artifact_path.parent / "startup_config.json").read_bytes() == alias_bytes
    assert list(artifact_path.parent.glob("startup_resync.*.json")) == []

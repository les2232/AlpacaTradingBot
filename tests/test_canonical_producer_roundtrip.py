from __future__ import annotations

import hashlib
import json

import pytest

from backtest_runner import build_backtest_canonical_strategy_evidence
from tradeos.strategy_contract import (
    PROMOTED_STRATEGY_FIELDS,
    RUNTIME_FIXED_STRATEGY_ASSUMPTIONS,
    CanonicalResolvedPromotedStrategy,
    ContractValidationError,
)
from trading_bot import BotConfig, build_runtime_canonical_strategy_evidence


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

_RESEARCH_SELECTORS = {
    "strategy_mode",
    "sma_bars",
    "symbol_strategy_modes",
    "bar_timeframe_minutes",
}


def _runtime_config() -> BotConfig:
    return BotConfig(
        symbols=list(PAPER_SYMBOLS),
        max_usd_per_trade=100.0,
        max_symbol_exposure_usd=200.0,
        max_open_positions=3,
        max_daily_loss_usd=50.0,
        sma_bars=15,
        bar_timeframe_minutes=15,
        strategy_mode="mean_reversion",
    )


def _research_evidence_from_runtime(
    runtime_evidence: dict[str, object],
    *,
    symbols: list[str] | None = None,
    promoted_overrides: dict[str, object] | None = None,
    fixed_overrides: dict[str, object] | None = None,
) -> dict[str, object]:
    contract = runtime_evidence["contract"]
    assert isinstance(contract, CanonicalResolvedPromotedStrategy)
    promoted = contract.promoted_values
    if promoted_overrides:
        promoted.update(promoted_overrides)
    active_symbols = list(symbols or contract.symbols)
    modes = dict(promoted.pop("symbol_strategy_modes"))
    if active_symbols != list(contract.symbols):
        base_mode = promoted["strategy_mode"]
        modes = {symbol: modes.get(symbol, base_mode) for symbol in active_symbols}
    strategy_mode = promoted.pop("strategy_mode")
    sma_bars = promoted.pop("sma_bars")
    timeframe = promoted.pop("bar_timeframe_minutes")
    assert set(promoted) == set(PROMOTED_STRATEGY_FIELDS) - _RESEARCH_SELECTORS

    fixed_values = dict(RUNTIME_FIXED_STRATEGY_ASSUMPTIONS)
    if fixed_overrides:
        fixed_values.update(fixed_overrides)
    return build_backtest_canonical_strategy_evidence(
        symbols=active_symbols,
        bar_timeframe_minutes=timeframe,
        sma_bars=sma_bars,
        strategy_mode=strategy_mode,
        symbol_strategy_modes=modes,
        strategy_values=promoted,
        runtime_fixed_strategy_values=fixed_values,
    )


def test_runtime_and_research_producers_have_identical_canonical_identity() -> None:
    runtime = build_runtime_canonical_strategy_evidence(_runtime_config())
    research = _research_evidence_from_runtime(runtime)

    runtime_contract = runtime["contract"]
    research_contract = research["contract"]
    assert isinstance(runtime_contract, CanonicalResolvedPromotedStrategy)
    assert isinstance(research_contract, CanonicalResolvedPromotedStrategy)
    assert runtime_contract.schema_name == research_contract.schema_name
    assert runtime_contract.schema_version == research_contract.schema_version
    assert runtime_contract.promoted_values == research_contract.promoted_values
    assert len(runtime_contract.promoted_values) == 65
    assert list(runtime_contract.symbols) == PAPER_SYMBOLS
    assert runtime_contract.symbols == research_contract.symbols
    assert runtime_contract.promoted_values["symbol_strategy_modes"] == {
        symbol: "mean_reversion" for symbol in PAPER_SYMBOLS
    }
    runtime_fixed = runtime_contract.to_canonical_dict()[
        "runtime_fixed_strategy_assumptions"
    ]
    research_fixed = research_contract.to_canonical_dict()[
        "runtime_fixed_strategy_assumptions"
    ]
    assert runtime_fixed == research_fixed
    assert len(runtime_fixed["values"]) == 13
    assert runtime["canonical_strategy_contract"] == runtime_contract.to_canonical_dict()
    assert runtime["canonical_strategy_json"] == research["canonical_strategy_json"]
    assert runtime["canonical_strategy_sha256"] == research["canonical_strategy_sha256"]
    assert runtime["canonical_strategy_sha256"] == runtime_contract.sha256()


def test_producer_identity_changes_with_promoted_field_or_symbol() -> None:
    runtime = build_runtime_canonical_strategy_evidence(_runtime_config())
    original_hash = runtime["canonical_strategy_sha256"]

    changed_field = _research_evidence_from_runtime(
        runtime,
        promoted_overrides={"entry_threshold_pct": 0.002},
    )
    changed_symbols = _research_evidence_from_runtime(
        runtime,
        symbols=[*PAPER_SYMBOLS[:-1], "NVDA"],
    )

    assert changed_field["canonical_strategy_sha256"] != original_hash
    assert changed_symbols["canonical_strategy_sha256"] != original_hash


def test_research_fixed_assumption_change_is_hash_visible_and_rejected() -> None:
    runtime = build_runtime_canonical_strategy_evidence(_runtime_config())
    canonical_payload = runtime["canonical_strategy_contract"]
    assert isinstance(canonical_payload, dict)
    changed_payload = json.loads(json.dumps(canonical_payload))
    changed_payload["runtime_fixed_strategy_assumptions"]["values"][
        "orb_target_multiple"
    ] = 1.25
    changed_json = json.dumps(
        changed_payload,
        sort_keys=True,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
    )
    changed_hash = hashlib.sha256(changed_json.encode("utf-8")).hexdigest()

    assert changed_hash != runtime["canonical_strategy_sha256"]
    with pytest.raises(ContractValidationError, match="runtime-fixed strategy"):
        CanonicalResolvedPromotedStrategy.from_payload(changed_payload)
    with pytest.raises(ContractValidationError, match="runtime-fixed strategy"):
        _research_evidence_from_runtime(
            runtime,
            fixed_overrides={"orb_target_multiple": 1.25},
        )


def test_runtime_canonical_evidence_is_environment_independent_after_resolution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _runtime_config()
    before = build_runtime_canonical_strategy_evidence(config)
    monkeypatch.setenv("STRATEGY_MODE", "breakout")
    monkeypatch.setenv("SYMBOLS", "SPY")
    monkeypatch.setenv("BAR_TIMEFRAME_MINUTES", "1")
    monkeypatch.setenv("VWAP_Z_ENTRY_THRESHOLD", "999")
    after = build_runtime_canonical_strategy_evidence(config)

    assert after["canonical_strategy_json"] == before["canonical_strategy_json"]
    assert after["canonical_strategy_sha256"] == before["canonical_strategy_sha256"]

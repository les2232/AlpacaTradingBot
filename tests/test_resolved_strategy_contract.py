from __future__ import annotations

import hashlib
import json
from dataclasses import FrozenInstanceError

import pytest

from tradeos.strategy_contract import (
    SCHEMA_NAME,
    SCHEMA_VERSION,
    CanonicalResolvedPromotedStrategy,
    ContractValidationError,
    PROMOTED_STRATEGY_ENUM_CHOICES,
    PROMOTED_STRATEGY_FIELDS,
    RUNTIME_FIXED_ASSUMPTIONS_SCHEMA_NAME,
    RUNTIME_FIXED_ASSUMPTIONS_SCHEMA_VERSION,
    RUNTIME_FIXED_STRATEGY_ASSUMPTIONS,
    assert_source_models_match_contract,
    source_model_promoted_fields,
)
from trading_bot import BotConfig


# Independent schema-v1 golden.  This must not be generated from the
# production constant: a source-model change requires an intentional contract
# review and schema-version decision.
GOLDEN_PROMOTED_STRATEGY_FIELDS = (
    "strategy_mode",
    "sma_bars",
    "symbol_strategy_modes",
    "bar_timeframe_minutes",
    "ml_probability_buy",
    "ml_probability_sell",
    "entry_threshold_pct",
    "threshold_mode",
    "atr_multiple",
    "atr_percentile_threshold",
    "time_window_mode",
    "regime_filter_enabled",
    "orb_filter_mode",
    "breakout_exit_style",
    "breakout_tight_stop_fraction",
    "breakout_max_stop_pct",
    "mean_reversion_exit_style",
    "mean_reversion_max_atr_percentile",
    "mean_reversion_stop_pct",
    "sma_stop_pct",
    "mean_reversion_trend_filter",
    "mean_reversion_trend_slope_filter",
    "vwap_z_entry_threshold",
    "vwap_z_stop_atr_multiple",
    "min_atr_percentile",
    "max_adx_threshold",
    "trend_pullback_min_adx",
    "trend_pullback_min_slope",
    "trend_pullback_entry_threshold",
    "trend_pullback_min_atr_percentile",
    "trend_pullback_max_atr_percentile",
    "trend_pullback_exit_style",
    "trend_pullback_hold_bars",
    "trend_pullback_take_profit_pct",
    "trend_pullback_stop_pct",
    "momentum_breakout_lookback_bars",
    "momentum_breakout_entry_buffer_pct",
    "momentum_breakout_min_adx",
    "momentum_breakout_min_slope",
    "momentum_breakout_min_atr_percentile",
    "momentum_breakout_exit_style",
    "momentum_breakout_hold_bars",
    "momentum_breakout_stop_pct",
    "momentum_breakout_take_profit_pct",
    "volatility_expansion_lookback_bars",
    "volatility_expansion_entry_buffer_pct",
    "volatility_expansion_max_atr_percentile",
    "volatility_expansion_trend_filter",
    "volatility_expansion_min_slope",
    "volatility_expansion_use_volume_confirm",
    "volatility_expansion_exit_style",
    "volatility_expansion_hold_bars",
    "volatility_expansion_stop_pct",
    "volatility_expansion_take_profit_pct",
    "bb_period",
    "bb_stddev_mult",
    "bb_width_lookback",
    "bb_squeeze_quantile",
    "bb_slope_lookback",
    "bb_use_volume_confirm",
    "bb_volume_mult",
    "bb_breakout_buffer_pct",
    "bb_min_mid_slope",
    "bb_trend_filter",
    "bb_exit_mode",
)

GOLDEN_PROMOTED_STRATEGY_ENUM_CHOICES = (
    (
        "strategy_mode",
        (
            "sma",
            "ml",
            "hybrid",
            "breakout",
            "mean_reversion",
            "trend_pullback",
            "momentum_breakout",
            "volatility_expansion",
            "orb",
            "wick_fade",
            "bollinger_squeeze",
            "hybrid_bb_mr",
        ),
    ),
    ("threshold_mode", ("static_pct", "atr_multiple")),
    (
        "time_window_mode",
        (
            "full_day",
            "morning_only",
            "afternoon_only",
            "combined_windows",
        ),
    ),
    ("orb_filter_mode", ("none", "volume_or_volatility")),
    (
        "breakout_exit_style",
        (
            "target_1x_stop_low",
            "target_1_5x_stop_low",
            "eod_only",
            "trailing_stop_half_range",
            "trailing_stop_full_range",
            "target_1x_tight_stop",
        ),
    ),
    ("mean_reversion_exit_style", ("sma", "midpoint", "eod")),
    (
        "trend_pullback_exit_style",
        ("fixed_bars", "take_profit", "hybrid_tp_or_time"),
    ),
    ("momentum_breakout_exit_style", ("fixed_bars",)),
    ("volatility_expansion_exit_style", ("fixed_bars",)),
    ("bb_exit_mode", ("middle_band",)),
)

GOLDEN_RUNTIME_FIXED_STRATEGY_ASSUMPTIONS = (
    ("orb_target_multiple", 1.0),
    ("orb_entry_buffer_pct", 0.0015),
    ("orb_min_or_size_pct", 0.003),
    ("orb_max_or_size_pct", 0.03),
    ("orb_hard_stop_pct", 0.015),
    ("breakout_gap_pct_min", 0.0),
    ("breakout_or_range_pct_min", 0.0),
    ("wick_fade_min_lower_wick_ratio", 0.4),
    ("wick_fade_min_close_position", 0.5),
    ("wick_fade_min_range_pct", 0.003),
    ("wick_fade_stop_atr_multiple", 1.5),
    ("wick_fade_target_atr_multiple", 1.0),
    ("wick_fade_max_hold_bars", 4),
)


def _bot_config(**overrides: object) -> BotConfig:
    values: dict[str, object] = {
        "symbols": ["AMD", "MSFT"],
        "max_usd_per_trade": 1_000.0,
        "max_symbol_exposure_usd": 1_000.0,
        "max_open_positions": 2,
        "max_daily_loss_usd": 500.0,
        "sma_bars": 20,
        "bar_timeframe_minutes": 15,
        "strategy_mode": "hybrid",
    }
    values.update(overrides)
    return BotConfig(**values)  # type: ignore[arg-type]


def _contract(**config_overrides: object) -> CanonicalResolvedPromotedStrategy:
    return CanonicalResolvedPromotedStrategy.from_bot_config(
        _bot_config(**config_overrides)
    )


def _promoted_values() -> dict[str, object]:
    return _contract().promoted_values


def _with_promoted_value(name: str, value: object) -> dict[str, object]:
    promoted = _promoted_values()
    promoted[name] = value
    return promoted


def test_schema_v1_matches_the_independent_65_field_golden() -> None:
    assert SCHEMA_NAME == "tradeos.canonical_resolved_promoted_strategy"
    assert SCHEMA_VERSION == 1
    assert len(GOLDEN_PROMOTED_STRATEGY_FIELDS) == 65
    assert PROMOTED_STRATEGY_FIELDS == GOLDEN_PROMOTED_STRATEGY_FIELDS
    assert source_model_promoted_fields() == GOLDEN_PROMOTED_STRATEGY_FIELDS
    assert assert_source_models_match_contract() is None


def test_schema_v1_enums_and_runtime_fixed_assumptions_match_independent_goldens() -> None:
    assert len(PROMOTED_STRATEGY_ENUM_CHOICES) == len(
        GOLDEN_PROMOTED_STRATEGY_ENUM_CHOICES
    )
    for actual_enum, expected_enum in zip(
        PROMOTED_STRATEGY_ENUM_CHOICES,
        GOLDEN_PROMOTED_STRATEGY_ENUM_CHOICES,
        strict=True,
    ):
        assert actual_enum[0] == expected_enum[0]
        assert len(actual_enum[1]) == len(expected_enum[1])
        for actual_token, expected_token in zip(
            actual_enum[1], expected_enum[1], strict=True
        ):
            assert type(actual_token) is str
            assert actual_token == expected_token
    assert RUNTIME_FIXED_ASSUMPTIONS_SCHEMA_NAME == (
        "tradeos.runtime_fixed_strategy_assumptions"
    )
    assert RUNTIME_FIXED_ASSUMPTIONS_SCHEMA_VERSION == 1
    assert len(RUNTIME_FIXED_STRATEGY_ASSUMPTIONS) == len(
        GOLDEN_RUNTIME_FIXED_STRATEGY_ASSUMPTIONS
    )
    for actual, expected in zip(
        RUNTIME_FIXED_STRATEGY_ASSUMPTIONS,
        GOLDEN_RUNTIME_FIXED_STRATEGY_ASSUMPTIONS,
        strict=True,
    ):
        assert actual[0] == expected[0]
        assert type(actual[1]) is type(expected[1])
        assert actual[1] == expected[1]


def test_schema_includes_strategy_families_and_excludes_runtime_infrastructure() -> None:
    assert {
        "breakout_max_stop_pct",
        "sma_stop_pct",
        "vwap_z_entry_threshold",
        "trend_pullback_entry_threshold",
        "momentum_breakout_entry_buffer_pct",
        "volatility_expansion_entry_buffer_pct",
        "bb_period",
        "bb_exit_mode",
        "symbol_strategy_modes",
        "bar_timeframe_minutes",
    } <= set(GOLDEN_PROMOTED_STRATEGY_FIELDS)
    assert {
        "symbols",
        "excluded_symbols",
        "ml_lookback_bars",
        "max_usd_per_trade",
        "max_daily_loss_usd",
        "historical_feed",
        "broker_backend",
        "max_orders_per_minute",
    }.isdisjoint(GOLDEN_PROMOTED_STRATEGY_FIELDS)


def test_from_bot_config_emits_exact_envelope_and_full_effective_mode_map() -> None:
    contract = _contract(symbol_strategy_modes={"MSFT": "sma"})
    payload = contract.to_canonical_dict()

    assert tuple(payload) == (
        "schema_name",
        "schema_version",
        "runtime_fixed_strategy_assumptions",
        "symbols",
        "promoted_values",
    )
    assert payload["schema_name"] == SCHEMA_NAME
    assert payload["schema_version"] == SCHEMA_VERSION
    assumptions = payload["runtime_fixed_strategy_assumptions"]
    assert tuple(assumptions) == ("schema_name", "schema_version", "values")
    assert assumptions["schema_name"] == (
        "tradeos.runtime_fixed_strategy_assumptions"
    )
    assert assumptions["schema_version"] == 1
    assert assumptions["values"] == dict(RUNTIME_FIXED_STRATEGY_ASSUMPTIONS)
    assert payload["symbols"] == ["AMD", "MSFT"]
    assert tuple(payload["promoted_values"]) == GOLDEN_PROMOTED_STRATEGY_FIELDS
    assert payload["promoted_values"]["symbol_strategy_modes"] == {
        "AMD": "hybrid",
        "MSFT": "sma",
    }


def test_sparse_and_explicit_default_completion_have_one_identity() -> None:
    implicit = _contract(symbol_strategy_modes=None)
    sparse = _contract(symbol_strategy_modes={"MSFT": "sma"})
    explicit = _contract(
        symbol_strategy_modes={"AMD": "hybrid", "MSFT": "sma"}
    )

    assert implicit.promoted_values["symbol_strategy_modes"] == {
        "AMD": "hybrid",
        "MSFT": "hybrid",
    }
    assert sparse == explicit
    assert sparse.canonical_bytes() == explicit.canonical_bytes()
    assert sparse.sha256() == explicit.sha256()


def test_from_bot_config_preserves_active_symbols_when_exclusions_are_disjoint_provenance() -> None:
    contract = _contract(excluded_symbols=("TSLA",))

    assert contract.symbols == ("AMD", "MSFT")
    assert contract.promoted_values["symbol_strategy_modes"] == {
        "AMD": "hybrid",
        "MSFT": "hybrid",
    }


def test_from_bot_config_rejects_mode_override_for_inactive_symbol() -> None:
    with pytest.raises(ContractValidationError, match="inactive symbols"):
        _contract(symbol_strategy_modes={"TSLA": "sma"})


def test_symbol_order_is_preserved_and_changes_the_identity() -> None:
    values = _promoted_values()
    forward = CanonicalResolvedPromotedStrategy.from_mapping(
        ["AMD", "MSFT"], values
    )
    reverse = CanonicalResolvedPromotedStrategy.from_mapping(
        ["MSFT", "AMD"], values
    )

    assert forward.symbols == ("AMD", "MSFT")
    assert reverse.symbols == ("MSFT", "AMD")
    assert forward.canonical_json() != reverse.canonical_json()
    assert forward.sha256() != reverse.sha256()
    assert list(reverse.promoted_values["symbol_strategy_modes"]) == [
        "MSFT",
        "AMD",
    ]


@pytest.mark.parametrize(
    "symbols",
    [
        ["amd", "MSFT"],
        [" AMD", "MSFT"],
        ["", "MSFT"],
        ["AMD", "AMD"],
        ["AMD", 7],
    ],
    ids=["lowercase", "whitespace", "blank", "duplicate", "non-string"],
)
def test_symbols_must_be_strict_canonical_ordered_values(symbols: list[object]) -> None:
    with pytest.raises(ContractValidationError):
        CanonicalResolvedPromotedStrategy.from_mapping(symbols, _promoted_values())


@pytest.mark.parametrize(
    "modes",
    [
        {"AMD": "hybrid"},
        {"AMD": "hybrid", "MSFT": "hybrid", "TSLA": "sma"},
        {"AMD": "hybrid", "msft": "hybrid"},
        None,
    ],
    ids=["missing", "extra", "noncanonical-key", "null"],
)
def test_mapping_boundary_requires_exact_full_symbol_mode_map(modes: object) -> None:
    with pytest.raises(ContractValidationError, match="symbol_strategy_modes"):
        CanonicalResolvedPromotedStrategy.from_mapping(
            ["AMD", "MSFT"],
            _with_promoted_value("symbol_strategy_modes", modes),
        )


def test_all_constructor_paths_round_trip_to_identical_bytes_and_digest() -> None:
    expected = _contract(symbol_strategy_modes={"MSFT": "sma"})
    values = expected.promoted_values
    instances = (
        CanonicalResolvedPromotedStrategy.from_mapping(expected.symbols, values),
        CanonicalResolvedPromotedStrategy.from_field_pairs(
            expected.symbols, values.items()
        ),
        CanonicalResolvedPromotedStrategy.from_payload(
            expected.to_canonical_dict()
        ),
        CanonicalResolvedPromotedStrategy.from_json(expected.canonical_json()),
    )

    assert all(instance == expected for instance in instances)
    assert all(instance.canonical_bytes() == expected.canonical_bytes() for instance in instances)
    assert all(instance.sha256() == expected.sha256() for instance in instances)
    assert expected.sha256() == hashlib.sha256(expected.canonical_bytes()).hexdigest()


def test_research_resolution_must_prove_runtime_fixed_strategy_values() -> None:
    expected = _contract()
    exact_assumptions = dict(RUNTIME_FIXED_STRATEGY_ASSUMPTIONS)
    resolved = CanonicalResolvedPromotedStrategy.from_research_resolution(
        expected.symbols,
        expected.promoted_values,
        exact_assumptions,
    )

    assert resolved == expected

    wrong = dict(exact_assumptions)
    wrong["breakout_gap_pct_min"] = 0.01
    with pytest.raises(ContractValidationError, match="breakout_gap_pct_min"):
        CanonicalResolvedPromotedStrategy.from_research_resolution(
            expected.symbols,
            expected.promoted_values,
            wrong,
        )

    missing = dict(exact_assumptions)
    missing.pop("wick_fade_max_hold_bars")
    with pytest.raises(ContractValidationError, match="missing"):
        CanonicalResolvedPromotedStrategy.from_research_resolution(
            expected.symbols,
            expected.promoted_values,
            missing,
        )


def test_mapping_and_mode_input_order_do_not_change_canonical_identity() -> None:
    expected = _contract(symbol_strategy_modes={"MSFT": "sma"})
    reversed_values = dict(reversed(tuple(expected.promoted_values.items())))
    reversed_values["symbol_strategy_modes"] = {
        "MSFT": "sma",
        "AMD": "hybrid",
    }

    reordered = CanonicalResolvedPromotedStrategy.from_mapping(
        expected.symbols, reversed_values
    )

    assert reordered.canonical_json() == expected.canonical_json()
    assert reordered.sha256() == expected.sha256()


def test_contract_does_not_retain_or_expose_mutable_input_state() -> None:
    values = _promoted_values()
    modes = values["symbol_strategy_modes"]
    contract = CanonicalResolvedPromotedStrategy.from_mapping(
        ["AMD", "MSFT"], values
    )
    original_digest = contract.sha256()

    values["strategy_mode"] = "sma"
    modes["AMD"] = "sma"  # type: ignore[index]
    exported = contract.to_canonical_dict()
    exported["symbols"].reverse()
    exported["promoted_values"]["symbol_strategy_modes"]["AMD"] = "sma"

    assert contract.symbols == ("AMD", "MSFT")
    assert contract.promoted_values["strategy_mode"] == "hybrid"
    assert contract.promoted_values["symbol_strategy_modes"]["AMD"] == "hybrid"
    assert contract.sha256() == original_digest
    with pytest.raises(FrozenInstanceError):
        contract.symbols = ("MSFT", "AMD")  # type: ignore[misc]


def test_from_mapping_rejects_a_missing_promoted_field() -> None:
    values = _promoted_values()
    values.pop("mean_reversion_stop_pct")

    with pytest.raises(ContractValidationError, match="missing=.*mean_reversion_stop_pct"):
        CanonicalResolvedPromotedStrategy.from_mapping(["AMD", "MSFT"], values)


def test_from_mapping_rejects_an_extra_promoted_field() -> None:
    values = _promoted_values()
    values["research_only_knob"] = 1.0

    with pytest.raises(ContractValidationError, match="unexpected=.*research_only_knob"):
        CanonicalResolvedPromotedStrategy.from_mapping(["AMD", "MSFT"], values)


def test_from_field_pairs_rejects_duplicate_promoted_field() -> None:
    pairs = list(_promoted_values().items())
    pairs.append(("mean_reversion_stop_pct", 0.1))

    with pytest.raises(ContractValidationError, match="Duplicate promoted field"):
        CanonicalResolvedPromotedStrategy.from_field_pairs(
            ["AMD", "MSFT"], pairs
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("regime_filter_enabled", 1),
        ("regime_filter_enabled", "true"),
        ("regime_filter_enabled", None),
        ("sma_bars", True),
        ("sma_bars", 20.0),
        ("sma_bars", "20"),
        ("entry_threshold_pct", True),
        ("entry_threshold_pct", 1),
        ("entry_threshold_pct", "0.001"),
        ("entry_threshold_pct", float("nan")),
        ("entry_threshold_pct", float("inf")),
        ("entry_threshold_pct", None),
    ],
)
def test_mapping_boundary_rejects_bool_int_string_null_and_nonfinite_confusion(
    field: str, value: object
) -> None:
    with pytest.raises(ContractValidationError):
        CanonicalResolvedPromotedStrategy.from_mapping(
            ["AMD", "MSFT"], _with_promoted_value(field, value)
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("strategy_mode", "HYBRID"),
        ("strategy_mode", " hybrid"),
        ("strategy_mode", "not_a_mode"),
        ("threshold_mode", "ATR_MULT"),
        ("bb_exit_mode", "middle-band"),
    ],
)
def test_enum_values_must_already_be_canonical(field: str, value: object) -> None:
    with pytest.raises(ContractValidationError):
        CanonicalResolvedPromotedStrategy.from_mapping(
            ["AMD", "MSFT"], _with_promoted_value(field, value)
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("sma_bars", 0),
        ("trend_pullback_hold_bars", -1),
        ("entry_threshold_pct", 1.01),
        ("atr_percentile_threshold", 100.01),
        ("atr_multiple", -0.01),
        ("bb_min_mid_slope", -0.01),
    ],
)
def test_numeric_domains_are_enforced(field: str, value: object) -> None:
    with pytest.raises(ContractValidationError):
        CanonicalResolvedPromotedStrategy.from_mapping(
            ["AMD", "MSFT"], _with_promoted_value(field, value)
        )


def test_cross_field_probability_order_is_enforced() -> None:
    with pytest.raises(ContractValidationError, match="ml_probability_sell"):
        CanonicalResolvedPromotedStrategy.from_mapping(
            ["AMD", "MSFT"],
            _with_promoted_value("ml_probability_sell", 0.75),
        )


def test_cross_field_enabled_atr_range_is_enforced() -> None:
    with pytest.raises(ContractValidationError, match="min_atr_percentile"):
        CanonicalResolvedPromotedStrategy.from_mapping(
            ["AMD", "MSFT"],
            _with_promoted_value("trend_pullback_max_atr_percentile", 10.0),
        )


@pytest.mark.parametrize(
    ("update", "error"),
    [
        ({"schema_name": "other.schema"}, "schema_name"),
        ({"schema_version": 2}, "schema_version"),
        ({"schema_version": True}, "schema_version"),
        ({"extra": "field"}, "envelope fields"),
    ],
)
def test_payload_requires_exact_versioned_envelope(
    update: dict[str, object], error: str
) -> None:
    payload = _contract().to_canonical_dict()
    payload.update(update)

    with pytest.raises(ContractValidationError, match=error):
        CanonicalResolvedPromotedStrategy.from_payload(payload)


def test_payload_rejects_unproven_runtime_fixed_strategy_assumptions() -> None:
    payload = _contract().to_canonical_dict()
    payload["runtime_fixed_strategy_assumptions"]["values"][
        "breakout_gap_pct_min"
    ] = 0.01

    with pytest.raises(ContractValidationError, match="breakout_gap_pct_min"):
        CanonicalResolvedPromotedStrategy.from_payload(payload)


@pytest.mark.parametrize("location", ["envelope", "promoted"])
def test_from_json_rejects_duplicate_object_keys(location: str) -> None:
    text = _contract().canonical_json()
    if location == "envelope":
        duplicate = f'{{"schema_name":{json.dumps(SCHEMA_NAME)},' + text[1:]
    else:
        duplicate = text.replace(
            '"promoted_values":{',
            '"promoted_values":{"strategy_mode":"hybrid",',
            1,
        )

    with pytest.raises(ContractValidationError, match="Duplicate JSON field"):
        CanonicalResolvedPromotedStrategy.from_json(duplicate)


@pytest.mark.parametrize(
    "text",
    [
        "{not-json",
        '{"value":NaN}',
        '{"value":Infinity}',
    ],
    ids=["malformed", "nan", "infinity"],
)
def test_from_json_rejects_malformed_and_nonstandard_json(text: str) -> None:
    with pytest.raises(ContractValidationError):
        CanonicalResolvedPromotedStrategy.from_json(text)


def test_signed_zero_has_one_canonical_spelling_and_digest() -> None:
    negative = CanonicalResolvedPromotedStrategy.from_mapping(
        ["AMD", "MSFT"], _with_promoted_value("sma_stop_pct", -0.0)
    )
    positive = CanonicalResolvedPromotedStrategy.from_mapping(
        ["AMD", "MSFT"], _with_promoted_value("sma_stop_pct", 0.0)
    )

    assert negative.canonical_json() == positive.canonical_json()
    assert negative.sha256() == positive.sha256()
    assert '"sma_stop_pct":0.0' in negative.canonical_json()


def test_meaningful_strategy_and_universe_changes_change_digest() -> None:
    baseline = _contract()
    strategy_change = _contract(bb_period=21)
    mode_change = _contract(symbol_strategy_modes={"MSFT": "sma"})
    universe_change = _contract(symbols=["AMD", "MSFT", "TSLA"])

    assert len(
        {
            baseline.sha256(),
            strategy_change.sha256(),
            mode_change.sha256(),
            universe_change.sha256(),
        }
    ) == 4


def test_schema_version_is_bound_into_the_content_identity() -> None:
    contract = _contract()
    future_payload = contract.to_canonical_dict()
    future_payload["schema_version"] = SCHEMA_VERSION + 1
    future_bytes = json.dumps(
        future_payload,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
    ).encode("utf-8")

    assert hashlib.sha256(future_bytes).hexdigest() != contract.sha256()
    with pytest.raises(ContractValidationError, match="schema_version"):
        CanonicalResolvedPromotedStrategy.from_payload(future_payload)


def test_resolved_contract_identity_does_not_read_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    effective_config = _bot_config()
    before = CanonicalResolvedPromotedStrategy.from_bot_config(effective_config)

    monkeypatch.setenv("STRATEGY_MODE", "sma")
    monkeypatch.setenv("SMA_BARS", "999")
    monkeypatch.setenv("SYMBOLS", "TSLA")
    after = CanonicalResolvedPromotedStrategy.from_bot_config(effective_config)

    assert after.canonical_bytes() == before.canonical_bytes()
    assert after.sha256() == before.sha256()


def test_known_paper_shadow_symbol_universe_is_representable() -> None:
    symbols = [
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
    contract = _contract(symbols=symbols)

    assert contract.symbols == tuple(symbols)
    assert list(contract.promoted_values["symbol_strategy_modes"]) == symbols
    assert len(contract.promoted_values) == 65

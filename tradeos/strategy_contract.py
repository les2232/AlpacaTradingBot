"""Versioned canonical contract for TradeOS's promoted strategy surface.

This module is the authoritative definition of the *promoted* strategy
contract.  It deliberately does not claim to contain every field accepted by
``StrategyConfig``: fields that cannot currently be supplied through
``BotConfig`` are outside schema version 1.

Symbol order is semantically significant.  It is preserved in canonical JSON
and therefore bound by the content digest.  ``symbol_strategy_modes`` is stored
as a fully resolved mapping with exactly one effective mode per active symbol.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, fields
import hashlib
import json
import math
import re
from typing import Any, ClassVar, get_type_hints


SCHEMA_NAME = "tradeos.canonical_resolved_promoted_strategy"
SCHEMA_VERSION = 1
SYMBOL_ORDER_SEMANTICS = "significant"
RUNTIME_FIXED_ASSUMPTIONS_SCHEMA_NAME = (
    "tradeos.runtime_fixed_strategy_assumptions"
)
RUNTIME_FIXED_ASSUMPTIONS_SCHEMA_VERSION = 1

# Frozen schema-version-1 order.  Do not derive or silently regenerate this
# tuple.  Any promoted field-surface change requires an explicit schema review
# and version decision.
PROMOTED_STRATEGY_FIELDS: tuple[str, ...] = (
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

# These StrategyConfig fields are not configurable through BotConfig in schema
# v1.  Live execution therefore uses these exact StrategyConfig defaults.
# Research must prove its resolved values match this manifest before emitting a
# canonical contract; the values are bound into canonical JSON and its digest.
RUNTIME_FIXED_STRATEGY_ASSUMPTIONS: tuple[tuple[str, object], ...] = (
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

_RUNTIME_STRATEGY_SELECTORS: tuple[str, ...] = (
    "sma_bars",
    "symbol_strategy_modes",
    "bar_timeframe_minutes",
)

# Enum tokens are part of schema v1 and are intentionally frozen here rather
# than borrowed dynamically from strategy.py.  Drift checks below compare them
# with the execution model so a source enum cannot silently redefine v1.
_STRATEGY_MODE_CHOICES = (
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
)
_THRESHOLD_MODE_CHOICES = ("static_pct", "atr_multiple")
_TIME_WINDOW_CHOICES = (
    "full_day",
    "morning_only",
    "afternoon_only",
    "combined_windows",
)
_ORB_FILTER_CHOICES = ("none", "volume_or_volatility")
_BREAKOUT_EXIT_CHOICES = (
    "target_1x_stop_low",
    "target_1_5x_stop_low",
    "eod_only",
    "trailing_stop_half_range",
    "trailing_stop_full_range",
    "target_1x_tight_stop",
)
_MEAN_REVERSION_EXIT_CHOICES = ("sma", "midpoint", "eod")
_TREND_PULLBACK_EXIT_CHOICES = (
    "fixed_bars",
    "take_profit",
    "hybrid_tp_or_time",
)
_MOMENTUM_BREAKOUT_EXIT_CHOICES = ("fixed_bars",)
_VOLATILITY_EXPANSION_EXIT_CHOICES = ("fixed_bars",)
_BOLLINGER_EXIT_CHOICES = ("middle_band",)

PROMOTED_STRATEGY_ENUM_CHOICES: tuple[
    tuple[str, tuple[str, ...]], ...
] = (
    ("strategy_mode", _STRATEGY_MODE_CHOICES),
    ("threshold_mode", _THRESHOLD_MODE_CHOICES),
    ("time_window_mode", _TIME_WINDOW_CHOICES),
    ("orb_filter_mode", _ORB_FILTER_CHOICES),
    ("breakout_exit_style", _BREAKOUT_EXIT_CHOICES),
    ("mean_reversion_exit_style", _MEAN_REVERSION_EXIT_CHOICES),
    ("trend_pullback_exit_style", _TREND_PULLBACK_EXIT_CHOICES),
    ("momentum_breakout_exit_style", _MOMENTUM_BREAKOUT_EXIT_CHOICES),
    ("volatility_expansion_exit_style", _VOLATILITY_EXPANSION_EXIT_CHOICES),
    ("bb_exit_mode", _BOLLINGER_EXIT_CHOICES),
)
_ENUM_CHOICES: dict[str, tuple[str, ...]] = dict(
    PROMOTED_STRATEGY_ENUM_CHOICES
)

_SOURCE_ENUM_NAMES = {
    "strategy_mode": "STRATEGY_MODE_CHOICES",
    "threshold_mode": "THRESHOLD_MODE_CHOICES",
    "time_window_mode": "TIME_WINDOW_CHOICES",
    "orb_filter_mode": "ORB_FILTER_CHOICES",
    "breakout_exit_style": "BREAKOUT_EXIT_CHOICES",
    "mean_reversion_exit_style": "MEAN_REVERSION_EXIT_CHOICES",
    "trend_pullback_exit_style": "TREND_PULLBACK_EXIT_CHOICES",
    "momentum_breakout_exit_style": "MOMENTUM_BREAKOUT_EXIT_CHOICES",
    "volatility_expansion_exit_style": "VOLATILITY_EXPANSION_EXIT_CHOICES",
    "bb_exit_mode": "BOLLINGER_EXIT_CHOICES",
}

_BOOL_FIELDS = frozenset(
    {
        "regime_filter_enabled",
        "mean_reversion_trend_filter",
        "mean_reversion_trend_slope_filter",
        "volatility_expansion_trend_filter",
        "volatility_expansion_use_volume_confirm",
        "bb_use_volume_confirm",
        "bb_trend_filter",
    }
)

_POSITIVE_INT_FIELDS = frozenset(
    {
        "sma_bars",
        "bar_timeframe_minutes",
        "momentum_breakout_lookback_bars",
        "volatility_expansion_lookback_bars",
        "bb_period",
        "bb_width_lookback",
        "bb_slope_lookback",
    }
)

_NONNEGATIVE_INT_FIELDS = frozenset(
    {
        "trend_pullback_hold_bars",
        "momentum_breakout_hold_bars",
        "volatility_expansion_hold_bars",
    }
)

_UNIT_INTERVAL_FLOAT_FIELDS = frozenset(
    {
        "ml_probability_buy",
        "ml_probability_sell",
        "entry_threshold_pct",
        "breakout_tight_stop_fraction",
        "breakout_max_stop_pct",
        "mean_reversion_stop_pct",
        "sma_stop_pct",
        "trend_pullback_entry_threshold",
        "trend_pullback_take_profit_pct",
        "trend_pullback_stop_pct",
        "momentum_breakout_entry_buffer_pct",
        "momentum_breakout_stop_pct",
        "momentum_breakout_take_profit_pct",
        "volatility_expansion_entry_buffer_pct",
        "volatility_expansion_stop_pct",
        "volatility_expansion_take_profit_pct",
        "bb_squeeze_quantile",
        "bb_breakout_buffer_pct",
    }
)

_PERCENTILE_FLOAT_FIELDS = frozenset(
    {
        "atr_percentile_threshold",
        "mean_reversion_max_atr_percentile",
        "min_atr_percentile",
        "max_adx_threshold",
        "trend_pullback_min_adx",
        "trend_pullback_min_atr_percentile",
        "trend_pullback_max_atr_percentile",
        "momentum_breakout_min_adx",
        "momentum_breakout_min_atr_percentile",
        "volatility_expansion_max_atr_percentile",
    }
)

_NONNEGATIVE_FLOAT_FIELDS = frozenset(
    {
        "atr_multiple",
        "vwap_z_entry_threshold",
        "vwap_z_stop_atr_multiple",
        "bb_stddev_mult",
        "bb_volume_mult",
        "bb_min_mid_slope",
    }
)

_FINITE_FLOAT_FIELDS = frozenset(
    {
        "trend_pullback_min_slope",
        "momentum_breakout_min_slope",
        "volatility_expansion_min_slope",
    }
)

_ENVELOPE_FIELDS = frozenset(
    {
        "schema_name",
        "schema_version",
        "runtime_fixed_strategy_assumptions",
        "symbols",
        "promoted_values",
    }
)
_ASSUMPTIONS_ENVELOPE_FIELDS = frozenset(
    {"schema_name", "schema_version", "values"}
)
_SYMBOL_PATTERN = re.compile(r"^[A-Z][A-Z0-9.-]*$")


class ContractValidationError(ValueError):
    """Raised when a resolved promoted-strategy contract is not canonical."""


def _assert_schema_definition() -> None:
    if len(PROMOTED_STRATEGY_FIELDS) != 65:
        raise RuntimeError(
            "Schema version 1 must contain exactly 65 promoted strategy fields; "
            f"found {len(PROMOTED_STRATEGY_FIELDS)}."
        )
    duplicates = tuple(
        name
        for index, name in enumerate(PROMOTED_STRATEGY_FIELDS)
        if name in PROMOTED_STRATEGY_FIELDS[:index]
    )
    if duplicates:
        raise RuntimeError(f"Duplicate promoted strategy fields: {duplicates!r}")

    assumption_names = tuple(
        name for name, _value in RUNTIME_FIXED_STRATEGY_ASSUMPTIONS
    )
    if len(assumption_names) != 13 or len(set(assumption_names)) != 13:
        raise RuntimeError(
            "Schema version 1 must define exactly 13 unique runtime-fixed "
            "strategy assumptions."
        )

    specified = (
        set(_ENUM_CHOICES)
        | set(_BOOL_FIELDS)
        | set(_POSITIVE_INT_FIELDS)
        | set(_NONNEGATIVE_INT_FIELDS)
        | set(_UNIT_INTERVAL_FLOAT_FIELDS)
        | set(_PERCENTILE_FLOAT_FIELDS)
        | set(_NONNEGATIVE_FLOAT_FIELDS)
        | set(_FINITE_FLOAT_FIELDS)
        | {"symbol_strategy_modes"}
    )
    expected = set(PROMOTED_STRATEGY_FIELDS)
    if specified != expected:
        raise RuntimeError(
            "Promoted strategy validator coverage does not match schema: "
            f"missing={sorted(expected - specified)!r}, "
            f"unexpected={sorted(specified - expected)!r}"
        )


_assert_schema_definition()


def source_model_promoted_fields() -> tuple[str, ...]:
    """Return the old dynamic surface solely for explicit drift detection.

    Production identity never derives its schema from this function.  Tests use
    it to force an intentional schema/version update when source models change.
    The local import avoids loading runtime/config code on normal contract use.
    """

    from strategy import StrategyConfig
    from trading_bot import BotConfig

    bot_fields = frozenset(field.name for field in fields(BotConfig))
    return tuple(
        dict.fromkeys(
            field_name
            for field_name in (
                "strategy_mode",
                *_RUNTIME_STRATEGY_SELECTORS,
                *(field.name for field in fields(StrategyConfig)),
            )
            if field_name in bot_fields
        )
    )


def assert_source_models_match_contract() -> None:
    """Fail if v1 fields, types, or enums drift from current source models."""

    from dataclasses import MISSING

    import strategy as source_strategy
    from strategy import StrategyConfig
    from trading_bot import BotConfig

    derived = source_model_promoted_fields()
    if derived != PROMOTED_STRATEGY_FIELDS:
        expected = set(PROMOTED_STRATEGY_FIELDS)
        current = set(derived)
        raise ContractValidationError(
            "Source-model promoted surface differs from the explicit schema; "
            "review the schema and bump its version intentionally. "
            f"missing_from_source={sorted(expected - current)!r}, "
            f"new_in_source={sorted(current - expected)!r}, "
            f"order_changed={expected == current}"
        )

    enum_drift = {
        field_name: {
            "schema": choices,
            "source": tuple(getattr(source_strategy, _SOURCE_ENUM_NAMES[field_name])),
        }
        for field_name, choices in _ENUM_CHOICES.items()
        if tuple(getattr(source_strategy, _SOURCE_ENUM_NAMES[field_name])) != choices
    }
    if enum_drift:
        raise ContractValidationError(
            "Source enum choices differ from the frozen schema-v1 tokens; "
            f"review and version intentionally: {enum_drift!r}"
        )

    expected_types: dict[str, object] = {
        **{name: str for name in _ENUM_CHOICES},
        **{name: bool for name in _BOOL_FIELDS},
        **{name: int for name in _POSITIVE_INT_FIELDS | _NONNEGATIVE_INT_FIELDS},
        **{
            name: float
            for name in (
                _UNIT_INTERVAL_FLOAT_FIELDS
                | _PERCENTILE_FLOAT_FIELDS
                | _NONNEGATIVE_FLOAT_FIELDS
                | _FINITE_FLOAT_FIELDS
            )
        },
        "symbol_strategy_modes": dict[str, str] | None,
    }
    bot_types = get_type_hints(BotConfig)
    strategy_types = get_type_hints(StrategyConfig)
    type_drift: dict[str, object] = {}
    for name, expected_type in expected_types.items():
        if bot_types.get(name) != expected_type:
            type_drift[f"BotConfig.{name}"] = bot_types.get(name)
        if name in strategy_types and strategy_types[name] != expected_type:
            type_drift[f"StrategyConfig.{name}"] = strategy_types[name]
    if type_drift:
        raise ContractValidationError(
            "Source field types differ from the frozen schema-v1 validators; "
            f"review and version intentionally: {type_drift!r}"
        )

    promoted_names = set(PROMOTED_STRATEGY_FIELDS)
    source_fixed_fields = tuple(
        field
        for field in fields(StrategyConfig)
        if field.name not in promoted_names
    )
    expected_fixed_names = tuple(
        name for name, _value in RUNTIME_FIXED_STRATEGY_ASSUMPTIONS
    )
    if tuple(field.name for field in source_fixed_fields) != expected_fixed_names:
        raise ContractValidationError(
            "Runtime-fixed StrategyConfig field surface differs from the frozen "
            "schema-v1 assumptions; review and version intentionally."
        )
    fixed_default_drift: dict[str, object] = {}
    expected_fixed_values = dict(RUNTIME_FIXED_STRATEGY_ASSUMPTIONS)
    for field in source_fixed_fields:
        expected_value = expected_fixed_values[field.name]
        if (
            field.default is MISSING
            or type(field.default) is not type(expected_value)
            or field.default != expected_value
        ):
            fixed_default_drift[field.name] = field.default
    if fixed_default_drift:
        raise ContractValidationError(
            "StrategyConfig defaults differ from the frozen schema-v1 runtime "
            f"assumptions; review and version intentionally: {fixed_default_drift!r}"
        )


def _validate_symbols(raw_symbols: object, *, label: str = "symbols") -> tuple[str, ...]:
    if not isinstance(raw_symbols, (list, tuple)):
        raise ContractValidationError(f"{label} must be an ordered list or tuple.")
    if not raw_symbols:
        raise ContractValidationError(f"{label} must not be empty.")

    normalized: list[str] = []
    seen: set[str] = set()
    for index, raw_symbol in enumerate(raw_symbols):
        if type(raw_symbol) is not str:
            raise ContractValidationError(f"{label}[{index}] must be a string.")
        if not raw_symbol or raw_symbol != raw_symbol.strip():
            raise ContractValidationError(
                f"{label}[{index}] must be nonblank and contain no surrounding whitespace."
            )
        if raw_symbol != raw_symbol.upper() or not _SYMBOL_PATTERN.fullmatch(raw_symbol):
            raise ContractValidationError(
                f"{label}[{index}] must be a canonical uppercase trading symbol."
            )
        if raw_symbol in seen:
            raise ContractValidationError(f"{label} contains duplicate symbol {raw_symbol!r}.")
        seen.add(raw_symbol)
        normalized.append(raw_symbol)
    return tuple(normalized)


def _validate_enum(name: str, value: object, choices: tuple[str, ...]) -> str:
    if type(value) is not str:
        raise ContractValidationError(f"{name} must be a string enum value.")
    if value not in choices:
        raise ContractValidationError(
            f"{name} must be one of {choices!r}; received {value!r}."
        )
    return value


def _validate_int(name: str, value: object, *, minimum: int) -> int:
    if type(value) is not int:
        raise ContractValidationError(f"{name} must be an integer.")
    if value < minimum:
        raise ContractValidationError(f"{name} must be >= {minimum}.")
    return value


def _validate_float(
    name: str,
    value: object,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float:
    if type(value) is not float:
        raise ContractValidationError(f"{name} must be a float.")
    if not math.isfinite(value):
        raise ContractValidationError(f"{name} must be finite.")
    if minimum is not None and value < minimum:
        raise ContractValidationError(f"{name} must be >= {minimum}.")
    if maximum is not None and value > maximum:
        raise ContractValidationError(f"{name} must be <= {maximum}.")
    # JSON distinguishes neither signed zero semantically nor portably.  Bind
    # both representations to the single canonical spelling 0.0.
    return 0.0 if value == 0.0 else value


def _validate_symbol_modes(
    raw_modes: object,
    symbols: tuple[str, ...],
) -> tuple[tuple[str, str], ...]:
    if not isinstance(raw_modes, Mapping):
        raise ContractValidationError(
            "symbol_strategy_modes must be an object with one mode per active symbol."
        )
    raw_keys = tuple(raw_modes.keys())
    if any(type(key) is not str for key in raw_keys):
        raise ContractValidationError("symbol_strategy_modes keys must be strings.")
    expected = set(symbols)
    actual = set(raw_keys)
    if actual != expected or len(raw_keys) != len(symbols):
        raise ContractValidationError(
            "symbol_strategy_modes must contain exactly the active symbol universe; "
            f"missing={sorted(expected - actual)!r}, "
            f"unexpected={sorted(actual - expected)!r}."
        )
    return tuple(
        (
            symbol,
            _validate_enum(
                f"symbol_strategy_modes[{symbol!r}]",
                raw_modes[symbol],
                _STRATEGY_MODE_CHOICES,
            ),
        )
        for symbol in symbols
    )


def _validate_scalar(name: str, value: object) -> object:
    if name in _ENUM_CHOICES:
        return _validate_enum(name, value, _ENUM_CHOICES[name])
    if name in _BOOL_FIELDS:
        if type(value) is not bool:
            raise ContractValidationError(f"{name} must be a boolean.")
        return value
    if name in _POSITIVE_INT_FIELDS:
        return _validate_int(name, value, minimum=1)
    if name in _NONNEGATIVE_INT_FIELDS:
        return _validate_int(name, value, minimum=0)
    if name in _UNIT_INTERVAL_FLOAT_FIELDS:
        return _validate_float(name, value, minimum=0.0, maximum=1.0)
    if name in _PERCENTILE_FLOAT_FIELDS:
        return _validate_float(name, value, minimum=0.0, maximum=100.0)
    if name in _NONNEGATIVE_FLOAT_FIELDS:
        return _validate_float(name, value, minimum=0.0)
    if name in _FINITE_FLOAT_FIELDS:
        return _validate_float(name, value)
    raise RuntimeError(f"No validator registered for promoted field {name!r}.")


def validate_runtime_fixed_strategy_values(
    raw_values: Mapping[str, object],
) -> dict[str, object]:
    """Validate research/backtest values against live's fixed v1 defaults.

    Research producers must call this with the values actually used by the
    resolved backtest.  Missing values are never inferred from this manifest.
    """

    if not isinstance(raw_values, Mapping):
        raise ContractValidationError(
            "runtime-fixed strategy values must be an object mapping."
        )
    raw_keys = tuple(raw_values.keys())
    if any(type(key) is not str for key in raw_keys):
        raise ContractValidationError(
            "runtime-fixed strategy value names must be strings."
        )
    expected_values = dict(RUNTIME_FIXED_STRATEGY_ASSUMPTIONS)
    expected_keys = set(expected_values)
    actual_keys = set(raw_keys)
    if actual_keys != expected_keys or len(raw_keys) != len(expected_keys):
        raise ContractValidationError(
            "runtime-fixed strategy values are not exact; "
            f"missing={sorted(expected_keys - actual_keys)!r}, "
            f"unexpected={sorted(actual_keys - expected_keys)!r}."
        )
    validated: dict[str, object] = {}
    for name, expected_value in RUNTIME_FIXED_STRATEGY_ASSUMPTIONS:
        actual_value = raw_values[name]
        if (
            type(actual_value) is not type(expected_value)
            or actual_value != expected_value
        ):
            raise ContractValidationError(
                f"runtime-fixed strategy value {name!r} must equal "
                f"{expected_value!r} with type {type(expected_value).__name__}."
            )
        validated[name] = actual_value
    return validated


def _runtime_fixed_assumptions_payload() -> dict[str, object]:
    return {
        "schema_name": RUNTIME_FIXED_ASSUMPTIONS_SCHEMA_NAME,
        "schema_version": RUNTIME_FIXED_ASSUMPTIONS_SCHEMA_VERSION,
        "values": dict(RUNTIME_FIXED_STRATEGY_ASSUMPTIONS),
    }


def _validate_runtime_fixed_assumptions_payload(raw_payload: object) -> None:
    if not isinstance(raw_payload, Mapping):
        raise ContractValidationError(
            "runtime_fixed_strategy_assumptions must be an object."
        )
    raw_keys = tuple(raw_payload.keys())
    if any(type(key) is not str for key in raw_keys):
        raise ContractValidationError(
            "runtime-fixed assumptions envelope field names must be strings."
        )
    if (
        set(raw_keys) != _ASSUMPTIONS_ENVELOPE_FIELDS
        or len(raw_keys) != len(_ASSUMPTIONS_ENVELOPE_FIELDS)
    ):
        raise ContractValidationError(
            "runtime-fixed assumptions envelope fields are not exact."
        )
    if (
        type(raw_payload["schema_name"]) is not str
        or raw_payload["schema_name"] != RUNTIME_FIXED_ASSUMPTIONS_SCHEMA_NAME
    ):
        raise ContractValidationError(
            "Unsupported runtime-fixed assumptions schema_name."
        )
    if (
        type(raw_payload["schema_version"]) is not int
        or raw_payload["schema_version"]
        != RUNTIME_FIXED_ASSUMPTIONS_SCHEMA_VERSION
    ):
        raise ContractValidationError(
            "Unsupported runtime-fixed assumptions schema_version."
        )
    values = raw_payload["values"]
    if not isinstance(values, Mapping):
        raise ContractValidationError(
            "runtime-fixed assumptions values must be an object mapping."
        )
    validate_runtime_fixed_strategy_values(values)


def _reject_duplicate_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ContractValidationError(f"Duplicate JSON field {key!r}.")
        result[key] = value
    return result


def _reject_nonstandard_json_number(token: str) -> None:
    raise ContractValidationError(f"Non-finite JSON number {token!r} is not allowed.")


@dataclass(frozen=True, init=False)
class CanonicalResolvedPromotedStrategy:
    """Immutable, validated schema-v1 promoted strategy identity."""

    schema_name: ClassVar[str] = SCHEMA_NAME
    schema_version: ClassVar[int] = SCHEMA_VERSION
    symbol_order_semantics: ClassVar[str] = SYMBOL_ORDER_SEMANTICS

    symbols: tuple[str, ...]
    _promoted_items: tuple[tuple[str, object], ...]

    def __init__(self) -> None:
        raise TypeError(
            "Use from_bot_config, from_research_resolution, from_mapping, "
            "from_field_pairs, from_payload, or from_json to construct a "
            "validated canonical contract."
        )

    @classmethod
    def _create(
        cls,
        symbols: tuple[str, ...],
        promoted_items: tuple[tuple[str, object], ...],
    ) -> CanonicalResolvedPromotedStrategy:
        instance = object.__new__(cls)
        object.__setattr__(instance, "symbols", symbols)
        object.__setattr__(instance, "_promoted_items", promoted_items)
        return instance

    @classmethod
    def from_mapping(
        cls,
        symbols: Sequence[str],
        promoted_values: Mapping[str, object],
    ) -> CanonicalResolvedPromotedStrategy:
        """Validate an already-resolved exact mapping without coercion."""

        canonical_symbols = _validate_symbols(symbols)
        if not isinstance(promoted_values, Mapping):
            raise ContractValidationError("promoted_values must be an object mapping.")
        raw_keys = tuple(promoted_values.keys())
        if any(type(key) is not str for key in raw_keys):
            raise ContractValidationError("promoted field names must be strings.")
        expected = set(PROMOTED_STRATEGY_FIELDS)
        actual = set(raw_keys)
        if actual != expected or len(raw_keys) != len(PROMOTED_STRATEGY_FIELDS):
            raise ContractValidationError(
                "promoted_values must contain exactly the schema-v1 field surface; "
                f"missing={sorted(expected - actual)!r}, "
                f"unexpected={sorted(actual - expected)!r}."
            )

        validated_items: list[tuple[str, object]] = []
        for name in PROMOTED_STRATEGY_FIELDS:
            raw_value = promoted_values[name]
            if name == "symbol_strategy_modes":
                value = _validate_symbol_modes(raw_value, canonical_symbols)
            else:
                value = _validate_scalar(name, raw_value)
            validated_items.append((name, value))

        validated = dict(validated_items)
        if validated["ml_probability_sell"] > validated["ml_probability_buy"]:
            raise ContractValidationError(
                "ml_probability_sell must be <= ml_probability_buy."
            )
        min_atr = validated["trend_pullback_min_atr_percentile"]
        max_atr = validated["trend_pullback_max_atr_percentile"]
        if max_atr > 0.0 and min_atr > max_atr:
            raise ContractValidationError(
                "trend_pullback_min_atr_percentile must be <= the enabled maximum."
            )

        return cls._create(canonical_symbols, tuple(validated_items))

    @classmethod
    def from_research_resolution(
        cls,
        symbols: Sequence[str],
        promoted_values: Mapping[str, object],
        runtime_fixed_strategy_values: Mapping[str, object],
    ) -> CanonicalResolvedPromotedStrategy:
        """Construct research identity only after proving fixed live parity."""

        validate_runtime_fixed_strategy_values(runtime_fixed_strategy_values)
        return cls.from_mapping(symbols, promoted_values)

    @classmethod
    def from_field_pairs(
        cls,
        symbols: Sequence[str],
        field_pairs: Iterable[tuple[str, object]],
    ) -> CanonicalResolvedPromotedStrategy:
        """Build from pairs while retaining the ability to reject duplicates."""

        promoted_values: dict[str, object] = {}
        try:
            iterator = iter(field_pairs)
        except TypeError as exc:
            raise ContractValidationError("field_pairs must be iterable.") from exc
        for index, pair in enumerate(iterator):
            if not isinstance(pair, (list, tuple)) or len(pair) != 2:
                raise ContractValidationError(f"field_pairs[{index}] must contain name and value.")
            name, value = pair
            if type(name) is not str:
                raise ContractValidationError(f"field_pairs[{index}] name must be a string.")
            if name in promoted_values:
                raise ContractValidationError(f"Duplicate promoted field {name!r}.")
            promoted_values[name] = value
        return cls.from_mapping(symbols, promoted_values)

    @classmethod
    def from_payload(cls, payload: Mapping[str, object]) -> CanonicalResolvedPromotedStrategy:
        """Validate a complete versioned canonical envelope."""

        if not isinstance(payload, Mapping):
            raise ContractValidationError("Canonical payload must be an object.")
        raw_keys = tuple(payload.keys())
        if any(type(key) is not str for key in raw_keys):
            raise ContractValidationError("Canonical envelope field names must be strings.")
        if set(raw_keys) != _ENVELOPE_FIELDS or len(raw_keys) != len(_ENVELOPE_FIELDS):
            raise ContractValidationError(
                "Canonical envelope fields are not exact; "
                f"missing={sorted(_ENVELOPE_FIELDS - set(raw_keys))!r}, "
                f"unexpected={sorted(set(raw_keys) - _ENVELOPE_FIELDS)!r}."
            )
        if type(payload["schema_name"]) is not str or payload["schema_name"] != SCHEMA_NAME:
            raise ContractValidationError(f"Unsupported schema_name {payload['schema_name']!r}.")
        if type(payload["schema_version"]) is not int or payload["schema_version"] != SCHEMA_VERSION:
            raise ContractValidationError(f"Unsupported schema_version {payload['schema_version']!r}.")
        _validate_runtime_fixed_assumptions_payload(
            payload["runtime_fixed_strategy_assumptions"]
        )
        promoted_values = payload["promoted_values"]
        if not isinstance(promoted_values, Mapping):
            raise ContractValidationError("promoted_values must be an object mapping.")
        symbols = payload["symbols"]
        if not isinstance(symbols, (list, tuple)):
            raise ContractValidationError("symbols must be an ordered list or tuple.")
        return cls.from_mapping(symbols, promoted_values)

    @classmethod
    def from_json(cls, text: str) -> CanonicalResolvedPromotedStrategy:
        """Parse strict JSON, rejecting duplicate keys and non-finite numbers."""

        if type(text) is not str:
            raise ContractValidationError("Canonical JSON input must be text.")
        try:
            payload = json.loads(
                text,
                object_pairs_hook=_reject_duplicate_json_object,
                parse_constant=_reject_nonstandard_json_number,
            )
        except json.JSONDecodeError as exc:
            raise ContractValidationError(f"Malformed canonical JSON: {exc.msg}.") from exc
        return cls.from_payload(payload)

    @classmethod
    def from_bot_config(cls, config: object) -> CanonicalResolvedPromotedStrategy:
        """Resolve an effective BotConfig into explicit canonical values.

        Environment/default parsing belongs before this boundary.  After an
        effective ``BotConfig`` enters this method, no environment is read and
        no implicit defaults remain in the resulting contract.
        """

        try:
            raw_symbols = getattr(config, "symbols")
            raw_excluded = getattr(config, "excluded_symbols")
            base_mode = getattr(config, "strategy_mode")
            raw_modes = getattr(config, "symbol_strategy_modes")
        except AttributeError as exc:
            raise ContractValidationError(
                "from_bot_config requires an effective BotConfig-compatible object."
            ) from exc

        # Runtime config resolution has already applied exclusions and retained
        # them only as provenance.  ``BotConfig.symbols`` is therefore the
        # authoritative active order at this boundary; never filter it twice.
        active_symbols = _validate_symbols(raw_symbols, label="BotConfig.symbols")
        if not isinstance(raw_excluded, (list, tuple)):
            raise ContractValidationError("BotConfig.excluded_symbols must be a list or tuple.")
        if raw_excluded:
            excluded = _validate_symbols(raw_excluded, label="BotConfig.excluded_symbols")
        else:
            excluded = ()
        active_excluded = set(excluded) & set(active_symbols)
        if active_excluded:
            raise ContractValidationError(
                "Effective BotConfig.symbols still contains excluded symbols "
                f"{sorted(active_excluded)!r}."
            )

        effective_base_mode = _validate_enum(
            "strategy_mode",
            base_mode,
            _STRATEGY_MODE_CHOICES,
        )
        if raw_modes is None:
            sparse_modes: Mapping[object, object] = {}
        elif isinstance(raw_modes, Mapping):
            sparse_modes = raw_modes
        else:
            raise ContractValidationError("BotConfig.symbol_strategy_modes must be a mapping or None.")
        if any(type(symbol) is not str for symbol in sparse_modes):
            raise ContractValidationError("BotConfig.symbol_strategy_modes keys must be strings.")
        inactive_mode_keys = set(sparse_modes) - set(active_symbols)
        if inactive_mode_keys:
            raise ContractValidationError(
                "BotConfig.symbol_strategy_modes contains inactive symbols "
                f"{sorted(inactive_mode_keys)!r}."
            )
        full_modes = {
            symbol: _validate_enum(
                f"BotConfig.symbol_strategy_modes[{symbol!r}]",
                sparse_modes.get(symbol, effective_base_mode),
                _STRATEGY_MODE_CHOICES,
            )
            for symbol in active_symbols
        }

        promoted_values: dict[str, object] = {}
        for name in PROMOTED_STRATEGY_FIELDS:
            if name == "symbol_strategy_modes":
                promoted_values[name] = full_modes
                continue
            try:
                promoted_values[name] = getattr(config, name)
            except AttributeError as exc:
                raise ContractValidationError(
                    f"BotConfig-compatible object is missing promoted field {name!r}."
                ) from exc
        return cls.from_mapping(active_symbols, promoted_values)

    @property
    def promoted_values(self) -> dict[str, object]:
        """Return a fresh JSON-compatible mapping in schema order."""

        result: dict[str, object] = {}
        for name, value in self._promoted_items:
            if name == "symbol_strategy_modes":
                result[name] = {symbol: mode for symbol, mode in value}  # type: ignore[misc]
            else:
                result[name] = value
        return result

    def to_canonical_dict(self) -> dict[str, object]:
        """Return the deterministic, versioned JSON envelope."""

        return {
            "schema_name": SCHEMA_NAME,
            "schema_version": SCHEMA_VERSION,
            "runtime_fixed_strategy_assumptions": (
                _runtime_fixed_assumptions_payload()
            ),
            "symbols": list(self.symbols),
            "promoted_values": self.promoted_values,
        }

    def canonical_json(self) -> str:
        """Serialize using compact UTF-8 JSON rules with finite numbers only."""

        return json.dumps(
            self.to_canonical_dict(),
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
        )

    def canonical_bytes(self) -> bytes:
        return self.canonical_json().encode("utf-8")

    def sha256(self) -> str:
        return hashlib.sha256(self.canonical_bytes()).hexdigest()


__all__ = [
    "CanonicalResolvedPromotedStrategy",
    "ContractValidationError",
    "PROMOTED_STRATEGY_FIELDS",
    "PROMOTED_STRATEGY_ENUM_CHOICES",
    "RUNTIME_FIXED_ASSUMPTIONS_SCHEMA_NAME",
    "RUNTIME_FIXED_ASSUMPTIONS_SCHEMA_VERSION",
    "RUNTIME_FIXED_STRATEGY_ASSUMPTIONS",
    "SCHEMA_NAME",
    "SCHEMA_VERSION",
    "SYMBOL_ORDER_SEMANTICS",
    "assert_source_models_match_contract",
    "source_model_promoted_fields",
    "validate_runtime_fixed_strategy_values",
]

"""Immutable, versioned research-promotion bundles for TradeOS.

The bundle is the durable evidence boundary between a resolved backtest and a
runtime promotion.  It consumes the exact canonical strategy JSON emitted by
the backtest, validates every binding, and records enough identity to reject
partial, contradictory, or cross-generation publication.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from importlib import metadata as importlib_metadata
import base64
import csv
import hashlib
import io
import json
import math
import os
from pathlib import Path, PurePosixPath
import platform
import re
import subprocess
import sys
from typing import Any
from uuid import uuid4

from dataset_integrity import load_verified_manifest
from tradeos.strategy_contract import (
    PROMOTED_STRATEGY_ENUM_CHOICES,
    PROMOTED_STRATEGY_FIELDS,
    RUNTIME_FIXED_ASSUMPTIONS_SCHEMA_NAME,
    RUNTIME_FIXED_ASSUMPTIONS_SCHEMA_VERSION,
    RUNTIME_FIXED_STRATEGY_ASSUMPTIONS,
    SCHEMA_NAME as STRATEGY_SCHEMA_NAME,
    SCHEMA_VERSION as STRATEGY_SCHEMA_VERSION,
    CanonicalResolvedPromotedStrategy,
    ContractValidationError,
)


BUNDLE_SCHEMA_NAME = "tradeos.research_promotion_bundle"
BUNDLE_SCHEMA_VERSION = 1
GENERATOR_NAME = "tradeos.run_research"
GENERATOR_VERSION = 1
SOURCE_MANIFEST_SCHEMA_NAME = "tradeos.versionable_source_manifest"
SOURCE_MANIFEST_SCHEMA_VERSION = 1
RUNTIME_ENVIRONMENT_SCHEMA_NAME = "tradeos.research_runtime_environment"
RUNTIME_ENVIRONMENT_SCHEMA_VERSION = 1
RUNTIME_ENVIRONMENT_DISTRIBUTIONS = (
    "alpaca-py",
    "joblib",
    "numpy",
    "pandas",
    "pyarrow",
    "python-dotenv",
    "pytz",
    "scikit-learn",
    "scipy",
    "threadpoolctl",
)
DATASET_PROVENANCE_SCHEMA_NAME = "tradeos.dataset_provenance"
DATASET_PROVENANCE_SCHEMA_VERSION = 1
DATASET_SET_SCHEMA_NAME = "tradeos.research_dataset_set"
DATASET_SET_SCHEMA_VERSION = 1
DATASET_CHECKPOINT_SCHEMA_NAME = "tradeos.dataset_preexecution_checkpoint"
DATASET_CHECKPOINT_SCHEMA_VERSION = 1
RESEARCH_CONFIG_SCHEMA_NAME = "tradeos.research_generation_config"
RESEARCH_CONFIG_SCHEMA_VERSION = 1
RESEARCH_GENERATION_SCHEMA_NAME = "tradeos.research_generation_identity"
RESEARCH_GENERATION_SCHEMA_VERSION = 1
PROMOTION_POINTER_SCHEMA_NAME = "tradeos.research_promotion_pointer"
PROMOTION_POINTER_SCHEMA_VERSION = 1
LATEST_RESEARCH_STATE_SCHEMA_NAME = "tradeos.latest_research_state"
LATEST_RESEARCH_STATE_SCHEMA_VERSION = 1
SELECTION_EVIDENCE_SCHEMA_NAME = "tradeos.backtest_selection_evidence"
SELECTION_EVIDENCE_SCHEMA_VERSION = 1
DRIFT_PROFILE_SCHEMA_NAME = "tradeos.promoted_drift_profile_projection"
DRIFT_PROFILE_SCHEMA_VERSION = 1

CANONICAL_STRATEGY_JSON_COLUMN = "canonical_strategy_json"
CANONICAL_STRATEGY_SHA256_COLUMN = "canonical_strategy_sha256"
MODEL_EXECUTION_EVIDENCE_JSON_COLUMN = "model_execution_evidence_json"
MODEL_EXECUTION_EVIDENCE_SHA256_COLUMN = "model_execution_evidence_sha256"
MODEL_EXECUTION_EVIDENCE_SCHEMA_NAME = "tradeos.backtest_model_execution_evidence"
MODEL_EXECUTION_EVIDENCE_SCHEMA_VERSION = 1

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_HEAD_RE = re.compile(r"^[0-9a-f]{40}$")
_UTC_RE = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?Z$")
_BUNDLE_KEYS = frozenset(
    {
        "schema_name",
        "schema_version",
        "generator",
        "generated_at_utc",
        "source_identity",
        "dataset_provenance",
        "research_config",
        "strategy",
        "model_execution",
        "metrics",
        "approval",
        "stability",
        "regime",
        "final_decision",
        "publication",
        "bundle_sha256",
    }
)


class ResearchBundleValidationError(RuntimeError):
    """Raised when research evidence cannot form one coherent bundle."""


class ResearchPublicationError(RuntimeError):
    """Raised when staged current-state publication fails or cannot roll back."""


@dataclass(frozen=True)
class DriftProfilePolicyV1:
    rejection_rate: float = 0.72
    avg_bar_age_s: float = 240.0
    stale_bar_rate: float = 0.0

    def metric_sources(self) -> dict[str, str]:
        return {
            "buy_signal_rate_per_symbol_per_day": (
                "bundle.metrics.trades_per_day / canonical_strategy.symbol_count"
            ),
            "sell_signal_rate_per_symbol_per_day": (
                "v1 long-only cadence equals buy signal rate"
            ),
            "rejection_rate": "tradeos.drift_profile_policy.v1.rejection_rate",
            "rejection_breakdown": "unavailable in bound promotion evidence",
            "hybrid_branch_participation": (
                "unavailable: branch shares are not bound promotion evidence"
            ),
            "avg_bar_age_s": "tradeos.drift_profile_policy.v1.avg_bar_age_s",
            "stale_bar_rate": "tradeos.drift_profile_policy.v1.stale_bar_rate",
        }


DRIFT_PROFILE_POLICY_V1 = DriftProfilePolicyV1()

RISK_EXECUTION_CONTEXT_FIELDS = (
    "max_usd_per_trade",
    "max_symbol_exposure_usd",
    "max_open_positions",
    "max_daily_loss_usd",
    "max_orders_per_minute",
    "max_price_deviation_bps",
    "max_data_delay_seconds",
    "max_live_price_age_seconds",
)


def validate_risk_execution_context(payload: object) -> dict[str, int | float]:
    """Validate the exact non-strategy limits that materially gate execution."""

    normalized = _normalize_json(payload, field="risk_execution_context")
    if not isinstance(normalized, dict) or set(normalized) != set(
        RISK_EXECUTION_CONTEXT_FIELDS
    ):
        raise ResearchBundleValidationError(
            "Risk execution context fields are not exact."
        )
    for field in (
        "max_usd_per_trade",
        "max_symbol_exposure_usd",
        "max_daily_loss_usd",
    ):
        value = normalized[field]
        if type(value) is not float or not math.isfinite(value) or value <= 0.0:
            raise ResearchBundleValidationError(
                f"Risk execution context {field} must be a finite float > 0.0."
            )
    deviation = normalized["max_price_deviation_bps"]
    if type(deviation) is not float or not math.isfinite(deviation) or deviation < 0.0:
        raise ResearchBundleValidationError(
            "Risk execution context max_price_deviation_bps must be a finite float >= 0.0."
        )
    for field in ("max_open_positions", "max_orders_per_minute"):
        value = normalized[field]
        if type(value) is not int or value < 1:
            raise ResearchBundleValidationError(
                f"Risk execution context {field} must be an integer >= 1."
            )
    for field in ("max_data_delay_seconds", "max_live_price_age_seconds"):
        value = normalized[field]
        if type(value) is not int or value < 0:
            raise ResearchBundleValidationError(
                f"Risk execution context {field} must be an integer >= 0."
            )
    return {field: normalized[field] for field in RISK_EXECUTION_CONTEXT_FIELDS}  # type: ignore[return-value]


def resolve_risk_execution_context(
    runtime: Mapping[str, object],
    environment: Mapping[str, object],
) -> dict[str, int | float]:
    """Resolve runtime JSON over dotenv/process values over code defaults."""

    if not isinstance(runtime, Mapping) or not isinstance(environment, Mapping):
        raise ResearchBundleValidationError(
            "Risk execution resolution requires runtime and environment mappings."
        )

    def env_value(name: str, default: str) -> object:
        value = environment.get(name)
        return default if value is None else value

    base_raw: dict[str, object] = {
        "max_usd_per_trade": env_value("MAX_USD_PER_TRADE", "200"),
        "max_open_positions": env_value("MAX_OPEN_POSITIONS", "3"),
        "max_daily_loss_usd": env_value("MAX_DAILY_LOSS_USD", "300"),
        "max_orders_per_minute": env_value("MAX_ORDERS_PER_MINUTE", "6"),
        "max_price_deviation_bps": env_value("MAX_PRICE_DEVIATION_BPS", "75"),
        "max_data_delay_seconds": env_value("MAX_DATA_DELAY_SECONDS", "300"),
        "max_live_price_age_seconds": env_value(
            "MAX_LIVE_PRICE_AGE_SECONDS",
            "60",
        ),
    }
    base_raw["max_symbol_exposure_usd"] = environment.get(
        "MAX_SYMBOL_EXPOSURE_USD",
        base_raw["max_usd_per_trade"],
    )
    raw = {
        field: (
            runtime[field]
            if field in runtime and runtime[field] is not None
            else base_raw[field]
        )
        for field in RISK_EXECUTION_CONTEXT_FIELDS
    }
    resolved: dict[str, int | float] = {}
    for field in (
        "max_usd_per_trade",
        "max_symbol_exposure_usd",
        "max_daily_loss_usd",
        "max_price_deviation_bps",
    ):
        try:
            resolved[field] = float(raw[field])
        except (TypeError, ValueError, OverflowError) as exc:
            if field == "max_price_deviation_bps" and field not in runtime:
                resolved[field] = 75.0
            else:
                raise ResearchBundleValidationError(
                    f"Effective risk execution value {field} is not numeric."
                ) from exc
    for field in (
        "max_open_positions",
        "max_orders_per_minute",
        "max_data_delay_seconds",
        "max_live_price_age_seconds",
    ):
        try:
            resolved[field] = int(raw[field])
        except (TypeError, ValueError, OverflowError) as exc:
            raise ResearchBundleValidationError(
                f"Effective risk execution value {field} is not an integer."
            ) from exc
    return validate_risk_execution_context(resolved)


def _canonical_json_bytes(value: object) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ResearchBundleValidationError(
            "Research evidence is not finite canonical JSON."
        ) from exc


def _sha256_bytes(contents: bytes) -> str:
    return hashlib.sha256(contents).hexdigest()


def generation_directory_component(generation_sha256: str) -> str:
    """Encode the full 256-bit generation digest as compact URL-safe base64."""

    generation = _require_sha256(
        generation_sha256,
        field="research_generation_sha256",
    )
    return base64.urlsafe_b64encode(bytes.fromhex(generation)).decode("ascii").rstrip("=")


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise ResearchBundleValidationError(f"Unable to hash artifact: {path}") from exc
    return digest.hexdigest()


def _require_sha256(value: object, *, field: str) -> str:
    if type(value) is not str or not _SHA256_RE.fullmatch(value):
        raise ResearchBundleValidationError(f"{field} must be a lowercase SHA-256 digest.")
    return value


def _normalize_json(value: object, *, field: str = "value") -> object:
    """Return plain finite JSON values without silently dropping information."""

    if value is None or type(value) in {str, bool, int}:
        return value
    if type(value) is float:
        if not math.isfinite(value):
            raise ResearchBundleValidationError(f"{field} must be finite.")
        return 0.0 if value == 0.0 else value
    # Pandas/numpy scalar values expose ``item``.  Converting that scalar is a
    # representation normalization, not a strategy/default inference.
    item = getattr(value, "item", None)
    if callable(item):
        try:
            converted = item()
        except (TypeError, ValueError):
            converted = value
        if converted is not value:
            return _normalize_json(converted, field=field)
    if isinstance(value, Mapping):
        normalized: dict[str, object] = {}
        for key, child in value.items():
            if type(key) is not str:
                raise ResearchBundleValidationError(f"{field} keys must be strings.")
            normalized[key] = _normalize_json(child, field=f"{field}.{key}")
        return normalized
    if isinstance(value, (list, tuple)):
        return [
            _normalize_json(child, field=f"{field}[{index}]")
            for index, child in enumerate(value)
        ]
    raise ResearchBundleValidationError(
        f"{field} contains unsupported value type {type(value).__name__}."
    )


def _finite_numeric_float(
    value: object,
    *,
    field: str,
    nonnegative: bool = False,
) -> float:
    if type(value) not in {int, float} or type(value) is bool:
        raise ResearchBundleValidationError(f"{field} must be numeric.")
    try:
        result = float(value)
    except (OverflowError, ValueError) as exc:
        raise ResearchBundleValidationError(
            f"{field} cannot be represented as a finite float."
        ) from exc
    if not math.isfinite(result) or (nonnegative and result < 0):
        qualifier = "finite nonnegative" if nonnegative else "finite"
        raise ResearchBundleValidationError(f"{field} must be {qualifier}.")
    return result


def _reject_duplicate_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ResearchBundleValidationError(f"Duplicate JSON field {key!r}.")
        result[key] = value
    return result


def _reject_nonfinite_json(token: str) -> None:
    raise ResearchBundleValidationError(f"Non-finite JSON number {token!r} is forbidden.")


def strict_json_loads(text: str, *, label: str) -> object:
    if type(text) is not str:
        raise ResearchBundleValidationError(f"{label} must be JSON text.")
    try:
        return json.loads(
            text,
            object_pairs_hook=_reject_duplicate_object,
            parse_constant=_reject_nonfinite_json,
        )
    except json.JSONDecodeError as exc:
        raise ResearchBundleValidationError(f"{label} is malformed JSON: {exc.msg}.") from exc


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _require_utc(value: object, *, field: str) -> str:
    if type(value) is not str or not _UTC_RE.fullmatch(value):
        raise ResearchBundleValidationError(f"{field} must be an ISO-8601 UTC timestamp.")
    try:
        parsed = datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError as exc:
        raise ResearchBundleValidationError(f"{field} is not a valid UTC timestamp.") from exc
    if parsed.tzinfo is None or parsed.utcoffset() != timezone.utc.utcoffset(parsed):
        raise ResearchBundleValidationError(f"{field} must resolve to UTC.")
    return value


def _canonicalize_utc(value: object, *, field: str) -> str:
    if type(value) is not str or not value:
        raise ResearchBundleValidationError(f"{field} must be an ISO-8601 timestamp.")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ResearchBundleValidationError(f"{field} is not a valid timestamp.") from exc
    if parsed.tzinfo is None or parsed.utcoffset() != timedelta(0):
        raise ResearchBundleValidationError(f"{field} must resolve to UTC.")
    return parsed.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def strategy_schema_sha256() -> str:
    descriptor = {
        "schema_name": STRATEGY_SCHEMA_NAME,
        "schema_version": STRATEGY_SCHEMA_VERSION,
        "promoted_fields": list(PROMOTED_STRATEGY_FIELDS),
        "enum_choices": [
            [field, list(choices)] for field, choices in PROMOTED_STRATEGY_ENUM_CHOICES
        ],
        "runtime_fixed_assumptions": {
            "schema_name": RUNTIME_FIXED_ASSUMPTIONS_SCHEMA_NAME,
            "schema_version": RUNTIME_FIXED_ASSUMPTIONS_SCHEMA_VERSION,
            "values": dict(RUNTIME_FIXED_STRATEGY_ASSUMPTIONS),
        },
    }
    return _sha256_bytes(_canonical_json_bytes(descriptor))


def strategy_config_sha256(contract: CanonicalResolvedPromotedStrategy) -> str:
    config_identity = {
        "symbols": list(contract.symbols),
        "promoted_values": contract.promoted_values,
    }
    return _sha256_bytes(_canonical_json_bytes(config_identity))


def build_research_config_identity(
    inputs: Mapping[str, object],
) -> dict[str, object]:
    """Bind the complete declared research/search configuration.

    Dataset content and source code are bound separately.  This identity is
    for the parameters that determine snapshot selection, search grid,
    ranking, stability, and regime policy.
    """

    normalized_inputs = _normalize_json(inputs, field="research_config.inputs")
    if not isinstance(normalized_inputs, dict) or not normalized_inputs:
        raise ResearchBundleValidationError(
            "Research configuration inputs must be a nonempty object."
        )
    _validate_research_config_inputs(normalized_inputs)
    body = {
        "schema_name": RESEARCH_CONFIG_SCHEMA_NAME,
        "schema_version": RESEARCH_CONFIG_SCHEMA_VERSION,
        "inputs": normalized_inputs,
    }
    return {
        **body,
        "config_sha256": _sha256_bytes(_canonical_json_bytes(body)),
    }


def _validate_research_config_inputs(inputs: object) -> dict[str, object]:
    normalized = _normalize_json(inputs, field="research_config.inputs")
    if not isinstance(normalized, dict) or set(normalized) != {
        "dataset_request",
        "search",
        "ranking_policy",
        "approval_policy",
        "stability_policy",
        "regime_policy",
        "evidence_contract",
        "resolved_live_inputs",
    }:
        raise ResearchBundleValidationError(
            "Research configuration input fields are not exact."
        )
    dataset_request = normalized["dataset_request"]
    if not isinstance(dataset_request, dict) or set(dataset_request) != {
        "mode",
        "research_anchor_utc",
        "validation_windows",
        "window_ranges",
        "existing_dataset_reference",
        "snapshot",
    }:
        raise ResearchBundleValidationError("Research dataset request fields are not exact.")
    mode = dataset_request["mode"]
    windows = dataset_request["validation_windows"]
    anchor = _require_utc(
        dataset_request["research_anchor_utc"],
        field="research_config.dataset_request.research_anchor_utc",
    )
    if mode not in {"fresh_snapshots", "existing_dataset"}:
        raise ResearchBundleValidationError("Research dataset request mode is unsupported.")
    if (
        not isinstance(windows, list)
        or not windows
        or any(type(value) is not int or value < 0 for value in windows)
        or len(set(windows)) != len(windows)
    ):
        raise ResearchBundleValidationError(
            "Research validation windows must be nonnegative integers."
        )
    window_ranges = dataset_request["window_ranges"]
    if not isinstance(window_ranges, list) or len(window_ranges) != len(windows):
        raise ResearchBundleValidationError(
            "Research validation window ranges must match ordered lookbacks."
        )
    normalized_ranges: list[tuple[int, str, str]] = []
    for index, window_range in enumerate(window_ranges):
        if not isinstance(window_range, dict) or set(window_range) != {
            "lookback_days",
            "start_utc",
            "end_utc",
        }:
            raise ResearchBundleValidationError(
                "Research validation window range fields are not exact."
            )
        lookback = window_range["lookback_days"]
        if type(lookback) is not int or lookback != windows[index]:
            raise ResearchBundleValidationError(
                "Research window range lookback/order is inconsistent."
            )
        start = _require_utc(
            window_range["start_utc"],
            field=f"research_config.window_ranges[{index}].start_utc",
        )
        end = _require_utc(
            window_range["end_utc"],
            field=f"research_config.window_ranges[{index}].end_utc",
        )
        start_time = datetime.fromisoformat(start[:-1] + "+00:00")
        end_time = datetime.fromisoformat(end[:-1] + "+00:00")
        if start_time >= end_time:
            raise ResearchBundleValidationError(
                "Research window range must have start before end."
            )
        normalized_ranges.append((lookback, start, end))
    snapshot = dataset_request["snapshot"]
    existing_reference = dataset_request["existing_dataset_reference"]
    if mode == "fresh_snapshots":
        if existing_reference is not None or not isinstance(snapshot, dict) or set(snapshot) != {
            "symbols",
            "timeframe",
            "feed",
            "adjustment",
            "align_mode",
        }:
            raise ResearchBundleValidationError(
                "Fresh research snapshot settings are incomplete."
            )
        if windows != sorted(windows) or any(value < 1 for value in windows):
            raise ResearchBundleValidationError(
                "Fresh research validation windows must be positive and increasing."
            )
        anchor_time = datetime.fromisoformat(anchor[:-1] + "+00:00")
        for lookback, start, end in normalized_ranges:
            start_time = datetime.fromisoformat(start[:-1] + "+00:00")
            end_time = datetime.fromisoformat(end[:-1] + "+00:00")
            if end != anchor or end_time - start_time != timedelta(days=lookback):
                raise ResearchBundleValidationError(
                    "Fresh research windows must share the anchor and exact lookback duration."
                )
        if any(
            datetime.fromisoformat(end[:-1] + "+00:00") != anchor_time
            for _lookback, _start, end in normalized_ranges
        ):
            raise ResearchBundleValidationError(
                "Fresh research window ends disagree with the UTC anchor."
            )
        if (
            not isinstance(snapshot["symbols"], list)
            or not snapshot["symbols"]
            or any(type(symbol) is not str or not symbol for symbol in snapshot["symbols"])
            or any(
                type(snapshot[field]) is not str or not snapshot[field]
                for field in ("timeframe", "feed", "adjustment", "align_mode")
            )
        ):
            raise ResearchBundleValidationError(
                "Fresh research snapshot values are invalid."
            )
    elif (
        type(existing_reference) is not str
        or not existing_reference
        or snapshot is not None
        or windows != [0]
    ):
        raise ResearchBundleValidationError(
            "Existing-dataset research request fields are invalid."
        )
    if mode == "existing_dataset":
        _validate_repo_relative_path(
            existing_reference,
            field="research_config.dataset_request.existing_dataset_reference",
        )

    resolved_inputs = normalized["resolved_live_inputs"]
    if not isinstance(resolved_inputs, dict) or set(resolved_inputs) != {
        "symbols",
        "bar_timeframe_minutes",
        "strategy_mode",
        "risk_execution_context",
        "input_sha256",
    }:
        raise ResearchBundleValidationError(
            "Resolved live research input fields are not exact."
        )
    resolved_symbols = resolved_inputs["symbols"]
    if (
        not isinstance(resolved_symbols, list)
        or not resolved_symbols
        or any(type(symbol) is not str or not symbol for symbol in resolved_symbols)
        or len(resolved_symbols) != len(set(resolved_symbols))
        or type(resolved_inputs["bar_timeframe_minutes"]) is not int
        or resolved_inputs["bar_timeframe_minutes"] < 1
        or type(resolved_inputs["strategy_mode"]) is not str
        or not resolved_inputs["strategy_mode"]
    ):
        raise ResearchBundleValidationError("Resolved live research inputs are invalid.")
    validate_risk_execution_context(resolved_inputs["risk_execution_context"])
    resolved_body = {
        key: resolved_inputs[key]
        for key in (
            "symbols",
            "bar_timeframe_minutes",
            "strategy_mode",
            "risk_execution_context",
        )
    }
    if resolved_inputs["input_sha256"] != _sha256_bytes(
        _canonical_json_bytes(resolved_body)
    ):
        raise ResearchBundleValidationError(
            "Resolved live research input hash is inconsistent."
        )

    search = normalized["search"]
    if not isinstance(search, dict) or set(search) != {
        "sweep_parameters",
        "fixed_parameters",
    }:
        raise ResearchBundleValidationError("Research search fields are not exact.")
    for section in ("sweep_parameters", "fixed_parameters"):
        entries = search[section]
        if not isinstance(entries, list) or not entries:
            raise ResearchBundleValidationError(
                f"Research search {section} must be a nonempty ordered list."
            )
        for entry in entries:
            if (
                not isinstance(entry, dict)
                or set(entry) != {"flag", "value"}
                or type(entry["flag"]) is not str
                or not entry["flag"]
                or type(entry["value"]) is not str
            ):
                raise ResearchBundleValidationError(
                    f"Research search {section} entry is invalid."
                )

    strategy_entries = [
        entry
        for entry in search["sweep_parameters"]
        if entry["flag"] == "strategy-mode-list"
    ]
    if (
        len(strategy_entries) != 1
        or strategy_entries[0]["value"] != resolved_inputs["strategy_mode"]
    ):
        raise ResearchBundleValidationError(
            "Research sweep strategy mode contradicts resolved live inputs."
        )
    if mode == "fresh_snapshots" and (
        snapshot["symbols"] != resolved_symbols
        or snapshot["timeframe"]
        != f"{resolved_inputs['bar_timeframe_minutes']}Min"
    ):
        raise ResearchBundleValidationError(
            "Research snapshot settings contradict resolved live inputs."
        )

    ranking = normalized["ranking_policy"]
    if not isinstance(ranking, dict) or set(ranking) != {
        "name",
        "version",
        "combined_score_formula",
        "tie_method",
        "metric_directions",
        "tie_breakers",
    }:
        raise ResearchBundleValidationError("Research ranking policy fields are not exact.")
    if (
        type(ranking["name"]) is not str
        or not ranking["name"]
        or type(ranking["version"]) is not int
        or ranking["version"] < 1
        or type(ranking["combined_score_formula"]) is not str
        or not ranking["combined_score_formula"]
        or ranking["tie_method"] != "min"
        or ranking["tie_breakers"]
        != [
            "canonical_strategy_sha256",
            "model_execution_evidence_sha256",
            "source_row_index",
        ]
        or ranking["metric_directions"]
        != {
            "max_drawdown_pct": "ascending",
            "profit_factor": "descending",
            "total_return_pct": "descending",
        }
    ):
        raise ResearchBundleValidationError("Research ranking policy is invalid.")

    ApprovalPolicy.from_payload(normalized["approval_policy"])
    StabilityPolicy.from_payload(normalized["stability_policy"])
    RegimePolicy.from_payload(normalized["regime_policy"])
    if normalized["evidence_contract"] != {
        "canonical_strategy_json_column": CANONICAL_STRATEGY_JSON_COLUMN,
        "canonical_strategy_sha256_column": CANONICAL_STRATEGY_SHA256_COLUMN,
        "model_execution_evidence_json_column": MODEL_EXECUTION_EVIDENCE_JSON_COLUMN,
        "model_execution_evidence_sha256_column": MODEL_EXECUTION_EVIDENCE_SHA256_COLUMN,
    }:
        raise ResearchBundleValidationError(
            "Research backtest evidence-column contract is unsupported."
        )
    return normalized


def _validate_research_config_identity(payload: object) -> dict[str, object]:
    normalized = _normalize_json(payload, field="research_config")
    if not isinstance(normalized, dict) or set(normalized) != {
        "schema_name",
        "schema_version",
        "inputs",
        "config_sha256",
    }:
        raise ResearchBundleValidationError("Research config identity fields are not exact.")
    if (
        normalized["schema_name"] != RESEARCH_CONFIG_SCHEMA_NAME
        or type(normalized["schema_version"]) is not int
        or normalized["schema_version"] != RESEARCH_CONFIG_SCHEMA_VERSION
        or not isinstance(normalized["inputs"], dict)
        or not normalized["inputs"]
    ):
        raise ResearchBundleValidationError("Research config identity is invalid.")
    _validate_research_config_inputs(normalized["inputs"])
    body = {key: normalized[key] for key in normalized if key != "config_sha256"}
    expected = _sha256_bytes(_canonical_json_bytes(body))
    if normalized["config_sha256"] != expected:
        raise ResearchBundleValidationError("Research config hash is inconsistent.")
    return normalized


@dataclass(frozen=True)
class BacktestCanonicalStrategy:
    """The exact canonical strategy envelope emitted by one backtest row."""

    canonical_json: str
    sha256: str
    contract: CanonicalResolvedPromotedStrategy


def canonical_strategy_from_backtest_row(
    row: Mapping[str, object],
) -> BacktestCanonicalStrategy:
    if not isinstance(row, Mapping):
        raise ResearchBundleValidationError("Backtest result row must be a mapping.")
    raw_json = row.get(CANONICAL_STRATEGY_JSON_COLUMN)
    raw_sha = row.get(CANONICAL_STRATEGY_SHA256_COLUMN)
    if type(raw_json) is not str or not raw_json:
        raise ResearchBundleValidationError(
            f"Backtest row is missing {CANONICAL_STRATEGY_JSON_COLUMN!r}."
        )
    expected_sha = _require_sha256(
        raw_sha,
        field=CANONICAL_STRATEGY_SHA256_COLUMN,
    )
    try:
        contract = CanonicalResolvedPromotedStrategy.from_json(raw_json)
    except ContractValidationError as exc:
        raise ResearchBundleValidationError(
            "Backtest canonical strategy envelope is invalid."
        ) from exc
    canonical_json = contract.canonical_json()
    if raw_json != canonical_json:
        raise ResearchBundleValidationError(
            "Backtest canonical strategy JSON is not in the exact canonical encoding."
        )
    actual_sha = contract.sha256()
    if actual_sha != expected_sha:
        raise ResearchBundleValidationError(
            "Backtest canonical strategy hash does not match its exact envelope."
        )
    return BacktestCanonicalStrategy(raw_json, actual_sha, contract)


@dataclass(frozen=True)
class BacktestModelExecutionEvidence:
    canonical_json: str
    sha256: str
    payload: Mapping[str, object]


def model_execution_evidence_from_backtest_row(
    row: Mapping[str, object],
) -> BacktestModelExecutionEvidence:
    if not isinstance(row, Mapping):
        raise ResearchBundleValidationError("Backtest result row must be a mapping.")
    raw_json = row.get(MODEL_EXECUTION_EVIDENCE_JSON_COLUMN)
    raw_sha = _require_sha256(
        row.get(MODEL_EXECUTION_EVIDENCE_SHA256_COLUMN),
        field=MODEL_EXECUTION_EVIDENCE_SHA256_COLUMN,
    )
    if type(raw_json) is not str:
        raise ResearchBundleValidationError(
            f"Backtest row is missing {MODEL_EXECUTION_EVIDENCE_JSON_COLUMN!r}."
        )
    payload = strict_json_loads(raw_json, label="model execution evidence")
    normalized = _normalize_json(payload, field="model_execution_evidence")
    if not isinstance(normalized, dict) or set(normalized) != {
        "schema_name",
        "schema_version",
        "used",
        "model_path",
        "model_sha256",
        "sidecar_path",
        "sidecar_declared_sha256",
        "sidecar_sha256",
        "feature_columns",
        "training_metadata",
        "training_dataset_identity",
    }:
        raise ResearchBundleValidationError(
            "Model execution evidence fields are not exact."
        )
    if (
        normalized["schema_name"] != MODEL_EXECUTION_EVIDENCE_SCHEMA_NAME
        or type(normalized["schema_version"]) is not int
        or normalized["schema_version"] != MODEL_EXECUTION_EVIDENCE_SCHEMA_VERSION
        or type(normalized["used"]) is not bool
        or not isinstance(normalized["feature_columns"], list)
        or not isinstance(normalized["training_metadata"], dict)
    ):
        raise ResearchBundleValidationError("Model execution evidence schema is invalid.")
    if normalized["used"] is False:
        if any(
            normalized[field] is not None
            for field in (
                "model_path",
                "model_sha256",
                "sidecar_path",
                "sidecar_declared_sha256",
                "sidecar_sha256",
                "training_dataset_identity",
            )
        ) or normalized["feature_columns"] or normalized["training_metadata"]:
            raise ResearchBundleValidationError(
                "Unused model evidence must use explicit empty/null values."
            )
    else:
        for field in ("model_path", "sidecar_path"):
            path = normalized[field]
            _validate_repo_relative_path(path, field=f"model_execution.{field}")
        if normalized["sidecar_path"] != f"{normalized['model_path']}.sha256":
            raise ResearchBundleValidationError(
                "Model sidecar path must identify the executed model sidecar."
            )
        for field in (
            "model_sha256",
            "sidecar_declared_sha256",
            "sidecar_sha256",
        ):
            _require_sha256(normalized[field], field=f"model_execution.{field}")
        if normalized["sidecar_declared_sha256"] != normalized["model_sha256"]:
            raise ResearchBundleValidationError(
                "Model sidecar declaration does not match the executed model content."
            )
        feature_columns = normalized["feature_columns"]
        if (
            not feature_columns
            or any(type(column) is not str or not column for column in feature_columns)
            or len(set(feature_columns)) != len(feature_columns)
        ):
            raise ResearchBundleValidationError(
                "Used model evidence feature_columns are invalid."
            )
        if not normalized["training_metadata"]:
            raise ResearchBundleValidationError(
                "Used model evidence requires explicit training metadata."
            )
        training_dataset = normalized["training_dataset_identity"]
        if not isinstance(training_dataset, dict) or set(training_dataset) != {
            "dataset_reference",
            "dataset_id",
            "manifest_sha256",
            "bars_sha256",
            "snapshot_sha256",
        }:
            raise ResearchBundleValidationError(
                "Used model training dataset identity fields are not exact."
            )
        for field in ("dataset_reference", "dataset_id"):
            if type(training_dataset[field]) is not str or not training_dataset[field]:
                raise ResearchBundleValidationError(
                    f"Model training dataset {field} must be nonblank."
                )
        _validate_repo_relative_path(
            training_dataset["dataset_reference"],
            field="model_execution.training_dataset_identity.dataset_reference",
        )
        if normalized["training_metadata"].get("dataset") not in {
            None,
            training_dataset["dataset_reference"],
        }:
            raise ResearchBundleValidationError(
                "Model training metadata dataset reference is inconsistent."
            )
        for field in ("manifest_sha256", "bars_sha256", "snapshot_sha256"):
            _require_sha256(
                training_dataset[field],
                field=f"model_execution.training_dataset_identity.{field}",
            )
    canonical_json = _canonical_json_bytes(normalized).decode("utf-8")
    if raw_json != canonical_json:
        raise ResearchBundleValidationError(
            "Model execution evidence is not in exact canonical JSON encoding."
        )
    actual_sha = _sha256_bytes(raw_json.encode("utf-8"))
    if raw_sha != actual_sha:
        raise ResearchBundleValidationError(
            "Model execution evidence hash contradicts its exact JSON."
        )
    return BacktestModelExecutionEvidence(raw_json, raw_sha, normalized)


def _validate_selection_evidence(payload: object) -> dict[str, object]:
    normalized = _normalize_json(payload, field="selection_evidence")
    if not isinstance(normalized, dict) or set(normalized) != {
        "schema_name",
        "schema_version",
        "sweep_csv_sha256",
        "leaderboard_csv_sha256",
        "row_count",
        "selected_source_row_index",
        "candidates",
        "evidence_sha256",
    }:
        raise ResearchBundleValidationError("Selection evidence fields are not exact.")
    if (
        normalized["schema_name"] != SELECTION_EVIDENCE_SCHEMA_NAME
        or type(normalized["schema_version"]) is not int
        or normalized["schema_version"] != SELECTION_EVIDENCE_SCHEMA_VERSION
    ):
        raise ResearchBundleValidationError("Selection evidence schema is unsupported.")
    _require_sha256(normalized["sweep_csv_sha256"], field="selection.sweep_csv_sha256")
    _require_sha256(
        normalized["leaderboard_csv_sha256"],
        field="selection.leaderboard_csv_sha256",
    )
    row_count = normalized["row_count"]
    selected_index = normalized["selected_source_row_index"]
    candidates = normalized["candidates"]
    if (
        type(row_count) is not int
        or row_count < 1
        or type(selected_index) is not int
        or not isinstance(candidates, list)
        or len(candidates) != row_count
    ):
        raise ResearchBundleValidationError("Selection evidence cardinality is invalid.")
    normalized_candidates: list[dict[str, object]] = []
    for expected_index, candidate in enumerate(candidates):
        if not isinstance(candidate, dict) or set(candidate) != {
            "source_row_index",
            "total_return_pct",
            "profit_factor",
            "max_drawdown_pct",
            "canonical_strategy_sha256",
            "model_execution_evidence_sha256",
        }:
            raise ResearchBundleValidationError("Selection candidate fields are not exact.")
        if candidate["source_row_index"] != expected_index or type(
            candidate["source_row_index"]
        ) is not int:
            raise ResearchBundleValidationError(
                "Selection candidates must retain exact source-row order."
            )
        normalized_candidates.append(
            {
                "source_row_index": expected_index,
                "total_return_pct": _finite_numeric_float(
                    candidate["total_return_pct"],
                    field=f"selection.candidates[{expected_index}].total_return_pct",
                ),
                "profit_factor": _finite_numeric_float(
                    candidate["profit_factor"],
                    field=f"selection.candidates[{expected_index}].profit_factor",
                    nonnegative=True,
                ),
                "max_drawdown_pct": _finite_numeric_float(
                    candidate["max_drawdown_pct"],
                    field=f"selection.candidates[{expected_index}].max_drawdown_pct",
                    nonnegative=True,
                ),
                "canonical_strategy_sha256": _require_sha256(
                    candidate["canonical_strategy_sha256"],
                    field=(
                        f"selection.candidates[{expected_index}]."
                        "canonical_strategy_sha256"
                    ),
                ),
                "model_execution_evidence_sha256": _require_sha256(
                    candidate["model_execution_evidence_sha256"],
                    field=(
                        f"selection.candidates[{expected_index}]."
                        "model_execution_evidence_sha256"
                    ),
                ),
            }
        )
    rank_keys: list[tuple[float, str, str, int]] = []
    for candidate in normalized_candidates:
        rank_return = 1 + sum(
            other["total_return_pct"] > candidate["total_return_pct"]
            for other in normalized_candidates
        )
        rank_pf = 1 + sum(
            other["profit_factor"] > candidate["profit_factor"]
            for other in normalized_candidates
        )
        rank_drawdown = 1 + sum(
            other["max_drawdown_pct"] < candidate["max_drawdown_pct"]
            for other in normalized_candidates
        )
        combined_score = (rank_return + rank_pf + rank_drawdown) / 3.0
        rank_keys.append(
            (
                combined_score,
                candidate["canonical_strategy_sha256"],  # type: ignore[arg-type]
                candidate["model_execution_evidence_sha256"],  # type: ignore[arg-type]
                candidate["source_row_index"],  # type: ignore[arg-type]
            )
        )
    expected_selected = min(range(row_count), key=lambda index: rank_keys[index])
    if selected_index != expected_selected:
        raise ResearchBundleValidationError(
            "Selected backtest row is not the deterministic ranking winner."
        )
    body = {
        key: normalized[key] for key in normalized if key != "evidence_sha256"
    }
    # Candidate numeric representations are canonicalized to floats above.
    body["candidates"] = normalized_candidates
    expected_hash = _sha256_bytes(_canonical_json_bytes(body))
    if normalized["evidence_sha256"] != expected_hash:
        raise ResearchBundleValidationError("Selection evidence SHA-256 is inconsistent.")
    return {**body, "evidence_sha256": expected_hash}


def build_backtest_selection_evidence(
    *,
    sweep_csv_sha256: str,
    leaderboard_csv_sha256: str,
    selected_source_row_index: int,
    candidates: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    body: dict[str, object] = {
        "schema_name": SELECTION_EVIDENCE_SCHEMA_NAME,
        "schema_version": SELECTION_EVIDENCE_SCHEMA_VERSION,
        "sweep_csv_sha256": sweep_csv_sha256,
        "leaderboard_csv_sha256": leaderboard_csv_sha256,
        "row_count": len(candidates),
        "selected_source_row_index": selected_source_row_index,
        "candidates": list(candidates),
    }
    payload = {
        **body,
        "evidence_sha256": _sha256_bytes(_canonical_json_bytes(body)),
    }
    return _validate_selection_evidence(payload)


def _csv_rows(contents: bytes, *, label: str) -> tuple[list[str], list[dict[str, str]]]:
    try:
        text = contents.decode("utf-8-sig", errors="strict")
    except UnicodeError as exc:
        raise ResearchBundleValidationError(f"{label} CSV is not valid UTF-8.") from exc
    reader = csv.DictReader(io.StringIO(text, newline=""), strict=True)
    fields = reader.fieldnames
    if (
        not isinstance(fields, list)
        or not fields
        or any(type(field) is not str or not field for field in fields)
        or len(fields) != len(set(fields))
    ):
        raise ResearchBundleValidationError(
            f"{label} CSV headers must be nonempty and unique."
        )
    try:
        rows = list(reader)
    except csv.Error as exc:
        raise ResearchBundleValidationError(f"{label} CSV is malformed.") from exc
    if any(None in row or any(value is None for value in row.values()) for row in rows):
        raise ResearchBundleValidationError(f"{label} CSV row widths are inconsistent.")
    return fields, rows  # type: ignore[return-value]


def _csv_float(value: object, *, field: str, nonnegative: bool = False) -> float:
    if type(value) is not str or not value.strip():
        raise ResearchBundleValidationError(f"{field} is missing from CSV evidence.")
    try:
        parsed = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ResearchBundleValidationError(f"{field} is not a finite CSV number.") from exc
    return _finite_numeric_float(parsed, field=field, nonnegative=nonnegative)


def validate_backtest_selection_artifacts(
    selection_evidence: Mapping[str, object],
    *,
    sweep_csv_bytes: bytes,
    leaderboard_csv_bytes: bytes,
) -> dict[str, object]:
    """Tie canonical selection evidence to the exact sweep and leaderboard bytes."""

    if not isinstance(sweep_csv_bytes, bytes) or not isinstance(
        leaderboard_csv_bytes,
        bytes,
    ):
        raise ResearchBundleValidationError("Selection artifacts must be exact bytes.")
    selection = _validate_selection_evidence(selection_evidence)
    if (
        selection["sweep_csv_sha256"] != _sha256_bytes(sweep_csv_bytes)
        or selection["leaderboard_csv_sha256"]
        != _sha256_bytes(leaderboard_csv_bytes)
    ):
        raise ResearchBundleValidationError(
            "Selection evidence hashes do not match exact CSV bytes."
        )
    required_evidence = {
        CANONICAL_STRATEGY_JSON_COLUMN,
        CANONICAL_STRATEGY_SHA256_COLUMN,
        MODEL_EXECUTION_EVIDENCE_JSON_COLUMN,
        MODEL_EXECUTION_EVIDENCE_SHA256_COLUMN,
        "total_return_pct",
        "profit_factor",
        "max_drawdown_pct",
    }
    sweep_fields, sweep_rows = _csv_rows(sweep_csv_bytes, label="sweep")
    if not required_evidence <= set(sweep_fields):
        raise ResearchBundleValidationError(
            "Sweep CSV is missing canonical selection evidence columns."
        )
    candidates = selection["candidates"]
    assert isinstance(candidates, list)
    if len(sweep_rows) != selection["row_count"]:
        raise ResearchBundleValidationError(
            "Sweep CSV row count contradicts selection evidence."
        )
    observed_candidates: list[dict[str, object]] = []
    for source_index, row in enumerate(sweep_rows):
        canonical = canonical_strategy_from_backtest_row(row)
        model = model_execution_evidence_from_backtest_row(row)
        observed_candidates.append(
            {
                "source_row_index": source_index,
                "total_return_pct": _csv_float(
                    row["total_return_pct"],
                    field=f"sweep[{source_index}].total_return_pct",
                ),
                "profit_factor": _csv_float(
                    row["profit_factor"],
                    field=f"sweep[{source_index}].profit_factor",
                    nonnegative=True,
                ),
                "max_drawdown_pct": _csv_float(
                    row["max_drawdown_pct"],
                    field=f"sweep[{source_index}].max_drawdown_pct",
                    nonnegative=True,
                ),
                "canonical_strategy_sha256": canonical.sha256,
                "model_execution_evidence_sha256": model.sha256,
            }
        )
    if observed_candidates != candidates:
        raise ResearchBundleValidationError(
            "Selection candidates contradict the exact sweep CSV rows."
        )

    leaderboard_fields, leaderboard_rows = _csv_rows(
        leaderboard_csv_bytes,
        label="leaderboard",
    )
    required_leaderboard = required_evidence | {
        "source_row_index",
        "rank_return",
        "rank_pf",
        "rank_drawdown",
        "combined_score",
    }
    if not required_leaderboard <= set(leaderboard_fields):
        raise ResearchBundleValidationError(
            "Leaderboard CSV is missing deterministic ranking evidence columns."
        )
    if len(leaderboard_rows) != len(observed_candidates):
        raise ResearchBundleValidationError(
            "Leaderboard row count contradicts the sweep CSV."
        )
    rank_evidence: dict[int, tuple[float, float, float, float]] = {}
    ranking_keys: dict[int, tuple[float, str, str, int]] = {}
    for candidate in observed_candidates:
        index = candidate["source_row_index"]
        assert isinstance(index, int)
        rank_return = float(
            1
            + sum(
                other["total_return_pct"] > candidate["total_return_pct"]
                for other in observed_candidates
            )
        )
        rank_pf = float(
            1
            + sum(
                other["profit_factor"] > candidate["profit_factor"]
                for other in observed_candidates
            )
        )
        rank_drawdown = float(
            1
            + sum(
                other["max_drawdown_pct"] < candidate["max_drawdown_pct"]
                for other in observed_candidates
            )
        )
        combined = (rank_return + rank_pf + rank_drawdown) / 3.0
        rank_evidence[index] = (rank_return, rank_pf, rank_drawdown, combined)
        ranking_keys[index] = (
            combined,
            candidate["canonical_strategy_sha256"],  # type: ignore[arg-type]
            candidate["model_execution_evidence_sha256"],  # type: ignore[arg-type]
            index,
        )
    expected_order = sorted(ranking_keys, key=ranking_keys.__getitem__)
    observed_order: list[int] = []
    for position, row in enumerate(leaderboard_rows):
        raw_index = row["source_row_index"]
        if type(raw_index) is not str or not re.fullmatch(r"\d+", raw_index.strip()):
            raise ResearchBundleValidationError(
                f"leaderboard[{position}].source_row_index is invalid."
            )
        source_index = int(raw_index)
        if source_index not in rank_evidence or source_index in observed_order:
            raise ResearchBundleValidationError(
                "Leaderboard source-row identities are invalid or duplicated."
            )
        observed_order.append(source_index)
        candidate = observed_candidates[source_index]
        canonical = canonical_strategy_from_backtest_row(row)
        model = model_execution_evidence_from_backtest_row(row)
        expected_ranks = rank_evidence[source_index]
        observed_ranks = (
            _csv_float(row["rank_return"], field=f"leaderboard[{position}].rank_return"),
            _csv_float(row["rank_pf"], field=f"leaderboard[{position}].rank_pf"),
            _csv_float(
                row["rank_drawdown"],
                field=f"leaderboard[{position}].rank_drawdown",
            ),
            _csv_float(
                row["combined_score"],
                field=f"leaderboard[{position}].combined_score",
            ),
        )
        observed_metrics = (
            _csv_float(
                row["total_return_pct"],
                field=f"leaderboard[{position}].total_return_pct",
            ),
            _csv_float(
                row["profit_factor"],
                field=f"leaderboard[{position}].profit_factor",
                nonnegative=True,
            ),
            _csv_float(
                row["max_drawdown_pct"],
                field=f"leaderboard[{position}].max_drawdown_pct",
                nonnegative=True,
            ),
        )
        if (
            canonical.sha256 != candidate["canonical_strategy_sha256"]
            or model.sha256 != candidate["model_execution_evidence_sha256"]
            or observed_metrics
            != (
                candidate["total_return_pct"],
                candidate["profit_factor"],
                candidate["max_drawdown_pct"],
            )
            or any(
                not math.isclose(observed, expected, rel_tol=0.0, abs_tol=1e-12)
                for observed, expected in zip(observed_ranks, expected_ranks, strict=True)
            )
        ):
            raise ResearchBundleValidationError(
                "Leaderboard row contradicts sweep-derived deterministic ranking."
            )
    if (
        observed_order != expected_order
        or observed_order[0] != selection["selected_source_row_index"]
    ):
        raise ResearchBundleValidationError(
            "Leaderboard ordering contradicts deterministic selection evidence."
        )
    return selection


@dataclass(frozen=True)
class ApprovalPolicy:
    name: str
    version: int
    min_profit_factor: float
    max_drawdown_pct: float
    min_total_return_pct: float
    min_trades_per_day: float

    def __post_init__(self) -> None:
        if type(self.name) is not str or not self.name.strip():
            raise ResearchBundleValidationError("Approval policy name must be nonblank.")
        if type(self.version) is not int or self.version < 1:
            raise ResearchBundleValidationError("Approval policy version must be a positive integer.")
        for field_name in (
            "min_profit_factor",
            "max_drawdown_pct",
            "min_total_return_pct",
            "min_trades_per_day",
        ):
            value = getattr(self, field_name)
            if type(value) is not float or not math.isfinite(value):
                raise ResearchBundleValidationError(
                    f"Approval threshold {field_name} must be a finite float."
                )

    def to_payload(self) -> dict[str, object]:
        return {
            "name": self.name,
            "version": self.version,
            "thresholds": {
                "min_profit_factor": self.min_profit_factor,
                "max_drawdown_pct": self.max_drawdown_pct,
                "min_total_return_pct": self.min_total_return_pct,
                "min_trades_per_day": self.min_trades_per_day,
            },
        }

    @classmethod
    def from_payload(cls, payload: object) -> ApprovalPolicy:
        if not isinstance(payload, Mapping):
            raise ResearchBundleValidationError("Approval policy must be an object.")
        if set(payload) != {"name", "version", "thresholds"}:
            raise ResearchBundleValidationError("Approval policy fields are not exact.")
        thresholds = payload["thresholds"]
        if not isinstance(thresholds, Mapping) or set(thresholds) != {
            "min_profit_factor",
            "max_drawdown_pct",
            "min_total_return_pct",
            "min_trades_per_day",
        }:
            raise ResearchBundleValidationError("Approval policy thresholds are not exact.")
        return cls(
            name=payload["name"],  # type: ignore[arg-type]
            version=payload["version"],  # type: ignore[arg-type]
            min_profit_factor=thresholds["min_profit_factor"],  # type: ignore[arg-type]
            max_drawdown_pct=thresholds["max_drawdown_pct"],  # type: ignore[arg-type]
            min_total_return_pct=thresholds["min_total_return_pct"],  # type: ignore[arg-type]
            min_trades_per_day=thresholds["min_trades_per_day"],  # type: ignore[arg-type]
        )


@dataclass(frozen=True)
class ApprovalOutcome:
    approved: bool
    rejection_reasons: tuple[str, ...]


@dataclass(frozen=True)
class StabilityPolicy:
    name: str
    version: int
    min_profit_factor: float
    max_drawdown_pct: float
    minimum_window_count: int
    required_lookback_days: tuple[int, ...]
    require_consistent_canonical_strategy: bool = True
    require_consistent_model_execution: bool = True

    def __post_init__(self) -> None:
        if type(self.name) is not str or not self.name.strip():
            raise ResearchBundleValidationError("Stability policy name must be nonblank.")
        if type(self.version) is not int or self.version < 1:
            raise ResearchBundleValidationError(
                "Stability policy version must be a positive integer."
            )
        if type(self.min_profit_factor) is not float or not math.isfinite(
            self.min_profit_factor
        ):
            raise ResearchBundleValidationError(
                "Stability min_profit_factor must be a finite float."
            )
        if type(self.max_drawdown_pct) is not float or not math.isfinite(
            self.max_drawdown_pct
        ):
            raise ResearchBundleValidationError(
                "Stability max_drawdown_pct must be a finite float."
            )
        if type(self.minimum_window_count) is not int or self.minimum_window_count < 2:
            raise ResearchBundleValidationError(
                "Stability minimum_window_count must be at least two."
            )
        if (
            not isinstance(self.required_lookback_days, tuple)
            or len(self.required_lookback_days) < self.minimum_window_count
            or any(
                type(value) is not int or value < 1
                for value in self.required_lookback_days
            )
            or tuple(sorted(set(self.required_lookback_days)))
            != self.required_lookback_days
        ):
            raise ResearchBundleValidationError(
                "Stability required lookbacks must be unique, positive, and increasing."
            )
        if type(self.require_consistent_canonical_strategy) is not bool:
            raise ResearchBundleValidationError(
                "Stability canonical-consistency requirement must be boolean."
            )
        if type(self.require_consistent_model_execution) is not bool:
            raise ResearchBundleValidationError(
                "Stability model-consistency requirement must be boolean."
            )

    def to_payload(self) -> dict[str, object]:
        return {
            "name": self.name,
            "version": self.version,
            "thresholds": {
                "min_profit_factor": self.min_profit_factor,
                "max_drawdown_pct": self.max_drawdown_pct,
            },
            "minimum_window_count": self.minimum_window_count,
            "required_lookback_days": list(self.required_lookback_days),
            "require_consistent_canonical_strategy": (
                self.require_consistent_canonical_strategy
            ),
            "require_consistent_model_execution": self.require_consistent_model_execution,
        }

    @classmethod
    def from_payload(cls, payload: object) -> StabilityPolicy:
        if not isinstance(payload, Mapping) or set(payload) != {
            "name",
            "version",
            "thresholds",
            "minimum_window_count",
            "required_lookback_days",
            "require_consistent_canonical_strategy",
            "require_consistent_model_execution",
        }:
            raise ResearchBundleValidationError("Stability policy fields are not exact.")
        thresholds = payload["thresholds"]
        if not isinstance(thresholds, Mapping) or set(thresholds) != {
            "min_profit_factor",
            "max_drawdown_pct",
        }:
            raise ResearchBundleValidationError(
                "Stability policy thresholds are not exact."
            )
        required_lookbacks = payload["required_lookback_days"]
        if not isinstance(required_lookbacks, list):
            raise ResearchBundleValidationError(
                "Stability required_lookback_days must be a list."
            )
        return cls(
            name=payload["name"],  # type: ignore[arg-type]
            version=payload["version"],  # type: ignore[arg-type]
            min_profit_factor=thresholds["min_profit_factor"],  # type: ignore[arg-type]
            max_drawdown_pct=thresholds["max_drawdown_pct"],  # type: ignore[arg-type]
            minimum_window_count=payload["minimum_window_count"],  # type: ignore[arg-type]
            required_lookback_days=tuple(required_lookbacks),
            require_consistent_canonical_strategy=payload[
                "require_consistent_canonical_strategy"
            ],  # type: ignore[arg-type]
            require_consistent_model_execution=payload[
                "require_consistent_model_execution"
            ],  # type: ignore[arg-type]
        )


@dataclass(frozen=True)
class RegimePolicy:
    name: str
    version: int
    volatility_period_bars: int
    trend_sma_period_bars: int
    high_volatility_threshold: float
    slope_threshold: float
    max_data_age_hours: float
    disable_on: tuple[str, ...]

    def __post_init__(self) -> None:
        if type(self.name) is not str or not self.name.strip():
            raise ResearchBundleValidationError("Regime policy name must be nonblank.")
        if type(self.version) is not int or self.version < 1:
            raise ResearchBundleValidationError(
                "Regime policy version must be a positive integer."
            )
        for field_name in ("volatility_period_bars", "trend_sma_period_bars"):
            value = getattr(self, field_name)
            if type(value) is not int or value < 1:
                raise ResearchBundleValidationError(
                    f"Regime policy {field_name} must be a positive integer."
                )
        for field_name in (
            "high_volatility_threshold",
            "slope_threshold",
            "max_data_age_hours",
        ):
            value = getattr(self, field_name)
            if (
                type(value) is not float
                or not math.isfinite(value)
                or value < 0
                or (field_name == "max_data_age_hours" and value == 0)
            ):
                raise ResearchBundleValidationError(
                    f"Regime policy {field_name} must be a finite nonnegative float."
                )
        if (
            not isinstance(self.disable_on, tuple)
            or any(type(value) is not str or not value for value in self.disable_on)
            or tuple(sorted(set(self.disable_on))) != self.disable_on
        ):
            raise ResearchBundleValidationError(
                "Regime disable_on labels must be unique and sorted."
            )
        if not {"unknown", "stale"} <= set(self.disable_on):
            raise ResearchBundleValidationError(
                "Regime policy must fail closed by disabling unknown and stale regimes."
            )

    def to_payload(self) -> dict[str, object]:
        return {
            "name": self.name,
            "version": self.version,
            "classifier": {
                "volatility_period_bars": self.volatility_period_bars,
                "trend_sma_period_bars": self.trend_sma_period_bars,
                "high_volatility_threshold": self.high_volatility_threshold,
                "slope_threshold": self.slope_threshold,
                "max_data_age_hours": self.max_data_age_hours,
            },
            "disable_on": list(self.disable_on),
        }

    @classmethod
    def from_payload(cls, payload: object) -> RegimePolicy:
        if not isinstance(payload, Mapping) or set(payload) != {
            "name",
            "version",
            "classifier",
            "disable_on",
        }:
            raise ResearchBundleValidationError("Regime policy fields are not exact.")
        classifier = payload["classifier"]
        if not isinstance(classifier, Mapping) or set(classifier) != {
            "volatility_period_bars",
            "trend_sma_period_bars",
            "high_volatility_threshold",
            "slope_threshold",
            "max_data_age_hours",
        }:
            raise ResearchBundleValidationError(
                "Regime classifier policy fields are not exact."
            )
        disable_on = payload["disable_on"]
        if not isinstance(disable_on, list):
            raise ResearchBundleValidationError("Regime disable_on must be a list.")
        return cls(
            name=payload["name"],  # type: ignore[arg-type]
            version=payload["version"],  # type: ignore[arg-type]
            volatility_period_bars=classifier["volatility_period_bars"],  # type: ignore[arg-type]
            trend_sma_period_bars=classifier["trend_sma_period_bars"],  # type: ignore[arg-type]
            high_volatility_threshold=classifier["high_volatility_threshold"],  # type: ignore[arg-type]
            slope_threshold=classifier["slope_threshold"],  # type: ignore[arg-type]
            max_data_age_hours=classifier["max_data_age_hours"],  # type: ignore[arg-type]
            disable_on=tuple(disable_on),
        )


_APPROVAL_METRICS = (
    "profit_factor",
    "max_drawdown_pct",
    "total_return_pct",
    "trades_per_day",
)


def normalize_metrics(metrics: Mapping[str, object]) -> dict[str, float]:
    if not isinstance(metrics, Mapping):
        raise ResearchBundleValidationError("metrics must be an object mapping.")
    normalized: dict[str, float] = {}
    for key in sorted(metrics):
        if type(key) is not str:
            raise ResearchBundleValidationError("Metric names must be strings.")
        raw = _normalize_json(metrics[key], field=f"metrics.{key}")
        if type(raw) not in {int, float} or type(raw) is bool:
            raise ResearchBundleValidationError(f"Metric {key!r} must be numeric.")
        try:
            value = float(raw)
        except (OverflowError, ValueError) as exc:
            raise ResearchBundleValidationError(
                f"Metric {key!r} cannot be represented as a finite float."
            ) from exc
        if not math.isfinite(value):
            raise ResearchBundleValidationError(f"Metric {key!r} must be finite.")
        normalized[key] = 0.0 if value == 0.0 else value
    missing = [name for name in _APPROVAL_METRICS if name not in normalized]
    if missing:
        raise ResearchBundleValidationError(
            f"Approval metrics are incomplete: missing={missing!r}."
        )
    for name in ("profit_factor", "max_drawdown_pct", "trades_per_day"):
        if normalized[name] < 0:
            raise ResearchBundleValidationError(
                f"Metric {name!r} must be nonnegative."
            )
    if "win_rate" in normalized and not 0.0 <= normalized["win_rate"] <= 100.0:
        raise ResearchBundleValidationError(
            "Metric 'win_rate' must be between 0 and 100 inclusive."
        )
    return normalized


def evaluate_approval_policy(
    metrics: Mapping[str, object],
    policy: ApprovalPolicy,
) -> ApprovalOutcome:
    values = normalize_metrics(metrics)
    checks = (
        (
            values["profit_factor"] >= policy.min_profit_factor,
            f"profit_factor {values['profit_factor']:.4f} >= {policy.min_profit_factor}",
        ),
        (
            values["max_drawdown_pct"] <= policy.max_drawdown_pct,
            f"max_drawdown_pct {values['max_drawdown_pct']:.4f}% <= {policy.max_drawdown_pct}%",
        ),
        (
            values["total_return_pct"] >= policy.min_total_return_pct,
            f"total_return_pct {values['total_return_pct']:.4f}% >= {policy.min_total_return_pct}%",
        ),
        (
            values["trades_per_day"] >= policy.min_trades_per_day,
            f"trades_per_day {values['trades_per_day']:.4f} >= {policy.min_trades_per_day}",
        ),
    )
    reasons = tuple(description for passed, description in checks if not passed)
    return ApprovalOutcome(approved=not reasons, rejection_reasons=reasons)


_STABILITY_AVERAGE_METRICS = (
    "total_return_pct",
    "profit_factor",
    "max_drawdown_pct",
    "sharpe_ratio",
    "win_rate",
    "trades_per_day",
    "realized_pnl",
)


def evaluate_stability_policy(
    windows: Sequence[Mapping[str, object]],
    policy: StabilityPolicy,
) -> dict[str, object]:
    """Recompute the complete stability verdict from bound window evidence."""

    if not isinstance(windows, Sequence) or isinstance(windows, (str, bytes)) or not windows:
        raise ResearchBundleValidationError("Stability requires at least one window.")
    normalized_windows: list[dict[str, object]] = []
    reasons: list[str] = []
    hashes: list[str] = []
    model_hashes: list[str] = []
    for index, raw_window in enumerate(windows):
        if not isinstance(raw_window, Mapping) or set(raw_window) != {
            "lookback_days",
            "canonical_strategy_sha256",
            "dataset_provenance_sha256",
            "model_execution_evidence_json",
            "model_execution_evidence_sha256",
            "selection_evidence",
            "metrics",
        }:
            raise ResearchBundleValidationError(
                "Stability window fields must be exact."
            )
        lookback = raw_window["lookback_days"]
        if type(lookback) is not int or lookback < 0:
            raise ResearchBundleValidationError(
                f"Stability window {index} has an invalid lookback."
            )
        digest = _require_sha256(
            raw_window["canonical_strategy_sha256"],
            field=f"stability.windows[{index}].canonical_strategy_sha256",
        )
        dataset_digest = _require_sha256(
            raw_window["dataset_provenance_sha256"],
            field=f"stability.windows[{index}].dataset_provenance_sha256",
        )
        model_evidence = model_execution_evidence_from_backtest_row(raw_window)
        model_hashes.append(model_evidence.sha256)
        selection_evidence = _validate_selection_evidence(
            raw_window["selection_evidence"]
        )
        metrics = normalize_metrics(raw_window["metrics"])  # type: ignore[arg-type]
        hashes.append(digest)
        normalized_windows.append(
            {
                "lookback_days": lookback,
                "canonical_strategy_sha256": digest,
                "dataset_provenance_sha256": dataset_digest,
                "model_execution_evidence_json": model_evidence.canonical_json,
                "model_execution_evidence_sha256": model_evidence.sha256,
                "selection_evidence": selection_evidence,
                "metrics": metrics,
            }
        )
        if metrics["profit_factor"] < policy.min_profit_factor:
            reasons.append(
                f"{lookback}d window: profit_factor {metrics['profit_factor']:.4f}"
                f" < {policy.min_profit_factor}"
            )
        if metrics["max_drawdown_pct"] > policy.max_drawdown_pct:
            reasons.append(
                f"{lookback}d window: max_drawdown_pct "
                f"{metrics['max_drawdown_pct']:.4f}% > {policy.max_drawdown_pct}%"
            )

    lookbacks = tuple(window["lookback_days"] for window in normalized_windows)
    if len(normalized_windows) < policy.minimum_window_count:
        reasons.append(
            f"Stability requires at least {policy.minimum_window_count} validation windows; "
            f"observed {len(normalized_windows)}"
        )
    if lookbacks != policy.required_lookback_days:
        reasons.append(
            "Stability lookback set/order differs from policy: "
            f"expected={list(policy.required_lookback_days)!r}, observed={list(lookbacks)!r}"
        )

    params_consistent = len(set(hashes)) == 1
    if policy.require_consistent_canonical_strategy and not params_consistent:
        reasons.append(
            "Canonical strategy identity differs across validation windows: "
            + ", ".join(
                f"{window['lookback_days']}d={window['canonical_strategy_sha256']}"
                for window in normalized_windows
            )
        )

    model_execution_consistent = len(set(model_hashes)) == 1
    if policy.require_consistent_model_execution and not model_execution_consistent:
        reasons.append(
            "Model execution evidence differs across validation windows: "
            + ", ".join(
                f"{window['lookback_days']}d={window['model_execution_evidence_sha256']}"
                for window in normalized_windows
            )
        )

    averages: dict[str, float] = {}
    for metric_name in _STABILITY_AVERAGE_METRICS:
        values = [
            window["metrics"][metric_name]  # type: ignore[index]
            for window in normalized_windows
            if metric_name in window["metrics"]  # type: ignore[operator]
        ]
        if values:
            averages[metric_name] = round(sum(values) / len(values), 6)

    return {
        "policy": policy.to_payload(),
        "stable": not reasons,
        "reasons": reasons,
        "params_consistent": params_consistent,
        "canonical_strategy_hashes": hashes,
        "model_execution_consistent": model_execution_consistent,
        "model_execution_evidence_hashes": model_hashes,
        "inconsistent_params": (
            {} if params_consistent else {"canonical_strategy_sha256": hashes}
        ),
        "average_metrics": averages,
        "windows": normalized_windows,
    }


def _run_git(repo_root: Path, *arguments: str) -> bytes:
    completed = subprocess.run(
        ["git", *arguments],
        cwd=repo_root,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if completed.returncode != 0:
        detail = completed.stderr.decode("utf-8", errors="replace").strip()
        raise ResearchBundleValidationError(
            f"git {' '.join(arguments)} failed while capturing source identity: {detail}"
        )
    return completed.stdout


def _validate_runtime_environment(payload: object) -> dict[str, object]:
    normalized = _normalize_json(payload, field="runtime_environment")
    if not isinstance(normalized, dict) or set(normalized) != {
        "schema_name",
        "schema_version",
        "python",
        "platform",
        "distributions",
        "environment_sha256",
    }:
        raise ResearchBundleValidationError(
            "Runtime environment identity fields are not exact."
        )
    if (
        normalized["schema_name"] != RUNTIME_ENVIRONMENT_SCHEMA_NAME
        or type(normalized["schema_version"]) is not int
        or normalized["schema_version"] != RUNTIME_ENVIRONMENT_SCHEMA_VERSION
    ):
        raise ResearchBundleValidationError(
            "Runtime environment identity schema is unsupported."
        )
    python_identity = normalized["python"]
    if not isinstance(python_identity, dict) or set(python_identity) != {
        "implementation",
        "version",
        "cache_tag",
    }:
        raise ResearchBundleValidationError(
            "Runtime Python identity fields are not exact."
        )
    if any(
        type(python_identity[field]) is not str or not python_identity[field]
        for field in ("implementation", "version", "cache_tag")
    ):
        raise ResearchBundleValidationError("Runtime Python identity is invalid.")
    platform_identity = normalized["platform"]
    if not isinstance(platform_identity, dict) or set(platform_identity) != {
        "system",
        "machine",
    }:
        raise ResearchBundleValidationError(
            "Runtime platform identity fields are not exact."
        )
    if any(
        type(platform_identity[field]) is not str or not platform_identity[field]
        for field in ("system", "machine")
    ):
        raise ResearchBundleValidationError("Runtime platform identity is invalid.")
    distributions = normalized["distributions"]
    if (
        not isinstance(distributions, list)
        or len(distributions) != len(RUNTIME_ENVIRONMENT_DISTRIBUTIONS)
    ):
        raise ResearchBundleValidationError(
            "Runtime distribution identity set is incomplete."
        )
    for expected_name, distribution in zip(
        RUNTIME_ENVIRONMENT_DISTRIBUTIONS,
        distributions,
        strict=True,
    ):
        if (
            not isinstance(distribution, dict)
            or set(distribution) != {"name", "version"}
            or distribution["name"] != expected_name
            or type(distribution["version"]) is not str
            or not distribution["version"]
        ):
            raise ResearchBundleValidationError(
                "Runtime distribution identities must be complete and ordered."
            )
    body = {
        key: normalized[key]
        for key in normalized
        if key != "environment_sha256"
    }
    if normalized["environment_sha256"] != _sha256_bytes(
        _canonical_json_bytes(body)
    ):
        raise ResearchBundleValidationError(
            "Runtime environment identity hash is inconsistent."
        )
    return normalized


def capture_runtime_environment() -> dict[str, object]:
    """Capture the exact behavior-relevant interpreter and dependency set."""

    cache_tag = getattr(sys.implementation, "cache_tag", None)
    distributions: list[dict[str, str]] = []
    for name in RUNTIME_ENVIRONMENT_DISTRIBUTIONS:
        try:
            version = importlib_metadata.version(name)
        except importlib_metadata.PackageNotFoundError as exc:
            raise ResearchBundleValidationError(
                f"Required research distribution is not installed: {name}"
            ) from exc
        if type(version) is not str or not version:
            raise ResearchBundleValidationError(
                f"Required research distribution has no exact version: {name}"
            )
        distributions.append({"name": name, "version": version})
    body: dict[str, object] = {
        "schema_name": RUNTIME_ENVIRONMENT_SCHEMA_NAME,
        "schema_version": RUNTIME_ENVIRONMENT_SCHEMA_VERSION,
        "python": {
            "implementation": platform.python_implementation(),
            "version": platform.python_version(),
            "cache_tag": cache_tag,
        },
        "platform": {
            "system": platform.system(),
            "machine": platform.machine(),
        },
        "distributions": distributions,
    }
    return _validate_runtime_environment(
        {
            **body,
            "environment_sha256": _sha256_bytes(_canonical_json_bytes(body)),
        }
    )


def capture_source_identity(repo_root: Path) -> dict[str, object]:
    """Bind HEAD plus all versionable Python/dependency source contents."""

    root = Path(repo_root).resolve(strict=True)
    head = _run_git(root, "rev-parse", "--verify", "HEAD").decode("ascii").strip().lower()
    if not _HEAD_RE.fullmatch(head):
        raise ResearchBundleValidationError("git returned an invalid HEAD commit identity.")
    branch = _run_git(root, "rev-parse", "--abbrev-ref", "HEAD").decode(
        "utf-8", errors="strict"
    ).strip()
    dirty = bool(_run_git(root, "status", "--porcelain"))
    raw_paths = _run_git(
        root,
        "ls-files",
        "-z",
        "--cached",
        "--others",
        "--exclude-standard",
        "--",
        "*.py",
        "requirements*.txt",
        "pyproject.toml",
    )
    paths: list[str] = []
    for encoded in raw_paths.split(b"\0"):
        if not encoded:
            continue
        relative = encoded.decode("utf-8", errors="surrogateescape").replace("\\", "/")
        paths.append(
            _validate_repo_relative_path(
                relative,
                field="source_identity.content_manifest.path",
            )
        )
    # The offline ML payload is execution input even when ignored by git.
    # Include both content and integrity sidecars in the same source identity.
    models_dir = root / "ml" / "models"
    if models_dir.is_dir():
        for model_path in models_dir.rglob("*"):
            if model_path.is_file():
                paths.append(model_path.relative_to(root).as_posix())
    paths = sorted(set(paths))
    if not paths:
        raise ResearchBundleValidationError("No versionable source files were found.")
    files: list[dict[str, str]] = []
    for relative in paths:
        path = root / PurePosixPath(relative)
        if not path.is_file():
            raise ResearchBundleValidationError(f"Versionable source file is missing: {relative}")
        files.append({"path": relative, "sha256": _file_sha256(path)})
    manifest_payload = {
        "schema_name": SOURCE_MANIFEST_SCHEMA_NAME,
        "schema_version": SOURCE_MANIFEST_SCHEMA_VERSION,
        "files": files,
    }
    return {
        "git_commit": head,
        "git_branch": branch,
        "repository_dirty": dirty,
        "content_manifest": manifest_payload,
        "content_sha256": _sha256_bytes(_canonical_json_bytes(manifest_payload)),
        "runtime_environment": capture_runtime_environment(),
    }


def assert_source_identity_unchanged(
    expected: Mapping[str, object],
    observed: Mapping[str, object],
) -> None:
    """Fail if executable source/model identity changed during research.

    Repository dirtiness is informative status rather than content identity;
    immutable bundle creation can change it in a clean checkout.  Commit,
    branch, the complete source/model manifest, and its digest must stay exact.
    """

    before = _validate_source_identity(expected)
    after = _validate_source_identity(observed)
    identity_fields = (
        "git_commit",
        "git_branch",
        "content_manifest",
        "content_sha256",
        "runtime_environment",
    )
    if any(before[field] != after[field] for field in identity_fields):
        raise ResearchBundleValidationError(
            "Executable source or model identity changed during the research run."
        )


def _timeframe_minutes(value: object) -> int:
    if type(value) is not str or not value.strip():
        raise ResearchBundleValidationError("Dataset manifest timeframe is missing or invalid.")
    normalized = value.strip().lower()
    match = re.fullmatch(r"(\d+)\s*(min|m|minute|minutes|h|hour|hours)", normalized)
    if not match:
        raise ResearchBundleValidationError(f"Unsupported dataset timeframe {value!r}.")
    quantity = int(match.group(1))
    if quantity < 1:
        raise ResearchBundleValidationError("Dataset timeframe must be positive.")
    return quantity * 60 if match.group(2) in {"h", "hour", "hours"} else quantity


def capture_dataset_provenance(
    dataset_path: Path,
    contract: CanonicalResolvedPromotedStrategy,
    *,
    require_versioned_hashes: bool = False,
) -> dict[str, object]:
    """Capture observed hashes while retaining absent legacy declarations as null."""

    root = Path(dataset_path).resolve(strict=True)
    if not root.is_dir():
        raise ResearchBundleValidationError(f"Dataset path is not a directory: {root}")
    manifest_path = root / "manifest.json"
    bars_path = root / "bars.parquet"
    if not manifest_path.is_file() or not bars_path.is_file():
        raise ResearchBundleValidationError(
            "Dataset promotion requires both manifest.json and bars.parquet."
        )
    try:
        manifest_text = manifest_path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ResearchBundleValidationError("Dataset manifest is unreadable.") from exc
    manifest = strict_json_loads(manifest_text, label="dataset manifest")
    if not isinstance(manifest, Mapping):
        raise ResearchBundleValidationError("Dataset manifest must contain an object.")
    # Reuse the repository's integrity verifier for its bars/snapshot semantics.
    try:
        verified_manifest = load_verified_manifest(root)
    except RuntimeError as exc:
        raise ResearchBundleValidationError("Dataset integrity verification failed.") from exc
    if dict(manifest) != verified_manifest:
        raise ResearchBundleValidationError(
            "Strict dataset manifest parsing disagrees with integrity verification."
        )

    manifest_symbols = manifest.get("symbols")
    if not isinstance(manifest_symbols, list) or any(type(item) is not str for item in manifest_symbols):
        raise ResearchBundleValidationError(
            "Dataset manifest must explicitly declare its ordered symbol universe."
        )
    if tuple(manifest_symbols) != contract.symbols:
        raise ResearchBundleValidationError(
            "Dataset manifest symbols do not exactly match the canonical strategy universe."
        )
    timeframe = manifest.get("timeframe")
    timeframe_minutes = _timeframe_minutes(timeframe)
    if timeframe_minutes != contract.promoted_values["bar_timeframe_minutes"]:
        raise ResearchBundleValidationError(
            "Dataset timeframe does not match canonical bar_timeframe_minutes."
        )

    observed_bars_hash = _file_sha256(bars_path)
    declared_bars_hash = manifest.get("bars_sha256")
    declared_snapshot_hash = manifest.get("snapshot_sha256")
    if (declared_bars_hash is None) != (declared_snapshot_hash is None):
        raise ResearchBundleValidationError(
            "Dataset manifest must declare bars_sha256 and snapshot_sha256 together."
        )
    if require_versioned_hashes and declared_bars_hash is None:
        raise ResearchBundleValidationError(
            "Promotable research requires versioned dataset bars_sha256 and snapshot_sha256; "
            "legacy observed-only provenance is nonpromotable."
        )
    if declared_bars_hash is not None:
        declared_bars_hash = _require_sha256(declared_bars_hash, field="manifest.bars_sha256")
        if declared_bars_hash != observed_bars_hash:
            raise ResearchBundleValidationError(
                "Dataset manifest bars_sha256 contradicts observed bars content."
            )
    snapshot_body = {
        key: value for key, value in manifest.items() if key != "snapshot_sha256"
    }
    observed_snapshot_hash = _sha256_bytes(_canonical_json_bytes(snapshot_body))
    if declared_snapshot_hash is not None:
        declared_snapshot_hash = _require_sha256(
            declared_snapshot_hash,
            field="manifest.snapshot_sha256",
        )
        if declared_snapshot_hash != observed_snapshot_hash:
            raise ResearchBundleValidationError(
                "Dataset manifest snapshot_sha256 contradicts its canonical fingerprint."
            )

    declared = {
        # Missing legacy fields stay null.  Do not infer them from directory
        # names, timestamps, current source, or observed hashes.
        "schema_version": manifest.get("schema_version"),
        "dataset_id": manifest.get("dataset_id"),
        "created_at_utc": manifest.get("created_at_utc"),
        "code_hash": manifest.get("code_hash"),
        "bars_sha256": declared_bars_hash,
        "snapshot_sha256": declared_snapshot_hash,
    }
    observed = {
        "directory_name": root.name,
        "manifest_sha256": _file_sha256(manifest_path),
        "bars_sha256": observed_bars_hash,
        "snapshot_sha256": observed_snapshot_hash,
        "symbols": list(contract.symbols),
        "timeframe": timeframe,
        "bar_timeframe_minutes": contract.promoted_values["bar_timeframe_minutes"],
        "start_utc": _canonicalize_utc(
            manifest.get("start_utc", manifest.get("start")),
            field="manifest.start_utc",
        ),
        "end_utc": _canonicalize_utc(
            manifest.get("end_utc", manifest.get("end")),
            field="manifest.end_utc",
        ),
        "feed": manifest.get("feed"),
        "adjustment": manifest.get("adjustment"),
        "align_mode": manifest.get("align_mode"),
        "row_count": manifest.get("row_count"),
    }
    body = {
        "schema_name": DATASET_PROVENANCE_SCHEMA_NAME,
        "schema_version": DATASET_PROVENANCE_SCHEMA_VERSION,
        "declared": _normalize_json(declared, field="dataset.declared"),
        "observed": _normalize_json(observed, field="dataset.observed"),
        "versioned_hashes_verified": declared_bars_hash is not None,
    }
    return {
        **body,
        "provenance_sha256": _sha256_bytes(_canonical_json_bytes(body)),
    }


def capture_dataset_checkpoint(dataset_path: Path) -> dict[str, object]:
    """Capture immutable dataset bytes before a backtest can consume them."""

    root = Path(dataset_path).resolve(strict=True)
    manifest_path = root / "manifest.json"
    bars_path = root / "bars.parquet"
    if not manifest_path.is_file() or not bars_path.is_file():
        raise ResearchBundleValidationError(
            "Dataset checkpoint requires manifest.json and bars.parquet."
        )
    try:
        manifest_text = manifest_path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ResearchBundleValidationError("Dataset manifest is unreadable.") from exc
    manifest = strict_json_loads(manifest_text, label="dataset checkpoint manifest")
    if not isinstance(manifest, Mapping):
        raise ResearchBundleValidationError("Dataset checkpoint manifest must be an object.")
    try:
        verified = load_verified_manifest(root)
    except RuntimeError as exc:
        raise ResearchBundleValidationError(
            "Dataset checkpoint integrity verification failed."
        ) from exc
    if dict(manifest) != verified:
        raise ResearchBundleValidationError(
            "Dataset checkpoint strict parsing disagrees with integrity verification."
        )
    snapshot_body = {
        key: value for key, value in manifest.items() if key != "snapshot_sha256"
    }
    return {
        "schema_name": DATASET_CHECKPOINT_SCHEMA_NAME,
        "schema_version": DATASET_CHECKPOINT_SCHEMA_VERSION,
        "manifest_sha256": _file_sha256(manifest_path),
        "bars_sha256": _file_sha256(bars_path),
        "snapshot_sha256": _sha256_bytes(_canonical_json_bytes(snapshot_body)),
    }


def assert_dataset_checkpoint_unchanged(
    expected: Mapping[str, object],
    observed: Mapping[str, object],
) -> None:
    exact_keys = {
        "schema_name",
        "schema_version",
        "manifest_sha256",
        "bars_sha256",
        "snapshot_sha256",
    }
    before = _normalize_json(expected, field="dataset_checkpoint.expected")
    after = _normalize_json(observed, field="dataset_checkpoint.observed")
    if (
        not isinstance(before, dict)
        or not isinstance(after, dict)
        or set(before) != exact_keys
        or set(after) != exact_keys
        or before["schema_name"] != DATASET_CHECKPOINT_SCHEMA_NAME
        or type(before["schema_version"]) is not int
        or before["schema_version"] != DATASET_CHECKPOINT_SCHEMA_VERSION
        or after["schema_name"] != DATASET_CHECKPOINT_SCHEMA_NAME
        or type(after["schema_version"]) is not int
        or after["schema_version"] != DATASET_CHECKPOINT_SCHEMA_VERSION
    ):
        raise ResearchBundleValidationError("Dataset checkpoint schema is invalid.")
    for checkpoint in (before, after):
        for field in ("manifest_sha256", "bars_sha256", "snapshot_sha256"):
            _require_sha256(checkpoint[field], field=f"dataset_checkpoint.{field}")
    if before != after:
        raise ResearchBundleValidationError(
            "Dataset content changed during the research run."
        )


def build_dataset_provenance_set(
    windows: Sequence[Mapping[str, object]],
    *,
    primary_window_index: int,
) -> dict[str, object]:
    """Bind ordered validation-window datasets into one content identity."""

    if not isinstance(windows, Sequence) or isinstance(windows, (str, bytes)) or not windows:
        raise ResearchBundleValidationError(
            "Research dataset provenance requires at least one window."
        )
    if (
        type(primary_window_index) is not int
        or primary_window_index < 0
        or primary_window_index >= len(windows)
    ):
        raise ResearchBundleValidationError("Primary dataset window index is invalid.")
    normalized_windows: list[dict[str, object]] = []
    for raw_window in windows:
        if not isinstance(raw_window, Mapping) or set(raw_window) != {
            "lookback_days",
            "provenance",
        }:
            raise ResearchBundleValidationError(
                "Dataset-window provenance fields must be exact."
            )
        lookback = raw_window["lookback_days"]
        if type(lookback) is not int or lookback < 0:
            raise ResearchBundleValidationError(
                "Dataset-window lookback_days must be a nonnegative integer."
            )
        normalized_windows.append(
            {
                "lookback_days": lookback,
                "provenance": _validate_dataset_provenance(
                    raw_window["provenance"]
                ),
            }
        )
    body = {
        "schema_name": DATASET_SET_SCHEMA_NAME,
        "schema_version": DATASET_SET_SCHEMA_VERSION,
        "primary_window_index": primary_window_index,
        "windows": normalized_windows,
    }
    return {
        **body,
        "provenance_sha256": _sha256_bytes(_canonical_json_bytes(body)),
    }


def build_live_config_payload(
    base_payload: Mapping[str, object],
    canonical_strategy: BacktestCanonicalStrategy,
    *,
    risk_execution_context: Mapping[str, object],
    source_updates: Mapping[str, object],
) -> dict[str, object]:
    """Merge canonical strategy keys while preserving non-contract runtime data."""

    normalized_base = _normalize_json(base_payload, field="live_config")
    if not isinstance(normalized_base, dict):
        raise ResearchBundleValidationError("Live config must contain a top-level object.")
    runtime = normalized_base.get("runtime")
    if not isinstance(runtime, dict):
        raise ResearchBundleValidationError("Live config must contain a runtime object.")
    raw_excluded = runtime.get("excluded_symbols")
    if raw_excluded is None:
        excluded_symbols: list[str] = []
    elif isinstance(raw_excluded, str):
        excluded_symbols = [
            value.strip().upper()
            for value in raw_excluded.split(",")
            if value.strip()
        ]
    elif isinstance(raw_excluded, list) and all(
        type(value) is str for value in raw_excluded
    ):
        excluded_symbols = [value.strip().upper() for value in raw_excluded if value.strip()]
    else:
        raise ResearchBundleValidationError(
            "Live config excluded_symbols must be a string or string list."
        )
    if len(excluded_symbols) != len(set(excluded_symbols)):
        raise ResearchBundleValidationError(
            "Live config excluded_symbols must be unique after normalization."
        )
    exclusion_overlap = set(excluded_symbols) & set(canonical_strategy.contract.symbols)
    if exclusion_overlap:
        raise ResearchBundleValidationError(
            "Preserved live-config exclusions would remove canonical active symbols: "
            f"{sorted(exclusion_overlap)!r}."
        )
    for field_name in PROMOTED_STRATEGY_FIELDS:
        runtime.pop(field_name, None)
    runtime["symbols"] = list(canonical_strategy.contract.symbols)
    # Preserve legitimate inactive exclusions as normalized, explicit runtime
    # metadata.  Any overlap with the canonical active universe failed above.
    runtime["excluded_symbols"] = excluded_symbols
    for field_name, value in canonical_strategy.contract.promoted_values.items():
        runtime[field_name] = value
    resolved_risk_context = validate_risk_execution_context(
        risk_execution_context
    )
    for field_name, value in resolved_risk_context.items():
        runtime[field_name] = value

    existing_source = normalized_base.get("source")
    if existing_source is None:
        existing_source = {}
    if not isinstance(existing_source, dict):
        raise ResearchBundleValidationError("Live config source must be an object when present.")
    source = dict(existing_source)
    normalized_updates = _normalize_json(source_updates, field="live_config.source_updates")
    if not isinstance(normalized_updates, dict):
        raise ResearchBundleValidationError("Live config source updates must be an object.")
    source.update(normalized_updates)
    source.update(
        {
            "canonical_strategy_schema_name": STRATEGY_SCHEMA_NAME,
            "canonical_strategy_schema_version": STRATEGY_SCHEMA_VERSION,
            "canonical_strategy_sha256": canonical_strategy.sha256,
            "canonical_strategy_config_sha256": strategy_config_sha256(
                canonical_strategy.contract
            ),
        }
    )
    normalized_base["source"] = source
    return normalized_base


def json_file_bytes(payload: Mapping[str, object]) -> bytes:
    normalized = _normalize_json(payload, field="artifact")
    if not isinstance(normalized, dict):
        raise ResearchBundleValidationError("Artifact must contain an object.")
    try:
        text = json.dumps(
            normalized,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise ResearchBundleValidationError("Artifact is not serializable JSON.") from exc
    return (text + "\n").encode("utf-8")


def _validate_source_identity(payload: object) -> dict[str, object]:
    normalized = _normalize_json(payload, field="source_identity")
    if not isinstance(normalized, dict) or set(normalized) != {
        "git_commit",
        "git_branch",
        "repository_dirty",
        "content_manifest",
        "content_sha256",
        "runtime_environment",
    }:
        raise ResearchBundleValidationError("Source identity fields are not exact.")
    if type(normalized["git_commit"]) is not str or not _HEAD_RE.fullmatch(normalized["git_commit"]):
        raise ResearchBundleValidationError("Source git_commit is invalid.")
    if type(normalized["git_branch"]) is not str or not normalized["git_branch"]:
        raise ResearchBundleValidationError("Source git_branch is invalid.")
    if type(normalized["repository_dirty"]) is not bool:
        raise ResearchBundleValidationError("Source repository_dirty must be boolean.")
    manifest = normalized["content_manifest"]
    if not isinstance(manifest, dict) or set(manifest) != {
        "schema_name",
        "schema_version",
        "files",
    }:
        raise ResearchBundleValidationError("Source content manifest is invalid.")
    if (
        manifest["schema_name"] != SOURCE_MANIFEST_SCHEMA_NAME
        or type(manifest["schema_version"]) is not int
        or manifest["schema_version"] != SOURCE_MANIFEST_SCHEMA_VERSION
    ):
        raise ResearchBundleValidationError("Source content manifest schema is unsupported.")
    files = manifest["files"]
    if not isinstance(files, list) or not files:
        raise ResearchBundleValidationError("Source content manifest files are missing.")
    previous = ""
    for entry in files:
        if not isinstance(entry, dict) or set(entry) != {"path", "sha256"}:
            raise ResearchBundleValidationError("Source content file identity is invalid.")
        path = entry["path"]
        if type(path) is not str or not path or path <= previous:
            raise ResearchBundleValidationError("Source content paths must be unique and sorted.")
        _validate_repo_relative_path(path, field="source_identity.content_manifest.path")
        previous = path
        _require_sha256(entry["sha256"], field=f"source file {path}")
    expected = _sha256_bytes(_canonical_json_bytes(manifest))
    if normalized["content_sha256"] != expected:
        raise ResearchBundleValidationError("Source content hash contradicts its manifest.")
    normalized["runtime_environment"] = _validate_runtime_environment(
        normalized["runtime_environment"]
    )
    return normalized


def _validate_dataset_provenance(payload: object) -> dict[str, object]:
    normalized = _normalize_json(payload, field="dataset_provenance")
    if not isinstance(normalized, dict) or set(normalized) != {
        "schema_name",
        "schema_version",
        "declared",
        "observed",
        "versioned_hashes_verified",
        "provenance_sha256",
    }:
        raise ResearchBundleValidationError("Dataset provenance fields are not exact.")
    if (
        normalized["schema_name"] != DATASET_PROVENANCE_SCHEMA_NAME
        or type(normalized["schema_version"]) is not int
        or normalized["schema_version"] != DATASET_PROVENANCE_SCHEMA_VERSION
    ):
        raise ResearchBundleValidationError("Dataset provenance schema is unsupported.")
    body = {key: normalized[key] for key in normalized if key != "provenance_sha256"}
    expected = _sha256_bytes(_canonical_json_bytes(body))
    if normalized["provenance_sha256"] != expected:
        raise ResearchBundleValidationError("Dataset provenance hash is inconsistent.")
    observed = normalized["observed"]
    declared = normalized["declared"]
    if not isinstance(observed, dict) or not isinstance(declared, dict):
        raise ResearchBundleValidationError("Dataset provenance sections must be objects.")
    if set(declared) != {
        "schema_version",
        "dataset_id",
        "created_at_utc",
        "code_hash",
        "bars_sha256",
        "snapshot_sha256",
    }:
        raise ResearchBundleValidationError("Declared dataset provenance fields are not exact.")
    if set(observed) != {
        "directory_name",
        "manifest_sha256",
        "bars_sha256",
        "snapshot_sha256",
        "symbols",
        "timeframe",
        "bar_timeframe_minutes",
        "start_utc",
        "end_utc",
        "feed",
        "adjustment",
        "align_mode",
        "row_count",
    }:
        raise ResearchBundleValidationError("Observed dataset provenance fields are not exact.")
    if type(normalized["versioned_hashes_verified"]) is not bool:
        raise ResearchBundleValidationError(
            "Dataset versioned_hashes_verified must be boolean."
        )
    for key in ("manifest_sha256", "bars_sha256", "snapshot_sha256"):
        _require_sha256(observed.get(key), field=f"dataset.observed.{key}")
    declared_bars = declared.get("bars_sha256")
    declared_snapshot = declared.get("snapshot_sha256")
    if (declared_bars is None) != (declared_snapshot is None):
        raise ResearchBundleValidationError(
            "Declared dataset bars and snapshot hashes must be present together."
        )
    hashes_verified = declared_bars is not None
    if hashes_verified:
        _require_sha256(declared_bars, field="dataset.declared.bars_sha256")
        _require_sha256(declared_snapshot, field="dataset.declared.snapshot_sha256")
        if declared_bars != observed["bars_sha256"]:
            raise ResearchBundleValidationError(
                "Declared and observed dataset bars hashes disagree."
            )
        if declared_snapshot != observed["snapshot_sha256"]:
            raise ResearchBundleValidationError(
                "Declared and observed dataset snapshot hashes disagree."
            )
    if normalized["versioned_hashes_verified"] is not hashes_verified:
        raise ResearchBundleValidationError(
            "Dataset versioned hash verification status is inconsistent."
        )
    return normalized


def _validate_dataset_provenance_set(payload: object) -> dict[str, object]:
    normalized = _normalize_json(payload, field="dataset_provenance")
    if not isinstance(normalized, dict) or set(normalized) != {
        "schema_name",
        "schema_version",
        "primary_window_index",
        "windows",
        "provenance_sha256",
    }:
        raise ResearchBundleValidationError(
            "Research dataset-set provenance fields are not exact."
        )
    if (
        normalized["schema_name"] != DATASET_SET_SCHEMA_NAME
        or type(normalized["schema_version"]) is not int
        or normalized["schema_version"] != DATASET_SET_SCHEMA_VERSION
    ):
        raise ResearchBundleValidationError(
            "Research dataset-set provenance schema is unsupported."
        )
    windows = normalized["windows"]
    primary_index = normalized["primary_window_index"]
    if (
        not isinstance(windows, list)
        or not windows
        or type(primary_index) is not int
        or primary_index < 0
        or primary_index >= len(windows)
    ):
        raise ResearchBundleValidationError(
            "Research dataset-set primary window is invalid."
        )
    for window in windows:
        if not isinstance(window, dict) or set(window) != {
            "lookback_days",
            "provenance",
        }:
            raise ResearchBundleValidationError(
                "Research dataset-window provenance fields are not exact."
            )
        if type(window["lookback_days"]) is not int or window["lookback_days"] < 0:
            raise ResearchBundleValidationError(
                "Research dataset-window lookback is invalid."
            )
        _validate_dataset_provenance(window["provenance"])
    body = {key: normalized[key] for key in normalized if key != "provenance_sha256"}
    expected = _sha256_bytes(_canonical_json_bytes(body))
    if normalized["provenance_sha256"] != expected:
        raise ResearchBundleValidationError(
            "Research dataset-set provenance hash is inconsistent."
        )
    return normalized


def compute_research_generation_sha256(
    *,
    generated_at_utc: str,
    source_content_sha256: str,
    runtime_environment_sha256: str,
    dataset_provenance_sha256: str,
    research_config_sha256: str,
    canonical_strategy_sha256: str,
    model_execution_evidence_sha256: str,
    metrics: Mapping[str, object],
    approval: Mapping[str, object],
    stability: Mapping[str, object],
    regime: Mapping[str, object],
    final_decision: Mapping[str, object],
) -> str:
    """Derive the shared pre-publication generation token without recursion."""

    body = {
        "schema_name": RESEARCH_GENERATION_SCHEMA_NAME,
        "schema_version": RESEARCH_GENERATION_SCHEMA_VERSION,
        "generated_at_utc": _require_utc(
            generated_at_utc,
            field="generation.generated_at_utc",
        ),
        "source_content_sha256": _require_sha256(
            source_content_sha256,
            field="generation.source_content_sha256",
        ),
        "runtime_environment_sha256": _require_sha256(
            runtime_environment_sha256,
            field="generation.runtime_environment_sha256",
        ),
        "dataset_provenance_sha256": _require_sha256(
            dataset_provenance_sha256,
            field="generation.dataset_provenance_sha256",
        ),
        "research_config_sha256": _require_sha256(
            research_config_sha256,
            field="generation.research_config_sha256",
        ),
        "canonical_strategy_sha256": _require_sha256(
            canonical_strategy_sha256,
            field="generation.canonical_strategy_sha256",
        ),
        "model_execution_evidence_sha256": _require_sha256(
            model_execution_evidence_sha256,
            field="generation.model_execution_evidence_sha256",
        ),
        "metrics": normalize_metrics(metrics),
        "approval": _normalize_json(approval, field="generation.approval"),
        "stability": _normalize_json(stability, field="generation.stability"),
        "regime": _normalize_json(regime, field="generation.regime"),
        "final_decision": _normalize_json(
            final_decision,
            field="generation.final_decision",
        ),
    }
    return _sha256_bytes(_canonical_json_bytes(body))


def _validate_bundle_payload(payload: Mapping[str, object]) -> dict[str, object]:
    normalized = _normalize_json(payload, field="bundle")
    if not isinstance(normalized, dict):
        raise ResearchBundleValidationError("Bundle must contain an object.")
    if set(normalized) != _BUNDLE_KEYS or len(normalized) != len(_BUNDLE_KEYS):
        raise ResearchBundleValidationError("Research bundle fields are not exact.")
    if (
        normalized["schema_name"] != BUNDLE_SCHEMA_NAME
        or type(normalized["schema_version"]) is not int
        or normalized["schema_version"] != BUNDLE_SCHEMA_VERSION
    ):
        raise ResearchBundleValidationError("Research bundle schema is unsupported.")
    generator = normalized["generator"]
    if (
        not isinstance(generator, dict)
        or set(generator) != {"name", "version"}
        or generator["name"] != GENERATOR_NAME
        or type(generator["version"]) is not int
        or generator["version"] != GENERATOR_VERSION
    ):
        raise ResearchBundleValidationError("Research bundle generator identity is unsupported.")
    _require_utc(normalized["generated_at_utc"], field="generated_at_utc")
    _validate_source_identity(normalized["source_identity"])
    dataset_provenance = _validate_dataset_provenance_set(
        normalized["dataset_provenance"]
    )
    research_config = _validate_research_config_identity(normalized["research_config"])

    strategy = normalized["strategy"]
    if not isinstance(strategy, dict) or set(strategy) != {
        "canonical_json",
        "canonical_sha256",
        "strategy_config_sha256",
        "schema_sha256",
    }:
        raise ResearchBundleValidationError("Bundle strategy identity fields are not exact.")
    canonical_json = strategy["canonical_json"]
    if type(canonical_json) is not str:
        raise ResearchBundleValidationError("Bundle canonical strategy must be JSON text.")
    try:
        contract = CanonicalResolvedPromotedStrategy.from_json(canonical_json)
    except ContractValidationError as exc:
        raise ResearchBundleValidationError("Bundle canonical strategy is invalid.") from exc
    if canonical_json != contract.canonical_json():
        raise ResearchBundleValidationError("Bundle canonical strategy encoding is not exact.")
    if strategy["canonical_sha256"] != contract.sha256():
        raise ResearchBundleValidationError("Bundle canonical strategy hash is inconsistent.")
    if strategy["strategy_config_sha256"] != strategy_config_sha256(contract):
        raise ResearchBundleValidationError("Bundle strategy config hash is inconsistent.")
    if strategy["schema_sha256"] != strategy_schema_sha256():
        raise ResearchBundleValidationError("Bundle strategy schema hash is inconsistent.")
    model_execution = normalized["model_execution"]
    if not isinstance(model_execution, dict) or set(model_execution) != {
        "canonical_json",
        "sha256",
    }:
        raise ResearchBundleValidationError(
            "Bundle model execution identity fields are not exact."
        )
    model_evidence = model_execution_evidence_from_backtest_row(
        {
            MODEL_EXECUTION_EVIDENCE_JSON_COLUMN: model_execution["canonical_json"],
            MODEL_EXECUTION_EVIDENCE_SHA256_COLUMN: model_execution["sha256"],
        }
    )
    symbol_modes = contract.promoted_values["symbol_strategy_modes"]
    if not isinstance(symbol_modes, dict):
        raise ResearchBundleValidationError(
            "Canonical symbol strategy modes are not an object."
        )
    expects_model = any(mode in {"ml", "hybrid"} for mode in symbol_modes.values())
    if model_evidence.payload["used"] is not expects_model:
        raise ResearchBundleValidationError(
            "Model execution evidence contradicts the canonical strategy modes."
        )
    source_manifest = normalized["source_identity"]["content_manifest"]  # type: ignore[index]
    assert isinstance(source_manifest, dict)
    source_files = source_manifest["files"]
    assert isinstance(source_files, list)
    source_hashes = {
        entry["path"]: entry["sha256"]
        for entry in source_files
        if isinstance(entry, dict)
    }
    if expects_model:
        if (
            source_hashes.get(model_evidence.payload["model_path"])
            != model_evidence.payload["model_sha256"]
            or source_hashes.get(model_evidence.payload["sidecar_path"])
            != model_evidence.payload["sidecar_sha256"]
        ):
            raise ResearchBundleValidationError(
                "Executed model content contradicts the captured source/model manifest."
            )
    dataset_windows = dataset_provenance["windows"]
    assert isinstance(dataset_windows, list)
    for dataset_window in dataset_windows:
        assert isinstance(dataset_window, dict)
        provenance = _validate_dataset_provenance(dataset_window["provenance"])
        observed_dataset = provenance["observed"]
        assert isinstance(observed_dataset, dict)
        if observed_dataset["symbols"] != list(contract.symbols):
            raise ResearchBundleValidationError(
                "Bundle dataset symbol universe contradicts the canonical strategy."
            )
        if (
            observed_dataset["bar_timeframe_minutes"]
            != contract.promoted_values["bar_timeframe_minutes"]
        ):
            raise ResearchBundleValidationError(
                "Bundle dataset timeframe contradicts the canonical strategy."
            )

    metrics = normalize_metrics(normalized["metrics"])  # type: ignore[arg-type]
    approval = normalized["approval"]
    if not isinstance(approval, dict) or set(approval) != {
        "policy",
        "approved",
        "rejection_reasons",
    }:
        raise ResearchBundleValidationError("Bundle approval fields are not exact.")
    policy = ApprovalPolicy.from_payload(approval["policy"])
    expected_outcome = evaluate_approval_policy(metrics, policy)
    reasons = approval["rejection_reasons"]
    if type(approval["approved"]) is not bool or not isinstance(reasons, list) or any(type(item) is not str for item in reasons):
        raise ResearchBundleValidationError("Bundle approval outcome types are invalid.")
    if approval["approved"] != expected_outcome.approved or tuple(reasons) != expected_outcome.rejection_reasons:
        raise ResearchBundleValidationError(
            "Approval outcome contradicts the bound metrics and policy."
        )

    stability = normalized["stability"]
    regime = normalized["regime"]
    decision = normalized["final_decision"]
    if not isinstance(stability, dict) or set(stability) != {
        "policy",
        "stable",
        "reasons",
        "params_consistent",
        "canonical_strategy_hashes",
        "model_execution_consistent",
        "model_execution_evidence_hashes",
        "inconsistent_params",
        "average_metrics",
        "windows",
    }:
        raise ResearchBundleValidationError("Bundle stability result is invalid.")
    stability_policy = StabilityPolicy.from_payload(stability["policy"])
    stability_windows = stability["windows"]
    if not isinstance(stability_windows, list):
        raise ResearchBundleValidationError("Bundle stability windows must be a list.")
    expected_stability = evaluate_stability_policy(
        stability_windows,
        stability_policy,
    )
    if stability != expected_stability:
        raise ResearchBundleValidationError(
            "Bundle stability outcome contradicts its bound window evidence."
        )
    for stability_window in stability_windows:
        assert isinstance(stability_window, dict)
        window_model = model_execution_evidence_from_backtest_row(stability_window)
        selection = _validate_selection_evidence(
            stability_window["selection_evidence"]
        )
        candidates = selection["candidates"]
        selected_index = selection["selected_source_row_index"]
        assert isinstance(candidates, list) and isinstance(selected_index, int)
        selected_candidate = candidates[selected_index]
        assert isinstance(selected_candidate, dict)
        window_metrics = stability_window["metrics"]
        assert isinstance(window_metrics, dict)
        if (
            selected_candidate["canonical_strategy_sha256"]
            != stability_window["canonical_strategy_sha256"]
            or selected_candidate["model_execution_evidence_sha256"]
            != window_model.sha256
            or any(
                selected_candidate[field] != window_metrics[field]
                for field in (
                    "total_return_pct",
                    "profit_factor",
                    "max_drawdown_pct",
                )
            )
        ):
            raise ResearchBundleValidationError(
                "Stability window does not match its deterministic selected candidate."
            )
        if window_model.payload["used"] is not expects_model:
            raise ResearchBundleValidationError(
                "A stability window's model evidence contradicts canonical modes."
            )
        if expects_model and (
            source_hashes.get(window_model.payload["model_path"])
            != window_model.payload["model_sha256"]
            or source_hashes.get(window_model.payload["sidecar_path"])
            != window_model.payload["sidecar_sha256"]
        ):
            raise ResearchBundleValidationError(
                "A stability window's model evidence contradicts captured model content."
            )
        if expects_model:
            metadata = window_model.payload["training_metadata"]
            training_dataset = window_model.payload["training_dataset_identity"]
            assert isinstance(metadata, Mapping)
            assert isinstance(training_dataset, Mapping)
            if (
                "dataset" in metadata
                and metadata["dataset"] != training_dataset["dataset_reference"]
            ):
                raise ResearchBundleValidationError(
                    "Model training metadata dataset reference is inconsistent."
                )
    if len(dataset_windows) != len(stability_windows):
        raise ResearchBundleValidationError(
            "Stability windows and dataset provenance windows differ in count."
        )
    for dataset_window, stability_window in zip(
        dataset_windows,
        stability_windows,
        strict=True,
    ):
        assert isinstance(dataset_window, dict)
        assert isinstance(stability_window, dict)
        provenance = dataset_window["provenance"]
        assert isinstance(provenance, dict)
        if (
            dataset_window["lookback_days"] != stability_window["lookback_days"]
            or provenance["provenance_sha256"]
            != stability_window["dataset_provenance_sha256"]
        ):
            raise ResearchBundleValidationError(
                "Stability window is not bound to its ordered dataset provenance."
            )
    primary_index = dataset_provenance["primary_window_index"]
    assert isinstance(primary_index, int)
    primary_stability = stability_windows[primary_index]
    assert isinstance(primary_stability, dict)
    if (
        primary_stability["canonical_strategy_sha256"] != contract.sha256()
        or primary_stability["model_execution_evidence_json"]
        != model_evidence.canonical_json
        or primary_stability["model_execution_evidence_sha256"]
        != model_evidence.sha256
    ):
        raise ResearchBundleValidationError(
            "Bundle strategy/model identity is not the selected primary stability window."
        )

    if not isinstance(regime, dict) or set(regime) != {
        "policy",
        "regime",
        "disabled_by_regime",
        "latest_bar_utc",
        "avg_volatility",
        "avg_sma_slope_pct",
        "per_symbol",
    }:
        raise ResearchBundleValidationError("Bundle regime result is invalid.")
    regime_policy = RegimePolicy.from_payload(regime["policy"])
    per_symbol = regime["per_symbol"]
    if not isinstance(per_symbol, dict) or set(per_symbol) != set(contract.symbols):
        raise ResearchBundleValidationError(
            "Bundle regime symbol evidence contradicts the canonical universe."
        )
    symbol_labels: list[str] = []
    valid_volatilities: list[float] = []
    slopes: list[float] = []
    for symbol, evidence in per_symbol.items():
        if not isinstance(evidence, dict) or set(evidence) != {
            "regime",
            "current_volatility",
            "sma_slope_pct",
        }:
            raise ResearchBundleValidationError(
                f"Regime evidence fields are invalid for {symbol}."
            )
        volatility = evidence["current_volatility"]
        slope = evidence["sma_slope_pct"]
        volatility_value = (
            None
            if volatility is None
            else _finite_numeric_float(
                volatility,
                field=f"regime.per_symbol.{symbol}.current_volatility",
                nonnegative=True,
            )
        )
        slope_value = _finite_numeric_float(
            slope,
            field=f"regime.per_symbol.{symbol}.sma_slope_pct",
        )
        expected_label = (
            "unknown"
            if volatility_value is None
            else "high_volatility"
            if volatility_value > regime_policy.high_volatility_threshold
            else "trending"
            if abs(slope_value) > regime_policy.slope_threshold
            else "sideways"
        )
        if evidence["regime"] != expected_label:
            raise ResearchBundleValidationError(
                f"Regime label contradicts bound signals for {symbol}."
            )
        if expected_label != "unknown":
            symbol_labels.append(expected_label)
        if volatility_value is not None:
            valid_volatilities.append(volatility_value)
        slopes.append(slope_value)
    if not symbol_labels:
        expected_regime = "unknown"
    elif symbol_labels.count("high_volatility") > len(symbol_labels) / 2:
        expected_regime = "high_volatility"
    elif symbol_labels.count("trending") > len(symbol_labels) / 2:
        expected_regime = "trending"
    else:
        expected_regime = "sideways"
    expected_avg_volatility = (
        round(sum(valid_volatilities) / len(valid_volatilities), 6)
        if valid_volatilities
        else None
    )
    expected_avg_slope = round(sum(slopes) / len(slopes), 6)
    latest_bar_utc = _require_utc(regime["latest_bar_utc"], field="regime.latest_bar_utc")
    generated_time = datetime.fromisoformat(
        str(normalized["generated_at_utc"])[:-1] + "+00:00"
    )
    latest_bar_time = datetime.fromisoformat(latest_bar_utc[:-1] + "+00:00")
    data_age_hours = (generated_time - latest_bar_time).total_seconds() / 3600.0
    if data_age_hours < 0 or data_age_hours > regime_policy.max_data_age_hours:
        expected_regime = "stale"
    if (
        type(regime["regime"]) is not str
        or type(regime["disabled_by_regime"]) is not bool
        or regime["regime"] != expected_regime
        or regime["disabled_by_regime"]
        is not (expected_regime in regime_policy.disable_on)
        or regime["avg_volatility"] != expected_avg_volatility
        or regime["avg_sma_slope_pct"] != expected_avg_slope
    ):
        raise ResearchBundleValidationError(
            "Bundle regime outcome contradicts its bound policy and symbol evidence."
        )
    config_inputs = research_config["inputs"]
    assert isinstance(config_inputs, dict)
    resolved_inputs = config_inputs["resolved_live_inputs"]
    assert isinstance(resolved_inputs, dict)
    if (
        resolved_inputs["symbols"] != list(contract.symbols)
        or resolved_inputs["bar_timeframe_minutes"]
        != contract.promoted_values["bar_timeframe_minutes"]
        or resolved_inputs["strategy_mode"]
        != contract.promoted_values["strategy_mode"]
    ):
        raise ResearchBundleValidationError(
            "Resolved live research inputs contradict the selected canonical strategy."
        )
    dataset_request = config_inputs["dataset_request"]
    assert isinstance(dataset_request, dict)
    configured_ranges = dataset_request["window_ranges"]
    assert isinstance(configured_ranges, list)
    if (
        approval["policy"] != config_inputs["approval_policy"]
        or stability["policy"] != config_inputs["stability_policy"]
        or regime["policy"] != config_inputs["regime_policy"]
        or [window["lookback_days"] for window in stability_windows]
        != dataset_request["validation_windows"]
        or dataset_provenance["primary_window_index"] != len(stability_windows) - 1
    ):
        raise ResearchBundleValidationError(
            "Research configuration contradicts bound policy or window evidence."
        )
    if len(configured_ranges) != len(dataset_windows):
        raise ResearchBundleValidationError(
            "Configured research windows and dataset provenance differ in count."
        )
    for configured_range, dataset_window in zip(
        configured_ranges,
        dataset_windows,
        strict=True,
    ):
        assert isinstance(configured_range, dict)
        assert isinstance(dataset_window, dict)
        provenance = dataset_window["provenance"]
        assert isinstance(provenance, dict)
        observed = provenance["observed"]
        assert isinstance(observed, dict)
        if (
            dataset_window["lookback_days"] != configured_range["lookback_days"]
            or observed["start_utc"] != configured_range["start_utc"]
            or observed["end_utc"] != configured_range["end_utc"]
        ):
            raise ResearchBundleValidationError(
                "Dataset provenance contradicts configured UTC research windows."
            )
    snapshot_request = dataset_request["snapshot"]
    if snapshot_request is not None:
        assert isinstance(snapshot_request, dict)
        for dataset_window in dataset_windows:
            assert isinstance(dataset_window, dict)
            provenance = dataset_window["provenance"]
            assert isinstance(provenance, dict)
            observed = provenance["observed"]
            assert isinstance(observed, dict)
            if (
                observed["symbols"] != snapshot_request["symbols"]
                or observed["timeframe"] != snapshot_request["timeframe"]
                or observed["feed"] != snapshot_request["feed"]
                or observed["adjustment"] != snapshot_request["adjustment"]
                or observed["align_mode"] != snapshot_request["align_mode"]
            ):
                raise ResearchBundleValidationError(
                    "Fresh dataset provenance contradicts its research snapshot request."
                )
    if not isinstance(decision, dict):
        raise ResearchBundleValidationError("Bundle final decision must be an object.")
    required_decision = {
        "should_trade",
        "approved",
        "stable",
        "disabled_by_regime",
        "market_regime",
        "canonical_strategy_sha256",
    }
    if set(decision) != required_decision:
        raise ResearchBundleValidationError("Bundle final decision fields are not exact.")
    expected_should_trade = bool(approval["approved"] and stability["stable"] and not regime["disabled_by_regime"])
    if (
        type(decision["should_trade"]) is not bool
        or type(decision["approved"]) is not bool
        or type(decision["stable"]) is not bool
        or type(decision["disabled_by_regime"]) is not bool
        or decision["should_trade"] != expected_should_trade
        or decision["approved"] != approval["approved"]
        or decision["stable"] != stability["stable"]
        or decision["disabled_by_regime"] != regime["disabled_by_regime"]
        or decision["market_regime"] != regime["regime"]
        or decision["canonical_strategy_sha256"] != contract.sha256()
    ):
        raise ResearchBundleValidationError(
            "Final decision contradicts approval, stability, regime, or strategy identity."
        )
    publication = normalized["publication"]
    if not isinstance(publication, dict) or set(publication) != {
        "generation_sha256",
        "eligible",
        "live_config_sha256",
    }:
        raise ResearchBundleValidationError("Bundle publication fields are not exact.")
    if type(publication["eligible"]) is not bool or publication["eligible"] != expected_should_trade:
        raise ResearchBundleValidationError("Bundle publication eligibility is inconsistent.")
    expected_generation_sha = compute_research_generation_sha256(
        generated_at_utc=normalized["generated_at_utc"],  # type: ignore[arg-type]
        source_content_sha256=normalized["source_identity"]["content_sha256"],  # type: ignore[index,arg-type]
        runtime_environment_sha256=normalized["source_identity"][
            "runtime_environment"
        ]["environment_sha256"],  # type: ignore[index,arg-type]
        dataset_provenance_sha256=dataset_provenance["provenance_sha256"],  # type: ignore[arg-type]
        research_config_sha256=research_config["config_sha256"],  # type: ignore[arg-type]
        canonical_strategy_sha256=contract.sha256(),
        model_execution_evidence_sha256=model_evidence.sha256,
        metrics=metrics,
        approval=approval,
        stability=stability,
        regime=regime,
        final_decision=decision,
    )
    if publication["generation_sha256"] != expected_generation_sha:
        raise ResearchBundleValidationError(
            "Bundle publication generation identity is inconsistent."
        )
    live_hash = publication["live_config_sha256"]
    if expected_should_trade:
        _require_sha256(live_hash, field="publication.live_config_sha256")
        if any(
            window["provenance"]["versioned_hashes_verified"] is not True
            for window in dataset_windows
        ):
            raise ResearchBundleValidationError(
                "Eligible publication requires verified declared dataset content hashes."
            )
    elif live_hash is not None:
        raise ResearchBundleValidationError(
            "Ineligible bundle must not claim a live config content hash."
        )

    expected_bundle_hash = _sha256_bytes(
        _canonical_json_bytes(
            {key: normalized[key] for key in normalized if key != "bundle_sha256"}
        )
    )
    if normalized["bundle_sha256"] != expected_bundle_hash:
        raise ResearchBundleValidationError("Research bundle SHA-256 is inconsistent.")
    return normalized


@dataclass(frozen=True)
class ResearchBundle:
    """Validated immutable bundle represented by canonical JSON text."""

    _canonical_json: str

    @classmethod
    def from_payload(cls, payload: Mapping[str, object]) -> ResearchBundle:
        normalized = _validate_bundle_payload(payload)
        return cls(_canonical_json_bytes(normalized).decode("utf-8"))

    @classmethod
    def from_json(cls, text: str) -> ResearchBundle:
        payload = strict_json_loads(text, label="research bundle")
        if not isinstance(payload, Mapping):
            raise ResearchBundleValidationError("Research bundle must contain an object.")
        return cls.from_payload(payload)

    @property
    def payload(self) -> dict[str, object]:
        payload = strict_json_loads(self._canonical_json, label="research bundle")
        assert isinstance(payload, dict)
        return payload

    @property
    def sha256(self) -> str:
        return self.payload["bundle_sha256"]  # type: ignore[return-value]

    def canonical_json(self) -> str:
        return self._canonical_json

    def file_bytes(self) -> bytes:
        payload = self.payload
        return json_file_bytes(payload)


_POINTER_ARTIFACT_NAMES = frozenset(
    {
        "best_config_latest",
        "live_config",
        "stability_report",
        "trade_decision",
        "drift_profile",
    }
)


def _validate_repo_relative_path(value: object, *, field: str) -> str:
    if type(value) is not str or not value:
        raise ResearchBundleValidationError(f"{field} must be a repo-relative path.")
    pure = PurePosixPath(value)
    if (
        pure.is_absolute()
        or ".." in pure.parts
        or pure.as_posix() != value
        or "\\" in value
        or (pure.parts and ":" in pure.parts[0])
    ):
        raise ResearchBundleValidationError(f"{field} is not a safe repo-relative path.")
    return value


def _validate_promotion_pointer_payload(payload: Mapping[str, object]) -> dict[str, object]:
    normalized = _normalize_json(payload, field="promotion_pointer")
    if not isinstance(normalized, dict) or set(normalized) != {
        "schema_name",
        "schema_version",
        "generator",
        "committed_at_utc",
        "research_generation_sha256",
        "bundle",
        "artifacts",
        "pointer_sha256",
    }:
        raise ResearchBundleValidationError("Promotion pointer fields are not exact.")
    if (
        normalized["schema_name"] != PROMOTION_POINTER_SCHEMA_NAME
        or type(normalized["schema_version"]) is not int
        or normalized["schema_version"] != PROMOTION_POINTER_SCHEMA_VERSION
    ):
        raise ResearchBundleValidationError("Promotion pointer schema is unsupported.")
    pointer_generator = normalized["generator"]
    if (
        not isinstance(pointer_generator, dict)
        or set(pointer_generator) != {"name", "version"}
        or pointer_generator["name"] != GENERATOR_NAME
        or type(pointer_generator["version"]) is not int
        or pointer_generator["version"] != GENERATOR_VERSION
    ):
        raise ResearchBundleValidationError("Promotion pointer generator is unsupported.")
    _require_utc(normalized["committed_at_utc"], field="pointer.committed_at_utc")
    _require_sha256(
        normalized["research_generation_sha256"],
        field="pointer.research_generation_sha256",
    )
    bundle = normalized["bundle"]
    if not isinstance(bundle, dict) or set(bundle) != {
        "path",
        "file_sha256",
        "bundle_sha256",
    }:
        raise ResearchBundleValidationError("Promotion pointer bundle fields are not exact.")
    _validate_repo_relative_path(bundle["path"], field="pointer.bundle.path")
    _require_sha256(bundle["file_sha256"], field="pointer.bundle.file_sha256")
    _require_sha256(bundle["bundle_sha256"], field="pointer.bundle.bundle_sha256")
    artifacts = normalized["artifacts"]
    if not isinstance(artifacts, dict) or set(artifacts) != _POINTER_ARTIFACT_NAMES:
        raise ResearchBundleValidationError("Promotion pointer artifact set is not exact.")
    for name, artifact in artifacts.items():
        if not isinstance(artifact, dict) or set(artifact) != {"path", "sha256"}:
            raise ResearchBundleValidationError(
                f"Promotion pointer artifact {name!r} fields are not exact."
            )
        _validate_repo_relative_path(
            artifact["path"],
            field=f"pointer.artifacts.{name}.path",
        )
        _require_sha256(
            artifact["sha256"],
            field=f"pointer.artifacts.{name}.sha256",
        )
    body = {key: normalized[key] for key in normalized if key != "pointer_sha256"}
    expected = _sha256_bytes(_canonical_json_bytes(body))
    if normalized["pointer_sha256"] != expected:
        raise ResearchBundleValidationError("Promotion pointer SHA-256 is inconsistent.")
    return normalized


@dataclass(frozen=True)
class PromotionPointer:
    _canonical_json: str

    @classmethod
    def from_payload(cls, payload: Mapping[str, object]) -> PromotionPointer:
        normalized = _validate_promotion_pointer_payload(payload)
        return cls(_canonical_json_bytes(normalized).decode("utf-8"))

    @classmethod
    def from_json(cls, text: str) -> PromotionPointer:
        payload = strict_json_loads(text, label="promotion pointer")
        if not isinstance(payload, Mapping):
            raise ResearchBundleValidationError(
                "Promotion pointer must contain an object."
            )
        return cls.from_payload(payload)

    @property
    def payload(self) -> dict[str, object]:
        payload = strict_json_loads(self._canonical_json, label="promotion pointer")
        assert isinstance(payload, dict)
        return payload

    @property
    def sha256(self) -> str:
        return self.payload["pointer_sha256"]  # type: ignore[return-value]

    def file_bytes(self) -> bytes:
        return json_file_bytes(self.payload)


def build_promotion_pointer(
    *,
    research_generation_sha256: str,
    bundle_path: str,
    bundle_sha256: str,
    bundle_file_sha256: str,
    artifacts: Mapping[str, tuple[str, bytes]],
    committed_at_utc: str | None = None,
) -> PromotionPointer:
    if set(artifacts) != _POINTER_ARTIFACT_NAMES:
        raise ResearchBundleValidationError("Promotion pointer artifact set is not exact.")
    artifact_payload: dict[str, dict[str, str]] = {}
    for name in sorted(artifacts):
        path, contents = artifacts[name]
        if not isinstance(contents, bytes):
            raise ResearchBundleValidationError(
                f"Promotion pointer artifact {name!r} contents must be bytes."
            )
        artifact_payload[name] = {
            "path": _validate_repo_relative_path(
                path,
                field=f"pointer.artifacts.{name}.path",
            ),
            "sha256": _sha256_bytes(contents),
        }
    body: dict[str, object] = {
        "schema_name": PROMOTION_POINTER_SCHEMA_NAME,
        "schema_version": PROMOTION_POINTER_SCHEMA_VERSION,
        "generator": {"name": GENERATOR_NAME, "version": GENERATOR_VERSION},
        "committed_at_utc": _require_utc(
            committed_at_utc or utc_now_iso(),
            field="pointer.committed_at_utc",
        ),
        "research_generation_sha256": _require_sha256(
            research_generation_sha256,
            field="pointer.research_generation_sha256",
        ),
        "bundle": {
            "path": _validate_repo_relative_path(
                bundle_path,
                field="pointer.bundle.path",
            ),
            "file_sha256": _require_sha256(
                bundle_file_sha256,
                field="pointer.bundle.file_sha256",
            ),
            "bundle_sha256": _require_sha256(
                bundle_sha256,
                field="pointer.bundle.bundle_sha256",
            ),
        },
        "artifacts": artifact_payload,
    }
    return PromotionPointer.from_payload(
        {
            **body,
            "pointer_sha256": _sha256_bytes(_canonical_json_bytes(body)),
        }
    )


def _resolve_pointer_target(repo_root: Path, relative_path: str) -> Path:
    root = Path(repo_root).resolve(strict=True)
    target = (root / PurePosixPath(relative_path)).resolve(strict=True)
    try:
        target.relative_to(root)
    except ValueError as exc:
        raise ResearchBundleValidationError(
            "Promotion pointer target escapes the repository root."
        ) from exc
    if not target.is_file():
        raise ResearchBundleValidationError(
            f"Promotion pointer target is not a file: {relative_path}"
        )
    return target


@dataclass(frozen=True)
class _VerifiedPromotion:
    pointer: PromotionPointer
    bundle: ResearchBundle
    bundle_path: Path
    bundle_file_sha256: str


def _verify_promotion_pointer_bytes(
    pointer_bytes: bytes,
    *,
    repo_root: Path,
) -> _VerifiedPromotion:
    """Verify pointer and generation evidence from one immutable byte buffer."""

    try:
        text = pointer_bytes.decode("utf-8", errors="strict")
    except UnicodeError as exc:
        raise ResearchBundleValidationError("Promotion pointer is unreadable.") from exc
    pointer = PromotionPointer.from_json(text)
    payload = pointer.payload
    generation_sha = payload["research_generation_sha256"]
    bundle_ref = payload["bundle"]
    assert isinstance(bundle_ref, dict)
    bundle_path = _resolve_pointer_target(repo_root, bundle_ref["path"])  # type: ignore[arg-type]
    try:
        bundle_bytes = bundle_path.read_bytes()
        bundle_text = bundle_bytes.decode("utf-8", errors="strict")
    except (OSError, UnicodeError) as exc:
        raise ResearchBundleValidationError("Promotion bundle is unreadable.") from exc
    if _sha256_bytes(bundle_bytes) != bundle_ref["file_sha256"]:
        raise ResearchBundleValidationError("Promotion pointer bundle bytes changed.")
    bundle = ResearchBundle.from_json(bundle_text)
    bundle_payload = bundle.payload
    assert_source_identity_unchanged(
        bundle_payload["source_identity"],  # type: ignore[arg-type]
        capture_source_identity(repo_root),
    )
    publication = bundle_payload["publication"]
    assert isinstance(publication, dict)
    if (
        bundle.sha256 != bundle_ref["bundle_sha256"]
        or publication["eligible"] is not True
        or publication["generation_sha256"] != generation_sha
    ):
        raise ResearchBundleValidationError(
            "Promotion pointer and immutable bundle generation disagree."
        )
    artifacts = payload["artifacts"]
    assert isinstance(artifacts, dict)
    artifact_payloads: dict[str, dict[str, object]] = {}
    for name, reference in artifacts.items():
        assert isinstance(reference, dict)
        target = _resolve_pointer_target(repo_root, reference["path"])  # type: ignore[arg-type]
        try:
            artifact_bytes = target.read_bytes()
            artifact_text = artifact_bytes.decode("utf-8", errors="strict")
        except (OSError, UnicodeError) as exc:
            raise ResearchBundleValidationError(
                f"Promotion artifact {name!r} is unreadable."
            ) from exc
        if _sha256_bytes(artifact_bytes) != reference["sha256"]:
            raise ResearchBundleValidationError(
                f"Promotion artifact {name!r} does not match the commit pointer."
            )
        parsed = strict_json_loads(
            artifact_text,
            label=f"promotion artifact {name}",
        )
        if not isinstance(parsed, dict):
            raise ResearchBundleValidationError(
                f"Promotion artifact {name!r} must contain an object."
            )
        artifact_payloads[name] = parsed
    for name, artifact in artifact_payloads.items():
        source = artifact.get("source") if name == "live_config" else artifact
        if not isinstance(source, dict) or source.get(
            "research_generation_sha256"
        ) != generation_sha:
            raise ResearchBundleValidationError(
                f"Promotion artifact {name!r} has a different generation identity."
            )
        if name != "live_config" and artifact.get("bundle_sha256") != bundle.sha256:
            raise ResearchBundleValidationError(
                f"Promotion artifact {name!r} has a different bundle identity."
            )
        if name != "live_config" and artifact.get("bundle_path") != bundle_ref["path"]:
            raise ResearchBundleValidationError(
                f"Promotion artifact {name!r} has a different bundle path."
            )

    strategy = bundle_payload["strategy"]
    approval = bundle_payload["approval"]
    assert isinstance(strategy, dict) and isinstance(approval, dict)
    contract = CanonicalResolvedPromotedStrategy.from_json(strategy["canonical_json"])  # type: ignore[arg-type]
    expected_config = {
        "symbols": list(contract.symbols),
        **contract.promoted_values,
    }
    best = artifact_payloads["best_config_latest"]
    if set(best) != {
        "schema_name",
        "schema_version",
        "saved_at",
        "research_generation_sha256",
        "bundle_path",
        "bundle_sha256",
        "approved",
        "rejection_reasons",
        "approval_policy",
        "dataset_provenance_sha256",
        "config",
        "performance",
        "canonical_strategy_contract",
        "canonical_strategy_json",
        "canonical_strategy_sha256",
        "strategy_config_sha256",
        "model_execution_evidence_sha256",
    }:
        raise ResearchBundleValidationError("Best-config projection fields are not exact.")
    expected_best = {
        "schema_name": "tradeos.promoted_best_config_projection",
        "schema_version": 1,
        "saved_at": bundle_payload["generated_at_utc"],
        "research_generation_sha256": generation_sha,
        "bundle_path": bundle_ref["path"],
        "bundle_sha256": bundle.sha256,
        "approved": approval["approved"],
        "rejection_reasons": approval["rejection_reasons"],
        "approval_policy": approval["policy"],
        "dataset_provenance_sha256": bundle_payload["dataset_provenance"][
            "provenance_sha256"
        ],
        "config": expected_config,
        "performance": bundle_payload["metrics"],
        "canonical_strategy_contract": contract.to_canonical_dict(),
        "canonical_strategy_json": strategy["canonical_json"],
        "canonical_strategy_sha256": strategy["canonical_sha256"],
        "strategy_config_sha256": strategy["strategy_config_sha256"],
        "model_execution_evidence_sha256": bundle_payload["model_execution"][
            "sha256"
        ],
    }
    if _canonical_json_bytes(best) != _canonical_json_bytes(expected_best):
        raise ResearchBundleValidationError(
            "Best-config projection contradicts the immutable bundle."
        )

    live = artifact_payloads["live_config"]
    runtime = live.get("runtime")
    source = live.get("source")
    if not isinstance(runtime, dict) or not isinstance(source, dict):
        raise ResearchBundleValidationError(
            "Live projection must contain runtime and source objects."
        )
    try:
        live_contract = CanonicalResolvedPromotedStrategy.from_mapping(
            runtime.get("symbols"),  # type: ignore[arg-type]
            {field: runtime[field] for field in PROMOTED_STRATEGY_FIELDS},
        )
    except (ContractValidationError, KeyError) as exc:
        raise ResearchBundleValidationError(
            "Live projection does not contain the complete canonical strategy."
        ) from exc
    raw_excluded = runtime.get("excluded_symbols")
    if (
        not isinstance(raw_excluded, list)
        or any(
            type(item) is not str
            or not item
            or item != item.strip().upper()
            for item in raw_excluded
        )
        or len(raw_excluded) != len(set(raw_excluded))
    ):
        raise ResearchBundleValidationError(
            "Live projection excluded_symbols must be a normalized unique list."
        )
    if set(raw_excluded) & set(contract.symbols):
        raise ResearchBundleValidationError(
            "Live projection exclusions remove canonical active symbols."
        )
    dataset_set = bundle_payload["dataset_provenance"]
    assert isinstance(dataset_set, dict)
    dataset_windows = dataset_set["windows"]
    primary_index = dataset_set["primary_window_index"]
    assert isinstance(dataset_windows, list) and isinstance(primary_index, int)
    primary_dataset = dataset_windows[primary_index]
    assert isinstance(primary_dataset, dict)
    primary_provenance = primary_dataset["provenance"]
    assert isinstance(primary_provenance, dict)
    primary_observed = primary_provenance["observed"]
    assert isinstance(primary_observed, dict)
    live_dataset_reference = source.get("dataset")
    if type(live_dataset_reference) is not str:
        raise ResearchBundleValidationError(
            "Live projection dataset reference is missing."
        )
    _validate_repo_relative_path(
        live_dataset_reference,
        field="live_config.source.dataset",
    )
    research_config_identity = bundle_payload["research_config"]
    assert isinstance(research_config_identity, dict)
    research_inputs = research_config_identity["inputs"]
    assert isinstance(research_inputs, dict)
    bound_resolved_inputs = research_inputs["resolved_live_inputs"]
    assert isinstance(bound_resolved_inputs, dict)
    expected_risk_context = validate_risk_execution_context(
        bound_resolved_inputs["risk_execution_context"]
    )
    observed_risk_context = validate_risk_execution_context(
        {field: runtime.get(field) for field in RISK_EXECUTION_CONTEXT_FIELDS}
    )
    if (
        live_contract.canonical_json() != contract.canonical_json()
        or _canonical_json_bytes(observed_risk_context)
        != _canonical_json_bytes(expected_risk_context)
        or live.get("saved_at") != bundle_payload["generated_at_utc"]
        or source.get("saved_at") != bundle_payload["generated_at_utc"]
        or PurePosixPath(live_dataset_reference).name
        != primary_observed["directory_name"]
        or source.get("canonical_strategy_schema_name") != STRATEGY_SCHEMA_NAME
        or type(source.get("canonical_strategy_schema_version")) is not int
        or source.get("canonical_strategy_schema_version") != STRATEGY_SCHEMA_VERSION
        or source.get("canonical_strategy_sha256") != strategy["canonical_sha256"]
        or source.get("canonical_strategy_config_sha256")
        != strategy["strategy_config_sha256"]
        or source.get("approved") is not True
        or source.get("rejection_reasons") != []
        or source.get("approval_policy") != approval["policy"]
        or source.get("research_generation_sha256") != generation_sha
        or source.get("dataset_provenance_sha256")
        != bundle_payload["dataset_provenance"]["provenance_sha256"]  # type: ignore[index]
        or source.get("research_config_sha256")
        != bundle_payload["research_config"]["config_sha256"]  # type: ignore[index]
        or source.get("source_git_commit")
        != bundle_payload["source_identity"]["git_commit"]  # type: ignore[index]
        or source.get("source_content_sha256")
        != bundle_payload["source_identity"]["content_sha256"]  # type: ignore[index]
        or source.get("model_execution_evidence_sha256")
        != bundle_payload["model_execution"]["sha256"]  # type: ignore[index]
    ):
        raise ResearchBundleValidationError(
            "Live projection contradicts the immutable bundle."
        )

    stability_projection = artifact_payloads["stability_report"]
    bound_stability = bundle_payload["stability"]
    bound_regime = bundle_payload["regime"]
    assert isinstance(bound_stability, dict) and isinstance(bound_regime, dict)
    bound_stability_windows = bound_stability["windows"]
    assert isinstance(bound_stability_windows, list)
    expected_stability_projection = {
        "schema_name": "tradeos.promoted_stability_projection",
        "schema_version": 1,
        "saved_at": bundle_payload["generated_at_utc"],
        "research_generation_sha256": generation_sha,
        "bundle_path": bundle_ref["path"],
        "bundle_sha256": bundle.sha256,
        "validation_windows": [
            window["lookback_days"] for window in bound_stability_windows
        ],
        "per_window": [
            {
                "lookback_days": window["lookback_days"],
                "dataset_provenance_sha256": window[
                    "dataset_provenance_sha256"
                ],
                "canonical_strategy_sha256": window[
                    "canonical_strategy_sha256"
                ],
                "model_execution_evidence_sha256": window[
                    "model_execution_evidence_sha256"
                ],
                "selection_evidence_sha256": window["selection_evidence"][
                    "evidence_sha256"
                ],
                "performance": window["metrics"],
            }
            for window in bound_stability_windows
        ],
        "average_performance": bound_stability["average_metrics"],
        "stability": bound_stability,
        "market_regime": bound_regime,
    }
    if _canonical_json_bytes(stability_projection) != _canonical_json_bytes(
        expected_stability_projection
    ):
        raise ResearchBundleValidationError(
            "Stability projection contradicts the immutable bundle."
        )
    decision_projection = artifact_payloads["trade_decision"]
    decision = bundle_payload["final_decision"]
    assert isinstance(decision, dict)
    expected_decision_projection = {
        "schema_name": "tradeos.promoted_trade_decision_projection",
        "schema_version": 1,
        "timestamp": bundle_payload["generated_at_utc"],
        "research_generation_sha256": generation_sha,
        "bundle_path": bundle_ref["path"],
        "bundle_sha256": bundle.sha256,
        **decision,
        "rejection_reasons": approval["rejection_reasons"],
        "stability_reasons": bound_stability["reasons"],
    }
    if _canonical_json_bytes(decision_projection) != _canonical_json_bytes(
        expected_decision_projection
    ):
        raise ResearchBundleValidationError(
            "Trade-decision projection contradicts the immutable bundle."
        )
    drift_projection = artifact_payloads["drift_profile"]
    metrics = bundle_payload["metrics"]
    assert isinstance(metrics, dict)
    drift_performance_names = (
        "trades_per_day",
        "win_rate",
        "profit_factor",
        "sharpe_ratio",
        "realized_pnl",
        "combined_score",
    )
    if any(name not in metrics for name in drift_performance_names):
        raise ResearchBundleValidationError(
            "Bundle lacks required promoted drift performance metrics."
        )
    signal_rate = metrics["trades_per_day"] / len(contract.symbols)
    expected_drift_projection = {
        "schema_name": DRIFT_PROFILE_SCHEMA_NAME,
        "schema_version": DRIFT_PROFILE_SCHEMA_VERSION,
        "generated_at": bundle_payload["generated_at_utc"],
        "approved": approval["approved"],
        "source_artifacts": {
            "dataset_provenance_sha256": dataset_set["provenance_sha256"],
            "best_config": artifacts["best_config_latest"]["path"],
            "trade_decision": artifacts["trade_decision"]["path"],
            "stability_report": artifacts["stability_report"]["path"],
            "live_config": artifacts["live_config"]["path"],
            "bundle": bundle_ref["path"],
        },
        "context": {
            "strategy_mode": contract.promoted_values["strategy_mode"],
            "symbols": list(contract.symbols),
            "symbol_count": len(contract.symbols),
            "timeframe": primary_observed["timeframe"],
            "bar_timeframe_minutes": contract.promoted_values[
                "bar_timeframe_minutes"
            ],
            "stable": bound_stability["stable"],
            "market_regime": bound_regime["regime"],
        },
        "performance": {
            name: metrics[name] for name in drift_performance_names
        },
        "metrics": {
            "buy_signal_rate_per_symbol_per_day": signal_rate,
            "sell_signal_rate_per_symbol_per_day": signal_rate,
            "rejection_rate": DRIFT_PROFILE_POLICY_V1.rejection_rate,
            "rejection_breakdown": None,
            "hybrid_branch_participation": None,
            "avg_bar_age_s": DRIFT_PROFILE_POLICY_V1.avg_bar_age_s,
            "stale_bar_rate": DRIFT_PROFILE_POLICY_V1.stale_bar_rate,
        },
        "metric_sources": DRIFT_PROFILE_POLICY_V1.metric_sources(),
        "research_generation_sha256": generation_sha,
        "bundle_path": bundle_ref["path"],
        "bundle_sha256": bundle.sha256,
    }
    if _canonical_json_bytes(drift_projection) != _canonical_json_bytes(
        expected_drift_projection
    ):
        raise ResearchBundleValidationError(
            "Drift projection contradicts the immutable bundle."
        )
    if artifacts["live_config"]["sha256"] != publication["live_config_sha256"]:  # type: ignore[index]
        raise ResearchBundleValidationError(
            "Promotion pointer live config hash contradicts the bundle."
        )
    return _VerifiedPromotion(
        pointer=pointer,
        bundle=bundle,
        bundle_path=bundle_path,
        bundle_file_sha256=_sha256_bytes(bundle_bytes),
    )


def _validate_latest_research_state_payload(
    payload: Mapping[str, object],
) -> dict[str, object]:
    normalized = _normalize_json(payload, field="latest_research_state")
    if not isinstance(normalized, dict) or set(normalized) != {
        "schema_name",
        "schema_version",
        "generator",
        "completed_at_utc",
        "research_generation_sha256",
        "eligible",
        "bundle",
        "promotion_pointer",
        "state_sha256",
    }:
        raise ResearchBundleValidationError(
            "Latest research state fields are not exact."
        )
    generator = normalized["generator"]
    if (
        normalized["schema_name"] != LATEST_RESEARCH_STATE_SCHEMA_NAME
        or type(normalized["schema_version"]) is not int
        or normalized["schema_version"] != LATEST_RESEARCH_STATE_SCHEMA_VERSION
        or not isinstance(generator, dict)
        or set(generator) != {"name", "version"}
        or generator["name"] != GENERATOR_NAME
        or type(generator["version"]) is not int
        or generator["version"] != GENERATOR_VERSION
    ):
        raise ResearchBundleValidationError(
            "Latest research state schema or generator is unsupported."
        )
    _require_utc(
        normalized["completed_at_utc"],
        field="latest_research_state.completed_at_utc",
    )
    _require_sha256(
        normalized["research_generation_sha256"],
        field="latest_research_state.research_generation_sha256",
    )
    if type(normalized["eligible"]) is not bool:
        raise ResearchBundleValidationError(
            "Latest research state eligibility must be boolean."
        )
    bundle = normalized["bundle"]
    if not isinstance(bundle, dict) or set(bundle) != {
        "path",
        "file_sha256",
        "bundle_sha256",
    }:
        raise ResearchBundleValidationError(
            "Latest research state bundle fields are not exact."
        )
    _validate_repo_relative_path(
        bundle["path"],
        field="latest_research_state.bundle.path",
    )
    _require_sha256(
        bundle["file_sha256"],
        field="latest_research_state.bundle.file_sha256",
    )
    _require_sha256(
        bundle["bundle_sha256"],
        field="latest_research_state.bundle.bundle_sha256",
    )
    pointer = normalized["promotion_pointer"]
    if normalized["eligible"] is True:
        if not isinstance(pointer, dict) or set(pointer) != {
            "path",
            "file_sha256",
            "pointer_sha256",
        }:
            raise ResearchBundleValidationError(
                "Eligible latest research state requires an exact promotion pointer."
            )
        _validate_repo_relative_path(
            pointer["path"],
            field="latest_research_state.promotion_pointer.path",
        )
        _require_sha256(
            pointer["file_sha256"],
            field="latest_research_state.promotion_pointer.file_sha256",
        )
        _require_sha256(
            pointer["pointer_sha256"],
            field="latest_research_state.promotion_pointer.pointer_sha256",
        )
    elif pointer is not None:
        raise ResearchBundleValidationError(
            "Ineligible latest research state cannot claim a promotion pointer."
        )
    body = {key: normalized[key] for key in normalized if key != "state_sha256"}
    expected_sha = _sha256_bytes(_canonical_json_bytes(body))
    if normalized["state_sha256"] != expected_sha:
        raise ResearchBundleValidationError(
            "Latest research state SHA-256 is inconsistent."
        )
    return normalized


@dataclass(frozen=True)
class LatestResearchState:
    """Authoritative commit marker for the most recently completed research."""

    _canonical_json: str

    @classmethod
    def from_payload(cls, payload: Mapping[str, object]) -> LatestResearchState:
        normalized = _validate_latest_research_state_payload(payload)
        return cls(_canonical_json_bytes(normalized).decode("utf-8"))

    @classmethod
    def from_json(cls, text: str) -> LatestResearchState:
        payload = strict_json_loads(text, label="latest research state")
        if not isinstance(payload, Mapping):
            raise ResearchBundleValidationError(
                "Latest research state must contain an object."
            )
        return cls.from_payload(payload)

    @property
    def payload(self) -> dict[str, object]:
        payload = strict_json_loads(
            self._canonical_json,
            label="latest research state",
        )
        assert isinstance(payload, dict)
        return payload

    @property
    def sha256(self) -> str:
        return self.payload["state_sha256"]  # type: ignore[return-value]

    def file_bytes(self) -> bytes:
        return json_file_bytes(self.payload)


def build_latest_research_state(
    *,
    research_generation_sha256: str,
    bundle_path: str,
    bundle_sha256: str,
    bundle_file_sha256: str,
    eligible: bool,
    promotion_pointer_path: str | None = None,
    promotion_pointer: PromotionPointer | None = None,
    promotion_pointer_file_sha256: str | None = None,
    completed_at_utc: str | None = None,
) -> LatestResearchState:
    if type(eligible) is not bool:
        raise ResearchBundleValidationError(
            "Latest research state eligibility must be boolean."
        )
    if eligible:
        if (
            type(promotion_pointer_path) is not str
            or not isinstance(promotion_pointer, PromotionPointer)
        ):
            raise ResearchBundleValidationError(
                "Eligible latest research state requires promotion pointer identity."
            )
        pointer_payload: dict[str, object] | None = {
            "path": _validate_repo_relative_path(
                promotion_pointer_path,
                field="latest_research_state.promotion_pointer.path",
            ),
            "file_sha256": _require_sha256(
                promotion_pointer_file_sha256,
                field="latest_research_state.promotion_pointer.file_sha256",
            ),
            "pointer_sha256": promotion_pointer.sha256,
        }
    else:
        if any(
            value is not None
            for value in (
                promotion_pointer_path,
                promotion_pointer,
                promotion_pointer_file_sha256,
            )
        ):
            raise ResearchBundleValidationError(
                "Ineligible latest research state cannot claim promotion pointer identity."
            )
        pointer_payload = None
    body: dict[str, object] = {
        "schema_name": LATEST_RESEARCH_STATE_SCHEMA_NAME,
        "schema_version": LATEST_RESEARCH_STATE_SCHEMA_VERSION,
        "generator": {"name": GENERATOR_NAME, "version": GENERATOR_VERSION},
        "completed_at_utc": _require_utc(
            completed_at_utc or utc_now_iso(),
            field="latest_research_state.completed_at_utc",
        ),
        "research_generation_sha256": _require_sha256(
            research_generation_sha256,
            field="latest_research_state.research_generation_sha256",
        ),
        "eligible": eligible,
        "bundle": {
            "path": _validate_repo_relative_path(
                bundle_path,
                field="latest_research_state.bundle.path",
            ),
            "file_sha256": _require_sha256(
                bundle_file_sha256,
                field="latest_research_state.bundle.file_sha256",
            ),
            "bundle_sha256": _require_sha256(
                bundle_sha256,
                field="latest_research_state.bundle.bundle_sha256",
            ),
        },
        "promotion_pointer": pointer_payload,
    }
    return LatestResearchState.from_payload(
        {
            **body,
            "state_sha256": _sha256_bytes(_canonical_json_bytes(body)),
        }
    )


def _verify_latest_research_state_bytes(
    state_bytes: bytes,
    *,
    repo_root: Path,
    promotion_pointer_path: Path | None = None,
    promotion_pointer_bytes: bytes | None = None,
    verified_promotion: _VerifiedPromotion | None = None,
) -> LatestResearchState:
    try:
        state_text = state_bytes.decode("utf-8", errors="strict")
    except UnicodeError as exc:
        raise ResearchBundleValidationError(
            "Latest research state is unreadable."
        ) from exc
    state = LatestResearchState.from_json(state_text)
    payload = state.payload
    root = Path(repo_root).resolve(strict=True)
    supplied_pointer = promotion_pointer_path is not None or promotion_pointer_bytes is not None
    if supplied_pointer and (
        promotion_pointer_path is None or not isinstance(promotion_pointer_bytes, bytes)
    ):
        raise ResearchBundleValidationError(
            "Staged promotion pointer path and bytes must be supplied together."
        )
    if verified_promotion is not None and not supplied_pointer:
        raise ResearchBundleValidationError(
            "Preverified promotion context requires exact pointer path and bytes."
        )

    bundle_ref = payload["bundle"]
    assert isinstance(bundle_ref, dict)
    if payload["eligible"] is True:
        pointer_ref = payload["promotion_pointer"]
        assert isinstance(pointer_ref, dict)
        expected_pointer_path = (
            root / PurePosixPath(pointer_ref["path"])  # type: ignore[arg-type]
        ).resolve(strict=False)
        if supplied_pointer:
            assert promotion_pointer_path is not None
            pointer_target = Path(promotion_pointer_path).resolve(strict=False)
            if pointer_target != expected_pointer_path:
                raise ResearchBundleValidationError(
                    "Staged promotion pointer path contradicts latest research state."
                )
            assert isinstance(promotion_pointer_bytes, bytes)
            pointer_bytes = promotion_pointer_bytes
        else:
            pointer_target = _resolve_pointer_target(
                root,
                pointer_ref["path"],  # type: ignore[arg-type]
            )
            try:
                pointer_bytes = pointer_target.read_bytes()
            except OSError as exc:
                raise ResearchBundleValidationError(
                    "Latest research promotion pointer is unreadable."
                ) from exc
        if _sha256_bytes(pointer_bytes) != pointer_ref["file_sha256"]:
            raise ResearchBundleValidationError(
                "Latest research promotion pointer bytes changed."
            )
        verified = verified_promotion or _verify_promotion_pointer_bytes(
            pointer_bytes,
            repo_root=root,
        )
        pointer_payload = verified.pointer.payload
        if (
            verified.pointer.sha256 != pointer_ref["pointer_sha256"]
            or pointer_payload["research_generation_sha256"]
            != payload["research_generation_sha256"]
            or pointer_payload["bundle"] != bundle_ref
        ):
            raise ResearchBundleValidationError(
                "Latest research state contradicts its verified promotion pointer."
            )
        bundle = verified.bundle
        bundle_file_sha256 = verified.bundle_file_sha256
    else:
        if supplied_pointer:
            raise ResearchBundleValidationError(
                "Ineligible latest research state cannot verify staged promotion bytes."
            )
        bundle_path = _resolve_pointer_target(
            root,
            bundle_ref["path"],  # type: ignore[arg-type]
        )
        try:
            bundle_bytes = bundle_path.read_bytes()
            bundle_text = bundle_bytes.decode("utf-8", errors="strict")
        except (OSError, UnicodeError) as exc:
            raise ResearchBundleValidationError(
                "Latest research bundle is unreadable."
            ) from exc
        bundle_file_sha256 = _sha256_bytes(bundle_bytes)
        bundle = ResearchBundle.from_json(bundle_text)
        assert_source_identity_unchanged(
            bundle.payload["source_identity"],  # type: ignore[arg-type]
            capture_source_identity(root),
        )
    publication = bundle.payload["publication"]
    assert isinstance(publication, dict)
    if (
        bundle_file_sha256 != bundle_ref["file_sha256"]
        or bundle.sha256 != bundle_ref["bundle_sha256"]
        or publication["generation_sha256"]
        != payload["research_generation_sha256"]
        or publication["eligible"] is not payload["eligible"]
        or bundle.payload["generated_at_utc"] != payload["completed_at_utc"]
    ):
        raise ResearchBundleValidationError(
            "Latest research state contradicts its immutable bundle."
        )
    return state


def verify_latest_research_state_bytes(
    state_bytes: bytes,
    *,
    repo_root: Path,
    promotion_pointer_path: Path | None = None,
    promotion_pointer_bytes: bytes | None = None,
) -> LatestResearchState:
    """Verify exact staged latest-state bytes before their atomic commit."""

    if not isinstance(state_bytes, bytes):
        raise ResearchBundleValidationError(
            "Latest research state candidate must be bytes."
        )
    return _verify_latest_research_state_bytes(
        state_bytes,
        repo_root=repo_root,
        promotion_pointer_path=promotion_pointer_path,
        promotion_pointer_bytes=promotion_pointer_bytes,
    )


def load_verified_latest_research_state(
    state_path: Path,
    *,
    repo_root: Path,
) -> LatestResearchState:
    """Load and verify the authoritative most-recent research commit marker."""

    try:
        state_bytes = Path(state_path).read_bytes()
    except OSError as exc:
        raise ResearchBundleValidationError(
            "Latest research state is unreadable."
        ) from exc
    return _verify_latest_research_state_bytes(
        state_bytes,
        repo_root=repo_root,
    )


def load_verified_promotion_pointer(
    pointer_path: Path,
    *,
    repo_root: Path,
    latest_state_path: Path | None = None,
    require_latest_state: bool = True,
) -> PromotionPointer:
    """Verify a promotion and require it to be the latest completed research."""

    if type(require_latest_state) is not bool:
        raise ResearchBundleValidationError(
            "require_latest_state must be boolean."
        )
    try:
        pointer_bytes = Path(pointer_path).read_bytes()
    except OSError as exc:
        raise ResearchBundleValidationError("Promotion pointer is unreadable.") from exc
    verified = _verify_promotion_pointer_bytes(
        pointer_bytes,
        repo_root=repo_root,
    )
    if require_latest_state:
        state_target = (
            Path(latest_state_path)
            if latest_state_path is not None
            else Path(repo_root) / "results" / "research_latest.json"
        )
        try:
            state_bytes = state_target.read_bytes()
        except OSError as exc:
            raise ResearchBundleValidationError(
                "Latest research state is unreadable."
            ) from exc
        state = _verify_latest_research_state_bytes(
            state_bytes,
            repo_root=repo_root,
            promotion_pointer_path=Path(pointer_path),
            promotion_pointer_bytes=pointer_bytes,
            verified_promotion=verified,
        )
        if state.payload["eligible"] is not True:
            raise ResearchBundleValidationError(
                "Latest completed research is not eligible for promotion."
            )
    return verified.pointer


def build_research_bundle(
    *,
    canonical_strategy: BacktestCanonicalStrategy,
    model_execution_evidence: BacktestModelExecutionEvidence,
    metrics: Mapping[str, object],
    approval_policy: ApprovalPolicy,
    approved: bool,
    rejection_reasons: Sequence[str],
    stability: Mapping[str, object],
    regime: Mapping[str, object],
    final_decision: Mapping[str, object],
    dataset_provenance: Mapping[str, object],
    source_identity: Mapping[str, object],
    research_config_inputs: Mapping[str, object],
    live_config_sha256: str | None,
    generated_at_utc: str | None = None,
) -> ResearchBundle:
    normalized_metrics = normalize_metrics(metrics)
    expected_outcome = evaluate_approval_policy(normalized_metrics, approval_policy)
    if type(approved) is not bool or approved != expected_outcome.approved:
        raise ResearchBundleValidationError(
            "Supplied approval contradicts the selected metrics and policy."
        )
    normalized_reasons = tuple(rejection_reasons)
    if any(type(reason) is not str for reason in normalized_reasons) or normalized_reasons != expected_outcome.rejection_reasons:
        raise ResearchBundleValidationError(
            "Supplied rejection reasons contradict the selected metrics and policy."
        )
    normalized_stability = _normalize_json(stability, field="stability")
    normalized_regime = _normalize_json(regime, field="regime")
    normalized_decision = _normalize_json(final_decision, field="final_decision")
    if not isinstance(normalized_stability, dict) or not isinstance(normalized_regime, dict) or not isinstance(normalized_decision, dict):
        raise ResearchBundleValidationError("Research verdict sections must be objects.")
    eligible = bool(
        approved
        and normalized_stability.get("stable") is True
        and normalized_regime.get("disabled_by_regime") is False
    )
    normalized_dataset = _validate_dataset_provenance_set(dataset_provenance)
    if eligible:
        _require_sha256(live_config_sha256, field="live_config_sha256")
        dataset_windows = normalized_dataset["windows"]
        assert isinstance(dataset_windows, list)
        if any(
            window["provenance"]["versioned_hashes_verified"] is not True
            for window in dataset_windows
        ):
            raise ResearchBundleValidationError(
                "Eligible research cannot use legacy observed-only dataset provenance."
            )
    elif live_config_sha256 is not None:
        raise ResearchBundleValidationError(
            "An ineligible research decision cannot bind a live config publication."
        )
    generated_value = _require_utc(
        generated_at_utc or utc_now_iso(),
        field="generated_at_utc",
    )
    normalized_source = _validate_source_identity(source_identity)
    research_config = build_research_config_identity(research_config_inputs)
    approval_payload = {
        "policy": approval_policy.to_payload(),
        "approved": approved,
        "rejection_reasons": list(normalized_reasons),
    }
    generation_sha256 = compute_research_generation_sha256(
        generated_at_utc=generated_value,
        source_content_sha256=normalized_source["content_sha256"],  # type: ignore[arg-type]
        runtime_environment_sha256=normalized_source["runtime_environment"][
            "environment_sha256"
        ],  # type: ignore[index,arg-type]
        dataset_provenance_sha256=normalized_dataset["provenance_sha256"],  # type: ignore[arg-type]
        research_config_sha256=research_config["config_sha256"],  # type: ignore[arg-type]
        canonical_strategy_sha256=canonical_strategy.sha256,
        model_execution_evidence_sha256=model_execution_evidence.sha256,
        metrics=normalized_metrics,
        approval=approval_payload,
        stability=normalized_stability,
        regime=normalized_regime,
        final_decision=normalized_decision,
    )
    body: dict[str, object] = {
        "schema_name": BUNDLE_SCHEMA_NAME,
        "schema_version": BUNDLE_SCHEMA_VERSION,
        "generator": {"name": GENERATOR_NAME, "version": GENERATOR_VERSION},
        "generated_at_utc": generated_value,
        "source_identity": normalized_source,
        "dataset_provenance": normalized_dataset,
        "research_config": research_config,
        "strategy": {
            "canonical_json": canonical_strategy.canonical_json,
            "canonical_sha256": canonical_strategy.sha256,
            "strategy_config_sha256": strategy_config_sha256(
                canonical_strategy.contract
            ),
            "schema_sha256": strategy_schema_sha256(),
        },
        "model_execution": {
            "canonical_json": model_execution_evidence.canonical_json,
            "sha256": model_execution_evidence.sha256,
        },
        "metrics": normalized_metrics,
        "approval": approval_payload,
        "stability": normalized_stability,
        "regime": normalized_regime,
        "final_decision": normalized_decision,
        "publication": {
            "generation_sha256": generation_sha256,
            "eligible": eligible,
            "live_config_sha256": live_config_sha256,
        },
    }
    payload = {
        **body,
        "bundle_sha256": _sha256_bytes(_canonical_json_bytes(body)),
    }
    return ResearchBundle.from_payload(payload)


def write_immutable_bundle(bundle: ResearchBundle, directory: Path) -> Path:
    """Publish a complete bundle without ever replacing an existing path."""

    target_dir = Path(directory)
    target_dir.mkdir(parents=True, exist_ok=True)
    payload = bundle.payload
    stamp = str(payload["generated_at_utc"]).replace("-", "").replace(":", "")
    path = target_dir / f"promotion_bundle.{stamp}.{bundle.sha256[:16]}.json"
    contents = bundle.file_bytes()
    if path.exists():
        if path.read_bytes() == contents:
            return path
        raise ResearchPublicationError(
            f"Immutable bundle path already exists with different content: {path}"
        )
    # Do not repeat the already descriptive target filename in the staging
    # component: nested Windows workspaces can otherwise exceed MAX_PATH.
    temporary = target_dir / f".{uuid4().hex}.tmp"
    try:
        with temporary.open("xb") as handle:
            handle.write(contents)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError:
            if path.read_bytes() != contents:
                raise ResearchPublicationError(
                    f"Immutable bundle path raced with different content: {path}"
                )
        except OSError as exc:
            # A hard link is the only operation here that guarantees a fully
            # written target without replacement.  Do not weaken immutability
            # by falling back to os.replace.
            raise ResearchPublicationError(
                f"Unable to atomically create immutable bundle {path}: {exc}"
            ) from exc
    finally:
        temporary.unlink(missing_ok=True)
    return path


PublicationFailureInjector = Callable[[str, int, Path], None]


def write_immutable_generation(
    generation_sha256: str,
    artifacts: Mapping[str, bytes],
    directory: Path,
) -> Path:
    """Publish a complete content-addressed projection directory once."""

    generation = _require_sha256(
        generation_sha256,
        field="research_generation_sha256",
    )
    if not artifacts:
        raise ResearchPublicationError("Immutable generation artifacts are empty.")
    normalized: dict[str, bytes] = {}
    for name, contents in artifacts.items():
        safe_name = (
            _validate_repo_relative_path(name, field="immutable_generation.artifact_name")
            if type(name) is str
            else ""
        )
        if (
            type(name) is not str
            or not name
            or safe_name != name
            or len(PurePosixPath(name).parts) != 1
            or name in {".", ".."}
            or not isinstance(contents, bytes)
        ):
            raise ResearchPublicationError(
                "Immutable generation artifact names/contents are invalid."
            )
        normalized[name] = contents
    root = Path(directory)
    root.mkdir(parents=True, exist_ok=True)
    target = root / generation_directory_component(generation)

    def _matches_existing() -> bool:
        if not target.is_dir():
            return False
        existing = {path.name for path in target.iterdir() if path.is_file()}
        if existing != set(normalized) or any(path.is_dir() for path in target.iterdir()):
            return False
        return all((target / name).read_bytes() == contents for name, contents in normalized.items())

    if target.exists():
        if _matches_existing():
            return target
        raise ResearchPublicationError(
            f"Immutable generation already exists with different content: {target}"
        )
    # Keep the staging component short enough for Windows' traditional path
    # limit; the final content-addressed directory still carries generation.
    temporary = root / f".{uuid4().hex}.tmp"
    temporary.mkdir(parents=False, exist_ok=False)
    try:
        for name, contents in normalized.items():
            with (temporary / name).open("xb") as handle:
                handle.write(contents)
                handle.flush()
                os.fsync(handle.fileno())
        try:
            os.replace(temporary, target)
        except OSError as exc:
            if target.exists() and _matches_existing():
                return target
            raise ResearchPublicationError(
                f"Unable to commit immutable generation {target}: {exc}"
            ) from exc
        return target
    finally:
        if temporary.exists():
            for child in temporary.iterdir():
                if child.is_file():
                    child.unlink(missing_ok=True)
            temporary.rmdir()


def publish_files_atomically(
    publications: Mapping[Path, bytes],
    *,
    commit_pointer: tuple[Path, bytes] | None = None,
    expected_originals: Mapping[Path, bytes | None] | None = None,
    failure_injector: PublicationFailureInjector | None = None,
) -> None:
    """Stage a publication set and restore every target after handled failure.

    No filesystem offers a cross-directory multi-file rename transaction.  This
    function therefore stages and fsyncs every file first, detects concurrent
    target changes, and uses exact rollback bytes if a replacement or injected
    failure occurs.  Rollback is compare-before-restore so it cannot overwrite
    a different writer's later bytes.  The immutable bundle remains the
    authoritative evidence.
    """

    if not publications and commit_pointer is None:
        return
    ordered: list[tuple[Path, bytes]] = []
    seen: set[Path] = set()
    for raw_path, contents in publications.items():
        path = Path(raw_path).resolve(strict=False)
        if path in seen:
            raise ResearchPublicationError(f"Duplicate publication target: {path}")
        if not isinstance(contents, bytes):
            raise ResearchPublicationError(f"Publication content must be bytes: {path}")
        seen.add(path)
        ordered.append((path, contents))
    ordered.sort(key=lambda item: str(item[0]))
    pointer_target: Path | None = None
    if commit_pointer is not None:
        raw_pointer_path, pointer_contents = commit_pointer
        pointer_target = Path(raw_pointer_path).resolve(strict=False)
        if pointer_target in seen or not isinstance(pointer_contents, bytes):
            raise ResearchPublicationError("Commit pointer target/content is invalid.")
        seen.add(pointer_target)
        # The authoritative commit marker is always replaced last.
        ordered.append((pointer_target, pointer_contents))

    normalized_expected = {
        Path(path).resolve(strict=False): contents
        for path, contents in (expected_originals or {}).items()
    }
    if not set(normalized_expected) <= seen:
        raise ResearchPublicationError(
            "Expected-original checks contain an unknown publication target."
        )

    staged: dict[Path, Path] = {}
    originals: dict[Path, bytes | None] = {}
    replaced: list[Path] = []
    published_contents = dict(ordered)
    try:
        for index, (target, contents) in enumerate(ordered):
            target.parent.mkdir(parents=True, exist_ok=True)
            originals[target] = target.read_bytes() if target.exists() else None
            if (
                target in normalized_expected
                and originals[target] != normalized_expected[target]
            ):
                raise ResearchPublicationError(
                    f"Publication target changed before staging: {target}"
                )
            temporary = target.parent / f".{target.name}.{uuid4().hex}.tmp"
            with temporary.open("xb") as handle:
                handle.write(contents)
                handle.flush()
                os.fsync(handle.fileno())
            staged[target] = temporary
            if failure_injector is not None:
                failure_injector("after_stage", index, target)

        if failure_injector is not None:
            failure_injector("before_commit", -1, ordered[0][0])
        for index, (target, _contents) in enumerate(ordered):
            original = originals[target]
            current = target.read_bytes() if target.exists() else None
            if current != original:
                raise ResearchPublicationError(
                    f"Publication target changed concurrently: {target}"
                )
            os.replace(staged[target], target)
            staged.pop(target, None)
            replaced.append(target)
            if failure_injector is not None:
                failure_injector("after_replace", index, target)
        if pointer_target is not None and replaced[-1] != pointer_target:
            raise ResearchPublicationError("Commit pointer was not published last.")
    except BaseException as exc:
        rollback_errors: list[str] = []
        for target in reversed(replaced):
            original = originals[target]
            try:
                published = published_contents[target]
                current = target.read_bytes() if target.exists() else None
                if current != published:
                    rollback_errors.append(
                        f"{target}: changed by another writer after publication"
                    )
                    continue
                if original is None:
                    target.unlink(missing_ok=True)
                else:
                    rollback = target.parent / f".{target.name}.{uuid4().hex}.rollback"
                    try:
                        with rollback.open("xb") as handle:
                            handle.write(original)
                            handle.flush()
                            os.fsync(handle.fileno())
                        os.replace(rollback, target)
                    finally:
                        rollback.unlink(missing_ok=True)
            except OSError as rollback_exc:
                rollback_errors.append(f"{target}: {rollback_exc}")
        if rollback_errors:
            raise ResearchPublicationError(
                "Publication failed and rollback was incomplete: "
                + "; ".join(rollback_errors)
            ) from exc
        if isinstance(exc, (ResearchPublicationError, ResearchBundleValidationError)):
            raise
        raise ResearchPublicationError("Publication failed; original files were restored.") from exc
    finally:
        for temporary in staged.values():
            temporary.unlink(missing_ok=True)


__all__ = [
    "ApprovalOutcome",
    "ApprovalPolicy",
    "BUNDLE_SCHEMA_NAME",
    "BUNDLE_SCHEMA_VERSION",
    "BacktestCanonicalStrategy",
    "BacktestModelExecutionEvidence",
    "CANONICAL_STRATEGY_JSON_COLUMN",
    "CANONICAL_STRATEGY_SHA256_COLUMN",
    "DRIFT_PROFILE_POLICY_V1",
    "DRIFT_PROFILE_SCHEMA_NAME",
    "DRIFT_PROFILE_SCHEMA_VERSION",
    "DriftProfilePolicyV1",
    "GENERATOR_NAME",
    "GENERATOR_VERSION",
    "LATEST_RESEARCH_STATE_SCHEMA_NAME",
    "LATEST_RESEARCH_STATE_SCHEMA_VERSION",
    "LatestResearchState",
    "MODEL_EXECUTION_EVIDENCE_JSON_COLUMN",
    "MODEL_EXECUTION_EVIDENCE_SCHEMA_NAME",
    "MODEL_EXECUTION_EVIDENCE_SCHEMA_VERSION",
    "MODEL_EXECUTION_EVIDENCE_SHA256_COLUMN",
    "PROMOTION_POINTER_SCHEMA_NAME",
    "PROMOTION_POINTER_SCHEMA_VERSION",
    "PromotionPointer",
    "RESEARCH_CONFIG_SCHEMA_NAME",
    "RESEARCH_CONFIG_SCHEMA_VERSION",
    "ResearchBundle",
    "ResearchBundleValidationError",
    "ResearchPublicationError",
    "RISK_EXECUTION_CONTEXT_FIELDS",
    "RUNTIME_ENVIRONMENT_DISTRIBUTIONS",
    "RUNTIME_ENVIRONMENT_SCHEMA_NAME",
    "RUNTIME_ENVIRONMENT_SCHEMA_VERSION",
    "RegimePolicy",
    "SELECTION_EVIDENCE_SCHEMA_NAME",
    "SELECTION_EVIDENCE_SCHEMA_VERSION",
    "StabilityPolicy",
    "assert_dataset_checkpoint_unchanged",
    "assert_source_identity_unchanged",
    "build_dataset_provenance_set",
    "build_backtest_selection_evidence",
    "build_live_config_payload",
    "build_latest_research_state",
    "build_promotion_pointer",
    "build_research_config_identity",
    "build_research_bundle",
    "canonical_strategy_from_backtest_row",
    "capture_dataset_provenance",
    "capture_dataset_checkpoint",
    "capture_runtime_environment",
    "capture_source_identity",
    "compute_research_generation_sha256",
    "evaluate_approval_policy",
    "evaluate_stability_policy",
    "generation_directory_component",
    "json_file_bytes",
    "load_verified_promotion_pointer",
    "load_verified_latest_research_state",
    "model_execution_evidence_from_backtest_row",
    "normalize_metrics",
    "publish_files_atomically",
    "resolve_risk_execution_context",
    "strategy_config_sha256",
    "strategy_schema_sha256",
    "strict_json_loads",
    "utc_now_iso",
    "validate_backtest_selection_artifacts",
    "validate_risk_execution_context",
    "verify_latest_research_state_bytes",
    "write_immutable_bundle",
    "write_immutable_generation",
]

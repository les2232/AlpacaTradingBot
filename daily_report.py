#!/usr/bin/env python3
"""
daily_report.py
---------------
Reads the bot's JSONL logs for one trading day and prints a diagnostic
report comparing live behavior to backtest expectations.

The comparison is fail-closed: when the saved baseline is not valid for
comparison (BACKTEST_SOURCE["valid_for_comparison"] is not True), BACKTEST
holds fallback numbers only, so the report says "comparison unavailable" with
the validation reasons and omits every backtest-relative statement.  Observed
metrics and absolute WARN thresholds are still reported.

Even with a valid baseline, only metrics loaded from a validated artifact
(research or live config) are shown as expectations.  avg_hold_bars,
rejection_rate and avg_slippage_bps are not carried by any research artifact,
so they never get a "(backtest ...)" note, comparison row, or JSON value.

Usage:
    python daily_report.py                      # today
    python daily_report.py --date 2026-04-09
    python daily_report.py --date 2026-04-09 --log-root logs
    python daily_report.py --date 2026-04-09 --json   # machine-readable output

Requires: pandas
"""

import argparse
import hashlib
import json
import math
import sys
from datetime import date
from pathlib import Path, PurePosixPath
from typing import Any

import pandas as pd

from tradeos.research_bundle import (
    RISK_EXECUTION_CONTEXT_FIELDS,
    ResearchBundleValidationError,
    load_verified_promotion_pointer,
    validate_risk_execution_context,
)
from tradeos.strategy_contract import (
    CanonicalResolvedPromotedStrategy,
    ContractValidationError,
    PROMOTED_STRATEGY_FIELDS,
)

# ---------------------------------------------------------------------------
# Backtest placeholders
# Historical OOS numbers from an old mean-reversion validation.  They are NOT
# expectations: load_backtest_baseline() overwrites the metrics that validated
# artifacts carry and records each metric's provenance in
# BACKTEST_SOURCE["metric_sources"].  The report only presents a metric as an
# expectation when its source is "research" or "live_config".
# ---------------------------------------------------------------------------

DEFAULT_BACKTEST: dict[str, Any] = {
    "symbols":                         15,
    "bars_per_day":                    26,     # 390-min session ÷ 15-min bars
    "signal_rate_per_symbol_per_day":  2.1,    # avg BUY signals per symbol per session
    "win_rate":                        0.65,   # fraction of closed trades that are winners
    "avg_hold_bars":                   4.2,    # bars held per trade
    "rejection_rate":                  0.72,   # fraction of evaluated bars → HOLD
    "avg_slippage_bps":                5.0,    # expected adverse slippage (bps)
}


# Per-metric provenance recorded in BACKTEST_SOURCE["metric_sources"].  Only a
# metric whose value was actually loaded from a validated artifact may be shown
# as an expectation; everything else keeps its DEFAULT_BACKTEST placeholder in
# BACKTEST (for shape compatibility) but is reported as unavailable.
METRIC_SOURCE_RESEARCH = "research"
METRIC_SOURCE_LIVE_CONFIG = "live_config"
METRIC_SOURCE_UNAVAILABLE = "unavailable"
_LOADED_METRIC_SOURCES = frozenset({METRIC_SOURCE_RESEARCH, METRIC_SOURCE_LIVE_CONFIG})

# The promotion bundle and best_config_latest.json "performance" block carry
# trades_per_day and win_rate, but no holding-period, rejection-rate, or
# slippage measurement, so these never have a research-backed expectation.
NON_RESEARCH_BACKED_METRICS: tuple[str, ...] = (
    "avg_hold_bars",
    "rejection_rate",
    "avg_slippage_bps",
)


# Backward-compatible name for report/drift consumers.  The explicit, frozen,
# versioned definition lives in tradeos.strategy_contract.
PROMOTED_RUNTIME_FIELDS: tuple[str, ...] = PROMOTED_STRATEGY_FIELDS


# These execution-affecting values intentionally remain outside the canonical
# strategy hash, but an operational baseline is comparable only when the
# selected runtime carries the exact context committed in the pointer-bound
# live projection.  Keeping this surface explicit makes additions fail closed.
EXECUTION_CONTEXT_FIELDS: tuple[str, ...] = (
    "historical_feed",
    "live_feed",
    "latest_bar_feed",
    "bar_build_mode",
    "apply_updated_bars",
    "post_bar_reconcile_poll",
    "block_trading_until_resync",
    "assert_feed_on_startup",
    "log_bar_components",
)
EXECUTION_CONTEXT_COMPARISON_FIELDS: tuple[str, ...] = (
    *EXECUTION_CONTEXT_FIELDS,
    *RISK_EXECUTION_CONTEXT_FIELDS,
)
_DATA_FEED_VALUES = frozenset({"iex", "sip", "delayed_sip"})
_BAR_BUILD_MODE_VALUES = frozenset(
    {"stream_minute_aggregate", "historical_preaggregated"}
)


def _strict_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    for key, value in pairs:
        if key in payload:
            raise ValueError(f"duplicate JSON field {key!r}")
        payload[key] = value
    return payload


def _reject_nonfinite_json_number(token: str) -> None:
    raise ValueError(f"non-finite JSON number {token!r}")


def _load_json_file_with_sha256(
    path: Path,
) -> tuple[dict[str, Any] | None, str | None]:
    if not path.exists():
        return None, None
    try:
        raw = path.read_bytes()
        payload = json.loads(
            raw.decode("utf-8", errors="strict"),
            object_pairs_hook=_strict_json_object,
            parse_constant=_reject_nonfinite_json_number,
        )
    except (OSError, UnicodeError, ValueError):
        return None, None
    if not isinstance(payload, dict):
        return None, None
    return payload, hashlib.sha256(raw).hexdigest()


def _load_json_file(path: Path) -> dict[str, Any] | None:
    payload, _sha256 = _load_json_file_with_sha256(path)
    return payload


def _load_pointer_bound_json(
    repo_root: Path,
    reference: object,
) -> tuple[Path, dict[str, Any]]:
    """Read and parse the exact artifact bytes committed by a promotion pointer."""

    if not isinstance(reference, dict) or set(reference) != {"path", "sha256"}:
        raise ValueError("promotion artifact reference fields are not exact")
    relative_path = reference["path"]
    expected_sha256 = reference["sha256"]
    if type(relative_path) is not str or not relative_path:
        raise ValueError("promotion artifact path is missing or invalid")
    pure_path = PurePosixPath(relative_path)
    if pure_path.is_absolute() or ".." in pure_path.parts or pure_path.as_posix() != relative_path:
        raise ValueError("promotion artifact path is not a safe repo-relative path")
    if (
        type(expected_sha256) is not str
        or len(expected_sha256) != 64
        or any(char not in "0123456789abcdef" for char in expected_sha256)
    ):
        raise ValueError("promotion artifact SHA-256 is missing or invalid")

    root = Path(repo_root).resolve(strict=True)
    artifact_path = (root / pure_path).resolve(strict=True)
    try:
        artifact_path.relative_to(root)
    except ValueError as exc:
        raise ValueError("promotion artifact path escapes the repository") from exc
    raw = artifact_path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected_sha256:
        raise ValueError("promotion artifact bytes do not match the commit pointer")
    try:
        payload = json.loads(
            raw.decode("utf-8", errors="strict"),
            object_pairs_hook=_strict_json_object,
            parse_constant=_reject_nonfinite_json_number,
        )
    except (UnicodeError, ValueError) as exc:
        raise ValueError("promotion artifact is not strict UTF-8 JSON") from exc
    if not isinstance(payload, dict):
        raise ValueError("promotion artifact must contain an object")
    return artifact_path, payload


def _canonical_sha256(value: object) -> str | None:
    if (
        type(value) is not str
        or len(value) != 64
        or value != value.lower()
        or any(character not in "0123456789abcdef" for character in value)
    ):
        return None
    return value


def _to_ratio(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    try:
        numeric = float(value)
    except (OverflowError, ValueError):
        return None
    if not math.isfinite(numeric) or not 0.0 <= numeric <= 100.0:
        return None
    return numeric / 100.0 if numeric > 1.0 else numeric


def _to_nonnegative_finite_number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    try:
        numeric = float(value)
    except (OverflowError, ValueError):
        return None
    if not math.isfinite(numeric) or numeric < 0.0:
        return None
    return numeric


def _normalize_runtime_for_comparison(runtime: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(runtime, dict):
        return {}
    normalized: dict[str, Any] = {}
    for key in PROMOTED_RUNTIME_FIELDS:
        if key not in runtime:
            continue
        value = runtime[key]
        normalized[key] = list(value) if isinstance(value, list) else value
    return normalized


def _normalize_research_config_for_comparison(config: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(config, dict):
        return {}
    normalized: dict[str, Any] = {}
    for key in PROMOTED_RUNTIME_FIELDS:
        if key in config:
            normalized[key] = config[key]
    return normalized


def _strict_execution_context(
    runtime: object,
    *,
    label: str,
) -> dict[str, str | bool | int | float]:
    if not isinstance(runtime, dict):
        raise ValueError(f"{label} runtime metadata is missing or invalid")
    raw_context = runtime.get("data_parity")
    if not isinstance(raw_context, dict):
        raise ValueError(f"{label} runtime data_parity is missing or invalid")
    if set(raw_context) != set(EXECUTION_CONTEXT_FIELDS):
        missing = [key for key in EXECUTION_CONTEXT_FIELDS if key not in raw_context]
        unexpected = sorted(set(raw_context) - set(EXECUTION_CONTEXT_FIELDS))
        details: list[str] = []
        if missing:
            details.append("missing " + ", ".join(missing))
        if unexpected:
            details.append("unexpected " + ", ".join(unexpected))
        raise ValueError(
            f"{label} runtime data_parity fields are not exact"
            + (f" ({'; '.join(details)})" if details else "")
        )

    context: dict[str, str | bool | int | float] = {}
    for key in EXECUTION_CONTEXT_FIELDS[:3]:
        value = raw_context[key]
        if type(value) is not str or value not in _DATA_FEED_VALUES:
            raise ValueError(
                f"{label} runtime data_parity.{key} is missing or invalid"
            )
        context[key] = value
    bar_build_mode = raw_context["bar_build_mode"]
    if type(bar_build_mode) is not str or bar_build_mode not in _BAR_BUILD_MODE_VALUES:
        raise ValueError(
            f"{label} runtime data_parity.bar_build_mode is missing or invalid"
        )
    context["bar_build_mode"] = bar_build_mode
    for key in EXECUTION_CONTEXT_FIELDS[4:]:
        value = raw_context[key]
        if type(value) is not bool:
            raise ValueError(
                f"{label} runtime data_parity.{key} is missing or invalid"
            )
        context[key] = value
    try:
        risk_context = validate_risk_execution_context(
            {field: runtime.get(field) for field in RISK_EXECUTION_CONTEXT_FIELDS}
        )
    except ResearchBundleValidationError as exc:
        raise ValueError(
            f"{label} runtime risk execution context is missing or invalid ({exc})"
        ) from exc
    context.update(risk_context)
    return context


def _strict_runtime_filters(
    runtime: object,
    *,
    label: str,
) -> dict[str, object]:
    if not isinstance(runtime, dict):
        raise ValueError(f"{label} runtime metadata is missing or invalid")
    raw_symbols = runtime.get("symbols")
    if (
        not isinstance(raw_symbols, list)
        or not raw_symbols
        or any(type(symbol) is not str or not symbol.strip() for symbol in raw_symbols)
    ):
        raise ValueError(f"{label} runtime symbols are missing or invalid")
    active_symbols = {symbol.strip().upper() for symbol in raw_symbols}

    raw_excluded = runtime.get("excluded_symbols")
    if raw_excluded is None:
        excluded_values: list[str] = []
    elif type(raw_excluded) is str:
        excluded_values = [
            value.strip().upper()
            for value in raw_excluded.split(",")
            if value.strip()
        ]
    elif isinstance(raw_excluded, list) and all(
        type(value) is str for value in raw_excluded
    ):
        excluded_values = [
            value.strip().upper() for value in raw_excluded if value.strip()
        ]
    else:
        raise ValueError(f"{label} runtime excluded_symbols is invalid")
    excluded = tuple(sorted(set(excluded_values)))
    overlap = sorted(active_symbols & set(excluded))
    if overlap:
        raise ValueError(
            f"{label} runtime exclusions remove active canonical symbols {overlap!r}"
        )

    raw_blackouts = runtime.get("symbol_event_blackouts")
    if raw_blackouts is None:
        blackout_values: list[dict[str, str]] = []
    elif isinstance(raw_blackouts, list):
        blackout_values = []
        for index, item in enumerate(raw_blackouts):
            if not isinstance(item, dict) or not set(item) <= {
                "symbol",
                "start_utc",
                "end_utc",
                "reason",
            }:
                raise ValueError(
                    f"{label} runtime symbol_event_blackouts[{index}] is invalid"
                )
            symbol = item.get("symbol")
            start_utc = item.get("start_utc")
            end_utc = item.get("end_utc")
            reason = item.get("reason", "event_blackout")
            if (
                type(symbol) is not str
                or not symbol.strip()
                or type(start_utc) is not str
                or not start_utc.strip()
                or type(end_utc) is not str
                or not end_utc.strip()
                or type(reason) is not str
                or not reason.strip()
            ):
                raise ValueError(
                    f"{label} runtime symbol_event_blackouts[{index}] is invalid"
                )
            blackout_values.append(
                {
                    "symbol": symbol.strip().upper(),
                    "start_utc": start_utc.strip(),
                    "end_utc": end_utc.strip(),
                    "reason": reason.strip(),
                }
            )
    else:
        raise ValueError(f"{label} runtime symbol_event_blackouts is invalid")
    return {
        "excluded_symbols": excluded,
        "symbol_event_blackouts": tuple(
            tuple(item.items()) for item in blackout_values
        ),
    }


def _strict_effective_runtime_binding(
    config: object,
    *,
    label: str,
) -> tuple[
    CanonicalResolvedPromotedStrategy,
    dict[str, str | bool | int | float],
    dict[str, object],
]:
    """Normalize the already-resolved in-memory BotConfig without file I/O."""

    try:
        raw_blackouts = getattr(config, "symbol_event_blackouts")
        raw_excluded = getattr(config, "excluded_symbols")
        raw_symbols = getattr(config, "symbols")
        data_parity = {
            field: getattr(config, field) for field in EXECUTION_CONTEXT_FIELDS
        }
        risk_context = {
            field: getattr(config, field) for field in RISK_EXECUTION_CONTEXT_FIELDS
        }
    except AttributeError as exc:
        raise ValueError(
            f"{label} is missing canonical execution or filter fields"
        ) from exc
    if not isinstance(raw_blackouts, (list, tuple)):
        raise ValueError(f"{label} symbol_event_blackouts is invalid")
    blackout_payloads: list[dict[str, object]] = []
    for index, item in enumerate(raw_blackouts):
        if isinstance(item, dict):
            values = {
                key: item.get(key)
                for key in ("symbol", "start_utc", "end_utc", "reason")
            }
        else:
            try:
                values = {
                    key: getattr(item, key)
                    for key in ("symbol", "start_utc", "end_utc", "reason")
                }
            except AttributeError as exc:
                raise ValueError(
                    f"{label} symbol_event_blackouts[{index}] is invalid"
                ) from exc
        blackout_payloads.append(values)
    if not isinstance(raw_excluded, (list, tuple)):
        raise ValueError(f"{label} excluded_symbols is invalid")
    runtime = {
        "symbols": raw_symbols,
        "excluded_symbols": list(raw_excluded),
        "symbol_event_blackouts": blackout_payloads,
        "data_parity": data_parity,
        **risk_context,
    }
    try:
        contract = CanonicalResolvedPromotedStrategy.from_bot_config(config)
    except ContractValidationError as exc:
        raise ValueError(f"{label} canonical strategy is invalid ({exc})") from exc
    return (
        contract,
        _strict_execution_context(runtime, label=label),
        _strict_runtime_filters(runtime, label=label),
    )


def load_backtest_baseline(
    repo_root: Path | None = None,
    runtime_config_path: str | Path | None = None,
    *,
    expected_runtime_config_sha256: str | None = None,
    effective_runtime_config: object | None = None,
    effective_canonical_strategy_sha256: str | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    base = dict(DEFAULT_BACKTEST)
    source = {
        "mode": "defaults",
        "live_config_path": None,
        "research_config_path": None,
        "promotion_pointer_path": None,
        "promotion_pointer_valid": False,
        "promotion_bundle_path": None,
        "research_generation_sha256": None,
        "promotion_bundle_sha256": None,
        "research_source_content_sha256": None,
        "research_metrics_used": False,
        "live_config_approved": None,
        "research_approved": None,
        "research_matches_live_runtime": None,
        "research_matches_live_symbols": None,
        "execution_context_matches_pointer_live": None,
        "execution_context_validation_errors": [],
        "selected_runtime_config_sha256": None,
        "effective_runtime_binding_valid": None,
        "effective_runtime_binding_errors": [],
        "validation_errors": [],
        "valid_for_comparison": False,
        "metric_sources": {key: METRIC_SOURCE_UNAVAILABLE for key in DEFAULT_BACKTEST},
    }
    metric_sources: dict[str, str] = source["metric_sources"]

    effective_binding_requested = any(
        value is not None
        for value in (
            expected_runtime_config_sha256,
            effective_runtime_config,
            effective_canonical_strategy_sha256,
        )
    )
    effective_binding_inputs_complete = all(
        value is not None
        for value in (
            expected_runtime_config_sha256,
            effective_runtime_config,
            effective_canonical_strategy_sha256,
        )
    )
    if effective_binding_requested and not effective_binding_inputs_complete:
        binding_error = (
            "effective runtime binding inputs are missing; source bytes, in-memory "
            "config, and canonical identity are all required"
        )
        source["effective_runtime_binding_valid"] = False
        source["effective_runtime_binding_errors"].append(binding_error)
        source["validation_errors"].append(binding_error)

    root = (repo_root or Path(__file__).resolve().parent).resolve(strict=False)
    pointer_path = (root / "results" / "promotion_current.json").resolve(strict=False)
    source["promotion_pointer_path"] = str(pointer_path)
    pointer_live_path: Path | None = None
    pointer_live_config: dict[str, Any] | None = None
    pointer_live_sha256: str | None = None
    pointer_research_path: Path | None = None
    pointer_research: dict[str, Any] | None = None
    pointer_bundle_contract: CanonicalResolvedPromotedStrategy | None = None
    try:
        pointer = load_verified_promotion_pointer(pointer_path, repo_root=root)
        pointer_payload = pointer.payload
        artifacts = pointer_payload.get("artifacts")
        bundle_reference = pointer_payload.get("bundle")
        if not isinstance(artifacts, dict) or not isinstance(bundle_reference, dict):
            raise ValueError("promotion pointer references are missing or invalid")
        pointer_bundle_path, pointer_bundle = _load_pointer_bound_json(
            root,
            {
                "path": bundle_reference.get("path"),
                "sha256": bundle_reference.get("file_sha256"),
            },
        )
        strategy_identity = pointer_bundle.get("strategy")
        if not isinstance(strategy_identity, dict):
            raise ValueError("promotion bundle strategy identity is missing or invalid")
        canonical_json = strategy_identity.get("canonical_json")
        canonical_sha256 = strategy_identity.get("canonical_sha256")
        if type(canonical_json) is not str or type(canonical_sha256) is not str:
            raise ValueError("promotion bundle canonical strategy is missing or invalid")
        try:
            pointer_bundle_contract = CanonicalResolvedPromotedStrategy.from_json(
                canonical_json
            )
        except ContractValidationError as exc:
            raise ValueError("promotion bundle canonical strategy is invalid") from exc
        if (
            canonical_json != pointer_bundle_contract.canonical_json()
            or canonical_sha256 != pointer_bundle_contract.sha256()
            or pointer_bundle.get("bundle_sha256")
            != bundle_reference.get("bundle_sha256")
        ):
            raise ValueError("promotion bundle strategy or identity is inconsistent")
        pointer_live_path, pointer_live_config = _load_pointer_bound_json(
            root,
            artifacts.get("live_config"),
        )
        live_reference = artifacts.get("live_config")
        if not isinstance(live_reference, dict):
            raise ValueError("promotion live artifact reference is missing or invalid")
        pointer_live_sha256 = _canonical_sha256(live_reference.get("sha256"))
        if pointer_live_sha256 is None:
            raise ValueError("promotion live artifact SHA-256 is missing or invalid")
        pointer_research_path, pointer_research = _load_pointer_bound_json(
            root,
            artifacts.get("best_config_latest"),
        )
        generation_sha256 = pointer_payload.get("research_generation_sha256")
        bundle_sha256 = bundle_reference.get("bundle_sha256")
        source_identity = pointer_bundle.get("source_identity")
        source_content_sha256 = (
            source_identity.get("content_sha256")
            if isinstance(source_identity, dict)
            else None
        )
        generation_sha256 = _canonical_sha256(generation_sha256)
        bundle_sha256 = _canonical_sha256(bundle_sha256)
        source_content_sha256 = _canonical_sha256(source_content_sha256)
        if generation_sha256 is None or bundle_sha256 is None:
            raise ValueError("promotion pointer identities are missing or invalid")
        if source_content_sha256 is None:
            raise ValueError("promotion research source identity is missing or invalid")
    except (OSError, RuntimeError, TypeError, UnicodeError, ValueError) as exc:
        source["validation_errors"].append(
            f"promotion pointer is missing or invalid ({exc})"
        )
        pointer_live_path = None
        pointer_live_config = None
        pointer_live_sha256 = None
        pointer_research_path = None
        pointer_research = None
        pointer_bundle_contract = None
    else:
        source["promotion_pointer_valid"] = True
        source["promotion_bundle_path"] = str(pointer_bundle_path)
        source["research_generation_sha256"] = generation_sha256
        source["promotion_bundle_sha256"] = bundle_sha256
        source["research_source_content_sha256"] = source_content_sha256

    selected_runtime_path: Path | None = None
    if runtime_config_path is None:
        selected_runtime_path = pointer_live_path or (
            root / "config" / "live_config.json"
        ).resolve(strict=False)
    else:
        try:
            candidate = Path(runtime_config_path).expanduser()
            if not candidate.is_absolute():
                candidate = root / candidate
            candidate = candidate.resolve(strict=False)
            candidate.relative_to(root)
        except (OSError, RuntimeError, TypeError, ValueError):
            source["validation_errors"].append(
                "runtime config path is invalid or outside the repository"
            )
        else:
            selected_runtime_path = candidate

    live_config = None
    selected_runtime_config_sha256: str | None = None
    if selected_runtime_path is not None:
        if pointer_live_path is not None and selected_runtime_path == pointer_live_path:
            live_config = pointer_live_config
            selected_runtime_config_sha256 = pointer_live_sha256
        else:
            live_config, selected_runtime_config_sha256 = (
                _load_json_file_with_sha256(selected_runtime_path)
            )
    source["selected_runtime_config_sha256"] = selected_runtime_config_sha256
    live_runtime: dict[str, Any] | None = None
    live_symbols: list[str] | None = None
    live_contract: CanonicalResolvedPromotedStrategy | None = None
    if selected_runtime_path is not None:
        source["live_config_path"] = str(selected_runtime_path)
    if live_config is None and selected_runtime_path is not None:
        source["validation_errors"].append("live config is missing or invalid")
    elif live_config is not None:
        runtime = live_config.get("runtime")
        if isinstance(runtime, dict):
            live_runtime = runtime
            symbols = runtime.get("symbols")
            if not isinstance(symbols, list) or not symbols:
                source["validation_errors"].append(
                    "live runtime symbols are missing or invalid"
                )
            source["mode"] = "live_config"
        else:
            source["validation_errors"].append("live config runtime metadata is missing or invalid")
        live_source = live_config.get("source")
        if isinstance(live_source, dict) and isinstance(live_source.get("approved"), bool):
            source["live_config_approved"] = live_source["approved"]
            if source["live_config_approved"] is not True:
                reasons = live_source.get("rejection_reasons") or []
                reasons_text = ", ".join(str(item) for item in reasons if str(item))
                detail = f"live config is approved=false"
                if reasons_text:
                    detail = f"{detail} ({reasons_text})"
                source["validation_errors"].append(detail)
        else:
            source["validation_errors"].append(
                "live config approval metadata is missing or invalid"
            )
        if source["promotion_pointer_valid"] is True:
            live_generation = (
                live_source.get("research_generation_sha256")
                if isinstance(live_source, dict)
                else None
            )
            if type(live_generation) is not str:
                source["validation_errors"].append(
                    "live config research generation is missing or invalid"
                )
            elif live_generation != source["research_generation_sha256"]:
                source["validation_errors"].append(
                    "live config research generation does not match the committed promotion"
                )

        live_runtime_cmp = _normalize_runtime_for_comparison(live_runtime)
        missing_live_fields = [
            key for key in PROMOTED_RUNTIME_FIELDS if key not in live_runtime_cmp
        ]
        if missing_live_fields:
            source["validation_errors"].append(
                "live runtime is missing promoted fields ("
                + ", ".join(missing_live_fields)
                + ")"
            )
        elif isinstance(live_runtime, dict):
            try:
                live_contract = CanonicalResolvedPromotedStrategy.from_mapping(
                    live_runtime.get("symbols"),
                    live_runtime_cmp,
                )
            except ContractValidationError as exc:
                source["validation_errors"].append(
                    f"live runtime canonical contract is invalid ({exc})"
                )
            else:
                live_symbols = list(live_contract.symbols)
                if (
                    pointer_bundle_contract is not None
                    and live_contract.canonical_bytes()
                    != pointer_bundle_contract.canonical_bytes()
                ):
                    source["validation_errors"].append(
                        "live runtime canonical contract does not match the committed promotion bundle"
                    )
                base["symbols"] = len(live_symbols)
                timeframe_minutes = live_contract.promoted_values[
                    "bar_timeframe_minutes"
                ]
                base["bars_per_day"] = max(
                    1,
                    int((6.5 * 60) / timeframe_minutes),
                )
                metric_sources["symbols"] = METRIC_SOURCE_LIVE_CONFIG
                metric_sources["bars_per_day"] = METRIC_SOURCE_LIVE_CONFIG

    selected_context: dict[str, str | bool | int | float] | None = None
    selected_filters: dict[str, object] | None = None
    if pointer_live_config is not None and live_config is not None:
        try:
            pointer_runtime = pointer_live_config.get("runtime")
            committed_context = _strict_execution_context(
                pointer_runtime,
                label="pointer live config",
            )
            committed_filters = _strict_runtime_filters(
                pointer_runtime,
                label="pointer live config",
            )
            selected_context = _strict_execution_context(
                live_runtime,
                label="selected live config",
            )
            selected_filters = _strict_runtime_filters(
                live_runtime,
                label="selected live config",
            )
            mismatches = [
                key
                for key in EXECUTION_CONTEXT_COMPARISON_FIELDS
                if selected_context[key] != committed_context[key]
            ]
            if mismatches:
                details = "; ".join(
                    f"{key}: selected={selected_context[key]!r} "
                    f"pointer={committed_context[key]!r}"
                    for key in mismatches
                )
                raise ValueError(
                    "selected runtime execution context does not match the "
                    f"pointer live config ({details})"
                )
            filter_mismatches = [
                key
                for key in ("excluded_symbols", "symbol_event_blackouts")
                if selected_filters[key] != committed_filters[key]
            ]
            if filter_mismatches:
                raise ValueError(
                    "selected runtime filters do not match the pointer live config "
                    f"({', '.join(filter_mismatches)})"
                )
        except (KeyError, TypeError, ValueError) as exc:
            context_error = f"runtime execution context is missing or invalid ({exc})"
            source["execution_context_validation_errors"].append(context_error)
            source["validation_errors"].append(context_error)
            source["execution_context_matches_pointer_live"] = False
        else:
            source["execution_context_matches_pointer_live"] = True

    if effective_binding_requested and effective_binding_inputs_complete:
        try:
            expected_source_sha256 = _canonical_sha256(
                expected_runtime_config_sha256
            )
            expected_canonical_sha256 = _canonical_sha256(
                effective_canonical_strategy_sha256
            )
            if expected_source_sha256 is None:
                raise ValueError(
                    "first-read runtime config SHA-256 is missing or invalid"
                )
            if expected_canonical_sha256 is None:
                raise ValueError(
                    "effective canonical strategy SHA-256 is missing or invalid"
                )
            if selected_runtime_config_sha256 != expected_source_sha256:
                raise ValueError(
                    "runtime config bytes changed after the effective BotConfig was resolved"
                )
            if live_contract is None or selected_context is None or selected_filters is None:
                raise ValueError(
                    "selected runtime canonical, execution, or filter context is unavailable"
                )
            (
                effective_contract,
                effective_context,
                effective_filters,
            ) = _strict_effective_runtime_binding(
                effective_runtime_config,
                label="effective in-memory config",
            )
            if (
                effective_contract.sha256() != expected_canonical_sha256
                or live_contract.sha256() != expected_canonical_sha256
            ):
                raise ValueError(
                    "effective in-memory canonical strategy does not match the selected runtime"
                )
            context_mismatches = [
                key
                for key in EXECUTION_CONTEXT_COMPARISON_FIELDS
                if effective_context[key] != selected_context[key]
            ]
            filter_mismatches = [
                key
                for key in ("excluded_symbols", "symbol_event_blackouts")
                if effective_filters[key] != selected_filters[key]
            ]
            if context_mismatches or filter_mismatches:
                details = ", ".join((*context_mismatches, *filter_mismatches))
                raise ValueError(
                    "effective in-memory execution/filter context does not match "
                    f"the selected runtime ({details})"
                )
        except (KeyError, TypeError, ValueError) as exc:
            binding_error = f"effective runtime binding failed ({exc})"
            source["effective_runtime_binding_errors"].append(binding_error)
            source["validation_errors"].append(binding_error)
            source["effective_runtime_binding_valid"] = False
        else:
            source["effective_runtime_binding_valid"] = True

    research_path = pointer_research_path or (
        root / "results" / "best_config_latest.json"
    ).resolve(strict=False)
    research = (
        pointer_research
        if pointer_research_path is not None
        else _load_json_file(research_path)
    )
    if research is None:
        source["validation_errors"].append("research baseline artifact is missing or invalid")
    else:
        source["research_config_path"] = str(research_path)
        if isinstance(research.get("approved"), bool):
            source["research_approved"] = research["approved"]
            if source["research_approved"] is not True:
                reasons = research.get("rejection_reasons") or []
                reasons_text = ", ".join(str(item) for item in reasons if str(item))
                detail = f"research artifact is approved=false"
                if reasons_text:
                    detail = f"{detail} ({reasons_text})"
                source["validation_errors"].append(detail)
        else:
            source["validation_errors"].append(
                "research approval metadata is missing or invalid"
            )

        persisted_research_contract: CanonicalResolvedPromotedStrategy | None = None
        try:
            persisted_research_contract = (
                CanonicalResolvedPromotedStrategy.from_payload(
                    research.get("canonical_strategy_contract")
                )
            )
        except ContractValidationError as exc:
            source["validation_errors"].append(
                "research canonical strategy contract is missing or invalid "
                f"({exc})"
            )

        raw_research_config = research.get("config")
        research_config = _normalize_research_config_for_comparison(raw_research_config)
        research_symbols: list[str] | None = None
        research_contract: CanonicalResolvedPromotedStrategy | None = None
        flat_research_contract: CanonicalResolvedPromotedStrategy | None = None
        if isinstance(raw_research_config, dict):
            raw_research_symbols = raw_research_config.get("symbols")
            if isinstance(raw_research_symbols, list) and raw_research_symbols:
                research_symbols = raw_research_symbols
        if research_symbols is None:
            source["validation_errors"].append(
                "research baseline symbols are missing or invalid"
            )
        missing_research_fields = [
            key for key in PROMOTED_RUNTIME_FIELDS if key not in research_config
        ]
        if missing_research_fields:
            source["validation_errors"].append(
                "research baseline is missing promoted fields ("
                + ", ".join(missing_research_fields)
                + ")"
            )
        elif research_symbols is not None:
            try:
                flat_research_contract = CanonicalResolvedPromotedStrategy.from_mapping(
                    research_symbols,
                    research_config,
                )
            except ContractValidationError as exc:
                source["validation_errors"].append(
                    f"research canonical contract is invalid ({exc})"
                )

        if (
            flat_research_contract is not None
            and persisted_research_contract is not None
        ):
            if (
                flat_research_contract.canonical_bytes()
                != persisted_research_contract.canonical_bytes()
            ):
                source["validation_errors"].append(
                    "research canonical strategy contract does not match its "
                    "flat compatibility config"
                )
            else:
                research_contract = persisted_research_contract

        if live_contract is not None and research_contract is not None:
            live_symbols = list(live_contract.symbols)
            research_symbols = list(research_contract.symbols)
            source["research_matches_live_symbols"] = (
                research_contract.symbols == live_contract.symbols
            )
            if source["research_matches_live_symbols"] is not True:
                source["validation_errors"].append(
                    "research symbols do not match live runtime "
                    f"(live={live_symbols!r} research={research_symbols!r})"
                )

            live_runtime_cmp = live_contract.promoted_values
            research_runtime_cmp = research_contract.promoted_values
            mismatches: list[str] = []
            for key in PROMOTED_RUNTIME_FIELDS:
                research_value = research_runtime_cmp[key]
                if live_runtime_cmp[key] != research_value:
                    mismatches.append(f"{key}: live={live_runtime_cmp[key]!r} research={research_value!r}")
            source["research_matches_live_runtime"] = len(mismatches) == 0
            if mismatches:
                source["validation_errors"].append(
                    "research config does not match live runtime (" + "; ".join(mismatches) + ")"
                )

        performance = research.get("performance")
        trades_per_day_value: float | None = None
        win_rate: float | None = None
        if not isinstance(performance, dict):
            source["validation_errors"].append(
                "research performance metadata is missing or invalid"
            )
        else:
            trades_per_day_value = _to_nonnegative_finite_number(
                performance.get("trades_per_day")
            )
            if trades_per_day_value is None:
                source["validation_errors"].append(
                    "research trades_per_day is missing or invalid"
                )
            win_rate = _to_ratio(performance.get("win_rate"))
            if win_rate is None:
                source["validation_errors"].append(
                    "research win_rate is missing or invalid"
                )

        research_valid = (
            isinstance(performance, dict)
            and source["promotion_pointer_valid"] is True
            and source["research_approved"] is True
            and source["research_matches_live_runtime"] is True
            and source["research_matches_live_symbols"] is True
            and source["execution_context_matches_pointer_live"] is True
            and source["live_config_approved"] is True
            and not source["validation_errors"]
            and trades_per_day_value is not None
            and win_rate is not None
        )
        if research_valid:
            if base["symbols"] > 0:
                base["signal_rate_per_symbol_per_day"] = trades_per_day_value / base["symbols"]
                metric_sources["signal_rate_per_symbol_per_day"] = METRIC_SOURCE_RESEARCH
            base["win_rate"] = win_rate
            metric_sources["win_rate"] = METRIC_SOURCE_RESEARCH
            source["research_metrics_used"] = True
            source["mode"] = "live_config+research" if source["mode"] == "live_config" else "research"

    source["valid_for_comparison"] = (
        len(source["validation_errors"]) == 0
        and source["promotion_pointer_valid"] is True
        and source["live_config_approved"] is True
        and source["research_approved"] is True
        and source["research_matches_live_runtime"] is True
        and source["research_matches_live_symbols"] is True
        and source["execution_context_matches_pointer_live"] is True
        and source["research_metrics_used"] is True
        and (
            not effective_binding_requested
            or source["effective_runtime_binding_valid"] is True
        )
    )

    return base, source


BACKTEST, BACKTEST_SOURCE = load_backtest_baseline()

# ---------------------------------------------------------------------------
# Alert thresholds
# ---------------------------------------------------------------------------

WARN: dict[str, Any] = {
    "avg_slippage_bps":          15.0,   # bps — execution cost is eroding edge
    "worst_slippage_bps":        40.0,   # bps — single bad fill
    "signal_rate_pct_of_bt":     0.50,   # live rate below 50% of backtest → filter drift
    "signal_rate_pct_of_bt_hi":  2.00,   # live rate above 200% of backtest → filter not working
    "win_rate_low":               0.50,   # below this → strategy deteriorating
    "win_rate_high":              0.90,   # above this → too few trades, not meaningful
    "eod_exit_rate":              0.30,   # >30% of exits from EOD flatten → exits not firing
    "stale_bar_count":            2,      # stale bars during market hours = data feed issue
    "avg_bar_age_s":              240.0,  # seconds — IEX feed has ~4-5 min publication delay; warn if approaching the 300s stale cutoff
    "min_trades_for_win_rate":    5,      # don't report win rate below this sample size
}


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------

def _load(path: Path) -> pd.DataFrame:
    """Read a JSONL file into a DataFrame. Returns empty DataFrame if missing."""
    if not path.exists():
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    malformed_lines = 0
    total_lines = 0
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        total_lines += 1
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            malformed_lines += 1
            continue
        if isinstance(payload, dict):
            rows.append(payload)

    df = pd.DataFrame(rows)
    df.attrs["source_path"] = str(path)
    df.attrs["raw_line_count"] = total_lines
    df.attrs["malformed_line_count"] = malformed_lines
    df.attrs["loaded_row_count"] = len(rows)
    return df


def _dedupe(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        df.attrs["duplicate_row_count"] = 0
        df.attrs["unique_row_count"] = 0
        return df

    # A trace identifies one decision context, not one immutable event. Partial
    # fills, lifecycle transitions, and retries can legitimately share the same
    # event/trace pair while carrying different state. Only discard exact rows.
    fingerprints = [
        json.dumps(record, sort_keys=True, default=str, separators=(",", ":"))
        for record in df.to_dict(orient="records")
    ]
    duplicate_mask = pd.Series(fingerprints, index=df.index).duplicated(keep="first")
    deduped = df.loc[~duplicate_mask].copy()
    deduped.attrs.update(df.attrs)
    deduped.attrs["duplicate_row_count"] = len(df) - len(deduped)
    deduped.attrs["unique_row_count"] = len(deduped)
    return deduped


def load_logs(log_dir: Path) -> dict[str, pd.DataFrame]:
    return {
        "bars":      _dedupe(_load(log_dir / "bars.jsonl")),
        "signals":   _dedupe(_load(log_dir / "signals.jsonl")),
        "risk":      _dedupe(_load(log_dir / "risk.jsonl")),
        "execution": _dedupe(_load(log_dir / "execution.jsonl")),
        "positions": _dedupe(_load(log_dir / "positions.jsonl")),
    }


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------

def _col(df: pd.DataFrame, col: str) -> pd.Series:
    """Return column or empty Series if it doesn't exist."""
    return df[col] if col in df.columns else pd.Series(dtype=object)


def _flag(value: float, warn_threshold: float, low_is_bad: bool = True) -> str:
    """Return 'WARN' or 'ok' based on whether value breaches threshold."""
    if low_is_bad:
        return "WARN" if value < warn_threshold else "ok"
    else:
        return "WARN" if value > warn_threshold else "ok"


def baseline_comparison_status() -> dict[str, Any]:
    """Fail-closed view of whether live metrics may be compared to BACKTEST.

    ``BACKTEST`` always holds numbers, falling back to ``DEFAULT_BACKTEST``
    when the saved baseline is missing or fails validation.  Those fallback
    numbers are not an approved expectation, so every live-vs-backtest
    statement must be gated on this status rather than on ``BACKTEST`` alone.

    ``sources`` maps every ``BACKTEST`` key to where its expectation came from:
    ``research`` (pointer-bound research artifact), ``live_config`` (validated
    live runtime), or ``unavailable`` (no validated artifact carries it, so the
    value in ``BACKTEST`` is only a ``DEFAULT_BACKTEST`` placeholder).  Even
    with a valid baseline, an ``unavailable`` metric has no expectation.
    """
    if BACKTEST_SOURCE.get("valid_for_comparison") is True:
        raw_sources = BACKTEST_SOURCE.get("metric_sources")
        if not isinstance(raw_sources, dict):
            raw_sources = {}
        sources = {
            key: (
                raw_sources[key]
                if raw_sources.get(key) in _LOADED_METRIC_SOURCES
                else METRIC_SOURCE_UNAVAILABLE
            )
            for key in BACKTEST
        }
        return {"available": True, "status": "available", "reasons": [], "sources": sources}
    reasons = [
        str(item)
        for item in (BACKTEST_SOURCE.get("validation_errors") or [])
        if str(item)
    ]
    return {
        "available": False,
        "status": "unavailable",
        "reasons": reasons or ["baseline validation failed"],
        "sources": {key: METRIC_SOURCE_UNAVAILABLE for key in BACKTEST},
    }


_REASON_SUMMARY_MAX_ITEMS = 5
_REASON_SUMMARY_PREVIEW_ITEMS = 3
_OPENERS = {")": "(", "]": "[", "}": "{"}


def _split_top_level(text: str, separator: str) -> list[str]:
    """Split on ``separator`` only outside (), [] and {} nesting."""
    parts: list[str] = []
    depth = 0
    start = 0
    index = 0
    while index < len(text):
        char = text[index]
        if char in "([{":
            depth += 1
        elif char in ")]}":
            depth = max(0, depth - 1)
        elif depth == 0 and text.startswith(separator, index):
            parts.append(text[start:index])
            index += len(separator)
            start = index
            continue
        index += 1
    parts.append(text[start:])
    return [part.strip() for part in parts if part.strip()]


def _trailing_parenthetical(text: str) -> tuple[str, str] | None:
    """Return (head, inner) for text ending in a balanced "(...)" group."""
    if not text.endswith(")"):
        return None
    stack: list[str] = []
    for index in range(len(text) - 1, -1, -1):
        char = text[index]
        if char in _OPENERS:
            stack.append(_OPENERS[char])
        elif char in "([{":
            if not stack or stack.pop() != char:
                return None
            if not stack:
                if char != "(":
                    return None
                return text[:index].rstrip(), text[index + 1:-1]
    return None


def summarize_validation_reason(reason: str) -> str:
    """Shorten one baseline validation error for human-facing output.

    Long trailing lists such as ``live runtime is missing promoted fields
    (a, b, ... 40 more)`` become ``live runtime is missing 43 promoted
    fields``; other long lists keep their first few items plus a count.
    Short reasons are returned unchanged.  The JSON ``comparison.reasons``
    always carries the full, unsummarized text.
    """
    text = str(reason)
    split = _trailing_parenthetical(text)
    if split is None:
        return text
    head, inner = split
    separator = "; "
    items = _split_top_level(inner, separator)
    if len(items) <= 1:
        separator = ", "
        items = _split_top_level(inner, separator)
    if len(items) <= 1:
        summarized_inner = summarize_validation_reason(inner)
        return text if summarized_inner == inner else f"{head} ({summarized_inner})"
    if len(items) <= _REASON_SUMMARY_MAX_ITEMS:
        return text
    subject, marker, noun = head.partition(" is missing ")
    if marker and noun.endswith("fields") and " or " not in noun:
        return f"{subject} is missing {len(items)} {noun}"
    preview = separator.join(items[:_REASON_SUMMARY_PREVIEW_ITEMS])
    remaining = len(items) - _REASON_SUMMARY_PREVIEW_ITEMS
    return f"{head} ({preview}{separator}and {remaining} more)"


def _expectation(comparison: dict[str, Any], key: str) -> Any:
    """Return BACKTEST[key] only when it is a validated, artifact-backed value."""
    if comparison.get("available") is not True:
        return None
    if comparison.get("sources", {}).get(key) not in _LOADED_METRIC_SOURCES:
        return None
    return BACKTEST.get(key)


def _signal_rate_comparable(comparison: dict[str, Any]) -> bool:
    """Signal-rate checks need the research rate and the live session shape."""
    return all(
        _expectation(comparison, key) is not None
        for key in ("signal_rate_per_symbol_per_day", "symbols", "bars_per_day")
    )


def _session_signal_rate(sig: dict) -> tuple[float | None, float | None, float | None]:
    total_evaluated = sig.get("total_evaluated", 0) or 0
    buy_signals = sig.get("buy_signals", 0) or 0
    if total_evaluated <= 0 or BACKTEST["symbols"] <= 0:
        return None, None, None

    bars_per_symbol_seen = total_evaluated / BACKTEST["symbols"]
    if bars_per_symbol_seen <= 0:
        return None, None, None

    coverage_pct = bars_per_symbol_seen / BACKTEST["bars_per_day"]
    full_day_equiv = (buy_signals / BACKTEST["symbols"]) * (BACKTEST["bars_per_day"] / bars_per_symbol_seen)
    return bars_per_symbol_seen, coverage_pct, full_day_equiv


# ---------------------------------------------------------------------------
# Section computations
# ---------------------------------------------------------------------------

def compute_bar_quality(bars: pd.DataFrame) -> dict:
    received = bars[_col(bars, "event") == "bar.received"] if not bars.empty else pd.DataFrame()
    if received.empty:
        return {"evaluated": 0, "avg_age_s": None, "stale_count": 0, "symbols": []}

    return {
        "evaluated":   len(received),
        "avg_age_s":   received["bar_age_s"].mean() if "bar_age_s" in received.columns else None,
        "stale_count": int(received["stale"].sum()) if "stale" in received.columns else 0,
        "symbols":     sorted(received["symbol"].unique().tolist()) if "symbol" in received.columns else [],
    }


def compute_signal_behavior(signals: pd.DataFrame) -> dict:
    evaluated = signals[_col(signals, "event") == "signal.evaluated"] if not signals.empty else pd.DataFrame()
    if evaluated.empty:
        return {
            "total_evaluated": 0, "buy_signals": 0, "sell_signals": 0,
            "hold_count": 0, "by_symbol": {}, "rejections": {}, "rejection_rate": None,
        }

    buys  = evaluated[_col(evaluated, "action") == "BUY"]
    sells = evaluated[_col(evaluated, "action") == "SELL"]
    holds = evaluated[_col(evaluated, "action") == "HOLD"]

    # Rejection breakdown (why HOLD fired when a signal might have been expected)
    rejections: dict[str, int] = {}
    if "rejection" in evaluated.columns:
        rej = evaluated["rejection"].dropna()
        rejections = rej.value_counts().to_dict()

    # Per-symbol BUY signal count
    by_symbol: dict[str, int] = {}
    if "symbol" in buys.columns:
        by_symbol = buys["symbol"].value_counts().to_dict()

    return {
        "total_evaluated": len(evaluated),
        "buy_signals":     len(buys),
        "sell_signals":    len(sells),
        "hold_count":      len(holds),
        "by_symbol":       by_symbol,
        "rejections":      rejections,
        "rejection_rate":  len(holds) / len(evaluated) if len(evaluated) > 0 else None,
    }


def compute_execution(execution: pd.DataFrame) -> dict:
    if execution.empty:
        return {"submitted_buys": 0, "submitted_sells": 0,
                "avg_slippage_bps": None, "worst_slippage_bps": None,
                "partial_fills": 0, "avg_latency_ms": None}

    submitted = execution[_col(execution, "event") == "order.submitted"]
    filled    = execution[_col(execution, "event") == "order.filled"]
    partials  = execution[_col(execution, "event") == "order.partial_fill"]

    buys  = submitted[_col(submitted, "side") == "buy"]  if not submitted.empty else pd.DataFrame()
    sells = submitted[_col(submitted, "side") == "sell"] if not submitted.empty else pd.DataFrame()

    slippage_series = (
        filled["slippage_bps"].dropna()
        if not filled.empty and "slippage_bps" in filled.columns
        else pd.Series(dtype=float)
    )
    latency_series = (
        filled["submit_to_fill_ms"].dropna()
        if not filled.empty and "submit_to_fill_ms" in filled.columns
        else pd.Series(dtype=float)
    )

    return {
        "submitted_buys":    len(buys),
        "submitted_sells":   len(sells),
        "avg_slippage_bps":  float(slippage_series.mean())  if len(slippage_series) > 0 else None,
        "worst_slippage_bps":float(slippage_series.max())   if len(slippage_series) > 0 else None,
        "partial_fills":     len(partials),
        "avg_latency_ms":    float(latency_series.mean())   if len(latency_series) > 0 else None,
    }


def compute_risk(risk: pd.DataFrame) -> dict:
    if risk.empty:
        return {
            "kill_switch_triggered": False,
            "buy_blocks_by_reason": {},
            "sell_blocks_by_reason": {},
        }

    checks = risk[_col(risk, "event") == "risk.check"] if not risk.empty else pd.DataFrame()
    triggered = risk[_col(risk, "event") == "kill_switch.triggered"] if not risk.empty else pd.DataFrame()

    buy_blocks: dict[str, int] = {}
    sell_blocks: dict[str, int] = {}
    if not checks.empty and "block_reason" in checks.columns:
        blocked = checks[checks["block_reason"].notna()]
        if not blocked.empty and "action" in blocked.columns:
            buy_blocked  = blocked[blocked["action"] == "BUY"]
            sell_blocked = blocked[blocked["action"] == "SELL"]
            buy_blocks   = buy_blocked["block_reason"].value_counts().to_dict()
            sell_blocks  = sell_blocked["block_reason"].value_counts().to_dict()
        else:
            buy_blocks = blocked["block_reason"].value_counts().to_dict()

    return {
        "kill_switch_triggered": len(triggered) > 0,
        "buy_blocks_by_reason":  buy_blocks,
        "sell_blocks_by_reason": sell_blocks,
    }


def compute_positions(positions: pd.DataFrame) -> dict:
    if positions.empty:
        return {
            "opened": 0, "closed": 0, "win_rate": None,
            "avg_hold_bars": None, "avg_pnl_usd": None,
            "exit_reasons": {}, "total_pnl_usd": None,
        }

    opened = positions[_col(positions, "event") == "position.opened"]
    closed = positions[_col(positions, "event") == "position.closed"]

    if closed.empty:
        return {
            "opened": len(opened), "closed": 0, "win_rate": None,
            "avg_hold_bars": None, "avg_pnl_usd": None,
            "exit_reasons": {}, "total_pnl_usd": None,
        }

    winners = closed["winner"].sum() if "winner" in closed.columns else 0
    win_rate = float(winners / len(closed)) if len(closed) > 0 else None

    avg_hold = float(closed["holding_bars"].mean()) if "holding_bars" in closed.columns else None
    avg_pnl  = float(closed["pnl_usd"].mean())     if "pnl_usd" in closed.columns else None
    total_pnl= float(closed["pnl_usd"].sum())      if "pnl_usd" in closed.columns else None

    exit_reasons: dict[str, int] = {}
    if "exit_reason" in closed.columns:
        exit_reasons = closed["exit_reason"].value_counts().to_dict()

    return {
        "opened":        len(opened),
        "closed":        len(closed),
        "win_rate":      win_rate,
        "avg_hold_bars": avg_hold,
        "avg_pnl_usd":   avg_pnl,
        "total_pnl_usd": total_pnl,
        "exit_reasons":  exit_reasons,
    }


def compute_log_health(logs: dict[str, pd.DataFrame]) -> dict:
    files: dict[str, dict[str, int]] = {}
    for name, df in logs.items():
        files[name] = {
            "raw_lines": int(df.attrs.get("raw_line_count", len(df))),
            "loaded_rows": int(df.attrs.get("loaded_row_count", len(df))),
            "unique_rows": int(df.attrs.get("unique_row_count", len(df))),
            "malformed_lines": int(df.attrs.get("malformed_line_count", 0)),
            "duplicate_rows": int(df.attrs.get("duplicate_row_count", 0)),
        }

    return {
        "files": files,
        "malformed_lines": sum(item["malformed_lines"] for item in files.values()),
        "duplicate_rows": sum(item["duplicate_rows"] for item in files.values()),
    }


# ---------------------------------------------------------------------------
# Concern detection
# ---------------------------------------------------------------------------

def detect_concerns(bar_q: dict, sig: dict, ex: dict, pos: dict, log_health: dict | None = None) -> list[str]:
    """
    Return a list of concern strings.  Empty list = everything looks normal.
    """
    concerns = []
    comparison = baseline_comparison_status()
    if not comparison["available"]:
        detail = "; ".join(
            summarize_validation_reason(item) for item in comparison["reasons"]
        )
        concerns.append(
            f"BASELINE INVALID: {detail} — comparison unavailable; "
            f"backtest-relative checks (signal rate, session coverage) suppressed"
        )

    # --- Slippage ---
    if ex["avg_slippage_bps"] is not None:
        if ex["avg_slippage_bps"] > WARN["avg_slippage_bps"]:
            concerns.append(
                f"SLIPPAGE HIGH: avg {ex['avg_slippage_bps']:.1f} bps "
                f"(threshold {WARN['avg_slippage_bps']} bps) — execution cost eroding edge"
            )
    if ex["worst_slippage_bps"] is not None:
        if ex["worst_slippage_bps"] > WARN["worst_slippage_bps"]:
            concerns.append(
                f"SLIPPAGE SPIKE: worst {ex['worst_slippage_bps']:.1f} bps "
                f"(threshold {WARN['worst_slippage_bps']} bps) — check that fill for anomalies"
            )

    # --- Signal rate (backtest-relative: only against a valid baseline) ---
    # Coverage and the full-day equivalent are both derived from the
    # baseline's symbols/bars_per_day, so they are suppressed with it.
    coverage_pct = full_day_signal_rate = None
    if _signal_rate_comparable(comparison):
        _, coverage_pct, full_day_signal_rate = _session_signal_rate(sig)
    if full_day_signal_rate is not None:
        live_rate = full_day_signal_rate
        bt_rate   = BACKTEST["signal_rate_per_symbol_per_day"]
        ratio     = live_rate / bt_rate if bt_rate > 0 else 0
        if ratio < WARN["signal_rate_pct_of_bt"]:
            concerns.append(
                f"SIGNAL RATE LOW: {live_rate:.1f}/symbol/day vs backtest {bt_rate} "
                f"({ratio:.0%} of backtest) — live filters may differ from backtest"
            )
        if ratio > WARN["signal_rate_pct_of_bt_hi"]:
            concerns.append(
                f"SIGNAL RATE HIGH: {live_rate:.1f}/symbol/day vs backtest {bt_rate} "
                f"({ratio:.0%} of backtest) — filter may not be applying correctly in live"
            )
        if coverage_pct is not None and coverage_pct < 1.0:
            concerns.append(
                f"PARTIAL SESSION: observed {coverage_pct:.0%} of a full session "
                f"({sig['total_evaluated']} unique signal evaluations) — normalized signal-rate estimate used"
            )

    # --- Win rate ---
    n_closed = pos.get("closed", 0)
    if n_closed >= WARN["min_trades_for_win_rate"] and pos["win_rate"] is not None:
        wr = pos["win_rate"]
        if wr < WARN["win_rate_low"]:
            concerns.append(
                f"WIN RATE LOW: {wr:.0%} (threshold {WARN['win_rate_low']:.0%}) "
                f"— strategy may not be working in current regime"
            )
        if wr > WARN["win_rate_high"]:
            concerns.append(
                f"WIN RATE HIGH: {wr:.0%} ({n_closed} trades) "
                f"— sample too small or streak, not statistically significant"
            )

    # --- EOD exit rate ---
    total_exits = sum(pos.get("exit_reasons", {}).values())
    eod_exits   = pos.get("exit_reasons", {}).get("eod_flatten", 0)
    if total_exits >= 3:
        eod_rate = eod_exits / total_exits
        if eod_rate > WARN["eod_exit_rate"]:
            concerns.append(
                f"EOD EXITS HIGH: {eod_exits}/{total_exits} exits ({eod_rate:.0%}) via EOD flatten "
                f"— positions not recovering within session; exit logic or SMA level may differ from backtest"
            )

    # --- Bar quality ---
    if bar_q["stale_count"] >= WARN["stale_bar_count"]:
        concerns.append(
            f"STALE BARS: {bar_q['stale_count']} stale bar(s) detected "
            f"— data feed latency or Alpaca connectivity issue"
        )
    if bar_q["avg_age_s"] is not None and bar_q["avg_age_s"] > WARN["avg_bar_age_s"]:
        concerns.append(
            f"BAR LATENCY: avg bar age {bar_q['avg_age_s']:.1f}s "
            f"(threshold {WARN['avg_bar_age_s']}s) — bot is processing stale data"
        )

    # --- Partial fills ---
    if ex["partial_fills"] > 0:
        concerns.append(
            f"PARTIAL FILLS: {ex['partial_fills']} partial fill(s) detected "
            f"— universe liquidity may have declined; check fill quality"
        )

    if log_health:
        malformed = log_health.get("malformed_lines", 0)
        duplicates = log_health.get("duplicate_rows", 0)
        if malformed > 0:
            concerns.append(
                f"LOG PARSE WARNINGS: skipped {malformed} malformed JSONL line(s) — inspect raw logs if this day matters"
            )
        if duplicates > 0:
            concerns.append(
                f"RESTART DUPLICATES: removed {duplicates} duplicate log row(s) by trace/event before analysis"
            )

    # --- Kill switch ---
    # Reported separately in the report, not as a "concern" here.

    return concerns


# ---------------------------------------------------------------------------
# Report formatter
# ---------------------------------------------------------------------------

def fmt(value: Any, fmt_str: str = "", fallback: str = "n/a") -> str:
    if value is None:
        return fallback
    if isinstance(value, float):
        return format(value, fmt_str) if fmt_str else f"{value:.2f}"
    return str(value)


def print_report(
    report_date: str,
    bar_q: dict,
    sig: dict,
    ex: dict,
    risk: dict,
    pos: dict,
    concerns: list[str],
    log_health: dict | None = None,
) -> None:
    comparison = baseline_comparison_status()
    comparable = comparison["available"]
    signal_comparable = _signal_rate_comparable(comparison)
    n_sym     = BACKTEST["symbols"]
    bt_bars   = BACKTEST["bars_per_day"]
    bt_win_rate     = _expectation(comparison, "win_rate")
    bt_hold_bars    = _expectation(comparison, "avg_hold_bars")
    bt_rejection    = _expectation(comparison, "rejection_rate")
    bt_slippage_bps = _expectation(comparison, "avg_slippage_bps")
    divider   = "─" * 60

    def section(title: str) -> None:
        print(f"\n{divider}")
        print(f"  {title}")
        print(divider)

    def row(label: str, value: str, note: str = "", flag: str = "") -> None:
        flag_str = f"  [{flag}]" if flag else ""
        note_str = f"  {note}" if note else ""
        print(f"  {label:<38} {value:<14}{flag_str}{note_str}")

    print(f"\n{'═' * 60}")
    print(f"  DAILY BOT REPORT  ·  {report_date}")
    print(f"{'═' * 60}")

    # ── Data quality ──────────────────────────────────────────────
    section("DATA QUALITY")
    if (
        _expectation(comparison, "symbols") is not None
        and _expectation(comparison, "bars_per_day") is not None
    ):
        expected_bars = n_sym * bt_bars
        row("Bars evaluated",        fmt(bar_q["evaluated"]),
            f"(expect ~{expected_bars} = {n_sym} syms × {bt_bars} bars)")
    else:
        row("Bars evaluated",        fmt(bar_q["evaluated"]))
    row("Baseline source",       str(BACKTEST_SOURCE.get("mode", "defaults")))
    reason_summaries = [
        summarize_validation_reason(item) for item in comparison["reasons"]
    ]
    if not comparable:
        row("Backtest comparison",   "unavailable", flag="WARN")
        print("  Baseline validation:")
        for item in reason_summaries:
            print(f"    - {item}")
        if reason_summaries != comparison["reasons"]:
            print("    (long lists summarized; full text in --json comparison.reasons)")
    avg_age = bar_q["avg_age_s"]
    row("Avg bar age (s)",        fmt(avg_age, ".1f"),
        flag=_flag(avg_age or 0, WARN["avg_bar_age_s"], low_is_bad=False) if avg_age else "")
    row("Stale bars",             fmt(bar_q["stale_count"]),
        flag=_flag(bar_q["stale_count"], WARN["stale_bar_count"] - 1, low_is_bad=False)
             if bar_q["stale_count"] > 0 else "")
    if log_health is not None:
        row("Malformed log lines", fmt(log_health.get("malformed_lines", 0)),
            flag="WARN" if log_health.get("malformed_lines", 0) > 0 else "")
        row("Duplicate rows removed", fmt(log_health.get("duplicate_rows", 0)),
            flag="WARN" if log_health.get("duplicate_rows", 0) > 0 else "")

    # ── Signal behavior ───────────────────────────────────────────
    section("SIGNAL BEHAVIOR")
    total_eval = sig["total_evaluated"]
    buy_count  = sig["buy_signals"]
    live_per_sym = None
    row("Total bars evaluated",   fmt(total_eval))
    if signal_comparable:
        bars_per_symbol_seen, coverage_pct, live_per_sym = _session_signal_rate(sig)
        if live_per_sym is None:
            live_per_sym = buy_count / n_sym if n_sym > 0 else 0
        bt_per_sym   = BACKTEST["signal_rate_per_symbol_per_day"]
        sig_ratio    = live_per_sym / bt_per_sym if bt_per_sym > 0 else 0

        row("BUY signals fired",      fmt(buy_count),
            f"({live_per_sym:.1f}/sym/day equiv  bt={bt_per_sym:.1f}  ratio={sig_ratio:.0%})",
            flag=_flag(sig_ratio, WARN["signal_rate_pct_of_bt"], low_is_bad=True)
                 if buy_count > 0 else "")
        row("SELL signals fired",     fmt(sig["sell_signals"]))
        if coverage_pct is not None:
            row("Session coverage",    f"{coverage_pct:.0%}",
                f"(~{bars_per_symbol_seen:.1f}/{BACKTEST['bars_per_day']} bars per symbol observed)")
    else:
        row("BUY signals fired",      fmt(buy_count))
        row("SELL signals fired",     fmt(sig["sell_signals"]))
    rej_rate = sig["rejection_rate"]
    row("Rejection rate",         fmt(rej_rate, ".0%") if rej_rate is not None else "n/a",
        f"(backtest ~{bt_rejection:.0%})" if bt_rejection is not None else "")

    if sig["rejections"]:
        print()
        print("  Rejection breakdown:")
        for reason, count in sorted(sig["rejections"].items(), key=lambda x: -x[1]):
            pct = count / total_eval * 100 if total_eval > 0 else 0
            print(f"    {reason:<32} {count:>4}  ({pct:.1f}%)")

    if sig["by_symbol"]:
        print()
        print("  BUY signals by symbol:")
        for sym, count in sorted(sig["by_symbol"].items(), key=lambda x: -x[1]):
            print(f"    {sym:<8} {count:>3}")

    # ── Execution ─────────────────────────────────────────────────
    section("EXECUTION")
    row("Orders submitted (buy)",  fmt(ex["submitted_buys"]))
    row("Orders submitted (sell)", fmt(ex["submitted_sells"]))
    row("Partial fills",           fmt(ex["partial_fills"]),
        flag="WARN" if ex["partial_fills"] > 0 else "")
    avg_slip = ex["avg_slippage_bps"]
    worst_slip = ex["worst_slippage_bps"]
    row("Avg slippage (bps)",      fmt(avg_slip, ".1f"),
        f"(backtest ~{bt_slippage_bps:.1f} bps)" if bt_slippage_bps is not None else "",
        flag=_flag(avg_slip or 0, WARN["avg_slippage_bps"], low_is_bad=False) if avg_slip else "")
    row("Worst slippage (bps)",    fmt(worst_slip, ".1f"),
        flag=_flag(worst_slip or 0, WARN["worst_slippage_bps"], low_is_bad=False) if worst_slip else "")
    row("Avg fill latency (ms)",   fmt(ex["avg_latency_ms"], ".0f"))

    # ── Risk ──────────────────────────────────────────────────────
    section("RISK CONTROLS")
    ks = "TRIGGERED ⚠" if risk["kill_switch_triggered"] else "not triggered"
    row("Kill switch",             ks,
        flag="WARN" if risk["kill_switch_triggered"] else "")
    if risk["buy_blocks_by_reason"]:
        print()
        print("  BUY blocks by reason:")
        for reason, count in sorted(risk["buy_blocks_by_reason"].items(), key=lambda x: -x[1]):
            print(f"    {reason:<38} {count:>3}")
    if risk["sell_blocks_by_reason"]:
        print()
        print("  SELL blocks by reason:")
        for reason, count in sorted(risk["sell_blocks_by_reason"].items(), key=lambda x: -x[1]):
            print(f"    {reason:<38} {count:>3}")

    # ── Trade outcomes ────────────────────────────────────────────
    section("TRADE OUTCOMES")
    n_closed = pos["closed"]
    row("Positions opened",        fmt(pos["opened"]))
    row("Positions closed",        fmt(n_closed))

    if n_closed >= WARN["min_trades_for_win_rate"] and pos["win_rate"] is not None:
        wr = pos["win_rate"]
        row("Win rate",            fmt(wr, ".0%"),
            f"(backtest {bt_win_rate:.0%})" if bt_win_rate is not None else "",
            flag=_flag(wr, WARN["win_rate_low"], low_is_bad=True))
    elif n_closed > 0:
        row("Win rate",            f"{pos['win_rate']:.0%}" if pos["win_rate"] is not None else "n/a",
            f"(only {n_closed} trade(s) — not meaningful yet)")
    else:
        row("Win rate",            "n/a", "(no closed positions today)")

    avg_hold = pos["avg_hold_bars"]
    row("Avg hold (bars)",         fmt(avg_hold, ".1f"),
        f"(backtest {bt_hold_bars:.1f})" if bt_hold_bars is not None else "")
    row("Avg PnL per trade ($)",   fmt(pos["avg_pnl_usd"], ".2f"))
    row("Total PnL today ($)",     fmt(pos["total_pnl_usd"], ".2f"))

    if pos["exit_reasons"]:
        print()
        print("  Exit reason breakdown:")
        total_exits = sum(pos["exit_reasons"].values())
        for reason, count in sorted(pos["exit_reasons"].items(), key=lambda x: -x[1]):
            pct = count / total_exits * 100 if total_exits > 0 else 0
            flag_str = ""
            if reason == "eod_flatten" and pct > WARN["eod_exit_rate"] * 100:
                flag_str = "  [WARN]"
            print(f"    {reason:<32} {count:>3}  ({pct:.0f}%){flag_str}")

    # ── Backtest comparison ───────────────────────────────────────
    section("vs BACKTEST COMPARISON")
    if comparable:
        _print_backtest_comparison(
            sig, ex, pos, live_per_sym if buy_count > 0 else None, comparison
        )
    else:
        print("  Comparison unavailable: baseline is not valid for comparison.")
        for item in reason_summaries:
            print(f"    - {item}")

    # ── Concerns ──────────────────────────────────────────────────
    section("CONCERNS")
    if not concerns:
        print("  No concerns. Live behavior is within expected parameters.")
    else:
        for i, c in enumerate(concerns, 1):
            print(f"  {i}. {c}")

    print(f"\n{'═' * 60}\n")


_COMPARISON_LABELS: dict[str, str] = {
    "signal_rate_per_symbol_per_day": "BUY signals/sym/day",
    "win_rate":                       "Win rate",
    "avg_hold_bars":                  "Avg hold (bars)",
    "rejection_rate":                 "Rejection rate",
    "avg_slippage_bps":               "Avg slippage (bps)",
}


def _print_backtest_comparison(
    sig: dict,
    ex: dict,
    pos: dict,
    live_per_sym: float | None,
    comparison: dict[str, Any] | None = None,
) -> None:
    """Live-vs-backtest table.  Callers must check baseline_comparison_status().

    Only metrics with an artifact-backed expectation get a row; the rest are
    listed as not research-backed instead of being compared to placeholders.
    """
    comparison = comparison if comparison is not None else baseline_comparison_status()
    signal_comparable = _signal_rate_comparable(comparison)
    print(f"  {'Metric':<32} {'Live':>10}  {'Backtest':>10}  {'Delta':>10}")
    print(f"  {'─'*32} {'─'*10}  {'─'*10}  {'─'*10}")

    def compare_row(label: str, live: Any, bt: Any, fmt_s: str = ".2f", invert: bool = False) -> None:
        if live is None:
            live_str, delta_str = "n/a", "n/a"
        else:
            live_str = format(live, fmt_s)
            delta = live - bt
            sign = "+" if delta >= 0 else ""
            delta_str = f"{sign}{delta:{fmt_s}}"
            if invert:
                delta_str = f"{'−' if delta >= 0 else '+'}{abs(delta):{fmt_s}}"
        bt_str = format(bt, fmt_s)
        print(f"  {label:<32} {live_str:>10}  {bt_str:>10}  {delta_str:>10}")

    rows: list[tuple[str, Any, str, bool, bool]] = [
        # (metric key, live value, format, invert, has expectation)
        ("signal_rate_per_symbol_per_day", live_per_sym, ".2f", False, signal_comparable),
        ("win_rate", pos["win_rate"], ".0%", False, True),
        ("avg_hold_bars", pos["avg_hold_bars"], ".2f", False, True),
        ("rejection_rate", sig["rejection_rate"], ".0%", False, True),
        ("avg_slippage_bps", ex["avg_slippage_bps"], ".2f", True, True),
    ]
    not_backed: list[str] = []
    for key, live, fmt_s, invert, has_expectation in rows:
        expected = _expectation(comparison, key) if has_expectation else None
        if expected is None:
            not_backed.append(_COMPARISON_LABELS[key])
            continue
        compare_row(_COMPARISON_LABELS[key], live, expected, fmt_s, invert=invert)
    if not_backed:
        print()
        print("  Not research-backed (no expectation): " + ", ".join(not_backed))


# ---------------------------------------------------------------------------
# JSON output (machine-readable)
# ---------------------------------------------------------------------------

def build_json_output(
    report_date: str,
    bar_q: dict,
    sig: dict,
    ex: dict,
    risk: dict,
    pos: dict,
    concerns: list[str],
    log_health: dict | None = None,
) -> dict:
    comparison = baseline_comparison_status()
    # Fail closed: never publish DEFAULT_BACKTEST fallbacks as expectations.
    # A metric is published only when the baseline is valid AND the metric was
    # loaded from a validated artifact (see comparison["sources"]).
    expected: dict[str, Any] = {
        key: _expectation(comparison, key) for key in BACKTEST
    }
    return {
        "date":          report_date,
        "backtest_source": BACKTEST_SOURCE,
        "comparison":    comparison,
        "bar_quality":   bar_q,
        "signals":       sig,
        "execution":     ex,
        "risk":          risk,
        "positions":     pos,
        "concerns":      concerns,
        "log_health":    log_health or {},
        "backtest":      expected,
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Daily bot diagnostic report")
    parser.add_argument(
        "--date", default=date.today().strftime("%Y-%m-%d"),
        help="Trading date to analyse (YYYY-MM-DD). Defaults to today."
    )
    parser.add_argument(
        "--log-root", default="logs",
        help="Root directory containing per-date log subdirectories."
    )
    parser.add_argument(
        "--json", action="store_true",
        help="Output machine-readable JSON instead of the formatted report."
    )
    parser.add_argument(
        "--require-approved-baseline", action="store_true",
        help="Exit with an error if the report baseline is unapproved or mismatched."
    )
    args = parser.parse_args()

    log_dir = Path(args.log_root) / args.date
    if not log_dir.exists():
        print(f"No log directory found for {args.date}: {log_dir}", file=sys.stderr)
        print("Has the bot run today?", file=sys.stderr)
        sys.exit(1)

    logs = load_logs(log_dir)

    bar_q      = compute_bar_quality(logs["bars"])
    sig        = compute_signal_behavior(logs["signals"])
    ex         = compute_execution(logs["execution"])
    risk_      = compute_risk(logs["risk"])
    pos        = compute_positions(logs["positions"])
    log_health = compute_log_health(logs)
    concerns   = detect_concerns(bar_q, sig, ex, pos, log_health)
    if args.require_approved_baseline and BACKTEST_SOURCE.get("valid_for_comparison") is not True:
        for item in BACKTEST_SOURCE.get("validation_errors", []):
            print(f"Baseline validation error: {item}", file=sys.stderr)
        sys.exit(2)

    if args.json:
        import json as _json
        output = build_json_output(args.date, bar_q, sig, ex, risk_, pos, concerns, log_health)
        print(_json.dumps(output, indent=2, default=str))
    else:
        print_report(args.date, bar_q, sig, ex, risk_, pos, concerns, log_health)


if __name__ == "__main__":
    main()

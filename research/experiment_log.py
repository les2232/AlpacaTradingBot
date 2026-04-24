from __future__ import annotations

import hashlib
import json
import math
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_LOG_PATH = PROJECT_ROOT / "results" / "experiment_log.jsonl"

_HIGHER_IS_BETTER = {
    "total_return_pct",
    "profit_factor",
    "sharpe",
    "win_rate",
    "trade_count",
    "expectancy",
    "realized_pnl",
}
_LOWER_IS_BETTER = {"max_drawdown_pct"}


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _coerce_jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, datetime):
        if value.tzinfo is None:
            value = value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    if isinstance(value, Mapping):
        return {str(key): _coerce_jsonable(val) for key, val in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_coerce_jsonable(item) for item in value]
    if isinstance(value, bool) or value is None:
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            return None
        return float(value)
    return value


def _clean_mapping(payload: Mapping[str, Any] | None) -> dict[str, Any]:
    if not payload:
        return {}
    cleaned = _coerce_jsonable(dict(payload))
    return {key: value for key, value in cleaned.items() if value not in (None, {}, [], "")}


def _try_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    return number


def _git_value(args: Sequence[str]) -> str | None:
    try:
        result = subprocess.run(
            args,
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            check=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None
    value = result.stdout.strip()
    return value or None


def collect_git_context() -> dict[str, Any]:
    commit = _git_value(["git", "rev-parse", "HEAD"])
    branch = _git_value(["git", "rev-parse", "--abbrev-ref", "HEAD"])
    dirty = _git_value(["git", "status", "--porcelain"])
    return {
        "commit": commit,
        "branch": branch,
        "is_dirty": bool(dirty) if dirty is not None else None,
    }


def load_dataset_context(dataset_path: str | Path | None) -> dict[str, Any]:
    if not dataset_path:
        return {}
    path = Path(dataset_path)
    context: dict[str, Any] = {
        "dataset_path": str(path),
        "dataset_name": path.name,
    }
    manifest_path = path / "manifest.json"
    if not manifest_path.exists():
        return context
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return context

    symbols = manifest.get("symbols")
    if isinstance(symbols, list):
        context["symbols"] = [str(symbol).strip().upper() for symbol in symbols if str(symbol).strip()]
    timeframe = manifest.get("timeframe")
    if timeframe:
        context["dataset_timeframe"] = str(timeframe)
    feed = manifest.get("feed")
    if feed:
        context["dataset_feed"] = str(feed)
    start = manifest.get("start")
    end = manifest.get("end")
    if start:
        context["dataset_start"] = str(start)
    if end:
        context["dataset_end"] = str(end)
    return context


def _parameter_hash(params: Mapping[str, Any]) -> str:
    canonical = json.dumps(_clean_mapping(params), sort_keys=True, separators=(",", ":"))
    return hashlib.sha1(canonical.encode("utf-8")).hexdigest()[:12]


def build_change_fingerprint(
    *,
    run_type: str,
    script_name: str,
    strategy_name: str | None,
    dataset_name: str | None,
    params: Mapping[str, Any],
    git_commit: str | None,
) -> dict[str, Any]:
    parameter_hash = _parameter_hash(params)
    base = {
        "run_type": run_type,
        "script_name": script_name,
        "strategy_name": strategy_name,
        "dataset_name": dataset_name,
        "parameter_hash": parameter_hash,
        "git_commit": git_commit,
    }
    canonical = json.dumps(base, sort_keys=True, separators=(",", ":"))
    return {
        **base,
        "fingerprint": hashlib.sha1(canonical.encode("utf-8")).hexdigest()[:16],
    }


def _as_text_list(values: Sequence[Any] | None) -> list[str]:
    if not values:
        return []
    normalized: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = str(value).strip().upper()
        if not text or text in seen:
            continue
        normalized.append(text)
        seen.add(text)
    return normalized


def _select_metrics(metrics: Mapping[str, Any] | None) -> dict[str, float]:
    raw = _clean_mapping(metrics)
    aliases = {
        "total_return_pct": ("total_return_pct", "total_return"),
        "profit_factor": ("profit_factor",),
        "sharpe": ("sharpe", "sharpe_ratio"),
        "win_rate": ("win_rate", "realized_win_rate"),
        "max_drawdown_pct": ("max_drawdown_pct", "max_drawdown"),
        "trade_count": ("trade_count", "total_trades", "closed_trade_count", "closed_trades", "signal_count"),
        "expectancy": ("expectancy", "realized_expectancy", "avg_forward_return_pct", "best_raw_net_expectancy_pct"),
        "realized_pnl": ("realized_pnl",),
    }
    selected: dict[str, float] = {}
    for canonical, names in aliases.items():
        for name in names:
            value = _try_float(raw.get(name))
            if value is not None:
                selected[canonical] = value
                break
    for key, value in raw.items():
        if key in selected:
            continue
        number = _try_float(value)
        if number is not None:
            selected[key] = number
    return selected


def load_experiment_log(log_path: str | Path = DEFAULT_LOG_PATH) -> list[dict[str, Any]]:
    path = Path(log_path)
    if not path.exists():
        return []
    entries: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            entries.append(payload)
    return entries


def _find_prior_comparable(entry: Mapping[str, Any], previous_entries: Sequence[Mapping[str, Any]]) -> dict[str, Any] | None:
    strategy_name = entry.get("strategy_name")
    script_name = entry.get("script_name")
    run_type = entry.get("run_type")
    dataset_name = entry.get("dataset_name")

    exact: list[Mapping[str, Any]] = []
    loose: list[Mapping[str, Any]] = []
    for prior in previous_entries:
        if prior.get("script_name") != script_name or prior.get("run_type") != run_type:
            continue
        if strategy_name and prior.get("strategy_name") != strategy_name:
            continue
        if dataset_name and prior.get("dataset_name") == dataset_name:
            exact.append(prior)
        else:
            loose.append(prior)
    candidates = exact or loose
    if not candidates:
        return None
    return dict(candidates[-1])


def summarize_against_prior(
    entry: Mapping[str, Any],
    prior_entry: Mapping[str, Any] | None,
) -> dict[str, Any]:
    if prior_entry is None:
        return {
            "label": "insufficient evidence",
            "reason": "No comparable prior run was found for this script and strategy.",
            "compared_run_id": None,
            "deltas": {},
        }

    current_metrics = {key: _try_float(value) for key, value in dict(entry.get("metrics") or {}).items()}
    prior_metrics = {key: _try_float(value) for key, value in dict(prior_entry.get("metrics") or {}).items()}
    deltas: dict[str, float] = {}
    score = 0
    comparable_metrics = 0

    current_trade_count = current_metrics.get("trade_count")
    prior_trade_count = prior_metrics.get("trade_count")
    if current_trade_count is not None and prior_trade_count is not None:
        deltas["trade_count"] = current_trade_count - prior_trade_count
        comparable_metrics += 1
        if prior_trade_count > 0 and current_trade_count <= prior_trade_count * 0.5:
            return {
                "label": "trade count collapsed",
                "reason": (
                    f"Trade count fell from {prior_trade_count:.0f} to {current_trade_count:.0f} "
                    f"against the most recent comparable run."
                ),
                "compared_run_id": prior_entry.get("run_id"),
                "deltas": deltas,
            }

    for metric_name in sorted(_HIGHER_IS_BETTER | _LOWER_IS_BETTER):
        current_value = current_metrics.get(metric_name)
        prior_value = prior_metrics.get(metric_name)
        if current_value is None or prior_value is None:
            continue
        delta = current_value - prior_value
        deltas[metric_name] = delta
        comparable_metrics += 1
        if abs(delta) < 1e-12:
            continue
        if metric_name in _HIGHER_IS_BETTER:
            score += 1 if delta > 0 else -1
        else:
            score += 1 if delta < 0 else -1

    if comparable_metrics < 2:
        return {
            "label": "insufficient evidence",
            "reason": "A prior run exists, but there are not enough overlapping metrics to compare safely.",
            "compared_run_id": prior_entry.get("run_id"),
            "deltas": deltas,
        }
    if score >= 2:
        return {
            "label": "improved vs prior",
            "reason": "Core overlapping metrics improved against the most recent comparable run.",
            "compared_run_id": prior_entry.get("run_id"),
            "deltas": deltas,
        }
    if score <= -2:
        return {
            "label": "worse than prior",
            "reason": "Core overlapping metrics degraded against the most recent comparable run.",
            "compared_run_id": prior_entry.get("run_id"),
            "deltas": deltas,
        }
    return {
        "label": "insufficient evidence",
        "reason": "Metric changes were mixed, so there is no clear directional conclusion.",
        "compared_run_id": prior_entry.get("run_id"),
        "deltas": deltas,
    }


def create_experiment_log_entry(
    *,
    run_type: str,
    script_path: str | Path,
    strategy_name: str | None = None,
    dataset_path: str | Path | None = None,
    symbols: Sequence[Any] | None = None,
    params: Mapping[str, Any] | None = None,
    metrics: Mapping[str, Any] | None = None,
    output_path: str | Path | None = None,
    summary_path: str | Path | None = None,
    entrypoint: str | Path | None = None,
    log_path: str | Path = DEFAULT_LOG_PATH,
    extra_fields: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    dataset_context = load_dataset_context(dataset_path)
    git = collect_git_context()
    key_parameters = _clean_mapping(params)
    selected_metrics = _select_metrics(metrics)
    strategy_text = str(strategy_name).strip() if strategy_name else None
    script = Path(script_path)
    entrypoint_path = Path(entrypoint) if entrypoint else None
    output_dir = Path(output_path) if output_path else None
    summary_file = Path(summary_path) if summary_path else None
    symbol_list = _as_text_list(symbols) or dataset_context.get("symbols", [])

    fingerprint = build_change_fingerprint(
        run_type=run_type,
        script_name=script.name,
        strategy_name=strategy_text,
        dataset_name=dataset_context.get("dataset_name"),
        params=key_parameters,
        git_commit=git.get("commit"),
    )
    entry = {
        "run_id": f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}_{fingerprint['fingerprint'][:8]}",
        "timestamp": utc_now_iso(),
        "run_type": run_type,
        "script_name": script.name,
        "script_path": str(script),
        "entrypoint": str(entrypoint_path) if entrypoint_path else None,
        "strategy_name": strategy_text,
        "symbols": symbol_list,
        "dataset_path": dataset_context.get("dataset_path"),
        "dataset_name": dataset_context.get("dataset_name"),
        "dataset_timeframe": dataset_context.get("dataset_timeframe"),
        "dataset_feed": dataset_context.get("dataset_feed"),
        "dataset_start": dataset_context.get("dataset_start"),
        "dataset_end": dataset_context.get("dataset_end"),
        "key_parameters": key_parameters,
        "metrics": selected_metrics,
        "output_path": str(output_dir) if output_dir else None,
        "summary_path": str(summary_file) if summary_file else None,
        "git": git,
        "change_fingerprint": fingerprint,
    }
    entry.update(_clean_mapping(extra_fields))

    prior_entry = _find_prior_comparable(entry, load_experiment_log(log_path))
    entry["auto_summary"] = summarize_against_prior(entry, prior_entry)
    return _clean_mapping(entry)


def append_experiment_log(entry: Mapping[str, Any], log_path: str | Path = DEFAULT_LOG_PATH) -> Path:
    path = Path(log_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(_coerce_jsonable(dict(entry)), sort_keys=True) + "\n")
    return path


def log_experiment_run(
    *,
    run_type: str,
    script_path: str | Path,
    strategy_name: str | None = None,
    dataset_path: str | Path | None = None,
    symbols: Sequence[Any] | None = None,
    params: Mapping[str, Any] | None = None,
    metrics: Mapping[str, Any] | None = None,
    output_path: str | Path | None = None,
    summary_path: str | Path | None = None,
    entrypoint: str | Path | None = None,
    extra_fields: Mapping[str, Any] | None = None,
    log_path: str | Path = DEFAULT_LOG_PATH,
) -> dict[str, Any]:
    entry = create_experiment_log_entry(
        run_type=run_type,
        script_path=script_path,
        strategy_name=strategy_name,
        dataset_path=dataset_path,
        symbols=symbols,
        params=params,
        metrics=metrics,
        output_path=output_path,
        summary_path=summary_path,
        entrypoint=entrypoint,
        extra_fields=extra_fields,
        log_path=log_path,
    )
    append_experiment_log(entry, log_path=log_path)
    return entry


def experiments_to_dataframe(entries: Sequence[Mapping[str, Any]]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for entry in entries:
        metrics = dict(entry.get("metrics") or {})
        summary = dict(entry.get("auto_summary") or {})
        fingerprint = dict(entry.get("change_fingerprint") or {})
        params = dict(entry.get("key_parameters") or {})
        rows.append(
            {
                "run_id": entry.get("run_id"),
                "timestamp": entry.get("timestamp"),
                "run_type": entry.get("run_type"),
                "script_name": entry.get("script_name"),
                "strategy_name": entry.get("strategy_name"),
                "symbols": ", ".join(entry.get("symbols") or []),
                "dataset_name": entry.get("dataset_name"),
                "dataset_path": entry.get("dataset_path"),
                "output_path": entry.get("output_path"),
                "branch": ((entry.get("git") or {}).get("branch")),
                "git_commit": ((entry.get("git") or {}).get("commit")),
                "summary_label": summary.get("label"),
                "summary_reason": summary.get("reason"),
                "fingerprint": fingerprint.get("fingerprint"),
                "parameter_hash": fingerprint.get("parameter_hash"),
                "parameter_preview": json.dumps(params, sort_keys=True)[:240] if params else "",
                **metrics,
            }
        )
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    return df.sort_values("timestamp", ascending=False).reset_index(drop=True)


def load_experiment_log_frame(log_path: str | Path = DEFAULT_LOG_PATH) -> pd.DataFrame:
    return experiments_to_dataframe(load_experiment_log(log_path))

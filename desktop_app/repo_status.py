from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any


DEFAULT_RUNTIME_CONFIG_PATH = Path("config") / "live_config.json"


@dataclass(frozen=True)
class RuntimeStatus:
    strategy_mode: str
    symbols_text: str
    timeframe_text: str
    account_mode: str
    execution_mode: str
    runtime_source: str
    promoted_config_source: str


def load_runtime_status(project_root: Path) -> RuntimeStatus:
    env_path = project_root / ".env"
    env_values = _read_dotenv(env_path)

    runtime_config_setting = _get_env_setting("BOT_RUNTIME_CONFIG_PATH", env_values)
    runtime_path = project_root / runtime_config_setting if runtime_config_setting else project_root / DEFAULT_RUNTIME_CONFIG_PATH
    payload, runtime = _read_runtime_payload(runtime_path)

    base_symbols_raw = _get_env_setting("BOT_SYMBOLS", env_values) or "AAPL,MSFT,NVDA"
    base_symbols = [symbol.strip().upper() for symbol in base_symbols_raw.split(",") if symbol.strip()]
    base_strategy_mode = _get_env_setting("STRATEGY_MODE", env_values) or "hybrid"
    base_timeframe = _get_env_setting("BAR_TIMEFRAME_MINUTES", env_values) or "15"

    effective_symbols = _runtime_symbols(runtime) or base_symbols
    effective_strategy_mode = str(runtime.get("strategy_mode", base_strategy_mode)) if runtime else base_strategy_mode
    effective_timeframe = str(runtime.get("bar_timeframe_minutes", base_timeframe)) if runtime else base_timeframe

    account_mode = "paper" if (_get_env_setting("ALPACA_PAPER", env_values) or "true").strip().lower() != "false" else "live"
    execution_mode = "live orders" if (_get_env_setting("EXECUTE_ORDERS", env_values) or "true").strip().lower() != "false" else "dry run"

    runtime_source = _describe_runtime_source(runtime_path=runtime_path, runtime=runtime)
    promoted_source = _describe_promoted_source(payload)

    return RuntimeStatus(
        strategy_mode=effective_strategy_mode,
        symbols_text=_format_symbols(effective_symbols),
        timeframe_text=f"{effective_timeframe}m",
        account_mode=account_mode,
        execution_mode=execution_mode,
        runtime_source=runtime_source,
        promoted_config_source=promoted_source,
    )


def _read_dotenv(env_path: Path) -> dict[str, str]:
    if not env_path.exists():
        return {}

    values: dict[str, str] = {}
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = _strip_inline_comment(value.strip())
        if key:
            values[key] = value
    return values


def _strip_inline_comment(value: str) -> str:
    for marker in (" #", "\t#"):
        if marker in value:
            return value.split(marker, 1)[0].rstrip()
    return value


def _get_env_setting(key: str, env_values: dict[str, str]) -> str | None:
    process_value = os.getenv(key)
    if process_value and process_value.strip():
        return process_value.strip()
    file_value = env_values.get(key)
    if file_value and file_value.strip():
        return file_value.strip()
    return None


def _read_runtime_payload(runtime_path: Path) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    if not runtime_path.exists():
        return None, None

    try:
        payload = json.loads(runtime_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None, None

    if not isinstance(payload, dict):
        return None, None

    runtime = payload.get("runtime", payload)
    if not isinstance(runtime, dict):
        return payload, None
    return payload, runtime


def _runtime_symbols(runtime: dict[str, Any] | None) -> list[str] | None:
    if not runtime:
        return None
    raw_symbols = runtime.get("symbols")
    if not isinstance(raw_symbols, list):
        return None
    return [str(symbol).strip().upper() for symbol in raw_symbols if str(symbol).strip()]


def _format_symbols(symbols: list[str]) -> str:
    if not symbols:
        return "n/a"
    if len(symbols) <= 8:
        return ", ".join(symbols)
    return ", ".join(symbols[:8]) + f", ... ({len(symbols)} total)"


def _describe_runtime_source(runtime_path: Path, runtime: dict[str, Any] | None) -> str:
    if runtime is None:
        return "Environment/.env only"
    return f"Environment/.env with runtime overrides from {runtime_path}"


def _describe_promoted_source(payload: dict[str, Any] | None) -> str:
    if not payload:
        return "n/a"
    source = payload.get("source")
    if not isinstance(source, dict):
        return "n/a"

    dataset = source.get("dataset")
    note = source.get("note")
    symbol_source = source.get("dataset_symbol_source")
    if dataset:
        return str(dataset)
    if symbol_source:
        return str(symbol_source)
    if note:
        return str(note)
    return "n/a"

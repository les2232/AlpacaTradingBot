"""Server-side configuration for the Ask TradeOS analyst.

Credentials are resolved from the process environment first, then from the
repository ``.env`` file.  Only the named keys below are read from ``.env``;
nothing is exported into ``os.environ`` (the same rule the dashboard already
applies to ``BOT_DB_PATH``).  The API key is never logged, rendered, or placed
into a model request body.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping

import dotenv

from tradeos.runtime_paths import TRADEOS_REPOSITORY_ROOT


API_KEY_ENV = "OPENAI_API_KEY"
MODEL_ENV = "TRADEOS_ASK_MODEL"
TIMEOUT_ENV = "TRADEOS_ASK_TIMEOUT_SECONDS"
DEFAULT_MODEL = "gpt-6.1-sol"
DEFAULT_TIMEOUT_SECONDS = 60.0
_DOTENV_KEYS = (API_KEY_ENV, MODEL_ENV, TIMEOUT_ENV)


@dataclass(frozen=True)
class AskSettings:
    model: str
    timeout_seconds: float
    api_key: str | None = field(default=None, repr=False)
    api_key_source: str | None = None

    @property
    def model_configured(self) -> bool:
        return bool(self.api_key)

    def describe(self) -> str:
        """Operator-facing status line. Never includes the key itself."""

        if not self.api_key:
            return (
                f"Model explanations are off: {API_KEY_ENV} is not set. "
                "Calculated diagnostics are still available."
            )
        return f"Model: {self.model} (key from {self.api_key_source})."


def _clean(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    text = value.strip()
    return text or None


def load_ask_settings(
    *,
    environment: Mapping[str, str] | None = None,
    dotenv_path: Path | None = None,
) -> AskSettings:
    source = os.environ if environment is None else environment
    try:
        file_values = dotenv.dotenv_values(
            dotenv_path or (TRADEOS_REPOSITORY_ROOT / ".env"), verbose=False
        )
    except (OSError, UnicodeError):
        file_values = {}
    file_values = {key: file_values.get(key) for key in _DOTENV_KEYS}

    def pick(name: str) -> tuple[str | None, str | None]:
        value = _clean(source.get(name))
        if value is not None:
            return value, "environment"
        value = _clean(file_values.get(name))
        if value is not None:
            return value, ".env"
        return None, None

    api_key, api_key_source = pick(API_KEY_ENV)
    model, _ = pick(MODEL_ENV)
    raw_timeout, _ = pick(TIMEOUT_ENV)
    timeout = DEFAULT_TIMEOUT_SECONDS
    if raw_timeout is not None:
        try:
            parsed = float(raw_timeout)
        except ValueError:
            parsed = DEFAULT_TIMEOUT_SECONDS
        if 1.0 <= parsed <= 600.0:
            timeout = parsed
    return AskSettings(
        model=model or DEFAULT_MODEL,
        timeout_seconds=timeout,
        api_key=api_key,
        api_key_source=api_key_source,
    )

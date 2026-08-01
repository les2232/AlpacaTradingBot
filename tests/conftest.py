from __future__ import annotations

import sys

import dotenv
import pytest


@pytest.fixture(autouse=True)
def prevent_local_dotenv_access(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep offline tests independent from operator credentials and settings."""

    monkeypatch.delenv("BOT_DB_PATH", raising=False)

    def disabled_load_dotenv(*args: object, **kwargs: object) -> bool:
        return False

    monkeypatch.setattr(dotenv, "load_dotenv", disabled_load_dotenv)
    monkeypatch.setattr(dotenv, "dotenv_values", lambda *args, **kwargs: {})
    for module_name in (
        "alpaca_trading_bot.cli",
        "dataset_snapshotter",
        "trading_bot",
    ):
        module = sys.modules.get(module_name)
        if module is not None and hasattr(module, "load_dotenv"):
            monkeypatch.setattr(module, "load_dotenv", disabled_load_dotenv)

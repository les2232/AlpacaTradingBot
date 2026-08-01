"""Stable runtime path authority for TradeOS entrypoints."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Mapping

import dotenv


DATABASE_PATH_ENV = "BOT_DB_PATH"
DEFAULT_DATABASE_FILENAME = "bot_history.db"
LOG_ROOT_ENV = "BOT_LOG_ROOT"
DEFAULT_LOG_ROOT_DIRECTORY = "logs"
TRADEOS_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


class DatabasePathError(ValueError):
    """Raised when a configured database path is not a safe authority."""


def _resolve_path_from_root(
    configured_path: str | os.PathLike[str],
    *,
    repository_root: Path,
    configuration_name: str,
) -> Path:
    """Resolve one repository-anchored path without creating anything."""

    raw_path = os.fspath(configured_path)
    path_text = raw_path.strip()
    if not path_text:
        raise ValueError(
            f"{configuration_name} cannot be blank; unset it to use the repository default."
        )

    canonical_root = repository_root.resolve(strict=False)
    supplied_path = Path(path_text)
    if supplied_path.is_absolute():
        return supplied_path.resolve(strict=False)

    resolved_path = (canonical_root / supplied_path).resolve(strict=False)
    try:
        resolved_path.relative_to(canonical_root)
    except ValueError as exc:
        raise ValueError(
            f"Relative {configuration_name} paths must remain inside the TradeOS repository; "
            "use an explicitly absolute path for an external location."
        ) from exc
    return resolved_path


def _resolve_database_path_from_root(
    configured_path: str | os.PathLike[str],
    *,
    repository_root: Path,
) -> Path:
    """Resolve one configured database path without creating or opening it."""

    try:
        return _resolve_path_from_root(
            configured_path,
            repository_root=repository_root,
            configuration_name=DATABASE_PATH_ENV,
        )
    except ValueError as exc:
        raise DatabasePathError(str(exc)) from exc


def resolve_database_path(
    configured_path: str | os.PathLike[str] | None = None,
    *,
    environment: Mapping[str, str] | None = None,
) -> Path:
    """Return the canonical absolute SQLite authority for this checkout.

    An explicit argument wins over ``BOT_DB_PATH``. When neither is supplied,
    the existing repository-root ``bot_history.db`` location remains the
    default. Relative values are repository-rooted and may not escape it.
    """

    if configured_path is None:
        source = os.environ if environment is None else environment
        configured_path = source.get(DATABASE_PATH_ENV, DEFAULT_DATABASE_FILENAME)
    return _resolve_database_path_from_root(
        configured_path,
        repository_root=TRADEOS_REPOSITORY_ROOT,
    )


def resolve_explicit_database_path(
    *,
    environment: Mapping[str, str] | None = None,
) -> Path | None:
    """Resolve the explicit process or repository database setting, if any.

    Only the ``BOT_DB_PATH`` value is retained from the repository dotenv file.
    Other settings and broker credentials are never exported.
    """

    source = os.environ if environment is None else environment
    if DATABASE_PATH_ENV in source:
        return resolve_database_path(source[DATABASE_PATH_ENV])

    configured_values = dotenv.dotenv_values(
        TRADEOS_REPOSITORY_ROOT / ".env",
        verbose=False,
    )
    if DATABASE_PATH_ENV not in configured_values:
        return None
    configured_path = configured_values[DATABASE_PATH_ENV]
    if configured_path is None:
        configured_path = ""
    return resolve_database_path(configured_path)


def resolve_log_root(
    configured_path: str | os.PathLike[str] | None = None,
    *,
    environment: Mapping[str, str] | None = None,
) -> Path:
    """Return the CWD-invariant root used to discover startup metadata."""

    if configured_path is None:
        source = os.environ if environment is None else environment
        configured_path = source.get(LOG_ROOT_ENV, DEFAULT_LOG_ROOT_DIRECTORY)
    return _resolve_path_from_root(
        configured_path,
        repository_root=TRADEOS_REPOSITORY_ROOT,
        configuration_name=LOG_ROOT_ENV,
    )


def require_canonical_database_path(
    configured_path: str | os.PathLike[str],
) -> Path:
    """Validate that storage received an absolute, normalized authority."""

    raw_path = os.fspath(configured_path)
    path_text = raw_path.strip()
    if not path_text:
        raise DatabasePathError("Storage requires a nonblank canonical database path.")

    supplied_path = Path(path_text)
    resolved_path = resolve_database_path(path_text)
    if not supplied_path.is_absolute() or supplied_path != resolved_path:
        raise DatabasePathError(
            "Storage requires an absolute, normalized database path from "
            "resolve_database_path()."
        )
    return resolved_path

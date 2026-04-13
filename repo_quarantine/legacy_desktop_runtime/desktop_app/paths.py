from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class PathResolution:
    path: Path
    latest_child: Path | None = None


def resolve_latest_logs(project_root: Path) -> PathResolution:
    logs_root = project_root / "logs"
    if not logs_root.exists():
        raise FileNotFoundError(f"Logs folder was not found: {logs_root}")

    dated_dirs = [
        path for path in logs_root.iterdir()
        if path.is_dir() and path.name != "launches" and _looks_like_date_folder(path.name)
    ]
    if not dated_dirs:
        raise FileNotFoundError(f"No dated log folders were found under: {logs_root}")

    latest = max(dated_dirs, key=lambda path: path.name)
    return PathResolution(path=latest, latest_child=latest)


def resolve_results(project_root: Path) -> PathResolution:
    results_root = project_root / "results"
    if not results_root.exists():
        raise FileNotFoundError(f"Results folder was not found: {results_root}")

    children = [path for path in results_root.iterdir()]
    latest = max(children, key=lambda path: path.stat().st_mtime, default=None)
    return PathResolution(path=results_root, latest_child=latest)


def open_in_windows(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Path was not found: {path}")

    if os.name == "nt":
        os.startfile(str(path))
        return

    subprocess.Popen(["xdg-open", str(path)])


def resolve_today_log(project_root: Path, filename: str) -> PathResolution:
    from datetime import date
    today = date.today().strftime("%Y-%m-%d")
    log_path = project_root / "logs" / today / filename
    if not log_path.exists():
        raise FileNotFoundError(f"Today's log file was not found: {log_path}")
    return PathResolution(path=log_path)


def _looks_like_date_folder(name: str) -> bool:
    parts = name.split("-")
    if len(parts) != 3:
        return False
    return all(part.isdigit() for part in parts)

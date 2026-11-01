"""Read-only diagnostics for one session, safe to paste into a support chat.

Usage:
    python -m tradeos.ask.inspect <session-id>

Prints where Ask TradeOS found (or did not find) the session's startup
artifact, log records, lifecycle steps, and database runs.  It never prints
account fingerprints, balances, order ids, credentials, or absolute paths.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from tradeos.ask.records import (
    list_log_days,
    load_day,
    load_db_sessions,
    load_startup_artifacts,
)
from tradeos.runtime_paths import resolve_database_path, resolve_explicit_database_path, resolve_log_root

_ARTIFACT_FIELDS = (
    "artifact_schema_name",
    "artifact_schema_version",
    "artifact_type",
    "launch_mode",
    "started_at_utc",
    "execution_enabled",
    "paper",
    "strategy_mode",
    "bar_timeframe_minutes",
    "symbol_count",
    "baseline_valid_for_comparison",
)
_LIFECYCLE_FIELDS = ("event", "stage", "ts", "result", "ready", "status", "reason", "reason_codes", "blocking_reasons")
_MAX_SCAN_BYTES = 50 * 1024 * 1024


def _safe_value(value: Any) -> Any:
    if isinstance(value, str) and len(value) > 160:
        return value[:157] + "..."
    return value


def inspect_session(session_id: str, log_root: Path, db_path: Path | None) -> dict[str, Any]:
    report: dict[str, Any] = {"session_id": session_id, "log_root_exists": log_root.is_dir(), "db_found": db_path is not None}

    artifacts = []
    for artifact in load_startup_artifacts(log_root):
        if artifact.session_id != session_id:
            continue
        payload = artifact.payload
        artifacts.append(
            {
                "file": artifact.path,
                "valid_for_ask_tradeos": artifact.valid,
                "errors": list(artifact.errors),
                "fields": {k: _safe_value(payload.get(k)) for k in _ARTIFACT_FIELDS},
                "symbols_listed": len(payload.get("symbols") or []) if isinstance(payload.get("symbols"), list) else repr(type(payload.get("symbols")).__name__),
                "total_keys": len(payload),
            }
        )
    report["startup_artifacts"] = artifacts

    records: dict[str, dict[str, int]] = {}
    lifecycle: list[dict[str, Any]] = []
    other_files: list[str] = []
    needle = session_id.encode("utf-8")
    for day in list_log_days(log_root):
        day_log = load_day(log_root, day)
        counts = Counter(r.stream for r in day_log.records if r.session_id == session_id)
        if counts:
            records[day] = dict(counts)
        for record in day_log.records:
            if record.session_id == session_id and record.stream == "lifecycle":
                lifecycle.append({k: _safe_value(record.payload.get(k)) for k in _LIFECYCLE_FIELDS if k in record.payload})
        day_dir = log_root / day
        try:
            entries = sorted(day_dir.iterdir())
        except OSError:
            continue
        for path in entries:
            if not path.is_file() or path.suffix not in {".json", ".jsonl", ".log"}:
                continue
            try:
                if path.stat().st_size > _MAX_SCAN_BYTES:
                    continue
                if needle in path.read_bytes():
                    other_files.append(f"{day}/{path.name}")
            except OSError:
                continue
    report["log_records_by_day_and_stream"] = records
    report["lifecycle_steps"] = lifecycle[-25:]
    report["files_mentioning_session_id"] = other_files

    db_runs = [s for s in load_db_sessions(db_path) if s.session_id == session_id]
    report["database_runs"] = [
        {"runs": s.run_count, "environments": list(s.environments), "first": s.first_ts, "last": s.last_ts} for s in db_runs
    ]
    return report


def list_sessions(log_root: Path, db_path: Path | None) -> list[str]:
    """One line per session: start, run type, record counts, config status, id."""

    from tradeos.ask.records import discover_sessions

    artifacts: dict[str, list[bool]] = {}
    for artifact in load_startup_artifacts(log_root):
        if artifact.session_id:
            artifacts.setdefault(artifact.session_id, []).append(artifact.valid)
    lines = []
    for ref in discover_sessions(log_root, db_path):
        if ref.kind != "bound":
            lines.append(f"{ref.days[0]}  unbound log day  logs={ref.log_record_count:>7,}  db_runs=      0  config=n/a      {ref.key}")
            continue
        valid = artifacts.get(ref.session_id or "", [])
        config = "valid" if any(valid) else ("invalid" if valid else "missing")
        lines.append(
            f"{(ref.first_ts or '')[:16]:16}  {ref.run_type:34}  logs={ref.log_record_count:>7,}  "
            f"db_runs={ref.db_run_count:>7,}  config={config:8} {ref.session_id}"
        )
    return lines


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("session_id", nargs="?", help="session id to inspect")
    parser.add_argument("--list", action="store_true", help="list every session with record counts")
    args = parser.parse_args(argv)
    if not args.list and not args.session_id:
        parser.error("give a session id or --list")
    log_root = resolve_log_root()
    try:
        db_path = resolve_explicit_database_path() or resolve_database_path()
    except ValueError:
        db_path = None
    if db_path is not None and not db_path.is_file():
        db_path = None
    if args.list:
        print("start (UTC)       run type                            log records   db runs        startup config / id")
        for line in list_sessions(log_root, db_path):
            print(line)
        return 0
    json.dump(inspect_session(args.session_id.strip(), log_root, db_path), sys.stdout, indent=2, default=str)
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Read-only loading of historical TradeOS records with source provenance.

Every record keeps the file and line it came from so that evidence shown to
the operator (and cited by the model) resolves to a concrete source.  Records
are attributed to a session only through an explicit ``session_id``; time
proximity is never treated as provenance (the same rule dashboard_state uses).
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable
from zoneinfo import ZoneInfo


STREAMS: tuple[str, ...] = ("signals", "execution", "positions", "risk", "bars", "lifecycle")
EXPECTED_STREAMS: tuple[str, ...] = ("signals", "execution", "positions", "risk", "bars")
_ET = ZoneInfo("America/New_York")
_SHA256_HEX = frozenset("0123456789abcdef")


def parse_utc(value: object) -> datetime | None:
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        parsed = datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return None
    return parsed.astimezone(timezone.utc)


def is_sha256(value: object) -> bool:
    return isinstance(value, str) and len(value) == 64 and set(value) <= _SHA256_HEX


def _session_id_of(payload: dict[str, Any]) -> str | None:
    value = payload.get("session_id")
    if isinstance(value, str) and value.strip():
        return value.strip()
    return None


@dataclass(frozen=True)
class LogRecord:
    stream: str
    day: str
    line: int
    payload: dict[str, Any] = field(repr=False)
    source: str

    @property
    def session_id(self) -> str | None:
        return _session_id_of(self.payload)

    @property
    def event(self) -> str:
        return str(self.payload.get("event", ""))

    @property
    def symbol(self) -> str | None:
        value = self.payload.get("symbol")
        if isinstance(value, str) and value.strip():
            return value.strip().upper()
        return None

    @property
    def ts(self) -> datetime | None:
        return parse_utc(self.payload.get("ts"))


@dataclass(frozen=True)
class StreamFile:
    stream: str
    day: str
    path: str
    present: bool
    raw_lines: int = 0
    malformed_lines: int = 0


@dataclass(frozen=True)
class DayLog:
    day: str
    records: tuple[LogRecord, ...]
    files: tuple[StreamFile, ...]


@dataclass(frozen=True)
class StartupArtifact:
    """A launch-time configuration record for one session."""

    path: str
    payload: dict[str, Any] = field(repr=False)
    session_id: str | None
    valid: bool
    errors: tuple[str, ...]


@dataclass(frozen=True)
class DbSessionSummary:
    session_id: str | None
    environments: tuple[str | None, ...]
    run_count: int
    first_ts: str | None
    last_ts: str | None
    run_ids: tuple[int, ...]


_RUN_TYPES: tuple[tuple[str, str], ...] = (
    ("paper-shadow-", "paper-shadow run (no submit)"),
    ("preview-", "preview run (no submit)"),
    ("live-", "trading run (orders enabled)"),
    ("reconcile-", "broker reconciliation, no trading"),
    ("repair-", "local state repair, no trading"),
)


@dataclass(frozen=True)
class SessionRef:
    key: str
    kind: str  # "bound" | "unbound_log_day"
    session_id: str | None
    days: tuple[str, ...]
    first_ts: str | None
    last_ts: str | None
    log_record_count: int
    startup_artifacts: tuple[str, ...]
    db_run_count: int

    @property
    def run_type(self) -> str:
        """Human description derived from the session-id prefix the CLI assigns."""

        sid = self.session_id or ""
        for prefix, text in _RUN_TYPES:
            if sid.startswith(prefix):
                return text
        return "unrecognised run type"

    @property
    def label(self) -> str:
        day_text = ", ".join(self.days) if self.days else "unknown day"
        if self.kind == "bound":
            start = parse_utc(self.first_ts)
            when = start.astimezone(_ET).strftime("%Y-%m-%d %H:%M ET") if start else day_text
            config = "config recorded" if self.startup_artifacts else "no startup config"
            return (
                f"{when} · {self.run_type} · {self.log_record_count:,} log records · {config} · {self.session_id}"
            )
        return f"{day_text} · unbound log day (no session id)"


# ---------------------------------------------------------------------------
# Log files
# ---------------------------------------------------------------------------

_DAY_CACHE: dict[tuple[str, tuple[tuple[str, int, int], ...]], DayLog] = {}


def _is_day_dir(path: Path) -> bool:
    if not path.is_dir():
        return False
    try:
        datetime.strptime(path.name, "%Y-%m-%d")
    except ValueError:
        return False
    return True


def list_log_days(log_root: Path) -> list[str]:
    if not log_root.is_dir():
        return []
    try:
        return sorted(path.name for path in log_root.iterdir() if _is_day_dir(path))
    except OSError:
        return []


def _relative(path: Path, base: Path) -> str:
    try:
        return path.relative_to(base).as_posix()
    except ValueError:
        return path.name


def load_day(log_root: Path, day: str, *, display_base: Path | None = None) -> DayLog:
    day_dir = log_root / day
    base = display_base or log_root.parent
    stats: list[tuple[str, int, int]] = []
    for stream in STREAMS:
        path = day_dir / f"{stream}.jsonl"
        try:
            stat = path.stat()
            stats.append((stream, stat.st_mtime_ns, stat.st_size))
        except OSError:
            stats.append((stream, -1, -1))
    cache_key = (str(day_dir.resolve(strict=False)), tuple(stats))
    cached = _DAY_CACHE.get(cache_key)
    if cached is not None:
        return cached

    records: list[LogRecord] = []
    files: list[StreamFile] = []
    for stream in STREAMS:
        path = day_dir / f"{stream}.jsonl"
        rel = _relative(path, base)
        if not path.is_file():
            files.append(StreamFile(stream, day, rel, present=False))
            continue
        try:
            lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        except OSError:
            files.append(StreamFile(stream, day, rel, present=False))
            continue
        raw = 0
        malformed = 0
        for index, line in enumerate(lines, start=1):
            if not line.strip():
                continue
            raw += 1
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                malformed += 1
                continue
            if not isinstance(payload, dict):
                malformed += 1
                continue
            records.append(LogRecord(stream, day, index, payload, f"{rel}:L{index}"))
        files.append(StreamFile(stream, day, rel, True, raw, malformed))
    day_log = DayLog(day, tuple(records), tuple(files))
    if len(_DAY_CACHE) > 64:
        _DAY_CACHE.clear()
    _DAY_CACHE[cache_key] = day_log
    return day_log


# ---------------------------------------------------------------------------
# Startup artifacts (launch-time configuration identity)
# ---------------------------------------------------------------------------

def validate_startup_payload(payload: object) -> tuple[str, ...]:
    if not isinstance(payload, dict):
        return ("artifact is not a JSON object",)
    errors: list[str] = []
    if payload.get("artifact_type") not in (None, "startup_config"):
        errors.append("artifact_type is not startup_config")
    if _session_id_of(payload) is None:
        errors.append("session_id is missing")
    if parse_utc(payload.get("started_at_utc")) is None:
        errors.append("started_at_utc is missing or not timezone-aware")
    strategy = payload.get("strategy_mode")
    if not isinstance(strategy, str) or not strategy.strip():
        errors.append("strategy_mode is missing")
    timeframe = payload.get("bar_timeframe_minutes")
    if type(timeframe) is not int or timeframe <= 0:
        errors.append("bar_timeframe_minutes is missing or invalid")
    symbols = payload.get("symbols")
    if (
        not isinstance(symbols, list)
        or not symbols
        or any(not isinstance(item, str) or not item.strip() for item in symbols)
    ):
        errors.append("symbols are missing or invalid")
    elif payload.get("symbol_count") != len(symbols):
        errors.append("symbol_count does not match symbols")
    if type(payload.get("paper")) is not bool:
        errors.append("paper flag is missing")
    return tuple(errors)


def load_startup_artifacts(log_root: Path, *, display_base: Path | None = None) -> list[StartupArtifact]:
    base = display_base or log_root.parent
    artifacts: list[StartupArtifact] = []
    for day in list_log_days(log_root):
        day_dir = log_root / day
        try:
            paths = sorted(day_dir.glob("startup_config.*.json"))
        except OSError:
            continue
        if not paths and (day_dir / "startup_config.json").is_file():
            # Legacy alias only when no stamped artifact exists.
            paths = [day_dir / "startup_config.json"]
        for path in paths:
            rel = _relative(path, base)
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, UnicodeError, json.JSONDecodeError):
                artifacts.append(StartupArtifact(rel, {}, None, False, ("artifact is unreadable",)))
                continue
            errors = validate_startup_payload(payload)
            session_id = _session_id_of(payload) if isinstance(payload, dict) else None
            artifacts.append(
                StartupArtifact(rel, payload if isinstance(payload, dict) else {}, session_id, not errors, errors)
            )
    return artifacts


# ---------------------------------------------------------------------------
# SQLite history (read-only)
# ---------------------------------------------------------------------------

def _connect_read_only(db_path: Path) -> sqlite3.Connection | None:
    if not db_path.is_file():
        return None
    try:
        connection = sqlite3.connect(f"file:{db_path.as_posix()}?mode=ro", uri=True, timeout=5.0)
    except sqlite3.Error:
        return None
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA query_only=ON")
    return connection


def _columns(connection: sqlite3.Connection, table: str) -> set[str]:
    try:
        return {row[1] for row in connection.execute(f"PRAGMA table_info({table})")}
    except sqlite3.Error:
        return set()


def load_db_sessions(db_path: Path | None) -> list[DbSessionSummary]:
    if db_path is None:
        return []
    connection = _connect_read_only(db_path)
    if connection is None:
        return []
    try:
        columns = _columns(connection, "bot_runs")
        if not {"id", "timestamp_utc"} <= columns:
            return []
        session_expr = "session_id" if "session_id" in columns else "NULL"
        env_expr = "environment" if "environment" in columns else "NULL"
        rows = connection.execute(
            f"SELECT id, {session_expr} AS session_id, {env_expr} AS environment, "
            "timestamp_utc FROM bot_runs ORDER BY id"
        ).fetchall()
    except sqlite3.Error:
        return []
    finally:
        connection.close()
    grouped: dict[str | None, list[sqlite3.Row]] = {}
    for row in rows:
        sid = row["session_id"].strip() if isinstance(row["session_id"], str) and row["session_id"].strip() else None
        grouped.setdefault(sid, []).append(row)
    summaries = []
    for sid, items in grouped.items():
        stamps = sorted(str(item["timestamp_utc"]) for item in items if item["timestamp_utc"])
        envs = tuple(sorted({item["environment"] for item in items}, key=lambda v: (v is None, str(v))))
        summaries.append(
            DbSessionSummary(
                session_id=sid,
                environments=envs,
                run_count=len(items),
                first_ts=stamps[0] if stamps else None,
                last_ts=stamps[-1] if stamps else None,
                run_ids=tuple(int(item["id"]) for item in items),
            )
        )
    return summaries


def load_db_symbol_runs(db_path: Path | None, run_ids: Iterable[int]) -> list[dict[str, Any]]:
    ids = list(run_ids)
    if db_path is None or not ids:
        return []
    connection = _connect_read_only(db_path)
    if connection is None:
        return []
    try:
        if not {"run_id", "symbol", "action"} <= _columns(connection, "symbol_runs"):
            return []
        placeholders = ",".join("?" for _ in ids)
        rows = connection.execute(
            "SELECT sr.id, sr.run_id, sr.symbol, sr.action, sr.price, sr.sma, sr.holding, "
            "sr.quantity, sr.error, br.timestamp_utc "
            "FROM symbol_runs sr JOIN bot_runs br ON br.id = sr.run_id "
            f"WHERE sr.run_id IN ({placeholders}) ORDER BY sr.id",
            ids,
        ).fetchall()
    except sqlite3.Error:
        return []
    finally:
        connection.close()
    return [dict(row) for row in rows]


# ---------------------------------------------------------------------------
# Session catalog
# ---------------------------------------------------------------------------

def _et_day(value: str | None) -> str | None:
    parsed = parse_utc(value)
    return parsed.astimezone(_ET).date().isoformat() if parsed else None


def discover_sessions(log_root: Path, db_path: Path | None = None) -> list[SessionRef]:
    """Return selectable sessions, newest first.

    ``bound`` sessions have an explicit session identifier.  Log days that
    contain records without one are offered separately as ``unbound_log_day``
    so their records are never folded into a session's conclusions.
    """

    bound: dict[str, dict[str, Any]] = {}
    unbound: dict[str, dict[str, Any]] = {}

    def bucket(sid: str) -> dict[str, Any]:
        return bound.setdefault(
            sid, {"days": set(), "stamps": [], "count": 0, "artifacts": [], "db_runs": 0}
        )

    for day in list_log_days(log_root):
        day_log = load_day(log_root, day)
        for record in day_log.records:
            stamp = record.payload.get("ts")
            if record.session_id is not None:
                info = bucket(record.session_id)
                info["days"].add(day)
                info["count"] += 1
                if isinstance(stamp, str):
                    info["stamps"].append(stamp)
            else:
                info = unbound.setdefault(day, {"stamps": [], "count": 0})
                info["count"] += 1
                if isinstance(stamp, str):
                    info["stamps"].append(stamp)

    for artifact in load_startup_artifacts(log_root):
        if artifact.session_id is None:
            continue
        info = bucket(artifact.session_id)
        info["artifacts"].append(artifact.path)
        started = artifact.payload.get("started_at_utc")
        if isinstance(started, str):
            info["stamps"].append(started)
            day = _et_day(started)
            if day:
                info["days"].add(day)

    for summary in load_db_sessions(db_path):
        if summary.session_id is None:
            continue
        info = bucket(summary.session_id)
        info["db_runs"] += summary.run_count
        for stamp in (summary.first_ts, summary.last_ts):
            if stamp:
                info["stamps"].append(stamp)
                day = _et_day(stamp)
                if day:
                    info["days"].add(day)

    def stamp_range(stamps: list[str]) -> tuple[str | None, str | None]:
        parsed = sorted((p, s) for s in stamps if (p := parse_utc(s)) is not None)
        if not parsed:
            return None, None
        return parsed[0][0].isoformat(), parsed[-1][0].isoformat()

    sessions: list[SessionRef] = []
    for sid, info in bound.items():
        first, last = stamp_range(info["stamps"])
        sessions.append(
            SessionRef(
                key=f"session:{sid}",
                kind="bound",
                session_id=sid,
                days=tuple(sorted(info["days"])),
                first_ts=first,
                last_ts=last,
                log_record_count=info["count"],
                startup_artifacts=tuple(sorted(info["artifacts"])),
                db_run_count=info["db_runs"],
            )
        )
    for day, info in unbound.items():
        first, last = stamp_range(info["stamps"])
        sessions.append(
            SessionRef(
                key=f"day:{day}",
                kind="unbound_log_day",
                session_id=None,
                days=(day,),
                first_ts=first,
                last_ts=last,
                log_record_count=info["count"],
                startup_artifacts=(),
                db_run_count=0,
            )
        )
    sessions.sort(key=lambda ref: (ref.last_ts or ref.first_ts or "", ref.key), reverse=True)
    return sessions

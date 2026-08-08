"""Guarded, append-only reconciliation of Automation V1 local position state.

This module is deliberately incapable of submitting or modifying broker orders.
It reads broker/account history, builds an evidence-backed repair plan, and only
mutates SQLite when the caller supplies the exact dry-run database and evidence
digests while holding the Automation V1 execution lock.
"""

from __future__ import annotations

import hashlib
import json
import re
import sqlite3
import tempfile
from contextlib import closing
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Callable, Iterable, Protocol, Sequence

from automation_safety import account_fingerprint, mask_identifier
from tradeos.brokers.base import (
    BrokerAccount,
    BrokerFillActivity,
    BrokerOrder,
    BrokerPosition,
)
from tradeos.execution_lock import ExecutionLock, LockState, LockStatus, inspect_lock


REPAIR_SYMBOL = "COP"
EXPECTED_LOCAL_QUANTITY = Decimal("-8")
HISTORY_START_UTC = datetime(2026, 4, 1, tzinfo=timezone.utc)
OUTCOME_A = "A"
OUTCOME_B = "B"
OUTCOME_C = "C"
BLOCKED_LOCAL_STATE_PROVENANCE_UNKNOWN = "BLOCKED_LOCAL_STATE_PROVENANCE_UNKNOWN"
READY_FOR_HUMAN_APPROVAL_TO_APPLY = "READY_FOR_HUMAN_APPROVAL_TO_APPLY"
TOOL_VERSION = "automation-v1-local-state-repair/2"

_TERMINAL_ORDER_STATUSES = frozenset(
    {"canceled", "cancelled", "expired", "filled", "rejected", "replaced"}
)
_AMBIGUOUS_INTENT_STATUSES = frozenset({"submitting", "ambiguous"})
_REPAIR_ID_RE = re.compile(r"^[0-9]{8}T[0-9]{6}Z-[0-9a-f]{12}$")
_SAFE_HISTORY_FAILURE_DETAILS = frozenset(
    {
        "Malformed Alpaca order history page.",
        "Alpaca repeated an order history page.",
        "Alpaca order history pages are out of order.",
        "Alpaca repeated an order history cursor.",
        "Malformed Alpaca FILL activity history page.",
        "Alpaca repeated a FILL activity history page.",
        "Alpaca FILL activity history pages are out of order.",
        "Alpaca repeated a FILL activity page token.",
    }
)


class RepairSafetyError(RuntimeError):
    """A sanitized, fail-closed repair failure."""


class HistoricalRepairBroker(Protocol):
    """The read-only broker surface used by the repair planner."""

    def get_account(self) -> BrokerAccount: ...

    def list_positions(self) -> list[BrokerPosition]: ...

    def list_open_orders(self) -> list[BrokerOrder]: ...

    def list_orders_since(self, symbol: str, start: datetime) -> list[BrokerOrder]: ...

    def list_fill_activities(
        self, symbol: str, start: datetime
    ) -> list[BrokerFillActivity]: ...


@dataclass(frozen=True)
class LocalRepairState:
    source_run_id: int
    source_timestamp_utc: str
    source_environment: str | None
    source_account_fingerprint: str | None
    source_session_id: str | None
    source_symbol_fingerprint: str | None
    source_run: dict[str, Any]
    symbol_rows: tuple[dict[str, Any], ...]
    local_positions: tuple[tuple[str, Decimal], ...]
    local_cop_quantity: Decimal
    local_orders: tuple[dict[str, Any], ...]
    ambiguous_intents: tuple[dict[str, Any], ...]
    current_scope_binding: dict[str, Any] | None
    next_run_id: int
    missing_schema_changes: tuple[str, ...]


@dataclass(frozen=True)
class HistoricalEvidence:
    local_order_count: int
    matched_local_order_count: int
    broker_order_count: int
    fill_activity_count: int
    local_net_quantity: Decimal
    matched_fill_net_quantity: Decimal
    later_fill_net_quantity: Decimal
    account_fill_net_quantity: Decimal
    matching_order_ids: tuple[str, ...] = ()
    closing_order_ids: tuple[str, ...] = ()
    supporting_fill_activity_ids: tuple[str, ...] = ()
    closing_fill_confirmed: bool = False
    exact_local_history_match: bool = False
    submission_outcome_ambiguous: bool = False
    summary: str = ""


@dataclass(frozen=True)
class RepairPlan:
    checked_at_utc: str
    audit_timestamp_utc: str
    repair_id: str
    database_path: Path
    database_raw_file_sha256_before: str
    database_state_sha256_before: str
    repair_plan_sha256: str
    evidence_sha256: str
    backup_path: Path
    environment: str
    account_id_masked: str
    account_fingerprint: str
    account_status: str
    authenticated_account_snapshot: dict[str, str] | None
    symbol: str
    broker_quantity: Decimal
    local_quantity: Decimal
    broker_position_count: int
    open_pending_order_count: int
    outcome: str
    outcome_reason: str
    evidence: HistoricalEvidence
    source_run_id: int
    repaired_run_id: int
    schema_changes: tuple[str, ...]
    proposed_changes: tuple[dict[str, Any], ...]
    audit_record: dict[str, Any]
    checks: tuple[tuple[str, bool], ...]
    application_allowed: bool
    apply_command: str | None
    applied: bool = False
    backup_created: bool = False
    backup_raw_file_sha256: str | None = None
    database_raw_file_sha256_after: str | None = None
    database_state_sha256_after: str | None = None

    @property
    def verdict(self) -> str:
        return (
            READY_FOR_HUMAN_APPROVAL_TO_APPLY
            if self.application_allowed and not self.applied
            else BLOCKED_LOCAL_STATE_PROVENANCE_UNKNOWN
        )


def _decimal(value: Any, *, field: str) -> Decimal:
    if value is None or isinstance(value, bool):
        raise RepairSafetyError(f"{field} is missing or invalid")
    try:
        result = Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError) as exc:
        raise RepairSafetyError(f"{field} is malformed") from exc
    if not result.is_finite():
        raise RepairSafetyError(f"{field} is not finite")
    return result


def _decimal_text(value: Decimal | Any) -> str:
    normalized = _decimal(value, field="quantity")
    text = format(normalized, "f")
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    return text or "0"


def _authenticated_account_snapshot(account: BrokerAccount) -> dict[str, str]:
    """Return finite broker-derived values safe for a clean scoped snapshot."""

    cash = _decimal(getattr(account, "cash", None), field="broker account cash")
    buying_power = _decimal(
        getattr(account, "buying_power", None),
        field="broker account buying power",
    )
    equity = _decimal(
        getattr(account, "equity", None), field="broker account equity"
    )
    last_equity = _decimal(
        getattr(account, "last_equity", None),
        field="broker account last equity",
    )
    return {
        "cash": _decimal_text(cash),
        "buying_power": _decimal_text(buying_power),
        "equity": _decimal_text(equity),
        "last_equity": _decimal_text(last_equity),
        "daily_pnl": _decimal_text(equity - last_equity),
    }


def _enum_text(value: Any) -> str:
    text = str(value or "").strip()
    if "." in text:
        text = text.rsplit(".", 1)[-1]
    return text.lower()


def _parse_time(value: Any, *, field: str) -> datetime:
    text = str(value or "").strip()
    if text.endswith("Z"):
        text = f"{text[:-1]}+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError as exc:
        raise RepairSafetyError(f"{field} is not a valid timestamp") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise RepairSafetyError(f"{field} is not timezone-aware")
    return parsed.astimezone(timezone.utc)


def _side_sign(side: Any) -> Decimal:
    normalized = _enum_text(side)
    if normalized.startswith("buy"):
        return Decimal("1")
    if normalized.startswith("sell"):
        return Decimal("-1")
    raise RepairSafetyError("broker or local history contains an unknown order side")


def _account_identity(account: BrokerAccount) -> tuple[str, tuple[str, ...]]:
    candidates = tuple(
        value
        for value in (
            str(getattr(account, "account_id", "") or "").strip(),
            str(getattr(account, "account_number", "") or "").strip(),
        )
        if value
    )
    return (candidates[0] if candidates else ""), candidates


def raw_file_sha256(path: Path) -> str:
    """Return the SHA-256 of exactly the bytes in one filesystem file."""

    resolved = Path(path).resolve(strict=True)
    digest = hashlib.sha256()
    with resolved.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def database_state_sha256(database_path: Path) -> str:
    """Hash the named SQLite main file plus any data-bearing WAL sidecar.

    The shared-memory file is intentionally excluded because it is coordination
    metadata rather than durable database content.  This is a repair concurrency
    token, not a raw file hash or a semantic SQLite-content digest: names, sizes,
    bytes, and an explicit missing-WAL marker are all covered.
    """

    path = Path(database_path).resolve(strict=True)
    digest = hashlib.sha256()
    for candidate in (path, Path(f"{path}-wal")):
        digest.update(candidate.name.encode("utf-8"))
        if not candidate.exists():
            digest.update(b"\x00missing\x00")
            continue
        digest.update(str(candidate.stat().st_size).encode("ascii"))
        with candidate.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
    return digest.hexdigest()


def _connect_read_only(database_path: Path) -> sqlite3.Connection:
    path = Path(database_path).resolve(strict=True)
    connection = sqlite3.connect(f"file:{path.as_posix()}?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA query_only=ON")
    return connection


def _table_names(connection: sqlite3.Connection) -> set[str]:
    return {
        str(row[0])
        for row in connection.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table'"
        ).fetchall()
    }


def _columns(connection: sqlite3.Connection, table: str) -> set[str]:
    return {
        str(row[1])
        for row in connection.execute(f'PRAGMA table_info("{table}")').fetchall()
    }


def _runtime_schema_delta(connection: sqlite3.Connection) -> tuple[str, ...]:
    tables = _table_names(connection)
    changes: list[str] = []
    bot_columns = _columns(connection, "bot_runs")
    order_columns = _columns(connection, "order_history")
    intent_columns = _columns(connection, "order_intents")
    if "environment" not in bot_columns:
        changes.append("ALTER bot_runs ADD environment")
    if "account_fingerprint" not in bot_columns:
        changes.append("ALTER bot_runs ADD account_fingerprint")
    if "environment" not in order_columns:
        changes.append("ALTER order_history ADD environment")
    if "account_fingerprint" not in order_columns:
        changes.append("ALTER order_history ADD account_fingerprint")
    if "environment" not in intent_columns:
        changes.append("ALTER order_intents ADD environment")
    if "runtime_state_bindings" not in tables:
        changes.append("CREATE runtime_state_bindings")
    if "local_state_reconciliation_events" not in tables:
        changes.append("CREATE immutable local_state_reconciliation_events")
    return tuple(changes)


def _read_binding(
    connection: sqlite3.Connection,
    *,
    environment: str,
    fingerprint: str,
) -> dict[str, Any] | None:
    if "runtime_state_bindings" not in _table_names(connection):
        return None
    row = connection.execute(
        """
        SELECT *
        FROM runtime_state_bindings
        WHERE environment = ? AND account_fingerprint = ?
        """,
        (environment, fingerprint),
    ).fetchone()
    return dict(row) if row is not None else None


def _read_local_state(
    database_path: Path,
    *,
    environment: str,
    fingerprint: str,
    symbol: str,
) -> LocalRepairState:
    with closing(_connect_read_only(database_path)) as connection:
        tables = _table_names(connection)
        required = {"bot_runs", "symbol_runs", "order_history", "order_intents"}
        if not required.issubset(tables):
            missing = ",".join(sorted(required - tables))
            raise RepairSafetyError(f"database is missing required tables: {missing}")

        bot_columns = _columns(connection, "bot_runs")
        binding = _read_binding(
            connection, environment=environment, fingerprint=fingerprint
        )
        if (
            binding is not None
            and {"environment", "account_fingerprint"}.issubset(bot_columns)
        ):
            source = connection.execute(
                """
                SELECT * FROM bot_runs
                WHERE environment = ? AND account_fingerprint = ?
                ORDER BY id DESC LIMIT 1
                """,
                (environment, fingerprint),
            ).fetchone()
        else:
            # Unbound legacy state is inspected for explicit migration only;
            # normal runtime readers never receive this fallback.
            source = connection.execute(
                "SELECT * FROM bot_runs ORDER BY id DESC LIMIT 1"
            ).fetchone()
        if source is None:
            raise RepairSafetyError("database contains no local runtime snapshot")
        source_run = dict(source)
        source_run_id = int(source_run["id"])
        source_environment = (
            str(source_run.get("environment") or "").strip().lower() or None
        )
        source_fingerprint = (
            str(source_run.get("account_fingerprint") or "").strip() or None
        )
        if bool(source_environment) != bool(source_fingerprint):
            raise RepairSafetyError("latest local snapshot has incomplete runtime scope")
        symbol_rows = tuple(
            dict(row)
            for row in connection.execute(
                "SELECT * FROM symbol_runs WHERE run_id = ? ORDER BY id",
                (source_run_id,),
            ).fetchall()
        )
        normalized_symbol = symbol.strip().upper()
        matching_rows = [
            row
            for row in symbol_rows
            if str(row.get("symbol", "")).strip().upper() == normalized_symbol
        ]
        if len(matching_rows) != 1:
            raise RepairSafetyError(
                "latest local snapshot does not contain exactly one COP row"
            )
        local_positions: list[tuple[str, Decimal]] = []
        for row in symbol_rows:
            qty = _decimal(row.get("quantity", 0), field="local position quantity")
            if qty != 0:
                local_positions.append(
                    (str(row.get("symbol", "")).strip().upper(), qty)
                )

        order_columns = _columns(connection, "order_history")
        has_order_scope = {
            "environment",
            "account_fingerprint",
        }.issubset(order_columns)
        has_partial_order_scope = bool(
            {"environment", "account_fingerprint"} & order_columns
        ) and not has_order_scope
        if has_partial_order_scope:
            raise RepairSafetyError("local order history has incomplete runtime scope")
        if has_order_scope and source_environment and source_fingerprint:
            order_scope_sql = (
                "AND environment = ? AND account_fingerprint = ?"
            )
            order_scope_parameters: tuple[Any, ...] = (
                source_environment,
                source_fingerprint,
            )
        elif has_order_scope:
            # Legacy state is reconciled only against legacy/unbound orders;
            # records already assigned to another namespace are never mixed in.
            order_scope_sql = (
                "AND environment IS NULL AND account_fingerprint IS NULL"
            )
            order_scope_parameters = ()
        else:
            order_scope_sql = ""
            order_scope_parameters = ()
        orders = tuple(
            dict(row)
            for row in connection.execute(
                f"""
                SELECT * FROM order_history
                WHERE UPPER(symbol) = ? {order_scope_sql}
                ORDER BY COALESCE(filled_at, submitted_at, observed_at_utc), order_id
                """,
                (normalized_symbol, *order_scope_parameters),
            ).fetchall()
        )
        ambiguous_intents = tuple(
            dict(row)
            for row in connection.execute(
                """
                SELECT * FROM order_intents
                WHERE LOWER(status) IN ('submitting', 'ambiguous')
                ORDER BY created_at_utc, intent_id
                """
            ).fetchall()
        )
        sequence = connection.execute(
            "SELECT seq FROM sqlite_sequence WHERE name = 'bot_runs'"
        ).fetchone()
        max_id = int(
            connection.execute("SELECT COALESCE(MAX(id), 0) FROM bot_runs").fetchone()[
                0
            ]
        )
        next_run_id = max(max_id, int(sequence[0]) if sequence else 0) + 1

        return LocalRepairState(
            source_run_id=source_run_id,
            source_timestamp_utc=str(source_run.get("timestamp_utc") or ""),
            source_environment=source_environment,
            source_account_fingerprint=source_fingerprint,
            source_session_id=str(source_run.get("session_id") or "") or None,
            source_symbol_fingerprint=str(
                source_run.get("symbol_fingerprint") or ""
            )
            or None,
            source_run=source_run,
            symbol_rows=symbol_rows,
            local_positions=tuple(sorted(local_positions)),
            local_cop_quantity=_decimal(
                matching_rows[0].get("quantity"), field="local COP quantity"
            ),
            local_orders=orders,
            ambiguous_intents=ambiguous_intents,
            current_scope_binding=binding,
            next_run_id=next_run_id,
            missing_schema_changes=_runtime_schema_delta(connection),
        )


def _order_identity_matches(local: dict[str, Any], broker: BrokerOrder) -> bool:
    try:
        local_qty = _decimal(local.get("qty"), field="local order quantity")
        broker_qty = _decimal(broker.qty, field="broker order quantity")
        local_filled = _decimal(
            local.get("filled_qty"), field="local filled quantity"
        )
        broker_filled = _decimal(
            broker.filled_qty, field="broker filled quantity"
        )
    except RepairSafetyError:
        return False
    return all(
        (
            str(local.get("symbol", "")).strip().upper()
            == str(broker.symbol or "").strip().upper(),
            _enum_text(local.get("side")) == _enum_text(broker.side),
            _enum_text(local.get("status")) == _enum_text(broker.status),
            local_qty == broker_qty,
            local_filled == broker_filled,
        )
    )


def _history_evidence(
    local_state: LocalRepairState,
    broker_orders: Sequence[BrokerOrder],
    fills: Sequence[BrokerFillActivity],
) -> HistoricalEvidence:
    source_snapshot_time = _parse_time(
        local_state.source_timestamp_utc,
        field="latest local snapshot time",
    )
    local_by_id = {
        str(row.get("order_id") or "").strip(): row
        for row in local_state.local_orders
        if str(row.get("order_id") or "").strip()
    }
    if len(local_by_id) != len(local_state.local_orders):
        raise RepairSafetyError("local COP history has missing or duplicate order IDs")
    broker_by_id: dict[str, BrokerOrder] = {}
    for order in broker_orders:
        order_id = str(order.order_id or "").strip()
        if not order_id or order_id in broker_by_id:
            raise RepairSafetyError("broker COP history has missing or duplicate order IDs")
        if str(order.symbol or "").strip().upper() != REPAIR_SYMBOL:
            raise RepairSafetyError("broker COP order history contains another symbol")
        _side_sign(order.side)
        broker_by_id[order_id] = order

    matching_ids = tuple(sorted(set(local_by_id) & set(broker_by_id)))
    exact_match = bool(local_by_id) and len(matching_ids) == len(local_by_id)
    if exact_match:
        exact_match = all(
            _order_identity_matches(local_by_id[order_id], broker_by_id[order_id])
            for order_id in matching_ids
        )

    local_net = Decimal("0")
    for row in local_state.local_orders:
        status = _enum_text(row.get("status"))
        if status != "filled":
            raise RepairSafetyError("local COP order history is not terminal-filled")
        filled = _decimal(row.get("filled_qty"), field="local filled quantity")
        local_net += _side_sign(row.get("side")) * filled
        _parse_time(
            row.get("filled_at") or row.get("submitted_at"),
            field="local order fill time",
        )

    seen_activities: set[str] = set()
    matched_fill_net = Decimal("0")
    later_fill_net = Decimal("0")
    total_fill_net = Decimal("0")
    closing_order_ids: list[str] = []
    supporting_activity_ids: list[str] = []
    fill_by_local_order: dict[str, Decimal] = {key: Decimal("0") for key in local_by_id}
    fill_side_by_local_order: dict[str, Decimal] = {}
    fill_qty_by_order: dict[str, Decimal] = {}
    for activity in fills:
        activity_id = str(activity.activity_id or "").strip()
        order_id = str(activity.order_id or "").strip()
        if not activity_id or activity_id in seen_activities or not order_id:
            raise RepairSafetyError("broker fill activity history is malformed or repeated")
        if str(activity.symbol or "").strip().upper() != REPAIR_SYMBOL:
            raise RepairSafetyError("broker COP fill history contains another symbol")
        broker_order = broker_by_id.get(order_id)
        if broker_order is None:
            raise RepairSafetyError("broker COP fill does not match a retrieved order")
        seen_activities.add(activity_id)
        qty = _decimal(activity.qty, field="broker fill activity quantity")
        signed = _side_sign(activity.side) * qty
        if _side_sign(activity.side) != _side_sign(broker_order.side):
            raise RepairSafetyError("broker COP order and fill sides conflict")
        fill_qty_by_order[order_id] = fill_qty_by_order.get(
            order_id, Decimal("0")
        ) + qty
        total_fill_net += signed
        activity_time = _parse_time(
            activity.transaction_time, field="broker fill activity time"
        )
        if order_id in local_by_id:
            matched_fill_net += signed
            fill_by_local_order[order_id] += qty
            fill_side_by_local_order.setdefault(order_id, _side_sign(activity.side))
            if fill_side_by_local_order[order_id] != _side_sign(activity.side):
                raise RepairSafetyError("one broker order has conflicting fill sides")
            supporting_activity_ids.append(activity_id)
        elif activity_time > source_snapshot_time:
            later_fill_net += signed
            closing_order_ids.append(order_id)
            supporting_activity_ids.append(activity_id)

    fill_match = exact_match
    if fill_match:
        for order_id, local in local_by_id.items():
            expected_qty = _decimal(
                local.get("filled_qty"), field="local filled quantity"
            )
            if fill_by_local_order.get(order_id, Decimal("0")) != expected_qty:
                fill_match = False
                break
            if fill_side_by_local_order.get(order_id) != _side_sign(local.get("side")):
                fill_match = False
                break

    unique_closing_ids = tuple(sorted(set(closing_order_ids)))
    closing_orders_valid = bool(unique_closing_ids)
    for order_id in unique_closing_ids:
        order = broker_by_id.get(order_id)
        order_qty = (
            _decimal(order.qty, field="closing broker order quantity")
            if order is not None
            else Decimal("0")
        )
        order_filled_qty = (
            _decimal(order.filled_qty, field="closing broker filled quantity")
            if order is not None
            else Decimal("0")
        )
        if (
            order is None
            or _enum_text(order.status) != "filled"
            or _side_sign(order.side) != Decimal("1")
            or str(order.symbol or "").strip().upper() != REPAIR_SYMBOL
            or order_qty != order_filled_qty
            or fill_qty_by_order.get(order_id, Decimal("0")) != order_filled_qty
        ):
            closing_orders_valid = False
            break
    closing_fill_confirmed = all(
        (
            fill_match,
            local_net == EXPECTED_LOCAL_QUANTITY,
            matched_fill_net == EXPECTED_LOCAL_QUANTITY,
            later_fill_net == -EXPECTED_LOCAL_QUANTITY,
            total_fill_net == Decimal("0"),
            closing_orders_valid,
        )
    )
    submission_ambiguous = any(
        _enum_text(order.status) not in _TERMINAL_ORDER_STATUSES
        for order in broker_orders
    )
    summary = (
        f"local_orders={len(local_by_id)} matched={len(matching_ids)}; "
        f"local_net={_decimal_text(local_net)}; "
        f"matched_fill_net={_decimal_text(matched_fill_net)}; "
        f"later_fill_net={_decimal_text(later_fill_net)}; "
        f"account_fill_net={_decimal_text(total_fill_net)}; "
        f"closing_fill_confirmed={'true' if closing_fill_confirmed else 'false'}"
    )
    return HistoricalEvidence(
        local_order_count=len(local_by_id),
        matched_local_order_count=len(matching_ids),
        broker_order_count=len(broker_orders),
        fill_activity_count=len(fills),
        local_net_quantity=local_net,
        matched_fill_net_quantity=matched_fill_net,
        later_fill_net_quantity=later_fill_net,
        account_fill_net_quantity=total_fill_net,
        matching_order_ids=matching_ids,
        closing_order_ids=unique_closing_ids,
        supporting_fill_activity_ids=tuple(sorted(set(supporting_activity_ids))),
        closing_fill_confirmed=closing_fill_confirmed,
        exact_local_history_match=fill_match,
        submission_outcome_ambiguous=submission_ambiguous,
        summary=summary,
    )


def _evidence_digest(
    *,
    environment: str,
    fingerprint: str,
    local_state: LocalRepairState,
    broker_orders: Sequence[BrokerOrder],
    fills: Sequence[BrokerFillActivity],
    evidence: HistoricalEvidence,
) -> str:
    payload = {
        "environment": environment,
        "account_fingerprint": fingerprint,
        "source_run_id": local_state.source_run_id,
        "source_timestamp_utc": local_state.source_timestamp_utc,
        "local_quantity": _decimal_text(local_state.local_cop_quantity),
        "orders": [
            {
                "id": order.order_id,
                "client_order_id": order.client_order_id,
                "symbol": order.symbol,
                "side": _enum_text(order.side),
                "status": _enum_text(order.status),
                "qty": _decimal_text(order.qty or 0),
                "filled_qty": _decimal_text(order.filled_qty or 0),
                "submitted_at": order.submitted_at,
                "filled_at": order.filled_at,
                "replaces": order.replaces,
                "replaced_by": order.replaced_by,
            }
            for order in broker_orders
        ],
        "fills": [
            {
                "id": fill.activity_id,
                "order_id": fill.order_id,
                "side": _enum_text(fill.side),
                "qty": _decimal_text(fill.qty),
                "transaction_time": fill.transaction_time,
            }
            for fill in fills
        ],
        "evidence": {
            "matching_order_ids": evidence.matching_order_ids,
            "closing_order_ids": evidence.closing_order_ids,
            "closing_fill_confirmed": evidence.closing_fill_confirmed,
        },
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )
    return hashlib.sha256(encoded).hexdigest()


def _audit_time_and_repair_id(
    database_state_sha256_before: str, repair_id: str | None
) -> tuple[str, str]:
    if repair_id is None:
        current = datetime.now(timezone.utc).replace(microsecond=0)
        repair_id = (
            f"{current.strftime('%Y%m%dT%H%M%SZ')}-"
            f"{database_state_sha256_before[:12]}"
        )
    if not _REPAIR_ID_RE.fullmatch(repair_id):
        raise RepairSafetyError("repair ID is malformed")
    timestamp = datetime.strptime(repair_id[:16], "%Y%m%dT%H%M%SZ").replace(
        tzinfo=timezone.utc
    )
    return timestamp.isoformat().replace("+00:00", "Z"), repair_id


def _default_backup_path(database_path: Path, repair_id: str) -> Path:
    return (
        Path(tempfile.gettempdir()).resolve()
        / "tradeos-local-state-backups"
        / f"{database_path.name}.{repair_id}.bak"
    )


def build_repair_plan(
    *,
    broker: HistoricalRepairBroker,
    database_path: Path,
    configured_paper: bool,
    expected_account_id: str | None,
    symbol: str,
    repair_id: str | None = None,
    lock_inspector: Callable[[], LockStatus] = inspect_lock,
    backup_root: Path | None = None,
) -> RepairPlan:
    normalized_symbol = str(symbol or "").strip().upper()
    if normalized_symbol != REPAIR_SYMBOL:
        raise RepairSafetyError("this repair command is restricted to symbol COP")
    if configured_paper is not True:
        raise RepairSafetyError("paper environment is not positively configured")
    expected = str(expected_account_id or "").strip()
    if not expected:
        raise RepairSafetyError("expected account ID is not configured")

    checked_at = datetime.now(timezone.utc).isoformat()
    try:
        account = broker.get_account()
    except Exception as exc:
        raise RepairSafetyError(
            f"broker account query failed: {type(exc).__name__}"
        ) from exc
    environment = _enum_text(getattr(account, "environment", "unknown")) or "unknown"
    actual_account_id, candidates = _account_identity(account)
    status = _enum_text(getattr(account, "status", "unknown")) or "unknown"
    if environment != "paper":
        raise RepairSafetyError("authenticated broker environment is not paper")
    if not actual_account_id:
        raise RepairSafetyError("authenticated broker account identifier is missing")
    if expected not in candidates:
        raise RepairSafetyError("authenticated account does not match expected account")
    if status != "active":
        raise RepairSafetyError("authenticated broker account is not active")
    for field_name in ("trading_blocked", "account_blocked", "trade_suspended_by_user"):
        if bool(getattr(account, field_name, False)):
            raise RepairSafetyError(f"authenticated account reports {field_name}")
    fingerprint = account_fingerprint(actual_account_id)

    database_path = Path(database_path).resolve(strict=True)
    raw_file_digest_before = raw_file_sha256(database_path)
    state_digest_before = database_state_sha256(database_path)
    local_state = _read_local_state(
        database_path,
        environment=environment,
        fingerprint=fingerprint,
        symbol=normalized_symbol,
    )

    broker_reads: tuple[tuple[str, Callable[[], Iterable[Any]]], ...] = (
        ("positions", broker.list_positions),
        ("open orders", broker.list_open_orders),
        (
            "COP order history",
            lambda: broker.list_orders_since(normalized_symbol, HISTORY_START_UTC),
        ),
        (
            "COP fill history",
            lambda: broker.list_fill_activities(normalized_symbol, HISTORY_START_UTC),
        ),
    )
    broker_results: list[list[Any]] = []
    for stage, reader in broker_reads:
        try:
            broker_results.append(list(reader()))
        except Exception as exc:
            detail = str(exc).strip()
            safe_detail = (
                f" ({detail})" if detail in _SAFE_HISTORY_FAILURE_DETAILS else ""
            )
            raise RepairSafetyError(
                f"read-only broker {stage} query failed: "
                f"{type(exc).__name__}{safe_detail}"
            ) from exc
    positions, open_orders, broker_orders, fills = broker_results

    raw_file_digest_after = raw_file_sha256(database_path)
    state_digest_after = database_state_sha256(database_path)
    if (
        raw_file_digest_after != raw_file_digest_before
        or state_digest_after != state_digest_before
    ):
        raise RepairSafetyError("database changed during dry-run inspection")

    broker_position_map: dict[str, Decimal] = {}
    for position in positions:
        position_symbol = str(position.symbol or "").strip().upper()
        qty = _decimal(position.qty, field="broker position quantity")
        if not position_symbol or qty == 0 or position_symbol in broker_position_map:
            raise RepairSafetyError("broker returned malformed or duplicate positions")
        broker_position_map[position_symbol] = qty
    for order in open_orders:
        if _enum_text(order.status) in _TERMINAL_ORDER_STATUSES:
            raise RepairSafetyError("open-order query returned a terminal order")
    broker_qty = broker_position_map.get(normalized_symbol, Decimal("0"))

    evidence = _history_evidence(local_state, broker_orders, fills)
    matched_count = evidence.matched_local_order_count
    local_count = evidence.local_order_count
    source_scope_mismatch = bool(
        (local_state.source_environment and local_state.source_environment != environment)
        or (
            local_state.source_account_fingerprint
            and local_state.source_account_fingerprint != fingerprint
        )
    )
    if source_scope_mismatch:
        outcome = OUTCOME_B
        outcome_reason = (
            "the latest local COP state is explicitly assigned to a different "
            "account or environment and must remain in its legacy namespace"
        )
    elif matched_count == local_count and local_count > 0 and evidence.closing_fill_confirmed:
        outcome = OUTCOME_A
        outcome_reason = (
            "the authenticated paper account contains the exact persisted COP short "
            "history and a later filled buy returned COP to flat"
        )
    elif matched_count == 0 and local_count > 0:
        outcome = OUTCOME_B
        outcome_reason = (
            "the local COP state is outside the authenticated account history and "
            "must remain legacy state while a clean scoped namespace is initialized"
        )
    else:
        outcome = OUTCOME_C
        outcome_reason = (
            "local and broker COP history cannot be matched completely and conclusively"
        )
    authenticated_snapshot = (
        _authenticated_account_snapshot(account) if outcome == OUTCOME_B else None
    )

    other_local_positions = tuple(
        (key, value)
        for key, value in local_state.local_positions
        if key != normalized_symbol and value != 0
    )
    lock_status = lock_inspector()
    lock_available = lock_status.state is LockState.UNLOCKED or (
        lock_status.state is LockState.HELD and lock_status.held_by_current_process
    )
    checks = (
        ("paper_environment_verified", environment == "paper"),
        ("expected_account_configured", bool(expected)),
        ("authenticated_account_matches_expected", expected in candidates),
        ("account_active", status == "active"),
        ("fresh_read_only_reconciliation_succeeded", True),
        ("no_broker_cop_position", broker_qty == 0),
        ("no_broker_positions", not broker_position_map),
        ("no_open_or_pending_cop_order", not any(str(o.symbol).upper() == normalized_symbol for o in open_orders)),
        ("no_unmanaged_position", not positions),
        ("no_unmanaged_order", not open_orders),
        ("no_ambiguous_order_intent", not local_state.ambiguous_intents),
        ("local_cop_quantity_is_exactly_minus_8", local_state.local_cop_quantity == EXPECTED_LOCAL_QUANTITY),
        ("no_other_local_open_position", not other_local_positions),
        (
            "database_unchanged_during_inspection",
            raw_file_digest_after == raw_file_digest_before
            and state_digest_after == state_digest_before,
        ),
        ("historical_evidence_supports_outcome", outcome in {OUTCOME_A, OUTCOME_B}),
        ("submission_outcomes_terminal", not evidence.submission_outcome_ambiguous),
        ("execution_lock_available", lock_available),
    )
    application_allowed = all(value for _, value in checks)

    evidence_sha = _evidence_digest(
        environment=environment,
        fingerprint=fingerprint,
        local_state=local_state,
        broker_orders=broker_orders,
        fills=fills,
        evidence=evidence,
    )
    audit_timestamp, selected_repair_id = _audit_time_and_repair_id(
        state_digest_before, repair_id
    )
    backup_path = (
        Path(backup_root).resolve() / f"{database_path.name}.{selected_repair_id}.bak"
        if backup_root is not None
        else _default_backup_path(database_path, selected_repair_id)
    )
    reason = (
        "broker-confirmed closing fill after latest local snapshot"
        if outcome == OUTCOME_A
        else "clean authenticated account namespace initialization"
    )
    audit_record: dict[str, Any] = {
        "repair_id": selected_repair_id,
        "timestamp_utc": audit_timestamp,
        "environment": environment,
        "account_fingerprint": fingerprint,
        "account_id_masked": mask_identifier(actual_account_id),
        "symbol": normalized_symbol,
        "previous_quantity": _decimal_text(local_state.local_cop_quantity),
        "new_quantity": "0",
        "outcome": outcome,
        "reason": reason,
        "source_run_id": local_state.source_run_id,
        "repaired_run_id": local_state.next_run_id,
        "supporting_broker_order_ids": list(
            dict.fromkeys(
                [*evidence.matching_order_ids, *evidence.closing_order_ids]
            )
        ),
        "supporting_fill_activity_ids": list(
            evidence.supporting_fill_activity_ids
        ),
        "evidence_sha256": evidence_sha,
        "database_raw_file_sha256_before": raw_file_digest_before,
        "database_state_sha256_before": state_digest_before,
        "backup_raw_file_sha256": None,
        "backup_path": str(backup_path),
        "tool_version": TOOL_VERSION,
    }
    binding_exists = local_state.current_scope_binding is not None
    copies_source_state = outcome == OUTCOME_A
    binding_history_cutoff = (
        local_state.source_timestamp_utc
        if copies_source_state
        else audit_timestamp
    )
    proposed_changes = (
        {
            "table": "bot_runs",
            "operation": "INSERT_COPY" if copies_source_state else "INSERT_INITIALIZED",
            "id": local_state.next_run_id,
            "session_id": f"repair-{selected_repair_id}",
            "environment": environment,
            "account_fingerprint": fingerprint,
            "state_source": (
                "local_source_run"
                if copies_source_state
                else "authenticated_broker_account"
            ),
            "source_run_id": (
                local_state.source_run_id if copies_source_state else None
            ),
            "account_snapshot": (
                None if copies_source_state else authenticated_snapshot
            ),
            "symbol_fingerprint": (
                local_state.source_symbol_fingerprint if copies_source_state else None
            ),
        },
        {
            "table": "symbol_runs",
            "operation": (
                "INSERT_COPY" if copies_source_state else "INSERT_INITIALIZED_FLAT"
            ),
            "row_count": len(local_state.symbol_rows) if copies_source_state else 1,
            "source_run_id": (
                local_state.source_run_id if copies_source_state else None
            ),
            "new_run_id": local_state.next_run_id,
            "symbol": normalized_symbol,
            "field_changes": {
                "action": "RECONCILED_FLAT",
                "holding": 0,
                "holding_minutes": None,
                "quantity": 0,
                "market_value": 0,
                "avg_entry_price": None,
                "unrealized_pl": None,
                "error": None,
            },
        },
        {
            "table": "runtime_state_bindings",
            "operation": "UNCHANGED" if binding_exists else "INSERT",
            "environment": environment,
            "account_fingerprint": fingerprint,
            "history_cutoff_utc": binding_history_cutoff,
        },
        {
            "table": "local_state_reconciliation_events",
            "operation": "INSERT",
            "repair_id": selected_repair_id,
        },
    )
    plan_digest_payload = {
        "tool_version": TOOL_VERSION,
        "repair_id": selected_repair_id,
        "audit_timestamp_utc": audit_timestamp,
        "database_path": str(database_path),
        "database_raw_file_sha256_before": raw_file_digest_before,
        "database_state_sha256_before": state_digest_before,
        "evidence_sha256": evidence_sha,
        "environment": environment,
        "account_fingerprint": fingerprint,
        "authenticated_account_snapshot": authenticated_snapshot,
        "symbol": normalized_symbol,
        "broker_quantity": _decimal_text(broker_qty),
        "local_quantity": _decimal_text(local_state.local_cop_quantity),
        "outcome": outcome,
        "outcome_reason": outcome_reason,
        "source_run_id": local_state.source_run_id,
        "repaired_run_id": local_state.next_run_id,
        "schema_changes": list(local_state.missing_schema_changes),
        "proposed_changes": list(proposed_changes),
        "audit_record_template": audit_record,
        "checks": list(checks),
        "application_allowed": application_allowed,
    }
    plan_sha = hashlib.sha256(
        json.dumps(
            plan_digest_payload,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    audit_record["repair_plan_sha256"] = plan_sha

    apply_command = None
    if application_allowed:
        apply_command = (
            "python -m tradeos repair-local-state --symbol COP --apply "
            f"--repair-id {selected_repair_id} "
            f"--approved-db-raw-sha256 {raw_file_digest_before} "
            f"--approved-db-state-sha256 {state_digest_before} "
            f"--approved-plan-sha256 {plan_sha} "
            f"--approved-evidence-sha256 {evidence_sha}"
        )
    return RepairPlan(
        checked_at_utc=checked_at,
        audit_timestamp_utc=audit_timestamp,
        repair_id=selected_repair_id,
        database_path=database_path,
        database_raw_file_sha256_before=raw_file_digest_before,
        database_state_sha256_before=state_digest_before,
        repair_plan_sha256=plan_sha,
        evidence_sha256=evidence_sha,
        backup_path=backup_path,
        environment=environment,
        account_id_masked=mask_identifier(actual_account_id),
        account_fingerprint=fingerprint,
        account_status=status,
        authenticated_account_snapshot=authenticated_snapshot,
        symbol=normalized_symbol,
        broker_quantity=broker_qty,
        local_quantity=local_state.local_cop_quantity,
        broker_position_count=len(positions),
        open_pending_order_count=len(open_orders),
        outcome=outcome,
        outcome_reason=outcome_reason,
        evidence=evidence,
        source_run_id=local_state.source_run_id,
        repaired_run_id=local_state.next_run_id,
        schema_changes=local_state.missing_schema_changes,
        proposed_changes=proposed_changes,
        audit_record=audit_record,
        checks=checks,
        application_allowed=application_allowed,
        apply_command=apply_command,
    )


def create_verified_database_backup(source_path: Path, backup_path: Path) -> Path:
    source = Path(source_path).resolve(strict=True)
    destination = Path(backup_path).resolve(strict=False)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise RepairSafetyError("refusing to overwrite an existing database backup")
    source_connection = sqlite3.connect(f"file:{source.as_posix()}?mode=ro", uri=True)
    destination_connection: sqlite3.Connection | None = None
    try:
        destination_connection = sqlite3.connect(destination)
        source_connection.backup(destination_connection)
        destination_connection.commit()
    except Exception:
        if destination.exists():
            destination.unlink(missing_ok=True)
        raise
    finally:
        source_connection.close()
        if destination_connection is not None:
            destination_connection.close()
    if not destination.is_file() or destination.stat().st_size <= 0:
        raise RepairSafetyError("database backup was not created")
    try:
        verification = sqlite3.connect(
            f"file:{destination.as_posix()}?mode=ro", uri=True
        )
        result = verification.execute("PRAGMA quick_check").fetchone()
        verification.close()
    except Exception as exc:
        raise RepairSafetyError("database backup is not readable") from exc
    if result is None or str(result[0]).lower() != "ok":
        raise RepairSafetyError("database backup integrity verification failed")
    return destination


def _ensure_audit_schema(connection: sqlite3.Connection) -> None:
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS local_state_reconciliation_events (
            repair_id TEXT PRIMARY KEY,
            timestamp_utc TEXT NOT NULL,
            environment TEXT NOT NULL,
            account_fingerprint TEXT NOT NULL,
            account_id_masked TEXT NOT NULL,
            symbol TEXT NOT NULL,
            previous_quantity TEXT NOT NULL,
            new_quantity TEXT NOT NULL,
            outcome TEXT NOT NULL,
            reason TEXT NOT NULL,
            source_run_id INTEGER NOT NULL,
            repaired_run_id INTEGER NOT NULL,
            supporting_broker_order_ids_json TEXT NOT NULL,
            supporting_fill_activity_ids_json TEXT NOT NULL,
            evidence_sha256 TEXT NOT NULL,
            repair_plan_sha256 TEXT NOT NULL,
            database_raw_file_sha256_before TEXT NOT NULL,
            database_state_sha256_before TEXT NOT NULL,
            backup_raw_file_sha256 TEXT NOT NULL,
            backup_path TEXT NOT NULL,
            tool_version TEXT NOT NULL
        )
        """
    )
    connection.execute(
        """
        CREATE TRIGGER IF NOT EXISTS local_state_reconciliation_events_no_update
        BEFORE UPDATE ON local_state_reconciliation_events
        BEGIN
            SELECT RAISE(ABORT, 'local state reconciliation events are immutable');
        END
        """
    )
    connection.execute(
        """
        CREATE TRIGGER IF NOT EXISTS local_state_reconciliation_events_no_delete
        BEFORE DELETE ON local_state_reconciliation_events
        BEGIN
            SELECT RAISE(ABORT, 'local state reconciliation events are immutable');
        END
        """
    )


def _apply_plan_transaction(
    plan: RepairPlan,
    *,
    before_commit: Callable[[sqlite3.Connection], None] | None = None,
) -> None:
    from storage import ensure_runtime_scope_schema

    connection = sqlite3.connect(plan.database_path, timeout=30.0)
    connection.row_factory = sqlite3.Row
    try:
        connection.execute("PRAGMA busy_timeout = 30000")
        connection.execute("BEGIN IMMEDIATE")
        current_raw_digest = raw_file_sha256(plan.database_path)
        current_state_digest = database_state_sha256(plan.database_path)
        if (
            current_raw_digest != plan.database_raw_file_sha256_before
            or current_state_digest != plan.database_state_sha256_before
        ):
            raise RepairSafetyError("database changed after approved dry-run state")
        source = connection.execute(
            "SELECT * FROM bot_runs WHERE id = ?", (plan.source_run_id,)
        ).fetchone()
        cop = connection.execute(
            """
            SELECT * FROM symbol_runs
            WHERE run_id = ? AND UPPER(symbol) = 'COP'
            """,
            (plan.source_run_id,),
        ).fetchall()
        if source is None or len(cop) != 1:
            raise RepairSafetyError("approved local source snapshot no longer exists")
        if _decimal(cop[0]["quantity"], field="current local COP quantity") != EXPECTED_LOCAL_QUANTITY:
            raise RepairSafetyError("local COP quantity changed after dry run")

        ensure_runtime_scope_schema(connection)
        _ensure_audit_schema(connection)
        if connection.execute(
            "SELECT 1 FROM bot_runs WHERE id = ?", (plan.repaired_run_id,)
        ).fetchone():
            raise RepairSafetyError("predicted repaired run ID is no longer available")

        source_payload = dict(source)
        copies_source_state = plan.outcome == OUTCOME_A
        if plan.outcome not in {OUTCOME_A, OUTCOME_B}:
            raise RepairSafetyError("approved repair outcome is not applicable")
        if copies_source_state:
            run_cash = source_payload["cash"]
            run_buying_power = source_payload["buying_power"]
            run_equity = source_payload["equity"]
            run_last_equity = source_payload["last_equity"]
            run_daily_pnl = source_payload["daily_pnl"]
            run_kill_switch = source_payload["kill_switch_triggered"]
            run_symbol_fingerprint = source_payload.get("symbol_fingerprint")
        else:
            # Outcome B proves only that the inspected legacy state is outside
            # this account's history. It is evidence, never a source of active
            # account balances, configuration identity, or per-symbol state.
            snapshot = plan.authenticated_account_snapshot
            if snapshot is None:
                raise RepairSafetyError(
                    "Outcome B is missing its authenticated account snapshot"
                )
            run_cash = snapshot["cash"]
            run_buying_power = snapshot["buying_power"]
            run_equity = snapshot["equity"]
            run_last_equity = snapshot["last_equity"]
            run_daily_pnl = snapshot["daily_pnl"]
            run_kill_switch = 0
            run_symbol_fingerprint = None
        connection.execute(
            """
            INSERT INTO bot_runs (
                id, timestamp_utc, cash, buying_power, equity, last_equity,
                daily_pnl, kill_switch_triggered, session_id,
                symbol_fingerprint, environment, account_fingerprint
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                plan.repaired_run_id,
                plan.audit_timestamp_utc,
                run_cash,
                run_buying_power,
                run_equity,
                run_last_equity,
                run_daily_pnl,
                run_kill_switch,
                f"repair-{plan.repair_id}",
                run_symbol_fingerprint,
                plan.environment,
                plan.account_fingerprint,
            ),
        )
        rows: Sequence[sqlite3.Row | dict[str, Any]]
        if copies_source_state:
            rows = connection.execute(
                "SELECT * FROM symbol_runs WHERE run_id = ? ORDER BY id",
                (plan.source_run_id,),
            ).fetchall()
        else:
            rows = (
                {
                    "symbol": plan.symbol,
                    "price": None,
                    "sma": None,
                    "ml_probability_up": None,
                    "ml_confidence": None,
                    "ml_training_rows": None,
                    "action": "RECONCILED_FLAT",
                    "holding": 0,
                    "holding_minutes": None,
                    "quantity": 0,
                    "market_value": 0,
                    "avg_entry_price": None,
                    "unrealized_pl": None,
                    "error": None,
                },
            )
        for raw_row in rows:
            row = dict(raw_row)
            is_cop = str(row.get("symbol") or "").strip().upper() == plan.symbol
            connection.execute(
                """
                INSERT INTO symbol_runs (
                    run_id, symbol, price, sma, ml_probability_up, ml_confidence,
                    ml_training_rows, action, holding, holding_minutes, quantity,
                    market_value, avg_entry_price, unrealized_pl, error
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    plan.repaired_run_id,
                    row.get("symbol"),
                    row.get("price"),
                    row.get("sma"),
                    row.get("ml_probability_up"),
                    row.get("ml_confidence"),
                    row.get("ml_training_rows"),
                    "RECONCILED_FLAT" if is_cop else row.get("action"),
                    0 if is_cop else row.get("holding"),
                    None if is_cop else row.get("holding_minutes"),
                    0 if is_cop else row.get("quantity"),
                    0 if is_cop else row.get("market_value"),
                    None if is_cop else row.get("avg_entry_price"),
                    None if is_cop else row.get("unrealized_pl"),
                    None if is_cop else row.get("error"),
                ),
            )

        provenance = json.dumps(
            {
                "repair_id": plan.repair_id,
                "outcome": plan.outcome,
                "source_run_id": plan.source_run_id,
                "repaired_run_id": plan.repaired_run_id,
                "evidence_sha256": plan.evidence_sha256,
                "source_state_copied": copies_source_state,
                "state_source": (
                    "local_source_run"
                    if copies_source_state
                    else "authenticated_broker_account"
                ),
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        existing_binding = connection.execute(
            """
            SELECT * FROM runtime_state_bindings
            WHERE environment = ? AND account_fingerprint = ?
            """,
            (plan.environment, plan.account_fingerprint),
        ).fetchone()
        if existing_binding is None:
            binding_history_cutoff = (
                str(source_payload["timestamp_utc"])
                if copies_source_state
                else plan.audit_timestamp_utc
            )
            connection.execute(
                """
                INSERT INTO runtime_state_bindings (
                    environment, account_fingerprint, provenance,
                    history_cutoff_utc, bound_at_utc
                ) VALUES (?, ?, ?, ?, ?)
                """,
                (
                    plan.environment,
                    plan.account_fingerprint,
                    provenance,
                    binding_history_cutoff,
                    plan.audit_timestamp_utc,
                ),
            )

        audit = plan.audit_record
        connection.execute(
            """
            INSERT INTO local_state_reconciliation_events (
                repair_id, timestamp_utc, environment, account_fingerprint,
                account_id_masked, symbol, previous_quantity, new_quantity,
                outcome, reason, source_run_id, repaired_run_id,
                supporting_broker_order_ids_json,
                supporting_fill_activity_ids_json, evidence_sha256,
                repair_plan_sha256, database_raw_file_sha256_before,
                database_state_sha256_before, backup_raw_file_sha256,
                backup_path, tool_version
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                audit["repair_id"],
                audit["timestamp_utc"],
                audit["environment"],
                audit["account_fingerprint"],
                audit["account_id_masked"],
                audit["symbol"],
                audit["previous_quantity"],
                audit["new_quantity"],
                audit["outcome"],
                audit["reason"],
                audit["source_run_id"],
                audit["repaired_run_id"],
                json.dumps(audit["supporting_broker_order_ids"], sort_keys=True),
                json.dumps(audit["supporting_fill_activity_ids"], sort_keys=True),
                audit["evidence_sha256"],
                audit["repair_plan_sha256"],
                audit["database_raw_file_sha256_before"],
                audit["database_state_sha256_before"],
                audit["backup_raw_file_sha256"],
                audit["backup_path"],
                audit["tool_version"],
            ),
        )
        if before_commit is not None:
            before_commit(connection)
        connection.commit()
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()


def execute_repair(
    *,
    broker: HistoricalRepairBroker,
    database_path: Path,
    configured_paper: bool,
    expected_account_id: str | None,
    symbol: str,
    apply: bool = False,
    repair_id: str | None = None,
    approved_db_raw_sha256: str | None = None,
    approved_db_state_sha256: str | None = None,
    approved_plan_sha256: str | None = None,
    approved_evidence_sha256: str | None = None,
    lock_factory: Callable[[], Any] = ExecutionLock,
    lock_inspector: Callable[[], LockStatus] = inspect_lock,
    backup_root: Path | None = None,
    backup_function: Callable[[Path, Path], Path] = create_verified_database_backup,
    before_commit: Callable[[sqlite3.Connection], None] | None = None,
) -> RepairPlan:
    if not apply:
        return build_repair_plan(
            broker=broker,
            database_path=database_path,
            configured_paper=configured_paper,
            expected_account_id=expected_account_id,
            symbol=symbol,
            repair_id=repair_id,
            lock_inspector=lock_inspector,
            backup_root=backup_root,
        )
    if not all(
        (
            repair_id,
            approved_db_raw_sha256,
            approved_db_state_sha256,
            approved_plan_sha256,
            approved_evidence_sha256,
        )
    ):
        raise RepairSafetyError(
            "--apply requires the repair ID and every approved dry-run digest"
        )
    with lock_factory():
        plan = build_repair_plan(
            broker=broker,
            database_path=database_path,
            configured_paper=configured_paper,
            expected_account_id=expected_account_id,
            symbol=symbol,
            repair_id=repair_id,
            lock_inspector=lock_inspector,
            backup_root=backup_root,
        )
        if plan.database_raw_file_sha256_before != str(
            approved_db_raw_sha256
        ).lower():
            raise RepairSafetyError(
                "database raw file does not match the approved dry-run digest"
            )
        if plan.database_state_sha256_before != str(
            approved_db_state_sha256
        ).lower():
            raise RepairSafetyError(
                "database state does not match the approved dry-run digest"
            )
        if plan.repair_plan_sha256 != str(approved_plan_sha256).lower():
            raise RepairSafetyError(
                "repair plan does not match the approved dry-run digest"
            )
        if plan.evidence_sha256 != str(approved_evidence_sha256).lower():
            raise RepairSafetyError("broker evidence does not match the approved dry-run digest")
        if not plan.application_allowed:
            raise RepairSafetyError("fresh repair checks do not allow application")
        try:
            backup = backup_function(plan.database_path, plan.backup_path)
        except RepairSafetyError:
            raise
        except Exception as exc:
            raise RepairSafetyError(
                f"database backup failed: {type(exc).__name__}"
            ) from exc
        if Path(backup).resolve() != plan.backup_path.resolve():
            raise RepairSafetyError("backup function returned an unexpected path")
        backup_raw_digest = raw_file_sha256(plan.backup_path)
        applied_audit = dict(plan.audit_record)
        applied_audit["backup_raw_file_sha256"] = backup_raw_digest
        plan_with_backup = replace(
            plan,
            audit_record=applied_audit,
            backup_created=True,
            backup_raw_file_sha256=backup_raw_digest,
        )
        try:
            _apply_plan_transaction(plan_with_backup, before_commit=before_commit)
        except RepairSafetyError:
            raise
        except Exception as exc:
            raise RepairSafetyError(
                f"database transaction failed and rolled back: {type(exc).__name__}"
            ) from exc
        return replace(
            plan_with_backup,
            applied=True,
            apply_command=None,
            database_raw_file_sha256_after=raw_file_sha256(plan.database_path),
            database_state_sha256_after=database_state_sha256(plan.database_path),
        )


def _masked_ids(values: Iterable[str]) -> list[str]:
    return [mask_identifier(value) for value in values]


def format_repair_plan(plan: RepairPlan) -> str:
    sanitized_audit = dict(plan.audit_record)
    sanitized_audit["account_fingerprint"] = (
        f"sha256:{plan.account_fingerprint[:12]}…"
    )
    sanitized_audit["supporting_broker_order_ids"] = _masked_ids(
        sanitized_audit["supporting_broker_order_ids"]
    )
    sanitized_audit["supporting_fill_activity_ids"] = _masked_ids(
        sanitized_audit["supporting_fill_activity_ids"]
    )
    sanitized_changes: list[dict[str, Any]] = []
    for change in plan.proposed_changes:
        item = dict(change)
        if "account_fingerprint" in item:
            item["account_fingerprint"] = (
                f"sha256:{plan.account_fingerprint[:12]}…"
            )
        sanitized_changes.append(item)
    checks = {name: passed for name, passed in plan.checks}
    lines = [
        "Mode: " + ("APPLIED" if plan.applied else "DRY_RUN"),
        f"Environment: {plan.environment}",
        f"Account: {plan.account_id_masked}",
        f"Account status: {plan.account_status}",
        f"Symbol: {plan.symbol}",
        f"Current broker quantity: {_decimal_text(plan.broker_quantity)}",
        f"Current local quantity: {_decimal_text(plan.local_quantity)}",
        f"Broker position count: {plan.broker_position_count}",
        f"Open/pending order count: {plan.open_pending_order_count}",
        f"Historical evidence summary: {plan.evidence.summary}",
        "Matched local broker order IDs: "
        + json.dumps(_masked_ids(plan.evidence.matching_order_ids)),
        "Confirmed closing broker order IDs: "
        + json.dumps(_masked_ids(plan.evidence.closing_order_ids)),
        f"Selected outcome: {plan.outcome}",
        f"Outcome reason: {plan.outcome_reason}",
        f"Database: {plan.database_path}",
        "Raw database file SHA-256 before repair: "
        f"{plan.database_raw_file_sha256_before}",
        "SQLite main+WAL state SHA-256 before repair: "
        f"{plan.database_state_sha256_before}",
        f"Canonical repair plan SHA-256: {plan.repair_plan_sha256}",
        f"Broker evidence SHA-256: {plan.evidence_sha256}",
        f"Database backup path: {plan.backup_path}",
        "Backup raw file SHA-256: "
        + (plan.backup_raw_file_sha256 or "NOT_CREATED"),
        "Raw database file SHA-256 after repair: "
        + (plan.database_raw_file_sha256_after or "NOT_APPLIED"),
        "SQLite main+WAL state SHA-256 after repair: "
        + (plan.database_state_sha256_after or "NOT_APPLIED"),
        "Schema changes that would occur: "
        + json.dumps(list(plan.schema_changes), sort_keys=True),
        "Exact proposed changes: "
        + json.dumps(sanitized_changes, sort_keys=True),
        "Exact audit record: "
        + json.dumps(sanitized_audit, sort_keys=True),
        "Safety checks: " + json.dumps(checks, sort_keys=True),
        f"Application allowed: {'yes' if plan.application_allowed else 'no'}",
        "Apply command: " + (plan.apply_command or "not available"),
        plan.verdict if not plan.applied else "LOCAL_STATE_REPAIR_APPLIED",
    ]
    return "\n".join(lines)


__all__ = [
    "BLOCKED_LOCAL_STATE_PROVENANCE_UNKNOWN",
    "EXPECTED_LOCAL_QUANTITY",
    "HISTORY_START_UTC",
    "HistoricalEvidence",
    "LocalRepairState",
    "OUTCOME_A",
    "OUTCOME_B",
    "OUTCOME_C",
    "READY_FOR_HUMAN_APPROVAL_TO_APPLY",
    "REPAIR_SYMBOL",
    "RepairPlan",
    "RepairSafetyError",
    "TOOL_VERSION",
    "build_repair_plan",
    "create_verified_database_backup",
    "database_state_sha256",
    "execute_repair",
    "format_repair_plan",
    "raw_file_sha256",
]

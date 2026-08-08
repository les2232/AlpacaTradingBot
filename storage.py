import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

from tradeos.runtime_paths import require_canonical_database_path


DECISION_CYCLE_CLAIMED = "claimed"
DECISION_CYCLE_RUNNING = "running"
DECISION_CYCLE_COMPLETED = "completed"
DECISION_CYCLE_FAILED = "failed"
DECISION_CYCLE_AMBIGUOUS = "ambiguous"
DECISION_CYCLE_TERMINAL_STATUSES = frozenset(
    {
        DECISION_CYCLE_COMPLETED,
        DECISION_CYCLE_FAILED,
        DECISION_CYCLE_AMBIGUOUS,
    }
)
DECISION_CYCLE_STATUSES = frozenset(
    {
        DECISION_CYCLE_CLAIMED,
        DECISION_CYCLE_RUNNING,
        *DECISION_CYCLE_TERMINAL_STATUSES,
    }
)

ORDER_INTENT_PREPARED = "prepared"
ORDER_INTENT_SUBMITTING = "submitting"
ORDER_INTENT_ACCEPTED = "accepted"
ORDER_INTENT_AMBIGUOUS = "ambiguous"
ORDER_INTENT_TERMINAL = "terminal"
ORDER_TERMINAL_BROKER_STATUSES = frozenset(
    {"canceled", "cancelled", "expired", "filled", "rejected", "replaced"}
)
ORDER_INTENT_STATUSES = frozenset(
    {
        ORDER_INTENT_PREPARED,
        ORDER_INTENT_SUBMITTING,
        ORDER_INTENT_ACCEPTED,
        ORDER_INTENT_AMBIGUOUS,
        ORDER_INTENT_TERMINAL,
    }
)

RUNTIME_ENVIRONMENTS = frozenset({"paper", "live"})


def _normalize_order_filled_quantity(value: Any) -> tuple[Decimal, str]:
    """Return a finite, nonnegative cumulative fill quantity and stable text."""

    try:
        quantity = Decimal(str(value).strip())
    except (InvalidOperation, TypeError, ValueError) as exc:
        raise ValueError("order intent filled quantity is malformed") from exc
    if not quantity.is_finite() or quantity < 0:
        raise ValueError("order intent filled quantity must be finite and nonnegative")
    text = format(quantity, "f")
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    return quantity, text or "0"


def _terminal_broker_status_identity(status: str) -> str:
    """Collapse spelling aliases when comparing sticky terminal outcomes."""

    return "canceled" if status == "cancelled" else status


def _merge_order_intent_broker_state(
    connection: sqlite3.Connection,
    row: sqlite3.Row,
    *,
    broker_order_id: str,
    broker_status: str,
    filled_qty: Any,
    updated_at_utc: str,
) -> None:
    """Monotonically merge one broker observation into a locked intent row."""

    payload = dict(row)
    intent_id = str(payload.get("intent_id", ""))
    current_local_status = str(payload.get("status", "")).strip().lower()
    merge_capable_statuses = {
        ORDER_INTENT_SUBMITTING,
        ORDER_INTENT_AMBIGUOUS,
        ORDER_INTENT_ACCEPTED,
        ORDER_INTENT_TERMINAL,
    }
    if current_local_status not in merge_capable_statuses:
        raise RuntimeError(
            f"order intent in status {current_local_status or 'unknown'} cannot merge broker state"
        )
    if int(payload.get("submission_attempts", 0) or 0) <= 0:
        raise RuntimeError("order intent has no durable submission attempt")

    normalized_order_id = str(broker_order_id or "").strip()
    if not normalized_order_id:
        raise ValueError("broker order id cannot be blank")
    existing_order_id = str(payload.get("broker_order_id") or "").strip()
    if existing_order_id and existing_order_id != normalized_order_id:
        raise RuntimeError("broker order id is already bound to this intent")

    normalized_broker_status = str(broker_status or "").strip().lower()
    if not normalized_broker_status:
        raise ValueError("broker order status cannot be blank")
    incoming_filled, incoming_filled_text = _normalize_order_filled_quantity(
        filled_qty
    )
    current_filled, _ = _normalize_order_filled_quantity(
        payload.get("filled_qty", "0")
    )
    incoming_is_terminal = normalized_broker_status in ORDER_TERMINAL_BROKER_STATUSES
    current_broker_status = str(payload.get("broker_status") or "").strip().lower()

    if current_local_status == ORDER_INTENT_TERMINAL:
        if current_broker_status not in ORDER_TERMINAL_BROKER_STATUSES:
            raise RuntimeError("terminal order intent has no terminal broker status")
        if not incoming_is_terminal:
            # Acceptance/list endpoints can lag a terminal broker observation.
            # Preserve the terminal row exactly instead of regressing it.
            return
        if _terminal_broker_status_identity(
            current_broker_status
        ) != _terminal_broker_status_identity(normalized_broker_status):
            raise RuntimeError("contradictory terminal broker status for order intent")

    if incoming_filled < current_filled:
        raise RuntimeError("order intent filled quantity cannot decrease")

    next_local_status = (
        ORDER_INTENT_TERMINAL if incoming_is_terminal else ORDER_INTENT_ACCEPTED
    )
    cursor = connection.execute(
        """
        UPDATE order_intents
        SET status = ?, broker_order_id = ?, broker_status = ?,
            filled_qty = ?, updated_at_utc = ?, error_category = NULL
        WHERE intent_id = ?
        """,
        (
            next_local_status,
            existing_order_id or normalized_order_id,
            normalized_broker_status,
            incoming_filled_text,
            str(updated_at_utc),
            intent_id,
        ),
    )
    if int(cursor.rowcount or 0) != 1:
        raise RuntimeError("order intent broker-state merge lost its locked row")


def normalize_runtime_scope(
    environment: str,
    account_fingerprint: str,
) -> tuple[str, str]:
    """Validate and normalize the non-secret identity of one broker runtime."""

    normalized_environment = str(environment or "").strip().lower()
    normalized_fingerprint = str(account_fingerprint or "").strip()
    if normalized_environment not in RUNTIME_ENVIRONMENTS:
        raise ValueError("runtime environment must be exactly 'paper' or 'live'")
    if not normalized_fingerprint:
        raise ValueError("runtime account fingerprint cannot be blank")
    return normalized_environment, normalized_fingerprint


def _optional_runtime_scope(
    environment: str | None,
    account_fingerprint: str | None,
) -> tuple[str, str] | None:
    if environment is None and account_fingerprint is None:
        return None
    if environment is None or account_fingerprint is None:
        raise ValueError(
            "environment and account_fingerprint must be supplied together"
        )
    return normalize_runtime_scope(environment, account_fingerprint)


def _table_columns(
    connection: sqlite3.Connection,
    table_name: str,
) -> set[str]:
    return {
        str(row[1])
        for row in connection.execute(f"PRAGMA table_info({table_name})").fetchall()
    }


def ensure_runtime_scope_schema(connection: sqlite3.Connection) -> None:
    """Ensure additive runtime-scope storage inside the caller's transaction.

    This helper deliberately neither begins nor commits a transaction. Repair
    tooling can therefore call it after ``BEGIN IMMEDIATE`` and make its scope
    binding and repair audit one atomic change. Existing operational tables
    must already exist; their legacy rows remain NULL/unbound.
    """

    required_tables = ("bot_runs", "order_history", "order_intents")
    columns = {name: _table_columns(connection, name) for name in required_tables}
    missing_tables = [name for name, names in columns.items() if not names]
    if missing_tables:
        raise RuntimeError(
            "runtime scope migration requires existing tables: "
            + ", ".join(sorted(missing_tables))
        )

    additions = {
        "bot_runs": {
            "environment": "ALTER TABLE bot_runs ADD COLUMN environment TEXT",
            "account_fingerprint": (
                "ALTER TABLE bot_runs ADD COLUMN account_fingerprint TEXT"
            ),
        },
        "order_history": {
            "environment": "ALTER TABLE order_history ADD COLUMN environment TEXT",
            "account_fingerprint": (
                "ALTER TABLE order_history ADD COLUMN account_fingerprint TEXT"
            ),
        },
        "order_intents": {
            "environment": "ALTER TABLE order_intents ADD COLUMN environment TEXT",
        },
    }
    for table_name, table_additions in additions.items():
        for column_name, statement in table_additions.items():
            if column_name not in columns[table_name]:
                connection.execute(statement)

    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS runtime_state_bindings (
            environment TEXT NOT NULL,
            account_fingerprint TEXT NOT NULL,
            provenance TEXT NOT NULL,
            history_cutoff_utc TEXT NOT NULL,
            bound_at_utc TEXT NOT NULL,
            PRIMARY KEY (environment, account_fingerprint),
            CHECK (environment IN ('paper', 'live'))
        )
        """
    )
    connection.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_bot_runs_runtime_scope
        ON bot_runs (environment, account_fingerprint, id DESC)
        """
    )
    connection.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_order_history_runtime_scope
        ON order_history (
            environment,
            account_fingerprint,
            observed_at_utc DESC
        )
        """
    )
    connection.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_order_intents_runtime_scope
        ON order_intents (
            environment,
            account_fingerprint,
            created_at_utc,
            intent_id
        )
        """
    )


def get_runtime_state_binding(
    connection: sqlite3.Connection,
    *,
    environment: str,
    account_fingerprint: str,
) -> dict[str, Any] | None:
    """Read one immutable runtime binding without changing transaction state."""

    normalized_environment, normalized_fingerprint = normalize_runtime_scope(
        environment,
        account_fingerprint,
    )
    row = connection.execute(
        """
        SELECT environment, account_fingerprint, provenance,
               history_cutoff_utc, bound_at_utc
        FROM runtime_state_bindings
        WHERE environment = ? AND account_fingerprint = ?
        """,
        (normalized_environment, normalized_fingerprint),
    ).fetchone()
    if row is None:
        return None
    if isinstance(row, sqlite3.Row):
        return dict(row)
    return {
        "environment": row[0],
        "account_fingerprint": row[1],
        "provenance": row[2],
        "history_cutoff_utc": row[3],
        "bound_at_utc": row[4],
    }


def bind_runtime_state_scope(
    connection: sqlite3.Connection,
    *,
    environment: str,
    account_fingerprint: str,
    provenance: str,
    history_cutoff_utc: str,
    bound_at_utc: str,
) -> bool:
    """Immutably bind a verified broker scope in the caller's transaction.

    Returns ``True`` when this call inserted the binding and ``False`` for an
    exact idempotent replay. A conflicting attempt fails closed.
    """

    ensure_runtime_scope_schema(connection)
    normalized_environment, normalized_fingerprint = normalize_runtime_scope(
        environment,
        account_fingerprint,
    )
    normalized_provenance = str(provenance or "").strip()
    normalized_cutoff = str(history_cutoff_utc or "").strip()
    normalized_bound_at = str(bound_at_utc or "").strip()
    if not normalized_provenance:
        raise ValueError("runtime scope binding provenance cannot be blank")
    if not normalized_cutoff:
        raise ValueError("runtime scope binding history cutoff cannot be blank")
    if not normalized_bound_at:
        raise ValueError("runtime scope binding timestamp cannot be blank")

    cursor = connection.execute(
        """
        INSERT OR IGNORE INTO runtime_state_bindings (
            environment,
            account_fingerprint,
            provenance,
            history_cutoff_utc,
            bound_at_utc
        ) VALUES (?, ?, ?, ?, ?)
        """,
        (
            normalized_environment,
            normalized_fingerprint,
            normalized_provenance,
            normalized_cutoff,
            normalized_bound_at,
        ),
    )
    existing = get_runtime_state_binding(
        connection,
        environment=normalized_environment,
        account_fingerprint=normalized_fingerprint,
    )
    expected = {
        "environment": normalized_environment,
        "account_fingerprint": normalized_fingerprint,
        "provenance": normalized_provenance,
        "history_cutoff_utc": normalized_cutoff,
        "bound_at_utc": normalized_bound_at,
    }
    if existing != expected:
        raise RuntimeError("runtime state scope is already bound with different provenance")
    return int(cursor.rowcount or 0) == 1


@dataclass(frozen=True)
class DecisionCycleClaim:
    decision_ts: str
    claimed: bool
    reason: str
    status: str | None
    owner_id: str | None = None
    latest_decision_ts: str | None = None


def normalize_decision_cycle_id(value: datetime | str) -> str:
    """Return the canonical UTC identity for one completed-bar cycle."""

    parsed = value if isinstance(value, datetime) else datetime.fromisoformat(value)
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError("decision cycle timestamp must be timezone-aware")
    return (
        parsed.astimezone(timezone.utc)
        .replace(microsecond=0)
        .isoformat(timespec="seconds")
    )


class BotStorage:
    def __init__(self, db_path: str | Path, *, read_only: bool = False) -> None:
        self.db_path = require_canonical_database_path(db_path)
        self.read_only = read_only is True
        if self.read_only:
            if not self.db_path.is_file():
                raise FileNotFoundError(
                    f"read-only database does not exist: {self.db_path}"
                )
        else:
            self._init_db()

    def _connect(self) -> sqlite3.Connection:
        if self.read_only:
            connection = sqlite3.connect(
                f"file:{self.db_path.as_posix()}?mode=ro",
                uri=True,
                timeout=30.0,
            )
        else:
            connection = sqlite3.connect(self.db_path, timeout=30.0)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA busy_timeout = 30000")
        if self.read_only:
            connection.execute("PRAGMA query_only=ON")
        return connection

    def _init_db(self) -> None:
        with self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS bot_runs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT,
                    symbol_fingerprint TEXT,
                    environment TEXT,
                    account_fingerprint TEXT,
                    timestamp_utc TEXT NOT NULL,
                    cash REAL NOT NULL,
                    buying_power REAL NOT NULL,
                    equity REAL NOT NULL,
                    last_equity REAL NOT NULL,
                    daily_pnl REAL NOT NULL,
                    kill_switch_triggered INTEGER NOT NULL
                );

                CREATE TABLE IF NOT EXISTS symbol_runs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id INTEGER NOT NULL,
                    symbol TEXT NOT NULL,
                    price REAL,
                    sma REAL,
                    ml_probability_up REAL,
                    ml_confidence REAL,
                    ml_training_rows INTEGER,
                    action TEXT NOT NULL,
                    holding INTEGER NOT NULL,
                    holding_minutes REAL,
                    quantity REAL NOT NULL,
                    market_value REAL NOT NULL,
                    avg_entry_price REAL,
                    unrealized_pl REAL,
                    error TEXT,
                    FOREIGN KEY(run_id) REFERENCES bot_runs(id)
                );

                CREATE TABLE IF NOT EXISTS order_history (
                    order_id TEXT PRIMARY KEY,
                    session_id TEXT,
                    environment TEXT,
                    account_fingerprint TEXT,
                    observed_at_utc TEXT NOT NULL,
                    submitted_at TEXT,
                    filled_at TEXT,
                    symbol TEXT NOT NULL,
                    side TEXT NOT NULL,
                    status TEXT NOT NULL,
                    qty REAL,
                    filled_qty REAL,
                    filled_avg_price REAL,
                    notional REAL,
                    client_order_id TEXT
                );

                CREATE TABLE IF NOT EXISTS processed_decisions (
                    decision_ts TEXT PRIMARY KEY,
                    claimed_at_utc TEXT NOT NULL,
                    owner_id TEXT NOT NULL DEFAULT 'legacy',
                    status TEXT NOT NULL DEFAULT 'ambiguous',
                    updated_at_utc TEXT NOT NULL DEFAULT '',
                    finished_at_utc TEXT,
                    reason TEXT
                );

                CREATE TABLE IF NOT EXISTS processed_order_fills (
                    order_id TEXT NOT NULL,
                    filled_qty REAL NOT NULL,
                    claimed_at_utc TEXT NOT NULL,
                    PRIMARY KEY (order_id, filled_qty)
                );

                CREATE TABLE IF NOT EXISTS order_intents (
                    intent_id TEXT PRIMARY KEY,
                    client_order_id TEXT NOT NULL UNIQUE,
                    environment TEXT,
                    account_fingerprint TEXT NOT NULL,
                    decision_ts TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    side TEXT NOT NULL,
                    intent_kind TEXT NOT NULL,
                    requested_qty TEXT NOT NULL,
                    final_qty TEXT NOT NULL,
                    status TEXT NOT NULL,
                    broker_order_id TEXT,
                    broker_status TEXT,
                    filled_qty TEXT NOT NULL DEFAULT '0',
                    submission_attempts INTEGER NOT NULL DEFAULT 0,
                    error_category TEXT,
                    created_at_utc TEXT NOT NULL,
                    updated_at_utc TEXT NOT NULL
                );
                """
            )
            # Serialize additive schema inspection and migration. Without this
            # write lock, two first-time owners opening the same legacy
            # database can both observe a missing column and race the ALTER.
            connection.execute("BEGIN IMMEDIATE")
            ensure_runtime_scope_schema(connection)
            symbol_columns = {
                row["name"]
                for row in connection.execute("PRAGMA table_info(symbol_runs)").fetchall()
            }
            bot_run_columns = {
                row["name"]
                for row in connection.execute("PRAGMA table_info(bot_runs)").fetchall()
            }
            bot_run_migrations = {
                "session_id": "ALTER TABLE bot_runs ADD COLUMN session_id TEXT",
                "symbol_fingerprint": "ALTER TABLE bot_runs ADD COLUMN symbol_fingerprint TEXT",
            }
            for column_name, statement in bot_run_migrations.items():
                if column_name not in bot_run_columns:
                    connection.execute(statement)
            column_migrations = {
                "ml_probability_up": "ALTER TABLE symbol_runs ADD COLUMN ml_probability_up REAL",
                "ml_confidence": "ALTER TABLE symbol_runs ADD COLUMN ml_confidence REAL",
                "ml_training_rows": "ALTER TABLE symbol_runs ADD COLUMN ml_training_rows INTEGER",
                "holding_minutes": "ALTER TABLE symbol_runs ADD COLUMN holding_minutes REAL",
                "avg_entry_price": "ALTER TABLE symbol_runs ADD COLUMN avg_entry_price REAL",
                "unrealized_pl": "ALTER TABLE symbol_runs ADD COLUMN unrealized_pl REAL",
            }
            for column_name, statement in column_migrations.items():
                if column_name not in symbol_columns:
                    connection.execute(statement)

            order_columns = {
                row["name"]
                for row in connection.execute("PRAGMA table_info(order_history)").fetchall()
            }
            if "session_id" not in order_columns:
                connection.execute("ALTER TABLE order_history ADD COLUMN session_id TEXT")
            if "filled_at" not in order_columns:
                connection.execute("ALTER TABLE order_history ADD COLUMN filled_at TEXT")
            if "client_order_id" not in order_columns:
                connection.execute("ALTER TABLE order_history ADD COLUMN client_order_id TEXT")

            processed_decision_columns = {
                row["name"]
                for row in connection.execute(
                    "PRAGMA table_info(processed_decisions)"
                ).fetchall()
            }
            processed_decision_migrations = {
                "owner_id": (
                    "ALTER TABLE processed_decisions "
                    "ADD COLUMN owner_id TEXT NOT NULL DEFAULT 'legacy'"
                ),
                "status": (
                    "ALTER TABLE processed_decisions "
                    "ADD COLUMN status TEXT NOT NULL DEFAULT 'ambiguous'"
                ),
                "updated_at_utc": (
                    "ALTER TABLE processed_decisions "
                    "ADD COLUMN updated_at_utc TEXT NOT NULL DEFAULT ''"
                ),
                "finished_at_utc": (
                    "ALTER TABLE processed_decisions "
                    "ADD COLUMN finished_at_utc TEXT"
                ),
                "reason": "ALTER TABLE processed_decisions ADD COLUMN reason TEXT",
            }
            for column_name, statement in processed_decision_migrations.items():
                if column_name not in processed_decision_columns:
                    connection.execute(statement)
            connection.execute(
                """
                UPDATE processed_decisions
                SET updated_at_utc = claimed_at_utc
                WHERE updated_at_utc = ''
                """
            )

            intent_columns = {
                row["name"]
                for row in connection.execute("PRAGMA table_info(order_intents)").fetchall()
            }
            intent_migrations = {
                "filled_qty": (
                    "ALTER TABLE order_intents "
                    "ADD COLUMN filled_qty TEXT NOT NULL DEFAULT '0'"
                ),
                "submission_attempts": (
                    "ALTER TABLE order_intents "
                    "ADD COLUMN submission_attempts INTEGER NOT NULL DEFAULT 0"
                ),
                "error_category": "ALTER TABLE order_intents ADD COLUMN error_category TEXT",
            }
            for column_name, statement in intent_migrations.items():
                if column_name not in intent_columns:
                    connection.execute(statement)
            connection.execute(
                """
                UPDATE processed_decisions
                SET finished_at_utc = claimed_at_utc,
                    reason = COALESCE(reason, 'legacy_claim_outcome_unknown')
                WHERE owner_id = 'legacy'
                  AND status = 'ambiguous'
                  AND finished_at_utc IS NULL
                """
            )

    def get_runtime_state_binding(
        self,
        *,
        environment: str,
        account_fingerprint: str,
    ) -> dict[str, Any] | None:
        with self._connect() as connection:
            return get_runtime_state_binding(
                connection,
                environment=environment,
                account_fingerprint=account_fingerprint,
            )

    def bind_runtime_state_scope(
        self,
        *,
        environment: str,
        account_fingerprint: str,
        provenance: str,
        history_cutoff_utc: str,
        bound_at_utc: str,
    ) -> bool:
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            return bind_runtime_state_scope(
                connection,
                environment=environment,
                account_fingerprint=account_fingerprint,
                provenance=provenance,
                history_cutoff_utc=history_cutoff_utc,
                bound_at_utc=bound_at_utc,
            )

    def _require_runtime_state_binding(
        self,
        *,
        environment: str,
        account_fingerprint: str,
    ) -> tuple[str, str]:
        normalized_environment, normalized_fingerprint = normalize_runtime_scope(
            environment,
            account_fingerprint,
        )
        if self.get_runtime_state_binding(
            environment=normalized_environment,
            account_fingerprint=normalized_fingerprint,
        ) is None:
            raise RuntimeError(
                "runtime state scope is not bound; refusing to read or write scoped state"
            )
        return normalized_environment, normalized_fingerprint

    def get_latest_unbound_open_positions(self) -> list[dict[str, Any]]:
        """Return nonzero state from the newest legacy/unbound snapshot only."""

        with self._connect() as connection:
            latest_run = connection.execute(
                """
                SELECT id, timestamp_utc, session_id
                FROM bot_runs
                WHERE environment IS NULL OR account_fingerprint IS NULL
                ORDER BY id DESC
                LIMIT 1
                """
            ).fetchone()
            if latest_run is None:
                return []
            rows = connection.execute(
                """
                SELECT id AS symbol_run_id, run_id, symbol, holding, quantity,
                       action, market_value, avg_entry_price, unrealized_pl
                FROM symbol_runs
                WHERE run_id = ? AND COALESCE(quantity, 0) != 0
                ORDER BY symbol, id
                """,
                (latest_run["id"],),
            ).fetchall()
        return [dict(row) for row in rows]

    def claim_decision_cycle(
        self,
        decision_ts: datetime | str,
        *,
        owner_id: str,
        claimed_at_utc: str,
    ) -> DecisionCycleClaim:
        """
        Atomically claim one global multi-symbol decision cycle.

        The canonical UTC decision timestamp is the cycle identity. Claims are
        strictly monotonic within one database: an unseen timestamp older than
        the durable high-water mark is rejected and is never inserted.
        """

        normalized_decision_ts = normalize_decision_cycle_id(decision_ts)
        requested_time = datetime.fromisoformat(normalized_decision_ts)
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            existing_rows = connection.execute(
                """
                SELECT decision_ts, owner_id, status
                FROM processed_decisions
                """
            ).fetchall()

            latest_time: datetime | None = None
            latest_decision_ts: str | None = None
            for row in existing_rows:
                persisted_decision_ts = str(row["decision_ts"])
                try:
                    persisted_time = datetime.fromisoformat(
                        normalize_decision_cycle_id(persisted_decision_ts)
                    )
                except (TypeError, ValueError):
                    return DecisionCycleClaim(
                        decision_ts=normalized_decision_ts,
                        claimed=False,
                        reason="invalid_persisted_cycle",
                        status=DECISION_CYCLE_AMBIGUOUS,
                        owner_id=str(row["owner_id"] or "") or None,
                        latest_decision_ts=persisted_decision_ts,
                    )
                if persisted_time == requested_time:
                    return DecisionCycleClaim(
                        decision_ts=normalized_decision_ts,
                        claimed=False,
                        reason="duplicate_cycle",
                        status=str(row["status"]),
                        owner_id=str(row["owner_id"] or "") or None,
                        latest_decision_ts=persisted_decision_ts,
                    )
                if latest_time is None or persisted_time > latest_time:
                    latest_time = persisted_time
                    latest_decision_ts = persisted_decision_ts

            if latest_time is not None and requested_time < latest_time:
                return DecisionCycleClaim(
                    decision_ts=normalized_decision_ts,
                    claimed=False,
                    reason="out_of_order_cycle",
                    status=None,
                    latest_decision_ts=latest_decision_ts,
                )

            connection.execute(
                """
                INSERT INTO processed_decisions (
                    decision_ts,
                    claimed_at_utc,
                    owner_id,
                    status,
                    updated_at_utc,
                    finished_at_utc,
                    reason
                ) VALUES (?, ?, ?, ?, ?, NULL, ?)
                """,
                (
                    normalized_decision_ts,
                    claimed_at_utc,
                    owner_id,
                    DECISION_CYCLE_CLAIMED,
                    claimed_at_utc,
                    "cycle_claimed",
                ),
            )
            return DecisionCycleClaim(
                decision_ts=normalized_decision_ts,
                claimed=True,
                reason="cycle_claimed",
                status=DECISION_CYCLE_CLAIMED,
                owner_id=owner_id,
                latest_decision_ts=latest_decision_ts,
            )

    def transition_decision_cycle(
        self,
        decision_ts: datetime | str,
        *,
        owner_id: str,
        status: str,
        updated_at_utc: str,
        reason: str,
    ) -> bool:
        """Compare-and-set a claimed cycle to its next durable lifecycle state."""

        if status not in DECISION_CYCLE_STATUSES:
            raise ValueError(f"unsupported decision cycle status: {status}")
        allowed_previous = {
            DECISION_CYCLE_RUNNING: (DECISION_CYCLE_CLAIMED,),
            DECISION_CYCLE_COMPLETED: (
                DECISION_CYCLE_CLAIMED,
                DECISION_CYCLE_RUNNING,
            ),
            DECISION_CYCLE_FAILED: (
                DECISION_CYCLE_CLAIMED,
                DECISION_CYCLE_RUNNING,
            ),
            DECISION_CYCLE_AMBIGUOUS: (
                DECISION_CYCLE_CLAIMED,
                DECISION_CYCLE_RUNNING,
            ),
        }.get(status)
        if allowed_previous is None:
            raise ValueError(
                "decision cycles can only transition out of the claimed state"
            )

        normalized_decision_ts = normalize_decision_cycle_id(decision_ts)
        placeholders = ", ".join("?" for _ in allowed_previous)
        finished_at_utc = (
            updated_at_utc
            if status in DECISION_CYCLE_TERMINAL_STATUSES
            else None
        )
        with self._connect() as connection:
            cursor = connection.execute(
                f"""
                UPDATE processed_decisions
                SET status = ?,
                    updated_at_utc = ?,
                    finished_at_utc = ?,
                    reason = ?
                WHERE decision_ts = ?
                  AND owner_id = ?
                  AND status IN ({placeholders})
                """,
                (
                    status,
                    updated_at_utc,
                    finished_at_utc,
                    reason,
                    normalized_decision_ts,
                    owner_id,
                    *allowed_previous,
                ),
            )
            return int(cursor.rowcount or 0) == 1

    def get_decision_cycle(
        self,
        decision_ts: datetime | str,
    ) -> dict[str, Any] | None:
        normalized_decision_ts = normalize_decision_cycle_id(decision_ts)
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT
                    decision_ts,
                    claimed_at_utc,
                    owner_id,
                    status,
                    updated_at_utc,
                    finished_at_utc,
                    reason
                FROM processed_decisions
                WHERE decision_ts = ?
                """,
                (normalized_decision_ts,),
            ).fetchone()
        return dict(row) if row is not None else None

    def claim_decision_timestamp(self, decision_ts: str, claimed_at_utc: str) -> bool:
        """Route the legacy method name through durable cycle authority."""

        return self.claim_decision_cycle(
            decision_ts,
            owner_id="legacy_api",
            claimed_at_utc=claimed_at_utc,
        ).claimed

    def claim_order_fill(self, order_id: str, filled_qty: float, claimed_at_utc: str) -> bool:
        normalized_qty = round(float(filled_qty), 6)
        with self._connect() as connection:
            cursor = connection.execute(
                """
                INSERT OR IGNORE INTO processed_order_fills (order_id, filled_qty, claimed_at_utc)
                VALUES (?, ?, ?)
                """,
                (order_id, normalized_qty, claimed_at_utc),
            )
            return int(cursor.rowcount or 0) == 1

    def reserve_order_intent(
        self,
        *,
        intent_id: str,
        client_order_id: str,
        account_fingerprint: str,
        environment: str | None = None,
        decision_ts: str,
        symbol: str,
        side: str,
        intent_kind: str,
        requested_qty: str,
        final_qty: str,
        created_at_utc: str,
    ) -> tuple[dict[str, Any], bool]:
        """Durably reserve one immutable logical order intent.

        The insert and subsequent read run under one SQLite write transaction so
        concurrent processes cannot both become the creator.  A repeated
        intent may be inspected, but conflicting immutable fields fail closed.
        """

        normalized_environment: str | None = None
        normalized_fingerprint = str(account_fingerprint).strip()
        if environment is not None:
            normalized_environment, normalized_fingerprint = (
                self._require_runtime_state_binding(
                    environment=environment,
                    account_fingerprint=account_fingerprint,
                )
            )
        normalized_symbol = str(symbol).strip().upper()
        normalized_side = str(side).strip().lower()
        immutable = {
            "intent_id": str(intent_id),
            "client_order_id": str(client_order_id),
            "environment": normalized_environment,
            "account_fingerprint": normalized_fingerprint,
            "decision_ts": str(decision_ts),
            "symbol": normalized_symbol,
            "side": normalized_side,
            "intent_kind": str(intent_kind),
            "requested_qty": str(requested_qty),
            "final_qty": str(final_qty),
        }
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            cursor = connection.execute(
                """
                INSERT OR IGNORE INTO order_intents (
                    intent_id, client_order_id, environment, account_fingerprint,
                    decision_ts, symbol, side, intent_kind,
                    requested_qty, final_qty, status,
                    created_at_utc, updated_at_utc
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    immutable["intent_id"],
                    immutable["client_order_id"],
                    immutable["environment"],
                    immutable["account_fingerprint"],
                    immutable["decision_ts"],
                    immutable["symbol"],
                    immutable["side"],
                    immutable["intent_kind"],
                    immutable["requested_qty"],
                    immutable["final_qty"],
                    ORDER_INTENT_PREPARED,
                    created_at_utc,
                    created_at_utc,
                ),
            )
            row = connection.execute(
                "SELECT * FROM order_intents WHERE intent_id = ?",
                (immutable["intent_id"],),
            ).fetchone()
            if row is None:
                conflicting = connection.execute(
                    "SELECT * FROM order_intents WHERE client_order_id = ?",
                    (immutable["client_order_id"],),
                ).fetchone()
                if conflicting is not None:
                    raise RuntimeError("client_order_id is already bound to another intent")
                raise RuntimeError("order intent reservation could not be read back")
            payload = dict(row)
            for key, expected in immutable.items():
                if key == "environment":
                    if payload.get(key) != expected:
                        raise RuntimeError(
                            "order intent conflict for immutable field environment"
                        )
                    continue
                if str(payload.get(key, "")) != expected:
                    raise RuntimeError(f"order intent conflict for immutable field {key}")
            return payload, int(cursor.rowcount or 0) == 1

    def mark_order_intent_submitting(self, intent_id: str, *, updated_at_utc: str) -> bool:
        """Record the point after which any outcome must be treated as uncertain."""

        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            cursor = connection.execute(
                """
                UPDATE order_intents
                SET status = ?,
                    submission_attempts = submission_attempts + 1,
                    updated_at_utc = ?,
                    error_category = NULL
                WHERE intent_id = ?
                  AND status = ?
                  AND submission_attempts = 0
                """,
                (
                    ORDER_INTENT_SUBMITTING,
                    updated_at_utc,
                    intent_id,
                    ORDER_INTENT_PREPARED,
                ),
            )
            return int(cursor.rowcount or 0) == 1

    def mark_order_intent_accepted(
        self,
        intent_id: str,
        *,
        broker_order_id: str,
        broker_status: str,
        filled_qty: str,
        updated_at_utc: str,
    ) -> None:
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT * FROM order_intents WHERE intent_id = ?",
                (intent_id,),
            ).fetchone()
            if row is None:
                raise RuntimeError("unknown order intent cannot be marked accepted")
            _merge_order_intent_broker_state(
                connection,
                row,
                broker_order_id=broker_order_id,
                broker_status=broker_status,
                filled_qty=filled_qty,
                updated_at_utc=updated_at_utc,
            )

    def mark_order_intent_ambiguous(
        self,
        intent_id: str,
        *,
        error_category: str,
        updated_at_utc: str,
    ) -> None:
        with self._connect() as connection:
            cursor = connection.execute(
                """
                UPDATE order_intents
                SET status = ?, error_category = ?, updated_at_utc = ?
                WHERE intent_id = ?
                  AND status IN (?, ?)
                """,
                (
                    ORDER_INTENT_AMBIGUOUS,
                    str(error_category),
                    updated_at_utc,
                    intent_id,
                    ORDER_INTENT_SUBMITTING,
                    ORDER_INTENT_AMBIGUOUS,
                ),
            )
            if int(cursor.rowcount or 0) != 1:
                raise RuntimeError("order intent is not in an ambiguous-capable state")

    def reconcile_order_intent(
        self,
        client_order_id: str,
        *,
        broker_order_id: str,
        broker_status: str,
        filled_qty: str,
        updated_at_utc: str,
        environment: str | None = None,
        account_fingerprint: str | None = None,
    ) -> bool:
        scope = _optional_runtime_scope(environment, account_fingerprint)
        if scope is not None:
            scope = self._require_runtime_state_binding(
                environment=scope[0],
                account_fingerprint=scope[1],
            )
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            if scope is None:
                row = connection.execute(
                    "SELECT * FROM order_intents WHERE client_order_id = ?",
                    (client_order_id,),
                ).fetchone()
            else:
                row = connection.execute(
                    """
                    SELECT * FROM order_intents
                    WHERE client_order_id = ?
                      AND environment = ?
                      AND account_fingerprint = ?
                    """,
                    (
                        client_order_id,
                        scope[0],
                        scope[1],
                    ),
                ).fetchone()
            if row is None:
                return False
            _merge_order_intent_broker_state(
                connection,
                row,
                broker_order_id=broker_order_id,
                broker_status=broker_status,
                filled_qty=filled_qty,
                updated_at_utc=updated_at_utc,
            )
            return True

    def get_order_intent(self, intent_id: str) -> dict[str, Any] | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM order_intents WHERE intent_id = ?",
                (intent_id,),
            ).fetchone()
        return dict(row) if row is not None else None

    def get_order_intent_by_client_id(self, client_order_id: str) -> dict[str, Any] | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM order_intents WHERE client_order_id = ?",
                (client_order_id,),
            ).fetchone()
        return dict(row) if row is not None else None

    def list_order_intents(
        self,
        *,
        environment: str | None = None,
        account_fingerprint: str | None = None,
    ) -> list[dict[str, Any]]:
        scope = _optional_runtime_scope(environment, account_fingerprint)
        if scope is not None:
            scope = self._require_runtime_state_binding(
                environment=scope[0],
                account_fingerprint=scope[1],
            )
        with self._connect() as connection:
            if scope is None:
                rows = connection.execute(
                    "SELECT * FROM order_intents ORDER BY created_at_utc, intent_id"
                ).fetchall()
            else:
                rows = connection.execute(
                    """
                    SELECT *
                    FROM order_intents
                    WHERE environment = ? AND account_fingerprint = ?
                    ORDER BY created_at_utc, intent_id
                    """,
                    scope,
                ).fetchall()
        return [dict(row) for row in rows]

    def save_snapshot(
        self,
        snapshot: Any,
        orders: list[Any],
        *,
        session_id: str | None = None,
        symbol_fingerprint: str | None = None,
        environment: str | None = None,
        account_fingerprint: str | None = None,
    ) -> int:
        scope = _optional_runtime_scope(environment, account_fingerprint)
        if scope is not None:
            scope = self._require_runtime_state_binding(
                environment=scope[0],
                account_fingerprint=scope[1],
            )
        stored_environment = scope[0] if scope is not None else None
        stored_fingerprint = scope[1] if scope is not None else None
        with self._connect() as connection:
            cursor = connection.execute(
                """
                INSERT INTO bot_runs (
                    session_id, symbol_fingerprint, environment,
                    account_fingerprint, timestamp_utc, cash, buying_power,
                    equity, last_equity, daily_pnl, kill_switch_triggered
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    session_id,
                    symbol_fingerprint,
                    stored_environment,
                    stored_fingerprint,
                    snapshot.timestamp_utc,
                    snapshot.cash,
                    snapshot.buying_power,
                    snapshot.equity,
                    snapshot.last_equity,
                    snapshot.daily_pnl,
                    int(snapshot.kill_switch_triggered),
                ),
            )
            run_id = int(cursor.lastrowid)

            connection.executemany(
                """
                INSERT INTO symbol_runs (
                    run_id, symbol, price, sma, ml_probability_up, ml_confidence, ml_training_rows, action, holding, holding_minutes, quantity, market_value, avg_entry_price, unrealized_pl, error
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        run_id,
                        item.symbol,
                        item.price,
                        item.sma,
                        item.ml_probability_up,
                        item.ml_confidence,
                        item.ml_training_rows,
                        item.action,
                        int(item.holding),
                        item.holding_minutes,
                        item.quantity,
                        item.market_value,
                        float(getattr(snapshot.positions.get(item.symbol), "avg_entry_price", 0.0))
                        if snapshot.positions.get(item.symbol) is not None and getattr(snapshot.positions.get(item.symbol), "avg_entry_price", None) is not None
                        else None,
                        float(getattr(snapshot.positions.get(item.symbol), "unrealized_pl", 0.0))
                        if snapshot.positions.get(item.symbol) is not None and getattr(snapshot.positions.get(item.symbol), "unrealized_pl", None) is not None
                        else None,
                        item.error,
                    )
                    for item in snapshot.symbols
                ],
            )

            connection.executemany(
                """
                INSERT INTO order_history (
                    order_id, session_id, environment, account_fingerprint,
                    observed_at_utc, submitted_at, filled_at, symbol, side,
                    status, qty, filled_qty, filled_avg_price, notional,
                    client_order_id
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(order_id) DO UPDATE SET
                    session_id=excluded.session_id,
                    environment=excluded.environment,
                    account_fingerprint=excluded.account_fingerprint,
                    observed_at_utc=excluded.observed_at_utc,
                    submitted_at=excluded.submitted_at,
                    filled_at=excluded.filled_at,
                    symbol=excluded.symbol,
                    side=excluded.side,
                    status=excluded.status,
                    qty=excluded.qty,
                    filled_qty=excluded.filled_qty,
                    filled_avg_price=excluded.filled_avg_price,
                    notional=excluded.notional,
                    client_order_id=excluded.client_order_id
                """,
                [
                    (
                        order.order_id,
                        session_id,
                        stored_environment,
                        stored_fingerprint,
                        snapshot.timestamp_utc,
                        order.submitted_at,
                        getattr(order, "filled_at", None),
                        order.symbol,
                        order.side,
                        order.status,
                        float(order.qty) if order.qty is not None else None,
                        float(order.filled_qty) if order.filled_qty is not None else None,
                        order.filled_avg_price,
                        order.notional,
                        getattr(order, "client_order_id", None),
                    )
                    for order in orders
                    if order.order_id
                ],
            )

            return run_id

    def get_run_history(self, limit: int = 200, session_id: str | None = None) -> list[dict[str, Any]]:
        with self._connect() as connection:
            if session_id is None:
                rows = connection.execute(
                    """
                    SELECT session_id, symbol_fingerprint, timestamp_utc, cash, buying_power, equity, last_equity, daily_pnl, kill_switch_triggered
                    FROM bot_runs
                    ORDER BY id DESC
                    LIMIT ?
                    """,
                    (limit,),
                ).fetchall()
            else:
                rows = connection.execute(
                    """
                    SELECT session_id, symbol_fingerprint, timestamp_utc, cash, buying_power, equity, last_equity, daily_pnl, kill_switch_triggered
                    FROM bot_runs
                    WHERE session_id = ?
                    ORDER BY id DESC
                    LIMIT ?
                    """,
                    (session_id, limit),
                ).fetchall()
        return [dict(row) for row in reversed(rows)]

    def get_latest_run(
        self,
        session_id: str | None = None,
        *,
        environment: str | None = None,
        account_fingerprint: str | None = None,
    ) -> dict[str, Any] | None:
        scope = _optional_runtime_scope(environment, account_fingerprint)
        if scope is not None:
            scope = self._require_runtime_state_binding(
                environment=scope[0],
                account_fingerprint=scope[1],
            )
        with self._connect() as connection:
            if scope is None and session_id is None:
                row = connection.execute(
                    """
                    SELECT id, session_id, symbol_fingerprint, environment,
                           account_fingerprint, timestamp_utc, cash,
                           buying_power, equity, last_equity, daily_pnl,
                           kill_switch_triggered
                    FROM bot_runs
                    ORDER BY id DESC
                    LIMIT 1
                    """
                ).fetchone()
            elif scope is None:
                row = connection.execute(
                    """
                    SELECT id, session_id, symbol_fingerprint, environment,
                           account_fingerprint, timestamp_utc, cash,
                           buying_power, equity, last_equity, daily_pnl,
                           kill_switch_triggered
                    FROM bot_runs
                    WHERE session_id = ?
                    ORDER BY id DESC
                    LIMIT 1
                    """,
                    (session_id,),
                ).fetchone()
            elif session_id is None:
                row = connection.execute(
                    """
                    SELECT id, session_id, symbol_fingerprint, environment,
                           account_fingerprint, timestamp_utc, cash,
                           buying_power, equity, last_equity, daily_pnl,
                           kill_switch_triggered
                    FROM bot_runs
                    WHERE environment = ? AND account_fingerprint = ?
                    ORDER BY id DESC
                    LIMIT 1
                    """,
                    scope,
                ).fetchone()
            else:
                row = connection.execute(
                    """
                    SELECT id, session_id, symbol_fingerprint, environment,
                           account_fingerprint, timestamp_utc, cash,
                           buying_power, equity, last_equity, daily_pnl,
                           kill_switch_triggered
                    FROM bot_runs
                    WHERE session_id = ?
                      AND environment = ?
                      AND account_fingerprint = ?
                    ORDER BY id DESC
                    LIMIT 1
                    """,
                    (session_id, *scope),
                ).fetchone()
        return dict(row) if row is not None else None

    def get_latest_symbol_snapshot(
        self,
        session_id: str | None = None,
        *,
        environment: str | None = None,
        account_fingerprint: str | None = None,
    ) -> list[dict[str, Any]]:
        latest_run = self.get_latest_run(
            session_id=session_id,
            environment=environment,
            account_fingerprint=account_fingerprint,
        )
        if latest_run is None:
            return []

        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT
                    id AS symbol_run_id,
                    run_id,
                    symbol,
                    price,
                    sma,
                    ml_probability_up,
                    ml_confidence,
                    ml_training_rows,
                    action,
                    holding,
                    holding_minutes,
                    quantity,
                    market_value,
                    avg_entry_price,
                    unrealized_pl,
                    error
                FROM symbol_runs
                WHERE run_id = ?
                ORDER BY
                    holding DESC,
                    market_value DESC,
                    symbol ASC
                """,
                (latest_run["id"],),
            ).fetchall()
        return [dict(row) for row in rows]

    def get_symbol_history(self, symbol: str, limit: int = 200, session_id: str | None = None) -> list[dict[str, Any]]:
        with self._connect() as connection:
            if session_id is None:
                rows = connection.execute(
                    """
                    SELECT
                        b.timestamp_utc,
                        b.session_id,
                        s.symbol,
                        s.price,
                        s.sma,
                        s.ml_probability_up,
                        s.ml_confidence,
                        s.ml_training_rows,
                        s.action,
                        s.holding,
                        s.holding_minutes,
                        s.quantity,
                        s.market_value,
                        s.avg_entry_price,
                        s.unrealized_pl,
                        s.error
                    FROM symbol_runs s
                    JOIN bot_runs b ON b.id = s.run_id
                    WHERE s.symbol = ?
                    ORDER BY s.id DESC
                    LIMIT ?
                    """,
                    (symbol, limit),
                ).fetchall()
            else:
                rows = connection.execute(
                    """
                    SELECT
                        b.timestamp_utc,
                        b.session_id,
                        s.symbol,
                        s.price,
                        s.sma,
                        s.ml_probability_up,
                        s.ml_confidence,
                        s.ml_training_rows,
                        s.action,
                        s.holding,
                        s.holding_minutes,
                        s.quantity,
                        s.market_value,
                        s.avg_entry_price,
                        s.unrealized_pl,
                        s.error
                    FROM symbol_runs s
                    JOIN bot_runs b ON b.id = s.run_id
                    WHERE s.symbol = ? AND b.session_id = ?
                    ORDER BY s.id DESC
                    LIMIT ?
                    """,
                    (symbol, session_id, limit),
                ).fetchall()
        return [dict(row) for row in reversed(rows)]

    def get_latest_snapshot_symbols(
        self,
        session_id: str | None = None,
        *,
        environment: str | None = None,
        account_fingerprint: str | None = None,
    ) -> list[str]:
        rows = self.get_latest_symbol_snapshot(
            session_id=session_id,
            environment=environment,
            account_fingerprint=account_fingerprint,
        )
        return [str(row.get("symbol", "")).strip().upper() for row in rows if str(row.get("symbol", "")).strip()]

    def get_order_history(
        self,
        limit: int = 50,
        session_id: str | None = None,
        *,
        environment: str | None = None,
        account_fingerprint: str | None = None,
    ) -> list[dict[str, Any]]:
        scope = _optional_runtime_scope(environment, account_fingerprint)
        if scope is not None:
            scope = self._require_runtime_state_binding(
                environment=scope[0],
                account_fingerprint=scope[1],
            )
        with self._connect() as connection:
            if scope is None and session_id is None:
                rows = connection.execute(
                    """
                    SELECT order_id, session_id, environment,
                           account_fingerprint, observed_at_utc, submitted_at,
                           filled_at, symbol, side, status, qty, filled_qty,
                           filled_avg_price, notional, client_order_id
                    FROM order_history
                    ORDER BY observed_at_utc DESC
                    LIMIT ?
                    """,
                    (limit,),
                ).fetchall()
            elif scope is None:
                rows = connection.execute(
                    """
                    SELECT order_id, session_id, environment,
                           account_fingerprint, observed_at_utc, submitted_at,
                           filled_at, symbol, side, status, qty, filled_qty,
                           filled_avg_price, notional, client_order_id
                    FROM order_history
                    WHERE session_id = ?
                    ORDER BY observed_at_utc DESC
                    LIMIT ?
                    """,
                    (session_id, limit),
                ).fetchall()
            elif session_id is None:
                rows = connection.execute(
                    """
                    SELECT order_id, session_id, environment,
                           account_fingerprint, observed_at_utc, submitted_at,
                           filled_at, symbol, side, status, qty, filled_qty,
                           filled_avg_price, notional, client_order_id
                    FROM order_history
                    WHERE environment = ? AND account_fingerprint = ?
                    ORDER BY observed_at_utc DESC
                    LIMIT ?
                    """,
                    (*scope, limit),
                ).fetchall()
            else:
                rows = connection.execute(
                    """
                    SELECT order_id, session_id, environment,
                           account_fingerprint, observed_at_utc, submitted_at,
                           filled_at, symbol, side, status, qty, filled_qty,
                           filled_avg_price, notional, client_order_id
                    FROM order_history
                    WHERE session_id = ?
                      AND environment = ?
                      AND account_fingerprint = ?
                    ORDER BY observed_at_utc DESC
                    LIMIT ?
                    """,
                    (session_id, *scope, limit),
                ).fetchall()
        return [dict(row) for row in rows]

    def get_session_first_prices(self, session_id: str | None = None) -> dict[str, float]:
        """Return the earliest recorded price per symbol for the given session.
        Used to compute intra-session % change for the ticker strip."""
        with self._connect() as connection:
            if session_id is None:
                rows = connection.execute(
                    """
                    SELECT s.symbol, s.price
                    FROM symbol_runs s
                    JOIN bot_runs b ON b.id = s.run_id
                    WHERE s.price IS NOT NULL
                    ORDER BY s.id ASC
                    """
                ).fetchall()
            else:
                rows = connection.execute(
                    """
                    SELECT s.symbol, s.price
                    FROM symbol_runs s
                    JOIN bot_runs b ON b.id = s.run_id
                    WHERE b.session_id = ? AND s.price IS NOT NULL
                    ORDER BY s.id ASC
                    """,
                    (session_id,),
                ).fetchall()
        first_prices: dict[str, float] = {}
        for row in rows:
            sym = str(row["symbol"])
            if sym not in first_prices:
                first_prices[sym] = float(row["price"])
        return first_prices

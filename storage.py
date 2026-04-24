import sqlite3
from pathlib import Path
from typing import Any


class BotStorage:
    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.db_path)
        connection.row_factory = sqlite3.Row
        return connection

    def _init_db(self) -> None:
        with self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS bot_runs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT,
                    symbol_fingerprint TEXT,
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
                    observed_at_utc TEXT NOT NULL,
                    submitted_at TEXT,
                    filled_at TEXT,
                    symbol TEXT NOT NULL,
                    side TEXT NOT NULL,
                    status TEXT NOT NULL,
                    qty REAL,
                    filled_qty REAL,
                    filled_avg_price REAL,
                    notional REAL
                );

                CREATE TABLE IF NOT EXISTS processed_decisions (
                    decision_ts TEXT PRIMARY KEY,
                    claimed_at_utc TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS processed_order_fills (
                    order_id TEXT NOT NULL,
                    filled_qty REAL NOT NULL,
                    claimed_at_utc TEXT NOT NULL,
                    PRIMARY KEY (order_id, filled_qty)
                );
                """
            )
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

    def claim_decision_timestamp(self, decision_ts: str, claimed_at_utc: str) -> bool:
        with self._connect() as connection:
            cursor = connection.execute(
                """
                INSERT OR IGNORE INTO processed_decisions (decision_ts, claimed_at_utc)
                VALUES (?, ?)
                """,
                (decision_ts, claimed_at_utc),
            )
            return int(cursor.rowcount or 0) == 1

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

    def save_snapshot(
        self,
        snapshot: Any,
        orders: list[Any],
        *,
        session_id: str | None = None,
        symbol_fingerprint: str | None = None,
    ) -> int:
        with self._connect() as connection:
            cursor = connection.execute(
                """
                INSERT INTO bot_runs (
                    session_id, symbol_fingerprint, timestamp_utc, cash, buying_power, equity, last_equity, daily_pnl, kill_switch_triggered
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    session_id,
                    symbol_fingerprint,
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
                    order_id, session_id, observed_at_utc, submitted_at, filled_at, symbol, side, status, qty, filled_qty, filled_avg_price, notional
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(order_id) DO UPDATE SET
                    session_id=excluded.session_id,
                    observed_at_utc=excluded.observed_at_utc,
                    submitted_at=excluded.submitted_at,
                    filled_at=excluded.filled_at,
                    symbol=excluded.symbol,
                    side=excluded.side,
                    status=excluded.status,
                    qty=excluded.qty,
                    filled_qty=excluded.filled_qty,
                    filled_avg_price=excluded.filled_avg_price,
                    notional=excluded.notional
                """,
                [
                    (
                        order.order_id,
                        session_id,
                        snapshot.timestamp_utc,
                        order.submitted_at,
                        getattr(order, "filled_at", None),
                        order.symbol,
                        order.side,
                        order.status,
                        order.qty,
                        order.filled_qty,
                        order.filled_avg_price,
                        order.notional,
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

    def get_latest_run(self, session_id: str | None = None) -> dict[str, Any] | None:
        with self._connect() as connection:
            if session_id is None:
                row = connection.execute(
                    """
                    SELECT id, session_id, symbol_fingerprint, timestamp_utc, cash, buying_power, equity, last_equity, daily_pnl, kill_switch_triggered
                    FROM bot_runs
                    ORDER BY id DESC
                    LIMIT 1
                    """
                ).fetchone()
            else:
                row = connection.execute(
                    """
                    SELECT id, session_id, symbol_fingerprint, timestamp_utc, cash, buying_power, equity, last_equity, daily_pnl, kill_switch_triggered
                    FROM bot_runs
                    WHERE session_id = ?
                    ORDER BY id DESC
                    LIMIT 1
                    """,
                    (session_id,),
                ).fetchone()
        return dict(row) if row is not None else None

    def get_latest_symbol_snapshot(self, session_id: str | None = None) -> list[dict[str, Any]]:
        latest_run = self.get_latest_run(session_id=session_id)
        if latest_run is None:
            return []

        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT
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

    def get_latest_snapshot_symbols(self, session_id: str | None = None) -> list[str]:
        rows = self.get_latest_symbol_snapshot(session_id=session_id)
        return [str(row.get("symbol", "")).strip().upper() for row in rows if str(row.get("symbol", "")).strip()]

    def get_order_history(self, limit: int = 50, session_id: str | None = None) -> list[dict[str, Any]]:
        with self._connect() as connection:
            if session_id is None:
                rows = connection.execute(
                    """
                    SELECT order_id, session_id, observed_at_utc, submitted_at, filled_at, symbol, side, status, qty, filled_qty, filled_avg_price, notional
                    FROM order_history
                    ORDER BY observed_at_utc DESC
                    LIMIT ?
                    """,
                    (limit,),
                ).fetchall()
            else:
                rows = connection.execute(
                    """
                    SELECT order_id, session_id, observed_at_utc, submitted_at, filled_at, symbol, side, status, qty, filled_qty, filled_avg_price, notional
                    FROM order_history
                    WHERE session_id = ?
                    ORDER BY observed_at_utc DESC
                    LIMIT ?
                    """,
                    (session_id, limit),
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

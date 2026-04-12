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
                    error TEXT,
                    FOREIGN KEY(run_id) REFERENCES bot_runs(id)
                );

                CREATE TABLE IF NOT EXISTS order_history (
                    order_id TEXT PRIMARY KEY,
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
                """
            )
            symbol_columns = {
                row["name"]
                for row in connection.execute("PRAGMA table_info(symbol_runs)").fetchall()
            }
            column_migrations = {
                "ml_probability_up": "ALTER TABLE symbol_runs ADD COLUMN ml_probability_up REAL",
                "ml_confidence": "ALTER TABLE symbol_runs ADD COLUMN ml_confidence REAL",
                "ml_training_rows": "ALTER TABLE symbol_runs ADD COLUMN ml_training_rows INTEGER",
                "holding_minutes": "ALTER TABLE symbol_runs ADD COLUMN holding_minutes REAL",
            }
            for column_name, statement in column_migrations.items():
                if column_name not in symbol_columns:
                    connection.execute(statement)

            order_columns = {
                row["name"]
                for row in connection.execute("PRAGMA table_info(order_history)").fetchall()
            }
            if "filled_at" not in order_columns:
                connection.execute("ALTER TABLE order_history ADD COLUMN filled_at TEXT")

    def save_snapshot(self, snapshot: Any, orders: list[Any]) -> int:
        with self._connect() as connection:
            cursor = connection.execute(
                """
                INSERT INTO bot_runs (
                    timestamp_utc, cash, buying_power, equity, last_equity, daily_pnl, kill_switch_triggered
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
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
                    run_id, symbol, price, sma, ml_probability_up, ml_confidence, ml_training_rows, action, holding, holding_minutes, quantity, market_value, error
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
                        item.error,
                    )
                    for item in snapshot.symbols
                ],
            )

            connection.executemany(
                """
                INSERT INTO order_history (
                    order_id, observed_at_utc, submitted_at, filled_at, symbol, side, status, qty, filled_qty, filled_avg_price, notional
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(order_id) DO UPDATE SET
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

    def get_run_history(self, limit: int = 200) -> list[dict[str, Any]]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT timestamp_utc, cash, buying_power, equity, last_equity, daily_pnl, kill_switch_triggered
                FROM bot_runs
                ORDER BY id DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
        return [dict(row) for row in reversed(rows)]

    def get_latest_run(self) -> dict[str, Any] | None:
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT id, timestamp_utc, cash, buying_power, equity, last_equity, daily_pnl, kill_switch_triggered
                FROM bot_runs
                ORDER BY id DESC
                LIMIT 1
                """
            ).fetchone()
        return dict(row) if row is not None else None

    def get_latest_symbol_snapshot(self) -> list[dict[str, Any]]:
        latest_run = self.get_latest_run()
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

    def get_symbol_history(self, symbol: str, limit: int = 200) -> list[dict[str, Any]]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT
                    b.timestamp_utc,
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
                    s.error
                FROM symbol_runs s
                JOIN bot_runs b ON b.id = s.run_id
                WHERE s.symbol = ?
                ORDER BY s.id DESC
                LIMIT ?
                """,
                (symbol, limit),
            ).fetchall()
        return [dict(row) for row in reversed(rows)]

    def get_order_history(self, limit: int = 50) -> list[dict[str, Any]]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT observed_at_utc, submitted_at, filled_at, symbol, side, status, qty, filled_qty, filled_avg_price, notional
                FROM order_history
                ORDER BY observed_at_utc DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
        return [dict(row) for row in rows]

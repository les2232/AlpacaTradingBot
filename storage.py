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
                    action TEXT NOT NULL,
                    holding INTEGER NOT NULL,
                    quantity REAL NOT NULL,
                    market_value REAL NOT NULL,
                    error TEXT,
                    FOREIGN KEY(run_id) REFERENCES bot_runs(id)
                );

                CREATE TABLE IF NOT EXISTS order_history (
                    order_id TEXT PRIMARY KEY,
                    observed_at_utc TEXT NOT NULL,
                    submitted_at TEXT,
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
                    run_id, symbol, price, sma, action, holding, quantity, market_value, error
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        run_id,
                        item.symbol,
                        item.price,
                        item.sma,
                        item.action,
                        int(item.holding),
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
                    order_id, observed_at_utc, submitted_at, symbol, side, status, qty, filled_qty, filled_avg_price, notional
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(order_id) DO UPDATE SET
                    observed_at_utc=excluded.observed_at_utc,
                    submitted_at=excluded.submitted_at,
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

    def get_symbol_history(self, symbol: str, limit: int = 200) -> list[dict[str, Any]]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT b.timestamp_utc, s.symbol, s.price, s.sma, s.action, s.holding, s.quantity, s.market_value, s.error
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
                SELECT observed_at_utc, submitted_at, symbol, side, status, qty, filled_qty, filled_avg_price, notional
                FROM order_history
                ORDER BY observed_at_utc DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
        return [dict(row) for row in rows]

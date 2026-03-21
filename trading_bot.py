import os
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, cast

from alpaca.data.enums import DataFeed
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest, StockLatestTradeRequest
from alpaca.data.timeframe import TimeFrame
from alpaca.trading.client import TradingClient
from alpaca.trading.enums import OrderSide, QueryOrderStatus, TimeInForce
from alpaca.trading.models import Position
from alpaca.trading.requests import GetOrdersRequest, MarketOrderRequest
from dotenv import load_dotenv

from storage import BotStorage


@dataclass(frozen=True)
class BotConfig:
    symbols: list[str]
    max_usd_per_trade: float
    max_open_positions: int
    max_daily_loss_usd: float
    sma_days: int
    paper: bool = True


@dataclass(frozen=True)
class SymbolSnapshot:
    symbol: str
    price: float | None
    sma: float | None
    action: str
    holding: bool
    quantity: float
    market_value: float
    error: str | None = None


@dataclass(frozen=True)
class BotSnapshot:
    timestamp_utc: str
    cash: float
    buying_power: float
    equity: float
    last_equity: float
    daily_pnl: float
    kill_switch_triggered: bool
    positions: dict[str, Position]
    symbols: list[SymbolSnapshot]


@dataclass(frozen=True)
class OrderSnapshot:
    order_id: str
    submitted_at: str | None
    symbol: str
    side: str
    status: str
    qty: float | None
    filled_qty: float | None
    filled_avg_price: float | None
    notional: float | None


def load_config() -> BotConfig:
    symbols_raw = os.getenv("BOT_SYMBOLS", "AAPL,MSFT,NVDA")
    symbols = [symbol.strip().upper() for symbol in symbols_raw.split(",") if symbol.strip()]
    if not symbols:
        raise RuntimeError("BOT_SYMBOLS must contain at least one ticker.")

    return BotConfig(
        symbols=symbols,
        max_usd_per_trade=float(os.getenv("MAX_USD_PER_TRADE", "200")),
        max_open_positions=int(os.getenv("MAX_OPEN_POSITIONS", "3")),
        max_daily_loss_usd=float(os.getenv("MAX_DAILY_LOSS_USD", "300")),
        sma_days=int(os.getenv("SMA_DAYS", "20")),
        paper=os.getenv("ALPACA_PAPER", "true").lower() != "false",
    )


class AlpacaTradingBot:
    def __init__(self, config: BotConfig) -> None:
        load_dotenv(Path.cwd() / ".env")

        api_key = os.getenv("ALPACA_API_KEY")
        api_secret = os.getenv("ALPACA_API_SECRET")
        if not api_key or not api_secret:
            raise RuntimeError(
                "Missing Alpaca credentials. Set ALPACA_API_KEY and ALPACA_API_SECRET in .env."
            )

        self.config = config
        self.trading = TradingClient(api_key, api_secret, paper=config.paper)
        self.data = StockHistoricalDataClient(api_key, api_secret)
        db_path = Path(os.getenv("BOT_DB_PATH", "bot_history.db"))
        self.storage = BotStorage(db_path)

    def get_account(self) -> Any:
        return cast(Any, self.trading).get_account()

    def get_latest_price(self, symbol: str) -> float:
        request = StockLatestTradeRequest(symbol_or_symbols=symbol, feed=DataFeed.IEX)
        latest = cast(dict[str, Any], cast(Any, self.data).get_stock_latest_trade(request))
        return float(latest[symbol].price)

    def get_positions_by_symbol(self) -> dict[str, Position]:
        positions = cast(list[Position], cast(Any, self.trading).get_all_positions())
        return {position.symbol: position for position in positions}

    def get_sma(self, symbol: str, days: int) -> float:
        end = datetime.now(timezone.utc)
        start = end - timedelta(days=days * 3)
        request = StockBarsRequest(
            symbol_or_symbols=[symbol],
            timeframe=cast(TimeFrame, TimeFrame.Day),
            start=start,
            end=end,
            limit=days,
            feed=DataFeed.IEX,
        )

        bars_response = cast(Any, self.data).get_stock_bars(request)
        bars = cast(list[Any], bars_response.data.get(symbol, []))
        closes = [float(bar.close) for bar in bars][-days:]
        if len(closes) < max(5, days // 2):
            raise RuntimeError(f"Not enough daily bars for {symbol}: got {len(closes)}")

        return sum(closes) / len(closes)

    def decide(self, symbol: str, positions: dict[str, Position]) -> str:
        price = self.get_latest_price(symbol)
        sma = self.get_sma(symbol, days=self.config.sma_days)
        holding = symbol in positions

        if price > sma and not holding:
            return "BUY"
        if price < sma and holding:
            return "SELL"
        return "HOLD"

    def daily_pnl(self) -> float:
        account = self.get_account()
        return float(account.equity) - float(account.last_equity)

    def kill_switch_triggered(self) -> bool:
        pnl = self.daily_pnl()
        print(f"Daily PnL: {pnl:.2f}")
        return pnl <= -self.config.max_daily_loss_usd

    def build_snapshot(self) -> BotSnapshot:
        account = self.get_account()
        positions = self.get_positions_by_symbol()
        symbols: list[SymbolSnapshot] = []

        for symbol in self.config.symbols:
            position = positions.get(symbol)
            try:
                price = self.get_latest_price(symbol)
                sma = self.get_sma(symbol, days=self.config.sma_days)
                holding = position is not None
                quantity = float(position.qty) if position is not None else 0.0
                market_value = float(position.market_value) if position is not None else 0.0

                if price > sma and not holding:
                    action = "BUY"
                elif price < sma and holding:
                    action = "SELL"
                else:
                    action = "HOLD"

                symbols.append(
                    SymbolSnapshot(
                        symbol=symbol,
                        price=price,
                        sma=sma,
                        action=action,
                        holding=holding,
                        quantity=quantity,
                        market_value=market_value,
                    )
                )
            except Exception as exc:
                symbols.append(
                    SymbolSnapshot(
                        symbol=symbol,
                        price=None,
                        sma=None,
                        action="ERROR",
                        holding=position is not None,
                        quantity=float(position.qty) if position is not None else 0.0,
                        market_value=float(position.market_value) if position is not None else 0.0,
                        error=str(exc),
                    )
                )

        daily_pnl = float(account.equity) - float(account.last_equity)
        return BotSnapshot(
            timestamp_utc=datetime.now(timezone.utc).isoformat(),
            cash=float(account.cash),
            buying_power=float(account.buying_power),
            equity=float(account.equity),
            last_equity=float(account.last_equity),
            daily_pnl=daily_pnl,
            kill_switch_triggered=daily_pnl <= -self.config.max_daily_loss_usd,
            positions=positions,
            symbols=symbols,
        )

    def get_recent_orders(self, limit: int = 10) -> list[OrderSnapshot]:
        request = GetOrdersRequest(status=QueryOrderStatus.ALL, limit=limit, nested=False)
        orders = cast(list[Any], cast(Any, self.trading).get_orders(filter=request))
        snapshots: list[OrderSnapshot] = []

        for order in orders:
            snapshots.append(
                OrderSnapshot(
                    order_id=str(getattr(order, "id", "")),
                    submitted_at=str(getattr(order, "submitted_at", None)),
                    symbol=str(getattr(order, "symbol", "")),
                    side=str(getattr(order, "side", "")),
                    status=str(getattr(order, "status", "")),
                    qty=float(order.qty) if getattr(order, "qty", None) else None,
                    filled_qty=float(order.filled_qty) if getattr(order, "filled_qty", None) else None,
                    filled_avg_price=(
                        float(order.filled_avg_price)
                        if getattr(order, "filled_avg_price", None)
                        else None
                    ),
                    notional=float(order.notional) if getattr(order, "notional", None) else None,
                )
            )

        return snapshots

    def record_state(self, snapshot: BotSnapshot, orders_limit: int = 20) -> list[OrderSnapshot]:
        orders = self.get_recent_orders(limit=orders_limit)
        self.storage.save_snapshot(snapshot, orders)
        return orders

    def capture_state(self, orders_limit: int = 20) -> tuple[BotSnapshot, list[OrderSnapshot]]:
        snapshot = self.build_snapshot()
        orders = self.record_state(snapshot, orders_limit=orders_limit)
        return snapshot, orders

    def place_market_buy(self, symbol: str) -> Any | None:
        price = self.get_latest_price(symbol)
        qty = int(self.config.max_usd_per_trade // price)
        if qty <= 0:
            print(f"Skip {symbol}: price {price:.2f} is above ${self.config.max_usd_per_trade:.2f}")
            return None

        order = cast(
            Any,
            self.trading.submit_order(
                MarketOrderRequest(
                    symbol=symbol,
                    qty=qty,
                    side=OrderSide.BUY,
                    time_in_force=TimeInForce.DAY,
                )
            ),
        )
        print(f"Submitted BUY {symbol} qty={qty} approx=${qty * price:.2f}")
        return order

    def place_market_sell(self, symbol: str, position: Position) -> Any | None:
        qty = int(float(position.qty))
        if qty <= 0:
            print(f"Skip SELL {symbol}: non-positive quantity {position.qty}")
            return None

        order = cast(
            Any,
            self.trading.submit_order(
                MarketOrderRequest(
                    symbol=symbol,
                    qty=qty,
                    side=OrderSide.SELL,
                    time_in_force=TimeInForce.DAY,
                )
            ),
        )
        print(f"Submitted SELL {symbol} qty={qty}")
        return order

    def run_once(self, execute_orders: bool = True) -> BotSnapshot:
        print("\n=== BOT TICK ===")
        snapshot = self.build_snapshot()
        print(f"Connected. Cash: {snapshot.cash} Buying power: {snapshot.buying_power}")
        print(f"Daily PnL: {snapshot.daily_pnl:.2f}")

        if snapshot.kill_switch_triggered:
            print("Kill switch triggered. No trades submitted.")
            self.record_state(snapshot)
            return snapshot

        if not execute_orders:
            for item in snapshot.symbols:
                suffix = f" ERROR: {item.error}" if item.error else ""
                print(f"{item.symbol} -> {item.action}{suffix}")
            self.record_state(snapshot)
            return snapshot

        positions = snapshot.positions.copy()
        open_positions = len(positions)

        for item in snapshot.symbols:
            try:
                symbol = item.symbol
                action = item.action
                if item.error:
                    print(f"{symbol} ERROR: {item.error}")
                    continue

                print(f"{symbol} -> {action}")

                if action == "BUY":
                    if symbol in positions:
                        print(f"Already holding {symbol}")
                        continue
                    if open_positions >= self.config.max_open_positions:
                        print("Max positions reached")
                        continue
                    order = self.place_market_buy(symbol)
                    if order is not None:
                        open_positions += 1

                elif action == "SELL" and symbol in positions:
                    order = self.place_market_sell(symbol, positions[symbol])
                    if order is not None:
                        open_positions = max(0, open_positions - 1)

            except Exception as exc:
                print(f"{symbol} ERROR: {exc}")

        snapshot = self.build_snapshot()
        self.record_state(snapshot)
        return snapshot


def main() -> None:
    config = load_config()
    bot = AlpacaTradingBot(config)
    execute_orders = os.getenv("EXECUTE_ORDERS", "true").lower() != "false"
    bot.run_once(execute_orders=execute_orders)


if __name__ == "__main__":
    main()

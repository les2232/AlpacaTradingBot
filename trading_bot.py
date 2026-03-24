import os
import math
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, cast

from alpaca.data.enums import DataFeed
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.live.stock import StockDataStream
from alpaca.data.requests import StockBarsRequest, StockLatestTradeRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
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
    sma_bars: int
    bar_timeframe_minutes: int
    paper: bool = True
    strategy_mode: str = "hybrid"
    ml_lookback_bars: int = 320
    ml_probability_buy: float = 0.55
    ml_probability_sell: float = 0.45
    ml_train_every_seconds: int = 900


@dataclass(frozen=True)
class MlSignal:
    probability_up: float
    confidence: float
    training_rows: int
    model_age_seconds: float
    feature_names: tuple[str, ...]


@dataclass(frozen=True)
class SymbolSnapshot:
    symbol: str
    price: float | None
    sma: float | None
    action: str
    holding: bool
    quantity: float
    market_value: float
    ml_probability_up: float | None = None
    ml_confidence: float | None = None
    ml_training_rows: int | None = None
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


@dataclass(frozen=True)
class TrainedLogisticModel:
    weights: tuple[float, ...]
    bias: float
    means: tuple[float, ...]
    scales: tuple[float, ...]
    trained_at_unix: float
    training_rows: int
    feature_names: tuple[str, ...]

    def predict_probability(self, features: list[float]) -> float:
        z = self.bias
        for value, mean, scale, weight in zip(features, self.means, self.scales, self.weights):
            normalized = (value - mean) / scale
            z += normalized * weight
        return 1.0 / (1.0 + math.exp(-max(min(z, 35.0), -35.0)))


FEATURE_NAMES = (
    "ret_1",
    "ret_3",
    "ret_5",
    "price_vs_sma_10",
    "price_vs_sma_20",
    "volatility_10",
    "volume_vs_avg_10",
)


def _safe_float(value: str | None, default: float) -> float:
    try:
        return float(value) if value is not None else default
    except (TypeError, ValueError):
        return default


def load_config() -> BotConfig:
    symbols_raw = os.getenv("BOT_SYMBOLS", "AAPL,MSFT,NVDA")
    symbols = [symbol.strip().upper() for symbol in symbols_raw.split(",") if symbol.strip()]
    if not symbols:
        raise RuntimeError("BOT_SYMBOLS must contain at least one ticker.")

    sma_bars_raw = os.getenv("SMA_BARS") or os.getenv("SMA_DAYS", "20")
    bar_timeframe_minutes = int(os.getenv("BAR_TIMEFRAME_MINUTES", "15"))

    return BotConfig(
        symbols=symbols,
        max_usd_per_trade=float(os.getenv("MAX_USD_PER_TRADE", "200")),
        max_open_positions=int(os.getenv("MAX_OPEN_POSITIONS", "3")),
        max_daily_loss_usd=float(os.getenv("MAX_DAILY_LOSS_USD", "300")),
        sma_bars=int(sma_bars_raw),
        bar_timeframe_minutes=bar_timeframe_minutes,
        paper=os.getenv("ALPACA_PAPER", "true").lower() != "false",
        strategy_mode=os.getenv("STRATEGY_MODE", "hybrid").strip().lower(),
        ml_lookback_bars=int(os.getenv("ML_LOOKBACK_BARS", "320")),
        ml_probability_buy=_safe_float(os.getenv("ML_PROBABILITY_BUY"), 0.55),
        ml_probability_sell=_safe_float(os.getenv("ML_PROBABILITY_SELL"), 0.45),
        ml_train_every_seconds=int(os.getenv("ML_TRAIN_EVERY_SECONDS", "900")),
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
        self.data_stream = StockDataStream(api_key, api_secret, feed=DataFeed.IEX)
        db_path = Path(os.getenv("BOT_DB_PATH", "bot_history.db"))
        self.storage = BotStorage(db_path)
        self._latest_prices: dict[str, float] = {}
        self._latest_price_times: dict[str, float] = {}
        self._price_lock = threading.Lock()
        self._stream_error: str | None = None
        self._stream_thread: threading.Thread | None = None
        self._model_cache: dict[str, TrainedLogisticModel] = {}
        self._start_price_stream()

    def get_account(self) -> Any:
        return cast(Any, self.trading).get_account()

    def get_price_feed_status(self) -> str:
        with self._price_lock:
            active_symbols = len(self._latest_prices)

        if self._stream_error:
            return f"stream error: {self._stream_error}"
        if active_symbols == 0:
            return "stream connecting"
        return f"live stream active for {active_symbols}/{len(self.config.symbols)} symbols"

    def _start_price_stream(self) -> None:
        if self._stream_thread is not None:
            return

        self.data_stream.subscribe_trades(self._handle_trade, *self.config.symbols)
        self._stream_thread = threading.Thread(target=self._run_price_stream, daemon=True)
        self._stream_thread.start()

    def _run_price_stream(self) -> None:
        try:
            self.data_stream.run()
        except Exception as exc:
            self._stream_error = str(exc)

    async def _handle_trade(self, trade: Any) -> None:
        symbol = str(getattr(trade, "symbol", ""))
        price = getattr(trade, "price", None)
        if not symbol or price is None:
            return

        with self._price_lock:
            self._latest_prices[symbol] = float(price)
            self._latest_price_times[symbol] = time.time()
            self._stream_error = None

    def get_latest_price(self, symbol: str) -> float:
        with self._price_lock:
            cached_price = self._latest_prices.get(symbol)
            cached_at = self._latest_price_times.get(symbol)

        if cached_price is not None and cached_at is not None and (time.time() - cached_at) <= 15:
            return cached_price

        request = StockLatestTradeRequest(symbol_or_symbols=symbol, feed=DataFeed.IEX)
        latest = cast(dict[str, Any], cast(Any, self.data).get_stock_latest_trade(request))
        price = float(latest[symbol].price)

        with self._price_lock:
            self._latest_prices[symbol] = price
            self._latest_price_times[symbol] = time.time()

        return price

    def get_positions_by_symbol(self) -> dict[str, Position]:
        positions = cast(list[Position], cast(Any, self.trading).get_all_positions())
        return {position.symbol: position for position in positions}

    def _get_intraday_bars(self, symbol: str, bars_needed: int) -> list[Any]:
        if bars_needed <= 0:
            raise RuntimeError("bars_needed must be greater than zero.")

        end = datetime.now(timezone.utc)
        trading_minutes_per_day = 390
        timeframe_minutes = self.config.bar_timeframe_minutes
        trading_days_needed = max(3, math.ceil((bars_needed * timeframe_minutes) / trading_minutes_per_day))
        start = end - timedelta(days=trading_days_needed * 6)
        request = StockBarsRequest(
            symbol_or_symbols=[symbol],
            timeframe=TimeFrame(timeframe_minutes, TimeFrameUnit.Minute),
            start=start,
            end=end,
            limit=bars_needed,
            feed=DataFeed.IEX,
        )

        bars_response = cast(Any, self.data).get_stock_bars(request)
        return cast(list[Any], bars_response.data.get(symbol, []))

    def get_sma(self, symbol: str, bars: int) -> float:
        if bars <= 0:
            raise RuntimeError("SMA_BARS must be greater than zero.")

        intraday_bars = self._get_intraday_bars(symbol, bars)
        closes = [float(bar.close) for bar in intraday_bars][-bars:]
        if len(closes) < max(5, bars // 2):
            raise RuntimeError(f"Not enough intraday bars for {symbol}: got {len(closes)}")

        return sum(closes) / len(closes)

    def _mean(self, values: list[float]) -> float:
        return sum(values) / len(values)

    def _stddev(self, values: list[float], mean_value: float) -> float:
        variance = sum((value - mean_value) ** 2 for value in values) / max(1, len(values))
        return math.sqrt(max(variance, 1e-12))

    def _build_feature_vector(self, closes: list[float], volumes: list[float], index: int) -> list[float]:
        price = closes[index]
        ret_1 = (price / closes[index - 1]) - 1.0
        ret_3 = (price / closes[index - 3]) - 1.0
        ret_5 = (price / closes[index - 5]) - 1.0
        sma_10 = self._mean(closes[index - 9 : index + 1])
        sma_20 = self._mean(closes[index - 19 : index + 1])
        vol_returns = [
            (closes[j] / closes[j - 1]) - 1.0
            for j in range(index - 9, index + 1)
            if j - 1 >= 0
        ]
        avg_volume_10 = self._mean(volumes[index - 9 : index + 1])
        return [
            ret_1,
            ret_3,
            ret_5,
            (price / sma_10) - 1.0,
            (price / sma_20) - 1.0,
            self._stddev(vol_returns, self._mean(vol_returns)),
            (volumes[index] / max(avg_volume_10, 1.0)) - 1.0,
        ]

    def _train_logistic_regression(
        self,
        feature_rows: list[list[float]],
        labels: list[int],
        feature_names: tuple[str, ...],
    ) -> TrainedLogisticModel:
        row_count = len(feature_rows)
        column_count = len(feature_rows[0])
        means: list[float] = []
        scales: list[float] = []

        for col in range(column_count):
            column_values = [row[col] for row in feature_rows]
            mean_value = self._mean(column_values)
            scale = self._stddev(column_values, mean_value)
            means.append(mean_value)
            scales.append(scale if scale > 1e-9 else 1.0)

        normalized_rows: list[list[float]] = []
        for row in feature_rows:
            normalized_rows.append(
                [(value - means[idx]) / scales[idx] for idx, value in enumerate(row)]
            )

        weights = [0.0] * column_count
        bias = 0.0
        learning_rate = 0.08
        epochs = 180

        for _ in range(epochs):
            grad_w = [0.0] * column_count
            grad_b = 0.0
            for row, label in zip(normalized_rows, labels):
                z = bias + sum(weight * value for weight, value in zip(weights, row))
                prob = 1.0 / (1.0 + math.exp(-max(min(z, 35.0), -35.0)))
                error = prob - label
                for idx, value in enumerate(row):
                    grad_w[idx] += error * value
                grad_b += error

            row_scale = 1.0 / row_count
            for idx in range(column_count):
                weights[idx] -= learning_rate * grad_w[idx] * row_scale
            bias -= learning_rate * grad_b * row_scale

        return TrainedLogisticModel(
            weights=tuple(weights),
            bias=bias,
            means=tuple(means),
            scales=tuple(scales),
            trained_at_unix=time.time(),
            training_rows=row_count,
            feature_names=feature_names,
        )

    def _get_or_train_ml_model(self, symbol: str) -> TrainedLogisticModel:
        cached_model = self._model_cache.get(symbol)
        if cached_model is not None and (time.time() - cached_model.trained_at_unix) < self.config.ml_train_every_seconds:
            return cached_model

        bars_needed = max(self.config.ml_lookback_bars, 80)
        intraday_bars = self._get_intraday_bars(symbol, bars_needed)
        closes = [float(bar.close) for bar in intraday_bars]
        volumes = [float(getattr(bar, "volume", 0.0) or 0.0) for bar in intraday_bars]

        if len(closes) < 40:
            raise RuntimeError(f"Not enough bars to train ML model for {symbol}: got {len(closes)}")

        feature_rows: list[list[float]] = []
        labels: list[int] = []
        max_index = len(closes) - 2
        for index in range(19, max_index + 1):
            features = self._build_feature_vector(closes, volumes, index)
            next_close = closes[index + 1]
            label = 1 if next_close > closes[index] else 0
            feature_rows.append(features)
            labels.append(label)

        if len(feature_rows) < 20:
            raise RuntimeError(f"Not enough ML training rows for {symbol}: got {len(feature_rows)}")

        model = self._train_logistic_regression(feature_rows, labels, FEATURE_NAMES)
        self._model_cache[symbol] = model
        return model

    def get_ml_signal(self, symbol: str) -> MlSignal:
        model = self._get_or_train_ml_model(symbol)
        bars_needed = max(self.config.sma_bars, 25)
        intraday_bars = self._get_intraday_bars(symbol, bars_needed)
        closes = [float(bar.close) for bar in intraday_bars]
        volumes = [float(getattr(bar, "volume", 0.0) or 0.0) for bar in intraday_bars]
        latest_index = len(closes) - 1
        if latest_index < 19:
            raise RuntimeError(f"Not enough bars to score ML signal for {symbol}: got {len(closes)}")

        features = self._build_feature_vector(closes, volumes, latest_index)
        probability = model.predict_probability(features)
        confidence = abs(probability - 0.5) * 2.0
        return MlSignal(
            probability_up=probability,
            confidence=confidence,
            training_rows=model.training_rows,
            model_age_seconds=time.time() - model.trained_at_unix,
            feature_names=model.feature_names,
        )

    def decide(self, symbol: str, positions: dict[str, Position]) -> str:
        price = self.get_latest_price(symbol)
        sma = self.get_sma(symbol, bars=self.config.sma_bars)
        ml_signal = self.get_ml_signal(symbol)
        holding = symbol in positions
        mode = self.config.strategy_mode

        if mode == "sma":
            if price > sma and not holding:
                return "BUY"
            if price < sma and holding:
                return "SELL"
            return "HOLD"

        if mode == "ml":
            if ml_signal.probability_up >= self.config.ml_probability_buy and not holding:
                return "BUY"
            if ml_signal.probability_up <= self.config.ml_probability_sell and holding:
                return "SELL"
            return "HOLD"

        # hybrid mode: require SMA trend confirmation for buys, allow either risk signal for sells
        if (
            price > sma
            and ml_signal.probability_up >= self.config.ml_probability_buy
            and not holding
        ):
            return "BUY"
        if holding and (
            price < sma or ml_signal.probability_up <= self.config.ml_probability_sell
        ):
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
                sma = self.get_sma(symbol, bars=self.config.sma_bars)
                ml_signal = self.get_ml_signal(symbol)
                holding = position is not None
                quantity = float(position.qty) if position is not None else 0.0
                market_value = float(position.market_value) if position is not None else 0.0

                mode = self.config.strategy_mode
                if mode == "sma":
                    if price > sma and not holding:
                        action = "BUY"
                    elif price < sma and holding:
                        action = "SELL"
                    else:
                        action = "HOLD"
                elif mode == "ml":
                    if ml_signal.probability_up >= self.config.ml_probability_buy and not holding:
                        action = "BUY"
                    elif ml_signal.probability_up <= self.config.ml_probability_sell and holding:
                        action = "SELL"
                    else:
                        action = "HOLD"
                else:
                    if price > sma and ml_signal.probability_up >= self.config.ml_probability_buy and not holding:
                        action = "BUY"
                    elif holding and (price < sma or ml_signal.probability_up <= self.config.ml_probability_sell):
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
                        ml_probability_up=ml_signal.probability_up,
                        ml_confidence=ml_signal.confidence,
                        ml_training_rows=ml_signal.training_rows,
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
        print(f"Strategy mode: {self.config.strategy_mode}")

        if snapshot.kill_switch_triggered:
            print("Kill switch triggered. No trades submitted.")
            self.record_state(snapshot)
            return snapshot

        if not execute_orders:
            for item in snapshot.symbols:
                suffix = f" ml_up={item.ml_probability_up:.3f}" if item.ml_probability_up is not None else ""
                error_suffix = f" ERROR: {item.error}" if item.error else ""
                print(f"{item.symbol} -> {item.action}{suffix}{error_suffix}")
            self.record_state(snapshot)
            return snapshot

        positions = snapshot.positions.copy()
        open_positions = len(positions)

        for item in snapshot.symbols:
            symbol = item.symbol
            try:
                action = item.action
                if item.error:
                    print(f"{symbol} ERROR: {item.error}")
                    continue

                ml_text = (
                    f" ml_up={item.ml_probability_up:.3f} conf={item.ml_confidence:.3f}"
                    if item.ml_probability_up is not None and item.ml_confidence is not None
                    else ""
                )
                print(f"{symbol} -> {action}{ml_text}")

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

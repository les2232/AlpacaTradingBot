import os
import math
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from datetime import time as dt_time
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
import pytz
from dotenv import load_dotenv

from storage import BotStorage
from strategy import Strategy, StrategyConfig, MlSignal


@dataclass(frozen=True)
class BotConfig:
    symbols: list[str]
    max_usd_per_trade: float
    max_symbol_exposure_usd: float
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
    ml_model_type: str = "logistic"
    ml_validation_fraction: float = 0.2
    max_orders_per_minute: int = 6
    max_price_deviation_bps: float = 75.0
    max_data_delay_seconds: int = 1800
    max_live_price_age_seconds: int = 30


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
    ml_buy_threshold: float | None = None
    ml_sell_threshold: float | None = None
    ml_model_name: str | None = None
    error: str | None = None


@dataclass(frozen=True)
class SymbolEvaluation:
    price: float
    sma: float
    ml_signal: MlSignal
    action: str
    latest_bar_close_utc: str


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
class TrainedSignalModel:
    estimator: Any
    trained_at_unix: float
    training_rows: int
    feature_names: tuple[str, ...]
    buy_threshold: float
    sell_threshold: float
    validation_rows: int
    model_name: str

    def predict_probability(self, features: list[float]) -> float:
        probabilities = cast(list[list[float]], self.estimator.predict_proba([features]))
        return float(probabilities[0][1])


_ET = pytz.timezone("America/New_York")
_SESSION_ENTRY_START = dt_time(9, 45)   # no new entries before this
_SESSION_ENTRY_END   = dt_time(15, 45)  # no new entries after this
_SESSION_FLATTEN_AT  = dt_time(15, 55)  # forced EOD flatten deadline


FEATURE_NAMES = (
    "ret_1",
    "ret_3",
    "ret_5",
    "price_vs_sma_10",
    "price_vs_sma_20",
    "volatility_10",
    "volume_vs_avg_10",
)


def _get_sklearn_components() -> tuple[Any, Any, Any, Any, Any]:
    try:
        from sklearn.calibration import CalibratedClassifierCV
        from sklearn.ensemble import GradientBoostingClassifier
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
    except ImportError as exc:
        raise RuntimeError(
            "Missing scikit-learn. Run `python -m pip install -r requirements.txt`."
        ) from exc

    return (
        CalibratedClassifierCV,
        GradientBoostingClassifier,
        LogisticRegression,
        Pipeline,
        StandardScaler,
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
        max_symbol_exposure_usd=float(
            os.getenv("MAX_SYMBOL_EXPOSURE_USD", os.getenv("MAX_USD_PER_TRADE", "200"))
        ),
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
        ml_model_type=os.getenv("ML_MODEL_TYPE", "logistic").strip().lower(),
        ml_validation_fraction=_safe_float(os.getenv("ML_VALIDATION_FRACTION"), 0.2),
        max_orders_per_minute=int(os.getenv("MAX_ORDERS_PER_MINUTE", "6")),
        max_price_deviation_bps=_safe_float(os.getenv("MAX_PRICE_DEVIATION_BPS"), 75.0),
        max_data_delay_seconds=int(os.getenv("MAX_DATA_DELAY_SECONDS", "1800")),
        max_live_price_age_seconds=int(os.getenv("MAX_LIVE_PRICE_AGE_SECONDS", "30")),
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
        self._api_key = api_key
        self._api_secret = api_secret
        db_path = Path(os.getenv("BOT_DB_PATH", "bot_history.db"))
        self.storage = BotStorage(db_path)
        self._latest_prices: dict[str, float] = {}
        self._latest_price_times: dict[str, float] = {}
        self._latest_trade_times: dict[str, float] = {}
        self._price_lock = threading.Lock()
        self._stream_enabled = os.getenv("ENABLE_PRICE_STREAM", "true").lower() != "false"
        self._stream_error: str | None = None
        self.data_stream: StockDataStream | None = None
        self._stream_thread: threading.Thread | None = None
        self._model_cache: dict[str, TrainedSignalModel] = {}
        self.strategy = Strategy(
            StrategyConfig(
                strategy_mode=config.strategy_mode,
                ml_probability_buy=config.ml_probability_buy,
                ml_probability_sell=config.ml_probability_sell,
            )
        )

    def get_account(self) -> Any:
        return cast(Any, self.trading).get_account()

    def get_price_feed_status(self) -> str:
        with self._price_lock:
            active_symbols = len(self._latest_prices)

        if not self._stream_enabled:
            return "live stream disabled"
        if self._stream_error:
            return f"stream error: {self._stream_error}"
        if self._stream_thread is None:
            return "stream idle"
        if active_symbols == 0:
            return "stream connecting"
        return f"live stream active for {active_symbols}/{len(self.config.symbols)} symbols"

    def _start_price_stream(self) -> None:
        if not self._stream_enabled or self._stream_thread is not None:
            return

        self.data_stream = StockDataStream(self._api_key, self._api_secret, feed=DataFeed.IEX)
        self.data_stream.subscribe_trades(self._handle_trade, *self.config.symbols)
        self._stream_thread = threading.Thread(target=self._run_price_stream, daemon=True)
        self._stream_thread.start()

    def _run_price_stream(self) -> None:
        try:
            if self.data_stream is None:
                return
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
            trade_timestamp = getattr(trade, "timestamp", None)
            if isinstance(trade_timestamp, datetime):
                if trade_timestamp.tzinfo is None:
                    normalized_trade_time = trade_timestamp.replace(tzinfo=timezone.utc)
                else:
                    normalized_trade_time = trade_timestamp.astimezone(timezone.utc)
                self._latest_trade_times[symbol] = normalized_trade_time.timestamp()
            else:
                self._latest_trade_times[symbol] = time.time()
            self._stream_error = None

    def get_latest_price_with_age(self, symbol: str) -> tuple[float, float]:
        self._start_price_stream()

        with self._price_lock:
            cached_price = self._latest_prices.get(symbol)
            cached_at = self._latest_price_times.get(symbol)
            cached_trade_at = self._latest_trade_times.get(symbol)

        if cached_price is not None and cached_at is not None and (time.time() - cached_at) <= 15:
            if cached_trade_at is not None:
                return cached_price, max(0.0, time.time() - cached_trade_at)
            return cached_price, time.time() - cached_at

        request = StockLatestTradeRequest(symbol_or_symbols=symbol, feed=DataFeed.IEX)
        latest = cast(dict[str, Any], cast(Any, self.data).get_stock_latest_trade(request))
        trade = latest[symbol]
        price = float(trade.price)
        timestamp = getattr(trade, "timestamp", None)
        if isinstance(timestamp, datetime):
            if timestamp.tzinfo is None:
                trade_timestamp = timestamp.replace(tzinfo=timezone.utc)
            else:
                trade_timestamp = timestamp.astimezone(timezone.utc)
            age_seconds = max(0.0, (datetime.now(timezone.utc) - trade_timestamp).total_seconds())
        else:
            age_seconds = 0.0

        with self._price_lock:
            self._latest_prices[symbol] = price
            self._latest_price_times[symbol] = time.time()
            self._latest_trade_times[symbol] = time.time() - age_seconds

        return price, age_seconds

    def get_latest_price(self, symbol: str) -> float:
        price, _ = self.get_latest_price_with_age(symbol)
        return price

    def get_positions_by_symbol(self) -> dict[str, Position]:
        positions = cast(list[Position], cast(Any, self.trading).get_all_positions())
        return {position.symbol: position for position in positions}

    def _bar_interval(self) -> timedelta:
        return timedelta(minutes=self.config.bar_timeframe_minutes)

    def get_decision_timestamp(self, now: datetime | None = None) -> datetime:
        current_time = now or datetime.now(timezone.utc)
        bar_seconds = int(self._bar_interval().total_seconds())
        decision_unix = int(current_time.timestamp()) // bar_seconds * bar_seconds
        return datetime.fromtimestamp(decision_unix, tz=timezone.utc)

    def _get_bar_start_time(self, bar: Any) -> datetime:
        raw_timestamp = getattr(bar, "timestamp", None)
        if not isinstance(raw_timestamp, datetime):
            raise RuntimeError("Bar is missing a timestamp.")
        if raw_timestamp.tzinfo is None:
            return raw_timestamp.replace(tzinfo=timezone.utc)
        return raw_timestamp.astimezone(timezone.utc)

    def _get_intraday_bars(
        self,
        symbol: str,
        bars_needed: int,
        decision_timestamp: datetime | None = None,
    ) -> list[Any]:
        if bars_needed <= 0:
            raise RuntimeError("bars_needed must be greater than zero.")

        aligned_decision_timestamp = decision_timestamp or self.get_decision_timestamp()
        trading_minutes_per_day = 390
        timeframe_minutes = self.config.bar_timeframe_minutes
        trading_days_needed = max(3, math.ceil((bars_needed * timeframe_minutes) / trading_minutes_per_day))
        start = aligned_decision_timestamp - timedelta(days=trading_days_needed * 6)
        request_end = aligned_decision_timestamp + self._bar_interval()
        request = StockBarsRequest(
            symbol_or_symbols=[symbol],
            timeframe=TimeFrame(timeframe_minutes, TimeFrameUnit.Minute),
            start=start,
            end=request_end,
            limit=bars_needed + 8,
            feed=DataFeed.IEX,
        )

        bars_response = cast(Any, self.data).get_stock_bars(request)
        bars = cast(list[Any], bars_response.data.get(symbol, []))
        completed_bars = [
            bar
            for bar in bars
            if (self._get_bar_start_time(bar) + self._bar_interval()) <= aligned_decision_timestamp
        ]
        return completed_bars[-bars_needed:]

    def _latest_bar_close_time(self, bars: list[Any]) -> datetime:
        if not bars:
            raise RuntimeError("No completed bars available.")
        return self._get_bar_start_time(bars[-1]) + self._bar_interval()

    def get_sma(self, symbol: str, bars: int, decision_timestamp: datetime | None = None) -> float:
        if bars <= 0:
            raise RuntimeError("SMA_BARS must be greater than zero.")

        intraday_bars = self._get_intraday_bars(symbol, bars, decision_timestamp=decision_timestamp)
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

    def _build_base_estimator(self) -> Any:
        _, GradientBoostingClassifier, LogisticRegression, Pipeline, StandardScaler = (
            _get_sklearn_components()
        )
        if self.config.ml_model_type == "gradient_boosting":
            return GradientBoostingClassifier(
                n_estimators=150,
                learning_rate=0.05,
                max_depth=2,
                random_state=42,
            )
        return Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                (
                    "model",
                    LogisticRegression(
                        C=0.5,
                        class_weight="balanced",
                        max_iter=1000,
                        random_state=42,
                    ),
                ),
            ]
        )

    def _has_both_classes(self, labels: list[int]) -> bool:
        return len(set(labels)) >= 2

    def _fit_calibrated_model(
        self,
        train_features: list[list[float]],
        train_labels: list[int],
        calibration_features: list[list[float]],
        calibration_labels: list[int],
    ) -> tuple[Any, str]:
        if not self._has_both_classes(train_labels):
            raise RuntimeError("Training split does not contain both classes.")

        CalibratedClassifierCV, _, _, _, _ = _get_sklearn_components()
        estimator = self._build_base_estimator()
        estimator.fit(train_features, train_labels)
        if not calibration_features or not self._has_both_classes(calibration_labels):
            return estimator, self.config.ml_model_type

        calibrator = CalibratedClassifierCV(estimator, method="sigmoid", cv="prefit")
        calibrator.fit(calibration_features, calibration_labels)
        return calibrator, f"{self.config.ml_model_type}+sigmoid_calibrated"

    def _split_ml_datasets(
        self,
        feature_rows: list[list[float]],
        labels: list[int],
    ) -> tuple[list[list[float]], list[int], list[list[float]], list[int], list[list[float]], list[int]]:
        row_count = len(feature_rows)
        validation_fraction = min(max(self.config.ml_validation_fraction, 0.1), 0.3)
        validation_rows = max(20, int(row_count * validation_fraction))
        calibration_rows = max(20, int(row_count * validation_fraction))
        training_rows = row_count - validation_rows - calibration_rows

        if training_rows < 40:
            raise RuntimeError(
                f"Not enough ML rows for train/calibration/validation split: got {row_count}"
            )

        train_end = training_rows
        calibration_end = train_end + calibration_rows
        return (
            feature_rows[:train_end],
            labels[:train_end],
            feature_rows[train_end:calibration_end],
            labels[train_end:calibration_end],
            feature_rows[calibration_end:],
            labels[calibration_end:],
        )

    def _f1_score(self, true_labels: list[int], predicted_labels: list[int], positive_label: int) -> float:
        true_positive = 0
        false_positive = 0
        false_negative = 0

        for truth, predicted in zip(true_labels, predicted_labels):
            if predicted == positive_label and truth == positive_label:
                true_positive += 1
            elif predicted == positive_label and truth != positive_label:
                false_positive += 1
            elif predicted != positive_label and truth == positive_label:
                false_negative += 1

        precision = true_positive / max(1, true_positive + false_positive)
        recall = true_positive / max(1, true_positive + false_negative)
        if precision + recall == 0:
            return 0.0
        return 2.0 * precision * recall / (precision + recall)

    def _select_validation_thresholds(
        self,
        probabilities_up: list[float],
        labels: list[int],
    ) -> tuple[float, float]:
        if len(probabilities_up) < 20:
            return self.config.ml_probability_buy, self.config.ml_probability_sell

        buy_thresholds = [round(value, 2) for value in [0.50, 0.52, 0.54, 0.56, 0.58, 0.60, 0.62, 0.64, 0.66, 0.68, 0.70]]
        sell_thresholds = [round(value, 2) for value in [0.30, 0.32, 0.34, 0.36, 0.38, 0.40, 0.42, 0.44, 0.46, 0.48, 0.50]]

        best_buy_threshold = self.config.ml_probability_buy
        best_sell_threshold = self.config.ml_probability_sell
        best_buy_score = -1.0
        best_sell_score = -1.0

        for threshold in buy_thresholds:
            predicted = [1 if probability >= threshold else 0 for probability in probabilities_up]
            score = self._f1_score(labels, predicted, positive_label=1)
            acted = sum(predicted)
            if score > best_buy_score or (math.isclose(score, best_buy_score) and acted > 0):
                best_buy_score = score
                best_buy_threshold = threshold

        for threshold in sell_thresholds:
            predicted = [0 if probability <= threshold else 1 for probability in probabilities_up]
            score = self._f1_score(labels, predicted, positive_label=0)
            acted = sum(1 for probability in probabilities_up if probability <= threshold)
            if score > best_sell_score or (math.isclose(score, best_sell_score) and acted > 0):
                best_sell_score = score
                best_sell_threshold = threshold

        if best_sell_threshold > best_buy_threshold:
            midpoint = (best_buy_threshold + best_sell_threshold) / 2.0
            best_sell_threshold = min(best_sell_threshold, midpoint)
            best_buy_threshold = max(best_buy_threshold, midpoint)

        return best_buy_threshold, best_sell_threshold

    def _get_or_train_ml_model(self, symbol: str) -> TrainedSignalModel:
        cached_model = self._model_cache.get(symbol)
        if cached_model is not None and (time.time() - cached_model.trained_at_unix) < self.config.ml_train_every_seconds:
            return cached_model

        bars_needed = max(self.config.ml_lookback_bars, 80)
        decision_timestamp = self.get_decision_timestamp()
        intraday_bars = self._get_intraday_bars(
            symbol,
            bars_needed,
            decision_timestamp=decision_timestamp,
        )
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

        (
            train_features,
            train_labels,
            calibration_features,
            calibration_labels,
            validation_features,
            validation_labels,
        ) = self._split_ml_datasets(feature_rows, labels)
        model_estimator, model_name = self._fit_calibrated_model(
            train_features,
            train_labels,
            calibration_features,
            calibration_labels,
        )
        validation_probabilities = [
            float(probability_row[1])
            for probability_row in cast(list[list[float]], model_estimator.predict_proba(validation_features))
        ]
        buy_threshold, sell_threshold = self._select_validation_thresholds(
            validation_probabilities,
            validation_labels,
        )

        model = TrainedSignalModel(
            estimator=model_estimator,
            trained_at_unix=time.time(),
            training_rows=len(train_features) + len(calibration_features),
            feature_names=FEATURE_NAMES,
            buy_threshold=buy_threshold,
            sell_threshold=sell_threshold,
            validation_rows=len(validation_features),
            model_name=model_name,
        )
        self._model_cache[symbol] = model
        return model

    def get_ml_signal(self, symbol: str, decision_timestamp: datetime | None = None) -> MlSignal:
        model = self._get_or_train_ml_model(symbol)
        bars_needed = max(self.config.sma_bars, 25)
        aligned_decision_timestamp = decision_timestamp or self.get_decision_timestamp()
        intraday_bars = self._get_intraday_bars(
            symbol,
            bars_needed,
            decision_timestamp=aligned_decision_timestamp,
        )
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
            buy_threshold=model.buy_threshold,
            sell_threshold=model.sell_threshold,
            validation_rows=model.validation_rows,
            model_name=model.model_name,
        )

    def evaluate_symbol(
        self,
        symbol: str,
        holding: bool,
        decision_timestamp: datetime | None = None,
    ) -> SymbolEvaluation:
        aligned_decision_timestamp = decision_timestamp or self.get_decision_timestamp()
        bars_needed = max(self.config.sma_bars, 25)
        intraday_bars = self._get_intraday_bars(
            symbol,
            bars_needed,
            decision_timestamp=aligned_decision_timestamp,
        )
        closes = [float(bar.close) for bar in intraday_bars]
        if len(closes) < max(20, self.config.sma_bars):
            raise RuntimeError(
                f"Not enough completed bars for {symbol} at {aligned_decision_timestamp.isoformat()}: got {len(closes)}"
            )
        latest_bar_close = self._latest_bar_close_time(intraday_bars)
        bar_delay_seconds = max(0.0, (aligned_decision_timestamp - latest_bar_close).total_seconds())
        if bar_delay_seconds > self.config.max_data_delay_seconds:
            raise RuntimeError(
                f"Stale completed bars for {symbol}: latest close {latest_bar_close.isoformat()}"
            )

        price = closes[-1]
        sma = sum(closes[-self.config.sma_bars :]) / self.config.sma_bars
        ml_signal = self.get_ml_signal(symbol, decision_timestamp=aligned_decision_timestamp)
        action = self.strategy.decide_action(price, sma, ml_signal, holding)
        return SymbolEvaluation(
            price=price,
            sma=sma,
            ml_signal=ml_signal,
            action=action,
            latest_bar_close_utc=latest_bar_close.isoformat(),
        )

    def decide(
        self,
        symbol: str,
        positions: dict[str, Position],
        decision_timestamp: datetime | None = None,
    ) -> str:
        holding = symbol in positions
        return self.evaluate_symbol(symbol, holding, decision_timestamp=decision_timestamp).action

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
        decision_timestamp = self.get_decision_timestamp()

        for symbol in self.config.symbols:
            position = positions.get(symbol)
            try:
                holding = position is not None
                evaluation = self.evaluate_symbol(symbol, holding, decision_timestamp=decision_timestamp)
                quantity = float(position.qty) if position is not None else 0.0
                market_value = float(position.market_value) if position is not None else 0.0

                symbols.append(
                    SymbolSnapshot(
                        symbol=symbol,
                        price=evaluation.price,
                        sma=evaluation.sma,
                        action=evaluation.action,
                        holding=holding,
                        quantity=quantity,
                        market_value=market_value,
                        ml_probability_up=evaluation.ml_signal.probability_up,
                        ml_confidence=evaluation.ml_signal.confidence,
                        ml_training_rows=evaluation.ml_signal.training_rows,
                        ml_buy_threshold=evaluation.ml_signal.buy_threshold,
                        ml_sell_threshold=evaluation.ml_signal.sell_threshold,
                        ml_model_name=evaluation.ml_signal.model_name,
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
            timestamp_utc=decision_timestamp.isoformat(),
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

    def get_open_orders(self) -> list[Any]:
        request = GetOrdersRequest(status=QueryOrderStatus.OPEN, nested=False)
        return cast(list[Any], cast(Any, self.trading).get_orders(filter=request))

    def _count_recent_orders(self, window_seconds: int = 60) -> int:
        orders = self.get_recent_orders(limit=100)
        now = datetime.now(timezone.utc)
        count = 0
        for order in orders:
            if not order.submitted_at or order.submitted_at == "None":
                continue
            raw_timestamp = order.submitted_at
            if raw_timestamp.endswith("Z"):
                raw_timestamp = raw_timestamp[:-1] + "+00:00"
            try:
                submitted_at = datetime.fromisoformat(raw_timestamp)
            except ValueError:
                continue
            if submitted_at.tzinfo is None:
                submitted_at = submitted_at.replace(tzinfo=timezone.utc)
            else:
                submitted_at = submitted_at.astimezone(timezone.utc)
            if (now - submitted_at).total_seconds() <= window_seconds:
                count += 1
        return count

    def _is_price_collar_breached(self, decision_price: float, live_price: float) -> bool:
        if decision_price <= 0:
            return True
        deviation_bps = abs((live_price / decision_price) - 1.0) * 10000.0
        return deviation_bps > self.config.max_price_deviation_bps

    def _is_symbol_exposure_exceeded(
        self,
        symbol: str,
        live_price: float,
        positions: dict[str, Position],
    ) -> bool:
        existing_position_value = 0.0
        if symbol in positions and getattr(positions[symbol], "market_value", None) is not None:
            existing_position_value = abs(float(positions[symbol].market_value))
        proposed_qty = int(self.config.max_usd_per_trade // live_price)
        proposed_value = proposed_qty * live_price
        return (existing_position_value + proposed_value) > self.config.max_symbol_exposure_usd

    def flatten_positions(self, positions: dict[str, Position], open_order_symbols: set[str]) -> None:
        for symbol, position in positions.items():
            if symbol in open_order_symbols:
                print(f"Skip flatten {symbol}: existing open order in flight")
                continue
            try:
                self.place_market_sell(symbol, position)
            except Exception as exc:
                print(f"Flatten {symbol} ERROR: {exc}")

    def record_state(self, snapshot: BotSnapshot, orders_limit: int = 20) -> list[OrderSnapshot]:
        orders = self.get_recent_orders(limit=orders_limit)
        self.storage.save_snapshot(snapshot, orders)
        return orders

    def capture_state(self, orders_limit: int = 20) -> tuple[BotSnapshot, list[OrderSnapshot]]:
        snapshot = self.build_snapshot()
        orders = self.record_state(snapshot, orders_limit=orders_limit)
        return snapshot, orders

    def place_market_buy(
        self,
        symbol: str,
        buying_power_available: float | None = None,
        price: float | None = None,
    ) -> Any | None:
        execution_price = price if price is not None else self.get_latest_price(symbol)
        available_buying_power = buying_power_available
        if available_buying_power is None:
            account = self.get_account()
            available_buying_power = float(account.buying_power)

        trade_budget = min(self.config.max_usd_per_trade, available_buying_power)
        if trade_budget < execution_price:
            print(
                f"Skip {symbol}: available buying power ${available_buying_power:.2f} "
                f"cannot fund one share at {execution_price:.2f}"
            )
            return None

        qty = int(trade_budget // execution_price)
        if qty <= 0:
            print(
                f"Skip {symbol}: price {execution_price:.2f} is above ${self.config.max_usd_per_trade:.2f}"
            )
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
        print(f"Submitted BUY {symbol} qty={qty} approx=${qty * execution_price:.2f}")
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

    def _et_now(self) -> datetime:
        return datetime.now(_ET)

    def _is_in_entry_window(self, now_et: datetime | None = None) -> bool:
        t = (now_et or self._et_now()).time()
        return _SESSION_ENTRY_START <= t <= _SESSION_ENTRY_END

    def _is_past_flatten_deadline(self, now_et: datetime | None = None) -> bool:
        t = (now_et or self._et_now()).time()
        return t >= _SESSION_FLATTEN_AT

    def run_once(self, execute_orders: bool = True) -> BotSnapshot:
        print("\n=== BOT TICK ===")
        now_et = self._et_now()
        snapshot = self.build_snapshot()
        print(f"Connected. Cash: {snapshot.cash} Buying power: {snapshot.buying_power}")
        print(f"Daily PnL: {snapshot.daily_pnl:.2f}")
        print(f"Strategy mode: {self.config.strategy_mode}")

        if execute_orders and self._is_past_flatten_deadline(now_et):
            print(f"EOD flatten: {now_et.strftime('%H:%M:%S')} ET >= 15:55, closing all positions")
            open_orders = self.get_open_orders()
            open_order_symbols = {
                str(getattr(order, "symbol", ""))
                for order in open_orders
                if getattr(order, "symbol", None)
            }
            self.flatten_positions(snapshot.positions, open_order_symbols)
            snapshot = self.build_snapshot()
            self.record_state(snapshot)
            return snapshot

        if snapshot.kill_switch_triggered:
            print("Kill switch triggered.")
            if execute_orders:
                open_orders = self.get_open_orders()
                open_order_symbols = {
                    str(getattr(order, "symbol", ""))
                    for order in open_orders
                    if getattr(order, "symbol", None)
                }
                self.flatten_positions(snapshot.positions, open_order_symbols)
                snapshot = self.build_snapshot()
            self.record_state(snapshot)
            return snapshot

        if not execute_orders:
            for item in snapshot.symbols:
                suffix = f" ml_up={item.ml_probability_up:.3f}" if item.ml_probability_up is not None else ""
                error_suffix = f" ERROR: {item.error}" if item.error else ""
                print(f"{item.symbol} -> {item.action}{suffix}{error_suffix}")
            self.record_state(snapshot)
            return snapshot

        in_entry_window = self._is_in_entry_window(now_et)
        if not in_entry_window:
            print(f"Outside entry window ({now_et.strftime('%H:%M:%S')} ET): new entries suppressed, exits still active")

        positions = snapshot.positions.copy()
        open_positions = len(positions)
        remaining_buying_power = snapshot.buying_power
        open_orders = self.get_open_orders()
        open_order_symbols = {
            str(getattr(order, "symbol", ""))
            for order in open_orders
            if getattr(order, "symbol", None)
        }
        recent_order_count = self._count_recent_orders(window_seconds=60)

        for item in snapshot.symbols:
            if self._is_past_flatten_deadline():
                print(f"EOD flatten triggered mid-cycle at {self._et_now().strftime('%H:%M:%S')} ET")
                self.flatten_positions(snapshot.positions, open_order_symbols)
                break

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

                if symbol in open_order_symbols:
                    print(f"Skip {symbol}: existing open order in flight")
                    continue

                if recent_order_count >= self.config.max_orders_per_minute:
                    print("Skip new orders: max order rate reached")
                    break

                if action == "BUY":
                    if not in_entry_window:
                        print(f"Skip {symbol} BUY: outside trading window ({now_et.strftime('%H:%M:%S')} ET)")
                        continue
                    if symbol in positions:
                        print(f"Already holding {symbol}")
                        continue
                    if open_positions >= self.config.max_open_positions:
                        print("Max positions reached")
                        continue
                    live_price, live_price_age = self.get_latest_price_with_age(symbol)
                    if live_price_age > self.config.max_live_price_age_seconds:
                        print(f"Skip {symbol}: stale live price age {live_price_age:.1f}s")
                        continue
                    if self._is_price_collar_breached(item.price or 0.0, live_price):
                        print(
                            f"Skip {symbol}: live price {live_price:.2f} breaches collar vs decision price {item.price:.2f}"
                        )
                        continue
                    if self._is_symbol_exposure_exceeded(symbol, live_price, positions):
                        print(f"Skip {symbol}: max symbol exposure would be exceeded")
                        continue
                    order = self.place_market_buy(
                        symbol,
                        buying_power_available=remaining_buying_power,
                        price=live_price,
                    )
                    if order is not None:
                        open_positions += 1
                        recent_order_count += 1
                        estimated_cost = int(self.config.max_usd_per_trade // live_price) * live_price
                        remaining_buying_power = max(0.0, remaining_buying_power - estimated_cost)

                elif action == "SELL" and symbol in positions:
                    live_price, live_price_age = self.get_latest_price_with_age(symbol)
                    if live_price_age > self.config.max_live_price_age_seconds:
                        print(f"Skip {symbol}: stale live price age {live_price_age:.1f}s")
                        continue
                    if self._is_price_collar_breached(item.price or 0.0, live_price):
                        print(
                            f"Skip {symbol}: live price {live_price:.2f} breaches collar vs decision price {item.price:.2f}"
                        )
                        continue
                    order = self.place_market_sell(symbol, positions[symbol])
                    if order is not None:
                        open_positions = max(0, open_positions - 1)
                        recent_order_count += 1

            except Exception as exc:
                print(f"{symbol} ERROR: {exc}")

        snapshot = self.build_snapshot()
        self.record_state(snapshot)
        return snapshot


def main() -> None:
    load_dotenv(Path.cwd() / ".env")
    config = load_config()
    bot = AlpacaTradingBot(config)
    execute_orders = os.getenv("EXECUTE_ORDERS", "true").lower() != "false"
    bar_interval_seconds = config.bar_timeframe_minutes * 60
    shutdown_event = threading.Event()

    def _eod_flatten_worker() -> None:
        now_et = bot._et_now()
        target_et = now_et.replace(hour=15, minute=55, second=0, microsecond=0)
        delay = (target_et - now_et).total_seconds()
        if delay <= 0:
            print("EOD flatten thread: already past 15:55 ET, signalling shutdown")
            shutdown_event.set()
            return
        print(f"EOD flatten thread: scheduled in {delay:.0f}s at 15:55 ET")
        # Wait until 15:55, but wake up early if the main loop shuts down first
        shutdown_event.wait(timeout=delay)
        if shutdown_event.is_set():
            # Kill switch or other early exit already handled the flatten
            return
        print(f"EOD flatten thread firing at {bot._et_now().strftime('%H:%M:%S')} ET")
        if execute_orders:
            try:
                positions = bot.get_positions_by_symbol()
                open_orders = bot.get_open_orders()
                open_order_symbols = {
                    str(getattr(order, "symbol", ""))
                    for order in open_orders
                    if getattr(order, "symbol", None)
                }
                bot.flatten_positions(positions, open_order_symbols)
            except Exception as exc:
                print(f"EOD flatten thread ERROR: {exc}")
        else:
            print("EOD flatten thread: execute_orders=False, skipping flatten")
        shutdown_event.set()

    flatten_thread = threading.Thread(target=_eod_flatten_worker, name="eod-flatten", daemon=True)
    flatten_thread.start()

    while not shutdown_event.is_set():
        if bot._is_past_flatten_deadline():
            print("Main loop: past 15:55 ET, exiting without new tick")
            break

        tick_start = time.time()
        try:
            snapshot = bot.run_once(execute_orders=execute_orders)
            if snapshot.kill_switch_triggered:
                print("Main loop: kill switch triggered, exiting")
                shutdown_event.set()
                break
        except Exception as exc:
            print(f"run_once ERROR: {exc}")

        if shutdown_event.is_set():
            break

        elapsed = time.time() - tick_start
        sleep_seconds = max(0.0, bar_interval_seconds - elapsed)
        print(f"Next tick in {sleep_seconds:.0f}s")
        # Wait for the next bar, but wake immediately if flatten thread fires
        shutdown_event.wait(timeout=sleep_seconds)

    print("Bot loop exited.")


if __name__ == "__main__":
    main()

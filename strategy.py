from dataclasses import dataclass


@dataclass(frozen=True)
class StrategyConfig:
    strategy_mode: str
    ml_probability_buy: float
    ml_probability_sell: float
    entry_threshold_pct: float = 0.0


@dataclass(frozen=True)
class MlSignal:
    probability_up: float
    confidence: float
    training_rows: int
    model_age_seconds: float
    feature_names: tuple[str, ...]
    buy_threshold: float
    sell_threshold: float
    validation_rows: int
    model_name: str


class Strategy:
    def __init__(self, config: StrategyConfig) -> None:
        self.config = config

    def decide_action(self, price: float, sma: float, ml_signal: MlSignal, holding: bool) -> str:
        mode = self.config.strategy_mode

        if mode == "sma":
            threshold_price = sma * (1 + self.config.entry_threshold_pct)
            if price > threshold_price and not holding:
                return "BUY"
            if price < sma and holding:
                return "SELL"
            return "HOLD"

        if mode == "ml":
            if ml_signal.probability_up >= ml_signal.buy_threshold and not holding:
                return "BUY"
            if ml_signal.probability_up <= ml_signal.sell_threshold and holding:
                return "SELL"
            return "HOLD"

        # hybrid mode: require SMA trend confirmation for buys, allow either risk signal for sells
        if (
            price > sma
            and ml_signal.probability_up >= ml_signal.buy_threshold
            and not holding
        ):
            return "BUY"
        if holding and (
            price < sma or ml_signal.probability_up <= ml_signal.sell_threshold
        ):
            return "SELL"
        return "HOLD"
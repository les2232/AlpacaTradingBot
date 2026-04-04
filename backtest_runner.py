import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from strategy import MlSignal, Strategy, StrategyConfig


# ---------------------------------------------------------------------------
# ML helpers — mirrors trading_bot.py so the backtest uses the same feature
# engineering and training pipeline as the live bot.
# ---------------------------------------------------------------------------

@dataclass
class _MlModel:
    estimator: Any
    buy_threshold: float
    sell_threshold: float

    def predict_up(self, features: list[float]) -> float:
        return float(self.estimator.predict_proba([features])[0][1])


def _mean(values: list[float]) -> float:
    return sum(values) / len(values)


def _stddev(values: list[float], mean_val: float) -> float:
    variance = sum((v - mean_val) ** 2 for v in values) / max(1, len(values))
    return math.sqrt(max(variance, 1e-12))


def _build_feature_vector(closes: list[float], volumes: list[float], idx: int) -> list[float]:
    """Identical feature set to trading_bot.py._build_feature_vector."""
    price = closes[idx]
    ret_1 = (price / closes[idx - 1]) - 1.0
    ret_3 = (price / closes[idx - 3]) - 1.0
    ret_5 = (price / closes[idx - 5]) - 1.0
    sma_10 = _mean(closes[idx - 9: idx + 1])
    sma_20 = _mean(closes[idx - 19: idx + 1])
    vol_returns = [
        (closes[j] / closes[j - 1]) - 1.0
        for j in range(idx - 9, idx + 1)
        if j - 1 >= 0
    ]
    avg_vol = _mean(vol_returns)
    avg_volume_10 = _mean(volumes[idx - 9: idx + 1])
    return [
        ret_1, ret_3, ret_5,
        (price / sma_10) - 1.0,
        (price / sma_20) - 1.0,
        _stddev(vol_returns, avg_vol),
        (volumes[idx] / max(avg_volume_10, 1.0)) - 1.0,
    ]


def _f1_score(true_labels: list[int], pred_labels: list[int], pos: int) -> float:
    tp = fp = fn = 0
    for t, p in zip(true_labels, pred_labels):
        if p == pos and t == pos:
            tp += 1
        elif p == pos:
            fp += 1
        elif t == pos:
            fn += 1
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    if precision + recall == 0:
        return 0.0
    return 2.0 * precision * recall / (precision + recall)


def _select_thresholds(
    probs: list[float],
    labels: list[int],
    default_buy: float,
    default_sell: float,
) -> tuple[float, float]:
    """Identical threshold-selection logic to trading_bot.py._select_validation_thresholds."""
    if len(probs) < 20:
        return default_buy, default_sell

    buy_candidates  = [round(v, 2) for v in [0.50, 0.52, 0.54, 0.56, 0.58, 0.60, 0.62, 0.64, 0.66, 0.68, 0.70]]
    sell_candidates = [round(v, 2) for v in [0.30, 0.32, 0.34, 0.36, 0.38, 0.40, 0.42, 0.44, 0.46, 0.48, 0.50]]

    best_buy, best_sell = default_buy, default_sell
    best_buy_score = best_sell_score = -1.0

    for thr in buy_candidates:
        predicted = [1 if p >= thr else 0 for p in probs]
        score = _f1_score(labels, predicted, 1)
        if score > best_buy_score or (math.isclose(score, best_buy_score) and sum(predicted) > 0):
            best_buy_score = score
            best_buy = thr

    for thr in sell_candidates:
        predicted = [0 if p <= thr else 1 for p in probs]
        score = _f1_score(labels, predicted, 0)
        if score > best_sell_score or (math.isclose(score, best_sell_score) and
                                       sum(1 for p in probs if p <= thr) > 0):
            best_sell_score = score
            best_sell = thr

    if best_sell > best_buy:
        mid = (best_buy + best_sell) / 2.0
        best_sell = min(best_sell, mid)
        best_buy = max(best_buy, mid)

    return best_buy, best_sell


def _train_ml_model(
    closes: list[float],
    volumes: list[float],
    current_idx: int,
    ml_lookback_bars: int,
    ml_probability_buy: float,
    ml_probability_sell: float,
) -> "_MlModel | None":
    """
    Train a logistic classifier on bars ending just before current_idx.

    Training window : [max(0, current_idx - ml_lookback_bars), current_idx - 1]
    Label for bar i : 1 if closes[i+1] > closes[i] else 0
    Last label      : closes[current_idx] > closes[current_idx-1] — the current
                      completed bar's close, so no look-ahead.

    Exactly mirrors _get_or_train_ml_model in trading_bot.py.
    """
    try:
        from sklearn.calibration import CalibratedClassifierCV
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
    except ImportError:
        return None

    start_i = max(0, current_idx - ml_lookback_bars)
    feature_rows: list[list[float]] = []
    labels: list[int] = []

    for i in range(max(start_i, 19), current_idx):
        feature_rows.append(_build_feature_vector(closes, volumes, i))
        labels.append(1 if closes[i + 1] > closes[i] else 0)

    if len(feature_rows) < 80 or len(set(labels)) < 2:
        return None

    val_rows = max(20, int(len(feature_rows) * 0.2))
    cal_rows = max(20, int(len(feature_rows) * 0.2))
    train_rows = len(feature_rows) - val_rows - cal_rows
    if train_rows < 40:
        return None

    X_train = feature_rows[:train_rows]
    y_train = labels[:train_rows]
    X_cal   = feature_rows[train_rows: train_rows + cal_rows]
    y_cal   = labels[train_rows: train_rows + cal_rows]
    X_val   = feature_rows[train_rows + cal_rows:]
    y_val   = labels[train_rows + cal_rows:]

    if len(set(y_train)) < 2:
        return None

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(C=0.5, class_weight="balanced", max_iter=1000, random_state=42)),
    ])
    pipe.fit(X_train, y_train)

    if X_cal and len(set(y_cal)) >= 2:
        cal = CalibratedClassifierCV(pipe, method="sigmoid", cv="prefit")
        cal.fit(X_cal, y_cal)
        estimator: Any = cal
    else:
        estimator = pipe

    val_probs = [float(row[1]) for row in estimator.predict_proba(X_val)]
    buy_thr, sell_thr = _select_thresholds(val_probs, list(y_val), ml_probability_buy, ml_probability_sell)

    return _MlModel(estimator=estimator, buy_threshold=buy_thr, sell_threshold=sell_thr)


# ---------------------------------------------------------------------------
# Dataset I/O
# ---------------------------------------------------------------------------

def load_dataset(dataset_path: Path) -> tuple[pd.DataFrame, dict]:
    bars_path = dataset_path / "bars.parquet"
    manifest_path = dataset_path / "manifest.json"
    df = pd.read_parquet(bars_path)
    with open(manifest_path) as f:
        manifest = json.load(f)
    return df, manifest


def _build_performance_summary(
    starting_capital: float,
    final_equity: float,
    max_drawdown_pct: float,
    total_trades: int,
    winning_pnls: list[float],
    losing_pnls: list[float],
    realized_pnl: float,
    final_cash: float,
) -> dict[str, float | int]:
    closed = winning_pnls + losing_pnls
    return {
        "starting_capital": starting_capital,
        "final_equity": final_equity,
        "total_return_pct": ((final_equity - starting_capital) / starting_capital * 100) if starting_capital > 0 else 0.0,
        "max_drawdown_pct": max_drawdown_pct,
        "total_trades": total_trades,
        "win_rate": len(winning_pnls) / max(1, len(closed)) * 100,
        "avg_winning_trade": sum(winning_pnls) / len(winning_pnls) if winning_pnls else 0.0,
        "avg_losing_trade": sum(losing_pnls) / len(losing_pnls) if losing_pnls else 0.0,
        "realized_pnl": realized_pnl,
        "final_cash": final_cash,
    }


# ---------------------------------------------------------------------------
# Core backtest loop
# ---------------------------------------------------------------------------

def run_backtest(
    dataset_path: Path,
    symbols: list[str] | None = None,
    sma_bars: int = 20,
    commission: float = 0.0,
    slippage: float = 0.0,
    entry_threshold_pct: float = 0.0,
    starting_capital: float = 10000.0,
    position_size: float = 1000.0,
    start_date: str | None = None,
    end_date: str | None = None,
    strategy_mode: str = "hybrid",
    ml_lookback_bars: int = 320,
    ml_retrain_every_bars: int = 15,
    ml_probability_buy: float = 0.55,
    ml_probability_sell: float = 0.45,
) -> dict:
    df, manifest = load_dataset(dataset_path)
    df = df.sort_values(["symbol", "timestamp"])

    if start_date:
        start_ts = pd.Timestamp(start_date, tz="UTC")
        df = df[df["timestamp"] >= start_ts]
    if end_date:
        end_ts = pd.Timestamp(end_date, tz="UTC") + pd.Timedelta(days=1) - pd.Timedelta(microseconds=1)
        df = df[df["timestamp"] <= end_ts]

    if symbols is None:
        symbols = manifest["symbols"]

    strategy = Strategy(StrategyConfig(
        strategy_mode=strategy_mode,
        ml_probability_buy=ml_probability_buy,
        ml_probability_sell=ml_probability_sell,
        entry_threshold_pct=entry_threshold_pct,
    ))

    _dummy_ml = MlSignal(
        probability_up=0.5, confidence=0.0, training_rows=0,
        model_age_seconds=0.0, feature_names=(),
        buy_threshold=ml_probability_buy, sell_threshold=ml_probability_sell,
        validation_rows=0, model_name="dummy",
    )

    cash = starting_capital
    open_positions_value = 0.0
    skipped_trades = 0
    current_equity = starting_capital
    peak_equity = starting_capital
    max_drawdown_pct = 0.0
    per_symbol_starting_capital = starting_capital / max(1, len(symbols))

    results: dict[str, Any] = {
        "strategy_mode": strategy_mode,
        "total_trades": 0,
        "realized_pnl": 0.0,
        "starting_capital": starting_capital,
        "final_equity": starting_capital,
        "total_return_pct": 0.0,
        "symbol_trade_counts": {symbol: 0 for symbol in symbols},
        "final_positions": {},
        "trades": [],
        "winning_pnls": [],
        "losing_pnls": [],
        "max_drawdown_pct": 0.0,
        "position_size": position_size,
        "final_cash": starting_capital,
        "skipped_trades": 0,
        "max_concurrent_positions": 0,
        "total_buy_candidates": 0,
        "competing_timestamps": 0,
        "per_symbol": {},
    }
    symbol_stats: dict[str, dict[str, Any]] = {
        symbol: {
            "realized_pnl": 0.0,
            "winning_pnls": [],
            "losing_pnls": [],
            "cash": per_symbol_starting_capital,
            "open_positions_value": 0.0,
            "peak_equity": per_symbol_starting_capital,
            "max_drawdown_pct": 0.0,
            "total_trades": 0,
        }
        for symbol in symbols
    }

    # Precompute per-symbol arrays — avoids repeated pandas overhead in the hot loop
    symbols_dfs: dict[str, pd.DataFrame] = {}
    close_arrs: dict[str, list[float]] = {}
    volume_arrs: dict[str, list[float]] = {}
    for symbol in symbols:
        sdf = df[df["symbol"] == symbol].sort_values("timestamp").reset_index(drop=True)
        symbols_dfs[symbol] = sdf
        close_arrs[symbol] = sdf["close"].tolist()
        volume_arrs[symbol] = sdf["volume"].tolist()

    # Per-symbol ML state
    ml_cache: dict[str, _MlModel | None] = {s: None for s in symbols}
    ml_trained_at: dict[str, int] = {s: -999 for s in symbols}

    pointers = {symbol: 0 for symbol in symbols}
    position = {symbol: False for symbol in symbols}
    entry_price = {symbol: 0.0 for symbol in symbols}
    shares_held = {symbol: 0.0 for symbol in symbols}
    trades = []

    while any(pointers[s] < len(symbols_dfs[s]) for s in symbols):
        next_times = [
            symbols_dfs[s].iloc[pointers[s]]["timestamp"]
            for s in symbols if pointers[s] < len(symbols_dfs[s])
        ]
        if not next_times:
            break
        current_time = min(next_times)

        buy_candidates = []
        sell_actions = []

        for symbol in symbols:
            if pointers[symbol] >= len(symbols_dfs[symbol]):
                continue
            row = symbols_dfs[symbol].iloc[pointers[symbol]]
            if row["timestamp"] != current_time:
                continue

            p = pointers[symbol]
            closes = close_arrs[symbol]
            volumes = volume_arrs[symbol]

            if p < sma_bars - 1:
                pointers[symbol] += 1
                continue

            sma = _mean(closes[p - sma_bars + 1: p + 1])
            price = closes[p]

            # --- ML signal ---
            if strategy_mode in ("hybrid", "ml"):
                # Retrain every ml_retrain_every_bars bars once we have enough history.
                # p >= 99 ensures enough samples for the train/cal/val split.
                if p >= 99 and (
                    ml_cache[symbol] is None
                    or p - ml_trained_at[symbol] >= ml_retrain_every_bars
                ):
                    new_model = _train_ml_model(
                        closes, volumes, p,
                        ml_lookback_bars, ml_probability_buy, ml_probability_sell,
                    )
                    if new_model is not None:
                        ml_cache[symbol] = new_model
                        ml_trained_at[symbol] = p

                model = ml_cache[symbol]
                if model is not None and p >= 19:
                    prob = model.predict_up(_build_feature_vector(closes, volumes, p))
                    ml_signal = MlSignal(
                        probability_up=prob,
                        confidence=abs(prob - 0.5) * 2.0,
                        training_rows=0,
                        model_age_seconds=0.0,
                        feature_names=(),
                        buy_threshold=model.buy_threshold,
                        sell_threshold=model.sell_threshold,
                        validation_rows=0,
                        model_name="logistic",
                    )
                else:
                    ml_signal = _dummy_ml
            else:
                ml_signal = _dummy_ml

            action = strategy.decide_action(price, sma, ml_signal, position[symbol])

            if action == "BUY" and not position[symbol]:
                buy_candidates.append(((price - sma) / sma, symbol, row, sma))
            elif action == "SELL" and position[symbol]:
                sell_actions.append((symbol, row, sma))

            pointers[symbol] += 1

        # --- Execute sells first to free capital ---
        for symbol, row, _ in sell_actions:
            price = row["close"]
            exit_price = price - slippage - commission
            pnl = (exit_price - entry_price[symbol]) * shares_held[symbol]
            results["realized_pnl"] += pnl
            (results["winning_pnls"] if pnl > 0 else results["losing_pnls"]).append(pnl)
            symbol_stat = symbol_stats[symbol]
            symbol_stat["realized_pnl"] += pnl
            (symbol_stat["winning_pnls"] if pnl > 0 else symbol_stat["losing_pnls"]).append(pnl)
            cash += position_size + pnl
            open_positions_value -= position_size
            current_equity = cash + open_positions_value
            peak_equity = max(peak_equity, current_equity)
            if peak_equity > 0:
                max_drawdown_pct = max(max_drawdown_pct, (peak_equity - current_equity) / peak_equity * 100)
            symbol_stat["cash"] += position_size + pnl
            symbol_stat["open_positions_value"] -= position_size
            symbol_equity = symbol_stat["cash"] + symbol_stat["open_positions_value"]
            symbol_stat["peak_equity"] = max(symbol_stat["peak_equity"], symbol_equity)
            if symbol_stat["peak_equity"] > 0:
                symbol_stat["max_drawdown_pct"] = max(
                    symbol_stat["max_drawdown_pct"],
                    (symbol_stat["peak_equity"] - symbol_equity) / symbol_stat["peak_equity"] * 100,
                )
            position[symbol] = False
            trades.append({"symbol": symbol, "side": "SELL", "price": price,
                           "shares": shares_held[symbol], "timestamp": str(row["timestamp"]), "pnl": pnl})
            results["symbol_trade_counts"][symbol] += 1
            results["total_trades"] += 1
            symbol_stat["total_trades"] += 1

        # --- Execute buys sorted by distance above SMA ---
        buy_candidates.sort(reverse=True)
        results["total_buy_candidates"] += len(buy_candidates)
        if len(buy_candidates) > 1:
            results["competing_timestamps"] += 1

        for _, symbol, row, _ in buy_candidates:
            if cash >= position_size:
                price = row["close"]
                position[symbol] = True
                shares_held[symbol] = position_size / price
                entry_price[symbol] = price + slippage + commission
                cash -= position_size
                open_positions_value += position_size
                symbol_stat = symbol_stats[symbol]
                symbol_stat["cash"] -= position_size
                symbol_stat["open_positions_value"] += position_size
                current_equity = cash + open_positions_value
                results["max_concurrent_positions"] = max(
                    results["max_concurrent_positions"], sum(position.values())
                )
                symbol_equity = symbol_stat["cash"] + symbol_stat["open_positions_value"]
                symbol_stat["peak_equity"] = max(symbol_stat["peak_equity"], symbol_equity)
                trades.append({"symbol": symbol, "side": "BUY", "price": price,
                               "shares": shares_held[symbol], "timestamp": str(row["timestamp"])})
                results["symbol_trade_counts"][symbol] += 1
                results["total_trades"] += 1
                symbol_stat["total_trades"] += 1
            else:
                skipped_trades += 1

    # --- Force-close any remaining open positions at the last bar ---
    for symbol in symbols:
        if position[symbol]:
            last_row = symbols_dfs[symbol].iloc[-1]
            price = last_row["close"]
            exit_price = price - slippage - commission
            pnl = (exit_price - entry_price[symbol]) * shares_held[symbol]
            results["realized_pnl"] += pnl
            (results["winning_pnls"] if pnl > 0 else results["losing_pnls"]).append(pnl)
            symbol_stat = symbol_stats[symbol]
            symbol_stat["realized_pnl"] += pnl
            (symbol_stat["winning_pnls"] if pnl > 0 else symbol_stat["losing_pnls"]).append(pnl)
            cash += position_size + pnl
            open_positions_value -= position_size
            current_equity = cash + open_positions_value
            peak_equity = max(peak_equity, current_equity)
            if peak_equity > 0:
                max_drawdown_pct = max(max_drawdown_pct, (peak_equity - current_equity) / peak_equity * 100)
            symbol_stat["cash"] += position_size + pnl
            symbol_stat["open_positions_value"] -= position_size
            symbol_equity = symbol_stat["cash"] + symbol_stat["open_positions_value"]
            symbol_stat["peak_equity"] = max(symbol_stat["peak_equity"], symbol_equity)
            if symbol_stat["peak_equity"] > 0:
                symbol_stat["max_drawdown_pct"] = max(
                    symbol_stat["max_drawdown_pct"],
                    (symbol_stat["peak_equity"] - symbol_equity) / symbol_stat["peak_equity"] * 100,
                )
            position[symbol] = False
            trades.append({"symbol": symbol, "side": "SELL", "price": price,
                           "shares": shares_held[symbol], "timestamp": str(last_row["timestamp"]),
                           "pnl": pnl, "forced_close": True})

    results["trades"].extend(trades)
    results.update(
        _build_performance_summary(
            starting_capital=starting_capital,
            final_equity=cash + open_positions_value,
            max_drawdown_pct=max_drawdown_pct,
            total_trades=results["total_trades"],
            winning_pnls=results["winning_pnls"],
            losing_pnls=results["losing_pnls"],
            realized_pnl=results["realized_pnl"],
            final_cash=cash,
        )
    )
    results["skipped_trades"] = skipped_trades
    results["per_symbol"] = {
        symbol: {
            "symbol": symbol,
            **_build_performance_summary(
                starting_capital=per_symbol_starting_capital,
                final_equity=stats["cash"] + stats["open_positions_value"],
                max_drawdown_pct=stats["max_drawdown_pct"],
                total_trades=stats["total_trades"],
                winning_pnls=stats["winning_pnls"],
                losing_pnls=stats["losing_pnls"],
                realized_pnl=stats["realized_pnl"],
                final_cash=stats["cash"],
            ),
        }
        for symbol, stats in symbol_stats.items()
    }
    return results


# ---------------------------------------------------------------------------
# CLI helpers
# ---------------------------------------------------------------------------

def extract_clean_columns(result_dict: dict) -> dict:
    clean_cols = [
        "regime", "start_date", "end_date",
        "strategy_mode", "entry_threshold_pct", "sma_bars",
        "total_return_pct", "max_drawdown_pct",
        "total_trades", "total_buy_candidates", "skipped_trades",
        "competing_timestamps", "max_concurrent_positions",
        "win_rate", "avg_winning_trade", "avg_losing_trade",
        "realized_pnl", "final_equity", "final_cash",
        "starting_capital", "position_size",
    ]
    return {k: result_dict[k] for k in clean_cols if k in result_dict}


def _build_per_symbol_rows(result_dict: dict) -> list[dict]:
    base_cols = {
        "regime": result_dict.get("regime"),
        "start_date": result_dict.get("start_date"),
        "end_date": result_dict.get("end_date"),
        "strategy_mode": result_dict.get("strategy_mode"),
        "sma_bars": result_dict.get("sma_bars"),
        "entry_threshold_pct": result_dict.get("entry_threshold_pct"),
    }
    rows = [{
        **base_cols,
        "scope": "portfolio",
        "symbol": "COMBINED",
        "total_return_pct": result_dict["total_return_pct"],
        "max_drawdown_pct": result_dict["max_drawdown_pct"],
        "win_rate": result_dict["win_rate"],
        "avg_winning_trade": result_dict["avg_winning_trade"],
        "avg_losing_trade": result_dict["avg_losing_trade"],
        "total_trades": result_dict["total_trades"],
    }]
    for symbol, symbol_result in result_dict.get("per_symbol", {}).items():
        rows.append({
            **base_cols,
            "scope": "symbol",
            "symbol": symbol,
            "total_return_pct": symbol_result["total_return_pct"],
            "max_drawdown_pct": symbol_result["max_drawdown_pct"],
            "win_rate": symbol_result["win_rate"],
            "avg_winning_trade": symbol_result["avg_winning_trade"],
            "avg_losing_trade": symbol_result["avg_losing_trade"],
            "total_trades": symbol_result["total_trades"],
        })
    return rows


def _write_per_symbol_csv(results_list: list[dict], output_csv: str) -> Path:
    output_path = Path(output_csv)
    suffix = output_path.suffix or ".csv"
    per_symbol_path = output_path.with_name(f"{output_path.stem}_per_symbol{suffix}")
    rows: list[dict] = []
    for result in results_list:
        rows.extend(_build_per_symbol_rows(result))
    pd.DataFrame(rows).to_csv(per_symbol_path, index=False)
    return per_symbol_path


def _format_config_label(result: dict) -> str:
    parts = [f"SMA={result['sma_bars']}", f"threshold={result['entry_threshold_pct']:.4f}"]
    if result.get("regime"):
        parts.append(f"regime={result['regime']}")
    return ", ".join(parts)


def _group_results_by_config(results_list: list[dict]) -> dict[tuple[int, float], list[dict]]:
    grouped: dict[tuple[int, float], list[dict]] = {}
    for result in results_list:
        key = (result["sma_bars"], result["entry_threshold_pct"])
        grouped.setdefault(key, []).append(result)
    return grouped


def _print_post_run_summary(results_list: list[dict]) -> None:
    if not results_list:
        return

    print("\nPost-Run Summary:")

    best_return = max(results_list, key=lambda r: r["total_return_pct"])
    print(
        f"Best Return:            {_format_config_label(best_return)} "
        f"({best_return['total_return_pct']:.2f}%)"
    )

    best_drawdown = min(results_list, key=lambda r: r["max_drawdown_pct"])
    print(
        f"Lowest Max Drawdown:    {_format_config_label(best_drawdown)} "
        f"({best_drawdown['max_drawdown_pct']:.1f}%)"
    )

    eligible_win_rate = [r for r in results_list if r["total_trades"] >= 10]
    if eligible_win_rate:
        best_win_rate = max(eligible_win_rate, key=lambda r: r["win_rate"])
        print(
            f"Best Win Rate:          {_format_config_label(best_win_rate)} "
            f"({best_win_rate['win_rate']:.1f}% with {best_win_rate['total_trades']} trades)"
        )
    else:
        print("Best Win Rate:          No config met the 10-trade minimum")

    regime_names = {result.get("regime") for result in results_list if result.get("regime")}
    if len(regime_names) > 1:
        consistent_candidates: list[tuple[float, int, float, list[dict]]] = []
        for (_, _), config_results in _group_results_by_config(results_list).items():
            config_regimes = {result.get("regime") for result in config_results}
            if config_regimes == regime_names:
                avg_return = sum(result["total_return_pct"] for result in config_results) / len(config_results)
                total_trades = sum(result["total_trades"] for result in config_results)
                avg_drawdown = sum(result["max_drawdown_pct"] for result in config_results) / len(config_results)
                consistent_candidates.append((avg_return, total_trades, avg_drawdown, config_results))

        if consistent_candidates:
            _, _, _, best_consistent_results = max(
                consistent_candidates,
                key=lambda item: (item[0], item[1], -item[2]),
            )
            sample = best_consistent_results[0]
            avg_return = sum(result["total_return_pct"] for result in best_consistent_results) / len(best_consistent_results)
            print(
                f"Most Consistent:        SMA={sample['sma_bars']}, "
                f"threshold={sample['entry_threshold_pct']:.4f} "
                f"(avg return {avg_return:.2f}% across {len(best_consistent_results)} regimes)"
            )
        else:
            print("Most Consistent:        No config appeared in all regimes")

    symbol_returns: dict[str, list[float]] = {}
    for result in results_list:
        for symbol, symbol_result in result.get("per_symbol", {}).items():
            symbol_returns.setdefault(symbol, []).append(symbol_result["total_return_pct"])

    if symbol_returns:
        avg_symbol_returns = {
            symbol: sum(returns) / len(returns)
            for symbol, returns in symbol_returns.items()
        }
        worst_symbol, worst_avg_return = min(avg_symbol_returns.items(), key=lambda item: item[1])
        best_symbol, best_avg_return = max(avg_symbol_returns.items(), key=lambda item: item[1])
        print(f"Worst Symbol Overall:   {worst_symbol} ({worst_avg_return:.2f}% avg return)")
        print(f"Best Symbol Overall:    {best_symbol} ({best_avg_return:.2f}% avg return)")


def _parse_regime_specs(regime_specs: list[str] | None, start_date: str | None, end_date: str | None) -> list[dict[str, str | None]]:
    if regime_specs:
        regimes = []
        for spec in regime_specs:
            parts = [part.strip() for part in spec.split(":")]
            if len(parts) != 3 or not all(parts):
                raise ValueError(
                    "Invalid --regime format. Use NAME:YYYY-MM-DD:YYYY-MM-DD, "
                    "for example Bull:2020-04-01:2021-12-31"
                )
            name, regime_start, regime_end = parts
            regimes.append({"regime": name, "start_date": regime_start, "end_date": regime_end})
        return regimes
    return [{"regime": "Custom", "start_date": start_date, "end_date": end_date}]


def _parse_csv_list(raw_value: str, cast: type[int] | type[float]) -> list[int] | list[float]:
    values = [item.strip() for item in raw_value.split(",") if item.strip()]
    if not values:
        raise ValueError("Sweep list cannot be empty.")
    return [cast(item) for item in values]


def _print_sweep_results(results_list: list[dict], strategy_mode: str, show_regime: bool = False) -> None:
    print(f"\nSweep Results - strategy_mode={strategy_mode} (sorted by Total Return)")
    regime_header = f"{'Regime':<12}  " if show_regime else ""
    print(f"{regime_header}{'SMA':>5}  {'Thresh':>8}  {'Return':>8}  {'Max DD':>7}  {'Win%':>6}  {'Trades':>7}")
    print("-" * (72 if show_regime else 58))
    for res in results_list:
        regime_prefix = f"{res.get('regime', ''):<12}  " if show_regime else ""
        print(
            f"{regime_prefix}"
            f"{res['sma_bars']:>5}  "
            f"{res['entry_threshold_pct']:>8.4f}  "
            f"{res['total_return_pct']:>7.2f}%  "
            f"{res['max_drawdown_pct']:>6.1f}%  "
            f"{res['win_rate']:>5.1f}%  "
            f"{res['total_trades']:>7}"
        )


def _print_single_result(results: dict) -> None:
    print(f"Regime:                 {results.get('regime', 'Custom')}")
    print(f"Date Range:             {results.get('start_date') or 'start'} to {results.get('end_date') or 'end'}")
    print("Combined Portfolio:")
    print(f"Strategy Mode:          {results['strategy_mode']}")
    print(f"Position Size:          ${results['position_size']:.2f}")
    print(f"Starting Capital:       ${results['starting_capital']:.2f}")
    print(f"Final Cash:             ${results['final_cash']:.2f}")
    print(f"Final Equity:           ${results['final_equity']:.2f}")
    print(f"Total Return:           {results['total_return_pct']:.2f}%")
    print(f"Max Drawdown:           {results['max_drawdown_pct']:.1f}%")
    print(f"Total Trades:           {results['total_trades']}")
    print(f"Skipped Trades:         {results['skipped_trades']}")
    print(f"Max Concurrent Pos:     {results['max_concurrent_positions']}")
    print(f"Total BUY Candidates:   {results['total_buy_candidates']}")
    print(f"Competing Timestamps:   {results['competing_timestamps']}")
    print(f"Win Rate:               {results['win_rate']:.1f}%")
    print(f"Average Winning Trade:  ${results['avg_winning_trade']:.2f}")
    print(f"Average Losing Trade:   ${results['avg_losing_trade']:.2f}")
    print("Symbol Trade Counts:")
    for symbol, count in results["symbol_trade_counts"].items():
        print(f"  {symbol}: {count}")
    print("Final Positions: All flat (forced close at end)")
    print("\nPer-Symbol Results:")
    for symbol, symbol_result in results.get("per_symbol", {}).items():
        print(f"\n{symbol}:")
        print(f"  Return:               {symbol_result['total_return_pct']:.2f}%")
        print(f"  Max Drawdown:         {symbol_result['max_drawdown_pct']:.1f}%")
        print(f"  Win Rate:             {symbol_result['win_rate']:.1f}%")
        print(f"  Average Winning Trade:${symbol_result['avg_winning_trade']:.2f}")
        print(f"  Average Losing Trade: ${symbol_result['avg_losing_trade']:.2f}")
        print(f"  Total Trades:         {symbol_result['total_trades']}")


def _print_regime_results(results_list: list[dict]) -> None:
    for idx, result in enumerate(results_list):
        if idx > 0:
            print("\n" + "=" * 72)
        _print_single_result(result)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a backtest on a historical dataset.")
    parser.add_argument("--dataset", required=True, help="Path to dataset directory")
    parser.add_argument("--symbols", nargs="*", help="Symbols to backtest (default: all in manifest)")
    parser.add_argument("--strategy-mode", default="hybrid", choices=["sma", "ml", "hybrid"],
                        help="Strategy mode (default: hybrid)")
    parser.add_argument("--sma-bars", type=int, default=20)
    parser.add_argument("--sma-bars-list", help="Comma-separated SMA values to sweep, e.g. 10,20,30,50")
    parser.add_argument("--entry-threshold-pct", type=float, default=0.0)
    parser.add_argument("--entry-threshold-pct-list",
                        help="Comma-separated entry thresholds to sweep")
    parser.add_argument("--ml-lookback-bars", type=int, default=320)
    parser.add_argument("--ml-retrain-every-bars", type=int, default=15)
    parser.add_argument("--ml-probability-buy", type=float, default=0.55)
    parser.add_argument("--ml-probability-sell", type=float, default=0.45)
    parser.add_argument("--commission-per-order", type=float, default=0.0)
    parser.add_argument("--slippage-per-share", type=float, default=0.0)
    parser.add_argument("--starting-capital", type=float, default=10000.0)
    parser.add_argument("--position-size", type=float, default=1000.0)
    parser.add_argument("--output-csv", help="CSV output path for sweep results")
    parser.add_argument("--start-date")
    parser.add_argument("--end-date")
    parser.add_argument(
        "--regime",
        action="append",
        help="Named regime in NAME:YYYY-MM-DD:YYYY-MM-DD format. Repeat for multiple ranges.",
    )

    args = parser.parse_args()
    dataset_path = Path(args.dataset)
    sma_values = _parse_csv_list(args.sma_bars_list, int) if args.sma_bars_list else [args.sma_bars]
    threshold_values = (
        _parse_csv_list(args.entry_threshold_pct_list, float)
        if args.entry_threshold_pct_list
        else [args.entry_threshold_pct]
    )
    regimes = _parse_regime_specs(args.regime, args.start_date, args.end_date)

    common_kwargs = dict(
        commission=args.commission_per_order,
        slippage=args.slippage_per_share,
        starting_capital=args.starting_capital,
        position_size=args.position_size,
        strategy_mode=args.strategy_mode,
        ml_lookback_bars=args.ml_lookback_bars,
        ml_retrain_every_bars=args.ml_retrain_every_bars,
        ml_probability_buy=args.ml_probability_buy,
        ml_probability_sell=args.ml_probability_sell,
    )

    if args.sma_bars_list or args.entry_threshold_pct_list or args.regime:
        results_list = []
        for regime in regimes:
            for sma in sma_values:
                for threshold in threshold_values:
                    print(
                        f"Running regime={regime['regime']}, SMA={sma}, threshold={threshold}...",
                        flush=True,
                    )
                    result = run_backtest(
                        dataset_path,
                        args.symbols,
                        sma,
                        entry_threshold_pct=threshold,
                        start_date=regime["start_date"],
                        end_date=regime["end_date"],
                        **common_kwargs,
                    )
                    result["regime"] = regime["regime"]
                    result["start_date"] = regime["start_date"]
                    result["end_date"] = regime["end_date"]
                    result["sma_bars"] = sma
                    result["entry_threshold_pct"] = threshold
                    results_list.append(result)

        results_list.sort(key=lambda x: x["total_return_pct"], reverse=True)
        if args.sma_bars_list or args.entry_threshold_pct_list:
            _print_sweep_results(results_list, args.strategy_mode, show_regime=len(regimes) > 1)
            _print_post_run_summary(results_list)
        else:
            _print_regime_results(results_list)

        if args.output_csv:
            pd.DataFrame([extract_clean_columns(r) for r in results_list]).to_csv(args.output_csv, index=False)
            print(f"\nResults written to {args.output_csv}")
            per_symbol_csv = _write_per_symbol_csv(results_list, args.output_csv)
            print(f"Per-symbol results written to {per_symbol_csv}")

    else:
        print("Running backtest...", flush=True)
        results = run_backtest(dataset_path, args.symbols, args.sma_bars,
                               entry_threshold_pct=args.entry_threshold_pct, **common_kwargs)
        results["regime"] = regimes[0]["regime"]
        results["start_date"] = regimes[0]["start_date"]
        results["end_date"] = regimes[0]["end_date"]
        results["sma_bars"] = args.sma_bars
        results["entry_threshold_pct"] = args.entry_threshold_pct
        print("\nBacktest Results:")
        _print_single_result(results)
        if args.output_csv:
            per_symbol_csv = _write_per_symbol_csv([results], args.output_csv)
            print(f"\nPer-symbol results written to {per_symbol_csv}")


if __name__ == "__main__":
    main()

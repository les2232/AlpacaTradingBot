import argparse
import json
from pathlib import Path

import pandas as pd

from strategy import MlSignal, Strategy, StrategyConfig


def load_dataset(dataset_path: Path) -> tuple[pd.DataFrame, dict]:
    bars_path = dataset_path / "bars.parquet"
    manifest_path = dataset_path / "manifest.json"

    df = pd.read_parquet(bars_path)
    with open(manifest_path) as f:
        manifest = json.load(f)

    return df, manifest


def run_backtest(dataset_path: Path, symbols: list[str] | None = None, sma_bars: int = 20, commission: float = 0.0, slippage: float = 0.0, entry_threshold_pct: float = 0.0, starting_capital: float = 10000.0, position_size: float = 1000.0, start_date: str | None = None, end_date: str | None = None) -> dict:
    df, manifest = load_dataset(dataset_path)
    df = df.sort_values(["symbol", "timestamp"])

    # Apply date range filter if provided
    if start_date:
        df = df[df["timestamp"] >= pd.Timestamp(start_date)]
    if end_date:
        df = df[df["timestamp"] <= pd.Timestamp(end_date)]

    if symbols is None:
        symbols = manifest["symbols"]

    strategy = Strategy(StrategyConfig(strategy_mode="sma", ml_probability_buy=0.55, ml_probability_sell=0.45, entry_threshold_pct=entry_threshold_pct))

    cash = starting_capital
    open_positions_value = 0.0
    skipped_trades = 0
    current_equity = starting_capital
    peak_equity = starting_capital
    max_drawdown_pct = 0.0

    # Dummy ML signal for SMA mode
    dummy_ml = MlSignal(
        probability_up=0.5,
        confidence=0.0,
        training_rows=0,
        model_age_seconds=0.0,
        feature_names=(),
        buy_threshold=0.55,
        sell_threshold=0.45,
        validation_rows=0,
        model_name="dummy",
    )

    results = {
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
    }

    symbols_dfs = {}
    for symbol in symbols:
        symbols_dfs[symbol] = df[df["symbol"] == symbol].sort_values("timestamp").reset_index(drop=True)

    pointers = {symbol: 0 for symbol in symbols}
    position = {symbol: False for symbol in symbols}
    entry_price = {symbol: 0.0 for symbol in symbols}
    shares = {symbol: 0.0 for symbol in symbols}
    trades = []

    while any(pointers[symbol] < len(symbols_dfs[symbol]) for symbol in symbols):
        # Find the next timestamp
        next_times = [symbols_dfs[symbol].iloc[pointers[symbol]]["timestamp"] for symbol in symbols if pointers[symbol] < len(symbols_dfs[symbol])]
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

            closes = symbols_dfs[symbol]["close"].iloc[: pointers[symbol] + 1]
            if len(closes) < sma_bars:
                pointers[symbol] += 1
                continue

            sma = closes.iloc[-sma_bars:].mean()
            price = row["close"]

            action = strategy.decide_action(price, sma, dummy_ml, position[symbol])

            if action == "BUY" and not position[symbol]:
                strength = (price - sma) / sma
                buy_candidates.append((strength, symbol, row, sma))
            elif action == "SELL" and position[symbol]:
                sell_actions.append((symbol, row, sma))

            pointers[symbol] += 1

        # Execute sells first to free capital
        for symbol, row, sma in sell_actions:
            price = row["close"]
            exit_price = price - slippage - commission
            pnl = (exit_price - entry_price[symbol]) * shares[symbol]
            results["realized_pnl"] += pnl
            if pnl > 0:
                results["winning_pnls"].append(pnl)
            else:
                results["losing_pnls"].append(pnl)
            cash += position_size + pnl
            open_positions_value -= position_size
            current_equity = cash + open_positions_value
            peak_equity = max(peak_equity, current_equity)
            if peak_equity > 0:
                drawdown_pct = (peak_equity - current_equity) / peak_equity * 100
                max_drawdown_pct = max(max_drawdown_pct, drawdown_pct)
            position[symbol] = False
            trades.append({
                "symbol": symbol,
                "side": "SELL",
                "price": price,
                "shares": shares[symbol],
                "timestamp": str(row["timestamp"]),
                "pnl": pnl,
            })
            results["symbol_trade_counts"][symbol] += 1
            results["total_trades"] += 1

        # Sort buys by strength descending
        buy_candidates.sort(reverse=True)
        results["total_buy_candidates"] += len(buy_candidates)
        if len(buy_candidates) > 1:
            results["competing_timestamps"] += 1
        for _, symbol, row, sma in buy_candidates:
            if cash >= position_size:
                price = row["close"]
                position[symbol] = True
                shares[symbol] = position_size / price
                entry_price[symbol] = price + slippage + commission
                cash -= position_size
                open_positions_value += position_size
                current_equity = cash + open_positions_value
                results["max_concurrent_positions"] = max(results["max_concurrent_positions"], sum(position.values()))
                trades.append({
                    "symbol": symbol,
                    "side": "BUY",
                    "price": price,
                    "shares": shares[symbol],
                    "timestamp": str(row["timestamp"]),
                })
                results["symbol_trade_counts"][symbol] += 1
                results["total_trades"] += 1
            else:
                skipped_trades += 1

    # Force close remaining positions
    for symbol in symbols:
        if position[symbol]:
            last_row = symbols_dfs[symbol].iloc[-1]
            price = last_row["close"]
            exit_price = price - slippage - commission
            pnl = (exit_price - entry_price[symbol]) * shares[symbol]
            results["realized_pnl"] += pnl
            if pnl > 0:
                results["winning_pnls"].append(pnl)
            else:
                results["losing_pnls"].append(pnl)
            cash += position_size + pnl
            open_positions_value -= position_size
            current_equity = cash + open_positions_value
            peak_equity = max(peak_equity, current_equity)
            if peak_equity > 0:
                drawdown_pct = (peak_equity - current_equity) / peak_equity * 100
                max_drawdown_pct = max(max_drawdown_pct, drawdown_pct)
            position[symbol] = False
            trades.append({
                "symbol": symbol,
                "side": "SELL",
                "price": price,
                "shares": shares[symbol],
                "timestamp": str(last_row["timestamp"]),
                "pnl": pnl,
                "forced_close": True,
            })

    results["trades"].extend(trades)

    results["final_equity"] = cash + open_positions_value
    results["total_return_pct"] = (results["final_equity"] - starting_capital) / starting_capital * 100 if starting_capital > 0 else 0.0
    results["max_drawdown_pct"] = max_drawdown_pct
    win_rate = len(results["winning_pnls"]) / max(1, len(results["winning_pnls"]) + len(results["losing_pnls"])) * 100
    avg_win = sum(results["winning_pnls"]) / len(results["winning_pnls"]) if results["winning_pnls"] else 0.0
    avg_loss = sum(results["losing_pnls"]) / len(results["losing_pnls"]) if results["losing_pnls"] else 0.0
    results["win_rate"] = win_rate
    results["avg_winning_trade"] = avg_win
    results["avg_losing_trade"] = avg_loss
    results["final_cash"] = cash
    results["skipped_trades"] = skipped_trades

    return results


def extract_clean_columns(result_dict: dict) -> dict:
    """Extract only clean scalar columns for CSV export."""
    clean_cols = [
        "entry_threshold_pct", "sma_bars",
        "total_return_pct", "max_drawdown_pct",
        "total_trades", "total_buy_candidates", "skipped_trades",
        "competing_timestamps", "max_concurrent_positions",
        "win_rate", "avg_winning_trade", "avg_losing_trade",
        "realized_pnl", "final_equity", "final_cash",
        "starting_capital", "position_size",
    ]
    return {k: result_dict[k] for k in clean_cols if k in result_dict}


def main():
    parser = argparse.ArgumentParser(description="Run a simple backtest on historical dataset.")
    parser.add_argument("--dataset", required=True, help="Path to dataset directory (with bars.parquet and manifest.json)")
    parser.add_argument("--symbols", nargs="*", help="Symbols to backtest (default: all in manifest)")
    parser.add_argument("--sma-bars", type=int, default=20, help="SMA lookback bars (default: 20)")
    parser.add_argument("--sma-bars-list", help="Comma-separated list of SMA bars to sweep (e.g., 10,15,20,30,50)")
    parser.add_argument("--entry-threshold-pct", type=float, default=0.0, help="Entry threshold percent above SMA (default: 0.0)")
    parser.add_argument("--entry-threshold-pct-list", help="Comma-separated list of entry threshold percents to sweep (e.g., 0,0.001,0.0025,0.005,0.01)")
    parser.add_argument("--commission-per-order", type=float, default=0.0, help="Commission per order (default: 0.0)")
    parser.add_argument("--slippage-per-share", type=float, default=0.0, help="Slippage per share (default: 0.0)")
    parser.add_argument("--starting-capital", type=float, default=10000.0, help="Starting capital for equity calculation (default: 10000.0)")
    parser.add_argument("--position-size", type=float, default=1000.0, help="Position size per trade in dollars (default: 1000.0)")
    parser.add_argument("--output-csv", help="Path to output CSV file for sweep results (sweep modes only)")
    parser.add_argument("--start-date", help="Start date for backtest in YYYY-MM-DD format (inclusive)")
    parser.add_argument("--end-date", help="End date for backtest in YYYY-MM-DD format (inclusive)")

    args = parser.parse_args()
    dataset_path = Path(args.dataset)

    if args.sma_bars_list:
        sma_values = [int(x.strip()) for x in args.sma_bars_list.split(",")]
        results_list = []
        for sma in sma_values:
            result = run_backtest(dataset_path, args.symbols, sma, args.commission_per_order, args.slippage_per_share, args.entry_threshold_pct, args.starting_capital, args.position_size)
            result["sma_bars"] = sma
            results_list.append(result)
        
        # Sort by realized_pnl descending
        results_list.sort(key=lambda x: x["realized_pnl"], reverse=True)
        
        print("SMA Sweep Results (sorted by Total Return % descending):")
        print(f"{'SMA Bars':<10} {'Total Return':<13} {'Max DD':<8} {'Total Trades':<12} {'Win Rate':<9} {'Avg Win':<8} {'Avg Loss':<9}")
        print("-" * 80)
        for res in results_list:
            print(f"{res['sma_bars']:<10} {res['total_return_pct']:<12.2f}% {res['max_drawdown_pct']:<7.1f}% {res['total_trades']:<12} {res['win_rate']:<8.1f}% ${res['avg_winning_trade']:<7.2f} ${res['avg_losing_trade']:<8.2f}")
        
        if args.output_csv:
            export_cols = ['sma_bars', 'total_return_pct', 'max_drawdown_pct', 'total_trades', 'total_buy_candidates', 'skipped_trades', 'competing_timestamps', 'max_concurrent_positions', 'win_rate', 'avg_winning_trade', 'avg_losing_trade', 'final_equity', 'final_cash', 'starting_capital', 'position_size']
            df_export = pd.DataFrame(results_list)
            df_export = df_export[[col for col in export_cols if col in df_export.columns]]
            df_export.to_csv(args.output_csv, index=False)
        
        best = results_list[0]
        print(f"\nBest SMA setting: {best['sma_bars']} bars (PnL: ${best['realized_pnl']:.2f})")
    elif args.entry_threshold_pct_list:
        threshold_values = [float(x.strip()) for x in args.entry_threshold_pct_list.split(",")]
        results_list = []
        for thresh in threshold_values:
            result = run_backtest(dataset_path, args.symbols, args.sma_bars, args.commission_per_order, args.slippage_per_share, thresh, args.starting_capital, args.position_size)
            result["entry_threshold_pct"] = thresh
            results_list.append(result)
        
        # Sort by total_return_pct descending
        results_list.sort(key=lambda x: x["total_return_pct"], reverse=True)

        print(f"{'Threshold %':<12} {'Total Return':<13} {'Max DD':<8} {'Total Trades':<12} {'Buy Cands':<10} {'Skipped':<8} {'Comp TS':<8} {'Max Pos':<8} {'Win Rate':<9} {'Avg Win':<8} {'Avg Loss':<9}")
        print("-" * 110)
        for res in results_list:
            print(f"{res['entry_threshold_pct']:<12.4f} {res['total_return_pct']:<12.2f}% {res['max_drawdown_pct']:<7.1f}% {res['total_trades']:<12} {res['total_buy_candidates']:<10} {res['skipped_trades']:<8} {res['competing_timestamps']:<8} {res['max_concurrent_positions']:<8} {res['win_rate']:<8.1f}% ${res['avg_winning_trade']:<7.2f} ${res['avg_losing_trade']:<8.2f}")
        
        if args.output_csv:
            export_cols = ['entry_threshold_pct', 'total_return_pct', 'max_drawdown_pct', 'total_trades', 'total_buy_candidates', 'skipped_trades', 'competing_timestamps', 'max_concurrent_positions', 'win_rate', 'avg_winning_trade', 'avg_losing_trade', 'final_equity', 'final_cash', 'starting_capital', 'position_size']
            df_export = pd.DataFrame(results_list)
            df_export = df_export[[col for col in export_cols if col in df_export.columns]]
            df_export.to_csv(args.output_csv, index=False)
        
        best = results_list[0]
        print(f"\nBest entry threshold: {best['entry_threshold_pct']:.4f} (Return: {best['total_return_pct']:.2f}%)")
    else:
        results = run_backtest(dataset_path, args.symbols, args.sma_bars, args.commission_per_order, args.slippage_per_share, args.entry_threshold_pct, args.starting_capital, args.position_size)
        print("Backtest Results:")
        print(f"Position Size: ${results['position_size']:.2f}")
        print(f"Starting Capital: ${results['starting_capital']:.2f}")
        print(f"Final Cash: ${results['final_cash']:.2f}")
        print(f"Final Equity: ${results['final_equity']:.2f}")
        print(f"Total Return: {results['total_return_pct']:.2f}%")
        print(f"Max Drawdown: {results['max_drawdown_pct']:.1f}%")
        print(f"Total Trades: {results['total_trades']}")
        print(f"Skipped Trades: {results['skipped_trades']}")
        print(f"Max Concurrent Positions: {results['max_concurrent_positions']}")
        print(f"Total BUY Candidates: {results['total_buy_candidates']}")
        print(f"Competing Timestamps: {results['competing_timestamps']}")
        print(f"Win Rate: {results['win_rate']:.1f}%")
        print(f"Average Winning Trade: ${results['avg_winning_trade']:.2f}")
        print(f"Average Losing Trade: ${results['avg_losing_trade']:.2f}")
        print("Symbol Trade Counts:")
        for symbol, count in results["symbol_trade_counts"].items():
            print(f"  {symbol}: {count}")
        print("Final Positions: All flat (forced close at end)")


if __name__ == "__main__":
    main()
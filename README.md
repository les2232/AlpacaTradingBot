# AlpacaTradingBot

Intraday Alpaca paper-trading bot with shared live, backtest, and offline-research components.

The current default operating rules are frozen in [TRADING_SPEC.md](TRADING_SPEC.md).

## Quick Start As A Program

This repo now has an installable package and a single top-level command.

If you want to open it by clicking instead of using the terminal, double-click:

`Open AlpacaTradingBot App.pyw`

That opens the native desktop dashboard directly on Windows with no terminal window.

Install it from the repo root:

```powershell
python -m pip install -e .
```

Then use the program entry point:

```powershell
alpaca-bot --help
```

Main commands:

- `alpaca-bot preview` runs the live bot with order execution disabled
- `alpaca-bot live` runs the live bot with normal execution behavior
- `alpaca-bot backtest ...` passes arguments through to `backtest_runner.py`
- `alpaca-bot snapshot ...` passes arguments through to `dataset_snapshotter.py`
- `alpaca-bot research` runs the research pipeline
- `alpaca-bot experiments ...` runs the backtest experiment batch
- `alpaca-bot report ...` runs the daily diagnostic report
- `alpaca-bot dashboard` launches the browser-based Streamlit dashboard
- `alpaca-bot control-panel` launches the desktop control panel

You can also run the package without installing the script wrapper:

```powershell
python -m alpaca_trading_bot --help
```

After install, Windows also gets a GUI launcher command:

```powershell
alpaca-bot-gui
```

If you want this to feel even more like a normal app, create a shortcut to [Launch AlpacaTradingBot.pyw](c:/Users/lesco/Desktop/AlpacaTradingBot-1/Launch%20AlpacaTradingBot.pyw) and pin that shortcut to Start or your taskbar.

## Build A Standalone Windows App

To package the native desktop dashboard as a standalone Windows `.exe`:

```powershell
python -m pip install pyinstaller
powershell -ExecutionPolicy Bypass -File .\build_exe.ps1
```

The built app will be here:

`dist\AlpacaTradingBot\AlpacaTradingBot.exe`

To create a desktop shortcut after building:

```powershell
powershell -ExecutionPolicy Bypass -File .\create_desktop_shortcut.ps1
```

## Current Behavior Snapshot

The live bot currently enforces these behaviors:

- decisions are based on completed bars only
- only one decision timestamp is processed per completed bar
- Alpaca market clock must be open before execution proceeds
- regular-hours trading window is enforced for execution
- no new entries after the configured session cutoff
- open positions are force-flattened into the end-of-day window
- the daily-loss kill switch can block new entries and trigger flattening
- symbol snapshots and order history are persisted to SQLite

One implementation detail is intentionally duplicated for safety:

- end-of-day flatten protection exists both inside the live decision path and in a background thread in [trading_bot.py](trading_bot.py)
- the in-loop guard protects normal decision flow
- the background thread acts as a wall-clock fail-safe if the loop drifts or stalls

## Repository Map

- [trading_bot.py](trading_bot.py)
  Live/paper trading entry point. Loads `.env`, builds the runtime `BotConfig`, evaluates completed 15-minute bars, applies execution safety checks, places Alpaca orders, and writes snapshots to SQLite.
- [strategy.py](strategy.py)
  Shared decision engine for `sma`, `ml`, `hybrid`, `breakout`, and `mean_reversion`. This is the main logic shared by live trading and backtests.
- [backtest_runner.py](backtest_runner.py)
  Offline backtest CLI. Replays saved datasets through the shared strategy layer, supports sweeps, and compares strategy modes.
- [dataset_snapshotter.py](dataset_snapshotter.py)
  Dataset-building CLI. Downloads Alpaca historical bars and writes versioned datasets under `datasets/`.
- [ml/feature_pipeline.py](ml/feature_pipeline.py)
  Offline feature engineering aligned to the live ML feature vector.
- [ml/train.py](ml/train.py)
  Offline model training entry point. Saves the logistic model used by live and backtest ML paths.
- [ml/predict.py](ml/predict.py)
  Offline-model inference helpers used by the live bot. The saved model in `ml/models/logistic_latest.pkl` is the source of truth for ML inference.
- [storage.py](storage.py)
  SQLite persistence for account snapshots, symbol snapshots, and order history.
- [dashboard.py](dashboard.py)
  Streamlit dashboard that reuses the live bot config and snapshot capture path.
- [run_research.py](run_research.py)
  Convenience research pipeline that snapshots data, runs sweeps, and writes timestamped results.
- [run_compare_suite.ps1](run_compare_suite.ps1)
  PowerShell wrapper for a standard strategy comparison suite.
- [universe.py](universe.py)
  Standalone universe builder module. It is intentionally independent and not yet wired into live trading.

## Entry Points

### Live paper trading

Install dependencies:

```powershell
python -m pip install -e .
```

Create a `.env` file:

```env
ALPACA_API_KEY=your_key
ALPACA_API_SECRET=your_secret
ALPACA_PAPER=true
BOT_SYMBOLS=AAPL,MSFT,NVDA
MAX_USD_PER_TRADE=200
MAX_OPEN_POSITIONS=3
MAX_DAILY_LOSS_USD=300
SMA_BARS=20
BAR_TIMEFRAME_MINUTES=15
STRATEGY_MODE=hybrid
```

Run the live bot:

```powershell
alpaca-bot live
```

Preview one or more decision cycles without placing orders:

```powershell
alpaca-bot preview
```

### Backtests

Run a single backtest:

```powershell
alpaca-bot backtest --dataset datasets\YOUR_DATASET --strategy-mode sma
```

Run the canned breakout comparison batch into an isolated experiment folder:

```powershell
alpaca-bot experiments --dataset datasets\YOUR_DATASET
```

Or use the wrapper with the current large breakout dataset already filled in:

```powershell
powershell -ExecutionPolicy Bypass -File run_breakout_research.ps1
```

That writes a timestamped folder under `results\experiments\` with:

- the individual backtest CSV outputs for each run
- `comparison_summary.csv`
- `comparison_per_symbol.csv`
- `comparison_report.md`

Compare `sma`, `ml`, and `hybrid`:

```powershell
alpaca-bot backtest --dataset datasets\YOUR_DATASET --strategy-mode-list sma,ml,hybrid
```

Run the standard comparison suite:

```powershell
powershell -ExecutionPolicy Bypass -File run_compare_suite.ps1
```

### Automated research pipeline

Run the broader research workflow:

```powershell
alpaca-bot research
```

Or use the PowerShell wrapper that logs output to `logs/`:

```powershell
powershell -ExecutionPolicy Bypass -File run_research.ps1
```

### Dataset generation

Build a versioned dataset snapshot:

```powershell
alpaca-bot snapshot --symbols AAPL MSFT NVDA --start 2026-01-01T00:00:00Z --end 2026-02-01T00:00:00Z --timeframe 15Min --feed iex
```

This writes a dataset directory under `datasets/` containing:

- `bars.parquet`
- `manifest.json`

### Dashboard

Run the local monitoring dashboard:

```powershell
alpaca-bot dashboard
```

Streamlit usually serves on `http://localhost:8501`.

## Desktop Control Panel

There is also a small Windows launcher that wraps the existing operational commands without changing the backend.

Install dependencies:

```powershell
python -m pip install -e .
```

Launch the control panel:

```powershell
alpaca-bot control-panel
```

Phase 1 buttons currently wrap:

- `bot.ps1 setup`
- `bot.ps1 show-config`
- `bot.ps1 preflight`
- `bot.ps1 paper-run`
- `bot.ps1 dashboard`

The app streams command output into an in-window console and uses the existing repo scripts as the source of truth.

## Config Flow

There are three distinct config paths in this repo:

### 1. Environment variables for live runtime

[trading_bot.py](trading_bot.py) reads `.env` and environment variables through `load_config()`, then builds a `BotConfig` dataclass. This path is for live and paper execution only.

Examples:

- `BOT_SYMBOLS`
- `STRATEGY_MODE`
- `SMA_BARS`
- `ML_PROBABILITY_BUY`
- `ML_PROBABILITY_SELL`
- `MAX_DAILY_LOSS_USD`
- `EXECUTE_ORDERS`

### 2. CLI arguments for offline tools

[backtest_runner.py](backtest_runner.py) and [dataset_snapshotter.py](dataset_snapshotter.py) are CLI-driven. They do not use the live bot `.env` flow for their main runtime parameters.

Examples:

- `--dataset`
- `--strategy-mode`
- `--strategy-mode-list`
- `--symbols`
- `--start`
- `--end`
- `--feed`

### 3. Runtime config objects

The code then narrows raw config into dataclasses or normalized internal structures:

- `BotConfig` in [trading_bot.py](trading_bot.py)
- `StrategyConfig` in [strategy.py](strategy.py)
- `BacktestConfig` in [backtest_runner.py](backtest_runner.py)
- `UniverseConfig` in [universe.py](universe.py)

That split is intentional:

- `.env` drives live execution
- CLI flags drive offline tools
- runtime dataclasses hold normalized values after parsing

## Shared Logic

The repo is organized around a small set of shared components rather than separate live and research implementations:

- [strategy.py](strategy.py) is shared by live trading and backtests.
- The ML feature vector is intentionally kept aligned between the live bot and the backtester.
- [storage.py](storage.py) is only for live/dashboard persistence, not backtest output.

## Known Overlap And Cleanup Notes

These areas are currently intentional but worth knowing about:

- End-of-day flatten logic exists in two places in [trading_bot.py](trading_bot.py):
  - inside `run_once()` as a fail-safe execution guard
  - inside the background thread started in `main()` as a wall-clock fail-safe
  This is protective overlap, not yet unified.
- The dashboard initializes the live bot directly instead of using a separate read-only service layer. That keeps behavior aligned, but it also means dashboard startup depends on the same Alpaca credentials and bot initialization path as live trading.
- [run_research.py](run_research.py) and [run_compare_suite.ps1](run_compare_suite.ps1) overlap somewhat as research orchestration tools. The Python script is a configurable pipeline; the PowerShell script is a fixed comparison suite.
- [universe.py](universe.py) is present but not yet integrated into live bot symbol selection. This is intentional and low-risk for now.

## Data And Results

- Live snapshots are written to `bot_history.db` by default. Override with `BOT_DB_PATH` if needed.
- Offline datasets live under `datasets/`.
- Backtest and research outputs live under `results/`.

Typical generated artifacts include:

- `datasets/<dataset_id>/bars.parquet`
- `datasets/<dataset_id>/manifest.json`
- `results/<name>.csv`
- `results/<name>_per_symbol.csv`
- `results/<name>_robust_top10.csv`
- `results/<name>_winner_by_symbol.csv`
- `results/best_config_latest.json`
- `results/stability_report.json`
- `results/trade_decision.json`
- `results/compare_suite_YYYYMMDD_HHMMSS/...`

## Practical Workflow

A safe working order for this repo is:

1. Build or refresh a dataset with [dataset_snapshotter.py](dataset_snapshotter.py).
2. Train or refresh the offline model with [ml/train.py](ml/train.py) if you are changing ML inputs.
3. Run either:
   [run_research.py](run_research.py) for a configurable multi-window research pipeline with approval, stability, and regime reporting, or
   [run_compare_suite.ps1](run_compare_suite.ps1) for a fixed comparison suite across known datasets.
4. Preview the live bot with `EXECUTE_ORDERS=false`.
5. Use [dashboard.py](dashboard.py) to inspect snapshots and recent order flow.

## Research Workflow Guide

Use [run_research.py](run_research.py) when you want a configurable workflow that:

- snapshots fresh data for one or more validation windows
- runs parameter sweeps
- ranks results into leaderboard CSVs
- writes decision artifacts such as:
  - `best_config_latest.json`
  - `stability_report.json`
  - `trade_decision.json`

Use [run_compare_suite.ps1](run_compare_suite.ps1) when you want a fixed repeatable benchmark run that:

- uses specific existing datasets
- compares `sma`, `ml`, and `hybrid`
- performs a fixed ML threshold sweep
- writes all outputs into one timestamped subfolder under `results/`

## Notes

- `IEX` data is easier to access on more Alpaca accounts, but it can distort volume-sensitive research compared with `SIP`.
- Live trading is long-only and intraday-only by default.
- The live bot is designed around completed bars, not partially formed bars.

# TradeOS

TradeOS is a strategy-driven trading workspace for Alpaca-based paper/live execution, Streamlit monitoring, offline backtesting, dataset generation, and research validation. The supported interfaces are the `tradeos` CLI and the Streamlit dashboard; the older desktop UI layer has been intentionally removed. The repo shares one strategy engine across live trading and offline analysis, so changes in core files can affect both operator workflows and research results.

## Start Here

Read these first:

1. [`OPERATIONS.md`](OPERATIONS.md) for the day-of operator workflow.
2. [`TRADING_SPEC.md`](TRADING_SPEC.md) for the frozen trading behavior and runtime safety expectations.
3. [`PROJECT_MAP.md`](PROJECT_MAP.md) for a full repo map and module guide.
4. [`CLEANUP_AUDIT.md`](CLEANUP_AUDIT.md) for current cleanup findings and deferred work.
5. [`DESKTOP_REMOVAL_AUDIT.md`](DESKTOP_REMOVAL_AUDIT.md) for the desktop UI removal audit.
6. [`ARTIFACT_STORAGE_POLICY.md`](ARTIFACT_STORAGE_POLICY.md) for the artifact and output storage rules.

## Main Modes

### Live or preview trading

- `tradeos live`
  Runs the live bot with normal execution behavior.
- `tradeos preview`
  Runs the same runtime path with execution disabled.
- `.\start_dashboard.ps1`
  Preferred operator launcher. Starts or reuses the live bot and dashboard together.

### Monitoring

- `tradeos dashboard`
  Starts the Streamlit dashboard directly.
- `.\start_dashboard.ps1 -DashboardOnly`
  Opens monitoring without starting a live bot.

Streamlit is now the only supported UI surface in the repository.

### Research and offline analysis

- `tradeos snapshot ...`
  Builds a versioned dataset under `datasets/`.
- `tradeos backtest ...`
  Replays a dataset through the shared strategy engine.
- `tradeos research`
  Runs the broader research pipeline.
- `tradeos experiments ...`
  Runs the backtest experiment batch.
- `tradeos report ...`
  Generates the daily diagnostic report.

There are also many focused `run_*.py` scripts at the repo root for specific validations and experiments. Treat those as research utilities, not as part of the normal operator path.

### Experiment logging and review

- successful research and backtest-oriented runs now append structured entries to `results/experiment_log.jsonl`
- the log is append-only JSONL so it is easy to diff, parse, and extend
- the Streamlit dashboard `Performance` tab now includes an `Experiment History` section for recent runs, metric trends, and top results

Useful commands:

```powershell
python run_research.py
python run_backtest_experiments.py --dataset datasets\YOUR_DATASET
python run_momentum_breakout_validation.py --output-dir results\momentum_breakout_validation
python -m streamlit run dashboard.py
```

## Minimal Setup

Install the project from the repo root:

```powershell
python -m pip install -e .
```

Create a `.env` with Alpaca credentials and runtime guardrails:

```env
ALPACA_API_KEY=your_key
ALPACA_API_SECRET=your_secret
ALPACA_PAPER=true
MAX_USD_PER_TRADE=200
MAX_OPEN_POSITIONS=3
MAX_DAILY_LOSS_USD=300
```

Keep active trading settings in [`config/live_config.json`](config/live_config.json). In this repo, `.env` is for secrets and risk limits, while `config/live_config.json` is the runtime trading-settings source of truth.

## Typical Commands

```powershell
tradeos --help
tradeos preview
tradeos live
tradeos dashboard
tradeos backtest --dataset datasets\YOUR_DATASET --strategy-mode sma
tradeos snapshot --symbols AAPL MSFT NVDA --start 2026-01-01T00:00:00Z --end 2026-02-01T00:00:00Z --timeframe 15Min --feed iex
tradeos research
```

PowerShell wrappers still used in practice:

```powershell
.\start_dashboard.ps1
powershell -ExecutionPolicy Bypass -File run_research.ps1
powershell -ExecutionPolicy Bypass -File run_compare_suite.ps1
```

## Repository Shape

The intended boundary is:

- `live runtime`: operator-facing execution, broker integration, locks, risk checks, persistence
- `shared strategy`: signal logic and indicators reused by live and offline paths
- `dashboard`: read-only monitoring and diagnostics
- `research`: validation, sweeps, audits, and experiment runners
- `artifacts`: datasets, logs, results, and disposable local scratch output

The current cleanup direction is a conservative `v2-in-place`: reduce repo-root noise, extract shared internals into smaller modules, and preserve the existing CLI and operator workflow while the internals become more structured.

### Core runtime

- [`alpaca_trading_bot/cli.py`](alpaca_trading_bot/cli.py)
  Main CLI implementation.
- [`trading_bot.py`](trading_bot.py)
  Live execution loop, risk gates, runtime protections, and broker-backed order flow.
- [`strategy.py`](strategy.py)
  Shared strategy engine used by live trading and backtests.
- [`tradeos/brokers/alpaca_broker.py`](tradeos/brokers/alpaca_broker.py)
  Broker adapter for Alpaca.
- [`storage.py`](storage.py)
  SQLite persistence for runtime history.
- [`botlog.py`](botlog.py)
  JSONL event logging under `logs/`.

### Dashboard and diagnostics

- [`dashboard.py`](dashboard.py)
  Streamlit monitoring UI.
- [`dashboard_state.py`](dashboard_state.py)
  Dashboard-side state loading and drilldown logic.
- [`daily_report.py`](daily_report.py)
  Diagnostic reporting.
- [`drift_monitor.py`](drift_monitor.py)
  Drift-analysis support used by the dashboard and reports.

### Offline research

- [`dataset_snapshotter.py`](dataset_snapshotter.py)
  Dataset creation into `datasets/`.
- [`backtest_runner.py`](backtest_runner.py)
  Offline simulation against saved datasets.
- [`research/`](research/)
  Current research-only validation, audit, and experiment scripts.
- [`research/legacy_experiments/`](research/legacy_experiments/)
  Historical sweep-era and archival experiment runners preserved for reproducibility.
- root-level `run_*.py` and helper wrappers
  Compatibility entry points that keep existing imports and commands stable while the real implementations live under `research/` or `research/legacy_experiments/`.

### Docs and runbooks

- [`OPERATIONS.md`](OPERATIONS.md)
- [`TRADING_SPEC.md`](TRADING_SPEC.md)
- [`PROJECT_MAP.md`](PROJECT_MAP.md)
- [`CLEANUP_AUDIT.md`](CLEANUP_AUDIT.md)
- [`V2_CLEANUP_EXECUTION_PLAN.md`](V2_CLEANUP_EXECUTION_PLAN.md)

### Legacy holding area

- [`repo_quarantine/`](repo_quarantine/)
  Holding area for intentionally removed or retired material. The legacy desktop UI runtime has already been removed from the repository.

## Data and Outputs

- `datasets/`
  Versioned offline datasets, usually `bars.parquet` plus `manifest.json`.
- `results/`
  Shared artifact area. Root-level files are current promoted artifacts; subdirectories hold active research runs and archived historical outputs.
- `logs/`
  Runtime JSONL logs and startup metadata.
- `bot_history.db`
  Default SQLite runtime history database.

The repo also generates many local test and analysis artifacts such as `test_botlog_*`, `test_logs_*`, `test_dashboard_logs_*`, `inspect_dataset_*`, `tmp_test_artifacts/`, and `output/`.

Artifact policy:

- keep current promoted artifacts such as `results/best_config_latest.json`, `results/trade_decision.json`, and `results/strategy_status.md` at the `results/` root for compatibility
- put active research runs in dedicated subdirectories under `results/`
- use `results/archive/` for reviewed historical outputs
- keep scratch and test artifacts ignored and disposable

See [`ARTIFACT_POLICY_AUDIT.md`](ARTIFACT_POLICY_AUDIT.md), [`ARTIFACT_STORAGE_POLICY.md`](ARTIFACT_STORAGE_POLICY.md), and [`results/README.md`](results/README.md) for the working policy.

## Experiment Log Notes

The experiment log currently records:

- timestamp
- run type and script name
- strategy, symbols, dataset metadata, and key parameters when available
- normalized core metrics such as return, profit factor, sharpe, win rate, drawdown, trade count, expectancy, and realized pnl when the source script exposes them
- output paths, summary artifact paths, and git branch/commit context
- a compact change fingerprint and a conservative auto-summary versus the most recent comparable run

How logs are generated:

- current research entrypoints append a log entry after a successful run completes
- the logger reuses metrics already produced by each script rather than recomputing performance
- if there is a recent comparable run for the same script and strategy, the logger labels the new run as `improved vs prior`, `worse than prior`, `trade count collapsed`, or `insufficient evidence`

How to extend the logger for a new research script:

1. Import `log_experiment_run` from [`research/experiment_log.py`](research/experiment_log.py).
2. Call it near the end of the script after final metrics and artifact paths are known.
3. Pass the script's existing result metrics and parameters directly; avoid recomputing anything just for logging.
4. If the script writes a JSON summary artifact, pass that path as `summary_path` so the dashboard has a stable anchor.

## Where New Code Goes

- Put runtime-sensitive code near the existing core files and packages.
- Put new validation, audit, sweep, and experiment scripts under [`research/`](research/).
- Put older archival experiment runners that still need to be preserved under [`research/legacy_experiments/`](research/legacy_experiments/).
- Keep root-level wrappers only when compatibility with existing imports, tests, or scripts still matters.

## Safety Boundaries

Treat these files as sensitive:

- [`trading_bot.py`](trading_bot.py)
- [`strategy.py`](strategy.py)
- [`alpaca_trading_bot/cli.py`](alpaca_trading_bot/cli.py)
- [`start_dashboard.ps1`](start_dashboard.ps1)
- [`tradeos/brokers/alpaca_broker.py`](tradeos/brokers/alpaca_broker.py)
- [`storage.py`](storage.py)
- [`dashboard_state.py`](dashboard_state.py)
- [`config/live_config.json`](config/live_config.json)

Important constraints already enforced in the repo:

- one live-process lock
- completed-bar-only decision cadence
- market-hours and session-cutoff guards
- end-of-day flatten protections
- daily-loss kill switch behavior
- runtime approval checks for `config/live_config.json`

Do not casually refactor or deduplicate those paths during cleanup work.

Recommended first cleanup slice:

1. ignore and artifact hygiene
2. boundary docs and repo orientation
3. shared config extraction between live and backtest paths
4. strategy helper extraction before touching the live loop

## Naming Notes

`tradeos` is the preferred current product and CLI name. Some compatibility layers still use the older `alpaca_trading_bot` package name or the `alpaca-bot` script alias. Those leftovers are intentional for compatibility unless changed in a dedicated migration.

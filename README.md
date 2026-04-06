# AlpacaTradingBot

Simple Alpaca paper-trading bot with a basic intraday SMA strategy.

Default operating rules are frozen in [TRADING_SPEC.md](TRADING_SPEC.md).

## Setup

Install dependencies in PowerShell:

```bash
python -m pip install -r requirements.txt
```

If `python` is not available on your machine, use:

```bash
py -m pip install -r requirements.txt
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
```

Run the bot once:

```bash
python trading_bot.py
```

Preview decisions without placing orders:

```bash
$env:EXECUTE_ORDERS="false"
python trading_bot.py
```

Run the local dashboard:

```bash
python -m streamlit run dashboard.py
```

If `python` is not available, use:

```bash
py -m streamlit run dashboard.py
```

Streamlit should print a local URL, usually `http://localhost:8501`.

The bot now records local history to `bot_history.db` by default. Override that with `BOT_DB_PATH` if you want the SQLite file somewhere else.

## Offline Dataset Snapshots

Build a versioned Parquet snapshot of historical bars:

```bash
python dataset_snapshotter.py --symbols AAPL MSFT NVDA --start 2026-01-01T00:00:00Z --end 2026-02-01T00:00:00Z --timeframe 15Min --feed iex
```

Feed notes:

- `--feed iex` works on more accounts, but it is single-exchange data and can understate or distort intraday volume-sensitive signals.
- `--feed sip` uses broader consolidated market data when your Alpaca account has SIP access enabled.
- If `--feed sip` fails because of permissions or subscription limits, the snapshotter now prints a clearer explanation and suggests falling back to `iex`.

This writes a dataset under `datasets/` with:

- `bars.parquet`
- `manifest.json`

The dataset version includes the date range, feed, and a code hash so offline training data can be tied back to the exact snapshotting logic.

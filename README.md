# AlpacaTradingBot

Simple Alpaca paper-trading bot with a basic SMA strategy.

## Setup

Install dependencies:

```bash
pip install -r requirements.txt
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
SMA_DAYS=20
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
streamlit run dashboard.py
```

The bot now records local history to `bot_history.db` by default. Override that with `BOT_DB_PATH` if you want the SQLite file somewhere else.

The original notebook prototype is still in `trading_bot.ipynb`.

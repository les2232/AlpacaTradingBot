from dataclasses import asdict

import pandas as pd
import streamlit as st

from trading_bot import AlpacaTradingBot, load_config


st.set_page_config(page_title="Alpaca Trading Bot", layout="wide")

st.markdown(
    """
    <style>
    .stApp {
        background:
            radial-gradient(circle at top left, rgba(166, 201, 255, 0.22), transparent 28%),
            radial-gradient(circle at top right, rgba(89, 166, 127, 0.18), transparent 25%),
            linear-gradient(180deg, #f6f4ee 0%, #ece6db 100%);
        color: #1d2a1f;
    }
    .hero {
        padding: 1.4rem 1.6rem;
        border: 1px solid rgba(29, 42, 31, 0.08);
        border-radius: 24px;
        background: rgba(255, 252, 245, 0.82);
        box-shadow: 0 18px 50px rgba(69, 58, 42, 0.10);
        margin-bottom: 1rem;
    }
    .eyebrow {
        font-size: 0.8rem;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        color: #6a6f57;
        margin-bottom: 0.45rem;
    }
    .hero h1 {
        margin: 0;
        font-size: 2.6rem;
        line-height: 1;
        color: #1d2a1f;
    }
    .hero p {
        margin: 0.7rem 0 0;
        max-width: 48rem;
        color: #435244;
        font-size: 1rem;
    }
    .status-pill {
        display: inline-block;
        margin-top: 1rem;
        padding: 0.4rem 0.7rem;
        border-radius: 999px;
        background: #e6f2e8;
        color: #245233;
        font-size: 0.9rem;
        font-weight: 600;
    }
    .status-pill.danger {
        background: #fde9e2;
        color: #8c2f1d;
    }
    .panel-title {
        font-size: 0.82rem;
        text-transform: uppercase;
        letter-spacing: 0.12em;
        color: #6f725d;
        margin-bottom: 0.35rem;
    }
    .metric-card {
        padding: 0.9rem 1rem;
        border-radius: 18px;
        background: rgba(255, 250, 240, 0.90);
        border: 1px solid rgba(29, 42, 31, 0.08);
        min-height: 6.4rem;
    }
    .metric-value {
        font-size: 1.65rem;
        font-weight: 700;
        color: #1d2a1f;
    }
    .metric-delta {
        color: #5d6b5f;
        font-size: 0.92rem;
    }
    .section-card {
        padding: 1rem 1rem 0.8rem;
        border-radius: 22px;
        background: rgba(255, 252, 246, 0.88);
        border: 1px solid rgba(29, 42, 31, 0.08);
        box-shadow: 0 12px 36px rgba(69, 58, 42, 0.06);
        height: 100%;
    }
    .section-card h3 {
        margin-top: 0.1rem;
        margin-bottom: 0.8rem;
        color: #1d2a1f;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def get_bot() -> AlpacaTradingBot:
    return AlpacaTradingBot(load_config())


def format_money(value: float) -> str:
    return f"${value:,.2f}"


def render_metric_card(label: str, value: str, delta: str | None = None) -> None:
    delta_html = f"<div class='metric-delta'>{delta}</div>" if delta else ""
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="panel-title">{label}</div>
            <div class="metric-value">{value}</div>
            {delta_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_section_title(title: str) -> None:
    st.markdown(
        f"""
        <div class="section-card">
            <h3>{title}</h3>
        </div>
        """,
        unsafe_allow_html=True,
    )


def main() -> None:
    bot = get_bot()

    if "snapshot" not in st.session_state:
        st.session_state.snapshot, st.session_state.recent_orders = bot.capture_state()

    kill_switch_class = "status-pill danger" if st.session_state.snapshot.kill_switch_triggered else "status-pill"
    kill_switch_text = "Kill Switch Active" if st.session_state.snapshot.kill_switch_triggered else "Kill Switch Clear"
    st.markdown(
        f"""
        <div class="hero">
            <div class="eyebrow">Paper Trading Control Surface</div>
            <h1>Alpaca Bot Dashboard</h1>
            <p>Monitor account health, inspect per-symbol decisions, and review recent order flow without digging through terminal output.</p>
            <div class="{kill_switch_class}">{kill_switch_text}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    controls_col, actions_col = st.columns([1.1, 1.4])
    with controls_col:
        execute_orders = st.toggle("Enable order execution", value=False)
        if execute_orders:
            st.warning("Running a bot cycle will submit paper orders.")
        st.caption("Keep this off while validating strategy output and dashboard state.")

    with actions_col:
        button_cols = st.columns(2)
        refresh = button_cols[0].button("Refresh snapshot", type="primary", use_container_width=True)
        run_cycle = button_cols[1].button("Run bot cycle", use_container_width=True)

    if refresh:
        st.session_state.snapshot, st.session_state.recent_orders = bot.capture_state()

    if run_cycle:
        st.session_state.snapshot = bot.run_once(execute_orders=execute_orders)
        st.session_state.recent_orders = bot.get_recent_orders(limit=12)

    snapshot = st.session_state.snapshot
    recent_orders = st.session_state.get("recent_orders", bot.get_recent_orders(limit=12))
    run_history = pd.DataFrame(bot.storage.get_run_history(limit=120))
    selected_symbol = st.selectbox("History symbol", bot.config.symbols, index=0)
    symbol_history = pd.DataFrame(bot.storage.get_symbol_history(selected_symbol, limit=120))

    metric_cols = st.columns(5)
    with metric_cols[0]:
        render_metric_card("Cash", format_money(snapshot.cash))
    with metric_cols[1]:
        render_metric_card("Buying Power", format_money(snapshot.buying_power))
    with metric_cols[2]:
        render_metric_card("Equity", format_money(snapshot.equity), f"Prev close {format_money(snapshot.last_equity)}")
    with metric_cols[3]:
        pnl_delta = "Loss limit reached" if snapshot.kill_switch_triggered else "Within daily limit"
        render_metric_card("Daily PnL", format_money(snapshot.daily_pnl), pnl_delta)
    with metric_cols[4]:
        render_metric_card("Tracked Symbols", str(len(snapshot.symbols)), ", ".join(item.symbol for item in snapshot.symbols))

    symbols_df = pd.DataFrame([asdict(item) for item in snapshot.symbols])
    if not symbols_df.empty:
        symbols_df = symbols_df[
            ["symbol", "price", "sma", "action", "holding", "quantity", "market_value", "error"]
        ]
        symbols_df = symbols_df.rename(
            columns={
                "price": "last_price",
                "market_value": "position_value",
            }
        )

    positions_rows = []
    for symbol, position in snapshot.positions.items():
        positions_rows.append(
            {
                "symbol": symbol,
                "qty": float(position.qty) if position.qty is not None else 0.0,
                "market_value": float(position.market_value) if position.market_value is not None else 0.0,
                "avg_entry_price": float(position.avg_entry_price) if position.avg_entry_price is not None else 0.0,
                "unrealized_pl": float(position.unrealized_pl) if position.unrealized_pl is not None else 0.0,
            }
        )

    positions_df = pd.DataFrame(positions_rows)
    orders_df = pd.DataFrame([asdict(order) for order in recent_orders])
    if not orders_df.empty:
        orders_df = orders_df[
            ["submitted_at", "symbol", "side", "status", "qty", "filled_qty", "filled_avg_price", "notional"]
        ]

    history_left, history_right = st.columns([1.3, 1])
    with history_left:
        render_section_title("Account History")
        if run_history.empty:
            st.info("No saved run history yet.")
        else:
            account_chart = run_history[["timestamp_utc", "equity", "cash", "daily_pnl"]].copy()
            account_chart["timestamp_utc"] = pd.to_datetime(account_chart["timestamp_utc"])
            account_chart = account_chart.set_index("timestamp_utc")
            st.line_chart(account_chart)
    with history_right:
        render_section_title(f"{selected_symbol} Trend")
        if symbol_history.empty:
            st.info("No symbol history yet.")
        else:
            symbol_chart = symbol_history[["timestamp_utc", "price", "sma"]].copy()
            symbol_chart = symbol_chart.dropna(how="all", subset=["price", "sma"])
            symbol_chart["timestamp_utc"] = pd.to_datetime(symbol_chart["timestamp_utc"])
            symbol_chart = symbol_chart.set_index("timestamp_utc")
            if symbol_chart.empty:
                st.info("No price/SMA points saved yet for this symbol.")
            else:
                st.line_chart(symbol_chart)
            action_counts = symbol_history["action"].value_counts().rename_axis("action").reset_index(name="count")
            st.dataframe(action_counts, use_container_width=True, hide_index=True)

    left_col, right_col = st.columns([1.5, 1])
    with left_col:
        render_section_title("Symbol Decisions")
        st.dataframe(symbols_df, use_container_width=True, hide_index=True)
    with right_col:
        render_section_title("Open Positions")
        if positions_df.empty:
            st.info("No open positions.")
        else:
            st.dataframe(positions_df, use_container_width=True, hide_index=True)

    render_section_title("Recent Orders")
    if orders_df.empty:
        st.info("No recent order activity returned by Alpaca.")
    else:
        st.dataframe(orders_df, use_container_width=True, hide_index=True)

    risk_col, notes_col = st.columns([1, 1])
    with risk_col:
        render_section_title("Risk Guardrails")
        st.write(f"Max per trade: {format_money(bot.config.max_usd_per_trade)}")
        st.write(f"Max open positions: {bot.config.max_open_positions}")
        st.write(f"Max daily loss: {format_money(bot.config.max_daily_loss_usd)}")
        st.write(f"SMA window: {bot.config.sma_days} days")
    with notes_col:
        render_section_title("Session Notes")
        st.write(f"Snapshot time: `{snapshot.timestamp_utc}`")
        st.write(f"Paper mode: `{bot.config.paper}`")
        st.write(f"Watched symbols: `{', '.join(bot.config.symbols)}`")
        st.write(f"Recent order count shown: `{len(recent_orders)}`")
        st.write(f"History database: `{bot.storage.db_path}`")


if __name__ == "__main__":
    main()

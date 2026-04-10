import os
import inspect
from dataclasses import asdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components
from dotenv import load_dotenv

from trading_bot import AlpacaTradingBot, load_config

st.set_page_config(page_title="Alpaca Bot", layout="wide")


# ── CSS ───────────────────────────────────────────────────────────────────────

def _inject_css() -> None:
    st.markdown(
        """
        <style>
        html, body, [class*="css"] { font-family: 'Segoe UI', Tahoma, sans-serif; }
        code, pre, .mono { font-family: 'Consolas', 'Courier New', monospace; }

        .stApp { background: #0f1117; color: #e2e8f0; }

        /* Kill switch banner */
        .ks-banner {
            background: #7f1d1d;
            border: 1px solid #dc2626;
            border-radius: 8px;
            padding: 0.6rem 1rem;
            color: #fca5a5;
            font-weight: 600;
            font-size: 0.9rem;
            margin-bottom: 1rem;
            letter-spacing: 0.03em;
        }

        /* Metric cards */
        .m-card {
            background: #1e2130;
            border: 1px solid #2d3148;
            border-radius: 12px;
            padding: 0.85rem 1rem;
            min-height: 5.5rem;
        }
        .m-label {
            font-size: 0.78rem;
            text-transform: uppercase;
            letter-spacing: 0.1em;
            color: #cbd5e1;
            margin-bottom: 0.3rem;
        }
        .m-value {
            font-size: 1.55rem;
            font-weight: 700;
            color: #f1f5f9;
            line-height: 1.1;
        }
        .m-delta { font-size: 0.88rem; color: #a8b4c7; margin-top: 0.25rem; }
        .m-pos   { color: #4ade80; }
        .m-neg   { color: #f87171; }

        /* Feed status dot */
        .dot {
            display: inline-block;
            width: 8px; height: 8px;
            border-radius: 50%;
            margin-right: 6px;
            vertical-align: middle;
        }
        .dot-live  { background: #4ade80; box-shadow: 0 0 6px #4ade80; }
        .dot-rest  { background: #facc15; }
        .dot-error { background: #f87171; }

        /* Symbol cards */
        .sym-card {
            background: #1a1d2e;
            border: 1px solid #2d3148;
            border-radius: 14px;
            padding: 1rem 1.1rem;
            margin-bottom: 0.75rem;
        }
        .sym-held {
            border-color: #166534;
            box-shadow: inset 0 0 0 1px rgba(74, 222, 128, 0.18);
        }
        .sym-header {
            display: flex;
            align-items: center;
            justify-content: space-between;
            margin-bottom: 0.5rem;
        }
        .sym-ticker {
            font-size: 1.05rem;
            font-weight: 700;
            color: #f1f5f9;
            font-family: 'DM Mono', monospace;
            letter-spacing: 0.05em;
        }
        .sym-price { font-size: 1rem; color: #d7e0ee; font-family: 'DM Mono', monospace; }
        .badge {
            display: inline-block;
            border-radius: 999px;
            padding: 0.2rem 0.6rem;
            font-size: 0.75rem;
            font-weight: 600;
            letter-spacing: 0.04em;
        }
        .b-buy  { background: #14532d; color: #86efac; }
        .b-sell { background: #7f1d1d; color: #fca5a5; }
        .b-hold { background: #1e3a5f; color: #93c5fd; }
        .b-err  { background: #451a03; color: #fed7aa; }
        .b-held { background: #052e16; color: #86efac; }
        .b-ready { background: #14532d; color: #dcfce7; }
        .b-blocked { background: #7f1d1d; color: #fecaca; }
        .b-nosignal { background: #334155; color: #cbd5e1; }

        .held-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(210px, 1fr));
            gap: 0.75rem;
            margin-bottom: 0.9rem;
        }
        .held-card {
            background: linear-gradient(180deg, #13261d, #0f172a);
            border: 1px solid #166534;
            border-radius: 14px;
            padding: 0.95rem 1rem;
        }
        .held-top {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 0.55rem;
        }
        .held-symbol {
            font-size: 1rem;
            font-weight: 700;
            color: #f8fafc;
            font-family: 'DM Mono', monospace;
            letter-spacing: 0.05em;
        }
        .held-main {
            font-size: 1.15rem;
            font-weight: 700;
            color: #f8fafc;
            margin-bottom: 0.45rem;
        }
        .held-kv {
            display: flex;
            justify-content: space-between;
            gap: 0.6rem;
            font-size: 0.85rem;
            color: #e2e8f0;
            padding: 0.18rem 0;
        }
        .held-kv-label { color: #c7d2e3; }

        /* ML probability bar */
        .ml-wrap { margin: 0.5rem 0 0.25rem; }
        .ml-label {
            font-size: 0.78rem;
            color: #c7d2e3;
            margin-bottom: 0.25rem;
            letter-spacing: 0.05em;
            text-transform: uppercase;
        }
        .ml-track {
            position: relative;
            height: 10px;
            background: #2d3148;
            border-radius: 5px;
            overflow: visible;
        }
        .ml-fill { height: 100%; border-radius: 5px; }
        .ml-tick {
            position: absolute;
            top: -3px;
            width: 2px;
            height: 16px;
            background: #f8fafc;
            border-radius: 1px;
            opacity: 0.7;
        }
        .ml-axis {
            display: flex;
            justify-content: space-between;
            font-size: 0.72rem;
            color: #a8b4c7;
            margin-top: 0.15rem;
            font-family: 'DM Mono', monospace;
        }

        /* Proximity bar */
        .prox-wrap { margin: 0.4rem 0; }
        .prox-label {
            font-size: 0.78rem;
            color: #c7d2e3;
            margin-bottom: 0.2rem;
            letter-spacing: 0.05em;
            text-transform: uppercase;
        }
        .prox-track { position: relative; height: 6px; background: #2d3148; border-radius: 3px; }
        .prox-fill  { height: 100%; border-radius: 3px; }
        .prox-above { background: linear-gradient(90deg, #166534, #4ade80); }
        .prox-below { background: linear-gradient(90deg, #7f1d1d, #f87171); }

        /* KV config layout */
        .kv {
            display: flex;
            justify-content: space-between;
            align-items: baseline;
            padding: 0.45rem 0;
            border-bottom: 1px solid #1e2130;
        }
        .kv-key { font-size: 0.88rem; color: #c7d2e3; }
        .kv-val { font-size: 0.9rem; font-family: 'DM Mono', monospace; color: #f1f5f9; }

        /* Section headings */
        .sec-head {
            font-size: 0.78rem;
            text-transform: uppercase;
            letter-spacing: 0.12em;
            color: #b6c2d4;
            margin: 1.2rem 0 0.5rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


# ── Bot singleton ─────────────────────────────────────────────────────────────

@st.cache_resource
def get_bot() -> AlpacaTradingBot:
    load_dotenv(Path.cwd() / ".env")
    return AlpacaTradingBot(load_config())


# ── Utilities ─────────────────────────────────────────────────────────────────

def _ensure_bot_capabilities(bot: AlpacaTradingBot) -> AlpacaTradingBot:
    required_methods = ("preview_execution", "get_last_run_cycle_report")
    run_once_params = ()
    try:
        run_once_params = tuple(inspect.signature(bot.run_once).parameters.keys())
    except (TypeError, ValueError, AttributeError):
        run_once_params = ()
    if all(hasattr(bot, name) for name in required_methods) and "force_process" in run_once_params:
        return bot
    get_bot.clear()
    return get_bot()


def parse_mixed_iso_timestamps(values: pd.Series) -> pd.Series:
    try:
        parsed = pd.to_datetime(values, utc=True, errors="coerce", format="mixed")
    except (TypeError, ValueError):
        parsed = pd.to_datetime(values, utc=True, errors="coerce")
    if parsed.isna().all():
        parsed = pd.to_datetime(values, utc=True, errors="coerce")
    return parsed


def _fmt_money(v: float) -> str:
    return f"${v:,.2f}"


def _pnl_cls(v: float) -> str:
    return "m-pos" if v >= 0 else "m-neg"


def _get_bar_timing(timeframe_minutes: int) -> dict[str, object]:
    now_utc = datetime.now(timezone.utc)
    timeframe_seconds = max(int(timeframe_minutes), 1) * 60
    current_bucket = int(now_utc.timestamp()) // timeframe_seconds
    next_bar_close_utc = datetime.fromtimestamp(
        (current_bucket + 1) * timeframe_seconds,
        tz=timezone.utc,
    )
    seconds_remaining = max(int((next_bar_close_utc - now_utc).total_seconds()), 0)
    countdown = str(timedelta(seconds=seconds_remaining))
    if seconds_remaining < 36000:
        countdown = f"0{countdown}"
    return {
        "current_time_utc": now_utc,
        "next_bar_close_utc": next_bar_close_utc,
        "seconds_remaining": seconds_remaining,
        "countdown": countdown,
        "timeframe_minutes": max(int(timeframe_minutes), 1),
    }


# ── HTML builders ─────────────────────────────────────────────────────────────

def _metric_html(label: str, value: str, delta: str = "", delta_cls: str = "") -> str:
    delta_part = f"<div class='m-delta {delta_cls}'>{delta}</div>" if delta else ""
    return f"<div class='m-card'><div class='m-label'>{label}</div><div class='m-value'>{value}</div>{delta_part}</div>"


def _ml_bar_html(
    prob: float | None,
    buy_thr: float | None,
    sell_thr: float | None,
    confidence: float | None,
) -> str:
    if prob is None:
        return "<div class='ml-wrap'><div class='ml-label'>ML Probability — n/a</div></div>"
    pct = min(max(prob * 100, 0), 100)
    if buy_thr is not None and prob >= buy_thr:
        fill_color = "#4ade80"
    elif sell_thr is not None and prob <= sell_thr:
        fill_color = "#f87171"
    else:
        fill_color = "#64748b"
    buy_tick = (
        f"<div class='ml-tick' style='left:{buy_thr * 100:.1f}%'></div>"
        if buy_thr is not None
        else ""
    )
    sell_tick = (
        f"<div class='ml-tick' style='left:{sell_thr * 100:.1f}%'></div>"
        if sell_thr is not None
        else ""
    )
    conf_str = f"&nbsp;&nbsp;conf {confidence:.2f}" if confidence is not None else ""
    return (
        f"<div class='ml-wrap'>"
        f"<div class='ml-label'>ML Probability &nbsp;"
        f"<span style=\"font-family:'DM Mono',monospace\">{prob:.3f}{conf_str}</span></div>"
        f"<div class='ml-track'>"
        f"<div class='ml-fill' style='width:{pct:.1f}%;background:{fill_color}'></div>"
        f"{sell_tick}{buy_tick}"
        f"</div>"
        f"<div class='ml-axis'><span>0.0</span><span>0.5</span><span>1.0</span></div>"
        f"</div>"
    )


def _prox_bar_html(price: float | None, sma: float | None) -> str:
    if price is None or sma is None or sma == 0:
        return ""
    diff_pct = (price - sma) / sma * 100
    above = diff_pct >= 0
    bar_pct = min(abs(diff_pct) * 10, 100)
    fill_cls = "prox-above" if above else "prox-below"
    sign = "+" if above else ""
    label = (
        f"Price vs SMA &nbsp;"
        f"<span style=\"font-family:'DM Mono',monospace\">{sign}{diff_pct:.2f}%</span>"
    )
    return (
        f"<div class='prox-wrap'>"
        f"<div class='prox-label'>{label}</div>"
        f"<div class='prox-track'>"
        f"<div class='prox-fill {fill_cls}' style='width:{bar_pct:.1f}%'></div>"
        f"</div></div>"
    )


def _badge_html(action: str) -> str:
    cls = {"BUY": "b-buy", "SELL": "b-sell", "ERROR": "b-err"}.get(
        (action or "").upper(), "b-hold"
    )
    return f"<span class='badge {cls}'>{action}</span>"


def _sym_card_html(item, extra_html: str = "") -> str:
    price_str = f"${item.price:,.2f}" if item.price is not None else "—"
    sma_str = f"SMA {item.sma:,.2f}" if item.sma is not None else ""
    held_str = (
        f"held {item.holding_minutes:.0f} min"
        if item.holding and item.holding_minutes is not None
        else ""
    )
    meta = "  ·  ".join(filter(None, [sma_str, held_str]))
    badge = _badge_html(item.action or "HOLD")
    ml_bar = _ml_bar_html(
        item.ml_probability_up,
        getattr(item, "ml_buy_threshold", None),
        getattr(item, "ml_sell_threshold", None),
        item.ml_confidence,
    )
    prox_bar = _prox_bar_html(item.price, item.sma)
    error_html = (
        f"<div style='font-size:0.78rem;color:#f87171;margin-top:0.4rem'>{item.error}</div>"
        if item.error
        else ""
    )
    return (
        f"<div class='sym-card'>"
        f"<div class='sym-header'><span class='sym-ticker'>{item.symbol}</span>{badge}</div>"
        f"<div style='display:flex;justify-content:space-between;align-items:baseline'>"
        f"<span class='sym-price'>{price_str}</span>"
        f"<span style='font-size:0.84rem;color:#b6c2d4'>{meta}</span>"
        f"</div>"
        f"{prox_bar}{ml_bar}{error_html}{extra_html}"
        f"</div>"
    )


def _kv(key: str, val: str) -> str:
    return (
        f"<div class='kv'>"
        f"<span class='kv-key'>{key}</span>"
        f"<span class='kv-val'>{val}</span>"
        f"</div>"
    )


def _render_bar_timing(bot) -> None:
    timing = _get_bar_timing(bot.config.bar_timeframe_minutes)
    current_time_utc = timing["current_time_utc"]
    next_bar_close_utc = timing["next_bar_close_utc"]
    seconds_remaining = timing["seconds_remaining"]
    timeframe_minutes = timing["timeframe_minutes"]

    st.markdown("<div class='sec-head'>Next Bar Refresh</div>", unsafe_allow_html=True)
    components.html(
        f"""
        <div style="background:#1a1d2e;border:1px solid #2d3148;border-radius:14px;padding:0.95rem 1rem;">
          <div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(160px,1fr));gap:0.75rem;">
            <div>
              <div style="font-size:0.78rem;text-transform:uppercase;letter-spacing:0.1em;color:#cbd5e1;">Current UTC</div>
              <div id="bot-timing-now" style="font-size:1.05rem;font-weight:700;color:#f1f5f9;font-family:'DM Mono',monospace;"></div>
            </div>
            <div>
              <div style="font-size:0.78rem;text-transform:uppercase;letter-spacing:0.1em;color:#cbd5e1;">Next Bar Close UTC</div>
              <div id="bot-timing-close" style="font-size:1.05rem;font-weight:700;color:#f1f5f9;font-family:'DM Mono',monospace;"></div>
            </div>
            <div>
              <div style="font-size:0.78rem;text-transform:uppercase;letter-spacing:0.1em;color:#cbd5e1;">Countdown</div>
              <div id="bot-timing-countdown" style="font-size:1.2rem;font-weight:700;color:#4ade80;font-family:'DM Mono',monospace;"></div>
            </div>
            <div>
              <div style="font-size:0.78rem;text-transform:uppercase;letter-spacing:0.1em;color:#cbd5e1;">Bar Timeframe</div>
              <div style="font-size:1.05rem;font-weight:700;color:#f1f5f9;font-family:'DM Mono',monospace;">{timeframe_minutes} min</div>
            </div>
          </div>
          <script>
            const baseNowMs = {int(current_time_utc.timestamp() * 1000)};
            const nextCloseMs = {int(next_bar_close_utc.timestamp() * 1000)};
            const baseRemaining = {seconds_remaining};

            function formatUtc(tsMs) {{
              return new Date(tsMs).toISOString().replace("T", " ").slice(0, 19);
            }}

            function formatCountdown(totalSeconds) {{
              const safe = Math.max(totalSeconds, 0);
              const hours = String(Math.floor(safe / 3600)).padStart(2, "0");
              const minutes = String(Math.floor((safe % 3600) / 60)).padStart(2, "0");
              const seconds = String(safe % 60).padStart(2, "0");
              return `${{hours}}:${{minutes}}:${{seconds}}`;
            }}

            function tick() {{
              const elapsed = Math.floor((Date.now() - baseNowMs) / 1000);
              const currentMs = baseNowMs + elapsed * 1000;
              const remaining = baseRemaining - elapsed;
              document.getElementById("bot-timing-now").textContent = formatUtc(currentMs);
              document.getElementById("bot-timing-close").textContent = formatUtc(nextCloseMs);
              document.getElementById("bot-timing-countdown").textContent = formatCountdown(remaining);
            }}

            tick();
            setInterval(tick, 1000);
          </script>
        </div>
        """,
        height=140,
    )


# ── Error page ────────────────────────────────────────────────────────────────

def render_startup_error(exc: Exception) -> None:
    st.markdown(
        "<div class='ks-banner'>Bot initialization failed — see details below</div>",
        unsafe_allow_html=True,
    )
    st.error(f"{type(exc).__name__}: {exc}")
    st.markdown(
        "**Common fixes**\n"
        "- Run `pip install -r requirements.txt`\n"
        "- Confirm `.env` contains `ALPACA_API_KEY` and `ALPACA_API_SECRET`\n"
        "- Relaunch with `python -m streamlit run dashboard.py`"
    )
    with st.expander("Traceback"):
        st.exception(exc)


# ── Orders helper ─────────────────────────────────────────────────────────────

def _orders_dataframe(recent_orders: list) -> pd.DataFrame:
    if not recent_orders:
        return pd.DataFrame()
    df = pd.DataFrame([asdict(o) for o in recent_orders])
    cols = [
        c
        for c in ["submitted_at", "symbol", "side", "status", "qty", "filled_qty", "filled_avg_price", "notional"]
        if c in df.columns
    ]
    return df[cols]


def _preview_reason_label(reason: str | None) -> str:
    mapping = {
        None: "",
        "outside_entry_window": "Outside entry window",
        "already_holding": "Already holding",
        "max_open_positions_reached": "Max open positions reached",
        "stale_live_price": "Live price too stale",
        "price_collar_breached": "Live price too far from signal price",
        "symbol_exposure_exceeded": "Symbol exposure limit exceeded",
        "max_orders_per_minute": "Order rate limit reached",
        "open_order_in_flight": "Open order already in flight",
        "kill_switch_active": "Kill switch active",
        "not_holding": "No position to sell",
        "insufficient_buying_power": "Not enough buying power",
        "live_price_unavailable": "Live price unavailable",
    }
    return mapping.get(reason, str(reason).replace("_", " ").title())


def _execution_preview_dataframe(previews: list) -> pd.DataFrame:
    if not previews:
        return pd.DataFrame()
    rows = []
    for preview in previews:
        rows.append(
            {
                "symbol": preview.symbol,
                "signal": preview.action,
                "execution": preview.status,
                "reason": (
                    f"{_preview_reason_label(preview.reason)}: {preview.detail}"
                    if getattr(preview, "detail", None)
                    else (_preview_reason_label(preview.reason) or "Ready to submit")
                ),
                "live_price": round(preview.live_price, 2) if preview.live_price is not None else None,
                "signal_price": round(preview.signal_price, 2) if preview.signal_price is not None else None,
                "deviation_bps": round(preview.price_deviation_bps, 1) if preview.price_deviation_bps is not None else None,
                "price_age_s": round(preview.live_price_age_s, 1) if preview.live_price_age_s is not None else None,
            }
        )
    return pd.DataFrame(rows)


def _execution_status_badge(status: str) -> str:
    key = (status or "").upper()
    cls = {
        "READY": "b-ready",
        "BLOCKED": "b-blocked",
        "NO_SIGNAL": "b-nosignal",
        "ERROR": "b-err",
    }.get(key, "b-nosignal")
    label = {
        "READY": "READY NOW",
        "BLOCKED": "BLOCKED",
        "NO_SIGNAL": "NO SIGNAL",
        "ERROR": "ERROR",
    }.get(key, key or "UNKNOWN")
    return f"<span class='badge {cls}'>{label}</span>"


def _cycle_reason_label(reason: str) -> str:
    mapping = {
        "duplicate_bar": "Skipped: this bar was already processed",
        "market_clock_error": "Stopped: market clock lookup failed",
        "market_closed": "Stopped: market is closed",
        "outside_regular_hours": "Stopped: outside regular trading hours",
        "eod_flatten_window": "Stopped: end-of-day flatten window",
        "kill_switch_active": "Stopped: kill switch active",
        "preview_only": "Preview only: no orders allowed",
        "signals_blocked_or_skipped": "Signals found, but no orders were submitted",
        "no_actionable_signals": "No BUY/SELL signals on this cycle",
        "execution_completed": "Execution completed",
    }
    return mapping.get(reason, reason.replace("_", " ").title())


def _render_last_cycle_report(report) -> None:
    if report is None:
        st.info("No cycle has been run in this dashboard session yet.")
        return
    mode = "Live execution enabled" if report.execute_orders else "Preview only"
    headline = (
        f"{mode} | {report.orders_submitted} orders submitted | "
        f"{report.buy_signals} buy | {report.sell_signals} sell"
    )
    st.markdown("<div class='sec-head'>Last Run Cycle</div>", unsafe_allow_html=True)
    st.markdown(
        f"<div style='margin:0.2rem 0 0.5rem;font-size:0.92rem;color:#e2e8f0'>{headline}</div>",
        unsafe_allow_html=True,
    )
    st.markdown(
        f"<div style='font-size:0.84rem;color:#94a3b8;margin-bottom:1rem'>"
        f"{_cycle_reason_label(report.skip_reason)} | bar {report.decision_timestamp}"
        f"</div>",
        unsafe_allow_html=True,
    )


def _render_operator_status(bot, last_cycle_report) -> None:
    env_execute = os.getenv("EXECUTE_ORDERS", "true").lower() != "false"
    mode_label = "Auto-run live" if env_execute else "Auto-run preview"
    timing = _get_bar_timing(bot.config.bar_timeframe_minutes)
    countdown = timing["countdown"]
    subtitle = (
        "Normal workflow: run the external bot loop and use this dashboard as a monitor. "
        "Manual controls are available below for debugging."
    )
    if last_cycle_report is not None and last_cycle_report.processed_bar:
        last_text = f"Last processed bar: {last_cycle_report.decision_timestamp}"
    elif last_cycle_report is not None:
        last_text = f"Last cycle result: {_cycle_reason_label(last_cycle_report.skip_reason)}"
    else:
        last_text = "No cycle recorded in this dashboard session yet"
    st.markdown("<div class='sec-head'>Operator Mode</div>", unsafe_allow_html=True)
    st.markdown(
        f"<div style='margin:0.15rem 0 0.45rem;font-size:0.95rem;color:#f1f5f9'>{mode_label}</div>",
        unsafe_allow_html=True,
    )
    st.markdown(
        f"<div style='font-size:0.84rem;color:#94a3b8;margin-bottom:0.25rem'>{subtitle}</div>",
        unsafe_allow_html=True,
    )
    st.markdown(
        f"<div style='font-size:0.84rem;color:#cbd5e1'>Next bar in {countdown} | {last_text}</div>",
        unsafe_allow_html=True,
    )


def _position_rows(snapshot) -> list[dict[str, float | str | None]]:
    rows = []
    for sym, pos in snapshot.positions.items():
        match = next((s for s in snapshot.symbols if s.symbol == sym), None)
        rows.append(
            {
                "symbol": sym,
                "qty": float(pos.qty) if pos.qty is not None else 0.0,
                "market_value": float(pos.market_value) if pos.market_value is not None else 0.0,
                "avg_entry": float(pos.avg_entry_price) if pos.avg_entry_price is not None else 0.0,
                "unrealized_pl": float(pos.unrealized_pl) if pos.unrealized_pl is not None else 0.0,
                "held_mins": (
                    round(float(match.holding_minutes), 1)
                    if match and match.holding_minutes is not None
                    else None
                ),
            }
        )
    return rows


def _held_card_html(row: dict[str, float | str | None]) -> str:
    pnl = float(row["unrealized_pl"] or 0.0)
    pnl_cls = "m-pos" if pnl >= 0 else "m-neg"
    held_text = (
        f"{float(row['held_mins']):.0f} min"
        if row.get("held_mins") is not None
        else "n/a"
    )
    return (
        "<div class='held-card'>"
        f"<div class='held-top'><span class='held-symbol'>{row['symbol']}</span>"
        f"<span class='badge b-held'>OPEN</span></div>"
        f"<div class='held-main'>{_fmt_money(float(row['market_value'] or 0.0))}</div>"
        f"<div class='held-kv'><span class='held-kv-label'>Qty</span><span>{float(row['qty'] or 0.0):.2f}</span></div>"
        f"<div class='held-kv'><span class='held-kv-label'>Avg entry</span><span>{_fmt_money(float(row['avg_entry'] or 0.0))}</span></div>"
        f"<div class='held-kv'><span class='held-kv-label'>Unrealized P/L</span><span class='{pnl_cls}'>{_fmt_money(pnl)}</span></div>"
        f"<div class='held-kv'><span class='held-kv-label'>Held</span><span>{held_text}</span></div>"
        "</div>"
    )


def _position_signal_card_html(item, row: dict[str, float | str | None] | None, execution_preview=None) -> str:
    execution_html = ""
    if execution_preview is not None:
        execution_html = (
            f"<div style='margin-top:0.45rem'>{_execution_status_badge(execution_preview.status)}</div>"
            f"<div style='font-size:0.78rem;color:#94a3b8;margin-top:0.35rem'>"
            f"{(f'{_preview_reason_label(execution_preview.reason)}: {execution_preview.detail}') if getattr(execution_preview, 'detail', None) else (_preview_reason_label(execution_preview.reason) or 'Signal passes current execution checks')}"
            f"</div>"
        )

    if row is None:
        return _sym_card_html(item, execution_html)

    price_str = f"${item.price:,.2f}" if item.price is not None else "-"
    sma_str = f"SMA {item.sma:,.2f}" if item.sma is not None else ""
    held_str = (
        f"held {item.holding_minutes:.0f} min"
        if item.holding and item.holding_minutes is not None
        else ""
    )
    meta = "  |  ".join(filter(None, [sma_str, held_str]))
    badge = _badge_html(item.action or "HOLD")
    ml_bar = _ml_bar_html(
        item.ml_probability_up,
        getattr(item, "ml_buy_threshold", None),
        getattr(item, "ml_sell_threshold", None),
        item.ml_confidence,
    )
    prox_bar = _prox_bar_html(item.price, item.sma)
    error_html = (
        f"<div style='font-size:0.78rem;color:#f87171;margin-top:0.4rem'>{item.error}</div>"
        if item.error
        else ""
    )
    pnl = float(row["unrealized_pl"] or 0.0)
    return (
        "<div class='sym-card sym-held'>"
        f"<div class='sym-header'><span class='sym-ticker'>{item.symbol}</span><div><span class='badge b-held'>HELD</span>&nbsp;{badge}</div></div>"
        f"<div style='display:flex;justify-content:space-between;align-items:baseline'>"
        f"<span class='sym-price'>{price_str}</span>"
        f"<span style='font-size:0.78rem;color:#475569'>{meta}</span>"
        f"</div>"
        f"<div style='font-size:0.86rem;color:#e2e8f0;margin-top:0.35rem'>"
        f"qty {float(row['qty'] or 0.0):.2f} &nbsp;|&nbsp; value {_fmt_money(float(row['market_value'] or 0.0))}"
        f"</div>"
        f"<div style='font-size:0.86rem;color:#e2e8f0;margin-bottom:0.35rem'>"
        f"uPnL <span class='{_pnl_cls(pnl)}'>{_fmt_money(pnl)}</span>"
        f"</div>"
        f"{prox_bar}{ml_bar}{error_html}{execution_html}"
        "</div>"
    )


# ── Tab: Live ─────────────────────────────────────────────────────────────────

def _render_live(bot, snapshot, recent_orders: list, execution_previews: list | None = None) -> None:
    if snapshot.kill_switch_triggered:
        st.markdown(
            "<div class='ks-banner'>Kill Switch Active — daily loss limit reached, no new entries</div>",
            unsafe_allow_html=True,
        )

    pnl_cls = _pnl_cls(snapshot.daily_pnl)
    eq_delta = f"Prev close {_fmt_money(snapshot.last_equity)}"
    action_counts: dict[str, int] = {}
    for item in snapshot.symbols:
        action_counts[item.action] = action_counts.get(item.action, 0) + 1
    open_pos = len(snapshot.positions)
    position_rows = _position_rows(snapshot)
    position_lookup = {str(row["symbol"]): row for row in position_rows}
    decision_str = f"{action_counts.get('BUY', 0)} buy · {action_counts.get('SELL', 0)} sell"

    cols = st.columns(5)
    cols[0].markdown(_metric_html("Cash", _fmt_money(snapshot.cash)), unsafe_allow_html=True)
    cols[1].markdown(_metric_html("Buying Power", _fmt_money(snapshot.buying_power)), unsafe_allow_html=True)
    cols[2].markdown(_metric_html("Equity", _fmt_money(snapshot.equity), eq_delta), unsafe_allow_html=True)
    cols[3].markdown(
        _metric_html(
            "Daily PnL",
            _fmt_money(snapshot.daily_pnl),
            "Kill switch active" if snapshot.kill_switch_triggered else "Within limit",
            pnl_cls,
        ),
        unsafe_allow_html=True,
    )
    cols[4].markdown(
        _metric_html("Positions / Signals", f"{open_pos} open", decision_str),
        unsafe_allow_html=True,
    )

    has_execution_preview = execution_previews is not None
    if execution_previews is None:
        execution_previews = []
    preview_counts: dict[str, int] = {}
    for preview in execution_previews:
        preview_counts[preview.status] = preview_counts.get(preview.status, 0) + 1

    try:
        feed_status = bot.get_price_feed_status()
        dot_cls = "dot-live" if "live" in feed_status.lower() else "dot-rest"
    except Exception:
        feed_status = "unavailable"
        dot_cls = "dot-error"
    st.markdown(
        f"<div style='margin:0.6rem 0 1rem'>"
        f"<span class='dot {dot_cls}'></span>"
        f"<span style='font-size:0.88rem;color:#c7d2e3'>"
        f"{feed_status} &nbsp;·&nbsp; {snapshot.timestamp_utc}"
        f"</span></div>",
        unsafe_allow_html=True,
    )

    if not snapshot.symbols:
        st.info("No symbol data in snapshot.")
        return

    st.markdown("<div class='sec-head'>If You Run With Orders Enabled</div>", unsafe_allow_html=True)
    if not has_execution_preview:
        st.info("Run one cycle to see which signals are ready now, blocked, or missing live pricing.")
    elif any(item.action == "SNAPSHOT_ONLY" for item in snapshot.symbols):
        st.info("No current execution preview. This snapshot did not evaluate a new bar yet.")
    else:
        summary = (
            f"{preview_counts.get('READY', 0)} ready now | "
            f"{preview_counts.get('BLOCKED', 0)} blocked | "
            f"{preview_counts.get('NO_SIGNAL', 0)} no signal"
        )
        if preview_counts.get("ERROR", 0):
            summary += f" | {preview_counts.get('ERROR', 0)} error"
        st.markdown(
            f"<div style='margin:0.25rem 0 1rem;font-size:0.92rem;color:#cbd5e1'>{summary}</div>",
            unsafe_allow_html=True,
        )
        preview_df = _execution_preview_dataframe(execution_previews)
        if not preview_df.empty:
            st.dataframe(preview_df, use_container_width=True, hide_index=True)

    if position_rows:
        total_exposure = sum(float(row["market_value"]) for row in position_rows)
        total_unrealized = sum(float(row["unrealized_pl"]) for row in position_rows)
        held_cols = st.columns(3)
        held_cols[0].markdown(
            _metric_html("Open Exposure", _fmt_money(total_exposure), f"{open_pos} positions"),
            unsafe_allow_html=True,
        )
        held_cols[1].markdown(
            _metric_html("Unrealized P/L", _fmt_money(total_unrealized), "Across held names", _pnl_cls(total_unrealized)),
            unsafe_allow_html=True,
        )
        avg_hold_mins = [float(row["held_mins"]) for row in position_rows if row["held_mins"] is not None]
        held_cols[2].markdown(
            _metric_html(
                "Avg Hold Time",
                f"{(sum(avg_hold_mins) / len(avg_hold_mins)):.0f} min" if avg_hold_mins else "n/a",
                "Current open positions",
            ),
            unsafe_allow_html=True,
        )

        st.markdown("<div class='sec-head'>Held Now</div>", unsafe_allow_html=True)
        held_cards = "".join(
            _held_card_html(row)
            for row in sorted(position_rows, key=lambda row: float(row["market_value"]), reverse=True)
        )
        st.markdown(f"<div class='held-grid'>{held_cards}</div>", unsafe_allow_html=True)

    st.markdown("<div class='sec-head'>Symbol Signals</div>", unsafe_allow_html=True)
    card_cols = st.columns(3)
    ordered_symbols = sorted(snapshot.symbols, key=lambda item: (not item.holding, item.symbol))
    preview_lookup = {preview.symbol: preview for preview in execution_previews}
    for idx, item in enumerate(ordered_symbols):
        card_cols[idx % 3].markdown(
            _position_signal_card_html(item, position_lookup.get(item.symbol), preview_lookup.get(item.symbol)),
            unsafe_allow_html=True,
        )

    if position_rows:
        st.markdown("<div class='sec-head'>Open Positions</div>", unsafe_allow_html=True)
        positions_df = pd.DataFrame(position_rows).sort_values("market_value", ascending=False)
        st.dataframe(positions_df, use_container_width=True, hide_index=True)

    orders_df = _orders_dataframe(recent_orders)
    if not orders_df.empty:
        st.markdown("<div class='sec-head'>Recent Orders</div>", unsafe_allow_html=True)
        st.dataframe(orders_df, use_container_width=True, hide_index=True)


# ── Tab: History ──────────────────────────────────────────────────────────────

def _render_history(bot, snapshot) -> None:
    selected = st.radio("Symbol", bot.config.symbols, horizontal=True)

    try:
        run_history = pd.DataFrame(bot.storage.get_run_history(limit=200))
        symbol_history = pd.DataFrame(bot.storage.get_symbol_history(selected, limit=200))
    except Exception as exc:
        st.error(f"Could not load history: {exc}")
        return

    left, right = st.columns([1.4, 1])

    with left:
        st.markdown("<div class='sec-head'>Account Equity</div>", unsafe_allow_html=True)
        if run_history.empty:
            st.info("No run history yet.")
        else:
            rh = run_history.copy()
            rh["ts"] = parse_mixed_iso_timestamps(rh["timestamp_utc"])
            rh = rh.dropna(subset=["ts"]).sort_values("ts")
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=rh["ts"], y=rh["equity"], name="Equity", line=dict(color="#60a5fa", width=2)))
            fig.add_trace(
                go.Scatter(
                    x=rh["ts"], y=rh["daily_pnl"], name="Daily PnL",
                    line=dict(color="#4ade80", width=1.5), yaxis="y2",
                )
            )
            fig.update_layout(
                paper_bgcolor="#1a1d2e", plot_bgcolor="#1a1d2e",
                font=dict(family="DM Sans, sans-serif", color="#94a3b8", size=11),
                margin=dict(l=0, r=0, t=20, b=0),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, bgcolor="rgba(0,0,0,0)"),
                xaxis=dict(gridcolor="#2d3148"),
                yaxis=dict(gridcolor="#2d3148", title="Equity"),
                yaxis2=dict(overlaying="y", side="right", title="Daily PnL", gridcolor="rgba(0,0,0,0)"),
                height=260,
            )
            st.plotly_chart(fig, use_container_width=True)

        st.markdown(f"<div class='sec-head'>{selected} — Price vs SMA</div>", unsafe_allow_html=True)
        if symbol_history.empty:
            st.info("No symbol history yet.")
        else:
            sh = symbol_history.copy()
            sh["ts"] = parse_mixed_iso_timestamps(sh["timestamp_utc"])
            sh = sh.dropna(subset=["ts", "price", "sma"]).sort_values("ts")
            if sh.empty:
                st.info("No price/SMA data saved yet.")
            else:
                fig2 = go.Figure()
                fig2.add_trace(go.Scatter(x=sh["ts"], y=sh["price"], name="Price", line=dict(color="#f1f5f9", width=2)))
                fig2.add_trace(
                    go.Scatter(x=sh["ts"], y=sh["sma"], name="SMA", line=dict(color="#facc15", width=1.5, dash="dot"))
                )
                fig2.update_layout(
                    paper_bgcolor="#1a1d2e", plot_bgcolor="#1a1d2e",
                    font=dict(family="DM Sans, sans-serif", color="#94a3b8", size=11),
                    margin=dict(l=0, r=0, t=20, b=0),
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, bgcolor="rgba(0,0,0,0)"),
                    xaxis=dict(gridcolor="#2d3148"),
                    yaxis=dict(gridcolor="#2d3148"),
                    height=240,
                )
                st.plotly_chart(fig2, use_container_width=True)

            if "ml_probability_up" in symbol_history.columns:
                ml_sh = sh.dropna(subset=["ml_probability_up"]) if not sh.empty else pd.DataFrame()
                if not ml_sh.empty:
                    fig3 = go.Figure()
                    fig3.add_trace(
                        go.Scatter(x=ml_sh["ts"], y=ml_sh["ml_probability_up"], name="ML Prob", line=dict(color="#a78bfa", width=2))
                    )
                    if "ml_confidence" in ml_sh.columns:
                        fig3.add_trace(
                            go.Scatter(
                                x=ml_sh["ts"], y=ml_sh["ml_confidence"], name="Confidence",
                                line=dict(color="#64748b", width=1, dash="dot"),
                            )
                        )
                    fig3.add_hline(y=0.5, line=dict(color="#475569", dash="dash"), annotation_text="0.5")
                    fig3.update_layout(
                        paper_bgcolor="#1a1d2e", plot_bgcolor="#1a1d2e",
                        font=dict(family="DM Sans, sans-serif", color="#94a3b8", size=11),
                        margin=dict(l=0, r=0, t=20, b=0),
                        legend=dict(orientation="h", yanchor="bottom", y=1.02, bgcolor="rgba(0,0,0,0)"),
                        xaxis=dict(gridcolor="#2d3148"),
                        yaxis=dict(gridcolor="#2d3148", range=[0, 1]),
                        height=200,
                    )
                    st.plotly_chart(fig3, use_container_width=True)

    with right:
        st.markdown(f"<div class='sec-head'>{selected} — Decision Mix</div>", unsafe_allow_html=True)
        if not symbol_history.empty and "action" in symbol_history.columns:
            mix = symbol_history["action"].value_counts().rename_axis("action").reset_index(name="count")
            st.dataframe(mix, use_container_width=True, hide_index=True)

        st.markdown(f"<div class='sec-head'>{selected} — Latest Rows</div>", unsafe_allow_html=True)
        if not symbol_history.empty:
            latest = symbol_history.tail(10).copy()
            if "holding_minutes" in latest.columns:
                latest["holding_minutes"] = pd.to_numeric(latest["holding_minutes"], errors="coerce").round(1)
            st.dataframe(latest, use_container_width=True, hide_index=True)


# ── Tab: Config ───────────────────────────────────────────────────────────────

def _render_config(bot, snapshot) -> None:
    cfg = bot.config
    left, right = st.columns(2)

    with left:
        st.markdown("<div class='sec-head'>Strategy</div>", unsafe_allow_html=True)
        st.markdown(
            "".join(
                _kv(k, v)
                for k, v in [
                    ("Mode", cfg.strategy_mode),
                    ("Symbols", ", ".join(cfg.symbols)),
                    ("Bar timeframe", f"{cfg.bar_timeframe_minutes} min"),
                    ("SMA bars", str(cfg.sma_bars)),
                    ("Paper mode", str(cfg.paper)),
                ]
            ),
            unsafe_allow_html=True,
        )

        st.markdown("<div class='sec-head'>Session</div>", unsafe_allow_html=True)
        session_rows = [
            ("Snapshot time", str(snapshot.timestamp_utc)),
            ("History DB", str(bot.storage.db_path)),
        ]
        try:
            session_rows.append(("Price feed", bot.get_price_feed_status()))
        except Exception:
            pass
        st.markdown("".join(_kv(k, v) for k, v in session_rows), unsafe_allow_html=True)

    with right:
        st.markdown("<div class='sec-head'>Risk Controls</div>", unsafe_allow_html=True)
        st.markdown(
            "".join(
                _kv(k, v)
                for k, v in [
                    ("Max per trade", _fmt_money(cfg.max_usd_per_trade)),
                    ("Max symbol exposure", _fmt_money(cfg.max_symbol_exposure_usd)),
                    ("Max open positions", str(cfg.max_open_positions)),
                    ("Max daily loss", _fmt_money(cfg.max_daily_loss_usd)),
                    ("Max orders/min", str(cfg.max_orders_per_minute)),
                    ("Price collar", f"{cfg.max_price_deviation_bps:.0f} bps"),
                    ("Max live price age", f"{cfg.max_live_price_age_seconds}s"),
                    ("Max bar delay", f"{cfg.max_data_delay_seconds}s"),
                ]
            ),
            unsafe_allow_html=True,
        )

        st.markdown("<div class='sec-head'>ML Thresholds</div>", unsafe_allow_html=True)
        model_names = sorted(
            {getattr(item, "ml_model_name", None) for item in snapshot.symbols} - {None}
        )
        if model_names:
            st.markdown(_kv("Model", ", ".join(model_names)), unsafe_allow_html=True)
        ml_rows = [
            f"{item.symbol}: buy ≥ {item.ml_buy_threshold:.2f}  sell ≤ {item.ml_sell_threshold:.2f}"
            for item in snapshot.symbols
            if getattr(item, "ml_buy_threshold", None) is not None
        ]
        if ml_rows:
            st.markdown("".join(_kv("Threshold", r) for r in ml_rows), unsafe_allow_html=True)
        else:
            st.markdown(_kv("Thresholds", "n/a"), unsafe_allow_html=True)


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    _inject_css()

    try:
        bot = get_bot()
        bot = _ensure_bot_capabilities(bot)
    except Exception as exc:
        render_startup_error(exc)
        return

    if "snapshot" not in st.session_state:
        try:
            st.session_state.snapshot, st.session_state.recent_orders = bot.capture_state()
        except Exception as exc:
            render_startup_error(exc)
            return

    if "execute_orders" not in st.session_state:
        st.session_state.execute_orders = False
    if "run_pending" not in st.session_state:
        st.session_state.run_pending = False
    if "force_run_pending" not in st.session_state:
        st.session_state.force_run_pending = False
    if "execution_previews" not in st.session_state:
        st.session_state.execution_previews = None
    if "last_cycle_report" not in st.session_state:
        st.session_state.last_cycle_report = None

    snapshot = st.session_state.snapshot
    recent_orders = st.session_state.get("recent_orders", [])
    execution_previews = st.session_state.execution_previews
    last_cycle_report = st.session_state.last_cycle_report
    current_decision_timestamp = bot.get_decision_timestamp().isoformat()
    already_ran_current_bar = (
        last_cycle_report is not None
        and getattr(last_cycle_report, "decision_timestamp", None) == current_decision_timestamp
    )

    # Header row
    hdr_l, hdr_r = st.columns([2, 1.4])
    with hdr_l:
        st.markdown(
            "<div style='font-size:1.4rem;font-weight:700;color:#f1f5f9;margin-bottom:0.1rem'>Alpaca Bot</div>",
            unsafe_allow_html=True,
        )
        st.markdown(
            f"<div style='font-size:0.86rem;color:#c7d2e3'>"
            f"{bot.config.strategy_mode} · {len(bot.config.symbols)} symbols · "
            f"{'paper' if bot.config.paper else 'live'}"
            f"</div>",
            unsafe_allow_html=True,
        )

    with hdr_r:
        _render_operator_status(bot, last_cycle_report)
        st.caption("Manual controls below are for debugging only.")
        ctl_a, ctl_b, ctl_c, ctl_d = st.columns([1, 1, 1, 1])
        refresh = ctl_a.button("Refresh View", use_container_width=True, type="primary")
        execute_orders = ctl_b.toggle("Allow order placement", value=st.session_state.execute_orders)
        st.session_state.execute_orders = execute_orders

        if not st.session_state.run_pending and not st.session_state.force_run_pending:
            run_cycle = ctl_c.button(
                "Run manual cycle",
                use_container_width=True,
                disabled=already_ran_current_bar,
            )
            force_run_cycle = ctl_d.button(
                "Force debug cycle",
                use_container_width=True,
            )
            if already_ran_current_bar:
                ctl_c.caption("Already ran this bar. Wait for next bar close.")
            else:
                ctl_c.caption("Normal operation should be automated.")
            ctl_d.caption("Bypasses duplicate-bar protection for debugging.")
            if run_cycle:
                st.session_state.run_pending = True
                st.rerun()
            if force_run_cycle:
                st.session_state.force_run_pending = True
                st.rerun()
        else:
            mode_label = "with order execution" if execute_orders else "preview — no orders"
            st.warning(f"Run one manual cycle ({mode_label}). Confirm?")
            is_force_run = st.session_state.force_run_pending
            conf_l, conf_r = st.columns(2)
            if conf_l.button("Confirm", type="primary", use_container_width=True):
                try:
                    st.session_state.snapshot = bot.run_once(
                        execute_orders=execute_orders,
                        force_process=is_force_run,
                    )
                    st.session_state.recent_orders = bot.get_recent_orders(limit=12)
                    st.session_state.last_cycle_report = bot.get_last_run_cycle_report()
                    last_cycle_report = st.session_state.last_cycle_report
                    try:
                        st.session_state.execution_previews = bot.preview_execution(st.session_state.snapshot)
                    except Exception:
                        st.session_state.execution_previews = []
                    execution_previews = st.session_state.execution_previews
                except Exception as exc:
                    render_startup_error(exc)
                    return
                finally:
                    st.session_state.run_pending = False
                    st.session_state.force_run_pending = False
                st.rerun()
            if conf_r.button("Cancel", use_container_width=True):
                st.session_state.run_pending = False
                st.session_state.force_run_pending = False
                st.rerun()

    if refresh:
        try:
            st.session_state.snapshot, st.session_state.recent_orders = bot.capture_state()
            st.session_state.execution_previews = None
            snapshot = st.session_state.snapshot
            recent_orders = st.session_state.recent_orders
            execution_previews = st.session_state.execution_previews
        except Exception as exc:
            render_startup_error(exc)
            return

    _render_bar_timing(bot)
    st.divider()

    live_tab, history_tab, config_tab = st.tabs(["Live", "History", "Config"])

    with live_tab:
        _render_last_cycle_report(last_cycle_report)
        _render_live(bot, snapshot, recent_orders, execution_previews)

    with history_tab:
        _render_history(bot, snapshot)

    with config_tab:
        _render_config(bot, snapshot)


if __name__ == "__main__":
    main()

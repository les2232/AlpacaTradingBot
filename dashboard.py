from dataclasses import asdict
from datetime import datetime, time as dt_time, timedelta, timezone
import html
from pathlib import Path
from types import SimpleNamespace
from zoneinfo import ZoneInfo

_ET = ZoneInfo("America/New_York")
_SESSION_FLATTEN_AT = dt_time(15, 55)
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components
from dashboard_state import DashboardState, DrilldownEvent, load_dashboard_state
from research.experiment_log import DEFAULT_LOG_PATH, load_experiment_log_frame

st.set_page_config(page_title="TradeOS", layout="wide")


def _strategy_mode_uses_ml(strategy_mode: str | None) -> bool:
    return str(strategy_mode or "").strip().lower() in {"ml", "hybrid"}


def _live_bot_process_health() -> tuple[str, str]:
    lock_path = Path.cwd() / ".live_bot.lock"
    try:
        from alpaca_trading_bot.cli import _live_lock_matches_running_process, _read_live_lock_metadata
    except Exception:
        return "unknown", "Live bot process health is unavailable in this dashboard context."

    try:
        metadata = _read_live_lock_metadata()
    except Exception:
        return "unknown", "Live bot process metadata could not be read safely."

    if not metadata:
        if lock_path.exists():
            return "stale_lock", "A live bot lock file exists but could not be parsed."
        return "missing", "No active TradeOS live process was detected."

    pid = int(metadata.get("pid", 0) or 0)
    if _live_lock_matches_running_process(metadata):
        return "running", f"Live bot process detected (pid={pid})."
    if pid > 0:
        return "stale_lock", f"Live bot lock points to pid={pid}, but that process is no longer healthy."
    return "stale_lock", "A stale live bot lock is present."


def _process_health_summary() -> tuple[str, str, str]:
    health, note = _live_bot_process_health()
    if health == "running":
        return "Process live", note, "good"
    if health == "unknown":
        return "Process unknown", note, "warn"
    return "Process offline", note, "err"


def _resync_status_summary(state: DashboardState) -> tuple[str, str, str]:
    cfg = state.startup_config
    status = str(getattr(cfg, "resync_status", "") or "").strip().upper()
    if not status:
        return "Resync unknown", "This session does not have persisted startup resync metadata yet.", "warn"
    if status == "RESYNC_OK":
        return "Resync OK", "Broker state was recovered and the bot is allowed to trade normally.", "good"
    if status == "RESYNC_DEGRADED":
        return "Resync degraded", "Broker exposure was recovered, but the bot should remain exits-only.", "warn"
    if status == "RESYNC_FAILED":
        return "Resync failed", "Startup reconciliation did not finish safely, so trading should remain blocked.", "err"
    if status == "RESYNC_LOCKED":
        return "Resync locked", "Startup reconciliation is still in progress, so new trading should remain blocked.", "warn"
    return f"Resync {status.lower()}", f"Startup reconciliation reported {status}.", "warn"


# â”€â”€ CSS â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def _inject_css() -> None:
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=DM+Sans:wght@400;500;700&display=swap');
        :root {
            --font-family-ui: Inter, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;

            --font-size-xxs: 12px;
            --font-size-xs: 13px;
            --font-size-sm: 15px;
            --font-size-md: 16px;
            --font-size-lg: 18px;
            --font-size-xl: 22px;
            --font-size-2xl: 30px;

            --font-weight-regular: 400;
            --font-weight-medium: 500;
            --font-weight-semibold: 600;
            --font-weight-bold: 700;

            --text-primary: #E6EDF3;
            --text-secondary: #9DA7B3;
            --text-muted: #6B7480;

            --line-height-tight: 1.2;
            --line-height-snug: 1.35;
            --line-height-normal: 1.48;
            --line-height-relaxed: 1.58;

            --letter-spacing-tight: -0.02em;
            --letter-spacing-normal: 0;
            --letter-spacing-wide: 0.08em;

            --bg: #0B0F14;
            --panel: #151B23;
            --panel-strong: #151B23;
            --panel-soft: #151B23;
            --line: #1F2733;
            --text: var(--text-primary);
            --text-soft: var(--text-secondary);
            --text-dim: var(--text-muted);
            --accent: #E6EDF3;
            --accent-2: #E6EDF3;
            --glow: none;
            --success: #3FB950;
            --warning: #D29922;
            --error: #F85149;
            --success-bg: rgba(63, 185, 80, 0.12);
            --warning-bg: rgba(210, 153, 34, 0.12);
            --error-bg: rgba(248, 81, 73, 0.12);
            --muted-bg: rgba(157, 167, 179, 0.10);
        }
        html, body, [class*="css"] {
            font-family: var(--font-family-ui);
            font-size: var(--font-size-md);
            font-weight: var(--font-weight-regular);
            line-height: var(--line-height-normal);
            letter-spacing: var(--letter-spacing-normal);
        }
        code, pre, .mono { font-family: 'DM Mono', 'Consolas', monospace; }

        .stApp {
            background: var(--bg);
            color: var(--text);
        }
        .block-container {
            padding-top: 1.2rem;
            padding-bottom: 2.8rem;
            max-width: 1600px;
        }

        /* Kill switch banner */
        .ks-banner {
            background: rgba(127,29,29,0.65);
            border: 0.5px solid rgba(220,38,38,0.45);
            border-radius: 8px;
            padding: 0.65rem 1rem;
            color: #fca5a5;
            font-weight: 600;
            font-size: 0.88rem;
            margin-bottom: 1.2rem;
            letter-spacing: 0.02em;
        }

        .m-pos { color: #4ade80; }
        .m-neg { color: #f87171; }

        /* Feed status dot */
        .dot {
            display: inline-block;
            width: 7px; height: 7px;
            border-radius: 50%;
            margin-right: 6px;
            vertical-align: middle;
        }
        .dot-live  { background: #4ade80; box-shadow: 0 0 5px rgba(74,222,128,0.55); }
        .dot-rest  { background: #facc15; }
        .dot-error { background: #f87171; }

        /* Badges â€” rectangular chip style, DM Mono, semi-transparent tints */
        .badge {
            display: inline-block;
            border-radius: 4px;
            padding: 0.16rem 0.58rem;
            font-size: 0.84rem;
            font-weight: 700;
            letter-spacing: 0.07em;
            font-family: 'DM Mono', monospace;
            text-transform: uppercase;
        }
        .b-buy     { background: rgba(74,222,128,0.13);  color: #4ade80; }
        .b-sell    { background: rgba(248,113,113,0.13); color: #f87171; }
        .b-hold    { background: rgba(96,165,250,0.11);  color: #7db6f7; }
        .b-err     { background: rgba(239,159,39,0.13);  color: #ef9f27; }
        .b-ready   { background: rgba(74,222,128,0.11);  color: #86efac; }
        .b-blocked { background: rgba(248,113,113,0.11); color: #fca5a5; }
        .b-nosignal { background: rgba(100,116,139,0.16); color: #8fa3ba; }

        /* ML probability bar */
        .ml-wrap { margin: 0.6rem 0 0.3rem; }
        .ml-label {
            font-size: 0.8rem;
            color: #4a5c6e;
            margin-bottom: 0.32rem;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            font-family: 'DM Sans', sans-serif;
        }
        .ml-track {
            position: relative;
            height: 8px;
            background: rgba(255,255,255,0.06);
            border-radius: 4px;
            overflow: visible;
        }
        .ml-fill { height: 100%; border-radius: 4px; }
        .ml-tick {
            position: absolute;
            top: -3px;
            width: 1.5px;
            height: 14px;
            background: rgba(248,250,252,0.65);
            border-radius: 1px;
        }
        .ml-axis {
            display: flex;
            justify-content: space-between;
            font-size: 0.75rem;
            color: #4a5c6e;
            margin-top: 0.24rem;
            font-family: 'DM Mono', monospace;
        }

        /* Proximity bar */
        .prox-wrap { margin: 0.4rem 0 0.65rem; }
        .prox-label {
            font-size: 0.8rem;
            color: #4a5c6e;
            margin-bottom: 0.3rem;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            font-family: 'DM Sans', sans-serif;
        }
        .prox-track { position: relative; height: 5px; background: rgba(255,255,255,0.06); border-radius: 3px; }
        .prox-fill  { height: 100%; border-radius: 3px; }
        .prox-above { background: linear-gradient(90deg, rgba(22,101,52,0.55), #4ade80); }
        .prox-below { background: linear-gradient(90deg, rgba(127,29,29,0.55), #f87171); }

        /* KV config layout */
        .kv {
            display: flex;
            justify-content: space-between;
            align-items: baseline;
            padding: 0.5rem 0;
            border-bottom: 0.5px solid rgba(255,255,255,0.05);
        }
        .kv-key { font-size: 1.02rem; color: #6b7a90; }
        .kv-val { font-size: 1.08rem; font-family: 'DM Mono', monospace; color: #d8e0ec; }

        /* Section headings */
        .sec-head {
            font-size: 0.86rem;
            text-transform: uppercase;
            letter-spacing: 0.18em;
            color: #86a3c4;
            margin: 1.1rem 0 0.72rem;
            font-family: 'DM Sans', sans-serif;
            font-weight: 600;
        }

        /* â”€â”€ v2 dashboard layout â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€ */
        .config-stack {
            display: flex;
            flex-direction: column;
            gap: 16px;
        }
        .config-card {
            position: relative;
            padding: 18px 18px 16px;
            border-radius: 20px;
            border: 1px solid rgba(126, 157, 194, 0.12);
            background:
                linear-gradient(180deg, rgba(17, 27, 41, 0.98), rgba(10, 16, 26, 0.98)),
                radial-gradient(circle at top right, rgba(103, 184, 255, 0.12), transparent 32%);
            box-shadow: 0 18px 44px rgba(3, 7, 15, 0.28);
            overflow: hidden;
        }
        .config-card::after {
            content: "";
            position: absolute;
            inset: 0;
            background: linear-gradient(135deg, rgba(255,255,255,0.04), transparent 24%, transparent 76%, rgba(255,255,255,0.02));
            pointer-events: none;
        }
        .config-card-top {
            display: flex;
            justify-content: space-between;
            align-items: flex-start;
            gap: 14px;
            margin-bottom: 14px;
        }
        .config-card-title {
            font-size: 0.86rem;
            text-transform: uppercase;
            letter-spacing: 0.16em;
            color: #86a3c4;
            font-weight: 700;
        }
        .config-card-note {
            font-size: 0.9rem;
            color: #93a7bf;
            line-height: 1.45;
            max-width: 24rem;
        }
        .config-hero {
            font-family: 'DM Mono', monospace;
            font-size: 0.82rem;
            color: #cde5ff;
            padding: 0.34rem 0.62rem;
            border-radius: 999px;
            background: rgba(103, 184, 255, 0.12);
            border: 1px solid rgba(103, 184, 255, 0.2);
            white-space: nowrap;
        }
        .config-stat-grid {
            display: grid;
            grid-template-columns: repeat(2, minmax(0, 1fr));
            gap: 12px;
        }
        .config-stat {
            padding: 13px 14px 12px;
            border-radius: 16px;
            background: rgba(255,255,255,0.032);
            border: 1px solid rgba(255,255,255,0.065);
            min-height: 92px;
        }
        .config-stat-label {
            font-size: 0.73rem;
            text-transform: uppercase;
            letter-spacing: 0.12em;
            color: #7590ae;
            margin-bottom: 0.48rem;
        }
        .config-stat-value {
            font-family: 'DM Mono', monospace;
            font-size: 1.08rem;
            line-height: 1.35;
            color: #edf4ff;
            font-weight: 700;
            overflow-wrap: anywhere;
        }
        .config-stat-sub {
            margin-top: 0.42rem;
            font-size: 0.88rem;
            color: #92a7bf;
            line-height: 1.4;
        }
        .config-band {
            margin-top: 14px;
            padding-top: 14px;
            border-top: 1px solid rgba(255,255,255,0.06);
        }
        .config-band-label {
            font-size: 0.72rem;
            text-transform: uppercase;
            letter-spacing: 0.12em;
            color: #7590ae;
            margin-bottom: 0.6rem;
        }
        .config-chip-cloud {
            display: flex;
            flex-wrap: wrap;
            gap: 8px;
        }
        .config-chip {
            display: inline-flex;
            align-items: center;
            padding: 0.38rem 0.68rem;
            border-radius: 999px;
            background: rgba(255,255,255,0.045);
            border: 1px solid rgba(255,255,255,0.075);
            color: #e6f0fb;
            font-family: 'DM Mono', monospace;
            font-size: 0.84rem;
            line-height: 1;
        }
        .config-chip.muted {
            color: #9eb2c8;
            background: rgba(255,255,255,0.03);
        }
        .config-list {
            display: flex;
            flex-direction: column;
            gap: 10px;
        }
        .config-row {
            display: grid;
            grid-template-columns: minmax(140px, 180px) minmax(0, 1fr);
            gap: 12px;
            align-items: start;
            padding: 11px 0;
            border-bottom: 1px solid rgba(255,255,255,0.055);
        }
        .config-row:last-child {
            border-bottom: none;
            padding-bottom: 0;
        }
        .config-row-key {
            font-size: 0.8rem;
            text-transform: uppercase;
            letter-spacing: 0.12em;
            color: #6f88a5;
        }
        .config-row-val {
            font-family: 'DM Mono', monospace;
            font-size: 0.93rem;
            line-height: 1.55;
            color: #edf4ff;
            overflow-wrap: anywhere;
        }
        .config-empty {
            padding: 16px;
            border-radius: 16px;
            background: rgba(255,255,255,0.032);
            border: 1px dashed rgba(126, 157, 194, 0.18);
            color: #9fb3c8;
            font-size: 0.96rem;
        }

        @media (max-width: 1100px) {
            .config-stat-grid { grid-template-columns: 1fr; }
            .config-row { grid-template-columns: 1fr; gap: 6px; }
        }

        .dash-metrics {
            display: grid;
            grid-template-columns: repeat(5, 1fr);
            background: linear-gradient(180deg, rgba(17, 26, 40, 0.96), rgba(12, 18, 29, 0.96));
            border: 1px solid rgba(123, 152, 187, 0.12);
            border-radius: 18px;
            overflow: hidden;
            margin-bottom: 1.25rem;
            box-shadow: var(--glow);
        }
        .dash-metric { padding: 18px 22px; border-right: 1px solid rgba(255,255,255,0.05); }
        .dash-metric:last-child { border-right: none; }
        .dash-metric-label { font-size: 12px; text-transform: uppercase; letter-spacing: .1em; color: #61758d; margin-bottom: 7px; font-family: 'DM Sans', sans-serif; }
        .dash-metric-val { font-size: 28px; font-weight: 700; font-family: 'DM Mono', monospace; }

        /* Watchlist cards */
        .sym-card-v2 { padding: 12px 15px; border-bottom: 0.5px solid rgba(255,255,255,0.05); cursor: pointer; border-left: 2px solid transparent; transition: background 0.1s; }
        .sym-card-v2:hover { background: rgba(255,255,255,0.03); }
        .sym-card-v2.sym-error  { border-left-color: #e24b4a; border-left-width: 3px; background: rgba(226,75,74,0.07); }
        .sym-card-v2.sym-active { border-left-color: #4a90d9; border-left-width: 3px; background: rgba(74,144,217,0.06); }
        .sym-card-v2.sym-s-buy  { border-left-color: #4ade80; border-left-width: 3px; }
        .sym-card-v2.sym-s-sell { border-left-color: #f87171; border-left-width: 3px; }
        .sym-card-v2-top { display: flex; justify-content: space-between; align-items: center; }
        .sym-card-v2-meta { font-size: 0.98rem; color: #4a5c6e; margin-top: 8px; display: flex; justify-content: space-between; }
        .sym-ticker { font-weight: 700; font-family: 'DM Mono', monospace; font-size: 1.14rem; color: #d8e0ec; }
        /* Card-specific badge and mini bar sizing */
        .sym-card-v2 .badge { font-size: 0.73rem; padding: 0.12rem 0.48rem; }
        .sym-card-v2 .ml-track { height: 4px; border-radius: 2px; margin: 8px 0 3px; }
        .sym-card-v2 .ml-fill  { border-radius: 2px; }
        .sym-card-v2 .ml-tick  { top: -4px; height: 12px; }
        .dm-mono { font-family: 'DM Mono', monospace; font-size: 13px; color: #4a5c6e; }

        /* Detail panel */
        .detail-error-banner { background: rgba(226,75,74,0.09); border: 0.5px solid rgba(226,75,74,0.28); border-radius: 8px; padding: 12px 15px; margin-bottom: 15px; font-family: 'DM Mono', monospace; font-size: 13px; color: #e05c5c; }
        .detail-info-grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: 10px; margin-bottom: 18px; }
        .detail-info-cell { background: rgba(255,255,255,0.025); border: 0.5px solid rgba(255,255,255,0.07); border-radius: 9px; padding: 13px 14px; }
        .detail-info-cell-label { font-size: 11px; text-transform: uppercase; letter-spacing: .11em; color: #5d738d; margin-bottom: 7px; font-family: 'DM Sans', sans-serif; }
        .detail-info-cell-val { font-size: 18px; font-weight: 700; font-family: 'DM Mono', monospace; color: #d8e0ec; }

        /* Right rail */
        .rail-alert { display: flex; gap: 8px; align-items: flex-start; padding: 10px 0; border-bottom: 0.5px solid rgba(255,255,255,0.05); font-size: 13px; }
        .rail-alert:last-child { border-bottom: none; }
        .rail-dot { width: 6px; height: 6px; border-radius: 50%; margin-top: 4px; flex-shrink: 0; }
        .rail-dot-err  { background: #e24b4a; box-shadow: 0 0 4px rgba(226,75,74,0.45); }
        .rail-dot-warn { background: #ef9f27; }
        .rail-alert-syms { font-size: 13px; font-weight: 600; color: #e05c5c; margin-top: 3px; font-family: 'DM Mono', monospace; }

        .kill-switch-row { display: flex; justify-content: space-between; align-items: center; padding: 10px 13px; background: rgba(226,75,74,0.08); border: 0.5px solid rgba(226,75,74,0.2); border-radius: 7px; margin-top: 10px; font-size: 13px; color: #e05c5c; font-weight: 500; }

        .rail-empty { font-size: 15px; color: #6a7f97; padding: 10px 0; font-style: italic; }
        .rail-row { padding: 11px 0; border-bottom: 0.5px solid rgba(255,255,255,0.05); }
        .rail-row-top { display: flex; justify-content: space-between; align-items: center; }
        .rail-row-meta { display: flex; justify-content: space-between; font-size: 13px; color: #6d8197; margin-top: 5px; }
        .rail-row-val { font-size: 1.04rem; color: #d8e0ec; font-family: 'DM Mono', monospace; }
        .rail-sym { font-weight: 700; font-family: 'DM Mono', monospace; font-size: 1.08rem; color: #d8e0ec; }
        .order-row {
            padding: 12px 0 13px;
            border-bottom: 1px solid rgba(255,255,255,0.045);
        }
        .order-row:last-child { border-bottom: none; }
        .order-row-top {
            display: flex;
            justify-content: space-between;
            align-items: center;
            gap: 10px;
        }
        .order-main {
            display: flex;
            align-items: center;
            gap: 10px;
            min-width: 0;
        }
        .order-action {
            font-family: 'DM Mono', monospace;
            font-size: 0.92rem;
            font-weight: 800;
            letter-spacing: 0.05em;
            text-transform: uppercase;
        }
        .order-action.buy { color: #4ade80; }
        .order-action.sell { color: #f87171; }
        .order-symbol {
            font-family: 'DM Mono', monospace;
            font-size: 0.96rem;
            font-weight: 700;
            color: #ecf2f9;
        }
        .order-price {
            font-family: 'DM Mono', monospace;
            font-size: 1rem;
            font-weight: 700;
            color: #f8fbff;
            text-align: right;
            white-space: nowrap;
        }
        .order-meta {
            display: flex;
            justify-content: space-between;
            align-items: center;
            gap: 10px;
            margin-top: 6px;
        }
        .order-detail {
            font-size: 0.8rem;
            color: #8399b0;
        }
        .order-time {
            font-family: 'DM Mono', monospace;
            font-size: 0.76rem;
            color: #71859c;
            white-space: nowrap;
        }

        /* Detail history table */
        .hist-wrap {
            max-height: 320px;
            overflow-y: auto;
            border: 1px solid rgba(255,255,255,0.07);
            background: linear-gradient(180deg, rgba(255,255,255,0.022), rgba(255,255,255,0.012));
            border-radius: 14px;
            box-shadow: inset 0 1px 0 rgba(255,255,255,0.02);
        }
        .hist-table { width: 100%; border-collapse: collapse; font-size: 13px; }
        .hist-table th {
            position: sticky;
            top: 0;
            background: rgba(12,16,23,0.97);
            padding: 11px 13px;
            text-align: left;
            font-size: 10px;
            text-transform: uppercase;
            letter-spacing: .14em;
            color: #7489a2;
            border-bottom: 1px solid rgba(255,255,255,0.07);
            font-weight: 700;
            font-family: 'DM Sans', sans-serif;
        }
        .hist-table td {
            padding: 12px 13px;
            border-bottom: 1px solid rgba(255,255,255,0.045);
            color: #d5dfeb;
            vertical-align: top;
        }
        .hist-table tbody tr:nth-child(odd) td {
            background: rgba(255,255,255,0.012);
        }
        .hist-table tr:last-child td { border-bottom: none; }
        .hist-table .mono { font-family: 'DM Mono', monospace; color: #b5c4d5; }
        .hist-action-buy  { color: #4ade80; font-weight: 600; }
        .hist-action-sell { color: #f87171; font-weight: 600; }
        .hist-action-hold { color: #96a9be; font-weight: 600; }
        .hist-action-err  { color: #ef9f27; font-weight: 600; }
        .hist-action-snapshot { color: #8fb8ff; font-weight: 700; }

        /* Tabs */
        [data-testid="stTabs"] { margin-top: 0.1rem; }

        /* Watchlist card+button overlay â€” button is an invisible full-cover click target */
        [data-testid="column"]:first-child > [data-testid="stVerticalBlock"] > [data-testid="stVerticalBlock"] {
            position: relative;
        }
        [data-testid="column"]:first-child > [data-testid="stVerticalBlock"] > [data-testid="stVerticalBlock"] [data-testid="stButton"] {
            position: absolute;
            inset: 0;
            z-index: 10;
        }
        [data-testid="column"]:first-child > [data-testid="stVerticalBlock"] > [data-testid="stVerticalBlock"] [data-testid="stButton"] > button {
            width: 100%;
            height: 100%;
            opacity: 0;
            cursor: pointer;
            background: transparent;
            border: none;
        }

        /* Workspace refresh */
        .top-shell {
            display: flex;
            justify-content: space-between;
            gap: 18px;
            align-items: flex-start;
            margin-bottom: 0.85rem;
        }
        .operator-header {
            display: grid;
            grid-template-columns: minmax(320px, 1.7fr) repeat(4, minmax(150px, 0.9fr));
            gap: 10px;
            margin: 0.35rem 0 0.9rem;
            align-items: stretch;
        }
        .operator-header-main,
        .operator-header-card {
            background: linear-gradient(180deg, rgba(15, 22, 34, 0.98), rgba(10, 15, 24, 0.98));
            border: 1px solid rgba(129, 151, 181, 0.14);
            border-radius: 16px;
            padding: 14px 16px;
            min-width: 0;
            box-shadow: 0 10px 24px rgba(5, 10, 20, 0.18);
        }
        .operator-header-card.warn,
        .operator-header-main.warn {
            border-color: rgba(250, 204, 21, 0.26);
        }
        .operator-header-card.err,
        .operator-header-main.err {
            border-color: rgba(248, 113, 113, 0.28);
        }
        .operator-header-main {
            display: flex;
            flex-direction: column;
            gap: 12px;
        }
        .operator-header-top {
            display: flex;
            justify-content: space-between;
            align-items: flex-start;
            gap: 14px;
        }
        .operator-title-wrap { min-width: 0; }
        .operator-kicker {
            font-size: 0.72rem;
            text-transform: uppercase;
            letter-spacing: 0.16em;
            color: #7f95ae;
            font-weight: 700;
            margin-bottom: 0.35rem;
        }
        .operator-title {
            font-size: 1.45rem;
            line-height: 1.15;
            font-weight: 700;
            color: #f5f8fc;
            letter-spacing: -0.02em;
        }
        .operator-subtitle {
            margin-top: 0.35rem;
            font-size: 0.88rem;
            color: #92a6bc;
            line-height: 1.4;
        }
        .operator-chip-row {
            display: flex;
            flex-wrap: wrap;
            gap: 8px;
            align-items: center;
        }
        .operator-chip {
            display: inline-flex;
            align-items: center;
            gap: 6px;
            border-radius: 999px;
            padding: 0.28rem 0.62rem;
            font-size: 0.74rem;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            font-weight: 700;
            font-family: 'DM Mono', monospace;
            color: #dce6f1;
            background: rgba(255,255,255,0.04);
            border: 1px solid rgba(129, 151, 181, 0.16);
            white-space: nowrap;
        }
        .operator-chip.good {
            color: #9bf0cf;
            border-color: rgba(76, 226, 197, 0.28);
            background: rgba(76, 226, 197, 0.07);
        }
        .operator-chip.warn {
            color: #fbd46e;
            border-color: rgba(250, 204, 21, 0.28);
            background: rgba(250, 204, 21, 0.08);
        }
        .operator-chip.err {
            color: #fca5a5;
            border-color: rgba(248, 113, 113, 0.3);
            background: rgba(248, 113, 113, 0.08);
        }
        .operator-main-grid {
            display: grid;
            grid-template-columns: repeat(4, minmax(0, 1fr));
            gap: 10px;
        }
        .operator-mini {
            border: 1px solid rgba(129, 151, 181, 0.12);
            border-radius: 12px;
            background: rgba(255,255,255,0.025);
            padding: 11px 12px;
            min-width: 0;
        }
        .operator-mini-label,
        .operator-card-label {
            font-size: 0.7rem;
            text-transform: uppercase;
            letter-spacing: 0.13em;
            color: #71859c;
            margin-bottom: 0.28rem;
            font-weight: 700;
        }
        .operator-mini-value,
        .operator-card-value {
            font-size: 1rem;
            line-height: 1.25;
            color: #f3f7fc;
            font-weight: 700;
            font-family: 'DM Mono', monospace;
            overflow-wrap: anywhere;
        }
        .operator-mini-note,
        .operator-card-note {
            margin-top: 0.32rem;
            font-size: 0.8rem;
            line-height: 1.35;
            color: #8ea3b8;
        }
        .operator-card-value.good { color: #86efac; }
        .operator-card-value.warn { color: #fcd34d; }
        .operator-card-value.err { color: #fda4af; }
        .operator-shell {
            margin: 0.2rem 0 0.75rem;
        }
        .operator-control-shell {
            display: flex;
            justify-content: space-between;
            align-items: flex-start;
            gap: 12px;
            padding: 10px 0 8px;
            border-bottom: 1px solid var(--line);
            margin-bottom: 10px;
        }
        .operator-control-main {
            min-width: 0;
        }
        .operator-control-title {
            margin-bottom: 6px;
        }
        .operator-meta-grid {
            display: grid;
            grid-template-columns: repeat(3, minmax(0, 1fr));
            gap: 8px;
            margin-bottom: 10px;
        }
        .operator-meta-item {
            min-width: 0;
        }
        .operator-status-grid {
            display: grid;
            grid-template-columns: repeat(5, minmax(0, 1fr));
            gap: 8px;
        }
        .operator-status-card {
            background: var(--panel);
            border: 1px solid var(--line);
            border-radius: 8px;
            padding: 12px;
            min-width: 0;
            display: flex;
            flex-direction: column;
            gap: 6px;
        }
        .operator-status-card.warn {
            border-color: rgba(210, 153, 34, 0.35);
        }
        .operator-status-card.err {
            border-color: rgba(248, 81, 73, 0.35);
        }
        .decision-log-summary {
            display: grid;
            grid-template-columns: repeat(3, minmax(0, 1fr));
            gap: 8px;
            margin: 0.25rem 0 0.8rem;
        }
        .decision-log-summary-card {
            background:
                radial-gradient(circle at top right, rgba(103, 184, 255, 0.08), transparent 34%),
                linear-gradient(180deg, rgba(21, 27, 35, 0.98), rgba(15, 21, 29, 0.98));
            border: 1px solid rgba(129, 151, 181, 0.16);
            border-radius: 16px;
            padding: 16px 18px;
            box-shadow: 0 10px 24px rgba(2, 8, 18, 0.2);
        }
        .decision-log-shell {
            display: grid;
            grid-template-columns: minmax(0, 1.25fr) minmax(360px, 0.95fr);
            gap: 16px;
            align-items: start;
        }
        .decision-log-pane {
            background:
                radial-gradient(circle at top right, rgba(103, 184, 255, 0.08), transparent 34%),
                linear-gradient(180deg, rgba(21, 27, 35, 0.98), rgba(15, 21, 29, 0.98));
            border: 1px solid rgba(129, 151, 181, 0.16);
            border-radius: 18px;
            padding: 16px 18px;
            min-width: 0;
            box-shadow: 0 12px 30px rgba(2, 8, 18, 0.2);
        }
        .decision-log-list-header,
        .decision-log-list-row {
            display: grid;
            grid-template-columns: 1.15fr 0.7fr 0.95fr 0.85fr 1.6fr 72px;
            gap: 12px;
            align-items: stretch;
        }
        .decision-log-list-header {
            padding: 0 4px 10px;
            border-bottom: 1px solid rgba(129, 151, 181, 0.14);
            margin-bottom: 8px;
        }
        .decision-log-list-row {
            padding: 12px 14px;
            border: 1px solid rgba(129, 151, 181, 0.14);
            border-radius: 16px;
            background: linear-gradient(180deg, rgba(255,255,255,0.03), rgba(255,255,255,0.015));
            margin-bottom: 10px;
            box-shadow: inset 0 1px 0 rgba(255,255,255,0.02);
        }
        .decision-log-list-row:last-child {
            margin-bottom: 0;
        }
        .decision-log-cell {
            min-width: 0;
        }
        .decision-log-cell-block {
            min-width: 0;
            display: flex;
            flex-direction: column;
            gap: 4px;
            justify-content: center;
        }
        .decision-log-col-label {
            font-size: 0.68rem;
            text-transform: uppercase;
            letter-spacing: 0.14em;
            color: #7f95ae;
            font-weight: 700;
        }
        .decision-log-row-value {
            font-size: 0.95rem;
            color: #e7eef8;
            line-height: 1.35;
            overflow-wrap: anywhere;
        }
        .decision-log-row-value.strong {
            font-size: 1.02rem;
            font-weight: 700;
            color: #f5f8fd;
        }
        .decision-log-row-meta {
            font-size: 0.8rem;
            color: #94a8bf;
            line-height: 1.35;
        }
        .decision-log-outcome {
            display: inline-flex;
            align-items: center;
            justify-content: center;
            width: fit-content;
            padding: 0.24rem 0.58rem;
            border-radius: 999px;
            font-size: 0.76rem;
            font-weight: 700;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            font-family: 'DM Mono', monospace;
            border: 1px solid rgba(129, 151, 181, 0.16);
            background: rgba(255,255,255,0.04);
            color: #dce6f1;
        }
        .decision-log-outcome.good {
            color: #9bf0cf;
            background: rgba(76, 226, 197, 0.08);
            border-color: rgba(76, 226, 197, 0.24);
        }
        .decision-log-outcome.warn {
            color: #fbd46e;
            background: rgba(250, 204, 21, 0.08);
            border-color: rgba(250, 204, 21, 0.24);
        }
        .decision-log-outcome.neutral {
            color: #d9e3ee;
            background: rgba(148, 163, 184, 0.08);
            border-color: rgba(129, 151, 181, 0.2);
        }
        .decision-log-button-cell {
            display: flex;
            align-items: center;
            justify-content: stretch;
            min-width: 0;
        }
        .decision-log-detail-hero {
            display: flex;
            flex-direction: column;
            gap: 10px;
            padding-bottom: 14px;
            margin-bottom: 14px;
            border-bottom: 1px solid rgba(129, 151, 181, 0.14);
        }
        .decision-log-detail-headline {
            font-size: 1.18rem;
            line-height: 1.3;
            font-weight: 700;
            color: #f5f8fd;
        }
        .decision-log-detail-chip-row {
            display: flex;
            flex-wrap: wrap;
            gap: 8px;
            align-items: center;
        }
        .decision-log-detail-chip {
            display: inline-flex;
            align-items: center;
            gap: 6px;
            padding: 0.24rem 0.62rem;
            border-radius: 999px;
            font-size: 0.75rem;
            font-weight: 700;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            font-family: 'DM Mono', monospace;
            color: #dce6f1;
            background: rgba(255,255,255,0.04);
            border: 1px solid rgba(129, 151, 181, 0.16);
        }
        .decision-log-detail-grid {
            display: grid;
            grid-template-columns: repeat(2, minmax(0, 1fr));
            gap: 10px;
        }
        .decision-log-detail-grid .kv {
            display: flex;
            flex-direction: column;
            align-items: stretch;
            gap: 6px;
            padding: 12px 14px;
            border-radius: 14px;
            border: 1px solid rgba(129, 151, 181, 0.14);
            background: linear-gradient(180deg, rgba(255,255,255,0.03), rgba(255,255,255,0.015));
        }
        .decision-log-detail-grid .kv-key {
            font-size: 0.7rem;
            text-transform: uppercase;
            letter-spacing: 0.14em;
            color: #7f95ae;
            font-weight: 700;
        }
        .decision-log-detail-grid .kv-val {
            font-size: 0.98rem;
            line-height: 1.45;
            color: #edf4ff;
            font-family: var(--font-family-ui);
            font-weight: 600;
            overflow-wrap: anywhere;
        }
        .decision-log-empty {
            padding: 18px 0 8px;
        }
        @media (max-width: 1200px) {
            .operator-meta-grid { grid-template-columns: 1fr; }
            .operator-status-grid { grid-template-columns: repeat(3, minmax(0, 1fr)); }
            .decision-log-shell { grid-template-columns: 1fr; }
        }
        @media (max-width: 900px) {
            .operator-control-shell { flex-direction: column; }
            .operator-status-grid,
            .decision-log-summary { grid-template-columns: 1fr; }
            .decision-log-list-header,
            .decision-log-list-row {
                grid-template-columns: 1fr;
                gap: 6px;
            }
        }
        .hero-card {
            background:
                radial-gradient(circle at top right, rgba(103, 184, 255, 0.12), transparent 22%),
                linear-gradient(135deg, rgba(15, 24, 38, 0.98), rgba(10, 15, 25, 0.98)),
                linear-gradient(180deg, rgba(255,255,255,0.02), rgba(255,255,255,0));
            border: 1px solid rgba(129, 151, 181, 0.14);
            border-radius: 24px;
            padding: 22px 24px;
            box-shadow: var(--glow);
            min-height: 118px;
            position: relative;
            overflow: hidden;
        }
        .hero-card:before {
            content: "";
            position: absolute;
            inset: 0;
            background: linear-gradient(90deg, rgba(103,184,255,0.10), transparent 30%, transparent 70%, rgba(76,226,197,0.08));
            pointer-events: none;
        }
        .hero-kicker {
            font-size: 0.76rem;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.18em;
            color: #7f95ae;
            margin-bottom: 0.45rem;
        }
        .top-title {
            font-size: 2.25rem;
            font-weight: 700;
            color: #f8fbff;
            letter-spacing: -0.02em;
        }
        .top-meta {
            display: flex;
            align-items: center;
            gap: 8px;
            flex-wrap: wrap;
            margin-top: 0.3rem;
        }
        .subtle-copy {
            font-size: 1rem;
            color: #7c90a6;
        }
        .hero-summary {
            display: grid;
            grid-template-columns: repeat(3, minmax(0, 1fr));
            gap: 10px;
            margin-top: 0.95rem;
        }
        .hero-stat {
            border: 1px solid rgba(129, 151, 181, 0.12);
            background: linear-gradient(180deg, rgba(255,255,255,0.04), rgba(255,255,255,0.02));
            border-radius: 16px;
            padding: 12px 14px;
            backdrop-filter: blur(10px);
        }
        .hero-stat-label {
            font-size: 0.72rem;
            text-transform: uppercase;
            letter-spacing: 0.13em;
            color: #7489a2;
            margin-bottom: 0.25rem;
        }
        .hero-stat-value {
            font-size: 1.12rem;
            font-weight: 700;
            color: #f3f7fc;
            font-family: 'DM Mono', monospace;
        }
        .hero-note {
            margin-top: 0.75rem;
            font-size: 0.9rem;
            color: #9bb0c7;
            max-width: 54rem;
            line-height: 1.45;
        }
        .status-inline {
            display: flex;
            align-items: center;
            gap: 8px;
            flex-wrap: wrap;
            justify-content: flex-end;
        }
        .action-cluster {
            display: flex;
            flex-direction: column;
            align-items: flex-end;
            gap: 10px;
        }
        .action-card {
            background: linear-gradient(180deg, rgba(18, 26, 38, 0.94), rgba(12, 17, 27, 0.94));
            border: 1px solid rgba(129, 151, 181, 0.14);
            border-radius: 18px;
            padding: 14px 16px;
            min-width: 260px;
            box-shadow: var(--glow);
        }
        .action-card .workspace-label { margin-bottom: 0.35rem; }
        .status-emphasis {
            font-size: 1.12rem;
            color: #eff5fb;
            font-weight: 600;
        }
        .status-strip {
            display: grid;
            grid-template-columns: minmax(250px, 1.45fr) repeat(5, minmax(120px, 0.72fr));
            gap: 10px;
            margin: 0.8rem 0 0.95rem;
        }
        .status-panel {
            background:
                radial-gradient(circle at top right, rgba(103, 184, 255, 0.1), transparent 26%),
                linear-gradient(180deg, rgba(18, 27, 41, 0.97), rgba(10, 16, 25, 0.97));
            border: 1px solid rgba(129, 151, 181, 0.15);
            border-radius: 18px;
            padding: 16px 18px;
            box-shadow: var(--glow);
            min-width: 0;
        }
        .status-panel.primary {
            border-color: rgba(103, 184, 255, 0.24);
        }
        .status-panel.warn {
            border-color: rgba(250, 204, 21, 0.3);
            background:
                radial-gradient(circle at top right, rgba(250, 204, 21, 0.12), transparent 22%),
                linear-gradient(180deg, rgba(31, 27, 16, 0.97), rgba(19, 17, 12, 0.97));
        }
        .status-panel.err {
            border-color: rgba(248, 113, 113, 0.3);
            background:
                radial-gradient(circle at top right, rgba(248, 113, 113, 0.14), transparent 22%),
                linear-gradient(180deg, rgba(34, 21, 24, 0.97), rgba(21, 12, 14, 0.97));
        }
        .status-panel-kicker {
            font-size: 0.74rem;
            text-transform: uppercase;
            letter-spacing: 0.16em;
            color: #7f95ae;
            margin-bottom: 0.35rem;
            font-weight: 700;
        }
        .status-panel-title {
            font-size: 1.38rem;
            color: #f4f8fd;
            font-weight: 700;
            margin-bottom: 0.28rem;
        }
        .status-panel-copy {
            font-size: 0.92rem;
            color: #a7bbd0;
            line-height: 1.4;
        }
        .status-metric-label {
            font-size: 0.72rem;
            text-transform: uppercase;
            letter-spacing: 0.12em;
            color: #71859c;
            margin-bottom: 0.25rem;
        }
        .status-metric-value {
            font-size: 1.24rem;
            color: #f2f7fd;
            font-weight: 700;
            font-family: 'DM Mono', monospace;
        }
        .status-metric-note {
            margin-top: 0.35rem;
            font-size: 0.82rem;
            color: #91a6bc;
            line-height: 1.35;
        }
        .approval-chip-row {
            display: flex;
            gap: 8px;
            align-items: center;
            flex-wrap: wrap;
            margin-top: 0.55rem;
        }
        .approval-chip {
            display: inline-flex;
            align-items: center;
            gap: 6px;
            border-radius: 999px;
            padding: 0.26rem 0.66rem;
            font-size: 0.76rem;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            font-weight: 700;
            font-family: 'DM Mono', monospace;
            border: 1px solid rgba(129, 151, 181, 0.16);
            color: #dce6f1;
            background: rgba(255,255,255,0.04);
        }
        .approval-chip.good {
            color: #9bf0cf;
            border-color: rgba(76, 226, 197, 0.28);
            background: rgba(76, 226, 197, 0.08);
        }
        .approval-chip.warn {
            color: #fbd46e;
            border-color: rgba(250, 204, 21, 0.28);
            background: rgba(250, 204, 21, 0.08);
        }
        .approval-chip.err {
            color: #fca5a5;
            border-color: rgba(248, 113, 113, 0.3);
            background: rgba(248, 113, 113, 0.08);
        }
        .compact-card {
            background: linear-gradient(180deg, rgba(23, 30, 42, 0.96), rgba(15, 18, 27, 0.96));
            border: 1px solid rgba(129, 151, 181, 0.14);
            border-radius: 16px;
            padding: 16px 18px;
            box-shadow: 0 14px 28px rgba(0, 0, 0, 0.16);
        }
        .compact-card.feed-card {
            background:
                radial-gradient(circle at left center, rgba(76, 226, 197, 0.09), transparent 18%),
                linear-gradient(135deg, rgba(18, 29, 42, 0.98), rgba(11, 17, 27, 0.98));
            border-color: rgba(96, 165, 250, 0.14);
            border-radius: 18px;
            padding: 14px 16px;
            margin-bottom: 0.35rem;
            box-shadow: var(--glow);
        }
        .feed-grid {
            display: grid;
            grid-template-columns: minmax(220px, 1.5fr) repeat(3, minmax(120px, 0.72fr));
            gap: 10px;
            align-items: center;
        }
        .feed-primary { min-width: 0; }
        .feed-title {
            display: flex;
            align-items: center;
            gap: 8px;
            font-size: 1.04rem;
            color: #eef4fb;
            font-weight: 600;
            margin-bottom: 0.2rem;
        }
        .feed-subline {
            font-size: 0.92rem;
            color: #9cb0c8;
            line-height: 1.32;
            max-width: 28rem;
        }
        .feed-stat {
            min-width: 0;
            padding-left: 12px;
            border-left: 1px solid rgba(255,255,255,0.08);
        }
        .feed-stat-label {
            font-size: 0.72rem;
            text-transform: uppercase;
            letter-spacing: 0.12em;
            color: #71859c;
            margin-bottom: 0.3rem;
        }
        .feed-stat-value {
            font-size: 1.1rem;
            color: #f1f6fc;
            font-weight: 700;
            font-family: 'DM Mono', monospace;
        }
        .status-card {
            display: flex;
            gap: 10px;
            align-items: flex-start;
            background: rgba(255,255,255,0.025);
            border: 1px solid rgba(129, 151, 181, 0.14);
            border-radius: 12px;
            padding: 10px 12px;
            margin-bottom: 0.55rem;
        }
        .status-card:last-child { margin-bottom: 0; }
        .status-card.warn {
            background: rgba(250, 204, 21, 0.08);
            border-color: rgba(250, 204, 21, 0.24);
        }
        .status-card.err {
            background: rgba(248, 113, 113, 0.08);
            border-color: rgba(248, 113, 113, 0.24);
        }
        .status-card.info {
            background: rgba(96, 165, 250, 0.08);
            border-color: rgba(96, 165, 250, 0.2);
        }
        .status-card-dot {
            width: 8px;
            height: 8px;
            border-radius: 50%;
            margin-top: 5px;
            flex-shrink: 0;
        }
        .status-card.warn .status-card-dot { background: #facc15; }
        .status-card.err .status-card-dot { background: #f87171; }
        .status-card.info .status-card-dot { background: #60a5fa; }
        .status-card-title {
            font-size: 0.92rem;
            color: #e8eef7;
            font-weight: 600;
            margin-bottom: 3px;
        }
        .status-card-body {
            font-size: 0.86rem;
            color: #a9bacd;
            line-height: 1.35;
        }
        .section-intro {
            margin: 0.4rem 0 1rem;
            padding: 16px 18px;
            border-radius: 20px;
            background:
                radial-gradient(circle at top right, rgba(103, 184, 255, 0.1), transparent 28%),
                linear-gradient(180deg, rgba(17, 26, 40, 0.96), rgba(11, 17, 28, 0.96));
            border: 1px solid rgba(129, 151, 181, 0.12);
            box-shadow: var(--glow);
        }
        .section-intro-kicker {
            font-size: 0.74rem;
            text-transform: uppercase;
            letter-spacing: 0.18em;
            color: #7f95ae;
            margin-bottom: 0.35rem;
            font-weight: 700;
        }
        .section-intro-title {
            font-size: 1.34rem;
            color: #f4f8fd;
            font-weight: 700;
            margin-bottom: 0.35rem;
        }
        .section-intro-body {
            font-size: 0.95rem;
            color: #9db0c8;
            line-height: 1.5;
            max-width: 50rem;
        }
        .workspace-map {
            display: grid;
            grid-template-columns: repeat(4, minmax(0, 1fr));
            gap: 12px;
            margin: 0.55rem 0 1.1rem;
        }
        .workspace-map-card {
            padding: 16px 16px 15px;
            border-radius: 18px;
            background: linear-gradient(180deg, rgba(18, 26, 39, 0.96), rgba(11, 17, 27, 0.96));
            border: 1px solid rgba(129, 151, 181, 0.12);
            box-shadow: 0 14px 30px rgba(2, 6, 14, 0.24);
        }
        .workspace-map-kicker {
            font-size: 0.72rem;
            text-transform: uppercase;
            letter-spacing: 0.14em;
            color: #7b91aa;
            margin-bottom: 0.42rem;
        }
        .workspace-map-title {
            font-size: 1rem;
            color: #eff5fb;
            font-weight: 700;
            margin-bottom: 0.35rem;
        }
        .workspace-map-copy {
            font-size: 0.88rem;
            color: #97abc1;
            line-height: 1.45;
        }
        .workspace-map-meta {
            margin-top: 0.75rem;
            font-family: 'DM Mono', monospace;
            font-size: 0.76rem;
            color: #cde5ff;
            letter-spacing: 0.05em;
        }
        .workspace-label {
            font-size: 0.82rem;
            text-transform: uppercase;
            letter-spacing: 0.16em;
            color: #768aa1;
            margin-bottom: 0.6rem;
            font-weight: 600;
        }
        .watchlist-shell {
            background: linear-gradient(180deg, rgba(18, 26, 39, 0.98), rgba(11, 17, 27, 0.97));
            border: 1px solid rgba(129, 151, 181, 0.12);
            border-radius: 22px;
            overflow: hidden;
            box-shadow: var(--glow);
        }
        .watchlist-header {
            padding: 16px 16px 12px;
            border-bottom: 1px solid rgba(255,255,255,0.06);
            background: linear-gradient(180deg, rgba(255,255,255,0.03), rgba(255,255,255,0.015));
        }
        .watchlist-body {
            max-height: 980px;
            overflow-y: auto;
        }
        .watchlist-subtext {
            font-size: 1rem;
            color: #8b9fb6;
            margin-top: 0.2rem;
        }
        .watchlist-tip {
            margin-top: 0.45rem;
            font-size: 0.82rem;
            color: #72849a;
            line-height: 1.35;
        }
        .live-zones {
            display: grid;
            grid-template-columns: 1.05fr 1.7fr 1.05fr;
            gap: 12px;
            margin-bottom: 0.85rem;
        }
        .live-zone {
            padding: 12px 14px;
            border-radius: 16px;
            background: rgba(255,255,255,0.028);
            border: 1px solid rgba(129, 151, 181, 0.1);
        }
        .live-zone-title {
            font-size: 0.8rem;
            text-transform: uppercase;
            letter-spacing: 0.14em;
            color: #7d92aa;
            margin-bottom: 0.32rem;
            font-weight: 700;
        }
        .live-zone-copy {
            font-size: 0.87rem;
            color: #9fb3c8;
            line-height: 1.45;
        }
        .history-shell {
            margin-top: 0.2rem;
        }
        .history-toolbar {
            display: grid;
            grid-template-columns: minmax(260px, 1.2fr) repeat(2, minmax(140px, 0.8fr));
            gap: 12px;
            margin-bottom: 1rem;
        }
        .history-stat {
            padding: 14px 16px;
            border-radius: 16px;
            background: rgba(255,255,255,0.028);
            border: 1px solid rgba(129, 151, 181, 0.12);
        }
        .history-stat-label {
            font-size: 0.74rem;
            text-transform: uppercase;
            letter-spacing: 0.14em;
            color: #7d92aa;
            margin-bottom: 0.36rem;
        }
        .history-stat-value {
            font-size: 1.08rem;
            font-family: 'DM Mono', monospace;
            color: #eff5fb;
            font-weight: 700;
        }
        .sym-card-v2 {
            padding: 16px 17px 15px;
            border-bottom: 1px solid rgba(255,255,255,0.045);
            cursor: pointer;
            border-left: 3px solid transparent;
            background: linear-gradient(180deg, rgba(255,255,255,0.018), rgba(255,255,255,0.01));
            transition: background 0.14s ease, border-color 0.14s ease, transform 0.14s ease, box-shadow 0.14s ease;
        }
        .sym-card-v2:last-child { border-bottom: none; }
        .sym-card-v2:hover {
            background: linear-gradient(180deg, rgba(255,255,255,0.04), rgba(255,255,255,0.018));
            transform: translateX(3px);
            box-shadow: inset 0 0 0 1px rgba(140, 167, 201, 0.08);
        }
        .sym-card-v2.sym-active {
            background:
                radial-gradient(circle at right center, rgba(96,165,250,0.12), transparent 38%),
                linear-gradient(90deg, rgba(59,130,246,0.18), rgba(59,130,246,0.045));
            border-left-color: #60a5fa;
            box-shadow: inset 0 0 0 1px rgba(96,165,250,0.12);
        }
        .sym-card-v2.sym-error {
            background:
                radial-gradient(circle at right center, rgba(248,113,113,0.1), transparent 40%),
                linear-gradient(90deg, rgba(248,113,113,0.16), rgba(248,113,113,0.045));
            border-left-color: #f87171;
        }
        .sym-card-v2-top {
            display: grid;
            grid-template-columns: 1fr auto auto;
            gap: 8px;
            align-items: center;
        }
        .sym-card-v2-meta {
            display: grid;
            grid-template-columns: 1fr auto;
            gap: 8px;
            margin-top: 9px;
            font-size: 0.88rem;
            color: #c6d4e2;
        }
        .sym-price {
            font-family: 'DM Mono', monospace;
            color: #e8eef7;
            font-size: 1.02rem;
        }
        .sym-change {
            font-family: 'DM Mono', monospace;
            color: #93a5bc;
            font-size: 0.98rem;
        }
        .sym-card-foot {
            display: flex;
            justify-content: space-between;
            align-items: center;
            gap: 8px;
            margin-top: 10px;
        }
        .tiny-label {
            font-size: 0.84rem;
            color: #6f8095;
            text-transform: uppercase;
            letter-spacing: 0.09em;
        }
        .detail-hero {
            background:
                radial-gradient(circle at top right, rgba(103,184,255,0.10), transparent 18%),
                linear-gradient(180deg, rgba(20, 29, 42, 0.98), rgba(11, 17, 27, 0.96));
            border: 1px solid rgba(129, 151, 181, 0.13);
            border-radius: 22px;
            padding: 24px 26px;
            box-shadow: var(--glow);
            margin-bottom: 0.9rem;
        }
        .detail-hero-top {
            display: flex;
            justify-content: space-between;
            align-items: flex-start;
            gap: 16px;
            margin-bottom: 1rem;
        }
        .detail-symbol {
            font-size: 2.9rem;
            line-height: 1;
            font-weight: 700;
            color: #fbfdff;
            font-family: 'DM Mono', monospace;
        }
        .detail-price {
            font-size: 2.45rem;
            line-height: 1;
            font-weight: 700;
            color: #f8fbff;
            font-family: 'DM Mono', monospace;
            text-align: right;
        }
        .detail-subline {
            display: flex;
            flex-wrap: wrap;
            align-items: center;
            gap: 8px;
            margin-top: 0.45rem;
            color: #8093a8;
            font-size: 0.92rem;
        }
        .detail-context {
            margin-top: 0.8rem;
            padding: 10px 12px;
            border-radius: 12px;
            background: rgba(255,255,255,0.03);
            border: 1px solid rgba(255,255,255,0.05);
            font-size: 0.9rem;
            color: #adc0d3;
            line-height: 1.4;
        }
        .detail-pill-row {
            display: flex;
            flex-wrap: wrap;
            gap: 8px;
            margin-top: 0.85rem;
        }
        .detail-info-grid {
            display: grid;
            grid-template-columns: repeat(3, minmax(0, 1fr));
            gap: 10px;
            margin-bottom: 14px;
        }
        .detail-info-cell {
            background: rgba(255,255,255,0.03);
            border: 1px solid rgba(255,255,255,0.06);
            border-radius: 12px;
            padding: 12px 13px;
        }
        .trend-strip {
            display: flex;
            align-items: flex-end;
            gap: 4px;
            height: 58px;
            margin: 0.75rem 0 0.15rem;
            padding: 10px 0 2px;
        }
        .trend-bar {
            flex: 1 1 0;
            min-width: 6px;
            border-radius: 999px;
            opacity: 0.95;
        }
        .detail-history-card,
        .rail-card {
            background: linear-gradient(180deg, rgba(18, 26, 39, 0.98), rgba(11, 17, 27, 0.96));
            border: 1px solid rgba(129, 151, 181, 0.12);
            border-radius: 20px;
            padding: 18px 18px 16px;
            margin-bottom: 0.95rem;
            box-shadow: var(--glow);
        }
        .rail-card:last-child { margin-bottom: 0; }
        .rail-card .sec-head,
        .detail-history-card .sec-head {
            margin-top: 0;
            margin-bottom: 0.85rem;
        }

        /* Trader Thoughts + Near Misses panels */
        .thought-row {
            padding: 9px 0;
            border-bottom: 0.5px solid rgba(255,255,255,0.05);
        }
        .thought-row:last-child { border-bottom: none; padding-bottom: 0; }
        .thought-meta {
            display: flex;
            align-items: center;
            gap: 7px;
            margin-bottom: 3px;
        }
        .thought-sym {
            font-family: 'DM Mono', monospace;
            font-size: 0.78rem;
            font-weight: 700;
            color: #7db6f7;
            letter-spacing: 0.04em;
        }
        /* Near-miss symbol chip â€” amber tint to signal "almost" */
        .miss-sym {
            font-family: 'DM Mono', monospace;
            font-size: 0.78rem;
            font-weight: 700;
            color: #f0c96a;
            letter-spacing: 0.04em;
        }
        .thought-ts {
            font-family: 'DM Mono', monospace;
            font-size: 0.74rem;
            color: #4a5c6e;
        }
        .thought-text {
            font-size: 0.9rem;
            color: #b8cfe8;
            line-height: 1.4;
        }

        .ticker-strip {
            display: flex;
            flex-direction: column;
            gap: 12px;
            margin-bottom: 1rem;
        }
        .ticker-strip-head {
            display: flex;
            justify-content: space-between;
            align-items: flex-end;
            gap: 14px;
            flex-wrap: wrap;
        }
        .ticker-strip-ribbon {
            display: flex;
            gap: 12px;
            overflow-x: auto;
            padding: 14px 16px;
            border-radius: 20px;
            background:
                radial-gradient(circle at left center, rgba(96,165,250,0.08), transparent 18%),
                linear-gradient(90deg, rgba(16, 24, 36, 0.98), rgba(10, 17, 27, 0.98));
            border: 1px solid rgba(129, 151, 181, 0.14);
            box-shadow: inset 0 1px 0 rgba(255,255,255,0.02);
            scrollbar-width: thin;
        }
        .ticker-chip {
            flex: 0 0 auto;
            min-width: 124px;
            display: grid;
            grid-template-columns: auto 1fr;
            grid-template-areas:
                "symbol price"
                "change change";
            align-items: center;
            column-gap: 10px;
            row-gap: 6px;
            padding: 10px 12px 11px;
            border-radius: 14px;
            background: linear-gradient(180deg, rgba(255,255,255,0.055), rgba(255,255,255,0.025));
            border: 1px solid rgba(255,255,255,0.075);
            box-shadow: inset 0 1px 0 rgba(255,255,255,0.025);
        }
        .ticker-chip.active {
            background:
                radial-gradient(circle at top right, rgba(96,165,250,0.12), transparent 34%),
                linear-gradient(180deg, rgba(96,165,250,0.14), rgba(96,165,250,0.06));
            border-color: rgba(96,165,250,0.28);
        }
        .ticker-chip-symbol {
            grid-area: symbol;
            font-family: 'DM Mono', monospace;
            font-size: 1.08rem;
            font-weight: 700;
            color: #f8fbff;
        }
        .ticker-chip-price {
            grid-area: price;
            font-family: 'DM Mono', monospace;
            justify-self: end;
            font-size: 0.98rem;
            color: #c8d5e4;
        }
        .ticker-chip-change {
            grid-area: change;
            font-family: 'DM Mono', monospace;
            display: inline-flex;
            align-items: baseline;
            gap: 6px;
            font-size: 1rem;
            font-weight: 700;
            letter-spacing: -0.01em;
        }
        .ticker-chip-note {
            font-size: 0.78rem;
            color: #6f849d;
            font-weight: 600;
            letter-spacing: 0.08em;
            text-transform: uppercase;
        }
        .exec-panel {
            background:
                radial-gradient(circle at right top, rgba(76, 226, 197, 0.08), transparent 18%),
                linear-gradient(180deg, rgba(21, 30, 44, 0.98), rgba(12, 18, 28, 0.98));
            border: 1px solid rgba(96, 165, 250, 0.18);
            border-radius: 20px;
            padding: 20px 21px;
            margin-bottom: 1rem;
            box-shadow: var(--glow);
        }
        .exec-list {
            display: grid;
            gap: 11px;
        }
        .activity-feed-list {
            display: grid;
            gap: 10px;
        }
        .activity-feed-row {
            display: grid;
            grid-template-columns: 76px minmax(0, 1fr);
            gap: 12px;
            align-items: start;
            padding: 12px 0;
            border-top: 1px solid rgba(255,255,255,0.06);
        }
        .activity-feed-row:first-child {
            border-top: none;
            padding-top: 4px;
        }
        .activity-time {
            font-size: 0.74rem;
            color: #7f95ae;
            font-family: 'DM Mono', monospace;
            letter-spacing: 0.06em;
            padding-top: 2px;
        }
        .activity-title {
            font-size: 0.97rem;
            color: #eef4fb;
            font-weight: 600;
            margin-bottom: 0.2rem;
        }
        .activity-body {
            font-size: 0.88rem;
            color: #9fb2c7;
            line-height: 1.4;
        }
        .activity-badges {
            display: flex;
            gap: 8px;
            flex-wrap: wrap;
            margin-bottom: 0.35rem;
        }
        .activity-tag {
            display: inline-block;
            border-radius: 999px;
            padding: 0.18rem 0.5rem;
            font-size: 0.7rem;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            font-weight: 700;
            font-family: 'DM Mono', monospace;
            background: rgba(255,255,255,0.05);
            color: #dce6f1;
        }
        .activity-tag.warn {
            background: rgba(250, 204, 21, 0.08);
            color: #fbd46e;
        }
        .activity-tag.err {
            background: rgba(248, 113, 113, 0.08);
            color: #fca5a5;
        }
        .activity-tag.info {
            background: rgba(103, 184, 255, 0.08);
            color: #9fd4ff;
        }
        .exec-row {
            display: grid;
            grid-template-columns: 58px minmax(220px, 1.3fr) minmax(120px, 0.8fr) minmax(180px, 1fr);
            gap: 12px;
            align-items: center;
            padding: 14px 15px 14px 17px;
            border: 1px solid rgba(255,255,255,0.06);
            border-radius: 12px;
            background: #171d28;
            position: relative;
        }
        .exec-row:before {
            content: "";
            position: absolute;
            left: 0;
            top: 0;
            bottom: 0;
            width: 4px;
            border-radius: 12px 0 0 12px;
            background: #64748b;
        }
        .exec-row.buy:before { background: #4ade80; }
        .exec-row.sell:before { background: #f87171; }
        .exec-row.hold-no-signal:before { background: #64748b; }
        .exec-row.hold-filtered:before { background: #facc15; }
        .exec-row.hold-rejected:before { background: #fb923c; }
        .exec-row.blocked:before { background: #f97316; }
        .exec-time, .exec-qty {
            font-family: 'DM Mono', monospace;
            font-size: 0.82rem;
            color: #8b9eb4;
        }
        .exec-main {
            display: flex;
            align-items: center;
            gap: 10px;
            min-width: 0;
        }
        .exec-action-wrap {
            display: flex;
            align-items: center;
            gap: 10px;
            min-width: 0;
        }
        .exec-action {
            font-family: 'DM Mono', monospace;
            font-size: 1.08rem;
            font-weight: 800;
            letter-spacing: 0.04em;
        }
        .exec-action.buy { color: #4ade80; }
        .exec-action.sell { color: #f87171; }
        .exec-action.hold { color: #cbd5e1; }
        .exec-action.blocked { color: #fb923c; }
        .exec-fill {
            display: flex;
            align-items: center;
            gap: 8px;
        }
        .exec-symbol {
            font-family: 'DM Mono', monospace;
            font-size: 1.08rem;
            color: #f5f8fd;
            font-weight: 700;
        }
        .exec-price {
            font-family: 'DM Mono', monospace;
            font-size: 1rem;
            color: #e2e8f0;
            font-weight: 600;
        }
        .exec-reason {
            font-size: 0.94rem;
            color: #afbfd1;
            line-height: 1.3;
        }
        .exec-type-badge {
            display: inline-flex;
            align-items: center;
            padding: 0.18rem 0.5rem;
            border-radius: 999px;
            font-size: 0.72rem;
            font-weight: 700;
            letter-spacing: 0.06em;
            text-transform: uppercase;
            font-family: 'DM Mono', monospace;
            background: rgba(255,255,255,0.06);
            color: #cbd5e1;
        }
        .exec-type-badge.filtered { background: rgba(250,204,21,0.12); color: #facc15; }
        .exec-type-badge.rejected { background: rgba(251,146,60,0.12); color: #fb923c; }
        .exec-type-badge.blocked { background: rgba(249,115,22,0.14); color: #fdba74; }
        .exec-type-badge.no-signal { background: rgba(100,116,139,0.16); color: #94a3b8; }
        .logic-list {
            display: grid;
            gap: 10px;
            margin-top: 0.95rem;
        }
        .logic-item {
            display: grid;
            grid-template-columns: minmax(118px, 0.72fr) minmax(0, 1.65fr);
            gap: 12px 16px;
            align-items: start;
            font-size: 0.98rem;
            color: #d8e3ee;
            line-height: 1.42;
            padding: 12px 14px;
            background: linear-gradient(180deg, rgba(255,255,255,0.03), rgba(255,255,255,0.018));
            border: 1px solid rgba(255,255,255,0.055);
            border-radius: 14px;
            box-shadow: inset 0 1px 0 rgba(255,255,255,0.02);
        }
        .logic-bullet {
            display: none;
        }
        .logic-key {
            color: #8ea4bc;
            font-size: 0.76rem;
            line-height: 1.25;
            letter-spacing: 0.12em;
            text-transform: uppercase;
            font-weight: 700;
            padding-top: 0.15rem;
        }
        .logic-body {
            min-width: 0;
        }
        .logic-main {
            color: #eef5fb;
            font-size: 0.98rem;
            line-height: 1.35;
        }
        .logic-note {
            margin-top: 0.32rem;
            color: #8fa2b7;
            font-size: 0.88rem;
            line-height: 1.42;
        }
        .logic-val-yes { color: #4ade80; font-weight: 700; }
        .logic-val-no { color: #f87171; font-weight: 700; }
        .logic-val-neutral { color: #d7e1ec; font-weight: 600; }
        .cycle-summary {
            margin-top: 0.95rem;
            padding: 15px 17px;
            border-radius: 16px;
            background: linear-gradient(90deg, rgba(22,30,42,0.98), rgba(17,22,32,0.98));
            border: 1px solid rgba(96,165,250,0.18);
        }
        .cycle-summary-line {
            font-size: 0.92rem;
            color: #e6edf6;
        }
        .cycle-summary-line + .cycle-summary-line {
            margin-top: 5px;
            color: #b1c0d2;
        }
        .timing-shell {
            margin: 0.5rem 0 0.75rem;
            padding: 16px 18px;
            border-radius: 18px;
            background: linear-gradient(135deg, rgba(18, 27, 41, 0.98), rgba(12, 16, 24, 0.98));
            border: 1px solid rgba(96,165,250,0.14);
            box-shadow: 0 16px 32px rgba(0, 0, 0, 0.18);
        }
        .timing-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            gap: 12px;
            margin-bottom: 0.9rem;
        }
        .timing-title {
            font-size: 0.9rem;
            text-transform: uppercase;
            letter-spacing: 0.14em;
            color: #7f93ab;
            font-weight: 700;
        }
        .timing-grid {
            display: grid;
            grid-template-columns: repeat(4, minmax(0, 1fr));
            gap: 12px;
        }
        .timing-item {
            background: rgba(255,255,255,0.03);
            border: 1px solid rgba(255,255,255,0.06);
            border-radius: 14px;
            padding: 12px 13px;
        }
        .timing-item.strong {
            background: rgba(96,165,250,0.09);
            border-color: rgba(96,165,250,0.24);
        }
        .timing-label {
            font-size: 0.72rem;
            text-transform: uppercase;
            letter-spacing: 0.12em;
            color: #7286a0;
            margin-bottom: 0.35rem;
        }
        .timing-value {
            font-size: 1.02rem;
            font-weight: 700;
            color: #f5f9ff;
            font-family: 'DM Mono', monospace;
        }
        .timing-help {
            font-size: 0.96rem;
            color: #96a8bd;
        }
        [data-testid="stButton"] > button[kind="secondary"] {
            border-radius: 12px;
            border-color: rgba(129, 151, 181, 0.28);
            background: rgba(255,255,255,0.03);
        }
        [data-testid="stTabs"] [role="tablist"] {
            gap: 10px;
            border-bottom: 1px solid rgba(255,255,255,0.07);
            padding-top: 0.05rem;
            padding-bottom: 0.2rem;
            margin-bottom: 0.35rem;
        }
        [data-testid="stTabs"] [role="tab"] {
            border-radius: 999px;
            padding: 0.22rem 0.75rem;
            color: #8ea2b8;
            background: linear-gradient(180deg, rgba(255,255,255,0.03), rgba(255,255,255,0.015));
            border: 1px solid rgba(255,255,255,0.05);
            font-size: 0.88rem;
            transition: all 0.15s ease;
        }
        [data-testid="stTabs"] [aria-selected="true"] {
            color: #f8fbff;
            background: linear-gradient(180deg, rgba(103,184,255,0.18), rgba(103,184,255,0.08));
            border-color: rgba(103,184,255,0.28);
            box-shadow: 0 8px 24px rgba(14, 90, 183, 0.24);
        }
        .rail-row { padding: 12px 0; border-bottom: 1px solid rgba(255,255,255,0.05); }
        .rail-row:last-child { border-bottom: none; }

        @media (max-width: 1200px) {
            .operator-header { grid-template-columns: repeat(3, minmax(0, 1fr)); }
            .operator-header-main { grid-column: span 3; }
            .operator-main-grid { grid-template-columns: repeat(2, minmax(0, 1fr)); }
            .dash-metrics { grid-template-columns: repeat(2, 1fr); }
            .detail-info-grid { grid-template-columns: repeat(2, minmax(0, 1fr)); }
            .status-strip { grid-template-columns: repeat(3, minmax(0, 1fr)); }
            .hero-summary,
            .timing-grid,
            .workspace-map,
            .history-toolbar,
            .live-zones,
            .feed-grid { grid-template-columns: repeat(2, minmax(0, 1fr)); }
        }
        @media (max-width: 900px) {
            .top-shell { flex-direction: column; }
            .status-inline { justify-content: flex-start; }
            .operator-header { grid-template-columns: 1fr; }
            .operator-header-main { grid-column: span 1; }
            .operator-main-grid { grid-template-columns: 1fr; }
            .operator-header-top { flex-direction: column; }
            .detail-hero-top { flex-direction: column; }
            .detail-price { text-align: left; }
            .dash-metrics { grid-template-columns: 1fr; }
            .detail-info-grid { grid-template-columns: 1fr; }
            .status-strip,
            .hero-summary,
            .workspace-map,
            .history-toolbar,
            .live-zones,
            .timing-grid,
            .feed-grid { grid-template-columns: 1fr; }
            .feed-stat {
                padding-left: 0;
                border-left: none;
                border-top: 1px solid rgba(255,255,255,0.08);
                padding-top: 12px;
            }
            .action-cluster { align-items: stretch; }
            .action-card { min-width: 0; }
        }

        /* â”€â”€ Trade Decision Drilldown â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€ */
        .dd-shell {
            padding: 22px 24px 20px;
            border-radius: 22px;
            background:
                radial-gradient(circle at top right, rgba(103, 184, 255, 0.1), transparent 32%),
                linear-gradient(180deg, rgba(17, 25, 38, 0.98), rgba(10, 15, 24, 0.98));
            border: 1px solid rgba(129, 151, 181, 0.18);
            box-shadow: 0 18px 40px rgba(0,0,0,0.24);
            margin-bottom: 1.2rem;
        }
        .dd-header {
            display: flex;
            align-items: center;
            gap: 12px;
            margin-bottom: 1.25rem;
            flex-wrap: wrap;
        }
        .dd-symbol {
            font-size: 1.45rem;
            font-weight: 800;
            color: #f5f8fd;
            font-family: 'DM Mono', monospace;
            letter-spacing: 0.04em;
        }
        .dd-ts {
            font-size: 0.9rem;
            color: #8ea4ba;
            margin-left: 4px;
        }
        .dd-section {
            margin-top: 1.2rem;
            padding-top: 1rem;
            border-top: 1px solid rgba(129, 151, 181, 0.14);
        }
        .dd-shell .kv {
            display: grid;
            grid-template-columns: minmax(120px, 0.7fr) minmax(0, 1fr);
            gap: 10px 18px;
            align-items: start;
            padding: 0.72rem 0;
            border-bottom: 1px solid rgba(129, 151, 181, 0.12);
        }
        .dd-shell .kv-key {
            font-size: 0.74rem;
            text-transform: uppercase;
            letter-spacing: 0.12em;
            color: #7f95ae;
            font-weight: 700;
            padding-top: 0.15rem;
        }
        .dd-shell .kv-val {
            font-size: 0.98rem;
            line-height: 1.45;
            color: #edf4ff;
            font-family: var(--font-family-ui);
            font-weight: 600;
            overflow-wrap: anywhere;
        }
        .dd-path { margin-top: 0.7rem; display: grid; gap: 10px; }
        .dd-path-step {
            display: grid;
            grid-template-columns: 28px minmax(140px, 0.8fr) minmax(0, 1fr);
            gap: 10px;
            align-items: start;
            padding: 12px 14px;
            border: 1px solid rgba(129, 151, 181, 0.12);
            border-radius: 14px;
            background: linear-gradient(180deg, rgba(255,255,255,0.035), rgba(255,255,255,0.02));
        }
        .dd-step-idx {
            font-family: 'DM Mono', monospace;
            font-size: 0.76rem;
            color: #99b2c9;
            width: 28px;
            height: 28px;
            border-radius: 999px;
            display: inline-flex;
            align-items: center;
            justify-content: center;
            background: rgba(103, 184, 255, 0.1);
            border: 1px solid rgba(103, 184, 255, 0.16);
        }
        .dd-step-key {
            font-size: 0.88rem;
            color: #9eb3c8;
            font-weight: 600;
            line-height: 1.4;
            padding-top: 0.3rem;
        }
        .dd-step-val { font-size: 0.96rem; color: #edf4ff; line-height: 1.45; }
        .dd-pass  { color: #4ade80; font-weight: 700; }
        .dd-reject { color: #f87171; font-weight: 700; }
        .dd-note  { display: inline-block; font-size: 0.82rem; color: #8ea4ba; margin-left: 8px; }

        .tradeos-page-title {
            font-family: var(--font-family-ui);
            font-size: var(--font-size-2xl);
            font-weight: var(--font-weight-bold);
            line-height: var(--line-height-tight);
            letter-spacing: var(--letter-spacing-tight);
            color: var(--text-primary);
        }
        .tradeos-section-label {
            font-family: var(--font-family-ui);
            font-size: var(--font-size-xs);
            font-weight: var(--font-weight-semibold);
            line-height: 1.2;
            letter-spacing: var(--letter-spacing-wide);
            text-transform: uppercase;
            color: var(--text-secondary);
        }
        .tradeos-primary-value {
            font-family: var(--font-family-ui);
            font-size: var(--font-size-lg);
            font-weight: var(--font-weight-bold);
            line-height: var(--line-height-snug);
            letter-spacing: -0.01em;
            color: var(--text-primary);
            font-variant-numeric: tabular-nums;
        }
        .tradeos-secondary-value {
            font-family: var(--font-family-ui);
            font-size: var(--font-size-md);
            font-weight: var(--font-weight-medium);
            line-height: var(--line-height-normal);
            color: var(--text-primary);
            font-variant-numeric: tabular-nums;
        }
        .tradeos-body-text {
            font-family: var(--font-family-ui);
            font-size: var(--font-size-sm);
            font-weight: var(--font-weight-regular);
            line-height: var(--line-height-relaxed);
            color: var(--text-secondary);
        }
        .tradeos-meta-text {
            font-family: var(--font-family-ui);
            font-size: var(--font-size-xs);
            font-weight: var(--font-weight-medium);
            line-height: var(--line-height-normal);
            color: var(--text-secondary);
            font-variant-numeric: tabular-nums;
        }
        .tradeos-muted-text {
            font-family: var(--font-family-ui);
            font-size: var(--font-size-xs);
            font-weight: var(--font-weight-regular);
            line-height: var(--line-height-normal);
            color: var(--text-muted);
        }

        /* Theme normalization overrides */
        .m-pos,
        .order-action.buy,
        .hist-action-buy,
        .exec-action.buy,
        .logic-val-yes,
        .dd-pass,
        .operator-card-value.good {
            color: var(--success) !important;
        }
        .m-neg,
        .order-action.sell,
        .hist-action-sell,
        .exec-action.sell,
        .logic-val-no,
        .dd-reject,
        .operator-card-value.err {
            color: var(--error) !important;
        }
        .operator-card-value.warn,
        .status-card.warn .status-card-dot,
        .activity-tag.warn {
            color: var(--warning) !important;
        }
        .dot-live,
        .exec-row.buy:before {
            background: var(--success) !important;
            box-shadow: none !important;
        }
        .dot-rest,
        .rail-dot-warn,
        .exec-row.hold-filtered:before {
            background: var(--warning) !important;
            box-shadow: none !important;
        }
        .dot-error,
        .rail-dot-err,
        .exec-row.sell:before,
        .exec-row.blocked:before {
            background: var(--error) !important;
            box-shadow: none !important;
        }
        .exec-row.hold-no-signal:before,
        .exec-row.hold-rejected:before,
        .status-card.info .status-card-dot {
            background: #6E7681 !important;
        }
        .badge,
        .approval-chip,
        .operator-chip,
        .config-chip,
        .activity-tag,
        .exec-type-badge,
        [data-testid="stTabs"] [role="tab"] {
            border-radius: 8px !important;
            box-shadow: none !important;
        }
        .b-buy,
        .b-ready,
        .approval-chip.good,
        .operator-chip.good {
            background: var(--success-bg) !important;
            color: var(--success) !important;
            border: 1px solid transparent !important;
        }
        .b-err,
        .b-blocked,
        .approval-chip.err,
        .operator-chip.err {
            background: var(--error-bg) !important;
            color: var(--error) !important;
            border: 1px solid transparent !important;
        }
        .b-sell,
        .b-hold,
        .b-nosignal,
        .approval-chip.warn,
        .operator-chip.warn,
        .exec-type-badge.no-signal,
        .activity-tag.info {
            background: var(--muted-bg) !important;
            color: var(--text-soft) !important;
            border: 1px solid transparent !important;
        }
        .exec-type-badge.filtered {
            background: var(--warning-bg) !important;
            color: var(--warning) !important;
        }
        .exec-type-badge.rejected,
        .exec-type-badge.blocked,
        .activity-tag.err {
            background: var(--error-bg) !important;
            color: var(--error) !important;
        }
        .activity-tag.warn,
        .status-card.warn {
            background: var(--warning-bg) !important;
        }
        .status-card.err {
            background: var(--error-bg) !important;
        }
        .status-card.info {
            background: var(--muted-bg) !important;
        }
        .config-card,
        .dash-metrics,
        .detail-history-card,
        .rail-card,
        .detail-hero,
        .watchlist-shell,
        .exec-panel,
        .dd-shell,
        .timing-shell,
        .section-intro,
        .workspace-map-card,
        .hero-card,
        .action-card,
        .status-panel,
        .compact-card,
        .compact-card.feed-card,
        .operator-header-main,
        .operator-header-card,
        .operator-mini,
        .config-stat,
        .config-empty,
        .history-stat,
        .live-zone,
        .detail-info-cell,
        .hist-wrap,
        .ticker-strip-ribbon,
        .ticker-chip,
        .exec-row,
        .logic-item,
        .cycle-summary,
        .timing-item,
        [data-testid="stButton"] > button[kind="secondary"],
        [data-testid="stTabs"] [role="tab"],
        [data-testid="stTabs"] [aria-selected="true"] {
            background: var(--panel) !important;
            border: 1px solid var(--line) !important;
            border-radius: 8px !important;
            box-shadow: none !important;
        }
        .config-card::after,
        .hero-card:before {
            content: none !important;
            display: none !important;
            background: none !important;
        }
        .stApp,
        .config-card,
        .dash-metrics,
        .detail-history-card,
        .rail-card,
        .detail-hero,
        .watchlist-shell,
        .exec-panel,
        .dd-shell,
        .timing-shell,
        .section-intro,
        .workspace-map-card,
        .hero-card,
        .action-card,
        .status-panel,
        .compact-card,
        .operator-header-main,
        .operator-header-card,
        .operator-mini,
        .config-stat,
        .config-empty,
        .history-stat,
        .live-zone,
        .detail-info-cell,
        .hist-wrap,
        .ticker-strip-ribbon,
        .ticker-chip,
        .exec-row,
        .logic-item,
        .cycle-summary,
        .timing-item {
            filter: none !important;
        }
        .top-title,
        .operator-title,
        .status-panel-title,
        .section-intro-title,
        .workspace-map-title,
        .detail-symbol,
        .order-price,
        .rail-row-val,
        .rail-sym,
        .sym-ticker,
        .exec-symbol,
        .timing-value,
        .operator-mini-value,
        .operator-card-value,
        .config-stat-value,
        .config-row-val,
        .kv-val,
        .dd-symbol,
        .dd-step-val,
        .hero-stat-value {
            color: var(--text) !important;
            font-family: var(--font-family-ui);
        }
        .subtle-copy,
        .hero-note,
        .status-panel-copy,
        .status-metric-note,
        .config-card-note,
        .config-stat-sub,
        .config-empty,
        .workspace-map-copy,
        .workspace-map-meta,
        .watchlist-subtext,
        .watchlist-tip,
        .live-zone-copy,
        .activity-body,
        .exec-reason,
        .thought-text,
        .timing-help,
        .operator-subtitle,
        .operator-mini-note,
        .operator-card-note,
        .rail-empty,
        .order-detail,
        .order-time,
        .history-stat-label,
        .feed-subline,
        .feed-stat-label,
        .feed-stat-value,
        .ticker-chip-price,
        .thought-ts,
        .dd-ts,
        .dd-step-key,
        .dd-note,
        .kv-key,
        .config-row-key,
        .config-band-label,
        .config-stat-label,
        .detail-info-cell-label,
        .dash-metric-label,
        .status-panel-kicker,
        .operator-kicker,
        .operator-mini-label,
        .operator-card-label,
        .activity-time,
        .activity-title,
        .sec-head,
        .ticker-strip-head .subtle-copy,
        .logic-key,
        .sym-card-v2-meta,
        .dm-mono,
        .ml-label,
        .ml-axis,
        .prox-label {
            color: var(--text-soft) !important;
        }
        .top-title,
        .operator-title {
            font-size: var(--font-size-xl) !important;
            font-weight: var(--font-weight-bold) !important;
            line-height: var(--line-height-tight) !important;
            letter-spacing: var(--letter-spacing-tight) !important;
        }
        .top-title {
            font-size: var(--font-size-2xl) !important;
        }
        .status-panel-title,
        .section-intro-title,
        .workspace-map-title,
        .detail-symbol,
        .exec-symbol,
        .operator-mini-value,
        .operator-card-value,
        .config-stat-value,
        .hero-stat-value {
            font-size: var(--font-size-lg) !important;
            font-weight: var(--font-weight-semibold) !important;
            line-height: var(--line-height-snug) !important;
        }
        .dash-metric-val,
        .detail-info-cell-val,
        .detail-price,
        .metric-val-big {
            font-size: var(--font-size-2xl) !important;
            font-weight: var(--font-weight-bold) !important;
            line-height: var(--line-height-tight) !important;
            letter-spacing: var(--letter-spacing-tight) !important;
        }
        .sec-head,
        .operator-kicker,
        .operator-mini-label,
        .operator-card-label,
        .status-panel-kicker,
        .status-metric-label,
        .config-card-title,
        .config-band-label,
        .config-stat-label,
        .config-row-key,
        .detail-info-cell-label,
        .dash-metric-label,
        .activity-tag,
        .badge {
            font-size: var(--font-size-xs) !important;
            font-weight: var(--font-weight-semibold) !important;
            letter-spacing: var(--letter-spacing-wide) !important;
            line-height: var(--line-height-snug) !important;
        }
        .subtle-copy,
        .hero-note,
        .status-panel-copy,
        .status-metric-note,
        .config-card-note,
        .config-stat-sub,
        .config-empty,
        .workspace-map-copy,
        .watchlist-tip,
        .activity-body,
        .exec-reason,
        .thought-text,
        .timing-help,
        .operator-subtitle,
        .operator-mini-note,
        .operator-card-note,
        .rail-empty {
            font-size: var(--font-size-sm) !important;
            line-height: var(--line-height-relaxed) !important;
        }
        .activity-time,
        .order-time,
        .thought-ts,
        .dd-ts,
        .config-row-key,
        .hist-table th,
        .detail-info-cell-label {
            font-size: var(--font-size-xs) !important;
        }
        .watchlist-subtext,
        .ticker-chip-price,
        .feed-subline,
        .feed-stat-label,
        .feed-stat-value,
        .order-detail,
        .rail-row-meta,
        .dd-note,
        .dd-step-key {
            font-size: var(--font-size-sm) !important;
            line-height: var(--line-height-normal) !important;
        }
        .block-container,
        .hist-table td,
        .exec-price,
        .exec-action.hold,
        .logic-val-neutral,
        .activity-tag,
        .ticker-chip-symbol,
        .order-symbol {
            color: var(--text) !important;
        }
        .kv,
        .config-row,
        .config-band,
        .dash-metric,
        .sym-card-v2,
        .rail-row,
        .order-row,
        .thought-row,
        .activity-feed-row,
        .dd-section,
        .dd-path-step,
        .hist-table th,
        .hist-table td,
        [data-testid="stTabs"] [role="tablist"] {
            border-color: var(--line) !important;
        }
        .sym-card-v2:hover,
        .sym-card-v2.sym-active,
        .ticker-chip.active {
            background: rgba(157, 167, 179, 0.08) !important;
        }
        .sym-card-v2.sym-error,
        .detail-error-banner,
        .kill-switch-row {
            background: var(--error-bg) !important;
            border-color: rgba(248, 81, 73, 0.3) !important;
            color: var(--error) !important;
        }
        .status-panel.warn,
        .operator-header-card.warn,
        .operator-header-main.warn {
            background: var(--panel) !important;
            border-color: rgba(210, 153, 34, 0.35) !important;
        }
        .status-panel.err,
        .operator-header-card.err,
        .operator-header-main.err {
            background: var(--panel) !important;
            border-color: rgba(248, 81, 73, 0.35) !important;
        }
        .prox-track,
        .ml-track {
            background: #11161D !important;
        }
        .prox-above {
            background: var(--success) !important;
        }
        .prox-below {
            background: var(--error) !important;
        }
        .hist-table th,
        .exec-row {
            background: var(--panel) !important;
        }
        .trend-bar {
            border-radius: 2px !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


# â”€â”€ Bot singleton â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

# â”€â”€ Utilities â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

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


def _fmt_optional_money(v) -> str:
    if v is None or pd.isna(v):
        return "â€”"
    return _fmt_money(float(v))


def _relative_age(ts) -> str:
    try:
        snap_dt = datetime.fromisoformat(str(ts).replace("Z", "+00:00"))
        age_s = int((datetime.now(timezone.utc) - snap_dt).total_seconds())
        if age_s < 60:
            return f"{age_s}s ago"
        return f"{age_s // 60}m ago"
    except Exception:
        return str(ts)


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


# â”€â”€ HTML builders â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def _ml_bar_html(
    prob: float | None,
    buy_thr: float | None,
    sell_thr: float | None,
    confidence: float | None,
) -> str:
    if prob is None:
        return "<div class='ml-wrap'><div class='ml-label'>ML Probability â€” n/a</div></div>"
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


def _ml_mini_bar_html(
    prob: float | None,
    buy_thr: float | None,
    sell_thr: float | None,
) -> str:
    """Compact track-only ML bar for watchlist cards â€” no labels, no axis."""
    if prob is None:
        return ""
    pct = min(max(prob * 100, 0), 100)
    if buy_thr is not None and prob >= buy_thr:
        fill_color = "#4ade80"
    elif sell_thr is not None and prob <= sell_thr:
        fill_color = "#f87171"
    else:
        fill_color = "#64748b"
    buy_tick = (
        f"<div class='ml-tick' style='left:{buy_thr * 100:.1f}%'></div>"
        if buy_thr is not None else ""
    )
    sell_tick = (
        f"<div class='ml-tick' style='left:{sell_thr * 100:.1f}%'></div>"
        if sell_thr is not None else ""
    )
    return (
        f"<div class='ml-track' style='margin:5px 0 2px'>"
        f"<div class='ml-fill' style='width:{pct:.1f}%;background:{fill_color}'></div>"
        f"{sell_tick}{buy_tick}"
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
    normalized = (action or "").upper()
    cls = {"BUY": "b-buy", "SELL": "b-sell", "ERROR": "b-err", "SNAPSHOT_ONLY": "b-nosignal"}.get(
        normalized, "b-hold"
    )
    label = "SNAPSHOT" if normalized == "SNAPSHOT_ONLY" else action
    return f"<span class='badge {cls}'>{label}</span>"


def _kv(key: str, val: str) -> str:
    return (
        f"<div class='kv'>"
        f"<span class='kv-key'>{key}</span>"
        f"<span class='kv-val'>{val}</span>"
        f"</div>"
    )


def _decision_log_cell(label: str, value_html: str, meta_html: str = "", strong: bool = False) -> str:
    value_cls = "decision-log-row-value strong" if strong else "decision-log-row-value"
    meta = f"<div class='decision-log-row-meta'>{meta_html}</div>" if meta_html else ""
    return (
        "<div class='decision-log-cell-block'>"
        f"<div class='decision-log-col-label'>{_escape_html(label)}</div>"
        f"<div class='{value_cls}'>{value_html}</div>"
        f"{meta}"
        "</div>"
    )


def _decision_log_outcome_badge(outcome: str, tone: str) -> str:
    tone_cls = {"good": "good", "warn": "warn", "neutral": "neutral"}.get(tone, "neutral")
    return f"<span class='decision-log-outcome {tone_cls}'>{_escape_html(outcome)}</span>"


def _escape_html(value) -> str:
    return html.escape(str(value if value not in (None, "") else "â€”"))


def _config_stat_html(label: str, value, subtext: str | None = None) -> str:
    sub = f"<div class='config-stat-sub'>{_escape_html(subtext)}</div>" if subtext else ""
    return (
        "<div class='config-stat'>"
        f"<div class='config-stat-label'>{_escape_html(label)}</div>"
        f"<div class='config-stat-value'>{_escape_html(value)}</div>"
        f"{sub}"
        "</div>"
    )


def _config_row_html(label: str, value) -> str:
    return (
        "<div class='config-row'>"
        f"<div class='config-row-key'>{_escape_html(label)}</div>"
        f"<div class='config-row-val'>{_escape_html(value)}</div>"
        "</div>"
    )


def _config_chip_cloud_html(values: list[str] | tuple[str, ...], muted: bool = False) -> str:
    if not values:
        values = ["n/a"]
        muted = True
    cls = "config-chip muted" if muted else "config-chip"
    chips = "".join(f"<span class='{cls}'>{_escape_html(value)}</span>" for value in values)
    return f"<div class='config-chip-cloud'>{chips}</div>"


def _render_bar_timing(bot) -> None:
    timeframe_minutes = getattr(bot.config, "bar_timeframe_minutes", 0) or 15
    timing = _get_bar_timing(int(timeframe_minutes))
    current_time_utc: datetime = timing["current_time_utc"]  # type: ignore[assignment]
    next_bar_close_utc: datetime = timing["next_bar_close_utc"]  # type: ignore[assignment]
    seconds_remaining = timing["seconds_remaining"]
    timeframe_minutes = timing["timeframe_minutes"]

    components.html(
        f"""
        <style>
          body {{
            background: transparent;
            margin: 0;
            padding: 0;
            font-family: 'DM Sans', sans-serif;
            color: #e8eef7;
          }}
        </style>
        <div style="margin:0.35rem 0 0.75rem;padding:12px;border-radius:8px;background:#151B23;border:1px solid #1F2733;box-shadow:none;">
          <div style="display:flex;justify-content:space-between;align-items:flex-start;gap:18px;flex-wrap:wrap;">
            <div style="min-width:240px;">
              <div style="font-size:0.78rem;text-transform:uppercase;letter-spacing:0.14em;color:#9DA7B3;font-weight:700;margin-bottom:0.45rem;">Next Bar In</div>
              <div id="bot-timing-countdown" style="font-size:2.2rem;line-height:1;font-weight:800;color:#E6EDF3;font-family:'DM Mono', monospace;"></div>
              <div style="font-size:0.9rem;color:#9DA7B3;margin-top:0.55rem;">Next bar closes at <span id="bot-timing-close-inline" style="color:#E6EDF3;font-family:'DM Mono', monospace;font-weight:700;"></span></div>
            </div>
            <div style="display:grid;grid-template-columns:repeat(auto-fit, minmax(180px, 1fr));gap:12px;flex:1 1 420px;">
              <div style="background:#151B23;border:1px solid #1F2733;border-radius:8px;padding:12px;">
                <div style="font-size:0.72rem;text-transform:uppercase;letter-spacing:0.12em;color:#9DA7B3;margin-bottom:0.35rem;">Current UTC</div>
                <div id="bot-timing-now" style="font-size:1rem;font-weight:700;color:#E6EDF3;font-family:'DM Mono', monospace;"></div>
              </div>
              <div style="background:#151B23;border:1px solid #1F2733;border-radius:8px;padding:12px;">
                <div style="font-size:0.72rem;text-transform:uppercase;letter-spacing:0.12em;color:#9DA7B3;margin-bottom:0.35rem;">Next Bar Close</div>
                <div id="bot-timing-close" style="font-size:1rem;font-weight:700;color:#E6EDF3;font-family:'DM Mono', monospace;"></div>
              </div>
              <div style="background:#151B23;border:1px solid #1F2733;border-radius:8px;padding:12px;">
                <div style="font-size:0.72rem;text-transform:uppercase;letter-spacing:0.12em;color:#9DA7B3;margin-bottom:0.35rem;">Bar Timeframe</div>
                <div style="font-size:1rem;font-weight:700;color:#E6EDF3;font-family:'DM Mono', monospace;">{timeframe_minutes} min</div>
              </div>
            </div>
          </div>
          <script>
            const baseNowMs = {int(current_time_utc.timestamp() * 1000)};
            const nextCloseMs = {int(next_bar_close_utc.timestamp() * 1000)};
            const baseRemaining = {seconds_remaining};

            function formatUtc(tsMs) {{
              const iso = new Date(tsMs).toISOString();
              const yyyy = iso.slice(0, 4);
              const mm = iso.slice(5, 7);
              const dd = iso.slice(8, 10);
              const timeBits = iso.slice(11, 19).split(":");
              let hour = Number(timeBits[0]);
              const meridiem = hour >= 12 ? "PM" : "AM";
              hour = hour % 12 || 12;
              const hourLabel = String(hour).padStart(2, "0");
              const time = `${{hourLabel}}:${{timeBits[1]}}:${{timeBits[2]}} ${{meridiem}}`;
              return `${{mm}}-${{dd}}-${{yyyy}} ${{time}} UTC`;
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
              document.getElementById("bot-timing-close-inline").textContent = formatUtc(nextCloseMs);
              document.getElementById("bot-timing-countdown").textContent = formatCountdown(remaining);
            }}

            tick();
            setInterval(tick, 1000);
          </script>
        </div>
        """,
        height=170,
    )


# â”€â”€ Drilldown panel â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def _drilldown_label(event: DrilldownEvent) -> str:
    """Short selector label for one drilldown event."""
    time_str = _compact_time_str(event.timestamp_utc)
    etype_short = {
        "signal.evaluated": "sig",
        "risk.check": "risk",
        "position.closed": "exit",
    }.get(event.event_type, event.event_type)
    action = event.action or "â€”"
    return f"{event.symbol}  {time_str}  [{etype_short}]  {action}"


def _drilldown_path_html(event: DrilldownEvent) -> str:
    """Compact decision path steps appropriate for the event type."""
    steps: list[str] = []
    idx = 1

    def _step(key: str, val_html: str, note: str = "") -> None:
        nonlocal idx
        note_html = (
            f"<span class='dd-note'>{_escape_html(note)}</span>" if note else ""
        )
        steps.append(
            f"<div class='dd-path-step'>"
            f"<span class='dd-step-idx'>{idx}</span>"
            f"<span class='dd-step-key'>{_escape_html(key)}</span>"
            f"<span class='dd-step-val'>{val_html}{note_html}</span>"
            f"</div>"
        )
        idx += 1

    et = event.event_type

    if et == "signal.evaluated":
        if event.deviation_pct is not None:
            sign = "+" if event.deviation_pct >= 0 else ""
            direction = "above" if event.deviation_pct >= 0 else "below"
            _step(
                "Deviation from SMA",
                f"<span class='dd-step-val'>{sign}{event.deviation_pct:.2f}%</span>",
                f"{direction} reference",
            )
        if event.trend_filter is not None:
            is_pass = event.trend_filter.lower() == "pass"
            cls = "dd-pass" if is_pass else "dd-reject"
            trend_note = (
                "above trend SMA" if event.above_trend_sma is True
                else ("below trend SMA" if event.above_trend_sma is False else "")
            )
            _step("Trend filter", f"<span class='{cls}'>{event.trend_filter.upper()}</span>", trend_note)
        if event.atr_filter is not None:
            is_pass = event.atr_filter.lower() == "pass"
            cls = "dd-pass" if is_pass else "dd-reject"
            atr_parts: list[str] = []
            if event.atr_pct is not None:
                atr_parts.append(f"ATR {event.atr_pct:.2%}")
            if event.atr_percentile is not None:
                atr_parts.append(f"{event.atr_percentile:.0f}th pct")
            _step("ATR filter", f"<span class='{cls}'>{event.atr_filter.upper()}</span>", " · ".join(atr_parts))
        if event.ml_prob is not None:
            pct = int(event.ml_prob * 100)
            color = "#4ade80" if event.ml_prob >= 0.6 else ("#f87171" if event.ml_prob <= 0.4 else "#d7e1ec")
            _step(
                "ML probability",
                f"<span style='color:{color};font-weight:700'>{event.ml_prob:.3f} ({pct}%)</span>",
            )
        if event.volume_ratio is not None:
            color = "#4ade80" if event.volume_ratio >= 1.0 else "#f87171"
            note = "above avg" if event.volume_ratio >= 1.0 else "below avg"
            _step("Volume ratio", f"<span style='color:{color}'>{event.volume_ratio:.2f}x</span>", note)
        if event.window_open is not None:
            cls = "dd-pass" if event.window_open else "dd-reject"
            _step("Entry window", f"<span class='{cls}'>{'OPEN' if event.window_open else 'CLOSED'}</span>")
        action = (event.action or "HOLD").upper()
        action_cls = {"BUY": "b-buy", "SELL": "b-sell"}.get(action, "b-hold")
        if action in {"BUY", "SELL"}:
            outcome_note = "cleared all filters"
        elif event.rejection:
            outcome_note = f"reason: {_pretty_reason(event.rejection)}"
        else:
            outcome_note = "no actionable signal"
        _step("â†’ Decision", f"<span class='badge {action_cls}'>{action}</span>", outcome_note)

    elif et == "risk.check":
        action = (event.action or "").upper()
        action_cls = {"BUY": "b-buy", "SELL": "b-sell"}.get(action, "b-hold")
        _step("Signal direction", f"<span class='badge {action_cls}'>{action or 'â€”'}</span>")
        if event.allowed is True:
            _step("Risk gate", "<span class='dd-pass'>PASSED</span>", "all checks cleared")
        elif event.allowed is False:
            block_note = _pretty_reason(event.block_reason) if event.block_reason else "reason unknown"
            _step("Risk gate", "<span class='dd-reject'>BLOCKED</span>", block_note)
        else:
            _step("Risk gate", "<span class='dd-step-val'>â€”</span>")

    elif et == "position.closed":
        exit_label = _escape_html(event.exit_reason or "â€”")
        exit_note = _pretty_reason(event.exit_reason) if event.exit_reason else ""
        _step("Exit trigger", f"<span class='dd-step-val'>{exit_label}</span>", exit_note)
        if event.pnl_usd is not None:
            sign = "+" if event.pnl_usd >= 0 else ""
            cls = "dd-pass" if event.pnl_usd >= 0 else "dd-reject"
            _step("Realized P&L", f"<span class='{cls}'>{sign}${abs(event.pnl_usd):.2f}</span>")

    if not steps:
        return (
            "<div class='dd-section'>"
            "<div class='sec-head' style='margin-top:0'>Decision Path</div>"
            "<div class='rail-empty'>No decision data available for this event type.</div>"
            "</div>"
        )
    return (
        "<div class='dd-section'>"
        "<div class='sec-head' style='margin-top:0'>Decision Path</div>"
        f"<div class='dd-path'>{''.join(steps)}</div>"
        "</div>"
    )


def _drilldown_panel_html(event: DrilldownEvent) -> str:
    """Full drilldown card for one decision event."""
    et = event.event_type
    action = (event.action or "").upper()

    etype_label = {
        "signal.evaluated": "SIGNAL",
        "risk.check": "RISK CHECK",
        "position.closed": "EXIT",
    }.get(et, et.upper().replace(".", " "))

    etype_cls = {
        "signal.evaluated": "b-hold",
        "risk.check": "b-blocked" if event.allowed is False else "b-ready",
        "position.closed": "b-sell",
    }.get(et, "b-nosignal")
    action_badge_cls = {"BUY": "b-buy", "SELL": "b-sell", "HOLD": "b-hold"}.get(action, "b-nosignal")

    ts_str = _format_datetime_pretty(event.timestamp_utc) if event.timestamp_utc else "â€”"

    # Allowed / blocked display
    if event.allowed is True:
        allowed_html = "<span class='dd-pass'>allowed</span>"
    elif event.allowed is False:
        allowed_html = "<span class='dd-reject'>blocked</span>"
    else:
        allowed_html = "<span style='color:#4a6070'>n/a</span>"

    # Primary rejection/block reason
    reason_raw = event.block_reason or event.rejection or event.exit_reason
    reason_html = _escape_html(_pretty_reason(reason_raw) if reason_raw else "â€”")

    fields: list[str] = [
        _kv("Symbol", _escape_html(event.symbol or "â€”")),
        _kv("Timestamp", ts_str),
        _kv("Event type", _escape_html(etype_label)),
        _kv("Action", _escape_html(action or "â€”")),
        _kv("Strategy mode", _escape_html(event.strategy_mode or "â€”")),
        _kv("Allowed / blocked", allowed_html),
        _kv("Rejection / reason", reason_html),
    ]

    if event.trend_filter is not None:
        cls = "dd-pass" if event.trend_filter.lower() == "pass" else "dd-reject"
        fields.append(_kv("Trend filter", f"<span class='{cls}'>{_escape_html(event.trend_filter.upper())}</span>"))
    if event.atr_filter is not None:
        cls = "dd-pass" if event.atr_filter.lower() == "pass" else "dd-reject"
        fields.append(_kv("ATR filter", f"<span class='{cls}'>{_escape_html(event.atr_filter.upper())}</span>"))
    if event.deviation_pct is not None:
        sign = "+" if event.deviation_pct >= 0 else ""
        fields.append(_kv("Deviation (% vs SMA)", f"{sign}{event.deviation_pct:.2f}%"))
    if event.ml_prob is not None:
        fields.append(_kv("ML probability", f"{event.ml_prob:.3f} ({int(event.ml_prob * 100)}%)"))
    if event.pnl_usd is not None:
        sign = "+" if event.pnl_usd >= 0 else ""
        cls = "dd-pass" if event.pnl_usd >= 0 else "dd-reject"
        fields.append(_kv("Realized P&L", f"<span class='{cls}'>{sign}${abs(event.pnl_usd):.2f}</span>"))

    return (
        "<div class='dd-shell'>"
        "<div class='dd-header'>"
        f"<span class='dd-symbol'>{_escape_html(event.symbol)}</span>"
        f"<span class='badge {etype_cls}'>{etype_label}</span>"
        f"<span class='badge {action_badge_cls}'>{action or 'â€”'}</span>"
        f"<span class='dd-ts'>{ts_str}</span>"
        "</div>"
        f"{''.join(fields)}"
        + _drilldown_path_html(event)
        + "</div>"
    )


def _decision_log_outcome(event: DrilldownEvent) -> tuple[str, str, str]:
    if event.event_type == "risk.check":
        if event.allowed is False:
            return "Blocked", "warn", _pretty_reason(event.block_reason)
        if event.allowed is True:
            return "Allowed", "good", "Passed risk checks"
        return "Unknown", "neutral", "Risk state unavailable"
    if event.event_type == "position.closed":
        reason = _pretty_reason(event.exit_reason) if event.exit_reason else "Position closed"
        return "Closed", "neutral", reason
    action = (event.action or "").upper()
    if action in {"BUY", "SELL"}:
        return action, "good", "Actionable signal"
    if event.rejection:
        return "Rejected", "warn", _pretty_reason(event.rejection)
    return "No signal", "neutral", "No actionable signal"


def _normalize_decision_log_rows(candidates: list[DrilldownEvent]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    type_labels = {
        "signal.evaluated": "Signal",
        "risk.check": "Risk Check",
        "position.closed": "Exit",
    }
    priority_map = {
        "Blocked": 0,
        "Rejected": 1,
        "Closed": 2,
        "BUY": 3,
        "SELL": 3,
        "Allowed": 4,
        "No signal": 5,
        "Unknown": 6,
    }
    for index, event in enumerate(candidates):
        outcome, tone, reason = _decision_log_outcome(event)
        timestamp = _parse_datetime(event.timestamp_utc)
        event_type_label = type_labels.get(event.event_type, event.event_type.replace(".", " ").title())
        summary = reason
        if event.event_type == "signal.evaluated" and outcome in {"BUY", "SELL"}:
            summary = f"{outcome} signal ready"
        elif event.event_type == "position.closed" and event.pnl_usd is not None:
            pnl_label = f"{'+' if event.pnl_usd >= 0 else '-'}${abs(event.pnl_usd):.2f}"
            summary = f"{reason} · {pnl_label}"
        rows.append(
            {
                "id": f"{event.symbol}|{event.event_type}|{event.timestamp_utc}|{index}",
                "event": event,
                "timestamp": timestamp,
                "timestamp_label": _compact_time_str(event.timestamp_utc),
                "timestamp_full": _format_datetime_pretty(event.timestamp_utc),
                "symbol": event.symbol or "â€”",
                "event_type": event.event_type,
                "event_type_label": event_type_label,
                "outcome": outcome,
                "tone": tone,
                "reason": reason,
                "summary": summary,
                "priority": priority_map.get(outcome, 7),
                "search_text": " ".join(
                    str(part or "")
                    for part in [
                        event.symbol,
                        event.event_type,
                        outcome,
                        reason,
                        event.block_reason,
                        event.rejection,
                        event.exit_reason,
                        event.strategy_mode,
                    ]
                ).lower(),
            }
        )
    rows.sort(
        key=lambda row: (
            int(row["priority"]),
            -(row["timestamp"].timestamp() if row["timestamp"] is not None else 0),
        )
    )
    return rows


def _render_decision_log_filters(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    event_options = ["All"] + sorted({str(row["event_type_label"]) for row in rows})
    symbol_options = ["All"] + sorted({str(row["symbol"]) for row in rows})
    outcome_options = ["All"] + sorted({str(row["outcome"]) for row in rows})
    filter_cols = st.columns([1.15, 0.9, 0.9, 0.75, 1.3])
    with filter_cols[0]:
        event_filter = st.selectbox("Event Type", event_options, index=0, key="decision_log_event_type")
    with filter_cols[1]:
        symbol_filter = st.selectbox("Symbol", symbol_options, index=0, key="decision_log_symbol")
    with filter_cols[2]:
        outcome_filter = st.selectbox("Outcome", outcome_options, index=0, key="decision_log_outcome")
    with filter_cols[3]:
        max_rows = st.selectbox("Rows", [10, 20, 30, 50], index=1, key="decision_log_rows")
    with filter_cols[4]:
        search_text = st.text_input("Search", value="", placeholder="reason, symbol, event", key="decision_log_search").strip().lower()

    filtered = rows
    if event_filter != "All":
        filtered = [row for row in filtered if row["event_type_label"] == event_filter]
    if symbol_filter != "All":
        filtered = [row for row in filtered if row["symbol"] == symbol_filter]
    if outcome_filter != "All":
        filtered = [row for row in filtered if row["outcome"] == outcome_filter]
    if search_text:
        filtered = [row for row in filtered if search_text in str(row["search_text"])]
    return filtered[: int(max_rows)]


def _render_decision_log_summary(rows: list[dict[str, object]]) -> None:
    blocked_count = sum(1 for row in rows if row["outcome"] == "Blocked")
    rejected_count = sum(1 for row in rows if row["outcome"] == "Rejected")
    exit_count = sum(1 for row in rows if row["event_type"] == "position.closed")
    summary_html = (
        "<div class='decision-log-summary'>"
        "<section class='decision-log-summary-card'>"
        "<div class='tradeos-section-label'>Blocked</div>"
        f"<div class='tradeos-primary-value'>{blocked_count}</div>"
        "<div class='tradeos-body-text'>Recent blocked decisions</div>"
        "</section>"
        "<section class='decision-log-summary-card'>"
        "<div class='tradeos-section-label'>Rejected</div>"
        f"<div class='tradeos-primary-value'>{rejected_count}</div>"
        "<div class='tradeos-body-text'>Filtered setups</div>"
        "</section>"
        "<section class='decision-log-summary-card'>"
        "<div class='tradeos-section-label'>Closures</div>"
        f"<div class='tradeos-primary-value'>{exit_count}</div>"
        "<div class='tradeos-body-text'>Recent position exits</div>"
        "</section>"
        "</div>"
    )
    st.markdown(summary_html, unsafe_allow_html=True)


def _render_decision_log_table(rows: list[dict[str, object]]) -> dict[str, object] | None:
    if not rows:
        st.markdown("<div class='decision-log-empty tradeos-body-text'>No events match the current filters.</div>", unsafe_allow_html=True)
        return None

    selected_id = st.session_state.get("decision_log_selected_id")
    row_ids = {str(row["id"]) for row in rows}
    if selected_id not in row_ids:
        selected_id = str(rows[0]["id"])
        st.session_state["decision_log_selected_id"] = selected_id

    header_cols = st.columns([1.15, 0.7, 0.95, 0.85, 1.6, 0.7])
    for col, label in zip(header_cols, ["Time", "Symbol", "Event", "Outcome", "Reason", "View"]):
        col.markdown(f"<div class='tradeos-section-label'>{label}</div>", unsafe_allow_html=True)

    selected_row: dict[str, object] | None = None
    for row in rows:
        row_cols = st.columns([1.15, 0.7, 0.95, 0.85, 1.6, 0.7])
        row_cols[0].markdown(
            _decision_log_cell("Time", str(row["timestamp_label"]), str(row["timestamp_full"])),
            unsafe_allow_html=True,
        )
        row_cols[1].markdown(
            _decision_log_cell("Symbol", _escape_html(row["symbol"]), strong=True),
            unsafe_allow_html=True,
        )
        row_cols[2].markdown(
            _decision_log_cell("Event", _escape_html(row["event_type_label"]), _escape_html(row["event_type"])),
            unsafe_allow_html=True,
        )
        row_cols[3].markdown(
            _decision_log_cell(
                "Outcome",
                _decision_log_outcome_badge(str(row["outcome"]), str(row["tone"])),
            ),
            unsafe_allow_html=True,
        )
        row_cols[4].markdown(
            _decision_log_cell("Why", _escape_html(row["summary"]), _escape_html(row["reason"])),
            unsafe_allow_html=True,
        )
        button_label = "Open" if str(row["id"]) != selected_id else "Selected"
        if row_cols[5].button(button_label, key=f"decision_log_{row['id']}", use_container_width=True, type="secondary"):
            st.session_state["decision_log_selected_id"] = str(row["id"])
            selected_id = str(row["id"])
        if str(row["id"]) == selected_id:
            selected_row = row
    return selected_row or rows[0]


def _render_decision_log_detail(selected_row: dict[str, object] | None) -> None:
    if selected_row is None:
        st.markdown("<div class='decision-log-empty tradeos-body-text'>Select an event to inspect it.</div>", unsafe_allow_html=True)
        return

    event = selected_row["event"]
    detail_meta = "".join(
        [
            _kv("Symbol", _escape_html(selected_row["symbol"])),
            _kv("Timestamp", str(selected_row["timestamp_full"])),
            _kv("Event", _escape_html(selected_row["event_type_label"])),
            _kv("Outcome", _escape_html(selected_row["outcome"])),
            _kv("Reason", _escape_html(selected_row["reason"])),
        ]
    )
    st.markdown(
        "<div class='decision-log-pane'>"
        "<div class='decision-log-detail-hero'>"
        "<div class='tradeos-section-label'>Selected Event</div>"
        f"<div class='decision-log-detail-headline'>{_escape_html(selected_row['summary'])}</div>"
        "<div class='decision-log-detail-chip-row'>"
        f"<span class='decision-log-detail-chip'>{_escape_html(selected_row['symbol'])}</span>"
        f"<span class='decision-log-detail-chip'>{_escape_html(selected_row['event_type_label'])}</span>"
        f"{_decision_log_outcome_badge(str(selected_row['outcome']), str(selected_row['tone']))}"
        "</div>"
        "</div>"
        f"<div class='decision-log-detail-grid'>{detail_meta}</div>"
        "</div>",
        unsafe_allow_html=True,
    )
    st.markdown(_drilldown_panel_html(event), unsafe_allow_html=True)


def _render_drilldown(state: DashboardState) -> None:
    """Render the Trade Decision Drilldown tab."""
    st.markdown(
        _section_intro_html(
            "Decision Log",
            "Trace a single cycle from signal to outcome",
            "Use this area when you need the why behind a decision, not just the latest state.",
        ),
        unsafe_allow_html=True,
    )
    candidates = list(state.drilldown_candidates)

    if not candidates:
        st.markdown(
            "<div class='rail-empty' style='margin-top:1.5rem'>"
            "No recent decision events are available yet. "
            "This usually means the current session has not written its first full bar-cycle logs."
            "</div>",
            unsafe_allow_html=True,
        )
        return

    rows = _normalize_decision_log_rows(candidates)
    filtered_rows = _render_decision_log_filters(rows)
    _render_decision_log_summary(filtered_rows)

    left_pane, right_pane = st.columns([1.2, 1])
    with left_pane:
        st.markdown("<div class='decision-log-pane'>", unsafe_allow_html=True)
        selected_row = _render_decision_log_table(filtered_rows)
        st.markdown("</div>", unsafe_allow_html=True)
    with right_pane:
        _render_decision_log_detail(selected_row)


# â”€â”€ Error page â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def render_startup_error(exc: Exception) -> None:
    st.markdown(
        "<div class='ks-banner'>Bot initialization failed â€” see details below</div>",
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


# â”€â”€ Orders helper â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

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
        "stale_market_data": "Stopped: market data was stale",
        "signals_blocked_or_skipped": "Signals found, but no orders were submitted",
        "no_actionable_signals": "No BUY/SELL signals on this cycle",
        "execution_completed": "Execution completed",
    }
    return mapping.get(reason, reason.replace("_", " ").title())


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


def _operator_chip_html(label: str, tone: str = "neutral") -> str:
    safe_tone = tone if tone in {"good", "warn", "err"} else ""
    cls = f"operator-chip {safe_tone}".strip()
    return f"<span class='{cls}'>{_escape_html(label)}</span>"


def _operator_card_html(label: str, value: str, note: str = "", tone: str = "neutral") -> str:
    safe_tone = tone if tone in {"good", "warn", "err"} else ""
    note_html = f"<div class='operator-card-note'>{_escape_html(note)}</div>" if note else ""
    tone_cls = f" {safe_tone}" if safe_tone else ""
    return (
        f"<section class='operator-header-card{tone_cls}'>"
        f"<div class='operator-card-label'>{_escape_html(label)}</div>"
        f"<div class='operator-card-value{tone_cls}'>{_escape_html(value)}</div>"
        f"{note_html}"
        "</section>"
    )


def _operator_data_freshness_status(state: DashboardState, snapshot) -> tuple[str, str, str, str, str]:
    freshness_text = _relative_age(snapshot.timestamp_utc) if state.has_persisted_snapshot else "No data yet"
    freshness_note = _format_datetime_pretty(snapshot.timestamp_utc) if state.has_persisted_snapshot else "Waiting for first snapshot"
    freshness_tone = "warn" if "stale" in (state.feed_status or "").lower() else ("good" if state.has_persisted_snapshot else "warn")
    data_value = state.feed_status or "No data yet"
    data_note = "Waiting for first snapshot" if not state.has_persisted_snapshot else freshness_text
    live_health, live_note = _live_bot_process_health()
    if state.has_persisted_snapshot and live_health != "running":
        freshness_note = f"{live_note} Last saved snapshot: {freshness_note}"
        freshness_tone = "err"
        data_value = "No live bot heartbeat"
        data_note = f"{live_note} Last saved snapshot: {freshness_text}"
    return freshness_text, freshness_note, freshness_tone, data_value, data_note


def _open_order_count(recent_orders: list) -> int:
    active_statuses = {
        "new",
        "accepted",
        "pending_new",
        "accepted_for_bidding",
        "partially_filled",
        "pending_replace",
    }
    return sum(1 for order in recent_orders if str(getattr(order, "status", "")).lower() in active_statuses)


def _header_warning_summary(state: DashboardState) -> tuple[str, str]:
    if state.snapshot.kill_switch_triggered:
        return "Kill switch active", "err"
    if any(not check.allowed for check in state.latest_cycle_risk_checks):
        return "Risk blocks active", "warn"
    if state.session_warnings:
        return str(state.session_warnings[0]), "warn"
    feed_status = (state.feed_status or "").lower()
    if "stale" in feed_status:
        return "Feed stale", "warn"
    return "No active blocks", "good"


def _last_cycle_summary(
    last_cycle_report,
    latest_signal_rows: list,
    latest_cycle_risk_checks: list,
) -> tuple[str, str, str]:
    if last_cycle_report is None:
        return "No recent cycle", "Waiting for the first persisted cycle summary.", "warn"

    evaluated = (
        int(last_cycle_report.buy_signals or 0)
        + int(last_cycle_report.sell_signals or 0)
        + int(last_cycle_report.hold_signals or 0)
        + int(last_cycle_report.error_signals or 0)
    )
    error_count = int(last_cycle_report.error_signals or 0)
    blocked = sum(1 for row in latest_cycle_risk_checks if not row.allowed)
    rejected = sum(
        1
        for row in latest_signal_rows
        if row.decision_timestamp == last_cycle_report.decision_timestamp and getattr(row, "rejection", None)
    )

    if not last_cycle_report.processed_bar:
        return (
            "No new bar",
            _cycle_reason_label(last_cycle_report.skip_reason),
            "warn",
        )
    if int(last_cycle_report.orders_submitted or 0) > 0:
        return (
            f"{int(last_cycle_report.orders_submitted)} order(s) sent",
            f"{evaluated} evaluated · {blocked} blocked · {rejected} rejected · {error_count} errors",
            "good",
        )
    if error_count > 0:
        return (
            "Evaluation errors",
            f"{evaluated} evaluated · {error_count} errors",
            "err",
        )
    if blocked > 0:
        return (
            "Signals blocked",
            f"{evaluated} evaluated · {blocked} blocked · {rejected} rejected",
            "warn",
        )
    if int(last_cycle_report.buy_signals or 0) or int(last_cycle_report.sell_signals or 0):
        return (
            "Signals found, no orders",
            f"{evaluated} evaluated · {rejected} rejected",
            "warn",
        )
    return (
        "No action",
        f"{evaluated} evaluated · {rejected} rejected",
        "neutral",
    )


def _operator_header_html(state: DashboardState, snapshot, recent_orders: list) -> str:
    runtime_mode = _runtime_mode_label(state)
    approval_label, approval_level, approval_note = _runtime_approval_text(state)
    status_title, status_level, status_copy = _status_overview(state)
    warning_text, warning_level = _header_warning_summary(state)
    next_cycle_value, next_cycle_note = _next_cycle_text(state)
    cfg = state.startup_config

    session_label = cfg.session_id or "Unavailable" if cfg is not None else "Unavailable"
    strategy_label = (
        str(cfg.strategy_mode).replace("_", " ")
        if cfg is not None and cfg.strategy_mode
        else "Unknown"
    )
    execution_label = "Execution enabled" if cfg is not None and cfg.execution_enabled and runtime_mode != "Preview" else "Read only"
    execution_tone = "good" if execution_label == "Execution enabled" else "warn"

    freshness_text, freshness_note, freshness_tone, data_value, data_note = _operator_data_freshness_status(state, snapshot)
    process_value, process_note, process_tone = _process_health_summary()
    resync_value, resync_note, resync_tone = _resync_status_summary(state)
    freshness_label = state.feed_status or "Unknown"
    if data_value == "No live bot heartbeat":
        freshness_label = "No live bot heartbeat"

    open_positions = len(snapshot.positions)
    open_orders = _open_order_count(recent_orders)
    exposure_note = (
        f"Cash {_fmt_money(snapshot.cash)} · Buying power {_fmt_money(snapshot.buying_power)}"
    )
    exposure_value = f"Eq {_fmt_money(snapshot.equity)}"
    exposure_pnl = _fmt_money(snapshot.daily_pnl)
    exposure_note = f"{exposure_note} · P/L {exposure_pnl}"
    account_tone = "good" if snapshot.daily_pnl >= 0 else "err"

    cycle_value, cycle_note, cycle_tone = _last_cycle_summary(
        state.last_cycle_report,
        list(state.latest_signal_rows),
        list(state.latest_cycle_risk_checks),
    )

    chips = "".join(
        [
            _operator_chip_html(runtime_mode, "good" if runtime_mode in {"Paper", "Live"} else "warn"),
            _operator_chip_html(approval_label, approval_level),
            _operator_chip_html(execution_label, execution_tone),
            _operator_chip_html(status_title, status_level),
        ]
    )

    mini_cards = "".join(
        [
            (
                "<div class='operator-mini'>"
                "<div class='operator-mini-label'>Strategy</div>"
                f"<div class='operator-mini-value'>{_escape_html(strategy_label.upper())}</div>"
                f"<div class='operator-mini-note'>{_escape_html(status_copy)}</div>"
                "</div>"
            ),
            (
                "<div class='operator-mini'>"
                "<div class='operator-mini-label'>Session</div>"
                f"<div class='operator-mini-value'>{_escape_html(session_label)}</div>"
                f"<div class='operator-mini-note'>{_escape_html(approval_note)}</div>"
                "</div>"
            ),
            (
                "<div class='operator-mini'>"
                "<div class='operator-mini-label'>Last Update</div>"
                f"<div class='operator-mini-value'>{_escape_html(freshness_text)}</div>"
                f"<div class='operator-mini-note'>{_escape_html(freshness_note)}</div>"
                "</div>"
            ),
            (
                "<div class='operator-mini'>"
                "<div class='operator-mini-label'>Exposure</div>"
                f"<div class='operator-mini-value'>{open_positions} pos · {open_orders} orders</div>"
                f"<div class='operator-mini-note'>{_escape_html(exposure_note)}</div>"
                "</div>"
            ),
        ]
    )

    return (
        f"<section class='operator-header {status_level if status_level in {'warn', 'err'} else ''}'>"
        f"<div class='operator-header-main {status_level if status_level in {'warn', 'err'} else ''}'>"
        "<div class='operator-header-top'>"
        "<div class='operator-title-wrap'>"
        "<div class='operator-kicker'>Operator Header</div>"
        "<div class='operator-title'>TradeOS Live Monitor</div>"
        f"<div class='operator-subtitle'>{_escape_html(status_copy)}</div>"
        "</div>"
        f"<div class='operator-chip-row'>{chips}</div>"
        "</div>"
        f"<div class='operator-main-grid'>{mini_cards}</div>"
        "</div>"
        f"{_operator_card_html('Safety', warning_text, 'Kill switch, stale data, and blocking conditions.', warning_level)}"
        f"{_operator_card_html('Freshness', freshness_label, freshness_note, freshness_tone)}"
        f"{_operator_card_html('Next Cycle', next_cycle_value, next_cycle_note, 'err' if next_cycle_value == 'No live bot heartbeat' else ('warn' if next_cycle_value == 'Overdue' else 'neutral'))}"
        f"{_operator_card_html('Account', exposure_value, exposure_note, account_tone)}"
        f"{_operator_card_html('Last Cycle', cycle_value, cycle_note, cycle_tone)}"
        "</section>"
    )


# â”€â”€ v2 HTML builders â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def _build_operator_header_summary(state: DashboardState, snapshot, recent_orders: list) -> dict[str, object]:
    runtime_mode = _runtime_mode_label(state)
    approval_label, approval_level, _approval_note = _runtime_approval_text(state)
    status_title, status_level, _status_copy = _status_overview(state)
    warning_text, warning_level = _header_warning_summary(state)
    next_cycle_value, next_cycle_note = _next_cycle_text(state)
    cfg = state.startup_config
    session_label = cfg.session_id or "Unavailable" if cfg is not None else "Unavailable"
    strategy_label = (
        str(cfg.strategy_mode).replace("_", " ")
        if cfg is not None and cfg.strategy_mode
        else "Unknown"
    )
    execution_label = "Execution enabled" if cfg is not None and cfg.execution_enabled and runtime_mode != "Preview" else "Read only"
    execution_tone = "good" if execution_label == "Execution enabled" else "warn"
    freshness_text, freshness_note, freshness_tone, data_value, data_note = _operator_data_freshness_status(state, snapshot)
    process_value, process_note, process_tone = _process_health_summary()
    resync_value, resync_note, resync_tone = _resync_status_summary(state)
    open_positions = len(snapshot.positions)
    open_orders = _open_order_count(recent_orders)
    account_value = _fmt_money(snapshot.equity)
    account_tone = "good" if snapshot.daily_pnl >= 0 else "err"
    cycle_value, cycle_note, cycle_tone = _last_cycle_summary(
        state.last_cycle_report,
        list(state.latest_signal_rows),
        list(state.latest_cycle_risk_checks),
    )
    chips: list[tuple[str, str]] = [
        (runtime_mode.upper(), "good" if runtime_mode in {"Paper", "Live"} else "warn"),
        (approval_label.replace(" Runtime", "").upper(), approval_level),
        (execution_label.upper(), execution_tone),
        (resync_value.upper(), resync_tone),
        (process_value.upper(), process_tone),
    ]
    if status_level in {"warn", "err"}:
        chips.append((status_title.upper(), status_level))
    return {
        "title": "TradeOS Live Monitor",
        "chips": chips,
        "last_updated_value": freshness_text,
        "last_updated_note": freshness_note,
        "metadata": [
            ("Strategy", strategy_label.upper()),
            ("Session", session_label),
            ("Exposure", f"{open_positions} positions · {open_orders} orders"),
        ],
        "status_cards": [
            ("Safety", "System healthy" if warning_text == "No active blocks" else warning_text, "No active blocks" if warning_text == "No active blocks" else "Review control state", warning_level),
            ("Data", data_value, data_note, freshness_tone),
            ("Runtime", process_value, process_note, process_tone),
            ("Resync", resync_value, resync_note, resync_tone),
            ("Next Cycle", next_cycle_value, "First cycle not completed" if next_cycle_value == "Waiting" else next_cycle_note, "err" if next_cycle_value == "No live bot heartbeat" else ("warn" if next_cycle_value == "Overdue" else "neutral")),
            ("Account", account_value, f"{open_positions} positions · {open_orders} orders", account_tone),
            ("Last Cycle", cycle_value, "Waiting for first cycle" if state.last_cycle_report is None else cycle_note, cycle_tone),
        ],
    }


def _render_operator_control_strip(summary: dict[str, object]) -> tuple[bool, bool]:
    left_col, meta_col, toggle_col, action_col = st.columns([7.2, 2.8, 1.8, 1.4])
    chips = "".join(_operator_chip_html(label, tone) for label, tone in summary["chips"])  # type: ignore[index]
    with left_col:
        st.markdown(
            "<div class='operator-control-shell'>"
            "<div class='operator-control-main'>"
            f"<div class='operator-control-title tradeos-page-title'>{_escape_html(summary['title'])}</div>"
            f"<div class='operator-chip-row'>{chips}</div>"
            "</div>"
            "</div>",
            unsafe_allow_html=True,
        )
    with meta_col:
        st.markdown(
            "<div class='operator-control-shell' style='justify-content:flex-end;border-bottom:none;margin-bottom:0;padding-bottom:0;'>"
            "<div class='operator-meta-item'>"
            "<div class='tradeos-section-label'>Last Updated</div>"
            f"<div class='tradeos-secondary-value'>{_escape_html(summary['last_updated_value'])}</div>"
            f"<div class='tradeos-meta-text'>{summary['last_updated_note']}</div>"
            "</div>"
            "</div>",
            unsafe_allow_html=True,
        )
    with toggle_col:
        auto_refresh = st.checkbox(
            "Auto refresh",
            value=st.session_state.get("operator_auto_refresh", True),
            key="operator_auto_refresh",
        )
    with action_col:
        refresh = st.button("Refresh", use_container_width=True, type="secondary")
    return refresh, auto_refresh


def _render_operator_metadata_grid(summary: dict[str, object]) -> None:
    items_html = "".join(
        (
            "<div class='operator-meta-item'>"
            f"<div class='tradeos-section-label'>{_escape_html(label)}</div>"
            f"<div class='tradeos-secondary-value'>{_escape_html(value)}</div>"
            "</div>"
        )
        for label, value in summary["metadata"]  # type: ignore[index]
    )
    st.markdown(f"<div class='operator-meta-grid'>{items_html}</div>", unsafe_allow_html=True)


def _render_operator_status_cards(summary: dict[str, object]) -> None:
    cards_html = "".join(
        (
            f"<section class='operator-status-card{(' ' + tone) if tone in {'warn', 'err'} else ''}'>"
            f"<div class='tradeos-section-label'>{_escape_html(label)}</div>"
            f"<div class='tradeos-primary-value'>{_escape_html(value)}</div>"
            f"<div class='tradeos-body-text'>{_escape_html(note)}</div>"
            "</section>"
        )
        for label, value, note, tone in summary["status_cards"]  # type: ignore[index]
    )
    st.markdown(f"<div class='operator-status-grid'>{cards_html}</div>", unsafe_allow_html=True)


def _render_operator_header(state: DashboardState, snapshot, recent_orders: list) -> bool:
    summary = _build_operator_header_summary(state, snapshot, recent_orders)
    st.markdown("<section class='operator-shell'>", unsafe_allow_html=True)
    refresh, auto_refresh = _render_operator_control_strip(summary)
    _render_operator_metadata_grid(summary)
    _render_operator_status_cards(summary)
    st.markdown("</section>", unsafe_allow_html=True)
    if auto_refresh:
        components.html(
            """
            <script>
            window.setTimeout(() => window.parent.location.reload(), 15000);
            </script>
            """,
            height=0,
        )
    return refresh


def _metrics_bar_html(snapshot) -> str:
    pnl_cls = _pnl_cls(snapshot.daily_pnl)
    action_counts: dict[str, int] = {}
    for item in snapshot.symbols:
        action_counts[item.action] = action_counts.get(item.action, 0) + 1
    open_pos = len(snapshot.positions)
    decision_str = f"{action_counts.get('BUY', 0)} buy · {action_counts.get('SELL', 0)} sell"
    snapshot_only_count = action_counts.get("SNAPSHOT_ONLY", 0)
    if snapshot_only_count == len(snapshot.symbols) and snapshot.symbols:
        decision_str = "snapshot only"
    last_eq = snapshot.last_equity if snapshot.last_equity else 0.0

    def _cell(label: str, value: str, delta: str = "", val_cls: str = "") -> str:
        delta_html = f"<div style='font-size:12px;color:#8a9bb0;margin-top:3px'>{delta}</div>" if delta else ""
        return (
            f"<div class='dash-metric'>"
            f"<div class='dash-metric-label'>{label}</div>"
            f"<div class='dash-metric-val {val_cls}'>{value}</div>"
            f"{delta_html}</div>"
        )

    return (
        "<div class='dash-metrics'>"
        + _cell("Cash Available", _fmt_money(snapshot.cash))
        + _cell("Buying Power", _fmt_money(snapshot.buying_power), "ready to deploy")
        + _cell("Account Value", _fmt_money(snapshot.equity), f"prev close {_fmt_money(last_eq)}")
        + _cell(
            "Today's P/L",
            _fmt_money(snapshot.daily_pnl),
            "Kill switch active" if snapshot.kill_switch_triggered else "Within limit",
            pnl_cls,
        )
        + _cell("Positions Open", f"{open_pos}", decision_str)
        + "</div>"
    )


def _hero_stat(label: str, value: str) -> str:
    return (
        "<div class='hero-stat'>"
        f"<div class='hero-stat-label'>{label}</div>"
        f"<div class='hero-stat-value'>{value}</div>"
        "</div>"
    )


def _section_intro_html(kicker: str, title: str, body: str) -> str:
    return (
        "<section class='section-intro'>"
        f"<div class='section-intro-kicker'>{_escape_html(kicker)}</div>"
        f"<div class='section-intro-title'>{_escape_html(title)}</div>"
        f"<div class='section-intro-body'>{_escape_html(body)}</div>"
        "</section>"
    )


def _workspace_card_html(kicker: str, title: str, copy: str, meta: str) -> str:
    return (
        "<section class='workspace-map-card'>"
        f"<div class='workspace-map-kicker'>{_escape_html(kicker)}</div>"
        f"<div class='workspace-map-title'>{_escape_html(title)}</div>"
        f"<div class='workspace-map-copy'>{_escape_html(copy)}</div>"
        f"<div class='workspace-map-meta'>{_escape_html(meta)}</div>"
        "</section>"
    )


def _workspace_map_html(state: DashboardState, snapshot) -> str:
    drilldown_count = len(list(state.drilldown_candidates))
    symbol_count = len(snapshot.symbols)
    config_status = "Ready" if state.has_persisted_startup_config else "Missing"
    return (
        "<div class='workspace-map'>"
        + _workspace_card_html(
            "Start Here",
            "Live Desk",
            "Scan the watchlist, focus one symbol, and understand what the bot sees right now.",
            f"{symbol_count} symbols in rotation",
        )
        + _workspace_card_html(
            "Look Back",
            "Performance",
            "Review equity behavior and saved symbol history when you want patterns instead of snapshots.",
            "Trend charts plus recent saved states",
        )
        + _workspace_card_html(
            "Sanity Check",
            "Setup",
            "Confirm runtime mode, guardrails, and model thresholds without reading raw config files.",
            f"Startup config: {config_status}",
        )
        + _workspace_card_html(
            "Explain It",
            "Decision Log",
            "Trace a recent decision event from signal context through execution outcome.",
            f"{drilldown_count} recent decision events",
        )
        + "</div>"
    )


def _status_panel_metric_html(label: str, value: str, note: str = "") -> str:
    note_html = f"<div class='status-metric-note'>{_escape_html(note)}</div>" if note else ""
    return (
        "<div class='status-panel'>"
        f"<div class='status-metric-label'>{_escape_html(label)}</div>"
        f"<div class='status-metric-value'>{_escape_html(value)}</div>"
        f"{note_html}"
        "</div>"
    )


def _runtime_mode_label(state: DashboardState) -> str:
    cfg = state.startup_config
    if cfg is None:
        return "Unknown"
    launch_mode = str(cfg.launch_mode or "").strip().lower()
    if launch_mode == "preview":
        return "Preview"
    if cfg.paper:
        return "Paper"
    return "Live"


def _runtime_approval_text(state: DashboardState) -> tuple[str, str, str]:
    cfg = state.startup_config
    if cfg is None:
        return "Approval Unknown", "warn", "No persisted startup metadata is available for this session."
    if cfg.runtime_config_approved is True:
        return "Approved Runtime", "good", "The persisted startup artifact marks this runtime as approved."
    if cfg.runtime_config_approved is False:
        reasons = "; ".join(cfg.runtime_config_rejection_reasons)
        detail = reasons if reasons else "No rejection reasons were saved."
        return "Unapproved Runtime", "err", detail
    return "Approval Unknown", "warn", "This startup artifact does not record a runtime approval decision."


def _next_cycle_text(state: DashboardState) -> tuple[str, str]:
    if _is_past_session_flatten_deadline():
        return (
            "Closed",
            "The live bot exits after the 3:55 PM ET flatten deadline and resumes next session.",
        )

    cfg = state.startup_config
    timeframe_minutes = (
        int(getattr(cfg, "bar_timeframe_minutes", 15) or 15)
        if cfg is not None else 15
    )
    reference_value = None
    if state.last_cycle_report is not None and state.last_cycle_report.decision_timestamp:
        reference_value = state.last_cycle_report.decision_timestamp
    elif state.has_persisted_snapshot:
        snapshot_ts = getattr(state.snapshot, "timestamp_utc", None)
        if snapshot_ts:
            reference_value = snapshot_ts
    if not reference_value:
        return "Waiting", "The next decision time will appear after the first saved cycle."

    reference_dt = _parse_datetime(reference_value)
    if reference_dt is None:
        return "Unknown", "The persisted timestamp could not be parsed safely."
    next_dt = reference_dt + timedelta(minutes=timeframe_minutes)
    now_utc = datetime.now(timezone.utc)
    delta_seconds = int((next_dt - now_utc).total_seconds())
    if delta_seconds >= 0:
        minutes, seconds = divmod(delta_seconds, 60)
        return (
            next_dt.astimezone(_ET).strftime("%I:%M:%S %p ET"),
            f"About {minutes}m {seconds:02d}s until the next scheduled bar close.",
        )
    overdue_seconds = abs(delta_seconds)
    minutes, seconds = divmod(overdue_seconds, 60)
    live_health, live_note = _live_bot_process_health()
    if live_health != "running":
        return (
            "No live bot heartbeat",
            f"{live_note} The last saved cycle is overdue by {minutes}m {seconds:02d}s.",
        )
    return (
        "Overdue",
        f"The next cycle would normally have arrived {minutes}m {seconds:02d}s ago.",
    )


def _is_past_session_flatten_deadline(now_utc: datetime | None = None) -> bool:
    current_utc = now_utc or datetime.now(timezone.utc)
    current_et = current_utc.astimezone(_ET)
    return current_et.time() >= _SESSION_FLATTEN_AT


def _status_overview(state: DashboardState) -> tuple[str, str, str]:
    if _is_past_session_flatten_deadline():
        freshness = _relative_age(state.snapshot.timestamp_utc) if state.has_persisted_snapshot else "n/a"
        return (
            "Closed",
            "primary",
            f"The trading session is past the 3:55 PM ET flatten deadline. Last saved snapshot is {freshness}.",
        )

    cfg = state.startup_config
    if cfg is None and not state.has_persisted_snapshot:
        return (
            "Unavailable",
            "err",
            "No recent session data is available yet. Start a TradeOS session or wait for persisted startup metadata.",
        )
    if cfg is not None and state.last_cycle_report is None and not state.has_persisted_snapshot:
        return (
            "Initializing",
            "warn",
            "The session has startup metadata, but no saved decision cycle has been recorded yet.",
        )
    if cfg is not None and str(cfg.resync_status or "").upper() == "RESYNC_FAILED":
        return (
            "Resync Failed",
            "err",
            "Startup reconciliation did not complete safely. The dashboard may still show persisted state, but trading should remain blocked.",
        )
    if cfg is not None and str(cfg.resync_status or "").upper() == "RESYNC_DEGRADED":
        return (
            "Degraded",
            "warn",
            "Startup reconciliation recovered broker exposure, but the runtime should remain exits-only until the discrepancy is resolved.",
        )
    if cfg is not None and str(cfg.resync_status or "").upper() == "RESYNC_LOCKED":
        return (
            "Resync Locked",
            "warn",
            "Startup reconciliation is still in progress, so the runtime should not place fresh entries yet.",
        )

    freshness = _relative_age(state.snapshot.timestamp_utc) if state.has_persisted_snapshot else "n/a"
    feed_status = (state.feed_status or "").lower()
    live_health, live_note = _live_bot_process_health()
    if state.has_persisted_snapshot and live_health != "running":
        return (
            "Live Bot Offline",
            "err",
            f"{live_note} The dashboard is showing persisted data from {freshness}.",
        )
    if "stale" in feed_status:
        return (
            "Stale",
            "warn",
            f"Persisted data looks behind the expected bar rhythm. Last saved snapshot is {freshness}.",
        )
    if state.last_cycle_report is not None and not state.last_cycle_report.processed_bar:
        return (
            "Waiting",
            "primary",
            f"The latest scheduler event did not process a new bar ({_cycle_reason_label(state.last_cycle_report.skip_reason).lower()}).",
        )
    if state.has_persisted_snapshot:
        return (
            "Running",
            "primary",
            f"Recent session data is available and the last saved snapshot is {freshness}.",
        )
    return (
        "Waiting",
        "warn",
        "TradeOS has session metadata, but it has not saved a snapshot for this session yet.",
    )


def _global_status_bar_html(state: DashboardState, snapshot) -> str:
    status_title, status_level, status_copy = _status_overview(state)
    position_count = len(snapshot.positions)
    symbol_count = len(state.startup_config.symbols) if state.startup_config is not None else len(snapshot.symbols)
    next_cycle_value, next_cycle_note = _next_cycle_text(state)
    approval_label, approval_level, approval_note = _runtime_approval_text(state)
    runtime_mode = _runtime_mode_label(state)
    active_config = (
        state.startup_config.strategy_mode
        if state.startup_config is not None else
        "No persisted runtime"
    )
    active_note = (
        "Currently active session metadata."
        if state.startup_config is not None else
        "The dashboard is showing only whatever persisted state is available."
    )
    position_value = "Open" if position_count else "Flat"
    position_note = (
        f"{position_count} open positions are reflected in the latest persisted snapshot."
        if position_count else
        "The latest persisted snapshot shows no open positions."
    )
    countdown_note = next_cycle_note
    approval_html = (
        "<div class='approval-chip-row'>"
        f"<span class='approval-chip {approval_level}'>{_escape_html(approval_label)}</span>"
        f"<span class='approval-chip'>{_escape_html(runtime_mode)}</span>"
        "</div>"
    )
    return (
        "<div class='status-strip'>"
        f"<div class='status-panel {status_level}'>"
        "<div class='status-panel-kicker'>Global Status</div>"
        f"<div class='status-panel-title'>{_escape_html(status_title)}</div>"
        f"<div class='status-panel-copy'>{_escape_html(status_copy)}</div>"
        f"{approval_html}"
        f"<div class='status-metric-note'>{_escape_html(approval_note)}</div>"
        "</div>"
        f"{_status_panel_metric_html('Mode', runtime_mode, 'Preview disables orders; paper/live reflects the account context saved at startup.')}"
        f"{_status_panel_metric_html('Watching', str(symbol_count), 'Symbols from the active startup config when available.')}"
        f"{_status_panel_metric_html('Positions', position_value, position_note)}"
        f"{_status_panel_metric_html('Next Bar', next_cycle_value, countdown_note)}"
        f"{_status_panel_metric_html('Active Config', active_config, active_note)}"
        "</div>"
    )


def _header_summary_html(state: DashboardState, snapshot, last_cycle_report) -> str:
    processed_text = "Waiting for first decision cycle"
    if last_cycle_report is not None and last_cycle_report.processed_bar:
        processed_text = _compact_time_str(last_cycle_report.decision_timestamp)
    elif last_cycle_report is not None:
        processed_text = _cycle_reason_label(last_cycle_report.skip_reason)

    symbol_count = len(state.startup_config.symbols) if state.startup_config is not None else len(snapshot.symbols)
    mode_label = (
        state.startup_config.strategy_mode
        if state.startup_config is not None
        else ("snapshot only" if state.has_persisted_snapshot else "config pending")
    )
    position_count = len(snapshot.positions)
    note = (
        "A calm control surface for the live bot: quick status up top, recent movement in the middle, and deep symbol context when you need it."
    )
    if state.startup_config is None:
        mode_badge = "<span class='badge b-nosignal'>CONFIG MISSING</span>"
    elif str(state.startup_config.launch_mode or "").strip().lower() == "preview":
        mode_badge = "<span class='badge b-nosignal'>PREVIEW</span>"
    elif state.startup_config.paper:
        mode_badge = "<span class='badge b-hold'>PAPER</span>"
    else:
        mode_badge = "<span class='badge b-sell'>LIVE</span>"
    return (
        "<div class='hero-card'>"
        "<div class='hero-kicker'>Live Bot Overview</div>"
        "<div class='top-title'>TradeOS</div>"
        f"<div class='top-meta'>{mode_badge}"
        f"<span class='subtle-copy'>{symbol_count} symbols · {mode_label}</span></div>"
        f"<div class='hero-summary'>{_hero_stat('Last Update', processed_text)}{_hero_stat('Positions', str(position_count))}{_hero_stat('Snapshot Age', _relative_age(snapshot.timestamp_utc))}</div>"
        f"<div class='hero-note'>{note}</div>"
        "</div>"
    )


def _sym_card_v2_html(
    item, pos_row: dict | None, is_selected: bool, is_error: bool, preview=None, *, ml_enabled: bool = True
) -> str:
    action = (item.action or "HOLD").upper()
    badge = _execution_status_badge(preview.status) if preview is not None else _badge_html(action)

    if is_error:
        extra_cls = "sym-error"
    elif is_selected:
        extra_cls = "sym-active"
    else:
        extra_cls = ""

    state_cls = {"BUY": "sym-s-buy", "SELL": "sym-s-sell"}.get(action, "")
    price_str = f"${item.price:,.2f}" if item.price is not None else "â€”"
    change_str = "n/a"
    if item.sma is not None and item.price is not None and item.sma != 0:
        delta = (item.price - item.sma) / item.sma * 100
        sign = "+" if delta >= 0 else ""
        change_str = f"{sign}{delta:.2f}%"

    if pos_row is not None:
        mv = _fmt_money(float(pos_row.get("market_value") or 0.0))
        qty = f"{float(pos_row.get('qty') or 0.0):.2f} sh"
        meta_right = f"<span style='color:#86efac'>{mv} · {qty}</span>"
    else:
        meta_right = "<span class='tiny-label'>Flat</span>"

    mini_bar = _ml_mini_bar_html(
        item.ml_probability_up if ml_enabled else None,
        getattr(item, "ml_buy_threshold", None),
        getattr(item, "ml_sell_threshold", None),
    )

    cls = " ".join(c for c in ("sym-card-v2", extra_cls, state_cls) if c)
    return (
        f"<div class='{cls}'>"
        f"<div class='sym-card-v2-top'>"
        f"<span class='sym-ticker'>{item.symbol}</span>"
        f"{badge}"
        f"</div>"
        f"<div class='sym-card-v2-meta'>"
        f"<span class='sym-price'>{price_str}</span>"
        f"<span class='sym-change'>{change_str}</span>"
        f"</div>"
        f"<div class='sym-card-foot'><span class='tiny-label'>vs sma</span>{meta_right}</div>"
        f"{mini_bar}"
        f"</div>"
    )


def _detail_info_cell(label: str, value: str, val_cls: str = "") -> str:
    return (
        f"<div class='detail-info-cell'>"
        f"<div class='detail-info-cell-label'>{label}</div>"
        f"<div class='detail-info-cell-val {val_cls}'>{value}</div>"
        f"</div>"
    )


def _status_card_html(level: str, title: str, body: str) -> str:
    safe_level = level if level in {"warn", "err", "info"} else "info"
    return (
        f"<div class='status-card {safe_level}'>"
        f"<div class='status-card-dot'></div>"
        f"<div>"
        f"<div class='status-card-title'>{title}</div>"
        f"<div class='status-card-body'>{body}</div>"
        f"</div>"
        f"</div>"
    )


def _drift_summary_html(report) -> str:
    if report is None:
        return ""

    baseline = report.baseline
    live = report.live
    if report.alerts:
        cards = "".join(
            _status_card_html(
                alert.severity,
                alert.summary,
                f"Observed: {alert.observed}. Expected: {alert.expected}. {alert.why_it_matters}",
            )
            for alert in report.alerts[:4]
        )
        status_line = f"{len(report.alerts)} active drift alert(s) in {live.window_label}."
    elif baseline.valid_for_comparison:
        cards = _status_card_html(
            "info",
            "Live behavior is within the validated profile",
            "Current session metrics are staying within the operator drift thresholds sourced from the validated research profile.",
        )
        status_line = f"No active drift alerts in {live.window_label}."
    else:
        detail = "; ".join(baseline.validation_errors) or "validated baseline is unavailable"
        cards = _status_card_html(
            "warn",
            "Drift comparison is running with an incomplete baseline",
            detail,
        )
        status_line = "Baseline needs attention before drift comparisons are fully trustworthy."

    metric_bits = []
    if live.buy_signal_rate_per_symbol_per_day is not None:
        metric_bits.append(f"Buy rate {live.buy_signal_rate_per_symbol_per_day:.2f}/sym/day")
    if live.rejection_rate is not None:
        metric_bits.append(f"Rejections {live.rejection_rate:.0%}")
    if live.avg_bar_age_s is not None:
        metric_bits.append(f"Avg bar age {live.avg_bar_age_s:.0f}s")

    metric_text = " | ".join(metric_bits) if metric_bits else "Metrics will populate after more persisted session activity."
    return (
        "<div class='detail-history-card'>"
        "<div class='sec-head'>Behavior Drift</div>"
        f"<div class='subtle-copy' style='margin-bottom:0.7rem'>{status_line}</div>"
        f"<div class='subtle-copy' style='margin-bottom:0.9rem'>Baseline source: {_escape_html(baseline.source_label)}. {metric_text}</div>"
        f"{cards}"
        "</div>"
    )


def _feed_health_html(feed_status: str, dot_cls: str, snapshot, timeframe_minutes: int) -> str:
    freshness = _relative_age(snapshot.timestamp_utc)
    symbol_count = len(snapshot.symbols)
    position_count = len(snapshot.positions)
    return (
        "<div class='compact-card feed-card'>"
        "<div class='feed-grid'>"
        "<div class='feed-primary'>"
        f"<div class='feed-title'><span class='dot {dot_cls}'></span><span>Bot status</span></div>"
        f"<div class='feed-subline'>{feed_status}. This screen follows the live bot and helps you understand what it is seeing.</div>"
        "</div>"
        "<div class='feed-stat'>"
        "<div class='feed-stat-label'>How fresh</div>"
        f"<div class='feed-stat-value'>{freshness}</div>"
        "</div>"
        "<div class='feed-stat'>"
        "<div class='feed-stat-label'>Update rhythm</div>"
        f"<div class='feed-stat-value'>{timeframe_minutes} min</div>"
        "</div>"
        "<div class='feed-stat'>"
        "<div class='feed-stat-label'>Watching</div>"
        f"<div class='feed-stat-value'>{symbol_count} syms / {position_count} open</div>"
        "</div>"
        "</div>"
        "</div>"
    )


def _trend_strip_html(sym_history: pd.DataFrame) -> str:
    if sym_history.empty or "price" not in sym_history.columns:
        return ""
    prices = pd.to_numeric(sym_history["price"], errors="coerce").dropna().tail(18)
    if len(prices) < 2:
        return ""

    low = float(prices.min())
    high = float(prices.max())
    span = max(high - low, 1e-9)
    prev = None
    bars: list[str] = []
    for price in prices:
        height_pct = 24 + ((float(price) - low) / span) * 76
        if prev is None:
            color = "#60a5fa"
        elif float(price) >= prev:
            color = "#4ade80"
        else:
            color = "#f87171"
        bars.append(
            f"<div class='trend-bar' style='height:{height_pct:.1f}%;background:{color}' "
            f"title='{float(price):.2f}'></div>"
        )
        prev = float(price)
    return (
        "<div>"
        "<div class='tiny-label'>Recent Trend</div>"
        f"<div class='trend-strip'>{''.join(bars)}</div>"
        "</div>"
    )


def _fmt_pct(v: float | None) -> str:
    if v is None:
        return "n/a"
    sign = "+" if v >= 0 else ""
    return f"{sign}{v:.2f}%"


def _timestamp_html(primary: str, secondary: str | None = None) -> str:
    primary_html = html.escape(primary)
    if not secondary:
        return primary_html
    secondary_html = html.escape(secondary)
    return f"{primary_html}<br><span style='opacity:0.6;font-size:0.88em'>{secondary_html}</span>"


def _compact_time_str(value: str | None) -> str:
    if not value:
        return "â€”"
    try:
        dt = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        if dt.tzinfo is not None:
            et = dt.astimezone(_ET)
            local = dt.astimezone()
            et_str = et.strftime("%I:%M %p ET").lstrip("0")
            local_str = local.strftime("%I:%M %p").lstrip("0") + " local"
            if et.hour == local.hour and et.minute == local.minute:
                return html.escape(et_str)
            return _timestamp_html(et_str, local_str)
        return html.escape(dt.strftime("%I:%M %p").lstrip("0"))
    except ValueError:
        fallback = str(value)[11:16] if len(str(value)) >= 16 else str(value)
        return html.escape(fallback)


def _parse_datetime(value) -> datetime | None:
    if value in (None, ""):
        return None
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None


def _format_datetime_pretty(value, include_time: bool = True) -> str:
    parsed = _parse_datetime(value)
    if parsed is not None:
        if parsed.tzinfo is not None and include_time:
            et = parsed.astimezone(_ET)
            local = parsed.astimezone()
            et_str = et.strftime("%b %d, %Y · %I:%M %p ET").replace(" 0", " ")
            local_str = local.strftime("%I:%M %p").lstrip("0") + " local"
            if et.hour == local.hour and et.minute == local.minute:
                return html.escape(et_str)
            return _timestamp_html(et_str, local_str)
        fmt = "%b %d, %Y · %I:%M %p ET" if include_time else "%b %d, %Y"
        return html.escape(parsed.strftime(fmt).replace(" 0", " "))
    if value in (None, ""):
        return "â€”"
    return html.escape(str(value))


def _pretty_reason(reason: str | None) -> str:
    if not reason:
        return "n/a"
    raw = str(reason).strip()
    if "|" in raw:
        parts = [_pretty_reason(part) for part in raw.split("|") if str(part).strip()]
        return " + ".join(parts) if parts else "n/a"
    mapping = {
        "no_signal": "no actionable signal",
        "trend_filter": "filtered by trend filter",
        "atr_filter": "filtered by ATR filter",
        "outside_entry_window": "outside entry window",
        "max_open_positions_reached": "blocked by max positions",
        "stale_live_price": "blocked by stale live price",
        "insufficient_buying_power": "blocked by buying power",
        "price_collar_breached": "blocked by price collar",
    }
    key = raw.lower().replace(" ", "_")
    return mapping.get(key, raw.replace("_", " "))


def _logic_value_html(text: str, state: str = "neutral") -> str:
    cls = {
        "yes": "logic-val-yes",
        "no": "logic-val-no",
        "neutral": "logic-val-neutral",
    }.get(state, "logic-val-neutral")
    return f"<span class='{cls}'>{text}</span>"


def _logic_line(label: str, value_html: str, note: str = "") -> str:
    note_html = f"<div class='logic-note'>{html.escape(note)}</div>" if note else ""
    return (
        "<div class='logic-item'>"
        "<span class='logic-bullet'>&bull;</span>"
        f"<span class='logic-key'>{label}</span>"
        "<div class='logic-body'>"
        f"<div class='logic-main'>{value_html}</div>"
        f"{note_html}"
        "</div>"
        "</div>"
    )


def _ticker_strip_html(snapshot, latest_signals: dict, session_first_prices: dict | None = None, selected_symbol: str | None = None) -> str:
    chips: list[str] = []
    first_prices = session_first_prices or {}
    for item in snapshot.symbols:
        signal_row = latest_signals.get(item.symbol)
        price_str = f"${item.price:,.2f}" if item.price is not None else "&mdash;"
        # Prefer intra-session % change computed from first recorded bar price.
        # Falls back to SMA20 deviation when no session baseline is available.
        first_price = first_prices.get(item.symbol)
        if first_price is not None and item.price is not None and first_price != 0:
            change_val = (item.price - first_price) / first_price * 100
            change_label = "session"
        else:
            change_val = getattr(signal_row, "deviation_pct", None)
            change_label = "vs SMA"
        color = "#93a5bc"
        if change_val is not None:
            color = "#4ade80" if change_val > 0 else ("#f87171" if change_val < 0 else "#93a5bc")
        change_html = (
            f"{_fmt_pct(change_val)}"
            f"<span class='ticker-chip-note'>{change_label}</span>"
            if change_val is not None else "&mdash;"
        )
        chip_cls = "ticker-chip active" if item.symbol == selected_symbol else "ticker-chip"
        chips.append(
            f"<div class='{chip_cls}'>"
            f"<span class='ticker-chip-symbol'>{item.symbol}</span>"
            f"<span class='ticker-chip-price'>{price_str}</span>"
            f"<span class='ticker-chip-change' style='color:{color}'>{change_html}</span>"
            "</div>"
        )
    data_note = "Session move from first recorded bar" if first_prices else "Latest move vs SMA20 because no session baseline is available yet"
    return (
        "<div class='ticker-strip'>"
        "<div class='ticker-strip-head'>"
        "<div class='workspace-label' style='margin-bottom:0'>Market snapshot</div>"
        f"<div class='subtle-copy'>Latest bar-close prices &middot; {data_note}</div>"
        "</div>"
        f"<div class='ticker-strip-ribbon'>{''.join(chips)}</div>"
        "</div>"
    )


def _execution_panel_html(recent_execution_activity: list) -> str:
    if not recent_execution_activity:
        return (
            "<div class='exec-panel'>"
            "<div class='sec-head' style='margin-top:0;margin-bottom:0.6rem'>What Changed Recently</div>"
            "<div class='rail-empty'>Waiting for recent execution-side events. The bot may still be warming up, between bars, or simply not taking action yet.</div>"
            "</div>"
        )
    rows: list[str] = []
    for item in recent_execution_activity[:5]:
        time_str = _compact_time_str(item.observed_at_utc)
        qty_str = f"{float(item.qty):.2f} sh" if item.qty is not None else "â€”"
        price_str = f"${float(item.price):,.2f}" if item.price is not None else "â€”"
        reason_text = _pretty_reason(item.reason)
        side = (item.side or "HOLD").upper()
        row_cls = side.lower()
        action_cls = row_cls
        type_badge = ""
        if side == "HOLD":
            if "filtered" in reason_text:
                row_cls = "hold-filtered"
                type_badge = "<span class='exec-type-badge filtered'>Filtered</span>"
            elif "rejected" in reason_text:
                row_cls = "hold-rejected"
                type_badge = "<span class='exec-type-badge rejected'>Rejected</span>"
            else:
                row_cls = "hold-no-signal"
                type_badge = "<span class='exec-type-badge no-signal'>No Signal</span>"
            action_cls = "hold"
        elif "blocked" in reason_text:
            row_cls = "blocked"
            action_cls = "blocked"
            type_badge = "<span class='exec-type-badge blocked'>Blocked</span>"

        action_display = side
        if row_cls == "blocked" and side in {"BUY", "SELL"}:
            action_display = side

        source_label = {
            "order.submitted": "submitted",
            "order.filled": "filled",
            "order.partial_fill": "partial fill",
            "order.rejected": "rejected",
            "risk.check": "risk check",
            "signal.evaluated": "signal",
        }.get(item.source_event or "", "")
        source_html = (
            f"<span style='font-size:0.74rem;color:#546778;margin-right:5px'>[{source_label}]</span>"
            if source_label else ""
        )

        rows.append(
            f"<div class='exec-row {row_cls}'>"
            f"<div class='exec-time'>{time_str}</div>"
            f"<div class='exec-main'>"
            f"<div class='exec-action-wrap'>"
            f"<span class='exec-action {action_cls}'>{action_display}</span>"
            f"<span class='exec-symbol'>{item.symbol}</span>"
            f"{type_badge}"
            f"</div>"
            f"</div>"
            f"<div class='exec-fill'><span class='exec-qty'>{qty_str}</span><span class='exec-price'>{price_str}</span></div>"
            f"<div class='exec-reason'>{source_html}{reason_text}</div>"
            "</div>"
        )
    return (
        "<div class='exec-panel'>"
        "<div class='sec-head' style='margin-top:0;margin-bottom:0.2rem'>What Changed Recently</div>"
        "<div class='subtle-copy' style='margin-bottom:0.85rem'>Recent submits, blocked trades, and holds that did not need action.</div>"
        f"<div class='exec-list'>{''.join(rows)}</div>"
        "</div>"
    )


def _recent_activity_feed_html(recent_activity: list) -> str:
    if not recent_activity:
        return (
            "<div class='exec-panel'>"
            "<div class='sec-head' style='margin-top:0;margin-bottom:0.25rem'>Session Activity Feed</div>"
            "<div class='subtle-copy' style='margin-bottom:0.85rem'>Recent saved signals, blocks, warnings, and state changes pulled from persisted logs.</div>"
            "<div class='rail-empty'>No recent session activity is available yet. That usually means the current session has not written its first meaningful events.</div>"
            "</div>"
        )
    rows: list[str] = []
    for item in recent_activity[:8]:
        badges = [f"<span class='activity-tag {item.level}'>{_escape_html(item.level)}</span>"]
        if item.symbol:
            badges.append(f"<span class='activity-tag'>{_escape_html(item.symbol)}</span>")
        rows.append(
            "<div class='activity-feed-row'>"
            f"<div class='activity-time'>{_compact_time_str(item.observed_at_utc)}</div>"
            "<div>"
            f"<div class='activity-badges'>{''.join(badges)}</div>"
            f"<div class='activity-title'>{_escape_html(item.title)}</div>"
            f"<div class='activity-body'>{_escape_html(item.body)}</div>"
            "</div>"
            "</div>"
        )
    return (
        "<div class='exec-panel'>"
        "<div class='sec-head' style='margin-top:0;margin-bottom:0.25rem'>Session Activity Feed</div>"
        "<div class='subtle-copy' style='margin-bottom:0.85rem'>Recent saved signals, blocks, warnings, and state changes pulled from persisted logs.</div>"
        f"<div class='activity-feed-list'>{''.join(rows)}</div>"
        "</div>"
    )


def _decision_logic_html(signal_row) -> str:
    if signal_row is None:
        return (
            "<div class='detail-history-card'>"
            "<div class='sec-head'>Why This Symbol Looks This Way</div>"
            "<div class='rail-empty'>No saved decision breakdown is available for this symbol yet. The latest persisted logs have not captured enough detail to explain this view safely.</div>"
            "</div>"
        )
    result_reason = _pretty_reason(signal_row.final_signal_reason or signal_row.rejection)
    if not signal_row.final_signal_reason and not signal_row.rejection:
        result_reason = "actionable signal present" if (signal_row.action or "").upper() in {"BUY", "SELL"} else "no explicit rejection persisted"
    trend_state = "YES" if signal_row.above_trend_sma else "NO"
    strategy_mode = (signal_row.strategy_mode or "unknown").replace("_", " ").upper()
    decision_summary = signal_row.decision_summary or "No richer saved explanation is available for this decision yet."
    items = "".join(
        [
            _logic_line(
                "Strategy",
                _logic_value_html(strategy_mode, "neutral"),
                f"regime {signal_row.regime_state}" if signal_row.regime_state else "",
            ),
            _logic_line(
                "Price vs SMA20",
                _logic_value_html(_fmt_pct(signal_row.deviation_pct), "neutral"),
                f"reference {f'${signal_row.sma:,.2f}' if signal_row.sma is not None else 'n/a'}",
            ),
            _logic_line(
                "Trend filter",
                _logic_value_html(
                    (signal_row.trend_filter or "n/a").upper(),
                    "yes" if signal_row.trend_filter == "pass" else ("no" if signal_row.trend_filter == "reject" else "neutral"),
                ),
                f"above trend SMA: {trend_state}",
            ),
            _logic_line(
                "ATR filter",
                _logic_value_html(
                    (signal_row.atr_filter or "n/a").upper(),
                    "yes" if signal_row.atr_filter == "pass" else ("no" if signal_row.atr_filter == "reject" else "neutral"),
                ),
                f"ATR pct {signal_row.atr_pct:.3%}, percentile {signal_row.atr_percentile:.1f}" if signal_row.atr_pct is not None and signal_row.atr_percentile is not None else "",
            ),
            _logic_line(
                "Volume ratio",
                _logic_value_html(
                    f"{signal_row.volume_ratio:.2f}x" if signal_row.volume_ratio is not None else "n/a",
                    "yes" if (signal_row.volume_ratio is not None and signal_row.volume_ratio >= 1.0) else "neutral",
                ),
                "above avg" if (signal_row.volume_ratio is not None and signal_row.volume_ratio >= 1.0) else ("below avg" if signal_row.volume_ratio is not None else ""),
            ),
            _logic_line(
                "Window open",
                _logic_value_html("YES" if signal_row.window_open else "NO", "yes" if signal_row.window_open else "no"),
            ),
            _logic_line(
                "Position held",
                _logic_value_html("YES" if signal_row.holding else "NO", "yes" if signal_row.holding else "neutral"),
            ),
            _logic_line(
                "Decision core",
                _logic_value_html(_pretty_reason(signal_row.final_signal_reason), "neutral"),
                decision_summary,
            ),
            _logic_line(
                "Reference levels",
                _logic_value_html(
                    f"${signal_row.entry_reference_price:,.2f}" if signal_row.entry_reference_price is not None else "n/a",
                    "neutral",
                ),
                " / ".join(
                    part
                    for part in [
                        f"exit {signal_row.exit_reference_price:,.2f}" if signal_row.exit_reference_price is not None else "",
                        f"stop {signal_row.stop_reference_price:,.2f}" if signal_row.stop_reference_price is not None else "",
                        f"VWAP {signal_row.vwap:,.2f}" if signal_row.vwap is not None else "",
                    ]
                    if part
                ),
            ),
            _logic_line(
                "Branch context",
                _logic_value_html(
                    (signal_row.hybrid_branch_active or signal_row.mr_signal or "n/a").replace("_", " ").upper(),
                    "neutral",
                ),
                " / ".join(
                    part
                    for part in [
                        f"entry {signal_row.hybrid_entry_branch}" if signal_row.hybrid_entry_branch else "",
                        f"regime {signal_row.hybrid_regime_branch}" if signal_row.hybrid_regime_branch else "",
                        f"trend slope {signal_row.trend_sma_slope:.4f}" if signal_row.trend_sma_slope is not None else "",
                        f"ADX {signal_row.adx:.1f}" if signal_row.adx is not None else "",
                    ]
                    if part
                ),
            ),
            _logic_line(
                "Result",
                _logic_value_html((signal_row.action or "HOLD").upper(), "neutral"),
                f"because {result_reason}",
            ),
        ]
    )
    return (
        "<div class='detail-history-card'>"
        "<div class='sec-head'>Why This Symbol Looks This Way</div>"
        f"<div class='logic-list'>{items}</div>"
        "</div>"
    )


def _cycle_summary_html(
    last_cycle_report,
    latest_signal_rows: list,
    latest_cycle_risk_checks: list | None = None,
    session_block_reason_counts: list | None = None,
) -> str:
    if last_cycle_report is None:
        return (
            "<div class='cycle-summary'>"
            "<div class='workspace-label' style='margin-bottom:0.4rem'>Cycle recap</div>"
            "<div class='cycle-summary-line'>Waiting for the first persisted decision cycle summary.</div>"
            "</div>"
        )
    latest_cycle_risk_checks = latest_cycle_risk_checks or []
    session_block_reason_counts = session_block_reason_counts or []
    matching = [
        row for row in latest_signal_rows
        if row.decision_timestamp == last_cycle_report.decision_timestamp and (row.action or "").upper() == "HOLD"
    ]
    hold_counts: dict[str, int] = {}
    if matching:
        for row in matching:
            reason = row.rejection or "unspecified_hold"
            hold_counts[reason] = hold_counts.get(reason, 0) + 1

    blocked_buys = [row for row in latest_cycle_risk_checks if not row.allowed and row.action == "BUY"]
    blocked_sells = [row for row in latest_cycle_risk_checks if not row.allowed and row.action == "SELL"]
    blocked_reason_counts: dict[str, int] = {}
    for row in blocked_buys + blocked_sells:
        key = row.block_reason or "unknown_block"
        blocked_reason_counts[key] = blocked_reason_counts.get(key, 0) + 1

    line_one_parts = [f"Last cycle {_compact_time_str(last_cycle_report.decision_timestamp)}"]
    if last_cycle_report.orders_submitted:
        line_one_parts.append(f"{last_cycle_report.orders_submitted} orders submitted")
    if blocked_buys:
        line_one_parts.append(f"{len(blocked_buys)} buy signals blocked")
    elif last_cycle_report.buy_signals:
        line_one_parts.append(f"{last_cycle_report.buy_signals} buys")
    if blocked_sells:
        line_one_parts.append(f"{len(blocked_sells)} sell signals blocked")
    elif last_cycle_report.sell_signals:
        line_one_parts.append(f"{last_cycle_report.sell_signals} sells")
    line_one_parts.append(f"{last_cycle_report.hold_signals} holds")

    line_two_parts: list[str] = []
    if blocked_reason_counts:
        top_block = ", ".join(
            f"{_pretty_reason(reason)} {count}"
            for reason, count in sorted(blocked_reason_counts.items(), key=lambda item: (-item[1], item[0]))[:2]
        )
        line_two_parts.append(f"Blocked: {top_block}")
    if hold_counts:
        hold_breakdown = ", ".join(
            f"{_pretty_reason(reason)} {count}"
            for reason, count in sorted(hold_counts.items(), key=lambda item: (-item[1], item[0]))[:2]
        )
        line_two_parts.append(f"Holds: {hold_breakdown}")
    line_three_parts: list[str] = []
    if session_block_reason_counts:
        session_breakdown = ", ".join(
            f"{_pretty_reason(reason)} {count}"
            for reason, count in session_block_reason_counts[:3]
        )
        line_three_parts.append(f"Session blocks: {session_breakdown}")
    line_three_html = (
        f"<div class='cycle-summary-line'>{' | '.join(line_three_parts)}</div>"
        if line_three_parts
        else ""
    )
    return (
        "<div class='cycle-summary'>"
        "<div class='workspace-label' style='margin-bottom:0.4rem'>Cycle recap</div>"
        f"<div class='cycle-summary-line'>{' | '.join(line_one_parts)}</div>"
        f"<div class='cycle-summary-line'>{' | '.join(line_two_parts) if line_two_parts else _cycle_reason_label(last_cycle_report.skip_reason)}</div>"
        f"{line_three_html}"
        "</div>"
    )


def _render_detail_panel(
    storage,
    item,
    pos_row: dict | None,
    preview,
    snapshot,
    signal_row=None,
    *,
    ml_enabled: bool = True,
) -> None:
    price_str = f"${item.price:,.2f}" if item.price is not None else "â€”"
    try:
        sym_history = pd.DataFrame(storage.get_symbol_history(item.symbol, limit=20))
    except Exception:
        sym_history = pd.DataFrame()

    if (item.action or "").upper() == "ERROR" or item.error:
        error_text = item.error or "Unknown error"
    else:
        error_text = ""

    pos_str = "â€”"
    if pos_row is not None:
        mv = _fmt_money(float(pos_row.get("market_value") or 0.0))
        qty = f"{float(pos_row.get('qty') or 0.0):.2f} sh"
        pos_str = f"{mv} · {qty}"
    hold_str = (
        f"{item.holding_minutes:.0f} min"
        if item.holding and item.holding_minutes is not None
        else "â€”"
    )
    sma_str = f"${item.sma:,.2f}" if item.sma is not None else "â€”"
    ml_probability = item.ml_probability_up if ml_enabled else None
    ml_confidence = item.ml_confidence if ml_enabled else None
    ml_str = f"{ml_probability:.3f}" if ml_probability is not None else "â€”"
    signal_str = preview.status if preview is not None else "â€”"
    delta_str = "â€”"
    if item.price is not None and item.sma not in (None, 0):
        diff_pct = ((item.price - item.sma) / item.sma) * 100
        sign = "+" if diff_pct >= 0 else ""
        delta_str = f"{sign}{diff_pct:.2f}%"

    action_badge = _badge_html(item.action or "â€”") if item.action else "â€”"
    signal_badge = _execution_status_badge(signal_str) if preview is not None else "â€”"
    context_copy = (
        f"Current action is {(item.action or 'HOLD').upper()} with {signal_str.lower().replace('_', ' ')} execution status."
        if preview is not None
        else f"Current action is {(item.action or 'HOLD').upper()}."
    )
    st.markdown(
        "<div class='detail-hero'>"
        "<div class='detail-hero-top'>"
        "<div>"
        f"<div class='detail-symbol'>{item.symbol}</div>"
        f"<div class='detail-subline'><span>Updated {_relative_age(snapshot.timestamp_utc)}</span><span>&bull;</span><span>Selected symbol</span></div>"
        f"<div class='detail-pill-row'>{action_badge}{signal_badge if signal_badge != 'â€”' else ''}</div>"
        "</div>"
        f"<div class='detail-price'>{price_str}</div>"
        "</div>"
        + (f"<div class='detail-error-banner'>{error_text}</div>" if error_text else "")
        + "<div class='detail-info-grid'>"
        + _detail_info_cell("Position", pos_str)
        + _detail_info_cell("Hold Time", hold_str)
        + _detail_info_cell("SMA", sma_str)
        + _detail_info_cell("ML Prob Up", ml_str)
        + _detail_info_cell("Execution", signal_badge)
        + _detail_info_cell("Price vs SMA", delta_str)
        + "</div>"
        + _trend_strip_html(sym_history)
        + _prox_bar_html(item.price, item.sma)
        + _ml_bar_html(
            ml_probability,
            getattr(item, "ml_buy_threshold", None),
            getattr(item, "ml_sell_threshold", None),
            ml_confidence,
        )
        + f"<div class='detail-context'>{context_copy}</div>"
        + "</div>",
        unsafe_allow_html=True,
    )

    st.markdown(
        "<div class='detail-history-card'><div class='sec-head'>Recent Snapshot History</div>",
        unsafe_allow_html=True,
    )
    try:
        if not sym_history.empty:
            cols_wanted = [c for c in ["timestamp_utc", "sma", "price", "action"] if c in sym_history.columns]
            if cols_wanted:
                hist_df = sym_history[cols_wanted].tail(20)
                _col_labels = {"timestamp_utc": "Time", "sma": "SMA", "price": "Price", "action": "Action"}
                _action_cls = {
                    "BUY": "hist-action-buy",
                    "SELL": "hist-action-sell",
                    "HOLD": "hist-action-hold",
                    "ERROR": "hist-action-err",
                    "SNAPSHOT_ONLY": "hist-action-snapshot",
                }
                thead = "".join(f"<th>{_col_labels.get(c, c)}</th>" for c in cols_wanted)
                tbody = ""
                for _, row in hist_df.iloc[::-1].iterrows():
                    tds = ""
                    for c in cols_wanted:
                        v = row[c]
                        if c == "timestamp_utc":
                            tds += f"<td class='mono'>{_format_datetime_pretty(v)}</td>"
                        elif c in ("sma", "price"):
                            tds += f"<td class='mono'>{_fmt_optional_money(v)}</td>"
                        elif c == "action":
                            cls = _action_cls.get(str(v or "").upper(), "hist-action-hold")
                            action_label = "Snapshot" if str(v or "").upper() == "SNAPSHOT_ONLY" else str(v or "—").replace("_", " ")
                            tds += f"<td class='{cls}'>{html.escape(action_label)}</td>"
                        else:
                            tds += f"<td>{html.escape(str(v or '—'))}</td>"
                    tbody += f"<tr>{tds}</tr>"
                st.markdown(
                    f"<div class='hist-wrap'><table class='hist-table'>"
                    f"<thead><tr>{thead}</tr></thead>"
                    f"<tbody>{tbody}</tbody>"
                    f"</table></div></div>",
                    unsafe_allow_html=True,
                )
            else:
                st.markdown("<div class='rail-empty'>No history columns available.</div></div>", unsafe_allow_html=True)
        else:
            st.markdown("<div class='rail-empty'>No history yet for this symbol.</div></div>", unsafe_allow_html=True)
    except Exception as exc:
        st.markdown(f"<div class='rail-empty'>History unavailable: {exc}</div></div>", unsafe_allow_html=True)
    st.markdown(_decision_logic_html(signal_row), unsafe_allow_html=True)


def _trader_thoughts_html(narrations: list) -> str:
    """Render the Trader Thoughts explainability panel for the right rail."""
    if not narrations:
        return (
            "<div class='rail-card'>"
            "<div class='sec-head'>Trader Thoughts</div>"
            "<div class='rail-empty'>No recent trader thoughts available.</div>"
            "</div>"
        )
    rows: list[str] = []
    for item in narrations:
        time_str = _compact_time_str(item.timestamp_utc)
        sym_html = (
            f"<span class='thought-sym'>{html.escape(item.symbol)}</span>"
            if item.symbol and item.symbol != "?"
            else ""
        )
        rows.append(
            f"<div class='thought-row'>"
            f"<div class='thought-meta'>{sym_html}"
            f"<span class='thought-ts'>{time_str}</span></div>"
            f"<div class='thought-text'>{html.escape(item.narration)}</div>"
            f"</div>"
        )
    return (
        "<div class='rail-card'>"
        "<div class='sec-head'>Trader Thoughts</div>"
        f"{''.join(rows)}"
        "</div>"
    )


def _near_misses_html(near_misses: list) -> str:
    """Render the Near Misses explainability panel for the right rail."""
    if not near_misses:
        return (
            "<div class='rail-card'>"
            "<div class='sec-head'>Near Misses</div>"
            "<div class='rail-empty'>No near-miss setups detected recently.</div>"
            "</div>"
        )
    rows: list[str] = []
    for item in near_misses:
        time_str = _compact_time_str(item.timestamp_utc)
        sym_html = (
            f"<span class='miss-sym'>{html.escape(item.symbol)}</span>"
            if item.symbol and item.symbol != "?"
            else ""
        )
        rows.append(
            f"<div class='thought-row'>"
            f"<div class='thought-meta'>{sym_html}"
            f"<span class='thought-ts'>{time_str}</span></div>"
            f"<div class='thought-text'>{html.escape(item.narration)}</div>"
            f"</div>"
        )
    return (
        "<div class='rail-card'>"
        "<div class='sec-head'>Near Misses</div>"
        f"{''.join(rows)}"
        "</div>"
    )


def _render_right_rail(
    snapshot, position_rows: list, recent_orders: list,
    execution_previews: list, preview_lookup: dict,
    recent_narrations: list | None = None,
    recent_near_misses: list | None = None,
    drift_report=None,
) -> None:
    alerts: list[tuple[str, str, list[str]]] = []
    error_syms = [
        item.symbol for item in snapshot.symbols
        if item.error or (item.action or "").upper() == "ERROR"
    ]
    if error_syms:
        alerts.append(("err", "Symbol errors detected", error_syms))

    blocked_syms = [
        p.symbol for p in execution_previews
        if (p.status or "").upper() == "BLOCKED"
    ]
    if blocked_syms:
        alerts.append(("warn", "Execution blocked", blocked_syms))

    if snapshot.kill_switch_triggered:
        alerts.append(("err", "Kill switch active", ["No new entries allowed"]))
    if drift_report is not None:
        for alert in drift_report.alerts[:3]:
            alerts.append((alert.severity, alert.summary, [alert.observed]))

    if alerts:
        alerts_html = "".join(
            _status_card_html(
                severity,
                desc,
                f"Watch closely{' · ' + ' · '.join(syms) if syms else ''}.",
            )
            for severity, desc, syms in alerts
        )
    else:
        alerts_html = _status_card_html("info", "Nothing urgent", "No active warnings in the latest snapshot.")
    st.markdown(
        f"<div class='rail-card'><div class='sec-head'>Alerts</div>{alerts_html}</div>",
        unsafe_allow_html=True,
    )

    positions_html = ""
    if position_rows:
        for row in sorted(position_rows, key=lambda r: float(r["market_value"]), reverse=True):
            pnl = float(row["unrealized_pl"] or 0.0)
            pnl_cls = "m-pos" if pnl >= 0 else "m-neg"
            held_text = (
                f"{float(row['held_mins']):.0f} min"
                if row.get("held_mins") is not None else "â€”"
            )
            positions_html += (
                f"<div class='rail-row'>"
                f"<div class='rail-row-top'>"
                f"<span class='rail-sym'>{row['symbol']}</span>"
                f"<span class='rail-row-val'>{_fmt_money(float(row['market_value'] or 0.0))}</span>"
                f"</div>"
                f"<div class='rail-row-meta'>"
                f"<span>{float(row['qty'] or 0.0):.2f} sh · held {held_text}</span>"
                f"<span class='{pnl_cls}'>{_fmt_money(pnl)}</span>"
                f"</div></div>"
            )
    else:
        positions_html = "<div class='rail-empty'>There are no open positions right now.</div>"
    st.markdown(
        f"<div class='rail-card'><div class='sec-head'>What Youâ€™re Holding</div>{positions_html}</div>",
        unsafe_allow_html=True,
    )

    orders_html = ""
    if recent_orders:
        for order in recent_orders[:8]:
            od = asdict(order)
            side = od.get("side", "")
            symbol = od.get("symbol", "?")
            filled_avg = od.get("filled_avg_price")
            qty = od.get("filled_qty") or od.get("qty")
            submitted = od.get("submitted_at", "")
            status = od.get("status", "")

            action_cls = "buy" if str(side).upper() == "BUY" else "sell"
            price_str = f"${float(filled_avg):,.2f}" if filled_avg is not None else "â€”"
            qty_str = f"{float(qty):.2f}" if qty is not None else "â€”"
            time_str = _compact_time_str(submitted)

            orders_html += (
                f"<div class='order-row'>"
                f"<div class='order-row-top'>"
                f"<div class='order-main'>"
                f"<span class='order-action {action_cls}'>{str(side).upper()}</span>"
                f"<span class='order-symbol'>{symbol}</span>"
                f"</div>"
                f"<span class='order-price'>{price_str}</span>"
                f"</div>"
                f"<div class='order-meta'>"
                f"<span class='order-detail'>{qty_str} sh · {status}</span>"
                f"<span class='order-time'>{time_str}</span>"
                f"</div></div>"
            )
    else:
        orders_html = "<div class='rail-empty'>No recent orders have been submitted.</div>"
    st.markdown(
        f"<div class='rail-card'><div class='sec-head'>Orders Sent Recently</div>{orders_html}</div>",
        unsafe_allow_html=True,
    )

    st.markdown(_trader_thoughts_html(recent_narrations or []), unsafe_allow_html=True)
    st.markdown(_near_misses_html(recent_near_misses or []), unsafe_allow_html=True)


# â”€â”€ Tab: Live â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def _render_live(
    storage,
    snapshot,
    recent_orders: list,
    execution_previews: list | None = None,
    latest_signals: dict | None = None,
    recent_execution_activity: list | None = None,
    recent_activity: list | None = None,
    last_cycle_report=None,
    latest_signal_rows: list | None = None,
    latest_cycle_risk_checks: list | None = None,
    *,
    session_first_prices: dict | None = None,
    ml_enabled: bool = False,
    recent_narrations: list | None = None,
    recent_near_misses: list | None = None,
    session_block_reason_counts: list | None = None,
    drift_report=None,
) -> None:
    position_rows = _position_rows(snapshot)
    position_lookup = {str(row["symbol"]): row for row in position_rows}
    latest_signals = latest_signals or {}
    recent_execution_activity = recent_execution_activity or []
    recent_activity = recent_activity or []
    latest_signal_rows = latest_signal_rows or []
    latest_cycle_risk_checks = latest_cycle_risk_checks or []

    if execution_previews is None:
        execution_previews = []
    preview_lookup = {preview.symbol: preview for preview in execution_previews}

    if not snapshot.symbols:
        st.info("No symbol snapshot is available yet. The session may still be initializing or has not persisted its first cycle.")
        return

    if "selected_symbol" not in st.session_state:
        st.session_state.selected_symbol = None

    # Validate selection is still in current list; apply fallback if not
    current_symbols = {s.symbol for s in snapshot.symbols}
    if st.session_state.selected_symbol not in current_symbols:
        action_map = {s.symbol: (s.action or "").upper() for s in snapshot.symbols}
        fallback = (
            next((s.symbol for s in snapshot.symbols if action_map[s.symbol] == "SELL"), None)
            or next((s.symbol for s in snapshot.symbols if action_map[s.symbol] == "BUY"), None)
            or next((s.symbol for s in snapshot.symbols if s.holding), None)
            or (snapshot.symbols[0].symbol if snapshot.symbols else None)
        )
        st.session_state.selected_symbol = fallback

    st.markdown(
        _ticker_strip_html(
            snapshot,
            latest_signals,
            session_first_prices,
            st.session_state.selected_symbol,
        ),
        unsafe_allow_html=True,
    )
    st.markdown(_recent_activity_feed_html(recent_activity), unsafe_allow_html=True)
    st.markdown(_execution_panel_html(recent_execution_activity), unsafe_allow_html=True)
    st.markdown(
        _cycle_summary_html(
            last_cycle_report,
            latest_signal_rows,
            latest_cycle_risk_checks,
            session_block_reason_counts,
        ),
        unsafe_allow_html=True,
    )
    drift_html = _drift_summary_html(drift_report)
    if drift_html:
        st.markdown(drift_html, unsafe_allow_html=True)
    st.markdown(
        _section_intro_html(
            "Live Desk",
            "Scan, inspect, and confirm the current market posture",
            "Read the workspace from left to right: pick a symbol, inspect the thesis in the center, then validate portfolio and execution context on the rail.",
        ),
        unsafe_allow_html=True,
    )
    st.markdown(
        "<div class='live-zones'>"
        "<section class='live-zone'><div class='live-zone-title'>1. Symbol Explorer</div><div class='live-zone-copy'>The watchlist is your fast scan. Holdings and stronger actions are easier to spot first.</div></section>"
        "<section class='live-zone'><div class='live-zone-title'>2. Focus Panel</div><div class='live-zone-copy'>The center column is the explanation surface for one symbol, including signals, state, and recent context.</div></section>"
        "<section class='live-zone'><div class='live-zone-title'>3. Context Rail</div><div class='live-zone-copy'>The right rail keeps orders, holdings, and narration close by without crowding the main explanation panel.</div></section>"
        "</div>",
        unsafe_allow_html=True,
    )

    left_col, center_col, right_col = st.columns([1.05, 1.85, 1.1])

    with left_col:
        st.markdown(
            f"<div class='watchlist-shell'><div class='watchlist-header'><div class='workspace-label'>Symbol Explorer</div><div class='watchlist-subtext'>{len(snapshot.symbols)} names in the live rotation</div><div class='watchlist-tip'>Pick a symbol here, then use the center panel to understand what the bot is seeing.</div></div><div class='watchlist-body'>",
            unsafe_allow_html=True,
        )
        ordered_symbols = sorted(
            snapshot.symbols, key=lambda s: (not s.holding, s.symbol)
        )
        def _select(sym: str) -> None:
            st.session_state.selected_symbol = sym

        for item in ordered_symbols:
            is_selected = st.session_state.selected_symbol == item.symbol
            is_error = (item.action or "").upper() == "ERROR" or bool(item.error)
            with st.container():
                st.markdown(
                    _sym_card_v2_html(
                        item,
                        position_lookup.get(item.symbol),
                        is_selected,
                        is_error,
                        preview_lookup.get(item.symbol),
                        ml_enabled=ml_enabled,
                    ),
                    unsafe_allow_html=True,
                )
                st.button(
                    item.symbol,
                    key=f"wl_{item.symbol}",
                    use_container_width=False,
                    on_click=_select,
                    args=(item.symbol,),
                )
        st.markdown("</div></div>", unsafe_allow_html=True)
    with center_col:
        sel = st.session_state.selected_symbol
        if sel is None:
            st.markdown(
                "<div style='color:#64748b;font-size:0.95rem;margin-top:3rem;text-align:center'>"
                "Select a symbol from the watchlist</div>",
                unsafe_allow_html=True,
            )
        else:
            sel_item = next((s for s in snapshot.symbols if s.symbol == sel), None)
            if sel_item is None:
                st.markdown(
                    "<div style='color:#f87171'>Symbol not found in current snapshot.</div>",
                    unsafe_allow_html=True,
                )
            else:
                _render_detail_panel(
                    storage,
                    sel_item,
                    position_lookup.get(sel),
                    preview_lookup.get(sel),
                    snapshot,
                    latest_signals.get(sel),
                    ml_enabled=ml_enabled,
                )

    with right_col:
        _render_right_rail(
            snapshot, position_rows, recent_orders, execution_previews, preview_lookup,
            recent_narrations=recent_narrations,
            recent_near_misses=recent_near_misses,
            drift_report=drift_report,
        )


# â”€â”€ Tab: History â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def _render_history(state: DashboardState, snapshot) -> None:
    st.markdown(
        _section_intro_html(
            "Performance",
            "Zoom out from the current cycle",
            "This view is for trend reading: account trajectory, one symbol's saved behavior, and the shape of decisions over time.",
        ),
        unsafe_allow_html=True,
    )
    symbols = list(state.startup_config.symbols) if state.startup_config is not None else [item.symbol for item in snapshot.symbols]
    ml_enabled = _strategy_mode_uses_ml(
        state.startup_config.strategy_mode if state.startup_config is not None else None
    )
    if not symbols:
        st.info("No recent session data is available for history yet. Wait for a startup artifact or the first saved snapshot.")
        return
    selected = st.selectbox("Focus symbol", symbols, index=0)

    try:
        session_id = state.startup_config.session_id if state.startup_config is not None else None
        run_history = pd.DataFrame(state.storage.get_run_history(limit=200, session_id=session_id))
        symbol_history = pd.DataFrame(state.storage.get_symbol_history(selected, limit=200, session_id=session_id))
    except Exception as exc:
        st.error(f"Could not load history: {exc}")
        return

    history_points = len(symbol_history.index)
    latest_action = "n/a"
    if not symbol_history.empty and "action" in symbol_history.columns:
        latest_actions = symbol_history["action"].dropna()
        if not latest_actions.empty:
            latest_action = str(latest_actions.iloc[-1]).upper()
    st.markdown(
        "<div class='history-shell'><div class='history-toolbar'>"
        "<section class='history-stat'>"
        "<div class='history-stat-label'>Current focus</div>"
        f"<div class='history-stat-value'>{_escape_html(selected)}</div>"
        "</section>"
        "<section class='history-stat'>"
        "<div class='history-stat-label'>Saved snapshots</div>"
        f"<div class='history-stat-value'>{history_points}</div>"
        "</section>"
        "<section class='history-stat'>"
        "<div class='history-stat-label'>Latest action</div>"
        f"<div class='history-stat-value'>{_escape_html(latest_action)}</div>"
        "</section>"
        "</div></div>",
        unsafe_allow_html=True,
    )

    left, right = st.columns([1.4, 1])

    with left:
        st.markdown("<div class='sec-head'>Account Equity</div>", unsafe_allow_html=True)
        if run_history.empty:
            st.info("Run history has not been written yet for this session.")
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

        st.markdown(f"<div class='sec-head'>{selected} â€” Price vs SMA</div>", unsafe_allow_html=True)
        if symbol_history.empty:
            st.info("This symbol does not have saved history for the current session yet.")
        else:
            sh = symbol_history.copy()
            sh["ts"] = parse_mixed_iso_timestamps(sh["timestamp_utc"])
            sh = sh.dropna(subset=["ts", "price", "sma"]).sort_values("ts")
            if sh.empty:
                st.info("Saved history exists, but no price/SMA series has been persisted for this symbol yet.")
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

            if ml_enabled and "ml_probability_up" in symbol_history.columns:
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
        st.markdown(f"<div class='sec-head'>{selected} â€” Decision Mix</div>", unsafe_allow_html=True)
        if not symbol_history.empty and "action" in symbol_history.columns:
            mix = symbol_history["action"].value_counts().rename_axis("action").reset_index(name="count")
            st.dataframe(mix, use_container_width=True, hide_index=True)

        st.markdown(f"<div class='sec-head'>{selected} â€” Recent Snapshots</div>", unsafe_allow_html=True)
        if not symbol_history.empty:
            latest = symbol_history.tail(10).copy()
            if "timestamp_utc" in latest.columns:
                latest["Time"] = parse_mixed_iso_timestamps(latest["timestamp_utc"]).dt.strftime("%m-%d-%Y %I:%M %p")
            if "price" in latest.columns:
                latest["Price"] = pd.to_numeric(latest["price"], errors="coerce").map(lambda v: f"${v:,.2f}" if pd.notna(v) else "â€”")
            if "sma" in latest.columns:
                latest["SMA"] = pd.to_numeric(latest["sma"], errors="coerce").map(lambda v: f"${v:,.2f}" if pd.notna(v) else "â€”")
            if "action" in latest.columns:
                latest["Action"] = latest["action"].fillna("â€”").astype(str).str.title()
            if ml_enabled and "ml_probability_up" in latest.columns:
                latest["ML Up"] = pd.to_numeric(latest["ml_probability_up"], errors="coerce").map(lambda v: f"{v:.2f}" if pd.notna(v) else "â€”")
            if "holding_minutes" in latest.columns:
                latest["Hold Min"] = pd.to_numeric(latest["holding_minutes"], errors="coerce").round(1).map(lambda v: f"{v:.1f}" if pd.notna(v) else "â€”")
            display_cols = [c for c in ["Time", "Action", "Price", "SMA", "ML Up", "Hold Min"] if c in latest.columns]
            st.dataframe(latest[display_cols].iloc[::-1].reset_index(drop=True), use_container_width=True, hide_index=True)
    _render_experiment_history()


def _experiment_metric_direction(metric_name: str) -> bool:
    return metric_name not in {"max_drawdown_pct"}


def _experiment_metric_label(metric_name: str) -> str:
    labels = {
        "total_return_pct": "Total Return %",
        "profit_factor": "Profit Factor",
        "sharpe": "Sharpe",
        "win_rate": "Win Rate",
        "max_drawdown_pct": "Max Drawdown %",
        "trade_count": "Trade Count",
        "expectancy": "Expectancy",
        "realized_pnl": "Realized PnL",
    }
    return labels.get(metric_name, metric_name.replace("_", " ").title())


def _build_recent_experiment_table(experiments_df: pd.DataFrame, limit: int = 12) -> pd.DataFrame:
    if experiments_df.empty:
        return pd.DataFrame()
    recent = experiments_df.head(limit).copy()
    recent["Time"] = recent["timestamp"].dt.tz_convert(_ET).dt.strftime("%m-%d %I:%M %p")
    recent["Strategy"] = recent["strategy_name"].fillna("n/a")
    recent["Run"] = recent["run_type"].fillna("n/a")
    recent["Summary"] = recent["summary_label"].fillna("n/a")
    for column in ["total_return_pct", "profit_factor", "sharpe", "max_drawdown_pct", "trade_count"]:
        if column in recent.columns:
            recent[column] = pd.to_numeric(recent[column], errors="coerce")
    selected = [column for column in ["Time", "Strategy", "Run", "Summary", "total_return_pct", "profit_factor", "sharpe", "max_drawdown_pct", "trade_count", "dataset_name"] if column in recent.columns]
    renamed = recent[selected].rename(
        columns={
            "total_return_pct": "Return %",
            "profit_factor": "PF",
            "sharpe": "Sharpe",
            "max_drawdown_pct": "Max DD %",
            "trade_count": "Trades",
            "dataset_name": "Dataset",
        }
    )
    return renamed.reset_index(drop=True)


def _build_top_experiment_table(experiments_df: pd.DataFrame, metric_name: str, limit: int = 8) -> pd.DataFrame:
    if experiments_df.empty or metric_name not in experiments_df.columns:
        return pd.DataFrame()
    metric_df = experiments_df.dropna(subset=[metric_name]).copy()
    if metric_df.empty:
        return pd.DataFrame()
    metric_df = metric_df.sort_values(metric_name, ascending=not _experiment_metric_direction(metric_name)).head(limit)
    metric_df["Time"] = metric_df["timestamp"].dt.tz_convert(_ET).dt.strftime("%m-%d %I:%M %p")
    metric_df["Strategy"] = metric_df["strategy_name"].fillna("n/a")
    metric_df["Summary"] = metric_df["summary_label"].fillna("n/a")
    metric_df["Run"] = metric_df["run_type"].fillna("n/a")
    metric_df["Metric"] = pd.to_numeric(metric_df[metric_name], errors="coerce")
    selected = [column for column in ["Time", "Strategy", "Run", "Summary", "Metric", "dataset_name"] if column in metric_df.columns]
    return metric_df[selected].rename(columns={"dataset_name": "Dataset"}).reset_index(drop=True)


def _render_experiment_history() -> None:
    experiments_df = load_experiment_log_frame(DEFAULT_LOG_PATH)
    st.markdown("<div class='sec-head'>Experiment History</div>", unsafe_allow_html=True)
    if experiments_df.empty:
        st.info(f"No experiment log entries found yet at {DEFAULT_LOG_PATH}.")
        return

    available_metrics = [
        metric for metric in [
            "total_return_pct",
            "profit_factor",
            "sharpe",
            "win_rate",
            "max_drawdown_pct",
            "trade_count",
            "expectancy",
            "realized_pnl",
        ]
        if metric in experiments_df.columns and experiments_df[metric].notna().any()
    ]
    if not available_metrics:
        st.info("Experiment log entries exist, but none of the tracked metrics are populated yet.")
        return
    metric_name = st.selectbox(
        "Experiment metric",
        available_metrics,
        format_func=_experiment_metric_label,
        key="experiment_history_metric",
    )

    latest_good = experiments_df[experiments_df["summary_label"] == "improved vs prior"].head(1)
    latest_good_text = "No clearly improved run logged yet."
    if not latest_good.empty:
        row = latest_good.iloc[0]
        metric_value = row.get(metric_name)
        metric_text = f"{metric_value:.3f}" if pd.notna(metric_value) else "n/a"
        latest_good_text = f"{row.get('strategy_name') or 'n/a'} at {row['timestamp'].tz_convert(_ET).strftime('%m-%d %I:%M %p')} with {_experiment_metric_label(metric_name)} {metric_text}."

    top_row = experiments_df.dropna(subset=[metric_name]).sort_values(
        metric_name,
        ascending=not _experiment_metric_direction(metric_name),
    ).head(1)
    top_metric_text = "n/a"
    if not top_row.empty:
        value = top_row.iloc[0][metric_name]
        top_metric_text = f"{value:.3f}" if pd.notna(value) else "n/a"

    hdr_left, hdr_mid, hdr_right = st.columns(3)
    with hdr_left:
        st.metric("Logged runs", len(experiments_df))
    with hdr_mid:
        st.metric(f"Best {_experiment_metric_label(metric_name)}", top_metric_text)
    with hdr_right:
        st.caption(f"Last clearly good result: {latest_good_text}")

    trend_df = experiments_df.dropna(subset=[metric_name]).sort_values("timestamp")
    if not trend_df.empty:
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=trend_df["timestamp"],
                y=trend_df[metric_name],
                mode="lines+markers",
                line=dict(color="#67b8ff", width=2),
                marker=dict(
                    size=8,
                    color=[
                        "#4ce2c5" if label == "improved vs prior"
                        else "#f59e0b" if label == "insufficient evidence"
                        else "#f87171"
                        for label in trend_df["summary_label"].fillna("")
                    ],
                ),
                text=trend_df["strategy_name"].fillna("n/a"),
                hovertemplate="%{text}<br>%{x}<br>%{y}<extra></extra>",
                name=_experiment_metric_label(metric_name),
            )
        )
        fig.update_layout(
            paper_bgcolor="#1a1d2e",
            plot_bgcolor="#1a1d2e",
            font=dict(family="DM Sans, sans-serif", color="#94a3b8", size=11),
            margin=dict(l=0, r=0, t=20, b=0),
            xaxis=dict(gridcolor="#2d3148"),
            yaxis=dict(gridcolor="#2d3148", title=_experiment_metric_label(metric_name)),
            height=280,
        )
        st.plotly_chart(fig, use_container_width=True)

    left, right = st.columns([1.25, 1])
    with left:
        st.markdown("<div class='sec-head'>Recent Experiments</div>", unsafe_allow_html=True)
        st.dataframe(_build_recent_experiment_table(experiments_df), use_container_width=True, hide_index=True)
    with right:
        st.markdown(f"<div class='sec-head'>Top Runs By {_experiment_metric_label(metric_name)}</div>", unsafe_allow_html=True)
        st.dataframe(_build_top_experiment_table(experiments_df, metric_name), use_container_width=True, hide_index=True)


# â”€â”€ Tab: Config â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def _render_config(state: DashboardState, snapshot) -> None:
    cfg = state.startup_config
    ml_enabled = _strategy_mode_uses_ml(cfg.strategy_mode if cfg is not None else None)
    left, right = st.columns(2)

    with left:
        st.markdown("<div class='sec-head'>How The Bot Is Set Up</div>", unsafe_allow_html=True)
        if cfg is None:
            st.markdown(_kv("Startup Config", "No saved startup config yet"), unsafe_allow_html=True)
        else:
            st.markdown(
                "".join(
                    _kv(k, v)
                    for k, v in [
                        ("Mode", cfg.strategy_mode),
                        ("Symbols", ", ".join(cfg.symbols)),
                        ("Bar timeframe", f"{cfg.bar_timeframe_minutes} min"),
                        ("SMA bars", str(cfg.sma_bars)),
                        ("Paper mode", str(cfg.paper)),
                        ("Launch mode", cfg.launch_mode),
                        ("Execution enabled", str(cfg.execution_enabled)),
                    ]
                ),
                unsafe_allow_html=True,
            )

        st.markdown("<div class='sec-head'>This Session</div>", unsafe_allow_html=True)
        session_rows = [
            ("Last snapshot", _format_datetime_pretty(snapshot.timestamp_utc)),
            ("History database", str(state.storage.db_path)),
            ("Dashboard mode", "Read-only live-bot view"),
            ("Feed status", state.feed_status),
            ("Saved symbol state", state.symbol_state_status.replace("_", " ")),
        ]
        if cfg is None:
            session_rows.append(("Startup config", "Missing"))
        else:
            session_rows.append(("Startup config", _format_datetime_pretty(cfg.started_at_utc) if cfg.started_at_utc else "Available"))
            session_rows.append(("Session ID", cfg.session_id or "n/a"))
            session_rows.append(("Startup artifact", cfg.artifact_path or "n/a"))
            session_rows.append(("Symbols in use", ", ".join(cfg.symbols) or "n/a"))
        if state.ignored_snapshot_symbols:
            session_rows.append(("Ignored snapshot symbols", ", ".join(state.ignored_snapshot_symbols)))
        if cfg is not None:
            approval_text = (
                "approved"
                if cfg.runtime_config_approved is True else
                "unapproved"
                if cfg.runtime_config_approved is False else
                "unknown"
            )
            session_rows.append(("Runtime config path", cfg.runtime_config_path or "n/a"))
            session_rows.append(("Runtime overrides", ", ".join(cfg.runtime_overrides) or "none"))
            session_rows.append(("Runtime approval", approval_text))
        st.markdown("".join(_kv(k, v) for k, v in session_rows), unsafe_allow_html=True)

    with right:
        st.markdown("<div class='sec-head'>Safety Limits</div>", unsafe_allow_html=True)
        if cfg is None:
            st.markdown(_kv("Safety limits", "No saved startup config yet"), unsafe_allow_html=True)
        else:
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

        st.markdown("<div class='sec-head'>Model Thresholds</div>", unsafe_allow_html=True)
        if not ml_enabled:
            st.markdown(_kv("Model", "n/a for current strategy mode"), unsafe_allow_html=True)
        else:
            model_names = sorted(
                name for item in snapshot.symbols
                if (name := getattr(item, "ml_model_name", None)) is not None
            )
            if model_names:
                st.markdown(_kv("Model", ", ".join(model_names)), unsafe_allow_html=True)
            ml_rows = [
                f"{item.symbol}: buy â‰¥ {item.ml_buy_threshold:.2f}  sell â‰¤ {item.ml_sell_threshold:.2f}"
                for item in snapshot.symbols
                if getattr(item, "ml_buy_threshold", None) is not None
            ]
            if ml_rows:
                st.markdown("".join(_kv("Threshold", r) for r in ml_rows), unsafe_allow_html=True)
            else:
                st.markdown(_kv("Thresholds", "n/a"), unsafe_allow_html=True)


# â”€â”€ Main â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def _render_config_v2(state: DashboardState, snapshot) -> None:
    st.markdown(
        _section_intro_html(
            "Setup",
            "See the runtime shape without digging through files",
            "This section groups the bot's operating mode, session context, guardrails, and ML gates so the configuration reads like a system overview.",
        ),
        unsafe_allow_html=True,
    )
    cfg = state.startup_config
    ml_enabled = _strategy_mode_uses_ml(cfg.strategy_mode if cfg is not None else None)
    left, right = st.columns(2)

    with left:
        if cfg is None:
            st.markdown(
                "<div class='config-stack'>"
                "<section class='config-card'>"
                "<div class='config-card-top'>"
                "<div><div class='config-card-title'>How The Bot Is Set Up</div>"
                "<div class='config-card-note'>Startup configuration has not been persisted yet.</div></div>"
                "<div class='config-hero'>CONFIG MISSING</div>"
                "</div>"
                "<div class='config-empty'>No saved startup config yet.</div>"
                "</section>",
                unsafe_allow_html=True,
            )
        else:
            setup_stats = [
                _config_stat_html("Strategy", cfg.strategy_mode, "Current trading style"),
                _config_stat_html("Bar cadence", f"{cfg.bar_timeframe_minutes} min", f"SMA uses {cfg.sma_bars} bars"),
                _config_stat_html("Launch mode", cfg.launch_mode, "Paper/live runtime context"),
                _config_stat_html("Execution", "Enabled" if cfg.execution_enabled else "Disabled", f"Paper mode: {cfg.paper}"),
            ]
            st.markdown(
                "<div class='config-stack'>"
                "<section class='config-card'>"
                "<div class='config-card-top'>"
                "<div><div class='config-card-title'>How The Bot Is Set Up</div>"
                "<div class='config-card-note'>Core runtime choices at a glance, without the spreadsheet feel.</div></div>"
                f"<div class='config-hero'>{_escape_html(cfg.strategy_mode)}</div>"
                "</div>"
                f"<div class='config-stat-grid'>{''.join(setup_stats)}</div>"
                "<div class='config-band'>"
                "<div class='config-band-label'>Symbol Universe</div>"
                f"{_config_chip_cloud_html(list(cfg.symbols))}"
                "</div>"
                "</section>",
                unsafe_allow_html=True,
            )

        session_rows = [
            ("Last snapshot", _format_datetime_pretty(snapshot.timestamp_utc)),
            ("History database", str(state.storage.db_path)),
            ("Dashboard mode", "Read-only live-bot view"),
            ("Feed status", state.feed_status),
            ("Saved symbol state", state.symbol_state_status.replace("_", " ")),
        ]
        if cfg is None:
            session_rows.append(("Startup config", "Missing"))
        else:
            session_rows.append(("Startup config", _format_datetime_pretty(cfg.started_at_utc) if cfg.started_at_utc else "Available"))
            session_rows.append(("Session ID", cfg.session_id or "n/a"))
            session_rows.append(("Startup artifact", cfg.artifact_path or "n/a"))
            session_rows.append(("Symbols in use", ", ".join(cfg.symbols) or "n/a"))
        if state.ignored_snapshot_symbols:
            session_rows.append(("Ignored snapshot symbols", ", ".join(state.ignored_snapshot_symbols)))
        if cfg is not None:
            approval_text = (
                "approved"
                if cfg.runtime_config_approved is True else
                "unapproved"
                if cfg.runtime_config_approved is False else
                "unknown"
            )
            session_rows.append(("Runtime config path", cfg.runtime_config_path or "n/a"))
            session_rows.append(("Runtime overrides", ", ".join(cfg.runtime_overrides) or "none"))
            session_rows.append(("Runtime approval", approval_text))
            session_rows.append(("ML lookback bars", str(cfg.ml_lookback_bars or "n/a")))
            if cfg.strategy_mode == "mean_reversion":
                session_rows.append(("MR exit style", cfg.mean_reversion_exit_style or "n/a"))
                session_rows.append(("MR max ATR pctile", f"{cfg.mean_reversion_max_atr_percentile:.1f}" if cfg.mean_reversion_max_atr_percentile else "off"))
                session_rows.append(("MR trend filter", "on" if cfg.mean_reversion_trend_filter else "off"))
                session_rows.append(("MR slope filter", "on" if cfg.mean_reversion_trend_slope_filter else "off"))
                session_rows.append(("MR stop", f"{cfg.mean_reversion_stop_pct * 100:.2f}%" if cfg.mean_reversion_stop_pct else "off"))
            session_rows.append(("SMA stop", f"{cfg.sma_stop_pct * 100:.2f}%" if cfg.sma_stop_pct else "off"))
            session_rows.append(("Breakout max stop", f"{cfg.breakout_max_stop_pct * 100:.2f}%" if cfg.breakout_max_stop_pct else "off"))
            if cfg.runtime_config_rejection_reasons:
                session_rows.append(("Approval notes", "; ".join(cfg.runtime_config_rejection_reasons)))
        st.markdown(
            "<section class='config-card'>"
            "<div class='config-card-top'>"
            "<div><div class='config-card-title'>This Session</div>"
            "<div class='config-card-note'>Live context, persisted artifacts, and the exact runtime this dashboard is reflecting.</div></div>"
            "<div class='config-hero'>SESSION</div>"
            "</div>"
            f"<div class='config-list'>{''.join(_config_row_html(k, v) for k, v in session_rows)}</div>"
            "</section>"
            "</div>",
            unsafe_allow_html=True,
        )

    with right:
        if cfg is None:
            st.markdown(
                "<div class='config-stack'>"
                "<section class='config-card'>"
                "<div class='config-card-top'>"
                "<div><div class='config-card-title'>Safety Limits</div>"
                "<div class='config-card-note'>Risk controls will appear here once a startup config is available.</div></div>"
                "<div class='config-hero'>SAFETY</div>"
                "</div>"
                "<div class='config-empty'>No saved startup config yet.</div>"
                "</section>",
                unsafe_allow_html=True,
            )
        else:
            safety_stats = [
                _config_stat_html("Max per trade", _fmt_money(cfg.max_usd_per_trade)),
                _config_stat_html("Max symbol exposure", _fmt_money(cfg.max_symbol_exposure_usd)),
                _config_stat_html("Max daily loss", _fmt_money(cfg.max_daily_loss_usd)),
                _config_stat_html("Max open positions", str(cfg.max_open_positions)),
                _config_stat_html("Orders per minute", str(cfg.max_orders_per_minute)),
                _config_stat_html("Price collar", f"{cfg.max_price_deviation_bps:.0f} bps"),
                _config_stat_html("Max live price age", f"{cfg.max_live_price_age_seconds}s"),
                _config_stat_html("Max bar delay", f"{cfg.max_data_delay_seconds}s"),
            ]
            st.markdown(
                "<div class='config-stack'>"
                "<section class='config-card'>"
                "<div class='config-card-top'>"
                "<div><div class='config-card-title'>Safety Limits</div>"
                "<div class='config-card-note'>Hard guardrails the runtime should respect before taking risk.</div></div>"
                "<div class='config-hero'>GUARDRAILS</div>"
                "</div>"
                f"<div class='config-stat-grid'>{''.join(safety_stats)}</div>"
                "</section>",
                unsafe_allow_html=True,
            )

        if not ml_enabled:
            st.markdown(
                "<section class='config-card'>"
                "<div class='config-card-top'>"
                "<div><div class='config-card-title'>Model Thresholds</div>"
                "<div class='config-card-note'>Machine-learning gates are inactive for the current strategy mode.</div></div>"
                "<div class='config-hero'>ML OFF</div>"
                "</div>"
                "<div class='config-empty'>n/a for current strategy mode</div>"
                "</section>"
                "</div>",
                unsafe_allow_html=True,
            )
        else:
            model_names = sorted(
                name for item in snapshot.symbols
                if (name := getattr(item, "ml_model_name", None)) is not None
            )
            ml_rows = [
                f"{item.symbol}: buy >= {item.ml_buy_threshold:.2f} | sell <= {item.ml_sell_threshold:.2f}"
                for item in snapshot.symbols
                if getattr(item, "ml_buy_threshold", None) is not None
            ]
            st.markdown(
                "<section class='config-card'>"
                "<div class='config-card-top'>"
                "<div><div class='config-card-title'>Model Thresholds</div>"
                "<div class='config-card-note'>Active model names and per-symbol probability gates for ML-assisted entries.</div></div>"
                "<div class='config-hero'>ML ACTIVE</div>"
                "</div>"
                "<div class='config-band' style='margin-top:0;padding-top:0;border-top:none;'>"
                "<div class='config-band-label'>Models</div>"
                f"{_config_chip_cloud_html(model_names, muted=not bool(model_names))}"
                "</div>"
                "<div class='config-band'>"
                "<div class='config-band-label'>Per-Symbol Thresholds</div>"
                f"{_config_chip_cloud_html(ml_rows, muted=not bool(ml_rows))}"
                "</div>"
                "</section>"
                "</div>",
                unsafe_allow_html=True,
            )


def main() -> None:
    _inject_css()

    try:
        state = load_dashboard_state()
    except Exception as exc:
        render_startup_error(exc)
        return

    snapshot = state.snapshot
    recent_orders = state.recent_orders
    execution_previews = None
    last_cycle_report = state.last_cycle_report
    dashboard_config = state.startup_config or SimpleNamespace(
        paper=False,
        symbols=[],
        strategy_mode="no persisted startup config",
        bar_timeframe_minutes=15,
    )
    bot = SimpleNamespace(
        config=dashboard_config,
        storage=state.storage,
        get_price_feed_status=lambda: state.feed_status,
    )

    refresh = _render_operator_header(state, snapshot, recent_orders)

    if refresh:
        try:
            state = load_dashboard_state()
            snapshot = state.snapshot
            recent_orders = state.recent_orders
            execution_previews = None
            last_cycle_report = state.last_cycle_report
            dashboard_config = state.startup_config or SimpleNamespace(
                paper=False,
                symbols=[],
                strategy_mode="no persisted startup config",
                bar_timeframe_minutes=15,
            )
            bot = SimpleNamespace(
                config=dashboard_config,
                storage=state.storage,
                get_price_feed_status=lambda: state.feed_status,
            )
        except Exception as exc:
            render_startup_error(exc)
            return

    status_cards: list[str] = []
    for message in state.session_warnings:
        status_cards.append(_status_card_html("warn", "Runtime Warning", message))
    if status_cards:
        st.markdown("".join(status_cards), unsafe_allow_html=True)

    live_tab, history_tab, config_tab, drill_tab = st.tabs(["Live Desk", "Performance", "Setup", "Decision Log"])

    with live_tab:
        _render_live(
            bot.storage,
            snapshot,
            recent_orders,
            execution_previews,
            state.latest_signals,
            list(state.recent_execution_activity),
            list(state.recent_activity),
            last_cycle_report,
            list(state.latest_signal_rows),
            list(state.latest_cycle_risk_checks),
            session_first_prices=state.session_first_prices,
            ml_enabled=_strategy_mode_uses_ml(
                state.startup_config.strategy_mode if state.startup_config is not None else None
            ),
            recent_narrations=list(state.recent_narrations),
            recent_near_misses=list(state.recent_near_misses),
            session_block_reason_counts=list(state.session_block_reason_counts),
            drift_report=state.drift_report,
        )

    with history_tab:
        _render_history(state, snapshot)

    with config_tab:
        _render_config_v2(state, snapshot)

    with drill_tab:
        _render_drilldown(state)


if __name__ == "__main__":
    main()


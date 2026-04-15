from dataclasses import asdict
from datetime import datetime, timedelta, timezone
import html
from types import SimpleNamespace
from zoneinfo import ZoneInfo

_ET = ZoneInfo("America/New_York")
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components
from dashboard_state import DashboardState, DrilldownEvent, load_dashboard_state

st.set_page_config(page_title="TradeOS", layout="wide")


def _strategy_mode_uses_ml(strategy_mode: str | None) -> bool:
    return str(strategy_mode or "").strip().lower() in {"ml", "hybrid"}


# ── CSS ───────────────────────────────────────────────────────────────────────

def _inject_css() -> None:
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=DM+Sans:wght@400;500;700&display=swap');
        :root {
            --bg: #0b111b;
            --panel: rgba(14, 22, 35, 0.92);
            --panel-strong: rgba(18, 28, 43, 0.98);
            --panel-soft: rgba(255,255,255,0.035);
            --line: rgba(148, 163, 184, 0.16);
            --text: #edf4ff;
            --text-soft: #adc1d9;
            --text-dim: #71859c;
            --accent: #67b8ff;
            --accent-2: #4ce2c5;
            --glow: 0 18px 50px rgba(5, 10, 20, 0.36);
        }
        html, body, [class*="css"] { font-family: 'DM Sans', 'Segoe UI', sans-serif; font-size: 20px; }
        code, pre, .mono { font-family: 'DM Mono', 'Consolas', monospace; }

        .stApp {
            background:
                radial-gradient(circle at top left, rgba(71, 132, 255, 0.14), transparent 28%),
                radial-gradient(circle at 85% 0%, rgba(76, 226, 197, 0.10), transparent 20%),
                radial-gradient(circle at 50% 20%, rgba(255,255,255,0.03), transparent 35%),
                linear-gradient(180deg, #08101a 0%, #0b111b 35%, #0b111b 100%);
            color: #d8e0ec;
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

        /* Badges — rectangular chip style, DM Mono, semi-transparent tints */
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

        /* ── v2 dashboard layout ─────────────────────────────────── */
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
        .hist-wrap { max-height: 280px; overflow-y: auto; border: 0.5px solid rgba(255,255,255,0.07); background: rgba(255,255,255,0.015); border-radius: 10px; }
        .hist-table { width: 100%; border-collapse: collapse; font-size: 13px; }
        .hist-table th { position: sticky; top: 0; background: rgba(15,17,23,0.96); padding: 9px 11px; text-align: left; font-size: 10px; text-transform: uppercase; letter-spacing: .11em; color: #647b96; border-bottom: 0.5px solid rgba(255,255,255,0.07); font-weight: 500; font-family: 'DM Sans', sans-serif; }
        .hist-table td { padding: 9px 11px; border-bottom: 0.5px solid rgba(255,255,255,0.04); color: #c7d3df; }
        .hist-table tr:last-child td { border-bottom: none; }
        .hist-table .mono { font-family: 'DM Mono', monospace; color: #8fa3ba; }
        .hist-action-buy  { color: #4ade80; font-weight: 600; }
        .hist-action-sell { color: #f87171; font-weight: 600; }
        .hist-action-hold { color: #4a5c6e; }
        .hist-action-err  { color: #ef9f27; font-weight: 600; }

        /* Tabs */
        [data-testid="stTabs"] { margin-top: 0.1rem; }

        /* Watchlist card+button overlay — button is an invisible full-cover click target */
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
        .sym-card-v2 {
            padding: 14px 16px;
            border-bottom: 1px solid rgba(255,255,255,0.045);
            cursor: pointer;
            border-left: 3px solid transparent;
            transition: background 0.14s ease, border-color 0.14s ease, transform 0.14s ease;
        }
        .sym-card-v2:last-child { border-bottom: none; }
        .sym-card-v2:hover {
            background: rgba(255,255,255,0.032);
            transform: translateX(3px);
        }
        .sym-card-v2.sym-active {
            background: linear-gradient(90deg, rgba(59,130,246,0.14), rgba(59,130,246,0.04));
            border-left-color: #60a5fa;
        }
        .sym-card-v2.sym-error {
            background: linear-gradient(90deg, rgba(248,113,113,0.14), rgba(248,113,113,0.04));
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
            margin-top: 7px;
            font-size: 0.9rem;
            color: #cad5e4;
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
            margin-top: 6px;
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
        /* Near-miss symbol chip — amber tint to signal "almost" */
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
            gap: 8px;
            margin-bottom: 0.8rem;
        }
        .ticker-strip-head {
            display: flex;
            justify-content: space-between;
            align-items: center;
            gap: 10px;
        }
        .ticker-strip-ribbon {
            display: flex;
            gap: 10px;
            overflow-x: auto;
            padding: 12px 14px;
            border-radius: 16px;
            background: linear-gradient(90deg, rgba(17, 25, 38, 0.98), rgba(10, 17, 27, 0.98));
            border: 1px solid rgba(129, 151, 181, 0.14);
            box-shadow: inset 0 1px 0 rgba(255,255,255,0.02);
        }
        .ticker-chip {
            flex: 0 0 auto;
            display: flex;
            align-items: center;
            gap: 9px;
            padding: 7px 9px;
            border-radius: 10px;
            background: rgba(255,255,255,0.045);
            border: 1px solid rgba(255,255,255,0.08);
        }
        .ticker-chip.active {
            background: rgba(96,165,250,0.11);
            border-color: rgba(96,165,250,0.28);
        }
        .ticker-chip-symbol {
            font-family: 'DM Mono', monospace;
            font-size: 0.95rem;
            font-weight: 700;
            color: #f8fbff;
        }
        .ticker-chip-price {
            font-family: 'DM Mono', monospace;
            font-size: 0.9rem;
            color: #dfe7f2;
        }
        .ticker-chip-change {
            font-family: 'DM Mono', monospace;
            font-size: 0.86rem;
            font-weight: 700;
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
            gap: 8px;
            margin-top: 0.9rem;
        }
        .logic-item {
            display: flex;
            gap: 8px;
            align-items: flex-start;
            font-size: 1rem;
            color: #c8d3e1;
            line-height: 1.35;
            padding: 9px 11px;
            background: rgba(255,255,255,0.025);
            border: 1px solid rgba(255,255,255,0.05);
            border-radius: 10px;
        }
        .logic-bullet {
            color: #60a5fa;
            font-family: 'DM Mono', monospace;
            margin-top: 1px;
        }
        .logic-key {
            color: #90a3ba;
            min-width: 134px;
            flex: 0 0 134px;
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
            .dash-metrics { grid-template-columns: repeat(2, 1fr); }
            .detail-info-grid { grid-template-columns: repeat(2, minmax(0, 1fr)); }
            .hero-summary,
            .timing-grid,
            .feed-grid { grid-template-columns: repeat(2, minmax(0, 1fr)); }
        }
        @media (max-width: 900px) {
            .top-shell { flex-direction: column; }
            .status-inline { justify-content: flex-start; }
            .detail-hero-top { flex-direction: column; }
            .detail-price { text-align: left; }
            .dash-metrics { grid-template-columns: 1fr; }
            .detail-info-grid { grid-template-columns: 1fr; }
            .hero-summary,
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

        /* ── Trade Decision Drilldown ────────────────────────────── */
        .dd-shell {
            padding: 20px 22px 18px;
            border-radius: 18px;
            background: linear-gradient(180deg, rgba(16,24,38,0.98), rgba(10,15,24,0.98));
            border: 1px solid rgba(103,184,255,0.14);
            box-shadow: 0 12px 32px rgba(0,0,0,0.22);
            margin-bottom: 1.2rem;
        }
        .dd-header {
            display: flex;
            align-items: center;
            gap: 10px;
            margin-bottom: 1.1rem;
            flex-wrap: wrap;
        }
        .dd-symbol {
            font-size: 1.35rem;
            font-weight: 800;
            color: #edf4ff;
            font-family: 'DM Mono', monospace;
            letter-spacing: 0.04em;
        }
        .dd-ts {
            font-size: 0.86rem;
            color: #4a6474;
            margin-left: 4px;
        }
        .dd-section {
            margin-top: 1.1rem;
            padding-top: 0.9rem;
            border-top: 1px solid rgba(255,255,255,0.06);
        }
        .dd-path { margin-top: 0.5rem; }
        .dd-path-step {
            display: flex;
            align-items: flex-start;
            gap: 10px;
            padding: 7px 0;
            border-bottom: 0.5px solid rgba(255,255,255,0.04);
        }
        .dd-path-step:last-child { border-bottom: none; }
        .dd-step-idx {
            font-family: 'DM Mono', monospace;
            font-size: 0.78rem;
            color: #3d5a78;
            min-width: 22px;
            padding-top: 2px;
        }
        .dd-step-key {
            font-size: 0.92rem;
            color: #8fa3ba;
            min-width: 150px;
            flex: 0 0 150px;
        }
        .dd-step-val { font-size: 0.92rem; color: #d7e1ec; }
        .dd-pass  { color: #4ade80; font-weight: 700; }
        .dd-reject { color: #f87171; font-weight: 700; }
        .dd-note  { font-size: 0.8rem; color: #4a6070; margin-left: 6px; }
        </style>
        """,
        unsafe_allow_html=True,
    )


# ── Bot singleton ─────────────────────────────────────────────────────────────

# ── Utilities ─────────────────────────────────────────────────────────────────

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


# ── HTML builders ─────────────────────────────────────────────────────────────

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


def _ml_mini_bar_html(
    prob: float | None,
    buy_thr: float | None,
    sell_thr: float | None,
) -> str:
    """Compact track-only ML bar for watchlist cards — no labels, no axis."""
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


def _escape_html(value) -> str:
    return html.escape(str(value if value not in (None, "") else "—"))


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
        <div style="margin:0.35rem 0 0.75rem;padding:18px 20px;border-radius:18px;background:linear-gradient(135deg, rgba(18, 27, 41, 0.98), rgba(12, 16, 24, 0.98));border:1px solid rgba(96,165,250,0.14);box-shadow:0 16px 32px rgba(0, 0, 0, 0.18);">
          <div style="display:flex;justify-content:space-between;align-items:flex-start;gap:18px;flex-wrap:wrap;">
            <div style="min-width:240px;">
              <div style="font-size:0.78rem;text-transform:uppercase;letter-spacing:0.14em;color:#7f93ab;font-weight:700;margin-bottom:0.45rem;">Next Bar In</div>
              <div id="bot-timing-countdown" style="font-size:2.2rem;line-height:1;font-weight:800;color:#f8fbff;font-family:'DM Mono', monospace;"></div>
              <div style="font-size:0.9rem;color:#9cb0c8;margin-top:0.55rem;">Next bar closes at <span id="bot-timing-close-inline" style="color:#f1f6fc;font-family:'DM Mono', monospace;font-weight:700;"></span></div>
            </div>
            <div style="display:grid;grid-template-columns:repeat(auto-fit, minmax(180px, 1fr));gap:12px;flex:1 1 420px;">
              <div style="background:rgba(255,255,255,0.03);border:1px solid rgba(255,255,255,0.06);border-radius:14px;padding:12px 13px;">
                <div style="font-size:0.72rem;text-transform:uppercase;letter-spacing:0.12em;color:#7286a0;margin-bottom:0.35rem;">Current UTC</div>
                <div id="bot-timing-now" style="font-size:1rem;font-weight:700;color:#f5f9ff;font-family:'DM Mono', monospace;"></div>
              </div>
              <div style="background:rgba(255,255,255,0.03);border:1px solid rgba(255,255,255,0.06);border-radius:14px;padding:12px 13px;">
                <div style="font-size:0.72rem;text-transform:uppercase;letter-spacing:0.12em;color:#7286a0;margin-bottom:0.35rem;">Next Bar Close</div>
                <div id="bot-timing-close" style="font-size:1rem;font-weight:700;color:#f5f9ff;font-family:'DM Mono', monospace;"></div>
              </div>
              <div style="background:rgba(255,255,255,0.03);border:1px solid rgba(255,255,255,0.06);border-radius:14px;padding:12px 13px;">
                <div style="font-size:0.72rem;text-transform:uppercase;letter-spacing:0.12em;color:#7286a0;margin-bottom:0.35rem;">Bar Timeframe</div>
                <div style="font-size:1rem;font-weight:700;color:#f5f9ff;font-family:'DM Mono', monospace;">{timeframe_minutes} min</div>
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


# ── Drilldown panel ───────────────────────────────────────────────────────────

def _drilldown_label(event: DrilldownEvent) -> str:
    """Short selector label for one drilldown event."""
    time_str = _compact_time_str(event.timestamp_utc)
    etype_short = {
        "signal.evaluated": "sig",
        "risk.check": "risk",
        "position.closed": "exit",
    }.get(event.event_type, event.event_type)
    action = event.action or "—"
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
        _step("→ Decision", f"<span class='badge {action_cls}'>{action}</span>", outcome_note)

    elif et == "risk.check":
        action = (event.action or "").upper()
        action_cls = {"BUY": "b-buy", "SELL": "b-sell"}.get(action, "b-hold")
        _step("Signal direction", f"<span class='badge {action_cls}'>{action or '—'}</span>")
        if event.allowed is True:
            _step("Risk gate", "<span class='dd-pass'>PASSED</span>", "all checks cleared")
        elif event.allowed is False:
            block_note = _pretty_reason(event.block_reason) if event.block_reason else "reason unknown"
            _step("Risk gate", "<span class='dd-reject'>BLOCKED</span>", block_note)
        else:
            _step("Risk gate", "<span class='dd-step-val'>—</span>")

    elif et == "position.closed":
        exit_label = _escape_html(event.exit_reason or "—")
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

    ts_str = _format_datetime_pretty(event.timestamp_utc) if event.timestamp_utc else "—"

    # Allowed / blocked display
    if event.allowed is True:
        allowed_html = "<span class='dd-pass'>allowed</span>"
    elif event.allowed is False:
        allowed_html = "<span class='dd-reject'>blocked</span>"
    else:
        allowed_html = "<span style='color:#4a6070'>n/a</span>"

    # Primary rejection/block reason
    reason_raw = event.block_reason or event.rejection or event.exit_reason
    reason_html = _escape_html(_pretty_reason(reason_raw) if reason_raw else "—")

    fields: list[str] = [
        _kv("Symbol", _escape_html(event.symbol or "—")),
        _kv("Timestamp", _escape_html(ts_str)),
        _kv("Event type", _escape_html(etype_label)),
        _kv("Action", _escape_html(action or "—")),
        _kv("Strategy mode", _escape_html(event.strategy_mode or "—")),
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
        f"<span class='badge {action_badge_cls}'>{action or '—'}</span>"
        f"<span class='dd-ts'>{_escape_html(ts_str)}</span>"
        "</div>"
        f"{''.join(fields)}"
        + _drilldown_path_html(event)
        + "</div>"
    )


def _render_drilldown(state: DashboardState) -> None:
    """Render the Trade Decision Drilldown tab."""
    candidates = list(state.drilldown_candidates)

    if not candidates:
        st.markdown(
            "<div class='rail-empty' style='margin-top:1.5rem'>"
            "No recent decision events found. "
            "Start the bot and wait for the first bar cycle to populate logs."
            "</div>",
            unsafe_allow_html=True,
        )
        return

    labels = [_drilldown_label(ev) for ev in candidates]
    idx = st.selectbox(
        "Decision event",
        range(len(labels)),
        format_func=lambda i: labels[i],
    )
    if idx is None:
        return

    selected = candidates[int(idx)]
    st.markdown(_drilldown_panel_html(selected), unsafe_allow_html=True)


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


# ── v2 HTML builders ─────────────────────────────────────────────────────────

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


def _header_summary_html(state: DashboardState, snapshot, last_cycle_report) -> str:
    processed_text = "No cycle yet"
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
    price_str = f"${item.price:,.2f}" if item.price is not None else "—"
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


def _compact_time_str(value: str | None) -> str:
    if not value:
        return "—"
    try:
        dt = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        if dt.tzinfo is not None:
            et = dt.astimezone(_ET)
            local = dt.astimezone()
            et_str = et.strftime("%I:%M %p ET").lstrip("0")
            local_str = local.strftime("%I:%M %p").lstrip("0") + " local"
            if et.hour == local.hour and et.minute == local.minute:
                return et_str
            return f"{et_str}<br><span style='opacity:0.6;font-size:0.88em'>{local_str}</span>"
        return dt.strftime("%I:%M %p").lstrip("0")
    except ValueError:
        return str(value)[11:16] if len(str(value)) >= 16 else str(value)


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
                return et_str
            return f"{et_str}<br><span style='opacity:0.6;font-size:0.88em'>{local_str}</span>"
        fmt = "%b %d, %Y · %I:%M %p ET" if include_time else "%b %d, %Y"
        return parsed.strftime(fmt).replace(" 0", " ")
    if value in (None, ""):
        return "—"
    return str(value)


def _pretty_reason(reason: str | None) -> str:
    if not reason:
        return "n/a"
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
    key = str(reason).strip().lower().replace(" ", "_")
    return mapping.get(key, str(reason).replace("_", " "))


def _logic_value_html(text: str, state: str = "neutral") -> str:
    cls = {
        "yes": "logic-val-yes",
        "no": "logic-val-no",
        "neutral": "logic-val-neutral",
    }.get(state, "logic-val-neutral")
    return f"<span class='{cls}'>{text}</span>"


def _logic_line(label: str, value_html: str, note: str = "") -> str:
    suffix = f" <span style='color:#8fa2b7'>{note}</span>" if note else ""
    return (
        "<div class='logic-item'>"
        "<span class='logic-bullet'>•</span>"
        f"<span class='logic-key'>{label}</span>"
        f"<span>{value_html}{suffix}</span>"
        "</div>"
    )


def _ticker_strip_html(snapshot, latest_signals: dict, session_first_prices: dict | None = None, selected_symbol: str | None = None) -> str:
    chips: list[str] = []
    first_prices = session_first_prices or {}
    for item in snapshot.symbols:
        signal_row = latest_signals.get(item.symbol)
        price_str = f"${item.price:,.2f}" if item.price is not None else "—"
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
            f"<span style='font-size:0.72em;color:#566778;font-weight:400;margin-left:3px'>{change_label}</span>"
            if change_val is not None else "—"
        )
        chip_cls = "ticker-chip active" if item.symbol == selected_symbol else "ticker-chip"
        chips.append(
            f"<div class='{chip_cls}'>"
            f"<span class='ticker-chip-symbol'>{item.symbol}</span>"
            f"<span class='ticker-chip-price'>{price_str}</span>"
            f"<span class='ticker-chip-change' style='color:{color}'>{change_html}</span>"
            "</div>"
        )
    data_note = "session % from first recorded bar" if first_prices else "vs SMA20 — no session baseline yet"
    return (
        "<div class='ticker-strip'>"
        "<div class='ticker-strip-head'>"
        "<div class='workspace-label' style='margin-bottom:0'>Market snapshot</div>"
        f"<div class='subtle-copy'>Latest bar-close prices · {data_note}</div>"
        "</div>"
        f"<div class='ticker-strip-ribbon'>{''.join(chips)}</div>"
        "</div>"
    )


def _execution_panel_html(recent_execution_activity: list) -> str:
    if not recent_execution_activity:
        return (
            "<div class='exec-panel'>"
            "<div class='sec-head' style='margin-top:0;margin-bottom:0.6rem'>What Changed Recently</div>"
            "<div class='rail-empty'>Nothing new has been written to the recent activity log yet.</div>"
            "</div>"
        )
    rows: list[str] = []
    for item in recent_execution_activity[:5]:
        time_str = _compact_time_str(item.observed_at_utc)
        qty_str = f"{float(item.qty):.2f} sh" if item.qty is not None else "—"
        price_str = f"${float(item.price):,.2f}" if item.price is not None else "—"
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


def _decision_logic_html(signal_row) -> str:
    if signal_row is None:
        return (
            "<div class='detail-history-card'>"
            "<div class='sec-head'>Why This Symbol Looks This Way</div>"
            "<div class='rail-empty'>There is no saved decision breakdown yet for this symbol.</div>"
            "</div>"
        )
    result_reason = _pretty_reason(signal_row.rejection)
    if not signal_row.rejection:
        result_reason = "actionable signal present" if (signal_row.action or "").upper() in {"BUY", "SELL"} else "no explicit rejection persisted"
    trend_state = "YES" if signal_row.above_trend_sma else "NO"
    items = "".join(
        [
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


def _cycle_summary_html(last_cycle_report, latest_signal_rows: list, latest_cycle_risk_checks: list | None = None) -> str:
    if last_cycle_report is None:
        return (
            "<div class='cycle-summary'>"
            "<div class='workspace-label' style='margin-bottom:0.4rem'>Cycle recap</div>"
            "<div class='cycle-summary-line'>No persisted cycle summary yet.</div>"
            "</div>"
        )
    latest_cycle_risk_checks = latest_cycle_risk_checks or []
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
    return (
        "<div class='cycle-summary'>"
        "<div class='workspace-label' style='margin-bottom:0.4rem'>Cycle recap</div>"
        f"<div class='cycle-summary-line'>{' | '.join(line_one_parts)}</div>"
        f"<div class='cycle-summary-line'>{' | '.join(line_two_parts) if line_two_parts else _cycle_reason_label(last_cycle_report.skip_reason)}</div>"
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
    price_str = f"${item.price:,.2f}" if item.price is not None else "—"
    try:
        sym_history = pd.DataFrame(storage.get_symbol_history(item.symbol, limit=20))
    except Exception:
        sym_history = pd.DataFrame()

    if (item.action or "").upper() == "ERROR" or item.error:
        error_text = item.error or "Unknown error"
    else:
        error_text = ""

    pos_str = "—"
    if pos_row is not None:
        mv = _fmt_money(float(pos_row.get("market_value") or 0.0))
        qty = f"{float(pos_row.get('qty') or 0.0):.2f} sh"
        pos_str = f"{mv} · {qty}"
    hold_str = (
        f"{item.holding_minutes:.0f} min"
        if item.holding and item.holding_minutes is not None
        else "—"
    )
    sma_str = f"${item.sma:,.2f}" if item.sma is not None else "—"
    ml_probability = item.ml_probability_up if ml_enabled else None
    ml_confidence = item.ml_confidence if ml_enabled else None
    ml_str = f"{ml_probability:.3f}" if ml_probability is not None else "—"
    signal_str = preview.status if preview is not None else "—"
    delta_str = "—"
    if item.price is not None and item.sma not in (None, 0):
        diff_pct = ((item.price - item.sma) / item.sma) * 100
        sign = "+" if diff_pct >= 0 else ""
        delta_str = f"{sign}{diff_pct:.2f}%"

    action_badge = _badge_html(item.action or "—") if item.action else "—"
    signal_badge = _execution_status_badge(signal_str) if preview is not None else "—"
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
        f"<div class='detail-pill-row'>{action_badge}{signal_badge if signal_badge != '—' else ''}</div>"
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
                _action_cls = {"BUY": "hist-action-buy", "SELL": "hist-action-sell", "HOLD": "hist-action-hold", "ERROR": "hist-action-err"}
                thead = "".join(f"<th>{_col_labels.get(c, c)}</th>" for c in cols_wanted)
                tbody = ""
                for _, row in hist_df.iloc[::-1].iterrows():
                    tds = ""
                    for c in cols_wanted:
                        v = row[c]
                        if c == "timestamp_utc":
                            tds += f"<td class='mono'>{_format_datetime_pretty(v)}</td>"
                        elif c in ("sma", "price"):
                            tds += f"<td class='mono'>{'$' + f'{float(v):,.2f}' if v is not None else '—'}</td>"
                        elif c == "action":
                            cls = _action_cls.get(str(v or "").upper(), "hist-action-hold")
                            tds += f"<td class='{cls}'>{str(v or '—')}</td>"
                        else:
                            tds += f"<td>{str(v or '—')}</td>"
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
                if row.get("held_mins") is not None else "—"
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
        f"<div class='rail-card'><div class='sec-head'>What You’re Holding</div>{positions_html}</div>",
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
            price_str = f"${float(filled_avg):,.2f}" if filled_avg is not None else "—"
            qty_str = f"{float(qty):.2f}" if qty is not None else "—"
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


# ── Tab: Live ─────────────────────────────────────────────────────────────────

def _render_live(
    storage,
    snapshot,
    recent_orders: list,
    execution_previews: list | None = None,
    latest_signals: dict | None = None,
    recent_execution_activity: list | None = None,
    last_cycle_report=None,
    latest_signal_rows: list | None = None,
    latest_cycle_risk_checks: list | None = None,
    *,
    session_first_prices: dict | None = None,
    ml_enabled: bool = False,
    recent_narrations: list | None = None,
    recent_near_misses: list | None = None,
) -> None:
    position_rows = _position_rows(snapshot)
    position_lookup = {str(row["symbol"]): row for row in position_rows}
    latest_signals = latest_signals or {}
    recent_execution_activity = recent_execution_activity or []
    latest_signal_rows = latest_signal_rows or []
    latest_cycle_risk_checks = latest_cycle_risk_checks or []

    if execution_previews is None:
        execution_previews = []
    preview_lookup = {preview.symbol: preview for preview in execution_previews}

    if not snapshot.symbols:
        st.info("No symbol data in snapshot.")
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
    st.markdown(_execution_panel_html(recent_execution_activity), unsafe_allow_html=True)
    st.markdown(_cycle_summary_html(last_cycle_report, latest_signal_rows, latest_cycle_risk_checks), unsafe_allow_html=True)

    left_col, center_col, right_col = st.columns([1, 3, 1.5])

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
        )


# ── Tab: History ──────────────────────────────────────────────────────────────

def _render_history(state: DashboardState, snapshot) -> None:
    symbols = list(state.startup_config.symbols) if state.startup_config is not None else [item.symbol for item in snapshot.symbols]
    ml_enabled = _strategy_mode_uses_ml(
        state.startup_config.strategy_mode if state.startup_config is not None else None
    )
    if not symbols:
        st.info("No persisted startup config or symbol snapshot yet.")
        return
    selected = st.radio("Symbol", symbols, horizontal=True)

    try:
        session_id = state.startup_config.session_id if state.startup_config is not None else None
        run_history = pd.DataFrame(state.storage.get_run_history(limit=200, session_id=session_id))
        symbol_history = pd.DataFrame(state.storage.get_symbol_history(selected, limit=200, session_id=session_id))
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
        st.markdown(f"<div class='sec-head'>{selected} — Decision Mix</div>", unsafe_allow_html=True)
        if not symbol_history.empty and "action" in symbol_history.columns:
            mix = symbol_history["action"].value_counts().rename_axis("action").reset_index(name="count")
            st.dataframe(mix, use_container_width=True, hide_index=True)

        st.markdown(f"<div class='sec-head'>{selected} — Recent Snapshots</div>", unsafe_allow_html=True)
        if not symbol_history.empty:
            latest = symbol_history.tail(10).copy()
            if "timestamp_utc" in latest.columns:
                latest["Time"] = parse_mixed_iso_timestamps(latest["timestamp_utc"]).dt.strftime("%m-%d-%Y %I:%M %p")
            if "price" in latest.columns:
                latest["Price"] = pd.to_numeric(latest["price"], errors="coerce").map(lambda v: f"${v:,.2f}" if pd.notna(v) else "—")
            if "sma" in latest.columns:
                latest["SMA"] = pd.to_numeric(latest["sma"], errors="coerce").map(lambda v: f"${v:,.2f}" if pd.notna(v) else "—")
            if "action" in latest.columns:
                latest["Action"] = latest["action"].fillna("—").astype(str).str.title()
            if ml_enabled and "ml_probability_up" in latest.columns:
                latest["ML Up"] = pd.to_numeric(latest["ml_probability_up"], errors="coerce").map(lambda v: f"{v:.2f}" if pd.notna(v) else "—")
            if "holding_minutes" in latest.columns:
                latest["Hold Min"] = pd.to_numeric(latest["holding_minutes"], errors="coerce").round(1).map(lambda v: f"{v:.1f}" if pd.notna(v) else "—")
            display_cols = [c for c in ["Time", "Action", "Price", "SMA", "ML Up", "Hold Min"] if c in latest.columns]
            st.dataframe(latest[display_cols].iloc[::-1].reset_index(drop=True), use_container_width=True, hide_index=True)


# ── Tab: Config ───────────────────────────────────────────────────────────────

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
            session_rows.append(("Runtime config path", cfg.runtime_config_path or "n/a"))
            session_rows.append(("Runtime overrides", ", ".join(cfg.runtime_overrides) or "none"))
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
                f"{item.symbol}: buy ≥ {item.ml_buy_threshold:.2f}  sell ≤ {item.ml_sell_threshold:.2f}"
                for item in snapshot.symbols
                if getattr(item, "ml_buy_threshold", None) is not None
            ]
            if ml_rows:
                st.markdown("".join(_kv("Threshold", r) for r in ml_rows), unsafe_allow_html=True)
            else:
                st.markdown(_kv("Thresholds", "n/a"), unsafe_allow_html=True)


# ── Main ──────────────────────────────────────────────────────────────────────

def _render_config_v2(state: DashboardState, snapshot) -> None:
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
            session_rows.append(("Runtime config path", cfg.runtime_config_path or "n/a"))
            session_rows.append(("Runtime overrides", ", ".join(cfg.runtime_overrides) or "none"))
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
    header_symbol_count = len(state.startup_config.symbols) if state.startup_config is not None else len(snapshot.symbols)
    header_mode_label = (
        state.startup_config.strategy_mode
        if state.startup_config is not None
        else ("snapshot only" if state.has_persisted_snapshot else "no persisted startup config")
    )
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

    hdr_l, hdr_r = st.columns([2.2, 1])
    with hdr_l:
        st.markdown(_header_summary_html(state, snapshot, last_cycle_report), unsafe_allow_html=True)

    with hdr_r:
        monitor_badge = "<span class='badge b-nosignal'>READ ONLY</span>"
        if last_cycle_report is not None and last_cycle_report.processed_bar:
            last_text = f"Last bar: {_format_datetime_pretty(last_cycle_report.decision_timestamp)}"
        elif last_cycle_report is not None:
            last_text = _cycle_reason_label(last_cycle_report.skip_reason)
        else:
            last_text = "No persisted cycle yet"
        st.markdown(
            "<div class='action-card'>"
            "<div class='workspace-label'>What You Can Do Here</div>"
            f"<div class='status-inline' style='justify-content:flex-start;margin-bottom:0.45rem'>{monitor_badge}</div>"
            f"<div class='status-emphasis'>{last_text}</div>"
            "<div class='subtle-copy' style='margin-top:0.35rem'>You can inspect the live bot here, but trades are still managed by the running bot process.</div>"
            "</div>",
            unsafe_allow_html=True,
        )
        refresh = st.button("Refresh", use_container_width=False, type="secondary")

    if refresh:
        try:
            state = load_dashboard_state()
            snapshot = state.snapshot
            recent_orders = state.recent_orders
            execution_previews = None
            last_cycle_report = state.last_cycle_report
            header_symbol_count = len(state.startup_config.symbols) if state.startup_config is not None else len(snapshot.symbols)
            header_mode_label = (
                state.startup_config.strategy_mode
                if state.startup_config is not None
                else ("snapshot only" if state.has_persisted_snapshot else "no persisted startup config")
            )
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

    st.markdown(_metrics_bar_html(snapshot), unsafe_allow_html=True)
    _render_bar_timing(bot)

    try:
        feed_status = bot.get_price_feed_status()
        if "stale" in feed_status.lower():
            dot_cls = "dot-error"
        elif "fresh" in feed_status.lower() or "received" in feed_status.lower():
            dot_cls = "dot-live"
        else:
            dot_cls = "dot-rest"
    except Exception:
        feed_status = "unavailable"
        dot_cls = "dot-error"
    st.markdown(
        _feed_health_html(feed_status, dot_cls, snapshot, getattr(bot.config, "bar_timeframe_minutes", 15) or 15),
        unsafe_allow_html=True,
    )

    status_cards: list[str] = []
    if not state.has_persisted_snapshot:
        status_cards.append(
            _status_card_html(
                "info",
                "Waiting for the first saved update",
                "Start `tradeos live` or wait for the bot to write its first cycle.",
            )
        )
    if not state.has_persisted_startup_config:
        if state.has_persisted_snapshot:
            status_cards.append(
                _status_card_html(
                    "warn",
                    "Setup details are still missing",
                    "Snapshot data exists, but startup_config.json has not been saved for this session yet.",
                )
            )
        else:
            status_cards.append(
                _status_card_html(
                    "warn",
                    "Setup details are still missing",
                    "The setup panels will stay partial until `tradeos live` writes startup_config.json.",
                )
            )
    for message in state.session_warnings:
        status_cards.append(_status_card_html("warn", "Heads up", message))
    if status_cards:
        st.markdown("".join(status_cards), unsafe_allow_html=True)

    live_tab, history_tab, config_tab, drill_tab = st.tabs(["Live", "History", "Config", "Drilldown"])

    with live_tab:
        _render_live(
            bot.storage,
            snapshot,
            recent_orders,
            execution_previews,
            state.latest_signals,
            list(state.recent_execution_activity),
            last_cycle_report,
            list(state.latest_signal_rows),
            list(state.latest_cycle_risk_checks),
            session_first_prices=state.session_first_prices,
            ml_enabled=_strategy_mode_uses_ml(
                state.startup_config.strategy_mode if state.startup_config is not None else None
            ),
            recent_narrations=list(state.recent_narrations),
            recent_near_misses=list(state.recent_near_misses),
        )

    with history_tab:
        _render_history(state, snapshot)

    with config_tab:
        _render_config_v2(state, snapshot)

    with drill_tab:
        _render_drilldown(state)


if __name__ == "__main__":
    main()

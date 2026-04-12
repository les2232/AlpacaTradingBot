from __future__ import annotations

import json
import os
import subprocess
import sys
import tkinter as tk
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import customtkinter as ctk

from desktop_app.paths import open_in_windows, resolve_latest_logs, resolve_results
from desktop_app.repo_status import load_runtime_status
from storage import BotStorage


_PROJECT_ROOT_OVERRIDE = os.getenv("ALPACA_BOT_PROJECT_ROOT")
PROJECT_ROOT = (
    Path(_PROJECT_ROOT_OVERRIDE).resolve()
    if _PROJECT_ROOT_OVERRIDE
    else Path(__file__).resolve().parent.parent
)
DB_PATH = PROJECT_ROOT / "bot_history.db"

# ── Semantic color tokens ──────────────────────────────────────────────────────
# Tuples are (light_mode, dark_mode).  Single strings apply to both.

_APP_BG = ("#E8EEF3", "#11161B")
_PANEL = ("#F6F9FC", "#182028")
_PANEL_ALT = ("#EEF4F8", "#1F2A33")
_BORDER = ("#C9D6E2", "#2B3945")
_TEXT = ("#10202D", "#EAF1F7")
_MUTED = ("#5C7285", "#9FB0BF")
_DIM = ("#758A9C", "#7E91A1")
_SECTION = ("#12212E", "#EAF1F7")

_C_TEAL = "#2EA8C9"
_C_TEAL_BG = ("#D7F2F8", "#123847")
_C_BLUE = "#60A5FA"
_C_BLUE_BG = ("#DBEAFE", "#172554")
_C_GREEN = "#24B36B"
_C_GREEN_BG = ("#DCF6E8", "#102A1E")
_C_AMBER = "#D6A043"
_C_AMBER_BG = ("#FAECCF", "#3A2A12")
_C_RED = "#D84C4C"
_C_RED_BG = ("#FBE0E0", "#3A1717")

_ROW_SEL = ("#CDEFF6", "#123847")
_ROW_HELD = ("#DDF4E7", "#163125")
_ROW_ERR = ("#F8D7D7", "#3A1717")
_ROW_BUY = ("#D6EFF4", "#173440")
_ROW_DEFAULT = ("#EAF1F6", "#22303A")
_ROW_HOVER_SEL = ("#BCE8F2", "#1A4A5B")
_ROW_HOVER_HELD = ("#CFEEDB", "#1B3C2D")
_ROW_HOVER_ERR = ("#F3C3C3", "#4A1D1D")
_ROW_HOVER_BUY = ("#C7E7EF", "#1C4250")
_ROW_HOVER_DEF = ("#DCE8F0", "#2A3944")


# ── Tooltip ───────────────────────────────────────────────────────────────────

class Tooltip:
    """Lightweight hover tooltip for any customtkinter widget.

    Usage (no need to store the return value — bindings keep it alive):
        Tooltip(some_widget, "Text shown on hover")
    """

    _DELAY_MS: int = 500

    def __init__(self, widget: Any, text: str) -> None:
        self._widget = widget
        self._text = text.strip()
        self._window: tk.Toplevel | None = None
        self._job: str | None = None

        widget.bind("<Enter>",       self._schedule, add="+")
        widget.bind("<Leave>",       self._cancel,   add="+")
        widget.bind("<ButtonPress>", self._cancel,   add="+")
        widget.bind("<Destroy>",     self._cancel,   add="+")

    def _schedule(self, _event: Any = None) -> None:
        self._cancel()
        self._job = self._widget.after(self._DELAY_MS, self._show)

    def _cancel(self, _event: Any = None) -> None:
        if self._job:
            try:
                self._widget.after_cancel(self._job)
            except Exception:
                pass
            self._job = None
        if self._window:
            try:
                self._window.destroy()
            except Exception:
                pass
            self._window = None

    def _show(self) -> None:
        if not self._text or self._window:
            return
        try:
            x = self._widget.winfo_rootx() + 14
            y = self._widget.winfo_rooty() + self._widget.winfo_height() + 6
        except Exception:
            return

        self._window = tw = tk.Toplevel(self._widget)
        tw.wm_overrideredirect(True)
        tw.wm_geometry(f"+{x}+{y}")
        tw.attributes("-topmost", True)

        is_dark = ctk.get_appearance_mode().lower() == "dark"
        bg = "#1e2a36" if is_dark else "#f8fafc"
        fg = "#d4e0ea" if is_dark else "#1e2a36"
        bd = "#2e3f50" if is_dark else "#b8ccd8"

        border_frame = tk.Frame(tw, bg=bd, padx=1, pady=1)
        border_frame.pack()
        tk.Label(
            border_frame,
            text=self._text,
            justify=tk.LEFT,
            bg=bg,
            fg=fg,
            font=("Segoe UI", 10),
            wraplength=340,
            padx=10,
            pady=6,
        ).pack()


# ── Tooltip text constants ─────────────────────────────────────────────────────

_TIP_REFRESH = (
    "Reload all data from the database: latest run, symbol snapshot, order history.\n"
    "The dashboard also auto-refreshes every 10 seconds."
)
_TIP_CONTROL_PANEL = (
    "Open the operator control panel in a separate window.\n"
    "Use it to run/stop the bot, stream live logs, view config, and run preflight checks."
)
_TIP_LOGS = (
    "Open the latest dated logs folder in Explorer.\n"
    "Contains signals.jsonl, bars.jsonl, execution.jsonl, and risk.jsonl."
)
_TIP_RESULTS = (
    "Open the results/ folder in Explorer.\n"
    "Contains backtest CSVs, experiment comparisons, and research outputs."
)
_TIP_PILL_ACCOUNT = (
    "Account mode — set by ALPACA_PAPER in .env\n"
    "\n"
    "PAPER  No real money moves. Uses Alpaca paper trading account.\n"
    "LIVE   Real Alpaca account. Real orders can execute."
)
_TIP_PILL_EXECUTION = (
    "Execution mode — set by EXECUTE_ORDERS in .env\n"
    "\n"
    "DRY RUN     Bot evaluates signals and logs decisions but submits no orders.\n"
    "LIVE ORDERS Bot submits real orders to Alpaca when signals fire."
)
_TIP_EQUITY = (
    "Total portfolio value: cash + open position market values.\n"
    "Pulled from the Alpaca account snapshot at the time of the last bot run."
)
_TIP_PNL = (
    "Change in equity since market open today.\n"
    "Green = profitable so far.  Red = currently at a loss.\n"
    "Resets each trading day."
)
_TIP_CASH = "Uninvested cash balance available for new trades."
_TIP_BUYING_POWER = (
    "Maximum capital you can deploy into new long positions.\n"
    "May differ from cash if margin is available."
)
_TIP_OPEN_POSITIONS = (
    "Number of symbols the bot is currently holding a long position in.\n"
    "Click a row in the watchlist to inspect any individual position."
)
_TIP_CHIP_ACTION = (
    "Primary state for the selected symbol."
)
_TIP_CHIP_POSITION = (
    "Current position snapshot for the selected symbol."
)
_TIP_CHIP_SIGNAL = (
    "Latest signal context for the selected symbol."
)
_TIP_SELECTED_PRICE = (
    "Latest saved snapshot price for the selected symbol."
)
_TIP_WATCHLIST = (
    "Each row shows one tracked symbol and one primary trading state."
)
_TIP_ALERTS_SECTION = (
    "Attention rail for current issues and notable state changes."
)
_TIP_POSITIONS_SECTION = (
    "Open positions rail."
)
_TIP_ORDERS_SECTION = (
    "Recent saved order events."
)


class NativeDashboardApp(ctk.CTk):
    def __init__(self) -> None:
        super().__init__()
        self.title("AlpacaTradingBot")
        self.geometry("1420x920")
        self.minsize(1200, 760)

        ctk.set_appearance_mode("System")
        ctk.set_default_color_theme("blue")

        self.storage = BotStorage(DB_PATH)
        self.auto_refresh_enabled = True
        self.selected_symbol: str | None = None
        self.symbol_rows: list[dict[str, Any]] = []
        self.symbol_buttons: dict[str, ctk.CTkButton] = {}
        self.latest_run: dict[str, Any] | None = None
        self.latest_launch: dict[str, Any] | None = None

        # Metric StringVars
        self.equity_var = ctk.StringVar(value="—")
        self.daily_pnl_var = ctk.StringVar(value="—")
        self.cash_var = ctk.StringVar(value="—")
        self.buying_power_var = ctk.StringVar(value="—")
        self.open_positions_var = ctk.StringVar(value="0")

        # Detail-panel StringVars
        self.selected_price_var = ctk.StringVar(value="—")
        self.selected_action_var = ctk.StringVar(value="—")
        self.selected_position_var = ctk.StringVar(value="—")
        self.selected_signal_var = ctk.StringVar(value="—")

        # Footer activity
        self.activity_var = ctk.StringVar(value="Ready")

        # Widget refs populated during layout (used in refresh)
        self._pnl_card: ctk.CTkLabel | None = None
        self._account_pill: ctk.CTkLabel | None = None
        self._execution_pill: ctk.CTkLabel | None = None
        self._strategy_label: ctk.CTkLabel | None = None
        self._snapshot_label: ctk.CTkLabel | None = None
        self._watchlist_count_label: ctk.CTkLabel | None = None
        self._alerts_count_label: ctk.CTkLabel | None = None
        self._positions_count_label: ctk.CTkLabel | None = None
        self._orders_count_label: ctk.CTkLabel | None = None
        self._selected_price_label: ctk.CTkLabel | None = None
        self._selected_action_card: ctk.CTkFrame | None = None
        self._selected_position_card: ctk.CTkFrame | None = None
        self._selected_signal_card: ctk.CTkFrame | None = None

        self._build_layout()
        self.refresh()
        self.after(10000, self._auto_refresh_tick)

    # ── Top-level layout ───────────────────────────────────────────────────────

    def _build_layout(self) -> None:
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(2, weight=1)

        self._build_header()
        self._build_metrics()
        self._build_body()

    # ── Header bar ────────────────────────────────────────────────────────────

    def _build_header(self) -> None:
        bar = ctk.CTkFrame(self, corner_radius=14)
        bar.grid(row=0, column=0, padx=16, pady=(16, 8), sticky="ew")
        bar.grid_columnconfigure(1, weight=1)

        # Left: title + mode pills ────────────────────────────────────────────
        left = ctk.CTkFrame(bar, fg_color="transparent")
        left.grid(row=0, column=0, padx=18, pady=14, sticky="w")

        ctk.CTkLabel(
            left,
            text="AlpacaTradingBot",
            font=ctk.CTkFont(size=22, weight="bold"),
        ).grid(row=0, column=0, padx=(0, 10), sticky="w")

        self._account_pill = ctk.CTkLabel(
            left,
            text="PAPER",
            font=ctk.CTkFont(size=11, weight="bold"),
            corner_radius=6,
            fg_color=_C_AMBER_BG,
            text_color=_C_AMBER,
            padx=8,
            pady=2,
        )
        self._account_pill.grid(row=0, column=1, padx=(0, 6), sticky="w")
        Tooltip(self._account_pill, _TIP_PILL_ACCOUNT)

        self._execution_pill = ctk.CTkLabel(
            left,
            text="DRY RUN",
            font=ctk.CTkFont(size=11, weight="bold"),
            corner_radius=6,
            fg_color=_C_BLUE_BG,
            text_color=_C_BLUE,
            padx=8,
            pady=2,
        )
        self._execution_pill.grid(row=0, column=2, sticky="w")
        Tooltip(self._execution_pill, _TIP_PILL_EXECUTION)

        # Center: strategy/symbols + snapshot freshness ───────────────────────
        mid = ctk.CTkFrame(bar, fg_color="transparent")
        mid.grid(row=0, column=1, padx=8, pady=14, sticky="w")

        self._strategy_label = ctk.CTkLabel(
            mid,
            text="—",
            font=ctk.CTkFont(size=13, weight="bold"),
            text_color=_MUTED,
        )
        self._strategy_label.grid(row=0, column=0, sticky="w")

        self._snapshot_label = ctk.CTkLabel(
            mid,
            text="Waiting for snapshot",
            font=ctk.CTkFont(size=11),
            text_color=_DIM,
        )
        self._snapshot_label.grid(row=1, column=0, sticky="w")

        ctk.CTkLabel(
            mid,
            text="auto-refreshes every 10 s",
            font=ctk.CTkFont(size=10),
            text_color=_DIM,
        ).grid(row=2, column=0, sticky="w")

        # Right: activity line + action buttons ───────────────────────────────
        right = ctk.CTkFrame(bar, fg_color="transparent")
        right.grid(row=0, column=2, padx=18, pady=14, sticky="e")

        ctk.CTkLabel(
            right,
            textvariable=self.activity_var,
            font=ctk.CTkFont(size=11),
            text_color=_DIM,
        ).grid(row=0, column=0, columnspan=4, pady=(0, 6), sticky="e")

        btn_specs: list[tuple[str, Any, bool, int, str]] = [
            ("Refresh",        self.refresh,              True,  100, _TIP_REFRESH),
            ("Control Panel",  self._open_control_panel,  False, 130, _TIP_CONTROL_PANEL),
            ("Logs",           self._open_logs,           False, 70,  _TIP_LOGS),
            ("Results",        self._open_results,        False, 80,  _TIP_RESULTS),
        ]
        for col, (label, cmd, primary, width, tip) in enumerate(btn_specs):
            btn = ctk.CTkButton(
                right,
                text=label,
                command=cmd,
                width=width,
                height=30,
                font=ctk.CTkFont(size=12),
                fg_color=None if primary else ("gray75", "gray25"),
                hover_color=None if primary else ("gray65", "gray35"),
                text_color=None if primary else _SECTION,
            )
            btn.grid(row=1, column=col, padx=(0 if col == 0 else 6, 0))
            Tooltip(btn, tip)

    # ── Metric cards row ──────────────────────────────────────────────────────

    def _build_metrics(self) -> None:
        row = ctk.CTkFrame(self, fg_color="transparent")
        row.grid(row=1, column=0, padx=16, pady=(0, 8), sticky="ew")
        for col in range(5):
            row.grid_columnconfigure(col, weight=1)

        self._metric_card(row, 0, "EQUITY",         self.equity_var,        tooltip=_TIP_EQUITY)
        self._pnl_card = self._metric_card(
            row, 1, "DAILY P/L", self.daily_pnl_var, return_label=True, tooltip=_TIP_PNL
        )
        self._metric_card(row, 2, "CASH",           self.cash_var,          tooltip=_TIP_CASH)
        self._metric_card(row, 3, "BUYING POWER",   self.buying_power_var,  tooltip=_TIP_BUYING_POWER)
        self._metric_card(row, 4, "OPEN POSITIONS", self.open_positions_var, tooltip=_TIP_OPEN_POSITIONS)

    def _metric_card(
        self,
        parent: ctk.CTkFrame,
        col: int,
        label: str,
        var: ctk.StringVar,
        *,
        return_label: bool = False,
        tooltip: str = "",
    ) -> ctk.CTkLabel | None:
        card = ctk.CTkFrame(parent, corner_radius=12)
        card.grid(row=0, column=col, padx=(0 if col == 0 else 8, 0), sticky="ew")
        if tooltip:
            Tooltip(card, tooltip)
        ctk.CTkLabel(
            card,
            text=label,
            font=ctk.CTkFont(size=10, weight="bold"),
            text_color=_MUTED,
        ).grid(row=0, column=0, padx=16, pady=(12, 2), sticky="w")
        value_lbl = ctk.CTkLabel(
            card,
            textvariable=var,
            font=ctk.CTkFont(size=22, weight="bold"),
        )
        value_lbl.grid(row=1, column=0, padx=16, pady=(0, 12), sticky="w")
        return value_lbl if return_label else None

    # ── Three-panel body ──────────────────────────────────────────────────────

    def _build_body(self) -> None:
        body = ctk.CTkFrame(self, fg_color="transparent")
        body.grid(row=2, column=0, padx=16, pady=(0, 16), sticky="nsew")
        body.grid_columnconfigure(0, weight=2)
        body.grid_columnconfigure(1, weight=3)
        body.grid_columnconfigure(2, weight=2)
        body.grid_rowconfigure(0, weight=1)

        self._build_watchlist(body)
        self._build_detail_panel(body)
        self._build_right_rail(body)

    # ── Watchlist (left) ──────────────────────────────────────────────────────

    def _build_watchlist(self, parent: ctk.CTkFrame) -> None:
        panel = ctk.CTkFrame(parent, corner_radius=14)
        panel.grid(row=0, column=0, padx=(0, 8), sticky="nsew")
        panel.grid_columnconfigure(0, weight=1)
        panel.grid_rowconfigure(1, weight=1)

        hdr = ctk.CTkFrame(panel, fg_color="transparent")
        hdr.grid(row=0, column=0, padx=14, pady=(14, 4), sticky="ew")
        hdr.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(
            hdr,
            text="WATCHLIST",
            font=ctk.CTkFont(size=10, weight="bold"),
            text_color=_MUTED,
        ).grid(row=0, column=0, sticky="w")
        Tooltip(hdr, _TIP_WATCHLIST)

        self._watchlist_count_label = ctk.CTkLabel(
            hdr,
            text="",
            font=ctk.CTkFont(size=10),
            text_color=_DIM,
        )
        self._watchlist_count_label.grid(row=0, column=1, sticky="e")

        ctk.CTkLabel(
            hdr,
            text="Primary row states: HELD, BUY, SELL, NO SIGNAL, MISSING, ERROR",
            font=ctk.CTkFont(size=9),
            text_color=_DIM,
        ).grid(row=1, column=0, columnspan=2, pady=(1, 0), sticky="w")

        ctk.CTkLabel(
            hdr,
            text="Click a symbol to inspect its current state and latest snapshot.",
            font=ctk.CTkFont(size=9),
            text_color=_DIM,
        ).grid(row=2, column=0, columnspan=2, pady=(1, 0), sticky="w")

        self.watchlist_frame = ctk.CTkScrollableFrame(
            panel, corner_radius=8, fg_color="transparent"
        )
        self.watchlist_frame.grid(row=1, column=0, padx=10, pady=(0, 10), sticky="nsew")
        self.watchlist_frame.grid_columnconfigure(0, weight=1)

    # ── Detail panel (center) ─────────────────────────────────────────────────

    def _build_detail_panel(self, parent: ctk.CTkFrame) -> None:
        panel = ctk.CTkFrame(parent, corner_radius=14)
        panel.grid(row=0, column=1, padx=(0, 8), sticky="nsew")
        panel.grid_columnconfigure(0, weight=1)
        panel.grid_rowconfigure(4, weight=2)  # detail_box expands

        # Title row: big symbol name + price ──────────────────────────────────
        title_row = ctk.CTkFrame(panel, fg_color="transparent")
        title_row.grid(row=0, column=0, padx=16, pady=(16, 0), sticky="ew")
        title_row.grid_columnconfigure(0, weight=1)

        self.selected_symbol_label = ctk.CTkLabel(
            title_row,
            text="—",
            font=ctk.CTkFont(size=34, weight="bold"),
        )
        self.selected_symbol_label.grid(row=0, column=0, sticky="w")

        self._selected_price_label = ctk.CTkLabel(
            title_row,
            textvariable=self.selected_price_var,
            font=ctk.CTkFont(size=26, weight="bold"),
            text_color=_MUTED,
        )
        self._selected_price_label.grid(row=0, column=1, sticky="e")
        Tooltip(self._selected_price_label, _TIP_SELECTED_PRICE)

        ctk.CTkLabel(
            panel,
            text="Latest snapshot",
            font=ctk.CTkFont(size=10),
            text_color=_DIM,
        ).grid(row=1, column=0, padx=16, pady=(2, 2), sticky="w")

        # Timestamp / activity line ───────────────────────────────────────────
        ctk.CTkLabel(
            panel,
            textvariable=self.activity_var,
            font=ctk.CTkFont(size=11),
            text_color=_DIM,
        ).grid(row=2, column=0, padx=16, pady=(0, 10), sticky="w")

        # Stat chips: action / position / signal ──────────────────────────────
        chips = ctk.CTkFrame(panel, fg_color="transparent")
        chips.grid(row=3, column=0, padx=16, pady=(0, 10), sticky="ew")
        for col in range(3):
            chips.grid_columnconfigure(col, weight=1)

        self._selected_action_card = self._stat_chip(
            chips, 0, "ACTION", self.selected_action_var, tooltip=_TIP_CHIP_ACTION
        )
        self._selected_position_card = self._stat_chip(
            chips, 1, "POSITION", self.selected_position_var, tooltip=_TIP_CHIP_POSITION
        )
        self._selected_signal_card = self._stat_chip(
            chips, 2, "SIGNAL", self.selected_signal_var, tooltip=_TIP_CHIP_SIGNAL
        )

        # Recent context ───────────────────────────────────────────────────────
        self._panel_section_label(
            panel, row=4,
            text="HISTORY",
            caption="Recent saved snapshots",
        )

        self.detail_box = ctk.CTkTextbox(
            panel,
            wrap="none",
            font=ctk.CTkFont(family="Consolas", size=11),
        )
        self.detail_box.grid(row=5, column=0, padx=16, pady=(0, 8), sticky="nsew")

        # System details (de-emphasized, compact) ─────────────────────────────
        self._panel_section_label(
            panel, row=6,
            text="SYSTEM",
            caption="Runtime",
        )

        self.system_box = ctk.CTkTextbox(
            panel,
            wrap="word",
            height=110,
            font=ctk.CTkFont(size=11),
            text_color=_MUTED,
        )
        self.system_box.grid(row=7, column=0, padx=16, pady=(0, 16), sticky="ew")

    # ── Right rail (alerts / positions / orders) ──────────────────────────────

    def _build_right_rail(self, parent: ctk.CTkFrame) -> None:
        panel = ctk.CTkFrame(parent, corner_radius=14)
        panel.grid(row=0, column=2, sticky="nsew")
        panel.grid_columnconfigure(0, weight=1)
        # alerts row 1 is fixed-height; positions (4) and orders (7) expand
        panel.grid_rowconfigure(4, weight=1)
        panel.grid_rowconfigure(7, weight=1)

        # Alerts ───────────────────────────────────────────────────────────────
        self._rail_section_label(
            panel, row=0,
            text="NEEDS ATTENTION",
            count_attr="_alerts_count_label",
            caption="ACTION / WATCH / INFO / OK / ISSUE attention states",
        )
        self.alerts_frame = ctk.CTkScrollableFrame(
            panel, corner_radius=8, fg_color="transparent", height=140
        )
        self.alerts_frame.grid(row=1, column=0, padx=10, pady=(0, 4), sticky="ew")
        self.alerts_frame.grid_columnconfigure(0, weight=1)
        Tooltip(self.alerts_frame, _TIP_ALERTS_SECTION)

        self._rail_divider(panel, row=2)

        # Positions ────────────────────────────────────────────────────────────
        self._rail_section_label(
            panel, row=3,
            text="OPEN POSITIONS",
            count_attr="_positions_count_label",
            caption="HELD = position open · SELL = exit bias in latest snapshot",
        )
        self.positions_frame = ctk.CTkScrollableFrame(
            panel, corner_radius=8, fg_color="transparent"
        )
        self.positions_frame.grid(row=4, column=0, padx=10, pady=(0, 4), sticky="nsew")
        self.positions_frame.grid_columnconfigure(0, weight=1)
        Tooltip(self.positions_frame, _TIP_POSITIONS_SECTION)

        self._rail_divider(panel, row=5)

        # Orders ───────────────────────────────────────────────────────────────
        self._rail_section_label(
            panel, row=6,
            text="RECENT ORDERS",
            count_attr="_orders_count_label",
            caption="BUY/SELL show order side · MISSING means limited order context",
        )
        self.orders_frame = ctk.CTkScrollableFrame(
            panel, corner_radius=8, fg_color="transparent"
        )
        self.orders_frame.grid(row=7, column=0, padx=10, pady=(0, 10), sticky="nsew")
        self.orders_frame.grid_columnconfigure(0, weight=1)
        Tooltip(self.orders_frame, _TIP_ORDERS_SECTION)

    # ── Widget-building helpers ───────────────────────────────────────────────

    def _stat_chip(
        self,
        parent: ctk.CTkFrame,
        col: int,
        label: str,
        var: ctk.StringVar,
        *,
        tooltip: str = "",
    ) -> ctk.CTkFrame:
        card = ctk.CTkFrame(parent, corner_radius=10)
        card.grid(row=0, column=col, padx=(0 if col == 0 else 8, 0), sticky="ew")
        if tooltip:
            Tooltip(card, tooltip)
        ctk.CTkLabel(
            card,
            text=label,
            font=ctk.CTkFont(size=10, weight="bold"),
            text_color=_MUTED,
        ).grid(row=0, column=0, padx=10, pady=(10, 2), sticky="w")
        ctk.CTkLabel(
            card,
            textvariable=var,
            font=ctk.CTkFont(size=16, weight="bold"),
            wraplength=160,
            justify="left",
        ).grid(row=1, column=0, padx=10, pady=(0, 10), sticky="w")
        return card

    def _panel_section_label(
        self,
        parent: ctk.CTkFrame,
        row: int,
        text: str,
        caption: str = "",
    ) -> None:
        frame = ctk.CTkFrame(parent, fg_color="transparent")
        frame.grid(row=row, column=0, padx=16, pady=(6, 2), sticky="ew")
        ctk.CTkLabel(
            frame,
            text=text,
            font=ctk.CTkFont(size=10, weight="bold"),
            text_color=_MUTED,
        ).grid(row=0, column=0, sticky="w")
        if caption:
            ctk.CTkLabel(
                frame,
                text=caption,
                font=ctk.CTkFont(size=9),
                text_color=_DIM,
            ).grid(row=1, column=0, sticky="w")

    def _rail_section_label(
        self,
        parent: ctk.CTkFrame,
        row: int,
        text: str,
        count_attr: str,
        caption: str = "",
    ) -> None:
        outer = ctk.CTkFrame(parent, fg_color="transparent")
        outer.grid(row=row, column=0, padx=14, pady=(12, 2), sticky="ew")
        outer.grid_columnconfigure(0, weight=1)

        hdr = ctk.CTkFrame(outer, fg_color="transparent")
        hdr.grid(row=0, column=0, sticky="ew")
        hdr.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(
            hdr,
            text=text,
            font=ctk.CTkFont(size=10, weight="bold"),
            text_color=_MUTED,
        ).grid(row=0, column=0, sticky="w")

        count_lbl = ctk.CTkLabel(
            hdr,
            text="",
            font=ctk.CTkFont(size=10),
            text_color=_DIM,
        )
        count_lbl.grid(row=0, column=1, sticky="e")
        setattr(self, count_attr, count_lbl)

        if caption:
            ctk.CTkLabel(
                outer,
                text=caption,
                font=ctk.CTkFont(size=9),
                text_color=_DIM,
            ).grid(row=1, column=0, sticky="w")

    def _rail_divider(self, parent: ctk.CTkFrame, row: int) -> None:
        ctk.CTkFrame(parent, height=1, fg_color=("gray80", "gray25")).grid(
            row=row, column=0, padx=14, pady=2, sticky="ew"
        )

    # ── Refresh ───────────────────────────────────────────────────────────────

    def refresh(self) -> None:
        runtime_status = load_runtime_status(PROJECT_ROOT)
        self.latest_run = self.storage.get_latest_run()
        self.symbol_rows = self.storage.get_latest_symbol_snapshot()
        recent_orders = self.storage.get_order_history(limit=12)
        self.latest_launch = _load_latest_launch(PROJECT_ROOT)

        if self.symbol_rows and (
            self.selected_symbol not in {row["symbol"] for row in self.symbol_rows}
        ):
            self.selected_symbol = str(self.symbol_rows[0]["symbol"])
        elif not self.symbol_rows:
            self.selected_symbol = None

        # Header updates
        if self._strategy_label:
            self._strategy_label.configure(
                text=(
                    f"{runtime_status.strategy_mode}"
                    f"  ·  {runtime_status.symbols_text}"
                    f"  ·  {runtime_status.timeframe_text}"
                )
            )
        if self._snapshot_label:
            self._snapshot_label.configure(
                text=_build_snapshot_text(self.latest_run, self.latest_launch)
            )
        self._update_mode_pills(runtime_status)

        # Metric cards
        run = self.latest_run
        self.equity_var.set(_fmt_money(run["equity"]) if run else "—")
        pnl = run["daily_pnl"] if run else None
        self.daily_pnl_var.set(_fmt_money(pnl) if pnl is not None else "—")
        if self._pnl_card is not None:
            if pnl is not None:
                self._pnl_card.configure(text_color=_C_GREEN if pnl >= 0 else _C_RED)
            else:
                self._pnl_card.configure(text_color=_DIM)
        self.cash_var.set(_fmt_money(run["cash"]) if run else "—")
        self.buying_power_var.set(_fmt_money(run["buying_power"]) if run else "—")
        self.open_positions_var.set(
            str(sum(1 for row in self.symbol_rows if _is_truthy(row.get("holding"))))
        )

        # Panels
        self._rebuild_watchlist()
        self._render_selected_symbol()
        self._rebuild_alerts(self.latest_run, self.symbol_rows, self.latest_launch)
        self._rebuild_positions(self.symbol_rows)
        self._rebuild_orders(recent_orders)

        self._replace_text(
            self.system_box,
            _build_system_text(runtime_status, self.latest_run, self.latest_launch),
        )
        self.activity_var.set(f"Updated {datetime.now().strftime('%H:%M:%S')}")

    def _update_mode_pills(self, runtime_status: Any) -> None:
        if self._account_pill is None or self._execution_pill is None:
            return
        account = runtime_status.account_mode.upper()
        execution = runtime_status.execution_mode.upper()

        if account == "LIVE":
            self._account_pill.configure(text=account, fg_color=_C_RED_BG, text_color=_C_RED)
        else:
            self._account_pill.configure(text=account, fg_color=_C_AMBER_BG, text_color=_C_AMBER)

        if "DRY" in execution or "DISABLED" in execution:
            self._execution_pill.configure(text=execution, fg_color=_C_BLUE_BG, text_color=_C_BLUE)
        else:
            self._execution_pill.configure(text=execution, fg_color=_C_GREEN_BG, text_color=_C_GREEN)

    # ── Watchlist rebuild ─────────────────────────────────────────────────────

    def _rebuild_watchlist(self) -> None:
        for child in self.watchlist_frame.winfo_children():
            child.destroy()
        self.symbol_buttons.clear()

        count = len(self.symbol_rows)
        if self._watchlist_count_label:
            self._watchlist_count_label.configure(
                text=f"{count} symbol{'s' if count != 1 else ''}"
            )

        if not self.symbol_rows:
            ctk.CTkLabel(
                self.watchlist_frame,
                text="No snapshot yet.\nRun or refresh the bot.",
                justify="left",
                text_color=_DIM,
                font=ctk.CTkFont(size=12),
            ).grid(row=0, column=0, padx=8, pady=16, sticky="w")
            return

        for idx, row in enumerate(self.symbol_rows):
            symbol = str(row["symbol"])
            is_sel = symbol == self.selected_symbol
            state_kind, state_label = _trading_state_details(row)
            btn = ctk.CTkButton(
                self.watchlist_frame,
                text=_watchlist_row_text(row),
                anchor="w",
                height=62,
                corner_radius=10,
                command=lambda s=symbol: self._select_symbol(s),
                fg_color=self._row_color(row, selected=is_sel),
                hover_color=self._row_hover(row),
                font=ctk.CTkFont(family="Consolas", size=12),
            )
            btn.grid(row=idx, column=0, padx=4, pady=(0, 6), sticky="ew")
            self.symbol_buttons[symbol] = btn
            Tooltip(btn, _watchlist_tooltip_text(row))
            fg_color, text_color, _ = _state_colors(state_kind)
            ctk.CTkLabel(
                btn,
                text=state_label,
                font=ctk.CTkFont(size=10, weight="bold"),
                corner_radius=999,
                fg_color=fg_color,
                text_color=text_color,
                padx=8,
                pady=2,
            ).place(relx=1.0, x=-10, y=10, anchor="ne")

    def _row_color(
        self, row: dict[str, Any], *, selected: bool
    ) -> tuple[str, str] | str:
        if selected:
            return _ROW_SEL
        state_kind, _state_label = _trading_state_details(row)
        if state_kind == "error":
            return _ROW_ERR
        if state_kind == "held":
            return _ROW_HELD
        if state_kind == "buy":
            return _ROW_BUY
        return _ROW_DEFAULT

    def _row_hover(self, row: dict[str, Any]) -> tuple[str, str] | str:
        state_kind, _state_label = _trading_state_details(row)
        if state_kind == "error":
            return _ROW_HOVER_ERR
        if state_kind == "held":
            return _ROW_HOVER_HELD
        if state_kind == "buy":
            return _ROW_HOVER_BUY
        return _ROW_HOVER_DEF

    # ── Detail panel render ───────────────────────────────────────────────────

    def _render_selected_symbol(self) -> None:
        if not self.selected_symbol:
            self.selected_symbol_label.configure(text="—")
            self.selected_price_var.set("—")
            self.selected_action_var.set("NO DATA")
            self.selected_position_var.set("NO DATA")
            self.selected_signal_var.set("NO DATA")
            self._replace_text(self.detail_box, "No symbol selected.")
            return

        row = next(
            (r for r in self.symbol_rows if str(r["symbol"]) == self.selected_symbol),
            None,
        )
        if row is None:
            self._replace_text(self.detail_box, "Symbol no longer in latest snapshot.")
            return

        history = self.storage.get_symbol_history(self.selected_symbol, limit=12)
        self.selected_symbol_label.configure(text=str(row["symbol"]))
        self.selected_price_var.set(_fmt_money_or_na(row.get("price")))
        state_kind = _primary_state_kind(row)
        state_label = _primary_state_label(row)
        self.selected_action_var.set(state_label)
        self.selected_position_var.set(_position_summary_explained(row))
        self.selected_signal_var.set(_signal_summary_explained(row))
        if self._selected_action_card is not None:
            self._selected_action_card.configure(fg_color=_state_colors(state_kind)[0])
            Tooltip(self._selected_action_card, _selected_action_tooltip_text(row))
        if self._selected_position_card is not None:
            pos_kind = "held" if _is_truthy(row.get("holding")) else "no_data"
            self._selected_position_card.configure(fg_color=_state_colors(pos_kind)[0])
            Tooltip(self._selected_position_card, _selected_position_tooltip_text(row))
        if self._selected_signal_card is not None:
            signal_kind = state_kind if state_kind in {"buy", "sell", "error"} else "no_data"
            self._selected_signal_card.configure(fg_color=_state_colors(signal_kind)[0])
            Tooltip(self._selected_signal_card, _selected_signal_tooltip_text(row))
        if self._selected_price_label is not None:
            Tooltip(self._selected_price_label, _selected_price_tooltip_text(row))
        self._replace_text(self.detail_box, _format_symbol_detail_explained(row, history))

    def _select_symbol(self, symbol: str) -> None:
        self.selected_symbol = symbol
        self._rebuild_watchlist()
        self._render_selected_symbol()
        self.activity_var.set(f"Selected {symbol}")

    # ── Right-rail row builders ───────────────────────────────────────────────

    def _rebuild_alerts(
        self,
        latest_run: dict[str, Any] | None,
        rows: list[dict[str, Any]],
        latest_launch: dict[str, Any] | None,
    ) -> None:
        alerts = _collect_alerts(latest_run, rows, latest_launch)
        _clear_frame(self.alerts_frame)
        if self._alerts_count_label:
            self._alerts_count_label.configure(
                text=f"{len(alerts)} item{'s' if len(alerts) != 1 else ''}"
            )

        if not alerts:
            _empty_label(self.alerts_frame, 0, "All clear")
            return

        color_map = {"error": _C_RED, "warn": _C_AMBER, "info": _C_BLUE, "ok": _C_GREEN}
        tip_map = {
            "error": "Red — action likely needed. Check logs or the bot's error output.",
            "warn":  "Amber — worth monitoring. May resolve on the next cycle.",
            "info":  "Blue — informational. The bot is acting on this signal now.",
            "ok":    "Green — this is normal and expected. No action required.",
        }
        for idx, (severity, text) in enumerate(alerts):
            badge_kind = {
                "error": "error",
                "warn": "no_data",
                "info": "buy",
                "ok": "held",
            }.get(severity, "no_data")
            badge_text = {
                "error": "ERROR",
                "warn": "NO DATA",
                "info": "BUY",
                "ok": "HELD",
            }.get(severity, "NO DATA")
            _stripe_row(
                self.alerts_frame, idx,
                color_map.get(severity, "#64748b"),
                text,
                subtext=_alert_subtext(severity, text),
                wrap=240,
                tooltip=_alert_tooltip_text(severity, text),
                badge_text=badge_text,
                badge_kind=badge_kind,
            )

    def _rebuild_positions(self, rows: list[dict[str, Any]]) -> None:
        held = [r for r in rows if _is_truthy(r.get("holding"))]
        _clear_frame(self.positions_frame)
        if self._positions_count_label:
            self._positions_count_label.configure(text=f"{len(held)} open")

        if not held:
            _empty_label(self.positions_frame, 0, "No open positions")
            return

        for idx, row in enumerate(held):
            sym = str(row.get("symbol", "?"))
            detail = _position_row_subtext(row)
            qty = _fmt_number(row.get("quantity"))
            value = _fmt_money_or_na(row.get("market_value"))
            mins = _fmt_number(row.get("holding_minutes"), digits=0)
            detail = f"{qty} shares  ·  {value}  ·  {mins} min held"
            detail = _position_row_subtext(row)
            badge_kind = "sell" if str(row.get("action") or "").upper() == "SELL" else "held"
            tip = _position_tooltip_text(row)
            _stripe_row(
                self.positions_frame,
                idx,
                _state_colors(badge_kind)[2],
                sym,
                subtext=detail,
                tooltip=tip,
                badge_text="SELL" if badge_kind == "sell" else "HELD",
                badge_kind=badge_kind,
            )

    def _rebuild_orders(self, rows: list[dict[str, Any]]) -> None:
        _clear_frame(self.orders_frame)
        shown = min(len(rows), 10)
        if self._orders_count_label:
            self._orders_count_label.configure(text=f"{shown} shown" if shown else "")

        if not rows:
            _empty_label(self.orders_frame, 0, "No saved orders yet")
            return

        for idx, row in enumerate(rows[:10]):
            side = str(row.get("side") or "").upper()
            badge_kind = "buy" if side == "BUY" else "sell" if side == "SELL" else "no_data"
            stripe_color = _state_colors(badge_kind)[2]
            sym = str(row.get("symbol", "?"))
            top = f"{sym}  {side or 'ORDER'}  Â·  {_friendly_order_status(row.get('status'))}"
            qty_val = (
                row.get("filled_qty")
                if row.get("filled_qty") not in (None, "")
                else row.get("qty")
            )
            top = f"{sym}  {side}  ·  {row.get('status', '?')}"
            detail = (
                f"qty {_fmt_number(qty_val)}  ·  "
                f"avg {_fmt_number(row.get('filled_avg_price'))}  ·  "
                f"{str(row.get('observed_at_utc', ''))[:16]}"
            )
            detail = _order_row_subtext(row, qty_val)
            top = f"{sym}  {side or 'ORDER'} | {_friendly_order_status(row.get('status'))}"
            tip = _order_tooltip_text(row)
            _stripe_row(
                self.orders_frame,
                idx,
                stripe_color,
                top,
                subtext=detail,
                tooltip=tip,
                badge_text=side or "NO DATA",
                badge_kind=badge_kind,
            )

    # ── Generic utilities ─────────────────────────────────────────────────────

    def _replace_text(self, widget: ctk.CTkTextbox, content: str) -> None:
        widget.configure(state="normal")
        widget.delete("1.0", "end")
        widget.insert("1.0", content)
        widget.configure(state="disabled")

    def _open_control_panel(self) -> None:
        env = os.environ.copy()
        env["ALPACA_BOT_PROJECT_ROOT"] = str(PROJECT_ROOT)
        if getattr(sys, "frozen", False):
            command = [sys.executable, "control-panel"]
        else:
            command = [sys.executable, "-m", "alpaca_trading_bot", "control-panel"]
        subprocess.Popen(command, cwd=str(PROJECT_ROOT), env=env)
        self.activity_var.set("Opened control panel in a separate window")

    def _open_logs(self) -> None:
        try:
            resolution = resolve_latest_logs(PROJECT_ROOT)
            open_in_windows(resolution.path)
            self.activity_var.set(f"Opened logs: {resolution.path.name}")
        except FileNotFoundError as exc:
            self.activity_var.set(f"Logs not found: {exc}")

    def _open_results(self) -> None:
        try:
            resolution = resolve_results(PROJECT_ROOT)
            open_in_windows(resolution.path)
            self.activity_var.set("Opened results folder")
        except FileNotFoundError as exc:
            self.activity_var.set(f"Results not found: {exc}")

    def _auto_refresh_tick(self) -> None:
        if self.auto_refresh_enabled:
            try:
                self.refresh()
            except Exception as exc:
                self.activity_var.set(f"Auto-refresh failed: {exc}")
        self.after(10000, self._auto_refresh_tick)


# ── Shared row-builder functions (module-level, stateless) ────────────────────

def _clear_frame(frame: ctk.CTkScrollableFrame) -> None:
    for child in frame.winfo_children():
        child.destroy()


def _primary_state_kind(row: dict[str, Any]) -> str:
    if row.get("error"):
        return "error"
    action = str(row.get("action") or "").upper()
    if _is_truthy(row.get("holding")):
        return "held"
    if action == "BUY":
        return "buy"
    if action == "SELL":
        return "sell"
    if row.get("price") in (None, "") or row.get("sma") in (None, ""):
        return "missing"
    return "no_signal"


def _primary_state_label(row: dict[str, Any]) -> str:
    return {
        "held": "HELD",
        "buy": "BUY",
        "sell": "SELL",
        "error": "ERROR",
        "flat": "FLAT",
        "no_signal": "NO SIGNAL",
        "missing": "MISSING",
        "stale": "STALE",
        "market_closed": "MARKET CLOSED",
        "no_data": "NO DATA",
    }[_primary_state_kind(row)]


def _state_colors(kind: str) -> tuple[Any, Any, str]:
    palette = {
        "held": (_C_GREEN_BG, _C_GREEN, _C_GREEN),
        "buy": (_C_BLUE_BG, _C_BLUE, _C_BLUE),
        "sell": (_C_AMBER_BG, _C_AMBER, _C_AMBER),
        "error": (_C_RED_BG, _C_RED, _C_RED),
        "flat": (_C_BLUE_BG, _DIM, _DIM),
        "no_signal": (_C_BLUE_BG, _DIM, _DIM),
        "missing": (_C_RED_BG, _C_RED, _C_RED),
        "stale": (_C_AMBER_BG, _C_AMBER, _C_AMBER),
        "market_closed": (_C_BLUE_BG, _DIM, _DIM),
        "no_data": (_C_BLUE_BG, _DIM, _DIM),
    }
    return palette.get(kind, (_C_BLUE_BG, _DIM, _DIM))


def _empty_label(parent: ctk.CTkScrollableFrame, row: int, text: str) -> None:
    ctk.CTkLabel(
        parent,
        text=text,
        font=ctk.CTkFont(size=12),
        text_color=_DIM,
    ).grid(row=row, column=0, padx=8, pady=12, sticky="w")


def _stripe_row(
    parent: ctk.CTkScrollableFrame,
    row_idx: int,
    stripe_color: str,
    text: str,
    *,
    subtext: str = "",
    wrap: int = 260,
    tooltip: str = "",
    badge_text: str = "",
    badge_kind: str = "no_data",
) -> None:
    """A card row with a colored left stripe, a bold title, and optional subtext."""
    frame = ctk.CTkFrame(parent, fg_color=("gray90", "gray18"), corner_radius=8)
    frame.grid(row=row_idx, column=0, padx=4, pady=(0, 5), sticky="ew")
    frame.grid_columnconfigure(1, weight=1)

    if tooltip:
        Tooltip(frame, tooltip)

    ctk.CTkFrame(frame, width=4, corner_radius=2, fg_color=stripe_color).grid(
        row=0, column=0, padx=(8, 0), pady=8, sticky="ns"
    )

    content = ctk.CTkFrame(frame, fg_color="transparent")
    content.grid(row=0, column=1, padx=(8, 10), pady=(7, 7), sticky="ew")
    content.grid_columnconfigure(0, weight=1)

    header = ctk.CTkFrame(content, fg_color="transparent")
    header.grid(row=0, column=0, sticky="ew")
    header.grid_columnconfigure(0, weight=1)

    ctk.CTkLabel(
        header,
        text=text,
        font=ctk.CTkFont(size=12, weight="bold"),
        wraplength=wrap,
        justify="left",
        anchor="w",
    ).grid(row=0, column=0, sticky="w")

    if badge_text:
        fg_color, text_color, _ = _state_colors(badge_kind)
        ctk.CTkLabel(
            header,
            text=badge_text,
            font=ctk.CTkFont(size=10, weight="bold"),
            corner_radius=999,
            fg_color=fg_color,
            text_color=text_color,
            padx=8,
            pady=2,
        ).grid(row=0, column=1, padx=(8, 0), sticky="e")

    if subtext:
        ctk.CTkLabel(
            content,
            text=subtext,
            font=ctk.CTkFont(size=11),
            text_color=_MUTED,
            wraplength=wrap,
            justify="left",
            anchor="w",
        ).grid(row=1, column=0, pady=(1, 0), sticky="w")


# ── Tooltip text builders (dynamic, per-row) ──────────────────────────────────

def _watchlist_tooltip_text(row: dict[str, Any]) -> str:
    parts: list[str] = []

    action = str(row.get("action") or "HOLD").upper()
    action_desc = {
        "BUY":  "BUY — strategy wants to open a long position this cycle",
        "SELL": "SELL — strategy wants to close the current position",
        "HOLD": "HOLD — no action this cycle; bot is watching",
    }.get(action, f"{action} — unrecognised action")
    parts.append(action_desc)

    held = _is_truthy(row.get("holding"))
    if held:
        qty = _fmt_number(row.get("quantity"))
        value = _fmt_money_or_na(row.get("market_value"))
        mins = _fmt_number(row.get("holding_minutes"), digits=0)
        parts.append(f"Holding {qty} shares  ·  {value}  ·  {mins} min")
    else:
        parts.append("Flat — no active position")

    if row.get("price") not in (None, "") and row.get("sma") not in (None, ""):
        try:
            p, s = float(row["price"]), float(row["sma"])
            if s != 0:
                pct = ((p - s) / s) * 100
                direction = "above" if pct >= 0 else "below"
                parts.append(f"Price is {abs(pct):.2f}% {direction} the SMA")
        except (TypeError, ValueError):
            pass

    if row.get("ml_probability_up") not in (None, ""):
        try:
            ml = float(row["ml_probability_up"])
            conf = row.get("ml_confidence")
            training = row.get("ml_training_rows")
            ml_line = f"ML probability up: {ml:.2f}"
            if conf not in (None, ""):
                ml_line += f"  (confidence {float(conf):.2f})"
            if training not in (None, ""):
                ml_line += f"  ·  {training} training rows"
            parts.append(ml_line)
        except (TypeError, ValueError):
            pass

    if row.get("error"):
        parts.append(f"Error: {row['error']}")

    return "\n".join(parts)


def _position_tooltip_text(row: dict[str, Any]) -> str:
    parts: list[str] = []
    parts.append("Active long position — the bot entered on a BUY signal.")
    action = str(row.get("action") or "HOLD").upper()
    if action == "SELL":
        parts.append("Sell signal is active — the bot may exit on the next cycle.")
    elif action == "HOLD":
        parts.append("No exit signal yet — bot is holding.")
    if row.get("ml_probability_up") not in (None, ""):
        try:
            ml = float(row["ml_probability_up"])
            parts.append(f"Current ML probability up: {ml:.2f}")
        except (TypeError, ValueError):
            pass
    if row.get("error"):
        parts.append(f"Note: last cycle had an error — {row['error']}")
    return "\n".join(parts)


def _order_tooltip_text(row: dict[str, Any]) -> str:
    side = str(row.get("side") or "").upper()
    status = str(row.get("status") or "").lower()

    side_desc = {
        "BUY":  "BUY — bot entered a long position.",
        "SELL": "SELL — bot exited a position.",
    }.get(side, f"{side} order")

    status_desc = {
        "filled":           "filled — order fully executed at the average price shown.",
        "partially_filled": "partially_filled — only part of the requested quantity was executed.",
        "canceled":         "canceled — order was cancelled before it could fill.",
        "rejected":         "rejected — Alpaca rejected the order (check buying power or symbol status).",
        "pending_new":      "pending_new — order submitted but not yet acknowledged by the exchange.",
        "new":              "new — order accepted and working at the exchange.",
    }.get(status, f"status: {status}")

    return f"{side_desc}\n{status_desc}"


def _selected_price_tooltip_text(row: dict[str, Any]) -> str:
    price = _fmt_money_or_na(row.get("price"))
    sma = _fmt_number(row.get("sma"))
    return (
        f"Latest saved snapshot price: {price}\n"
        f"SMA reference: {sma}\n"
        "Use this with the SIGNAL chip to see whether price is above or below trend."
    )


def _selected_action_tooltip_text(row: dict[str, Any]) -> str:
    action = str(row.get("action") or "HOLD").upper()
    if action == "BUY":
        meaning = "BUY means the strategy wants to open a long position on this cycle."
    elif action == "SELL":
        meaning = "SELL means the strategy wants to exit the current long position."
    elif action == "HOLD":
        meaning = "HOLD means the bot is watching this symbol but not changing exposure now."
    else:
        meaning = f"{action} is the raw action saved for this cycle."
    held_context = (
        "This symbol is already held." if _is_truthy(row.get("holding")) else "No active position is open."
    )
    return f"{meaning}\n{held_context}"


def _selected_position_tooltip_text(row: dict[str, Any]) -> str:
    if not _is_truthy(row.get("holding")):
        return (
            "Flat means the bot does not currently own this symbol.\n"
            "A BUY action would open a new position if execution mode allows orders."
        )
    return (
        f"Held position: {_fmt_number(row.get('quantity'))} shares\n"
        f"Market value: {_fmt_money_or_na(row.get('market_value'))}\n"
        f"Held for about {_fmt_number(row.get('holding_minutes'), digits=0)} minutes"
    )


def _selected_signal_tooltip_text(row: dict[str, Any]) -> str:
    parts = ["Signal summary for the current snapshot."]
    if row.get("price") not in (None, "") and row.get("sma") not in (None, ""):
        try:
            price = float(row["price"])
            sma = float(row["sma"])
            if sma != 0:
                pct = ((price - sma) / sma) * 100
                parts.append(
                    f"Price is {abs(pct):.2f}% {'above' if pct >= 0 else 'below'} the SMA trend line."
                )
        except (TypeError, ValueError):
            pass
    if row.get("ml_probability_up") not in (None, ""):
        try:
            ml = float(row["ml_probability_up"])
            parts.append(
                f"ML probability up is {ml:.2f}. Closer to 1.00 is more bullish; closer to 0.00 is more bearish."
            )
        except (TypeError, ValueError):
            pass
    if len(parts) == 1:
        parts.append("Not enough saved signal context for this row yet.")
    return "\n".join(parts)


def _alert_subtext(severity: str, text: str) -> str:
    hints = {
        "error": "Action likely needed. Check logs, broker status, or symbol-specific errors.",
        "warn": "Monitor this before trusting fresh trades or automation.",
        "info": "Informational summary of current bot activity.",
        "ok": "Healthy status. No operator action is usually required.",
    }
    if "stale" in text.lower():
        return "The latest snapshot is aging out, so the screen may not reflect the current market."
    return hints.get(severity, "")


def _friendly_order_status(status: object) -> str:
    normalized = str(status or "").strip().lower()
    return {
        "filled": "filled",
        "partially_filled": "partial fill",
        "canceled": "canceled",
        "rejected": "rejected",
        "pending_new": "pending submit",
        "new": "working",
    }.get(normalized, normalized or "unknown")


def _signal_summary_explained(row: dict[str, Any]) -> str:
    parts: list[str] = []
    if row.get("price") not in (None, "") and row.get("sma") not in (None, ""):
        try:
            price, sma = float(row["price"]), float(row["sma"])
            if sma != 0:
                parts.append(f"vs SMA {((price - sma) / sma) * 100:+.2f}%")
        except (TypeError, ValueError):
            pass
    if row.get("ml_probability_up") not in (None, ""):
        try:
            parts.append(f"ML up {float(row['ml_probability_up']):.2f}")
        except (TypeError, ValueError):
            pass
    return " | ".join(parts) if parts else "â€”"


def _format_symbol_detail_explained(row: dict[str, Any], history: list[dict[str, Any]]) -> str:
    lines = [
        "  What this panel shows:",
        "  - action = strategy decision for the latest saved cycle",
        "  - sma = moving-average reference used by the signal",
        "  - ml_prob_up = model probability of an upward next move",
        "",
        f"  action            {row.get('action') or 'â€”'}",
        f"  price             {_fmt_number(row.get('price'))}",
        f"  sma               {_fmt_number(row.get('sma'))}",
        f"  holding           {'Yes' if _is_truthy(row.get('holding')) else 'No'}",
        f"  holding_minutes   {_fmt_number(row.get('holding_minutes'), digits=0)}",
        f"  quantity          {_fmt_number(row.get('quantity'))}",
        f"  market_value      {_fmt_money_or_na(row.get('market_value'))}",
        f"  ml_prob_up        {_fmt_number(row.get('ml_probability_up'))}",
        f"  ml_confidence     {_fmt_number(row.get('ml_confidence'))}",
        f"  ml_training_rows  {row.get('ml_training_rows') if row.get('ml_training_rows') not in (None, '') else 'â€”'}",
        f"  error             {row.get('error') or 'None'}",
        "",
        "  recent history:",
    ]
    if not history:
        lines.append("  No saved symbol history yet.")
    else:
        for item in history[-8:]:
            ts = str(item.get("timestamp_utc", ""))[:16]
            price = _fmt_number(item.get("price"))
            sma = _fmt_number(item.get("sma"))
            action = str(item.get("action") or "â€”").rjust(5)
            lines.append(f"  {ts}  {price:>9}  sma {sma:>9}  {action}")
    return "\n".join(lines)


# ── Pure data / formatting functions ──────────────────────────────────────────

def _fmt_money(value: float | int | None) -> str:
    if value is None:
        return "—"
    return f"${float(value):,.2f}"


def _fmt_money_or_na(value: object) -> str:
    if value in (None, ""):
        return "—"
    return _fmt_money(float(value))  # type: ignore[arg-type]


def _fmt_number(value: object, digits: int = 2) -> str:
    if value in (None, ""):
        return "—"
    return f"{float(value):.{digits}f}"  # type: ignore[arg-type]


def _is_truthy(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    return str(value).strip().lower() in {"1", "true", "yes"}


def _format_symbol_detail(row: dict[str, Any], history: list[dict[str, Any]]) -> str:
    lines = [
        f"  action            {row.get('action') or '—'}",
        f"  price             {_fmt_number(row.get('price'))}",
        f"  sma               {_fmt_number(row.get('sma'))}",
        f"  holding           {'Yes' if _is_truthy(row.get('holding')) else 'No'}",
        f"  holding_minutes   {_fmt_number(row.get('holding_minutes'), digits=0)}",
        f"  quantity          {_fmt_number(row.get('quantity'))}",
        f"  market_value      {_fmt_money_or_na(row.get('market_value'))}",
        f"  ml_prob_up        {_fmt_number(row.get('ml_probability_up'))}",
        f"  ml_confidence     {_fmt_number(row.get('ml_confidence'))}",
        f"  ml_training_rows  {row.get('ml_training_rows') if row.get('ml_training_rows') not in (None, '') else '—'}",
        f"  error             {row.get('error') or 'None'}",
        "",
        "  ── recent history ──────────────────────────────────────",
    ]
    if not history:
        lines.append("  No saved symbol history yet.")
    else:
        for item in history[-8:]:
            ts = str(item.get("timestamp_utc", ""))[:16]
            price = _fmt_number(item.get("price"))
            sma = _fmt_number(item.get("sma"))
            action = str(item.get("action") or "—").rjust(5)
            lines.append(f"  {ts}  {price:>9}  sma {sma:>9}  {action}")
    return "\n".join(lines)


def _collect_alerts(
    latest_run: dict[str, Any] | None,
    rows: list[dict[str, Any]],
    latest_launch: dict[str, Any] | None,
) -> list[tuple[str, str]]:
    alerts: list[tuple[str, str]] = []

    if latest_run is None:
        alerts.append(("warn", "No saved bot snapshot yet. Run the bot to populate."))
    else:
        ts = _parse_utc_timestamp(str(latest_run["timestamp_utc"]))
        if ts is not None:
            age = (datetime.now(timezone.utc) - ts).total_seconds() / 60
            if age > 20:
                alerts.append(("warn", f"Snapshot stale — {age:.0f} min old"))
        if latest_run.get("kill_switch_triggered"):
            alerts.append(("error", "Kill switch active — new orders blocked"))

    error_rows = [r for r in rows if r.get("error")]
    buy_rows = [r for r in rows if str(r.get("action") or "").upper() == "BUY"]
    held_rows = [r for r in rows if _is_truthy(r.get("holding"))]

    if error_rows:
        syms = ", ".join(str(r["symbol"]) for r in error_rows[:4])
        alerts.append(("error", f"{len(error_rows)} symbol error{'s' if len(error_rows) > 1 else ''}: {syms}"))
    if buy_rows:
        syms = ", ".join(str(r["symbol"]) for r in buy_rows[:4])
        alerts.append(("info", f"{len(buy_rows)} BUY signal{'s' if len(buy_rows) > 1 else ''}: {syms}"))
    if held_rows:
        n = len(held_rows)
        alerts.append(("ok", f"{n} position{'s' if n > 1 else ''} being held"))
    if latest_launch and str(latest_launch.get("ready_for_live", "")).lower() == "false":
        alerts.append(("warn", "Last launch: not ready for live"))

    return alerts


def _build_snapshot_text(
    latest_run: dict[str, Any] | None,
    latest_launch: dict[str, Any] | None,
) -> str:
    parts: list[str] = []
    if latest_run:
        parts.append(f"Snapshot {str(latest_run['timestamp_utc'])[:16]}")
        if latest_run.get("kill_switch_triggered"):
            parts.append("Kill switch active")
    if latest_launch:
        lt = str(latest_launch.get("timestamp", ""))[:16]
        if lt:
            parts.append(f"Launched {lt}")
    return "  ·  ".join(parts) if parts else "No snapshot yet"


def _build_system_text(
    runtime_status: Any,
    latest_run: dict[str, Any] | None,
    latest_launch: dict[str, Any] | None,
) -> str:
    lines = [
        (
            f"Strategy: {runtime_status.strategy_mode}"
            f"  ·  Symbols: {runtime_status.symbols_text}"
            f"  ·  {runtime_status.timeframe_text}"
        ),
        f"Account: {runtime_status.account_mode}  ·  Execution: {runtime_status.execution_mode}",
        f"Config: {runtime_status.runtime_source}",
        f"DB: {DB_PATH}",
    ]
    if latest_run:
        ks = "ACTIVE" if latest_run.get("kill_switch_triggered") else "clear"
        lines.append(
            f"Kill switch: {ks}"
            f"  ·  Equity: {_fmt_money(latest_run.get('equity'))}"
            f"  ·  P/L: {_fmt_money(latest_run.get('daily_pnl'))}"
        )
    if latest_launch:
        lines.append(
            f"Last launch: {latest_launch.get('launch_mode', '?')}"
            f" / {latest_launch.get('strategy_mode', '?')}"
            f" @ {str(latest_launch.get('timestamp', ''))[:16]}"
        )
    return "\n".join(lines)


def _load_latest_launch(project_root: Path) -> dict[str, object] | None:
    launches_root = project_root / "logs" / "launches"
    if not launches_root.exists():
        return None
    launch_files = sorted(
        launches_root.glob("launch_*.json"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not launch_files:
        return None
    try:
        return json.loads(launch_files[0].read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _parse_utc_timestamp(value: str) -> datetime | None:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _watchlist_row_text(row: dict[str, Any]) -> str:
    symbol = str(row.get("symbol") or "").ljust(6)
    _state_kind, state_label_raw = _trading_state_details(row)
    state_label = state_label_raw.ljust(7)
    price = _fmt_number(row.get("price"))
    sma = _fmt_number(row.get("sma"))
    pct = ""
    if row.get("price") not in (None, "") and row.get("sma") not in (None, ""):
        try:
            p, s = float(row["price"]), float(row["sma"])
            if s != 0:
                pct = f"  {((p - s) / s * 100):+.1f}%"
        except (TypeError, ValueError):
            pass
    ml_hint = ""
    if row.get("ml_probability_up") not in (None, ""):
        try:
            ml_hint = f"  ML {float(row['ml_probability_up']):.2f}"
        except (TypeError, ValueError):
            pass
    return f" {symbol}  {state_label}\n ${price}  SMA {sma}{pct}{ml_hint}"


def _position_summary_explained(row: dict[str, Any]) -> str:
    if not _is_truthy(row.get("holding")):
        return "FLAT"
    qty = _fmt_number(row.get("quantity"))
    value = _fmt_money_or_na(row.get("market_value"))
    mins = _fmt_number(row.get("holding_minutes"), digits=0)
    return f"{qty} sh | {value} | {mins}m"


def _signal_summary_explained(row: dict[str, Any]) -> str:
    if row.get("price") in (None, "") or row.get("sma") in (None, ""):
        return "MISSING"
    parts: list[str] = []
    try:
        price, sma = float(row["price"]), float(row["sma"])
        if sma != 0:
            parts.append(f"SMA {((price - sma) / sma) * 100:+.2f}%")
    except (TypeError, ValueError):
        return "MISSING"
    if row.get("ml_probability_up") not in (None, ""):
        try:
            parts.append(f"ML {float(row['ml_probability_up']):.2f}")
        except (TypeError, ValueError):
            pass
    return " | ".join(parts) if parts else "NO SIGNAL"


def _position_row_subtext(row: dict[str, Any]) -> str:
    qty = _fmt_number(row.get("quantity"))
    value = _fmt_money_or_na(row.get("market_value"))
    mins = _fmt_number(row.get("holding_minutes"), digits=0)
    action = str(row.get("action") or "").upper()
    next_step = "SELL queued" if action == "SELL" else "Holding"
    return f"{qty} shares | {value} | {mins} min | {next_step}"


def _dashboard_render_selected_symbol(self: NativeDashboardApp) -> None:
    if not self.selected_symbol:
        self.selected_symbol_label.configure(text="â€”")
        self.selected_price_var.set("â€”")
        self.selected_action_var.set("NO DATA")
        self.selected_position_var.set("FLAT")
        self.selected_signal_var.set("NO SIGNAL")
        for card in (self._selected_action_card, self._selected_position_card, self._selected_signal_card):
            if card is not None:
                card.configure(fg_color=_state_colors("no_data")[0])
        self._replace_text(self.detail_box, "No symbol selected.")
        return

    row = next((r for r in self.symbol_rows if str(r["symbol"]) == self.selected_symbol), None)
    if row is None:
        self.selected_action_var.set("NO DATA")
        self.selected_position_var.set("FLAT")
        self.selected_signal_var.set("MISSING")
        self._replace_text(self.detail_box, "Symbol no longer in latest snapshot.")
        return

    history = self.storage.get_symbol_history(self.selected_symbol, limit=12)
    action_kind, action_label = _trading_state_details(row)
    position_kind, _ = _position_state_details(row)
    signal_kind, _ = _signal_state_details(row)
    self.selected_symbol_label.configure(text=str(row["symbol"]))
    self.selected_price_var.set(_fmt_money_or_na(row.get("price")))
    self.selected_action_var.set(action_label)
    self.selected_position_var.set(_position_summary_explained(row))
    self.selected_signal_var.set(_signal_summary_explained(row))

    if self._selected_action_card is not None:
        self._selected_action_card.configure(fg_color=_state_colors(action_kind)[0])
        Tooltip(self._selected_action_card, _selected_action_tooltip_text(row))
    if self._selected_position_card is not None:
        self._selected_position_card.configure(fg_color=_state_colors(position_kind)[0])
        Tooltip(self._selected_position_card, _selected_position_tooltip_text(row))
    if self._selected_signal_card is not None:
        self._selected_signal_card.configure(fg_color=_state_colors(signal_kind)[0])
        Tooltip(self._selected_signal_card, _selected_signal_tooltip_text(row))
    if self._selected_price_label is not None:
        Tooltip(self._selected_price_label, _selected_price_tooltip_text(row))

    self._replace_text(self.detail_box, _format_symbol_detail_explained(row, history))


def _dashboard_rebuild_positions(self: NativeDashboardApp, rows: list[dict[str, Any]]) -> None:
    held = [r for r in rows if _is_truthy(r.get("holding"))]
    _clear_frame(self.positions_frame)
    if self._positions_count_label:
        self._positions_count_label.configure(text=f"{len(held)} open")

    if not held:
        _empty_label(self.positions_frame, 0, "FLAT")
        return

    for idx, row in enumerate(held):
        badge_kind, badge_text = _position_state_details(row)
        _stripe_row(
            self.positions_frame,
            idx,
            _state_colors(badge_kind)[2],
            str(row.get("symbol", "?")),
            subtext=_position_row_subtext(row),
            tooltip=_position_tooltip_text(row),
            badge_text=badge_text,
            badge_kind=badge_kind,
        )


def _dashboard_rebuild_orders(self: NativeDashboardApp, rows: list[dict[str, Any]]) -> None:
    _clear_frame(self.orders_frame)
    shown = min(len(rows), 10)
    if self._orders_count_label:
        self._orders_count_label.configure(text=f"{shown} shown" if shown else "")

    if not rows:
        _empty_label(self.orders_frame, 0, "NO DATA")
        return

    for idx, row in enumerate(rows[:10]):
        badge_kind, badge_text = _order_state_details(row)
        qty_val = row.get("filled_qty") if row.get("filled_qty") not in (None, "") else row.get("qty")
        _stripe_row(
            self.orders_frame,
            idx,
            _state_colors(badge_kind)[2],
            f"{str(row.get('symbol', '?'))} | {_friendly_order_status(row.get('status'))}",
            subtext=_order_row_subtext(row, qty_val),
            tooltip=_order_tooltip_text(row),
            badge_text=badge_text,
            badge_kind=badge_kind,
        )


NativeDashboardApp._render_selected_symbol = _dashboard_render_selected_symbol
NativeDashboardApp._rebuild_alerts = _dashboard_rebuild_alerts
NativeDashboardApp._rebuild_positions = _dashboard_rebuild_positions
NativeDashboardApp._rebuild_orders = _dashboard_rebuild_orders


_TIP_CHIP_ACTION = (
    "Primary state for the selected symbol.\n"
    "HELD = currently holding a position.\n"
    "BUY = latest snapshot favors entry.\n"
    "SELL = latest snapshot favors exit.\n"
    "ERROR = signal or data needs attention.\n"
    "NO SIGNAL = current snapshot is neutral.\n"
    "MISSING = key snapshot fields are unavailable.\n"
    "NO DATA = fallback when no better state can be inferred."
)
_TIP_CHIP_POSITION = (
    "Position snapshot for the selected symbol.\n"
    "HELD means a position is open.\n"
    "SELL means the latest snapshot is favoring exit.\n"
    "FLAT means no open position is shown.\n"
    "NO DATA is only used when position context cannot be inferred."
)
_TIP_CHIP_SIGNAL = (
    "Latest signal context behind the visible state.\n"
    "BUY = entry bias.\n"
    "SELL = exit bias.\n"
    "ERROR = signal evaluation failed or needs review.\n"
    "NO SIGNAL = neutral snapshot.\n"
    "MISSING = not enough current signal fields."
)
_TIP_SELECTED_PRICE = (
    "Latest saved snapshot price for the selected symbol.\n"
    "If nearby cards show MISSING, NO SIGNAL, or STALE, price alone may not be actionable."
)
_TIP_WATCHLIST = (
    "Each row shows one tracked symbol and one primary state.\n"
    "HELD = position open. BUY = entry bias. SELL = exit bias.\n"
    "NO SIGNAL = neutral. MISSING = incomplete snapshot. ERROR = needs attention."
)
_TIP_ALERTS_SECTION = (
    "Attention rail for current issues and notable state changes.\n"
    "ACTION = needs review now. WATCH = monitor. INFO = informational. OK = healthy. ISSUE = system or data problem."
)
_TIP_POSITIONS_SECTION = (
    "Open positions rail.\n"
    "HELD means a position is open. SELL means the latest snapshot is favoring exit."
)
_TIP_ORDERS_SECTION = (
    "Recent saved order events.\n"
    "BUY and SELL show order side. MISSING means the row has no usable side or current order context."
)


def _state_tooltip_blurb(kind: str) -> str:
    blurbs = {
        "held": "HELD = currently holding a position. Informational unless the row also shows an issue.",
        "buy": "BUY = latest snapshot is favoring entry or buy bias. Review only if that is unexpected.",
        "sell": "SELL = latest snapshot is favoring exit or sell bias. Review if a held position should be reduced.",
        "error": "ERROR = something needs attention or signal/data evaluation failed.",
        "flat": "FLAT = no open position is shown for this symbol.",
        "no_signal": "NO SIGNAL = current snapshot is neutral. There is no entry or exit bias right now.",
        "missing": "MISSING = key snapshot fields are unavailable or incomplete.",
        "stale": "STALE = the latest snapshot is old enough that state may no longer be current.",
        "market_closed": "MARKET CLOSED = no actionable market state is expected right now.",
        "no_data": "NO DATA = no current actionable state or not enough current data. This is not necessarily a crash.",
    }
    return blurbs.get(kind, blurbs["no_data"])


def _trading_state_details(row: dict[str, Any]) -> tuple[str, str]:
    kind = _primary_state_kind(row)
    return kind, _primary_state_label(row)


def _position_state_details(row: dict[str, Any]) -> tuple[str, str]:
    if _is_truthy(row.get("holding")):
        action = str(row.get("action") or "").upper()
        if action == "SELL":
            return "sell", "SELL"
        return "held", "HELD"
    return "flat", "FLAT"


def _signal_state_details(row: dict[str, Any]) -> tuple[str, str]:
    kind = _primary_state_kind(row)
    if kind in {"buy", "sell", "error", "missing", "no_signal"}:
        return kind, _primary_state_label(row)
    return "no_signal", "NO SIGNAL"


def _order_state_details(row: dict[str, Any]) -> tuple[str, str]:
    side = str(row.get("side") or "").upper()
    if side == "BUY":
        return "buy", "BUY"
    if side == "SELL":
        return "sell", "SELL"
    return "missing", "MISSING"


def _alert_badge_kind(severity: str, text: str) -> str:
    normalized = severity.strip().lower()
    if normalized == "error":
        if "kill switch" in text.lower():
            return "ACTION"
        return "ISSUE"
    if normalized == "warn":
        return "WATCH"
    if normalized == "info":
        return "INFO"
    if normalized == "ok":
        return "OK"
    return "WATCH"


def _alert_badge_color(kind: str) -> tuple[Any, Any, str]:
    palette = {
        "ACTION": (_C_RED_BG, _C_RED, _C_RED),
        "ISSUE": (_C_RED_BG, _C_RED, _C_RED),
        "WATCH": (_C_AMBER_BG, _C_AMBER, _C_AMBER),
        "INFO": (_C_BLUE_BG, _C_BLUE, _C_BLUE),
        "OK": (_C_GREEN_BG, _C_GREEN, _C_GREEN),
    }
    return palette.get(kind, (_C_AMBER_BG, _C_AMBER, _C_AMBER))


def _alert_tooltip_blurb(kind: str) -> str:
    blurbs = {
        "ACTION": "ACTION = review now. This alert may affect safety, execution, or readiness.",
        "WATCH": "WATCH = monitor closely. Not urgent yet, but worth checking.",
        "INFO": "INFO = informational context. Useful for awareness, not an instruction to trade.",
        "OK": "OK = healthy or expected status. No action usually required.",
        "ISSUE": "ISSUE = something needs attention or data/system evaluation did not complete cleanly.",
    }
    return blurbs.get(kind, blurbs["WATCH"])


def _watchlist_tooltip_text(row: dict[str, Any]) -> str:
    kind, _label = _trading_state_details(row)
    parts = [_state_tooltip_blurb(kind), "Watchlist row state from the latest snapshot."]
    if row.get("price") not in (None, ""):
        parts.append(f"Price: {_fmt_money_or_na(row.get('price'))}")
    if row.get("price") not in (None, "") and row.get("sma") not in (None, ""):
        try:
            p, s = float(row["price"]), float(row["sma"])
            if s != 0:
                parts.append(f"Signal context: price vs SMA {((p - s) / s) * 100:+.2f}%")
        except (TypeError, ValueError):
            pass
    if row.get("error"):
        parts.append(f"Attention: {row['error']}")
    return "\n".join(parts)


def _position_tooltip_text(row: dict[str, Any]) -> str:
    action = str(row.get("action") or "").upper()
    if action == "SELL":
        return (
            f"{_state_tooltip_blurb('sell')}\n"
            "Position state: currently holding a position.\n"
            "Action: review exit bias on the latest snapshot."
        )
    return (
        f"{_state_tooltip_blurb('held')}\n"
        "Position state: currently holding a position.\n"
        "Action: no immediate action unless strategy behavior looks wrong."
    )


def _order_tooltip_text(row: dict[str, Any]) -> str:
    kind, _label = _order_state_details(row)
    status = _friendly_order_status(row.get("status"))
    return (
        f"{_state_tooltip_blurb(kind)}\n"
        f"Order state: {status}.\n"
        "This row reflects order side and fill status, not current position size."
    )


def _selected_action_tooltip_text(row: dict[str, Any]) -> str:
    kind, _label = _trading_state_details(row)
    return (
        f"{_state_tooltip_blurb(kind)}\n"
        "This card reflects the selected symbol's latest primary state."
    )


def _selected_position_tooltip_text(row: dict[str, Any]) -> str:
    if _is_truthy(row.get("holding")):
        return (
            f"{_state_tooltip_blurb('held')}\n"
            "This card reflects position state.\n"
            f"Size: {_fmt_number(row.get('quantity'))} shares | Value: {_fmt_money_or_na(row.get('market_value'))}"
        )
    return (
        f"{_state_tooltip_blurb('flat')}\n"
        "This card reflects position state.\n"
        "No current open position is shown in the latest snapshot."
    )


def _selected_signal_tooltip_text(row: dict[str, Any]) -> str:
    kind, _label = _signal_state_details(row)
    parts = [_state_tooltip_blurb(kind), "This card reflects signal state from the latest snapshot."]
    if row.get("price") not in (None, "") and row.get("sma") not in (None, ""):
        try:
            price = float(row["price"])
            sma = float(row["sma"])
            if sma != 0:
                parts.append(f"SMA context: {((price - sma) / sma) * 100:+.2f}%")
        except (TypeError, ValueError):
            pass
    if row.get("ml_probability_up") not in (None, ""):
        try:
            parts.append(f"ML up: {float(row['ml_probability_up']):.2f}")
        except (TypeError, ValueError):
            pass
    return "\n".join(parts)


def _selected_price_tooltip_text(row: dict[str, Any]) -> str:
    return (
        f"Latest saved snapshot price: {_fmt_money_or_na(row.get('price'))}\n"
        "Data availability: this is a saved snapshot, not a live tick.\n"
        "If other cards show NO SIGNAL, MISSING, or STALE, price alone may not be enough for an actionable state."
    )


def _format_symbol_detail_explained(row: dict[str, Any], history: list[dict[str, Any]]) -> str:
    lines = [
        f"price       {_fmt_money_or_na(row.get('price'))}",
        f"sma         {_fmt_number(row.get('sma'))}",
        f"ml up       {_fmt_number(row.get('ml_probability_up'))}",
        f"confidence  {_fmt_number(row.get('ml_confidence'))}",
    ]
    if row.get("error"):
        lines.append(f"error       {row.get('error')}")
    lines.extend(["", "recent"])
    if not history:
        lines.append("no saved history")
    else:
        for item in history[-6:]:
            ts = str(item.get("timestamp_utc", ""))[:16]
            price = _fmt_number(item.get("price"))
            action = str(item.get("action") or "-").rjust(5)
            lines.append(f"{ts}  {price:>8}  {action}")
    return "\n".join(lines)


def _alert_tooltip_text(severity: str, text: str) -> str:
    badge = _alert_badge_kind(severity, text)
    return f"{_alert_tooltip_blurb(badge)}\n{text}"


def _dashboard_rebuild_alerts(
    self: NativeDashboardApp,
    latest_run: dict[str, Any] | None,
    rows: list[dict[str, Any]],
    latest_launch: dict[str, Any] | None,
) -> None:
    alerts = _collect_alerts(latest_run, rows, latest_launch)
    _clear_frame(self.alerts_frame)
    if self._alerts_count_label:
        self._alerts_count_label.configure(text=f"{len(alerts)} item{'s' if len(alerts) != 1 else ''}")

    if not alerts:
        _empty_label(self.alerts_frame, 0, "All clear")
        return

    for idx, (severity, text) in enumerate(alerts):
        badge = _alert_badge_kind(severity, text)
        fg_color, _text_color, stripe = _alert_badge_color(badge)
        _stripe_row(
            self.alerts_frame,
            idx,
            stripe,
            text,
            subtext=_alert_subtext(severity, text),
            wrap=240,
            tooltip=_alert_tooltip_text(severity, text),
            badge_text=badge,
            badge_kind="error" if badge in {"ACTION", "ISSUE"} else "buy" if badge == "INFO" else "held" if badge == "OK" else "sell",
        )


def main() -> None:
    app = NativeDashboardApp()
    app.mainloop()


if __name__ == "__main__":
    main()

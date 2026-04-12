from __future__ import annotations

import os
import queue
from dataclasses import dataclass
from pathlib import Path

import customtkinter as ctk

from desktop_app.log_formatter import format_log_line
from desktop_app.paths import open_in_windows, resolve_latest_logs, resolve_results, resolve_today_log
from desktop_app.repo_status import RuntimeStatus, load_runtime_status
from desktop_app.runner import ProcessRunner, ProcessUpdate


@dataclass(frozen=True)
class CommandSpec:
    label: str
    command: list[str]
    description: str
    safety_level: str
    action: str = "run"


_PROJECT_ROOT_OVERRIDE = os.getenv("ALPACA_BOT_PROJECT_ROOT")
PROJECT_ROOT = (
    Path(_PROJECT_ROOT_OVERRIDE).resolve()
    if _PROJECT_ROOT_OVERRIDE
    else Path(__file__).resolve().parent.parent
)
POWERSHELL_EXE = "powershell"
BOT_SCRIPT = PROJECT_ROOT / "bot.ps1"

COMMAND_SPECS: list[CommandSpec] = [
    CommandSpec(
        label="Setup",
        command=[POWERSHELL_EXE, "-ExecutionPolicy", "Bypass", "-File", "bot.ps1", "setup"],
        description="Verify local runtime dependencies and required files.",
        safety_level="Read-only",
    ),
    CommandSpec(
        label="Show Config",
        command=[POWERSHELL_EXE, "-ExecutionPolicy", "Bypass", "-File", "bot.ps1", "show-config"],
        description="Display the promoted live runtime config.",
        safety_level="Read-only",
    ),
    CommandSpec(
        label="Preflight",
        command=[POWERSHELL_EXE, "-ExecutionPolicy", "Bypass", "-File", "bot.ps1", "preflight"],
        description="Run readiness checks before launch.",
        safety_level="Read-only",
    ),
    CommandSpec(
        label="Paper Run",
        command=[POWERSHELL_EXE, "-ExecutionPolicy", "Bypass", "-File", "bot.ps1", "paper-run"],
        description="Run the bot with paper-safe execution disabled for orders.",
        safety_level="Paper",
    ),
    CommandSpec(
        label="Live Run",
        command=[POWERSHELL_EXE, "-ExecutionPolicy", "Bypass", "-File", "bot.ps1", "live-run", "--confirm-live"],
        description="Run the live-order launch path after explicit confirmation.",
        safety_level="Live-sensitive",
    ),
    CommandSpec(
        label="Open Browser Dashboard",
        command=[POWERSHELL_EXE, "-ExecutionPolicy", "Bypass", "-File", "bot.ps1", "dashboard"],
        description="Launch the browser-based Streamlit dashboard.",
        safety_level="Read-only",
    ),
    CommandSpec(
        label="Open Latest Logs Folder",
        command=[],
        description="Open the newest dated logs folder under logs/ for review.",
        safety_level="Read-only",
        action="open_latest_logs",
    ),
    CommandSpec(
        label="Open Results Folder",
        command=[],
        description="Open the results/ folder that holds research and backtest outputs.",
        safety_level="Read-only",
        action="open_results",
    ),
    CommandSpec(
        label="Run Daily Report",
        command=[],
        description="Run daily_report.py against the newest dated logs folder and stream the report here.",
        safety_level="Read-only",
        action="daily_report",
    ),
    CommandSpec(
        label="Watch Signals",
        command=[],
        description="Tail today's signals.jsonl in the console. Streams new entries as the bot writes them.",
        safety_level="Read-only",
        action="watch_log:signals.jsonl",
    ),
    CommandSpec(
        label="Watch Bars",
        command=[],
        description="Tail today's bars.jsonl in the console. Shows each completed bar the bot receives.",
        safety_level="Read-only",
        action="watch_log:bars.jsonl",
    ),
]

PRIMARY_COMMAND_LABELS = {
    "Setup",
    "Show Config",
    "Preflight",
    "Paper Run",
    "Live Run",
    "Open Browser Dashboard",
}

LOG_REPORT_COMMAND_LABELS = {
    "Open Latest Logs Folder",
    "Open Results Folder",
    "Run Daily Report",
}

LIVE_LOG_COMMAND_LABELS = {
    "Watch Signals",
    "Watch Bars",
}


class ControlPanelApp(ctk.CTk):
    def __init__(self) -> None:
        super().__init__()
        self.title("AlpacaTradingBot Control Panel")
        self.geometry("1100x760")
        self.minsize(960, 680)

        ctk.set_appearance_mode("System")
        ctk.set_default_color_theme("blue")

        self.runner = ProcessRunner()
        self._events: queue.Queue[tuple[str, object]] = queue.Queue()
        self._command_buttons: list[ctk.CTkButton] = []
        self._selected_command = COMMAND_SPECS[0]
        self._runtime_status = load_runtime_status(PROJECT_ROOT)
        self._preflight_ok_in_session = False
        self._log_format_active = False  # True while a watch_log command is streaming

        self.status_var = ctk.StringVar(value="Idle")
        self.preflight_gate_var = ctk.StringVar(value="Required before Live Run")
        self.command_name_var = ctk.StringVar(value=self._selected_command.label)
        self.command_desc_var = ctk.StringVar(value=self._selected_command.description)
        self.command_text_var = ctk.StringVar(value=" ".join(self._selected_command.command))
        self.command_safety_var = ctk.StringVar(value=self._selected_command.safety_level)
        self.strategy_var = ctk.StringVar(value=self._runtime_status.strategy_mode)
        self.symbols_var = ctk.StringVar(value=self._runtime_status.symbols_text)
        self.timeframe_var = ctk.StringVar(value=self._runtime_status.timeframe_text)
        self.config_source_var = ctk.StringVar(value=self._runtime_status.runtime_source)
        self.promoted_source_var = ctk.StringVar(value=self._runtime_status.promoted_config_source)
        self.repo_root_var = ctk.StringVar(value=str(PROJECT_ROOT))
        self.activity_var = ctk.StringVar(value="No command run yet")
        self._format_var = ctk.BooleanVar(value=True)  # Formatted log output toggle

        self._build_layout()
        self._apply_status_badge("idle")
        self.after(100, self._drain_events)

    def _build_layout(self) -> None:
        self.grid_columnconfigure(0, weight=0)
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        sidebar = ctk.CTkFrame(self, corner_radius=0)
        sidebar.grid(row=0, column=0, sticky="nsew")
        sidebar.grid_columnconfigure(0, weight=1)

        header = ctk.CTkLabel(
            sidebar,
            text="Control Panel",
            font=ctk.CTkFont(size=22, weight="bold"),
        )
        header.grid(row=0, column=0, padx=18, pady=(18, 6), sticky="w")

        subheader = ctk.CTkLabel(
            sidebar,
            text="Thin wrapper over existing repo commands",
            text_color=("gray40", "gray70"),
        )
        subheader.grid(row=1, column=0, padx=18, pady=(0, 18), sticky="w")

        row = 2
        row = self._add_sidebar_section(
            sidebar=sidebar,
            row=row,
            title="Command Actions",
            specs=[spec for spec in COMMAND_SPECS if spec.label in PRIMARY_COMMAND_LABELS],
        )
        row = self._add_sidebar_section(
            sidebar=sidebar,
            row=row,
            title="Logs And Reports",
            specs=[spec for spec in COMMAND_SPECS if spec.label in LOG_REPORT_COMMAND_LABELS],
        )
        row = self._add_sidebar_section(
            sidebar=sidebar,
            row=row,
            title="Live Logs",
            specs=[spec for spec in COMMAND_SPECS if spec.label in LIVE_LOG_COMMAND_LABELS],
        )

        ctk.CTkLabel(
            sidebar,
            text="Process Control",
            font=ctk.CTkFont(size=12, weight="bold"),
            text_color=("gray35", "gray70"),
        ).grid(row=row, column=0, padx=18, pady=(2, 8), sticky="w")
        row += 1

        ctk.CTkFrame(sidebar, height=1, fg_color=("gray78", "gray28")).grid(
            row=row, column=0, padx=18, pady=(0, 10), sticky="ew"
        )
        row += 1

        stop_button = ctk.CTkButton(
            sidebar,
            text="Stop Running Command",
            command=self._stop_process,
            fg_color="#b45309",
            hover_color="#92400e",
            height=38,
        )
        stop_button.grid(row=row, column=0, padx=18, pady=(0, 12), sticky="ew")
        self.stop_button = stop_button
        self.stop_button.configure(state="disabled")
        row += 1

        ctk.CTkLabel(
            sidebar,
            text="About This Launcher",
            font=ctk.CTkFont(size=12, weight="bold"),
            text_color=("gray35", "gray70"),
        ).grid(row=row, column=0, padx=18, pady=(0, 8), sticky="w")
        row += 1

        info = ctk.CTkTextbox(sidebar, width=280, height=180)
        info.grid(row=row, column=0, padx=18, pady=(0, 18), sticky="nsew")
        info.insert(
            "1.0",
            "Wrapped commands:\n"
            "- bot.ps1 setup\n"
            "- bot.ps1 show-config\n"
            "- bot.ps1 preflight\n"
            "- bot.ps1 paper-run\n"
            "- bot.ps1 live-run --confirm-live\n"
            "- bot.ps1 dashboard (browser)\n\n"
            "This launcher is a thin wrapper over the repo's existing workflows.\n"
            "It streams live process output below and does not change bot logic.",
        )
        info.configure(state="disabled")

        main = ctk.CTkFrame(self)
        main.grid(row=0, column=1, sticky="nsew", padx=16, pady=16)
        main.grid_columnconfigure(0, weight=1)
        main.grid_rowconfigure(2, weight=1)

        header_frame = ctk.CTkFrame(main)
        header_frame.grid(row=0, column=0, padx=16, pady=(16, 10), sticky="ew")
        header_frame.grid_columnconfigure(0, weight=1)
        header_frame.grid_columnconfigure(1, weight=1)

        ctk.CTkLabel(
            header_frame,
            text="Configured Runtime",
            font=ctk.CTkFont(size=20, weight="bold"),
        ).grid(row=0, column=0, padx=14, pady=(14, 8), sticky="w")

        self.process_badge = ctk.CTkLabel(
            header_frame,
            text="IDLE",
            corner_radius=999,
            padx=10,
            pady=4,
        )
        self.process_badge.grid(row=0, column=1, padx=14, pady=(14, 8), sticky="e")

        self.account_badge = ctk.CTkLabel(
            header_frame,
            text=self._runtime_status.account_mode.upper(),
            corner_radius=999,
            padx=10,
            pady=4,
        )
        self.account_badge.grid(row=1, column=0, padx=(14, 8), pady=(0, 12), sticky="w")
        self.execution_badge = ctk.CTkLabel(
            header_frame,
            text=self._runtime_status.execution_mode.upper(),
            corner_radius=999,
            padx=10,
            pady=4,
        )
        self.execution_badge.grid(row=1, column=1, padx=(8, 14), pady=(0, 12), sticky="e")
        self._apply_mode_badge(self.account_badge, self._runtime_status.account_mode)
        self._apply_mode_badge(self.execution_badge, self._runtime_status.execution_mode)

        details_left = ctk.CTkFrame(header_frame, fg_color="transparent")
        details_left.grid(row=2, column=0, padx=14, pady=(0, 14), sticky="nsew")
        details_right = ctk.CTkFrame(header_frame, fg_color="transparent")
        details_right.grid(row=2, column=1, padx=14, pady=(0, 14), sticky="nsew")
        details_left.grid_columnconfigure(1, weight=1)
        details_right.grid_columnconfigure(1, weight=1)

        self._add_value_row(details_left, 0, "Strategy", self.strategy_var)
        self._add_value_row(details_left, 1, "Symbols", self.symbols_var)
        self._add_value_row(details_left, 2, "Timeframe", self.timeframe_var)
        self._add_value_row(details_left, 3, "Live Gate", self.preflight_gate_var)
        self._add_value_row(details_right, 0, "Config Source", self.config_source_var)
        self._add_value_row(details_right, 1, "Promoted Source", self.promoted_source_var)

        middle_frame = ctk.CTkFrame(main, fg_color="transparent")
        middle_frame.grid(row=1, column=0, padx=16, pady=(0, 10), sticky="ew")
        middle_frame.grid_columnconfigure(0, weight=3)
        middle_frame.grid_columnconfigure(1, weight=2)

        command_frame = ctk.CTkFrame(middle_frame)
        command_frame.grid(row=0, column=0, padx=(0, 8), sticky="nsew")
        command_frame.grid_columnconfigure(1, weight=1)

        ctk.CTkLabel(
            command_frame,
            text="Command Details",
            font=ctk.CTkFont(size=18, weight="bold"),
        ).grid(row=0, column=0, columnspan=2, padx=14, pady=(14, 8), sticky="w")

        self._add_value_row(command_frame, 1, "Selected", self.command_name_var)
        self._add_value_row(command_frame, 2, "What It Does", self.command_desc_var)
        self._add_value_row(command_frame, 3, "Wrapped Command", self.command_text_var)

        ctk.CTkLabel(
            command_frame,
            text="Safety",
            text_color=("gray35", "gray70"),
        ).grid(row=4, column=0, padx=(14, 10), pady=(0, 14), sticky="nw")
        self.command_safety_badge = ctk.CTkLabel(command_frame, text="", corner_radius=999, padx=10, pady=4)
        self.command_safety_badge.grid(row=4, column=1, padx=(0, 14), pady=(0, 14), sticky="w")
        self._apply_command_details(self._selected_command)

        summary_frame = ctk.CTkFrame(middle_frame)
        summary_frame.grid(row=0, column=1, padx=(8, 0), sticky="nsew")
        summary_frame.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(
            summary_frame,
            text="Launcher Status",
            font=ctk.CTkFont(size=18, weight="bold"),
        ).grid(row=0, column=0, padx=14, pady=(14, 8), sticky="w")

        self.status_label = ctk.CTkLabel(
            summary_frame,
            textvariable=self.status_var,
            justify="left",
            wraplength=280,
            text_color=("gray35", "gray70"),
        )
        self.status_label.grid(row=1, column=0, padx=14, pady=(0, 8), sticky="w")

        ctk.CTkLabel(
            summary_frame,
            text="Last Or Active Command",
            text_color=("gray35", "gray70"),
        ).grid(row=2, column=0, padx=14, pady=(4, 4), sticky="w")
        ctk.CTkLabel(
            summary_frame,
            textvariable=self.activity_var,
            justify="left",
            wraplength=280,
        ).grid(row=3, column=0, padx=14, pady=(0, 14), sticky="w")

        console_frame = ctk.CTkFrame(main)
        console_frame.grid(row=2, column=0, padx=16, pady=(0, 10), sticky="nsew")
        console_frame.grid_columnconfigure(0, weight=1)
        console_frame.grid_columnconfigure(1, weight=0)
        console_frame.grid_rowconfigure(1, weight=1)

        ctk.CTkLabel(
            console_frame,
            text="Output Console",
            font=ctk.CTkFont(size=20, weight="bold"),
        ).grid(row=0, column=0, padx=16, pady=(14, 8), sticky="w")

        ctk.CTkCheckBox(
            console_frame,
            text="Formatted logs",
            variable=self._format_var,
            width=140,
        ).grid(row=0, column=1, padx=(0, 16), pady=(14, 8), sticky="e")

        self.console = ctk.CTkTextbox(console_frame, wrap="none")
        self.console.grid(row=1, column=0, columnspan=2, padx=16, pady=(0, 16), sticky="nsew")
        self.console.insert("1.0", "Ready.\n")
        self.console.configure(state="disabled")

        footer_frame = ctk.CTkFrame(main)
        footer_frame.grid(row=3, column=0, padx=16, pady=(0, 16), sticky="ew")
        footer_frame.grid_columnconfigure(1, weight=1)

        ctk.CTkLabel(
            footer_frame,
            text="Repo Root",
            text_color=("gray35", "gray70"),
        ).grid(row=0, column=0, padx=(14, 10), pady=10, sticky="w")
        ctk.CTkLabel(
            footer_frame,
            textvariable=self.repo_root_var,
            justify="left",
            wraplength=760,
        ).grid(row=0, column=1, padx=(0, 14), pady=10, sticky="w")

    def _launch_command(self, spec: CommandSpec) -> None:
        self._apply_command_details(spec)
        if self.runner.is_running():
            self._append_line("A process is already running. Stop it before launching another command.")
            return

        if not BOT_SCRIPT.exists():
            self._append_line(f"Missing required script: {BOT_SCRIPT}")
            return

        if spec.action == "open_latest_logs":
            self._open_latest_logs()
            return
        if spec.action == "open_results":
            self._open_results_folder()
            return
        if spec.action == "daily_report":
            spec = self._build_daily_report_command(spec)
            self._apply_command_details(spec)
            latest_date = spec.command[3] if len(spec.command) >= 4 else "unknown"
            self._append_line(f"Using inferred daily report date from latest logs folder: {latest_date}")
        if spec.action.startswith("watch_log:"):
            filename = spec.action.split(":", 1)[1]
            spec = self._build_watch_log_command(spec, filename)
            if spec is None:
                return
            self._apply_command_details(spec)
            self._log_format_active = True

        if spec.safety_level == "Live-sensitive":
            if not self._preflight_ok_in_session:
                self._append_line("Live Run blocked: Preflight must succeed in this app session first.")
                return
            if not self._confirm_live_run(spec):
                self._append_line("Live Run cancelled before launch.")
                return

        pretty = " ".join(spec.command)
        self.activity_var.set(pretty)
        self._append_line("")
        self._append_line(f"$ {pretty}")

        try:
            self.runner.start(
                spec.command,
                cwd=PROJECT_ROOT,
                on_output=lambda stream, line: self._events.put(("output", (stream, line))),
                on_update=lambda update: self._events.put(("update", update)),
            )
        except Exception as exc:
            self._apply_runner_update(
                ProcessUpdate(status="idle", command=[], detail="Ready")
            )
            self._append_line(f"Failed to start command: {exc}")

    def _stop_process(self) -> None:
        update = self.runner.stop()
        if update is None:
            self._append_line("No running process to stop.")
            return

        self._apply_runner_update(update)
        if update.detail:
            self._append_line(update.detail)

    def _append_line(self, text: str) -> None:
        self.console.configure(state="normal")
        self.console.insert("end", text + "\n")
        self.console.see("end")
        self.console.configure(state="disabled")

    def _apply_runner_update(self, update: ProcessUpdate) -> None:
        command_text = " ".join(update.command) if update.command else "none"
        if update.command:
            self.activity_var.set(command_text)

        if update.status == "running":
            self.status_var.set(f"Running: {command_text}")
            self._set_command_buttons_enabled(False)
            self.stop_button.configure(state="normal")
            self._apply_status_badge("running")
            return

        if update.status == "stopping":
            self.status_var.set(f"Stopping: {command_text}")
            self._set_command_buttons_enabled(False)
            self.stop_button.configure(state="disabled")
            self._apply_status_badge("stopping")
            return

        self._set_command_buttons_enabled(True)
        self.stop_button.configure(state="disabled")
        self._log_format_active = False
        self._refresh_runtime_status()
        self._update_live_gate(update)

        if update.status == "completed":
            self.status_var.set(f"Idle (completed, exit {update.exit_code})")
            self._apply_status_badge("success")
        elif update.status == "failed":
            self.status_var.set(f"Idle (failed, exit {update.exit_code})")
            self._apply_status_badge("failed")
        elif update.status == "stopped":
            self.status_var.set(f"Idle (stopped, exit {update.exit_code})")
            self._apply_status_badge("idle")
        else:
            self.status_var.set("Idle")
            self._apply_status_badge("idle")

    def _set_command_buttons_enabled(self, enabled: bool) -> None:
        state = "normal" if enabled else "disabled"
        for button in self._command_buttons:
            button.configure(state=state)

    def _apply_command_details(self, spec: CommandSpec) -> None:
        self._selected_command = spec
        self.command_name_var.set(spec.label)
        self.command_desc_var.set(spec.description)
        self.command_text_var.set(" ".join(spec.command) if spec.command else self._describe_nonprocess_action(spec))
        self.command_safety_var.set(spec.safety_level)
        self._apply_mode_badge(self.command_safety_badge, spec.safety_level)
        self.command_safety_badge.configure(text=spec.safety_level.upper())

    def _add_value_row(self, parent: ctk.CTkFrame, row: int, label: str, variable: ctk.StringVar) -> None:
        ctk.CTkLabel(
            parent,
            text=label,
            text_color=("gray35", "gray70"),
        ).grid(row=row, column=0, padx=(0, 10), pady=(0, 8), sticky="nw")
        ctk.CTkLabel(
            parent,
            textvariable=variable,
            justify="left",
            wraplength=400,
        ).grid(row=row, column=1, pady=(0, 8), sticky="w")

    def _add_sidebar_section(
        self,
        sidebar: ctk.CTkFrame,
        row: int,
        title: str,
        specs: list[CommandSpec],
    ) -> int:
        ctk.CTkLabel(
            sidebar,
            text=title,
            font=ctk.CTkFont(size=12, weight="bold"),
            text_color=("gray35", "gray70"),
        ).grid(row=row, column=0, padx=18, pady=(0, 8), sticky="w")
        row += 1

        ctk.CTkFrame(sidebar, height=1, fg_color=("gray78", "gray28")).grid(
            row=row, column=0, padx=18, pady=(0, 10), sticky="ew"
        )
        row += 1

        for spec in specs:
            button = ctk.CTkButton(
                sidebar,
                text=spec.label,
                command=lambda current=spec: self._launch_command(current),
                height=38,
                fg_color=self._button_fg(spec.safety_level),
                hover_color=self._button_hover(spec.safety_level),
            )
            button.grid(row=row, column=0, padx=18, pady=(0, 6), sticky="ew")
            self._command_buttons.append(button)
            row += 1

            ctk.CTkLabel(
                sidebar,
                text=spec.description,
                justify="left",
                wraplength=250,
                text_color=("gray35", "gray70"),
            ).grid(row=row, column=0, padx=18, pady=(0, 10), sticky="w")
            row += 1

        return row

    def _apply_status_badge(self, status: str) -> None:
        palette = {
            "idle": ("#475569", "#e2e8f0"),
            "running": ("#1d4ed8", "#dbeafe"),
            "stopping": ("#b45309", "#fef3c7"),
            "success": ("#15803d", "#dcfce7"),
            "failed": ("#b91c1c", "#fee2e2"),
        }
        fg_color, text_color = palette.get(status, ("#475569", "#e2e8f0"))
        self.process_badge.configure(text=status.upper(), fg_color=fg_color, text_color=text_color)

    def _apply_mode_badge(self, label: ctk.CTkLabel, mode: str) -> None:
        normalized = mode.strip().lower()
        palette = {
            "paper": ("#0369a1", "#e0f2fe"),
            "live": ("#b91c1c", "#fee2e2"),
            "dry run": ("#166534", "#dcfce7"),
            "live orders": ("#b91c1c", "#fee2e2"),
            "read-only": ("#475569", "#e2e8f0"),
            "live-sensitive": ("#b91c1c", "#fee2e2"),
        }
        fg_color, text_color = palette.get(normalized, ("#475569", "#e2e8f0"))
        label.configure(fg_color=fg_color, text_color=text_color)

    def _button_fg(self, safety_level: str) -> str:
        palette = {
            "Read-only": "#475569",
            "Paper": "#0369a1",
            "Live-sensitive": "#b91c1c",
        }
        return palette.get(safety_level, "#475569")

    def _button_hover(self, safety_level: str) -> str:
        palette = {
            "Read-only": "#334155",
            "Paper": "#075985",
            "Live-sensitive": "#991b1b",
        }
        return palette.get(safety_level, "#334155")

    def _refresh_runtime_status(self) -> None:
        self._runtime_status = load_runtime_status(PROJECT_ROOT)
        self.strategy_var.set(self._runtime_status.strategy_mode)
        self.symbols_var.set(self._runtime_status.symbols_text)
        self.timeframe_var.set(self._runtime_status.timeframe_text)
        self.config_source_var.set(self._runtime_status.runtime_source)
        self.promoted_source_var.set(self._runtime_status.promoted_config_source)
        self.account_badge.configure(text=self._runtime_status.account_mode.upper())
        self.execution_badge.configure(text=self._runtime_status.execution_mode.upper())
        self._apply_mode_badge(self.account_badge, self._runtime_status.account_mode)
        self._apply_mode_badge(self.execution_badge, self._runtime_status.execution_mode)

    def _open_latest_logs(self) -> None:
        try:
            resolution = resolve_latest_logs(PROJECT_ROOT)
            open_in_windows(resolution.path)
            self.activity_var.set(f"Open folder: {resolution.path}")
            self._append_line(f"Opened latest logs folder: {resolution.path}")
        except Exception as exc:
            self._append_line(f"Unable to open latest logs folder: {exc}")

    def _open_results_folder(self) -> None:
        try:
            resolution = resolve_results(PROJECT_ROOT)
            open_in_windows(resolution.path)
            self.activity_var.set(f"Open folder: {resolution.path}")
            latest_text = f" Latest child: {resolution.latest_child}" if resolution.latest_child is not None else ""
            self._append_line(f"Opened results folder: {resolution.path}.{latest_text}")
        except Exception as exc:
            self._append_line(f"Unable to open results folder: {exc}")

    def _update_live_gate(self, update: ProcessUpdate) -> None:
        preflight_command = next((spec.command for spec in COMMAND_SPECS if spec.label == "Preflight"), None)
        if preflight_command is not None and update.command == preflight_command:
            self._preflight_ok_in_session = update.status == "completed"
        self.preflight_gate_var.set(
            "Passed in this session" if self._preflight_ok_in_session else "Required before Live Run"
        )

    def _confirm_live_run(self, spec: CommandSpec) -> bool:
        confirmed = {"value": False}

        dialog = ctk.CTkToplevel(self)
        dialog.title("Confirm Live Run")
        dialog.geometry("660x380")
        dialog.resizable(False, False)
        dialog.transient(self)
        dialog.grab_set()
        dialog.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(
            dialog,
            text="LIVE ORDER WARNING",
            font=ctk.CTkFont(size=22, weight="bold"),
            text_color="#b91c1c",
        ).grid(row=0, column=0, padx=18, pady=(18, 10), sticky="w")

        ctk.CTkLabel(
            dialog,
            text=(
                "This action launches the live-order workflow through bot.ps1.\n"
                "The backend still enforces its own safety checks, but this path is intended for real-order execution."
            ),
            justify="left",
            wraplength=610,
        ).grid(row=1, column=0, padx=18, pady=(0, 12), sticky="w")

        ctk.CTkLabel(
            dialog,
            text="Exact command to be launched",
            text_color=("gray35", "gray70"),
        ).grid(row=2, column=0, padx=18, pady=(0, 6), sticky="w")

        command_box = ctk.CTkTextbox(dialog, height=72)
        command_box.grid(row=3, column=0, padx=18, pady=(0, 12), sticky="ew")
        command_box.insert("1.0", " ".join(spec.command))
        command_box.configure(state="disabled")

        ctk.CTkLabel(
            dialog,
            text='Type LIVE to confirm',
            text_color=("gray35", "gray70"),
        ).grid(row=4, column=0, padx=18, pady=(0, 6), sticky="w")

        confirmation_entry = ctk.CTkEntry(dialog, placeholder_text="LIVE")
        confirmation_entry.grid(row=5, column=0, padx=18, pady=(0, 12), sticky="ew")

        acknowledge_var = ctk.BooleanVar(value=False)
        ctk.CTkCheckBox(
            dialog,
            text="I understand this uses the live-order launch path.",
            variable=acknowledge_var,
        ).grid(row=6, column=0, padx=18, pady=(0, 12), sticky="w")

        feedback_var = ctk.StringVar(value="")
        ctk.CTkLabel(
            dialog,
            textvariable=feedback_var,
            text_color="#b91c1c",
        ).grid(row=7, column=0, padx=18, pady=(0, 12), sticky="w")

        button_row = ctk.CTkFrame(dialog, fg_color="transparent")
        button_row.grid(row=8, column=0, padx=18, pady=(0, 18), sticky="e")

        def cancel() -> None:
            dialog.destroy()

        def confirm() -> None:
            if confirmation_entry.get().strip().upper() != "LIVE":
                feedback_var.set("Typed confirmation must be exactly LIVE.")
                return
            if not acknowledge_var.get():
                feedback_var.set("You must acknowledge the live-order warning.")
                return
            confirmed["value"] = True
            dialog.destroy()

        ctk.CTkButton(
            button_row,
            text="Cancel",
            command=cancel,
            fg_color="#475569",
            hover_color="#334155",
        ).pack(side="left", padx=(0, 10))
        ctk.CTkButton(
            button_row,
            text="Confirm Live Run",
            command=confirm,
            fg_color="#b91c1c",
            hover_color="#991b1b",
        ).pack(side="left")

        confirmation_entry.focus()
        self.wait_window(dialog)
        return confirmed["value"]

    def _build_daily_report_command(self, spec: CommandSpec) -> CommandSpec:
        resolution = resolve_latest_logs(PROJECT_ROOT)
        latest_date = resolution.path.name
        command = [
            "python",
            "daily_report.py",
            "--date",
            latest_date,
            "--log-root",
            "logs",
        ]
        description = f"{spec.description} Latest log date: {latest_date}."
        return CommandSpec(
            label=spec.label,
            command=command,
            description=description,
            safety_level=spec.safety_level,
            action="run",
        )

    def _build_watch_log_command(self, spec: CommandSpec, filename: str) -> CommandSpec | None:
        try:
            resolution = resolve_today_log(PROJECT_ROOT, filename)
        except FileNotFoundError as exc:
            self._append_line(f"Cannot watch log: {exc}")
            return None
        log_path = str(resolution.path)
        command = [
            POWERSHELL_EXE,
            "-NoProfile",
            "-Command",
            f"Get-Content -Wait -Path '{log_path}'",
        ]
        return CommandSpec(
            label=spec.label,
            command=command,
            description=f"{spec.description} File: {log_path}",
            safety_level=spec.safety_level,
            action="run",
        )

    def _describe_nonprocess_action(self, spec: CommandSpec) -> str:
        if spec.action == "open_latest_logs":
            try:
                return f"Open folder: {resolve_latest_logs(PROJECT_ROOT).path}"
            except Exception:
                return "Open latest dated logs folder"
        if spec.action == "open_results":
            try:
                resolution = resolve_results(PROJECT_ROOT)
                latest_text = f" (latest child: {resolution.latest_child.name})" if resolution.latest_child is not None else ""
                return f"Open folder: {resolution.path}{latest_text}"
            except Exception:
                return "Open results folder"
        if spec.action == "daily_report":
            try:
                latest_logs = resolve_latest_logs(PROJECT_ROOT).path.name
                return f"python daily_report.py --date {latest_logs} --log-root logs"
            except Exception:
                return "python daily_report.py --date <latest-log-date> --log-root logs"
        if spec.action.startswith("watch_log:"):
            filename = spec.action.split(":", 1)[1]
            try:
                log_path = resolve_today_log(PROJECT_ROOT, filename).path
                return f"Get-Content -Wait '{log_path}'"
            except Exception:
                return f"Get-Content -Wait logs/<today>/{filename}"
        return "Manual action"

    def _drain_events(self) -> None:
        while not self._events.empty():
            event_type, payload = self._events.get()
            if event_type == "output":
                stream, line = payload
                if self._log_format_active and self._format_var.get() and stream == "stdout":
                    formatted = format_log_line(line)
                    if formatted:  # empty string means suppressed event
                        self._append_line(formatted)
                else:
                    prefix = "[stderr]" if stream == "stderr" else ""
                    self._append_line(f"{prefix} {line}".strip())
            elif event_type == "update" and isinstance(payload, ProcessUpdate):
                self._apply_runner_update(payload)
                if payload.status in {"completed", "failed", "stopped"}:
                    command_text = " ".join(payload.command) if payload.command else "unknown command"
                    self._append_line(
                        f"[{payload.status}] {command_text} exited with code {payload.exit_code}"
                    )
        self.after(100, self._drain_events)


def main() -> None:
    app = ControlPanelApp()
    app.mainloop()


if __name__ == "__main__":
    main()

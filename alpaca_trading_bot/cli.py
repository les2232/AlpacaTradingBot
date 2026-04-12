from __future__ import annotations

import argparse
import contextlib
import os
import subprocess
import sys
from pathlib import Path
from typing import Callable


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="alpaca-bot",
        description="Single entry point for live trading, research, and the desktop control panel.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("live", help="Run the live bot with normal order-execution behavior.")
    subparsers.add_parser("preview", help="Run the live bot with order execution disabled.")

    dashboard_parser = subparsers.add_parser(
        "dashboard",
        help="Launch the Streamlit dashboard.",
    )
    dashboard_parser.add_argument(
        "--port",
        type=int,
        default=None,
        help="Optional Streamlit port override.",
    )

    subparsers.add_parser("control-panel", help="Launch the desktop control panel.")

    for name, help_text in [
        ("backtest", "Run the offline backtest CLI."),
        ("snapshot", "Build a dataset snapshot."),
        ("research", "Run the broader research pipeline."),
        ("experiments", "Run the backtest experiment batch."),
        ("report", "Generate the daily diagnostic report."),
    ]:
        subparsers.add_parser(name, help=help_text)

    return parser


@contextlib.contextmanager
def _temporary_argv(program_name: str, args: list[str]):
    original_argv = sys.argv[:]
    sys.argv = [program_name, *args]
    try:
        yield
    finally:
        sys.argv = original_argv


@contextlib.contextmanager
def _temporary_env(name: str, value: str | None):
    original = os.environ.get(name)
    if value is None:
        os.environ.pop(name, None)
    else:
        os.environ[name] = value
    try:
        yield
    finally:
        if original is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = original


def _run_module_main(program_name: str, args: list[str], entrypoint: Callable[[], None]) -> int:
    with _temporary_argv(program_name, args):
        entrypoint()
    return 0


def _run_live(preview: bool) -> int:
    import trading_bot

    env_value = "false" if preview else "true"
    with _temporary_env("EXECUTE_ORDERS", env_value):
        trading_bot.main()
    return 0


def _run_dashboard(port: int | None) -> int:
    command = [sys.executable, "-m", "streamlit", "run", "dashboard.py"]
    if port is not None:
        command.extend(["--server.port", str(port)])
    completed = subprocess.run(command, cwd=str(PROJECT_ROOT), check=False)
    return int(completed.returncode)


def main(argv: list[str] | None = None) -> int:
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    parser = build_parser()
    passthrough_commands = {"backtest", "snapshot", "research", "experiments", "report"}
    passthrough_args: list[str] = []
    if raw_argv and raw_argv[0] in passthrough_commands:
        passthrough_args = raw_argv[1:]
        args = parser.parse_args([raw_argv[0]])
    else:
        args = parser.parse_args(raw_argv)

    if args.command == "live":
        return _run_live(preview=False)
    if args.command == "preview":
        return _run_live(preview=True)
    if args.command == "dashboard":
        return _run_dashboard(args.port)
    if args.command == "control-panel":
        from desktop_app.app import main as control_panel_main

        control_panel_main()
        return 0
    if args.command == "backtest":
        from backtest_runner import main as backtest_main

        return _run_module_main("backtest_runner.py", passthrough_args, backtest_main)
    if args.command == "snapshot":
        from dataset_snapshotter import main as dataset_snapshotter_main

        return _run_module_main("dataset_snapshotter.py", passthrough_args, dataset_snapshotter_main)
    if args.command == "research":
        from run_research import main as research_main

        return _run_module_main("run_research.py", passthrough_args, research_main)
    if args.command == "experiments":
        from run_backtest_experiments import main as experiments_main

        return _run_module_main("run_backtest_experiments.py", passthrough_args, experiments_main)
    if args.command == "report":
        from daily_report import main as daily_report_main

        return _run_module_main("daily_report.py", passthrough_args, daily_report_main)

    parser.error(f"Unsupported command: {args.command}")
    return 2

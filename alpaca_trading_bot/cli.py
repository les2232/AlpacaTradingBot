from __future__ import annotations

import argparse
import contextlib
import json
import os
import socket
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable
from uuid import uuid4

from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parent.parent
LIVE_BOT_LOCK_PATH = PROJECT_ROOT / ".live_bot.lock"
LOG_ROOT = PROJECT_ROOT / "logs"
DEFAULT_DASHBOARD_PORT = 8501
DASHBOARD_READY_TIMEOUT_SECONDS = 15.0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="tradeos",
        description="TradeOS entry point for live trading, monitoring, and research workflows.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("live", help="Run the live bot with normal order-execution behavior.")
    subparsers.add_parser("paper", help="Alias for `live` while using the current paper-trading account.")
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


def _read_live_lock_metadata() -> dict[str, object]:
    try:
        raw_text = LIVE_BOT_LOCK_PATH.read_text(encoding="utf-8")
    except FileNotFoundError:
        return {}
    except OSError:
        return {}
    try:
        payload = json.loads(raw_text)
    except json.JSONDecodeError:
        return {}
    return payload if isinstance(payload, dict) else {}


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _startup_artifact_dir() -> Path:
    day_dir = LOG_ROOT / datetime.now().strftime("%Y-%m-%d")
    day_dir.mkdir(parents=True, exist_ok=True)
    return day_dir


def _startup_artifact_path(started_at_utc: str) -> Path:
    stamp = started_at_utc.replace("+00:00", "Z").replace(":", "").replace("-", "")
    return _startup_artifact_dir() / f"startup_config.{stamp}.json"


def _startup_artifact_latest_path() -> Path:
    return _startup_artifact_dir() / "startup_config.json"


def _pid_is_running(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


@contextlib.contextmanager
def _live_instance_lock():
    current_pid = os.getpid()
    lock_payload = {
        "pid": current_pid,
        "command": "tradeos live",
        "created_at_utc": _utcnow_iso(),
        "workspace": str(PROJECT_ROOT),
    }
    for _ in range(2):
        try:
            fd = os.open(str(LIVE_BOT_LOCK_PATH), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            existing = _read_live_lock_metadata()
            existing_pid = int(existing.get("pid", 0) or 0)
            if existing_pid and existing_pid != current_pid and _pid_is_running(existing_pid):
                raise RuntimeError(
                    "Refusing to start a second live bot instance. "
                    f"Another live process appears to be running with pid={existing_pid}. "
                    f"Lock file: {LIVE_BOT_LOCK_PATH}"
                )
            if existing_pid:
                print(
                    f"Recovered stale live bot lock at {LIVE_BOT_LOCK_PATH} "
                    f"(pid={existing_pid} no longer running)"
                )
            else:
                print(
                    f"Recovered malformed live bot lock at {LIVE_BOT_LOCK_PATH} "
                    "(missing or unreadable pid)"
                )
            try:
                LIVE_BOT_LOCK_PATH.unlink()
            except FileNotFoundError:
                continue
            except OSError as exc:
                raise RuntimeError(
                    f"Could not clear stale live lock at {LIVE_BOT_LOCK_PATH}: {exc}"
                ) from exc
            continue
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(lock_payload, handle)
            break
        except Exception:
            try:
                LIVE_BOT_LOCK_PATH.unlink()
            except OSError:
                pass
            raise
    else:
        raise RuntimeError(f"Could not acquire live bot lock at {LIVE_BOT_LOCK_PATH}")

    try:
        print(f"Live instance lock acquired: {LIVE_BOT_LOCK_PATH} (pid={current_pid})")
        yield LIVE_BOT_LOCK_PATH
    finally:
        current_lock = _read_live_lock_metadata()
        if int(current_lock.get("pid", 0) or 0) != current_pid:
            return
        try:
            LIVE_BOT_LOCK_PATH.unlink()
        except FileNotFoundError:
            pass


def _render_runtime_summary(config: object, preview: bool) -> None:
    mode = "preview" if preview else "live"
    paper_text = "paper" if bool(getattr(config, "paper", False)) else "live-account"
    symbols = list(getattr(config, "symbols", []))
    execution_text = "disabled" if preview else "enabled"
    symbols_preview = ", ".join(symbols[:5])
    if len(symbols) > 5:
        symbols_preview = f"{symbols_preview}, +{len(symbols) - 5} more"
    print(
        f"Starting {mode} mode | account={paper_text} | "
        f"strategy={getattr(config, 'strategy_mode', 'unknown')} | "
        f"timeframe={getattr(config, 'bar_timeframe_minutes', '?')}m | "
        f"execution={execution_text} | symbols={len(symbols)}"
    )
    if symbols_preview:
        print(f"Runtime symbols preview: {symbols_preview}")


def _persist_startup_config(details: object, preview: bool, *, session_id: str) -> Path:
    config = getattr(details, "config")
    started_at_utc = _utcnow_iso()
    payload = {
        "session_id": session_id,
        "started_at_utc": started_at_utc,
        "launch_mode": "preview" if preview else "live",
        "execution_enabled": not preview,
        "paper": bool(getattr(config, "paper", False)),
        "account_mode": "paper" if bool(getattr(config, "paper", False)) else "live-account",
        "strategy_mode": getattr(config, "strategy_mode", ""),
        "broker_backend": getattr(config, "broker_backend", "alpaca"),
        "bar_timeframe_minutes": int(getattr(config, "bar_timeframe_minutes", 0) or 0),
        "sma_bars": int(getattr(config, "sma_bars", 0) or 0),
        "symbols": list(getattr(config, "symbols", [])),
        "symbol_count": len(list(getattr(config, "symbols", []))),
        "runtime_config_path": getattr(details, "runtime_config_path", None),
        "runtime_overrides": list(getattr(details, "overridden_fields", ()) or ()),
        "max_usd_per_trade": float(getattr(config, "max_usd_per_trade", 0.0) or 0.0),
        "max_symbol_exposure_usd": float(getattr(config, "max_symbol_exposure_usd", 0.0) or 0.0),
        "max_open_positions": int(getattr(config, "max_open_positions", 0) or 0),
        "max_daily_loss_usd": float(getattr(config, "max_daily_loss_usd", 0.0) or 0.0),
        "max_orders_per_minute": int(getattr(config, "max_orders_per_minute", 0) or 0),
        "max_price_deviation_bps": float(getattr(config, "max_price_deviation_bps", 0.0) or 0.0),
        "max_live_price_age_seconds": int(getattr(config, "max_live_price_age_seconds", 0) or 0),
        "max_data_delay_seconds": int(getattr(config, "max_data_delay_seconds", 0) or 0),
        "db_path": os.getenv("BOT_DB_PATH", "bot_history.db"),
    }
    artifact_text = json.dumps(payload, indent=2)
    artifact_path = _startup_artifact_path(started_at_utc)
    artifact_path.write_text(artifact_text, encoding="utf-8")
    latest_path = _startup_artifact_latest_path()
    latest_path.write_text(artifact_text, encoding="utf-8")
    print(f"Persisted startup config: {artifact_path}")
    return artifact_path


def _validate_live_runtime(config: object, preview: bool) -> None:
    if preview:
        return
    if bool(getattr(config, "paper", False)):
        return
    raise RuntimeError(
        "Refusing to start live order execution with ALPACA_PAPER=false. "
        "This TradeOS workspace is currently being operated in paper-trading mode."
    )


def _run_live(preview: bool) -> int:
    import trading_bot

    load_dotenv(PROJECT_ROOT / ".env")
    details = trading_bot.load_config_details()
    config = details.config
    session_id = f"live-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}-{uuid4().hex[:8]}"
    _validate_live_runtime(config, preview=preview)
    _render_runtime_summary(config, preview=preview)
    env_value = "false" if preview else "true"
    live_lock = contextlib.nullcontext()
    if not preview:
        live_lock = _live_instance_lock()
    with live_lock:
        if not preview:
            _persist_startup_config(details, preview=preview, session_id=session_id)
        with _temporary_env("EXECUTE_ORDERS", env_value):
            trading_bot.main(config=config, session_id=session_id)
    return 0


def _dashboard_url(port: int) -> str:
    return f"http://localhost:{port}"


def _wait_for_dashboard_server(
    port: int,
    process: subprocess.Popen[bytes] | subprocess.Popen[str],
    *,
    timeout_seconds: float = DASHBOARD_READY_TIMEOUT_SECONDS,
) -> bool:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        if process.poll() is not None:
            return False
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=0.5):
                return True
        except OSError:
            time.sleep(0.2)
    return False


def _run_dashboard(port: int | None) -> int:
    command = [sys.executable, "-m", "streamlit", "run", "dashboard.py"]
    command.extend(["--server.headless", "true"])
    if port is not None:
        command.extend(["--server.port", str(port)])
    effective_port = port or DEFAULT_DASHBOARD_PORT
    process = subprocess.Popen(command, cwd=str(PROJECT_ROOT))
    url = _dashboard_url(effective_port)
    if _wait_for_dashboard_server(effective_port, process):
        print(f"Dashboard available at {url}")
    else:
        print(f"Dashboard is starting without auto-opening a browser. URL: {url}")
    return int(process.wait())


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

    if args.command in {"live", "paper"}:
        return _run_live(preview=False)
    if args.command == "preview":
        return _run_live(preview=True)
    if args.command == "dashboard":
        return _run_dashboard(args.port)
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

from __future__ import annotations

import argparse
import contextlib
import dataclasses
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
from daily_report import load_backtest_baseline


PROJECT_ROOT = Path(__file__).resolve().parent.parent
LIVE_BOT_LOCK_PATH = PROJECT_ROOT / ".live_bot.lock"
LOG_ROOT = PROJECT_ROOT / "logs"
DEFAULT_DASHBOARD_PORT = 8501
DASHBOARD_READY_TIMEOUT_SECONDS = 15.0
STARTUP_ARTIFACT_PATH_ENV = "TRADEOS_STARTUP_ARTIFACT_PATH"
ALLOW_UNAPPROVED_RUNTIME_ENV = "TRADEOS_ALLOW_UNAPPROVED_RUNTIME"
ALLOW_BASELINE_MISMATCH_ENV = "TRADEOS_ALLOW_BASELINE_MISMATCH"


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
    if os.name == "nt":
        identity = _read_process_identity(pid)
        return int(identity.get("pid", 0) or 0) == pid
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def _read_process_identity(pid: int) -> dict[str, object]:
    if pid <= 0:
        return {}
    if os.name == "nt":
        script = (
            "$proc = Get-CimInstance Win32_Process -Filter \"ProcessId = %d\" -ErrorAction SilentlyContinue; "
            "if ($null -eq $proc) { exit 0 }; "
            "$started = $null; "
            "if ($proc.CreationDate) { "
            "$started = ([System.Management.ManagementDateTimeConverter]::ToDateTime($proc.CreationDate)).ToUniversalTime().ToString('o') "
            "}; "
            "$payload = @{ "
            "pid = [int]$proc.ProcessId; "
            "started_at_utc = $started; "
            "command_line = $proc.CommandLine "
            "}; "
            "$payload | ConvertTo-Json -Compress"
        ) % pid
        try:
            result = subprocess.run(
                ["powershell", "-NoProfile", "-Command", script],
                capture_output=True,
                text=True,
                timeout=5,
                check=False,
            )
        except (OSError, subprocess.SubprocessError):
            return {}
        if result.returncode != 0 or not result.stdout.strip():
            fallback_script = (
                "$proc = Get-Process -Id %d -ErrorAction SilentlyContinue; "
                "if ($null -eq $proc) { exit 0 }; "
                "$started = $null; "
                "if ($proc.StartTime) { "
                "$started = $proc.StartTime.ToUniversalTime().ToString('o') "
                "}; "
                "$payload = @{ "
                "pid = [int]$proc.Id; "
                "started_at_utc = $started; "
                "command_line = 'python -m tradeos live' "
                "}; "
                "$payload | ConvertTo-Json -Compress"
            ) % pid
            try:
                fallback = subprocess.run(
                    ["powershell", "-NoProfile", "-Command", fallback_script],
                    capture_output=True,
                    text=True,
                    timeout=5,
                    check=False,
                )
            except (OSError, subprocess.SubprocessError):
                return {}
            if fallback.returncode != 0 or not fallback.stdout.strip():
                return {}
            try:
                payload = json.loads(fallback.stdout)
            except json.JSONDecodeError:
                return {}
            return payload if isinstance(payload, dict) else {}
        try:
            payload = json.loads(result.stdout)
        except json.JSONDecodeError:
            return {}
        return payload if isinstance(payload, dict) else {}
    try:
        result = subprocess.run(
            ["ps", "-o", "pid=", "-o", "lstart=", "-o", "command=", "-p", str(pid)],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return {}
    output = result.stdout.strip()
    if result.returncode != 0 or not output:
        return {}
    parts = output.split(None, 6)
    if len(parts) < 7:
        return {}
    try:
        resolved_pid = int(parts[0])
    except ValueError:
        return {}
    started_at_text = " ".join(parts[1:6])
    command_line = parts[6]
    try:
        started_at_utc = datetime.strptime(started_at_text, "%a %b %d %H:%M:%S %Y").replace(
            tzinfo=timezone.utc
        ).isoformat()
    except ValueError:
        started_at_utc = None
    return {
        "pid": resolved_pid,
        "started_at_utc": started_at_utc,
        "command_line": command_line,
    }


def _process_looks_like_tradeos_live(identity: dict[str, object]) -> bool:
    command_line = str(identity.get("command_line", "") or "").lower()
    return "tradeos" in command_line and "live" in command_line


def _live_lock_matches_running_process(metadata: dict[str, object], *, expected_pid: int | None = None) -> bool:
    pid = int(metadata.get("pid", 0) or 0)
    if pid <= 0 or not _pid_is_running(pid):
        return False
    if expected_pid is not None and pid != expected_pid:
        return False
    identity = _read_process_identity(pid)
    if not identity:
        return False
    if int(identity.get("pid", 0) or 0) != pid:
        return False
    if not _process_looks_like_tradeos_live(identity):
        return False
    locked_started_at = str(metadata.get("process_started_at_utc", "") or "").strip()
    if locked_started_at:
        return str(identity.get("started_at_utc", "") or "").strip() == locked_started_at
    return str(metadata.get("command", "") or "").strip().lower() == "tradeos live"


@contextlib.contextmanager
def _live_instance_lock():
    current_pid = os.getpid()
    process_identity = _read_process_identity(current_pid)
    lock_payload = {
        "pid": current_pid,
        "command": "tradeos live",
        "created_at_utc": _utcnow_iso(),
        "workspace": str(PROJECT_ROOT),
        "process_started_at_utc": process_identity.get("started_at_utc"),
    }
    for _ in range(2):
        try:
            fd = os.open(str(LIVE_BOT_LOCK_PATH), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            existing = _read_live_lock_metadata()
            existing_pid = int(existing.get("pid", 0) or 0)
            if existing_pid and existing_pid != current_pid and _live_lock_matches_running_process(existing):
                raise RuntimeError(
                    "Refusing to start a second live bot instance. "
                    f"Another live process appears to be running with pid={existing_pid}. "
                    f"Lock file: {LIVE_BOT_LOCK_PATH}"
                )
            if existing_pid:
                print(
                    f"Recovered stale live bot lock at {LIVE_BOT_LOCK_PATH} "
                    f"(pid={existing_pid} was not a matching tradeos live process)"
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
    print(
        "Data parity | "
        f"historical={getattr(config, 'historical_feed', 'unknown')} | "
        f"live={getattr(config, 'live_feed', 'unknown')} | "
        f"latest={getattr(config, 'latest_bar_feed', 'unknown')} | "
        f"bar_build={getattr(config, 'bar_build_mode', 'unknown')}"
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
        "runtime_config_approved": getattr(details, "runtime_config_approved", None),
        "runtime_config_rejection_reasons": list(getattr(details, "runtime_config_rejection_reasons", ()) or ()),
        "baseline_valid_for_comparison": getattr(details, "baseline_valid_for_comparison", None),
        "baseline_validation_errors": list(getattr(details, "baseline_validation_errors", ()) or ()),
        "max_usd_per_trade": float(getattr(config, "max_usd_per_trade", 0.0) or 0.0),
        "max_symbol_exposure_usd": float(getattr(config, "max_symbol_exposure_usd", 0.0) or 0.0),
        "max_open_positions": int(getattr(config, "max_open_positions", 0) or 0),
        "max_daily_loss_usd": float(getattr(config, "max_daily_loss_usd", 0.0) or 0.0),
        "max_orders_per_minute": int(getattr(config, "max_orders_per_minute", 0) or 0),
        "max_price_deviation_bps": float(getattr(config, "max_price_deviation_bps", 0.0) or 0.0),
        "max_live_price_age_seconds": int(getattr(config, "max_live_price_age_seconds", 0) or 0),
        "max_data_delay_seconds": int(getattr(config, "max_data_delay_seconds", 0) or 0),
        "ml_lookback_bars": int(getattr(config, "ml_lookback_bars", 0) or 0),
        "breakout_max_stop_pct": float(getattr(config, "breakout_max_stop_pct", 0.0) or 0.0),
        "sma_stop_pct": float(getattr(config, "sma_stop_pct", 0.0) or 0.0),
        "mean_reversion_exit_style": str(getattr(config, "mean_reversion_exit_style", "") or ""),
        "mean_reversion_max_atr_percentile": float(getattr(config, "mean_reversion_max_atr_percentile", 0.0) or 0.0),
        "mean_reversion_trend_filter": bool(getattr(config, "mean_reversion_trend_filter", False)),
        "mean_reversion_trend_slope_filter": bool(getattr(config, "mean_reversion_trend_slope_filter", False)),
        "mean_reversion_stop_pct": float(getattr(config, "mean_reversion_stop_pct", 0.0) or 0.0),
        "db_path": os.getenv("BOT_DB_PATH", "bot_history.db"),
        "historical_feed": str(getattr(config, "historical_feed", "") or ""),
        "live_feed": str(getattr(config, "live_feed", "") or ""),
        "latest_bar_feed": str(getattr(config, "latest_bar_feed", "") or ""),
        "bar_build_mode": str(getattr(config, "bar_build_mode", "") or ""),
        "apply_updated_bars": bool(getattr(config, "apply_updated_bars", False)),
        "post_bar_reconcile_poll": bool(getattr(config, "post_bar_reconcile_poll", False)),
        "block_trading_until_resync": bool(getattr(config, "block_trading_until_resync", False)),
        "assert_feed_on_startup": bool(getattr(config, "assert_feed_on_startup", False)),
        "log_bar_components": bool(getattr(config, "log_bar_components", False)),
        "resync_status": None,
        "resync_started_at_utc": None,
        "resync_completed_at_utc": None,
        "resync_reason_codes": [],
        "resync_positions_recovered": [],
        "resync_open_orders_recovered": [],
        "resync_recent_fills_recovered": [],
        "resync_discrepancies": [],
        "gate_allows_entries": None,
        "gate_allows_exits": None,
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


def _validate_runtime_config_approval(details: object, preview: bool) -> None:
    if preview:
        return
    approved = getattr(details, "runtime_config_approved", None)
    if approved is not False:
        return
    if os.getenv(ALLOW_UNAPPROVED_RUNTIME_ENV, "").strip().lower() in {"1", "true", "yes"}:
        print(
            f"WARNING: runtime config approval override enabled via {ALLOW_UNAPPROVED_RUNTIME_ENV}; "
            "continuing with an unapproved runtime config."
        )
        return
    runtime_path = getattr(details, "runtime_config_path", None) or "runtime config"
    reasons = list(getattr(details, "runtime_config_rejection_reasons", ()) or ())
    reasons_text = "; ".join(reasons) if reasons else "no rejection reasons were recorded"
    raise RuntimeError(
        "Refusing to start live trading with an unapproved runtime config. "
        f"{runtime_path} is marked approved=false ({reasons_text}). "
        f"To override intentionally, set {ALLOW_UNAPPROVED_RUNTIME_ENV}=true."
    )


def _attach_baseline_validation(details: object) -> object:
    _, source = load_backtest_baseline(PROJECT_ROOT)
    baseline_valid = source.get("valid_for_comparison", True)
    baseline_errors = tuple(
        str(item) for item in source.get("validation_errors", []) if str(item).strip()
    )
    if dataclasses.is_dataclass(details):
        field_names = {field.name for field in dataclasses.fields(details)}
        updates: dict[str, object] = {}
        if "baseline_valid_for_comparison" in field_names:
            updates["baseline_valid_for_comparison"] = baseline_valid
        if "baseline_validation_errors" in field_names:
            updates["baseline_validation_errors"] = baseline_errors
        if updates:
            return dataclasses.replace(details, **updates)
        return details
    setattr(details, "baseline_valid_for_comparison", baseline_valid)
    setattr(details, "baseline_validation_errors", baseline_errors)
    return details


def _validate_runtime_baseline(details: object, preview: bool) -> None:
    if preview:
        return
    valid = getattr(details, "baseline_valid_for_comparison", True)
    if valid is not False:
        return
    if os.getenv(ALLOW_BASELINE_MISMATCH_ENV, "").strip().lower() in {"1", "true", "yes"}:
        print(
            f"WARNING: baseline mismatch override enabled via {ALLOW_BASELINE_MISMATCH_ENV}; "
            "continuing despite runtime/research drift."
        )
        return
    runtime_path = getattr(details, "runtime_config_path", None) or "runtime config"
    reasons = list(getattr(details, "baseline_validation_errors", ()) or ())
    reasons_text = "; ".join(reasons) if reasons else "baseline validation failed"
    raise RuntimeError(
        "Refusing to start live trading with a runtime config that does not match the promoted baseline. "
        f"{runtime_path} failed baseline validation ({reasons_text}). "
        f"To override intentionally, set {ALLOW_BASELINE_MISMATCH_ENV}=true."
    )


def _run_live(preview: bool) -> int:
    load_dotenv(PROJECT_ROOT / ".env")
    live_lock = contextlib.nullcontext()
    if not preview:
        live_lock = _live_instance_lock()
    with live_lock:
        import trading_bot

        details = _attach_baseline_validation(trading_bot.load_config_details())
        config = details.config
        session_id = f"live-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}-{uuid4().hex[:8]}"
        _validate_live_runtime(config, preview=preview)
        _validate_runtime_config_approval(details, preview=preview)
        _validate_runtime_baseline(details, preview=preview)
        _render_runtime_summary(config, preview=preview)
        env_value = "false" if preview else "true"
        startup_artifact_path: Path | None = None
        if not preview:
            startup_artifact_path = _persist_startup_config(details, preview=preview, session_id=session_id)
        startup_artifact_env = str(startup_artifact_path) if startup_artifact_path is not None else ""
        with _temporary_env("EXECUTE_ORDERS", env_value):
            with _temporary_env(STARTUP_ARTIFACT_PATH_ENV, startup_artifact_env):
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

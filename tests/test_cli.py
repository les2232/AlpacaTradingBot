import sys
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

import pytest

from alpaca_trading_bot import cli
from trading_bot import BotConfig, RuntimeConfigDetails


def _cleanup_tree(path: Path) -> None:
    if not path.exists():
        return
    for child in sorted(path.rglob("*"), reverse=True):
        if child.is_file():
            child.unlink()
        else:
            child.rmdir()
    path.rmdir()


def test_cli_parser_accepts_preview() -> None:
    args = cli.build_parser().parse_args(["preview"])
    assert args.command == "preview"


def test_cli_parser_accepts_paper_shadow_and_unambiguous_repair_digests() -> None:
    shadow = cli.build_parser().parse_args(["paper-shadow"])
    repair = cli.build_parser().parse_args(
        [
            "repair-local-state",
            "--symbol",
            "COP",
            "--apply",
            "--repair-id",
            "20260806T000000Z-123456789abc",
            "--approved-db-raw-sha256",
            "a" * 64,
            "--approved-db-state-sha256",
            "b" * 64,
            "--approved-plan-sha256",
            "c" * 64,
            "--approved-evidence-sha256",
            "d" * 64,
        ]
    )

    assert shadow.command == "paper-shadow"
    assert repair.approved_db_raw_sha256 == "a" * 64
    assert repair.approved_db_state_sha256 == "b" * 64
    assert repair.approved_plan_sha256 == "c" * 64


def test_paper_shadow_uses_named_profile_writer_locks_and_explicit_deny(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from automation_safety import account_fingerprint

    expected_account = "paper-account-test"
    calls: dict[str, object] = {"locks": []}
    config = SimpleNamespace(
        paper=True,
        broker_backend="alpaca",
        strategy_mode="mean_reversion",
        bar_timeframe_minutes=15,
        symbols=["AMD"],
        historical_feed="iex",
        live_feed="iex",
        latest_bar_feed="iex",
        bar_build_mode="stream_minute_aggregate",
    )
    details = SimpleNamespace(
        config=config,
        runtime_config_path=str(cli.PAPER_SHADOW_CONFIG_PATH),
        overridden_fields=("symbols",),
        runtime_config_approved=True,
        runtime_config_rejection_reasons=(),
        baseline_valid_for_comparison=True,
        baseline_validation_errors=(),
        launch_profile="paper_shadow",
        broker_environment="paper",
        submission_policy="deny",
        expected_account_fingerprint=account_fingerprint(expected_account),
        dashboard_mode="read_only",
    )

    class TrackingLock:
        def __init__(self, name: str) -> None:
            self.name = name

        def __enter__(self):
            calls["locks"].append(f"enter:{self.name}")  # type: ignore[union-attr]
            return self

        def __exit__(self, exc_type, exc, tb) -> None:
            calls["locks"].append(f"exit:{self.name}")  # type: ignore[union-attr]

    def fake_main(**kwargs: object) -> None:
        calls["main"] = kwargs
        calls["runtime_path"] = cli.os.environ.get("BOT_RUNTIME_CONFIG_PATH")
        calls["execute_env"] = cli.os.environ.get("EXECUTE_ORDERS")

    fake_module = SimpleNamespace(
        load_config_details=lambda: details,
        main=fake_main,
    )
    artifact = tmp_path / "shadow-startup.json"
    artifact.write_text("{}", encoding="utf-8")
    monkeypatch.setitem(sys.modules, "trading_bot", fake_module)
    monkeypatch.setattr(cli, "load_dotenv", lambda *args, **kwargs: None)
    monkeypatch.setattr(cli, "_attach_baseline_validation", lambda value: value)
    monkeypatch.setattr(cli, "ExecutionLock", lambda: TrackingLock("kernel"))
    monkeypatch.setattr(cli, "_live_instance_lock", lambda: TrackingLock("legacy"))
    monkeypatch.setattr(
        cli,
        "_persist_startup_config",
        lambda *args, **kwargs: artifact,
    )
    monkeypatch.setenv("ALPACA_EXPECTED_ACCOUNT_ID", expected_account)
    monkeypatch.setenv("ALPACA_PAPER", "true")
    monkeypatch.setenv("BOT_RUNTIME_CONFIG_PATH", "ambient-wrong-config.json")

    assert cli._run_live(preview=True, paper_shadow=True) == 0

    assert calls["runtime_path"] == str(cli.PAPER_SHADOW_CONFIG_PATH)
    assert calls["execute_env"] == "false"
    assert calls["locks"] == [
        "enter:kernel",
        "enter:legacy",
        "exit:legacy",
        "exit:kernel",
    ]
    main_call = calls["main"]
    assert isinstance(main_call, dict)
    assert main_call["execute_orders"] is False
    assert main_call["require_reconciliation_ready"] is True
    assert main_call["durable_decision_cycles"] is True
    assert cli.os.environ["BOT_RUNTIME_CONFIG_PATH"] == "ambient-wrong-config.json"


def test_paper_shadow_account_binding_mismatch_fails_before_bot_main(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = SimpleNamespace(paper=True, broker_backend="alpaca")
    details = SimpleNamespace(
        config=config,
        runtime_config_path=str(cli.PAPER_SHADOW_CONFIG_PATH),
        launch_profile="paper_shadow",
        broker_environment="paper",
        submission_policy="deny",
        expected_account_fingerprint="0" * 64,
        dashboard_mode="read_only",
    )
    called = False

    def fake_main(**kwargs: object) -> None:
        nonlocal called
        called = True

    monkeypatch.setenv("ALPACA_EXPECTED_ACCOUNT_ID", "different-account")
    with pytest.raises(RuntimeError, match="binding does not match"):
        cli._validate_paper_shadow_runtime(details)

    assert called is False


def test_cli_parser_rejects_retired_control_panel() -> None:
    with pytest.raises(SystemExit):
        cli.build_parser().parse_args(["control-panel"])


def test_main_forwards_backtest_passthrough_args(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def fake_run_module_main(program_name: str, args: list[str], entrypoint) -> int:
        captured["program_name"] = program_name
        captured["args"] = args
        return 0

    monkeypatch.setattr(cli, "_run_module_main", fake_run_module_main)

    exit_code = cli.main(["backtest", "--dataset", "datasets/sample", "--strategy-mode", "sma"])

    assert exit_code == 0
    assert captured["program_name"] == "backtest_runner.py"
    assert captured["args"] == ["--dataset", "datasets/sample", "--strategy-mode", "sma"]


def test_run_dashboard_starts_streamlit_headless_and_reports_url(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    captured: dict[str, object] = {}

    class FakeProcess:
        def poll(self):
            return None

        def wait(self) -> int:
            return 0

    def fake_popen(command, cwd=None, env=None):
        captured["command"] = command
        captured["cwd"] = cwd
        captured["env"] = env
        return FakeProcess()

    monkeypatch.setattr(cli.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(cli, "_dashboard_port_is_open", lambda port: False)
    monkeypatch.setattr(cli, "_wait_for_dashboard_server", lambda port, process: True)

    exit_code = cli._run_dashboard(port=None)
    output = capsys.readouterr().out

    assert exit_code == 0
    assert captured["command"] == [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        "dashboard.py",
        "--server.headless",
        "true",
    ]
    assert captured["cwd"] == str(cli.PROJECT_ROOT)
    assert captured["env"] is None
    assert "Dashboard available at http://localhost:8501" in output


def test_run_dashboard_honors_custom_port(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}

    class FakeProcess:
        def poll(self):
            return None

        def wait(self) -> int:
            return 0

    def fake_popen(command, cwd=None, env=None):
        captured["command"] = command
        return FakeProcess()

    monkeypatch.setattr(cli.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(cli, "_dashboard_port_is_open", lambda port: False)
    monkeypatch.setattr(cli, "_wait_for_dashboard_server", lambda port, process: False)

    exit_code = cli._run_dashboard(port=9123)

    assert exit_code == 0
    assert captured["command"] == [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        "dashboard.py",
        "--server.headless",
        "true",
        "--server.port",
        "9123",
    ]


def test_run_dashboard_refuses_unverified_occupied_port(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(cli, "_dashboard_port_is_open", lambda port: True)
    monkeypatch.setattr(
        cli.subprocess,
        "Popen",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("dashboard process must not start")
        ),
    )

    with pytest.raises(RuntimeError, match="database authority cannot be verified"):
        cli._run_dashboard(port=8501)


def test_run_live_refuses_non_paper_execution(monkeypatch: pytest.MonkeyPatch) -> None:
    config = SimpleNamespace(
        paper=False,
        strategy_mode="mean_reversion",
        bar_timeframe_minutes=15,
        symbols=["AMD", "MSFT"],
    )
    fake_module = SimpleNamespace(
        load_config_details=lambda: SimpleNamespace(
            config=config,
            runtime_config_path=None,
            overridden_fields=(),
            runtime_config_approved=None,
            runtime_config_rejection_reasons=(),
        ),
        main=lambda **kwargs: None,
    )
    monkeypatch.setitem(sys.modules, "trading_bot", fake_module)
    monkeypatch.setattr(cli, "_live_instance_lock", cli.contextlib.nullcontext)

    with pytest.raises(RuntimeError, match="ALPACA_PAPER=false"):
        cli._run_live(preview=False)


def test_run_preview_refuses_non_paper_without_starting_bot(monkeypatch: pytest.MonkeyPatch) -> None:
    called: dict[str, bool] = {"main": False}
    config = SimpleNamespace(
        paper=False,
        strategy_mode="mean_reversion",
        bar_timeframe_minutes=15,
        symbols=["AMD"],
    )

    def fake_main(
        *,
        config=None,
        session_id=None,
        execute_orders=None,
        require_reconciliation_ready=None,
        durable_decision_cycles=None,
    ) -> None:
        called["main"] = True
        assert cli.os.environ["EXECUTE_ORDERS"] == "false"
        assert execute_orders is False
        assert require_reconciliation_ready is False
        assert config is not None
        assert session_id is not None

    fake_module = SimpleNamespace(
        load_config_details=lambda: SimpleNamespace(
            config=config,
            runtime_config_path=None,
            overridden_fields=(),
            runtime_config_approved=None,
            runtime_config_rejection_reasons=(),
        ),
        main=fake_main,
    )
    monkeypatch.setitem(sys.modules, "trading_bot", fake_module)

    with pytest.raises(RuntimeError, match="ALPACA_PAPER=false"):
        cli._run_live(preview=True)

    assert called["main"] is False


def test_run_live_refuses_second_live_instance(monkeypatch: pytest.MonkeyPatch) -> None:
    lock_path = cli.PROJECT_ROOT / f"test_live_lock_{uuid4().hex}.lock"
    log_root = cli.PROJECT_ROOT / f"test_logs_{uuid4().hex}"
    lock_path.write_text(
        '{"pid": 4242, "command": "tradeos live", "process_started_at_utc": "2026-04-16T14:00:00+00:00"}',
        encoding="utf-8",
    )
    config = SimpleNamespace(
        paper=True,
        strategy_mode="mean_reversion",
        bar_timeframe_minutes=15,
        symbols=["AMD"],
        sma_bars=20,
        max_usd_per_trade=200.0,
        max_symbol_exposure_usd=200.0,
        max_open_positions=3,
        max_daily_loss_usd=300.0,
        max_orders_per_minute=6,
        max_price_deviation_bps=75.0,
        max_live_price_age_seconds=60,
        max_data_delay_seconds=300,
    )

    fake_module = SimpleNamespace(
        load_config_details=lambda: SimpleNamespace(
            config=config,
            runtime_config_path="config/live_config.json",
            overridden_fields=("symbols",),
            runtime_config_approved=True,
            runtime_config_rejection_reasons=(),
        ),
        main=lambda **kwargs: None,
    )
    monkeypatch.setitem(sys.modules, "trading_bot", fake_module)
    monkeypatch.setattr(cli, "LIVE_BOT_LOCK_PATH", lock_path)
    monkeypatch.setattr(cli, "LOG_ROOT", log_root)
    monkeypatch.setattr(cli, "_pid_is_running", lambda pid: pid == 4242)
    monkeypatch.setattr(
        cli,
        "_read_process_identity",
        lambda pid: {
            "pid": pid,
            "started_at_utc": "2026-04-16T14:00:00+00:00",
            "command_line": "python -m tradeos live",
        },
    )

    try:
        with pytest.raises(RuntimeError, match="second live bot instance"):
            cli._run_live(preview=False)
    finally:
        lock_path.unlink(missing_ok=True)
        _cleanup_tree(log_root)


def test_run_live_replaces_stale_lock_and_cleans_up(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    lock_path = cli.PROJECT_ROOT / f"test_live_lock_{uuid4().hex}.lock"
    log_root = cli.PROJECT_ROOT / f"test_logs_{uuid4().hex}"
    lock_path.write_text('{"pid": 999999, "command": "tradeos live"}', encoding="utf-8")
    called: dict[str, bool] = {"main": False}
    config = SimpleNamespace(
        paper=True,
        strategy_mode="mean_reversion",
        bar_timeframe_minutes=15,
        symbols=["AMD"],
        sma_bars=20,
        max_usd_per_trade=200.0,
        max_symbol_exposure_usd=200.0,
        max_open_positions=3,
        max_daily_loss_usd=300.0,
        max_orders_per_minute=6,
        max_price_deviation_bps=75.0,
        max_live_price_age_seconds=60,
        max_data_delay_seconds=300,
    )

    def fake_main(
        *,
        config=None,
        session_id=None,
        execute_orders=None,
        require_reconciliation_ready=None,
        durable_decision_cycles=None,
    ) -> None:
        called["main"] = True
        assert lock_path.exists()
        assert config is not None
        assert session_id is not None

    fake_module = SimpleNamespace(
        load_config_details=lambda: SimpleNamespace(
            config=config,
            runtime_config_path="config/live_config.json",
            overridden_fields=("symbols",),
            runtime_config_approved=True,
            runtime_config_rejection_reasons=(),
        ),
        main=fake_main,
    )
    monkeypatch.setitem(sys.modules, "trading_bot", fake_module)
    monkeypatch.setattr(cli, "LIVE_BOT_LOCK_PATH", lock_path)
    monkeypatch.setattr(cli, "LOG_ROOT", log_root)
    monkeypatch.setattr(cli, "_pid_is_running", lambda pid: False)
    monkeypatch.setattr(
        cli,
        "_read_process_identity",
        lambda pid: {
            "pid": pid,
            "started_at_utc": "2026-04-16T14:00:00+00:00",
            "command_line": "python -m tradeos live",
        }
        if pid == cli.os.getpid()
        else {},
    )

    try:
        exit_code = cli._run_live(preview=False)

        assert exit_code == 0
        assert called["main"] is True
        assert not lock_path.exists()
    finally:
        lock_path.unlink(missing_ok=True)
        _cleanup_tree(log_root)


def test_run_live_reports_lock_acquired(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    lock_path = cli.PROJECT_ROOT / f"test_live_lock_{uuid4().hex}.lock"
    log_root = cli.PROJECT_ROOT / f"test_logs_{uuid4().hex}"
    config = SimpleNamespace(
        paper=True,
        strategy_mode="mean_reversion",
        bar_timeframe_minutes=15,
        symbols=["AMD", "MSFT", "QCOM", "JPM", "HD", "XOM"],
        sma_bars=20,
        max_usd_per_trade=200.0,
        max_symbol_exposure_usd=200.0,
        max_open_positions=3,
        max_daily_loss_usd=300.0,
        max_orders_per_minute=6,
        max_price_deviation_bps=75.0,
        max_live_price_age_seconds=60,
        max_data_delay_seconds=300,
    )

    fake_module = SimpleNamespace(
        load_config_details=lambda: SimpleNamespace(
            config=config,
            runtime_config_path="config/live_config.json",
            overridden_fields=("symbols", "strategy_mode"),
            runtime_config_approved=True,
            runtime_config_rejection_reasons=(),
        ),
        main=lambda **kwargs: None,
    )
    monkeypatch.setitem(sys.modules, "trading_bot", fake_module)
    monkeypatch.setattr(cli, "LIVE_BOT_LOCK_PATH", lock_path)
    monkeypatch.setattr(cli, "LOG_ROOT", log_root)
    monkeypatch.setattr(
        cli,
        "_read_process_identity",
        lambda pid: {
            "pid": pid,
            "started_at_utc": "2026-04-16T14:00:00+00:00",
            "command_line": "python -m tradeos live",
        },
    )

    try:
        exit_code = cli._run_live(preview=False)
        output = capsys.readouterr().out

        assert exit_code == 0
        assert "execution=enabled" in output
        assert "Runtime symbols preview: AMD, MSFT, QCOM, JPM, HD, +1 more" in output
        assert "Live instance lock acquired" in output
    finally:
        lock_path.unlink(missing_ok=True)
        _cleanup_tree(log_root)


def test_live_instance_lock_never_suppresses_body_error_when_metadata_changes(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    lock_path = tmp_path / "live.lock"
    monkeypatch.setattr(cli, "LIVE_BOT_LOCK_PATH", lock_path)

    with pytest.raises(RuntimeError, match="runtime failed"):
        with cli._live_instance_lock():
            lock_path.write_text('{"pid": 999999}', encoding="utf-8")
            raise RuntimeError("runtime failed")

    assert lock_path.exists()


def test_run_live_loads_dotenv_before_resolving_config(monkeypatch: pytest.MonkeyPatch) -> None:
    called: list[str] = []
    config = SimpleNamespace(
        paper=True,
        strategy_mode="mean_reversion",
        bar_timeframe_minutes=15,
        symbols=["AMD"],
        sma_bars=20,
        max_usd_per_trade=1000.0,
        max_symbol_exposure_usd=1000.0,
        max_open_positions=5,
        max_daily_loss_usd=300.0,
        max_orders_per_minute=6,
        max_price_deviation_bps=75.0,
        max_live_price_age_seconds=60,
        max_data_delay_seconds=300,
    )

    def fake_load_config_details():
        called.append("load_config_details")
        return SimpleNamespace(
            config=config,
            runtime_config_path="config/live_config.json",
            overridden_fields=("symbols",),
            runtime_config_approved=True,
            runtime_config_rejection_reasons=(),
        )

    fake_module = SimpleNamespace(
        load_config_details=fake_load_config_details,
        main=lambda **kwargs: None,
    )

    def fake_load_dotenv(path):
        called.append("load_dotenv")
        return False

    monkeypatch.setitem(sys.modules, "trading_bot", fake_module)
    monkeypatch.setattr(cli, "load_dotenv", fake_load_dotenv)
    monkeypatch.setattr(
        cli,
        "_read_process_identity",
        lambda pid: {
            "pid": pid,
            "started_at_utc": "2026-04-16T14:00:00+00:00",
            "command_line": "python -m tradeos live",
        },
    )

    exit_code = cli._run_live(preview=True)

    assert exit_code == 0
    assert called[:2] == ["load_dotenv", "load_config_details"]


def test_run_live_acquires_lock_before_loading_config(monkeypatch: pytest.MonkeyPatch) -> None:
    lock_path = cli.PROJECT_ROOT / f"test_live_lock_{uuid4().hex}.lock"
    log_root = cli.PROJECT_ROOT / f"test_logs_{uuid4().hex}"
    observed: dict[str, bool] = {"lock_seen_during_load": False, "lock_seen_during_main": False}
    config = SimpleNamespace(
        paper=True,
        strategy_mode="mean_reversion",
        bar_timeframe_minutes=15,
        symbols=["AMD"],
        sma_bars=20,
        max_usd_per_trade=200.0,
        max_symbol_exposure_usd=200.0,
        max_open_positions=3,
        max_daily_loss_usd=300.0,
        max_orders_per_minute=6,
        max_price_deviation_bps=75.0,
        max_live_price_age_seconds=60,
        max_data_delay_seconds=300,
    )

    def fake_load_config_details():
        observed["lock_seen_during_load"] = lock_path.exists()
        return SimpleNamespace(
            config=config,
            runtime_config_path="config/live_config.json",
            overridden_fields=("symbols",),
            runtime_config_approved=True,
            runtime_config_rejection_reasons=(),
        )

    def fake_main(
        *,
        config=None,
        session_id=None,
        execute_orders=None,
        require_reconciliation_ready=None,
        durable_decision_cycles=None,
    ) -> None:
        observed["lock_seen_during_main"] = lock_path.exists()
        assert config is not None
        assert session_id is not None

    fake_module = SimpleNamespace(
        load_config_details=fake_load_config_details,
        main=fake_main,
    )
    monkeypatch.setitem(sys.modules, "trading_bot", fake_module)
    monkeypatch.setattr(cli, "LIVE_BOT_LOCK_PATH", lock_path)
    monkeypatch.setattr(cli, "LOG_ROOT", log_root)
    monkeypatch.setattr(
        cli,
        "_read_process_identity",
        lambda pid: {
            "pid": pid,
            "started_at_utc": "2026-04-16T14:00:00+00:00",
            "command_line": "python -m tradeos live",
        },
    )

    try:
        exit_code = cli._run_live(preview=False)

        assert exit_code == 0
        assert observed["lock_seen_during_load"] is True
        assert observed["lock_seen_during_main"] is True
        assert not lock_path.exists()
    finally:
        lock_path.unlink(missing_ok=True)
        _cleanup_tree(log_root)


def test_run_live_recovers_malformed_lock(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    lock_path = cli.PROJECT_ROOT / f"test_live_lock_{uuid4().hex}.lock"
    log_root = cli.PROJECT_ROOT / f"test_logs_{uuid4().hex}"
    lock_path.write_text("not-json", encoding="utf-8")
    config = SimpleNamespace(
        paper=True,
        strategy_mode="mean_reversion",
        bar_timeframe_minutes=15,
        symbols=["AMD"],
        sma_bars=20,
        max_usd_per_trade=200.0,
        max_symbol_exposure_usd=200.0,
        max_open_positions=3,
        max_daily_loss_usd=300.0,
        max_orders_per_minute=6,
        max_price_deviation_bps=75.0,
        max_live_price_age_seconds=60,
        max_data_delay_seconds=300,
    )

    fake_module = SimpleNamespace(
        load_config_details=lambda: SimpleNamespace(
            config=config,
            runtime_config_path="config/live_config.json",
            overridden_fields=("symbols",),
            runtime_config_approved=True,
            runtime_config_rejection_reasons=(),
        ),
        main=lambda **kwargs: None,
    )
    monkeypatch.setitem(sys.modules, "trading_bot", fake_module)
    monkeypatch.setattr(cli, "LIVE_BOT_LOCK_PATH", lock_path)
    monkeypatch.setattr(cli, "LOG_ROOT", log_root)
    monkeypatch.setattr(
        cli,
        "_read_process_identity",
        lambda pid: {
            "pid": pid,
            "started_at_utc": "2026-04-16T14:00:00+00:00",
            "command_line": "python -m tradeos live",
        },
    )

    try:
        exit_code = cli._run_live(preview=False)
        output = capsys.readouterr().out

        assert exit_code == 0
        assert "Recovered malformed live bot lock" in output
        assert not lock_path.exists()
    finally:
        lock_path.unlink(missing_ok=True)
        _cleanup_tree(log_root)


def test_run_live_refuses_unapproved_runtime_config(monkeypatch: pytest.MonkeyPatch) -> None:
    config = SimpleNamespace(
        paper=True,
        strategy_mode="mean_reversion",
        bar_timeframe_minutes=15,
        symbols=["AMD"],
    )
    fake_module = SimpleNamespace(
        load_config_details=lambda: SimpleNamespace(
            config=config,
            runtime_config_path="config/live_config.json",
            overridden_fields=("symbols",),
            runtime_config_approved=False,
            runtime_config_rejection_reasons=("profit_factor 1.0169 >= 1.2",),
            baseline_valid_for_comparison=True,
            baseline_validation_errors=(),
        ),
        main=lambda **kwargs: None,
    )
    monkeypatch.setitem(sys.modules, "trading_bot", fake_module)
    monkeypatch.delenv(cli.ALLOW_UNAPPROVED_RUNTIME_ENV, raising=False)
    monkeypatch.setattr(cli, "_attach_baseline_validation", lambda details: details)
    monkeypatch.setattr(cli, "_live_instance_lock", cli.contextlib.nullcontext)

    with pytest.raises(RuntimeError, match="unapproved runtime config"):
        cli._run_live(preview=False)


def test_live_runtime_requires_explicit_approval_when_config_file_exists(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv(cli.ALLOW_UNAPPROVED_RUNTIME_ENV, raising=False)
    details = SimpleNamespace(
        runtime_config_path="config/live_config.json",
        runtime_config_approved=None,
        runtime_config_rejection_reasons=(),
    )

    with pytest.raises(RuntimeError, match="not explicitly approved"):
        cli._validate_runtime_config_approval(details, preview=False)


def test_run_live_allows_unapproved_runtime_with_explicit_override(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    lock_path = cli.PROJECT_ROOT / f"test_live_lock_{uuid4().hex}.lock"
    log_root = cli.PROJECT_ROOT / f"test_logs_{uuid4().hex}"
    called: dict[str, bool] = {"main": False}
    config = SimpleNamespace(
        paper=True,
        strategy_mode="mean_reversion",
        bar_timeframe_minutes=15,
        symbols=["AMD"],
        sma_bars=20,
        max_usd_per_trade=200.0,
        max_symbol_exposure_usd=200.0,
        max_open_positions=3,
        max_daily_loss_usd=300.0,
        max_orders_per_minute=6,
        max_price_deviation_bps=75.0,
        max_live_price_age_seconds=60,
        max_data_delay_seconds=300,
    )

    def fake_main(
        *,
        config=None,
        session_id=None,
        execute_orders=None,
        require_reconciliation_ready=None,
        durable_decision_cycles=None,
    ) -> None:
        called["main"] = True
        assert config is not None
        assert session_id is not None

    fake_module = SimpleNamespace(
        load_config_details=lambda: SimpleNamespace(
            config=config,
            runtime_config_path="config/live_config.json",
            overridden_fields=("symbols",),
            runtime_config_approved=False,
            runtime_config_rejection_reasons=("profit_factor 1.0169 >= 1.2",),
            baseline_valid_for_comparison=True,
            baseline_validation_errors=(),
        ),
        main=fake_main,
    )
    monkeypatch.setitem(sys.modules, "trading_bot", fake_module)
    monkeypatch.setattr(cli, "LIVE_BOT_LOCK_PATH", lock_path)
    monkeypatch.setattr(cli, "LOG_ROOT", log_root)
    monkeypatch.setenv(cli.ALLOW_UNAPPROVED_RUNTIME_ENV, "true")
    monkeypatch.setattr(cli, "_attach_baseline_validation", lambda details: details)
    monkeypatch.setattr(
        cli,
        "_read_process_identity",
        lambda pid: {
            "pid": pid,
            "started_at_utc": "2026-04-16T14:00:00+00:00",
            "command_line": "python -m tradeos live",
        },
    )

    try:
        exit_code = cli._run_live(preview=False)

        assert exit_code == 0
        assert called["main"] is True
        assert not lock_path.exists()
    finally:
        lock_path.unlink(missing_ok=True)
        _cleanup_tree(log_root)


def test_run_live_refuses_baseline_mismatch(monkeypatch: pytest.MonkeyPatch) -> None:
    config = SimpleNamespace(
        paper=True,
        strategy_mode="mean_reversion",
        bar_timeframe_minutes=15,
        symbols=["AMD"],
    )
    fake_module = SimpleNamespace(
        load_config_details=lambda: SimpleNamespace(
            config=config,
            runtime_config_path="config/live_config.json",
            overridden_fields=("symbols",),
            runtime_config_approved=True,
            runtime_config_rejection_reasons=(),
            baseline_valid_for_comparison=False,
            baseline_validation_errors=("research config does not match live runtime",),
        ),
        main=lambda **kwargs: None,
    )
    monkeypatch.setitem(sys.modules, "trading_bot", fake_module)
    monkeypatch.delenv(cli.ALLOW_BASELINE_MISMATCH_ENV, raising=False)
    monkeypatch.setattr(cli, "_attach_baseline_validation", lambda details: details)
    monkeypatch.setattr(cli, "_live_instance_lock", cli.contextlib.nullcontext)

    with pytest.raises(RuntimeError, match="does not match the promoted baseline"):
        cli._run_live(preview=False)


def test_run_live_allows_baseline_mismatch_with_explicit_override(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    called: dict[str, bool] = {"main": False}
    config = SimpleNamespace(
        paper=True,
        strategy_mode="mean_reversion",
        bar_timeframe_minutes=15,
        symbols=["AMD"],
        sma_bars=20,
        max_usd_per_trade=200.0,
        max_symbol_exposure_usd=200.0,
        max_open_positions=3,
        max_daily_loss_usd=300.0,
        max_orders_per_minute=6,
        max_price_deviation_bps=75.0,
        max_live_price_age_seconds=60,
        max_data_delay_seconds=300,
    )

    def fake_main(
        *,
        config=None,
        session_id=None,
        execute_orders=None,
        require_reconciliation_ready=None,
        durable_decision_cycles=None,
    ) -> None:
        called["main"] = True
        assert config is not None
        assert session_id is not None

    fake_module = SimpleNamespace(
        load_config_details=lambda: SimpleNamespace(
            config=config,
            runtime_config_path="config/live_config.json",
            overridden_fields=("symbols",),
            runtime_config_approved=True,
            runtime_config_rejection_reasons=(),
            baseline_valid_for_comparison=False,
            baseline_validation_errors=("research config does not match live runtime",),
        ),
        main=fake_main,
    )
    monkeypatch.setitem(sys.modules, "trading_bot", fake_module)
    monkeypatch.setenv(cli.ALLOW_BASELINE_MISMATCH_ENV, "true")
    monkeypatch.setattr(cli, "_attach_baseline_validation", lambda details: details)
    monkeypatch.setattr(cli, "_live_instance_lock", cli.contextlib.nullcontext)
    monkeypatch.setattr(cli, "LOG_ROOT", tmp_path / "logs")

    exit_code = cli._run_live(preview=False)

    assert exit_code == 0
    assert called["main"] is True


def test_attach_baseline_validation_supports_frozen_runtime_details(monkeypatch: pytest.MonkeyPatch) -> None:
    details = RuntimeConfigDetails(
        config=BotConfig(
            symbols=["AMD"],
            max_usd_per_trade=200.0,
            max_symbol_exposure_usd=200.0,
            max_open_positions=3,
            max_daily_loss_usd=300.0,
            sma_bars=20,
            bar_timeframe_minutes=15,
        ),
        runtime_config_path="config/live_config.json",
        overridden_fields=("symbols",),
        runtime_config_approved=True,
        runtime_config_rejection_reasons=(),
    )
    monkeypatch.setattr(
        cli,
        "load_backtest_baseline",
        lambda project_root: (
            None,
            {
                "valid_for_comparison": False,
                "validation_errors": ["research config does not match live runtime"],
            },
        ),
    )

    updated = cli._attach_baseline_validation(details)

    assert updated is not details
    assert updated.baseline_valid_for_comparison is False
    assert updated.baseline_validation_errors == ("research config does not match live runtime",)


def test_run_live_recovers_reused_pid_lock(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    lock_path = cli.PROJECT_ROOT / f"test_live_lock_{uuid4().hex}.lock"
    log_root = cli.PROJECT_ROOT / f"test_logs_{uuid4().hex}"
    lock_path.write_text(
        '{"pid": 4242, "command": "tradeos live", "process_started_at_utc": "2026-04-16T09:30:00+00:00"}',
        encoding="utf-8",
    )
    config = SimpleNamespace(
        paper=True,
        strategy_mode="mean_reversion",
        bar_timeframe_minutes=15,
        symbols=["AMD"],
        sma_bars=20,
        max_usd_per_trade=200.0,
        max_symbol_exposure_usd=200.0,
        max_open_positions=3,
        max_daily_loss_usd=300.0,
        max_orders_per_minute=6,
        max_price_deviation_bps=75.0,
        max_live_price_age_seconds=60,
        max_data_delay_seconds=300,
    )

    fake_module = SimpleNamespace(
        load_config_details=lambda: SimpleNamespace(
            config=config,
            runtime_config_path="config/live_config.json",
            overridden_fields=("symbols",),
            runtime_config_approved=True,
            runtime_config_rejection_reasons=(),
        ),
        main=lambda **kwargs: None,
    )
    monkeypatch.setitem(sys.modules, "trading_bot", fake_module)
    monkeypatch.setattr(cli, "LIVE_BOT_LOCK_PATH", lock_path)
    monkeypatch.setattr(cli, "LOG_ROOT", log_root)
    monkeypatch.setattr(cli, "_pid_is_running", lambda pid: True)
    monkeypatch.setattr(
        cli,
        "_read_process_identity",
        lambda pid: {
            "pid": pid,
            "started_at_utc": "2026-04-16T10:45:00+00:00" if pid == 4242 else "2026-04-16T14:00:00+00:00",
            "command_line": "python -m tradeos live" if pid == cli.os.getpid() else "python unrelated.py",
        },
    )

    try:
        exit_code = cli._run_live(preview=False)
        output = capsys.readouterr().out

        assert exit_code == 0
        assert "Recovered stale live bot lock" in output
        assert not lock_path.exists()
    finally:
        lock_path.unlink(missing_ok=True)
        _cleanup_tree(log_root)


def test_persist_startup_config_writes_live_artifact(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    log_root = tmp_path / "logs"
    details = SimpleNamespace(
        config=SimpleNamespace(
            paper=True,
            strategy_mode="mean_reversion",
            bar_timeframe_minutes=15,
            sma_bars=20,
            symbols=["AMD", "MSFT"],
            historical_feed="iex",
            live_feed="iex",
            latest_bar_feed="iex",
            bar_build_mode="stream_minute_aggregate",
            apply_updated_bars=True,
            post_bar_reconcile_poll=True,
            block_trading_until_resync=True,
            assert_feed_on_startup=True,
            log_bar_components=True,
            max_usd_per_trade=200.0,
            max_symbol_exposure_usd=200.0,
            max_open_positions=3,
            max_daily_loss_usd=300.0,
                max_orders_per_minute=6,
                max_price_deviation_bps=75.0,
                max_live_price_age_seconds=60,
                max_data_delay_seconds=300,
                ml_lookback_bars=320,
                breakout_max_stop_pct=0.03,
                sma_stop_pct=0.0,
                mean_reversion_exit_style="sma",
                mean_reversion_max_atr_percentile=0.0,
                mean_reversion_trend_filter=False,
                mean_reversion_trend_slope_filter=False,
                mean_reversion_stop_pct=0.0,
            ),
        runtime_config_path="config/live_config.json",
        overridden_fields=("symbols", "strategy_mode"),
        runtime_config_approved=True,
        runtime_config_rejection_reasons=(),
    )
    monkeypatch.setattr(cli, "LOG_ROOT", log_root)
    monkeypatch.setenv("BOT_DB_PATH", "bot_history.db")

    try:
        path = cli._persist_startup_config(details, preview=False, session_id="live-test-session")
        payload = cli.json.loads(path.read_text(encoding="utf-8"))
        latest_payload = cli.json.loads((path.parent / "startup_config.json").read_text(encoding="utf-8"))

        assert path.name.startswith("startup_config.")
        assert path.name.endswith(".json")
        assert payload["launch_mode"] == "live"
        assert payload["session_id"] == "live-test-session"
        assert payload["execution_enabled"] is True
        assert payload["paper"] is True
        assert payload["symbols"] == ["AMD", "MSFT"]
        assert payload["historical_feed"] == "iex"
        assert payload["live_feed"] == "iex"
        assert payload["latest_bar_feed"] == "iex"
        assert payload["bar_build_mode"] == "stream_minute_aggregate"
        assert payload["apply_updated_bars"] is True
        assert payload["post_bar_reconcile_poll"] is True
        assert payload["block_trading_until_resync"] is True
        assert payload["runtime_overrides"] == ["symbols", "strategy_mode"]
        assert payload["runtime_config_approved"] is True
        assert payload["runtime_config_rejection_reasons"] == []
        assert payload["ml_lookback_bars"] == 320
        assert payload["breakout_max_stop_pct"] == 0.03
        assert payload["sma_stop_pct"] == 0.0
        assert payload["mean_reversion_exit_style"] == "sma"
        assert payload["mean_reversion_trend_filter"] is False
        assert payload["mean_reversion_trend_slope_filter"] is False
        assert payload["mean_reversion_stop_pct"] == 0.0
        assert payload["db_path"] == str((cli.PROJECT_ROOT / "bot_history.db").resolve())
        assert latest_payload == payload
    finally:
        if log_root.exists():
            for child in sorted(log_root.rglob("*"), reverse=True):
                if child.is_file():
                    child.unlink()
                else:
                    child.rmdir()


def test_pid_is_running_uses_process_identity_on_windows(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cli.os, "name", "nt")
    monkeypatch.setattr(cli, "_read_process_identity", lambda pid: {"pid": pid, "command_line": "python -m tradeos live"})

    assert cli._pid_is_running(15592) is True
    assert cli._pid_is_running(0) is False
def test_parser_exposes_read_only_reconcile_command():
    from alpaca_trading_bot import cli

    args = cli.build_parser().parse_args(["reconcile"])

    assert args.command == "reconcile"


def test_live_command_rejects_non_paper_config_before_main_or_bot_construction(
    monkeypatch, tmp_path
):
    from types import SimpleNamespace

    import pytest

    import trading_bot
    from alpaca_trading_bot import cli

    class NoopContext:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, traceback):
            return False

    def unexpected_call(*args, **kwargs):
        pytest.fail("non-paper configuration reached an order-capable entrypoint")

    config = SimpleNamespace(
        paper=False,
        base_url="https://api.alpaca.markets",
    )
    monkeypatch.setattr(cli, "load_dotenv", lambda *args, **kwargs: None)
    monkeypatch.setattr(cli, "resolve_database_path", lambda *_args: tmp_path / "bot.db")
    monkeypatch.setattr(cli, "ExecutionLock", NoopContext)
    monkeypatch.setattr(cli, "_live_instance_lock", NoopContext)
    monkeypatch.setattr(
        trading_bot,
        "load_config_details",
        lambda: SimpleNamespace(config=config),
    )
    monkeypatch.setattr(trading_bot, "main", unexpected_call)
    monkeypatch.setattr(trading_bot, "TradeOSBot", unexpected_call)

    with pytest.raises(RuntimeError, match="ALPACA_PAPER=false"):
        cli.main(["live"])


def test_reconcile_live_config_reports_block_without_constructing_bot(
    monkeypatch, tmp_path, capsys
):
    from types import SimpleNamespace

    import pytest

    import automation_safety
    import storage
    import trading_bot
    from alpaca_trading_bot import cli

    report = SimpleNamespace(
        ready=False,
        final_code="BLOCKED_LIVE_CONFIGURATION",
    )

    class FakeStorage:
        def __init__(self, *args, **kwargs):
            pass

    class FakeSafetyReconciler:
        def __init__(self, *, broker, storage, configured_paper):
            assert broker is None
            assert isinstance(storage, FakeStorage)
            assert configured_paper is False

        def reconcile(self):
            return report

    def unexpected_bot(*args, **kwargs):
        pytest.fail("live reconciliation constructed TradeOSBot")

    config = SimpleNamespace(
        paper=False,
        base_url="https://api.alpaca.markets",
    )
    monkeypatch.setattr(cli, "load_dotenv", lambda *args, **kwargs: None)
    monkeypatch.setattr(cli, "resolve_database_path", lambda: tmp_path / "bot.db")
    monkeypatch.setattr(
        trading_bot,
        "load_config_details",
        lambda: SimpleNamespace(config=config),
    )
    monkeypatch.setattr(trading_bot, "TradeOSBot", unexpected_bot)
    monkeypatch.setattr(storage, "BotStorage", FakeStorage)
    monkeypatch.setattr(automation_safety, "SafetyReconciler", FakeSafetyReconciler)
    monkeypatch.setattr(
        automation_safety,
        "format_reconciliation",
        lambda value: value.final_code,
    )

    assert cli.main(["reconcile"]) == 2
    assert capsys.readouterr().out.strip() == "BLOCKED_LIVE_CONFIGURATION"


def test_paper_reconcile_prints_safe_final_code_from_bot_report(
    monkeypatch, tmp_path, capsys
):
    from types import SimpleNamespace

    import automation_safety
    import trading_bot
    from alpaca_trading_bot import cli

    report = SimpleNamespace(
        ready=True,
        final_code="SAFE_FOR_PAPER_AUTOMATION",
    )
    constructed = []

    class FakeTradeOSBot:
        def __init__(self, *args, **kwargs):
            constructed.append((args, kwargs))

        def reconcile_broker_safety(self):
            return report

    config = SimpleNamespace(
        paper=True,
        base_url="https://paper-api.alpaca.markets",
    )
    monkeypatch.setattr(cli, "load_dotenv", lambda *args, **kwargs: None)
    monkeypatch.setattr(cli, "resolve_database_path", lambda: tmp_path / "bot.db")
    monkeypatch.setattr(
        trading_bot,
        "load_config_details",
        lambda: SimpleNamespace(config=config),
    )
    monkeypatch.setattr(trading_bot, "TradeOSBot", FakeTradeOSBot)
    monkeypatch.setattr(
        automation_safety,
        "format_reconciliation",
        lambda value: value.final_code,
    )

    assert cli.main(["reconcile"]) == 0
    assert constructed
    assert capsys.readouterr().out.strip() == "SAFE_FOR_PAPER_AUTOMATION"


def test_live_run_holds_execution_lock_while_trading_main_executes(
    monkeypatch, tmp_path
):
    from types import SimpleNamespace

    import trading_bot
    from alpaca_trading_bot import cli

    state = {"execution_lock_held": False, "live_lock_held": False}
    events = []

    class FakeExecutionLock:
        def __enter__(self):
            assert state["execution_lock_held"] is False
            state["execution_lock_held"] = True
            events.append("execution-enter")
            return self

        def __exit__(self, exc_type, exc, traceback):
            assert state["execution_lock_held"] is True
            state["execution_lock_held"] = False
            events.append("execution-exit")
            return False

    class FakeLiveLock:
        def __enter__(self):
            assert state["execution_lock_held"] is True
            state["live_lock_held"] = True
            events.append("live-enter")
            return self

        def __exit__(self, exc_type, exc, traceback):
            assert state["live_lock_held"] is True
            state["live_lock_held"] = False
            events.append("live-exit")
            return False

    def fake_main(*args, **kwargs):
        assert state == {"execution_lock_held": True, "live_lock_held": True}
        events.append("main")

    config = SimpleNamespace(
        paper=True,
        base_url="https://paper-api.alpaca.markets",
    )
    monkeypatch.setattr(cli, "load_dotenv", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        cli, "resolve_database_path", lambda *_args: tmp_path / "bot.db"
    )
    monkeypatch.setattr(cli, "ExecutionLock", FakeExecutionLock)
    monkeypatch.setattr(cli, "_live_instance_lock", FakeLiveLock)
    monkeypatch.setattr(cli, "LOG_ROOT", tmp_path / "logs")
    monkeypatch.setattr(
        trading_bot,
        "load_config_details",
        lambda: SimpleNamespace(config=config),
    )
    monkeypatch.setattr(trading_bot, "main", fake_main)

    assert cli.main(["live"]) == 0
    assert events == [
        "execution-enter",
        "live-enter",
        "main",
        "live-exit",
        "execution-exit",
    ]
    assert state == {"execution_lock_held": False, "live_lock_held": False}

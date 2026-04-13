import sys
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

import pytest

from alpaca_trading_bot import cli


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


def test_run_live_refuses_non_paper_execution(monkeypatch: pytest.MonkeyPatch) -> None:
    config = SimpleNamespace(
        paper=False,
        strategy_mode="mean_reversion",
        bar_timeframe_minutes=15,
        symbols=["AMD", "MSFT"],
    )
    fake_module = SimpleNamespace(
        load_config_details=lambda: SimpleNamespace(config=config, runtime_config_path=None, overridden_fields=()),
        main=lambda **kwargs: None,
    )
    monkeypatch.setitem(sys.modules, "trading_bot", fake_module)

    with pytest.raises(RuntimeError, match="ALPACA_PAPER=false"):
        cli._run_live(preview=False)


def test_run_preview_allows_non_paper_but_disables_execution(monkeypatch: pytest.MonkeyPatch) -> None:
    called: dict[str, bool] = {"main": False}
    config = SimpleNamespace(
        paper=False,
        strategy_mode="mean_reversion",
        bar_timeframe_minutes=15,
        symbols=["AMD"],
    )

    def fake_main(*, config=None, session_id=None) -> None:
        called["main"] = True
        assert cli.os.environ["EXECUTE_ORDERS"] == "false"
        assert config is not None
        assert session_id is not None

    fake_module = SimpleNamespace(
        load_config_details=lambda: SimpleNamespace(config=config, runtime_config_path=None, overridden_fields=()),
        main=fake_main,
    )
    monkeypatch.setitem(sys.modules, "trading_bot", fake_module)

    exit_code = cli._run_live(preview=True)

    assert exit_code == 0
    assert called["main"] is True


def test_run_live_refuses_second_live_instance(monkeypatch: pytest.MonkeyPatch) -> None:
    lock_path = cli.PROJECT_ROOT / f"test_live_lock_{uuid4().hex}.lock"
    log_root = cli.PROJECT_ROOT / f"test_logs_{uuid4().hex}"
    lock_path.write_text('{"pid": 4242, "command": "alpaca-bot live"}', encoding="utf-8")
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
        load_config_details=lambda: SimpleNamespace(config=config, runtime_config_path="config/live_config.json", overridden_fields=("symbols",)),
        main=lambda **kwargs: None,
    )
    monkeypatch.setitem(sys.modules, "trading_bot", fake_module)
    monkeypatch.setattr(cli, "LIVE_BOT_LOCK_PATH", lock_path)
    monkeypatch.setattr(cli, "LOG_ROOT", log_root)
    monkeypatch.setattr(cli, "_pid_is_running", lambda pid: pid == 4242)

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
    lock_path.write_text('{"pid": 999999, "command": "alpaca-bot live"}', encoding="utf-8")
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

    def fake_main(*, config=None, session_id=None) -> None:
        called["main"] = True
        assert lock_path.exists()
        assert config is not None
        assert session_id is not None

    fake_module = SimpleNamespace(
        load_config_details=lambda: SimpleNamespace(config=config, runtime_config_path="config/live_config.json", overridden_fields=("symbols",)),
        main=fake_main,
    )
    monkeypatch.setitem(sys.modules, "trading_bot", fake_module)
    monkeypatch.setattr(cli, "LIVE_BOT_LOCK_PATH", lock_path)
    monkeypatch.setattr(cli, "LOG_ROOT", log_root)
    monkeypatch.setattr(cli, "_pid_is_running", lambda pid: False)

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
        load_config_details=lambda: SimpleNamespace(config=config, runtime_config_path="config/live_config.json", overridden_fields=("symbols", "strategy_mode")),
            main=lambda **kwargs: None,
    )
    monkeypatch.setitem(sys.modules, "trading_bot", fake_module)
    monkeypatch.setattr(cli, "LIVE_BOT_LOCK_PATH", lock_path)
    monkeypatch.setattr(cli, "LOG_ROOT", log_root)

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
        load_config_details=lambda: SimpleNamespace(config=config, runtime_config_path="config/live_config.json", overridden_fields=("symbols",)),
            main=lambda **kwargs: None,
    )
    monkeypatch.setitem(sys.modules, "trading_bot", fake_module)
    monkeypatch.setattr(cli, "LIVE_BOT_LOCK_PATH", lock_path)
    monkeypatch.setattr(cli, "LOG_ROOT", log_root)

    try:
        exit_code = cli._run_live(preview=False)
        output = capsys.readouterr().out

        assert exit_code == 0
        assert "Recovered malformed live bot lock" in output
        assert not lock_path.exists()
    finally:
        lock_path.unlink(missing_ok=True)
        _cleanup_tree(log_root)


def test_persist_startup_config_writes_live_artifact(monkeypatch: pytest.MonkeyPatch) -> None:
    log_root = cli.PROJECT_ROOT / f"test_logs_{uuid4().hex}"
    details = SimpleNamespace(
        config=SimpleNamespace(
            paper=True,
            strategy_mode="mean_reversion",
            bar_timeframe_minutes=15,
            sma_bars=20,
            symbols=["AMD", "MSFT"],
            max_usd_per_trade=200.0,
            max_symbol_exposure_usd=200.0,
            max_open_positions=3,
            max_daily_loss_usd=300.0,
            max_orders_per_minute=6,
            max_price_deviation_bps=75.0,
            max_live_price_age_seconds=60,
            max_data_delay_seconds=300,
        ),
        runtime_config_path="config/live_config.json",
        overridden_fields=("symbols", "strategy_mode"),
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
        assert payload["runtime_overrides"] == ["symbols", "strategy_mode"]
        assert latest_payload == payload
    finally:
        if log_root.exists():
            for child in sorted(log_root.rglob("*"), reverse=True):
                if child.is_file():
                    child.unlink()
                else:
                    child.rmdir()

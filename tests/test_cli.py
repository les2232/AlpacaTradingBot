from alpaca_trading_bot import cli


def test_cli_parser_accepts_preview() -> None:
    args = cli.build_parser().parse_args(["preview"])
    assert args.command == "preview"


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

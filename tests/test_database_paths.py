from __future__ import annotations

import hashlib
import json
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest

import dashboard_state
import trading_bot
import tradeos.runtime_paths as runtime_paths
from alpaca_trading_bot import cli
from storage import BotStorage
from tradeos.runtime_paths import (
    DATABASE_PATH_ENV,
    DEFAULT_DATABASE_FILENAME,
    TRADEOS_REPOSITORY_ROOT,
    DatabasePathError,
    _resolve_database_path_from_root,
    resolve_database_path,
    resolve_explicit_database_path,
    resolve_log_root,
)


def test_dashboard_storage_is_sqlite_query_only_and_does_not_change_database(
    tmp_path: Path,
) -> None:
    database_path = (tmp_path / "dashboard-read-only.db").resolve()
    BotStorage(database_path)
    before_bytes = database_path.read_bytes()
    before_tables = sqlite3.connect(database_path).execute(
        "SELECT name, sql FROM sqlite_master ORDER BY name"
    ).fetchall()

    storage = BotStorage(database_path, read_only=True)
    assert storage.get_latest_run() is None
    with pytest.raises(sqlite3.OperationalError, match="write"):
        storage.bind_runtime_state_scope(
            environment="paper",
            account_fingerprint="f" * 64,
            provenance="must-not-write",
            history_cutoff_utc="2026-08-06T00:00:00+00:00",
            bound_at_utc="2026-08-06T00:00:00+00:00",
        )

    after_tables = sqlite3.connect(database_path).execute(
        "SELECT name, sql FROM sqlite_master ORDER BY name"
    ).fetchall()
    assert database_path.read_bytes() == before_bytes
    assert after_tables == before_tables


def test_dashboard_read_only_storage_projects_legacy_scope_columns_without_migration(
    tmp_path: Path,
) -> None:
    database_path = (tmp_path / "legacy-dashboard.db").resolve()
    with sqlite3.connect(database_path) as connection:
        connection.executescript(
            """
            CREATE TABLE bot_runs (
                id INTEGER PRIMARY KEY,
                session_id TEXT,
                symbol_fingerprint TEXT,
                timestamp_utc TEXT NOT NULL,
                cash REAL NOT NULL,
                buying_power REAL NOT NULL,
                equity REAL NOT NULL,
                last_equity REAL NOT NULL,
                daily_pnl REAL NOT NULL,
                kill_switch_triggered INTEGER NOT NULL
            );
            CREATE TABLE symbol_runs (
                id INTEGER PRIMARY KEY,
                run_id INTEGER NOT NULL,
                symbol TEXT NOT NULL,
                price REAL,
                sma REAL,
                ml_probability_up REAL,
                ml_confidence REAL,
                ml_training_rows INTEGER,
                action TEXT NOT NULL,
                holding INTEGER NOT NULL,
                holding_minutes REAL,
                quantity REAL NOT NULL,
                market_value REAL NOT NULL,
                avg_entry_price REAL,
                unrealized_pl REAL,
                error TEXT
            );
            CREATE TABLE order_history (
                order_id TEXT PRIMARY KEY,
                session_id TEXT,
                observed_at_utc TEXT NOT NULL,
                submitted_at TEXT,
                filled_at TEXT,
                symbol TEXT NOT NULL,
                side TEXT NOT NULL,
                status TEXT NOT NULL,
                qty REAL,
                filled_qty REAL,
                filled_avg_price REAL,
                notional REAL,
                client_order_id TEXT
            );
            CREATE TABLE order_intents (
                intent_id TEXT PRIMARY KEY,
                client_order_id TEXT NOT NULL,
                account_fingerprint TEXT NOT NULL,
                decision_ts TEXT NOT NULL,
                symbol TEXT NOT NULL,
                side TEXT NOT NULL,
                intent_kind TEXT NOT NULL,
                requested_qty TEXT NOT NULL,
                final_qty TEXT NOT NULL,
                status TEXT NOT NULL,
                broker_order_id TEXT,
                broker_status TEXT,
                filled_qty TEXT NOT NULL,
                submission_attempts INTEGER NOT NULL,
                error_category TEXT,
                created_at_utc TEXT NOT NULL,
                updated_at_utc TEXT NOT NULL
            );
            INSERT INTO bot_runs VALUES (
                1, 'legacy-session', 'legacy-symbols',
                '2026-04-24T19:45:00+00:00', 100, 200, 100, 100, 0, 0
            );
            INSERT INTO symbol_runs VALUES (
                1, 1, 'COP', 121.0, 122.0, NULL, NULL, NULL,
                'SNAPSHOT_ONLY', 1, NULL, -8, -968, NULL, NULL, NULL
            );
            INSERT INTO order_history VALUES (
                'order-1', 'legacy-session', '2026-04-24T19:40:00+00:00',
                NULL, NULL, 'COP', 'sell', 'FILLED', 8, 8, 121, NULL, NULL
            );
            """
        )
    before_bytes = database_path.read_bytes()
    before_schema = sqlite3.connect(database_path).execute(
        "SELECT name, sql FROM sqlite_master ORDER BY name"
    ).fetchall()

    storage = BotStorage(database_path, read_only=True)
    latest = storage.get_latest_run()
    orders = storage.get_order_history()

    assert latest is not None
    assert latest["session_id"] == "legacy-session"
    assert latest["environment"] is None
    assert latest["account_fingerprint"] is None
    assert storage.get_latest_symbol_snapshot()[0]["quantity"] == -8
    assert orders[0]["environment"] is None
    assert orders[0]["account_fingerprint"] is None
    assert storage.get_runtime_state_binding(
        environment="paper",
        account_fingerprint="f" * 64,
    ) is None

    after_schema = sqlite3.connect(database_path).execute(
        "SELECT name, sql FROM sqlite_master ORDER BY name"
    ).fetchall()
    assert database_path.read_bytes() == before_bytes
    assert after_schema == before_schema


def test_dashboard_launcher_defaults_to_dashboard_only_and_never_invokes_live() -> None:
    launcher = (TRADEOS_REPOSITORY_ROOT / "start_dashboard.ps1").read_text(
        encoding="utf-8"
    )

    assert 'if ($PaperShadow) {' in launcher
    assert '-ArgumentList @("-m", "tradeos", "paper-shadow")' in launcher
    assert '-ArgumentList @("-m", "tradeos", "live")' not in launcher


def test_default_database_path_is_launcher_invariant(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.delenv(DATABASE_PATH_ENV, raising=False)
    expected = (TRADEOS_REPOSITORY_ROOT / DEFAULT_DATABASE_FILENAME).resolve()

    monkeypatch.chdir(TRADEOS_REPOSITORY_ROOT)
    from_repository = resolve_database_path()
    monkeypatch.chdir(tmp_path)
    from_unrelated_directory = resolve_database_path()

    assert from_repository == expected
    assert from_unrelated_directory == expected
    assert from_repository.is_absolute()


def test_runtime_config_and_log_root_are_launcher_invariant(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.delenv("BOT_RUNTIME_CONFIG_PATH", raising=False)
    monkeypatch.delenv(runtime_paths.LOG_ROOT_ENV, raising=False)

    monkeypatch.chdir(tmp_path)
    runtime_path, _, _, _ = trading_bot._load_runtime_config_payload()

    assert runtime_path == (
        TRADEOS_REPOSITORY_ROOT / "config" / "live_config.json"
    ).resolve()
    assert resolve_log_root() == (TRADEOS_REPOSITORY_ROOT / "logs").resolve()


def test_relative_runtime_config_path_is_repository_anchored(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("BOT_RUNTIME_CONFIG_PATH", "config/live_config.json")

    runtime_path, runtime, payload, runtime_sha256 = (
        trading_bot._load_runtime_config_payload()
    )

    assert runtime_path == (
        TRADEOS_REPOSITORY_ROOT / "config" / "live_config.json"
    ).resolve()
    assert isinstance(runtime, dict)
    assert isinstance(payload, dict)
    assert runtime_path is not None
    assert runtime_sha256 == hashlib.sha256(runtime_path.read_bytes()).hexdigest()


def test_relative_database_path_is_repository_anchored_from_every_cwd(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    configured = "runtime-data/paper authority.db"
    monkeypatch.setenv(DATABASE_PATH_ENV, configured)

    monkeypatch.chdir(TRADEOS_REPOSITORY_ROOT)
    from_repository = resolve_database_path()
    monkeypatch.chdir(tmp_path)
    from_unrelated_directory = resolve_database_path()

    expected = (TRADEOS_REPOSITORY_ROOT / configured).resolve()
    assert from_repository == expected
    assert from_unrelated_directory == expected
    assert not expected.exists()


def test_explicit_absolute_database_path_is_safely_normalized(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    configured = tmp_path / "unused" / ".." / "paper.db"
    monkeypatch.setenv(DATABASE_PATH_ENV, str(tmp_path / "environment.db"))

    assert resolve_database_path(configured) == (tmp_path / "paper.db").resolve()
    assert not (tmp_path / "paper.db").exists()


def test_repository_path_with_spaces_resolves_without_side_effects(tmp_path: Path) -> None:
    repository_root = tmp_path / "TradeOS checkout with spaces"

    resolved = _resolve_database_path_from_root(
        "state/paper history.db",
        repository_root=repository_root,
    )

    assert resolved == (repository_root / "state" / "paper history.db").resolve()
    assert not repository_root.exists()


def test_unset_uses_default_but_blank_values_fail_closed() -> None:
    expected = (TRADEOS_REPOSITORY_ROOT / DEFAULT_DATABASE_FILENAME).resolve()
    assert resolve_database_path(environment={}) == expected

    for blank_value in ("", " ", "\t"):
        with pytest.raises(DatabasePathError, match="cannot be blank"):
            resolve_database_path(environment={DATABASE_PATH_ENV: blank_value})


def test_relative_database_path_cannot_escape_repository(tmp_path: Path) -> None:
    repository_root = tmp_path / "checkout"
    outside_path = tmp_path / "outside.db"

    with pytest.raises(DatabasePathError, match="must remain inside"):
        _resolve_database_path_from_root(
            "../outside.db",
            repository_root=repository_root,
        )

    assert not repository_root.exists()
    assert not outside_path.exists()


def test_storage_rejects_noncanonical_authority_before_connecting(tmp_path: Path) -> None:
    with pytest.raises(DatabasePathError, match="absolute, normalized"):
        BotStorage("relative-test-authority.db")

    noncanonical_absolute = tmp_path / "unused" / ".." / "paper.db"
    with pytest.raises(DatabasePathError, match="absolute, normalized"):
        BotStorage(noncanonical_absolute)

    assert not (TRADEOS_REPOSITORY_ROOT / "relative-test-authority.db").exists()
    assert not (tmp_path / "paper.db").exists()


def test_cli_and_direct_runtime_resolve_the_same_database(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    configured = str(tmp_path / "shared.db")
    expected = (tmp_path / "shared.db").resolve()
    observed: dict[str, Path] = {}

    class RuntimeStorageCaptured(RuntimeError):
        pass

    def capture_runtime_storage(path: Path) -> None:
        observed["runtime"] = path
        raise RuntimeStorageCaptured

    monkeypatch.setenv(DATABASE_PATH_ENV, configured)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(trading_bot, "load_dotenv", lambda *args, **kwargs: None)
    monkeypatch.setattr(trading_bot, "BotStorage", capture_runtime_storage)
    monkeypatch.setattr(trading_bot.TradeOSBot, "_build_broker", lambda self: object())
    config = trading_bot.BotConfig(
        symbols=["AMD"],
        max_usd_per_trade=200.0,
        max_symbol_exposure_usd=200.0,
        max_open_positions=3,
        max_daily_loss_usd=300.0,
        sma_bars=20,
        bar_timeframe_minutes=15,
    )

    with pytest.raises(RuntimeStorageCaptured):
        trading_bot.TradeOSBot(config)

    def capture_cli_main(**kwargs: object) -> None:
        observed["cli"] = Path(cli.os.environ[DATABASE_PATH_ENV])

    fake_module = SimpleNamespace(
        BotConfig=trading_bot.BotConfig,
        load_config_details=lambda: SimpleNamespace(
            config=config,
            runtime_config_path=None,
            overridden_fields=(),
            runtime_config_approved=True,
            runtime_config_rejection_reasons=(),
        ),
        build_runtime_canonical_strategy_evidence=(
            trading_bot.build_runtime_canonical_strategy_evidence
        ),
        main=capture_cli_main,
    )
    monkeypatch.setitem(sys.modules, "trading_bot", fake_module)
    monkeypatch.setattr(cli, "load_dotenv", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        cli,
        "_attach_baseline_validation",
        lambda details, **_kwargs: details,
    )

    assert cli._run_live(preview=True) == 0
    assert observed == {"runtime": expected, "cli": expected}
    assert not expected.exists()


def test_dashboard_rejects_conflicting_authority_before_storage(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    artifact_path = (tmp_path / "artifact.db").resolve()
    environment_path = (tmp_path / "environment.db").resolve()
    startup_config = SimpleNamespace(db_path=str(artifact_path))
    storage_called = False

    def unexpected_storage(path: Path) -> None:
        nonlocal storage_called
        storage_called = True

    monkeypatch.setenv(DATABASE_PATH_ENV, str(environment_path))
    monkeypatch.setattr(dashboard_state, "_load_startup_configs", lambda: [startup_config])
    monkeypatch.setattr(dashboard_state, "BotStorage", unexpected_storage)

    with pytest.raises(RuntimeError, match="Conflicting database authorities"):
        dashboard_state.load_dashboard_state()

    assert storage_called is False
    assert not artifact_path.exists()
    assert not environment_path.exists()


def test_dashboard_accepts_matching_canonical_authorities(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    database_path = (tmp_path / "shared.db").resolve()
    startup_config = SimpleNamespace(db_path=str(database_path))
    monkeypatch.setenv(DATABASE_PATH_ENV, str(database_path))

    assert dashboard_state._resolve_dashboard_database_path(startup_config) == database_path
    assert not database_path.exists()


def test_dashboard_uses_only_repository_database_key_without_startup_artifact(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    requested_paths: list[tuple[Path, bool]] = []

    def fake_dotenv_values(path: Path, *, verbose: bool) -> dict[str, str]:
        requested_paths.append((Path(path), verbose))
        return {DATABASE_PATH_ENV: "runtime-data/preview.db"}

    monkeypatch.delenv(DATABASE_PATH_ENV, raising=False)
    monkeypatch.setattr(runtime_paths.dotenv, "dotenv_values", fake_dotenv_values)

    resolved = dashboard_state._resolve_dashboard_database_path(None)

    assert resolved == (
        TRADEOS_REPOSITORY_ROOT / "runtime-data" / "preview.db"
    ).resolve()
    assert requested_paths == [(TRADEOS_REPOSITORY_ROOT / ".env", False)]
    assert not resolved.exists()


def test_explicit_process_database_path_precedes_repository_key(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    database_path = (tmp_path / "process.db").resolve()
    monkeypatch.setenv(DATABASE_PATH_ENV, str(database_path))
    monkeypatch.setattr(
        runtime_paths.dotenv,
        "dotenv_values",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("repository dotenv must not be read")
        ),
    )

    assert resolve_explicit_database_path() == database_path
    assert not database_path.exists()


def test_repository_database_key_absence_and_blank_values_are_distinct(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv(DATABASE_PATH_ENV, raising=False)
    monkeypatch.setattr(runtime_paths.dotenv, "dotenv_values", lambda *args, **kwargs: {})
    assert resolve_explicit_database_path() is None

    for blank_value in (None, "", " "):
        monkeypatch.setattr(
            runtime_paths.dotenv,
            "dotenv_values",
            lambda *args, value=blank_value, **kwargs: {DATABASE_PATH_ENV: value},
        )
        with pytest.raises(DatabasePathError, match="cannot be blank"):
            resolve_explicit_database_path()


def test_dashboard_startup_artifact_discovery_is_cwd_invariant(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    repository_root = tmp_path / "TradeOS checkout with spaces"
    day_dir = repository_root / "logs" / "2026-04-23"
    day_dir.mkdir(parents=True)
    database_path = (tmp_path / "authority.db").resolve()
    BotStorage(database_path)
    artifact_path = day_dir / "startup_config.20260423T154500Z.json"
    artifact_path.write_text(
        json.dumps(
            {
                "session_id": "session-a",
                "started_at_utc": "2026-04-23T15:45:00+00:00",
                "launch_mode": "live",
                "execution_enabled": True,
                "paper": True,
                "account_mode": "paper",
                "strategy_mode": "mean_reversion",
                "bar_timeframe_minutes": 15,
                "symbols": ["AMD"],
                "symbol_count": 1,
                "db_path": str(database_path),
            }
        ),
        encoding="utf-8",
    )
    unrelated_cwd = tmp_path / "unrelated"
    unrelated_cwd.mkdir()
    monkeypatch.setattr(runtime_paths, "TRADEOS_REPOSITORY_ROOT", repository_root)
    monkeypatch.delenv("BOT_LOG_ROOT", raising=False)
    monkeypatch.setenv("BOT_DB_PATH", str(database_path))

    monkeypatch.chdir(repository_root)
    from_repository = dashboard_state._load_startup_configs()
    monkeypatch.chdir(unrelated_cwd)
    from_unrelated_directory = dashboard_state._load_startup_configs()

    assert [config.artifact_path for config in from_repository] == [str(artifact_path)]
    assert [config.artifact_path for config in from_unrelated_directory] == [
        str(artifact_path)
    ]
    assert from_repository[0].db_path == str(database_path)


def test_powershell_launcher_delegates_database_authority_to_tradeos() -> None:
    launcher_text = (TRADEOS_REPOSITORY_ROOT / "start_dashboard.ps1").read_text(
        encoding="utf-8"
    )

    assert '@("-m", "tradeos", "paper-shadow")' in launcher_text
    assert '@("-m", "tradeos", "live")' not in launcher_text
    assert '@("-m", "tradeos", "dashboard", "--port", "$Port")' in launcher_text
    assert launcher_text.count("-WorkingDirectory $projectRoot") >= 3
    assert DATABASE_PATH_ENV not in launcher_text
    assert "Reusing existing dashboard" not in launcher_text
    assert "database authority cannot be verified" in launcher_text
    assert "-PassThru" in launcher_text
    assert "$dashboardProcess.HasExited" in launcher_text
    startup_sequence = launcher_text.split(
        "$python = Resolve-PythonExecutable -RequestedPython $PythonExe",
        maxsplit=1,
    )[1]
    assert 'if ($PaperShadow) {' in startup_sequence
    assert startup_sequence.index("Assert-DashboardPortAvailable") < startup_sequence.index(
        "Start-LiveBot"
    )


def test_storage_instances_share_one_temporary_authority_across_cwds(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    database_path = (tmp_path / "shared-storage.db").resolve()

    monkeypatch.chdir(TRADEOS_REPOSITORY_ROOT)
    first = BotStorage(database_path)
    monkeypatch.chdir(tmp_path)
    second = BotStorage(database_path)

    assert first.db_path == database_path
    assert second.db_path == database_path


def test_decision_cycle_claim_is_shared_across_working_directories(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    database_path = (tmp_path / "decision-authority.db").resolve()
    decision_time = datetime(2026, 4, 23, 15, 45, tzinfo=timezone.utc)

    monkeypatch.chdir(TRADEOS_REPOSITORY_ROOT)
    first_storage = BotStorage(database_path)
    first_claim = first_storage.claim_decision_cycle(
        decision_time,
        owner_id="repository-launch",
        claimed_at_utc="2026-04-23T15:45:05+00:00",
    )

    monkeypatch.chdir(tmp_path)
    second_storage = BotStorage(database_path)
    duplicate_claim = second_storage.claim_decision_cycle(
        decision_time,
        owner_id="alternate-launch",
        claimed_at_utc="2026-04-23T15:45:06+00:00",
    )

    assert first_claim.claimed is True
    assert duplicate_claim.claimed is False
    assert duplicate_claim.reason == "duplicate_cycle"

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import dashboard
from dashboard_state import (
    BrokerSafetyPanelState,
    _build_broker_safety_panel_state,
    load_broker_safety_panel_state,
)


NOW = datetime(2026, 8, 5, 18, 0, tzinfo=timezone.utc)
ACCOUNT_FINGERPRINT = "a" * 64


def _lock(state: str = "unlocked", *, metadata_valid: bool = True):
    return SimpleNamespace(
        state=SimpleNamespace(value=state),
        held_by_current_process=False,
        metadata_valid=metadata_valid,
    )


def _safe_payload(**updates):
    position = {
        "symbol": "AAPL",
        "qty": "1",
        "managed": True,
        "reason": "managed_client_order_history",
    }
    order = {
        "order_id": "order-1",
        "client_order_id": "tradeos-order-1",
        "symbol": "AAPL",
        "side": "buy",
        "status": "accepted",
        "qty": "1",
        "filled_qty": "0",
        "remaining_qty": "1",
        "managed": True,
        "reason": "managed_client_order_id",
    }
    payload = {
        "environment": "paper",
        "account_id_masked": "****1234",
        "account_fingerprint": ACCOUNT_FINGERPRINT,
        "account_status": "active",
        "checked_at_utc": NOW.isoformat(),
        "reconciliation_succeeded": True,
        "current_broker_positions": [position],
        "managed_positions": [position],
        "unmanaged_positions": [],
        "current_open_pending_orders": [order],
        "managed_orders": [order],
        "unmanaged_orders": [],
        "local_positions": {"AAPL": "1"},
        "pending_buy_qty": {"AAPL": "1"},
        "pending_sell_qty": {},
        "ambiguous_intents": [],
        "state_mismatches": [],
        "blocking_reasons": [],
        "automation_readiness": "ready",
        "result": "SAFE_FOR_PAPER_AUTOMATION",
    }
    payload.update(updates)
    return payload


def _paper_shadow_config_payload(
    expected_account_fingerprint: object = ACCOUNT_FINGERPRINT,
) -> dict[str, object]:
    return {
        "launch": {
            "profile": "paper_shadow",
            "broker_environment": "paper",
            "submission_policy": "deny",
            "expected_account_fingerprint": expected_account_fingerprint,
            "dashboard_mode": "read_only",
        }
    }


def test_broker_safety_projection_accepts_only_fresh_positive_paper_state() -> None:
    panel = _build_broker_safety_panel_state(
        _safe_payload(),
        _lock(),
        now_utc=NOW,
        expected_account_fingerprint=ACCOUNT_FINGERPRINT,
    )

    assert panel.environment == "paper"
    assert panel.account_id_masked == "****1234"
    assert panel.lock_status == "unlocked"
    assert panel.managed_position_count == 1
    assert panel.unmanaged_position_count == 0
    assert panel.managed_order_count == 1
    assert panel.unmanaged_order_count == 0
    assert panel.mismatch_status == "clear"
    assert panel.automation_readiness == "ready"
    assert panel.blocking_reasons == ()
    assert panel.stale is False


def test_broker_safety_projection_rejects_missing_account_fingerprint() -> None:
    payload = _safe_payload()
    payload.pop("account_fingerprint")

    panel = _build_broker_safety_panel_state(
        payload,
        _lock(),
        now_utc=NOW,
        expected_account_fingerprint=ACCOUNT_FINGERPRINT,
    )

    assert panel.automation_readiness == "blocked"
    assert "reconciliation_artifact_malformed" in panel.blocking_reasons


def test_broker_safety_projection_rejects_invalid_account_fingerprint_string() -> None:
    panel = _build_broker_safety_panel_state(
        _safe_payload(account_fingerprint="not-a-sha256-fingerprint"),
        _lock(),
        now_utc=NOW,
        expected_account_fingerprint=ACCOUNT_FINGERPRINT,
    )

    assert panel.automation_readiness == "blocked"
    assert "reconciliation_account_fingerprint_invalid" in panel.blocking_reasons


def test_broker_safety_projection_rejects_account_fingerprint_mismatch() -> None:
    panel = _build_broker_safety_panel_state(
        _safe_payload(account_fingerprint="b" * 64),
        _lock(),
        now_utc=NOW,
        expected_account_fingerprint=ACCOUNT_FINGERPRINT,
    )

    assert panel.automation_readiness == "blocked"
    assert "reconciliation_account_fingerprint_mismatch" in panel.blocking_reasons


def test_broker_safety_projection_fails_closed_for_missing_or_malformed_artifact() -> None:
    missing = _build_broker_safety_panel_state(
        None,
        _lock(),
        now_utc=NOW,
        expected_account_fingerprint=ACCOUNT_FINGERPRINT,
    )
    malformed = _build_broker_safety_panel_state(
        {"environment": "paper", "blocking_reasons": []},
        _lock(),
        now_utc=NOW,
        expected_account_fingerprint=ACCOUNT_FINGERPRINT,
    )

    assert missing.automation_readiness == "blocked"
    assert missing.environment == "unknown"
    assert missing.managed_position_count is None
    assert "reconciliation_artifact_missing_or_malformed" in missing.blocking_reasons
    assert malformed.automation_readiness == "blocked"
    assert malformed.environment == "unknown"
    assert "reconciliation_artifact_malformed" in malformed.blocking_reasons


def test_broker_safety_projection_blocks_stale_artifact() -> None:
    stale_payload = _safe_payload(checked_at_utc=(NOW - timedelta(seconds=301)).isoformat())

    panel = _build_broker_safety_panel_state(
        stale_payload,
        _lock(),
        now_utc=NOW,
        expected_account_fingerprint=ACCOUNT_FINGERPRINT,
    )

    assert panel.automation_readiness == "blocked"
    assert panel.stale is True
    assert "reconciliation_stale_over_5_minutes" in panel.blocking_reasons


def test_broker_safety_projection_preserves_and_derives_exact_blocking_reasons() -> None:
    unmanaged_position = {
        "symbol": "COP",
        "qty": "1",
        "managed": False,
        "reason": "unknown_position",
    }
    unmanaged_order = {
        "order_id": "order-2",
        "client_order_id": "external-order",
        "symbol": "COP",
        "side": "sell",
        "status": "accepted",
        "qty": "1",
        "filled_qty": "0",
        "remaining_qty": "1",
        "managed": False,
        "reason": "unknown_client_order_id",
    }
    payload = _safe_payload(
        current_broker_positions=[
            _safe_payload()["managed_positions"][0],
            unmanaged_position,
        ],
        unmanaged_positions=[unmanaged_position],
        current_open_pending_orders=[
            _safe_payload()["managed_orders"][0],
            unmanaged_order,
        ],
        unmanaged_orders=[unmanaged_order],
        local_positions={"AAPL": "1", "COP": "0"},
        pending_sell_qty={"COP": "1"},
        state_mismatches=["COP broker=-10 local=10"],
        blocking_reasons=["unmanaged broker exposure"],
        automation_readiness="blocked",
        result="BLOCKED_UNMANAGED_POSITION",
    )

    panel = _build_broker_safety_panel_state(
        payload,
        _lock("unknown"),
        now_utc=NOW,
        expected_account_fingerprint=ACCOUNT_FINGERPRINT,
    )

    assert panel.automation_readiness == "blocked"
    assert panel.unmanaged_position_count == 1
    assert panel.unmanaged_order_count == 1
    assert panel.mismatch_status == "mismatch (1)"
    assert panel.blocking_reasons[0] == "unmanaged broker exposure"
    assert "trading_lock_status_unknown" in panel.blocking_reasons
    assert "unmanaged_positions_present" in panel.blocking_reasons
    assert "unmanaged_orders_present" in panel.blocking_reasons
    assert "broker_local_state_mismatch" in panel.blocking_reasons


def test_redacted_loader_reads_only_requested_artifact_and_lock_probe(monkeypatch, tmp_path) -> None:
    artifact = tmp_path / "reconciliation.json"
    artifact.write_text(json.dumps(_safe_payload()), encoding="utf-8")
    config_path = tmp_path / "paper_shadow_config.json"
    config_path.write_text(
        json.dumps(_paper_shadow_config_payload()),
        encoding="utf-8",
    )

    import tradeos.execution_lock as execution_lock
    import tradeos.readiness as readiness

    monkeypatch.setattr(execution_lock, "inspect_lock", lambda: _lock("held"))
    monkeypatch.setattr(
        readiness,
        "select_latest_valid_evidence",
        lambda *args, **kwargs: SimpleNamespace(evidence=object(), rejected=()),
    )

    panel = load_broker_safety_panel_state(
        artifact,
        now_utc=NOW,
        readiness_config_path=config_path,
    )

    assert panel.automation_readiness == "ready"
    assert panel.lock_status == "held by another process"
    assert panel.account_id_masked == "****1234"


def test_redacted_loader_fails_closed_for_invalid_expected_account_config(
    monkeypatch, tmp_path
) -> None:
    artifact = tmp_path / "reconciliation.json"
    artifact.write_text(json.dumps(_safe_payload()), encoding="utf-8")
    config_path = tmp_path / "paper_shadow_config.json"
    config_path.write_text(
        json.dumps(_paper_shadow_config_payload(expected_account_fingerprint=1234)),
        encoding="utf-8",
    )

    import tradeos.execution_lock as execution_lock

    monkeypatch.setattr(execution_lock, "inspect_lock", lambda: _lock())

    panel = load_broker_safety_panel_state(
        artifact,
        now_utc=NOW,
        readiness_config_path=config_path,
    )

    assert panel.automation_readiness == "blocked"
    assert "expected_account_fingerprint_missing_or_invalid" in panel.blocking_reasons


def test_broker_safety_projection_rejects_conflicting_complete_metadata() -> None:
    payload = _safe_payload(
        local_positions={"AAPL": "0"},
        state_mismatches=[],
    )

    panel = _build_broker_safety_panel_state(
        payload,
        _lock(),
        now_utc=NOW,
        expected_account_fingerprint=ACCOUNT_FINGERPRINT,
    )

    assert panel.automation_readiness == "blocked"
    assert "reconciliation_metadata_inconsistent" in panel.blocking_reasons


def test_broker_safety_projection_requires_repository_readiness() -> None:
    panel = _build_broker_safety_panel_state(
        _safe_payload(),
        _lock(),
        now_utc=NOW,
        repository_readiness_reasons=("repository_readiness_evidence_missing_or_invalid",),
        expected_account_fingerprint=ACCOUNT_FINGERPRINT,
    )

    assert panel.automation_readiness == "blocked"
    assert "repository_readiness_evidence_missing_or_invalid" in panel.blocking_reasons


def test_broker_safety_panel_html_is_compact_complete_and_escaped() -> None:
    panel = BrokerSafetyPanelState(
        environment="unknown",
        account_id_masked="unknown",
        lock_status="unknown",
        reconciliation_time="unknown",
        managed_position_count=None,
        unmanaged_position_count=None,
        managed_order_count=None,
        unmanaged_order_count=None,
        mismatch_status="unknown",
        automation_readiness="blocked",
        blocking_reasons=("bad <script>alert('x')</script>",),
        result="BLOCKED_RECONCILIATION_FAILURE",
        stale=True,
    )

    rendered = dashboard._broker_safety_panel_html(panel)

    assert "Broker Safety (read only)" in rendered
    assert "Masked Account" in rendered
    assert "Trading Lock" in rendered
    assert "Managed Positions" in rendered
    assert "Unmanaged Positions" in rendered
    assert "Managed Open Orders" in rendered
    assert "Unmanaged Open Orders" in rendered
    assert "Local vs Broker" in rendered
    assert "Automation Readiness" in rendered
    assert "Exact Blocking Reasons" in rendered
    assert "BLOCKED" in rendered
    assert "bad &lt;script&gt;alert(&#x27;x&#x27;)&lt;/script&gt;" in rendered
    assert "<script>" not in rendered

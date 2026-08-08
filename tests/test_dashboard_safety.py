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


def _lock(state: str = "unlocked", *, metadata_valid: bool = True):
    return SimpleNamespace(
        state=SimpleNamespace(value=state),
        held_by_current_process=False,
        metadata_valid=metadata_valid,
    )


def _safe_payload(**updates):
    payload = {
        "environment": "paper",
        "account_id_masked": "****1234",
        "checked_at_utc": NOW.isoformat(),
        "reconciliation_succeeded": True,
        "managed_positions": [{"symbol": "AAPL"}],
        "unmanaged_positions": [],
        "managed_orders": [{"symbol": "AAPL"}],
        "unmanaged_orders": [],
        "state_mismatches": [],
        "blocking_reasons": [],
        "automation_readiness": "ready",
        "result": "SAFE_FOR_PAPER_AUTOMATION",
    }
    payload.update(updates)
    return payload


def test_broker_safety_projection_accepts_only_fresh_positive_paper_state() -> None:
    panel = _build_broker_safety_panel_state(_safe_payload(), _lock(), now_utc=NOW)

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


def test_broker_safety_projection_fails_closed_for_missing_or_malformed_artifact() -> None:
    missing = _build_broker_safety_panel_state(None, _lock(), now_utc=NOW)
    malformed = _build_broker_safety_panel_state(
        {"environment": "paper", "blocking_reasons": []},
        _lock(),
        now_utc=NOW,
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

    panel = _build_broker_safety_panel_state(stale_payload, _lock(), now_utc=NOW)

    assert panel.automation_readiness == "blocked"
    assert panel.stale is True
    assert "reconciliation_stale_over_5_minutes" in panel.blocking_reasons


def test_broker_safety_projection_preserves_and_derives_exact_blocking_reasons() -> None:
    payload = _safe_payload(
        unmanaged_positions=[{"symbol": "COP"}],
        unmanaged_orders=[{"symbol": "COP"}],
        state_mismatches=["COP broker=-10 local=10"],
        blocking_reasons=["unmanaged broker exposure"],
        automation_readiness="blocked",
        result="BLOCKED_UNMANAGED_POSITION",
    )

    panel = _build_broker_safety_panel_state(payload, _lock("unknown"), now_utc=NOW)

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

    import tradeos.execution_lock as execution_lock

    monkeypatch.setattr(execution_lock, "inspect_lock", lambda: _lock("held"))

    panel = load_broker_safety_panel_state(artifact, now_utc=NOW)

    assert panel.automation_readiness == "ready"
    assert panel.lock_status == "held by another process"
    assert panel.account_id_masked == "****1234"


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

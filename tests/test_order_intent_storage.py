from __future__ import annotations

from pathlib import Path

import pytest

from storage import (
    ORDER_INTENT_ACCEPTED,
    ORDER_INTENT_SUBMITTING,
    ORDER_INTENT_TERMINAL,
    BotStorage,
)


FINGERPRINT = "a" * 64
OTHER_FINGERPRINT = "b" * 64
T0 = "2026-08-06T15:00:00+00:00"
T1 = "2026-08-06T15:00:01+00:00"
T2 = "2026-08-06T15:00:02+00:00"
T3 = "2026-08-06T15:00:03+00:00"


@pytest.fixture
def storage(tmp_path: Path) -> BotStorage:
    result = BotStorage((tmp_path / "order-intents.db").resolve())
    result.bind_runtime_state_scope(
        environment="paper",
        account_fingerprint=FINGERPRINT,
        provenance="order-intent-storage-test",
        history_cutoff_utc=T0,
        bound_at_utc=T0,
    )
    return result


def _reserve(
    storage: BotStorage,
    *,
    suffix: str,
    mark_submitting: bool = True,
) -> tuple[str, str]:
    intent_id = f"intent-{suffix}"
    client_order_id = f"client-{suffix}"
    storage.reserve_order_intent(
        intent_id=intent_id,
        client_order_id=client_order_id,
        environment="paper",
        account_fingerprint=FINGERPRINT,
        decision_ts=f"decision-{suffix}",
        symbol="AMD",
        side="buy",
        intent_kind="entry",
        requested_qty="5",
        final_qty="5",
        created_at_utc=T0,
    )
    if mark_submitting:
        assert storage.mark_order_intent_submitting(
            intent_id,
            updated_at_utc=T1,
        )
    return intent_id, client_order_id


def _intent(storage: BotStorage, intent_id: str) -> dict[str, object]:
    payload = storage.get_order_intent(intent_id)
    assert payload is not None
    return payload


def test_normal_order_intent_progression_is_monotonic(storage: BotStorage) -> None:
    intent_id, client_order_id = _reserve(storage, suffix="normal")
    assert _intent(storage, intent_id)["status"] == ORDER_INTENT_SUBMITTING

    storage.mark_order_intent_accepted(
        intent_id,
        broker_order_id="broker-normal",
        broker_status="new",
        filled_qty="0",
        updated_at_utc=T1,
    )
    accepted = _intent(storage, intent_id)
    assert accepted["status"] == ORDER_INTENT_ACCEPTED
    assert accepted["broker_status"] == "new"
    assert accepted["filled_qty"] == "0"

    assert storage.reconcile_order_intent(
        client_order_id,
        broker_order_id="broker-normal",
        broker_status="partially_filled",
        filled_qty="2.000",
        updated_at_utc=T2,
        environment="paper",
        account_fingerprint=FINGERPRINT,
    )
    partial = _intent(storage, intent_id)
    assert partial["status"] == ORDER_INTENT_ACCEPTED
    assert partial["broker_status"] == "partially_filled"
    assert partial["filled_qty"] == "2"

    assert storage.reconcile_order_intent(
        client_order_id,
        broker_order_id="broker-normal",
        broker_status="filled",
        filled_qty="5",
        updated_at_utc=T3,
        environment="paper",
        account_fingerprint=FINGERPRINT,
    )
    terminal = _intent(storage, intent_id)
    assert terminal["status"] == ORDER_INTENT_TERMINAL
    assert terminal["broker_status"] == "filled"
    assert terminal["filled_qty"] == "5"


def test_stale_acceptance_after_fill_cannot_regress_terminal_state(
    storage: BotStorage,
) -> None:
    intent_id, _ = _reserve(storage, suffix="stale-acceptance")
    storage.mark_order_intent_accepted(
        intent_id,
        broker_order_id="broker-stale-acceptance",
        broker_status="filled",
        filled_qty="5",
        updated_at_utc=T2,
    )
    before = _intent(storage, intent_id)

    storage.mark_order_intent_accepted(
        intent_id,
        broker_order_id="broker-stale-acceptance",
        broker_status="new",
        filled_qty="0",
        updated_at_utc=T3,
    )

    assert _intent(storage, intent_id) == before


def test_stale_reconciliation_after_fill_cannot_regress_terminal_state(
    storage: BotStorage,
) -> None:
    intent_id, client_order_id = _reserve(storage, suffix="stale-reconcile")
    assert storage.reconcile_order_intent(
        client_order_id,
        broker_order_id="broker-stale-reconcile",
        broker_status="filled",
        filled_qty="5",
        updated_at_utc=T2,
        environment="paper",
        account_fingerprint=FINGERPRINT,
    )
    before = _intent(storage, intent_id)

    assert storage.reconcile_order_intent(
        client_order_id,
        broker_order_id="broker-stale-reconcile",
        broker_status="accepted",
        filled_qty="0",
        updated_at_utc=T3,
        environment="paper",
        account_fingerprint=FINGERPRINT,
    )

    assert _intent(storage, intent_id) == before


def test_decreasing_cumulative_fill_is_rejected_without_mutation(
    storage: BotStorage,
) -> None:
    intent_id, client_order_id = _reserve(storage, suffix="decreasing-fill")
    storage.mark_order_intent_accepted(
        intent_id,
        broker_order_id="broker-decreasing-fill",
        broker_status="partially_filled",
        filled_qty="3",
        updated_at_utc=T2,
    )
    before = _intent(storage, intent_id)

    with pytest.raises(RuntimeError, match="filled quantity cannot decrease"):
        storage.reconcile_order_intent(
            client_order_id,
            broker_order_id="broker-decreasing-fill",
            broker_status="partially_filled",
            filled_qty="2",
            updated_at_utc=T3,
            environment="paper",
            account_fingerprint=FINGERPRINT,
        )

    assert _intent(storage, intent_id) == before


def test_broker_order_id_cannot_be_rebound(storage: BotStorage) -> None:
    intent_id, client_order_id = _reserve(storage, suffix="order-rebind")
    storage.mark_order_intent_accepted(
        intent_id,
        broker_order_id="broker-original",
        broker_status="new",
        filled_qty="0",
        updated_at_utc=T2,
    )
    before = _intent(storage, intent_id)

    with pytest.raises(RuntimeError, match="broker order id is already bound"):
        storage.reconcile_order_intent(
            client_order_id,
            broker_order_id="broker-contradiction",
            broker_status="new",
            filled_qty="0",
            updated_at_utc=T3,
            environment="paper",
            account_fingerprint=FINGERPRINT,
        )

    assert _intent(storage, intent_id) == before


def test_contradictory_terminal_status_is_rejected(storage: BotStorage) -> None:
    intent_id, client_order_id = _reserve(storage, suffix="terminal-conflict")
    storage.mark_order_intent_accepted(
        intent_id,
        broker_order_id="broker-terminal-conflict",
        broker_status="filled",
        filled_qty="5",
        updated_at_utc=T2,
    )
    before = _intent(storage, intent_id)

    with pytest.raises(RuntimeError, match="contradictory terminal broker status"):
        storage.reconcile_order_intent(
            client_order_id,
            broker_order_id="broker-terminal-conflict",
            broker_status="canceled",
            filled_qty="5",
            updated_at_utc=T3,
            environment="paper",
            account_fingerprint=FINGERPRINT,
        )

    assert _intent(storage, intent_id) == before


@pytest.mark.parametrize("filled_qty", ["-1", "NaN", "Infinity", "not-a-number"])
def test_invalid_filled_quantity_is_rejected(
    storage: BotStorage,
    filled_qty: str,
) -> None:
    intent_id, _ = _reserve(storage, suffix=f"invalid-{filled_qty}")
    before = _intent(storage, intent_id)

    with pytest.raises(ValueError, match="filled quantity"):
        storage.mark_order_intent_accepted(
            intent_id,
            broker_order_id=f"broker-invalid-{filled_qty}",
            broker_status="new",
            filled_qty=filled_qty,
            updated_at_utc=T2,
        )

    assert _intent(storage, intent_id) == before


def test_prepared_intent_cannot_skip_submission_lifecycle(storage: BotStorage) -> None:
    intent_id, _ = _reserve(
        storage,
        suffix="prepared",
        mark_submitting=False,
    )
    before = _intent(storage, intent_id)

    with pytest.raises(RuntimeError, match="cannot merge broker state"):
        storage.mark_order_intent_accepted(
            intent_id,
            broker_order_id="broker-prepared",
            broker_status="new",
            filled_qty="0",
            updated_at_utc=T2,
        )

    assert _intent(storage, intent_id) == before


def test_scoped_reconciliation_does_not_touch_another_bound_scope(
    storage: BotStorage,
) -> None:
    intent_id, client_order_id = _reserve(storage, suffix="scope")
    storage.bind_runtime_state_scope(
        environment="paper",
        account_fingerprint=OTHER_FINGERPRINT,
        provenance="other-order-intent-storage-test",
        history_cutoff_utc=T0,
        bound_at_utc=T0,
    )
    before = _intent(storage, intent_id)

    assert not storage.reconcile_order_intent(
        client_order_id,
        broker_order_id="broker-wrong-scope",
        broker_status="new",
        filled_qty="0",
        updated_at_utc=T2,
        environment="paper",
        account_fingerprint=OTHER_FINGERPRINT,
    )

    assert _intent(storage, intent_id) == before

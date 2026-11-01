from __future__ import annotations

import hashlib
import json
import os
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Callable, Literal

from storage import (
    ORDER_INTENT_ACCEPTED,
    ORDER_INTENT_AMBIGUOUS,
    ORDER_INTENT_SUBMITTING,
    BotStorage,
)
from tradeos.brokers.base import BrokerAccount, BrokerClient, BrokerOrder, BrokerPosition
from tradeos.runtime_paths import TRADEOS_REPOSITORY_ROOT


CLIENT_ORDER_ID_PREFIX = "av1-"
EXPECTED_ACCOUNT_ID_ENV = "ALPACA_EXPECTED_ACCOUNT_ID"
DEFAULT_RECONCILIATION_PATH = (
    TRADEOS_REPOSITORY_ROOT / "logs" / "safety" / "reconciliation.json"
)

SAFE_FOR_PAPER_AUTOMATION = "SAFE_FOR_PAPER_AUTOMATION"
BLOCKED_UNMANAGED_POSITION = "BLOCKED_UNMANAGED_POSITION"
BLOCKED_UNMANAGED_ORDER = "BLOCKED_UNMANAGED_ORDER"
BLOCKED_STATE_MISMATCH = "BLOCKED_STATE_MISMATCH"
BLOCKED_LIVE_CONFIGURATION = "BLOCKED_LIVE_CONFIGURATION"
BLOCKED_RECONCILIATION_FAILURE = "BLOCKED_RECONCILIATION_FAILURE"
BLOCKED_AMBIGUOUS_SUBMISSION = "BLOCKED_AMBIGUOUS_SUBMISSION"

ENTRY_INTENT = "entry"
LONG_EXIT_INTENT = "long_exit"
SHORT_COVER_INTENT = "short_cover"
FORCED_LONG_EXIT_INTENT = "forced_long_exit"
_PROCESS_SUBMISSION_MUTEX = threading.Lock()

# Alpaca can expose several intermediate order states.  Treat only states that
# positively mean the original order can no longer fill as terminal.  Unknown
# states fail reconciliation rather than being guessed safe.
TERMINAL_ORDER_STATUSES = frozenset(
    {"canceled", "cancelled", "expired", "filled", "rejected", "replaced"}
)
KNOWN_ORDER_STATUSES = frozenset(
    {
        "accepted",
        "accepted_for_bidding",
        "calculated",
        "canceled",
        "cancelled",
        "done_for_day",
        "expired",
        "filled",
        "new",
        "partially_filled",
        "pending_cancel",
        "pending_new",
        "pending_replace",
        "rejected",
        "replaced",
        "stopped",
        "suspended",
    }
)


class ReconciliationError(RuntimeError):
    pass


class AmbiguousSubmissionError(RuntimeError):
    """The broker may have accepted an order; automatic retry is forbidden."""


def _enum_text(value: Any) -> str:
    text = str(value or "").strip()
    if "." in text:
        text = text.rsplit(".", 1)[-1]
    return text.lower()


def to_decimal(value: Any, *, field_name: str = "quantity") -> Decimal:
    if value is None or isinstance(value, bool):
        raise ReconciliationError(f"{field_name} is missing or invalid")
    try:
        normalized = Decimal(str(value))
    except (InvalidOperation, ValueError, TypeError) as exc:
        raise ReconciliationError(f"{field_name} is malformed") from exc
    if not normalized.is_finite():
        raise ReconciliationError(f"{field_name} must be finite")
    return normalized


def positive_decimal(value: Any, *, field_name: str = "quantity") -> Decimal:
    normalized = to_decimal(value, field_name=field_name)
    if normalized <= 0:
        raise ReconciliationError(f"{field_name} must be positive")
    return normalized


def decimal_text(value: Decimal | Any) -> str:
    normalized = to_decimal(value)
    text = format(normalized, "f")
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    return text or "0"


def mask_identifier(value: str | None) -> str:
    raw = str(value or "").strip()
    if not raw:
        return "unknown"
    if len(raw) <= 4:
        return "*" * len(raw)
    return f"{'*' * min(8, len(raw) - 4)}{raw[-4:]}"


def account_fingerprint(value: str) -> str:
    raw = str(value).strip()
    if not raw:
        return ""
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def build_intent_identity(
    *,
    account_fingerprint_value: str,
    decision_id: str,
    symbol: str,
    side: str,
    intent_kind: str,
) -> tuple[str, str]:
    logical = "|".join(
        (
            "automation-v1",
            account_fingerprint_value,
            str(decision_id).strip(),
            str(symbol).strip().upper(),
            str(side).strip().lower(),
            str(intent_kind).strip().lower(),
        )
    )
    digest = hashlib.sha256(logical.encode("utf-8")).hexdigest()
    return digest, f"{CLIENT_ORDER_ID_PREFIX}{digest[:40]}"


def remaining_order_quantity(order: BrokerOrder) -> Decimal:
    qty = positive_decimal(getattr(order, "qty", None), field_name="order quantity")
    filled = to_decimal(
        getattr(order, "filled_qty", 0) or 0,
        field_name="filled quantity",
    )
    if filled < 0 or filled > qty:
        raise ReconciliationError("filled quantity is outside the order quantity")
    return qty - filled


def normalized_order_status(order: BrokerOrder) -> str:
    status = _enum_text(getattr(order, "status", ""))
    if status not in KNOWN_ORDER_STATUSES:
        raise ReconciliationError("broker order status is missing or unknown")
    return status


def _order_immutable_identity(order: BrokerOrder) -> tuple[str, str, str, Decimal]:
    order_id = str(getattr(order, "order_id", "") or "").strip()
    symbol = str(getattr(order, "symbol", "") or "").strip().upper()
    side = _enum_text(getattr(order, "side", ""))
    qty = positive_decimal(getattr(order, "qty", None), field_name="order quantity")
    if not order_id or not symbol or side not in {"buy", "sell"}:
        raise ReconciliationError("broker order is missing required identity fields")
    normalized_order_status(order)
    remaining_order_quantity(order)
    return order_id, symbol, side, qty


def _same_order_identity(left: BrokerOrder, right: BrokerOrder) -> bool:
    return _order_immutable_identity(left) == _order_immutable_identity(right)


def available_to_sell(long_quantity: Any, pending_sell_quantity: Any) -> Decimal:
    long_qty = to_decimal(long_quantity, field_name="broker long quantity")
    pending = to_decimal(pending_sell_quantity, field_name="pending sell quantity")
    if long_qty <= 0 or pending < 0:
        return Decimal("0")
    return max(long_qty - pending, Decimal("0"))


def available_to_cover(short_quantity: Any, pending_cover_quantity: Any) -> Decimal:
    position_qty = to_decimal(short_quantity, field_name="broker short quantity")
    pending = to_decimal(pending_cover_quantity, field_name="pending cover quantity")
    if position_qty >= 0 or pending < 0:
        return Decimal("0")
    return max(abs(position_qty) - pending, Decimal("0"))


@dataclass(frozen=True)
class ClassifiedPosition:
    symbol: str
    qty: Decimal
    managed: bool
    reason: str


@dataclass(frozen=True)
class ClassifiedOrder:
    order_id: str
    client_order_id: str
    symbol: str
    side: str
    status: str
    qty: Decimal
    filled_qty: Decimal
    remaining_qty: Decimal
    managed: bool
    reason: str


@dataclass(frozen=True)
class ReconciliationReport:
    checked_at_utc: str
    environment: str
    account_id_masked: str = "unknown"
    account_status: str = "unknown"
    account_fingerprint: str = field(default="", repr=False)
    reconciliation_succeeded: bool = False
    broker_positions: tuple[ClassifiedPosition, ...] = ()
    open_orders: tuple[ClassifiedOrder, ...] = ()
    local_positions: tuple[tuple[str, Decimal], ...] = ()
    pending_buy_qty: tuple[tuple[str, Decimal], ...] = ()
    pending_sell_qty: tuple[tuple[str, Decimal], ...] = ()
    state_mismatches: tuple[str, ...] = ()
    ambiguous_intents: tuple[str, ...] = ()
    blocking_reasons: tuple[str, ...] = ("reconciliation_not_completed",)
    result: str = BLOCKED_RECONCILIATION_FAILURE
    ready: bool = False

    @property
    def managed_positions(self) -> tuple[ClassifiedPosition, ...]:
        return tuple(item for item in self.broker_positions if item.managed)

    @property
    def unmanaged_positions(self) -> tuple[ClassifiedPosition, ...]:
        return tuple(item for item in self.broker_positions if not item.managed)

    @property
    def managed_orders(self) -> tuple[ClassifiedOrder, ...]:
        return tuple(item for item in self.open_orders if item.managed)

    @property
    def unmanaged_orders(self) -> tuple[ClassifiedOrder, ...]:
        return tuple(item for item in self.open_orders if not item.managed)

    def position_qty(self, symbol: str) -> Decimal:
        normalized = str(symbol).strip().upper()
        for item in self.broker_positions:
            if item.symbol == normalized:
                return item.qty
        return Decimal("0")

    def position_is_managed(self, symbol: str) -> bool:
        normalized = str(symbol).strip().upper()
        return any(item.symbol == normalized and item.managed for item in self.broker_positions)

    def pending_buy_for(self, symbol: str) -> Decimal:
        return dict(self.pending_buy_qty).get(str(symbol).strip().upper(), Decimal("0"))

    def pending_sell_for(self, symbol: str) -> Decimal:
        return dict(self.pending_sell_qty).get(str(symbol).strip().upper(), Decimal("0"))

    def to_redacted_dict(self) -> dict[str, Any]:
        def position_payload(item: ClassifiedPosition) -> dict[str, Any]:
            return {
                "symbol": item.symbol,
                "qty": decimal_text(item.qty),
                "managed": item.managed,
                "reason": item.reason,
            }

        def order_payload(item: ClassifiedOrder) -> dict[str, Any]:
            return {
                "order_id": item.order_id,
                "client_order_id": item.client_order_id,
                "symbol": item.symbol,
                "side": item.side,
                "status": item.status,
                "qty": decimal_text(item.qty),
                "filled_qty": decimal_text(item.filled_qty),
                "remaining_qty": decimal_text(item.remaining_qty),
                "managed": item.managed,
                "reason": item.reason,
            }

        return {
            "checked_at_utc": self.checked_at_utc,
            "environment": self.environment,
            "account_id_masked": self.account_id_masked,
            "account_fingerprint": self.account_fingerprint,
            "account_status": self.account_status,
            "reconciliation_succeeded": self.reconciliation_succeeded,
            "current_broker_positions": [position_payload(item) for item in self.broker_positions],
            "managed_positions": [position_payload(item) for item in self.managed_positions],
            "unmanaged_positions": [position_payload(item) for item in self.unmanaged_positions],
            "current_open_pending_orders": [order_payload(item) for item in self.open_orders],
            "managed_orders": [order_payload(item) for item in self.managed_orders],
            "unmanaged_orders": [order_payload(item) for item in self.unmanaged_orders],
            "pending_buy_qty": {key: decimal_text(value) for key, value in self.pending_buy_qty},
            "pending_sell_qty": {key: decimal_text(value) for key, value in self.pending_sell_qty},
            "local_positions": {key: decimal_text(value) for key, value in self.local_positions},
            "state_mismatches": list(self.state_mismatches),
            "ambiguous_intents": list(self.ambiguous_intents),
            "blocking_reasons": list(self.blocking_reasons),
            "automation_readiness": "ready" if self.ready else "blocked",
            "result": self.result,
        }


@dataclass(frozen=True)
class OrderRequest:
    symbol: str
    side: Literal["buy", "sell"]
    qty: Any
    intent_kind: str
    decision_id: str
    explicit_cover: bool = False


@dataclass(frozen=True)
class SubmissionOutcome:
    allowed: bool
    reason: str
    requested_qty: Decimal
    final_qty: Decimal = Decimal("0")
    intent_id: str | None = None
    client_order_id: str | None = None
    order: BrokerOrder | None = None


def _account_identity(account: BrokerAccount) -> tuple[str, tuple[str, ...]]:
    candidates = tuple(
        value
        for value in (
            str(getattr(account, "account_id", "") or "").strip(),
            str(getattr(account, "account_number", "") or "").strip(),
        )
        if value
    )
    return (candidates[0] if candidates else ""), candidates


def _local_positions(
    storage: BotStorage,
    *,
    environment: str,
    account_fingerprint_value: str,
) -> dict[str, Decimal]:
    result: dict[str, Decimal] = {}
    for row in storage.get_latest_symbol_snapshot(
        environment=environment,
        account_fingerprint=account_fingerprint_value,
    ):
        symbol = str(row.get("symbol", "")).strip().upper()
        if not symbol:
            continue
        qty = to_decimal(row.get("quantity", 0), field_name=f"local quantity for {symbol}")
        if qty != 0:
            result[symbol] = qty
    return result


def _intent_matches_order(
    intent: dict[str, Any],
    order: BrokerOrder,
    fingerprint: str,
    environment: str,
) -> bool:
    try:
        order_qty = positive_decimal(getattr(order, "qty", None), field_name="order quantity")
        expected_qty = positive_decimal(intent.get("final_qty"), field_name="intent quantity")
    except ReconciliationError:
        return False
    return all(
        (
            str(intent.get("environment", "")).strip().lower() == environment,
            str(intent.get("account_fingerprint", "")) == fingerprint,
            str(intent.get("client_order_id", ""))
            == str(getattr(order, "client_order_id", "") or ""),
            str(intent.get("symbol", "")).strip().upper()
            == str(getattr(order, "symbol", "")).strip().upper(),
            str(intent.get("side", "")).strip().lower()
            == _enum_text(getattr(order, "side", "")),
            expected_qty == order_qty,
            int(intent.get("submission_attempts", 0) or 0) > 0,
        )
    )


class SafetyReconciler:
    """Read broker state and conservatively bind it to durable Automation V1 state."""

    def __init__(
        self,
        *,
        broker: BrokerClient | None,
        storage: BotStorage,
        configured_paper: bool,
        expected_account_id: str | None = None,
        report_path: Path | None = DEFAULT_RECONCILIATION_PATH,
    ) -> None:
        self.broker = broker
        self.storage = storage
        self.configured_paper = configured_paper
        self.expected_account_id = (
            str(expected_account_id).strip()
            if expected_account_id is not None
            else os.getenv(EXPECTED_ACCOUNT_ID_ENV, "").strip()
        )
        self.report_path = report_path
        self._last_report: ReconciliationReport | None = None

    @property
    def last_report(self) -> ReconciliationReport | None:
        return self._last_report

    def _persist(self, report: ReconciliationReport) -> ReconciliationReport:
        self._last_report = report
        if self.report_path is None:
            return report
        self.report_path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = self.report_path.with_suffix(f"{self.report_path.suffix}.tmp")
        temporary_path.write_text(
            json.dumps(report.to_redacted_dict(), indent=2, sort_keys=True),
            encoding="utf-8",
        )
        temporary_path.replace(self.report_path)
        return report

    def _blocked_without_broker(self, *, environment: str, reason: str, result: str) -> ReconciliationReport:
        return self._persist(
            ReconciliationReport(
                checked_at_utc=datetime.now(timezone.utc).isoformat(),
                environment=environment,
                blocking_reasons=(reason,),
                result=result,
                ready=False,
            )
        )

    def reconcile(self) -> ReconciliationReport:
        if not self.configured_paper:
            return self._blocked_without_broker(
                environment="live",
                reason="configured_environment_is_not_paper",
                result=BLOCKED_LIVE_CONFIGURATION,
            )
        if self.broker is None:
            return self._blocked_without_broker(
                environment="unknown",
                reason="paper_broker_is_unavailable",
                result=BLOCKED_RECONCILIATION_FAILURE,
            )

        checked_at = datetime.now(timezone.utc).isoformat()
        try:
            account = self.broker.get_account()
        except Exception as exc:
            return self._persist(
                ReconciliationReport(
                    checked_at_utc=checked_at,
                    environment="paper",
                    blocking_reasons=(
                        f"broker_account_query_failed:{type(exc).__name__}",
                    ),
                    result=BLOCKED_RECONCILIATION_FAILURE,
                )
            )

        environment = _enum_text(getattr(account, "environment", "unknown")) or "unknown"
        actual_account_id, account_candidates = _account_identity(account)
        fingerprint = account_fingerprint(actual_account_id)
        account_status = _enum_text(getattr(account, "status", "unknown")) or "unknown"

        def query_failure(stage: str, exc: Exception) -> ReconciliationReport:
            return self._persist(
                ReconciliationReport(
                    checked_at_utc=checked_at,
                    environment=environment,
                    account_id_masked=mask_identifier(actual_account_id),
                    account_status=account_status,
                    blocking_reasons=(f"{stage}_failed:{type(exc).__name__}",),
                    result=BLOCKED_RECONCILIATION_FAILURE,
                )
            )

        try:
            broker_positions_raw = list(self.broker.list_positions())
        except Exception as exc:
            return query_failure("broker_positions_query", exc)
        try:
            open_orders_raw = list(self.broker.list_open_orders())
        except Exception as exc:
            return query_failure("broker_open_orders_query", exc)
        try:
            recent_orders_raw = list(self.broker.list_recent_orders(limit=500))
        except Exception as exc:
            return query_failure("broker_recent_orders_query", exc)
        blocking: list[str] = []
        mismatches: list[str] = []

        if environment != "paper":
            blocking.append("broker_environment_not_verified_as_paper")
        if not account_candidates:
            blocking.append("broker_account_identifier_missing")
        if not self.expected_account_id:
            blocking.append("expected_account_id_not_configured")
        elif self.expected_account_id not in account_candidates:
            blocking.append("broker_account_does_not_match_expected_account")
        if account_status != "active":
            blocking.append(f"broker_account_status_{account_status}")
        for field_name in (
            "trading_blocked",
            "account_blocked",
            "trade_suspended_by_user",
        ):
            if bool(getattr(account, field_name, False)):
                blocking.append(field_name)

        identity_authenticated = bool(
            environment == "paper"
            and account_candidates
            and self.expected_account_id
            and self.expected_account_id in account_candidates
        )
        account_operational = bool(
            identity_authenticated
            and account_status == "active"
            and not any(
                bool(getattr(account, field_name, False))
                for field_name in (
                    "trading_blocked",
                    "account_blocked",
                    "trade_suspended_by_user",
                )
            )
        )
        local_positions: dict[str, Decimal] = {}
        intents: list[dict[str, Any]] = []
        scope_binding: dict[str, Any] | None = None
        if identity_authenticated:
            try:
                scope_binding = self.storage.get_runtime_state_binding(
                    environment=environment,
                    account_fingerprint=fingerprint,
                )
                if scope_binding is None:
                    legacy_open = self.storage.get_latest_unbound_open_positions()
                    if legacy_open:
                        blocking.append("unbound_legacy_open_state")
                        legacy_symbols = sorted(
                            {
                                str(row.get("symbol", "")).strip().upper()
                                for row in legacy_open
                                if str(row.get("symbol", "")).strip()
                            }
                        )
                        mismatches.extend(
                            f"unbound_legacy_open_state:{symbol}"
                            for symbol in legacy_symbols
                        )
                    elif account_operational:
                        provenance = json.dumps(
                            {
                                "schema_version": 1,
                                "source": "authenticated_reconciliation",
                                "legacy_open_state": "none",
                            },
                            sort_keys=True,
                            separators=(",", ":"),
                        )
                        self.storage.bind_runtime_state_scope(
                            environment=environment,
                            account_fingerprint=fingerprint,
                            provenance=provenance,
                            history_cutoff_utc=checked_at,
                            bound_at_utc=checked_at,
                        )
                        scope_binding = self.storage.get_runtime_state_binding(
                            environment=environment,
                            account_fingerprint=fingerprint,
                        )
                    else:
                        blocking.append("runtime_state_scope_unbound")

                if scope_binding is not None:
                    local_positions = _local_positions(
                        self.storage,
                        environment=environment,
                        account_fingerprint_value=fingerprint,
                    )
                    intents = self.storage.list_order_intents(
                        environment=environment,
                        account_fingerprint=fingerprint,
                    )
            except Exception as exc:
                return query_failure("runtime_scope_query", exc)

        # Build one conservative broker-order view.  A nonterminal order seen
        # in recent results or by deterministic client ID still reserves its
        # remaining quantity even if list_open_orders momentarily omits it.
        observed_orders: list[BrokerOrder] = [*recent_orders_raw, *open_orders_raw]
        order_by_client: dict[str, BrokerOrder] = {}
        try:
            for order in open_orders_raw:
                if normalized_order_status(order) in TERMINAL_ORDER_STATUSES:
                    raise ReconciliationError("terminal order returned by open-order query")
            for order in observed_orders:
                _order_immutable_identity(order)
                client_id = str(getattr(order, "client_order_id", "") or "").strip()
                if not client_id:
                    continue
                previous = order_by_client.get(client_id)
                if previous is not None and not _same_order_identity(previous, order):
                    raise ReconciliationError("conflicting broker orders share a client order id")
                # Open-order results occur last and therefore win over a stale
                # recent-order copy with the same immutable identity.
                order_by_client[client_id] = order
        except ReconciliationError:
            return self._persist(
                ReconciliationReport(
                    checked_at_utc=checked_at,
                    environment=environment,
                    account_id_masked=mask_identifier(actual_account_id),
                    account_status=account_status,
                    blocking_reasons=("malformed_or_conflicting_order_response",),
                    result=BLOCKED_RECONCILIATION_FAILURE,
                )
            )

        intents_by_client = {
            str(intent.get("client_order_id", "")): intent
            for intent in intents
            if str(intent.get("client_order_id", ""))
        }
        unresolved_intent_ids: list[str] = []
        for client_id, intent in intents_by_client.items():
            order = order_by_client.get(client_id)
            if order is None and str(intent.get("status", "")) in {
                ORDER_INTENT_SUBMITTING,
                ORDER_INTENT_AMBIGUOUS,
                ORDER_INTENT_ACCEPTED,
            }:
                try:
                    order = self.broker.get_order_by_client_order_id(client_id)
                except Exception:
                    return self._persist(
                        ReconciliationReport(
                            checked_at_utc=checked_at,
                            environment=environment,
                            account_id_masked=mask_identifier(actual_account_id),
                            account_status=account_status,
                            blocking_reasons=("ambiguous_order_lookup_failed",),
                            result=BLOCKED_RECONCILIATION_FAILURE,
                        )
                    )
            if order is None:
                if str(intent.get("status", "")) in {
                    ORDER_INTENT_SUBMITTING,
                    ORDER_INTENT_AMBIGUOUS,
                    ORDER_INTENT_ACCEPTED,
                }:
                    unresolved_intent_ids.append(str(intent.get("intent_id", "")))
                continue
            try:
                _order_immutable_identity(order)
                previous = order_by_client.get(client_id)
                if previous is not None and not _same_order_identity(previous, order):
                    raise ReconciliationError("client-order lookup conflicts with list results")
            except ReconciliationError:
                return self._persist(
                    ReconciliationReport(
                        checked_at_utc=checked_at,
                        environment=environment,
                        account_id_masked=mask_identifier(actual_account_id),
                        account_status=account_status,
                        blocking_reasons=("malformed_or_conflicting_order_response",),
                        result=BLOCKED_RECONCILIATION_FAILURE,
                    )
                )
            if order not in observed_orders:
                observed_orders.append(order)
            order_by_client[client_id] = order
            if not _intent_matches_order(
                intent,
                order,
                fingerprint,
                environment,
            ):
                mismatches.append(
                    f"intent_order_mismatch:{str(intent.get('symbol', '')).upper()}:{client_id}"
                )
                continue
            try:
                self.storage.reconcile_order_intent(
                    client_id,
                    broker_order_id=str(getattr(order, "order_id", "") or ""),
                    broker_status=_enum_text(getattr(order, "status", "")),
                    filled_qty=decimal_text(
                        to_decimal(getattr(order, "filled_qty", 0) or 0)
                    ),
                    updated_at_utc=checked_at,
                    environment=environment,
                    account_fingerprint=fingerprint,
                )
            except Exception:
                return self._persist(
                    ReconciliationReport(
                        checked_at_utc=checked_at,
                        environment=environment,
                        account_id_masked=mask_identifier(actual_account_id),
                        account_status=account_status,
                        blocking_reasons=("local_intent_reconciliation_failed",),
                        result=BLOCKED_RECONCILIATION_FAILURE,
                    )
                )

        if scope_binding is not None:
            intents = self.storage.list_order_intents(
                environment=environment,
                account_fingerprint=fingerprint,
            )
        intents_by_client = {
            str(intent.get("client_order_id", "")): intent
            for intent in intents
            if str(intent.get("client_order_id", ""))
        }
        ambiguous_intents = tuple(
            dict.fromkeys(
                [
                    str(intent.get("intent_id", ""))
                    for intent in intents
                    if str(intent.get("status", ""))
                    in {ORDER_INTENT_SUBMITTING, ORDER_INTENT_AMBIGUOUS}
                ]
                + unresolved_intent_ids
            )
        )
        if ambiguous_intents:
            blocking.append("ambiguous_prior_submission")

        pending_buy: dict[str, Decimal] = {}
        pending_sell: dict[str, Decimal] = {}
        classified_orders: list[ClassifiedOrder] = []
        try:
            pending_order_by_id: dict[str, BrokerOrder] = {}
            for order in observed_orders:
                order_id, _, _, _ = _order_immutable_identity(order)
                if normalized_order_status(order) in TERMINAL_ORDER_STATUSES:
                    continue
                previous = pending_order_by_id.get(order_id)
                if previous is not None and not _same_order_identity(previous, order):
                    raise ReconciliationError("conflicting broker order identity")
                if previous is None or remaining_order_quantity(order) > remaining_order_quantity(previous):
                    # When two read-only endpoints disagree on fill progress,
                    # reserve the larger remainder until a later reconciliation.
                    pending_order_by_id[order_id] = order

            for order in pending_order_by_id.values():
                order_id = str(getattr(order, "order_id", "") or "").strip()
                client_id = str(getattr(order, "client_order_id", "") or "").strip()
                symbol = str(getattr(order, "symbol", "") or "").strip().upper()
                side = _enum_text(getattr(order, "side", ""))
                status = normalized_order_status(order)
                if not order_id or not symbol or side not in {"buy", "sell"} or not status:
                    raise ReconciliationError("open order is missing required identity fields")
                qty = positive_decimal(getattr(order, "qty", None), field_name="order quantity")
                filled = to_decimal(getattr(order, "filled_qty", 0) or 0)
                remaining = remaining_order_quantity(order)
                target = pending_buy if side == "buy" else pending_sell
                target[symbol] = target.get(symbol, Decimal("0")) + remaining
                intent = intents_by_client.get(client_id)
                managed = bool(
                    client_id.startswith(CLIENT_ORDER_ID_PREFIX)
                    and intent is not None
                    and _intent_matches_order(
                        intent,
                        order,
                        fingerprint,
                        environment,
                    )
                )
                classified_orders.append(
                    ClassifiedOrder(
                        order_id=order_id,
                        client_order_id=client_id or "missing",
                        symbol=symbol,
                        side=side,
                        status=status,
                        qty=qty,
                        filled_qty=filled,
                        remaining_qty=remaining,
                        managed=managed,
                        reason="matched_durable_intent" if managed else "no_confident_intent_link",
                    )
                )
        except ReconciliationError:
            return self._persist(
                ReconciliationReport(
                    checked_at_utc=checked_at,
                    environment=environment,
                    account_id_masked=mask_identifier(actual_account_id),
                    account_status=account_status,
                    blocking_reasons=("malformed_open_or_pending_order_response",),
                    result=BLOCKED_RECONCILIATION_FAILURE,
                )
            )

        if any(not item.managed for item in classified_orders):
            blocking.append("unmanaged_open_order")

        managed_net: dict[str, Decimal] = {}
        for intent in intents:
            if str(intent.get("account_fingerprint", "")) != fingerprint:
                continue
            try:
                filled = to_decimal(intent.get("filled_qty", "0"), field_name="intent filled quantity")
            except ReconciliationError:
                mismatches.append(f"malformed_intent_fill:{intent.get('intent_id', '')}")
                continue
            symbol = str(intent.get("symbol", "")).strip().upper()
            side = str(intent.get("side", "")).strip().lower()
            if not symbol or side not in {"buy", "sell"}:
                mismatches.append(f"malformed_intent_identity:{intent.get('intent_id', '')}")
                continue
            managed_net[symbol] = managed_net.get(symbol, Decimal("0")) + (
                filled if side == "buy" else -filled
            )

        classified_positions: list[ClassifiedPosition] = []
        broker_position_map: dict[str, Decimal] = {}
        try:
            for position in broker_positions_raw:
                symbol = str(getattr(position, "symbol", "") or "").strip().upper()
                qty = to_decimal(getattr(position, "qty", None), field_name="position quantity")
                if not symbol or qty == 0:
                    raise ReconciliationError("broker position is missing symbol or has zero quantity")
                if symbol in broker_position_map:
                    raise ReconciliationError("broker returned duplicate positions for one symbol")
                broker_position_map[symbol] = qty
                local_qty = local_positions.get(symbol)
                intent_qty = managed_net.get(symbol, Decimal("0"))
                local_matches = local_qty == qty
                intent_matches = intent_qty == qty
                managed = local_matches and intent_matches
                reasons: list[str] = []
                if not local_matches:
                    reasons.append("local_snapshot_quantity_mismatch")
                    mismatches.append(
                        f"local_broker_position_mismatch:{symbol}:local={decimal_text(local_qty or Decimal('0'))}:broker={decimal_text(qty)}"
                    )
                if not intent_matches:
                    reasons.append("managed_fill_lineage_mismatch")
                classified_positions.append(
                    ClassifiedPosition(
                        symbol=symbol,
                        qty=qty,
                        managed=managed,
                        reason="matched_local_state_and_managed_fills"
                        if managed
                        else "+".join(reasons),
                    )
                )
        except ReconciliationError:
            return self._persist(
                ReconciliationReport(
                    checked_at_utc=checked_at,
                    environment=environment,
                    account_id_masked=mask_identifier(actual_account_id),
                    account_status=account_status,
                    blocking_reasons=("malformed_position_response",),
                    result=BLOCKED_RECONCILIATION_FAILURE,
                )
            )

        for symbol, local_qty in local_positions.items():
            if broker_position_map.get(symbol, Decimal("0")) != local_qty:
                detail = f"local_position_not_confirmed:{symbol}"
                if detail not in mismatches:
                    mismatches.append(detail)
        for symbol, expected_qty in managed_net.items():
            if expected_qty != 0 and broker_position_map.get(symbol, Decimal("0")) != expected_qty:
                mismatches.append(f"managed_fill_position_mismatch:{symbol}")

        if any(not item.managed for item in classified_positions):
            blocking.append("unmanaged_broker_position")
        if mismatches:
            blocking.append("local_broker_state_mismatch")

        blocking = list(dict.fromkeys(blocking))
        result = SAFE_FOR_PAPER_AUTOMATION
        ready = not blocking
        if not ready:
            if environment != "paper":
                result = BLOCKED_LIVE_CONFIGURATION
            elif any(not item.managed for item in classified_positions):
                result = BLOCKED_UNMANAGED_POSITION
            elif any(not item.managed for item in classified_orders):
                result = BLOCKED_UNMANAGED_ORDER
            elif mismatches or any(
                reason.startswith("broker_account")
                or reason.startswith("expected_account")
                or reason.startswith("unbound_legacy")
                or reason.startswith("runtime_state_scope")
                for reason in blocking
            ):
                result = BLOCKED_STATE_MISMATCH
            elif ambiguous_intents:
                result = BLOCKED_AMBIGUOUS_SUBMISSION
            else:
                result = BLOCKED_RECONCILIATION_FAILURE

        report = ReconciliationReport(
            checked_at_utc=checked_at,
            environment=environment,
            account_id_masked=mask_identifier(actual_account_id),
            account_status=account_status,
            account_fingerprint=fingerprint,
            reconciliation_succeeded=True,
            broker_positions=tuple(classified_positions),
            open_orders=tuple(classified_orders),
            local_positions=tuple(sorted(local_positions.items())),
            pending_buy_qty=tuple(sorted(pending_buy.items())),
            pending_sell_qty=tuple(sorted(pending_sell.items())),
            state_mismatches=tuple(dict.fromkeys(mismatches)),
            ambiguous_intents=ambiguous_intents,
            blocking_reasons=tuple(blocking),
            result=result,
            ready=ready,
        )
        return self._persist(report)


class SubmissionGate:
    """The sole order-capable gateway used by Automation V1."""

    def __init__(
        self,
        *,
        broker: BrokerClient,
        storage: BotStorage,
        reconciler: SafetyReconciler,
        submission_enabled: bool = False,
        lock_checker: Callable[[], bool] | None = None,
    ) -> None:
        self.broker = broker
        self.storage = storage
        self.reconciler = reconciler
        self.submission_enabled = submission_enabled is True
        self._lock_checker = lock_checker
        self._mutex = _PROCESS_SUBMISSION_MUTEX

    def _blocked(self, request: OrderRequest, reason: str) -> SubmissionOutcome:
        try:
            requested = positive_decimal(request.qty)
        except ReconciliationError:
            requested = Decimal("0")
        return SubmissionOutcome(False, reason, requested)

    def submit(self, request: OrderRequest) -> SubmissionOutcome:
        with self._mutex:
            if not self.submission_enabled:
                return self._blocked(request, "order_submission_disabled")
            try:
                requested = positive_decimal(request.qty)
            except ReconciliationError as exc:
                return self._blocked(request, str(exc))
            symbol = str(request.symbol).strip().upper()
            side = str(request.side).strip().lower()
            decision_id = str(request.decision_id).strip()
            if not symbol or side not in {"buy", "sell"} or not decision_id:
                return self._blocked(request, "invalid_order_identity")

            if self._lock_checker is not None:
                try:
                    lock_held = bool(self._lock_checker())
                except Exception:
                    lock_held = False
            else:
                try:
                    from tradeos.execution_lock import current_process_holds_lock

                    lock_held = current_process_holds_lock()
                except Exception:
                    lock_held = False
            if not lock_held:
                return self._blocked(request, "trading_lock_not_held")

            report = self.reconciler.reconcile()
            if not report.ready:
                return self._blocked(request, f"reconciliation_blocked:{report.result}")

            broker_qty = report.position_qty(symbol)
            final_qty = requested
            if request.intent_kind in {LONG_EXIT_INTENT, FORCED_LONG_EXIT_INTENT}:
                if side != "sell":
                    return self._blocked(request, "long_exit_requires_sell_side")
                if broker_qty <= 0:
                    return self._blocked(request, "ordinary_sell_requires_broker_long")
                if not report.position_is_managed(symbol):
                    return self._blocked(request, "ordinary_sell_requires_managed_position")
                available = available_to_sell(broker_qty, report.pending_sell_for(symbol))
                if available <= 0:
                    return self._blocked(request, "no_safely_available_long_quantity")
                final_qty = min(requested, available)
            elif request.intent_kind == SHORT_COVER_INTENT:
                if side != "buy" or not request.explicit_cover:
                    return self._blocked(request, "cover_requires_explicit_cover_action")
                if broker_qty >= 0:
                    return self._blocked(request, "cover_requires_broker_short")
                if not report.position_is_managed(symbol):
                    return self._blocked(request, "cover_requires_managed_short")
                available = available_to_cover(broker_qty, report.pending_buy_for(symbol))
                if available <= 0:
                    return self._blocked(request, "no_safely_available_short_quantity")
                final_qty = min(requested, available)
            elif request.intent_kind == ENTRY_INTENT:
                if side != "buy":
                    return self._blocked(request, "entry_requires_buy_side")
                if broker_qty != 0:
                    return self._blocked(request, "entry_requires_flat_broker_position")
                if report.pending_buy_for(symbol) > 0 or report.pending_sell_for(symbol) > 0:
                    return self._blocked(request, "entry_requires_no_pending_symbol_orders")
            else:
                return self._blocked(request, "unsupported_intent_kind")

            intent_id, client_order_id = build_intent_identity(
                account_fingerprint_value=report.account_fingerprint,
                decision_id=decision_id,
                symbol=symbol,
                side=side,
                intent_kind=request.intent_kind,
            )
            now = datetime.now(timezone.utc).isoformat()
            try:
                intent, created = self.storage.reserve_order_intent(
                    intent_id=intent_id,
                    client_order_id=client_order_id,
                    environment=report.environment,
                    account_fingerprint=report.account_fingerprint,
                    decision_ts=decision_id,
                    symbol=symbol,
                    side=side,
                    intent_kind=request.intent_kind,
                    requested_qty=decimal_text(requested),
                    final_qty=decimal_text(final_qty),
                    created_at_utc=now,
                )
            except Exception:
                return SubmissionOutcome(
                    False,
                    "durable_intent_reservation_failed",
                    requested,
                    final_qty,
                    intent_id,
                    client_order_id,
                )
            if not created:
                return SubmissionOutcome(
                    False,
                    f"order_intent_already_exists:{intent.get('status', 'unknown')}",
                    requested,
                    final_qty,
                    intent_id,
                    client_order_id,
                )
            if not self.storage.mark_order_intent_submitting(intent_id, updated_at_utc=now):
                return SubmissionOutcome(
                    False,
                    "order_intent_not_submit_capable",
                    requested,
                    final_qty,
                    intent_id,
                    client_order_id,
                )

            try:
                order = self.broker.submit_market_order(
                    symbol=symbol,
                    qty=final_qty,
                    side=side,
                    client_order_id=client_order_id,
                )
            except Exception as exc:
                try:
                    self.storage.mark_order_intent_ambiguous(
                        intent_id,
                        error_category=type(exc).__name__,
                        updated_at_utc=datetime.now(timezone.utc).isoformat(),
                    )
                finally:
                    raise AmbiguousSubmissionError(
                        f"submission outcome is ambiguous for {symbol}; reconciliation is required"
                    ) from exc

            response_client_id = str(getattr(order, "client_order_id", "") or "")
            response_order_id = str(getattr(order, "order_id", "") or "")
            response_symbol = str(getattr(order, "symbol", "") or "").strip().upper()
            response_side = _enum_text(getattr(order, "side", ""))
            try:
                response_status = normalized_order_status(order)
                response_qty = positive_decimal(
                    getattr(order, "qty", None), field_name="broker acceptance quantity"
                )
                response_filled_qty = to_decimal(
                    getattr(order, "filled_qty", 0) or 0,
                    field_name="broker acceptance filled quantity",
                )
                response_quantities_valid = (
                    response_qty == final_qty
                    and Decimal("0") <= response_filled_qty <= response_qty
                )
            except ReconciliationError:
                response_status = ""
                response_filled_qty = Decimal("0")
                response_quantities_valid = False
            if (
                response_client_id != client_order_id
                or not response_order_id
                or response_symbol != symbol
                or response_side != side
                or not response_status
                or not response_quantities_valid
            ):
                self.storage.mark_order_intent_ambiguous(
                    intent_id,
                    error_category="malformed_broker_acceptance",
                    updated_at_utc=datetime.now(timezone.utc).isoformat(),
                )
                raise AmbiguousSubmissionError(
                    f"broker acceptance could not be verified for {symbol}; reconciliation is required"
                )

            try:
                self.storage.mark_order_intent_accepted(
                    intent_id,
                    broker_order_id=response_order_id,
                    broker_status=response_status,
                    filled_qty=decimal_text(response_filled_qty),
                    updated_at_utc=datetime.now(timezone.utc).isoformat(),
                )
            except Exception as exc:
                raise AmbiguousSubmissionError(
                    f"broker accepted {symbol} but local acceptance persistence failed"
                ) from exc

            if response_status in TERMINAL_ORDER_STATUSES - {"filled"}:
                self.storage.reconcile_order_intent(
                    client_order_id,
                    broker_order_id=response_order_id,
                    broker_status=response_status,
                    filled_qty=decimal_text(response_filled_qty),
                    updated_at_utc=datetime.now(timezone.utc).isoformat(),
                    environment=report.environment,
                    account_fingerprint=report.account_fingerprint,
                )
                return SubmissionOutcome(
                    False,
                    f"broker_order_not_active:{response_status}",
                    requested,
                    final_qty,
                    intent_id,
                    client_order_id,
                    order,
                )

            return SubmissionOutcome(
                True,
                "submitted",
                requested,
                final_qty,
                intent_id,
                client_order_id,
                order,
            )


def load_redacted_reconciliation(path: Path = DEFAULT_RECONCILIATION_PATH) -> dict[str, Any] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def format_reconciliation(report: ReconciliationReport) -> str:
    payload = report.to_redacted_dict()
    lines = [
        f"Environment: {payload['environment']}",
        f"Account: {payload['account_id_masked']}",
        f"Account status: {payload['account_status']}",
        "Broker positions: "
        + json.dumps(payload["current_broker_positions"], sort_keys=True),
        "Open/pending orders: "
        + json.dumps(payload["current_open_pending_orders"], sort_keys=True),
        "Managed positions: " + json.dumps(payload["managed_positions"], sort_keys=True),
        "Unmanaged positions: "
        + json.dumps(payload["unmanaged_positions"], sort_keys=True),
        "Managed open orders: " + json.dumps(payload["managed_orders"], sort_keys=True),
        "Unmanaged open orders: "
        + json.dumps(payload["unmanaged_orders"], sort_keys=True),
        "Pending buys: " + json.dumps(payload["pending_buy_qty"], sort_keys=True),
        "Pending sells: " + json.dumps(payload["pending_sell_qty"], sort_keys=True),
        "Local positions: " + json.dumps(payload["local_positions"], sort_keys=True),
        "State mismatches: " + json.dumps(payload["state_mismatches"], sort_keys=True),
        "Ambiguous intents: " + json.dumps(payload["ambiguous_intents"], sort_keys=True),
        "Blocking reasons: " + (
            ", ".join(payload["blocking_reasons"]) or "none"
        ),
        str(payload["result"]),
    ]
    return "\n".join(lines)

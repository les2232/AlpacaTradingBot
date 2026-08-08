from __future__ import annotations

from dataclasses import dataclass

from tradeos.brokers.base import BrokerClient, BrokerOrder

from .decisions import OrderIntent
from .models import PositionState


@dataclass(frozen=True)
class ExecutionResult:
    intent: OrderIntent
    status: str
    order: BrokerOrder | None = None
    reason: str | None = None


class BrokerIntentExecutor:
    """Non-live clean-room adapter.

    V2 intents do not carry the durable identity required by Automation V1's
    SubmissionGate, so this adapter is deliberately incapable of reaching a
    broker submission method.
    """

    def __init__(self, broker: BrokerClient) -> None:
        self.broker = broker

    def execute_intents(
        self,
        intents: tuple[OrderIntent, ...],
        positions: dict[str, PositionState],
    ) -> tuple[ExecutionResult, ...]:
        results: list[ExecutionResult] = []
        for intent in intents:
            position = positions.get(intent.symbol)
            if intent.reduce_only:
                if intent.side == "sell" and (position is None or not position.is_long or abs(position.qty) < intent.qty):
                    results.append(
                        ExecutionResult(
                            intent=intent,
                            status="rejected",
                            reason="reduce_only_sell_without_sufficient_long",
                        )
                    )
                    continue
                if intent.side == "buy" and (position is None or not position.is_short or abs(position.qty) < intent.qty):
                    results.append(
                        ExecutionResult(
                            intent=intent,
                            status="rejected",
                            reason="reduce_only_buy_without_sufficient_short",
                        )
                    )
                    continue
            results.append(
                ExecutionResult(
                    intent=intent,
                    status="rejected",
                    reason="v2_broker_submission_disabled",
                )
            )
        return tuple(results)

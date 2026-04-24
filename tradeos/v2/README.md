# TradeOS V2

This package is the clean-room foundation for a replacement trading engine.

Design rules:

- One explicit source of truth for runtime config.
- Unknown config keys fail fast instead of being ignored.
- Positions are signed and always modeled as `long`, `short`, or `flat`.
- Strategy evaluation is pure: input state in, decision out.
- Engine output is deterministic and side-effect free: decisions, intents, and events.
- Broker submission, persistence, and dashboards should consume engine output rather than re-derive it.

Current scope:

- Strict config parsing in `config.py`
- Explicit market/account/position models in `models.py`
- Structured decisions and order intents in `decisions.py`
- Structured engine events in `events.py`
- Minimal long-only reference strategy in `strategy.py`
- Deterministic evaluation engine in `engine.py`
- Replay harness with deterministic position evolution in `replay.py`
- Broker intent adapter with reduce-only safety checks in `broker_adapter.py`

Recommended next steps:

1. Add a real position book that tracks partial fills and realized PnL.
2. Add a live orchestrator that owns scheduling, resync, and artifact persistence.
3. Add a log-to-replay translator for existing `signals.jsonl` / `execution.jsonl` style runs.
4. Port one strategy at a time into the v2 protocol and validate each with replay tests.

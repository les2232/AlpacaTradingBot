# Rebuild Starting Point

This note captures the intended starting point if we decide to restart the project architecture for maximum clarity, safety, and optimization.

## Core Idea

Do not rewrite `trading_bot.py` directly.

Start a new clean core beside the current live system, then migrate behavior into it gradually after tests prove parity.

The current live path should remain intact while the new core is built and validated.

## What "Migrate Behavior" Means

Migrating behavior means moving the bot's observable decision-making rules into a cleaner module without changing what the bot would decide for the same inputs.

It does not mean blindly moving files.

It means extracting logic such as:

- completed-bar alignment
- strategy signal generation
- stale data checks
- price collar checks
- open-order blocking
- max position limits
- daily loss kill switch behavior
- session window checks
- forced flatten rules
- buy, sell, hold, and block decisions

into a deterministic engine that can be tested and replayed.

## First New Module

The first new module should likely be:

```text
tradeos/core/decision_engine.py
```

Its responsibility should be narrow:

```python
def evaluate_decision_cycle(snapshot, config) -> DecisionResult:
    ...
```

The decision engine should not:

- call Alpaca
- write to SQLite
- render dashboard UI
- read environment variables
- manage threads
- start schedulers
- submit orders directly

It should only answer:

```text
Given the current market/account/order/position/config/session state,
what should the bot do?
```

## Target Flow

The old live runtime keeps doing external work:

```text
load config
fetch Alpaca data
read positions and orders
build snapshot
call new decision engine
submit returned orders
persist audit records
update dashboard/logging state
```

The new core owns decision logic:

```text
calculate indicators
evaluate strategy
evaluate risk guards
produce buy/sell/hold/block decisions
explain why each decision happened
```

## Why This Is The Best Starting Point

The optimized rebuild should prioritize correctness, replayability, and safety before speed.

For this project, "optimized" means:

- live and backtest use the same decision logic
- decisions are deterministic and replayable
- every order can be traced back to a decision record
- incomplete bars are never used for signals
- safety checks are testable without Alpaca
- config drift between live and research is reduced
- the dashboard reads state instead of influencing trading behavior

## Migration Rule

For any behavior moved into the new core:

1. Capture the existing behavior in a test.
2. Implement the behavior in the new core.
3. Prove the new core returns the same result for the same input.
4. Only then route the live/runtime path through the new core.

The goal is a clean rebuild without pulling the floor out from under the current bot.

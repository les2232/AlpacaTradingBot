# Automation V1 execution safety

Automation V1 is restricted to Alpaca paper trading.  Its order-capable
runtime now fails closed unless it can positively establish every submission
precondition described below.

## Confirmed COP incident path

The prior protection was a JSON/PID file whose ownership check recognized only
command lines containing both `tradeos` and `live`.  Other valid Automation V1
entrypoints could therefore classify a still-running owner as stale, remove the
metadata file, and start a second order-capable process.  Decision-cycle claims
were keyed by bar timestamp, so two overlapping processes evaluating different
bars were each permitted to proceed.

The April 21 lifecycle/order logs show two such sessions evaluating COP SELL
decisions for the 17:15 and 17:30 UTC bars.  Each submitted the full locally
observed eight-share quantity.  The first sell filled before the second request
arrived, so the second full sell opened the unintended short.  That path had no
cross-process kernel lock, pending-sell reservation, deterministic client order
ID, or durable pre-submission intent.  Its broker-refresh exception path also
fell back to stale local state.  The duplicated BAC and C buys in the same
window confirm that the defect was systemic session overlap, not a COP strategy
rule.

## Read-only reconciliation

Set `ALPACA_EXPECTED_ACCOUNT_ID` in `.env` to the full Alpaca paper account ID
or account number that the operator has independently verified.  The value is
used only for an exact comparison and is never written to reconciliation
artifacts or logs.  Do not guess this value and do not use first-contact
auto-binding for an unknown account.

Run:

```powershell
python -m tradeos reconcile
```

This command does not acquire the trading lock and does not submit, cancel,
replace, or modify broker orders.  It reads the configured account, positions,
open/pending orders, recent orders, and local durable state.  Output and the
dashboard artifact contain only a masked account identifier.

The command prints one final result code.  Automation is allowed only for
`SAFE_FOR_PAPER_AUTOMATION`.  All other results are blocking, including:

- `BLOCKED_UNMANAGED_POSITION`
- `BLOCKED_UNMANAGED_ORDER`
- `BLOCKED_STATE_MISMATCH`
- `BLOCKED_LIVE_CONFIGURATION`
- `BLOCKED_RECONCILIATION_FAILURE`
- `BLOCKED_AMBIGUOUS_SUBMISSION`

An API error, malformed response, unknown environment/account, missing expected
account binding, unmanaged exposure, stale local position, or uncertain prior
submission therefore remains blocked.

## Instance authority

`.automation_v1.lock` is a persistent audit file.  Its presence does not mean
the bot is running.  Exclusion comes from a kernel file lock held on an open
handle for the entire order-capable process lifetime.  The operating system
releases that lock after a crash.  A new owner may replace stale metadata only
after it has acquired the free kernel lock, and records the recovery in the
metadata.  Never delete the file to force a takeover.

Read-only reconciliation and dashboard monitoring do not acquire this lock.
Every submission gateway call proves that the current process owns it.

## Durable order intents

Before any broker call, Automation V1 writes an immutable logical intent to
SQLite and assigns a deterministic `av1-...` client order ID.  The intent is
marked `submitting` before the request leaves the process and `accepted` only
after a matching broker response is durably recorded.  Any exception after the
submitting transition is ambiguous and is never retried automatically.

On restart, reconciliation looks up the deterministic client order ID.  A
matching broker order resolves local uncertainty without creating another
order.  An absent or unqueryable result remains blocked for operator review.

## Sell and cover rules

Ordinary exits use current broker state, never a contradictory local quantity:

```text
available_to_sell = max(broker_long_qty - pending_open_sell_remaining, 0)
```

The requested sell is capped to that value.  Flat/short positions, stale broker
data, unmanaged positions, or zero availability block the order.  Partial fills
reserve only the open order's remaining quantity.

Short covering is a distinct explicitly authorized method:

```text
available_to_cover = max(abs(broker_short_qty) - pending_open_buy_remaining, 0)
```

It is capped to that value and is never called by EOD or kill-switch automatic
flattening.  Unmanaged shorts require a separate human-directed recovery action;
there is no automatic liquidation or gate bypass.

## Manual paper restart checklist

1. Confirm `.env` says `ALPACA_PAPER=true`.
2. Independently verify the intended paper account and configure its exact ID
   as `ALPACA_EXPECTED_ACCOUNT_ID` without copying it into logs or tickets.
3. Confirm no older Automation V1 process is expected to be order-capable.
4. Run `python -m tradeos reconcile`.
5. Review the masked account suffix, account status, broker positions, pending
   quantities, managed/unmanaged classifications, mismatches, and blockers.
6. Continue only when the final line is exactly
   `SAFE_FOR_PAPER_AUTOMATION` and the reconciliation is current.
7. Start one paper runtime with `python -m tradeos paper`.
8. Confirm the dashboard shows the execution lock held, a current successful
   reconciliation, no unmanaged exposure, and readiness `ready`.
9. If startup or a submission is ambiguous, stop and reconcile; do not retry.

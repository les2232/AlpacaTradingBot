"""
avatar.py
---------
Rule-based trade narration generator for the Trader Thoughts panel.

Produces short, deterministic, analytical sentences describing trade events.
No external API calls or AI dependencies required.

Usage
-----
    from avatar import generate_trade_narration

    narration = generate_trade_narration(event_dict)
    # -> "Entered AMD. Regime bullish, volatility in range."
"""

from __future__ import annotations


# ---------------------------------------------------------------------------
# Lookup tables — deterministic labels for known reason codes
# ---------------------------------------------------------------------------

# risk.check → block_reason values emitted by BotLogger.risk_check()
_BLOCK_REASON_LABELS: dict[str, str] = {
    "max_positions": "Position limit reached.",
    "kill_switch": "Kill switch is active.",
    "daily_loss_limit": "Daily loss limit reached.",
    "position_size": "Trade size exceeds limit.",
    "price_deviation": "Live price drifted from signal.",
    "live_price_stale": "Live price data is stale.",
    "data_delay": "Market data delay too high.",
    "symbol_already_held": "Position already open.",
    "max_symbol_exposure": "Symbol exposure limit hit.",
}

# position.closed → exit_reason values emitted by BotLogger.position_closed()
_EXIT_REASON_LABELS: dict[str, str] = {
    "sma_recovery": "Mean reversion signal weakened.",
    "eod_flatten": "End-of-day flatten.",
    "kill_switch": "Kill switch triggered.",
    "time_stop": "Time stop reached.",
    "target_1x_stop": "Profit target hit.",
    "trailing_stop": "Trailing stop triggered.",
    "breakout_exit": "Breakout target reached.",
}

# signal.evaluated → rejection values emitted by BotLogger.signal()
_REJECTION_LABELS: dict[str, str] = {
    "trend_filter": "Regime filter blocked entry.",
    "atr_filter": "Volatility outside threshold.",
    "no_signal": "No actionable signal.",
    "window_closed": "Entry window is closed.",
    "holding": "Position already held.",
}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _sym(event: dict) -> str:
    """Return the symbol from an event dict, uppercased. Returns '?' if absent."""
    return str(event.get("symbol") or "?").upper()


def _narrate_signal_active(event: dict) -> str:
    """Narrate a BUY or SELL signal that cleared all filters."""
    symbol = _sym(event)
    action = str(event.get("action") or "").upper()

    context_parts: list[str] = []

    # Regime / trend filter context
    trend_filter = str(event.get("trend_filter") or "")
    above_trend = event.get("above_trend_sma")
    if trend_filter == "pass":
        if above_trend is True:
            context_parts.append("regime bullish")
        elif above_trend is False:
            context_parts.append("regime bearish")

    # Volatility context — use percentile bucket rather than raw value
    atr_percentile = event.get("atr_percentile")
    if atr_percentile is not None:
        try:
            pct = float(atr_percentile)
            if pct < 30:
                context_parts.append("low volatility")
            elif pct > 75:
                context_parts.append("elevated volatility")
        except (TypeError, ValueError):
            pass

    # ML classifier context
    ml_prob = event.get("ml_prob")
    if ml_prob is not None:
        try:
            prob_pct = int(float(ml_prob) * 100)
            context_parts.append(f"ML {prob_pct}%")
        except (TypeError, ValueError):
            pass

    if context_parts:
        raw = ", ".join(context_parts)
        # Uppercase first letter only; preserve casing of the rest (e.g. "ML 72%")
        context = raw[0].upper() + raw[1:]
    else:
        context = "Signal conditions met"

    if action == "BUY":
        return f"Entered {symbol}. {context}."
    else:
        return f"Sell signal on {symbol}. {context}."


def _narrate_signal_hold(event: dict) -> str:
    """Narrate a HOLD outcome — filtered entry or no signal."""
    symbol = _sym(event)
    rejection = str(event.get("rejection") or "").lower().strip()

    label = _REJECTION_LABELS.get(rejection)
    if label:
        return f"Skipped {symbol}. {label}"

    if event.get("window_open") is False:
        return f"Skipped {symbol}. Entry window is closed."

    return f"No signal on {symbol}."


def _narrate_risk_block(event: dict) -> str:
    """Narrate a risk check that blocked a trade."""
    symbol = _sym(event)
    block_reason = str(event.get("block_reason") or "").lower().strip()

    label = _BLOCK_REASON_LABELS.get(block_reason, "Risk check failed.")
    return f"Blocked {symbol}. {label}"


def _narrate_position_closed(event: dict) -> str:
    """Narrate a position exit, optionally appending P&L."""
    symbol = _sym(event)
    exit_reason = str(event.get("exit_reason") or "").lower().strip()

    label = _EXIT_REASON_LABELS.get(exit_reason, "Exit condition met.")

    # Append P&L if available
    pnl_str = ""
    pnl_usd = event.get("pnl_usd")
    if pnl_usd is not None:
        try:
            pnl_val = float(pnl_usd)
            sign = "+" if pnl_val >= 0 else "-"
            pnl_str = f" Closed {sign}${abs(pnl_val):.2f}."
        except (TypeError, ValueError):
            pass

    return f"Exited {symbol}. {label}{pnl_str}"


# ---------------------------------------------------------------------------
# Near-miss helpers
# ---------------------------------------------------------------------------

def _near_miss_reason(rejection: str, trend_filter: str, atr_filter: str) -> str | None:
    """
    Return the human-readable blocking reason for a near-miss HOLD signal,
    or None if the event does not represent a genuine near-miss.

    Priority: explicit rejection key > individual filter flags.
    """
    if rejection == "trend_filter" or trend_filter == "reject":
        return "Regime filter blocked entry."
    if rejection == "atr_filter" or atr_filter == "reject":
        return "ATR filter blocked entry."
    return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_near_miss_narration(event: dict) -> str:
    """
    Generate a short, deterministic narration for a near-miss signal event.

    A near-miss is a ``signal.evaluated`` HOLD event where a specific entry
    filter explicitly rejected a setup that had directional intent — i.e.
    the bot evaluated the symbol and a filter blocked the trade, rather than
    there being no signal at all.

    Parameters
    ----------
    event : dict
        A ``signal.evaluated`` JSONL event with ``action=HOLD`` and at least
        one of ``rejection``, ``trend_filter``, or ``atr_filter`` set.

    Returns
    -------
    str
        A single concise sentence. Returns a generic fallback when not enough
        context is present to build a useful narration.

    Examples
    --------
    >>> generate_near_miss_narration({
    ...     "event": "signal.evaluated", "symbol": "AAPL",
    ...     "action": "HOLD", "rejection": "atr_filter",
    ...     "atr_filter": "reject", "deviation_pct": 1.4
    ... })
    'Almost entered AAPL. ATR filter blocked entry.'

    >>> generate_near_miss_narration({
    ...     "event": "signal.evaluated", "symbol": "NVDA",
    ...     "action": "HOLD", "rejection": "trend_filter",
    ...     "trend_filter": "reject", "ml_prob": 0.63
    ... })
    'Almost entered NVDA. ML at 63%, but regime filter blocked entry.'
    """
    if not isinstance(event, dict):
        return "Near-miss data unavailable."

    symbol = _sym(event)
    rejection = str(event.get("rejection") or "").lower().strip()
    trend_filter = str(event.get("trend_filter") or "").lower().strip()
    atr_filter = str(event.get("atr_filter") or "").lower().strip()

    reason = _near_miss_reason(rejection, trend_filter, atr_filter)
    if reason is None:
        return f"{symbol} showed no qualifying near-miss setup."

    # When ML probability is available it gives the best sense of "how close"
    ml_prob = event.get("ml_prob")
    if ml_prob is not None:
        try:
            prob_pct = int(float(ml_prob) * 100)
            # Strip trailing period from reason for inline use
            reason_inline = reason.rstrip(".")
            return f"Almost entered {symbol}. ML at {prob_pct}%, but {reason_inline.lower()}."
        except (TypeError, ValueError):
            pass

    # Fall back to deviation_pct as a secondary "how close" hint
    deviation_pct = event.get("deviation_pct")
    if deviation_pct is not None:
        try:
            dev = float(deviation_pct)
            if abs(dev) >= 0.5:
                direction = "above" if dev > 0 else "below"
                reason_inline = reason.rstrip(".")
                return (
                    f"Almost entered {symbol}. "
                    f"Price {abs(dev):.1f}% {direction} SMA, but {reason_inline.lower()}."
                )
        except (TypeError, ValueError):
            pass

    return f"Almost entered {symbol}. {reason}"


def generate_trade_narration(event: dict) -> str:
    """
    Generate a short, deterministic narration sentence for a trade event.

    Parameters
    ----------
    event : dict
        A JSONL log event dict. Must contain at least an ``event`` key.

        Handled event types:
            signal.evaluated  — per-bar signal decision (BUY/SELL/HOLD)
            risk.check        — risk gate outcome (only blocks are narrated)
            position.closed   — position exit with PnL

    Returns
    -------
    str
        A single concise sentence in analytical tone. Returns a safe fallback
        when the event type is unrecognised or required fields are absent.

    Examples
    --------
    >>> generate_trade_narration({
    ...     "event": "signal.evaluated", "symbol": "AMD",
    ...     "action": "BUY", "above_trend_sma": True, "trend_filter": "pass"
    ... })
    'Entered AMD. Regime bullish.'

    >>> generate_trade_narration({
    ...     "event": "risk.check", "symbol": "AAPL",
    ...     "allowed": False, "block_reason": "max_positions"
    ... })
    'Blocked AAPL. Position limit reached.'

    >>> generate_trade_narration({
    ...     "event": "position.closed", "symbol": "NVDA",
    ...     "exit_reason": "sma_recovery", "pnl_usd": 18.5
    ... })
    'Exited NVDA. Mean reversion signal weakened. Closed +$18.50.'
    """
    if not isinstance(event, dict):
        return "Event data unavailable."

    event_type = str(event.get("event") or "").lower().strip()

    if event_type == "signal.evaluated":
        action = str(event.get("action") or "").upper()
        if action in {"BUY", "SELL"}:
            return _narrate_signal_active(event)
        return _narrate_signal_hold(event)

    if event_type == "risk.check":
        if event.get("allowed") is False:
            return _narrate_risk_block(event)
        return f"Risk check passed for {_sym(event)}."

    if event_type == "position.closed":
        return _narrate_position_closed(event)

    # Fallback for unknown or unhandled event types
    symbol = _sym(event)
    sym_part = f" for {symbol}" if symbol and symbol != "?" else ""
    return f"Event recorded{sym_part}."

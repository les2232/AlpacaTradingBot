"""
log_formatter.py
----------------
Converts raw JSONL log lines (written by botlog.BotLogger) into concise,
human-readable strings for display in the desktop app console.

Rules:
- Called per-line; returns a formatted string or the original line on failure.
- Returning an empty string suppresses the line (used for redundant events).
- No UI imports; no side effects.
- All individual formatters are wrapped by format_log_line's try/except so a
  bug in one formatter never crashes the console — it just falls back to raw.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def format_log_line(line: str) -> str:
    """
    Parse a JSONL log line and return a readable string.

    Returns:
        Formatted string       — on success
        Empty string ""        — to suppress the line (e.g. redundant events)
        Original line (raw)    — on any parse or format error
    """
    stripped = line.strip()
    if not stripped:
        return stripped

    try:
        d = json.loads(stripped)
    except (json.JSONDecodeError, ValueError):
        return stripped  # not JSON → pass through as-is

    if not isinstance(d, dict):
        return stripped

    event = d.get("event", "")
    try:
        return _dispatch(event, d)
    except Exception:
        return stripped  # formatter bug → raw fallback


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------

_FORMATTERS: dict[str, Any] = {}  # populated below after function definitions


def _dispatch(event: str, d: dict[str, Any]) -> str:
    formatter = _FORMATTERS.get(event)
    if formatter is not None:
        return formatter(d)
    return _fmt_unknown(event, d)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _bar_time(bar_ts: str) -> str:
    """ISO bar timestamp → 'HH:MM UTC'."""
    try:
        dt = datetime.fromisoformat(bar_ts).astimezone(timezone.utc)
        return dt.strftime("%H:%M")
    except Exception:
        return bar_ts[:16] if len(bar_ts) >= 16 else bar_ts


def _opt(value: Any, fmt: str = "{}") -> str:
    """Format a value only if it is not None."""
    if value is None:
        return ""
    return fmt.format(value)


def _sign(value: float) -> str:
    return f"+{value}" if value >= 0 else str(value)


# ---------------------------------------------------------------------------
# Bar events
# ---------------------------------------------------------------------------

def _fmt_bar_received(d: dict[str, Any]) -> str:
    sym    = d.get("symbol", "?")
    close  = d.get("bar_close", "?")
    vol    = d.get("bar_volume", "?")
    bar_ts = d.get("bar_ts", "")
    age    = d.get("bar_age_s", "?")
    stale  = d.get("stale", False)
    time_s = _bar_time(bar_ts)
    stale_s = "  STALE" if stale else ""
    return f"[BAR]    {sym:<6} {time_s}  close={close}  vol={vol:,}  age={age}s{stale_s}"


def _fmt_bar_stale_warning(_d: dict[str, Any]) -> str:
    # Suppressed: the STALE flag on bar.received already conveys this.
    return ""


# ---------------------------------------------------------------------------
# Signal events
# ---------------------------------------------------------------------------

def _fmt_signal(d: dict[str, Any]) -> str:
    sym       = d.get("symbol", "?")
    action    = d.get("action", "?")
    close     = d.get("bar_close")
    dev       = d.get("deviation_pct")
    rejection = d.get("rejection")
    trend     = d.get("trend_filter")
    atr       = d.get("atr_filter")
    ml_prob   = d.get("ml_prob")
    holding   = d.get("holding", False)

    parts: list[str] = [f"[SIGNAL] {sym:<6} {action:<5}"]

    if close is not None:
        parts.append(f"close={close}")
    if dev is not None:
        parts.append(f"dev={_sign(dev)}%")
    if rejection:
        parts.append(f"rejection={rejection}")
    if trend:
        parts.append(f"trend={trend}")
    if atr:
        parts.append(f"atr={atr}")
    if ml_prob is not None:
        parts.append(f"ml={ml_prob:.3f}")
    if holding:
        parts.append("(holding)")

    return "  ".join(parts)


# ---------------------------------------------------------------------------
# Execution events
# ---------------------------------------------------------------------------

def _fmt_order_submitted(d: dict[str, Any]) -> str:
    sym      = d.get("symbol", "?")
    side     = (d.get("side") or "?").upper()
    qty      = d.get("qty", "?")
    price    = d.get("live_price", "?")
    notional = d.get("notional")
    notional_s = f"  ~${notional:,.2f}" if notional is not None else ""
    return f"[ORDER]  {side} {sym:<6} qty={qty} @ ${price}{notional_s}"


def _fmt_order_filled(d: dict[str, Any]) -> str:
    sym      = d.get("symbol", "?")
    side     = (d.get("side") or "?").upper()
    qty      = d.get("fill_qty", "?")
    price    = d.get("fill_price", "?")
    slip     = d.get("slippage_bps")
    latency  = d.get("submit_to_fill_ms")
    slip_s   = f"  slip={_sign(slip)}bps" if slip is not None else ""
    lat_s    = f"  latency={int(latency)}ms" if latency is not None else ""
    return f"[FILL]   {side} {sym:<6} qty={qty} @ ${price}{slip_s}{lat_s}"


def _fmt_order_partial_fill(d: dict[str, Any]) -> str:
    sym       = d.get("symbol", "?")
    price     = d.get("fill_price", "?")
    filled    = d.get("filled_qty", "?")
    requested = d.get("requested_qty", "?")
    rate      = d.get("fill_rate")
    rate_s    = f"  rate={rate:.0%}" if rate is not None else ""
    return f"[FILL]   {sym:<6} PARTIAL  {filled}/{requested} @ ${price}{rate_s}"


def _fmt_order_rejected(d: dict[str, Any]) -> str:
    sym    = d.get("symbol", "?")
    side   = (d.get("side") or "?").upper()
    reason = d.get("reason", "unknown")
    return f"[REJECT] {side} {sym:<6}  reason={reason}"


# ---------------------------------------------------------------------------
# Position events
# ---------------------------------------------------------------------------

def _fmt_position_opened(d: dict[str, Any]) -> str:
    sym      = d.get("symbol", "?")
    entry    = d.get("entry_price", "?")
    qty      = d.get("qty", "?")
    notional = d.get("notional")
    mode     = d.get("strategy_mode", "")
    notional_s = f"  ~${notional:,.2f}" if notional is not None else ""
    mode_s     = f"  [{mode}]" if mode else ""
    return f"[OPEN]   {sym:<6} entry={entry}  qty={qty}{notional_s}{mode_s}"


def _fmt_position_closed(d: dict[str, Any]) -> str:
    sym     = d.get("symbol", "?")
    entry   = d.get("entry_price", "?")
    exit_p  = d.get("exit_price", "?")
    qty     = d.get("qty", "?")
    pnl_usd = d.get("pnl_usd")
    pnl_pct = d.get("pnl_pct")
    bars    = d.get("holding_bars")
    reason  = d.get("exit_reason", "?")

    pnl_s  = f"  pnl={_sign(pnl_usd)}" if pnl_usd is not None else ""
    pct_s  = f" ({_sign(pnl_pct)}%)" if pnl_pct is not None else ""
    bars_s = f"  bars={bars}" if bars is not None else ""

    return f"[CLOSE]  {sym:<6} entry={entry} -> exit={exit_p}  qty={qty}{pnl_s}{pct_s}{bars_s}  reason={reason}"


# ---------------------------------------------------------------------------
# Risk events
# ---------------------------------------------------------------------------

def _fmt_risk_check(d: dict[str, Any]) -> str:
    sym     = d.get("symbol", "?")
    action  = (d.get("action") or "?").upper()
    allowed = d.get("allowed", True)
    block   = d.get("block_reason")
    pos     = d.get("open_positions")
    max_pos = d.get("max_positions")
    pnl     = d.get("daily_pnl")
    kill    = d.get("kill_switch_pct_consumed")

    status_s = "BLOCKED" if not allowed else "OK"
    block_s  = f"  reason={block}" if block else ""
    pos_s    = f"  pos={pos}/{max_pos}" if pos is not None and max_pos is not None else ""
    pnl_s    = f"  pnl=${pnl:+.2f}" if pnl is not None else ""
    kill_s   = f"  kill={kill:.0f}%" if kill is not None else ""

    return f"[RISK]   {sym:<6} {action}  {status_s}{block_s}{pos_s}{pnl_s}{kill_s}"


def _fmt_kill_switch(d: dict[str, Any]) -> str:
    trigger  = d.get("trigger", False)
    pnl      = d.get("daily_pnl")
    limit    = d.get("daily_limit")
    consumed = d.get("pct_consumed")
    reason   = d.get("reason", "")

    label    = "[KILL!]" if trigger else "[KILL] "
    pnl_s    = f"  pnl=${pnl:+.2f}" if pnl is not None else ""
    limit_s  = f"  limit=${limit:.2f}" if limit is not None else ""
    cons_s   = f"  consumed={consumed:.0f}%" if consumed is not None else ""
    reason_s = f"  {reason}" if reason else ""

    return f"{label}{pnl_s}{limit_s}{cons_s}{reason_s}".strip()


# ---------------------------------------------------------------------------
# Fallback for unrecognised events
# ---------------------------------------------------------------------------

def _fmt_unknown(event: str, d: dict[str, Any]) -> str:
    """Best-effort abbreviated dump for events not in the formatter table."""
    symbol = d.get("symbol", "")
    skip = {"event", "ts", "trace", "symbol"}
    kv_pairs = [
        f"{k}={v}"
        for k, v in d.items()
        if k not in skip
    ][:5]  # cap at 5 fields to avoid long lines
    symbol_s = f" {symbol}" if symbol else ""
    fields_s = "  " + "  ".join(kv_pairs) if kv_pairs else ""
    return f"[{event}]{symbol_s}{fields_s}"


# ---------------------------------------------------------------------------
# Wire up dispatcher table (after all functions are defined)
# ---------------------------------------------------------------------------

_FORMATTERS.update({
    "bar.received":        _fmt_bar_received,
    "bar.stale_warning":   _fmt_bar_stale_warning,
    "signal.evaluated":    _fmt_signal,
    "order.submitted":     _fmt_order_submitted,
    "order.filled":        _fmt_order_filled,
    "order.partial_fill":  _fmt_order_partial_fill,
    "order.rejected":      _fmt_order_rejected,
    "position.opened":     _fmt_position_opened,
    "position.closed":     _fmt_position_closed,
    "risk.check":          _fmt_risk_check,
    "kill_switch.triggered": _fmt_kill_switch,
    "kill_switch.check":     _fmt_kill_switch,
})

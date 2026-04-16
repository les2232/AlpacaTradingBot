"""
botlog.py
---------
Structured JSON-lines logging for TradeOS.

Usage:
    from botlog import BotLogger
    blog = BotLogger()          # call once at bot startup
    blog.signal(...)
    blog.order_submitted(...)
    blog.order_filled(...)
    blog.position_closed(...)
    blog.risk_check(...)
    blog.bar_received(...)
    blog.kill_switch(...)

Each method writes one JSON line to a per-concern file under logs/<date>/.
All log files are newline-delimited JSON (JSONL) for easy grep and pandas parsing.
"""

import json
import logging
import logging.handlers
import os
from datetime import datetime, timezone
from pathlib import Path


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def _trace_id(symbol: str, decision_ts: str) -> str:
    """
    Deterministic trace ID that links all log events for one bar evaluation.
    Format: AMD_1744214700  (symbol + unix epoch of the decision bar)
    """
    # decision_ts is an ISO string like "2026-04-09T14:45:00+00:00"
    try:
        dt = datetime.fromisoformat(decision_ts)
        return f"{symbol}_{int(dt.timestamp())}"
    except Exception:
        return f"{symbol}_unknown"


def _make_file_logger(name: str, path: Path) -> logging.Logger:
    path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    logger.propagate = False
    if not logger.handlers:
        handler = logging.FileHandler(path, encoding="utf-8")
        handler.setFormatter(logging.Formatter("%(message)s"))
        logger.addHandler(handler)
    return logger


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

class BotLogger:
    """
    One instance per bot session.  Writes one JSONL file per concern per day
    under logs/<YYYY-MM-DD>/.

    Files produced:
        logs/2026-04-09/signals.jsonl      -- every bar evaluation
        logs/2026-04-09/execution.jsonl    -- orders submitted + filled
        logs/2026-04-09/positions.jsonl    -- position open / close with PnL
        logs/2026-04-09/risk.jsonl         -- kill switch + position limit checks
        logs/2026-04-09/bars.jsonl         -- bar timing / stale detection
    """

    def __init__(self, log_root: str | Path = "logs") -> None:
        date_str = datetime.now().strftime("%Y-%m-%d")
        day_dir = Path(log_root) / date_str

        self._signals  = _make_file_logger("bot.signals",  day_dir / "signals.jsonl")
        self._exec     = _make_file_logger("bot.exec",     day_dir / "execution.jsonl")
        self._pos      = _make_file_logger("bot.positions",day_dir / "positions.jsonl")
        self._risk     = _make_file_logger("bot.risk",     day_dir / "risk.jsonl")
        self._bars     = _make_file_logger("bot.bars",     day_dir / "bars.jsonl")

        # Track signal bar prices for slippage calculation at fill time
        self._signal_bar_prices: dict[str, float] = {}

    def _emit(self, logger: logging.Logger, event: str, **fields) -> None:
        fields["event"] = event
        fields["ts"] = _utcnow()
        logger.info(json.dumps(fields, default=str))

    # ------------------------------------------------------------------
    # 1. Bar received / timing
    # ------------------------------------------------------------------

    def bar_received(
        self,
        symbol: str,
        bar_close: float,
        bar_volume: float,
        bar_ts: str,          # ISO timestamp of bar open
        bar_age_s: float,     # seconds since bar close time (how late we received it)
        decision_ts: str,
    ) -> None:
        stale = bar_age_s > 60  # warn if bar arrived more than 60s late
        self._emit(self._bars, "bar.received",
            trace=_trace_id(symbol, decision_ts),
            symbol=symbol,
            bar_close=bar_close,
            bar_volume=int(bar_volume),
            bar_ts=bar_ts,
            bar_age_s=round(bar_age_s, 2),
            stale=stale,
        )
        if stale:
            self._emit(self._bars, "bar.stale_warning",
                trace=_trace_id(symbol, decision_ts),
                symbol=symbol,
                bar_age_s=round(bar_age_s, 2),
                threshold_s=60,
            )

    # ------------------------------------------------------------------
    # 2. Signal evaluation — one call per symbol per bar
    # ------------------------------------------------------------------

    def signal(
        self,
        symbol: str,
        decision_ts: str,
        bar_close: float,
        sma: float,
        trend_sma: float | None,
        atr_pct: float | None,
        atr_percentile: float | None,
        volume_ratio: float | None,
        action: str,          # "BUY" | "SELL" | "HOLD"
        holding: bool,
        # filter outcomes — pass True if filter PASSED (trade allowed), False if blocked
        trend_filter_pass: bool | None,
        atr_filter_pass: bool | None,
        window_open: bool,
        rejection: str | None,  # None if action is BUY/SELL, else the reason it's HOLD
        ml_prob: float | None = None,
    ) -> None:
        deviation_pct = round((bar_close - sma) / sma * 100, 4) if sma > 0 else None
        trace = _trace_id(symbol, decision_ts)

        # Cache bar close so we can compute slippage when fill arrives
        if action in ("BUY", "SELL"):
            self._signal_bar_prices[symbol] = bar_close

        self._emit(self._signals, "signal.evaluated",
            trace=trace,
            symbol=symbol,
            decision_ts=decision_ts,
            bar_close=bar_close,
            sma=round(sma, 4),
            deviation_pct=deviation_pct,
            trend_sma=round(trend_sma, 4) if trend_sma is not None else None,
            above_trend_sma=(bar_close >= trend_sma) if trend_sma is not None else None,
            atr_pct=round(atr_pct, 5) if atr_pct is not None else None,
            atr_percentile=round(atr_percentile, 1) if atr_percentile is not None else None,
            volume_ratio=round(volume_ratio, 3) if volume_ratio is not None else None,
            ml_prob=round(ml_prob, 4) if ml_prob is not None else None,
            trend_filter=("pass" if trend_filter_pass else "reject") if trend_filter_pass is not None else None,
            atr_filter=("pass" if atr_filter_pass else "reject") if atr_filter_pass is not None else None,
            window_open=window_open,
            action=action,
            holding=holding,
            rejection=rejection,
        )

    # ------------------------------------------------------------------
    # 3. Risk checks — one call per symbol per BUY attempt
    # ------------------------------------------------------------------

    def risk_check(
        self,
        symbol: str,
        decision_ts: str,
        action: str,
        allowed: bool,
        block_reason: str | None,
        open_positions: int,
        max_positions: int,
        daily_pnl: float,
        daily_limit: float,
        live_price: float | None = None,
        signal_price: float | None = None,
        price_deviation_bps: float | None = None,
        live_price_age_s: float | None = None,
        detail: str | None = None,
        in_entry_window: bool | None = None,
        remaining_buying_power: float | None = None,
        trade_budget: float | None = None,
        recent_order_count: int | None = None,
        max_orders_per_minute: int | None = None,
    ) -> None:
        kill_pct = round(-daily_pnl / daily_limit * 100, 1) if daily_limit > 0 else 0.0
        self._emit(self._risk, "risk.check",
            trace=_trace_id(symbol, decision_ts),
            symbol=symbol,
            action=action,
            allowed=allowed,
            block_reason=block_reason,
            open_positions=open_positions,
            max_positions=max_positions,
            daily_pnl=round(daily_pnl, 2),
            kill_switch_pct_consumed=kill_pct,
            live_price=round(live_price, 4) if live_price is not None else None,
            signal_price=round(signal_price, 4) if signal_price is not None else None,
            price_deviation_bps=round(price_deviation_bps, 1) if price_deviation_bps is not None else None,
            live_price_age_s=round(live_price_age_s, 1) if live_price_age_s is not None else None,
            detail=detail,
            in_entry_window=in_entry_window,
            remaining_buying_power=round(remaining_buying_power, 2) if remaining_buying_power is not None else None,
            trade_budget=round(trade_budget, 2) if trade_budget is not None else None,
            recent_order_count=recent_order_count,
            max_orders_per_minute=max_orders_per_minute,
        )

    def cycle_summary(
        self,
        decision_ts: str,
        execute_orders: bool,
        processed_bar: bool,
        skip_reason: str,
        buy_signals: int,
        sell_signals: int,
        hold_signals: int,
        error_signals: int,
        orders_submitted: int,
    ) -> None:
        self._emit(self._risk, "cycle.summary",
            trace=_trace_id("CYCLE", decision_ts),
            decision_ts=decision_ts,
            execute_orders=execute_orders,
            processed_bar=processed_bar,
            skip_reason=skip_reason,
            buy_signals=buy_signals,
            sell_signals=sell_signals,
            hold_signals=hold_signals,
            error_signals=error_signals,
            orders_submitted=orders_submitted,
        )

    def symbol_state_mismatch(
        self,
        *,
        current_symbols: list[str],
        persisted_symbols: list[str],
        persisted_snapshot_ts: str | None,
        action: str,
        session_id: str | None = None,
    ) -> None:
        self._emit(
            self._risk,
            "state.symbol_mismatch",
            current_symbols=current_symbols,
            persisted_symbols=persisted_symbols,
            persisted_snapshot_ts=persisted_snapshot_ts,
            action=action,
            session_id=session_id,
        )

    def execution_error(
        self,
        symbol: str,
        decision_ts: str,
        action: str,
        error: str,
    ) -> None:
        self._emit(self._risk, "execution.error",
            trace=_trace_id(symbol, decision_ts),
            symbol=symbol,
            action=action,
            error=error,
        )

    # ------------------------------------------------------------------
    # 4. Kill switch
    # ------------------------------------------------------------------

    def kill_switch(
        self,
        daily_pnl: float,
        daily_limit: float,
        trigger: bool,
        reason: str = "",
    ) -> None:
        event = "kill_switch.triggered" if trigger else "kill_switch.check"
        self._emit(self._risk, event,
            daily_pnl=round(daily_pnl, 2),
            daily_limit=daily_limit,
            pct_consumed=round(-daily_pnl / daily_limit * 100, 1) if daily_limit > 0 else 0.0,
            trigger=trigger,
            reason=reason,
        )

    # ------------------------------------------------------------------
    # 5. Order submitted
    # ------------------------------------------------------------------

    def order_submitted(
        self,
        symbol: str,
        decision_ts: str,
        side: str,        # "buy" | "sell"
        qty: int,
        live_price: float,
        order_id: str,
        signal_bar_close: float | None = None,
    ) -> None:
        signal_price = signal_bar_close
        if signal_price is None:
            signal_price = self._signal_bar_prices.get(symbol)
        self._emit(self._exec, "order.submitted",
            trace=_trace_id(symbol, decision_ts),
            symbol=symbol,
            side=side,
            qty=qty,
            live_price=round(live_price, 4),
            signal_bar_close=round(signal_price, 4) if signal_price is not None else None,
            notional=round(qty * live_price, 2),
            order_id=order_id,
        )

    # ------------------------------------------------------------------
    # 6. Order filled — call when you poll and detect a fill
    # ------------------------------------------------------------------

    def order_filled(
        self,
        symbol: str,
        decision_ts: str,
        side: str,
        fill_price: float,
        fill_qty: float,
        order_id: str,
        submitted_at: str,    # ISO string from order object
        filled_at: str,       # ISO string from order object
        signal_bar_close: float | None = None,
    ) -> None:
        signal_price = signal_bar_close
        if signal_price is None:
            signal_price = self._signal_bar_prices.get(symbol)
        slippage_bps: float | None = None
        if signal_price and signal_price > 0:
            # Adverse slippage is always positive regardless of side.
            # BUY:  fill above signal price = paid too much = positive (bad)
            # SELL: fill below signal price = received too little = positive (bad)
            if side == "buy":
                slippage_bps = round((fill_price - signal_price) / signal_price * 10_000, 2)
            else:
                slippage_bps = round((signal_price - fill_price) / signal_price * 10_000, 2)

        try:
            sub_dt = datetime.fromisoformat(submitted_at)
            fill_dt = datetime.fromisoformat(filled_at)
            latency_ms = round((fill_dt - sub_dt).total_seconds() * 1000, 0)
        except Exception:
            latency_ms = None

        self._emit(self._exec, "order.filled",
            trace=_trace_id(symbol, decision_ts),
            symbol=symbol,
            side=side,
            fill_price=round(fill_price, 4),
            fill_qty=fill_qty,
            signal_bar_close=round(signal_price, 4) if signal_price is not None else None,
            slippage_bps=slippage_bps,
            submit_to_fill_ms=latency_ms,
            order_id=order_id,
            submitted_at=submitted_at,
            filled_at=filled_at,
        )

    def order_partial_fill(
        self,
        symbol: str,
        decision_ts: str,
        fill_price: float,
        filled_qty: float,
        requested_qty: float,
        order_id: str,
    ) -> None:
        self._emit(self._exec, "order.partial_fill",
            trace=_trace_id(symbol, decision_ts),
            symbol=symbol,
            fill_price=round(fill_price, 4),
            filled_qty=filled_qty,
            requested_qty=requested_qty,
            fill_rate=round(filled_qty / requested_qty, 3) if requested_qty > 0 else None,
            order_id=order_id,
        )

    def order_rejected(
        self,
        symbol: str,
        decision_ts: str,
        side: str,
        reason: str,
        order_id: str = "",
    ) -> None:
        self._emit(self._exec, "order.rejected",
            trace=_trace_id(symbol, decision_ts),
            symbol=symbol,
            side=side,
            reason=reason,
            order_id=order_id,
        )

    # ------------------------------------------------------------------
    # 7. Position open / close with P&L, MAE, MFE
    # ------------------------------------------------------------------

    def position_opened(
        self,
        symbol: str,
        decision_ts: str,
        entry_price: float,
        qty: float,
        strategy_mode: str,
    ) -> None:
        self._emit(self._pos, "position.opened",
            trace=_trace_id(symbol, decision_ts),
            symbol=symbol,
            entry_price=round(entry_price, 4),
            qty=qty,
            notional=round(entry_price * qty, 2),
            strategy_mode=strategy_mode,
        )

    def position_closed(
        self,
        symbol: str,
        decision_ts: str,
        entry_price: float,
        exit_price: float,
        qty: float,
        holding_bars: int,
        holding_minutes: float,
        exit_reason: str,     # "sma_recovery" | "eod_flatten" | "kill_switch" | "time_stop"
        mae_pct: float | None = None,    # max adverse excursion %
        mfe_pct: float | None = None,    # max favorable excursion %
    ) -> None:
        pnl_usd = round((exit_price - entry_price) * qty, 2)
        pnl_pct = round((exit_price - entry_price) / entry_price * 100, 4) if entry_price > 0 else None
        self._emit(self._pos, "position.closed",
            trace=_trace_id(symbol, decision_ts),
            symbol=symbol,
            entry_price=round(entry_price, 4),
            exit_price=round(exit_price, 4),
            qty=qty,
            pnl_usd=pnl_usd,
            pnl_pct=pnl_pct,
            winner=pnl_usd > 0,
            holding_bars=holding_bars,
            holding_minutes=round(holding_minutes, 1),
            exit_reason=exit_reason,
            mae_pct=round(mae_pct, 4) if mae_pct is not None else None,
            mfe_pct=round(mfe_pct, 4) if mfe_pct is not None else None,
        )
        # Clear cached signal price once position is closed
        self._signal_bar_prices.pop(symbol, None)

from __future__ import annotations

import threading
from types import SimpleNamespace

from tradeos.brokers.alpaca_broker import _install_default_request_timeout, AlpacaBroker


def test_install_default_request_timeout_injects_default_timeout() -> None:
    calls: list[object] = []

    def _request(method: str, url: str, **kwargs: object) -> dict[str, object]:
        calls.append(kwargs.get("timeout"))
        return {"ok": True}

    client = SimpleNamespace(_session=SimpleNamespace(request=_request))

    _install_default_request_timeout(
        client,
        connect_timeout_seconds=5.0,
        read_timeout_seconds=20.0,
    )

    result = client._session.request("GET", "https://example.test")

    assert result == {"ok": True}
    assert calls == [(5.0, 20.0)]


def test_install_default_request_timeout_preserves_explicit_timeout() -> None:
    calls: list[object] = []

    def _request(method: str, url: str, **kwargs: object) -> dict[str, object]:
        calls.append(kwargs.get("timeout"))
        return {"ok": True}

    client = SimpleNamespace(_session=SimpleNamespace(request=_request))

    _install_default_request_timeout(
        client,
        connect_timeout_seconds=5.0,
        read_timeout_seconds=20.0,
    )

    client._session.request("GET", "https://example.test", timeout=(1.0, 2.0))

    assert calls == [(1.0, 2.0)]


def test_start_price_stream_restarts_after_stream_error(monkeypatch) -> None:
    started: list[str] = []

    class _FakeStream:
        def __init__(self, api_key: str, api_secret: str, feed: object) -> None:
            self.feed = feed

        def subscribe_trades(self, handler, *symbols: str) -> None:
            started.append(f"subscribed:{','.join(symbols)}")

        def run(self) -> None:
            started.append("run")

        def stop(self) -> None:
            started.append("stop")

    real_thread = threading.Thread

    class _ImmediateThread:
        def __init__(self, target, daemon: bool = False) -> None:
            self._target = target
            self._alive = False

        def start(self) -> None:
            self._alive = True
            try:
                self._target()
            finally:
                self._alive = False

        def is_alive(self) -> bool:
            return self._alive

    monkeypatch.setattr("tradeos.brokers.alpaca_broker.StockDataStream", _FakeStream)
    monkeypatch.setattr("tradeos.brokers.alpaca_broker.threading.Thread", _ImmediateThread)

    broker = AlpacaBroker.__new__(AlpacaBroker)
    broker._api_key = "key"
    broker._api_secret = "secret"
    broker._symbols = ["AMD"]
    broker._stream_enabled = True
    broker._stream_error = None
    broker._latest_prices = {}
    broker._latest_price_times = {}
    broker._latest_trade_times = {}
    broker._price_lock = threading.Lock()
    broker._data_stream = None
    broker._stream_thread = None
    broker._stream_restart_lock = threading.Lock()
    broker._preferred_price_stream_feed = lambda: "iex"
    broker._run_price_stream = lambda: started.append("worker")

    try:
        broker._start_price_stream()
        broker._stream_error = "WinError 121"
        broker._start_price_stream()
    finally:
        monkeypatch.setattr("tradeos.brokers.alpaca_broker.threading.Thread", real_thread)

    assert started == [
        "subscribed:AMD",
        "worker",
        "stop",
        "subscribed:AMD",
        "worker",
    ]


def test_price_feed_status_includes_configured_feed_names() -> None:
    broker = AlpacaBroker.__new__(AlpacaBroker)
    broker._stream_enabled = True
    broker._stream_error = None
    broker._latest_prices = {"AMD": 123.0}
    broker._latest_price_times = {}
    broker._latest_trade_times = {}
    broker._price_lock = threading.Lock()
    broker._stream_thread = object()
    broker._symbols = ["AMD", "MSFT"]
    broker._price_stream_feed_name = "iex"
    broker._latest_data_feed_name = "sip"

    status = broker.get_price_feed_status()

    assert "stream=iex" in status
    assert "latest=sip" in status

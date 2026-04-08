import logging
from dataclasses import dataclass, field


logger = logging.getLogger(__name__)

UNIVERSE_MODE_MANUAL = "manual"
UNIVERSE_MODE_FILTERED = "filtered"
UNIVERSE_MODE_HYBRID = "hybrid"
UNIVERSE_MODES = (
    UNIVERSE_MODE_MANUAL,
    UNIVERSE_MODE_FILTERED,
    UNIVERSE_MODE_HYBRID,
)


@dataclass(frozen=True)
class UniverseAsset:
    symbol: str
    exchange: str | None
    tradable: bool
    status: str
    asset_class: str
    avg_price: float | None
    avg_volume: float | None
    avg_dollar_volume: float | None
    is_otc: bool = False


@dataclass(frozen=True)
class UniverseConfig:
    mode: str = UNIVERSE_MODE_MANUAL
    manual_symbols: list[str] = field(default_factory=list)
    include_otc: bool = False
    exchanges: list[str] | None = None
    min_price: float | None = None
    max_price: float | None = None
    min_avg_volume: float | None = None
    min_dollar_volume: float | None = None
    max_symbols: int | None = None
    exclude_etfs: bool = False


def _normalize_mode(mode: str) -> str:
    return mode.strip().lower()


def _normalize_symbols(symbols: list[str]) -> list[str]:
    seen: set[str] = set()
    normalized: list[str] = []
    for raw_symbol in symbols:
        symbol = raw_symbol.strip().upper()
        if not symbol or symbol in seen:
            continue
        seen.add(symbol)
        normalized.append(symbol)
    return normalized


def _normalize_exchanges(exchanges: list[str] | None) -> set[str] | None:
    if exchanges is None:
        return None
    normalized = {exchange.strip().upper() for exchange in exchanges if exchange.strip()}
    return normalized or None


def _passes_filters(config: UniverseConfig, asset: UniverseAsset, exchanges: set[str] | None) -> bool:
    if not asset.tradable:
        return False
    if str(asset.status).lower() != "active":
        return False
    if not config.include_otc and asset.is_otc:
        return False
    if exchanges is not None and (asset.exchange or "").upper() not in exchanges:
        return False
    if config.exclude_etfs and str(asset.asset_class).lower() == "etf":
        return False
    if config.min_price is not None and (asset.avg_price is None or asset.avg_price < config.min_price):
        return False
    if config.max_price is not None and (asset.avg_price is None or asset.avg_price > config.max_price):
        return False
    if config.min_avg_volume is not None and (
        asset.avg_volume is None or asset.avg_volume < config.min_avg_volume
    ):
        return False
    if config.min_dollar_volume is not None and (
        asset.avg_dollar_volume is None or asset.avg_dollar_volume < config.min_dollar_volume
    ):
        return False
    return True


def build_universe(config: UniverseConfig, assets: list[UniverseAsset]) -> list[str]:
    mode = _normalize_mode(config.mode)
    if mode not in UNIVERSE_MODES:
        raise ValueError(
            f"Unsupported universe mode {config.mode!r}. "
            f"Choose from {', '.join(UNIVERSE_MODES)}."
        )

    manual_symbols = _normalize_symbols(config.manual_symbols)
    exchanges = _normalize_exchanges(config.exchanges)

    logger.info("Universe assets loaded: %d", len(assets))

    filtered_assets = [asset for asset in assets if _passes_filters(config, asset, exchanges)]
    logger.info("Universe assets after filtering: %d", len(filtered_assets))

    filtered_symbols = _normalize_symbols([asset.symbol for asset in filtered_assets])

    if mode == UNIVERSE_MODE_MANUAL:
        final_symbols = manual_symbols
    elif mode == UNIVERSE_MODE_FILTERED:
        final_symbols = filtered_symbols
    else:
        manual_set = set(manual_symbols)
        final_symbols = [symbol for symbol in filtered_symbols if symbol in manual_set]

    if config.max_symbols is not None:
        if config.max_symbols < 0:
            raise ValueError("max_symbols must be >= 0")
        final_symbols = final_symbols[: config.max_symbols]

    logger.info("Final universe symbols (%d): %s", len(final_symbols), ", ".join(final_symbols))
    return final_symbols


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    sample_assets = [
        UniverseAsset(
            symbol="AAPL",
            exchange="NASDAQ",
            tradable=True,
            status="active",
            asset_class="us_equity",
            avg_price=185.0,
            avg_volume=55_000_000,
            avg_dollar_volume=10_175_000_000,
            is_otc=False,
        ),
        UniverseAsset(
            symbol="MSFT",
            exchange="NASDAQ",
            tradable=True,
            status="active",
            asset_class="us_equity",
            avg_price=420.0,
            avg_volume=22_000_000,
            avg_dollar_volume=9_240_000_000,
            is_otc=False,
        ),
        UniverseAsset(
            symbol="SPY",
            exchange="ARCA",
            tradable=True,
            status="active",
            asset_class="etf",
            avg_price=510.0,
            avg_volume=70_000_000,
            avg_dollar_volume=35_700_000_000,
            is_otc=False,
        ),
    ]

    sample_config = UniverseConfig(
        mode=UNIVERSE_MODE_HYBRID,
        manual_symbols=["AAPL", "MSFT", "NVDA"],
        exchanges=["NASDAQ", "NYSE"],
        min_price=10.0,
        min_avg_volume=1_000_000,
        min_dollar_volume=100_000_000,
        max_symbols=10,
        exclude_etfs=True,
    )

    result = build_universe(sample_config, sample_assets)
    print(result)

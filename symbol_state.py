from __future__ import annotations

import hashlib
from typing import Iterable


def normalize_symbols(raw_symbols: Iterable[object]) -> list[str]:
    symbols: list[str] = []
    seen: set[str] = set()
    for raw_symbol in raw_symbols:
        symbol = str(raw_symbol).strip().upper()
        if not symbol or symbol in seen:
            continue
        seen.add(symbol)
        symbols.append(symbol)
    return symbols


def canonical_symbol_tuple(raw_symbols: Iterable[object]) -> tuple[str, ...]:
    return tuple(sorted(normalize_symbols(raw_symbols)))


def symbols_match(left: Iterable[object], right: Iterable[object]) -> bool:
    return canonical_symbol_tuple(left) == canonical_symbol_tuple(right)


def symbol_fingerprint(raw_symbols: Iterable[object]) -> str:
    canonical = canonical_symbol_tuple(raw_symbols)
    payload = "|".join(canonical).encode("utf-8")
    return hashlib.sha1(payload).hexdigest()[:16]


def format_symbol_list(raw_symbols: Iterable[object]) -> str:
    symbols = normalize_symbols(raw_symbols)
    return ", ".join(symbols) if symbols else "(none)"

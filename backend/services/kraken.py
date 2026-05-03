"""Kraken OHLC client helpers."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import List

from backend.services.http_client import request_json_with_retries

KRAKEN_MAP = {
    # Core majors with legacy Kraken codes
    "BTC-USD": "XXBTZUSD",
    "ETH-USD": "XETHZUSD",
    "LTC-USD": "XLTCZUSD",
    "XLM-USD": "XXLMZUSD",
    "XRP-USD": "XXRPZUSD",
    # Majors and L1s that follow the BASEUSD pattern
    "SOL-USD": "SOLUSD",
    "AVAX-USD": "AVAXUSD",
    "ADA-USD": "ADAUSD",
    "LINK-USD": "LINKUSD",
    "BCH-USD": "BCHUSD",
    "DOGE-USD": "XDGUSD",
    "HBAR-USD": "HBARUSD",
    "TRX-USD": "TRXUSD",
    "SUI-USD": "SUIUSD",
    # BTC derivatives / wrapped & stables
    "WBTC-USD": "WBTCUSD",
    "USDT-USD": "USDTZUSD",
    "USDC-USD": "USDCUSD",
    "USDE-USD": "USDEUSD",
    "USDS-USD": "USDSUSD",
    # BNB explicit attempt
    "BNB-USD": "BNBUSD",
}


@dataclass
class OhlcBar:
    """Simple OHLCV container for Kraken responses."""

    ts: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float


def resolve_kraken_pair(symbol: str) -> str | None:
    """
    Resolve a Trident symbol like 'BTC-USD' into a Kraken pair name.

    Resolution order:
    1) If the symbol is in KRAKEN_MAP, return that.
    2) Else, if symbol looks like 'BASE-USD', return f"{BASE}USD".
    3) Otherwise, return None (unsupported).
    """
    pair = KRAKEN_MAP.get(symbol)
    if pair:
        return pair

    if "-" in symbol:
        base, quote = symbol.split("-", 1)
        if quote == "USD":
            return f"{base}USD"

    return None


def _select_result_key(result: dict, preferred: str) -> str:
    """Select the correct OHLC key from Kraken's result payload."""
    if preferred in result:
        return preferred
    # Fallback: choose the first key that is not "last"
    for key in result:
        if key != "last":
            return key
    raise RuntimeError("No OHLC data present in Kraken response.")


def fetch_ohlc(base_url: str, symbol: str, interval_minutes: int, lookback: int) -> List[OhlcBar]:
    """
    Fetch up to `lookback` OHLC bars from Kraken for the given symbol and interval.
    - base_url: from settings.kraken_base, e.g. https://api.kraken.com/0/public
    - symbol: Trident symbol like 'BTC-USD'
    - interval_minutes: e.g. 60 for 1h
    - lookback: number of bars requested (we can request more from Kraken and then truncate)
    """
    pair = resolve_kraken_pair(symbol)
    if not pair:
        print(f"[crypto:backfill] Skip: cannot resolve Kraken pair for {symbol}")
        return []

    print(f"[crypto:backfill] Fetching {symbol} as Kraken pair {pair}")

    url = f"{base_url.rstrip('/')}/OHLC"
    try:
        payload, _meta = request_json_with_retries(
            url,
            params={"pair": pair, "interval": interval_minutes},
            timeout=10.0,
            max_attempts=4,
            retry_budget_seconds=20.0,
            backoff_base_seconds=0.5,
            backoff_cap_seconds=3.0,
            seed=f"kraken:{pair}:{interval_minutes}",
        )
    except Exception as exc:
        print(f"[crypto:backfill] Error: Unsupported or empty candles for {symbol} ({pair})")
        return []
    errors = payload.get("error", [])
    if errors:
        print(f"[crypto:backfill] Error: Unsupported or empty candles for {symbol} ({pair})")
        return []

    result = payload.get("result", {})
    key = _select_result_key(result, pair)
    rows = result.get(key)
    if not rows:
        print(f"[crypto:backfill] Error: Unsupported or empty candles for {symbol} ({pair})")
        return []

    bars: List[OhlcBar] = []
    for row in rows:
        ts = datetime.fromtimestamp(row[0], tz=timezone.utc)
        bars.append(
            OhlcBar(
                ts=ts,
                open=float(row[1]),
                high=float(row[2]),
                low=float(row[3]),
                close=float(row[4]),
                volume=float(row[6]),
            )
        )

    bars.sort(key=lambda b: b.ts)
    if lookback > 0:
        bars = bars[-lookback:]
    return bars

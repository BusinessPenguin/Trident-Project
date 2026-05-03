"""Crypto OHLCV backfill orchestration."""

from __future__ import annotations

from backend.config.env import get_settings
from backend.db.schema import get_connection
from backend.services.kraken import fetch_ohlc


_INTERVAL_MINUTES = {
    "1m": 1,
    "5m": 5,
    "15m": 15,
    "30m": 30,
    "1h": 60,
    "4h": 240,
    "1d": 1440,
}


def _interval_to_minutes(interval: str) -> int:
    """Map textual interval to minutes."""
    normalized = interval.lower()
    if normalized in _INTERVAL_MINUTES:
        return _INTERVAL_MINUTES[normalized]
    supported = ", ".join(sorted(_INTERVAL_MINUTES.keys()))
    raise ValueError(f"Unsupported interval: {interval}. Supported: {supported}")


def backfill_candles(symbol: str, interval: str, lookback: int) -> int:
    """
    Backfill OHLCV data for the given symbol/interval into DuckDB.
    - symbol: Trident symbol, e.g. 'BTC-USD'
    - interval: currently only '1h' is supported.
    - lookback: number of bars to keep (last N).
    Returns the number of rows inserted or updated.
    """
    settings = get_settings()
    interval_minutes = _interval_to_minutes(interval)
    print(f"[crypto:backfill] interval={interval} -> {interval_minutes}m")

    bars = fetch_ohlc(
        base_url=settings.kraken_base,
        symbol=symbol,
        interval_minutes=interval_minutes,
        lookback=lookback,
    )

    if not bars:
        return 0

    conn = get_connection(settings.database_path)
    try:
        conn.executemany(
            """
            INSERT INTO candles (symbol, interval, ts, open, high, low, close, volume)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT (symbol, interval, ts) DO UPDATE SET
                open=excluded.open,
                high=excluded.high,
                low=excluded.low,
                close=excluded.close,
                volume=excluded.volume;
            """,
            [
                (
                    symbol,
                    interval,
                    bar.ts,
                    bar.open,
                    bar.high,
                    bar.low,
                    bar.close,
                    bar.volume,
                )
                for bar in bars
            ],
        )
    finally:
        conn.close()

    return len(bars)

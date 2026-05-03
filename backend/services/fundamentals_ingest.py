"""Ingestion helpers for fundamentals data."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Dict, Optional


def ingest_fundamentals(con, symbol: str, fundamentals_dict: Dict[str, Optional[float]]) -> int:
    """
    Insert or replace latest fundamentals for a symbol and append to history.
    Returns number of rows written.
    """
    if not fundamentals_dict:
        return 0

    staged = [(k, v) for k, v in fundamentals_dict.items() if v is not None]
    if not staged:
        return 0

    ts = datetime.now(timezone.utc)
    try:
        con.execute("BEGIN TRANSACTION")
        con.execute("CREATE TEMP TABLE IF NOT EXISTS _fundamentals_stage (key VARCHAR, value DOUBLE)")
        con.execute("DELETE FROM _fundamentals_stage")
        con.executemany(
            "INSERT INTO _fundamentals_stage(key, value) VALUES (?, ?)",
            staged,
        )
        con.execute("DELETE FROM fundamentals WHERE symbol = ?", [symbol])
        con.execute(
            """
            INSERT OR REPLACE INTO fundamentals(symbol, ts, key, value)
            SELECT ?, ?, key, value
            FROM _fundamentals_stage
            """,
            [symbol, ts],
        )
        con.execute(
            """
            INSERT OR REPLACE INTO fundamentals_history(symbol, ts, key, value)
            SELECT ?, ?, key, value
            FROM _fundamentals_stage
            """,
            [symbol, ts],
        )
        con.execute("COMMIT")
    except Exception:
        try:
            con.execute("ROLLBACK")
        except Exception:
            pass
        raise
    return len(staged)

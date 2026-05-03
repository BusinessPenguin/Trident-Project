"""Smoke check for calendar country filtering (US/JP only)."""

from datetime import datetime, timezone, timedelta

from backend.config.env import get_settings
from backend.db.schema import get_connection


def main() -> None:
    settings = get_settings()
    con = get_connection(settings.database_path)
    now_utc = datetime.now(timezone.utc)
    start_ts = now_utc - timedelta(days=7)
    end_ts = now_utc + timedelta(days=14)
    rows = con.execute(
        """
        SELECT country, COUNT(*)
        FROM economic_calendar_events
        WHERE event_ts_utc >= ? AND event_ts_utc <= ?
        GROUP BY country
        ORDER BY COUNT(*) DESC, country ASC
        """,
        [start_ts, end_ts],
    ).fetchall()
    print(rows)


if __name__ == "__main__":
    main()

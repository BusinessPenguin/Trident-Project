"""Smoke check for calendar:features output keys."""

from datetime import datetime, timezone

from backend.config.env import get_settings
from backend.db.schema import get_connection, apply_all_migrations
from backend.features.calendar_features import compute_calendar_features


def main() -> None:
    settings = get_settings()
    con = get_connection(settings.database_path)
    apply_all_migrations(con)
    now_utc = datetime.now(timezone.utc)
    data = compute_calendar_features(con, now_utc)
    required = {
        "window",
        "freshness",
        "counts",
        "theme_pressure",
        "top_upcoming",
        "top_recent",
        "calendar_summary",
    }
    missing = required - set(data.keys())
    if missing:
        raise SystemExit(f"Missing keys: {sorted(missing)}")
    print("calendar_features smoke OK")


if __name__ == "__main__":
    main()

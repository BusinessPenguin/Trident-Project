"""FRED data fetcher and cache utilities."""

from __future__ import annotations

from datetime import date, datetime, time, timezone
from typing import Any, Dict, List, Tuple

import requests

FRED_OBSERVATIONS_URL = "https://api.stlouisfed.org/fred/series/observations"

SERIES_META: Dict[str, Dict[str, Any]] = {
    "WALCL": {"multiplier": 1e6, "unit": "USD", "raw_unit": "millions"},
    "WRESBAL": {"multiplier": 1e6, "unit": "USD", "raw_unit": "millions"},
    "RRPONTSYD": {"multiplier": 1e9, "unit": "USD", "raw_unit": "billions"},
}


def _series_meta(series_id: str) -> Dict[str, Any]:
    meta = SERIES_META.get(series_id)
    if meta:
        return meta
    return {"multiplier": 1.0, "unit": "USD", "raw_unit": "units"}


def _parse_value(val_str: Any) -> float | None:
    if val_str in (".", "", None):
        return None
    try:
        return float(val_str)
    except Exception:
        return None


def fred_fetch_observations(
    series_id: str,
    api_key: str,
    start_date: date,
    timeout: int = 10,
) -> List[Dict[str, Any]]:
    """Fetch FRED observations for a series starting at start_date."""
    params = {
        "series_id": series_id,
        "api_key": api_key,
        "file_type": "json",
        "observation_start": start_date.isoformat(),
    }
    resp = requests.get(FRED_OBSERVATIONS_URL, params=params, timeout=timeout)
    if resp.status_code != 200:
        raise RuntimeError(f"FRED error {resp.status_code}: {resp.text[:200]}")
    data = resp.json()
    obs_list = data.get("observations")
    if obs_list is None:
        raise RuntimeError("FRED response missing observations")

    meta = _series_meta(series_id)
    multiplier = float(meta.get("multiplier", 1.0))
    unit = str(meta.get("unit", "USD"))

    rows: List[Dict[str, Any]] = []
    for obs in obs_list:
        obs_date_str = obs.get("date")
        if not obs_date_str:
            continue
        try:
            obs_date = date.fromisoformat(obs_date_str)
        except ValueError:
            continue
        value_raw = _parse_value(obs.get("value"))
        value_norm = None if value_raw is None else value_raw * multiplier
        rows.append(
            {
                "series_id": series_id,
                "obs_date": obs_date,
                "value_raw": value_raw,
                "value_norm": value_norm,
                "unit": unit,
                "multiplier": multiplier,
            }
        )
    return rows


def _float_equal(a: float | None, b: float | None, tol: float = 1e-9) -> bool:
    if a is None and b is None:
        return True
    if a is None or b is None:
        return False
    return abs(a - b) <= tol


def upsert_fred_series(
    con,
    series_id: str,
    rows: List[Dict[str, Any]],
    fetched_at_utc: datetime,
) -> Tuple[int, int, int]:
    """Upsert rows into macro_fred_series; returns (inserted, updated, skipped)."""
    inserted = 0
    updated = 0
    skipped = 0
    fetched_at_utc = fetched_at_utc.astimezone(timezone.utc)

    for row in rows:
        obs_date = row.get("obs_date")
        value_raw = row.get("value_raw")
        value_norm = row.get("value_norm")
        unit = row.get("unit")
        multiplier = row.get("multiplier")
        if obs_date is None:
            continue
        existing = con.execute(
            """
            SELECT value_raw, value_norm, unit, multiplier
            FROM macro_fred_series
            WHERE series_id = ? AND obs_date = ?
            """,
            [series_id, obs_date],
        ).fetchone()
        ts_utc = datetime.combine(obs_date, time(0, 0), tzinfo=timezone.utc)
        if existing is None:
            con.execute(
                """
                INSERT INTO macro_fred_series (
                    series_id, obs_date, value, value_raw, value_norm, unit, multiplier,
                    series, ts_utc, fetched_at_utc, source
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    series_id,
                    obs_date,
                    value_raw,
                    value_raw,
                    value_norm,
                    unit,
                    multiplier,
                    series_id,
                    ts_utc,
                    fetched_at_utc,
                    "fred",
                ],
            )
            inserted += 1
            continue
        existing_raw, existing_norm, existing_unit, existing_multiplier = existing
        needs_update = False
        if not _float_equal(existing_raw, value_raw):
            needs_update = True
        if not _float_equal(existing_norm, value_norm):
            needs_update = True
        if (existing_unit or "") != (unit or ""):
            needs_update = True
        if not _float_equal(existing_multiplier, multiplier):
            needs_update = True
        if not needs_update:
            skipped += 1
            continue
        con.execute(
            """
            UPDATE macro_fred_series
            SET value = ?, value_raw = ?, value_norm = ?, unit = ?, multiplier = ?,
                series = ?, ts_utc = ?, fetched_at_utc = ?
            WHERE series_id = ? AND obs_date = ?
            """,
            [
                value_raw,
                value_raw,
                value_norm,
                unit,
                multiplier,
                series_id,
                ts_utc,
                fetched_at_utc,
                series_id,
                obs_date,
            ],
        )
        updated += 1
    return inserted, updated, skipped


def get_latest_fred_value(con, series_id: str) -> Dict[str, Any] | None:
    """Return latest obs_date/value_norm for the series_id from cache."""
    row = con.execute(
        """
        SELECT obs_date, value_norm, value_raw, unit, multiplier
        FROM macro_fred_series
        WHERE series_id = ?
        ORDER BY obs_date DESC
        LIMIT 1
        """,
        [series_id],
    ).fetchone()
    if not row:
        return None
    obs_date, value_norm, value_raw, unit, multiplier = row
    if value_norm is None and value_raw is not None and multiplier is not None:
        value_norm = value_raw * multiplier
    return {
        "obs_date": obs_date,
        "value_norm": value_norm,
        "value_raw": value_raw,
        "unit": unit,
        "multiplier": multiplier,
    }

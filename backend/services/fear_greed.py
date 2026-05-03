"""Fear & Greed Index fetcher (Alternative.me)."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Dict, Any

import requests


def fetch_alternative_me_fng(timeout: int = 10) -> Dict[str, Any]:
    """
    Fetch the latest Fear & Greed Index reading from Alternative.me.
    Returns a dict with ts_utc, value, label, source, fetched_at_utc.
    """
    url = "https://api.alternative.me/fng/"
    now_utc = datetime.now(timezone.utc)
    try:
        resp = requests.get(url, timeout=timeout)
    except Exception as exc:
        raise RuntimeError(f"Alternative.me request failed: {exc}") from exc
    if resp.status_code != 200:
        raise RuntimeError(f"Alternative.me request failed: status={resp.status_code}")
    try:
        payload = resp.json()
        data = payload["data"][0]
        value = int(data["value"])
        label = str(data["value_classification"])
        ts_utc = datetime.fromtimestamp(int(data["timestamp"]), tz=timezone.utc)
    except Exception as exc:
        raise RuntimeError(f"Alternative.me response parse error: {exc}") from exc

    return {
        "ts_utc": ts_utc,
        "value": value,
        "label": label,
        "source": "alternative_me",
        "fetched_at_utc": now_utc,
    }

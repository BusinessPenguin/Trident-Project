from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple

EPS = 1e-12

ORDER = [
    "price",
    "mkt_cap",
    "fdv",
    "circ_supply",
    "total_supply",
    "circ_ratio",
    "mkt_cap_rank",
    "pct_change_1h",
    "pct_change_24h",
    "pct_change_7d",
    "pct_change_30d",
    "pct_change_1y",
    "ath_price",
    "ath_change_pct",
    "atl_price",
    "atl_change_pct",
    "vol_24h",
    "vol_mcap_ratio",
    "mcap_change_1d",
    "dominance_pct",
    "liquidity_score",
    "cg_score",
    "dev_score",
    "community_score",
    "mktcap_fdv_ratio",
    "mkt_cap_change_7d",
    "mkt_cap_change_30d",
    "vol_24h_change_7d",
    "vol_mcap_ratio_z_30d",
    "fundamental_trend_score",
]


def _fmt_val(v):
    if v is None:
        return "n/a"
    if isinstance(v, int):
        return format(v, ",")
    if isinstance(v, float):
        s = format(v, "f")
        s = s.rstrip("0").rstrip(".")
        return s if s else "0"
    return v


def _safe_float(v) -> Optional[float]:
    try:
        if v is None:
            return None
        return float(v)
    except Exception:
        return None


def _as_utc(ts) -> Optional[datetime]:
    if not isinstance(ts, datetime):
        return None
    if ts.tzinfo is None:
        return ts.replace(tzinfo=timezone.utc)
    return ts.astimezone(timezone.utc)


def _latest_at_or_before(series: List[Tuple[datetime, float]], cutoff: datetime) -> Optional[float]:
    value = None
    for ts, v in series:
        if ts <= cutoff:
            value = v
        else:
            break
    return value


def _pct_change(latest: Optional[float], prior: Optional[float]) -> Optional[float]:
    if latest is None or prior is None:
        return None
    if abs(prior) <= EPS:
        return None
    return float((latest - prior) / prior)


def _clip_signed(x: float, bound: float = 1.0) -> float:
    if x > bound:
        return bound
    if x < -bound:
        return -bound
    return float(x)


def _history_derived(
    symbol: str,
    con,
    latest_ts: Optional[datetime],
    latest_data: Dict[str, float],
) -> Dict[str, Optional[float]]:
    out: Dict[str, Optional[float]] = {
        "mkt_cap_change_7d": None,
        "mkt_cap_change_30d": None,
        "vol_24h_change_7d": None,
        "vol_mcap_ratio_z_30d": None,
        "fundamental_trend_score": None,
    }
    if latest_ts is None:
        return out

    keys = ["mkt_cap", "vol_24h", "vol_mcap_ratio"]
    placeholders = ", ".join(["?"] * len(keys))
    try:
        rows = con.execute(
            f"""
            SELECT key, ts, value
            FROM fundamentals_history
            WHERE symbol = ? AND key IN ({placeholders})
            ORDER BY key ASC, ts ASC
            """,
            [symbol, *keys],
        ).fetchall()
    except Exception:
        return out

    series: Dict[str, List[Tuple[datetime, float]]] = {}
    for key, ts_raw, value_raw in rows:
        ts = _as_utc(ts_raw)
        value = _safe_float(value_raw)
        if ts is None or value is None:
            continue
        series.setdefault(str(key), []).append((ts, value))

    cutoff_7d = latest_ts - timedelta(days=7)
    cutoff_30d = latest_ts - timedelta(days=30)

    mkt_cap_now = _safe_float(latest_data.get("mkt_cap"))
    vol_24h_now = _safe_float(latest_data.get("vol_24h"))
    vmr_now = _safe_float(latest_data.get("vol_mcap_ratio"))

    mkt_cap_7d = _latest_at_or_before(series.get("mkt_cap", []), cutoff_7d)
    mkt_cap_30d = _latest_at_or_before(series.get("mkt_cap", []), cutoff_30d)
    vol_24h_7d = _latest_at_or_before(series.get("vol_24h", []), cutoff_7d)

    out["mkt_cap_change_7d"] = _pct_change(mkt_cap_now, mkt_cap_7d)
    out["mkt_cap_change_30d"] = _pct_change(mkt_cap_now, mkt_cap_30d)
    out["vol_24h_change_7d"] = _pct_change(vol_24h_now, vol_24h_7d)

    vmr_series = [
        v for ts, v in series.get("vol_mcap_ratio", []) if ts >= cutoff_30d and ts <= latest_ts
    ]
    if vmr_now is not None and len(vmr_series) >= 5:
        mean_v = sum(vmr_series) / len(vmr_series)
        var = sum((x - mean_v) ** 2 for x in vmr_series) / len(vmr_series)
        std = var ** 0.5
        if std > EPS:
            out["vol_mcap_ratio_z_30d"] = float((vmr_now - mean_v) / std)

    trend_components: List[float] = []
    if out["mkt_cap_change_7d"] is not None:
        trend_components.append(_clip_signed(float(out["mkt_cap_change_7d"]) / 0.10))
    if out["mkt_cap_change_30d"] is not None:
        trend_components.append(_clip_signed(float(out["mkt_cap_change_30d"]) / 0.20))
    if out["vol_24h_change_7d"] is not None:
        trend_components.append(_clip_signed(float(out["vol_24h_change_7d"]) / 0.20))

    mcap_fdv_ratio = _safe_float(latest_data.get("mktcap_fdv_ratio"))
    if mcap_fdv_ratio is not None:
        trend_components.append(_clip_signed((mcap_fdv_ratio - 0.5) / 0.5))

    if trend_components:
        out["fundamental_trend_score"] = float(sum(trend_components) / len(trend_components))

    return out


def get_fundamentals_features(symbol: str, con):
    rows = con.execute(
        """
        SELECT key, ts, value
        FROM fundamentals
        WHERE symbol = ?
        ORDER BY key ASC, ts DESC
        """,
        (symbol,),
    ).fetchall()

    data: Dict[str, float] = {}
    latest_ts: Optional[datetime] = None
    for key, ts_raw, value in rows:
        if key not in data:
            data[key] = value
        ts = _as_utc(ts_raw)
        if ts is not None and (latest_ts is None or ts > latest_ts):
            latest_ts = ts

    # Raw values for derived summary logic.
    mkt_cap = _safe_float(data.get("mkt_cap"))
    circ_supply = _safe_float(data.get("circ_supply"))

    if mkt_cap is not None and mkt_cap > 10_000_000_000 and circ_supply:
        summary = "large-cap asset with strong network effects"
    elif mkt_cap is not None and mkt_cap > 1_000_000_000:
        summary = "mid-cap asset with active trading dynamics"
    else:
        summary = "early stage or low-liquidity asset"

    history_derived = _history_derived(symbol, con, latest_ts, data)

    merged_data: Dict[str, object] = dict(data)
    for key, value in history_derived.items():
        if value is not None:
            merged_data[key] = value

    features: Dict[str, object] = {}
    for key in ORDER:
        if key in merged_data:
            features[key] = _fmt_val(merged_data[key])

    # Always include vol_change_1d if present in DB even though not in ORDER.
    if "vol_change_1d" in merged_data and "vol_change_1d" not in features:
        features["vol_change_1d"] = _fmt_val(merged_data["vol_change_1d"])

    features["summary"] = summary
    return features


def compute_fundamentals_features(symbol: str, con):
    return get_fundamentals_features(symbol, con)


def build_fundamentals(con, symbol: str) -> dict:
    return get_fundamentals_features(symbol, con)

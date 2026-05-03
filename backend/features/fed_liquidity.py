"""Fed liquidity feature computation from cached FRED data."""

from __future__ import annotations

from datetime import date, datetime, time, timedelta, timezone
import math
from typing import Any, Dict, Tuple


WEEKLY_SERIES = {"WALCL", "WRESBAL"}
DAILY_SERIES = {"RRPONTSYD"}

DEFAULT_WALCL_SCALE_PCT_4W = 1.0
DEFAULT_RESERVES_SCALE_USD_4W = 100_000_000_000.0
DEFAULT_RRP_SCALE_USD_4W = 200_000_000_000.0
ADAPTIVE_LOOKBACK_DAYS = 1095


def clamp01(val: float) -> float:
    try:
        return float(max(0.0, min(1.0, val)))
    except Exception:
        return 0.0


def _row_value(row: Tuple) -> float | None:
    if not row:
        return None
    value_norm = row[1]
    value_raw = row[2]
    multiplier = row[3]
    legacy_val = row[4]
    if value_norm is not None:
        return value_norm
    if value_raw is None:
        value_raw = legacy_val
    if value_raw is None:
        return None
    if multiplier is None:
        multiplier = 1.0
    return value_raw * multiplier


def _series_history(con, series_id: str, cutoff: date | None = None) -> list[tuple[date, float]]:
    rows = con.execute(
        """
        SELECT obs_date, value_norm, value_raw, multiplier, value
        FROM macro_fred_series
        WHERE series_id = ? AND (? IS NULL OR obs_date >= ?)
        ORDER BY obs_date ASC
        """,
        [series_id, cutoff, cutoff],
    ).fetchall()
    history: list[tuple[date, float]] = []
    for row in rows or []:
        obs_date = row[0]
        val = _row_value(row)
        if obs_date is None or val is None:
            continue
        history.append((obs_date, float(val)))
    return history


def _stddev(values: list[float]) -> float:
    if not values:
        return 0.0
    mean = sum(values) / len(values)
    var = sum((v - mean) ** 2 for v in values) / len(values)
    return math.sqrt(var)


def _adaptive_scale(
    con,
    series_id: str,
    latest_date: date | None,
    floor: float,
    pct_change: bool,
) -> float:
    if latest_date is None:
        return floor
    cutoff = latest_date - timedelta(days=ADAPTIVE_LOOKBACK_DAYS)
    history = _series_history(con, series_id, cutoff)
    if len(history) < 6:
        return floor
    dates = [d for d, _ in history]
    values = [v for _, v in history]
    changes: list[float] = []
    for idx, (d, v) in enumerate(history):
        target = d - timedelta(days=28)
        # find latest index with date <= target
        j = -1
        lo, hi = 0, idx
        while lo < hi:
            mid = (lo + hi) // 2
            if dates[mid] <= target:
                lo = mid + 1
            else:
                hi = mid
        j = lo - 1
        if j < 0:
            continue
        prev = values[j]
        if prev == 0:
            continue
        if pct_change:
            changes.append(100.0 * (v - prev) / prev)
        else:
            changes.append(v - prev)
    if len(changes) < 5:
        return floor
    scale = _stddev(changes)
    return max(scale, floor)


def _get_latest_obs(con, series_id: str) -> Tuple[date | None, float | None]:
    row = con.execute(
        """
        SELECT obs_date, value_norm, value_raw, multiplier, value
        FROM macro_fred_series
        WHERE series_id = ?
        ORDER BY obs_date DESC
        LIMIT 1
        """,
        [series_id],
    ).fetchone()
    if not row:
        return None, None
    return row[0], _row_value(row)


def _get_obs_at_or_before(con, series_id: str, target: date) -> Tuple[date | None, float | None]:
    row = con.execute(
        """
        SELECT obs_date, value_norm, value_raw, multiplier, value
        FROM macro_fred_series
        WHERE series_id = ? AND obs_date <= ?
        ORDER BY obs_date DESC
        LIMIT 1
        """,
        [series_id, target],
    ).fetchone()
    if not row:
        return None, None
    return row[0], _row_value(row)


def _pct_change(cur: float | None, prev: float | None) -> float | None:
    if cur is None or prev is None:
        return None
    if prev == 0:
        return None
    return 100.0 * (cur - prev) / prev


def _direction_from_change(change: float | None, pos_label: str, neg_label: str) -> str:
    if change is None:
        return "unknown"
    if change > 0:
        return pos_label
    if change < 0:
        return neg_label
    return "flat"


def _fmt_interpretation_balance(direction: str) -> str:
    if direction == "contracting":
        return "The Fed balance sheet continues to contract on a 4-week basis, consistent with quantitative tightening dynamics."
    if direction == "expanding":
        return "The Fed balance sheet continues to expand on a 4-week basis, consistent with liquidity injection dynamics."
    if direction == "flat":
        return "The Fed balance sheet is broadly flat over the last month, indicating neutral balance sheet momentum."
    return "Balance sheet data is insufficient to infer liquidity direction."


def _fmt_interpretation_reserves(direction: str) -> str:
    if direction == "declining":
        return "Bank reserve balances have declined over the last month, indicating reduced system liquidity."
    if direction == "rising":
        return "Bank reserve balances have risen over the last month, indicating improving system liquidity."
    if direction == "flat":
        return "Bank reserve balances are flat over the last month, suggesting stable system liquidity."
    return "Bank reserve data is insufficient to infer liquidity direction."


def _fmt_interpretation_rrp(direction: str) -> str:
    if direction == "rising":
        return "Rising usage of the reverse repo facility suggests liquidity absorption rather than injection."
    if direction == "declining":
        return "Declining reverse repo usage suggests liquidity is being released back into the system."
    if direction == "flat":
        return "Reverse repo usage is flat, suggesting steady short-term liquidity absorption."
    return "Reverse repo data is insufficient to infer liquidity direction."


def _age_hours(obs_date: date | None, now_utc: datetime) -> float | None:
    if obs_date is None:
        return None
    obs_dt = datetime.combine(obs_date, time(0, 0), tzinfo=timezone.utc)
    return (now_utc - obs_dt).total_seconds() / 3600.0


def _staleness_label(series_id: str, age_hours: float | None) -> str:
    if age_hours is None:
        return "very_stale"
    if series_id in WEEKLY_SERIES:
        if age_hours <= 240:
            return "acceptable"
        if age_hours <= 336:
            return "stale"
        return "very_stale"
    if series_id in DAILY_SERIES:
        if age_hours <= 72:
            return "acceptable"
        if age_hours <= 120:
            return "stale"
        return "very_stale"
    if age_hours <= 96:
        return "acceptable"
    if age_hours <= 168:
        return "stale"
    return "very_stale"


def _worst_staleness(labels: Dict[str, str]) -> str:
    order = {"acceptable": 0, "stale": 1, "very_stale": 2}
    worst = "acceptable"
    for label in labels.values():
        if order.get(label, 2) > order.get(worst, 2):
            worst = label
    return worst


def compute_fed_liquidity_features(con) -> Dict[str, Any]:
    """Compute Fed liquidity regime features from cached FRED series."""
    now_utc = datetime.now(timezone.utc)

    walcl_date, walcl_latest = _get_latest_obs(con, "WALCL")
    res_date, res_latest = _get_latest_obs(con, "WRESBAL")
    rrp_date, rrp_latest = _get_latest_obs(con, "RRPONTSYD")

    if walcl_date is None:
        return {
            "asof_date": None,
            "liquidity_regime": {"label": "Neutral", "strength": 0.0, "confidence": 0.0},
            "balance_sheet": {
                "walcl": {"latest": None, "change_1w_pct": None, "change_4w_pct": None, "direction": "unknown"},
                "interpretation": "Balance sheet data is unavailable.",
            },
            "bank_reserves": {
                "reserves": {"latest": None, "change_4w": None, "direction": "unknown"},
                "interpretation": "Bank reserve data is unavailable.",
            },
            "reverse_repo": {
                "rrp": {"latest": None, "change_4w": None, "direction": "unknown"},
                "interpretation": "Reverse repo data is unavailable.",
            },
            "liquidity_signals": {"signals_in_agreement": 0, "signals_total": 0, "agreement_ratio": None},
            "freshness": {"avg_age_hours": None, "staleness_label": "very_stale"},
            "summary": {
                "one_liner": "Fed liquidity conditions are unavailable due to missing data.",
                "policy_note": "Liquidity regime influences confidence and tail risk, not directional bias.",
            },
        }

    walcl_1w_date, walcl_1w = (None, None)
    walcl_4w_date, walcl_4w = (None, None)
    if walcl_date is not None:
        walcl_1w_date, walcl_1w = _get_obs_at_or_before(con, "WALCL", walcl_date - timedelta(days=7))
        walcl_4w_date, walcl_4w = _get_obs_at_or_before(con, "WALCL", walcl_date - timedelta(days=28))

    walcl_change_1w_pct = _pct_change(walcl_latest, walcl_1w)
    walcl_change_4w_pct = _pct_change(walcl_latest, walcl_4w)
    walcl_direction = _direction_from_change(
        walcl_change_4w_pct, pos_label="expanding", neg_label="contracting"
    )

    res_4w_date, res_4w = (None, None)
    if res_date is not None:
        res_4w_date, res_4w = _get_obs_at_or_before(con, "WRESBAL", res_date - timedelta(days=28))
    reserves_change_4w = None
    if res_latest is not None and res_4w is not None:
        reserves_change_4w = res_latest - res_4w
    reserves_direction = _direction_from_change(
        reserves_change_4w, pos_label="rising", neg_label="declining"
    )

    rrp_4w_date, rrp_4w = (None, None)
    if rrp_date is not None:
        rrp_4w_date, rrp_4w = _get_obs_at_or_before(con, "RRPONTSYD", rrp_date - timedelta(days=28))
    rrp_change_4w = None
    if rrp_latest is not None and rrp_4w is not None:
        rrp_change_4w = rrp_latest - rrp_4w
    rrp_direction = _direction_from_change(
        rrp_change_4w, pos_label="rising", neg_label="declining"
    )

    pctW_4w = None
    if walcl_latest is not None and walcl_4w is not None and walcl_4w != 0:
        pctW_4w = (walcl_latest - walcl_4w) / walcl_4w

    strength = 0.0 if pctW_4w is None else clamp01(abs(pctW_4w) / 0.01)

    def _effect_from_change(change: float | None, pos_is_tight: bool) -> str | None:
        if change is None:
            return None
        if change > 0:
            return "tightening" if pos_is_tight else "easing"
        if change < 0:
            return "easing" if pos_is_tight else "tightening"
        return None

    walcl_effect = _effect_from_change(walcl_change_4w_pct, pos_is_tight=False)
    reserves_effect = _effect_from_change(reserves_change_4w, pos_is_tight=False)
    rrp_effect = _effect_from_change(rrp_change_4w, pos_is_tight=True)

    count_tightening = sum(1 for eff in (walcl_effect, reserves_effect, rrp_effect) if eff == "tightening")
    count_easing = sum(1 for eff in (walcl_effect, reserves_effect, rrp_effect) if eff == "easing")
    signals_total = count_tightening + count_easing
    signals_in_agreement = max(count_tightening, count_easing)
    agreement_ratio = None if signals_total == 0 else signals_in_agreement / signals_total

    if count_tightening >= 2:
        regime_label = "QT-like"
    elif count_easing >= 2:
        regime_label = "QE-like"
    else:
        regime_label = "Neutral"

    ages = {
        "WALCL": _age_hours(walcl_date, now_utc),
        "WRESBAL": _age_hours(res_date, now_utc),
        "RRPONTSYD": _age_hours(rrp_date, now_utc),
    }
    staleness_by_series = {
        series_id: _staleness_label(series_id, age)
        for series_id, age in ages.items()
        if age is not None
    }
    staleness_label = _worst_staleness(staleness_by_series) if staleness_by_series else "very_stale"
    age_vals = [age for age in ages.values() if age is not None]
    avg_age_hours = None if not age_vals else sum(age_vals) / len(age_vals)

    confidence = 0.6
    if agreement_ratio is not None and agreement_ratio >= 0.66:
        confidence += 0.2
    if staleness_label == "stale":
        confidence -= 0.10
    elif staleness_label == "very_stale":
        confidence -= 0.25
    confidence = clamp01(confidence)

    latest_dates = [d for d in (walcl_date, res_date, rrp_date) if d is not None]
    asof_date = min(latest_dates).isoformat() if latest_dates else None

    summary_one_liner = (
        f"Fed liquidity conditions remain {regime_label.replace('-', ' ').lower()}, "
        f"with balance sheet {walcl_direction}, reserves {reserves_direction}, "
        f"and reverse repo {rrp_direction} reinforcing a {regime_label} regime."
    )

    return {
        "asof_date": asof_date,
        "liquidity_regime": {
            "label": regime_label,
            "strength": round(float(strength), 2),
            "confidence": round(float(confidence), 2),
        },
        "balance_sheet": {
            "walcl": {
                "latest": walcl_latest,
                "change_1w_pct": None if walcl_change_1w_pct is None else round(walcl_change_1w_pct, 2),
                "change_4w_pct": None if walcl_change_4w_pct is None else round(walcl_change_4w_pct, 2),
                "direction": walcl_direction,
            },
            "interpretation": _fmt_interpretation_balance(walcl_direction),
        },
        "bank_reserves": {
            "reserves": {
                "latest": res_latest,
                "change_4w": None if reserves_change_4w is None else round(reserves_change_4w, 2),
                "direction": reserves_direction,
            },
            "interpretation": _fmt_interpretation_reserves(reserves_direction),
        },
        "reverse_repo": {
            "rrp": {
                "latest": rrp_latest,
                "change_4w": None if rrp_change_4w is None else round(rrp_change_4w, 2),
                "direction": rrp_direction,
            },
            "interpretation": _fmt_interpretation_rrp(rrp_direction),
        },
        "liquidity_signals": {
            "signals_in_agreement": int(signals_in_agreement),
            "signals_total": int(signals_total),
            "agreement_ratio": None if agreement_ratio is None else round(float(agreement_ratio), 2),
        },
        "freshness": {
            "avg_age_hours": None if avg_age_hours is None else round(float(avg_age_hours), 2),
            "staleness_label": staleness_label,
        },
        "summary": {
            "one_liner": summary_one_liner,
            "policy_note": "Liquidity regime influences confidence and tail risk, not directional bias.",
        },
    }


def compute_fed_liquidity_features_v2(con) -> Dict[str, Any]:
    """Compute Fed liquidity features with the explainable schema for fed:features."""
    now_utc = datetime.now(timezone.utc)

    walcl_date, walcl_latest = _get_latest_obs(con, "WALCL")
    res_date, res_latest = _get_latest_obs(con, "WRESBAL")
    rrp_date, rrp_latest = _get_latest_obs(con, "RRPONTSYD")

    walcl_1w_date, walcl_1w = (None, None)
    walcl_4w_date, walcl_4w = (None, None)
    if walcl_date is not None:
        walcl_1w_date, walcl_1w = _get_obs_at_or_before(con, "WALCL", walcl_date - timedelta(days=7))
        walcl_4w_date, walcl_4w = _get_obs_at_or_before(con, "WALCL", walcl_date - timedelta(days=28))

    res_4w_date, res_4w = (None, None)
    if res_date is not None:
        res_4w_date, res_4w = _get_obs_at_or_before(con, "WRESBAL", res_date - timedelta(days=28))

    rrp_4w_date, rrp_4w = (None, None)
    if rrp_date is not None:
        rrp_4w_date, rrp_4w = _get_obs_at_or_before(con, "RRPONTSYD", rrp_date - timedelta(days=28))

    missing_required = (
        walcl_date is None
        or res_date is None
        or rrp_date is None
        or walcl_latest is None
        or res_latest is None
        or rrp_latest is None
        or walcl_4w is None
        or res_4w is None
        or rrp_4w is None
    )

    if missing_required:
        return {
            "source": "fred",
            "asof_date": None,
            "age_hours": None,
            "inputs": {
                "walcl_change_4w_pct": None,
                "walcl_change_1w_pct": None,
                "reserves_change_4w_usd": None,
                "rrp_change_4w_usd": None,
            },
            "normalization": {
                "walcl_scale_pct_4w": 1.0,
                "reserves_scale_usd_4w": 100_000_000_000.0,
                "rrp_scale_usd_4w": 200_000_000_000.0,
            },
            "subscores_tightening_0_to_1": {"walcl": None, "reserves": None, "rrp": None},
            "weights": {"walcl": 0.45, "reserves": 0.45, "rrp": 0.10},
            "strength": 0.5,
            "regime": "neutral",
            "classification_rules": {
                "qt_like_if_strength_gte": 0.55,
                "neutral_if_between": [0.45, 0.55],
                "qe_like_if_strength_lte": 0.45,
            },
            "signals_in_agreement": 0,
            "signals_total": 3,
            "agreement_ratio": 0.0,
            "liquidity_confidence": 0.3,
            "explain": "Insufficient data to determine liquidity regime from cached FRED series.",
        }

    walcl_change_1w_pct = _pct_change(walcl_latest, walcl_1w)
    walcl_change_4w_pct = _pct_change(walcl_latest, walcl_4w)
    reserves_change_4w_usd = res_latest - res_4w
    rrp_change_4w_usd = rrp_latest - rrp_4w

    walcl_scale_pct_4w = _adaptive_scale(
        con, "WALCL", walcl_date, DEFAULT_WALCL_SCALE_PCT_4W, pct_change=True
    )
    reserves_scale_usd_4w = _adaptive_scale(
        con, "WRESBAL", res_date, DEFAULT_RESERVES_SCALE_USD_4W, pct_change=False
    )
    rrp_scale_usd_4w = _adaptive_scale(
        con, "RRPONTSYD", rrp_date, DEFAULT_RRP_SCALE_USD_4W, pct_change=False
    )

    walcl_tight = clamp01((-(walcl_change_4w_pct or 0.0)) / walcl_scale_pct_4w)
    reserves_tight = clamp01((-(reserves_change_4w_usd or 0.0)) / reserves_scale_usd_4w)
    rrp_tight = clamp01((rrp_change_4w_usd or 0.0) / rrp_scale_usd_4w)

    strength = (0.45 * walcl_tight) + (0.45 * reserves_tight) + (0.10 * rrp_tight)
    strength_out = round(float(strength), 3)

    if strength >= 0.55:
        regime = "qt_like"
    elif strength <= 0.45:
        regime = "qe_like"
    else:
        regime = "neutral"

    def _vote_from_change(change: float | None, tightening_if_positive: bool) -> str:
        if change is None:
            return "neutral"
        if change > 0:
            return "tightening" if tightening_if_positive else "easing"
        if change < 0:
            return "easing" if tightening_if_positive else "tightening"
        return "neutral"

    walcl_vote = _vote_from_change(walcl_change_4w_pct, tightening_if_positive=False)
    reserves_vote = _vote_from_change(reserves_change_4w_usd, tightening_if_positive=False)
    rrp_vote = _vote_from_change(rrp_change_4w_usd, tightening_if_positive=True)
    votes = [walcl_vote, reserves_vote, rrp_vote]

    if regime == "qt_like":
        signals_in_agreement = sum(1 for v in votes if v == "tightening")
    elif regime == "qe_like":
        signals_in_agreement = sum(1 for v in votes if v == "easing")
    else:
        subs = [walcl_tight, reserves_tight, rrp_tight]
        signals_in_agreement = sum(
            1 for v, s in zip(votes, subs) if v == "neutral" or abs(s) < 0.1
        )

    agreement_ratio = round(signals_in_agreement / 3.0, 3)

    latest_dates = [d for d in (walcl_date, res_date, rrp_date) if d is not None]
    asof_date = min(latest_dates).isoformat() if latest_dates else None
    age_hours = _age_hours(min(latest_dates) if latest_dates else None, now_utc)
    age_hours_out = None if age_hours is None else round(float(age_hours), 2)

    freshness_factor = clamp01(1 - (age_hours or 0.0) / 336.0)
    liquidity_confidence = round((0.6 * agreement_ratio) + (0.4 * freshness_factor), 3)

    def _fmt_pct(val: float | None) -> str:
        if val is None:
            return "n/a"
        return f"{val:.2f}%"

    def _fmt_billions(val: float | None) -> str:
        if val is None:
            return "n/a"
        return f"{val/1e9:+.2f}B"

    walcl_dir = "contracting" if (walcl_change_4w_pct or 0.0) < 0 else "expanding"
    res_dir = "declining" if reserves_change_4w_usd < 0 else "rising"
    rrp_dir = "rising" if rrp_change_4w_usd > 0 else "declining"

    explain = (
        f"WALCL is {walcl_dir} over 4 weeks ({_fmt_pct(walcl_change_4w_pct)}), "
        f"reserves are {res_dir} ({_fmt_billions(reserves_change_4w_usd)}), "
        f"and RRP usage is {rrp_dir} ({_fmt_billions(rrp_change_4w_usd)}). "
        f"This implies a {regime} liquidity regime with strength {strength_out:.3f}, "
        "reflecting liquidity conditions rather than price direction."
    )

    return {
        "source": "fred",
        "asof_date": asof_date,
        "age_hours": age_hours_out,
        "inputs": {
            "walcl_change_4w_pct": None if walcl_change_4w_pct is None else round(walcl_change_4w_pct, 3),
            "walcl_change_1w_pct": None if walcl_change_1w_pct is None else round(walcl_change_1w_pct, 3),
            "reserves_change_4w_usd": round(reserves_change_4w_usd, 3),
            "rrp_change_4w_usd": round(rrp_change_4w_usd, 3),
        },
        "normalization": {
            "walcl_scale_pct_4w": walcl_scale_pct_4w,
            "reserves_scale_usd_4w": reserves_scale_usd_4w,
            "rrp_scale_usd_4w": rrp_scale_usd_4w,
        },
        "subscores_tightening_0_to_1": {
            "walcl": round(walcl_tight, 3),
            "reserves": round(reserves_tight, 3),
            "rrp": round(rrp_tight, 3),
        },
        "weights": {"walcl": 0.45, "reserves": 0.45, "rrp": 0.10},
        "strength": strength_out,
        "regime": regime,
        "classification_rules": {
            "qt_like_if_strength_gte": 0.55,
            "neutral_if_between": [0.45, 0.55],
            "qe_like_if_strength_lte": 0.45,
        },
        "signals_in_agreement": int(signals_in_agreement),
        "signals_total": 3,
        "agreement_ratio": agreement_ratio,
        "liquidity_confidence": liquidity_confidence,
        "explain": explain,
    }

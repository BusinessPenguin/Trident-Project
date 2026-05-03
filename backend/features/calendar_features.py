from __future__ import annotations

import json
import os
import sys
import bisect
import math
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional, Tuple

from backend.services.economic_calendar import (
    classify_category,
    classify_impact,
    macro_tags_for_event,
    get_static_event_explainer,
)


IMPACT_RANK = {"low": 1, "medium": 2, "high": 3}
DRIVER_THEMES = ["real_rates", "global_liquidity", "money_supply"]
OVERLAY_THEMES = ["risk_regime"]
THEMES = DRIVER_THEMES + OVERLAY_THEMES
LEVEL_LOOKBACK_DAYS = 1095
DEEP_HISTORY_DAYS = 3650


def _ensure_utc(ts: datetime) -> datetime:
    if ts.tzinfo is None:
        return ts.replace(tzinfo=timezone.utc)
    return ts.astimezone(timezone.utc)


def _normalize_title(title: str) -> str:
    return " ".join((title or "").lower().split())


def _parse_macro_tags(value: Optional[str]) -> List[str]:
    if not value:
        return []
    try:
        parsed = json.loads(value)
        if isinstance(parsed, list):
            return [str(x) for x in parsed if isinstance(x, str)]
    except Exception:
        return []
    return []


def _impact_rank(impact: Optional[str]) -> int:
    return IMPACT_RANK.get((impact or "").lower(), 0)


def _allowed_countries() -> set[str]:
    raw = os.getenv("TRIDENT_CALENDAR_COUNTRIES")
    if raw is None or not raw.strip():
        raw = "US,JP"
    codes = [code.strip().upper() for code in raw.split(",") if code.strip()]
    return set(codes or ["US", "JP"])


def _surprise_strength(surprise_pct: Optional[float]) -> str:
    if surprise_pct is None:
        return "none"
    mag = abs(surprise_pct)
    if mag <= 0.0025:
        return "small"
    if mag <= 0.01:
        return "medium"
    return "large"


def _mapped_theme_from_tags(tags: List[str]) -> tuple[str, float, str]:
    core = [t for t in tags if t in {"real_rates", "global_liquidity", "money_supply"}]
    if len(core) == 1:
        return core[0], 0.35, "tag_singleton"
    if len(core) > 1:
        for pref in ["real_rates", "global_liquidity", "money_supply"]:
            if pref in core:
                return pref, 0.30, "tag_multi"
    if "risk_regime" in tags:
        return "risk_regime", 0.25, "risk_regime_only"
    return "other", 0.20, "insufficient_data"


def _normalize_family(title: str) -> str:
    t = _normalize_title(title)
    for suffix in ["final", "prelim", "preliminary", "adv", "advance"]:
        t = t.replace(suffix, "")
    return " ".join(t.split())



def _normalize_history_family(title: str) -> str:
    t = _normalize_title(title)
    tokens = t.split()
    drop = {"mom", "m/m", "mm", "qoq", "q/q", "qq", "yoy", "y/y", "yy", "sa", "s.a", "s.a.", "nsa", "n.s.a", "final", "prelim", "preliminary", "adv", "advance"}
    kept = []
    for tok in tokens:
        clean = tok.strip("()[],:")
        clean_norm = clean.replace("/", "").replace(".", "")
        if clean in drop or clean_norm in drop:
            continue
        kept.append(tok)
    return " ".join(kept)
def _is_jobs_title(title: str) -> bool:
    t = _normalize_title(title)
    return any(k in t for k in ["nonfarm", "payroll", "unemployment", "jobless", "employment"])


def _is_growth_title(title: str) -> bool:
    t = _normalize_title(title)
    return any(k in t for k in ["gdp", "pmi", "retail", "industrial", "inventor", "productivity"])


def _apply_macro_fallback(
    tags: List[str],
    category: str,
    impact: str,
    title: str,
) -> List[str]:
    if tags:
        return tags
    t = _normalize_title(title)
    impact_val = (impact or "").lower()

    if category == "growth":
        mapped: List[str] = []
        if "money" in t:
            mapped.append("money_supply")
        if "liquidity" in t or "balance sheet" in t:
            mapped.append("global_liquidity")
        if mapped:
            return mapped

    if impact_val != "high":
        return []

    if category in {"jobs", "growth"} or _is_jobs_title(title) or _is_growth_title(title):
        return ["risk_regime"]

    if any(k in t for k in ["inventories", "inventory", "productivity"]):
        return ["risk_regime"]

    return []




def _history_values(
    con,
    title: str,
    country: str | None,
    cutoff_ts: datetime,
    cache: Dict[Tuple[str, str], List[float]],
) -> List[float]:
    family = _normalize_history_family(title or "")
    country_key = country or ""
    key = (family, country_key)
    if key in cache:
        return cache[key]

    loaded_key = ("__loaded__", country_key)
    if loaded_key not in cache:
        deep_cutoff = cutoff_ts
        if DEEP_HISTORY_DAYS > LEVEL_LOOKBACK_DAYS:
            deep_cutoff = cutoff_ts - timedelta(days=(DEEP_HISTORY_DAYS - LEVEL_LOOKBACK_DAYS))
        if country:
            rows = con.execute(
                """
                SELECT title, actual, previous
                FROM economic_calendar_events
                WHERE country = ? AND (actual IS NOT NULL OR previous IS NOT NULL) AND event_ts_utc >= ?
                """,
                [country, cutoff_ts],
            ).fetchall()
            archive_rows = con.execute(
                """
                SELECT title, actual, previous
                FROM economic_calendar_events_archive
                WHERE country = ? AND (actual IS NOT NULL OR previous IS NOT NULL) AND event_ts_utc >= ?
                """,
                [country, deep_cutoff],
            ).fetchall()
        else:
            rows = con.execute(
                """
                SELECT title, actual, previous
                FROM economic_calendar_events
                WHERE country IS NULL AND (actual IS NOT NULL OR previous IS NOT NULL) AND event_ts_utc >= ?
                """,
                [cutoff_ts],
            ).fetchall()
            archive_rows = con.execute(
                """
                SELECT title, actual, previous
                FROM economic_calendar_events_archive
                WHERE country IS NULL AND (actual IS NOT NULL OR previous IS NOT NULL) AND event_ts_utc >= ?
                """,
                [deep_cutoff],
            ).fetchall()
        rows = (rows or []) + (archive_rows or [])
        grouped: Dict[str, List[float]] = {}
        for row_title, actual, previous in rows or []:
            value = actual if actual is not None else previous
            if value is None:
                continue
            fam = _normalize_history_family(row_title or "")
            grouped.setdefault(fam, []).append(float(value))
        for fam, values in grouped.items():
            cache[(fam, country_key)] = values
        cache[loaded_key] = []

    return cache.get(key, [])


def _level_context(
    value: float | None,
    history: List[float],
    basis: str,
) -> Dict[str, Any] | None:
    if value is None:
        return None
    n = len(history)
    if n < 3:
        return {
            "basis": basis,
            "history_count": n,
            "percentile": None,
            "zscore": None,
            "label": "insufficient_history",
        }
    history_sorted = sorted(history)
    rank = bisect.bisect_right(history_sorted, float(value))
    percentile = (rank - 0.5) / n if n > 0 else 0.5
    percentile = max(0.0, min(1.0, percentile))
    mean = sum(history) / n
    var = sum((x - mean) ** 2 for x in history) / n
    stdev = math.sqrt(var) if var > 0 else 0.0
    zscore = 0.0 if stdev == 0 else (float(value) - mean) / stdev
    if percentile <= 0.33:
        label = "low"
    elif percentile >= 0.67:
        label = "high"
    else:
        label = "normal"
    return {
        "basis": basis,
        "history_count": n,
        "percentile": round(percentile, 2),
        "zscore": round(zscore, 2),
        "label": label,
    }


def _level_context_note(
    category: str,
    impact: str,
    level_context: Dict[str, Any],
    is_recent: bool,
    effect_hint: Optional[Dict[str, Any]] = None,
    explainer: Optional[str] = None,
) -> Optional[str]:
    if not level_context:
        return None
    impact_val = (impact or "medium").lower()
    if impact_val == "high":
        impact_phrase = "High-impact"
    elif impact_val == "low":
        impact_phrase = "Low-impact"
    else:
        impact_phrase = "Medium-impact"
    cat = (category or "other").lower()
    cat_map = {
        "monetary_policy": "policy",
        "inflation": "inflation",
        "jobs": "jobs",
        "growth": "growth",
        "liquidity": "liquidity",
        "money_supply": "money supply",
        "surveys": "survey",
    }
    cat_phrase = cat_map.get(cat, "macro")

    basis = (level_context.get("basis") or "").lower()
    if basis == "forecast":
        basis_phrase = "expected level"
    elif basis == "previous":
        basis_phrase = "prior level"
    else:
        basis_phrase = "reported level"

    label = (level_context.get("label") or "").lower()

    prefix = ""
    if explainer:
        cleaned = " ".join(str(explainer).split()).strip().rstrip(".")
        if cleaned:
            if cleaned.lower().startswith("measures "):
                prefix = f"{cleaned}. "
            else:
                prefix = f"Measures {cleaned}. "
    if label == "high":
        level_phrase = f"{basis_phrase} is high vs recent history"
    elif label == "low":
        level_phrase = f"{basis_phrase} is low vs recent history"
    elif label == "normal":
        level_phrase = f"{basis_phrase} is near normal vs recent history"
    else:
        level_phrase = "history is limited; treat level as contextual only"

    if label == "insufficient_history":
        if is_recent:
            return (
                prefix
                + f"{impact_phrase} {cat_phrase} release; history is limited, "
                "so it mainly provides background context for scenario analysis."
            )
        return (
            prefix
            + f"{impact_phrase} {cat_phrase} event; history is limited, "
            "so it mainly adds timing/tail-risk context around the release."
        )

    bias_phrase = ""
    if is_recent and effect_hint:
        bias = (effect_hint.get("bias") or "neutral").lower()
        bias_map = {
            "tightening_pressure": "suggests firmer policy pressure",
            "easing_pressure": "suggests easing pressure",
            "risk_on": "tilts risk sentiment higher",
            "risk_off": "tilts risk sentiment lower",
            "liquidity_up": "signals improving liquidity",
            "liquidity_down": "signals tightening liquidity",
            "momentum_up": "signals upside momentum",
            "momentum_down": "signals downside momentum",
        }
        bias_phrase = bias_map.get(bias, "")

    if is_recent:
        if bias_phrase:
            return (
                prefix
                + f"{impact_phrase} {cat_phrase} release; {level_phrase} and {bias_phrase}, "
                "which can influence confidence/tail-risk context."
            )
        return (
            prefix
            + f"{impact_phrase} {cat_phrase} release; {level_phrase}, "
            "so it primarily informs confidence/tail-risk context."
        )

    return (
        prefix
        + f"{impact_phrase} {cat_phrase} event; {level_phrase}, "
        "which can raise macro sensitivity around the release."
    )


def _effect_hint(
    category: str,
    title: str,
    tags: List[str],
    actual: Optional[float],
    forecast: Optional[float],
    previous: Optional[float],
    surprise_strength: str,
) -> Dict[str, Any]:
    base_conf_map = {"none": 0.2, "small": 0.4, "medium": 0.6, "large": 0.8}
    strength_conf = base_conf_map.get(surprise_strength, 0.2)
    mapped_theme, base_conf, rationale = _mapped_theme_from_tags(tags)
    cat = (category or "other").lower()

    bias = "neutral"
    has_surprise_inputs = actual is not None and forecast is not None
    surprise = actual - forecast if has_surprise_inputs else None

    if not has_surprise_inputs:
        if actual is not None and previous is not None:
            try:
                delta = float(actual) - float(previous)
            except Exception:
                delta = None
            if delta is None:
                return {
                    "bias": "neutral",
                    "confidence": round(float(min(base_conf, 0.3)), 2),
                    "mapped_theme": mapped_theme,
                    "rationale": "no_surprise_inputs",
                }
            if cat in {"inflation", "monetary_policy"}:
                if delta > 0:
                    bias = "tightening_pressure"
                    rationale = "policy_momentum_up"
                elif delta < 0:
                    bias = "easing_pressure"
                    rationale = "policy_momentum_down"
                else:
                    bias = "neutral"
                    rationale = "policy_flat"
            elif cat in {"liquidity", "money_supply"}:
                if delta > 0:
                    bias = "liquidity_up"
                    rationale = "liquidity_momentum_up"
                elif delta < 0:
                    bias = "liquidity_down"
                    rationale = "liquidity_momentum_down"
                else:
                    bias = "neutral"
                    rationale = "liquidity_flat"
            elif cat in {"jobs", "growth"}:
                if delta > 0:
                    bias = "risk_on"
                    rationale = "growth_momentum_up"
                elif delta < 0:
                    bias = "risk_off"
                    rationale = "growth_momentum_down"
                else:
                    bias = "neutral"
                    rationale = "growth_flat"
            else:
                if delta > 0:
                    bias = "momentum_up"
                    rationale = "momentum_up"
                elif delta < 0:
                    bias = "momentum_down"
                    rationale = "momentum_down"
                else:
                    bias = "neutral"
                    rationale = "momentum_flat"
            conf = min(max(base_conf, 0.25), 0.35)
            return {
                "bias": bias,
                "confidence": round(float(conf), 2),
                "mapped_theme": mapped_theme,
                "rationale": rationale,
            }
        return {
            "bias": "neutral",
            "confidence": round(float(min(base_conf, 0.3)), 2),
            "mapped_theme": mapped_theme,
            "rationale": "no_surprise_inputs",
        }

    if cat in {"inflation", "monetary_policy"}:
        if surprise is not None and surprise > 0:
            bias = "tightening_pressure"
            rationale = "policy_hotter"
        elif surprise is not None and surprise < 0:
            bias = "easing_pressure"
            rationale = "policy_cooler"
        else:
            bias = "neutral"
            rationale = "policy_flat"
    elif cat in {"liquidity", "money_supply"}:
        if surprise is not None and surprise > 0:
            bias = "liquidity_up"
            rationale = "liquidity_up"
        elif surprise is not None and surprise < 0:
            bias = "liquidity_down"
            rationale = "liquidity_down"
        else:
            bias = "neutral"
            rationale = "liquidity_flat"
    base_conf = max(base_conf, strength_conf)
    return {
        "bias": bias,
        "confidence": round(float(max(0.0, min(1.0, base_conf))), 2),
        "mapped_theme": mapped_theme,
        "rationale": rationale,
    }


def compute_calendar_features(
    con,
    now_utc: datetime,
    lookback_hours: int = 168,
    lookahead_hours: int = 168,
    min_impact: str = "medium",
    max_upcoming_items: int = 15,
    max_recent_items: int = 5,
    debug: bool = False,
) -> Dict[str, Any]:
    now_utc = _ensure_utc(now_utc)
    min_rank = IMPACT_RANK.get((min_impact or "medium").lower(), 2)
    start_ts = now_utc - timedelta(hours=lookback_hours)
    end_ts = now_utc + timedelta(hours=lookahead_hours)
    level_cache: Dict[Tuple[str, str], List[float]] = {}
    level_cutoff = now_utc - timedelta(days=LEVEL_LOOKBACK_DAYS)

    explainer_map: Dict[Tuple[str, str, str], str] = {}
    try:
        rows_expl = con.execute(
            """
            SELECT event_family, country, category, explanation
            FROM economic_calendar_explainers
            """
        ).fetchall()
        for fam, ctry, cat, expl in rows_expl or []:
            key = (fam or "", (ctry or "").upper(), cat or "")
            if expl and key not in explainer_map:
                explainer_map[key] = str(expl)
    except Exception:
        explainer_map = {}


    rows = con.execute(
        """
        SELECT
            event_ts_utc,
            country,
            title,
            category,
            impact,
            macro_tags,
            actual,
            forecast,
            previous,
            unit,
            fetched_at_utc
        FROM economic_calendar_events
        WHERE event_ts_utc >= ? AND event_ts_utc <= ?
        """,
        [start_ts, end_ts],
    ).fetchall()

    events: List[Dict[str, Any]] = []
    empty_tags_before = 0
    empty_tags_after = 0
    allowed_countries = _allowed_countries()
    for (
        event_ts,
        country,
        title,
        category,
        impact,
        macro_tags,
        actual,
        forecast,
        previous,
        unit,
        fetched_at,
    ) in rows or []:
        country_code = (country or "").upper()
        if country_code and country_code not in allowed_countries:
            continue
        normalized_title = title or ""
        derived_category = classify_category(normalized_title)
        derived_impact = classify_impact(country, normalized_title, derived_category)
        impact_val = derived_impact
        if _impact_rank(impact_val) < min_rank:
            continue
        ts = _ensure_utc(event_ts)
        fetch_ts = _ensure_utc(fetched_at) if fetched_at else None
        derived_tags = macro_tags_for_event(
            normalized_title,
            derived_category,
            impact=impact_val,
            event_ts_utc=ts,
            now_utc=now_utc,
        )
        if not derived_tags:
            empty_tags_before += 1
        tags = _apply_macro_fallback(derived_tags, derived_category, impact_val, normalized_title)
        if not tags:
            empty_tags_after += 1
        events.append(
            {
                "event_ts_utc": ts,
                "country": country,
                "title": normalized_title,
                "category": derived_category or "other",
                "impact": impact_val,
                "macro_tags": tags,
                "actual": actual,
                "forecast": forecast,
                "previous": previous,
                "unit": unit,
                "fetched_at_utc": fetch_ts,
            }
        )

    window = {
        "now_utc": now_utc.isoformat(),
        "lookback_hours": lookback_hours,
        "lookahead_hours": lookahead_hours,
        "min_impact": (min_impact or "medium").lower(),
    }

    rows_used = len(events)
    fetch_ages = []
    for ev in events:
        fetch_ts = ev.get("fetched_at_utc")
        if fetch_ts:
            age = (now_utc - fetch_ts).total_seconds() / 3600.0
            fetch_ages.append(age)
    if fetch_ages:
        latest_age = round(min(fetch_ages), 2)
        avg_age = round(sum(fetch_ages) / len(fetch_ages), 2)
    else:
        latest_age = None
        avg_age = None

    freshness = {
        "calendar_latest_fetch_age_hours": latest_age,
        "calendar_avg_fetch_age_hours": avg_age,
        "rows_used": rows_used,
    }

    by_impact = {"high": 0, "medium": 0, "low": 0}
    by_country: Dict[str, int] = {}
    for ev in events:
        impact_val = ev["impact"]
        if impact_val in by_impact:
            by_impact[impact_val] += 1
        c = ev.get("country")
        if c:
            by_country[c] = by_country.get(c, 0) + 1
    by_country_sorted = sorted(by_country.items(), key=lambda x: (-x[1], x[0]))[:5]
    by_country_top = {k: v for k, v in by_country_sorted}

    counts = {
        "rows_used": rows_used,
        "by_impact": by_impact,
        "by_country_top": by_country_top,
    }

    if debug:
        actual_count = sum(1 for ev in events if ev.get("actual") is not None)
        forecast_count = sum(1 for ev in events if ev.get("forecast") is not None)
        previous_count = sum(1 for ev in events if ev.get("previous") is not None)
        print(
            "[calendar:features][debug] empty_macro_tags "
            f"before_fallback={empty_tags_before} after_fallback={empty_tags_after}",
            file=sys.stderr,
        )
        print(
            "[calendar:features][debug] non_null actual="
            f"{actual_count} forecast={forecast_count} previous={previous_count}",
            file=sys.stderr,
        )
        samples = [
            ev for ev in events if ev.get("forecast") is not None or ev.get("previous") is not None
        ][:3]
        for ev in samples:
            print(
                "[calendar:features][debug] sample "
                f"{ev.get('title')} forecast={ev.get('forecast')} previous={ev.get('previous')}",
                file=sys.stderr,
            )

    theme_weighted = {t: 0.0 for t in THEMES}
    theme_events = {t: 0 for t in THEMES}
    impact_w = {"high": 1.0, "medium": 0.5, "low": 0.25}
    for ev in events:
        tags = ev.get("macro_tags") or []
        if not tags:
            continue
        w = impact_w.get(ev["impact"], 0.25)
        for tag in tags:
            if tag in theme_weighted:
                theme_weighted[tag] += w
                theme_events[tag] += 1

    theme_pressure: Dict[str, Any] = {}
    driver_max_weighted = (
        max(theme_weighted[t] for t in DRIVER_THEMES) if DRIVER_THEMES else 0.0
    )
    for theme in THEMES:
        if theme in DRIVER_THEMES:
            score = (
                theme_weighted[theme] / driver_max_weighted
                if driver_max_weighted > 0
                else 0.0
            )
        else:
            # risk_regime is an overlay/context signal; excluded from dominance to prevent structural bias.
            score = (
                theme_weighted[theme] / driver_max_weighted
                if driver_max_weighted > 0
                else (1.0 if theme_weighted[theme] > 0 else 0.0)
            )
        score = max(0.0, min(1.0, score))
        theme_pressure[theme] = {
            "score": round(score, 3),
            "events": theme_events[theme],
            "weighted_events": round(theme_weighted[theme], 3),
        }

    upcoming = [ev for ev in events if ev["event_ts_utc"] >= now_utc]

    theme_next_high: Dict[str, Optional[float]] = {t: None for t in THEMES}
    for ev in upcoming:
        if ev.get("impact") != "high":
            continue
        tags = ev.get("macro_tags") or []
        if not tags:
            continue
        in_hours = (ev["event_ts_utc"] - now_utc).total_seconds() / 3600.0
        if in_hours < 0 or in_hours > 72:
            continue
        for tag in tags:
            if tag in theme_next_high:
                cur = theme_next_high[tag]
                if cur is None or in_hours < cur:
                    theme_next_high[tag] = in_hours

    if DRIVER_THEMES:
        max_score = max(theme_pressure[t]["score"] for t in DRIVER_THEMES)
        candidates = [
            t
            for t in DRIVER_THEMES
            if abs(theme_pressure[t]["score"] - max_score) <= 0.02
        ]
        if len(candidates) == 1:
            dominant = candidates[0]
        else:
            def _tie_key(t: str) -> Any:
                weighted = theme_pressure[t]["weighted_events"]
                has_high = 1 if theme_next_high.get(t) is not None else 0
                next_high = theme_next_high.get(t)
                next_val = next_high if next_high is not None else float("inf")
                return (-weighted, -has_high, next_val, t)

            dominant = sorted(candidates, key=_tie_key)[0]
    else:
        dominant = ""
        max_score = 0.0
    theme_pressure["dominant_theme"] = dominant
    theme_pressure["dominant_score"] = round(float(max_score), 3)
    theme_pressure["dominant_weighted_events"] = round(
        float(theme_pressure.get(dominant, {}).get("weighted_events", 0.0)), 3
    )
    overlay_score = theme_pressure.get("risk_regime", {}).get("score", 0.0)
    overlay_events = theme_pressure.get("risk_regime", {}).get("events", 0)
    overlay_weighted = theme_pressure.get("risk_regime", {}).get("weighted_events", 0.0)

    def _priority_score_upcoming(ev: Dict[str, Any]) -> float:
        in_hours = max(
            0.0, (ev["event_ts_utc"] - now_utc).total_seconds() / 3600.0
        )
        base = {"high": 1.0, "medium": 0.6, "low": 0.3}.get(ev["impact"], 0.3)
        time_factor = 1.0 / (1.0 + (in_hours / 72.0))
        tags = ev.get("macro_tags") or []
        has_any_numeric = any(ev.get(k) is not None for k in ["actual", "forecast", "previous"])
        has_estimates = ev.get("forecast") is not None or ev.get("previous") is not None
        numeric_bonus = 0.15 if has_any_numeric else -0.10
        if ev.get("impact") == "high" and ev.get("category") == "monetary_policy" and not has_any_numeric:
            numeric_bonus = 0.0
        estimates_bonus = 0.10 if has_estimates else 0.0
        theme_bonus = 0.10 if any(t in {"real_rates", "global_liquidity", "money_supply"} for t in tags) else 0.0
        overlay_bonus = 0.03 if "risk_regime" in tags else 0.0
        score = base * time_factor + numeric_bonus + estimates_bonus + theme_bonus + overlay_bonus
        return max(0.0, min(1.0, score))

    def _priority_score_recent(ev: Dict[str, Any]) -> float:
        base = {"high": 1.0, "medium": 0.6, "low": 0.3}.get(ev["impact"], 0.3)
        tags = ev.get("macro_tags") or []
        has_any_numeric = any(ev.get(k) is not None for k in ["actual", "forecast", "previous"])
        has_estimates = ev.get("forecast") is not None or ev.get("previous") is not None
        has_surprise_inputs = ev.get("actual") is not None and ev.get("forecast") is not None
        numeric_bonus = 0.15 if has_any_numeric else -0.10
        estimates_bonus = 0.10 if has_estimates else 0.0
        surprise_bonus = 0.10 if has_surprise_inputs else 0.0
        theme_bonus = 0.10 if any(t in {"real_rates", "global_liquidity", "money_supply"} for t in tags) else 0.0
        overlay_bonus = 0.03 if "risk_regime" in tags else 0.0
        score = base + numeric_bonus + estimates_bonus + surprise_bonus + theme_bonus + overlay_bonus
        return max(0.0, min(1.0, score))

    for ev in upcoming:
        ev["_priority_score"] = _priority_score_upcoming(ev)
        ev["_dedupe_title"] = _normalize_family(ev.get("title") or "")

    upcoming.sort(key=lambda ev: (-ev["_priority_score"], ev["event_ts_utc"]))
    upcoming_before = len(upcoming)
    # De-dup exact and near-duplicate titles within same day
    exact_map: Dict[Tuple[str, str, datetime], Dict[str, Any]] = {}
    for ev in upcoming:
        key = (ev.get("country") or "", ev.get("title") or "", ev["event_ts_utc"])
        existing = exact_map.get(key)
        if existing is None or ev["_priority_score"] > existing["_priority_score"]:
            exact_map[key] = ev

    near_map: Dict[Tuple[str, str, datetime.date, str], Dict[str, Any]] = {}
    for ev in exact_map.values():
        key = (
            ev.get("country") or "",
            ev["_dedupe_title"],
            ev["event_ts_utc"].date(),
            ev.get("impact") or "",
        )
        existing = near_map.get(key)
        if existing is None or ev["_priority_score"] > existing["_priority_score"]:
            near_map[key] = ev

    selected = list(near_map.values())
    selected.sort(key=lambda ev: (-ev["_priority_score"], ev["event_ts_utc"]))
    upcoming_after = len(selected)
    if debug:
        print(
            "[calendar:features][debug] top_upcoming dedupe "
            f"before={upcoming_before} after={upcoming_after}",
            file=sys.stderr,
        )

    top_upcoming: List[Dict[str, Any]] = []
    for ev in selected[: max_upcoming_items]:
        delta_hours = (ev["event_ts_utc"] - now_utc).total_seconds() / 3600.0
        has_estimates = ev.get("forecast") is not None or ev.get("previous") is not None
        priority_score = ev["_priority_score"]
        has_any_numeric = any(ev.get(k) is not None for k in ["actual", "forecast", "previous"])
        level_basis = None
        level_value = None
        if ev.get("forecast") is not None:
            level_basis = "forecast"
            level_value = ev.get("forecast")
        elif ev.get("previous") is not None:
            level_basis = "previous"
            level_value = ev.get("previous")
        level_context = None
        if level_basis is not None:
            level_history = _history_values(con, ev.get("title"), ev.get("country"), level_cutoff, level_cache)
            level_context = _level_context(level_value, level_history, level_basis)
            if level_context is not None:
                family = _normalize_history_family(ev.get("title") or "")
                country_key = (ev.get("country") or "").upper()
                cat_key = ev.get("category") or ""
                static_explainer = get_static_event_explainer(ev.get("title"), ev.get("category"))
                explainer = (
                    explainer_map.get((family, country_key, cat_key))
                    or explainer_map.get((family, country_key, ""))
                    or explainer_map.get((family, "", cat_key))
                    or explainer_map.get((family, "", ""))
                    or static_explainer
                )
                level_context["note"] = _level_context_note(
                    ev.get("category"),
                    ev.get("impact"),
                    level_context,
                    is_recent=False,
                    explainer=explainer,
                )
        top_upcoming.append(
            {
                "ts_utc": ev["event_ts_utc"].isoformat(),
                "in_hours": round(delta_hours, 2),
                "country": ev.get("country"),
                "impact": ev.get("impact"),
                "category": ev.get("category"),
                "title": ev.get("title"),
                "macro_tags": ev.get("macro_tags") or [],
                "forecast": ev.get("forecast"),
                "previous": ev.get("previous"),
                "unit": ev.get("unit"),
                "priority_score": round(priority_score, 3),
                "has_estimates": has_estimates,
                "has_any_numeric": has_any_numeric,
                "level_context": level_context,
            }
        )

    recent = [ev for ev in events if ev["event_ts_utc"] < now_utc]
    for ev in recent:
        ev["_priority_score"] = _priority_score_recent(ev)
        ev["_dedupe_title"] = _normalize_family(ev.get("title") or "")

    recent.sort(
        key=lambda ev: (
            -_impact_rank(ev["impact"]),
            abs((ev["event_ts_utc"] - now_utc).total_seconds() / 3600.0),
            ev["title"],
        )
    )

    recent_before = len(recent)
    exact_recent: Dict[Tuple[str, str, datetime], Dict[str, Any]] = {}
    for ev in recent:
        key = (ev.get("country") or "", ev.get("title") or "", ev["event_ts_utc"])
        existing = exact_recent.get(key)
        if existing is None or ev["_priority_score"] > existing["_priority_score"]:
            exact_recent[key] = ev

    near_recent: Dict[Tuple[str, str, datetime.date, str], Dict[str, Any]] = {}
    for ev in exact_recent.values():
        key = (
            ev.get("country") or "",
            ev["_dedupe_title"],
            ev["event_ts_utc"].date(),
            ev.get("impact") or "",
        )
        existing = near_recent.get(key)
        if existing is None or ev["_priority_score"] > existing["_priority_score"]:
            near_recent[key] = ev

    recent = list(near_recent.values())
    recent.sort(
        key=lambda ev: (
            -_impact_rank(ev["impact"]),
            abs((ev["event_ts_utc"] - now_utc).total_seconds() / 3600.0),
            ev["title"],
        )
    )
    recent_after = len(recent)
    if debug:
        print(
            "[calendar:features][debug] top_recent dedupe "
            f"before={recent_before} after={recent_after}",
            file=sys.stderr,
        )

    top_recent: List[Dict[str, Any]] = []
    for ev in recent[: max_recent_items]:
        delta_hours = (now_utc - ev["event_ts_utc"]).total_seconds() / 3600.0
        actual = ev.get("actual")
        forecast = ev.get("forecast")
        previous = ev.get("previous")
        has_surprise_inputs = actual is not None and forecast is not None
        surprise = None
        surprise_pct = None
        surprise_sign = None
        if has_surprise_inputs:
            surprise = actual - forecast
            if abs(surprise) < 1e-9:
                surprise_sign = "flat"
            elif surprise > 0:
                surprise_sign = "up"
            else:
                surprise_sign = "down"
            if forecast != 0:
                surprise_pct = surprise / abs(forecast)
        strength = _surprise_strength(surprise_pct) if has_surprise_inputs else "none"
        effect_hint = _effect_hint(
            ev.get("category"),
            ev.get("title"),
            ev.get("macro_tags") or [],
            actual,
            forecast,
            previous,
            strength,
        )
        level_basis = None
        level_value = None
        if actual is not None:
            level_basis = "actual"
            level_value = actual
        elif forecast is not None:
            level_basis = "forecast"
            level_value = forecast
        elif previous is not None:
            level_basis = "previous"
            level_value = previous
        level_context = None
        if level_basis is not None:
            level_history = _history_values(con, ev.get("title"), ev.get("country"), level_cutoff, level_cache)
            level_context = _level_context(level_value, level_history, level_basis)
            if level_context is not None:
                family = _normalize_history_family(ev.get("title") or "")
                country_key = (ev.get("country") or "").upper()
                cat_key = ev.get("category") or ""
                static_explainer = get_static_event_explainer(ev.get("title"), ev.get("category"))
                explainer = (
                    explainer_map.get((family, country_key, cat_key))
                    or explainer_map.get((family, country_key, ""))
                    or explainer_map.get((family, "", cat_key))
                    or explainer_map.get((family, "", ""))
                    or static_explainer
                )
                level_context["note"] = _level_context_note(
                    ev.get("category"),
                    ev.get("impact"),
                    level_context,
                    is_recent=True,
                    effect_hint=effect_hint,
                    explainer=explainer,
                )
        has_any_numeric = any(ev.get(k) is not None for k in ["actual", "forecast", "previous"])
        top_recent.append(
            {
                "ts_utc": ev["event_ts_utc"].isoformat(),
                "ago_hours": round(delta_hours, 2),
                "country": ev.get("country"),
                "impact": ev.get("impact"),
                "category": ev.get("category"),
                "title": ev.get("title"),
                "macro_tags": ev.get("macro_tags") or [],
                "actual": actual,
                "forecast": forecast,
                "previous": previous,
                "unit": ev.get("unit"),
                "surprise": surprise,
                "surprise_pct": surprise_pct,
                "surprise_sign": surprise_sign,
                "surprise_strength": strength,
                "effect_hint": effect_hint,
                "has_surprise_inputs": has_surprise_inputs,
                "has_any_numeric": has_any_numeric,
                "level_context": level_context,
            }
        )

    # Conservative risk_regime overlay classification based on directional evidence.
    impact_w = {"high": 1.0, "medium": 0.5, "low": 0.25}
    evidence_count = 0
    evidence_weight = 0.0
    weighted_sum = 0.0
    for ev in top_recent:
        if not ev.get("has_surprise_inputs"):
            continue
        evidence_count += 1
        bias = (ev.get("effect_hint") or {}).get("bias") or "neutral"
        conf = float((ev.get("effect_hint") or {}).get("confidence") or 0.0)
        impact_val = impact_w.get(ev.get("impact") or "low", 0.25)
        sign = 0.0
        if bias in {"tightening_pressure", "risk_off", "liquidity_down", "momentum_down"}:
            sign = -1.0
        elif bias in {"easing_pressure", "risk_on", "liquidity_up", "momentum_up"}:
            sign = 1.0
        weight = impact_val * conf
        evidence_weight += weight
        weighted_sum += sign * weight

    if evidence_count < 3 or evidence_weight < 0.5:
        overlay_class = "macro_sensitive"
        net_bias = 0.0
    else:
        net_bias = weighted_sum / max(evidence_weight, 1e-9)
        if net_bias <= -0.15:
            overlay_class = "risk_off"
        elif net_bias >= 0.15:
            overlay_class = "risk_on"
        else:
            overlay_class = "neutral"

    upcoming_7d = [ev for ev in upcoming if 0 <= (ev["event_ts_utc"] - now_utc).total_seconds() / 3600.0 <= 168]
    next_high = None
    next_high_title = None
    high_events_next_7d = 0
    liquidity_events_next_7d = 0
    money_supply_events_next_7d = 0
    for ev in upcoming_7d:
        in_hours = (ev["event_ts_utc"] - now_utc).total_seconds() / 3600.0
        if ev.get("impact") == "high":
            high_events_next_7d += 1
            if next_high is None or in_hours < next_high:
                next_high = in_hours
                next_high_title = ev.get("title")
        if ev.get("category") == "liquidity" or "global_liquidity" in (ev.get("macro_tags") or []):
            liquidity_events_next_7d += 1
        if ev.get("category") == "money_supply" or "money_supply" in (ev.get("macro_tags") or []):
            money_supply_events_next_7d += 1

    calendar_summary = {
        "next_high_event_in_hours": round(next_high, 2) if next_high is not None else None,
        "next_high_event_title": next_high_title,
        "high_events_next_7d": high_events_next_7d,
        "liquidity_events_next_7d": liquidity_events_next_7d,
        "money_supply_events_next_7d": money_supply_events_next_7d,
        "notes": "deterministic_summary",
    }

    return {
        "window": window,
        "freshness": freshness,
        "counts": counts,
        "theme_pressure": theme_pressure,
        "risk_regime_overlay": {
            "score": round(float(overlay_score), 3) if overlay_score is not None else None,
            "events": overlay_events,
            "weighted_events": round(float(overlay_weighted), 3)
            if overlay_weighted is not None
            else None,
            "classification": overlay_class,
            "directional_evidence": {
                "events_with_surprise": evidence_count,
                "net_bias": round(float(net_bias), 3),
                "evidence_weight": round(float(evidence_weight), 3),
            },
            "method": "calendar_tag_density",
            "notes": "overlay_only_not_in_dominance",
        },
        "top_upcoming": top_upcoming,
        "top_recent": top_recent,
        "calendar_summary": calendar_summary,
    }

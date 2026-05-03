"""Economic calendar fetcher and tagging helpers."""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional, Tuple

import requests


def _to_utc(dt: datetime) -> datetime:
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _parse_ts(value: Any) -> Optional[datetime]:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        try:
            return datetime.fromtimestamp(int(value), tz=timezone.utc)
        except Exception:
            return None
    if isinstance(value, str):
        s = value.strip()
        if not s:
            return None
        try:
            dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
            return _to_utc(dt)
        except Exception:
            pass
        for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d"):
            try:
                dt = datetime.strptime(s, fmt)
                return _to_utc(dt)
            except Exception:
                continue
    return None


def _parse_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        if isinstance(value, str):
            v = value.strip().replace(",", "")
            if v == "":
                return None
            return float(v)
        return float(value)
    except Exception:
        return None


def _normalize_title(title: str) -> str:
    return " ".join((title or "").lower().split())

def normalize_event_family(title: str) -> str:
    t = _normalize_title(title)
    tokens = t.split()
    drop = {
        "mom",
        "m/m",
        "mm",
        "qoq",
        "q/q",
        "qq",
        "yoy",
        "y/y",
        "yy",
        "sa",
        "s.a",
        "s.a.",
        "nsa",
        "n.s.a",
        "final",
        "prelim",
        "preliminary",
        "adv",
        "advance",
    }
    kept = []
    for tok in tokens:
        clean = tok.strip("()[],:")
        clean_norm = clean.replace("/", "").replace(".", "")
        if clean in drop or clean_norm in drop:
            continue
        kept.append(tok)
    return " ".join(kept)

STATIC_EVENT_EXPLAINERS: list[tuple[tuple[str, ...], str]] = [
    (("fomc",), "Fed policy statement and rate decision guidance."),
    (("federal open market committee",), "Fed policy statement and rate decision guidance."),
    (("fed interest rate decision",), "Fed policy rate decision setting the federal funds target."),
    (("interest rate decision", "fed"), "Fed policy rate decision setting the federal funds target."),
    (("interest rate decision", "boj"), "Bank of Japan policy rate decision."),
    (("interest rate decision", "bank of japan"), "Bank of Japan policy rate decision."),
    (("interest rate decision", "ecb"), "ECB policy rate decision."),
    (("interest rate decision", "european central bank"), "ECB policy rate decision."),
    (("press conference", "fed"), "Central bank press conference on policy outlook."),
    (("press conference", "ecb"), "Central bank press conference on policy outlook."),
    (("press conference", "boj"), "Central bank press conference on policy outlook."),
    (("core cpi",), "Core CPI measures inflation excluding food and energy."),
    (("cpi",), "CPI measures consumer price inflation."),
    (("core pce",), "Core PCE measures inflation excluding food and energy."),
    (("pce price index",), "PCE price index measures consumer inflation used by the Fed."),
    (("pce",), "PCE price index measures consumer inflation used by the Fed."),
    (("ppi",), "PPI measures producer price inflation."),
    (("nonfarm payroll",), "Nonfarm payrolls measure monthly US job growth."),
    (("payrolls", "nonfarm"), "Nonfarm payrolls measure monthly US job growth."),
    (("unemployment rate",), "Unemployment rate measures share of labor force unemployed."),
    (("jobless claims",), "Jobless claims track new unemployment benefit filings."),
    (("gdp",), "GDP measures total economic output."),
    (("pmi",), "PMI surveys business activity and sentiment."),
    (("ism",), "PMI/ISM surveys business activity and sentiment."),
    (("retail sales",), "Retail sales measure consumer spending."),
    (("balance sheet",), "Central bank balance sheet size reflects system liquidity."),
    (("money supply",), "Money supply measures broad money in circulation."),
    (("m2",), "Money supply measures broad money in circulation."),
    (("reverse repo",), "Reverse repo usage reflects cash parked at the Fed."),
    (("rrp",), "Reverse repo usage reflects cash parked at the Fed."),
    (("treasury general account",), "Treasury general account reflects government cash held at the Fed."),
    (("tga",), "Treasury general account reflects government cash held at the Fed."),
]


def get_static_event_explainer(title: str, category: Optional[str] = None) -> Optional[str]:
    t = _normalize_title(title)
    if not t:
        return None
    for keys, expl in STATIC_EVENT_EXPLAINERS:
        if all(k in t for k in keys):
            return _sanitize_explainer(expl)
    return None



def _sanitize_explainer(text: str) -> Optional[str]:
    if not text:
        return None
    cleaned = " ".join(str(text).split()).strip()
    cleaned = cleaned.strip('"').strip("'")
    words = cleaned.split()
    if len(words) > 15:
        cleaned = " ".join(words[:15])
    return cleaned.rstrip(".")


def generate_event_explainer(
    title: str,
    category: str,
    country: Optional[str],
    api_key: str,
    model: str = "gpt-5.2",
) -> Optional[str]:
    from openai import OpenAI

    client = OpenAI(api_key=api_key)
    system_msg = (
        "You explain what an economic release measures in 15 words or fewer. "
        "No forecasts, no advice. Output plain text only."
    )
    user_msg = (
        f"Title: {title}\n"
        f"Country: {country or 'N/A'}\n"
        f"Category: {category or 'other'}\n"
        "Explain what this release measures."
    )
    resp = client.chat.completions.create(
        model=model,
        temperature=0.0,
        max_completion_tokens=40,
        messages=[
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
        ],
    )
    content = resp.choices[0].message.content if resp.choices else None
    return _sanitize_explainer(content)


def ensure_calendar_explainers(
    con,
    candidates: Dict[Tuple[str, str, str], str],
    api_key: Optional[str],
    model: str = "gpt-5.2",
    now_utc: Optional[datetime] = None,
    debug: bool = False,
) -> int:
    if not api_key or not candidates:
        return 0
    existing = con.execute(
        "SELECT event_family, country, category FROM economic_calendar_explainers"
    ).fetchall()
    existing_set = {(r[0] or "", r[1] or "", r[2] or "") for r in existing or []}
    created = 0
    empty_count = 0
    first_error: Optional[Exception] = None
    logged_error = False
    if now_utc is None:
        now_utc = datetime.now(timezone.utc)
    for (family, country, category), title in candidates.items():
        key = (family or "", country or "", category or "")
        if key in existing_set:
            continue
        try:
            explanation = generate_event_explainer(
                title=title,
                category=category,
                country=country,
                api_key=api_key,
                model=model,
            )
        except Exception as exc:
            if first_error is None:
                first_error = exc
            if debug and not logged_error:
                print(
                    f"[calendar:pull][debug] explainer_error={type(exc).__name__}: {exc}",
                    file=sys.stderr,
                )
                logged_error = True
            continue
        if not explanation:
            empty_count += 1
            continue
        con.execute(
            """
            INSERT INTO economic_calendar_explainers (
                event_family, country, category, explanation, model, source, updated_at_utc
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            [family, country, category, explanation, model, "openai", now_utc],
        )
        existing_set.add(key)
        created += 1
    if debug and created == 0:
        err = (
            f"{type(first_error).__name__}: {first_error}"
            if first_error is not None
            else "none"
        )
        print(
            "[calendar:pull][debug] explainers_created=0 "
            f"candidates={len(candidates)} empty={empty_count} error={err}",
            file=sys.stderr,
        )
    return created


def pick_first(raw: dict, keys: List[str]) -> Any:
    for key in keys:
        if key in raw and raw.get(key) not in (None, "", "null"):
            return raw.get(key)
    return None


def classify_category(title: str) -> str:
    t = _normalize_title(title)
    if any(k in t for k in ["mortgage applications", "mortgage market", "mortgage rate", "mba mortgage", "mba 30"]):
        return "growth"
    if any(
        k in t
        for k in [
            "balance sheet",
            "quantitative easing",
            "quantitative tightening",
            "repo",
            "bill purchases",
            "liquidity",
            "h.4.1",
        ]
    ):
        return "liquidity"
    if any(k in t for k in ["money supply", "m2", "credit", "bank lending"]):
        return "money_supply"
    if any(k in t for k in ["cpi", "pce", "ppi", "inflation"]):
        return "inflation"
    if any(k in t for k in ["nonfarm", "payroll", "unemployment", "jobless"]):
        return "jobs"
    if any(k in t for k in ["gdp", "pmi", "retail", "industrial"]):
        return "growth"
    if "fed" in t and any(
        k in t
        for k in [
            "manufacturing index",
            "activity index",
            "survey",
            "conditions",
            "business outlook",
            "services index",
            "employment index",
            "new orders",
        ]
    ):
        return "surveys"
    is_cb_speech = any(k in t for k in ["speech", "remarks", "testimony"]) and any(
        k in t for k in ["fed", "federal reserve", "ecb", "boj", "bank of japan", "pboc", "european central bank"]
    )
    if is_cb_speech:
        return "monetary_policy"
    is_fomc = "fomc" in t or "federal open market committee" in t
    is_policy_meeting = "monetary policy meeting" in t or "policy meeting" in t
    is_rate_decision = "interest rate decision" in t or "rate decision" in t
    is_press_conf = "press conference" in t
    is_dot_plot = "dot plot" in t or "summary of economic projections" in t
    is_statement = "statement" in t and "fomc" in t
    is_ycc = ("ycc" in t or "yield curve control" in t) and ("boj" in t or "bank of japan" in t)
    is_minutes = "minutes" in t and any(k in t for k in ["fed", "ecb", "boj", "bank of japan"])
    if (
        is_fomc
        or is_policy_meeting
        or is_rate_decision
        or is_press_conf
        or is_dot_plot
        or is_statement
        or is_ycc
        or is_minutes
    ):
        return "monetary_policy"
    return "other"


def macro_tags_for_event(
    title: str,
    category: str,
    impact: Optional[str] = None,
    event_ts_utc: Optional[datetime] = None,
    now_utc: Optional[datetime] = None,
) -> List[str]:
    t = _normalize_title(title)
    impact_val = (impact or "medium").lower()
    tags: List[str] = []

    liquidity_ops = any(
        k in t
        for k in [
            "balance sheet",
            "quantitative easing",
            "quantitative tightening",
            "bill purchases",
            "repo",
            "liquidity",
            "h.4.1",
        ]
    )

    if category == "monetary_policy":
        tags.append("real_rates")
        if liquidity_ops:
            tags.append("global_liquidity")
    elif category == "inflation":
        tags.append("real_rates")
    elif category == "liquidity":
        tags.append("global_liquidity")
    elif category == "money_supply":
        tags.extend(["money_supply", "global_liquidity"])

    # risk_regime gating (overlay only) - avoid tagging medium/low broadly.
    risk_regime_allowed = False
    if impact_val == "high":
        risk_regime_allowed = True
    elif impact_val == "medium":
        if category in {"liquidity", "money_supply"}:
            risk_regime_allowed = True
    if risk_regime_allowed:
        tags.append("risk_regime")

    # Deduplicate while preserving order
    seen = set()
    out: List[str] = []
    for tag in tags:
        if tag not in seen:
            seen.add(tag)
            out.append(tag)
    return out


def classify_impact(country: Optional[str], title: str, category: str) -> str:
    c = (country or "").upper()
    t = _normalize_title(title)
    high_countries = {"US", "JP", "EU", "GB", "CN"}
    med_countries = {"US", "JP", "EU", "GB", "CN", "CA", "AU", "CH"}

    # Explicit low-impact overrides
    low_overrides = [
        "mortgage applications",
        "mortgage market index",
        "mortgage rate",
        "mba mortgage",
        "mba 30-year",
        "mba 30 year",
    ]
    if any(term in t for term in low_overrides):
        return "low"

    # Hard exclusions for high impact
    high_exclusions = [
        "chicago fed",
        "dallas fed",
        "richmond fed",
        "kansas city fed",
        "manufacturing index",
        "activity index",
        "survey",
        "auction",
        "bond auction",
        "bill auction",
        "ny fed bill purchases",
        "bill purchases",
        "repo operation",
        "repo operations",
    ]
    high_blocked = any(ex in t for ex in high_exclusions)

    central_banks = [
        "fed",
        "federal reserve",
        "ecb",
        "boj",
        "pboc",
        "bank of japan",
        "european central bank",
    ]

    def _has_any(terms: List[str]) -> bool:
        return any(term in t for term in terms)

    is_fomc = "fomc" in t or "federal open market committee" in t
    is_rate_decision = "interest rate decision" in t and _has_any(central_banks)
    is_press_conf = "press conference" in t and _has_any(["fed", "ecb", "boj", "bank of japan"])
    is_speech = any(k in t for k in ["speech", "remarks", "testimony"]) and _has_any(central_banks)
    is_dot_plot = ("dot plot" in t or "summary of economic projections" in t) and "fomc" in t
    is_fomc_statement = "statement" in t and "fomc" in t
    is_ycc = ("ycc" in t or "yield curve control" in t) and _has_any(["boj", "bank of japan"])
    is_cpi = "cpi" in t
    is_core_cpi = "core cpi" in t
    is_pce = "pce price index" in t or "core pce" in t
    is_inflation_rate = "inflation rate" in t and category == "inflation" and c in high_countries
    is_nfp = "nonfarm payroll" in t or "nfp" in t
    is_unemployment = "unemployment rate" in t and category == "jobs" and c in {"US", "EU", "JP", "GB"}
    is_balance_sheet = _has_any(
        ["fed balance sheet", "h.4.1", "quantitative tightening", "quantitative easing"]
    )

    if (not high_blocked) or is_balance_sheet:
        if (
            is_fomc
            or is_rate_decision
            or is_press_conf
            or is_dot_plot
            or is_fomc_statement
            or is_ycc
            or is_cpi
            or is_core_cpi
            or is_pce
            or is_inflation_rate
            or is_nfp
            or is_unemployment
            or is_balance_sheet
        ):
            return "high"

    if is_speech and c in med_countries:
        return "medium"

    medium_terms = [
        "pmi",
        "ism",
        "retail sales",
        "gdp",
        "ppi",
        "jobless claims",
        "initial jobless claims",
        "repo",
        "bill purchases",
        "treasury general account",
        "tga",
        "money supply",
        "m2",
        "credit",
        "bank lending",
    ]
    if c in med_countries and category in {"inflation", "jobs", "growth", "money_supply", "liquidity"}:
        return "medium"
    if c in med_countries and _has_any(medium_terms):
        return "medium"
    return "low"


if __name__ == "__main__":
    samples = {
        "HIGH": [
            "FOMC Interest Rate Decision",
            "US CPI (YoY)",
            "Core PCE Price Index",
            "Nonfarm Payrolls",
            "BoJ Interest Rate Decision",
            "ECB Press Conference",
        ],
        "MEDIUM": [
            "Initial Jobless Claims",
            "US ISM Manufacturing PMI",
            "Dallas Fed Manufacturing Index",
            "Chicago Fed National Activity Index",
            "NY Fed Bill Purchases 1 to 4 months",
            "Retail Sales (MoM)",
        ],
        "LOW": [
            "MBA Mortgage Applications",
            "EIA Natural Gas Storage Change",
        ],
    }
    for label, titles in samples.items():
        print(f"{label}:")
        for t in titles:
            category = classify_category(t)
            impact = classify_impact("US", t, category)
            print(f"  {t} -> {category} / {impact}")


def fetch_finnhub_calendar(
    api_key: str,
    start_date: str,
    end_date: str,
    timeout: int = 10,
) -> List[Dict[str, Any]]:
    url = "https://finnhub.io/api/v1/calendar/economic"
    params = {"from": start_date, "to": end_date, "token": api_key}
    try:
        resp = requests.get(url, params=params, timeout=timeout)
    except Exception as exc:
        raise RuntimeError(f"Finnhub request failed: {exc}") from exc
    if resp.status_code != 200:
        raise RuntimeError(f"Finnhub request failed: status={resp.status_code}")
    try:
        payload = resp.json()
    except Exception as exc:
        raise RuntimeError(f"Finnhub response parse error: {exc}") from exc

    events = payload.get("economicCalendar")
    if events is None and isinstance(payload, dict):
        events = payload.get("data")
    if events is None and isinstance(payload, list):
        events = payload
    if not events:
        return []

    normalized: List[Dict[str, Any]] = []
    for event in events:
        raw = event or {}
        title = raw.get("event") or raw.get("title") or ""
        if not title:
            continue
        ts = _parse_ts(raw.get("date") or raw.get("time") or raw.get("timestamp"))
        if ts is None:
            continue
        actual = _parse_float(pick_first(raw, ["actual", "value"]))
        forecast = _parse_float(pick_first(raw, ["estimate"]))
        previous = _parse_float(pick_first(raw, ["previous", "prev", "prior"]))
        unit = pick_first(raw, ["unit"]) or ""
        normalized.append(
            {
                "event_ts_utc": ts,
                "title": title,
                "country": raw.get("country"),
                "actual": actual,
                "forecast": forecast,
                "previous": previous,
                "unit": unit,
                "raw": raw,
            }
        )
    return normalized


def get_upcoming_calendar_events(
    con,
    now_utc: datetime,
    hours_ahead: int = 168,
    min_impact: Optional[str] = None,
    limit: int = 10,
) -> List[Dict[str, Any]]:
    end_ts = now_utc + timedelta(hours=hours_ahead)
    if min_impact:
        rows = con.execute(
            """
            SELECT event_ts_utc, country, impact, title
            FROM economic_calendar_events
            WHERE event_ts_utc >= ? AND event_ts_utc <= ? AND impact = ?
            ORDER BY event_ts_utc ASC
            LIMIT ?
            """,
            [now_utc, end_ts, min_impact, limit],
        ).fetchall()
    else:
        rows = con.execute(
            """
            SELECT event_ts_utc, country, impact, title
            FROM economic_calendar_events
            WHERE event_ts_utc >= ? AND event_ts_utc <= ?
            ORDER BY event_ts_utc ASC
            LIMIT ?
            """,
            [now_utc, end_ts, limit],
        ).fetchall()
    results: List[Dict[str, Any]] = []
    for ts, country, impact, title in rows or []:
        results.append(
            {
                "event_ts_utc": ts,
                "country": country,
                "impact": impact,
                "title": title,
            }
        )
    return results

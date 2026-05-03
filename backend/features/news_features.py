from __future__ import annotations

import json
import re
import math
from collections import Counter
from datetime import datetime, timezone, timedelta
from typing import List, Dict

from backend.services.news_api import (
    BAD_SOURCES,
    MACRO_CONTEXT_TERMS,
    is_macro_trusted_source,
    is_spam,
    is_macro_noise,
    is_listicle,
    ACTIVE_SYMBOLS,
)
from backend.features.news_source_quality import get_source_quality, source_family

ACTIVE_UNIVERSE = {"BTC-USD", "ETH-USD", "SOL-USD", "XRP-USD", "SUI-USD"}

BULLISH_KEYWORDS = [
    "rally", "surge", "recover", "bounce", "rebound",
    "growth", "bullish", "uptrend", "gain", "gains",
    "strong", "accumulate", "optimistic", "upgrade",
    "momentum", "inflow", "buying", "support holding",
]

BEARISH_KEYWORDS = [
    "crash", "plunge", "drop", "drops", "down", "dip",
    "selloff", "sell-off", "liquidation", "liquidations",
    "bearish", "fear", "concern", "warning",
    "downtrend", "slump", "bleed", "bleeding",
    "decline", "collapse", "hack", "exploit",
    "market in red", "sell pressure", "volatility spike",
    "downward", "worst month", "suffers", "losses",
]

# Strong negative phrases that should override category bias
STRONG_NEGATIVE = [
    "slips", "slides", "tumbles", "slump", "selloff", "sell-off", "capitulation",
    "risk increases", "risk-off", "panic", "bloodbath", "crashing", "plunging",
]

CRYPTO_SIGNAL_TERMS = [
    "crypto", "cryptocurrency", "blockchain", "digital asset", "digital assets",
    "bitcoin", "btc", "ethereum", "eth", "solana", "sol", "xrp", "ripple", "sui",
    "token", "defi", "stablecoin", "exchange", "web3", "coinbase",
]

POLICY_TERMS = [
    "congress", "senate", "house", "vote", "voted", "bill", "legislation",
    "regulation", "regulator", "committee", "hearing",
    "sec", "cftc", "treasury", "white house", "federal",
]

US_MACRO_TERMS = [
    "federal reserve", "fed", "fomc", "treasury", "white house",
    "congress", "senate", "house", "u.s.", "united states",
    "cpi", "pce", "nonfarm", "jobs report", "unemployment",
    "gdp", "yield", "bond", "rate decision", "interest rate",
]

# High-impact macro/geopolitics triggers that can dominate market tone.
MACRO_OVERRIDE_TERMS = [
    "war", "invasion", "attack", "missile", "nuclear",
    "sanction", "sanctions", "tariff", "tariffs", "embargo",
    "blockade", "coup", "martial law",
]

# Macro severity grading terms for context weighting.
MACRO_SEVERITY_HIGH = [
    "war", "invasion", "nuclear", "missile", "attack", "blockade", "coup", "martial law",
]
MACRO_SEVERITY_MEDIUM = [
    "sanction", "sanctions", "tariff", "tariffs", "embargo", "strike", "escalation",
    "conflict", "trade war", "military",
]
MACRO_SEVERITY_LOW = [
    "election", "referendum", "policy", "central bank", "interest rate",
    "rate hike", "rate cut", "inflation", "recession", "gdp",
    "treasury", "bond yield", "yields", "opec", "oil", "energy shock",
]

MACRO_MIN_SOURCE_QUALITY = 2
PRIMARY_WEIGHT_SCALE = 0.6
MACRO_WEIGHT_SCALE = 1.0

# Market impact language (non-crypto) that signals risk-on/off conditions.
MACRO_SIGNAL_TERMS = [
    "market", "risk", "volatility", "selloff", "panic",
    "liquidity", "crisis", "shock", "risk-off",
]
ECOSYSTEM_TERMS = {
    "BTC-USD": ["bitcoin", "btc", "halving", "etf", "miner", "mining", "hashrate"],
    "ETH-USD": ["ethereum", "eth", "staking", "eip", "layer 2", "arbitrum", "optimism", "polygon", "base"],
    "SOL-USD": ["solana", "sol", "validator", "tps", "nft", "magic eden", "jito"],
    "XRP-USD": ["xrp", "ripple", "xrpl", "xrp ledger"],
    "SUI-USD": ["sui", "sui network", "mysten labs", "move language"],
}

CATEGORIES = {
    "etf": ["etf", "spot etf", "approval", "sec filing"],
    "selloff": ["crash", "plunge", "selloff", "liquidation", "dump", "down"],
    "upgrade": ["upgrade", "hard fork", "eip", "fusaka", "halving"],
    "hack": ["hack", "exploit", "breach", "security incident"],
    "institutional": ["blackrock", "fidelity", "inflows", "treasury", "institutional"],
    "regulation": ["sec", "cftc", "lawsuit", "ban", "regulation", "compliance"],
    "market": ["market", "crypto market", "trend", "rally", "pullback"],
}


def recency_decay(dt, now):
    age_minutes = (now - dt).total_seconds() / 60
    return math.exp(-age_minutes / 720)


def to_utc(ts):
    if ts is None:
        return None
    if ts.tzinfo is None:
        return ts.replace(tzinfo=timezone.utc)
    return ts.astimezone(timezone.utc)


def classify_category(title: str) -> List[str]:
    t = (title or "").lower()
    hits = []
    for cat, words in CATEGORIES.items():
        if any(w in t for w in words):
            hits.append(cat)
    return hits or ["general"]


def relevance_score(symbol: str, title: str) -> float:
    t = (title or "").lower()
    score = 1.0
    base = symbol.split("-")[0].lower()

    if base in t:
        score += 1.0

    for term in ECOSYSTEM_TERMS.get(symbol, []):
        if term in t:
            score += 1.0
            break
    return score


def compute_news_features(symbol: str, con) -> dict:
    """
    Compute news features using a 48h window with 72h fallback and weighted sentiment.
    """
    if symbol not in ACTIVE_UNIVERSE:
        return {
            "direction": "neutral",
            "intensity": 0.0,
            "article_count": 0,
            "window_hours": 48,
            "category_breakdown": {},
            "weighted_stats": {"weighted_bullish": 0.0, "weighted_bearish": 0.0, "weighted_raw": 0.0},
            "articles": [],
        }
    now_utc = datetime.now(timezone.utc)
    cutoff_primary = now_utc - timedelta(hours=48)
    cutoff_fallback = now_utc - timedelta(hours=72)
    cutoff_secondary = cutoff_fallback

    try:
        rows_raw = con.execute(
            """
            SELECT source, title, polarity, published_at, ai_meta
            FROM news_items
            WHERE symbol = ?
              AND lane = 'symbol'
            """,
            [symbol],
        ).fetchall()
    except Exception:
        rows_raw = con.execute(
            """
            SELECT source, title, polarity, published_at
            FROM news_items
            WHERE symbol = ?
            """,
            [symbol],
        ).fetchall()

    try:
        undated_rows = int(
            con.execute(
                """
                SELECT COUNT(*)
                FROM news_items
                WHERE symbol = ? AND lane = 'symbol' AND published_at IS NULL
                """,
                [symbol],
            ).fetchone()[0]
            or 0
        )
    except Exception:
        undated_rows = int(
            con.execute(
                """
                SELECT COUNT(*)
                FROM news_items
                WHERE symbol = ? AND published_at IS NULL
                """,
                [symbol],
            ).fetchone()[0]
            or 0
        )

    rows = []
    for r in rows_raw:
        ai_meta = None
        if len(r) > 4:
            ai_meta = r[4]
        rows.append(
            {
                "source": r[0],
                "title": r[1],
                "polarity": r[2],
                "published_at": to_utc(r[3]),
                "ai_meta": ai_meta,
            }
        )

    rows_recent = [r for r in rows if r["published_at"] and r["published_at"] >= cutoff_primary]
    if len(rows_recent) < 20:
        rows_recent = [r for r in rows if r["published_at"] and r["published_at"] >= cutoff_fallback]

    rows_recent.sort(key=lambda r: r["published_at"], reverse=True)

    # Deduplicate
    seen = set()
    unique = []
    for r in rows_recent:
        key = ((r["source"] or "").lower(), (r["title"] or "").lower())
        if key in seen:
            continue
        seen.add(key)
        unique.append(r)
    rows_recent = unique

    if len(rows_recent) < 5:
        # Continue with small samples; stability relies on market context spillover later.
        pass

    def _has_token(term: str, text: str) -> bool:
        term = term.lower()
        if not term:
            return False
        # Use word boundaries for short tokens to avoid false positives (e.g., "etf" in "netflix").
        if term.isalnum() and len(term) <= 4:
            return re.search(rf"\\b{re.escape(term)}\\b", text) is not None
        return term in text

    def _clamp01(val: float) -> float:
        try:
            return float(max(0.0, min(1.0, val)))
        except Exception:
            return 0.0

    def _title_has_crypto_terms(symbol: str, title: str) -> bool:
        t = (title or "").lower()
        if not t:
            return False

        base = symbol.split("-")[0].lower()
        if _has_token(base, t):
            return True
        if any(_has_token(term, t) for term in ECOSYSTEM_TERMS.get(symbol, [])):
            return True
        generic_terms = [
            "crypto", "cryptocurrency", "blockchain", "digital asset",
            "bitcoin", "btc", "ethereum", "eth", "solana", "sol",
            "xrp", "ripple", "sui", "token", "defi", "etf", "coinbase",
        ]
        return any(_has_token(term, t) for term in generic_terms)

    def _title_has_policy_terms(title: str) -> bool:
        t = (title or "").lower()
        if not t:
            return False
        policy_terms = [
            "congress", "senate", "house", "vote", "voted", "bill", "legislation",
            "regulation", "regulator", "committee", "hearing",
        ]
        return any(_has_token(term, t) for term in policy_terms)

    def _extract_labels(items):
        labels = []
        for item in items or []:
            if isinstance(item, str):
                labels.append(item)
            elif isinstance(item, dict):
                label = item.get("label")
                if isinstance(label, dict):
                    labels.append(label.get("eng") or "")
                elif isinstance(label, str):
                    labels.append(label)
                else:
                    labels.append(item.get("name") or item.get("title") or "")
        return [l for l in labels if l]

    def _meta_has_crypto_context(ai_meta) -> bool:
        if not ai_meta:
            return False
        try:
            if isinstance(ai_meta, str):
                meta = json.loads(ai_meta)
            elif isinstance(ai_meta, dict):
                meta = ai_meta
            else:
                return False
        except Exception:
            return False

        crypto_markers = [
            "crypto", "cryptocurrency", "blockchain", "digital asset",
            "bitcoin", "btc", "ethereum", "eth", "solana", "sol",
            "xrp", "ripple", "sui", "defi", "token", "etf",
        ]

        categories = _extract_labels(meta.get("categories") or [])
        if categories:
            cat_text = " ".join(categories).lower()
            if any(k in cat_text for k in crypto_markers):
                return True
            # Categories exist but are not crypto-related: reject.
            return False

        # If no categories, do not accept meta-only context.
        return False

    def _meta_summary(ai_meta) -> str:
        if not ai_meta:
            return ""
        try:
            if isinstance(ai_meta, str):
                meta = json.loads(ai_meta)
            elif isinstance(ai_meta, dict):
                meta = ai_meta
            else:
                return ""
        except Exception:
            return ""
        summary = meta.get("summary") or meta.get("bodyAbstract") or meta.get("abstract")
        return summary.strip() if isinstance(summary, str) else ""

    def _parse_ai_meta(ai_meta) -> Dict:
        if not ai_meta:
            return {}
        try:
            if isinstance(ai_meta, str):
                return json.loads(ai_meta)
            if isinstance(ai_meta, dict):
                return ai_meta
        except Exception:
            return {}
        return {}

    def _meta_labels(meta: Dict) -> List[str]:
        labels = []
        for key in ("categories", "topics", "concepts", "entities"):
            labels.extend(_extract_labels(meta.get(key) or []))
        return [l for l in labels if l]

    def _meta_event_uri(meta: Dict) -> str:
        ev = meta.get("eventUri") if isinstance(meta, dict) else None
        return ev or ""

    def _crypto_signal_score(symbol: str, title: str, summary: str, meta: Dict) -> float:
        text = f"{title or ''} {summary or ''}".lower()
        score = 0.0
        if any(_has_token(term, text) for term in CRYPTO_SIGNAL_TERMS):
            score += 1.0
        base = symbol.split("-")[0].lower()
        if _has_token(base, text):
            score += 1.0
        if any(_has_token(term, text) for term in ECOSYSTEM_TERMS.get(symbol, [])):
            score += 1.0
        labels = _meta_labels(meta)
        if labels:
            label_text = " ".join(labels).lower()
            if any(_has_token(term, label_text) for term in CRYPTO_SIGNAL_TERMS):
                score += 1.0
        return score

    def _asset_focus_score(symbol: str, title: str, summary: str, meta: Dict) -> float:
        text = f"{title or ''} {summary or ''}".lower()
        base = symbol.split("-")[0].lower()
        score = 0.0
        if _has_token(base, text):
            score += 1.0
        if any(_has_token(term, text) for term in ECOSYSTEM_TERMS.get(symbol, [])):
            score += 1.0
        labels = _meta_labels(meta)
        if labels:
            label_text = " ".join(labels).lower()
            if _has_token(base, label_text) or any(
                _has_token(term, label_text) for term in ECOSYSTEM_TERMS.get(symbol, [])
            ):
                score += 1.0
        return score

    def _policy_hit(text: str) -> bool:
        return any(_has_token(term, text) for term in POLICY_TERMS)

    def _crypto_macro_hit(text: str, meta: Dict) -> bool:
        if any(_has_token(term, text) for term in CRYPTO_SIGNAL_TERMS):
            return True
        labels = _meta_labels(meta)
        if labels:
            label_text = " ".join(labels).lower()
            return any(_has_token(term, label_text) for term in CRYPTO_SIGNAL_TERMS)
        return False

    def _us_macro_hit(text: str) -> bool:
        return any(_has_token(term, text) for term in US_MACRO_TERMS)

    def _policy_classify(symbol: str, title: str, summary: str, crypto_score: float) -> Dict | None:
        text = f"{title or ''} {summary or ''}".lower()
        if not text.strip():
            return None
        if not _policy_hit(text):
            return None
        if crypto_score < 1.0:
            return None

        stage = "unknown"
        if "vote" in text or "voted" in text:
            stage = "vote"
        elif "passed" in text or "passage" in text:
            stage = "passed"
        elif "hearing" in text:
            stage = "hearing"
        elif "introduced" in text or "introduces" in text:
            stage = "introduced"

        branch = "unknown"
        if "senate" in text:
            branch = "senate"
        elif "house" in text:
            branch = "house"
        elif "congress" in text:
            branch = "congress"
        elif "sec" in text or "cftc" in text:
            branch = "agency"
        elif "white house" in text or "president" in text:
            branch = "executive"
        elif "supreme court" in text or "court" in text:
            branch = "court"

        us_federal = branch in {"senate", "house", "congress", "agency", "executive", "court"} or "u.s." in text or " us " in text
        crypto_related = crypto_score >= 1.0
        conf = 0.4
        conf += 0.2 if stage != "unknown" else 0.0
        conf += 0.2 if branch != "unknown" else 0.0
        conf += 0.2 * _clamp01(crypto_score / 2.0)
        conf = _clamp01(conf)

        return {
            "us_federal": us_federal,
            "crypto_related": crypto_related,
            "stage": stage,
            "branch": branch,
            "confidence": conf,
        }

    def _extract_policy_meta(ai_meta) -> Dict | None:
        if not ai_meta:
            return None
        try:
            if isinstance(ai_meta, str):
                meta = json.loads(ai_meta)
            elif isinstance(ai_meta, dict):
                meta = ai_meta
            else:
                return None
        except Exception:
            return None
        policy = meta.get("policy_meta")
        return policy if isinstance(policy, dict) else None

    scored_rows = []
    for r in rows_recent:
        src = (r["source"] or "").lower()
        if src in BAD_SOURCES:
            continue
        if is_spam(r["title"]) or is_listicle(r["title"]):
            continue

        meta = _parse_ai_meta(r.get("ai_meta"))
        summary = _meta_summary(meta)
        crypto_score = _crypto_signal_score(symbol, r["title"], summary, meta)
        asset_focus = _asset_focus_score(symbol, r["title"], summary, meta)
        policy_meta = _extract_policy_meta(meta)
        if policy_meta is None:
            policy_meta = _policy_classify(symbol, r["title"], summary, crypto_score)

        if is_macro_noise(r["title"]) and crypto_score < 2.0:
            continue
        if crypto_score < 1.0 and not _meta_has_crypto_context(meta):
            continue

        r["summary"] = summary
        r["event_uri"] = _meta_event_uri(meta)
        r["crypto_score"] = crypto_score
        r["asset_focus_score"] = asset_focus
        r["policy_meta"] = policy_meta
        scored_rows.append(r)

    rows_recent = scored_rows

    # Cluster by eventUri to reduce duplicates; keep top sources per event.
    by_event = {}
    no_event = []
    for r in rows_recent:
        ev = r.get("event_uri")
        if ev:
            by_event.setdefault(ev, []).append(r)
        else:
            no_event.append(r)

    clustered = []
    for ev, items in by_event.items():
        items_sorted = sorted(
            items,
            key=lambda rr: (
                -get_source_quality(rr["source"]),
                -(rr["published_at"].timestamp() if rr.get("published_at") else 0.0),
                (rr.get("title") or "").lower(),
            ),
        )
        keep = []
        if items_sorted:
            keep.append(items_sorted[0])
        for cand in items_sorted[1:]:
            if len(keep) >= 2:
                break
            if (cand.get("source") or "").lower() != (keep[0].get("source") or "").lower():
                keep.append(cand)
        clustered.extend(keep)

    rows_recent = no_event + clustered

    weighted_bullish = 0.0
    weighted_bearish = 0.0

    # Primary weighting first
    def _score_primary(rows, weight_scale: float):
        bull = 0.0
        bear = 0.0
        for r in rows:
            t = (r["title"] or "").lower()
            bull_hits = sum(1 for w in BULLISH_KEYWORDS if w in t)
            bear_hits = sum(1 for w in BEARISH_KEYWORDS if w in t)

            # Strong negative override: ensure bearish weight when stress language appears
            if any(sn in t for sn in STRONG_NEGATIVE):
                bear_hits = max(bear_hits, 1)
                bull_hits = 0

            crypto_mod = 1.0 + 0.15 * min(3.0, float(r.get("crypto_score") or 0.0))
            asset_mod = 1.0 + 0.10 * min(3.0, float(r.get("asset_focus_score") or 0.0))
            weight = (
                relevance_score(symbol, t)
                * get_source_quality(r["source"])
                * recency_decay(r["published_at"], now_utc)
                * weight_scale
                * crypto_mod
                * asset_mod
            )
            bull += bull_hits * weight
            bear += bear_hits * weight
        return bull, bear

    # Market context spillover: macro/geopolitics always, crypto-market context only when thin coverage
    context_bullish = 0.0
    context_bearish = 0.0
    macro_bullish = 0.0
    macro_bearish = 0.0
    market_context = []
    MIN_PRIMARY_ARTICLES = 3
    context_cap = 0.20
    macro_cap = 0.20

    context_rows = []
    try:
        context_rows = con.execute(
            """
            SELECT source, title, polarity, published_at, ai_meta, macro_tag
            FROM news_items
            WHERE published_at >= ?
              AND lane = 'macro'
            ORDER BY published_at DESC
            """,
            [cutoff_primary],
        ).fetchall()
    except Exception:
        context_rows = con.execute(
            """
            SELECT source, title, polarity, published_at, ai_meta
            FROM news_items
            WHERE published_at >= ?
              AND symbol = 'MACRO'
            ORDER BY published_at DESC
            """,
            [cutoff_primary],
        ).fetchall()

    MARKET_TERMS = ["market", "crypto", "selloff", "regulation", "macro", "risk", "liquidity", "volatility"]
    macro_trigger_terms = MACRO_CONTEXT_TERMS + MACRO_OVERRIDE_TERMS

    def _token_match(text: str, term: str) -> bool:
        term = term.lower()
        if not term:
            return False
        # Use word boundaries for short tokens to avoid false positives (e.g., "etf" in "netflix").
        if term.isalnum() and len(term) <= 4:
            return re.search(rf"\b{re.escape(term)}\b", text) is not None
        return term in text

    def _macro_hit(text: str, terms: List[str]) -> bool:
        return any(_token_match(text, mt) for mt in terms)

    def _macro_severity(text: str) -> float:
        if _macro_hit(text, MACRO_SEVERITY_HIGH):
            return 1.6
        if _macro_hit(text, MACRO_SEVERITY_MEDIUM):
            return 1.3
        if _macro_hit(text, MACRO_SEVERITY_LOW):
            return 1.1
        return 1.0

    def _macro_source_multiplier(src: str) -> float:
        q = get_source_quality(src)
        if q >= 3:
            return 1.25
        if q == 2:
            return 1.0
        if q == 1:
            return 0.85
        return 0.7

    def _macro_priority(item) -> tuple:
        src, title, _, ts, _, _, asset_focus, crypto_hit, us_hit, _ = item
        return (
            -float(asset_focus),
            -(1 if crypto_hit else 0),
            -(1 if us_hit else 0),
            -get_source_quality(src),
            -(to_utc(ts).timestamp() if ts else 0.0),
            (title or "").lower(),
        )

    macro_candidates = []
    macro_candidates_trusted = []
    for row in context_rows:
        if len(row) == 6:
            src, title, pol, ts, ai_meta, _macro_tag = row
        else:
            src, title, pol, ts, ai_meta = row
        if not ts:
            continue
        tctx = (title or "").lower()
        if not _macro_hit(tctx, macro_trigger_terms):
            continue
        is_trusted = is_macro_trusted_source(src)
        if not is_trusted:
            continue
        meta = _parse_ai_meta(ai_meta)
        summary = _meta_summary(meta)
        combined = f"{tctx} {summary}".lower()
        asset_focus = _asset_focus_score(symbol, title, summary, meta)
        crypto_hit = _crypto_macro_hit(combined, meta) or asset_focus > 0
        us_hit = _us_macro_hit(combined)
        macro_candidates.append((src, title, pol, ts, tctx, ai_meta, asset_focus, crypto_hit, us_hit, summary))
        if is_trusted:
            macro_candidates_trusted.append((src, title, pol, ts, tctx, ai_meta, asset_focus, crypto_hit, us_hit, summary))

    # Fallback: if no MACRO rows, use macro-tagged primary headlines as context
    if not macro_candidates_trusted:
        for r in rows_recent:
            tprim = (r["title"] or "").lower()
            if _macro_hit(tprim, macro_trigger_terms):
                is_trusted = is_macro_trusted_source(r["source"])
                if not is_trusted:
                    continue
                meta = _parse_ai_meta(r.get("ai_meta"))
                summary = r.get("summary") or _meta_summary(meta)
                combined = f"{tprim} {summary}".lower()
                asset_focus = _asset_focus_score(symbol, r["title"], summary, meta)
                crypto_hit = _crypto_macro_hit(combined, meta) or asset_focus > 0
                us_hit = _us_macro_hit(combined)
                macro_candidates.append((r["source"], r["title"], r["polarity"], r["published_at"], tprim, r.get("ai_meta"), asset_focus, crypto_hit, us_hit, summary))
                if is_trusted:
                    macro_candidates_trusted.append((r["source"], r["title"], r["polarity"], r["published_at"], tprim, r.get("ai_meta"), asset_focus, crypto_hit, us_hit, summary))

    # Prefer trusted macro sources; fall back to quality-gated macro if none exist.
    if macro_candidates_trusted:
        macro_candidates = macro_candidates_trusted
    macro_candidates = sorted(macro_candidates, key=_macro_priority)

    # Determine if macro should be a dominant influencer
    macro_strong_hits = sum(1 for _, _, _, _, t, *_ in macro_candidates if _macro_hit(t, MACRO_OVERRIDE_TERMS))
    macro_signal_hits = sum(1 for _, _, _, _, t, *_ in macro_candidates if _macro_hit(t, MACRO_SIGNAL_TERMS))
    macro_override = macro_strong_hits >= 1 and macro_signal_hits >= 1
    macro_weight_scale = MACRO_WEIGHT_SCALE * (1.25 if macro_override else 1.0)

    macro_keys = set()
    macro_display_families = set()
    for src, title, pol, ts, tctx, ai_meta, asset_focus, crypto_hit, us_hit, summary in macro_candidates:
        macro_keys.add(((src or "").lower(), (title or "").lower()))
        bull_hits = sum(1 for w in BULLISH_KEYWORDS if w in tctx)
        bear_hits = sum(1 for w in BEARISH_KEYWORDS if w in tctx)
        if any(sn in tctx for sn in STRONG_NEGATIVE):
            bear_hits = max(bear_hits, 1)
            bull_hits = 0

        severity_mult = _macro_severity(tctx)
        if severity_mult >= 1.6:
            bear_hits = max(bear_hits, 2)
            bull_hits = 0
        elif severity_mult >= 1.3:
            bear_hits = max(bear_hits, 1)
            bull_hits = 0

        macro_focus_mult = 1.0
        if asset_focus > 0:
            macro_focus_mult += 0.20
        elif crypto_hit:
            macro_focus_mult += 0.12
        elif us_hit:
            macro_focus_mult += 0.08

        w = (
            relevance_score(symbol, tctx)
            * get_source_quality(src)
            * recency_decay(to_utc(ts), now_utc)
            * severity_mult
            * _macro_source_multiplier(src)
            * macro_weight_scale
            * macro_focus_mult
        )

        macro_bullish += bull_hits * w
        macro_bearish += bear_hits * w
        # Display policy: macro/geopolitical list should be ex-crypto context only.
        if not crypto_hit and asset_focus <= 0:
            family = source_family(src)
            if family in macro_display_families:
                continue
            macro_display_families.add(family)
            market_context.append(
                {
                    "source": src,
                    "title": title,
                    "published_at": to_utc(ts).strftime("%Y-%m-%d %H:%M UTC"),
                    "summary": summary,
                }
            )

    # Remove macro-duplicated items from primary feed (macro is primary)
    if macro_keys:
        rows_primary = [
            r for r in rows_recent
            if ((r["source"] or "").lower(), (r["title"] or "").lower()) not in macro_keys
        ]
    else:
        rows_primary = rows_recent

    primary_weighted_bullish, primary_weighted_bearish = _score_primary(
        rows_primary,
        PRIMARY_WEIGHT_SCALE,
    )
    primary_raw = primary_weighted_bullish - primary_weighted_bearish

    # Combine: macro/geopolitical is primary; recent relevant news is secondary.
    weighted_bullish = macro_bullish + primary_weighted_bullish
    weighted_bearish = macro_bearish + primary_weighted_bearish
    weighted_raw = weighted_bullish - weighted_bearish
    macro_raw = macro_bullish - macro_bearish
    total_count = len(rows_primary) + len(macro_candidates)
    threshold = 0.10 * total_count

    if weighted_raw > threshold:
        direction = "bullish"
    elif weighted_raw < -threshold:
        direction = "bearish"
    else:
        direction = "neutral"

    intensity = min(1.0, abs(weighted_raw) / (total_count or 1))
    timestamp_quality_factor = 1.0
    if undated_rows > 0:
        denom = max(20.0, float(len(rows_raw) or 1))
        timestamp_quality_factor = max(0.70, 1.0 - min(0.30, float(undated_rows) / denom))
        intensity = min(1.0, float(intensity) * float(timestamp_quality_factor))

    category_counts = Counter()
    for r in rows_primary:
        for cat in classify_category(r["title"]):
            category_counts[cat] += 1
    for src, title, *_ in macro_candidates:
        for cat in classify_category(title):
            category_counts[cat] += 1

    policy_events = []
    for r in rows_primary:
        policy = r.get("policy_meta") or _extract_policy_meta(r.get("ai_meta"))
        if not policy:
            continue
        if not policy.get("crypto_related") or not policy.get("us_federal"):
            continue
        try:
            conf = float(policy.get("confidence") or 0.0)
        except Exception:
            conf = 0.0
        if conf < 0.6:
            continue
        if get_source_quality(r["source"]) < 2:
            continue
        policy_events.append(
            {
                "source": r["source"],
                "title": r["title"],
                "published_at": r["published_at"].strftime("%Y-%m-%d %H:%M UTC"),
                "stage": policy.get("stage") or "unknown",
                "branch": policy.get("branch") or "unknown",
                "confidence": round(conf, 2),
            }
        )

    policy_events.sort(key=lambda x: x["published_at"], reverse=True)
    policy_events = policy_events[:10]

    rows_primary_sorted = sorted(
        rows_primary,
        key=lambda r: (
            -get_source_quality(r["source"]),
            -float(r.get("asset_focus_score") or 0.0),
            -float(r.get("crypto_score") or 0.0),
            -(r["published_at"].timestamp() if r.get("published_at") else 0.0),
            (r.get("title") or "").lower(),
        ),
    )

    articles_out = [
        {
            "source": r["source"],
            "title": r["title"],
            "published_at": r["published_at"].strftime("%Y-%m-%d %H:%M UTC"),
            "weight": get_source_quality(r["source"].strip() if isinstance(r["source"], str) else r["source"]),
            "summary": r.get("summary") or _meta_summary(r.get("ai_meta")),
        }
        for r in rows_primary_sorted[:15]
    ]

    sector_highlights = []
    macro_highlights = []
    try:
        sector_rows = con.execute(
            """
            SELECT source, title, published_at
            FROM news_items
            WHERE lane = 'sector' AND published_at >= ?
            ORDER BY published_at DESC
            """,
            [cutoff_primary],
        ).fetchall()
        sector_rows_sorted = sorted(
            sector_rows,
            key=lambda r: (
                -(
                    1
                    if (
                        r[2]
                        and (now_utc - to_utc(r[2])).total_seconds() <= 24 * 3600
                    )
                    else 0
                ),
                -(to_utc(r[2]).timestamp() if r[2] else 0.0),
                -get_source_quality(r[0]),
            ),
        )
        for src, title, ts in sector_rows_sorted[:3]:
            macro_ts = to_utc(ts)
            sector_highlights.append(
                {
                    "source": src,
                    "title": title,
                    "published_at": macro_ts.strftime("%Y-%m-%d %H:%M UTC") if macro_ts else "unknown",
                }
            )

        macro_rows = con.execute(
            """
            SELECT source, title, published_at, macro_tag, ai_meta
            FROM news_items
            WHERE lane = 'macro' AND published_at >= ?
            ORDER BY published_at DESC
            """,
            [cutoff_primary],
        ).fetchall()
        tag_best = {}
        for src, title, ts, tag, ai_meta in macro_rows:
            meta = _parse_ai_meta(ai_meta)
            summary = _meta_summary(meta)
            combined = f"{(title or '').lower()} {summary.lower()}".strip()
            if _crypto_macro_hit(combined, meta):
                continue
            tag_key = tag or "unknown"
            is_recent = 1 if (ts and (now_utc - to_utc(ts)).total_seconds() <= 24 * 3600) else 0
            score = (
                is_recent,
                to_utc(ts).timestamp() if ts else 0.0,
                get_source_quality(src),
            )
            prev = tag_best.get(tag_key)
            if prev is None or score > prev["score"]:
                tag_best[tag_key] = {
                    "source": src,
                    "title": title,
                    "published_at": ts,
                    "macro_tag": tag_key,
                    "score": score,
                }
        macro_best = list(tag_best.values())
        macro_best.sort(
            key=lambda r: (
                -(to_utc(r["published_at"]).timestamp() if r["published_at"] else 0.0),
                -get_source_quality(r["source"]),
            )
        )
        macro_highlight_families = set()
        for item in macro_best[:3]:
            family = source_family(item["source"])
            if family in macro_highlight_families:
                continue
            macro_highlight_families.add(family)
            macro_ts = to_utc(item["published_at"])
            macro_highlights.append(
                {
                    "source": item["source"],
                    "title": item["title"],
                    "published_at": macro_ts.strftime("%Y-%m-%d %H:%M UTC") if macro_ts else "unknown",
                    "macro_tag": item["macro_tag"],
                }
            )
    except Exception:
        sector_highlights = []
        macro_highlights = []

    return {
        "direction": direction,
        "intensity": round(float(intensity), 3),
        "article_count": total_count,
        "window_hours": 48,
        "category_breakdown": dict(category_counts),
        "weighted_stats": {
            "weighted_bullish": round(weighted_bullish, 3),
            "weighted_bearish": round(weighted_bearish, 3),
            "weighted_raw": round(weighted_raw, 3),
        },
        "policy_events": policy_events,
        "articles": articles_out,
        "market_context": market_context,
        "sector_highlights": sector_highlights,
        "macro_highlights": macro_highlights,
        "signal_freshness": {
            "timestamp_quality": {
                "undated_rows": int(undated_rows),
                "quality_factor": round(float(timestamp_quality_factor), 3),
                "recency_eligible_rows": int(len(rows_recent)),
            }
        },
    }


def build_news(con, symbol: str, max_news: int = 5) -> dict:
    return compute_news_features(symbol, con)


def load_recent_news_titles(symbol: str, con, limit: int = 15):
    try:
        rows = con.execute(
            """
            SELECT title
            FROM news_items
            WHERE symbol = ? AND lane = 'symbol'
            ORDER BY published_at DESC
            LIMIT ?
            """,
            (symbol, limit),
        ).fetchall()
    except Exception:
        rows = con.execute(
            """
            SELECT title
            FROM news_items
            WHERE symbol = ?
            ORDER BY published_at DESC
            LIMIT ?
            """,
            (symbol, limit),
        ).fetchall()

    return [r[0] for r in rows]


def compute_keyword_sentiment(rows: List[tuple]) -> Dict[str, object]:
    # Deprecated path retained for compatibility; uses simple counts.
    bullish_score = 0
    bearish_score = 0

    for _, title, _, _ in rows:
        t = (title or "").lower()
        for word in BULLISH_KEYWORDS:
            if word in t:
                bullish_score += 1
        for word in BEARISH_KEYWORDS:
            if word in t:
                bearish_score += 1

    article_count = len(rows)
    raw_score = bullish_score - bearish_score
    direction = "neutral"
    if raw_score > 0:
        direction = "bullish"
    elif raw_score < 0:
        direction = "bearish"

    intensity = min(1.0, abs(raw_score) / (article_count or 1))

    return {
        "direction": direction,
        "intensity": intensity,
        "bull_count": bullish_score,
        "bear_count": bearish_score,
        "neutral_count": 0,
        "evaluated_articles": article_count,
    }

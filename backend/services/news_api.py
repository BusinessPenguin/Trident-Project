"""EventRegistry / NewsAPI.ai client for Project Trident with streamlined filtering."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime as _dt, timezone as _tz
from typing import Dict, List, Optional
import math
import os
import hashlib

from backend.features.news_source_quality import get_source_quality, source_family
from backend.services.http_client import request_json_with_retries

ACTIVE_SYMBOLS = {"BTC-USD", "ETH-USD", "SOL-USD", "XRP-USD", "SUI-USD"}

MACRO_CONTEXT_TERMS = [
    "trump", "tariff", "tariffs", "sanction", "sanctions", "election",
    "geopolitic", "geopolitical", "war", "conflict", "greenland", "nato", "ukraine",
    "china", "taiwan", "middle east", "iran", "israel", "gaza",
    "opec", "oil", "oil shock", "energy shock", "embargo",
    "federal reserve", "fed", "central bank", "interest rate",
    "rate hike", "rate cut", "inflation", "recession", "gdp",
    "treasury", "bond yield", "yields", "trade war",
]

MACRO_QUERY_SEEDS = [
    "war",
    "invasion",
    "sanctions",
    "tariffs",
    "trade war",
    "central bank",
    "interest rate",
    "inflation",
    "recession",
    "geopolitical",
    "election",
    "oil",
    "energy shock",
]

MACRO_TRUSTED_SOURCES = [
    "reuters",
    "bloomberg",
    "associated press",
    "ap news",
    "apnews.com",
    "wall street journal",
    "wsj",
    "financial times",
    "ft.com",
    "cnbc",
    "marketwatch",
    "investing.com",
    "fxstreet",
    "seeking alpha",
    "barron's",
    "axios",
    "the hill",
    "the economist",
    "bbc",
    "bbc news",
    "new york times",
    "nytimes",
    "the guardian",
    "washington post",
    "politico",
    "fortune",
    "coindesk",
    "the block",
    "beincrypto",
    "beincrypto.com",
    "nikkei",
    "nikkei asia",
    "zerohedge",
    "zerohedge.com",
]

SOURCE_QUALITY = {
    "Reuters": 3,
    "Bloomberg": 3,
    "Yahoo! Finance": 3,
    "CoinDesk": 3,
    "Cointelegraph": 3,
    "Markets Insider": 2,
    "The Motley Fool": 2,
}

BAD_SOURCES = [
    "ambcrypto",
    "coincu news",
    "cryptopotato",
    "cryptopolitan",
    "thecoinrise.com",
    "coingape",
    "coinrank",
    "cointurk news",
    "the coin republic",
    "bitcoin sistemi",
    "watcherguru",
    "u.today",
    "coingape",
    "cryptodaily",
    "dailycoin",
    "blockchain.news",
]

SPAM_KEYWORDS = [
    "100x", "200x", "500x", "700x", "10x gains", "viral",
    "moon", "parabolic", "presale", "airdrop", "whitelist",
    "token sale", "pre-sale", "pump", "dump", "smashes"
]

LIST_KEYWORDS = [
    "top ", "top-", "winners", "losers", "weekly", "market update",
    "best crypto", "top crypto"
]

MACRO_NOISE = [
    "nvda", "nvidia", "federal reserve", "fed ",
    "inflation", "jobs report", "macro"
]


POLICY_TERMS = [
    "congress", "senate", "house", "vote", "voted", "bill", "legislation",
    "regulation", "regulator", "committee", "hearing",
    "sec", "cftc", "treasury", "white house", "federal",
]

CRYPTO_TERMS = [
    "crypto", "cryptocurrency", "digital asset", "digital assets", "blockchain",
    "bitcoin", "btc", "ethereum", "eth", "solana", "sol", "xrp", "ripple", "sui",
    "token", "defi", "stablecoin", "exchange", "web3", "coinbase",
]

SECTOR_ANCHORS = [
    "crypto", "cryptocurrency", "digital asset", "stablecoin", "bitcoin etf", "spot etf",
]

SECTOR_CO_TERMS = [
    "sec", "cftc", "congress", "bill", "act", "legislation", "guidance", "enforcement", "lawsuit",
    "exchange", "custody", "depeg", "etf inflow", "etf outflow",
    "funding", "open interest", "liquidations",
    "hack", "exploit", "outage",
]

SECTOR_PRIMARY_MAX_PAGES = 2
SECTOR_FALLBACK_MAX_PAGES = 2
SECTOR_PAGE_SIZE = 30
SECTOR_FALLBACK_QUERY = "crypto OR stablecoin OR \"bitcoin ETF\" OR \"spot ETF\""
SECTOR_PRIMARY_SEEDS = [
    "crypto regulation",
    "stablecoin policy",
    "bitcoin etf",
    "crypto enforcement",
]
SECTOR_SUPPLEMENTAL_SEEDS = [
    "crypto regulation",
    "stablecoin legislation",
    "bitcoin etf inflow",
    "crypto exchange lawsuit",
    "crypto hack exploit",
    "open interest liquidations",
]
SECTOR_SUPPLEMENTAL_MAX_QUERIES = 4
SECTOR_SUPPLEMENTAL_PAGE_SIZE = 25
SECTOR_MIN_KEEP = 4

SYMBOL_MAX_PAGES = 3
SYMBOL_PAGE_SIZE = 40

MACRO_TAG_RULES = {
    "macro_liquidity": [
        "fomc", "fed", "qt", "qe", "balance sheet", "reserves", "reverse repo", "real rates",
        "treasury buyback", "repo", "overnight reverse repo",
    ],
    "macro_policy_us": [
        "shutdown", "debt ceiling", "spending bill", "appropriations", "treasury issuance",
        "refunding", "deficit", "fiscal", "house", "senate", "continuing resolution",
    ],
    "macro_risk_regime": [
        "risk-off", "volatility spike", "credit spreads", "banking stress", "liquidity crunch",
        "nonfarm payrolls", "nfp", "cpi", "pce", "unemployment", "treasury auction",
        "bond yield", "yields", "vix",
    ],
    "macro_geopolitics": [
        "sanctions", "tariffs", "trade restrictions", "trade war",
        "conflict escalation", "war", "invasion", "ceasefire",
        "trump", "white house", "election",
        "middle east", "iran", "israel", "gaza", "ukraine",
        "china", "taiwan", "nato", "opec", "oil", "energy shock", "embargo",
    ],
}

MACRO_TAG_PRIORITY = [
    "macro_liquidity",
    "macro_policy_us",
    "macro_risk_regime",
    "macro_geopolitics",
]

MACRO_PRIMARY_MAX_PAGES = 3
MACRO_FALLBACK_MAX_PAGES = 3
MACRO_PAGE_SIZE = 40
MACRO_FALLBACK_QUERY = "FOMC OR Fed OR \"Treasury issuance\" OR \"balance sheet\" OR sanctions OR tariffs"
MACRO_PRIMARY_SEED_MAX_QUERIES = 6
SECTOR_PRIMARY_SEED_MAX_QUERIES = 4
MACRO_SUPPLEMENTAL_SEEDS = [
    "Trump tariffs",
    "US election policy",
    "Middle East conflict oil",
    "Ukraine sanctions",
    "China Taiwan trade",
    "Fed balance sheet",
    "FOMC rate decision",
    "Treasury issuance",
    "nonfarm payrolls",
    "CPI inflation",
    "PCE inflation",
    "bond yields",
    "debt ceiling",
    "tariffs sanctions",
]
MACRO_SUPPLEMENTAL_MAX_QUERIES = 6
MACRO_SUPPLEMENTAL_PAGE_SIZE = 30
MACRO_MIN_KEEP = 10
MACRO_MIN_FAMILY_DIVERSITY = 5
MACRO_DIVERSITY_RESCUE_MAX_QUERIES = 4
MACRO_DIVERSITY_RESCUE_PAGE_SIZE = 30

MACRO_TAG_FALLBACK_RULES = {
    "macro_liquidity": [
        "federal reserve", "fed", "fomc", "rate hike", "rate cut", "interest rate",
        "inflation", "cpi", "pce", "real rates", "treasury yield", "bond yield",
        "liquidity", "repo", "reverse repo", "balance sheet", "qt", "qe",
    ],
    "macro_policy_us": [
        "white house", "congress", "senate", "house", "bill", "legislation",
        "appropriations", "debt ceiling", "shutdown", "fiscal", "treasury issuance",
        "deficit", "election",
    ],
    "macro_risk_regime": [
        "risk-off", "risk appetite", "volatility", "vix", "selloff", "drawdown",
        "credit spread", "banking stress", "liquidity crunch", "recession", "gdp",
        "jobs report", "nonfarm payrolls", "unemployment",
    ],
    "macro_geopolitics": [
        "trump", "tariff", "tariffs", "sanction", "sanctions", "trade war", "war",
        "conflict", "invasion", "ceasefire", "middle east", "iran", "israel", "gaza",
        "ukraine", "china", "taiwan", "nato", "opec", "oil", "energy shock",
    ],
}

# Secondary, trusted-source-only fallback to reduce false negatives in macro tagging.
# This pass is only used when strict/fallback tag rules fail and source is trusted.
MACRO_TRUSTED_TAG_FALLBACK_RULES = {
    "macro_liquidity": [
        "monetary policy", "central bank", "interest rates", "rate decision",
        "rate outlook", "policy rate", "bond market", "treasury yields", "yield curve",
        "inflation outlook", "cpi", "pce", "jobs data", "labor market",
        "liquidity", "money markets",
    ],
    "macro_policy_us": [
        "white house", "administration", "congress", "senate", "house", "lawmakers",
        "budget", "fiscal policy", "government funding", "spending", "debt ceiling",
        "treasury secretary", "executive order",
    ],
    "macro_risk_regime": [
        "risk sentiment", "risk appetite", "volatility", "market stress", "drawdown",
        "sell-off", "selloff", "recession fears", "slowdown", "financial conditions",
        "credit conditions", "banking risk",
    ],
    "macro_geopolitics": [
        "geopolitical", "diplomacy", "tariff threats", "trade tensions", "sanctions",
        "conflict", "ceasefire", "military", "middle east", "ukraine", "china",
        "taiwan", "opec", "oil prices", "energy markets", "trump",
    ],
}


def is_bad_source(src: str) -> bool:
    return src in BAD_SOURCES


def is_spam(title: str) -> bool:
    t = (title or "").lower()
    return any(k in t for k in SPAM_KEYWORDS)


def is_listicle(title: str) -> bool:
    t = (title or "").lower()
    return any(k in t for k in LIST_KEYWORDS)


def is_macro_noise(title: str) -> bool:
    t = (title or "").lower()
    return any(k in t for k in MACRO_NOISE)


def is_btc_relevant(title: str) -> bool:
    """
    A headline is BTC-relevant if it mentions:
    - bitcoin / btc
    - crypto markets
    - crypto industry
    - miners / mining
    - exchanges (if crypto-related)
    - ETFs with digital asset context
    """
    if not title:
        return False

    t = title.lower()

    btc_terms = [
        "bitcoin", "btc", "satoshi",
        "crypto", "cryptocurrency",
        "digital asset", "digital currency",
        "miners", "mining",
        "spot etf", "btc etf", "crypto etf",
        "blockchain"
    ]

    return any(term in t for term in btc_terms)


def contains_other_assets(symbol: str, title: str) -> bool:
    """Return True only if the title mentions other major assets EXCEPT the target."""
    t = (title or "").lower()
    target = symbol.split("-")[0].lower()
    for asset in OTHER_ASSETS:
        if asset == target:
            continue
        if asset.lower() in t:
            return True
    return False


def is_macro_trusted_source(src: str) -> bool:
    if not src:
        return False
    s = src.lower()
    return any(token in s for token in MACRO_TRUSTED_SOURCES)


def compute_relevance(symbol: str, title: str) -> int:
    """
    Symbol-agnostic relevance for the Phase 2E universe (BTC, ETH, SOL, XRP, SUI).
    Score range: 0–3; market-wide crypto headlines count as relevant.
    """
    if not title:
        return 0
    t = title.lower()

    crypto_terms = [
        "crypto", "cryptocurrency", "digital asset", "digital currency",
        "blockchain", "exchange", "etf", "miners", "market", "trading",
        "defi", "web3",
    ]
    policy_terms = POLICY_TERMS

    base = symbol.split("-")[0].lower()
    ticker = base.upper().replace("USD", "")
    names = SYMBOL_NAME_MAP.get(symbol) or []
    name_hits = [n.lower() for n in names]

    score = 0

    # Direct mention (name/ticker/base)
    if base in t or (ticker and ticker.lower() in t) or any(n in t for n in name_hits):
        score += 2

    # General crypto headlines
    if any(term in t for term in crypto_terms + policy_terms):
        score += 1

    # Market linkage (BTC mention + target)
    if ("bitcoin" in t or "btc" in t) and (base in t or (ticker and ticker.lower() in t)):
        score += 1

    return min(score, 3)


def _policy_candidate(title: str, summary: str) -> bool:
    text = f"{title or ''} {summary or ''}".lower()
    if not text.strip():
        return False
    return any(term in text for term in POLICY_TERMS)


def _has_crypto_context(title: str, summary: str, ai_meta: Dict) -> bool:
    text = f"{title or ''} {summary or ''}".lower()
    if any(term in text for term in CRYPTO_TERMS):
        return True
    cats = ai_meta.get("categories") or []
    cat_text = " ".join(str(c) for c in cats).lower()
    return any(term in cat_text for term in CRYPTO_TERMS)


@dataclass
class NewsArticle:
    ts: Optional[_dt]
    source: str
    url: str
    title: str
    polarity: float
    ai_meta: Optional[Dict] = None
    macro_tag: Optional[str] = None


SYMBOL_NAME_MAP = {
    "BTC-USD": ["Bitcoin", "BTC"],
    "ETH-USD": ["Ethereum", "ETH"],
    "SOL-USD": ["Solana", "SOL"],
    "XRP-USD": ["XRP", "Ripple"],
    "SUI-USD": ["Sui", "SUI"],
}


def build_keyword_query(symbol: str) -> str:
    names = SYMBOL_NAME_MAP.get(symbol)
    if not names:
        base = symbol.split("-")[0]
        names = [base]
    return " OR ".join(names)


def build_macro_query() -> str:
    terms = []
    for term in MACRO_CONTEXT_TERMS:
        if " " in term:
            terms.append(f"\"{term}\"")
        else:
            terms.append(term)
    return " OR ".join(terms)


def _window_days(window_hours: int) -> int:
    try:
        hours = max(1, int(window_hours))
    except Exception:
        hours = 72
    return max(1, min(30, int(math.ceil(hours / 24))))


def _normalize_text(title: str, summary: str) -> str:
    return f"{title or ''} {summary or ''}".lower()


def _extract_label_text(ai_meta: Optional[Dict]) -> str:
    if not isinstance(ai_meta, dict):
        return ""
    out: List[str] = []
    for key in ("categories", "topics", "concepts", "entities"):
        items = ai_meta.get(key) or []
        if not isinstance(items, list):
            continue
        for item in items:
            if isinstance(item, str):
                out.append(item)
                continue
            if not isinstance(item, dict):
                continue
            label = item.get("label")
            if isinstance(label, dict):
                lbl = label.get("eng")
                if isinstance(lbl, str) and lbl:
                    out.append(lbl)
            elif isinstance(label, str) and label:
                out.append(label)
            for alt in ("name", "title"):
                v = item.get(alt)
                if isinstance(v, str) and v:
                    out.append(v)
    return " ".join(out).lower()


def _context_text(title: str, summary: str, ai_meta: Optional[Dict]) -> str:
    return f"{title or ''} {summary or ''} {_extract_label_text(ai_meta)}".lower()


def _matches_any(text: str, terms: List[str]) -> bool:
    if not text:
        return False
    for term in terms:
        if term in text:
            return True
    return False


def classify_macro_tag(title: str, summary: str, ai_meta: Optional[Dict] = None) -> Optional[str]:
    text = _context_text(title, summary, ai_meta)
    if not text.strip():
        return None
    for tag in MACRO_TAG_PRIORITY:
        terms = MACRO_TAG_RULES.get(tag) or []
        if _matches_any(text, terms):
            return tag
    # Fallback pass: keep deterministic tagging for broad macro rows that miss strict terms.
    for tag in MACRO_TAG_PRIORITY:
        terms = MACRO_TAG_FALLBACK_RULES.get(tag) or []
        if _matches_any(text, terms):
            return tag
    if _matches_any(text, MACRO_CONTEXT_TERMS):
        return "macro_geopolitics"
    return None


def classify_macro_tag_for_source(
    source: str,
    title: str,
    summary: str,
    ai_meta: Optional[Dict] = None,
) -> tuple[Optional[str], bool]:
    """
    Source-aware macro tag classifier.
    Returns (tag, used_trusted_fallback).
    """
    tag = classify_macro_tag(title, summary, ai_meta)
    if tag:
        return tag, False
    if not is_macro_trusted_source(source):
        return None, False
    text = _context_text(title, summary, ai_meta)
    if not text.strip():
        return None, False
    for candidate in MACRO_TAG_PRIORITY:
        terms = MACRO_TRUSTED_TAG_FALLBACK_RULES.get(candidate) or []
        if _matches_any(text, terms):
            return candidate, True
    return None, False


def _rotating_seeds(seeds: List[str], max_queries: int, rotation_key: Optional[str] = None) -> List[str]:
    if not seeds or max_queries <= 0:
        return []
    key_material = (
        rotation_key
        or os.getenv("TRIDENT_RUN_ID")
        or os.getenv("TRIDENT_ASOF_UTC")
        or "default_seed_rotation"
    )
    digest = hashlib.sha256(str(key_material).encode("utf-8")).hexdigest()
    start = int(digest[:8], 16) % len(seeds)
    take = min(len(seeds), max_queries)
    return [seeds[(start + i) % len(seeds)] for i in range(take)]


def _rank_articles_by_quality(articles: List[NewsArticle]) -> List[NewsArticle]:
    return sorted(
        articles,
        key=lambda a: (
            -get_source_quality(a.source),
            -(a.ts.timestamp() if a.ts else 0.0),
        ),
    )


def _apply_quality_overflow(
    articles: List[NewsArticle],
    base_limit: int,
    overflow_limit: int,
    quality_floor: int,
) -> List[NewsArticle]:
    ranked = _rank_articles_by_quality(articles)
    if base_limit <= 0:
        base_limit = len(ranked)
    base = ranked[:base_limit]
    if overflow_limit <= 0:
        return base
    extra: List[NewsArticle] = []
    for art in ranked[base_limit:]:
        if get_source_quality(art.source) >= quality_floor:
            extra.append(art)
            if len(extra) >= overflow_limit:
                break
    return base + extra


def _merge_raw_articles_unique(raw_groups: List[List[dict]]) -> List[dict]:
    merged: List[dict] = []
    seen = set()
    for group in raw_groups:
        for article in group:
            if not isinstance(article, dict):
                continue
            key = (
                article.get("url")
                or article.get("link")
                or article.get("clean_url")
                or article.get("uri")
                or article.get("eventUri")
                or article.get("title")
            )
            if not key or key in seen:
                continue
            seen.add(key)
            merged.append(article)
    return merged


def _raw_article_ts(article: dict) -> Optional[_dt]:
    if not isinstance(article, dict):
        return None
    published_at_str = (
        article.get("dateTimePub")
        or article.get("dateTime")
        or None
    )
    if not published_at_str:
        d = article.get("date")
        t = article.get("time")
        if d and t:
            published_at_str = f"{d}T{t}Z"
    if not published_at_str:
        return None
    try:
        dt = _dt.fromisoformat(str(published_at_str).replace("Z", "+00:00"))
    except Exception:
        return None
    if dt.tzinfo is None:
        return dt.replace(tzinfo=_tz.utc)
    return dt.astimezone(_tz.utc)


def _apply_since_watermark(raw_articles: List[dict], since_ts: Optional[_dt]) -> tuple[List[dict], int]:
    if since_ts is None:
        return raw_articles, 0
    watermark = since_ts if since_ts.tzinfo else since_ts.replace(tzinfo=_tz.utc)
    kept: List[dict] = []
    rejected = 0
    for article in raw_articles:
        ts = _raw_article_ts(article)
        if ts is None:
            kept.append(article)
            continue
        if ts < watermark:
            rejected += 1
            continue
        kept.append(article)
    return kept, rejected


def _fetch_articles_by_seed_fanout(
    api_key: str,
    seeds: List[str],
    window_hours: int,
    max_queries: int,
    max_pages_per_seed: int,
    size: int,
    debug: bool,
    rotation_key: Optional[str] = None,
) -> List[dict]:
    raw_groups: List[List[dict]] = []
    for seed in _rotating_seeds(seeds, max_queries, rotation_key=rotation_key):
        raw_groups.append(
            _fetch_articles_by_keyword(
                api_key,
                seed,
                window_hours,
                max_pages=max_pages_per_seed,
                size=size,
                debug=debug,
            )
        )
    return _merge_raw_articles_unique(raw_groups)


def _select_diverse_macro_articles(
    articles: List[NewsArticle],
    desired_count: int,
) -> List[NewsArticle]:
    """
    Select macro rows with deterministic diversity priority:
    1) unique source families
    2) then unique macro tags
    3) then quality/recency fill
    """
    ranked = _rank_articles_by_quality(articles)
    if desired_count <= 0 or not ranked:
        return []
    target = min(desired_count, len(ranked))

    selected: List[NewsArticle] = []
    selected_keys = set()
    seen_families = set()
    seen_tags = set()

    for art in ranked:
        key = art.url or art.title
        family = source_family(art.source)
        if key in selected_keys or family in seen_families:
            continue
        selected.append(art)
        selected_keys.add(key)
        seen_families.add(family)
        seen_tags.add(art.macro_tag or "unknown")
        if len(selected) >= target:
            return selected

    for art in ranked:
        key = art.url or art.title
        tag = art.macro_tag or "unknown"
        if key in selected_keys or tag in seen_tags:
            continue
        selected.append(art)
        selected_keys.add(key)
        seen_tags.add(tag)
        if len(selected) >= target:
            return selected

    for art in ranked:
        key = art.url or art.title
        if key in selected_keys:
            continue
        selected.append(art)
        selected_keys.add(key)
        if len(selected) >= target:
            return selected

    return selected


def _select_diverse_sector_articles(
    articles: List[NewsArticle],
    desired_count: int,
) -> List[NewsArticle]:
    ranked = _rank_articles_by_quality(articles)
    if desired_count <= 0 or not ranked:
        return []
    target = min(desired_count, len(ranked))

    selected: List[NewsArticle] = []
    selected_keys = set()
    seen_families = set()

    for art in ranked:
        key = art.url or art.title
        family = source_family(art.source)
        if key in selected_keys or family in seen_families:
            continue
        selected.append(art)
        selected_keys.add(key)
        seen_families.add(family)
        if len(selected) >= target:
            return selected

    for art in ranked:
        key = art.url or art.title
        if key in selected_keys:
            continue
        selected.append(art)
        selected_keys.add(key)
        if len(selected) >= target:
            return selected
    return selected


def _fetch_articles_by_keyword(
    api_key: str,
    keyword: str,
    window_hours: int,
    max_pages: int = 4,
    size: int = 50,
    debug: bool = False,
) -> List[dict]:
    url = "https://eventregistry.org/api/v1/article/getArticles"
    params = {
        "apiKey": api_key,
        "keyword": keyword,
        "keywordOper": "or",
        "lang": "eng",
        "resultType": "articles",
        "dataType": ["news"],
        "articlesSortBy": "date",
        "articlesSortByAsc": False,
        "forceMaxDataTimeWindow": _window_days(window_hours),
        "size": size,
    }

    all_articles: List[dict] = []
    retry_total = 0
    for page in range(1, max_pages + 1):
        params["page"] = page
        try:
            data, meta = request_json_with_retries(
                url,
                params=params,
                timeout=10.0,
                max_attempts=4,
                retry_budget_seconds=25.0,
                backoff_base_seconds=0.5,
                backoff_cap_seconds=3.0,
                seed=f"eventregistry:keyword:{keyword}:page:{page}",
            )
            retry_total += int(meta.get("retries") or 0)
        except Exception:
            if debug:
                print(f"[news:pull][debug] request error keyword page={page}")
            break
        if debug and isinstance(data, dict) and data.get("error"):
            print(f"[news:pull][debug] api error page={page}: {data.get('error')}")
        page_articles = (
            data.get("articles", {}).get("results", [])
            or data.get("articles")
            or data.get("results")
            or []
        )
        if debug:
            print(f"[news:pull][debug] keyword page={page} returned {len(page_articles)}")
        if not page_articles:
            break
        # Guard against unexpected non-dict payloads.
        all_articles.extend([a for a in page_articles if isinstance(a, dict)])

    if debug and retry_total > 0:
        print(f"[news:pull][debug] keyword retry_total={retry_total}")

    return all_articles


def _article_to_news(article: dict, url_suffix: str | None = None, context_type: str | None = None) -> Optional[NewsArticle]:
    title = article.get("title") or ""
    raw_source = article.get("source") or {}
    source_name = (
        raw_source.get("title")
        or raw_source.get("name")
        or raw_source.get("clean_url")
        or "Unknown"
    ).strip()
    source_name_clean = source_name.lower()

    url_val = (
        article.get("url")
        or article.get("link")
        or article.get("clean_url")
        or None
    )

    published_at_str = (
        article.get("dateTimePub")
        or article.get("dateTime")
        or None
    )
    if not published_at_str:
        d = article.get("date")
        t = article.get("time")
        if d and t:
            published_at_str = f"{d}T{t}Z"

    if not url_val:
        return None
    if url_suffix:
        url_val = f"{url_val}{url_suffix}"

    if source_name_clean in BAD_SOURCES:
        return None
    if is_spam(title) or is_listicle(title):
        return None

    ai_meta = _extract_ai_meta(article, context_type=context_type)
    ts: Optional[_dt] = None
    ts_quality = "unknown"
    if published_at_str:
        try:
            ts = _dt.fromisoformat(str(published_at_str).replace("Z", "+00:00"))
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=_tz.utc)
            else:
                ts = ts.astimezone(_tz.utc)
            ts_quality = "provided"
        except Exception:
            ts = None
            ts_quality = "invalid"
    ai_meta["timestamp_quality"] = ts_quality
    return NewsArticle(ts, source_name, url_val, title, 0.0, ai_meta)


def fetch_sector_news(
    api_key: str,
    window_hours: int = 72,
    max_count: int = 120,
    overflow_good_count: int = 0,
    quality_floor: int = 2,
    extra_pages: int = 0,
    since_ts: Optional[_dt] = None,
    stale_mode: bool = False,
    debug: bool = False,
    rotation_key: Optional[str] = None,
    stats: Optional[Dict[str, int]] = None,
) -> List[NewsArticle]:
    base_quality_floor = int(max(0, quality_floor))
    relaxed_quality_floor = max(1, base_quality_floor - 1) if stale_mode else base_quality_floor
    keyword = " OR ".join(
        [f"\"{t}\"" if " " in t else t for t in SECTOR_ANCHORS]
    )
    max_pages = SECTOR_PRIMARY_MAX_PAGES + max(0, int(extra_pages))
    primary_keyword_raw = _fetch_articles_by_keyword(
        api_key,
        keyword,
        window_hours,
        max_pages=max_pages,
        size=SECTOR_PAGE_SIZE,
        debug=debug,
    )
    primary_seed_queries = SECTOR_PRIMARY_SEED_MAX_QUERIES + (2 if stale_mode else 0)
    primary_seed_raw = _fetch_articles_by_seed_fanout(
        api_key=api_key,
        seeds=SECTOR_PRIMARY_SEEDS,
        window_hours=window_hours,
        max_queries=primary_seed_queries,
        max_pages_per_seed=1,
        size=SECTOR_PAGE_SIZE,
        debug=debug,
        rotation_key=rotation_key,
    )
    raw_articles = _merge_raw_articles_unique([primary_keyword_raw, primary_seed_raw])

    fallback_raw: List[dict] = []
    if not raw_articles:
        # Fallback: single compact query to keep cost low.
        fallback_raw = _fetch_articles_by_keyword(
            api_key,
            SECTOR_FALLBACK_QUERY,
            window_hours,
            max_pages=SECTOR_FALLBACK_MAX_PAGES,
            size=SECTOR_PAGE_SIZE,
            debug=debug,
        )
    raw_articles = _merge_raw_articles_unique([raw_articles, fallback_raw])
    raw_total_before_watermark = len(raw_articles)
    raw_articles, watermark_rejects = _apply_since_watermark(raw_articles, since_ts=since_ts)
    raw_total = len(raw_articles)
    rescue_pool: List[dict] = [a for a in raw_articles if isinstance(a, dict)]

    seen = set()
    filtered: List[NewsArticle] = []
    anchor_hits = 0
    co_term_hits = 0
    sample_anchor_rejects: List[str] = []
    sample_coterm_rejects: List[str] = []
    supplemental_queries = 0
    supplemental_raw = 0
    quality_rejects = 0
    dupe_rejects = 0
    relaxed_rescue_added = 0
    relaxed_quality_kept = 0
    trusted_relaxed_kept = 0
    for article in raw_articles:
        if not isinstance(article, dict):
            continue
        title = article.get("title") or ""
        ai_meta = _extract_ai_meta(article, context_type="sector")
        summary = ai_meta.get("summary") or ""
        text = _context_text(title, summary, ai_meta)
        if not _matches_any(text, SECTOR_ANCHORS):
            if len(sample_anchor_rejects) < 3:
                sample_anchor_rejects.append(title[:120])
            continue
        anchor_hits += 1
        if not _matches_any(text, SECTOR_CO_TERMS):
            if len(sample_coterm_rejects) < 3:
                sample_coterm_rejects.append(title[:120])
            continue
        co_term_hits += 1

        news = _article_to_news(article, url_suffix="#sector", context_type="sector")
        if not news:
            continue
        if get_source_quality(news.source) < base_quality_floor:
            quality_rejects += 1
            continue
        key = news.url or news.title
        if key in seen:
            dupe_rejects += 1
            continue
        seen.add(key)
        filtered.append(news)

    if len(filtered) < SECTOR_MIN_KEEP:
        for seed in _rotating_seeds(
            SECTOR_SUPPLEMENTAL_SEEDS,
            SECTOR_SUPPLEMENTAL_MAX_QUERIES,
            rotation_key=f"{rotation_key or ''}:sector_supp",
        ):
            supplemental_queries += 1
            extra_raw = _fetch_articles_by_keyword(
                api_key,
                seed,
                window_hours,
                max_pages=1,
                size=SECTOR_SUPPLEMENTAL_PAGE_SIZE,
                debug=debug,
            )
            supplemental_raw += len(extra_raw)
            raw_total += len(extra_raw)
            rescue_pool.extend([a for a in extra_raw if isinstance(a, dict)])
            for article in extra_raw:
                if not isinstance(article, dict):
                    continue
                title = article.get("title") or ""
                ai_meta = _extract_ai_meta(article, context_type="sector")
                summary = ai_meta.get("summary") or ""
                text = _context_text(title, summary, ai_meta)
                has_anchor = _matches_any(text, SECTOR_ANCHORS)
                has_coterm = _matches_any(text, SECTOR_CO_TERMS)
                if stale_mode:
                    gate_pass = has_anchor or has_coterm
                else:
                    gate_pass = has_anchor and has_coterm
                if not gate_pass:
                    continue
                news = _article_to_news(article, url_suffix="#sector", context_type="sector")
                if not news:
                    continue
                source_quality = get_source_quality(news.source)
                if source_quality < base_quality_floor:
                    if stale_mode and source_quality >= relaxed_quality_floor:
                        relaxed_quality_kept += 1
                        if is_macro_trusted_source(news.source):
                            trusted_relaxed_kept += 1
                    else:
                        quality_rejects += 1
                        continue
                key = news.url or news.title
                if key in seen:
                    dupe_rejects += 1
                    continue
                seen.add(key)
                filtered.append(news)

    # Relaxed rescue: if strict gates are thin, salvage anchor-matching rows from all
    # already-fetched payload (primary + supplemental) with quality floor intact.
    if len(filtered) < SECTOR_MIN_KEEP and rescue_pool:
        for article in rescue_pool:
            if not isinstance(article, dict):
                continue
            title = article.get("title") or ""
            ai_meta = _extract_ai_meta(article, context_type="sector")
            summary = ai_meta.get("summary") or ""
            text = _context_text(title, summary, ai_meta)
            has_anchor = _matches_any(text, SECTOR_ANCHORS)
            has_coterm = _matches_any(text, SECTOR_CO_TERMS)
            if stale_mode:
                gate_pass = has_anchor or has_coterm
            else:
                gate_pass = has_anchor
            if not gate_pass:
                continue
            news = _article_to_news(article, url_suffix="#sector", context_type="sector")
            if not news:
                continue
            source_quality = get_source_quality(news.source)
            if source_quality < base_quality_floor:
                if stale_mode and source_quality >= relaxed_quality_floor:
                    relaxed_quality_kept += 1
                    if is_macro_trusted_source(news.source):
                        trusted_relaxed_kept += 1
                else:
                    quality_rejects += 1
                    continue
            key = news.url or news.title
            if key in seen:
                dupe_rejects += 1
                continue
            seen.add(key)
            filtered.append(news)
            relaxed_rescue_added += 1
            if relaxed_rescue_added >= 12:
                break

    base_limit = min(max_count, SECTOR_PAGE_SIZE * SECTOR_PRIMARY_MAX_PAGES)
    filtered = _apply_quality_overflow(
        filtered,
        base_limit=base_limit,
        overflow_limit=int(max(0, overflow_good_count)),
        quality_floor=base_quality_floor,
    )
    desired_count = min(
        len(filtered),
        int(max(0, base_limit)) + int(max(0, overflow_good_count)) + (6 if stale_mode else 0),
    )
    filtered = _select_diverse_sector_articles(filtered, desired_count=desired_count)

    if stats is not None:
        stats.clear()
        stats.update(
            {
                "raw": int(len(primary_keyword_raw)),
                "primary_raw": int(len(primary_keyword_raw)),
                "primary_seed_raw": int(len(primary_seed_raw)),
                "fallback_raw": int(len(fallback_raw)),
                "raw_total_before_watermark": int(raw_total_before_watermark),
                "watermark_rejects": int(watermark_rejects),
                "raw_total": int(raw_total),
                "anchor_hits": int(anchor_hits),
                "coterm_hits": int(co_term_hits),
                "quality_rejects": int(quality_rejects),
                "dupe_rejects": int(dupe_rejects),
                "supplemental_queries": int(supplemental_queries),
                "supplemental_raw": int(supplemental_raw),
                "relaxed_rescue_added": int(relaxed_rescue_added),
                "relaxed_quality_kept": int(relaxed_quality_kept),
                "trusted_relaxed_kept": int(trusted_relaxed_kept),
                "quality_floor_used": int(base_quality_floor),
                "relaxed_quality_floor": int(relaxed_quality_floor),
                "stale_mode": int(bool(stale_mode)),
                "pre_ingest_kept": int(len(filtered)),
                "sources": int(len({(a.source or '').strip().lower() for a in filtered if a.source})),
                "kept": int(len(filtered)),
            }
        )

    if debug:
        print(
            "[news:pull_sector][debug] "
            f"raw={len(raw_articles)} anchor_hits={anchor_hits} "
            f"co_term_hits={co_term_hits} kept={len(filtered)} "
            f"supplemental_queries={supplemental_queries} supplemental_raw={supplemental_raw}"
        )
        if sample_anchor_rejects:
            print("[news:pull_sector][debug] sample_anchor_rejects:")
            for item in sample_anchor_rejects:
                print(f"  - {item}")
        if sample_coterm_rejects:
            print("[news:pull_sector][debug] sample_coterm_rejects:")
            for item in sample_coterm_rejects:
                print(f"  - {item}")

    return filtered


def fetch_macro_news(
    api_key: str,
    window_hours: int = 72,
    max_count: int = 120,
    overflow_good_count: int = 0,
    quality_floor: int = 2,
    extra_pages: int = 0,
    since_ts: Optional[_dt] = None,
    stale_mode: bool = False,
    debug: bool = False,
    rotation_key: Optional[str] = None,
    stats: Optional[Dict[str, int]] = None,
) -> List[NewsArticle]:
    base_quality_floor = int(max(0, quality_floor))
    relaxed_quality_floor = max(1, base_quality_floor - 1) if stale_mode else base_quality_floor
    tag_terms = []
    for tag in MACRO_TAG_PRIORITY:
        tag_terms.extend(MACRO_TAG_RULES.get(tag) or [])
    keyword = " OR ".join([f"\"{t}\"" if " " in t else t for t in tag_terms])
    max_pages = MACRO_PRIMARY_MAX_PAGES + max(0, int(extra_pages))
    primary_keyword_raw = _fetch_articles_by_keyword(
        api_key,
        keyword,
        window_hours,
        max_pages=max_pages,
        size=MACRO_PAGE_SIZE,
        debug=debug,
    )
    primary_seed_queries = MACRO_PRIMARY_SEED_MAX_QUERIES + (2 if stale_mode else 0)
    primary_seed_raw = _fetch_articles_by_seed_fanout(
        api_key=api_key,
        seeds=MACRO_QUERY_SEEDS,
        window_hours=window_hours,
        max_queries=primary_seed_queries,
        max_pages_per_seed=1,
        size=MACRO_PAGE_SIZE,
        debug=debug,
        rotation_key=rotation_key,
    )
    raw_articles = _merge_raw_articles_unique([primary_keyword_raw, primary_seed_raw])

    fallback_raw: List[dict] = []
    if not raw_articles or (stale_mode and len(raw_articles) < 20):
        # Fallback: single compact query to keep cost low.
        fallback_raw = _fetch_articles_by_keyword(
            api_key,
            MACRO_FALLBACK_QUERY,
            window_hours,
            max_pages=MACRO_FALLBACK_MAX_PAGES,
            size=MACRO_PAGE_SIZE,
            debug=debug,
        )
    raw_articles = _merge_raw_articles_unique([raw_articles, fallback_raw])
    raw_total_before_watermark = len(raw_articles)
    raw_articles, watermark_rejects = _apply_since_watermark(raw_articles, since_ts=since_ts)
    raw_total = len(raw_articles)

    seen = set()
    filtered: List[NewsArticle] = []
    tag_counts: Dict[str, int] = {}
    sample_tag_rejects: List[str] = []
    supplemental_queries = 0
    supplemental_raw = 0
    diversity_rescue_queries = 0
    diversity_rescue_raw = 0
    diversity_rescue_added = 0
    tag_rejects = 0
    quality_rejects = 0
    dupe_rejects = 0
    trusted_fallback_tags = 0
    relaxed_quality_kept = 0
    trusted_relaxed_kept = 0
    for article in raw_articles:
        if not isinstance(article, dict):
            continue
        title = article.get("title") or ""
        news = _article_to_news(article, url_suffix="#macro", context_type="macro")
        if not news:
            continue
        ai_meta = news.ai_meta or {}
        summary = ai_meta.get("summary") or ""
        macro_tag, used_trusted_fallback = classify_macro_tag_for_source(
            news.source, title, summary, ai_meta
        )
        if not macro_tag:
            tag_rejects += 1
            if len(sample_tag_rejects) < 3:
                sample_tag_rejects.append(title[:120])
            continue
        if used_trusted_fallback:
            trusted_fallback_tags += 1
        tag_counts[macro_tag] = tag_counts.get(macro_tag, 0) + 1
        source_quality = get_source_quality(news.source)
        if source_quality < base_quality_floor:
            if stale_mode and source_quality >= relaxed_quality_floor and is_macro_trusted_source(news.source):
                relaxed_quality_kept += 1
                trusted_relaxed_kept += 1
            else:
                quality_rejects += 1
                continue
        news.macro_tag = macro_tag
        key = news.url or news.title
        if key in seen:
            dupe_rejects += 1
            continue
        seen.add(key)
        filtered.append(news)

    if len(filtered) < MACRO_MIN_KEEP:
        supplemental_limit = MACRO_SUPPLEMENTAL_MAX_QUERIES + (2 if stale_mode else 0)
        for seed in _rotating_seeds(
            MACRO_SUPPLEMENTAL_SEEDS,
            supplemental_limit,
            rotation_key=f"{rotation_key or ''}:macro_supp",
        ):
            supplemental_queries += 1
            extra_raw = _fetch_articles_by_keyword(
                api_key,
                seed,
                window_hours,
                max_pages=1,
                size=MACRO_SUPPLEMENTAL_PAGE_SIZE,
                debug=debug,
            )
            supplemental_raw += len(extra_raw)
            for article in extra_raw:
                if not isinstance(article, dict):
                    continue
                title = article.get("title") or ""
                news = _article_to_news(article, url_suffix="#macro", context_type="macro")
                if not news:
                    continue
                ai_meta = news.ai_meta or {}
                summary = ai_meta.get("summary") or ""
                macro_tag, used_trusted_fallback = classify_macro_tag_for_source(
                    news.source, title, summary, ai_meta
                )
                if not macro_tag:
                    tag_rejects += 1
                    continue
                if used_trusted_fallback:
                    trusted_fallback_tags += 1
                source_quality = get_source_quality(news.source)
                if source_quality < base_quality_floor:
                    if stale_mode and source_quality >= relaxed_quality_floor and is_macro_trusted_source(news.source):
                        relaxed_quality_kept += 1
                        trusted_relaxed_kept += 1
                    else:
                        quality_rejects += 1
                        continue
                news.macro_tag = macro_tag
                key = news.url or news.title
                if key in seen:
                    dupe_rejects += 1
                    continue
                seen.add(key)
                filtered.append(news)
                tag_counts[macro_tag] = tag_counts.get(macro_tag, 0) + 1

    # Diversity rescue: bounded extra calls only when families are overly concentrated.
    seen_families = {source_family(a.source) for a in filtered if a.source}
    if len(seen_families) < MACRO_MIN_FAMILY_DIVERSITY:
        rescue_limit = MACRO_DIVERSITY_RESCUE_MAX_QUERIES + (2 if stale_mode else 0)
        for seed in _rotating_seeds(
            MACRO_QUERY_SEEDS,
            rescue_limit,
            rotation_key=f"{rotation_key or ''}:macro_rescue",
        ):
            diversity_rescue_queries += 1
            extra_raw = _fetch_articles_by_keyword(
                api_key,
                seed,
                window_hours,
                max_pages=1,
                size=MACRO_DIVERSITY_RESCUE_PAGE_SIZE,
                debug=debug,
            )
            diversity_rescue_raw += len(extra_raw)
            for article in extra_raw:
                if not isinstance(article, dict):
                    continue
                title = article.get("title") or ""
                news = _article_to_news(article, url_suffix="#macro", context_type="macro")
                if not news:
                    continue
                ai_meta = news.ai_meta or {}
                summary = ai_meta.get("summary") or ""
                macro_tag, used_trusted_fallback = classify_macro_tag_for_source(
                    news.source, title, summary, ai_meta
                )
                if not macro_tag:
                    tag_rejects += 1
                    continue
                if used_trusted_fallback:
                    trusted_fallback_tags += 1
                source_quality = get_source_quality(news.source)
                if source_quality < base_quality_floor:
                    if stale_mode and source_quality >= relaxed_quality_floor and is_macro_trusted_source(news.source):
                        relaxed_quality_kept += 1
                        trusted_relaxed_kept += 1
                    else:
                        quality_rejects += 1
                        continue
                fam = source_family(news.source)
                if fam in seen_families:
                    dupe_rejects += 1
                    continue
                news.macro_tag = macro_tag
                key = news.url or news.title
                if key in seen:
                    dupe_rejects += 1
                    continue
                seen.add(key)
                seen_families.add(fam)
                filtered.append(news)
                diversity_rescue_added += 1
                tag_counts[macro_tag] = tag_counts.get(macro_tag, 0) + 1
                if len(seen_families) >= MACRO_MIN_FAMILY_DIVERSITY:
                    break
            if len(seen_families) >= MACRO_MIN_FAMILY_DIVERSITY:
                break

    base_limit = min(max_count, MACRO_PAGE_SIZE * MACRO_PRIMARY_MAX_PAGES)
    desired_count = min(
        len(filtered),
        max(0, int(base_limit)) + max(0, int(overflow_good_count)),
    )
    filtered = _select_diverse_macro_articles(filtered, desired_count=desired_count)

    if stats is not None:
        stats.clear()
        stats.update(
            {
                "raw": int(len(primary_keyword_raw)),
                "primary_raw": int(len(primary_keyword_raw)),
                "primary_seed_raw": int(len(primary_seed_raw)),
                "fallback_raw": int(len(fallback_raw)),
                "raw_total_before_watermark": int(raw_total_before_watermark),
                "watermark_rejects": int(watermark_rejects),
                "raw_total": int(raw_total),
                "tagged": int(sum(tag_counts.values())),
                "tag_rejects": int(tag_rejects),
                "trusted_fallback_tags": int(trusted_fallback_tags),
                "quality_rejects": int(quality_rejects),
                "dupe_rejects": int(dupe_rejects),
                "supplemental_queries": int(supplemental_queries),
                "supplemental_raw": int(supplemental_raw),
                "rescue_queries": int(diversity_rescue_queries),
                "rescue_raw": int(diversity_rescue_raw),
                "rescue_added": int(diversity_rescue_added),
                "families": int(len({source_family(a.source) for a in filtered})),
                "sources": int(len({(a.source or '').strip().lower() for a in filtered if a.source})),
                "tags": int(len({a.macro_tag or 'unknown' for a in filtered})),
                "quality_floor_used": int(base_quality_floor),
                "relaxed_quality_floor": int(relaxed_quality_floor),
                "stale_mode": int(bool(stale_mode)),
                "relaxed_quality_kept": int(relaxed_quality_kept),
                "trusted_relaxed_kept": int(trusted_relaxed_kept),
                "pre_ingest_kept": int(len(filtered)),
                "kept": int(len(filtered)),
            }
        )

    if debug:
        fam_count = len({source_family(a.source) for a in filtered})
        tag_count = len({a.macro_tag or "unknown" for a in filtered})
        print(
            "[news:pull_macro][debug] "
            f"raw={len(raw_articles)} tagged={sum(tag_counts.values())} kept={len(filtered)} "
            f"families={fam_count} tags={tag_count} "
            f"trusted_fallback_tags={trusted_fallback_tags} "
            f"supplemental_queries={supplemental_queries} supplemental_raw={supplemental_raw} "
            f"rescue_queries={diversity_rescue_queries} rescue_raw={diversity_rescue_raw} "
            f"rescue_added={diversity_rescue_added}"
        )
        if tag_counts:
            print("[news:pull_macro][debug] tag_counts:")
            for tag, cnt in sorted(tag_counts.items(), key=lambda x: (-x[1], x[0])):
                print(f"  - {tag}: {cnt}")
        if sample_tag_rejects:
            print("[news:pull_macro][debug] sample_tag_rejects:")
            for item in sample_tag_rejects:
                print(f"  - {item}")

    return filtered


def _extract_ai_meta(article: dict, context_type: Optional[str] = None) -> Dict:
    ai_meta = {
        "sentimentScore": article.get("sentimentScore") or article.get("sentiment"),
        "relevance": article.get("relevance"),
        "eventUri": article.get("eventUri"),
        "categories": article.get("categories") or [],
        "concepts": article.get("concepts") or [],
        "topics": article.get("topics") or [],
        "entities": article.get("entities") or [],
        "summary": article.get("bodyAbstract") or article.get("summary") or "",
        "wgt": article.get("wgt"),
    }
    if context_type:
        ai_meta["context_type"] = context_type
    return ai_meta


def fetch_macro_context(api_key: str, max_count: int = 50) -> List[NewsArticle]:
    """
    Fetch macro/geopolitical context articles (context-only layer).
    These are stored under a dedicated symbol and used only when primary coverage is thin.
    """
    url = "https://eventregistry.org/api/v1/article/getArticles"

    params = {
        "apiKey": api_key,
        "keywordOper": "or",
        "lang": "eng",
        "resultType": "articles",
        "dataType": ["news"],
        "articlesSortBy": "date",
        "articlesSortByAsc": False,
        "forceMaxDataTimeWindow": 7,
        "size": 50,
    }

    all_articles: List[dict] = []
    seen = set()
    max_per_seed = max(5, max_count // max(len(MACRO_QUERY_SEEDS), 1))
    for seed in MACRO_QUERY_SEEDS:
        params["keyword"] = seed
        params["page"] = 1
        print(f"[news:pull][debug] requesting macro seed='{seed}' page=1")

        try:
            data, _meta = request_json_with_retries(
                url,
                params=params,
                timeout=10.0,
                max_attempts=4,
                retry_budget_seconds=20.0,
                backoff_base_seconds=0.5,
                backoff_cap_seconds=3.0,
                seed=f"macro_context:{seed}",
            )
        except Exception as e:
            print(f"[news:pull][warn] macro request error on seed='{seed}': {e}")
            continue
        page_articles = (
            data.get("articles", {}).get("results", [])
            or data.get("articles")
            or data.get("results")
            or []
        )

        print(f"[news:pull][debug] macro seed='{seed}' returned {len(page_articles)} articles")

        added = 0
        for article in page_articles:
            url_val = (
                article.get("url")
                or article.get("link")
                or article.get("clean_url")
                or ""
            )
            key = url_val or article.get("uri") or article.get("title") or ""
            if not key or key in seen:
                continue
            raw_source = article.get("source") or {}
            source_name = (
                raw_source.get("title")
                or raw_source.get("name")
                or raw_source.get("clean_url")
                or "Unknown"
            ).strip()
            if not is_macro_trusted_source(source_name):
                continue
            seen.add(key)
            all_articles.append(article)
            added += 1
            if added >= max_per_seed:
                break
        if len(all_articles) >= max_count:
            break

    all_articles.sort(
        key=lambda a: a.get("dateTimePub") or a.get("dateTime") or "",
        reverse=True,
    )
    if len(all_articles) > max_count:
        all_articles = all_articles[:max_count]

    filtered: List[NewsArticle] = []

    for article in all_articles:
        title = article.get("title") or ""

        raw_source = article.get("source") or {}
        source_name = (
            raw_source.get("title")
            or raw_source.get("name")
            or raw_source.get("clean_url")
            or "Unknown"
        ).strip()
        source_name_clean = source_name.lower()

        url_val = (
            article.get("url")
            or article.get("link")
            or article.get("clean_url")
            or None
        )

        published_at_str = (
            article.get("dateTimePub")
            or article.get("dateTime")
            or None
        )
        if not published_at_str:
            d = article.get("date")
            t = article.get("time")
            if d and t:
                published_at_str = f"{d}T{t}Z"

        if not url_val:
            continue
        # Ensure macro context rows don't collide with primary symbol URLs.
        url_val = f"{url_val}#macro"
        if source_name_clean in BAD_SOURCES:
            continue
        if is_spam(title) or is_listicle(title):
            continue

        ai_meta = _extract_ai_meta(article, context_type="macro")
        ts: Optional[_dt] = None
        ts_quality = "unknown"
        if published_at_str:
            try:
                ts = _dt.fromisoformat(str(published_at_str).replace("Z", "+00:00"))
                if ts.tzinfo is None:
                    ts = ts.replace(tzinfo=_tz.utc)
                else:
                    ts = ts.astimezone(_tz.utc)
                ts_quality = "provided"
            except Exception:
                ts = None
                ts_quality = "invalid"
        ai_meta["timestamp_quality"] = ts_quality
        summary = ai_meta.get("summary") or ""
        macro_tag = classify_macro_tag(title, summary, ai_meta)
        if not macro_tag:
            continue
        news = NewsArticle(ts, source_name, url_val, title, 0.0, ai_meta)
        news.macro_tag = macro_tag
        filtered.append(news)

    return filtered


def fetch_news_for_symbol(
    api_key: str,
    symbol: str,
    max_count: int = 50,
    debug: bool = False,
) -> List[NewsArticle]:
    """
    Fetch news for a symbol from EventRegistry with streamlined filtering.
    Returns a filtered list of NewsArticle ready for DB insertion by caller.
    """
    if symbol not in ACTIVE_SYMBOLS:
        print(f"[news:pull] SKIP {symbol}: not in active Phase 2E universe")
        return []

    keyword_str = build_keyword_query(symbol)
    url = "https://eventregistry.org/api/v1/article/getArticles"

    params = {
        "apiKey": api_key,
        "keyword": keyword_str,
        "lang": "eng",
        "resultType": "articles",
        "dataType": ["news"],
        "articlesSortBy": "date",
        "articlesSortByAsc": False,
        "forceMaxDataTimeWindow": 7,
        "size": SYMBOL_PAGE_SIZE,
    }

    all_articles: List[dict] = []
    def _dbg(msg: str) -> None:
        if debug:
            print(msg)

    for page in range(1, SYMBOL_MAX_PAGES + 1):
        params["page"] = page
        _dbg(f"[news:pull][debug] requesting page={page}")

        try:
            data, _meta = request_json_with_retries(
                url,
                params=params,
                timeout=10.0,
                max_attempts=4,
                retry_budget_seconds=20.0,
                backoff_base_seconds=0.5,
                backoff_cap_seconds=3.0,
                seed=f"symbol_news:{symbol}:page:{page}",
            )
        except Exception as e:
            print(f"[news:pull][warn] request error on page={page}: {e}")
            break
        page_articles = (
            data.get("articles", {}).get("results", [])
            or data.get("articles")
            or data.get("results")
            or []
        )

        _dbg(f"[news:pull][debug] page={page} returned {len(page_articles)} articles")

        if not page_articles:
            break

        all_articles.extend(page_articles)

    articles = all_articles
    _dbg(f"[news:pull][debug] total raw_articles={len(articles)} before filtering")
    for a in articles[:5]:
        src = (a.get("source_name") or a.get("clean_url") or "Unknown")
        ttl = (a.get("title") or "")[:140]
        _dbg(f"[news:pull][debug] raw: {src} - {ttl}")

    count_inserted = 0
    filtered: List[NewsArticle] = []

    for article in articles:
        _dbg(f"[news:pull][debug] symbol={symbol} article keys={list(article.keys())[:10]}")

        # --- Correct field mappings for NewsAPI.ai / EventRegistry ---
        title = article.get("title") or ""

        raw_source = article.get("source") or {}
        source_name = (
            raw_source.get("title")
            or raw_source.get("name")
            or raw_source.get("clean_url")
            or "Unknown"
        ).strip()
        source_name_clean = source_name.lower()

        url_val = (
            article.get("url")
            or article.get("link")
            or article.get("clean_url")
            or None
        )

        # Primary publication timestamp from EventRegistry
        published_at_str = (
            article.get("dateTimePub")   # actual pub time
            or article.get("dateTime")   # backup
            or None
        )

        # Reconstruct from date + time if needed
        if not published_at_str:
            d = article.get("date")
            t = article.get("time")
            if d and t:
                published_at_str = f"{d}T{t}Z"

        _dbg(
            f"[news:pull][debug] symbol={symbol} fields: "
            f"source_name={source_name!r}, url={url_val!r}, "
            f"published_at_str={published_at_str!r}, title={title[:80]!r}"
        )

        if not url_val:
            url_val = article.get("clean_url")
            if not url_val:
                _dbg(f"[news:pull][debug] symbol={symbol} SKIP: no url")
                continue

        # Basic filters (reuse existing helpers)
        if source_name_clean in BAD_SOURCES:
            _dbg(f"[news:pull][debug] symbol={symbol} SKIP: bad source {source_name!r}")
            continue
        if is_spam(title):
            _dbg(f"[news:pull][debug] symbol={symbol} SKIP: spam title={title[:80]!r}")
            continue
        if is_listicle(title):
            _dbg(f"[news:pull][debug] symbol={symbol} SKIP: listicle title={title[:80]!r}")
            continue

        # Relevance scoring
        relevance = compute_relevance(symbol, title)
        if relevance < 1:
            _dbg(f"[news:pull][debug] relevance={relevance} SKIP: {title[:80]!r}")
            continue
        else:
            _dbg(f"[news:pull][debug] relevance={relevance} KEEPING: {title[:80]!r}")

        ai_meta = _extract_ai_meta(article)
        ts: Optional[_dt] = None
        ts_quality = "unknown"
        if published_at_str:
            try:
                ts = _dt.fromisoformat(str(published_at_str).replace("Z", "+00:00"))
                if ts.tzinfo is None:
                    ts = ts.replace(tzinfo=_tz.utc)
                else:
                    ts = ts.astimezone(_tz.utc)
                ts_quality = "provided"
            except Exception as e:
                _dbg(f"[news:pull][debug] symbol={symbol} bad timestamp {published_at_str!r} ({e})")
                ts = None
                ts_quality = "invalid"
        ai_meta["timestamp_quality"] = ts_quality
        filtered.append(NewsArticle(ts, source_name, url_val, title, 0.0, ai_meta))
        count_inserted += 1
        _dbg(f"[news:pull][debug] symbol={symbol} INSERTED: {source_name!r} - {title[:80]!r}")

    return filtered

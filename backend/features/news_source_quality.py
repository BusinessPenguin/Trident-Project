"""
Source quality mapping for news items.

Score:
    3 = reputable / top-tier
    2 = medium quality
    1 = low quality / unknown (default)
    0 = flagged / avoid (optional future use)
"""

from __future__ import annotations

import re
from urllib.parse import urlparse

SOURCE_QUALITY = {
    # Reputable / top-tier crypto + finance
    "Bloomberg": 3,
    "Reuters": 3,
    "Associated Press": 3,
    "AP News": 3,
    "CoinDesk": 3,
    "The Block": 3,
    "CoinTelegraph": 3,
    "Fortune": 3,
    "Decrypt": 3,
    "Yahoo! Finance": 3,
    "Yahoo Finance": 3,
    "Business Insider": 3,
    "Markets Insider": 3,
    "Financial Times": 3,
    "The Wall Street Journal": 3,
    "Wall Street Journal": 3,
    "WSJ": 3,
    "The Economist": 3,
    "BBC": 3,
    "BBC News": 3,
    "The Guardian": 3,
    "New York Times": 3,
    "NYTimes": 3,
    "Washington Post": 3,
    "Politico": 3,
    "Nikkei": 3,
    "Nikkei Asia": 3,
    "CNBC": 3,
    "Forbes": 3,
    "The Motley Fool": 3,
    "CryptoRank": 3,
    "CoinMarketCap": 3,
    "crypto.news": 3,
    "MarketWatch": 3,
    "Bloomberg Markets": 3,
    "Investing.com": 2,
    "Investing.com India": 2,
    "Investing.com UK": 2,
    "Investing.com South Africa": 2,
    "FXStreet": 2,
    "Seeking Alpha": 2,
    "Barron's": 2,
    "The Hill": 2,
    "Axios": 2,
    "Aol": 2,

    # Medium quality / common crypto media
    "AMBCrypto": 2,
    "CryptoSlate": 2,
    "BeInCrypto": 2,
    "U.Today": 2,
    "CryptoPotato": 2,
    "CoinGape": 2,
    "CoinCu News": 2,
    "The Coin Republic": 2,
    "Analytics Insight": 2,
    "Crypto Economy": 2,
    "Blockonomi": 2,
    "TokenPost": 2,
    "CryptoBriefing": 2,
    "NewsBTC": 2,

    # Lower-tier / more sensational outlets (example)
    "DailyCoin": 1,
    "WatcherGuru": 1,
    "Crypto News Flash": 1,
    "CryptoTicker": 1,
    "Cointribune": 1,
    "Live Bitcoin News": 1,
    "NFT Plazas": 1,
    "FinanceFeeds": 1,
    "TechBullion": 1,
    "The Manila times": 1,
    "Coindoo": 1,
    "cryptodaily.co.uk": 1,

    # Banned / PR-heavy
    "GlobeNewswire": 0,
    "PRNewswire": 0,
    "BusinessWire": 0,
    "Press release": 0,
    "Times Bull": 0,
    "CryptoReporter": 0,
    # Add more as needed
}

_SOURCE_QUALITY_NORM = {k.lower(): v for k, v in SOURCE_QUALITY.items()}


def _normalize_source_text(name: str) -> str:
    text = (name or "").strip()
    if not text:
        return ""
    if text.startswith("http://") or text.startswith("https://"):
        parsed = urlparse(text)
        host = (parsed.netloc or "").strip().lower()
        if host.startswith("www."):
            host = host[4:]
        return host
    lowered = text.lower()
    if lowered.startswith("www."):
        lowered = lowered[4:]
    lowered = lowered.split("/", 1)[0]
    lowered = re.sub(r"\s+", " ", lowered).strip()
    return lowered


def canonical_source_name(name: str) -> str:
    """
    Map source variants/domains to a canonical display name used for quality.
    """
    norm = _normalize_source_text(name)
    if not norm:
        return ""

    # Domain-based canonicalization
    if "investing.com" in norm:
        return "Investing.com"
    if "wsj.com" in norm or "wall street journal" in norm:
        return "Wall Street Journal"
    if "bloomberg" in norm:
        return "Bloomberg"
    if "reuters" in norm:
        return "Reuters"
    if "coindesk" in norm:
        return "CoinDesk"
    if "theblock.co" in norm or norm == "the block":
        return "The Block"
    if "economist.com" in norm or norm == "the economist":
        return "The Economist"
    if "marketwatch" in norm:
        return "MarketWatch"
    if "seekingalpha" in norm:
        return "Seeking Alpha"
    if "ft.com" in norm:
        return "Financial Times"
    if "nytimes.com" in norm:
        return "New York Times"
    if "washingtonpost.com" in norm:
        return "Washington Post"
    if "apnews.com" in norm:
        return "AP News"
    if "axios.com" in norm:
        return "Axios"
    if "thehill.com" in norm:
        return "The Hill"
    if "cnbc.com" in norm:
        return "CNBC"
    if "bbc." in norm:
        return "BBC"
    if "fortune.com" in norm:
        return "Fortune"
    if "fxstreet.com" in norm:
        return "FXStreet"
    if "nikkei.com" in norm:
        return "Nikkei Asia"

    # Name-based canonicalization
    title = (name or "").strip()
    title_norm = re.sub(r"\s+", " ", title).strip()
    if title_norm in SOURCE_QUALITY:
        return title_norm

    return title_norm


def source_family(name: str) -> str:
    """
    Family key used for diversity controls (lower-case canonical source name).
    """
    canon = canonical_source_name(name)
    return canon.lower() if canon else ""


def get_source_quality(name: str) -> int:
    """
    Return quality score for a given source name.
    Default is 1 (low) if unknown.
    """
    if not name:
        return 1
    canon = canonical_source_name(name)
    if not canon:
        return 1
    exact = SOURCE_QUALITY.get(canon)
    if exact is not None:
        return exact
    return _SOURCE_QUALITY_NORM.get(canon.lower(), 1)

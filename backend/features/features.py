from backend.db.schema import get_connection
from backend.config.env import get_settings
from .technicals import build_technicals
from .fundamentals_features import build_fundamentals
from .news_features import build_news


def build(symbol: str, max_news: int = 5) -> dict:
    """
    Build combined features for a symbol.

    Returns a dict with three top-level keys:
        - 'technicals': dict
        - 'fundamentals': dict
        - 'news': dict

    This function MUST NOT raise if one or more modalities fail.
    It logs a warning and returns partial data instead.
    """
    settings = get_settings()
    con = get_connection(settings.database_path)

    tech: dict = {}
    funds: dict = {}
    news: dict = {}

    # Technicals
    try:
        tech = build_technicals(con, symbol) or {}
    except Exception as e:
        print(f"[features] WARNING: technicals failed for {symbol}: {e}")
        tech = {}

    # Fundamentals (may use technicals for ATR-based flags)
    try:
        funds = build_fundamentals(con, symbol) or {}
    except Exception as e:
        print(f"[features] WARNING: fundamentals failed for {symbol}: {e}")
        funds = {}

    # News
    try:
        news = build_news(con, symbol, max_news=max_news) or {}
    except Exception as e:
        print(f"[features] WARNING: news failed for {symbol}: {e}")
        news = {}

    return {
        "technicals": tech,
        "fundamentals": funds,
        "news": news,
    }

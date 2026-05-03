"""CoinGecko fundamentals client for Project Trident."""

from __future__ import annotations

from typing import Dict, Optional

from backend.services.http_client import request_json_with_retries

# Mapping from Trident symbol to CoinGecko coin id.
SYMBOL_TO_ID: Dict[str, str] = {
    "BTC-USD": "bitcoin",
    "ETH-USD": "ethereum",
    "USDT-USD": "tether",
    "BNB-USD": "binancecoin",
    "XRP-USD": "ripple",
    "SOL-USD": "solana",
    "USDC-USD": "usd-coin",
    "DOGE-USD": "dogecoin",
    "TRX-USD": "tron",
    "ADA-USD": "cardano",
    "WBTC-USD": "wrapped-bitcoin",
    "LINK-USD": "chainlink",
    "USDE-USD": "ethena-usde",
    "BCH-USD": "bitcoin-cash",
    "XLM-USD": "stellar",
    "SUI-USD": "sui",
    "AVAX-USD": "avalanche-2",
    "USDS-USD": "stably-usds",
    "HBAR-USD": "hedera-hashgraph",
    "LTC-USD": "litecoin",
}


def _safe_get(container: dict, *keys, default=None):
    """Retrieve a nested value from dicts safely."""
    current = container
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            return default
        current = current[key]
    return current


def _compute_fdv(total_supply: Optional[float], price: Optional[float], fdv_raw: Optional[float]) -> Optional[float]:
    if fdv_raw is not None:
        return fdv_raw
    if total_supply is not None and total_supply > 0 and price is not None:
        return total_supply * price
    return None


def fetch_fundamentals_for_symbol(symbol: str, timeout: int = 10) -> Dict[str, Optional[float]]:
    """
    Fetch and normalize fundamentals for a single symbol from CoinGecko.
    Returns a dict key -> value, or empty dict on failure.
    """
    coin_id = SYMBOL_TO_ID.get(symbol)
    if not coin_id:
        print(f"[fundamentals] WARNING: No CoinGecko ID for symbol {symbol}, skipping.")
        return {}

    url = f"https://api.coingecko.com/api/v3/coins/{coin_id}"
    params = {
        "localization": "false",
        "tickers": "false",
        "market_data": "true",
    }

    try:
        data, _meta = request_json_with_retries(
            url,
            params=params,
            timeout=float(timeout),
            max_attempts=4,
            retry_budget_seconds=18.0,
            backoff_base_seconds=0.5,
            backoff_cap_seconds=3.0,
            seed=f"coingecko:{coin_id}",
        )
    except Exception:
        print(f"[fundamentals] WARNING: CoinGecko request failed for {symbol} ({coin_id}).")
        return {}

    market_data = data.get("market_data", {}) or {}
    if not market_data:
        print(f"[fundamentals] WARNING: CoinGecko returned no market_data for {symbol} ({coin_id}).")
        return {}

    price = _safe_get(market_data, "current_price", "usd")
    mkt_cap = _safe_get(market_data, "market_cap", "usd")
    vol_24h = _safe_get(market_data, "total_volume", "usd")
    circ_supply = market_data.get("circulating_supply")
    total_supply = market_data.get("total_supply")
    fdv_raw = _safe_get(market_data, "fully_diluted_valuation", "usd")
    fdv = _compute_fdv(total_supply, price, fdv_raw)

    fundamentals: Dict[str, Optional[float]] = {
        "price": price,
        "mkt_cap": mkt_cap,
        "mkt_cap_rank": data.get("market_cap_rank"),
        "vol_24h": vol_24h,
        "circ_supply": circ_supply,
        "total_supply": total_supply,
        "fdv": fdv,
        "pct_change_1h": _safe_get(market_data, "price_change_percentage_1h_in_currency", "usd"),
        "pct_change_24h": _safe_get(market_data, "price_change_percentage_24h_in_currency", "usd"),
        "pct_change_7d": _safe_get(market_data, "price_change_percentage_7d_in_currency", "usd"),
        "pct_change_30d": _safe_get(market_data, "price_change_percentage_30d_in_currency", "usd"),
        "pct_change_1y": _safe_get(market_data, "price_change_percentage_1y_in_currency", "usd"),
        "mcap_change_1d": (
            _safe_get(market_data, "market_cap_change_percentage_24h_in_currency", "usd")
            if _safe_get(market_data, "market_cap_change_percentage_24h_in_currency", "usd") is not None
            else market_data.get("market_cap_change_percentage_24h")
        ),
        "ath_price": _safe_get(market_data, "ath", "usd"),
        "ath_change_pct": _safe_get(market_data, "ath_change_percentage", "usd"),
        "atl_price": _safe_get(market_data, "atl", "usd"),
        "atl_change_pct": _safe_get(market_data, "atl_change_percentage", "usd"),
        "dominance_pct": _safe_get(market_data, "market_cap_dominance"),
        "liquidity_score": data.get("liquidity_score"),
        "cg_score": data.get("coingecko_score"),
        "dev_score": data.get("developer_score"),
        "community_score": data.get("community_score"),
    }

    # Derived metrics
    if mkt_cap and mkt_cap > 0 and vol_24h is not None:
        fundamentals["vol_mcap_ratio"] = vol_24h / mkt_cap
    else:
        fundamentals["vol_mcap_ratio"] = None

    if circ_supply and circ_supply > 0 and total_supply and total_supply > 0:
        fundamentals["circ_ratio"] = circ_supply / total_supply
    else:
        fundamentals["circ_ratio"] = None

    if fdv and fdv > 0 and mkt_cap is not None:
        fundamentals["mktcap_fdv_ratio"] = mkt_cap / fdv
    else:
        fundamentals["mktcap_fdv_ratio"] = None

    return fundamentals

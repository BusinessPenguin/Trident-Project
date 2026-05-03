from typing import Dict


# Keyword lists for simple polarity scoring
POSITIVE_KEYWORDS = [
    "upgrade", "upgrades", "partnership", "partnerships", "integration", "integrations",
    "collaboration", "collaborations", "approval", "approved", "etf", "etfs",
    "launch", "launches", "growth", "investment", "investments", "surge", "surges",
    "bullish", "support", "supports", "mainnet", "adoption", "record high",
]

NEGATIVE_KEYWORDS = [
    "hack", "hacked", "exploit", "exploited", "breach", "breached", "stolen",
    "attack", "attacked", "malicious", "lawsuit", "lawsuits", "sues",
    "ban", "banned", "ban on", "delay", "delayed", "reversal", "bearish",
    "liquidation", "liquidations", "unregistered", "scam", "scams",
    "halt", "halts", "halted", "downgrade", "downgraded",
]


# Narrative clusters: topic categories for headlines
CLUSTERS: Dict[str, list] = {
    "regulation": [
        "sec", "lawsuit", "approval", "approved", "reject", "rejected", "filing",
        "rule", "rules", "etf", "commission", "fine", "fined", "regulator",
    ],
    "security": [
        "hack", "hacked", "exploit", "exploited", "breach", "breached",
        "stolen", "drain", "drained", "attack", "attacked", "malicious",
        "vulnerability", "bug", "bugs",
    ],
    "partnership": [
        "partnership", "partnerships", "integration", "integrations",
        "collaboration", "collaborations", "alliance", "alliances",
        "joins", "joined", "teams up",
    ],
    "upgrade": [
        "upgrade", "upgrades", "update", "updates", "roadmap", "mainnet",
        "testnet", "fork", "forks", "merge", "hard fork", "hardfork",
        "protocol upgrade",
    ],
    "market_events": [
        "liquidation", "liquidations", "whale", "whales", "accumulation",
        "selloff", "sell-off", "pump", "dump", "breakout", "spike",
        "crash", "rally",
    ],
    "institutional": [
        "institution", "institutions", "fidelity", "blackrock", "bank", "banks",
        "fund", "funds", "allocation", "allocations", "etf", "etfs",
    ],
    "exchange": [
        "exchange", "exchanges", "binance", "kraken", "coinbase",
        "withdrawal", "withdrawals", "freeze", "freezes", "suspends",
        "suspension", "halt", "halts", "delist", "delists", "delisted",
    ],
    "stablecoin": [
        "stablecoin", "stablecoins", "peg", "depeg", "de-pegged", "reserve",
        "reserves", "mint", "minted", "redeem", "redeems", "redemption",
        "supply",
    ],
    "defi": [
        "defi", "dex", "dexes", "yield", "liquidity pool", "pools",
        "protocol", "protocols", "amm", "staking", "borrow", "lending",
    ],
}


def compute_polarity(text: str) -> float | None:
    """
    Compute a simple polarity score in [-1, 1] from a news title.
    Uses keyword hits from POSITIVE_KEYWORDS and NEGATIVE_KEYWORDS.
    Returns 0.0 if no hits; None if text is empty.
    """
    if not text:
        return None

    t = text.lower()
    pos_hits = sum(1 for w in POSITIVE_KEYWORDS if w in t)
    neg_hits = sum(1 for w in NEGATIVE_KEYWORDS if w in t)
    total = pos_hits + neg_hits
    if total == 0:
        return 0.0
    return (pos_hits - neg_hits) / float(total)


def detect_clusters(text: str) -> Dict[str, int]:
    """
    Return a dict cluster_name -> hit count for a given text.
    """
    counts: Dict[str, int] = {name: 0 for name in CLUSTERS.keys()}
    if not text:
        return counts
    t = text.lower()
    for name, keywords in CLUSTERS.items():
        for kw in keywords:
            if kw in t:
                counts[name] += 1
    return counts


def primary_cluster(cluster_counts: Dict[str, int]) -> str:
    """
    Return the name of the cluster with the highest count, or "none".
    """
    if not cluster_counts:
        return "none"
    name, value = max(cluster_counts.items(), key=lambda kv: kv[1])
    return name if value > 0 else "none"

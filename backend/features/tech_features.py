from __future__ import annotations

from typing import Optional

from backend.features.technical_features import compute_technical_features


def compute_tech_features(symbol: str, con, interval: Optional[str] = None) -> dict:
    """Compatibility wrapper for technical feature computation."""
    return compute_technical_features(symbol, con, interval=interval)

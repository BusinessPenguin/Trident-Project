"""Compatibility shim for legacy fundamentals entrypoint."""

from __future__ import annotations

from typing import Optional, Dict, Any

from .fundamentals_features import get_fundamentals_features


def build_fundamentals(con, symbol: str, tech: Optional[Dict[str, Any]] = None) -> dict:
    _ = tech
    return get_fundamentals_features(symbol, con)


"""
Feature engineering package for Project Trident v2.0.0.

Exposes a single public entrypoint:

    from backend.features.features import build

which returns a combined feature dict for a symbol.
"""

from .features import build

__all__ = ["build"]

from __future__ import annotations

import math
from typing import Dict, List

from backend.features.tech_features import compute_tech_features
from backend.features.fundamentals_features import compute_fundamentals_features
from backend.features.news_features import compute_news_features, ACTIVE_UNIVERSE


def _bias_from_direction(direction: str) -> str:
    d = (direction or "").lower()
    if d == "bullish":
        return "bullish"
    if d == "bearish":
        return "bearish"
    return "neutral"


def _bias_to_sign(bias: str) -> int:
    if bias == "bullish":
        return 1
    if bias == "bearish":
        return -1
    return 0


def _clip_strength(val) -> float:
    try:
        return float(max(0.0, min(1.0, val)))
    except Exception:
        return 0.0


def _parse_pct(val) -> float | None:
    if val is None:
        return None
    try:
        s = str(val).replace("%", "").replace(",", "").strip()
        return float(s) / 100.0
    except Exception:
        return None


def _technical_state(tech: Dict) -> Dict[str, object]:
    ema_cross = tech.get("ema_cross")
    trend_strength = tech.get("trend_strength")
    bias = "neutral"
    strength = 0.0
    if ema_cross == "bullish":
        bias = "bullish"
    elif ema_cross == "bearish":
        bias = "bearish"
    strength = _clip_strength(abs(trend_strength)) if trend_strength is not None else strength
    return {"bias": bias, "strength": strength}


def _news_state(news: Dict) -> Dict[str, object]:
    bias = _bias_from_direction(news.get("direction"))
    strength = _clip_strength(news.get("intensity", 0.0))
    return {"bias": bias, "strength": strength}


def _fundamental_state(fund: Dict) -> Dict[str, object]:
    change = _parse_pct(fund.get("mcap_change_1d"))
    bias = "neutral"
    strength = 0.0
    if change is not None:
        if change > 0:
            bias = "bullish"
        elif change < 0:
            bias = "bearish"
        strength = _clip_strength(abs(change))
    return {"bias": bias, "strength": strength}


def _composite(tech_state: Dict, news_state: Dict, fund_state: Dict) -> Dict[str, object]:
    weights = {"tech": 0.4, "news": 0.3, "fund": 0.3}
    signals = {
        "tech": _bias_to_sign(tech_state["bias"]) * tech_state["strength"],
        "news": _bias_to_sign(news_state["bias"]) * news_state["strength"],
        "fund": _bias_to_sign(fund_state["bias"]) * fund_state["strength"],
    }
    weighted_sum = sum(signals[k] * weights[k] for k in weights)
    weight_total = sum(weights.values())
    composite_bias = "neutral"
    if weighted_sum > 0:
        composite_bias = "bullish"
    elif weighted_sum < 0:
        composite_bias = "bearish"
    composite_confidence = _clip_strength((abs(signals["tech"]) + abs(signals["news"]) + abs(signals["fund"])) / 3.0)
    superscore = _clip_strength(abs(weighted_sum) / weight_total)
    return {
        "composite_bias": composite_bias,
        "composite_confidence": composite_confidence,
        "superscore": superscore,
    }


def _agreement(tech_state: Dict, news_state: Dict, fund_state: Dict) -> Dict[str, object]:
    def pair_align(a: str, b: str) -> str:
        if a == b and a != "neutral":
            return "aligned"
        if a == "neutral" or b == "neutral":
            return "partial"
        return "divergent"

    pairs = [
        ("tech_vs_news", tech_state["bias"], news_state["bias"]),
        ("tech_vs_fundamentals", tech_state["bias"], fund_state["bias"]),
        ("news_vs_fundamentals", news_state["bias"], fund_state["bias"]),
    ]
    results = {}
    aligned = 0.0
    for name, a, b in pairs:
        status = pair_align(a, b)
        results[name] = status
        if status == "aligned":
            aligned += 1.0
        elif status == "partial":
            aligned += 0.5
    results["overall_alignment_score"] = round(aligned / len(pairs), 2)
    return results


def _signal_freshness() -> Dict[str, object]:
    return {
        "technical": {"half_life_hours": 12, "status": "current"},
        "news": {"half_life_hours": 36, "status": "recent"},
        "fundamentals": {"half_life_hours": 168, "status": "slow"},
    }


def _base_case_invalidation() -> List[str]:
    return [
        "Technical bias shifts away from current alignment",
        "News sentiment changes direction in consecutive windows",
        "Fundamental bias materially changes from current state",
        "Overall alignment score falls below neutral",
    ]


def _scenario_invalidation_best(tech_state: Dict, news_state: Dict, fund_state: Dict, agreement: Dict) -> List[str]:
    return [
        "Technical bias fails to improve from current state",
        "News sentiment deteriorates or turns negative",
        "Overall alignment weakens instead of strengthening",
    ]


def _scenario_invalidation_base(tech_state: Dict, news_state: Dict, fund_state: Dict, agreement: Dict) -> List[str]:
    return [
        "Technical bias flips direction from current state",
        "News sentiment changes direction across consecutive windows",
        "Composite agreement score materially declines",
    ]


def _scenario_invalidation_worst(tech_state: Dict, news_state: Dict, fund_state: Dict, agreement: Dict) -> List[str]:
    return [
        "Technicals stabilize or reverse positively",
        "News sentiment improves meaningfully",
        "Volatility contracts instead of expanding",
    ]


def _scenarios(symbol: str, tech_state: Dict, news_state: Dict, fund_state: Dict, composite: Dict, agreement: Dict) -> List[Dict]:
    # Static likelihoods summing to ~1
    base_like = 0.5
    best_like = 0.25
    worst_like = 0.25
    def drivers():
        return [
            f"technical: {tech_state['bias']} ({tech_state['strength']:.2f})",
            f"news: {news_state['bias']} ({news_state['strength']:.2f})",
            f"fundamental: {fund_state['bias']} ({fund_state['strength']:.2f})",
        ]
    return [
        {
            "name": "best_case",
            "description": f"{symbol} improves if technicals and news strengthen together.",
            "likelihood": best_like,
            "intensity": 0.7,
            "drivers": drivers(),
            "invalidation_conditions": _scenario_invalidation_best(tech_state, news_state, fund_state, agreement),
        },
        {
            "name": "base_case",
            "description": f"{symbol} stays aligned with current composite bias ({composite['composite_bias']}).",
            "likelihood": base_like,
            "intensity": 0.5,
            "drivers": drivers(),
            "invalidation_conditions": _scenario_invalidation_base(tech_state, news_state, fund_state, agreement),
        },
        {
            "name": "worst_case",
            "description": f"{symbol} weakens if opposing signals grow or volatility rises.",
            "likelihood": worst_like,
            "intensity": 0.8,
            "drivers": drivers(),
            "invalidation_conditions": _scenario_invalidation_worst(tech_state, news_state, fund_state, agreement),
        },
    ]


def _state_summary(symbol: str, tech_state: Dict, news_state: Dict, fund_state: Dict, composite: Dict) -> str:
    return (
        f"{symbol} base case is {composite['composite_bias']} with confidence {composite['composite_confidence']:.2f}; "
        f"tech {tech_state['bias']} ({tech_state['strength']:.2f}), "
        f"news {news_state['bias']} ({news_state['strength']:.2f}), "
        f"fundamentals {fund_state['bias']} ({fund_state['strength']:.2f})."
    )


def _scenario_interpretation(
    scenarios: List[Dict],
    tech_state: Dict,
    news_state: Dict,
    fund_state: Dict,
    agreement: Dict,
    freshness: Dict,
) -> Dict[str, Dict]:
    def narrative(s):
        return (
            f"This scenario could play out if the current mix of technical ({tech_state['bias']}) "
            f"and news ({news_state['bias']}) tone persists while fundamentals stay {fund_state['bias']}. "
            f"Agreement across modalities is {agreement.get('overall_alignment_score', 0):.2f}, "
            f"so the outlook remains conditional and may shift with new information."
        )

    def why_exists(s):
        return (
            f"It exists because technicals are {tech_state['bias']} with strength {tech_state['strength']:.2f}, "
            f"news is {news_state['bias']} with strength {news_state['strength']:.2f}, "
            f"and fundamentals are {fund_state['bias']} with strength {fund_state['strength']:.2f}."
        )

    def supports():
        return [
            f"Technical posture: {tech_state['bias']} ({tech_state['strength']:.2f})",
            f"News tone: {news_state['bias']} ({news_state['strength']:.2f})",
            f"Fundamentals: {fund_state['bias']} ({fund_state['strength']:.2f})",
            f"Alignment: {agreement.get('overall_alignment_score', 0):.2f}",
            f"Freshness: tech {freshness['technical']['status']}, news {freshness['news']['status']}, fund {freshness['fundamentals']['status']}",
        ]

    def fragilities(inv_conditions: List[str]):
        frags = list(inv_conditions)
        frags.append("Signals could diverge if alignment weakens further")
        return frags

    out = {}
    for s in scenarios:
        name = s.get("name")
        inv = s.get("invalidation_conditions") or []
        out[name] = {
            "narrative_explanation": narrative(s),
            "why_this_scenario_exists": why_exists(s),
            "key_supporting_factors": supports(),
            "key_fragilities": fragilities(inv),
        }
    return out


def compute_signal_fusion(symbol: str, con) -> Dict[str, object]:
    """
    Phase 3A fusion: static, deterministic aggregation of tech/news/fund signals.
    """
    if symbol not in ACTIVE_UNIVERSE:
        return {"error": "symbol out of active universe"}

    tech = compute_tech_features(symbol, con)
    fund = compute_fundamentals_features(symbol, con)
    news = compute_news_features(symbol, con)

    tech_state = _technical_state(tech)
    news_state = _news_state(news)
    fund_state = _fundamental_state(fund)

    agreement = _agreement(tech_state, news_state, fund_state)
    freshness = _signal_freshness()
    composite = _composite(tech_state, news_state, fund_state)
    scenarios = _scenarios(symbol, tech_state, news_state, fund_state, composite, agreement)
    scenario_interpretation = _scenario_interpretation(scenarios, tech_state, news_state, fund_state, agreement, freshness)

    market_state = {
        "technical_bias": tech_state["bias"],
        "technical_strength": tech_state["strength"],
        "news_bias": news_state["bias"],
        "news_strength": news_state["strength"],
        "fundamental_bias": fund_state["bias"],
        "fundamental_strength": fund_state["strength"],
        "composite_bias": composite["composite_bias"],
        "composite_confidence": composite["composite_confidence"],
        "superscore": composite["superscore"],
    }

    return {
        "symbol": symbol,
        "market_state": market_state,
        "agreement": agreement,
        "signal_freshness": freshness,
        "scenarios": scenarios,
        "scenario_interpretation": scenario_interpretation,
        "state_summary": _state_summary(symbol, tech_state, news_state, fund_state, composite),
    }

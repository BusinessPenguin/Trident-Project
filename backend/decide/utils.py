from __future__ import annotations

from typing import Any, Dict, Optional


AGGRESSION_TIER_KNOBS: Dict[str, Dict[str, float]] = {
    "very_defensive": {
        "risk_mult": 0.70,
        "stop_mult": 1.30,
        "hold_mult": 1.20,
        "notional_cap": 0.10,
    },
    "defensive": {
        "risk_mult": 0.85,
        "stop_mult": 1.15,
        "hold_mult": 1.10,
        "notional_cap": 0.11,
    },
    "balanced": {
        "risk_mult": 1.00,
        "stop_mult": 1.00,
        "hold_mult": 1.00,
        "notional_cap": 0.12,
    },
    "assertive": {
        "risk_mult": 1.15,
        "stop_mult": 0.90,
        "hold_mult": 0.85,
        "notional_cap": 0.135,
    },
    "very_aggressive": {
        "risk_mult": 1.30,
        "stop_mult": 0.80,
        "hold_mult": 0.75,
        "notional_cap": 0.15,
    },
}


def _clamp(value: float, low: float, high: float) -> float:
    if value < low:
        return float(low)
    if value > high:
        return float(high)
    return float(value)


def normalize_aggression_tier(tier: Any) -> str:
    t = str(tier or "").strip().lower()
    return t if t in AGGRESSION_TIER_KNOBS else "balanced"


def aggression_knobs_for_tier(tier: Any, baseline: Optional[Dict[str, Any]] = None) -> Dict[str, float]:
    label = normalize_aggression_tier(tier)
    base = dict(AGGRESSION_TIER_KNOBS[label])
    baseline = baseline or {}

    risk_scale = _clamp(float(baseline.get("risk_mult", 1.0)), 0.80, 1.20)
    stop_scale = _clamp(float(baseline.get("stop_mult", 1.0)), 0.80, 1.20)
    hold_scale = _clamp(float(baseline.get("hold_mult", 1.0)), 0.80, 1.20)
    notional_bias = float(baseline.get("exposure_cap", base["notional_cap"])) - AGGRESSION_TIER_KNOBS["balanced"][
        "notional_cap"
    ]
    notional_cap = _clamp(base["notional_cap"] + notional_bias, 0.10, 0.15)

    return {
        "risk_mult": round(_clamp(base["risk_mult"] * risk_scale, 0.50, 1.60), 6),
        "stop_mult": round(_clamp(base["stop_mult"] * stop_scale, 0.70, 1.60), 6),
        "hold_mult": round(_clamp(base["hold_mult"] * hold_scale, 0.60, 1.50), 6),
        "notional_cap": round(notional_cap, 6),
    }


def compute_hold_window_hours(
    conviction_label: str,
    regime_label: str,
    event_risk: str,
    hold_multiplier: float = 1.0,
    min_hours: int = 24,
    max_hours: int = 96,
    hard_cap_hours: int = 168,
) -> int:
    conv = str(conviction_label or "").lower()
    regime = str(regime_label or "").lower()
    risk = str(event_risk or "").lower()

    if conv == "high":
        base = 24.0
    elif conv == "medium":
        base = 48.0
    else:
        base = 96.0

    regime_factor = 1.0
    if regime in {"bull_trend", "bear_trend", "trend"}:
        regime_factor = 0.85
    elif regime == "high_vol_chop":
        regime_factor = 1.15
    elif regime == "range":
        regime_factor = 1.0

    risk_factor = 0.90 if risk == "elevated" else 1.0
    hold_mult = max(0.10, float(hold_multiplier or 1.0))

    hours = base * regime_factor * risk_factor * hold_mult
    hours = _clamp(hours, float(min_hours), float(max_hours))
    hours = _clamp(hours, float(min_hours), float(hard_cap_hours))
    return int(round(hours))


def compute_position_size(
    effective_confidence: float,
    raw_confidence: float,
    agreement_score: float,
    margin_vs_second: float,
    regime: str,
    volatility_regime: str,
    event_risk: str,
    fear_greed_value: int,
    account_equity: float = 100000,
    base_risk_pct: float = 0.0125,
) -> Dict[str, float]:
    _ = volatility_regime
    risk_pct = float(base_risk_pct) * (0.60 + 1.60 * float(effective_confidence))
    risk_pct *= (0.80 + 0.40 * float(raw_confidence))

    edge_factor = max(0.60, min(1.40, float(margin_vs_second) / 0.20))
    risk_pct *= edge_factor

    if agreement_score > 0.80:
        risk_pct *= 1.30
    elif agreement_score >= 0.65:
        risk_pct *= 1.15
    else:
        risk_pct *= 0.90

    if regime == "trend":
        risk_pct *= 1.35
    elif regime == "high_vol_chop":
        risk_pct *= 0.90
    elif regime == "squeeze":
        risk_pct *= 0.75

    if int(fear_greed_value) < 15:
        risk_pct *= 0.95

    cap = 0.03
    if regime == "high_vol_chop" and str(event_risk).lower() == "elevated":
        cap = 0.015

    risk_pct = min(max(risk_pct, 0.0), cap)
    return {
        "risk_pct_of_equity": round(risk_pct, 6),
        "risk_dollars": round(float(account_equity) * risk_pct, 2),
    }


def compute_entry_plan(confidence: float, regime: str, atr_value: float) -> Dict[str, Optional[float]]:
    _ = atr_value
    if confidence >= 0.55:
        entry_type = "market"
        offset = None
    elif confidence >= 0.45:
        entry_type = "limit"
        offset = 0.25
    else:
        entry_type = "limit"
        offset = 0.50

    if regime == "high_vol_chop":
        validity_hours = 4
    elif regime == "trend":
        validity_hours = 12
    else:
        validity_hours = 8

    return {
        "type": entry_type,
        "limit_offset_atr": offset,
        "validity_hours": int(validity_hours),
    }


def compute_stop(action: str, entry_price: float, atr_value: float, regime: str) -> Dict[str, float]:
    multiple = 1.5
    if regime == "high_vol_chop":
        multiple = 1.8
    elif regime == "trend":
        multiple = 1.2
    elif regime == "squeeze":
        multiple = 2.0

    if action == "SHORT":
        stop = float(entry_price) + (float(atr_value) * multiple)
    else:
        stop = float(entry_price) - (float(atr_value) * multiple)

    return {
        "stop_multiple_atr": float(multiple),
        "stop_price_estimate": round(float(stop), 2),
        "atr_used": float(atr_value),
    }


def compute_target(action: str, entry_price: float, stop_price: float, confidence: float) -> Dict[str, float]:
    if confidence < 0.5:
        r_multiple = 2.0
    elif confidence < 0.65:
        r_multiple = 2.5
    else:
        r_multiple = 3.0

    risk_distance = abs(float(entry_price) - float(stop_price))
    if action == "SHORT":
        target = float(entry_price) - risk_distance * r_multiple
    else:
        target = float(entry_price) + risk_distance * r_multiple

    return {
        "risk_reward_multiple": float(r_multiple),
        "target_price_estimate": round(float(target), 2),
    }


def compute_exit_conditions(regime: str) -> Dict[str, object]:
    if regime == "high_vol_chop":
        time_stop = 6
        breakeven_trigger_r = 0.6
        trailing_stop_atr = 1.4
    elif regime == "trend":
        time_stop = 20
        breakeven_trigger_r = 0.9
        trailing_stop_atr = 0.9
    elif regime == "squeeze":
        time_stop = 4
        breakeven_trigger_r = 0.5
        trailing_stop_atr = 1.6
    else:
        time_stop = 10
        breakeven_trigger_r = 0.75
        trailing_stop_atr = 1.2

    return {
        "time_stop_hours": int(time_stop),
        "breakeven_trigger_r": float(breakeven_trigger_r),
        "partial_take_profit": [
            {"at_r": 1.0, "close_pct": 0.40},
            {"at_r": 2.0, "close_pct": 0.30},
        ],
        "trailing_stop_atr": float(trailing_stop_atr),
        "invalidation_rules": [
            "agreement_score < 0.50",
            "regime shift detected",
            "event_risk escalates",
        ],
    }

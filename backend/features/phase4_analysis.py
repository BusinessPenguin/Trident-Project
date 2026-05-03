from __future__ import annotations

import os
import sys
import json
import math
import re
import uuid
import hashlib
from datetime import datetime, timezone, timedelta
from typing import Dict, Any, Tuple, List, Optional

from openai import OpenAI

from backend.features.tech_features import compute_tech_features
from backend.features.news_features import compute_news_features
from backend.features.fundamentals_features import compute_fundamentals_features
from backend.features.news_source_quality import get_source_quality
from backend.features.calendar_features import compute_calendar_features
from backend.features.fed_liquidity import compute_fed_liquidity_features_v2


def clamp01(x: float) -> float:
    """Clamp a numeric value into [0, 1]."""
    try:
        return float(max(0.0, min(1.0, x)))
    except Exception:
        return 0.0


def safe_z(x: float, lo: float, hi: float) -> float:
    """Normalize x into [0,1] given a (lo, hi) range, with clipping."""
    try:
        if hi <= lo:
            return 0.0
        return clamp01((x - lo) / (hi - lo))
    except Exception:
        return 0.0


def _clip(v: float) -> float:
    return clamp01(v)


def _bias_from_direction(direction: str) -> str:
    d = (direction or "").lower()
    if d == "bullish":
        return "bullish"
    if d == "bearish":
        return "bearish"
    return "neutral"


def _parse_float(val) -> float | None:
    if val is None:
        return None
    try:
        s = str(val).replace(",", "").strip()
        if s.lower() == "n/a":
            return None
        return float(s)
    except Exception:
        return None


def _parse_pct(val) -> float | None:
    if val is None:
        return None
    try:
        s = str(val).replace("%", "").replace(",", "").strip()
        return float(s) / 100.0
    except Exception:
        return None


REQUIRED_MODALITY_MAX_AGE_HOURS = 72.0


def _stable_json_hash(payload: Dict[str, Any]) -> str:
    try:
        raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    except Exception:
        raw = str(payload)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _new_run_id(prefix: str) -> str:
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    return f"{prefix}_{ts}_{uuid.uuid4().hex[:8]}"


def _resolve_scenario_use_gpt(use_gpt: Optional[bool] = None) -> bool:
    if use_gpt is not None:
        return bool(use_gpt)
    explicit = os.getenv("TRIDENT_SCENARIO_USE_GPT")
    if explicit is not None:
        return str(explicit).strip().lower() in {"1", "true", "yes", "y", "on"}
    deterministic = os.getenv("TRIDENT_SCENARIO_DETERMINISTIC")
    if deterministic is not None and str(deterministic).strip().lower() in {"1", "true", "yes", "y", "on"}:
        return False
    return bool(os.getenv("OPENAI_API_KEY"))


def _sanitize_untrusted_text(text: Any, max_chars: int = 800) -> tuple[str, bool]:
    raw = str(text or "")
    suspicious_patterns = [
        r"(?i)\bignore\b.*\binstruction",
        r"(?i)\bsystem prompt\b",
        r"(?i)\bdeveloper message\b",
        r"(?i)\bact as\b",
        r"(?i)\byou are chatgpt\b",
        r"(?i)<\s*script",
        r"(?i)\btool call\b",
    ]
    flagged = any(re.search(pat, raw) is not None for pat in suspicious_patterns)
    cleaned = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", " ", raw)
    cleaned = " ".join(cleaned.split())
    if len(cleaned) > max_chars:
        cleaned = cleaned[:max_chars].rstrip()
        flagged = True
    return cleaned, flagged


def evaluate_required_modality_status(
    freshness: Dict[str, Any],
    stale_hours: float = REQUIRED_MODALITY_MAX_AGE_HOURS,
) -> Dict[str, Any]:
    checks = [
        ("technicals", "tech_avg_age_hours"),
        ("fundamentals", "fundamentals_avg_age_hours"),
    ]
    failures: List[Dict[str, Any]] = []
    for label, key in checks:
        age = freshness.get(key)
        if age is None:
            failures.append(
                {
                    "modality": label,
                    "code": "MISSING",
                    "detail": f"{key}=None",
                }
            )
            continue
        try:
            age_h = float(age)
        except Exception:
            failures.append(
                {
                    "modality": label,
                    "code": "INVALID",
                    "detail": f"{key}={age!r}",
                }
            )
            continue
        if age_h >= float(stale_hours):
            failures.append(
                {
                    "modality": label,
                    "code": "STALE",
                    "detail": f"{key}={age_h:.2f}h >= {float(stale_hours):.2f}h",
                }
            )
    return {
        "ok": len(failures) == 0,
        "stale_hours_threshold": float(stale_hours),
        "failures": failures,
    }


def _alignment_score(tb: str, nb: str, fb: str) -> float:
    def align(a, b):
        if a == b and a != "neutral":
            return 1.0
        if a == "neutral" or b == "neutral":
            return 0.5
        return 0.0

    return round((align(tb, nb) + align(tb, fb) + align(nb, fb)) / 3.0, 2)


def _bias_from_tech(tech: Dict[str, Any]) -> str:
    ema20_above_50 = _get_tech_value(tech, "ema20_above_50")
    price_above_20sma = _get_tech_value(tech, "price_above_20sma")
    if ema20_above_50 is True or price_above_20sma is True:
        return "bullish"
    if ema20_above_50 is False and price_above_20sma is False:
        return "bearish"
    return "neutral"


def _bias_from_fundamentals(fund: Dict[str, Any]) -> str:
    pct_24h = _parse_float(fund.get("pct_change_24h"))
    if pct_24h is None:
        return "neutral"
    if pct_24h > 0.5:
        return "bullish"
    if pct_24h < -0.5:
        return "bearish"
    return "neutral"


def _agreement_fallback(tech: Dict[str, Any], news: Dict[str, Any], fund: Dict[str, Any]) -> float:
    return _continuous_agreement_score(tech, news, fund)


def _continuous_agreement_score(
    tech: Dict[str, Any], news: Dict[str, Any], fund: Dict[str, Any]
) -> float:
    trend_strength = float(_get_tech_value(tech, "trend_strength") or 0.0)
    ema_cross = (tech.get("ema_cross") or "").lower()
    if ema_cross not in ("bullish", "bearish", "neutral"):
        ema_cross = _bias_from_tech(tech)
    tech_sign = 1.0 if ema_cross == "bullish" else -1.0 if ema_cross == "bearish" else 0.0
    tech_strength = clamp01(abs(trend_strength) / 0.02)
    tech_score = tech_sign * tech_strength

    news_dir = _bias_from_direction(news.get("direction"))
    weighted_bullish = _parse_float(news.get("weighted_bullish"))
    weighted_bearish = _parse_float(news.get("weighted_bearish"))
    if news_dir == "neutral" and weighted_bullish is not None and weighted_bearish is not None:
        skew = weighted_bullish - weighted_bearish
        if abs(skew) >= 1.0:
            news_dir = "bullish" if skew > 0 else "bearish"
    news_sign = 1.0 if news_dir == "bullish" else -1.0 if news_dir == "bearish" else 0.0
    news_intensity = float(news.get("intensity") or 0.0)
    news_strength = clamp01(news_intensity / 0.5)
    news_score = news_sign * news_strength

    fund_change = _parse_float(fund.get("pct_change_24h"))
    if fund_change is None:
        fund_change = _parse_float(fund.get("mcap_change_1d"))
    fund_sign = 1.0 if (fund_change or 0.0) > 0 else -1.0 if (fund_change or 0.0) < 0 else 0.0
    fund_strength = clamp01(abs(fund_change or 0.0) / 2.0)
    fund_score = fund_sign * fund_strength

    if abs(tech_score) < 1e-6 and abs(news_score) < 1e-6 and abs(fund_score) < 1e-6:
        return 0.5

    pair_sim = (tech_score * news_score + tech_score * fund_score + news_score * fund_score) / 3.0
    return clamp01(0.5 + 0.5 * pair_sim)


def _get_tech_value(tech: Dict[str, Any], key: str):
    if key in tech:
        return tech.get(key)
    for block in ("trend", "momentum", "volatility", "ema", "volume", "relative_strength", "breadth"):
        b = tech.get(block)
        if isinstance(b, dict) and key in b:
            return b.get(key)
    return None


def _horizon_alignment_from_multi_horizon(tech: Dict[str, Any]) -> Dict[str, Any] | None:
    multi_horizon = tech.get("multi_horizon")
    if not isinstance(multi_horizon, dict):
        return None
    horizon_signals = multi_horizon.get("horizon_signals")
    if not isinstance(horizon_signals, dict) or not horizon_signals:
        return None

    windows_raw = multi_horizon.get("windows")
    windows: List[str] = []
    if isinstance(windows_raw, list):
        for w in windows_raw:
            ws = str(w)
            if ws in horizon_signals:
                windows.append(ws)
    if not windows:
        windows = [w for w in ("6h", "2d", "14d") if w in horizon_signals]
    if not windows:
        windows = [str(k) for k in horizon_signals.keys()]
    if not windows:
        return None

    dir_map = {"bullish": 1.0, "bearish": -1.0, "neutral": 0.0}
    s_val = 0.0
    w_sum = 0.0
    signals_used: Dict[str, Dict[str, Any]] = {}
    for w in windows:
        signal = horizon_signals.get(w) or {}
        if not isinstance(signal, dict):
            signal = {}
        trend_bias = (signal.get("trend_bias") or "").lower()
        direction = dir_map.get(trend_bias, 0.0)
        comp_raw = signal.get("composite_strength")
        try:
            weight = clamp01(float(comp_raw))
        except Exception:
            weight = 0.0
            comp_raw = None
        s_val += weight * direction
        w_sum += weight
        # Keep full signal payload so output can mirror tech:features exactly.
        signal_copy = dict(signal)
        if isinstance(comp_raw, (int, float)):
            signal_copy["composite_strength"] = round(float(comp_raw), 3)
        signals_used[w] = signal_copy

    if w_sum <= 1e-12:
        return None

    alignment_raw = clamp01(abs(s_val) / max(w_sum, 1e-12))
    if alignment_raw >= 0.67:
        label = "aligned"
    elif alignment_raw >= 0.34:
        label = "mixed"
    else:
        label = "conflict"

    if abs(s_val) < 0.05 * w_sum:
        dominant = "neutral"
    elif s_val > 0:
        dominant = "bullish"
    elif s_val < 0:
        dominant = "bearish"
    else:
        dominant = "neutral"

    return {
        "horizon_alignment_score": float(alignment_raw),
        "horizon_alignment_label": label,
        "horizon_dominant_bias": dominant,
        "horizon_windows": windows,
        "horizon_signals_used": signals_used,
    }


def _horizon_alignment_detail(info: Dict[str, Any]) -> str:
    windows = info.get("horizon_windows") or []
    signals = info.get("horizon_signals_used") or {}
    parts: List[str] = []
    for w in windows:
        sig = signals.get(w) or {}
        trend = sig.get("trend_bias")
        comp = sig.get("composite_strength")
        trend_txt = trend if trend is not None else "null"
        comp_txt = f"{float(comp):.3f}" if isinstance(comp, (int, float)) else "n/a"
        parts.append(f"{w} {trend_txt} ({comp_txt})")
    align = info.get("horizon_alignment_score")
    label = info.get("horizon_alignment_label")
    align_txt = f"{float(align):.2f}" if isinstance(align, (int, float)) else "n/a"
    return f"{', '.join(parts)} => alignment={align_txt} label={label}"


def compute_likelihoods(snapshot: Dict[str, Any]) -> Dict[str, float]:
    """
    Deterministic likelihood calibration using only existing features.
    """
    tech = snapshot.get("technical_features", {}) or {}
    news = snapshot.get("news_features", {}) or {}
    fund = snapshot.get("fundamental_features", {}) or {}
    meta = snapshot.get("meta_features", {}) or {}

    best = 0.25
    base = 0.50
    worst = 0.25

    agreement_score = meta.get("agreement_score")
    if agreement_score is None:
        agreement_score = _agreement_fallback(tech, news, fund)
    try:
        agreement_score = float(agreement_score)
    except Exception:
        agreement_score = 0.5
    agreement_score = clamp01(agreement_score)
    base += 0.20 * agreement_score
    best += 0.10 * agreement_score
    worst -= 0.10 * agreement_score

    vol_regime = (_get_tech_value(tech, "vol_regime") or "").lower()
    if vol_regime == "high":
        worst += 0.15
        base -= 0.10
    elif vol_regime == "low":
        base += 0.10

    ema_cross = (_get_tech_value(tech, "ema_cross") or "").lower()
    trend_strength = float(_get_tech_value(tech, "trend_strength") or 0.0)
    TREND_STRENGTH_THRESHOLD = 0.02
    if ema_cross == "bullish" and trend_strength > TREND_STRENGTH_THRESHOLD:
        best += 0.15
    elif ema_cross == "bearish" and trend_strength > TREND_STRENGTH_THRESHOLD:
        worst += 0.15

    weighted_bullish = float(news.get("weighted_bullish") or 0.0)
    weighted_bearish = float(news.get("weighted_bearish") or 0.0)
    news_skew = weighted_bullish - weighted_bearish
    NEWS_SKEW_K = 0.01
    if news_skew > 0:
        best += min(0.10, NEWS_SKEW_K * abs(news_skew))
    elif news_skew < 0:
        worst += min(0.10, NEWS_SKEW_K * abs(news_skew))

    pct_vals = [
        _parse_float(fund.get("pct_change_24h")),
        _parse_float(fund.get("pct_change_7d")),
        _parse_float(fund.get("pct_change_30d")),
    ]
    pct_vals = [v for v in pct_vals if v is not None]
    if pct_vals:
        avg = sum(pct_vals) / len(pct_vals)
        if avg > 0:
            best += 0.05
        elif avg < 0:
            worst += 0.05

    # Additive technical context: VWAP positioning, cross-asset relative strength,
    # market breadth, and squeeze state.
    price_above_vwap = _get_tech_value(tech, "price_above_vwap")
    vwap_distance_pct = _parse_float(_get_tech_value(tech, "vwap_distance_pct"))
    rs_vs_btc_7d = _parse_float(_get_tech_value(tech, "rs_vs_btc_7d"))
    breadth_score = _parse_float(_get_tech_value(tech, "breadth_score"))
    squeeze_on = bool(_get_tech_value(tech, "squeeze_on") is True)
    squeeze_fired = bool(_get_tech_value(tech, "squeeze_fired") is True)

    if price_above_vwap is True and vwap_distance_pct is not None and vwap_distance_pct > 0:
        best += min(0.05, abs(vwap_distance_pct) * 2.0)
    elif price_above_vwap is False and vwap_distance_pct is not None and vwap_distance_pct < 0:
        worst += min(0.05, abs(vwap_distance_pct) * 2.0)

    if rs_vs_btc_7d is not None:
        if rs_vs_btc_7d > 0:
            best += min(0.06, abs(rs_vs_btc_7d) * 1.5)
        elif rs_vs_btc_7d < 0:
            worst += min(0.06, abs(rs_vs_btc_7d) * 1.5)

    if breadth_score is not None:
        if breadth_score >= 0.60:
            best += 0.04
        elif breadth_score <= 0.40:
            worst += 0.04

    if squeeze_on:
        base += 0.04
        best -= 0.02
        worst -= 0.02
    if squeeze_fired:
        if ema_cross == "bullish":
            best += 0.05
        elif ema_cross == "bearish":
            worst += 0.05

    conflict_score = 0.0
    macd_hist = _get_tech_value(tech, "macd_hist")
    if macd_hist is not None and ema_cross in ("bullish", "bearish"):
        macd_sign = 1 if macd_hist > 0 else -1 if macd_hist < 0 else 0
        ema_sign = 1 if ema_cross == "bullish" else -1
        if macd_sign != 0 and macd_sign != ema_sign:
            conflict_score += 0.34

    ret_7d = _get_tech_value(tech, "ret_7d")
    ret_1d = _get_tech_value(tech, "ret_1d")
    if ret_1d is not None and ret_7d is not None:
        if (ret_1d > 0) != (ret_7d > 0):
            conflict_score += 0.33

    if ema_cross in ("bullish", "bearish") and news_skew != 0:
        news_sign = 1 if news_skew > 0 else -1
        ema_sign = 1 if ema_cross == "bullish" else -1
        if news_sign != ema_sign:
            conflict_score += 0.33
    elif ret_1d is not None and news_skew != 0:
        news_sign = 1 if news_skew > 0 else -1
        ret_sign = 1 if ret_1d > 0 else -1 if ret_1d < 0 else 0
        if ret_sign != 0 and news_sign != ret_sign:
            conflict_score += 0.33

    conflict_score = _clip(conflict_score)
    base += 0.10 * conflict_score
    best -= 0.05 * conflict_score
    worst -= 0.05 * conflict_score

    # Likelihood guardrails based on agreement strength
    tech_bias = _bias_from_tech(tech)
    news_bias = _bias_from_direction(news.get("direction"))
    fund_bias = _bias_from_fundamentals(fund)
    strong_alignment = (tech_bias == news_bias == fund_bias) and tech_bias != "neutral"

    if agreement_score < 0.55:
        base = min(base, 0.60)
    if agreement_score < 0.45:
        base = min(base, 0.55)
        best = max(best, 0.20)
        worst = max(worst, 0.20)
    if agreement_score > 0.75 and strong_alignment:
        base = min(base, 0.70)

    best = max(best, 0.01)
    base = max(base, 0.01)
    worst = max(worst, 0.01)

    total = best + base + worst
    best /= total
    base /= total
    worst /= total

    best = round(best, 3)
    base = round(base, 3)
    worst = round(worst, 3)
    diff = round(1.0 - (best + base + worst), 3)
    base = round(clamp01(base + diff), 3)

    return {
        "best_case": best,
        "base_case": base,
        "worst_case": worst,
    }


def compute_scenario_alignment(snapshot: Dict[str, Any]) -> Dict[str, float]:
    tech = snapshot.get("technical_features", {}) or {}
    news = snapshot.get("news_features", {}) or {}
    meta = snapshot.get("meta_features", {}) or {}
    fng_value = (
        snapshot.get("macro_sentiment", {}).get("fear_greed", {}).get("value")
    )

    ema_cross = (_get_tech_value(tech, "ema_cross") or "").lower()
    trend_strength = float(_get_tech_value(tech, "trend_strength") or 0.0)
    macd_hist = _get_tech_value(tech, "macd_hist")
    rsi14 = _get_tech_value(tech, "rsi14")
    ret_1d = _get_tech_value(tech, "ret_1d")
    ret_3d = _get_tech_value(tech, "ret_3d")
    ret_7d = _get_tech_value(tech, "ret_7d")
    vol_regime = (_get_tech_value(tech, "vol_regime") or "").lower()
    news_dir = (news.get("direction") or "").lower()
    agreement_score = float(meta.get("agreement_score") or 0.0)

    best_raw = 0.0
    if ema_cross == "bullish":
        best_raw += 0.35
    if trend_strength >= 0.010:
        best_raw += 0.15
    if rsi14 is not None and rsi14 >= 50:
        best_raw += 0.10
    if macd_hist is not None and macd_hist < 0:
        best_raw -= 0.10
    if news_dir == "bullish":
        best_raw += 0.20
    elif news_dir == "neutral":
        best_raw += 0.05
    elif news_dir == "bearish":
        best_raw -= 0.20
    if fng_value is not None and (fng_value <= 20 or fng_value >= 80):
        best_raw -= 0.05

    worst_raw = 0.0
    if ret_7d is not None and ret_7d < 0:
        worst_raw += 0.20
    if ret_3d is not None and ret_3d < 0:
        worst_raw += 0.10
    if ret_1d is not None and ret_1d < 0:
        worst_raw += 0.10
    if macd_hist is not None and macd_hist < 0:
        worst_raw += 0.20
    if ema_cross != "bullish":
        worst_raw += 0.15
    if news_dir == "bearish":
        worst_raw += 0.20
    elif news_dir == "neutral":
        worst_raw += 0.05
    elif news_dir == "bullish":
        worst_raw -= 0.10
    if fng_value is not None and (fng_value <= 20 or fng_value >= 80):
        worst_raw -= 0.05

    base_raw = 0.0
    if ema_cross == "neutral":
        base_raw += 0.25
    if trend_strength <= 0.003:
        base_raw += 0.25
    if vol_regime in {"low", "normal"}:
        base_raw += 0.15
    if rsi14 is not None and abs(rsi14 - 50) <= 7:
        base_raw += 0.15
    if agreement_score <= 0.60:
        base_raw += 0.20

    def _align(raw: float, max_raw: float) -> float:
        if max_raw <= 0:
            return 0.0
        norm = clamp01(raw / max_raw)
        return max(-1.0, min(1.0, (2.0 * norm) - 1.0))

    return {
        "best_case": _align(best_raw, 0.80),
        "base_case": _align(base_raw, 1.00),
        "worst_case": _align(worst_raw, 0.95),
    }


def _scenario_penalty_sums(penalties: Dict[str, List[Dict[str, Any]]]) -> Dict[str, float]:
    totals = {"best_case": 0.0, "base_case": 0.0, "worst_case": 0.0}
    tail_adjustments = {"best_case": 0.0, "base_case": 0.0, "worst_case": 0.0}
    tail_codes = {"FEAR_RISK", "GREED_RISK", "EXTREME_FEAR_RISK", "EXTREME_GREED_RISK"}
    for group in penalties.values():
        for p in group:
            impact = float(p.get("impact") or 0.0)
            if impact >= 0:
                continue
            magnitude = -impact
            code = p.get("code") or ""
            base_mult = 0.7 if code in {"LOW_AGREEMENT", "LOW_LIQUIDITY"} else 1.0
            totals["best_case"] += magnitude
            totals["base_case"] += magnitude * base_mult
            totals["worst_case"] += magnitude
            if code in tail_codes:
                totals["best_case"] += 0.02
                tail_adjustments["best_case"] += 0.02
                totals["worst_case"] += 0.02
                tail_adjustments["worst_case"] += 0.02
    return {"totals": totals, "tail_adjustments": tail_adjustments}


def _scenario_confidence(
    overall: float,
    scenario_likelihood: float,
    scenario_intensity: float,
    edge_score: float,
    agreement_score: float,
    horizon_alignment_score: float | None = None,
) -> float:
    """
    Deterministic per-scenario confidence.
    Uses final overall confidence (already post-penalty) to avoid double-penalty subtraction.
    """
    o_val = clamp01(float(overall))
    p_i = clamp01(float(scenario_likelihood))
    i_i = clamp01(float(scenario_intensity))
    e_val = clamp01(float(edge_score))
    a_val = clamp01(float(agreement_score))

    # Base mapping from edge + intensity + alignment quality.
    intensity_factor = 0.60 + 0.40 * (1.0 - i_i)
    alignment_factor = 0.80 + 0.20 * a_val
    if isinstance(horizon_alignment_score, (int, float)):
        h_val = clamp01(float(horizon_alignment_score))
        alignment_factor *= (0.85 + 0.15 * h_val)
    likelihood_factor = 0.85 + 0.15 * p_i

    raw_conf = o_val * (0.55 + 0.45 * e_val) * alignment_factor * intensity_factor * likelihood_factor
    if not math.isfinite(raw_conf):
        return 0.0
    if o_val <= 0.01:
        return 0.0
    return max(0.05, min(0.95, raw_conf))


def _scenario_confidences_from_outputs(
    overall: float,
    scenarios: Dict[str, Dict[str, Any]],
    agreement_score: float,
    horizon_alignment_score: float | None = None,
) -> Dict[str, float]:
    likelihoods: Dict[str, float] = {}
    for name in ("best_case", "base_case", "worst_case"):
        try:
            likelihoods[name] = clamp01(float((scenarios.get(name) or {}).get("likelihood") or 0.0))
        except Exception:
            likelihoods[name] = 0.0
    ordered = sorted(likelihoods.values(), reverse=True)
    edge = (ordered[0] - ordered[1]) if len(ordered) >= 2 else 0.0
    edge_score = clamp01(edge / 0.20)

    out: Dict[str, float] = {}
    for name in ("best_case", "base_case", "worst_case"):
        intensity = (scenarios.get(name) or {}).get("intensity")
        try:
            i_val = float(intensity)
        except Exception:
            i_val = 0.5
        out[name] = _scenario_confidence(
            overall=float(overall),
            scenario_likelihood=likelihoods.get(name, 0.0),
            scenario_intensity=i_val,
            edge_score=edge_score,
            agreement_score=agreement_score,
            horizon_alignment_score=horizon_alignment_score,
        )
    return out


def _debug_confidence_case(
    name: str,
    overall_post_penalties: float,
    scenario_likelihood: float,
    scenario_intensity: float,
    edge_score: float,
    agreement_score: float,
    horizon_alignment_score: float | None,
    scenario_confidence: float,
) -> None:
    if os.environ.get("TRIDENT_DEBUG_CONF") != "1":
        return
    h_txt = (
        f"{float(horizon_alignment_score):.3f}"
        if isinstance(horizon_alignment_score, (int, float))
        else "none"
    )
    print(
        "[scenario_confidence] "
        f"{name} overall_post_penalties={overall_post_penalties:.3f} "
        f"likelihood={scenario_likelihood:.3f} intensity={scenario_intensity:.3f} "
        f"edge_score={edge_score:.3f} agreement={agreement_score:.3f} horizon_alignment={h_txt} "
        f"scenario_confidence={scenario_confidence:.3f}",
        file=sys.stderr,
    )


def _ensure_analysis_runs_table(con) -> None:
    con.execute(
        """
        CREATE TABLE IF NOT EXISTS analysis_runs (
            run_ts_utc TIMESTAMP,
            symbol VARCHAR,
            run_id VARCHAR,
            snapshot_id VARCHAR,
            input_hash VARCHAR,
            confidence_overall DOUBLE,
            scenario_conf_best DOUBLE,
            scenario_conf_base DOUBLE,
            scenario_conf_worst DOUBLE,
            scenario_like_best DOUBLE,
            scenario_like_base DOUBLE,
            scenario_like_worst DOUBLE,
            scenario_intensity_best DOUBLE,
            scenario_intensity_base DOUBLE,
            scenario_intensity_worst DOUBLE,
            PRIMARY KEY(run_ts_utc, symbol)
        );
        """
    )
    try:
        con.execute("ALTER TABLE analysis_runs ADD COLUMN IF NOT EXISTS run_id VARCHAR;")
        con.execute("ALTER TABLE analysis_runs ADD COLUMN IF NOT EXISTS snapshot_id VARCHAR;")
        con.execute("ALTER TABLE analysis_runs ADD COLUMN IF NOT EXISTS input_hash VARCHAR;")
        con.execute("ALTER TABLE analysis_runs ADD COLUMN IF NOT EXISTS scenario_like_best DOUBLE;")
        con.execute("ALTER TABLE analysis_runs ADD COLUMN IF NOT EXISTS scenario_like_base DOUBLE;")
        con.execute("ALTER TABLE analysis_runs ADD COLUMN IF NOT EXISTS scenario_like_worst DOUBLE;")
        con.execute("ALTER TABLE analysis_runs ADD COLUMN IF NOT EXISTS scenario_intensity_best DOUBLE;")
        con.execute("ALTER TABLE analysis_runs ADD COLUMN IF NOT EXISTS scenario_intensity_base DOUBLE;")
        con.execute("ALTER TABLE analysis_runs ADD COLUMN IF NOT EXISTS scenario_intensity_worst DOUBLE;")
    except Exception:
        # Best-effort migration; don't fail if unsupported.
        pass


def _insert_analysis_run(
    con,
    run_ts_utc: datetime,
    symbol: str,
    run_id: Optional[str],
    snapshot_id: Optional[str],
    input_hash: Optional[str],
    confidence_overall: float,
    scenario_conf_best: float,
    scenario_conf_base: float,
    scenario_conf_worst: float,
    scenario_like_best: float | None,
    scenario_like_base: float | None,
    scenario_like_worst: float | None,
    scenario_intensity_best: float | None,
    scenario_intensity_base: float | None,
    scenario_intensity_worst: float | None,
) -> None:
    _ensure_analysis_runs_table(con)
    exists = con.execute(
        """
        SELECT 1
        FROM analysis_runs
        WHERE run_ts_utc = ? AND symbol = ?
        """,
        [run_ts_utc, symbol],
    ).fetchone()
    if exists:
        return
    con.execute(
        """
        INSERT INTO analysis_runs
        (
            run_ts_utc,
            symbol,
            run_id,
            snapshot_id,
            input_hash,
            confidence_overall,
            scenario_conf_best,
            scenario_conf_base,
            scenario_conf_worst,
            scenario_like_best,
            scenario_like_base,
            scenario_like_worst,
            scenario_intensity_best,
            scenario_intensity_base,
            scenario_intensity_worst
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        [
            run_ts_utc,
            symbol,
            run_id,
            snapshot_id,
            input_hash,
            confidence_overall,
            scenario_conf_best,
            scenario_conf_base,
            scenario_conf_worst,
            scenario_like_best,
            scenario_like_base,
            scenario_like_worst,
            scenario_intensity_best,
            scenario_intensity_base,
            scenario_intensity_worst,
        ],
    )


def _fetch_recent_runs(con, symbol: str, limit: int) -> List[Tuple[Any, ...]]:
    _ensure_analysis_runs_table(con)
    rows = con.execute(
        """
        SELECT run_ts_utc, confidence_overall, scenario_conf_best, scenario_conf_base, scenario_conf_worst
        FROM analysis_runs
        WHERE symbol = ?
        ORDER BY run_ts_utc DESC
        LIMIT ?
        """,
        [symbol, limit],
    ).fetchall()
    return rows or []


def _fetch_recent_scenario_history(con, symbol: str, limit: int) -> List[Tuple[Any, ...]]:
    _ensure_analysis_runs_table(con)
    rows = con.execute(
        """
        SELECT run_ts_utc,
               scenario_like_best, scenario_like_base, scenario_like_worst,
               scenario_intensity_best, scenario_intensity_base, scenario_intensity_worst
        FROM analysis_runs
        WHERE symbol = ?
        ORDER BY run_ts_utc DESC
        LIMIT ?
        """,
        [symbol, limit],
    ).fetchall()
    return rows or []


def _compute_confidence_trend(
    con,
    symbol: str,
    current_overall: float,
    penalties_sum: float,
    agreement_score: float,
    lookback: int = 5,
) -> Tuple[Dict[str, Any], Dict[str, float] | None]:
    runs = _fetch_recent_runs(con, symbol, lookback)
    lookback_runs_fast = len(runs)
    prev_scenarios = None
    if lookback_runs_fast >= 2:
        _, _, prev_best, prev_base, prev_worst = runs[1]
        prev_scenarios = {
            "best_case": float(prev_best),
            "base_case": float(prev_base),
            "worst_case": float(prev_worst),
        }

    if lookback_runs_fast < 4:
        return (
            {
                "lookback_runs_fast": lookback_runs_fast,
                "previous_avg_fast": None,
                "delta_1": None,
                "delta_fast": None,
                "sigma_fast": None,
                "z_delta_fast": None,
                "slope_per_hour_fast": None,
                "slope_24h_fast": None,
                "threshold_used": None,
                "label": "insufficient_history",
                "explanation": "Insufficient history to compute confidence trend.",
            },
            prev_scenarios,
        )

    current = runs[0]
    prior = runs[1:]
    prev_overall = float(prior[0][1])
    prior_vals = [float(r[1]) for r in prior]
    previous_avg_fast = sum(prior_vals) / len(prior_vals)
    delta_1 = current_overall - prev_overall
    delta_fast = current_overall - previous_avg_fast
    mean = previous_avg_fast
    variance = sum((v - mean) ** 2 for v in prior_vals) / max(len(prior_vals), 1)
    sigma_fast = math.sqrt(variance)
    if sigma_fast < 0.02:
        sigma_fast = 0.02
    z_delta_fast = delta_fast / sigma_fast

    runs_asc = sorted(runs, key=lambda r: r[0])
    t0 = runs_asc[0][0]
    xs = []
    ys = []
    for ts, val, *_ in runs_asc:
        hours = (ts - t0).total_seconds() / 3600.0
        xs.append(hours)
        ys.append(float(val))
    x_bar = sum(xs) / len(xs)
    y_bar = sum(ys) / len(ys)
    denom = sum((x - x_bar) ** 2 for x in xs)
    if denom == 0:
        slope = 0.0
    else:
        slope = sum((x - x_bar) * (y - y_bar) for x, y in zip(xs, ys)) / denom
    slope_24h = slope * 24.0
    threshold_used = max(0.02, 0.8 * sigma_fast)

    if slope_24h > threshold_used:
        label = "improving"
    elif slope_24h < -threshold_used:
        label = "deteriorating"
    else:
        label = "stable"

    if label == "deteriorating" and penalties_sum > 0.10:
        explanation = "Confidence is deteriorating and penalty load is high."
    elif agreement_score < 0.55:
        explanation = "Confidence is pressured by weak signal agreement."
    elif label == "improving":
        explanation = "Confidence is improving across recent runs."
    else:
        explanation = "Confidence is stable within recent variability."

    return (
        {
            "lookback_runs_fast": lookback_runs_fast,
            "previous_avg_fast": round(previous_avg_fast, 3),
            "delta_1": round(delta_1, 3),
            "delta_fast": round(delta_fast, 3),
            "sigma_fast": round(sigma_fast, 3),
            "z_delta_fast": round(z_delta_fast, 3),
            "slope_per_hour_fast": round(slope, 3),
            "slope_24h_fast": round(slope_24h, 3),
            "threshold_used": round(threshold_used, 3),
            "label": label,
            "explanation": explanation,
        },
        prev_scenarios,
    )


def get_latest_fng(con) -> Dict[str, Any] | None:
    """Return latest Fear & Greed Index row or None."""
    try:
        row = con.execute(
            """
            SELECT ts_utc, value, label, source, fetched_at_utc
            FROM fear_greed_index
            WHERE source = ?
            ORDER BY ts_utc DESC
            LIMIT 1
            """,
            ["alternative_me"],
        ).fetchone()
    except Exception:
        return None
    if not row:
        return None
    ts_utc, value, label, source, fetched_at_utc = row
    return {
        "ts_utc": ts_utc,
        "value": value,
        "label": label,
        "source": source,
        "fetched_at_utc": fetched_at_utc,
    }


def _compute_sentiment_regime(snapshot: Dict[str, Any]) -> Dict[str, Any]:
    fng = snapshot.get("macro_sentiment", {}).get("fear_greed", {}) or {}
    value = fng.get("value")
    label = fng.get("label")
    if value is None:
        return {
            "fear_greed_value": None,
            "fear_greed_label": None,
            "sentiment_extreme": None,
            "convexity_boost": None,
            "policy": "convexity_only_no_direction",
        }
    sentiment_extreme = abs(float(value) - 50.0) / 50.0
    convexity_boost = min(0.08, 0.15 * sentiment_extreme)
    return {
        "fear_greed_value": int(value),
        "fear_greed_label": label,
        "sentiment_extreme": round(sentiment_extreme, 3),
        "convexity_boost": round(convexity_boost, 3),
        "policy": "convexity_only_no_direction",
    }


def _apply_convexity(
    likelihoods: Dict[str, float],
    sentiment_regime: Dict[str, Any],
) -> Dict[str, float]:
    value = sentiment_regime.get("fear_greed_value")
    if value is None:
        return likelihoods
    sentiment_extreme = abs(float(value) - 50.0) / 50.0
    convexity_boost = min(0.08, 0.15 * sentiment_extreme)

    best = float(likelihoods.get("best_case", 0.0))
    base = float(likelihoods.get("base_case", 0.0))
    worst = float(likelihoods.get("worst_case", 0.0))

    base -= convexity_boost
    best += convexity_boost / 2.0
    worst += convexity_boost / 2.0

    best = max(best, 0.05)
    base = max(base, 0.05)
    worst = max(worst, 0.05)

    total = best + base + worst
    best /= total
    base /= total
    worst /= total

    best = round(best, 3)
    base = round(base, 3)
    worst = round(worst, 3)
    diff = round(1.0 - (best + base + worst), 3)
    base = round(clamp01(base + diff), 3)

    return {
        "best_case": best,
        "base_case": base,
        "worst_case": worst,
    }


def _apply_liquidity_convexity(
    likelihoods: Dict[str, float],
    macro_liquidity: Dict[str, Any] | None,
    macro_calendar: Dict[str, Any] | None,
) -> Tuple[Dict[str, float], float, float | None]:
    if not macro_liquidity:
        return likelihoods, 0.0, None
    strength = macro_liquidity.get("tightening_score")
    confidence = macro_liquidity.get("confidence")
    if strength is None or confidence is None:
        return likelihoods, 0.0, None
    try:
        strength_val = float(strength)
        conf_val = float(confidence)
    except Exception:
        return likelihoods, 0.0, None

    distance = abs(strength_val - 0.5) / 0.5
    boost = min(0.06, 0.10 * distance * clamp01(conf_val))
    if macro_calendar:
        next_high = macro_calendar.get("next_high_event_in_hours")
        try:
            if next_high is not None and float(next_high) <= 72.0:
                boost = min(0.06, boost * 1.25)
        except Exception:
            pass
    if boost <= 0:
        return likelihoods, 0.0, distance

    best = float(likelihoods.get("best_case", 0.0))
    base = float(likelihoods.get("base_case", 0.0))
    worst = float(likelihoods.get("worst_case", 0.0))

    base -= boost
    best += boost / 2.0
    worst += boost / 2.0

    best = max(best, 0.05)
    base = max(base, 0.05)
    worst = max(worst, 0.05)

    total = best + base + worst
    best /= total
    base /= total
    worst /= total

    best = round(best, 3)
    base = round(base, 3)
    worst = round(worst, 3)
    diff = round(1.0 - (best + base + worst), 3)
    base = round(clamp01(base + diff), 3)

    return {
        "best_case": best,
        "base_case": base,
        "worst_case": worst,
    }, boost, distance




def compute_freshness(symbol: str, con, window_hours: int) -> Dict[str, Any]:
    """
    Compute average age (hours) for tech, news, fundamentals, plus news count used.
    """
    now_utc = datetime.now(timezone.utc)

    def _to_utc(ts):
        if ts is None:
            return None
        if isinstance(ts, str):
            try:
                ts = datetime.fromisoformat(ts.replace("Z", "+00:00"))
            except Exception:
                return None
        if ts.tzinfo is None:
            return ts.replace(tzinfo=timezone.utc)
        return ts.astimezone(timezone.utc)

    # Tech freshness: use the most recent candle across 15m/30m/1h
    tech_dt = None
    for interval in ("15m", "30m", "1h"):
        tech_ts = con.execute(
            """
            SELECT MAX(ts)
            FROM candles
            WHERE symbol = ? AND interval = ?
            """,
            [symbol, interval],
        ).fetchone()
        candidate = _to_utc(tech_ts[0]) if tech_ts else None
        if candidate and (tech_dt is None or candidate > tech_dt):
            tech_dt = candidate
    tech_age = (now_utc - tech_dt).total_seconds() / 3600.0 if tech_dt else None

    # Fundamentals freshness: latest snapshot
    fund_ts = con.execute(
        """
        SELECT MAX(ts)
        FROM fundamentals
        WHERE symbol = ?
        """,
        [symbol],
    ).fetchone()
    fund_dt = _to_utc(fund_ts[0]) if fund_ts else None
    fund_age = (now_utc - fund_dt).total_seconds() / 3600.0 if fund_dt else None

    # News freshness: windowed rows that are actually used
    cutoff = now_utc - timedelta(hours=window_hours)
    news_rows = []
    try:
        news_rows = con.execute(
            """
            SELECT published_at, ai_meta, relevance
            FROM news_items
            WHERE symbol = ? AND published_at >= ?
            """,
            [symbol, cutoff],
        ).fetchall()
    except Exception:
        try:
            news_rows = con.execute(
                """
                SELECT published_at, ai_meta
                FROM news_items
                WHERE symbol = ? AND published_at >= ?
                """,
                [symbol, cutoff],
            ).fetchall()
        except Exception:
            news_rows = con.execute(
                """
                SELECT published_at
                FROM news_items
                WHERE symbol = ? AND published_at >= ?
                """,
                [symbol, cutoff],
            ).fetchall()

    ages: list[float] = []
    count_used = 0
    for row in news_rows:
        if len(row) == 3:
            published_at, ai_meta, relevance = row
            if relevance == 0:
                continue
        elif len(row) == 2:
            published_at, ai_meta = row
            relevance = None
        else:
            published_at = row[0]
            ai_meta = None
            relevance = None

        ai_data = {}
        if ai_meta:
            try:
                ai_data = json.loads(ai_meta)
            except Exception:
                ai_data = {}
        if relevance is None and ai_data.get("relevance") == 0:
            continue

        ts = _to_utc(published_at)
        if ts is None:
            continue
        age = (now_utc - ts).total_seconds() / 3600.0
        ages.append(age)
        count_used += 1

    news_age = (sum(ages) / len(ages)) if ages else None

    return {
        "tech_avg_age_hours": tech_age,
        "news_avg_age_hours": news_age,
        "fundamentals_avg_age_hours": fund_age,
        "news_count_used": count_used,
    }


def _freshness_score(freshness: Dict[str, Any]) -> Tuple[float, Dict[str, Any]]:
    """
    Compute a deterministic freshness_score in [0,1] with modality reweighting.
    Returns (freshness_score, freshness_detail).
    """
    tech_hl = 6.0
    news_hl = 24.0
    fund_hl = 72.0

    tech_age = freshness.get("tech_avg_age_hours")
    news_age = freshness.get("news_avg_age_hours")
    fund_age = freshness.get("fundamentals_avg_age_hours")
    news_count_used = int(freshness.get("news_count_used") or 0)

    subscores: Dict[str, float] = {}
    if tech_age is not None:
        subscores["tech"] = math.exp(-math.log(2) * tech_age / tech_hl)
    if news_age is not None and news_count_used > 0:
        subscores["news"] = math.exp(-math.log(2) * news_age / news_hl)
    if fund_age is not None:
        subscores["fundamentals"] = math.exp(-math.log(2) * fund_age / fund_hl)

    base_weights = {
        "tech": 0.45,
        "news": 0.25,
        "fundamentals": 0.30,
    }

    if not subscores:
        detail = {
            "tech_weight_used": 0.0,
            "news_weight_used": 0.0,
            "fundamentals_weight_used": 0.0,
            "tech_avg_age_hours": tech_age,
            "news_avg_age_hours": news_age,
            "fundamentals_avg_age_hours": fund_age,
        }
        return 0.0, detail

    weight_sum = sum(base_weights[k] for k in subscores.keys()) or 1.0
    weights_used = {k: base_weights[k] / weight_sum for k in subscores.keys()}
    score = sum(subscores[k] * weights_used[k] for k in subscores.keys())

    detail = {
        "tech_weight_used": round(weights_used.get("tech", 0.0), 3),
        "news_weight_used": round(weights_used.get("news", 0.0), 3),
        "fundamentals_weight_used": round(weights_used.get("fundamentals", 0.0), 3),
        "tech_avg_age_hours": tech_age,
        "news_avg_age_hours": news_age,
        "fundamentals_avg_age_hours": fund_age,
    }
    return round(float(score), 3), detail


def _compute_intensities(snapshot: Dict[str, Any]) -> Dict[str, float]:
    tech = snapshot.get("technical_features", {}) or {}
    news = snapshot.get("news_features", {}) or {}
    fund = snapshot.get("fundamental_features", {}) or {}

    atr = float(_get_tech_value(tech, "atr_pct") or 0.0)
    trend = abs(float(_get_tech_value(tech, "trend_strength") or 0.0))
    macd = abs(float(_get_tech_value(tech, "macd_hist") or 0.0))
    dhi = abs(float(_get_tech_value(tech, "dist_from_local_high") or 0.0))
    dlo = abs(float(_get_tech_value(tech, "dist_from_local_low") or 0.0))

    weighted_bullish = float(news.get("weighted_bullish") or 0.0)
    weighted_bearish = float(news.get("weighted_bearish") or 0.0)
    wraw = abs(weighted_bullish - weighted_bearish)
    acnt = float(news.get("article_count") or 0)

    pc24 = abs(_parse_float(fund.get("pct_change_24h")) or 0.0)
    vmr = abs(_parse_float(fund.get("vol_mcap_ratio")) or 0.0)
    vwap_dist = abs(_parse_float(_get_tech_value(tech, "vwap_distance_pct")) or 0.0)
    rs_vs_btc = abs(_parse_float(_get_tech_value(tech, "rs_vs_btc_7d")) or 0.0)
    breadth = _parse_float(_get_tech_value(tech, "breadth_score"))
    squeeze_on = bool(_get_tech_value(tech, "squeeze_on") is True)
    squeeze_fired = bool(_get_tech_value(tech, "squeeze_fired") is True)

    atr_n = clamp01(atr / 0.02)
    trend_n = clamp01(trend / 0.02)
    macd_n = clamp01(macd / 500.0)
    dist_n = clamp01((dhi + dlo) / 0.20)

    wraw_n = clamp01(wraw / 20.0)
    acnt_n = clamp01(math.log1p(acnt) / math.log1p(100))

    pc24_n = clamp01(pc24 / 10.0)
    vmr_n = clamp01(vmr / 0.20)
    vwap_n = clamp01(vwap_dist / 0.03)
    rs_n = clamp01(rs_vs_btc / 0.05)
    breadth_n = 0.0 if breadth is None else clamp01(abs(float(breadth) - 0.5) / 0.5)

    tech_I = clamp01(
        0.35 * atr_n
        + 0.25 * trend_n
        + 0.12 * macd_n
        + 0.10 * dist_n
        + 0.08 * vwap_n
        + 0.05 * rs_n
        + 0.05 * breadth_n
    )
    news_I = clamp01(0.60 * wraw_n + 0.40 * acnt_n)
    fund_I = clamp01(0.70 * pc24_n + 0.30 * vmr_n)

    I0 = clamp01(0.45 * tech_I + 0.35 * news_I + 0.20 * fund_I)
    if squeeze_on and not squeeze_fired:
        I0 = clamp01(I0 * 0.88)

    base_I = clamp01(I0 * 0.85)
    best_I = clamp01(I0 * (1.00 + 0.20 * wraw_n))
    worst_I = clamp01(I0 * (1.00 + 0.25 * macd_n))
    if squeeze_fired:
        best_I = clamp01(best_I * 1.08)
        worst_I = clamp01(worst_I * 1.08)
        base_I = clamp01(base_I * 0.95)

    return {
        "best_case": round(best_I, 3),
        "base_case": round(base_I, 3),
        "worst_case": round(worst_I, 3),
    }


def _confidence_block(
    snapshot: Dict[str, Any],
    scenarios: Dict[str, Any],
    freshness_score: float,
) -> Dict[str, Any]:
    DEFAULT_ATTRIBUTION_WEIGHTS = {
        "agreement": 0.40,
        "freshness": 0.25,
        "coverage": 0.15,
        "volatility": 0.20,
    }
    penalty_explanations = {
        "TECH_STALE": "Technicals are based on old candles; signal may lag the current market.",
        "TECH_MISSING": "Technical data is missing; directional confidence is reduced.",
        "NEWS_STALE": "News inputs are aging; sentiment may not reflect the latest context.",
        "FUND_STALE": "Fundamentals are outdated; on-chain context may have shifted.",
        "FUND_MISSING": "Fundamental data is missing; structural confidence is reduced.",
        "NEWS_STARVATION": "No recent news in window; sentiment is under-informed.",
        "LOW_COVERAGE": "Few relevant articles; news signal is less reliable.",
        "LOW_AGREEMENT": "Signals disagree; combined confidence is weaker.",
        "MODEL_EDGE_WEAK": "Top scenarios are too close in likelihood; ranking is unstable.",
        "HIGH_VOLATILITY": "Volatility is elevated; outcomes are less predictable.",
        "LOW_LIQUIDITY": "Liquidity is weak; price moves may be erratic.",
        "RANGE_TRAP_RISK": "Weak trend with flat returns; price may be range-bound.",
        "NEWS_PRICE_DIVERGENCE": "News skew conflicts with recent price direction.",
        "HORIZON_CONFLICT": "Time horizons disagree; short-term strength may be counter-trend versus higher timeframe.",
        "DUPLICATION_RISK": "Headline clustering may overstate a single narrative.",
        "SOURCE_CONCENTRATION": "Coverage is concentrated in few sources; bias risk higher.",
        "CATEGORY_SKEW": "News is skewed to one category; context may be unbalanced.",
        "EXTREME_FEAR_RISK": "Extreme fear can widen downside tails and raise fragility risk.",
        "FEAR_RISK": "Elevated fear can increase downside tail risk.",
        "GREED_RISK": "Elevated greed can increase downside reversal risk.",
        "EXTREME_GREED_RISK": "Extreme greed can increase reversal risk and tail outcomes.",
        "FED_LIQUIDITY_TIGHTENING": "Fed liquidity is tightening, which can increase tail risk.",
        "FED_LIQUIDITY_EASING": "Fed liquidity is easing, which can reduce stress but add convexity.",
        "FED_LIQUIDITY_STALE": "Fed liquidity data is stale; macro context is less reliable.",
        "LIQUIDITY_STALE": "Fed liquidity data is stale; confidence in macro context is reduced.",
        "LIQUIDITY_LOW_CONFIDENCE": "Fed liquidity signals are weak or mixed; confidence is reduced.",
        "WEAK_BREADTH": "Breadth across the active symbol set is weak, reducing confidence in directional follow-through.",
        "RELATIVE_WEAKNESS": "Relative strength versus BTC is weak, increasing underperformance risk for long setups.",
        "PRE_BREAKOUT_COMPRESSION": "Price is compressed inside a squeeze; direction is less reliable until expansion confirms.",
    }
    tech = snapshot.get("technical_features", {}) or {}
    news = snapshot.get("news_features", {}) or {}
    meta = snapshot.get("meta_features", {}) or {}
    freshness = meta.get("freshness", {}) or {}

    agreement_score = meta.get("agreement_score")
    try:
        agreement_score = float(agreement_score)
    except Exception:
        agreement_score = 0.5
    agreement_score = clamp01(agreement_score)
    horizon_alignment = _horizon_alignment_from_multi_horizon(tech)
    if horizon_alignment is not None:
        alignment_raw = float(horizon_alignment.get("horizon_alignment_score") or 0.0)
        agreement_score = clamp01(0.70 * agreement_score + 0.30 * alignment_raw)

    freshness_component = clamp01(freshness_score)
    news_count_used = int(freshness.get("news_count_used") or 0)
    coverage_component = clamp01(min(1.0, news_count_used / 30.0))

    atr_pct = _get_tech_value(tech, "atr_pct")
    if atr_pct is None:
        volatility_component = 0.5
    else:
        volatility_component = clamp01(1.0 - min(1.0, float(atr_pct) / 0.03))

    base_conf = (
        0.35 * agreement_score
        + 0.30 * freshness_component
        + 0.20 * coverage_component
        + 0.15 * volatility_component
    )

    penalties = {
        "data_quality": [],
        "signal_consistency": [],
        "market_structure": [],
        "macro_event": [],
        "sentiment_regime": [],
        "macro_liquidity": [],
    }

    def add_penalty(group: str, code: str, severity: str, impact: float, detail: str) -> None:
        penalties[group].append(
            {
                "code": code,
                "severity": severity,
                "impact": impact,
                "detail": detail,
                "explanation": penalty_explanations.get(code, "(missing explanation)"),
            }
        )

    # Data quality & coverage
    tech_age = freshness.get("tech_avg_age_hours")
    if tech_age is None:
        add_penalty(
            "data_quality",
            "TECH_MISSING",
            "high",
            -0.20,
            "tech_avg_age_hours=None",
        )
    else:
        if tech_age > 12.0:
            add_penalty(
                "data_quality",
                "TECH_STALE",
                "high",
                -0.18,
                f"tech_avg_age_hours={tech_age:.2f}h > 12.00h (high)",
            )
        elif tech_age > 6.0:
            add_penalty(
                "data_quality",
                "TECH_STALE",
                "medium",
                -0.10,
                f"tech_avg_age_hours={tech_age:.2f}h > 6.00h (medium)",
            )

    news_age = freshness.get("news_avg_age_hours")
    if news_age is not None:
        if news_age > 36.0:
            add_penalty(
                "data_quality",
                "NEWS_STALE",
                "medium",
                -0.07,
                f"news_avg_age_hours={news_age:.2f}h > 36.00h (medium)",
            )
        elif news_age > 24.0:
            add_penalty(
                "data_quality",
                "NEWS_STALE",
                "low",
                -0.05,
                f"news_avg_age_hours={news_age:.2f}h > 24.00h (low)",
            )

    fund_age = freshness.get("fundamentals_avg_age_hours")
    if fund_age is None:
        add_penalty(
            "data_quality",
            "FUND_MISSING",
            "medium",
            -0.10,
            "fundamentals_avg_age_hours=None",
        )
    elif fund_age >= REQUIRED_MODALITY_MAX_AGE_HOURS:
        add_penalty(
            "data_quality",
            "FUND_STALE",
            "high",
            -0.12,
            f"fundamentals_avg_age_hours={fund_age:.2f}h >= {REQUIRED_MODALITY_MAX_AGE_HOURS:.2f}h",
        )

    if news_count_used == 0:
        add_penalty("data_quality", "NEWS_STARVATION", "high", -0.15, "news_count_used=0")
    elif 0 < news_count_used < 10:
        add_penalty("data_quality", "LOW_COVERAGE", "low", -0.05, f"news_count_used={news_count_used}")

    # Macro liquidity
    macro_liquidity_raw = snapshot.get("macro_liquidity", {}) or {}
    macro_liquidity, _, staleness_label = _macro_liquidity_block(macro_liquidity_raw)
    regime = macro_liquidity.get("regime") if macro_liquidity else None
    tightening_score = macro_liquidity.get("tightening_score") if macro_liquidity else None
    liquidity_conf = macro_liquidity.get("confidence") if macro_liquidity else None
    age_hours = macro_liquidity.get("age_hours") if macro_liquidity else None

    if isinstance(tightening_score, (int, float)):
        if regime == "qt_like":
            if tightening_score >= 0.65:
                add_penalty(
                    "macro_liquidity",
                    "FED_LIQUIDITY_TIGHTENING",
                    "high",
                    -0.10,
                    f"regime=qt_like score={tightening_score:.3f} conf={liquidity_conf}",
                )
            elif tightening_score >= 0.55:
                add_penalty(
                    "macro_liquidity",
                    "FED_LIQUIDITY_TIGHTENING",
                    "medium",
                    -0.05,
                    f"regime=qt_like score={tightening_score:.3f} conf={liquidity_conf}",
                )
    if isinstance(age_hours, (int, float)):
        if age_hours > 336:
            add_penalty(
                "macro_liquidity",
                "LIQUIDITY_STALE",
                "high",
                -0.10,
                f"age_h={age_hours:.1f}",
            )
        elif age_hours > 168:
            add_penalty(
                "macro_liquidity",
                "LIQUIDITY_STALE",
                "medium",
                -0.05,
                f"age_h={age_hours:.1f}",
            )
    if isinstance(liquidity_conf, (int, float)) and liquidity_conf < 0.45:
        agreement_ratio = macro_liquidity.get("agreement_ratio") if macro_liquidity else None
        add_penalty(
            "macro_liquidity",
            "LIQUIDITY_LOW_CONFIDENCE",
            "medium",
            -0.05,
            f"conf={liquidity_conf:.2f} agreement={agreement_ratio}",
        )


    # Signal agreement & consistency
    if agreement_score < 0.45:
        add_penalty("signal_consistency", "LOW_AGREEMENT", "high", -0.12, f"agreement_score={agreement_score:.2f}")
    elif agreement_score < 0.55:
        add_penalty("signal_consistency", "LOW_AGREEMENT", "medium", -0.07, f"agreement_score={agreement_score:.2f}")
    if horizon_alignment is not None:
        label = horizon_alignment.get("horizon_alignment_label")
        if label == "conflict":
            add_penalty(
                "signal_consistency",
                "HORIZON_CONFLICT",
                "high",
                -0.10,
                _horizon_alignment_detail(horizon_alignment),
            )

    ema_cross = (_get_tech_value(tech, "ema_cross") or "neutral").lower()
    ema50_above_200 = _get_tech_value(tech, "ema50_above_200")
    ret_7d = _get_tech_value(tech, "ret_7d")
    macd_hist = _get_tech_value(tech, "macd_hist")
    rsi14 = _get_tech_value(tech, "rsi14")

    case_a = (
        (ema_cross == "bullish" or ema50_above_200 is True)
        and ret_7d is not None
        and ret_7d < 0
        and ((macd_hist is not None and macd_hist < 0) or (rsi14 is not None and rsi14 < 45))
    )
    case_b = (
        (ema_cross == "bearish" and ema50_above_200 is False)
        and ret_7d is not None
        and ret_7d > 0
        and ((macd_hist is not None and macd_hist > 0) or (rsi14 is not None and rsi14 > 55))
    )
    if case_a or case_b:
        case_label = "A" if case_a else "B"
        detail = (
            f"case={case_label} ema_cross={ema_cross} ema50_above_200={ema50_above_200} "
            f"ret_7d={ret_7d} macd_hist={macd_hist} rsi14={rsi14}"
        )
        add_penalty("signal_consistency", "DIVERGENT_MOMENTUM", "medium", -0.06, detail)

    if news_count_used >= 10:
        weighted_bullish = float(news.get("weighted_bullish") or 0.0)
        weighted_bearish = float(news.get("weighted_bearish") or 0.0)
        news_skew = weighted_bullish - weighted_bearish
        ret_3d = _get_tech_value(tech, "ret_3d")
        if ret_3d is not None:
            if news_skew > 5 and ret_3d < 0:
                detail = f"news_skew={news_skew:.2f} ret_3d={ret_3d} wb={weighted_bullish} wbr={weighted_bearish}"
                add_penalty("signal_consistency", "NEWS_PRICE_DIVERGENCE", "medium", -0.06, detail)
            elif news_skew < -5 and ret_3d > 0:
                detail = f"news_skew={news_skew:.2f} ret_3d={ret_3d} wb={weighted_bullish} wbr={weighted_bearish}"
                add_penalty("signal_consistency", "NEWS_PRICE_DIVERGENCE", "medium", -0.06, detail)

    # Model edge weak
    likelihoods = {k: float(v.get("likelihood", 0.0)) for k, v in scenarios.items()}
    ordered = sorted(likelihoods.items(), key=lambda kv: kv[1], reverse=True)
    if len(ordered) >= 2:
        margin = ordered[0][1] - ordered[1][1]
        detail = f"margin={margin:.3f} best={ordered[0][1]:.3f} second={ordered[1][1]:.3f}"
        if margin < 0.05:
            add_penalty("signal_consistency", "MODEL_EDGE_WEAK", "high", -0.10, detail)
        elif margin < 0.10:
            add_penalty("signal_consistency", "MODEL_EDGE_WEAK", "medium", -0.06, detail)

    # Market structure & regime risk
    vol_regime = (_get_tech_value(tech, "vol_regime") or "").lower()
    atr_pct = _get_tech_value(tech, "atr_pct") or 0.0
    if vol_regime == "high" or atr_pct > 0.02:
        add_penalty("market_structure", "HIGH_VOLATILITY", "medium", -0.08, f"vol_regime={vol_regime} atr_pct={atr_pct}")

    vol_z = _get_tech_value(tech, "vol_z")
    if vol_z is not None and vol_z < -1.0:
        add_penalty("market_structure", "LOW_LIQUIDITY", "low", -0.05, f"vol_z={vol_z}")

    trend_strength = _get_tech_value(tech, "trend_strength") or 0.0
    ret_7d = _get_tech_value(tech, "ret_7d")
    if ret_7d is not None and trend_strength < 0.01 and abs(ret_7d) < 0.02:
        add_penalty(
            "market_structure",
            "RANGE_TRAP_RISK",
            "low",
            -0.04,
            f"trend_strength={trend_strength} ret_7d={ret_7d}",
        )

    breadth_score = _parse_float(_get_tech_value(tech, "breadth_score"))
    if breadth_score is not None:
        if breadth_score < 0.40:
            add_penalty("market_structure", "WEAK_BREADTH", "high", -0.08, f"breadth_score={breadth_score:.3f}")
        elif breadth_score < 0.50:
            add_penalty("market_structure", "WEAK_BREADTH", "medium", -0.05, f"breadth_score={breadth_score:.3f}")

    rs_vs_btc_7d = _parse_float(_get_tech_value(tech, "rs_vs_btc_7d"))
    if rs_vs_btc_7d is not None and rs_vs_btc_7d < -0.02:
        add_penalty("market_structure", "RELATIVE_WEAKNESS", "medium", -0.06, f"rs_vs_btc_7d={rs_vs_btc_7d:.4f}")

    squeeze_on = bool(_get_tech_value(tech, "squeeze_on") is True)
    vwap_distance_pct = _parse_float(_get_tech_value(tech, "vwap_distance_pct"))
    if squeeze_on and vwap_distance_pct is not None and abs(vwap_distance_pct) < 0.01:
        add_penalty(
            "market_structure",
            "PRE_BREAKOUT_COMPRESSION",
            "low",
            -0.04,
            f"squeeze_on={squeeze_on} vwap_distance_pct={vwap_distance_pct:.4f}",
        )

    # Macro / event risk (only if metadata exists)
    event_risk = meta.get("event_risk", {}) if isinstance(meta.get("event_risk"), dict) else {}
    macro_hours = event_risk.get("macro_event_hours")
    policy_hours = event_risk.get("policy_event_hours")
    if macro_hours is not None and macro_hours <= 72:
        add_penalty("macro_event", "MACRO_EVENT_IMMINENT", "medium", -0.06, f"macro_event_hours={macro_hours}")
    if policy_hours is not None and policy_hours <= 168:
        add_penalty("macro_event", "POLICY_EVENT_RISK", "low", -0.04, f"policy_event_hours={policy_hours}")

    # Sentiment regime (Fear & Greed Index)
    fng = snapshot.get("macro_sentiment", {}).get("fear_greed", {}) or {}
    fng_value = fng.get("value")
    fng_label = fng.get("label")
    fng_age = fng.get("age_hours")
    if fng_value is not None:
        detail = f"fng={fng_value} label={fng_label} age_h={fng_age}"
        if fng_value <= 20:
            add_penalty("sentiment_regime", "EXTREME_FEAR_RISK", "high", -0.10, detail)
        elif fng_value <= 30:
            add_penalty("sentiment_regime", "FEAR_RISK", "medium", -0.05, detail)
        elif fng_value >= 80:
            add_penalty("sentiment_regime", "EXTREME_GREED_RISK", "high", -0.10, detail)
        elif fng_value >= 70:
            add_penalty("sentiment_regime", "GREED_RISK", "medium", -0.05, detail)

    total_impact = 0.0
    for group in penalties.values():
        for p in group:
            total_impact += float(p.get("impact") or 0.0)

    if os.environ.get("TRIDENT_DEBUG_CONF") == "1":
        h_align = None
        if horizon_alignment is not None:
            try:
                h_align = float(horizon_alignment.get("horizon_alignment_score"))
            except Exception:
                h_align = None
        h_txt = f"{h_align:.3f}" if isinstance(h_align, (int, float)) else "none"
        print(
            "[confidence] "
            f"overall_conf_pre_penalties={base_conf:.3f} penalties_total={total_impact:.3f} "
            f"overall_conf_post_penalties={clamp01(base_conf + total_impact):.3f} "
            f"agreement_score={agreement_score:.3f} horizon_alignment_score={h_txt}",
            file=sys.stderr,
        )

    overall = clamp01(base_conf + total_impact)
    overall = round(overall, 3)

    components = {
        "agreement": round(agreement_score, 3),
        "freshness": round(freshness_component, 3),
        "coverage": round(coverage_component, 3),
        "volatility": round(volatility_component, 3),
    }

    penalties_contribution = 0.0
    for group in penalties.values():
        for p in group:
            penalties_contribution += float(p.get("impact") or 0.0)

    attribution = {}
    for key, weight in DEFAULT_ATTRIBUTION_WEIGHTS.items():
        score = float(components.get(key, 0.0))
        contribution = (score - 0.50) * weight
        attribution[key] = round(contribution, 3)
    attribution["penalties"] = round(penalties_contribution, 3)

    _ = sum(attribution.values())  # internal consistency check only

    return {
        "overall": overall,
        "components": components,
        "penalties": penalties,
        "attribution": attribution,
    }

def build_phase4_snapshot(symbol: str, con) -> Dict[str, Any]:
    """
    Build a canonical, feature-level snapshot for Phase 4.
    Contains only raw features (technical/news/fundamental/meta).
    """
    now_utc = datetime.now(timezone.utc)
    tech = compute_tech_features(symbol, con) or {}
    news = compute_news_features(symbol, con) or {}
    fund = compute_fundamentals_features(symbol, con) or {}

    tech_bias = tech.get("ema_cross") or "neutral"
    tech_strength = tech.get("trend_strength") or 0.0
    news_bias = _bias_from_direction(news.get("direction"))
    news_strength = news.get("intensity", 0.0) or 0.0
    fund_change = _parse_pct(fund.get("mcap_change_1d"))
    fund_bias = "neutral"
    fund_strength = 0.0
    if fund_change is not None:
        if fund_change > 0:
            fund_bias = "bullish"
        elif fund_change < 0:
            fund_bias = "bearish"
        fund_strength = abs(fund_change)

    agreement_score = _continuous_agreement_score(tech, news, fund)
    superscore = _clip((abs(tech_strength) + abs(news_strength) + abs(fund_strength)) / 3.0)

    window_hours = int(news.get("window_hours") or 48)
    freshness = compute_freshness(symbol, con, window_hours)
    fng_row = get_latest_fng(con)
    if fng_row and fng_row.get("ts_utc"):
        ts_utc = fng_row["ts_utc"]
        if ts_utc.tzinfo is None:
            ts_utc = ts_utc.replace(tzinfo=timezone.utc)
        else:
            ts_utc = ts_utc.astimezone(timezone.utc)
        age_hours = (now_utc - ts_utc).total_seconds() / 3600.0
        fng = {
            "value": int(fng_row.get("value")),
            "label": fng_row.get("label"),
            "ts_utc": ts_utc.isoformat(),
            "age_hours": round(age_hours, 2),
            "source": "alternative_me",
        }
    else:
        fng = {
            "value": None,
            "label": None,
            "ts_utc": None,
            "age_hours": None,
            "source": "alternative_me",
        }

    def _build_summary_list(items: list[dict], limit: int | None = 15) -> list[dict]:
        summaries = []
        for item in items or []:
            source = item.get("source") or ""
            summaries.append(
                {
                    "source": source,
                    "title": item.get("title") or "",
                    "published_at": item.get("published_at"),
                    "summary": item.get("summary") or "",
                    "source_quality": get_source_quality(source),
                }
            )
        summaries.sort(
            key=lambda x: (-x["source_quality"], x.get("published_at") or "", x.get("title") or "")
        )
        if limit is None:
            return summaries
        return summaries[:limit]

    try:
        cal_features = compute_calendar_features(
            con,
            now_utc,
            lookback_hours=168,
            lookahead_hours=168,
            min_impact="medium",
            max_upcoming_items=25,
            max_recent_items=10,
            debug=False,
        )
    except Exception:
        cal_features = {}

    try:
        fed_liquidity = compute_fed_liquidity_features_v2(con)
    except Exception:
        fed_liquidity = {}

    macro_calendar, macro_calendar_raw = _build_macro_calendar(cal_features)
    required_modalities = evaluate_required_modality_status(freshness)

    hash_payload = {
        "symbol": symbol,
        "technical_features": tech,
        "news_features": news,
        "fundamental_features": fund,
        "freshness": freshness,
        "macro_calendar": macro_calendar,
        "macro_liquidity": fed_liquidity,
    }
    input_hash = _stable_json_hash(hash_payload)
    snapshot_id = f"snap_{input_hash[:16]}"
    run_id = os.getenv("TRIDENT_RUN_ID") or _new_run_id("snap")
    created_at_utc = now_utc.replace(microsecond=0).isoformat().replace("+00:00", "Z")

    snapshot = {
        "symbol": symbol,
        "run_id": run_id,
        "snapshot_id": snapshot_id,
        "input_hash": input_hash,
        "provenance": {
            "run_id": run_id,
            "snapshot_id": snapshot_id,
            "input_hash": input_hash,
            "created_at_utc": created_at_utc,
            "source": "phase4_snapshot_v2",
        },
        "technical_features": tech,
        "news_features": {
            "direction": news.get("direction"),
            "intensity": news.get("intensity"),
            "article_count": news.get("article_count"),
            "weighted_bullish": news.get("weighted_stats", {}).get("weighted_bullish"),
            "weighted_bearish": news.get("weighted_stats", {}).get("weighted_bearish"),
            "category_breakdown": news.get("category_breakdown"),
            "window_hours": window_hours,
        },
        "news_summaries": _build_summary_list(news.get("articles", []), limit=None),
        "macro_summaries": _build_summary_list(news.get("market_context", []), limit=None),
        "macro_sentiment": {
            "fear_greed": fng,
        },
        "macro_calendar": macro_calendar,
        "macro_calendar_raw": macro_calendar_raw,
        "macro_liquidity": fed_liquidity,
        "fundamental_features": fund,
        "meta_features": {
            "signal_freshness": news.get("signal_freshness", {}),  # may be empty if not provided
            "agreement_score": agreement_score,
            "superscore": superscore,
            "freshness": freshness,
            "required_modalities": required_modalities,
        },
    }
    return snapshot


def _scenario_from_features(
    name: str,
    snapshot: Dict[str, Any],
    likelihood: float,
    intensity: float,
) -> Dict[str, Any]:
    news = snapshot.get("news_features", {})
    tech = snapshot.get("technical_features", {})
    fund = snapshot.get("fundamental_features", {})
    meta = snapshot.get("meta_features", {})
    news_intensity = news.get("intensity") or 0.0

    # Build drivers referencing raw features
    drivers = [
        f"Technical ema_cross={tech.get('ema_cross')} trend_strength={tech.get('trend_strength')}",
        f"News direction={news.get('direction')} intensity={news_intensity} article_count={news.get('article_count')}",
        f"Fundamentals mcap_change_1d={fund.get('mcap_change_1d')}",
        f"Agreement score={meta.get('agreement_score')}",
        f"atr_pct={_get_tech_value(tech, 'atr_pct')}, vol_regime={_get_tech_value(tech, 'vol_regime')}, dist_from_local_high={tech.get('dist_from_local_high')}, dist_from_local_low={tech.get('dist_from_local_low')}",
        f"VWAP price_above={_get_tech_value(tech, 'price_above_vwap')} vwap_distance_pct={_get_tech_value(tech, 'vwap_distance_pct')}",
        f"Relative strength rs_vs_btc_7d={_get_tech_value(tech, 'rs_vs_btc_7d')} breadth_score={_get_tech_value(tech, 'breadth_score')}",
        f"Squeeze squeeze_on={_get_tech_value(tech, 'squeeze_on')} squeeze_fired={_get_tech_value(tech, 'squeeze_fired')}",
        f"Fundamental trend score={fund.get('fundamental_trend_score')} mkt_cap_change_7d={fund.get('mkt_cap_change_7d')}",
    ]

    fragilities = [
        "Technical signals could flip direction",
        "News tone could reverse or intensity could fade",
        "Fundamental change could shift bias",
        "Alignment could weaken further",
    ]

    summary = (
        f"This {name.replace('_', ' ')} reflects current technical posture ({tech.get('ema_cross')}), "
        f"trend_strength {tech.get('trend_strength')}, news tone ({news.get('direction')}, intensity {news_intensity}), "
        f"and fundamentals (mcap_change_1d={fund.get('mcap_change_1d')}, trend_score={fund.get('fundamental_trend_score')}). "
        f"Agreement score {meta.get('agreement_score')} bounds this scenario."
    )

    return {
        "likelihood": likelihood,
        "intensity": round(float(intensity), 3),
        "summary": summary,
        "key_drivers": drivers,
        "key_fragilities": fragilities,
    }


def _sanitize_summary_rows(rows: Any, max_items: int = 20) -> tuple[List[Dict[str, Any]], int]:
    cleaned: List[Dict[str, Any]] = []
    markers = 0
    for item in (rows or [])[:max_items]:
        if not isinstance(item, dict):
            continue
        src, src_mark = _sanitize_untrusted_text(item.get("source"), max_chars=80)
        title, title_mark = _sanitize_untrusted_text(item.get("title"), max_chars=240)
        summary, sum_mark = _sanitize_untrusted_text(item.get("summary"), max_chars=600)
        published_at, pub_mark = _sanitize_untrusted_text(item.get("published_at"), max_chars=60)
        markers += int(src_mark) + int(title_mark) + int(sum_mark) + int(pub_mark)
        cleaned.append(
            {
                "source": src,
                "title": title,
                "summary": summary,
                "published_at": published_at,
            }
        )
    return cleaned, markers


def _validated_ai_scenario_result(ai_result: Any, scenario_names: List[str]) -> Dict[str, Any]:
    if not isinstance(ai_result, dict):
        return {}
    scenarios_raw = ai_result.get("scenarios")
    out_scenarios: Dict[str, Any] = {}
    if isinstance(scenarios_raw, dict):
        for name in scenario_names:
            row = scenarios_raw.get(name)
            if not isinstance(row, dict):
                continue
            summary, _ = _sanitize_untrusted_text(row.get("summary"), max_chars=1600)
            drivers = row.get("key_drivers")
            fragilities = row.get("key_fragilities")
            if not isinstance(drivers, list):
                drivers = []
            if not isinstance(fragilities, list):
                fragilities = []
            clean_drivers = []
            clean_fragilities = []
            for d in drivers[:8]:
                txt, _ = _sanitize_untrusted_text(d, max_chars=220)
                if txt:
                    clean_drivers.append(txt)
            for f in fragilities[:8]:
                txt, _ = _sanitize_untrusted_text(f, max_chars=220)
                if txt:
                    clean_fragilities.append(txt)
            out_scenarios[name] = {
                "summary": summary,
                "key_drivers": clean_drivers,
                "key_fragilities": clean_fragilities,
            }

    macro = ai_result.get("macro_interpretation")
    if not isinstance(macro, dict):
        macro = {}
    final_summary, _ = _sanitize_untrusted_text(ai_result.get("final_summary"), max_chars=2400)
    return {
        "scenarios": out_scenarios,
        "macro_interpretation": macro,
        "final_summary": final_summary,
    }


def _call_openai_for_scenarios(
    snapshot: Dict[str, Any],
    scenarios: Dict[str, Any],
    analysis_context: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Call OpenAI gpt-5.2 to generate differentiated scenario narratives.
    Numeric fields (likelihood, intensity) are provided and must be echoed unchanged.
    """
    client = OpenAI()

    summary_instruction = (
        "You are writing a research-grade market synthesis for an analytical engine. "
        "Use only the provided feature values, scenario likelihoods, confidence metrics, "
        "and macro/sentiment context. You may use conditional language such as 'may' or 'could' "
        "but you must not issue advice, instructions, or certainty. "
        "Your goal is to explain why the base case is currently the anchor, how tail risks are shaped, "
        "and what developments would plausibly change this assessment. "
        "Do not restate raw values mechanically; integrate them naturally into a coherent narrative. "
        "Emphasize regime characterization, dominant drivers, scenario distribution, "
        "confidence constraints, and change-of-view conditions. "
        "Avoid buy/sell language, imperatives, and rigid templated phrasing. "
        "Write in a professional, neutral research tone. "
        "final_summary must be 7-8 sentences, a single paragraph, and must not repeat "
        "likelihood/intensity numbers in the first sentence. "
        "Begin with a simple, investor-friendly snapshot that includes current price (last_close) "
        "and 1d/7d returns and the volatility regime if available. "
        "Use simpler language and briefly explain any technical terms in plain words "
        "(e.g., ema_cross as trend direction, rsi as momentum). "
        "When signals conflict, explicitly label the divergence (e.g., trend vs momentum) "
        "so the reader understands why the view is mixed. "
        "Include one sentence that explains what likelihood and intensity mean in plain language, "
        "then state scenario_distribution_pct and intensities for best/base/worst using provided values. "
        "If scenario_history.has_history is true, mention which scenario's likelihood moved "
        "most and by how many percentage points (use scenario_history.delta.likelihood_pct), "
        "and note any meaningful intensity change using scenario_history.delta.intensity, "
        "with a short clause tying that shift to current evidence (no new data). "
        "Explain confidence and confidence_trend in plain language (confidence reflects signal clarity; "
        "trend shows whether confidence is improving or deteriorating). "
        "Combine macro event density and the next key event into one concise sentence. If macro_liquidity is provided, mention the liquidity regime (qt_like/qe_like/neutral) with walcl_4w and tightening_score, and note it affects tail risk/confidence not direction. "
        "Use final_summary_inputs.confidence_label explicitly. "
        "Include a sentence that begins 'What would change this view' and names "
        "2-3 parallel, testable triggers from final_summary_inputs.change_of_view_triggers, "
        "with a brief non-directive near-term expectation tied to the anchor scenario. "
        "End with a final sentence stating whether the analysis skew is bullish, neutral, or bearish "
        "using final_summary_inputs.analysis_skew."
    )

    system_msg = (
        "You are an analyst. You must base all reasoning strictly on the provided feature values. "
        "Do not assume trends, do not invent numbers, and do not add external context. "
        "Use conditional language (could/may/if). Do not give predictions or advice. "
        "Likelihood and intensity numbers are fixed; you must not alter them. "
        "You may reference the provided news_summaries and macro_summaries for context. "
        "Do not compute, change, or reinterpret likelihood/intensity; only use provided numbers. "
        "Use fear/greed only as convexity context; do not claim it implies tops or bottoms. "
        "Macro interpretation must reference next_high_event timing, derived_bias, macro_confidence, and event_risk. "
        "Do not infer risk_on/risk_off from tag density alone; use provided directional evidence only. "
        "You may use macro_event_digest to interpret recent actual vs previous and upcoming forecast vs previous. "
        "Do not compute or change any liquidity numbers; only explain macro_liquidity and liquidity_regime as confidence/tail-risk context, not direction. "
        "Summaries must be meaningfully different per scenario: "
        "best_case focuses on improvement/positive alignment; "
        "base_case focuses on continuation/mixed signals; "
        "worst_case focuses on deterioration/downside. "
        "Do not provide trade positioning or long/short/flat recommendations. "
        "Key drivers must explicitly reference provided feature values. "
        "Key fragilities must reference what would invalidate the scenario. "
        "Also return macro_interpretation with fields: summary, scenario_implications "
        "(best_case/base_case/worst_case), and directional_note. "
        "Return final_summary as a single string. "
        "Do not compute or change any numeric values; only explain using provided fields. "
        f"{summary_instruction} "
        "Return strict JSON with fields: scenarios, macro_interpretation, final_summary. "
        "Return JSON only, no markdown or extra text."
    )

    macro_ctx = analysis_context.get("macro_context", {}) or {}
    macro_context_map = {name: macro_ctx for name in scenarios}
    macro_drivers_map = {
        name: macro_ctx.get("macro_drivers", [])
        for name in scenarios
    }

    safe_news_rows, news_markers = _sanitize_summary_rows(snapshot.get("news_summaries", []), max_items=25)
    safe_macro_rows, macro_markers = _sanitize_summary_rows(snapshot.get("macro_summaries", []), max_items=25)
    total_markers = int(news_markers + macro_markers)

    user_payload = {
        "symbol": snapshot.get("symbol"),
        "technical_features": snapshot.get("technical_features", {}),
        "news_features": snapshot.get("news_features", {}),
        "news_summaries_untrusted": safe_news_rows,
        "macro_summaries_untrusted": safe_macro_rows,
        "macro_sentiment": snapshot.get("macro_sentiment", {}),
        "macro_calendar": snapshot.get("macro_calendar", {}),
        "macro_liquidity": snapshot.get("macro_liquidity", {}),
        "sentiment_regime": snapshot.get("sentiment_regime", {}),
        "fundamental_features": snapshot.get("fundamental_features", {}),
        "meta_features": snapshot.get("meta_features", {}),
        "llm_safety": {
            "untrusted_text_delimited": True,
            "injection_markers_detected": int(total_markers),
            "max_summary_chars": 600,
        },
        "macro_context": macro_context_map,
        "macro_drivers": macro_drivers_map,
        "liquidity_regime": analysis_context.get("liquidity_regime"),
        "scenarios": {
            name: {
                "likelihood": data.get("likelihood"),
                "intensity": data.get("intensity"),
            }
            for name, data in scenarios.items()
        },
        "final_summary_inputs": analysis_context,
    }

    print(">>> CALLING OPENAI GPT-5.2")
    try:
        resp = client.chat.completions.create(
            model="gpt-5.2",
            temperature=0.2,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": json.dumps(user_payload)},
            ],
        )
    except TypeError:
        resp = client.chat.completions.create(
            model="gpt-5.2",
            temperature=0.2,
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": json.dumps(user_payload)},
            ],
        )
    content = resp.choices[0].message.content
    try:
        parsed = json.loads(content)
    except json.JSONDecodeError as exc:
        start = content.find("{")
        end = content.rfind("}")
        if start != -1 and end != -1 and end > start:
            parsed = json.loads(content[start : end + 1])
        else:
            raise RuntimeError("OpenAI returned non-JSON output") from exc
    validated = _validated_ai_scenario_result(parsed, list(scenarios.keys()))
    validated["llm_safety"] = {
        "injection_markers_detected": int(total_markers),
        "sanitized_inputs": True,
    }
    return validated

def _format_source_summary(item: Dict[str, Any], max_len: int = 100) -> str:
    source = (item.get("source") or "Unknown").strip()
    published_at = item.get("published_at") or ""
    text = item.get("summary") or item.get("title") or ""
    text = " ".join(str(text).split())
    if published_at:
        line = f"[{source}] {published_at} - {text}".strip()
    else:
        line = f"[{source}] {text}".strip()
    if len(line) > max_len:
        line = line[: max_len - 3].rstrip() + "..."
    return line


def _sources_used(snapshot: Dict[str, Any], limit: int = 10, max_len: int = 100) -> Dict[str, List[str]]:
    news_list = snapshot.get("news_summaries", []) or []
    macro_list = snapshot.get("macro_summaries", []) or []
    return {
        "news_summaries": [_format_source_summary(item, max_len) for item in news_list[:limit]],
        "macro_summaries": [_format_source_summary(item, max_len) for item in macro_list[:limit]],
    }


def _build_macro_calendar(cal_features: Dict[str, Any]) -> tuple[Dict[str, Any], Dict[str, Any]]:
    summary = cal_features.get("calendar_summary", {}) or {}
    freshness = cal_features.get("freshness", {}) or {}
    overlay = cal_features.get("risk_regime_overlay", {}) or {}
    directional = overlay.get("directional_evidence", {}) or {}
    top_recent = cal_features.get("top_recent", []) or []
    top_upcoming = cal_features.get("top_upcoming", []) or []
    counts = cal_features.get("counts", {}) or {}
    theme_pressure = cal_features.get("theme_pressure", {}) or {}

    events_with_surprise = int(directional.get("events_with_surprise") or 0)
    net_bias = float(directional.get("net_bias") or 0.0)
    evidence_weight = float(directional.get("evidence_weight") or 0.0)
    momentum_events = 0
    momentum_weight = 0.0
    momentum_sum = 0.0
    impact_w = {"high": 1.0, "medium": 0.5, "low": 0.25}
    for ev in top_recent:
        actual = ev.get("actual")
        prev = ev.get("previous")
        if actual is None or prev is None:
            continue
        try:
            delta = float(actual) - float(prev)
        except Exception:
            continue
        sign = 1.0 if delta > 0 else -1.0 if delta < 0 else 0.0
        w = impact_w.get(ev.get("impact") or "low", 0.25)
        momentum_events += 1
        momentum_weight += w
        momentum_sum += sign * w

    derived_bias = "unknown"
    if events_with_surprise >= 2 and evidence_weight >= 0.5:
        if net_bias >= 0.15:
            derived_bias = "easing_pressure"
        elif net_bias <= -0.15:
            derived_bias = "tightening_pressure"
        else:
            derived_bias = "neutral"
    elif momentum_events >= 2 and momentum_weight >= 0.5:
        momentum_bias = momentum_sum / max(momentum_weight, 1e-6)
        if momentum_bias >= 0.15:
            derived_bias = "easing_pressure"
        elif momentum_bias <= -0.15:
            derived_bias = "tightening_pressure"
        else:
            derived_bias = "neutral"

    recent_events_used = len(top_recent)
    surprise_eligible = sum(1 for ev in top_recent if ev.get("has_surprise_inputs"))
    ago_vals = [float(ev.get("ago_hours")) for ev in top_recent if ev.get("ago_hours") is not None]
    avg_ago = round(sum(ago_vals) / len(ago_vals), 2) if ago_vals else None
    high_next_24h = 0
    high_next_48h = 0
    for ev in top_upcoming:
        if ev.get("impact") != "high":
            continue
        in_hours = ev.get("in_hours")
        if in_hours is None:
            continue
        if float(in_hours) <= 24:
            high_next_24h += 1
        if float(in_hours) <= 48:
            high_next_48h += 1

    macro_calendar = {
        "next_high_event_in_hours": summary.get("next_high_event_in_hours"),
        "next_high_event_title": summary.get("next_high_event_title"),
        "counts_next_7d": {
            "high": int(summary.get("high_events_next_7d") or 0),
            "liquidity": int(summary.get("liquidity_events_next_7d") or 0),
            "money_supply": int(summary.get("money_supply_events_next_7d") or 0),
        },
        "event_stack": {
            "high_next_24h": high_next_24h,
            "high_next_48h": high_next_48h,
            "high_next_7d": int(summary.get("high_events_next_7d") or 0),
        },
        "directional_evidence": {
            "events_with_surprise": events_with_surprise,
            "net_bias": round(net_bias, 3),
            "evidence_weight": round(evidence_weight, 3),
            "derived_bias": derived_bias,
            "momentum_events": momentum_events,
            "momentum_weight": round(momentum_weight, 3),
        },
        "data_usability": {
            "recent_events_used": recent_events_used,
            "surprise_eligible": surprise_eligible,
            "avg_ago_hours_recent": avg_ago,
        },
        "policy": "macro_influences_confidence_and_tail_risk_not_direction",
    }

    macro_raw = {
        "calendar_summary": summary,
        "directional_evidence": directional,
        "counts": counts,
        "theme_pressure": theme_pressure,
        "top_recent": top_recent,
        "top_upcoming": top_upcoming,
        "freshness": freshness,
    }
    return macro_calendar, macro_raw


def _macro_event_digest(macro_raw: Dict[str, Any], max_recent: int = 6, max_upcoming: int = 8) -> Dict[str, Any]:
    recent = macro_raw.get("top_recent", []) or []
    upcoming = macro_raw.get("top_upcoming", []) or []

    def _pick_recent(ev: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "title": ev.get("title"),
            "impact": ev.get("impact"),
            "category": ev.get("category"),
            "ago_hours": ev.get("ago_hours"),
            "actual": ev.get("actual"),
            "forecast": ev.get("forecast"),
            "previous": ev.get("previous"),
            "unit": ev.get("unit"),
            "macro_tags": ev.get("macro_tags") or [],
        }

    def _pick_upcoming(ev: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "title": ev.get("title"),
            "impact": ev.get("impact"),
            "category": ev.get("category"),
            "in_hours": ev.get("in_hours"),
            "forecast": ev.get("forecast"),
            "previous": ev.get("previous"),
            "unit": ev.get("unit"),
            "macro_tags": ev.get("macro_tags") or [],
        }

    recent_sorted = sorted(recent, key=lambda ev: ev.get("ago_hours") or 1e9)
    upcoming_sorted = sorted(upcoming, key=lambda ev: ev.get("in_hours") or 1e9)

    return {
        "recent_events": [_pick_recent(ev) for ev in recent_sorted[:max_recent]],
        "upcoming_events": [_pick_upcoming(ev) for ev in upcoming_sorted[:max_upcoming]],
    }


def _macro_liquidity_block(liq_features: Dict[str, Any]) -> Tuple[Dict[str, Any], str | None, str | None]:
    if not liq_features:
        return {}, None, None

    # v2 schema (fed:features)
    if "inputs" in liq_features and (
        "regime" in liq_features
        or "strength" in liq_features
        or "liquidity_confidence" in liq_features
    ):
        regime = liq_features.get("regime")
        strength = liq_features.get("strength")
        inputs = liq_features.get("inputs") or {}
        walcl_4w = inputs.get("walcl_change_4w_pct")
        walcl_1w = inputs.get("walcl_change_1w_pct")
        reserves_4w = inputs.get("reserves_change_4w_usd")
        rrp_4w = inputs.get("rrp_change_4w_usd")
        asof_date = liq_features.get("asof_date")
        age_hours = liq_features.get("age_hours")
        staleness_label = None
        confidence_val = liq_features.get("liquidity_confidence")
        agreement_ratio = liq_features.get("agreement_ratio")
    else:
        regime_label = (liq_features.get("liquidity_regime") or {}).get("label")
        strength = (liq_features.get("liquidity_regime") or {}).get("strength")
        walcl = (liq_features.get("balance_sheet") or {}).get("walcl") or {}
        reserves = (liq_features.get("bank_reserves") or {}).get("reserves") or {}
        rrp = (liq_features.get("reverse_repo") or {}).get("rrp") or {}
        freshness = liq_features.get("freshness") or {}

        label_map = {"QT-like": "qt_like", "QE-like": "qe_like", "Neutral": "neutral"}
        regime = label_map.get(regime_label)

        walcl_4w = walcl.get("change_4w_pct")
        walcl_1w = walcl.get("change_1w_pct")
        reserves_4w = reserves.get("change_4w")
        rrp_4w = rrp.get("change_4w")
        asof_date = liq_features.get("asof_date")
        age_hours = freshness.get("avg_age_hours")
        staleness_label = freshness.get("staleness_label")
        confidence_val = liq_features.get("liquidity_confidence")
        agreement_ratio = liq_features.get("agreement_ratio")

    macro_liquidity = {
        "source": "fred",
        "asof_date": asof_date,
        "age_hours": age_hours,
        "regime": regime,
        "tightening_score": strength,
        "confidence": confidence_val,
        "agreement_ratio": agreement_ratio,
        "inputs": {
            "walcl_change_4w_pct": walcl_4w,
            "reserves_change_4w_usd": reserves_4w,
            "rrp_change_4w_usd": rrp_4w,
        },
        "policy": "liquidity_influences_confidence_and_tail_risk_not_direction",
    }

    driver_line = None
    if regime is not None:
        strength_txt = f"{float(strength):.3f}" if isinstance(strength, (int, float)) else "n/a"
        conf_txt = f"{float(confidence_val):.3f}" if isinstance(confidence_val, (int, float)) else "n/a"
        agreement_txt = (
            f"{float(agreement_ratio):.3f}"
            if isinstance(agreement_ratio, (int, float))
            else "n/a"
        )
        walcl_txt = f"{walcl_4w:+.2f}%" if isinstance(walcl_4w, (int, float)) else "n/a"
        reserves_txt = _fmt_large(reserves_4w)
        rrp_txt = _fmt_large(rrp_4w)
        age_txt = f"{float(age_hours):.1f}" if isinstance(age_hours, (int, float)) else "n/a"
        driver_line = (
            f"Fed liquidity: regime={regime} score={strength_txt} conf={conf_txt} agree={agreement_txt} "
            f"(age_h={age_txt})"
        )
        driver_line += f" | Drivers: WALCL 4w {walcl_txt}, reserves 4w {reserves_txt}, RRP 4w {rrp_txt}"

    return macro_liquidity, driver_line, staleness_label




def _macro_context_from_calendar(macro_calendar: Dict[str, Any], macro_raw: Dict[str, Any]) -> Dict[str, Any]:
    next_high = macro_calendar.get("next_high_event_in_hours")
    if next_high is None:
        event_risk = "unknown"
    elif float(next_high) <= 48:
        event_risk = "elevated"
    else:
        event_risk = "normal"

    derived_bias = macro_calendar.get("directional_evidence", {}).get("derived_bias", "unknown")
    dir_evidence = macro_calendar.get("directional_evidence", {}) or {}
    events_with_surprise = float(dir_evidence.get("events_with_surprise") or 0.0)
    evidence_weight = float(dir_evidence.get("evidence_weight") or 0.0)
    recent_events_used = float(macro_calendar.get("data_usability", {}).get("recent_events_used") or 0.0)
    surprise_eligible = float(macro_calendar.get("data_usability", {}).get("surprise_eligible") or 0.0)
    avg_ago = macro_calendar.get("data_usability", {}).get("avg_ago_hours_recent")
    avg_ago_val = float(avg_ago) if avg_ago is not None else 168.0

    evidence_events = events_with_surprise
    evidence_w = evidence_weight
    stale_penalty = 0.0
    if avg_ago_val > 72.0:
        stale_penalty = min(0.15, ((avg_ago_val - 72.0) / 96.0) * 0.15)

    conf = clamp01(
        0.25
        + 0.20 * min(1.0, evidence_events / 4.0)
        + 0.20 * min(1.0, evidence_w / 5.0)
        + 0.20 * min(1.0, surprise_eligible / max(1.0, recent_events_used))
        + 0.15 * (1.0 - min(1.0, avg_ago_val / 168.0))
        - stale_penalty
    )

    recent = macro_raw.get("top_recent", []) or []
    recent_sorted = sorted(recent, key=lambda ev: ev.get("ago_hours") or 1e9)
    drivers: List[str] = []
    seen_titles = set()
    max_total = 5
    target_us = 3
    target_jp = 2

    def _country(ev: Dict[str, Any]) -> Optional[str]:
        c = ev.get("country")
        return c if c in {"US", "JP"} else None

    def _fmt_recent_surprise(ev: Dict[str, Any]) -> str:
        title = ev.get("title") or ""
        surprise_sign = ev.get("surprise_sign") or "none"
        effect = ev.get("effect_hint") or {}
        bias = effect.get("bias") or "unknown"
        conf_str = effect.get("confidence")
        conf_txt = f"{conf_str:.2f}" if isinstance(conf_str, (int, float)) else "null"
        ago = ev.get("ago_hours")
        ago_txt = f"{float(ago):.2f}" if ago is not None else "null"
        actual = ev.get("actual")
        forecast = ev.get("forecast")
        actual_txt = f"{float(actual):.4f}" if isinstance(actual, (int, float)) else "null"
        forecast_txt = f"{float(forecast):.4f}" if isinstance(forecast, (int, float)) else "null"
        category = ev.get("category") or "other"
        level_txt = _level_context_suffix(ev.get("level_context"))
        return (
            f"{title} (type={category}): actual={actual_txt} forecast={forecast_txt} "
            f"surprise_sign={surprise_sign} bias={bias} conf={conf_txt} (ago_h={ago_txt})"
            f"{level_txt}"
        )

    def _fmt_recent_basic(ev: Dict[str, Any]) -> str:
        title = ev.get("title") or ""
        ago = ev.get("ago_hours")
        ago_txt = f"{float(ago):.2f}" if ago is not None else "null"
        actual = ev.get("actual")
        previous = ev.get("previous")
        actual_txt = f"{float(actual):.4f}" if isinstance(actual, (int, float)) else "null"
        previous_txt = f"{float(previous):.4f}" if isinstance(previous, (int, float)) else "null"
        effect = ev.get("effect_hint") or {}
        bias = effect.get("bias") or "neutral"
        conf_str = effect.get("confidence")
        conf_txt = f"{conf_str:.2f}" if isinstance(conf_str, (int, float)) else "null"
        category = ev.get("category") or "other"
        level_txt = _level_context_suffix(ev.get("level_context"))
        return (
            f"{title} (type={category}): actual={actual_txt} previous={previous_txt} "
            f"bias={bias} conf={conf_txt} (ago_h={ago_txt})"
            f"{level_txt}"
        )

    def _fmt_upcoming(ev: Dict[str, Any]) -> str:
        title = ev.get("title") or ""
        in_hours = float(ev.get("in_hours"))
        forecast = ev.get("forecast")
        previous = ev.get("previous")
        forecast_txt = f"{float(forecast):.4f}" if isinstance(forecast, (int, float)) else "null"
        previous_txt = f"{float(previous):.4f}" if isinstance(previous, (int, float)) else "null"
        category = ev.get("category") or "other"
        expected = "unknown"
        if isinstance(forecast, (int, float)) and isinstance(previous, (int, float)):
            delta = float(forecast) - float(previous)
            if abs(delta) < 1e-9:
                expected = "flat"
            elif delta > 0:
                expected = "up"
            else:
                expected = "down"
        level_txt = _level_context_suffix(ev.get("level_context"))
        return (
            f"Upcoming high-impact: {title} (type={category}) (in {in_hours:.2f}h) "
            f"forecast={forecast_txt} previous={previous_txt} expected={expected}"
            f"{level_txt}"
        )

    candidates: List[tuple] = []
    # Priority tiers: lower number = higher priority
    recent_high_surprise = [
        ev for ev in recent_sorted if ev.get("impact") == "high" and ev.get("has_surprise_inputs")
    ]
    for ev in recent_high_surprise:
        if _country(ev):
            candidates.append((1, float(ev.get("ago_hours") or 1e9), _country(ev), _fmt_recent_surprise, ev))

    recent_high = [ev for ev in recent_sorted if ev.get("impact") == "high"]
    for ev in recent_high:
        if _country(ev):
            candidates.append((2, float(ev.get("ago_hours") or 1e9), _country(ev), _fmt_recent_basic, ev))

    recent_with_surprise = [
        ev
        for ev in recent_sorted
        if ev.get("has_surprise_inputs") and ev.get("impact") in {"high", "medium"}
    ]
    for ev in recent_with_surprise:
        if _country(ev):
            candidates.append((3, float(ev.get("ago_hours") or 1e9), _country(ev), _fmt_recent_surprise, ev))

    for ev in recent_sorted:
        if _country(ev):
            candidates.append((4, float(ev.get("ago_hours") or 1e9), _country(ev), _fmt_recent_basic, ev))

    upcoming = macro_raw.get("top_upcoming", []) or []
    upcoming_high = [
        ev for ev in upcoming if ev.get("impact") == "high" and ev.get("in_hours") is not None
    ]
    upcoming_sorted = sorted(upcoming_high, key=lambda ev: float(ev.get("in_hours")))
    for ev in upcoming_sorted:
        if _country(ev):
            candidates.append((5, float(ev.get("in_hours") or 1e9), _country(ev), _fmt_upcoming, ev))

    counts = {"US": 0, "JP": 0}
    for _, _, country, fmt_fn, ev in sorted(candidates, key=lambda x: (x[0], x[1])):
        if len(drivers) >= max_total:
            break
        title = ev.get("title") or ""
        if title in seen_titles:
            continue
        if country == "US" and counts["US"] >= target_us:
            continue
        if country == "JP" and counts["JP"] >= target_jp:
            continue
        seen_titles.add(title)
        drivers.append(fmt_fn(ev))
        counts[country] += 1

    if len(drivers) < max_total:
        for _, _, country, fmt_fn, ev in sorted(candidates, key=lambda x: (x[0], x[1])):
            if len(drivers) >= max_total:
                break
            title = ev.get("title") or ""
            if title in seen_titles:
                continue
            seen_titles.add(title)
            drivers.append(fmt_fn(ev))
            if country in counts:
                counts[country] += 1

    return {
        "event_risk": event_risk,
        "macro_bias": derived_bias,
        "macro_confidence": round(float(conf), 3),
        "macro_drivers": drivers,
    }


def _level_context_suffix(level: Any) -> str:
    if not isinstance(level, dict):
        return ""
    label = level.get("label")
    pct = level.get("percentile")
    zscore = level.get("zscore")
    basis = level.get("basis")
    parts = []
    if label:
        parts.append(f"level={label}")
    if isinstance(pct, (int, float)):
        parts.append(f"p={float(pct):.2f}")
    if isinstance(zscore, (int, float)):
        parts.append(f"z={float(zscore):.2f}")
    if basis:
        parts.append(f"basis={basis}")
    if not parts:
        return ""
    return " " + " ".join(parts)


def _fmt_num(val: Any, nd: int = 3) -> str:
    if val is None:
        return "n/a"
    try:
        return f"{float(val):.{nd}f}"
    except Exception:
        return str(val)




def _fmt_large(val: Any) -> str:
    if val is None:
        return "n/a"
    try:
        num = float(val)
    except Exception:
        return str(val)
    abs_val = abs(num)
    if abs_val >= 1e12:
        return f"{num / 1e12:.2f}T"
    if abs_val >= 1e9:
        return f"{num / 1e9:.2f}B"
    if abs_val >= 1e6:
        return f"{num / 1e6:.2f}M"
    return f"{num:.2f}"

def _fmt_bool(val: Any) -> str:
    if isinstance(val, bool):
        return str(val).lower()
    return "n/a"


def _confidence_label(overall_conf: float) -> str:
    if overall_conf >= 0.75:
        return "higher conviction"
    if overall_conf >= 0.55:
        return "moderate conviction"
    return "low conviction / fragile"


def _agreement_label(score: float | None) -> str:
    if score is None:
        return "unknown"
    if score >= 0.75:
        return "high"
    if score >= 0.55:
        return "moderate"
    return "low"


def _analysis_skew_label(scenarios: Dict[str, Any]) -> Tuple[str, str]:
    best = scenarios.get("best_case", {}).get("likelihood")
    worst = scenarios.get("worst_case", {}).get("likelihood")
    try:
        best_val = float(best)
        worst_val = float(worst)
    except Exception:
        return ("neutral", "neutral")
    diff = best_val - worst_val
    abs_diff = abs(diff)
    if abs_diff < 0.05:
        return ("neutral", "neutral")
    if abs_diff < 0.10:
        strength = "slightly"
    elif abs_diff < 0.20:
        strength = "moderately"
    else:
        strength = "clearly"
    label = "bullish" if diff > 0 else "bearish"
    return (label, strength)


def _change_view_triggers(
    tech: Dict[str, Any], news: Dict[str, Any], macro_calendar: Dict[str, Any]
) -> List[str]:
    ema_cross = tech.get("ema_cross") or "unknown"
    vol_regime = tech.get("vol_regime") or "unknown"
    news_dir = news.get("direction") or "unknown"
    weighted_bullish = news.get("weighted_bullish")
    weighted_bearish = news.get("weighted_bearish")
    next_title = macro_calendar.get("next_high_event_title")

    ema_trigger = "ema_cross resolving clearly (bullish or bearish)"
    if ema_cross == "bearish":
        ema_trigger = "ema_cross flipping to neutral/bullish"
    elif ema_cross == "bullish":
        ema_trigger = "ema_cross slipping to neutral/bearish"

    news_trigger = "news direction and intensity shifting materially"
    if weighted_bearish is not None and weighted_bullish is not None:
        if news_dir == "bearish":
            news_trigger = "news skew reversing (weighted_bullish overtakes weighted_bearish)"
        elif news_dir == "bullish":
            news_trigger = "news skew deteriorating (weighted_bearish overtakes weighted_bullish)"

    if next_title:
        macro_trigger = f"{next_title} passes without surprise as event risk fades"
    else:
        macro_trigger = f"vol_regime shifts away from {vol_regime}"

    return [ema_trigger, news_trigger, macro_trigger]


def build_final_summary(snapshot: Dict[str, Any], analysis_json: Dict[str, Any]) -> str:
    tech = snapshot.get("technical_features", {}) or {}
    news = snapshot.get("news_features", {}) or {}
    macro_calendar = analysis_json.get("macro_calendar") or snapshot.get("macro_calendar") or {}
    sentiment = analysis_json.get("sentiment_regime") or snapshot.get("sentiment_regime") or {}
    confidence = analysis_json.get("confidence", {}) or {}
    scenarios = analysis_json.get("scenarios", {}) or {}
    scenario_history = analysis_json.get("scenario_history", {}) or {}

    ema_cross = tech.get("ema_cross") or "unknown"
    last_close = tech.get("last_close")
    price_above_20sma = tech.get("price_above_20sma")
    ret_1d = tech.get("ret_1d")
    ret_3d = tech.get("ret_3d")
    ret_7d = tech.get("ret_7d")
    trend_strength = tech.get("trend_strength")
    vol_regime = tech.get("vol_regime") or "unknown"
    atr_pct = tech.get("atr_pct")

    news_dir = news.get("direction") or "unknown"
    news_intensity = news.get("intensity")
    weighted_bullish = news.get("weighted_bullish")
    weighted_bearish = news.get("weighted_bearish")
    article_count = news.get("article_count")

    agreement_score = snapshot.get("meta_features", {}).get("agreement_score")
    try:
        agreement_score = float(agreement_score)
    except Exception:
        agreement_score = None

    overall_conf = float(confidence.get("overall") or 0.0)
    trend_label = (confidence.get("trend") or {}).get("label") or "unknown"
    freshness_score = analysis_json.get("freshness_score")

    conf_label = _confidence_label(overall_conf)
    skew_label, skew_strength = _analysis_skew_label(scenarios)

    anchor = "base_case"
    max_like = -1.0
    for name, data in scenarios.items():
        try:
            like = float(data.get("likelihood"))
        except Exception:
            like = -1.0
        if like > max_like:
            max_like = like
            anchor = name
    anchor = anchor or "base_case"

    sentences: List[str] = []

    headline_bits = []
    if last_close is not None:
        headline_bits.append(f"price={_fmt_num(last_close, 2)}")
    if ret_1d is not None:
        headline_bits.append(f"1d_return={_fmt_num(ret_1d)}")
    if ret_7d is not None:
        headline_bits.append(f"7d_return={_fmt_num(ret_7d)}")
    headline_bits.append(f"vol_regime={vol_regime}")
    sentences.append(
        f"Current conditions are summarized by {', '.join(headline_bits)}."
    )

    regime_parts = [f"ema_cross={ema_cross}"]
    if price_above_20sma is not None:
        regime_parts.append(f"price_above_20sma={_fmt_bool(price_above_20sma)}")
    trend_bits = [f"trend_strength={_fmt_num(trend_strength)}"]
    if ret_3d is not None:
        trend_bits.append(f"ret_3d={_fmt_num(ret_3d)}")
    trend_bits.append(f"vol_regime={vol_regime}")
    if atr_pct is not None:
        trend_bits.append(f"atr_pct={_fmt_num(atr_pct)}")
    sentences.append(
        "The setup is mixed: short-term trend and momentum signals diverge rather than align "
        f"({', '.join(regime_parts)}; {', '.join(trend_bits)})."
    )

    if article_count is not None and int(article_count) == 0:
        sentences.append("News coverage is thin (article_count=0), so sentiment input is weak.")
    else:
        news_bits = [
            f"direction={news_dir}",
            f"intensity={_fmt_num(news_intensity)}",
        ]
        if weighted_bearish is not None and weighted_bullish is not None:
            news_bits.append(f"weighted_bearish={_fmt_num(weighted_bearish)} vs weighted_bullish={_fmt_num(weighted_bullish)}")
        if article_count is not None:
            news_bits.append(f"article_count={int(article_count)}")
        sentences.append(
            f"News tone is {news_dir} with {', '.join(news_bits)}, which can shape sentiment pressure."
        )

    next_high = macro_calendar.get("next_high_event_in_hours")
    next_title = macro_calendar.get("next_high_event_title")
    derived_bias = (macro_calendar.get("directional_evidence") or {}).get("derived_bias") or "unknown"
    macro_ctx = analysis_json.get("macro_context") or {}
    macro_conf = macro_ctx.get("macro_confidence")
    if next_high is None:
        macro_sentence = "Macro calendar is sparse, so event risk is unknown."
    else:
        event_risk = macro_ctx.get("event_risk") or ("elevated" if float(next_high) <= 48 else "normal")
        macro_sentence = (
            f"Macro calendar shows next high-impact in {float(next_high):.2f}h "
            f"({next_title}); derived_bias={derived_bias} with macro_confidence={_fmt_num(macro_conf)} "
            f"and event_risk={event_risk}."
        )
    fng_val = sentiment.get("fear_greed_value")
    fng_label = sentiment.get("fear_greed_label")
    convexity = sentiment.get("convexity_boost")
    if fng_val is not None:
        macro_sentence += (
            f" Sentiment is {fng_label} (value={int(fng_val)}) with convexity_boost={_fmt_num(convexity)}, "
            "which implies fatter tails without a directional call."
        )
    else:
        macro_sentence += " Sentiment regime is unavailable, so convexity guidance is limited."
    sentences.append(macro_sentence)

    sentences.append(
        f"Overall confidence is {conf_label} (overall={overall_conf:.3f}) with agreement_score={_fmt_num(agreement_score)} "
        f"and trend={trend_label}, meaning confidence reflects signal clarity while the trend shows recent confidence drift; "
        f"{anchor} remains the anchor scenario."
    )

    scenario_bits = []
    for name in ["best_case", "base_case", "worst_case"]:
        sc = scenarios.get(name, {}) or {}
        like = sc.get("likelihood")
        inten = sc.get("intensity")
        like_pct = f"{float(like) * 100:.1f}%" if isinstance(like, (int, float)) else "n/a"
        inten_txt = f"{float(inten):.3f}" if isinstance(inten, (int, float)) else "n/a"
        label = name.replace("_case", "")
        scenario_bits.append(f"{label} {like_pct} (intensity {inten_txt})")
    history_clause = ""
    if scenario_history.get("has_history"):
        deltas = scenario_history.get("delta", {}).get("likelihood_pct", {}) or {}
        best_key = None
        best_delta = None
        for key in ["best_case", "base_case", "worst_case"]:
            val = deltas.get(key)
            if isinstance(val, (int, float)) and (best_delta is None or val > best_delta):
                best_delta = val
                best_key = key
        if best_key is not None and best_delta is not None:
            inten_delta = (
                scenario_history.get("delta", {}).get("intensity", {}).get(best_key)
            )
            inten_txt = ""
            if isinstance(inten_delta, (int, float)) and abs(inten_delta) > 0:
                inten_txt = f"; intensity change {inten_delta:+.3f}"
            history_clause = (
                f" Since last run, {best_key.replace('_case', '')} likelihood moved {best_delta:+.1f}pp"
                f"{inten_txt}."
            )
    sentences.append(
        "Likelihoods describe relative scenario odds and intensity describes potential magnitude; "
        f"current mix is {', '.join(scenario_bits)}.{history_clause}"
    )

    triggers = _change_view_triggers(tech, news, macro_calendar)

    sentences.append(
        f"What would change this view is {triggers[0]}, {triggers[1]}, or {triggers[2]}; "
        f"otherwise the {anchor.replace('_case', '')} case remains the most plausible near-term path."
    )

    if skew_label == "neutral":
        sentences.append("Overall analysis skew is neutral.")
    else:
        sentences.append(f"Overall analysis skew is {skew_strength} {skew_label}.")

    summary = " ".join(sentences)
    return summary


def _build_shadow_intelligence_for_scenario(
    snapshot: Dict[str, Any],
    scenarios: Dict[str, Any],
    agreement_detail: Dict[str, Any],
    confidence: Dict[str, Any],
    freshness_score: float,
    event_risk: str,
    con,
    interval: str = "1h",
) -> Dict[str, Any]:
    base = {
        "stage": "shadow",
        "enabled": False,
        "interval": str(interval or "1h"),
        "prediction_enabled": False,
        "pattern_enabled": False,
        "influence_weights": {
            "prediction": 0.0,
            "pattern": 0.0,
        },
        "prediction": {},
        "patterns": {},
        "intel_score_delta": 0.0,
        "intel_blockers": [],
        "intel_used_for_entry": False,
        "status": "unavailable",
        "error": None,
    }
    if con is None:
        return base
    try:
        from backend.services.paper_engine import (
            apply_intelligence,
            build_pattern_snapshot,
            build_prediction_snapshot,
        )

        symbol = str(snapshot.get("symbol") or "")
        tech = snapshot.get("technical_features", {}) or {}
        top_two = sorted(
            [
                float((scenarios.get("best_case") or {}).get("likelihood") or 0.0),
                float((scenarios.get("base_case") or {}).get("likelihood") or 0.0),
                float((scenarios.get("worst_case") or {}).get("likelihood") or 0.0),
            ],
            reverse=True,
        )
        margin = (top_two[0] - top_two[1]) if len(top_two) >= 2 else 0.0
        agreement_score = float(agreement_detail.get("agreement_score") or 0.0)
        effective_conf = float(confidence.get("overall") or 0.0)
        dominant_bias = str(agreement_detail.get("horizon_dominant_bias") or "neutral").lower()
        side = "LONG" if dominant_bias == "bullish" else "SHORT" if dominant_bias == "bearish" else "NO_TRADE"
        vol_regime = str(_get_tech_value(tech, "vol_regime") or "")
        regime_label = "high_vol_chop" if vol_regime == "high" else "range"
        technical_context = {
            "price_above_vwap": _get_tech_value(tech, "price_above_vwap"),
            "vwap_distance_pct": _parse_float(_get_tech_value(tech, "vwap_distance_pct")),
            "rs_vs_btc_7d": _parse_float(_get_tech_value(tech, "rs_vs_btc_7d")),
            "breadth_score": _parse_float(_get_tech_value(tech, "breadth_score")),
            "squeeze_on": bool(_get_tech_value(tech, "squeeze_on") is True),
        }
        decision_like = {
            "symbol": symbol,
            "strategy_inputs": {
                "regime_classification": {
                    "market_regime": regime_label,
                },
                "scenario_snapshot": {
                    "margin_vs_second": round(float(margin), 6),
                },
                "signal_quality": {
                    "effective_confidence": round(float(effective_conf), 6),
                    "agreement_score": round(float(agreement_score), 6),
                    "freshness_score": round(float(freshness_score), 6),
                    "event_risk": str(event_risk or "normal"),
                },
                "technical_context": technical_context,
            },
        }
        candidate = {
            "symbol": symbol,
            "side": side,
            "effective_confidence": float(effective_conf),
            "agreement_score": float(agreement_score),
            "freshness_score": float(freshness_score),
        }
        pred = build_prediction_snapshot(
            conn=con,
            symbol=symbol,
            decision_json=decision_like,
            candidate=candidate,
            interval=str(interval or "1h"),
        )
        pat = build_pattern_snapshot(
            conn=con,
            symbol=symbol,
            decision_json=decision_like,
            interval=str(interval or "1h"),
        )
        intel = apply_intelligence(
            candidate=candidate,
            prediction=pred,
            patterns=pat,
            stage_cfg={
                "stage": "shadow",
                "prediction_weight": 0.0,
                "pattern_weight": 0.0,
            },
        )
        return {
            **base,
            "enabled": True,
            "prediction_enabled": True,
            "pattern_enabled": True,
            "prediction": intel.get("prediction") or pred,
            "patterns": intel.get("patterns") or pat,
            "intel_score_delta": float(intel.get("intel_score_delta") or 0.0),
            "intel_blockers": list(intel.get("intel_blockers") or []),
            "intel_used_for_entry": bool(intel.get("intel_used_for_entry")),
            "status": "ok",
            "error": None,
        }
    except Exception as exc:
        failed = dict(base)
        failed["status"] = "error"
        failed["error"] = str(exc)
        return failed


def interpret_snapshot(
    snapshot: Dict[str, Any],
    con=None,
    use_gpt: Optional[bool] = None,
) -> Dict[str, Any]:
    """
    Analysis-only interpretation from raw features.
    No summaries or narratives from Phase 3 are consumed.
    """
    if "error" in snapshot:
        return snapshot

    likelihoods = compute_likelihoods(snapshot)
    sentiment_regime = _compute_sentiment_regime(snapshot)
    likelihoods = _apply_convexity(likelihoods, sentiment_regime)
    macro_liquidity_raw = snapshot.get("macro_liquidity", {}) or {}
    macro_liquidity_ctx, _, _ = _macro_liquidity_block(macro_liquidity_raw)
    macro_calendar = snapshot.get("macro_calendar", {}) or {}
    likelihoods, liq_boost, liq_distance = _apply_liquidity_convexity(
        likelihoods, macro_liquidity_ctx, macro_calendar
    )
    snapshot["sentiment_regime"] = sentiment_regime
    intensities = _compute_intensities(snapshot)
    scenarios = {
        "best_case": _scenario_from_features(
            "best_case",
            snapshot,
            likelihoods.get("best_case", 0.33),
            intensities.get("best_case", 0.0),
        ),
        "base_case": _scenario_from_features(
            "base_case",
            snapshot,
            likelihoods.get("base_case", 0.34),
            intensities.get("base_case", 0.0),
        ),
        "worst_case": _scenario_from_features(
            "worst_case",
            snapshot,
            likelihoods.get("worst_case", 0.33),
            intensities.get("worst_case", 0.0),
        ),
    }
    macro_calendar = snapshot.get("macro_calendar", {}) or {}
    macro_raw = snapshot.get("macro_calendar_raw", {}) or {}
    macro_context = _macro_context_from_calendar(macro_calendar, macro_raw)

    macro_liquidity_raw = snapshot.get("macro_liquidity", {}) or {}
    macro_liquidity_ctx, liq_driver, liq_staleness = _macro_liquidity_block(macro_liquidity_raw)
    if macro_liquidity_ctx:
        try:
            regime = macro_liquidity_ctx.get("regime")
            score = macro_liquidity_ctx.get("tightening_score")
            conf = macro_liquidity_ctx.get("confidence")
            agree = macro_liquidity_ctx.get("agreement_ratio")
            age_h = macro_liquidity_ctx.get("age_hours")
            inputs = macro_liquidity_ctx.get("inputs") or {}
            walcl_4w = inputs.get("walcl_change_4w_pct")
            reserves_4w = inputs.get("reserves_change_4w_usd")
            rrp_4w = inputs.get("rrp_change_4w_usd")
            missing = any(v is None for v in [walcl_4w, reserves_4w, rrp_4w])
            line1 = (
                f"Fed liquidity: regime={regime} score={score:.3f} conf={conf:.3f} agree={agree:.3f} "
                f"(age_h={float(age_h):.1f})"
                if isinstance(score, (int, float))
                and isinstance(conf, (int, float))
                and isinstance(agree, (int, float))
                and isinstance(age_h, (int, float))
                else f"Fed liquidity: regime={regime} score={score} conf={conf} agree={agree} (age_h={age_h})"
            )
            walcl_txt = f"{walcl_4w:+.2f}%" if isinstance(walcl_4w, (int, float)) else "n/a"
            line2 = (
                f"Drivers: WALCL 4w {walcl_txt}, reserves 4w {_fmt_large(reserves_4w)}, "
                f"RRP 4w {_fmt_large(rrp_4w)}"
            )
            line3 = "Policy: affects confidence/tail risk, not direction"
            line4 = "Data: incomplete inputs" if missing else "Data: complete inputs"
            # Keep macro_liquidity detail under macro_liquidity block only; avoid duplication in macro_context.
        except Exception:
            pass
    if liq_driver:
        drivers = macro_context.get("macro_drivers")
        if isinstance(drivers, list):
            drivers.append(liq_driver)
        else:
            macro_context["macro_drivers"] = [liq_driver]

    liquidity_regime = {}
    if macro_liquidity_ctx:
        try:
            liquidity_regime = {
                "tightening_score": macro_liquidity_ctx.get("tightening_score"),
                "regime": macro_liquidity_ctx.get("regime"),
                "confidence": macro_liquidity_ctx.get("confidence"),
                "distance_from_neutral": None if liq_distance is None else round(float(liq_distance), 3),
                "convexity_boost": round(float(liq_boost), 3),
                "policy": "convexity_only_no_direction",
            }
        except Exception:
            liquidity_regime = {}

    freshness = snapshot.get("meta_features", {}).get("freshness", {}) or {}
    freshness_score, freshness_detail = _freshness_score(freshness)
    required_modalities = evaluate_required_modality_status(freshness)
    meta_features = snapshot.get("meta_features") or {}
    meta_features["required_modalities"] = required_modalities
    snapshot["meta_features"] = meta_features
    for data in scenarios.values():
        data["freshness"] = {
            "tech_avg_age_hours": freshness.get("tech_avg_age_hours"),
            "news_avg_age_hours": freshness.get("news_avg_age_hours"),
            "fundamentals_avg_age_hours": freshness.get("fundamentals_avg_age_hours"),
            "news_count_used": freshness.get("news_count_used", 0),
        }

    confidence = _confidence_block(snapshot, scenarios, freshness_score)

    tech_now = snapshot.get("technical_features", {}) or {}
    horizon_alignment_info = _horizon_alignment_from_multi_horizon(tech_now)
    horizon_alignment_score = None
    if horizon_alignment_info is not None:
        try:
            horizon_alignment_score = float(horizon_alignment_info.get("horizon_alignment_score"))
        except Exception:
            horizon_alignment_score = None
    agreement_component = float(confidence.get("components", {}).get("agreement") or 0.5)
    likelihood_vals = []
    for name in ("best_case", "base_case", "worst_case"):
        try:
            likelihood_vals.append(float((scenarios.get(name) or {}).get("likelihood") or 0.0))
        except Exception:
            likelihood_vals.append(0.0)
    likelihood_vals = sorted(likelihood_vals, reverse=True)
    edge = (likelihood_vals[0] - likelihood_vals[1]) if len(likelihood_vals) >= 2 else 0.0
    edge_score = clamp01(edge / 0.20)

    if os.environ.get("TRIDENT_DEBUG_CONF") == "1":
        penalties_total = 0.0
        for group in confidence.get("penalties", {}).values():
            for p in group:
                penalties_total += float(p.get("impact") or 0.0)
        h_txt = (
            f"{float(horizon_alignment_score):.3f}"
            if isinstance(horizon_alignment_score, (int, float))
            else "none"
        )
        print(
            "[scenario_confidence] "
            f"base_likelihoods={{best:{scenarios['best_case'].get('likelihood')},"
            f"base:{scenarios['base_case'].get('likelihood')},"
            f"worst:{scenarios['worst_case'].get('likelihood')}}} "
            f"intensities={{best:{scenarios['best_case'].get('intensity')},"
            f"base:{scenarios['base_case'].get('intensity')},"
            f"worst:{scenarios['worst_case'].get('intensity')}}} "
            f"edge={edge:.3f} edge_score={edge_score:.3f} penalties_total={penalties_total:.3f} "
            f"agreement_score={agreement_component:.3f} horizon_alignment_score={h_txt}",
            file=sys.stderr,
        )

    overall_conf = float(confidence.get("overall") or 0.0)
    scenario_conf_map = _scenario_confidences_from_outputs(
        overall=overall_conf,
        scenarios=scenarios,
        agreement_score=agreement_component,
        horizon_alignment_score=horizon_alignment_score,
    )
    for name, data in scenarios.items():
        scenario_conf = float(scenario_conf_map.get(name, 0.0))
        if scenario_conf <= 0.0 and os.environ.get("TRIDENT_DEBUG_CONF") == "1":
            print(f"[scenario_confidence] {name} clamped to 0.0", file=sys.stderr)
        _debug_confidence_case(
            name=name,
            overall_post_penalties=overall_conf,
            scenario_likelihood=float(data.get("likelihood") or 0.0),
            scenario_intensity=float(data.get("intensity") or 0.0),
            edge_score=edge_score,
            agreement_score=agreement_component,
            horizon_alignment_score=horizon_alignment_score,
            scenario_confidence=scenario_conf,
        )
        if os.environ.get("TRIDENT_DEBUG_CONF") == "1":
            if abs(scenario_conf - overall_conf) > 0.35:
                print(
                    f"[scenario_confidence] {name} deviates from overall by >0.35 "
                    f"(overall={overall_conf:.3f} scenario={scenario_conf:.3f})",
                    file=sys.stderr,
                )
        data["scenario_confidence"] = round(float(scenario_conf), 3)

    penalties_sum = 0.0
    for group in confidence.get("penalties", {}).values():
        for p in group:
            impact = float(p.get("impact") or 0.0)
            if impact < 0:
                penalties_sum += -impact

    if con is not None:
        now_utc = datetime.now(timezone.utc)
        provenance = snapshot.get("provenance") or {}
        run_id = str(provenance.get("run_id") or snapshot.get("run_id") or "")
        snapshot_id = str(provenance.get("snapshot_id") or snapshot.get("snapshot_id") or "")
        input_hash = str(provenance.get("input_hash") or snapshot.get("input_hash") or "")
        _insert_analysis_run(
            con,
            now_utc,
            snapshot.get("symbol") or "",
            run_id,
            snapshot_id,
            input_hash,
            overall_conf,
            scenarios["best_case"]["scenario_confidence"],
            scenarios["base_case"]["scenario_confidence"],
            scenarios["worst_case"]["scenario_confidence"],
            scenarios["best_case"].get("likelihood"),
            scenarios["base_case"].get("likelihood"),
            scenarios["worst_case"].get("likelihood"),
            scenarios["best_case"].get("intensity"),
            scenarios["base_case"].get("intensity"),
            scenarios["worst_case"].get("intensity"),
        )
        try:
            con.execute(
                """
                INSERT OR REPLACE INTO run_provenance
                    (run_id, snapshot_id, input_hash, command, symbol, asof_utc, status, notes, created_at_utc)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    run_id,
                    snapshot_id,
                    input_hash,
                    "scenario:analysis",
                    str(snapshot.get("symbol") or ""),
                    now_utc,
                    "ok",
                    None,
                    now_utc,
                ],
            )
        except Exception:
            pass
        agreement_score = float(confidence.get("components", {}).get("agreement") or 0.0)
        trend, prev_scenarios = _compute_confidence_trend(
            con,
            snapshot.get("symbol") or "",
            overall_conf,
            penalties_sum,
            agreement_score,
        )
        history_rows = _fetch_recent_scenario_history(con, snapshot.get("symbol") or "", 2)
    else:
        trend = {
            "lookback_runs_fast": 0,
            "previous_avg_fast": None,
            "delta_1": None,
            "delta_fast": None,
            "sigma_fast": None,
            "z_delta_fast": None,
            "slope_per_hour_fast": None,
            "slope_24h_fast": None,
            "threshold_used": None,
            "label": "insufficient_history",
            "explanation": "Insufficient history to compute confidence trend.",
        }
        prev_scenarios = None
        history_rows = []

    confidence["trend"] = trend
    for name, data in scenarios.items():
        if prev_scenarios is None:
            data["confidence_trend_bias"] = "neutral"
            continue
        prev_val = float(prev_scenarios.get(name, 0.0))
        delta = float(data["scenario_confidence"]) - prev_val
        if delta > 0.03:
            data["confidence_trend_bias"] = "strengthening"
        elif delta < -0.03:
            data["confidence_trend_bias"] = "weakening"
        else:
            data["confidence_trend_bias"] = "neutral"

    scenario_confidence_summary = {
        "best_case": {
            "confidence": scenarios["best_case"]["scenario_confidence"],
            "trend_bias": scenarios["best_case"]["confidence_trend_bias"],
        },
        "base_case": {
            "confidence": scenarios["base_case"]["scenario_confidence"],
            "trend_bias": scenarios["base_case"]["confidence_trend_bias"],
        },
        "worst_case": {
            "confidence": scenarios["worst_case"]["scenario_confidence"],
            "trend_bias": scenarios["worst_case"]["confidence_trend_bias"],
        },
    }

    for data in scenarios.values():
        data.pop("scenario_confidence", None)
        data.pop("confidence_trend_bias", None)

    scenario_history = {
        "has_history": False,
        "previous": {},
        "delta": {},
    }
    if len(history_rows) >= 2:
        _, prev_lb, prev_lbase, prev_lw, prev_ib, prev_ibase, prev_iw = history_rows[1]
        prev_like = {
            "best_case": prev_lb,
            "base_case": prev_lbase,
            "worst_case": prev_lw,
        }
        prev_int = {
            "best_case": prev_ib,
            "base_case": prev_ibase,
            "worst_case": prev_iw,
        }
        delta_like = {}
        delta_int = {}
        delta_like_pct = {}
        for key in ["best_case", "base_case", "worst_case"]:
            cur_like = scenarios[key].get("likelihood")
            cur_int = scenarios[key].get("intensity")
            try:
                delta_like[key] = round(float(cur_like) - float(prev_like.get(key)), 3)
            except Exception:
                delta_like[key] = None
            try:
                delta_int[key] = round(float(cur_int) - float(prev_int.get(key)), 3)
            except Exception:
                delta_int[key] = None
            try:
                delta_like_pct[key] = round((float(cur_like) - float(prev_like.get(key))) * 100.0, 1)
            except Exception:
                delta_like_pct[key] = None
        scenario_history = {
            "has_history": True,
            "previous": {
                "likelihood": prev_like,
                "intensity": prev_int,
            },
            "delta": {
                "likelihood": delta_like,
                "intensity": delta_int,
                "likelihood_pct": delta_like_pct,
            },
        }

    confidence = {
        "overall": confidence.get("overall"),
        "components": confidence.get("components"),
        "scenario_confidences": scenario_confidence_summary,
        "trend": confidence.get("trend"),
        "penalties": confidence.get("penalties"),
        "attribution": confidence.get("attribution"),
    }

    # Build analysis context for GPT final_summary
    tech = snapshot.get("technical_features", {}) or {}
    multi_horizon = tech.get("multi_horizon")
    if not isinstance(multi_horizon, dict):
        multi_horizon = None
    news = snapshot.get("news_features", {}) or {}
    agreement_score = confidence.get("components", {}).get("agreement")
    try:
        agreement_score = float(agreement_score)
    except Exception:
        agreement_score = snapshot.get("meta_features", {}).get("agreement_score")
        try:
            agreement_score = float(agreement_score)
        except Exception:
            agreement_score = None
    overall_conf = float(confidence.get("overall") or 0.0)
    conf_label = _confidence_label(overall_conf)
    anchor = "base_case"
    max_like = -1.0
    for name, data in scenarios.items():
        try:
            like = float(data.get("likelihood"))
        except Exception:
            like = -1.0
        if like > max_like:
            max_like = like
            anchor = name
    triggers = _change_view_triggers(tech, news, macro_calendar)
    macro_event_digest = _macro_event_digest(snapshot.get("macro_calendar_raw", {}) or {})
    skew_label, skew_strength = _analysis_skew_label(scenarios)
    analysis_context = {
        "confidence_overall": overall_conf,
        "confidence_label": conf_label,
        "confidence_trend": (confidence.get("trend") or {}).get("label"),
        "agreement_score": agreement_score,
        "freshness_score": freshness_score,
        "freshness_detail": freshness_detail,
        "headline_metrics": {
            "last_close": tech.get("last_close"),
            "ret_1d": tech.get("ret_1d"),
            "ret_7d": tech.get("ret_7d"),
            "vol_regime": tech.get("vol_regime"),
        },
        "macro_calendar": macro_calendar,
        "macro_context": macro_context,
        "macro_liquidity": macro_liquidity_raw,
        "liquidity_regime": liquidity_regime,
        "macro_event_digest": macro_event_digest,
        "sentiment_regime": sentiment_regime,
        "anchor_scenario": anchor,
        "analysis_skew": {
            "label": skew_label,
            "strength": skew_strength,
        },
        "scenario_distribution": {
            name: {
                "likelihood": data.get("likelihood"),
                "intensity": data.get("intensity"),
            }
            for name, data in scenarios.items()
        },
        "scenario_distribution_pct": {
            name: (
                round(float(data.get("likelihood")) * 100.0, 1)
                if isinstance(data.get("likelihood"), (int, float))
                else None
            )
            for name, data in scenarios.items()
        },
        "scenario_history": scenario_history,
        "change_of_view_triggers": triggers,
        "required_modalities": required_modalities,
    }

    # Call OpenAI for differentiated narratives and final_summary
    ai_result: Dict[str, Any] = {}
    llm_status = {"used_gpt": False, "error": None}
    use_gpt_effective = _resolve_scenario_use_gpt(use_gpt)
    if use_gpt_effective:
        try:
            ai_result = _call_openai_for_scenarios(snapshot, scenarios, analysis_context)
            llm_status["used_gpt"] = True
        except Exception as exc:
            llm_status["error"] = str(exc)
            print(f"[scenario:analysis][warn] gpt_interpretation_failed error={exc}", file=sys.stderr)
    ai_scenarios = ai_result.get("scenarios", {}) if isinstance(ai_result, dict) else {}
    macro_interpretation = ai_result.get("macro_interpretation", {}) if isinstance(ai_result, dict) else {}
    if not isinstance(macro_interpretation, dict):
        macro_interpretation = {}
    if not macro_interpretation:
        macro_interpretation = {
            "summary": "",
            "scenario_implications": {
                "best_case": "",
                "base_case": "",
                "worst_case": "",
            },
            "directional_note": "",
        }

    for name, data in scenarios.items():
        ai_data = ai_scenarios.get(name, {})
        # Preserve deterministic numbers; only override textual fields if provided
        summary = ai_data.get("summary")
        drivers = ai_data.get("key_drivers")
        fragilities = ai_data.get("key_fragilities")
        if summary:
            data["summary"] = summary
        if isinstance(drivers, list):
            data["key_drivers"] = drivers
        if isinstance(fragilities, list):
            data["key_fragilities"] = fragilities

    # Build final_summary
    horizon_alignment = _horizon_alignment_from_multi_horizon(tech)
    agreement_detail = {
        "agreement_score": agreement_score,
        "agreement_label": _agreement_label(agreement_score),
    }
    if horizon_alignment is not None:
        agreement_detail.update(
            {
                "horizon_alignment_score": round(
                    float(horizon_alignment.get("horizon_alignment_score") or 0.0), 3
                ),
                "horizon_alignment_label": horizon_alignment.get("horizon_alignment_label"),
                "horizon_dominant_bias": horizon_alignment.get("horizon_dominant_bias"),
                "multi_horizon": {
                    "windows": horizon_alignment.get("horizon_windows"),
                    "horizon_signals": horizon_alignment.get("horizon_signals_used"),
                },
            }
        )

    shadow_intelligence = _build_shadow_intelligence_for_scenario(
        snapshot=snapshot,
        scenarios=scenarios,
        agreement_detail=agreement_detail,
        confidence=confidence,
        freshness_score=float(freshness_score),
        event_risk=str((macro_context or {}).get("event_risk") or "normal"),
        con=con,
        interval="1h",
    )

    analysis_json = {
        "symbol": snapshot.get("symbol"),
        "run_id": snapshot.get("run_id"),
        "snapshot_id": snapshot.get("snapshot_id"),
        "input_hash": snapshot.get("input_hash"),
        "scenarios": scenarios,
        "multi_horizon": multi_horizon,
        "macro_context": macro_context,
        "macro_calendar": macro_calendar,
        "macro_liquidity": macro_liquidity_raw,
        "liquidity_regime": liquidity_regime,
        "sources_used": _sources_used(snapshot, limit=10, max_len=100),
        "freshness_score": freshness_score,
        "freshness_detail": freshness_detail,
        "required_modality_status": required_modalities,
        "agreement_detail": agreement_detail,
        "confidence": confidence,
        "sentiment_regime": sentiment_regime,
        "scenario_history": scenario_history,
        "shadow_intelligence": shadow_intelligence,
        "llm_status": llm_status,
        "llm_safety": ai_result.get("llm_safety") if isinstance(ai_result, dict) else {},
    }
    final_summary = ""
    if isinstance(ai_result, dict):
        fs = ai_result.get("final_summary")
        if isinstance(fs, str):
            final_summary = fs.strip()
    if not final_summary:
        final_summary = build_final_summary(snapshot, analysis_json)

    return {
        "symbol": snapshot.get("symbol"),
        "run_id": snapshot.get("run_id"),
        "snapshot_id": snapshot.get("snapshot_id"),
        "input_hash": snapshot.get("input_hash"),
        "scenarios": scenarios,
        "multi_horizon": multi_horizon,
        "macro_context": macro_context,
        "macro_calendar": macro_calendar,
        "macro_liquidity": macro_liquidity_raw,
        "liquidity_regime": liquidity_regime,
        "macro_interpretation": macro_interpretation,
        "sources_used": _sources_used(snapshot, limit=10, max_len=100),
        "freshness_score": freshness_score,
        "freshness_detail": freshness_detail,
        "required_modality_status": required_modalities,
        "agreement_detail": agreement_detail,
        "confidence": confidence,
        "sentiment_regime": sentiment_regime,
        "shadow_intelligence": shadow_intelligence,
        "final_summary": final_summary,
        "llm_status": llm_status,
        "llm_safety": ai_result.get("llm_safety") if isinstance(ai_result, dict) else {},
    }

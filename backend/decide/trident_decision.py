from __future__ import annotations

import json
import sys
import uuid
import hashlib
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from backend.decide.gates import build_gate_result
from backend.decide.utils import (
    compute_hold_window_hours,
    compute_entry_plan,
    compute_exit_conditions,
    compute_position_size,
    compute_stop,
    compute_target,
)
from backend.features.phase4_analysis import (
    _apply_convexity,
    _apply_liquidity_convexity,
    _compute_intensities,
    _compute_sentiment_regime,
    _confidence_block,
    _continuous_agreement_score,
    _freshness_score,
    evaluate_required_modality_status,
    _horizon_alignment_from_multi_horizon,
    _macro_context_from_calendar,
    _macro_liquidity_block,
    build_phase4_snapshot,
    clamp01,
    compute_likelihoods,
)
from backend.features.tech_features import compute_tech_features
from backend.services.policy_ai import (
    render_decision_narrative_gpt52,
    render_trade_explanation_gpt52,
)


ALLOWED_ACTIONS = {"LONG", "SHORT", "HOLD", "NO_TRADE"}
ALLOWED_REGIMES = {"bull_trend", "bear_trend", "range", "high_vol_chop"}
DECISION_LOGIC_VERSION = "5C.1"
ALLOWED_POSITION_STATES = {"FLAT", "LONG", "SHORT"}


def _fmt_utc(ts: datetime) -> str:
    return ts.astimezone(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _parse_utc(ts: Any) -> Optional[datetime]:
    if ts is None:
        return None
    try:
        s = str(ts).strip()
        if not s:
            return None
        if s.endswith("Z"):
            s = s[:-1] + "+00:00"
        dt = datetime.fromisoformat(s)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    except Exception:
        return None


def _safe_float(v: Any) -> Optional[float]:
    try:
        if v is None:
            return None
        return float(v)
    except Exception:
        return None


def _safe_int(v: Any, default: int = 0) -> int:
    try:
        if v is None:
            return int(default)
        return int(v)
    except Exception:
        return int(default)


def _stable_json_hash(payload: Dict[str, Any]) -> str:
    try:
        raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    except Exception:
        raw = str(payload)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _new_run_id(prefix: str) -> str:
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    return f"{prefix}_{ts}_{uuid.uuid4().hex[:8]}"


def _normalize_position_state(v: Any) -> str:
    s = str(v or "").strip().upper()
    if s in {"LONG", "SHORT", "FLAT"}:
        return s
    return "FLAT"


def _load_recent_decision_payloads(con, symbol: str, limit: int = 8) -> List[Dict[str, Any]]:
    try:
        rows = con.execute(
            """
            SELECT payload_json
            FROM trident_decisions
            WHERE symbol = ?
            ORDER BY asof_utc DESC, created_at_utc DESC
            LIMIT ?
            """,
            [symbol, int(limit)],
        ).fetchall()
    except Exception:
        return []
    payloads: List[Dict[str, Any]] = []
    for r in rows:
        if not r:
            continue
        raw = r[0]
        if not raw:
            continue
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, dict):
                payloads.append(parsed)
        except Exception:
            continue
    return payloads


def _infer_position_state(con, symbol: str) -> str:
    payloads = _load_recent_decision_payloads(con, symbol, limit=3)
    for p in payloads:
        state = _normalize_position_state(p.get("position_state"))
        if state in {"LONG", "SHORT"}:
            return state
        decision = (p.get("decision") or {}).get("action")
        decision_s = str(decision or "").upper()
        if decision_s in {"LONG", "SHORT"}:
            return decision_s
        if decision_s == "NO_TRADE":
            return "FLAT"
    return "FLAT"


def _flip_persistence_runs(con, symbol: str, current_state: str, opposite_action: str) -> int:
    payloads = _load_recent_decision_payloads(con, symbol, limit=8)
    runs = 0
    for p in payloads:
        p_state = _normalize_position_state(p.get("position_state"))
        p_hyp = str(((p.get("paper_trade_preview") or {}).get("hypothetical_action") or "")).upper()
        if p_state == current_state and p_hyp == opposite_action:
            runs += 1
        else:
            break
    return runs


def _flip_strength(
    margin: float,
    effective_conf: float,
    agreement_score: Optional[float],
    horizon_alignment: Optional[float],
    regime_label: str,
    event_risk: str,
) -> Tuple[float, str]:
    score = 0.35
    score += 0.25 * min(1.0, max(0.0, float(margin) / 0.20))
    score += 0.20 * clamp01(float(effective_conf))
    score += 0.10 * clamp01(float(agreement_score if agreement_score is not None else 0.5))
    score += 0.10 * clamp01(float(horizon_alignment if horizon_alignment is not None else 0.5))
    if regime_label == "high_vol_chop":
        score -= 0.10
    if str(event_risk).lower() == "elevated":
        score -= 0.10
    score = round(clamp01(score), 3)
    if score >= 0.70:
        tier = "strong"
    elif score >= 0.50:
        tier = "medium"
    else:
        tier = "mild"
    return score, tier


def _extract_anchor_from_payload(payload: Dict[str, Any], side: str) -> Optional[Dict[str, Any]]:
    decision = payload.get("decision") or {}
    action = str(decision.get("action") or "").upper()
    if action != side:
        return None
    trade_plan = payload.get("trade_plan")
    if not isinstance(trade_plan, dict):
        return None
    risk_mgmt = trade_plan.get("risk_management") or {}
    target_plan = trade_plan.get("target_plan") or {}
    entry_price = _safe_float(risk_mgmt.get("entry_price_estimate"))
    stop_price = _safe_float(risk_mgmt.get("stop_price_estimate"))
    if entry_price is None:
        entry_price = _safe_float(payload.get("price"))
    target_price = _safe_float(target_plan.get("target_price_estimate"))
    if entry_price is None or stop_price is None:
        return None
    return {
        "asof": payload.get("asof"),
        "entry_price": float(entry_price),
        "stop_price": float(stop_price),
        "target_price": float(target_price) if target_price is not None else None,
    }


def _load_position_anchor(con, symbol: str, side: str) -> Optional[Dict[str, Any]]:
    if con is None:
        return None
    for payload in _load_recent_decision_payloads(con, symbol, limit=24):
        anchor = _extract_anchor_from_payload(payload, side=side)
        if anchor is not None:
            return anchor
    return None


def _anchor_age_hours(anchor_asof: Any) -> Optional[float]:
    dt = _parse_utc(anchor_asof)
    if dt is None:
        return None
    age = (datetime.now(timezone.utc) - dt).total_seconds() / 3600.0
    return float(age) if age >= 0 else 0.0


def _build_shadow_intelligence_block(
    con,
    symbol: str,
    decision_payload: Dict[str, Any],
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

        strategy_inputs = decision_payload.get("strategy_inputs") or {}
        signal_quality = strategy_inputs.get("signal_quality") or {}
        hypothetical_action = str(
            ((decision_payload.get("paper_trade_preview") or {}).get("hypothetical_action") or "NO_TRADE")
        ).upper()
        side = hypothetical_action if hypothetical_action in {"LONG", "SHORT"} else "NO_TRADE"
        candidate = {
            "symbol": symbol,
            "side": side,
            "effective_confidence": float(_safe_float(signal_quality.get("effective_confidence")) or 0.0),
            "agreement_score": float(_safe_float(signal_quality.get("agreement_score")) or 0.0),
            "freshness_score": float(_safe_float(signal_quality.get("freshness_score")) or 0.0),
        }
        pred = build_prediction_snapshot(
            conn=con,
            symbol=symbol,
            decision_json=decision_payload,
            candidate=candidate,
            interval=str(interval or "1h"),
        )
        pat = build_pattern_snapshot(
            conn=con,
            symbol=symbol,
            decision_json=decision_payload,
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


def _consecutive_hold_breach_runs(
    con,
    symbol: str,
    side: str,
    metric: str,
    threshold: float,
) -> int:
    if con is None:
        return 0
    runs = 0
    for payload in _load_recent_decision_payloads(con, symbol, limit=24):
        p_state = _normalize_position_state(payload.get("position_state"))
        if p_state != side:
            break
        action = str(((payload.get("decision") or {}).get("action") or "")).upper()
        if action != "HOLD":
            break
        sq = ((payload.get("strategy_inputs") or {}).get("signal_quality") or {})
        value = _safe_float(sq.get(metric))
        if value is None:
            break
        if value < float(threshold):
            runs += 1
        else:
            break
    return runs


def _build_hold_escalation_state(
    con,
    symbol: str,
    side: str,
    gate_active: bool,
    agreement_score: Optional[float],
    effective_confidence: Optional[float],
) -> Dict[str, Any]:
    agreement_breach_runs = 0
    confidence_breach_runs = 0
    if agreement_score is not None and agreement_score < 0.50:
        agreement_breach_runs = 1 + _consecutive_hold_breach_runs(
            con=con,
            symbol=symbol,
            side=side,
            metric="agreement_score",
            threshold=0.50,
        )
    if effective_confidence is not None and effective_confidence < 0.35:
        confidence_breach_runs = 1 + _consecutive_hold_breach_runs(
            con=con,
            symbol=symbol,
            side=side,
            metric="effective_confidence",
            threshold=0.35,
        )

    agreement_trigger = agreement_breach_runs >= 2
    confidence_trigger = confidence_breach_runs >= 2
    triggered = bool(gate_active or agreement_trigger or confidence_trigger)
    return {
        "agreement_breach_runs": int(agreement_breach_runs),
        "confidence_breach_runs": int(confidence_breach_runs),
        "gate_active_trigger": bool(gate_active),
        "agreement_trigger": bool(agreement_trigger),
        "confidence_trigger": bool(confidence_trigger),
        "triggered": bool(triggered),
        "recommended_step": "REDUCE_OR_EXIT" if triggered else "MAINTAIN",
    }


def _get_tech_value(tech: Dict[str, Any], key: str) -> Any:
    if key in tech and tech.get(key) is not None:
        return tech.get(key)
    for block_key in ("ema", "trend", "momentum", "volatility", "volume", "relative_strength", "breadth", "market_structure"):
        block = tech.get(block_key)
        if isinstance(block, dict) and key in block and block.get(key) is not None:
            return block.get(key)
    return None


def _regime_for_trade_plan(regime_label: str) -> str:
    if regime_label in {"bull_trend", "bear_trend"}:
        return "trend"
    if regime_label == "high_vol_chop":
        return "high_vol_chop"
    return "range"


def _expected_rr(confidence: float) -> float:
    if confidence < 0.5:
        return 2.0
    if confidence < 0.65:
        return 2.5
    return 3.0


def _resolve_entry_price(snapshot: Dict[str, Any], tech: Dict[str, Any]) -> float:
    for key in ("last_close", "ema20", "sma20", "ema50"):
        px = _safe_float(_get_tech_value(tech, key))
        if px is not None and px > 0:
            return float(px)
    fund = snapshot.get("fundamental_features", {}) or {}
    for key in ("price", "last_price", "close", "price_usd"):
        v = _safe_float(fund.get(key))
        if v is not None and v > 0:
            return float(v)
    return 1.0


def _resolve_atr_value(entry_price: float, tech: Dict[str, Any]) -> float:
    atr_pct = _safe_float(_get_tech_value(tech, "atr_pct"))
    if atr_pct is not None and atr_pct > 0 and entry_price > 0:
        return float(entry_price) * float(atr_pct)
    rv_24h = _safe_float(_get_tech_value(tech, "rv_24h"))
    if rv_24h is not None and rv_24h > 0 and entry_price > 0:
        return float(entry_price) * float(rv_24h)
    if entry_price > 0:
        return max(float(entry_price) * 0.001, 1e-6)
    return 1e-6


def _top_two_likelihoods(scenarios: Dict[str, Dict[str, Any]]) -> Tuple[Optional[str], float, float]:
    vals: List[Tuple[str, float]] = []
    for name in ("best_case", "base_case", "worst_case"):
        like = _safe_float((scenarios.get(name) or {}).get("likelihood"))
        vals.append((name, like if like is not None else 0.0))
    vals.sort(key=lambda x: x[1], reverse=True)
    if not vals:
        return None, 0.0, 0.0
    top_name, top_val = vals[0]
    second_val = vals[1][1] if len(vals) > 1 else 0.0
    return top_name, float(top_val), float(second_val)


def _infer_horizon_bias(multi_horizon: Optional[Dict[str, Any]]) -> str:
    if not isinstance(multi_horizon, dict):
        return "mixed"
    hs = multi_horizon.get("horizon_signals")
    if not isinstance(hs, dict):
        return "mixed"
    h6 = hs.get("6h") or {}
    h2d = hs.get("2d") or {}
    t6 = (h6.get("trend_bias") or "").lower()
    t2 = (h2d.get("trend_bias") or "").lower()
    c6 = _safe_float(h6.get("composite_strength")) or 0.0
    c2 = _safe_float(h2d.get("composite_strength")) or 0.0
    if c6 >= 0.65 and c2 >= 0.65 and t6 in {"bullish", "bearish"} and t6 == t2:
        return t6
    return "mixed"


def _conviction_label(confidence: float) -> str:
    if confidence >= 0.70:
        return "high"
    if confidence >= 0.45:
        return "medium"
    return "low"


def _validity_hours(conviction_label: str, regime_label: str, event_risk: str, hold_multiplier: float = 1.0) -> int:
    return compute_hold_window_hours(
        conviction_label=conviction_label,
        regime_label=regime_label,
        event_risk=event_risk,
        hold_multiplier=hold_multiplier,
        min_hours=24,
        max_hours=96,
        hard_cap_hours=168,
    )


def classify_regime(
    tech: Dict[str, Any],
    agreement_detail: Dict[str, Any],
    multi_horizon: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    vol_regime = (str(_get_tech_value(tech, "vol_regime") or "")).lower()
    atr_pct = _safe_float(_get_tech_value(tech, "atr_pct"))
    rv_24h = _safe_float(_get_tech_value(tech, "rv_24h"))
    adx14 = _safe_float(_get_tech_value(tech, "adx14"))
    trend_strength = _safe_float(_get_tech_value(tech, "trend_strength"))
    ema_cross = (str(_get_tech_value(tech, "ema_cross") or "")).lower()
    plus_di14 = _safe_float(_get_tech_value(tech, "plus_di14"))
    minus_di14 = _safe_float(_get_tech_value(tech, "minus_di14"))
    bb_bandwidth = _safe_float(_get_tech_value(tech, "bb_bandwidth_20_2"))
    price_above_vwap = _get_tech_value(tech, "price_above_vwap")
    vwap_distance_pct = _safe_float(_get_tech_value(tech, "vwap_distance_pct"))
    breadth_score = _safe_float(_get_tech_value(tech, "breadth_score"))
    rs_vs_btc_7d = _safe_float(_get_tech_value(tech, "rs_vs_btc_7d"))
    squeeze_on = bool(_get_tech_value(tech, "squeeze_on") is True)
    squeeze_fired = bool(_get_tech_value(tech, "squeeze_fired") is True)

    is_high_vol = (
        vol_regime in {"high", "extreme"}
        or (rv_24h is not None and rv_24h >= 0.012)
        or (atr_pct is not None and atr_pct >= 0.018)
    )
    is_strong_trend = (
        adx14 is not None
        and adx14 >= 25.0
        and trend_strength is not None
        and trend_strength >= 0.04
    )

    di_bias = "flat"
    if plus_di14 is not None and minus_di14 is not None:
        if plus_di14 - minus_di14 >= 5.0:
            di_bias = "bull"
        elif minus_di14 - plus_di14 >= 5.0:
            di_bias = "bear"

    horizon_bias = agreement_detail.get("horizon_dominant_bias")
    if horizon_bias not in {"bullish", "bearish", "neutral"}:
        horizon_bias = _infer_horizon_bias(multi_horizon)

    breadth_bull_ok = breadth_score is None or breadth_score >= 0.45
    breadth_bear_ok = breadth_score is None or breadth_score <= 0.55
    rs_bull_ok = rs_vs_btc_7d is None or rs_vs_btc_7d >= -0.01
    rs_bear_ok = rs_vs_btc_7d is None or rs_vs_btc_7d <= 0.01

    if is_high_vol and not is_strong_trend:
        label = "high_vol_chop"
    elif is_strong_trend and ema_cross == "bullish" and di_bias == "bull" and breadth_bull_ok and rs_bull_ok:
        label = "bull_trend"
    elif is_strong_trend and ema_cross == "bearish" and di_bias == "bear" and breadth_bear_ok and rs_bear_ok:
        label = "bear_trend"
    else:
        label = "range"

    conf = 0.50
    if is_high_vol and label == "high_vol_chop":
        conf += 0.15
    if is_strong_trend and label in {"bull_trend", "bear_trend"}:
        conf += 0.15
    if (label == "bull_trend" and di_bias == "bull") or (label == "bear_trend" and di_bias == "bear"):
        conf += 0.10
    horizon_alignment_score = _safe_float(agreement_detail.get("horizon_alignment_score"))
    if horizon_alignment_score is not None and horizon_alignment_score >= 0.65:
        conf += 0.10
    agreement_score = _safe_float(agreement_detail.get("agreement_score"))
    if agreement_score is not None and agreement_score < 0.55:
        conf -= 0.15
    if breadth_score is not None:
        if breadth_score >= 0.60:
            conf += 0.05
        elif breadth_score < 0.40:
            conf -= 0.07
    if rs_vs_btc_7d is not None:
        if rs_vs_btc_7d >= 0.02:
            conf += 0.05
        elif rs_vs_btc_7d <= -0.02:
            conf -= 0.07
    if price_above_vwap is True and ema_cross == "bullish":
        conf += 0.03
    elif price_above_vwap is False and ema_cross == "bearish":
        conf += 0.03
    if squeeze_on and vwap_distance_pct is not None and abs(vwap_distance_pct) < 0.01:
        conf -= 0.05
    if squeeze_fired:
        conf += 0.03
    conf = round(clamp01(conf), 3)

    drivers: List[str] = [
        f"vol_regime={vol_regime} atr_pct={atr_pct} rv_24h={rv_24h} => is_high_vol={is_high_vol}",
        (
            f"adx14={adx14} trend_strength={trend_strength} "
            f"ema_cross={ema_cross} di_bias={di_bias} => is_strong_trend={is_strong_trend}"
        ),
        f"horizon_bias={horizon_bias} horizon_alignment_score={horizon_alignment_score}",
        (
            "vwap/rs/breadth "
            f"price_above_vwap={price_above_vwap} vwap_distance_pct={vwap_distance_pct} "
            f"rs_vs_btc_7d={rs_vs_btc_7d} breadth_score={breadth_score}"
        ),
        f"squeeze_on={squeeze_on} squeeze_fired={squeeze_fired}",
    ]
    if bb_bandwidth is not None:
        drivers.append(f"bb_bandwidth_20_2={bb_bandwidth}")

    return {
        "label": label,
        "confidence": conf,
        "drivers": drivers[:5],
        "is_high_vol": is_high_vol,
        "is_strong_trend": is_strong_trend,
        "di_bias": di_bias,
        "horizon_bias": horizon_bias,
    }


def _hypothetical_action(dominant_bias: str, longer_bias: Optional[str], regime_label: str) -> str:
    _ = regime_label
    if dominant_bias == "bullish":
        return "LONG"
    if dominant_bias == "bearish":
        return "SHORT"
    return "NO_TRADE"


def _is_14d_conflict(candidate_action: str, longer_bias: Optional[str]) -> bool:
    lb = str(longer_bias or "").lower()
    if candidate_action == "LONG" and lb == "bearish":
        return True
    if candidate_action == "SHORT" and lb == "bullish":
        return True
    return False


def _select_hypothetical_action(
    dominant_bias: str,
    longer_bias: Optional[str],
    regime_label: str,
    best_likelihood: float,
    worst_likelihood: float,
    agreement_score: Optional[float],
    margin: Optional[float],
    thresholds: Dict[str, Any],
) -> str:
    candidate = _hypothetical_action(dominant_bias, longer_bias, regime_label)
    agreement = _safe_float(agreement_score)
    marg = _safe_float(margin)
    min_margin = float(thresholds.get("min_margin", 0.06))
    min_agree = float(thresholds.get("min_agreement", 0.52))
    edge_ok = bool(agreement is not None and marg is not None and agreement >= min_agree and marg >= min_margin)
    soft_conflict_ok = bool(
        agreement is not None
        and marg is not None
        and agreement >= (min_agree + 0.03)
        and marg >= (min_margin + 0.02)
    )

    conflict_14d = _is_14d_conflict(candidate, longer_bias)
    hard_veto = bool(conflict_14d and regime_label in {"bull_trend", "bear_trend"})
    soft_conflict = bool(conflict_14d and regime_label in {"range", "high_vol_chop"})

    if hard_veto:
        return "NO_TRADE"
    if candidate == "NO_TRADE":
        if edge_ok:
            if float(best_likelihood) > float(worst_likelihood):
                return "LONG"
            if float(worst_likelihood) > float(best_likelihood):
                return "SHORT"
        return "NO_TRADE"
    if soft_conflict and not soft_conflict_ok:
        return "NO_TRADE"
    return candidate


def _strong_move_structure_pass(
    effective_confidence: Optional[float],
    agreement_score: Optional[float],
    margin: Optional[float],
) -> bool:
    eff = _safe_float(effective_confidence)
    agree = _safe_float(agreement_score)
    marg = _safe_float(margin)
    return bool(
        eff is not None
        and agree is not None
        and marg is not None
        and eff >= 0.60
        and agree >= 0.65
        and marg >= 0.08
    )


def _weighted_gate_override_pass(
    deadband: Dict[str, Any],
    effective_confidence: Optional[float],
    agreement_score: Optional[float],
    margin: Optional[float],
) -> bool:
    if not bool(deadband.get("active")):
        return True
    if str(deadband.get("activation_basis") or "") != "penalty_threshold":
        return False
    if deadband.get("hard_blockers"):
        return False
    return _strong_move_structure_pass(effective_confidence, agreement_score, margin)


def _target_valid_for_side(side: str, entry_price: float, target_price: Optional[float]) -> bool:
    if target_price is None:
        return False
    if side == "LONG":
        return bool(target_price > entry_price)
    if side == "SHORT":
        return bool(target_price < entry_price)
    return False


def _build_hold_trade_plan(
    symbol: str,
    side: str,
    con,
    plan_regime: str,
    entry_price_now: float,
    atr_value: float,
    confidence_for_plan: float,
    confidence_overall: float,
    agreement_for_plan: float,
    margin: float,
    vol_regime: str,
    event_risk: str,
    fear_greed_value: int,
    validity_hours: int,
) -> Dict[str, Any]:
    anchor = _load_position_anchor(con, symbol=symbol, side=side)
    position_sizing = compute_position_size(
        effective_confidence=float(confidence_for_plan),
        raw_confidence=float(confidence_overall),
        agreement_score=float(agreement_for_plan),
        margin_vs_second=float(margin),
        regime=plan_regime,
        volatility_regime=vol_regime,
        event_risk=event_risk,
        fear_greed_value=fear_greed_value,
    )
    stop_now = compute_stop(
        action=side,
        entry_price=float(entry_price_now),
        atr_value=float(atr_value),
        regime=plan_regime,
    )
    exit_conditions = compute_exit_conditions(plan_regime)
    exit_conditions["time_stop_hours"] = min(168, max(24, int(validity_hours or 24)))
    source = "entry_anchored" if anchor is not None else "derived_fallback"
    anchor_age = _anchor_age_hours((anchor or {}).get("asof"))
    anchor_stale = bool(anchor_age is not None and anchor_age > 72.0)
    if source == "entry_anchored" and anchor_stale:
        source = "derived_fallback"
    anchor_entry = _safe_float((anchor or {}).get("entry_price"))
    anchor_stop = _safe_float((anchor or {}).get("stop_price"))
    anchor_target = _safe_float((anchor or {}).get("target_price"))

    computed_stop_candidate = float(stop_now["stop_price_estimate"])
    atr_mode = "applied"
    stop_source = "recomputed"
    if source == "entry_anchored" and anchor_entry is not None and anchor_stop is not None:
        managed_entry = float(anchor_entry)
        if side == "LONG":
            managed_stop = max(float(anchor_stop), float(stop_now["stop_price_estimate"]))
            stop_not_widened = bool(managed_stop >= float(anchor_stop))
        else:
            managed_stop = min(float(anchor_stop), float(stop_now["stop_price_estimate"]))
            stop_not_widened = bool(managed_stop <= float(anchor_stop))
        if abs(managed_stop - float(anchor_stop)) < 1e-9:
            stop_source = "entry_anchored"
            atr_mode = "not_applied"
        else:
            stop_source = "recomputed"
            atr_mode = "applied"
        if _target_valid_for_side(side, managed_entry, anchor_target):
            target_price = float(anchor_target)
            risk_distance = abs(float(managed_entry) - float(managed_stop))
            rr_mult = None if risk_distance <= 1e-12 else abs(float(target_price) - float(managed_entry)) / risk_distance
            target_plan: Dict[str, Any] = {
                "risk_reward_multiple": round(float(rr_mult), 3) if rr_mult is not None else None,
                "target_price_estimate": round(float(target_price), 2),
                "validations": {
                    "target_valid_for_side": True,
                    "target_recomputed": False,
                },
            }
        else:
            target_plan = compute_target(
                action=side,
                entry_price=float(managed_entry),
                stop_price=float(managed_stop),
                confidence=float(confidence_for_plan),
            )
            target_plan["validations"] = {
                "target_valid_for_side": _target_valid_for_side(
                    side, float(managed_entry), _safe_float(target_plan.get("target_price_estimate"))
                ),
                "target_recomputed": True,
            }
    else:
        managed_entry = float(entry_price_now)
        managed_stop = float(stop_now["stop_price_estimate"])
        stop_not_widened = True
        stop_source = "recomputed"
        atr_mode = "applied"
        target_plan = compute_target(
            action=side,
            entry_price=float(managed_entry),
            stop_price=float(managed_stop),
            confidence=float(confidence_for_plan),
        )
        target_plan["validations"] = {
            "target_valid_for_side": _target_valid_for_side(
                side, float(managed_entry), _safe_float(target_plan.get("target_price_estimate"))
            ),
            "target_recomputed": True,
        }

    stop_distance = abs(float(managed_entry) - float(managed_stop))
    max_risk_pct = 0.015 if (plan_regime == "high_vol_chop" and str(event_risk).lower() == "elevated") else 0.03
    risk_management = {
        "stop_multiple_atr": float(stop_now["stop_multiple_atr"]) if atr_mode == "applied" else None,
        "stop_price_estimate": round(float(managed_stop), 2),
        "atr_used": float(stop_now["atr_used"]) if atr_mode == "applied" else None,
        "stop_source": stop_source,
        "atr_mode": atr_mode,
        "computed_stop_candidate": round(float(computed_stop_candidate), 2),
        "anchor_age_hours": round(float(anchor_age), 2) if anchor_age is not None else None,
        "anchor_stale": bool(anchor_stale),
        "entry_price_estimate": round(float(managed_entry), 2),
        "stop_distance": round(float(stop_distance), 6),
        "validations": {
            "stop_distance_positive": bool(stop_distance > 0.0),
            "risk_dollars_within_cap": bool(position_sizing["risk_dollars"] <= 100000 * max_risk_pct + 1e-9),
            "stop_not_widened": bool(stop_not_widened),
            "position_side_matches_state": bool(side in {"LONG", "SHORT"}),
        },
    }
    exit_conditions["validations"] = {
        "time_stop_positive": bool(int(exit_conditions.get("time_stop_hours") or 0) > 0),
    }
    exit_conditions["escalation_rules"] = [
        "gate_active_immediate",
        "agreement_below_0.50_for_2_runs",
        "effective_confidence_below_0.35_for_2_runs",
    ]
    exit_conditions["escalation_state"] = _build_hold_escalation_state(
        con=con,
        symbol=symbol,
        side=side,
        gate_active=False,
        agreement_score=agreement_for_plan,
        effective_confidence=confidence_for_plan,
    )

    return {
        "mode": "position_management",
        "source": source,
        "position_side": side,
        "anchor": {
            "asof": (anchor or {}).get("asof"),
            "entry_price": round(float(anchor_entry), 2) if anchor_entry is not None else None,
            "stop_price": round(float(anchor_stop), 2) if anchor_stop is not None else None,
            "target_price": round(float(anchor_target), 2) if anchor_target is not None else None,
        },
        "position_sizing": position_sizing,
        "risk_management": risk_management,
        "target_plan": target_plan,
        "exit_conditions": exit_conditions,
    }


def compute_deadband(
    confidence_overall: Optional[float],
    agreement_score: Optional[float],
    horizon_alignment_score: Optional[float],
    margin: Optional[float],
    event_risk: str,
    regime: Dict[str, Any],
    context: Optional[Dict[str, Any]] = None,
    gate_overrides: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    return build_gate_result(
        raw_confidence=_safe_float(confidence_overall),
        agreement_score=_safe_float(agreement_score),
        horizon_alignment_score=_safe_float(horizon_alignment_score),
        margin_vs_second=_safe_float(margin),
        event_risk=str(event_risk or "unknown"),
        regime_label=str(regime.get("label") or ""),
        is_high_vol=bool(regime.get("is_high_vol")),
        context=context or {},
        thresholds_override=gate_overrides,
    )


def _build_scenarios(snapshot: Dict[str, Any]) -> Dict[str, Dict[str, float]]:
    likelihoods = compute_likelihoods(snapshot)
    sentiment_regime = _compute_sentiment_regime(snapshot)
    likelihoods = _apply_convexity(likelihoods, sentiment_regime)
    macro_liquidity_raw = snapshot.get("macro_liquidity", {}) or {}
    macro_liquidity_ctx, _, _ = _macro_liquidity_block(macro_liquidity_raw)
    macro_calendar = snapshot.get("macro_calendar", {}) or {}
    likelihoods, _, _ = _apply_liquidity_convexity(likelihoods, macro_liquidity_ctx, macro_calendar)
    intensities = _compute_intensities(snapshot)
    out: Dict[str, Dict[str, float]] = {}
    for name in ("best_case", "base_case", "worst_case"):
        out[name] = {
            "likelihood": float(likelihoods.get(name, 0.0)),
            "intensity": float(intensities.get(name, 0.0)),
        }
    return out


def _build_agreement_detail(snapshot: Dict[str, Any], confidence_block: Dict[str, Any]) -> Dict[str, Any]:
    tech = snapshot.get("technical_features", {}) or {}
    horizon_info = _horizon_alignment_from_multi_horizon(tech)
    agreement_score = _safe_float((confidence_block.get("components") or {}).get("agreement"))
    out: Dict[str, Any] = {
        "agreement_score": round(agreement_score, 3) if agreement_score is not None else None,
        "agreement_label": (
            "high"
            if (agreement_score is not None and agreement_score >= 0.75)
            else "moderate"
            if (agreement_score is not None and agreement_score >= 0.55)
            else "low"
        )
        if agreement_score is not None
        else "unknown",
    }
    if horizon_info is not None:
        out["horizon_alignment_score"] = round(float(horizon_info.get("horizon_alignment_score") or 0.0), 3)
        out["horizon_alignment_label"] = horizon_info.get("horizon_alignment_label")
        out["horizon_dominant_bias"] = horizon_info.get("horizon_dominant_bias")
    return out


def _fallback_narrative(result: Dict[str, Any]) -> Dict[str, Any]:
    decision = result.get("decision", {}) or {}
    strategy = result.get("strategy_inputs", {}) or {}
    gate = result.get("no_trade_gate", {}) or {}
    regime = (strategy.get("regime_classification") or {}).get("market_regime")
    conf = decision.get("confidence")
    top = (strategy.get("scenario_snapshot") or {}).get("top_scenario")
    summary = (
        f"{result.get('symbol')} is currently in {regime} with confidence {conf}. "
        f"Top scenario is {top}. "
        "Decision is deterministic and gated by signal quality thresholds."
    )
    what_change = [
        "agreement_score rises above threshold",
        "horizon alignment improves above threshold",
        "event risk normalizes after key releases",
    ]
    risks = [
        "High volatility can cause fast signal reversals",
        "Event-driven shocks can invalidate setup",
    ]
    if gate.get("active"):
        summary = (
            f"{result.get('symbol')} is in {regime} and currently blocked by the no-trade gate. "
            "The model sees low-quality alignment despite scenario ranking."
        )
    return {
        "summary": summary,
        "what_would_change": what_change,
        "top_risks": risks,
    }


def _fallback_trade_explanation(result: Dict[str, Any], trade_plan: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    action = ((result.get("decision") or {}).get("action") or "").upper()
    signal_quality = ((result.get("strategy_inputs") or {}).get("signal_quality") or {})
    effective_conf = _safe_float(signal_quality.get("effective_confidence"))
    event_risk = signal_quality.get("event_risk")
    regime = ((result.get("strategy_inputs") or {}).get("regime_classification") or {}).get("market_regime")

    if action == "NO_TRADE" or not isinstance(trade_plan, dict):
        reason = (
            "No trade plan generated because the deterministic decision is NO_TRADE."
        )
        return {
            "rationale": reason,
            "risk_context": (
                f"Current regime={regime}, event_risk={event_risk}, "
                f"effective_confidence={effective_conf} keep execution gated."
            ),
            "positioning_logic": "Wait for gate conditions to clear before sizing, entries, or risk targets are activated.",
        }

    position = trade_plan.get("position_sizing", {}) or {}
    risk_mgmt = trade_plan.get("risk_management", {}) or {}
    target = trade_plan.get("target_plan", {}) or {}
    exit_cond = trade_plan.get("exit_conditions", {}) or {}
    entry = trade_plan.get("entry_plan", {}) or {}

    if action == "HOLD":
        escal = (exit_cond.get("escalation_state") or {})
        return {
            "rationale": (
                f"HOLD management uses source={trade_plan.get('source')} with side={trade_plan.get('position_side')} "
                f"and stop={risk_mgmt.get('stop_price_estimate')} target={target.get('target_price_estimate')}."
            ),
            "risk_context": (
                f"Regime={regime}, event_risk={event_risk}, effective_confidence={effective_conf}; "
                f"stop_source={risk_mgmt.get('stop_source')} atr_mode={risk_mgmt.get('atr_mode')} "
                f"anchor_age_hours={risk_mgmt.get('anchor_age_hours')} anchor_stale={risk_mgmt.get('anchor_stale')} "
                f"stop_not_widened={((risk_mgmt.get('validations') or {}).get('stop_not_widened'))}."
            ),
            "positioning_logic": (
                f"Maintain open position with time_stop={exit_cond.get('time_stop_hours')}h, "
                f"invalidation_rules={exit_cond.get('invalidation_rules')}, "
                f"and escalation_recommendation={escal.get('recommended_step')}."
            ),
        }

    return {
        "rationale": (
            f"Position size scales with effective confidence ({effective_conf}) and agreement, "
            f"producing risk_pct={position.get('risk_pct_of_equity')} "
            f"and risk_dollars={position.get('risk_dollars')}."
        ),
        "risk_context": (
            f"Regime={regime} and event_risk={event_risk} drive a stop at "
            f"{risk_mgmt.get('stop_price_estimate')} using {risk_mgmt.get('stop_multiple_atr')} ATR, "
            f"with target RR={target.get('risk_reward_multiple')}."
        ),
        "positioning_logic": (
            f"Entry type={entry.get('type')} (offset_atr={entry.get('limit_offset_atr')}), "
            f"target={target.get('target_price_estimate')}, "
            f"time_stop={exit_cond.get('time_stop_hours')}h."
        ),
    }


def build_decision_output(
    symbol: str,
    snapshot: Dict[str, Any],
    use_gpt: bool = False,
    gpt_model: str = "gpt-5.2",
    current_position_state: str = "FLAT",
    con=None,
    gate_overrides: Optional[Dict[str, Any]] = None,
    intel_interval: str = "1h",
) -> Dict[str, Any]:
    scenarios = _build_scenarios(snapshot)
    freshness = (snapshot.get("meta_features") or {}).get("freshness", {}) or {}
    required_modalities = (
        ((snapshot.get("meta_features") or {}).get("required_modalities") or {})
        if isinstance(snapshot.get("meta_features"), dict)
        else {}
    )
    if not required_modalities:
        required_modalities = evaluate_required_modality_status(freshness)
    freshness_score, _ = _freshness_score(freshness)
    conf_block = _confidence_block(snapshot, scenarios, freshness_score)
    agreement_detail = _build_agreement_detail(snapshot, conf_block)

    tech = snapshot.get("technical_features", {}) or {}
    news = snapshot.get("news_features", {}) or {}
    multi_horizon = tech.get("multi_horizon")
    regime = classify_regime(tech, agreement_detail, multi_horizon)

    top_name, top_like, second_like = _top_two_likelihoods(scenarios)
    margin = float(top_like - second_like)

    macro_calendar = snapshot.get("macro_calendar", {}) or {}
    macro_context = _macro_context_from_calendar(macro_calendar, snapshot.get("macro_calendar_raw", {}) or {})
    event_risk = macro_context.get("event_risk") or "unknown"

    technical_context = {
        "price_above_vwap": _get_tech_value(tech, "price_above_vwap"),
        "vwap_distance_pct": _safe_float(_get_tech_value(tech, "vwap_distance_pct")),
        "rs_vs_btc_7d": _safe_float(_get_tech_value(tech, "rs_vs_btc_7d")),
        "rs_rank": _safe_float(_get_tech_value(tech, "rs_rank")),
        "rs_percentile": _safe_float(_get_tech_value(tech, "rs_percentile")),
        "breadth_score": _safe_float(_get_tech_value(tech, "breadth_score")),
        "pct_symbols_above_ema20": _safe_float(_get_tech_value(tech, "pct_symbols_above_ema20")),
        "pct_symbols_bullish_ema_cross": _safe_float(_get_tech_value(tech, "pct_symbols_bullish_ema_cross")),
        "squeeze_on": bool(_get_tech_value(tech, "squeeze_on") is True),
        "squeeze_fired": bool(_get_tech_value(tech, "squeeze_fired") is True),
        "squeeze_level": _get_tech_value(tech, "squeeze_level"),
    }
    fund_now = snapshot.get("fundamental_features", {}) or {}
    fundamental_context = {
        "mkt_cap_change_7d": _safe_float(fund_now.get("mkt_cap_change_7d")),
        "mkt_cap_change_30d": _safe_float(fund_now.get("mkt_cap_change_30d")),
        "vol_24h_change_7d": _safe_float(fund_now.get("vol_24h_change_7d")),
        "vol_mcap_ratio_z_30d": _safe_float(fund_now.get("vol_mcap_ratio_z_30d")),
        "fundamental_trend_score": _safe_float(fund_now.get("fundamental_trend_score")),
        "mktcap_fdv_ratio": _safe_float(fund_now.get("mktcap_fdv_ratio")),
        "liquidity_score": _safe_float(fund_now.get("liquidity_score")),
    }

    confidence_overall = round(float(conf_block.get("overall") or 0.0), 3)
    if not bool(required_modalities.get("ok", True)):
        confidence_overall = round(max(0.0, confidence_overall - 0.20), 3)
    conviction_label = _conviction_label(confidence_overall)
    validity_hours = _validity_hours(conviction_label, str(regime.get("label")), str(event_risk), hold_multiplier=1.0)
    deadband_context = dict(technical_context)
    deadband_context["required_modality_ok"] = bool(required_modalities.get("ok", True))
    deadband_context["required_modality_failures"] = list(required_modalities.get("failures") or [])
    deadband = compute_deadband(
        confidence_overall=confidence_overall,
        agreement_score=_safe_float(agreement_detail.get("agreement_score")),
        horizon_alignment_score=_safe_float(agreement_detail.get("horizon_alignment_score")),
        margin=margin,
        event_risk=event_risk,
        regime=regime,
        context=deadband_context,
        gate_overrides=gate_overrides,
    )

    dominant_bias = agreement_detail.get("horizon_dominant_bias")
    if dominant_bias not in {"bullish", "bearish", "neutral"}:
        dominant_bias = regime.get("horizon_bias")
    if dominant_bias not in {"bullish", "bearish", "neutral"}:
        dominant_bias = "neutral"

    longer_bias = None
    hs = ((multi_horizon or {}).get("horizon_signals") or {}) if isinstance(multi_horizon, dict) else {}
    if isinstance(hs, dict):
        longer_bias = (hs.get("14d") or {}).get("trend_bias")

    best_like = float((scenarios.get("best_case") or {}).get("likelihood") or 0.0)
    worst_like = float((scenarios.get("worst_case") or {}).get("likelihood") or 0.0)
    hypothetical_action = _select_hypothetical_action(
        dominant_bias=dominant_bias,
        longer_bias=longer_bias,
        regime_label=str(regime.get("label")),
        best_likelihood=best_like,
        worst_likelihood=worst_like,
        agreement_score=_safe_float(agreement_detail.get("agreement_score")),
        margin=margin,
        thresholds=deadband.get("thresholds") or {},
    )
    current_position_state = _normalize_position_state(current_position_state)
    next_position_state = current_position_state
    flip_alert = False
    flip_strength_score: Optional[float] = None
    flip_tier: Optional[str] = None
    flip_persistence_runs = 0

    if current_position_state == "FLAT":
        action = "NO_TRADE" if deadband.get("active") else hypothetical_action
        if action in {"LONG", "SHORT"}:
            next_position_state = action
        else:
            next_position_state = "FLAT"
    else:
        if hypothetical_action in {"NO_TRADE", current_position_state}:
            action = "HOLD"
            next_position_state = current_position_state
        else:
            flip_alert = True
            prior_runs = _flip_persistence_runs(con, symbol, current_position_state, hypothetical_action) if con is not None else 0
            flip_persistence_runs = prior_runs + 1
            flip_strength_score, flip_tier = _flip_strength(
                margin=margin,
                effective_conf=float(deadband.get("effective_confidence") or confidence_overall),
                agreement_score=_safe_float(agreement_detail.get("agreement_score")),
                horizon_alignment=_safe_float(agreement_detail.get("horizon_alignment_score")),
                regime_label=str(regime.get("label") or ""),
                event_risk=event_risk,
            )
            eff_for_flip = _safe_float(deadband.get("effective_confidence"))
            flip_gate_clear = _weighted_gate_override_pass(
                deadband=deadband,
                effective_confidence=eff_for_flip,
                agreement_score=_safe_float(agreement_detail.get("agreement_score")),
                margin=margin,
            )
            strong_confirmed = bool(
                flip_tier == "strong"
                and flip_persistence_runs >= 2
                and (eff_for_flip is not None and eff_for_flip >= 0.55)
                and ((_safe_float(agreement_detail.get("agreement_score")) or 0.0) >= 0.60)
                and float(margin) >= 0.08
            )
            if strong_confirmed and flip_gate_clear:
                action = hypothetical_action
                next_position_state = hypothetical_action
            else:
                action = "HOLD"
                next_position_state = current_position_state

    if action not in ALLOWED_ACTIONS:
        action = "NO_TRADE"

    top_intensity = 0.0
    if top_name in scenarios:
        top_intensity = float((scenarios[top_name] or {}).get("intensity") or 0.0)

    reasons = deadband.get("reasons", []) or []
    blocked_codes: List[str] = []
    if bool(deadband.get("active")):
        blocked_codes = [r.get("code") for r in reasons[:3] if r.get("code")]
    quality_flags: List[str] = []
    if not bool(deadband.get("active")):
        weighted = deadband.get("weighted_reasons") or []
        quality_flags = [str(r.get("code")) for r in weighted if r.get("code")]
        quality_flags = list(dict.fromkeys([c for c in quality_flags if c]))
    if flip_alert and action == "HOLD":
        if flip_tier == "strong" and deadband.get("active"):
            blocked_codes.append("REVERSAL_BLOCKED_BY_GATES")
        elif flip_tier != "strong" or flip_persistence_runs < 2:
            blocked_codes.append("REVERSAL_NOT_CONFIRMED")
    blocked_codes = list(dict.fromkeys([c for c in blocked_codes if c]))
    effective_confidence = _safe_float(deadband.get("effective_confidence"))
    confidence_for_plan = effective_confidence if effective_confidence is not None else confidence_overall
    agreement_for_plan = _safe_float(agreement_detail.get("agreement_score")) or 0.0
    regime_label = str(regime.get("label") or "range")
    plan_regime = _regime_for_trade_plan(regime_label)
    vol_regime = str(_get_tech_value(tech, "vol_regime") or "")
    fear_greed_value = _safe_int(((snapshot.get("macro_sentiment") or {}).get("fear_greed") or {}).get("value"), default=50)
    entry_price = _resolve_entry_price(snapshot, tech)
    atr_value = _resolve_atr_value(entry_price, tech)

    trade_plan: Optional[Dict[str, Any]] = None
    if action in {"LONG", "SHORT"}:
        position_sizing = compute_position_size(
            effective_confidence=float(confidence_for_plan),
            raw_confidence=float(confidence_overall),
            agreement_score=float(agreement_for_plan),
            margin_vs_second=float(margin),
            regime=plan_regime,
            volatility_regime=vol_regime,
            event_risk=event_risk,
            fear_greed_value=fear_greed_value,
        )
        entry_plan = compute_entry_plan(
            confidence=float(confidence_for_plan),
            regime=plan_regime,
            atr_value=float(atr_value),
        )
        stop_plan = compute_stop(
            action=action,
            entry_price=float(entry_price),
            atr_value=float(atr_value),
            regime=plan_regime,
        )
        target_plan = compute_target(
            action=action,
            entry_price=float(entry_price),
            stop_price=float(stop_plan.get("stop_price_estimate") or 0.0),
            confidence=float(confidence_for_plan),
        )
        exit_conditions = compute_exit_conditions(plan_regime)
        exit_conditions["time_stop_hours"] = min(168, max(24, int(validity_hours or 24)))
        stop_distance = abs(float(entry_price) - float(stop_plan.get("stop_price_estimate") or 0.0))
        rr_expected = _expected_rr(float(confidence_for_plan))
        max_risk_pct = 0.015 if (plan_regime == "high_vol_chop" and str(event_risk).lower() == "elevated") else 0.03
        trade_plan = {
            "mode": "new_entry",
            "position_sizing": position_sizing,
            "entry_plan": entry_plan,
            "risk_management": {
                **stop_plan,
                "entry_price_estimate": round(float(entry_price), 2),
                "stop_distance": round(stop_distance, 6),
                "validations": {
                    "stop_distance_positive": bool(stop_distance > 0.0),
                    "risk_dollars_within_cap": bool(position_sizing["risk_dollars"] <= 100000 * max_risk_pct + 1e-9),
                },
            },
            "target_plan": {
                **target_plan,
                "validations": {
                    "rr_multiple_matches_confidence": bool(abs(float(target_plan["risk_reward_multiple"]) - rr_expected) < 1e-9),
                },
            },
            "exit_conditions": {
                **exit_conditions,
                "validations": {
                    "time_stop_positive": bool(int(exit_conditions.get("time_stop_hours") or 0) > 0),
                },
            },
        }
    elif action == "HOLD" and current_position_state in {"LONG", "SHORT"}:
        trade_plan = _build_hold_trade_plan(
            symbol=symbol,
            side=current_position_state,
            con=con,
            plan_regime=plan_regime,
            entry_price_now=float(entry_price),
            atr_value=float(atr_value),
            confidence_for_plan=float(confidence_for_plan),
            confidence_overall=float(confidence_overall),
            agreement_for_plan=float(agreement_for_plan),
            margin=float(margin),
            vol_regime=vol_regime,
            event_risk=event_risk,
            fear_greed_value=int(fear_greed_value),
            validity_hours=int(validity_hours or 24),
        )
        if isinstance(trade_plan, dict):
            exit_cond = trade_plan.get("exit_conditions") or {}
            exit_cond["escalation_state"] = _build_hold_escalation_state(
                con=con,
                symbol=symbol,
                side=current_position_state,
                gate_active=bool(deadband.get("active")),
                agreement_score=_safe_float(agreement_detail.get("agreement_score")),
                effective_confidence=effective_confidence,
            )
            trade_plan["exit_conditions"] = exit_cond

    snapshot_provenance = snapshot.get("provenance") or {}
    run_id = str(snapshot.get("run_id") or snapshot_provenance.get("run_id") or _new_run_id("dec"))
    snapshot_id = str(snapshot.get("snapshot_id") or snapshot_provenance.get("snapshot_id") or "")
    input_hash = str(snapshot.get("input_hash") or snapshot_provenance.get("input_hash") or "")
    if not input_hash:
        input_hash = _stable_json_hash(
            {
                "symbol": symbol,
                "technical_features": snapshot.get("technical_features") or {},
                "news_features": snapshot.get("news_features") or {},
                "fundamental_features": snapshot.get("fundamental_features") or {},
                "freshness": freshness,
            }
        )
    if not snapshot_id:
        snapshot_id = f"snap_{input_hash[:16]}"

    result: Dict[str, Any] = {
        "symbol": symbol,
        "run_id": run_id,
        "snapshot_id": snapshot_id,
        "input_hash": input_hash,
        "price": round(float(entry_price), 2) if float(entry_price) > 0 else None,
        "position_state": current_position_state,
        "next_position_state": next_position_state,
        "asof": _fmt_utc(datetime.now(timezone.utc)),
        "decision_logic_version": DECISION_LOGIC_VERSION,
        "decision": {
            "action": action,
            "confidence": confidence_overall,
            "conviction_label": conviction_label,
            "validity_hours": int(validity_hours),
        },
        "strategy_inputs": {
            "regime_classification": {
                "market_regime": regime.get("label"),
                "confidence": regime.get("confidence"),
            },
            "scenario_snapshot": {
                "top_scenario": top_name,
                "likelihood": round(top_like, 3),
                "intensity": round(top_intensity, 3),
                "margin_vs_second": round(margin, 3),
            },
            "signal_quality": {
                "raw_confidence": confidence_overall,
                "effective_confidence": deadband.get("effective_confidence"),
                "agreement_score": _safe_float(agreement_detail.get("agreement_score")),
                "horizon_alignment_score": _safe_float(agreement_detail.get("horizon_alignment_score")),
                "event_risk": event_risk,
                "freshness_score": round(float(freshness_score), 3),
                "hold_window_source": "deterministic_swing_v1",
            },
            "data_quality": {
                "required_modalities": required_modalities,
            },
            "technical_context": technical_context,
            "fundamental_context": fundamental_context,
        },
        "no_trade_gate": {
            "active": bool(deadband.get("active")),
            "gate_model": deadband.get("gate_model"),
            "gate_policy_version": deadband.get("gate_policy_version"),
            "hard_blockers": deadband.get("hard_blockers") or [],
            "weighted_reasons": deadband.get("weighted_reasons") or [],
            "active_blocking_checks": deadband.get("active_blocking_checks") or [],
            "diagnostic_only_checks": deadband.get("diagnostic_only_checks") or [],
            "disabled_checks": deadband.get("disabled_checks") or [],
            "penalty_total": deadband.get("penalty_total"),
            "penalty_threshold": deadband.get("penalty_threshold"),
            "penalty_ratio": deadband.get("penalty_ratio"),
            "passed_checks": deadband.get("passed_checks") or [],
            "activation_basis": deadband.get("activation_basis"),
            "reasons": reasons,
            "quality_adjustments": deadband.get("quality_adjustments") or [],
            "override_conditions": [
                "no hard blockers",
                "penalty_total < penalty_threshold",
            ],
        },
        "paper_trade_preview": {
            "would_trade_if_allowed": bool(current_position_state == "FLAT" and hypothetical_action in {"LONG", "SHORT"}),
            "hypothetical_action": hypothetical_action,
            "blocked_by": blocked_codes,
            "quality_flags": quality_flags,
            "flip_alert": flip_alert,
            "flip_strength": flip_strength_score,
            "flip_tier": flip_tier,
            "flip_persistence_runs": flip_persistence_runs,
            "state_transition": {
                "from": current_position_state,
                "to": next_position_state,
            },
        },
        "trade_plan": trade_plan if action in {"LONG", "SHORT", "HOLD"} else None,
        "trade_explanation": {},
        "explain_like_human": {},
        "evidence_audit": {
            "highlights": [
                (
                    f"top_scenario={top_name} (like={round(top_like, 3)}, "
                    f"intensity={round(top_intensity, 3)}, margin={round(margin, 3)})"
                ),
                f"agreement_score={_safe_float(agreement_detail.get('agreement_score'))}",
                f"horizon_alignment_score={_safe_float(agreement_detail.get('horizon_alignment_score'))}",
                f"market_regime={regime.get('label')} (regime_conf={regime.get('confidence')})",
                (
                    "context "
                    f"vwap_distance_pct={technical_context.get('vwap_distance_pct')} "
                    f"rs_vs_btc_7d={technical_context.get('rs_vs_btc_7d')} "
                    f"breadth_score={technical_context.get('breadth_score')} "
                    f"squeeze_on={technical_context.get('squeeze_on')}"
                ),
                (
                    "fundamentals "
                    f"trend_score={fundamental_context.get('fundamental_trend_score')} "
                    f"mkt_cap_change_7d={fundamental_context.get('mkt_cap_change_7d')} "
                    f"vol_mcap_ratio_z_30d={fundamental_context.get('vol_mcap_ratio_z_30d')}"
                ),
                (
                    f"event_risk={event_risk} "
                    f"(requires confidence >= {deadband['thresholds']['min_confidence_elevated_event_risk']:.2f} when elevated)"
                ),
                (
                    "required_modalities "
                    f"ok={bool(required_modalities.get('ok', True))} "
                    f"failures={required_modalities.get('failures') or []}"
                ),
            ],
            "thresholds": deadband["thresholds"],
        },
    }

    shadow_intel = _build_shadow_intelligence_block(
        con=con,
        symbol=symbol,
        decision_payload=result,
        interval=str(intel_interval or "1h"),
    )
    result["strategy_inputs"]["shadow_intelligence"] = shadow_intel
    try:
        result["evidence_audit"]["highlights"].append(
            "shadow_intel "
            f"status={shadow_intel.get('status')} "
            f"stage={shadow_intel.get('stage')} "
            f"pred_conf={((shadow_intel.get('prediction') or {}).get('confidence'))}"
        )
    except Exception:
        pass

    narrative = _fallback_narrative(result)
    narrative_source = "deterministic_fallback"
    trade_explanation = _fallback_trade_explanation(result, result.get("trade_plan"))
    trade_expl_source = "deterministic_fallback"

    if use_gpt:
        payload = {
            "symbol": symbol,
            "action": result["decision"]["action"],
            "confidence": result["decision"]["confidence"],
            "conviction_label": result["decision"]["conviction_label"],
            "regime": result["strategy_inputs"]["regime_classification"],
            "scenario_snapshot": result["strategy_inputs"]["scenario_snapshot"],
            "signal_quality": result["strategy_inputs"]["signal_quality"],
            "no_trade_gate": result["no_trade_gate"],
            "evidence_highlights": result["evidence_audit"]["highlights"][:5],
        }
        gpt_narrative = render_decision_narrative_gpt52(payload, model=gpt_model)
        if gpt_narrative:
            narrative = gpt_narrative
            narrative_source = "gpt"
        else:
            narrative_source = "fallback_after_gpt_failure"
        if result.get("trade_plan") is not None:
            trade_payload = {
                "symbol": symbol,
                "action": result["decision"]["action"],
                "regime": result["strategy_inputs"]["regime_classification"],
                "signal_quality": result["strategy_inputs"]["signal_quality"],
                "scenario_snapshot": result["strategy_inputs"]["scenario_snapshot"],
                "trade_plan": result.get("trade_plan"),
                "event_risk": event_risk,
            }
            gpt_trade_explanation = render_trade_explanation_gpt52(trade_payload, model=gpt_model)
            if gpt_trade_explanation:
                trade_explanation = gpt_trade_explanation
                trade_expl_source = "gpt"
            else:
                trade_expl_source = "fallback_after_gpt_failure"

    narrative["source"] = narrative_source
    narrative["model"] = gpt_model if narrative_source == "gpt" else None
    result["explain_like_human"] = narrative
    trade_explanation["source"] = trade_expl_source
    trade_explanation["model"] = gpt_model if trade_expl_source == "gpt" else None
    result["trade_explanation"] = trade_explanation
    return result


def persist_decision_output(con, result: Dict[str, Any]) -> None:
    symbol = str(result.get("symbol") or "")
    asof = str(result.get("asof") or "")
    decision = result.get("decision", {}) or {}
    strategy_inputs = result.get("strategy_inputs", {}) or {}
    regime = (strategy_inputs.get("regime_classification") or {}) or {}
    scenario = (strategy_inputs.get("scenario_snapshot") or {}) or {}
    gate = result.get("no_trade_gate", {}) or {}
    preview = result.get("paper_trade_preview", {}) or {}
    persistence = ((result.get("strategy_inputs") or {}).get("persistence") or {})

    con.execute(
        """
        INSERT OR REPLACE INTO trident_decisions (
            asof_utc, symbol, action, confidence, conviction_label, validity_hours,
            hypothetical_action, deadband_active, blocked_by_json,
            regime_label, regime_confidence, top_scenario, top_likelihood,
            likelihood_margin, logic_version, run_id, snapshot_id, input_hash, persistence_status,
            payload_json, created_at_utc
        ) VALUES (
            CAST(? AS TIMESTAMP), ?, ?, ?, ?, ?,
            ?, ?, ?,
            ?, ?, ?, ?,
            ?, ?, ?, ?, ?, ?, ?, ?
        )
        """,
        [
            asof,
            symbol,
            str(decision.get("action") or ""),
            _safe_float(decision.get("confidence")),
            str(decision.get("conviction_label") or ""),
            int(decision.get("validity_hours") or 0),
            str(preview.get("hypothetical_action") or ""),
            bool(gate.get("active")),
            json.dumps(preview.get("blocked_by") or []),
            str(regime.get("market_regime") or ""),
            _safe_float(regime.get("confidence")),
            str(scenario.get("top_scenario") or ""),
            _safe_float(scenario.get("likelihood")),
            _safe_float(scenario.get("margin_vs_second")),
            str(result.get("decision_logic_version") or DECISION_LOGIC_VERSION),
            str(result.get("run_id") or ""),
            str(result.get("snapshot_id") or ""),
            str(result.get("input_hash") or ""),
            str(persistence.get("status") or "ok"),
            json.dumps(result, separators=(",", ":")),
            datetime.now(timezone.utc),
        ],
    )
    try:
        con.execute(
            """
            INSERT OR REPLACE INTO run_provenance
                (run_id, snapshot_id, input_hash, command, symbol, asof_utc, status, notes, created_at_utc)
            VALUES (?, ?, ?, ?, ?, CAST(? AS TIMESTAMP), ?, ?, ?)
            """,
            [
                str(result.get("run_id") or ""),
                str(result.get("snapshot_id") or ""),
                str(result.get("input_hash") or ""),
                "decision:trident",
                symbol,
                asof,
                str(persistence.get("status") or "ok"),
                None,
                datetime.now(timezone.utc),
            ],
        )
    except Exception:
        pass


def load_latest_phase4_snapshot(symbol: str, con, interval: Optional[str] = None) -> Dict[str, Any]:
    snapshot = build_phase4_snapshot(symbol, con)
    if interval:
        tech = compute_tech_features(symbol, con, interval=interval) or {}
        snapshot["technical_features"] = tech
        news = snapshot.get("news_features", {}) or {}
        fund = snapshot.get("fundamental_features", {}) or {}
        meta = snapshot.get("meta_features", {}) or {}
        meta["agreement_score"] = _continuous_agreement_score(tech, news, fund)
        snapshot["meta_features"] = meta
    return snapshot


def run_decision_trident(
    symbol: str,
    con,
    interval: Optional[str] = None,
    use_gpt: bool = False,
    gpt_model: str = "gpt-5.2",
    current_position_state_override: Optional[str] = None,
    gate_overrides: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    snapshot = load_latest_phase4_snapshot(symbol=symbol, con=con, interval=interval)
    if current_position_state_override is not None:
        current_position_state = _normalize_position_state(current_position_state_override)
    else:
        current_position_state = _infer_position_state(con, symbol)
    result = build_decision_output(
        symbol=symbol,
        snapshot=snapshot,
        use_gpt=use_gpt,
        gpt_model=gpt_model,
        current_position_state=current_position_state,
        con=con,
        gate_overrides=gate_overrides,
        intel_interval=str(interval or "1h"),
    )
    strategy_inputs = result.get("strategy_inputs") or {}
    persist_meta = {
        "attempted": True,
        "status": "pending",
        "run_id": str(result.get("run_id") or _new_run_id("dec")),
        "snapshot_id": str(result.get("snapshot_id") or ""),
        "error": None,
    }
    strategy_inputs["persistence"] = persist_meta
    result["strategy_inputs"] = strategy_inputs
    try:
        persist_meta["status"] = "ok"
        persist_decision_output(con, result)
    except Exception as exc:
        persist_meta["status"] = "error"
        persist_meta["error"] = str(exc)
        print(
            f"[decision:trident][warn] decision_persist_failed run_id={persist_meta['run_id']} error={exc}",
            file=sys.stderr,
        )
    return result

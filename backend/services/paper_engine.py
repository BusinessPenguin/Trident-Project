"""Deterministic paper trading execution helpers."""

from __future__ import annotations

import math
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any, Callable, Dict, List, Optional, Tuple

from backend.decide.utils import aggression_knobs_for_tier, normalize_aggression_tier
from backend.db.paper_repo import (
    close_position_and_exit_fill,
    compute_equity_snapshot,
    get_last_position_activity_ts,
    insert_mark,
    record_replay_event,
)


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return float(default)
        return float(value)
    except Exception:
        return float(default)


def _parse_ts(value: Any) -> Optional[datetime]:
    if value is None:
        return None
    if isinstance(value, datetime):
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc)
    s = str(value).strip()
    if not s:
        return None
    try:
        if s.endswith("Z"):
            s = s[:-1] + "+00:00"
        dt = datetime.fromisoformat(s)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    except Exception:
        return None


def _to_utc_naive(ts: datetime) -> datetime:
    dt = _parse_ts(ts) or datetime.now(timezone.utc)
    return dt.astimezone(timezone.utc).replace(tzinfo=None)


def _strong_move_structure_pass(
    effective_confidence: float,
    agreement_score: float,
    margin_vs_second: float,
) -> bool:
    return bool(
        float(effective_confidence) >= 0.60
        and float(agreement_score) >= 0.65
        and float(margin_vs_second) >= 0.08
    )


def fallback_aggression_profile(
    baseline: Optional[Dict[str, Any]] = None,
    rationale: str = "deterministic balanced fallback",
) -> Dict[str, Any]:
    knobs = aggression_knobs_for_tier("balanced", baseline=baseline)
    return {
        "tier": "balanced",
        "source": "deterministic_fallback",
        "override_weighted_gate": False,
        "quick_exit_bias": "none",
        "confidence": 0.0,
        "rationale": str(rationale),
        "knobs_applied": knobs,
    }


def resolve_aggression_profile(
    decision_json: Dict[str, Any],
    gpt_profile: Optional[Dict[str, Any]] = None,
    baseline: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    fallback = fallback_aggression_profile(baseline=baseline)
    if not isinstance(gpt_profile, dict) or not gpt_profile:
        return fallback

    tier = normalize_aggression_tier(gpt_profile.get("tier"))
    quick_exit_bias = str(gpt_profile.get("quick_exit_bias") or "none").lower()
    if quick_exit_bias not in {"none", "take_profit", "cut_loss"}:
        quick_exit_bias = "none"

    conf = _safe_float(gpt_profile.get("confidence"), 0.0)
    conf = max(0.0, min(1.0, conf))
    rationale = str(gpt_profile.get("reasoning_summary") or "").strip() or "gpt aggression profile"

    return {
        "tier": tier,
        "source": "gpt",
        "override_weighted_gate": bool(gpt_profile.get("override_weighted_gate")),
        "quick_exit_bias": quick_exit_bias,
        "confidence": round(conf, 6),
        "rationale": rationale,
        "knobs_applied": aggression_knobs_for_tier(tier, baseline=baseline),
    }


def gate_override_allowed(
    decision_json: Dict[str, Any],
    aggression_profile: Optional[Dict[str, Any]],
    allow_weighted_gate_override: bool = True,
) -> Tuple[bool, bool]:
    strategy_inputs = decision_json.get("strategy_inputs") or {}
    signal_quality = strategy_inputs.get("signal_quality") or {}
    scenario = strategy_inputs.get("scenario_snapshot") or {}
    gate = decision_json.get("no_trade_gate") or {}
    hard_blockers = gate.get("hard_blockers") or []
    activation_basis = str(gate.get("activation_basis") or "")

    effective_conf = _safe_float(signal_quality.get("effective_confidence"), 0.0)
    agreement = _safe_float(signal_quality.get("agreement_score"), 0.0)
    margin = _safe_float(scenario.get("margin_vs_second"), 0.0)
    strong_pass = _strong_move_structure_pass(effective_conf, agreement, margin)

    tier = normalize_aggression_tier((aggression_profile or {}).get("tier"))
    override_requested = bool((aggression_profile or {}).get("override_weighted_gate"))
    allowed = bool(
        allow_weighted_gate_override
        and
        override_requested
        and tier in {"assertive", "very_aggressive"}
        and activation_basis == "penalty_threshold"
        and not hard_blockers
        and strong_pass
    )
    return allowed, strong_pass


def extract_candidate_from_decision(
    decision_json: Dict[str, Any],
    penalty_high_vol_chop: float = 0.9,
    penalty_elevated_event_risk: float = 0.9,
    aggression_profile: Optional[Dict[str, Any]] = None,
    allow_weighted_gate_override: bool = True,
) -> Dict[str, Any]:
    decision = decision_json.get("decision") or {}
    strategy_inputs = decision_json.get("strategy_inputs") or {}
    signal_quality = strategy_inputs.get("signal_quality") or {}
    regime_block = strategy_inputs.get("regime_classification") or {}
    gate = decision_json.get("no_trade_gate") or {}
    preview = decision_json.get("paper_trade_preview") or {}
    gate_active = bool(gate.get("active"))

    profile = aggression_profile or fallback_aggression_profile()
    action = str(decision.get("action") or "NO_TRADE").upper()
    side = "NO_TRADE"
    if action in {"LONG", "SHORT"}:
        side = action
    elif action == "HOLD":
        hyp = str(preview.get("hypothetical_action") or "").upper()
        side = hyp if hyp in {"LONG", "SHORT"} else "NO_TRADE"
    elif action == "NO_TRADE":
        hyp = str(preview.get("hypothetical_action") or "").upper()
        if hyp in {"LONG", "SHORT"}:
            side = hyp

    confidence = _safe_float(decision.get("confidence"), 0.0)
    effective_conf = _safe_float(signal_quality.get("effective_confidence"), confidence)
    agreement_score = _safe_float(signal_quality.get("agreement_score"), 0.0)
    freshness_score = _safe_float(signal_quality.get("freshness_score"), 0.0)
    regime = str(regime_block.get("market_regime") or "")
    event_risk = str(signal_quality.get("event_risk") or "").lower()

    gate_reasons = gate.get("reasons") or []
    gate_reason_codes = [str(r.get("code")) for r in gate_reasons if isinstance(r, dict) and r.get("code")]
    override_weighted_gate, strong_move_structure_pass = gate_override_allowed(
        decision_json=decision_json,
        aggression_profile=profile,
        allow_weighted_gate_override=allow_weighted_gate_override,
    )
    gate_enforced = bool(gate_active and not override_weighted_gate)

    gates_blocking = gate_reason_codes if gate_enforced else []
    quality_flags = gate_reason_codes if not gate_enforced else []
    score = effective_conf * agreement_score * freshness_score
    if regime == "high_vol_chop":
        score *= float(penalty_high_vol_chop)
    if event_risk == "elevated":
        score *= float(penalty_elevated_event_risk)
    technical_context = strategy_inputs.get("technical_context") or {}
    breadth_score = _safe_float(technical_context.get("breadth_score"), float("nan"))
    rs_vs_btc_7d = _safe_float(technical_context.get("rs_vs_btc_7d"), float("nan"))
    vwap_distance_pct = _safe_float(technical_context.get("vwap_distance_pct"), float("nan"))
    squeeze_on = bool(technical_context.get("squeeze_on"))
    quality_set = {str(c) for c in quality_flags if c}
    if not gate_enforced:
        if "LOW_BREADTH" in quality_set:
            score *= 0.92
        if "RELATIVE_WEAKNESS" in quality_set and side == "LONG":
            score *= 0.90
        if "PRE_BREAKOUT_COMPRESSION" in quality_set:
            score *= 0.95
        if not math.isnan(breadth_score) and breadth_score < 0.40:
            score *= 0.95
        if not math.isnan(rs_vs_btc_7d):
            if side == "LONG" and rs_vs_btc_7d < -0.02:
                score *= 0.90
            elif side == "SHORT" and rs_vs_btc_7d > 0.02:
                score *= 0.90
        if squeeze_on and not math.isnan(vwap_distance_pct) and abs(vwap_distance_pct) < 0.01:
            score *= 0.97
    if gate_enforced:
        score = 0.0
        side = "NO_TRADE"

    return {
        "symbol": str(decision_json.get("symbol") or ""),
        "side": side,
        "confidence": round(confidence, 6),
        "effective_confidence": round(effective_conf, 6),
        "agreement_score": round(agreement_score, 6),
        "freshness_score": round(freshness_score, 6),
        "candidate_score": round(score, 6),
        "gate_active": gate_active,
        "gates_blocking": gates_blocking,
        "quality_flags": quality_flags,
        "override_weighted_gate": bool(override_weighted_gate),
        "strong_move_structure_pass": bool(strong_move_structure_pass),
        "aggression_tier": str(profile.get("tier") or "balanced"),
        "aggression_source": str(profile.get("source") or "deterministic_fallback"),
        "aggression_quick_exit_bias": str(profile.get("quick_exit_bias") or "none"),
        "aggression_knobs": profile.get("knobs_applied") or {},
        "regime": regime,
        "event_risk": event_risk,
    }


def _recent_candle_rows(conn, symbol: str, interval: str, limit: int = 220) -> List[Dict[str, Any]]:
    rows = conn.execute(
        """
        SELECT ts, open, high, low, close
        FROM candles
        WHERE symbol = ? AND interval = ?
        ORDER BY ts DESC
        LIMIT ?
        """,
        [symbol, interval, int(max(10, limit))],
    ).fetchall()
    out: List[Dict[str, Any]] = []
    for r in rows:
        out.append(
            {
                "ts": _parse_ts(r[0]),
                "open": _safe_float(r[1], 0.0),
                "high": _safe_float(r[2], 0.0),
                "low": _safe_float(r[3], 0.0),
                "close": _safe_float(r[4], 0.0),
            }
        )
    out.reverse()
    return out


def _clip(x: float, low: float, high: float) -> float:
    return max(float(low), min(float(high), float(x)))


def _interval_minutes(interval: str) -> int:
    i = str(interval or "15m").strip().lower()
    if i.endswith("m"):
        try:
            return max(1, int(i[:-1]))
        except Exception:
            return 15
    if i.endswith("h"):
        try:
            return max(1, int(i[:-1])) * 60
        except Exception:
            return 60
    return 15


def resolve_intel_stage(
    closed_trades: int,
    mode: str,
    bootstrap: int = 25,
    promotion: int = 50,
) -> Dict[str, Any]:
    requested = str(mode or "auto").strip().lower()
    if requested not in {"auto", "shadow", "soft", "hard"}:
        requested = "auto"
    ct = max(0, int(closed_trades or 0))
    bs = max(1, int(bootstrap or 25))
    pm = max(bs + 1, int(promotion or 50))
    if requested == "hard":
        stage = "hard"
    elif requested == "shadow":
        stage = "shadow"
    elif requested == "soft":
        stage = "soft" if ct >= bs else "shadow"
    else:
        if ct < bs:
            stage = "shadow"
        elif ct < pm:
            stage = "soft"
        else:
            stage = "hard"
    return {
        "mode": requested,
        "stage": stage,
        "closed_trades": ct,
        "bootstrap_trades": bs,
        "promotion_trades": pm,
        "bootstrap_guard_active": bool(ct < bs and requested != "hard"),
    }


def build_prediction_snapshot(
    conn,
    symbol: str,
    decision_json: Dict[str, Any],
    candidate: Dict[str, Any],
    interval: str = "15m",
) -> Dict[str, Any]:
    strategy_inputs = decision_json.get("strategy_inputs") or {}
    signal_quality = strategy_inputs.get("signal_quality") or {}
    scenario = strategy_inputs.get("scenario_snapshot") or {}
    technical = strategy_inputs.get("technical_context") or {}
    regime = str((strategy_inputs.get("regime_classification") or {}).get("market_regime") or "range")
    event_risk = str(signal_quality.get("event_risk") or "normal").lower()

    bars = _recent_candle_rows(conn, symbol=symbol, interval=interval, limit=220)
    latest_close = _safe_float(bars[-1]["close"], 0.0) if bars else 0.0
    min_6h = int(round(360.0 / _interval_minutes(interval)))
    min_24h = int(round(1440.0 / _interval_minutes(interval)))
    mom_6h = 0.0
    mom_24h = 0.0
    if len(bars) > min_6h and latest_close > 0:
        ref = _safe_float(bars[-1 - min_6h]["close"], latest_close)
        if ref > 0:
            mom_6h = (latest_close / ref) - 1.0
    if len(bars) > min_24h and latest_close > 0:
        ref = _safe_float(bars[-1 - min_24h]["close"], latest_close)
        if ref > 0:
            mom_24h = (latest_close / ref) - 1.0

    effective_conf = _clip(_safe_float(candidate.get("effective_confidence"), 0.0), 0.0, 1.0)
    agreement = _clip(_safe_float(candidate.get("agreement_score"), 0.0), 0.0, 1.0)
    freshness = _clip(_safe_float(candidate.get("freshness_score"), 0.0), 0.0, 1.0)
    margin = _clip(_safe_float(scenario.get("margin_vs_second"), 0.0), 0.0, 0.30)
    breadth = _clip(_safe_float(technical.get("breadth_score"), 0.5), 0.0, 1.0)
    rs_7d = _clip(_safe_float(technical.get("rs_vs_btc_7d"), 0.0), -0.20, 0.20)
    vwap_dist = _clip(_safe_float(technical.get("vwap_distance_pct"), 0.0), -0.10, 0.10)

    # Fixed-weight deterministic score.
    score = (
        (effective_conf * 0.30)
        + (agreement * 0.18)
        + (freshness * 0.10)
        + ((margin / 0.06) * 0.10)
        + (_clip(mom_6h, -0.05, 0.05) / 0.05 * 0.10)
        + (_clip(mom_24h, -0.12, 0.12) / 0.12 * 0.12)
        + (((breadth * 2.0) - 1.0) * 0.07)
        + ((rs_7d / 0.20) * 0.08)
        - ((abs(vwap_dist) / 0.10) * 0.03)
    )
    if regime == "high_vol_chop":
        score -= 0.08
    if event_risk == "elevated":
        score -= 0.08
    score = _clip(score, -1.5, 1.5)

    if score >= 0.12:
        d24 = "LONG"
    elif score <= -0.12:
        d24 = "SHORT"
    else:
        d24 = "FLAT"
    if score >= 0.18:
        d48 = "LONG"
    elif score <= -0.18:
        d48 = "SHORT"
    else:
        d48 = "FLAT"

    confidence = _clip((abs(score) / 1.2), 0.0, 1.0)
    exp24 = _clip(score * 220.0, -600.0, 600.0)
    exp48 = _clip(score * 360.0, -900.0, 900.0)
    ev_r = _clip((confidence * abs(exp24) / 100.0) - ((1.0 - confidence) * 0.60), -1.0, 1.2)

    return {
        "version": "pred_v1",
        "score": round(float(score), 6),
        "direction_24h": d24,
        "direction_48h": d48,
        "confidence": round(float(confidence), 6),
        "expected_move_bps_24h": round(float(exp24), 4),
        "expected_move_bps_48h": round(float(exp48), 4),
        "ev_r": round(float(ev_r), 6),
        "features": {
            "mom_6h": round(float(mom_6h), 6),
            "mom_24h": round(float(mom_24h), 6),
            "breadth_score": round(float(breadth), 6),
            "rs_vs_btc_7d": round(float(rs_7d), 6),
            "vwap_distance_pct": round(float(vwap_dist), 6),
            "margin_vs_second": round(float(margin), 6),
        },
    }


def build_pattern_snapshot(
    conn,
    symbol: str,
    decision_json: Dict[str, Any],
    interval: str = "15m",
) -> Dict[str, Any]:
    strategy_inputs = decision_json.get("strategy_inputs") or {}
    technical = strategy_inputs.get("technical_context") or {}
    bars = _recent_candle_rows(conn, symbol=symbol, interval=interval, limit=220)
    if not bars:
        return {
            "version": "pat_v1",
            "hits": [],
            "support_long": 0.0,
            "support_short": 0.0,
            "conflict_ratio": 0.0,
        }

    closes = [float(b.get("close") or 0.0) for b in bars if _safe_float(b.get("close"), 0.0) > 0]
    highs = [float(b.get("high") or 0.0) for b in bars]
    lows = [float(b.get("low") or 0.0) for b in bars]
    last_close = closes[-1] if closes else 0.0
    ema21 = sum(closes[-21:]) / 21.0 if len(closes) >= 21 else (sum(closes) / len(closes) if closes else 0.0)
    ema55 = sum(closes[-55:]) / 55.0 if len(closes) >= 55 else (sum(closes) / len(closes) if closes else 0.0)
    prev20_high = max(highs[-21:-1]) if len(highs) >= 21 else max(highs[:-1] or [0.0])
    prev20_low = min(lows[-21:-1]) if len(lows) >= 21 else min(lows[:-1] or [0.0])
    std20 = 0.0
    if len(closes) >= 20:
        mean20 = sum(closes[-20:]) / 20.0
        var20 = sum((c - mean20) ** 2 for c in closes[-20:]) / 20.0
        std20 = math.sqrt(max(0.0, var20))
        bb_z = (last_close - mean20) / std20 if std20 > 0 else 0.0
    else:
        bb_z = 0.0
    rs_7d = _safe_float(technical.get("rs_vs_btc_7d"), 0.0)
    breadth = _safe_float(technical.get("breadth_score"), 0.5)
    squeeze_on = bool(technical.get("squeeze_on"))
    vwap_dist = _safe_float(technical.get("vwap_distance_pct"), 0.0)
    mom_6h = 0.0
    mom_24h = 0.0
    iv = _interval_minutes(interval)
    i6 = int(round(360.0 / iv))
    i24 = int(round(1440.0 / iv))
    if len(closes) > i6 and closes[-1 - i6] > 0:
        mom_6h = (last_close / closes[-1 - i6]) - 1.0
    if len(closes) > i24 and closes[-1 - i24] > 0:
        mom_24h = (last_close / closes[-1 - i24]) - 1.0

    gains = []
    losses = []
    for idx in range(max(1, len(closes) - 2), len(closes)):
        d = closes[idx] - closes[idx - 1]
        gains.append(max(0.0, d))
        losses.append(max(0.0, -d))
    rs = (sum(gains) / max(1.0, len(gains))) / max(1e-9, (sum(losses) / max(1.0, len(losses))))
    rsi2 = 100.0 - (100.0 / (1.0 + rs))

    hits: List[str] = []
    long_hits = 0
    short_hits = 0

    def _hit(code: str, side: str, cond: bool) -> None:
        nonlocal long_hits, short_hits
        if not cond:
            return
        hits.append(code)
        if side == "LONG":
            long_hits += 1
        elif side == "SHORT":
            short_hits += 1

    _hit("TC_EMA_STACK_LONG", "LONG", ema21 > ema55 and mom_6h > 0 and mom_24h > 0)
    _hit("TC_EMA_STACK_SHORT", "SHORT", ema21 < ema55 and mom_6h < 0 and mom_24h < 0)
    _hit("TC_PULLBACK_EMA21_LONG", "LONG", ema21 > ema55 and ema21 > 0 and abs((last_close - ema21) / ema21) <= 0.004)
    _hit(
        "TC_PULLBACK_EMA21_SHORT",
        "SHORT",
        ema21 < ema55 and ema21 > 0 and abs((last_close - ema21) / ema21) <= 0.004,
    )
    _hit("BO_RANGE_BREAKOUT_LONG", "LONG", last_close > prev20_high * 1.001 if prev20_high > 0 else False)
    _hit("BO_RANGE_BREAKDOWN_SHORT", "SHORT", last_close < prev20_low * 0.999 if prev20_low > 0 else False)
    _hit("BO_DONCHIAN20_LONG", "LONG", last_close >= prev20_high if prev20_high > 0 else False)
    _hit("BO_DONCHIAN20_SHORT", "SHORT", last_close <= prev20_low if prev20_low > 0 else False)
    _hit("MR_BB_REVERT_LONG", "LONG", bb_z <= -1.8 and mom_6h > 0)
    _hit("MR_BB_REVERT_SHORT", "SHORT", bb_z >= 1.8 and mom_6h < 0)
    _hit("MR_RSI2_EXTREME_LONG", "LONG", rsi2 <= 10.0)
    _hit("MR_RSI2_EXTREME_SHORT", "SHORT", rsi2 >= 90.0)
    _hit("VOL_SQUEEZE_RELEASE_LONG", "LONG", (not squeeze_on) and abs(vwap_dist) >= 0.012 and mom_6h > 0)
    _hit("VOL_SQUEEZE_RELEASE_SHORT", "SHORT", (not squeeze_on) and abs(vwap_dist) >= 0.012 and mom_6h < 0)
    _hit("RS_OUTPERFORM_BTC_LONG", "LONG", rs_7d > 0.01 and breadth >= 0.45)
    _hit("RS_UNDERPERFORM_BTC_SHORT", "SHORT", rs_7d < -0.01 and breadth <= 0.55)

    support_long = _clip(long_hits / 8.0, 0.0, 1.0)
    support_short = _clip(short_hits / 8.0, 0.0, 1.0)
    mx = max(support_long, support_short)
    mn = min(support_long, support_short)
    conflict = (mn / mx) if mx > 0 else 0.0
    return {
        "version": "pat_v1",
        "hits": hits,
        "support_long": round(float(support_long), 6),
        "support_short": round(float(support_short), 6),
        "conflict_ratio": round(float(conflict), 6),
    }


def apply_intelligence(
    candidate: Dict[str, Any],
    prediction: Optional[Dict[str, Any]],
    patterns: Optional[Dict[str, Any]],
    stage_cfg: Dict[str, Any],
) -> Dict[str, Any]:
    side = str(candidate.get("side") or "NO_TRADE")
    stage = str((stage_cfg or {}).get("stage") or "shadow")
    pred_w = _safe_float((stage_cfg or {}).get("prediction_weight"), 0.0)
    pat_w = _safe_float((stage_cfg or {}).get("pattern_weight"), 0.0)
    pred = prediction or {}
    pat = patterns or {}

    delta = 0.0
    blockers: List[str] = []
    used_for_entry = stage in {"soft", "hard"} and side in {"LONG", "SHORT"}
    pred_dir = str(pred.get("direction_24h") or "FLAT")
    pred_conf = _clip(_safe_float(pred.get("confidence"), 0.0), 0.0, 1.0)
    ev_r = _safe_float(pred.get("ev_r"), 0.0)
    support_long = _clip(_safe_float(pat.get("support_long"), 0.0), 0.0, 1.0)
    support_short = _clip(_safe_float(pat.get("support_short"), 0.0), 0.0, 1.0)
    support_side = support_long if side == "LONG" else support_short
    support_opp = support_short if side == "LONG" else support_long

    if side in {"LONG", "SHORT"} and stage in {"soft", "hard"}:
        if pred_dir == side:
            delta += pred_w * max(0.0, pred_conf - 0.5)
        elif pred_dir in {"LONG", "SHORT"}:
            delta -= pred_w * (0.25 + max(0.0, pred_conf - 0.5))
        delta += pat_w * (support_side - 0.5)
        delta -= pat_w * max(0.0, support_opp - 0.55)
        delta = _clip(delta, -0.15, 0.15)

    if side in {"LONG", "SHORT"} and stage == "hard":
        if pred_conf < 0.52:
            blockers.append("INTEL_PREDICTION_LOW_CONF")
        if pred_dir in {"LONG", "SHORT"} and pred_dir != side and pred_conf >= 0.58:
            blockers.append("INTEL_PREDICTION_MISMATCH")
        if support_side < 0.35:
            blockers.append("INTEL_PATTERN_SUPPORT_LOW")
        if ev_r < -0.05:
            blockers.append("INTEL_NEGATIVE_EV")

    if stage == "shadow":
        used_for_entry = False
        delta = 0.0
        blockers = []

    return {
        "prediction": pred,
        "patterns": pat,
        "intel_score_delta": round(float(delta), 6),
        "intel_blockers": blockers,
        "intel_used_for_entry": bool(used_for_entry),
    }


def get_latest_mid_price(conn, symbol: str, interval: str = "1h") -> Tuple[Optional[float], Optional[datetime]]:
    row = conn.execute(
        """
        SELECT close, ts
        FROM candles
        WHERE symbol = ? AND interval = ?
        ORDER BY ts DESC
        LIMIT 1
        """,
        [symbol, interval],
    ).fetchone()
    if not row:
        return None, None
    close_px = _safe_float(row[0], default=0.0)
    if close_px <= 0:
        return None, _parse_ts(row[1])
    return close_px, _parse_ts(row[1])


def load_replay_bars(
    conn,
    symbol: str,
    interval: str,
    from_ts: datetime,
    to_ts: datetime,
) -> List[Dict[str, Any]]:
    from_bound = _parse_ts(from_ts) or datetime.now(timezone.utc)
    to_bound = _parse_ts(to_ts) or datetime.now(timezone.utc)
    if from_bound > to_bound:
        from_bound, to_bound = to_bound, from_bound
    query_from = _to_utc_naive(from_bound)
    query_to = _to_utc_naive(to_bound)
    rows = conn.execute(
        """
        SELECT ts, open, high, low, close
        FROM candles
        WHERE symbol = ? AND interval = ? AND ts >= ? AND ts <= ?
        ORDER BY ts ASC
        """,
        [symbol, interval, query_from, query_to],
    ).fetchall()
    out: List[Dict[str, Any]] = []
    for row in rows:
        ts = _parse_ts(row[0])
        if ts is None:
            continue
        # Guard against DuckDB/local-tz coercion differences: only keep bars
        # strictly inside the intended UTC window.
        if ts < from_bound or ts > to_bound:
            continue
        out.append(
            {
                "ts": ts,
                "open": _safe_float(row[1], 0.0),
                "high": _safe_float(row[2], 0.0),
                "low": _safe_float(row[3], 0.0),
                "close": _safe_float(row[4], 0.0),
            }
        )
    return out


def _apply_replay_fill_policy(
    side: str,
    trigger_type: str,
    trigger_price: float,
    bar_open: float,
) -> Tuple[float, bool]:
    side_u = str(side).upper()
    t = str(trigger_type).upper()
    tp = float(trigger_price)
    bo = float(bar_open)
    gap_fill = False
    if t == "STOP_HIT_REPLAY":
        if side_u == "LONG" and bo <= tp:
            gap_fill = True
            return bo, gap_fill
        if side_u == "SHORT" and bo >= tp:
            gap_fill = True
            return bo, gap_fill
        return tp, gap_fill
    if t == "TP_HIT_REPLAY":
        if side_u == "LONG" and bo >= tp:
            gap_fill = True
            return bo, gap_fill
        if side_u == "SHORT" and bo <= tp:
            gap_fill = True
            return bo, gap_fill
        return tp, gap_fill
    # TIME_STOP_REPLAY uses bar open by policy.
    return bo, False


def resolve_replay_exit_for_position(
    position: Dict[str, Any],
    bars: List[Dict[str, Any]],
    rule_config: Optional[Dict[str, Any]] = None,
) -> Optional[Dict[str, Any]]:
    if not bars:
        return None
    rules = rule_config or {}
    side = str(position.get("side") or "").upper()
    stop = _safe_float(position.get("stop_price"), math.nan)
    tp_raw = position.get("take_profit_price")
    tp = _safe_float(tp_raw, math.nan) if tp_raw is not None else math.nan
    time_stop_ts = _parse_ts(position.get("time_stop_ts"))
    resolution_rule = str(rules.get("ambiguous_resolution") or "stop_first")

    for bar in bars:
        bts = _parse_ts(bar.get("ts"))
        if bts is None:
            continue
        bar_open = _safe_float(bar.get("open"), 0.0)
        bar_high = _safe_float(bar.get("high"), 0.0)
        bar_low = _safe_float(bar.get("low"), 0.0)

        if time_stop_ts is not None and bts >= time_stop_ts:
            fill_price, gap_fill = _apply_replay_fill_policy(side, "TIME_STOP_REPLAY", bar_open, bar_open)
            return {
                "bar_ts": bts,
                "trigger_type": "TIME_STOP_REPLAY",
                "trigger_price": bar_open,
                "fill_price": fill_price,
                "gap_fill": gap_fill,
                "ambiguous_bar": False,
                "resolution_rule": resolution_rule,
                "notes": "time_stop<=bar_ts",
            }

        stop_hit = False
        tp_hit = False
        if side == "LONG":
            stop_hit = (not math.isnan(stop)) and (bar_low <= stop)
            tp_hit = (not math.isnan(tp)) and (bar_high >= tp)
        elif side == "SHORT":
            stop_hit = (not math.isnan(stop)) and (bar_high >= stop)
            tp_hit = (not math.isnan(tp)) and (bar_low <= tp)
        else:
            continue

        if not stop_hit and not tp_hit:
            continue

        ambiguous = bool(stop_hit and tp_hit)
        if ambiguous:
            trigger = "STOP_HIT_REPLAY" if resolution_rule == "stop_first" else "TP_HIT_REPLAY"
        elif stop_hit:
            trigger = "STOP_HIT_REPLAY"
        else:
            trigger = "TP_HIT_REPLAY"

        trigger_price = stop if trigger == "STOP_HIT_REPLAY" else tp
        fill_price, gap_fill = _apply_replay_fill_policy(side, trigger, trigger_price, bar_open)
        return {
            "bar_ts": bts,
            "trigger_type": trigger,
            "trigger_price": float(trigger_price),
            "fill_price": float(fill_price),
            "gap_fill": bool(gap_fill),
            "ambiguous_bar": ambiguous,
            "resolution_rule": resolution_rule,
            "notes": "intrabar_touch",
        }
    return None


def replay_open_positions(
    conn,
    positions: List[Dict[str, Any]],
    config: Dict[str, Any],
    command: str,
    run_id: Optional[str] = None,
    interval: str = "15m",
    now_utc: Optional[datetime] = None,
) -> Dict[str, Any]:
    now = now_utc or datetime.now(timezone.utc)
    interval_eff = str(interval or "15m")
    risk_cfg = config.get("risk_limits") or {}
    lookback_bars = max(1, int(_safe_float(risk_cfg.get("replay_lookback_bars"), 672)))
    fallback_interval = "1h" if interval_eff != "1h" else interval_eff
    replay_events: List[Dict[str, Any]] = []
    exits_triggered: List[Dict[str, Any]] = []
    fee_bps = _safe_float(config.get("fee_bps"), 5.0)

    minutes = 15
    if interval_eff.endswith("m"):
        try:
            minutes = max(1, int(interval_eff[:-1]))
        except Exception:
            minutes = 15
    elif interval_eff.endswith("h"):
        try:
            minutes = max(1, int(interval_eff[:-1])) * 60
        except Exception:
            minutes = 60
    max_lookback = timedelta(minutes=minutes * lookback_bars)

    windows: List[Tuple[datetime, datetime]] = []

    for pos in positions:
        position_id = str(pos.get("position_id") or "")
        entry_ts = _parse_ts(pos.get("entry_ts")) or now
        last_activity_ts = get_last_position_activity_ts(conn, position_id)
        from_ts = entry_ts
        if last_activity_ts is not None and last_activity_ts > from_ts:
            from_ts = last_activity_ts
        if now - from_ts > max_lookback:
            from_ts = now - max_lookback
        if from_ts > now:
            from_ts = now

        windows.append((from_ts, now))
        symbol = str(pos.get("symbol") or "")
        bars = load_replay_bars(conn, symbol=symbol, interval=interval_eff, from_ts=from_ts, to_ts=now)
        used_interval = interval_eff
        if not bars and fallback_interval != interval_eff:
            bars = load_replay_bars(conn, symbol=symbol, interval=fallback_interval, from_ts=from_ts, to_ts=now)
            used_interval = fallback_interval

        event = resolve_replay_exit_for_position(
            position=pos,
            bars=bars,
            rule_config={"ambiguous_resolution": "stop_first"},
        )
        if not event:
            continue

        qty = _safe_float(pos.get("qty"), 0.0)
        side = str(pos.get("side") or "").upper()
        fill_price = _safe_float(event.get("fill_price"), 0.0)
        trigger_price = _safe_float(event.get("trigger_price"), fill_price)
        fees_usd = abs(qty * fill_price) * (fee_bps / 10000.0)
        slippage_usd = abs(fill_price - trigger_price) * abs(qty)
        ts_event = event.get("bar_ts") or now
        reason = str(event.get("trigger_type") or "REPLAY_EXIT")

        close_position_and_exit_fill(
            conn=conn,
            position_id=position_id,
            exit_dict={"exit_ts": ts_event, "exit_price": fill_price, "exit_reason": reason},
            fill_dict={
                "fill_id": f"fill_{uuid.uuid4().hex}",
                "ts": ts_event,
                "fill_price": fill_price,
                "fees_usd": fees_usd,
                "slippage_usd": slippage_usd,
                "qty": qty,
                "type": "EXIT",
            },
        )

        replay_event = {
            "event_id": f"replay_{uuid.uuid4().hex}",
            "ts": now,
            "command": str(command or ""),
            "run_id": run_id,
            "position_id": position_id,
            "symbol": symbol,
            "side": side,
            "interval": used_interval,
            "window_from": from_ts,
            "window_to": now,
            "bar_ts": ts_event,
            "trigger_type": reason,
            "trigger_price": trigger_price,
            "fill_price": fill_price,
            "gap_fill": bool(event.get("gap_fill")),
            "ambiguous_bar": bool(event.get("ambiguous_bar")),
            "resolution_rule": str(event.get("resolution_rule") or "stop_first"),
            "notes": str(event.get("notes") or ""),
        }
        record_replay_event(conn, replay_event)
        replay_events.append(
            {
                "position_id": position_id,
                "symbol": symbol,
                "trigger_type": reason,
                "bar_ts": ts_event,
                "fill_price": round(fill_price, 8),
                "gap_fill": bool(event.get("gap_fill")),
                "ambiguous_bar": bool(event.get("ambiguous_bar")),
            }
        )
        exits_triggered.append(
            {
                "position_id": position_id,
                "symbol": symbol,
                "reason": reason,
                "exit_price": round(fill_price, 8),
                "exit_origin": "replay",
            }
        )

    window_from = None
    if windows:
        window_from = min(w[0] for w in windows)
    return {
        "interval": interval_eff,
        "window_from": window_from,
        "window_to": now,
        "positions_checked": len(positions),
        "exits_replayed": len(exits_triggered),
        "events": replay_events,
        "exits_triggered": exits_triggered,
    }


def smart_refresh_symbol(
    conn,
    symbol: str,
    interval: str = "1h",
    force_refresh: bool = False,
    max_age_hours: float = 2.0,
    refresh_callback: Optional[Callable[[str, str], None]] = None,
    now_utc: Optional[datetime] = None,
) -> List[str]:
    notes: List[str] = []
    if now_utc is None:
        now_utc = datetime.now(timezone.utc)
    _, latest_ts = get_latest_mid_price(conn, symbol=symbol, interval=interval)
    is_stale = latest_ts is None or ((now_utc - latest_ts).total_seconds() / 3600.0 > float(max_age_hours))
    if force_refresh or is_stale:
        notes.append("crypto_backfill")
        if refresh_callback:
            refresh_callback(symbol, interval)
    return notes


def simulate_fill(
    mid_price: float,
    side: str,
    qty: float,
    fee_bps: float,
    slippage_bps: float,
    fill_type: str,
) -> Dict[str, Any]:
    side_u = str(side).upper()
    fill_type_u = str(fill_type).upper()
    px = float(mid_price)
    slip = float(slippage_bps) / 10000.0
    if fill_type_u == "ENTRY":
        if side_u == "LONG":
            fill_px = px * (1.0 + slip)
        else:
            fill_px = px * (1.0 - slip)
    else:
        # Exit slippage is adverse to the position side.
        if side_u == "LONG":
            fill_px = px * (1.0 - slip)
        else:
            fill_px = px * (1.0 + slip)
    notional = abs(float(qty) * fill_px)
    fees = notional * (float(fee_bps) / 10000.0)
    slippage_usd = abs(fill_px - px) * abs(float(qty))
    return {
        "fill_price": round(fill_px, 8),
        "fees_usd": round(fees, 8),
        "slippage_usd": round(slippage_usd, 8),
        "qty": float(qty),
        "type": fill_type_u,
    }


def compute_position_plan(
    entry_price: float,
    side: str,
    equity: float,
    risk_cfg: Dict[str, Any],
    atr_pct: Optional[float] = None,
    validity_hours: int = 8,
    stop_mult_override: Optional[float] = None,
    aggression_knobs: Optional[Dict[str, Any]] = None,
    hard_max_hold_hours: int = 168,
    now_utc: Optional[datetime] = None,
) -> Dict[str, Any]:
    now = now_utc or datetime.now(timezone.utc)
    ep = float(entry_price)
    side_u = str(side).upper()
    knobs = aggression_knobs or {}
    risk_mult = max(0.10, _safe_float(knobs.get("risk_mult"), 1.0))
    stop_mult_aggr = max(0.10, _safe_float(knobs.get("stop_mult"), 1.0))
    hold_mult = max(0.10, _safe_float(knobs.get("hold_mult"), 1.0))
    notional_cap_aggr = _safe_float(knobs.get("notional_cap"), 0.0)

    max_risk_pct_base = _safe_float(risk_cfg.get("max_risk_per_trade_pct"), 0.01)
    max_risk_pct = min(0.02, max_risk_pct_base * risk_mult)
    max_total_exposure_pct = _safe_float(risk_cfg.get("max_total_exposure_pct"), 0.60)
    if notional_cap_aggr > 0:
        max_total_exposure_pct = min(max_total_exposure_pct, notional_cap_aggr)
    fallback_stop_pct = _safe_float(risk_cfg.get("stop_distance_pct_fallback"), 0.015)
    stop_mult_base = _safe_float(stop_mult_override, _safe_float(risk_cfg.get("stop_distance_atr_mult"), 1.0))
    stop_mult = max(0.1, stop_mult_base * stop_mult_aggr)
    atr_pct_v = _safe_float(atr_pct, 0.0)

    if atr_pct_v > 0:
        stop_distance_pct = max(atr_pct_v * max(stop_mult, 0.1), 1e-6)
        stop_method = "atr_pct"
    else:
        stop_distance_pct = max(fallback_stop_pct, 1e-6)
        stop_method = "fallback_pct"

    if side_u == "LONG":
        stop_price = ep * (1.0 - stop_distance_pct)
    else:
        stop_price = ep * (1.0 + stop_distance_pct)
    risk_distance = abs(ep - stop_price)
    risk_usd = float(equity) * max_risk_pct
    qty = risk_usd / risk_distance if risk_distance > 0 else 0.0
    qty_cap = (float(equity) * max_total_exposure_pct) / ep if ep > 0 else 0.0
    qty = max(0.0, min(qty, qty_cap))
    if side_u == "LONG":
        tp_price = ep + (risk_distance * 2.0)
    else:
        tp_price = ep - (risk_distance * 2.0)
    effective_validity_hours = int(round(float(validity_hours or 24) * hold_mult))
    effective_validity_hours = max(24, min(int(hard_max_hold_hours), effective_validity_hours))
    time_stop_ts = now + timedelta(hours=effective_validity_hours)
    return {
        "entry_price": round(ep, 8),
        "stop_price": round(stop_price, 8),
        "take_profit_price": round(tp_price, 8),
        "time_stop_ts": time_stop_ts,
        "qty": round(qty, 8),
        "risk_distance": round(risk_distance, 8),
        "risk_usd": round(risk_usd, 8),
        "stop_distance_pct": round(stop_distance_pct, 8),
        "stop_method": stop_method,
        "validity_hours_applied": int(effective_validity_hours),
        "aggression_knobs": {
            "risk_mult": round(risk_mult, 6),
            "stop_mult": round(stop_mult_aggr, 6),
            "hold_mult": round(hold_mult, 6),
            "notional_cap": round(max_total_exposure_pct, 6),
        },
    }


def evaluate_exit(
    position: Dict[str, Any],
    mid_price: float,
    now_utc: datetime,
    decision_flip: Optional[Dict[str, Any]] = None,
    allow_flip_exit: bool = False,
    allow_quick_exit: bool = True,
) -> Optional[str]:
    side = str(position.get("side") or "").upper()
    stop = _safe_float(position.get("stop_price"), math.nan)
    tp = position.get("take_profit_price")
    tp_v = _safe_float(tp, math.nan) if tp is not None else math.nan
    time_stop_ts = _parse_ts(position.get("time_stop_ts"))
    mid = float(mid_price)

    if side == "LONG" and not math.isnan(stop) and mid <= stop:
        return "STOP_HIT"
    if side == "SHORT" and not math.isnan(stop) and mid >= stop:
        return "STOP_HIT"
    if side == "LONG" and not math.isnan(tp_v) and mid >= tp_v:
        return "TP_HIT"
    if side == "SHORT" and not math.isnan(tp_v) and mid <= tp_v:
        return "TP_HIT"
    if time_stop_ts is not None and now_utc >= time_stop_ts:
        return "TIME_STOP"

    if decision_flip and allow_quick_exit:
        bias = str(decision_flip.get("quick_exit_bias") or "none").lower()
        quick_allowed = bool(
            bias in {"take_profit", "cut_loss"}
            and _safe_float(decision_flip.get("effective_confidence"), 0.0) >= 0.65
            and _safe_float(decision_flip.get("agreement_score"), 0.0) >= 0.65
            and _safe_float(decision_flip.get("penalty_ratio"), 1.0) < 0.20
            and not bool(decision_flip.get("hard_blockers"))
        )
        entry_ts = _parse_ts(position.get("entry_ts"))
        same_day = bool(entry_ts is not None and (now_utc - entry_ts).total_seconds() <= 86400)
        entry_px = _safe_float(position.get("entry_price"), mid)
        if quick_allowed and same_day:
            if side == "LONG" and bias == "take_profit" and mid > entry_px:
                return "QUICK_TAKE_PROFIT"
            if side == "SHORT" and bias == "take_profit" and mid < entry_px:
                return "QUICK_TAKE_PROFIT"
            if side == "LONG" and bias == "cut_loss" and mid < entry_px:
                return "QUICK_CUT_LOSS"
            if side == "SHORT" and bias == "cut_loss" and mid > entry_px:
                return "QUICK_CUT_LOSS"

    if decision_flip and allow_flip_exit:
        action = str(decision_flip.get("action") or "").upper()
        conf = _safe_float(decision_flip.get("confidence"), 0.0)
        agreement = _safe_float(decision_flip.get("agreement_score"), 0.0)
        margin = _safe_float(decision_flip.get("margin_vs_second"), 0.0)
        gate_active = bool(decision_flip.get("gate_active"))
        gate_override = bool(decision_flip.get("override_weighted_gate"))
        if conf >= 0.55 and agreement >= 0.60 and margin >= 0.08 and (not gate_active or gate_override):
            if side == "LONG" and action == "SHORT":
                return "FLIP_EXIT"
            if side == "SHORT" and action == "LONG":
                return "FLIP_EXIT"
    return None


def evaluate_entry_eligibility(
    candidate: Dict[str, Any],
    decision_json: Dict[str, Any],
    cfg: Dict[str, Any],
    intel_result: Optional[Dict[str, Any]] = None,
    quality_veto_mode: str = "full",
) -> Tuple[bool, List[str]]:
    blockers: List[str] = []
    side = str(candidate.get("side") or "NO_TRADE")
    if side not in {"LONG", "SHORT"}:
        blockers.append("side_not_directional")
        return False, blockers

    min_score = _safe_float(cfg.get("entry_min_score"), 0.18)
    min_effective_conf = _safe_float(cfg.get("entry_min_effective_confidence"), 0.40)
    min_agreement = _safe_float(cfg.get("entry_min_agreement"), 0.60)
    min_margin = _safe_float(cfg.get("entry_min_margin"), 0.06)
    quality_veto_enabled = bool(cfg.get("quality_veto_enabled", True))

    score = _safe_float(candidate.get("candidate_score"), 0.0)
    effective_conf = _safe_float(candidate.get("effective_confidence"), 0.0)
    agreement = _safe_float(candidate.get("agreement_score"), 0.0)
    margin = _safe_float(
        ((decision_json.get("strategy_inputs") or {}).get("scenario_snapshot") or {}).get("margin_vs_second"),
        0.0,
    )
    gate = decision_json.get("no_trade_gate") or {}
    hard_blockers = gate.get("hard_blockers") or []
    gate_active = bool(gate.get("active"))
    override_weighted_gate = bool(candidate.get("override_weighted_gate"))
    if gate_active and not override_weighted_gate:
        blockers.append("gate_active")
    if score < min_score:
        blockers.append("entry_min_score")
    if effective_conf < min_effective_conf:
        blockers.append("entry_min_effective_confidence")
    if agreement < min_agreement:
        blockers.append("entry_min_agreement")
    if margin < min_margin:
        blockers.append("entry_min_margin")

    mode = str(quality_veto_mode or "full").lower()
    if quality_veto_enabled:
        q = {str(x) for x in (candidate.get("quality_flags") or [])}
        combo_cfg = cfg.get("quality_veto_combos")
        combos: List[List[str]] = []
        if isinstance(combo_cfg, list):
            for raw in combo_cfg:
                if not isinstance(raw, list):
                    continue
                combo = [str(code or "").strip().upper() for code in raw if str(code or "").strip()]
                if combo:
                    combos.append(combo)
        if not combos:
            # Month-1 simplification: keep only confidence/edge-centric quality veto.
            combos = [["LOW_CONFIDENCE", "MODEL_EDGE_WEAK"]]
        if mode == "bootstrap":
            # Bootstrap remains tighter by only applying the primary combo.
            combos = combos[:1]

        veto_combo = None
        for combo in combos:
            if set(combo).issubset(q):
                veto_combo = f"QUALITY_VETO_{'_'.join(combo)}"
                break
        if veto_combo:
            escape_hatch = bool(
                effective_conf >= 0.55
                and agreement >= 0.80
                and score >= 0.24
                and not hard_blockers
            )
            if not escape_hatch:
                blockers.append(veto_combo)

    if intel_result and bool(intel_result.get("intel_used_for_entry")):
        for code in list(intel_result.get("intel_blockers") or []):
            if code and code not in blockers:
                blockers.append(str(code))

    return len(blockers) == 0, blockers


def run_paper_cycle(
    decisions: List[Dict[str, Any]],
    max_trades: int = 1,
    penalty_high_vol_chop: float = 0.9,
    penalty_elevated_event_risk: float = 0.9,
) -> Dict[str, Any]:
    candidates = [
        extract_candidate_from_decision(
            d,
            penalty_high_vol_chop=penalty_high_vol_chop,
            penalty_elevated_event_risk=penalty_elevated_event_risk,
        )
        for d in decisions
    ]
    valid = [c for c in candidates if c.get("side") in {"LONG", "SHORT"} and _safe_float(c.get("candidate_score")) > 0]
    valid.sort(
        key=lambda c: (
            -_safe_float(c.get("candidate_score")),
            -_safe_float(c.get("confidence")),
            str(c.get("symbol") or ""),
        )
    )
    selected = valid[0] if valid and int(max_trades or 0) > 0 else None
    return {
        "candidates": candidates,
        "selected": selected,
        "placed_trade": False,
        "reasons": [],
    }


def mark_open_positions(
    conn,
    positions: List[Dict[str, Any]],
    config: Dict[str, Any],
    now_utc: Optional[datetime] = None,
    exit_on_flip: bool = False,
    allow_quick_exit: bool = True,
    decision_flip_getter: Optional[Callable[[str], Optional[Dict[str, Any]]]] = None,
) -> Dict[str, Any]:
    now = now_utc or datetime.now(timezone.utc)
    fee_bps = _safe_float(config.get("fee_bps"), 5.0)
    slippage_bps = _safe_float(config.get("slippage_bps"), 8.0)
    exits: List[Dict[str, Any]] = []
    marks_to_insert: List[Dict[str, Any]] = []

    for pos in positions:
        symbol = str(pos.get("symbol") or "")
        side = str(pos.get("side") or "").upper()
        mid, _ = get_latest_mid_price(conn, symbol=symbol, interval="1h")
        if mid is None:
            continue
        entry = _safe_float(pos.get("entry_price"), 0.0)
        qty = _safe_float(pos.get("qty"), 0.0)
        if side == "LONG":
            unrealized = (mid - entry) * qty
        else:
            unrealized = (entry - mid) * qty

        decision_flip = None
        if decision_flip_getter and (exit_on_flip or allow_quick_exit):
            decision_flip = decision_flip_getter(symbol)
        reason = evaluate_exit(
            pos,
            mid_price=mid,
            now_utc=now,
            decision_flip=decision_flip,
            allow_flip_exit=bool(exit_on_flip),
            allow_quick_exit=bool(allow_quick_exit),
        )
        if reason:
            fill = simulate_fill(
                mid_price=mid,
                side=side,
                qty=qty,
                fee_bps=fee_bps,
                slippage_bps=slippage_bps,
                fill_type="EXIT",
            )
            fill_id = f"fill_{uuid.uuid4().hex}"
            close_position_and_exit_fill(
                conn=conn,
                position_id=str(pos["position_id"]),
                exit_dict={"exit_ts": now, "exit_price": fill["fill_price"], "exit_reason": reason},
                fill_dict={
                    "fill_id": fill_id,
                    "ts": now,
                    "fill_price": fill["fill_price"],
                    "fees_usd": fill["fees_usd"],
                    "slippage_usd": fill["slippage_usd"],
                    "qty": qty,
                    "type": "EXIT",
                },
            )
            exits.append(
                {
                    "position_id": str(pos["position_id"]),
                    "symbol": symbol,
                    "reason": reason,
                    "exit_price": fill["fill_price"],
                }
            )

        marks_to_insert.append(
            {
                "mark_id": f"mark_{uuid.uuid4().hex}",
                "ts": now,
                "symbol": symbol,
                "mid_price": float(mid),
                "position_id": str(pos["position_id"]),
                "unrealized_pnl_usd": float(unrealized),
            }
        )

    equity = compute_equity_snapshot(conn, _safe_float(config.get("starting_equity"), 10000.0))
    for mark in marks_to_insert:
        insert_mark(
            conn,
            {
                **mark,
                "equity": equity["equity"],
                "drawdown_pct": equity["drawdown_pct"],
            },
        )
    # Recompute after marks are inserted for a stable post-mark snapshot.
    equity_post = compute_equity_snapshot(conn, _safe_float(config.get("starting_equity"), 10000.0))
    return {
        "exits_triggered": exits,
        "marks_written": len(marks_to_insert),
        "equity": equity_post,
    }

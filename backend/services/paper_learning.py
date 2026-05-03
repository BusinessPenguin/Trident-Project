"""Bounded deterministic learning helpers for paper trading."""

from __future__ import annotations

import json
import math
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple


TAXONOMY = [
    "STOP_TOO_TIGHT",
    "STOP_TOO_WIDE",
    "REGIME_MISCLASSIFIED",
    "EVENT_RISK_UNDERWEIGHTED",
    "SLIPPAGE_TOO_OPTIMISTIC",
    "CONFIDENCE_CALIBRATION",
    "SIGNAL_DISAGREEMENT",
    "TIMING_BAD",
    "NEWS_REVERSAL",
]


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return float(default)
        return float(value)
    except Exception:
        return float(default)


def _json_loads(raw: Any, default: Any) -> Any:
    if raw is None:
        return default
    try:
        parsed = json.loads(str(raw))
    except Exception:
        return default
    return parsed if isinstance(parsed, type(default)) else default


def _clamp(value: float, low: float, high: float) -> float:
    if value < low:
        return float(low)
    if value > high:
        return float(high)
    return float(value)


def load_closed_trades(conn, since_hours: Optional[int] = None, last_n: Optional[int] = None) -> List[Dict[str, Any]]:
    params: List[Any] = []
    sql = """
        SELECT position_id, symbol, side, qty, entry_ts, entry_price, stop_price, take_profit_price,
               exit_ts, exit_price, exit_reason, linked_run_id
        FROM paper_positions
        WHERE status = 'CLOSED'
    """
    if since_hours is not None and since_hours > 0:
        cutoff = datetime.now(timezone.utc) - timedelta(hours=int(since_hours))
        sql += " AND exit_ts >= ?"
        params.append(cutoff)
    sql += " ORDER BY exit_ts DESC"
    if last_n is not None and last_n > 0:
        sql += " LIMIT ?"
        params.append(int(last_n))
    rows = conn.execute(sql, params).fetchall()

    trades: List[Dict[str, Any]] = []
    for row in rows:
        position_id = str(row[0])
        symbol = str(row[1] or "")
        side = str(row[2] or "").upper()
        qty = _safe_float(row[3], 0.0)
        entry_price = _safe_float(row[5], 0.0)
        exit_price = _safe_float(row[9], 0.0)
        if side == "LONG":
            pnl = (exit_price - entry_price) * qty
        else:
            pnl = (entry_price - exit_price) * qty

        fills = conn.execute(
            """
            SELECT type, fill_price, fees_usd, slippage_usd, ts
            FROM paper_fills
            WHERE position_id = ?
            ORDER BY ts ASC
            """,
            [position_id],
        ).fetchall()
        fees_total = sum(_safe_float(f[2], 0.0) for f in fills)
        slippage_total = sum(_safe_float(f[3], 0.0) for f in fills)

        decision_row = conn.execute(
            """
            SELECT decision_json
            FROM paper_decisions
            WHERE run_id = ? AND symbol = ?
            ORDER BY asof_utc DESC
            LIMIT 1
            """,
            [str(row[11] or ""), symbol],
        ).fetchone()
        decision_json = _json_loads(decision_row[0] if decision_row else None, {})
        candidate_row = conn.execute(
            """
            SELECT candidate_json
            FROM paper_candidates
            WHERE run_id = ? AND symbol = ?
            ORDER BY candidate_score DESC
            LIMIT 1
            """,
            [str(row[11] or ""), symbol],
        ).fetchone()
        candidate_json = _json_loads(candidate_row[0] if candidate_row else None, {})
        signal_quality = ((decision_json.get("strategy_inputs") or {}).get("signal_quality") or {})
        regime = ((decision_json.get("strategy_inputs") or {}).get("regime_classification") or {}).get("market_regime")
        top_scenario = ((decision_json.get("strategy_inputs") or {}).get("scenario_snapshot") or {}).get("top_scenario")

        trades.append(
            {
                "position_id": position_id,
                "symbol": symbol,
                "side": side,
                "qty": qty,
                "entry_ts": row[4],
                "entry_price": entry_price,
                "stop_price": _safe_float(row[6], 0.0),
                "take_profit_price": _safe_float(row[7], 0.0) if row[7] is not None else None,
                "exit_ts": row[8],
                "exit_price": exit_price,
                "exit_reason": str(row[10] or ""),
                "linked_run_id": str(row[11] or ""),
                "gross_pnl": float(pnl),
                "fees_total": float(fees_total),
                "slippage_total": float(slippage_total),
                "event_risk": str(signal_quality.get("event_risk") or "").lower(),
                "agreement_score": _safe_float(signal_quality.get("agreement_score"), 0.0),
                "effective_confidence": _safe_float(signal_quality.get("effective_confidence"), 0.0),
                "regime": str(regime or ""),
                "top_scenario": str(top_scenario or ""),
                "aggression_tier": str(candidate_json.get("aggression_tier") or "balanced"),
                "aggression_source": str(candidate_json.get("aggression_source") or "unknown"),
            }
        )
    return trades


def classify_trade_failure(trade: Dict[str, Any]) -> Optional[str]:
    pnl = _safe_float(trade.get("gross_pnl"), 0.0)
    if pnl >= 0:
        return None
    exit_reason = str(trade.get("exit_reason") or "").upper()
    slippage_total = _safe_float(trade.get("slippage_total"), 0.0)
    effective_conf = _safe_float(trade.get("effective_confidence"), 0.0)
    agreement = _safe_float(trade.get("agreement_score"), 0.0)
    event_risk = str(trade.get("event_risk") or "")
    regime = str(trade.get("regime") or "")
    top_scenario = str(trade.get("top_scenario") or "")

    if exit_reason in {"STOP_HIT", "STOP_HIT_REPLAY"}:
        if slippage_total > abs(pnl) * 0.15:
            return "SLIPPAGE_TOO_OPTIMISTIC"
        return "STOP_TOO_TIGHT"
    if exit_reason in {"TIME_STOP", "TIME_STOP_REPLAY"}:
        return "TIMING_BAD"
    if exit_reason == "FLIP_EXIT":
        return "SIGNAL_DISAGREEMENT"
    if event_risk == "elevated":
        return "EVENT_RISK_UNDERWEIGHTED"
    if regime == "high_vol_chop":
        return "REGIME_MISCLASSIFIED"
    if top_scenario == "best_case" and pnl < 0:
        return "NEWS_REVERSAL"
    if effective_conf < 0.45 or agreement < 0.52:
        return "CONFIDENCE_CALIBRATION"
    return "TIMING_BAD"


def classify_failures(trades: List[Dict[str, Any]]) -> Tuple[Dict[str, int], List[Dict[str, Any]]]:
    counts = {code: 0 for code in TAXONOMY}
    labeled: List[Dict[str, Any]] = []
    for trade in trades:
        code = classify_trade_failure(trade)
        if code is None:
            continue
        counts[code] = counts.get(code, 0) + 1
        labeled.append({"position_id": trade.get("position_id"), "symbol": trade.get("symbol"), "code": code})
    return counts, labeled


def _step_limited(value_old: float, value_prop: float, max_abs: float) -> float:
    delta = value_prop - value_old
    if delta > max_abs:
        return value_old + max_abs
    if delta < -max_abs:
        return value_old - max_abs
    return value_prop


def _step_limited_rel(value_old: float, value_prop: float, rel_limit: float) -> float:
    max_delta = abs(value_old) * rel_limit
    return _step_limited(value_old, value_prop, max_delta)


def propose_parameter_updates(
    risk_limits: Dict[str, Any],
    learning_policy: Dict[str, Any],
    counts: Dict[str, int],
    trades: Optional[List[Dict[str, Any]]] = None,
    gpt_policy_proposal: Optional[Dict[str, Any]] = None,
    max_gpt_influence: float = 0.30,
    min_trades_for_gpt_influence: int = 10,
    learn_scope: str = "full",
    freeze_aggression_baseline: bool = False,
    alpha: float = 0.2,
) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any], str, Dict[str, Any], Dict[str, Any]]:
    old_min_conf = _safe_float(risk_limits.get("min_confidence"), 0.33)
    old_stop_mult = _safe_float(risk_limits.get("stop_distance_atr_mult"), 1.0)
    old_entry_min_score = _safe_float(risk_limits.get("entry_min_score"), 0.18)
    old_entry_min_effective_conf = _safe_float(risk_limits.get("entry_min_effective_confidence"), 0.40)
    old_entry_min_agreement = _safe_float(risk_limits.get("entry_min_agreement"), 0.60)
    old_entry_min_margin = _safe_float(risk_limits.get("entry_min_margin"), 0.06)
    old_penalty_chop = _safe_float(learning_policy.get("penalty_high_vol_chop"), 0.9)
    old_penalty_event = _safe_float(learning_policy.get("penalty_elevated_event_risk"), 0.9)
    baseline = dict(learning_policy.get("aggression_baseline") or {})
    old_aggr_risk = _safe_float(baseline.get("risk_mult"), 1.0)
    old_aggr_stop = _safe_float(baseline.get("stop_mult"), 1.0)
    old_aggr_hold = _safe_float(baseline.get("hold_mult"), 1.0)
    old_aggr_exposure = _safe_float(baseline.get("exposure_cap"), 0.12)

    prop_min_conf = old_min_conf
    prop_stop_mult = old_stop_mult
    prop_penalty_chop = old_penalty_chop
    prop_penalty_event = old_penalty_event
    prop_entry_min_score = old_entry_min_score
    prop_entry_min_effective_conf = old_entry_min_effective_conf
    prop_entry_min_agreement = old_entry_min_agreement
    prop_entry_min_margin = old_entry_min_margin
    prop_aggr_risk = old_aggr_risk
    prop_aggr_stop = old_aggr_stop
    prop_aggr_hold = old_aggr_hold
    prop_aggr_exposure = old_aggr_exposure

    if counts.get("STOP_TOO_TIGHT", 0) > counts.get("STOP_TOO_WIDE", 0):
        prop_stop_mult = old_stop_mult * 1.05
    elif counts.get("STOP_TOO_WIDE", 0) > counts.get("STOP_TOO_TIGHT", 0):
        prop_stop_mult = old_stop_mult * 0.95

    if counts.get("EVENT_RISK_UNDERWEIGHTED", 0) > 0:
        prop_penalty_event = old_penalty_event * 0.95
    if counts.get("REGIME_MISCLASSIFIED", 0) > 0:
        prop_penalty_chop = old_penalty_chop * 0.95
    if counts.get("CONFIDENCE_CALIBRATION", 0) + counts.get("SIGNAL_DISAGREEMENT", 0) > 0:
        prop_min_conf = old_min_conf + 0.02

    if counts.get("STOP_TOO_TIGHT", 0) > counts.get("STOP_TOO_WIDE", 0):
        prop_aggr_stop = old_aggr_stop * 1.05
        prop_aggr_risk = old_aggr_risk * 0.98
    elif counts.get("STOP_TOO_WIDE", 0) > counts.get("STOP_TOO_TIGHT", 0):
        prop_aggr_stop = old_aggr_stop * 0.95
        prop_aggr_risk = old_aggr_risk * 1.02

    if counts.get("TIMING_BAD", 0) + counts.get("SIGNAL_DISAGREEMENT", 0) > 0:
        prop_aggr_hold = old_aggr_hold * 1.03

    if counts.get("EVENT_RISK_UNDERWEIGHTED", 0) > 0:
        prop_aggr_exposure = old_aggr_exposure * 0.97

    aggressive_losses = 0
    defensive_losses = 0
    for trade in trades or []:
        pnl = _safe_float((trade or {}).get("gross_pnl"), 0.0)
        if pnl >= 0:
            continue
        tier = str((trade or {}).get("aggression_tier") or "balanced")
        if tier in {"assertive", "very_aggressive"}:
            aggressive_losses += 1
        elif tier in {"defensive", "very_defensive"}:
            defensive_losses += 1
    if aggressive_losses > defensive_losses:
        prop_aggr_risk = prop_aggr_risk * 0.97
        prop_aggr_exposure = prop_aggr_exposure * 0.98

    deterministic_proposal = {
        "min_confidence": prop_min_conf,
        "stop_distance_atr_mult": prop_stop_mult,
        "penalty_high_vol_chop": prop_penalty_chop,
        "penalty_elevated_event_risk": prop_penalty_event,
        "entry_min_score": prop_entry_min_score,
        "entry_min_effective_confidence": prop_entry_min_effective_conf,
        "entry_min_agreement": prop_entry_min_agreement,
        "entry_min_margin": prop_entry_min_margin,
        "aggression_risk_mult": prop_aggr_risk,
        "aggression_stop_mult": prop_aggr_stop,
        "aggression_hold_mult": prop_aggr_hold,
        "aggression_exposure_cap": prop_aggr_exposure,
    }
    gpt_deltas = dict((gpt_policy_proposal or {}).get("deltas") or {})
    gpt_confidence = _safe_float((gpt_policy_proposal or {}).get("confidence"), 0.0)
    closed_count = len(trades or [])
    min_gpt_trades = max(1, int(min_trades_for_gpt_influence or 10))
    if closed_count < min_gpt_trades:
        influence_weight = 0.0
    elif closed_count < (min_gpt_trades * 2):
        influence_weight = 0.2
    else:
        influence_weight = 0.3
    influence_weight = max(0.0, min(float(max_gpt_influence), influence_weight))

    merged = dict(deterministic_proposal)
    merged["min_confidence"] = deterministic_proposal["min_confidence"] + influence_weight * _safe_float(
        gpt_deltas.get("min_confidence"), 0.0
    )
    merged["stop_distance_atr_mult"] = deterministic_proposal["stop_distance_atr_mult"] * (
        1.0 + influence_weight * _safe_float(gpt_deltas.get("stop_distance_atr_mult_pct"), 0.0)
    )
    merged["penalty_high_vol_chop"] = deterministic_proposal["penalty_high_vol_chop"] * (
        1.0 + influence_weight * _safe_float(gpt_deltas.get("penalty_high_vol_chop_pct"), 0.0)
    )
    merged["penalty_elevated_event_risk"] = deterministic_proposal["penalty_elevated_event_risk"] * (
        1.0 + influence_weight * _safe_float(gpt_deltas.get("penalty_elevated_event_risk_pct"), 0.0)
    )
    merged["entry_min_score"] = deterministic_proposal["entry_min_score"] + influence_weight * _safe_float(
        gpt_deltas.get("entry_min_score"), 0.0
    )
    merged["entry_min_effective_confidence"] = deterministic_proposal[
        "entry_min_effective_confidence"
    ] + influence_weight * _safe_float(gpt_deltas.get("entry_min_effective_confidence"), 0.0)
    merged["entry_min_agreement"] = deterministic_proposal["entry_min_agreement"] + influence_weight * _safe_float(
        gpt_deltas.get("entry_min_agreement"), 0.0
    )
    merged["entry_min_margin"] = deterministic_proposal["entry_min_margin"] + influence_weight * _safe_float(
        gpt_deltas.get("entry_min_margin"), 0.0
    )
    merged["aggression_risk_mult"] = deterministic_proposal["aggression_risk_mult"] * (
        1.0 + influence_weight * _safe_float(gpt_deltas.get("aggression_risk_mult_pct"), 0.0)
    )
    merged["aggression_stop_mult"] = deterministic_proposal["aggression_stop_mult"] * (
        1.0 + influence_weight * _safe_float(gpt_deltas.get("aggression_stop_mult_pct"), 0.0)
    )
    merged["aggression_hold_mult"] = deterministic_proposal["aggression_hold_mult"] * (
        1.0 + influence_weight * _safe_float(gpt_deltas.get("aggression_hold_mult_pct"), 0.0)
    )
    merged["aggression_exposure_cap"] = deterministic_proposal["aggression_exposure_cap"] * (
        1.0 + influence_weight * _safe_float(gpt_deltas.get("aggression_exposure_cap_pct"), 0.0)
    )

    bounded_min_conf = _clamp(_step_limited(old_min_conf, merged["min_confidence"], 0.02), 0.25, 0.70)
    bounded_stop_mult = _clamp(_step_limited_rel(old_stop_mult, merged["stop_distance_atr_mult"], 0.05), 0.75, 2.50)
    bounded_penalty_chop = _clamp(_step_limited_rel(old_penalty_chop, merged["penalty_high_vol_chop"], 0.05), 0.70, 1.00)
    bounded_penalty_event = _clamp(
        _step_limited_rel(old_penalty_event, merged["penalty_elevated_event_risk"], 0.05), 0.70, 1.00
    )
    bounded_entry_min_score = _clamp(_step_limited(old_entry_min_score, merged["entry_min_score"], 0.02), 0.05, 0.60)
    bounded_entry_min_effective_conf = _clamp(
        _step_limited(old_entry_min_effective_conf, merged["entry_min_effective_confidence"], 0.02), 0.25, 0.80
    )
    bounded_entry_min_agreement = _clamp(
        _step_limited(old_entry_min_agreement, merged["entry_min_agreement"], 0.02), 0.40, 0.90
    )
    bounded_entry_min_margin = _clamp(
        _step_limited(old_entry_min_margin, merged["entry_min_margin"], 0.02), 0.02, 0.20
    )
    bounded_aggr_risk = _clamp(_step_limited_rel(old_aggr_risk, merged["aggression_risk_mult"], 0.05), 0.80, 1.20)
    bounded_aggr_stop = _clamp(_step_limited_rel(old_aggr_stop, merged["aggression_stop_mult"], 0.05), 0.80, 1.20)
    bounded_aggr_hold = _clamp(_step_limited_rel(old_aggr_hold, merged["aggression_hold_mult"], 0.05), 0.80, 1.20)
    bounded_aggr_exposure = _clamp(
        _step_limited_rel(old_aggr_exposure, merged["aggression_exposure_cap"], 0.05), 0.10, 0.15
    )

    new_min_conf = old_min_conf * (1.0 - alpha) + bounded_min_conf * alpha
    new_stop_mult = old_stop_mult * (1.0 - alpha) + bounded_stop_mult * alpha
    new_penalty_chop = old_penalty_chop * (1.0 - alpha) + bounded_penalty_chop * alpha
    new_penalty_event = old_penalty_event * (1.0 - alpha) + bounded_penalty_event * alpha
    new_entry_min_score = old_entry_min_score * (1.0 - alpha) + bounded_entry_min_score * alpha
    new_entry_min_effective_conf = old_entry_min_effective_conf * (1.0 - alpha) + bounded_entry_min_effective_conf * alpha
    new_entry_min_agreement = old_entry_min_agreement * (1.0 - alpha) + bounded_entry_min_agreement * alpha
    new_entry_min_margin = old_entry_min_margin * (1.0 - alpha) + bounded_entry_min_margin * alpha
    new_aggr_risk = old_aggr_risk * (1.0 - alpha) + bounded_aggr_risk * alpha
    new_aggr_stop = old_aggr_stop * (1.0 - alpha) + bounded_aggr_stop * alpha
    new_aggr_hold = old_aggr_hold * (1.0 - alpha) + bounded_aggr_hold * alpha
    new_aggr_exposure = old_aggr_exposure * (1.0 - alpha) + bounded_aggr_exposure * alpha

    frozen_params: List[str] = []
    scope = str(learn_scope or "full").lower()
    if scope == "stop_only":
        frozen_params = [
            "min_confidence",
            "penalty_high_vol_chop",
            "penalty_elevated_event_risk",
            "entry_min_score",
            "entry_min_effective_confidence",
            "entry_min_agreement",
            "entry_min_margin",
            "aggression_risk_mult",
            "aggression_stop_mult",
            "aggression_hold_mult",
            "aggression_exposure_cap",
        ]
    elif scope == "stop_plus_entry":
        frozen_params = [
            "min_confidence",
            "penalty_high_vol_chop",
            "penalty_elevated_event_risk",
            "aggression_risk_mult",
            "aggression_stop_mult",
            "aggression_hold_mult",
            "aggression_exposure_cap",
        ]

    if freeze_aggression_baseline:
        for k in ["aggression_risk_mult", "aggression_stop_mult", "aggression_hold_mult", "aggression_exposure_cap"]:
            if k not in frozen_params:
                frozen_params.append(k)

    if "min_confidence" in frozen_params:
        new_min_conf = old_min_conf
        bounded_min_conf = old_min_conf
        merged["min_confidence"] = old_min_conf
    if "penalty_high_vol_chop" in frozen_params:
        new_penalty_chop = old_penalty_chop
        bounded_penalty_chop = old_penalty_chop
        merged["penalty_high_vol_chop"] = old_penalty_chop
    if "penalty_elevated_event_risk" in frozen_params:
        new_penalty_event = old_penalty_event
        bounded_penalty_event = old_penalty_event
        merged["penalty_elevated_event_risk"] = old_penalty_event
    if "entry_min_score" in frozen_params:
        new_entry_min_score = old_entry_min_score
        bounded_entry_min_score = old_entry_min_score
        merged["entry_min_score"] = old_entry_min_score
    if "entry_min_effective_confidence" in frozen_params:
        new_entry_min_effective_conf = old_entry_min_effective_conf
        bounded_entry_min_effective_conf = old_entry_min_effective_conf
        merged["entry_min_effective_confidence"] = old_entry_min_effective_conf
    if "entry_min_agreement" in frozen_params:
        new_entry_min_agreement = old_entry_min_agreement
        bounded_entry_min_agreement = old_entry_min_agreement
        merged["entry_min_agreement"] = old_entry_min_agreement
    if "entry_min_margin" in frozen_params:
        new_entry_min_margin = old_entry_min_margin
        bounded_entry_min_margin = old_entry_min_margin
        merged["entry_min_margin"] = old_entry_min_margin
    if "aggression_risk_mult" in frozen_params:
        new_aggr_risk = old_aggr_risk
        bounded_aggr_risk = old_aggr_risk
        merged["aggression_risk_mult"] = old_aggr_risk
    if "aggression_stop_mult" in frozen_params:
        new_aggr_stop = old_aggr_stop
        bounded_aggr_stop = old_aggr_stop
        merged["aggression_stop_mult"] = old_aggr_stop
    if "aggression_hold_mult" in frozen_params:
        new_aggr_hold = old_aggr_hold
        bounded_aggr_hold = old_aggr_hold
        merged["aggression_hold_mult"] = old_aggr_hold
    if "aggression_exposure_cap" in frozen_params:
        new_aggr_exposure = old_aggr_exposure
        bounded_aggr_exposure = old_aggr_exposure
        merged["aggression_exposure_cap"] = old_aggr_exposure

    changes = {
        "min_confidence": {
            "old": round(old_min_conf, 6),
            "proposed": round(merged["min_confidence"], 6),
            "bounded": round(bounded_min_conf, 6),
            "ema_new": round(new_min_conf, 6),
        },
        "stop_distance_atr_mult": {
            "old": round(old_stop_mult, 6),
            "proposed": round(merged["stop_distance_atr_mult"], 6),
            "bounded": round(bounded_stop_mult, 6),
            "ema_new": round(new_stop_mult, 6),
        },
        "penalty_high_vol_chop": {
            "old": round(old_penalty_chop, 6),
            "proposed": round(merged["penalty_high_vol_chop"], 6),
            "bounded": round(bounded_penalty_chop, 6),
            "ema_new": round(new_penalty_chop, 6),
        },
        "penalty_elevated_event_risk": {
            "old": round(old_penalty_event, 6),
            "proposed": round(merged["penalty_elevated_event_risk"], 6),
            "bounded": round(bounded_penalty_event, 6),
            "ema_new": round(new_penalty_event, 6),
        },
        "entry_min_score": {
            "old": round(old_entry_min_score, 6),
            "proposed": round(merged["entry_min_score"], 6),
            "bounded": round(bounded_entry_min_score, 6),
            "ema_new": round(new_entry_min_score, 6),
        },
        "entry_min_effective_confidence": {
            "old": round(old_entry_min_effective_conf, 6),
            "proposed": round(merged["entry_min_effective_confidence"], 6),
            "bounded": round(bounded_entry_min_effective_conf, 6),
            "ema_new": round(new_entry_min_effective_conf, 6),
        },
        "entry_min_agreement": {
            "old": round(old_entry_min_agreement, 6),
            "proposed": round(merged["entry_min_agreement"], 6),
            "bounded": round(bounded_entry_min_agreement, 6),
            "ema_new": round(new_entry_min_agreement, 6),
        },
        "entry_min_margin": {
            "old": round(old_entry_min_margin, 6),
            "proposed": round(merged["entry_min_margin"], 6),
            "bounded": round(bounded_entry_min_margin, 6),
            "ema_new": round(new_entry_min_margin, 6),
        },
        "aggression_baseline": {
            "old": {
                "risk_mult": round(old_aggr_risk, 6),
                "stop_mult": round(old_aggr_stop, 6),
                "hold_mult": round(old_aggr_hold, 6),
                "exposure_cap": round(old_aggr_exposure, 6),
            },
            "proposed": {
                "risk_mult": round(merged["aggression_risk_mult"], 6),
                "stop_mult": round(merged["aggression_stop_mult"], 6),
                "hold_mult": round(merged["aggression_hold_mult"], 6),
                "exposure_cap": round(merged["aggression_exposure_cap"], 6),
            },
            "bounded": {
                "risk_mult": round(bounded_aggr_risk, 6),
                "stop_mult": round(bounded_aggr_stop, 6),
                "hold_mult": round(bounded_aggr_hold, 6),
                "exposure_cap": round(bounded_aggr_exposure, 6),
            },
            "ema_new": {
                "risk_mult": round(new_aggr_risk, 6),
                "stop_mult": round(new_aggr_stop, 6),
                "hold_mult": round(new_aggr_hold, 6),
                "exposure_cap": round(new_aggr_exposure, 6),
            },
        },
    }

    new_risk = dict(risk_limits)
    new_policy = dict(learning_policy)
    new_risk["min_confidence"] = round(new_min_conf, 6)
    new_risk["stop_distance_atr_mult"] = round(new_stop_mult, 6)
    new_risk["entry_min_score"] = round(new_entry_min_score, 6)
    new_risk["entry_min_effective_confidence"] = round(new_entry_min_effective_conf, 6)
    new_risk["entry_min_agreement"] = round(new_entry_min_agreement, 6)
    new_risk["entry_min_margin"] = round(new_entry_min_margin, 6)
    new_policy["penalty_high_vol_chop"] = round(new_penalty_chop, 6)
    new_policy["penalty_elevated_event_risk"] = round(new_penalty_event, 6)
    new_policy["gpt_learn_max_influence"] = round(float(max_gpt_influence), 6)
    new_policy["aggression_baseline"] = {
        "risk_mult": round(new_aggr_risk, 6),
        "stop_mult": round(new_aggr_stop, 6),
        "hold_mult": round(new_aggr_hold, 6),
        "exposure_cap": round(new_aggr_exposure, 6),
    }
    gate_overrides = dict(new_policy.get("gate_overrides") or {})
    gate_overrides["min_confidence"] = round(new_min_conf, 6)
    new_policy["gate_overrides"] = gate_overrides

    rejected_or_clamped: List[str] = []
    if gpt_deltas and influence_weight <= 0.0:
        rejected_or_clamped.append(f"gpt_influence_disabled:closed_trades_below_{min_gpt_trades}")
    bounded_map = {
        "min_confidence": bounded_min_conf,
        "stop_distance_atr_mult": bounded_stop_mult,
        "penalty_high_vol_chop": bounded_penalty_chop,
        "penalty_elevated_event_risk": bounded_penalty_event,
        "entry_min_score": bounded_entry_min_score,
        "entry_min_effective_confidence": bounded_entry_min_effective_conf,
        "entry_min_agreement": bounded_entry_min_agreement,
        "entry_min_margin": bounded_entry_min_margin,
        "aggression_risk_mult": bounded_aggr_risk,
        "aggression_stop_mult": bounded_aggr_stop,
        "aggression_hold_mult": bounded_aggr_hold,
        "aggression_exposure_cap": bounded_aggr_exposure,
    }
    for key, merged_val in merged.items():
        bounded_val = bounded_map.get(key)
        if bounded_val is None:
            continue
        if abs(float(merged_val) - float(bounded_val)) > 1e-12:
            rejected_or_clamped.append(f"{key}:bounded_or_step_limited")

    gpt_strategy = {
        "proposal_source": "gpt" if gpt_policy_proposal else "none",
        "proposal_confidence": round(gpt_confidence, 6) if gpt_policy_proposal else 0.0,
        "proposal_deltas": gpt_deltas if gpt_policy_proposal else {},
        "influence_weight": round(influence_weight, 6),
    }
    arbiter = {
        "deterministic_proposal": {k: round(v, 6) for k, v in deterministic_proposal.items()},
        "gpt_proposal": dict(gpt_deltas),
        "merged_proposal": {k: round(v, 6) for k, v in merged.items()},
        "bounded_proposal": {k: round(v, 6) for k, v in bounded_map.items()},
        "applied_values": {
            "risk_limits": {
                "min_confidence": round(new_min_conf, 6),
                "stop_distance_atr_mult": round(new_stop_mult, 6),
                "entry_min_score": round(new_entry_min_score, 6),
                "entry_min_effective_confidence": round(new_entry_min_effective_conf, 6),
                "entry_min_agreement": round(new_entry_min_agreement, 6),
                "entry_min_margin": round(new_entry_min_margin, 6),
            },
            "learning_policy": {
                "penalty_high_vol_chop": round(new_penalty_chop, 6),
                "penalty_elevated_event_risk": round(new_penalty_event, 6),
                "aggression_baseline": {
                    "risk_mult": round(new_aggr_risk, 6),
                    "stop_mult": round(new_aggr_stop, 6),
                    "hold_mult": round(new_aggr_hold, 6),
                    "exposure_cap": round(new_aggr_exposure, 6),
                },
            },
        },
        "rejected_or_clamped_reasons": rejected_or_clamped,
        "learn_scope": scope,
        "frozen_params": sorted(set(frozen_params)),
    }

    diff_text = (
        f"min_confidence: {old_min_conf:.4f}->{new_min_conf:.4f}; "
        f"stop_distance_atr_mult: {old_stop_mult:.4f}->{new_stop_mult:.4f}; "
        f"penalty_high_vol_chop: {old_penalty_chop:.4f}->{new_penalty_chop:.4f}; "
        f"penalty_elevated_event_risk: {old_penalty_event:.4f}->{new_penalty_event:.4f}; "
        f"entry_min_score: {old_entry_min_score:.4f}->{new_entry_min_score:.4f}; "
        f"entry_min_effective_confidence: {old_entry_min_effective_conf:.4f}->{new_entry_min_effective_conf:.4f}; "
        f"entry_min_agreement: {old_entry_min_agreement:.4f}->{new_entry_min_agreement:.4f}; "
        f"entry_min_margin: {old_entry_min_margin:.4f}->{new_entry_min_margin:.4f}; "
        f"aggr_risk_mult: {old_aggr_risk:.4f}->{new_aggr_risk:.4f}; "
        f"aggr_stop_mult: {old_aggr_stop:.4f}->{new_aggr_stop:.4f}; "
        f"aggr_hold_mult: {old_aggr_hold:.4f}->{new_aggr_hold:.4f}; "
        f"aggr_exposure_cap: {old_aggr_exposure:.4f}->{new_aggr_exposure:.4f}"
    )
    return changes, new_risk, new_policy, diff_text, gpt_strategy, arbiter


def _band(v: float, low: float, high: float) -> str:
    if v < low:
        return "low"
    if v > high:
        return "high"
    return "mid"


def _trade_metrics(trades: List[Dict[str, Any]]) -> Dict[str, float]:
    n = max(1, len(trades))
    pnls = [float(_safe_float((t or {}).get("gross_pnl"), 0.0)) for t in trades]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p < 0]
    expectancy = sum(pnls) / float(n)
    win_rate = float(len(wins)) / float(n)
    gain_sum = sum(wins)
    loss_sum = abs(sum(losses))
    profit_factor = (gain_sum / loss_sum) if loss_sum > 1e-9 else (2.0 if gain_sum > 0 else 0.0)

    # Drawdown proxy on closed-trade PnL path.
    cum = 0.0
    peak = 0.0
    max_dd = 0.0
    for p in pnls:
        cum += float(p)
        peak = max(peak, cum)
        max_dd = max(max_dd, peak - cum)
    denom = max(100.0, abs(peak) + 1e-9)
    drawdown_proxy = max_dd / denom

    mean = expectancy
    var = sum((p - mean) ** 2 for p in pnls) / float(n)
    pnl_std = math.sqrt(max(0.0, var))
    instability = min(1.0, pnl_std / max(50.0, abs(expectancy) + 1e-9))

    flip_rate = float(
        sum(1 for t in trades if str((t or {}).get("exit_reason") or "").upper() == "FLIP_EXIT")
    ) / float(n)
    confidence_collapse_rate = float(
        sum(1 for t in trades if float(_safe_float((t or {}).get("effective_confidence"), 0.0)) < 0.25)
    ) / float(n)

    return {
        "count": float(len(trades)),
        "expectancy": float(expectancy),
        "win_rate": float(win_rate),
        "profit_factor": float(profit_factor),
        "drawdown_proxy": float(drawdown_proxy),
        "pnl_std": float(pnl_std),
        "instability": float(instability),
        "flip_rate": float(flip_rate),
        "confidence_collapse_rate": float(confidence_collapse_rate),
        "gross_pnl_total": float(sum(pnls)),
    }


def _build_cohort_key(trade: Dict[str, Any], pattern_hits: Optional[List[str]] = None) -> str:
    regime = str((trade or {}).get("regime") or "unknown")
    event = str((trade or {}).get("event_risk") or "unknown")
    top_scenario = str((trade or {}).get("top_scenario") or "unknown")
    conf = float(_safe_float((trade or {}).get("effective_confidence"), 0.0))
    agree = float(_safe_float((trade or {}).get("agreement_score"), 0.0))
    pattern_tag = "none"
    hits = list(pattern_hits or [])
    if hits:
        pattern_tag = str(sorted(hits)[0])
    return (
        f"regime={regime}|event={event}|scenario={top_scenario}|"
        f"conf={_band(conf, 0.35, 0.60)}|agree={_band(agree, 0.50, 0.70)}|pat={pattern_tag}"
    )


def _summarize_cohorts(
    trades: List[Dict[str, Any]],
    audit_rows: Optional[List[Dict[str, Any]]] = None,
    min_count: int = 5,
) -> Dict[str, Any]:
    key_to_patterns: Dict[Tuple[str, str], List[str]] = {}
    for row in audit_rows or []:
        run_id = str((row or {}).get("run_id") or "")
        symbol = str((row or {}).get("symbol") or "")
        hits = list(((row or {}).get("patterns") or {}).get("hits") or [])
        if run_id and symbol:
            key_to_patterns[(run_id, symbol)] = [str(h) for h in hits if str(h)]

    agg: Dict[str, Dict[str, float]] = {}
    for t in trades:
        key = _build_cohort_key(
            t,
            key_to_patterns.get((str(t.get("linked_run_id") or ""), str(t.get("symbol") or ""))),
        )
        row = agg.setdefault(
            key,
            {
                "count": 0.0,
                "wins": 0.0,
                "losses": 0.0,
                "pnl_sum": 0.0,
            },
        )
        pnl = float(_safe_float(t.get("gross_pnl"), 0.0))
        row["count"] += 1.0
        row["pnl_sum"] += pnl
        if pnl > 0:
            row["wins"] += 1.0
        elif pnl < 0:
            row["losses"] += 1.0

    cohorts: List[Dict[str, Any]] = []
    for key, row in agg.items():
        cnt = int(row["count"])
        if cnt < int(max(1, min_count)):
            continue
        exp = float(row["pnl_sum"]) / float(max(1, cnt))
        win_rate = float(row["wins"]) / float(max(1, cnt))
        cohorts.append(
            {
                "cohort": key,
                "count": cnt,
                "expectancy": round(exp, 6),
                "win_rate": round(win_rate, 6),
                "pnl_sum": round(float(row["pnl_sum"]), 6),
            }
        )
    worked = [c for c in cohorts if c["expectancy"] > 0]
    failed = [c for c in cohorts if c["expectancy"] <= 0]
    worked.sort(key=lambda x: (-float(x["expectancy"]), -int(x["count"]), x["cohort"]))
    failed.sort(key=lambda x: (float(x["expectancy"]), -int(x["count"]), x["cohort"]))
    return {
        "worked": worked[:8],
        "failed": failed[:8],
    }


def _default_gate_policy_template() -> Dict[str, Any]:
    return {
        "policy_version": "smart_adjust_v1_baseline",
        "enabled_hard_checks": [
            "CRITICAL_STALE_REQUIRED_MODALITY",
            "CRITICAL_LOW_CONFIDENCE",
        ],
        "enabled_weighted_checks": [
            "LOW_CONFIDENCE",
            "LOW_HORIZON_ALIGNMENT",
            "MODEL_EDGE_WEAK",
            "RISK_CLUSTER_LOW_CONF",
        ],
        "diagnostic_only_checks": [
            "LOW_AGREEMENT",
            "LOW_BREADTH",
            "RELATIVE_WEAKNESS",
            "PRE_BREAKOUT_COMPRESSION",
        ],
        "merged_overlap_mode": True,
    }


def evaluate_killswitch_state(
    trades: List[Dict[str, Any]],
    learning_policy: Dict[str, Any],
) -> Dict[str, Any]:
    metrics = _trade_metrics(trades)
    kill_cfg = dict((learning_policy or {}).get("kill_switch") or {})
    drawdown_limit = float(_safe_float(kill_cfg.get("drawdown_spike"), 0.22))
    flip_limit = float(_safe_float(kill_cfg.get("flip_rate"), 0.35))
    conf_limit = float(_safe_float(kill_cfg.get("confidence_collapse"), 0.55))

    reasons: List[Dict[str, Any]] = []
    if float(metrics["drawdown_proxy"]) >= drawdown_limit:
        reasons.append(
            {
                "code": "KILLSWITCH_DRAWDOWN_SPIKE",
                "detail": f"drawdown_proxy={metrics['drawdown_proxy']:.3f} >= {drawdown_limit:.3f}",
            }
        )
    if float(metrics["flip_rate"]) >= flip_limit:
        reasons.append(
            {
                "code": "KILLSWITCH_FLIP_RATE",
                "detail": f"flip_rate={metrics['flip_rate']:.3f} >= {flip_limit:.3f}",
            }
        )
    if float(metrics["confidence_collapse_rate"]) >= conf_limit:
        reasons.append(
            {
                "code": "KILLSWITCH_CONF_COLLAPSE",
                "detail": f"confidence_collapse_rate={metrics['confidence_collapse_rate']:.3f} >= {conf_limit:.3f}",
            }
        )
    return {
        "active": bool(reasons),
        "reasons": reasons,
        "limits": {
            "drawdown_spike": drawdown_limit,
            "flip_rate": flip_limit,
            "confidence_collapse": conf_limit,
        },
        "metrics": metrics,
    }


def _composite_alpha_score(metrics: Dict[str, float], score_weights: Dict[str, Any]) -> float:
    w_exp = float(_safe_float(score_weights.get("expectancy"), 0.45))
    w_win = float(_safe_float(score_weights.get("win_rate"), 0.30))
    w_dd = float(_safe_float(score_weights.get("drawdown_penalty"), 0.40))
    w_inst = float(_safe_float(score_weights.get("instability_penalty"), 0.25))

    exp_n = max(-1.0, min(1.0, float(metrics.get("expectancy", 0.0)) / 120.0))
    win_n = (float(metrics.get("win_rate", 0.5)) - 0.5) * 2.0
    dd_n = max(0.0, min(1.0, float(metrics.get("drawdown_proxy", 0.0))))
    inst_n = max(0.0, min(1.0, float(metrics.get("instability", 0.0))))
    return float((w_exp * exp_n) + (w_win * win_n) - (w_dd * dd_n) - (w_inst * inst_n))


def _project_success_fail_ratio(
    base_win_rate: float,
    score_delta: float,
    sample_size: int,
) -> Dict[str, float]:
    p = float(base_win_rate) + (0.12 * float(score_delta))
    p = max(0.05, min(0.95, p))
    fail = 1.0 - p
    conf = max(0.1, min(0.95, 0.35 + (float(sample_size) / 60.0)))
    return {
        "success_pct": round(p * 100.0, 2),
        "fail_pct": round(fail * 100.0, 2),
        "success_to_fail_ratio": round(p / max(1e-9, fail), 4),
        "confidence": round(conf, 3),
    }


def build_smart_adjustment_candidates(
    trades: List[Dict[str, Any]],
    counts: Dict[str, int],
    audit_rows: List[Dict[str, Any]],
    risk_limits: Dict[str, Any],
    learning_policy: Dict[str, Any],
    numeric_changes: Dict[str, Any],
    numeric_risk_new: Dict[str, Any],
    numeric_policy_new: Dict[str, Any],
    numeric_diff_text: str,
    gpt_strategy: Dict[str, Any],
    arbiter: Dict[str, Any],
) -> Dict[str, Any]:
    metrics = _trade_metrics(trades)
    min_cohort_trades = int(_safe_float((learning_policy or {}).get("min_cohort_trades"), 5))
    cohorts = _summarize_cohorts(trades, audit_rows=audit_rows, min_count=min_cohort_trades)
    score_weights = dict((learning_policy or {}).get("score_weights") or {})
    base_alpha = _composite_alpha_score(metrics, score_weights)

    current_gate_policy = dict(
        ((learning_policy or {}).get("gate_overrides") or {}).get("gate_policy") or {}
    )
    simplified_policy = _default_gate_policy_template()

    candidates: List[Dict[str, Any]] = []

    candidates.append(
        {
            "candidate_id": "cand_numeric_base",
            "candidate_type": "numeric",
            "score": base_alpha,
            "confidence": 0.40 + min(0.45, float(metrics.get("count", 0.0)) / 100.0),
            "reasoning": "Bounded numeric adjustment from deterministic+arbiter path.",
            "risk_limits": numeric_risk_new,
            "learning_policy": numeric_policy_new,
            "gate_policy_changes": {
                "from": current_gate_policy or simplified_policy,
                "to": (numeric_policy_new.get("gate_overrides") or {}).get("gate_policy") or current_gate_policy or simplified_policy,
            },
            "diff_text": numeric_diff_text,
            "gpt_strategy": gpt_strategy,
            "arbiter": arbiter,
        }
    )

    policy_struct = dict(learning_policy or {})
    gate_overrides_struct = dict(policy_struct.get("gate_overrides") or {})
    gate_overrides_struct["gate_policy"] = simplified_policy
    policy_struct["gate_overrides"] = gate_overrides_struct

    structural_bonus = 0.0
    if int(counts.get("CONFIDENCE_CALIBRATION", 0)) + int(counts.get("SIGNAL_DISAGREEMENT", 0)) > 0:
        structural_bonus += 0.04
    if int(counts.get("EVENT_RISK_UNDERWEIGHTED", 0)) > 0:
        structural_bonus -= 0.01

    candidates.append(
        {
            "candidate_id": "cand_structural_simplified",
            "candidate_type": "structural",
            "score": base_alpha + structural_bonus,
            "confidence": 0.36 + min(0.40, float(metrics.get("count", 0.0)) / 120.0),
            "reasoning": "Simplified gate composition to reduce over-blocking while preserving immutable hard safety checks.",
            "risk_limits": dict(risk_limits or {}),
            "learning_policy": policy_struct,
            "gate_policy_changes": {
                "from": current_gate_policy or simplified_policy,
                "to": simplified_policy,
            },
        }
    )

    hybrid_policy = dict(numeric_policy_new or {})
    hybrid_gate_overrides = dict(hybrid_policy.get("gate_overrides") or {})
    hybrid_gate_overrides["gate_policy"] = simplified_policy
    hybrid_policy["gate_overrides"] = hybrid_gate_overrides
    candidates.append(
        {
            "candidate_id": "cand_hybrid_numeric_structural",
            "candidate_type": "hybrid",
            "score": base_alpha + structural_bonus + 0.015,
            "confidence": 0.38 + min(0.45, float(metrics.get("count", 0.0)) / 110.0),
            "reasoning": "Hybrid candidate combines bounded numeric changes with simplified gate policy for balanced alpha/risk adaptation.",
            "risk_limits": numeric_risk_new,
            "learning_policy": hybrid_policy,
            "gate_policy_changes": {
                "from": current_gate_policy or simplified_policy,
                "to": simplified_policy,
            },
        }
    )

    # Optional adaptive reintroduction candidate if edge appears stable.
    if (
        float(metrics.get("count", 0.0)) >= 25.0
        and float(metrics.get("win_rate", 0.0)) >= 0.56
        and float(metrics.get("expectancy", 0.0)) > 0.0
    ):
        reintro = dict(simplified_policy)
        enabled_weighted = list(reintro.get("enabled_weighted_checks") or [])
        diag = list(reintro.get("diagnostic_only_checks") or [])
        if "LOW_AGREEMENT" in diag and "LOW_AGREEMENT" not in enabled_weighted:
            diag = [c for c in diag if c != "LOW_AGREEMENT"]
            enabled_weighted.append("LOW_AGREEMENT")
        reintro["enabled_weighted_checks"] = enabled_weighted
        reintro["diagnostic_only_checks"] = diag
        reintro_policy = dict(hybrid_policy)
        reintro_go = dict(reintro_policy.get("gate_overrides") or {})
        reintro_go["gate_policy"] = reintro
        reintro_policy["gate_overrides"] = reintro_go
        candidates.append(
            {
                "candidate_id": "cand_reintroduce_low_agreement",
                "candidate_type": "structural_reintro",
                "score": base_alpha + structural_bonus + 0.010,
                "confidence": 0.42 + min(0.45, float(metrics.get("count", 0.0)) / 100.0),
                "reasoning": "Reintroduce LOW_AGREEMENT as weighted gate due to stronger recent edge stability.",
                "risk_limits": numeric_risk_new,
                "learning_policy": reintro_policy,
                "gate_policy_changes": {
                    "from": simplified_policy,
                    "to": reintro,
                },
            }
        )

    # Rank and project outcomes.
    candidates.sort(
        key=lambda c: (-float(c.get("score") or 0.0), -float(c.get("confidence") or 0.0), str(c.get("candidate_id") or "")),
    )
    base_win = float(metrics.get("win_rate", 0.5))
    for idx, cand in enumerate(candidates, start=1):
        cand["rank"] = idx
        delta = float(cand.get("score") or 0.0) - float(base_alpha)
        proj = _project_success_fail_ratio(base_win_rate=base_win, score_delta=delta, sample_size=int(metrics.get("count", 0.0)))
        cand["projected_success_ratio"] = proj["success_pct"]
        cand["projected_fail_ratio"] = proj["fail_pct"]
        cand["projected_success_fail_ratio"] = proj
        cand["selected"] = False
        cand["applied"] = False

    selected = candidates[0] if candidates else None
    if selected is not None:
        selected["selected"] = True

    projected_overall = (
        dict((selected or {}).get("projected_success_fail_ratio") or {})
        if selected
        else _project_success_fail_ratio(base_win_rate=base_win, score_delta=0.0, sample_size=int(metrics.get("count", 0.0)))
    )

    return {
        "metrics": metrics,
        "base_alpha_score": round(base_alpha, 6),
        "cohorts": cohorts,
        "candidates": candidates,
        "selected": selected,
        "projected_success_fail_ratio": projected_overall,
    }
